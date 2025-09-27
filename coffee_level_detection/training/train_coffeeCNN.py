import torch, argparse, json
from coffee_level_detection.training.coffee import coffeeCNN, CoffeeImageDataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import pandas as pd


def train(dataset: CoffeeImageDataset, batch_size: int = 1, epochs: int = 20, checkpoint: int = 5):
    """
    Train a coffeeCNN model on the provided CoffeeImageDataset.
    Args:
        dataset (CoffeeImageDataset): The dataset to train on.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        checkpoint (int): Checkpoint interval for saving the model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = coffeeCNN(num_classes=11).to(device)
    train_loader, val_loader = __handle_dataset(dataset, 0.8, 0.2, batch_size)
    # weights = __weights(dataset.df['coffee_level'].values)
    criterion = torch.nn.CrossEntropyLoss()  # class balancing
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    #train
    __train_loop(model,device,train_loader, criterion, optimizer, epochs, checkpoint)
    torch.save(model, "coffeeCNN")
    #eval
    __eval_loop(model, val_loader)
    
    
def __train_loop(model: coffeeCNN, device, train_loader, criterion, optimizer, epochs, checkpoint):
    """
    Internal training loop for coffeeCNN.
    Args:
        model (coffeeCNN): The model to train.
        device: Device to use (CPU or CUDA).
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        epochs (int): Number of epochs.
        checkpoint (int): Checkpoint interval.
    """
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for i, epoch in enumerate(epoch_bar):
        running_loss = 0.0
        total_samples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for inputs, labels in batch_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate and display per-batch
            batch_loss = loss.item()
            running_loss += batch_loss * (labels.size(0) if hasattr(labels, "size") else 1)
            batch_bar.set_postfix(loss=f"{batch_loss:.4f}")
        if (i+1)%checkpoint == 0:
            torch.save(model, f"checkpoint")
        # compute epoch average loss and show on epoch bar
        avg_loss = running_loss / total_samples if total_samples > 0 else running_loss / max(len(train_loader), 1)
        epoch_bar.set_postfix(epoch_loss=f"{avg_loss:.4f}")
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def __eval_loop(model, val_loader):
    """
    Internal evaluation loop for coffeeCNN.
    Args:
        model: Trained model to evaluate.
        val_loader: DataLoader for validation data.
    Returns:
        Tuple (correct, total): Number of correct predictions and total samples.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"correct: {correct}, total: {total}")
    print(f"accuracy: {correct/total}")
    return correct, total

def __weights(y: np.ndarray):
    """
    Compute class weights for balancing the loss function.
    Args:
        y (np.ndarray): Array of class labels.
    Returns:
        torch.Tensor: Tensor of class weights.
    """
    classes = np.arange(11)
    raw_weights = compute_class_weight('balanced', classes=classes, y=y)
    weights = torch.sqrt(raw_weights)
    weights = weights / weights.mean()
    class_weights = torch.clamp(weights, min=0.5, max=5.0)
    return class_weights

def __handle_dataset(dataset: CoffeeImageDataset, train_size, val_size, batch_size):
    """
    Split dataset into training and validation sets and create DataLoaders.
    Args:
        dataset (CoffeeImageDataset): The dataset to split.
        train_size (float): Proportion of training data.
        val_size (float): Proportion of validation data.
    Returns:
        Tuple (train_loader, val_loader): DataLoaders for train and validation sets.
    """
    train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
    )
    
    train_indices = train_dataset.indices if hasattr(train_dataset, "indices") else train_dataset._indices
    train_labels = dataset.df.iloc[train_indices]['coffee_level'].values

    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    weights = np.array([weight[t] for t in train_labels])

    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def __load_dataset(dataset_path, img_path):
    """
    Load dataset from JSON annotation file and image directory.
    Args:
        dataset_path (str): Path to the annotation JSON file.
        img_path (str): Path to the image directory.
    Returns:
        CoffeeImageDataset: Loaded dataset object.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data["annotation_data"])
    df = df.dropna(subset=['coffee_level'])
    dataset = CoffeeImageDataset(df, img_path)
    return dataset

def main():
    """
    Main entry point for training coffeeCNN from command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--f", type=str, default="compiled_coffee_level_annotations.json")
    parser.add_argument("--batch", type=int, default="10")
    parser.add_argument("--epochs",type=int, default="10")
    parser.add_argument("--checkpoint", type=int, default="5")
    parser.add_argument("--img_dir", type=str, default="processed_images")
    parser.add_argument("--checkpoint_dir",type=str, default="checkpoint")
    
    args = parser.parse_args()

    dataset = __load_dataset(args.f, args.img_dir)
    train(dataset, args.batch, args.epochs, args.checkpoint)

if __name__=="__main__":
    main()



