"""
Fine-tuning script for coffee level detection CNN.
Loads pretrained weights and fine-tunes on newly annotated data.
"""
import torch
import argparse
import json
import os
from datetime import datetime
from coffee_level_detection.training.coffee import coffeeCNN, CoffeeImageDataset
from coffee_level_detection.training.train_coffeeCNN import train, __load_dataset
from coffee_level_detection.inference.tools import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_new_annotations(manual_levels_dir='manual_levels', min_samples_per_class=5):
    """
    Load newly annotated data from manual_levels directory.
    
    Args:
        manual_levels_dir: Directory containing manual annotation JSON files
        min_samples_per_class: Minimum samples required per class for training
    
    Returns:
        pandas.DataFrame: DataFrame with filenames and coffee levels
    """
    annotations = []
    
    if not os.path.exists(manual_levels_dir):
        print(f"Warning: {manual_levels_dir} directory not found!")
        return pd.DataFrame()
    
    # Load all annotation files
    for filename in os.listdir(manual_levels_dir):
        if filename.startswith('level_') and filename.endswith('.json'):
            filepath = os.path.join(manual_levels_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract annotation data
                if data.get('coffee_level') is not None:
                    annotations.append({
                        'filename': data['filename'],
                        'coffee_level': data['coffee_level'],
                        'annotator': data.get('annotator', 'unknown'),
                        'timestamp': data.get('timestamp', '')
                    })
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    if not annotations:
        print("No valid annotations found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(annotations)
    
    # Filter out null values
    df = df[df['coffee_level'].notna()]
    df['coffee_level'] = df['coffee_level'].astype(int)
    
    # Check class distribution
    class_counts = df['coffee_level'].value_counts().sort_index()
    print(f"\nClass distribution in new annotations:")
    for level, count in class_counts.items():
        print(f"  Level {level}: {count} samples")
    
    # Filter classes with insufficient samples
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df_filtered = df[df['coffee_level'].isin(valid_classes)]
    
    if len(df_filtered) < len(df):
        removed = len(df) - len(df_filtered)
        print(f"\nFiltered out {removed} samples from classes with < {min_samples_per_class} samples")
        print(f"Remaining: {len(df_filtered)} samples from classes: {sorted(valid_classes.tolist())}")
    
    return df_filtered


def create_balanced_subset(df, max_samples_per_class=100, min_samples_per_class=10):
    """
    Create a balanced subset for fine-tuning to avoid class imbalance.
    
    Args:
        df: DataFrame with annotations
        max_samples_per_class: Maximum samples to use per class
        min_samples_per_class: Minimum samples required per class
    
    Returns:
        pandas.DataFrame: Balanced subset
    """
    balanced_samples = []
    
    for level in sorted(df['coffee_level'].unique()):
        level_data = df[df['coffee_level'] == level]
        
        if len(level_data) < min_samples_per_class:
            print(f"Warning: Level {level} has only {len(level_data)} samples (< {min_samples_per_class}), skipping")
            continue
        
        # Sample up to max_samples_per_class
        n_samples = min(len(level_data), max_samples_per_class)
        sampled = level_data.sample(n=n_samples, random_state=42)
        balanced_samples.append(sampled)
        
        print(f"Level {level}: Using {n_samples}/{len(level_data)} samples")
    
    if balanced_samples:
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    else:
        return pd.DataFrame()


def fine_tune_model(pretrained_path, dataset, output_path, 
                   learning_rate=1e-6, epochs=5, batch_size=8, 
                   freeze_early_layers=True, num_classes=11,
                   early_stopping_patience=3, dropout_rate=0.3):
    """
    Fine-tune a pretrained model on new data with overfitting prevention.
    
    Args:
        pretrained_path: Path to pretrained model weights
        dataset: CoffeeImageDataset for fine-tuning
        output_path: Path to save fine-tuned model
        learning_rate: Learning rate for fine-tuning (very low to prevent overfitting)
        epochs: Number of fine-tuning epochs (reduced default)
        batch_size: Batch size for training
        freeze_early_layers: Whether to freeze early conv layers
        num_classes: Number of output classes
        early_stopping_patience: Stop training if no improvement for N epochs
        dropout_rate: Dropout rate for regularization
    """
    # Device selection with Intel XPU support
    if torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Using device: {device} (Intel XPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU)")
    
    # Load pretrained model
    print(f"Loading pretrained model from {pretrained_path}")
    model = coffeeCNN(num_classes=num_classes).to(device)
    
    # Add dropout for regularization
    model.dropout = torch.nn.Dropout(dropout_rate)
    
    try:
        # Load pretrained weights
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
        print("✅ Loaded pretrained weights successfully")
    except Exception as e:
        print(f"❌ Error loading pretrained weights: {e}")
        print("Starting from random initialization...")
    
    # Freeze more layers for conservative fine-tuning
    if freeze_early_layers:
        print("🔒 Freezing early layers for conservative fine-tuning")
        model.conv1.requires_grad_(False)
        model.bn1.requires_grad_(False)
        model.conv2.requires_grad_(False)
        # Only fine-tune the fully connected layers
        print("   Only training FC layers to prevent overfitting")
    
    # Split dataset with larger validation set
    train_size = int(0.7 * len(dataset))  # Reduced from 0.8
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup for conservative fine-tuning
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)  # Higher weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    print(f"\n🚀 Starting conservative fine-tuning:")
    print(f"  Dataset size: {len(dataset)} ({train_size} train, {val_size} val)")
    print(f"  Learning rate: {learning_rate} (very conservative)")
    print(f"  Max epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Label smoothing: 0.1")
    print(f"  Weight decay: 1e-3")
    
    # Fine-tuning loop with early stopping
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_accuracies = []
    no_improvement_count = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply dropout during training
            inputs = model.dropout(inputs) if hasattr(model, 'dropout') else inputs
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling based on validation accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.2e}")
        
        # Check for overfitting
        if len(train_losses) > 1:
            train_loss_trend = train_losses[-1] - train_losses[-2]
            val_acc_trend = val_accuracies[-1] - val_accuracies[-2] if len(val_accuracies) > 1 else 0
            
            if train_loss_trend < -0.01 and val_acc_trend < -1.0:  # Training loss decreasing but val acc dropping
                print(f"  ⚠️ Potential overfitting detected!")
        
        # Early stopping and best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improvement_count = 0
            torch.save(model.state_dict(), output_path)
            print(f"  💾 Saved new best model (Val Acc: {val_acc:.2f}%)")
        else:
            no_improvement_count += 1
            print(f"  📉 No improvement for {no_improvement_count} epochs")
            
            if no_improvement_count >= early_stopping_patience:
                print(f"  🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs")
                print(f"  📊 Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch + 1}")
                break
        
        print()
    
    print(f"🎉 Fine-tuning complete!")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch + 1})")
    print(f"💾 Fine-tuned model saved to: {output_path}")
    
    return model, {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1
    }


def main():
    """Main function for fine-tuning."""
    parser = argparse.ArgumentParser(description='Fine-tune coffee level detection CNN')
    
    # Model paths
    parser.add_argument('--pretrained', type=str, default='coffeeCNN.pth',
                       help='Path to pretrained model weights')
    parser.add_argument('--output', type=str, default='coffeeCNN_finetuned.pth',
                       help='Path to save fine-tuned model')
    
    # Data paths
    parser.add_argument('--manual-levels', type=str, default='manual_levels',
                       help='Directory with manual annotations')
    parser.add_argument('--img-dir', type=str, default='processed_images',
                       help='Directory with images')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-6,
                       help='Learning rate for fine-tuning (very conservative)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Maximum number of fine-tuning epochs (reduced default)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num-classes', type=int, default=11,
                       help='Number of output classes')
    parser.add_argument('--early-stopping', type=int, default=3,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for regularization')
    
    # Data filtering
    parser.add_argument('--max-per-class', type=int, default=100,
                       help='Maximum samples per class')
    parser.add_argument('--min-per-class', type=int, default=5,
                       help='Minimum samples per class')
    
    # Options
    parser.add_argument('--no-freeze', action='store_true',
                       help='Do not freeze early layers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show data statistics without training')
    
    args = parser.parse_args()
    
    print("🎯 Coffee Level Detection - Model Fine-tuning")
    print("=" * 50)
    
    # Load new annotations
    print("📂 Loading new annotations...")
    df = load_new_annotations(args.manual_levels, args.min_per_class)
    
    if df.empty:
        print("❌ No valid annotations found for fine-tuning!")
        return
    
    # Create balanced subset
    print(f"\n⚖️ Creating balanced subset...")
    balanced_df = create_balanced_subset(df, args.max_per_class, args.min_per_class)
    
    if balanced_df.empty:
        print("❌ No valid data after balancing!")
        return
    
    print(f"\n📊 Final dataset statistics:")
    print(f"Total samples: {len(balanced_df)}")
    class_dist = balanced_df['coffee_level'].value_counts().sort_index()
    for level, count in class_dist.items():
        print(f"  Level {level}: {count} samples")
    
    if args.dry_run:
        print("\n🏃 Dry run complete - no training performed")
        return
    
    # Create dataset
    print(f"\n📦 Creating dataset from {args.img_dir}...")
    try:
        dataset = CoffeeImageDataset(balanced_df, args.img_dir)
        print(f"✅ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return
    
    # Check if pretrained model exists
    if not os.path.exists(args.pretrained):
        print(f"⚠️ Pretrained model {args.pretrained} not found!")
        print("Training from scratch...")
    
    # Fine-tune model
    print(f"\n🔧 Starting fine-tuning...")
    freeze_layers = not args.no_freeze
    
    model, history = fine_tune_model(
        pretrained_path=args.pretrained,
        dataset=dataset,
        output_path=args.output,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        freeze_early_layers=freeze_layers,
        num_classes=args.num_classes,
        early_stopping_patience=args.early_stopping,
        dropout_rate=args.dropout
    )
    
    # Save training history
    history_path = args.output.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'args': vars(args),
            'data_stats': class_dist.to_dict(),
            'training_history': history
        }, f, indent=2)
    
    print(f"📈 Training history saved to: {history_path}")
    print("\n✅ Fine-tuning complete!")


if __name__ == "__main__":
    main()