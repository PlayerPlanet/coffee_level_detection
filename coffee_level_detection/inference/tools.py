# Inference infrastructure for coffeeCNN
import torch
import cv2
import numpy as np
from coffee_level_detection.training.coffee import coffeeCNN
import torchvision.transforms as T

def load_model(model_path="coffeeCNN.pth", device=None):
	"""Load trained coffeeCNN model from file."""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = coffeeCNN(num_classes=11).to(device)
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

def preprocess_image(img_paths):
	"""
	Load and preprocess one or more images for inference.
	Args:
		img_paths (str or list): Path(s) to image(s).
	Returns:
		torch.Tensor: Batch tensor of images (N, C, H, W)
	"""
	if isinstance(img_paths, str):
		img_paths = [img_paths]
	transform = T.Compose([
		T.ToTensor(),
	])
	tensors = []
	for img_path in img_paths:
		img = cv2.imread(img_path)
		if img is None:
			raise FileNotFoundError(f"Image not found: {img_path}")
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_tensor = transform(img)
		tensors.append(img_tensor)
	batch_tensor = torch.stack(tensors, dim=0)
	return batch_tensor

def predict_coffee_level(model, img_tensor, device=None):
	"""
	Predict coffee level(s) from image tensor(s) using coffeeCNN model.
	Args:
		model: Loaded coffeeCNN model
		img_tensor: torch.Tensor (N, C, H, W)
		device: torch.device
	Returns:
		List[int] if batch, int if single
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	img_tensor = img_tensor.to(device)
	with torch.no_grad():
		outputs = model(img_tensor)
		_, preds = torch.max(outputs, 1)
		preds = preds.cpu().numpy().tolist()
		if len(preds) == 1:
			return int(preds[0])
		return preds

def infer_coffee_level(img_path, model=None, device=None):
	"""
	End-to-end inference: preprocess image, predict coffee level using loaded model.
	Args:
		img_path: Path to image
		model: Loaded coffeeCNN model
		device: torch.device
	Returns:
		int: Predicted coffee level
	"""
	if model is None:
		raise ValueError("Model must be provided for efficient inference.")
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	img_tensor = preprocess_image(img_path)
	level = predict_coffee_level(model, img_tensor, device)
	return level

def infer_coffee_level_batch(img_paths, model, device=None):
	"""
	Batch inference for multiple images.
	Args:
		img_paths: List of image paths
		model: Loaded coffeeCNN model
		device: torch.device
	Returns:
		List[int]: Predicted coffee levels
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	img_tensor = preprocess_image(img_paths)
	levels = predict_coffee_level(model, img_tensor, device)
	return levels
