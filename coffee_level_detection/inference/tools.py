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

def preprocess_image(img_path):
	"""Load and preprocess image for inference."""
	img = cv2.imread(img_path)
	if img is None:
		raise FileNotFoundError(f"Image not found: {img_path}")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	transform = T.Compose([
		T.ToTensor(),
	])
	img_tensor = transform(img)
	img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
	return img_tensor

def predict_coffee_level(model, img_tensor, device=None):
	"""Predict coffee level from image tensor using coffeeCNN model."""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	img_tensor = img_tensor.to(device)
	with torch.no_grad():
		outputs = model(img_tensor)
		_, pred = torch.max(outputs, 1)
		return int(pred.item())

def infer_coffee_level(img_path, model_path="coffeeCNN.pth"):
	"""End-to-end inference: load model, preprocess image, predict coffee level."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = load_model(model_path, device)
	img_tensor = preprocess_image(img_path)
	level = predict_coffee_level(model, img_tensor, device)
	return level
