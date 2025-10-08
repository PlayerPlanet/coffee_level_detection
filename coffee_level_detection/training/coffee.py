
"""
coffee.py
Module for CoffeeImageDataset and related utilities for coffee level detection.
Provides a PyTorch Dataset for loading coffee images and labels from a DataFrame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import os
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import torchvision.models as models
from math import sqrt

class CoffeeImageDataset(Dataset):
    """
    PyTorch Dataset for coffee level detection images and labels.
    Loads images and coffee level labels from a pandas DataFrame and image directory.
    Args:
        df (pd.DataFrame): DataFrame with columns 'filename' and 'coffee_level'.
        images_dir (str): Directory containing image files.
        transform: Optional torchvision transform to apply to images.
    """
    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        """
        Initialize CoffeeImageDataset.
        Args:
            df (pd.DataFrame): DataFrame with image filenames and coffee levels.
            images_dir (str): Path to image directory.
            transform: Optional transform for images.
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform or T.Compose([T.ToTensor()])
        self.__validate_dataset(self.df)

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get image and label for a given index.
        Args:
            idx (int): Index of sample.
        Returns:
            Tuple (image, label): Transformed image and integer coffee level label.
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, str(row['filename']))
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['coffee_level'])
        return img, label

    def __validate_dataset(self, dataset: pd.DataFrame):
        """
        Validate that the DataFrame contains required columns and valid coffee level values.
        Args:
            dataset (pd.DataFrame): DataFrame to validate.
        Raises:
            ValueError: If required columns are missing or values are out of range.
        """
        required = {'coffee_level', 'filename'}
        missing = required - set(dataset.columns.astype(str))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
        
        if not np.issubdtype(dataset['coffee_level'].dtype, np.integer):
            if not np.all(np.mod(dataset['coffee_level'], 1) == 0):
                raise ValueError("coffee_level column must contain integer values (0-10).")
        vals = dataset['coffee_level'].astype(int).unique()
        if vals.min() < 0 or vals.max() > 10:
            raise ValueError("coffee_level values must be in range 0..10.")
        
class coffeeCNN(torch.nn.Module):
    def __init__(self, num_classes=11, H=480, W=320):
        super(coffeeCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.fc1 = torch.nn.Linear(64 * (H//4)* (W//4), 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.bn1   = torch.nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class coffeeCNNv2(torch.nn.Module):
    """
    Improved coffeeCNNv2:
    - BatchNorm after each conv
    - Optional 3rd conv for capacity
    - AdaptiveAvgPool2d to give spatial-invariant features
    - Dropout and Kaiming initialization for stability
    """
    def __init__(self, num_classes=11, H=480, W=320, dropout=0.4, pretrained_backbone=False):
        super(coffeeCNNv2, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)

        # optional additional conv layer to increase capacity
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = torch.nn.Linear(128, 128)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(128, num_classes)

        # initialize weights
        self._init_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        # one more pool is optional; adaptive pool makes final size invariant
        x = self.global_pool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    def _init_weights(self):
        """Kaiming init for conv/linear, sensible BatchNorm init."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=sqrt(5))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

class coffeeResNet(nn.Module):
    """ResNet-based coffee level classifier.

    Wraps a torchvision ResNet (default: resnet18) and replaces the final
    fully-connected layer to output `num_classes` logits.

    Args:
        num_classes (int): Number of output classes (default 11 for levels 0-10).
        backbone (str): Which ResNet backbone to use: 'resnet18' or 'resnet34'.
        pretrained (bool): If True, load ImageNet pretrained weights for the backbone.
    """
    def __init__(self, num_classes=11, backbone='resnet18', pretrained=False):
        super(coffeeResNet, self).__init__()

        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_features = self.model.fc.in_features
        # Replace the final fully-connected layer with one that outputs num_classes
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Forward pass returning raw logits (no softmax)."""
        return self.model(x)