# Modified from model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        # May add Dropout to prevent overfitting
        # self.dropout = nn.Dropout(p=0.5) 
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)  # Implement Dropout before the output layer
        x = self.fc2(x)
        return x

# Define additional models for SplitFed
class SplitClientCNN(nn.Module):
    def __init__(self):
        super(SplitClientCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x

class SplitServerCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SplitServerCNN, self).__init__()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        # May add Dropout to prevent overfitting
        # self.dropout = nn.Dropout(p=0.5) 
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)  # Implement Dropout before the output layer
        x = self.fc2(x)
        return x

def get_model(mode="centralized", num_classes=10):
    if mode in ["centralized", "ring"]:
        return CNN(num_classes=num_classes)
    if mode == "splitfed":
        return SplitClientCNN(), SplitServerCNN(num_classes=num_classes)
    raise ValueError(f"Unknown mode: {mode}")
