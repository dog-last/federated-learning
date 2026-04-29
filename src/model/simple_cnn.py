"""SimpleCNN model for MNIST (1-channel 28x28) and CIFAR-10 (3-channel 32x32)."""

import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.model.base import BaseModel


class _MNISTNet(nn.Module):
    """CNN architecture for MNIST (1x28x28 input)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x: object) -> object:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class _CIFARNet(nn.Module):
    """CNN architecture for CIFAR-10 (3x32x32 input)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x: object) -> object:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_simple_cnn(input_channels: int = 1, num_classes: int = 10) -> BaseModel:
    """Create a SimpleCNN model for the given dataset type.

    Args:
        input_channels: Number of input channels (1=MNIST, 3=CIFAR-10).
        num_classes: Number of output classes.

    Returns:
        BaseModel: Wrapped model.
    """
    net = _MNISTNet() if input_channels == 1 else _CIFARNet()
    return BaseModel(net)
