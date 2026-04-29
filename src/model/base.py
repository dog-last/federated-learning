"""Model base class wrapping a torch.nn.Module with IModel interface."""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.interfaces import IModel
from src.core.types import TrainingResult, Weights


class BaseModel(IModel):
    """Base model wrapper for nn.Module.

    Subclasses must set self.net in __init__.

    Attributes:
        net: The underlying nn.Module.
        device: Device the model is on.
    """

    def __init__(self, net: nn.Module, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)

    def get_weights(self) -> Weights:
        """Get model weights.

        Returns:
            Weights: Model state dict.
        """
        return {k: v.cpu() for k, v in self.net.state_dict().items()}

    def set_weights(self, weights: Weights) -> None:
        """Set model weights.

        Args:
            weights: Model state dict.
        """
        self.net.load_state_dict({k: v.to(self.device) for k, v in weights.items()})

    def train_epoch(self, dataloader: DataLoader, lr: float, epoch: int = 1) -> TrainingResult:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.
            lr: Learning rate.
            epoch: Current epoch number.

        Returns:
            TrainingResult: Training result.
        """
        self.net.train()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        elapsed = time.time() - start
        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)

        return TrainingResult(
            loss=avg_loss,
            accuracy=accuracy,
            num_samples=total,
            training_time=elapsed,
        )

    def evaluate(self, dataloader: DataLoader) -> TrainingResult:
        """Evaluate the model.

        Args:
            dataloader: Evaluation data loader.

        Returns:
            TrainingResult: Evaluation result.
        """
        self.net.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                loss = criterion(output, target)

                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        elapsed = time.time() - start
        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)

        return TrainingResult(
            loss=avg_loss,
            accuracy=accuracy,
            num_samples=total,
            training_time=elapsed,
        )

    @property
    def model_size(self) -> int:
        """Number of model parameters.

        Returns:
            int: Parameter count.
        """
        return sum(p.numel() for p in self.net.parameters())
