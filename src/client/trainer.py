"""Local trainer for federated clients."""

from torch.utils.data import DataLoader

from src.core.interfaces import IModel, ITrainer
from src.core.types import TrainingResult


class LocalTrainer(ITrainer):
    """Local trainer that runs multiple epochs on a model.

    Attributes:
        momentum: SGD momentum.
        weight_decay: L2 regularization coefficient.
    """

    def __init__(self, momentum: float = 0.9, weight_decay: float = 0.0001) -> None:
        self.momentum = momentum
        self.weight_decay = weight_decay

    def train(
        self, model: IModel, dataloader: DataLoader, epochs: int, lr: float
    ) -> TrainingResult:
        """Execute local training.

        Args:
            model: Model instance.
            dataloader: Training data.
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            TrainingResult: Aggregated training result over all epochs.
        """
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        total_time = 0.0

        for epoch in range(1, epochs + 1):
            result = model.train_epoch(dataloader, lr, epoch)
            total_loss += result.loss * result.num_samples
            total_acc += result.accuracy * result.num_samples / 100.0
            total_samples += result.num_samples
            total_time += result.training_time

        return TrainingResult(
            loss=total_loss / max(total_samples, 1),
            accuracy=100.0 * total_acc / max(total_samples, 1),
            num_samples=total_samples // epochs if epochs > 0 else 0,
            training_time=total_time,
        )
