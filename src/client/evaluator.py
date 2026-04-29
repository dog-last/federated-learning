"""Model evaluator for federated clients."""

from torch.utils.data import DataLoader

from src.core.interfaces import IModel
from src.core.types import TrainingResult


class Evaluator:
    """Evaluate a model on a test dataset.

    Attributes:
        test_loader: DataLoader for the test set.
    """

    def __init__(self, test_loader: DataLoader) -> None:
        self.test_loader = test_loader

    def evaluate(self, model: IModel) -> TrainingResult:
        """Evaluate the model.

        Args:
            model: Model to evaluate.

        Returns:
            TrainingResult: Evaluation result.
        """
        return model.evaluate(self.test_loader)
