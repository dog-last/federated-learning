"""Unit tests for client: base, trainer, evaluator."""

import torch

from src.client.base import BaseClient
from src.client.evaluator import Evaluator
from src.client.trainer import LocalTrainer
from src.core.types import Weights
from src.model.simple_cnn import create_simple_cnn


class MockClient(BaseClient):
    """Mock client for testing BaseClient."""

    def connect(self, host: str, port: int) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def register(self) -> None:
        pass

    def receive_model(self) -> Weights:
        return {}

    def send_update(self, weights: Weights) -> None:
        pass

    def run(self, rounds: int) -> None:
        pass


class TestBaseClient:
    """Tests for BaseClient."""

    def test_initial_state(self) -> None:
        c = MockClient()
        assert c.client_id == -1
        assert not c.is_connected


class TestLocalTrainer:
    """Tests for LocalTrainer."""

    def test_train(self) -> None:
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(32, 1, 28, 28)
        target = torch.randint(0, 10, (32,))
        dataset = torch.utils.data.TensorDataset(data, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        trainer = LocalTrainer()
        result = trainer.train(model, loader, epochs=2, lr=0.01)
        assert result.loss > 0
        assert result.training_time > 0
        assert result.num_samples > 0

    def test_custom_momentum(self) -> None:
        trainer = LocalTrainer(momentum=0.5, weight_decay=0.001)
        assert trainer.momentum == 0.5
        assert trainer.weight_decay == 0.001


class TestEvaluator:
    """Tests for Evaluator."""

    def test_evaluate(self) -> None:
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(data, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        evaluator = Evaluator(loader)
        result = evaluator.evaluate(model)
        assert result.loss > 0
        assert 0 <= result.accuracy <= 100
