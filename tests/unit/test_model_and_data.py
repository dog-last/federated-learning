"""Unit tests for model: base, simple_cnn, registry."""

import pytest
import torch

from src.model.base import BaseModel
from src.model.registry import get_model, register_model
from src.model.simple_cnn import _CIFARNet, _MNISTNet, create_simple_cnn


class TestBaseModel:
    """Tests for BaseModel."""

    def _make_model(self, channels: int = 1) -> BaseModel:
        return create_simple_cnn(input_channels=channels)

    def test_get_set_weights(self) -> None:
        model = self._make_model()
        w = model.get_weights()
        assert isinstance(w, dict)
        assert len(w) > 0

        model.set_weights(w)
        w2 = model.get_weights()
        for k in w:
            assert torch.allclose(w[k], w2[k])

    def test_model_size(self) -> None:
        model = self._make_model()
        assert model.model_size > 0

    def test_train_epoch(self) -> None:
        model = self._make_model()
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(data, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        result = model.train_epoch(loader, lr=0.01)
        assert result.loss > 0
        assert 0 <= result.accuracy <= 100
        assert result.num_samples == 16
        assert result.training_time > 0

    def test_evaluate(self) -> None:
        model = self._make_model()
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(data, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        result = model.evaluate(loader)
        assert result.loss > 0
        assert 0 <= result.accuracy <= 100

    def test_cifar_model(self) -> None:
        model = self._make_model(channels=3)
        data = torch.randn(8, 3, 32, 32)
        target = torch.randint(0, 10, (8,))
        dataset = torch.utils.data.TensorDataset(data, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        result = model.train_epoch(loader, lr=0.01)
        assert result.num_samples == 8

        eval_result = model.evaluate(loader)
        assert eval_result.num_samples == 8


class TestSimpleCNN:
    """Tests for SimpleCNN architectures."""

    def test_mnist_net_forward(self) -> None:
        net = _MNISTNet()
        x = torch.randn(2, 1, 28, 28)
        y = net(x)
        assert y.shape == (2, 10)

    def test_cifar_net_forward(self) -> None:
        net = _CIFARNet()
        x = torch.randn(2, 3, 32, 32)
        y = net(x)
        assert y.shape == (2, 10)

    def test_create_simple_cnn_mnist(self) -> None:
        model = create_simple_cnn(input_channels=1)
        assert isinstance(model, BaseModel)

    def test_create_simple_cnn_cifar(self) -> None:
        model = create_simple_cnn(input_channels=3)
        assert isinstance(model, BaseModel)


class TestRegistry:
    """Tests for model registry."""

    def test_get_simple_cnn(self) -> None:
        model = get_model("simple_cnn", input_channels=1)
        assert isinstance(model, BaseModel)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError):
            get_model("nonexistent")

    def test_register_custom(self) -> None:
        def custom_factory(**kwargs: object) -> BaseModel:
            return create_simple_cnn(input_channels=1)

        register_model("custom_test", custom_factory)
        model = get_model("custom_test")
        assert isinstance(model, BaseModel)
