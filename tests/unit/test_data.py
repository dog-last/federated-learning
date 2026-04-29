"""Unit tests for data: dataset, partitioner, loader."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import TensorDataset

from src.data.dataset import (
    get_input_channels,
    get_num_classes,
    get_transforms,
    load_and_partition,
    load_dataset,
)
from src.data.loader import create_dataloader
from src.data.partitioner import Partitioner


def _fake_torchvision_dataset(n: int = 100, num_classes: int = 10, channels: int = 1):
    """Return a MagicMock that mimics a torchvision dataset with `n` samples."""
    data = torch.randn(n, channels, 28, 28)
    targets = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(data, targets)
    mock = MagicMock(return_value=ds)
    return mock


class TestPartitioner:
    """Tests for Partitioner."""

    def _make_dataset(self, n: int = 100, num_classes: int = 10) -> TensorDataset:
        data = torch.randn(n, 1, 28, 28)
        targets = torch.randint(0, num_classes, (n,))
        return TensorDataset(data, targets)

    def test_iid_partition(self) -> None:
        ds = self._make_dataset(100)
        p = Partitioner()
        parts = p.partition(ds, 3, strategy="iid")
        assert len(parts) == 3
        total = sum(len(p) for p in parts)
        assert total == 100

    def test_non_iid_partition(self) -> None:
        ds = self._make_dataset(100)
        p = Partitioner()
        parts = p.partition(ds, 3, strategy="non_iid")
        assert len(parts) == 3
        total = sum(len(p) for p in parts)
        assert total == 100

    def test_get_client_data(self) -> None:
        ds = self._make_dataset(100)
        p = Partitioner()
        p.partition(ds, 3, strategy="iid")
        indices = p.get_client_data(0)
        assert indices is not None
        assert len(indices) > 0

    def test_get_client_data_not_partitioned(self) -> None:
        p = Partitioner()
        assert p.get_client_data(0) is None

    def test_unknown_strategy_raises(self) -> None:
        ds = self._make_dataset(10)
        p = Partitioner()
        with pytest.raises(ValueError):
            p.partition(ds, 2, strategy="unknown")

    def test_iid_balanced(self) -> None:
        ds = self._make_dataset(99)
        p = Partitioner()
        parts = p.partition(ds, 3, strategy="iid")
        sizes = [len(part) for part in parts]
        assert max(sizes) - min(sizes) <= 1

    def test_non_iid_with_targets_attr(self) -> None:
        """Test Non-IID partition with a dataset that has .targets."""
        n = 100
        data = torch.randn(n, 1, 28, 28)
        targets = torch.tensor([i % 10 for i in range(n)])

        class MockDataset:
            def __init__(self) -> None:
                self.data = data
                self.targets = targets
                self._size = n

            def __len__(self) -> int:
                return self._size

            def __getitem__(self, idx: int) -> tuple:
                return self.data[idx], self.targets[idx]

        ds = MockDataset()
        p = Partitioner()
        parts = p.partition(ds, 3, strategy="non_iid")
        assert len(parts) == 3


class TestDataset:
    """Tests for dataset factory functions."""

    def test_get_transforms_mnist(self) -> None:
        t = get_transforms("mnist")
        assert t is not None

    def test_get_transforms_cifar10(self) -> None:
        t = get_transforms("cifar10")
        assert t is not None

    def test_get_transforms_unknown(self) -> None:
        with pytest.raises(ValueError):
            get_transforms("unknown")

    def test_get_input_channels(self) -> None:
        assert get_input_channels("mnist") == 1
        assert get_input_channels("cifar10") == 3
        with pytest.raises(ValueError):
            get_input_channels("unknown")

    def test_get_num_classes(self) -> None:
        assert get_num_classes("mnist") == 10
        assert get_num_classes("cifar10") == 10
        with pytest.raises(ValueError):
            get_num_classes("unknown")

    @patch("src.data.dataset.datasets.CIFAR10", _fake_torchvision_dataset(channels=3))
    @patch("src.data.dataset.datasets.MNIST", _fake_torchvision_dataset())
    def test_load_dataset_mnist(self, tmp_dir: str) -> None:
        ds = load_dataset("mnist", tmp_dir, train=True)
        assert len(ds) > 0

    @patch("src.data.dataset.datasets.CIFAR10", _fake_torchvision_dataset(channels=3))
    @patch("src.data.dataset.datasets.MNIST", _fake_torchvision_dataset())
    def test_load_dataset_cifar10(self, tmp_dir: str) -> None:
        ds = load_dataset("cifar10", tmp_dir, train=True)
        assert len(ds) > 0

    def test_load_dataset_unknown(self, tmp_dir: str) -> None:
        with pytest.raises(ValueError):
            load_dataset("unknown", tmp_dir)

    @patch("src.data.dataset.datasets.CIFAR10", _fake_torchvision_dataset(channels=3))
    @patch("src.data.dataset.datasets.MNIST", _fake_torchvision_dataset())
    def test_load_and_partition(self, tmp_dir: str) -> None:
        parts, test = load_and_partition("mnist", tmp_dir, num_clients=2, strategy="iid")
        assert len(parts) == 2
        assert len(test) > 0

    @patch("src.data.dataset.datasets.CIFAR10", _fake_torchvision_dataset(channels=3))
    @patch("src.data.dataset.datasets.MNIST", _fake_torchvision_dataset())
    def test_load_and_partition_non_iid(self, tmp_dir: str) -> None:
        parts, test = load_and_partition("mnist", tmp_dir, num_clients=2, strategy="non_iid")
        assert len(parts) == 2


class TestDataLoader:
    """Tests for create_dataloader."""

    def test_create(self) -> None:
        data = torch.randn(20, 1, 28, 28)
        target = torch.randint(0, 10, (20,))
        ds = TensorDataset(data, target)
        loader = create_dataloader(ds, batch_size=8, shuffle=False)
        assert len(loader) == 3  # 20 / 8 = 2.5 -> 3 batches

    def test_create_shuffle(self) -> None:
        data = torch.randn(20, 1, 28, 28)
        target = torch.randint(0, 10, (20,))
        ds = TensorDataset(data, target)
        loader = create_dataloader(ds, batch_size=10, shuffle=True)
        assert len(loader) == 2
