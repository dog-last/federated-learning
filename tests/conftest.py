"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def mock_mnist_datasets():
    """Mock torchvision.datasets.MNIST to avoid network downloads."""
    num_samples = 10
    np_rng = np.random.RandomState(42)
    from PIL import Image
    _images = [Image.fromarray(np_rng.randint(0, 256, (28, 28), dtype=np.uint8), mode='L') for _ in range(num_samples)]
    _labels = torch.randint(0, 10, (num_samples,)).tolist()

    class MockDataset:
        def __init__(self, root, train, download, transform):
            self.root = root
            self.train = train
            self.download = download
            self.transform = transform

        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            img = _images[idx]
            label = _labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

        def __iter__(self):
            for i in range(len(self)):
                yield self[i]

    with patch('torchvision.datasets.MNIST', MockDataset):
        yield


@pytest.fixture(autouse=True, scope="session")
def block_monitor_post():
    """Block all MonitorReporter HTTP requests for the entire test session."""
    with patch('utils.monitoring.MonitorReporter.post'):
        yield


@pytest.fixture(scope="session")
def test_data_dir():
    """Session-scoped temporary directory for test data."""
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp(prefix="fl_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
