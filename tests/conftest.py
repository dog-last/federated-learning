"""Shared pytest fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options for pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (may be slow and require network resources)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection based on command line options."""
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(skip_integration)


@pytest.fixture
def tmp_dir() -> Generator[str]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def tmp_yaml(tmp_dir: str) -> str:
    """Write a minimal centralized config YAML and return its path."""
    import yaml

    cfg = {
        "mode": "centralized",
        "model": {"name": "simple_cnn", "input_channels": 1, "num_classes": 10},
        "dataset": {
            "name": "mnist",
            "data_dir": tmp_dir,
            "num_clients": 2,
            "partition_strategy": "iid",
            "alpha": 0.5,
        },
        "training": {
            "rounds": 2,
            "epochs_per_round": 1,
            "learning_rate": 0.01,
            "batch_size": 32,
            "momentum": 0.9,
            "weight_decay": 0.0,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 9000,
            "address": "127.0.0.1",
            "timeouts": {"connect": 5.0, "send": 5.0, "recv": 5.0, "round": 30.0},
        },
        "clients": {
            "nodes": [
                {"id": 1, "host": "127.0.0.1"},
                {"id": 2, "host": "127.0.0.1"},
            ],
        },
        "aggregator": {"name": "fedavg", "fedprox_mu": 0.01},
        "logging": {
            "level": "WARNING",
            "log_dir": tmp_dir,
            "console_output": False,
            "file_output": False,
        },
        "output": {"checkpoint_dir": tmp_dir, "figure_dir": tmp_dir, "save_checkpoint_every": 1},
    }
    path = Path(tmp_dir) / "test_config.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)
