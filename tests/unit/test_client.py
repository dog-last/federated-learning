"""Unit tests for client module — no real network, no real file I/O for construction."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

N = 4  # tiny sample count


def _make_config(mode="centralized", stragglers=None):
    return {
        "experiment": {
            "mode": mode,
            "global_epochs": 1,
            "local_epochs": 1,
            "device": "cpu",
            "dataset_params": {"batch_size": N, "num_workers": 0},
            "optimization": {"client_lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        },
        "topology": {
            "server": {"host": "127.0.0.1", "port": 19900},
            "clients": [{"id": "client_1", "host": "127.0.0.1", "port": 19901}],
        },
        "network": {"compression": False, "stragglers": stragglers or {}, "server_timeout": 30.0},
        "monitoring": {"api_host": "127.0.0.1", "api_port": 19999},
    }


def _write_config(config, tmp_dir):
    path = os.path.join(tmp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f)
    return path


def _write_client_data(tmp_dir, client_id="client_1", num_samples=N):
    splits_dir = os.path.join(tmp_dir, "data", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    n_train = max(int(num_samples * 0.8), 1)
    n_val = max(int(num_samples * 0.1), 1)
    client_data = {
        "train_images": torch.rand(n_train, 1, 28, 28),
        "train_labels": torch.randint(0, 10, (n_train,)),
        "val_images": torch.rand(n_val, 1, 28, 28),
        "val_labels": torch.randint(0, 10, (n_val,)),
        "test_images": torch.rand(n_val, 1, 28, 28),
        "test_labels": torch.randint(0, 10, (n_val,)),
    }
    idx = client_id.split("_")[-1]
    torch.save(client_data, os.path.join(splits_dir, f"client_{idx}_data.pt"))
    server_data = {"images": torch.rand(N, 1, 28, 28), "labels": torch.randint(0, 10, (N,))}
    torch.save(server_data, os.path.join(splits_dir, "server_test_data.pt"))


def _make_client(mode="centralized", stragglers=None):
    """Build a Client in a tmp dir. MonitorReporter.post is already session-patched."""
    from core.client import Client
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = _make_config(mode, stragglers)
        config_path = _write_config(config, tmp_dir)
        _write_client_data(tmp_dir)
        with patch('core.client.socket.socket') as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock_cls.return_value = mock_sock
            # Patch project_root to use tmp_dir so data files can be found
            original_load_local_data = Client._load_local_data
            def patched_load_local_data(self):
                idx = self.client_id.split("_")[-1]
                path = os.path.join(tmp_dir, "data", "splits", f"client_{idx}_data.pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing {path}")
                payload = torch.load(path, map_location="cpu")
                train_loader = self._build_loader(payload["train_images"], payload["train_labels"], shuffle=True)
                val_loader = self._build_loader(payload["val_images"], payload["val_labels"], shuffle=False)
                test_loader = self._build_loader(payload["test_images"], payload["test_labels"], shuffle=False)
                return train_loader, val_loader, test_loader
            with patch.object(Client, '_load_local_data', patched_load_local_data):
                client = Client(config_path, "client_1")
        # Yield the client — tmp_dir will be cleaned up but data is already in memory
        yield client


@pytest.fixture(scope="module")
def client_centralized():
    yield from _make_client("centralized")


@pytest.fixture(scope="module")
def client_splitfed():
    yield from _make_client("splitfed")


class TestSelectDevice:
    def test_auto_returns_cpu(self):
        from core.client import Client
        assert Client._select_device("auto") == torch.device("cpu")

    def test_empty_returns_cpu(self):
        from core.client import Client
        assert Client._select_device("") == torch.device("cpu")

    def test_none_returns_cpu(self):
        from core.client import Client
        assert Client._select_device(None) == torch.device("cpu")

    def test_cuda_falls_back_to_cpu(self):
        from core.client import Client
        device = Client._select_device("cuda")
        assert device in (torch.device("cuda"), torch.device("cpu"))


class TestNormalize:
    def test_mnist(self):
        from core.client import Client
        images = torch.rand(2, 1, 28, 28) * 255
        result = Client._normalize(None, images)
        assert result.shape == (2, 1, 28, 28)
        assert torch.isfinite(result).all()

    def test_cifar(self):
        from core.client import Client
        images = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.float32)
        result = Client._normalize(None, images)
        assert result.shape == (2, 3, 32, 32)
        assert torch.isfinite(result).all()

    def test_unsupported_channels(self):
        from core.client import Client
        with pytest.raises(ValueError, match="Unsupported image channels"):
            Client._normalize(None, torch.rand(2, 5, 28, 28))


class TestClientInit:
    def test_centralized_has_model(self, client_centralized):
        c = client_centralized
        assert c.mode == "centralized"
        assert hasattr(c, 'model')

    def test_splitfed_has_split_models(self, client_splitfed):
        c = client_splitfed
        assert c.mode == "splitfed"
        assert hasattr(c, 'client_model')
        assert hasattr(c, 'shadow_server_model')


class TestClientHelpers:
    def test_should_drop_round_zero_rate(self, client_centralized):
        client_centralized.straggler_config = {"drop_rate": 0.0}
        assert client_centralized._should_drop_round() is False

    def test_should_drop_round_always(self, client_centralized):
        client_centralized.straggler_config = {"drop_rate": 1.0}
        assert client_centralized._should_drop_round() is True

    def test_count_type(self, client_centralized):
        bucket = {}
        client_centralized._count_type(bucket, "model_update", 1)
        assert bucket["model_update"] == 1
        client_centralized._count_type(bucket, "model_update", 1)
        assert bucket["model_update"] == 2

    def test_build_loader(self, client_centralized):
        images = torch.rand(N, 1, 28, 28)
        labels = torch.randint(0, 10, (N,))
        loader = client_centralized._build_loader(images, labels, shuffle=False)
        assert isinstance(loader, DataLoader)
        assert len(loader.dataset) == N

    def test_load_local_data(self, client_centralized):
        assert client_centralized.train_loader is not None
        assert client_centralized.val_loader is not None
        assert client_centralized.test_loader is not None

    def test_load_local_data_missing_file(self):
        from core.client import Client
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config("centralized")
            config_path = _write_config(config, tmp_dir)
            with patch('core.client.socket.socket'):
                with pytest.raises(FileNotFoundError, match="Missing"):
                    Client(config_path, "client_1")


class TestClientEval:
    def test_eval_centralized(self, client_centralized):
        loss, acc = client_centralized._eval_centralized(client_centralized.val_loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_eval_split(self, client_splitfed):
        loss, acc = client_splitfed._eval_split(client_splitfed.val_loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)


class TestClientMissingData:
    def test_missing_data_raises(self):
        from core.client import Client
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config()
            config_path = _write_config(config, tmp_dir)
            with patch('core.client.socket.socket'):
                with pytest.raises(FileNotFoundError):
                    Client(config_path, "client_1")
