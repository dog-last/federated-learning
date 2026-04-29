"""Unit tests for server module — mock socket binding to avoid port conflicts."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

N = 4  # tiny sample count


def _make_config(mode="centralized"):
    return {
        "experiment": {
            "mode": mode,
            "global_epochs": 1,
            "local_epochs": 1,
            "device": "cpu",
            "dataset_params": {"batch_size": N, "num_workers": 0},
            "optimization": {"client_lr": 0.01, "server_lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        },
        "topology": {
            "server": {"host": "127.0.0.1", "port": 0},
            "clients": [{"id": "client_1", "host": "127.0.0.1", "port": 0}],
        },
        "network": {"compression": False, "stragglers": {}, "server_timeout": 30.0},
        "monitoring": {"api_host": "127.0.0.1", "api_port": 0},
    }


def _write_config_and_data(config, tmp_dir):
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    splits_dir = os.path.join(tmp_dir, "data", "splits")
    os.makedirs(splits_dir, exist_ok=True)

    server_data = {"images": torch.rand(N, 1, 28, 28), "labels": torch.randint(0, 10, (N,))}
    torch.save(server_data, os.path.join(splits_dir, "server_test_data.pt"))

    for client in config["topology"]["clients"]:
        idx = client["id"].split("_")[-1]
        n_train = max(int(N * 0.8), 1)
        n_val = max(int(N * 0.1), 1)
        client_data = {
            "train_images": torch.rand(n_train, 1, 28, 28),
            "train_labels": torch.randint(0, 10, (n_train,)),
            "val_images": torch.rand(n_val, 1, 28, 28),
            "val_labels": torch.randint(0, 10, (n_val,)),
            "test_images": torch.rand(n_val, 1, 28, 28),
            "test_labels": torch.randint(0, 10, (n_val,)),
        }
        torch.save(client_data, os.path.join(splits_dir, f"client_{idx}_data.pt"))

    return config_path


def _mock_socket():
    """Return a mock socket that supports bind/listen/setsockopt."""
    mock = MagicMock()
    mock.setsockopt = MagicMock()
    mock.bind = MagicMock()
    mock.listen = MagicMock()
    return mock


def _make_server(mode="centralized"):
    """Build a Server with mocked TCP socket. MonitorReporter.post already session-patched."""
    from core.server import Server
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = _make_config(mode)
        config_path = _write_config_and_data(config, tmp_dir)
        with patch('core.server.socket.socket', return_value=_mock_socket()):
            server = Server(config_path)
        yield server


@pytest.fixture(scope="module")
def server_centralized():
    yield from _make_server("centralized")


@pytest.fixture(scope="module")
def server_splitfed():
    yield from _make_server("splitfed")


class TestServerSelectDevice:
    def test_auto_returns_cpu(self):
        from core.server import Server
        assert Server._select_device("auto") == torch.device("cpu")

    def test_none_returns_cpu(self):
        from core.server import Server
        assert Server._select_device(None) == torch.device("cpu")


class TestServerInit:
    def test_centralized_mode(self, server_centralized):
        s = server_centralized
        assert s.mode == "centralized"
        assert s.global_model is not None

    def test_splitfed_mode(self, server_splitfed):
        s = server_splitfed
        assert s.mode == "splitfed"
        assert s.server_model is not None
        assert s.global_client_model is not None

    def test_missing_data_raises(self):
        from core.server import Server
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _make_config("centralized")
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)
            with patch('core.server.socket.socket', return_value=_mock_socket()):
                with pytest.raises(FileNotFoundError, match="Missing"):
                    Server(config_path)


class TestServerNormalize:
    def test_mnist(self):
        from core.server import Server
        images = torch.rand(2, 1, 28, 28) * 255
        result = Server._normalize(None, images)
        assert result.shape == (2, 1, 28, 28)
        assert torch.isfinite(result).all()

    def test_cifar(self):
        from core.server import Server
        images = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.float32)
        result = Server._normalize(None, images)
        assert result.shape == (2, 3, 32, 32)
        assert torch.isfinite(result).all()

    def test_unsupported_channels(self):
        from core.server import Server
        with pytest.raises(ValueError, match="Unsupported image channels"):
            Server._normalize(None, torch.rand(2, 5, 28, 28))


class TestServerAggregation:
    def test_aggregate_weighted(self, server_centralized):
        sd = server_centralized.global_model.state_dict()
        updates = [
            {"weights": {k: v.clone() for k, v in sd.items()}, "num_samples": 10},
            {"weights": {k: v.clone() for k, v in sd.items()}, "num_samples": 20},
        ]
        agg = server_centralized._aggregate_weighted(updates)
        assert agg is not None
        for key in sd.keys():
            assert key in agg

    def test_aggregate_weighted_empty(self, server_centralized):
        assert server_centralized._aggregate_weighted([]) is None


class TestServerEval:
    def test_evaluate_centralized(self, server_centralized):
        loss, acc = server_centralized._evaluate_centralized()
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_evaluate_splitfed(self, server_splitfed):
        loss, acc = server_splitfed._evaluate_splitfed()
        assert isinstance(loss, float)
        assert isinstance(acc, float)


class TestServerHelpers:
    def test_snapshot_net(self, server_centralized):
        snap = server_centralized._snapshot_net()
        assert "bytes_sent" in snap
        assert "bytes_recv" in snap
        assert "messages_sent" in snap
        assert "messages_recv" in snap

    def test_delta_net(self):
        from core.server import Server
        before = {"bytes_sent": 10, "bytes_recv": 5, "messages_sent": 2, "messages_recv": 1}
        after = {"bytes_sent": 20, "bytes_recv": 15, "messages_sent": 4, "messages_recv": 3}
        delta = Server._delta_net(after, before)
        assert delta["bytes_sent"] == 10
        assert delta["bytes_recv"] == 10
        assert delta["messages_sent"] == 2
        assert delta["messages_recv"] == 2

    def test_count_type(self, server_centralized):
        bucket = {}
        server_centralized._count_type(bucket, "test", 1)
        assert bucket["test"] == 1
        server_centralized._count_type(bucket, "test", 2)
        assert bucket["test"] == 3

    def test_report_metric(self, server_centralized):
        # MonitorReporter.post is session-patched, just verify it doesn't crash
        server_centralized.report_metric({"source": "server", "type": "test_metric"})

    def test_load_server_test_loader(self, server_centralized):
        assert server_centralized.test_loader is not None
        assert len(server_centralized.test_loader.dataset) > 0
