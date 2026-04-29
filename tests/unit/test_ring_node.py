"""Unit tests for ring_node module — mock sockets, no real network."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

N = 4


def _make_ring_config():
    return {
        "experiment": {
            "mode": "ring",
            "global_epochs": 2,
            "local_epochs": 1,
            "device": "cpu",
            "target_accuracy": 0.99,
            "dataset_params": {"batch_size": N, "num_workers": 0},
            "optimization": {
                "client_lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
        },
        "topology": {
            "nodes": [
                {"id": 1, "host": "127.0.0.1", "port": 19001},
                {"id": 2, "host": "127.0.0.1", "port": 19002},
                {"id": 3, "host": "127.0.0.1", "port": 19003},
            ]
        },
        "network": {
            "compression": False,
            "stragglers": {
                "client_1": {"delay": 0.0, "drop_rate": 0.0},
                "client_2": {"delay": 5.0, "drop_rate": 0.0},
                "client_3": {"delay": 0.0, "drop_rate": 0.0},
            },
            "server_timeout": 30.0,
        },
        "monitoring": {"api_host": "127.0.0.1", "api_port": 19999},
    }


def _write_config_and_data(config, tmp_dir):
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    splits_dir = os.path.join(tmp_dir, "data", "splits")
    os.makedirs(splits_dir, exist_ok=True)

    server_data = {
        "images": torch.rand(N, 1, 28, 28),
        "labels": torch.randint(0, 10, (N,)),
    }
    torch.save(server_data, os.path.join(splits_dir, "server_test_data.pt"))

    for i in range(1, 4):
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
        torch.save(client_data, os.path.join(splits_dir, f"client_{i}_data.pt"))

    return config_path


def _mock_socket():
    mock = MagicMock()
    mock.setsockopt = MagicMock()
    mock.bind = MagicMock()
    mock.listen = MagicMock()
    mock.close = MagicMock()
    mock.settimeout = MagicMock()
    return mock


def _make_ring_node(node_id=1, stragglers=None):
    from core.ring_node import RingNode

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = _make_ring_config()
        if stragglers is not None:
            config["network"]["stragglers"] = stragglers
        config_path = _write_config_and_data(config, tmp_dir)

        def patched_load_local_data(self):
            idx = self.node_id
            path = os.path.join(tmp_dir, "data", "splits", f"client_{idx}_data.pt")
            payload = torch.load(path, map_location="cpu")
            train_loader = self._build_loader(
                payload["train_images"], payload["train_labels"], shuffle=True
            )
            val_loader = self._build_loader(
                payload["val_images"], payload["val_labels"], shuffle=False
            )
            test_loader = self._build_loader(
                payload["test_images"], payload["test_labels"], shuffle=False
            )
            return train_loader, val_loader, test_loader

        def patched_load_server_test_data(self):
            path = os.path.join(tmp_dir, "data", "splits", "server_test_data.pt")
            payload = torch.load(path, map_location="cpu")
            x = self._normalize(payload["images"])
            y = payload["labels"].long()
            ds = torch.utils.data.TensorDataset(x, y)
            self.server_test_loader = torch.utils.data.DataLoader(
                ds, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

        with patch("core.ring_node.socket.socket", return_value=_mock_socket()):
            with patch.object(RingNode, "_load_local_data", patched_load_local_data):
                with patch.object(
                    RingNode, "_load_server_test_data", patched_load_server_test_data
                ):
                    node = RingNode(config_path, node_id)
        yield node


@pytest.fixture(scope="module")
def ring_node_1():
    yield from _make_ring_node(1)


@pytest.fixture(scope="module")
def ring_node_2():
    yield from _make_ring_node(2, stragglers={"client_2": {"delay": 5.0, "drop_rate": 0.0}})


class TestRingNodeSelectDevice:
    def test_auto_returns_valid_device(self):
        from core.ring_node import RingNode
        device = RingNode._select_device("auto")
        assert device.type in ("cpu", "cuda", "mps")

    def test_none_returns_valid_device(self):
        from core.ring_node import RingNode
        device = RingNode._select_device(None)
        assert device.type in ("cpu", "cuda", "mps")

    def test_explicit_cpu(self):
        from core.ring_node import RingNode
        assert RingNode._select_device("cpu") == torch.device("cpu")


class TestRingNodeInit:
    def test_node_id(self, ring_node_1):
        assert ring_node_1.node_id == 1

    def test_has_model(self, ring_node_1):
        assert ring_node_1.model is not None

    def test_is_initiator(self, ring_node_1):
        assert ring_node_1.node_id == 1

    def test_not_initiator(self, ring_node_2):
        assert ring_node_2.node_id != 1

    def test_next_node_addr(self, ring_node_1):
        assert ring_node_1.next_host == "127.0.0.1"
        assert ring_node_1.next_port == 19002

    def test_next_node_addr_wraps(self, ring_node_2):
        # Node 2's successor is Node 3
        assert ring_node_2.next_port == 19003


class TestRingNodeStraggler:
    def test_straggler_config(self, ring_node_2):
        assert ring_node_2.straggler_cfg["delay"] == 5.0

    def test_no_straggler(self, ring_node_1):
        assert ring_node_1.straggler_cfg["delay"] == 0.0

    def test_should_drop_round_zero(self, ring_node_1):
        ring_node_1.straggler_cfg = {"drop_rate": 0.0}
        assert ring_node_1._should_drop_round() is False

    def test_should_drop_round_always(self, ring_node_1):
        ring_node_1.straggler_cfg = {"drop_rate": 1.0}
        assert ring_node_1._should_drop_round() is True


class TestRingNodeNormalize:
    def test_mnist(self):
        from core.ring_node import RingNode
        images = torch.rand(2, 1, 28, 28) * 255
        result = RingNode._normalize(None, images)
        assert result.shape == (2, 1, 28, 28)
        assert torch.isfinite(result).all()

    def test_cifar(self):
        from core.ring_node import RingNode
        images = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.float32)
        result = RingNode._normalize(None, images)
        assert result.shape == (2, 3, 32, 32)
        assert torch.isfinite(result).all()

    def test_unsupported_channels(self):
        from core.ring_node import RingNode
        with pytest.raises(ValueError, match="Unsupported image channels"):
            RingNode._normalize(None, torch.rand(2, 5, 28, 28))


class TestRingNodeTraining:
    def test_train_local(self, ring_node_1):
        train_loss, train_acc, test_loss, test_acc = ring_node_1._train_local(round_id=1)
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0.0 <= train_acc <= 1.0

    def test_evaluate(self, ring_node_1):
        loss, acc = ring_node_1._evaluate(ring_node_1.val_loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_has_data_loaders(self, ring_node_1):
        assert ring_node_1.train_loader is not None
        assert ring_node_1.val_loader is not None
        assert ring_node_1.test_loader is not None

    def test_has_server_test_loader(self, ring_node_1):
        assert ring_node_1.server_test_loader is not None


class TestRingNodeNetwork:
    def test_net_stats_initialized(self, ring_node_1):
        assert ring_node_1.net_stats["bytes_sent"] == 0
        assert ring_node_1.net_stats["bytes_recv"] == 0

    def test_record_network_out(self, ring_node_1):
        ring_node_1._record_network("out", "ring_pass", 1024)
        assert ring_node_1.net_stats["bytes_sent"] == 1024
        assert ring_node_1.net_stats["messages_sent"] == 1

    def test_record_network_in(self, ring_node_1):
        ring_node_1._record_network("in", "ring_pass", 512)
        assert ring_node_1.net_stats["bytes_recv"] == 512
        assert ring_node_1.net_stats["messages_recv"] == 1
