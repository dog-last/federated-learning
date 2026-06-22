"""Unit tests for server straggler/timeout handling."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import time
import threading

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

N = 4


def _make_config(num_clients=3, server_timeout=15.0, min_clients=1):
    clients = [
        {"id": f"client_{i+1}", "host": "127.0.0.1", "port": 0}
        for i in range(num_clients)
    ]
    return {
        "experiment": {
            "mode": "centralized",
            "global_epochs": 1,
            "local_epochs": 1,
            "device": "cpu",
            "dataset_params": {"batch_size": N, "num_workers": 0},
            "optimization": {
                "client_lr": 0.01,
                "server_lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
        },
        "topology": {
            "server": {"host": "127.0.0.1", "port": 0},
            "clients": clients,
        },
        "network": {
            "compression": False,
            "stragglers": {},
            "server_timeout": server_timeout,
            "min_clients": min_clients,
        },
        "monitoring": {"api_host": "127.0.0.1", "api_port": 0},
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
    mock = MagicMock()
    mock.setsockopt = MagicMock()
    mock.bind = MagicMock()
    mock.listen = MagicMock()
    return mock


def _make_server(num_clients=3, server_timeout=2.0, min_clients=1):
    from core.server import Server

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = _make_config(num_clients, server_timeout, min_clients)
        config_path = _write_config_and_data(config, tmp_dir)
        with patch("core.server.socket.socket", return_value=_mock_socket()):
            server = Server(config_path)
        yield server


@pytest.fixture(scope="module")
def server_3clients():
    yield from _make_server(num_clients=3, server_timeout=2.0, min_clients=1)


@pytest.fixture(scope="module")
def server_3clients_min2():
    yield from _make_server(num_clients=3, server_timeout=2.0, min_clients=2)


class TestServerMinClientsConfig:
    def test_min_clients_default(self, server_3clients):
        assert server_3clients.min_clients == 1

    def test_min_clients_custom(self, server_3clients_min2):
        assert server_3clients_min2.min_clients == 2

    def test_dropped_clients_initialized(self, server_3clients):
        assert isinstance(server_3clients.dropped_clients, set)
        assert len(server_3clients.dropped_clients) == 0


class TestServerWaitRoundUpdates:
    def test_timeout_drops_stragglers(self, server_3clients):
        """Simulate 2 of 3 clients responding before timeout; 1 straggler dropped."""
        server = server_3clients
        server.timeout = 1.0  # short timeout for fast test

        # Register 3 mock clients
        server.active_clients = {
            "client_1": MagicMock(),
            "client_2": MagicMock(),
            "client_3": MagicMock(),
        }

        round_id = 100
        server.round_updates[round_id] = {}

        # Simulate 2 clients responding quickly in a background thread
        def add_updates():
            time.sleep(0.3)
            with server.update_cv:
                server.round_updates[round_id]["client_1"] = {
                    "weights": {k: v.clone() for k, v in server.global_model.state_dict().items()},
                    "num_samples": 100,
                }
                server.round_updates[round_id]["client_3"] = {
                    "weights": {k: v.clone() for k, v in server.global_model.state_dict().items()},
                    "num_samples": 100,
                }
                server.update_cv.notify_all()

        t = threading.Thread(target=add_updates, daemon=True)
        t.start()

        got = server._wait_round_updates(round_id)
        t.join(timeout=3.0)

        # client_1 and client_3 responded; client_2 timed out
        assert len(got) == 2
        assert "client_1" in got
        assert "client_3" in got
        assert "client_2" not in got
        assert "client_2" in server.dropped_clients

    def test_all_clients_respond(self, server_3clients):
        """When all clients respond, no stragglers are dropped."""
        server = server_3clients
        server.timeout = 5.0
        server.dropped_clients = set()

        server.active_clients = {
            "client_1": MagicMock(),
            "client_2": MagicMock(),
            "client_3": MagicMock(),
        }

        round_id = 200
        server.round_updates[round_id] = {}

        def add_all_updates():
            time.sleep(0.2)
            with server.update_cv:
                for cid in ["client_1", "client_2", "client_3"]:
                    server.round_updates[round_id][cid] = {
                        "weights": {k: v.clone() for k, v in server.global_model.state_dict().items()},
                        "num_samples": 100,
                    }
                server.update_cv.notify_all()

        t = threading.Thread(target=add_all_updates, daemon=True)
        t.start()
        got = server._wait_round_updates(round_id)
        t.join(timeout=5.0)

        assert len(got) == 3
        assert len(server.dropped_clients) == 0


class TestServerStragglerAggregation:
    def test_aggregate_with_dropped_client(self, server_3clients):
        """Server can aggregate even when some clients are dropped."""
        server = server_3clients
        sd = server.global_model.state_dict()

        # Only 2 of 3 clients submitted updates
        updates = [
            {"weights": {k: v.clone() for k, v in sd.items()}, "num_samples": 100},
            {"weights": {k: v.clone() for k, v in sd.items()}, "num_samples": 150},
        ]
        agg = server._aggregate_weighted(updates)
        assert agg is not None
        for key in sd.keys():
            assert key in agg

    def test_aggregate_with_one_client(self, server_3clients):
        """Server can aggregate with just 1 client (min_clients=1)."""
        server = server_3clients
        sd = server.global_model.state_dict()
        updates = [
            {"weights": {k: v.clone() for k, v in sd.items()}, "num_samples": 100},
        ]
        agg = server._aggregate_weighted(updates)
        assert agg is not None

    def test_round_proceeds_with_partial_updates(self, server_3clients):
        """Verify that a round can proceed when some clients are dropped."""
        server = server_3clients
        server.active_clients = {
            "client_1": MagicMock(),
            "client_2": MagicMock(),
            "client_3": MagicMock(),
        }
        # Simulate round with only 2 updates
        server.round_updates = {
            1: {
                "client_1": {
                    "weights": {k: v.clone() for k, v in server.global_model.state_dict().items()},
                    "num_samples": 100,
                    "type": "model_update",
                    "round": 1,
                },
                "client_3": {
                    "weights": {k: v.clone() for k, v in server.global_model.state_dict().items()},
                    "num_samples": 100,
                    "type": "model_update",
                    "round": 1,
                },
            }
        }

        # With min_clients=1, this should proceed normally
        update_list = list(server.round_updates[1].values())
        assert len(update_list) == 2
        assert len(update_list) >= server.min_clients

        agg = server._aggregate_weighted(update_list)
        assert agg is not None
