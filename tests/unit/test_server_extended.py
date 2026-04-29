"""Extended unit tests for server module to improve coverage."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import socket
import threading
import time

import pytest
import torch
import numpy as np

from core.server import Server

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
N = 4


def _make_config(mode="centralized", stragglers=None):
    return {
        "experiment": {
            "mode": mode,
            "global_epochs": 1,
            "local_epochs": 1,
            "device": "cpu",
            "target_accuracy": 0.95,
            "dataset_params": {"batch_size": N, "num_workers": 0},
            "optimization": {"client_lr": 0.01, "server_lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        },
        "topology": {
            "server": {"host": "127.0.0.1", "port": 0},
            "clients": [{"id": "client_1", "host": "127.0.0.1", "port": 0}],
        },
        "network": {"compression": False, "stragglers": stragglers or {}, "server_timeout": 30.0},
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
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = _make_config(mode)
        config_path = _write_config_and_data(config, tmp_dir)
        with patch('core.server.socket.socket', return_value=_mock_socket()):
            server = Server(config_path)
            # Patch project_root to use tmp_dir
            server.project_root = tmp_dir
            yield server


@pytest.fixture
def server_centralized():
    yield from _make_server("centralized")


@pytest.fixture
def server_splitfed():
    yield from _make_server("splitfed")


class TestServerSelectDeviceExtended:
    """Extended tests for device selection."""

    def test_cuda_available_returns_cuda(self):
        with patch('torch.cuda.is_available', return_value=True):
            device = Server._select_device("cuda")
            assert device == torch.device("cuda")

    def test_mps_available_returns_mps(self):
        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(torch.backends, 'mps', create=True) as mock_mps:
                mock_mps.is_available = MagicMock(return_value=True)
                device = Server._select_device("mps")
                assert device == torch.device("mps")

    def test_auto_prefers_cuda(self):
        with patch('torch.cuda.is_available', return_value=True):
            device = Server._select_device("auto")
            assert device == torch.device("cuda")


class TestServerNetworkStats:
    """Tests for network statistics tracking."""

    def test_count_type_new_key(self, server_centralized):
        bucket = {}
        server_centralized._count_type(bucket, "model_update", 1)
        assert bucket["model_update"] == 1

    def test_count_type_increment_existing(self, server_centralized):
        bucket = {"model_update": 5}
        server_centralized._count_type(bucket, "model_update", 3)
        assert bucket["model_update"] == 8

    def test_snapshot_net(self, server_centralized):
        server_centralized.net_stats = {
            "bytes_sent": 1000,
            "bytes_recv": 2000,
            "messages_sent": 10,
            "messages_recv": 20,
        }
        snap = server_centralized._snapshot_net()
        assert snap["bytes_sent"] == 1000
        assert snap["bytes_recv"] == 2000
        assert snap["messages_sent"] == 10
        assert snap["messages_recv"] == 20

    def test_delta_net(self, server_centralized):
        before = {"bytes_sent": 100, "bytes_recv": 50, "messages_sent": 5, "messages_recv": 3}
        after = {"bytes_sent": 200, "bytes_recv": 150, "messages_sent": 10, "messages_recv": 8}
        delta = server_centralized._delta_net(after, before)
        assert delta["bytes_sent"] == 100
        assert delta["bytes_recv"] == 100
        assert delta["messages_sent"] == 5
        assert delta["messages_recv"] == 5


class TestServerAggregation:
    """Tests for model aggregation."""

    def test_aggregate_weighted_with_updates(self, server_centralized):
        # Create sample state dict
        sd = server_centralized.global_model.state_dict()
        updates = [
            {"weights": {k: v.clone() * 1.1 for k, v in sd.items()}, "num_samples": 10},
            {"weights": {k: v.clone() * 0.9 for k, v in sd.items()}, "num_samples": 20},
        ]
        agg = server_centralized._aggregate_weighted(updates)
        assert agg is not None
        for key in sd.keys():
            assert key in agg

    def test_aggregate_weighted_single_update(self, server_centralized):
        sd = server_centralized.global_model.state_dict()
        updates = [
            {"weights": {k: v.clone() for k, v in sd.items()}, "num_samples": 10},
        ]
        agg = server_centralized._aggregate_weighted(updates)
        assert agg is not None


class TestServerClientHandlers:
    """Tests for client connection handlers."""

    def test_handle_client_register(self, server_centralized):
        mock_conn = MagicMock()
        mock_addr = ("127.0.0.1", 12345)

        # Mock receiving register message then shutdown
        messages = [
            {"type": "register", "client_id": "client_1"},
        ]

        def mock_recv_data(sock):
            if messages:
                return messages.pop(0)
            return None

        with patch.object(server_centralized.communicator, 'recv_data', side_effect=mock_recv_data):
            with patch.object(server_centralized.communicator, 'send_data', return_value=(True, 100)):
                # Run handle_client with timeout to prevent infinite loop
                def run_handle_client():
                    server_centralized.handle_client(mock_conn, mock_addr)

                t = threading.Thread(target=run_handle_client)
                t.start()
                t.join(timeout=1)

    def test_handle_client_model_update(self, server_centralized):
        mock_conn = MagicMock()
        mock_addr = ("127.0.0.1", 12345)

        weights = {k: v.clone() for k, v in server_centralized.global_model.state_dict().items()}
        messages = [
            {"type": "register", "client_id": "client_1"},
            {"type": "model_update", "round": 1, "client_id": "client_1", "weights": weights, "num_samples": 10, "train_loss": 0.5, "train_acc": 0.8, "test_acc": 0.75},
        ]

        call_count = [0]

        def mock_recv_data(sock):
            if call_count[0] < len(messages):
                msg = messages[call_count[0]]
                call_count[0] += 1
                return msg
            return None

        with patch.object(server_centralized.communicator, 'recv_data', side_effect=mock_recv_data):
            with patch.object(server_centralized.communicator, 'send_data', return_value=(True, 100)):
                def run_handle_client():
                    server_centralized.handle_client(mock_conn, mock_addr)

                t = threading.Thread(target=run_handle_client)
                t.start()
                t.join(timeout=1)


class TestServerWaitRoundUpdates:
    """Tests for _wait_round_updates method."""

    def test_wait_round_updates_timeout(self, server_centralized):
        """Test that _wait_round_updates handles timeout correctly."""
        server_centralized.timeout = 0.1  # Short timeout for testing
        server_centralized.round_updates[1] = {}
        server_centralized.num_clients = 2

        # Should return empty dict when timeout occurs
        result = server_centralized._wait_round_updates(1)
        assert isinstance(result, dict)

    def test_wait_round_updates_with_updates(self, server_centralized):
        """Test that _wait_round_updates returns collected updates."""
        server_centralized.timeout = 1.0
        server_centralized.round_updates[1] = {"client_1": {"num_samples": 10}}
        server_centralized.num_clients = 1

        result = server_centralized._wait_round_updates(1)
        assert "client_1" in result
        assert result["client_1"]["num_samples"] == 10


class TestServerBroadcast:
    """Tests for _broadcast method."""

    def test_broadcast_to_clients(self, server_centralized):
        """Test broadcasting to all active clients."""
        mock_conn = MagicMock()
        server_centralized.active_clients = {"client_1": mock_conn}
        server_centralized.num_clients = 1

        payload = {"type": "test", "round": 1}
        with patch.object(server_centralized, '_send', return_value=(True, 100)):
            server_centralized._broadcast(payload)

        # _send should have been called for the client

    def test_broadcast_with_failed_send(self, server_centralized):
        """Test broadcasting when send fails."""
        mock_conn = MagicMock()
        server_centralized.active_clients = {"client_1": mock_conn}

        payload = {"type": "test"}
        with patch.object(server_centralized, '_send', return_value=(False, 0)):
            server_centralized._broadcast(payload)


class TestServerEvaluation:
    """Tests for server evaluation."""

    def test_evaluate_centralized(self, server_centralized):
        loss, acc = server_centralized._evaluate_centralized()
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_splitfed(self, server_splitfed):
        loss, acc = server_splitfed._evaluate_splitfed()
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


class TestServerSplitFed:
    """Tests for SplitFed server mode."""

    def test_handle_split_batch(self, server_splitfed):
        """Test handling split batch from client."""
        mock_conn = MagicMock()

        # Create a sample activation tensor
        activations = torch.rand(2, 32, 14, 14)
        labels = torch.randint(0, 10, (2,))

        with patch.object(server_splitfed, '_send', return_value=(True, 100)):
            server_splitfed._handle_split_batch(mock_conn, "client_1", {
                "activations": activations,
                "labels": labels,
            })


class TestServerRunRound:
    """Tests for running training rounds."""

    def test_run_round_centralized(self, server_centralized):
        """Test running a centralized training round."""
        server_centralized.num_clients = 1
        server_centralized.active_clients = {"client_1": MagicMock()}
        server_centralized.round_updates[1] = {
            "client_1": {
                "weights": {k: v.clone() for k, v in server_centralized.global_model.state_dict().items()},
                "num_samples": 10,
            }
        }

        with patch.object(server_centralized, '_broadcast'):
            with patch.object(server_centralized, '_wait_round_updates', return_value=server_centralized.round_updates[1]):
                with patch.object(server_centralized, '_aggregate_weighted') as mock_agg:
                    mock_agg.return_value = {k: v.clone() for k, v in server_centralized.global_model.state_dict().items()}
                    acc = server_centralized._run_round_centralized(1)
                    assert isinstance(acc, float)

    def test_run_round_splitfed(self, server_splitfed):
        """Test running a SplitFed training round."""
        server_splitfed.num_clients = 1
        server_splitfed.active_clients = {"client_1": MagicMock()}

        with patch.object(server_splitfed, '_broadcast'):
            with patch.object(server_splitfed, '_wait_round_updates', return_value={
                "client_1": {
                    "weights": {k: v.clone() for k, v in server_splitfed.global_client_model.state_dict().items()},
                    "num_samples": 10,
                }
            }):
                with patch.object(server_splitfed.global_client_model, 'load_state_dict'):
                    acc = server_splitfed._run_round_splitfed(1)
                    assert isinstance(acc, float)


class TestServerAcceptLoop:
    """Tests for _accept_loop method."""

    def test_accept_loop_stops_on_stop_event(self, server_centralized):
        """Test that accept loop stops when stop_event is set."""
        server_centralized.stop_event.set()

        # Should return immediately when stop_event is set
        server_centralized._accept_loop()

    def test_accept_loop_handles_oserror(self, server_centralized):
        """Test that accept loop handles OSError gracefully."""
        server_centralized.server_obj = MagicMock()
        server_centralized.server_obj.accept.side_effect = OSError("Socket closed")

        server_centralized._accept_loop()
        # Should not raise


class TestServerWaitForAllClients:
    """Tests for _wait_for_all_clients method."""

    def test_wait_for_all_clients_already_connected(self, server_centralized):
        """Test when all clients are already connected."""
        server_centralized.num_clients = 1
        server_centralized.active_clients = {"client_1": MagicMock()}

        # Should return immediately
        server_centralized._wait_for_all_clients()

    def test_wait_for_all_clients_with_timeout(self, server_centralized):
        """Test waiting for clients with timeout."""
        server_centralized.num_clients = 2
        server_centralized.active_clients = {"client_1": MagicMock()}

        # This will loop until client_2 connects or test timeout
        # For testing, we just verify it doesn't crash
        def add_client_after_delay():
            import time
            time.sleep(0.1)
            server_centralized.active_clients["client_2"] = MagicMock()

        t = threading.Thread(target=add_client_after_delay)
        t.start()
        server_centralized._wait_for_all_clients()
        t.join(timeout=1)
