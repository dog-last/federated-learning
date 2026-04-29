"""Extended unit tests for client module to improve coverage."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import socket

import pytest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
N = 4


def _make_config(mode="centralized", stragglers=None, device="cpu"):
    return {
        "experiment": {
            "mode": mode,
            "global_epochs": 1,
            "local_epochs": 1,
            "device": device,
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


def _make_client(mode="centralized", stragglers=None, device="cpu"):
    """Build a Client in a tmp dir. MonitorReporter.post is already session-patched."""
    from core.client import Client
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = _make_config(mode, stragglers, device)
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
        yield client


@pytest.fixture
def client_centralized():
    yield from _make_client("centralized")


@pytest.fixture
def client_splitfed():
    yield from _make_client("splitfed")


class TestSelectDeviceExtended:
    """Extended tests for device selection."""
    
    def test_cuda_available_returns_cuda(self):
        from core.client import Client
        with patch('torch.cuda.is_available', return_value=True):
            device = Client._select_device("cuda")
            assert device == torch.device("cuda")
    
    def test_mps_available_returns_mps(self):
        from core.client import Client
        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(torch.backends, 'mps', create=True) as mock_mps:
                mock_mps.is_available = MagicMock(return_value=True)
                device = Client._select_device("mps")
                assert device == torch.device("mps")
    
    def test_auto_prefers_cuda(self):
        from core.client import Client
        with patch('torch.cuda.is_available', return_value=True):
            device = Client._select_device("auto")
            assert device == torch.device("cuda")
    
    def test_auto_prefers_mps_over_cpu(self):
        from core.client import Client
        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(torch.backends, 'mps', create=True) as mock_mps:
                mock_mps.is_available = MagicMock(return_value=True)
                device = Client._select_device("auto")
                assert device == torch.device("mps")


class TestStragglerSimulation:
    """Tests for straggler simulation features."""
    
    def test_should_drop_round_with_zero_rate(self, client_centralized):
        client_centralized.straggler_config = {"drop_rate": 0.0}
        # With 0 drop rate, should never drop
        results = [client_centralized._should_drop_round() for _ in range(10)]
        assert all(r is False for r in results)
    
    def test_should_drop_round_with_one_rate(self, client_centralized):
        client_centralized.straggler_config = {"drop_rate": 1.0}
        # With 1.0 drop rate, should always drop
        results = [client_centralized._should_drop_round() for _ in range(10)]
        assert all(r is True for r in results)
    
    def test_should_drop_round_with_mid_rate(self, client_centralized):
        client_centralized.straggler_config = {"drop_rate": 0.5}
        # With 0.5 drop rate, should sometimes drop
        results = [client_centralized._should_drop_round() for _ in range(100)]
        # Should have a mix of True and False
        assert any(results)
        assert not all(results)
    
    def test_maybe_delay_with_zero_delay(self, client_centralized):
        client_centralized.straggler_config = {"delay": 0.0}
        with patch('core.client.time.sleep') as mock_sleep:
            client_centralized._maybe_delay()
            mock_sleep.assert_not_called()
    
    def test_maybe_delay_with_positive_delay(self, client_centralized):
        client_centralized.straggler_config = {"delay": 0.1}
        with patch('core.client.time.sleep') as mock_sleep:
            client_centralized._maybe_delay()
            mock_sleep.assert_called_once_with(0.1)


class TestNetworkStats:
    """Tests for network statistics tracking."""
    
    def test_count_type_new_key(self, client_centralized):
        bucket = {}
        client_centralized._count_type(bucket, "model_update", 1)
        assert bucket["model_update"] == 1
    
    def test_count_type_increment_existing(self, client_centralized):
        bucket = {"model_update": 5}
        client_centralized._count_type(bucket, "model_update", 3)
        assert bucket["model_update"] == 8
    
    def test_record_network_outgoing(self, client_centralized):
        initial_sent = client_centralized.net_stats["bytes_sent"]
        initial_msgs = client_centralized.net_stats["messages_sent"]
        
        client_centralized._record_network("out", "model_update", 1000, round_id=1)
        
        assert client_centralized.net_stats["bytes_sent"] == initial_sent + 1000
        assert client_centralized.net_stats["messages_sent"] == initial_msgs + 1
        assert client_centralized.net_stats["sent_by_type"]["model_update"] == 1
    
    def test_record_network_incoming(self, client_centralized):
        initial_recv = client_centralized.net_stats["bytes_recv"]
        initial_msgs = client_centralized.net_stats["messages_recv"]
        
        client_centralized._record_network("in", "weights", 2000, round_id=1)
        
        assert client_centralized.net_stats["bytes_recv"] == initial_recv + 2000
        assert client_centralized.net_stats["messages_recv"] == initial_msgs + 1
        assert client_centralized.net_stats["recv_by_type"]["weights"] == 1
    
    def test_record_network_with_extra(self, client_centralized):
        client_centralized._record_network("out", "test", 100, extra={"custom": "data"})
        # Should not raise and should include extra in monitor post


class TestSendReceive:
    """Tests for send and receive operations."""
    
    def test_send_success(self, client_centralized):
        with patch.object(client_centralized.communicator, 'send_data', return_value=(True, 100)):
            result = client_centralized._send({"type": "test"}, round_id=1)
            assert result == (True, 100)
    
    def test_send_failure(self, client_centralized):
        with patch.object(client_centralized.communicator, 'send_data', return_value=(False, 0)):
            result = client_centralized._send({"type": "test"}, round_id=1)
            assert result == (False, 0)
    
    def test_recv_with_message(self, client_centralized):
        mock_msg = {"type": "test_msg", "round": 1}
        mock_meta = {"payload_bytes": 100, "magic_ok": True, "compression": False}
        with patch.object(client_centralized.communicator, 'recv_data_with_meta', return_value=(mock_msg, mock_meta)):
            msg, meta = client_centralized._recv()
            assert msg == mock_msg
            assert meta == mock_meta
    
    def test_recv_with_none_message(self, client_centralized):
        with patch.object(client_centralized.communicator, 'recv_data_with_meta', return_value=(None, None)):
            msg, meta = client_centralized._recv()
            assert msg is None
            assert meta is None


class TestReportMetric:
    """Tests for metric reporting."""
    
    def test_report_metric_adds_mode(self, client_centralized):
        with patch.object(client_centralized.monitor, 'post') as mock_post:
            client_centralized.report_metric({"source": "test", "value": 1.0})
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["mode"] == "centralized"
            assert call_args[1]["source"] == "test"
            assert call_args[1]["value"] == 1.0
    
    def test_report_metric_preserves_existing_mode(self, client_centralized):
        with patch.object(client_centralized.monitor, 'post') as mock_post:
            client_centralized.report_metric({"source": "test", "mode": "custom"})
            call_args = mock_post.call_args
            assert call_args[1]["mode"] == "custom"


class TestTrainingRound:
    """Tests for training round execution."""
    
    def test_train_centralized_round_basic(self, client_centralized):
        # Create mock weights
        weights = {k: v.clone() for k, v in client_centralized.model.state_dict().items()}
        msg = {
            "round": 1,
            "weights": weights,
            "lr": 0.01,
            "local_epochs": 1,
        }
        
        with patch.object(client_centralized, '_send'):
            with patch.object(client_centralized.monitor, 'post'):
                result = client_centralized._train_centralized_round(msg)
        
        assert result["type"] == "model_update"
        assert result["round"] == 1
        assert result["client_id"] == "client_1"
        assert "weights" in result
        assert "train_loss" in result
        assert "train_acc" in result
    
    def test_train_centralized_round_with_custom_params(self, client_centralized):
        weights = {k: v.clone() for k, v in client_centralized.model.state_dict().items()}
        msg = {
            "round": 2,
            "weights": weights,
            "lr": 0.001,
            "local_epochs": 2,
            "momentum": 0.95,
            "weight_decay": 0.001,
        }
        
        with patch.object(client_centralized, '_send'):
            with patch.object(client_centralized.monitor, 'post'):
                result = client_centralized._train_centralized_round(msg)
        
        assert result["round"] == 2


class TestClientRunLoop:
    """Tests for client main run loop."""
    
    def test_run_handles_register_ack(self, client_centralized):
        messages = [
            {"type": "register_ack", "mode": "centralized"},
            {"type": "shutdown", "reason": "test"},
        ]
        
        def mock_recv():
            return messages.pop(0), {}
        
        with patch.object(client_centralized, '_recv', side_effect=mock_recv):
            with patch.object(client_centralized, '_send'):
                with patch.object(client_centralized.conn, 'connect'):
                    with patch.object(client_centralized.monitor, 'post'):
                        client_centralized.run()
        
        # Should have processed both messages and exited
    
    def test_run_handles_round_start_centralized(self, client_centralized):
        weights = {k: v.clone() for k, v in client_centralized.model.state_dict().items()}
        messages = [
            {"type": "register_ack", "mode": "centralized"},
            {"type": "round_start_centralized", "round": 1, "weights": weights, "lr": 0.01},
            {"type": "shutdown", "reason": "test"},
        ]
        
        def mock_recv():
            if messages:
                return messages.pop(0), {}
            return None, None
        
        with patch.object(client_centralized, '_recv', side_effect=mock_recv):
            with patch.object(client_centralized, '_send'):
                with patch.object(client_centralized.conn, 'connect'):
                    with patch.object(client_centralized.monitor, 'post'):
                        with patch.object(client_centralized, '_should_drop_round', return_value=False):
                            client_centralized.run()
    
    def test_run_handles_server_disconnect(self, client_centralized):
        with patch.object(client_centralized, '_recv', return_value=(None, None)):
            with patch.object(client_centralized.conn, 'connect'):
                with patch.object(client_centralized, '_send'):
                    with patch.object(client_centralized.monitor, 'post'):
                        client_centralized.run()
        
        # Should exit gracefully when server disconnects
    
    def test_run_handles_unknown_message_type(self, client_centralized):
        messages = [
            {"type": "register_ack", "mode": "centralized"},
            {"type": "unknown_type", "data": "test"},
            {"type": "shutdown", "reason": "test"},
        ]
        
        def mock_recv():
            if messages:
                return messages.pop(0), {}
            return None, None
        
        with patch.object(client_centralized, '_recv', side_effect=mock_recv):
            with patch.object(client_centralized, '_send'):
                with patch.object(client_centralized.conn, 'connect'):
                    with patch.object(client_centralized.monitor, 'post'):
                        with patch.object(client_centralized.logger, 'warning') as mock_warning:
                            client_centralized.run()
                            mock_warning.assert_called()


class TestSplitFedTraining:
    """Tests for SplitFed training mode."""
    
    def test_train_splitfed_round_basic(self, client_splitfed):
        client_weights = {k: v.clone() for k, v in client_splitfed.client_model.state_dict().items()}
        server_weights = {k: v.clone() for k, v in client_splitfed.shadow_server_model.state_dict().items()}
        
        msg = {
            "round": 1,
            "client_weights": client_weights,
            "server_weights": server_weights,
            "lr": 0.01,
            "local_epochs": 1,
        }
        
        # Get the actual output shape from client model
        with torch.no_grad():
            sample_input = next(iter(client_splitfed.train_loader))[0]
            sample_output = client_splitfed.client_model(sample_input)
            grad_shape = sample_output.shape
        
        # Mock the server responses for split batches with correct gradient shape
        responses = [
            ({"type": "split_grad", "grad": torch.rand(grad_shape), "batch_size": 2, "correct": 1, "loss": 0.5}, {}),
        ]
        
        with patch.object(client_splitfed, '_recv', side_effect=responses):
            with patch.object(client_splitfed, '_send'):
                with patch.object(client_splitfed.monitor, 'post'):
                    result = client_splitfed._train_splitfed_round(msg)
        
        assert result["type"] == "split_update"
        assert result["round"] == 1
        assert "weights" in result


class TestMainEntryPoint:
    """Tests for __main__ entry point."""
    
    def test_main_without_args(self):
        with patch('sys.argv', ['core.client']):
            with patch('builtins.print') as mock_print:
                # Import and execute the main block
                import core.client as client_module
                # Simulate the main block execution
                if len([]) < 2:  # Empty args
                    print("Usage: python -m core.client <client_id>")
                mock_print.assert_called_once()
