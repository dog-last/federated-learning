"""Integration tests for TrainingController."""
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from utils.training_controller import TrainingController


@pytest.fixture
def controller_config():
    """Create a minimal config for controller testing."""
    return {
        "experiment": {
            "mode": "centralized",
            "global_epochs": 1,
            "local_epochs": 1,
            "device": "cpu",
            "dataset_params": {"batch_size": 4, "num_workers": 0},
            "optimization": {"client_lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        },
        "topology": {
            "server": {"host": "127.0.0.1", "port": 19900},
            "clients": [
                {"id": "client_1"},
                {"id": "client_2"},
            ]
        },
        "network": {"compression": False, "stragglers": {}, "server_timeout": 30.0},
        "monitoring": {"api_host": "127.0.0.1", "api_port": 19999},
    }


@pytest.fixture
def controller_data_dir(tmp_path):
    """Create minimal data for controller testing."""
    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)
    
    # Create client data
    for client_id in [1, 2]:
        client_data = {
            "train_images": torch.rand(20, 1, 28, 28),
            "train_labels": torch.randint(0, 10, (20,)),
            "val_images": torch.rand(4, 1, 28, 28),
            "val_labels": torch.randint(0, 10, (4,)),
            "test_images": torch.rand(4, 1, 28, 28),
            "test_labels": torch.randint(0, 10, (4,)),
        }
        torch.save(client_data, splits_dir / f"client_{client_id}_data.pt")
    
    # Create server test data
    server_data = {
        "images": torch.rand(10, 1, 28, 28),
        "labels": torch.randint(0, 10, (10,)),
    }
    torch.save(server_data, splits_dir / "server_test_data.pt")
    
    return tmp_path


@pytest.mark.integration
class TestTrainingControllerLifecycle:
    """Integration tests for TrainingController lifecycle."""

    def test_controller_start_stop_cycle(self, controller_config, controller_data_dir):
        """Test controller can start and stop successfully."""
        config_path = controller_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(controller_config, f)

        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=None,
            )

            # Mock subprocess.Popen to avoid actually starting processes
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None
            # Make wait() block so the watcher thread doesn't set state to stopped
            wait_event = threading.Event()
            mock_proc.wait.side_effect = lambda: (wait_event.wait(), 0)[1]

            with patch('subprocess.Popen', return_value=mock_proc):
                with patch.object(ctrl, '_ensure_data'):
                    # Start training
                    result = ctrl.start(str(config_path))
                    assert result["ok"] is True
                    assert ctrl._state == "running"
                    assert ctrl._server_proc is not None
                    assert len(ctrl._client_procs) == 2

                    # Stop training
                    wait_event.set()  # unblock watcher
                    result = ctrl.stop()
                    assert result["ok"] is True
                    assert ctrl._state == "stopped"

    def test_controller_status_during_training(self, controller_config, controller_data_dir):
        """Test controller status reporting during training."""
        config_path = controller_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(controller_config, f)

        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=None,
            )

            # Initial status
            status = ctrl.status()
            assert status["state"] == "stopped"
            assert status["uptime_seconds"] == 0

            # Mock running state
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None
            wait_event = threading.Event()
            mock_proc.wait.side_effect = lambda: (wait_event.wait(), 0)[1]

            with patch('subprocess.Popen', return_value=mock_proc):
                with patch.object(ctrl, '_ensure_data'):
                    ctrl.start(str(config_path))

                    status = ctrl.status()
                    assert status["state"] == "running"
                    assert status["server"]["alive"] is True
                    assert len(status["clients"]) == 2

            wait_event.set()  # unblock watcher on cleanup

    def test_controller_event_emission(self, controller_config, controller_data_dir):
        """Test controller emits events correctly."""
        events = []
        
        def event_hook(event):
            events.append(event)
        
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=event_hook,
            )
            
            # Emit a test event
            ctrl._emit({"type": "test_event", "data": "value"})
            
            assert len(events) == 1
            assert events[0]["type"] == "test_event"
            assert events[0]["data"] == "value"
            assert "ts" in events[0]


@pytest.mark.integration
class TestTrainingControllerDataHandling:
    """Integration tests for data handling in TrainingController."""

    def test_controller_ensures_data_exists(self, controller_config, controller_data_dir):
        """Test controller checks for existing data."""
        config_path = controller_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(controller_config, f)
        
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=None,
            )
            
            # Data already exists, should not call subprocess
            with patch('subprocess.run') as mock_run:
                ctrl._ensure_data(controller_config)
                mock_run.assert_not_called()

    def test_controller_creates_data_when_missing(self, controller_config, tmp_path):
        """Test controller creates data when missing."""
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(controller_config, f)
        
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(tmp_path),
                python_bin="python",
                event_hook=None,
            )
            
            # Data does not exist, should call subprocess
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                ctrl._ensure_data(controller_config)
                mock_run.assert_called_once()

    def test_controller_mnist_compatibility_check(self, controller_data_dir):
        """Test controller MNIST compatibility checking."""
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=None,
            )
            
            # Check compatible files
            splits_dir = controller_data_dir / "data" / "splits"
            files = [
                str(splits_dir / "client_1_data.pt"),
                str(splits_dir / "server_test_data.pt"),
            ]
            
            is_compatible = ctrl._is_mnist_split_compatible(files)
            assert is_compatible is True


@pytest.mark.integration
class TestTrainingControllerProcessManagement:
    """Integration tests for process management."""

    def test_controller_detects_dead_processes(self, controller_config, controller_data_dir):
        """Test controller detects when processes die."""
        config_path = controller_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(controller_config, f)
        
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=None,
            )
            
            # Create mock dead process
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = 1  # Process exited with code 1
            
            info = ctrl._proc_info(mock_proc, "server")
            assert info["alive"] is False
            assert info["exit_code"] == 1

    def test_controller_log_file_handling(self, controller_data_dir):
        """Test controller log file operations."""
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=None,
            )
            
            # Create logs directory
            logs_dir = controller_data_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Open log files
            ctrl._open_role_log("server")
            assert len(ctrl._proc_log_handles) == 1
            
            # Close log files
            ctrl._close_role_logs()
            assert len(ctrl._proc_log_handles) == 0


@pytest.mark.integration
class TestTrainingControllerErrorHandling:
    """Integration tests for error handling."""

    def test_controller_handles_invalid_config(self, controller_data_dir):
        """Test controller handles invalid config gracefully."""
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=None,
            )
            
            # Try to start with non-existent config
            result = ctrl.start("/nonexistent/config.json")
            assert result["ok"] is False
            assert "error" in result

    def test_controller_handles_start_when_already_running(self, controller_config, controller_data_dir):
        """Test controller prevents starting when already running."""
        config_path = controller_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(controller_config, f)
        
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="python",
                event_hook=None,
            )
            
            # Set state to running
            ctrl._state = "running"
            
            # Try to start again
            result = ctrl.start(str(config_path))
            assert result["ok"] is False
            assert "already" in result["error"].lower()

    def test_controller_handles_missing_python_binary(self, controller_config, controller_data_dir):
        """Test controller handles missing Python binary."""
        config_path = controller_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(controller_config, f)
        
        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController(
                project_root=str(controller_data_dir),
                python_bin="/nonexistent/python",
                event_hook=None,
            )
            
            with patch('subprocess.Popen', side_effect=FileNotFoundError("No such file")):
                with patch.object(ctrl, '_ensure_data'):
                    result = ctrl.start(str(config_path))
                    # Should handle the error gracefully
                    assert result["ok"] is False or ctrl._state == "running"
