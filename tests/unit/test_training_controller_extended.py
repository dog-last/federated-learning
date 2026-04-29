"""Extended unit tests for training controller to improve coverage."""
import os
import signal
import tempfile
from unittest.mock import MagicMock, patch, mock_open
import subprocess

import pytest
import torch

from utils.training_controller import TrainingController


@pytest.fixture
def ctrl():
    """Create a fresh TrainingController for each test with mocked time.sleep."""
    with patch('utils.training_controller.time.sleep'):
        yield TrainingController(
            project_root=tempfile.gettempdir(),
            python_bin="python",
            event_hook=None,
        )


class TestTrainingControllerStart:
    """Tests for starting training."""

    def test_start_creates_procs(self, ctrl):
        """Test that start creates server and client processes."""
        config = {
            "topology": {
                "server": {"host": "127.0.0.1", "port": 8080},
                "clients": [
                    {"id": "client_1"},
                    {"id": "client_2"},
                ]
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w") as f:
                import json
                json.dump(config, f)

            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None

            with patch('subprocess.Popen', return_value=mock_proc):
                with patch('threading.Thread'):
                    with patch.object(ctrl, '_ensure_data'):
                        with patch.object(ctrl, '_emit'):
                            result = ctrl.start(config_path)

            assert result["ok"] is True
            assert ctrl._state == "running"
            assert ctrl._server_proc is not None
            assert len(ctrl._client_procs) == 2

    def test_start_when_already_running(self, ctrl):
        """Test starting when already running returns error."""
        ctrl._state = "running"

        result = ctrl.start("/fake/config.json")
        assert result["ok"] is False
        assert "already" in result["error"].lower()

    def test_start_with_invalid_config(self, ctrl):
        """Test starting with invalid config file."""
        result = ctrl.start("/nonexistent/config.json")
        assert result["ok"] is False
        assert "config" in result["error"].lower()


class TestTrainingControllerStopRunning:
    """Tests for stopping running training."""

    def test_stop_running_processes(self, ctrl):
        """Test stopping running server and client processes."""
        ctrl._state = "running"

        # Create mock processes
        mock_server = MagicMock()
        mock_server.pid = 12345
        mock_server.poll.return_value = None
        mock_server.terminate = MagicMock()
        mock_server.wait = MagicMock(return_value=0)

        mock_client = MagicMock()
        mock_client.pid = 12346
        mock_client.poll.return_value = None
        mock_client.terminate = MagicMock()
        mock_client.wait = MagicMock(return_value=0)

        ctrl._server_proc = mock_server
        ctrl._client_procs = [mock_client]

        with patch.object(ctrl, '_close_role_logs'):
            with patch.object(ctrl, '_emit'):
                result = ctrl.stop()

        assert result["ok"] is True
        assert ctrl._state == "stopped"
        mock_server.send_signal.assert_called_once_with(signal.SIGTERM)
        mock_client.send_signal.assert_called_once_with(signal.SIGTERM)

    def test_stop_with_dead_processes(self, ctrl):
        """Test stopping when processes are already dead."""
        ctrl._state = "running"

        mock_server = MagicMock()
        mock_server.poll.return_value = 0  # Already exited

        ctrl._server_proc = mock_server
        ctrl._client_procs = []

        with patch.object(ctrl, '_close_role_logs'):
            with patch.object(ctrl, '_emit'):
                result = ctrl.stop()

        assert result["ok"] is True


class TestTrainingControllerEnsureData:
    """Tests for data preparation."""

    def test_ensure_data_creates_splits(self, ctrl):
        """Test that ensure_data creates data splits if needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ctrl.project_root = tmp_dir
            config = {
                "topology": {
                    "clients": [{"id": "client_1"}]
                }
            }

            with patch('utils.training_controller.subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                ctrl._ensure_data(config)

            # Should have called subprocess to prepare data
            mock_run.assert_called_once()

    def test_ensure_data_with_existing_splits(self, ctrl):
        """Test ensure_data when splits already exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ctrl.project_root = tmp_dir
            splits_dir = os.path.join(tmp_dir, "data", "splits")
            os.makedirs(splits_dir, exist_ok=True)

            # Create existing data files
            torch.save({"images": torch.rand(10, 1, 28, 28)},
                      os.path.join(splits_dir, "server_test_data.pt"))
            torch.save({"train_images": torch.rand(10, 1, 28, 28)},
                      os.path.join(splits_dir, "client_1_data.pt"))

            config = {
                "topology": {
                    "clients": [{"id": "client_1"}]
                }
            }

            # Should not raise and should not call subprocess
            with patch('utils.training_controller.subprocess.run') as mock_run:
                ctrl._ensure_data(config)
                mock_run.assert_not_called()


class TestTrainingControllerEmit:
    """Tests for event emission."""

    def test_emit_with_event_hook(self):
        """Test that events are passed to the hook."""
        events = []

        def hook(event):
            events.append(event)

        with patch('utils.training_controller.time.sleep'):
            ctrl = TrainingController("/tmp", "python", event_hook=hook)
            ctrl._emit({"type": "test_event"})

        assert len(events) == 1
        assert events[0]["type"] == "test_event"

    def test_emit_without_hook(self, ctrl):
        """Test that emit works without a hook."""
        # Should not raise
        ctrl._emit({"type": "test_event"})


class TestTrainingControllerLogs:
    """Tests for log handling."""

    def test_open_role_logs(self, ctrl):
        """Test opening log files for server and clients."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ctrl.project_root = tmp_dir
            logs_dir = os.path.join(tmp_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)

            ctrl._open_role_log("server")
            assert len(ctrl._proc_log_handles) == 1

            ctrl._open_role_log("client_1")
            assert len(ctrl._proc_log_handles) == 2

            # Clean up
            for handle in ctrl._proc_log_handles:
                handle.close()

    def test_close_role_logs(self, ctrl):
        """Test closing log file handles."""
        mock_handle = MagicMock()
        ctrl._proc_log_handles = [mock_handle]

        ctrl._close_role_logs()

        mock_handle.close.assert_called_once()
        assert ctrl._proc_log_handles == []


class TestTrainingControllerProcInfoExtended:
    """Extended tests for process info."""

    def test_proc_info_with_none(self, ctrl):
        """Test proc_info with None process."""
        info = ctrl._proc_info(None, "server")
        assert info["role"] == "server"
        assert info["pid"] is None
        assert info["alive"] is False
        assert info["exit_code"] is None

    def test_proc_info_with_running_process(self, ctrl):
        """Test proc_info with running process."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        info = ctrl._proc_info(mock_proc, "client")
        assert info["pid"] == 12345
        assert info["alive"] is True
        assert info["exit_code"] is None

    def test_proc_info_with_exited_process(self, ctrl):
        """Test proc_info with exited process."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0

        info = ctrl._proc_info(mock_proc, "server")
        assert info["pid"] == 12345
        assert info["alive"] is False
        assert info["exit_code"] == 0


class TestTrainingControllerStatusExtended:
    """Extended tests for status."""

    def test_status_with_uptime(self, ctrl):
        """Test status calculation with uptime."""
        import time
        ctrl._state = "running"
        ctrl._started_at = time.time() - 100  # Started 100 seconds ago

        status = ctrl.status()
        assert status["state"] == "running"
        assert status["uptime_seconds"] >= 100

    def test_status_with_error(self, ctrl):
        """Test status includes last error."""
        ctrl._last_error = "Test error message"

        status = ctrl.status()
        assert status["last_error"] == "Test error message"


class TestTrainingControllerCompatibility:
    """Tests for split compatibility checking."""

    def test_is_mnist_split_compatible_valid(self, ctrl):
        """Test compatibility check with valid MNIST data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            client_path = os.path.join(tmp_dir, "client.pt")
            server_path = os.path.join(tmp_dir, "server.pt")

            # Create valid MNIST-shaped data
            torch.save({
                "train_images": torch.rand(100, 1, 28, 28),
                "train_labels": torch.randint(0, 10, (100,))
            }, client_path)

            torch.save({
                "images": torch.rand(100, 1, 28, 28),
                "labels": torch.randint(0, 10, (100,))
            }, server_path)

            assert ctrl._is_mnist_split_compatible([client_path, server_path]) is True

    def test_is_mnist_split_compatible_wrong_shape(self, ctrl):
        """Test compatibility check with wrong image shape."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            client_path = os.path.join(tmp_dir, "client.pt")

            # Create data with wrong shape (not 28x28)
            torch.save({
                "train_images": torch.rand(100, 1, 32, 32),  # Wrong size
            }, client_path)

            assert ctrl._is_mnist_split_compatible([client_path]) is False

    def test_is_mnist_split_compatible_missing_files(self, ctrl):
        """Test compatibility check with missing files."""
        assert ctrl._is_mnist_split_compatible(["/nonexistent/file.pt"]) is False


class TestTrainingControllerStartProcesses:
    """Tests for starting server and client processes."""

    def test_start_assigns_server_proc(self, ctrl):
        """Test that start assigns the server process."""
        config = {
            "topology": {
                "server": {"host": "127.0.0.1", "port": 8080},
                "clients": [{"id": "client_1"}]
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w") as f:
                import json
                json.dump(config, f)

            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None

            with patch('subprocess.Popen', return_value=mock_proc):
                with patch('threading.Thread'):
                    with patch.object(ctrl, '_ensure_data'):
                        with patch.object(ctrl, '_emit'):
                            ctrl.start(config_path)

            assert ctrl._server_proc is not None
            assert ctrl._server_proc.pid == 12345

    def test_start_assigns_client_procs(self, ctrl):
        """Test that start assigns client processes."""
        config = {
            "topology": {
                "server": {"host": "127.0.0.1", "port": 8080},
                "clients": [{"id": "c1"}, {"id": "c2"}]
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w") as f:
                import json
                json.dump(config, f)

            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None

            with patch('subprocess.Popen', return_value=mock_proc):
                with patch('threading.Thread'):
                    with patch.object(ctrl, '_ensure_data'):
                        with patch.object(ctrl, '_emit'):
                            ctrl.start(config_path)

            assert len(ctrl._client_procs) == 2
