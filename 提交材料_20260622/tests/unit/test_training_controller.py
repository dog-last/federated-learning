"""Unit tests for training controller."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from utils.training_controller import TrainingController

N = 4


@pytest.fixture
def ctrl():
    """Create a fresh TrainingController for each test with mocked time.sleep."""
    with patch('utils.training_controller.time.sleep'):
        yield TrainingController(
            project_root=tempfile.gettempdir(),
            python_bin="python",
            event_hook=None,
        )


class TestInit:
    def test_state(self, ctrl):
        assert ctrl._state == "stopped"

    def test_no_procs(self, ctrl):
        assert ctrl._server_proc is None
        assert ctrl._client_procs == []


class TestLogsDir:
    def test_logs_dir_exists(self, ctrl):
        assert os.path.isdir(ctrl._logs_dir())


class TestProcInfo:
    def test_none_proc(self, ctrl):
        info = ctrl._proc_info(None, "server")
        assert info["role"] == "server"
        assert info["pid"] is None
        assert info["alive"] is False
        assert info["exit_code"] is None

    def test_alive_proc(self, ctrl):
        mock = MagicMock()
        mock.pid = 12345
        mock.poll.return_value = None
        info = ctrl._proc_info(mock, "client")
        assert info["pid"] == 12345
        assert info["alive"] is True

    def test_dead_proc(self, ctrl):
        mock = MagicMock()
        mock.pid = 12345
        mock.poll.return_value = 0
        info = ctrl._proc_info(mock, "client")
        assert info["alive"] is False
        assert info["exit_code"] == 0


class TestStatus:
    def test_stopped(self, ctrl):
        status = ctrl.status()
        assert status["state"] == "stopped"
        assert status["uptime_seconds"] == 0
        assert status["last_error"] is None

    def test_with_error(self, ctrl):
        ctrl._last_error = "test error"
        status = ctrl.status()
        assert status["last_error"] == "test error"
        ctrl._last_error = None


class TestStop:
    def test_stop_when_stopped(self, ctrl):
        result = ctrl.stop()
        assert result["ok"] is True
        assert result["status"]["state"] == "stopped"


class TestEnsureData:
    def test_ensure_data_exists(self, ctrl):
        with tempfile.TemporaryDirectory() as tmp_dir:
            splits_dir = os.path.join(tmp_dir, "data", "splits")
            os.makedirs(splits_dir, exist_ok=True)

            for cid in ["client_1", "client_2"]:
                torch.save(
                    {"train_images": torch.rand(N, 1, 28, 28), "train_labels": torch.randint(0, 10, (N,))},
                    os.path.join(splits_dir, f"{cid}_data.pt"),
                )
            torch.save(
                {"images": torch.rand(N, 1, 28, 28), "labels": torch.randint(0, 10, (N,))},
                os.path.join(splits_dir, "server_test_data.pt"),
            )

            old = ctrl.project_root
            ctrl.project_root = tmp_dir
            try:
                ctrl._ensure_data({"topology": {"clients": [{"id": "client_1"}, {"id": "client_2"}]}})
            finally:
                ctrl.project_root = old

    def test_is_mnist_split_compatible_valid(self, ctrl):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cp = os.path.join(tmp_dir, "client.pt")
            sp = os.path.join(tmp_dir, "server.pt")
            torch.save({"train_images": torch.rand(N, 1, 28, 28), "train_labels": torch.randint(0, 10, (N,))}, cp)
            torch.save({"images": torch.rand(N, 1, 28, 28), "labels": torch.randint(0, 10, (N,))}, sp)
            assert ctrl._is_mnist_split_compatible([cp, sp]) is True

    def test_is_mnist_split_compatible_invalid(self, ctrl):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cp = os.path.join(tmp_dir, "client.pt")
            sp = os.path.join(tmp_dir, "server.pt")
            torch.save({"train_images": torch.rand(N, 3, 32, 32)}, cp)
            torch.save({"images": torch.rand(N, 1, 28, 28)}, sp)
            assert ctrl._is_mnist_split_compatible([cp, sp]) is False

    def test_is_mnist_split_compatible_missing(self, ctrl):
        assert ctrl._is_mnist_split_compatible(["/no/file.pt", "/no/server.pt"]) is False


class TestEventHook:
    def test_emit_with_hook(self):
        events = []
        with patch('utils.training_controller.time.sleep'):
            c = TrainingController("/tmp", "python", event_hook=lambda e: events.append(e))
            c._emit({"event_type": "test"})
        assert len(events) == 1

    def test_emit_without_hook(self, ctrl):
        ctrl._emit({"event_type": "test"})


class TestCloseLogs:
    def test_close_role_logs(self, ctrl):
        mock = MagicMock()
        ctrl._proc_log_handles = [mock]
        ctrl._close_role_logs()
        mock.close.assert_called_once()
        assert ctrl._proc_log_handles == []
