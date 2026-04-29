"""Unit tests for federated server."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from src.core.types import EarlyStoppingConfig, RoundStats, ServerTimeouts
from src.server.aggregator import FedAvg
from src.server.checkpoint import load_checkpoint, save_checkpoint
from src.server.federated_server import FederatedServer
from src.server.round_coordinator import RoundCoordinator
from src.utils.logger import FedLogger
from src.utils.metrics import MetricsCollector


class TestFederatedServer:
    """Tests for FederatedServer."""

    def test_server_initialization(self) -> None:
        """Test server initialization with default parameters."""
        server = FederatedServer()
        assert server._running is False
        assert isinstance(server._aggregator, FedAvg)
        assert isinstance(server._metrics, MetricsCollector)
        assert isinstance(server._timeouts, ServerTimeouts)
        assert server._checkpoint_dir == "./outputs/checkpoints"
        assert server._save_checkpoint_every == 5

    def test_server_initialization_with_custom_params(self) -> None:
        """Test server initialization with custom parameters."""
        aggregator = FedAvg()
        logger = FedLogger(name="TestServer", console_output=False, file_output=False)
        timeouts = ServerTimeouts(connect=5.0, send=10.0, recv=20.0, round=60.0)
        early_stopping = EarlyStoppingConfig(enabled=True, patience=3)

        server = FederatedServer(
            aggregator=aggregator,
            logger=logger,
            timeouts=timeouts,
            early_stopping=early_stopping,
            checkpoint_dir="./custom_checkpoints",
            save_checkpoint_every=3,
        )

        assert server._aggregator is aggregator
        assert server._logger is logger
        assert server._timeouts is timeouts
        assert server._early_stopping._config.enabled is True
        assert server._checkpoint_dir == "./custom_checkpoints"
        assert server._save_checkpoint_every == 3

    def test_set_test_dataloader(self) -> None:
        """Test setting test dataloader."""
        server = FederatedServer()
        mock_dataloader = MagicMock()

        server.set_test_dataloader(mock_dataloader)
        assert server._test_dataloader is mock_dataloader

    def test_set_model(self) -> None:
        """Test setting model."""
        server = FederatedServer()
        mock_model = MagicMock()

        server.set_model(mock_model)
        assert server._model is mock_model

    def test_format_size(self) -> None:
        """Test size formatting utility."""
        assert FederatedServer._format_size(100) == "100.0B"
        assert FederatedServer._format_size(1024) == "1.0KB"
        assert FederatedServer._format_size(1536) == "1.5KB"
        assert FederatedServer._format_size(1024 * 1024) == "1.0MB"
        assert FederatedServer._format_size(1024 * 1024 * 1024) == "1.0GB"

    def test_save_checkpoint(self) -> None:
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            server = FederatedServer(checkpoint_dir=tmpdir)
            server._global_weights = {"weight": torch.tensor([1.0, 2.0])}

            server._save_checkpoint(5)

            checkpoint_path = Path(tmpdir) / "round_5.pt"
            assert checkpoint_path.exists()

            # Verify checkpoint can be loaded
            weights, round_id = load_checkpoint(str(checkpoint_path))
            assert round_id == 5
            assert "weight" in weights

    def test_save_checkpoint_failure(self) -> None:
        """Test checkpoint saving failure handling."""
        server = FederatedServer(checkpoint_dir="/invalid/path/that/does/not/exist")
        server._global_weights = {"weight": torch.tensor([1.0])}

        # Should not raise exception
        server._save_checkpoint(1)

    def test_evaluate_global_without_model(self) -> None:
        """Test global evaluation when model is not set."""
        server = FederatedServer()
        server._global_weights = {}

        acc, loss = server._evaluate_global()
        assert acc == 0.0
        assert loss == 0.0

    def test_evaluate_global_without_dataloader(self) -> None:
        """Test global evaluation when dataloader is not set."""
        server = FederatedServer()
        server._model = MagicMock()
        server._global_weights = {}

        acc, loss = server._evaluate_global()
        assert acc == 0.0
        assert loss == 0.0

    def test_evaluate_global_with_mock(self) -> None:
        """Test global evaluation with mock model."""
        server = FederatedServer()
        server._global_weights = {"weight": torch.tensor([1.0])}

        mock_model = MagicMock()
        mock_model.evaluate.return_value = MagicMock(accuracy=85.5, loss=0.5)
        server._model = mock_model
        server._test_dataloader = MagicMock()

        acc, loss = server._evaluate_global()
        assert acc == 85.5
        assert loss == 0.5
        mock_model.set_weights.assert_called_once_with(server._global_weights)


class TestRoundCoordinator:
    """Tests for RoundCoordinator."""

    def test_start_round(self) -> None:
        """Test starting a new round."""
        mock_client_manager = MagicMock()
        mock_aggregator = MagicMock()
        coordinator = RoundCoordinator(mock_client_manager, mock_aggregator)

        coordinator.start_round(5)

        assert coordinator._current_round == 5
        assert coordinator._network_stats == {}
        assert coordinator._broadcast_payload_size == 0

    def test_broadcast_model(self) -> None:
        """Test broadcasting model to clients."""
        mock_client_manager = MagicMock()
        mock_aggregator = MagicMock()
        coordinator = RoundCoordinator(mock_client_manager, mock_aggregator)
        coordinator._current_round = 1

        weights = {"layer": torch.tensor([1.0, 2.0])}
        duration = coordinator.broadcast_model(weights)

        assert duration >= 0
        assert coordinator._broadcast_payload_size > 0
        mock_client_manager.broadcast_message.assert_called_once()

    def test_collect_updates_with_timeouts(self) -> None:
        """Test collecting updates with some timeouts."""
        mock_client_manager = MagicMock()
        mock_aggregator = MagicMock()
        coordinator = RoundCoordinator(mock_client_manager, mock_aggregator)
        coordinator._current_round = 1

        # Simulate one successful response and one timeout
        from src.core.types import MsgType
        from src.protocol.message import Message

        success_msg = Message(
            msg_type=MsgType.MODEL_UPDATE,
            payload={"weights": b"serialized_weights"},
        )

        mock_client_manager.collect_messages.return_value = {
            1: success_msg,
            2: None,  # Timeout
        }

        updates = coordinator.collect_updates(timeout=1.0)

        assert 1 in updates
        assert 2 in updates
        assert updates[2] is None
        assert 1 in coordinator._network_stats
        assert 2 in coordinator._network_stats
        assert coordinator._network_stats[2].success is False


class TestCheckpoint:
    """Tests for checkpoint functionality."""

    def test_save_and_load_checkpoint(self) -> None:
        """Test saving and loading checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {"layer1": torch.tensor([1.0, 2.0, 3.0])}
            path = os.path.join(tmpdir, "checkpoint.pt")

            save_checkpoint(weights, path, round_id=10)

            loaded_weights, loaded_round = load_checkpoint(path)

            assert loaded_round == 10
            assert "layer1" in loaded_weights
            assert torch.allclose(loaded_weights["layer1"], weights["layer1"])

    def test_save_checkpoint_creates_directory(self) -> None:
        """Test that save_checkpoint creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {"layer": torch.tensor([1.0])}
            nested_path = os.path.join(tmpdir, "nested", "dir", "checkpoint.pt")

            save_checkpoint(weights, nested_path)

            assert Path(nested_path).exists()

    def test_load_checkpoint_without_round_id(self) -> None:
        """Test loading checkpoint without round_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {"layer": torch.tensor([1.0])}
            path = os.path.join(tmpdir, "checkpoint.pt")

            save_checkpoint(weights, path)  # No round_id

            loaded_weights, loaded_round = load_checkpoint(path)

            assert loaded_round is None
            assert "layer" in loaded_weights


class TestServerIntegration:
    """Integration tests for server components."""

    def test_server_with_mock_training(self) -> None:
        """Test server with mocked training loop."""
        server = FederatedServer(
            logger=FedLogger(name="TestServer", console_output=False, file_output=False),
            checkpoint_dir=tempfile.mkdtemp(),
            save_checkpoint_every=1,
        )

        # Setup mock model
        mock_model = MagicMock()
        mock_model.evaluate.return_value = MagicMock(accuracy=80.0, loss=0.5)
        server._model = mock_model
        server._test_dataloader = MagicMock()
        server._global_weights = {"weight": torch.tensor([1.0])}

        # Mock the coordinator methods
        with (
            patch.object(server._coordinator, "start_round") as mock_start,
            patch.object(
                server._coordinator, "broadcast_model", return_value=0.1
            ) as mock_broadcast,
            patch.object(
                server._coordinator,
                "collect_updates",
                return_value={1: {"weight": torch.tensor([1.1])}},
            ) as mock_collect,
            patch.object(
                server._aggregator, "aggregate", return_value={"weight": torch.tensor([1.05])}
            ),
        ):
            stats = server.run_round(1)

            assert isinstance(stats, RoundStats)
            assert stats.round_id == 1
            mock_start.assert_called_once_with(1)
            mock_broadcast.assert_called_once()
            mock_collect.assert_called_once()

    def test_run_round_with_no_participants(self) -> None:
        """Test running round with no participating clients."""
        server = FederatedServer(
            logger=FedLogger(name="TestServer", console_output=False, file_output=False),
        )

        server._global_weights = {"weight": torch.tensor([1.0])}

        with (
            patch.object(server._coordinator, "start_round"),
            patch.object(server._coordinator, "broadcast_model", return_value=0.1),
            patch.object(server._coordinator, "collect_updates", return_value={}),
            patch.object(server, "_evaluate_global", return_value=(0.0, 0.0)),
        ):
            stats = server.run_round(1)

            assert stats.participating_clients == []
            assert stats.timeout_clients == []
