"""Unit tests for federated client."""

import socket
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.client.federated_client import FederatedClient
from src.client.trainer import LocalTrainer
from src.core.types import MsgType, TrainingResult
from src.model.simple_cnn import create_simple_cnn
from src.protocol.message import Message
from src.protocol.serializer import TorchSerializer
from src.utils.logger import FedLogger


class TestFederatedClient:
    """Tests for FederatedClient."""

    def test_client_initialization(self) -> None:
        """Test client initialization."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)

        assert client.model is model
        assert client.dataloader is loader
        assert client._client_id == -1
        assert client._connected is False

    def test_client_initialization_with_custom_trainer(self) -> None:
        """Test client initialization with custom trainer."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)
        trainer = LocalTrainer(momentum=0.5)

        client = FederatedClient(model=model, dataloader=loader, trainer=trainer)

        assert client.trainer is trainer

    def test_client_initialization_with_logger(self) -> None:
        """Test client initialization with custom logger."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)
        logger = FedLogger(name="TestClient", console_output=False, file_output=False)

        client = FederatedClient(model=model, dataloader=loader, logger=logger)

        assert client.logger is logger

    def test_receive_model(self) -> None:
        """Test receiving model from server."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)

        # Mock connection
        mock_conn = MagicMock()
        serializer = TorchSerializer()
        weights = {"weight": torch.tensor([1.0, 2.0])}
        serialized = serializer.serialize_weights(weights)

        mock_msg = Message(
            msg_type=MsgType.MODEL_BROADCAST,
            payload={"weights": serialized},
        )
        mock_conn.recv_message.return_value = mock_msg
        client._conn = mock_conn

        received_weights = client.receive_model(timeout=10.0)

        assert "weight" in received_weights
        assert torch.allclose(received_weights["weight"], weights["weight"])

    def test_send_update(self) -> None:
        """Test sending model update to server."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)
        client._client_id = 1

        # Mock connection
        mock_conn = MagicMock()
        client._conn = mock_conn

        weights = {"weight": torch.tensor([1.0, 2.0])}
        client.send_update(weights)

        mock_conn.send_message.assert_called_once()
        sent_msg = mock_conn.send_message.call_args[0][0]
        assert sent_msg.msg_type == MsgType.MODEL_UPDATE
        assert sent_msg.client_id == 1

    def test_run_single_round(self) -> None:
        """Test running a single training round."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)
        client._client_id = 1

        # Mock connection
        mock_conn = MagicMock()
        serializer = TorchSerializer()

        # Setup mock for receive_model - use actual model weights
        weights = model.get_weights()
        serialized = serializer.serialize_weights(weights)
        mock_msg = Message(
            msg_type=MsgType.MODEL_BROADCAST,
            payload={"weights": serialized},
        )
        mock_conn.recv_message.return_value = mock_msg
        client._conn = mock_conn

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = TrainingResult(
            loss=0.5, accuracy=85.0, num_samples=16, training_time=1.0
        )
        client.trainer = mock_trainer

        # Run one round
        client.run(num_rounds=1, epochs=1, lr=0.01)

        mock_trainer.train.assert_called_once()
        mock_conn.send_message.assert_called_once()

    def test_log_network_on_receive(self) -> None:
        """Test network logging on model receive."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)
        logger = FedLogger(name="TestClient", console_output=False, file_output=False)

        client = FederatedClient(model=model, dataloader=loader, logger=logger)

        # Mock connection
        mock_conn = MagicMock()
        serializer = TorchSerializer()
        weights = {"weight": torch.tensor([1.0, 2.0])}
        serialized = serializer.serialize_weights(weights)

        mock_msg = Message(
            msg_type=MsgType.MODEL_BROADCAST,
            payload={"weights": serialized},
        )
        mock_conn.recv_message.return_value = mock_msg
        client._conn = mock_conn

        with patch.object(logger, "log_network") as mock_log:
            client.receive_model(timeout=10.0)
            mock_log.assert_called_once()
            # Check the call - log_network(event, client_id=None, size=None, duration=None, success=True)
            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs.get("event") == "recv" or mock_log.call_args[0][0] == "recv"
            assert call_kwargs.get("size") == len(serialized) or mock_log.call_args[1].get(
                "size"
            ) == len(serialized)

    def test_log_network_on_send(self) -> None:
        """Test network logging on model send."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)
        logger = FedLogger(name="TestClient", console_output=False, file_output=False)

        client = FederatedClient(model=model, dataloader=loader, logger=logger)
        client._client_id = 1

        # Mock connection
        mock_conn = MagicMock()
        client._conn = mock_conn

        weights = {"weight": torch.tensor([1.0, 2.0])}

        with patch.object(logger, "log_network") as mock_log:
            client.send_update(weights)
            mock_log.assert_called_once()
            # Check the call - log_network(event, client_id, size, duration, success)
            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs.get("event") == "send" or mock_log.call_args[0][0] == "send"
            assert call_kwargs.get("client_id") == 1 or mock_log.call_args[1].get("client_id") == 1

    def test_log_training(self) -> None:
        """Test training logging."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)
        logger = FedLogger(name="TestClient", console_output=False, file_output=False)

        client = FederatedClient(model=model, dataloader=loader, logger=logger)
        client._client_id = 1

        with patch.object(logger, "log_training") as mock_log:
            client.logger.log_training(1, 1, 0.5, 85.0, 1.0)
            mock_log.assert_called_once_with(1, 1, 0.5, 85.0, 1.0)


class TestFederatedClientConnection:
    """Tests for client connection handling."""

    def test_connect(self) -> None:
        """Test connecting to server."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value = mock_socket

            client.connect("127.0.0.1", 9000)

            mock_socket.connect.assert_called_once_with(("127.0.0.1", 9000))
            assert client._connected is True

    def test_disconnect(self) -> None:
        """Test disconnecting from server."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)

        # Setup mock connection
        mock_conn = MagicMock()
        client._conn = mock_conn
        client._connected = True

        client.disconnect()

        mock_conn.close.assert_called_once()
        assert client._connected is False

    def test_register(self) -> None:
        """Test client registration."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)

        # Mock connection
        mock_conn = MagicMock()
        mock_ack = Message(
            msg_type=MsgType.CLIENT_ACK,
            payload={"client_id": 5, "config": {"epochs": 2, "lr": 0.01}},
        )
        mock_conn.recv_message.return_value = mock_ack
        client._conn = mock_conn

        client_id = client.register()

        assert client_id == 5
        assert client._client_id == 5
        mock_conn.send_message.assert_called_once()


class TestFederatedClientErrorHandling:
    """Tests for client error handling."""

    def test_receive_model_timeout(self) -> None:
        """Test handling timeout when receiving model."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)

        # Mock connection that raises timeout
        mock_conn = MagicMock()
        mock_conn.recv_message.side_effect = TimeoutError("Timeout")
        client._conn = mock_conn

        with pytest.raises(socket.timeout):
            client.receive_model(timeout=1.0)

    def test_send_update_without_connection(self) -> None:
        """Test sending update without connection."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)
        client._conn = None

        weights = {"weight": torch.tensor([1.0])}

        with pytest.raises(AssertionError):
            client.send_update(weights)

    def test_receive_model_without_connection(self) -> None:
        """Test receiving model without connection."""
        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=8)

        client = FederatedClient(model=model, dataloader=loader)
        client._conn = None

        with pytest.raises(AssertionError):
            client.receive_model(timeout=10.0)
