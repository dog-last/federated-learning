"""Federated learning client with TCP communication."""

import socket
import time

from torch.utils.data import DataLoader

from src.client.base import BaseClient
from src.client.trainer import LocalTrainer
from src.core.interfaces import IModel
from src.core.types import MsgType, Weights
from src.protocol.codec import Codec
from src.protocol.message import Message
from src.protocol.serializer import TorchSerializer
from src.transport.connection import Connection
from src.utils.logger import FedLogger


class FederatedClient(BaseClient):
    """Federated learning client that communicates over TCP.

    Attributes:
        model: Local model.
        dataloader: Local training data loader.
        trainer: Local trainer.
        logger: Logger instance.
        _conn: TCP connection.
        _codec: Message codec.
        _serializer: Weight serializer.
    """

    def __init__(
        self,
        model: IModel,
        dataloader: DataLoader,
        trainer: LocalTrainer | None = None,
        logger: FedLogger | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.trainer = trainer or LocalTrainer()
        self.logger = logger or FedLogger(name="Client", console_output=True, file_output=False)
        self._conn: Connection | None = None
        self._codec = Codec()
        self._serializer = TorchSerializer()

    def connect(self, host: str, port: int) -> None:
        """Connect to the server.

        Args:
            host: Server address.
            port: Server port.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        self._conn = Connection(sock, self._codec)
        self._connected = True
        self.logger.info(f"Connected to {host}:{port}")

    def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._connected = False
        self.logger.info("Disconnected")

    def register(self) -> int:
        """Register with the server.

        Returns:
            int: Assigned client ID.
        """
        assert self._conn is not None
        msg = Message(
            msg_type=MsgType.CLIENT_REGISTER,
            payload={"data_size": len(self.dataloader.dataset)},
        )
        self._conn.send_message(msg)

        ack = self._conn.recv_message(timeout=10.0)
        self._client_id = ack.payload.get("client_id", -1)
        self.logger.info(f"Registered as Client-{self._client_id}")
        return self._client_id

    def receive_model(self, timeout: float = 120.0) -> Weights:
        """Receive the global model.

        Args:
            timeout: Receive timeout in seconds.

        Returns:
            Weights: Model weights.
        """
        assert self._conn is not None
        start = time.time()
        msg = self._conn.recv_message(timeout=timeout)
        payload = msg.payload
        serialized = payload["weights"]
        weights = self._serializer.deserialize_weights(serialized)
        elapsed = time.time() - start
        self.logger.log_network("recv", size=len(serialized), duration=elapsed, success=True)
        return weights

    def send_update(self, weights: Weights) -> None:
        """Send model update.

        Args:
            weights: Trained model weights.
        """
        assert self._conn is not None
        serialized = self._serializer.serialize_weights(weights)
        start = time.time()
        msg = Message(
            msg_type=MsgType.MODEL_UPDATE,
            client_id=self._client_id,
            payload={"weights": serialized},
        )
        self._conn.send_message(msg)
        elapsed = time.time() - start
        self.logger.log_network(
            "send", client_id=self._client_id, size=len(serialized), duration=elapsed, success=True
        )

    def run(self, num_rounds: int, epochs: int = 2, lr: float = 0.01) -> None:
        """Run the client main loop.

        Args:
            num_rounds: Number of rounds to participate in.
            epochs: Number of local training epochs per round.
            lr: Local learning rate.
        """
        assert self._conn is not None
        for round_id in range(1, num_rounds + 1):
            # Receive global model
            weights = self.receive_model()
            self.model.set_weights(weights)

            # Local training
            result = self.trainer.train(
                self.model,
                self.dataloader,
                epochs=epochs,
                lr=lr,
            )
            self.logger.log_training(
                self._client_id, round_id, result.loss, result.accuracy, result.training_time
            )

            # Send update
            self.send_update(self.model.get_weights())

        self.logger.info("All rounds completed")
