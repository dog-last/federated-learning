"""Abstract interfaces for the federated learning system."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from torch.utils.data import DataLoader

from src.core.types import RoundStats, TrainingResult, Weights

# --- Model layer ---


class IModel(ABC):
    """Model interface."""

    @abstractmethod
    def get_weights(self) -> Weights:
        """Get model weights.

        Returns:
            Weights: Model weights dictionary.
        """
        ...

    @abstractmethod
    def set_weights(self, weights: Weights) -> None:
        """Set model weights.

        Args:
            weights: Model weights dictionary.
        """
        ...

    @abstractmethod
    def train_epoch(self, dataloader: DataLoader, lr: float, epoch: int = 1) -> TrainingResult:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.
            lr: Learning rate.
            epoch: Current epoch number.

        Returns:
            TrainingResult: Training result.
        """
        ...

    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> TrainingResult:
        """Evaluate the model.

        Args:
            dataloader: Evaluation data loader.

        Returns:
            TrainingResult: Evaluation result.
        """
        ...

    @property
    @abstractmethod
    def model_size(self) -> int:
        """Number of model parameters.

        Returns:
            int: Parameter count.
        """
        ...


# --- Data layer ---


class IPartitioner(ABC):
    """Data partitioner interface."""

    @abstractmethod
    def partition(self, dataset: Any, num_clients: int, strategy: str = "iid") -> list[Any]:
        """Partition a dataset among clients.

        Args:
            dataset: Original dataset.
            num_clients: Number of clients.
            strategy: Partitioning strategy ("iid" or "non_iid").

        Returns:
            List[Any]: List of partitioned datasets.
        """
        ...

    @abstractmethod
    def get_client_data(self, client_id: int) -> Any:
        """Get the dataset for a specific client.

        Args:
            client_id: Client ID.

        Returns:
            Client dataset.
        """
        ...


# --- Client layer ---


class IClient(ABC):
    """Client interface."""

    @abstractmethod
    def connect(self, host: str, port: int) -> None:
        """Connect to the server.

        Args:
            host: Server address.
            port: Server port.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the server."""
        ...

    @abstractmethod
    def register(self) -> int:
        """Register with the server.

        Returns:
            int: Assigned client ID.
        """
        ...

    @abstractmethod
    def receive_model(self, timeout: float = 30.0) -> Weights:
        """Receive the global model.

        Args:
            timeout: Receive timeout in seconds.

        Returns:
            Weights: Model weights.
        """
        ...

    @abstractmethod
    def send_update(self, weights: Weights) -> None:
        """Send model update.

        Args:
            weights: Trained model weights.
        """
        ...

    @abstractmethod
    def run(self, num_rounds: int) -> None:
        """Run the client main loop.

        Args:
            num_rounds: Number of rounds to participate in.
        """
        ...

    @property
    @abstractmethod
    def client_id(self) -> int:
        """Client ID.

        Returns:
            int: The client ID.
        """
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the client is connected.

        Returns:
            bool: Connection status.
        """
        ...


class ITrainer(ABC):
    """Local trainer interface."""

    @abstractmethod
    def train(
        self, model: IModel, dataloader: DataLoader, epochs: int, lr: float
    ) -> TrainingResult:
        """Execute local training.

        Args:
            model: Model instance.
            dataloader: Training data.
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            TrainingResult: Training result.
        """
        ...


# --- Server layer ---


class IServer(ABC):
    """Server interface."""

    @abstractmethod
    def start(self, port: int) -> None:
        """Start the server.

        Args:
            port: Listening port.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the server."""
        ...

    @abstractmethod
    def wait_for_clients(self, num_clients: int, timeout: float = 60.0) -> list[int]:
        """Wait for clients to connect.

        Args:
            num_clients: Expected number of clients.
            timeout: Wait timeout in seconds.

        Returns:
            List[int]: List of connected client IDs.
        """
        ...

    @abstractmethod
    def run_round(self, round_id: int) -> RoundStats:
        """Execute one round of federated learning.

        Args:
            round_id: Round ID.

        Returns:
            RoundStats: Round statistics.
        """
        ...

    @abstractmethod
    def run(self, num_rounds: int) -> None:
        """Run the server main loop.

        Args:
            num_rounds: Total number of rounds.
        """
        ...

    @property
    @abstractmethod
    def global_weights(self) -> Weights:
        """Current global model weights.

        Returns:
            Weights: Global model weights.
        """
        ...


class IAggregator(ABC):
    """Aggregator interface."""

    @abstractmethod
    def aggregate(
        self, weights_list: list[Weights], client_sizes: list[int] | None = None
    ) -> Weights:
        """Aggregate multiple model weights.

        Args:
            weights_list: List of model weights.
            client_sizes: Number of samples per client (for weighted averaging).

        Returns:
            Weights: Aggregated weights.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Aggregator name.

        Returns:
            str: Name of the aggregator.
        """
        ...


class IClientManager(ABC):
    """Client connection manager interface."""

    @abstractmethod
    def add_client(self, conn: Any) -> int:
        """Add a client connection.

        Args:
            conn: Connection object.

        Returns:
            int: Assigned client ID.
        """
        ...

    @abstractmethod
    def remove_client(self, client_id: int) -> None:
        """Remove a client.

        Args:
            client_id: Client ID.
        """
        ...

    @abstractmethod
    def get_connection(self, client_id: int) -> Any | None:
        """Get a client connection.

        Args:
            client_id: Client ID.

        Returns:
            Connection object, or None if not found.
        """
        ...

    @abstractmethod
    def broadcast(self, data: bytes, exclude: list[int] | None = None) -> dict[int, bool]:
        """Broadcast data to all clients.

        Args:
            data: Data to broadcast.
            exclude: List of client IDs to exclude.

        Returns:
            Dict[int, bool]: Send status per client.
        """
        ...

    @abstractmethod
    def collect(self, timeout: float) -> dict[int, bytes | None]:
        """Collect data from all clients.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Dict[int, Optional[bytes]]: Data per client; None for timeouts.
        """
        ...

    @property
    @abstractmethod
    def client_ids(self) -> list[int]:
        """All client IDs.

        Returns:
            List[int]: List of client IDs.
        """
        ...

    @property
    @abstractmethod
    def num_clients(self) -> int:
        """Current number of clients.

        Returns:
            int: Client count.
        """
        ...


class IRoundCoordinator(ABC):
    """Round coordinator interface."""

    @abstractmethod
    def start_round(self, round_id: int) -> None:
        """Start a new round.

        Args:
            round_id: Round ID.
        """
        ...

    @abstractmethod
    def broadcast_model(self, weights: Weights) -> float:
        """Broadcast global model.

        Args:
            weights: Model weights.

        Returns:
            float: Broadcast duration in seconds.
        """
        ...

    @abstractmethod
    def collect_updates(self, timeout: float) -> dict[int, Weights | None]:
        """Collect client updates.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Dict[int, Optional[Weights]]: Updates per client; None for timeouts.
        """
        ...

    @abstractmethod
    def end_round(self, aggregated_weights: Weights) -> RoundStats:
        """End the round.

        Args:
            aggregated_weights: Aggregated weights.

        Returns:
            RoundStats: Round statistics.
        """
        ...


# --- P2P layer ---


class IP2PNode(ABC):
    """P2P node interface."""

    @abstractmethod
    def start(self, port: int) -> None:
        """Start the node.

        Args:
            port: Listening port.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the node."""
        ...

    @abstractmethod
    def join_ring(self, bootstrap_node: tuple[str, int]) -> None:
        """Join a ring network.

        Args:
            bootstrap_node: Bootstrap node address (host, port).
        """
        ...

    @abstractmethod
    def leave_ring(self) -> None:
        """Leave the ring network."""
        ...

    @abstractmethod
    def pass_model(self, weights: Weights) -> bool:
        """Pass model to next node.

        Args:
            weights: Model weights.

        Returns:
            bool: Whether the pass succeeded.
        """
        ...

    @abstractmethod
    def run(self, num_rounds: int) -> None:
        """Run the node main loop.

        Args:
            num_rounds: Number of rounds to participate in.
        """
        ...

    @property
    @abstractmethod
    def node_id(self) -> int:
        """Node ID.

        Returns:
            int: The node ID.
        """
        ...

    @property
    @abstractmethod
    def next_node(self) -> tuple[str, int] | None:
        """Next node address.

        Returns:
            Optional[Tuple[str, int]]: Next node address, or None.
        """
        ...

    @property
    @abstractmethod
    def prev_node(self) -> tuple[str, int] | None:
        """Previous node address.

        Returns:
            Optional[Tuple[str, int]]: Previous node address, or None.
        """
        ...


class ITopologyManager(ABC):
    """Topology manager interface."""

    @abstractmethod
    def get_next_node(self, current_id: int) -> int | None:
        """Get the next node ID in the ring.

        Args:
            current_id: Current node ID.

        Returns:
            Optional[int]: Next node ID, or None if ring is broken.
        """
        ...

    @abstractmethod
    def handle_failure(self, failed_id: int) -> None:
        """Handle node failure and restructure topology.

        Args:
            failed_id: Failed node ID.
        """
        ...

    @abstractmethod
    def add_node(self, node_id: int, address: tuple[str, int]) -> None:
        """Add a node to the topology.

        Args:
            node_id: Node ID.
            address: Node address (host, port).
        """
        ...

    @abstractmethod
    def remove_node(self, node_id: int) -> None:
        """Remove a node from the topology.

        Args:
            node_id: Node ID.
        """
        ...

    @property
    @abstractmethod
    def ring_order(self) -> list[int]:
        """Ring order (list of node IDs).

        Returns:
            List[int]: Node IDs in ring order.
        """
        ...


class IFailureDetector(ABC):
    """Failure detector interface."""

    @abstractmethod
    def start(self) -> None:
        """Start failure detection."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop failure detection."""
        ...

    @abstractmethod
    def heartbeat(self, target_id: int) -> bool:
        """Send heartbeat to target node.

        Args:
            target_id: Target node ID.

        Returns:
            bool: Whether a response was received.
        """
        ...

    @abstractmethod
    def check(self, node_id: int, timeout: float) -> bool:
        """Check if a node is alive.

        Args:
            node_id: Node ID.
            timeout: Timeout in seconds.

        Returns:
            bool: Whether the node is alive.
        """
        ...

    @abstractmethod
    def get_failed_nodes(self) -> list[int]:
        """Get the list of failed nodes.

        Returns:
            List[int]: Failed node IDs.
        """
        ...

    @abstractmethod
    def on_failure(self, callback: Callable[[int], None]) -> None:
        """Register a failure callback.

        Args:
            callback: Callback function taking a failed node ID.
        """
        ...


# --- Protocol layer ---


class IMessage(ABC):
    """Message interface.

    Implementations should provide these as instance attributes
    (e.g. via dataclass fields) or properties.
    """

    msg_type: int
    client_id: int | None
    payload: Any
    timestamp: float
    round_id: int | None


class ICodec(ABC):
    """Codec interface."""

    @abstractmethod
    def encode(self, message: "IMessage") -> bytes:
        """Encode a message to bytes.

        Args:
            message: Message object.

        Returns:
            bytes: Encoded byte stream.
        """
        ...

    @abstractmethod
    def decode(self, data: bytes) -> "IMessage":
        """Decode bytes to a message.

        Args:
            data: Byte stream.

        Returns:
            IMessage: Decoded message.
        """
        ...


class ISerializer(ABC):
    """Serializer interface."""

    @abstractmethod
    def serialize_weights(self, weights: Weights) -> bytes:
        """Serialize model weights.

        Args:
            weights: Model weights.

        Returns:
            bytes: Serialized byte stream.
        """
        ...

    @abstractmethod
    def deserialize_weights(self, data: bytes) -> Weights:
        """Deserialize model weights.

        Args:
            data: Byte stream.

        Returns:
            Weights: Model weights.
        """
        ...

    @abstractmethod
    def get_size(self, weights: Weights) -> int:
        """Get the serialized size of weights.

        Args:
            weights: Model weights.

        Returns:
            int: Size in bytes.
        """
        ...


# --- Transport layer ---


class IConnection(ABC):
    """Connection interface."""

    @abstractmethod
    def send(self, data: bytes) -> None:
        """Send data.

        Args:
            data: Byte stream to send.
        """
        ...

    @abstractmethod
    def recv(self, size: int, timeout: float | None = None) -> bytes:
        """Receive a specified number of bytes.

        Args:
            size: Number of bytes to receive.
            timeout: Timeout in seconds.

        Returns:
            bytes: Received data.

        Raises:
            NetworkTimeoutError: Receive timeout.
        """
        ...

    @abstractmethod
    def recv_all(self, timeout: float | None = None) -> bytes:
        """Receive a complete message (handles TCP framing).

        Receives 8-byte length prefix first, then the complete data.

        Args:
            timeout: Timeout in seconds.

        Returns:
            bytes: Complete message data.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the connection."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the connection is active.

        Returns:
            bool: Connection status.
        """
        ...

    @property
    @abstractmethod
    def remote_address(self) -> tuple[str, int]:
        """Remote address.

        Returns:
            Tuple[str, int]: (host, port) of the remote endpoint.
        """
        ...


class IListener(ABC):
    """Listener interface."""

    @abstractmethod
    def bind(self, host: str, port: int) -> None:
        """Bind to an address.

        Args:
            host: Host address.
            port: Port number.
        """
        ...

    @abstractmethod
    def accept(self, timeout: float | None = None) -> "IConnection":
        """Accept a connection.

        Args:
            timeout: Timeout in seconds.

        Returns:
            IConnection: Connection object.

        Raises:
            NetworkTimeoutError: Accept timeout.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the listener."""
        ...

    @property
    @abstractmethod
    def is_listening(self) -> bool:
        """Whether the listener is active.

        Returns:
            bool: Listening status.
        """
        ...

    @property
    @abstractmethod
    def address(self) -> tuple[str, int]:
        """Listening address.

        Returns:
            Tuple[str, int]: (host, port) of the listener.
        """
        ...


# --- Utils layer ---


class ILogger(ABC):
    """Logger interface."""

    @abstractmethod
    def info(self, msg: str, **kwargs: Any) -> None:
        """Log an INFO message."""
        ...

    @abstractmethod
    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log a WARNING message."""
        ...

    @abstractmethod
    def error(self, msg: str, **kwargs: Any) -> None:
        """Log an ERROR message."""
        ...

    @abstractmethod
    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log a DEBUG message."""
        ...

    @abstractmethod
    def log_round(self, stats: RoundStats) -> None:
        """Log round statistics.

        Args:
            stats: Round statistics.
        """
        ...

    @abstractmethod
    def log_network(
        self,
        event: str,
        client_id: int | None = None,
        size: int | None = None,
        duration: float | None = None,
        success: bool = True,
    ) -> None:
        """Log a network event.

        Args:
            event: Event type.
            client_id: Client ID.
            size: Data size in bytes.
            duration: Duration in seconds.
            success: Whether the event succeeded.
        """
        ...


class IMetricsCollector(ABC):
    """Metrics collector interface."""

    @abstractmethod
    def record_round(self, stats: RoundStats) -> None:
        """Record round statistics.

        Args:
            stats: Round statistics.
        """
        ...

    @abstractmethod
    def get_accuracy_history(self) -> list[float]:
        """Get accuracy history.

        Returns:
            List[float]: Accuracy per round.
        """
        ...

    @abstractmethod
    def get_loss_history(self) -> list[float]:
        """Get loss history.

        Returns:
            List[float]: Loss per round.
        """
        ...

    @abstractmethod
    def get_round_times(self) -> list[float]:
        """Get round durations.

        Returns:
            List[float]: Duration per round.
        """
        ...

    @abstractmethod
    def export(self, path: str) -> None:
        """Export metrics to a file.

        Args:
            path: Output file path.
        """
        ...

    @abstractmethod
    def plot_accuracy(self, path: str) -> None:
        """Plot accuracy curve and save.

        Args:
            path: Output file path for the plot.
        """
        ...
