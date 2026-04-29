"""Core type definitions for the federated learning system."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MsgType(IntEnum):
    """Message type enumeration."""

    # Centralized mode
    MODEL_BROADCAST = 1
    MODEL_UPDATE = 2
    CLIENT_REGISTER = 3
    CLIENT_ACK = 4

    # P2P mode
    NODE_JOIN = 10
    NODE_LEAVE = 11
    RING_PASS = 12
    HEARTBEAT = 13
    HEARTBEAT_ACK = 14
    TOPOLOGY_UPDATE = 15
    NODE_ACK = 16

    # Control messages
    TRAIN_START = 20
    TRAIN_COMPLETE = 21
    ROUND_END = 22
    ERROR = 99


class ErrorCode(IntEnum):
    """Error code enumeration."""

    # Connection errors (1-99)
    CONNECTION_REFUSED = 1
    CONNECTION_TIMEOUT = 2
    CONNECTION_CLOSED = 3

    # Protocol errors (100-199)
    INVALID_MESSAGE = 100
    DECODE_ERROR = 101
    INVALID_CHECKSUM = 102

    # Business errors (200-299)
    UNKNOWN_CLIENT = 200
    ROUND_MISMATCH = 201
    MODEL_VERSION_MISMATCH = 202

    # System errors (300-399)
    INTERNAL_ERROR = 300
    OUT_OF_MEMORY = 301


@dataclass
class TrainingResult:
    """Training result data."""

    loss: float
    accuracy: float
    num_samples: int
    training_time: float


@dataclass
class NetworkStats:
    """Network statistics for a single client/node."""

    client_id: int
    recv_size_bytes: int = 0
    recv_time_ms: float = 0.0
    send_size_bytes: int = 0
    send_time_ms: float = 0.0
    success: bool = True


@dataclass
class RoundStats:
    """Round statistics."""

    round_id: int
    broadcast_time: float
    training_times: dict[int, float] = field(default_factory=dict)
    collect_times: dict[int, float] = field(default_factory=dict)
    aggregate_time: float = 0.0
    total_time: float = 0.0
    participating_clients: list[int] = field(default_factory=list)
    timeout_clients: list[int] = field(default_factory=list)
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    network_stats: dict[int, NetworkStats] = field(default_factory=dict)
    broadcast_payload_size: int = 0


Weights = dict[str, Any]


# --- Config dataclasses ---


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "simple_cnn"
    input_channels: int = 1
    num_classes: int = 10


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str = "mnist"
    data_dir: str = "./data"
    num_clients: int = 3
    partition_strategy: str = "iid"
    alpha: float = 0.5


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.001
    monitor: str = "accuracy"  # "accuracy" or "loss"
    mode: str = "max"  # "max" for accuracy, "min" for loss


@dataclass
class TrainingConfig:
    """Training configuration."""

    rounds: int = 10
    epochs_per_round: int = 2
    learning_rate: float = 0.01
    batch_size: int = 32
    momentum: float = 0.9
    weight_decay: float = 0.0001
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class ServerTimeouts:
    """Server timeout configuration."""

    connect: float = 10.0
    send: float = 30.0
    recv: float = 30.0
    round: float = 60.0


@dataclass
class ServerConfig:
    """Centralized server configuration."""

    host: str = "0.0.0.0"
    port: int = 9000
    address: str = "127.0.0.1"
    timeouts: ServerTimeouts = field(default_factory=ServerTimeouts)


@dataclass
class ClientNodeConfig:
    """Single client node configuration."""

    id: int = 0
    host: str = "127.0.0.1"


@dataclass
class ClientsConfig:
    """Centralized clients configuration."""

    server_address: str | None = None
    nodes: list[ClientNodeConfig] = field(default_factory=list)


@dataclass
class PeerNodeConfig:
    """Single P2P peer node configuration."""

    id: int = 0
    host: str = "127.0.0.1"
    port: int = 9001


@dataclass
class PeersConfig:
    """Decentralized peers configuration."""

    local: bool = True
    nodes: list[PeerNodeConfig] = field(default_factory=list)


@dataclass
class P2PConfig:
    """P2P configuration."""

    topology: str = "ring"
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 10.0
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class AggregatorConfig:
    """Aggregator configuration."""

    name: str = "fedavg"
    fedprox_mu: float = 0.01


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_dir: str = "./logs"
    console_output: bool = True
    file_output: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""

    checkpoint_dir: str = "./outputs/checkpoints"
    figure_dir: str = "./outputs/figures"
    save_checkpoint_every: int = 5


@dataclass
class Config:
    """Top-level configuration."""

    mode: str = "centralized"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    server: ServerConfig | None = None
    clients: ClientsConfig | None = None
    p2p: P2PConfig | None = None
    peers: PeersConfig | None = None
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def num_clients(self) -> int:
        """Derive client count from mode-specific config."""
        if self.mode == "centralized" and self.clients is not None:
            return len(self.clients.nodes)
        if self.mode == "decentralized" and self.peers is not None:
            return len(self.peers.nodes)
        return self.dataset.num_clients

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Config: Parsed configuration object.

        Raises:
            ValueError: If mode-specific required sections are missing
                or num_clients is inconsistent.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        mode = data.get("mode", "centralized")

        # Parse training config with early stopping
        training_data = data.get("training", {})
        early_stopping_data = training_data.pop("early_stopping", {})
        training_config = TrainingConfig(
            **training_data,
            early_stopping=EarlyStoppingConfig(**early_stopping_data),
        )

        cfg = cls(
            mode=mode,
            model=ModelConfig(**data.get("model", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            training=training_config,
            aggregator=AggregatorConfig(**data.get("aggregator", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            output=OutputConfig(**data.get("output", {})),
        )

        # Parse mode-specific sections
        if mode == "centralized":
            server_data = data.get("server", {})
            cfg.server = ServerConfig(
                host=server_data.get("host", "0.0.0.0"),
                port=server_data.get("port", 9000),
                address=server_data.get("address", "127.0.0.1"),
                timeouts=ServerTimeouts(**server_data.get("timeouts", {})),
            )
            clients_data = data.get("clients", {})
            cfg.clients = ClientsConfig(
                server_address=clients_data.get("server_address"),
                nodes=[ClientNodeConfig(**n) for n in clients_data.get("nodes", [])],
            )
            if not cfg.clients.nodes:
                raise ValueError("Centralized mode requires 'clients.nodes' in config")
            # Validate num_clients consistency
            if cfg.dataset.num_clients != len(cfg.clients.nodes):
                raise ValueError(
                    f"dataset.num_clients ({cfg.dataset.num_clients}) must match "
                    f"clients.nodes count ({len(cfg.clients.nodes)})"
                )
        elif mode == "decentralized":
            p2p_data = data.get("p2p", {})
            cfg.p2p = P2PConfig(**p2p_data)
            peers_data = data.get("peers", {})
            cfg.peers = PeersConfig(
                local=peers_data.get("local", True),
                nodes=[PeerNodeConfig(**n) for n in peers_data.get("nodes", [])],
            )
            if not cfg.peers.nodes:
                raise ValueError("Decentralized mode requires 'peers.nodes' in config")
            if cfg.dataset.num_clients != len(cfg.peers.nodes):
                raise ValueError(
                    f"dataset.num_clients ({cfg.dataset.num_clients}) must match "
                    f"peers.nodes count ({len(cfg.peers.nodes)})"
                )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'centralized' or 'decentralized'")

        return cfg
