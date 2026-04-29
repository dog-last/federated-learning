"""Round coordinator for managing federated learning rounds."""

import time

from src.core.interfaces import IAggregator, IClientManager, IRoundCoordinator
from src.core.types import MsgType, NetworkStats, RoundStats, Weights
from src.protocol.message import Message
from src.protocol.serializer import TorchSerializer
from src.utils.logger import FedLogger
from src.utils.timer import Timer


class RoundCoordinator(IRoundCoordinator):
    """Coordinates a single round: broadcast, collect, aggregate.

    Attributes:
        _client_manager: Client connection manager.
        _aggregator: Aggregation strategy.
        _serializer: Weight serializer.
        _logger: Logger instance.
        _current_round: Current round ID.
        _round_start_time: Timestamp when the round started.
        _network_stats: Network statistics per client for current round.
        _broadcast_payload_size: Size of broadcast payload in bytes.
    """

    def __init__(
        self,
        client_manager: IClientManager,
        aggregator: IAggregator,
        logger: FedLogger | None = None,
    ) -> None:
        self._client_manager = client_manager
        self._aggregator = aggregator
        self._serializer = TorchSerializer()
        self._logger = logger or FedLogger(
            name="RoundCoordinator", console_output=True, file_output=False
        )
        self._current_round: int = 0
        self._round_start_time: float = 0.0
        self._network_stats: dict[int, NetworkStats] = {}
        self._broadcast_payload_size: int = 0

    def start_round(self, round_id: int) -> None:
        """Start a new round.

        Args:
            round_id: Round ID.
        """
        self._current_round = round_id
        self._round_start_time = time.time()
        self._network_stats = {}
        self._broadcast_payload_size = 0
        self._logger.info(f"Round {round_id} started")

    def broadcast_model(self, weights: Weights) -> float:
        """Broadcast global model to all clients.

        Args:
            weights: Model weights.

        Returns:
            float: Broadcast duration in seconds.
        """
        with Timer() as timer:
            serialized = self._serializer.serialize_weights(weights)
            self._broadcast_payload_size = len(serialized)
            msg = Message(
                msg_type=MsgType.MODEL_BROADCAST,
                payload={"weights": serialized, "round_id": self._current_round},
                round_id=self._current_round,
            )
            self._client_manager.broadcast_message(msg)

        self._logger.log_network(
            "broadcast", size=self._broadcast_payload_size, duration=timer.elapsed, success=True
        )
        return timer.elapsed

    def collect_updates(self, timeout: float) -> dict[int, Weights | None]:
        """Collect client updates with per-client timing and size logging.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Dict[int, Optional[Weights]]: Updates per client; None for timeouts.
        """
        start = time.time()
        raw = self._client_manager.collect_messages(timeout)
        results: dict[int, Weights | None] = {}

        for cid, msg in raw.items():
            elapsed = time.time() - start
            if msg is not None and msg.payload is not None:
                try:
                    payload_size = len(msg.payload["weights"]) if "weights" in msg.payload else 0
                    weights = self._serializer.deserialize_weights(msg.payload["weights"])
                    results[cid] = weights
                    # Store detailed network stats
                    self._network_stats[cid] = NetworkStats(
                        client_id=cid,
                        recv_size_bytes=payload_size,
                        recv_time_ms=elapsed * 1000,
                        success=True,
                    )
                    self._logger.log_network(
                        "recv", client_id=cid, size=payload_size, duration=elapsed, success=True
                    )
                except Exception:
                    results[cid] = None
                    self._network_stats[cid] = NetworkStats(
                        client_id=cid, recv_time_ms=elapsed * 1000, success=False
                    )
                    self._logger.log_network("recv", client_id=cid, duration=elapsed, success=False)
            else:
                results[cid] = None
                self._network_stats[cid] = NetworkStats(
                    client_id=cid, recv_time_ms=timeout * 1000, success=False
                )
                self._logger.log_network("recv", client_id=cid, duration=timeout, success=False)

        return results

    def end_round(self, aggregated_weights: Weights) -> RoundStats:
        """End the round and compute statistics.

        Args:
            aggregated_weights: Aggregated weights.

        Returns:
            RoundStats: Round statistics.
        """
        total_time = time.time() - self._round_start_time
        stats = RoundStats(
            round_id=self._current_round,
            broadcast_time=0.0,
            aggregate_time=0.0,
            total_time=total_time,
            participating_clients=list(self._client_manager.client_ids),
            timeout_clients=[],
        )
        self._logger.log_round(stats)
        return stats
