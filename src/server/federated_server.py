"""Federated learning server with TCP communication."""

import os
import threading

from torch.utils.data import DataLoader

from src.core.interfaces import IAggregator
from src.core.types import (
    EarlyStoppingConfig,
    MsgType,
    NetworkStats,
    RoundStats,
    ServerTimeouts,
)
from src.protocol.serializer import TorchSerializer
from src.server.aggregator import create_aggregator
from src.server.base import BaseServer
from src.server.checkpoint import save_checkpoint
from src.server.client_manager import ClientManager
from src.server.round_coordinator import RoundCoordinator
from src.transport.listener import Listener
from src.utils.early_stopping import EarlyStopping
from src.utils.logger import FedLogger
from src.utils.metrics import MetricsCollector
from src.utils.timer import Timer


class FederatedServer(BaseServer):
    """Federated learning server that orchestrates training rounds.

    Attributes:
        _listener: TCP listener.
        _client_manager: Client connection manager.
        _coordinator: Round coordinator.
        _aggregator: Aggregation strategy.
        _metrics: Metrics collector.
        _logger: Logger instance.
        _serializer: Weight serializer.
        _test_dataloader: Test set for global evaluation.
        _model: Global model for evaluation.
        _timeouts: Server timeout configuration.
        _early_stopping: Early stopping handler.
        _checkpoint_dir: Directory for saving checkpoints.
        _save_checkpoint_every: Save checkpoint every N rounds.
    """

    def __init__(
        self,
        aggregator: IAggregator | None = None,
        logger: FedLogger | None = None,
        timeouts: ServerTimeouts | None = None,
        early_stopping: EarlyStoppingConfig | None = None,
        checkpoint_dir: str = "./outputs/checkpoints",
        save_checkpoint_every: int = 5,
    ) -> None:
        super().__init__()
        self._listener = Listener()
        self._client_manager = ClientManager(logger=logger)
        self._aggregator = aggregator or create_aggregator("fedavg")
        self._coordinator = RoundCoordinator(self._client_manager, self._aggregator, logger=logger)
        self._metrics = MetricsCollector()
        self._logger = logger or FedLogger(name="Server")
        self._serializer = TorchSerializer()
        self._accept_thread: threading.Thread | None = None
        self._test_dataloader: DataLoader | None = None
        self._model = None
        self._timeouts = timeouts or ServerTimeouts()
        self._early_stopping = EarlyStopping(early_stopping or EarlyStoppingConfig())
        self._checkpoint_dir = checkpoint_dir
        self._save_checkpoint_every = save_checkpoint_every

    def set_test_dataloader(self, dataloader: DataLoader) -> None:
        """Set the test dataloader for global evaluation."""
        self._test_dataloader = dataloader

    def set_model(self, model: object) -> None:
        """Set the global model reference for evaluation."""
        self._model = model

    def start(self, port: int) -> None:
        """Start the server.

        Args:
            port: Listening port.
        """
        self._listener.bind("0.0.0.0", port)
        self._running = True
        self._logger.info(f"Server started on port {port}")

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        self._listener.close()
        for cid in self._client_manager.client_ids:
            self._client_manager.remove_client(cid)
        self._logger.info("Server stopped")

    def wait_for_clients(self, num_clients: int, timeout: float = 60.0) -> list[int]:
        """Wait for clients to connect.

        Args:
            num_clients: Expected number of clients.
            timeout: Wait timeout in seconds.

        Returns:
            List[int]: List of connected client IDs.
        """
        self._logger.info(f"Waiting for {num_clients} clients...")

        while self._client_manager.num_clients < num_clients and self._running:
            try:
                conn = self._listener.accept(timeout=5.0)
                cid = self._client_manager.add_client(conn)
                # Read and discard the CLIENT_REGISTER message from the client
                _ = conn.recv_message(timeout=10.0)
                ack = _make_ack(cid)
                conn.send_message(ack)
            except Exception:
                continue

        ids = self._client_manager.client_ids
        self._logger.info(f"All {len(ids)} clients connected: {ids}")
        return ids

    def _evaluate_global(self) -> tuple[float, float]:
        """Evaluate global model on test set.

        Returns:
            Tuple of (accuracy, loss).
        """
        if self._model is None or self._test_dataloader is None:
            return 0.0, 0.0
        self._model.set_weights(self._global_weights)
        result = self._model.evaluate(self._test_dataloader)
        return result.accuracy, result.loss

    def _save_checkpoint(self, round_id: int) -> None:
        """Save model checkpoint.

        Args:
            round_id: Current round ID.
        """
        try:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(self._checkpoint_dir, f"round_{round_id}.pt")
            save_checkpoint(self._global_weights, checkpoint_path, round_id)
            self._logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self._logger.error(f"Failed to save checkpoint: {e}")

    @staticmethod
    def _format_size(size: float) -> str:
        """Format a byte count for human reading.

        Args:
            size: Size in bytes.

        Returns:
            str: Formatted size string.
        """
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def run_round(self, round_id: int) -> RoundStats:
        """Execute one round of federated learning.

        Args:
            round_id: Round ID.

        Returns:
            RoundStats: Round statistics.
        """
        self._coordinator.start_round(round_id)

        # Broadcast
        with Timer() as broadcast_timer:
            self._coordinator.broadcast_model(self._global_weights)

        # Collect updates
        with Timer() as collect_timer:
            updates = self._coordinator.collect_updates(timeout=self._timeouts.recv)

        participating = [cid for cid, w in updates.items() if w is not None]
        timeout_clients = [cid for cid, w in updates.items() if w is None]

        # Aggregate
        with Timer() as agg_timer:
            weights_list = [updates[cid] for cid in participating if updates[cid] is not None]
            if weights_list:
                self._global_weights = self._aggregator.aggregate(weights_list)

        # Evaluate global model
        global_accuracy, global_loss = self._evaluate_global()

        # Build network stats from coordinator
        network_stats = self._coordinator._network_stats.copy()
        for cid in participating:
            if cid not in network_stats:
                network_stats[cid] = NetworkStats(client_id=cid, success=True)
        for cid in timeout_clients:
            if cid not in network_stats:
                network_stats[cid] = NetworkStats(client_id=cid, success=False)

        stats = RoundStats(
            round_id=round_id,
            broadcast_time=broadcast_timer.elapsed,
            aggregate_time=agg_timer.elapsed,
            total_time=broadcast_timer.elapsed + collect_timer.elapsed + agg_timer.elapsed,
            participating_clients=participating,
            timeout_clients=timeout_clients,
            global_accuracy=global_accuracy,
            global_loss=global_loss,
            network_stats=network_stats,
            broadcast_payload_size=self._coordinator._broadcast_payload_size,
        )
        self._metrics.record_round(stats)
        self._logger.log_round(stats)

        # Log detailed network status for each client
        self._logger.info(f"[Round {round_id}] Network Status:")
        self._logger.info(
            f"  - Broadcast payload size: {self._format_size(stats.broadcast_payload_size)}"
        )
        self._logger.info(f"  - Broadcast time: {stats.broadcast_time:.3f}s")
        for cid in sorted(network_stats.keys()):
            ns = network_stats[cid]
            status = "SUCCESS" if ns.success else "TIMEOUT"
            size_str = (
                f", Size: {self._format_size(ns.recv_size_bytes)}" if ns.recv_size_bytes > 0 else ""
            )
            self._logger.info(
                f"  - Client-{cid} update: {status}, Wait time: {ns.recv_time_ms:.1f}ms{size_str}"
            )

        # Update early stopping
        if self._early_stopping._config.enabled:
            metric_value = (
                global_accuracy
                if self._early_stopping._config.monitor == "accuracy"
                else global_loss
            )
            if self._early_stopping(metric_value):
                self._logger.info(
                    f"Early stopping condition met. No improvement for {self._early_stopping.counter} rounds."
                )

        # Save checkpoint periodically
        if round_id % self._save_checkpoint_every == 0:
            self._save_checkpoint(round_id)

        return stats

    def run(self, num_rounds: int) -> None:
        """Run the server main loop.

        Args:
            num_rounds: Total number of rounds.
        """
        self._logger.info(f"Starting {num_rounds} rounds of federated learning")

        for round_id in range(1, num_rounds + 1):
            if not self._running:
                break
            self.run_round(round_id)

            # Check early stopping
            if self._early_stopping.should_stop:
                self._logger.info(
                    f"Early stopping triggered at round {round_id}. "
                    f"Best {self._early_stopping._config.monitor}: {self._early_stopping.best_value:.4f}"
                )
                break

        # Final evaluation
        final_acc, final_loss = self._evaluate_global()
        self._logger.info(
            f"Training completed. Final accuracy: {final_acc:.2f}%, Loss: {final_loss:.4f}"
        )

        # Export metrics and plots
        self._metrics.export("outputs/metrics.json")
        self._metrics.plot_accuracy("outputs/figures/accuracy.png")
        self._metrics.plot_loss("outputs/figures/loss.png")


def _make_ack(client_id: int):
    """Create a CLIENT_ACK message."""
    from src.protocol.message import Message

    return Message(
        msg_type=MsgType.CLIENT_ACK,
        payload={"client_id": client_id, "config": {"epochs": 2, "lr": 0.01, "batch_size": 32}},
    )
