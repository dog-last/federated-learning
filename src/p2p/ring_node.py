"""Ring node for decentralized federated learning."""

import os
import socket
import threading
import time

from torch.utils.data import DataLoader

from src.core.interfaces import IModel, IP2PNode
from src.core.types import EarlyStoppingConfig, MsgType, NetworkStats, Weights
from src.p2p.failure_detector import HeartbeatFailureDetector
from src.p2p.recovery import RecoveryManager
from src.p2p.topology import RingTopology
from src.protocol.codec import Codec
from src.protocol.message import Message
from src.protocol.serializer import TorchSerializer
from src.server.checkpoint import save_checkpoint
from src.transport.connection import Connection
from src.transport.listener import Listener
from src.utils.early_stopping import EarlyStopping
from src.utils.logger import FedLogger
from src.utils.timer import Timer


class RingNode(IP2PNode):
    """P2P node in a ring topology for decentralized federated learning.

    Attributes:
        _node_id: This node's ID.
        _topology: Ring topology manager.
        _failure_detector: Failure detector.
        _recovery: Recovery manager.
        _model: Local model.
        _dataloader: Local training data.
        _test_dataloader: Test set for evaluation.
        _logger: Logger instance.
        _serializer: Weight serializer.
        _listener: TCP listener.
        _running: Whether the node is active.
        _accept_thread: Background thread for handling connections.
        _early_stopping: Early stopping handler.
        _checkpoint_dir: Directory for saving checkpoints.
        _save_checkpoint_every: Save checkpoint every N rounds.
        _network_stats: Network statistics for current round.
        _current_round: Current round ID.
    """

    def __init__(
        self,
        node_id: int,
        model: IModel,
        dataloader: object,
        logger: FedLogger | None = None,
        early_stopping: EarlyStoppingConfig | None = None,
        checkpoint_dir: str = "./outputs/checkpoints",
        save_checkpoint_every: int = 5,
    ) -> None:
        self._node_id = node_id
        self._topology = RingTopology()
        self._failure_detector = HeartbeatFailureDetector()
        self._recovery = RecoveryManager(self._topology, self._failure_detector)
        self._model = model
        self._dataloader = dataloader
        self._test_dataloader: DataLoader | None = None
        self._logger = logger or FedLogger(
            name=f"Node-{node_id}", console_output=True, file_output=False
        )
        self._serializer = TorchSerializer()
        self._listener = Listener()
        self._running = False
        self._accept_thread: threading.Thread | None = None
        self._early_stopping = EarlyStopping(early_stopping or EarlyStoppingConfig())
        self._checkpoint_dir = checkpoint_dir
        self._save_checkpoint_every = save_checkpoint_every
        self._network_stats: dict[int, NetworkStats] = {}
        self._current_round: int = 0

    def start(self, port: int) -> None:
        """Start the node.

        Args:
            port: Listening port (0 for auto-assign).
        """
        self._listener.bind("0.0.0.0", port)
        actual_port = self._listener.address[1]
        self._topology.add_node(self._node_id, ("127.0.0.1", actual_port))
        self._failure_detector.start()
        self._running = True
        self._logger.info(f"Node-{self._node_id} started on port {actual_port}")

        # Start background thread to handle incoming connections
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def stop(self) -> None:
        """Stop the node."""
        self._running = False
        self._failure_detector.stop()
        # Close listener first to unblock accept()
        self._listener.close()
        # Wait for accept thread with a longer timeout
        if self._accept_thread is not None and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=3.0)
        self._logger.info(f"Node-{self._node_id} stopped")

    def set_test_dataloader(self, dataloader: DataLoader) -> None:
        """Set the test dataloader for evaluation.

        Args:
            dataloader: Test data loader.
        """
        self._test_dataloader = dataloader

    def _evaluate(self) -> tuple[float, float]:
        """Evaluate model on test set.

        Returns:
            Tuple of (accuracy, loss).
        """
        if self._model is None or self._test_dataloader is None:
            return 0.0, 0.0
        result = self._model.evaluate(self._test_dataloader)
        return result.accuracy, result.loss

    def _accept_loop(self) -> None:
        """Background loop to accept and handle incoming connections."""
        while self._running:
            try:
                conn = self._listener.accept(timeout=0.5)
                # Handle each connection in a separate thread
                handler_thread = threading.Thread(
                    target=self._handle_connection, args=(conn,), daemon=True
                )
                handler_thread.start()
            except Exception:
                # Timeout or error, continue loop if still running
                if not self._running:
                    break
                continue

    def _handle_connection(self, conn: Connection) -> None:
        """Handle an incoming connection.

        Args:
            conn: The incoming connection.
        """
        try:
            msg = conn.recv_message(timeout=10.0)
            if msg.msg_type == MsgType.NODE_JOIN:
                self._handle_node_join(conn, msg)
            elif msg.msg_type == MsgType.NODE_LEAVE:
                self._handle_node_leave(conn, msg)
            elif msg.msg_type == MsgType.RING_PASS:
                self._handle_ring_pass(conn, msg)
            else:
                self._logger.debug(f"Unknown message type: {msg.msg_type}")
        except Exception as e:
            self._logger.debug(f"Error handling connection: {e}")
        finally:
            conn.close()

    def _handle_node_join(self, conn: Connection, msg: Message) -> None:
        """Handle a node join request.

        Args:
            conn: The connection to respond on.
            msg: The join message.
        """
        new_node_id = msg.payload.get("node_id")
        new_address = msg.payload.get("address")

        if new_node_id is not None and new_address is not None:
            self._topology.add_node(new_node_id, tuple(new_address))
            self._logger.info(f"Node {new_node_id} joined the ring")

        # Send back the current topology
        node_addresses = {}
        for nid in self._topology.ring_order:
            addr = self._topology.get_address(nid)
            if addr is not None:
                node_addresses[nid] = list(addr)

        resp = Message(
            msg_type=MsgType.NODE_ACK,
            client_id=self._node_id,
            payload={
                "ring_order": self._topology.ring_order,
                "node_addresses": node_addresses,
            },
        )
        conn.send_message(resp)

    def _handle_node_leave(self, conn: Connection, msg: Message) -> None:
        """Handle a node leave notification.

        Args:
            conn: The connection.
            msg: The leave message.
        """
        node_id = msg.payload.get("node_id")
        if node_id is not None:
            self._topology.remove_node(node_id)
            self._logger.info(f"Node {node_id} left the ring")

    def _handle_ring_pass(self, conn: Connection, msg: Message) -> None:
        """Handle a model pass in the ring.

        Args:
            conn: The connection.
            msg: The ring pass message.
        """
        # For now, just acknowledge receipt
        # In a full implementation, this would trigger local aggregation
        self._logger.debug(f"Received model from node {msg.payload.get('origin_id')}")

    def join_ring(self, bootstrap_node: tuple[str, int]) -> None:
        """Join a ring network via a bootstrap node.

        Args:
            bootstrap_node: Bootstrap node address (host, port).
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(bootstrap_node)
        conn = Connection(sock, Codec())

        msg = Message(
            msg_type=MsgType.NODE_JOIN,
            client_id=self._node_id,
            payload={"node_id": self._node_id, "address": ("127.0.0.1", self._listener.address[1])},
        )
        conn.send_message(msg)

        # Receive topology info
        resp = conn.recv_message(timeout=10.0)
        if resp.payload and "ring_order" in resp.payload:
            for nid, addr in resp.payload.get("node_addresses", {}).items():
                self._topology.add_node(int(nid), tuple(addr))
            self._logger.info(f"Joined ring: {self._topology.ring_order}")

        conn.close()

    def leave_ring(self) -> None:
        """Leave the ring network."""
        next_addr = self.next_node
        if next_addr is not None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(next_addr)
                conn = Connection(sock, Codec())
                msg = Message(
                    msg_type=MsgType.NODE_LEAVE,
                    client_id=self._node_id,
                    payload={"node_id": self._node_id, "reason": "voluntary"},
                )
                conn.send_message(msg)
                conn.close()
            except Exception:
                pass

        self._topology.remove_node(self._node_id)
        self._logger.info(f"Node-{self._node_id} left the ring")

    def pass_model(self, weights: Weights) -> tuple[bool, int, float]:
        """Pass model to next node in the ring with network monitoring.

        Args:
            weights: Model weights.

        Returns:
            Tuple[bool, int, float]: (success, payload_size_bytes, duration_ms)
        """
        next_addr = self.next_node
        next_id = self._topology.get_next_node(self._node_id)
        if next_addr is None:
            self._logger.warning("No next node available")
            return False, 0, 0.0

        start_time = time.time()
        payload_size = 0
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30.0)
            sock.connect(next_addr)
            conn = Connection(sock, Codec())

            serialized = self._serializer.serialize_weights(weights)
            payload_size = len(serialized)
            msg = Message(
                msg_type=MsgType.RING_PASS,
                client_id=self._node_id,
                payload={
                    "weights": serialized,
                    "origin_id": self._node_id,
                    "round_id": self._current_round,
                },
            )
            conn.send_message(msg)
            conn.close()

            elapsed_ms = (time.time() - start_time) * 1000
            self._logger.log_network(
                "ring_pass_send",
                client_id=next_id,
                size=payload_size,
                duration=elapsed_ms / 1000,
                success=True,
            )
            return True, payload_size, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self._logger.log_network(
                "ring_pass_send",
                client_id=next_id,
                size=payload_size,
                duration=elapsed_ms / 1000,
                success=False,
            )
            self._logger.warning(f"Failed to pass model to Node-{next_id}: {e}")
            return False, payload_size, elapsed_ms

    def run(self, num_rounds: int) -> None:
        """Run the node main loop with network monitoring.

        Args:
            num_rounds: Number of rounds to participate in.
        """
        for round_id in range(1, num_rounds + 1):
            if not self._running:
                break

            self._current_round = round_id
            self._network_stats = {}
            round_start_time = time.time()

            self._logger.info("=" * 60)
            self._logger.info(f"[Round {round_id}/{num_rounds}] Node-{self._node_id}")
            self._logger.info("=" * 60)

            # Local training
            with Timer() as timer:
                result = self._model.train_epoch(self._dataloader, lr=0.01)
            train_time = timer.elapsed

            self._logger.log_training(
                self._node_id, round_id, result.loss, result.accuracy, train_time
            )

            # Pass model to next node with network monitoring
            weights = self._model.get_weights()
            pass_success, payload_size, pass_time_ms = self.pass_model(weights)

            # Store network stats
            next_id = self._topology.get_next_node(self._node_id)
            if next_id is not None:
                self._network_stats[next_id] = NetworkStats(
                    client_id=next_id,
                    send_size_bytes=payload_size,
                    send_time_ms=pass_time_ms,
                    success=pass_success,
                )

            # Log network status
            self._logger.info(f"[Round {round_id}] Network Status:")
            self._logger.info(f"  - Ring topology: {self._topology.ring_order}")
            self._logger.info(f"  - Next node: Node-{next_id}")
            self._logger.info(f"  - Model pass: {'SUCCESS' if pass_success else 'FAILED'}")
            self._logger.info(f"  - Payload size: {self._format_size(payload_size)}")
            self._logger.info(f"  - Pass time: {pass_time_ms:.1f}ms")
            self._logger.info(f"  - Training time: {train_time:.3f}s")

            # Evaluate on test set if available
            if self._test_dataloader is not None:
                test_acc, test_loss = self._evaluate()
                self._logger.info(
                    f"[Round {round_id}] Test - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}"
                )

                # Update early stopping
                if self._early_stopping._config.enabled:
                    metric_value = (
                        test_acc
                        if self._early_stopping._config.monitor == "accuracy"
                        else test_loss
                    )
                    if self._early_stopping(metric_value):
                        self._logger.info(
                            f"Early stopping triggered at round {round_id}. "
                            f"Best {self._early_stopping._config.monitor}: {self._early_stopping.best_value:.4f}"
                        )
                        break

            # Save checkpoint periodically
            if round_id % self._save_checkpoint_every == 0:
                self._save_checkpoint(round_id)

            round_total_time = time.time() - round_start_time
            self._logger.info(f"[Round {round_id}] Total time: {round_total_time:.3f}s")
            self._logger.info("=" * 60)

        self._logger.info("All rounds completed")

    def _save_checkpoint(self, round_id: int) -> None:
        """Save model checkpoint.

        Args:
            round_id: Current round ID.
        """
        try:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self._checkpoint_dir, f"node_{self._node_id}_round_{round_id}.pt"
            )
            weights = self._model.get_weights()
            save_checkpoint(weights, checkpoint_path, round_id)
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

    @property
    def node_id(self) -> int:
        """Node ID.

        Returns:
            int: The node ID.
        """
        return self._node_id

    @property
    def next_node(self) -> tuple[str, int] | None:
        """Next node address.

        Returns:
            Optional[Tuple[str, int]]: Next node address, or None.
        """
        next_id = self._topology.get_next_node(self._node_id)
        if next_id is not None:
            return self._topology.get_address(next_id)
        return None

    @property
    def prev_node(self) -> tuple[str, int] | None:
        """Previous node address.

        Returns:
            Optional[Tuple[str, int]]: Previous node address, or None.
        """
        ring = self._topology.ring_order
        if self._node_id not in ring or len(ring) < 2:
            return None
        idx = ring.index(self._node_id)
        prev_id = ring[(idx - 1) % len(ring)]
        return self._topology.get_address(prev_id)
