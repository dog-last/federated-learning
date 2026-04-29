"""Failure detector using heartbeat messages."""

import threading
from collections.abc import Callable

from src.core.interfaces import IFailureDetector
from src.utils.logger import FedLogger


class HeartbeatFailureDetector(IFailureDetector):
    """Detects node failures via heartbeat timeouts.

    Attributes:
        _running: Whether detection is active.
        _failed_nodes: Set of failed node IDs.
        _callbacks: Registered failure callbacks.
        _logger: Logger instance.
        _heartbeat_interval: Seconds between heartbeats.
        _heartbeat_timeout: Seconds before declaring failure.
    """

    def __init__(
        self,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 10.0,
        logger: FedLogger | None = None,
    ) -> None:
        self._running = False
        self._failed_nodes: set[int] = set()
        self._callbacks: list[Callable[[int], None]] = []
        self._logger = logger or FedLogger(
            name="FailureDetector", console_output=True, file_output=False
        )
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._last_heartbeat: dict[int, float] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start failure detection."""
        self._running = True

    def stop(self) -> None:
        """Stop failure detection."""
        self._running = False

    def heartbeat(self, target_id: int) -> bool:
        """Send heartbeat to target node.

        Args:
            target_id: Target node ID.

        Returns:
            bool: Whether a response was received.
        """
        # In a real implementation this would send a HEARTBEAT message
        # over the network. Here we just update the timestamp.
        import time

        with self._lock:
            self._last_heartbeat[target_id] = time.time()
        return True

    def check(self, node_id: int, timeout: float) -> bool:
        """Check if a node is alive.

        Args:
            node_id: Node ID.
            timeout: Timeout in seconds.

        Returns:
            bool: Whether the node is alive.
        """
        import time

        with self._lock:
            last = self._last_heartbeat.get(node_id, 0)
        return (time.time() - last) < timeout

    def get_failed_nodes(self) -> list[int]:
        """Get the list of failed nodes.

        Returns:
            List[int]: Failed node IDs.
        """
        with self._lock:
            return list(self._failed_nodes)

    def on_failure(self, callback: Callable[[int], None]) -> None:
        """Register a failure callback.

        Args:
            callback: Callback function taking a failed node ID.
        """
        self._callbacks.append(callback)

    def mark_failed(self, node_id: int) -> None:
        """Mark a node as failed and trigger callbacks.

        Args:
            node_id: Failed node ID.
        """
        with self._lock:
            if node_id in self._failed_nodes:
                return
            self._failed_nodes.add(node_id)

        self._logger.warning(f"Node-{node_id} detected as failed")
        for cb in self._callbacks:
            cb(node_id)

    def record_heartbeat(self, node_id: int) -> None:
        """Record a received heartbeat from a node.

        Args:
            node_id: Node ID that sent the heartbeat.
        """
        import time

        with self._lock:
            self._last_heartbeat[node_id] = time.time()
            self._failed_nodes.discard(node_id)
