"""Fault recovery for ring topology."""

from src.p2p.failure_detector import HeartbeatFailureDetector
from src.p2p.topology import RingTopology
from src.utils.logger import FedLogger


class RecoveryManager:
    """Handles fault recovery by restructuring the ring topology.

    When a node failure is detected, this manager:
    1. Removes the failed node from the topology.
    2. Updates the ring so the predecessor skips to the successor.

    Attributes:
        _topology: Ring topology manager.
        _failure_detector: Failure detector.
        _logger: Logger instance.
    """

    def __init__(
        self,
        topology: RingTopology,
        failure_detector: HeartbeatFailureDetector,
        logger: FedLogger | None = None,
    ) -> None:
        self._topology = topology
        self._failure_detector = failure_detector
        self._logger = logger or FedLogger(name="Recovery", console_output=True, file_output=False)

        # Register callback
        self._failure_detector.on_failure(self._on_failure)

    def _on_failure(self, failed_id: int) -> None:
        """Handle a node failure event.

        Args:
            failed_id: Failed node ID.
        """
        self._logger.warning(f"Recovering from failure of Node-{failed_id}")
        self._topology.handle_failure(failed_id)
        self._logger.info(f"Topology restructured. New ring: {self._topology.ring_order}")
