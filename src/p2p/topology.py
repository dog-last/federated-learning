"""Ring topology manager."""

from src.core.interfaces import ITopologyManager


class RingTopology(ITopologyManager):
    """Manages a ring topology for P2P communication.

    Attributes:
        _ring: Ordered list of node IDs.
        _addresses: Mapping of node_id -> (host, port).
    """

    def __init__(self) -> None:
        self._ring: list[int] = []
        self._addresses: dict[int, tuple[str, int]] = {}

    def get_next_node(self, current_id: int) -> int | None:
        """Get the next node ID in the ring.

        Args:
            current_id: Current node ID.

        Returns:
            Optional[int]: Next node ID, or None if ring is broken.
        """
        if not self._ring or current_id not in self._ring:
            return None
        idx = self._ring.index(current_id)
        return self._ring[(idx + 1) % len(self._ring)]

    def handle_failure(self, failed_id: int) -> None:
        """Handle node failure and restructure topology.

        Args:
            failed_id: Failed node ID.
        """
        self.remove_node(failed_id)

    def add_node(self, node_id: int, address: tuple[str, int]) -> None:
        """Add a node to the topology.

        Args:
            node_id: Node ID.
            address: Node address (host, port).
        """
        if node_id not in self._ring:
            self._ring.append(node_id)
        self._addresses[node_id] = address

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the topology.

        Args:
            node_id: Node ID.
        """
        if node_id in self._ring:
            self._ring.remove(node_id)
        self._addresses.pop(node_id, None)

    def get_address(self, node_id: int) -> tuple[str, int] | None:
        """Get the address of a node.

        Args:
            node_id: Node ID.

        Returns:
            Optional[Tuple[str, int]]: Node address, or None.
        """
        return self._addresses.get(node_id)

    @property
    def ring_order(self) -> list[int]:
        """Ring order (list of node IDs).

        Returns:
            List[int]: Node IDs in ring order.
        """
        return list(self._ring)

    @property
    def size(self) -> int:
        """Number of nodes in the ring.

        Returns:
            int: Node count.
        """
        return len(self._ring)
