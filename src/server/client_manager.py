"""Client connection manager for the federated server."""

import threading

from src.core.interfaces import IClientManager
from src.protocol.message import Message
from src.protocol.serializer import TorchSerializer
from src.transport.connection import Connection
from src.utils.logger import FedLogger


class ClientManager(IClientManager):
    """Manages client connections with thread-safe access.

    Attributes:
        _connections: Mapping of client_id -> Connection.
        _next_id: Next client ID to assign.
        _lock: Thread lock for safe access.
        _logger: Logger instance.
        _serializer: Weight serializer.
    """

    def __init__(self, logger: FedLogger | None = None) -> None:
        self._connections: dict[int, Connection] = {}
        self._next_id: int = 1
        self._lock = threading.Lock()
        self._logger = logger or FedLogger(
            name="ClientManager", console_output=True, file_output=False
        )
        self._serializer = TorchSerializer()

    def add_client(self, conn: Connection) -> int:
        """Add a client connection.

        Args:
            conn: Connection object.

        Returns:
            int: Assigned client ID.
        """
        with self._lock:
            cid = self._next_id
            self._next_id += 1
            self._connections[cid] = conn
        self._logger.info(f"Client-{cid} connected from {conn.remote_address}")
        return cid

    def remove_client(self, client_id: int) -> None:
        """Remove a client.

        Args:
            client_id: Client ID.
        """
        with self._lock:
            conn = self._connections.pop(client_id, None)
        if conn is not None:
            conn.close()
        self._logger.info(f"Client-{client_id} removed")

    def get_connection(self, client_id: int) -> Connection | None:
        """Get a client connection.

        Args:
            client_id: Client ID.

        Returns:
            Connection object, or None if not found.
        """
        return self._connections.get(client_id)

    def broadcast(self, data: bytes, exclude: list[int] | None = None) -> dict[int, bool]:
        """Broadcast data to all clients.

        Args:
            data: Data to broadcast.
            exclude: List of client IDs to exclude.

        Returns:
            Dict[int, bool]: Send status per client.
        """
        exclude = exclude or []
        results: dict[int, bool] = {}
        with self._lock:
            targets = {cid: conn for cid, conn in self._connections.items() if cid not in exclude}
        for cid, conn in targets.items():
            try:
                conn.send(data)
                results[cid] = True
            except Exception:
                results[cid] = False
        return results

    def broadcast_message(
        self, message: Message, exclude: list[int] | None = None
    ) -> dict[int, bool]:
        """Broadcast a Message object to all clients.

        Args:
            message: Message to broadcast.
            exclude: List of client IDs to exclude.

        Returns:
            Dict[int, bool]: Send status per client.
        """
        from src.protocol.codec import Codec

        codec = Codec()
        data = codec.encode(message)
        return self.broadcast(data, exclude)

    def collect(self, timeout: float) -> dict[int, bytes | None]:
        """Collect data from all clients.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Dict[int, Optional[bytes]]: Data per client; None for timeouts.
        """
        results: dict[int, bytes | None] = {}
        threads: list[threading.Thread] = []

        def _recv(cid: int, conn: Connection) -> None:
            try:
                results[cid] = conn.recv_all(timeout=timeout)
            except Exception:
                results[cid] = None

        with self._lock:
            targets = dict(self._connections)

        for cid, conn in targets.items():
            t = threading.Thread(target=_recv, args=(cid, conn))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=timeout + 1.0)

        return results

    def collect_messages(self, timeout: float) -> dict[int, Message | None]:
        """Collect Message objects from all clients.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Dict[int, Optional[Message]]: Message per client; None for timeouts.
        """
        from src.protocol.codec import Codec

        codec = Codec()
        raw = self.collect(timeout)
        results: dict[int, Message | None] = {}
        for cid, data in raw.items():
            if data is not None:
                try:
                    results[cid] = codec.decode(data)
                except Exception:
                    results[cid] = None
            else:
                results[cid] = None
        return results

    @property
    def client_ids(self) -> list[int]:
        """All client IDs.

        Returns:
            List[int]: List of client IDs.
        """
        with self._lock:
            return list(self._connections.keys())

    @property
    def num_clients(self) -> int:
        """Current number of clients.

        Returns:
            int: Client count.
        """
        with self._lock:
            return len(self._connections)
