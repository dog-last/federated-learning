"""TCP listener for accepting incoming connections."""

import contextlib
import socket

from src.core.exceptions import NetworkTimeoutError
from src.core.interfaces import IConnection, IListener
from src.transport.connection import Connection

_BACKLOG = 128


class Listener(IListener):
    """TCP listener that accepts connections and wraps them in Connection objects.

    Applies TCP optimizations (NODELAY, buffer sizes) to all accepted sockets.

    Attributes:
        _sock: Server socket.
        _listening: Whether the listener is active.
        _address: Bound address.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listening = False
        self._address: tuple[str, int] = ("", 0)

    def bind(self, host: str, port: int) -> None:
        """Bind to an address.

        Args:
            host: Host address.
            port: Port number (0 for auto-assign).
        """
        self._sock.bind((host, port))
        self._sock.listen(_BACKLOG)
        self._address = self._sock.getsockname()
        self._listening = True

    def accept(self, timeout: float | None = None) -> IConnection:
        """Accept a connection.

        Args:
            timeout: Timeout in seconds.

        Returns:
            IConnection: Connection object.

        Raises:
            NetworkTimeoutError: Accept timeout.
        """
        if timeout is not None:
            self._sock.settimeout(timeout)
        try:
            client_sock, _ = self._sock.accept()
            return Connection(client_sock)
        except TimeoutError:
            raise NetworkTimeoutError("accept", timeout or 0) from None

    def close(self) -> None:
        """Close the listener."""
        if self._listening:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._listening = False

    @property
    def is_listening(self) -> bool:
        """Whether the listener is active.

        Returns:
            bool: Listening status.
        """
        return self._listening

    @property
    def address(self) -> tuple[str, int]:
        """Listening address.

        Returns:
            Tuple[str, int]: (host, port) of the listener.
        """
        return self._address
