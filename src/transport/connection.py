"""TCP connection wrapper with message-level framing and TCP optimizations."""

import contextlib
import socket

from src.core.exceptions import ConnectionError, NetworkTimeoutError
from src.core.interfaces import IConnection
from src.protocol.codec import Codec
from src.protocol.constants import LENGTH_PREFIX_SIZE
from src.protocol.message import Message
from src.transport.timeout import _recv_exact

# TCP tuning constants
_TCP_NODELAY = True
_RECV_BUF = 256 * 1024  # 256 KB socket receive buffer
_SEND_BUF = 256 * 1024  # 256 KB socket send buffer


def _optimize_socket(sock: socket.socket) -> None:
    """Apply TCP performance optimizations to a socket.

    Args:
        sock: Socket to optimize.
    """
    if _TCP_NODELAY:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUF)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SEND_BUF)
    except OSError:
        pass


class Connection(IConnection):
    """TCP connection with length-prefix message framing and zero-copy I/O.

    Attributes:
        _sock: Underlying socket.
        _codec: Codec for encode/decode.
        _connected: Whether the connection is active.
    """

    def __init__(self, sock: socket.socket, codec: Codec | None = None) -> None:
        _optimize_socket(sock)
        self._sock = sock
        self._codec = codec or Codec()
        self._connected = True

    def send_message(self, message: Message) -> None:
        """Send a Message object over the wire.

        Args:
            message: Message to send.

        Raises:
            ConnectionError: If the connection is closed.
        """
        if not self._connected:
            raise ConnectionError("Connection is closed")
        data = self._codec.encode(message)
        self._sock.sendall(data)

    def recv_message(self, timeout: float | None = None) -> Message:
        """Receive a Message object from the wire.

        Args:
            timeout: Receive timeout in seconds.

        Returns:
            Message: Received message.

        Raises:
            ConnectionError: If the connection is closed.
            NetworkTimeoutError: If the receive times out.
        """
        if not self._connected:
            raise ConnectionError("Connection is closed")
        if timeout is not None:
            self._sock.settimeout(timeout)

        length_data = _recv_exact(self._sock, LENGTH_PREFIX_SIZE)
        length = self._codec.get_message_length(length_data)
        body = _recv_exact(self._sock, length)
        return self._codec.decode(body)

    # -- IConnection methods (raw bytes) --

    def send(self, data: bytes) -> None:
        """Send raw bytes.

        Args:
            data: Byte stream to send.
        """
        if not self._connected:
            raise ConnectionError("Connection is closed")
        self._sock.sendall(data)

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
        if timeout is not None:
            self._sock.settimeout(timeout)
        try:
            return _recv_exact(self._sock, size)
        except TimeoutError:
            raise NetworkTimeoutError("recv", timeout or 0) from None
        except OSError:
            self._connected = False
            raise ConnectionError("Connection is closed") from None

    def recv_all(self, timeout: float | None = None) -> bytes:
        """Receive a complete framed message.

        Reads 8-byte length prefix, then the full body.

        Args:
            timeout: Timeout in seconds.

        Returns:
            bytes: Complete message body.
        """
        if timeout is not None:
            self._sock.settimeout(timeout)
        length_data = _recv_exact(self._sock, LENGTH_PREFIX_SIZE)
        length = Codec.get_message_length(length_data)
        return _recv_exact(self._sock, length)

    def close(self) -> None:
        """Close the connection."""
        if self._connected:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the connection is active.

        Returns:
            bool: Connection status.
        """
        return self._connected

    @property
    def remote_address(self) -> tuple[str, int]:
        """Remote address.

        Returns:
            Tuple[str, int]: (host, port) of the remote endpoint.
        """
        return self._sock.getpeername()

    @property
    def socket(self) -> socket.socket:
        """Underlying socket (for advanced use).

        Returns:
            socket.socket: The raw socket.
        """
        return self._sock
