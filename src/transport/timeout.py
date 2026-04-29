"""Timeout control utilities with zero-copy recv."""

import socket

from src.core.exceptions import NetworkTimeoutError

# Pre-allocated read buffer size increments for _recv_exact
_RECV_BUF_SIZE = 65536  # 64 KB


def set_socket_timeout(sock: socket.socket, timeout: float | None) -> None:
    """Set a socket timeout.

    Args:
        sock: Socket object.
        timeout: Timeout in seconds, or None for blocking.
    """
    sock.settimeout(timeout)


def send_with_timeout(sock: socket.socket, data: bytes, timeout: float) -> None:
    """Send data with a timeout.

    Args:
        sock: Socket object.
        data: Data to send.
        timeout: Timeout in seconds.

    Raises:
        NetworkTimeoutError: If the send operation times out.
    """
    sock.settimeout(timeout)
    try:
        sock.sendall(data)
    except TimeoutError:
        raise NetworkTimeoutError("send", timeout) from None


def recv_with_timeout(sock: socket.socket, size: int, timeout: float) -> bytes:
    """Receive data with a timeout.

    Args:
        sock: Socket object.
        size: Number of bytes to receive.
        timeout: Timeout in seconds.

    Returns:
        bytes: Received data.

    Raises:
        NetworkTimeoutError: If the recv operation times out.
    """
    sock.settimeout(timeout)
    try:
        return _recv_exact(sock, size)
    except TimeoutError:
        raise NetworkTimeoutError("recv", timeout) from None


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly *size* bytes from a socket using pre-allocated buffer.

    Uses a pre-allocated bytearray with memoryview slices to avoid
    repeated concatenation overhead and reduce memory copies.

    Args:
        sock: Socket object.
        size: Exact number of bytes to read.

    Returns:
        bytes: The received data of exactly *size* bytes.

    Raises:
        ConnectionError: If the peer closes the connection.
    """
    buf = bytearray(size)
    view = memoryview(buf)
    offset = 0
    while offset < size:
        chunk = sock.recv_into(view[offset:], min(size - offset, _RECV_BUF_SIZE))
        if chunk == 0:
            raise ConnectionError("Connection closed by peer")
        offset += chunk
    return bytes(buf)
