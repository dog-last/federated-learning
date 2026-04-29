"""Unit tests for transport: connection, listener, timeout."""

import socket
import threading

import pytest

from src.core.exceptions import NetworkTimeoutError
from src.core.types import MsgType
from src.protocol.message import Message
from src.transport.connection import Connection
from src.transport.listener import Listener
from src.transport.timeout import recv_with_timeout, send_with_timeout, set_socket_timeout


class TestListener:
    """Tests for Listener."""

    def test_bind_and_address(self) -> None:
        listener = Listener()
        listener.bind("127.0.0.1", 0)
        addr = listener.address
        assert addr[0] == "127.0.0.1"
        assert addr[1] > 0
        assert listener.is_listening
        listener.close()
        assert not listener.is_listening

    def test_accept_connection(self) -> None:
        listener = Listener()
        listener.bind("127.0.0.1", 0)
        _, port = listener.address

        accepted = []

        def accept_thread() -> None:
            conn = listener.accept(timeout=5.0)
            accepted.append(conn)

        t = threading.Thread(target=accept_thread)
        t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))

        t.join(timeout=5.0)
        assert len(accepted) == 1
        assert accepted[0].is_connected

        sock.close()
        accepted[0].close()
        listener.close()

    def test_accept_timeout(self) -> None:
        listener = Listener()
        listener.bind("127.0.0.1", 0)
        with pytest.raises(NetworkTimeoutError):
            listener.accept(timeout=0.1)
        listener.close()


class TestConnection:
    """Tests for Connection."""

    def _make_pair(self) -> tuple[Connection, Connection]:
        """Create a connected pair of sockets wrapped in Connection."""
        listener = Listener()
        listener.bind("127.0.0.1", 0)
        _, port = listener.address

        result = []

        def accept() -> None:
            result.append(listener.accept())

        t = threading.Thread(target=accept)
        t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))

        t.join(timeout=5.0)
        server_conn = result[0]
        client_conn = Connection(sock)
        listener.close()
        return client_conn, server_conn

    def test_send_recv_message(self) -> None:
        client, server = self._make_pair()
        msg = Message(msg_type=MsgType.CLIENT_REGISTER, payload={"data_size": 100})
        client.send_message(msg)

        received = server.recv_message(timeout=5.0)
        assert received.msg_type == MsgType.CLIENT_REGISTER
        assert received.payload["data_size"] == 100

        client.close()
        server.close()

    def test_is_connected(self) -> None:
        client, server = self._make_pair()
        assert client.is_connected
        assert server.is_connected
        client.close()
        assert not client.is_connected
        server.close()

    def test_remote_address(self) -> None:
        client, server = self._make_pair()
        addr = server.remote_address
        assert isinstance(addr, tuple)
        assert len(addr) == 2
        client.close()
        server.close()

    def test_recv_all(self) -> None:
        client, server = self._make_pair()
        msg = Message(msg_type=MsgType.HEARTBEAT, payload="ping")
        client.send_message(msg)

        data = server.recv_all(timeout=5.0)
        assert len(data) > 0

        client.close()
        server.close()

    def test_send_on_closed_connection(self) -> None:
        from src.core.exceptions import ConnectionError

        client, server = self._make_pair()
        client.close()
        with pytest.raises(ConnectionError):
            client.send(b"data")
        server.close()

    def test_recv_on_closed_connection(self) -> None:
        from src.core.exceptions import ConnectionError

        client, server = self._make_pair()
        client.close()
        with pytest.raises(ConnectionError):
            client.recv(10)
        server.close()

    def test_send_message_on_closed_connection(self) -> None:
        from src.core.exceptions import ConnectionError

        client, server = self._make_pair()
        client.close()
        with pytest.raises(ConnectionError):
            client.send_message(Message(msg_type=MsgType.ERROR))
        server.close()

    def test_recv_message_on_closed_connection(self) -> None:
        from src.core.exceptions import ConnectionError

        client, server = self._make_pair()
        client.close()
        with pytest.raises(ConnectionError):
            client.recv_message()
        server.close()

    def test_double_close(self) -> None:
        client, server = self._make_pair()
        client.close()
        client.close()  # should not raise
        server.close()

    def test_recv_timeout(self) -> None:
        client, server = self._make_pair()
        with pytest.raises(NetworkTimeoutError):
            server.recv(10, timeout=0.1)
        client.close()
        server.close()


class TestTimeout:
    """Tests for timeout utilities."""

    def test_set_socket_timeout(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        set_socket_timeout(sock, 5.0)
        assert sock.gettimeout() == 5.0
        set_socket_timeout(sock, None)
        assert sock.gettimeout() is None
        sock.close()

    def test_send_with_timeout(self) -> None:
        listener = Listener()
        listener.bind("127.0.0.1", 0)
        _, port = listener.address

        conn_result = []

        def accept() -> None:
            conn_result.append(listener.accept())

        t = threading.Thread(target=accept)
        t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        t.join(timeout=5.0)

        send_with_timeout(sock, b"hello", 5.0)
        data = conn_result[0].recv(5, timeout=5.0)
        assert data == b"hello"

        sock.close()
        conn_result[0].close()
        listener.close()

    def test_recv_with_timeout(self) -> None:
        listener = Listener()
        listener.bind("127.0.0.1", 0)
        _, port = listener.address

        conn_result = []

        def accept() -> None:
            conn_result.append(listener.accept())

        t = threading.Thread(target=accept)
        t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        t.join(timeout=5.0)

        server_conn = conn_result[0]
        server_conn.send(b"world")
        data = recv_with_timeout(sock, 5, 5.0)
        assert data == b"world"

        sock.close()
        server_conn.close()
        listener.close()
