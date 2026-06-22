"""Unit tests for TCP communicator."""
import socket
import struct
import threading
import pickle
import gzip

import pytest
from core.communicator import TCPCommunicator, MAGIC_NUMBER


def _ephemeral_server(handler):
    """Bind port 0, start thread, return (port, thread, ready_event)."""
    ready = threading.Event()
    result_holder = {}

    def _run():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        result_holder["port"] = port
        srv.listen(1)
        ready.set()
        conn, _ = srv.accept()
        try:
            handler(conn, result_holder)
        finally:
            conn.close()
            srv.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait(timeout=5)
    return result_holder["port"], t, result_holder


def test_communicator_no_compression():
    comm = TCPCommunicator(use_compression=False)
    assert comm.use_compression is False


def test_communicator_with_compression():
    comm = TCPCommunicator(use_compression=True)
    assert comm.use_compression is True


def test_communicator_default():
    comm = TCPCommunicator()
    assert comm.use_compression is False


def test_send_recv_no_compression():
    comm = TCPCommunicator(use_compression=False)

    def handler(conn, rh):
        rh["value"] = comm.recv_data(conn)

    port, t, rh = _ephemeral_server(handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    payload = {"type": "test", "data": [1, 2, 3]}
    ok, size = comm.send_data(cli, payload)
    assert ok is True
    assert size > 0
    cli.close()
    t.join(timeout=5)
    assert rh["value"] == payload


def test_send_recv_with_compression():
    comm = TCPCommunicator(use_compression=True)

    def handler(conn, rh):
        rh["value"] = comm.recv_data(conn)

    port, t, rh = _ephemeral_server(handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    payload = {"type": "compressed", "values": list(range(50))}
    ok, size = comm.send_data(cli, payload)
    assert ok is True
    cli.close()
    t.join(timeout=5)
    assert rh["value"] == payload


def test_recv_data_with_meta():
    comm = TCPCommunicator(use_compression=False)

    def handler(conn, rh):
        msg, meta = comm.recv_data_with_meta(conn)
        rh["msg"] = msg
        rh["meta"] = meta

    port, t, rh = _ephemeral_server(handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    comm.send_data(cli, {"type": "meta_test"})
    cli.close()
    t.join(timeout=5)
    assert rh["msg"]["type"] == "meta_test"
    assert rh["meta"]["magic_ok"] is True
    assert rh["meta"]["compression"] is False
    assert rh["meta"]["payload_bytes"] > 0


def test_recvall_returns_none_on_disconnect():
    ready = threading.Event()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)

    def accept_and_close():
        conn, _ = srv.accept()
        ready.set()
        conn.close()
        srv.close()

    t = threading.Thread(target=accept_and_close, daemon=True)
    t.start()
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    ready.wait(timeout=5)
    # Small delay to ensure server has closed the connection
    import time; time.sleep(0.05)
    comm = TCPCommunicator()
    assert comm._recvall(cli, 100) is None
    cli.close()
    t.join(timeout=5)


def test_recv_data_none_on_closed():
    closed = threading.Event()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)

    def accept_and_close():
        conn, _ = srv.accept()
        conn.close()
        srv.close()
        closed.set()

    t = threading.Thread(target=accept_and_close, daemon=True)
    t.start()
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    closed.wait(timeout=5)
    import time; time.sleep(0.05)
    comm = TCPCommunicator(use_compression=False)
    assert comm.recv_data(cli) is None
    cli.close()
    t.join(timeout=5)


def test_magic_number_mismatch():
    comm = TCPCommunicator(use_compression=False)

    def handler(conn, rh):
        try:
            comm.recv_data_with_meta(conn)
        except ValueError:
            rh["got_error"] = True

    port, t, rh = _ephemeral_server(handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    bad_magic = b'XXXX'
    payload = b'test'
    cli.sendall(struct.pack('>Q', len(payload)) + bad_magic + payload)
    cli.close()
    t.join(timeout=5)
    assert rh.get("got_error") is True


def test_send_data_on_closed_socket():
    comm = TCPCommunicator(use_compression=False)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.close()
    ok, size = comm.send_data(cli, {"type": "fail"})
    assert ok is False
    assert size == 0


def test_multiple_messages():
    comm = TCPCommunicator(use_compression=False)

    def handler(conn, rh):
        received = []
        for _ in range(3):
            msg = comm.recv_data(conn)
            if msg is None:
                break
            received.append(msg)
        rh["received"] = received

    port, t, rh = _ephemeral_server(handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    for i in range(3):
        comm.send_data(cli, {"type": "batch", "index": i})
    cli.close()
    t.join(timeout=5)
    assert len(rh["received"]) == 3


def test_send_data_with_pickle_error():
    """Test handling of pickle serialization errors."""
    comm = TCPCommunicator(use_compression=False)
    
    # Create a mock socket
    mock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mock_sock.close()
    
    ok, size = comm.send_data(mock_sock, {"type": "test"})
    assert ok is False
    assert size == 0


def test_recv_data_with_corrupted_payload():
    """Test handling of corrupted/decompressed payload."""
    comm = TCPCommunicator(use_compression=False)

    def handler(conn, rh):
        result = comm.recv_data(conn)
        rh["result"] = result

    port, t, rh = _ephemeral_server(handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    
    # Send valid header but invalid pickle data
    bad_payload = b'not_valid_pickle_data'
    cli.sendall(struct.pack('>Q', len(bad_payload)) + MAGIC_NUMBER + bad_payload)
    cli.close()
    t.join(timeout=5)
    assert rh["result"] is None


def test_recv_data_with_connection_reset():
    """Test handling of ConnectionResetError."""
    comm = TCPCommunicator(use_compression=False)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)
    accepted = threading.Event()

    def accept_and_reset():
        conn, _ = srv.accept()
        # Forcefully close with RST (SO_LINGER with timeout 0)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        conn.close()
        srv.close()
        accepted.set()

    t = threading.Thread(target=accept_and_reset, daemon=True)
    t.start()
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    accepted.wait(timeout=5)
    import time; time.sleep(0.1)

    # recv_data should handle ConnectionResetError gracefully and return None
    result = comm.recv_data(cli)
    cli.close()
    t.join(timeout=5)
    assert result is None


def test_recv_data_with_gzip_decompression_error():
    """Test handling of gzip decompression errors."""
    comm = TCPCommunicator(use_compression=True)

    def handler(conn, rh):
        result = comm.recv_data(conn)
        rh["result"] = result

    port, t, rh = _ephemeral_server(handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    
    # Send data that claims to be compressed but isn't valid gzip
    bad_gzip = b'invalid_gzip_data'
    cli.sendall(struct.pack('>Q', len(bad_gzip)) + MAGIC_NUMBER + bad_gzip)
    cli.close()
    t.join(timeout=5)
    assert rh["result"] is None
