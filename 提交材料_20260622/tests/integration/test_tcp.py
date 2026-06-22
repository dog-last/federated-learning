"""Integration tests for TCP communication — ephemeral ports, Event-based sync."""
import socket
import threading

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _start_server(host, handler):
    """Bind port 0 for ephemeral port, return (port, thread, result_dict)."""
    result = {}

    def _run():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, 0))
        port = srv.getsockname()[1]
        result["port"] = port
        result["ready"] = True
        srv.listen(1)
        result["ready_event"].set()
        conn, _ = srv.accept()
        try:
            handler(conn, result)
        finally:
            conn.close()
            srv.close()

    ready = threading.Event()
    result["ready_event"] = ready
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait(timeout=5)
    return result["port"], t, result


@pytest.mark.integration
def test_tcp_communicator_p2p_framing():
    from core.communicator import TCPCommunicator

    comm = TCPCommunicator(use_compression=False)

    def handler(conn, rh):
        rh["value"] = comm.recv_data(conn)

    port, t, rh = _start_server("127.0.0.1", handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    payload = {"type": "smoke", "target": "success"}
    comm.send_data(cli, payload)
    cli.close()
    t.join(timeout=5)
    assert rh["value"] == payload


@pytest.mark.integration
def test_tcp_communicator_compressed():
    from core.communicator import TCPCommunicator

    comm = TCPCommunicator(use_compression=True)

    def handler(conn, rh):
        rh["value"] = comm.recv_data(conn)

    port, t, rh = _start_server("127.0.0.1", handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    payload = {"type": "compression_test", "data": list(range(50))}
    comm.send_data(cli, payload)
    cli.close()
    t.join(timeout=5)
    assert rh["value"] == payload


@pytest.mark.integration
def test_tcp_send_recv_with_meta():
    from core.communicator import TCPCommunicator

    comm = TCPCommunicator(use_compression=False)

    def handler(conn, rh):
        msg, meta = comm.recv_data_with_meta(conn)
        rh["msg"] = msg
        rh["meta"] = meta

    port, t, rh = _start_server("127.0.0.1", handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    comm.send_data(cli, {"type": "meta_test", "round": 1})
    cli.close()
    t.join(timeout=5)
    assert rh["msg"]["type"] == "meta_test"
    assert rh["meta"]["magic_ok"] is True
    assert rh["meta"]["payload_bytes"] > 0


@pytest.mark.integration
def test_tcp_multiple_messages():
    from core.communicator import TCPCommunicator

    comm = TCPCommunicator(use_compression=False)

    def handler(conn, rh):
        received = []
        for _ in range(3):
            msg = comm.recv_data(conn)
            if msg is None:
                break
            received.append(msg)
        rh["received"] = received

    port, t, rh = _start_server("127.0.0.1", handler)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", port))
    for i in range(3):
        comm.send_data(cli, {"type": "batch", "index": i})
    cli.close()
    t.join(timeout=5)
    assert len(rh["received"]) == 3
