"""Integration test: centralized federated learning round over real TCP."""

import contextlib
import socket
import threading
import time

import pytest
import torch

from src.core.types import MsgType, Weights
from src.model.simple_cnn import create_simple_cnn
from src.protocol.codec import Codec
from src.protocol.message import Message
from src.protocol.serializer import TorchSerializer
from src.server.aggregator import FedAvg
from src.transport.connection import Connection
from src.transport.listener import Listener
from src.utils.logger import FedLogger


class TestCentralizedIntegration:
    """Integration tests for centralized FL: server + 2 clients over TCP."""

    def test_full_round_trip(self) -> None:
        """Test: server broadcasts model, clients receive, train, send back, server aggregates."""
        FedLogger(name="IntegrationTest", console_output=False, file_output=False)
        serializer = TorchSerializer()
        aggregator = FedAvg()

        listener = Listener()
        server_conns: list[Connection] = []
        client_conns: list[Connection] = []
        client_socks: list[socket.socket] = []
        accept_thread: threading.Thread | None = None

        try:
            # Start server listener on random port
            listener.bind("127.0.0.1", 0)
            host, port = listener.address
            assert port != 0, "Listener should be assigned a real port"

            # Accept clients in background thread with timeout
            accept_error: list[Exception | None] = [None]

            def accept_clients() -> None:
                try:
                    for _ in range(2):
                        conn = listener.accept(timeout=5.0)
                        server_conns.append(conn)
                except Exception as e:
                    accept_error[0] = e

            accept_thread = threading.Thread(target=accept_clients, daemon=True)
            accept_thread.start()

            # Give server time to start listening
            time.sleep(0.1)

            # Clients connect
            for _ in range(2):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((host, port))
                client_conns.append(Connection(sock, Codec()))
                client_socks.append(sock)

            # Wait for server to accept connections
            accept_thread.join(timeout=10.0)
            if accept_thread.is_alive():
                raise TimeoutError("Server failed to accept connections within timeout")
            if accept_error[0]:
                raise accept_error[0]

            assert len(server_conns) == 2, f"Expected 2 server connections, got {len(server_conns)}"

            # Server broadcasts global model
            global_model = create_simple_cnn(input_channels=1)
            global_weights = global_model.get_weights()
            serialized = serializer.serialize_weights(global_weights)

            broadcast_msg = Message(
                msg_type=MsgType.MODEL_BROADCAST,
                payload={"weights": serialized, "round_id": 1},
                round_id=1,
            )

            # Start server-side recv threads BEFORE clients send,
            # so TCP buffers don't fill up and block sendall.
            received_weights: list[Weights] = [None, None]
            server_errors: list[Exception | None] = [None, None]

            def server_recv(idx: int, conn: Connection) -> None:
                try:
                    msg = conn.recv_message(timeout=10.0)
                    assert msg.msg_type == MsgType.MODEL_UPDATE
                    w = serializer.deserialize_weights(msg.payload["weights"])
                    received_weights[idx] = w
                except Exception as e:
                    server_errors[idx] = e

            server_recv_threads = []
            for i, conn in enumerate(server_conns):
                t = threading.Thread(target=server_recv, args=(i, conn), daemon=True)
                t.start()
                server_recv_threads.append(t)

            # Clients receive and train in background threads to avoid
            # deadlock when large payloads exceed TCP buffer size.
            client_updated_weights: list[Weights] = [None, None]
            client_errors: list[Exception | None] = [None, None]

            def client_worker(idx: int, conn: Connection) -> None:
                try:
                    msg = conn.recv_message(timeout=10.0)
                    assert msg.msg_type == MsgType.MODEL_BROADCAST

                    weights = serializer.deserialize_weights(msg.payload["weights"])
                    local_model = create_simple_cnn(input_channels=1)
                    local_model.set_weights(weights)

                    # Simple "training" — just perturb weights slightly
                    updated = local_model.get_weights()
                    for k in updated:
                        updated[k] = updated[k] + torch.randn_like(updated[k]) * 0.01
                    client_updated_weights[idx] = updated

                    # Send update back
                    serialized_update = serializer.serialize_weights(updated)
                    update_msg = Message(
                        msg_type=MsgType.MODEL_UPDATE,
                        client_id=1,
                        payload={"weights": serialized_update},
                        round_id=1,
                    )
                    conn.send_message(update_msg)
                except Exception as e:
                    client_errors[idx] = e

            client_threads = []
            for i, conn in enumerate(client_conns):
                t = threading.Thread(target=client_worker, args=(i, conn), daemon=True)
                t.start()
                client_threads.append(t)

            # Server broadcasts (clients are already receiving in threads)
            for conn in server_conns:
                conn.send_message(broadcast_msg)

            # Wait for clients to finish sending updates
            for t in client_threads:
                t.join(timeout=15.0)
            for _i, err in enumerate(client_errors):
                if err:
                    raise err
            assert all(w is not None for w in client_updated_weights)

            # Wait for server to finish receiving updates
            for t in server_recv_threads:
                t.join(timeout=15.0)
            for _i, err in enumerate(server_errors):
                if err:
                    raise err
            assert all(w is not None for w in received_weights)

            # Server aggregates
            aggregated = aggregator.aggregate(received_weights)
            assert isinstance(aggregated, dict)
            assert len(aggregated) > 0

            # Verify aggregated weights are different from global
            for k in global_weights:
                assert not torch.allclose(global_weights[k], aggregated[k])

        finally:
            # Cleanup in reverse order
            for conn in client_conns:
                with contextlib.suppress(Exception):
                    conn.close()
            for sock in client_socks:
                with contextlib.suppress(Exception):
                    sock.close()
            for conn in server_conns:
                with contextlib.suppress(Exception):
                    conn.close()
            with contextlib.suppress(Exception):
                listener.close()

    def test_connection_timeout(self) -> None:
        """Test that connection operations respect timeouts."""
        listener = Listener()

        try:
            listener.bind("127.0.0.1", 0)

            # Test accept timeout
            from src.core.exceptions import NetworkTimeoutError

            with pytest.raises(NetworkTimeoutError):
                listener.accept(timeout=0.1)

        finally:
            listener.close()

    def test_message_exchange(self) -> None:
        """Test basic message exchange between server and client."""
        listener = Listener()
        server_conn: Connection | None = None
        client_conn: Connection | None = None
        client_sock: socket.socket | None = None

        try:
            listener.bind("127.0.0.1", 0)
            host, port = listener.address

            server_conn_future: list[Connection | None] = [None]

            def accept_client() -> None:
                with contextlib.suppress(Exception):
                    server_conn_future[0] = listener.accept(timeout=5.0)

            thread = threading.Thread(target=accept_client, daemon=True)
            thread.start()
            time.sleep(0.1)

            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_sock.settimeout(5.0)
            client_sock.connect((host, port))
            client_conn = Connection(client_sock, Codec())

            thread.join(timeout=5.0)
            server_conn = server_conn_future[0]
            assert server_conn is not None

            # Send message from server to client
            test_msg = Message(
                msg_type=MsgType.MODEL_BROADCAST,
                payload={"data": "hello"},
                round_id=1,
            )
            server_conn.send_message(test_msg)

            # Receive on client
            received = client_conn.recv_message(timeout=5.0)
            assert received.msg_type == MsgType.MODEL_BROADCAST
            assert received.payload["data"] == "hello"

        finally:
            if client_conn:
                client_conn.close()
            if client_sock:
                client_sock.close()
            if server_conn:
                server_conn.close()
            listener.close()
