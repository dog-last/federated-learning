"""Unit tests for server: aggregator, client_manager, round_coordinator, checkpoint."""

import threading
import time

import pytest
import torch

from src.core.exceptions import AggregationError
from src.core.types import MsgType
from src.protocol.message import Message
from src.server.aggregator import FedAvg, FedProx, create_aggregator
from src.server.checkpoint import load_checkpoint, save_checkpoint
from src.server.client_manager import ClientManager
from src.server.round_coordinator import RoundCoordinator
from src.transport.connection import Connection
from src.transport.listener import Listener
from src.utils.logger import FedLogger


class TestFedAvg:
    """Tests for FedAvg aggregator."""

    def test_name(self) -> None:
        assert FedAvg().name == "fedavg"

    def test_equal_average(self) -> None:
        w1 = {"w": torch.tensor([1.0, 2.0])}
        w2 = {"w": torch.tensor([3.0, 4.0])}
        result = FedAvg().aggregate([w1, w2])
        assert torch.allclose(result["w"], torch.tensor([2.0, 3.0]))

    def test_weighted_average(self) -> None:
        w1 = {"w": torch.tensor([1.0])}
        w2 = {"w": torch.tensor([3.0])}
        result = FedAvg().aggregate([w1, w2], client_sizes=[3, 1])
        assert torch.allclose(result["w"], torch.tensor([1.5]))

    def test_empty_list_raises(self) -> None:
        with pytest.raises(AggregationError):
            FedAvg().aggregate([])

    def test_multiple_keys(self) -> None:
        w1 = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        w2 = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}
        result = FedAvg().aggregate([w1, w2])
        assert torch.allclose(result["a"], torch.tensor([2.0]))
        assert torch.allclose(result["b"], torch.tensor([3.0]))


class TestFedProx:
    """Tests for FedProx aggregator."""

    def test_name(self) -> None:
        assert FedProx().name == "fedprox"

    def test_aggregate_same_as_fedavg(self) -> None:
        w1 = {"w": torch.tensor([1.0, 2.0])}
        w2 = {"w": torch.tensor([3.0, 4.0])}
        result = FedProx(mu=0.01).aggregate([w1, w2])
        assert torch.allclose(result["w"], torch.tensor([2.0, 3.0]))

    def test_custom_mu(self) -> None:
        fp = FedProx(mu=0.1)
        assert fp.mu == 0.1


class TestCreateAggregator:
    """Tests for create_aggregator factory."""

    def test_fedavg(self) -> None:
        agg = create_aggregator("fedavg")
        assert agg.name == "fedavg"

    def test_fedprox(self) -> None:
        agg = create_aggregator("fedprox", fedprox_mu=0.05)
        assert agg.name == "fedprox"
        assert isinstance(agg, FedProx)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            create_aggregator("unknown")


class TestClientManager:
    """Tests for ClientManager."""

    def _make_conn(self) -> Connection:
        """Create a pair and return the server side connection."""
        listener = Listener()
        listener.bind("127.0.0.1", 0)
        _, port = listener.address

        result = []

        def accept() -> None:
            result.append(listener.accept())

        t = threading.Thread(target=accept)
        t.start()

        import socket as _socket

        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))

        t.join(timeout=5.0)
        listener.close()
        return result[0]

    def test_add_client(self) -> None:
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn = self._make_conn()
        cid = cm.add_client(conn)
        assert cid == 1
        assert cm.num_clients == 1
        conn.close()

    def test_remove_client(self) -> None:
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn = self._make_conn()
        cid = cm.add_client(conn)
        cm.remove_client(cid)
        assert cm.num_clients == 0

    def test_get_connection(self) -> None:
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn = self._make_conn()
        cid = cm.add_client(conn)
        assert cm.get_connection(cid) is conn
        assert cm.get_connection(999) is None
        conn.close()

    def test_client_ids(self) -> None:
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn1 = self._make_conn()
        conn2 = self._make_conn()
        cm.add_client(conn1)
        cm.add_client(conn2)
        assert len(cm.client_ids) == 2
        conn1.close()
        conn2.close()

    def test_broadcast(self) -> None:
        """Test broadcasting data to all clients."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn = self._make_conn()
        cm.add_client(conn)

        # Broadcast should return results dict
        results = cm.broadcast(b"test data")
        assert len(results) == 1
        assert results[1] is True  # Should succeed

        cm.remove_client(1)

    def test_broadcast_with_exclude(self) -> None:
        """Test broadcasting with exclusion list."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn1 = self._make_conn()
        conn2 = self._make_conn()
        cm.add_client(conn1)
        cm.add_client(conn2)

        # Broadcast excluding client 1
        results = cm.broadcast(b"test data", exclude=[1])
        assert len(results) == 1
        assert 1 not in results
        assert 2 in results

        cm.remove_client(1)
        cm.remove_client(2)

    def test_broadcast_message(self) -> None:
        """Test broadcasting a Message object."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn = self._make_conn()
        cm.add_client(conn)

        msg = Message(msg_type=MsgType.MODEL_BROADCAST, payload={"data": "test"}, round_id=1)
        results = cm.broadcast_message(msg)
        assert len(results) == 1

        cm.remove_client(1)

    def test_broadcast_with_failed_send(self) -> None:
        """Test broadcast handles send failures gracefully."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn = self._make_conn()
        cid = cm.add_client(conn)

        # Close connection to cause send failure
        conn.close()
        time.sleep(0.1)

        results = cm.broadcast(b"test data")
        assert results[cid] is False  # Should report failure

    def test_remove_nonexistent_client(self) -> None:
        """Test removing a client that doesn't exist."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        # Should not raise error
        cm.remove_client(999)

    def test_collect_with_no_clients(self) -> None:
        """Test collect when no clients connected."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        results = cm.collect(timeout=0.1)
        assert results == {}

    def test_collect_messages_with_no_clients(self) -> None:
        """Test collect_messages when no clients connected."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        results = cm.collect_messages(timeout=0.1)
        assert results == {}

    def test_collect_with_timeout(self) -> None:
        """Test collect handles timeout correctly."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn = self._make_conn()
        cid = cm.add_client(conn)

        # Collect will timeout since no data is sent
        results = cm.collect(timeout=0.1)
        assert results[cid] is None  # Should be None on timeout

        cm.remove_client(cid)

    def test_num_clients_property(self) -> None:
        """Test num_clients property."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        assert cm.num_clients == 0

        conn = self._make_conn()
        cm.add_client(conn)
        assert cm.num_clients == 1

        cm.remove_client(1)
        assert cm.num_clients == 0


class TestRoundCoordinator:
    """Tests for RoundCoordinator."""

    def _make_conn(self) -> Connection:
        """Create a pair and return the server side connection."""
        listener = Listener()
        listener.bind("127.0.0.1", 0)
        _, port = listener.address

        result = []

        def accept() -> None:
            result.append(listener.accept())

        t = threading.Thread(target=accept)
        t.start()

        import socket as _socket

        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))

        t.join(timeout=5.0)
        listener.close()
        return result[0]

    def test_start_round(self) -> None:
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        agg = FedAvg()
        rc = RoundCoordinator(
            cm, agg, logger=FedLogger(name="test", console_output=False, file_output=False)
        )
        rc.start_round(1)
        assert rc._current_round == 1

    def test_start_round_resets_stats(self) -> None:
        """Test that start_round resets network stats and timing."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        agg = FedAvg()
        rc = RoundCoordinator(
            cm, agg, logger=FedLogger(name="test", console_output=False, file_output=False)
        )

        # Set some previous state
        rc._network_stats = {1: "some_stat"}
        rc._broadcast_payload_size = 1000

        rc.start_round(5)

        assert rc._current_round == 5
        assert rc._network_stats == {}
        assert rc._broadcast_payload_size == 0
        assert rc._round_start_time > 0

    def test_end_round(self) -> None:
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        agg = FedAvg()
        rc = RoundCoordinator(
            cm, agg, logger=FedLogger(name="test", console_output=False, file_output=False)
        )
        rc.start_round(1)
        stats = rc.end_round({})
        assert stats.round_id == 1

    def test_end_round_returns_stats(self) -> None:
        """Test that end_round returns correct RoundStats."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        agg = FedAvg()
        rc = RoundCoordinator(
            cm, agg, logger=FedLogger(name="test", console_output=False, file_output=False)
        )

        rc.start_round(3)
        # Wait a bit to ensure total_time > 0
        time.sleep(0.05)
        stats = rc.end_round({"layer": torch.tensor([1.0])})

        assert stats.round_id == 3
        assert stats.broadcast_time == 0.0
        assert stats.aggregate_time == 0.0
        assert stats.total_time > 0
        assert stats.participating_clients == []
        assert stats.timeout_clients == []

    def test_broadcast_model(self) -> None:
        """Test broadcasting model to clients."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        agg = FedAvg()
        rc = RoundCoordinator(
            cm, agg, logger=FedLogger(name="test", console_output=False, file_output=False)
        )
        rc.start_round(1)

        weights = {"layer": torch.tensor([1.0, 2.0])}
        duration = rc.broadcast_model(weights)

        assert duration >= 0
        assert rc._broadcast_payload_size > 0

    def test_collect_updates_empty(self) -> None:
        """Test collect_updates with no clients."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        agg = FedAvg()
        rc = RoundCoordinator(
            cm, agg, logger=FedLogger(name="test", console_output=False, file_output=False)
        )
        rc.start_round(1)

        updates = rc.collect_updates(timeout=0.1)

        assert updates == {}

    def test_collect_updates_with_timeout(self) -> None:
        """Test collect_updates times out correctly."""
        cm = ClientManager()
        cm._logger = FedLogger(name="test", console_output=False, file_output=False)
        conn = self._make_conn()
        cm.add_client(conn)

        agg = FedAvg()
        rc = RoundCoordinator(
            cm, agg, logger=FedLogger(name="test", console_output=False, file_output=False)
        )
        rc.start_round(1)

        # Will timeout since no data is sent
        updates = rc.collect_updates(timeout=0.1)

        # Should have entry for the client but with None (timeout)
        assert len(updates) == 1
        assert list(updates.values())[0] is None

        cm.remove_client(1)


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_save_and_load(self, tmp_dir: str) -> None:
        weights = {"w": torch.tensor([1.0, 2.0, 3.0])}
        path = str(tmp_dir) + "/ckpt.pt"
        save_checkpoint(weights, path, round_id=5)

        loaded_weights, rid = load_checkpoint(path)
        assert rid == 5
        assert torch.allclose(loaded_weights["w"], weights["w"])

    def test_save_without_round(self, tmp_dir: str) -> None:
        weights = {"w": torch.tensor([1.0])}
        path = str(tmp_dir) + "/ckpt2.pt"
        save_checkpoint(weights, path)

        loaded_weights, rid = load_checkpoint(path)
        assert rid is None
