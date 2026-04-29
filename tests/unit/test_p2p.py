"""Unit tests for P2P: topology, failure_detector, recovery, ring_node."""

from src.p2p.failure_detector import HeartbeatFailureDetector
from src.p2p.recovery import RecoveryManager
from src.p2p.ring_node import RingNode
from src.p2p.topology import RingTopology
from src.utils.logger import FedLogger


class TestRingTopology:
    """Tests for RingTopology."""

    def test_add_and_ring_order(self) -> None:
        rt = RingTopology()
        rt.add_node(1, ("127.0.0.1", 9001))
        rt.add_node(2, ("127.0.0.1", 9002))
        rt.add_node(3, ("127.0.0.1", 9003))
        assert rt.ring_order == [1, 2, 3]
        assert rt.size == 3

    def test_get_next_node(self) -> None:
        rt = RingTopology()
        rt.add_node(1, ("127.0.0.1", 9001))
        rt.add_node(2, ("127.0.0.1", 9002))
        rt.add_node(3, ("127.0.0.1", 9003))
        assert rt.get_next_node(1) == 2
        assert rt.get_next_node(2) == 3
        assert rt.get_next_node(3) == 1  # wraps around

    def test_get_next_node_empty(self) -> None:
        rt = RingTopology()
        assert rt.get_next_node(1) is None

    def test_remove_node(self) -> None:
        rt = RingTopology()
        rt.add_node(1, ("127.0.0.1", 9001))
        rt.add_node(2, ("127.0.0.1", 9002))
        rt.add_node(3, ("127.0.0.1", 9003))
        rt.remove_node(2)
        assert rt.ring_order == [1, 3]
        assert rt.get_next_node(1) == 3

    def test_handle_failure(self) -> None:
        rt = RingTopology()
        rt.add_node(1, ("127.0.0.1", 9001))
        rt.add_node(2, ("127.0.0.1", 9002))
        rt.add_node(3, ("127.0.0.1", 9003))
        rt.handle_failure(2)
        assert 2 not in rt.ring_order

    def test_get_address(self) -> None:
        rt = RingTopology()
        rt.add_node(1, ("127.0.0.1", 9001))
        assert rt.get_address(1) == ("127.0.0.1", 9001)
        assert rt.get_address(99) is None

    def test_add_duplicate(self) -> None:
        rt = RingTopology()
        rt.add_node(1, ("127.0.0.1", 9001))
        rt.add_node(1, ("127.0.0.1", 9009))  # updates address
        assert rt.size == 1
        assert rt.get_address(1) == ("127.0.0.1", 9009)

    def test_remove_nonexistent(self) -> None:
        rt = RingTopology()
        rt.remove_node(99)  # should not raise


class TestHeartbeatFailureDetector:
    """Tests for HeartbeatFailureDetector."""

    def test_start_stop(self) -> None:
        fd = HeartbeatFailureDetector()
        fd.start()
        assert fd._running
        fd.stop()
        assert not fd._running

    def test_heartbeat(self) -> None:
        fd = HeartbeatFailureDetector()
        assert fd.heartbeat(1)

    def test_check_alive(self) -> None:
        fd = HeartbeatFailureDetector()
        fd.record_heartbeat(1)
        assert fd.check(1, timeout=10.0)

    def test_check_dead(self) -> None:
        fd = HeartbeatFailureDetector()
        assert not fd.check(999, timeout=0.001)

    def test_mark_failed(self) -> None:
        fd = HeartbeatFailureDetector()
        fd.mark_failed(1)
        assert 1 in fd.get_failed_nodes()

    def test_mark_failed_no_duplicate(self) -> None:
        fd = HeartbeatFailureDetector()
        fd.mark_failed(1)
        fd.mark_failed(1)  # should not duplicate
        assert fd.get_failed_nodes().count(1) == 1

    def test_on_failure_callback(self) -> None:
        fd = HeartbeatFailureDetector()
        failed = []
        fd.on_failure(lambda nid: failed.append(nid))
        fd.mark_failed(1)
        assert failed == [1]

    def test_record_heartbeat_clears_failure(self) -> None:
        fd = HeartbeatFailureDetector()
        fd.mark_failed(1)
        assert 1 in fd.get_failed_nodes()
        fd.record_heartbeat(1)
        assert 1 not in fd.get_failed_nodes()


class TestRecoveryManager:
    """Tests for RecoveryManager."""

    def test_failure_triggers_removal(self) -> None:
        topo = RingTopology()
        topo.add_node(1, ("127.0.0.1", 9001))
        topo.add_node(2, ("127.0.0.1", 9002))
        topo.add_node(3, ("127.0.0.1", 9003))

        fd = HeartbeatFailureDetector()
        logger = FedLogger(name="test", console_output=False, file_output=False)
        RecoveryManager(topo, fd, logger=logger)

        fd.mark_failed(2)
        assert 2 not in topo.ring_order
        assert topo.ring_order == [1, 3]


class TestRingNode:
    """Tests for RingNode."""

    def test_start_stop(self) -> None:
        import torch

        from src.model.simple_cnn import create_simple_cnn

        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(data, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        node = RingNode(
            node_id=1,
            model=model,
            dataloader=loader,
            logger=FedLogger(name="test", console_output=False, file_output=False),
        )
        node.start(0)  # port 0 = random
        assert node._running
        assert node.node_id == 1
        node.stop()
        assert not node._running

    def test_next_node_none(self) -> None:
        import torch

        from src.model.simple_cnn import create_simple_cnn

        model = create_simple_cnn(input_channels=1)
        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(data, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        node = RingNode(
            node_id=1,
            model=model,
            dataloader=loader,
            logger=FedLogger(name="test", console_output=False, file_output=False),
        )
        assert node.next_node is None
        assert node.prev_node is None
