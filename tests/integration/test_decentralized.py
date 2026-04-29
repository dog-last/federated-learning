"""Integration test: decentralized (ring P2P) federated learning."""

import time

import torch

from src.core.types import MsgType
from src.model.simple_cnn import create_simple_cnn
from src.p2p.failure_detector import HeartbeatFailureDetector
from src.p2p.ring_node import RingNode
from src.p2p.topology import RingTopology
from src.server.aggregator import FedAvg
from src.utils.logger import FedLogger


class TestRingTopologyIntegration:
    """Integration tests for ring topology with failure detection."""

    def test_ring_topology_with_failure(self) -> None:
        """Test ring restructuring when a node fails."""
        topo = RingTopology()
        topo.add_node(1, ("127.0.0.1", 9001))
        topo.add_node(2, ("127.0.0.1", 9002))
        topo.add_node(3, ("127.0.0.1", 9003))

        # Node 2 fails
        fd = HeartbeatFailureDetector()
        logger = FedLogger(name="test", console_output=False, file_output=False)

        from src.p2p.recovery import RecoveryManager

        RecoveryManager(topo, fd, logger=logger)

        fd.mark_failed(2)

        # Ring should skip node 2
        assert topo.ring_order == [1, 3]
        assert topo.get_next_node(1) == 3
        assert topo.get_next_node(3) == 1

    def test_ring_full_failure(self) -> None:
        """Test ring when all nodes except one fail."""
        topo = RingTopology()
        topo.add_node(1, ("127.0.0.1", 9001))
        topo.add_node(2, ("127.0.0.1", 9002))
        topo.add_node(3, ("127.0.0.1", 9003))

        topo.handle_failure(2)
        topo.handle_failure(3)

        assert topo.ring_order == [1]
        assert topo.get_next_node(1) == 1

    def test_failure_detector_lifecycle(self) -> None:
        """Test heartbeat detection and recovery."""
        fd = HeartbeatFailureDetector(heartbeat_interval=0.1, heartbeat_timeout=0.2)

        try:
            fd.start()
            fd.record_heartbeat(1)
            fd.record_heartbeat(2)

            # Use shorter timeout for faster test
            assert fd.check(1, timeout=1.0)
            assert fd.check(2, timeout=1.0)
            assert not fd.check(999, timeout=0.001)

            fd.mark_failed(2)
            assert 2 in fd.get_failed_nodes()

            fd.record_heartbeat(2)
            assert 2 not in fd.get_failed_nodes()
        finally:
            fd.stop()

    def test_node_start_stop(self) -> None:
        """Test ring node start/stop lifecycle."""
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

        try:
            node.start(0)
            assert node._running
            assert node.node_id == 1
            assert node._topology.ring_order == [1]
        finally:
            node.stop()
            assert not node._running

    def test_two_node_ring(self) -> None:
        """Test two ring nodes connecting and exchanging model."""
        logger1 = FedLogger(name="Node-1", console_output=False, file_output=False)
        logger2 = FedLogger(name="Node-2", console_output=False, file_output=False)

        model1 = create_simple_cnn(input_channels=1)
        model2 = create_simple_cnn(input_channels=1)

        data = torch.randn(16, 1, 28, 28)
        target = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(data, target)
        loader1 = torch.utils.data.DataLoader(dataset, batch_size=8)
        loader2 = torch.utils.data.DataLoader(dataset, batch_size=8)

        node1 = RingNode(node_id=1, model=model1, dataloader=loader1, logger=logger1)
        node2 = RingNode(node_id=2, model=model2, dataloader=loader2, logger=logger2)

        # Capture messages received by node2
        received: list[object] = []

        original_handle_ring_pass = node2._handle_ring_pass

        def capture_ring_pass(conn, msg) -> None:
            received.append(msg)
            original_handle_ring_pass(conn, msg)

        node2._handle_ring_pass = capture_ring_pass

        try:
            node1.start(0)
            node2.start(0)

            # Manually set up topology
            addr1 = node1._listener.address
            addr2 = node2._listener.address

            # Verify addresses are valid
            assert addr1[1] != 0, f"Node 1 got invalid port: {addr1}"
            assert addr2[1] != 0, f"Node 2 got invalid port: {addr2}"

            node1._topology.add_node(2, addr2)
            node2._topology.add_node(1, addr1)

            # Wait a bit for nodes to be ready
            time.sleep(0.1)

            # Node 1 passes model to node 2
            weights = model1.get_weights()
            success, payload_size, duration_ms = node1.pass_model(weights)
            assert success is True

            # Wait for node2's background thread to process the message
            time.sleep(0.5)

            assert len(received) == 1
            assert received[0].msg_type == MsgType.RING_PASS

            # Verify node 2's next node
            assert node2.next_node is not None
            assert node2.prev_node is not None
        finally:
            node1.stop()
            node2.stop()

    def test_aggregation_with_ring(self) -> None:
        """Test that FedAvg works with weights from ring topology nodes."""
        model1 = create_simple_cnn(input_channels=1)
        model2 = create_simple_cnn(input_channels=1)
        model3 = create_simple_cnn(input_channels=1)

        # Simulate local training
        w1 = model1.get_weights()
        w2 = model2.get_weights()
        w3 = model3.get_weights()

        # Slightly perturb to simulate training
        for k in w2:
            w2[k] = w2[k] + torch.randn_like(w2[k]) * 0.01
        for k in w3:
            w3[k] = w3[k] + torch.randn_like(w3[k]) * 0.01

        aggregator = FedAvg()
        aggregated = aggregator.aggregate([w1, w2, w3])

        assert isinstance(aggregated, dict)
        assert len(aggregated) == len(w1)

        # Verify aggregated weights differ from any single node
        for k in w1:
            assert not torch.allclose(w1[k], aggregated[k])
