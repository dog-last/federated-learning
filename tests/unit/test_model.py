"""Unit tests for core types and config."""

from pathlib import Path

import pytest
import yaml

from src.core.types import (
    ClientNodeConfig,
    ClientsConfig,
    Config,
    DatasetConfig,
    ErrorCode,
    ModelConfig,
    MsgType,
    PeerNodeConfig,
    PeersConfig,
    RoundStats,
    ServerConfig,
    ServerTimeouts,
    TrainingResult,
)


class TestMsgType:
    """Tests for MsgType enum."""

    def test_centralized_messages(self) -> None:
        assert MsgType.MODEL_BROADCAST == 1
        assert MsgType.MODEL_UPDATE == 2
        assert MsgType.CLIENT_REGISTER == 3
        assert MsgType.CLIENT_ACK == 4

    def test_p2p_messages(self) -> None:
        assert MsgType.NODE_JOIN == 10
        assert MsgType.NODE_LEAVE == 11
        assert MsgType.RING_PASS == 12
        assert MsgType.HEARTBEAT == 13
        assert MsgType.HEARTBEAT_ACK == 14
        assert MsgType.TOPOLOGY_UPDATE == 15

    def test_control_messages(self) -> None:
        assert MsgType.TRAIN_START == 20
        assert MsgType.TRAIN_COMPLETE == 21
        assert MsgType.ROUND_END == 22
        assert MsgType.ERROR == 99


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_connection_errors(self) -> None:
        assert ErrorCode.CONNECTION_REFUSED == 1
        assert ErrorCode.CONNECTION_TIMEOUT == 2
        assert ErrorCode.CONNECTION_CLOSED == 3

    def test_protocol_errors(self) -> None:
        assert ErrorCode.INVALID_MESSAGE == 100
        assert ErrorCode.DECODE_ERROR == 101


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_creation(self) -> None:
        r = TrainingResult(loss=0.5, accuracy=85.0, num_samples=100, training_time=1.2)
        assert r.loss == 0.5
        assert r.accuracy == 85.0
        assert r.num_samples == 100
        assert r.training_time == 1.2


class TestRoundStats:
    """Tests for RoundStats dataclass."""

    def test_defaults(self) -> None:
        s = RoundStats(round_id=1, broadcast_time=0.1)
        assert s.round_id == 1
        assert s.training_times == {}
        assert s.collect_times == {}
        assert s.aggregate_time == 0.0
        assert s.participating_clients == []
        assert s.timeout_clients == []
        assert s.global_accuracy == 0.0

    def test_with_data(self) -> None:
        s = RoundStats(
            round_id=2,
            broadcast_time=0.05,
            participating_clients=[1, 2],
            timeout_clients=[3],
            global_accuracy=85.0,
        )
        assert len(s.participating_clients) == 2
        assert s.timeout_clients == [3]


class TestConfig:
    """Tests for Config dataclasses."""

    def test_from_yaml(self, tmp_yaml: str) -> None:
        c = Config.from_yaml(tmp_yaml)
        assert c.mode == "centralized"
        assert c.model.input_channels == 1
        assert c.dataset.num_clients == 2
        assert c.training.rounds == 2
        assert c.server is not None
        assert c.server.port == 9000
        assert c.server.address == "127.0.0.1"
        assert c.clients is not None
        assert len(c.clients.nodes) == 2
        assert c.num_clients == 2

    def test_from_yaml_centralized_validates_num_clients(self, tmp_dir: str) -> None:
        path = Path(tmp_dir) / "bad.yaml"
        path.write_text(
            yaml.dump(
                {
                    "mode": "centralized",
                    "dataset": {"num_clients": 5},
                    "server": {"port": 9000},
                    "clients": {"nodes": [{"id": 1}, {"id": 2}]},
                }
            )
        )
        with pytest.raises(ValueError, match="num_clients"):
            Config.from_yaml(str(path))

    def test_from_yaml_centralized_requires_clients(self, tmp_dir: str) -> None:
        path = Path(tmp_dir) / "no_clients.yaml"
        path.write_text(
            yaml.dump(
                {
                    "mode": "centralized",
                    "server": {"port": 9000},
                }
            )
        )
        with pytest.raises(ValueError, match="clients.nodes"):
            Config.from_yaml(str(path))

    def test_from_yaml_decentralized(self, tmp_dir: str) -> None:
        path = Path(tmp_dir) / "decentralized.yaml"
        path.write_text(
            yaml.dump(
                {
                    "mode": "decentralized",
                    "dataset": {"num_clients": 2},
                    "p2p": {"topology": "ring"},
                    "peers": {
                        "nodes": [
                            {"id": 1, "host": "127.0.0.1", "port": 9001},
                            {"id": 2, "host": "127.0.0.1", "port": 9002},
                        ]
                    },
                }
            )
        )
        c = Config.from_yaml(str(path))
        assert c.mode == "decentralized"
        assert c.p2p is not None
        assert c.peers is not None
        assert len(c.peers.nodes) == 2
        assert c.num_clients == 2

    def test_from_yaml_unknown_mode(self, tmp_dir: str) -> None:
        path = Path(tmp_dir) / "bad_mode.yaml"
        path.write_text(yaml.dump({"mode": "hybrid"}))
        with pytest.raises(ValueError, match="Unknown mode"):
            Config.from_yaml(str(path))


class TestConfigDataclasses:
    """Tests for individual config dataclasses."""

    def test_model_config(self) -> None:
        mc = ModelConfig(name="resnet", input_channels=3, num_classes=100)
        assert mc.name == "resnet"
        assert mc.input_channels == 3

    def test_dataset_config(self) -> None:
        dc = DatasetConfig(name="cifar10", num_clients=5, partition_strategy="non_iid", alpha=0.1)
        assert dc.name == "cifar10"
        assert dc.alpha == 0.1

    def test_server_config(self) -> None:
        sc = ServerConfig(host="0.0.0.0", port=8080, address="192.168.1.1")
        assert sc.host == "0.0.0.0"
        assert sc.port == 8080
        assert sc.address == "192.168.1.1"

    def test_server_timeouts(self) -> None:
        st = ServerTimeouts(connect=5.0, round=120.0)
        assert st.connect == 5.0
        assert st.round == 120.0

    def test_client_node_config(self) -> None:
        cn = ClientNodeConfig(id=1, host="192.168.1.10")
        assert cn.id == 1
        assert cn.host == "192.168.1.10"

    def test_clients_config(self) -> None:
        cc = ClientsConfig(
            server_address="10.0.0.1:9999",
            nodes=[ClientNodeConfig(id=1), ClientNodeConfig(id=2)],
        )
        assert cc.server_address == "10.0.0.1:9999"
        assert len(cc.nodes) == 2

    def test_peer_node_config(self) -> None:
        pn = PeerNodeConfig(id=1, host="192.168.1.20", port=9001)
        assert pn.id == 1
        assert pn.host == "192.168.1.20"
        assert pn.port == 9001

    def test_peers_config(self) -> None:
        pc = PeersConfig(
            nodes=[
                PeerNodeConfig(id=1, host="127.0.0.1", port=9001),
                PeerNodeConfig(id=2, host="127.0.0.1", port=9002),
            ]
        )
        assert len(pc.nodes) == 2
