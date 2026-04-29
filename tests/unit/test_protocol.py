"""Unit tests for protocol: message, codec, serializer, constants."""

import struct
import time

import pytest

from src.core.types import MsgType
from src.protocol.codec import Codec
from src.protocol.constants import HEADER_SIZE, LENGTH_PREFIX_SIZE
from src.protocol.message import Message
from src.protocol.serializer import TorchSerializer


class TestMessage:
    """Tests for Message dataclass."""

    def test_creation(self) -> None:
        msg = Message(msg_type=MsgType.MODEL_BROADCAST, client_id=1, payload={"key": "val"})
        assert msg.msg_type == MsgType.MODEL_BROADCAST
        assert msg.client_id == 1
        assert msg.payload == {"key": "val"}
        assert msg.timestamp > 0

    def test_auto_timestamp(self) -> None:
        before = time.time()
        msg = Message(msg_type=MsgType.ERROR)
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_optional_fields(self) -> None:
        msg = Message(msg_type=MsgType.HEARTBEAT)
        assert msg.client_id is None
        assert msg.round_id is None
        assert msg.payload is None


class TestTorchSerializer:
    """Tests for TorchSerializer."""

    def setup_method(self) -> None:
        self.ser = TorchSerializer()

    def test_roundtrip(self) -> None:
        import torch

        weights = {"layer1": torch.randn(3, 3), "layer2": torch.randn(5)}
        data = self.ser.serialize_weights(weights)
        recovered = self.ser.deserialize_weights(data)
        assert set(recovered.keys()) == set(weights.keys())
        for k in weights:
            assert torch.allclose(weights[k], recovered[k])

    def test_get_size(self) -> None:
        import torch

        weights = {"w": torch.randn(10, 10)}
        size = self.ser.get_size(weights)
        assert size > 0
        assert size == len(self.ser.serialize_weights(weights))

    def test_empty_weights(self) -> None:
        data = self.ser.serialize_weights({})
        recovered = self.ser.deserialize_weights(data)
        assert recovered == {}


class TestCodec:
    """Tests for Codec."""

    def setup_method(self) -> None:
        self.codec = Codec()

    def test_encode_decode_roundtrip(self) -> None:
        msg = Message(
            msg_type=MsgType.MODEL_BROADCAST,
            client_id=3,
            payload={"round": 1, "data": [1, 2, 3]},
            round_id=5,
        )
        encoded = self.codec.encode(msg)
        assert isinstance(encoded, bytes)

        # Extract length prefix + body
        length = struct.unpack("!Q", encoded[:8])[0]
        body = encoded[8:]
        assert len(body) == length

        decoded = self.codec.decode(body)
        assert decoded.msg_type == MsgType.MODEL_BROADCAST
        assert decoded.client_id == 3
        assert decoded.payload == {"round": 1, "data": [1, 2, 3]}
        assert decoded.round_id == 5

    def test_none_client_id(self) -> None:
        msg = Message(msg_type=MsgType.ERROR, client_id=None)
        encoded = self.codec.encode(msg)
        body = encoded[8:]
        decoded = self.codec.decode(body)
        assert decoded.client_id is None

    def test_none_round_id(self) -> None:
        msg = Message(msg_type=MsgType.HEARTBEAT, round_id=None)
        encoded = self.codec.encode(msg)
        body = encoded[8:]
        decoded = self.codec.decode(body)
        assert decoded.round_id is None

    def test_get_message_length(self) -> None:
        length_data = struct.pack("!Q", 12345)
        assert Codec.get_message_length(length_data) == 12345

    def test_get_message_length_too_short(self) -> None:
        from src.core.exceptions import ProtocolError

        with pytest.raises(ProtocolError):
            Codec.get_message_length(b"\x00\x01")

    def test_decode_too_short(self) -> None:
        from src.core.exceptions import ProtocolError

        with pytest.raises(ProtocolError):
            self.codec.decode(b"\x00\x01")

    def test_header_size(self) -> None:
        assert HEADER_SIZE == 24

    def test_length_prefix_size(self) -> None:
        assert LENGTH_PREFIX_SIZE == 8

    def test_encode_with_bytes_payload(self) -> None:
        msg = Message(msg_type=MsgType.MODEL_UPDATE, payload=b"raw_bytes_data")
        encoded = self.codec.encode(msg)
        body = encoded[8:]
        decoded = self.codec.decode(body)
        assert decoded.payload == b"raw_bytes_data"

    def test_encode_with_dict_payload(self) -> None:
        msg = Message(
            msg_type=MsgType.CLIENT_ACK,
            payload={"client_id": 42, "config": {"lr": 0.01}},
        )
        encoded = self.codec.encode(msg)
        body = encoded[8:]
        decoded = self.codec.decode(body)
        assert decoded.payload["client_id"] == 42
