"""Message codec with length-prefix framing, gzip compression, and zero-copy."""

import gzip
import pickle
import struct

from src.core.exceptions import ProtocolError
from src.core.interfaces import ICodec, IMessage
from src.protocol.constants import LENGTH_PREFIX_SIZE
from src.protocol.message import Message
from src.protocol.serializer import TorchSerializer

# Header: msg_type(4) + client_id(4) + timestamp(8) + round_id(4) + flags(4)
# Bit 0 of flags: payload is gzip-compressed
_HEADER_FMT = "!IIdII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 24
_COMPRESSED_BIT = 1 << 0

# Default: compress payloads above this threshold (bytes)
_COMPRESS_THRESHOLD = 256


class Codec(ICodec):
    """Encode/decode messages with length-prefix framing and gzip compression.

    Wire format:
        [8 bytes: body length (uint64 big-endian)]
        [4 bytes: msg_type  (uint32)]
        [4 bytes: client_id (uint32, 0=None)]
        [8 bytes: timestamp (float64)]
        [4 bytes: round_id  (uint32, 0=None)]
        [4 bytes: flags     (uint32, bit 0 = gzip)]
        [N bytes: payload   (pickle, optionally gzip-compressed)]

    Attributes:
        serializer: Weight serializer instance.
        compress_threshold: Minimum payload size to trigger gzip compression.
    """

    def __init__(
        self,
        serializer: TorchSerializer | None = None,
        compress_threshold: int = _COMPRESS_THRESHOLD,
    ) -> None:
        self.serializer = serializer or TorchSerializer()
        self.compress_threshold = compress_threshold

    def encode(self, message: IMessage) -> bytes:
        """Encode a message to bytes.

        Args:
            message: Message object.

        Returns:
            bytes: Encoded byte stream with length prefix.
        """
        payload_bytes = pickle.dumps(message.payload, protocol=pickle.HIGHEST_PROTOCOL)

        flags = 0
        if len(payload_bytes) >= self.compress_threshold:
            payload_bytes = gzip.compress(payload_bytes, compresslevel=1)
            flags |= _COMPRESSED_BIT

        header = struct.pack(
            _HEADER_FMT,
            int(message.msg_type),
            message.client_id or 0,
            message.timestamp,
            message.round_id or 0,
            flags,
        )

        body = header + payload_bytes
        length_prefix = struct.pack("!Q", len(body))
        return length_prefix + body

    def decode(self, data: bytes) -> IMessage:
        """Decode bytes to a message.

        Args:
            data: Byte stream (body only, without length prefix).

        Returns:
            IMessage: Decoded message.

        Raises:
            ProtocolError: If the data is too short.
        """
        if len(data) < _HEADER_SIZE:
            raise ProtocolError(f"Data too short: {len(data)} < {_HEADER_SIZE}")

        msg_type, client_id, timestamp, round_id, flags = struct.unpack(
            _HEADER_FMT, data[:_HEADER_SIZE]
        )

        payload_bytes = data[_HEADER_SIZE:]
        if flags & _COMPRESSED_BIT:
            payload_bytes = gzip.decompress(payload_bytes)

        payload = pickle.loads(payload_bytes)

        return Message(
            msg_type=msg_type,
            client_id=client_id if client_id != 0 else None,
            payload=payload,
            timestamp=timestamp,
            round_id=round_id if round_id != 0 else None,
        )

    @staticmethod
    def get_message_length(length_data: bytes) -> int:
        """Parse the length prefix.

        Args:
            length_data: 8-byte length prefix.

        Returns:
            int: Message body length.

        Raises:
            ProtocolError: If length_data is too short.
        """
        if len(length_data) < LENGTH_PREFIX_SIZE:
            raise ProtocolError(f"Length data too short: {len(length_data)}")
        return struct.unpack("!Q", length_data)[0]
