"""Protocol constants."""

from src.core.types import ErrorCode, MsgType

# Header with flags field: msg_type(4) + client_id(4) + timestamp(8) + round_id(4) + flags(4)
HEADER_SIZE = 24
LENGTH_PREFIX_SIZE = 8

__all__ = ["MsgType", "ErrorCode", "HEADER_SIZE", "LENGTH_PREFIX_SIZE"]
