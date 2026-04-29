"""Message data structure."""

from dataclasses import dataclass
from typing import Any

from src.core.interfaces import IMessage
from src.core.types import MsgType


@dataclass
class Message(IMessage):
    """TCP message structure.

    Attributes:
        msg_type: Message type.
        client_id: Client ID (None if not applicable).
        payload: Message payload.
        timestamp: Send timestamp.
        round_id: Round ID (None if not applicable).
    """

    msg_type: MsgType
    client_id: int | None = None
    payload: Any = None
    timestamp: float = 0.0
    round_id: int | None = None

    def __post_init__(self) -> None:
        import time

        if self.timestamp == 0.0:
            self.timestamp = time.time()
