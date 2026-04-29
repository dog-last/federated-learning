"""Client base class."""

from src.core.interfaces import IClient


class BaseClient(IClient):
    """Abstract base client with common fields.

    Attributes:
        _client_id: Assigned client ID.
        _connected: Connection status.
    """

    def __init__(self) -> None:
        self._client_id: int = -1
        self._connected: bool = False

    @property
    def client_id(self) -> int:
        """Client ID.

        Returns:
            int: The client ID.
        """
        return self._client_id

    @property
    def is_connected(self) -> bool:
        """Whether the client is connected.

        Returns:
            bool: Connection status.
        """
        return self._connected
