"""Server base class."""

from src.core.interfaces import IServer
from src.core.types import Weights


class BaseServer(IServer):
    """Abstract base server with common fields.

    Attributes:
        _global_weights: Current global model weights.
        _running: Whether the server is running.
    """

    def __init__(self) -> None:
        self._global_weights: Weights = {}
        self._running: bool = False

    @property
    def global_weights(self) -> Weights:
        """Current global model weights.

        Returns:
            Weights: Global model weights.
        """
        return self._global_weights
