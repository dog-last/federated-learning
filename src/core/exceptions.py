"""Custom exceptions for the federated learning system."""


class FedError(Exception):
    """Base exception for the federated learning system."""


class NetworkTimeoutError(FedError):
    """Network timeout exception.

    Args:
        operation: The network operation that timed out.
        timeout: The timeout value in seconds.
    """

    def __init__(self, operation: str, timeout: float) -> None:
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"{operation} timed out after {timeout}s")


class ConnectionError(FedError):
    """Connection error."""

    def __init__(self, message: str = "Connection error") -> None:
        super().__init__(message)


class ProtocolError(FedError):
    """Protocol encoding/decoding error."""

    def __init__(self, message: str = "Protocol error") -> None:
        super().__init__(message)


class AggregationError(FedError):
    """Aggregation error."""

    def __init__(self, message: str = "Aggregation error") -> None:
        super().__init__(message)


class ConfigError(FedError):
    """Configuration error."""

    def __init__(self, message: str = "Configuration error") -> None:
        super().__init__(message)
