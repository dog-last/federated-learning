"""Unit tests for custom exceptions."""

from src.core.exceptions import (
    AggregationError,
    ConfigError,
    ConnectionError,
    FedError,
    NetworkTimeoutError,
    ProtocolError,
)


class TestFedError:
    """Tests for base FedError."""

    def test_is_exception(self) -> None:
        assert issubclass(FedError, Exception)

    def test_message(self) -> None:
        e = FedError("test")
        assert str(e) == "test"


class TestNetworkTimeoutError:
    """Tests for NetworkTimeoutError."""

    def test_attributes(self) -> None:
        e = NetworkTimeoutError("send", 30.0)
        assert e.operation == "send"
        assert e.timeout == 30.0

    def test_message(self) -> None:
        e = NetworkTimeoutError("recv", 10.0)
        assert "recv" in str(e)
        assert "10" in str(e)

    def test_is_fed_error(self) -> None:
        assert issubclass(NetworkTimeoutError, FedError)


class TestConnectionError:
    """Tests for ConnectionError."""

    def test_default_message(self) -> None:
        e = ConnectionError()
        assert str(e) == "Connection error"

    def test_custom_message(self) -> None:
        e = ConnectionError("closed")
        assert str(e) == "closed"


class TestProtocolError:
    """Tests for ProtocolError."""

    def test_default_message(self) -> None:
        e = ProtocolError()
        assert "Protocol" in str(e)


class TestAggregationError:
    """Tests for AggregationError."""

    def test_message(self) -> None:
        e = AggregationError("empty list")
        assert "empty list" in str(e)


class TestConfigError:
    """Tests for ConfigError."""

    def test_message(self) -> None:
        e = ConfigError("bad yaml")
        assert "bad yaml" in str(e)
