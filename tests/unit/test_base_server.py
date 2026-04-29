"""Unit tests for server base class."""

from src.server.base import BaseServer


class ConcreteServer(BaseServer):
    """Concrete implementation of BaseServer for testing."""

    def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the server."""
        self._running = True

    def stop(self) -> None:
        """Stop the server."""
        self._running = False

    def run_round(self) -> None:
        """Run one training round."""
        pass

    def run(self, rounds: int) -> None:
        """Run multiple rounds."""
        for _ in range(rounds):
            self.run_round()

    def wait_for_clients(self, min_clients: int, timeout: float | None = None) -> bool:
        """Wait for minimum number of clients."""
        return True


class TestBaseServer:
    """Tests for BaseServer."""

    def test_initial_state(self) -> None:
        """Test initial state of BaseServer."""
        server = ConcreteServer()
        assert server.global_weights == {}
        assert server._running is False

    def test_global_weights_property(self) -> None:
        """Test global_weights property."""
        server = ConcreteServer()
        # Initial state
        assert server.global_weights == {}
        # After setting
        server._global_weights = {"w": [1.0, 2.0]}
        assert server.global_weights == {"w": [1.0, 2.0]}

    def test_running_attribute(self) -> None:
        """Test _running attribute exists and can be modified."""
        server = ConcreteServer()
        assert server._running is False
        server._running = True
        assert server._running is True

    def test_start_and_stop(self) -> None:
        """Test start and stop methods."""
        server = ConcreteServer()
        assert server._running is False
        server.start()
        assert server._running is True
        server.stop()
        assert server._running is False
