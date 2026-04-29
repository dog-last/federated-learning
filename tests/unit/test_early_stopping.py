"""Unit tests for early stopping utility."""

from src.core.types import EarlyStoppingConfig
from src.utils.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Tests for EarlyStopping class."""

    def test_disabled_early_stopping(self) -> None:
        """Test that disabled early stopping always returns False."""
        config = EarlyStoppingConfig(enabled=False, patience=3, mode="max")
        es = EarlyStopping(config)
        # Even with many calls, should always return False
        for i in range(10):
            assert es(0.5 + i * 0.1) is False

    def test_first_call_sets_best_value(self) -> None:
        """Test that first call initializes best_value."""
        config = EarlyStoppingConfig(enabled=True, patience=2, mode="max")
        es = EarlyStopping(config)
        result = es(0.5)
        assert result is False
        assert es.best_value == 0.5

    def test_max_mode_improvement(self) -> None:
        """Test max mode detects improvement correctly."""
        config = EarlyStoppingConfig(enabled=True, patience=2, mode="max", min_delta=0.01)
        es = EarlyStopping(config)
        # First value
        es(0.5)
        # Significant improvement
        result = es(0.6)
        assert result is False
        assert es.best_value == 0.6
        assert es.counter == 0  # Counter reset

    def test_max_mode_no_improvement(self) -> None:
        """Test max mode detects lack of improvement."""
        config = EarlyStoppingConfig(enabled=True, patience=2, mode="max", min_delta=0.01)
        es = EarlyStopping(config)
        es(0.5)
        # No improvement (below min_delta)
        result = es(0.505)
        assert result is False
        assert es.counter == 1

    def test_min_mode_improvement(self) -> None:
        """Test min mode detects improvement correctly."""
        config = EarlyStoppingConfig(enabled=True, patience=2, mode="min", min_delta=0.01)
        es = EarlyStopping(config)
        # First value (loss)
        es(1.0)
        # Significant improvement (lower loss)
        result = es(0.8)
        assert result is False
        assert es.best_value == 0.8
        assert es.counter == 0  # Counter reset

    def test_min_mode_no_improvement(self) -> None:
        """Test min mode detects lack of improvement."""
        config = EarlyStoppingConfig(enabled=True, patience=2, mode="min", min_delta=0.01)
        es = EarlyStopping(config)
        es(1.0)
        # No improvement (below min_delta)
        result = es(0.99)
        assert result is False
        assert es.counter == 1

    def test_patience_reached_should_stop(self) -> None:
        """Test that should_stop becomes True when patience is reached."""
        config = EarlyStoppingConfig(enabled=True, patience=3, mode="max", min_delta=0.01)
        es = EarlyStopping(config)
        # First value - sets best_value
        es(0.5)
        assert es.best_value == 0.5
        # No improvement for patience rounds
        # 0.5 is not > 0.5 + 0.01 (0.51), so no improvement
        result = es(0.5)  # Same value, no improvement - counter=1
        assert result is False
        assert es.counter == 1
        result = es(0.5)  # counter=2
        assert es.counter == 2
        result = es(0.5)  # counter=3, should_stop=True
        assert es.should_stop is True
        assert result is True  # Now returns True because should_stop is True

    def test_should_stop_property(self) -> None:
        """Test should_stop property."""
        config = EarlyStoppingConfig(enabled=True, patience=1, mode="max")
        es = EarlyStopping(config)
        assert es.should_stop is False
        es(0.5)
        es(0.5)  # No improvement
        assert es.should_stop is True

    def test_best_value_property(self) -> None:
        """Test best_value property."""
        config = EarlyStoppingConfig(enabled=True, patience=3, mode="max")
        es = EarlyStopping(config)
        assert es.best_value is None
        es(0.5)
        assert es.best_value == 0.5
        es(0.7)
        assert es.best_value == 0.7

    def test_counter_property(self) -> None:
        """Test counter property."""
        config = EarlyStoppingConfig(enabled=True, patience=3, mode="max", min_delta=0.001)
        es = EarlyStopping(config)
        assert es.counter == 0
        es(0.5)  # Sets best_value
        assert es.counter == 0
        es(0.5)  # Same value, no improvement (not > 0.501)
        assert es.counter == 1
        es(0.5005)  # Still no improvement
        assert es.counter == 2
        es(0.6)  # Improvement (0.6 > 0.501) - reset
        assert es.counter == 0

    def test_reset(self) -> None:
        """Test reset method."""
        config = EarlyStoppingConfig(enabled=True, patience=3, mode="max", min_delta=0.01)
        es = EarlyStopping(config)
        es(0.5)  # Sets best_value = 0.5
        es(0.5)  # No improvement (0.5 is not > 0.51), counter = 1
        assert es.best_value == 0.5
        assert es.counter == 1
        assert es.should_stop is False

        es.reset()

        assert es.best_value is None
        assert es.counter == 0
        assert es.should_stop is False
