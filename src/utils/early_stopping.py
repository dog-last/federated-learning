"""Early stopping utility for federated learning."""

from src.core.types import EarlyStoppingConfig


class EarlyStopping:
    """Early stopping handler.

    Monitors a metric and stops training when it stops improving.

    Attributes:
        _config: Early stopping configuration.
        _best_value: Best metric value seen so far.
        _counter: Number of rounds without improvement.
        _should_stop: Whether training should stop.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        """Initialize early stopping.

        Args:
            config: Early stopping configuration.
        """
        self._config = config
        self._best_value: float | None = None
        self._counter = 0
        self._should_stop = False

    def __call__(self, metric_value: float) -> bool:
        """Check if training should stop.

        Args:
            metric_value: Current metric value (accuracy or loss).

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if not self._config.enabled:
            return False

        if self._best_value is None:
            self._best_value = metric_value
            return False

        if self._config.mode == "max":
            # For accuracy: higher is better
            improved = metric_value > self._best_value + self._config.min_delta
        else:
            # For loss: lower is better
            improved = metric_value < self._best_value - self._config.min_delta

        if improved:
            self._best_value = metric_value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self._config.patience:
                self._should_stop = True

        return self._should_stop

    @property
    def should_stop(self) -> bool:
        """Whether training should stop.

        Returns:
            bool: True if training should stop.
        """
        return self._should_stop

    @property
    def best_value(self) -> float | None:
        """Best metric value seen so far.

        Returns:
            Optional[float]: Best value or None if not set.
        """
        return self._best_value

    @property
    def counter(self) -> int:
        """Number of rounds without improvement.

        Returns:
            int: Counter value.
        """
        return self._counter

    def reset(self) -> None:
        """Reset early stopping state."""
        self._best_value = None
        self._counter = 0
        self._should_stop = False
