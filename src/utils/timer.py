"""Timer utility for measuring elapsed time."""

import time


class Timer:
    """Simple context-manager timer.

    Attributes:
        start_time: Timestamp when the timer started.
        elapsed: Elapsed time in seconds.
    """

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.time() - self.start_time

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns:
            float: Elapsed time in seconds.
        """
        self.elapsed = time.time() - self.start_time
        return self.elapsed
