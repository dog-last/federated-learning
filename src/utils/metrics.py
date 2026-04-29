"""Metrics collector and visualizer."""

import json
from pathlib import Path
from typing import Any

from src.core.interfaces import IMetricsCollector
from src.core.types import RoundStats


class MetricsCollector(IMetricsCollector):
    """Collects and exports training metrics.

    Attributes:
        _accuracy_history: Accuracy per round.
        _loss_history: Loss per round.
        _round_times: Duration per round.
        _round_stats: Full RoundStats per round.
    """

    def __init__(self) -> None:
        self._accuracy_history: list[float] = []
        self._loss_history: list[float] = []
        self._round_times: list[float] = []
        self._round_stats: list[RoundStats] = []

    def record_round(self, stats: RoundStats) -> None:
        """Record round statistics.

        Args:
            stats: Round statistics.
        """
        self._round_stats.append(stats)
        self._accuracy_history.append(stats.global_accuracy)
        self._loss_history.append(stats.global_loss)
        self._round_times.append(stats.total_time)

    def get_accuracy_history(self) -> list[float]:
        """Get accuracy history.

        Returns:
            List[float]: Accuracy per round.
        """
        return list(self._accuracy_history)

    def get_loss_history(self) -> list[float]:
        """Get loss history.

        Returns:
            List[float]: Loss per round.
        """
        return list(self._loss_history)

    def get_round_times(self) -> list[float]:
        """Get round durations.

        Returns:
            List[float]: Duration per round.
        """
        return list(self._round_times)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dict[str, Any]: Summary dictionary.
        """
        if not self._accuracy_history:
            return {}
        return {
            "total_rounds": len(self._accuracy_history),
            "final_accuracy": self._accuracy_history[-1],
            "best_accuracy": max(self._accuracy_history),
            "avg_round_time": sum(self._round_times) / len(self._round_times),
            "total_time": sum(self._round_times),
            "convergence_round": self._find_convergence(),
        }

    def _find_convergence(self, threshold: float = 0.001) -> int:
        """Find the convergence round.

        Args:
            threshold: Accuracy change threshold for convergence.

        Returns:
            int: Convergence round index (1-based).
        """
        for i in range(1, len(self._accuracy_history)):
            if abs(self._accuracy_history[i] - self._accuracy_history[i - 1]) < threshold:
                return i
        return len(self._accuracy_history)

    def export(self, path: str) -> None:
        """Export metrics to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "accuracy_history": self._accuracy_history,
            "loss_history": self._loss_history,
            "round_times": self._round_times,
            "summary": self.get_summary(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def plot_accuracy(self, path: str) -> None:
        """Plot accuracy curve and save.

        Args:
            path: Output file path for the plot.
        """
        try:
            import matplotlib.pyplot as plt

            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self._accuracy_history) + 1), self._accuracy_history)
            plt.xlabel("Round")
            plt.ylabel("Accuracy (%)")
            plt.title("Global Model Accuracy")
            plt.grid(True)
            plt.savefig(path)
            plt.close()
        except ImportError:
            pass

    def plot_loss(self, path: str) -> None:
        """Plot loss curve and save.

        Args:
            path: Output file path for the plot.
        """
        try:
            import matplotlib.pyplot as plt

            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self._loss_history) + 1), self._loss_history)
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)
            plt.savefig(path)
            plt.close()
        except ImportError:
            pass
