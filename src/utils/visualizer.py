"""Visualization utilities (thin wrapper around MetricsCollector plotting)."""

from src.utils.metrics import MetricsCollector


def plot_accuracy(metrics: MetricsCollector, path: str = "outputs/figures/accuracy.png") -> None:
    """Plot and save accuracy curve.

    Args:
        metrics: Metrics collector with recorded data.
        path: Output file path.
    """
    metrics.plot_accuracy(path)


def plot_loss(metrics: MetricsCollector, path: str = "outputs/figures/loss.png") -> None:
    """Plot and save loss curve.

    Args:
        metrics: Metrics collector with recorded data.
        path: Output file path.
    """
    metrics.plot_loss(path)
