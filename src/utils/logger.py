"""Structured logger for the federated learning system."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.interfaces import ILogger
from src.core.types import RoundStats


class FedLogger(ILogger):
    """Federated learning logger with console and file output.

    Attributes:
        log_dir: Directory for log files.
        logger: Underlying Python logger.
    """

    def __init__(
        self,
        name: str = "FedLearning",
        level: str = "INFO",
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()

        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        if console_output:
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, level.upper(), logging.INFO))
            ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            self.logger.addHandler(ch)

        if file_output:
            date_str = datetime.now().strftime("%Y%m%d")
            fh = logging.FileHandler(self.log_dir / f"fed_{date_str}.log")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
                    datefmt=datefmt,
                )
            )
            self.logger.addHandler(fh)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log an INFO message."""
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log a WARNING message."""
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log an ERROR message."""
        self.logger.error(msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log a DEBUG message."""
        self.logger.debug(msg, **kwargs)

    def log_round(self, stats: RoundStats) -> None:
        """Log round statistics.

        Args:
            stats: Round statistics.
        """
        self.info(f"[Round {stats.round_id}] Summary:")
        self.info(f"  - Participating clients: {stats.participating_clients}")
        if stats.timeout_clients:
            self.info(f"  - Timeout clients: {stats.timeout_clients}")
        self.info(f"  - Broadcast time: {stats.broadcast_time:.3f}s")
        for cid, t in stats.collect_times.items():
            self.info(f"  - Client-{cid} recv time: {t:.3f}s")
        self.info(f"  - Aggregate time: {stats.aggregate_time:.3f}s")
        self.info(f"  - Total time: {stats.total_time:.3f}s")
        self.info(f"  - Global accuracy: {stats.global_accuracy:.2f}%")
        self.info(f"  - Global loss: {stats.global_loss:.4f}")

    def log_network(
        self,
        event: str,
        client_id: int | None = None,
        size: int | None = None,
        duration: float | None = None,
        success: bool = True,
    ) -> None:
        """Log a network event.

        Args:
            event: Event type.
            client_id: Client ID.
            size: Data size in bytes.
            duration: Duration in seconds.
            success: Whether the event succeeded.
        """
        client_str = f"Client-{client_id}" if client_id else "Unknown"
        status = "SUCCESS" if success else "FAILED"
        parts = [f"[Network] {event.upper()} {client_str} {status}"]
        if size is not None:
            parts.append(f"Size: {self._format_size(size)}")
        if duration is not None:
            parts.append(f"Time: {duration * 1000:.1f}ms")
        msg = " | ".join(parts)
        if success:
            self.info(msg)
        else:
            self.warning(msg)

    def log_training(
        self,
        client_id: int,
        round_id: int,
        loss: float,
        accuracy: float,
        duration: float,
    ) -> None:
        """Log a training event.

        Args:
            client_id: Client ID.
            round_id: Round ID.
            loss: Training loss.
            accuracy: Training accuracy.
            duration: Training duration.
        """
        self.info(
            f"[Round {round_id}] Client-{client_id} training complete. "
            f"Loss: {loss:.4f}, Acc: {accuracy:.2f}%, Time: {duration:.2f}s"
        )

    @staticmethod
    def _format_size(size: float) -> str:
        """Format a byte count for human reading.

        Args:
            size: Size in bytes.

        Returns:
            str: Formatted size string.
        """
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


def get_logger(**kwargs: Any) -> FedLogger:
    """Create a FedLogger.

    Args:
        **kwargs: Keyword arguments forwarded to FedLogger.

    Returns:
        FedLogger: A new logger instance.
    """
    return FedLogger(**kwargs)
