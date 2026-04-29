"""Unit tests for utils: logger, metrics, timer, config_loader, visualizer."""

import json
import time
from pathlib import Path

from src.core.types import RoundStats
from src.utils.config_loader import load_config
from src.utils.logger import FedLogger, get_logger
from src.utils.metrics import MetricsCollector
from src.utils.timer import Timer
from src.utils.visualizer import plot_accuracy, plot_loss


class TestTimer:
    """Tests for Timer."""

    def test_context_manager(self) -> None:
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04

    def test_start_stop(self) -> None:
        t = Timer()
        t.start()
        time.sleep(0.05)
        elapsed = t.stop()
        assert elapsed >= 0.04
        assert t.elapsed == elapsed

    def test_zero_duration(self) -> None:
        t = Timer()
        t.start()
        elapsed = t.stop()
        assert elapsed < 0.1


class TestFedLogger:
    """Tests for FedLogger."""

    def test_creation(self, tmp_dir: str) -> None:
        logger = FedLogger(name="test", log_dir=tmp_dir, console_output=False, file_output=True)
        logger.info("hello")
        # Should have created a log file
        log_files = list(Path(tmp_dir).glob("fed_*.log"))
        assert len(log_files) == 1

    def test_log_levels(self, tmp_dir: str) -> None:
        logger = FedLogger(
            name="test", level="DEBUG", log_dir=tmp_dir, console_output=False, file_output=True
        )
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")

    def test_log_round(self, tmp_dir: str) -> None:
        logger = FedLogger(name="test", log_dir=tmp_dir, console_output=False, file_output=True)
        stats = RoundStats(
            round_id=1, broadcast_time=0.1, participating_clients=[1, 2], global_accuracy=50.0
        )
        logger.log_round(stats)

    def test_log_network(self, tmp_dir: str) -> None:
        logger = FedLogger(name="test", log_dir=tmp_dir, console_output=False, file_output=True)
        logger.log_network("send", client_id=1, size=1024, duration=0.5, success=True)
        logger.log_network("recv", client_id=2, success=False)

    def test_log_training(self, tmp_dir: str) -> None:
        logger = FedLogger(name="test", log_dir=tmp_dir, console_output=False, file_output=True)
        logger.log_training(client_id=1, round_id=1, loss=0.5, accuracy=85.0, duration=3.2)

    def test_get_logger(self) -> None:
        logger = get_logger(name="test2", console_output=False, file_output=False)
        assert isinstance(logger, FedLogger)

    def test_format_size(self) -> None:
        assert FedLogger._format_size(100) == "100.0B"
        assert FedLogger._format_size(2048) == "2.0KB"
        assert FedLogger._format_size(1048576) == "1.0MB"

    def test_console_only(self) -> None:
        logger = FedLogger(name="test", console_output=True, file_output=False)
        logger.info("console only")

    def test_no_output(self) -> None:
        logger = FedLogger(name="test", console_output=False, file_output=False)
        logger.info("nowhere")


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_and_get(self) -> None:
        mc = MetricsCollector()
        mc.record_round(
            RoundStats(
                round_id=1,
                broadcast_time=0.1,
                total_time=1.0,
                global_accuracy=50.0,
                global_loss=1.5,
            )
        )
        mc.record_round(
            RoundStats(
                round_id=2,
                broadcast_time=0.1,
                total_time=0.9,
                global_accuracy=70.0,
                global_loss=0.8,
            )
        )

        assert mc.get_accuracy_history() == [50.0, 70.0]
        assert mc.get_loss_history() == [1.5, 0.8]
        assert mc.get_round_times() == [1.0, 0.9]

    def test_summary(self) -> None:
        mc = MetricsCollector()
        mc.record_round(
            RoundStats(round_id=1, broadcast_time=0.1, total_time=1.0, global_accuracy=50.0)
        )
        mc.record_round(
            RoundStats(round_id=2, broadcast_time=0.1, total_time=0.8, global_accuracy=80.0)
        )

        s = mc.get_summary()
        assert s["total_rounds"] == 2
        assert s["final_accuracy"] == 80.0
        assert s["best_accuracy"] == 80.0
        assert s["total_time"] == 1.8

    def test_summary_empty(self) -> None:
        mc = MetricsCollector()
        assert mc.get_summary() == {}

    def test_export(self, tmp_dir: str) -> None:
        mc = MetricsCollector()
        mc.record_round(
            RoundStats(round_id=1, broadcast_time=0.1, total_time=1.0, global_accuracy=50.0)
        )

        path = str(Path(tmp_dir) / "metrics.json")
        mc.export(path)

        data = json.loads(Path(path).read_text())
        assert "accuracy_history" in data
        assert "summary" in data

    def test_plot_accuracy(self, tmp_dir: str) -> None:
        mc = MetricsCollector()
        mc.record_round(
            RoundStats(round_id=1, broadcast_time=0.1, total_time=1.0, global_accuracy=50.0)
        )
        mc.record_round(
            RoundStats(round_id=2, broadcast_time=0.1, total_time=1.0, global_accuracy=80.0)
        )

        path = str(Path(tmp_dir) / "acc.png")
        mc.plot_accuracy(path)
        # If matplotlib is installed, file should exist
        try:
            import matplotlib  # noqa: F401

            assert Path(path).exists()
        except ImportError:
            pass

    def test_plot_loss(self, tmp_dir: str) -> None:
        mc = MetricsCollector()
        mc.record_round(
            RoundStats(
                round_id=1,
                broadcast_time=0.1,
                total_time=1.0,
                global_accuracy=50.0,
                global_loss=1.0,
            )
        )

        path = str(Path(tmp_dir) / "loss.png")
        mc.plot_loss(path)


class TestConfigLoader:
    """Tests for config_loader."""

    def test_load_config(self, tmp_yaml: str) -> None:
        config = load_config(tmp_yaml)
        assert config.model.name == "simple_cnn"
        assert config.dataset.num_clients == 2


class TestVisualizer:
    """Tests for visualizer."""

    def test_plot_accuracy(self, tmp_dir: str) -> None:
        mc = MetricsCollector()
        mc.record_round(
            RoundStats(round_id=1, broadcast_time=0.1, total_time=1.0, global_accuracy=50.0)
        )
        path = str(Path(tmp_dir) / "acc.png")
        plot_accuracy(mc, path)

    def test_plot_loss(self, tmp_dir: str) -> None:
        mc = MetricsCollector()
        mc.record_round(
            RoundStats(
                round_id=1,
                broadcast_time=0.1,
                total_time=1.0,
                global_accuracy=50.0,
                global_loss=1.0,
            )
        )
        path = str(Path(tmp_dir) / "loss.png")
        plot_loss(mc, path)
