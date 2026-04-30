"""Unit tests for monitor_api module (pure functions only, no server startup)."""
import asyncio
import copy
import os
import pytest
import shutil
import tempfile
from unittest.mock import patch

from utils.monitor_api import (
    _fmt_float, _fmt_seconds, _fmt_int, _fmt_bytes, _fmt_rate,
    _apply_config_patch, _editable_field_schema,
    _update_summary, _summary_snapshot, _clear_monitor_state,
    _get_state_snapshot, _get_metrics_history, _collect_metrics,
    _save_charts_to_output,
    _append_event_sync,
    PROJECT_ROOT,
    progress_renderer,
)


def test_fmt_float():
    assert _fmt_float(3.14159) == "3.1416"
    assert _fmt_float(None) == "-"
    assert _fmt_float(0.0) == "0.0000"


def test_fmt_seconds():
    assert _fmt_seconds(1.234) == "1.234s"
    assert _fmt_seconds(None) == "-"
    assert _fmt_seconds(0.0) == "0.000s"


def test_fmt_int():
    assert _fmt_int(42) == "42"
    assert _fmt_int(None) == "-"


def test_fmt_bytes():
    assert _fmt_bytes(512) == "512.0B"
    assert _fmt_bytes(1024) == "1.0KB"
    assert _fmt_bytes(1048576) == "1.0MB"
    assert _fmt_bytes(1073741824) == "1.0GB"
    assert _fmt_bytes(None) == "-"


def test_fmt_rate():
    assert _fmt_rate(3.5) == "3.50/s"
    assert _fmt_rate(None) == "-"


def test_editable_field_schema():
    schema = _editable_field_schema()
    assert "experiment" in schema
    assert "network" in schema
    assert "mode" in schema["experiment"]


class TestApplyConfigPatch:
    def test_patch_mode(self):
        base = {"experiment": {"mode": "centralized"}}
        result = _apply_config_patch(base, {"experiment": {"mode": "splitfed"}})
        assert result["experiment"]["mode"] == "splitfed"

    def test_patch_invalid_mode(self):
        with pytest.raises(ValueError, match="centralized or splitfed"):
            _apply_config_patch({"experiment": {"mode": "centralized"}}, {"experiment": {"mode": "invalid"}})

    def test_patch_global_epochs(self):
        result = _apply_config_patch({"experiment": {}}, {"experiment": {"global_epochs": 20}})
        assert result["experiment"]["global_epochs"] == 20

    def test_patch_invalid_epochs(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"global_epochs": 0}})

    def test_patch_target_accuracy(self):
        result = _apply_config_patch({"experiment": {}}, {"experiment": {"target_accuracy": 0.95}})
        assert result["experiment"]["target_accuracy"] == 0.95

    def test_patch_invalid_accuracy(self):
        with pytest.raises(ValueError, match="must be in"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"target_accuracy": 2.0}})

    def test_patch_optimization(self):
        result = _apply_config_patch({"experiment": {}}, {
            "experiment": {"optimization": {"client_lr": 0.02, "momentum": 0.5}}
        })
        assert result["experiment"]["optimization"]["client_lr"] == 0.02

    def test_patch_invalid_optimization_key(self):
        with pytest.raises(ValueError, match="Unsupported optimization"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"optimization": {"bad_key": 1.0}}})

    def test_patch_optimization_invalid_lr(self):
        with pytest.raises(ValueError, match="must be > 0"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"optimization": {"client_lr": -1}}})

    def test_patch_optimization_invalid_momentum(self):
        with pytest.raises(ValueError, match="momentum must be in"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"optimization": {"momentum": 2.0}}})

    def test_patch_optimization_invalid_weight_decay(self):
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"optimization": {"weight_decay": -0.1}}})

    def test_patch_dataset_params(self):
        result = _apply_config_patch({"experiment": {}}, {"experiment": {"dataset_params": {"batch_size": 128}}})
        assert result["experiment"]["dataset_params"]["batch_size"] == 128

    def test_patch_invalid_dataset_key(self):
        with pytest.raises(ValueError, match="Unsupported dataset_params"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"dataset_params": {"bad_key": 1}}})

    def test_patch_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"dataset_params": {"batch_size": 0}}})

    def test_patch_unsupported_experiment_key(self):
        with pytest.raises(ValueError, match="Unsupported experiment"):
            _apply_config_patch({"experiment": {}}, {"experiment": {"bad_key": "value"}})

    def test_patch_network_timeout(self):
        result = _apply_config_patch({"network": {}}, {"network": {"server_timeout": 60.0}})
        assert result["network"]["server_timeout"] == 60.0

    def test_patch_invalid_timeout(self):
        with pytest.raises(ValueError, match="server_timeout must be > 0"):
            _apply_config_patch({"network": {}}, {"network": {"server_timeout": 0}})

    def test_patch_stragglers(self):
        result = _apply_config_patch({"network": {"stragglers": {}}}, {
            "network": {"stragglers": {"client_2": {"delay": 1.5, "drop_rate": 0.3}}}
        })
        assert result["network"]["stragglers"]["client_2"]["delay"] == 1.5

    def test_patch_invalid_straggler_delay(self):
        with pytest.raises(ValueError, match="straggler delay must be >= 0"):
            _apply_config_patch({"network": {"stragglers": {}}}, {
                "network": {"stragglers": {"c1": {"delay": -1.0}}}
            })

    def test_patch_invalid_straggler_drop_rate(self):
        with pytest.raises(ValueError, match="straggler drop_rate must be in"):
            _apply_config_patch({"network": {"stragglers": {}}}, {
                "network": {"stragglers": {"c1": {"drop_rate": 2.0}}}
            })

    def test_patch_invalid_straggler_key(self):
        with pytest.raises(ValueError, match="Unsupported straggler"):
            _apply_config_patch({"network": {"stragglers": {}}}, {
                "network": {"stragglers": {"c1": {"bad_key": 1.0}}}
            })

    def test_patch_unsupported_network_key(self):
        with pytest.raises(ValueError, match="Unsupported network"):
            _apply_config_patch({"network": {}}, {"network": {"bad_key": "value"}})

    def test_patch_unsupported_top_level_key(self):
        with pytest.raises(ValueError, match="Unsupported top-level"):
            _apply_config_patch({}, {"bad_top": {}})

    def test_patch_does_not_mutate_original(self):
        base = {"experiment": {"mode": "centralized"}}
        original = copy.deepcopy(base)
        _apply_config_patch(base, {"experiment": {"global_epochs": 5}})
        assert base == original


class TestSummary:
    def setup_method(self):
        _clear_monitor_state()

    def test_update_summary_out(self):
        _update_summary({"source": "client_1", "event_type": "send", "direction": "out", "payload_bytes": 100, "payload_label": "model_weights"})
        snap = _summary_snapshot()
        assert snap["total_events"] == 1
        assert snap["network"]["bytes_sent"] == 100
        assert snap["network"]["bytes_recv"] == 0

    def test_update_summary_in(self):
        _update_summary({"source": "server", "event_type": "recv", "direction": "in", "payload_bytes": 200, "payload_label": "update"})
        snap = _summary_snapshot()
        assert snap["network"]["bytes_recv"] == 200

    def test_summary_snapshot_empty(self):
        snap = _summary_snapshot()
        assert snap["total_events"] == 0
        assert snap["last_event"] is None

    def test_clear_monitor_state(self):
        _update_summary({"source": "test", "event_type": "test", "direction": "out", "payload_bytes": 50, "payload_label": "test"})
        _clear_monitor_state()
        snap = _summary_snapshot()
        assert snap["total_events"] == 0


class TestProgressRenderer:
    def test_progress_bar(self):
        result = progress_renderer._progress_bar(0.5, 20)
        assert "50.0%" in result

    def test_progress_bar_boundaries(self):
        assert "0.0%" in progress_renderer._progress_bar(0.0)
        assert "100.0%" in progress_renderer._progress_bar(1.0)

    def test_avg(self):
        assert progress_renderer._avg([1.0, 2.0, 3.0]) == 2.0
        assert progress_renderer._avg([]) is None

    def test_reset(self):
        progress_renderer.current_round = 5
        progress_renderer.phase = "test"
        progress_renderer.reset()
        assert progress_renderer.current_round == 0
        assert progress_renderer.phase == "idle"

    def test_ensure_client_state(self):
        progress_renderer.reset()
        progress_renderer._ensure_client_state("client_1")
        assert "client_1" in progress_renderer.client_states
        assert progress_renderer.client_states["client_1"]["status"] == "idle"

    def test_ensure_client_state_empty(self):
        progress_renderer._ensure_client_state("")
        # Should not add empty client_id

    def test_handle_network_io(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "network_io",
            "source": "client_1",
            "bytes_sent_total": 100,
            "bytes_recv_total": 50,
        })
        assert progress_renderer.total_bytes_sent >= 100

    def test_handle_round_start(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "round_start",
            "mode": "centralized",
            "round": 1,
            "total_epochs": 5,
            "expected_clients": 2,
        })
        assert progress_renderer.current_round == 1
        assert progress_renderer.phase == "broadcast"

    def test_handle_batch_progress(self):
        progress_renderer.reset()
        progress_renderer.known_clients = ["client_1"]
        progress_renderer._ensure_client_state("client_1")
        progress_renderer.handle({
            "event_type": "batch_progress",
            "client_id": "client_1",
            "total_batches": 10,
            "batch_idx": 5,
            "local_epoch_idx": 1,
            "local_epochs": 2,
            "batch_loss": 0.5,
            "batch_acc": 0.8,
        })
        assert progress_renderer.client_states["client_1"]["status"] == "training"
        assert progress_renderer.client_states["client_1"]["train_loss"] == 0.5

    def test_handle_local_round_done(self):
        progress_renderer.reset()
        progress_renderer._ensure_client_state("client_1")
        progress_renderer.handle({
            "event_type": "local_round_done",
            "client_id": "client_1",
            "round": 1,
            "train_loss": 0.3,
            "train_acc": 0.9,
            "test_acc": 0.85,
        })
        assert progress_renderer.client_states["client_1"]["status"] == "done"
        assert progress_renderer.client_states["client_1"]["test_acc"] == 0.85

    def test_handle_round_end(self):
        progress_renderer.reset()
        progress_renderer.current_round_started_at = 0.0
        progress_renderer.handle({
            "event_type": "round_end",
            "round": 1,
            "test_loss": 0.4,
            "test_acc": 0.88,
        })
        assert progress_renderer.last_round_acc == 0.88

    def test_handle_global_eval_metric(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "metric",
            "source": "server",
            "type": "global_eval",
            "test_loss": 0.3,
            "test_acc": 0.92,
        })
        assert progress_renderer.last_round_acc == 0.92

    def test_handle_client_disconnect(self):
        progress_renderer.reset()
        progress_renderer._ensure_client_state("client_1")
        progress_renderer.active_clients = ["client_1"]
        progress_renderer.handle({
            "event_type": "client_disconnect",
            "client_id": "client_1",
        })
        assert progress_renderer.client_states["client_1"]["status"] == "disconnected"
        assert "client_1" not in progress_renderer.active_clients

    def test_handle_startup(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "startup",
            "client_id": "client_1",
            "mode": "centralized",
            "device": "cpu",
        })
        assert progress_renderer.client_states["client_1"]["status"] == "online"

    def test_handle_topology_update(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "topology_update",
            "active_clients": ["client_1", "client_2"],
            "expected_count": 2,
        })
        assert len(progress_renderer.active_clients) == 2

    def test_handle_shutdown(self):
        progress_renderer.reset()
        progress_renderer.handle({"event_type": "shutdown", "source": "server"})
        assert progress_renderer.phase == "shutdown"

    def test_handle_training_events(self):
        progress_renderer.reset()
        progress_renderer.handle({"event_type": "training_started", "training": {"state": "running"}})
        assert "training started" in progress_renderer.phase

    def test_handle_wait_clients(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "wait_clients",
            "active_clients": ["client_1"],
            "expected_count": 2,
        })
        assert progress_renderer.phase == "waiting clients"

    def test_handle_round_wait_result(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "round_wait_result",
            "received_count": 2,
            "expected_count": 2,
            "wait_seconds": 5.0,
            "timeout_seconds": 30.0,
        })
        assert progress_renderer.phase == "waiting updates"

    def test_handle_round_transport(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "round_transport",
            "elapsed_seconds": 3.0,
        })
        assert progress_renderer.phase == "transport summary"

    def test_handle_manager_start(self):
        progress_renderer.reset()
        progress_renderer.handle({
            "event_type": "manager_start",
            "experiment": {"global_epochs": 10, "mode": "splitfed"},
            "topology": {"clients": [{"id": "c1"}], "client_count": 1},
        })
        assert progress_renderer.epoch_total == 10
        assert progress_renderer.mode == "splitfed"

    def test_load_render_mode_invalid(self):
        with patch.dict('os.environ', {'MONITOR_RENDER_MODE': 'invalid'}):
            assert progress_renderer._load_render_mode() == "auto"

    def test_load_render_mode_from_env(self):
        with patch.dict('os.environ', {'MONITOR_RENDER_MODE': 'plain'}):
            assert progress_renderer._load_render_mode() == "plain"


def test_load_render_mode_web_from_env():
    with patch.dict('os.environ', {'MONITOR_RENDER_MODE': 'web'}):
        assert progress_renderer._load_render_mode() == "web"


def test_web_mode_disables_rich_rendering():
    progress_renderer.reset()
    with patch.dict('os.environ', {'MONITOR_RENDER_MODE': 'web'}):
        progress_renderer.render_mode = progress_renderer._load_render_mode()
        progress_renderer.live_rendering = progress_renderer.render_mode not in {"plain", "web"} and progress_renderer.console.is_terminal
    assert progress_renderer.render_mode == "web"
    assert progress_renderer.live_rendering is False


class TestWebSocketBroadcast:
    def setup_method(self):
        _clear_monitor_state()
        progress_renderer.reset()

    def test_ws_clients_exists(self):
        assert hasattr(progress_renderer, 'ws_clients')
        assert isinstance(progress_renderer.ws_clients, set)

    def test_broadcast_event_no_crash_without_clients(self):
        progress_renderer.render_mode = "web"
        progress_renderer.ws_clients = set()
        progress_renderer._ws_loop = None
        progress_renderer._broadcast_event({"event_type": "test", "source": "unit_test"})

    def test_broadcast_event_with_loop_but_no_clients(self):
        progress_renderer.render_mode = "web"
        progress_renderer.ws_clients = set()
        progress_renderer._ws_loop = asyncio.new_event_loop()
        progress_renderer._broadcast_event({"event_type": "test", "source": "unit_test"})
        progress_renderer._ws_loop.close()
        progress_renderer._ws_loop = None


class TestStateSnapshot:
    def setup_method(self):
        _clear_monitor_state()
        progress_renderer.reset()

    def test_state_snapshot_structure(self):
        progress_renderer.phase = "training"
        progress_renderer.mode = "centralized"
        progress_renderer.current_round = 3
        progress_renderer.epoch_total = 10
        state = _get_state_snapshot()
        assert state["phase"] == "training"
        assert state["mode"] == "centralized"
        assert state["current_round"] == 3
        assert state["epoch_total"] == 10
        assert "clients" in state
        assert "key_events" in state
        assert "source_net_totals" in state

    def test_metrics_history_empty(self):
        history = _get_metrics_history()
        assert history["rounds"] == []
        assert history["train_loss"] == []
        assert "per_client" in history

    def test_metrics_history_collects_round_end(self):
        progress_renderer.render_mode = "web"
        progress_renderer.epoch_total = 3
        _append_event_sync({"event_type": "round_end", "source": "server", "round": 1, "test_loss": 2.3, "test_acc": 0.3, "train_loss": 2.5, "train_acc": 0.2})
        _append_event_sync({"event_type": "round_end", "source": "server", "round": 2, "test_loss": 1.5, "test_acc": 0.5, "train_loss": 1.8, "train_acc": 0.4})
        history = _get_metrics_history()
        assert history["rounds"] == [1, 2]
        assert history["test_loss"] == [2.3, 1.5]
        assert history["test_acc"] == [0.3, 0.5]


class TestWebModeAPI:
    def setup_method(self):
        _clear_monitor_state()
        progress_renderer.reset()
        progress_renderer.render_mode = "web"
        progress_renderer.live_rendering = False

    def teardown_method(self):
        progress_renderer.reset()

    def test_state_snapshot_after_events(self):
        _append_event_sync({"event_type": "manager_start", "source": "manager", "experiment": {"mode": "centralized", "global_epochs": 5}, "topology": {"clients": [{"id": "client_1"}], "client_count": 1}})
        _append_event_sync({"event_type": "round_start", "source": "server", "round": 1, "total_epochs": 5, "expected_clients": 1})
        _append_event_sync({"event_type": "round_end", "source": "server", "round": 1, "test_loss": 2.0, "test_acc": 0.3})
        state = _get_state_snapshot()
        assert state["mode"] == "centralized"
        assert state["current_round"] == 1
        assert state["epoch_total"] == 5

    def test_metrics_history_after_rounds(self):
        # Set epoch_total high enough to avoid auto-save triggering during test
        progress_renderer.epoch_total = 10
        _append_event_sync({"event_type": "round_end", "source": "server", "round": 1, "test_loss": 2.0, "test_acc": 0.3, "train_loss": 2.3, "train_acc": 0.2})
        _append_event_sync({"event_type": "round_end", "source": "server", "round": 2, "test_loss": 1.5, "test_acc": 0.5, "train_loss": 1.8, "train_acc": 0.4})
        history = _get_metrics_history()
        assert history["rounds"] == [1, 2]
        assert history["test_loss"] == [2.0, 1.5]
        assert history["train_acc"] == [0.2, 0.4]

    def test_web_mode_no_rich_rendering(self):
        assert progress_renderer.live_rendering is False

    def test_per_client_metrics(self):
        _append_event_sync({"event_type": "local_round_done", "source": "client_1", "client_id": "client_1", "round": 1, "train_loss": 1.5, "train_acc": 0.6, "test_acc": 0.55})
        history = _get_metrics_history()
        assert "client_1" in history["per_client"]
        assert history["per_client"]["client_1"]["train_loss"] == [1.5]

    def test_metrics_cleared_on_reset(self):
        _append_event_sync({"event_type": "round_end", "source": "server", "round": 1, "test_loss": 2.0, "test_acc": 0.3})
        _clear_monitor_state()
        history = _get_metrics_history()
        assert history["rounds"] == []

    def test_auto_save_on_training_end(self):
        output_dir = tempfile.mkdtemp()
        try:
            progress_renderer.epoch_total = 2
            _append_event_sync({"event_type": "round_end", "source": "server", "round": 1, "test_loss": 2.0, "test_acc": 0.3, "train_loss": 2.3, "train_acc": 0.2})
            _append_event_sync({"event_type": "round_end", "source": "server", "round": 2, "test_loss": 1.0, "test_acc": 0.7, "train_loss": 1.2, "train_acc": 0.6})
            _save_charts_to_output(output_dir)
            assert os.path.exists(os.path.join(output_dir, "loss_curve.png"))
            assert os.path.exists(os.path.join(output_dir, "accuracy_curve.png"))
            assert os.path.exists(os.path.join(output_dir, "metrics.json"))
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
