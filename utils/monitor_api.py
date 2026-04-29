import asyncio
import copy
import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
try:
    from rich.cells import cell_len as _cell_len
except Exception:
    def _cell_len(value):
        return len(str(value))

from utils.training_controller import TrainingController

logging.basicConfig(level=logging.INFO, format="%(asctime)s - MONITOR - %(levelname)s - %(message)s")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logs = []
lock = threading.Lock()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)

WEB_STATIC_DIR = os.path.join(PROJECT_ROOT, "web", "static")

def _is_web_mode():
    return progress_renderer.render_mode == "web"

summary = {
    "total_events": 0,
    "by_source": defaultdict(int),
    "by_event_type": defaultdict(int),
    "network": {
        "bytes_sent": 0,
        "bytes_recv": 0,
        "by_label": defaultdict(int),
    },
}

metrics_history = {
    "rounds": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "test_loss": [],
    "test_acc": [],
    "per_client": defaultdict(lambda: {"rounds": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": []}),
}

def _fmt_float(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_seconds(value, digits=3):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}s"


def _fmt_int(value):
    if value is None:
        return "-"
    return str(int(value))


def _fmt_bytes(value):
    if value is None:
        return "-"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024.0


def _fmt_rate(value):
    if value is None:
        return "-"
    return f"{float(value):.2f}/s"


from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box

class ProgressRenderer:
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch_total = 1
        self.mode = "-"
        self.current_round = 0
        self.phase = "idle"
        self.expected_clients = 0
        self.active_clients = []
        self.known_clients = []
        self.client_pct: Dict[str, float] = {}
        self.round_durations = deque(maxlen=5)
        self.wait_durations = deque(maxlen=5)
        self.transport_durations = deque(maxlen=5)
        self.current_round_started_at = None
        self.last_round_loss = None
        self.last_round_acc = None

        self.total_bytes_sent = 0
        self.total_bytes_recv = 0
        self.source_net_totals: Dict[str, Dict[str, int]] = {}

        self.client_states: Dict[str, Dict[str, object]] = {}
        self.current_round_pct = 0.0
        self.epoch_progress = 0

        self.last_refresh_at = 0.0
        self.refresh_interval = 0.7
        self.key_events = deque(maxlen=8)

        self.key_log_path = os.path.join(PROJECT_ROOT, "logs", "monitor_key_events.log")
        os.makedirs(os.path.dirname(self.key_log_path), exist_ok=True)
        
        self.render_mode = self._load_render_mode()
        self.console = Console(highlight=False)
        self.live_rendering = self.render_mode not in {"plain", "web"} and self.console.is_terminal
        self.live = None
        self.ws_clients: set = set()
        self._ws_loop: asyncio.AbstractEventLoop | None = None

    def _write(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {line}"
        self.key_events.append(entry)
        with open(self.key_log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")

    def _load_render_mode(self):
        env_mode = os.environ.get("MONITOR_RENDER_MODE")
        if env_mode:
            mode = env_mode.strip().lower()
            return mode if mode in {"auto", "live", "plain", "web"} else "auto"
        try:
            config = _read_config()
        except Exception:
            return "auto"
        mode = str((config.get("monitoring", {}) or {}).get("render_mode", "auto")).strip().lower()
        if mode not in {"auto", "live", "plain", "web"}:
            mode = "auto"
        if mode == "auto" and not self.console.is_terminal:
            mode = "web"
        return mode

    def _avg(self, values):
        if not values:
            return None
        return sum(values) / len(values)

    def _progress_bar(self, ratio: float, width: int = 16) -> str:
        ratio = max(0.0, min(1.0, float(ratio)))
        pct = f" {ratio * 100:5.1f}%"
        bar_width = max(width - len(pct), 1)
        done = int(round(ratio * bar_width))
        return ("[green]" + ("█" * done) + "[/green]") + ("[dim]" + ("░" * max(bar_width - done, 0)) + "[/dim]") + pct

    def _ensure_client_state(self, client_id: str):
        if not client_id:
            return
        if client_id not in self.client_states:
            self.client_states[client_id] = {
                "status": "idle",
                "progress": 0.0,
                "train_loss": None,
                "train_acc": None,
                "test_acc": None,
                "local_epoch": "-",
            }
        if client_id not in self.known_clients:
            self.known_clients.append(client_id)

    def _update_global_totals_from_source(self, source: str, item: Dict[str, object]):
        src = source or "unknown"
        bucket = self.source_net_totals.setdefault(
            src,
            {
                "bytes_sent": 0,
                "bytes_recv": 0,
            },
        )
        pairs = [
            ("bytes_sent_total", "bytes_sent"),
            ("bytes_recv_total", "bytes_recv"),
        ]
        for incoming_key, bucket_key in pairs:
            if incoming_key in item and item.get(incoming_key) is not None:
                value = int(item.get(incoming_key) or 0)
                if value > bucket[bucket_key]:
                    bucket[bucket_key] = value

        self.total_bytes_sent = sum(v["bytes_sent"] for v in self.source_net_totals.values())
        self.total_bytes_recv = sum(v["bytes_recv"] for v in self.source_net_totals.values())

    def _generate_dashboard(self):
        epoch_ratio = self.epoch_progress / max(self.epoch_total, 1)
        if self.client_pct:
            self.current_round_pct = sum(self.client_pct.values()) / len(self.client_pct)

        # MAIN TABLE
        main_table = Table(
            title=f"Monitor Dashboard  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  (refresh={self.refresh_interval:.1f}s)",
            title_justify="left",
            box=box.ROUNDED,
            expand=True
        )
        main_table.add_column("Key", style="cyan", no_wrap=True)
        main_table.add_column("Value", style="none")
        main_table.add_column("Key ", style="cyan", no_wrap=True)
        main_table.add_column("Value ", style="none")

        main_table.add_row("phase", str(self.phase), "mode", str(self.mode))
        main_table.add_row("epoch", f"{self.current_round}/{self.epoch_total} " + self._progress_bar(epoch_ratio, 20), "batch", self._progress_bar(self.current_round_pct, 40))
        main_table.add_row("avg_round", _fmt_seconds(self._avg(self.round_durations)), "avg_wait", _fmt_seconds(self._avg(self.wait_durations)))
        main_table.add_row("avg_xfer", _fmt_seconds(self._avg(self.transport_durations)), "network", f"tx={_fmt_bytes(self.total_bytes_sent)} rx={_fmt_bytes(self.total_bytes_recv)}")
        main_table.add_row("metrics", f"loss={_fmt_float(self.last_round_loss)} acc={_fmt_float(self.last_round_acc)}", "", "")

        # CLIENT TABLE
        client_table = Table(box=box.ROUNDED, expand=True)
        client_table.add_column("id", style="cyan")
        client_table.add_column("status")
        client_table.add_column("progress")
        client_table.add_column("loss", justify="right")
        client_table.add_column("acc", justify="right")
        client_table.add_column("network", justify="right")

        client_ids = [cid for cid in self.known_clients if cid]
        target_rows = max(self.expected_clients, len(client_ids), 1)
        while len(client_ids) < target_rows:
            client_ids.append(f"client_{len(client_ids) + 1}")

        for cid in client_ids[:target_rows]:
            self._ensure_client_state(cid)
            state = self.client_states.get(cid, {})
            net = self.source_net_totals.get(cid, {})
            net_text = f"tx={_fmt_bytes(net.get('bytes_sent', 0))} rx={_fmt_bytes(net.get('bytes_recv', 0))}"
            client_table.add_row(
                cid,
                str(state.get("status", "idle")),
                self._progress_bar(float(state.get("progress", 0.0)), 20),
                _fmt_float(state.get("train_loss")),
                _fmt_float(state.get("train_acc")),
                net_text
            )

        # EVENTS PANEL
        events_str = "\n".join(self.key_events) if self.key_events else "No events yet."
        events_panel = Panel(events_str, title="Recent key events", title_align="left", box=box.ROUNDED, expand=True)

        return Group(main_table, client_table, events_panel)

    def _render_status(self, force: bool = False):
        now = time.time()
        if not force and now - self.last_refresh_at < self.refresh_interval:
            return

        dashboard = self._generate_dashboard()

        if self.live_rendering:
            if self.live is None:
                self.live = Live(dashboard, console=self.console, auto_refresh=False, transient=False)
                self.live.start()
            else:
                self.live.update(dashboard, refresh=True)
        else:
            self.console.print(dashboard)

        self.last_refresh_at = now

    def reset(self):
        with self.lock:
            self.current_round = 0
            self.epoch_total = 1
            self.phase = "idle"
            self.expected_clients = 0
            self.active_clients = []
            self.known_clients = []
            self.client_pct.clear()
            self.round_durations.clear()
            self.wait_durations.clear()
            self.transport_durations.clear()
            self.current_round_started_at = None
            self.last_round_loss = None
            self.last_round_acc = None
            self.total_bytes_sent = 0
            self.total_bytes_recv = 0
            self.source_net_totals.clear()
            self.client_states.clear()
            self.current_round_pct = 0.0
            self.last_refresh_at = 0.0
            self.key_events.clear()
            self.epoch_progress = 0
            if self.live is not None:
                self.live.stop()
                self.live = None

    def _close_client_bars(self):
        return

    def _ensure_epoch_bar(self):
        return

    def _ensure_round_bar(self):
        return

    def _ensure_client_order(self, client_id: str):
        self._ensure_client_state(client_id)

    def _get_or_create_client_bar(self, client_id: str, total_batches: int):
        self._ensure_client_state(client_id)
        return None

    def _set_round_progress(self, value: int, postfix: Dict[str, str] = None):
        return

    def _update_epoch_postfix(self, loss=None, acc=None):
        return

    def handle(self, item):
        event_type = item.get("event_type", "")
        source = str(item.get("source", "") or "unknown")

        if event_type in {"network_io", "send_ack", "recv_ack"}:
            with self.lock:
                self._update_global_totals_from_source(source, item)
                if source.startswith("client"):
                    self._ensure_client_state(source)
                self._render_status()
            return

        with self.lock:
            if event_type == "manager_start":
                exp = item.get("experiment", {}) or {}
                self.epoch_total = max(int(exp.get("global_epochs", self.epoch_total) or 1), 1)
                self.mode = exp.get("mode", self.mode)
                topo = item.get("topology", {}) or {}
                if self.mode == "ring":
                    nodes = topo.get("nodes", [])
                    self.expected_clients = int(topo.get("node_count", len(nodes)) or len(nodes))
                    for n in nodes:
                        nid = n.get("id") if isinstance(n, dict) else None
                        if nid:
                            self._ensure_client_state(f"client_{nid}")
                else:
                    clients = topo.get("clients", [])
                    self.expected_clients = int(topo.get("client_count", len(clients)) or len(clients))
                    for c in clients:
                        cid = c.get("id") if isinstance(c, dict) else None
                        if cid:
                            self._ensure_client_state(cid)
                self._render_status(force=True)
                return

            if event_type == "topology_update":
                listed = [c for c in (item.get("active_clients", []) or []) if c]
                self.active_clients = listed
                self.expected_clients = int(item.get("expected_count", self.expected_clients or 0) or self.expected_clients or 0)
                for cid in listed:
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "connected"
                for cid in self.known_clients:
                    if cid not in listed and self.client_states.get(cid, {}).get("status") == "connected":
                        self.client_states[cid]["status"] = "idle"
                self._render_status(force=True)
                return

            if event_type == "startup":
                node = item.get("client_id") or item.get("source", "unknown")
                self._write(f"[START] {node} online mode={item.get('mode', '-')} device={item.get('device', '-')}")
                if isinstance(node, str) and node.startswith("client"):
                    self._ensure_client_state(node)
                    self.client_states[node]["status"] = "online"
                self._render_status()
                return

            if event_type == "round_start":
                self.mode = item.get("mode", self.mode)
                self.current_round = int(item.get("round", 0) or 0)
                total_epochs = int(item.get("total_epochs") or self.epoch_total or 1)
                self.epoch_total = max(total_epochs, 1)
                self.phase = "broadcast"
                self.current_round_started_at = time.time()
                self.expected_clients = int(item.get("expected_clients", self.expected_clients or 0) or self.expected_clients or 0)
                self.client_pct.clear()
                self.current_round_pct = 0.0
                self.epoch_progress = max(self.current_round - 1, 0)

                for cid in self.known_clients:
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "scheduled"
                    self.client_states[cid]["progress"] = 0.0

                self._write(
                    f"[ROUND] start round={self.current_round}/{self.epoch_total} mode={self.mode} "
                    f"clients={item.get('expected_clients', '-')} local_epochs={item.get('local_epochs', '-')}"
                )
                self._render_status(force=True)
                return

            if event_type == "wait_clients":
                self.phase = "waiting clients"
                self.active_clients = list(item.get("active_clients", []) or [])
                self.expected_clients = int(item.get("expected_count", self.expected_clients or 0) or self.expected_clients or 0)
                for cid in self.active_clients:
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "connected"
                self._render_status(force=True)
                return

            if event_type == "batch_progress":
                client_id = item.get("client_id") or item.get("source", "unknown")
                total_batches = max(int(item.get("total_batches", 1) or 1), 1)
                batch_idx = int(item.get("batch_idx", 0) or 0)
                local_epoch_idx = int(item.get("local_epoch_idx", 0) or 0)
                local_epochs = int(item.get("local_epochs", 0) or 0)

                progress = min(1.0, max(0.0, batch_idx / max(total_batches, 1)))
                self.client_pct[client_id] = progress
                self._ensure_client_state(client_id)
                self.client_states[client_id]["status"] = "training"
                self.client_states[client_id]["progress"] = progress
                self.client_states[client_id]["local_epoch"] = f"{local_epoch_idx}/{max(local_epochs, 1)}"
                self.client_states[client_id]["train_loss"] = item.get("batch_loss")
                self.client_states[client_id]["train_acc"] = item.get("batch_acc")

                if self.client_pct:
                    avg_pct = sum(self.client_pct.values()) / len(self.client_pct)
                    self.current_round_pct = avg_pct
                    self.phase = f"local training {int(avg_pct * 100)}%"
                    self._render_status()
                return

            if event_type == "round_wait_result":
                received = int(item.get("received_count", 0) or 0)
                expected = max(int(item.get("expected_count", 1) or 1), 1)
                self.phase = "waiting updates"
                self.wait_durations.append(float(item.get("wait_seconds", 0.0) or 0.0))
                self._write(
                    f"[ROUND] waiting round={item.get('round', '-')} updates={received}/{expected} "
                    f"wait={_fmt_seconds(item.get('wait_seconds'))} timeout={_fmt_seconds(item.get('timeout_seconds'))}"
                )
                self._render_status(force=True)
                return

            if event_type == "local_round_done":
                client_id = item.get("client_id") or item.get("source", "unknown")
                self.phase = f"client {client_id} done"
                self._ensure_client_state(client_id)
                self.client_states[client_id]["status"] = "done"
                self.client_states[client_id]["progress"] = 1.0
                self.client_states[client_id]["train_loss"] = item.get("train_loss")
                self.client_states[client_id]["train_acc"] = item.get("train_acc")
                self.client_states[client_id]["test_acc"] = item.get("test_acc")
                self._write(
                    f"[CLIENT] round={item.get('round', '-')} {client_id} done "
                    f"train_loss={_fmt_float(item.get('train_loss'))} train_acc={_fmt_float(item.get('train_acc'))} "
                    f"test_acc={_fmt_float(item.get('test_acc'))}"
                )
                self._render_status(force=True)
                return

            if event_type == "metric" and item.get("source") == "server" and item.get("type") == "global_eval":
                self.last_round_loss = item.get("test_loss")
                self.last_round_acc = item.get("test_acc")
                self.phase = "global evaluation"
                self._render_status(force=True)
                return

            if event_type == "round_end":
                round_id = int(item.get("round", 0) or 0)
                self.epoch_progress = min(max(round_id, 0), max(self.epoch_total, 1))
                self.phase = "round complete"
                self.last_round_loss = item.get("test_loss")
                self.last_round_acc = item.get("test_acc")
                self.current_round_pct = 1.0
                if self.current_round_started_at is not None:
                    self.round_durations.append(time.time() - self.current_round_started_at)
                self._write(
                    f"[ROUND] done round={round_id}/{self.epoch_total} loss={_fmt_float(item.get('test_loss'))} "
                    f"acc={_fmt_float(item.get('test_acc'))} avg_round={_fmt_seconds(self._avg(self.round_durations))} "
                    f"sent={_fmt_bytes(self.total_bytes_sent)} recv={_fmt_bytes(self.total_bytes_recv)}"
                )
                self._render_status(force=True)
                return

            if event_type == "round_transport":
                self.phase = "transport summary"
                self.transport_durations.append(float(item.get("elapsed_seconds", 0.0) or 0.0))
                self._render_status()
                return

            if event_type == "client_disconnect":
                client_id = item.get("client_id")
                if client_id:
                    self._ensure_client_state(client_id)
                    self.client_states[client_id]["status"] = "disconnected"
                    self.active_clients = [c for c in self.active_clients if c != client_id]
                self._render_status(force=True)
                return

            # --- Ring mode events ---

            if event_type == "ring_node_startup":
                nid = item.get("node_id")
                if nid is not None:
                    cid = f"client_{nid}"
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "online"
                self._render_status()
                return

            if event_type == "ring_node_ready":
                nid = item.get("node_id")
                if nid is not None:
                    cid = f"client_{nid}"
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "ready"
                    if item.get("is_initiator"):
                        self._write(f"[RING] node {nid} ready (initiator), total_nodes={item.get('total_nodes', '-')}")
                    else:
                        self._write(f"[RING] node {nid} ready, total_nodes={item.get('total_nodes', '-')}")
                self._render_status()
                return

            if event_type == "ring_all_joined":
                self._write(f"[RING] all nodes joined {item.get('joined_count', '-')}/{item.get('expected_count', '-')}")
                self.phase = "ring synchronized"
                self._render_status(force=True)
                return

            if event_type == "ring_round_start":
                round_id = int(item.get("round", 0) or 0)
                nid = item.get("node_id")
                role = item.get("role", "")
                self.current_round = round_id
                self.epoch_progress = round_id - 1
                if role == "initiator":
                    self.current_round_started_at = time.time()
                    self.client_pct.clear()
                    self.current_round_pct = 0.0
                self.phase = f"ring round {round_id}" if role == "initiator" else f"ring round {round_id} waiting"
                if nid is not None:
                    cid = f"client_{nid}"
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "training" if role == "initiator" else "waiting"
                    self.client_states[cid]["progress"] = 0.0
                if role == "initiator":
                    self._write(f"[RING] round {round_id}/{self.epoch_total} start (initiator=node {nid})")
                self._render_status(force=True)
                return

            if event_type == "ring_local_train_done":
                nid = item.get("node_id")
                if nid is not None:
                    cid = f"client_{nid}"
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "trained"
                    self.client_states[cid]["progress"] = 1.0
                    self.client_states[cid]["train_loss"] = item.get("train_loss")
                    self.client_states[cid]["train_acc"] = item.get("train_acc")
                    self.client_states[cid]["test_acc"] = item.get("test_acc")
                self._write(
                    f"[RING] node {nid} local train done round={item.get('round', '-')} "
                    f"loss={_fmt_float(item.get('train_loss'))} acc={_fmt_float(item.get('train_acc'))} "
                    f"val_acc={_fmt_float(item.get('val_acc'))}"
                )
                self.phase = f"ring node {nid} trained"
                self._render_status(force=True)
                return

            if event_type == "ring_send":
                nid = item.get("node_id")
                if nid is not None:
                    cid = f"client_{nid}"
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "sending"
                self._render_status()
                return

            if event_type == "ring_recv":
                nid = item.get("node_id")
                if nid is not None:
                    cid = f"client_{nid}"
                    self._ensure_client_state(cid)
                    self.client_states[cid]["status"] = "receiving"
                self._render_status()
                return

            if event_type == "ring_global_eval":
                round_id = int(item.get("round", 0) or 0)
                self.last_round_loss = item.get("test_loss")
                self.last_round_acc = item.get("test_acc")
                self.phase = "ring global evaluation"
                self._write(
                    f"[RING] round {round_id} global eval loss={_fmt_float(item.get('test_loss'))} "
                    f"acc={_fmt_float(item.get('test_acc'))}"
                )
                self._render_status(force=True)
                return

            if event_type == "ring_round_end":
                round_id = int(item.get("round", 0) or 0)
                self.epoch_progress = min(max(round_id, 0), max(self.epoch_total, 1))
                self.phase = "ring round complete"
                self.current_round_pct = 1.0
                if self.current_round_started_at is not None:
                    self.round_durations.append(time.time() - self.current_round_started_at)
                for cid in self.known_clients:
                    self._ensure_client_state(cid)
                    if self.client_states[cid].get("status") in {"training", "trained", "sending", "receiving", "waiting", "ready"}:
                        self.client_states[cid]["status"] = "done"
                        self.client_states[cid]["progress"] = 1.0
                self._write(
                    f"[RING] round {round_id}/{self.epoch_total} done "
                    f"loss={_fmt_float(self.last_round_loss)} acc={_fmt_float(self.last_round_acc)} "
                    f"avg_round={_fmt_seconds(self._avg(self.round_durations))} "
                    f"sent={_fmt_bytes(self.total_bytes_sent)} recv={_fmt_bytes(self.total_bytes_recv)}"
                )
                self._render_status(force=True)
                return

            if event_type in {"ring_pass_failed", "ring_recv_timeout"}:
                self._write(f"[RING] {event_type} node={item.get('node_id', '-')} round={item.get('round', '-')}")
                self._render_status()
                return

            if event_type in {"training_started", "training_stopped", "training_start_requested", "training_stop_requested"}:
                state = (item.get("training", {}) or {}).get("state", "-")
                self.phase = event_type.replace("_", " ")
                self._write(f"[CTRL] {event_type.replace('_', ' ')} state={state}")
                self._render_status(force=True)
                return

            if event_type in {"target_reached", "manager_stop", "shutdown"}:
                label = event_type.replace("_", " ")
                self.phase = label
                self._write(f"[CTRL] {label} source={source}")
                self._render_status(force=True)


progress_renderer = ProgressRenderer()


def _clear_monitor_state():
    with lock:
        logs.clear()
        summary["total_events"] = 0
        summary["by_source"].clear()
        summary["by_event_type"].clear()
        summary["network"]["bytes_sent"] = 0
        summary["network"]["bytes_recv"] = 0
        summary["network"]["by_label"].clear()
    progress_renderer.reset()


def _read_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_config(config):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    os.replace(tmp, CONFIG_PATH)


def _editable_field_schema():
    return {
        "experiment": {
            "mode": ["centralized", "splitfed"],
            "global_epochs": "int>0",
            "local_epochs": "int>0",
            "target_accuracy": "0~1",
            "optimization": {
                "client_lr": "float>0",
                "server_lr": "float>0",
                "momentum": "0~1",
                "weight_decay": "float>=0",
            },
            "dataset_params": {
                "batch_size": "int>0",
            },
        },
        "network": {
            "server_timeout": "float>0",
            "stragglers": {
                "<client_id>": {
                    "delay": "float>=0",
                    "drop_rate": "0~1",
                }
            },
        },
    }


def _apply_config_patch(config, patch):
    out = copy.deepcopy(config)

    for top_key in patch.keys():
        if top_key not in {"experiment", "network"}:
            raise ValueError(f"Unsupported top-level key: {top_key}")

    if "experiment" in patch:
        exp_patch = patch["experiment"]
        exp = out.setdefault("experiment", {})
        for key, value in exp_patch.items():
            if key == "mode":
                if value not in {"centralized", "splitfed"}:
                    raise ValueError("experiment.mode must be centralized or splitfed")
                exp[key] = value
            elif key in {"global_epochs", "local_epochs"}:
                iv = int(value)
                if iv <= 0:
                    raise ValueError(f"experiment.{key} must be > 0")
                exp[key] = iv
            elif key == "target_accuracy":
                fv = float(value)
                if fv < 0 or fv > 1:
                    raise ValueError("experiment.target_accuracy must be in [0, 1]")
                exp[key] = fv
            elif key == "optimization":
                opt = exp.setdefault("optimization", {})
                for ok, ov in value.items():
                    if ok not in {"client_lr", "server_lr", "momentum", "weight_decay"}:
                        raise ValueError(f"Unsupported optimization key: {ok}")
                    fv = float(ov)
                    if ok in {"client_lr", "server_lr"} and fv <= 0:
                        raise ValueError(f"optimization.{ok} must be > 0")
                    if ok == "momentum" and (fv < 0 or fv > 1):
                        raise ValueError("optimization.momentum must be in [0, 1]")
                    if ok == "weight_decay" and fv < 0:
                        raise ValueError("optimization.weight_decay must be >= 0")
                    opt[ok] = fv
            elif key == "dataset_params":
                ds = exp.setdefault("dataset_params", {})
                for dk, dv in value.items():
                    if dk != "batch_size":
                        raise ValueError(f"Unsupported dataset_params key: {dk}")
                    iv = int(dv)
                    if iv <= 0:
                        raise ValueError("dataset_params.batch_size must be > 0")
                    ds[dk] = iv
            else:
                raise ValueError(f"Unsupported experiment key: {key}")

    if "network" in patch:
        net_patch = patch["network"]
        net = out.setdefault("network", {})
        for key, value in net_patch.items():
            if key == "server_timeout":
                fv = float(value)
                if fv <= 0:
                    raise ValueError("network.server_timeout must be > 0")
                net[key] = fv
            elif key == "stragglers":
                st = net.setdefault("stragglers", {})
                for client_id, setting in value.items():
                    c = st.setdefault(client_id, {})
                    for sk, sv in setting.items():
                        if sk == "delay":
                            fv = float(sv)
                            if fv < 0:
                                raise ValueError("straggler delay must be >= 0")
                            c[sk] = fv
                        elif sk == "drop_rate":
                            fv = float(sv)
                            if fv < 0 or fv > 1:
                                raise ValueError("straggler drop_rate must be in [0, 1]")
                            c[sk] = fv
                        else:
                            raise ValueError(f"Unsupported straggler key: {sk}")
            else:
                raise ValueError(f"Unsupported network key: {key}")

    return out


def _update_summary(item):
    summary["total_events"] += 1
    summary["by_source"][item.get("source", "unknown")] += 1
    summary["by_event_type"][item.get("event_type", item.get("type", "unknown"))] += 1

    direction = item.get("direction")
    bytes_count = int(item.get("payload_bytes", 0) or 0)
    label = item.get("payload_label", "unknown")
    if direction == "out":
        summary["network"]["bytes_sent"] += bytes_count
    elif direction == "in":
        summary["network"]["bytes_recv"] += bytes_count
    if bytes_count > 0:
        summary["network"]["by_label"][label] += bytes_count


def _summary_snapshot():
    return {
        "total_events": summary["total_events"],
        "by_source": dict(summary["by_source"]),
        "by_event_type": dict(summary["by_event_type"]),
        "network": {
            "bytes_sent": summary["network"]["bytes_sent"],
            "bytes_recv": summary["network"]["bytes_recv"],
            "by_label": dict(summary["network"]["by_label"]),
        },
        "last_event": logs[-1] if logs else None,
    }


def _append_event_sync(item):
    with lock:
        logs.append(item)
        _update_summary(item)
    progress_renderer.handle(item)


def _emit_control_event(item):
    payload = {
        "ts": time.time(),
        "source": "control_api",
        **item,
    }
    _append_event_sync(payload)


controller = TrainingController(PROJECT_ROOT, PYTHON_BIN, event_hook=_emit_control_event)


@app.get("/dashboard")
async def dashboard():
    if not _is_web_mode():
        raise HTTPException(status_code=404, detail="Web mode is not enabled. Set monitoring.render_mode to 'web'.")
    index_path = os.path.join(WEB_STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard not built. Run 'npm run build' in web/ directory.")
    return FileResponse(index_path)


@app.on_event("startup")
async def on_startup():
    if _is_web_mode():
        try:
            config = _read_config()
            host = config.get("monitoring", {}).get("api_host", "127.0.0.1")
            port = config.get("monitoring", {}).get("api_port", 9000)
            print(f"\n  Dashboard: http://{host}:{port}/dashboard\n", flush=True)
        except Exception:
            pass


@app.post("/report")
async def report_metric(request: Request):
    data = await request.json()
    _append_event_sync(data)
    return {"status": "success", "length": len(logs), "total_events": summary["total_events"]}

@app.get("/logs")
def get_logs(limit: int = 500, source: str = "", event_type: str = ""):
    with lock:
        data = logs
        if source:
            data = [x for x in data if x.get("source") == source]
        if event_type:
            data = [x for x in data if x.get("event_type") == event_type]
        if limit > 0:
            data = data[-limit:]
    return {"status": "success", "count": len(data), "logs": data}


@app.get("/summary")
def get_summary():
    with lock:
        out = _summary_snapshot()
    return {"status": "success", "summary": out}


@app.get("/health")
def health():
    return {"status": "ok", "log_count": len(logs)}


@app.post("/clear")
async def clear_logs():
    _clear_monitor_state()
    return {"status": "success", "message": "logs cleared"}


@app.get("/config")
def get_config():
    config = _read_config()
    return {
        "status": "success",
        "config": config,
        "editable_schema": _editable_field_schema(),
    }


@app.put("/config")
async def update_config(request: Request):
    if controller.status()["state"] in {"running", "starting", "stopping"}:
        raise HTTPException(status_code=409, detail="Training is running; stop training before updating config.")
    patch = await request.json()
    config = _read_config()
    try:
        new_cfg = _apply_config_patch(config, patch)
        _write_config(new_cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _append_event_sync(
        {
            "ts": time.time(),
            "source": "control_api",
            "event_type": "config_updated",
            "applied_patch": patch,
        }
    )
    return {"status": "success", "config": new_cfg}


@app.get("/training/status")
def training_status():
    return {"status": "success", "training": controller.status()}


@app.post("/training/start")
async def training_start():
    if controller.status().get("state") in {"running", "starting", "stopping"}:
        raise HTTPException(status_code=409, detail={"ok": False, "reason": "already_running", "status": controller.status()})

    _clear_monitor_state()

    config = _read_config()
    result = await asyncio.to_thread(controller.start, config)
    if not result.get("ok"):
        code = 409 if result.get("reason") == "already_running" else 500
        raise HTTPException(status_code=code, detail=result)

    _append_event_sync(
        {
            "ts": time.time(),
            "source": "control_api",
            "event_type": "training_start_requested",
            "training": result.get("status", {}),
        }
    )
    return {"status": "success", "training": result.get("status", {})}


@app.post("/training/stop")
async def training_stop():
    result = await asyncio.to_thread(controller.stop)
    _append_event_sync(
        {
            "ts": time.time(),
            "source": "control_api",
            "event_type": "training_stop_requested",
            "training": result.get("status", {}),
        }
    )
    return {"status": "success", "training": result.get("status", {})}

if os.path.isdir(WEB_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=WEB_STATIC_DIR), name="static")

if __name__ == "__main__":
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    host = config['monitoring']['api_host']
    port = config['monitoring']['api_port']
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="warning")
