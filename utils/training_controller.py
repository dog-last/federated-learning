import json
import os
import signal
import subprocess
import threading
import time
import torch
from typing import Any, Dict, Optional, Union


class TrainingController:
    def __init__(self, project_root: str, python_bin: str, event_hook=None):
        self.project_root = project_root
        self.python_bin = python_bin
        self.event_hook = event_hook

        self._lock = threading.Lock()
        self._state = "stopped"
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._last_error: Optional[str] = None

        self._server_proc = None
        self._client_procs = []
        self._ring_mode = False
        self._proc_log_handles = []
        self._watcher = None
        self._manual_stop_requested = False
        self._run_id: Optional[str] = None
        self._run_log_dir: Optional[str] = None

    def _logs_dir(self) -> str:
        path = os.path.join(self.project_root, "logs")
        os.makedirs(path, exist_ok=True)
        return path

    def _new_run_id(self) -> str:
        return time.strftime("%Y%m%d%H%M%S")

    def _run_logs_dir(self) -> str:
        if not self._run_id:
            self._run_id = self._new_run_id()
        self._run_log_dir = os.path.join(self._logs_dir(), self._run_id)
        os.makedirs(self._run_log_dir, exist_ok=True)
        return self._run_log_dir

    def _open_role_log(self, role: str):
        path = os.path.join(self._run_logs_dir(), f"{role}.log")
        handle = open(path, "a", encoding="utf-8")
        handle.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - LOG - INFO - "
            f"Opened process log for role={role}, run_id={self._run_id}\n"
        )
        handle.flush()
        self._proc_log_handles.append(handle)
        return handle

    def _close_role_logs(self) -> None:
        for handle in self._proc_log_handles:
            try:
                handle.close()
            except OSError:
                pass
        self._proc_log_handles = []

    def _emit(self, event: Dict[str, Any]) -> None:
        if self.event_hook:
            event["ts"] = time.time()
            self.event_hook(event)

    def _proc_info(self, proc, role: str) -> Dict[str, Any]:
        if proc is None:
            return {"role": role, "pid": None, "alive": False, "exit_code": None}
        return {
            "role": role,
            "pid": proc.pid,
            "alive": proc.poll() is None,
            "exit_code": proc.poll(),
        }

    def _is_mnist_split_compatible(self, required_files) -> bool:
        try:
            client_payload = torch.load(required_files[0], map_location="cpu")
            server_payload = torch.load(required_files[-1], map_location="cpu")
            client_images = client_payload["train_images"]
            server_images = server_payload["images"]
            return client_images.ndim == 4 and client_images.shape[1:] == (1, 28, 28) and server_images.shape[1:] == (1, 28, 28)
        except (OSError, RuntimeError, KeyError, IndexError, ValueError, TypeError):
            return False

    def _ensure_data(self, config: Dict[str, Any]) -> None:
        clients = config.get("topology", {}).get("clients", [])
        required = [
            os.path.join(self.project_root, "data", "splits", f"{client['id']}_data.pt")
            for client in clients
            if isinstance(client, dict) and client.get("id")
        ]
        required.append(os.path.join(self.project_root, "data", "splits", "server_test_data.pt"))
        if all(os.path.exists(path) for path in required) and self._is_mnist_split_compatible(required):
            return

        subprocess.run(
            [self.python_bin, "scripts/prepare_mnist.py"],
            cwd=self.project_root,
            check=True,
        )

    def _watch_server(self) -> None:
        proc = self._server_proc
        if proc is None:
            return
        exit_code = proc.wait()

        should_emit_server_exit = True
        with self._lock:
            if self._state == "stopped":
                return
            if self._manual_stop_requested:
                should_emit_server_exit = False
            self._state = "stopped"
            self._stopped_at = time.time()

            if should_emit_server_exit:
                for p in self._client_procs:
                    if p.poll() is None:
                        p.send_signal(signal.SIGTERM)
                time.sleep(0.2)
                for p in self._client_procs:
                    if p.poll() is None:
                        p.kill()

            self._server_proc = None
            self._client_procs = []
            self._ring_mode = False

        self._close_role_logs()

        if should_emit_server_exit:
            self._emit(
                {
                    "event_type": "training_stopped",
                    "reason": "server_exit",
                    "server_exit_code": exit_code,
                }
            )

    def _resolve_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config, str):
            if not os.path.isfile(config):
                raise FileNotFoundError(f"Config file not found: {config}")
            with open(config, "r", encoding="utf-8") as f:
                return json.load(f)
        return config

    def start(self, config: Union[str, Dict[str, Any]], run_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self._state in {"running", "starting"}:
                state = self._state
                started_at = self._started_at
                stopped_at = self._stopped_at
                last_error = self._last_error
                return {
                    "ok": False,
                    "reason": "already_running",
                    "error": "already_running",
                    "status": {
                        "state": state,
                        "started_at": started_at,
                        "stopped_at": stopped_at,
                        "uptime_seconds": 0,
                        "server": self._proc_info(None, "server"),
                        "clients": [],
                        "last_error": last_error,
                        "run_id": self._run_id,
                        "log_dir": self._run_log_dir,
                    },
                }
            self._state = "starting"
            self._last_error = None
            self._manual_stop_requested = False
            self._run_id = run_id or self._new_run_id()
            self._run_log_dir = os.path.join(self._logs_dir(), self._run_id)
            os.makedirs(self._run_log_dir, exist_ok=True)

        try:
            resolved = self._resolve_config(config)
            # Determine a config path to pass to subprocesses. If the caller
            # provided a path string, reuse it; if they provided a dict, write
            # it to a temp config file under the project root so child processes
            # can receive a path via --config.
            if isinstance(config, str):
                config_path = config
            else:
                config_path = os.path.join(self.project_root, ".generated_config.json")
                with open(config_path, "w", encoding="utf-8") as cf:
                    json.dump(resolved, cf)

            self._ensure_data(resolved)
            env = os.environ.copy()
            env["PYTHONPATH"] = self.project_root
            env["FED_RUN_ID"] = self._run_id or ""
            env["FED_LOG_DIR"] = self._run_log_dir or self._run_logs_dir()
            env.setdefault("FED_LOG_LEVEL", "INFO")
            mode = str(resolved.get("experiment", {}).get("mode", "centralized"))
            self._ring_mode = mode == "ring"

            if self._ring_mode:
                nodes = resolved.get("topology", {}).get("nodes", [])
                if not nodes:
                    raise ValueError("Ring mode requires topology.nodes")
                client_procs = []
                for node in nodes:
                    node_id = str(node["id"])
                    p = subprocess.Popen(
                        [self.python_bin, "-m", "core.ring_node", node_id, "--config", config_path, "--data-path", self.project_root],
                        cwd=self.project_root,
                        env=env,
                        stdout=self._open_role_log(f"ring-node-{node_id}"),
                        stderr=subprocess.STDOUT,
                    )
                    client_procs.append(p)
                server_proc = client_procs[0]
            else:
                server_proc = subprocess.Popen(
                    [self.python_bin, "-m", "core.server", "--config", config_path, "--data-path", self.project_root],
                    cwd=self.project_root,
                    env=env,
                    stdout=self._open_role_log("server"),
                    stderr=subprocess.STDOUT,
                )
                time.sleep(1)

                client_procs = []
                for client in resolved["topology"]["clients"]:
                    cid = client["id"]
                    p = subprocess.Popen(
                        [self.python_bin, "-m", "core.client", cid, "--config", config_path, "--data-path", self.project_root],
                        cwd=self.project_root,
                        env=env,
                        stdout=self._open_role_log(f"client-{cid}"),
                        stderr=subprocess.STDOUT,
                    )
                    client_procs.append(p)

            with self._lock:
                self._server_proc = server_proc
                self._client_procs = client_procs
                self._state = "running"
                self._started_at = time.time()
                self._stopped_at = None

            self._watcher = threading.Thread(target=self._watch_server, daemon=True)
            self._watcher.start()

            self._emit(
                {
                    "event_type": "training_started",
                    "mode": mode,
                    "server": self._proc_info(server_proc, "server"),
                    "clients": [self._proc_info(p, "client") for p in client_procs],
                    "run_id": self._run_id,
                    "log_dir": self._run_log_dir,
                }
            )
            return {"ok": True, "status": self.status()}
        except (FileNotFoundError, OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            self._close_role_logs()
            with self._lock:
                self._state = "stopped"
                self._last_error = str(exc)
            return {"ok": False, "reason": "start_failed", "error": str(exc), "status": self.status()}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if self._state == "stopped":
                # Build status dict directly to avoid re-acquiring lock
                status = {
                    "state": self._state,
                    "started_at": self._started_at,
                    "stopped_at": self._stopped_at,
                    "uptime_seconds": 0,
                    "server": self._proc_info(self._server_proc, "server"),
                    "clients": [self._proc_info(p, "client") for p in self._client_procs],
                    "last_error": self._last_error,
                    "run_id": self._run_id,
                    "log_dir": self._run_log_dir,
                }
                return {"ok": True, "status": status}
            self._state = "stopping"
            self._manual_stop_requested = True
            server_proc = self._server_proc
            client_procs = list(self._client_procs)

        for p in client_procs:
            if p is not None and p.poll() is None:
                p.send_signal(signal.SIGTERM)
        if server_proc is not None and server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)

        time.sleep(0.2)
        for p in client_procs:
            if p is not None and p.poll() is None:
                p.kill()
        if server_proc is not None and server_proc.poll() is None:
            server_proc.kill()

        with self._lock:
            self._state = "stopped"
            self._stopped_at = time.time()
            self._server_proc = None
            self._client_procs = []
            self._ring_mode = False
            self._manual_stop_requested = False

        self._close_role_logs()

        self._emit({"event_type": "training_stopped", "reason": "manual_stop"})
        return {"ok": True, "status": self.status()}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            state = self._state
            started_at = self._started_at
            stopped_at = self._stopped_at
            server_proc = self._server_proc
            client_procs = list(self._client_procs)
            last_error = self._last_error

        return {
            "state": state,
            "started_at": started_at,
            "stopped_at": stopped_at,
            "uptime_seconds": (time.time() - started_at) if started_at and state in {"running", "starting", "stopping"} else 0,
            "server": self._proc_info(server_proc, "server"),
            "clients": [self._proc_info(p, "client") for p in client_procs],
            "last_error": last_error,
            "run_id": self._run_id,
            "log_dir": self._run_log_dir,
        }
