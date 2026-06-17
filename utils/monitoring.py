import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def payload_label(message_type: str) -> str:
    labels = {
        "register": "control_register",
        "register_ack": "control_ack",
        "shutdown": "control_shutdown",
        "round_start_centralized": "model_weights",
        "round_start_splitfed": "model_weights_split",
        "model_update": "model_weights_update",
        "split_update": "split_client_weights_update",
        "split_batch": "activations_and_labels",
        "split_grad": "activation_gradients",
        "ring_pass": "ring_model_weights",
        "ring_join": "ring_control_join",
        "ring_shutdown": "ring_control_shutdown",
    }
    return labels.get(message_type, "unknown")


class MonitorReporter:
    def __init__(self, api_host: str, api_port: int, source: str, timeout: float = 1.0):
        self.url = f"http://{api_host}:{api_port}/report"
        self.source = source
        self.session_seq = 0
        self.timeout = float(timeout)

    def _build_payload(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        self.session_seq += 1
        return {
            "ts": time.time(),
            "source": self.source,
            "event_type": event_type,
            "seq": self.session_seq,
            **fields,
        }

    def post(self, event_type: str, **fields: Any) -> None:
        payload = self._build_payload(event_type, **fields)
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                return
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            logging.warning("Monitor event dropped: source=%s event=%s error=%s", self.source, event_type, exc)
            return

    def post_best_effort(self, event_type: str, timeout: Optional[float] = None, **fields: Any) -> bool:
        payload = self._build_payload(event_type, **fields)
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout if timeout is None else float(timeout)):
                return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False


def wait_for_monitor_ready(api_host: str, api_port: int, timeout: float = 10.0, poll_interval: float = 0.2) -> bool:
    deadline = time.time() + max(float(timeout), 0.0)
    health_url = f"http://{api_host}:{api_port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=min(max(float(poll_interval), 0.1), 2.0)) as resp:
                if getattr(resp, "status", 200) == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            time.sleep(max(float(poll_interval), 0.05))
    return False


def compact_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    topo = config.get("topology", {})
    mode = config.get("experiment", {}).get("mode", "centralized")
    result: Dict[str, Any] = {"mode": mode}
    if mode == "ring":
        nodes = topo.get("nodes", [])
        result["topology"] = "ring"
        result["node_count"] = len(nodes)
        result["nodes"] = [
            {"id": n.get("id"), "host": n.get("host"), "port": n.get("port")}
            for n in nodes
        ]
    else:
        clients = topo.get("clients", [])
        result["server"] = topo.get("server", {})
        result["client_count"] = len(clients)
        result["clients"] = [
            {"id": c.get("id"), "host": c.get("host"), "port": c.get("port")}
            for c in clients
        ]
    return result
