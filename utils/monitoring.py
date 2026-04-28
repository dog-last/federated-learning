import json
import time
import urllib.request
from typing import Any, Dict


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
    }
    return labels.get(message_type, "unknown")


class MonitorReporter:
    def __init__(self, api_host: str, api_port: int, source: str):
        self.url = f"http://{api_host}:{api_port}/report"
        self.source = source
        self.session_seq = 0

    def post(self, event_type: str, **fields: Any) -> None:
        self.session_seq += 1
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "source": self.source,
            "event_type": event_type,
            "seq": self.session_seq,
            **fields,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1.0):
            # Fail fast: monitoring channel errors should surface immediately.
            return


def compact_topology(config: Dict[str, Any]) -> Dict[str, Any]:
    topo = config.get("topology", {})
    clients = topo.get("clients", [])
    return {
        "server": topo.get("server", {}),
        "client_count": len(clients),
        "clients": [{"id": c.get("id"), "host": c.get("host"), "port": c.get("port")} for c in clients],
    }
