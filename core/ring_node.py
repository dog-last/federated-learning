"""Decentralized ring node for federated learning without a central server.

Nodes form a ring topology: Client1 → Client2 → Client3 → Client1.
Each node receives model weights from its predecessor, trains locally,
then passes the updated model to its successor via TCP.
"""

import json
import logging
import os
import socket
import threading
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.communicator import TCPCommunicator
from model import get_model
from utils.monitoring import MonitorReporter, payload_label


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - RING-%(name)s - %(levelname)s - %(message)s",
)


class RingNode:
    """P2P node in a ring topology for decentralized federated learning.

    Each node:
    1. Listens on its own port for incoming model weights from the predecessor.
    2. Connects to the successor node to pass trained model weights.
    3. Trains locally on its partition of the data before passing.

    Attributes:
        node_id: Integer node identifier (1-based).
        config: Parsed config.json dict.
        model: Local CNN model.
        communicator: TCP communicator for sending/receiving messages.
        listener: Server socket for accepting connections from predecessor.
    """

    @staticmethod
    def _select_device(preferred):
        pref = str(preferred or "auto").lower()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if pref in {"", "auto"}:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if has_mps:
                return torch.device("mps")
            return torch.device("cpu")
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and has_mps:
            return torch.device("mps")
        return torch.device("cpu")

    def __init__(self, config_path, node_id, project_root=None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.node_id = node_id
        # Rule: data directory must reside under the project root containing `manager.py`.
        # Allow tests or callers to override by passing `project_root` explicitly.
        self.project_root = project_root
        preferred_device = self.config.get("experiment", {}).get("device", "auto")
        self.device = self._select_device(preferred_device)

        nodes = self.config["topology"]["nodes"]
        self.my_host = nodes[node_id - 1]["host"]
        self.my_port = int(nodes[node_id - 1]["port"])
        next_idx = node_id % len(nodes)
        self.next_host = nodes[next_idx]["host"]
        self.next_port = int(nodes[next_idx]["port"])

        exp = self.config.get("experiment", {})
        opt = exp.get("optimization", {})
        ds = exp.get("dataset_params", self.config.get("dataset_params", {}))

        self.max_epochs = int(exp.get("global_epochs", 10))
        self.local_epochs = int(exp.get("local_epochs", 1))
        self.client_lr = float(opt.get("client_lr", 0.01))
        self.momentum = float(opt.get("momentum", 0.9))
        self.weight_decay = float(opt.get("weight_decay", 5e-4))
        self.batch_size = int(ds.get("batch_size", 64))
        self.num_workers = int(ds.get("num_workers", 0))
        self.target_accuracy = float(exp.get("target_accuracy", 0.99))

        stragglers = self.config.get("network", {}).get("stragglers", {})
        self.straggler_cfg = stragglers.get(
            f"client_{node_id}", {"delay": 0.0, "drop_rate": 0.0}
        )

        self.communicator = TCPCommunicator(
            self.config.get("network", {}).get("compression", False)
        )
        self.model = get_model("ring")
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.client_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        monitor_cfg = self.config["monitoring"]
        self.monitor = MonitorReporter(
            monitor_cfg["api_host"], int(monitor_cfg["api_port"]), f"ring_node_{node_id}"
        )

        self.logger = logging.getLogger(str(node_id))

        self.train_loader, self.val_loader, self.test_loader = self._load_local_data()

        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.my_host, self.my_port))
        self.listener.listen(8)

        self._running = False
        self._stop_event = threading.Event()
        self.net_stats = {
            "bytes_sent": 0,
            "bytes_recv": 0,
            "messages_sent": 0,
            "messages_recv": 0,
        }

        self._load_server_test_data()

        self.monitor.post(
            "ring_node_startup",
            node_id=self.node_id,
            device=str(self.device),
            my_addr=f"{self.my_host}:{self.my_port}",
            next_addr=f"{self.next_host}:{self.next_port}",
            straggler=self.straggler_cfg,
            train_samples=len(self.train_loader.dataset),
        )

    def _normalize(self, images):
        x = images.float()
        if x.shape[1] == 1:
            mean = torch.tensor([0.1307], dtype=x.dtype).view(1, 1, 1, 1)
            std = torch.tensor([0.3081], dtype=x.dtype).view(1, 1, 1, 1)
        elif x.shape[1] == 3:
            x = x.div(255.0)
            mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=x.dtype).view(1, 3, 1, 1)
        else:
            raise ValueError(f"Unsupported image channels: {x.shape[1]}")
        return (x - mean) / std

    def _build_loader(self, images, labels, shuffle):
        ds = TensorDataset(self._normalize(images), labels.long())
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers
        )

    def _load_local_data(self):
        path = os.path.join(
            self.project_root, "data", "splits", f"client_{self.node_id}_data.pt"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Please run scripts/prepare_mnist.py first."
            )
        payload = torch.load(path, map_location="cpu")
        self.logger.info(
            "Loaded data: train=%d val=%d test=%d",
            payload["train_images"].shape[0],
            payload["val_images"].shape[0],
            payload["test_images"].shape[0],
        )
        train_loader = self._build_loader(
            payload["train_images"], payload["train_labels"], shuffle=True
        )
        val_loader = self._build_loader(
            payload["val_images"], payload["val_labels"], shuffle=False
        )
        test_loader = self._build_loader(
            payload["test_images"], payload["test_labels"], shuffle=False
        )
        return train_loader, val_loader, test_loader

    def _load_server_test_data(self):
        path = os.path.join(
            self.project_root, "data", "splits", "server_test_data.pt"
        )
        if not os.path.exists(path):
            self.server_test_loader = None
            return
        payload = torch.load(path, map_location="cpu")
        x = self._normalize(payload["images"])
        y = payload["labels"].long()
        ds = TensorDataset(x, y)
        self.server_test_loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def _maybe_delay(self):
        delay = float(self.straggler_cfg.get("delay", 0.0))
        if delay > 0:
            self.logger.info("Simulating straggler delay %.1fs", delay)
            self.monitor.post(
                "straggler_delay",
                node_id=self.node_id,
                delay_seconds=delay,
            )
            time.sleep(delay)

    def _should_drop_round(self):
        drop_rate = float(self.straggler_cfg.get("drop_rate", 0.0))
        return bool(torch.rand(1).item() < drop_rate)

    def _record_network(self, direction, message_type, payload_bytes, extra=None):
        if direction == "out":
            self.net_stats["bytes_sent"] += int(payload_bytes)
            self.net_stats["messages_sent"] += 1
        else:
            self.net_stats["bytes_recv"] += int(payload_bytes)
            self.net_stats["messages_recv"] += 1
        payload = {
            "mode": "ring",
            "direction": direction,
            "node_id": self.node_id,
            "message_type": message_type,
            "payload_bytes": int(payload_bytes),
            **self.net_stats,
        }
        if extra:
            payload.update(extra)
        self.monitor.post("network_io", **payload)

    def _send_to(self, sock, message_dict, peer_label=""):
        ok, size = self.communicator.send_data(sock, message_dict)
        msg_type = message_dict.get("type", "unknown")
        self._record_network("out", msg_type, size)
        self.monitor.post(
            "ring_send",
            node_id=self.node_id,
            peer=peer_label,
            message_type=msg_type,
            payload_bytes=int(size),
            success=bool(ok),
        )
        return ok, size

    def _recv_from(self, sock, peer_label=""):
        msg, meta = self.communicator.recv_data_with_meta(sock)
        if msg is None:
            return None, None
        msg_type = msg.get("type", "unknown")
        payload_bytes = int((meta or {}).get("payload_bytes", 0))
        self._record_network("in", msg_type, payload_bytes)
        self.monitor.post(
            "ring_recv",
            node_id=self.node_id,
            peer=peer_label,
            message_type=msg_type,
            payload_bytes=payload_bytes,
        )
        return msg, meta

    def _evaluate(self, loader):
        self.model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss_sum += float(loss.item()) * y.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += y.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def _train_local(self, round_id):
        """Train model locally for configured local epochs."""
        self.model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                loss_sum += float(loss.item()) * y.size(0)
                correct += int((logits.argmax(dim=1) == y).sum().item())
                total += y.size(0)

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = self._evaluate(self.val_loader)
        test_loss, test_acc = self._evaluate(self.test_loader)

        self.monitor.post(
            "ring_local_train_done",
            node_id=self.node_id,
            round=round_id,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )
        self.logger.info(
            "Round %d local train: loss=%.4f acc=%.4f",
            round_id,
            train_loss,
            train_acc,
        )
        return train_loss, train_acc, test_loss, test_acc

    def _pass_model_to_next(self, round_id):
        """Send current model weights to the successor node in the ring."""
        payload = {
            "type": "ring_pass",
            "round": round_id,
            "origin_id": self.node_id,
            "weights": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "num_samples": len(self.train_loader.dataset),
        }
        peer_label = f"node_{self.node_id % 3 + 1}"

        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect((self.next_host, self.next_port))
                ok, size = self._send_to(sock, payload, peer_label=peer_label)
                sock.close()
                if ok:
                    self.logger.info(
                        "Passed model to %s:%d (%d bytes)",
                        self.next_host,
                        self.next_port,
                        size,
                    )
                    return True
            except (OSError, ConnectionError) as e:
                self.logger.warning(
                    "Pass attempt %d failed: %s", attempt + 1, e
                )
                time.sleep(1.0)
        self.logger.error("Failed to pass model to successor after 3 attempts")
        self.monitor.post(
            "ring_pass_failed",
            node_id=self.node_id,
            round=round_id,
            target=f"{self.next_host}:{self.next_port}",
        )
        return False

    def _wait_for_model_from_prev(self, round_id, timeout=120.0):
        """Wait for the predecessor to send model weights."""
        self.listener.settimeout(timeout)
        self.logger.info("Waiting for model from predecessor (timeout=%.0fs)", timeout)
        try:
            conn, addr = self.listener.accept()
            msg, _ = self._recv_from(conn, peer_label=str(addr))
            conn.close()
            if msg and msg.get("type") == "ring_pass":
                recv_round = int(msg.get("round", -1))
                self.logger.info(
                    "Received model from node %s for round %d",
                    msg.get("origin_id"),
                    recv_round,
                )
                return msg
        except socket.timeout:
            self.logger.warning(
                "Timeout waiting for predecessor in round %d", round_id
            )
            self.monitor.post(
                "ring_recv_timeout",
                node_id=self.node_id,
                round=round_id,
                timeout_seconds=timeout,
            )
        except (OSError, ConnectionError) as e:
            self.logger.error("Error receiving model: %s", e)
        return None

    def run(self):
        """Run the ring node main loop.

        Node 1 initiates the ring by training first and passing its model.
        Nodes 2 and 3 wait for the model from their predecessor before training.
        After each full ring cycle, node 1 evaluates the global model.
        """
        self._running = True
        num_nodes = len(self.config["topology"]["nodes"])
        is_initiator = self.node_id == 1

        self.monitor.post(
            "ring_node_ready",
            node_id=self.node_id,
            is_initiator=is_initiator,
            total_nodes=num_nodes,
        )
        self.logger.info(
            "Ring node %d ready. Initiator=%s. Next=%s:%d",
            self.node_id,
            is_initiator,
            self.next_host,
            self.next_port,
        )

        # Synchronize: wait for all nodes to be ready
        self._sync_start()

        for round_id in range(1, self.max_epochs + 1):
            if self._stop_event.is_set():
                break

            if self._should_drop_round():
                self.logger.warning("Round %d dropped by straggler simulation", round_id)
                self.monitor.post(
                    "ring_round_dropped",
                    node_id=self.node_id,
                    round=round_id,
                )
                # Still need to pass the model along to keep the ring going
                if not is_initiator:
                    prev_msg = self._wait_for_model_from_prev(round_id)
                    if prev_msg:
                        self._pass_model_to_next(round_id)
                else:
                    self._pass_model_to_next(round_id)
                continue

            started = time.time()

            if is_initiator:
                # Node 1 starts the round by training first
                self.monitor.post(
                    "ring_round_start",
                    node_id=self.node_id,
                    round=round_id,
                    role="initiator",
                )
                train_loss, train_acc, test_loss, test_acc = self._train_local(round_id)
                self._maybe_delay()
                self._pass_model_to_next(round_id)

                # Wait for model to come back around the ring (from node 3)
                prev_msg = self._wait_for_model_from_prev(round_id)
                if prev_msg:
                    weights = prev_msg["weights"]
                    self.model.load_state_dict(weights)
                    self.logger.info("Loaded aggregated model from ring pass")
                else:
                    self.logger.warning(
                        "No model received after ring pass, keeping local model"
                    )

                # Evaluate global model on server test set (only node 1 does this)
                if self.server_test_loader is not None:
                    global_loss, global_acc = self._evaluate(self.server_test_loader)
                    self.monitor.post(
                        "ring_global_eval",
                        node_id=self.node_id,
                        round=round_id,
                        test_loss=global_loss,
                        test_acc=global_acc,
                    )
                    self.logger.info(
                        "Round %d global eval: loss=%.4f acc=%.4f",
                        round_id,
                        global_loss,
                        global_acc,
                    )
                    if global_acc >= self.target_accuracy:
                        self.logger.info("Target accuracy reached: %.4f", global_acc)
                        self._notify_ring_shutdown(round_id)
                        break
            else:
                # Non-initiator nodes wait for model from predecessor
                self.monitor.post(
                    "ring_round_start",
                    node_id=self.node_id,
                    round=round_id,
                    role="receiver",
                )
                prev_msg = self._wait_for_model_from_prev(round_id)
                if prev_msg:
                    weights = prev_msg["weights"]
                    self.model.load_state_dict(weights)
                    self.logger.info("Loaded model from predecessor")

                train_loss, train_acc, test_loss, test_acc = self._train_local(round_id)
                self._maybe_delay()
                self._pass_model_to_next(round_id)

            elapsed = time.time() - started
            self.monitor.post(
                "ring_round_end",
                node_id=self.node_id,
                round=round_id,
                elapsed_seconds=elapsed,
            )
            self.logger.info("Round %d finished in %.2fs", round_id, elapsed)

        self._running = False
        self._notify_ring_shutdown(round_id)
        self.logger.info("Ring node %d finished all rounds", self.node_id)

    def _sync_start(self):
        """Synchronize ring nodes before training begins.

        Node 1 waits briefly for other nodes to start their listeners.
        Other nodes signal readiness by connecting to node 1.
        """
        is_initiator = self.node_id == 1
        if is_initiator:
            self.logger.info("Initiator waiting for other nodes to join...")
            expected = len(self.config["topology"]["nodes"]) - 1
            joined = 0
            self.listener.settimeout(30.0)
            while joined < expected:
                try:
                    conn, addr = self.listener.accept()
                    msg, _ = self._recv_from(conn, peer_label=str(addr))
                    conn.close()
                    if msg and msg.get("type") == "ring_join":
                        joined += 1
                        self.logger.info(
                            "Node %s joined the ring (%d/%d)",
                            msg.get("node_id"),
                            joined,
                            expected,
                        )
                except socket.timeout:
                    self.logger.warning("Timeout waiting for node join")
                    break
            self.monitor.post(
                "ring_all_joined",
                node_id=self.node_id,
                joined_count=joined + 1,
                expected_count=expected + 1,
            )
        else:
            # Non-initiator: signal readiness to node 1
            time.sleep(1.0)
            node1_cfg = self.config["topology"]["nodes"][0]
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect((node1_cfg["host"], int(node1_cfg["port"])))
                self._send_to(
                    sock,
                    {"type": "ring_join", "node_id": self.node_id},
                    peer_label="node_1",
                )
                sock.close()
                self.logger.info("Sent join request to initiator")
            except (OSError, ConnectionError) as e:
                self.logger.error("Failed to join ring: %s", e)

    def _notify_ring_shutdown(self, round_id):
        """Notify successor that training is complete."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.next_host, self.next_port))
            self._send_to(
                sock,
                {
                    "type": "ring_shutdown",
                    "round": round_id,
                    "origin_id": self.node_id,
                    "reason": "training_finished",
                },
                peer_label=f"node_{self.node_id % 3 + 1}",
            )
            sock.close()
        except (OSError, ConnectionError):
            pass
        self.listener.close()

    def stop(self):
        """Stop the ring node."""
        self._stop_event.set()
        try:
            self.listener.close()
        except OSError:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ring node for decentralized federated learning")
    parser.add_argument("node_id", type=int, help="Node ID (1-based)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--data-path", type=str, default=None, help="Project root path containing data (overrides default)")
    args = parser.parse_args()

    node = RingNode(args.config, args.node_id, project_root=args.data_path)
    node.run()
