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
    level=getattr(logging, os.environ.get("FED_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - CLIENT %(name)s - %(levelname)s - %(message)s",
)


class Client:
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

    def __init__(self, config_path, client_id, project_root=None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Default data lookup to the directory containing the resolved config.
        # manager.py and TrainingController pass the repository root explicitly;
        # this fallback keeps direct module use and tests with temporary configs working.
        self.project_root = os.path.abspath(project_root) if project_root is not None else os.path.dirname(os.path.abspath(config_path))
        self.client_id = client_id
        preferred_device = self.config.get("experiment", {}).get("device", "auto")
        self.device = self._select_device(preferred_device)
        self.mode = self.config.get("experiment", {}).get("mode", "centralized")
        self.server_host = self.config["topology"]["server"]["host"]
        self.server_port = int(self.config["topology"]["server"]["port"])
        self.monitor = MonitorReporter(
            self.config["monitoring"]["api_host"],
            int(self.config["monitoring"]["api_port"]),
            client_id,
        )

        exp = self.config.get("experiment", {})
        ds = exp.get("dataset_params", self.config.get("dataset_params", {}))
        opt = exp.get("optimization", {})

        self.batch_size = int(ds.get("batch_size", 64))
        self.num_workers = int(ds.get("num_workers", 0))
        self.global_epochs = int(exp.get("global_epochs", 1))
        self.default_local_epochs = int(exp.get("local_epochs", 1))
        self.default_lr = float(opt.get("client_lr", 0.01))
        self.default_momentum = float(opt.get("momentum", 0.9))
        self.default_weight_decay = float(opt.get("weight_decay", 5e-4))

        self.straggler_config = self.config["network"].get("stragglers", {}).get(client_id, {"delay": 0.0, "drop_rate": 0.0})
        self.communicator = TCPCommunicator(self.config["network"].get("compression", False))
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lock = threading.RLock()

        self.criterion = nn.CrossEntropyLoss()
        if self.mode == "splitfed":
            self.client_model, self.shadow_server_model = get_model("splitfed")
            self.client_model.to(self.device)
            self.shadow_server_model.to(self.device)
            self.optimizer = torch.optim.SGD(
                self.client_model.parameters(),
                lr=self.default_lr,
                momentum=self.default_momentum,
                weight_decay=self.default_weight_decay,
            )
        else:
            self.model = get_model("centralized")
            self.model.to(self.device)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.default_lr,
                momentum=self.default_momentum,
                weight_decay=self.default_weight_decay,
            )

        self.logger = logging.getLogger(client_id)
        self.train_loader, self.val_loader, self.test_loader = self._load_local_data()
        self.net_stats = {
            "bytes_sent": 0,
            "bytes_recv": 0,
            "messages_sent": 0,
            "messages_recv": 0,
            "sent_by_type": {},
            "recv_by_type": {},
        }
        self.monitor.post(
            "startup",
            mode=self.mode,
            client_id=self.client_id,
            device=str(self.device),
            preferred_device=preferred_device,
            server_host=self.server_host,
            server_port=self.server_port,
            straggler=self.straggler_config,
            dataset={
                "train_samples": len(self.train_loader.dataset),
                "val_samples": len(self.val_loader.dataset),
                "test_samples": len(self.test_loader.dataset),
            },
            batch_size=self.batch_size,
        )

    def _normalize(self, images):
        """Normalize images based on their channel count (MNIST: 1ch, CIFAR-10: 3ch)."""
        x = images.float()
        # Check number of channels from image shape
        if x.shape[1] == 1:  # MNIST: [N, 1, 28, 28]
            mean = torch.tensor([0.1307], dtype=x.dtype).view(1, 1, 1, 1)
            std = torch.tensor([0.3081], dtype=x.dtype).view(1, 1, 1, 1)
        elif x.shape[1] == 3:  # CIFAR-10: [N, 3, 32, 32]
            x = x.div(255.0)  # CIFAR loaded as uint8
            mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=x.dtype).view(1, 3, 1, 1)
        else:
            raise ValueError(f"Unsupported image channels: {x.shape[1]}")
        return (x - mean) / std

    def _build_loader(self, images, labels, shuffle):
        ds = TensorDataset(self._normalize(images), labels.long())
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def _load_local_data(self):
        idx = self.client_id.split("_")[-1]
        path = os.path.join(self.project_root, "data", "splits", f"client_{idx}_data.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Please run scripts/prepare_mnist.py first.")
        payload = torch.load(path, map_location="cpu")
        self.logger.info(
            "Loaded data: train=%d val=%d test=%d",
            payload["train_images"].shape[0],
            payload["val_images"].shape[0],
            payload["test_images"].shape[0],
        )
        train_loader = self._build_loader(payload["train_images"], payload["train_labels"], shuffle=True)
        val_loader = self._build_loader(payload["val_images"], payload["val_labels"], shuffle=False)
        test_loader = self._build_loader(payload["test_images"], payload["test_labels"], shuffle=False)
        return train_loader, val_loader, test_loader

    def _should_drop_round(self):
        drop_rate = float(self.straggler_config.get("drop_rate", 0.0))
        return bool(torch.rand(1).item() < drop_rate)

    def _maybe_delay(self):
        delay = float(self.straggler_config.get("delay", 0.0))
        if delay > 0:
            self.logger.info("Simulating delay %.2fs", delay)
            self.monitor.post("straggler_delay", delay_seconds=delay, mode=self.mode)
            time.sleep(delay)

    def report_metric(self, metric_data):
        payload = dict(metric_data)
        payload.setdefault("mode", self.mode)
        self.monitor.post("metric", **payload)

    def _count_type(self, bucket, key, inc):
        bucket[key] = int(bucket.get(key, 0)) + int(inc)

    def _record_network(self, direction, message_type, payload_bytes, round_id=None, extra=None):
        with self.lock:
            if direction == "out":
                self.net_stats["bytes_sent"] += int(payload_bytes)
                self.net_stats["messages_sent"] += 1
                self._count_type(self.net_stats["sent_by_type"], message_type, 1)
            else:
                self.net_stats["bytes_recv"] += int(payload_bytes)
                self.net_stats["messages_recv"] += 1
                self._count_type(self.net_stats["recv_by_type"], message_type, 1)

            payload = {
                "mode": self.mode,
                "direction": direction,
                "peer": "server",
                "message_type": message_type,
                "payload_label": payload_label(message_type),
                "payload_bytes": int(payload_bytes),
                "round": round_id,
                "bytes_sent_total": self.net_stats["bytes_sent"],
                "bytes_recv_total": self.net_stats["bytes_recv"],
                "messages_sent_total": self.net_stats["messages_sent"],
                "messages_recv_total": self.net_stats["messages_recv"],
            }
        if extra:
            payload.update(extra)
        self.monitor.post("network_io", **payload)
        self.logger.debug(
            "net direction=%s peer=server msg=%s label=%s payload=%dB tx=%dB rx=%dB tx_msg=%d rx_msg=%d round=%s",
            direction,
            message_type,
            payload["payload_label"],
            int(payload_bytes),
            int(payload["bytes_sent_total"]),
            int(payload["bytes_recv_total"]),
            int(payload["messages_sent_total"]),
            int(payload["messages_recv_total"]),
            round_id if round_id is not None else "-",
        )

    def _send(self, payload, round_id=None, extra=None):
        ok, size = self.communicator.send_data(self.conn, payload)
        msg_type = payload.get("type", "unknown")
        self._record_network("out", msg_type, size, round_id=round_id, extra=extra)
        self.monitor.post(
            "send_ack",
            mode=self.mode,
            peer="server",
            message_type=msg_type,
            payload_label=payload_label(msg_type),
            payload_bytes=int(size),
            send_success=bool(ok),
            round=round_id,
        )
        return ok, size

    def _recv(self):
        msg, meta = self.communicator.recv_data_with_meta(self.conn)
        if msg is None:
            return None, None
        msg_type = msg.get("type", "unknown")
        payload_bytes = int((meta or {}).get("payload_bytes", 0))
        round_id = msg.get("round")
        self._record_network("in", msg_type, payload_bytes, round_id=round_id)
        self.monitor.post(
            "recv_ack",
            mode=self.mode,
            peer="server",
            message_type=msg_type,
            payload_label=payload_label(msg_type),
            payload_bytes=payload_bytes,
            recv_success=True,
            round=round_id,
            magic_ok=bool((meta or {}).get("magic_ok", False)),
            compression=bool((meta or {}).get("compression", False)),
        )
        return msg, meta

    def _eval_centralized(self, loader):
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
                correct += int((logits.argmax(dim=1) == y).sum().item())
                total += y.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def _eval_split(self, loader):
        self.client_model.eval()
        self.shadow_server_model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.shadow_server_model(self.client_model(x))
                loss = self.criterion(logits, y)
                loss_sum += float(loss.item()) * y.size(0)
                correct += int((logits.argmax(dim=1) == y).sum().item())
                total += y.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def _train_centralized_round(self, msg):
        round_id = int(msg["round"])
        self.model.load_state_dict(msg["weights"])
        lr = float(msg.get("lr", self.default_lr))
        local_epochs = int(msg.get("local_epochs", self.default_local_epochs))
        momentum = float(msg.get("momentum", self.default_momentum))
        weight_decay = float(msg.get("weight_decay", self.default_weight_decay))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.model.train()
        batches_per_local_epoch = max(len(self.train_loader), 1)
        total_batches = max(len(self.train_loader) * max(local_epochs, 1), 1)
        batch_idx = 0

        total = 0
        correct = 0
        loss_sum = 0.0
        for local_epoch_idx in range(1, local_epochs + 1):
            for batch_in_local_epoch, (x, y) in enumerate(self.train_loader, start=1):
                batch_idx += 1
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()

                batch_correct = int((logits.argmax(dim=1) == y).sum().item())
                batch_acc = float(batch_correct) / max(int(y.size(0)), 1)
                self.monitor.post(
                    "batch_progress",
                    mode="centralized",
                    round=round_id,
                    client_id=self.client_id,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    batch_loss=float(loss.item()),
                    batch_acc=batch_acc,
                    total_epochs=self.global_epochs,
                    local_epoch_idx=local_epoch_idx,
                    local_epochs=local_epochs,
                    batch_in_local_epoch=batch_in_local_epoch,
                    batches_per_local_epoch=batches_per_local_epoch,
                )

                loss_sum += float(loss.item()) * y.size(0)
                correct += batch_correct
                total += y.size(0)

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = self._eval_centralized(self.val_loader)
        test_loss, test_acc = self._eval_centralized(self.test_loader)

        self.report_metric(
            {
                "source": self.client_id,
                "mode": "centralized",
                "round": round_id,
                "type": "local_eval",
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        payload = {
            "type": "model_update",
            "round": round_id,
            "client_id": self.client_id,
            "weights": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "num_samples": len(self.train_loader.dataset),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        self.monitor.post(
            "local_round_done",
            mode="centralized",
            round=round_id,
            client_id=self.client_id,
            train_samples=len(self.train_loader.dataset),
            val_samples=len(self.val_loader.dataset),
            test_samples=len(self.test_loader.dataset),
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )
        return payload

    def _train_splitfed_round(self, msg):
        round_id = int(msg["round"])
        self.client_model.load_state_dict(msg["client_weights"])
        self.shadow_server_model.load_state_dict(msg["server_weights"])

        lr = float(msg.get("lr", self.default_lr))
        local_epochs = int(msg.get("local_epochs", self.default_local_epochs))
        momentum = float(msg.get("momentum", self.default_momentum))
        weight_decay = float(msg.get("weight_decay", self.default_weight_decay))

        optimizer = torch.optim.SGD(self.client_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.client_model.train()
        batches_per_local_epoch = max(len(self.train_loader), 1)
        total_batches = max(len(self.train_loader) * max(local_epochs, 1), 1)
        batch_idx = 0

        total = 0
        correct = 0
        loss_sum = 0.0
        split_batches = 0
        for local_epoch_idx in range(1, local_epochs + 1):
            for batch_in_local_epoch, (x, y) in enumerate(self.train_loader, start=1):
                batch_idx += 1
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                smashed = self.client_model(x)
                send_payload = {
                    "type": "split_batch",
                    "round": round_id,
                    "client_id": self.client_id,
                    "activations": smashed.detach().cpu(),
                    "labels": y.detach().cpu(),
                }
                self._send(send_payload, round_id=round_id, extra={"batch_size": int(y.size(0))})
                resp, _ = self._recv()
                if resp is None or resp.get("type") != "split_grad":
                    raise RuntimeError(f"Invalid split response in round {round_id}: {resp}")
                if "error" in resp:
                    raise RuntimeError(resp["error"])

                grad = resp["grad"].to(self.device)
                smashed.backward(grad)
                optimizer.step()

                batch_size = int(resp.get("batch_size", y.size(0)))
                batch_correct = int(resp.get("correct", 0))
                batch_loss = float(resp.get("loss", 0.0))
                batch_acc = float(batch_correct) / max(batch_size, 1)

                self.monitor.post(
                    "batch_progress",
                    mode="splitfed",
                    round=round_id,
                    client_id=self.client_id,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    batch_loss=batch_loss,
                    batch_acc=batch_acc,
                    total_epochs=self.global_epochs,
                    local_epoch_idx=local_epoch_idx,
                    local_epochs=local_epochs,
                    batch_in_local_epoch=batch_in_local_epoch,
                    batches_per_local_epoch=batches_per_local_epoch,
                )

                total += batch_size
                correct += batch_correct
                loss_sum += batch_loss * y.size(0)
                split_batches += 1

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = self._eval_split(self.val_loader)
        test_loss, test_acc = self._eval_split(self.test_loader)

        self.report_metric(
            {
                "source": self.client_id,
                "mode": "splitfed",
                "round": round_id,
                "type": "local_eval",
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        payload = {
            "type": "split_update",
            "round": round_id,
            "client_id": self.client_id,
            "weights": {k: v.detach().cpu() for k, v in self.client_model.state_dict().items()},
            "num_samples": len(self.train_loader.dataset),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        self.monitor.post(
            "local_round_done",
            mode="splitfed",
            round=round_id,
            client_id=self.client_id,
            split_batches=split_batches,
            train_samples=len(self.train_loader.dataset),
            val_samples=len(self.val_loader.dataset),
            test_samples=len(self.test_loader.dataset),
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )
        return payload

    def run(self):
        self.logger.info("Connecting to %s:%d", self.server_host, self.server_port)
        self.conn.connect((self.server_host, self.server_port))
        self._send({"type": "register", "client_id": self.client_id})

        while True:
            msg, _ = self._recv()
            if msg is None:
                self.logger.warning("Server disconnected")
                self.monitor.post("disconnect", mode=self.mode, reason="server_disconnected")
                break

            msg_type = msg.get("type")
            if msg_type == "register_ack":
                self.logger.info("Registered. Mode=%s", msg.get("mode"))
                self.monitor.post("registered", mode=msg.get("mode"), client_id=self.client_id)
                continue

            if msg_type == "shutdown":
                self.logger.info("Shutdown received: %s", msg.get("reason", ""))
                self.monitor.post(
                    "shutdown",
                    mode=self.mode,
                    reason=msg.get("reason", ""),
                    net_totals={
                        "bytes_sent": self.net_stats["bytes_sent"],
                        "bytes_recv": self.net_stats["bytes_recv"],
                        "messages_sent": self.net_stats["messages_sent"],
                        "messages_recv": self.net_stats["messages_recv"],
                    },
                )
                break

            if msg_type == "round_start_centralized":
                if self._should_drop_round():
                    self.logger.warning("Round %s dropped by simulation.", msg.get("round"))
                    self.monitor.post(
                        "round_dropped",
                        mode="centralized",
                        round=msg.get("round"),
                        reason="drop_rate_triggered",
                    )
                    continue
                payload = self._train_centralized_round(msg)
                self._maybe_delay()
                self._send(payload, round_id=payload.get("round"))
                continue

            if msg_type == "round_start_splitfed":
                if self._should_drop_round():
                    self.logger.warning("Round %s dropped by simulation.", msg.get("round"))
                    self.monitor.post(
                        "round_dropped",
                        mode="splitfed",
                        round=msg.get("round"),
                        reason="drop_rate_triggered",
                    )
                    continue
                payload = self._train_splitfed_round(msg)
                self._maybe_delay()
                self._send(payload, round_id=payload.get("round"))
                continue

            self.logger.warning("Unknown message type: %s", msg_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client for federated learning")
    parser.add_argument("client_id", type=str, help="Client identifier (e.g. client_1)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--data-path", type=str, default=None, help="Project root path containing data (overrides default)")
    args = parser.parse_args()

    client = Client(args.config, args.client_id, project_root=args.data_path)
    client.run()
