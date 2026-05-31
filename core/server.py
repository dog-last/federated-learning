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
from utils.monitoring import MonitorReporter, compact_topology, payload_label


logging.basicConfig(
    level=getattr(logging, os.environ.get("FED_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - SERVER - %(levelname)s - %(message)s",
)


class Server:
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

    def __init__(self, config_path, project_root=None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.config_path = os.path.abspath(config_path)
        # Default data lookup to the directory containing the resolved config.
        # manager.py and TrainingController pass the repository root explicitly;
        # this fallback keeps direct module use and tests with temporary configs working.
        self.project_root = os.path.abspath(project_root) if project_root is not None else os.path.dirname(self.config_path)
        preferred_device = self.config.get("experiment", {}).get("device", "auto")
        self.device = self._select_device(preferred_device)

        exp = self.config.get("experiment", {})
        opt = exp.get("optimization", {})
        ds = exp.get("dataset_params", self.config.get("dataset_params", {}))

        self.mode = exp.get("mode", "centralized")
        self.max_epochs = int(exp.get("global_epochs", 5))
        self.local_epochs = int(exp.get("local_epochs", 1))
        self.target_accuracy = float(exp.get("target_accuracy", 0.85))
        self.batch_size = int(ds.get("batch_size", 64))
        self.num_workers = int(ds.get("num_workers", 0))

        self.client_lr = float(opt.get("client_lr", 0.01))
        self.server_lr = float(opt.get("server_lr", 0.01))
        self.momentum = float(opt.get("momentum", 0.9))
        self.weight_decay = float(opt.get("weight_decay", 5e-4))

        self.host = self.config["topology"]["server"]["host"]
        self.port = int(self.config["topology"]["server"]["port"])
        self.client_configs = self.config["topology"]["clients"]
        self.num_clients = len(self.client_configs)
        self.timeout = float(self.config["network"].get("server_timeout", 15.0))
        self.monitor = MonitorReporter(
            self.config["monitoring"]["api_host"],
            int(self.config["monitoring"]["api_port"]),
            "server",
        )

        self.communicator = TCPCommunicator(self.config["network"].get("compression", False))
        self.server_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_obj.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_obj.bind((self.host, self.port))
        self.server_obj.listen(32)

        self.active_clients = {}
        self.lock = threading.RLock()
        self.update_cv = threading.Condition(self.lock)
        self.round_updates = {}
        self.current_round = 0
        self.stop_event = threading.Event()
        self.stop_reason = None
        self.min_clients = int(self.config.get("network", {}).get("min_clients", 1))
        self.dropped_clients = set()
        self.net_stats = {
            "bytes_sent": 0,
            "bytes_recv": 0,
            "messages_sent": 0,
            "messages_recv": 0,
            "sent_by_type": {},
            "recv_by_type": {},
        }

        self.criterion = nn.CrossEntropyLoss()
        if self.mode == "splitfed":
            self.global_client_model, self.server_model = get_model("splitfed")
            self.global_client_model.to(self.device)
            self.server_model.to(self.device)
            self.server_optimizer = torch.optim.SGD(
                self.server_model.parameters(),
                lr=self.server_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            self.split_lock = threading.Lock()
        else:
            self.global_model = get_model("centralized")
            self.global_model.to(self.device)

        self.test_loader = self._load_server_test_loader()
        self.monitor.post(
            "startup",
            mode=self.mode,
            device=str(self.device),
            preferred_device=preferred_device,
            topology=compact_topology(self.config),
            experiment=self.config.get("experiment", {}),
            network=self.config.get("network", {}),
            test_samples=len(self.test_loader.dataset),
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

    def _load_server_test_loader(self):
        path = os.path.join(self.project_root, "data", "splits", "server_test_data.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Please run scripts/prepare_mnist.py first.")
        payload = torch.load(path, map_location="cpu")
        x = self._normalize(payload["images"])
        y = payload["labels"].long()
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def report_metric(self, metric_data):
        payload = dict(metric_data)
        payload.setdefault("mode", self.mode)
        self.monitor.post("metric", **payload)

    def _count_type(self, bucket, key, inc):
        bucket[key] = int(bucket.get(key, 0)) + int(inc)

    def _record_network(self, direction, message_type, payload_bytes, peer="", round_id=None, extra=None):
        with self.lock:
            if direction == "out":
                self.net_stats["bytes_sent"] += int(payload_bytes)
                self.net_stats["messages_sent"] += 1
                self._count_type(self.net_stats["sent_by_type"], message_type, 1)
            else:
                self.net_stats["bytes_recv"] += int(payload_bytes)
                self.net_stats["messages_recv"] += 1
                self._count_type(self.net_stats["recv_by_type"], message_type, 1)
            totals = {
                "bytes_sent_total": self.net_stats["bytes_sent"],
                "bytes_recv_total": self.net_stats["bytes_recv"],
                "messages_sent_total": self.net_stats["messages_sent"],
                "messages_recv_total": self.net_stats["messages_recv"],
            }
        payload = {
            "mode": self.mode,
            "direction": direction,
            "peer": peer,
            "message_type": message_type,
            "payload_label": payload_label(message_type),
            "payload_bytes": int(payload_bytes),
            "round": round_id,
            **totals,
        }
        if extra:
            payload.update(extra)
        self.monitor.post("network_io", **payload)
        logging.log(
            logging.DEBUG if direction in {"out", "in"} else logging.INFO,
            "net direction=%s peer=%s msg=%s label=%s payload=%dB tx=%dB rx=%dB tx_msg=%d rx_msg=%d round=%s",
            direction,
            peer or "-",
            message_type,
            payload["payload_label"],
            int(payload_bytes),
            int(totals["bytes_sent_total"]),
            int(totals["bytes_recv_total"]),
            int(totals["messages_sent_total"]),
            int(totals["messages_recv_total"]),
            round_id if round_id is not None else "-",
        )

    def _send(self, conn, payload, peer="", round_id=None, extra=None):
        ok, size = self.communicator.send_data(conn, payload)
        msg_type = payload.get("type", "unknown")
        self._record_network("out", msg_type, size, peer=peer, round_id=round_id, extra=extra)
        self.monitor.post(
            "send_ack",
            mode=self.mode,
            peer=peer,
            message_type=msg_type,
            payload_label=payload_label(msg_type),
            payload_bytes=int(size),
            send_success=bool(ok),
            round=round_id,
        )
        return ok, size

    def _recv(self, conn, peer_hint=""):
        msg, meta = self.communicator.recv_data_with_meta(conn)
        if msg is None:
            return None, None
        msg_type = msg.get("type", "unknown")
        payload_bytes = int((meta or {}).get("payload_bytes", 0))
        round_id = msg.get("round")
        self._record_network("in", msg_type, payload_bytes, peer=peer_hint, round_id=round_id)
        self.monitor.post(
            "recv_ack",
            mode=self.mode,
            peer=peer_hint,
            message_type=msg_type,
            payload_label=payload_label(msg_type),
            payload_bytes=payload_bytes,
            recv_success=True,
            round=round_id,
            magic_ok=bool((meta or {}).get("magic_ok", False)),
            compression=bool((meta or {}).get("compression", False)),
        )
        return msg, meta

    def _snapshot_net(self):
        with self.lock:
            return {
                "bytes_sent": int(self.net_stats["bytes_sent"]),
                "bytes_recv": int(self.net_stats["bytes_recv"]),
                "messages_sent": int(self.net_stats["messages_sent"]),
                "messages_recv": int(self.net_stats["messages_recv"]),
            }

    @staticmethod
    def _delta_net(after, before):
        return {
            "bytes_sent": after["bytes_sent"] - before["bytes_sent"],
            "bytes_recv": after["bytes_recv"] - before["bytes_recv"],
            "messages_sent": after["messages_sent"] - before["messages_sent"],
            "messages_recv": after["messages_recv"] - before["messages_recv"],
        }

    def _aggregate_weighted(self, updates):
        if not updates:
            return None
        total = sum(max(int(u["num_samples"]), 1) for u in updates)
        agg = {}
        keys = updates[0]["weights"].keys()
        for key in keys:
            weighted = None
            for item in updates:
                w = max(int(item["num_samples"]), 1) / total
                t = item["weights"][key].float() * w
                weighted = t if weighted is None else weighted + t
            agg[key] = weighted
        return agg

    def _evaluate_centralized(self):
        self.global_model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.global_model(x)
                loss = self.criterion(logits, y)
                loss_sum += float(loss.item()) * y.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += y.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def _evaluate_splitfed(self):
        self.global_client_model.eval()
        self.server_model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                smashed = self.global_client_model(x)
                logits = self.server_model(smashed)
                loss = self.criterion(logits, y)
                loss_sum += float(loss.item()) * y.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += y.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def handle_client(self, conn, addr):
        client_id = None
        logging.info("Connected: %s", addr)
        try:
            while not self.stop_event.is_set():
                msg, _ = self._recv(conn, peer_hint=client_id or str(addr))
                if msg is None:
                    break
                msg_type = msg.get("type")

                if msg_type == "register":
                    client_id = msg.get("client_id")
                    with self.lock:
                        self.active_clients[client_id] = conn
                    self._send(conn, {"type": "register_ack", "mode": self.mode}, peer=client_id)
                    self.monitor.post(
                        "topology_update",
                        active_clients=sorted(self.active_clients.keys()),
                        active_count=len(self.active_clients),
                        expected_count=self.num_clients,
                        client_id=client_id,
                    )
                    logging.info("Client %s registered.", client_id)
                    continue

                if msg_type in {"model_update", "split_update"}:
                    round_id = int(msg.get("round", -1))
                    with self.update_cv:
                        if round_id not in self.round_updates:
                            self.round_updates[round_id] = {}
                        self.round_updates[round_id][client_id] = msg
                        self.update_cv.notify_all()
                    self.monitor.post(
                        "client_round_update",
                        mode=self.mode,
                        round=round_id,
                        client_id=client_id,
                        num_samples=int(msg.get("num_samples", 0)),
                        train_loss=float(msg.get("train_loss", 0.0)),
                        train_acc=float(msg.get("train_acc", 0.0)),
                        test_acc=float(msg.get("test_acc", 0.0)),
                    )
                    continue

                if msg_type == "split_batch":
                    self._handle_split_batch(conn, client_id, msg)
                    continue

                logging.warning("Unknown message type from %s: %s", client_id, msg_type)
        except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
            logging.error("Client handler error (%s): %s", client_id or addr, exc)
        finally:
            with self.lock:
                if client_id in self.active_clients:
                    self.active_clients.pop(client_id, None)
            conn.close()
            self.monitor.post(
                "client_disconnect",
                mode=self.mode,
                client_id=client_id,
                addr=str(addr),
                active_count=len(self.active_clients),
            )
            logging.info("Connection closed: %s", client_id or addr)

    def _handle_split_batch(self, conn, client_id, msg):
        round_id = int(msg.get("round", -1))
        if round_id != self.current_round:
            self._send(
                conn,
                {
                    "type": "split_grad",
                    "round": round_id,
                    "error": f"stale round, server current round is {self.current_round}",
                },
                peer=client_id,
                round_id=round_id,
            )
            return

        activations = msg["activations"].to(self.device).detach().requires_grad_(True)
        labels = msg["labels"].to(self.device).long()

        with self.split_lock:
            self.server_model.train()
            self.server_optimizer.zero_grad()
            logits = self.server_model(activations)
            loss = self.criterion(logits, labels)
            loss.backward()
            grad = activations.grad.detach().cpu()
            self.server_optimizer.step()
            correct = int((logits.argmax(dim=1) == labels).sum().item())

        self.monitor.post(
            "split_batch_processed",
            mode=self.mode,
            round=round_id,
            client_id=client_id,
            batch_size=int(labels.size(0)),
            loss=float(loss.item()),
            correct=correct,
            acc=float(correct) / max(int(labels.size(0)), 1),
        )
        self._send(
            conn,
            {
                "type": "split_grad",
                "round": round_id,
                "grad": grad,
                "loss": float(loss.item()),
                "correct": correct,
                "batch_size": int(labels.size(0)),
                "client_id": client_id,
            },
            peer=client_id,
            round_id=round_id,
            extra={"batch_size": int(labels.size(0))},
        )

    def _accept_loop(self):
        while not self.stop_event.is_set():
            try:
                conn, addr = self.server_obj.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                thread.start()
            except OSError:
                break

    def _wait_for_all_clients(self):
        while len(self.active_clients) < self.num_clients:
            logging.info("Waiting clients: %d/%d", len(self.active_clients), self.num_clients)
            self.monitor.post(
                "wait_clients",
                mode=self.mode,
                active_count=len(self.active_clients),
                expected_count=self.num_clients,
                active_clients=sorted(self.active_clients.keys()),
            )
            time.sleep(1)

    def _wait_round_updates(self, round_id):
        """Wait for client updates with timeout-based straggler dropping.

        If timeout expires before all clients respond, the server drops
        unresponsive clients for this round and aggregates only the
        received updates, as long as at least min_clients responded.
        """
        started = time.time()
        deadline = time.time() + self.timeout
        with self.update_cv:
            while len(self.round_updates.get(round_id, {})) < self.num_clients:
                remain = deadline - time.time()
                if remain <= 0:
                    break
                self.update_cv.wait(timeout=remain)
            got = dict(self.round_updates.get(round_id, {}))

        # Identify dropped (straggler) clients
        expected = set(self.active_clients.keys())
        responded = set(got.keys())
        dropped = expected - responded
        round_dropped = set()
        if dropped:
            for cid in dropped:
                round_dropped.add(cid)
                self.dropped_clients.add(cid)
                logging.warning(
                    "STRAGGLER DROP: client %s timed out after %.1fs in round %d",
                    cid,
                    self.timeout,
                    round_id,
                )
                self.monitor.post(
                    "straggler_dropped",
                    mode=self.mode,
                    round=round_id,
                    client_id=cid,
                    timeout_seconds=self.timeout,
                    reason="update_timeout",
                )

        can_proceed = len(got) >= self.min_clients
        self.monitor.post(
            "round_wait_result",
            mode=self.mode,
            round=round_id,
            wait_seconds=time.time() - started,
            received_count=len(got),
            expected_count=self.num_clients,
            clients=sorted(got.keys()),
            dropped_clients=sorted(round_dropped),
            can_proceed=can_proceed,
            timeout_seconds=self.timeout,
        )
        return got

    def _broadcast(self, payload):
        with self.lock:
            for client_id, conn in list(self.active_clients.items()):
                ok, size = self._send(conn, payload, peer=client_id, round_id=payload.get("round"))
                if not ok:
                    logging.warning("Failed to send payload to %s", client_id)
                else:
                    logging.info("Broadcast to %s payload=%d bytes", client_id, size)

    def _run_round_centralized(self, round_id):
        self.monitor.post(
            "round_start",
            mode="centralized",
            round=round_id,
            total_epochs=self.max_epochs,
            expected_clients=self.num_clients,
            local_epochs=self.local_epochs,
            client_lr=self.client_lr,
            batch_size=self.batch_size,
        )
        self.round_updates[round_id] = {}
        self._broadcast(
            {
                "type": "round_start_centralized",
                "round": round_id,
                "weights": {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()},
                "local_epochs": self.local_epochs,
                "lr": self.client_lr,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
            }
        )

        updates = self._wait_round_updates(round_id)
        update_list = list(updates.values())
        if len(update_list) < self.num_clients:
            dropped = self.num_clients - len(update_list)
            logging.warning(
                "Round %d straggler handling: received %d/%d updates, dropped %d client(s)",
                round_id,
                len(update_list),
                self.num_clients,
                dropped,
            )
            if len(update_list) < self.min_clients:
                logging.error(
                    "Round %d aborted: only %d update(s) received, need at least %d",
                    round_id,
                    len(update_list),
                    self.min_clients,
                )
                self.monitor.post(
                    "round_aborted",
                    mode="centralized",
                    round=round_id,
                    reason="insufficient_client_updates",
                    received_updates=len(update_list),
                    expected_count=self.num_clients,
                    min_clients=self.min_clients,
                    dropped_clients=dropped,
                )
                self.stop_reason = "insufficient_client_updates"
                return 0.0

        agg = self._aggregate_weighted(update_list)
        if agg is not None:
            self.global_model.load_state_dict(agg)

        test_loss, test_acc = self._evaluate_centralized()
        
        # Aggregate client metrics
        total_samples = sum(int(x.get("num_samples", 0)) for x in update_list)
        if total_samples > 0:
            train_loss = sum(float(x.get("train_loss", 0.0)) * int(x.get("num_samples", 0)) for x in update_list) / total_samples
            train_acc = sum(float(x.get("train_acc", 0.0)) * int(x.get("num_samples", 0)) for x in update_list) / total_samples
            val_loss = sum(float(x.get("val_loss", 0.0)) * int(x.get("num_samples", 0)) for x in update_list) / total_samples
            val_acc = sum(float(x.get("val_acc", 0.0)) * int(x.get("num_samples", 0)) for x in update_list) / total_samples
        else:
            train_loss = train_acc = val_loss = val_acc = None
        
        self.report_metric(
            {
                "source": "server",
                "mode": "centralized",
                "round": round_id,
                "type": "global_eval",
                "test_loss": test_loss,
                "test_acc": test_acc,
                "received_updates": len(update_list),
            }
        )
        self.monitor.post(
            "round_end",
            mode="centralized",
            round=round_id,
            total_epochs=self.max_epochs,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            received_updates=len(update_list),
            expected_updates=self.num_clients,
            dropped_clients=max(self.num_clients - len(update_list), 0),
            sample_total=total_samples,
        )
        logging.info("Round %d centralized eval: loss=%.4f acc=%.4f", round_id, test_loss, test_acc)
        return test_acc

    def _run_round_splitfed(self, round_id):
        self.monitor.post(
            "round_start",
            mode="splitfed",
            round=round_id,
            total_epochs=self.max_epochs,
            expected_clients=self.num_clients,
            local_epochs=self.local_epochs,
            client_lr=self.client_lr,
            server_lr=self.server_lr,
            batch_size=self.batch_size,
        )
        self.round_updates[round_id] = {}
        self._broadcast(
            {
                "type": "round_start_splitfed",
                "round": round_id,
                "client_weights": {k: v.detach().cpu() for k, v in self.global_client_model.state_dict().items()},
                "server_weights": {k: v.detach().cpu() for k, v in self.server_model.state_dict().items()},
                "local_epochs": self.local_epochs,
                "lr": self.client_lr,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
            }
        )

        updates = self._wait_round_updates(round_id)
        update_list = list(updates.values())
        if len(update_list) < self.num_clients:
            dropped = self.num_clients - len(update_list)
            logging.warning(
                "Round %d splitfed straggler handling: received %d/%d updates, dropped %d client(s)",
                round_id,
                len(update_list),
                self.num_clients,
                dropped,
            )
            if len(update_list) < self.min_clients:
                logging.error(
                    "Round %d splitfed aborted: only %d update(s) received, need at least %d",
                    round_id,
                    len(update_list),
                    self.min_clients,
                )
                self.monitor.post(
                    "round_aborted",
                    mode="splitfed",
                    round=round_id,
                    reason="insufficient_client_updates",
                    received_updates=len(update_list),
                    expected_count=self.num_clients,
                    min_clients=self.min_clients,
                    dropped_clients=dropped,
                )
                self.stop_reason = "insufficient_client_updates"
                return 0.0

        agg = self._aggregate_weighted(update_list)
        if agg is not None:
            self.global_client_model.load_state_dict(agg)

        test_loss, test_acc = self._evaluate_splitfed()
        
        # Aggregate client metrics
        total_samples = sum(int(x.get("num_samples", 0)) for x in update_list)
        if total_samples > 0:
            train_loss = sum(float(x.get("train_loss", 0.0)) * int(x.get("num_samples", 0)) for x in update_list) / total_samples
            train_acc = sum(float(x.get("train_acc", 0.0)) * int(x.get("num_samples", 0)) for x in update_list) / total_samples
            val_loss = sum(float(x.get("val_loss", 0.0)) * int(x.get("num_samples", 0)) for x in update_list) / total_samples
            val_acc = sum(float(x.get("val_acc", 0.0)) * int(x.get("num_samples", 0)) for x in update_list) / total_samples
        else:
            train_loss = train_acc = val_loss = val_acc = None
        
        self.report_metric(
            {
                "source": "server",
                "mode": "splitfed",
                "round": round_id,
                "type": "global_eval",
                "test_loss": test_loss,
                "test_acc": test_acc,
                "received_updates": len(update_list),
            }
        )
        self.monitor.post(
            "round_end",
            mode="splitfed",
            round=round_id,
            total_epochs=self.max_epochs,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            received_updates=len(update_list),
            expected_updates=self.num_clients,
            dropped_clients=max(self.num_clients - len(update_list), 0),
            sample_total=total_samples,
        )
        logging.info("Round %d splitfed eval: loss=%.4f acc=%.4f", round_id, test_loss, test_acc)
        return test_acc

    def run(self):
        logging.info("Server start on %s:%d mode=%s", self.host, self.port, self.mode)
        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()
        self._wait_for_all_clients()
        logging.info("All clients registered. Start training.")

        try:
            completed_all_rounds = True
            for round_id in range(1, self.max_epochs + 1):
                self.current_round = round_id
                started = time.time()
                net_before = self._snapshot_net()

                if self.mode == "splitfed":
                    test_acc = self._run_round_splitfed(round_id)
                else:
                    test_acc = self._run_round_centralized(round_id)

                elapsed = time.time() - started
                net_after = self._snapshot_net()
                net_delta = self._delta_net(net_after, net_before)
                self.monitor.post(
                    "round_transport",
                    mode=self.mode,
                    round=round_id,
                    total_epochs=self.max_epochs,
                    elapsed_seconds=elapsed,
                    bytes_sent=net_delta["bytes_sent"],
                    bytes_recv=net_delta["bytes_recv"],
                    messages_sent=net_delta["messages_sent"],
                    messages_recv=net_delta["messages_recv"],
                )
                logging.info("Round %d finished in %.2fs", round_id, elapsed)
                if self.stop_reason:
                    completed_all_rounds = False
                    break
                if test_acc >= self.target_accuracy:
                    self.stop_reason = "target_accuracy_reached"
                    logging.info("Target accuracy reached: %.4f >= %.4f", test_acc, self.target_accuracy)
                    self.monitor.post(
                        "target_reached",
                        mode=self.mode,
                        round=round_id,
                        target_accuracy=self.target_accuracy,
                        actual_accuracy=test_acc,
                    )
                    completed_all_rounds = False
                    break
        finally:
            if self.stop_reason is None:
                self.stop_reason = "rounds_completed" if completed_all_rounds else "stopped"
            self.stop_event.set()
            self._broadcast({"type": "shutdown", "reason": self.stop_reason})
            self.monitor.post(
                "shutdown",
                mode=self.mode,
                reason=self.stop_reason,
                net_totals=self._snapshot_net(),
                active_clients=sorted(self.active_clients.keys()),
            )
            self.server_obj.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Central server for federated learning")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--data-path", type=str, default=None, help="Project root path containing data (overrides default)")
    args = parser.parse_args()

    server = Server(args.config, project_root=args.data_path)
    server.run()
