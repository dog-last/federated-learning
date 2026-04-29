import subprocess
import json
import time
import os
import signal
import torch
import sys
import argparse

from utils.monitoring import MonitorReporter, compact_topology


def _project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _logs_dir(root):
    path = os.path.join(root, "logs")
    os.makedirs(path, exist_ok=True)
    return path


def _open_role_log(root, role):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(_logs_dir(root), f"{role}-{stamp}.log")
    return open(path, "a", encoding="utf-8")


def _python_bin():
    return os.environ.get("PYTHON_BIN", sys.executable)


def _is_mnist_split_compatible(required_files):
    try:
        client_payload = torch.load(required_files[0], map_location="cpu")
        server_payload = torch.load(required_files[-1], map_location="cpu")

        client_images = client_payload["train_images"]
        server_images = server_payload["images"]
        return client_images.ndim == 4 and client_images.shape[1:] == (1, 28, 28) and server_images.shape[1:] == (1, 28, 28)
    except (OSError, RuntimeError, KeyError, IndexError, ValueError, TypeError):
        return False


def _ensure_data(root, py_bin):
    required = [
        os.path.join(root, "data", "splits", "client_1_data.pt"),
        os.path.join(root, "data", "splits", "client_2_data.pt"),
        os.path.join(root, "data", "splits", "client_3_data.pt"),
        os.path.join(root, "data", "splits", "server_test_data.pt"),
    ]
    if all(os.path.exists(path) for path in required) and _is_mnist_split_compatible(required):
        return {"prepared": False, "files": required}

    print("Dataset splits not found. Running data preparation...")
    started = time.time()
    subprocess.run([py_bin, "scripts/prepare_mnist.py"], cwd=root, check=True)

    return {
        "prepared": True,
        "elapsed_seconds": time.time() - started,
        "files": required,
    }


def _proc_info(proc, role):
    return {
        "role": role,
        "pid": proc.pid,
        "alive": proc.poll() is None,
        "exit_code": proc.poll(),
    }


def _graceful_stop(proc, wait_seconds=2.0):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            proc.kill()

def start_experiment(config_path=None):
    root = _project_root()
    py_bin = _python_bin()
    log_handles = []

    if config_path is None:
        config_path = os.path.join(root, 'config.json')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    mode = config.get("experiment", {}).get("mode", "centralized")
    data_prepare = _ensure_data(root, py_bin)

    env = os.environ.copy()
    env["PYTHONPATH"] = root

    monitor_proc = subprocess.Popen(
        [
            py_bin,
            "-m",
            "uvicorn",
            "utils.monitor_api:app",
            "--host",
            config['monitoring']['api_host'],
            "--port",
            str(config['monitoring']['api_port']),
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        env=env,
        cwd=root,
    )

    time.sleep(2)
    reporter = MonitorReporter(config['monitoring']['api_host'], int(config['monitoring']['api_port']), "manager")
    reporter.post(
        "manager_start",
        topology=compact_topology(config),
        experiment=config.get("experiment", {}),
        network=config.get("network", {}),
        data_prepare=data_prepare,
        monitor_process=_proc_info(monitor_proc, "monitor"),
        config_path=config_path,
    )

    if mode == "ring":
        _start_ring_mode(root, py_bin, env, config, config_path, reporter, log_handles, monitor_proc)
    else:
        _start_centralized_mode(root, py_bin, env, config, config_path, reporter, log_handles, monitor_proc)


def _start_ring_mode(root, py_bin, env, config, config_path, reporter, log_handles, monitor_proc):
    """Launch decentralized ring topology: 3 ring nodes, no central server."""
    nodes = config["topology"]["nodes"]
    node_procs = []

    for node_cfg in nodes:
        node_id = str(node_cfg["id"])
        node_log = _open_role_log(root, f"ring-node-{node_id}")
        log_handles.append(node_log)
        proc = subprocess.Popen(
            [py_bin, "-m", "core.ring_node", node_id, "--config", config_path, "--data-path", root],
            env=env,
            cwd=root,
            stdout=node_log,
            stderr=subprocess.STDOUT,
        )
        node_procs.append(proc)
        reporter.post(
            "process_spawn",
            process=_proc_info(proc, f"ring_node_{node_id}"),
            node_id=node_id,
        )

    reporter.post(
        "ring_mode_started",
        node_count=len(nodes),
        nodes=[{"id": n["id"], "addr": f"{n['host']}:{n['port']}"} for n in nodes],
    )

    try:
        # Wait for first node (initiator) to finish — it controls the ring lifecycle
        if node_procs:
            node_procs[0].wait()
    except KeyboardInterrupt:
        print("Terminating ring experiment...")
    finally:
        reporter.post(
            "manager_stopping",
            mode="ring",
            processes=[_proc_info(monitor_proc, "monitor")]
            + [_proc_info(p, f"ring_node_{i+1}") for i, p in enumerate(node_procs)],
        )
        for p in node_procs:
            if p is not None and p.poll() is None:
                _graceful_stop(p, wait_seconds=2.0)
        reporter.post(
            "manager_stop",
            mode="ring",
            processes=[_proc_info(monitor_proc, "monitor")]
            + [_proc_info(p, f"ring_node_{i+1}") for i, p in enumerate(node_procs)],
        )
        for h in log_handles:
            h.close()
        _graceful_stop(monitor_proc, wait_seconds=0.5)


def _start_centralized_mode(root, py_bin, env, config, config_path, reporter, log_handles, monitor_proc):
    """Launch centralized or splitfed mode: 1 server + N clients."""
    server_log = _open_role_log(root, "server")
    log_handles.append(server_log)
    server_proc = subprocess.Popen(
        [py_bin, "-m", "core.server", "--config", config_path, "--data-path", root],
        env=env,
        cwd=root,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    reporter.post("process_spawn", process=_proc_info(server_proc, "server"))

    time.sleep(1)

    client_procs = []
    for client in config['topology']['clients']:
        cid = client['id']
        client_log = _open_role_log(root, f"client-{cid}")
        log_handles.append(client_log)
        proc = subprocess.Popen(
            [py_bin, "-m", "core.client", cid, "--config", config_path, "--data-path", root],
            env=env,
            cwd=root,
            stdout=client_log,
            stderr=subprocess.STDOUT,
        )
        client_procs.append(proc)
        reporter.post("process_spawn", process=_proc_info(proc, f"client:{cid}"), client_id=cid)

    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("Terminating experiment...")
    finally:
        reporter.post(
            "manager_stopping",
            processes=[_proc_info(monitor_proc, "monitor"), _proc_info(server_proc, "server")]
            + [_proc_info(p, "client") for p in client_procs],
        )
        worker_procs = [server_proc, *client_procs]
        for p in worker_procs:
            if p is not None and p.poll() is None:
                try:
                    p.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    _graceful_stop(p, wait_seconds=1.0)
        reporter.post(
            "manager_stop",
            processes=[_proc_info(monitor_proc, "monitor"), _proc_info(server_proc, "server")]
            + [_proc_info(p, "client") for p in client_procs],
        )
        for h in log_handles:
            h.close()
        _graceful_stop(monitor_proc, wait_seconds=0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Manager")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    args = parser.parse_args()
    # If config path is not provided, it will default to 'config.json' in the same directory as this script
    if args.config is None:
        args.config = os.path.join(_project_root(), 'config.json')
    print(f"Using config file: {args.config}")
    start_experiment(config_path=args.config)
