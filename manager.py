import subprocess
import json
import time
import os
import signal
import torch
import sys
import argparse

from utils.monitoring import MonitorReporter, compact_topology, wait_for_monitor_ready


def _project_root():
    return os.path.dirname(os.path.abspath(__file__))


def _logs_dir(root):
    path = os.path.join(root, "logs")
    os.makedirs(path, exist_ok=True)
    return path


def _new_run_id():
    return time.strftime("%Y%m%d%H%M%S")


def _run_logs_dir(root, run_id):
    path = os.path.join(_logs_dir(root), run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _open_role_log(root, run_id, role):
    path = os.path.join(_run_logs_dir(root, run_id), f"{role}.log")
    handle = open(path, "a", encoding="utf-8")
    handle.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - LOG - INFO - "
        f"Opened process log for role={role}, run_id={run_id}\n"
    )
    handle.flush()
    return handle


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


def _wait_monitor_or_raise(config, monitor_proc, timeout: float = 10.0):
    host = config["monitoring"]["api_host"]
    port = int(config["monitoring"]["api_port"])
    if wait_for_monitor_ready(host, port, timeout=timeout):
        return

    proc_state = _proc_info(monitor_proc, "monitor")
    raise RuntimeError(
        f"Monitor service failed to become ready within {timeout:.1f}s "
        f"at http://{host}:{port}/health, process={proc_state}"
    )

def start_experiment(config_path=None):
    root = _project_root()
    py_bin = _python_bin()
    run_id = _new_run_id()
    run_log_dir = _run_logs_dir(root, run_id)
    log_handles = []

    if config_path is None:
        config_path = os.path.join(root, 'config.json')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    mode = config.get("experiment", {}).get("mode", "centralized")
    data_prepare = _ensure_data(root, py_bin)

    env = os.environ.copy()
    env["PYTHONPATH"] = root
    env["FED_CONFIG_PATH"] = os.path.abspath(config_path)
    env["FED_RUN_ID"] = run_id
    env["FED_LOG_DIR"] = run_log_dir
    env.setdefault("FED_LOG_LEVEL", "INFO")

    monitor_log = _open_role_log(root, run_id, "monitor")
    log_handles.append(monitor_log)
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
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
    )

    monitor_ready_timeout = float(os.environ.get("FED_MONITOR_READY_TIMEOUT", "10.0"))
    _wait_monitor_or_raise(config, monitor_proc, timeout=monitor_ready_timeout)
    reporter = MonitorReporter(
        config['monitoring']['api_host'],
        int(config['monitoring']['api_port']),
        "manager",
        timeout=3.0,
    )
    reporter.post_best_effort(
        "manager_start",
        topology=compact_topology(config),
        experiment=config.get("experiment", {}),
        network=config.get("network", {}),
        data_prepare=data_prepare,
        monitor_process=_proc_info(monitor_proc, "monitor"),
        config_path=config_path,
        run_id=run_id,
        log_dir=run_log_dir,
    )

    if mode == "ring":
        _start_ring_mode(root, run_id, py_bin, env, config, config_path, reporter, log_handles, monitor_proc)
    else:
        _start_centralized_mode(root, run_id, py_bin, env, config, config_path, reporter, log_handles, monitor_proc)


def _start_ring_mode(root, run_id, py_bin, env, config, config_path, reporter, log_handles, monitor_proc):
    """Launch decentralized ring topology: 3 ring nodes, no central server."""
    nodes = config["topology"]["nodes"]
    node_procs = []

    for node_cfg in nodes:
        node_id = str(node_cfg["id"])
        node_log = _open_role_log(root, run_id, f"ring-node-{node_id}")
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


def _start_centralized_mode(root, run_id, py_bin, env, config, config_path, reporter, log_handles, monitor_proc):
    """Launch centralized or splitfed mode: 1 server + N clients."""
    server_log = _open_role_log(root, run_id, "server")
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
        client_log = _open_role_log(root, run_id, f"client-{cid}")
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
