"""One-command federated learning launcher.

Reads a config file, pre-downloads data, partitions it, then starts
all processes (server + clients or P2P peers).

Usage:
    python scripts/run_fed.py --config config/centralized.yaml
    python scripts/run_fed.py --config config/decentralized.yaml
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import torch

from src.core.types import Config
from src.data.dataset import load_and_partition


def _ensure_dirs(data_dir: str) -> str:
    """Ensure partition directory exists and return its path."""
    part_dir = os.path.join(data_dir, "partitioned")
    os.makedirs(part_dir, exist_ok=True)
    return part_dir


def _pre_download_and_partition(config: Config) -> str:
    """Download dataset and save partitions. Returns partition dir path.

    Args:
        config: Parsed configuration.

    Returns:
        str: Path to the partitioned data directory.
    """
    part_dir = _ensure_dirs(config.dataset.data_dir)

    # Check if partitions already exist
    first_partition = os.path.join(part_dir, "client_1.pt")
    if os.path.exists(first_partition):
        print("[Phase 1/3] Dataset already partitioned, skipping download")
        return part_dir

    # Phase 1: Download
    print(f"[Phase 1/3] Downloading {config.dataset.name} dataset...")
    client_datasets, test_dataset = load_and_partition(
        name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        num_clients=config.dataset.num_clients,
        strategy=config.dataset.partition_strategy,
        alpha=config.dataset.alpha,
    )

    # Phase 2: Save partitions
    print(
        f"[Phase 2/3] Partitioning data for {config.dataset.num_clients} clients ({config.dataset.partition_strategy})..."
    )
    for cid, ds in enumerate(client_datasets):
        path = os.path.join(part_dir, f"client_{cid + 1}.pt")
        torch.save(ds, path)
        print(f"  Client {cid + 1}: {len(ds)} samples -> {path}")

    test_path = os.path.join(part_dir, "test.pt")
    torch.save(test_dataset, test_path)
    print(f"  Test set: {len(test_dataset)} samples -> {test_path}")

    return part_dir


def _start_process(args: list[str], log_file: str | None = None) -> subprocess.Popen:
    """Start a subprocess with optional log file output.

    Args:
        args: Command line arguments.
        log_file: Optional path to log file for stdout/stderr redirection.

    Returns:
        subprocess.Popen: The started process.
    """
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            f.write(f"=== Process started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        stdout = open(log_file, "a")  # noqa: SIM115
        stderr = subprocess.STDOUT
    else:
        stdout = None
        stderr = None

    proc = subprocess.Popen([sys.executable, *args], stdout=stdout, stderr=stderr)
    proc._log_file_handle = stdout if log_file else None
    return proc


def _terminate_all(processes: list[subprocess.Popen]) -> None:
    """Terminate all child processes gracefully."""
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 5.0
    for proc in processes:
        remaining = deadline - time.time()
        if remaining > 0:
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
    for proc in processes:
        proc.wait()
        # Close log file handle if opened
        if hasattr(proc, "_log_file_handle") and proc._log_file_handle:
            proc._log_file_handle.close()


def _get_timestamped_log_dir(base_log_dir: str) -> str:
    """Create a timestamped log directory.

    Args:
        base_log_dir: Base log directory path.

    Returns:
        str: Path to timestamped log directory.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def run_centralized(config: Config, config_path: str, part_dir: str) -> None:
    """Launch server + all clients for centralized mode."""
    processes: list[subprocess.Popen] = []
    labels: dict[int, str] = {}

    def _shutdown(sig: int, frame: object) -> None:
        print("\nShutting down all processes...")
        _terminate_all(processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    assert config.server is not None
    assert config.clients is not None

    # Create timestamped log directory
    log_dir = _get_timestamped_log_dir(config.logging.log_dir)

    print("[Phase 3/3] Starting centralized training...")
    print(f"  Logs will be saved to: {log_dir}")

    # Start server
    print(f"  Starting server on {config.server.host}:{config.server.port}")
    server_log = os.path.join(log_dir, "server.log")
    server_proc = _start_process(
        [
            "scripts/run_server.py",
            "--config",
            config_path,
            "--partition-dir",
            part_dir,
        ],
        log_file=server_log,
    )
    processes.append(server_proc)
    labels[server_proc.pid] = "Server"
    print(f"    Server PID: {server_proc.pid}, Log: {server_log}")

    # Wait for server to bind
    time.sleep(2.0)

    # Start clients
    for node in config.clients.nodes:
        print(f"  Starting client {node.id}")
        client_log = os.path.join(log_dir, f"client_{node.id}.log")
        client_proc = _start_process(
            [
                "scripts/run_client.py",
                "--config",
                config_path,
                "--client-id",
                str(node.id),
                "--partition-dir",
                part_dir,
            ],
            log_file=client_log,
        )
        processes.append(client_proc)
        labels[client_proc.pid] = f"Client-{node.id}"
        print(f"    Client-{node.id} PID: {client_proc.pid}, Log: {client_log}")

    # Wait for all processes to finish
    try:
        _wait_for_all(processes, labels)
    except KeyboardInterrupt:
        _terminate_all(processes)

    for proc in processes:
        label = labels.get(proc.pid, "Unknown")
        code = proc.returncode
        print(f"{label} (PID {proc.pid}) exited with code {code}")


def _wait_for_all(processes: list[subprocess.Popen], labels: dict[int, str]) -> None:
    """Wait for all processes to complete, checking them in a round-robin fashion."""
    remaining = set(processes)
    while remaining:
        done = set()
        for proc in remaining:
            ret = proc.poll()
            if ret is not None:
                done.add(proc)
                label = labels.get(proc.pid, "Unknown")
                print(f"{label} (PID {proc.pid}) exited with code {ret}")
        remaining -= done
        if remaining:
            time.sleep(0.1)


def run_decentralized(config: Config, config_path: str, part_dir: str) -> None:
    """Launch all P2P peers for decentralized mode."""
    processes: list[subprocess.Popen] = []
    labels: dict[int, str] = {}

    def _shutdown(sig: int, frame: object) -> None:
        print("\nShutting down all processes...")
        _terminate_all(processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    assert config.peers is not None

    nodes = config.peers.nodes
    if not nodes:
        print("Error: no peers configured")
        sys.exit(1)

    # Create timestamped log directory
    log_dir = _get_timestamped_log_dir(config.logging.log_dir)

    print("[Phase 3/3] Starting decentralized training...")
    print(f"  Logs will be saved to: {log_dir}")

    if not config.peers.local:
        # Distributed mode: only start local peers
        local_nodes = [n for n in nodes if n.host in ("127.0.0.1", "localhost")]
        remote_nodes = [n for n in nodes if n.host not in ("127.0.0.1", "localhost")]
        if remote_nodes:
            print("  Remote peers (start manually on their machines):")
            for n in remote_nodes:
                print(
                    f"    Peer-{n.id}: python scripts/run_p2p_node.py --config {config_path} "
                    f"--node-id {n.id} --port {n.port} --partition-dir <local_part_dir>"
                )
            print()
        if not local_nodes:
            print("  No local peers to start. Exiting.")
            return
        nodes = local_nodes

    # Start first node (bootstrap)
    first = nodes[0]
    print(f"  Starting peer {first.id} on {first.host}:{first.port}")
    first_log = os.path.join(log_dir, f"node_{first.id}.log")
    first_proc = _start_process(
        [
            "scripts/run_p2p_node.py",
            "--config",
            config_path,
            "--node-id",
            str(first.id),
            "--port",
            str(first.port),
            "--partition-dir",
            part_dir,
        ],
        log_file=first_log,
    )
    processes.append(first_proc)
    labels[first_proc.pid] = f"Peer-{first.id}"
    print(f"    Peer-{first.id} PID: {first_proc.pid}, Log: {first_log}")

    # Wait for first node to be ready
    time.sleep(1.0)

    # Start remaining nodes with bootstrap
    bootstrap_addr = f"{first.host}:{first.port}"
    for node in nodes[1:]:
        print(f"  Starting peer {node.id} on {node.host}:{node.port}")
        peer_log = os.path.join(log_dir, f"node_{node.id}.log")
        peer_proc = _start_process(
            [
                "scripts/run_p2p_node.py",
                "--config",
                config_path,
                "--node-id",
                str(node.id),
                "--port",
                str(node.port),
                "--bootstrap",
                bootstrap_addr,
                "--partition-dir",
                part_dir,
            ],
            log_file=peer_log,
        )
        processes.append(peer_proc)
        labels[peer_proc.pid] = f"Peer-{node.id}"
        print(f"    Peer-{node.id} PID: {peer_proc.pid}, Log: {peer_log}")

    # Wait for all processes to finish
    try:
        _wait_for_all(processes, labels)
    except KeyboardInterrupt:
        _terminate_all(processes)

    for proc in processes:
        label = labels.get(proc.pid, "Unknown")
        code = proc.returncode
        print(f"{label} (PID {proc.pid}) exited with code {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch federated learning from config")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    print(f"Mode: {config.mode}")
    print(f"Dataset: {config.dataset.name}, Clients: {config.num_clients}")
    print(f"Training: {config.training.rounds} rounds")
    print()

    # Pre-download and partition data
    part_dir = _pre_download_and_partition(config)
    print()

    if config.mode == "centralized":
        run_centralized(config, args.config, part_dir)
    elif config.mode == "decentralized":
        run_decentralized(config, args.config, part_dir)
    else:
        print(f"Unknown mode: {config.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
