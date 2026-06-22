import subprocess
import sys
import json


def get_ports_from_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ports = []
    topo = config.get("topology", {})

    # 监控端口
    ports.append(int(config["monitoring"]["api_port"]))

    # Centralized / SplitFed 模式
    if "server" in topo:
        ports.append(int(topo["server"]["port"]))

    # Ring 模式
    for node in topo.get("nodes", []):
        ports.append(int(node["port"]))

    return ports


def kill_port(port):
    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            parts = line.strip().split()
            pid = parts[-1]
            if pid == "0":
                continue
            subprocess.run(["taskkill", "/PID", pid, "/F"],
                           capture_output=True)
            print(f"Killed PID {pid} on port {port}")
            return
    print(f"Port {port}: already free")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    print(f"Reading ports from {config_path}...")
    try:
        ports = get_ports_from_config(config_path)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    print(f"Cleaning up ports: {ports}")
    for port in ports:
        kill_port(port)
    print("Done. Safe to start now.")
