import subprocess
import sys

PORTS = [9000, 8000, 8001, 8002, 8003, 8101, 8102, 8103]

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
    print("Cleaning up ports...")
    for port in PORTS:
        kill_port(port)
    print("Done. Safe to start now.")