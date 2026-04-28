#!/usr/bin/env python3
"""Reusable TCP framing smoke test for the communicator.

This verifies that the magic number and length prefix framing still work.
"""

import json
import socket
import threading
import time
import unittest
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.communicator import TCPCommunicator


class TCPCommunicatorSmokeTest(unittest.TestCase):
    def test_p2p_framing(self):
        config_path = PROJECT_ROOT / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        host = config["topology"]["server"]["host"]
        port = int(config["topology"]["server"]["port"]) + 10
        communicator = TCPCommunicator(use_compression=False)
        received_data = {"value": None}

        def mock_server():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(1)
            conn, _ = srv.accept()
            try:
                received_data["value"] = communicator.recv_data(conn)
            finally:
                conn.close()
                srv.close()

        thread = threading.Thread(target=mock_server, daemon=True)
        thread.start()
        time.sleep(0.5)

        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.connect((host, port))
        test_payload = {"type": "smoke", "target": "success"}
        communicator.send_data(cli, test_payload)
        cli.close()
        thread.join(timeout=5)

        self.assertEqual(received_data["value"], test_payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
