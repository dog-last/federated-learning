"""Start a federated learning client.

Usage:
    python scripts/run_client.py --config config/centralized.yaml --client-id 1
"""

import argparse
import os
import sys
import time

import torch

from src.client.federated_client import FederatedClient
from src.core.types import Config
from src.data.dataset import get_input_channels, get_num_classes
from src.data.loader import create_dataloader
from src.model.registry import get_model
from src.utils.logger import FedLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Start federated client")
    parser.add_argument("--config", default="config/centralized.yaml")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--partition-dir", default="data/partitioned")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if config.mode != "centralized" or config.server is None:
        print("Error: config must have mode=centralized with a server section")
        sys.exit(1)

    logger = FedLogger(
        name=f"Client-{args.client_id}",
        level=config.logging.level,
        log_dir=config.logging.log_dir,
        console_output=config.logging.console_output,
        file_output=config.logging.file_output,
    )

    # Load pre-saved partition
    partition_path = os.path.join(args.partition_dir, f"client_{args.client_id}.pt")
    t0 = time.time()
    if os.path.exists(partition_path):
        client_data = torch.load(partition_path, weights_only=False)
    else:
        # Fallback: load and partition from source
        logger.info("Pre-partitioned data not found, loading from source...")
        from src.data.dataset import load_and_partition

        client_datasets, _ = load_and_partition(
            name=config.dataset.name,
            data_dir=config.dataset.data_dir,
            num_clients=config.dataset.num_clients,
            strategy=config.dataset.partition_strategy,
            alpha=config.dataset.alpha,
        )
        client_data = client_datasets[args.client_id - 1]

    dataloader = create_dataloader(client_data, batch_size=config.training.batch_size)
    logger.info(f"Data loaded: {len(client_data)} samples ({time.time() - t0:.1f}s)")

    # Initialize model
    input_channels = get_input_channels(config.dataset.name)
    num_classes = get_num_classes(config.dataset.name)
    model = get_model("simple_cnn", input_channels=input_channels, num_classes=num_classes)

    # Determine server address
    server_host = config.server.address
    server_port = config.server.port
    if config.clients and config.clients.server_address:
        parts = config.clients.server_address.split(":")
        server_host = parts[0]
        server_port = int(parts[1]) if len(parts) > 1 else server_port

    # Start client
    client = FederatedClient(model=model, dataloader=dataloader, logger=logger)
    client.connect(server_host, server_port)
    client.register()
    logger.info("Waiting for global model from server...")
    client.run(
        num_rounds=config.training.rounds,
        epochs=config.training.epochs_per_round,
        lr=config.training.learning_rate,
    )
    client.disconnect()


if __name__ == "__main__":
    main()
