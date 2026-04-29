"""Start a P2P ring node for decentralized federated learning.

Usage:
    python scripts/run_p2p_node.py --config config/decentralized.yaml --node-id 1 --port 9001
    python scripts/run_p2p_node.py --config config/decentralized.yaml --node-id 2 --port 9002 --bootstrap 127.0.0.1:9001
"""

import argparse
import os
import sys
import time

import torch

from src.core.types import Config
from src.data.dataset import get_input_channels, get_num_classes, load_and_partition, load_dataset
from src.data.loader import create_dataloader
from src.model.registry import get_model
from src.p2p.ring_node import RingNode
from src.utils.logger import FedLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Start P2P ring node")
    parser.add_argument("--config", default="config/decentralized.yaml")
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bootstrap", default=None, help="Bootstrap node host:port")
    parser.add_argument("--partition-dir", default="data/partitioned")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if config.mode != "decentralized":
        print("Error: config must have mode=decentralized")
        sys.exit(1)

    logger = FedLogger(
        name=f"Node-{args.node_id}",
        level=config.logging.level,
        log_dir=config.logging.log_dir,
        console_output=config.logging.console_output,
        file_output=config.logging.file_output,
    )

    # Load pre-saved partition
    partition_path = os.path.join(args.partition_dir, f"client_{args.node_id}.pt")
    t0 = time.time()
    if os.path.exists(partition_path):
        client_data = torch.load(partition_path, weights_only=False)
    else:
        # Fallback: load and partition from source
        logger.info("Pre-partitioned data not found, loading from source...")
        client_datasets, _ = load_and_partition(
            name=config.dataset.name,
            data_dir=config.dataset.data_dir,
            num_clients=config.dataset.num_clients,
            strategy=config.dataset.partition_strategy,
            alpha=config.dataset.alpha,
        )
        client_data = client_datasets[(args.node_id - 1) % len(client_datasets)]

    dataloader = create_dataloader(client_data, batch_size=config.training.batch_size)
    logger.info(f"Data loaded: {len(client_data)} samples ({time.time() - t0:.1f}s)")

    # Initialize model
    input_channels = get_input_channels(config.dataset.name)
    num_classes = get_num_classes(config.dataset.name)
    model = get_model("simple_cnn", input_channels=input_channels, num_classes=num_classes)

    # Load test dataset for evaluation
    test_path = os.path.join(args.partition_dir, "test.pt")
    if os.path.exists(test_path):
        test_dataset = torch.load(test_path, weights_only=False)
    else:
        logger.info("Pre-partitioned test data not found, loading from source...")
        test_dataset = load_dataset(config.dataset.name, config.dataset.data_dir, train=False)
    test_dataloader = create_dataloader(
        test_dataset, batch_size=config.training.batch_size, shuffle=False
    )
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")

    # Start node with early stopping and checkpoint config
    node = RingNode(
        node_id=args.node_id,
        model=model,
        dataloader=dataloader,
        logger=logger,
        early_stopping=config.training.early_stopping,
        checkpoint_dir=config.output.checkpoint_dir,
        save_checkpoint_every=config.output.save_checkpoint_every,
    )
    node.start(args.port)
    node.set_test_dataloader(test_dataloader)

    # Join ring via bootstrap or config
    bootstrap = args.bootstrap
    if bootstrap is None and config.peers and config.peers.nodes:
        for peer in config.peers.nodes:
            if peer.id != args.node_id:
                bootstrap = f"{peer.host}:{peer.port}"
                break

    if bootstrap:
        host, port = bootstrap.split(":")
        node.join_ring((host, int(port)))

    node.run(config.training.rounds)
    node.leave_ring()
    node.stop()


if __name__ == "__main__":
    main()
