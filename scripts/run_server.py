"""Start the federated learning server.

Usage:
    python scripts/run_server.py --config config/centralized.yaml
"""

import argparse
import signal
import sys

import torch

from src.core.types import Config
from src.data.dataset import get_input_channels, get_num_classes
from src.data.loader import create_dataloader
from src.model.registry import get_model
from src.server.aggregator import create_aggregator
from src.server.federated_server import FederatedServer
from src.utils.logger import FedLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Start federated server")
    parser.add_argument("--config", default="config/centralized.yaml")
    parser.add_argument("--partition-dir", default="data/partitioned")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if config.mode != "centralized" or config.server is None:
        print("Error: config must have mode=centralized with a server section")
        sys.exit(1)

    logger = FedLogger(
        name="Server",
        level=config.logging.level,
        log_dir=config.logging.log_dir,
        console_output=config.logging.console_output,
        file_output=config.logging.file_output,
    )

    # Initialize global model
    input_channels = get_input_channels(config.dataset.name)
    num_classes = get_num_classes(config.dataset.name)
    model = get_model("simple_cnn", input_channels=input_channels, num_classes=num_classes)
    logger.info(f"Model created: {model.model_size} parameters")

    # Load pre-saved test dataset
    import os

    test_path = os.path.join(args.partition_dir, "test.pt")
    if os.path.exists(test_path):
        test_dataset = torch.load(test_path, weights_only=False)
    else:
        from src.data.dataset import load_dataset

        logger.info("Pre-partitioned test data not found, loading from source...")
        test_dataset = load_dataset(config.dataset.name, config.dataset.data_dir, train=False)
    test_dataloader = create_dataloader(
        test_dataset, batch_size=config.training.batch_size, shuffle=False
    )
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")

    # Initialize aggregator
    aggregator = create_aggregator(config.aggregator.name, fedprox_mu=config.aggregator.fedprox_mu)

    # Start server with early stopping and checkpoint config
    server = FederatedServer(
        aggregator=aggregator,
        logger=logger,
        timeouts=config.server.timeouts,
        early_stopping=config.training.early_stopping,
        checkpoint_dir=config.output.checkpoint_dir,
        save_checkpoint_every=config.output.save_checkpoint_every,
    )
    server._global_weights = model.get_weights()
    server.set_test_dataloader(test_dataloader)
    server.set_model(model)

    def shutdown(sig: int, frame: object) -> None:
        logger.info("Shutdown signal received")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    server.start(config.server.port)
    server.wait_for_clients(config.num_clients)
    server.run(config.training.rounds)
    server.stop()


if __name__ == "__main__":
    main()
