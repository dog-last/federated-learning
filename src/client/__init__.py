"""Client module: base, federated_client, trainer, evaluator."""

from src.client.base import BaseClient
from src.client.evaluator import Evaluator
from src.client.federated_client import FederatedClient
from src.client.trainer import LocalTrainer

__all__ = ["BaseClient", "FederatedClient", "LocalTrainer", "Evaluator"]
