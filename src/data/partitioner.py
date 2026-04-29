"""Data partitioner with IID and Non-IID (Dirichlet) strategies."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from src.core.interfaces import IPartitioner


class Partitioner(IPartitioner):
    """Partition a dataset among federated learning clients.

    Supports IID (uniform shuffle-split) and Non-IID (Dirichlet distribution).

    Attributes:
        _partitions: Cached partitioned index lists.
    """

    def __init__(self) -> None:
        self._partitions: list[list[int]] | None = None

    def partition(self, dataset: Any, num_clients: int, strategy: str = "iid") -> list[Subset]:
        """Partition a dataset among clients.

        Args:
            dataset: Original dataset (must support indexing and have .targets).
            num_clients: Number of clients.
            strategy: "iid" or "non_iid".

        Returns:
            List[Subset]: Partitioned datasets.
        """
        if strategy == "iid":
            self._partitions = self._partition_iid(len(dataset), num_clients)
        elif strategy == "non_iid":
            labels = self._get_labels(dataset)
            self._partitions = self._partition_non_iid(labels, num_clients, alpha=0.5)
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")

        return [Subset(dataset, indices) for indices in self._partitions]

    def get_client_data(self, client_id: int) -> list[int] | None:
        """Get the sample indices for a specific client.

        Args:
            client_id: Client ID (0-based).

        Returns:
            List of sample indices, or None if not partitioned yet.
        """
        if self._partitions is None or client_id >= len(self._partitions):
            return None
        return self._partitions[client_id]

    @staticmethod
    def _partition_iid(total: int, num_clients: int) -> list[list[int]]:
        """IID partition: shuffle and split evenly.

        Args:
            total: Total number of samples.
            num_clients: Number of clients.

        Returns:
            List[List[int]]: Index lists per client.
        """
        indices = np.arange(total)
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        return [split.tolist() for split in splits]

    @staticmethod
    def _partition_non_iid(
        labels: np.ndarray, num_clients: int, alpha: float = 0.5
    ) -> list[list[int]]:
        """Non-IID partition using Dirichlet distribution.

        Args:
            labels: Array of label values for all samples.
            num_clients: Number of clients.
            alpha: Dirichlet concentration parameter.

        Returns:
            List[List[int]]: Index lists per client.
        """
        num_classes = int(labels.max() + 1)
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            np.random.shuffle(class_indices)

            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_indices)).astype(int)

            # Distribute remainder to the last client
            proportions[-1] = len(class_indices) - proportions[:-1].sum()

            start = 0
            for cid, count in enumerate(proportions):
                client_indices[cid].extend(class_indices[start : start + count].tolist())
                start += count

        return client_indices

    @staticmethod
    def _get_labels(dataset: Any) -> np.ndarray:
        """Extract label array from a dataset.

        Args:
            dataset: Dataset with .targets attribute or indexable.

        Returns:
            np.ndarray: Label array.
        """
        if hasattr(dataset, "targets"):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                return targets.numpy()
            if isinstance(targets, list):
                return np.array(targets)
            return np.array(targets)
        # Fallback: iterate
        return np.array([dataset[i][1] for i in range(len(dataset))])
