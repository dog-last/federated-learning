"""Federated aggregation algorithms (FedAvg, FedProx)."""

from src.core.exceptions import AggregationError
from src.core.interfaces import IAggregator
from src.core.types import Weights


class FedAvg(IAggregator):
    """Federated Averaging aggregator.

    Aggregation formula: w_{t+1} = sum(n_k / n) * w_k^t
    """

    @property
    def name(self) -> str:
        """Aggregator name.

        Returns:
            str: Name of the aggregator.
        """
        return "fedavg"

    def aggregate(
        self, weights_list: list[Weights], client_sizes: list[int] | None = None
    ) -> Weights:
        """Aggregate multiple model weights using weighted averaging.

        Args:
            weights_list: List of model weights.
            client_sizes: Number of samples per client (for weighted averaging).

        Returns:
            Weights: Aggregated weights.

        Raises:
            AggregationError: If weights_list is empty.
        """
        if not weights_list:
            raise AggregationError("Cannot aggregate empty weights list")

        if client_sizes is None:
            # Equal weighting
            return self._equal_average(weights_list)

        total = sum(client_sizes)
        result: Weights = {}
        for key in weights_list[0]:
            tensors = [w[key].float() for w in weights_list]
            weighted_sum = sum(size * t for size, t in zip(client_sizes, tensors, strict=False))
            result[key] = weighted_sum / total
        return result

    @staticmethod
    def _equal_average(weights_list: list[Weights]) -> Weights:
        """Simple equal-weight average.

        Args:
            weights_list: List of model weights.

        Returns:
            Weights: Averaged weights.
        """
        n = len(weights_list)
        result: Weights = {}
        for key in weights_list[0]:
            result[key] = sum(w[key].float() for w in weights_list) / n
        return result


class FedProx(IAggregator):
    """FedProx aggregator (FedAvg with proximal term support).

    The proximal term is applied during local training, not during
    aggregation. The aggregation itself is identical to FedAvg.

    Attributes:
        mu: Proximal term coefficient.
    """

    def __init__(self, mu: float = 0.01) -> None:
        self.mu = mu

    @property
    def name(self) -> str:
        """Aggregator name.

        Returns:
            str: Name of the aggregator.
        """
        return "fedprox"

    def aggregate(
        self, weights_list: list[Weights], client_sizes: list[int] | None = None
    ) -> Weights:
        """Aggregate using FedAvg (identical for FedProx).

        Args:
            weights_list: List of model weights.
            client_sizes: Number of samples per client.

        Returns:
            Weights: Aggregated weights.
        """
        return FedAvg().aggregate(weights_list, client_sizes)


def create_aggregator(name: str, **kwargs: object) -> IAggregator:
    """Create an aggregator by name.

    Args:
        name: Aggregator name ("fedavg" or "fedprox").
        **kwargs: Additional arguments.

    Returns:
        IAggregator: Aggregator instance.

    Raises:
        ValueError: If the name is unknown.
    """
    if name == "fedavg":
        return FedAvg()
    elif name == "fedprox":
        return FedProx(mu=float(kwargs.get("fedprox_mu", 0.01)))
    else:
        raise ValueError(f"Unknown aggregator: {name}")
