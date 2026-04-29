"""Model checkpoint save/load."""

from pathlib import Path

import torch

from src.core.types import Weights


def save_checkpoint(weights: Weights, path: str, round_id: int | None = None) -> None:
    """Save model weights to a checkpoint file.

    Args:
        weights: Model weights.
        path: Output file path.
        round_id: Optional round ID stored in the checkpoint.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": weights, "round_id": round_id}, path)


def load_checkpoint(path: str) -> tuple[Weights, int | None]:
    """Load model weights from a checkpoint file.

    Args:
        path: Checkpoint file path.

    Returns:
        Tuple[Weights, Optional[int]]: (weights, round_id).
    """
    data = torch.load(path, weights_only=False)
    return data["weights"], data.get("round_id")
