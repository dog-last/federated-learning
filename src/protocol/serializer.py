"""Weight serializer using gzip + torch.save/load for compact transfer."""

import gzip
import io

import torch

from src.core.interfaces import ISerializer
from src.core.types import Weights


class TorchSerializer(ISerializer):
    """Serialize/deserialize model weights with gzip compression.

    Uses torch.save into a BytesIO, then gzip-compresses with
    compresslevel=1 (fastest) for minimal CPU overhead while
    achieving significant size reduction on float tensors.
    """

    def serialize_weights(self, weights: Weights) -> bytes:
        """Serialize model weights.

        Args:
            weights: Model weights dictionary.

        Returns:
            bytes: Gzip-compressed serialized byte stream.
        """
        buffer = io.BytesIO()
        torch.save(weights, buffer)
        return gzip.compress(buffer.getvalue(), compresslevel=1)

    def deserialize_weights(self, data: bytes) -> Weights:
        """Deserialize model weights.

        Args:
            data: Gzip-compressed byte stream.

        Returns:
            Weights: Model weights dictionary.
        """
        raw = gzip.decompress(data)
        buffer = io.BytesIO(raw)
        return torch.load(buffer, weights_only=True)

    def get_size(self, weights: Weights) -> int:
        """Get the serialized size of weights.

        Args:
            weights: Model weights.

        Returns:
            int: Size in bytes (compressed).
        """
        return len(self.serialize_weights(weights))
