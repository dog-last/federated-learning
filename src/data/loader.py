"""DataLoader factory."""

from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for a dataset.

    Args:
        dataset: Dataset object.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of worker processes.

    Returns:
        DataLoader: Configured data loader.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
