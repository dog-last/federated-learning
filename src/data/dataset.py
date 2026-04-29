"""Dataset factory: load MNIST or CIFAR-10 with appropriate transforms."""

from typing import Any

from torchvision import datasets, transforms

from src.data.partitioner import Partitioner


def get_transforms(name: str) -> transforms.Compose:
    """Get standard transforms for a dataset.

    Args:
        name: Dataset name ("mnist" or "cifar10").

    Returns:
        transforms.Compose: Composed transforms.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if name == "mnist":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    elif name == "cifar10":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: mnist, cifar10")


def load_dataset(name: str, data_dir: str = "./data", train: bool = True) -> Any:
    """Load a full dataset.

    Args:
        name: Dataset name ("mnist" or "cifar10").
        data_dir: Root directory for data storage.
        train: Whether to load the training set.

    Returns:
        Dataset object.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    transform = get_transforms(name)
    if name == "mnist":
        return datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    elif name == "cifar10":
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def load_and_partition(
    name: str,
    data_dir: str = "./data",
    num_clients: int = 3,
    strategy: str = "iid",
    alpha: float = 0.5,
) -> tuple[list, Any]:
    """Load a dataset and partition it for federated learning.

    Args:
        name: Dataset name ("mnist" or "cifar10").
        data_dir: Root directory for data storage.
        num_clients: Number of clients.
        strategy: Partitioning strategy ("iid" or "non_iid").
        alpha: Dirichlet concentration for non_iid.

    Returns:
        Tuple[list, Dataset]: (list of partitioned Subsets, test dataset).
    """
    train_dataset = load_dataset(name, data_dir, train=True)
    test_dataset = load_dataset(name, data_dir, train=False)

    partitioner = Partitioner()
    client_datasets = partitioner.partition(train_dataset, num_clients, strategy)

    return client_datasets, test_dataset


def get_input_channels(name: str) -> int:
    """Get the number of input channels for a dataset.

    Args:
        name: Dataset name.

    Returns:
        int: Number of input channels.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if name == "mnist":
        return 1
    elif name == "cifar10":
        return 3
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_num_classes(name: str) -> int:
    """Get the number of classes for a dataset.

    Args:
        name: Dataset name.

    Returns:
        int: Number of classes.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if name in ("mnist", "cifar10"):
        return 10
    raise ValueError(f"Unknown dataset: {name}")
