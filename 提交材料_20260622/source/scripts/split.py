#!/usr/bin/env python3
"""Compatibility wrapper for MNIST data preparation.

Deprecated: use scripts/prepare_mnist.py directly.
"""

from scripts.prepare_mnist import prepare_mnist_federated


def split_mnist_for_federated(root="./data", num_clients=3, seed=42):
    return prepare_mnist_federated(root_dir=root, num_clients=num_clients, seed=seed)


if __name__ == "__main__":
    split_mnist_for_federated(root="./data", num_clients=3, seed=42)
