"""Download MNIST or CIFAR-10 dataset.

Usage:
    python scripts/download_cifar10.py --dataset mnist
    python scripts/download_cifar10.py --dataset cifar10
"""

import argparse

from src.data.dataset import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    print(f"Downloading {args.dataset}...")
    ds = load_dataset(args.dataset, args.data_dir, train=True)
    print(f"Training set size: {len(ds)}")

    ds_test = load_dataset(args.dataset, args.data_dir, train=False)
    print(f"Test set size: {len(ds_test)}")
    print("Done.")


if __name__ == "__main__":
    main()
