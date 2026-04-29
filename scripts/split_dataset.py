"""Split dataset for federated learning.

Usage:
    python scripts/split_dataset.py --dataset mnist --num-clients 3 --strategy iid
    python scripts/split_dataset.py --dataset cifar10 --num-clients 3 --strategy non_iid --alpha 0.5
"""

import argparse

import torch

from src.data.dataset import load_and_partition


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset for federated learning")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--strategy", choices=["iid", "non_iid"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    print(
        f"Loading {args.dataset} and partitioning ({args.strategy}) for {args.num_clients} clients..."
    )
    client_datasets, test_dataset = load_and_partition(
        name=args.dataset,
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        strategy=args.strategy,
        alpha=args.alpha,
    )

    for cid, ds in enumerate(client_datasets):
        path = f"data/partitioned/client_{cid + 1}.pt"
        torch.save(ds, path)
        print(f"Client {cid + 1}: {len(ds)} samples -> {path}")

    torch.save(test_dataset, "data/partitioned/test.pt")
    print(f"Test set: {len(test_dataset)} samples -> data/partitioned/test.pt")
    print("Done.")


if __name__ == "__main__":
    main()
