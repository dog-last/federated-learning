#!/usr/bin/env python3
"""
Prepare federated MNIST dataset for split learning / centralized training.
Saves data in format compatible with core/client.py and core/server.py.
"""
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path


def prepare_mnist_federated(root_dir="./data", num_clients=3, seed=42):
    """
    Prepare MNIST dataset split for federated learning.
    
    Args:
        root_dir: Root data directory (will create splits/ subdirectory)
        num_clients: Number of federated clients
        seed: Random seed for reproducibility
    
    Returns:
        dict with keys: client_datasets, server_test_dataset, statistics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    splits_dir = os.path.join(root_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    # MNIST specific normalization (single channel)
    # Mean and std computed for MNIST training set
    mnist_mean = 0.1307
    mnist_std = 0.3081
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,))
    ])
    
    # 1. Load full MNIST training and test datasets
    print("[MNIST] Loading training data...")
    full_train_dataset = datasets.MNIST(
        root=root_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    print("[MNIST] Loading test data...")
    full_test_dataset = datasets.MNIST(
        root=root_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # 2. Create indices grouped by class for non-IID distribution
    print("[MNIST] Organizing data by class...")
    train_images = []
    train_labels = []
    for img, label in full_train_dataset:
        train_images.append(img)
        train_labels.append(label)
    
    train_images = torch.stack(train_images)  # [N, 1, 28, 28]
    train_labels = torch.tensor(train_labels)  # [N]
    
    test_images = []
    test_labels = []
    for img, label in full_test_dataset:
        test_images.append(img)
        test_labels.append(label)
    
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)
    
    print(f"[MNIST] Train images shape: {train_images.shape}, Test images shape: {test_images.shape}")
    
    # 3. Non-IID split: assign class ranges to clients
    class_indices = [[] for _ in range(10)]
    for idx in range(len(train_labels)):
        label = train_labels[idx].item()
        class_indices[label].append(idx)
    
    # Define class ranges for each client (non-IID)
    # Distribute 10 classes across num_clients as evenly as possible
    classes_per_client = max(1, 10 // num_clients)
    remainder = 10 % num_clients
    client_class_ranges = []
    start = 0
    for i in range(num_clients):
        count = classes_per_client + (1 if i < remainder else 0)
        client_class_ranges.append(list(range(start, start + count)))
        start += count
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx, indices in enumerate(class_indices):
        # Find which client(s) this class belongs to
        primary_client = None
        for client_id, class_list in enumerate(client_class_ranges):
            if class_idx in class_list:
                primary_client = client_id
                break
        
        if primary_client is None:
            primary_client = class_idx % num_clients
        
        # Shuffle and split indices: 80% to primary, distribute remainder
        np.random.shuffle(indices)
        split1 = int(len(indices) * 0.8)
        split2 = int(len(indices) * 0.9)
        
        client_indices[primary_client].extend(indices[:split1])
        client_indices[(primary_client + 1) % num_clients].extend(indices[split1:split2])
        client_indices[(primary_client + 2) % num_clients].extend(indices[split2:])
    
    # 4. Further split each client's training data into train/val
    client_datasets = []
    stats = {"clients": []}
    
    for client_id in range(num_clients):
        indices = np.array(client_indices[client_id])
        np.random.shuffle(indices)
        
        # 80% train, 10% val, 10% test
        train_split = int(len(indices) * 0.8)
        val_split = int(len(indices) * 0.9)
        
        train_idx = indices[:train_split]
        val_idx = indices[train_split:val_split]
        test_idx = indices[val_split:]
        
        # Extract data
        client_train_images = train_images[train_idx]
        client_train_labels = train_labels[train_idx]
        
        client_val_images = train_images[val_idx]
        client_val_labels = train_labels[val_idx]
        
        client_test_images = train_images[test_idx]
        client_test_labels = train_labels[test_idx]
        
        # Save as dict
        client_data = {
            "train_images": client_train_images,
            "train_labels": client_train_labels,
            "val_images": client_val_images,
            "val_labels": client_val_labels,
            "test_images": client_test_images,
            "test_labels": client_test_labels,
        }
        
        client_path = os.path.join(splits_dir, f"client_{client_id + 1}_data.pt")
        torch.save(client_data, client_path)
        
        print(f"[CLIENT {client_id + 1}] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} -> {client_path}")
        
        client_datasets.append(client_data)
        stats["clients"].append({
            "client_id": f"client_{client_id + 1}",
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
            "unique_classes": len(set(client_train_labels.tolist())),
        })
    
    # 5. Save server test data (use global test set)
    server_test_data = {
        "images": test_images,
        "labels": test_labels,
    }
    
    server_test_path = os.path.join(splits_dir, "server_test_data.pt")
    torch.save(server_test_data, server_test_path)
    
    print(f"[SERVER] test={len(test_labels)} -> {server_test_path}")
    
    stats["server_test_samples"] = len(test_labels)
    stats["total_train_samples"] = sum(s["train_samples"] for s in stats["clients"])
    
    return {
        "client_datasets": client_datasets,
        "server_test_dataset": server_test_data,
        "statistics": stats
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare MNIST for federated learning")
    parser.add_argument("--data-dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of federated clients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    result = prepare_mnist_federated(
        root_dir=args.data_dir,
        num_clients=args.num_clients,
        seed=args.seed
    )
    
    print("\n=== Dataset Preparation Summary ===")
    for stat_key, stat_val in result["statistics"].items():
        print(f"{stat_key}: {stat_val}")
