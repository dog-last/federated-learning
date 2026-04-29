"""Unit tests for prepare_mnist script."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import numpy as np

# Import the functions from prepare_mnist
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from prepare_mnist import prepare_mnist_federated


class TestPrepareMNISTFederated:
    """Tests for prepare_mnist_federated function."""

    def test_prepare_mnist_creates_files(self):
        """Test that prepare_mnist_federated creates the expected files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('torchvision.datasets.MNIST') as mock_mnist:
                # Create mock dataset with proper iteration behavior
                num_train = 100
                num_test = 20
                
                # Create actual tensors for data and targets
                train_data = torch.randint(0, 255, (num_train, 28, 28), dtype=torch.uint8)
                train_targets = torch.randint(0, 10, (num_train,))
                test_data = torch.randint(0, 255, (num_test, 28, 28), dtype=torch.uint8)
                test_targets = torch.randint(0, 10, (num_test,))
                
                mock_train = MagicMock()
                mock_train.data = train_data
                mock_train.targets = train_targets
                # Make the mock iterable
                mock_train.__iter__ = lambda self: iter([
                    (train_data[i].unsqueeze(0).float() / 255.0, train_targets[i].item()) 
                    for i in range(num_train)
                ])
                
                mock_test = MagicMock()
                mock_test.data = test_data
                mock_test.targets = test_targets
                mock_test.__iter__ = lambda self: iter([
                    (test_data[i].unsqueeze(0).float() / 255.0, test_targets[i].item()) 
                    for i in range(num_test)
                ])

                mock_mnist.side_effect = [mock_train, mock_test]

                result = prepare_mnist_federated(
                    root_dir=tmp_dir,
                    num_clients=3,
                    seed=42
                )

                # Check that files were created
                splits_dir = Path(tmp_dir) / "splits"
                assert (splits_dir / "client_1_data.pt").exists()
                assert (splits_dir / "client_2_data.pt").exists()
                assert (splits_dir / "client_3_data.pt").exists()
                assert (splits_dir / "server_test_data.pt").exists()

                # Check result structure
                assert "client_datasets" in result
                assert "server_test_dataset" in result
                assert "statistics" in result
                assert len(result["client_datasets"]) == 3

    def test_prepare_mnist_statistics(self):
        """Test that statistics are correctly computed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('torchvision.datasets.MNIST') as mock_mnist:
                num_train = 100
                num_test = 20
                
                # Ensure we have samples from all 10 classes
                train_data = torch.randint(0, 255, (num_train, 28, 28), dtype=torch.uint8)
                train_targets = torch.tensor([i % 10 for i in range(num_train)])
                test_data = torch.randint(0, 255, (num_test, 28, 28), dtype=torch.uint8)
                test_targets = torch.randint(0, 10, (num_test,))
                
                mock_train = MagicMock()
                mock_train.data = train_data
                mock_train.targets = train_targets
                mock_train.__iter__ = lambda self: iter([
                    (train_data[i].unsqueeze(0).float() / 255.0, train_targets[i].item()) 
                    for i in range(num_train)
                ])
                
                mock_test = MagicMock()
                mock_test.data = test_data
                mock_test.targets = test_targets
                mock_test.__iter__ = lambda self: iter([
                    (test_data[i].unsqueeze(0).float() / 255.0, test_targets[i].item()) 
                    for i in range(num_test)
                ])

                mock_mnist.side_effect = [mock_train, mock_test]

                result = prepare_mnist_federated(root_dir=tmp_dir, num_clients=2, seed=42)

                stats = result["statistics"]
                assert "clients" in stats
                assert "server_test_samples" in stats
                assert "total_train_samples" in stats

                for client_stat in stats["clients"]:
                    assert "client_id" in client_stat
                    assert "train_samples" in client_stat
                    assert "val_samples" in client_stat
                    assert "test_samples" in client_stat
                    assert "unique_classes" in client_stat


class TestPrepareMNISTMain:
    """Tests for __main__ block."""

    def test_main_execution(self):
        """Test the main execution block."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('torchvision.datasets.MNIST') as mock_mnist:
                num_train = 100
                num_test = 20
                
                # Ensure we have samples from all 10 classes
                train_data = torch.randint(0, 255, (num_train, 28, 28), dtype=torch.uint8)
                train_targets = torch.tensor([i % 10 for i in range(num_train)])
                test_data = torch.randint(0, 255, (num_test, 28, 28), dtype=torch.uint8)
                test_targets = torch.randint(0, 10, (num_test,))
                
                mock_train = MagicMock()
                mock_train.data = train_data
                mock_train.targets = train_targets
                mock_train.__iter__ = lambda self: iter([
                    (train_data[i].unsqueeze(0).float() / 255.0, train_targets[i].item()) 
                    for i in range(num_train)
                ])
                
                mock_test = MagicMock()
                mock_test.data = test_data
                mock_test.targets = test_targets
                mock_test.__iter__ = lambda self: iter([
                    (test_data[i].unsqueeze(0).float() / 255.0, test_targets[i].item()) 
                    for i in range(num_test)
                ])

                mock_mnist.side_effect = [mock_train, mock_test]

                # Import and execute
                import scripts.prepare_mnist as pm

                result = pm.prepare_mnist_federated(
                    root_dir=tmp_dir,
                    num_clients=2,
                    seed=123
                )

                assert result is not None
                assert len(result["client_datasets"]) == 2
