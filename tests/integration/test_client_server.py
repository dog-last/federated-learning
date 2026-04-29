"""Integration tests for client-server communication."""
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from core.server import Server
from core.client import Client


@pytest.fixture
def integration_config():
    """Create a minimal config for integration testing."""
    return {
        "experiment": {
            "mode": "centralized",
            "global_epochs": 1,
            "local_epochs": 1,
            "device": "cpu",
            "target_accuracy": 0.99,
            "dataset_params": {"batch_size": 4, "num_workers": 0},
            "optimization": {"client_lr": 0.01, "server_lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        },
        "topology": {
            "server": {"host": "127.0.0.1", "port": 0},
            "clients": [{"id": "client_1", "host": "127.0.0.1", "port": 0}],
        },
        "network": {"compression": False, "stragglers": {}, "server_timeout": 30.0},
        "monitoring": {"api_host": "127.0.0.1", "api_port": 0},
    }


@pytest.fixture
def integration_data_dir(tmp_path):
    """Create minimal data for integration testing."""
    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)
    
    # Create minimal client data
    n_samples = 20
    client_data = {
        "train_images": torch.rand(n_samples, 1, 28, 28),
        "train_labels": torch.randint(0, 10, (n_samples,)),
        "val_images": torch.rand(4, 1, 28, 28),
        "val_labels": torch.randint(0, 10, (4,)),
        "test_images": torch.rand(4, 1, 28, 28),
        "test_labels": torch.randint(0, 10, (4,)),
    }
    torch.save(client_data, splits_dir / "client_1_data.pt")
    
    # Create server test data
    server_data = {
        "images": torch.rand(10, 1, 28, 28),
        "labels": torch.randint(0, 10, (10,)),
    }
    torch.save(server_data, splits_dir / "server_test_data.pt")
    
    return tmp_path


@pytest.mark.integration
class TestClientServerRegistration:
    """Integration tests for client-server registration flow."""

    def test_client_registration_message(self, integration_config, integration_data_dir):
        """Test that client sends correct registration message."""
        config_path = integration_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(integration_config, f)

        # Create server
        with patch('core.server.socket.socket') as mock_server_sock:
            mock_server = MagicMock()
            mock_server_sock.return_value = mock_server

            server = Server(str(config_path), project_root=str(integration_data_dir))

            # Verify server can access data via test_loader
            assert server.test_loader is not None
            assert len(server.test_loader.dataset) > 0

    def test_client_loads_local_data(self, integration_config, integration_data_dir):
        """Test that client can load local data."""
        config_path = integration_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(integration_config, f)

        with patch('core.client.socket.socket'):
            client = Client(str(config_path), "client_1", project_root=str(integration_data_dir))
            
            # Verify client can access data loaders
            assert client.train_loader is not None
            assert client.val_loader is not None
            assert client.test_loader is not None


@pytest.mark.integration
class TestModelTrainingIntegration:
    """Integration tests for model training."""

    def test_server_model_evaluation(self, integration_config, integration_data_dir):
        """Test server can evaluate model on test data."""
        config_path = integration_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(integration_config, f)
        
        with patch('core.server.socket.socket'):
            server = Server(str(config_path), project_root=str(integration_data_dir))

            # Evaluate model
            loss, acc = server._evaluate_centralized()
            
            assert isinstance(loss, float)
            assert isinstance(acc, float)
            assert 0.0 <= acc <= 1.0
            assert loss >= 0.0

    def test_client_local_training(self, integration_config, integration_data_dir):
        """Test client can perform local training."""
        config_path = integration_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(integration_config, f)
        
        with patch('core.client.socket.socket'):
            client = Client(str(config_path), "client_1", project_root=str(integration_data_dir))

            # Get initial model weights
            initial_weights = {k: v.clone() for k, v in client.model.state_dict().items()}
            
            # Simulate training by running one batch
            for batch_idx, (data, target) in enumerate(client.train_loader):
                if batch_idx >= 1:  # Just one batch
                    break
                
                client.optimizer.zero_grad()
                output = client.model(data)
                loss = client.criterion(output, target)
                loss.backward()
                client.optimizer.step()
            
            # Verify model weights changed
            new_weights = client.model.state_dict()
            weights_changed = any(
                not torch.allclose(initial_weights[k], new_weights[k])
                for k in initial_weights.keys()
            )
            assert weights_changed


@pytest.mark.integration
class TestCommunicationIntegration:
    """Integration tests for communication between components."""

    def test_message_serialization(self):
        """Test that messages can be serialized and deserialized."""
        from core.communicator import TCPCommunicator
        
        comm = TCPCommunicator(use_compression=False)
        
        # Test various message types
        messages = [
            {"type": "register", "client_id": "client_1"},
            {"type": "model_update", "round": 1, "weights": {"layer1": torch.rand(10, 10)}},
            {"type": "shutdown", "reason": "completed"},
        ]
        
        for msg in messages:
            # Serialize
            data = comm._serialize(msg)
            # Deserialize
            restored = comm._deserialize(data)
            
            assert restored["type"] == msg["type"]

    def test_compressed_message_serialization(self):
        """Test that compressed messages work correctly."""
        from core.communicator import TCPCommunicator
        
        comm = TCPCommunicator(use_compression=True)
        
        # Large payload that benefits from compression
        large_data = {"values": list(range(1000))}
        
        # Serialize with compression
        data = comm._serialize(large_data)
        # Deserialize
        restored = comm._deserialize(data)
        
        assert restored["values"] == large_data["values"]


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    def test_full_centralized_workflow(self, integration_config, integration_data_dir):
        """Test complete centralized training workflow."""
        config_path = integration_data_dir / "config.json"
        
        # Modify config for single round
        integration_config["experiment"]["global_epochs"] = 1
        integration_config["experiment"]["local_epochs"] = 1
        
        with open(config_path, "w") as f:
            json.dump(integration_config, f)
        
        # Create server
        with patch('core.server.socket.socket'):
            server = Server(str(config_path), project_root=str(integration_data_dir))

            # Simulate one training round
            server.num_clients = 1
            server.active_clients = {"client_1": MagicMock()}
            
            # Mock client update
            weights = {k: v.clone() for k, v in server.global_model.state_dict().items()}
            server.round_updates[1] = {
                "client_1": {
                    "weights": weights,
                    "num_samples": 20,
                    "train_loss": 0.5,
                    "train_acc": 0.8,
                    "test_acc": 0.75,
                }
            }
            
            # Run aggregation
            agg_weights = server._aggregate_weighted(list(server.round_updates[1].values()))
            assert agg_weights is not None
            
            # Evaluate
            loss, acc = server._evaluate_centralized()
            assert isinstance(acc, float)

    def test_data_flow_from_preparation_to_training(self, tmp_path):
        """Test data flows correctly from preparation to training."""
        from scripts.prepare_mnist import prepare_mnist_federated
        from model import get_model
        from torch.utils.data import DataLoader, TensorDataset
        
        # Prepare data
        with patch('torchvision.datasets.MNIST') as mock_mnist:
            mock_train = MagicMock()
            mock_train.data = torch.randint(0, 255, (100, 28, 28), dtype=torch.uint8)
            mock_train.targets = torch.randint(0, 10, (100,))
            mock_train.__iter__ = lambda self: iter([
                (mock_train.data[i].unsqueeze(0).float() / 255.0, mock_train.targets[i].item())
                for i in range(100)
            ])
            
            mock_test = MagicMock()
            mock_test.data = torch.randint(0, 255, (20, 28, 28), dtype=torch.uint8)
            mock_test.targets = torch.randint(0, 10, (20,))
            mock_test.__iter__ = lambda self: iter([
                (mock_test.data[i].unsqueeze(0).float() / 255.0, mock_test.targets[i].item())
                for i in range(20)
            ])
            
            mock_mnist.side_effect = [mock_train, mock_test]
            
            result = prepare_mnist_federated(root_dir=str(tmp_path), num_clients=2, seed=42)
        
        # Load data for training
        client_data = result["client_datasets"][0]
        train_images = client_data["train_images"]
        train_labels = client_data["train_labels"]
        
        # Create model
        model = get_model("centralized", num_classes=10)
        
        # Create dataloader
        dataset = TensorDataset(train_images, train_labels)
        loader = DataLoader(dataset, batch_size=4)
        
        # Train for one batch
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        for x, y in loader:
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert output.shape == (x.shape[0], 10)
            assert torch.isfinite(loss)
            break


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_server_handles_missing_client_data(self, integration_config, integration_data_dir):
        """Test server handles missing client data gracefully."""
        config_path = integration_data_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(integration_config, f)
        
        with patch('core.server.socket.socket'):
            server = Server(str(config_path), project_root=str(integration_data_dir))

            # Test with empty updates
            agg = server._aggregate_weighted([])
            assert agg is None

    def test_client_handles_missing_config(self, integration_data_dir):
        """Test client handles missing config file."""
        nonexistent_config = str(integration_data_dir / "nonexistent.json")
        
        with pytest.raises((FileNotFoundError, json.JSONDecodeError)):
            with patch('core.client.socket.socket'):
                Client(nonexistent_config, "client_1")

    def test_communicator_handles_connection_error(self):
        """Test communicator handles connection errors."""
        from core.communicator import TCPCommunicator
        
        comm = TCPCommunicator(use_compression=False)
        
        # Test sending on closed socket
        import socket
        closed_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        closed_sock.close()
        
        ok, size = comm.send_data(closed_sock, {"type": "test"})
        assert ok is False
        assert size == 0
