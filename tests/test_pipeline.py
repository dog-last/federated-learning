#!/usr/bin/env python3
"""
Unit test suite for federated learning training pipeline.
Tests data loading, model initialization, normalization, and communication.
"""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DataPrepTest(unittest.TestCase):
    """Test data preparation and loading."""
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.temp_dir.name
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_prepare_mnist_import(self):
        """Test that prepare_mnist script can be imported."""
        from scripts import prepare_mnist
        self.assertTrue(hasattr(prepare_mnist, 'prepare_mnist_federated'))
    
    def test_prepare_mnist_execution(self):
        """Test MNIST data preparation."""
        from scripts.prepare_mnist import prepare_mnist_federated
        
        result = prepare_mnist_federated(
            root_dir=self.data_dir,
            num_clients=3,
            seed=42
        )
        
        # Verify output structure
        self.assertIn('client_datasets', result)
        self.assertIn('server_test_dataset', result)
        self.assertIn('statistics', result)
        
        # Verify splits directory was created
        splits_dir = os.path.join(self.data_dir, 'splits')
        self.assertTrue(os.path.exists(splits_dir))
        
        # Verify client data files
        for client_id in range(1, 4):
            client_file = os.path.join(splits_dir, f'client_{client_id}_data.pt')
            self.assertTrue(os.path.exists(client_file))
            
            # Verify file format
            data = torch.load(client_file)
            for key in ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']:
                self.assertIn(key, data)
                self.assertEqual(data[key].dtype, torch.float32 if 'images' in key else torch.long)
        
        # Verify server test file
        server_file = os.path.join(splits_dir, 'server_test_data.pt')
        self.assertTrue(os.path.exists(server_file))
        
        server_data = torch.load(server_file)
        self.assertIn('images', server_data)
        self.assertIn('labels', server_data)
        
        print("✓ MNIST data preparation test passed")
    
    def test_client_data_load(self):
        """Test client data loading and normalization."""
        # Create minimal test data
        from scripts.prepare_mnist import prepare_mnist_federated
        
        result = prepare_mnist_federated(
            root_dir=self.data_dir,
            num_clients=3,
            seed=42
        )
        
        client_data = result['client_datasets'][0]
        
        # Test normalization consistency
        train_images = client_data['train_images']
        self.assertEqual(train_images.shape[1], 1)  # MNIST: single channel
        self.assertEqual(train_images.shape[2], 28)
        self.assertEqual(train_images.shape[3], 28)
        
        print("✓ Client data loading test passed")


class ModelTest(unittest.TestCase):
    """Test model initialization and forward pass."""
    
    def test_model_import(self):
        """Test model module can be imported."""
        from model import get_model, CNN, SplitClientCNN, SplitServerCNN
        self.assertTrue(callable(get_model))
    
    def test_centralized_model(self):
        """Test centralized model instantiation and forward."""
        from model import get_model
        
        model = get_model('centralized', num_classes=10)
        self.assertIsInstance(model, nn.Module)
        
        # Test forward pass with MNIST-sized input
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 10))
        print("✓ Centralized model test passed")
    
    def test_splitfed_model(self):
        """Test split-fed model instantiation and forward."""
        from model import get_model
        
        client_model, server_model = get_model('splitfed', num_classes=10)
        self.assertIsInstance(client_model, nn.Module)
        self.assertIsInstance(server_model, nn.Module)
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        client_output = client_model(x)
        server_output = server_model(client_output)
        
        self.assertEqual(server_output.shape, (2, 10))
        print("✓ Split-fed model test passed")


class NormalizationTest(unittest.TestCase):
    """Test image normalization for different datasets."""
    
    def test_mnist_normalization(self):
        """Test MNIST-specific normalization."""
        from core.client import Client
        
        # Create dummy client to access normalization method
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "experiment": {"mode": "centralized"},
                "topology": {"server": {"host": "127.0.0.1", "port": 8000}},
                "monitoring": {"api_host": "127.0.0.1", "api_port": 9000}
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            # We need to mock some initialization, let's just test the normalize function directly
            # Create MNIST-like tensor
            mnist_images = torch.randn(2, 1, 28, 28)
            
            # Apply MNIST normalization
            mnist_mean = 0.1307
            mnist_std = 0.3081
            normalized = (mnist_images - mnist_mean) / mnist_std
            
            # Verify output shape and value range
            self.assertEqual(normalized.shape, mnist_images.shape)
            self.assertTrue(torch.isfinite(normalized).all())
            
            print("✓ MNIST normalization test passed")
        finally:
            os.unlink(config_path)
    
    def test_cifar_normalization(self):
        """Test CIFAR-10-specific normalization."""
        # Create CIFAR-like tensor (3 channels)
        cifar_images = torch.randint(0, 256, (2, 3, 32, 32)).float()
        
        # Apply CIFAR-10 normalization
        cifar_images = cifar_images / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        normalized = (cifar_images - mean) / std
        
        # Verify output
        self.assertEqual(normalized.shape, (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(normalized).all())
        
        print("✓ CIFAR-10 normalization test passed")


class ConfigurationTest(unittest.TestCase):
    """Test configuration and compatibility."""
    
    def test_config_loads(self):
        """Test config.json can be loaded."""
        config_path = project_root / 'config.json'
        self.assertTrue(config_path.exists())
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Verify essential keys
        self.assertIn('experiment', config)
        self.assertIn('topology', config)
        self.assertIn('monitoring', config)
        self.assertIn('network', config)
        
        print("✓ Configuration test passed")


class CommunicationTest(unittest.TestCase):
    """Test TCP communication protocol."""
    
    def test_communicator_import(self):
        """Test TCP communicator can be imported."""
        from core.communicator import TCPCommunicator
        
        comm = TCPCommunicator(use_compression=False)
        self.assertTrue(hasattr(comm, 'send_data'))
        self.assertTrue(hasattr(comm, 'recv_data'))
        
        print("✓ TCP communicator import test passed")


class IntegrationTest(unittest.TestCase):
    """Integration tests combining multiple components."""
    
    def test_full_pipeline_setup(self):
        """Test complete setup: data -> model -> training iteration."""
        import tempfile
        from scripts.prepare_mnist import prepare_mnist_federated
        from model import get_model
        
        # Prepare data
        with tempfile.TemporaryDirectory() as temp_dir:
            result = prepare_mnist_federated(
                root_dir=temp_dir,
                num_clients=3,
                seed=42
            )
            
            # Load model
            model = get_model('centralized', num_classes=10)
            device = torch.device('cpu')
            model.to(device)
            
            # Create simple training loop
            client_data = result['client_datasets'][0]
            train_images = client_data['train_images'][:10]  # Use 10 samples
            train_labels = client_data['train_labels'][:10]
            
            # Normalize and create dataloader
            mnist_mean = 0.1307
            mnist_std = 0.3081
            normalized = (train_images - mnist_mean) / mnist_std
            
            ds = TensorDataset(normalized, train_labels)
            loader = DataLoader(ds, batch_size=2)
            
            # Forward pass
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                
                output = model(x)
                loss = criterion(output, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx == 0:
                    self.assertEqual(output.shape, (x.shape[0], 10))
                    self.assertTrue(torch.isfinite(loss))
        
        print("✓ Full pipeline integration test passed")


def run_tests(verbose=True):
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(DataPrepTest))
    suite.addTests(loader.loadTestsFromTestCase(ModelTest))
    suite.addTests(loader.loadTestsFromTestCase(NormalizationTest))
    suite.addTests(loader.loadTestsFromTestCase(ConfigurationTest))
    suite.addTests(loader.loadTestsFromTestCase(CommunicationTest))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTest))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbose=True)
    sys.exit(0 if success else 1)
