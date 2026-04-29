"""Integration tests for data preparation."""
import os
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.integration
def test_mnist_data_preparation(test_data_dir):
    from scripts.prepare_mnist import prepare_mnist_federated

    result = prepare_mnist_federated(root_dir=str(test_data_dir), num_clients=2, seed=42)

    assert 'client_datasets' in result
    assert 'server_test_dataset' in result
    assert 'statistics' in result
    assert len(result['client_datasets']) == 2

    splits_dir = test_data_dir / 'splits'
    assert splits_dir.exists()

    for client_id in range(1, 3):
        client_file = splits_dir / f'client_{client_id}_data.pt'
        assert client_file.exists()
        data = torch.load(client_file)
        for key in ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']:
            assert key in data


@pytest.mark.integration
def test_data_loading_after_preparation(test_data_dir):
    from scripts.prepare_mnist import prepare_mnist_federated

    result = prepare_mnist_federated(root_dir=str(test_data_dir), num_clients=2, seed=42)
    client_data = result['client_datasets'][0]
    train_images = client_data['train_images']
    assert train_images.shape[1] == 1
    assert train_images.shape[2] == 28
    assert train_images.shape[3] == 28


@pytest.mark.integration
def test_full_training_loop(test_data_dir):
    from scripts.prepare_mnist import prepare_mnist_federated
    from model import get_model
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    result = prepare_mnist_federated(root_dir=str(test_data_dir), num_clients=2, seed=42)

    model = get_model('centralized', num_classes=10)
    model.to(torch.device('cpu'))

    client_data = result['client_datasets'][0]
    train_images = client_data['train_images'][:4]
    train_labels = client_data['train_labels'][:4]

    ds = TensorDataset(train_images, train_labels)
    loader = DataLoader(ds, batch_size=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert output.shape == (x.shape[0], 10)
        assert torch.isfinite(loss)
        break  # one batch is enough


@pytest.mark.integration
def test_split_mnist_wrapper(test_data_dir):
    from scripts.split import split_mnist_for_federated

    result = split_mnist_for_federated(root=str(test_data_dir), num_clients=2, seed=42)
    assert 'client_datasets' in result
    assert len(result['client_datasets']) == 2


@pytest.mark.integration
def test_non_iid_split_statistics(test_data_dir):
    from scripts.prepare_mnist import prepare_mnist_federated

    result = prepare_mnist_federated(root_dir=str(test_data_dir), num_clients=2, seed=42)
    stats = result['statistics']
    assert 'clients' in stats
    assert len(stats['clients']) == 2
    for cs in stats['clients']:
        assert 'client_id' in cs
        assert 'train_samples' in cs
        assert 'unique_classes' in cs
