"""Unit tests for model module."""
import pytest
import torch
import torch.nn as nn
from model import get_model, CNN, SplitClientCNN, SplitServerCNN

B = 1  # batch size


def test_get_model_centralized():
    model = get_model('centralized', num_classes=10)
    assert isinstance(model, CNN)
    assert isinstance(model, nn.Module)
    x = torch.randn(B, 1, 28, 28)
    out = model(x)
    assert out.shape == (B, 10)


def test_get_model_ring():
    model = get_model('ring', num_classes=10)
    assert isinstance(model, CNN)
    x = torch.randn(B, 1, 28, 28)
    out = model(x)
    assert out.shape == (B, 10)


def test_get_model_splitfed():
    client_model, server_model = get_model('splitfed', num_classes=10)
    assert isinstance(client_model, SplitClientCNN)
    assert isinstance(server_model, SplitServerCNN)
    x = torch.randn(B, 1, 28, 28)
    smashed = client_model(x)
    out = server_model(smashed)
    assert out.shape == (B, 10)


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown mode"):
        get_model('unknown_mode')


def test_model_device_cpu():
    model = get_model('centralized', num_classes=10)
    model.to(torch.device('cpu'))
    for p in model.parameters():
        assert p.device == torch.device('cpu')


def test_cnn_internal_layers():
    model = CNN(num_classes=10)
    x = torch.randn(B, 1, 28, 28)
    out = model.pool(torch.relu(model.conv1(x)))
    assert out.shape == (B, 32, 14, 14)
    out = model.pool(torch.relu(model.conv2(out)))
    assert out.shape == (B, 64, 7, 7)


def test_splitfed_custom_classes():
    client_model, server_model = get_model('splitfed', num_classes=5)
    x = torch.randn(B, 1, 28, 28)
    out = server_model(client_model(x))
    assert out.shape == (B, 5)
