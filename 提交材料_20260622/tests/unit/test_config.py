"""Unit tests for configuration."""
import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_config_loads():
    config_path = PROJECT_ROOT / 'config.json'
    assert config_path.exists()
    with open(config_path) as f:
        config = json.load(f)
    for key in ['experiment', 'topology', 'monitoring', 'network']:
        assert key in config


def test_config_structure():
    with open(PROJECT_ROOT / 'config.json') as f:
        config = json.load(f)
    assert 'mode' in config['experiment']
    assert config['experiment']['global_epochs'] > 0
    assert 'server' in config['topology']
    assert 'clients' in config['topology']
    assert len(config['topology']['clients']) > 0
    assert 'api_host' in config['monitoring']
    assert 'api_port' in config['monitoring']


def test_config_modes():
    with open(PROJECT_ROOT / 'config.json') as f:
        config = json.load(f)
    assert config['experiment']['mode'] in ('centralized', 'splitfed', 'ring')
