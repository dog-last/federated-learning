"""Unit tests for scripts module."""
from scripts.split import split_mnist_for_federated


def test_split_mnist_import():
    assert callable(split_mnist_for_federated)


def test_run_directory_tests_import():
    from scripts.run_directory_tests import main
    assert callable(main)


def test_prepare_mnist_import():
    from scripts.prepare_mnist import prepare_mnist_federated
    assert callable(prepare_mnist_federated)
