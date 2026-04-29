"""Model registry for creating models by name."""

from src.model.base import BaseModel
from src.model.simple_cnn import create_simple_cnn

_REGISTRY: dict[str, type] = {}


def register_model(name: str, cls: type) -> None:
    """Register a model class under a name.

    Args:
        name: Model name.
        cls: Model class.
    """
    _REGISTRY[name] = cls


def get_model(name: str, **kwargs: object) -> BaseModel:
    """Create a model by name.

    Args:
        name: Registered model name.
        **kwargs: Arguments forwarded to the factory.

    Returns:
        BaseModel: Instantiated model.

    Raises:
        ValueError: If the model name is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)


# Register built-in models
register_model("simple_cnn", create_simple_cnn)
