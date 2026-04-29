"""Configuration loader."""

from src.core.types import Config


def load_config(path: str) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    return Config.from_yaml(path)
