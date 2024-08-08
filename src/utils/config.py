from typing import Dict, Any
import yaml


def get_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
