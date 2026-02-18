"""
Configuration loading utilities.
"""

import tomllib
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """
    Load configuration TOML file.
    Args:
        config_path: Path to configuration file (default: config.toml)
    Returns:
        Dictionary with configuration
    """
    logger.info(f"Loading configuration from {config_path}")
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "rb") as f:
        config = tomllib.load(f)
    logger.info("Configuration loaded successfully")
    return config
