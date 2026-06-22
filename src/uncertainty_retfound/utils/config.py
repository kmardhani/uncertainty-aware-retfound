"""Utilities for loading YAML configuration files."""

from pathlib import Path
from typing import Any, cast

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the YAML file is empty or does not contain a YAML mapping.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_config: Any = yaml.safe_load(file)

    if raw_config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {path}")

    return cast(dict[str, Any], raw_config)