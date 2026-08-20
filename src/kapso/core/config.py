# Configuration Loading Utilities
#
# Helper functions for loading and parsing YAML configuration files.

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# The platform-generic base layer every mode resolves over: the `defaults:`
# mapping in the package's own config file. Resolved relative to the package
# (src/kapso/config.yaml), never from the environment.
PLATFORM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Parsed configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``override`` onto ``base`` without mutating either.

    Nested dicts merge key-by-key; scalars AND lists replace wholesale.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_platform_defaults() -> Dict[str, Any]:
    """The platform config's `defaults:` mapping — the base config layer."""
    platform_config = load_config(str(PLATFORM_CONFIG_PATH))
    if 'defaults' not in platform_config:
        raise ValueError(
            f"Platform config {PLATFORM_CONFIG_PATH} has no 'defaults' section"
        )
    return platform_config['defaults']


def load_mode_config(
    config_path: Optional[str],
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve a mode's effective configuration: the file's modes[mode]
    deep-merged over the platform `defaults:` layer, so benchmark configs
    carry overrides only and inherit everything else.

    Args:
        config_path: Path to config.yaml (None returns {} — the documented
            no-config default)
        mode: Configuration mode to resolve (if None, uses default_mode
            from the config file)

    Returns:
        The resolved configuration dictionary. An unknown mode raises.
    """
    if config_path is None:
        return {}

    config_data = load_config(config_path)
    mode = mode or config_data.get('default_mode', 'MINIMAL')
    modes = config_data.get('modes', {})
    if mode not in modes:
        raise ValueError(
            f"Unknown mode {mode!r} in {config_path}; "
            f"available: {sorted(modes)}"
        )
    return deep_merge(load_platform_defaults(), modes[mode] or {})
