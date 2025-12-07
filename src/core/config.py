# Configuration Loading Utilities
#
# Helper functions for loading and parsing YAML configuration files.

import yaml
from typing import Any, Dict, Optional


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


def load_mode_config(
    config_path: Optional[str],
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load mode-specific configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        mode: Configuration mode to load (if None, uses default_mode from config)
        
    Returns:
        Mode configuration dictionary
    """
    if config_path is None:
        return {}
    
    config_data = load_config(config_path)
    mode = mode or config_data.get('default_mode', 'MINIMAL')
    return config_data.get('modes', {}).get(mode, {})

