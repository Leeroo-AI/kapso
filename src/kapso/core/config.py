"""Strict YAML configuration loading and cross-run config composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from kapso.cross_run.contracts import CrossRunTaskBindingSettings
from kapso.cross_run.settings import (
    CrossRunConfigurationError,
    CrossRunSettings,
    EffectiveConfig,
    compose_runtime_config,
    validate_runtime_registry,
)


class StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: StrictSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise CrossRunConfigurationError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML mapping and validate any cross-run configuration tree."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.load(config_file, Loader=StrictSafeLoader)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise CrossRunConfigurationError("configuration root must be an object")
    result = dict(loaded)
    if "cross_run" in result:
        CrossRunSettings.from_dict(result["cross_run"])
    return result


def load_effective_config(
    config_path: str,
    mode: Optional[str] = None,
) -> EffectiveConfig:
    """Load one selected mode together with its pinned cross-run registry."""
    config_data = load_config(config_path)
    if "default_mode" not in config_data and mode is None:
        raise CrossRunConfigurationError(
            "configuration requires default_mode when mode is not supplied"
        )
    selected_mode = mode if mode is not None else config_data["default_mode"]
    if not isinstance(selected_mode, str) or not selected_mode:
        raise CrossRunConfigurationError("selected mode must be non-empty text")
    modes = config_data.get("modes")
    if not isinstance(modes, Mapping) or selected_mode not in modes:
        raise CrossRunConfigurationError(f"unknown configuration mode: {selected_mode}")
    mode_config = modes[selected_mode]
    if not isinstance(mode_config, Mapping):
        raise CrossRunConfigurationError(f"mode {selected_mode} must be an object")
    if (
        "cross_run_registry_fingerprint" in config_data
        and "cross_run" not in config_data
    ):
        raise CrossRunConfigurationError(
            "cross_run_registry_fingerprint requires cross_run settings"
        )
    cross_run = None
    registry_fingerprint = None
    cross_run_binding = None
    if "cross_run" in config_data:
        cross_run = CrossRunSettings.from_dict(config_data["cross_run"])
        declared_fingerprint = config_data.get("cross_run_registry_fingerprint")
        if declared_fingerprint is not None:
            validate_runtime_registry(config_data, cross_run)
            registry_fingerprint = declared_fingerprint
        else:
            registry_fingerprint = cross_run.scopes.fingerprint
    if "cross_run_binding" in mode_config:
        cross_run_binding = CrossRunTaskBindingSettings.from_dict(
            mode_config["cross_run_binding"]
        )
    return EffectiveConfig(
        mode_name=selected_mode,
        mode=mode_config,
        cross_run=cross_run,
        registry_source_fingerprint=registry_fingerprint,
        cross_run_binding=cross_run_binding,
    )


def load_mode_config(
    config_path: Optional[str],
    mode: Optional[str] = None,
) -> dict[str, Any]:
    """Return only the selected workload mode after validating global settings."""
    if config_path is None:
        return {}
    return dict(load_effective_config(config_path, mode).mode)


__all__ = [
    "StrictSafeLoader",
    "compose_runtime_config",
    "load_config",
    "load_effective_config",
    "load_mode_config",
]
