from __future__ import annotations

import pytest
import yaml

from kapso.core.config import load_config, load_effective_config
from kapso.cross_run.embedding_space import EmbeddingSpace
from kapso.cross_run.launch.production import build_production_launch_services
from kapso.cross_run.launch.starting_artifacts import (
    LaunchStartingArtifactSetProvider,
)
from kapso.cross_run.settings import CrossRunSettings, EmbeddingSettings

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _embedding_space(config, mode):
    profile_name = config["modes"][mode]["ideation_profile"]
    settings = EmbeddingSettings.from_dict(
        config["ideation_profiles"][profile_name]["embeddings"]
    )
    return EmbeddingSpace.mint(
        provider=settings.provider,
        model=settings.model,
        dimensions=settings.dimensions,
        canonicalizer_version=settings.canonicalizer_version,
    )


@pytest.mark.parametrize(
    ("config_path", "mode"),
    (
        ("benchmarks/posttrain/config.yaml", "POSTTRAIN"),
        ("benchmarks/relbench/config.yaml", "RELBENCH_GENERIC"),
    ),
)
def test_benchmark_bindings_compose_the_same_configured_repository_triple(
    tmp_path,
    config_path,
    mode,
):
    canonical = load_config(_CANONICAL_CONFIG_PATH)
    workload = load_config(config_path)
    canonical_settings = CrossRunSettings.from_dict(canonical["cross_run"])
    workload["cross_run"] = canonical["cross_run"]
    workload["cross_run_registry_fingerprint"] = canonical_settings.scopes.fingerprint
    runtime_path = tmp_path / f"{mode}.yaml"
    runtime_path.write_text(
        yaml.safe_dump(workload, sort_keys=False),
        encoding="utf-8",
    )
    effective = load_effective_config(str(runtime_path), mode)
    settings = effective.cross_run
    services = build_production_launch_services(
        settings=settings,
        binding=effective.cross_run_binding,
        experiment_embedding_space=_embedding_space(canonical, "GENERIC"),
        starting_artifacts=LaunchStartingArtifactSetProvider((), settings.launch),
        state_root=(tmp_path / mode).absolute(),
    )

    repositories = services.github_resolver.repositories_for_scope(
        effective.cross_run_binding.scope_id
    )

    assert repositories == canonical_settings.scopes.resolve("ml_ai")
    assert services.task_adapter_store.settings is settings.expert.task_adapters
