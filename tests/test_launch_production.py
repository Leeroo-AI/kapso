from __future__ import annotations

import pytest
import yaml

from kapso.core.config import load_config, load_effective_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.embedding_space import EmbeddingSpace
from kapso.cross_run.launch.contracts import LaunchTaskContextRequest
from kapso.cross_run.launch.production import (
    build_production_launch_preparation,
    build_production_launch_services,
    ProductionLaunchCompositionError,
)
from kapso.cross_run.launch.starting_artifacts import (
    LaunchStartingArtifactSetProvider,
)
from kapso.cross_run.settings import CrossRunSettings, EmbeddingSettings

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _task_context():
    fingerprint = tree_or_blob_digest(b"production launch fixture")
    return LaunchTaskContextRequest.mint(
        capability_tags=("predict",),
        input_contract_fingerprint=fingerprint,
        target_contract_fingerprint=fingerprint,
        starting_artifact_refs=(),
        method_fingerprint=fingerprint,
        toolchain_fingerprint=fingerprint,
        dependency_runtime_fingerprint=fingerprint,
        budget_hardware_envelope={"hardware": "cpu"},
        transfer_dimensions={},
    )


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


@pytest.mark.parametrize(
    ("config_path", "mode", "expected_family", "expected_adapter"),
    (
        (
            "benchmarks/posttrain/config.yaml",
            "POSTTRAIN",
            "language_model_post_training",
            "posttrain",
        ),
        (
            "benchmarks/relbench/config.yaml",
            "RELBENCH_GENERIC",
            "relational_tabular_prediction",
            "relbench",
        ),
    ),
)
def test_benchmark_launch_preparation_obtains_binding_from_selected_mode(
    tmp_path,
    config_path,
    mode,
    expected_family,
    expected_adapter,
):
    canonical = load_config(_CANONICAL_CONFIG_PATH)
    workload = load_config(config_path)
    settings = CrossRunSettings.from_dict(canonical["cross_run"])
    workload["cross_run"] = canonical["cross_run"]
    workload["cross_run_registry_fingerprint"] = settings.scopes.fingerprint
    runtime_path = tmp_path / f"{mode}-preparation.yaml"
    runtime_path.write_text(
        yaml.safe_dump(workload, sort_keys=False),
        encoding="utf-8",
    )
    effective = load_effective_config(str(runtime_path), mode)

    prepared = build_production_launch_preparation(
        effective_config=effective,
        goal="Improve the synthetic task",
        additional_context="",
        task_context_request=_task_context(),
        starting_artifact_sources={},
        dependency_runtime_contract={"runtime": "python"},
        budget_fidelity_envelope={"fidelity": "full"},
        scope_id=None,
        task_family_id=None,
        task_adapter_id=None,
        requested_coding_agent="codex",
    )

    assert prepared.binding.scope_id == "ml_ai"
    assert prepared.binding.task_family_id == expected_family
    assert prepared.binding.task_adapter_id == expected_adapter
    assert prepared.request.binding == prepared.binding
    assert prepared.request.requested_coding_agent == "codex"
    assert dict(prepared.request.starting_artifact_content_ids) == {}


def test_selected_benchmark_rejects_an_explicit_binding_override(tmp_path):
    canonical = load_config(_CANONICAL_CONFIG_PATH)
    workload = load_config("benchmarks/posttrain/config.yaml")
    settings = CrossRunSettings.from_dict(canonical["cross_run"])
    workload["cross_run"] = canonical["cross_run"]
    workload["cross_run_registry_fingerprint"] = settings.scopes.fingerprint
    runtime_path = tmp_path / "posttrain.yaml"
    runtime_path.write_text(
        yaml.safe_dump(workload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ProductionLaunchCompositionError, match="differs"):
        build_production_launch_preparation(
            effective_config=load_effective_config(str(runtime_path), "POSTTRAIN"),
            goal="Improve the synthetic task",
            additional_context="",
            task_context_request=_task_context(),
            starting_artifact_sources={},
            dependency_runtime_contract={"runtime": "python"},
            budget_fidelity_envelope={"fidelity": "full"},
            scope_id="ml_ai",
            task_family_id="relational_tabular_prediction",
            task_adapter_id="relbench",
            requested_coding_agent="codex",
        )


def test_generic_launch_requires_and_accepts_one_explicit_binding():
    prepared = build_production_launch_preparation(
        effective_config=load_effective_config(_CANONICAL_CONFIG_PATH, "GENERIC"),
        goal="Improve the synthetic task",
        additional_context="",
        task_context_request=_task_context(),
        starting_artifact_sources={},
        dependency_runtime_contract={"runtime": "python"},
        budget_fidelity_envelope={"fidelity": "full"},
        scope_id="ml_ai",
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
        requested_coding_agent="codex",
    )

    assert prepared.request.binding == prepared.binding
    assert prepared.binding.task_family_id == "relational_tabular_prediction"
