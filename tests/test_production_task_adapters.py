"""Public transport task adapters remain deterministic and replayable."""

from dataclasses import replace

import pytest

import kapso.cross_run.production_smoke as smoke_module
from kapso.core.config import load_effective_config
from kapso.cross_run.production_task_adapters import (
    ProductionTaskAdapterError,
    bootstrap_production_task_adapters,
)
from kapso.cross_run.settings import CodingAgentImageSettings
from kapso.cross_run.task_adapter_authority import CanonicalTaskAdapterAuthority
from kapso.cross_run.task_adapter_store import (
    TaskAdapterAuthorityRegistry,
    TaskAdapterPackageStore,
)

_CONFIG_PATH = "src/kapso/config.yaml"
_MANIFEST_DIGEST = "sha256:" + "1" * 64
_CONFIG_DIGEST = "sha256:" + "2" * 64


def _settings_with_image():
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    return replace(
        settings,
        launch=replace(
            settings.launch,
            coding_agent_image=CodingAgentImageSettings(
                image_reference=(
                    "ghcr.io/leeroo-ai/kapso-coding-agent@" + _MANIFEST_DIGEST
                ),
                image_config_digest=_CONFIG_DIGEST,
                operating_system="linux",
                architecture="amd64",
                architecture_variant=None,
            ),
        ),
    )


def _image_inspection():
    return {
        "Config": {
            "Cmd": None,
            "Entrypoint": None,
            "Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C"],
            "Healthcheck": None,
            "Volumes": None,
        }
    }


def test_bootstrap_publishes_every_scope_binding_and_replays(tmp_path):
    settings = _settings_with_image()
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    state_root = tmp_path.resolve()

    first = bootstrap_production_task_adapters(
        settings=settings,
        state_root=state_root,
        scope_contract=scope_contract,
        image_inspection=_image_inspection(),
    )
    replayed = bootstrap_production_task_adapters(
        settings=settings,
        state_root=state_root,
        scope_contract=scope_contract,
        image_inspection=_image_inspection(),
    )

    assert replayed == first
    assert tuple(
        (item["task_family_id"], item["task_adapter_id"]) for item in first["adapters"]
    ) == (
        ("language_model_post_training", "posttrain"),
        ("relational_tabular_prediction", "relbench"),
    )
    adapter_settings = settings.expert.task_adapters
    store = TaskAdapterPackageStore(
        state_root / adapter_settings.state_path,
        state_root,
        adapter_settings,
        TaskAdapterAuthorityRegistry(
            adapter_settings,
            tuple(
                CanonicalTaskAdapterAuthority(authority)
                for authority in adapter_settings.trusted_authorities
            ),
        ),
    )
    for evidence in first["adapters"]:
        active = store.resolve_active_binding(
            scope_contract_id=scope_contract.scope_contract_id,
            task_family_id=evidence["task_family_id"],
            task_adapter_id=evidence["task_adapter_id"],
        )
        assert active.activation.activation_id == evidence["activation_id"]
        assert len(active.verified_adapter.manifest.release_matrix_cases) == 2
        assert active.verified_adapter.manifest.runtime.image_manifest_digest == (
            _MANIFEST_DIGEST
        )
        promotion_dimensions = {
            dimension.dimension_id: dimension.direction
            for dimension in settings.expert.validation.policy.promotion.pareto_dimensions
        }
        comparison_bindings = (
            active.verified_adapter.manifest.task_evaluator.metric_comparison_bindings
        )
        assert {
            binding.comparison_dimension_id: binding.objective_direction
            for binding in comparison_bindings
        } == promotion_dimensions
        assert {
            fingerprint.metric_name: fingerprint.objective_direction
            for case in active.verified_adapter.manifest.release_matrix_cases
            for fingerprint in case.evaluation_fingerprints
        } == promotion_dimensions
        assert all(
            len(case.evaluation_fingerprints) == len(promotion_dimensions)
            for case in active.verified_adapter.manifest.release_matrix_cases
        )


def test_bootstrap_fails_without_pinned_image(tmp_path):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    settings = replace(
        settings,
        launch=replace(settings.launch, coding_agent_image=None),
    )
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )

    with pytest.raises(ProductionTaskAdapterError, match="pinned coding-agent image"):
        bootstrap_production_task_adapters(
            settings=settings,
            state_root=tmp_path.resolve(),
            scope_contract=scope_contract,
            image_inspection=_image_inspection(),
        )
