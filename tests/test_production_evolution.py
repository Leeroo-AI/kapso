from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kapso.core.config import load_config, load_effective_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch import production_evolution
from kapso.cross_run.launch.contracts import LaunchTaskContextRequest
from kapso.cross_run.launch.production_evolution import (
    execute_production_evolution,
    ProductionEvolutionError,
)
from test_launch_resolver import resolver_case
from test_run_frontier_action_gate import _action_case, _reserve_ideation_agent
from test_run_state_publisher import publisher_case

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _runtime_config(tmp_path: Path):
    config = load_config(_CANONICAL_CONFIG_PATH)
    config["cross_run"]["launch"]["coding_agent_image"] = {
        "image_reference": (
            "registry.example.com/kapso/coding-agent@sha256:" + "a" * 64
        ),
        "image_config_digest": "sha256:" + "b" * 64,
        "operating_system": "linux",
        "architecture": "amd64",
        "architecture_variant": None,
    }
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return load_effective_config(str(path), "GENERIC")


def test_runtime_config_requires_and_builds_one_pinned_action_image(tmp_path):
    effective = _runtime_config(tmp_path)

    authority = production_evolution._configured_image_authority(effective)

    assert authority.image_reference.endswith("@sha256:" + "a" * 64)
    assert authority.image_config_digest == "sha256:" + "b" * 64


def test_fresh_input_validation_precedes_any_embedding_or_launch_call(
    tmp_path,
    monkeypatch,
):
    def forbidden_embedding(*_arguments, **_keywords):
        raise AssertionError("embedding constructed before prepared handoff")

    monkeypatch.setattr(
        production_evolution,
        "OpenAIEmbeddingProvider",
        forbidden_embedding,
    )

    with pytest.raises(ProductionEvolutionError, match="fresh evolution requires"):
        execute_production_evolution(
            effective_config=_runtime_config(tmp_path),
            goal="Preserve this complete goal.",
            run_root=(tmp_path / "run").absolute(),
            state_root=tmp_path.absolute(),
            task_context_request=None,
            starting_artifact_sources={},
            dependency_runtime_contract=None,
            budget_fidelity_envelope=None,
            scope_id="ml_ai",
            task_family_id="language_model_post_training",
            task_adapter_id="posttrain",
            requested_coding_agent="codex",
            objective_direction="maximize",
            additional_context="",
            resume=False,
        )


def test_resume_rejects_rederived_fresh_inputs_before_external_access(tmp_path):
    with pytest.raises(ProductionEvolutionError, match="pinned local launch inputs"):
        execute_production_evolution(
            effective_config=_runtime_config(tmp_path),
            goal="Resume the pinned goal.",
            run_root=(tmp_path / "run").absolute(),
            state_root=tmp_path.absolute(),
            task_context_request=None,
            starting_artifact_sources={},
            dependency_runtime_contract={"runtime": "python"},
            budget_fidelity_envelope=None,
            scope_id="ml_ai",
            task_family_id="language_model_post_training",
            task_adapter_id="posttrain",
            requested_coding_agent="codex",
            objective_direction="maximize",
            additional_context="",
            resume=True,
        )


def test_evolution_prompt_preserves_full_goal_context_and_repository_memory():
    goal = "goal-prefix\n" + "g" * 200_000 + "\ngoal-suffix"
    context = "context-prefix\n" + "c" * 200_000 + "\ncontext-suffix"
    memory = b'{"summary":"complete repository memory"}'

    prompt = production_evolution._evolution_prompt(
        goal=goal,
        additional_context=context,
        repository_memory=memory,
    )

    assert goal in prompt
    assert context in prompt
    assert memory.decode("utf-8") in prompt


def test_resume_prompt_inputs_must_match_the_pinned_fresh_launch(tmp_path):
    goal = "Preserve this complete goal."
    context = "Preserve this complete caller context."
    preparation = production_evolution.build_production_launch_preparation(
        effective_config=_runtime_config(tmp_path),
        goal=goal,
        additional_context=context,
        task_context_request=LaunchTaskContextRequest.mint(
            capability_tags=("predict",),
            input_contract_fingerprint=tree_or_blob_digest(b"input"),
            target_contract_fingerprint=tree_or_blob_digest(b"target"),
            starting_artifact_refs=(),
            method_fingerprint=tree_or_blob_digest(b"method"),
            toolchain_fingerprint=tree_or_blob_digest(b"toolchain"),
            dependency_runtime_fingerprint=tree_or_blob_digest(b"runtime"),
            budget_hardware_envelope={"hardware": "cpu"},
            transfer_dimensions={},
        ),
        starting_artifact_sources={},
        dependency_runtime_contract={"runtime": "python"},
        budget_fidelity_envelope={"fidelity": "full"},
        scope_id="ml_ai",
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
        requested_coding_agent="codex",
    )

    production_evolution._require_pinned_prompt_inputs(
        preparation.request,
        goal=goal,
        additional_context=context,
    )
    with pytest.raises(ProductionEvolutionError, match="prompt inputs differ"):
        production_evolution._require_pinned_prompt_inputs(
            preparation.request,
            goal=goal,
            additional_context=context + " changed",
        )


def test_resume_recovers_only_actions_ahead_of_the_checkpoint(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    projected = frontier.projection.action_ledger

    assert not production_evolution._unprojected_action_tails(
        projected,
        projected,
    )

    reservation = _reserve_ideation_agent(gate, frontier)
    unprojected = production_evolution._unprojected_action_tails(
        projected,
        gate._publisher.action_ledger_snapshot(),
    )

    assert tuple(tail.operation_id for tail in unprojected) == (
        reservation.intent.operation_id,
    )
    publisher_case["active"].close()
