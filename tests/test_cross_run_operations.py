"""Operational commands remain thin and emit complete non-secret receipts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.cli as cli_module
import kapso.cross_run.operations as operations_module
from kapso.cli import main
from kapso.core.config import load_effective_config
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import (
    CompletionState,
    ExpertPromotionState,
    ExpertValidationStage,
)
from kapso.cross_run.expert.providers import GitHubExpertCurrentReleaseProvider
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderRegistry,
    project_prepared_task_evaluation_cases,
)
from kapso.cross_run.operations import (
    GitHubOperationServices,
    capture_cross_run,
    propose_expert_cross_run,
    publish_knowledge_cross_run,
    resolve_launch_cross_run,
    validate_expert_cross_run,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStore
from cross_run_capture_fixtures import make_capture_fixture
from test_cross_run_retrieval import source_fixture
from test_expert_proposal import BootstrapProposalRunner, bootstrap_output
from test_expert_review import _review_fixture
from test_expert_publication_eligibility import _coordinator, _publish_matrix
from test_expert_promotion_decision import _settings as promotion_settings
from test_expert_release_matrix_reservation import _bootstrap_release_matrix_fixture
from test_expert_task_evaluation_execution_store import (
    _DenylistAuthority,
    _Provider,
)
from test_expert_triggers import trigger_packet, trigger_settings
from test_knowledge_snapshot_publisher import RecordingPublicationAuthority
from test_launch_bootstrap import _fresh_coordinator
from test_launch_handoff import _DescendantSecurityAuthority
from test_launch_resolver import resolver_case

_CONFIG_PATH = "src/kapso/config.yaml"
_COMMITTED_AT = "2026-07-27T12:00:00Z"


class _AbsentExpertResolver:
    def diagnose_repository(self, scope_id, artifact_kind):
        assert scope_id == "ml_ai"
        assert artifact_kind.value == "expert_base_release"
        return SimpleNamespace(
            repository_full_name="Leeroo-AI/kapso-expert",
            repository_node_id="expert_repo_node",
        )

    def read_current_pointer_state(self, scope_id, artifact_kind, *, allow_missing):
        assert scope_id == "ml_ai"
        assert artifact_kind.value == "expert_base_release"
        assert allow_missing is True
        return SimpleNamespace(pointer=None, head_commit_sha="a" * 40)


def test_capture_command_runs_the_real_pipeline_and_reports_exact_bundle(tmp_path):
    fixture_root = tmp_path / "workspace"
    fixture_root.mkdir()
    fixture = make_capture_fixture(fixture_root)
    request_path = tmp_path / "capture-request.json"
    request_path.write_text(
        json.dumps(_capture_request_payload(fixture.request), sort_keys=True),
        encoding="utf-8",
    )

    result = capture_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
    )

    assert result["operation"] == "capture"
    assert result["run_id"] == fixture.request.run_id
    assert result["campaign_id"] == fixture.request.campaign_id
    assert result["bundle_id"].startswith("run-bundle:sha256:")
    assert result["completion_state"] == CompletionState.STOPPED.value
    assert result["artifact_digests"]


def test_expert_validation_services_use_the_canonical_task_adapter_root(tmp_path):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    state_root = tmp_path / "state"
    state_root.mkdir(mode=0o700)
    services = operations_module._expert_validation_services(
        settings,
        state_root,
        GitHubOperationServices(
            resolver=object(),
            materializer=object(),
            publisher=object(),
        ),
    )

    assert services.task_adapter_store.state_root == state_root
    assert services.task_adapter_store.state_path == (
        state_root / settings.expert.task_adapters.state_path
    )


def test_publish_knowledge_delegates_exact_empty_generation_to_m2(
    tmp_path,
    monkeypatch,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    scope_contract = source_fixture()[0]
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    generation = CrossRunCatalog(
        catalog_root,
        scope_contract,
        settings.catalog,
    ).store.read_current()
    authority = RecordingPublicationAuthority()
    monkeypatch.setattr(
        operations_module,
        "_github_services",
        lambda _settings, _state_root: GitHubOperationServices(
            resolver=object(),
            materializer=object(),
            publisher=authority,
        ),
    )
    request_path = tmp_path / "publish.json"
    request_path.write_text(
        json.dumps(
            {
                "catalog_root": "catalog",
                "scope_contract": scope_contract.to_dict(),
                "expected_parent_sha": "a" * 40,
                "expected_current_snapshot_id": None,
                "committed_at": _COMMITTED_AT,
                "validation_closure_ids": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = publish_knowledge_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
        state_root=tmp_path,
    )

    assert result["operation"] == "publish-knowledge"
    assert result["snapshot_id"].startswith("knowledge-snapshot:sha256:")
    assert result["catalog_generation_id"] == generation.catalog_generation_id
    assert result["embedding"] is None
    assert result["commit_sha"] == "b" * 40


def test_cli_wrapper_only_parses_and_prints_operation_receipt(
    tmp_path,
    monkeypatch,
    capsysbinary,
):
    expected = {
        "operation": "inspect",
        "scope_id": "ml_ai",
        "artifacts": {},
    }
    calls = []

    def inspect(**arguments):
        calls.append(arguments)
        return expected

    monkeypatch.setattr(cli_module, "inspect_cross_run", inspect)

    main(
        [
            "cross-run",
            "inspect",
            "--config",
            _CONFIG_PATH,
            "--mode",
            "GENERIC",
            "--scope-id",
            "ml_ai",
            "--state-root",
            str(tmp_path),
        ]
    )

    assert calls == [
        {
            "config_path": _CONFIG_PATH,
            "mode": "GENERIC",
            "scope_id": "ml_ai",
            "state_root": Path(tmp_path),
        }
    ]
    assert json.loads(capsysbinary.readouterr().out) == expected


def test_propose_expert_runs_existing_architect_and_seals_candidate(
    tmp_path,
    monkeypatch,
):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    request_path = tmp_path / "propose.json"
    request_path.write_text(
        json.dumps({"evidence_packet": packet.to_dict()}, sort_keys=True),
        encoding="utf-8",
    )
    runner = BootstrapProposalRunner(
        tmp_path / "fixture-agent-artifacts",
        bootstrap_output(),
        {
            "src/execution.py": b"def execute(task):\n    return task.run()\n",
            "tests/test_execution.py": b"def test_execute():\n    assert True\n",
        },
    )
    monkeypatch.setattr(
        operations_module,
        "_coding_agent_runner",
        lambda _settings, _state_root: runner,
    )

    result = propose_expert_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
        state_root=tmp_path,
    )

    assert result["operation"] == "propose-expert"
    assert result["candidate_id"].startswith("expert-candidate:sha256:")
    assert result["source_base_release_id"] is None
    assert result["change_kind"] == "repository_architecture"
    assert len(runner.calls) == 1


def test_resolve_launch_preserves_complete_request_and_pins_workspace(
    tmp_path,
    resolver_case,
    monkeypatch,
):
    launch_request = resolver_case["request"]
    launch_values = launch_request.to_dict()
    artifact_inputs = {}
    for artifact_ref in launch_request.task_context_request.starting_artifact_refs:
        source = tmp_path / artifact_ref
        source.mkdir()
        artifact_inputs[artifact_ref] = {
            "source": artifact_ref,
            "mount_path": f"inputs/{artifact_ref}",
        }
    request_path = tmp_path / "launch.json"
    request_path.write_text(
        json.dumps(
            {
                "goal": "goal-prefix\n" + "g" * 20_000 + "\ngoal-suffix",
                "additional_context": "context-prefix\ncontext-suffix",
                "task_context_request": (launch_request.task_context_request.to_dict()),
                "starting_artifacts": artifact_inputs,
                "dependency_runtime_contract": launch_values[
                    "dependency_runtime_contract"
                ],
                "budget_fidelity_envelope": launch_values["budget_fidelity_envelope"],
                "scope_id": launch_request.binding.scope_id,
                "task_family_id": launch_request.binding.task_family_id,
                "task_adapter_id": launch_request.binding.task_adapter_id,
                "requested_coding_agent": launch_request.requested_coding_agent,
                "objective_direction": "maximize",
                "empty_scope_bootstrap_authorization_id": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    captured = {}

    def prepare(**arguments):
        captured.update(arguments)
        return SimpleNamespace(
            binding=launch_request.binding,
            experiment_embedding_space=object(),
            starting_artifacts=object(),
            request=launch_request,
        )

    monkeypatch.setattr(
        operations_module,
        "load_effective_config",
        lambda _path, _mode: SimpleNamespace(
            cross_run=resolver_case["resolver"].settings
        ),
    )
    monkeypatch.setattr(
        operations_module,
        "build_production_launch_preparation",
        prepare,
    )
    monkeypatch.setattr(
        operations_module,
        "build_production_launch_services",
        lambda **_arguments: SimpleNamespace(
            coordinator=_fresh_coordinator(resolver_case),
            security_authority=_DescendantSecurityAuthority(),
        ),
    )

    result = resolve_launch_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
    )

    assert captured["goal"].endswith("goal-suffix")
    assert len(captured["goal"]) > 20_000
    assert captured["additional_context"].endswith("context-suffix")
    assert result["operation"] == "resolve-launch"
    assert result["expert_release_id"] == (
        resolver_case["expert_package"].manifest.release_id
    )
    assert result["knowledge_snapshot_id"] == (
        resolver_case["knowledge_package"].manifest.snapshot_id
    )


def test_validate_expert_runs_the_existing_restart_aware_review_stage(
    tmp_path,
    monkeypatch,
):
    review_root = tmp_path / "review-system"
    _coordinator, _prepared, _workspace, _runner, snapshot, store = _review_fixture(
        review_root,
        monkeypatch,
    )
    store = ExpertValidationStore(
        store.root,
        store.state_root,
        store.settings,
        store.reducer,
    )
    services = operations_module.ExpertValidationOperationServices(
        candidate_store=store.reducer.candidate_store,
        validation_store=store,
        task_adapter_store=store.reducer.task_adapter_provider,
    )
    monkeypatch.setattr(
        operations_module,
        "_github_services",
        lambda _settings, _state_root: GitHubOperationServices(
            resolver=object(),
            materializer=object(),
            publisher=object(),
        ),
    )
    monkeypatch.setattr(
        operations_module,
        "_expert_validation_services",
        lambda _settings, _state_root, _github: services,
    )
    request_path = tmp_path / "validate.json"
    request_path.write_text(
        json.dumps(
            {
                "candidate_id": snapshot.state.candidate_id,
                "expected_transition_id": snapshot.transition.transition_id,
                "evaluator_result": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = validate_expert_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
        state_root=review_root,
    )

    assert result["operation"] == "validate-expert"
    assert result["candidate_id"] == snapshot.state.candidate_id
    assert result["transition_id"] != snapshot.transition.transition_id
    assert result["accepted_stage_result_ids"][-1].startswith(
        "expert-automated-review-stage-result:sha256:"
    )


def test_validate_expert_recovers_and_publishes_bootstrap_release_matrix(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, _prepared_plan, adapter_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    current_release = GitHubExpertCurrentReleaseProvider(_AbsentExpertResolver())
    validation_store.reducer.current_release_provider = current_release
    services = operations_module.ExpertValidationOperationServices(
        candidate_store=validation_store.reducer.candidate_store,
        validation_store=validation_store,
        task_adapter_store=adapter_provider,
    )
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    validation_settings = validation_store.settings
    settings = replace(
        settings,
        docker=validation_settings.task_evaluation_provider.runtime,
        expert=replace(
            settings.expert,
            validation=validation_settings,
        ),
    )
    github = GitHubOperationServices(
        resolver=current_release.resolver,
        materializer=object(),
        publisher=object(),
    )
    runtime = {}
    providers = []

    def registry(*, prepared_request, workspace_root):
        assert workspace_root == (tmp_path / "state").resolve()
        runtime["prepared"] = prepared_request
        provider_keys = tuple(
            sorted(
                {
                    case.provider_key
                    for case in project_prepared_task_evaluation_cases(prepared_request)
                },
                key=lambda provider_key: provider_key.identity,
            )
        )
        created = tuple(
            _Provider(validation_store.root, provider_key)
            for provider_key in provider_keys
        )
        providers.extend(created)
        return TaskEvaluationExecutionProviderRegistry(prepared_request, created)

    monkeypatch.setattr(operations_module, "_settings", lambda *_arguments: settings)
    monkeypatch.setattr(
        operations_module,
        "_github_services",
        lambda _settings, _state_root: github,
    )
    monkeypatch.setattr(
        operations_module,
        "_expert_validation_services",
        lambda _settings, _state_root, _github: services,
    )
    monkeypatch.setattr(
        operations_module,
        "build_task_evaluation_docker_provider_registry",
        registry,
    )
    monkeypatch.setattr(
        operations_module,
        "_policy_services",
        lambda _settings, _state_root, _github: SimpleNamespace(
            security_authority=_DenylistAuthority(runtime["prepared"]),
        ),
    )
    request_path = tmp_path / "matrix.json"
    request_path.write_text(
        json.dumps(
            {
                "candidate_id": snapshot.state.candidate_id,
                "expected_transition_id": snapshot.transition.transition_id,
                "evaluator_result": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    coordinator = operations_module.ExpertReleaseMatrixStageCoordinator
    original_publish = coordinator.publish_completed

    def interrupt_after_execution(self, **_arguments):
        raise RuntimeError("interrupt before typed matrix publication")

    monkeypatch.setattr(coordinator, "publish_completed", interrupt_after_execution)
    with pytest.raises(RuntimeError, match="interrupt before typed"):
        validate_expert_cross_run(
            config_path=_CONFIG_PATH,
            mode="GENERIC",
            request_path=request_path,
            state_root=tmp_path / "state",
        )
    execution_count = sum(len(provider.execution_calls) for provider in providers)
    assert execution_count > 0
    monkeypatch.setattr(coordinator, "publish_completed", original_publish)
    validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    services = operations_module.ExpertValidationOperationServices(
        candidate_store=validation_store.reducer.candidate_store,
        validation_store=validation_store,
        task_adapter_store=adapter_provider,
    )

    result = validate_expert_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
        state_root=tmp_path / "state",
    )

    assert result["transition_id"] != snapshot.transition.transition_id
    assert result["accepted_stage_result_ids"][-1].startswith(
        "expert-release-matrix-stage-result:sha256:"
    )
    assert result["next_stage"] == ExpertValidationStage.PUBLICATION_ELIGIBILITY.value
    assert (
        sum(len(provider.execution_calls) for provider in providers) == execution_count
    )


def test_validate_expert_executes_typed_publication_eligibility(
    tmp_path,
    monkeypatch,
):
    matrix_root = tmp_path / "matrix"
    matrix_root.mkdir()
    case = _publish_matrix(
        matrix_root,
        monkeypatch,
        bootstrap=True,
        settings=promotion_settings(minimum_replicates=1, minimum_pairs=1),
    )
    services = operations_module.ExpertValidationOperationServices(
        candidate_store=case.validation_store.reducer.candidate_store,
        validation_store=case.validation_store,
        task_adapter_store=case.validation_store.reducer.task_adapter_provider,
    )
    authority = _coordinator(case)
    monkeypatch.setattr(
        operations_module,
        "_github_services",
        lambda _settings, _state_root: GitHubOperationServices(
            resolver=object(),
            materializer=object(),
            publisher=object(),
        ),
    )
    monkeypatch.setattr(
        operations_module,
        "_expert_validation_services",
        lambda _settings, _state_root, _github: services,
    )
    monkeypatch.setattr(
        operations_module,
        "_policy_services",
        lambda _settings, _state_root, _github: object(),
    )
    monkeypatch.setattr(
        operations_module,
        "_publication_eligibility_coordinator",
        lambda _services, _policies: authority.coordinator,
    )
    snapshot = case.matrix_commit.snapshot
    request_path = tmp_path / "eligibility.json"
    request_path.write_text(
        json.dumps(
            {
                "candidate_id": snapshot.state.candidate_id,
                "expected_transition_id": snapshot.transition.transition_id,
                "evaluator_result": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = validate_expert_cross_run(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        request_path=request_path,
        state_root=tmp_path / "state",
    )

    assert result["promotion_state"] == ExpertPromotionState.APPROVED.value
    assert result["next_stage"] is None
    assert result["next_action"] == "publish-expert"


def _capture_request_payload(request):
    return {
        "workspace_dir": str(request.workspace_dir),
        "idea_archive_path": str(request.idea_archive_path),
        "scope_contract_id": request.scope_contract_id,
        "scope_id": request.scope_id,
        "run_id": request.run_id,
        "campaign_id": request.campaign_id,
        "configuration_fingerprint": request.configuration_fingerprint,
        "completion_state": request.completion_state.value,
        "started_at": request.started_at,
        "kapso_commit": request.kapso_commit,
        "launch_manifest_id": request.launch_manifest_id,
        "knowledge_snapshot_id": request.knowledge_snapshot_id,
        "expert_base_release_id": request.expert_base_release_id,
        "task_context_binding": request.task_context_binding.to_dict(),
        "artifact_environment": request.artifact_environment.to_dict(),
        "evaluation_fingerprints": [
            fingerprint.to_dict() for fingerprint in request.evaluation_fingerprints
        ],
        "run_log_paths": list(request.run_log_paths),
    }
