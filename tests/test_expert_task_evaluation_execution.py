from dataclasses import fields, replace

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationInvocationAllocation,
    TaskEvaluationLegKind,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    ExecutableTaskEvaluationCase,
    ResolvedTaskEvaluationCase,
    TaskEvaluationExecutionError,
    TaskEvaluationExecutionProviderKey,
    TaskEvaluationExecutionProviderRegistry,
    TaskEvaluationLegInvocation,
    TaskEvaluationProviderCompletion,
    TaskEvaluationProviderSupportRequirements,
    project_prepared_task_evaluation_cases,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessResult,
)
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
    _release_matrix_fixture,
)
from test_expert_task_evaluation_preflight import (
    _CurrentAuthority,
    _coordinator,
    _current_observation,
    _expert_sources,
)
from test_expert_task_evaluation_reservation import _parent_prepared


class _Provider:
    def __init__(self, dispatch_key, *, supported=True, execute=None):
        self.dispatch_key = dispatch_key
        self.supported = supported
        self.execute = execute
        self.support_calls = []
        self.execution_calls = []
        self.cleanup_calls = []

    def require_supported_execution(self, requirements):
        self.support_calls.append(requirements)
        if not self.supported:
            raise TaskEvaluationExecutionError(
                "provider rejects this deterministic case authority"
            )

    def execute_leg(self, invocation):
        self.execution_calls.append(invocation)
        if self.execute is None:
            raise AssertionError("registry resolution must not execute a leg")
        return self.execute(invocation)

    def cleanup_interrupted(self, provider_handle):
        self.cleanup_calls.append(provider_handle)


def _bootstrap_prepared(tmp_path, monkeypatch):
    validation_store, snapshot, prepared_plan, _active_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    observation = _current_observation(prepared_plan)
    coordinator, _candidate_reader, source_base_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        source_base=None,
        current_authority=_CurrentAuthority((observation, observation)),
    )
    return coordinator.build(plan_reservation), source_base_provider


def _parent_prepared_with_additional_case(
    tmp_path,
    monkeypatch,
    *,
    source_fixture=None,
    released_source=None,
    source_adapter=None,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        add_active_case=True,
        source_fixture=source_fixture,
        released_source=released_source,
        source_adapter=source_adapter,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    _candidate, source_base = _expert_sources(
        prepared_plan,
        source_base_contents=(None if released_source is None else released_source[2]),
    )
    observation = _current_observation(prepared_plan)
    coordinator, *_providers = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        source_base=source_base,
        current_authority=_CurrentAuthority((observation, observation)),
    )
    return validation_store, snapshot, coordinator.build(plan_reservation)


def _reserve(validation_store, snapshot, prepared):
    return validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    ).reservation


def _allocation(reservation_snapshot, executable_case, leg_position=0):
    return TaskEvaluationInvocationAllocation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        evaluation_case_id=executable_case.evaluation_case_id,
        evaluation_leg_id=executable_case.legs[leg_position].authority.leg_id,
        invocation_nonce="a" * 32,
    )


def _process_result(tmp_path):
    trusted_root = tmp_path.resolve()
    return BoundedProcessResult(
        request=BoundedProcessRequest(
            argv=("true",),
            trusted_root=trusted_root,
            cwd=trusted_root,
            timeout_seconds=1,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1,
            stderr_byte_limit=1,
            environment={},
        ),
        outcome=BoundedProcessOutcome.COMPLETED,
        returncode=0,
        stdout=b"",
        stderr=b"",
        stdout_bytes_observed=0,
        stderr_bytes_observed=0,
        duration_seconds=0.5,
    )


def _completion(tmp_path, invocation, *, provider_handle_id=None):
    return TaskEvaluationProviderCompletion(
        provider_handle_id=(
            invocation.provider_handle.provider_handle_id
            if provider_handle_id is None
            else provider_handle_id
        ),
        process_result=_process_result(tmp_path),
        result_payload=b"result",
    )


def test_registry_erases_matrix_provenance_and_resolves_every_parent_case(
    tmp_path,
    monkeypatch,
):
    _store, _snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    executable_cases = project_prepared_task_evaluation_cases(prepared)
    provider = _Provider(executable_cases[0].provider_key)

    registry = TaskEvaluationExecutionProviderRegistry(prepared, (provider,))
    resolved = registry.resolve_all()

    assert len(resolved) == len(executable_cases) == len(prepared.cases)
    assert all(type(item) is ResolvedTaskEvaluationCase for item in resolved)
    assert tuple(item.executable_case for item in resolved) == executable_cases
    assert len(provider.support_calls) == len(executable_cases)
    assert all(
        type(requirements) is TaskEvaluationProviderSupportRequirements
        for requirements in provider.support_calls
    )
    assert provider.execution_calls == []
    assert all(
        type(case) is ExecutableTaskEvaluationCase
        and not hasattr(case, "provenance_binding_id")
        and not hasattr(case, "evaluation_cell_ids")
        and {leg.authority.kind for leg in case.legs}
        == {
            TaskEvaluationLegKind.SOURCE_BASE_CONTROL,
            TaskEvaluationLegKind.CANDIDATE,
        }
        for case in executable_cases
    )
    for executable_case, materialized_case in zip(
        executable_cases,
        prepared.cases,
        strict=True,
    ):
        signed_case = next(
            signed
            for signed in materialized_case.adapter.manifest.release_matrix_cases
            if signed.release_matrix_case_id
            == materialized_case.request_case.release_matrix_case_id
        )
        assert executable_case.task_context_binding == (
            signed_case.task_context_binding
        )
        assert executable_case.evaluation_fingerprints == (
            signed_case.evaluation_fingerprints
        )
        assert executable_case.adapter_runtime == materialized_case.adapter_runtime
        assert executable_case.starting_artifacts == (
            materialized_case.starting_artifacts
        )


def test_bootstrap_registry_resolves_only_candidate_legs(
    tmp_path,
    monkeypatch,
):
    prepared, source_base_provider = _bootstrap_prepared(tmp_path, monkeypatch)
    executable_cases = project_prepared_task_evaluation_cases(prepared)
    provider = _Provider(executable_cases[0].provider_key)

    resolved = TaskEvaluationExecutionProviderRegistry(
        prepared,
        (provider,),
    ).resolve_all()

    assert len(resolved) == len(prepared.cases)
    assert all(
        tuple(leg.authority.kind for leg in item.executable_case.legs)
        == (TaskEvaluationLegKind.CANDIDATE,)
        for item in resolved
    )
    assert source_base_provider.calls == []


def test_provider_key_excludes_case_mode_schedule_and_scientific_identity(
    tmp_path,
    monkeypatch,
):
    parent_root = tmp_path / "source_base"
    parent_root.mkdir()
    _store, _snapshot, parent_prepared, *_providers = _parent_prepared(
        parent_root,
        monkeypatch,
    )
    bootstrap_root = tmp_path / "bootstrap"
    bootstrap_root.mkdir()
    bootstrap_prepared, _parent_provider = _bootstrap_prepared(
        bootstrap_root,
        monkeypatch,
    )

    parent_case = project_prepared_task_evaluation_cases(parent_prepared)[0]
    bootstrap_case = project_prepared_task_evaluation_cases(bootstrap_prepared)[0]

    assert parent_case.provider_key == bootstrap_case.provider_key
    assert set(parent_case.compute_binding.leg_order) != set(
        bootstrap_case.compute_binding.leg_order
    )
    assert parent_case.evaluation_case_id != bootstrap_case.evaluation_case_id


def test_registry_rejects_missing_duplicate_and_mutated_provider_identity(
    tmp_path,
    monkeypatch,
):
    _store, _snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    executable_case = project_prepared_task_evaluation_cases(prepared)[0]
    key = executable_case.provider_key
    foreign_key = replace(key, execution_provider_version="unsupported_provider_v2")

    with pytest.raises(TaskEvaluationExecutionError, match="unsupported"):
        TaskEvaluationExecutionProviderRegistry(
            prepared,
            (_Provider(foreign_key),),
        )
    with pytest.raises(TaskEvaluationExecutionError, match="duplicated"):
        TaskEvaluationExecutionProviderRegistry(
            prepared,
            (_Provider(key), _Provider(key)),
        )
    with pytest.raises(TaskEvaluationExecutionError, match="exact required key set"):
        TaskEvaluationExecutionProviderRegistry(
            prepared,
            (_Provider(key), _Provider(foreign_key)),
        )

    provider = _Provider(key)
    registry = TaskEvaluationExecutionProviderRegistry(prepared, (provider,))
    provider.dispatch_key = foreign_key
    with pytest.raises(TaskEvaluationExecutionError, match="identity changed"):
        registry.resolve_all()


def test_deterministic_provider_incompatibility_fails_during_resolution(
    tmp_path,
    monkeypatch,
):
    _store, _snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    executable_case = project_prepared_task_evaluation_cases(prepared)[0]
    provider = _Provider(executable_case.provider_key, supported=False)

    with pytest.raises(TaskEvaluationExecutionError, match="deterministic"):
        TaskEvaluationExecutionProviderRegistry(prepared, (provider,))

    assert len(provider.support_calls) == 1
    assert provider.support_calls[0].dispatch_key == executable_case.provider_key
    assert provider.execution_calls == []


def test_support_check_receives_only_non_scientific_runtime_requirements(
    tmp_path,
    monkeypatch,
):
    _store, _snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    executable_case = project_prepared_task_evaluation_cases(prepared)[0]
    provider = _Provider(executable_case.provider_key)

    TaskEvaluationExecutionProviderRegistry(prepared, (provider,))

    assert {field.name for field in fields(provider.support_calls[0])} == {
        "dispatch_key",
        "runtime_contract",
        "task_evaluator_executable_path",
        "leg_wall_time_limit_seconds",
        "termination_grace_seconds",
        "cpu_millicore_limit",
        "memory_byte_limit",
        "shared_memory_byte_limit",
        "process_limit",
        "open_file_limit",
        "writable_inode_limit",
        "writable_storage_byte_limit",
        "output_entry_limit",
        "output_byte_limit",
        "stdout_byte_limit",
        "stderr_byte_limit",
        "accelerator_class_id",
        "accelerator_count",
    }
    requirements = provider.support_calls[0]
    for forbidden_name in (
        "evaluation_case_id",
        "task_context_binding",
        "evaluation_fingerprints",
        "starting_artifacts",
        "legs",
        "expert_source",
        "source_contents",
        "leg_order",
    ):
        assert not hasattr(requirements, forbidden_name)


def test_registry_is_bound_to_one_prepared_authority_and_has_no_spawn_surface(
    tmp_path,
    monkeypatch,
):
    parent_root = tmp_path / "source_base"
    parent_root.mkdir()
    _store, _snapshot, parent_prepared, *_providers = _parent_prepared(
        parent_root,
        monkeypatch,
    )
    bootstrap_root = tmp_path / "bootstrap"
    bootstrap_root.mkdir()
    bootstrap_prepared, _parent_provider = _bootstrap_prepared(
        bootstrap_root,
        monkeypatch,
    )
    executable_case = project_prepared_task_evaluation_cases(parent_prepared)[0]
    provider = _Provider(executable_case.provider_key)
    registry = TaskEvaluationExecutionProviderRegistry(
        parent_prepared,
        (provider,),
    )

    registry.require_exact_prepared_authority(parent_prepared)
    with pytest.raises(TaskEvaluationExecutionError, match="prepared authority"):
        registry.require_exact_prepared_authority(bootstrap_prepared)

    resolved = registry.resolve_all()[0]
    assert not hasattr(resolved, "_execute_leg")
    assert "executable_case" not in {
        field.name for field in fields(TaskEvaluationLegInvocation)
    }
    with pytest.raises(TypeError):
        TaskEvaluationLegInvocation()


def test_registry_resolves_repeated_leg_ids_by_exact_case(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _parent_prepared_with_additional_case(
        tmp_path,
        monkeypatch,
    )
    reservation_snapshot = _reserve(validation_store, snapshot, prepared)
    executable_cases = project_prepared_task_evaluation_cases(prepared)
    assert len(executable_cases) == 2
    repeated_leg_ids = {
        leg.authority.leg_id for leg in executable_cases[0].legs
    }.intersection(leg.authority.leg_id for leg in executable_cases[1].legs)
    assert repeated_leg_ids
    repeated_leg_id = next(iter(repeated_leg_ids))
    provider = _Provider(executable_cases[0].provider_key)
    registry = TaskEvaluationExecutionProviderRegistry(prepared, (provider,))
    allocation = TaskEvaluationInvocationAllocation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        evaluation_case_id=executable_cases[1].evaluation_case_id,
        evaluation_leg_id=repeated_leg_id,
        invocation_nonce="b" * 32,
    )

    resolved_case = registry._resolved_case_for_allocation(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
    )

    assert resolved_case.executable_case == executable_cases[1]
    assert resolved_case.executable_case.evaluation_case_id == (
        allocation.evaluation_case_id
    )


def test_journal_execution_rejects_a_foreign_registry_resolution_before_call(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    reservation_snapshot = _reserve(validation_store, snapshot, prepared)
    executable_case = project_prepared_task_evaluation_cases(prepared)[0]
    allocation = _allocation(reservation_snapshot, executable_case)
    first_provider = _Provider(executable_case.provider_key)
    second_provider = _Provider(executable_case.provider_key)
    first_registry = TaskEvaluationExecutionProviderRegistry(
        prepared,
        (first_provider,),
    )
    second_registry = TaskEvaluationExecutionProviderRegistry(
        prepared,
        (second_provider,),
    )
    foreign_resolution = second_registry._resolved_case_for_allocation(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
    )

    with pytest.raises(TaskEvaluationExecutionError, match="foreign resolution"):
        first_registry._execute_journal_leg(
            prepared_request=prepared,
            reservation_snapshot=reservation_snapshot,
            resolved_case=foreign_resolution,
            invocation_allocation=allocation,
        )

    assert first_provider.execution_calls == []
    assert second_provider.execution_calls == []


@pytest.mark.parametrize("drift_phase", ("before", "after"))
def test_journal_execution_rejects_provider_identity_drift_without_retry(
    tmp_path,
    monkeypatch,
    drift_phase,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    reservation_snapshot = _reserve(validation_store, snapshot, prepared)
    executable_case = project_prepared_task_evaluation_cases(prepared)[0]
    foreign_key = replace(
        executable_case.provider_key,
        execution_provider_version="drifted_provider_v2",
    )
    provider = _Provider(executable_case.provider_key)
    registry = TaskEvaluationExecutionProviderRegistry(prepared, (provider,))
    allocation = _allocation(reservation_snapshot, executable_case)
    resolved_case = registry._resolved_case_for_allocation(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
    )
    if drift_phase == "before":
        provider.dispatch_key = foreign_key
    else:
        def drift_after_call(invocation):
            provider.dispatch_key = foreign_key
            return _completion(tmp_path, invocation)

        provider.execute = drift_after_call

    with pytest.raises(TaskEvaluationExecutionError, match="identity changed"):
        registry._execute_journal_leg(
            prepared_request=prepared,
            reservation_snapshot=reservation_snapshot,
            resolved_case=resolved_case,
            invocation_allocation=allocation,
        )

    assert len(provider.execution_calls) == (0 if drift_phase == "before" else 1)


@pytest.mark.parametrize("wrong_completion_kind", ("untyped", "foreign_handle"))
def test_journal_execution_rejects_wrong_completion_without_retry(
    tmp_path,
    monkeypatch,
    wrong_completion_kind,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    reservation_snapshot = _reserve(validation_store, snapshot, prepared)
    executable_case = project_prepared_task_evaluation_cases(prepared)[0]
    provider = _Provider(executable_case.provider_key)
    registry = TaskEvaluationExecutionProviderRegistry(prepared, (provider,))
    allocation = _allocation(reservation_snapshot, executable_case)
    resolved_case = registry._resolved_case_for_allocation(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
    )

    def wrong_completion(invocation):
        if wrong_completion_kind == "untyped":
            return object()
        return _completion(
            tmp_path,
            invocation,
            provider_handle_id=content_id(
                "task-evaluation-provider-execution-handle",
                {"foreign": True},
            ),
        )

    provider.execute = wrong_completion

    with pytest.raises(TaskEvaluationExecutionError, match="foreign completion"):
        registry._execute_journal_leg(
            prepared_request=prepared,
            reservation_snapshot=reservation_snapshot,
            resolved_case=resolved_case,
            invocation_allocation=allocation,
        )

    assert len(provider.execution_calls) == 1


def test_journal_execution_constructs_one_private_invocation_and_calls_once(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    reservation_snapshot = _reserve(validation_store, snapshot, prepared)
    executable_case = project_prepared_task_evaluation_cases(prepared)[0]
    provider = _Provider(executable_case.provider_key)
    registry = TaskEvaluationExecutionProviderRegistry(prepared, (provider,))
    allocation = _allocation(reservation_snapshot, executable_case)
    resolved_case = registry._resolved_case_for_allocation(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
    )
    provider.execute = lambda invocation: _completion(tmp_path, invocation)

    completion = registry._execute_journal_leg(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        resolved_case=resolved_case,
        invocation_allocation=allocation,
    )

    assert type(completion) is TaskEvaluationProviderCompletion
    assert len(provider.execution_calls) == 1
    invocation = provider.execution_calls[0]
    assert type(invocation) is TaskEvaluationLegInvocation
    assert invocation.invocation_allocation == allocation
    assert invocation.selected_leg is resolved_case.executable_case.legs[0]
    assert completion.provider_handle_id == (
        invocation.provider_handle.provider_handle_id
    )
    assert not hasattr(resolved_case, "execute_leg")
    assert not hasattr(resolved_case, "_execute_leg")


def test_provider_key_requires_exact_complete_identity():
    with pytest.raises(TaskEvaluationExecutionError, match="digest"):
        TaskEvaluationExecutionProviderKey(
            execution_protocol_version="task_execution_v1",
            execution_provider_id="provider",
            execution_provider_version="provider_v1",
            execution_provider_settings_digest="sha256:not-a-digest",
            sandbox_policy_version="sandbox_v1",
            task_adapter_runtime_protocol_version="adapter_runtime_v1",
            task_evaluator_protocol_version="task_evaluator_v1",
        )
