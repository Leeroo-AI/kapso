from dataclasses import fields, replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.expert.task_evaluation_authority_projection import (
    build_task_evaluation_spawn_authority_fence,
    task_evaluation_adapter_trust_observations,
    task_evaluation_spawn_security_subject_ids,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationInvocationAllocation,
    TaskEvaluationLegKind,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    project_prepared_task_evaluation_cases,
    task_evaluation_provider_execution_handle,
)
from kapso.cross_run.expert.task_evaluation_execution_journal import (
    TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION,
    TaskEvaluationExecutionJournalError,
    TaskEvaluationExecutionJournalEvent,
    TaskEvaluationExecutionJournalEventKind,
    TaskEvaluationExecutionPrefixState,
    TaskEvaluationProcessObservation,
    task_evaluation_execution_schedule,
    validate_task_evaluation_execution_prefix,
)
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalResultBlob,
)
from kapso.cross_run.expert.task_evaluation_protocol import (
    build_task_evaluation_evaluator_request,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TaskEvaluatorFingerprintResult,
    TaskEvaluatorResult,
)
from kapso.cross_run.process import BoundedProcessOutcome
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
    _release_matrix_fixture,
)
from test_expert_task_evaluation_authority import _denylist
from test_expert_task_evaluation_preflight import (
    _CurrentAuthority,
    _coordinator,
    _current_observation,
    _expert_sources,
)
from test_expert_task_evaluation_reservation import _parent_prepared


def _reserve(validation_store, snapshot, prepared):
    return validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    ).reservation


def _parent_journal_authority(tmp_path, monkeypatch):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    return prepared, _reserve(validation_store, snapshot, prepared)


def _bootstrap_journal_authority(tmp_path, monkeypatch):
    validation_store, snapshot, prepared_plan, _active_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    observation = _current_observation(prepared_plan)
    coordinator, _candidate_reader, _parent_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        source_base=None,
        current_authority=_CurrentAuthority((observation, observation)),
    )
    prepared = coordinator.build(plan_reservation)
    return prepared, _reserve(validation_store, snapshot, prepared)


def _multi_case_parent_journal_authority(tmp_path, monkeypatch):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        add_active_case=True,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    _candidate, source_base = _expert_sources(prepared_plan)
    observation = _current_observation(prepared_plan)
    coordinator, _candidate_reader, _parent_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        source_base=source_base,
        current_authority=_CurrentAuthority((observation, observation)),
    )
    prepared = coordinator.build(plan_reservation)
    return prepared, _reserve(validation_store, snapshot, prepared)


def _remint(record, **changes):
    values = {field.name: getattr(record, field.name) for field in fields(record)}
    values.pop(record.IDENTITY_FIELD)
    values.update(changes)
    return type(record).mint(**values)


def _event(
    *,
    event_number,
    predecessor_event_id,
    event_kind,
    request_id,
    allocation,
    **payload,
):
    values = {
        "schema_version": TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION,
        "event_number": event_number,
        "predecessor_event_id": predecessor_event_id,
        "event_kind": event_kind,
        "request_id": request_id,
        "invocation_allocation": allocation,
        "spawn_authority_fence": None,
        "execution_provider_key": None,
        "provider_execution_handle": None,
        "task_evaluator_request": None,
        "aggregate_tolerance": None,
        "process_observation": None,
        "result_blob": None,
        "task_evaluator_result": None,
    }
    values.update(payload)
    return TaskEvaluationExecutionJournalEvent.mint(**values)


def _result(task_evaluator_request):
    return TaskEvaluatorResult(
        protocol_version=task_evaluator_request.protocol_version,
        opaque_invocation_id=task_evaluator_request.opaque_invocation_id,
        fingerprint_results=tuple(
            TaskEvaluatorFingerprintResult(
                evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
                aggregate_value=1.0,
                replicate_values={
                    replicate_id: 1.0
                    for replicate_id in fingerprint.seed_or_replicate_ids
                },
            )
            for fingerprint in task_evaluator_request.evaluation_fingerprints
        ),
    )


def _leg_events(
    prepared,
    reservation_snapshot,
    *,
    schedule_position,
    predecessor_event_id,
    invocation_nonce,
    result_payload=None,
):
    case_id, leg_id = task_evaluation_execution_schedule(
        reservation_snapshot,
        prepared,
    )[schedule_position]
    allocation = TaskEvaluationInvocationAllocation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        evaluation_case_id=case_id,
        evaluation_leg_id=leg_id,
        invocation_nonce=invocation_nonce,
    )
    adapter_observations = task_evaluation_adapter_trust_observations(prepared)
    security_subject_ids = task_evaluation_spawn_security_subject_ids(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
        stable_current_release_observation=prepared.current_release_observation,
        task_adapter_trust_observations=adapter_observations,
    )
    fence = build_task_evaluation_spawn_authority_fence(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
        stable_current_release_observation=prepared.current_release_observation,
        task_adapter_trust_observations=adapter_observations,
        security_denylist_observation=_denylist(prepared, security_subject_ids),
    )
    executable_case = next(
        case
        for case in project_prepared_task_evaluation_cases(prepared)
        if case.evaluation_case_id == case_id
    )
    task_evaluator_request = build_task_evaluation_evaluator_request(
        prepared,
        reservation_snapshot,
        allocation,
    )
    first_event_number = schedule_position * 4 + 1
    allocated = _event(
        event_number=first_event_number,
        predecessor_event_id=predecessor_event_id,
        event_kind=(TaskEvaluationExecutionJournalEventKind.INVOCATION_ALLOCATED),
        request_id=reservation_snapshot.request.request_id,
        allocation=allocation,
    )
    spawned = _event(
        event_number=first_event_number + 1,
        predecessor_event_id=allocated.event_id,
        event_kind=TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED,
        request_id=reservation_snapshot.request.request_id,
        allocation=allocation,
        spawn_authority_fence=fence,
        execution_provider_key=executable_case.provider_key,
        provider_execution_handle=task_evaluation_provider_execution_handle(
            executable_case.provider_key,
            allocation,
        ),
        task_evaluator_request=task_evaluator_request,
        aggregate_tolerance=(
            prepared.plan_join.settings.policy.task_evaluation_aggregate_tolerance
        ),
    )
    if result_payload is None:
        result_payload = _result(task_evaluator_request).to_json_bytes()
    result_blob = ExecutionJournalResultBlob(
        digest=tree_or_blob_digest(result_payload),
        size=len(result_payload),
    )
    received = _event(
        event_number=first_event_number + 2,
        predecessor_event_id=spawned.event_id,
        event_kind=TaskEvaluationExecutionJournalEventKind.RESULT_RECEIVED,
        request_id=reservation_snapshot.request.request_id,
        allocation=allocation,
        process_observation=TaskEvaluationProcessObservation(
            outcome=BoundedProcessOutcome.COMPLETED,
            returncode=0,
            stdout_bytes_observed=0,
            stderr_bytes_observed=0,
            duration_seconds=1.0,
        ),
        result_blob=result_blob,
    )
    return allocated, spawned, received, result_payload


def _first_leg_events(prepared, reservation_snapshot, *, result_payload=None):
    return _leg_events(
        prepared,
        reservation_snapshot,
        schedule_position=0,
        predecessor_event_id=None,
        invocation_nonce="1" * 32,
        result_payload=result_payload,
    )


def _accepted_event(events, result_payload):
    allocated, spawned, received = events
    result = TaskEvaluatorResult.from_json_bytes(result_payload)
    return _event(
        event_number=received.event_number + 1,
        predecessor_event_id=received.event_id,
        event_kind=TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED,
        request_id=allocated.request_id,
        allocation=allocated.invocation_allocation,
        task_evaluator_result=result,
    )


def _payloads(received, result_payload):
    return ((received.result_blob, result_payload),)


def test_schedule_uses_canonical_case_order_and_each_semantic_leg_order(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "source_base").mkdir()
    parent_prepared, parent_reservation = _parent_journal_authority(
        tmp_path / "source_base",
        monkeypatch,
    )
    (tmp_path / "bootstrap").mkdir()
    bootstrap_prepared, bootstrap_reservation = _bootstrap_journal_authority(
        tmp_path / "bootstrap",
        monkeypatch,
    )

    parent_schedule = task_evaluation_execution_schedule(
        parent_reservation,
        parent_prepared,
    )
    expected_parent_schedule = tuple(
        (case.evaluation_case_id, legs_by_kind[leg_kind])
        for case in parent_reservation.request.cases
        for legs_by_kind in ({leg.kind: leg.leg_id for leg in case.legs},)
        for leg_kind in case.compute_binding.leg_order
    )
    bootstrap_schedule = task_evaluation_execution_schedule(
        bootstrap_reservation,
        bootstrap_prepared,
    )

    assert parent_schedule == expected_parent_schedule
    assert all(
        len(
            tuple(
                item for item in parent_schedule if item[0] == case.evaluation_case_id
            )
        )
        == 2
        for case in parent_reservation.request.cases
    )
    assert bootstrap_schedule == tuple(
        (case.evaluation_case_id, case.legs[0].leg_id)
        for case in bootstrap_reservation.request.cases
    )
    assert all(
        case.compute_binding.leg_order == (TaskEvaluationLegKind.CANDIDATE,)
        for case in bootstrap_reservation.request.cases
    )


def test_full_four_event_block_reduces_from_exact_durable_authority(
    tmp_path,
    monkeypatch,
):
    prepared, reservation_snapshot = _parent_journal_authority(
        tmp_path,
        monkeypatch,
    )
    allocated, spawned, received, result_payload = _first_leg_events(
        prepared,
        reservation_snapshot,
    )
    accepted = _accepted_event((allocated, spawned, received), result_payload)
    events = (allocated, spawned, received, accepted)

    prefix = validate_task_evaluation_execution_prefix(
        reservation_snapshot=reservation_snapshot,
        prepared_request=prepared,
        events=events,
        result_payloads=_payloads(received, result_payload),
    )

    assert prefix.events == events
    assert prefix.schedule == task_evaluation_execution_schedule(
        reservation_snapshot,
        prepared,
    )
    assert prefix.complete is (len(prefix.schedule) == 1)
    assert (
        TaskEvaluationExecutionJournalEvent.from_json_bytes(spawned.to_json_bytes())
        == spawned
    )
    assert (
        TaskEvaluationExecutionPrefixState(
            events=events,
            schedule=prefix.schedule,
        )
        == prefix
    )


def test_multi_case_prefix_keeps_repeated_leg_ids_case_scoped(
    tmp_path,
    monkeypatch,
):
    prepared, reservation_snapshot = _multi_case_parent_journal_authority(
        tmp_path,
        monkeypatch,
    )
    schedule = task_evaluation_execution_schedule(reservation_snapshot, prepared)
    candidate_pairs = tuple(
        (case.evaluation_case_id, leg.leg_id)
        for case in reservation_snapshot.request.cases
        for leg in case.legs
        if leg.kind is TaskEvaluationLegKind.CANDIDATE
    )

    assert len(candidate_pairs) >= 2
    assert len({case_id for case_id, _leg_id in candidate_pairs}) == len(
        candidate_pairs
    )
    assert len({leg_id for _case_id, leg_id in candidate_pairs}) == 1
    assert set(candidate_pairs).issubset(schedule)

    target_pair = candidate_pairs[1]
    target_position = schedule.index(target_pair)
    events = []
    result_payloads = []
    predecessor_event_id = None
    for schedule_position in range(target_position + 1):
        allocated, spawned, received, result_payload = _leg_events(
            prepared,
            reservation_snapshot,
            schedule_position=schedule_position,
            predecessor_event_id=predecessor_event_id,
            invocation_nonce=f"{schedule_position + 1:032x}",
        )
        accepted = _accepted_event((allocated, spawned, received), result_payload)
        events.extend((allocated, spawned, received, accepted))
        result_payloads.append((received.result_blob, result_payload))
        predecessor_event_id = accepted.event_id

    prefix = validate_task_evaluation_execution_prefix(
        reservation_snapshot=reservation_snapshot,
        prepared_request=prepared,
        events=tuple(events),
        result_payloads=tuple(sorted(result_payloads, key=lambda item: item[0].digest)),
    )

    target_allocation = prefix.events[target_position * 4].invocation_allocation
    assert (
        target_allocation.evaluation_case_id,
        target_allocation.evaluation_leg_id,
    ) == target_pair


def test_prefix_rejects_schedule_substitution_reused_nonce_and_spawn_drift(
    tmp_path,
    monkeypatch,
):
    prepared, reservation_snapshot = _parent_journal_authority(
        tmp_path,
        monkeypatch,
    )
    allocated, spawned, received, result_payload = _first_leg_events(
        prepared,
        reservation_snapshot,
    )
    accepted = _accepted_event((allocated, spawned, received), result_payload)
    schedule = task_evaluation_execution_schedule(reservation_snapshot, prepared)
    second_case_id, second_leg_id = schedule[1]
    substituted_allocation = replace(
        allocated.invocation_allocation,
        evaluation_case_id=second_case_id,
        evaluation_leg_id=second_leg_id,
    )
    substituted_first = _remint(
        allocated,
        invocation_allocation=substituted_allocation,
    )
    with pytest.raises(TaskEvaluationExecutionJournalError, match="schedule prefix"):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(substituted_first,),
            result_payloads=(),
        )

    reused_allocation = TaskEvaluationInvocationAllocation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        evaluation_case_id=second_case_id,
        evaluation_leg_id=second_leg_id,
        invocation_nonce=allocated.invocation_allocation.invocation_nonce,
    )
    reused = _event(
        event_number=5,
        predecessor_event_id=accepted.event_id,
        event_kind=(TaskEvaluationExecutionJournalEventKind.INVOCATION_ALLOCATED),
        request_id=reservation_snapshot.request.request_id,
        allocation=reused_allocation,
    )
    with pytest.raises(TaskEvaluationExecutionJournalError, match="reuses"):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(allocated, spawned, received, accepted, reused),
            result_payloads=_payloads(received, result_payload),
        )

    drifted_key = replace(
        spawned.execution_provider_key,
        execution_provider_version="substituted_provider_v2",
    )
    drifted_spawn = _remint(
        spawned,
        execution_provider_key=drifted_key,
        provider_execution_handle=task_evaluation_provider_execution_handle(
            drifted_key,
            spawned.invocation_allocation,
        ),
    )
    with pytest.raises(TaskEvaluationExecutionJournalError, match="prepared authority"):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(allocated, drifted_spawn),
            result_payloads=(),
        )


def test_received_malformed_result_is_reopenable_but_cannot_be_accepted(
    tmp_path,
    monkeypatch,
):
    prepared, reservation_snapshot = _parent_journal_authority(
        tmp_path,
        monkeypatch,
    )
    malformed_payload = b"not canonical evaluator JSON"
    allocated, spawned, received, _payload = _first_leg_events(
        prepared,
        reservation_snapshot,
        result_payload=malformed_payload,
    )

    reopened = validate_task_evaluation_execution_prefix(
        reservation_snapshot=reservation_snapshot,
        prepared_request=prepared,
        events=(allocated, spawned, received),
        result_payloads=_payloads(received, malformed_payload),
    )

    assert reopened.events[-1].event_kind is (
        TaskEvaluationExecutionJournalEventKind.RESULT_RECEIVED
    )
    valid_result = _result(spawned.task_evaluator_request)
    invalid_acceptance = _event(
        event_number=4,
        predecessor_event_id=received.event_id,
        event_kind=TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED,
        request_id=allocated.request_id,
        allocation=allocated.invocation_allocation,
        task_evaluator_result=valid_result,
    )
    with pytest.raises(ValueError):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(allocated, spawned, received, invalid_acceptance),
            result_payloads=_payloads(received, malformed_payload),
        )


def test_prefix_requires_exact_blob_set_and_persisted_compute_bounds(
    tmp_path,
    monkeypatch,
):
    prepared, reservation_snapshot = _parent_journal_authority(
        tmp_path,
        monkeypatch,
    )
    allocated, spawned, received, result_payload = _first_leg_events(
        prepared,
        reservation_snapshot,
    )

    result_byte_limit = (
        prepared.plan_join.settings.policy.task_evaluation_result_byte_limit
    )
    first_case = reservation_snapshot.request.cases[0]
    assert result_byte_limit < first_case.compute_binding.output_byte_limit
    oversized_payload = b"x" * (result_byte_limit + 1)
    (
        oversized_allocated,
        oversized_spawned,
        oversized_received,
        _oversized_payload,
    ) = _first_leg_events(
        prepared,
        reservation_snapshot,
        result_payload=oversized_payload,
    )
    with pytest.raises(TaskEvaluationExecutionJournalError, match="configured bound"):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(oversized_allocated, oversized_spawned, oversized_received),
            result_payloads=_payloads(oversized_received, oversized_payload),
        )

    with pytest.raises(TaskEvaluationExecutionJournalError, match="referenced blobs"):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(allocated, spawned, received),
            result_payloads=(),
        )
    extra_payload = b"orphan"
    extra_blob = ExecutionJournalResultBlob(
        digest=tree_or_blob_digest(extra_payload),
        size=len(extra_payload),
    )
    payloads = tuple(
        sorted(
            (
                (received.result_blob, result_payload),
                (extra_blob, extra_payload),
            ),
            key=lambda item: item[0].digest,
        )
    )
    with pytest.raises(TaskEvaluationExecutionJournalError, match="referenced blobs"):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(allocated, spawned, received),
            result_payloads=payloads,
        )

    oversized_observation = replace(
        received.process_observation,
        stdout_bytes_observed=first_case.compute_binding.stdout_byte_limit + 1,
    )
    oversized_received = _remint(
        received,
        process_observation=oversized_observation,
    )
    with pytest.raises(TaskEvaluationExecutionJournalError, match="compute bounds"):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(allocated, spawned, oversized_received),
            result_payloads=_payloads(received, result_payload),
        )

    foreign_predecessor = content_id(
        "task-evaluation-execution-journal-event",
        {"foreign": True},
    )
    replaced_spawn = _remint(
        spawned,
        predecessor_event_id=foreign_predecessor,
    )
    with pytest.raises(TaskEvaluationExecutionJournalError, match="schedule prefix"):
        validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared,
            events=(allocated, replaced_spawn),
            result_payloads=(),
        )
