from __future__ import annotations

import sys
from dataclasses import replace
from multiprocessing import get_context

import pytest

from kapso.cross_run.canonical import (
    content_id,
)
from kapso.cross_run.contracts import (
    ExpertSourceReplayExecutionLegKind,
    ObjectiveDirection,
)
from kapso.cross_run.expert.replay_comparison import (
    build_expert_source_replay_paired_comparison_receipt,
)
from kapso.cross_run.expert.replay_comparison_contracts import (
    ExpertSourceReplayFingerprintComparison,
    ExpertSourceReplayPairedComparisonReceipt,
)
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayProviderCompletion,
    expert_source_replay_execution_provider_key,
)
from kapso.cross_run.expert.replay_execution_store import (
    CompletedExpertSourceReplayExecution,
    ExpertSourceReplayExecutionStore,
    ExpertSourceReplayExecutionStoreError,
    SourceReplayExecutionJournalEventKind,
    source_replay_execution_schedule,
)
from kapso.cross_run.expert.replay_protocol_contracts import (
    ExpertSourceReplayProtocolError,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TaskEvaluatorFingerprintResult,
    TaskEvaluatorProtocolError,
    TaskEvaluatorResult,
    stable_arithmetic_mean,
)
from test_expert_replay_execution_store import (
    _MatchedLegProvider,
    _commit_spawn,
    _journal_fixture,
    _process_result,
    _remint,
)
from test_expert_source_replay_request import _prepared, _request_fixture


def _reject_completed_execution_after_fork(
    completed,
    execution_store,
    reservation,
    prepared_request,
    result_queue,
):
    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="authority"):
        completed.require_exact(execution_store, reservation, prepared_request)
    result_queue.put("rejected")


class _SemanticMatchedLegProvider(_MatchedLegProvider):
    def __init__(self, *args, aggregate_by_leg_kind, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregate_by_leg_kind = aggregate_by_leg_kind

    def execute_leg(self, invocation):
        self.invocations.append(invocation)
        request_case = invocation.materialized_case.request_case
        leg_kind = (
            ExpertSourceReplayExecutionLegKind.CONTROL_PARENT
            if invocation.invocation_allocation.execution_leg_id
            == request_case.control_leg.execution_leg_id
            else ExpertSourceReplayExecutionLegKind.CANDIDATE
        )
        aggregate = self.aggregate_by_leg_kind[leg_kind]
        result = TaskEvaluatorResult(
            protocol_version=invocation.task_evaluator_request.protocol_version,
            opaque_invocation_id=(
                invocation.task_evaluator_request.opaque_invocation_id
            ),
            fingerprint_results=tuple(
                TaskEvaluatorFingerprintResult(
                    evaluation_fingerprint_id=(fingerprint.evaluation_fingerprint_id),
                    aggregate_value=aggregate,
                    replicate_values={
                        replicate_id: aggregate
                        for replicate_id in fingerprint.seed_or_replicate_ids
                    },
                )
                for fingerprint in (
                    invocation.task_evaluator_request.evaluation_fingerprints
                )
            ),
        )
        return ExpertSourceReplayProviderCompletion(
            provider_handle_id=invocation.provider_handle.provider_handle_id,
            process_result=_process_result(
                self.trusted_root,
                compute_binding=request_case.compute_binding,
            ),
            result_payload=result.to_json_bytes(),
        )


def _provider(fixture, prepared, allocation, aggregate_by_leg_kind):
    materialized_case = next(
        case
        for case in prepared.cases
        if case.request_case.execution_case_id == allocation.execution_case_id
    )
    return _SemanticMatchedLegProvider(
        fixture.validation_store.root,
        expert_source_replay_execution_provider_key(materialized_case),
        aggregate_by_leg_kind=aggregate_by_leg_kind,
    )


def _complete_execution(
    tmp_path,
    validation_settings=None,
    aggregate_by_leg_kind=None,
    candidate_first_execution=False,
    contract_records=None,
    source_adapter=None,
    evaluation_evidence=None,
):
    if (
        validation_settings is None
        and not candidate_first_execution
        and contract_records is None
        and source_adapter is None
        and evaluation_evidence is None
    ):
        fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    else:
        fixture = _request_fixture(
            tmp_path,
            validation_settings=validation_settings,
            candidate_first_execution=candidate_first_execution,
            contract_records=contract_records,
            source_adapter=source_adapter,
            evaluation_evidence=evaluation_evidence,
        )
        prepared = _prepared(fixture)
        snapshot = fixture.validation_store.snapshot(prepared.request.candidate_id)
        assert snapshot is not None
        committed = fixture.validation_store.reserve_source_replay(
            expected_transition_id=snapshot.transition.transition_id,
            prepared_request=prepared,
        )
        reservation = committed.reservation
        store = ExpertSourceReplayExecutionStore(
            (fixture.validation_store.root / "source-replay-executions").resolve(),
            fixture.validation_store.root,
            prepared.settings.policy,
        )
    selected_aggregates = (
        {
            ExpertSourceReplayExecutionLegKind.CONTROL_PARENT: 0.7,
            ExpertSourceReplayExecutionLegKind.CANDIDATE: 0.8,
        }
        if aggregate_by_leg_kind is None
        else aggregate_by_leg_kind
    )
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        schedule = source_replay_execution_schedule(reservation, prepared.request)
        for _case_id, _leg_id in schedule:
            permit = session.allocate_expected_leg()
            allocation = permit.require_current_allocation(store)
            execution, _ = _commit_spawn(
                fixture,
                prepared,
                reservation,
                store,
                permit,
                _provider(
                    fixture,
                    prepared,
                    allocation,
                    selected_aggregates,
                ),
            )
            session.record_result_received(execution.execute())
            session.accept_received_result()
        completed = session.completed_execution()
    return fixture, prepared, reservation, store, completed


def test_completed_journal_builds_one_deterministic_factual_receipt(tmp_path):
    _, prepared, reservation, store, completed = _complete_execution(tmp_path)
    if prepared.request.cases[0].compute_binding.leg_order[0] is not (
        ExpertSourceReplayExecutionLegKind.CANDIDATE
    ):
        candidate_first_root = tmp_path / "candidate-first"
        candidate_first_root.mkdir()
        _, prepared, reservation, store, completed = _complete_execution(
            candidate_first_root,
            candidate_first_execution=True,
        )
    request_case = prepared.request.cases[0]
    assert request_case.compute_binding.leg_order[0] is (
        ExpertSourceReplayExecutionLegKind.CANDIDATE
    )

    receipt = build_expert_source_replay_paired_comparison_receipt(
        completed_execution=completed,
        execution_store=store,
        reservation=reservation,
        prepared_request=prepared,
    )
    case_comparison = receipt.case_comparisons[0]
    fingerprint_comparison = case_comparison.fingerprint_comparisons[0]
    accepted_events = tuple(
        event
        for event in completed.events
        if event.event_kind is SourceReplayExecutionJournalEventKind.RESULT_ACCEPTED
    )
    accepted_by_leg = {event.execution_leg_id: event for event in accepted_events}
    legs_by_kind = {
        request_case.control_leg.kind: request_case.control_leg.execution_leg_id,
        request_case.candidate_leg.kind: request_case.candidate_leg.execution_leg_id,
    }
    expected_dependencies = {
        reservation.reservation_id,
        *reservation.exact_dependency_ids,
        prepared.request.execution_request_id,
        *prepared.request.exact_dependency_ids,
        *(event.event_id for event in completed.events),
    }

    assert fingerprint_comparison.control_result.aggregate_value == 0.7
    assert fingerprint_comparison.candidate_result.aggregate_value == 0.8
    assert fingerprint_comparison.aggregate_raw_delta == 0.8 - 0.7
    assert fingerprint_comparison.aggregate_direction_aligned_delta == 0.8 - 0.7
    assert fingerprint_comparison.aggregate_normalized_effect == 0.8 - 0.7
    assert (
        accepted_events[0].execution_leg_id
        == legs_by_kind[request_case.compute_binding.leg_order[0]]
    )
    assert (
        case_comparison.control_result_accepted_event_id
        == accepted_by_leg[request_case.control_leg.execution_leg_id].event_id
    )
    assert (
        case_comparison.candidate_result_accepted_event_id
        == accepted_by_leg[request_case.candidate_leg.execution_leg_id].event_id
    )
    assert set(receipt.exact_dependency_ids) == expected_dependencies
    assert receipt.reservation_dependency_ids == reservation.exact_dependency_ids
    assert receipt.request_dependency_ids == prepared.request.exact_dependency_ids
    assert receipt.execution_journal_event_ids == tuple(
        event.event_id for event in completed.events
    )
    assert (
        ExpertSourceReplayPairedComparisonReceipt.from_json_bytes(
            receipt.to_json_bytes()
        )
        == receipt
    )

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as reopened:
        rebuilt = build_expert_source_replay_paired_comparison_receipt(
            completed_execution=reopened.completed_execution(),
            execution_store=store,
            reservation=reservation,
            prepared_request=prepared,
        )
    assert rebuilt == receipt
    assert rebuilt.to_json_bytes() == receipt.to_json_bytes()


def test_receipt_dependency_closure_rejects_malformed_omitted_and_extra_ids(tmp_path):
    _, prepared, reservation, store, completed = _complete_execution(tmp_path)
    receipt = build_expert_source_replay_paired_comparison_receipt(
        completed_execution=completed,
        execution_store=store,
        reservation=reservation,
        prepared_request=prepared,
    )

    with pytest.raises(ValueError, match="form"):
        _remint(
            receipt,
            request_dependency_ids=tuple(
                sorted(
                    (
                        *receipt.request_dependency_ids,
                        "not-a-content-id",
                    )
                )
            ),
        )
    with pytest.raises(ExpertSourceReplayProtocolError, match="omit"):
        _remint(
            receipt,
            reservation_dependency_ids=(reservation.authorization_state_id,),
        )
    with pytest.raises(ExpertSourceReplayProtocolError, match="not exact"):
        _remint(
            receipt,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        *receipt.exact_dependency_ids,
                        content_id("unexpected-receipt-dependency", {"extra": True}),
                    }
                )
            ),
        )


@pytest.mark.parametrize("event_count", (0, 4, 5, 6, 7))
def test_incomplete_journal_cannot_mint_completed_execution(tmp_path, event_count):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    aggregate_by_leg_kind = {
        ExpertSourceReplayExecutionLegKind.CONTROL_PARENT: 0.7,
        ExpertSourceReplayExecutionLegKind.CANDIDATE: 0.8,
    }
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        pending_execution = None
        while len(session.events) < event_count:
            phase = len(session.events) % 4
            if phase == 0:
                session.allocate_expected_leg()
            elif phase == 1:
                permit = session.allocate_expected_leg()
                allocation = permit.require_current_allocation(store)
                execution, _ = _commit_spawn(
                    fixture,
                    prepared,
                    reservation,
                    store,
                    permit,
                    _provider(
                        fixture,
                        prepared,
                        allocation,
                        aggregate_by_leg_kind,
                    ),
                )
                pending_execution = execution
            elif phase == 2:
                session.record_result_received(pending_execution.execute())
            else:
                session.accept_received_result()
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="incomplete",
        ):
            session.completed_execution()


def test_completed_execution_is_sealed_immutable_and_store_bound(tmp_path):
    fixture, prepared, reservation, store, completed = _complete_execution(tmp_path)
    with pytest.raises(
        ExpertSourceReplayExecutionStoreError,
        match="not journal sealed",
    ):
        CompletedExpertSourceReplayExecution(
            object(),
            store,
            reservation,
            prepared,
            completed.events,
        )
    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="immutable"):
        completed.events = ()
    foreign_store = type(store)(
        (fixture.validation_store.root / "foreign-source-replay-executions").resolve(),
        fixture.validation_store.root,
        prepared.settings.policy,
    )
    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="authority"):
        completed.require_exact(foreign_store, reservation, prepared)

    process_context = get_context("fork")
    result_queue = process_context.Queue()
    child = process_context.Process(
        target=_reject_completed_execution_after_fork,
        args=(completed, store, reservation, prepared, result_queue),
    )
    child.start()
    child.join()
    assert child.exitcode == 0
    assert result_queue.get() == "rejected"
    result_queue.close()
    result_queue.join_thread()


def test_comparison_aligns_minimize_direction_and_uses_declared_scale(tmp_path):
    _, prepared, _, _ = _journal_fixture(tmp_path)
    materialized_case = prepared.cases[0]
    terminal_attempt = materialized_case.episode.attempts[
        materialized_case.episode.terminal_attempt_revision
    ]
    fingerprint = _remint(
        terminal_attempt.evaluation_fingerprints[0],
        objective_direction=ObjectiveDirection.MINIMIZE,
    )
    binding = replace(
        materialized_case.task_adapter.manifest.task_evaluator.metric_comparison_bindings[
            0
        ],
        objective_direction=ObjectiveDirection.MINIMIZE,
        comparison_scale=2.0,
    )
    control = TaskEvaluatorFingerprintResult(
        evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
        aggregate_value=1.0,
        replicate_values={"seed-1": 1.0},
    )
    candidate = TaskEvaluatorFingerprintResult(
        evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
        aggregate_value=0.5,
        replicate_values={"seed-1": 0.5},
    )

    comparison = ExpertSourceReplayFingerprintComparison(
        evaluation_fingerprint=fingerprint,
        metric_comparison_binding=binding,
        control_result=control,
        candidate_result=candidate,
        aggregate_raw_delta=-0.5,
        aggregate_direction_aligned_delta=0.5,
        aggregate_normalized_effect=0.25,
    )

    comparison.validate_aggregates(0.0)


def test_comparison_rejects_nonfinite_derived_values_and_signed_zero(tmp_path):
    _, prepared, _, _ = _journal_fixture(tmp_path)
    materialized_case = prepared.cases[0]
    terminal_attempt = materialized_case.episode.attempts[
        materialized_case.episode.terminal_attempt_revision
    ]
    fingerprint = terminal_attempt.evaluation_fingerprints[0]
    binding = materialized_case.task_adapter.manifest.task_evaluator.metric_comparison_bindings[
        0
    ]
    maximum = sys.float_info.max
    control = TaskEvaluatorFingerprintResult(
        evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
        aggregate_value=-maximum,
        replicate_values={"seed-1": -maximum},
    )
    candidate = TaskEvaluatorFingerprintResult(
        evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
        aggregate_value=maximum,
        replicate_values={"seed-1": maximum},
    )
    with pytest.raises(ValueError, match="finite"):
        ExpertSourceReplayFingerprintComparison(
            evaluation_fingerprint=fingerprint,
            metric_comparison_binding=binding,
            control_result=control,
            candidate_result=candidate,
            aggregate_raw_delta=maximum - -maximum,
            aggregate_direction_aligned_delta=maximum - -maximum,
            aggregate_normalized_effect=maximum - -maximum,
        )

    zero = TaskEvaluatorFingerprintResult(
        evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
        aggregate_value=0.0,
        replicate_values={"seed-1": 0.0},
    )
    with pytest.raises(ExpertSourceReplayProtocolError, match="signed zero"):
        ExpertSourceReplayFingerprintComparison(
            evaluation_fingerprint=fingerprint,
            metric_comparison_binding=binding,
            control_result=zero,
            candidate_result=zero,
            aggregate_raw_delta=-0.0,
            aggregate_direction_aligned_delta=0.0,
            aggregate_normalized_effect=0.0,
        )


def test_stable_arithmetic_mean_rejects_empty_and_nonfinite_values():
    with pytest.raises(TaskEvaluatorProtocolError, match="finite"):
        stable_arithmetic_mean(())
    with pytest.raises(TaskEvaluatorProtocolError, match="finite"):
        stable_arithmetic_mean((float("inf"),))
