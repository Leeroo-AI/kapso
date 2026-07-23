from __future__ import annotations

import pytest

import kapso.cross_run.expert.promotion_evidence as evidence_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalStoreError,
)
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixContractError,
    ExpertReleaseMatrixMode,
    ExpertReleaseMatrixProvenanceKind,
    ExpertReleaseMatrixReport,
    ExpertReleaseMatrixTaskExecutionEvidence,
)
from kapso.cross_run.expert.promotion_evidence import (
    ExpertReleaseMatrixEvidenceError,
    _merge_expert_release_matrix_rows,
    derive_expert_release_matrix_report,
)
from kapso.cross_run.expert.task_evaluation_authority import (
    TaskEvaluationFreshAuthorityCoordinator,
)
from kapso.cross_run.expert.task_evaluation_contracts import TaskEvaluationLegKind
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderRegistry,
    project_prepared_task_evaluation_cases,
)
from kapso.cross_run.expert.task_evaluation_execution_journal import (
    TaskEvaluationExecutionJournalEventKind,
    task_evaluation_execution_schedule,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    ExpertTaskEvaluationExecutionStore,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TaskEvaluatorFingerprintResult,
    TaskEvaluatorResult,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStore
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
)
from test_expert_task_evaluation_execution import (
    _parent_prepared_with_additional_case,
)
from test_expert_task_evaluation_execution_store import (
    _AdapterAuthority,
    _CurrentAuthority,
    _DenylistAuthority,
    _Provider,
    _commit_spawn,
)
from test_expert_task_evaluation_preflight import (
    _CurrentAuthority as _PreflightCurrentAuthority,
)
from test_expert_task_evaluation_preflight import (
    _coordinator,
    _current_observation,
)


class _SemanticProvider(_Provider):
    def __init__(self, trusted_root, dispatch_key, values_by_case_and_kind):
        super().__init__(trusted_root, dispatch_key)
        self.values_by_case_and_kind = values_by_case_and_kind

    def execute_leg(self, invocation):
        value = self.values_by_case_and_kind[
            invocation.invocation_allocation.evaluation_case_id
        ][invocation.selected_leg.authority.kind]
        request = invocation.task_evaluator_request
        self.result_payload = TaskEvaluatorResult(
            protocol_version=request.protocol_version,
            opaque_invocation_id=request.opaque_invocation_id,
            fingerprint_results=tuple(
                TaskEvaluatorFingerprintResult(
                    evaluation_fingerprint_id=(fingerprint.evaluation_fingerprint_id),
                    aggregate_value=value,
                    replicate_values={
                        replicate_id: value
                        for replicate_id in fingerprint.seed_or_replicate_ids
                    },
                )
                for fingerprint in request.evaluation_fingerprints
            ),
        ).to_json_bytes()
        return super().execute_leg(invocation)


def _bootstrap_prepared_with_store(tmp_path, monkeypatch):
    validation_store, snapshot, prepared_plan, _active_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    observation = _current_observation(prepared_plan)
    coordinator, *_providers = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=None,
        current_authority=_PreflightCurrentAuthority((observation, observation)),
    )
    return validation_store, snapshot, coordinator.build(plan_reservation)


def _execution_runtime(
    validation_store,
    snapshot,
    prepared,
    provider_factory=None,
):
    reservation_snapshot = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    ).reservation
    execution_store = ExpertTaskEvaluationExecutionStore(
        ExpertTaskEvaluationExecutionStore.canonical_root(
            validation_store.root
        ).resolve(),
        validation_store.root,
        prepared.plan_join.settings.policy,
    )
    executable_cases = project_prepared_task_evaluation_cases(prepared)
    provider_keys = tuple(
        sorted(
            {case.provider_key for case in executable_cases},
            key=lambda provider_key: provider_key.identity,
        )
    )
    providers = tuple(
        (
            _Provider(validation_store.root, provider_key)
            if provider_factory is None
            else provider_factory(validation_store.root, provider_key)
        )
        for provider_key in provider_keys
    )
    registry = TaskEvaluationExecutionProviderRegistry(prepared, providers)
    authority_coordinator = TaskEvaluationFreshAuthorityCoordinator(
        reservation_authority=validation_store,
        execution_store=execution_store,
        current_release_authority=_CurrentAuthority(
            prepared.current_release_observation
        ),
        task_adapter_authority=_AdapterAuthority(prepared),
        security_denylist_authority=_DenylistAuthority(prepared),
    )
    return reservation_snapshot, execution_store, registry, authority_coordinator


def _complete_execution(
    *,
    prepared,
    reservation_snapshot,
    execution_store,
    registry,
    authority_coordinator,
):
    schedule = task_evaluation_execution_schedule(
        reservation_snapshot,
        prepared,
    )
    with execution_store.reservation_session(
        reservation_snapshot=reservation_snapshot,
        prepared_request=prepared,
    ) as session:
        while len(session.events) < 4 * len(schedule):
            spawn = _commit_spawn(
                session,
                prepared,
                reservation_snapshot,
                registry,
                authority_coordinator,
            )
            session.record_result_received(spawn.execute())
            session.accept_received_result()
        return session.completed_execution()


def _derive_report(
    validation_store,
    snapshot,
    prepared,
    provider_factory=None,
):
    reservation, execution_store, registry, authority = _execution_runtime(
        validation_store,
        snapshot,
        prepared,
        provider_factory,
    )
    completed = _complete_execution(
        prepared=prepared,
        reservation_snapshot=reservation,
        execution_store=execution_store,
        registry=registry,
        authority_coordinator=authority,
    )
    report = derive_expert_release_matrix_report(
        validation_store=validation_store,
        execution_store=execution_store,
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    return report, completed, reservation, execution_store


def _remint_task_evidence(task_evidence, case_evidence):
    return ExpertReleaseMatrixTaskExecutionEvidence.mint(
        mode=task_evidence.mode,
        reservation_id=task_evidence.reservation_id,
        request_id=task_evidence.request_id,
        aggregate_recomputation_tolerance=(
            task_evidence.aggregate_recomputation_tolerance
        ),
        execution_journal_event_ids=task_evidence.execution_journal_event_ids,
        reservation_dependency_ids=task_evidence.reservation_dependency_ids,
        request_dependency_ids=task_evidence.request_dependency_ids,
        case_evidence=case_evidence,
        exact_dependency_ids=task_evidence.exact_dependency_ids,
    )


def test_parent_report_preserves_semantic_values_after_both_stores_reopen(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _parent_prepared_with_additional_case(
        tmp_path,
        monkeypatch,
    )
    values_by_case_and_kind = {
        case.evaluation_case_id: {
            TaskEvaluationLegKind.CANDIDATE: float(position * 10 + 7),
            TaskEvaluationLegKind.PARENT_CONTROL: float(position * 10 + 3),
        }
        for position, case in enumerate(prepared.plan_join.request.cases, start=1)
    }
    report, _completed, reservation, execution_store = _derive_report(
        validation_store,
        snapshot,
        prepared,
        lambda trusted_root, provider_key: _SemanticProvider(
            trusted_root,
            provider_key,
            values_by_case_and_kind,
        ),
    )
    plan = prepared.plan_join.plan_reservation.evaluation_plan
    task_case_by_provenance = {
        case.provenance_binding_id: case for case in prepared.plan_join.request.cases
    }
    for cell, row in zip(plan.evaluation_cells, report.evidence_rows, strict=True):
        case = task_case_by_provenance.get(cell.provenance_binding_id)
        if case is None:
            continue
        expected_values = values_by_case_and_kind[case.evaluation_case_id]
        assert set(row.candidate_replicate_values.values()) == {
            expected_values[TaskEvaluationLegKind.CANDIDATE]
        }
        assert set(row.parent_replicate_values.values()) == {
            expected_values[TaskEvaluationLegKind.PARENT_CONTROL]
        }

    reopened_validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened_reservation = reopened_validation_store.reopen_task_evaluation_reservation(
        reservation_id=reservation.reservation.reservation_id,
        prepared_request=prepared,
    )
    reopened_execution_store = ExpertTaskEvaluationExecutionStore(
        execution_store.root,
        execution_store.trusted_root,
        execution_store.policy_settings,
    )
    with reopened_execution_store.reservation_session(
        reservation_snapshot=reopened_reservation,
        prepared_request=prepared,
    ) as session:
        reopened_completion = session.completed_execution()
    reopened_report = derive_expert_release_matrix_report(
        validation_store=reopened_validation_store,
        execution_store=reopened_execution_store,
        completed_execution=reopened_completion,
        reservation_snapshot=reopened_reservation,
        prepared_request=prepared,
    )

    assert reopened_report.to_json_bytes() == report.to_json_bytes()


def test_parent_report_merges_source_and_case_scoped_task_evidence(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _parent_prepared_with_additional_case(
        tmp_path,
        monkeypatch,
    )
    report, completed, reservation, _execution_store = _derive_report(
        validation_store,
        snapshot,
        prepared,
    )
    plan = prepared.plan_join.plan_reservation.evaluation_plan
    provenance_by_id = {
        provenance.provenance_binding_id: provenance
        for provenance in plan.provenance_bindings
    }

    assert type(report) is ExpertReleaseMatrixReport
    assert tuple(row.evaluation_cell_id for row in report.evidence_rows) == tuple(
        cell.evaluation_cell_id for cell in plan.evaluation_cells
    )
    assert report.task_execution_evidence is not None
    assert report.task_execution_evidence.execution_journal_event_ids == tuple(
        event.event_id for event in completed.events
    )
    assert report.task_execution_evidence.reservation_id == (
        reservation.reservation.reservation_id
    )
    assert report.task_execution_evidence.task_execution_evidence_id in (
        report.exact_dependency_ids
    )
    for cell, row in zip(plan.evaluation_cells, report.evidence_rows, strict=True):
        expected_namespace = (
            "source-replay-execution-journal-event"
            if provenance_by_id[cell.provenance_binding_id].provenance_kind
            is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
            else "task-evaluation-execution-journal-event"
        )
        assert row.candidate_observation_event_id.startswith(f"{expected_namespace}:")

    request_cases = prepared.plan_join.request.cases
    assert len(request_cases) == 2
    assert set(leg.leg_id for leg in request_cases[0].legs).intersection(
        leg.leg_id for leg in request_cases[1].legs
    )
    accepted_events = {
        (
            event.invocation_allocation.evaluation_case_id,
            event.invocation_allocation.evaluation_leg_id,
        ): event
        for event in completed.events
        if event.event_kind is TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED
    }
    case_evidence_by_provenance = {
        case.provenance_binding_id: case
        for case in report.task_execution_evidence.case_evidence
    }
    for request_case in request_cases:
        legs_by_kind = {leg.kind: leg for leg in request_case.legs}
        case_evidence = case_evidence_by_provenance[request_case.provenance_binding_id]
        assert (
            case_evidence.candidate_result_accepted_event_id
            == accepted_events[
                (
                    request_case.evaluation_case_id,
                    legs_by_kind[TaskEvaluationLegKind.CANDIDATE].leg_id,
                )
            ].event_id
        )
        assert (
            case_evidence.parent_result_accepted_event_id
            == accepted_events[
                (
                    request_case.evaluation_case_id,
                    legs_by_kind[TaskEvaluationLegKind.PARENT_CONTROL].leg_id,
                )
            ].event_id
        )

    task_evidence = report.task_execution_evidence
    first_case, second_case = task_evidence.case_evidence
    with pytest.raises(
        ExpertReleaseMatrixContractError,
        match="canonical and unique",
    ):
        _remint_task_evidence(
            task_evidence,
            (
                first_case,
                type(second_case)(
                    evaluation_case_id=first_case.evaluation_case_id,
                    provenance_binding_id=second_case.provenance_binding_id,
                    candidate_result_accepted_event_id=(
                        second_case.candidate_result_accepted_event_id
                    ),
                    parent_result_accepted_event_id=(
                        second_case.parent_result_accepted_event_id
                    ),
                    evaluation_fingerprint_ids=(second_case.evaluation_fingerprint_ids),
                ),
            ),
        )
    with pytest.raises(
        ExpertReleaseMatrixContractError,
        match="request dependencies",
    ):
        _remint_task_evidence(
            task_evidence,
            (
                type(first_case)(
                    evaluation_case_id=content_id(
                        "task-evaluation-case",
                        {"foreign": True},
                    ),
                    provenance_binding_id=first_case.provenance_binding_id,
                    candidate_result_accepted_event_id=(
                        first_case.candidate_result_accepted_event_id
                    ),
                    parent_result_accepted_event_id=(
                        first_case.parent_result_accepted_event_id
                    ),
                    evaluation_fingerprint_ids=(first_case.evaluation_fingerprint_ids),
                ),
                second_case,
            ),
        )

    source_rows = tuple(
        row
        for cell, row in zip(plan.evaluation_cells, report.evidence_rows, strict=True)
        if provenance_by_id[cell.provenance_binding_id].provenance_kind
        is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
    )
    task_rows = tuple(
        row
        for cell, row in zip(plan.evaluation_cells, report.evidence_rows, strict=True)
        if provenance_by_id[cell.provenance_binding_id].provenance_kind
        is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
    )
    with pytest.raises(ExpertReleaseMatrixEvidenceError, match="reuse"):
        _merge_expert_release_matrix_rows(
            plan_reservation=prepared.plan_join.plan_reservation,
            source_rows=(source_rows[0],),
            task_rows=(source_rows[0], *task_rows),
        )
    with pytest.raises(ExpertReleaseMatrixEvidenceError, match="exactly cover"):
        _merge_expert_release_matrix_rows(
            plan_reservation=prepared.plan_join.plan_reservation,
            source_rows=(),
            task_rows=task_rows,
        )


def test_bootstrap_report_is_candidate_only_and_does_not_reopen_source(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(
        evidence_module,
        "derive_expert_release_matrix_source_rows",
        lambda **_arguments: pytest.fail("bootstrap must not reopen source evidence"),
    )

    report, completed, _reservation, _execution_store = _derive_report(
        validation_store,
        snapshot,
        prepared,
    )

    assert report.mode is ExpertReleaseMatrixMode.BOOTSTRAP
    assert report.task_execution_evidence is not None
    assert report.task_execution_evidence.mode is ExpertReleaseMatrixMode.BOOTSTRAP
    assert len(report.task_execution_evidence.execution_journal_event_ids) == (
        4 * len(report.task_execution_evidence.case_evidence)
    )
    assert all(
        case.parent_result_accepted_event_id is None
        for case in report.task_execution_evidence.case_evidence
    )
    assert all(row.parent_observation_event_id is None for row in report.evidence_rows)
    assert all(
        event.event_kind is TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED
        for event in completed.events[3::4]
    )


def test_report_requires_sealed_completion_and_its_exact_execution_store(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
        tmp_path,
        monkeypatch,
    )
    report, completed, reservation, execution_store = _derive_report(
        validation_store,
        snapshot,
        prepared,
    )
    foreign_store = ExpertTaskEvaluationExecutionStore(
        execution_store.root,
        execution_store.trusted_root,
        execution_store.policy_settings,
    )

    assert report.task_execution_evidence is not None
    with pytest.raises(ExecutionJournalStoreError, match="journal authority"):
        derive_expert_release_matrix_report(
            validation_store=validation_store,
            execution_store=foreign_store,
            completed_execution=completed,
            reservation_snapshot=reservation,
            prepared_request=prepared,
        )
    with pytest.raises(ExpertReleaseMatrixEvidenceError, match="completed"):
        derive_expert_release_matrix_report(
            validation_store=validation_store,
            execution_store=execution_store,
            completed_execution=completed.events,
            reservation_snapshot=reservation,
            prepared_request=prepared,
        )
