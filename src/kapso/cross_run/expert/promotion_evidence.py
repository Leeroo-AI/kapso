"""Exact factual evidence reduction for a precommitted expert release matrix."""

from __future__ import annotations

from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixComparisonRow,
    ExpertReleaseMatrixMode,
    ExpertReleaseMatrixProvenanceKind,
    ExpertReleaseMatrixReport,
    ExpertReleaseMatrixTaskCaseEvidence,
    ExpertReleaseMatrixTaskExecutionEvidence,
)
from kapso.cross_run.expert.promotion_plan import (
    validate_expert_release_matrix_source_joins,
)
from kapso.cross_run.expert.validation_snapshots import (
    ExpertReleaseMatrixPlanReservationSnapshot,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationCase,
    TaskEvaluationLegKind,
)
from kapso.cross_run.expert.task_evaluation_execution_journal import (
    TaskEvaluationExecutionJournalEvent,
    TaskEvaluationExecutionJournalEventKind,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    CompletedTaskEvaluationExecution,
    ExpertTaskEvaluationExecutionStore,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.expert.task_evaluator_protocol import TaskEvaluatorResult
from kapso.cross_run.expert.validation_store import (
    ExpertReleaseMatrixSourceEvidenceSnapshot,
    ExpertValidationStore,
)


class ExpertReleaseMatrixEvidenceError(ValueError):
    """Accepted evidence cannot cover its precommitted release-matrix cells."""


def derive_expert_release_matrix_report(
    *,
    validation_store: ExpertValidationStore,
    execution_store: ExpertTaskEvaluationExecutionStore,
    completed_execution: CompletedTaskEvaluationExecution,
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    prepared_request: PreparedTaskEvaluationRequest,
) -> ExpertReleaseMatrixReport:
    """Reduce one completed task journal and accepted source facts into a report."""

    if type(validation_store) is not ExpertValidationStore:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix report requires the canonical validation store"
        )
    if type(execution_store) is not ExpertTaskEvaluationExecutionStore:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix report requires the canonical task execution store"
        )
    if type(completed_execution) is not CompletedTaskEvaluationExecution:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix report requires a completed task execution"
        )
    if type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix report requires an exact task reservation"
        )
    if type(prepared_request) is not PreparedTaskEvaluationRequest:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix report requires an exact prepared task request"
        )
    reopened_reservation = validation_store.reopen_task_evaluation_reservation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        prepared_request=prepared_request,
    )
    if reopened_reservation != reservation_snapshot:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix task reservation differs from durable validation authority"
        )
    events = completed_execution.require_exact(
        execution_store,
        reservation_snapshot,
        prepared_request,
    )
    if (
        reservation_snapshot.request != prepared_request.plan_join.request
        or reservation_snapshot.plan_reservation
        != prepared_request.plan_join.plan_reservation
    ):
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix task authorities do not share one exact plan and request"
        )
    plan_reservation = reservation_snapshot.plan_reservation
    plan = plan_reservation.evaluation_plan
    aggregate_tolerance = (
        prepared_request.plan_join.settings.policy.task_evaluation_aggregate_tolerance
    )
    task_rows, task_case_evidence = _derive_expert_release_matrix_task_rows(
        reservation_snapshot=reservation_snapshot,
        events=events,
        aggregate_tolerance=aggregate_tolerance,
    )
    task_evidence_dependencies = {
        reservation_snapshot.reservation.reservation_id,
        reservation_snapshot.request.request_id,
        *reservation_snapshot.reservation.exact_dependency_ids,
        *reservation_snapshot.request.exact_dependency_ids,
        *(event.event_id for event in events),
    }
    task_execution_evidence = ExpertReleaseMatrixTaskExecutionEvidence.mint(
        mode=plan.mode,
        reservation_id=reservation_snapshot.reservation.reservation_id,
        request_id=reservation_snapshot.request.request_id,
        aggregate_recomputation_tolerance=aggregate_tolerance,
        execution_journal_event_ids=tuple(event.event_id for event in events),
        reservation_dependency_ids=(
            reservation_snapshot.reservation.exact_dependency_ids
        ),
        request_dependency_ids=reservation_snapshot.request.exact_dependency_ids,
        case_evidence=task_case_evidence,
        exact_dependency_ids=tuple(sorted(task_evidence_dependencies)),
    )
    has_source_provenance = any(
        provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
        for provenance in plan.provenance_bindings
    )
    source_rows = (
        derive_expert_release_matrix_source_rows(
            validation_store=validation_store,
            plan_reservation=plan_reservation,
        )
        if has_source_provenance
        else ()
    )
    evidence_rows = _merge_expert_release_matrix_rows(
        plan_reservation=plan_reservation,
        source_rows=source_rows,
        task_rows=task_rows,
    )
    dependencies = {
        plan.validation_attempt_id,
        plan.candidate_id,
        plan.candidate_commit_record_id,
        plan.scope_contract_id,
        plan.validation_policy_id,
        plan_reservation.operation.operation_id,
        plan.evaluation_plan_id,
        *plan.exact_dependency_ids,
        *(row.comparison_row_id for row in evidence_rows),
        *(
            dependency_id
            for row in evidence_rows
            for dependency_id in row.exact_dependency_ids
        ),
        task_execution_evidence.task_execution_evidence_id,
        *task_execution_evidence.exact_dependency_ids,
    }
    if plan.parent_release_id is not None:
        dependencies.add(plan.parent_release_id)
    return ExpertReleaseMatrixReport.mint(
        mode=plan.mode,
        validation_attempt_id=plan.validation_attempt_id,
        candidate_id=plan.candidate_id,
        candidate_commit_record_id=plan.candidate_commit_record_id,
        candidate_tree_hash=plan.candidate_tree_hash,
        scope_contract_id=plan.scope_contract_id,
        parent_release_id=plan.parent_release_id,
        parent_tree_hash=plan.parent_tree_hash,
        validation_policy_id=plan.validation_policy_id,
        configuration_fingerprint=plan.configuration_fingerprint,
        plan_reservation_operation_id=plan_reservation.operation.operation_id,
        evaluation_plan=plan,
        task_execution_evidence=task_execution_evidence,
        evidence_rows=evidence_rows,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def derive_expert_release_matrix_source_rows(
    *,
    validation_store: ExpertValidationStore,
    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
) -> tuple[ExpertReleaseMatrixComparisonRow, ...]:
    """Project accepted source-stage results without invoking an evaluator again."""

    if type(validation_store) is not ExpertValidationStore:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix source rows require the canonical validation store"
        )
    source_evidence = validation_store.reopen_release_matrix_source_evidence(
        plan_reservation=plan_reservation,
    )
    if type(source_evidence) is not ExpertReleaseMatrixSourceEvidenceSnapshot:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix source rows lack accepted source evidence"
        )
    plan = source_evidence.plan_reservation.evaluation_plan
    stage_result = source_evidence.stage_result
    request = source_evidence.request
    validate_expert_release_matrix_source_joins(
        plan,
        stage_result,
        request,
    )
    provenance_by_id = {
        provenance.provenance_binding_id: provenance
        for provenance in plan.provenance_bindings
    }
    comparison_by_case_id = {
        comparison.execution_case_id: comparison
        for comparison in stage_result.paired_comparison_receipt.case_comparisons
    }
    rows = []
    for cell in plan.evaluation_cells:
        provenance = provenance_by_id[cell.provenance_binding_id]
        if (
            provenance.provenance_kind
            is not ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
        ):
            continue
        case_comparison = comparison_by_case_id.get(provenance.source_execution_case_id)
        if case_comparison is None:
            raise ExpertReleaseMatrixEvidenceError(
                "release matrix source cell has no accepted case comparison"
            )
        fingerprint_comparisons = {
            comparison.evaluation_fingerprint.evaluation_fingerprint_id: comparison
            for comparison in case_comparison.fingerprint_comparisons
        }
        fingerprint_comparison = fingerprint_comparisons.get(
            cell.evaluation_fingerprint.evaluation_fingerprint_id
        )
        if fingerprint_comparison is None:
            raise ExpertReleaseMatrixEvidenceError(
                "release matrix source cell has no accepted fingerprint comparison"
            )
        if (
            fingerprint_comparison.evaluation_fingerprint != cell.evaluation_fingerprint
            or fingerprint_comparison.metric_comparison_binding
            != cell.metric_comparison_binding
        ):
            raise ExpertReleaseMatrixEvidenceError(
                "release matrix source comparison differs from its planned cell"
            )
        rows.append(
            ExpertReleaseMatrixComparisonRow.mint(
                evaluation_cell_id=cell.evaluation_cell_id,
                candidate_observation_event_id=(
                    case_comparison.candidate_result_accepted_event_id
                ),
                parent_observation_event_id=(
                    case_comparison.control_result_accepted_event_id
                ),
                candidate_replicate_values=dict(
                    fingerprint_comparison.candidate_result.replicate_values
                ),
                parent_replicate_values=dict(
                    fingerprint_comparison.control_result.replicate_values
                ),
            )
        )
    if not rows:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix plan contains no reusable source cells"
        )
    return tuple(rows)


def _derive_expert_release_matrix_task_rows(
    *,
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    events: tuple[TaskEvaluationExecutionJournalEvent, ...],
    aggregate_tolerance: float,
) -> tuple[
    tuple[ExpertReleaseMatrixComparisonRow, ...],
    tuple[ExpertReleaseMatrixTaskCaseEvidence, ...],
]:
    if type(events) is not tuple or any(
        type(event) is not TaskEvaluationExecutionJournalEvent for event in events
    ):
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix task evidence requires exact journal events"
        )
    request = reservation_snapshot.request
    plan = reservation_snapshot.plan_reservation.evaluation_plan
    expected_event_keys = {
        (case.evaluation_case_id, leg.leg_id)
        for case in request.cases
        for leg in case.legs
    }
    event_counts = {event_key: 0 for event_key in expected_event_keys}
    accepted_events = {}
    for event in events:
        allocation = event.invocation_allocation
        event_key = allocation.evaluation_case_id, allocation.evaluation_leg_id
        if (
            event_key not in event_counts
            or event.request_id != request.request_id
            or allocation.reservation_id
            != reservation_snapshot.reservation.reservation_id
        ):
            raise ExpertReleaseMatrixEvidenceError(
                "release matrix task journal differs from its reservation schedule"
            )
        event_counts[event_key] += 1
        if (
            event.event_kind is TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED
            and event.aggregate_tolerance != aggregate_tolerance
        ):
            raise ExpertReleaseMatrixEvidenceError(
                "release matrix task journal tolerance differs from configuration"
            )
        if (
            event.event_kind
            is not TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED
        ):
            continue
        if (
            event_key in accepted_events
            or type(event.task_evaluator_result) is not TaskEvaluatorResult
        ):
            raise ExpertReleaseMatrixEvidenceError(
                "release matrix accepted task events differ from their reservation"
            )
        accepted_events[event_key] = event
    if set(accepted_events) != expected_event_keys or any(
        event_count != 4 for event_count in event_counts.values()
    ):
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix accepted task event coverage is not exact"
        )
    adapter_provenances = tuple(
        provenance
        for provenance in plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
    )
    cases_by_provenance_id = {
        case.provenance_binding_id: case for case in request.cases
    }
    if len(cases_by_provenance_id) != len(request.cases) or set(
        cases_by_provenance_id
    ) != {provenance.provenance_binding_id for provenance in adapter_provenances}:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix task cases do not exactly cover adapter provenances"
        )
    cells_by_provenance_id = {
        provenance.provenance_binding_id: tuple(
            cell
            for cell in plan.evaluation_cells
            if cell.provenance_binding_id == provenance.provenance_binding_id
        )
        for provenance in adapter_provenances
    }
    rows = []
    case_evidence = []
    for provenance in adapter_provenances:
        case = cases_by_provenance_id[provenance.provenance_binding_id]
        cells = cells_by_provenance_id[provenance.provenance_binding_id]
        _validate_expert_release_matrix_task_case(case, provenance, cells)
        legs_by_kind = {leg.kind: leg for leg in case.legs}
        expected_leg_kinds = (
            {TaskEvaluationLegKind.CANDIDATE}
            if plan.mode is ExpertReleaseMatrixMode.BOOTSTRAP
            else set(TaskEvaluationLegKind)
        )
        if len(legs_by_kind) != len(case.legs) or set(legs_by_kind) != (
            expected_leg_kinds
        ):
            raise ExpertReleaseMatrixEvidenceError(
                "release matrix task legs differ from the matrix mode"
            )
        candidate_event = accepted_events[
            (
                case.evaluation_case_id,
                legs_by_kind[TaskEvaluationLegKind.CANDIDATE].leg_id,
            )
        ]
        parent_event = None
        if plan.mode is ExpertReleaseMatrixMode.PARENT_COMPARISON:
            parent_event = accepted_events[
                (
                    case.evaluation_case_id,
                    legs_by_kind[TaskEvaluationLegKind.PARENT_CONTROL].leg_id,
                )
            ]
        candidate_results = _task_fingerprint_results(candidate_event, case)
        parent_results = (
            None
            if parent_event is None
            else _task_fingerprint_results(parent_event, case)
        )
        case_evidence.append(
            ExpertReleaseMatrixTaskCaseEvidence(
                evaluation_case_id=case.evaluation_case_id,
                provenance_binding_id=case.provenance_binding_id,
                candidate_result_accepted_event_id=candidate_event.event_id,
                parent_result_accepted_event_id=(
                    None if parent_event is None else parent_event.event_id
                ),
                evaluation_fingerprint_ids=case.evaluation_fingerprint_ids,
            )
        )
        for cell in cells:
            fingerprint_id = cell.evaluation_fingerprint.evaluation_fingerprint_id
            candidate_result = candidate_results[fingerprint_id]
            parent_result = (
                None if parent_results is None else parent_results[fingerprint_id]
            )
            rows.append(
                ExpertReleaseMatrixComparisonRow.mint(
                    evaluation_cell_id=cell.evaluation_cell_id,
                    candidate_observation_event_id=candidate_event.event_id,
                    parent_observation_event_id=(
                        None if parent_event is None else parent_event.event_id
                    ),
                    candidate_replicate_values=dict(candidate_result.replicate_values),
                    parent_replicate_values=(
                        None
                        if parent_result is None
                        else dict(parent_result.replicate_values)
                    ),
                )
            )
    return tuple(rows), tuple(
        sorted(case_evidence, key=lambda evidence: evidence.canonical_key)
    )


def _validate_expert_release_matrix_task_case(case, provenance, cells) -> None:
    signed_case = provenance.adapter_case
    if (
        type(case) is not TaskEvaluationCase
        or signed_case is None
        or case.adapter_authority_id != provenance.adapter_authority_id
        or case.provenance_binding_id != provenance.provenance_binding_id
        or case.release_matrix_case_id != signed_case.release_matrix_case_id
        or case.task_context_binding_id
        != provenance.task_context_binding.task_context_binding_id
        or case.independence_group_id
        != signed_case.independence_group.independence_group_id
        or case.independence_group_id != provenance.independence_identity_id
        or case.evaluation_cell_ids
        != tuple(sorted(cell.evaluation_cell_id for cell in cells))
        or case.evaluation_fingerprint_ids != provenance.evaluation_fingerprint_ids
        or case.evaluation_fingerprint_ids
        != tuple(
            sorted(
                cell.evaluation_fingerprint.evaluation_fingerprint_id for cell in cells
            )
        )
        or case.starting_artifact_ids != provenance.starting_artifact_ids
    ):
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix task case differs from its planned provenance"
        )


def _task_fingerprint_results(
    event: TaskEvaluationExecutionJournalEvent,
    case: TaskEvaluationCase,
):
    result = event.task_evaluator_result
    if type(result) is not TaskEvaluatorResult:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix accepted task event lacks an exact evaluator result"
        )
    results_by_fingerprint_id = {
        fingerprint_result.evaluation_fingerprint_id: fingerprint_result
        for fingerprint_result in result.fingerprint_results
    }
    if len(results_by_fingerprint_id) != len(result.fingerprint_results) or set(
        results_by_fingerprint_id
    ) != set(case.evaluation_fingerprint_ids):
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix task fingerprint coverage is not exact"
        )
    return results_by_fingerprint_id


def _merge_expert_release_matrix_rows(
    *,
    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    source_rows: tuple[ExpertReleaseMatrixComparisonRow, ...],
    task_rows: tuple[ExpertReleaseMatrixComparisonRow, ...],
) -> tuple[ExpertReleaseMatrixComparisonRow, ...]:
    if type(plan_reservation) is not ExpertReleaseMatrixPlanReservationSnapshot:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix row merge requires an exact plan reservation"
        )
    if (
        type(source_rows) is not tuple
        or type(task_rows) is not tuple
        or any(
            type(row) is not ExpertReleaseMatrixComparisonRow
            for row in (*source_rows, *task_rows)
        )
    ):
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix row merge requires exact comparison rows"
        )
    rows_by_cell_id = {}
    for row in (*source_rows, *task_rows):
        if row.evaluation_cell_id in rows_by_cell_id:
            raise ExpertReleaseMatrixEvidenceError(
                "release matrix source and task evidence reuse a planned cell"
            )
        rows_by_cell_id[row.evaluation_cell_id] = row
    plan = plan_reservation.evaluation_plan
    expected_cell_ids = {cell.evaluation_cell_id for cell in plan.evaluation_cells}
    if set(rows_by_cell_id) != expected_cell_ids:
        raise ExpertReleaseMatrixEvidenceError(
            "release matrix source and task rows do not exactly cover the plan"
        )
    return tuple(
        rows_by_cell_id[cell.evaluation_cell_id] for cell in plan.evaluation_cells
    )
