"""Exact factual evidence reduction for a precommitted expert release matrix."""

from __future__ import annotations

from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixComparisonRow,
    ExpertReleaseMatrixProvenanceKind,
)
from kapso.cross_run.expert.promotion_plan import (
    validate_expert_release_matrix_source_joins,
)
from kapso.cross_run.expert.validation_snapshots import (
    ExpertReleaseMatrixPlanReservationSnapshot,
)
from kapso.cross_run.expert.validation_store import (
    ExpertReleaseMatrixSourceEvidenceSnapshot,
    ExpertValidationStore,
)


class ExpertReleaseMatrixEvidenceError(ValueError):
    """Accepted evidence cannot cover its precommitted release-matrix cells."""


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
