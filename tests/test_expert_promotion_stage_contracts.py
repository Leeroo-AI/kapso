from __future__ import annotations

from dataclasses import replace

import pytest

from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixMode,
    ExpertReleaseMatrixReport,
    ExpertReleaseMatrixTaskExecutionEvidence,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageContractError,
    ExpertReleaseMatrixStageResultRecord,
)
from test_expert_promotion import _id, _matrix

_AUTHORIZATION_TRANSITION_ID = _id(
    "expert-validation-transition",
    "release-matrix-stage",
)
_AUTHORIZATION_STATE_ID = _id(
    "expert-candidate-validation-state",
    "release-matrix-stage",
)


def _remint(record, **changes):
    values = record.to_dict()
    values.pop(record.IDENTITY_FIELD)
    values.update(changes)
    return type(record).mint(**values)


def _report_with_task_authority(
    *,
    omit_request_authority_id: str | None = None,
    omit_reservation_authority_id: str | None = None,
) -> ExpertReleaseMatrixReport:
    _, _, _, report = _matrix(ExpertReleaseMatrixMode.BOOTSTRAP)
    evidence = report.task_execution_evidence
    assert evidence is not None
    required_authority = {
        _AUTHORIZATION_TRANSITION_ID,
        _AUTHORIZATION_STATE_ID,
        report.plan_reservation_operation_id,
        report.evaluation_plan.evaluation_plan_id,
    }
    reservation_dependencies = tuple(
        sorted(
            {*evidence.reservation_dependency_ids, *required_authority}
            - (
                {omit_reservation_authority_id}
                if omit_reservation_authority_id
                else set()
            )
        )
    )
    request_dependencies = tuple(
        sorted(
            {
                *evidence.request_dependency_ids,
                *required_authority,
            }
            - ({omit_request_authority_id} if omit_request_authority_id else set())
        )
    )
    evidence_dependencies = tuple(
        sorted(
            {
                evidence.reservation_id,
                evidence.request_id,
                *reservation_dependencies,
                *request_dependencies,
                *evidence.execution_journal_event_ids,
            }
        )
    )
    authorized_evidence = ExpertReleaseMatrixTaskExecutionEvidence.mint(
        mode=evidence.mode,
        reservation_id=evidence.reservation_id,
        request_id=evidence.request_id,
        aggregate_recomputation_tolerance=evidence.aggregate_recomputation_tolerance,
        execution_journal_event_ids=evidence.execution_journal_event_ids,
        reservation_dependency_ids=reservation_dependencies,
        request_dependency_ids=request_dependencies,
        case_evidence=evidence.case_evidence,
        exact_dependency_ids=evidence_dependencies,
    )
    report_dependencies = {
        *report.exact_dependency_ids,
        *authorized_evidence.exact_dependency_ids,
        authorized_evidence.task_execution_evidence_id,
    }
    report_dependencies.remove(evidence.task_execution_evidence_id)
    return _remint(
        report,
        task_execution_evidence=authorized_evidence,
        exact_dependency_ids=tuple(sorted(report_dependencies)),
    )


def _stage(
    report: ExpertReleaseMatrixReport | None = None,
) -> ExpertReleaseMatrixStageResultRecord:
    selected_report = report or _report_with_task_authority()
    evidence = selected_report.task_execution_evidence
    reservation_id = (
        evidence.reservation_id
        if evidence is not None
        else _id("task-evaluation-reservation", "missing-task-evidence")
    )
    dependencies = {
        selected_report.validation_attempt_id,
        _AUTHORIZATION_TRANSITION_ID,
        _AUTHORIZATION_STATE_ID,
        selected_report.candidate_id,
        selected_report.scope_contract_id,
        selected_report.validation_policy_id,
        selected_report.plan_reservation_operation_id,
        reservation_id,
        selected_report.release_matrix_report_id,
        *selected_report.exact_dependency_ids,
    }
    if selected_report.parent_release_id is not None:
        dependencies.add(selected_report.parent_release_id)
    return ExpertReleaseMatrixStageResultRecord.mint(
        validation_attempt_id=selected_report.validation_attempt_id,
        authorization_transition_id=_AUTHORIZATION_TRANSITION_ID,
        authorization_state_id=_AUTHORIZATION_STATE_ID,
        candidate_id=selected_report.candidate_id,
        candidate_tree_hash=selected_report.candidate_tree_hash,
        scope_contract_id=selected_report.scope_contract_id,
        parent_release_id=selected_report.parent_release_id,
        validation_policy_id=selected_report.validation_policy_id,
        configuration_fingerprint=selected_report.configuration_fingerprint,
        plan_reservation_operation_id=(selected_report.plan_reservation_operation_id),
        task_evaluation_reservation_id=reservation_id,
        release_matrix_report=selected_report,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def test_release_matrix_stage_result_round_trips_exact_embedded_report():
    result = _stage()

    assert result.release_matrix_report.task_execution_evidence is not None
    assert (
        result.release_matrix_report.release_matrix_report_id
        in result.exact_dependency_ids
    )
    assert (
        ExpertReleaseMatrixStageResultRecord.from_json_bytes(result.to_json_bytes())
        == result
    )


def test_release_matrix_stage_result_rejects_subject_and_namespace_substitution():
    result = _stage()

    with pytest.raises(ExpertReleaseMatrixStageContractError, match="subjects differ"):
        replace(result, candidate_tree_hash="sha256:" + "1" * 64)
    with pytest.raises(ExpertReleaseMatrixStageContractError, match="wrong namespace"):
        _remint(
            result,
            authorization_transition_id=_id("untrusted-transition", "substitute"),
        )


@pytest.mark.parametrize("projection", ("request", "reservation"))
def test_release_matrix_stage_result_requires_authority_in_both_task_projections(
    projection: str,
):
    report = _report_with_task_authority(
        omit_request_authority_id=(
            _AUTHORIZATION_STATE_ID if projection == "request" else None
        ),
        omit_reservation_authority_id=(
            _AUTHORIZATION_STATE_ID if projection == "reservation" else None
        ),
    )

    with pytest.raises(ExpertReleaseMatrixStageContractError, match="omits stage"):
        _stage(report)


def test_release_matrix_stage_result_requires_task_evidence_and_exact_dependencies():
    result = _stage()
    _, _, _, report_without_task_evidence = _matrix()
    with pytest.raises(ExpertReleaseMatrixStageContractError, match="requires task"):
        _stage(report_without_task_evidence)

    with pytest.raises(ExpertReleaseMatrixStageContractError, match="task reservation"):
        replace(
            result,
            task_evaluation_reservation_id=_id(
                "task-evaluation-reservation",
                "substitute",
            ),
        )

    with pytest.raises(ExpertReleaseMatrixStageContractError, match="not exact"):
        replace(
            result,
            exact_dependency_ids=result.exact_dependency_ids[:-1],
        )
