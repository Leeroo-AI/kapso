"""Join task-evaluation requests to their reserved release-matrix authority."""

from __future__ import annotations

from dataclasses import dataclass

from kapso.cross_run.contracts import ExpertValidationStage
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixProvenanceKind,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationRequest,
)
from kapso.cross_run.expert.validation_store import (
    ExpertReleaseMatrixPlanReservationSnapshot,
)
from kapso.cross_run.settings import ExpertValidationSettings


class TaskEvaluationRequestPreparationError(ValueError):
    """A task-evaluation request differs from its reserved matrix plan."""


@dataclass(frozen=True)
class PlanJoinedTaskEvaluationRequest:
    """Runtime proof that a request is the exact adapter-case plan projection.

    This closure proves plan and configured-evaluator joins only. It is not an
    execution capability: dispatch additionally requires materialized package
    bytes, configured compute bindings, and fresh adapter authority.
    """

    request: TaskEvaluationRequest
    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot
    settings: ExpertValidationSettings

    def __post_init__(self) -> None:
        if (
            type(self.request) is not TaskEvaluationRequest
            or type(self.plan_reservation)
            is not ExpertReleaseMatrixPlanReservationSnapshot
            or type(self.settings) is not ExpertValidationSettings
        ):
            raise TaskEvaluationRequestPreparationError(
                "task evaluation plan join requires exact typed inputs"
            )
        self._validate_plan_subjects()
        self._validate_evaluator()
        self._validate_adapter_cases()

    def _validate_plan_subjects(self) -> None:
        request = self.request
        reservation = self.plan_reservation
        plan = reservation.evaluation_plan
        snapshot = reservation.snapshot
        attempt = snapshot.latest_attempt
        if attempt is None:
            raise TaskEvaluationRequestPreparationError(
                "task evaluation plan join lacks a validation attempt"
            )
        if (
            request.plan_reservation_operation_id != reservation.operation.operation_id
            or request.evaluation_plan_id != plan.evaluation_plan_id
            or request.mode is not plan.mode
            or request.authorization_transition_id != snapshot.transition.transition_id
            or request.authorization_state_id != snapshot.state.validation_state_id
            or request.validation_attempt_id != plan.validation_attempt_id
            or request.validation_attempt_id != attempt.validation_attempt_id
            or request.candidate_id != plan.candidate_id
            or request.candidate_commit_record_id != plan.candidate_commit_record_id
            or request.candidate_tree_hash != plan.candidate_tree_hash
            or request.scope_contract_id != plan.scope_contract_id
            or request.parent_release_id != plan.parent_release_id
            or request.parent_tree_hash != plan.parent_tree_hash
            or request.validation_policy_id != plan.validation_policy_id
            or request.configuration_fingerprint != plan.configuration_fingerprint
            or request.configuration_fingerprint
            != self.settings.configuration_fingerprint
            or request.plan_dependency_ids != plan.exact_dependency_ids
        ):
            raise TaskEvaluationRequestPreparationError(
                "task evaluation request differs from its reserved plan subjects"
            )

    def _validate_evaluator(self) -> None:
        evaluators = tuple(
            evaluator
            for evaluator in self.settings.policy.evaluators
            if evaluator.stage is ExpertValidationStage.RELEASE_MATRIX
        )
        if len(evaluators) != 1:
            raise TaskEvaluationRequestPreparationError(
                "task evaluation requires one configured release-matrix evaluator"
            )
        evaluator = evaluators[0]
        request = self.request
        if (
            request.release_matrix_evaluator_id != evaluator.evaluator_id
            or request.release_matrix_evaluator_role != evaluator.evaluator_role
            or request.release_matrix_evaluator_version != evaluator.evaluator_version
        ):
            raise TaskEvaluationRequestPreparationError(
                "task evaluation evaluator differs from configuration"
            )

    def _validate_adapter_cases(self) -> None:
        plan = self.plan_reservation.evaluation_plan
        expected_provenances = tuple(
            provenance
            for provenance in plan.provenance_bindings
            if provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
        )
        cases_by_provenance_id = {
            case.provenance_binding_id: case for case in self.request.cases
        }
        if len(cases_by_provenance_id) != len(self.request.cases) or set(
            cases_by_provenance_id
        ) != {provenance.provenance_binding_id for provenance in expected_provenances}:
            raise TaskEvaluationRequestPreparationError(
                "task evaluation request adapter provenance coverage is not exact"
            )
        cells_by_provenance_id = {
            provenance.provenance_binding_id: tuple(
                cell
                for cell in plan.evaluation_cells
                if cell.provenance_binding_id == provenance.provenance_binding_id
            )
            for provenance in expected_provenances
        }
        for provenance in expected_provenances:
            adapter_case = provenance.adapter_case
            if adapter_case is None:
                raise TaskEvaluationRequestPreparationError(
                    "task evaluation adapter provenance lacks its signed case"
                )
            case = cases_by_provenance_id[provenance.provenance_binding_id]
            cells = cells_by_provenance_id[provenance.provenance_binding_id]
            if (
                case.adapter_authority_id != provenance.adapter_authority_id
                or case.release_matrix_case_id != adapter_case.release_matrix_case_id
                or case.task_context_binding_id
                != provenance.task_context_binding.task_context_binding_id
                or case.independence_group_id
                != adapter_case.independence_group.independence_group_id
                or case.evaluation_cell_ids
                != tuple(sorted(cell.evaluation_cell_id for cell in cells))
                or case.evaluation_fingerprint_ids
                != provenance.evaluation_fingerprint_ids
                or case.evaluation_fingerprint_ids
                != tuple(
                    sorted(
                        cell.evaluation_fingerprint.evaluation_fingerprint_id
                        for cell in cells
                    )
                )
                or case.starting_artifact_ids != provenance.starting_artifact_ids
            ):
                raise TaskEvaluationRequestPreparationError(
                    "task evaluation case differs from its reserved provenance"
                )
