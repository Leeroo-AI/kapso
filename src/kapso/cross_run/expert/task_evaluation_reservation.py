"""Exact durable task-evaluation reservation authority."""

from __future__ import annotations

from dataclasses import dataclass

from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationRequest,
    TaskEvaluationReservation,
)
from kapso.cross_run.expert.validation_operation_contracts import (
    ExpertValidationOperation,
    ExpertValidationOperationKind,
)
from kapso.cross_run.expert.validation_snapshots import (
    ExpertReleaseMatrixPlanReservationSnapshot,
)


class TaskEvaluationReservationError(ValueError):
    """A durable task-evaluation reservation closure is inconsistent."""


@dataclass(frozen=True)
class ExpertTaskEvaluationReservationSnapshot:
    """One task request durably bound to its plan and admission observation."""

    operation: ExpertValidationOperation
    reservation: TaskEvaluationReservation
    request: TaskEvaluationRequest
    current_release_observation: TaskEvaluationCurrentReleaseObservation
    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot

    def __post_init__(self) -> None:
        if (
            type(self.operation) is not ExpertValidationOperation
            or type(self.reservation) is not TaskEvaluationReservation
            or type(self.request) is not TaskEvaluationRequest
            or type(self.current_release_observation)
            is not TaskEvaluationCurrentReleaseObservation
            or type(self.plan_reservation)
            is not ExpertReleaseMatrixPlanReservationSnapshot
        ):
            raise TaskEvaluationReservationError(
                "task evaluation reservation snapshot is not typed"
            )
        operation = self.operation
        reservation = self.reservation
        request = self.request
        observation = self.current_release_observation
        plan_reservation = self.plan_reservation
        plan = plan_reservation.evaluation_plan
        validation_snapshot = plan_reservation.snapshot
        attempt = validation_snapshot.latest_attempt
        scope_ids = {
            provenance.task_context_binding.scope_id
            for provenance in plan.provenance_bindings
        }
        if (
            attempt is None
            or scope_ids != {request.scope_id}
            or operation.operation_kind
            is not ExpertValidationOperationKind.TASK_EVALUATION_RESERVATION
            or operation.request_record_id != reservation.reservation_id
            or operation.expected_transition_id
            != validation_snapshot.transition.transition_id
            or operation.candidate_id != request.candidate_id
            or reservation.request_id != request.request_id
            or reservation.plan_reservation_operation_id
            != plan_reservation.operation.operation_id
            or reservation.evaluation_plan_id != plan.evaluation_plan_id
            or reservation.mode is not request.mode
            or reservation.authorization_transition_id
            != request.authorization_transition_id
            or reservation.authorization_transition_id
            != validation_snapshot.transition.transition_id
            or reservation.authorization_state_id != request.authorization_state_id
            or reservation.authorization_state_id
            != validation_snapshot.state.validation_state_id
            or reservation.validation_attempt_id != request.validation_attempt_id
            or reservation.validation_attempt_id != attempt.validation_attempt_id
            or reservation.candidate_id != request.candidate_id
            or reservation.candidate_tree_hash != request.candidate_tree_hash
            or reservation.scope_contract_id != request.scope_contract_id
            or reservation.scope_id != request.scope_id
            or reservation.current_release_observation_id != observation.observation_id
            or reservation.observed_current_release_id != request.source_base_release_id
            or reservation.observed_current_release_id != observation.release_id
            or observation.scope_id != request.scope_id
        ):
            raise TaskEvaluationReservationError(
                "task evaluation reservation snapshot authority is inconsistent"
            )
