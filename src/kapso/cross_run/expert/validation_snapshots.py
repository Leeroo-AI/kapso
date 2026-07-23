"""Shared immutable expert-validation transition and snapshot authorities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import (
    ContractValidationError,
    ExpertCandidateValidationState,
    ExpertEvaluatorResultRecord,
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertSourceReplayExecutionRequest,
    ExpertSourceReplayExecutionReservation,
    ExpertValidationAttempt,
    ExpertValidationStage,
    StrictContract,
)
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixEvaluationPlan,
    ExpertReleaseMatrixMode,
    ExpertReleaseMatrixProvenanceKind,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.expert.promotion_authority_contracts import (
    ExpertPublicationEligibilityStageResultRecord,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayStageResultRecord,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewStageResultRecord,
)
from kapso.cross_run.expert.validation import ExpertValidationPredecessor
from kapso.cross_run.expert.validation_operation_contracts import (
    ExpertValidationOperation,
    ExpertValidationOperationKind,
)


class ExpertValidationSnapshotError(ValueError):
    """A validation snapshot does not close over one exact transition."""


@dataclass(frozen=True)
class ExpertValidationTransition(StrictContract):
    transition_id: str
    candidate_id: str
    candidate_tree_hash: str
    transition_number: int
    predecessor_transition_id: str | None
    predecessor_state_id: str | None
    target_state_id: str
    latest_attempt_id: str | None
    operation_id: str
    validation_policy_id: str
    configuration_fingerprint: str
    eligibility_decision_id: str | None
    created_attempt_id: str | None
    accepted_stage_result_record_ids: tuple[str, ...]
    transition_stage_result_record_id: str | None
    transition_authority_invalidation_id: str | None
    transition_release_use_block_decision_id: str | None
    transition_release_activation_receipt_id: str | None
    transition_release_revocation_receipt_id: str | None

    CONTENT_NAMESPACE: ClassVar[str] = "expert-validation-transition"
    IDENTITY_FIELD: ClassVar[str] = "transition_id"

    def _validate(self) -> None:
        for value, name in (
            (self.candidate_id, "transition candidate_id"),
            (self.target_state_id, "transition target_state_id"),
            (self.operation_id, "transition operation_id"),
            (self.validation_policy_id, "transition validation_policy_id"),
        ):
            require_content_id(value, name)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.candidate_tree_hash) is None:
            raise ContractValidationError("transition candidate tree hash is invalid")
        if self.transition_number <= 0:
            raise ContractValidationError("transition number must be positive")
        first = self.transition_number == 1
        if (self.predecessor_transition_id is None) != (
            self.predecessor_state_id is None
        ) or first != (self.predecessor_transition_id is None):
            raise ContractValidationError(
                "only the first transition may omit both predecessors"
            )
        for value, name in (
            (self.predecessor_transition_id, "predecessor_transition_id"),
            (self.predecessor_state_id, "predecessor_state_id"),
            (self.latest_attempt_id, "latest_attempt_id"),
            (self.eligibility_decision_id, "eligibility_decision_id"),
            (self.created_attempt_id, "created_attempt_id"),
            (
                self.transition_stage_result_record_id,
                "transition_stage_result_record_id",
            ),
            (
                self.transition_authority_invalidation_id,
                "transition_authority_invalidation_id",
            ),
            (
                self.transition_release_use_block_decision_id,
                "transition_release_use_block_decision_id",
            ),
            (
                self.transition_release_activation_receipt_id,
                "transition_release_activation_receipt_id",
            ),
            (
                self.transition_release_revocation_receipt_id,
                "transition_release_revocation_receipt_id",
            ),
        ):
            if value is not None:
                require_content_id(value, name)
        if (
            re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                self.configuration_fingerprint,
            )
            is None
        ):
            raise ContractValidationError(
                "transition configuration fingerprint is invalid"
            )
        if len(self.accepted_stage_result_record_ids) != len(
            set(self.accepted_stage_result_record_ids)
        ):
            raise ContractValidationError(
                "accepted stage result records must be unique"
            )
        for result_record_id in self.accepted_stage_result_record_ids:
            require_content_id(
                result_record_id,
                "accepted_stage_result_record_ids",
            )
        request_count = sum(
            value is not None
            for value in (
                self.eligibility_decision_id,
                self.transition_stage_result_record_id,
                self.transition_authority_invalidation_id,
                self.transition_release_use_block_decision_id,
                self.transition_release_activation_receipt_id,
                self.transition_release_revocation_receipt_id,
            )
        )
        if request_count != 1:
            raise ContractValidationError(
                "transition must contain exactly one start, result, or invalidation request"
            )
        start = self.eligibility_decision_id is not None
        if not start and self.created_attempt_id is not None:
            raise ContractValidationError(
                "only a start transition may create a validation attempt"
            )
        if self.created_attempt_id is not None and (
            self.latest_attempt_id != self.created_attempt_id
        ):
            raise ContractValidationError(
                "created validation attempt must become the latest attempt"
            )


@dataclass(frozen=True)
class ExpertValidationSnapshot:
    transition: ExpertValidationTransition
    state: ExpertCandidateValidationState
    latest_attempt: ExpertValidationAttempt | None
    accepted_stage_results: tuple[
        ExpertEvaluatorResultRecord
        | ExpertSourceReplayStageResultRecord
        | ExpertAutomatedReviewStageResultRecord
        | ExpertReleaseMatrixStageResultRecord
        | ExpertPublicationEligibilityStageResultRecord,
        ...,
    ]

    @property
    def predecessor(self) -> ExpertValidationPredecessor:
        return ExpertValidationPredecessor(
            latest_attempt=self.latest_attempt,
            state=self.state,
        )


@dataclass(frozen=True)
class ExpertReleaseMatrixPlanReservationSnapshot:
    """A plan bound to one unchanged release-matrix validation head."""

    operation: ExpertValidationOperation
    evaluation_plan: ExpertReleaseMatrixEvaluationPlan
    snapshot: ExpertValidationSnapshot

    def __post_init__(self) -> None:
        if (
            type(self.operation) is not ExpertValidationOperation
            or type(self.evaluation_plan) is not ExpertReleaseMatrixEvaluationPlan
            or type(self.snapshot) is not ExpertValidationSnapshot
        ):
            raise ExpertValidationSnapshotError(
                "release matrix plan reservation is not typed"
            )
        attempt = self.snapshot.latest_attempt
        transition = self.snapshot.transition
        state = self.snapshot.state
        plan = self.evaluation_plan
        operation = self.operation
        if (
            attempt is None
            or operation.operation_kind
            is not ExpertValidationOperationKind.RELEASE_MATRIX_PLAN_RESERVATION
            or operation.request_record_id != plan.evaluation_plan_id
            or operation.expected_transition_id != transition.transition_id
            or operation.candidate_id != plan.candidate_id
            or plan.validation_attempt_id != attempt.validation_attempt_id
            or plan.candidate_id != state.candidate_id
            or plan.candidate_id != transition.candidate_id
            or plan.candidate_tree_hash != state.candidate_tree_hash
            or plan.candidate_tree_hash != transition.candidate_tree_hash
            or plan.candidate_commit_record_id != attempt.candidate_commit_record_id
            or plan.scope_contract_id != attempt.scope_contract_id
            or plan.source_base_release_id != attempt.source_base_release_id
            or plan.validation_policy_id != attempt.validation_policy_id
            or plan.configuration_fingerprint != attempt.configuration_fingerprint
            or state.validation_attempt_id != attempt.validation_attempt_id
            or state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.next_stage is not ExpertValidationStage.RELEASE_MATRIX
        ):
            raise ExpertValidationSnapshotError(
                "release matrix plan reservation authority is inconsistent"
            )


@dataclass(frozen=True)
class ExpertReleaseMatrixSourceEvidenceSnapshot:
    """Accepted source facts reopened under one unchanged matrix-plan head."""

    plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot
    stage_result: ExpertSourceReplayStageResultRecord
    reservation: ExpertSourceReplayExecutionReservation
    request: ExpertSourceReplayExecutionRequest

    def __post_init__(self) -> None:
        if (
            type(self.plan_reservation)
            is not ExpertReleaseMatrixPlanReservationSnapshot
            or type(self.stage_result) is not ExpertSourceReplayStageResultRecord
            or not isinstance(
                self.reservation,
                ExpertSourceReplayExecutionReservation,
            )
            or not isinstance(self.request, ExpertSourceReplayExecutionRequest)
        ):
            raise ExpertValidationSnapshotError(
                "release matrix source evidence snapshot is not typed"
            )
        plan = self.plan_reservation.evaluation_plan
        attempt = self.plan_reservation.snapshot.latest_attempt
        accepted_source_results = tuple(
            result
            for result in self.plan_reservation.snapshot.accepted_stage_results
            if type(result) is ExpertSourceReplayStageResultRecord
        )
        source_provenances = tuple(
            provenance
            for provenance in plan.provenance_bindings
            if provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
        )
        if (
            attempt is None
            or plan.mode is not ExpertReleaseMatrixMode.CONTROL_COMPARISON
            or accepted_source_results != (self.stage_result,)
            or self.stage_result.outcome is not ExpertEvaluatorOutcome.PASSED
            or not source_provenances
            or {
                provenance.source_replay_stage_result_id
                for provenance in source_provenances
            }
            != {self.stage_result.stage_result_record_id}
            or {
                provenance.paired_comparison_receipt_id
                for provenance in source_provenances
            }
            != {
                self.stage_result.paired_comparison_receipt.paired_comparison_receipt_id
            }
            or self.stage_result.validation_attempt_id != plan.validation_attempt_id
            or self.stage_result.candidate_id != plan.candidate_id
            or self.stage_result.candidate_tree_hash != plan.candidate_tree_hash
            or self.stage_result.validation_policy_id != plan.validation_policy_id
            or self.stage_result.configuration_fingerprint
            != plan.configuration_fingerprint
            or self.stage_result.reservation_id != self.reservation.reservation_id
            or self.stage_result.execution_request_id
            != self.request.execution_request_id
            or self.stage_result.paired_comparison_receipt.reservation_id
            != self.reservation.reservation_id
            or self.stage_result.paired_comparison_receipt.execution_request_id
            != self.request.execution_request_id
            or self.reservation.execution_request_id
            != self.request.execution_request_id
            or self.reservation.validation_attempt_id
            != self.request.validation_attempt_id
            or self.reservation.candidate_id != self.request.candidate_id
            or self.reservation.candidate_tree_hash != self.request.candidate_tree_hash
            or self.reservation.expected_current_release_id
            != self.request.source_base_release_id
            or self.request.validation_attempt_id != plan.validation_attempt_id
            or self.request.candidate_id != plan.candidate_id
            or self.request.candidate_tree_hash != plan.candidate_tree_hash
            or self.request.candidate_commit_record_id
            != plan.candidate_commit_record_id
            or self.request.scope_contract_id != plan.scope_contract_id
            or self.request.source_base_release_id != plan.source_base_release_id
            or self.request.source_base_tree_hash != plan.source_base_tree_hash
            or self.request.validation_policy_id != plan.validation_policy_id
            or self.request.configuration_fingerprint != plan.configuration_fingerprint
        ):
            raise ExpertValidationSnapshotError(
                "release matrix source evidence closure is inconsistent"
            )


@dataclass(frozen=True)
class ExpertPublicationEligibilitySnapshot:
    """Exact accepted release-matrix head awaiting terminal promotion."""

    snapshot: ExpertValidationSnapshot
    release_matrix_stage_result: ExpertReleaseMatrixStageResultRecord

    def __post_init__(self) -> None:
        if (
            type(self.snapshot) is not ExpertValidationSnapshot
            or type(self.release_matrix_stage_result)
            is not ExpertReleaseMatrixStageResultRecord
        ):
            raise ExpertValidationSnapshotError(
                "publication eligibility snapshot is not typed"
            )
        transition = self.snapshot.transition
        state = self.snapshot.state
        attempt = self.snapshot.latest_attempt
        matrix_result = self.release_matrix_stage_result
        if (
            attempt is None
            or state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.next_stage is not ExpertValidationStage.PUBLICATION_ELIGIBILITY
            or state.validation_attempt_id != attempt.validation_attempt_id
            or state.candidate_id != attempt.candidate_id
            or state.candidate_tree_hash != attempt.candidate_tree_hash
            or transition.latest_attempt_id != attempt.validation_attempt_id
            or transition.target_state_id != state.validation_state_id
            or not self.snapshot.accepted_stage_results
            or self.snapshot.accepted_stage_results[-1] != matrix_result
            or not state.accepted_stage_results
            or state.accepted_stage_results[-1].stage
            is not ExpertValidationStage.RELEASE_MATRIX
            or state.accepted_stage_results[-1].stage_result_record_id
            != matrix_result.stage_result_record_id
            or transition.accepted_stage_result_record_ids[-1]
            != matrix_result.stage_result_record_id
            or matrix_result.validation_attempt_id != attempt.validation_attempt_id
            or matrix_result.candidate_id != attempt.candidate_id
            or matrix_result.candidate_tree_hash != attempt.candidate_tree_hash
            or matrix_result.scope_contract_id != attempt.scope_contract_id
            or matrix_result.source_base_release_id != attempt.source_base_release_id
            or matrix_result.validation_policy_id != attempt.validation_policy_id
            or matrix_result.configuration_fingerprint
            != attempt.configuration_fingerprint
        ):
            raise ExpertValidationSnapshotError(
                "publication eligibility snapshot authority is inconsistent"
            )
