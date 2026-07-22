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
    ExpertPromotionState,
    ExpertValidationAttempt,
    ExpertValidationStage,
    StrictContract,
)
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixEvaluationPlan,
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
        | ExpertAutomatedReviewStageResultRecord,
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
            or plan.parent_release_id != attempt.parent_release_id
            or plan.validation_policy_id != attempt.validation_policy_id
            or plan.configuration_fingerprint != attempt.configuration_fingerprint
            or state.validation_attempt_id != attempt.validation_attempt_id
            or state.promotion_state is not ExpertPromotionState.VALIDATING
            or state.next_stage is not ExpertValidationStage.RELEASE_MATRIX
        ):
            raise ExpertValidationSnapshotError(
                "release matrix plan reservation authority is inconsistent"
            )
