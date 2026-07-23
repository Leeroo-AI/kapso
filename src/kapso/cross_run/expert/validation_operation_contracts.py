"""Content-addressed operations bound to expert-validation transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import StrictContract


class ExpertValidationOperationKind(str, Enum):
    START = "start"
    EVALUATOR_RESULT = "evaluator_result"
    SOURCE_REPLAY_RESERVATION = "source_replay_reservation"
    SOURCE_REPLAY_STAGE_RESULT = "source_replay_stage_result"
    AUTOMATED_REVIEW_STAGE_RESULT = "automated_review_stage_result"
    RELEASE_MATRIX_PLAN_RESERVATION = "release_matrix_plan_reservation"
    TASK_EVALUATION_RESERVATION = "task_evaluation_reservation"
    RELEASE_MATRIX_STAGE_RESULT = "release_matrix_stage_result"
    PUBLICATION_ELIGIBILITY_STAGE_RESULT = "publication_eligibility_stage_result"
    RELEASE_ACTIVATION = "release_activation"
    RELEASE_REVOCATION = "release_revocation"
    AUTHORITY_INVALIDATION = "authority_invalidation"


@dataclass(frozen=True)
class ExpertValidationOperation(StrictContract):
    operation_id: str
    operation_kind: ExpertValidationOperationKind
    candidate_id: str
    expected_transition_id: str | None
    request_record_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-validation-operation"
    IDENTITY_FIELD: ClassVar[str] = "operation_id"

    def _validate(self) -> None:
        require_content_id(self.candidate_id, "operation candidate_id")
        if self.expected_transition_id is not None:
            require_content_id(
                self.expected_transition_id,
                "operation expected_transition_id",
            )
        require_content_id(self.request_record_id, "operation request_record_id")
