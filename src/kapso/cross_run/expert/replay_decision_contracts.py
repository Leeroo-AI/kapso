"""Durable policy decision contracts for completed expert source replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import ExpertEvaluatorOutcome, StrictContract


class ExpertSourceReplayDecisionError(ValueError):
    """A factual source replay receipt cannot produce an exact stage decision."""


@dataclass(frozen=True)
class ExpertSourceReplayComparisonReference(StrictContract):
    execution_case_id: str
    evaluation_fingerprint_id: str

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.execution_case_id,
                "expert-source-replay-execution-case",
                "source replay decision execution_case_id",
            ),
            (
                self.evaluation_fingerprint_id,
                "evaluation-fingerprint",
                "source replay decision evaluation_fingerprint_id",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayDecisionError(
                    f"{name} uses the wrong namespace"
                )


@dataclass(frozen=True)
class ExpertSourceReplayStageDecision(StrictContract):
    source_replay_stage_decision_id: str
    paired_comparison_receipt_id: str
    validation_policy_id: str
    decision_policy_version: str
    outcome: ExpertEvaluatorOutcome
    hard_regression_comparisons: tuple[ExpertSourceReplayComparisonReference, ...]
    paired_comparison_dependency_ids: tuple[str, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-stage-decision"
    IDENTITY_FIELD: ClassVar[str] = "source_replay_stage_decision_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.paired_comparison_receipt_id,
                "expert-source-replay-paired-comparison",
                "source replay decision paired_comparison_receipt_id",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "source replay decision validation_policy_id",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayDecisionError(
                    f"{name} uses the wrong namespace"
                )
        require_identifier(
            self.decision_policy_version,
            "source replay decision policy version",
        )
        comparison_keys = tuple(
            (reference.execution_case_id, reference.evaluation_fingerprint_id)
            for reference in self.hard_regression_comparisons
        )
        if comparison_keys != tuple(sorted(set(comparison_keys))):
            raise ExpertSourceReplayDecisionError(
                "source replay hard regressions must be sorted and unique"
            )
        expected_outcome = (
            ExpertEvaluatorOutcome.CANDIDATE_FAILED
            if comparison_keys
            else ExpertEvaluatorOutcome.PASSED
        )
        if self.outcome is not expected_outcome:
            raise ExpertSourceReplayDecisionError(
                "source replay decision outcome differs from its hard regressions"
            )
        for dependency_ids, name in (
            (
                self.paired_comparison_dependency_ids,
                "source replay decision receipt dependencies",
            ),
            (self.exact_dependency_ids, "source replay decision exact dependencies"),
        ):
            if not dependency_ids or dependency_ids != tuple(
                sorted(set(dependency_ids))
            ):
                raise ExpertSourceReplayDecisionError(
                    f"{name} must be non-empty, sorted, and unique"
                )
            for dependency_id in dependency_ids:
                require_content_id(dependency_id, name)
        expected_dependencies = {
            self.paired_comparison_receipt_id,
            self.validation_policy_id,
            *self.paired_comparison_dependency_ids,
        }
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertSourceReplayDecisionError(
                "source replay decision dependency closure is not exact"
            )
