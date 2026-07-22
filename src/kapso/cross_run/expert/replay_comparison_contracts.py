"""Durable factual comparison contracts for completed expert source replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import (
    EvaluationFingerprint,
    ObjectiveDirection,
    StrictContract,
    TaskEvaluatorMetricComparisonBinding,
)
from kapso.cross_run.expert.replay_protocol_contracts import (
    ExpertSourceReplayProtocolError,
    TaskEvaluatorFingerprintResult,
    require_finite_float,
    stable_arithmetic_mean,
)


def normalized_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


@dataclass(frozen=True)
class ExpertSourceReplayFingerprintComparison(StrictContract):
    evaluation_fingerprint: EvaluationFingerprint
    metric_comparison_binding: TaskEvaluatorMetricComparisonBinding
    control_result: TaskEvaluatorFingerprintResult
    candidate_result: TaskEvaluatorFingerprintResult
    aggregate_raw_delta: float
    aggregate_direction_aligned_delta: float
    aggregate_normalized_effect: float

    def _validate(self) -> None:
        fingerprint = self.evaluation_fingerprint
        binding = self.metric_comparison_binding
        fingerprint_id = fingerprint.evaluation_fingerprint_id
        expected_replicates = set(fingerprint.seed_or_replicate_ids)
        if (
            binding.evaluator_fingerprint != fingerprint.evaluator_fingerprint
            or binding.metric_name != fingerprint.metric_name
            or binding.objective_direction is not fingerprint.objective_direction
            or self.control_result.evaluation_fingerprint_id != fingerprint_id
            or self.candidate_result.evaluation_fingerprint_id != fingerprint_id
            or set(self.control_result.replicate_values) != expected_replicates
            or set(self.candidate_result.replicate_values) != expected_replicates
        ):
            raise ExpertSourceReplayProtocolError(
                "source replay comparison authority differs from its fingerprint"
            )
        for value, name in (
            (self.aggregate_raw_delta, "source replay aggregate raw delta"),
            (
                self.aggregate_direction_aligned_delta,
                "source replay aggregate direction-aligned delta",
            ),
            (
                self.aggregate_normalized_effect,
                "source replay aggregate normalized effect",
            ),
        ):
            require_finite_float(value, name)
            if value == 0.0 and math.copysign(1.0, value) < 0.0:
                raise ExpertSourceReplayProtocolError(
                    f"{name} must normalize signed zero"
                )
        expected_raw_delta = normalized_zero(
            self.candidate_result.aggregate_value - self.control_result.aggregate_value
        )
        expected_direction_aligned_delta = normalized_zero(
            expected_raw_delta
            if fingerprint.objective_direction is ObjectiveDirection.MAXIMIZE
            else -expected_raw_delta
        )
        expected_normalized_effect = normalized_zero(
            expected_direction_aligned_delta / binding.comparison_scale
        )
        if not all(
            math.isfinite(value)
            for value in (
                expected_raw_delta,
                expected_direction_aligned_delta,
                expected_normalized_effect,
            )
        ):
            raise ExpertSourceReplayProtocolError(
                "source replay comparison arithmetic is not finite"
            )
        if (
            self.aggregate_raw_delta != expected_raw_delta
            or self.aggregate_direction_aligned_delta
            != expected_direction_aligned_delta
            or self.aggregate_normalized_effect != expected_normalized_effect
        ):
            raise ExpertSourceReplayProtocolError(
                "source replay comparison arithmetic differs from accepted results"
            )

    def validate_aggregates(self, aggregate_recomputation_tolerance: float) -> None:
        if (
            type(aggregate_recomputation_tolerance) is not float
            or not math.isfinite(aggregate_recomputation_tolerance)
            or aggregate_recomputation_tolerance < 0.0
        ):
            raise ExpertSourceReplayProtocolError(
                "source replay aggregate recomputation tolerance is invalid"
            )
        fingerprint = self.evaluation_fingerprint
        for result in (self.control_result, self.candidate_result):
            recomputed = stable_arithmetic_mean(
                tuple(
                    result.replicate_values[replicate_id]
                    for replicate_id in fingerprint.seed_or_replicate_ids
                )
            )
            if not math.isclose(
                result.aggregate_value,
                recomputed,
                rel_tol=0.0,
                abs_tol=aggregate_recomputation_tolerance,
            ):
                raise ExpertSourceReplayProtocolError(
                    "source replay accepted aggregate differs from its replicates"
                )


@dataclass(frozen=True)
class ExpertSourceReplayCaseComparison(StrictContract):
    execution_case_id: str
    score_of_record_fingerprint_id: str
    control_result_accepted_event_id: str
    candidate_result_accepted_event_id: str
    fingerprint_comparisons: tuple[ExpertSourceReplayFingerprintComparison, ...]

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.execution_case_id,
                "expert-source-replay-execution-case",
                "source replay comparison execution_case_id",
            ),
            (
                self.score_of_record_fingerprint_id,
                "evaluation-fingerprint",
                "source replay comparison score_of_record_fingerprint_id",
            ),
            (
                self.control_result_accepted_event_id,
                "source-replay-execution-journal-event",
                "source replay comparison control accepted event",
            ),
            (
                self.candidate_result_accepted_event_id,
                "source-replay-execution-journal-event",
                "source replay comparison candidate accepted event",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayProtocolError(
                    f"{name} uses the wrong namespace"
                )
        fingerprint_ids = tuple(
            comparison.evaluation_fingerprint.evaluation_fingerprint_id
            for comparison in self.fingerprint_comparisons
        )
        if (
            not fingerprint_ids
            or fingerprint_ids != tuple(sorted(set(fingerprint_ids)))
            or self.score_of_record_fingerprint_id not in fingerprint_ids
            or self.control_result_accepted_event_id
            == self.candidate_result_accepted_event_id
        ):
            raise ExpertSourceReplayProtocolError(
                "source replay case comparisons are incomplete or noncanonical"
            )


@dataclass(frozen=True)
class ExpertSourceReplayPairedComparisonReceipt(StrictContract):
    paired_comparison_receipt_id: str
    reservation_id: str
    execution_request_id: str
    aggregate_recomputation_tolerance: float
    execution_journal_event_ids: tuple[str, ...]
    reservation_dependency_ids: tuple[str, ...]
    request_dependency_ids: tuple[str, ...]
    case_comparisons: tuple[ExpertSourceReplayCaseComparison, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-paired-comparison"
    IDENTITY_FIELD: ClassVar[str] = "paired_comparison_receipt_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.reservation_id,
                "expert-source-replay-execution-reservation",
                "source replay comparison reservation_id",
            ),
            (
                self.execution_request_id,
                "expert-source-replay-execution-request",
                "source replay comparison execution_request_id",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayProtocolError(
                    f"{name} uses the wrong namespace"
                )
        tolerance = self.aggregate_recomputation_tolerance
        if (
            type(tolerance) is not float
            or not math.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ExpertSourceReplayProtocolError(
                "source replay aggregate recomputation tolerance is invalid"
            )
        if not self.execution_journal_event_ids or len(
            self.execution_journal_event_ids
        ) != len(set(self.execution_journal_event_ids)):
            raise ExpertSourceReplayProtocolError(
                "source replay comparison journal event IDs must be non-empty and unique"
            )
        for event_id in self.execution_journal_event_ids:
            require_content_id(event_id, "source replay comparison journal event")
            if event_id.split(":sha256:", 1)[0] != (
                "source-replay-execution-journal-event"
            ):
                raise ExpertSourceReplayProtocolError(
                    "source replay comparison journal event uses the wrong namespace"
                )
        case_ids = tuple(
            comparison.execution_case_id for comparison in self.case_comparisons
        )
        if (
            not case_ids
            or case_ids != tuple(sorted(set(case_ids)))
            or len(self.execution_journal_event_ids) != 8 * len(case_ids)
        ):
            raise ExpertSourceReplayProtocolError(
                "source replay receipt cases or journal cardinality are noncanonical"
            )
        accepted_event_ids = tuple(
            event_id
            for comparison in self.case_comparisons
            for event_id in (
                comparison.control_result_accepted_event_id,
                comparison.candidate_result_accepted_event_id,
            )
        )
        if len(accepted_event_ids) != len(set(accepted_event_ids)) or not set(
            accepted_event_ids
        ).issubset(self.execution_journal_event_ids):
            raise ExpertSourceReplayProtocolError(
                "source replay receipt accepted events differ from its journal"
            )
        for dependency_ids, name in (
            (
                self.reservation_dependency_ids,
                "source replay reservation dependencies",
            ),
            (
                self.request_dependency_ids,
                "source replay request dependencies",
            ),
            (self.exact_dependency_ids, "source replay exact dependencies"),
        ):
            if not dependency_ids or dependency_ids != tuple(
                sorted(set(dependency_ids))
            ):
                raise ExpertSourceReplayProtocolError(
                    f"{name} must be non-empty, sorted, and unique"
                )
            for dependency_id in dependency_ids:
                require_content_id(dependency_id, name)
        if self.execution_request_id not in self.reservation_dependency_ids:
            raise ExpertSourceReplayProtocolError(
                "source replay reservation dependencies omit the execution request"
            )
        expected_dependencies = {
            self.reservation_id,
            self.execution_request_id,
            *self.reservation_dependency_ids,
            *self.request_dependency_ids,
            *self.execution_journal_event_ids,
        }
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertSourceReplayProtocolError(
                "source replay comparison dependency closure is not exact"
            )
        for case_comparison in self.case_comparisons:
            for fingerprint_comparison in case_comparison.fingerprint_comparisons:
                fingerprint_comparison.validate_aggregates(tolerance)
