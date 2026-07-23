"""Typed Pareto decisions derived from factual expert release-matrix reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import ExpertValidationTrack, StrictContract
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode


class ExpertReleaseMatrixDecisionError(ValueError):
    """A release-matrix promotion decision is structurally inconsistent."""


class ExpertReleaseMatrixReplicateClassification(str, Enum):
    """Direction-aligned classification of one precommitted replicate."""

    GAIN = "gain"
    MATERIAL_REGRESSION = "material_regression"
    TIE = "tie"


class ExpertReleaseMatrixDecisionOutcome(str, Enum):
    """Terminal outcome of the release-matrix Pareto policy."""

    FAILED = "failed"
    PARETO_RETAINED = "pareto_retained"
    APPROVED = "approved"


class ExpertReleaseMatrixDecisionReason(str, Enum):
    """Exclusive structural reason for one Pareto outcome."""

    HARD_REGRESSION = "hard_regression"
    UNDERPOWERED_EVIDENCE = "underpowered_evidence"
    BOOTSTRAP_STANDALONE_COVERAGE = "bootstrap_standalone_coverage"
    CONFIRMED_BENEFIT = "confirmed_benefit"
    MECHANICAL_NON_REGRESSION = "mechanical_non_regression"
    GAIN_REGRESSION_TRADEOFF = "gain_regression_tradeoff"
    NOISY_GAIN = "noisy_gain"
    UNCOMPENSATED_REGRESSION = "uncompensated_regression"
    NO_BENEFIT = "no_benefit"


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertReleaseMatrixDecisionError(f"{name} uses the wrong namespace")


def _require_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ExpertReleaseMatrixDecisionError(f"{name} must be a sha256 digest")


def _require_sorted_identifiers(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ExpertReleaseMatrixDecisionError(f"{name} must be sorted and unique")
    for value in values:
        require_identifier(value, name)


def _require_sorted_content_ids(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ExpertReleaseMatrixDecisionError(
            f"{name} must be non-empty, sorted, and unique"
        )
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class ExpertReleaseMatrixReplicateAssessment(StrictContract):
    """One exact direction-aligned replicate assessment."""

    evaluation_cell_id: str
    comparison_dimension_id: str
    replicate_id: str
    normalized_effect: float
    classification: ExpertReleaseMatrixReplicateClassification
    hard_regression: bool

    def _validate(self) -> None:
        _require_namespaced_id(
            self.evaluation_cell_id,
            "expert-release-matrix-evaluation-cell",
            "release matrix assessment evaluation_cell_id",
        )
        require_identifier(
            self.comparison_dimension_id,
            "release matrix assessment comparison_dimension_id",
        )
        require_identifier(
            self.replicate_id,
            "release matrix assessment replicate_id",
        )
        if not math.isfinite(self.normalized_effect):
            raise ExpertReleaseMatrixDecisionError(
                "release matrix assessment normalized_effect must be finite"
            )
        if (
            self.normalized_effect == 0.0
            and math.copysign(1.0, self.normalized_effect) < 0.0
        ):
            raise ExpertReleaseMatrixDecisionError(
                "release matrix assessment normalized_effect must normalize signed zero"
            )
        if (
            self.classification is ExpertReleaseMatrixReplicateClassification.GAIN
            and self.normalized_effect <= 0.0
        ) or (
            self.classification
            is ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION
            and self.normalized_effect >= 0.0
        ):
            raise ExpertReleaseMatrixDecisionError(
                "release matrix assessment classification contradicts effect direction"
            )
        if self.hard_regression and self.normalized_effect >= 0.0:
            raise ExpertReleaseMatrixDecisionError(
                "release matrix hard regression must have a negative effect"
            )

    @property
    def canonical_key(self) -> tuple[str, str]:
        return self.evaluation_cell_id, self.replicate_id


@dataclass(frozen=True)
class ExpertReleaseMatrixPromotionDecision(StrictContract):
    """Pareto decision over one accepted release-matrix stage result."""

    promotion_decision_id: str
    release_matrix_stage_result_id: str
    release_matrix_report_id: str
    plan_reservation_operation_id: str
    validation_attempt_id: str
    validation_policy_id: str
    promotion_policy_version: str
    configuration_fingerprint: str
    mode: ExpertReleaseMatrixMode
    validation_track: ExpertValidationTrack
    outcome: ExpertReleaseMatrixDecisionOutcome
    reason: ExpertReleaseMatrixDecisionReason
    replicate_assessments: tuple[ExpertReleaseMatrixReplicateAssessment, ...]
    underpowered_dimension_ids: tuple[str, ...]
    confirmed_benefit_dimension_ids: tuple[str, ...]
    release_matrix_stage_dependency_ids: tuple[str, ...]
    attempt_dependency_ids: tuple[str, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-promotion-decision"
    IDENTITY_FIELD: ClassVar[str] = "promotion_decision_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.release_matrix_stage_result_id,
                "expert-release-matrix-stage-result",
                "promotion decision release_matrix_stage_result_id",
            ),
            (
                self.release_matrix_report_id,
                "expert-release-matrix-report",
                "promotion decision release_matrix_report_id",
            ),
            (
                self.plan_reservation_operation_id,
                "expert-validation-operation",
                "promotion decision plan_reservation_operation_id",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "promotion decision validation_attempt_id",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "promotion decision validation_policy_id",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        require_identifier(
            self.promotion_policy_version,
            "promotion decision policy version",
        )
        _require_digest(
            self.configuration_fingerprint,
            "promotion decision configuration_fingerprint",
        )
        assessment_keys = tuple(
            assessment.canonical_key for assessment in self.replicate_assessments
        )
        if assessment_keys != tuple(sorted(set(assessment_keys))):
            raise ExpertReleaseMatrixDecisionError(
                "promotion decision replicate assessments must be canonical and unique"
            )
        _require_sorted_identifiers(
            self.underpowered_dimension_ids,
            "promotion decision underpowered dimensions",
        )
        _require_sorted_identifiers(
            self.confirmed_benefit_dimension_ids,
            "promotion decision confirmed benefit dimensions",
        )
        if set(self.underpowered_dimension_ids) & set(
            self.confirmed_benefit_dimension_ids
        ):
            raise ExpertReleaseMatrixDecisionError(
                "promotion decision dimension projections must be disjoint"
            )
        for values, name in (
            (
                self.release_matrix_stage_dependency_ids,
                "promotion decision release matrix stage dependencies",
            ),
            (
                self.attempt_dependency_ids,
                "promotion decision attempt dependencies",
            ),
            (
                self.exact_dependency_ids,
                "promotion decision exact dependencies",
            ),
        ):
            _require_sorted_content_ids(values, name)
        required_stage_dependencies = {
            self.release_matrix_report_id,
            self.plan_reservation_operation_id,
            self.validation_attempt_id,
            self.validation_policy_id,
        }
        if not required_stage_dependencies.issubset(
            self.release_matrix_stage_dependency_ids
        ):
            raise ExpertReleaseMatrixDecisionError(
                "promotion decision release matrix stage dependencies omit required authority"
            )
        expected_dependencies = {
            self.release_matrix_stage_result_id,
            self.release_matrix_report_id,
            self.plan_reservation_operation_id,
            self.validation_attempt_id,
            self.validation_policy_id,
            *self.release_matrix_stage_dependency_ids,
            *self.attempt_dependency_ids,
        }
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertReleaseMatrixDecisionError(
                "promotion decision dependency closure is not exact"
            )
        gain_dimension_ids = {
            assessment.comparison_dimension_id
            for assessment in self.replicate_assessments
            if assessment.classification
            is ExpertReleaseMatrixReplicateClassification.GAIN
        }
        material_regression_dimension_ids = {
            assessment.comparison_dimension_id
            for assessment in self.replicate_assessments
            if assessment.classification
            is ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION
        }
        has_hard_regression = any(
            assessment.hard_regression for assessment in self.replicate_assessments
        )
        if not set(self.confirmed_benefit_dimension_ids).issubset(gain_dimension_ids):
            raise ExpertReleaseMatrixDecisionError(
                "confirmed benefit dimensions must be gain dimensions"
            )
        if self.mode is ExpertReleaseMatrixMode.BOOTSTRAP:
            self._validate_bootstrap_decision()
            return
        self._validate_control_comparison_decision(
            has_hard_regression=has_hard_regression,
            gain_dimension_ids=gain_dimension_ids,
            material_regression_dimension_ids=material_regression_dimension_ids,
        )

    def _validate_bootstrap_decision(self) -> None:
        if (
            self.validation_track is not ExpertValidationTrack.REPOSITORY_ARCHITECTURE
            or self.replicate_assessments
            or self.confirmed_benefit_dimension_ids
        ):
            raise ExpertReleaseMatrixDecisionError(
                "bootstrap decisions require standalone repository-architecture evidence"
            )
        if (
            self.reason
            is ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE
        ):
            if (
                self.outcome is not ExpertReleaseMatrixDecisionOutcome.APPROVED
                or self.underpowered_dimension_ids
            ):
                raise ExpertReleaseMatrixDecisionError(
                    "bootstrap standalone coverage requires an approved powered decision"
                )
            return
        if self.reason is ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE:
            if (
                self.outcome is not ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED
                or not self.underpowered_dimension_ids
            ):
                raise ExpertReleaseMatrixDecisionError(
                    "bootstrap underpowered evidence requires retained missing power"
                )
            return
        raise ExpertReleaseMatrixDecisionError(
            "bootstrap decision uses an unsupported reason"
        )

    def _validate_control_comparison_decision(
        self,
        *,
        has_hard_regression: bool,
        gain_dimension_ids: set[str],
        material_regression_dimension_ids: set[str],
    ) -> None:
        if not self.replicate_assessments:
            raise ExpertReleaseMatrixDecisionError(
                "control comparison decisions require replicate assessments"
            )
        if (
            self.reason
            is ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE
        ):
            raise ExpertReleaseMatrixDecisionError(
                "control comparison decision uses an unsupported reason"
            )
        if self.reason is ExpertReleaseMatrixDecisionReason.HARD_REGRESSION:
            if (
                self.outcome is not ExpertReleaseMatrixDecisionOutcome.FAILED
                or not has_hard_regression
                or self.underpowered_dimension_ids
                or self.confirmed_benefit_dimension_ids
            ):
                raise ExpertReleaseMatrixDecisionError(
                    "hard regression requires a failed hard-gated decision"
                )
            return
        if has_hard_regression:
            raise ExpertReleaseMatrixDecisionError(
                "non-hard-regression reason cannot retain a hard regression"
            )
        if self.reason is ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE:
            if (
                self.outcome is not ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED
                or not self.underpowered_dimension_ids
            ):
                raise ExpertReleaseMatrixDecisionError(
                    "underpowered evidence requires a retained underpowered decision"
                )
            return
        if self.underpowered_dimension_ids:
            raise ExpertReleaseMatrixDecisionError(
                "powered control decision cannot name underpowered dimensions"
            )
        if self.reason is ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT:
            valid = (
                self.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
                and not material_regression_dimension_ids
                and bool(self.confirmed_benefit_dimension_ids)
            )
        elif self.reason is ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION:
            valid = (
                self.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
                and self.validation_track
                is ExpertValidationTrack.MECHANICAL_GENERAL_FIX
                and not material_regression_dimension_ids
                and not self.confirmed_benefit_dimension_ids
            )
        elif self.reason is ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF:
            valid = (
                self.outcome is ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED
                and bool(gain_dimension_ids)
                and bool(material_regression_dimension_ids)
            )
        elif self.reason is ExpertReleaseMatrixDecisionReason.NOISY_GAIN:
            valid = (
                self.outcome is ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED
                and bool(gain_dimension_ids)
                and not material_regression_dimension_ids
                and not self.confirmed_benefit_dimension_ids
            )
        elif self.reason is ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION:
            valid = (
                self.outcome is ExpertReleaseMatrixDecisionOutcome.FAILED
                and bool(material_regression_dimension_ids)
                and not gain_dimension_ids
                and not self.confirmed_benefit_dimension_ids
            )
        elif self.reason is ExpertReleaseMatrixDecisionReason.NO_BENEFIT:
            valid = (
                self.outcome is ExpertReleaseMatrixDecisionOutcome.FAILED
                and not gain_dimension_ids
                and not material_regression_dimension_ids
                and not self.confirmed_benefit_dimension_ids
            )
        else:
            raise ExpertReleaseMatrixDecisionError(
                "control comparison decision uses an unsupported reason"
            )
        if not valid:
            raise ExpertReleaseMatrixDecisionError(
                "control comparison decision contradicts its reason"
            )
