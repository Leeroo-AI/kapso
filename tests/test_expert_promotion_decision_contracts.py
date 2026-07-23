from __future__ import annotations

from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import ContractValidationError, ExpertValidationTrack
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
    ExpertReleaseMatrixDecisionReason,
    ExpertReleaseMatrixPromotionDecision,
    ExpertReleaseMatrixReplicateAssessment,
    ExpertReleaseMatrixReplicateClassification,
)


def _id(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


def _assessment(
    label: str,
    dimension_id: str,
    classification: ExpertReleaseMatrixReplicateClassification,
    *,
    hard_regression: bool = False,
) -> ExpertReleaseMatrixReplicateAssessment:
    effect_by_classification = {
        ExpertReleaseMatrixReplicateClassification.GAIN: 0.25,
        ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION: -0.25,
        ExpertReleaseMatrixReplicateClassification.TIE: 0.0,
    }
    return ExpertReleaseMatrixReplicateAssessment(
        evaluation_cell_id=_id(
            "expert-release-matrix-evaluation-cell",
            label,
        ),
        comparison_dimension_id=dimension_id,
        replicate_id=f"replicate-{label}",
        normalized_effect=effect_by_classification[classification],
        classification=classification,
        hard_regression=hard_regression,
    )


def _canonical_assessments(
    *assessments: ExpertReleaseMatrixReplicateAssessment,
) -> tuple[ExpertReleaseMatrixReplicateAssessment, ...]:
    return tuple(sorted(assessments, key=lambda assessment: assessment.canonical_key))


def _base_payload() -> dict[str, object]:
    stage_result_id = _id("expert-release-matrix-stage-result", "stage-result")
    report_id = _id("expert-release-matrix-report", "report")
    plan_operation_id = _id("expert-validation-operation", "plan-reservation")
    attempt_id = _id("expert-validation-attempt", "attempt")
    policy_id = _id("expert-validation-policy", "policy")
    stage_dependency_ids = tuple(
        sorted(
            {
                report_id,
                plan_operation_id,
                attempt_id,
                policy_id,
                _id("evaluation-fingerprint", "stage-evidence"),
            }
        )
    )
    attempt_dependency_ids = (
        _id("expert-candidate", "candidate"),
        _id("expert-scope-contract", "scope"),
    )
    exact_dependency_ids = tuple(
        sorted(
            {
                stage_result_id,
                report_id,
                plan_operation_id,
                attempt_id,
                policy_id,
                *stage_dependency_ids,
                *attempt_dependency_ids,
            }
        )
    )
    return {
        "release_matrix_stage_result_id": stage_result_id,
        "release_matrix_report_id": report_id,
        "plan_reservation_operation_id": plan_operation_id,
        "validation_attempt_id": attempt_id,
        "validation_policy_id": policy_id,
        "promotion_policy_version": "promotion.v1",
        "configuration_fingerprint": tree_or_blob_digest(b"configuration"),
        "mode": ExpertReleaseMatrixMode.CONTROL_COMPARISON,
        "validation_track": ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
        "outcome": ExpertReleaseMatrixDecisionOutcome.FAILED,
        "reason": ExpertReleaseMatrixDecisionReason.NO_BENEFIT,
        "replicate_assessments": _canonical_assessments(
            _assessment(
                "quality-tie",
                "quality",
                ExpertReleaseMatrixReplicateClassification.TIE,
            )
        ),
        "underpowered_dimension_ids": (),
        "confirmed_benefit_dimension_ids": (),
        "release_matrix_stage_dependency_ids": stage_dependency_ids,
        "attempt_dependency_ids": attempt_dependency_ids,
        "exact_dependency_ids": exact_dependency_ids,
    }


def _mint(**changes: object) -> ExpertReleaseMatrixPromotionDecision:
    payload = _base_payload()
    payload.update(changes)
    return ExpertReleaseMatrixPromotionDecision.mint(**payload)


def _valid_parent_changes(
    reason: ExpertReleaseMatrixDecisionReason,
) -> dict[str, object]:
    gain = _assessment(
        "quality-gain",
        "quality",
        ExpertReleaseMatrixReplicateClassification.GAIN,
    )
    regression = _assessment(
        "cost-regression",
        "cost",
        ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION,
    )
    tie = _assessment(
        "quality-tie",
        "quality",
        ExpertReleaseMatrixReplicateClassification.TIE,
    )
    if reason is ExpertReleaseMatrixDecisionReason.HARD_REGRESSION:
        return {
            "outcome": ExpertReleaseMatrixDecisionOutcome.FAILED,
            "replicate_assessments": _canonical_assessments(
                replace(tie, normalized_effect=-0.01, hard_regression=True)
            ),
        }
    if reason is ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE:
        return {
            "outcome": ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            "replicate_assessments": _canonical_assessments(tie),
            "underpowered_dimension_ids": ("quality",),
        }
    if reason is ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT:
        return {
            "outcome": ExpertReleaseMatrixDecisionOutcome.APPROVED,
            "replicate_assessments": _canonical_assessments(gain),
            "confirmed_benefit_dimension_ids": ("quality",),
        }
    if reason is ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION:
        return {
            "outcome": ExpertReleaseMatrixDecisionOutcome.APPROVED,
            "validation_track": ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
            "replicate_assessments": _canonical_assessments(tie),
        }
    if reason is ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF:
        return {
            "outcome": ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            "replicate_assessments": _canonical_assessments(gain, regression),
        }
    if reason is ExpertReleaseMatrixDecisionReason.NOISY_GAIN:
        return {
            "outcome": ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            "replicate_assessments": _canonical_assessments(gain, tie),
        }
    if reason is ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION:
        return {
            "outcome": ExpertReleaseMatrixDecisionOutcome.FAILED,
            "replicate_assessments": _canonical_assessments(regression),
        }
    if reason is ExpertReleaseMatrixDecisionReason.NO_BENEFIT:
        return {
            "outcome": ExpertReleaseMatrixDecisionOutcome.FAILED,
            "replicate_assessments": _canonical_assessments(tie),
        }
    raise AssertionError(f"unsupported parent fixture reason: {reason}")


@pytest.mark.parametrize(
    ("reason", "outcome", "underpowered_dimension_ids"),
    (
        (
            ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE,
            ExpertReleaseMatrixDecisionOutcome.APPROVED,
            (),
        ),
        (
            ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
            ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            ("quality",),
        ),
    ),
)
def test_bootstrap_truth_table_accepts_only_standalone_coverage_outcomes(
    reason,
    outcome,
    underpowered_dimension_ids,
):
    decision = _mint(
        mode=ExpertReleaseMatrixMode.BOOTSTRAP,
        validation_track=ExpertValidationTrack.REPOSITORY_ARCHITECTURE,
        outcome=outcome,
        reason=reason,
        replicate_assessments=(),
        underpowered_dimension_ids=underpowered_dimension_ids,
    )

    assert decision.reason is reason
    assert decision.outcome is outcome


@pytest.mark.parametrize(
    "reason",
    (
        ExpertReleaseMatrixDecisionReason.HARD_REGRESSION,
        ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
        ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT,
        ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION,
        ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF,
        ExpertReleaseMatrixDecisionReason.NOISY_GAIN,
        ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION,
        ExpertReleaseMatrixDecisionReason.NO_BENEFIT,
    ),
)
def test_parent_truth_table_accepts_every_supported_reason(reason):
    decision = _mint(reason=reason, **_valid_parent_changes(reason))

    assert decision.reason is reason


def test_tradeoff_preserves_a_confirmed_benefit_in_another_dimension():
    changes = _valid_parent_changes(
        ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF
    )
    decision = _mint(
        reason=ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF,
        confirmed_benefit_dimension_ids=("quality",),
        **changes,
    )

    assert decision.confirmed_benefit_dimension_ids == ("quality",)


def test_decision_serialization_roundtrip_preserves_typed_identity():
    decision = _mint(
        reason=ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT,
        **_valid_parent_changes(ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT),
    )

    reopened = ExpertReleaseMatrixPromotionDecision.from_json_bytes(
        decision.to_json_bytes()
    )

    assert reopened == decision
    assert reopened.to_json_bytes() == decision.to_json_bytes()
    assert reopened.promotion_decision_id.startswith(
        "expert-release-matrix-promotion-decision:sha256:"
    )


def test_dependency_closure_requires_the_stage_capability_and_both_projections():
    decision = _mint()
    expected = tuple(
        sorted(
            {
                decision.release_matrix_stage_result_id,
                decision.release_matrix_report_id,
                decision.plan_reservation_operation_id,
                decision.validation_attempt_id,
                decision.validation_policy_id,
                *decision.release_matrix_stage_dependency_ids,
                *decision.attempt_dependency_ids,
            }
        )
    )

    assert decision.exact_dependency_ids == expected

    with pytest.raises(ValueError, match="dependency closure is not exact"):
        _mint(exact_dependency_ids=expected[1:])
    with pytest.raises(ValueError, match="dependency closure is not exact"):
        _mint(
            exact_dependency_ids=tuple(
                sorted((*expected, _id("unrelated-evidence", "extra")))
            )
        )


@pytest.mark.parametrize(
    "required_field",
    (
        "release_matrix_report_id",
        "plan_reservation_operation_id",
        "validation_attempt_id",
        "validation_policy_id",
    ),
)
def test_stage_dependency_projection_must_contain_required_authority(required_field):
    payload = _base_payload()
    required_id = payload[required_field]
    payload["release_matrix_stage_dependency_ids"] = tuple(
        dependency_id
        for dependency_id in payload["release_matrix_stage_dependency_ids"]
        if dependency_id != required_id
    )

    with pytest.raises(ValueError, match="omit required authority"):
        ExpertReleaseMatrixPromotionDecision.mint(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("release_matrix_stage_dependency_ids", ()),
        ("attempt_dependency_ids", ()),
        ("exact_dependency_ids", ()),
    ),
)
def test_dependency_projections_are_nonempty(field, value):
    with pytest.raises(ValueError, match="non-empty, sorted, and unique"):
        _mint(**{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "release_matrix_stage_dependency_ids",
        "attempt_dependency_ids",
        "exact_dependency_ids",
    ),
)
def test_dependency_projections_are_canonical(field):
    values = tuple(_base_payload()[field])

    with pytest.raises(ValueError, match="non-empty, sorted, and unique"):
        _mint(**{field: tuple(reversed(values))})
    with pytest.raises(ValueError, match="non-empty, sorted, and unique"):
        _mint(**{field: (*values, values[-1])})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "release_matrix_stage_result_id",
            _id("wrong", "stage-result"),
            "wrong namespace",
        ),
        (
            "release_matrix_report_id",
            _id("wrong", "report"),
            "wrong namespace",
        ),
        (
            "plan_reservation_operation_id",
            _id("wrong", "operation"),
            "wrong namespace",
        ),
        (
            "validation_attempt_id",
            _id("wrong", "attempt"),
            "wrong namespace",
        ),
        (
            "validation_policy_id",
            _id("wrong", "policy"),
            "wrong namespace",
        ),
        ("promotion_policy_version", "invalid version!", "qualified identifier"),
        ("configuration_fingerprint", "sha256:short", "sha256 digest"),
    ),
)
def test_decision_authority_fields_are_typed(field, value, message):
    with pytest.raises(ValueError, match=message):
        _mint(**{field: value})


def test_replicate_assessment_rejects_signed_zero_and_non_boolean_hard_flag():
    with pytest.raises(ValueError, match="signed zero"):
        replace(
            _assessment(
                "signed-zero",
                "quality",
                ExpertReleaseMatrixReplicateClassification.TIE,
            ),
            normalized_effect=-0.0,
        )
    with pytest.raises(ContractValidationError, match="boolean"):
        ExpertReleaseMatrixReplicateAssessment(
            evaluation_cell_id=_id(
                "expert-release-matrix-evaluation-cell",
                "hard-flag",
            ),
            comparison_dimension_id="quality",
            replicate_id="replicate-hard-flag",
            normalized_effect=0.0,
            classification=ExpertReleaseMatrixReplicateClassification.TIE,
            hard_regression=1,
        )


@pytest.mark.parametrize(
    ("classification", "normalized_effect", "hard_regression"),
    (
        (ExpertReleaseMatrixReplicateClassification.GAIN, 0.0, False),
        (
            ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION,
            0.0,
            False,
        ),
        (ExpertReleaseMatrixReplicateClassification.TIE, 0.1, True),
    ),
)
def test_replicate_assessment_rejects_impossible_effect_direction(
    classification,
    normalized_effect,
    hard_regression,
):
    assessment = _assessment(
        "impossible-direction",
        "quality",
        ExpertReleaseMatrixReplicateClassification.TIE,
    )

    with pytest.raises(ValueError, match="effect direction|negative effect"):
        replace(
            assessment,
            classification=classification,
            normalized_effect=normalized_effect,
            hard_regression=hard_regression,
        )


def test_replicate_assessments_are_canonical_and_unique_by_cell_and_replicate():
    gain = _assessment(
        "gain",
        "quality",
        ExpertReleaseMatrixReplicateClassification.GAIN,
    )
    regression = _assessment(
        "regression",
        "cost",
        ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION,
    )
    canonical = _canonical_assessments(gain, regression)

    with pytest.raises(ValueError, match="canonical and unique"):
        _mint(replicate_assessments=tuple(reversed(canonical)))
    with pytest.raises(ValueError, match="canonical and unique"):
        _mint(replicate_assessments=(gain, gain))


def test_dimension_projections_are_canonical_and_disjoint():
    with pytest.raises(ValueError, match="sorted and unique"):
        _mint(
            reason=ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
            outcome=ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            underpowered_dimension_ids=("quality", "cost"),
        )
    with pytest.raises(ValueError, match="dimension projections must be disjoint"):
        _mint(
            reason=ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
            outcome=ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            replicate_assessments=_canonical_assessments(
                _assessment(
                    "quality-gain",
                    "quality",
                    ExpertReleaseMatrixReplicateClassification.GAIN,
                )
            ),
            underpowered_dimension_ids=("quality",),
            confirmed_benefit_dimension_ids=("quality",),
        )


def test_confirmed_benefit_dimensions_must_be_observed_gain_dimensions():
    with pytest.raises(ValueError, match="must be gain dimensions"):
        _mint(confirmed_benefit_dimension_ids=("quality",))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {
                "validation_track": ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
            },
            "repository-architecture",
        ),
        (
            {
                "replicate_assessments": _canonical_assessments(
                    _assessment(
                        "bootstrap-tie",
                        "quality",
                        ExpertReleaseMatrixReplicateClassification.TIE,
                    )
                ),
            },
            "standalone repository-architecture",
        ),
        (
            {
                "reason": ExpertReleaseMatrixDecisionReason.NO_BENEFIT,
            },
            "unsupported reason",
        ),
    ),
)
def test_bootstrap_rejects_unsupported_structural_combinations(changes, message):
    bootstrap_changes = {
        "mode": ExpertReleaseMatrixMode.BOOTSTRAP,
        "validation_track": ExpertValidationTrack.REPOSITORY_ARCHITECTURE,
        "outcome": ExpertReleaseMatrixDecisionOutcome.APPROVED,
        "reason": ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE,
        "replicate_assessments": (),
    }
    bootstrap_changes.update(changes)
    with pytest.raises(ValueError, match=message):
        _mint(**bootstrap_changes)


def test_parent_rejects_bootstrap_reason_and_empty_assessments():
    with pytest.raises(ValueError, match="unsupported reason"):
        _mint(reason=ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE)
    with pytest.raises(ValueError, match="require replicate assessments"):
        _mint(replicate_assessments=())


@pytest.mark.parametrize(
    "reason",
    (
        ExpertReleaseMatrixDecisionReason.HARD_REGRESSION,
        ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
        ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT,
        ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION,
        ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF,
        ExpertReleaseMatrixDecisionReason.NOISY_GAIN,
        ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION,
        ExpertReleaseMatrixDecisionReason.NO_BENEFIT,
    ),
)
def test_parent_truth_table_rejects_wrong_outcome_for_every_reason(reason):
    changes = _valid_parent_changes(reason)
    valid_outcome = changes["outcome"]
    changes["outcome"] = next(
        outcome
        for outcome in ExpertReleaseMatrixDecisionOutcome
        if outcome is not valid_outcome
    )

    with pytest.raises(ValueError, match="requires|contradicts"):
        _mint(reason=reason, **changes)


@pytest.mark.parametrize(
    ("reason", "underpowered_dimension_ids"),
    (
        (
            ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE,
            (),
        ),
        (
            ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
            ("quality",),
        ),
    ),
)
def test_bootstrap_truth_table_rejects_wrong_outcome(
    reason,
    underpowered_dimension_ids,
):
    with pytest.raises(ValueError, match="requires"):
        _mint(
            mode=ExpertReleaseMatrixMode.BOOTSTRAP,
            validation_track=ExpertValidationTrack.REPOSITORY_ARCHITECTURE,
            outcome=ExpertReleaseMatrixDecisionOutcome.FAILED,
            reason=reason,
            replicate_assessments=(),
            underpowered_dimension_ids=underpowered_dimension_ids,
        )


@pytest.mark.parametrize(
    "reason",
    (
        ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
        ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT,
        ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION,
        ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF,
        ExpertReleaseMatrixDecisionReason.NOISY_GAIN,
        ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION,
        ExpertReleaseMatrixDecisionReason.NO_BENEFIT,
    ),
)
def test_only_hard_regression_reason_can_retain_a_hard_assessment(reason):
    changes = _valid_parent_changes(reason)
    assessments = tuple(changes["replicate_assessments"])
    changes["replicate_assessments"] = _canonical_assessments(
        *assessments,
        _assessment(
            "forced-hard-regression",
            "hard-gate",
            ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION,
            hard_regression=True,
        ),
    )

    with pytest.raises(ValueError, match="cannot retain a hard regression"):
        _mint(reason=reason, **changes)


def test_hard_regression_cannot_advertise_confirmed_benefits():
    changes = _valid_parent_changes(ExpertReleaseMatrixDecisionReason.HARD_REGRESSION)
    changes["replicate_assessments"] = _canonical_assessments(
        *changes["replicate_assessments"],
        _assessment(
            "forged-confirmed-gain",
            "benefit",
            ExpertReleaseMatrixReplicateClassification.GAIN,
        ),
    )
    changes["confirmed_benefit_dimension_ids"] = ("benefit",)

    with pytest.raises(ValueError, match="hard-gated"):
        _mint(reason=ExpertReleaseMatrixDecisionReason.HARD_REGRESSION, **changes)


@pytest.mark.parametrize(
    "reason",
    (
        ExpertReleaseMatrixDecisionReason.HARD_REGRESSION,
        ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT,
        ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION,
        ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF,
        ExpertReleaseMatrixDecisionReason.NOISY_GAIN,
        ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION,
        ExpertReleaseMatrixDecisionReason.NO_BENEFIT,
    ),
)
def test_non_underpowered_parent_reasons_reject_underpowered_dimensions(reason):
    changes = _valid_parent_changes(reason)
    changes["underpowered_dimension_ids"] = ("unpowered",)

    with pytest.raises(ValueError, match="underpowered|hard-gated"):
        _mint(reason=reason, **changes)


@pytest.mark.parametrize(
    ("reason", "changes"),
    (
        (
            ExpertReleaseMatrixDecisionReason.HARD_REGRESSION,
            {
                "replicate_assessments": _canonical_assessments(
                    _assessment(
                        "not-hard",
                        "quality",
                        ExpertReleaseMatrixReplicateClassification.TIE,
                    )
                )
            },
        ),
        (
            ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
            {"underpowered_dimension_ids": ()},
        ),
        (
            ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT,
            {"confirmed_benefit_dimension_ids": ()},
        ),
        (
            ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION,
            {"validation_track": ExpertValidationTrack.BEHAVIORAL_CAPABILITY},
        ),
        (
            ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF,
            {
                "replicate_assessments": _canonical_assessments(
                    _assessment(
                        "gain-only",
                        "quality",
                        ExpertReleaseMatrixReplicateClassification.GAIN,
                    )
                )
            },
        ),
        (
            ExpertReleaseMatrixDecisionReason.NOISY_GAIN,
            {
                "replicate_assessments": _canonical_assessments(
                    _assessment(
                        "tie-only",
                        "quality",
                        ExpertReleaseMatrixReplicateClassification.TIE,
                    )
                )
            },
        ),
        (
            ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION,
            {
                "replicate_assessments": _canonical_assessments(
                    _assessment(
                        "regression-gain",
                        "quality",
                        ExpertReleaseMatrixReplicateClassification.GAIN,
                    )
                )
            },
        ),
        (
            ExpertReleaseMatrixDecisionReason.NO_BENEFIT,
            {
                "replicate_assessments": _canonical_assessments(
                    _assessment(
                        "benefit",
                        "quality",
                        ExpertReleaseMatrixReplicateClassification.GAIN,
                    )
                )
            },
        ),
    ),
)
def test_each_parent_reason_rejects_missing_required_evidence(reason, changes):
    valid_changes = _valid_parent_changes(reason)
    valid_changes.update(changes)

    with pytest.raises(ValueError, match="requires|contradicts"):
        _mint(reason=reason, **valid_changes)
