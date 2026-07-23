from __future__ import annotations

from dataclasses import replace

import pytest

from kapso.cross_run.contracts import (
    EvaluationFingerprint,
    ExpertEvaluatorOutcome,
    ExpertSourceReplayExecutionLegKind,
    ObjectiveDirection,
    TaskAdapterManifest,
)
from kapso.cross_run.expert.replay_comparison import (
    build_expert_source_replay_paired_comparison_receipt,
)
from kapso.cross_run.expert.replay_comparison_contracts import (
    ExpertSourceReplayFingerprintComparison,
    ExpertSourceReplayPairedComparisonReceipt,
    normalized_zero,
)
from kapso.cross_run.expert.replay_decision import (
    decide_expert_source_replay_stage,
)
from kapso.cross_run.expert.replay_decision_contracts import (
    ExpertSourceReplayDecisionError,
    ExpertSourceReplayStageDecision,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TaskEvaluatorFingerprintResult,
)
from test_expert_replay_execution_store import _remint
from test_expert_source_replay import _validation_policy
from test_expert_source_replay_comparison import _complete_execution
from test_cross_run_contracts import (
    build_records,
    digest,
    verified_test_task_adapter,
)


def _receipt(
    tmp_path,
    validation_settings=None,
    aggregate_by_leg_kind=None,
    contract_records=None,
    source_adapter=None,
    evaluation_evidence=None,
):
    _, prepared, reservation, store, completed = _complete_execution(
        tmp_path,
        validation_settings,
        aggregate_by_leg_kind,
        contract_records=contract_records,
        source_adapter=source_adapter,
        evaluation_evidence=evaluation_evidence,
    )
    receipt = build_expert_source_replay_paired_comparison_receipt(
        completed_execution=completed,
        execution_store=store,
        reservation=reservation,
        prepared_request=prepared,
    )
    return prepared, receipt


def _dual_fingerprint_authority():
    base_records = build_records()
    primary_fingerprint = next(
        record for record in base_records if isinstance(record, EvaluationFingerprint)
    )
    base_manifest = next(
        record for record in base_records if isinstance(record, TaskAdapterManifest)
    )
    auxiliary_fingerprint = _remint(
        primary_fingerprint,
        evaluator_fingerprint=digest("auxiliary-evaluator"),
    )
    primary_binding = base_manifest.task_evaluator.metric_comparison_bindings[0]
    auxiliary_binding = replace(
        primary_binding,
        evaluator_fingerprint=auxiliary_fingerprint.evaluator_fingerprint,
    )
    task_evaluator = replace(
        base_manifest.task_evaluator,
        supported_evaluator_fingerprints=tuple(
            sorted(
                (
                    primary_fingerprint.evaluator_fingerprint,
                    auxiliary_fingerprint.evaluator_fingerprint,
                )
            )
        ),
        metric_comparison_bindings=tuple(
            sorted(
                (primary_binding, auxiliary_binding),
                key=lambda binding: (
                    binding.evaluator_fingerprint,
                    binding.metric_name,
                ),
            )
        ),
    )
    records = build_records(task_evaluator=task_evaluator)
    source_manifest = next(
        record for record in records if isinstance(record, TaskAdapterManifest)
    )
    return (
        records,
        verified_test_task_adapter(source_manifest),
        (
            (primary_fingerprint, 0.5),
            (auxiliary_fingerprint, 0.4),
        ),
        auxiliary_fingerprint.evaluation_fingerprint_id,
    )


def _comparison_with_effect(
    template,
    normalized_effect,
    metric_comparison_binding=None,
):
    selected_binding = metric_comparison_binding or (template.metric_comparison_binding)
    scale = selected_binding.comparison_scale
    aligned_delta = normalized_effect * scale
    raw_delta = (
        aligned_delta
        if template.evaluation_fingerprint.objective_direction
        is ObjectiveDirection.MAXIMIZE
        else -aligned_delta
    )
    replicate_ids = template.evaluation_fingerprint.seed_or_replicate_ids
    control_result = TaskEvaluatorFingerprintResult(
        evaluation_fingerprint_id=(
            template.evaluation_fingerprint.evaluation_fingerprint_id
        ),
        aggregate_value=0.0,
        replicate_values={replicate_id: 0.0 for replicate_id in replicate_ids},
    )
    candidate_result = TaskEvaluatorFingerprintResult(
        evaluation_fingerprint_id=(
            template.evaluation_fingerprint.evaluation_fingerprint_id
        ),
        aggregate_value=raw_delta,
        replicate_values={replicate_id: raw_delta for replicate_id in replicate_ids},
    )
    return ExpertSourceReplayFingerprintComparison(
        evaluation_fingerprint=template.evaluation_fingerprint,
        metric_comparison_binding=selected_binding,
        control_result=control_result,
        candidate_result=candidate_result,
        aggregate_raw_delta=normalized_zero(raw_delta),
        aggregate_direction_aligned_delta=normalized_zero(aligned_delta),
        aggregate_normalized_effect=normalized_zero(normalized_effect),
    )


def _receipt_with_comparisons(receipt, comparisons):
    case = receipt.case_comparisons[0]
    changed_case = replace(
        case,
        fingerprint_comparisons=tuple(
            sorted(
                comparisons,
                key=lambda comparison: (
                    comparison.evaluation_fingerprint.evaluation_fingerprint_id
                ),
            )
        ),
    )
    return _remint(receipt, case_comparisons=(changed_case,))


def test_stage_decision_is_deterministic_non_regression_evidence(tmp_path):
    prepared, receipt = _receipt(tmp_path)
    policy = prepared.settings.policy.validation_policy()

    decision = decide_expert_source_replay_stage(
        paired_comparison_receipt=receipt,
        prepared_request=prepared,
    )

    assert decision.outcome is ExpertEvaluatorOutcome.PASSED
    assert decision.hard_regression_comparisons == ()
    assert decision.paired_comparison_dependency_ids == receipt.exact_dependency_ids
    assert set(decision.exact_dependency_ids) == {
        receipt.paired_comparison_receipt_id,
        policy.validation_policy_id,
        *receipt.exact_dependency_ids,
    }
    assert (
        ExpertSourceReplayStageDecision.from_json_bytes(decision.to_json_bytes())
        == decision
    )
    assert (
        decide_expert_source_replay_stage(
            paired_comparison_receipt=receipt,
            prepared_request=prepared,
        ).to_json_bytes()
        == decision.to_json_bytes()
    )


def test_every_governed_fingerprint_is_a_hard_regression_veto(tmp_path):
    records, source_adapter, evidence, auxiliary_fingerprint_id = (
        _dual_fingerprint_authority()
    )
    prepared, receipt = _receipt(
        tmp_path,
        contract_records=records,
        source_adapter=source_adapter,
        evaluation_evidence=evidence,
    )
    case_comparison = receipt.case_comparisons[0]
    comparisons_by_id = {
        comparison.evaluation_fingerprint.evaluation_fingerprint_id: comparison
        for comparison in case_comparison.fingerprint_comparisons
    }
    changed_receipt = _receipt_with_comparisons(
        receipt,
        tuple(
            _comparison_with_effect(
                comparisons_by_id[fingerprint_id],
                (
                    1.0
                    if fingerprint_id == case_comparison.score_of_record_fingerprint_id
                    else -0.01
                ),
            )
            for fingerprint_id in sorted(comparisons_by_id)
        ),
    )

    decision = decide_expert_source_replay_stage(
        paired_comparison_receipt=changed_receipt,
        prepared_request=prepared,
    )

    assert decision.outcome is ExpertEvaluatorOutcome.CANDIDATE_FAILED
    assert len(decision.hard_regression_comparisons) == 1
    assert (
        decision.hard_regression_comparisons[0].evaluation_fingerprint_id
        == auxiliary_fingerprint_id
    )


@pytest.mark.parametrize(
    "binding_changes",
    (
        {"comparison_scale": 2.0},
        {"comparison_dimension_id": "cost"},
    ),
)
def test_decision_rejects_metric_authority_not_in_prepared_adapter(
    tmp_path,
    binding_changes,
):
    prepared, receipt = _receipt(tmp_path)
    comparison = receipt.case_comparisons[0].fingerprint_comparisons[0]
    forged_binding = replace(
        comparison.metric_comparison_binding,
        **binding_changes,
    )
    forged_receipt = _receipt_with_comparisons(
        receipt,
        (
            _comparison_with_effect(
                comparison,
                -0.01,
                forged_binding,
            ),
        ),
    )

    with pytest.raises(ExpertSourceReplayDecisionError, match="metric authority"):
        decide_expert_source_replay_stage(
            paired_comparison_receipt=forged_receipt,
            prepared_request=prepared,
        )


def test_hard_regression_bound_is_strict_and_never_uses_noise_floor(tmp_path):
    settings = _validation_policy()
    changed_dimensions = tuple(
        replace(dimension, hard_regression_ratio=0.125, noise_floor_ratio=0.5)
        for dimension in settings.policy.promotion.pareto_dimensions
    )
    settings = replace(
        settings,
        policy=replace(
            settings.policy,
            promotion=replace(
                settings.policy.promotion,
                minimum_replicates_per_cell=10,
                pareto_dimensions=changed_dimensions,
            ),
        ),
    )
    prepared, receipt = _receipt(
        tmp_path,
        settings,
        {
            ExpertSourceReplayExecutionLegKind.CONTROL_PARENT: 0.0,
            ExpertSourceReplayExecutionLegKind.CANDIDATE: -0.125,
        },
    )
    policy = prepared.settings.policy.validation_policy()

    boundary_decision = decide_expert_source_replay_stage(
        paired_comparison_receipt=receipt,
        prepared_request=prepared,
    )
    assert boundary_decision.outcome is ExpertEvaluatorOutcome.PASSED

    comparison = receipt.case_comparisons[0].fingerprint_comparisons[0]
    regressing_receipt = _receipt_with_comparisons(
        receipt,
        (_comparison_with_effect(comparison, -0.25),),
    )
    failed_decision = decide_expert_source_replay_stage(
        paired_comparison_receipt=regressing_receipt,
        prepared_request=prepared,
    )
    assert failed_decision.outcome is ExpertEvaluatorOutcome.CANDIDATE_FAILED


def test_decision_rejects_unpinned_policy_and_malformed_outcome(tmp_path):
    prepared, receipt = _receipt(tmp_path)
    policy = prepared.settings.policy.validation_policy()
    changed_policy = replace(
        prepared.settings.policy,
        source_replay_stage_decision_policy_version=(
            "kapso.expert_source_replay_stage_decision.changed"
        ),
    ).validation_policy()

    injected_request_dependencies = tuple(
        sorted({*receipt.request_dependency_ids, changed_policy.validation_policy_id})
    )
    injected_exact_dependencies = tuple(
        sorted(
            {
                receipt.reservation_id,
                receipt.execution_request_id,
                *receipt.reservation_dependency_ids,
                *injected_request_dependencies,
                *receipt.execution_journal_event_ids,
            }
        )
    )
    injected_receipt = _remint(
        receipt,
        request_dependency_ids=injected_request_dependencies,
        exact_dependency_ids=injected_exact_dependencies,
    )

    with pytest.raises(
        ExpertSourceReplayDecisionError, match="exact execution request"
    ):
        decide_expert_source_replay_stage(
            paired_comparison_receipt=injected_receipt,
            prepared_request=prepared,
        )

    decision = decide_expert_source_replay_stage(
        paired_comparison_receipt=receipt,
        prepared_request=prepared,
    )
    with pytest.raises(ExpertSourceReplayDecisionError, match="outcome"):
        _remint(decision, outcome=ExpertEvaluatorOutcome.CANDIDATE_FAILED)
