"""Pure policy reduction of one factual expert source replay receipt."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
)
from kapso.cross_run.expert.replay_comparison_contracts import (
    ExpertSourceReplayPairedComparisonReceipt,
)
from kapso.cross_run.expert.replay_decision_contracts import (
    ExpertSourceReplayComparisonReference,
    ExpertSourceReplayDecisionError,
    ExpertSourceReplayStageDecision,
)

if TYPE_CHECKING:
    from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest


def decide_expert_source_replay_stage(
    *,
    paired_comparison_receipt: ExpertSourceReplayPairedComparisonReceipt,
    prepared_request: PreparedExpertSourceReplayRequest,
) -> ExpertSourceReplayStageDecision:
    execution_request = prepared_request.request
    validation_policy = prepared_request.settings.policy.validation_policy()
    if (
        paired_comparison_receipt.execution_request_id
        != execution_request.execution_request_id
        or paired_comparison_receipt.request_dependency_ids
        != execution_request.exact_dependency_ids
    ):
        raise ExpertSourceReplayDecisionError(
            "source replay receipt differs from the exact execution request"
        )
    if execution_request.validation_policy_id != validation_policy.validation_policy_id:
        raise ExpertSourceReplayDecisionError(
            "source replay request does not pin the supplied validation policy"
        )
    if paired_comparison_receipt.aggregate_recomputation_tolerance != (
        validation_policy.policy.source_replay_score_comparison_tolerance
    ):
        raise ExpertSourceReplayDecisionError(
            "source replay receipt tolerance differs from validation policy"
        )
    request_cases = {
        request_case.execution_case_id: request_case
        for request_case in execution_request.cases
    }
    receipt_cases = {
        case_comparison.execution_case_id: case_comparison
        for case_comparison in paired_comparison_receipt.case_comparisons
    }
    if set(receipt_cases) != set(request_cases):
        raise ExpertSourceReplayDecisionError(
            "source replay receipt cases differ from the execution request"
        )
    prepared_cases = {
        prepared_case.request_case.execution_case_id: prepared_case
        for prepared_case in prepared_request.cases
    }
    if set(prepared_cases) != set(request_cases):
        raise ExpertSourceReplayDecisionError(
            "source replay prepared cases differ from the execution request"
        )
    for execution_case_id, case_comparison in receipt_cases.items():
        request_case = request_cases[execution_case_id]
        prepared_case = prepared_cases[execution_case_id]
        fingerprint_ids = tuple(
            comparison.evaluation_fingerprint.evaluation_fingerprint_id
            for comparison in case_comparison.fingerprint_comparisons
        )
        if (
            case_comparison.score_of_record_fingerprint_id
            != request_case.source_score_of_record_fingerprint_id
            or fingerprint_ids != request_case.source_evaluation_fingerprint_ids
        ):
            raise ExpertSourceReplayDecisionError(
                "source replay receipt fingerprints differ from the execution request"
            )
        terminal_attempt = prepared_case.episode.attempts[
            prepared_case.episode.terminal_attempt_revision
        ]
        fingerprints = {
            fingerprint.evaluation_fingerprint_id: fingerprint
            for fingerprint in terminal_attempt.evaluation_fingerprints
        }
        bindings = {
            (binding.evaluator_fingerprint, binding.metric_name): binding
            for binding in (
                prepared_case.task_adapter.manifest.task_evaluator.metric_comparison_bindings
            )
        }
        for comparison in case_comparison.fingerprint_comparisons:
            fingerprint = comparison.evaluation_fingerprint
            binding = comparison.metric_comparison_binding
            expected_fingerprint = fingerprints.get(
                fingerprint.evaluation_fingerprint_id
            )
            expected_binding = bindings.get(
                (fingerprint.evaluator_fingerprint, fingerprint.metric_name)
            )
            if fingerprint != expected_fingerprint or binding != expected_binding:
                raise ExpertSourceReplayDecisionError(
                    "source replay receipt metric authority differs from prepared input"
                )
    dimensions = {
        dimension.dimension_id: dimension
        for dimension in validation_policy.policy.promotion.pareto_dimensions
    }
    hard_regressions = []
    for case_comparison in paired_comparison_receipt.case_comparisons:
        for comparison in case_comparison.fingerprint_comparisons:
            binding = comparison.metric_comparison_binding
            dimension = dimensions.get(binding.comparison_dimension_id)
            if dimension is None:
                raise ExpertSourceReplayDecisionError(
                    "source replay comparison dimension is absent from validation policy"
                )
            if dimension.direction is not binding.objective_direction:
                raise ExpertSourceReplayDecisionError(
                    "source replay comparison direction differs from validation policy"
                )
            if comparison.aggregate_normalized_effect < (
                -dimension.hard_regression_ratio
            ):
                hard_regressions.append(
                    ExpertSourceReplayComparisonReference(
                        execution_case_id=case_comparison.execution_case_id,
                        evaluation_fingerprint_id=(
                            comparison.evaluation_fingerprint.evaluation_fingerprint_id
                        ),
                    )
                )
    ordered_hard_regressions = tuple(
        sorted(
            hard_regressions,
            key=lambda reference: (
                reference.execution_case_id,
                reference.evaluation_fingerprint_id,
            ),
        )
    )
    outcome = (
        ExpertEvaluatorOutcome.CANDIDATE_FAILED
        if ordered_hard_regressions
        else ExpertEvaluatorOutcome.PASSED
    )
    exact_dependency_ids = tuple(
        sorted(
            {
                paired_comparison_receipt.paired_comparison_receipt_id,
                validation_policy.validation_policy_id,
                *paired_comparison_receipt.exact_dependency_ids,
            }
        )
    )
    return ExpertSourceReplayStageDecision.mint(
        paired_comparison_receipt_id=(
            paired_comparison_receipt.paired_comparison_receipt_id
        ),
        validation_policy_id=validation_policy.validation_policy_id,
        decision_policy_version=(
            validation_policy.policy.source_replay_stage_decision_policy_version
        ),
        outcome=outcome,
        hard_regression_comparisons=ordered_hard_regressions,
        paired_comparison_dependency_ids=(
            paired_comparison_receipt.exact_dependency_ids
        ),
        exact_dependency_ids=exact_dependency_ids,
    )
