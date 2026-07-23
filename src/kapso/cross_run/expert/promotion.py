"""Deterministic Pareto promotion over an accepted expert release matrix."""

from __future__ import annotations

import math

from kapso.cross_run.contracts import (
    ExpertValidationAttempt,
    ExpertValidationStage,
    ExpertValidationTrack,
    ObjectiveDirection,
)
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixEvaluationCell,
    ExpertReleaseMatrixMode,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
    ExpertReleaseMatrixDecisionReason,
    ExpertReleaseMatrixPromotionDecision,
    ExpertReleaseMatrixReplicateAssessment,
    ExpertReleaseMatrixReplicateClassification,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.settings import (
    ExpertParetoDimensionSettings,
    ExpertValidationSettings,
)


class ExpertReleaseMatrixPromotionError(ValueError):
    """Accepted release-matrix evidence cannot yield a Pareto decision."""


def decide_expert_release_matrix_promotion(
    *,
    stage_result: ExpertReleaseMatrixStageResultRecord,
    attempt: ExpertValidationAttempt,
    settings: ExpertValidationSettings,
) -> ExpertReleaseMatrixPromotionDecision:
    """Reduce one accepted factual matrix without aggregation or weighted scores."""

    if (
        type(stage_result) is not ExpertReleaseMatrixStageResultRecord
        or type(attempt) is not ExpertValidationAttempt
        or type(settings) is not ExpertValidationSettings
    ):
        raise ExpertReleaseMatrixPromotionError(
            "promotion requires exact typed stage, attempt, and settings authority"
        )
    report = stage_result.release_matrix_report
    policy = settings.policy.validation_policy()
    if (
        stage_result.validation_attempt_id != attempt.validation_attempt_id
        or stage_result.candidate_id != attempt.candidate_id
        or stage_result.candidate_tree_hash != attempt.candidate_tree_hash
        or stage_result.scope_contract_id != attempt.scope_contract_id
        or stage_result.parent_release_id != attempt.parent_release_id
        or stage_result.validation_policy_id != attempt.validation_policy_id
        or stage_result.configuration_fingerprint != attempt.configuration_fingerprint
        or report.candidate_commit_record_id != attempt.candidate_commit_record_id
        or attempt.validation_policy_id != policy.validation_policy_id
        or attempt.configuration_fingerprint != settings.configuration_fingerprint
        or ExpertValidationStage.RELEASE_MATRIX not in attempt.required_stages
        or ExpertValidationStage.PUBLICATION_ELIGIBILITY not in attempt.required_stages
        or attempt.required_stages[-2:]
        != (
            ExpertValidationStage.RELEASE_MATRIX,
            ExpertValidationStage.PUBLICATION_ELIGIBILITY,
        )
    ):
        raise ExpertReleaseMatrixPromotionError(
            "accepted release matrix differs from its configured validation attempt"
        )
    dimensions = {
        dimension.dimension_id: dimension
        for dimension in settings.policy.promotion.pareto_dimensions
    }
    cells = report.evaluation_plan.evaluation_cells
    observed_dimensions = {
        cell.metric_comparison_binding.comparison_dimension_id for cell in cells
    }
    if observed_dimensions != set(dimensions) or any(
        cell.metric_comparison_binding.objective_direction
        is not dimensions[
            cell.metric_comparison_binding.comparison_dimension_id
        ].direction
        for cell in cells
    ):
        raise ExpertReleaseMatrixPromotionError(
            "release matrix differs from configured Pareto dimensions"
        )
    if report.mode is ExpertReleaseMatrixMode.BOOTSTRAP:
        return _decide_bootstrap_promotion(
            stage_result=stage_result,
            attempt=attempt,
            settings=settings,
            dimensions=dimensions,
        )
    assessments = _derive_replicate_assessments(
        stage_result,
        dimensions,
    )
    if any(assessment.hard_regression for assessment in assessments):
        return _mint_promotion_decision(
            stage_result=stage_result,
            attempt=attempt,
            settings=settings,
            outcome=ExpertReleaseMatrixDecisionOutcome.FAILED,
            reason=ExpertReleaseMatrixDecisionReason.HARD_REGRESSION,
            assessments=assessments,
            underpowered_dimension_ids=(),
            confirmed_benefit_dimension_ids=(),
        )
    underpowered = _underpowered_dimensions(
        cells,
        dimensions,
        minimum_replicates_per_cell=(
            settings.policy.promotion.minimum_replicates_per_cell
        ),
        minimum_distinct_pairs=(
            settings.policy.promotion.minimum_distinct_context_lineage_pairs
        ),
    )
    assessments_by_cell = _assessments_by_cell(assessments)
    confirmed = _confirmed_benefit_dimensions(
        cells,
        assessments_by_cell,
        dimensions,
        underpowered,
        minimum_distinct_pairs=(
            settings.policy.promotion.minimum_distinct_context_lineage_pairs
        ),
    )
    if underpowered:
        return _mint_promotion_decision(
            stage_result=stage_result,
            attempt=attempt,
            settings=settings,
            outcome=ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            reason=ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE,
            assessments=assessments,
            underpowered_dimension_ids=underpowered,
            confirmed_benefit_dimension_ids=confirmed,
        )
    has_gain = any(
        assessment.classification is ExpertReleaseMatrixReplicateClassification.GAIN
        for assessment in assessments
    )
    has_material_regression = any(
        assessment.classification
        is ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION
        for assessment in assessments
    )
    if confirmed and not has_material_regression:
        outcome = ExpertReleaseMatrixDecisionOutcome.APPROVED
        reason = ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT
    elif (
        attempt.validation_track is ExpertValidationTrack.MECHANICAL_GENERAL_FIX
        and not has_material_regression
    ):
        outcome = ExpertReleaseMatrixDecisionOutcome.APPROVED
        reason = ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION
    elif has_gain and has_material_regression:
        outcome = ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED
        reason = ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF
    elif has_gain:
        outcome = ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED
        reason = ExpertReleaseMatrixDecisionReason.NOISY_GAIN
    elif has_material_regression:
        outcome = ExpertReleaseMatrixDecisionOutcome.FAILED
        reason = ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION
    else:
        outcome = ExpertReleaseMatrixDecisionOutcome.FAILED
        reason = ExpertReleaseMatrixDecisionReason.NO_BENEFIT
    return _mint_promotion_decision(
        stage_result=stage_result,
        attempt=attempt,
        settings=settings,
        outcome=outcome,
        reason=reason,
        assessments=assessments,
        underpowered_dimension_ids=(),
        confirmed_benefit_dimension_ids=confirmed,
    )


def _decide_bootstrap_promotion(
    *,
    stage_result: ExpertReleaseMatrixStageResultRecord,
    attempt: ExpertValidationAttempt,
    settings: ExpertValidationSettings,
    dimensions: dict[str, ExpertParetoDimensionSettings],
) -> ExpertReleaseMatrixPromotionDecision:
    if (
        attempt.validation_track is not ExpertValidationTrack.REPOSITORY_ARCHITECTURE
        or attempt.parent_release_id is not None
    ):
        raise ExpertReleaseMatrixPromotionError(
            "bootstrap promotion requires repository-architecture authority"
        )
    underpowered = _underpowered_dimensions(
        stage_result.release_matrix_report.evaluation_plan.evaluation_cells,
        dimensions,
        minimum_replicates_per_cell=(
            settings.policy.promotion.minimum_replicates_per_cell
        ),
        minimum_distinct_pairs=(
            settings.policy.promotion.minimum_distinct_context_lineage_pairs
        ),
    )
    if underpowered:
        outcome = ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED
        reason = ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE
    else:
        outcome = ExpertReleaseMatrixDecisionOutcome.APPROVED
        reason = ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE
    return _mint_promotion_decision(
        stage_result=stage_result,
        attempt=attempt,
        settings=settings,
        outcome=outcome,
        reason=reason,
        assessments=(),
        underpowered_dimension_ids=underpowered,
        confirmed_benefit_dimension_ids=(),
    )


def _derive_replicate_assessments(
    stage_result: ExpertReleaseMatrixStageResultRecord,
    dimensions: dict[str, ExpertParetoDimensionSettings],
) -> tuple[ExpertReleaseMatrixReplicateAssessment, ...]:
    report = stage_result.release_matrix_report
    assessments = []
    for cell, row in zip(
        report.evaluation_plan.evaluation_cells,
        report.evidence_rows,
        strict=True,
    ):
        parent_values = row.parent_replicate_values
        if parent_values is None:
            raise ExpertReleaseMatrixPromotionError(
                "parent comparison lacks paired control replicates"
            )
        binding = cell.metric_comparison_binding
        dimension = dimensions[binding.comparison_dimension_id]
        for replicate_id in cell.evaluation_fingerprint.seed_or_replicate_ids:
            candidate_value = row.candidate_replicate_values[replicate_id]
            parent_value = parent_values[replicate_id]
            raw_effect = candidate_value - parent_value
            if not math.isfinite(raw_effect):
                raise ExpertReleaseMatrixPromotionError(
                    "release matrix replicate subtraction is nonfinite"
                )
            aligned_effect = (
                raw_effect
                if dimension.direction is ObjectiveDirection.MAXIMIZE
                else -raw_effect
            )
            if not math.isfinite(aligned_effect):
                raise ExpertReleaseMatrixPromotionError(
                    "release matrix direction-aligned effect is nonfinite"
                )
            normalized_effect = aligned_effect / binding.comparison_scale
            if not math.isfinite(normalized_effect):
                raise ExpertReleaseMatrixPromotionError(
                    "release matrix normalized effect is nonfinite"
                )
            if normalized_effect == 0.0:
                normalized_effect = 0.0
            hard_regression = normalized_effect < -dimension.hard_regression_ratio
            if normalized_effect > dimension.noise_floor_ratio:
                classification = ExpertReleaseMatrixReplicateClassification.GAIN
            elif normalized_effect < -dimension.noise_floor_ratio:
                classification = (
                    ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION
                )
            else:
                classification = ExpertReleaseMatrixReplicateClassification.TIE
            assessments.append(
                ExpertReleaseMatrixReplicateAssessment(
                    evaluation_cell_id=cell.evaluation_cell_id,
                    comparison_dimension_id=binding.comparison_dimension_id,
                    replicate_id=replicate_id,
                    normalized_effect=normalized_effect,
                    classification=classification,
                    hard_regression=hard_regression,
                )
            )
    return tuple(sorted(assessments, key=lambda item: item.canonical_key))


def _underpowered_dimensions(
    cells: tuple[ExpertReleaseMatrixEvaluationCell, ...],
    dimensions: dict[str, ExpertParetoDimensionSettings],
    *,
    minimum_replicates_per_cell: int,
    minimum_distinct_pairs: int,
) -> tuple[str, ...]:
    underpowered = []
    for dimension_id in sorted(dimensions):
        dimension_cells = tuple(
            cell
            for cell in cells
            if cell.metric_comparison_binding.comparison_dimension_id == dimension_id
        )
        too_few_replicates = any(
            len(cell.evaluation_fingerprint.seed_or_replicate_ids)
            < minimum_replicates_per_cell
            for cell in dimension_cells
        )
        matching_size = _maximum_context_lineage_matching_size(dimension_cells)
        if too_few_replicates or matching_size < minimum_distinct_pairs:
            underpowered.append(dimension_id)
    return tuple(underpowered)


def _assessments_by_cell(
    assessments: tuple[ExpertReleaseMatrixReplicateAssessment, ...],
) -> dict[str, tuple[ExpertReleaseMatrixReplicateAssessment, ...]]:
    grouped: dict[str, list[ExpertReleaseMatrixReplicateAssessment]] = {}
    for assessment in assessments:
        grouped.setdefault(assessment.evaluation_cell_id, []).append(assessment)
    return {
        cell_id: tuple(cell_assessments)
        for cell_id, cell_assessments in grouped.items()
    }


def _confirmed_benefit_dimensions(
    cells: tuple[ExpertReleaseMatrixEvaluationCell, ...],
    assessments_by_cell: dict[
        str,
        tuple[ExpertReleaseMatrixReplicateAssessment, ...],
    ],
    dimensions: dict[str, ExpertParetoDimensionSettings],
    underpowered_dimension_ids: tuple[str, ...],
    *,
    minimum_distinct_pairs: int,
) -> tuple[str, ...]:
    underpowered = set(underpowered_dimension_ids)
    confirmed = []
    for dimension_id in sorted(dimensions):
        if dimension_id in underpowered:
            continue
        gain_cells = tuple(
            cell
            for cell in cells
            if cell.metric_comparison_binding.comparison_dimension_id == dimension_id
            and all(
                assessment.classification
                is ExpertReleaseMatrixReplicateClassification.GAIN
                for assessment in assessments_by_cell[cell.evaluation_cell_id]
            )
        )
        if _maximum_context_lineage_matching_size(gain_cells) >= (
            minimum_distinct_pairs
        ):
            confirmed.append(dimension_id)
    return tuple(confirmed)


def _maximum_context_lineage_matching_size(
    cells: tuple[ExpertReleaseMatrixEvaluationCell, ...],
) -> int:
    adjacency: dict[str, set[str]] = {}
    for cell in cells:
        context_id = cell.task_context_binding.task_context_binding_id
        adjacency.setdefault(context_id, set()).add(cell.independence_identity_id)
    ordered_adjacency = {
        context_id: tuple(sorted(lineage_ids))
        for context_id, lineage_ids in adjacency.items()
    }
    matched_lineage_by_context: dict[str, str] = {}
    matched_context_by_lineage: dict[str, str] = {}
    for starting_context_id in sorted(ordered_adjacency):
        if starting_context_id in matched_lineage_by_context:
            continue
        queued_context_ids = [starting_context_id]
        queue_position = 0
        visited_context_ids = {starting_context_id}
        visited_lineage_ids = set()
        preceding_context_by_lineage: dict[str, str] = {}
        free_lineage_id = None
        while queue_position < len(queued_context_ids) and free_lineage_id is None:
            context_id = queued_context_ids[queue_position]
            queue_position += 1
            for lineage_id in ordered_adjacency[context_id]:
                if lineage_id in visited_lineage_ids:
                    continue
                visited_lineage_ids.add(lineage_id)
                preceding_context_by_lineage[lineage_id] = context_id
                matched_context_id = matched_context_by_lineage.get(lineage_id)
                if matched_context_id is None:
                    free_lineage_id = lineage_id
                    break
                if matched_context_id not in visited_context_ids:
                    visited_context_ids.add(matched_context_id)
                    queued_context_ids.append(matched_context_id)
        if free_lineage_id is None:
            continue
        lineage_id = free_lineage_id
        while True:
            context_id = preceding_context_by_lineage[lineage_id]
            previous_lineage_id = matched_lineage_by_context.get(context_id)
            matched_lineage_by_context[context_id] = lineage_id
            matched_context_by_lineage[lineage_id] = context_id
            if previous_lineage_id is None:
                break
            lineage_id = previous_lineage_id
    return len(matched_lineage_by_context)


def _mint_promotion_decision(
    *,
    stage_result: ExpertReleaseMatrixStageResultRecord,
    attempt: ExpertValidationAttempt,
    settings: ExpertValidationSettings,
    outcome: ExpertReleaseMatrixDecisionOutcome,
    reason: ExpertReleaseMatrixDecisionReason,
    assessments: tuple[ExpertReleaseMatrixReplicateAssessment, ...],
    underpowered_dimension_ids: tuple[str, ...],
    confirmed_benefit_dimension_ids: tuple[str, ...],
) -> ExpertReleaseMatrixPromotionDecision:
    report = stage_result.release_matrix_report
    dependencies = {
        stage_result.stage_result_record_id,
        report.release_matrix_report_id,
        report.plan_reservation_operation_id,
        attempt.validation_attempt_id,
        attempt.validation_policy_id,
        *stage_result.exact_dependency_ids,
        *attempt.eligibility_dependency_ids,
    }
    return ExpertReleaseMatrixPromotionDecision.mint(
        release_matrix_stage_result_id=stage_result.stage_result_record_id,
        release_matrix_report_id=report.release_matrix_report_id,
        plan_reservation_operation_id=report.plan_reservation_operation_id,
        validation_attempt_id=attempt.validation_attempt_id,
        validation_policy_id=attempt.validation_policy_id,
        promotion_policy_version=settings.policy.promotion.policy_version,
        configuration_fingerprint=attempt.configuration_fingerprint,
        mode=report.mode,
        validation_track=attempt.validation_track,
        outcome=outcome,
        reason=reason,
        replicate_assessments=assessments,
        underpowered_dimension_ids=underpowered_dimension_ids,
        confirmed_benefit_dimension_ids=confirmed_benefit_dimension_ids,
        release_matrix_stage_dependency_ids=stage_result.exact_dependency_ids,
        attempt_dependency_ids=attempt.eligibility_dependency_ids,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )
