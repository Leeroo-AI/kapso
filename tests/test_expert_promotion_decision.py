from __future__ import annotations

import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

import test_expert_release_matrix_reservation as reservation_fixture_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import (
    ExpertValidationTrack,
    ObjectiveDirection,
)
from kapso.cross_run.expert.promotion import (
    ExpertReleaseMatrixPromotionError,
    _decide_bootstrap_promotion,
    _derive_replicate_assessments,
    _maximum_context_lineage_matching_size,
    _underpowered_dimensions,
    decide_expert_release_matrix_promotion,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
    ExpertReleaseMatrixDecisionReason,
    ExpertReleaseMatrixReplicateClassification,
)
from kapso.cross_run.expert.promotion_stage import (
    ExpertReleaseMatrixStageCoordinator,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.settings import ExpertParetoDimensionSettings
from task_adapter_matrix_fixtures import (
    task_adapter_release_matrix_case as original_release_matrix_case,
)
from test_expert_promotion_stage import _completed_runtime
from test_expert_promotion_evidence import _bootstrap_prepared_with_store
from test_expert_release_matrix_reservation import (
    _quality_only_validation_settings,
)
from test_expert_task_evaluation_execution import (
    _parent_prepared_with_additional_case,
)


def _remint(record, **changes):
    values = record.to_dict()
    values.pop(record.IDENTITY_FIELD)
    values.update(changes)
    return type(record).mint(**values)


def _settings(*, minimum_replicates: int, minimum_pairs: int):
    settings = _quality_only_validation_settings()
    quality = replace(
        settings.policy.promotion.pareto_dimensions[0],
        hard_regression_ratio=0.1,
    )
    return replace(
        settings,
        policy=replace(
            settings.policy,
            promotion=replace(
                settings.policy.promotion,
                minimum_replicates_per_cell=minimum_replicates,
                minimum_distinct_context_lineage_pairs=minimum_pairs,
                pareto_dimensions=(quality,),
            ),
        ),
    )


def _two_repeat_release_matrix_case(**arguments):
    selected_arguments = dict(arguments)
    selected_arguments["seed_or_replicate_ids"] = ("repeat_1", "repeat_2")
    return original_release_matrix_case(**selected_arguments)


def _publish_parent_stage(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(
        reservation_fixture_module,
        "_quality_only_validation_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        reservation_fixture_module,
        "task_adapter_release_matrix_case",
        _two_repeat_release_matrix_case,
    )
    validation_store, snapshot, prepared = _parent_prepared_with_additional_case(
        tmp_path,
        monkeypatch,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    committed = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    ).publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    assert snapshot.latest_attempt is not None
    return committed.stage_result, snapshot.latest_attempt, validation_store.settings


def _publish_bootstrap_stage(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(
        reservation_fixture_module,
        "_quality_only_validation_settings",
        lambda: settings,
    )
    validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
        tmp_path,
        monkeypatch,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    committed = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    ).publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    assert snapshot.latest_attempt is not None
    return committed.stage_result, snapshot.latest_attempt, validation_store.settings


@pytest.fixture(scope="module")
def accepted_release_matrices(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        parent = _publish_parent_stage(
            tmp_path_factory.mktemp("parent-promotion-decision"),
            monkeypatch,
            _settings(minimum_replicates=1, minimum_pairs=2),
        )
        bootstrap = _publish_bootstrap_stage(
            tmp_path_factory.mktemp("bootstrap-promotion-decision"),
            monkeypatch,
            _settings(minimum_replicates=1, minimum_pairs=1),
        )
    return SimpleNamespace(parent=parent, bootstrap=bootstrap)


def _stage_dependencies(report, stage_result):
    dependencies = {
        report.validation_attempt_id,
        stage_result.authorization_transition_id,
        stage_result.authorization_state_id,
        report.candidate_id,
        report.scope_contract_id,
        report.validation_policy_id,
        report.plan_reservation_operation_id,
        stage_result.task_evaluation_reservation_id,
        report.release_matrix_report_id,
        *report.exact_dependency_ids,
    }
    if report.source_base_release_id is not None:
        dependencies.add(report.source_base_release_id)
    return tuple(sorted(dependencies))


def _report_dependencies(
    report,
    plan,
    rows,
    task_evidence,
    *,
    validation_attempt_id=None,
):
    dependencies = {
        (
            report.validation_attempt_id
            if validation_attempt_id is None
            else validation_attempt_id
        ),
        report.candidate_id,
        report.candidate_commit_record_id,
        report.scope_contract_id,
        report.validation_policy_id,
        report.plan_reservation_operation_id,
        plan.evaluation_plan_id,
        *plan.exact_dependency_ids,
        *(row.comparison_row_id for row in rows),
        *(dependency for row in rows for dependency in row.exact_dependency_ids),
    }
    if report.source_base_release_id is not None:
        dependencies.add(report.source_base_release_id)
    if task_evidence is not None:
        dependencies.update(
            {
                task_evidence.task_execution_evidence_id,
                *task_evidence.exact_dependency_ids,
            }
        )
    return tuple(sorted(dependencies))


def _stage_with_normalized_effects(stage_result, normalized_effects):
    report = stage_result.release_matrix_report
    effect_iterator = iter(normalized_effects)
    rows = []
    for cell, row in zip(
        report.evaluation_plan.evaluation_cells,
        report.evidence_rows,
        strict=True,
    ):
        candidate_values = {}
        source_base_values = {}
        binding = cell.metric_comparison_binding
        for replicate_id in cell.evaluation_fingerprint.seed_or_replicate_ids:
            normalized_effect = next(effect_iterator)
            aligned_effect = normalized_effect * binding.comparison_scale
            raw_effect = (
                aligned_effect
                if binding.objective_direction is ObjectiveDirection.MAXIMIZE
                else -aligned_effect
            )
            candidate_values[replicate_id] = raw_effect
            source_base_values[replicate_id] = 0.0
        rows.append(
            _remint(
                row,
                candidate_replicate_values=candidate_values,
                control_replicate_values=source_base_values,
            )
        )
    with pytest.raises(StopIteration):
        next(effect_iterator)
    selected_rows = tuple(rows)
    reminted_report = _remint(
        report,
        evidence_rows=selected_rows,
        exact_dependency_ids=_report_dependencies(
            report,
            report.evaluation_plan,
            selected_rows,
            report.task_execution_evidence,
        ),
    )
    return _remint(
        stage_result,
        release_matrix_report=reminted_report,
        exact_dependency_ids=_stage_dependencies(reminted_report, stage_result),
    )


def _rebind_validation_track(stage_result, attempt, validation_track):
    reminted_attempt = _remint(attempt, validation_track=validation_track)
    report = stage_result.release_matrix_report
    plan = report.evaluation_plan
    cells = tuple(
        _remint(
            cell,
            validation_attempt_id=reminted_attempt.validation_attempt_id,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        *(
                            dependency
                            for dependency in cell.exact_dependency_ids
                            if dependency != attempt.validation_attempt_id
                        ),
                        reminted_attempt.validation_attempt_id,
                    }
                )
            ),
        )
        for cell in plan.evaluation_cells
    )
    reminted_plan = _remint(
        plan,
        validation_attempt_id=reminted_attempt.validation_attempt_id,
        evaluation_cells=cells,
        external_dependency_ids=tuple(
            sorted(
                {
                    *(
                        dependency
                        for dependency in plan.external_dependency_ids
                        if dependency != attempt.validation_attempt_id
                    ),
                    reminted_attempt.validation_attempt_id,
                }
            )
        ),
    )
    rows = tuple(
        _remint(row, evaluation_cell_id=cell.evaluation_cell_id)
        for row, cell in zip(report.evidence_rows, cells, strict=True)
    )
    task_evidence = report.task_execution_evidence
    assert task_evidence is not None
    reservation_dependencies = tuple(
        sorted(
            (
                reminted_plan.evaluation_plan_id
                if dependency == plan.evaluation_plan_id
                else dependency
            )
            for dependency in task_evidence.reservation_dependency_ids
        )
    )
    request_dependencies = tuple(
        sorted(
            (
                reminted_plan.evaluation_plan_id
                if dependency == plan.evaluation_plan_id
                else dependency
            )
            for dependency in task_evidence.request_dependency_ids
        )
    )
    reminted_task_evidence = _remint(
        task_evidence,
        reservation_dependency_ids=reservation_dependencies,
        request_dependency_ids=request_dependencies,
        exact_dependency_ids=tuple(
            sorted(
                {
                    task_evidence.reservation_id,
                    task_evidence.request_id,
                    *reservation_dependencies,
                    *request_dependencies,
                    *task_evidence.execution_journal_event_ids,
                }
            )
        ),
    )
    report_dependencies = _report_dependencies(
        report,
        reminted_plan,
        rows,
        reminted_task_evidence,
        validation_attempt_id=reminted_attempt.validation_attempt_id,
    )
    reminted_report = _remint(
        report,
        validation_attempt_id=reminted_attempt.validation_attempt_id,
        evaluation_plan=reminted_plan,
        task_execution_evidence=reminted_task_evidence,
        evidence_rows=rows,
        exact_dependency_ids=report_dependencies,
    )
    reminted_stage = _remint(
        stage_result,
        validation_attempt_id=reminted_attempt.validation_attempt_id,
        release_matrix_report=reminted_report,
        exact_dependency_ids=_stage_dependencies(reminted_report, stage_result),
    )
    return reminted_stage, reminted_attempt


def _replicate_count(stage_result):
    return sum(
        len(cell.evaluation_fingerprint.seed_or_replicate_ids)
        for cell in stage_result.release_matrix_report.evaluation_plan.evaluation_cells
    )


@pytest.mark.parametrize(
    ("direction", "candidate_value", "parent_value", "expected_effect"),
    (
        (ObjectiveDirection.MAXIMIZE, 3.0, 1.0, 1.0),
        (ObjectiveDirection.MINIMIZE, 8.0, 10.0, 1.0),
        (ObjectiveDirection.MINIMIZE, -1.0, -1.0, 0.0),
    ),
)
def test_replicate_math_aligns_direction_scale_and_signed_zero(
    direction,
    candidate_value,
    parent_value,
    expected_effect,
):
    dimension = ExpertParetoDimensionSettings(
        dimension_id="quality",
        direction=direction,
        hard_regression_ratio=0.1,
        noise_floor_ratio=0.01,
    )
    cell_id = content_id("expert-release-matrix-evaluation-cell", {"case": 1})
    cell = SimpleNamespace(
        evaluation_cell_id=cell_id,
        evaluation_fingerprint=SimpleNamespace(seed_or_replicate_ids=("repeat_1",)),
        metric_comparison_binding=SimpleNamespace(
            comparison_dimension_id="quality",
            comparison_scale=2.0,
        ),
    )
    row = SimpleNamespace(
        candidate_replicate_values={"repeat_1": candidate_value},
        control_replicate_values={"repeat_1": parent_value},
    )
    stage_result = SimpleNamespace(
        release_matrix_report=SimpleNamespace(
            evaluation_plan=SimpleNamespace(evaluation_cells=(cell,)),
            evidence_rows=(row,),
        )
    )

    assessment = _derive_replicate_assessments(
        stage_result,
        {"quality": dimension},
    )[0]

    assert assessment.normalized_effect == expected_effect
    if expected_effect == 0.0:
        assert math.copysign(1.0, assessment.normalized_effect) == 1.0


def test_exact_noise_and_hard_boundaries_are_not_crossed(
    accepted_release_matrices,
):
    stage_result, attempt, settings = accepted_release_matrices.parent
    dimension = settings.policy.promotion.pareto_dimensions[0]
    effects = [
        -dimension.hard_regression_ratio,
        dimension.noise_floor_ratio,
        -dimension.noise_floor_ratio,
        math.nextafter(dimension.noise_floor_ratio, math.inf),
    ]
    assert len(effects) == _replicate_count(stage_result)
    selected_stage = _stage_with_normalized_effects(stage_result, effects)

    decision = decide_expert_release_matrix_promotion(
        stage_result=selected_stage,
        attempt=attempt,
        settings=settings,
    )
    assessments = {
        assessment.normalized_effect: assessment
        for assessment in decision.replicate_assessments
    }

    assert assessments[-dimension.hard_regression_ratio].hard_regression is False
    assert (
        assessments[-dimension.hard_regression_ratio].classification
        is ExpertReleaseMatrixReplicateClassification.MATERIAL_REGRESSION
    )
    assert (
        assessments[dimension.noise_floor_ratio].classification
        is ExpertReleaseMatrixReplicateClassification.TIE
    )
    assert (
        assessments[-dimension.noise_floor_ratio].classification
        is ExpertReleaseMatrixReplicateClassification.TIE
    )
    assert any(
        assessment.classification is ExpertReleaseMatrixReplicateClassification.GAIN
        for assessment in decision.replicate_assessments
    )


def test_one_bad_repeat_trips_the_hard_gate(accepted_release_matrices):
    stage_result, attempt, settings = accepted_release_matrices.parent
    dimension = settings.policy.promotion.pareto_dimensions[0]
    effects = [dimension.noise_floor_ratio * 2] * _replicate_count(stage_result)
    effects[-1] = math.nextafter(-dimension.hard_regression_ratio, -math.inf)

    decision = decide_expert_release_matrix_promotion(
        stage_result=_stage_with_normalized_effects(stage_result, effects),
        attempt=attempt,
        settings=settings,
    )

    assert decision.outcome is ExpertReleaseMatrixDecisionOutcome.FAILED
    assert decision.reason is ExpertReleaseMatrixDecisionReason.HARD_REGRESSION
    assert sum(item.hard_regression for item in decision.replicate_assessments) == 1


def test_nonfinite_derived_effect_fails_loud(accepted_release_matrices):
    stage_result, attempt, settings = accepted_release_matrices.parent
    effects = [0.0] * _replicate_count(stage_result)
    selected_stage = _stage_with_normalized_effects(stage_result, effects)
    report = selected_stage.release_matrix_report
    first_row = report.evidence_rows[0]
    replicate_id = next(iter(first_row.candidate_replicate_values))
    overflow_row = _remint(
        first_row,
        candidate_replicate_values={replicate_id: sys.float_info.max},
        control_replicate_values={replicate_id: -sys.float_info.max},
    )
    rows = (overflow_row, *report.evidence_rows[1:])
    overflow_report = _remint(
        report,
        evidence_rows=rows,
        exact_dependency_ids=_report_dependencies(
            report,
            report.evaluation_plan,
            rows,
            report.task_execution_evidence,
        ),
    )
    overflow_stage = _remint(
        selected_stage,
        release_matrix_report=overflow_report,
        exact_dependency_ids=_stage_dependencies(overflow_report, selected_stage),
    )

    with pytest.raises(ExpertReleaseMatrixPromotionError, match="nonfinite"):
        decide_expert_release_matrix_promotion(
            stage_result=overflow_stage,
            attempt=attempt,
            settings=settings,
        )


@pytest.mark.parametrize(
    ("effects", "outcome", "reason"),
    (
        (
            (0.02, 0.02, 0.02, 0.02),
            ExpertReleaseMatrixDecisionOutcome.APPROVED,
            ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT,
        ),
        (
            (0.02, 0.0, 0.0, 0.0),
            ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            ExpertReleaseMatrixDecisionReason.NOISY_GAIN,
        ),
        (
            (0.02, -0.02, 0.0, 0.0),
            ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            ExpertReleaseMatrixDecisionReason.GAIN_REGRESSION_TRADEOFF,
        ),
        (
            (-0.02, 0.0, 0.0, 0.0),
            ExpertReleaseMatrixDecisionOutcome.FAILED,
            ExpertReleaseMatrixDecisionReason.UNCOMPENSATED_REGRESSION,
        ),
    ),
)
def test_parent_pareto_outcomes_are_deterministic(
    accepted_release_matrices,
    effects,
    outcome,
    reason,
):
    stage_result, attempt, settings = accepted_release_matrices.parent
    stage_result, attempt = _rebind_validation_track(
        stage_result,
        attempt,
        ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
    )
    selected_stage = _stage_with_normalized_effects(stage_result, effects)

    first = decide_expert_release_matrix_promotion(
        stage_result=selected_stage,
        attempt=attempt,
        settings=settings,
    )
    second = decide_expert_release_matrix_promotion(
        stage_result=selected_stage,
        attempt=attempt,
        settings=settings,
    )

    assert first.outcome is outcome
    assert first.reason is reason
    assert first.to_json_bytes() == second.to_json_bytes()


@pytest.mark.parametrize(
    ("validation_track", "outcome", "reason"),
    (
        (
            ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
            ExpertReleaseMatrixDecisionOutcome.APPROVED,
            ExpertReleaseMatrixDecisionReason.MECHANICAL_NON_REGRESSION,
        ),
        (
            ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
            ExpertReleaseMatrixDecisionOutcome.FAILED,
            ExpertReleaseMatrixDecisionReason.NO_BENEFIT,
        ),
    ),
)
def test_all_ties_only_receive_the_trusted_mechanical_exception(
    accepted_release_matrices,
    validation_track,
    outcome,
    reason,
):
    stage_result, attempt, settings = accepted_release_matrices.parent
    if validation_track is not attempt.validation_track:
        stage_result, attempt = _rebind_validation_track(
            stage_result,
            attempt,
            validation_track,
        )
    selected_stage = _stage_with_normalized_effects(
        stage_result,
        [0.0] * _replicate_count(stage_result),
    )

    decision = decide_expert_release_matrix_promotion(
        stage_result=selected_stage,
        attempt=attempt,
        settings=settings,
    )

    assert decision.outcome is outcome
    assert decision.reason is reason


def test_replicate_and_context_lineage_power_are_independent(
    accepted_release_matrices,
):
    stage_result, _attempt, settings = accepted_release_matrices.parent
    cells = stage_result.release_matrix_report.evaluation_plan.evaluation_cells
    dimensions = {
        dimension.dimension_id: dimension
        for dimension in settings.policy.promotion.pareto_dimensions
    }

    assert _underpowered_dimensions(
        cells,
        dimensions,
        minimum_replicates_per_cell=2,
        minimum_distinct_pairs=1,
    ) == ("quality",)
    assert _underpowered_dimensions(
        cells,
        dimensions,
        minimum_replicates_per_cell=1,
        minimum_distinct_pairs=4,
    ) == ("quality",)
    assert (
        _underpowered_dimensions(
            cells,
            dimensions,
            minimum_replicates_per_cell=1,
            minimum_distinct_pairs=2,
        )
        == ()
    )


def test_maximum_matching_finds_an_augmenting_path_instead_of_greedy_counting():
    def cell(context_id, lineage_id):
        return SimpleNamespace(
            task_context_binding=SimpleNamespace(task_context_binding_id=context_id),
            independence_identity_id=lineage_id,
        )

    cells = (
        cell("context-a", "lineage-a"),
        cell("context-a", "lineage-b"),
        cell("context-b", "lineage-a"),
    )

    assert _maximum_context_lineage_matching_size(cells) == 2


def test_bootstrap_approves_powered_coverage_and_retains_missing_power(
    accepted_release_matrices,
):
    stage_result, attempt, settings = accepted_release_matrices.bootstrap

    approved = decide_expert_release_matrix_promotion(
        stage_result=stage_result,
        attempt=attempt,
        settings=settings,
    )
    underpowered_settings = replace(
        settings,
        policy=replace(
            settings.policy,
            promotion=replace(
                settings.policy.promotion,
                minimum_replicates_per_cell=2,
                minimum_distinct_context_lineage_pairs=2,
            ),
        ),
    )
    retained = _decide_bootstrap_promotion(
        stage_result=stage_result,
        attempt=attempt,
        settings=underpowered_settings,
        dimensions={
            dimension.dimension_id: dimension
            for dimension in underpowered_settings.policy.promotion.pareto_dimensions
        },
    )

    assert approved.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
    assert (
        approved.reason
        is ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE
    )
    assert approved.replicate_assessments == ()
    assert retained.outcome is ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED
    assert retained.reason is ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE
    assert retained.underpowered_dimension_ids == ("quality",)


def test_public_reducer_rejects_attempt_or_configuration_substitution(
    accepted_release_matrices,
):
    stage_result, attempt, settings = accepted_release_matrices.parent

    with pytest.raises(ExpertReleaseMatrixPromotionError, match="differs"):
        decide_expert_release_matrix_promotion(
            stage_result=stage_result,
            attempt=_remint(
                attempt,
                validation_track=ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
            ),
            settings=settings,
        )

    changed_settings = replace(
        settings,
        policy=replace(
            settings.policy,
            promotion=replace(
                settings.policy.promotion,
                policy_version="promotion.substituted.v1",
            ),
        ),
    )
    with pytest.raises(ExpertReleaseMatrixPromotionError, match="differs"):
        decide_expert_release_matrix_promotion(
            stage_result=stage_result,
            attempt=attempt,
            settings=changed_settings,
        )


def test_fixture_uses_the_sealed_stage_record_boundary(accepted_release_matrices):
    stage_result, _attempt, _settings_value = accepted_release_matrices.parent

    assert type(stage_result) is ExpertReleaseMatrixStageResultRecord
    assert stage_result.release_matrix_report.task_execution_evidence is not None
