from __future__ import annotations

import pytest

import kapso.cross_run.expert.promotion_stage as promotion_stage_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import (
    ExpertPromotionState,
    ExpertValidationStage,
)
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalStoreError,
)
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode
from kapso.cross_run.expert.promotion_evidence import (
    derive_expert_release_matrix_report,
)
from kapso.cross_run.expert.promotion_stage import (
    ExpertReleaseMatrixStageCoordinator,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    ExpertTaskEvaluationExecutionStore,
)
from kapso.cross_run.expert.validation_store import (
    ExpertValidationCompareAndSwapError,
    ExpertValidationStore,
    ExpertValidationStoreError,
)
from test_expert_promotion_evidence import (
    _bootstrap_prepared_with_store,
    _complete_execution,
    _execution_runtime,
)
from test_expert_task_evaluation_execution import (
    _parent_prepared_with_additional_case,
)


def _completed_runtime(validation_store, snapshot, prepared):
    reservation, execution_store, registry, authority = _execution_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    completed = _complete_execution(
        prepared=prepared,
        reservation_snapshot=reservation,
        execution_store=execution_store,
        registry=registry,
        authority_coordinator=authority,
    )
    return reservation, execution_store, completed


def test_bootstrap_stage_publishes_exact_report_and_advances_typed_prefix(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
        tmp_path,
        monkeypatch,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    expected_report = derive_expert_release_matrix_report(
        validation_store=validation_store,
        execution_store=execution_store,
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    coordinator = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    )

    with pytest.raises(ExpertValidationStoreError, match="sealed execution"):
        validation_store.publish_release_matrix_stage(expected_report)

    committed = coordinator.publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )

    assert committed.replayed is False
    assert committed.stage_result.release_matrix_report.to_json_bytes() == (
        expected_report.to_json_bytes()
    )
    assert committed.snapshot.state.promotion_state is ExpertPromotionState.VALIDATING
    assert (
        committed.snapshot.state.next_stage
        is ExpertValidationStage.PUBLICATION_ELIGIBILITY
    )
    assert committed.snapshot.accepted_stage_results[-1] == committed.stage_result
    assert (
        committed.snapshot.state.accepted_stage_results[-1].stage
        is ExpertValidationStage.RELEASE_MATRIX
    )
    assert (
        committed.snapshot.state.accepted_stage_results[-1].stage_result_record_id
        == committed.stage_result.stage_result_record_id
    )
    with pytest.raises(ExpertValidationStoreError, match="sealed execution"):
        validation_store.publish_release_matrix_stage(committed.stage_result)


def test_stage_replays_lost_response_and_reopens_offline_after_both_stores_restart(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
        tmp_path,
        monkeypatch,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    coordinator = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    )
    committed = coordinator.publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )

    same_process_replay = coordinator.publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    assert same_process_replay.replayed is True
    assert same_process_replay.stage_result.to_json_bytes() == (
        committed.stage_result.to_json_bytes()
    )
    assert same_process_replay.snapshot == committed.snapshot

    reopened_validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened_execution_store = ExpertTaskEvaluationExecutionStore(
        execution_store.root,
        execution_store.trusted_root,
        execution_store.policy_settings,
    )
    reopened_coordinator = ExpertReleaseMatrixStageCoordinator(
        validation_store=reopened_validation_store,
        execution_store=reopened_execution_store,
    )
    monkeypatch.setattr(
        promotion_stage_module,
        "derive_expert_release_matrix_report",
        lambda **_arguments: pytest.fail(
            "durable release-matrix replay must not recompute evidence"
        ),
    )

    restarted_replay = reopened_coordinator.publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    reopened_snapshot = reopened_validation_store.snapshot(
        committed.snapshot.state.candidate_id
    )

    assert restarted_replay.replayed is True
    assert restarted_replay.stage_result.to_json_bytes() == (
        committed.stage_result.to_json_bytes()
    )
    assert reopened_snapshot == committed.snapshot


def test_unresolved_stage_rejects_foreign_execution_store_and_coordinator(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
        tmp_path,
        monkeypatch,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    coordinator = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    )
    foreign_execution_store = ExpertTaskEvaluationExecutionStore(
        execution_store.root,
        execution_store.trusted_root,
        execution_store.policy_settings,
    )
    with foreign_execution_store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        foreign_completed = session.completed_execution()

    with pytest.raises(ExecutionJournalStoreError, match="journal authority"):
        coordinator.publish_completed(
            completed_execution=foreign_completed,
            reservation_snapshot=reservation,
            prepared_request=prepared,
        )
    with pytest.raises(ExpertValidationStoreError, match="conflicting"):
        ExpertReleaseMatrixStageCoordinator(
            validation_store=validation_store,
            execution_store=execution_store,
        )

    committed = coordinator.publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    assert committed.replayed is False


def test_stale_parent_reservation_cannot_publish_after_authority_invalidation(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _parent_prepared_with_additional_case(
        tmp_path,
        monkeypatch,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    coordinator = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    )
    validation_store.reducer.current_release_provider.release_id = content_id(
        "expert-base-release",
        {"changed": True},
    )
    invalidated = validation_store.publish_current_release_authority_invalidation(
        candidate_id=snapshot.state.candidate_id,
        expected_validation_state_id=snapshot.state.validation_state_id,
    )

    assert invalidated.snapshot.state.promotion_state is ExpertPromotionState.FAILED
    with pytest.raises(ExpertValidationCompareAndSwapError, match="head"):
        coordinator.publish_completed(
            completed_execution=completed,
            reservation_snapshot=reservation,
            prepared_request=prepared,
        )


def test_parent_stage_publishes_candidate_and_control_evidence(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _parent_prepared_with_additional_case(
        tmp_path,
        monkeypatch,
    )
    reservation, execution_store, completed = _completed_runtime(
        validation_store,
        snapshot,
        prepared,
    )
    coordinator = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    )

    committed = coordinator.publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )

    report = committed.stage_result.release_matrix_report
    assert committed.replayed is False
    assert report.mode is ExpertReleaseMatrixMode.PARENT_COMPARISON
    assert all(row.parent_replicate_values is not None for row in report.evidence_rows)
    assert (
        committed.snapshot.state.next_stage
        is ExpertValidationStage.PUBLICATION_ELIGIBILITY
    )
