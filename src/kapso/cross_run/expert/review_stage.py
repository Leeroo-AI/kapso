"""Restart-aware orchestration for one automated expert-review stage."""

from __future__ import annotations

import tempfile
from pathlib import Path

from kapso.cross_run.contracts import (
    ExpertPromotionState,
    ExpertValidationAttempt,
    ExpertValidationStage,
)
from kapso.cross_run.expert.review import ExpertAutomatedReviewCoordinator
from kapso.cross_run.expert.store import ExpertCandidateStore
from kapso.cross_run.expert.validation_snapshots import ExpertValidationSnapshot
from kapso.cross_run.expert.validation_store import ExpertValidationStore


class ExpertAutomatedReviewStageError(ValueError):
    """The automated review stage lacks exact current authority."""


class ExpertAutomatedReviewStageOrchestrator:
    """Resume, execute, and atomically publish one automated review round."""

    def __init__(
        self,
        *,
        coordinator: ExpertAutomatedReviewCoordinator,
        candidate_store: ExpertCandidateStore,
        validation_store: ExpertValidationStore,
    ) -> None:
        if (
            candidate_store.validator.settings != coordinator.settings
            or candidate_store.root
            != coordinator.workspace_root / coordinator.settings.candidate_path
            or candidate_store.state_root != candidate_store.root.parent
            or validation_store.root
            != coordinator.workspace_root / coordinator.settings.validation.state_path
            or validation_store.state_root != validation_store.root.parent
            or validation_store.settings != coordinator.settings.validation
            or validation_store.reducer.candidate_store is not candidate_store
        ):
            raise ExpertAutomatedReviewStageError(
                "automated review components do not share exact authority"
            )
        self.coordinator = coordinator
        self.candidate_store = candidate_store
        self.validation_store = validation_store
        validation_store._bind_automated_review_publication_authority(coordinator)

    def run(
        self,
        attempt: ExpertValidationAttempt,
    ) -> ExpertValidationSnapshot:
        if type(attempt) is not ExpertValidationAttempt:
            raise ExpertAutomatedReviewStageError(
                "automated review requires a typed validation attempt"
            )
        with self.validation_store.automated_review_stage_lock(attempt.candidate_id):
            return self._run_locked(attempt)

    def _run_locked(
        self,
        attempt: ExpertValidationAttempt,
    ) -> ExpertValidationSnapshot:
        snapshot = self.validation_store.snapshot(attempt.candidate_id)
        if (
            snapshot is None
            or snapshot.latest_attempt != attempt
            or snapshot.state.validation_attempt_id != attempt.validation_attempt_id
        ):
            raise ExpertAutomatedReviewStageError(
                "automated review attempt is not the current candidate authority"
            )
        if self._review_is_resolved(snapshot):
            return snapshot
        if (
            snapshot.state.promotion_state is not ExpertPromotionState.VALIDATING
            or snapshot.state.next_stage is not ExpertValidationStage.AUTOMATED_REVIEW
        ):
            raise ExpertAutomatedReviewStageError(
                "automated review is not the current validation stage"
            )
        stored_candidate = self.candidate_store.read(attempt.candidate_id)
        prepared = self.coordinator.prepare(
            stored_candidate=stored_candidate,
            validation_attempt=attempt,
            authorization_transition_id=snapshot.transition.transition_id,
            authorization_state=snapshot.state,
            accepted_stage_results=snapshot.accepted_stage_results,
        )
        replay = self.validation_store.reopen_or_replay_automated_review(
            prepared.packet
        )
        if replay is not None:
            return replay.snapshot
        with tempfile.TemporaryDirectory(
            prefix=".automated-review-",
            dir=self.validation_store.staging_root,
        ) as workspace_text:
            workspace = Path(workspace_text).resolve()
            execution = self.coordinator.execute(
                prepared,
                workspace=workspace,
            )
        return self.validation_store.publish_automated_review_stage(execution).snapshot

    @staticmethod
    def _review_is_resolved(snapshot: ExpertValidationSnapshot) -> bool:
        accepted = any(
            reference.stage is ExpertValidationStage.AUTOMATED_REVIEW
            for reference in snapshot.state.accepted_stage_results
        )
        terminal_review = (
            snapshot.transition.transition_stage_result_record_id is not None
            and snapshot.transition.transition_stage_result_record_id.split(
                ":sha256:",
                1,
            )[0]
            == "expert-automated-review-stage-result"
            and snapshot.state.promotion_state
            in {ExpertPromotionState.FAILED, ExpertPromotionState.DISPUTED}
        )
        return accepted or terminal_review
