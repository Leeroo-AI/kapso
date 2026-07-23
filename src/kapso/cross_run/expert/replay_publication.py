"""Fresh final authority and atomic validation publication for source replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from kapso.cross_run.contracts import ExpertSourceReplayExecutionReservation
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplayCurrentReleaseObservation,
    source_replay_task_adapter_trust_observations,
)
from kapso.cross_run.expert.replay_comparison import (
    build_expert_source_replay_paired_comparison_receipt,
)
from kapso.cross_run.expert.replay_decision import decide_expert_source_replay_stage
from kapso.cross_run.expert.replay_execution_store import (
    CompletedExpertSourceReplayExecution,
    ExpertSourceReplayExecutionStore,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayPublicationError,
    SourceReplayDecisionPublicationFence,
    _build_expert_source_replay_stage_result_record,
    source_replay_publication_security_subject_ids,
)
from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)

if TYPE_CHECKING:
    from kapso.cross_run.expert.validation_store import (
        ExpertSourceReplayStageCommitResult,
        ExpertValidationStore,
    )


class ExpertSourceReplayPublicationCurrentAuthority(Protocol):
    """Fetch and authenticate the live expert CURRENT without stale fallback."""

    def current_release_observation(
        self,
        scope_id: str,
    ) -> SourceReplayCurrentReleaseObservation: ...


class ExpertSourceReplayPublicationDenylistAuthority(Protocol):
    """Authenticate current non-rollback denylist state for exact subjects."""

    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SecurityDenylistObservation: ...


class ExpertSourceReplayDecisionPublicationCoordinator:
    """Rebuild scientific evidence and commit it under fresh final authority."""

    def __init__(
        self,
        validation_store: ExpertValidationStore,
        execution_store: ExpertSourceReplayExecutionStore,
        current_release_authority: ExpertSourceReplayPublicationCurrentAuthority,
        task_adapter_authority: VerifiedTaskAdapterProvider,
        security_denylist_authority: ExpertSourceReplayPublicationDenylistAuthority,
    ) -> None:
        if not isinstance(execution_store, ExpertSourceReplayExecutionStore):
            raise ExpertSourceReplayPublicationError(
                "source replay publication requires its canonical execution store"
            )
        self.validation_store = validation_store
        self.execution_store = execution_store
        self.current_release_authority = current_release_authority
        self.task_adapter_authority = task_adapter_authority
        self.security_denylist_authority = security_denylist_authority
        self.validation_store._bind_source_replay_publication_authority(self)

    def publish_completed(
        self,
        *,
        completed_execution: CompletedExpertSourceReplayExecution,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> ExpertSourceReplayStageCommitResult:
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertSourceReplayPublicationError(
                "source replay publication requires a prepared request"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            parent=prepared_request.parent,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        replayed, first = (
            self.validation_store.reopen_or_replay_source_replay_publication(
                reservation=reservation,
                prepared_request=prepared,
            )
        )
        if replayed is not None:
            return replayed
        if first is None:
            raise ExpertSourceReplayPublicationError(
                "source replay publication has no reservation authority"
            )
        if first.reservation != reservation or first.request != prepared.request:
            raise ExpertSourceReplayPublicationError(
                "source replay publication reopen differs from completed authority"
            )
        receipt = build_expert_source_replay_paired_comparison_receipt(
            completed_execution=completed_execution,
            execution_store=self.execution_store,
            reservation=reservation,
            prepared_request=prepared,
        )
        decision = decide_expert_source_replay_stage(
            paired_comparison_receipt=receipt,
            prepared_request=prepared,
        )
        request = prepared.request
        parent = prepared.parent.release_manifest
        current = self.current_release_authority.current_release_observation(
            parent.scope_id
        )
        if (
            not isinstance(current, SourceReplayCurrentReleaseObservation)
            or current.scope_id != parent.scope_id
            or current.release_id != request.parent_release_id
        ):
            raise ExpertSourceReplayPublicationError(
                "source replay publication current release differs from its parent"
            )
        adapter_observations = self._reverify_adapters(prepared)
        events = completed_execution.require_exact(
            self.execution_store,
            reservation,
            prepared,
        )
        security_subject_ids = source_replay_publication_security_subject_ids(
            prepared_request=prepared,
            reservation=reservation,
            paired_comparison_receipt=receipt,
            stage_decision=decision,
            execution_events=events,
            current_release_observation=current,
            task_adapter_trust_observations=adapter_observations,
        )
        denylist = self.security_denylist_authority.observe_exact(
            scope_id=parent.scope_id,
            scope_contract_id=request.scope_contract_id,
            checked_subject_ids=security_subject_ids,
        )
        if (
            not isinstance(denylist, SecurityDenylistObservation)
            or denylist.checked_subject_ids != security_subject_ids
            or denylist.matched_revocations
        ):
            raise ExpertSourceReplayPublicationError(
                "source replay publication denylist rejected the exact closure"
            )
        replayed, second = (
            self.validation_store.reopen_or_replay_source_replay_publication(
                reservation=reservation,
                prepared_request=prepared,
            )
        )
        if replayed is not None:
            return replayed
        if second is None:
            raise ExpertSourceReplayPublicationError(
                "source replay publication lost its reservation authority"
            )
        if second != first:
            raise ExpertSourceReplayPublicationError(
                "source replay reservation changed during publication checks"
            )
        fence = SourceReplayDecisionPublicationFence.mint(
            reservation_id=reservation.reservation_id,
            execution_request_id=request.execution_request_id,
            authorization_transition_id=reservation.authorization_transition_id,
            authorization_state_id=reservation.authorization_state_id,
            validation_attempt_id=reservation.validation_attempt_id,
            candidate_id=reservation.candidate_id,
            candidate_tree_hash=reservation.candidate_tree_hash,
            scope_id=parent.scope_id,
            scope_contract_id=request.scope_contract_id,
            expected_parent_release_id=request.parent_release_id,
            validation_policy_id=request.validation_policy_id,
            configuration_fingerprint=request.configuration_fingerprint,
            paired_comparison_receipt_id=receipt.paired_comparison_receipt_id,
            source_replay_stage_decision_id=(decision.source_replay_stage_decision_id),
            outcome=decision.outcome,
            current_release_observation=current,
            task_adapter_trust_observations=adapter_observations,
            security_denylist_observation=denylist,
        )
        result = _build_expert_source_replay_stage_result_record(
            reservation=reservation,
            prepared_request=prepared,
            paired_comparison_receipt=receipt,
            stage_decision=decision,
            publication_authority_fence=fence,
        )
        permit = self.validation_store._seal_source_replay_publication_authority(
            coordinator=self,
            reservation_snapshot=second,
            prepared_request=prepared,
            stage_result=result,
        )
        return self.validation_store._commit_source_replay_publication(
            coordinator=self,
            publication_permit=permit,
        )

    def _reverify_adapters(
        self,
        prepared: PreparedExpertSourceReplayRequest,
    ) -> tuple[TaskAdapterTrustObservation, ...]:
        adapters = {
            (
                item.task_adapter.manifest.task_adapter_manifest_id,
                item.task_adapter.verification_receipt.verification_receipt_id,
            ): item.task_adapter
            for item in prepared.cases
        }
        for (manifest_id, receipt_id), expected in sorted(adapters.items()):
            observed = self.task_adapter_authority.resolve_exact(
                task_adapter_manifest_id=manifest_id,
                verification_receipt_id=receipt_id,
            )
            if not isinstance(observed, VerifiedTaskAdapter) or observed != expected:
                raise ExpertSourceReplayPublicationError(
                    "source replay publication adapter differs from prepared authority"
                )
        return source_replay_task_adapter_trust_observations(prepared)
