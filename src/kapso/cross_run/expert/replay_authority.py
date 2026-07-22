"""Fresh external authority fencing for expert source-replay execution."""

from __future__ import annotations

from typing import Protocol

from kapso.cross_run.expert.replay_authority_contracts import (
    ExpertSourceReplayFreshAuthorityError,
    SourceReplayCurrentReleaseObservation,
    SourceReplaySecurityDenylistObservation,
    SourceReplaySpawnAuthorityFence,
    SourceReplayTaskAdapterTrustObservation,
    source_replay_spawn_security_subject_ids,
    source_replay_task_adapter_trust_observations,
)
from kapso.cross_run.expert.replay_protocol_contracts import (
    ExpertSourceReplayInvocationAllocation,
)
from kapso.cross_run.expert.replay_execution import (
    ResolvedExpertSourceReplayExecutionCase,
)
from kapso.cross_run.expert.replay_execution_store import (
    ExpertSourceReplayExecutionStore,
    SourceReplayInvocationAllocationPermit,
    SourceReplaySpawnPermit,
)
from kapso.cross_run.expert.replay_request import (
    MaterializedExpertSourceReplayCase,
    PreparedExpertSourceReplayRequest,
)
from kapso.cross_run.expert.validation_store import (
    ExpertSourceReplayReservationSnapshot,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)


class ExpertSourceReplayReservationAuthority(Protocol):
    """Reopen the exact current local reservation under a short shared lock."""

    def reopen_source_replay_reservation(
        self,
        *,
        reservation_id: str,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> ExpertSourceReplayReservationSnapshot: ...


class ExpertSourceReplayCurrentReleaseAuthority(Protocol):
    """Fetch and authenticate the live expert CURRENT without a stale fallback."""

    def current_release_observation(
        self,
        scope_id: str,
    ) -> SourceReplayCurrentReleaseObservation: ...


class ExpertSourceReplaySecurityDenylistAuthority(Protocol):
    """Authenticate current non-rollback denylist state for the exact subjects."""

    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SourceReplaySecurityDenylistObservation: ...


class ExpertSourceReplayFreshAuthorityCoordinator:
    """Build one moment-in-time spawn fence between exact local reopens."""

    def __init__(
        self,
        reservation_authority: ExpertSourceReplayReservationAuthority,
        execution_store: ExpertSourceReplayExecutionStore,
        current_release_authority: ExpertSourceReplayCurrentReleaseAuthority,
        task_adapter_authority: VerifiedTaskAdapterProvider,
        security_denylist_authority: ExpertSourceReplaySecurityDenylistAuthority,
    ) -> None:
        self.reservation_authority = reservation_authority
        if not isinstance(execution_store, ExpertSourceReplayExecutionStore):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn authority requires its canonical execution store"
            )
        self.execution_store = execution_store
        self.current_release_authority = current_release_authority
        self.task_adapter_authority = task_adapter_authority
        self.security_denylist_authority = security_denylist_authority
        self.execution_store._bind_spawn_authority(type(self))

    def commit_spawn(
        self,
        *,
        prepared_request: PreparedExpertSourceReplayRequest,
        reservation_id: str,
        invocation_permit: SourceReplayInvocationAllocationPermit,
        resolved_case: ResolvedExpertSourceReplayExecutionCase,
    ) -> SourceReplaySpawnPermit:
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn authority requires a prepared request"
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
        if not isinstance(
            invocation_permit,
            SourceReplayInvocationAllocationPermit,
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn authority requires a live invocation allocation permit"
            )
        if type(resolved_case) is not ResolvedExpertSourceReplayExecutionCase:
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn authority requires an exact provider resolution"
            )
        invocation_allocation = invocation_permit.require_current_allocation(
            self.execution_store
        )
        first = self.reservation_authority.reopen_source_replay_reservation(
            reservation_id=reservation_id,
            prepared_request=prepared,
        )
        request = prepared.request
        if (
            not isinstance(first, ExpertSourceReplayReservationSnapshot)
            or first.request != request
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn reopen differs from the prepared request"
            )
        reservation = first.reservation
        if invocation_allocation.reservation_id != reservation.reservation_id:
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn allocation differs from the reopened reservation"
            )
        materialized_case = _allocated_materialized_case(
            prepared,
            invocation_allocation,
        )
        if resolved_case.materialized_case != materialized_case:
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn provider resolution differs from the allocated case"
            )
        resolved_case.require_current_provider_identity()
        scope_id = prepared.parent.release_manifest.scope_id
        if (
            prepared.parent.release_manifest.scope_contract_id
            != request.scope_contract_id
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn parent uses another scope contract"
            )
        current_observation = (
            self.current_release_authority.current_release_observation(scope_id)
        )
        if (
            not isinstance(
                current_observation,
                SourceReplayCurrentReleaseObservation,
            )
            or current_observation.scope_id != scope_id
            or current_observation.release_id != request.parent_release_id
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn current release differs from reserved authority"
            )
        adapter_observations = self._reverify_adapters(prepared)
        if not any(
            observation.task_adapter_manifest_id
            == materialized_case.task_adapter.manifest.task_adapter_manifest_id
            and observation.verification_receipt_id
            == materialized_case.task_adapter.verification_receipt.verification_receipt_id
            for observation in adapter_observations
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn allocated case lacks adapter trust authority"
            )
        checked_subject_ids = source_replay_spawn_security_subject_ids(
            prepared,
            reservation,
            current_observation,
            adapter_observations,
        )
        denylist_observation = self.security_denylist_authority.observe_exact(
            scope_id=scope_id,
            scope_contract_id=request.scope_contract_id,
            checked_subject_ids=checked_subject_ids,
        )
        if (
            not isinstance(
                denylist_observation,
                SourceReplaySecurityDenylistObservation,
            )
            or denylist_observation.checked_subject_ids != checked_subject_ids
            or denylist_observation.denied_subject_ids
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn denylist authority rejected the exact dependency closure"
            )
        second = self.reservation_authority.reopen_source_replay_reservation(
            reservation_id=reservation_id,
            prepared_request=prepared,
        )
        if (
            not isinstance(second, ExpertSourceReplayReservationSnapshot)
            or second != first
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn reservation changed during external checks"
            )
        fence = SourceReplaySpawnAuthorityFence.mint(
            reservation_id=reservation.reservation_id,
            execution_request_id=request.execution_request_id,
            authorization_transition_id=reservation.authorization_transition_id,
            authorization_state_id=reservation.authorization_state_id,
            candidate_id=reservation.candidate_id,
            scope_id=scope_id,
            scope_contract_id=request.scope_contract_id,
            expected_parent_release_id=request.parent_release_id,
            invocation_allocation=invocation_allocation,
            current_release_observation=current_observation,
            task_adapter_trust_observations=adapter_observations,
            security_denylist_observation=denylist_observation,
        )
        authorization = self.execution_store._seal_spawn_authorization(
            coordinator=self,
            allocation_permit=invocation_permit,
            prepared_request=prepared,
            resolved_case=resolved_case,
            fence=fence,
            aggregate_tolerance=(
                prepared.settings.policy.task_evaluation_aggregate_tolerance
            ),
        )
        return self.execution_store._commit_spawn_authorization(
            coordinator=self,
            authorization=authorization,
        )

    def _reverify_adapters(
        self,
        prepared: PreparedExpertSourceReplayRequest,
    ) -> tuple[SourceReplayTaskAdapterTrustObservation, ...]:
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
                raise ExpertSourceReplayFreshAuthorityError(
                    "fresh spawn adapter differs from the prepared byte closure"
                )
        return source_replay_task_adapter_trust_observations(prepared)


def _allocated_materialized_case(
    prepared: PreparedExpertSourceReplayRequest,
    allocation: ExpertSourceReplayInvocationAllocation,
) -> MaterializedExpertSourceReplayCase:
    matches = tuple(
        item
        for item in prepared.cases
        if item.request_case.execution_case_id == allocation.execution_case_id
    )
    if len(matches) != 1:
        raise ExpertSourceReplayFreshAuthorityError(
            "fresh spawn allocation names no unique prepared case"
        )
    request_case = matches[0].request_case
    if allocation.execution_leg_id not in {
        request_case.control_leg.execution_leg_id,
        request_case.candidate_leg.execution_leg_id,
    }:
        raise ExpertSourceReplayFreshAuthorityError(
            "fresh spawn allocation names no prepared leg"
        )
    return matches[0]
