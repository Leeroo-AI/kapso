"""Fresh external authority coordination for one task-evaluation spawn."""

from __future__ import annotations

from typing import Protocol

from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationAuthorityError,
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.task_evaluation_authority_projection import (
    build_task_evaluation_spawn_authority_fence,
    task_evaluation_adapter_trust_observations,
    task_evaluation_spawn_security_subject_ids,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderRegistry,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    ExpertTaskEvaluationExecutionStore,
    TaskEvaluationInvocationAllocationPermit,
    TaskEvaluationSpawnPermit,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    PreparedTaskEvaluationRequest,
    TaskEvaluationCurrentReleaseAuthority,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)


class TaskEvaluationReservationAuthority(Protocol):
    """Reopen the exact current task-evaluation reservation."""

    def reopen_task_evaluation_reservation(
        self,
        *,
        reservation_id: str,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> ExpertTaskEvaluationReservationSnapshot: ...


class TaskEvaluationSecurityDenylistAuthority(Protocol):
    """Authenticate current non-rollback state for the exact subject set."""

    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SecurityDenylistObservation: ...


class TaskEvaluationFreshAuthorityCoordinator:
    """Commit one spawn after stable remote checks and exact local reopens."""

    def __init__(
        self,
        *,
        reservation_authority: TaskEvaluationReservationAuthority,
        execution_store: ExpertTaskEvaluationExecutionStore,
        current_release_authority: TaskEvaluationCurrentReleaseAuthority,
        task_adapter_authority: VerifiedTaskAdapterProvider,
        security_denylist_authority: TaskEvaluationSecurityDenylistAuthority,
    ) -> None:
        if not isinstance(execution_store, ExpertTaskEvaluationExecutionStore):
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation authority requires its canonical execution store"
            )
        self.reservation_authority = reservation_authority
        self.execution_store = execution_store
        self.current_release_authority = current_release_authority
        self.task_adapter_authority = task_adapter_authority
        self.security_denylist_authority = security_denylist_authority
        self.execution_store._bind_spawn_authority(type(self))

    def commit_spawn(
        self,
        *,
        prepared_request: PreparedTaskEvaluationRequest,
        reservation_id: str,
        invocation_permit: TaskEvaluationInvocationAllocationPermit,
        provider_registry: TaskEvaluationExecutionProviderRegistry,
    ) -> TaskEvaluationSpawnPermit:
        prepared = _reconstruct_prepared_request(prepared_request)
        if type(invocation_permit) is not TaskEvaluationInvocationAllocationPermit:
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation authority requires a live allocation permit"
            )
        if type(provider_registry) is not TaskEvaluationExecutionProviderRegistry:
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation authority requires the exact provider registry"
            )
        allocation = invocation_permit.require_current_allocation(self.execution_store)
        provider_registry.require_exact_prepared_authority(prepared)

        first_reservation = self._reopen_reservation(
            reservation_id,
            prepared,
        )
        invocation_permit.require_current_reservation_snapshot(
            self.execution_store,
            first_reservation,
        )
        if allocation.reservation_id != first_reservation.reservation.reservation_id:
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation allocation differs from the reopened reservation"
            )

        request = first_reservation.request
        current_before = self._observe_current(
            request.scope_id,
            request.expected_current_release_id,
        )
        self._reverify_adapters(prepared)
        adapter_observations = task_evaluation_adapter_trust_observations(prepared)
        checked_subject_ids = task_evaluation_spawn_security_subject_ids(
            prepared_request=prepared,
            reservation_snapshot=first_reservation,
            invocation_allocation=allocation,
            stable_current_release_observation=current_before,
            task_adapter_trust_observations=adapter_observations,
        )
        denylist_observation = self.security_denylist_authority.observe_exact(
            scope_id=request.scope_id,
            scope_contract_id=request.scope_contract_id,
            checked_subject_ids=checked_subject_ids,
        )
        if (
            type(denylist_observation) is not SecurityDenylistObservation
            or denylist_observation.scope_id != request.scope_id
            or denylist_observation.scope_contract_id != request.scope_contract_id
            or denylist_observation.checked_subject_ids != checked_subject_ids
            or not set(denylist_observation.matched_subject_ids).issubset(
                request.allowed_control_security_subject_ids
            )
        ):
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation denylist differs from exact security authority"
            )

        current_after = self._observe_current(
            request.scope_id,
            request.expected_current_release_id,
        )
        if current_after != current_before:
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation current authority changed during external checks"
            )
        second_reservation = self._reopen_reservation(
            reservation_id,
            prepared,
        )
        if second_reservation != first_reservation:
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation reservation changed during external checks"
            )

        fence = build_task_evaluation_spawn_authority_fence(
            prepared_request=prepared,
            reservation_snapshot=first_reservation,
            invocation_allocation=allocation,
            stable_current_release_observation=current_before,
            task_adapter_trust_observations=adapter_observations,
            security_denylist_observation=denylist_observation,
        )
        authorization = self.execution_store._seal_spawn_authorization(
            coordinator=self,
            allocation_permit=invocation_permit,
            prepared_request=prepared,
            provider_registry=provider_registry,
            fence=fence,
        )
        return self.execution_store._commit_spawn_authorization(
            coordinator=self,
            authorization=authorization,
        )

    def _reopen_reservation(
        self,
        reservation_id: str,
        prepared: PreparedTaskEvaluationRequest,
    ) -> ExpertTaskEvaluationReservationSnapshot:
        reopened = self.reservation_authority.reopen_task_evaluation_reservation(
            reservation_id=reservation_id,
            prepared_request=prepared,
        )
        if (
            type(reopened) is not ExpertTaskEvaluationReservationSnapshot
            or reopened.reservation.reservation_id != reservation_id
            or reopened.request != prepared.plan_join.request
            or reopened.plan_reservation != prepared.plan_join.plan_reservation
        ):
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation reopen differs from prepared authority"
            )
        return reopened

    def _observe_current(
        self,
        scope_id: str,
        expected_release_id: str | None,
    ) -> TaskEvaluationCurrentReleaseObservation:
        observation = self.current_release_authority.observe_task_evaluation_current(
            scope_id
        )
        if (
            type(observation) is not TaskEvaluationCurrentReleaseObservation
            or observation.scope_id != scope_id
            or observation.release_id != expected_release_id
        ):
            raise TaskEvaluationAuthorityError(
                "fresh task-evaluation current release differs from reserved authority"
            )
        return observation

    def _reverify_adapters(
        self,
        prepared: PreparedTaskEvaluationRequest,
    ) -> None:
        for expected in prepared.adapters:
            observed = self.task_adapter_authority.resolve_exact(
                task_adapter_manifest_id=(expected.manifest.task_adapter_manifest_id),
                verification_receipt_id=(
                    expected.verification_receipt.verification_receipt_id
                ),
            )
            if type(observed) is not VerifiedTaskAdapter or observed != expected:
                raise TaskEvaluationAuthorityError(
                    "fresh task-evaluation adapter differs from prepared authority"
                )


def _reconstruct_prepared_request(
    prepared_request: PreparedTaskEvaluationRequest,
) -> PreparedTaskEvaluationRequest:
    if type(prepared_request) is not PreparedTaskEvaluationRequest:
        raise TaskEvaluationAuthorityError(
            "fresh task-evaluation authority requires exact prepared bytes"
        )
    return PreparedTaskEvaluationRequest(
        plan_join=prepared_request.plan_join,
        stored_candidate=prepared_request.stored_candidate,
        candidate=prepared_request.candidate,
        source_base=prepared_request.source_base,
        current_release_observation=prepared_request.current_release_observation,
        cases=prepared_request.cases,
    )
