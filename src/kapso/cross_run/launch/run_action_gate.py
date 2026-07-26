"""Durable current-frontier reservation authority for run-scoped actions."""

from __future__ import annotations

import os
from contextlib import ExitStack
from typing import Protocol

from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointStatus,
)
from kapso.cross_run.launch.resume_contracts import (
    RunEligibilityDisposition,
    RunSafetyBoundary,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionContractError,
    RunActionIntent,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_RECOVERY_COORDINATOR_AUTHORITY,
    RunActionRecoveryCoordinator,
    RunActionRecoveryImplementationRegistry,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionFrontierBinding,
    RunActionReservation,
    RunActionViewBinding,
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_resource_finalization import (
    require_run_action_resource_finalization_authority,
    RunActionResourceFinalizationAuthority,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_RESERVATION_AUTHORITY,
    RunActionStoreInspection,
)
from kapso.cross_run.launch.run_state_publisher import (
    ReconciledRunFrontier,
    RunStatePublisher,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
    RunWorkspaceFrontierIdentity,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)

RunFrontierActionError = RunActionContractError


def bind_run_action_frontier(
    frontier: ReconciledRunFrontier,
    workspace_before: RunWorkspaceFrontierIdentity | None,
) -> RunActionFrontierBinding:
    """Bind durable action authority to reconciled content and workspace state."""
    if type(frontier) is not ReconciledRunFrontier or (
        workspace_before is not None
        and type(workspace_before) is not RunWorkspaceFrontierIdentity
    ):
        raise RunFrontierActionError(
            "run action binding requires exact reconciled authorities"
        )
    if workspace_before is not None:
        evidence = frontier.checkpoint.safety_state.derivative_frontier.evidence
        expected_commit = evidence.branch_heads.get(workspace_before.branch)
        if expected_commit != workspace_before.commit_sha:
            raise RunFrontierActionError(
                "run action workspace differs from its checkpoint branch frontier"
            )
    return RunActionFrontierBinding.mint(
        bootstrap_pin_id=(
            frontier.checkpoint.safety_state.bootstrap_pin.bootstrap_pin_id
        ),
        run_checkpoint_id=frontier.run_checkpoint_id,
        safety_state_id=frontier.checkpoint.safety_state.safety_state_id,
        security_observation_id=(
            frontier.checkpoint.safety_state.security_observation.observation_id
        ),
        generation_id=frontier.generation_id,
        journal_head_id=frontier.journal_head_id,
        journal_size_bytes=frontier.journal_size_bytes,
        bundle_digest=frontier.bundle_digest,
        bundle_size_bytes=frontier.bundle_size_bytes,
        view_bindings=tuple(
            RunActionViewBinding(
                relative_path=identity.relative_path,
                digest=identity.digest,
                size_bytes=identity.size_bytes,
            )
            for identity in frontier.view_identities
        ),
        workspace_before=(
            None
            if workspace_before is None
            else RunActionWorkspaceBinding.from_identity(workspace_before)
        ),
    )


class RunActionSecurityAuthority(Protocol):
    """Fresh authenticated denylist authority used by action recovery."""

    def observe_exact_descendant_of(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
        required_ancestor: SecurityDenylistObservation,
    ) -> SecurityDenylistObservation: ...


class RunActionCredentialValidityAuthority(Protocol):
    """Trusted broker authority retained outside lifecycle adapter control."""

    def observe_exact(
        self,
        *,
        activated_credential_file_observation_id: str,
        credential_lease_authority_id: str,
    ) -> object: ...


class RunFrontierActionGate:
    """Reserve event 1 and issue the sole coordinator for later transitions."""

    def __init__(
        self,
        *,
        active_workspace: ActiveLaunchWorkspace,
        publisher: RunStatePublisher,
        security_authority: RunActionSecurityAuthority,
        credential_validity_authority: RunActionCredentialValidityAuthority | None,
        resource_finalization_authority: RunActionResourceFinalizationAuthority,
    ) -> None:
        if (
            type(active_workspace) is not ActiveLaunchWorkspace
            or type(publisher) is not RunStatePublisher
            or publisher._authority is not active_workspace
            or not hasattr(security_authority, "observe_exact_descendant_of")
            or (
                credential_validity_authority is not None
                and not hasattr(credential_validity_authority, "observe_exact")
            )
        ):
            raise RunFrontierActionError(
                "run frontier action gate authorities are incompatible"
            )
        active_workspace.require_control_authority()
        require_run_action_resource_finalization_authority(
            resource_finalization_authority,
            publisher._action_store,
            publisher._settings,
        )
        publisher._bind_action_resource_finalization_authority(
            resource_finalization_authority
        )
        self._active_workspace = active_workspace
        self._publisher = publisher
        self._action_store = publisher._action_store
        self._security_authority = security_authority
        self._credential_validity_authority = credential_validity_authority
        self._resource_finalization_authority = resource_finalization_authority
        self._owner_process_id = os.getpid()

    def recovery_coordinator(
        self,
        implementation_registry: RunActionRecoveryImplementationRegistry,
    ) -> RunActionRecoveryCoordinator:
        """Issue the sole recovery authority sharing this gate's live runtime."""
        self._require_owner_process()
        if type(implementation_registry) is not RunActionRecoveryImplementationRegistry:
            raise RunFrontierActionError(
                "run action recovery requires an issued implementation registry"
            )
        return RunActionRecoveryCoordinator(
            active_workspace=self._active_workspace,
            publisher=self._publisher,
            security_authority=self._security_authority,
            credential_validity_authority=self._credential_validity_authority,
            resource_finalization_authority=self._resource_finalization_authority,
            implementation_registry=implementation_registry,
            _authority=_RUN_ACTION_RECOVERY_COORDINATOR_AUTHORITY,
        )

    def reserve(
        self,
        frontier: ReconciledRunFrontier,
        *,
        kind: RunFrontierActionKind,
        boundary: RunSafetyBoundary,
        operation_id: str,
        request_payload: bytes,
        workspace_access: RunFrontierWorkspaceAccess,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> RunActionReservation:
        """Persist complete request bytes and return their durable reservation."""
        self._require_owner_process()
        intent = RunActionIntent.from_request(
            kind=kind,
            boundary=boundary,
            operation_id=operation_id,
            request_payload=request_payload,
            workspace_access=workspace_access,
            boundary_identity=boundary_identity,
        )
        with ExitStack() as descriptors:
            checkpoint = self._publisher._hold_current(frontier, descriptors)
            self._action_store.lock_workspace(
                RunFrontierWorkspaceAccess.READ_ONLY,
                descriptors,
            )
            self._require_actionable(checkpoint, intent)
            inspection = self._action_store.inspect()
            expected_terminal_workspace = self._require_frontier_available(
                frontier,
                inspection,
            )
            observed_workspace = self._inspect_workspace(
                checkpoint,
                descriptors,
            )
            if (
                expected_terminal_workspace is not None
                and expected_terminal_workspace.to_identity() != observed_workspace
            ):
                raise RunFrontierActionError(
                    "run action terminal workspace differs from the live workspace"
                )
            workspace_frontier = (
                None
                if intent.workspace_access is RunFrontierWorkspaceAccess.NONE
                else observed_workspace
            )
            reservation = RunActionReservation.build(
                intent=intent,
                frontier=bind_run_action_frontier(
                    frontier,
                    workspace_frontier,
                ),
                predecessor_ledger=inspection.ledger,
            )
            self._action_store._reserve_action(
                reservation,
                request_payload,
                _authority=_RUN_ACTION_RESERVATION_AUTHORITY,
            )
        return reservation

    @staticmethod
    def _require_actionable(
        checkpoint: RunCheckpoint,
        intent: RunActionIntent,
    ) -> None:
        if (
            type(checkpoint) is not RunCheckpoint
            or checkpoint.status is not RunCheckpointStatus.ACTIVE
            or checkpoint.last_stop is not None
        ):
            raise RunFrontierActionError(
                "stopped or completed run cannot execute an external action"
            )
        safety_state = checkpoint.safety_state
        if safety_state.boundary is not intent.boundary:
            raise RunFrontierActionError(
                "run safety boundary does not authorize this action"
            )
        if safety_state.disposition is RunEligibilityDisposition.SECURITY_BLOCKED:
            raise RunFrontierActionError(
                "security-blocked run cannot execute an external action"
            )

    def _require_frontier_available(
        self,
        frontier: ReconciledRunFrontier,
        inspection: RunActionStoreInspection,
    ) -> RunActionWorkspaceBinding | None:
        inspection.ledger.require_predecessor(frontier.projection.action_ledger)
        terminal_kinds = {
            RunActionExecutionEventKind.PROVIDER_TERMINATED,
            RunActionExecutionEventKind.RESULT_ACCEPTED,
            RunActionExecutionEventKind.CANCELLED,
            RunActionExecutionEventKind.FRONTIER_INVALIDATED,
        }
        if any(
            tail.tail_kind not in terminal_kinds
            for tail in inspection.ledger.operation_tails
        ):
            raise RunFrontierActionError(
                "run frontier has an unresolved durable action"
            )
        ordered = inspection.operations_since(
            frontier.projection.action_ledger,
        )
        for events in ordered:
            reservation = events[0].reservation
            binding = reservation.frontier
            expected_binding = bind_run_action_frontier(
                frontier,
                (
                    None
                    if binding.workspace_before is None
                    else binding.workspace_before.to_identity()
                ),
            )
            if (
                reservation.intent.boundary
                is not frontier.checkpoint.safety_state.boundary
                or binding != expected_binding
            ):
                raise RunFrontierActionError(
                    "run frontier action ledger contains another frontier"
                )
        self._publisher._require_terminal_resource_absence(ordered)
        workspace_pairs = inspection.workspace_chain(ordered)
        if any(before != after for before, after in workspace_pairs):
            raise RunFrontierActionError(
                "run checkpoint workspace frontier awaits reconciliation"
            )
        return None if not workspace_pairs else workspace_pairs[-1][1]

    def _inspect_workspace(
        self,
        checkpoint: RunCheckpoint,
        descriptors: ExitStack,
    ) -> RunWorkspaceFrontierIdentity:
        workspace_descriptor, _identity = (
            self._active_workspace._open_execution_workspace(descriptors)
        )
        return inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=self._publisher._settings,
            expected_commit_sha=self._checkpoint_workspace_commit(checkpoint),
        )

    def _checkpoint_workspace_commit(self, checkpoint: RunCheckpoint) -> str:
        branch = self._publisher._settings.workspace_git_branch
        commit_sha = (
            checkpoint.safety_state.derivative_frontier.evidence.branch_heads.get(
                branch
            )
        )
        if commit_sha is None:
            raise RunFrontierActionError(
                "run checkpoint omits its configured workspace branch"
            )
        return commit_sha

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise RunFrontierActionError(
                "run frontier action gate cannot cross a process boundary"
            )


__all__ = [
    "RunActionSecurityAuthority",
    "bind_run_action_frontier",
    "RunFrontierActionError",
    "RunFrontierActionGate",
    "RunFrontierActionKind",
    "RunFrontierWorkspaceAccess",
]
