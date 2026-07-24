"""One-shot, current-frontier authority for run-scoped external actions."""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import Condition, Lock
from typing import Protocol

from kapso.cross_run.launch.run_action_contracts import (
    RunActionContractError,
    RunActionIntent,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_store import (
    RunActionFrontierBinding,
    RunActionViewBinding,
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointStatus,
)
from kapso.cross_run.launch.resume_contracts import (
    RunEligibilityDisposition,
    RunSafetyBoundary,
)
from kapso.cross_run.launch.run_state_publisher import (
    ReconciledRunFrontier,
    RunStatePublisher,
    RunStateViewIdentity,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
    RunWorkspaceFrontierIdentity,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)

_USE_PERMIT_AUTHORITY = object()
_USE_LEASE_AUTHORITY = object()


RunFrontierActionError = RunActionContractError


def bind_run_action_frontier(
    frontier: ReconciledRunFrontier,
    workspace_before: RunWorkspaceFrontierIdentity | None,
) -> RunActionFrontierBinding:
    """Bind durable action authority to reconciled content and workspace state."""
    if (
        type(frontier) is not ReconciledRunFrontier
        or (
            workspace_before is not None
            and type(workspace_before) is not RunWorkspaceFrontierIdentity
        )
    ):
        raise RunFrontierActionError(
            "run action binding requires exact reconciled authorities"
        )
    if workspace_before is not None:
        expected_commit = (
            frontier.checkpoint.safety_state.derivative_frontier.evidence.branch_heads.get(
                workspace_before.branch
            )
        )
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
    """Fresh authenticated denylist authority required at action consumption."""

    def observe_exact_descendant_of(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
        required_ancestor: SecurityDenylistObservation,
    ) -> SecurityDenylistObservation: ...


@dataclass(frozen=True)
class RunFrontierUsePermit:
    """Nonserializable one-shot authority for one exact external request."""

    action_intent_id: str
    run_checkpoint_id: str
    safety_state_id: str
    generation_id: str
    journal_head_id: str
    journal_size_bytes: int
    bundle_digest: str
    bundle_size_bytes: int
    view_identities: tuple[RunStateViewIdentity, ...]
    workspace_frontier: RunWorkspaceFrontierIdentity | None
    _intent: RunActionIntent = field(repr=False, compare=False)
    _frontier: ReconciledRunFrontier = field(repr=False, compare=False)
    _gate_identity: object = field(repr=False, compare=False)
    _owner_process_id: int = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        frontier = self._frontier
        if (
            type(self._intent) is not RunActionIntent
            or type(frontier) is not ReconciledRunFrontier
            or self.action_intent_id != self._intent.action_intent_id
            or self.run_checkpoint_id != frontier.run_checkpoint_id
            or self.safety_state_id != frontier.checkpoint.safety_state.safety_state_id
            or self.generation_id != frontier.generation_id
            or self.journal_head_id != frontier.journal_head_id
            or self.journal_size_bytes != frontier.journal_size_bytes
            or self.bundle_digest != frontier.bundle_digest
            or self.bundle_size_bytes != frontier.bundle_size_bytes
            or self.view_identities != frontier.view_identities
            or type(self._gate_identity) is not object
            or type(self._owner_process_id) is not int
            or self._owner_process_id <= 0
            or self._authority is not _USE_PERMIT_AUTHORITY
        ):
            raise RunFrontierActionError(
                "run frontier use permit lacks exact sealed authority"
            )
        if self._intent.workspace_access is RunFrontierWorkspaceAccess.NONE:
            if self.workspace_frontier is not None:
                raise RunFrontierActionError(
                    "workspace-free action permit carries workspace authority"
                )
        elif type(self.workspace_frontier) is not RunWorkspaceFrontierIdentity:
            raise RunFrontierActionError(
                "workspace action permit lacks an exact workspace frontier"
            )


class RunFrontierUseLease:
    """Live shared-lock authority valid only inside one boundary invocation."""

    __slots__ = (
        "_active",
        "_authority",
        "_claimed",
        "_descriptors",
        "_frontier",
        "_gate",
        "_intent",
        "_owner_process_id",
        "_security_observation",
        "_workspace_descriptor",
        "_workspace_frontier",
        "__weakref__",
    )

    def __init__(
        self,
        seal: object,
        *,
        gate: "RunFrontierActionGate",
        intent: RunActionIntent,
        frontier: ReconciledRunFrontier,
        security_observation: SecurityDenylistObservation,
        workspace_descriptor: int | None,
        workspace_frontier: RunWorkspaceFrontierIdentity | None,
        descriptors: ExitStack,
    ) -> None:
        if seal is not _USE_LEASE_AUTHORITY:
            raise RunFrontierActionError(
                "run frontier use lease is not action-gate sealed"
            )
        object.__setattr__(self, "_active", True)
        object.__setattr__(self, "_authority", _USE_LEASE_AUTHORITY)
        object.__setattr__(self, "_claimed", False)
        object.__setattr__(self, "_descriptors", descriptors)
        object.__setattr__(self, "_frontier", frontier)
        object.__setattr__(self, "_gate", gate)
        object.__setattr__(self, "_intent", intent)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(
            self,
            "_security_observation",
            security_observation,
        )
        object.__setattr__(
            self,
            "_workspace_descriptor",
            workspace_descriptor,
        )
        object.__setattr__(self, "_workspace_frontier", workspace_frontier)

    def __setattr__(self, name, value) -> None:
        raise RunFrontierActionError("run frontier use lease is immutable")

    @property
    def action_intent_id(self) -> str:
        return self._intent.action_intent_id

    @property
    def run_checkpoint_id(self) -> str:
        return self._frontier.run_checkpoint_id

    @property
    def safety_state_id(self) -> str:
        return self._frontier.checkpoint.safety_state.safety_state_id


class _RunFrontierUseContext:
    """Single-entry context that keeps the publisher lock for the action."""

    __slots__ = ("_entered", "_gate", "_lease", "_permit", "_request_payload")

    def __init__(
        self,
        gate: "RunFrontierActionGate",
        permit: RunFrontierUsePermit,
        request_payload: bytes,
    ) -> None:
        self._gate = gate
        self._permit = permit
        self._request_payload = request_payload
        self._entered = False
        self._lease = None

    def __enter__(self) -> RunFrontierUseLease:
        if self._entered:
            raise RunFrontierActionError(
                "run frontier use context cannot be entered twice"
            )
        self._entered = True
        lease = self._gate._enter(self._permit, self._request_payload)
        self._lease = lease
        return lease

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if self._lease is None:
            raise RunFrontierActionError("run frontier use context was not entered")
        self._gate._exit(self._lease)
        self._lease = None
        return False


@dataclass(frozen=True)
class _PendingWorkspaceAdvance:
    """One completed edit that must be represented by the next checkpoint."""

    action_intent_id: str
    predecessor_checkpoint_id: str
    before: RunWorkspaceFrontierIdentity
    after: RunWorkspaceFrontierIdentity


class RunFrontierActionGate:
    """Issue and consume action authority from one current reconciled frontier."""

    def __init__(
        self,
        *,
        active_workspace: ActiveLaunchWorkspace,
        publisher: RunStatePublisher,
        security_authority: RunActionSecurityAuthority,
    ) -> None:
        if (
            type(active_workspace) is not ActiveLaunchWorkspace
            or type(publisher) is not RunStatePublisher
            or publisher._authority is not active_workspace
            or not hasattr(security_authority, "observe_exact_descendant_of")
        ):
            raise RunFrontierActionError(
                "run frontier action gate authorities are incompatible"
            )
        active_workspace.require_control_authority()
        self._active_workspace = active_workspace
        self._publisher = publisher
        self._security_authority = security_authority
        self._gate_identity = object()
        self._owner_process_id = os.getpid()
        self._registry_lock = Lock()
        self._action_condition = Condition(self._registry_lock)
        self._issued_permits: dict[int, RunFrontierUsePermit] = {}
        self._active_leases: dict[int, RunFrontierUseLease] = {}
        self._reserved_action_intent_ids: set[str] = set()
        self._reserved_operation_ids: set[str] = set()
        self._blocked_checkpoint_ids: set[str] = set()
        self._pending_workspace_advances: dict[
            str,
            _PendingWorkspaceAdvance,
        ] = {}
        self._active_workspace_readers = 0
        self._workspace_edit_active = False
        publisher._bind_action_publication_guard(self)

    def issue(
        self,
        frontier: ReconciledRunFrontier,
        *,
        kind: RunFrontierActionKind,
        boundary: RunSafetyBoundary,
        operation_id: str,
        request_payload: bytes,
        workspace_access: RunFrontierWorkspaceAccess,
    ) -> RunFrontierUsePermit:
        """Bind complete request bytes to the exact current run-state receipt."""
        self._require_owner_process()
        intent = RunActionIntent.from_request(
            kind=kind,
            boundary=boundary,
            operation_id=operation_id,
            request_payload=request_payload,
            workspace_access=workspace_access,
        )
        self._reserve_intent(intent)
        with ExitStack() as descriptors:
            self._acquire_action_slot(
                RunFrontierWorkspaceAccess.READ_ONLY,
                descriptors,
            )
            checkpoint = self._publisher.require_current(frontier)
            self._require_frontier_available(checkpoint.run_checkpoint_id)
            self._require_actionable(checkpoint, intent)
            workspace_frontier = (
                None
                if intent.workspace_access is RunFrontierWorkspaceAccess.NONE
                else self._inspect_workspace(
                    checkpoint,
                    descriptors,
                )
            )
        permit = RunFrontierUsePermit(
            action_intent_id=intent.action_intent_id,
            run_checkpoint_id=frontier.run_checkpoint_id,
            safety_state_id=checkpoint.safety_state.safety_state_id,
            generation_id=frontier.generation_id,
            journal_head_id=frontier.journal_head_id,
            journal_size_bytes=frontier.journal_size_bytes,
            bundle_digest=frontier.bundle_digest,
            bundle_size_bytes=frontier.bundle_size_bytes,
            view_identities=frontier.view_identities,
            workspace_frontier=workspace_frontier,
            _intent=intent,
            _frontier=frontier,
            _gate_identity=self._gate_identity,
            _owner_process_id=self._owner_process_id,
            _authority=_USE_PERMIT_AUTHORITY,
        )
        with self._registry_lock:
            self._issued_permits[id(permit)] = permit
        return permit

    def hold(
        self,
        permit: RunFrontierUsePermit,
        request_payload: bytes,
    ) -> _RunFrontierUseContext:
        """Return a one-entry context that consumes and locks one permit."""
        if type(permit) is not RunFrontierUsePermit:
            raise RunFrontierActionError(
                "run action requires one exact frontier use permit"
            )
        if type(request_payload) is not bytes or not request_payload:
            raise RunFrontierActionError(
                "run action request must be complete non-empty bytes"
            )
        return _RunFrontierUseContext(self, permit, request_payload)

    def claim(
        self,
        lease: RunFrontierUseLease,
        *,
        kind: RunFrontierActionKind,
    ) -> int | None:
        """Consume a live lease at its exact provider/process boundary."""
        if type(lease) is not RunFrontierUseLease:
            raise RunFrontierActionError(
                "run action boundary requires one exact live lease"
            )
        with self._registry_lock:
            issued = self._active_leases.get(id(lease))
            if (
                issued is not lease
                or lease._gate is not self
                or lease._authority is not _USE_LEASE_AUTHORITY
                or not lease._active
                or lease._claimed
                or lease._owner_process_id != os.getpid()
                or lease._intent.kind is not kind
            ):
                raise RunFrontierActionError(
                    "run frontier use lease is foreign, expired, claimed, or mismatched"
                )
            object.__setattr__(lease, "_claimed", True)
        return lease._workspace_descriptor

    def _enter(
        self,
        permit: RunFrontierUsePermit,
        request_payload: bytes,
    ) -> RunFrontierUseLease:
        self._require_owner_process()
        with self._registry_lock:
            issued = self._issued_permits.pop(id(permit), None)
        if (
            issued is not permit
            or permit._authority is not _USE_PERMIT_AUTHORITY
            or permit._gate_identity is not self._gate_identity
            or permit._owner_process_id != os.getpid()
        ):
            raise RunFrontierActionError(
                "run frontier use permit is cloned, foreign, consumed, or expired"
            )
        intent = permit._intent
        observed_intent = RunActionIntent.from_request(
            kind=intent.kind,
            boundary=intent.boundary,
            operation_id=intent.operation_id,
            request_payload=request_payload,
            workspace_access=intent.workspace_access,
        )
        if observed_intent != intent:
            raise RunFrontierActionError(
                "run frontier use permit authorizes another request"
            )
        with ExitStack() as descriptors:
            self._acquire_action_slot(intent.workspace_access, descriptors)
            checkpoint = self._publisher._hold_current(
                permit._frontier,
                descriptors,
            )
            if not self._permit_matches(permit):
                raise RunFrontierActionError(
                    "run frontier use permit differs from its current receipt"
                )
            self._require_frontier_available(checkpoint.run_checkpoint_id)
            self._require_actionable(checkpoint, intent)
            workspace_descriptor = None
            workspace_frontier = None
            if intent.workspace_access is not RunFrontierWorkspaceAccess.NONE:
                workspace_descriptor, _workspace_identity = (
                    self._active_workspace._open_execution_workspace(descriptors)
                )
                workspace_frontier = inspect_run_workspace_frontier(
                    workspace_descriptor,
                    settings=self._publisher._settings,
                    expected_commit_sha=self._checkpoint_workspace_commit(checkpoint),
                )
            if workspace_frontier != permit.workspace_frontier:
                raise RunFrontierActionError(
                    "run action workspace frontier changed after permit issuance"
                )
            security = checkpoint.safety_state.security_observation
            current_security = self._security_authority.observe_exact_descendant_of(
                scope_id=security.scope_id,
                scope_contract_id=security.scope_contract_id,
                checked_subject_ids=security.checked_subject_ids,
                required_ancestor=security,
            )
            if (
                type(current_security) is not SecurityDenylistObservation
                or current_security != security
            ):
                raise RunFrontierActionError(
                    "run safety state must be refreshed before external action"
                )
            retained_descriptors = descriptors.pop_all()
        lease = RunFrontierUseLease(
            _USE_LEASE_AUTHORITY,
            gate=self,
            intent=intent,
            frontier=permit._frontier,
            security_observation=current_security,
            workspace_descriptor=workspace_descriptor,
            workspace_frontier=workspace_frontier,
            descriptors=retained_descriptors,
        )
        with self._registry_lock:
            self._active_leases[id(lease)] = lease
        return lease

    def _exit(self, lease: RunFrontierUseLease) -> None:
        with self._registry_lock:
            issued = self._active_leases.pop(id(lease), None)
        if (
            issued is not lease
            or lease._gate is not self
            or lease._authority is not _USE_LEASE_AUTHORITY
            or not lease._active
            or lease._owner_process_id != os.getpid()
        ):
            raise RunFrontierActionError("run frontier use lease is foreign or expired")
        object.__setattr__(lease, "_active", False)
        with lease._descriptors:
            with ExitStack() as verification_descriptors:
                self._publisher._hold_current(
                    lease._frontier,
                    verification_descriptors,
                )
            if lease._intent.workspace_access is not RunFrontierWorkspaceAccess.NONE:
                self._block_checkpoint(lease.run_checkpoint_id)
                self._active_workspace._require_execution_workspace(
                    lease._workspace_descriptor,
                    lease._workspace_frontier.workspace_identity,
                )
                after = inspect_run_workspace_frontier(
                    lease._workspace_descriptor,
                    settings=self._publisher._settings,
                    expected_commit_sha=(
                        None
                        if lease._intent.workspace_access
                        is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                        else lease._workspace_frontier.commit_sha
                    ),
                )
                if (
                    lease._intent.workspace_access
                    is RunFrontierWorkspaceAccess.READ_ONLY
                ):
                    if after != lease._workspace_frontier:
                        raise RunFrontierActionError(
                            "read-only action changed its workspace frontier"
                        )
                    self._unblock_checkpoint(lease.run_checkpoint_id)
                else:
                    if (
                        after.commit_sha == lease._workspace_frontier.commit_sha
                        or after.parent_commit_shas
                        != (lease._workspace_frontier.commit_sha,)
                    ):
                        raise RunFrontierActionError(
                            "workspace edit must produce one direct successor commit"
                        )
                    self._record_pending_workspace_advance(
                        _PendingWorkspaceAdvance(
                            action_intent_id=lease.action_intent_id,
                            predecessor_checkpoint_id=lease.run_checkpoint_id,
                            before=lease._workspace_frontier,
                            after=after,
                        )
                    )

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise RunFrontierActionError(
                "run frontier action gate cannot cross a process boundary"
            )

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

    def _reserve_intent(self, intent: RunActionIntent) -> None:
        with self._action_condition:
            if (
                intent.action_intent_id in self._reserved_action_intent_ids
                or intent.operation_id in self._reserved_operation_ids
            ):
                raise RunFrontierActionError(
                    "run action intent or operation was already reserved"
                )
            self._reserved_action_intent_ids.add(intent.action_intent_id)
            self._reserved_operation_ids.add(intent.operation_id)

    def _acquire_action_slot(
        self,
        access: RunFrontierWorkspaceAccess,
        descriptors: ExitStack,
    ) -> None:
        with self._action_condition:
            if access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE:
                while self._workspace_edit_active or self._active_workspace_readers:
                    self._action_condition.wait()
                self._workspace_edit_active = True
            else:
                while self._workspace_edit_active:
                    self._action_condition.wait()
                self._active_workspace_readers += 1
        descriptors.callback(self._release_action_slot, access)

    def _release_action_slot(
        self,
        access: RunFrontierWorkspaceAccess,
    ) -> None:
        with self._action_condition:
            if access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE:
                if not self._workspace_edit_active:
                    raise RunFrontierActionError(
                        "workspace edit slot was released twice"
                    )
                self._workspace_edit_active = False
            else:
                if self._active_workspace_readers <= 0:
                    raise RunFrontierActionError(
                        "workspace reader slot was released twice"
                    )
                self._active_workspace_readers -= 1
            self._action_condition.notify_all()

    def _require_frontier_available(self, checkpoint_id: str) -> None:
        with self._action_condition:
            if (
                checkpoint_id in self._blocked_checkpoint_ids
                or checkpoint_id in self._pending_workspace_advances
            ):
                raise RunFrontierActionError(
                    "run checkpoint workspace frontier awaits reconciliation"
                )

    def _block_checkpoint(self, checkpoint_id: str) -> None:
        with self._action_condition:
            self._blocked_checkpoint_ids.add(checkpoint_id)

    def _unblock_checkpoint(self, checkpoint_id: str) -> None:
        with self._action_condition:
            self._blocked_checkpoint_ids.discard(checkpoint_id)

    def _record_pending_workspace_advance(
        self,
        pending: _PendingWorkspaceAdvance,
    ) -> None:
        with self._action_condition:
            if pending.predecessor_checkpoint_id in self._pending_workspace_advances:
                raise RunFrontierActionError(
                    "run checkpoint already has a pending workspace advance"
                )
            self._pending_workspace_advances[pending.predecessor_checkpoint_id] = (
                pending
            )
            self._blocked_checkpoint_ids.discard(pending.predecessor_checkpoint_id)

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

    def _require_publication_candidate(
        self,
        current_checkpoint: RunCheckpoint | None,
        candidate: RunCheckpoint,
    ) -> None:
        """Bind publication to the live workspace and any completed edit."""
        if current_checkpoint is None:
            return
        checkpoint_id = current_checkpoint.run_checkpoint_id
        with self._action_condition:
            if checkpoint_id in self._blocked_checkpoint_ids:
                raise RunFrontierActionError(
                    "blocked workspace frontier cannot publish a checkpoint"
                )
            pending = self._pending_workspace_advances.get(checkpoint_id)
        expected_workspace = (
            self._checkpoint_workspace_commit(current_checkpoint)
            if pending is None
            else pending.after.commit_sha
        )
        with ExitStack() as descriptors:
            workspace_descriptor, _identity = (
                self._active_workspace._open_execution_workspace(descriptors)
            )
            observed = inspect_run_workspace_frontier(
                workspace_descriptor,
                settings=self._publisher._settings,
                expected_commit_sha=expected_workspace,
            )
        if pending is not None and observed != pending.after:
            raise RunFrontierActionError(
                "pending workspace advance changed before publication"
            )
        self._require_candidate_workspace_evidence(
            current_checkpoint,
            candidate,
            pending,
        )

    def _require_candidate_workspace_evidence(
        self,
        current_checkpoint: RunCheckpoint,
        candidate: RunCheckpoint,
        pending: _PendingWorkspaceAdvance | None,
    ) -> None:
        branch = self._publisher._settings.workspace_git_branch
        current_evidence = current_checkpoint.safety_state.derivative_frontier.evidence
        candidate_evidence = candidate.safety_state.derivative_frontier.evidence
        current_ids = {
            advance.branch_advance_id for advance in current_evidence.branch_advances
        }
        new_branch_advances = tuple(
            advance
            for advance in candidate_evidence.branch_advances
            if advance.branch_advance_id not in current_ids and advance.branch == branch
        )
        if pending is None:
            if (
                candidate_evidence.branch_heads.get(branch)
                != current_evidence.branch_heads.get(branch)
                or new_branch_advances
            ):
                raise RunFrontierActionError(
                    "checkpoint changes workspace evidence without a completed edit"
                )
            return
        terminal = tuple(
            advance
            for advance in current_evidence.branch_advances
            if advance.branch == branch
            and advance.commit_sha == pending.before.commit_sha
        )
        predecessor_advance_id = (
            None
            if current_evidence.branch_origin_heads[branch] == pending.before.commit_sha
            else terminal[0].branch_advance_id if len(terminal) == 1 else ""
        )
        if (
            candidate_evidence.branch_heads.get(branch) != pending.after.commit_sha
            or len(new_branch_advances) != 1
            or new_branch_advances[0].predecessor_commit_sha
            != pending.before.commit_sha
            or new_branch_advances[0].commit_sha != pending.after.commit_sha
            or new_branch_advances[0].predecessor_branch_advance_id
            != predecessor_advance_id
            or new_branch_advances[0].authorization_safety_state_id
            != current_checkpoint.safety_state.safety_state_id
        ):
            raise RunFrontierActionError(
                "checkpoint does not exactly account for its workspace edit"
            )

    def _commit_publication(
        self,
        predecessor_checkpoint_id: str | None,
        candidate: RunCheckpoint,
    ) -> None:
        if predecessor_checkpoint_id is None:
            return
        with self._action_condition:
            pending = self._pending_workspace_advances.get(predecessor_checkpoint_id)
            if pending is None:
                return
            branch = self._publisher._settings.workspace_git_branch
            if (
                candidate.predecessor_checkpoint_id != predecessor_checkpoint_id
                or candidate.safety_state.derivative_frontier.evidence.branch_heads.get(
                    branch
                )
                != pending.after.commit_sha
            ):
                raise RunFrontierActionError(
                    "published checkpoint differs from its pending workspace edit"
                )
            del self._pending_workspace_advances[predecessor_checkpoint_id]

    @staticmethod
    def _permit_matches(permit: RunFrontierUsePermit) -> bool:
        frontier = permit._frontier
        return (
            permit.run_checkpoint_id == frontier.run_checkpoint_id
            and permit.safety_state_id
            == frontier.checkpoint.safety_state.safety_state_id
            and permit.generation_id == frontier.generation_id
            and permit.journal_head_id == frontier.journal_head_id
            and permit.journal_size_bytes == frontier.journal_size_bytes
            and permit.bundle_digest == frontier.bundle_digest
            and permit.bundle_size_bytes == frontier.bundle_size_bytes
            and permit.view_identities == frontier.view_identities
        )


__all__ = [
    "RunActionSecurityAuthority",
    "bind_run_action_frontier",
    "RunFrontierActionError",
    "RunFrontierActionGate",
    "RunFrontierActionKind",
    "RunFrontierUseLease",
    "RunFrontierUsePermit",
    "RunFrontierWorkspaceAccess",
]
