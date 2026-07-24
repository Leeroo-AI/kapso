"""One-shot, durable current-frontier authority for run-scoped actions."""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import Lock
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
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
    RunActionLedgerSnapshot,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_RECOVERY_COORDINATOR_AUTHORITY,
    RunActionRecoveryCoordinator,
    RunActionRecoveryImplementationRegistry,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_MUTATION_AUTHORITY,
    RunActionAcceptance,
    _RunActionExecutionSession,
    RunActionFrontierBinding,
    RunActionReservation,
    RunActionResultDisposition,
    RunActionResultReceipt,
    RunActionStoreInspection,
    RunActionTerminalReason,
    RunActionViewBinding,
    RunActionWorkspaceBinding,
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
    """Nonserializable one-shot capability for one durable reservation."""

    action_intent_id: str
    reservation_id: str
    run_checkpoint_id: str
    safety_state_id: str
    generation_id: str
    journal_head_id: str
    journal_size_bytes: int
    bundle_digest: str
    bundle_size_bytes: int
    view_identities: tuple[RunStateViewIdentity, ...]
    workspace_frontier: RunWorkspaceFrontierIdentity | None
    _reservation: RunActionReservation = field(repr=False, compare=False)
    _predecessor_ledger: RunActionLedgerSnapshot = field(
        repr=False,
        compare=False,
    )
    _frontier: ReconciledRunFrontier = field(repr=False, compare=False)
    _gate_identity: object = field(repr=False, compare=False)
    _owner_process_id: int = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        reservation = self._reservation
        frontier = self._frontier
        workspace_binding = reservation.frontier.workspace_before
        if (
            type(reservation) is not RunActionReservation
            or type(self._predecessor_ledger) is not RunActionLedgerSnapshot
            or type(frontier) is not ReconciledRunFrontier
            or self.action_intent_id != reservation.intent.action_intent_id
            or self.reservation_id != reservation.reservation_id
            or reservation.predecessor_ledger_snapshot_id
            != self._predecessor_ledger.ledger_snapshot_id
            or self.run_checkpoint_id != frontier.run_checkpoint_id
            or self.safety_state_id != frontier.checkpoint.safety_state.safety_state_id
            or self.generation_id != frontier.generation_id
            or self.journal_head_id != frontier.journal_head_id
            or self.journal_size_bytes != frontier.journal_size_bytes
            or self.bundle_digest != frontier.bundle_digest
            or self.bundle_size_bytes != frontier.bundle_size_bytes
            or self.view_identities != frontier.view_identities
            or reservation.frontier
            != bind_run_action_frontier(frontier, self.workspace_frontier)
            or (None if workspace_binding is None else workspace_binding.to_identity())
            != self.workspace_frontier
            or type(self._gate_identity) is not object
            or type(self._owner_process_id) is not int
            or self._owner_process_id <= 0
            or self._authority is not _USE_PERMIT_AUTHORITY
        ):
            raise RunFrontierActionError(
                "run frontier use permit lacks exact sealed authority"
            )

    @property
    def intent(self) -> RunActionIntent:
        return self._reservation.intent


class RunFrontierUseLease:
    """Live shared-checkpoint authority valid inside one provider invocation."""

    __slots__ = (
        "_active",
        "_authority",
        "_claimed",
        "_descriptors",
        "_frontier",
        "_gate",
        "_owner_process_id",
        "_reservation",
        "_result_receipt",
        "_security_observation",
        "_session",
        "_spawn_commit",
        "_workspace_descriptor",
        "_workspace_frontier",
        "__weakref__",
    )

    def __init__(
        self,
        seal: object,
        *,
        gate: "RunFrontierActionGate",
        reservation: RunActionReservation,
        frontier: ReconciledRunFrontier,
        security_observation: SecurityDenylistObservation,
        workspace_descriptor: int | None,
        workspace_frontier: RunWorkspaceFrontierIdentity | None,
        session: _RunActionExecutionSession,
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
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_reservation", reservation)
        object.__setattr__(self, "_result_receipt", None)
        object.__setattr__(
            self,
            "_security_observation",
            security_observation,
        )
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_spawn_commit", None)
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
        return self._reservation.intent.action_intent_id

    @property
    def run_checkpoint_id(self) -> str:
        return self._frontier.run_checkpoint_id

    @property
    def safety_state_id(self) -> str:
        return self._frontier.checkpoint.safety_state.safety_state_id


class _RunFrontierUseContext:
    """Single-entry context retaining checkpoint and workspace locks."""

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
        self._gate._exit(self._lease, exception_type)
        self._lease = None
        return False


class RunFrontierActionGate:
    """Issue and consume durable action authority from a reconciled frontier."""

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
        self._action_store = publisher._action_store
        self._security_authority = security_authority
        self._gate_identity = object()
        self._owner_process_id = os.getpid()
        self._registry_lock = Lock()
        self._issued_permits: dict[int, RunFrontierUsePermit] = {}
        self._active_leases: dict[int, RunFrontierUseLease] = {}

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
            implementation_registry=implementation_registry,
            _authority=_RUN_ACTION_RECOVERY_COORDINATOR_AUTHORITY,
        )

    def issue(
        self,
        frontier: ReconciledRunFrontier,
        *,
        kind: RunFrontierActionKind,
        boundary: RunSafetyBoundary,
        operation_id: str,
        request_payload: bytes,
        workspace_access: RunFrontierWorkspaceAccess,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> RunFrontierUsePermit:
        """Persist complete request bytes before returning an action capability."""
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
            with self._action_store._session(
                reservation,
                _authority=_RUN_ACTION_MUTATION_AUTHORITY,
            ) as session:
                session.reserve(request_payload)
        permit = RunFrontierUsePermit(
            action_intent_id=intent.action_intent_id,
            reservation_id=reservation.reservation_id,
            run_checkpoint_id=frontier.run_checkpoint_id,
            safety_state_id=checkpoint.safety_state.safety_state_id,
            generation_id=frontier.generation_id,
            journal_head_id=frontier.journal_head_id,
            journal_size_bytes=frontier.journal_size_bytes,
            bundle_digest=frontier.bundle_digest,
            bundle_size_bytes=frontier.bundle_size_bytes,
            view_identities=frontier.view_identities,
            workspace_frontier=workspace_frontier,
            _reservation=reservation,
            _predecessor_ledger=inspection.ledger,
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
        """Return a one-entry context that consumes one durable reservation."""
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
        provider_execution_id: str,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> int | None:
        """Durably commit the exact provider invocation before it may spawn."""
        self._require_live_lease(lease, kind=kind, require_claimed=False)
        if (
            type(boundary_identity) is not RunActionBoundaryIdentity
            or boundary_identity != lease._reservation.intent.boundary_identity
        ):
            raise RunFrontierActionError(
                "run action claim boundary differs from its reservation"
            )
        required_security = lease._security_observation
        current_security = self._security_authority.observe_exact_descendant_of(
            scope_id=required_security.scope_id,
            scope_contract_id=required_security.scope_contract_id,
            checked_subject_ids=required_security.checked_subject_ids,
            required_ancestor=required_security,
        )
        if (
            type(current_security) is not SecurityDenylistObservation
            or current_security != required_security
        ):
            raise RunFrontierActionError(
                "run safety state must be refreshed before external action"
            )
        spawn_commit = lease._session.commit_spawn(
            provider_execution_id=provider_execution_id,
            security_observation_id=(current_security.observation_id),
            boundary_identity=boundary_identity,
        )
        object.__setattr__(lease, "_claimed", True)
        object.__setattr__(lease, "_spawn_commit", spawn_commit)
        return lease._workspace_descriptor

    def record_result(
        self,
        lease: RunFrontierUseLease,
        *,
        result_payload: bytes,
    ) -> RunActionResultReceipt:
        """Persist complete provider output before adapter interpretation."""
        self._require_live_lease(
            lease,
            kind=lease._reservation.intent.kind,
            require_claimed=True,
        )
        if lease._result_receipt is not None:
            raise RunFrontierActionError("run action result was already recorded")
        receipt = lease._session.record_result(
            spawn_commit=lease._spawn_commit,
            result_payload=result_payload,
        )
        object.__setattr__(lease, "_result_receipt", receipt)
        return receipt

    def accept_result(
        self,
        lease: RunFrontierUseLease,
        *,
        result_receipt: RunActionResultReceipt,
        disposition: RunActionResultDisposition,
        accepted_result_payload: bytes,
    ) -> RunActionAcceptance:
        """Persist adapter acceptance and the exact post-action workspace."""
        self._require_live_lease(
            lease,
            kind=lease._reservation.intent.kind,
            require_claimed=True,
        )
        if result_receipt is not lease._result_receipt:
            raise RunFrontierActionError(
                "run action acceptance requires its live result receipt"
            )
        after = self._inspect_workspace_after(lease)
        return lease._session.accept_result(
            result_receipt=result_receipt,
            disposition=disposition,
            accepted_result_payload=accepted_result_payload,
            workspace_after=after,
        )

    def interrupt(
        self,
        lease: RunFrontierUseLease,
        *,
        reason: RunActionTerminalReason,
    ) -> None:
        """Close a committed spawn whose complete result cannot be recovered."""
        self._require_live_lease(
            lease,
            kind=lease._reservation.intent.kind,
            require_claimed=True,
        )
        if lease._result_receipt is not None:
            raise RunFrontierActionError(
                "received run action result cannot be marked interrupted"
            )
        lease._session.interrupt(
            reason=reason,
            workspace_after=self._inspect_workspace_after(lease),
        )

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
        intent = permit.intent
        observed_intent = RunActionIntent.from_request(
            kind=intent.kind,
            boundary=intent.boundary,
            operation_id=intent.operation_id,
            request_payload=request_payload,
            workspace_access=intent.workspace_access,
            boundary_identity=intent.boundary_identity,
        )
        if observed_intent != intent:
            raise RunFrontierActionError(
                "run frontier use permit authorizes another request"
            )
        with ExitStack() as descriptors:
            checkpoint = self._publisher._hold_current(
                permit._frontier,
                descriptors,
            )
            self._action_store.lock_workspace(
                intent.workspace_access,
                descriptors,
            )
            if not self._permit_matches(permit):
                raise RunFrontierActionError(
                    "run frontier use permit differs from its current receipt"
                )
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
            session = descriptors.enter_context(
                self._action_store._session(
                    permit._reservation,
                    _authority=_RUN_ACTION_MUTATION_AUTHORITY,
                )
            )
            if (
                len(session.events) != 1
                or session.events[0].event_kind
                is not RunActionExecutionEventKind.INTENT_RESERVED
            ):
                raise RunFrontierActionError(
                    "run action reservation is no longer spawnable"
                )
            inspection = self._action_store.inspect()
            self._require_exact_reserved_prefix(permit, inspection)
            retained_descriptors = descriptors.pop_all()
        lease = RunFrontierUseLease(
            _USE_LEASE_AUTHORITY,
            gate=self,
            reservation=permit._reservation,
            frontier=permit._frontier,
            security_observation=security,
            workspace_descriptor=workspace_descriptor,
            workspace_frontier=workspace_frontier,
            session=session,
            descriptors=retained_descriptors,
        )
        with self._registry_lock:
            self._active_leases[id(lease)] = lease
        return lease

    def _exit(
        self,
        lease: RunFrontierUseLease,
        exception_type,
    ) -> None:
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
            if lease._session.events and lease._session.events[-1].event_kind in {
                RunActionExecutionEventKind.RESULT_ACCEPTED,
                RunActionExecutionEventKind.INTERRUPTED,
            }:
                self._require_terminal_workspace_unchanged(lease)
            if exception_type is None and (
                not lease._session.events
                or lease._session.events[-1].event_kind
                not in {
                    RunActionExecutionEventKind.RESULT_ACCEPTED,
                    RunActionExecutionEventKind.INTERRUPTED,
                }
            ):
                raise RunFrontierActionError(
                    "normal run action exit requires one durable terminal result"
                )

    def _require_terminal_workspace_unchanged(
        self,
        lease: RunFrontierUseLease,
    ) -> None:
        terminal = lease._session.events[-1]
        expected = (
            terminal.acceptance.workspace_after
            if terminal.event_kind is RunActionExecutionEventKind.RESULT_ACCEPTED
            else terminal.workspace_after
        )
        observed = self._inspect_workspace_after(lease)
        if (
            None
            if observed is None
            else RunActionWorkspaceBinding.from_identity(observed)
        ) != expected:
            raise RunFrontierActionError(
                "run action workspace changed after its durable terminal event"
            )

    def _require_live_lease(
        self,
        lease: RunFrontierUseLease,
        *,
        kind: RunFrontierActionKind,
        require_claimed: bool,
    ) -> None:
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
            or lease._owner_process_id != os.getpid()
            or lease._reservation.intent.kind is not kind
            or lease._claimed is not require_claimed
        ):
            state = "claimed" if not require_claimed else "unclaimed"
            raise RunFrontierActionError(
                f"run frontier use lease is foreign, expired, {state}, or mismatched"
            )

    def _inspect_workspace_after(
        self,
        lease: RunFrontierUseLease,
    ) -> RunWorkspaceFrontierIdentity | None:
        access = lease._reservation.intent.workspace_access
        if access is RunFrontierWorkspaceAccess.NONE:
            return None
        self._active_workspace._require_execution_workspace(
            lease._workspace_descriptor,
            lease._workspace_frontier.workspace_identity,
        )
        return inspect_run_workspace_frontier(
            lease._workspace_descriptor,
            settings=self._publisher._settings,
            expected_commit_sha=(
                None
                if access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                else lease._workspace_frontier.commit_sha
            ),
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

    def _require_frontier_available(
        self,
        frontier: ReconciledRunFrontier,
        inspection: RunActionStoreInspection,
    ) -> RunActionWorkspaceBinding | None:
        inspection.ledger.require_predecessor(frontier.projection.action_ledger)
        terminal_kinds = {
            RunActionExecutionEventKind.RESULT_ACCEPTED,
            RunActionExecutionEventKind.CANCELLED,
            RunActionExecutionEventKind.INTERRUPTED,
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
        workspace_pairs = inspection.workspace_chain(ordered)
        if any(before != after for before, after in workspace_pairs):
            raise RunFrontierActionError(
                "run checkpoint workspace frontier awaits reconciliation"
            )
        return None if not workspace_pairs else workspace_pairs[-1][1]

    @staticmethod
    def _require_exact_reserved_prefix(
        permit: RunFrontierUsePermit,
        inspection: RunActionStoreInspection,
    ) -> None:
        inspection.ledger.require_predecessor(permit._predecessor_ledger)
        expected_event_count = permit._predecessor_ledger.event_count + 1
        operation_events = inspection.events_for(
            permit._reservation.intent.operation_id
        )
        if (
            inspection.ledger.event_count != expected_event_count
            or len(operation_events) != 1
            or operation_events[0].reservation != permit._reservation
            or operation_events[0].event_kind
            is not RunActionExecutionEventKind.INTENT_RESERVED
        ):
            raise RunFrontierActionError(
                "run action reservation is not the exact live ledger successor"
            )

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
