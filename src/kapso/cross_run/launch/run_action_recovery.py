"""Fail-closed recovery of durable nonterminal run-action prefixes."""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from threading import get_ident, Lock
from typing import Protocol
from weakref import WeakKeyDictionary, WeakValueDictionary

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
    RunActionLedgerSnapshot,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_RECOVERY_AUTHORITY,
    RunActionAcceptance,
    RunActionExecutionEvent,
    RunActionExecutionStore,
    RunActionReservation,
    RunActionResultDisposition,
    RunActionSpawnCommit,
    RunActionStoreInspection,
    RunActionTerminalReason,
    RunActionViewBinding,
    RunActionWorkspaceBinding,
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

_RUN_ACTION_RECOVERY_COORDINATOR_AUTHORITY = object()
_RUN_ACTION_RECOVERY_ADAPTER_REGISTRY_AUTHORITY = object()
_RUN_ACTION_FRESH_SPAWN_AUTHORITY = object()
_ISSUED_RECOVERY_COORDINATORS: WeakValueDictionary[int, object] = WeakValueDictionary()
_ISSUED_RECOVERY_ADAPTER_REGISTRIES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_ISSUED_RECOVERY_ADAPTER_BINDINGS: WeakKeyDictionary[object, tuple] = (
    WeakKeyDictionary()
)
_ISSUED_FRESH_SPAWN_CAPABILITIES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_RECOVERY_COORDINATOR_LOCK = Lock()
_RECOVERY_ADAPTER_REGISTRY_LOCK = Lock()
_FRESH_SPAWN_CAPABILITY_LOCK = Lock()
_TERMINAL_KINDS = {
    RunActionExecutionEventKind.RESULT_ACCEPTED,
    RunActionExecutionEventKind.CANCELLED,
    RunActionExecutionEventKind.INTERRUPTED,
}
_RECOVERY_ADAPTER_METHOD_NAMES = (
    "prepare_fresh",
    "start_once",
    "inspect_committed",
    "reattach",
    "accept_result",
)


class RunActionRecoveryError(RuntimeError):
    """Durable action recovery is unsafe, ambiguous, or incompatible."""


class RunActionCommittedSpawnState(str, Enum):
    """Provider facts admitted after a spawn was durably committed."""

    RESULT_AVAILABLE = "result_available"
    RUNNING_REATTACHABLE = "running_reattachable"
    PROVEN_QUIESCENT_WITHOUT_RESULT = "proven_quiescent_without_result"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunActionPreparedSpawn:
    """Local-only allocation made before a durable spawn commitment."""

    provider_execution_id: str
    boundary_identity: RunActionBoundaryIdentity

    def __post_init__(self) -> None:
        require_identifier(
            self.provider_execution_id,
            "prepared run action provider execution ID",
        )
        if type(self.boundary_identity) is not RunActionBoundaryIdentity:
            raise RunActionRecoveryError(
                "prepared run action lacks its exact boundary identity"
            )


@dataclass(frozen=True)
class RunActionProviderResult:
    """Complete raw bytes returned by one exact provider execution."""

    result_payload: bytes

    def __post_init__(self) -> None:
        if type(self.result_payload) is not bytes or not self.result_payload:
            raise RunActionRecoveryError(
                "recovered provider result must be complete non-empty bytes"
            )


@dataclass(frozen=True)
class RunActionAdapterAcceptance:
    """Deterministic adapter interpretation of one durable raw result."""

    disposition: RunActionResultDisposition
    accepted_result_payload: bytes

    def __post_init__(self) -> None:
        if (
            type(self.disposition) is not RunActionResultDisposition
            or type(self.accepted_result_payload) is not bytes
            or not self.accepted_result_payload
        ):
            raise RunActionRecoveryError("recovered adapter acceptance is invalid")


@dataclass(frozen=True)
class RunActionCommittedSpawnObservation:
    """Read-only provider observation that cannot authorize a fresh spawn."""

    state: RunActionCommittedSpawnState
    result: RunActionProviderResult | None
    reattach_token: str | None

    def __post_init__(self) -> None:
        if type(self.state) is not RunActionCommittedSpawnState:
            raise RunActionRecoveryError(
                "committed-spawn observation uses an unknown state"
            )
        expected_shape = {
            RunActionCommittedSpawnState.RESULT_AVAILABLE: (True, False),
            RunActionCommittedSpawnState.RUNNING_REATTACHABLE: (False, True),
            RunActionCommittedSpawnState.PROVEN_QUIESCENT_WITHOUT_RESULT: (
                False,
                False,
            ),
            RunActionCommittedSpawnState.UNKNOWN: (False, False),
        }[self.state]
        if (
            self.result is not None,
            self.reattach_token is not None,
        ) != expected_shape or (
            self.result is not None and type(self.result) is not RunActionProviderResult
        ):
            raise RunActionRecoveryError(
                "committed-spawn observation payload differs from its state"
            )
        if self.reattach_token is not None:
            require_identifier(
                self.reattach_token,
                "run action reattach token",
            )


class RunActionFreshSpawnCapability:
    """One coordinator-sealed capability that only a fresh adapter path accepts."""

    def __init__(
        self,
        *,
        reservation: RunActionReservation,
        spawn_commit: RunActionSpawnCommit,
        request_payload: bytes,
        workspace_descriptor: int | None,
        _authority: object,
    ) -> None:
        if (
            type(reservation) is not RunActionReservation
            or type(spawn_commit) is not RunActionSpawnCommit
            or spawn_commit.reservation_id != reservation.reservation_id
            or spawn_commit.boundary_identity != reservation.intent.boundary_identity
            or type(request_payload) is not bytes
            or not request_payload
            or tree_or_blob_digest(request_payload) != reservation.request_blob.digest
            or len(request_payload) != reservation.request_blob.size_bytes
            or (reservation.intent.workspace_access is RunFrontierWorkspaceAccess.NONE)
            != (workspace_descriptor is None)
            or (
                workspace_descriptor is not None
                and (type(workspace_descriptor) is not int or workspace_descriptor < 0)
            )
            or _authority is not _RUN_ACTION_FRESH_SPAWN_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "fresh run action spawn capability lacks exact authority"
            )
        self._reservation = reservation
        self._spawn_commit = spawn_commit
        self._request_payload = request_payload
        self._workspace_descriptor = (
            None if workspace_descriptor is None else os.dup(workspace_descriptor)
        )
        if self._workspace_descriptor is not None:
            os.set_inheritable(self._workspace_descriptor, False)
        self._owner_process_id = os.getpid()
        self._invoking_thread_id = None
        self._state = "ready"
        with _FRESH_SPAWN_CAPABILITY_LOCK:
            _ISSUED_FRESH_SPAWN_CAPABILITIES[id(self)] = self

    @property
    def reservation(self) -> RunActionReservation:
        self._require_active_invocation()
        return self._reservation

    @property
    def spawn_commit(self) -> RunActionSpawnCommit:
        self._require_active_invocation()
        return self._spawn_commit

    @property
    def request_payload(self) -> bytes:
        self._require_active_invocation()
        return self._request_payload

    @property
    def workspace_descriptor(self) -> int | None:
        self._require_active_invocation()
        return self._workspace_descriptor

    def _invoke_once(
        self,
        adapter: "RunActionRecoveryAdapter",
    ) -> RunActionProviderResult:
        with self._begin_invocation():
            return adapter.start_once(self)

    def _begin_invocation(self) -> "_RunActionFreshSpawnInvocation":
        with _FRESH_SPAWN_CAPABILITY_LOCK:
            issued = _ISSUED_FRESH_SPAWN_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "ready"
            ):
                raise RunActionRecoveryError(
                    "fresh run action spawn capability is spent, cloned, or foreign"
                )
            self._state = "invoking"
            self._invoking_thread_id = get_ident()
        return _RunActionFreshSpawnInvocation(self)

    def _require_active_invocation(self) -> None:
        with _FRESH_SPAWN_CAPABILITY_LOCK:
            issued = _ISSUED_FRESH_SPAWN_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
            ):
                raise RunActionRecoveryError(
                    "fresh run action spawn capability is not in its one invocation"
                )

    def _finish_invocation(self) -> None:
        with _FRESH_SPAWN_CAPABILITY_LOCK:
            issued = _ISSUED_FRESH_SPAWN_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
            ):
                raise RunActionRecoveryError(
                    "fresh run action spawn capability invocation changed"
                )
            self._state = "spent"
            self._invoking_thread_id = None
            _ISSUED_FRESH_SPAWN_CAPABILITIES.pop(id(self))
        if self._workspace_descriptor is not None:
            os.close(self._workspace_descriptor)
            self._workspace_descriptor = None


class _RunActionFreshSpawnInvocation:
    """Burn one fresh capability on all callback exits."""

    def __init__(self, capability: RunActionFreshSpawnCapability) -> None:
        self._capability = capability

    def __enter__(self) -> RunActionFreshSpawnCapability:
        return self._capability

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self._capability._finish_invocation()
        return False


@dataclass(frozen=True)
class RunActionCommittedSpawnQuery:
    """Read-only execution identity with no request or fresh-spawn authority."""

    reservation: RunActionReservation
    spawn_commit: RunActionSpawnCommit

    def __post_init__(self) -> None:
        if (
            type(self.reservation) is not RunActionReservation
            or type(self.spawn_commit) is not RunActionSpawnCommit
            or self.spawn_commit.reservation_id != self.reservation.reservation_id
            or self.spawn_commit.boundary_identity
            != self.reservation.intent.boundary_identity
        ):
            raise RunActionRecoveryError(
                "committed run action query lacks exact durable identity"
            )


class RunActionRecoveryAdapter(Protocol):
    """Exact adapter boundary used by the deterministic recovery state machine."""

    boundary_identity: RunActionBoundaryIdentity

    def prepare_fresh(
        self,
        reservation: RunActionReservation,
    ) -> RunActionPreparedSpawn: ...

    def start_once(
        self,
        capability: RunActionFreshSpawnCapability,
    ) -> RunActionProviderResult: ...

    def inspect_committed(
        self,
        query: RunActionCommittedSpawnQuery,
    ) -> RunActionCommittedSpawnObservation: ...

    def reattach(
        self,
        query: RunActionCommittedSpawnQuery,
        observation: RunActionCommittedSpawnObservation,
    ) -> RunActionProviderResult | None: ...

    def accept_result(
        self,
        *,
        request_payload: bytes,
        result_payload: bytes,
        workspace_before: RunActionWorkspaceBinding | None,
        workspace_after: RunActionWorkspaceBinding | None,
    ) -> RunActionAdapterAcceptance: ...


class RunActionRecoveryAdapterRegistry:
    """Process-bound exact-object catalog issued only by boundary composition."""

    def __init__(
        self,
        adapters: tuple[RunActionRecoveryAdapter, ...],
        *,
        _authority: object,
    ) -> None:
        if (
            type(adapters) is not tuple
            or not adapters
            or _authority is not _RUN_ACTION_RECOVERY_ADAPTER_REGISTRY_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action recovery adapter registry lacks issuance authority"
            )
        indexed = {}
        bindings = []
        for adapter in adapters:
            if not hasattr(adapter, "boundary_identity"):
                raise RunActionRecoveryError(
                    "run action recovery adapter lacks a boundary identity"
                )
            identity = adapter.boundary_identity
            implementation = tuple(
                getattr(type(adapter), name, None)
                for name in _RECOVERY_ADAPTER_METHOD_NAMES
            )
            if (
                type(identity) is not RunActionBoundaryIdentity
                or identity.boundary_identity_id in indexed
                or any(method is None for method in implementation)
                or any(
                    getattr(getattr(adapter, name), "__self__", None) is not adapter
                    or getattr(getattr(adapter, name), "__func__", None) is not method
                    for name, method in zip(
                        _RECOVERY_ADAPTER_METHOD_NAMES,
                        implementation,
                    )
                )
            ):
                raise RunActionRecoveryError(
                    "run action recovery adapter registry is ambiguous or invalid"
                )
            indexed[identity.boundary_identity_id] = adapter
            bindings.append(
                (
                    adapter,
                    identity,
                    type(adapter),
                    implementation,
                )
            )
        self._adapters = adapters
        self._owner_process_id = os.getpid()
        with _RECOVERY_ADAPTER_REGISTRY_LOCK:
            _ISSUED_RECOVERY_ADAPTER_REGISTRIES[id(self)] = self
            _ISSUED_RECOVERY_ADAPTER_BINDINGS[self] = tuple(bindings)

    def resolve(
        self,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> RunActionRecoveryAdapter:
        self._require_owner_process()
        if type(boundary_identity) is not RunActionBoundaryIdentity:
            raise RunActionRecoveryError(
                "run action adapter lookup requires an exact boundary identity"
            )
        with _RECOVERY_ADAPTER_REGISTRY_LOCK:
            bindings = _ISSUED_RECOVERY_ADAPTER_BINDINGS.get(self)
        matching = tuple(
            binding
            for binding in bindings
            if binding[1].boundary_identity_id == boundary_identity.boundary_identity_id
        )
        if (
            len(matching) != 1
            or matching[0][1] != boundary_identity
            or matching[0][0].boundary_identity != boundary_identity
            or type(matching[0][0]) is not matching[0][2]
            or any(
                getattr(getattr(matching[0][0], name), "__self__", None)
                is not matching[0][0]
                or getattr(getattr(matching[0][0], name), "__func__", None)
                is not method
                for name, method in zip(
                    _RECOVERY_ADAPTER_METHOD_NAMES,
                    matching[0][3],
                )
            )
        ):
            raise RunActionRecoveryError(
                "run action recovery adapter is absent or substituted"
            )
        return matching[0][0]

    def _require_owner_process(self) -> None:
        with _RECOVERY_ADAPTER_REGISTRY_LOCK:
            issued = _ISSUED_RECOVERY_ADAPTER_REGISTRIES.get(id(self))
            bindings = _ISSUED_RECOVERY_ADAPTER_BINDINGS.get(self)
        if (
            issued is not self
            or bindings is None
            or type(self._adapters) is not tuple
            or len(bindings) != len(self._adapters)
            or any(
                adapter is not binding[0]
                for adapter, binding in zip(self._adapters, bindings)
            )
            or self._owner_process_id != os.getpid()
        ):
            raise RunActionRecoveryError(
                "run action recovery adapter registry is cloned, foreign, or altered"
            )


@dataclass(frozen=True)
class RunActionRecoveryPlan:
    """Read-only classification of actions not yet in the checkpoint projection."""

    projected_ledger: RunActionLedgerSnapshot
    live_ledger: RunActionLedgerSnapshot
    ordered_operation_ids: tuple[str, ...]
    pending_operation_id: str | None

    def __post_init__(self) -> None:
        if (
            type(self.projected_ledger) is not RunActionLedgerSnapshot
            or type(self.live_ledger) is not RunActionLedgerSnapshot
            or self.ordered_operation_ids
            != tuple(dict.fromkeys(self.ordered_operation_ids))
            or (
                self.pending_operation_id is not None
                and (
                    self.pending_operation_id not in self.ordered_operation_ids
                    or self.pending_operation_id != self.ordered_operation_ids[-1]
                )
            )
        ):
            raise RunActionRecoveryError("run action recovery plan is invalid")
        self.live_ledger.require_predecessor(self.projected_ledger)


@dataclass(frozen=True)
class RunActionRecoveredOperation:
    """One terminal durable prefix and its complete accepted bytes, if any."""

    events: tuple[RunActionExecutionEvent, ...]
    accepted_result_payload: bytes | None

    def __post_init__(self) -> None:
        if (
            not self.events
            or any(type(event) is not RunActionExecutionEvent for event in self.events)
            or self.events[-1].event_kind not in _TERMINAL_KINDS
            or (
                self.events[-1].event_kind
                is RunActionExecutionEventKind.RESULT_ACCEPTED
            )
            != (self.accepted_result_payload is not None)
            or (
                self.accepted_result_payload is not None
                and (
                    type(self.accepted_result_payload) is not bytes
                    or not self.accepted_result_payload
                    or tree_or_blob_digest(self.accepted_result_payload)
                    != self.events[-1].acceptance.accepted_result_blob.digest
                    or len(self.accepted_result_payload)
                    != self.events[-1].acceptance.accepted_result_blob.size_bytes
                )
            )
        ):
            raise RunActionRecoveryError("recovered run action operation is invalid")

    @property
    def operation_id(self) -> str:
        return self.events[0].reservation.intent.operation_id


@dataclass(frozen=True)
class RunActionRecoveryReport:
    """Exact terminal replay inputs plus any still-ambiguous operation."""

    frontier_run_checkpoint_id: str
    live_ledger: RunActionLedgerSnapshot
    recovered_operations: tuple[RunActionRecoveredOperation, ...]
    unresolved_operation_id: str | None

    def __post_init__(self) -> None:
        require_content_id(
            self.frontier_run_checkpoint_id,
            "run action recovery frontier checkpoint",
        )
        if (
            self.frontier_run_checkpoint_id.split(":sha256:", 1)[0] != "run-checkpoint"
            or type(self.live_ledger) is not RunActionLedgerSnapshot
            or any(
                type(operation) is not RunActionRecoveredOperation
                for operation in self.recovered_operations
            )
            or tuple(operation.operation_id for operation in self.recovered_operations)
            != tuple(
                dict.fromkeys(
                    operation.operation_id for operation in self.recovered_operations
                )
            )
            or (
                self.unresolved_operation_id is not None
                and self.unresolved_operation_id
                in {operation.operation_id for operation in self.recovered_operations}
            )
        ):
            raise RunActionRecoveryError("run action recovery report is invalid")

    @property
    def is_complete(self) -> bool:
        return self.unresolved_operation_id is None


class RunActionRecoveryCoordinator:
    """Recover one exact live run without ever replaying a committed spawn."""

    def __init__(
        self,
        *,
        active_workspace: ActiveLaunchWorkspace,
        publisher: RunStatePublisher,
        security_authority: object,
        adapter_registry: RunActionRecoveryAdapterRegistry,
        _authority: object,
    ) -> None:
        if (
            type(active_workspace) is not ActiveLaunchWorkspace
            or type(publisher) is not RunStatePublisher
            or publisher._authority is not active_workspace
            or not hasattr(security_authority, "observe_exact_descendant_of")
            or type(adapter_registry) is not RunActionRecoveryAdapterRegistry
            or _authority is not _RUN_ACTION_RECOVERY_COORDINATOR_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action recovery authorities are incompatible"
            )
        active_workspace.require_control_authority()
        self._active_workspace = active_workspace
        self._publisher = publisher
        self._store = publisher._action_store
        self._security_authority = security_authority
        adapter_registry._require_owner_process()
        self._adapter_registry = adapter_registry
        self._owner_process_id = os.getpid()
        with _RECOVERY_COORDINATOR_LOCK:
            _ISSUED_RECOVERY_COORDINATORS[id(self)] = self

    def inspect(
        self,
        frontier: ReconciledRunFrontier,
    ) -> RunActionRecoveryPlan:
        """Classify unprojected operations without contacting any adapter."""
        self._require_owner_process()
        with ExitStack() as descriptors:
            self._publisher._hold_current(frontier, descriptors)
            self._store.lock_workspace(
                RunFrontierWorkspaceAccess.READ_ONLY,
                descriptors,
            )
            inspection = self._store.inspect()
            return self._plan(frontier, inspection)

    def recover(
        self,
        frontier: ReconciledRunFrontier,
    ) -> RunActionRecoveryReport:
        """Advance only the exact admitted durable tail, then replay terminals."""
        self._require_owner_process()
        self._adapter_registry._require_owner_process()
        with ExitStack() as descriptors:
            self._publisher._hold_current(frontier, descriptors)
            self._store.lock_workspace(
                RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
                descriptors,
            )
            inspection = self._store.inspect()
            plan = self._plan(frontier, inspection)
            if plan.pending_operation_id is not None:
                events = inspection.events_for(plan.pending_operation_id)
                with self._store._recovery_session(
                    events[0].reservation,
                    _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
                ) as session:
                    if session.events != events:
                        raise RunActionRecoveryError(
                            "run action changed after recovery inspection"
                        )
                    self._recover_session(
                        frontier,
                        session,
                        descriptors,
                    )
            return self._report(frontier)

    def _recover_session(
        self,
        frontier: ReconciledRunFrontier,
        session,
        descriptors: ExitStack,
    ) -> None:
        reservation = session.reservation
        self._require_reservation_frontier(frontier, reservation)
        events = session.events
        tail_kind = events[-1].event_kind
        if tail_kind is RunActionExecutionEventKind.INTENT_RESERVED:
            if not self._security_is_current(frontier):
                session.cancel(RunActionTerminalReason.STALE_FRONTIER)
                return
            request_payload = session.read_request()
            workspace_descriptor, observed_workspace = self._inspect_workspace(
                reservation,
                descriptors,
                allow_edit_successor=False,
            )
            expected_workspace = reservation.frontier.workspace_before
            if (
                None
                if observed_workspace is None
                else RunActionWorkspaceBinding.from_identity(observed_workspace)
            ) != expected_workspace:
                session.cancel(RunActionTerminalReason.STALE_FRONTIER)
                return
            adapter = self._resolve_adapter(self._adapter_registry, reservation)
            prepared = adapter.prepare_fresh(reservation)
            if (
                type(prepared) is not RunActionPreparedSpawn
                or prepared.boundary_identity != reservation.intent.boundary_identity
            ):
                raise RunActionRecoveryError(
                    "adapter prepared another run action boundary"
                )
            _descriptor, confirmed_workspace = self._inspect_workspace(
                reservation,
                descriptors,
                allow_edit_successor=False,
            )
            if (
                None
                if confirmed_workspace is None
                else RunActionWorkspaceBinding.from_identity(confirmed_workspace)
            ) != expected_workspace:
                session.cancel(RunActionTerminalReason.STALE_FRONTIER)
                return
            if not self._security_is_current(frontier):
                session.cancel(RunActionTerminalReason.STALE_FRONTIER)
                return
            spawn_commit = session.commit_spawn(
                provider_execution_id=prepared.provider_execution_id,
                security_observation_id=(reservation.frontier.security_observation_id),
                boundary_identity=reservation.intent.boundary_identity,
            )
            capability = RunActionFreshSpawnCapability(
                reservation=reservation,
                spawn_commit=spawn_commit,
                request_payload=request_payload,
                workspace_descriptor=workspace_descriptor,
                _authority=_RUN_ACTION_FRESH_SPAWN_AUTHORITY,
            )
            result = capability._invoke_once(adapter)
            self._record_and_accept(
                session,
                adapter,
                result,
                descriptors,
            )
            return
        adapter = self._resolve_adapter(self._adapter_registry, reservation)
        if tail_kind is RunActionExecutionEventKind.SPAWN_COMMITTED:
            spawn_commit = events[-1].spawn_commit
            query = RunActionCommittedSpawnQuery(
                reservation=reservation,
                spawn_commit=spawn_commit,
            )
            observation = adapter.inspect_committed(query)
            if type(observation) is not RunActionCommittedSpawnObservation:
                raise RunActionRecoveryError(
                    "adapter returned an invalid committed-spawn observation"
                )
            if observation.state is RunActionCommittedSpawnState.RESULT_AVAILABLE:
                self._record_and_accept(
                    session,
                    adapter,
                    observation.result,
                    descriptors,
                )
            elif (
                observation.state is RunActionCommittedSpawnState.RUNNING_REATTACHABLE
                and self._security_is_current(frontier)
            ):
                result = adapter.reattach(query, observation)
                if result is not None:
                    self._record_and_accept(
                        session,
                        adapter,
                        result,
                        descriptors,
                    )
            elif (
                observation.state
                is RunActionCommittedSpawnState.PROVEN_QUIESCENT_WITHOUT_RESULT
            ):
                _descriptor, workspace = self._inspect_workspace(
                    reservation,
                    descriptors,
                    allow_edit_successor=True,
                )
                session.interrupt(
                    reason=RunActionTerminalReason.PROVIDER_INTERRUPTED,
                    workspace_after=workspace,
                )
            return
        if tail_kind is RunActionExecutionEventKind.RESULT_RECEIVED:
            self._accept_received(
                session,
                adapter,
                descriptors,
            )
            return
        raise RunActionRecoveryError(
            "run action recovery received a terminal operation"
        )

    def _record_and_accept(
        self,
        session,
        adapter: RunActionRecoveryAdapter,
        result: RunActionProviderResult,
        descriptors: ExitStack,
    ) -> RunActionAcceptance:
        if type(result) is not RunActionProviderResult:
            raise RunActionRecoveryError("adapter returned an invalid provider result")
        spawn_commit = session.events[-1].spawn_commit
        session.record_result(
            spawn_commit=spawn_commit,
            result_payload=result.result_payload,
        )
        return self._accept_received(session, adapter, descriptors)

    def _accept_received(
        self,
        session,
        adapter: RunActionRecoveryAdapter,
        descriptors: ExitStack,
    ) -> RunActionAcceptance:
        reservation = session.reservation
        result_receipt = session.events[-1].result_receipt
        request_payload = session.read_request()
        result_payload = session.read_result(result_receipt)
        _descriptor, observed_workspace = self._inspect_workspace(
            reservation,
            descriptors,
            allow_edit_successor=True,
        )
        after = (
            None
            if observed_workspace is None
            else RunActionWorkspaceBinding.from_identity(observed_workspace)
        )
        acceptance = adapter.accept_result(
            request_payload=request_payload,
            result_payload=result_payload,
            workspace_before=reservation.frontier.workspace_before,
            workspace_after=after,
        )
        repeated_acceptance = adapter.accept_result(
            request_payload=request_payload,
            result_payload=result_payload,
            workspace_before=reservation.frontier.workspace_before,
            workspace_after=after,
        )
        if (
            type(acceptance) is not RunActionAdapterAcceptance
            or repeated_acceptance != acceptance
        ):
            raise RunActionRecoveryError(
                "adapter acceptance is invalid or nondeterministic"
            )
        _descriptor, confirmed_workspace = self._inspect_workspace(
            reservation,
            descriptors,
            allow_edit_successor=True,
        )
        if confirmed_workspace != observed_workspace:
            raise RunActionRecoveryError(
                "workspace changed during run action result acceptance"
            )
        durable_acceptance = session.accept_result(
            result_receipt=result_receipt,
            disposition=acceptance.disposition,
            accepted_result_payload=acceptance.accepted_result_payload,
            workspace_after=confirmed_workspace,
        )
        _descriptor, terminal_workspace = self._inspect_workspace(
            reservation,
            descriptors,
            allow_edit_successor=True,
        )
        if terminal_workspace != confirmed_workspace:
            raise RunActionRecoveryError(
                "workspace changed after durable run action acceptance"
            )
        return durable_acceptance

    def _inspect_workspace(
        self,
        reservation: RunActionReservation,
        descriptors: ExitStack,
        *,
        allow_edit_successor: bool,
    ) -> tuple[int | None, RunWorkspaceFrontierIdentity | None]:
        access = reservation.intent.workspace_access
        if access is RunFrontierWorkspaceAccess.NONE:
            return None, None
        descriptor, _identity = self._active_workspace._open_execution_workspace(
            descriptors
        )
        before = reservation.frontier.workspace_before
        expected_commit_sha = (
            None
            if (
                access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                and allow_edit_successor
            )
            else before.commit_sha
        )
        observed = inspect_run_workspace_frontier(
            descriptor,
            settings=self._publisher._settings,
            expected_commit_sha=expected_commit_sha,
        )
        return descriptor, observed

    def _report(
        self,
        frontier: ReconciledRunFrontier,
    ) -> RunActionRecoveryReport:
        inspection = self._store.inspect()
        operations = inspection.operations_since(frontier.projection.action_ledger)
        recovered = []
        unresolved_operation_id = None
        for events in operations:
            if events[-1].event_kind not in _TERMINAL_KINDS:
                if unresolved_operation_id is not None:
                    raise RunActionRecoveryError(
                        "run action recovery has multiple unresolved operations"
                    )
                unresolved_operation_id = events[0].reservation.intent.operation_id
                continue
            with self._store._recovery_session(
                events[0].reservation,
                _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
            ) as session:
                if session.events != events:
                    raise RunActionRecoveryError(
                        "terminal run action changed during report construction"
                    )
                accepted_payload = (
                    session.read_accepted_result(events[-1].acceptance)
                    if events[-1].event_kind
                    is RunActionExecutionEventKind.RESULT_ACCEPTED
                    else None
                )
            recovered.append(
                RunActionRecoveredOperation(
                    events=events,
                    accepted_result_payload=accepted_payload,
                )
            )
        return RunActionRecoveryReport(
            frontier_run_checkpoint_id=frontier.run_checkpoint_id,
            live_ledger=inspection.ledger,
            recovered_operations=tuple(recovered),
            unresolved_operation_id=unresolved_operation_id,
        )

    @staticmethod
    def _plan(
        frontier: ReconciledRunFrontier,
        inspection: RunActionStoreInspection,
    ) -> RunActionRecoveryPlan:
        if (
            type(frontier) is not ReconciledRunFrontier
            or type(inspection) is not RunActionStoreInspection
        ):
            raise RunActionRecoveryError(
                "run action recovery planning requires exact authorities"
            )
        operations = inspection.operations_since(frontier.projection.action_ledger)
        pending = tuple(
            events[0].reservation.intent.operation_id
            for events in operations
            if events[-1].event_kind not in _TERMINAL_KINDS
        )
        if len(pending) > 1 or (
            pending and pending[0] != operations[-1][0].reservation.intent.operation_id
        ):
            raise RunActionRecoveryError(
                "run action recovery requires one final nonterminal operation"
            )
        for events in operations:
            RunActionRecoveryCoordinator._require_reservation_frontier(
                frontier,
                events[0].reservation,
            )
        return RunActionRecoveryPlan(
            projected_ledger=frontier.projection.action_ledger,
            live_ledger=inspection.ledger,
            ordered_operation_ids=tuple(
                events[0].reservation.intent.operation_id for events in operations
            ),
            pending_operation_id=None if not pending else pending[0],
        )

    @staticmethod
    def _resolve_adapter(
        registry: RunActionRecoveryAdapterRegistry,
        reservation: RunActionReservation,
    ) -> RunActionRecoveryAdapter:
        identity = reservation.intent.boundary_identity
        adapter = registry.resolve(identity)
        required_methods = (
            "prepare_fresh",
            "start_once",
            "inspect_committed",
            "reattach",
            "accept_result",
        )
        if (
            not hasattr(adapter, "boundary_identity")
            or adapter.boundary_identity != identity
            or any(not hasattr(adapter, name) for name in required_methods)
        ):
            raise RunActionRecoveryError(
                "run action recovery adapter differs from its durable identity"
            )
        return adapter

    @staticmethod
    def _require_reservation_frontier(
        frontier: ReconciledRunFrontier,
        reservation: RunActionReservation,
    ) -> None:
        binding = reservation.frontier
        if (
            binding.bootstrap_pin_id
            != frontier.checkpoint.safety_state.bootstrap_pin.bootstrap_pin_id
            or binding.run_checkpoint_id != frontier.run_checkpoint_id
            or binding.safety_state_id
            != frontier.checkpoint.safety_state.safety_state_id
            or binding.security_observation_id
            != frontier.checkpoint.safety_state.security_observation.observation_id
            or binding.generation_id != frontier.generation_id
            or binding.journal_head_id != frontier.journal_head_id
            or binding.journal_size_bytes != frontier.journal_size_bytes
            or binding.bundle_digest != frontier.bundle_digest
            or binding.bundle_size_bytes != frontier.bundle_size_bytes
            or binding.view_bindings
            != tuple(
                RunActionViewBinding(
                    relative_path=identity.relative_path,
                    digest=identity.digest,
                    size_bytes=identity.size_bytes,
                )
                for identity in frontier.view_identities
            )
        ):
            raise RunActionRecoveryError(
                "run action reservation differs from the current frontier"
            )

    def _security_is_current(
        self,
        frontier: ReconciledRunFrontier,
    ) -> bool:
        required = frontier.checkpoint.safety_state.security_observation
        current = self._security_authority.observe_exact_descendant_of(
            scope_id=required.scope_id,
            scope_contract_id=required.scope_contract_id,
            checked_subject_ids=required.checked_subject_ids,
            required_ancestor=required,
        )
        if type(current) is not SecurityDenylistObservation:
            raise RunActionRecoveryError(
                "run action recovery security authority returned another type"
            )
        return current == required

    def _require_owner_process(self) -> None:
        with _RECOVERY_COORDINATOR_LOCK:
            issued = _ISSUED_RECOVERY_COORDINATORS.get(id(self))
        if issued is not self or self._owner_process_id != os.getpid():
            raise RunActionRecoveryError(
                "run action recovery coordinator is cloned or foreign"
            )


__all__ = [
    "RunActionAdapterAcceptance",
    "RunActionCommittedSpawnObservation",
    "RunActionCommittedSpawnQuery",
    "RunActionCommittedSpawnState",
    "RunActionFreshSpawnCapability",
    "RunActionPreparedSpawn",
    "RunActionProviderResult",
    "RunActionRecoveredOperation",
    "RunActionRecoveryAdapter",
    "RunActionRecoveryAdapterRegistry",
    "RunActionRecoveryCoordinator",
    "RunActionRecoveryError",
    "RunActionRecoveryPlan",
    "RunActionRecoveryReport",
]
