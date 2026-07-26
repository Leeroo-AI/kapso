"""Fail-closed recovery of durable nonterminal run-action prefixes."""

from __future__ import annotations

import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import get_ident, Lock
from typing import Protocol
from weakref import WeakKeyDictionary, WeakValueDictionary

from kapso.cross_run.canonical import (
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.launch.checkpoint_contracts import RunCheckpointStatus
from kapso.cross_run.launch.resume_contracts import RunEligibilityDisposition
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionExecutionLifecycleIdentity,
    RunActionResultInterpreterIdentity,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionResolvedWorkloadObservation,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
    RunActionLedgerSnapshot,
)
from kapso.cross_run.launch.run_action_release_authority import (
    mint_run_action_workload_release_receipt,
    require_run_action_workload_release_receipt_matches_event,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    open_run_action_release_inspection,
    RunActionReleasePresence,
)
from kapso.cross_run.launch.run_action_release_candidate import (
    _RELEASE_CANDIDATE_AUTHORIZATION_AUTHORITY,
    _RunActionFrozenReleaseCandidate,
    _SystemRunActionReleaseClock,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionCredentialValidityObservation,
    RunActionReleaseAuthorizationObservation,
    RunActionWorkloadReleaseAdoption,
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_result_authority import (
    run_action_terminal_result_evidence_matches,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
    RunActionViewBinding,
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    open_run_action_result_workspace,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_RECOVERY_AUTHORITY,
    RunActionAcceptance,
    RunActionExecutionEvent,
    RunActionExecutionStore,
    RunActionResultDecision,
    RunActionResultDisposition,
    RunActionStoreInspection,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionExecutionPolicy,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
    RunActionResultCaptureReceipt,
    RunActionTerminalObservation,
)
from kapso.cross_run.launch.run_action_workspace_promotion import (
    _RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY,
    RunActionWorkspacePromotion,
    RunActionWorkspacePromoter,
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
_RUN_ACTION_RECOVERY_IMPLEMENTATION_REGISTRY_AUTHORITY = object()
_RUN_ACTION_PREPARATION_AUTHORITY = object()
_RUN_ACTION_ACTIVATION_AUTHORITY = object()
_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY = object()
_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY = object()
_RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY = object()
_RUN_ACTION_RESULT_CAPTURE_AUTHORITY = object()
_ISSUED_RECOVERY_COORDINATORS: WeakValueDictionary[int, object] = WeakValueDictionary()
_ISSUED_RECOVERY_IMPLEMENTATION_REGISTRIES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_ISSUED_RECOVERY_IMPLEMENTATION_BINDINGS: WeakKeyDictionary[object, tuple] = (
    WeakKeyDictionary()
)
_ISSUED_PREPARATION_CAPABILITIES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_ISSUED_ACTIVATION_CAPABILITIES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_ISSUED_COMMITTED_CONTINUATION_CAPABILITIES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES: WeakKeyDictionary[
    object,
    "_RunActionIssuedReleaseAuthorities",
] = WeakKeyDictionary()
_RECOVERY_COORDINATOR_LOCK = Lock()
_RECOVERY_IMPLEMENTATION_REGISTRY_LOCK = Lock()
_PREPARATION_CAPABILITY_LOCK = Lock()
_ACTIVATION_CAPABILITY_LOCK = Lock()
_COMMITTED_CONTINUATION_CAPABILITY_LOCK = Lock()
_TERMINAL_KINDS = {
    RunActionExecutionEventKind.PROVIDER_TERMINATED,
    RunActionExecutionEventKind.RESULT_ACCEPTED,
    RunActionExecutionEventKind.CANCELLED,
    RunActionExecutionEventKind.FRONTIER_INVALIDATED,
}
_EXECUTION_ADAPTER_METHOD_NAMES = (
    "prepared_event_size_bound",
    "activation_event_size_bound",
    "release_receipt_size_bound",
    "prepare",
    "stage_activation",
    "inspect_unactivated",
    "inspect_committed",
    "continue_committed_once",
)
_RESULT_INTERPRETER_METHOD_NAMES = ("interpret",)
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_NANOSECONDS_PER_SECOND = 1_000_000_000


class RunActionRecoveryError(RuntimeError):
    """Durable action recovery is unsafe, ambiguous, or incompatible."""


@dataclass(frozen=True)
class _RunActionIssuedReleaseAuthorities:
    """Coordinator-held authorities never accepted at the adapter call site."""

    required_security_observation: SecurityDenylistObservation
    security_authority: object
    credential_validity_authority: object | None
    clock: _SystemRunActionReleaseClock

    def __post_init__(self) -> None:
        if (
            type(self.required_security_observation) is not SecurityDenylistObservation
            or self.required_security_observation.matched_revocations
            or not hasattr(self.security_authority, "observe_exact_descendant_of")
            or (
                self.credential_validity_authority is not None
                and not hasattr(
                    self.credential_validity_authority,
                    "observe_exact",
                )
            )
            or type(self.clock) is not _SystemRunActionReleaseClock
        ):
            raise RunActionRecoveryError(
                "issued release authorities are incomplete or unsafe"
            )


def _require_workspace_source_path(path: Path, descriptor: int) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise RunActionRecoveryError(
            "run action workspace source path is not absolute and symlink-free"
        )
    descriptor_metadata = os.fstat(descriptor)
    path_metadata = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or not stat.S_ISDIR(path_metadata.st_mode)
        or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        != (path_metadata.st_dev, path_metadata.st_ino)
    ):
        raise RunActionRecoveryError(
            "run action workspace source path differs from its live descriptor"
        )


class RunActionPreparationMode(str, Enum):
    """How an adapter must reconcile one already-durable allocation."""

    CREATE_ALLOCATED = "create_allocated"
    REOPEN_ALLOCATED = "reopen_allocated"
    REVALIDATE_PREPARED = "revalidate_prepared"


class RunActionPreparationOrigin(str, Enum):
    """How one exact prepared occurrence was obtained."""

    NEWLY_MATERIALIZED = "newly_materialized"
    REOPENED_ALLOCATION = "reopened_allocation"
    MATERIALIZED_AFTER_PROVEN_ABSENCE = "materialized_after_proven_absence"
    REVALIDATED_PREPARED = "revalidated_prepared"


class RunActionPreparationState(str, Enum):
    """Exact positive states admitted from preparation reconciliation."""

    EXACT_PREPARED = "exact_prepared"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunActionPreparationObservation:
    """Typed preparation result; uncertainty cannot become a terminal claim."""

    state: RunActionPreparationState
    prepared_execution: RunActionPreparedExecution | None
    origin: RunActionPreparationOrigin | None

    def __post_init__(self) -> None:
        if type(self.state) is not RunActionPreparationState:
            raise RunActionRecoveryError(
                "run action preparation observation uses an unknown state"
            )
        if self.state is RunActionPreparationState.EXACT_PREPARED:
            if (
                type(self.prepared_execution) is not RunActionPreparedExecution
                or type(self.origin) is not RunActionPreparationOrigin
            ):
                raise RunActionRecoveryError(
                    "exact preparation observation lacks its occurrence and origin"
                )
        elif self.prepared_execution is not None or self.origin is not None:
            raise RunActionRecoveryError(
                "non-exact preparation observation carries an occurrence"
            )


def _preparation_origin_matches_mode(
    observation: RunActionPreparationObservation,
    mode: RunActionPreparationMode,
) -> bool:
    if (
        type(observation) is not RunActionPreparationObservation
        or type(mode) is not RunActionPreparationMode
    ):
        return False
    if observation.state is not RunActionPreparationState.EXACT_PREPARED:
        return True
    admitted_origins = {
        RunActionPreparationMode.CREATE_ALLOCATED: {
            RunActionPreparationOrigin.NEWLY_MATERIALIZED,
        },
        RunActionPreparationMode.REOPEN_ALLOCATED: {
            RunActionPreparationOrigin.REOPENED_ALLOCATION,
            RunActionPreparationOrigin.MATERIALIZED_AFTER_PROVEN_ABSENCE,
        },
        RunActionPreparationMode.REVALIDATE_PREPARED: {
            RunActionPreparationOrigin.REVALIDATED_PREPARED,
        },
    }
    return observation.origin in admitted_origins[mode]


class RunActionUnactivatedSpawnState(str, Enum):
    """Facts admitted before one durable activation receipt is selected."""

    INERT_ACTIVATABLE = "inert_activatable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunActionUnactivatedSpawnObservation:
    """Closed observation that can never adopt an unauthorized provider start."""

    state: RunActionUnactivatedSpawnState

    def __post_init__(self) -> None:
        if type(self.state) is not RunActionUnactivatedSpawnState:
            raise RunActionRecoveryError(
                "unactivated-spawn observation uses an unknown state"
            )


class RunActionCommittedSpawnState(str, Enum):
    """Provider facts admitted after activation was durably selected."""

    INERT_CONTINUABLE = "inert_continuable"
    RUNNING_CONTINUABLE = "running_continuable"
    TERMINAL_CONTINUABLE = "terminal_continuable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunActionProviderResult:
    """Complete captured bytes and physical evidence from one provider execution."""

    terminal_observation: RunActionTerminalObservation
    result_capture_receipt: RunActionResultCaptureReceipt
    result_payload: bytes

    def __post_init__(self) -> None:
        if (
            type(self.terminal_observation) is not RunActionTerminalObservation
            or type(self.result_capture_receipt) is not RunActionResultCaptureReceipt
            or self.result_capture_receipt.terminal_observation_id
            != self.terminal_observation.terminal_observation_id
            or self.result_capture_receipt.runtime_volume_authority_id
            != self.terminal_observation.runtime_volume_authority_id
            or self.result_capture_receipt.generation_nonce
            != self.terminal_observation.generation_nonce
            or type(self.result_payload) is not bytes
            or not self.result_payload
            or self.result_capture_receipt.size_bytes != len(self.result_payload)
            or self.result_capture_receipt.content_digest
            != tree_or_blob_digest(self.result_payload)
        ):
            raise RunActionRecoveryError(
                "recovered provider result lacks exact terminal capture evidence"
            )


@dataclass(frozen=True)
class RunActionInterpretedResult:
    """Dependency-pure interpretation of one complete request and raw result."""

    disposition: RunActionResultDisposition
    accepted_result_payload: bytes

    def __post_init__(self) -> None:
        if (
            type(self.disposition) is not RunActionResultDisposition
            or type(self.accepted_result_payload) is not bytes
            or not self.accepted_result_payload
        ):
            raise RunActionRecoveryError("interpreted run action result is invalid")


@dataclass(frozen=True)
class RunActionCommittedSpawnObservation:
    """Read-only provider state whose token can seal one exact continuation."""

    state: RunActionCommittedSpawnState
    observation_token: str | None

    def __post_init__(self) -> None:
        if type(self.state) is not RunActionCommittedSpawnState:
            raise RunActionRecoveryError(
                "committed-spawn observation uses an unknown state"
            )
        continuable = self.state in {
            RunActionCommittedSpawnState.INERT_CONTINUABLE,
            RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
        }
        if continuable != (self.observation_token is not None):
            raise RunActionRecoveryError(
                "committed-spawn observation payload differs from its state"
            )
        if self.observation_token is not None:
            if (
                type(self.observation_token) is not str
                or _SHA256_DIGEST_PATTERN.fullmatch(self.observation_token) is None
            ):
                raise RunActionRecoveryError(
                    "run action committed observation token is invalid"
                )


class RunActionContinuationState(str, Enum):
    """Outcome of consuming one exact committed-observation capability."""

    PENDING = "pending"
    RESULT_CAPTURED = "result_captured"


@dataclass(frozen=True)
class RunActionContinuationOutcome:
    """Typed continuation outcome; pending work carries no provider bytes."""

    state: RunActionContinuationState
    result: RunActionProviderResult | None

    def __post_init__(self) -> None:
        if type(self.state) is not RunActionContinuationState or (
            self.state is RunActionContinuationState.RESULT_CAPTURED
        ) != (self.result is not None):
            raise RunActionRecoveryError(
                "run action continuation outcome differs from its state"
            )
        if self.result is not None and type(self.result) is not RunActionProviderResult:
            raise RunActionRecoveryError(
                "run action continuation outcome carries an invalid result"
            )


class RunActionPreparationCapability:
    """Single-use authority for one durable preparation transition."""

    def __init__(
        self,
        *,
        preparation_allocation: RunActionPreparationAllocation,
        mode: RunActionPreparationMode,
        durable_prepared_execution: RunActionPreparedExecution | None,
        workspace_descriptor: int | None,
        workspace_source_path: Path | None,
        _authority: object,
    ) -> None:
        if type(preparation_allocation) is not RunActionPreparationAllocation:
            raise RunActionRecoveryError(
                "run action preparation capability lacks exact authority"
            )
        claim = preparation_allocation.preparation_claim
        if (
            type(mode) is not RunActionPreparationMode
            or (mode is RunActionPreparationMode.REVALIDATE_PREPARED)
            != (durable_prepared_execution is not None)
            or (
                durable_prepared_execution is not None
                and (
                    type(durable_prepared_execution) is not RunActionPreparedExecution
                    or durable_prepared_execution.preparation_claim != claim
                    or durable_prepared_execution.runtime_volume_authority
                    != preparation_allocation.runtime_volume_authority
                )
            )
            or (
                claim.reservation.intent.workspace_access
                is RunFrontierWorkspaceAccess.NONE
            )
            != (workspace_descriptor is None)
            or (workspace_descriptor is None) != (workspace_source_path is None)
            or (
                workspace_descriptor is not None
                and (type(workspace_descriptor) is not int or workspace_descriptor < 0)
            )
            or _authority is not _RUN_ACTION_PREPARATION_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action preparation capability lacks exact authority"
            )
        if workspace_descriptor is not None:
            _require_workspace_source_path(
                workspace_source_path,
                workspace_descriptor,
            )
        self._preparation_allocation = preparation_allocation
        self._mode = mode
        self._durable_prepared_execution = durable_prepared_execution
        self._workspace_source_path = workspace_source_path
        self._workspace_descriptor = (
            None if workspace_descriptor is None else os.dup(workspace_descriptor)
        )
        if self._workspace_descriptor is not None:
            os.set_inheritable(self._workspace_descriptor, False)
        self._owner_process_id = os.getpid()
        self._invoking_thread_id = None
        self._state = "ready"
        with _PREPARATION_CAPABILITY_LOCK:
            _ISSUED_PREPARATION_CAPABILITIES[id(self)] = self

    @property
    def preparation_allocation(self) -> RunActionPreparationAllocation:
        self._require_active_invocation()
        return self._preparation_allocation

    @property
    def mode(self) -> RunActionPreparationMode:
        self._require_active_invocation()
        return self._mode

    @property
    def durable_prepared_execution(self) -> RunActionPreparedExecution | None:
        self._require_active_invocation()
        return self._durable_prepared_execution

    @property
    def workspace_descriptor(self) -> int | None:
        self._require_active_invocation()
        return self._workspace_descriptor

    @property
    def workspace_source_path(self) -> Path | None:
        self._require_active_invocation()
        return self._workspace_source_path

    def _invoke_once(
        self,
        execution_adapter: "RunActionExecutionAdapter",
    ) -> RunActionPreparationObservation:
        with self._begin_invocation():
            return execution_adapter.prepare(self)

    def _begin_invocation(self) -> "_RunActionPreparationInvocation":
        with _PREPARATION_CAPABILITY_LOCK:
            issued = _ISSUED_PREPARATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "ready"
            ):
                raise RunActionRecoveryError(
                    "run action preparation capability is spent, cloned, or foreign"
                )
            self._state = "invoking"
            self._invoking_thread_id = get_ident()
        return _RunActionPreparationInvocation(self)

    def _require_active_invocation(self) -> None:
        with _PREPARATION_CAPABILITY_LOCK:
            issued = _ISSUED_PREPARATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
            ):
                raise RunActionRecoveryError(
                    "run action preparation capability is not in its one invocation"
                )

    def _finish_invocation(self) -> None:
        with _PREPARATION_CAPABILITY_LOCK:
            issued = _ISSUED_PREPARATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
            ):
                raise RunActionRecoveryError(
                    "run action preparation capability invocation changed"
                )
            self._state = "spent"
            self._invoking_thread_id = None
            _ISSUED_PREPARATION_CAPABILITIES.pop(id(self))
        if self._workspace_descriptor is not None:
            os.close(self._workspace_descriptor)
            self._workspace_descriptor = None


class _RunActionPreparationInvocation:
    """Burn one preparation capability on every callback exit."""

    def __init__(self, capability: RunActionPreparationCapability) -> None:
        self._capability = capability

    def __enter__(self) -> RunActionPreparationCapability:
        return self._capability

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self._capability._finish_invocation()
        return False


class RunActionActivationCapability:
    """Single-use delivery and inert-revalidation authority before event 5."""

    def __init__(
        self,
        *,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        request_payload: bytes,
        workspace_descriptor: int | None,
        _authority: object,
    ) -> None:
        if type(prepared_execution) is not RunActionPreparedExecution:
            raise RunActionRecoveryError(
                "run action activation requires one exact prepared execution"
            )
        reservation = prepared_execution.preparation_claim.reservation
        if (
            type(spawn_commit) is not RunActionSpawnCommit
            or spawn_commit.reservation_id != reservation.reservation_id
            or spawn_commit.prepared_execution_id
            != prepared_execution.prepared_execution_id
            or spawn_commit.provider_execution_id
            != prepared_execution.inert_container_evidence.container_id
            or spawn_commit.boundary_identity != reservation.intent.boundary_identity
            or spawn_commit.security_observation_id
            != reservation.frontier.security_observation_id
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
            or _authority is not _RUN_ACTION_ACTIVATION_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action activation capability lacks exact authority"
            )
        self._prepared_execution = prepared_execution
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
        with _ACTIVATION_CAPABILITY_LOCK:
            _ISSUED_ACTIVATION_CAPABILITIES[id(self)] = self

    @property
    def prepared_execution(self) -> RunActionPreparedExecution:
        self._require_active_invocation()
        return self._prepared_execution

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
        execution_adapter: "RunActionExecutionAdapter",
    ) -> RunActionActivationRevalidationReceipt:
        with self._begin_invocation():
            return execution_adapter.stage_activation(self)

    def _begin_invocation(self) -> "_RunActionActivationInvocation":
        with _ACTIVATION_CAPABILITY_LOCK:
            issued = _ISSUED_ACTIVATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "ready"
            ):
                raise RunActionRecoveryError(
                    "run action activation capability is spent, cloned, or foreign"
                )
            self._state = "invoking"
            self._invoking_thread_id = get_ident()
        return _RunActionActivationInvocation(self)

    def _require_active_invocation(self) -> None:
        with _ACTIVATION_CAPABILITY_LOCK:
            issued = _ISSUED_ACTIVATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
            ):
                raise RunActionRecoveryError(
                    "run action activation capability is not in its one invocation"
                )

    def _finish_invocation(self) -> None:
        with _ACTIVATION_CAPABILITY_LOCK:
            issued = _ISSUED_ACTIVATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
            ):
                raise RunActionRecoveryError(
                    "run action activation capability invocation changed"
                )
            self._state = "spent"
            self._invoking_thread_id = None
            _ISSUED_ACTIVATION_CAPABILITIES.pop(id(self))
        if self._workspace_descriptor is not None:
            os.close(self._workspace_descriptor)
            self._workspace_descriptor = None


class _RunActionActivationInvocation:
    """Burn one activation capability on every callback exit."""

    def __init__(self, capability: RunActionActivationCapability) -> None:
        self._capability = capability

    def __enter__(self) -> RunActionActivationCapability:
        return self._capability

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self._capability._finish_invocation()
        return False


class RunActionCommittedContinuationCapability:
    """Single-use authority for one exact observed event-5 continuation."""

    def __init__(
        self,
        *,
        query: "RunActionCommittedSpawnQuery",
        observation: RunActionCommittedSpawnObservation,
        required_security_observation: SecurityDenylistObservation,
        security_authority: object,
        credential_validity_authority: object | None,
        release_clock: _SystemRunActionReleaseClock,
        _authority: object,
    ) -> None:
        if type(query) is not RunActionCommittedSpawnQuery:
            raise RunActionRecoveryError(
                "committed continuation capability lacks its exact query"
            )
        activation_event = query.activation_event
        if (
            type(activation_event) is not RunActionExecutionEvent
            or activation_event.event_number != 5
            or activation_event.event_kind
            is not RunActionExecutionEventKind.ACTIVATION_COMMITTED
            or type(activation_event.activation_revalidation_receipt)
            is not RunActionActivationRevalidationReceipt
        ):
            raise RunActionRecoveryError(
                "committed activation requires one exact durable event"
            )
        activation = activation_event.activation_revalidation_receipt
        reservation = activation.prepared_execution.preparation_claim.reservation
        if (
            activation_event.reservation != reservation
            or type(observation) is not RunActionCommittedSpawnObservation
            or observation.state
            not in {
                RunActionCommittedSpawnState.INERT_CONTINUABLE,
                RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
                RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            }
            or type(required_security_observation) is not SecurityDenylistObservation
            or required_security_observation.observation_id
            != reservation.frontier.security_observation_id
            or required_security_observation.matched_revocations
            or not hasattr(security_authority, "observe_exact_descendant_of")
            or (
                credential_validity_authority is not None
                and not hasattr(credential_validity_authority, "observe_exact")
            )
            or type(release_clock) is not _SystemRunActionReleaseClock
            or _authority is not _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "committed continuation capability lacks exact authority"
            )
        self._query = query
        self._observation = observation
        self._owner_process_id = os.getpid()
        self._invoking_thread_id = None
        self._state = "ready"
        self._release_publication_state = "ready"
        self._terminal_inspection_state = "ready"
        self._terminal_observation: RunActionTerminalObservation | None = None
        self._result_capture_state = "ready"
        self._captured_result: RunActionProviderResult | None = None
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES[id(self)] = self
            _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES[self] = (
                _RunActionIssuedReleaseAuthorities(
                    required_security_observation=required_security_observation,
                    security_authority=security_authority,
                    credential_validity_authority=credential_validity_authority,
                    clock=release_clock,
                )
            )

    @property
    def activation_event(self) -> RunActionExecutionEvent:
        self._require_active_invocation()
        return self._query.activation_event

    @property
    def query(self) -> "RunActionCommittedSpawnQuery":
        self._require_active_invocation()
        return self._query

    @property
    def activation_revalidation_receipt(
        self,
    ) -> RunActionActivationRevalidationReceipt:
        self._require_active_invocation()
        return self._query.activation_revalidation_receipt

    @property
    def prepared_execution(self) -> RunActionPreparedExecution:
        return self.activation_revalidation_receipt.prepared_execution

    @property
    def spawn_commit(self) -> RunActionSpawnCommit:
        return self.activation_revalidation_receipt.spawn_commit

    @property
    def observation(self) -> RunActionCommittedSpawnObservation:
        self._require_active_invocation()
        return self._observation

    @property
    def workload_release_adoption(
        self,
    ) -> RunActionWorkloadReleaseAdoption | None:
        self._require_active_invocation()
        return self._query.workload_release_adoption

    def _begin_release_publication(
        self,
        resolved_workload_observation: RunActionResolvedWorkloadObservation,
        *,
        _authority: object,
    ) -> "_RunActionReleasePublicationAuthorization":
        """Open the trusted publisher's sole event-5 release authorization."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            release_authority = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(
                self
            )
            if (
                issued is not self
                or release_authority is None
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._release_publication_state != "ready"
                or self._observation.state
                is not RunActionCommittedSpawnState.RUNNING_CONTINUABLE
                or type(resolved_workload_observation)
                is not RunActionResolvedWorkloadObservation
                or resolved_workload_observation.activation_revalidation_receipt
                != self._query.activation_revalidation_receipt
                or resolved_workload_observation.running_container_observation.complete_inspection_digest
                != self._observation.observation_token
                or resolved_workload_observation.release_present is not False
                or self._query.workload_release_adoption is not None
                or _authority is not _RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "workload release publication lacks exact live authority"
                )
            self._release_publication_state = "preparing"
        return _RunActionReleasePublicationAuthorization(
            capability=self,
            resolved_workload_observation=resolved_workload_observation,
            required_security_observation=(
                release_authority.required_security_observation
            ),
            _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
        )

    def _take_terminal_inspection_authority(
        self,
        *,
        _authority: object,
    ) -> tuple["RunActionCommittedSpawnQuery", str]:
        """Consume the trusted terminal leaf's sole sealed reinspection."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            observation_token = self._observation.observation_token
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._terminal_inspection_state != "ready"
                or self._observation.state
                is not RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
                or type(observation_token) is not str
                or self._query.workload_release_adoption is None
                or _authority is not _RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "terminal reinspection lacks exact live continuation authority"
                )
            self._terminal_inspection_state = "inspecting"
        return self._query, observation_token

    def _complete_terminal_inspection(
        self,
        terminal_observation: RunActionTerminalObservation,
        *,
        _authority: object,
    ) -> None:
        """Register the trusted terminal leaf's exact completed observation."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            query = self._query
            adoption = query.workload_release_adoption
            activation = query.activation_revalidation_receipt
            prepared = query.prepared_execution
            spawn = query.spawn_commit
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._terminal_inspection_state != "inspecting"
                or self._terminal_observation is not None
                or self._observation.state
                is not RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
                or adoption is None
                or type(terminal_observation) is not RunActionTerminalObservation
                or terminal_observation.complete_inspection_digest
                != self._observation.observation_token
                or terminal_observation.activation_revalidation_receipt_id
                != activation.activation_revalidation_receipt_id
                or terminal_observation.workload_release_adoption_id
                != adoption.workload_release_adoption_id
                or terminal_observation.prepared_execution_id
                != prepared.prepared_execution_id
                or terminal_observation.spawn_commit_id != spawn.spawn_commit_id
                or terminal_observation.provider_execution_id
                != spawn.provider_execution_id
                or terminal_observation.runtime_volume_authority_id
                != prepared.runtime_volume_authority.runtime_volume_authority_id
                or terminal_observation.generation_nonce
                != prepared.runtime_volume_authority.generation_nonce
                or terminal_observation.observed_inspect_projection
                != prepared.inert_container_evidence.issued_create_projection
                or _authority is not _RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "terminal reinspection completion lacks exact live authority"
                )
            self._terminal_observation = terminal_observation
            self._terminal_inspection_state = "complete"

    def _take_result_capture_authority(
        self,
        *,
        _authority: object,
    ) -> tuple["RunActionCommittedSpawnQuery", RunActionTerminalObservation]:
        """Consume the trusted result leaf's sole descriptor capture."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            terminal = self._terminal_observation
            adoption = self._query.workload_release_adoption
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._observation.state
                is not RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
                or self._terminal_inspection_state != "complete"
                or type(terminal) is not RunActionTerminalObservation
                or terminal.exit_code != 0
                or terminal.oom_killed is not False
                or adoption is None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or _authority is not _RUN_ACTION_RESULT_CAPTURE_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "result capture lacks exact live terminal authority"
                )
            clock = authorities.clock
            deadline = (
                adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
            )
        current_boottime = _read_positive_release_clock(
            clock.boottime_nanoseconds(),
            "result capture BOOTTIME",
        )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._terminal_observation != terminal
                or self._result_capture_state != "ready"
                or current_boottime > deadline
            ):
                raise RunActionRecoveryError(
                    "result capture started outside its exact terminal deadline"
                )
            self._result_capture_state = "capturing"
        return self._query, terminal

    def _complete_result_capture(
        self,
        result: RunActionProviderResult,
        *,
        _authority: object,
    ) -> None:
        """Register the trusted descriptor leaf's exact captured result."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            query = self._query
            adoption = query.workload_release_adoption
            terminal = self._terminal_observation
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._result_capture_state != "capturing"
                or self._captured_result is not None
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or adoption is None
                or type(terminal) is not RunActionTerminalObservation
                or type(result) is not RunActionProviderResult
                or result.terminal_observation != terminal
                or not run_action_terminal_result_evidence_matches(
                    terminal,
                    result.result_capture_receipt,
                    query.activation_revalidation_receipt,
                    adoption,
                )
                or _authority is not _RUN_ACTION_RESULT_CAPTURE_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "result capture completion lacks exact live authority"
                )
            self._captured_result = result
            self._result_capture_state = "complete"

    def _invoke_once(
        self,
        execution_adapter: "RunActionExecutionAdapter",
    ) -> RunActionContinuationOutcome:
        with self._begin_invocation():
            outcome = execution_adapter.continue_committed_once(self)
            self._require_terminal_outcome_authority(outcome)
            return outcome

    def _require_terminal_outcome_authority(
        self,
        outcome: RunActionContinuationOutcome,
    ) -> None:
        """Bind terminal continuation outcomes to the trusted terminal leaf."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or type(outcome) is not RunActionContinuationOutcome
            ):
                raise RunActionRecoveryError(
                    "execution adapter returned an invalid committed continuation"
                )
            if (
                self._observation.state
                is RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
            ):
                if (
                    self._terminal_inspection_state != "complete"
                    or type(self._terminal_observation)
                    is not RunActionTerminalObservation
                    or (
                        outcome.state is RunActionContinuationState.RESULT_CAPTURED
                        and (
                            type(outcome.result) is not RunActionProviderResult
                            or outcome.result.terminal_observation
                            != self._terminal_observation
                        )
                    )
                ):
                    raise RunActionRecoveryError(
                        "terminal continuation lacks its trusted reinspection"
                    )
                if (
                    outcome.state is RunActionContinuationState.PENDING
                    and (
                        self._result_capture_state != "ready"
                        or self._captured_result is not None
                    )
                ) or (
                    outcome.state is RunActionContinuationState.RESULT_CAPTURED
                    and (
                        self._result_capture_state != "complete"
                        or type(self._captured_result) is not RunActionProviderResult
                        or outcome.result != self._captured_result
                    )
                ):
                    raise RunActionRecoveryError(
                        "terminal continuation lacks its trusted result capture"
                    )
            elif (
                outcome.state is RunActionContinuationState.RESULT_CAPTURED
                or self._terminal_inspection_state != "ready"
                or self._terminal_observation is not None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
            ):
                raise RunActionRecoveryError(
                    "nonterminal continuation consumed terminal or result authority"
                )

    def _begin_invocation(self) -> "_RunActionCommittedContinuationInvocation":
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "ready"
            ):
                raise RunActionRecoveryError(
                    "committed continuation capability is spent, cloned, or foreign"
                )
            self._state = "invoking"
            self._invoking_thread_id = get_ident()
        return _RunActionCommittedContinuationInvocation(self)

    def _require_active_invocation(self) -> None:
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._release_publication_state == "authorizing"
            ):
                raise RunActionRecoveryError(
                    "committed continuation capability is not in its one invocation"
                )

    def _finish_invocation(self) -> None:
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            if (
                issued is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
            ):
                raise RunActionRecoveryError(
                    "committed continuation capability invocation changed"
                )
            self._state = "spent"
            self._invoking_thread_id = None
            self._release_publication_state = "spent"
            self._terminal_inspection_state = "spent"
            self._result_capture_state = "spent"
            _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.pop(id(self))
            _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.pop(self, None)


class _RunActionCommittedContinuationInvocation:
    """Burn one committed continuation capability on every callback exit."""

    def __init__(self, capability: RunActionCommittedContinuationCapability) -> None:
        self._capability = capability

    def __enter__(self) -> RunActionCommittedContinuationCapability:
        return self._capability

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self._capability._finish_invocation()
        return False


class _RunActionReleasePublicationAuthorization:
    """Publisher-private security authority for one already-frozen receipt."""

    def __init__(
        self,
        *,
        capability: RunActionCommittedContinuationCapability,
        resolved_workload_observation: RunActionResolvedWorkloadObservation,
        required_security_observation: SecurityDenylistObservation,
        _authority: object,
    ) -> None:
        if (
            type(capability) is not RunActionCommittedContinuationCapability
            or type(resolved_workload_observation)
            is not RunActionResolvedWorkloadObservation
            or type(required_security_observation) is not SecurityDenylistObservation
            or _authority is not _RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "release publication authorization lacks issuance authority"
            )
        self._capability = capability
        self._resolved_workload_observation = resolved_workload_observation
        self._required_security_observation = required_security_observation
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._receipt = None
        self._closed = False

    def _mint_receipt(
        self,
        *,
        _authority: object,
    ) -> RunActionWorkloadReleaseReceipt:
        self._require_preparing(_authority)
        if self._receipt is not None:
            raise RunActionRecoveryError(
                "release publication authorization already minted its receipt"
            )
        authorities = self._release_authorities()
        anchor_boottime = _read_positive_release_clock(
            authorities.clock.boottime_nanoseconds(),
            "release BOOTTIME anchor",
        )
        anchor_realtime = _read_positive_release_clock(
            authorities.clock.realtime_nanoseconds(),
            "release REALTIME anchor",
        )
        credential_validity = _observe_release_credential_validity(
            self._resolved_workload_observation,
            authorities.credential_validity_authority,
            authorities.clock,
            anchor_realtime,
        )
        release_authorization = RunActionReleaseAuthorizationObservation.mint(
            security_observation=self._required_security_observation,
            authorized_at_boottime_nanoseconds=anchor_boottime,
            authorized_at_realtime_nanoseconds=anchor_realtime,
            credential_validity_observation=credential_validity,
        )
        self._receipt = mint_run_action_workload_release_receipt(
            activation_event=self._capability.activation_event,
            resolved_workload_observation=self._resolved_workload_observation,
            release_authorization_observation=release_authorization,
        )
        return self._receipt

    def _authorize_frozen_release_once(
        self,
        *,
        candidate: _RunActionFrozenReleaseCandidate,
        _authority: object,
    ) -> RunActionWorkloadReleaseReceipt | None:
        self._require_preparing(_authority)
        if (
            type(candidate) is not _RunActionFrozenReleaseCandidate
            or type(self._receipt) is not RunActionWorkloadReleaseReceipt
        ):
            raise RunActionRecoveryError(
                "release authorization lacks one exact frozen receipt candidate"
            )
        receipt = candidate._begin_authorization(
            self._receipt,
            _authority=_RELEASE_CANDIDATE_AUTHORIZATION_AUTHORITY,
        )
        if (
            receipt != self._receipt
            or receipt.release_authorization_observation.security_observation
            != self._required_security_observation
        ):
            raise RunActionRecoveryError(
                "frozen release candidate differs from its issued authorization"
            )
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            capability._release_publication_state = "authorizing"
        authorities = self._release_authorities()
        with _RunActionReleaseLinkInvocation(self):
            _revalidate_release_credential_validity(
                self._resolved_workload_observation,
                authorities.credential_validity_authority,
                receipt,
                authorities.clock,
            )
            required = authorities.required_security_observation
            current = authorities.security_authority.observe_exact_descendant_of(
                scope_id=required.scope_id,
                scope_contract_id=required.scope_contract_id,
                checked_subject_ids=required.checked_subject_ids,
                required_ancestor=required,
            )
            if type(current) is not SecurityDenylistObservation:
                raise RunActionRecoveryError(
                    "workload release security authority returned another type"
                )
            if current != required:
                return None
            current_boottime_nanoseconds = authorities.clock.boottime_nanoseconds()
            if (
                type(current_boottime_nanoseconds) is not int
                or current_boottime_nanoseconds <= 0
                or current_boottime_nanoseconds
                > receipt.release_commit_deadline_boottime_nanoseconds
            ):
                return None
            return candidate._link_authorized_once(
                _authority=_RELEASE_CANDIDATE_AUTHORIZATION_AUTHORITY,
            )

    def _release_authorities(self) -> _RunActionIssuedReleaseAuthorities:
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(
                self._capability
            )
        if type(authorities) is not _RunActionIssuedReleaseAuthorities:
            raise RunActionRecoveryError(
                "release publication lost its coordinator-issued authorities"
            )
        return authorities

    def _require_preparing(self, _authority: object) -> None:
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(capability))
            release_authority = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(
                capability
            )
            if (
                self._closed
                or self._owner_process_id != os.getpid()
                or self._owner_thread_id != get_ident()
                or issued is not capability
                or release_authority is None
                or capability._owner_process_id != os.getpid()
                or capability._state != "invoking"
                or capability._invoking_thread_id != get_ident()
                or capability._release_publication_state != "preparing"
                or release_authority.required_security_observation
                != self._required_security_observation
                or _authority is not _RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "release publication authorization is spent, inactive, or foreign"
                )

    def _finish_link_invocation(self) -> None:
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                self._closed
                or self._owner_process_id != os.getpid()
                or self._owner_thread_id != get_ident()
                or capability._state != "invoking"
                or capability._invoking_thread_id != get_ident()
                or capability._release_publication_state != "authorizing"
            ):
                raise RunActionRecoveryError(
                    "release link authorization invocation changed"
                )
            self._closed = True
            capability._release_publication_state = "spent"

    def _close(self) -> None:
        if self._closed:
            return
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                self._owner_process_id != os.getpid()
                or self._owner_thread_id != get_ident()
                or capability._state != "invoking"
                or capability._invoking_thread_id != get_ident()
                or capability._release_publication_state != "preparing"
            ):
                raise RunActionRecoveryError(
                    "release publication authorization close changed"
                )
            self._closed = True
            capability._release_publication_state = "spent"

    def __enter__(self) -> "_RunActionReleasePublicationAuthorization":
        self._require_preparing(_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY)
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self._close()
        return False


class _RunActionReleaseLinkInvocation:
    """Burn the private link authorization on every final-check exit."""

    def __init__(
        self, authorization: _RunActionReleasePublicationAuthorization
    ) -> None:
        self._authorization = authorization

    def __enter__(self) -> _RunActionReleasePublicationAuthorization:
        return self._authorization

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self._authorization._finish_link_invocation()
        return False


def _observe_release_credential_validity(
    resolved: RunActionResolvedWorkloadObservation,
    credential_authority: object | None,
    clock: _SystemRunActionReleaseClock,
    anchor_realtime_nanoseconds: int,
) -> RunActionCredentialValidityObservation | None:
    activation = resolved.activation_revalidation_receipt
    policy = activation.prepared_execution.preparation_claim.execution_policy
    credential_file = activation.credential_file_observation
    if policy.credential_policy.mode is RunActionCredentialMode.NONE:
        if credential_file is not None:
            raise RunActionRecoveryError(
                "credential-free release carries a credential file"
            )
        return None
    if credential_file is None or not hasattr(credential_authority, "observe_exact"):
        raise RunActionRecoveryError(
            "credentialed release lacks its coordinator-held validity authority"
        )
    observed_before = _read_positive_release_clock(
        clock.realtime_nanoseconds(),
        "credential observation start",
    )
    validity = credential_authority.observe_exact(
        activated_credential_file_observation_id=(
            credential_file.activated_file_observation_id
        ),
        credential_lease_authority_id=credential_file.content_authority_id,
    )
    observed_after = _read_positive_release_clock(
        clock.realtime_nanoseconds(),
        "credential observation finish",
    )
    required_valid_until = (
        anchor_realtime_nanoseconds
        + (
            policy.supervisor_limits.execution_timeout_seconds
            + policy.supervisor_limits.termination_grace_seconds
        )
        * _NANOSECONDS_PER_SECOND
    )
    if (
        type(validity) is not RunActionCredentialValidityObservation
        or validity.activated_credential_file_observation_id
        != credential_file.activated_file_observation_id
        or validity.credential_lease_authority_id
        != credential_file.content_authority_id
        or not observed_before
        <= validity.observed_at_realtime_nanoseconds
        <= observed_after
        or validity.valid_until_realtime_nanoseconds < required_valid_until
        or (
            validity.valid_until_realtime_nanoseconds
            - validity.observed_at_realtime_nanoseconds
        )
        > policy.credential_policy.maximum_lease_seconds * _NANOSECONDS_PER_SECOND
    ):
        raise RunActionRecoveryError(
            "credential validity cannot authorize the release deadline"
        )
    return validity


def _revalidate_release_credential_validity(
    resolved: RunActionResolvedWorkloadObservation,
    credential_authority: object | None,
    receipt: RunActionWorkloadReleaseReceipt,
    clock: _SystemRunActionReleaseClock,
) -> None:
    activation = resolved.activation_revalidation_receipt
    policy = activation.prepared_execution.preparation_claim.execution_policy
    credential_file = activation.credential_file_observation
    required_validity = (
        receipt.release_authorization_observation.credential_validity_observation
    )
    if policy.credential_policy.mode is RunActionCredentialMode.NONE:
        if credential_file is not None or required_validity is not None:
            raise RunActionRecoveryError(
                "credential-free release gained credential evidence"
            )
        return
    if (
        type(required_validity) is not RunActionCredentialValidityObservation
        or credential_file is None
        or not hasattr(credential_authority, "observe_exact")
    ):
        raise RunActionRecoveryError(
            "credential revalidation lacks exact coordinator authority"
        )
    observed_before = _read_positive_release_clock(
        clock.realtime_nanoseconds(),
        "credential revalidation start",
    )
    current = credential_authority.observe_exact(
        activated_credential_file_observation_id=(
            credential_file.activated_file_observation_id
        ),
        credential_lease_authority_id=credential_file.content_authority_id,
    )
    observed_after = _read_positive_release_clock(
        clock.realtime_nanoseconds(),
        "credential revalidation finish",
    )
    containment_deadline_realtime = (
        receipt.release_authorization_observation.authorized_at_realtime_nanoseconds
        + (
            policy.supervisor_limits.execution_timeout_seconds
            + policy.supervisor_limits.termination_grace_seconds
        )
        * _NANOSECONDS_PER_SECOND
    )
    if (
        type(current) is not RunActionCredentialValidityObservation
        or current.activated_credential_file_observation_id
        != required_validity.activated_credential_file_observation_id
        or current.credential_lease_authority_id
        != required_validity.credential_lease_authority_id
        or not observed_before
        <= current.observed_at_realtime_nanoseconds
        <= observed_after
        or current.observed_at_realtime_nanoseconds
        < required_validity.observed_at_realtime_nanoseconds
        or current.valid_until_realtime_nanoseconds < containment_deadline_realtime
        or (
            current.valid_until_realtime_nanoseconds
            - current.observed_at_realtime_nanoseconds
        )
        > policy.credential_policy.maximum_lease_seconds * _NANOSECONDS_PER_SECOND
    ):
        raise RunActionRecoveryError("credential authority changed before release link")


def _read_positive_release_clock(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RunActionRecoveryError(f"{name} is invalid")
    return value


@dataclass(frozen=True)
class RunActionUnactivatedSpawnQuery:
    """Read-only identity for a spawn that has no durable activation receipt."""

    prepared_execution: RunActionPreparedExecution
    spawn_commit: RunActionSpawnCommit

    def __post_init__(self) -> None:
        if type(self.prepared_execution) is not RunActionPreparedExecution:
            raise RunActionRecoveryError(
                "committed run action query lacks its prepared execution"
            )
        reservation = self.prepared_execution.preparation_claim.reservation
        if (
            type(self.spawn_commit) is not RunActionSpawnCommit
            or self.spawn_commit.reservation_id != reservation.reservation_id
            or self.spawn_commit.prepared_execution_id
            != self.prepared_execution.prepared_execution_id
            or self.spawn_commit.provider_execution_id
            != self.prepared_execution.inert_container_evidence.container_id
            or self.spawn_commit.boundary_identity
            != reservation.intent.boundary_identity
            or self.spawn_commit.security_observation_id
            != reservation.frontier.security_observation_id
        ):
            raise RunActionRecoveryError(
                "committed run action query lacks exact durable identity"
            )

    @property
    def reservation(self) -> RunActionReservation:
        return self.prepared_execution.preparation_claim.reservation


@dataclass(frozen=True)
class RunActionCommittedSpawnQuery:
    """Read-only identity for one durably selected activation occurrence."""

    preparation_allocation: RunActionPreparationAllocation
    activation_event: RunActionExecutionEvent
    workload_release_adoption: RunActionWorkloadReleaseAdoption | None

    def __post_init__(self) -> None:
        if (
            type(self.preparation_allocation) is not RunActionPreparationAllocation
            or type(self.activation_event) is not RunActionExecutionEvent
            or self.activation_event.event_number != 5
            or self.activation_event.event_kind
            is not RunActionExecutionEventKind.ACTIVATION_COMMITTED
            or type(self.activation_event.activation_revalidation_receipt)
            is not RunActionActivationRevalidationReceipt
            or (
                self.workload_release_adoption is not None
                and type(self.workload_release_adoption)
                is not RunActionWorkloadReleaseAdoption
            )
        ):
            raise RunActionRecoveryError(
                "committed run action query lacks its durable activation"
            )
        if self.workload_release_adoption is not None:
            require_run_action_workload_release_receipt_matches_event(
                self.workload_release_adoption.workload_release_receipt,
                self.activation_event,
            )
        unactivated = RunActionUnactivatedSpawnQuery(
            prepared_execution=self.prepared_execution,
            spawn_commit=self.spawn_commit,
        )
        if (
            self.activation_event.reservation
            != self.preparation_allocation.preparation_claim.reservation
            or self.prepared_execution.preparation_claim
            != self.preparation_allocation.preparation_claim
            or self.prepared_execution.runtime_volume_authority
            != self.preparation_allocation.runtime_volume_authority
            or self.activation_revalidation_receipt.prepared_execution
            != unactivated.prepared_execution
            or self.activation_revalidation_receipt.spawn_commit
            != unactivated.spawn_commit
            or (
                self.workload_release_adoption is not None
                and self.workload_release_adoption.workload_release_receipt.activation_event_id
                != self.activation_event.event_id
            )
        ):
            raise RunActionRecoveryError(
                "committed run action query lacks its durable activation"
            )

    @property
    def prepared_execution(self) -> RunActionPreparedExecution:
        return self.activation_revalidation_receipt.prepared_execution

    @property
    def spawn_commit(self) -> RunActionSpawnCommit:
        return self.activation_revalidation_receipt.spawn_commit

    @property
    def activation_revalidation_receipt(
        self,
    ) -> RunActionActivationRevalidationReceipt:
        return self.activation_event.activation_revalidation_receipt

    @property
    def reservation(self) -> RunActionReservation:
        return self.preparation_allocation.preparation_claim.reservation


class RunActionExecutionAdapter(Protocol):
    """Provider execution lifecycle with no result-interpretation authority."""

    execution_lifecycle_identity: RunActionExecutionLifecycleIdentity
    execution_policy: DockerRunActionExecutionPolicy

    def prepared_event_size_bound(
        self,
        *,
        preparation_allocation: RunActionPreparationAllocation,
        predecessor_event_id: str,
    ) -> int: ...

    def activation_event_size_bound(
        self,
        *,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        predecessor_event_id: str,
    ) -> int: ...

    def release_receipt_size_bound(
        self,
        *,
        reservation: RunActionReservation,
    ) -> int: ...

    def prepare(
        self,
        capability: RunActionPreparationCapability,
    ) -> RunActionPreparationObservation: ...

    def stage_activation(
        self,
        capability: RunActionActivationCapability,
    ) -> RunActionActivationRevalidationReceipt: ...

    def inspect_unactivated(
        self,
        query: RunActionUnactivatedSpawnQuery,
    ) -> RunActionUnactivatedSpawnObservation: ...

    def inspect_committed(
        self,
        query: RunActionCommittedSpawnQuery,
    ) -> RunActionCommittedSpawnObservation: ...

    def continue_committed_once(
        self,
        capability: RunActionCommittedContinuationCapability,
    ) -> RunActionContinuationOutcome: ...


class RunActionResultInterpreter(Protocol):
    """Dependency-pure interpreter with no provider or workspace authority."""

    result_interpreter_identity: RunActionResultInterpreterIdentity

    def interpret(
        self,
        *,
        request_payload: bytes,
        result_payload: bytes,
    ) -> RunActionInterpretedResult: ...


@dataclass(frozen=True)
class RunActionRecoveryImplementation:
    """Exact lifecycle/interpreter objects composing one durable boundary."""

    boundary_identity: RunActionBoundaryIdentity
    execution_adapter: RunActionExecutionAdapter
    result_interpreter: RunActionResultInterpreter

    def __post_init__(self) -> None:
        if (
            type(self.boundary_identity) is not RunActionBoundaryIdentity
            or self.execution_adapter is self.result_interpreter
            or not hasattr(self.execution_adapter, "execution_lifecycle_identity")
            or self.execution_adapter.execution_lifecycle_identity
            != self.boundary_identity.execution_lifecycle_identity
            or not hasattr(self.execution_adapter, "execution_policy")
            or type(self.execution_adapter.execution_policy)
            is not DockerRunActionExecutionPolicy
            or self.execution_adapter.execution_policy.kind
            is not self.boundary_identity.kind
            or self.execution_adapter.execution_policy.docker_execution_policy_id
            != self.boundary_identity.execution_lifecycle_identity.execution_policy_id
            or not hasattr(self.result_interpreter, "result_interpreter_identity")
            or self.result_interpreter.result_interpreter_identity
            != self.boundary_identity.result_interpreter_identity
        ):
            raise RunActionRecoveryError(
                "run action recovery implementation differs from its boundary"
            )


class RunActionRecoveryImplementationRegistry:
    """Process-bound exact lifecycle/interpreter composition catalog."""

    def __init__(
        self,
        implementations: tuple[RunActionRecoveryImplementation, ...],
        *,
        _authority: object,
    ) -> None:
        if (
            type(implementations) is not tuple
            or not implementations
            or _authority is not _RUN_ACTION_RECOVERY_IMPLEMENTATION_REGISTRY_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action recovery implementation registry lacks issuance authority"
            )
        indexed = {}
        bindings = []
        for implementation in implementations:
            if type(implementation) is not RunActionRecoveryImplementation:
                raise RunActionRecoveryError(
                    "run action recovery implementation has an invalid type"
                )
            boundary_identity = implementation.boundary_identity
            execution_adapter = implementation.execution_adapter
            result_interpreter = implementation.result_interpreter
            execution_methods = tuple(
                getattr(type(execution_adapter), name, None)
                for name in _EXECUTION_ADAPTER_METHOD_NAMES
            )
            interpreter_methods = tuple(
                getattr(type(result_interpreter), name, None)
                for name in _RESULT_INTERPRETER_METHOD_NAMES
            )
            if (
                boundary_identity.boundary_identity_id in indexed
                or any(method is None for method in execution_methods)
                or any(
                    getattr(getattr(execution_adapter, name), "__self__", None)
                    is not execution_adapter
                    or getattr(getattr(execution_adapter, name), "__func__", None)
                    is not method
                    for name, method in zip(
                        _EXECUTION_ADAPTER_METHOD_NAMES,
                        execution_methods,
                    )
                )
                or any(method is None for method in interpreter_methods)
                or any(
                    getattr(getattr(result_interpreter, name), "__self__", None)
                    is not result_interpreter
                    or getattr(getattr(result_interpreter, name), "__func__", None)
                    is not method
                    for name, method in zip(
                        _RESULT_INTERPRETER_METHOD_NAMES,
                        interpreter_methods,
                    )
                )
            ):
                raise RunActionRecoveryError(
                    "run action recovery implementation registry is ambiguous or invalid"
                )
            indexed[boundary_identity.boundary_identity_id] = implementation
            bindings.append(
                (
                    implementation,
                    boundary_identity,
                    execution_adapter,
                    type(execution_adapter),
                    execution_methods,
                    execution_adapter.execution_policy,
                    result_interpreter,
                    type(result_interpreter),
                    interpreter_methods,
                )
            )
        self._implementations = implementations
        self._owner_process_id = os.getpid()
        with _RECOVERY_IMPLEMENTATION_REGISTRY_LOCK:
            _ISSUED_RECOVERY_IMPLEMENTATION_REGISTRIES[id(self)] = self
            _ISSUED_RECOVERY_IMPLEMENTATION_BINDINGS[self] = tuple(bindings)

    def resolve_execution(
        self,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> RunActionExecutionAdapter:
        self._require_owner_process()
        matching = self._matching_binding(boundary_identity)
        if (
            matching[0].boundary_identity != boundary_identity
            or matching[0].execution_adapter is not matching[2]
            or matching[2].execution_lifecycle_identity
            != boundary_identity.execution_lifecycle_identity
            or matching[2].execution_policy is not matching[5]
            or matching[5].docker_execution_policy_id
            != boundary_identity.execution_lifecycle_identity.execution_policy_id
            or type(matching[2]) is not matching[3]
            or any(
                getattr(getattr(matching[2], name), "__self__", None) is not matching[2]
                or getattr(getattr(matching[2], name), "__func__", None) is not method
                for name, method in zip(
                    _EXECUTION_ADAPTER_METHOD_NAMES,
                    matching[4],
                )
            )
        ):
            raise RunActionRecoveryError(
                "run action execution adapter is absent or substituted"
            )
        return matching[2]

    def resolve_interpreter(
        self,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> RunActionResultInterpreter:
        self._require_owner_process()
        matching = self._matching_binding(boundary_identity)
        if (
            matching[0].boundary_identity != boundary_identity
            or matching[0].result_interpreter is not matching[6]
            or matching[6].result_interpreter_identity
            != boundary_identity.result_interpreter_identity
            or type(matching[6]) is not matching[7]
            or any(
                getattr(getattr(matching[6], name), "__self__", None) is not matching[6]
                or getattr(getattr(matching[6], name), "__func__", None) is not method
                for name, method in zip(
                    _RESULT_INTERPRETER_METHOD_NAMES,
                    matching[8],
                )
            )
        ):
            raise RunActionRecoveryError(
                "run action result interpreter is absent or substituted"
            )
        return matching[6]

    def _matching_binding(
        self,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> tuple:
        if type(boundary_identity) is not RunActionBoundaryIdentity:
            raise RunActionRecoveryError(
                "run action implementation lookup requires an exact boundary identity"
            )
        with _RECOVERY_IMPLEMENTATION_REGISTRY_LOCK:
            bindings = _ISSUED_RECOVERY_IMPLEMENTATION_BINDINGS.get(self)
        matching = tuple(
            binding
            for binding in bindings
            if binding[1].boundary_identity_id == boundary_identity.boundary_identity_id
        )
        if len(matching) != 1 or matching[0][1] != boundary_identity:
            raise RunActionRecoveryError(
                "run action recovery implementation is absent or substituted"
            )
        return matching[0]

    def _require_owner_process(self) -> None:
        with _RECOVERY_IMPLEMENTATION_REGISTRY_LOCK:
            issued = _ISSUED_RECOVERY_IMPLEMENTATION_REGISTRIES.get(id(self))
            bindings = _ISSUED_RECOVERY_IMPLEMENTATION_BINDINGS.get(self)
        if (
            issued is not self
            or bindings is None
            or type(self._implementations) is not tuple
            or len(bindings) != len(self._implementations)
            or any(
                implementation is not binding[0]
                for implementation, binding in zip(self._implementations, bindings)
            )
            or self._owner_process_id != os.getpid()
        ):
            raise RunActionRecoveryError(
                "run action recovery implementation registry is cloned, foreign, or altered"
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
        accepted_decision = (
            self.events[-2].result_decision
            if (
                len(self.events) == 8
                and all(type(event) is RunActionExecutionEvent for event in self.events)
                and self.events[-1].event_kind
                is RunActionExecutionEventKind.RESULT_ACCEPTED
                and self.events[-2].event_kind
                is RunActionExecutionEventKind.RESULT_DECIDED
                and type(self.events[-2].result_decision) is RunActionResultDecision
            )
            else None
        )
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
                    or accepted_decision is None
                    or tree_or_blob_digest(self.accepted_result_payload)
                    != accepted_decision.accepted_result_blob.digest
                    or len(self.accepted_result_payload)
                    != accepted_decision.accepted_result_blob.size_bytes
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
        credential_validity_authority: object | None,
        implementation_registry: RunActionRecoveryImplementationRegistry,
        _authority: object,
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
            or type(implementation_registry)
            is not RunActionRecoveryImplementationRegistry
            or _authority is not _RUN_ACTION_RECOVERY_COORDINATOR_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action recovery authorities are incompatible"
            )
        active_workspace.require_control_authority()
        self._active_workspace = active_workspace
        self._publisher = publisher
        self._store = publisher._action_store
        self._workspace_promoter = RunActionWorkspacePromoter(
            active_workspace=active_workspace,
            settings=publisher._settings,
        )
        self._security_authority = security_authority
        self._credential_validity_authority = credential_validity_authority
        self._release_clock = _SystemRunActionReleaseClock()
        implementation_registry._require_owner_process()
        self._implementation_registry = implementation_registry
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
        self._implementation_registry._require_owner_process()
        with ExitStack() as descriptors:
            self._publisher._hold_current(frontier, descriptors)
            workspace_lock_descriptor = self._store.lock_workspace(
                RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
                descriptors,
            )
            inspection = self._store.inspect()
            plan = self._plan(frontier, inspection)
            pending_events = (
                ()
                if plan.pending_operation_id is None
                else inspection.events_for(plan.pending_operation_id)
            )
            pending_owns_durable_stage = (
                bool(pending_events)
                and pending_events[-1].event_kind
                is RunActionExecutionEventKind.RESULT_DECIDED
                and pending_events[-1].result_decision.workspace_promotion is not None
            )
            if not pending_owns_durable_stage:
                self._cleanup_latest_accepted_promotion(
                    workspace_lock_descriptor,
                    only_if_owned=True,
                )
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
                        workspace_lock_descriptor,
                    )
            self._cleanup_latest_accepted_promotion(
                workspace_lock_descriptor,
            )
            return self._report(frontier)

    def _recover_session(
        self,
        frontier: ReconciledRunFrontier,
        session,
        descriptors: ExitStack,
        workspace_lock_descriptor: int,
    ) -> None:
        reservation = session.reservation
        self._require_reservation_frontier(frontier, reservation)
        events = session.events
        tail_kind = events[-1].event_kind
        if tail_kind in {
            RunActionExecutionEventKind.INTENT_RESERVED,
            RunActionExecutionEventKind.PREPARATION_ALLOCATED,
            RunActionExecutionEventKind.EXECUTION_PREPARED,
        }:
            security_is_current = self._security_is_current(frontier)
            workspace_descriptor, observed_workspace = (
                self._inspect_pre_spawn_workspace(
                    reservation,
                    descriptors,
                )
                if tail_kind is not RunActionExecutionEventKind.INTENT_RESERVED
                else self._inspect_workspace(
                    reservation,
                    descriptors,
                )
            )
            expected_workspace = reservation.frontier.workspace_before
            workspace_is_current = (
                None
                if observed_workspace is None
                else RunActionWorkspaceBinding.from_identity(observed_workspace)
            ) == expected_workspace
            if not workspace_is_current:
                if tail_kind is RunActionExecutionEventKind.INTENT_RESERVED:
                    session.cancel()
                return
            if not security_is_current:
                if tail_kind is RunActionExecutionEventKind.INTENT_RESERVED:
                    session.cancel()
                else:
                    session.invalidate_frontier()
                return
            execution_adapter = self._resolve_execution_adapter(
                self._implementation_registry,
                reservation,
            )
            release_receipt_size_bound = self._release_receipt_size_bound(
                execution_adapter,
                reservation,
            )
            if tail_kind is RunActionExecutionEventKind.INTENT_RESERVED:
                preparation_allocation = session.allocate_preparation(
                    execution_adapter.execution_policy,
                )
                preparation_mode = RunActionPreparationMode.CREATE_ALLOCATED
                durable_prepared_execution = None
                allocated_workspace_descriptor, allocated_workspace = (
                    self._inspect_pre_spawn_workspace(
                        reservation,
                        descriptors,
                    )
                )
                if (
                    None
                    if allocated_workspace is None
                    else RunActionWorkspaceBinding.from_identity(allocated_workspace)
                ) != expected_workspace:
                    return
                if not self._security_is_current(frontier):
                    session.invalidate_frontier()
                    return
                workspace_descriptor = allocated_workspace_descriptor
            elif tail_kind is RunActionExecutionEventKind.PREPARATION_ALLOCATED:
                preparation_allocation = events[-1].preparation_allocation
                preparation_mode = RunActionPreparationMode.REOPEN_ALLOCATED
                durable_prepared_execution = None
            else:
                durable_prepared_execution = events[-1].prepared_execution
                preparation_allocation = events[1].preparation_allocation
                preparation_mode = RunActionPreparationMode.REVALIDATE_PREPARED
            claim = preparation_allocation.preparation_claim
            prepared_event_size_bound = None
            if durable_prepared_execution is None:
                prepared_event_size_bound = execution_adapter.prepared_event_size_bound(
                    preparation_allocation=preparation_allocation,
                    predecessor_event_id=session.events[-1].event_id,
                )
                repeated_size_bound = execution_adapter.prepared_event_size_bound(
                    preparation_allocation=preparation_allocation,
                    predecessor_event_id=session.events[-1].event_id,
                )
                if (
                    type(prepared_event_size_bound) is not int
                    or prepared_event_size_bound <= 0
                    or repeated_size_bound != prepared_event_size_bound
                    or prepared_event_size_bound
                    > self._publisher._settings.run_action_event_size_bytes
                ):
                    raise RunActionRecoveryError(
                        "run action preparation event envelope is invalid or too large"
                    )
            preparation_capability = RunActionPreparationCapability(
                preparation_allocation=preparation_allocation,
                mode=preparation_mode,
                durable_prepared_execution=durable_prepared_execution,
                workspace_descriptor=workspace_descriptor,
                workspace_source_path=self._workspace_source_path(
                    reservation,
                    workspace_descriptor,
                ),
                _authority=_RUN_ACTION_PREPARATION_AUTHORITY,
            )
            preparation = preparation_capability._invoke_once(execution_adapter)
            if type(
                preparation
            ) is not RunActionPreparationObservation or not _preparation_origin_matches_mode(
                preparation,
                preparation_mode,
            ):
                raise RunActionRecoveryError(
                    "execution adapter returned an invalid preparation observation"
                )
            if preparation.state is RunActionPreparationState.UNKNOWN:
                return
            prepared = preparation.prepared_execution
            if (
                prepared.preparation_claim != claim
                or prepared.runtime_volume_authority
                != preparation_allocation.runtime_volume_authority
                or (
                    durable_prepared_execution is not None
                    and prepared != durable_prepared_execution
                )
            ):
                raise RunActionRecoveryError(
                    "execution adapter returned another prepared execution"
                )
            if durable_prepared_execution is None:
                if (
                    session._prepared_event_size_bytes(prepared)
                    > prepared_event_size_bound
                ):
                    raise RunActionRecoveryError(
                        "prepared run action exceeded its pre-materialization event envelope"
                    )
                session.commit_prepared_execution(prepared)
            confirmed_descriptor, confirmed_workspace = (
                self._inspect_pre_spawn_workspace(
                    reservation,
                    descriptors,
                )
            )
            if (
                None
                if confirmed_workspace is None
                else RunActionWorkspaceBinding.from_identity(confirmed_workspace)
            ) != expected_workspace:
                return
            if not self._security_is_current(frontier):
                session.invalidate_frontier()
                return
            if release_receipt_size_bound != self._release_receipt_size_bound(
                execution_adapter,
                reservation,
            ):
                raise RunActionRecoveryError(
                    "run action release-receipt envelope changed before spawn"
                )
            spawn_commit = session.commit_spawn(
                security_observation_id=(reservation.frontier.security_observation_id),
                boundary_identity=reservation.intent.boundary_identity,
            )
            self._stage_and_activate(
                session,
                execution_adapter,
                prepared,
                spawn_commit,
                confirmed_descriptor,
                frontier,
                descriptors,
                workspace_lock_descriptor,
            )
            return
        if tail_kind is RunActionExecutionEventKind.SPAWN_COMMITTED:
            execution_adapter = self._resolve_execution_adapter(
                self._implementation_registry,
                reservation,
            )
            prepared_execution = events[2].prepared_execution
            spawn_commit = events[-1].spawn_commit
            query = RunActionUnactivatedSpawnQuery(
                prepared_execution=prepared_execution,
                spawn_commit=spawn_commit,
            )
            observation = execution_adapter.inspect_unactivated(query)
            if type(observation) is not RunActionUnactivatedSpawnObservation:
                raise RunActionRecoveryError(
                    "execution adapter returned an invalid unactivated-spawn observation"
                )
            if observation.state is RunActionUnactivatedSpawnState.INERT_ACTIVATABLE:
                workspace_descriptor, observed_workspace = self._inspect_workspace(
                    reservation,
                    descriptors,
                )
                expected_workspace = reservation.frontier.workspace_before
                if (
                    None
                    if observed_workspace is None
                    else RunActionWorkspaceBinding.from_identity(observed_workspace)
                ) != expected_workspace:
                    return
                if not self._security_is_current(frontier):
                    return
                self._stage_and_activate(
                    session,
                    execution_adapter,
                    prepared_execution,
                    spawn_commit,
                    workspace_descriptor,
                    frontier,
                    descriptors,
                    workspace_lock_descriptor,
                )
            return
        if tail_kind is RunActionExecutionEventKind.ACTIVATION_COMMITTED:
            execution_adapter = self._resolve_execution_adapter(
                self._implementation_registry,
                reservation,
            )
            self._recover_committed(
                session,
                execution_adapter,
                frontier,
                descriptors,
                workspace_lock_descriptor,
            )
            return
        if tail_kind is RunActionExecutionEventKind.RESULT_RECEIVED:
            self._interpret_received(
                session,
                descriptors,
                workspace_lock_descriptor,
            )
            return
        if tail_kind is RunActionExecutionEventKind.RESULT_DECIDED:
            self._accept_decided(
                session,
                descriptors,
                workspace_lock_descriptor,
            )
            return
        raise RunActionRecoveryError(
            "run action recovery received a terminal operation"
        )

    def _stage_and_activate(
        self,
        session,
        execution_adapter: RunActionExecutionAdapter,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        workspace_descriptor: int | None,
        frontier: ReconciledRunFrontier,
        descriptors: ExitStack,
        workspace_lock_descriptor: int,
    ) -> RunActionAcceptance | None:
        predecessor_event_id = session.events[-1].event_id
        activation_event_size_bound = execution_adapter.activation_event_size_bound(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            predecessor_event_id=predecessor_event_id,
        )
        repeated_size_bound = execution_adapter.activation_event_size_bound(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            predecessor_event_id=predecessor_event_id,
        )
        if (
            type(activation_event_size_bound) is not int
            or activation_event_size_bound <= 0
            or repeated_size_bound != activation_event_size_bound
            or activation_event_size_bound
            > self._publisher._settings.run_action_event_size_bytes
        ):
            raise RunActionRecoveryError(
                "run action activation event envelope is invalid or too large"
            )
        release_receipt_size_bound = self._release_receipt_size_bound(
            execution_adapter,
            session.reservation,
        )
        if (
            activation_event_size_bound
            + self._publisher._settings.run_action_process_snapshot_size_bytes
            >= release_receipt_size_bound
        ):
            raise RunActionRecoveryError(
                "run action activation and resolved evidence cannot fit "
                "the release-receipt envelope"
            )
        capability = RunActionActivationCapability(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            request_payload=session.read_request(),
            workspace_descriptor=workspace_descriptor,
            _authority=_RUN_ACTION_ACTIVATION_AUTHORITY,
        )
        activation = capability._invoke_once(execution_adapter)
        if (
            type(activation) is not RunActionActivationRevalidationReceipt
            or activation.prepared_execution != prepared_execution
            or activation.spawn_commit != spawn_commit
            or session.activation_event_size_bytes(activation)
            > activation_event_size_bound
        ):
            raise RunActionRecoveryError(
                "execution adapter returned another or oversized activation"
            )
        _post_stage_descriptor, observed_workspace = self._inspect_workspace(
            session.reservation,
            descriptors,
        )
        expected_workspace = session.reservation.frontier.workspace_before
        if (
            None
            if observed_workspace is None
            else RunActionWorkspaceBinding.from_identity(observed_workspace)
        ) != expected_workspace or not self._security_is_current(frontier):
            return None
        session.commit_activation(activation)
        return self._recover_committed(
            session,
            execution_adapter,
            frontier,
            descriptors,
            workspace_lock_descriptor,
        )

    def _recover_committed(
        self,
        session,
        execution_adapter: RunActionExecutionAdapter,
        frontier: ReconciledRunFrontier,
        descriptors: ExitStack,
        workspace_lock_descriptor: int,
    ) -> RunActionAcceptance | None:
        activation_event = session.events[-1]
        if (
            activation_event.event_kind
            is not RunActionExecutionEventKind.ACTIVATION_COMMITTED
        ):
            raise RunActionRecoveryError(
                "provider continuation requires the durable activation tail"
            )
        with open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=self._publisher._settings,
        ) as release_inspection:
            workload_release_adoption = (
                release_inspection.adoption
                if release_inspection.presence is RunActionReleasePresence.PRESENT
                else None
            )
        query = RunActionCommittedSpawnQuery(
            preparation_allocation=session.events[1].preparation_allocation,
            activation_event=activation_event,
            workload_release_adoption=workload_release_adoption,
        )
        observation = execution_adapter.inspect_committed(query)
        if type(observation) is not RunActionCommittedSpawnObservation:
            raise RunActionRecoveryError(
                "execution adapter returned an invalid committed-spawn observation"
            )
        if observation.state is RunActionCommittedSpawnState.UNKNOWN:
            return None
        self._require_unchanged_host_workspace(
            session.reservation,
            descriptors,
            "host workspace changed before committed provider continuation",
        )
        if (
            observation.state
            in {
                RunActionCommittedSpawnState.INERT_CONTINUABLE,
                RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            }
            and workload_release_adoption is None
            and not self._security_is_current(frontier)
        ):
            return None
        capability = RunActionCommittedContinuationCapability(
            query=query,
            observation=observation,
            required_security_observation=(
                frontier.checkpoint.safety_state.security_observation
            ),
            security_authority=self._security_authority,
            credential_validity_authority=self._credential_validity_authority,
            release_clock=self._release_clock,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
        )
        outcome = capability._invoke_once(execution_adapter)
        if type(
            outcome
        ) is not RunActionContinuationOutcome or not self._continuation_outcome_allowed(
            observation, outcome
        ):
            raise RunActionRecoveryError(
                "execution adapter returned an invalid committed continuation"
            )
        self._require_unchanged_host_workspace(
            session.reservation,
            descriptors,
            "host workspace changed during committed provider continuation",
        )
        if outcome.state is RunActionContinuationState.PENDING:
            return None
        if outcome.state is RunActionContinuationState.RESULT_CAPTURED:
            final_release_adoption = self._require_release_adoption(
                activation_event,
                workload_release_adoption,
            )
            return self._record_and_interpret(
                session,
                outcome.result,
                final_release_adoption,
                descriptors,
                workspace_lock_descriptor,
            )
        raise RunActionRecoveryError(
            "execution adapter returned an unknown committed continuation"
        )

    @staticmethod
    def _continuation_outcome_allowed(
        observation: RunActionCommittedSpawnObservation,
        outcome: RunActionContinuationOutcome,
    ) -> bool:
        if (
            type(observation) is not RunActionCommittedSpawnObservation
            or type(outcome) is not RunActionContinuationOutcome
        ):
            return False
        admitted = {
            RunActionCommittedSpawnState.INERT_CONTINUABLE: {
                RunActionContinuationState.PENDING,
            },
            RunActionCommittedSpawnState.RUNNING_CONTINUABLE: {
                RunActionContinuationState.PENDING,
            },
            RunActionCommittedSpawnState.TERMINAL_CONTINUABLE: {
                RunActionContinuationState.PENDING,
                RunActionContinuationState.RESULT_CAPTURED,
            },
            RunActionCommittedSpawnState.UNKNOWN: set(),
        }
        return outcome.state in admitted[observation.state]

    def _require_unchanged_host_workspace(
        self,
        reservation: RunActionReservation,
        descriptors: ExitStack,
        message: str,
    ) -> RunWorkspaceFrontierIdentity | None:
        _workspace_descriptor, observed_workspace = self._inspect_workspace(
            reservation,
            descriptors,
        )
        expected_workspace = reservation.frontier.workspace_before
        if (
            None
            if observed_workspace is None
            else RunActionWorkspaceBinding.from_identity(observed_workspace)
        ) != expected_workspace:
            raise RunActionRecoveryError(message)
        return observed_workspace

    def _record_and_interpret(
        self,
        session,
        result: RunActionProviderResult,
        workload_release_adoption: RunActionWorkloadReleaseAdoption,
        descriptors: ExitStack,
        workspace_lock_descriptor: int,
    ) -> RunActionAcceptance:
        if type(result) is not RunActionProviderResult:
            raise RunActionRecoveryError(
                "execution adapter returned an invalid provider result"
            )
        spawn_commit = session.events[3].spawn_commit
        activation = session.events[4].activation_revalidation_receipt
        if not run_action_terminal_result_evidence_matches(
            result.terminal_observation,
            result.result_capture_receipt,
            activation,
            workload_release_adoption,
        ):
            raise RunActionRecoveryError(
                "execution adapter result differs from durable activation"
            )
        session.record_result(
            spawn_commit=spawn_commit,
            workload_release_adoption=workload_release_adoption,
            terminal_observation=result.terminal_observation,
            result_capture_receipt=result.result_capture_receipt,
            result_payload=result.result_payload,
        )
        return self._interpret_received(
            session,
            descriptors,
            workspace_lock_descriptor,
        )

    def _require_release_adoption(
        self,
        activation_event: RunActionExecutionEvent,
        expected: RunActionWorkloadReleaseAdoption | None,
    ) -> RunActionWorkloadReleaseAdoption:
        with open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=self._publisher._settings,
        ) as release_inspection:
            if release_inspection.presence is not RunActionReleasePresence.PRESENT:
                raise RunActionRecoveryError(
                    "captured provider result lacks its published workload release"
                )
            adoption = release_inspection.adoption
            if expected is not None and adoption != expected:
                raise RunActionRecoveryError(
                    "published workload release changed before result event"
                )
            return adoption

    def _interpret_received(
        self,
        session,
        descriptors: ExitStack,
        workspace_lock_descriptor: int,
    ) -> RunActionAcceptance:
        reservation = session.reservation
        result_receipt = session.events[-1].result_receipt
        request_payload = session.read_request()
        result_payload = session.read_result(result_receipt)
        _descriptor, observed_workspace = self._inspect_workspace(
            reservation,
            descriptors,
        )
        interpreter = self._resolve_result_interpreter(
            self._implementation_registry,
            reservation,
        )
        interpreted = interpreter.interpret(
            request_payload=request_payload,
            result_payload=result_payload,
        )
        repeated_interpretation = interpreter.interpret(
            request_payload=request_payload,
            result_payload=result_payload,
        )
        if (
            type(interpreted) is not RunActionInterpretedResult
            or repeated_interpretation != interpreted
        ):
            raise RunActionRecoveryError(
                "run action result interpretation is invalid or nondeterministic"
            )
        successful_edit = (
            reservation.intent.workspace_access
            is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
            and interpreted.disposition is RunActionResultDisposition.SUCCEEDED
        )
        _descriptor, confirmed_workspace = self._inspect_workspace(
            reservation,
            descriptors,
        )
        if confirmed_workspace != observed_workspace:
            raise RunActionRecoveryError(
                "workspace changed during run action result interpretation"
            )
        workspace_promotion = None
        if successful_edit:
            prepared_execution = session.events[2].prepared_execution
            with open_run_action_result_workspace(
                prepared_execution,
                result_receipt.result_capture_receipt,
            ) as candidate:
                workspace_promotion = self._workspace_promoter.stage(
                    result_receipt_id=result_receipt.result_receipt_id,
                    prepared_workspace_proof_id=(
                        prepared_execution.workspace_proof.prepared_workspace_proof_id
                    ),
                    predecessor=reservation.frontier.workspace_before,
                    candidate_descriptor=candidate.workspace_descriptor,
                    workspace_lock_descriptor=workspace_lock_descriptor,
                    _authority=_RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY,
                )
                candidate.require_current()
        session.decide_result(
            result_interpreter_identity=interpreter.result_interpreter_identity,
            disposition=interpreted.disposition,
            accepted_result_payload=interpreted.accepted_result_payload,
            workspace_promotion=workspace_promotion,
        )
        return self._accept_decided(
            session,
            descriptors,
            workspace_lock_descriptor,
        )

    def _accept_decided(
        self,
        session,
        descriptors: ExitStack,
        workspace_lock_descriptor: int,
    ) -> RunActionAcceptance:
        reservation = session.reservation
        decision = session.events[-1].result_decision
        before = reservation.frontier.workspace_before
        promotion = decision.workspace_promotion
        if promotion is None:
            _descriptor, confirmed_workspace = self._inspect_workspace(
                reservation,
                descriptors,
            )
            if (
                None
                if confirmed_workspace is None
                else RunActionWorkspaceBinding.from_identity(confirmed_workspace)
            ) != before:
                raise RunActionRecoveryError(
                    "workspace changed before durable run action acceptance"
                )
            workspace_after = confirmed_workspace
        else:
            prepared_execution = session.events[2].prepared_execution
            result_receipt = session.events[5].result_receipt
            workspace_after = self._workspace_promoter._promote_decided(
                predecessor=before,
                promotion=promotion,
                result_receipt_id=result_receipt.result_receipt_id,
                prepared_workspace_proof_id=(
                    prepared_execution.workspace_proof.prepared_workspace_proof_id
                ),
                workspace_lock_descriptor=workspace_lock_descriptor,
                _authority=_RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY,
            )
        durable_acceptance = session.accept_decision(
            workspace_after=workspace_after,
        )
        terminal_workspace = self._inspect_workspace_binding(
            durable_acceptance.workspace_after,
            descriptors,
        )
        if terminal_workspace != workspace_after:
            raise RunActionRecoveryError(
                "workspace changed after durable run action acceptance"
            )
        return durable_acceptance

    def _cleanup_latest_accepted_promotion(
        self,
        workspace_lock_descriptor: int,
        *,
        only_if_owned: bool = False,
    ) -> None:
        inspection = self._store.inspect()
        ordered = inspection.operations_since(RunActionLedgerSnapshot.empty())
        accepted_promotions = tuple(
            events
            for events in ordered
            if (
                len(events) == 8
                and events[-1].event_kind is RunActionExecutionEventKind.RESULT_ACCEPTED
                and events[-2].result_decision.workspace_promotion is not None
            )
        )
        if not accepted_promotions:
            return
        events = accepted_promotions[-1]
        promotion = events[-2].result_decision.workspace_promotion
        arguments = {
            "predecessor": events[0].reservation.frontier.workspace_before,
            "promotion": promotion,
            "result_receipt_id": events[5].result_receipt.result_receipt_id,
            "prepared_workspace_proof_id": (
                events[2].prepared_execution.workspace_proof.prepared_workspace_proof_id
            ),
            "workspace_lock_descriptor": workspace_lock_descriptor,
            "_authority": _RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY,
        }
        if only_if_owned:
            self._workspace_promoter._cleanup_accepted_if_owned(**arguments)
        else:
            self._workspace_promoter._cleanup_accepted(**arguments)

    def _inspect_workspace(
        self,
        reservation: RunActionReservation,
        descriptors: ExitStack,
    ) -> tuple[int | None, RunWorkspaceFrontierIdentity | None]:
        access = reservation.intent.workspace_access
        if access is RunFrontierWorkspaceAccess.NONE:
            return None, None
        descriptor, _identity = self._active_workspace._open_execution_workspace(
            descriptors
        )
        before = reservation.frontier.workspace_before
        observed = inspect_run_workspace_frontier(
            descriptor,
            settings=self._publisher._settings,
            expected_commit_sha=before.commit_sha,
        )
        return descriptor, observed

    def _inspect_workspace_binding(
        self,
        binding: RunActionWorkspaceBinding | None,
        descriptors: ExitStack,
    ) -> RunWorkspaceFrontierIdentity | None:
        if binding is None:
            return None
        descriptor, identity = self._active_workspace._open_execution_workspace(
            descriptors
        )
        observed = inspect_run_workspace_frontier(
            descriptor,
            settings=self._publisher._settings,
            expected_commit_sha=binding.commit_sha,
        )
        if (
            identity
            != (
                binding.workspace_device,
                binding.workspace_inode,
            )
            or observed != binding.to_identity()
        ):
            raise RunActionRecoveryError(
                "public workspace differs from its accepted binding"
            )
        return observed

    def _inspect_pre_spawn_workspace(
        self,
        reservation: RunActionReservation,
        descriptors: ExitStack,
    ) -> tuple[int | None, RunWorkspaceFrontierIdentity | None]:
        if reservation.intent.workspace_access is RunFrontierWorkspaceAccess.NONE:
            return None, None
        descriptor, _identity = self._active_workspace._open_execution_workspace(
            descriptors
        )
        observed = inspect_run_workspace_frontier(
            descriptor,
            settings=self._publisher._settings,
            expected_commit_sha=None,
        )
        return descriptor, observed

    def _workspace_source_path(
        self,
        reservation: RunActionReservation,
        descriptor: int | None,
    ) -> Path | None:
        access = reservation.intent.workspace_access
        if access is RunFrontierWorkspaceAccess.NONE:
            if descriptor is not None:
                raise RunActionRecoveryError(
                    "workspace-free preparation retained a workspace descriptor"
                )
            return None
        if type(descriptor) is not int or descriptor < 0:
            raise RunActionRecoveryError(
                "workspace preparation lacks its live descriptor"
            )
        layout = self._active_workspace.bootstrap_pin.installation_receipt.layout
        path = self._active_workspace.run_root / layout.workspace_relative_path
        _require_workspace_source_path(path, descriptor)
        return path

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
                    session.read_decided_result(events[-2].result_decision)
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
    def _resolve_execution_adapter(
        registry: RunActionRecoveryImplementationRegistry,
        reservation: RunActionReservation,
    ) -> RunActionExecutionAdapter:
        identity = reservation.intent.boundary_identity
        execution_adapter = registry.resolve_execution(identity)
        if (
            not hasattr(execution_adapter, "execution_lifecycle_identity")
            or execution_adapter.execution_lifecycle_identity
            != identity.execution_lifecycle_identity
            or not hasattr(execution_adapter, "execution_policy")
            or type(execution_adapter.execution_policy)
            is not DockerRunActionExecutionPolicy
            or execution_adapter.execution_policy.docker_execution_policy_id
            != identity.execution_lifecycle_identity.execution_policy_id
            or any(
                not hasattr(execution_adapter, name)
                for name in _EXECUTION_ADAPTER_METHOD_NAMES
            )
        ):
            raise RunActionRecoveryError(
                "run action execution adapter differs from its durable identity"
            )
        return execution_adapter

    @staticmethod
    def _resolve_result_interpreter(
        registry: RunActionRecoveryImplementationRegistry,
        reservation: RunActionReservation,
    ) -> RunActionResultInterpreter:
        identity = reservation.intent.boundary_identity
        interpreter = registry.resolve_interpreter(identity)
        if (
            not hasattr(interpreter, "result_interpreter_identity")
            or interpreter.result_interpreter_identity
            != identity.result_interpreter_identity
            or any(
                not hasattr(interpreter, name)
                for name in _RESULT_INTERPRETER_METHOD_NAMES
            )
        ):
            raise RunActionRecoveryError(
                "run action result interpreter differs from its durable identity"
            )
        return interpreter

    @staticmethod
    def _require_reservation_frontier(
        frontier: ReconciledRunFrontier,
        reservation: RunActionReservation,
    ) -> None:
        binding = reservation.frontier
        checkpoint = frontier.checkpoint
        if (
            checkpoint.status is not RunCheckpointStatus.ACTIVE
            or checkpoint.last_stop is not None
            or checkpoint.safety_state.disposition
            is RunEligibilityDisposition.SECURITY_BLOCKED
            or reservation.intent.boundary is not checkpoint.safety_state.boundary
            or binding.bootstrap_pin_id
            != checkpoint.safety_state.bootstrap_pin.bootstrap_pin_id
            or binding.run_checkpoint_id != frontier.run_checkpoint_id
            or binding.safety_state_id != checkpoint.safety_state.safety_state_id
            or binding.security_observation_id
            != checkpoint.safety_state.security_observation.observation_id
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

    def _release_receipt_size_bound(
        self,
        execution_adapter: RunActionExecutionAdapter,
        reservation: RunActionReservation,
    ) -> int:
        policy_bound = (
            execution_adapter.execution_policy.supervisor_limits.release_receipt_size_bytes
        )
        policy_commit_timeout = (
            execution_adapter.execution_policy.supervisor_limits.release_commit_timeout_seconds
        )
        configured_bound = (
            self._publisher._settings.run_action_release_receipt_size_bytes
        )
        configured_commit_timeout = (
            self._publisher._settings.run_action_release_commit_timeout_seconds
        )
        first = execution_adapter.release_receipt_size_bound(
            reservation=reservation,
        )
        second = execution_adapter.release_receipt_size_bound(
            reservation=reservation,
        )
        if (
            type(first) is not int
            or first <= self._publisher._settings.run_action_process_snapshot_size_bytes
            or second != first
            or policy_bound != configured_bound
            or policy_commit_timeout != configured_commit_timeout
            or first > policy_bound
        ):
            raise RunActionRecoveryError(
                "run action release-receipt envelope is invalid or too large"
            )
        return first

    def _require_owner_process(self) -> None:
        with _RECOVERY_COORDINATOR_LOCK:
            issued = _ISSUED_RECOVERY_COORDINATORS.get(id(self))
        if issued is not self or self._owner_process_id != os.getpid():
            raise RunActionRecoveryError(
                "run action recovery coordinator is cloned or foreign"
            )


__all__ = [
    "RunActionActivationCapability",
    "RunActionCommittedContinuationCapability",
    "RunActionCommittedSpawnObservation",
    "RunActionCommittedSpawnQuery",
    "RunActionCommittedSpawnState",
    "RunActionContinuationOutcome",
    "RunActionContinuationState",
    "RunActionUnactivatedSpawnObservation",
    "RunActionUnactivatedSpawnQuery",
    "RunActionUnactivatedSpawnState",
    "RunActionExecutionAdapter",
    "RunActionInterpretedResult",
    "RunActionPreparationCapability",
    "RunActionPreparationMode",
    "RunActionPreparationObservation",
    "RunActionPreparationOrigin",
    "RunActionPreparationState",
    "RunActionProviderResult",
    "RunActionRecoveredOperation",
    "RunActionRecoveryCoordinator",
    "RunActionRecoveryError",
    "RunActionRecoveryImplementation",
    "RunActionRecoveryImplementationRegistry",
    "RunActionRecoveryPlan",
    "RunActionRecoveryReport",
    "RunActionResultInterpreter",
]
