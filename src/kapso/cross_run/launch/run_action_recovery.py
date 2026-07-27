"""Fail-closed recovery of durable nonterminal run-action prefixes."""

from __future__ import annotations

import os
import re
import stat
from contextlib import ExitStack
from copy import deepcopy
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
from kapso.cross_run.launch.run_action_containment_contracts import (
    RunActionTimeoutContainmentResult,
    RunActionTimeoutContainmentSignal,
    RunActionTimeoutContainmentState,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_credential_broker import (
    _RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
    burn_run_action_credential_materialization,
    RunActionCredentialBrokerRegistry,
    RunActionCredentialIssueResponse,
    RunActionCredentialLeaseStatus,
    RunActionCredentialMaterialization,
)
from kapso.cross_run.launch.run_action_credential_contracts import (
    credential_retirement_intent_matches_activation,
    RunActionCredentialRetirementIntent,
    RunActionPreReleaseCredentialObservation,
    RunActionPreReleaseCredentialState,
)
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_control_candidate import (
    _CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
    _RunActionControlFileTransition,
    _RunActionFrozenControlFileCandidate,
    _RunActionLinkedControlFileEvidence,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierRunningContainerObservation,
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
)
from kapso.cross_run.launch.run_action_timeout_adoption import (
    open_run_action_timeout_inspection,
    RunActionTimeoutInspectionLease,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionCredentialValidityObservation,
    RunActionReleaseAuthorizationObservation,
    RunActionWorkloadReleaseAdoption,
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_release_envelope import (
    workload_release_receipt_size_bound,
)
from kapso.cross_run.launch.run_action_result_authority import (
    run_action_terminal_result_evidence_matches,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
    RunActionViewBinding,
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_resource_finalization import (
    require_run_action_resource_finalization_authority,
    RunActionResourceFinalizationAuthority,
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
from kapso.cross_run.launch.run_action_credential_expiry_envelope import (
    credential_expiry_termination_event_size_bound,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionExecutionPolicy,
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
    RunActionResultCaptureReceipt,
    RunActionTerminalObservation,
    run_action_credential_lease_authority_id,
    run_action_credential_lease_request,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    provider_termination_matches_durable_activation,
    run_action_pre_release_main_loss_observation_token,
    run_action_pre_release_main_terminal_observation_token,
    run_action_running_container_occurrence_matches,
    run_action_timeout_directive_evidence_matches,
    run_action_timeout_publication_evidence_matches,
    RunActionPreReleaseMainTerminalObservation,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
    RunActionTimeoutDirective,
    RunActionTimeoutDirectivePublicationReceipt,
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
_RUN_ACTION_PREPARATION_AUTHORITY = object()
_RUN_ACTION_ACTIVATION_AUTHORITY = object()
_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY = object()
_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY = object()
_RUN_ACTION_START_AUTHORITY = object()
_RUN_ACTION_CREDENTIAL_RETIREMENT_AUTHORITY = object()
_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY = object()
_RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY = object()
_RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY = object()
_RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY = object()
_RUN_ACTION_RESULT_CAPTURE_AUTHORITY = object()
_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY = object()
_RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY = object()
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
_ISSUED_ACTIVATION_OWNERSHIP: WeakKeyDictionary[
    object,
    "_RunActionActivationOwnedResources",
] = WeakKeyDictionary()
_ISSUED_COMMITTED_CONTINUATION_CAPABILITIES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES: WeakKeyDictionary[
    object,
    "_RunActionIssuedReleaseAuthorities",
] = WeakKeyDictionary()
_ISSUED_PROVIDER_TERMINATION_PUBLICATION_FENCES: WeakKeyDictionary[
    object,
    "_RunActionProviderTerminationPublicationFenceIssuance",
] = WeakKeyDictionary()
_RECOVERY_COORDINATOR_LOCK = Lock()
_RECOVERY_IMPLEMENTATION_REGISTRY_LOCK = Lock()
_PREPARATION_CAPABILITY_LOCK = Lock()
_ACTIVATION_CAPABILITY_LOCK = Lock()
_COMMITTED_CONTINUATION_CAPABILITY_LOCK = Lock()
_PROVIDER_TERMINATION_PUBLICATION_FENCE_LOCK = Lock()
_TERMINAL_KINDS = {
    RunActionExecutionEventKind.PROVIDER_TERMINATED,
    RunActionExecutionEventKind.RESULT_ACCEPTED,
    RunActionExecutionEventKind.CANCELLED,
    RunActionExecutionEventKind.FRONTIER_INVALIDATED,
}
_EXECUTION_ADAPTER_METHOD_NAMES = (
    "prepared_event_size_bound",
    "activation_event_size_bound",
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
    """Coordinator-held continuation identity and release authorities."""

    query: "RunActionCommittedSpawnQuery"
    adapter_query: "RunActionCommittedSpawnQuery"
    observation: RunActionCommittedSpawnObservation
    adapter_observation: RunActionCommittedSpawnObservation
    pre_release_credential_observation: RunActionPreReleaseCredentialObservation | None
    adapter_pre_release_credential_observation: (
        RunActionPreReleaseCredentialObservation | None
    )
    required_security_observation: SecurityDenylistObservation
    security_authority: object
    credential_broker_registry: RunActionCredentialBrokerRegistry
    clock: _SystemRunActionClock
    release_receipt_size_bound_bytes: int | None
    provider_termination_registration: "_RunActionProviderTerminationRegistration"
    terminal_result_registration: "_RunActionTerminalResultRegistration"
    capability_lifecycle: "_RunActionContinuationCapabilityLifecycle"

    def __post_init__(self) -> None:
        if (
            type(self.query) is not RunActionCommittedSpawnQuery
            or type(self.adapter_query) is not RunActionCommittedSpawnQuery
            or type(self.observation) is not RunActionCommittedSpawnObservation
            or type(self.adapter_observation) is not RunActionCommittedSpawnObservation
            or self.adapter_query != self.query
            or self.adapter_observation != self.observation
            or (
                self.pre_release_credential_observation is not None
                and type(self.pre_release_credential_observation)
                is not RunActionPreReleaseCredentialObservation
            )
            or (
                self.adapter_pre_release_credential_observation is not None
                and type(self.adapter_pre_release_credential_observation)
                is not RunActionPreReleaseCredentialObservation
            )
            or self.adapter_pre_release_credential_observation
            != self.pre_release_credential_observation
            or type(self.required_security_observation)
            is not SecurityDenylistObservation
            or self.required_security_observation.matched_revocations
            or not hasattr(self.security_authority, "observe_exact_descendant_of")
            or type(self.credential_broker_registry)
            is not RunActionCredentialBrokerRegistry
            or type(self.clock) is not _SystemRunActionClock
            or (
                self.release_receipt_size_bound_bytes is not None
                and (
                    type(self.release_receipt_size_bound_bytes) is not int
                    or self.release_receipt_size_bound_bytes <= 0
                )
            )
            or type(self.provider_termination_registration)
            is not _RunActionProviderTerminationRegistration
            or type(self.terminal_result_registration)
            is not _RunActionTerminalResultRegistration
            or type(self.capability_lifecycle)
            is not _RunActionContinuationCapabilityLifecycle
        ):
            raise RunActionRecoveryError(
                "issued release authorities are incomplete or unsafe"
            )


@dataclass
class _RunActionProviderTerminationRegistration:
    """Coordinator-private terminal receipt and retained publication fence."""

    state: str = "ready"
    receipt: RunActionProviderTerminationReceipt | None = None
    publication_fence: "RunActionProviderTerminationPublicationFence | None" = None
    publication_fence_handed_off: bool = False


@dataclass
class _RunActionTerminalResultRegistration:
    """Coordinator-private canonical terminal and result evidence."""

    terminal_observation: RunActionTerminalObservation | None = None
    captured_result: "RunActionProviderResult | None" = None


@dataclass
class _RunActionContinuationCapabilityLifecycle:
    """Coordinator-private invocation ownership for one callback."""

    owner_process_id: int
    invoking_thread_id: int | None = None
    state: str = "ready"
    start_state: str = "ready"
    credential_retirement_state: str = "ready"
    started_running_observation: RunActionBarrierRunningContainerObservation | None = (
        None
    )
    release_publication_state: str = "ready"
    timeout_publication_state: str = "ready"
    timeout_directive_publication: (
        RunActionTimeoutDirectivePublicationReceipt | None
    ) = None
    timeout_containment_state: str = "ready"
    timeout_containment_result: RunActionTimeoutContainmentResult | None = None
    terminal_inspection_state: str = "ready"
    result_capture_state: str = "ready"


@dataclass
class _RunActionProviderTerminationPublicationFenceIssuance:
    """Coordinator-private ownership of one retained physical proof."""

    source: object | None
    owner_process_id: int
    owner_thread_id: int
    closed: bool


def _issued_continuation_authorities(
    capability: object,
    *,
    _authority: object,
) -> _RunActionIssuedReleaseAuthorities:
    """Open mutable continuation state only to coordinator-internal call sites."""

    if _authority is not _RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY:
        raise RunActionRecoveryError(
            "committed continuation private state authority is unavailable"
        )
    issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(capability))
    authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(capability)
    if (
        issued is not capability
        or type(authorities) is not _RunActionIssuedReleaseAuthorities
    ):
        raise RunActionRecoveryError(
            "committed continuation private state authority is unavailable"
        )
    return authorities


def _continuation_lifecycle(
    capability: object,
    *,
    _authority: object,
) -> _RunActionContinuationCapabilityLifecycle:
    """Open one mutable lifecycle only with coordinator-private authority."""

    return _issued_continuation_authorities(
        capability,
        _authority=_authority,
    ).capability_lifecycle


def _provider_termination_registration(
    capability: object,
    *,
    _authority: object,
) -> _RunActionProviderTerminationRegistration:
    """Open one mutable termination registration only inside the coordinator."""

    return _issued_continuation_authorities(
        capability,
        _authority=_authority,
    ).provider_termination_registration


def _terminal_result_registration(
    capability: object,
    *,
    _authority: object,
) -> _RunActionTerminalResultRegistration:
    """Open one mutable result registration only inside the coordinator."""

    return _issued_continuation_authorities(
        capability,
        _authority=_authority,
    ).terminal_result_registration


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
    PRE_RELEASE_MAIN_LOSS_CONTINUABLE = "pre_release_main_loss_continuable"
    PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE = "pre_release_main_terminal_continuable"
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
    operation_id: str
    accepted_result_payload: bytes
    expected_workspace_before_source_tree_digest: str | None = None
    expected_workspace_after_source_tree_digest: str | None = None

    def __post_init__(self) -> None:
        workspace_expectations = (
            self.expected_workspace_before_source_tree_digest,
            self.expected_workspace_after_source_tree_digest,
        )
        if (
            type(self.disposition) is not RunActionResultDisposition
            or not isinstance(self.operation_id, str)
            or not self.operation_id
            or type(self.accepted_result_payload) is not bytes
            or not self.accepted_result_payload
            or (workspace_expectations[0] is None)
            != (workspace_expectations[1] is None)
            or any(
                digest is not None
                and (
                    type(digest) is not str
                    or _SHA256_DIGEST_PATTERN.fullmatch(digest) is None
                )
                for digest in workspace_expectations
            )
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
            RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
            RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE,
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
    TIMEOUT_PUBLISHED = "timeout_published"
    RESULT_CAPTURED = "result_captured"
    PROVIDER_TERMINATED = "provider_terminated"


class RunActionProviderTerminationPublicationFence:
    """Coordinator-owned physical fence retained through terminal publication."""

    __slots__ = ("__weakref__",)

    def __init__(self, *, source: object, _authority: object) -> None:
        if (
            not hasattr(source, "require_current")
            or not hasattr(source, "close")
            or _authority
            is not _RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "provider termination publication fence lacks issued authority"
            )
        with _PROVIDER_TERMINATION_PUBLICATION_FENCE_LOCK:
            _ISSUED_PROVIDER_TERMINATION_PUBLICATION_FENCES[self] = (
                _RunActionProviderTerminationPublicationFenceIssuance(
                    source=source,
                    owner_process_id=os.getpid(),
                    owner_thread_id=get_ident(),
                    closed=False,
                )
            )
        self.require_current()

    def require_current(self) -> None:
        issuance = _ISSUED_PROVIDER_TERMINATION_PUBLICATION_FENCES.get(self)
        if (
            type(issuance) is not _RunActionProviderTerminationPublicationFenceIssuance
            or issuance.owner_process_id != os.getpid()
        ):
            raise RunActionRecoveryError(
                "provider termination publication fence is closed or foreign"
            )
        with _PROVIDER_TERMINATION_PUBLICATION_FENCE_LOCK:
            issuance = _ISSUED_PROVIDER_TERMINATION_PUBLICATION_FENCES.get(self)
            if (
                type(issuance)
                is not _RunActionProviderTerminationPublicationFenceIssuance
                or issuance.closed
                or issuance.owner_process_id != os.getpid()
                or issuance.owner_thread_id != get_ident()
                or not hasattr(issuance.source, "require_current")
            ):
                raise RunActionRecoveryError(
                    "provider termination publication fence is closed or foreign"
                )
            source = issuance.source
        source.require_current()
        with _PROVIDER_TERMINATION_PUBLICATION_FENCE_LOCK:
            if (
                _ISSUED_PROVIDER_TERMINATION_PUBLICATION_FENCES.get(self)
                is not issuance
                or issuance.closed
                or issuance.source is not source
            ):
                raise RunActionRecoveryError(
                    "provider termination publication fence changed during validation"
                )

    def close(self) -> None:
        issuance = _ISSUED_PROVIDER_TERMINATION_PUBLICATION_FENCES.get(self)
        if (
            type(issuance) is not _RunActionProviderTerminationPublicationFenceIssuance
            or issuance.owner_process_id != os.getpid()
        ):
            raise RunActionRecoveryError(
                "provider termination publication fence is already closed or foreign"
            )
        with _PROVIDER_TERMINATION_PUBLICATION_FENCE_LOCK:
            issuance = _ISSUED_PROVIDER_TERMINATION_PUBLICATION_FENCES.get(self)
            if (
                type(issuance)
                is not _RunActionProviderTerminationPublicationFenceIssuance
                or issuance.closed
                or issuance.owner_process_id != os.getpid()
                or issuance.owner_thread_id != get_ident()
                or not hasattr(issuance.source, "close")
            ):
                raise RunActionRecoveryError(
                    "provider termination publication fence is already closed or foreign"
                )
            source = issuance.source
            issuance.closed = True
            issuance.source = None
        source.close()

    def __copy__(self):
        raise RunActionRecoveryError(
            "provider termination publication fence cannot be copied"
        )

    def __deepcopy__(self, memo):
        raise RunActionRecoveryError(
            "provider termination publication fence cannot be copied"
        )

    def __reduce__(self):
        raise RunActionRecoveryError(
            "provider termination publication fence cannot be serialized"
        )


@dataclass(frozen=True)
class RunActionContinuationOutcome:
    """Typed continuation outcome with exactly one registered terminal branch."""

    state: RunActionContinuationState
    result: RunActionProviderResult | None
    provider_termination_receipt: RunActionProviderTerminationReceipt | None
    timeout_directive_publication: RunActionTimeoutDirectivePublicationReceipt | None
    provider_termination_publication_fence: (
        RunActionProviderTerminationPublicationFence | None
    ) = None

    def __post_init__(self) -> None:
        if type(self.state) is not RunActionContinuationState:
            raise RunActionRecoveryError(
                "run action continuation outcome differs from its state"
            )
        result_present = type(self.result) is RunActionProviderResult
        termination_present = (
            type(self.provider_termination_receipt)
            is RunActionProviderTerminationReceipt
        )
        timeout_present = (
            type(self.timeout_directive_publication)
            is RunActionTimeoutDirectivePublicationReceipt
        )
        fence_present = (
            type(self.provider_termination_publication_fence)
            is RunActionProviderTerminationPublicationFence
        )
        expected_presence = {
            RunActionContinuationState.PENDING: (False, False, False, False),
            RunActionContinuationState.TIMEOUT_PUBLISHED: (
                False,
                False,
                True,
                False,
            ),
            RunActionContinuationState.RESULT_CAPTURED: (True, False, False, False),
            RunActionContinuationState.PROVIDER_TERMINATED: (
                False,
                True,
                False,
                (
                    self.provider_termination_receipt is not None
                    and self.provider_termination_receipt.reason
                    in {
                        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
                        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL,
                        RunActionProviderTerminationReason.CREDENTIAL_EXPIRED,
                    }
                ),
            ),
        }
        if (
            (result_present, termination_present, timeout_present, fence_present)
            != expected_presence[self.state]
            or (
                self.result is not None
                and type(self.result) is not RunActionProviderResult
            )
            or (
                self.provider_termination_receipt is not None
                and type(self.provider_termination_receipt)
                is not RunActionProviderTerminationReceipt
            )
            or (
                self.timeout_directive_publication is not None
                and type(self.timeout_directive_publication)
                is not RunActionTimeoutDirectivePublicationReceipt
            )
            or (
                self.provider_termination_publication_fence is not None
                and type(self.provider_termination_publication_fence)
                is not RunActionProviderTerminationPublicationFence
            )
        ):
            raise RunActionRecoveryError(
                "run action continuation outcome differs from its state"
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
        RunActionPreparationCapability._finish_invocation(self._capability)
        return False


@dataclass
class _RunActionActivationOwnedResources:
    """Coordinator-private activation inputs and resources."""

    owner_process_id: int
    invoking_thread_id: int | None
    state: str
    prepared_execution: RunActionPreparedExecution
    spawn_commit: RunActionSpawnCommit
    adapter_prepared_execution: RunActionPreparedExecution
    adapter_spawn_commit: RunActionSpawnCommit
    request_payload: bytes
    credential_issue_response: RunActionCredentialIssueResponse | None
    credential_broker_registry: RunActionCredentialBrokerRegistry
    credential_materialization: RunActionCredentialMaterialization | None
    workspace_descriptor: int | None
    workspace_descriptor_identity: tuple[int, int] | None


class RunActionActivationCapability:
    """Single-use delivery and inert-revalidation authority before event 5."""

    __slots__ = ("_creation_process_id", "__weakref__")

    def __init__(
        self,
        *,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        request_payload: bytes,
        credential_issue_response: RunActionCredentialIssueResponse | None,
        credential_broker_registry: RunActionCredentialBrokerRegistry,
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
            or (
                credential_issue_response is not None
                and type(credential_issue_response)
                is not RunActionCredentialIssueResponse
            )
            or (
                (credential_issue_response is None)
                != (
                    prepared_execution.preparation_claim.execution_policy.credential_policy.mode
                    is RunActionCredentialMode.NONE
                )
            )
            or type(credential_broker_registry) is not RunActionCredentialBrokerRegistry
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
        credential_broker_registry.require_policy(
            prepared_execution.preparation_claim.execution_policy.credential_policy
        )
        owned_workspace_descriptor = (
            None if workspace_descriptor is None else os.dup(workspace_descriptor)
        )
        workspace_status = (
            None
            if owned_workspace_descriptor is None
            else os.fstat(owned_workspace_descriptor)
        )
        private_prepared_execution = deepcopy(prepared_execution)
        private_spawn_commit = deepcopy(spawn_commit)
        owned_resources = _RunActionActivationOwnedResources(
            owner_process_id=os.getpid(),
            invoking_thread_id=None,
            state="ready",
            prepared_execution=private_prepared_execution,
            spawn_commit=private_spawn_commit,
            adapter_prepared_execution=deepcopy(private_prepared_execution),
            adapter_spawn_commit=deepcopy(private_spawn_commit),
            request_payload=request_payload,
            credential_issue_response=credential_issue_response,
            credential_broker_registry=credential_broker_registry,
            credential_materialization=None,
            workspace_descriptor=owned_workspace_descriptor,
            workspace_descriptor_identity=(
                None
                if workspace_status is None
                else (workspace_status.st_dev, workspace_status.st_ino)
            ),
        )
        self._creation_process_id = os.getpid()
        with _ACTIVATION_CAPABILITY_LOCK:
            _ISSUED_ACTIVATION_CAPABILITIES[id(self)] = self
            _ISSUED_ACTIVATION_OWNERSHIP[self] = owned_resources

    @property
    def prepared_execution(self) -> RunActionPreparedExecution:
        return self._require_active_invocation().adapter_prepared_execution

    @property
    def spawn_commit(self) -> RunActionSpawnCommit:
        return self._require_active_invocation().adapter_spawn_commit

    @property
    def request_payload(self) -> bytes:
        return self._require_active_invocation().request_payload

    @property
    def credential_materialization(
        self,
    ) -> RunActionCredentialMaterialization | None:
        return self._require_active_invocation().credential_materialization

    @property
    def workspace_descriptor(self) -> int | None:
        return self._require_active_invocation().workspace_descriptor

    def _invoke_once(
        self,
        execution_adapter: "RunActionExecutionAdapter",
    ) -> RunActionActivationRevalidationReceipt:
        with _RunActionActivationOwnership(self):
            with self._begin_invocation():
                return execution_adapter.stage_activation(self)

    def _begin_invocation(self) -> "_RunActionActivationInvocation":
        if self._creation_process_id != os.getpid():
            raise RunActionRecoveryError(
                "run action activation capability is spent, cloned, or foreign"
            )
        activation_inputs_changed = False
        with _ACTIVATION_CAPABILITY_LOCK:
            issued = _ISSUED_ACTIVATION_CAPABILITIES.get(id(self))
            owned_resources = _ISSUED_ACTIVATION_OWNERSHIP.get(self)
            if (
                issued is not self
                or type(owned_resources) is not _RunActionActivationOwnedResources
                or owned_resources.owner_process_id != os.getpid()
                or owned_resources.state != "ready"
            ):
                raise RunActionRecoveryError(
                    "run action activation capability is spent, cloned, or foreign"
                )
            owned_resources.state = "invoking"
            owned_resources.invoking_thread_id = get_ident()
            if owned_resources.credential_issue_response is not None:
                owned_resources.credential_materialization = (
                    owned_resources.credential_broker_registry.materialize_exact(
                        owned_resources.credential_issue_response,
                        owned_resources.prepared_execution,
                        owned_resources.spawn_commit,
                    )
                )
                owned_resources.credential_issue_response = None
        return _RunActionActivationInvocation(
            self,
            owner_process_id=os.getpid(),
            invoking_thread_id=get_ident(),
        )

    def _require_active_invocation(self) -> _RunActionActivationOwnedResources:
        if self._creation_process_id != os.getpid():
            raise RunActionRecoveryError(
                "run action activation capability is not in its one invocation"
            )
        with _ACTIVATION_CAPABILITY_LOCK:
            issued = _ISSUED_ACTIVATION_CAPABILITIES.get(id(self))
            owned_resources = _ISSUED_ACTIVATION_OWNERSHIP.get(self)
            if (
                issued is not self
                or type(owned_resources) is not _RunActionActivationOwnedResources
                or owned_resources.owner_process_id != os.getpid()
                or owned_resources.state != "invoking"
                or owned_resources.invoking_thread_id != get_ident()
            ):
                raise RunActionRecoveryError(
                    "run action activation capability is not in its one invocation"
                )
            return owned_resources

    def _finish_invocation(
        self,
        *,
        owner_process_id: int,
        invoking_thread_id: int,
        _authority: object,
    ) -> None:
        if (
            owner_process_id != os.getpid()
            or invoking_thread_id != get_ident()
            or _authority is not _RUN_ACTION_ACTIVATION_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action activation capability invocation changed"
            )
        with _ACTIVATION_CAPABILITY_LOCK:
            issued = _ISSUED_ACTIVATION_CAPABILITIES.get(id(self))
            owned_resources = _ISSUED_ACTIVATION_OWNERSHIP.get(self)
            if (
                issued is not self
                or type(owned_resources) is not _RunActionActivationOwnedResources
                or owned_resources.owner_process_id != owner_process_id
                or owned_resources.state != "invoking"
                or owned_resources.invoking_thread_id != invoking_thread_id
            ):
                raise RunActionRecoveryError(
                    "run action activation capability invocation changed"
                )
            activation_inputs_changed = (
                owned_resources.adapter_prepared_execution
                != owned_resources.prepared_execution
                or owned_resources.adapter_spawn_commit != owned_resources.spawn_commit
            )
            owned_resources.state = "spent"
            owned_resources.invoking_thread_id = None
            _ISSUED_ACTIVATION_CAPABILITIES.pop(id(self))
        RunActionActivationCapability._release_owned_resources(self)
        if activation_inputs_changed:
            raise RunActionRecoveryError(
                "run action activation inputs changed during adapter invocation"
            )

    def _abandon(
        self,
        *,
        owner_process_id: int,
        owner_thread_id: int,
        _authority: object,
    ) -> None:
        if (
            owner_process_id != os.getpid()
            or owner_thread_id != get_ident()
            or _authority is not _RUN_ACTION_ACTIVATION_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action activation capability ownership changed"
            )
        with _ACTIVATION_CAPABILITY_LOCK:
            issued = _ISSUED_ACTIVATION_CAPABILITIES.get(id(self))
            owned_resources = _ISSUED_ACTIVATION_OWNERSHIP.get(self)
            if issued is None and owned_resources is None:
                return
            if (
                issued is not self
                or type(owned_resources) is not _RunActionActivationOwnedResources
                or owned_resources.owner_process_id != owner_process_id
                or owned_resources.state not in {"ready", "invoking"}
                or (
                    owned_resources.state == "invoking"
                    and owned_resources.invoking_thread_id != owner_thread_id
                )
            ):
                raise RunActionRecoveryError(
                    "run action activation capability ownership changed"
                )
            owned_resources.state = "spent"
            owned_resources.invoking_thread_id = None
            _ISSUED_ACTIVATION_CAPABILITIES.pop(id(self), None)
        self._release_owned_resources()

    def _release_owned_resources(self) -> None:
        with _ACTIVATION_CAPABILITY_LOCK:
            owned_resources = _ISSUED_ACTIVATION_OWNERSHIP.get(self)
        if type(owned_resources) is not _RunActionActivationOwnedResources:
            raise RunActionRecoveryError("run action activation ownership is missing")
        if owned_resources.credential_materialization is not None:
            burn_run_action_credential_materialization(
                owned_resources.credential_materialization,
                _authority=_RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
            )
            owned_resources.credential_materialization = None
        if owned_resources.credential_issue_response is not None:
            owned_resources.credential_broker_registry.burn_issue_response_exact(
                owned_resources.credential_issue_response
            )
            owned_resources.credential_issue_response = None
        if owned_resources.workspace_descriptor is not None:
            workspace_descriptor = owned_resources.workspace_descriptor
            owned_resources.workspace_descriptor = None
            workspace_descriptor_path = Path(f"/proc/self/fd/{workspace_descriptor}")
            if workspace_descriptor_path.exists():
                workspace_descriptor_status = workspace_descriptor_path.stat()
                if owned_resources.workspace_descriptor_identity == (
                    workspace_descriptor_status.st_dev,
                    workspace_descriptor_status.st_ino,
                ):
                    os.close(workspace_descriptor)
            owned_resources.workspace_descriptor_identity = None
        with _ACTIVATION_CAPABILITY_LOCK:
            if _ISSUED_ACTIVATION_OWNERSHIP.get(self) is not owned_resources:
                raise RunActionRecoveryError(
                    "run action activation ownership changed during close"
                )
            _ISSUED_ACTIVATION_OWNERSHIP.pop(self)


class _RunActionActivationInvocation:
    """Burn one activation capability on every callback exit."""

    def __init__(
        self,
        capability: RunActionActivationCapability,
        *,
        owner_process_id: int,
        invoking_thread_id: int,
    ) -> None:
        self._capability = capability
        self._owner_process_id = owner_process_id
        self._invoking_thread_id = invoking_thread_id

    def __enter__(self) -> RunActionActivationCapability:
        return self._capability

    def __exit__(self, exception_type, exception, traceback) -> bool:
        RunActionActivationCapability._finish_invocation(
            self._capability,
            owner_process_id=self._owner_process_id,
            invoking_thread_id=self._invoking_thread_id,
            _authority=_RUN_ACTION_ACTIVATION_AUTHORITY,
        )
        return False


class _RunActionActivationOwnership:
    """Close pre-invocation response, materialization, and descriptor authority."""

    def __init__(self, capability: RunActionActivationCapability) -> None:
        self._capability = capability
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()

    def __enter__(self) -> RunActionActivationCapability:
        return self._capability

    def __exit__(self, exception_type, exception, traceback) -> bool:
        RunActionActivationCapability._abandon(
            self._capability,
            owner_process_id=self._owner_process_id,
            owner_thread_id=self._owner_thread_id,
            _authority=_RUN_ACTION_ACTIVATION_AUTHORITY,
        )
        return False


class RunActionCommittedContinuationCapability:
    """Single-use authority for one exact observed event-5 continuation."""

    __slots__ = (
        "_creation_process_id",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        query: "RunActionCommittedSpawnQuery",
        observation: RunActionCommittedSpawnObservation,
        required_security_observation: SecurityDenylistObservation,
        security_authority: object,
        credential_broker_registry: RunActionCredentialBrokerRegistry,
        release_clock: _SystemRunActionClock,
        pre_release_credential_observation: (
            RunActionPreReleaseCredentialObservation | None
        ) = None,
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
        credential_observation_required = (
            type(observation) is RunActionCommittedSpawnObservation
            and query.credential_retirement_intent is None
            and query.control_directory_topology
            is RunActionControlDirectoryTopology.EMPTY
            and observation.state
            in {
                RunActionCommittedSpawnState.INERT_CONTINUABLE,
                RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            }
            and activation.prepared_execution.preparation_claim.execution_policy.credential_policy.mode
            is RunActionCredentialMode.SUPERVISOR_FILE
        )
        release_bound_required = (
            query.credential_retirement_intent is None
            and query.control_directory_topology
            is RunActionControlDirectoryTopology.EMPTY
        )
        if (
            activation_event.reservation != reservation
            or type(observation) is not RunActionCommittedSpawnObservation
            or observation.state
            not in {
                RunActionCommittedSpawnState.INERT_CONTINUABLE,
                RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
                RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
                RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
                RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE,
            }
            or (
                observation.state
                in {
                    RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
                    RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE,
                }
                and (
                    query.workload_release_adoption is not None
                    or query.timeout_directive_publication is not None
                    or query.control_directory_topology
                    is not RunActionControlDirectoryTopology.EMPTY
                )
            )
            or (
                pre_release_credential_observation is not None
                and type(pre_release_credential_observation)
                is not RunActionPreReleaseCredentialObservation
            )
            or (pre_release_credential_observation is not None)
            != credential_observation_required
            or (
                pre_release_credential_observation is not None
                and (
                    pre_release_credential_observation.activation_revalidation_receipt
                    != activation
                    or pre_release_credential_observation.state
                    is RunActionPreReleaseCredentialState.RENEWAL_PENDING
                )
            )
            or type(required_security_observation) is not SecurityDenylistObservation
            or required_security_observation.observation_id
            != reservation.frontier.security_observation_id
            or required_security_observation.matched_revocations
            or not hasattr(security_authority, "observe_exact_descendant_of")
            or type(credential_broker_registry) is not RunActionCredentialBrokerRegistry
            or type(release_clock) is not _SystemRunActionClock
            or _authority is not _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "committed continuation capability lacks exact authority"
            )
        release_receipt_size_bound_bytes = (
            workload_release_receipt_size_bound(
                prepared_execution=activation.prepared_execution,
                spawn_commit=activation.spawn_commit,
                required_security_observation=required_security_observation,
            )
            if release_bound_required
            else None
        )
        if (release_receipt_size_bound_bytes is not None) != release_bound_required or (
            release_receipt_size_bound_bytes is not None
            and (
                type(release_receipt_size_bound_bytes) is not int
                or release_receipt_size_bound_bytes <= 0
                or release_receipt_size_bound_bytes
                > activation.prepared_execution.preparation_claim.execution_policy.supervisor_limits.release_receipt_size_bytes
                or release_receipt_size_bound_bytes
                != workload_release_receipt_size_bound(
                    prepared_execution=activation.prepared_execution,
                    spawn_commit=activation.spawn_commit,
                    required_security_observation=(required_security_observation),
                )
            )
        ):
            raise RunActionRecoveryError(
                "committed continuation formal release envelope is invalid"
            )
        self._creation_process_id = os.getpid()
        private_query = deepcopy(query)
        private_observation = deepcopy(observation)
        private_pre_release_credential_observation = deepcopy(
            pre_release_credential_observation
        )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES[id(self)] = self
            _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES[self] = (
                _RunActionIssuedReleaseAuthorities(
                    query=private_query,
                    adapter_query=deepcopy(private_query),
                    observation=private_observation,
                    adapter_observation=deepcopy(private_observation),
                    pre_release_credential_observation=(
                        private_pre_release_credential_observation
                    ),
                    adapter_pre_release_credential_observation=deepcopy(
                        private_pre_release_credential_observation
                    ),
                    required_security_observation=required_security_observation,
                    security_authority=security_authority,
                    credential_broker_registry=credential_broker_registry,
                    clock=release_clock,
                    release_receipt_size_bound_bytes=(release_receipt_size_bound_bytes),
                    provider_termination_registration=(
                        _RunActionProviderTerminationRegistration()
                    ),
                    terminal_result_registration=(
                        _RunActionTerminalResultRegistration()
                    ),
                    capability_lifecycle=_RunActionContinuationCapabilityLifecycle(
                        owner_process_id=os.getpid(),
                        timeout_publication_state=(
                            "complete"
                            if type(query.timeout_directive_publication)
                            is RunActionTimeoutDirectivePublicationReceipt
                            else "ready"
                        ),
                        timeout_directive_publication=(
                            deepcopy(query.timeout_directive_publication)
                        ),
                    ),
                )
            )

    @property
    def _query(self) -> "RunActionCommittedSpawnQuery":
        authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
        if type(authorities) is not _RunActionIssuedReleaseAuthorities:
            raise RunActionRecoveryError(
                "committed continuation query authority is unavailable"
            )
        if authorities.adapter_query != authorities.query:
            raise RunActionRecoveryError(
                "committed continuation query changed during adapter invocation"
            )
        return deepcopy(authorities.query)

    @property
    def _observation(self) -> RunActionCommittedSpawnObservation:
        authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
        if type(authorities) is not _RunActionIssuedReleaseAuthorities:
            raise RunActionRecoveryError(
                "committed continuation observation authority is unavailable"
            )
        if authorities.adapter_observation != authorities.observation:
            raise RunActionRecoveryError(
                "committed continuation observation changed during adapter invocation"
            )
        return deepcopy(authorities.observation)

    @property
    def _pre_release_credential_observation(
        self,
    ) -> RunActionPreReleaseCredentialObservation | None:
        authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
        if type(authorities) is not _RunActionIssuedReleaseAuthorities:
            raise RunActionRecoveryError(
                "committed continuation credential authority is unavailable"
            )
        if (
            authorities.adapter_pre_release_credential_observation
            != authorities.pre_release_credential_observation
        ):
            raise RunActionRecoveryError(
                "committed continuation credential changed during adapter invocation"
            )
        return deepcopy(authorities.pre_release_credential_observation)

    @property
    def _provider_termination_state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).provider_termination_registration.state

    @property
    def _provider_termination_receipt(
        self,
    ) -> RunActionProviderTerminationReceipt | None:
        return deepcopy(
            _issued_continuation_authorities(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).provider_termination_registration.receipt
        )

    @property
    def _provider_termination_publication_fence_handed_off(self) -> bool:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).provider_termination_registration.publication_fence_handed_off

    @property
    def _terminal_observation(self) -> RunActionTerminalObservation | None:
        return deepcopy(
            _issued_continuation_authorities(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).terminal_result_registration.terminal_observation
        )

    @property
    def _captured_result(self) -> RunActionProviderResult | None:
        return deepcopy(
            _issued_continuation_authorities(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).terminal_result_registration.captured_result
        )

    @property
    def _owner_process_id(self) -> int:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.owner_process_id

    @property
    def _invoking_thread_id(self) -> int | None:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.invoking_thread_id

    @property
    def _state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.state

    @property
    def _start_state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.start_state

    @property
    def _credential_retirement_state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.credential_retirement_state

    @property
    def _started_running_observation(
        self,
    ) -> RunActionBarrierRunningContainerObservation | None:
        return deepcopy(
            _issued_continuation_authorities(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).capability_lifecycle.started_running_observation
        )

    @property
    def _release_publication_state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.release_publication_state

    @property
    def _timeout_publication_state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.timeout_publication_state

    @property
    def _timeout_directive_publication(
        self,
    ) -> RunActionTimeoutDirectivePublicationReceipt | None:
        return deepcopy(
            _issued_continuation_authorities(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).capability_lifecycle.timeout_directive_publication
        )

    @property
    def _timeout_containment_state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.timeout_containment_state

    @property
    def _timeout_containment_result(
        self,
    ) -> RunActionTimeoutContainmentResult | None:
        return deepcopy(
            _issued_continuation_authorities(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).capability_lifecycle.timeout_containment_result
        )

    @property
    def _terminal_inspection_state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.terminal_inspection_state

    @property
    def _result_capture_state(self) -> str:
        return _issued_continuation_authorities(
            self,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
        ).capability_lifecycle.result_capture_state

    @property
    def activation_event(self) -> RunActionExecutionEvent:
        authorities = self._require_active_invocation()
        return authorities.adapter_query.activation_event

    @property
    def query(self) -> "RunActionCommittedSpawnQuery":
        authorities = self._require_active_invocation()
        return authorities.adapter_query

    @property
    def activation_revalidation_receipt(
        self,
    ) -> RunActionActivationRevalidationReceipt:
        authorities = self._require_active_invocation()
        return authorities.adapter_query.activation_revalidation_receipt

    @property
    def prepared_execution(self) -> RunActionPreparedExecution:
        return self.activation_revalidation_receipt.prepared_execution

    @property
    def spawn_commit(self) -> RunActionSpawnCommit:
        return self.activation_revalidation_receipt.spawn_commit

    @property
    def observation(self) -> RunActionCommittedSpawnObservation:
        authorities = self._require_active_invocation()
        return authorities.adapter_observation

    @property
    def workload_release_adoption(
        self,
    ) -> RunActionWorkloadReleaseAdoption | None:
        authorities = self._require_active_invocation()
        return authorities.adapter_query.workload_release_adoption

    def _pre_release_credentials_allow_workload(self) -> bool:
        authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
        if type(authorities) is not _RunActionIssuedReleaseAuthorities:
            return False
        query = authorities.query
        observation = authorities.pre_release_credential_observation
        if query.credential_retirement_intent is not None:
            return False
        policy = (
            query.prepared_execution.preparation_claim.execution_policy.credential_policy
        )
        if policy.mode is RunActionCredentialMode.NONE:
            return observation is None
        return (
            type(observation) is RunActionPreReleaseCredentialObservation
            and observation.state is RunActionPreReleaseCredentialState.VALID
        )

    def _take_credential_retirement_authority(
        self,
        observation_token: str,
        control_inspection: RunActionTimeoutInspectionLease,
        *,
        _authority: object,
    ) -> tuple[
        "RunActionCommittedSpawnQuery",
        str,
        RunActionCredentialRetirementIntent,
    ]:
        """Consume exact durable expiry authority for one physical retirement."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            query = (
                authorities.query
                if type(authorities) is _RunActionIssuedReleaseAuthorities
                else None
            )
            observation = (
                authorities.observation
                if type(authorities) is _RunActionIssuedReleaseAuthorities
                else None
            )
            retirement_intent = (
                query.credential_retirement_intent
                if type(query) is RunActionCommittedSpawnQuery
                else None
            )
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or self._query != query
                or self._observation != observation
                or self._pre_release_credential_observation is not None
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or observation.state
                not in {
                    RunActionCommittedSpawnState.INERT_CONTINUABLE,
                    RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
                }
                or type(observation_token) is not str
                or observation_token != observation.observation_token
                or type(retirement_intent) is not RunActionCredentialRetirementIntent
                or retirement_intent.activation_event_id
                != query.activation_event.event_id
                or self._credential_retirement_state != "ready"
                or self._start_state != "ready"
                or self._started_running_observation is not None
                or self._release_publication_state != "ready"
                or self._timeout_publication_state != "ready"
                or self._timeout_directive_publication is not None
                or self._timeout_containment_state != "ready"
                or self._timeout_containment_result is not None
                or self._terminal_inspection_state != "ready"
                or self._terminal_observation is not None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
                or self._provider_termination_state != "ready"
                or self._provider_termination_receipt is not None
                or query.control_directory_topology
                is not RunActionControlDirectoryTopology.EMPTY
                or query.workload_release_adoption is not None
                or query.timeout_directive_publication is not None
                or type(control_inspection) is not RunActionTimeoutInspectionLease
                or control_inspection.activation_event != query.activation_event
                or control_inspection.topology
                is not RunActionControlDirectoryTopology.EMPTY
                or control_inspection.workload_release_adoption is not None
                or control_inspection.timeout_directive_publication is not None
                or _authority is not _RUN_ACTION_CREDENTIAL_RETIREMENT_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "credential retirement lacks exact durable expiry authority"
                )
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).credential_retirement_state = "retiring"
        return query, observation_token, retirement_intent

    def _complete_credential_retirement(
        self,
        observation_token: str,
        retirement_intent: RunActionCredentialRetirementIntent,
        *,
        _authority: object,
    ) -> None:
        """Record only that one exact physical retirement command was attempted."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or self._query != authorities.query
                or self._observation != authorities.observation
                or authorities.query.credential_retirement_intent != retirement_intent
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or observation_token != authorities.observation.observation_token
                or self._credential_retirement_state != "retiring"
                or _authority is not _RUN_ACTION_CREDENTIAL_RETIREMENT_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "credential retirement completion changed authority"
                )
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).credential_retirement_state = "complete"

    def _require_current_release_receipt_size_bound(self) -> int:
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or authorities.release_receipt_size_bound_bytes is None
            ):
                raise RunActionRecoveryError(
                    "committed continuation lacks its formal release envelope"
                )
            required_security_observation = authorities.required_security_observation
            expected_bound = authorities.release_receipt_size_bound_bytes
            activation = self._query.activation_revalidation_receipt
        current_bound = workload_release_receipt_size_bound(
            prepared_execution=activation.prepared_execution,
            spawn_commit=activation.spawn_commit,
            required_security_observation=required_security_observation,
        )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
                is not authorities
                or current_bound != expected_bound
            ):
                raise RunActionRecoveryError(
                    "committed continuation release envelope changed"
                )
        return current_bound

    def _take_start_authority(
        self,
        observation_token: str,
        *,
        _authority: object,
    ) -> tuple["RunActionCommittedSpawnQuery", str]:
        """Consume the trusted start leaf's exact event-5 inert seal."""

        release_receipt_size_bound_bytes = (
            self._require_current_release_receipt_size_bound()
        )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            query = self._query
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or authorities.release_receipt_size_bound_bytes
                != release_receipt_size_bound_bytes
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._observation.state
                is not RunActionCommittedSpawnState.INERT_CONTINUABLE
                or type(observation_token) is not str
                or observation_token != self._observation.observation_token
                or self._start_state != "ready"
                or self._credential_retirement_state != "ready"
                or not self._pre_release_credentials_allow_workload()
                or self._started_running_observation is not None
                or self._release_publication_state != "ready"
                or self._timeout_publication_state != "ready"
                or self._timeout_directive_publication is not None
                or self._timeout_containment_state != "ready"
                or self._timeout_containment_result is not None
                or self._terminal_inspection_state != "ready"
                or self._terminal_observation is not None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
                or self._provider_termination_state != "ready"
                or self._provider_termination_receipt is not None
                or query.control_directory_topology
                is not RunActionControlDirectoryTopology.EMPTY
                or query.workload_release_adoption is not None
                or query.timeout_directive_publication is not None
                or _authority is not _RUN_ACTION_START_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "container start lacks exact live event-5 authority"
                )
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).start_state = "validating_credentials"
            credential_broker_registry = authorities.credential_broker_registry
            clock = authorities.clock
        credential_anchor = _read_positive_release_clock(
            clock.realtime_nanoseconds(),
            "pre-start credential anchor",
        )
        _observe_activation_credential_validity(
            query.activation_revalidation_receipt,
            credential_broker_registry,
            clock,
            credential_anchor,
        )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
                is not authorities
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._start_state != "validating_credentials"
            ):
                raise RunActionRecoveryError(
                    "container start credential authority changed"
                )
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).start_state = "starting"
        return query, observation_token

    def _complete_start(
        self,
        running_observation: RunActionBarrierRunningContainerObservation,
        observation_token: str,
        *,
        _authority: object,
    ) -> None:
        """Register the trusted start leaf's stable blocked-wrapper occurrence."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            query = self._query
            prepared = query.prepared_execution
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._observation.state
                is not RunActionCommittedSpawnState.INERT_CONTINUABLE
                or observation_token != self._observation.observation_token
                or self._start_state != "starting"
                or self._started_running_observation is not None
                or type(running_observation)
                is not RunActionBarrierRunningContainerObservation
                or running_observation.container_id
                != query.spawn_commit.provider_execution_id
                or running_observation.observed_inspect_projection
                != prepared.inert_container_evidence.issued_create_projection
                or _authority is not _RUN_ACTION_START_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "container start completion lacks its exact blocked occurrence"
                )
            lifecycle = _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            )
            lifecycle.started_running_observation = deepcopy(running_observation)
            lifecycle.start_state = "complete"

    def _begin_release_publication(
        self,
        resolved_workload_observation: RunActionResolvedWorkloadObservation,
        *,
        _authority: object,
    ) -> "_RunActionReleasePublicationAuthorization":
        """Open the trusted publisher's sole event-5 release authorization."""

        release_receipt_size_bound_bytes = (
            self._require_current_release_receipt_size_bound()
        )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            release_authority = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(
                self
            )
            if (
                issued is not self
                or release_authority is None
                or release_authority.release_receipt_size_bound_bytes
                != release_receipt_size_bound_bytes
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._release_publication_state != "ready"
                or self._credential_retirement_state != "ready"
                or not self._pre_release_credentials_allow_workload()
                or self._timeout_publication_state != "ready"
                or self._timeout_directive_publication is not None
                or self._timeout_containment_state != "ready"
                or self._timeout_containment_result is not None
                or self._provider_termination_state != "ready"
                or self._provider_termination_receipt is not None
                or self._observation.state
                is not RunActionCommittedSpawnState.RUNNING_CONTINUABLE
                or type(resolved_workload_observation)
                is not RunActionResolvedWorkloadObservation
                or resolved_workload_observation.activation_revalidation_receipt
                != self._query.activation_revalidation_receipt
                or resolved_workload_observation.running_container_observation.complete_inspection_digest
                != self._observation.observation_token
                or resolved_workload_observation.control_directory_topology
                is not RunActionControlDirectoryTopology.EMPTY
                or self._query.control_directory_topology
                is not RunActionControlDirectoryTopology.EMPTY
                or self._query.workload_release_adoption is not None
                or self._query.timeout_directive_publication is not None
                or _authority is not _RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "workload release publication lacks exact live authority"
                )
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).release_publication_state = "preparing"
        return _RunActionReleasePublicationAuthorization(
            capability=self,
            resolved_workload_observation=resolved_workload_observation,
            required_security_observation=(
                release_authority.required_security_observation
            ),
            release_receipt_size_bound_bytes=release_receipt_size_bound_bytes,
            _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
        )

    def _begin_timeout_publication(
        self,
        control_inspection: RunActionTimeoutInspectionLease,
        *,
        _authority: object,
    ) -> "_RunActionTimeoutPublicationAuthorization | None":
        """Open the trusted publisher's sole released-to-timeout transition."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            query = self._query
            adoption = query.workload_release_adoption
            if (
                issued is not self
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._observation.state
                is not RunActionCommittedSpawnState.RUNNING_CONTINUABLE
                or self._release_publication_state != "ready"
                or self._timeout_publication_state != "ready"
                or self._timeout_directive_publication is not None
                or self._timeout_containment_state != "ready"
                or self._timeout_containment_result is not None
                or self._terminal_inspection_state != "ready"
                or self._terminal_observation is not None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
                or self._provider_termination_state != "ready"
                or self._provider_termination_receipt is not None
                or query.control_directory_topology
                is not RunActionControlDirectoryTopology.RELEASED
                or type(adoption) is not RunActionWorkloadReleaseAdoption
                or adoption.workload_release_receipt.resolved_workload_observation.running_container_observation.complete_inspection_digest
                != self._observation.observation_token
                or query.timeout_directive_publication is not None
                or type(control_inspection) is not RunActionTimeoutInspectionLease
                or control_inspection.topology
                is not RunActionControlDirectoryTopology.RELEASED
                or control_inspection.workload_release_adoption != adoption
                or control_inspection.timeout_directive_publication is not None
                or _authority is not _RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "timeout publication lacks exact live released authority"
                )
            clock = authorities.clock
            deadline = (
                adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
            )
        observed_before = _read_positive_release_clock(
            clock.boottime_nanoseconds(),
            "timeout observation start",
        )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._timeout_publication_state != "ready"
                or self._timeout_directive_publication is not None
            ):
                raise RunActionRecoveryError(
                    "timeout publication authority changed during deadline observation"
                )
            if observed_before < deadline:
                _continuation_lifecycle(
                    self,
                    _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
                ).timeout_publication_state = "checked_not_due"
                return None
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).timeout_publication_state = "preparing"
        return _RunActionTimeoutPublicationAuthorization(
            capability=self,
            observed_before_boottime_nanoseconds=observed_before,
            _authority=_RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY,
        )

    def _begin_timeout_containment(
        self,
        control_inspection: RunActionTimeoutInspectionLease,
        *,
        _authority: object,
    ) -> "_RunActionTimeoutContainmentAuthorization":
        """Open one trusted at-least-once signal attempt for a timed-out run."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            query = self._query
            adoption = query.workload_release_adoption
            publication = query.timeout_directive_publication
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or type(_ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self))
                is not _RunActionIssuedReleaseAuthorities
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._observation.state
                is not RunActionCommittedSpawnState.RUNNING_CONTINUABLE
                or self._release_publication_state != "ready"
                or self._timeout_publication_state != "complete"
                or self._timeout_directive_publication != publication
                or type(publication) is not RunActionTimeoutDirectivePublicationReceipt
                or self._timeout_containment_state != "ready"
                or self._timeout_containment_result is not None
                or self._terminal_inspection_state != "ready"
                or self._terminal_observation is not None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
                or self._provider_termination_state != "ready"
                or self._provider_termination_receipt is not None
                or query.control_directory_topology
                is not RunActionControlDirectoryTopology.TIMED_OUT
                or type(adoption) is not RunActionWorkloadReleaseAdoption
                or type(control_inspection) is not RunActionTimeoutInspectionLease
                or control_inspection.topology
                is not RunActionControlDirectoryTopology.TIMED_OUT
                or control_inspection.workload_release_adoption != adoption
                or control_inspection.timeout_directive_publication != publication
                or _authority is not _RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "timeout containment lacks exact live timed-out authority"
                )
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).timeout_containment_state = "preparing"
        return _RunActionTimeoutContainmentAuthorization(
            capability=self,
            _authority=_RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY,
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
                or (
                    (
                        self._query.control_directory_topology
                        is RunActionControlDirectoryTopology.RELEASED
                        and (
                            self._timeout_publication_state != "ready"
                            or self._timeout_directive_publication is not None
                        )
                    )
                    or (
                        self._query.control_directory_topology
                        is RunActionControlDirectoryTopology.TIMED_OUT
                        and (
                            self._timeout_publication_state != "complete"
                            or self._timeout_directive_publication
                            != self._query.timeout_directive_publication
                        )
                    )
                )
                or self._observation.state
                is not RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
                or type(observation_token) is not str
                or self._query.workload_release_adoption is None
                or self._query.control_directory_topology
                not in {
                    RunActionControlDirectoryTopology.RELEASED,
                    RunActionControlDirectoryTopology.TIMED_OUT,
                }
                or _authority is not _RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "terminal reinspection lacks exact live continuation authority"
                )
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).terminal_inspection_state = "inspecting"
        return self._query, observation_token

    def _complete_terminal_inspection(
        self,
        terminal_observation: RunActionTerminalObservation,
        *,
        _authority: object,
    ) -> None:
        """Register the trusted terminal leaf's exact completed observation."""

        if type(terminal_observation) is not RunActionTerminalObservation:
            raise RunActionRecoveryError(
                "terminal reinspection completion lacks exact live authority"
            )
        canonical_terminal_observation = RunActionTerminalObservation.from_json_bytes(
            terminal_observation.to_json_bytes()
        )
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
                or canonical_terminal_observation.complete_inspection_digest
                != self._observation.observation_token
                or canonical_terminal_observation.activation_revalidation_receipt_id
                != activation.activation_revalidation_receipt_id
                or canonical_terminal_observation.workload_release_adoption_id
                != adoption.workload_release_adoption_id
                or canonical_terminal_observation.prepared_execution_id
                != prepared.prepared_execution_id
                or canonical_terminal_observation.spawn_commit_id
                != spawn.spawn_commit_id
                or canonical_terminal_observation.provider_execution_id
                != spawn.provider_execution_id
                or canonical_terminal_observation.runtime_volume_authority_id
                != prepared.runtime_volume_authority.runtime_volume_authority_id
                or canonical_terminal_observation.generation_nonce
                != prepared.runtime_volume_authority.generation_nonce
                or canonical_terminal_observation.observed_inspect_projection
                != prepared.inert_container_evidence.issued_create_projection
                or _authority is not _RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "terminal reinspection completion lacks exact live authority"
                )
            _terminal_result_registration(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).terminal_observation = canonical_terminal_observation
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).terminal_inspection_state = "complete"

    def _take_result_capture_authority(
        self,
        *,
        _authority: object,
    ) -> tuple["RunActionCommittedSpawnQuery", RunActionTerminalObservation]:
        """Consume the trusted result leaf's sole descriptor capture."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
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
                or self._query.control_directory_topology
                is not RunActionControlDirectoryTopology.RELEASED
                or self._query.timeout_directive_publication is not None
                or self._timeout_publication_state != "ready"
                or self._timeout_directive_publication is not None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
                or self._provider_termination_state != "ready"
                or self._provider_termination_receipt is not None
                or type(_ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self))
                is not _RunActionIssuedReleaseAuthorities
                or _authority is not _RUN_ACTION_RESULT_CAPTURE_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "result capture lacks exact live terminal authority"
                )
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).result_capture_state = "capturing"
            _provider_termination_registration(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).state = "blocked_by_result_capture"
        return self._query, terminal

    def _complete_result_capture(
        self,
        result: RunActionProviderResult,
        *,
        _authority: object,
    ) -> None:
        """Register the trusted descriptor leaf's exact captured result."""

        if type(result) is not RunActionProviderResult:
            raise RunActionRecoveryError(
                "result capture completion lacks exact live authority"
            )
        canonical_result = RunActionProviderResult(
            terminal_observation=RunActionTerminalObservation.from_json_bytes(
                result.terminal_observation.to_json_bytes()
            ),
            result_capture_receipt=RunActionResultCaptureReceipt.from_json_bytes(
                result.result_capture_receipt.to_json_bytes()
            ),
            result_payload=bytes(result.result_payload),
        )
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
                or self._provider_termination_state != "blocked_by_result_capture"
                or self._provider_termination_receipt is not None
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or adoption is None
                or query.control_directory_topology
                is not RunActionControlDirectoryTopology.RELEASED
                or query.timeout_directive_publication is not None
                or self._timeout_publication_state != "ready"
                or self._timeout_directive_publication is not None
                or type(terminal) is not RunActionTerminalObservation
                or canonical_result.terminal_observation != terminal
                or not run_action_terminal_result_evidence_matches(
                    terminal,
                    canonical_result.result_capture_receipt,
                    query.activation_revalidation_receipt,
                    adoption,
                )
                or _authority is not _RUN_ACTION_RESULT_CAPTURE_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "result capture completion lacks exact live authority"
                )
            _terminal_result_registration(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).captured_result = canonical_result
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).result_capture_state = "complete"

    def _take_provider_termination_authority(
        self,
        *,
        _authority: object,
    ) -> tuple[
        "RunActionCommittedSpawnQuery",
        RunActionTerminalObservation | None,
        str | None,
    ]:
        """Consume the trusted termination leaf's sole receipt registration."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            query = self._query
            observation_state = self._observation.state
            terminal = self._terminal_observation
            pre_release_observation_token = (
                self._observation.observation_token
                if observation_state
                in {
                    RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
                    RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE,
                }
                else None
            )
            released_terminal_ready = (
                observation_state is RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
                and self._terminal_inspection_state == "complete"
                and type(terminal) is RunActionTerminalObservation
                and query.workload_release_adoption is not None
                and query.control_directory_topology
                in {
                    RunActionControlDirectoryTopology.RELEASED,
                    RunActionControlDirectoryTopology.TIMED_OUT,
                }
                and (
                    (
                        query.control_directory_topology
                        is RunActionControlDirectoryTopology.RELEASED
                        and self._timeout_publication_state == "ready"
                        and self._timeout_directive_publication is None
                    )
                    or (
                        query.control_directory_topology
                        is RunActionControlDirectoryTopology.TIMED_OUT
                        and self._timeout_publication_state == "complete"
                        and self._timeout_directive_publication
                        == query.timeout_directive_publication
                    )
                )
                and pre_release_observation_token is None
            )
            pre_release_termination_ready = (
                observation_state
                in {
                    RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
                    RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE,
                }
                and self._terminal_inspection_state == "ready"
                and terminal is None
                and query.workload_release_adoption is None
                and query.control_directory_topology
                is RunActionControlDirectoryTopology.EMPTY
                and query.timeout_directive_publication is None
                and self._timeout_publication_state == "ready"
                and self._timeout_directive_publication is None
                and type(pre_release_observation_token) is str
                and _SHA256_DIGEST_PATTERN.fullmatch(pre_release_observation_token)
                is not None
            )
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._provider_termination_state != "ready"
                or self._provider_termination_receipt is not None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
                or self._release_publication_state != "ready"
                or self._timeout_publication_state
                not in {
                    "ready",
                    "complete",
                }
                or not (released_terminal_ready or pre_release_termination_ready)
                or _authority is not _RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "provider termination registration lacks exact live authority"
                )
            _provider_termination_registration(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).state = "registering"
            _continuation_lifecycle(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).result_capture_state = "blocked_by_provider_termination"
            if pre_release_termination_ready:
                _continuation_lifecycle(
                    self,
                    _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
                ).terminal_inspection_state = "blocked_by_provider_termination"
        return query, terminal, pre_release_observation_token

    def _complete_provider_termination(
        self,
        receipt: RunActionProviderTerminationReceipt,
        publication_fence: RunActionProviderTerminationPublicationFence | None = None,
        *,
        _authority: object,
    ) -> None:
        """Register the trusted termination leaf's exact retained evidence."""

        if type(publication_fence) is RunActionProviderTerminationPublicationFence:
            publication_fence.require_current()
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if type(receipt) is not RunActionProviderTerminationReceipt:
                raise RunActionRecoveryError(
                    "provider termination completion lacks exact live authority"
                )
            query = self._query
            observation_state = self._observation.state
            retained_terminal = self._terminal_observation
            released_terminal_matches = (
                observation_state is RunActionCommittedSpawnState.TERMINAL_CONTINUABLE
                and self._terminal_inspection_state == "complete"
                and type(retained_terminal) is RunActionTerminalObservation
                and receipt.reason
                not in {
                    RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
                    RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL,
                    RunActionProviderTerminationReason.CREDENTIAL_EXPIRED,
                }
                and receipt.workload_release_adoption == query.workload_release_adoption
                and receipt.terminal_observation == retained_terminal
                and (
                    (
                        receipt.reason is RunActionProviderTerminationReason.TIMEOUT
                        and query.control_directory_topology
                        is RunActionControlDirectoryTopology.TIMED_OUT
                        and receipt.timeout_directive_publication
                        == query.timeout_directive_publication
                        and self._timeout_publication_state == "complete"
                        and self._timeout_directive_publication
                        == query.timeout_directive_publication
                    )
                    or (
                        receipt.reason is not RunActionProviderTerminationReason.TIMEOUT
                        and query.control_directory_topology
                        is RunActionControlDirectoryTopology.RELEASED
                        and query.timeout_directive_publication is None
                        and self._timeout_publication_state == "ready"
                        and self._timeout_directive_publication is None
                    )
                )
            )
            loss = receipt.pre_release_main_loss_observation
            expected_loss_reason = (
                RunActionProviderTerminationReason.CREDENTIAL_EXPIRED
                if query.credential_retirement_intent is not None
                else RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
            )
            pre_release_loss_matches = (
                observation_state
                is RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE
                and self._terminal_inspection_state == "blocked_by_provider_termination"
                and retained_terminal is None
                and receipt.reason is expected_loss_reason
                and receipt.credential_retirement_intent
                == query.credential_retirement_intent
                and query.control_directory_topology
                is RunActionControlDirectoryTopology.EMPTY
                and query.workload_release_adoption is None
                and query.timeout_directive_publication is None
                and self._timeout_publication_state == "ready"
                and self._timeout_directive_publication is None
                and loss is not None
                and run_action_pre_release_main_loss_observation_token(loss)
                == self._observation.observation_token
            )
            pre_release_terminal = receipt.terminal_observation
            expected_terminal_reason = (
                RunActionProviderTerminationReason.CREDENTIAL_EXPIRED
                if query.credential_retirement_intent is not None
                else RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL
            )
            pre_release_terminal_matches = (
                observation_state
                is RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE
                and self._terminal_inspection_state == "blocked_by_provider_termination"
                and retained_terminal is None
                and receipt.reason is expected_terminal_reason
                and receipt.credential_retirement_intent
                == query.credential_retirement_intent
                and query.control_directory_topology
                is RunActionControlDirectoryTopology.EMPTY
                and query.workload_release_adoption is None
                and query.timeout_directive_publication is None
                and self._timeout_publication_state == "ready"
                and self._timeout_directive_publication is None
                and type(pre_release_terminal)
                is RunActionPreReleaseMainTerminalObservation
                and run_action_pre_release_main_terminal_observation_token(
                    pre_release_terminal
                )
                == self._observation.observation_token
            )
            pre_release_termination_matches = (
                pre_release_loss_matches or pre_release_terminal_matches
            )
            registration = _provider_termination_registration(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            )
            if (
                _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self)) is not self
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._provider_termination_state != "registering"
                or self._provider_termination_receipt is not None
                or registration.publication_fence is not None
                or self._provider_termination_publication_fence_handed_off
                or self._result_capture_state != "blocked_by_provider_termination"
                or self._captured_result is not None
                or not provider_termination_matches_durable_activation(
                    receipt,
                    query.activation_event.event_id,
                    query.preparation_allocation,
                    query.activation_revalidation_receipt,
                )
                or not (released_terminal_matches or pre_release_termination_matches)
                or (
                    pre_release_termination_matches
                    != (
                        type(publication_fence)
                        is RunActionProviderTerminationPublicationFence
                    )
                )
                or (released_terminal_matches and publication_fence is not None)
                or _authority is not _RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "provider termination completion lacks exact live authority"
                )
            registration.receipt = deepcopy(receipt)
            registration.publication_fence = publication_fence
            registration.state = "complete"

    def _invoke_once(
        self,
        execution_adapter: "RunActionExecutionAdapter",
    ) -> RunActionContinuationOutcome:
        with self._begin_invocation():
            outcome = execution_adapter.continue_committed_once(self)
            RunActionCommittedContinuationCapability._require_continuation_outcome_authority(
                self,
                outcome,
            )
            registration = _provider_termination_registration(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            )
            terminal_result_registration = _terminal_result_registration(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            )
            return RunActionContinuationOutcome(
                state=outcome.state,
                result=deepcopy(
                    terminal_result_registration.captured_result
                    if outcome.state is RunActionContinuationState.RESULT_CAPTURED
                    else outcome.result
                ),
                provider_termination_receipt=deepcopy(
                    registration.receipt
                    if outcome.state is RunActionContinuationState.PROVIDER_TERMINATED
                    else outcome.provider_termination_receipt
                ),
                timeout_directive_publication=deepcopy(
                    _continuation_lifecycle(
                        self,
                        _authority=(_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY),
                    ).timeout_directive_publication
                    if outcome.state is RunActionContinuationState.TIMEOUT_PUBLISHED
                    else outcome.timeout_directive_publication
                ),
                provider_termination_publication_fence=(
                    registration.publication_fence
                    if outcome.state is RunActionContinuationState.PROVIDER_TERMINATED
                    else None
                ),
            )

    def _require_continuation_outcome_authority(
        self,
        outcome: RunActionContinuationOutcome,
    ) -> None:
        """Bind every continuation branch to its exact consumed authorities."""

        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            termination_registration = _provider_termination_registration(
                self,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            )
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
                    self._start_state != "ready"
                    or self._credential_retirement_state != "ready"
                    or self._started_running_observation is not None
                    or outcome.state
                    not in {
                        RunActionContinuationState.PENDING,
                        RunActionContinuationState.RESULT_CAPTURED,
                        RunActionContinuationState.PROVIDER_TERMINATED,
                    }
                    or self._terminal_inspection_state != "complete"
                    or self._timeout_containment_state != "ready"
                    or self._timeout_containment_result is not None
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
                    (
                        outcome.state is RunActionContinuationState.PENDING
                        and (
                            self._result_capture_state != "ready"
                            or self._captured_result is not None
                            or self._provider_termination_state != "ready"
                            or self._provider_termination_receipt is not None
                        )
                    )
                    or (
                        outcome.state is RunActionContinuationState.RESULT_CAPTURED
                        and (
                            self._result_capture_state != "complete"
                            or type(self._captured_result)
                            is not RunActionProviderResult
                            or outcome.result != self._captured_result
                            or self._provider_termination_state
                            != "blocked_by_result_capture"
                            or self._provider_termination_receipt is not None
                        )
                    )
                    or (
                        outcome.state is RunActionContinuationState.PROVIDER_TERMINATED
                        and (
                            self._result_capture_state
                            != "blocked_by_provider_termination"
                            or self._captured_result is not None
                            or self._provider_termination_state != "complete"
                            or type(self._provider_termination_receipt)
                            is not RunActionProviderTerminationReceipt
                            or outcome.provider_termination_receipt
                            != self._provider_termination_receipt
                            or outcome.provider_termination_publication_fence
                            is not termination_registration.publication_fence
                        )
                    )
                ):
                    raise RunActionRecoveryError(
                        "terminal continuation lacks its trusted outcome registration"
                    )
            elif self._observation.state in {
                RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
                RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE,
            }:
                if (
                    self._start_state != "ready"
                    or self._credential_retirement_state != "ready"
                    or self._started_running_observation is not None
                    or self._release_publication_state != "ready"
                    or self._timeout_publication_state != "ready"
                    or self._timeout_directive_publication is not None
                    or self._timeout_containment_state != "ready"
                    or self._timeout_containment_result is not None
                    or self._terminal_observation is not None
                    or self._captured_result is not None
                    or (
                        outcome.state is RunActionContinuationState.PENDING
                        and (
                            self._terminal_inspection_state != "ready"
                            or self._result_capture_state != "ready"
                            or self._provider_termination_state != "ready"
                            or self._provider_termination_receipt is not None
                        )
                    )
                    or (
                        outcome.state is RunActionContinuationState.PROVIDER_TERMINATED
                        and (
                            self._terminal_inspection_state
                            != "blocked_by_provider_termination"
                            or self._result_capture_state
                            != "blocked_by_provider_termination"
                            or self._provider_termination_state != "complete"
                            or type(self._provider_termination_receipt)
                            is not RunActionProviderTerminationReceipt
                            or outcome.provider_termination_receipt
                            != self._provider_termination_receipt
                            or outcome.provider_termination_publication_fence
                            is not termination_registration.publication_fence
                        )
                    )
                    or outcome.state
                    not in {
                        RunActionContinuationState.PENDING,
                        RunActionContinuationState.PROVIDER_TERMINATED,
                    }
                ):
                    raise RunActionRecoveryError(
                        "pre-release continuation lacks its trusted termination"
                    )
            elif (
                self._observation.state
                is RunActionCommittedSpawnState.INERT_CONTINUABLE
            ):
                credential_expired = (
                    type(self._query.credential_retirement_intent)
                    is RunActionCredentialRetirementIntent
                )
                started = (
                    not credential_expired
                    and self._start_state == "complete"
                    and type(self._started_running_observation)
                    is RunActionBarrierRunningContainerObservation
                    and self._credential_retirement_state == "ready"
                )
                retired = (
                    credential_expired
                    and self._start_state == "ready"
                    and self._started_running_observation is None
                    and self._credential_retirement_state == "complete"
                )
                if (
                    outcome.state is not RunActionContinuationState.PENDING
                    or not (started or retired)
                    or self._release_publication_state != "ready"
                    or self._timeout_publication_state != "ready"
                    or self._timeout_directive_publication is not None
                    or self._timeout_containment_state != "ready"
                    or self._timeout_containment_result is not None
                    or self._terminal_inspection_state != "ready"
                    or self._terminal_observation is not None
                    or self._result_capture_state != "ready"
                    or self._captured_result is not None
                    or self._provider_termination_state != "ready"
                    or self._provider_termination_receipt is not None
                ):
                    raise RunActionRecoveryError(
                        "inert continuation lacks exact start authority"
                    )
            elif (
                self._observation.state
                is RunActionCommittedSpawnState.RUNNING_CONTINUABLE
            ):
                topology = self._query.control_directory_topology
                credential_expired = (
                    type(self._query.credential_retirement_intent)
                    is RunActionCredentialRetirementIntent
                )
                empty_pending = (
                    topology is RunActionControlDirectoryTopology.EMPTY
                    and outcome.state is RunActionContinuationState.PENDING
                    and (
                        (
                            credential_expired
                            and self._credential_retirement_state == "complete"
                            and self._release_publication_state == "ready"
                        )
                        or (
                            not credential_expired
                            and self._credential_retirement_state == "ready"
                            and self._release_publication_state in {"ready", "spent"}
                        )
                    )
                    and self._timeout_publication_state == "ready"
                    and self._timeout_directive_publication is None
                    and self._query.workload_release_adoption is None
                    and self._query.timeout_directive_publication is None
                    and self._timeout_containment_state == "ready"
                    and self._timeout_containment_result is None
                )
                released_pending = (
                    topology is RunActionControlDirectoryTopology.RELEASED
                    and outcome.state is RunActionContinuationState.PENDING
                    and self._release_publication_state == "ready"
                    and self._timeout_publication_state
                    in {
                        "ready",
                        "checked_not_due",
                    }
                    and self._timeout_directive_publication is None
                    and type(self._query.workload_release_adoption)
                    is RunActionWorkloadReleaseAdoption
                    and self._query.timeout_directive_publication is None
                    and self._timeout_containment_state == "ready"
                    and self._timeout_containment_result is None
                    and self._credential_retirement_state == "ready"
                )
                released_timeout_published = (
                    topology is RunActionControlDirectoryTopology.RELEASED
                    and outcome.state is RunActionContinuationState.TIMEOUT_PUBLISHED
                    and self._release_publication_state == "ready"
                    and self._timeout_publication_state == "complete"
                    and type(self._timeout_directive_publication)
                    is RunActionTimeoutDirectivePublicationReceipt
                    and outcome.timeout_directive_publication
                    == self._timeout_directive_publication
                    and self._query.timeout_directive_publication is None
                    and self._timeout_containment_state == "ready"
                    and self._timeout_containment_result is None
                    and self._credential_retirement_state == "ready"
                )
                timed_out_pending = (
                    topology is RunActionControlDirectoryTopology.TIMED_OUT
                    and outcome.state is RunActionContinuationState.PENDING
                    and self._release_publication_state == "ready"
                    and self._timeout_publication_state == "complete"
                    and self._timeout_directive_publication
                    == self._query.timeout_directive_publication
                    and type(self._timeout_directive_publication)
                    is RunActionTimeoutDirectivePublicationReceipt
                    and self._timeout_containment_state == "complete"
                    and type(self._timeout_containment_result)
                    is RunActionTimeoutContainmentResult
                    and self._credential_retirement_state == "ready"
                )
                if (
                    self._start_state != "ready"
                    or self._started_running_observation is not None
                    or self._terminal_inspection_state != "ready"
                    or self._terminal_observation is not None
                    or self._result_capture_state != "ready"
                    or self._captured_result is not None
                    or self._provider_termination_state != "ready"
                    or self._provider_termination_receipt is not None
                    or not (
                        empty_pending
                        or released_pending
                        or released_timeout_published
                        or timed_out_pending
                    )
                ):
                    raise RunActionRecoveryError(
                        "nonterminal continuation consumed terminal outcome authority"
                    )
            elif (
                outcome.state is not RunActionContinuationState.PENDING
                or self._start_state != "ready"
                or self._started_running_observation is not None
                or self._timeout_publication_state != "ready"
                or self._timeout_directive_publication is not None
                or self._timeout_containment_state != "ready"
                or self._timeout_containment_result is not None
                or self._terminal_inspection_state != "ready"
                or self._terminal_observation is not None
                or self._result_capture_state != "ready"
                or self._captured_result is not None
                or self._provider_termination_state != "ready"
                or self._provider_termination_receipt is not None
            ):
                raise RunActionRecoveryError(
                    "nonterminal continuation consumed terminal outcome authority"
                )
            if (
                type(outcome.provider_termination_publication_fence)
                is RunActionProviderTerminationPublicationFence
            ):
                _provider_termination_registration(
                    self,
                    _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
                ).publication_fence_handed_off = True

    def _begin_invocation(self) -> "_RunActionCommittedContinuationInvocation":
        if self._creation_process_id != os.getpid():
            raise RunActionRecoveryError(
                "committed continuation capability is spent, cloned, or foreign"
            )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            if (
                issued is not self
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or self._query != authorities.query
                or self._observation != authorities.observation
                or self._pre_release_credential_observation
                != authorities.pre_release_credential_observation
                or self._owner_process_id != os.getpid()
                or self._state != "ready"
            ):
                raise RunActionRecoveryError(
                    "committed continuation capability is spent, cloned, or foreign"
                )
            lifecycle = authorities.capability_lifecycle
            lifecycle.state = "invoking"
            lifecycle.invoking_thread_id = get_ident()
        return _RunActionCommittedContinuationInvocation(
            self,
            owner_process_id=os.getpid(),
            invoking_thread_id=get_ident(),
        )

    def _require_active_invocation(self) -> _RunActionIssuedReleaseAuthorities:
        if self._creation_process_id != os.getpid():
            raise RunActionRecoveryError(
                "committed continuation capability is not in its one invocation"
            )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            if (
                issued is not self
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or self._query != authorities.query
                or self._observation != authorities.observation
                or self._pre_release_credential_observation
                != authorities.pre_release_credential_observation
                or self._owner_process_id != os.getpid()
                or self._state != "invoking"
                or self._invoking_thread_id != get_ident()
                or self._start_state == "starting"
                or self._credential_retirement_state == "retiring"
                or self._release_publication_state == "authorizing"
                or self._timeout_publication_state
                in {
                    "preparing",
                    "authorizing",
                    "published_awaiting_adoption",
                }
                or self._timeout_containment_state
                in {
                    "preparing",
                    "authorizing",
                }
            ):
                raise RunActionRecoveryError(
                    "committed continuation capability is not in its one invocation"
                )
            return authorities

    def _finish_invocation(
        self,
        *,
        owner_process_id: int,
        invoking_thread_id: int,
        _authority: object,
    ) -> None:
        if (
            owner_process_id != os.getpid()
            or invoking_thread_id != get_ident()
            or _authority is not _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "committed continuation capability invocation changed"
            )
        abandoned_publication_fence = None
        continuation_inputs_changed = False
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            issued = _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(self))
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(self)
            if (
                issued is not self
                or type(authorities) is not _RunActionIssuedReleaseAuthorities
                or authorities.capability_lifecycle.owner_process_id != owner_process_id
                or authorities.capability_lifecycle.state != "invoking"
                or authorities.capability_lifecycle.invoking_thread_id
                != invoking_thread_id
            ):
                raise RunActionRecoveryError(
                    "committed continuation capability invocation changed"
                )
            continuation_inputs_changed = (
                authorities.adapter_query != authorities.query
                or authorities.adapter_observation != authorities.observation
                or authorities.adapter_pre_release_credential_observation
                != authorities.pre_release_credential_observation
            )
            termination_registration = authorities.provider_termination_registration
            if not termination_registration.publication_fence_handed_off:
                abandoned_publication_fence = termination_registration.publication_fence
            termination_registration.publication_fence = None
            authorities.capability_lifecycle.state = "spent"
            authorities.capability_lifecycle.invoking_thread_id = None
            authorities.capability_lifecycle.start_state = "spent"
            authorities.capability_lifecycle.credential_retirement_state = "spent"
            authorities.capability_lifecycle.release_publication_state = "spent"
            authorities.capability_lifecycle.timeout_publication_state = "spent"
            authorities.capability_lifecycle.timeout_containment_state = "spent"
            authorities.capability_lifecycle.terminal_inspection_state = "spent"
            authorities.capability_lifecycle.result_capture_state = "spent"
            termination_registration.state = "spent"
            _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.pop(id(self))
            _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.pop(self, None)
        if abandoned_publication_fence is not None:
            abandoned_publication_fence.close()
        if continuation_inputs_changed:
            raise RunActionRecoveryError(
                "committed continuation inputs changed during adapter invocation"
            )


class _RunActionCommittedContinuationInvocation:
    """Burn one committed continuation capability on every callback exit."""

    def __init__(
        self,
        capability: RunActionCommittedContinuationCapability,
        *,
        owner_process_id: int,
        invoking_thread_id: int,
    ) -> None:
        self._capability = capability
        self._owner_process_id = owner_process_id
        self._invoking_thread_id = invoking_thread_id

    def __enter__(self) -> RunActionCommittedContinuationCapability:
        return self._capability

    def __exit__(self, exception_type, exception, traceback) -> bool:
        RunActionCommittedContinuationCapability._finish_invocation(
            self._capability,
            owner_process_id=self._owner_process_id,
            invoking_thread_id=self._invoking_thread_id,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
        )
        return False


class _RunActionReleasePublicationAuthorization:
    """Publisher-private security authority for one already-frozen receipt."""

    def __init__(
        self,
        *,
        capability: RunActionCommittedContinuationCapability,
        resolved_workload_observation: RunActionResolvedWorkloadObservation,
        required_security_observation: SecurityDenylistObservation,
        release_receipt_size_bound_bytes: int,
        _authority: object,
    ) -> None:
        if (
            type(capability) is not RunActionCommittedContinuationCapability
            or type(resolved_workload_observation)
            is not RunActionResolvedWorkloadObservation
            or type(required_security_observation) is not SecurityDenylistObservation
            or type(release_receipt_size_bound_bytes) is not int
            or release_receipt_size_bound_bytes <= 0
            or _authority is not _RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "release publication authorization lacks issuance authority"
            )
        self._capability = capability
        self._resolved_workload_observation = resolved_workload_observation
        self._required_security_observation = required_security_observation
        self._release_receipt_size_bound_bytes = release_receipt_size_bound_bytes
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
            authorities.credential_broker_registry,
            authorities.clock,
            anchor_realtime,
        )
        release_authorization = RunActionReleaseAuthorizationObservation.mint(
            security_observation=self._required_security_observation,
            authorized_at_boottime_nanoseconds=anchor_boottime,
            authorized_at_realtime_nanoseconds=anchor_realtime,
            credential_validity_observation=credential_validity,
        )
        receipt = mint_run_action_workload_release_receipt(
            activation_event=self._capability.activation_event,
            resolved_workload_observation=self._resolved_workload_observation,
            release_authorization_observation=release_authorization,
        )
        if len(receipt.to_json_bytes()) > self._release_receipt_size_bound_bytes:
            raise RunActionRecoveryError(
                "workload release receipt exceeded its formal envelope"
            )
        self._receipt = receipt
        return self._receipt

    def _receipt_size_bound(
        self,
        *,
        _authority: object,
    ) -> int:
        self._require_preparing(_authority)
        return self._release_receipt_size_bound_bytes

    def _authorize_frozen_release_once(
        self,
        *,
        candidate: _RunActionFrozenControlFileCandidate,
        _authority: object,
    ) -> RunActionWorkloadReleaseReceipt | None:
        self._require_preparing(_authority)
        if (
            type(candidate) is not _RunActionFrozenControlFileCandidate
            or type(self._receipt) is not RunActionWorkloadReleaseReceipt
        ):
            raise RunActionRecoveryError(
                "release authorization lacks one exact frozen receipt candidate"
            )
        receipt_payload = candidate._begin_publication(
            self._receipt.to_json_bytes(),
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        if (
            receipt_payload != self._receipt.to_json_bytes()
            or self._receipt.release_authorization_observation.security_observation
            != self._required_security_observation
        ):
            raise RunActionRecoveryError(
                "frozen release candidate differs from its issued authorization"
            )
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            _continuation_lifecycle(
                capability,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).release_publication_state = "authorizing"
        authorities = self._release_authorities()
        with _RunActionReleaseLinkInvocation(self):
            _revalidate_release_credential_validity(
                self._resolved_workload_observation,
                authorities.credential_broker_registry,
                self._receipt,
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
            candidate._prepare_authorized_link_once(
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )
            current_boottime_nanoseconds = authorities.clock.boottime_nanoseconds()
            if (
                type(current_boottime_nanoseconds) is not int
                or current_boottime_nanoseconds <= 0
                or current_boottime_nanoseconds
                > self._receipt.release_commit_deadline_boottime_nanoseconds
            ):
                return None
            candidate._link_prepared_once(
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )
            return self._receipt

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
            _continuation_lifecycle(
                capability,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).release_publication_state = "spent"

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
            _continuation_lifecycle(
                capability,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).release_publication_state = "spent"

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


class _RunActionTimeoutPublicationAuthorization:
    """Publisher-private authority for one released-to-timeout transition."""

    def __init__(
        self,
        *,
        capability: RunActionCommittedContinuationCapability,
        observed_before_boottime_nanoseconds: int,
        _authority: object,
    ) -> None:
        if (
            type(capability) is not RunActionCommittedContinuationCapability
            or type(observed_before_boottime_nanoseconds) is not int
            or observed_before_boottime_nanoseconds <= 0
            or capability._timeout_publication_state != "preparing"
            or _authority is not _RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "timeout publication authorization lacks issuance authority"
            )
        self._capability = capability
        self._observed_before_boottime_nanoseconds = (
            observed_before_boottime_nanoseconds
        )
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._directive: RunActionTimeoutDirective | None = None
        self._linked_evidence: _RunActionLinkedControlFileEvidence | None = None
        self._closed = False

    def _mint_timeout_directive(
        self,
        running_container_observation: RunActionBarrierRunningContainerObservation,
        host_boot_id: str,
        *,
        _authority: object,
    ) -> RunActionTimeoutDirective:
        """Bind one fresh running occurrence inside the sealed clock sandwich."""

        self._require_state("preparing", _authority)
        if (
            self._directive is not None
            or self._linked_evidence is not None
            or type(host_boot_id) is not str
            or type(running_container_observation)
            is not RunActionBarrierRunningContainerObservation
        ):
            raise RunActionRecoveryError(
                "timeout publication authorization already minted its directive"
            )
        capability = self._capability
        query = capability._query
        adoption = query.workload_release_adoption
        authorities = self._timeout_authorities()
        observed_after = _read_positive_release_clock(
            authorities.clock.boottime_nanoseconds(),
            "timeout observation finish",
        )
        if type(adoption) is not RunActionWorkloadReleaseAdoption:
            raise RunActionRecoveryError(
                "timeout publication lost its released occurrence"
            )
        release = adoption.workload_release_receipt
        directive = RunActionTimeoutDirective.mint(
            activation_event_id=query.activation_event.event_id,
            workload_release_receipt_id=release.workload_release_receipt_id,
            workload_release_adoption_id=adoption.workload_release_adoption_id,
            host_boot_id=host_boot_id,
            execution_deadline_boottime_nanoseconds=(
                release.execution_deadline_boottime_nanoseconds
            ),
            containment_deadline_boottime_nanoseconds=(
                release.containment_deadline_boottime_nanoseconds
            ),
            observed_before_boottime_nanoseconds=(
                self._observed_before_boottime_nanoseconds
            ),
            running_container_observation=running_container_observation,
            observed_after_boottime_nanoseconds=observed_after,
        )
        if (
            observed_after < self._observed_before_boottime_nanoseconds
            or host_boot_id != release.host_boot_id
            or not run_action_timeout_directive_evidence_matches(
                directive,
                query.activation_event.event_id,
                query.activation_revalidation_receipt,
                adoption,
            )
        ):
            raise RunActionRecoveryError(
                "timeout directive differs from its exact running release"
            )
        self._directive = directive
        return directive

    def _authorize_frozen_timeout_once(
        self,
        *,
        candidate: _RunActionFrozenControlFileCandidate,
        _authority: object,
    ) -> _RunActionLinkedControlFileEvidence:
        """Perform the final internal deadline check and irreversible link."""

        self._require_state("preparing", _authority)
        directive = self._directive
        if (
            type(candidate) is not _RunActionFrozenControlFileCandidate
            or type(directive) is not RunActionTimeoutDirective
            or self._linked_evidence is not None
            or candidate._begin_publication(
                directive.to_json_bytes(),
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )
            != directive.to_json_bytes()
        ):
            raise RunActionRecoveryError(
                "timeout authorization lacks one exact frozen directive"
            )
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            _continuation_lifecycle(
                capability,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).timeout_publication_state = "authorizing"
        candidate._prepare_authorized_link_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        final_boottime = _read_positive_release_clock(
            self._timeout_authorities().clock.boottime_nanoseconds(),
            "timeout link authorization",
        )
        if final_boottime < directive.observed_after_boottime_nanoseconds:
            raise RunActionRecoveryError("timeout link authorization clock regressed")
        evidence = candidate._link_prepared_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        if (
            evidence.transition is not _RunActionControlFileTransition.TIMEOUT
            or evidence.final_file_name != "timeout"
            or evidence.content_digest != tree_or_blob_digest(directive.to_json_bytes())
        ):
            raise RunActionRecoveryError(
                "linked timeout differs from its frozen directive"
            )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                capability._timeout_publication_state != "authorizing"
                or capability._timeout_directive_publication is not None
            ):
                raise RunActionRecoveryError(
                    "timeout publication authority changed across its link"
                )
            _continuation_lifecycle(
                capability,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).timeout_publication_state = "published_awaiting_adoption"
        self._linked_evidence = evidence
        return evidence

    def _complete_timeout_publication(
        self,
        adopted_timeout: RunActionTimeoutInspectionLease,
        *,
        _authority: object,
    ) -> None:
        """Register the fresh descriptor adoption of the just-linked directive."""

        self._require_state("published_awaiting_adoption", _authority)
        capability = self._capability
        query = capability._query
        adoption = query.workload_release_adoption
        evidence = self._linked_evidence
        if type(adopted_timeout) is not RunActionTimeoutInspectionLease:
            raise RunActionRecoveryError(
                "timeout publication completion lacks a retained adoption"
            )
        adopted_timeout.require_current()
        publication = adopted_timeout.timeout_directive_publication
        if (
            adopted_timeout.topology is not RunActionControlDirectoryTopology.TIMED_OUT
            or adopted_timeout.workload_release_adoption != adoption
            or type(publication) is not RunActionTimeoutDirectivePublicationReceipt
            or type(adoption) is not RunActionWorkloadReleaseAdoption
            or type(evidence) is not _RunActionLinkedControlFileEvidence
            or publication.timeout_directive != self._directive
            or evidence.mount_id != publication.timeout_mount_id
            or evidence.device != publication.timeout_device
            or evidence.inode != publication.timeout_inode
            or evidence.owner_user_id != publication.owner_user_id
            or evidence.owner_group_id != publication.owner_group_id
            or evidence.mode != publication.mode
            or evidence.link_count != publication.link_count
            or evidence.size_bytes != publication.size_bytes
            or evidence.content_digest != publication.content_digest
            or not run_action_timeout_publication_evidence_matches(
                publication,
                query.activation_event.event_id,
                query.activation_revalidation_receipt,
                adoption,
            )
        ):
            raise RunActionRecoveryError(
                "adopted timeout differs from its exact linked directive"
            )
        adopted_timeout.require_current()
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                capability._timeout_publication_state != "published_awaiting_adoption"
                or capability._timeout_directive_publication is not None
            ):
                raise RunActionRecoveryError(
                    "timeout publication completion changed before registration"
                )
            lifecycle = _continuation_lifecycle(
                capability,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            )
            lifecycle.timeout_directive_publication = deepcopy(publication)
            lifecycle.timeout_publication_state = "complete"

    def _timeout_authorities(self) -> _RunActionIssuedReleaseAuthorities:
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(
                self._capability
            )
        if type(authorities) is not _RunActionIssuedReleaseAuthorities:
            raise RunActionRecoveryError(
                "timeout publication lost its coordinator-issued clock"
            )
        return authorities

    def _require_state(self, expected_state: str, _authority: object) -> None:
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                self._closed
                or self._owner_process_id != os.getpid()
                or self._owner_thread_id != get_ident()
                or _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(capability))
                is not capability
                or capability._owner_process_id != os.getpid()
                or capability._state != "invoking"
                or capability._invoking_thread_id != get_ident()
                or capability._timeout_publication_state != expected_state
                or _authority is not _RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "timeout publication authorization is spent, inactive, or foreign"
                )

    def __enter__(self) -> "_RunActionTimeoutPublicationAuthorization":
        self._require_state(
            "preparing",
            _RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY,
        )
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                self._closed
                or self._owner_process_id != os.getpid()
                or self._owner_thread_id != get_ident()
                or capability._state != "invoking"
                or capability._invoking_thread_id != get_ident()
                or capability._timeout_publication_state
                not in {
                    "preparing",
                    "authorizing",
                    "published_awaiting_adoption",
                    "complete",
                }
            ):
                raise RunActionRecoveryError(
                    "timeout publication authorization close changed"
                )
            if capability._timeout_publication_state != "complete":
                _continuation_lifecycle(
                    capability,
                    _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
                ).timeout_publication_state = "spent"
            self._closed = True
        return False


class _RunActionTimeoutContainmentAuthorization:
    """Containment-private authority for one exact at-least-once signal."""

    def __init__(
        self,
        *,
        capability: RunActionCommittedContinuationCapability,
        _authority: object,
    ) -> None:
        if (
            type(capability) is not RunActionCommittedContinuationCapability
            or capability._timeout_containment_state != "preparing"
            or _authority is not _RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "timeout containment authorization lacks issuance authority"
            )
        self._capability = capability
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._signal: RunActionTimeoutContainmentSignal | None = None
        self._selected_at_boottime_nanoseconds: int | None = None
        self._closed = False

    def _select_signal(
        self,
        running_observation: RunActionBarrierRunningContainerObservation,
        host_boot_id: str,
        *,
        _authority: object,
    ) -> tuple[RunActionTimeoutContainmentSignal, int]:
        """Choose TERM or KILL from the original absolute deadline."""

        self._require_state("preparing", _authority)
        capability = self._capability
        query = capability._query
        publication = query.timeout_directive_publication
        adoption = query.workload_release_adoption
        if (
            self._signal is not None
            or self._selected_at_boottime_nanoseconds is not None
            or type(publication) is not RunActionTimeoutDirectivePublicationReceipt
            or type(adoption) is not RunActionWorkloadReleaseAdoption
            or type(running_observation)
            is not RunActionBarrierRunningContainerObservation
            or running_observation.complete_inspection_digest
            != capability._observation.observation_token
            or not run_action_running_container_occurrence_matches(
                running_observation,
                publication.timeout_directive.running_container_observation,
            )
            or host_boot_id != publication.timeout_directive.host_boot_id
        ):
            raise RunActionRecoveryError(
                "timeout containment lost its exact running occurrence"
            )
        authorities = _ISSUED_COMMITTED_CONTINUATION_RELEASE_AUTHORITIES.get(capability)
        if type(authorities) is not _RunActionIssuedReleaseAuthorities:
            raise RunActionRecoveryError(
                "timeout containment lost its coordinator-issued clock"
            )
        selected_at = _read_positive_release_clock(
            authorities.clock.boottime_nanoseconds(),
            "timeout containment signal selection",
        )
        directive = publication.timeout_directive
        if selected_at < directive.observed_after_boottime_nanoseconds:
            raise RunActionRecoveryError(
                "timeout containment signal-selection clock regressed"
            )
        signal = (
            RunActionTimeoutContainmentSignal.KILL
            if selected_at >= directive.containment_deadline_boottime_nanoseconds
            else RunActionTimeoutContainmentSignal.TERMINATE
        )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                capability._timeout_containment_state != "preparing"
                or capability._timeout_containment_result is not None
            ):
                raise RunActionRecoveryError(
                    "timeout containment authority changed before signal selection"
                )
            _continuation_lifecycle(
                capability,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            ).timeout_containment_state = "authorizing"
        self._signal = signal
        self._selected_at_boottime_nanoseconds = selected_at
        return signal, selected_at

    def _complete(
        self,
        result: RunActionTimeoutContainmentResult,
        *,
        _authority: object,
    ) -> None:
        """Register the exact stable occurrence observed after the command."""

        expected_state = "authorizing" if self._signal is not None else "preparing"
        self._require_state(expected_state, _authority)
        capability = self._capability
        query = capability._query
        adoption = query.workload_release_adoption
        publication = query.timeout_directive_publication
        if (
            type(result) is not RunActionTimeoutContainmentResult
            or result.signal is not self._signal
            or result.selected_at_boottime_nanoseconds
            != self._selected_at_boottime_nanoseconds
            or (
                self._signal is None
                and (
                    result.state is not RunActionTimeoutContainmentState.TERMINAL
                    or result.signal_dispatch_confirmed
                )
            )
            or type(adoption) is not RunActionWorkloadReleaseAdoption
            or type(publication) is not RunActionTimeoutDirectivePublicationReceipt
            or not self._post_signal_occurrence_matches(
                result,
                query,
                adoption,
                publication,
            )
        ):
            raise RunActionRecoveryError(
                "timeout containment result differs from its signal authority"
            )
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                capability._timeout_containment_state != expected_state
                or capability._timeout_containment_result is not None
            ):
                raise RunActionRecoveryError(
                    "timeout containment result changed before registration"
                )
            lifecycle = _continuation_lifecycle(
                capability,
                _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
            )
            lifecycle.timeout_containment_result = deepcopy(result)
            lifecycle.timeout_containment_state = "complete"

    @staticmethod
    def _post_signal_occurrence_matches(
        result: RunActionTimeoutContainmentResult,
        query: "RunActionCommittedSpawnQuery",
        adoption: RunActionWorkloadReleaseAdoption,
        publication: RunActionTimeoutDirectivePublicationReceipt,
    ) -> bool:
        if result.state is RunActionTimeoutContainmentState.RUNNING:
            running = result.running_observation
            return type(
                running
            ) is RunActionBarrierRunningContainerObservation and run_action_running_container_occurrence_matches(
                running,
                publication.timeout_directive.running_container_observation,
            )
        terminal = result.terminal_observation
        prepared = query.prepared_execution
        spawn = query.spawn_commit
        released_running = (
            adoption.workload_release_receipt.resolved_workload_observation.running_container_observation
        )
        return (
            type(terminal) is RunActionTerminalObservation
            and terminal.activation_revalidation_receipt_id
            == query.activation_revalidation_receipt.activation_revalidation_receipt_id
            and terminal.workload_release_adoption_id
            == adoption.workload_release_adoption_id
            and terminal.prepared_execution_id == prepared.prepared_execution_id
            and terminal.spawn_commit_id == spawn.spawn_commit_id
            and terminal.provider_execution_id == spawn.provider_execution_id
            and terminal.runtime_volume_authority_id
            == prepared.runtime_volume_authority.runtime_volume_authority_id
            and terminal.generation_nonce
            == prepared.runtime_volume_authority.generation_nonce
            and terminal.observed_inspect_projection
            == prepared.inert_container_evidence.issued_create_projection
            and terminal.started_at == released_running.started_at
        )

    def _require_state(self, expected_state: str, _authority: object) -> None:
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                self._closed
                or self._owner_process_id != os.getpid()
                or self._owner_thread_id != get_ident()
                or _ISSUED_COMMITTED_CONTINUATION_CAPABILITIES.get(id(capability))
                is not capability
                or capability._owner_process_id != os.getpid()
                or capability._state != "invoking"
                or capability._invoking_thread_id != get_ident()
                or capability._timeout_containment_state != expected_state
                or _authority is not _RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY
            ):
                raise RunActionRecoveryError(
                    "timeout containment authorization is spent, inactive, or foreign"
                )

    def __enter__(self) -> "_RunActionTimeoutContainmentAuthorization":
        self._require_state(
            "preparing",
            _RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY,
        )
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        capability = self._capability
        with _COMMITTED_CONTINUATION_CAPABILITY_LOCK:
            if (
                self._closed
                or self._owner_process_id != os.getpid()
                or self._owner_thread_id != get_ident()
                or capability._state != "invoking"
                or capability._invoking_thread_id != get_ident()
                or capability._timeout_containment_state
                not in {
                    "preparing",
                    "authorizing",
                    "complete",
                }
            ):
                raise RunActionRecoveryError(
                    "timeout containment authorization close changed"
                )
            if capability._timeout_containment_state != "complete":
                _continuation_lifecycle(
                    capability,
                    _authority=_RUN_ACTION_COMMITTED_CONTINUATION_STATE_AUTHORITY,
                ).timeout_containment_state = "spent"
            self._closed = True
        return False


def _observe_release_credential_validity(
    resolved: RunActionResolvedWorkloadObservation,
    credential_broker_registry: RunActionCredentialBrokerRegistry,
    clock: _SystemRunActionClock,
    anchor_realtime_nanoseconds: int,
) -> RunActionCredentialValidityObservation | None:
    return _observe_activation_credential_validity(
        resolved.activation_revalidation_receipt,
        credential_broker_registry,
        clock,
        anchor_realtime_nanoseconds,
    )


def _observe_pre_release_credential_state(
    activation: RunActionActivationRevalidationReceipt,
    credential_broker_registry: RunActionCredentialBrokerRegistry,
    clock: _SystemRunActionClock,
) -> RunActionPreReleaseCredentialObservation | None:
    if (
        type(activation) is not RunActionActivationRevalidationReceipt
        or type(credential_broker_registry) is not RunActionCredentialBrokerRegistry
        or type(clock) is not _SystemRunActionClock
    ):
        raise RunActionRecoveryError(
            "pre-release credential observation requires exact authority"
        )
    policy = activation.prepared_execution.preparation_claim.execution_policy
    credential_file = activation.credential_file_observation
    if policy.credential_policy.mode is RunActionCredentialMode.NONE:
        if credential_file is not None:
            raise RunActionRecoveryError(
                "credential-free activation carries credential evidence"
            )
        return None
    expected_authority_id = run_action_credential_lease_authority_id(
        activation.prepared_execution,
        activation.spawn_commit,
    )
    if (
        credential_file is None
        or credential_file.content_authority_id != expected_authority_id
    ):
        raise RunActionRecoveryError(
            "pre-release activation lacks its deterministic credential authority"
        )
    observed_before = _read_positive_release_clock(
        clock.realtime_nanoseconds(),
        "pre-release credential observation start",
    )
    status = credential_broker_registry.observe_exact(
        prepared_execution=activation.prepared_execution,
        spawn_commit=activation.spawn_commit,
        activated_credential_file_observation_id=(
            credential_file.activated_file_observation_id
        ),
    )
    observed_after = _read_positive_release_clock(
        clock.realtime_nanoseconds(),
        "pre-release credential observation finish",
    )
    required_valid_until = (
        observed_after
        + (
            policy.supervisor_limits.execution_timeout_seconds
            + policy.supervisor_limits.termination_grace_seconds
        )
        * _NANOSECONDS_PER_SECOND
    )
    if required_valid_until > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER:
        raise RunActionRecoveryError(
            "pre-release credential deadline exceeds its clock domain"
        )
    state = (
        RunActionPreReleaseCredentialState.EXPIRED
        if status.valid_until_realtime_nanoseconds <= observed_after
        else (
            RunActionPreReleaseCredentialState.VALID
            if status.valid_until_realtime_nanoseconds >= required_valid_until
            else RunActionPreReleaseCredentialState.RENEWAL_PENDING
        )
    )
    return RunActionPreReleaseCredentialObservation.mint(
        state=state,
        activation_revalidation_receipt=activation,
        credential_lease_status=status,
        observed_before_realtime_nanoseconds=observed_before,
        observed_after_realtime_nanoseconds=observed_after,
        required_valid_until_realtime_nanoseconds=required_valid_until,
    )


class _RunActionCredentialIssueResponseOwnership:
    """Burn one unused broker response unless its exact owner releases it."""

    def __init__(
        self,
        registry: RunActionCredentialBrokerRegistry,
        response: RunActionCredentialIssueResponse | None,
    ) -> None:
        if type(registry) is not RunActionCredentialBrokerRegistry or (
            response is not None
            and type(response) is not RunActionCredentialIssueResponse
        ):
            raise RunActionRecoveryError(
                "credential issue response ownership lacks exact authority"
            )
        self._registry = registry
        self._response = response
        self._released = False

    @property
    def response(self) -> RunActionCredentialIssueResponse | None:
        if self._released:
            raise RunActionRecoveryError(
                "credential issue response ownership is already released"
            )
        return self._response

    def release(self) -> RunActionCredentialIssueResponse | None:
        response = self.response
        self._released = True
        return response

    def __enter__(self) -> "_RunActionCredentialIssueResponseOwnership":
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if not self._released and self._response is not None:
            self._registry.burn_issue_response_exact(self._response)
        self._released = True
        self._response = None
        return False


def _issue_activation_credential_response(
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    credential_broker_registry: RunActionCredentialBrokerRegistry,
    clock: _SystemRunActionClock,
) -> RunActionCredentialIssueResponse | None:
    policy = prepared_execution.preparation_claim.execution_policy
    credential_policy = policy.credential_policy
    credential_broker_registry.require_policy(credential_policy)
    if credential_policy.mode is RunActionCredentialMode.NONE:
        return None
    observed_before = _read_positive_release_clock(
        clock.realtime_nanoseconds(),
        "credential issuance start",
    )
    response = credential_broker_registry.issue_or_replay_exact(
        prepared_execution,
        spawn_commit,
    )
    with _RunActionCredentialIssueResponseOwnership(
        credential_broker_registry,
        response,
    ) as ownership:
        observed_after = _read_positive_release_clock(
            clock.realtime_nanoseconds(),
            "credential issuance finish",
        )
        valid_until_realtime_nanoseconds = response.valid_until_realtime_nanoseconds
        maximum_valid_until = (
            observed_after
            + credential_policy.maximum_lease_seconds * _NANOSECONDS_PER_SECOND
        )
        if (
            observed_after < observed_before
            or maximum_valid_until > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            or (
                valid_until_realtime_nanoseconds > observed_after
                and valid_until_realtime_nanoseconds > maximum_valid_until
            )
        ):
            raise RunActionRecoveryError(
                "credential issuance exceeds its bounded lease interval"
            )
        return ownership.release()


def _observe_activation_credential_validity(
    activation: RunActionActivationRevalidationReceipt,
    credential_broker_registry: RunActionCredentialBrokerRegistry,
    clock: _SystemRunActionClock,
    anchor_realtime_nanoseconds: int,
) -> RunActionCredentialValidityObservation | None:
    if (
        type(activation) is not RunActionActivationRevalidationReceipt
        or type(credential_broker_registry) is not RunActionCredentialBrokerRegistry
        or type(clock) is not _SystemRunActionClock
    ):
        raise RunActionRecoveryError(
            "credential validity requires exact coordinator authority"
        )
    anchor_realtime_nanoseconds = _read_positive_release_clock(
        anchor_realtime_nanoseconds,
        "credential validity anchor",
    )
    policy = activation.prepared_execution.preparation_claim.execution_policy
    credential_file = activation.credential_file_observation
    if policy.credential_policy.mode is RunActionCredentialMode.NONE:
        if credential_file is not None:
            raise RunActionRecoveryError(
                "credential-free release carries a credential file"
            )
        return None
    expected_lease_authority_id = run_action_credential_lease_authority_id(
        activation.prepared_execution,
        activation.spawn_commit,
    )
    if (
        credential_file is None
        or credential_file.content_authority_id != expected_lease_authority_id
    ):
        raise RunActionRecoveryError(
            "credentialed activation lacks its deterministic lease authority"
        )
    observed_before = _read_positive_release_clock(
        clock.realtime_nanoseconds(),
        "credential observation start",
    )
    status = credential_broker_registry.observe_exact(
        prepared_execution=activation.prepared_execution,
        spawn_commit=activation.spawn_commit,
        activated_credential_file_observation_id=(
            credential_file.activated_file_observation_id
        ),
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
    if required_valid_until > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER:
        raise RunActionRecoveryError(
            "credential validity deadline exceeds its unsigned-64 clock domain"
        )
    if type(status) is not RunActionCredentialLeaseStatus:
        raise RunActionRecoveryError(
            "credential broker returned another validity status"
        )
    if (
        observed_after < observed_before
        or status.valid_until_realtime_nanoseconds <= observed_after
        or status.valid_until_realtime_nanoseconds < required_valid_until
        or (
            status.valid_until_realtime_nanoseconds - observed_after
            > policy.credential_policy.maximum_lease_seconds * _NANOSECONDS_PER_SECOND
        )
    ):
        raise RunActionRecoveryError(
            "credential validity cannot authorize the release deadline"
        )
    validity = RunActionCredentialValidityObservation.mint(
        activated_credential_file_observation_id=(
            credential_file.activated_file_observation_id
        ),
        credential_lease_authority_id=expected_lease_authority_id,
        observed_at_realtime_nanoseconds=observed_after,
        valid_until_realtime_nanoseconds=(status.valid_until_realtime_nanoseconds),
    )
    return validity


def _revalidate_release_credential_validity(
    resolved: RunActionResolvedWorkloadObservation,
    credential_broker_registry: RunActionCredentialBrokerRegistry,
    receipt: RunActionWorkloadReleaseReceipt,
    clock: _SystemRunActionClock,
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
    ):
        raise RunActionRecoveryError(
            "credential revalidation lacks exact coordinator authority"
        )
    current = _observe_activation_credential_validity(
        activation,
        credential_broker_registry,
        clock,
        receipt.release_authorization_observation.authorized_at_realtime_nanoseconds,
    )
    containment_deadline_realtime = (
        receipt.release_authorization_observation.authorized_at_realtime_nanoseconds
        + (
            policy.supervisor_limits.execution_timeout_seconds
            + policy.supervisor_limits.termination_grace_seconds
        )
        * _NANOSECONDS_PER_SECOND
    )
    if containment_deadline_realtime > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER:
        raise RunActionRecoveryError(
            "credential revalidation deadline exceeds its unsigned-64 clock domain"
        )
    if (
        type(current) is not RunActionCredentialValidityObservation
        or current.activated_credential_file_observation_id
        != required_validity.activated_credential_file_observation_id
        or current.credential_lease_authority_id
        != required_validity.credential_lease_authority_id
        or current.observed_at_realtime_nanoseconds
        < required_validity.observed_at_realtime_nanoseconds
        or current.valid_until_realtime_nanoseconds < containment_deadline_realtime
    ):
        raise RunActionRecoveryError("credential authority changed before release link")


def _read_positive_release_clock(value: int, name: str) -> int:
    if (
        type(value) is not int
        or value <= 0
        or value > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
    ):
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
    credential_retirement_intent: RunActionCredentialRetirementIntent | None
    workload_release_adoption: RunActionWorkloadReleaseAdoption | None
    timeout_directive_publication: RunActionTimeoutDirectivePublicationReceipt | None

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
                self.credential_retirement_intent is not None
                and type(self.credential_retirement_intent)
                is not RunActionCredentialRetirementIntent
            )
            or (
                self.workload_release_adoption is not None
                and type(self.workload_release_adoption)
                is not RunActionWorkloadReleaseAdoption
            )
            or (
                self.timeout_directive_publication is not None
                and type(self.timeout_directive_publication)
                is not RunActionTimeoutDirectivePublicationReceipt
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
        if self.credential_retirement_intent is not None and (
            self.workload_release_adoption is not None
            or self.timeout_directive_publication is not None
            or not credential_retirement_intent_matches_activation(
                self.credential_retirement_intent,
                self.activation_event.event_id,
                self.activation_revalidation_receipt,
            )
        ):
            raise RunActionRecoveryError(
                "committed run action query differs from credential retirement"
            )
        if (
            self.workload_release_adoption is None
            and self.timeout_directive_publication is not None
        ) or (
            self.timeout_directive_publication is not None
            and not run_action_timeout_publication_evidence_matches(
                self.timeout_directive_publication,
                self.activation_event.event_id,
                self.activation_revalidation_receipt,
                self.workload_release_adoption,
            )
        ):
            raise RunActionRecoveryError(
                "committed run action query differs from its control topology"
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
    def control_directory_topology(self) -> RunActionControlDirectoryTopology:
        if self.timeout_directive_publication is not None:
            return RunActionControlDirectoryTopology.TIMED_OUT
        if self.workload_release_adoption is not None:
            return RunActionControlDirectoryTopology.RELEASED
        return RunActionControlDirectoryTopology.EMPTY

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
        operation_id: str,
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
    ) -> None:
        if type(implementations) is not tuple or not implementations:
            raise RunActionRecoveryError(
                "run action recovery implementation registry is empty or malformed"
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
        credential_broker_registry: RunActionCredentialBrokerRegistry,
        implementation_registry: RunActionRecoveryImplementationRegistry,
        resource_finalization_authority: RunActionResourceFinalizationAuthority,
        _authority: object,
    ) -> None:
        if (
            type(active_workspace) is not ActiveLaunchWorkspace
            or type(publisher) is not RunStatePublisher
            or publisher._authority is not active_workspace
            or not hasattr(security_authority, "observe_exact_descendant_of")
            or type(credential_broker_registry) is not RunActionCredentialBrokerRegistry
            or type(implementation_registry)
            is not RunActionRecoveryImplementationRegistry
            or _authority is not _RUN_ACTION_RECOVERY_COORDINATOR_AUTHORITY
        ):
            raise RunActionRecoveryError(
                "run action recovery authorities are incompatible"
            )
        active_workspace.require_control_authority()
        require_run_action_resource_finalization_authority(
            resource_finalization_authority,
            publisher._action_store,
            publisher._settings,
        )
        self._active_workspace = active_workspace
        self._publisher = publisher
        self._settings = publisher._settings
        self._store = publisher._action_store
        self._workspace_promoter = RunActionWorkspacePromoter(
            active_workspace=active_workspace,
            settings=publisher._settings,
        )
        self._security_authority = security_authority
        credential_broker_registry._require_owner_process()
        self._credential_broker_registry = credential_broker_registry
        self._release_clock = _SystemRunActionClock()
        implementation_registry._require_owner_process()
        self._implementation_registry = implementation_registry
        self._resource_finalization_authority = resource_finalization_authority
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
            self._require_release_receipt_policy(execution_adapter)
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
            self._require_release_receipt_policy(execution_adapter)
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
            self._formal_release_receipt_size_bound(
                execution_adapter,
                prepared_execution,
                spawn_commit,
                frontier.checkpoint.safety_state.security_observation,
            )
            query = RunActionUnactivatedSpawnQuery(
                prepared_execution=prepared_execution,
                spawn_commit=spawn_commit,
            )
            adapter_query = deepcopy(query)
            observation = execution_adapter.inspect_unactivated(adapter_query)
            if adapter_query != query:
                raise RunActionRecoveryError(
                    "execution adapter changed its unactivated-spawn query"
                )
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
        if tail_kind is RunActionExecutionEventKind.CREDENTIAL_RETIREMENT_REQUESTED:
            execution_adapter = self._resolve_execution_adapter(
                self._implementation_registry,
                reservation,
            )
            self._recover_credential_retirement(
                session,
                execution_adapter,
                frontier,
                descriptors,
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
        self._formal_release_receipt_size_bound(
            execution_adapter,
            prepared_execution,
            spawn_commit,
            frontier.checkpoint.safety_state.security_observation,
        )
        credential_policy = (
            prepared_execution.preparation_claim.execution_policy.credential_policy
        )
        if credential_policy.mode is RunActionCredentialMode.SUPERVISOR_FILE and (
            session.credential_retirement_event_size_bytes(
                prepared_execution,
                spawn_commit,
            )
            > self._publisher._settings.run_action_event_size_bytes
            or credential_expiry_termination_event_size_bound(
                reservation=session.reservation,
                preparation_allocation=(session.events[1].preparation_allocation),
                prepared_execution=prepared_execution,
                spawn_commit=spawn_commit,
            )
            > self._publisher._settings.run_action_event_size_bytes
        ):
            raise RunActionRecoveryError(
                "credential expiry events exceed their durable envelope"
            )
        if not self._security_is_current(frontier):
            return None
        request_payload = session.read_request()
        credential_issue_response = _issue_activation_credential_response(
            prepared_execution,
            spawn_commit,
            self._credential_broker_registry,
            self._release_clock,
        )
        with _RunActionCredentialIssueResponseOwnership(
            self._credential_broker_registry,
            credential_issue_response,
        ) as credential_ownership:
            if not self._security_is_current(frontier):
                return None
            capability = RunActionActivationCapability(
                prepared_execution=prepared_execution,
                spawn_commit=spawn_commit,
                request_payload=request_payload,
                credential_issue_response=credential_ownership.response,
                credential_broker_registry=self._credential_broker_registry,
                workspace_descriptor=workspace_descriptor,
                _authority=_RUN_ACTION_ACTIVATION_AUTHORITY,
            )
            adapter_activation = capability._invoke_once(execution_adapter)
        if type(adapter_activation) is not RunActionActivationRevalidationReceipt:
            raise RunActionRecoveryError(
                "execution adapter returned another or oversized activation"
            )
        activation = RunActionActivationRevalidationReceipt.from_json_bytes(
            adapter_activation.to_json_bytes()
        )
        if (
            activation.prepared_execution != prepared_execution
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
        activation = activation_event.activation_revalidation_receipt
        with open_run_action_timeout_inspection(
            activation_event=activation_event,
            launch_settings=self._publisher._settings,
        ) as control_inspection:
            workload_release_adoption = control_inspection.workload_release_adoption
            timeout_directive_publication = (
                control_inspection.timeout_directive_publication
            )
            query = RunActionCommittedSpawnQuery(
                preparation_allocation=session.events[1].preparation_allocation,
                activation_event=activation_event,
                credential_retirement_intent=None,
                workload_release_adoption=workload_release_adoption,
                timeout_directive_publication=timeout_directive_publication,
            )
            if query.control_directory_topology is not control_inspection.topology:
                raise RunActionRecoveryError(
                    "committed query differs from retained control topology"
                )
            if (
                query.control_directory_topology
                is RunActionControlDirectoryTopology.EMPTY
            ):
                self._formal_release_receipt_size_bound(
                    execution_adapter,
                    activation.prepared_execution,
                    activation.spawn_commit,
                    frontier.checkpoint.safety_state.security_observation,
                )
            adapter_query = deepcopy(query)
            adapter_observation = execution_adapter.inspect_committed(adapter_query)
            if adapter_query != query:
                raise RunActionRecoveryError(
                    "execution adapter changed its committed-spawn query"
                )
            control_inspection.require_current()
        if type(adapter_observation) is not RunActionCommittedSpawnObservation:
            raise RunActionRecoveryError(
                "execution adapter returned an invalid committed-spawn observation"
            )
        observation = RunActionCommittedSpawnObservation(
            state=adapter_observation.state,
            observation_token=adapter_observation.observation_token,
        )
        if observation.state is RunActionCommittedSpawnState.UNKNOWN:
            return None
        self._require_unchanged_host_workspace(
            session.reservation,
            descriptors,
            "host workspace changed before committed provider continuation",
        )
        pre_release_credential_observation = None
        if (
            query.control_directory_topology is RunActionControlDirectoryTopology.EMPTY
            and observation.state
            in {
                RunActionCommittedSpawnState.INERT_CONTINUABLE,
                RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            }
        ):
            pre_release_credential_observation = _observe_pre_release_credential_state(
                activation,
                self._credential_broker_registry,
                self._release_clock,
            )
            if (
                pre_release_credential_observation is not None
                and pre_release_credential_observation.state
                is RunActionPreReleaseCredentialState.RENEWAL_PENDING
            ):
                return None
        credential_is_expired = (
            pre_release_credential_observation is not None
            and pre_release_credential_observation.state
            is RunActionPreReleaseCredentialState.EXPIRED
        )
        if credential_is_expired:
            session.commit_credential_retirement(
                RunActionCredentialRetirementIntent.mint(
                    activation_event_id=activation_event.event_id,
                    pre_release_credential_observation_id=(
                        pre_release_credential_observation.pre_release_credential_observation_id
                    ),
                    credential_lease_status=(
                        pre_release_credential_observation.credential_lease_status
                    ),
                    observed_before_realtime_nanoseconds=(
                        pre_release_credential_observation.observed_before_realtime_nanoseconds
                    ),
                    observed_after_realtime_nanoseconds=(
                        pre_release_credential_observation.observed_after_realtime_nanoseconds
                    ),
                    required_valid_until_realtime_nanoseconds=(
                        pre_release_credential_observation.required_valid_until_realtime_nanoseconds
                    ),
                )
            )
            return None
        if (
            observation.state
            in {
                RunActionCommittedSpawnState.INERT_CONTINUABLE,
                RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            }
            and workload_release_adoption is None
            and not credential_is_expired
            and not self._security_is_current(frontier)
        ):
            return None
        capability = RunActionCommittedContinuationCapability(
            query=query,
            observation=observation,
            pre_release_credential_observation=(pre_release_credential_observation),
            required_security_observation=(
                frontier.checkpoint.safety_state.security_observation
            ),
            security_authority=self._security_authority,
            credential_broker_registry=self._credential_broker_registry,
            release_clock=self._release_clock,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
        )
        outcome = capability._invoke_once(execution_adapter)
        publication_fence = (
            outcome.provider_termination_publication_fence
            if type(outcome) is RunActionContinuationOutcome
            else None
        )
        if type(publication_fence) is RunActionProviderTerminationPublicationFence:
            descriptors.callback(publication_fence.close)
            publication_fence.require_current()
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
        if outcome.state is RunActionContinuationState.TIMEOUT_PUBLISHED:
            with open_run_action_timeout_inspection(
                activation_event=activation_event,
                launch_settings=self._publisher._settings,
            ) as timeout_inspection:
                if (
                    timeout_inspection.topology
                    is not RunActionControlDirectoryTopology.TIMED_OUT
                    or timeout_inspection.workload_release_adoption
                    != workload_release_adoption
                    or timeout_inspection.timeout_directive_publication
                    != outcome.timeout_directive_publication
                ):
                    raise RunActionRecoveryError(
                        "published timeout differs from fresh recovery adoption"
                    )
                timeout_inspection.require_current()
            return None
        if outcome.state is RunActionContinuationState.RESULT_CAPTURED:
            return self._record_and_interpret(
                session,
                outcome.result,
                workload_release_adoption,
                descriptors,
                workspace_lock_descriptor,
            )
        if outcome.state is RunActionContinuationState.PROVIDER_TERMINATED:
            self._record_provider_termination(
                session,
                outcome.provider_termination_receipt,
                workload_release_adoption,
                publication_fence,
                descriptors,
            )
            return None
        raise RunActionRecoveryError(
            "execution adapter returned an unknown committed continuation"
        )

    def _recover_credential_retirement(
        self,
        session,
        execution_adapter: RunActionExecutionAdapter,
        frontier: ReconciledRunFrontier,
        descriptors: ExitStack,
    ) -> None:
        """Enforce a durable expiry decision without consulting mutable authority."""

        retirement_event = session.events[-1]
        activation_event = session.events[4]
        if (
            retirement_event.event_kind
            is not RunActionExecutionEventKind.CREDENTIAL_RETIREMENT_REQUESTED
            or type(retirement_event.credential_retirement_intent)
            is not RunActionCredentialRetirementIntent
        ):
            raise RunActionRecoveryError(
                "credential retirement recovery requires its durable intent tail"
            )
        with open_run_action_timeout_inspection(
            activation_event=activation_event,
            launch_settings=self._publisher._settings,
        ) as control_inspection:
            query = RunActionCommittedSpawnQuery(
                preparation_allocation=session.events[1].preparation_allocation,
                activation_event=activation_event,
                credential_retirement_intent=(
                    retirement_event.credential_retirement_intent
                ),
                workload_release_adoption=None,
                timeout_directive_publication=None,
            )
            if (
                control_inspection.topology
                is not RunActionControlDirectoryTopology.EMPTY
                or control_inspection.workload_release_adoption is not None
                or control_inspection.timeout_directive_publication is not None
            ):
                raise RunActionRecoveryError(
                    "credential retirement lost its empty control topology"
                )
            adapter_query = deepcopy(query)
            adapter_observation = execution_adapter.inspect_committed(adapter_query)
            if adapter_query != query:
                raise RunActionRecoveryError(
                    "execution adapter changed its retirement query"
                )
            control_inspection.require_current()
        if type(adapter_observation) is not RunActionCommittedSpawnObservation:
            raise RunActionRecoveryError(
                "execution adapter returned an invalid retirement observation"
            )
        observation = RunActionCommittedSpawnObservation(
            state=adapter_observation.state,
            observation_token=adapter_observation.observation_token,
        )
        if observation.state is RunActionCommittedSpawnState.UNKNOWN:
            return
        self._require_unchanged_host_workspace(
            session.reservation,
            descriptors,
            "host workspace changed before credential retirement",
        )
        capability = RunActionCommittedContinuationCapability(
            query=query,
            observation=observation,
            pre_release_credential_observation=None,
            required_security_observation=(
                frontier.checkpoint.safety_state.security_observation
            ),
            security_authority=self._security_authority,
            credential_broker_registry=self._credential_broker_registry,
            release_clock=self._release_clock,
            _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
        )
        outcome = capability._invoke_once(execution_adapter)
        publication_fence = (
            outcome.provider_termination_publication_fence
            if type(outcome) is RunActionContinuationOutcome
            else None
        )
        if type(publication_fence) is RunActionProviderTerminationPublicationFence:
            descriptors.callback(publication_fence.close)
            publication_fence.require_current()
        if (
            type(outcome) is not RunActionContinuationOutcome
            or not self._continuation_outcome_allowed(observation, outcome)
            or outcome.state
            not in {
                RunActionContinuationState.PENDING,
                RunActionContinuationState.PROVIDER_TERMINATED,
            }
        ):
            raise RunActionRecoveryError(
                "credential retirement returned an unauthorized continuation"
            )
        self._require_unchanged_host_workspace(
            session.reservation,
            descriptors,
            "host workspace changed during credential retirement",
        )
        if outcome.state is RunActionContinuationState.PENDING:
            return
        receipt = outcome.provider_termination_receipt
        if (
            type(receipt) is not RunActionProviderTerminationReceipt
            or receipt.reason
            is not RunActionProviderTerminationReason.CREDENTIAL_EXPIRED
            or receipt.credential_retirement_intent
            != retirement_event.credential_retirement_intent
        ):
            raise RunActionRecoveryError(
                "credential retirement termination lost its durable cause"
            )
        self._record_provider_termination(
            session,
            receipt,
            None,
            publication_fence,
            descriptors,
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
                RunActionContinuationState.TIMEOUT_PUBLISHED,
            },
            RunActionCommittedSpawnState.TERMINAL_CONTINUABLE: {
                RunActionContinuationState.PENDING,
                RunActionContinuationState.RESULT_CAPTURED,
                RunActionContinuationState.PROVIDER_TERMINATED,
            },
            RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE: {
                RunActionContinuationState.PENDING,
                RunActionContinuationState.PROVIDER_TERMINATED,
            },
            RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE: {
                RunActionContinuationState.PENDING,
                RunActionContinuationState.PROVIDER_TERMINATED,
            },
            RunActionCommittedSpawnState.UNKNOWN: set(),
        }
        return outcome.state in admitted[observation.state]

    def _record_provider_termination(
        self,
        session,
        receipt: RunActionProviderTerminationReceipt,
        expected_release_adoption: RunActionWorkloadReleaseAdoption | None,
        publication_fence: RunActionProviderTerminationPublicationFence | None,
        descriptors: ExitStack,
    ) -> None:
        """Publish only a registered receipt under one retained release fence."""

        fence_required = type(
            receipt
        ) is RunActionProviderTerminationReceipt and receipt.reason in {
            RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
            RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL,
            RunActionProviderTerminationReason.CREDENTIAL_EXPIRED,
        }
        if type(
            receipt
        ) is not RunActionProviderTerminationReceipt or fence_required != (
            type(publication_fence) is RunActionProviderTerminationPublicationFence
        ):
            raise RunActionRecoveryError(
                "provider termination publication lacks a registered receipt"
            )
        if publication_fence is not None:
            publication_fence.require_current()
        activation_event = session.events[4]
        self._require_unchanged_host_workspace(
            session.reservation,
            descriptors,
            "host workspace changed before provider termination event",
        )
        with open_run_action_timeout_inspection(
            activation_event=activation_event,
            launch_settings=self._publisher._settings,
        ) as control_inspection:
            self._require_provider_termination_release_fence(
                control_inspection,
                receipt,
                expected_release_adoption,
            )
            control_inspection.require_current()
            if publication_fence is not None:
                publication_fence.require_current()
            session.terminate_provider(receipt)
            if publication_fence is not None:
                publication_fence.require_current()
            control_inspection.require_current()
        self._require_unchanged_host_workspace(
            session.reservation,
            descriptors,
            "host workspace changed across provider termination event",
        )

    @staticmethod
    def _require_provider_termination_release_fence(
        control_inspection,
        receipt: RunActionProviderTerminationReceipt,
        expected_release_adoption: RunActionWorkloadReleaseAdoption | None,
    ) -> None:
        required_topology = {
            RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS: (
                RunActionControlDirectoryTopology.EMPTY
            ),
            RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL: (
                RunActionControlDirectoryTopology.EMPTY
            ),
            RunActionProviderTerminationReason.CREDENTIAL_EXPIRED: (
                RunActionControlDirectoryTopology.EMPTY
            ),
            RunActionProviderTerminationReason.TIMEOUT: (
                RunActionControlDirectoryTopology.TIMED_OUT
            ),
            RunActionProviderTerminationReason.OOM: (
                RunActionControlDirectoryTopology.RELEASED
            ),
            RunActionProviderTerminationReason.NONZERO_EXIT: (
                RunActionControlDirectoryTopology.RELEASED
            ),
            RunActionProviderTerminationReason.MISSING_RESULT: (
                RunActionControlDirectoryTopology.RELEASED
            ),
        }[receipt.reason]
        if control_inspection.topology is not required_topology:
            raise RunActionRecoveryError(
                "provider termination lost its exact control topology"
            )
        if required_topology is RunActionControlDirectoryTopology.EMPTY:
            if (
                expected_release_adoption is not None
                or receipt.workload_release_adoption is not None
                or control_inspection.workload_release_adoption is not None
                or control_inspection.timeout_directive_publication is not None
            ):
                raise RunActionRecoveryError(
                    "pre-release termination carries release adoption"
                )
            return
        if (
            expected_release_adoption is None
            or control_inspection.workload_release_adoption != expected_release_adoption
            or receipt.workload_release_adoption != expected_release_adoption
            or (
                required_topology is RunActionControlDirectoryTopology.TIMED_OUT
                and (
                    control_inspection.timeout_directive_publication
                    != receipt.timeout_directive_publication
                )
            )
            or (
                required_topology is RunActionControlDirectoryTopology.RELEASED
                and control_inspection.timeout_directive_publication is not None
            )
        ):
            raise RunActionRecoveryError(
                "released termination lost its exact release adoption"
            )

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
        expected_release_adoption: RunActionWorkloadReleaseAdoption | None,
        descriptors: ExitStack,
        workspace_lock_descriptor: int,
    ) -> RunActionAcceptance:
        if type(result) is not RunActionProviderResult:
            raise RunActionRecoveryError(
                "execution adapter returned an invalid provider result"
            )
        spawn_commit = session.events[3].spawn_commit
        activation_event = session.events[4]
        activation = activation_event.activation_revalidation_receipt
        with open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=self._publisher._settings,
        ) as release_inspection:
            if (
                release_inspection.topology
                is not RunActionControlDirectoryTopology.RELEASED
                or (
                    expected_release_adoption is not None
                    and release_inspection.adoption != expected_release_adoption
                )
            ):
                raise RunActionRecoveryError(
                    "captured provider result lacks its exact released topology"
                )
            workload_release_adoption = release_inspection.adoption
            if not run_action_terminal_result_evidence_matches(
                result.terminal_observation,
                result.result_capture_receipt,
                activation,
                workload_release_adoption,
            ):
                raise RunActionRecoveryError(
                    "execution adapter result differs from durable activation"
                )
            release_inspection.require_current()
            session.record_result(
                spawn_commit=spawn_commit,
                workload_release_adoption=workload_release_adoption,
                terminal_observation=result.terminal_observation,
                result_capture_receipt=result.result_capture_receipt,
                result_payload=result.result_payload,
            )
            release_inspection.require_current()
        return self._interpret_received(
            session,
            descriptors,
            workspace_lock_descriptor,
        )

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
            operation_id=reservation.intent.operation_id,
            request_payload=request_payload,
            result_payload=result_payload,
        )
        repeated_interpretation = interpreter.interpret(
            operation_id=reservation.intent.operation_id,
            request_payload=request_payload,
            result_payload=result_payload,
        )
        if (
            type(interpreted) is not RunActionInterpretedResult
            or repeated_interpretation != interpreted
            or interpreted.operation_id != reservation.intent.operation_id
        ):
            raise RunActionRecoveryError(
                "run action result interpretation is invalid or nondeterministic"
            )
        successful_edit = (
            reservation.intent.workspace_access
            is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
            and interpreted.disposition is RunActionResultDisposition.SUCCEEDED
        )
        has_workspace_expectations = (
            interpreted.expected_workspace_before_source_tree_digest is not None
        )
        if successful_edit and not has_workspace_expectations:
            raise RunActionRecoveryError(
                "successful edit result lacks workspace expectations"
            )
        if not successful_edit and has_workspace_expectations:
            raise RunActionRecoveryError(
                "non-edit result carries workspace expectations"
            )
        workspace_before = reservation.frontier.workspace_before
        if successful_edit and (
            workspace_before is None
            or interpreted.expected_workspace_before_source_tree_digest
            != workspace_before.source_tree_digest
        ):
            raise RunActionRecoveryError(
                "edit result expects another workspace predecessor"
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
                candidate_frontier = inspect_run_workspace_frontier(
                    candidate.workspace_descriptor,
                    settings=self._settings,
                    expected_commit_sha=None,
                )
                candidate.require_current()
                if (
                    interpreted.expected_workspace_after_source_tree_digest
                    != candidate_frontier.source_tree_digest
                ):
                    raise RunActionRecoveryError(
                        "edit result expects another workspace successor"
                    )
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
            if (
                interpreted.expected_workspace_after_source_tree_digest
                != workspace_promotion.candidate_workspace.source_tree_digest
            ):
                raise RunActionRecoveryError(
                    "edit result expects another workspace successor"
                )
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
        unresolved_operation_id = None
        for events in operations:
            if events[-1].event_kind not in _TERMINAL_KINDS:
                if unresolved_operation_id is not None:
                    raise RunActionRecoveryError(
                        "run action recovery has multiple unresolved operations"
                    )
                unresolved_operation_id = events[0].reservation.intent.operation_id
                continue
            self._resource_finalization_authority.finalize_terminal(
                events[0].reservation.intent.operation_id
            )
        finalized_inspection = self._store.inspect()
        if finalized_inspection != inspection:
            raise RunActionRecoveryError(
                "run action ledger changed during terminal resource finalization"
            )
        recovered = []
        for events in operations:
            if events[-1].event_kind not in _TERMINAL_KINDS:
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

    def _require_release_receipt_policy(
        self,
        execution_adapter: RunActionExecutionAdapter,
    ) -> None:
        policy_bound = (
            execution_adapter.execution_policy.supervisor_limits.release_receipt_size_bytes
        )
        policy_commit_timeout = (
            execution_adapter.execution_policy.supervisor_limits.release_commit_timeout_seconds
        )
        policy_timeout_directive_bound = (
            execution_adapter.execution_policy.supervisor_limits.timeout_directive_size_bytes
        )
        configured_bound = (
            self._publisher._settings.run_action_release_receipt_size_bytes
        )
        configured_commit_timeout = (
            self._publisher._settings.run_action_release_commit_timeout_seconds
        )
        configured_timeout_directive_bound = (
            self._publisher._settings.run_action_timeout_directive_size_bytes
        )
        configured_process_snapshot_bound = (
            self._publisher._settings.run_action_process_snapshot_size_bytes
        )
        if (
            policy_bound != configured_bound
            or policy_commit_timeout != configured_commit_timeout
            or policy_timeout_directive_bound != configured_timeout_directive_bound
            or execution_adapter.execution_policy.supervisor_limits.process_snapshot_size_bytes
            != configured_process_snapshot_bound
        ):
            raise RunActionRecoveryError(
                "run action release-receipt or timeout policy differs from "
                "launch configuration"
            )
        self._credential_broker_registry.require_policy(
            execution_adapter.execution_policy.credential_policy
        )

    def _formal_release_receipt_size_bound(
        self,
        execution_adapter: RunActionExecutionAdapter,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        required_security_observation: SecurityDenylistObservation,
    ) -> int:
        self._require_release_receipt_policy(execution_adapter)
        if (
            execution_adapter.execution_policy
            != prepared_execution.preparation_claim.execution_policy
        ):
            raise RunActionRecoveryError(
                "run action formal release envelope uses another execution policy"
            )
        first = workload_release_receipt_size_bound(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            required_security_observation=required_security_observation,
        )
        second = workload_release_receipt_size_bound(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            required_security_observation=required_security_observation,
        )
        if (
            type(first) is not int
            or first <= 0
            or second != first
            or first
            > execution_adapter.execution_policy.supervisor_limits.release_receipt_size_bytes
        ):
            raise RunActionRecoveryError(
                "run action formal release-receipt envelope is invalid or too large"
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
    "RunActionPreReleaseCredentialObservation",
    "RunActionPreReleaseCredentialState",
    "RunActionProviderResult",
    "RunActionProviderTerminationPublicationFence",
    "RunActionRecoveredOperation",
    "RunActionRecoveryCoordinator",
    "RunActionRecoveryError",
    "RunActionRecoveryImplementation",
    "RunActionRecoveryImplementationRegistry",
    "RunActionRecoveryPlan",
    "RunActionRecoveryReport",
    "RunActionResultInterpreter",
]
