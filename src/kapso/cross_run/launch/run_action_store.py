"""Create-only durable execution state for run-scoped external actions."""

from __future__ import annotations

import ctypes
import fcntl
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import ClassVar

from kapso.cross_run.canonical import (
    content_id,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionContractError,
    RunActionIntent,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
    RunActionLedgerSnapshot,
    RunActionOperationTail,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.launch.workspace_frontier import RunWorkspaceFrontierIdentity
from kapso.cross_run.settings import LaunchSettings

_RENAME_NOREPLACE = 1
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_NONCE_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_EVENT_NAME_PATTERN = re.compile(
    r"^operation-(?P<operation>[0-9a-f]{64})-event-(?P<number>[0-9]{4})[.]json$"
)
_RESULT_NAME_PATTERN = re.compile(r"^result-(?P<digest>[0-9a-f]{64})[.]blob$")
_ACCEPTED_NAME_PATTERN = re.compile(r"^accepted-(?P<digest>[0-9a-f]{64})[.]blob$")
_INPUT_NAME_PATTERN = re.compile(r"^input-(?P<digest>[0-9a-f]{64})[.]blob$")
_STAGING_NAME_PATTERN = re.compile(
    r"^[.](?P<kind>accepted|event|input|result)-[0-9a-f]{32}[.]tmp$"
)
_MAXIMUM_EVENT_COUNT = 4
_RUN_ACTION_STORE_AUTHORITY = object()
_RUN_ACTION_MUTATION_AUTHORITY = object()
_RUN_ACTION_RECOVERY_AUTHORITY = object()


class RunActionStoreError(RunActionContractError):
    """The durable run-action execution prefix is unsafe or conflicting."""


class RunActionTerminalReason(str, Enum):
    """Recovery or pre-spawn reasons that permanently close an operation."""

    STALE_FRONTIER = "stale_frontier"
    PROVIDER_INTERRUPTED = "provider_interrupted"
    PROVIDER_FAILED = "provider_failed"


class RunActionResultDisposition(str, Enum):
    """Adapter-level meaning of one durably received provider response."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RunActionViewBinding(StrictContract):
    """Serializable content identity of one checkpoint-owned mutable view."""

    relative_path: str
    digest: str
    size_bytes: int

    def _validate(self) -> None:
        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path
            or "\x00" in self.relative_path
            or _DIGEST_PATTERN.fullmatch(self.digest) is None
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
        ):
            raise RunActionStoreError("run action view binding is invalid")


@dataclass(frozen=True)
class RunActionWorkspaceBinding(StrictContract):
    """Serializable clean source/Git identity before or after one action."""

    workspace_device: int
    workspace_inode: int
    branch: str
    commit_sha: str
    parent_commit_shas: tuple[str, ...]
    git_tree_sha: str
    source_tree_digest: str
    git_closure_digest: str
    source_entry_count: int
    source_size_bytes: int

    def _validate(self) -> None:
        identity = RunWorkspaceFrontierIdentity(
            workspace_identity=(self.workspace_device, self.workspace_inode),
            branch=self.branch,
            commit_sha=self.commit_sha,
            parent_commit_shas=self.parent_commit_shas,
            git_tree_sha=self.git_tree_sha,
            source_tree_digest=self.source_tree_digest,
            git_closure_digest=self.git_closure_digest,
            source_entry_count=self.source_entry_count,
            source_size_bytes=self.source_size_bytes,
        )
        if identity.workspace_identity != (
            self.workspace_device,
            self.workspace_inode,
        ):
            raise RunActionStoreError("run action workspace binding is invalid")

    @classmethod
    def from_identity(
        cls,
        identity: RunWorkspaceFrontierIdentity,
    ) -> "RunActionWorkspaceBinding":
        if type(identity) is not RunWorkspaceFrontierIdentity:
            raise RunActionStoreError(
                "run action workspace binding requires one exact frontier"
            )
        return cls(
            workspace_device=identity.workspace_identity[0],
            workspace_inode=identity.workspace_identity[1],
            branch=identity.branch,
            commit_sha=identity.commit_sha,
            parent_commit_shas=identity.parent_commit_shas,
            git_tree_sha=identity.git_tree_sha,
            source_tree_digest=identity.source_tree_digest,
            git_closure_digest=identity.git_closure_digest,
            source_entry_count=identity.source_entry_count,
            source_size_bytes=identity.source_size_bytes,
        )

    def to_identity(self) -> RunWorkspaceFrontierIdentity:
        return RunWorkspaceFrontierIdentity(
            workspace_identity=(self.workspace_device, self.workspace_inode),
            branch=self.branch,
            commit_sha=self.commit_sha,
            parent_commit_shas=self.parent_commit_shas,
            git_tree_sha=self.git_tree_sha,
            source_tree_digest=self.source_tree_digest,
            git_closure_digest=self.git_closure_digest,
            source_entry_count=self.source_entry_count,
            source_size_bytes=self.source_size_bytes,
        )


@dataclass(frozen=True)
class RunActionFrontierBinding(StrictContract):
    """Complete durable identity of the reconciled frontier authorizing an action."""

    frontier_binding_id: str
    bootstrap_pin_id: str
    run_checkpoint_id: str
    safety_state_id: str
    security_observation_id: str
    generation_id: str
    journal_head_id: str
    journal_size_bytes: int
    bundle_digest: str
    bundle_size_bytes: int
    view_bindings: tuple[RunActionViewBinding, ...]
    workspace_before: RunActionWorkspaceBinding | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-frontier-binding"
    IDENTITY_FIELD: ClassVar[str] = "frontier_binding_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.bootstrap_pin_id, "bootstrap-pin", "bootstrap pin"),
            (self.run_checkpoint_id, "run-checkpoint", "checkpoint"),
            (self.safety_state_id, "run-safety-state", "safety state"),
            (
                self.security_observation_id,
                "security-denylist-observation",
                "security observation",
            ),
            (
                self.generation_id,
                "run-derived-state-generation",
                "derived generation",
            ),
            (self.journal_head_id, "run-checkpoint-head", "checkpoint head"),
        ):
            _require_namespaced_id(value, namespace, f"run action {name}")
        if (
            type(self.journal_size_bytes) is not int
            or self.journal_size_bytes <= 0
            or _DIGEST_PATTERN.fullmatch(self.bundle_digest) is None
            or type(self.bundle_size_bytes) is not int
            or self.bundle_size_bytes <= 0
            or any(
                type(binding) is not RunActionViewBinding
                for binding in self.view_bindings
            )
            or tuple(binding.relative_path for binding in self.view_bindings)
            != tuple(sorted({binding.relative_path for binding in self.view_bindings}))
            or (
                self.workspace_before is not None
                and type(self.workspace_before) is not RunActionWorkspaceBinding
            )
        ):
            raise RunActionStoreError("run action frontier binding is invalid")


@dataclass(frozen=True)
class RunActionReservation(StrictContract):
    """One operation identity durably reserved against one exact run frontier."""

    reservation_id: str
    intent: RunActionIntent
    frontier: RunActionFrontierBinding
    request_blob: RunActionRequestBlob
    predecessor_ledger_snapshot_id: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-reservation"
    IDENTITY_FIELD: ClassVar[str] = "reservation_id"

    def _validate(self) -> None:
        if (
            type(self.intent) is not RunActionIntent
            or type(self.frontier) is not RunActionFrontierBinding
            or type(self.request_blob) is not RunActionRequestBlob
        ):
            raise RunActionStoreError(
                "run action reservation requires intent and frontier"
            )
        _require_namespaced_id(
            self.predecessor_ledger_snapshot_id,
            RunActionLedgerSnapshot.CONTENT_NAMESPACE,
            "run action predecessor ledger",
        )
        expected = {
            self.intent.action_intent_id,
            self.frontier.frontier_binding_id,
            self.frontier.bootstrap_pin_id,
            self.frontier.run_checkpoint_id,
            self.frontier.safety_state_id,
            self.frontier.security_observation_id,
            self.frontier.generation_id,
            self.frontier.journal_head_id,
            self.request_blob.request_blob_id,
            self.predecessor_ledger_snapshot_id,
        }
        if (
            self.exact_dependency_ids != tuple(sorted(set(self.exact_dependency_ids)))
            or set(self.exact_dependency_ids) != expected
        ):
            raise RunActionStoreError(
                "run action reservation dependency closure is not exact"
            )
        if (self.intent.workspace_access is RunFrontierWorkspaceAccess.NONE) != (
            self.frontier.workspace_before is None
        ):
            raise RunActionStoreError(
                "run action reservation workspace authority differs from its intent"
            )
        if (
            self.request_blob.digest != self.intent.request_digest
            or self.request_blob.size_bytes != self.intent.request_size_bytes
        ):
            raise RunActionStoreError(
                "run action reservation request blob differs from its intent"
            )

    @classmethod
    def build(
        cls,
        *,
        intent: RunActionIntent,
        frontier: RunActionFrontierBinding,
        predecessor_ledger: RunActionLedgerSnapshot,
    ) -> "RunActionReservation":
        if (
            type(intent) is not RunActionIntent
            or type(frontier) is not RunActionFrontierBinding
            or type(predecessor_ledger) is not RunActionLedgerSnapshot
        ):
            raise RunActionStoreError(
                "run action reservation requires exact typed inputs"
            )
        request_blob = RunActionRequestBlob.mint(
            digest=intent.request_digest,
            size_bytes=intent.request_size_bytes,
        )
        return cls.mint(
            intent=intent,
            frontier=frontier,
            request_blob=request_blob,
            predecessor_ledger_snapshot_id=(predecessor_ledger.ledger_snapshot_id),
            exact_dependency_ids=tuple(
                sorted(
                    {
                        intent.action_intent_id,
                        frontier.frontier_binding_id,
                        frontier.bootstrap_pin_id,
                        frontier.run_checkpoint_id,
                        frontier.safety_state_id,
                        frontier.security_observation_id,
                        frontier.generation_id,
                        frontier.journal_head_id,
                        request_blob.request_blob_id,
                        predecessor_ledger.ledger_snapshot_id,
                    }
                )
            ),
        )


@dataclass(frozen=True)
class RunActionRequestBlob(StrictContract):
    """Content descriptor for the complete untruncated provider request."""

    request_blob_id: str
    digest: str
    size_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-request-blob"
    IDENTITY_FIELD: ClassVar[str] = "request_blob_id"

    def _validate(self) -> None:
        if (
            _DIGEST_PATTERN.fullmatch(self.digest) is None
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
        ):
            raise RunActionStoreError("run action request blob is invalid")


@dataclass(frozen=True)
class RunActionSpawnCommit(StrictContract):
    """Pre-spawn durable fence for one exact provider invocation."""

    spawn_commit_id: str
    reservation_id: str
    provider_execution_id: str
    invocation_nonce: str
    security_observation_id: str
    boundary_identity: RunActionBoundaryIdentity

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-spawn-commit"
    IDENTITY_FIELD: ClassVar[str] = "spawn_commit_id"

    def _validate(self) -> None:
        _require_namespaced_id(
            self.reservation_id,
            RunActionReservation.CONTENT_NAMESPACE,
            "run action spawn reservation",
        )
        require_identifier(
            self.provider_execution_id,
            "run action provider execution ID",
        )
        _require_namespaced_id(
            self.security_observation_id,
            "security-denylist-observation",
            "run action spawn security observation",
        )
        if _NONCE_PATTERN.fullmatch(self.invocation_nonce) is None:
            raise RunActionStoreError(
                "run action spawn nonce must be 128-bit lowercase hex"
            )
        if type(self.boundary_identity) is not RunActionBoundaryIdentity:
            raise RunActionStoreError(
                "run action spawn requires one exact boundary identity"
            )

    @classmethod
    def build(
        cls,
        *,
        reservation_id: str,
        provider_execution_id: str,
        security_observation_id: str,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> "RunActionSpawnCommit":
        return cls.mint(
            reservation_id=reservation_id,
            provider_execution_id=provider_execution_id,
            invocation_nonce=secrets.token_hex(16),
            security_observation_id=security_observation_id,
            boundary_identity=boundary_identity,
        )


@dataclass(frozen=True)
class RunActionResultBlob(StrictContract):
    """Content descriptor for complete provider or accepted result bytes."""

    digest: str
    size_bytes: int

    def _validate(self) -> None:
        if (
            _DIGEST_PATTERN.fullmatch(self.digest) is None
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
        ):
            raise RunActionStoreError("run action result blob is invalid")


@dataclass(frozen=True)
class RunActionResultReceipt(StrictContract):
    """Durable receipt that the committed provider invocation returned."""

    result_receipt_id: str
    spawn_commit_id: str
    provider_execution_id: str
    result_blob: RunActionResultBlob

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-result-receipt"
    IDENTITY_FIELD: ClassVar[str] = "result_receipt_id"

    def _validate(self) -> None:
        _require_namespaced_id(
            self.spawn_commit_id,
            RunActionSpawnCommit.CONTENT_NAMESPACE,
            "run action result spawn commit",
        )
        require_identifier(
            self.provider_execution_id,
            "run action result provider execution ID",
        )
        if type(self.result_blob) is not RunActionResultBlob:
            raise RunActionStoreError(
                "run action result receipt requires one result blob"
            )


@dataclass(frozen=True)
class RunActionAcceptance(StrictContract):
    """Adapter acceptance and post-action workspace proof."""

    acceptance_id: str
    result_receipt_id: str
    disposition: RunActionResultDisposition
    accepted_result_blob: RunActionResultBlob
    workspace_after: RunActionWorkspaceBinding | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-acceptance"
    IDENTITY_FIELD: ClassVar[str] = "acceptance_id"

    def _validate(self) -> None:
        _require_namespaced_id(
            self.result_receipt_id,
            RunActionResultReceipt.CONTENT_NAMESPACE,
            "run action acceptance result",
        )
        if (
            type(self.disposition) is not RunActionResultDisposition
            or type(self.accepted_result_blob) is not RunActionResultBlob
            or (
                self.workspace_after is not None
                and type(self.workspace_after) is not RunActionWorkspaceBinding
            )
        ):
            raise RunActionStoreError("run action acceptance is invalid")


@dataclass(frozen=True)
class RunActionExecutionEvent(StrictContract):
    """One create-only event in an exact per-operation execution prefix."""

    event_id: str
    event_number: int
    predecessor_event_id: str | None
    event_kind: RunActionExecutionEventKind
    reservation: RunActionReservation
    spawn_commit: RunActionSpawnCommit | None
    result_receipt: RunActionResultReceipt | None
    acceptance: RunActionAcceptance | None
    terminal_reason: RunActionTerminalReason | None
    workspace_after: RunActionWorkspaceBinding | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-execution-event"
    IDENTITY_FIELD: ClassVar[str] = "event_id"

    def _validate(self) -> None:
        if (
            type(self.event_number) is not int
            or not 1 <= self.event_number <= _MAXIMUM_EVENT_COUNT
            or (self.predecessor_event_id is None) != (self.event_number == 1)
            or type(self.event_kind) is not RunActionExecutionEventKind
            or type(self.reservation) is not RunActionReservation
        ):
            raise RunActionStoreError("run action execution event prefix is invalid")
        if self.predecessor_event_id is not None:
            _require_namespaced_id(
                self.predecessor_event_id,
                self.CONTENT_NAMESPACE,
                "run action event predecessor",
            )
        shape = (
            self.spawn_commit is not None,
            self.result_receipt is not None,
            self.acceptance is not None,
            self.terminal_reason is not None,
            self.workspace_after is not None,
        )
        expected = {
            RunActionExecutionEventKind.INTENT_RESERVED: (
                False,
                False,
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.SPAWN_COMMITTED: (
                True,
                False,
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.RESULT_RECEIVED: (
                False,
                True,
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.RESULT_ACCEPTED: (
                False,
                False,
                True,
                False,
                False,
            ),
            RunActionExecutionEventKind.CANCELLED: (
                False,
                False,
                False,
                True,
                False,
            ),
            RunActionExecutionEventKind.INTERRUPTED: (
                False,
                False,
                False,
                True,
                self.workspace_after is not None,
            ),
        }[self.event_kind]
        if shape != expected:
            raise RunActionStoreError(
                "run action execution event payload differs from its kind"
            )
        if (
            self.event_kind is RunActionExecutionEventKind.CANCELLED
            and self.terminal_reason is not RunActionTerminalReason.STALE_FRONTIER
        ) or (
            self.event_kind is RunActionExecutionEventKind.INTERRUPTED
            and self.terminal_reason
            not in {
                RunActionTerminalReason.PROVIDER_INTERRUPTED,
                RunActionTerminalReason.PROVIDER_FAILED,
            }
        ):
            raise RunActionStoreError(
                "run action terminal reason differs from its event kind"
            )


@dataclass(frozen=True)
class RunActionStoreInspection:
    """One registry-locked view of the ledger and its complete event prefixes."""

    ledger: RunActionLedgerSnapshot
    operation_events: tuple[tuple[RunActionExecutionEvent, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.ledger) is not RunActionLedgerSnapshot
            or type(self.operation_events) is not tuple
            or any(
                not events
                or any(type(event) is not RunActionExecutionEvent for event in events)
                for events in self.operation_events
            )
            or tuple(
                events[0].reservation.intent.operation_id
                for events in self.operation_events
            )
            != tuple(
                sorted(
                    events[0].reservation.intent.operation_id
                    for events in self.operation_events
                )
            )
        ):
            raise RunActionStoreError("run action store inspection is invalid")
        observed_tails = tuple(
            RunActionOperationTail(
                operation_id=events[0].reservation.intent.operation_id,
                reservation_id=events[0].reservation.reservation_id,
                event_ids=tuple(event.event_id for event in events),
                tail_kind=events[-1].event_kind,
            )
            for events in self.operation_events
        )
        if RunActionLedgerSnapshot.build(observed_tails) != self.ledger:
            raise RunActionStoreError(
                "run action store inspection differs from its ledger"
            )

    def events_for(
        self,
        operation_id: str,
    ) -> tuple[RunActionExecutionEvent, ...]:
        """Return one exact prefix from this immutable inspection."""
        for events in self.operation_events:
            if events[0].reservation.intent.operation_id == operation_id:
                return events
        raise RunActionStoreError(
            "run action operation is absent from the store inspection"
        )

    def operations_since(
        self,
        predecessor: RunActionLedgerSnapshot,
    ) -> tuple[tuple[RunActionExecutionEvent, ...], ...]:
        """Order new terminal prefixes by their durable ledger CAS chain."""
        self.ledger.require_predecessor(predecessor)
        previous_operation_ids = {
            tail.operation_id for tail in predecessor.operation_tails
        }
        remaining = [
            events
            for events in self.operation_events
            if events[0].reservation.intent.operation_id not in previous_operation_ids
        ]
        ordered = []
        current = predecessor
        tails = {tail.operation_id: tail for tail in predecessor.operation_tails}
        while remaining:
            matching = tuple(
                events
                for events in remaining
                if events[0].reservation.predecessor_ledger_snapshot_id
                == current.ledger_snapshot_id
            )
            if len(matching) != 1:
                raise RunActionStoreError(
                    "run action reservations do not form one exact ledger chain"
                )
            events = matching[0]
            operation_id = events[0].reservation.intent.operation_id
            tails[operation_id] = RunActionOperationTail(
                operation_id=operation_id,
                reservation_id=events[0].reservation.reservation_id,
                event_ids=tuple(event.event_id for event in events),
                tail_kind=events[-1].event_kind,
            )
            current = RunActionLedgerSnapshot.build(tuple(tails.values()))
            ordered.append(events)
            remaining.remove(events)
        if current != self.ledger:
            raise RunActionStoreError(
                "ordered run action reservations differ from the live ledger"
            )
        return tuple(ordered)

    @staticmethod
    def workspace_chain(
        operations: tuple[tuple[RunActionExecutionEvent, ...], ...],
    ) -> tuple[tuple[RunActionWorkspaceBinding, RunActionWorkspaceBinding], ...]:
        """Return the exact workspace before/after chain of ordered terminals."""
        pairs = []
        previous_after = None
        for events in operations:
            before = events[0].reservation.frontier.workspace_before
            terminal = events[-1]
            if terminal.event_kind is RunActionExecutionEventKind.RESULT_ACCEPTED:
                after = terminal.acceptance.workspace_after
            elif terminal.event_kind is RunActionExecutionEventKind.INTERRUPTED:
                after = terminal.workspace_after
            elif terminal.event_kind is RunActionExecutionEventKind.CANCELLED:
                after = before
            else:
                raise RunActionStoreError(
                    "workspace chain contains a nonterminal run action"
                )
            if before is None:
                if after is not None:
                    raise RunActionStoreError(
                        "workspace-free action carries terminal workspace state"
                    )
                continue
            if after is None:
                raise RunActionStoreError(
                    "workspace action lacks concrete terminal workspace state"
                )
            if previous_after is not None and before != previous_after:
                raise RunActionStoreError(
                    "run action terminal workspaces do not form one exact chain"
                )
            pairs.append((before, after))
            previous_after = after
        return tuple(pairs)


class _RunActionExecutionSession:
    """One process-owned, exclusively locked operation prefix."""

    def __init__(
        self,
        store: "RunActionExecutionStore",
        reservation: RunActionReservation,
        events: tuple[RunActionExecutionEvent, ...],
        descriptors: ExitStack,
    ) -> None:
        self._store = store
        self.reservation = reservation
        self._events = events
        self._descriptors = descriptors
        self._owner_process_id = os.getpid()
        self._active = True

    @property
    def events(self) -> tuple[RunActionExecutionEvent, ...]:
        self._require_active()
        return self._events

    def reserve(self, request_payload: bytes) -> RunActionExecutionEvent:
        self._require_active()
        if self._events:
            raise RunActionStoreError("run action operation is already reserved")
        if (
            type(request_payload) is not bytes
            or not request_payload
            or tree_or_blob_digest(request_payload)
            != self.reservation.request_blob.digest
            or len(request_payload) != self.reservation.request_blob.size_bytes
        ):
            raise RunActionStoreError(
                "run action reservation requires its complete request bytes"
            )
        event = self._event(
            RunActionExecutionEventKind.INTENT_RESERVED,
        )
        self._store._reserve(
            self._descriptors,
            self.reservation,
            request_payload,
            event,
        )
        self._events = (event,)
        return event

    def commit_spawn(
        self,
        *,
        provider_execution_id: str,
        security_observation_id: str,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> RunActionSpawnCommit:
        self._require_tail(RunActionExecutionEventKind.INTENT_RESERVED)
        if (
            type(boundary_identity) is not RunActionBoundaryIdentity
            or boundary_identity != self.reservation.intent.boundary_identity
        ):
            raise RunActionStoreError(
                "run action spawn boundary differs from its reservation"
            )
        spawn_commit = RunActionSpawnCommit.build(
            reservation_id=self.reservation.reservation_id,
            provider_execution_id=provider_execution_id,
            security_observation_id=security_observation_id,
            boundary_identity=boundary_identity,
        )
        event = self._event(
            RunActionExecutionEventKind.SPAWN_COMMITTED,
            spawn_commit=spawn_commit,
        )
        self._append(event)
        return spawn_commit

    def record_result(
        self,
        *,
        spawn_commit: RunActionSpawnCommit,
        result_payload: bytes,
    ) -> RunActionResultReceipt:
        self._require_tail(RunActionExecutionEventKind.SPAWN_COMMITTED)
        durable_spawn = self._events[-1].spawn_commit
        if (
            type(spawn_commit) is not RunActionSpawnCommit
            or spawn_commit != durable_spawn
        ):
            raise RunActionStoreError(
                "run action result differs from its durable spawn"
            )
        if type(result_payload) is not bytes or not result_payload:
            raise RunActionStoreError(
                "run action result must be complete non-empty bytes"
            )
        result_blob = RunActionResultBlob(
            digest=tree_or_blob_digest(result_payload),
            size_bytes=len(result_payload),
        )
        result_receipt = RunActionResultReceipt.mint(
            spawn_commit_id=spawn_commit.spawn_commit_id,
            provider_execution_id=spawn_commit.provider_execution_id,
            result_blob=result_blob,
        )
        event = self._event(
            RunActionExecutionEventKind.RESULT_RECEIVED,
            result_receipt=result_receipt,
        )
        self._store._publish_result_event(
            self._descriptors,
            operation_id=self.reservation.intent.operation_id,
            result_payload=result_payload,
            result_blob=result_blob,
            kind="result",
            event=event,
        )
        self._events = (*self._events, event)
        return result_receipt

    def accept_result(
        self,
        *,
        result_receipt: RunActionResultReceipt,
        disposition: RunActionResultDisposition,
        accepted_result_payload: bytes,
        workspace_after: RunWorkspaceFrontierIdentity | None,
    ) -> RunActionAcceptance:
        self._require_tail(RunActionExecutionEventKind.RESULT_RECEIVED)
        if (
            type(result_receipt) is not RunActionResultReceipt
            or result_receipt != self._events[-1].result_receipt
            or type(accepted_result_payload) is not bytes
            or not accepted_result_payload
            or type(disposition) is not RunActionResultDisposition
        ):
            raise RunActionStoreError(
                "run action acceptance differs from its received result"
            )
        before = self.reservation.frontier.workspace_before
        after = (
            None
            if workspace_after is None
            else RunActionWorkspaceBinding.from_identity(workspace_after)
        )
        _require_workspace_acceptance(
            self.reservation.intent.workspace_access,
            before,
            after,
        )
        accepted_result_blob = RunActionResultBlob(
            digest=tree_or_blob_digest(accepted_result_payload),
            size_bytes=len(accepted_result_payload),
        )
        acceptance = RunActionAcceptance.mint(
            result_receipt_id=result_receipt.result_receipt_id,
            disposition=disposition,
            accepted_result_blob=accepted_result_blob,
            workspace_after=after,
        )
        event = self._event(
            RunActionExecutionEventKind.RESULT_ACCEPTED,
            acceptance=acceptance,
        )
        self._store._publish_result_event(
            self._descriptors,
            operation_id=self.reservation.intent.operation_id,
            result_payload=accepted_result_payload,
            result_blob=accepted_result_blob,
            kind="accepted",
            event=event,
        )
        self._events = (*self._events, event)
        return acceptance

    def cancel(self, reason: RunActionTerminalReason) -> None:
        self._require_tail(RunActionExecutionEventKind.INTENT_RESERVED)
        if reason is not RunActionTerminalReason.STALE_FRONTIER:
            raise RunActionStoreError(
                "only stale-frontier reservation may be cancelled"
            )
        self._append(
            self._event(
                RunActionExecutionEventKind.CANCELLED,
                terminal_reason=reason,
            )
        )

    def interrupt(
        self,
        *,
        reason: RunActionTerminalReason,
        workspace_after: RunWorkspaceFrontierIdentity | None,
    ) -> None:
        self._require_tail(RunActionExecutionEventKind.SPAWN_COMMITTED)
        if reason not in {
            RunActionTerminalReason.PROVIDER_INTERRUPTED,
            RunActionTerminalReason.PROVIDER_FAILED,
        }:
            raise RunActionStoreError("run action interruption reason is invalid")
        after = (
            None
            if workspace_after is None
            else RunActionWorkspaceBinding.from_identity(workspace_after)
        )
        _require_interrupted_workspace(
            self.reservation.intent.workspace_access,
            self.reservation.frontier.workspace_before,
            after,
        )
        self._append(
            self._event(
                RunActionExecutionEventKind.INTERRUPTED,
                terminal_reason=reason,
                workspace_after=after,
            )
        )

    def read_result(self, result_receipt: RunActionResultReceipt) -> bytes:
        self._require_active()
        if type(result_receipt) is not RunActionResultReceipt:
            raise RunActionStoreError(
                "run action result read requires one exact receipt"
            )
        return self._store._read_result(
            self._descriptors,
            result_receipt.result_blob,
            kind="result",
        )

    def read_request(self) -> bytes:
        self._require_active()
        return self._store._read_request(
            self._descriptors,
            self.reservation.request_blob,
        )

    def read_accepted_result(self, acceptance: RunActionAcceptance) -> bytes:
        self._require_active()
        if type(acceptance) is not RunActionAcceptance:
            raise RunActionStoreError(
                "run action accepted-result read requires one exact acceptance"
            )
        return self._store._read_result(
            self._descriptors,
            acceptance.accepted_result_blob,
            kind="accepted",
        )

    def _event(
        self,
        event_kind: RunActionExecutionEventKind,
        *,
        spawn_commit: RunActionSpawnCommit | None = None,
        result_receipt: RunActionResultReceipt | None = None,
        acceptance: RunActionAcceptance | None = None,
        terminal_reason: RunActionTerminalReason | None = None,
        workspace_after: RunActionWorkspaceBinding | None = None,
    ) -> RunActionExecutionEvent:
        return RunActionExecutionEvent.mint(
            event_number=len(self._events) + 1,
            predecessor_event_id=(
                None if not self._events else self._events[-1].event_id
            ),
            event_kind=event_kind,
            reservation=self.reservation,
            spawn_commit=spawn_commit,
            result_receipt=result_receipt,
            acceptance=acceptance,
            terminal_reason=terminal_reason,
            workspace_after=workspace_after,
        )

    def _append(self, event: RunActionExecutionEvent) -> None:
        self._store._publish_event(
            self._descriptors,
            self.reservation.intent.operation_id,
            event,
        )
        self._events = (*self._events, event)

    def _require_tail(self, kind: RunActionExecutionEventKind) -> None:
        self._require_active()
        if not self._events or self._events[-1].event_kind is not kind:
            raise RunActionStoreError(
                f"run action operation requires a {kind.value} tail"
            )

    def _require_active(self) -> None:
        if (
            not self._active
            or self._owner_process_id != os.getpid()
            or self._store._active_sessions.get(self.reservation.intent.operation_id)
            is not self
        ):
            raise RunActionStoreError(
                "run action session lacks its creator process and store authority"
            )

    def _close(self) -> None:
        self._active = False


class _RunActionSessionContext:
    def __init__(
        self,
        store: "RunActionExecutionStore",
        reservation: RunActionReservation,
    ) -> None:
        self.store = store
        self.reservation = reservation
        self.session = None

    def __enter__(self) -> _RunActionExecutionSession:
        with ExitStack() as descriptors:
            store_descriptor, _identity = self.store._open_store(descriptors)
            events = self.store._open_operation(
                store_descriptor,
                self.reservation.intent.operation_id,
                descriptors,
            )
            if events and events[0].reservation != self.reservation:
                raise RunActionStoreError(
                    "run action operation ID was reserved for another request"
                )
            session = _RunActionExecutionSession(
                self.store,
                self.reservation,
                events,
                descriptors,
            )
            self.store._register_session(session)
            session._descriptors = descriptors.pop_all()
        self.session = session
        return session

    def __exit__(self, exception_type, exception, traceback) -> bool:
        session = self.session
        self.store._unregister_session(session)
        session._close()
        session._descriptors.close()
        self.session = None
        return False


class RunActionExecutionStore:
    """Descriptor-safe owner of bounded per-operation action event prefixes."""

    def __init__(
        self,
        *,
        active_workspace: ActiveLaunchWorkspace,
        settings: LaunchSettings,
        _authority: object,
    ) -> None:
        if (
            type(active_workspace) is not ActiveLaunchWorkspace
            or type(settings) is not LaunchSettings
            or _authority is not _RUN_ACTION_STORE_AUTHORITY
            or content_id("launch-settings", settings.to_dict())
            != active_workspace.bootstrap_pin.installation_receipt.launch_settings_id
        ):
            raise RunActionStoreError(
                "run action store requires the active launch settings"
            )
        active_workspace.require_control_authority()
        self._active_workspace = active_workspace
        self._settings = settings
        self._owner_process_id = os.getpid()
        self._registry_lock = Lock()
        self._active_sessions: dict[str, _RunActionExecutionSession] = {}
        with ExitStack() as descriptors:
            store_descriptor, _identity = self._open_store(descriptors)
            self._lock_registry(store_descriptor, descriptors)
            self._clean_staging(store_descriptor)
            self._clean_orphan_blobs(store_descriptor)
            self._validate_store(store_descriptor)

    def _session(
        self,
        reservation: RunActionReservation,
        *,
        _authority: object,
    ) -> _RunActionSessionContext:
        self._require_owner_process()
        if (
            type(reservation) is not RunActionReservation
            or _authority is not _RUN_ACTION_MUTATION_AUTHORITY
        ):
            raise RunActionStoreError(
                "run action session requires sealed mutation authority"
            )
        return _RunActionSessionContext(self, reservation)

    def _recovery_session(
        self,
        reservation: RunActionReservation,
        *,
        _authority: object,
    ) -> _RunActionSessionContext:
        self._require_owner_process()
        if (
            type(reservation) is not RunActionReservation
            or _authority is not _RUN_ACTION_RECOVERY_AUTHORITY
        ):
            raise RunActionStoreError(
                "run action recovery session requires sealed recovery authority"
            )
        return _RunActionSessionContext(self, reservation)

    def snapshot(self) -> RunActionLedgerSnapshot:
        """Read and validate every durable operation prefix."""
        return self.inspect().ledger

    def inspect(self) -> RunActionStoreInspection:
        """Read the ledger and every event prefix under one registry lock."""
        self._require_owner_process()
        with ExitStack() as descriptors:
            return self.inspect_locked(descriptors)

    def inspect_locked(
        self,
        descriptors: ExitStack,
    ) -> RunActionStoreInspection:
        """Inspect while retaining the registry lock in the caller's stack."""
        self._require_owner_process()
        if type(descriptors) is not ExitStack:
            raise RunActionStoreError(
                "run action locked inspection requires one descriptor stack"
            )
        store_descriptor, _identity = self._open_store(descriptors)
        self._lock_registry(store_descriptor, descriptors)
        self._clean_staging(store_descriptor)
        self._clean_orphan_blobs(store_descriptor)
        event_names = self._validate_store(store_descriptor)
        ledger = self._snapshot_from_event_names(store_descriptor, event_names)
        return RunActionStoreInspection(
            ledger=ledger,
            operation_events=tuple(
                self._read_operation_events(
                    store_descriptor,
                    tail.operation_id,
                )
                for tail in ledger.operation_tails
            ),
        )

    def lock_workspace(
        self,
        access: RunFrontierWorkspaceAccess,
        descriptors: ExitStack,
    ) -> int:
        """Hold the receipt-pinned cross-process workspace action lock."""
        self._require_owner_process()
        if (
            type(access) is not RunFrontierWorkspaceAccess
            or type(descriptors) is not ExitStack
        ):
            raise RunActionStoreError(
                "run action workspace lock requires exact typed authority"
            )
        store_descriptor, _identity = self._open_store(descriptors)
        descriptor = os.open(
            "workspace.lock",
            os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=store_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        metadata = _require_private_file(
            descriptor,
            mode=0o600,
            maximum_size_bytes=0,
            allow_empty=True,
            name="run action workspace lock",
        )
        receipt = self._active_workspace.bootstrap_pin.installation_receipt
        if (metadata.st_dev, metadata.st_ino) != (
            receipt.run_action_workspace_lock_device,
            receipt.run_action_workspace_lock_inode,
        ):
            raise RunActionStoreError(
                "run action workspace lock differs from its receipt"
            )
        fcntl.flock(
            descriptor,
            (
                fcntl.LOCK_EX
                if access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                else fcntl.LOCK_SH
            ),
        )
        return descriptor

    def _open_store(
        self,
        descriptors: ExitStack,
    ) -> tuple[int, tuple[int, int]]:
        if type(descriptors) is not ExitStack:
            raise RunActionStoreError(
                "run action store access requires one descriptor stack"
            )
        self._require_owner_process()
        return self._active_workspace._open_run_action_store(descriptors)

    def _validate_store(self, store_descriptor: int) -> tuple[str, ...]:
        names = []
        total_size_bytes = 0
        staging_entry_count = 0
        event_operation_digests = set()
        with os.scandir(store_descriptor) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > self._settings.run_action_store_entry_limit:
                    raise RunActionStoreError(
                        "run action store exceeds its configured entry limit"
                    )
                if entry.is_symlink():
                    raise RunActionStoreError("run action store contains a symlink")
        event_names = []
        for name in names:
            if _EVENT_NAME_PATTERN.fullmatch(name) is not None:
                required_mode = 0o400
                maximum_size = self._settings.run_action_event_size_bytes
                event_names.append(name)
                event_operation_digests.add(
                    _EVENT_NAME_PATTERN.fullmatch(name).group("operation")
                )
            elif _RESULT_NAME_PATTERN.fullmatch(name) is not None:
                required_mode = 0o400
                maximum_size = self._settings.run_action_result_size_bytes
            elif _ACCEPTED_NAME_PATTERN.fullmatch(name) is not None:
                required_mode = 0o400
                maximum_size = self._settings.run_action_result_size_bytes
            elif _INPUT_NAME_PATTERN.fullmatch(name) is not None:
                required_mode = 0o400
                maximum_size = self._settings.run_action_request_size_bytes
            elif name in {"registry.lock", "workspace.lock"}:
                required_mode = 0o600
                maximum_size = 0
            elif _STAGING_NAME_PATTERN.fullmatch(name) is not None:
                required_mode = None
                maximum_size = max(
                    self._settings.run_action_event_size_bytes,
                    self._settings.run_action_request_size_bytes,
                    self._settings.run_action_result_size_bytes,
                )
                staging_entry_count += 1
            else:
                raise RunActionStoreError(
                    "run action store contains an unexpected entry"
                )
            with ExitStack() as entry_descriptors:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=store_descriptor,
                )
                entry_descriptors.callback(os.close, descriptor)
                metadata = os.fstat(descriptor)
            total_size_bytes += metadata.st_size
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_size > maximum_size
                or (
                    required_mode is not None
                    and stat.S_IMODE(metadata.st_mode) != required_mode
                )
                or (
                    required_mode is None
                    and stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
                )
            ):
                raise RunActionStoreError("run action store entry is unsafe")
        if (
            len(event_operation_digests) > self._settings.run_action_operation_limit
            or staging_entry_count > self._settings.run_action_staging_entry_limit
            or total_size_bytes > self._settings.run_action_store_size_bytes
            or "registry.lock" not in names
            or "workspace.lock" not in names
        ):
            raise RunActionStoreError(
                "run action store exceeds bounds or lacks fixed locks"
            )
        return tuple(event_names)

    def _open_operation(
        self,
        store_descriptor: int,
        operation_id: str,
        descriptors: ExitStack,
    ) -> tuple[RunActionExecutionEvent, ...]:
        first_event_name = self._event_name(operation_id, 1)
        with ExitStack() as registry_descriptors:
            self._lock_registry(store_descriptor, registry_descriptors)
            self._clean_staging(store_descriptor)
            self._clean_orphan_blobs(store_descriptor)
            self._validate_store(store_descriptor)
            if not os.access(
                first_event_name,
                os.F_OK,
                dir_fd=store_descriptor,
                follow_symlinks=False,
            ):
                return ()
            descriptor = os.open(
                first_event_name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=store_descriptor,
            )
            descriptors.callback(os.close, descriptor)
            metadata = _require_private_file(
                descriptor,
                mode=0o400,
                maximum_size_bytes=self._settings.run_action_event_size_bytes,
                allow_empty=False,
                name="run action first event lock",
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        rebound = os.stat(
            first_event_name,
            dir_fd=store_descriptor,
            follow_symlinks=False,
        )
        if (rebound.st_dev, rebound.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise RunActionStoreError(
                "run action first event lock changed before acquisition"
            )
        return self._read_operation_events(
            store_descriptor,
            operation_id,
        )

    def _snapshot_from_event_names(
        self,
        store_descriptor: int,
        event_names: tuple[str, ...],
    ) -> RunActionLedgerSnapshot:
        grouped: dict[str, list[tuple[int, str]]] = {}
        for name in event_names:
            match = _EVENT_NAME_PATTERN.fullmatch(name)
            if match is None:
                raise RunActionStoreError(
                    "validated run action event name became invalid"
                )
            grouped.setdefault(match.group("operation"), []).append(
                (int(match.group("number")), name)
            )
        tails = []
        provider_execution_ids = set()
        invocation_nonces = set()
        for operation_digest, numbered_names in sorted(grouped.items()):
            events = self._read_named_events(
                store_descriptor,
                operation_digest,
                tuple(sorted(numbered_names)),
            )
            reservation = events[0].reservation
            if _operation_digest(reservation.intent.operation_id) != (operation_digest):
                raise RunActionStoreError(
                    "run action operation filename differs from its reservation"
                )
            if len(events) >= 2 and events[1].event_kind is (
                RunActionExecutionEventKind.SPAWN_COMMITTED
            ):
                spawn = events[1].spawn_commit
                if (
                    spawn.provider_execution_id in provider_execution_ids
                    or spawn.invocation_nonce in invocation_nonces
                ):
                    raise RunActionStoreError(
                        "run action provider execution identity was reused"
                    )
                provider_execution_ids.add(spawn.provider_execution_id)
                invocation_nonces.add(spawn.invocation_nonce)
            request_payload = _read_file(
                store_descriptor,
                (
                    "input-"
                    f"{reservation.request_blob.digest.removeprefix('sha256:')}.blob"
                ),
                mode=0o400,
                maximum_size_bytes=self._settings.run_action_request_size_bytes,
                name_description="run action request",
            )
            if (
                len(request_payload) != reservation.request_blob.size_bytes
                or tree_or_blob_digest(request_payload)
                != reservation.request_blob.digest
            ):
                raise RunActionStoreError(
                    "run action request differs from its reservation"
                )
            for event in events:
                payload_descriptors = []
                if event.result_receipt is not None:
                    payload_descriptors.append(
                        ("result", event.result_receipt.result_blob)
                    )
                if event.acceptance is not None:
                    payload_descriptors.append(
                        ("accepted", event.acceptance.accepted_result_blob)
                    )
                for kind, result_blob in payload_descriptors:
                    result_payload = _read_file(
                        store_descriptor,
                        (
                            f"{kind}-"
                            f"{result_blob.digest.removeprefix('sha256:')}.blob"
                        ),
                        mode=0o400,
                        maximum_size_bytes=(
                            self._settings.run_action_result_size_bytes
                        ),
                        name_description=f"run action {kind} result",
                    )
                    if (
                        len(result_payload) != result_blob.size_bytes
                        or tree_or_blob_digest(result_payload) != result_blob.digest
                    ):
                        raise RunActionStoreError(
                            f"run action {kind} result differs from its receipt"
                        )
            tails.append(
                RunActionOperationTail(
                    operation_id=reservation.intent.operation_id,
                    reservation_id=reservation.reservation_id,
                    event_ids=tuple(event.event_id for event in events),
                    tail_kind=events[-1].event_kind,
                )
            )
        return RunActionLedgerSnapshot.build(tuple(tails))

    def _reserve(
        self,
        descriptors: ExitStack,
        reservation: RunActionReservation,
        request_payload: bytes,
        event: RunActionExecutionEvent,
    ) -> None:
        store_descriptor, _identity = self._open_store(descriptors)
        with ExitStack() as registry_descriptors:
            self._lock_registry(store_descriptor, registry_descriptors)
            self._clean_staging(store_descriptor)
            self._clean_orphan_blobs(store_descriptor)
            event_names = self._validate_store(store_descriptor)
            snapshot = self._snapshot_from_event_names(
                store_descriptor,
                event_names,
            )
            if (
                snapshot.ledger_snapshot_id
                != reservation.predecessor_ledger_snapshot_id
            ):
                raise RunActionStoreError(
                    "run action reservation predecessor ledger moved"
                )
            if (
                len(snapshot.operation_tails) + 1
                > self._settings.run_action_operation_limit
            ):
                raise RunActionStoreError(
                    "run action store reached its operation limit"
                )
            existing_intent_ids = {
                self._read_operation_events(
                    store_descriptor,
                    tail.operation_id,
                )[0].reservation.intent.action_intent_id
                for tail in snapshot.operation_tails
            }
            if (
                any(
                    tail.operation_id == reservation.intent.operation_id
                    for tail in snapshot.operation_tails
                )
                or reservation.intent.action_intent_id in existing_intent_ids
            ):
                raise RunActionStoreError(
                    "run action operation or intent was already reserved"
                )
            request_name = (
                "input-"
                f"{reservation.request_blob.digest.removeprefix('sha256:')}.blob"
            )
            request_exists = os.access(
                request_name,
                os.F_OK,
                dir_fd=store_descriptor,
                follow_symlinks=False,
            )
            if request_exists and (
                self._read_blob(
                    store_descriptor,
                    request_name,
                    maximum_size_bytes=(self._settings.run_action_request_size_bytes),
                    name="run action request",
                )
                != request_payload
            ):
                raise RunActionStoreError(
                    "existing run action input differs from its content name"
                )
            event_payload = event.to_json_bytes()
            self._require_store_capacity(
                store_descriptor,
                additional_size_bytes=(
                    len(event_payload) + (0 if request_exists else len(request_payload))
                ),
                additional_entry_count=1 + (0 if request_exists else 1),
            )
            self._publish_request_locked(
                store_descriptor,
                reservation.request_blob,
                request_payload,
            )
            self._publish_event_locked(
                store_descriptor,
                reservation.intent.operation_id,
                event,
            )
            first_event_name = self._event_name(
                reservation.intent.operation_id,
                1,
            )
            operation_descriptor = os.open(
                first_event_name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=store_descriptor,
            )
            descriptors.callback(os.close, operation_descriptor)
            _require_private_file(
                operation_descriptor,
                mode=0o400,
                maximum_size_bytes=self._settings.run_action_event_size_bytes,
                allow_empty=False,
                name="run action first event lock",
            )
            fcntl.flock(
                operation_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )

    def _read_operation_events(
        self,
        store_descriptor: int,
        operation_id: str,
    ) -> tuple[RunActionExecutionEvent, ...]:
        digest = _operation_digest(operation_id)
        numbered_names = []
        with os.scandir(store_descriptor) as entries:
            for entry in entries:
                match = _EVENT_NAME_PATTERN.fullmatch(entry.name)
                if match is not None and match.group("operation") == digest:
                    numbered_names.append((int(match.group("number")), entry.name))
                    if len(numbered_names) > _MAXIMUM_EVENT_COUNT:
                        raise RunActionStoreError(
                            "run action operation has too many events"
                        )
        return self._read_named_events(
            store_descriptor,
            digest,
            tuple(sorted(numbered_names)),
        )

    def _read_named_events(
        self,
        store_descriptor: int,
        operation_digest: str,
        numbered_names: tuple[tuple[int, str], ...],
    ) -> tuple[RunActionExecutionEvent, ...]:
        events = []
        for expected_number, name in numbered_names:
            if expected_number != len(events) + 1:
                raise RunActionStoreError(
                    "run action operation event sequence has a gap"
                )
            payload = _read_file(
                store_descriptor,
                name,
                mode=0o400,
                maximum_size_bytes=self._settings.run_action_event_size_bytes,
                name_description="run action event",
            )
            event = RunActionExecutionEvent.from_json_bytes(payload)
            if (
                event.to_json_bytes() != payload
                or event.event_number != expected_number
                or _operation_digest(event.reservation.intent.operation_id)
                != operation_digest
            ):
                raise RunActionStoreError(
                    "run action event bytes differ from their filename"
                )
            events.append(event)
        result = tuple(events)
        _validate_event_prefix(result)
        return result

    def _publish_event(
        self,
        descriptors: ExitStack,
        operation_id: str,
        event: RunActionExecutionEvent,
    ) -> None:
        store_descriptor, _identity = self._open_store(descriptors)
        with ExitStack() as registry_descriptors:
            self._lock_registry(store_descriptor, registry_descriptors)
            self._clean_staging(store_descriptor)
            self._clean_orphan_blobs(store_descriptor)
            event_names = self._validate_store(store_descriptor)
            self._require_unique_spawn_identity(
                store_descriptor,
                event_names,
                event=event,
            )
            self._require_store_capacity(
                store_descriptor,
                additional_size_bytes=len(event.to_json_bytes()),
                additional_entry_count=1,
            )
            self._publish_event_locked(
                store_descriptor,
                operation_id,
                event,
            )

    def _publish_event_locked(
        self,
        store_descriptor: int,
        operation_id: str,
        event: RunActionExecutionEvent,
    ) -> None:
        if (
            type(event) is not RunActionExecutionEvent
            or event.reservation.intent.operation_id != operation_id
        ):
            raise RunActionStoreError("run action event differs from its operation")
        payload = event.to_json_bytes()
        _publish_create_only(
            store_descriptor,
            temporary_name=f".event-{secrets.token_hex(16)}.tmp",
            destination_name=self._event_name(operation_id, event.event_number),
            payload=payload,
            maximum_size_bytes=self._settings.run_action_event_size_bytes,
            name="run action event",
        )

    def _publish_result_event(
        self,
        descriptors: ExitStack,
        *,
        operation_id: str,
        result_payload: bytes,
        result_blob: RunActionResultBlob,
        kind: str,
        event: RunActionExecutionEvent,
    ) -> None:
        if (
            type(result_payload) is not bytes
            or not result_payload
            or type(result_blob) is not RunActionResultBlob
            or result_blob.digest != tree_or_blob_digest(result_payload)
            or result_blob.size_bytes != len(result_payload)
            or kind not in {"accepted", "result"}
            or type(event) is not RunActionExecutionEvent
            or event.reservation.intent.operation_id != operation_id
        ):
            raise RunActionStoreError(
                "run action result event requires exact complete inputs"
            )
        store_descriptor, _identity = self._open_store(descriptors)
        destination_name = f"{kind}-{result_blob.digest.removeprefix('sha256:')}.blob"
        with ExitStack() as registry_descriptors:
            self._lock_registry(store_descriptor, registry_descriptors)
            self._clean_staging(store_descriptor)
            self._clean_orphan_blobs(store_descriptor)
            event_names = self._validate_store(store_descriptor)
            result_exists = os.access(
                destination_name,
                os.F_OK,
                dir_fd=store_descriptor,
                follow_symlinks=False,
            )
            if result_exists and (
                _read_file(
                    store_descriptor,
                    destination_name,
                    mode=0o400,
                    maximum_size_bytes=(self._settings.run_action_result_size_bytes),
                    name_description=f"run action {kind} result",
                )
                != result_payload
            ):
                raise RunActionStoreError(
                    f"existing run action {kind} result differs from its content name"
                )
            self._require_store_capacity(
                store_descriptor,
                additional_size_bytes=(
                    len(event.to_json_bytes())
                    + (0 if result_exists else len(result_payload))
                ),
                additional_entry_count=1 + (0 if result_exists else 1),
            )
            self._publish_result_locked(
                store_descriptor,
                result_blob,
                destination_name,
                result_payload,
                kind,
            )
            self._publish_event_locked(
                store_descriptor,
                operation_id,
                event,
            )

    def _require_unique_spawn_identity(
        self,
        store_descriptor: int,
        event_names: tuple[str, ...],
        *,
        event: RunActionExecutionEvent,
    ) -> None:
        if event.spawn_commit is None:
            return
        for tail in self._snapshot_from_event_names(
            store_descriptor,
            event_names,
        ).operation_tails:
            events = self._read_operation_events(
                store_descriptor,
                tail.operation_id,
            )
            if (
                len(events) < 2
                or events[1].event_kind
                is not RunActionExecutionEventKind.SPAWN_COMMITTED
            ):
                continue
            spawn = events[1].spawn_commit
            if (
                spawn.provider_execution_id == event.spawn_commit.provider_execution_id
                or spawn.invocation_nonce == event.spawn_commit.invocation_nonce
            ):
                raise RunActionStoreError(
                    "run action provider execution identity was reused"
                )

    def _publish_result_locked(
        self,
        store_descriptor: int,
        blob: RunActionResultBlob,
        destination_name: str,
        payload: bytes,
        kind: str,
    ) -> RunActionResultBlob:
        if os.access(
            destination_name,
            os.F_OK,
            dir_fd=store_descriptor,
            follow_symlinks=False,
        ):
            if (
                _read_file(
                    store_descriptor,
                    destination_name,
                    mode=0o400,
                    maximum_size_bytes=self._settings.run_action_result_size_bytes,
                    name_description=f"run action {kind} result",
                )
                != payload
            ):
                raise RunActionStoreError(
                    f"existing run action {kind} result differs from its content name"
                )
            return blob
        _publish_create_only(
            store_descriptor,
            temporary_name=f".{kind}-{secrets.token_hex(16)}.tmp",
            destination_name=destination_name,
            payload=payload,
            maximum_size_bytes=self._settings.run_action_result_size_bytes,
            name=f"run action {kind} result",
        )
        return blob

    def _publish_request_locked(
        self,
        store_descriptor: int,
        request_blob: RunActionRequestBlob,
        payload: bytes,
    ) -> None:
        destination_name = f"input-{request_blob.digest.removeprefix('sha256:')}.blob"
        if os.access(
            destination_name,
            os.F_OK,
            dir_fd=store_descriptor,
            follow_symlinks=False,
        ):
            if (
                self._read_blob(
                    store_descriptor,
                    destination_name,
                    maximum_size_bytes=self._settings.run_action_request_size_bytes,
                    name="run action request",
                )
                != payload
            ):
                raise RunActionStoreError(
                    "existing run action input differs from its content name"
                )
            return
        _publish_create_only(
            store_descriptor,
            temporary_name=f".input-{secrets.token_hex(16)}.tmp",
            destination_name=destination_name,
            payload=payload,
            maximum_size_bytes=self._settings.run_action_request_size_bytes,
            name="run action request",
        )

    def _lock_registry(
        self,
        store_descriptor: int,
        descriptors: ExitStack,
    ) -> int:
        descriptor = os.open(
            "registry.lock",
            os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=store_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        metadata = _require_private_file(
            descriptor,
            mode=0o600,
            maximum_size_bytes=0,
            allow_empty=True,
            name="run action registry lock",
        )
        receipt = self._active_workspace.bootstrap_pin.installation_receipt
        if (metadata.st_dev, metadata.st_ino) != (
            receipt.run_action_registry_lock_device,
            receipt.run_action_registry_lock_inode,
        ):
            raise RunActionStoreError(
                "run action registry lock differs from its receipt"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor

    def _require_store_capacity(
        self,
        store_descriptor: int,
        *,
        additional_size_bytes: int,
        additional_entry_count: int,
    ) -> None:
        if (
            type(additional_size_bytes) is not int
            or additional_size_bytes < 0
            or type(additional_entry_count) is not int
            or additional_entry_count <= 0
        ):
            raise RunActionStoreError("run action capacity reservation is invalid")
        entry_count = 0
        total_size_bytes = 0
        with os.scandir(store_descriptor) as entries:
            for entry in entries:
                entry_count += 1
                metadata = entry.stat(follow_symlinks=False)
                total_size_bytes += metadata.st_size
        if (
            entry_count + additional_entry_count
            > self._settings.run_action_store_entry_limit
            or total_size_bytes + additional_size_bytes
            > self._settings.run_action_store_size_bytes
        ):
            raise RunActionStoreError(
                "run action store lacks capacity for another durable object"
            )

    def _clean_orphan_blobs(self, store_descriptor: int) -> None:
        event_names = []
        blob_names = []
        with os.scandir(store_descriptor) as entries:
            for entry in entries:
                if _EVENT_NAME_PATTERN.fullmatch(entry.name) is not None:
                    event_names.append(entry.name)
                elif any(
                    pattern.fullmatch(entry.name) is not None
                    for pattern in (
                        _INPUT_NAME_PATTERN,
                        _RESULT_NAME_PATTERN,
                        _ACCEPTED_NAME_PATTERN,
                    )
                ):
                    blob_names.append(entry.name)
        referenced = set()
        for event_name in sorted(event_names):
            payload = _read_file(
                store_descriptor,
                event_name,
                mode=0o400,
                maximum_size_bytes=self._settings.run_action_event_size_bytes,
                name_description="run action event",
            )
            event = RunActionExecutionEvent.from_json_bytes(payload)
            if event.to_json_bytes() != payload or _operation_digest(
                event.reservation.intent.operation_id
            ) != _EVENT_NAME_PATTERN.fullmatch(event_name).group("operation"):
                raise RunActionStoreError(
                    "run action event is invalid during orphan cleanup"
                )
            referenced.add(
                "input-"
                f"{event.reservation.request_blob.digest.removeprefix('sha256:')}.blob"
            )
            if event.result_receipt is not None:
                referenced.add(
                    "result-"
                    f"{event.result_receipt.result_blob.digest.removeprefix('sha256:')}.blob"
                )
            if event.acceptance is not None:
                referenced.add(
                    "accepted-"
                    f"{event.acceptance.accepted_result_blob.digest.removeprefix('sha256:')}.blob"
                )
        orphan_names = tuple(name for name in blob_names if name not in referenced)
        for orphan_name in orphan_names:
            pattern = (
                _INPUT_NAME_PATTERN
                if _INPUT_NAME_PATTERN.fullmatch(orphan_name) is not None
                else (
                    _RESULT_NAME_PATTERN
                    if _RESULT_NAME_PATTERN.fullmatch(orphan_name) is not None
                    else _ACCEPTED_NAME_PATTERN
                )
            )
            maximum_size_bytes = (
                self._settings.run_action_request_size_bytes
                if pattern is _INPUT_NAME_PATTERN
                else self._settings.run_action_result_size_bytes
            )
            _read_file(
                store_descriptor,
                orphan_name,
                mode=0o400,
                maximum_size_bytes=maximum_size_bytes,
                name_description="orphaned run action blob",
            )
            os.unlink(orphan_name, dir_fd=store_descriptor)
        if orphan_names:
            os.fsync(store_descriptor)

    def _clean_staging(self, store_descriptor: int) -> None:
        staging_names = []
        observed_entry_count = 0
        with os.scandir(store_descriptor) as entries:
            for entry in entries:
                observed_entry_count += 1
                if observed_entry_count > self._settings.run_action_store_entry_limit:
                    raise RunActionStoreError(
                        "run action store exceeds its configured entry limit"
                    )
                if _STAGING_NAME_PATTERN.fullmatch(entry.name) is not None:
                    staging_names.append(entry.name)
        for staging_name in staging_names:
            with ExitStack() as descriptors:
                descriptor = os.open(
                    staging_name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=store_descriptor,
                )
                descriptors.callback(os.close, descriptor)
                metadata = os.fstat(descriptor)
                rebound = os.stat(
                    staging_name,
                    dir_fd=store_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
                    or metadata.st_size
                    > max(
                        self._settings.run_action_event_size_bytes,
                        self._settings.run_action_request_size_bytes,
                        self._settings.run_action_result_size_bytes,
                    )
                    or (metadata.st_dev, metadata.st_ino)
                    != (rebound.st_dev, rebound.st_ino)
                ):
                    raise RunActionStoreError("run action staging entry is unsafe")
            os.unlink(staging_name, dir_fd=store_descriptor)
        if staging_names:
            os.fsync(store_descriptor)

    def _read_request(
        self,
        descriptors: ExitStack,
        request_blob: RunActionRequestBlob,
    ) -> bytes:
        if type(request_blob) is not RunActionRequestBlob:
            raise RunActionStoreError("run action request read requires one exact blob")
        store_descriptor, _identity = self._open_store(descriptors)
        payload = self._read_blob(
            store_descriptor,
            f"input-{request_blob.digest.removeprefix('sha256:')}.blob",
            maximum_size_bytes=self._settings.run_action_request_size_bytes,
            name="run action request",
        )
        if (
            len(payload) != request_blob.size_bytes
            or tree_or_blob_digest(payload) != request_blob.digest
        ):
            raise RunActionStoreError("run action request differs from its descriptor")
        return payload

    @staticmethod
    def _read_blob(
        store_descriptor: int,
        filename: str,
        *,
        maximum_size_bytes: int,
        name: str,
    ) -> bytes:
        return _read_file(
            store_descriptor,
            filename,
            mode=0o400,
            maximum_size_bytes=maximum_size_bytes,
            name_description=name,
        )

    def _read_result(
        self,
        descriptors: ExitStack,
        result_blob: RunActionResultBlob,
        *,
        kind: str,
    ) -> bytes:
        if type(result_blob) is not RunActionResultBlob or kind not in {
            "accepted",
            "result",
        }:
            raise RunActionStoreError(
                "run action result read requires one exact typed blob"
            )
        store_descriptor, _identity = self._open_store(descriptors)
        payload = _read_file(
            store_descriptor,
            f"{kind}-{result_blob.digest.removeprefix('sha256:')}.blob",
            mode=0o400,
            maximum_size_bytes=self._settings.run_action_result_size_bytes,
            name_description=f"run action {kind} result",
        )
        if (
            len(payload) != result_blob.size_bytes
            or tree_or_blob_digest(payload) != result_blob.digest
        ):
            raise RunActionStoreError(
                f"run action {kind} result differs from its descriptor"
            )
        return payload

    def _register_session(self, session: _RunActionExecutionSession) -> None:
        operation_id = session.reservation.intent.operation_id
        with self._registry_lock:
            if session._store is not self or operation_id in self._active_sessions:
                raise RunActionStoreError(
                    "run action operation already has a live session"
                )
            self._active_sessions[operation_id] = session

    def _unregister_session(self, session: _RunActionExecutionSession) -> None:
        operation_id = session.reservation.intent.operation_id
        with self._registry_lock:
            if self._active_sessions.get(operation_id) is not session:
                raise RunActionStoreError("run action session registration changed")
            del self._active_sessions[operation_id]

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise RunActionStoreError(
                "run action store cannot cross a process boundary"
            )

    @staticmethod
    def _event_name(operation_id: str, event_number: int) -> str:
        return (
            f"operation-{_operation_digest(operation_id)}"
            f"-event-{event_number:04d}.json"
        )


def _validate_event_prefix(events: tuple[RunActionExecutionEvent, ...]) -> None:
    if not events:
        return
    reservation = events[0].reservation
    expected_kinds = (
        RunActionExecutionEventKind.INTENT_RESERVED,
        RunActionExecutionEventKind.SPAWN_COMMITTED,
        RunActionExecutionEventKind.RESULT_RECEIVED,
        RunActionExecutionEventKind.RESULT_ACCEPTED,
    )
    previous_event_id = None
    for position, event in enumerate(events, start=1):
        if (
            event.event_number != position
            or event.predecessor_event_id != previous_event_id
            or event.reservation != reservation
        ):
            raise RunActionStoreError(
                "run action event chain changed identity or predecessor"
            )
        if position == 1:
            expected_kind = expected_kinds[0]
        elif events[-1].event_kind is RunActionExecutionEventKind.CANCELLED:
            expected_kind = RunActionExecutionEventKind.CANCELLED
        elif events[-1].event_kind is RunActionExecutionEventKind.INTERRUPTED:
            expected_kind = (
                expected_kinds[1]
                if position == 2
                else RunActionExecutionEventKind.INTERRUPTED
            )
        else:
            expected_kind = expected_kinds[position - 1]
        if event.event_kind is not expected_kind:
            raise RunActionStoreError(
                "run action event chain is not an admitted prefix"
            )
        previous_event_id = event.event_id
    if len(events) >= 2:
        spawn = events[1].spawn_commit
        if events[1].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED and (
            spawn.reservation_id != reservation.reservation_id
            or spawn.security_observation_id
            != reservation.frontier.security_observation_id
            or spawn.boundary_identity != reservation.intent.boundary_identity
        ):
            raise RunActionStoreError("run action spawn differs from its reservation")
    if len(events) >= 3 and events[2].event_kind is (
        RunActionExecutionEventKind.RESULT_RECEIVED
    ):
        spawn = events[1].spawn_commit
        result = events[2].result_receipt
        if (
            result.spawn_commit_id != spawn.spawn_commit_id
            or result.provider_execution_id != spawn.provider_execution_id
        ):
            raise RunActionStoreError("run action result differs from its spawn")
    if len(events) == 4:
        result = events[2].result_receipt
        acceptance = events[3].acceptance
        if acceptance.result_receipt_id != result.result_receipt_id:
            raise RunActionStoreError("run action acceptance differs from its result")


def _require_workspace_acceptance(
    access: RunFrontierWorkspaceAccess,
    before: RunActionWorkspaceBinding | None,
    after: RunActionWorkspaceBinding | None,
) -> None:
    if access is RunFrontierWorkspaceAccess.NONE:
        if before is not None or after is not None:
            raise RunActionStoreError(
                "workspace-free action acceptance carries workspace authority"
            )
        return
    if before is None or after is None:
        raise RunActionStoreError(
            "workspace action acceptance lacks a workspace frontier"
        )
    if access is RunFrontierWorkspaceAccess.READ_ONLY:
        if after != before:
            raise RunActionStoreError(
                "read-only action acceptance changed its workspace"
            )
        return
    if (
        after.branch != before.branch
        or after.commit_sha == before.commit_sha
        or after.parent_commit_shas != (before.commit_sha,)
        or (
            after.workspace_device,
            after.workspace_inode,
        )
        != (
            before.workspace_device,
            before.workspace_inode,
        )
    ):
        raise RunActionStoreError(
            "editing action acceptance lacks one direct workspace successor"
        )


def _require_interrupted_workspace(
    access: RunFrontierWorkspaceAccess,
    before: RunActionWorkspaceBinding | None,
    after: RunActionWorkspaceBinding | None,
) -> None:
    if access is RunFrontierWorkspaceAccess.NONE:
        if after is not None:
            raise RunActionStoreError(
                "workspace-free interruption carries workspace authority"
            )
        return
    if before is None:
        raise RunActionStoreError(
            "workspace interruption lacks its predecessor frontier"
        )
    if after is None:
        return
    if access is RunFrontierWorkspaceAccess.READ_ONLY:
        if after != before:
            raise RunActionStoreError(
                "interrupted read-only action changed its workspace"
            )
        return
    if after != before and (
        after.branch != before.branch
        or after.parent_commit_shas != (before.commit_sha,)
        or (
            after.workspace_device,
            after.workspace_inode,
        )
        != (
            before.workspace_device,
            before.workspace_inode,
        )
    ):
        raise RunActionStoreError(
            "interrupted edit has an unaccountable workspace frontier"
        )


def _operation_digest(operation_id: str) -> str:
    require_identifier(operation_id, "run action operation ID")
    return tree_or_blob_digest(operation_id.encode("utf-8")).removeprefix("sha256:")


def _require_namespaced_id(
    value: str,
    namespace: str,
    name: str,
) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionStoreError(f"{name} uses the wrong namespace")


def _require_private_file(
    descriptor: int,
    *,
    mode: int,
    maximum_size_bytes: int,
    allow_empty: bool,
    name: str,
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != mode
        or metadata.st_size > maximum_size_bytes
        or (not allow_empty and metadata.st_size == 0)
    ):
        raise RunActionStoreError(f"{name} is not one bounded private file")
    return metadata


def _read_file(
    store_descriptor: int,
    filename: str,
    *,
    mode: int,
    maximum_size_bytes: int,
    name_description: str,
) -> bytes:
    with ExitStack() as descriptors:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=store_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        metadata = _require_private_file(
            descriptor,
            mode=mode,
            maximum_size_bytes=maximum_size_bytes,
            allow_empty=False,
            name=name_description,
        )
        chunks = []
        observed_size_bytes = 0
        while observed_size_bytes <= maximum_size_bytes:
            chunk = os.read(
                descriptor,
                min(
                    1024 * 1024,
                    maximum_size_bytes + 1 - observed_size_bytes,
                ),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed_size_bytes += len(chunk)
        payload = b"".join(chunks)
        reopened = os.fstat(descriptor)
        rebound = os.stat(
            filename,
            dir_fd=store_descriptor,
            follow_symlinks=False,
        )
    if (
        len(payload) > maximum_size_bytes
        or (
            reopened.st_dev,
            reopened.st_ino,
            reopened.st_size,
            stat.S_IMODE(reopened.st_mode),
        )
        != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            stat.S_IMODE(metadata.st_mode),
        )
        or (rebound.st_dev, rebound.st_ino)
        != (
            metadata.st_dev,
            metadata.st_ino,
        )
    ):
        raise RunActionStoreError(f"{name_description} changed while reading")
    return payload


def _publish_create_only(
    store_descriptor: int,
    *,
    temporary_name: str,
    destination_name: str,
    payload: bytes,
    maximum_size_bytes: int,
    name: str,
) -> None:
    if type(payload) is not bytes or not payload or len(payload) > maximum_size_bytes:
        raise RunActionStoreError(f"{name} exceeds its configured bound")
    with ExitStack() as descriptors:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=store_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        written = 0
        while written < len(payload):
            chunk_size = os.write(descriptor, payload[written:])
            if chunk_size <= 0:
                raise RunActionStoreError(f"staging {name} made no write progress")
            written += chunk_size
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = _require_private_file(
            descriptor,
            mode=0o400,
            maximum_size_bytes=maximum_size_bytes,
            allow_empty=False,
            name=f"staged {name}",
        )
    if (
        _read_file(
            store_descriptor,
            temporary_name,
            mode=0o400,
            maximum_size_bytes=maximum_size_bytes,
            name_description=f"staged {name}",
        )
        != payload
    ):
        raise RunActionStoreError(f"staged {name} differs from its payload")
    _rename_no_replace(
        store_descriptor,
        temporary_name,
        destination_name,
    )
    os.fsync(store_descriptor)
    rebound = os.stat(
        destination_name,
        dir_fd=store_descriptor,
        follow_symlinks=False,
    )
    if (rebound.st_dev, rebound.st_ino) != (
        metadata.st_dev,
        metadata.st_ino,
    ):
        raise RunActionStoreError(f"published {name} changed inode")


def _rename_no_replace(
    directory_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(
        ctypes.c_int(directory_descriptor),
        ctypes.c_char_p(os.fsencode(source_name)),
        ctypes.c_int(directory_descriptor),
        ctypes.c_char_p(os.fsencode(destination_name)),
        ctypes.c_uint(_RENAME_NOREPLACE),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


__all__ = [
    "RunActionAcceptance",
    "RunActionExecutionEvent",
    "RunActionExecutionEventKind",
    "RunActionExecutionStore",
    "RunActionFrontierBinding",
    "RunActionLedgerSnapshot",
    "RunActionOperationTail",
    "RunActionReservation",
    "RunActionResultBlob",
    "RunActionResultDisposition",
    "RunActionResultReceipt",
    "RunActionSpawnCommit",
    "RunActionStoreInspection",
    "RunActionStoreError",
    "RunActionTerminalReason",
    "RunActionViewBinding",
    "RunActionWorkspaceBinding",
]
