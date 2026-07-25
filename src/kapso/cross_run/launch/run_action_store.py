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
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
    RunActionLedgerSnapshot,
    RunActionOperationTail,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionRequestBlob as _RunActionRequestBlob,
    RunActionReservation as _RunActionReservation,
    RunActionWorkspaceBinding as _RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_spawn_contracts import (
    RunActionSpawnCommit as _RunActionSpawnCommit,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionExecutionPolicy,
    RunActionActivationRevalidationReceipt,
    RunActionPreparationAllocation,
    RunActionPreparationClaim,
    RunActionPreparedExecution,
    RunActionResultCaptureReceipt,
    RunActionTerminalObservation,
    issue_runtime_volume_authority,
    run_action_terminal_result_evidence_matches,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.launch.workspace_frontier import RunWorkspaceFrontierIdentity
from kapso.cross_run.settings import LaunchSettings

_RENAME_NOREPLACE = 1
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVENT_NAME_PATTERN = re.compile(
    r"^operation-(?P<operation>[0-9a-f]{64})-event-(?P<number>[0-9]{4})[.]json$"
)
_RESULT_NAME_PATTERN = re.compile(r"^result-(?P<digest>[0-9a-f]{64})[.]blob$")
_ACCEPTED_NAME_PATTERN = re.compile(r"^accepted-(?P<digest>[0-9a-f]{64})[.]blob$")
_INPUT_NAME_PATTERN = re.compile(r"^input-(?P<digest>[0-9a-f]{64})[.]blob$")
_STAGING_NAME_PATTERN = re.compile(
    r"^[.](?P<kind>accepted|event|input|result)-[0-9a-f]{32}[.]tmp$"
)
_MAXIMUM_EVENT_COUNT = 7
_RUN_ACTION_STORE_AUTHORITY = object()
_RUN_ACTION_RESERVATION_AUTHORITY = object()
_RUN_ACTION_RECOVERY_AUTHORITY = object()
_TERMINAL_EVENT_KINDS = {
    RunActionExecutionEventKind.RESULT_ACCEPTED,
    RunActionExecutionEventKind.CANCELLED,
    RunActionExecutionEventKind.INTERRUPTED,
}
_FUTURE_EVENT_COUNT_BY_TAIL = {
    RunActionExecutionEventKind.INTENT_RESERVED: 6,
    RunActionExecutionEventKind.PREPARATION_ALLOCATED: 5,
    RunActionExecutionEventKind.EXECUTION_PREPARED: 4,
    RunActionExecutionEventKind.SPAWN_COMMITTED: 3,
    RunActionExecutionEventKind.ACTIVATION_COMMITTED: 2,
    RunActionExecutionEventKind.RESULT_RECEIVED: 1,
    RunActionExecutionEventKind.RESULT_ACCEPTED: 0,
    RunActionExecutionEventKind.CANCELLED: 0,
    RunActionExecutionEventKind.INTERRUPTED: 0,
}
_FUTURE_RESULT_BLOB_COUNT_BY_TAIL = {
    RunActionExecutionEventKind.INTENT_RESERVED: 2,
    RunActionExecutionEventKind.PREPARATION_ALLOCATED: 2,
    RunActionExecutionEventKind.EXECUTION_PREPARED: 2,
    RunActionExecutionEventKind.SPAWN_COMMITTED: 2,
    RunActionExecutionEventKind.ACTIVATION_COMMITTED: 2,
    RunActionExecutionEventKind.RESULT_RECEIVED: 1,
    RunActionExecutionEventKind.RESULT_ACCEPTED: 0,
    RunActionExecutionEventKind.CANCELLED: 0,
    RunActionExecutionEventKind.INTERRUPTED: 0,
}


def _issue_preparation_allocation(
    claim: RunActionPreparationClaim,
) -> RunActionPreparationAllocation:
    """Mint the sole random occurrence decision inside the durable store boundary."""

    if type(claim) is not RunActionPreparationClaim:
        raise RunActionStoreError(
            "preparation allocation requires an exact preparation claim"
        )
    return RunActionPreparationAllocation.mint(
        preparation_claim=claim,
        runtime_volume_authority=issue_runtime_volume_authority(
            claim,
            secrets.token_hex(16),
        ),
    )


class RunActionStoreError(RunActionContractError):
    """The durable run-action execution prefix is unsafe or conflicting."""


class RunActionTerminalReason(str, Enum):
    """Recovery or pre-spawn reasons that permanently close an operation."""

    STALE_FRONTIER = "stale_frontier"
    SUPERVISOR_RESOURCE_LOST_BEFORE_SPAWN = "supervisor_resource_lost_before_spawn"
    FRONTIER_INVALIDATED_BEFORE_SPAWN = "frontier_invalidated_before_spawn"
    PROVIDER_INTERRUPTED = "provider_interrupted"
    PROVIDER_FAILED = "provider_failed"


class RunActionResultDisposition(str, Enum):
    """Adapter-level meaning of one durably received provider response."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


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
    activation_revalidation_receipt_id: str
    terminal_observation: RunActionTerminalObservation
    result_capture_receipt: RunActionResultCaptureReceipt
    result_blob: RunActionResultBlob

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-result-receipt"
    IDENTITY_FIELD: ClassVar[str] = "result_receipt_id"

    def _validate(self) -> None:
        _require_namespaced_id(
            self.spawn_commit_id,
            _RunActionSpawnCommit.CONTENT_NAMESPACE,
            "run action result spawn commit",
        )
        require_identifier(
            self.provider_execution_id,
            "run action result provider execution ID",
        )
        _require_namespaced_id(
            self.activation_revalidation_receipt_id,
            RunActionActivationRevalidationReceipt.CONTENT_NAMESPACE,
            "run action result activation revalidation",
        )
        if (
            type(self.terminal_observation) is not RunActionTerminalObservation
            or type(self.result_capture_receipt) is not RunActionResultCaptureReceipt
            or self.terminal_observation.spawn_commit_id != self.spawn_commit_id
            or self.terminal_observation.provider_execution_id
            != self.provider_execution_id
            or self.terminal_observation.activation_revalidation_receipt_id
            != self.activation_revalidation_receipt_id
            or self.result_capture_receipt.terminal_observation_id
            != self.terminal_observation.terminal_observation_id
            or self.result_capture_receipt.runtime_volume_authority_id
            != self.terminal_observation.runtime_volume_authority_id
            or self.result_capture_receipt.generation_nonce
            != self.terminal_observation.generation_nonce
            or type(self.result_blob) is not RunActionResultBlob
            or self.result_capture_receipt.size_bytes != self.result_blob.size_bytes
            or self.result_capture_receipt.content_digest != self.result_blob.digest
        ):
            raise RunActionStoreError(
                "run action result receipt lacks exact terminal capture evidence"
            )


@dataclass(frozen=True)
class RunActionAcceptance(StrictContract):
    """Adapter acceptance and post-action workspace proof."""

    acceptance_id: str
    result_receipt_id: str
    disposition: RunActionResultDisposition
    accepted_result_blob: RunActionResultBlob
    workspace_after: _RunActionWorkspaceBinding | None

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
                and type(self.workspace_after) is not _RunActionWorkspaceBinding
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
    reservation: _RunActionReservation
    preparation_allocation: RunActionPreparationAllocation | None
    prepared_execution: RunActionPreparedExecution | None
    spawn_commit: _RunActionSpawnCommit | None
    activation_revalidation_receipt: RunActionActivationRevalidationReceipt | None
    result_receipt: RunActionResultReceipt | None
    acceptance: RunActionAcceptance | None
    terminal_reason: RunActionTerminalReason | None
    workspace_after: _RunActionWorkspaceBinding | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-execution-event"
    IDENTITY_FIELD: ClassVar[str] = "event_id"

    def _validate(self) -> None:
        if (
            type(self.event_number) is not int
            or not 1 <= self.event_number <= _MAXIMUM_EVENT_COUNT
            or (self.predecessor_event_id is None) != (self.event_number == 1)
            or type(self.event_kind) is not RunActionExecutionEventKind
            or type(self.reservation) is not _RunActionReservation
        ):
            raise RunActionStoreError("run action execution event prefix is invalid")
        if self.predecessor_event_id is not None:
            _require_namespaced_id(
                self.predecessor_event_id,
                self.CONTENT_NAMESPACE,
                "run action event predecessor",
            )
        optional_payloads = (
            (self.preparation_allocation, RunActionPreparationAllocation),
            (self.prepared_execution, RunActionPreparedExecution),
            (self.spawn_commit, _RunActionSpawnCommit),
            (
                self.activation_revalidation_receipt,
                RunActionActivationRevalidationReceipt,
            ),
            (self.result_receipt, RunActionResultReceipt),
            (self.acceptance, RunActionAcceptance),
            (self.terminal_reason, RunActionTerminalReason),
            (self.workspace_after, _RunActionWorkspaceBinding),
        )
        if any(
            payload is not None and type(payload) is not expected_type
            for payload, expected_type in optional_payloads
        ):
            raise RunActionStoreError(
                "run action execution event carries an invalid payload type"
            )
        shape = (
            self.preparation_allocation is not None,
            self.prepared_execution is not None,
            self.spawn_commit is not None,
            self.activation_revalidation_receipt is not None,
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
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.PREPARATION_ALLOCATED: (
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.EXECUTION_PREPARED: (
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.SPAWN_COMMITTED: (
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.ACTIVATION_COMMITTED: (
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.RESULT_RECEIVED: (
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ),
            RunActionExecutionEventKind.RESULT_ACCEPTED: (
                False,
                False,
                False,
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
            and (
                self.event_number != 2
                or self.terminal_reason is not RunActionTerminalReason.STALE_FRONTIER
            )
        ) or (
            self.event_kind is RunActionExecutionEventKind.INTERRUPTED
            and (
                (
                    self.event_number in {3, 4}
                    and self.terminal_reason
                    not in {
                        RunActionTerminalReason.SUPERVISOR_RESOURCE_LOST_BEFORE_SPAWN,
                        RunActionTerminalReason.FRONTIER_INVALIDATED_BEFORE_SPAWN,
                    }
                )
                or (
                    self.event_number in {5, 6}
                    and self.terminal_reason
                    not in {
                        RunActionTerminalReason.PROVIDER_INTERRUPTED,
                        RunActionTerminalReason.PROVIDER_FAILED,
                    }
                )
                or self.event_number not in {3, 4, 5, 6}
            )
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
    ) -> tuple[tuple[_RunActionWorkspaceBinding, _RunActionWorkspaceBinding], ...]:
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
        reservation: _RunActionReservation,
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

    def allocate_preparation(
        self,
        execution_policy: DockerRunActionExecutionPolicy,
    ) -> RunActionPreparationAllocation:
        """Durably issue the exact occurrence authority before Docker mutation."""
        self._require_tail(RunActionExecutionEventKind.INTENT_RESERVED)
        lifecycle = (
            self.reservation.intent.boundary_identity.execution_lifecycle_identity
        )
        if (
            type(execution_policy) is not DockerRunActionExecutionPolicy
            or execution_policy.kind is not self.reservation.intent.kind
            or execution_policy.docker_execution_policy_id
            != lifecycle.execution_policy_id
        ):
            raise RunActionStoreError(
                "run action preparation policy differs from its lifecycle"
            )
        claim = RunActionPreparationClaim.mint(
            reservation=self.reservation,
            execution_policy=execution_policy,
        )
        allocation = _issue_preparation_allocation(claim)
        self._append(
            self._event(
                RunActionExecutionEventKind.PREPARATION_ALLOCATED,
                preparation_allocation=allocation,
            )
        )
        return allocation

    def commit_prepared_execution(
        self,
        prepared_execution: RunActionPreparedExecution,
    ) -> RunActionPreparedExecution:
        """Persist the sole concrete inert occurrence for the durable allocation."""
        self._append(self._prepared_event(prepared_execution))
        return prepared_execution

    def _prepared_event_size_bytes(
        self,
        prepared_execution: RunActionPreparedExecution,
    ) -> int:
        """Measure the exact event that would index one prepared occurrence."""
        return len(self._prepared_event(prepared_execution).to_json_bytes())

    def _prepared_event(
        self,
        prepared_execution: RunActionPreparedExecution,
    ) -> RunActionExecutionEvent:
        self._require_tail(RunActionExecutionEventKind.PREPARATION_ALLOCATED)
        allocation = self._events[-1].preparation_allocation
        if (
            type(prepared_execution) is not RunActionPreparedExecution
            or prepared_execution.preparation_claim != allocation.preparation_claim
            or prepared_execution.runtime_volume_authority
            != allocation.runtime_volume_authority
        ):
            raise RunActionStoreError(
                "prepared run action execution differs from its durable allocation"
            )
        return self._event(
            RunActionExecutionEventKind.EXECUTION_PREPARED,
            prepared_execution=prepared_execution,
        )

    def commit_spawn(
        self,
        *,
        security_observation_id: str,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> _RunActionSpawnCommit:
        self._require_tail(RunActionExecutionEventKind.EXECUTION_PREPARED)
        if (
            type(boundary_identity) is not RunActionBoundaryIdentity
            or boundary_identity != self.reservation.intent.boundary_identity
            or security_observation_id
            != self.reservation.frontier.security_observation_id
        ):
            raise RunActionStoreError(
                "run action spawn security or boundary differs from its reservation"
            )
        prepared_execution = self._events[-1].prepared_execution
        spawn_commit = _RunActionSpawnCommit.build(
            reservation_id=self.reservation.reservation_id,
            prepared_execution_id=prepared_execution.prepared_execution_id,
            provider_execution_id=(
                prepared_execution.inert_container_evidence.container_id
            ),
            security_observation_id=security_observation_id,
            boundary_identity=boundary_identity,
        )
        event = self._event(
            RunActionExecutionEventKind.SPAWN_COMMITTED,
            spawn_commit=spawn_commit,
        )
        self._append(event)
        return spawn_commit

    def activation_event_size_bytes(
        self,
        activation_revalidation_receipt: RunActionActivationRevalidationReceipt,
    ) -> int:
        """Measure the exact durable activation selection before publication."""

        return len(
            self._activation_event(activation_revalidation_receipt).to_json_bytes()
        )

    def commit_activation(
        self,
        activation_revalidation_receipt: RunActionActivationRevalidationReceipt,
    ) -> RunActionExecutionEvent:
        """Select the sole revalidation receipt that may precede provider start."""

        event = self._activation_event(activation_revalidation_receipt)
        self._append(event)
        return event

    def _activation_event(
        self,
        activation_revalidation_receipt: RunActionActivationRevalidationReceipt,
    ) -> RunActionExecutionEvent:
        self._require_tail(RunActionExecutionEventKind.SPAWN_COMMITTED)
        prepared_execution = self._events[2].prepared_execution
        spawn_commit = self._events[3].spawn_commit
        if (
            type(activation_revalidation_receipt)
            is not RunActionActivationRevalidationReceipt
            or activation_revalidation_receipt.prepared_execution != prepared_execution
            or activation_revalidation_receipt.spawn_commit != spawn_commit
        ):
            raise RunActionStoreError(
                "run action activation differs from its durable spawn"
            )
        return self._event(
            RunActionExecutionEventKind.ACTIVATION_COMMITTED,
            activation_revalidation_receipt=activation_revalidation_receipt,
        )

    def record_result(
        self,
        *,
        spawn_commit: _RunActionSpawnCommit,
        terminal_observation: RunActionTerminalObservation,
        result_capture_receipt: RunActionResultCaptureReceipt,
        result_payload: bytes,
    ) -> RunActionResultReceipt:
        self._require_tail(RunActionExecutionEventKind.ACTIVATION_COMMITTED)
        durable_spawn = self._events[3].spawn_commit
        durable_activation = self._events[4].activation_revalidation_receipt
        if (
            type(spawn_commit) is not _RunActionSpawnCommit
            or spawn_commit != durable_spawn
            or type(terminal_observation) is not RunActionTerminalObservation
            or type(result_capture_receipt) is not RunActionResultCaptureReceipt
            or terminal_observation.spawn_commit_id != spawn_commit.spawn_commit_id
            or terminal_observation.provider_execution_id
            != spawn_commit.provider_execution_id
            or result_capture_receipt.terminal_observation_id
            != terminal_observation.terminal_observation_id
            or terminal_observation.activation_revalidation_receipt_id
            != durable_activation.activation_revalidation_receipt_id
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
            activation_revalidation_receipt_id=(
                durable_activation.activation_revalidation_receipt_id
            ),
            terminal_observation=terminal_observation,
            result_capture_receipt=result_capture_receipt,
            result_blob=result_blob,
        )
        event = self._event(
            RunActionExecutionEventKind.RESULT_RECEIVED,
            result_receipt=result_receipt,
        )
        _validate_event_prefix((*self._events, event))
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
            else _RunActionWorkspaceBinding.from_identity(workspace_after)
        )
        _require_workspace_acceptance(
            self.reservation.intent.workspace_access,
            disposition,
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
        _validate_event_prefix((*self._events, event))
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
        self._require_active()
        if self.events[-1].event_kind not in {
            RunActionExecutionEventKind.SPAWN_COMMITTED,
            RunActionExecutionEventKind.ACTIVATION_COMMITTED,
        }:
            raise RunActionStoreError(
                "provider interruption requires committed spawn or activation"
            )
        if reason not in {
            RunActionTerminalReason.PROVIDER_INTERRUPTED,
            RunActionTerminalReason.PROVIDER_FAILED,
        }:
            raise RunActionStoreError("run action interruption reason is invalid")
        after = (
            None
            if workspace_after is None
            else _RunActionWorkspaceBinding.from_identity(workspace_after)
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

    def interrupt_pre_spawn(
        self,
        *,
        reason: RunActionTerminalReason,
    ) -> None:
        """Close allocated or prepared work before any provider spend."""
        self._require_active()
        if self.events[-1].event_kind not in {
            RunActionExecutionEventKind.PREPARATION_ALLOCATED,
            RunActionExecutionEventKind.EXECUTION_PREPARED,
        }:
            raise RunActionStoreError(
                "pre-spawn interruption requires allocated or prepared work"
            )
        if reason not in {
            RunActionTerminalReason.SUPERVISOR_RESOURCE_LOST_BEFORE_SPAWN,
            RunActionTerminalReason.FRONTIER_INVALIDATED_BEFORE_SPAWN,
        }:
            raise RunActionStoreError("pre-spawn interruption reason is invalid")
        after = self.reservation.frontier.workspace_before
        _require_unchanged_pre_spawn_workspace(
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
        if (
            not self._events
            or self._events[-1].event_kind
            not in {
                RunActionExecutionEventKind.SPAWN_COMMITTED,
                RunActionExecutionEventKind.ACTIVATION_COMMITTED,
                RunActionExecutionEventKind.RESULT_RECEIVED,
                RunActionExecutionEventKind.RESULT_ACCEPTED,
                RunActionExecutionEventKind.INTERRUPTED,
            }
            or (
                self._events[-1].event_kind is RunActionExecutionEventKind.INTERRUPTED
                and len(self._events) < 5
            )
        ):
            raise RunActionStoreError(
                "run action request is unavailable before spawn commitment"
            )
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
        preparation_allocation: RunActionPreparationAllocation | None = None,
        prepared_execution: RunActionPreparedExecution | None = None,
        spawn_commit: _RunActionSpawnCommit | None = None,
        activation_revalidation_receipt: (
            RunActionActivationRevalidationReceipt | None
        ) = None,
        result_receipt: RunActionResultReceipt | None = None,
        acceptance: RunActionAcceptance | None = None,
        terminal_reason: RunActionTerminalReason | None = None,
        workspace_after: _RunActionWorkspaceBinding | None = None,
    ) -> RunActionExecutionEvent:
        return RunActionExecutionEvent.mint(
            event_number=len(self._events) + 1,
            predecessor_event_id=(
                None if not self._events else self._events[-1].event_id
            ),
            event_kind=event_kind,
            reservation=self.reservation,
            preparation_allocation=preparation_allocation,
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            activation_revalidation_receipt=activation_revalidation_receipt,
            result_receipt=result_receipt,
            acceptance=acceptance,
            terminal_reason=terminal_reason,
            workspace_after=workspace_after,
        )

    def _append(self, event: RunActionExecutionEvent) -> None:
        _validate_event_prefix((*self._events, event))
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
        reservation: _RunActionReservation,
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
            if not events:
                raise RunActionStoreError(
                    "run action recovery requires a durable reservation"
                )
            if events[0].reservation != self.reservation:
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
            self._prepare_store_locked(store_descriptor)

    def _reserve_action(
        self,
        reservation: _RunActionReservation,
        request_payload: bytes,
        *,
        _authority: object,
    ) -> RunActionExecutionEvent:
        self._require_owner_process()
        if (
            type(reservation) is not _RunActionReservation
            or _authority is not _RUN_ACTION_RESERVATION_AUTHORITY
        ):
            raise RunActionStoreError(
                "run action reservation requires sealed reservation authority"
            )
        if (
            type(request_payload) is not bytes
            or not request_payload
            or tree_or_blob_digest(request_payload) != reservation.request_blob.digest
            or len(request_payload) != reservation.request_blob.size_bytes
        ):
            raise RunActionStoreError(
                "run action reservation requires its complete request bytes"
            )
        event = RunActionExecutionEvent.mint(
            event_number=1,
            predecessor_event_id=None,
            event_kind=RunActionExecutionEventKind.INTENT_RESERVED,
            reservation=reservation,
            preparation_allocation=None,
            prepared_execution=None,
            spawn_commit=None,
            activation_revalidation_receipt=None,
            result_receipt=None,
            acceptance=None,
            terminal_reason=None,
            workspace_after=None,
        )
        with ExitStack() as descriptors:
            self._reserve(
                descriptors,
                reservation,
                request_payload,
                event,
            )
        return event

    def _recovery_session(
        self,
        reservation: _RunActionReservation,
        *,
        _authority: object,
    ) -> _RunActionSessionContext:
        self._require_owner_process()
        if (
            type(reservation) is not _RunActionReservation
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
        _event_names, ledger = self._prepare_store_locked(store_descriptor)
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

    def _validate_store(
        self,
        store_descriptor: int,
        *,
        before_staging_cleanup: bool,
    ) -> tuple[str, ...]:
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
            or (
                not before_staging_cleanup
                and staging_entry_count > self._settings.run_action_staging_entry_limit
            )
            or total_size_bytes > self._settings.run_action_store_size_bytes
            or "registry.lock" not in names
            or "workspace.lock" not in names
        ):
            raise RunActionStoreError(
                "run action store exceeds bounds or lacks fixed locks"
            )
        return tuple(sorted(event_names))

    def _open_operation(
        self,
        store_descriptor: int,
        operation_id: str,
        descriptors: ExitStack,
    ) -> tuple[RunActionExecutionEvent, ...]:
        first_event_name = self._event_name(operation_id, 1)
        with ExitStack() as registry_descriptors:
            self._lock_registry(store_descriptor, registry_descriptors)
            self._prepare_store_locked(store_descriptor)
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
        preparation_claim_ids = set()
        preparation_allocation_ids = set()
        allocated_volume_authority_ids = set()
        allocated_volume_names = set()
        allocated_generation_nonces = set()
        allocated_sentinel_identities = set()
        prepared_execution_ids = set()
        prepared_container_ids = set()
        prepared_container_names = set()
        prepared_keeper_container_ids = set()
        prepared_keeper_container_names = set()
        prepared_file_ids = set()
        provider_execution_ids = set()
        invocation_nonces = set()
        activation_revalidation_receipt_ids = set()
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
                RunActionExecutionEventKind.PREPARATION_ALLOCATED
            ):
                allocation = events[1].preparation_allocation
                claim_id = allocation.preparation_claim.preparation_claim_id
                volume = allocation.runtime_volume_authority
                if (
                    claim_id in preparation_claim_ids
                    or allocation.preparation_allocation_id
                    in preparation_allocation_ids
                    or volume.runtime_volume_authority_id
                    in allocated_volume_authority_ids
                    or volume.volume_name in allocated_volume_names
                    or volume.generation_nonce in allocated_generation_nonces
                    or volume.sentinel_identity in allocated_sentinel_identities
                ):
                    raise RunActionStoreError(
                        "run action preparation allocation authority was reused"
                    )
                preparation_claim_ids.add(claim_id)
                preparation_allocation_ids.add(allocation.preparation_allocation_id)
                allocated_volume_authority_ids.add(volume.runtime_volume_authority_id)
                allocated_volume_names.add(volume.volume_name)
                allocated_generation_nonces.add(volume.generation_nonce)
                allocated_sentinel_identities.add(volume.sentinel_identity)
            if (
                len(events) >= 3
                and events[2].event_kind
                is RunActionExecutionEventKind.EXECUTION_PREPARED
            ):
                prepared = events[2].prepared_execution
                prepared_files = tuple(
                    prepared_file
                    for prepared_file in (
                        prepared.input_file,
                        prepared.result_file,
                        prepared.credential_file,
                    )
                    if prepared_file is not None
                )
                file_ids = {
                    prepared_file.prepared_file_id for prepared_file in prepared_files
                }
                evidence = prepared.inert_container_evidence
                keeper = prepared.volume_keeper_evidence
                if (
                    prepared.prepared_execution_id in prepared_execution_ids
                    or evidence.container_id in prepared_container_ids
                    or evidence.container_id in prepared_keeper_container_ids
                    or evidence.container_name in prepared_container_names
                    or evidence.container_name in prepared_keeper_container_names
                    or keeper.container_id in prepared_keeper_container_ids
                    or keeper.container_id in prepared_container_ids
                    or keeper.container_name in prepared_keeper_container_names
                    or keeper.container_name in prepared_container_names
                    or prepared_file_ids & file_ids
                ):
                    raise RunActionStoreError(
                        "run action prepared occurrence authority was reused"
                    )
                prepared_execution_ids.add(prepared.prepared_execution_id)
                prepared_container_ids.add(evidence.container_id)
                prepared_container_names.add(evidence.container_name)
                prepared_keeper_container_ids.add(keeper.container_id)
                prepared_keeper_container_names.add(keeper.container_name)
                prepared_file_ids.update(file_ids)
            if len(events) >= 4 and events[3].event_kind is (
                RunActionExecutionEventKind.SPAWN_COMMITTED
            ):
                spawn = events[3].spawn_commit
                if (
                    spawn.provider_execution_id in provider_execution_ids
                    or spawn.invocation_nonce in invocation_nonces
                ):
                    raise RunActionStoreError(
                        "run action provider execution identity was reused"
                    )
                provider_execution_ids.add(spawn.provider_execution_id)
                invocation_nonces.add(spawn.invocation_nonce)
            if len(events) >= 5 and events[4].event_kind is (
                RunActionExecutionEventKind.ACTIVATION_COMMITTED
            ):
                activation_id = events[
                    4
                ].activation_revalidation_receipt.activation_revalidation_receipt_id
                if activation_id in activation_revalidation_receipt_ids:
                    raise RunActionStoreError(
                        "run action activation receipt identity was reused"
                    )
                activation_revalidation_receipt_ids.add(activation_id)
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
        snapshot = RunActionLedgerSnapshot.build(tuple(tails))
        if (
            sum(
                tail.tail_kind not in _TERMINAL_EVENT_KINDS
                for tail in snapshot.operation_tails
            )
            > 1
        ):
            raise RunActionStoreError(
                "run action store has multiple nonterminal operations"
            )
        return snapshot

    def _reserve(
        self,
        descriptors: ExitStack,
        reservation: _RunActionReservation,
        request_payload: bytes,
        event: RunActionExecutionEvent,
    ) -> None:
        store_descriptor, _identity = self._open_store(descriptors)
        with ExitStack() as registry_descriptors:
            self._lock_registry(store_descriptor, registry_descriptors)
            _event_names, snapshot = self._prepare_store_locked(store_descriptor)
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
            if any(
                tail.tail_kind not in _TERMINAL_EVENT_KINDS
                for tail in snapshot.operation_tails
            ):
                raise RunActionStoreError(
                    "run action store already has a nonterminal operation"
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
                prospective_tail_kind=event.event_kind,
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
            event_names, _snapshot = self._prepare_store_locked(store_descriptor)
            self._require_unique_allocation_authority(
                store_descriptor,
                event_names,
                event=event,
            )
            self._require_unique_prepared_authority(
                store_descriptor,
                event_names,
                event=event,
            )
            self._require_unique_spawn_identity(
                store_descriptor,
                event_names,
                event=event,
            )
            self._require_unique_activation_identity(
                store_descriptor,
                event_names,
                event=event,
            )
            self._require_store_capacity(
                store_descriptor,
                additional_size_bytes=len(event.to_json_bytes()),
                additional_entry_count=1,
                prospective_tail_kind=event.event_kind,
            )
            self._publish_event_locked(
                store_descriptor,
                operation_id,
                event,
            )

    def _require_unique_allocation_authority(
        self,
        store_descriptor: int,
        event_names: tuple[str, ...],
        *,
        event: RunActionExecutionEvent,
    ) -> None:
        candidate = event.preparation_allocation
        if candidate is None:
            return
        candidate_volume = candidate.runtime_volume_authority
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
                is not RunActionExecutionEventKind.PREPARATION_ALLOCATED
            ):
                continue
            existing = events[1].preparation_allocation
            existing_volume = existing.runtime_volume_authority
            if (
                existing.preparation_allocation_id
                == candidate.preparation_allocation_id
                or existing.preparation_claim.preparation_claim_id
                == candidate.preparation_claim.preparation_claim_id
                or existing_volume.runtime_volume_authority_id
                == candidate_volume.runtime_volume_authority_id
                or existing_volume.volume_name == candidate_volume.volume_name
                or existing_volume.generation_nonce == candidate_volume.generation_nonce
                or existing_volume.sentinel_identity
                == candidate_volume.sentinel_identity
            ):
                raise RunActionStoreError(
                    "run action preparation allocation authority was reused"
                )

    def _require_unique_prepared_authority(
        self,
        store_descriptor: int,
        event_names: tuple[str, ...],
        *,
        event: RunActionExecutionEvent,
    ) -> None:
        candidate = event.prepared_execution
        if candidate is None:
            return
        candidate_files = tuple(
            prepared_file
            for prepared_file in (
                candidate.input_file,
                candidate.result_file,
                candidate.credential_file,
            )
            if prepared_file is not None
        )
        candidate_file_ids = {
            prepared_file.prepared_file_id for prepared_file in candidate_files
        }
        for tail in self._snapshot_from_event_names(
            store_descriptor,
            event_names,
        ).operation_tails:
            events = self._read_operation_events(
                store_descriptor,
                tail.operation_id,
            )
            if (
                len(events) < 3
                or events[2].event_kind
                is not RunActionExecutionEventKind.EXECUTION_PREPARED
            ):
                continue
            existing = events[2].prepared_execution
            existing_files = tuple(
                prepared_file
                for prepared_file in (
                    existing.input_file,
                    existing.result_file,
                    existing.credential_file,
                )
                if prepared_file is not None
            )
            if (
                existing.prepared_execution_id == candidate.prepared_execution_id
                or existing.inert_container_evidence.container_id
                == candidate.inert_container_evidence.container_id
                or existing.inert_container_evidence.container_name
                == candidate.inert_container_evidence.container_name
                or existing.inert_container_evidence.container_id
                == candidate.volume_keeper_evidence.container_id
                or existing.inert_container_evidence.container_name
                == candidate.volume_keeper_evidence.container_name
                or existing.volume_keeper_evidence.container_id
                == candidate.volume_keeper_evidence.container_id
                or existing.volume_keeper_evidence.container_name
                == candidate.volume_keeper_evidence.container_name
                or existing.volume_keeper_evidence.container_id
                == candidate.inert_container_evidence.container_id
                or existing.volume_keeper_evidence.container_name
                == candidate.inert_container_evidence.container_name
                or existing.runtime_volume_authority.runtime_volume_authority_id
                == candidate.runtime_volume_authority.runtime_volume_authority_id
                or existing.runtime_volume_authority.volume_name
                == candidate.runtime_volume_authority.volume_name
                or existing.runtime_volume_authority.generation_nonce
                == candidate.runtime_volume_authority.generation_nonce
                or existing.runtime_volume_authority.sentinel_identity
                == candidate.runtime_volume_authority.sentinel_identity
                or candidate_file_ids
                & {prepared_file.prepared_file_id for prepared_file in existing_files}
            ):
                raise RunActionStoreError(
                    "run action prepared occurrence authority was reused"
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

    def _require_unique_activation_identity(
        self,
        store_descriptor: int,
        event_names: tuple[str, ...],
        *,
        event: RunActionExecutionEvent,
    ) -> None:
        candidate = event.activation_revalidation_receipt
        if candidate is None:
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
                len(events) >= 5
                and events[4].event_kind
                is RunActionExecutionEventKind.ACTIVATION_COMMITTED
                and events[
                    4
                ].activation_revalidation_receipt.activation_revalidation_receipt_id
                == candidate.activation_revalidation_receipt_id
            ):
                raise RunActionStoreError(
                    "run action activation receipt identity was reused"
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
            event_names, _snapshot = self._prepare_store_locked(store_descriptor)
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
                prospective_tail_kind=event.event_kind,
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
                len(events) < 4
                or events[3].event_kind
                is not RunActionExecutionEventKind.SPAWN_COMMITTED
            ):
                continue
            spawn = events[3].spawn_commit
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
        request_blob: _RunActionRequestBlob,
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
        prospective_tail_kind: RunActionExecutionEventKind,
    ) -> None:
        if (
            type(additional_size_bytes) is not int
            or additional_size_bytes < 0
            or type(additional_entry_count) is not int
            or additional_entry_count < 0
            or type(prospective_tail_kind) is not RunActionExecutionEventKind
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
            entry_count
            + additional_entry_count
            + _FUTURE_EVENT_COUNT_BY_TAIL[prospective_tail_kind]
            + _FUTURE_RESULT_BLOB_COUNT_BY_TAIL[prospective_tail_kind]
            > self._settings.run_action_store_entry_limit
            or total_size_bytes
            + additional_size_bytes
            + (
                _FUTURE_EVENT_COUNT_BY_TAIL[prospective_tail_kind]
                * self._settings.run_action_event_size_bytes
            )
            + (
                _FUTURE_RESULT_BLOB_COUNT_BY_TAIL[prospective_tail_kind]
                * self._settings.run_action_result_size_bytes
            )
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

    def _prepare_store_locked(
        self,
        store_descriptor: int,
    ) -> tuple[tuple[str, ...], RunActionLedgerSnapshot]:
        event_names = self._validate_store(
            store_descriptor,
            before_staging_cleanup=True,
        )
        snapshot = self._snapshot_from_event_names(
            store_descriptor,
            event_names,
        )
        self._clean_staging(store_descriptor)
        self._clean_orphan_blobs(store_descriptor)
        cleaned_event_names = self._validate_store(
            store_descriptor,
            before_staging_cleanup=False,
        )
        if cleaned_event_names != event_names:
            raise RunActionStoreError(
                "run action cleanup changed the durable event set"
            )
        nonterminal_tails = tuple(
            tail
            for tail in snapshot.operation_tails
            if tail.tail_kind not in _TERMINAL_EVENT_KINDS
        )
        self._require_store_capacity(
            store_descriptor,
            additional_size_bytes=0,
            additional_entry_count=0,
            prospective_tail_kind=(
                RunActionExecutionEventKind.RESULT_ACCEPTED
                if not nonterminal_tails
                else nonterminal_tails[0].tail_kind
            ),
        )
        return cleaned_event_names, snapshot

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
        request_blob: _RunActionRequestBlob,
    ) -> bytes:
        if type(request_blob) is not _RunActionRequestBlob:
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
    normal_kinds = (
        RunActionExecutionEventKind.INTENT_RESERVED,
        RunActionExecutionEventKind.PREPARATION_ALLOCATED,
        RunActionExecutionEventKind.EXECUTION_PREPARED,
        RunActionExecutionEventKind.SPAWN_COMMITTED,
        RunActionExecutionEventKind.ACTIVATION_COMMITTED,
        RunActionExecutionEventKind.RESULT_RECEIVED,
        RunActionExecutionEventKind.RESULT_ACCEPTED,
    )
    if events[-1].event_kind is RunActionExecutionEventKind.CANCELLED:
        expected_kinds = (
            RunActionExecutionEventKind.INTENT_RESERVED,
            RunActionExecutionEventKind.CANCELLED,
        )
    elif events[-1].event_kind is RunActionExecutionEventKind.INTERRUPTED:
        expected_kinds = (
            *normal_kinds[: len(events) - 1],
            RunActionExecutionEventKind.INTERRUPTED,
        )
    else:
        expected_kinds = normal_kinds[: len(events)]
    if len(expected_kinds) != len(events):
        raise RunActionStoreError("run action event chain is not an admitted prefix")
    previous_event_id = None
    for position, event in enumerate(events, start=1):
        if (
            event.event_number != position
            or event.predecessor_event_id != previous_event_id
            or event.reservation != reservation
            or event.event_kind is not expected_kinds[position - 1]
        ):
            raise RunActionStoreError(
                "run action event chain changed identity or predecessor"
            )
        previous_event_id = event.event_id
    if len(events) >= 2 and events[1].event_kind is (
        RunActionExecutionEventKind.PREPARATION_ALLOCATED
    ):
        allocation = events[1].preparation_allocation
        if allocation.preparation_claim.reservation != reservation:
            raise RunActionStoreError(
                "run action preparation allocation differs from its reservation"
            )
    if (
        len(events) >= 3
        and events[2].event_kind is RunActionExecutionEventKind.EXECUTION_PREPARED
    ):
        allocation = events[1].preparation_allocation
        prepared = events[2].prepared_execution
        if (
            prepared.preparation_claim != allocation.preparation_claim
            or prepared.runtime_volume_authority != allocation.runtime_volume_authority
        ):
            raise RunActionStoreError(
                "prepared run action execution differs from its allocation"
            )
    if (
        len(events) >= 4
        and events[3].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED
    ):
        prepared = events[2].prepared_execution
        spawn = events[3].spawn_commit
        if (
            spawn.reservation_id != reservation.reservation_id
            or spawn.security_observation_id
            != reservation.frontier.security_observation_id
            or spawn.boundary_identity != reservation.intent.boundary_identity
            or spawn.prepared_execution_id != prepared.prepared_execution_id
            or spawn.provider_execution_id
            != prepared.inert_container_evidence.container_id
        ):
            raise RunActionStoreError("run action spawn differs from its reservation")
    if len(events) >= 5 and events[4].event_kind is (
        RunActionExecutionEventKind.ACTIVATION_COMMITTED
    ):
        prepared = events[2].prepared_execution
        spawn = events[3].spawn_commit
        activation = events[4].activation_revalidation_receipt
        if (
            activation.prepared_execution != prepared
            or activation.spawn_commit != spawn
        ):
            raise RunActionStoreError("run action activation differs from its spawn")
    if len(events) >= 6 and events[5].event_kind is (
        RunActionExecutionEventKind.RESULT_RECEIVED
    ):
        prepared = events[2].prepared_execution
        spawn = events[3].spawn_commit
        activation = events[4].activation_revalidation_receipt
        result = events[5].result_receipt
        if (
            result.spawn_commit_id != spawn.spawn_commit_id
            or result.provider_execution_id != spawn.provider_execution_id
            or result.activation_revalidation_receipt_id
            != activation.activation_revalidation_receipt_id
            or result.terminal_observation.prepared_execution_id
            != prepared.prepared_execution_id
            or result.terminal_observation.spawn_commit_id != spawn.spawn_commit_id
            or result.terminal_observation.provider_execution_id
            != spawn.provider_execution_id
            or result.terminal_observation.runtime_volume_authority_id
            != prepared.runtime_volume_authority.runtime_volume_authority_id
            or result.terminal_observation.generation_nonce
            != prepared.runtime_volume_authority.generation_nonce
            or result.terminal_observation.observed_inspect_projection
            != prepared.inert_container_evidence.issued_create_projection
            or not run_action_terminal_result_evidence_matches(
                result.terminal_observation,
                result.result_capture_receipt,
                activation,
            )
        ):
            raise RunActionStoreError("run action result differs from its spawn")
    if len(events) == 7:
        result = events[5].result_receipt
        acceptance = events[6].acceptance
        if acceptance.result_receipt_id != result.result_receipt_id:
            raise RunActionStoreError("run action acceptance differs from its result")
        _require_workspace_acceptance(
            reservation.intent.workspace_access,
            acceptance.disposition,
            reservation.frontier.workspace_before,
            acceptance.workspace_after,
        )
    elif events[-1].event_kind is RunActionExecutionEventKind.INTERRUPTED:
        if len(events) < 5:
            if events[-1].terminal_reason not in {
                RunActionTerminalReason.SUPERVISOR_RESOURCE_LOST_BEFORE_SPAWN,
                RunActionTerminalReason.FRONTIER_INVALIDATED_BEFORE_SPAWN,
            }:
                raise RunActionStoreError(
                    "pre-spawn interruption uses a provider terminal reason"
                )
            _require_unchanged_pre_spawn_workspace(
                reservation.intent.workspace_access,
                reservation.frontier.workspace_before,
                events[-1].workspace_after,
            )
        else:
            if events[-1].terminal_reason not in {
                RunActionTerminalReason.PROVIDER_INTERRUPTED,
                RunActionTerminalReason.PROVIDER_FAILED,
            }:
                raise RunActionStoreError(
                    "provider interruption uses a pre-spawn terminal reason"
                )
            _require_interrupted_workspace(
                reservation.intent.workspace_access,
                reservation.frontier.workspace_before,
                events[-1].workspace_after,
            )


def _require_workspace_acceptance(
    access: RunFrontierWorkspaceAccess,
    disposition: RunActionResultDisposition,
    before: _RunActionWorkspaceBinding | None,
    after: _RunActionWorkspaceBinding | None,
) -> None:
    if type(disposition) is not RunActionResultDisposition:
        raise RunActionStoreError(
            "workspace acceptance lacks one exact result disposition"
        )
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
    if disposition is RunActionResultDisposition.FAILED:
        if after != before:
            raise RunActionStoreError(
                "failed editing action acceptance changed its workspace"
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
    before: _RunActionWorkspaceBinding | None,
    after: _RunActionWorkspaceBinding | None,
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


def _require_unchanged_pre_spawn_workspace(
    access: RunFrontierWorkspaceAccess,
    before: _RunActionWorkspaceBinding | None,
    after: _RunActionWorkspaceBinding | None,
) -> None:
    if access is RunFrontierWorkspaceAccess.NONE:
        if before is not None or after is not None:
            raise RunActionStoreError(
                "workspace-free pre-spawn interruption carries workspace authority"
            )
        return
    if before is None or after != before:
        raise RunActionStoreError(
            "pre-spawn interruption lacks its exact unchanged workspace"
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
    "RunActionLedgerSnapshot",
    "RunActionOperationTail",
    "RunActionResultBlob",
    "RunActionResultDisposition",
    "RunActionResultReceipt",
    "RunActionStoreInspection",
    "RunActionStoreError",
    "RunActionTerminalReason",
]
