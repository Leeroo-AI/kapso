"""Metadata-only checkpoint floor for durable run-action executions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_contracts import RunActionContractError

_MAXIMUM_EVENT_COUNT = 7


class RunActionLedgerError(RunActionContractError):
    """A run-action ledger snapshot is invalid or rolled back."""


class RunActionExecutionEventKind(str, Enum):
    """Exact phases admitted by one operation journal."""

    INTENT_RESERVED = "intent_reserved"
    PREPARATION_CLAIMED = "preparation_claimed"
    EXECUTION_PREPARED = "execution_prepared"
    SPAWN_COMMITTED = "spawn_committed"
    ACTIVATION_COMMITTED = "activation_committed"
    RESULT_RECEIVED = "result_received"
    RESULT_ACCEPTED = "result_accepted"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class RunActionOperationTail(StrictContract):
    """Metadata-only durable floor for one operation event prefix."""

    operation_id: str
    reservation_id: str
    event_ids: tuple[str, ...]
    tail_kind: RunActionExecutionEventKind

    def _validate(self) -> None:
        admitted_tail_kinds = {
            1: {RunActionExecutionEventKind.INTENT_RESERVED},
            2: {
                RunActionExecutionEventKind.PREPARATION_CLAIMED,
                RunActionExecutionEventKind.CANCELLED,
            },
            3: {
                RunActionExecutionEventKind.EXECUTION_PREPARED,
                RunActionExecutionEventKind.INTERRUPTED,
            },
            4: {
                RunActionExecutionEventKind.SPAWN_COMMITTED,
                RunActionExecutionEventKind.INTERRUPTED,
            },
            5: {
                RunActionExecutionEventKind.ACTIVATION_COMMITTED,
                RunActionExecutionEventKind.INTERRUPTED,
            },
            6: {
                RunActionExecutionEventKind.RESULT_RECEIVED,
                RunActionExecutionEventKind.INTERRUPTED,
            },
            7: {RunActionExecutionEventKind.RESULT_ACCEPTED},
        }
        require_identifier(self.operation_id, "run action tail operation ID")
        _require_namespaced_id(
            self.reservation_id,
            "run-action-reservation",
            "run action tail reservation",
        )
        if (
            type(self.tail_kind) is not RunActionExecutionEventKind
            or not 1 <= len(self.event_ids) <= _MAXIMUM_EVENT_COUNT
            or len(self.event_ids) != len(set(self.event_ids))
            or self.tail_kind not in admitted_tail_kinds[len(self.event_ids)]
        ):
            raise RunActionLedgerError("run action operation tail is invalid")
        for event_id in self.event_ids:
            _require_namespaced_id(
                event_id,
                "run-action-execution-event",
                "run action event floor",
            )

    @property
    def tail_event_id(self) -> str:
        return self.event_ids[-1]

    @property
    def event_count(self) -> int:
        return len(self.event_ids)


@dataclass(frozen=True)
class RunActionLedgerSnapshot(StrictContract):
    """Canonical metadata projection of all durable operation prefixes."""

    ledger_snapshot_id: str
    event_count: int
    operation_tails: tuple[RunActionOperationTail, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-ledger-snapshot"
    IDENTITY_FIELD: ClassVar[str] = "ledger_snapshot_id"

    def _validate(self) -> None:
        if (
            type(self.event_count) is not int
            or self.event_count < 0
            or self.event_count
            != sum(tail.event_count for tail in self.operation_tails)
            or any(
                type(tail) is not RunActionOperationTail
                for tail in self.operation_tails
            )
            or tuple(tail.operation_id for tail in self.operation_tails)
            != tuple(sorted({tail.operation_id for tail in self.operation_tails}))
        ):
            raise RunActionLedgerError("run action ledger snapshot is invalid")

    @classmethod
    def build(
        cls,
        operation_tails: tuple[RunActionOperationTail, ...],
    ) -> "RunActionLedgerSnapshot":
        ordered = tuple(sorted(operation_tails, key=lambda item: item.operation_id))
        return cls.mint(
            event_count=sum(tail.event_count for tail in ordered),
            operation_tails=ordered,
        )

    @classmethod
    def empty(cls) -> "RunActionLedgerSnapshot":
        return cls.build(())

    def require_predecessor(
        self,
        predecessor: "RunActionLedgerSnapshot",
    ) -> None:
        if type(predecessor) is not RunActionLedgerSnapshot:
            raise RunActionLedgerError(
                "run action ledger predecessor must be one exact snapshot"
            )
        current_by_operation = {
            tail.operation_id: tail for tail in self.operation_tails
        }
        terminal_kinds = {
            RunActionExecutionEventKind.RESULT_ACCEPTED,
            RunActionExecutionEventKind.CANCELLED,
            RunActionExecutionEventKind.INTERRUPTED,
        }
        for previous in predecessor.operation_tails:
            current = current_by_operation.get(previous.operation_id)
            if (
                current is None
                or current.reservation_id != previous.reservation_id
                or current.event_ids[: previous.event_count] != previous.event_ids
                or (previous.tail_kind in terminal_kinds and current != previous)
            ):
                raise RunActionLedgerError(
                    "run action ledger changed or extended a terminal predecessor"
                )


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionLedgerError(f"{name} uses the wrong namespace")


__all__ = [
    "RunActionExecutionEventKind",
    "RunActionLedgerError",
    "RunActionLedgerSnapshot",
    "RunActionOperationTail",
]
