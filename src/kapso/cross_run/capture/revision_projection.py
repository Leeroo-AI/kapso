"""Pure canonical projection of an execution-revision journal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_utc_timestamp,
    require_identifier,
)
from kapso.cross_run.contracts import (
    EpisodeEvaluationStatus,
    ExecutionStatus,
)
from kapso.cross_run.record_contracts import (
    EXECUTION_REVISION_EVENT_SCHEMA,
    ExecutionRevisionEvent,
)


class RevisionProjectionError(ValueError):
    """An execution-revision sequence or its canonical bytes are invalid."""


class RevisionProjectionConflictError(RevisionProjectionError):
    """A projected event conflicts with an immutable journal revision."""


@dataclass(frozen=True)
class ExecutionRevisionProjection:
    """One immutable, identity-bound execution-revision event sequence."""

    run_id: str
    campaign_id: str
    require_contiguous_node_ids: bool
    events: tuple[ExecutionRevisionEvent, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        require_identifier(self.campaign_id, "campaign_id")
        if type(self.require_contiguous_node_ids) is not bool:
            raise RevisionProjectionError(
                "execution revision node-identity policy must be explicit"
            )
        if type(self.events) is not tuple or any(
            type(event) is not ExecutionRevisionEvent for event in self.events
        ):
            raise RevisionProjectionError(
                "execution revision events must be an immutable event tuple"
            )
        self._validate_sequence()

    @classmethod
    def from_jsonl_bytes(
        cls,
        payload: bytes,
        *,
        run_id: str,
        campaign_id: str,
        require_contiguous_node_ids: bool,
    ) -> ExecutionRevisionProjection:
        """Parse only the exact canonical JSONL representation."""
        if not isinstance(payload, bytes):
            raise RevisionProjectionError("execution revision payload must be bytes")
        if not payload:
            return cls(
                run_id=run_id,
                campaign_id=campaign_id,
                require_contiguous_node_ids=require_contiguous_node_ids,
            )
        if not payload.endswith(b"\n"):
            raise RevisionProjectionError(
                "execution revision payload has an incomplete tail"
            )
        lines = payload.split(b"\n")[:-1]
        if any(not line for line in lines):
            raise RevisionProjectionError(
                "execution revision payload contains a blank event"
            )
        events = tuple(ExecutionRevisionEvent.from_json_bytes(line) for line in lines)
        for line, event in zip(lines, events, strict=True):
            if line != canonical_json_bytes(event.to_dict()):
                raise RevisionProjectionError(
                    "execution revision event is not canonical JSON"
                )
        projection = cls(
            run_id=run_id,
            campaign_id=campaign_id,
            require_contiguous_node_ids=require_contiguous_node_ids,
            events=events,
        )
        if projection.jsonl_bytes != payload:
            raise RevisionProjectionError(
                "execution revision payload is not canonical JSONL"
            )
        return projection

    @property
    def jsonl_bytes(self) -> bytes:
        """Return exactly one canonical event per line, including final newlines."""
        return b"".join(
            canonical_json_bytes(event.to_dict()) + b"\n" for event in self.events
        )

    @property
    def watermark(self) -> int:
        """Return the number of immutable revisions in the projection."""
        return len(self.events)

    @property
    def terminal_events(self) -> tuple[ExecutionRevisionEvent, ...]:
        """Return each node's latest projected revision in node order."""
        terminal: dict[int, ExecutionRevisionEvent] = {}
        first_seen_node_ids: list[int] = []
        for event in self.events:
            if event.node_id not in terminal:
                first_seen_node_ids.append(event.node_id)
            terminal[event.node_id] = event
        return tuple(terminal[node_id] for node_id in first_seen_node_ids)

    def append_event(
        self,
        event: ExecutionRevisionEvent,
    ) -> tuple[ExecutionRevisionProjection, ExecutionRevisionEvent]:
        """Purely append one exact event or return its idempotent predecessor."""
        if type(event) is not ExecutionRevisionEvent:
            raise RevisionProjectionError(
                "execution revision append requires an exact event"
            )
        if event.run_id != self.run_id or event.campaign_id != self.campaign_id:
            raise RevisionProjectionConflictError(
                "execution revision event identity differs from the projection"
            )
        matching = tuple(
            existing
            for existing in self.events
            if existing.node_id == event.node_id
            and existing.execution_revision == event.execution_revision
        )
        if matching:
            existing = matching[0]
            if canonical_json_bytes(existing.semantic_payload()) != (
                canonical_json_bytes(event.semantic_payload())
            ):
                raise RevisionProjectionConflictError(
                    "execution revision conflicts with prior semantic content"
                )
            return self, existing
        projected = type(self)(
            run_id=self.run_id,
            campaign_id=self.campaign_id,
            require_contiguous_node_ids=self.require_contiguous_node_ids,
            events=self.events + (event,),
        )
        return projected, event

    def append_projection(
        self,
        *,
        node_id: int,
        execution_revision: int,
        idea_id: str | None,
        selection_batch_id: str | None,
        parent_node_id: int | None,
        started_at: str,
        recorded_at: str,
        execution_status: ExecutionStatus,
        evaluation_status: EpisodeEvaluationStatus,
        evaluator_fingerprint_ids: tuple[str, ...],
        measurements: Mapping[str, float],
        feedback: str,
        technical_difficulties: str,
        artifact_refs: Mapping[str, str],
        projection: Mapping[str, Any],
    ) -> tuple[ExecutionRevisionProjection, ExecutionRevisionEvent]:
        """Mint from complete explicit semantics, then purely project the event."""
        event = ExecutionRevisionEvent.mint(
            schema=EXECUTION_REVISION_EVENT_SCHEMA,
            run_id=self.run_id,
            campaign_id=self.campaign_id,
            node_id=node_id,
            execution_revision=execution_revision,
            idea_id=idea_id,
            selection_batch_id=selection_batch_id,
            parent_node_id=parent_node_id,
            started_at=started_at,
            recorded_at=recorded_at,
            execution_status=execution_status,
            evaluation_status=evaluation_status,
            evaluator_fingerprint_ids=evaluator_fingerprint_ids,
            measurements=measurements,
            feedback=feedback,
            technical_difficulties=technical_difficulties,
            artifact_refs=artifact_refs,
            projection=projection,
        )
        return self.append_event(event)

    def _validate_sequence(self) -> None:
        seen_keys: set[tuple[int, int]] = set()
        next_revision: dict[int, int] = {}
        first_seen_nodes: list[int] = []
        previous_recorded_at = None
        for event in self.events:
            if event.run_id != self.run_id or event.campaign_id != self.campaign_id:
                raise RevisionProjectionConflictError(
                    "execution revision projection identity changed"
                )
            started_at = parse_utc_timestamp(
                event.started_at,
                "execution revision started_at",
            )
            recorded_at = parse_utc_timestamp(
                event.recorded_at,
                "execution revision recorded_at",
            )
            if recorded_at < started_at:
                raise RevisionProjectionConflictError(
                    "execution revision was recorded before it started"
                )
            if previous_recorded_at is not None and recorded_at < previous_recorded_at:
                raise RevisionProjectionConflictError(
                    "execution revision recording chronology moved backwards"
                )
            key = (event.node_id, event.execution_revision)
            if key in seen_keys:
                raise RevisionProjectionConflictError(
                    "execution revision projection contains a duplicate revision"
                )
            expected_revision = next_revision.get(event.node_id, 0)
            if event.execution_revision != expected_revision:
                raise RevisionProjectionConflictError(
                    "execution revision projection revisions are not gap-free"
                )
            if event.node_id not in next_revision:
                first_seen_nodes.append(event.node_id)
            next_revision[event.node_id] = expected_revision + 1
            seen_keys.add(key)
            previous_recorded_at = recorded_at
        if self.require_contiguous_node_ids and first_seen_nodes != list(
            range(len(first_seen_nodes))
        ):
            raise RevisionProjectionConflictError(
                "execution revision projection node ids are not contiguous"
            )
