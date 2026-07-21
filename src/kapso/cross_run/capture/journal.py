"""Crash-atomic, logically append-only execution-revision journal."""

from __future__ import annotations

import fcntl
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    normalize_utc_timestamp,
    require_identifier,
    to_json_value,
)
from kapso.cross_run.contracts import (
    EpisodeEvaluationStatus,
    ExecutionStatus,
    StrictContract,
)

EXECUTION_REVISION_EVENT_SCHEMA = "kapso.execution_revision_event.v1"


class JournalConflictError(ValueError):
    """A journal append conflicts with its immutable event sequence."""


def _utc_now() -> str:
    current = datetime.now(timezone.utc)
    timespec = "microseconds" if current.microsecond else "seconds"
    return current.isoformat(timespec=timespec).replace("+00:00", "Z")


def _normalized_utc(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("execution revision timestamp must include a UTC offset")
    current = parsed.astimezone(timezone.utc)
    timespec = "microseconds" if current.microsecond else "seconds"
    return current.isoformat(timespec=timespec).replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise JournalConflictError("execution journal path contains a symlink")


@dataclass(frozen=True)
class ExecutionRevisionEvent(StrictContract):
    """One immutable projection of one executed node revision."""

    event_id: str
    schema: str
    run_id: str
    campaign_id: str
    node_id: int
    execution_revision: int
    idea_id: str | None
    selection_batch_id: str | None
    parent_node_id: int | None
    started_at: str
    recorded_at: str
    execution_status: ExecutionStatus
    evaluation_status: EpisodeEvaluationStatus
    evaluator_fingerprint_ids: tuple[str, ...]
    measurements: Mapping[str, float]
    feedback: str
    technical_difficulties: str
    artifact_refs: Mapping[str, str]
    projection: Mapping[str, Any]

    CONTENT_NAMESPACE = "execution-revision-event"
    IDENTITY_FIELD = "event_id"

    def _validate(self) -> None:
        if self.schema != EXECUTION_REVISION_EVENT_SCHEMA:
            raise ValueError("execution revision event schema is incompatible")
        require_identifier(self.run_id, "run_id")
        require_identifier(self.campaign_id, "campaign_id")
        for value, name in (
            (self.node_id, "node_id"),
            (self.execution_revision, "execution_revision"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"execution revision {name} must be non-negative")
        if self.parent_node_id is not None and (
            type(self.parent_node_id) is not int or self.parent_node_id < 0
        ):
            raise ValueError("execution revision parent_node_id is invalid")
        if (self.idea_id is None) != (self.selection_batch_id is None):
            raise ValueError("execution revision idea and batch must appear together")
        if self.idea_id is not None:
            require_identifier(self.idea_id, "idea_id")
            require_identifier(self.selection_batch_id, "selection_batch_id")
        normalize_utc_timestamp(self.started_at, "started_at")
        normalize_utc_timestamp(self.recorded_at, "recorded_at")
        if not isinstance(self.execution_status, ExecutionStatus) or not isinstance(
            self.evaluation_status, EpisodeEvaluationStatus
        ):
            raise ValueError("execution revision statuses are invalid")
        if not isinstance(self.feedback, str) or not isinstance(
            self.technical_difficulties, str
        ):
            raise ValueError("execution revision observations must be strings")
        if self.evaluator_fingerprint_ids != tuple(
            sorted(set(self.evaluator_fingerprint_ids))
        ):
            raise ValueError("evaluator fingerprints must be sorted and unique")
        for evaluator_id in self.evaluator_fingerprint_ids:
            require_identifier(evaluator_id, "evaluator_fingerprint_id")
        if not isinstance(self.artifact_refs, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.artifact_refs.items()
        ):
            raise ValueError("execution revision artifact refs must be non-empty")
        if not isinstance(self.measurements, Mapping) or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for key, value in self.measurements.items()
        ):
            raise ValueError("execution revision measurements are invalid")
        projection = self.projection
        if not isinstance(projection, Mapping):
            raise ValueError("execution revision projection must be an object")
        expected_links = {
            "node_id": self.node_id,
            "execution_revision": self.execution_revision,
            "idea_id": self.idea_id,
            "selection_batch_id": self.selection_batch_id,
            "parent_node_id": self.parent_node_id,
            "feedback": self.feedback,
            "technical_difficulties": self.technical_difficulties,
        }
        conflicts = tuple(
            sorted(
                name
                for name, expected in expected_links.items()
                if projection.get(name) != expected
            )
        )
        if conflicts:
            raise ValueError(
                f"execution revision projection conflicts with event: {conflicts}"
            )
        had_error = projection.get("had_error")
        evaluation_valid = projection.get("evaluation_valid")
        if type(had_error) is not bool or type(evaluation_valid) is not bool:
            raise ValueError("execution revision projection states are invalid")
        if had_error and self.execution_status is ExecutionStatus.COMPLETED:
            raise ValueError("failed projection cannot have completed execution")
        if not had_error and self.execution_status is not ExecutionStatus.COMPLETED:
            raise ValueError("successful projection must have completed execution")
        has_measurement = projection.get("raw_score") is not None and bool(
            projection.get("evaluation_attempts")
        )
        expected_evaluation_status = (
            EpisodeEvaluationStatus.NOT_RUN
            if had_error
            else (
                EpisodeEvaluationStatus.INVALID
                if not evaluation_valid
                else (
                    EpisodeEvaluationStatus.VALID
                    if has_measurement
                    else EpisodeEvaluationStatus.PARTIAL
                )
            )
        )
        if self.evaluation_status is not expected_evaluation_status:
            raise ValueError("execution revision evaluation status is inconsistent")
        if had_error and self.evaluation_status is not EpisodeEvaluationStatus.NOT_RUN:
            raise ValueError("failed execution cannot claim an evaluation")

    def semantic_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("event_id")
        payload.pop("recorded_at")
        return payload


class ExecutionRevisionJournal:
    """Serialize an immutable event sequence using atomic copy-on-write append."""

    def __init__(self, path: str | Path, *, run_id: str, campaign_id: str):
        self.path = Path(path)
        self.run_id = require_identifier(run_id, "run_id")
        self.campaign_id = require_identifier(campaign_id, "campaign_id")
        _reject_symlink_components(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        _reject_symlink_components(self.lock_path)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            self.lock_path.chmod(0o600)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if not self.path.exists():
                self._write_events(())
            self._validate_sequence(self.read_events())
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def read_events(self) -> tuple[ExecutionRevisionEvent, ...]:
        raw = self.path.read_bytes()
        if not raw:
            return ()
        if not raw.endswith(b"\n"):
            raise JournalConflictError("execution journal has an incomplete tail")
        lines = raw.splitlines()
        if any(not line for line in lines):
            raise JournalConflictError("execution journal contains a blank event")
        events = tuple(ExecutionRevisionEvent.from_json_bytes(line) for line in lines)
        self._validate_sequence(events)
        return events

    @property
    def watermark(self) -> int:
        return len(self.read_events())

    def terminal_events(self) -> tuple[ExecutionRevisionEvent, ...]:
        terminal: dict[int, ExecutionRevisionEvent] = {}
        for event in self.read_events():
            terminal[event.node_id] = event
        return tuple(terminal[node_id] for node_id in sorted(terminal))

    def append_projection(
        self,
        *,
        projection: Mapping[str, Any],
        execution_status: ExecutionStatus,
        evaluation_status: EpisodeEvaluationStatus,
        evaluator_fingerprint_ids: tuple[str, ...],
        measurements: Mapping[str, float],
        artifact_refs: Mapping[str, str],
    ) -> ExecutionRevisionEvent:
        semantic = {
            "schema": EXECUTION_REVISION_EVENT_SCHEMA,
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "node_id": projection["node_id"],
            "execution_revision": projection["execution_revision"],
            "idea_id": projection["idea_id"],
            "selection_batch_id": projection["selection_batch_id"],
            "parent_node_id": projection["parent_node_id"],
            "started_at": _normalized_utc(projection["timestamp"]),
            "execution_status": execution_status,
            "evaluation_status": evaluation_status,
            "evaluator_fingerprint_ids": tuple(sorted(set(evaluator_fingerprint_ids))),
            "measurements": dict(measurements),
            "feedback": projection["feedback"],
            "technical_difficulties": projection["technical_difficulties"],
            "artifact_refs": dict(artifact_refs),
            "projection": dict(projection),
        }
        _reject_symlink_components(self.lock_path)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            self.lock_path.chmod(0o600)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            events = self.read_events()
            matching = tuple(
                event
                for event in events
                if event.node_id == semantic["node_id"]
                and event.execution_revision == semantic["execution_revision"]
            )
            if matching:
                if matching[0].semantic_payload() != to_json_value(semantic):
                    raise JournalConflictError(
                        "execution journal revision conflicts with prior content"
                    )
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                return matching[0]
            event = ExecutionRevisionEvent.mint(
                **semantic,
                recorded_at=_utc_now(),
            )
            proposed = events + (event,)
            self._validate_sequence(proposed)
            self._write_events(proposed)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return event

    def _validate_sequence(self, events: tuple[ExecutionRevisionEvent, ...]) -> None:
        seen_keys: set[tuple[int, int]] = set()
        next_revision: dict[int, int] = {}
        first_seen_nodes: list[int] = []
        for event in events:
            if event.run_id != self.run_id or event.campaign_id != self.campaign_id:
                raise JournalConflictError("execution journal identity changed")
            key = (event.node_id, event.execution_revision)
            if key in seen_keys:
                raise JournalConflictError(
                    "execution journal contains duplicate revision"
                )
            expected_revision = next_revision.get(event.node_id, 0)
            if event.execution_revision != expected_revision:
                raise JournalConflictError(
                    "execution journal revisions are not gap-free"
                )
            if event.node_id not in next_revision:
                first_seen_nodes.append(event.node_id)
            next_revision[event.node_id] = expected_revision + 1
            seen_keys.add(key)
        if first_seen_nodes != list(range(len(first_seen_nodes))):
            raise JournalConflictError("execution journal node ids are not contiguous")

    def _write_events(self, events: tuple[ExecutionRevisionEvent, ...]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self.path.parent,
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            for event in events:
                handle.write(canonical_json_bytes(event.to_dict()))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(0o600)
        os.replace(temporary, self.path)
        self.path.chmod(0o600)
        _fsync_directory(self.path.parent)
