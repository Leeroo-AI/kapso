"""Pure canonical projection of checkpoint-owned experiment history."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

from kapso.core.embedding_contracts import EmbeddingRecord, complete_input_hash
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    normalize_utc_timestamp,
    parse_json_bytes,
    require_content_id,
    require_identifier,
)
from kapso.execution.memories.experiment_memory.record import (
    ExperimentNodeProjection,
    ExperimentRecord,
)

RUN_EXPERIMENT_HISTORY_SCHEMA = "kapso.run_experiment_history.v1"

_HISTORY_FIELDS = {
    "campaign_id",
    "embedding_dimensions",
    "embedding_canonicalizer_version",
    "embedding_model",
    "embedding_provider",
    "embedding_space_id",
    "objective_direction",
    "records",
    "require_idea_links",
    "revision",
    "run_id",
    "schema",
}


@dataclass(frozen=True)
class ExperimentHistoryProjection:
    """One exact, filesystem-independent experiment-history payload."""

    schema: str
    run_id: str
    campaign_id: str
    embedding_space_id: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    embedding_canonicalizer_version: str
    revision: int
    objective_direction: str
    require_idea_links: bool
    records: tuple[ExperimentRecord, ...]

    def __post_init__(self) -> None:
        if self.schema != RUN_EXPERIMENT_HISTORY_SCHEMA:
            raise ValueError("experiment-history projection schema is incompatible")
        require_identifier(self.run_id, "experiment-history projection run_id")
        require_identifier(
            self.campaign_id,
            "experiment-history projection campaign_id",
        )
        require_content_id(
            self.embedding_space_id,
            "experiment-history projection embedding_space_id",
        )
        if self.embedding_space_id.split(":sha256:", 1)[0] != "embedding-space":
            raise ValueError("experiment-history projection embedding space is invalid")
        for value, name in (
            (self.embedding_provider, "embedding provider"),
            (self.embedding_model, "embedding model"),
            (
                self.embedding_canonicalizer_version,
                "embedding canonicalizer version",
            ),
        ):
            require_identifier(
                value,
                f"experiment-history projection {name}",
            )
        if type(self.embedding_dimensions) is not int or self.embedding_dimensions <= 0:
            raise ValueError(
                "experiment-history projection embedding dimensions must be positive"
            )
        if self.embedding_space_id != content_id(
            "embedding-space",
            {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
                "canonicalizer_version": self.embedding_canonicalizer_version,
            },
        ):
            raise ValueError(
                "experiment-history projection embedding space identity differs"
            )
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError(
                "experiment-history projection revision must be non-negative"
            )
        if self.objective_direction not in {"maximize", "minimize"}:
            raise ValueError(
                "experiment-history projection objective direction is invalid"
            )
        if type(self.require_idea_links) is not bool:
            raise ValueError(
                "experiment-history projection idea-link policy is invalid"
            )
        if type(self.records) is not tuple or any(
            type(record) is not ExperimentRecord for record in self.records
        ):
            raise ValueError(
                "experiment-history projection records must be exact records"
            )
        node_ids = tuple(record.node_id for record in self.records)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("experiment-history projection node ids must be unique")
        if self.require_idea_links and node_ids != tuple(range(len(self.records))):
            raise ValueError(
                "generic experiment-history projection node ids must be contiguous"
            )
        if self.revision != sum(
            record.execution_revision + 1 for record in self.records
        ):
            raise ValueError(
                "experiment-history projection revision differs from its records"
            )
        if any(
            record.objective_direction != self.objective_direction
            for record in self.records
        ):
            raise ValueError(
                "experiment-history projection record objective direction changed"
            )
        for record in self.records:
            normalize_utc_timestamp(
                record.timestamp,
                "experiment-history projection record timestamp",
            )
            if len(record.solution_embedding) != self.embedding_dimensions:
                raise ValueError(
                    "experiment-history projection embedding dimensions differ"
                )
        if self.require_idea_links and any(
            record.idea_id is None or record.selection_batch_id is None
            for record in self.records
        ):
            raise ValueError("experiment-history projection requires record idea links")

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible projection document."""
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "embedding_space_id": self.embedding_space_id,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_canonicalizer_version": (self.embedding_canonicalizer_version),
            "revision": self.revision,
            "objective_direction": self.objective_direction,
            "require_idea_links": self.require_idea_links,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json_bytes(self) -> bytes:
        """Return canonical bytes without formatting or a trailing newline."""
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentHistoryProjection":
        """Decode one exact JSON-compatible projection document."""
        if not isinstance(data, Mapping) or set(data) != _HISTORY_FIELDS:
            raise ValueError("experiment-history projection fields are invalid")
        raw_records = data["records"]
        if not isinstance(raw_records, list):
            raise ValueError("experiment-history projection records must be a list")
        return cls(
            schema=data["schema"],
            run_id=data["run_id"],
            campaign_id=data["campaign_id"],
            embedding_space_id=data["embedding_space_id"],
            embedding_provider=data["embedding_provider"],
            embedding_model=data["embedding_model"],
            embedding_dimensions=data["embedding_dimensions"],
            embedding_canonicalizer_version=data["embedding_canonicalizer_version"],
            revision=data["revision"],
            objective_direction=data["objective_direction"],
            require_idea_links=data["require_idea_links"],
            records=tuple(ExperimentRecord.from_dict(item) for item in raw_records),
        )

    @classmethod
    def from_json_bytes(
        cls,
        payload: bytes,
    ) -> "ExperimentHistoryProjection":
        """Decode only the strict canonical bytes of one exact projection."""
        if type(payload) is not bytes:
            raise ValueError("experiment-history projection payload must be bytes")
        parsed = parse_json_bytes(payload)
        if not isinstance(parsed, Mapping):
            raise ValueError("experiment-history projection must be an object")
        projection = cls.from_dict(parsed)
        if projection.to_json_bytes() != payload:
            raise ValueError("experiment-history projection bytes must be canonical")
        return projection


def build_experiment_history_genesis(
    *,
    run_id: str,
    campaign_id: str,
    embedding_space_id: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
    embedding_canonicalizer_version: str,
    objective_direction: str,
    require_idea_links: bool,
) -> ExperimentHistoryProjection:
    """Build the empty canonical projection for one immutable run identity."""
    return ExperimentHistoryProjection(
        schema=RUN_EXPERIMENT_HISTORY_SCHEMA,
        run_id=run_id,
        campaign_id=campaign_id,
        embedding_space_id=embedding_space_id,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        embedding_canonicalizer_version=embedding_canonicalizer_version,
        revision=0,
        objective_direction=objective_direction,
        require_idea_links=require_idea_links,
        records=(),
    )


def _normalize_embeddings(
    embeddings_by_node_revision: Mapping[tuple[int, int], EmbeddingRecord],
) -> dict[tuple[int, int], EmbeddingRecord]:
    if not isinstance(embeddings_by_node_revision, Mapping):
        raise ValueError("experiment projection embeddings must be a mapping")
    normalized: dict[tuple[int, int], EmbeddingRecord] = {}
    for key, embedding in embeddings_by_node_revision.items():
        if (
            type(key) is not tuple
            or len(key) != 2
            or any(type(value) is not int or value < 0 for value in key)
        ):
            raise ValueError(
                "experiment projection embedding keys must be node-revision pairs"
            )
        if type(embedding) is not EmbeddingRecord:
            raise ValueError("experiment projection embedding must be one exact record")
        normalized[key] = embedding
    return normalized


def _record_identity(record: ExperimentRecord) -> tuple[Any, ...]:
    return (
        record.idea_id,
        record.selection_batch_id,
        record.parent_node_id,
        record.solution,
        record.objective_direction,
    )


def require_experiment_record_successor(
    predecessor: ExperimentRecord,
    candidate: ExperimentRecord,
) -> None:
    """Require the exact persisted form of one legal Generic node revision."""

    if (
        type(predecessor) is not ExperimentRecord
        or type(candidate) is not ExperimentRecord
    ):
        raise ValueError("experiment successor requires exact records")
    if candidate.execution_revision != predecessor.execution_revision + 1:
        raise ValueError("experiment projection node revision contains a gap")
    if (
        _record_identity(candidate) != _record_identity(predecessor)
        or candidate.solution_embedding != predecessor.solution_embedding
        or (
            predecessor.branch_name and candidate.branch_name != predecessor.branch_name
        )
        or candidate.evaluation_attempts[: len(predecessor.evaluation_attempts)]
        != predecessor.evaluation_attempts
    ):
        raise ValueError("experiment projection node identity changed")
    appended_attempt_count = len(candidate.evaluation_attempts) - len(
        predecessor.evaluation_attempts
    )
    if predecessor.recoverable_error:
        if appended_attempt_count not in {0, 1}:
            raise ValueError(
                "experiment recovery revision appends too many evaluations"
            )
        if candidate.raw_score is not None and (
            appended_attempt_count != 1
            or candidate.raw_score != candidate.evaluation_attempts[-1].score
        ):
            raise ValueError("experiment recovery score lacks its appended evaluation")
        for name in ("duration_seconds", "cost_usd"):
            previous = getattr(predecessor, name)
            current = getattr(candidate, name)
            if previous is not None and (current is None or current < previous):
                raise ValueError(
                    "experiment recovery cumulative resources moved backwards"
                )
        for phase, previous_measurements in predecessor.phase_telemetry.items():
            current_measurements = candidate.phase_telemetry.get(phase)
            if current_measurements is None or any(
                measurement not in current_measurements
                or current_measurements[measurement] < previous_value
                for measurement, previous_value in previous_measurements.items()
            ):
                raise ValueError("experiment recovery phase telemetry moved backwards")
        return
    mutable_fields = {
        "evaluation_attempts",
        "evaluation_valid",
        "execution_revision",
        "normalized_utility",
        "raw_score",
    }
    if any(
        getattr(candidate, item.name) != getattr(predecessor, item.name)
        for item in fields(ExperimentRecord)
        if item.name not in mutable_fields
    ):
        raise ValueError(
            "nonrecoverable experiment revision changes immutable evidence"
        )
    if appended_attempt_count == 1:
        appended_score = candidate.evaluation_attempts[-1].score
        if candidate.raw_score not in {
            None,
            predecessor.raw_score,
            appended_score,
        } or candidate.evaluation_valid not in {
            predecessor.evaluation_valid,
            True,
        }:
            raise ValueError(
                "experiment validation revision has an invalid score transition"
            )
        if predecessor.evaluation_valid and not candidate.evaluation_valid:
            raise ValueError(
                "experiment validation revision rolls back evaluation validity"
            )
        return
    if not (
        appended_attempt_count == 0
        and predecessor.raw_score is not None
        and candidate.raw_score is None
        and candidate.evaluation_valid == predecessor.evaluation_valid
    ):
        raise ValueError(
            "experiment validation revision has no legal evidence transition"
        )


def project_records(
    *,
    predecessor: ExperimentHistoryProjection,
    nodes: Sequence[ExperimentNodeProjection],
    embeddings_by_node_revision: Mapping[tuple[int, int], EmbeddingRecord],
) -> ExperimentHistoryProjection:
    """Purely apply an exact batch of node revisions to a prior projection.

    A new node consumes exactly one caller-produced embedding at revision zero.
    Later revisions reuse that immutable embedding, so no provider call or other
    side effect can occur while constructing the projection.
    """
    if type(predecessor) is not ExperimentHistoryProjection:
        raise ValueError(
            "experiment projection predecessor must be an exact projection"
        )
    if not isinstance(nodes, (list, tuple)) or any(
        not isinstance(node, ExperimentNodeProjection) for node in nodes
    ):
        raise ValueError("experiment projection nodes must be a node sequence")
    embeddings = _normalize_embeddings(embeddings_by_node_revision)
    consumed_embeddings: set[tuple[int, int]] = set()
    seen_revisions: set[tuple[int, int]] = set()
    records = list(predecessor.records)
    revision = predecessor.revision

    for node in nodes:
        key = (node.node_id, node.execution_revision)
        if key in seen_revisions:
            raise ValueError(
                "experiment projection batch contains a duplicate node revision"
            )
        seen_revisions.add(key)
        normalize_utc_timestamp(
            node.started_at,
            "experiment projection node timestamp",
        )

        matching_positions = tuple(
            position
            for position, record in enumerate(records)
            if record.node_id == node.node_id
        )
        if matching_positions:
            record_position = matching_positions[0]
            prior = records[record_position]
            embedding = prior.solution_embedding
        else:
            record_position = len(records)
            if predecessor.require_idea_links and node.node_id != len(records):
                raise ValueError("experiment projection node ids must be contiguous")
            if node.execution_revision != 0:
                raise ValueError(
                    "new experiment projection nodes must start at revision zero"
                )
            if key not in embeddings:
                raise ValueError(
                    "new experiment projection node requires one exact embedding"
                )
            embedding_record = embeddings[key]
            if (
                embedding_record.embedding_space_id.value
                != predecessor.embedding_space_id
                or embedding_record.provider != predecessor.embedding_provider
                or embedding_record.model != predecessor.embedding_model
                or embedding_record.dimensions != predecessor.embedding_dimensions
                or embedding_record.canonicalizer_version
                != predecessor.embedding_canonicalizer_version
            ):
                raise ValueError(
                    "new experiment projection embedding belongs to another space"
                )
            if embedding_record.input_hash != complete_input_hash(node.solution):
                raise ValueError(
                    "new experiment projection embedding belongs to another input"
                )
            embedding = embedding_record.vector
            consumed_embeddings.add(key)

        record = ExperimentRecord.from_node(
            node,
            predecessor.objective_direction,
            predecessor.require_idea_links,
            embedding,
        )
        if not matching_positions:
            records.append(record)
            revision += 1
            continue

        prior = records[record_position]
        if record.execution_revision < prior.execution_revision:
            raise ValueError("experiment projection node revision moved backwards")
        if record.execution_revision == prior.execution_revision:
            if record != prior:
                raise ValueError(
                    "experiment projection node revision conflicts with prior content"
                )
            continue
        if record.execution_revision != prior.execution_revision + 1:
            raise ValueError("experiment projection node revision contains a gap")
        require_experiment_record_successor(prior, record)
        records[record_position] = record
        revision += 1

    if set(embeddings) != consumed_embeddings:
        raise ValueError(
            "experiment projection embeddings must match new node revisions exactly"
        )
    return ExperimentHistoryProjection(
        schema=predecessor.schema,
        run_id=predecessor.run_id,
        campaign_id=predecessor.campaign_id,
        embedding_space_id=predecessor.embedding_space_id,
        embedding_provider=predecessor.embedding_provider,
        embedding_model=predecessor.embedding_model,
        embedding_dimensions=predecessor.embedding_dimensions,
        embedding_canonicalizer_version=(predecessor.embedding_canonicalizer_version),
        revision=revision,
        objective_direction=predecessor.objective_direction,
        require_idea_links=predecessor.require_idea_links,
        records=tuple(records),
    )


__all__ = [
    "ExperimentHistoryProjection",
    "RUN_EXPERIMENT_HISTORY_SCHEMA",
    "build_experiment_history_genesis",
    "project_records",
    "require_experiment_record_successor",
]
