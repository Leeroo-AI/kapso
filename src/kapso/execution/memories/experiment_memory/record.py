"""Dependency-pure executed-experiment record contract."""

import json
import math
import re
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from kapso.cross_run.canonical import normalize_utc_timestamp
from kapso.execution.fidelity import EvaluationAttempt

EXPERIMENT_HISTORY_SCHEMA = "kapso.experiment_history.v5"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*_[0-9a-f]{32}$")
_RECORD_FIELDS = {
    "node_id",
    "execution_revision",
    "idea_id",
    "selection_batch_id",
    "parent_node_id",
    "solution",
    "solution_embedding",
    "raw_score",
    "normalized_utility",
    "objective_direction",
    "feedback",
    "branch_name",
    "had_error",
    "recoverable_error",
    "error_message",
    "timestamp",
    "technical_difficulties",
    "metrics",
    "primary_metric",
    "external_evaluation_metadata",
    "external_evaluation_error",
    "evaluation_valid",
    "evaluation_provenance",
    "evaluation_integrity_error",
    "build_fidelity",
    "eval_fidelity",
    "validation_tier",
    "evaluation_attempts",
    "phase_telemetry",
    "duration_seconds",
    "cost_usd",
}


def _finite_optional(value: Any, name: str, minimum: float | None = None):
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"experiment {name} must be finite or null")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"experiment {name} must be >= {minimum}")
    return numeric


def _typed_identifier(value: Any, prefix: str) -> str:
    if (
        not isinstance(value, str)
        or not _IDENTIFIER.fullmatch(value)
        or not value.startswith(prefix + "_")
    ):
        raise ValueError(f"experiment {prefix} id is invalid")
    return value


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Return cosine similarity and reject incompatible vector dimensions."""
    left = tuple(a)
    right = tuple(b)
    if len(left) != len(right):
        raise ValueError(f"Embedding dimensions differ: {len(left)} vs {len(right)}")
    dot = sum(x * y for x, y in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


@runtime_checkable
class ExperimentNodeProjection(Protocol):
    """Structural node evidence required to construct an experiment record."""

    node_id: int
    execution_revision: int
    idea_id: Optional[str]
    selection_batch_id: Optional[str]
    parent_node_id: Optional[int]
    solution: str
    score: Optional[float]
    evaluation_valid: bool
    had_error: bool
    recoverable_error: bool
    branch_name: str
    feedback: str
    error_message: str
    started_at: str
    technical_difficulties: str
    metrics: Mapping[str, float]
    primary_metric: Optional[str]
    external_evaluation_metadata: Mapping[str, Any]
    external_evaluation_error: str
    evaluation_provenance: str
    evaluation_integrity_error: str
    build_fidelity: str
    eval_fidelity: str
    evaluation_attempts: Sequence[EvaluationAttempt]
    phase_telemetry: Mapping[str, Mapping[str, float]]
    duration_seconds: Optional[float]
    cost_usd: Optional[float]


@dataclass(frozen=True)
class ExperimentRecord:
    """One executed node, separate from unexecuted idea candidates."""

    node_id: int
    execution_revision: int
    idea_id: Optional[str]
    selection_batch_id: Optional[str]
    parent_node_id: Optional[int]
    solution: str
    solution_embedding: Tuple[float, ...]
    raw_score: Optional[float]
    normalized_utility: Optional[float]
    objective_direction: str
    feedback: str
    branch_name: str
    had_error: bool
    recoverable_error: bool
    error_message: str
    timestamp: str
    technical_difficulties: str
    metrics: Dict[str, float]
    primary_metric: Optional[str]
    external_evaluation_metadata: Dict[str, Any]
    external_evaluation_error: str
    evaluation_valid: bool
    evaluation_provenance: str
    evaluation_integrity_error: str
    build_fidelity: str
    eval_fidelity: str
    validation_tier: str
    evaluation_attempts: Tuple[EvaluationAttempt, ...]
    phase_telemetry: Dict[str, Dict[str, float]]
    duration_seconds: Optional[float]
    cost_usd: Optional[float]

    def __post_init__(self) -> None:
        for value, name in (
            (self.node_id, "node id"),
            (self.execution_revision, "execution revision"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"experiment {name} must be non-negative")
        if not isinstance(self.phase_telemetry, dict):
            raise ValueError("experiment phase telemetry must be an object")
        phase_telemetry = {}
        for phase_name, measurements in self.phase_telemetry.items():
            if not isinstance(phase_name, str) or not phase_name:
                raise ValueError("experiment phase telemetry name is invalid")
            if not isinstance(measurements, dict):
                raise ValueError("experiment phase telemetry values must be objects")
            phase_telemetry[phase_name] = {}
            for measurement, value in measurements.items():
                if (
                    not isinstance(measurement, str)
                    or not measurement
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    raise ValueError("experiment phase telemetry value is invalid")
                phase_telemetry[phase_name][measurement] = float(value)
        object.__setattr__(self, "phase_telemetry", phase_telemetry)
        if not isinstance(self.solution_embedding, (list, tuple)) or not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in self.solution_embedding
        ):
            raise ValueError("experiment solution embedding is invalid")
        object.__setattr__(
            self,
            "solution_embedding",
            tuple(float(value) for value in self.solution_embedding),
        )
        if self.parent_node_id is not None and (
            isinstance(self.parent_node_id, bool)
            or not isinstance(self.parent_node_id, int)
            or self.parent_node_id < 0
        ):
            raise ValueError("experiment parent node id must be non-negative or null")
        if (self.idea_id is None) != (self.selection_batch_id is None):
            raise ValueError("experiment idea and batch links must appear together")
        if self.idea_id is not None:
            _typed_identifier(self.idea_id, "idea")
            _typed_identifier(self.selection_batch_id, "batch")
        for value, name in (
            (self.solution, "solution"),
            (self.branch_name, "branch"),
            (self.timestamp, "timestamp"),
            (self.objective_direction, "objective direction"),
            (self.build_fidelity, "build fidelity"),
            (self.eval_fidelity, "evaluation fidelity"),
            (self.validation_tier, "validation tier"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"experiment {name} must be non-empty")
        for value, name in (
            (self.feedback, "feedback"),
            (self.error_message, "error message"),
            (self.technical_difficulties, "technical difficulties"),
            (self.external_evaluation_error, "external evaluation error"),
            (self.evaluation_integrity_error, "evaluation integrity error"),
        ):
            if not isinstance(value, str):
                raise ValueError(f"experiment {name} must be a string")
        if self.objective_direction not in {"maximize", "minimize"}:
            raise ValueError("experiment objective direction is invalid")
        if self.build_fidelity not in {"fast", "full"} or self.eval_fidelity not in {
            "fast",
            "full",
        }:
            raise ValueError("experiment fidelity is invalid")
        if self.validation_tier not in {"probe", "validated", "full"}:
            raise ValueError("experiment validation tier is invalid")
        for value, name in (
            (self.had_error, "error status"),
            (self.recoverable_error, "recoverability"),
            (self.evaluation_valid, "evaluation validity"),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"experiment {name} must be boolean")
        if self.recoverable_error and not self.had_error:
            raise ValueError("only failed experiments can be recoverable")
        normalize_utc_timestamp(self.timestamp, "experiment timestamp")
        object.__setattr__(
            self,
            "raw_score",
            _finite_optional(self.raw_score, "raw score"),
        )
        object.__setattr__(
            self,
            "normalized_utility",
            _finite_optional(self.normalized_utility, "normalized utility"),
        )
        if self.raw_score is None and self.normalized_utility is not None:
            raise ValueError("normalized utility requires a raw score")
        if self.raw_score is not None:
            sign = 1.0 if self.objective_direction == "maximize" else -1.0
            if self.normalized_utility != sign * self.raw_score:
                raise ValueError(
                    "normalized utility conflicts with objective direction"
                )
        if self.had_error and (self.raw_score is not None or self.evaluation_attempts):
            raise ValueError("failed experiments cannot contain evaluation evidence")
        if not isinstance(self.metrics, dict) or not all(
            isinstance(key, str)
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for key, value in self.metrics.items()
        ):
            raise ValueError("experiment metrics are invalid")
        object.__setattr__(
            self,
            "metrics",
            {key: float(value) for key, value in self.metrics.items()},
        )
        if self.primary_metric is not None and (
            not isinstance(self.primary_metric, str)
            or self.primary_metric not in self.metrics
        ):
            raise ValueError("experiment primary metric is invalid")
        if not isinstance(self.external_evaluation_metadata, dict):
            raise ValueError("experiment external metadata must be an object")
        json.dumps(
            self.external_evaluation_metadata,
            sort_keys=True,
            allow_nan=False,
        )
        if self.evaluation_provenance not in {"provided", "agent_generated"}:
            raise ValueError("experiment evaluation provenance is invalid")
        if not isinstance(self.evaluation_attempts, (list, tuple)) or not all(
            isinstance(attempt, EvaluationAttempt)
            for attempt in self.evaluation_attempts
        ):
            raise ValueError("experiment evaluation attempts are invalid")
        object.__setattr__(self, "evaluation_attempts", tuple(self.evaluation_attempts))
        object.__setattr__(
            self,
            "duration_seconds",
            _finite_optional(self.duration_seconds, "duration", 0.0),
        )
        object.__setattr__(
            self,
            "cost_usd",
            _finite_optional(self.cost_usd, "cost", 0.0),
        )

    @classmethod
    def from_node(
        cls,
        node: ExperimentNodeProjection,
        objective_direction: str,
        require_idea_links: bool,
        solution_embedding: Iterable[float] = (),
    ) -> "ExperimentRecord":
        idea_id = getattr(node, "idea_id", None)
        batch_id = getattr(node, "selection_batch_id", None)
        if require_idea_links:
            idea_id = _typed_identifier(idea_id, "idea")
            batch_id = _typed_identifier(batch_id, "batch")
        raw_score = node.score if node.evaluation_valid and not node.had_error else None
        sign = 1.0 if objective_direction == "maximize" else -1.0
        normalized = None if raw_score is None else sign * raw_score
        if node.had_error:
            validation_tier = "probe"
        elif node.eval_fidelity == "full" and node.build_fidelity == "full":
            validation_tier = "full"
        elif node.eval_fidelity == "full":
            validation_tier = "validated"
        else:
            validation_tier = "probe"
        return cls(
            node_id=node.node_id,
            execution_revision=node.execution_revision,
            idea_id=idea_id,
            selection_batch_id=batch_id,
            parent_node_id=node.parent_node_id,
            solution=node.solution,
            solution_embedding=tuple(solution_embedding),
            raw_score=raw_score,
            normalized_utility=normalized,
            objective_direction=objective_direction,
            feedback=node.feedback,
            branch_name=node.branch_name,
            had_error=node.had_error,
            recoverable_error=node.recoverable_error,
            error_message=node.error_message,
            timestamp=node.started_at,
            technical_difficulties=node.technical_difficulties,
            metrics=dict(node.metrics),
            primary_metric=node.primary_metric,
            external_evaluation_metadata=dict(node.external_evaluation_metadata),
            external_evaluation_error=node.external_evaluation_error,
            evaluation_valid=node.evaluation_valid,
            evaluation_provenance=node.evaluation_provenance,
            evaluation_integrity_error=node.evaluation_integrity_error,
            build_fidelity=node.build_fidelity,
            eval_fidelity=node.eval_fidelity,
            validation_tier=validation_tier,
            evaluation_attempts=tuple(node.evaluation_attempts),
            phase_telemetry={
                phase: dict(measurements)
                for phase, measurements in node.phase_telemetry.items()
            },
            duration_seconds=node.duration_seconds,
            cost_usd=node.cost_usd,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "execution_revision": self.execution_revision,
            "idea_id": self.idea_id,
            "selection_batch_id": self.selection_batch_id,
            "parent_node_id": self.parent_node_id,
            "solution": self.solution,
            "solution_embedding": list(self.solution_embedding),
            "raw_score": self.raw_score,
            "normalized_utility": self.normalized_utility,
            "objective_direction": self.objective_direction,
            "feedback": self.feedback,
            "branch_name": self.branch_name,
            "had_error": self.had_error,
            "recoverable_error": self.recoverable_error,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "technical_difficulties": self.technical_difficulties,
            "metrics": dict(self.metrics),
            "primary_metric": self.primary_metric,
            "external_evaluation_metadata": dict(self.external_evaluation_metadata),
            "external_evaluation_error": self.external_evaluation_error,
            "evaluation_valid": self.evaluation_valid,
            "evaluation_provenance": self.evaluation_provenance,
            "evaluation_integrity_error": self.evaluation_integrity_error,
            "build_fidelity": self.build_fidelity,
            "eval_fidelity": self.eval_fidelity,
            "validation_tier": self.validation_tier,
            "evaluation_attempts": [
                attempt.to_dict() for attempt in self.evaluation_attempts
            ],
            "phase_telemetry": {
                phase: dict(measurements)
                for phase, measurements in self.phase_telemetry.items()
            },
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRecord":
        if not isinstance(data, Mapping) or set(data) != _RECORD_FIELDS:
            raise ValueError("experiment record fields are invalid")
        raw_attempts = data["evaluation_attempts"]
        if not isinstance(raw_attempts, list):
            raise ValueError("experiment evaluation attempts must be a list")
        attempt_fields = {
            "commit_sha",
            "evaluator_id",
            "fidelity",
            "fraction",
            "seed",
            "score",
            "duration_seconds",
            "metrics",
        }
        if any(
            not isinstance(attempt, Mapping) or set(attempt) != attempt_fields
            for attempt in raw_attempts
        ):
            raise ValueError("experiment evaluation attempt fields are invalid")
        values = dict(data)
        if not isinstance(values["solution_embedding"], list):
            raise ValueError("experiment solution embedding must be a list")
        values["solution_embedding"] = tuple(values["solution_embedding"])
        values["evaluation_attempts"] = tuple(
            EvaluationAttempt.from_dict(attempt) for attempt in raw_attempts
        )
        return cls(**values)

    def __str__(self) -> str:
        status = "failed" if self.had_error else f"utility={self.normalized_utility}"
        return f"Experiment {self.node_id} ({status}): {self.solution}"


__all__ = [
    "EXPERIMENT_HISTORY_SCHEMA",
    "ExperimentNodeProjection",
    "ExperimentRecord",
    "cosine_similarity",
]
