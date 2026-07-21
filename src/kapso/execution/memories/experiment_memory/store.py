"""Strict executed-experiment memory for evidence-directed campaigns."""

import fcntl
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from kapso.core.llm import LLMBackend
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_identifier,
    to_json_value,
)
from kapso.cross_run.capture.journal import (
    ExecutionRevisionJournal,
    JournalConflictError,
)
from kapso.cross_run.contracts import EpisodeEvaluationStatus, ExecutionStatus
from kapso.cross_run.git_command import BoundedGitCommand
from kapso.cross_run.git_refs import require_git_ref_name
from kapso.cross_run.github.command import CommandOutputKind, CommandRunner
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


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate experiment-history key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"experiment history contains non-finite value: {value}")


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
        timestamp = datetime.fromisoformat(self.timestamp)
        if timestamp.utcoffset() is None:
            raise ValueError("experiment timestamp must include a UTC offset")
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
        node: Any,
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
        timestamp = node.started_at
        if not timestamp and not require_idea_links:
            timestamp = datetime.now(timezone.utc).isoformat()
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
            timestamp=timestamp,
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


class ExperimentHistoryStore:
    """Atomic, objective-aware storage for executed nodes only."""

    def __init__(
        self,
        json_path: str,
        objective_direction: Optional[str] = None,
        require_idea_links: Optional[bool] = None,
        goal: Optional[str] = None,
        llm: Optional[LLMBackend] = None,
        run_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        journal_path: Optional[str] = None,
        git_command_timeout_seconds: Optional[int] = None,
        git_command_output_bytes: Optional[int] = None,
        git_command_runner: Optional[CommandRunner] = None,
    ):
        self.path = Path(json_path)
        self.goal = goal
        self._llm = llm
        self.experiments: List[ExperimentRecord] = []
        self.revision = 0
        self.run_id = run_id
        self.campaign_id = campaign_id
        self.objective_direction = objective_direction
        self.require_idea_links = require_idea_links
        if (git_command_timeout_seconds is None) != (git_command_output_bytes is None):
            raise ValueError("Git command bounds must be provided together")
        self._git_command = (
            BoundedGitCommand(
                timeout_seconds=git_command_timeout_seconds,
                maximum_output_bytes=git_command_output_bytes,
                runner=git_command_runner,
            )
            if git_command_timeout_seconds is not None
            else None
        )
        if self.path.exists():
            self._load()
            if (
                objective_direction is not None
                and objective_direction != self.objective_direction
            ):
                raise ValueError("experiment-history objective direction changed")
            if (
                require_idea_links is not None
                and require_idea_links != self.require_idea_links
            ):
                raise ValueError("experiment-history idea-link policy changed")
            if run_id is not None and run_id != self.run_id:
                raise ValueError("experiment-history run identity changed")
            if campaign_id is not None and campaign_id != self.campaign_id:
                raise ValueError("experiment-history campaign identity changed")
        elif objective_direction not in {"maximize", "minimize"}:
            raise ValueError(
                "new experiment history requires maximize or minimize direction"
            )
        elif not isinstance(require_idea_links, bool):
            raise ValueError("new experiment history requires an idea-link policy")
        elif journal_path is not None:
            self.run_id = require_identifier(run_id, "run_id")
            self.campaign_id = require_identifier(campaign_id, "campaign_id")
        self.revision_journal = None
        if journal_path is not None:
            if self.run_id is None or self.campaign_id is None:
                raise ValueError("journaled experiment history requires run identities")
            self.revision_journal = ExecutionRevisionJournal(
                journal_path,
                run_id=self.run_id,
                campaign_id=self.campaign_id,
            )
            journal_file = Path(journal_path)
            self._transaction_lock_path = journal_file.with_name(
                journal_file.name + ".transaction.lock"
            )
            for component in (
                self._transaction_lock_path,
                *self._transaction_lock_path.parents,
            ):
                if component.is_symlink():
                    raise ValueError("experiment-history transaction path is a symlink")
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self._transaction_lock_path.open("a+b") as lock:
                self._transaction_lock_path.chmod(0o600)
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                self._bind_transaction_lock(lock)
                self._refresh_journaled_authorities()
                if not self.path.exists():
                    self._save(self.experiments, self.revision)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def add_experiment(self, node: Any) -> ExperimentRecord:
        if self.revision_journal is None:
            raise ValueError("experiment-history writes require a revision journal")
        with self._transaction_lock_path.open("a+b") as lock:
            self._transaction_lock_path.chmod(0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._bind_transaction_lock(lock)
            self._refresh_journaled_authorities()
            record = self._add_experiment_locked(node)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return record

    def _add_experiment_locked(self, node: Any) -> ExperimentRecord:
        existing = tuple(
            item for item in self.experiments if item.node_id == node.node_id
        )
        solution_embedding: Iterable[float] = ()
        if existing and existing[0].solution == node.solution:
            solution_embedding = existing[0].solution_embedding
        elif self._llm is not None and node.solution.strip():
            solution_embedding = self._llm.create_embedding(node.solution)
        record = ExperimentRecord.from_node(
            node,
            self.objective_direction,
            self.require_idea_links,
            solution_embedding,
        )
        if existing:
            prior = existing[0]
            if prior != record:
                if record.execution_revision < prior.execution_revision:
                    raise ValueError("experiment node revision moved backwards")
                stable_identity = (
                    prior.idea_id,
                    prior.selection_batch_id,
                    prior.parent_node_id,
                    prior.solution,
                    prior.objective_direction,
                )
                next_identity = (
                    record.idea_id,
                    record.selection_batch_id,
                    record.parent_node_id,
                    record.solution,
                    record.objective_direction,
                )
                if record.execution_revision > prior.execution_revision and (
                    stable_identity != next_identity
                    or record.execution_revision != prior.execution_revision + 1
                ):
                    raise ValueError("experiment node identity or revision changed")
                proposed = (
                    [
                        record if item.node_id == record.node_id else item
                        for item in self.experiments
                    ]
                    if record.execution_revision > prior.execution_revision
                    else self.experiments
                )
            else:
                proposed = self.experiments
        else:
            if record.node_id != len(self.experiments):
                raise ValueError("experiment node ids must be contiguous")
            proposed = self.experiments + [record]
        execution_status = (
            ExecutionStatus.INTERRUPTED
            if record.recoverable_error
            else (
                ExecutionStatus.FAILED_TECHNICAL
                if record.had_error
                else ExecutionStatus.COMPLETED
            )
        )
        evaluation_status = (
            EpisodeEvaluationStatus.NOT_RUN
            if record.had_error
            else (
                EpisodeEvaluationStatus.INVALID
                if not record.evaluation_valid
                else (
                    EpisodeEvaluationStatus.VALID
                    if record.raw_score is not None and record.evaluation_attempts
                    else EpisodeEvaluationStatus.PARTIAL
                )
            )
        )
        measurements = dict(record.metrics)
        if record.raw_score is not None:
            measurements["raw_score"] = record.raw_score
        artifact_refs = {
            name: value
            for name, value in {
                "branch": record.branch_name,
                "parent_branch": getattr(node, "parent_branch_name", ""),
                "implementation_base": getattr(node, "implementation_base_ref", ""),
                "diff_base": getattr(node, "diff_base_ref", ""),
                "feedback_base": getattr(node, "feedback_base_ref", ""),
            }.items()
            if value
        }
        candidate_commit, candidate_ref = self._pin_revision_commit(node, record)
        if candidate_commit is not None:
            artifact_refs["candidate_commit"] = candidate_commit
            for name, commit in self._resolve_revision_base_commits(
                Path(node.workspace_dir),
                candidate_commit,
                {
                    "implementation": getattr(node, "implementation_base_ref", ""),
                    "diff": getattr(node, "diff_base_ref", ""),
                    "feedback": getattr(node, "feedback_base_ref", ""),
                },
            ).items():
                artifact_refs[f"{name}_base_commit"] = commit
        if candidate_ref:
            artifact_refs["candidate_ref"] = candidate_ref
        for position, attempt in enumerate(record.evaluation_attempts):
            artifact_refs[f"evaluation_commit_{position}"] = attempt.commit_sha
        self.revision_journal.append_projection(
            projection=record.to_dict(),
            execution_status=execution_status,
            evaluation_status=evaluation_status,
            evaluator_fingerprint_ids=tuple(
                sorted({attempt.evaluator_id for attempt in record.evaluation_attempts})
            ),
            measurements=measurements,
            artifact_refs=artifact_refs,
        )
        if existing:
            prior = existing[0]
            if prior == record:
                return prior
            if record.execution_revision == prior.execution_revision:
                raise JournalConflictError(
                    "execution journal revision conflicts with prior content"
                )
        self._save(proposed, self.revision + 1)
        self.experiments = proposed
        self.revision += 1
        return record

    def _resolve_revision_base_commits(
        self,
        workspace: Path,
        candidate_commit: str,
        base_refs: Mapping[str, str],
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for name, base_ref in base_refs.items():
            if not base_ref:
                continue
            git_command = self._require_git_command()
            result = git_command.run(
                workspace,
                (
                    "rev-parse",
                    "--verify",
                    f"{base_ref}^{{commit}}",
                ),
                output_kind=CommandOutputKind.TEXT,
            )
            if result.returncode != 0:
                raise ValueError(
                    result.stderr.decode("utf-8").strip()
                    or f"could not resolve {name} base commit"
                )
            commit = result.output.strip()
            if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                raise ValueError(f"resolved {name} base commit is invalid")
            ancestry = git_command.run(
                workspace,
                (
                    "merge-base",
                    "--is-ancestor",
                    commit,
                    candidate_commit,
                ),
                output_kind=CommandOutputKind.BINARY,
            )
            if ancestry.returncode == 1:
                raise ValueError(f"{name} base is not a candidate ancestor")
            if ancestry.returncode != 0:
                raise ValueError(ancestry.stderr.decode("utf-8", errors="strict"))
            resolved[name] = commit
        return resolved

    def _pin_revision_commit(
        self,
        node: Any,
        record: ExperimentRecord,
    ) -> tuple[str | None, str]:
        attempt_commits = tuple(
            sorted({attempt.commit_sha for attempt in record.evaluation_attempts})
        )
        if len(attempt_commits) > 1:
            raise ValueError("one execution revision cannot evaluate multiple commits")
        workspace_text = getattr(node, "workspace_dir", "")
        if not workspace_text:
            return (attempt_commits[0], "") if attempt_commits else (None, "")
        workspace = Path(workspace_text)
        if not workspace.is_dir() or not (workspace / ".git").exists():
            raise ValueError("execution revision workspace is not a Git repository")
        branch_name = record.branch_name
        require_git_ref_name(
            branch_name,
            "experiment branch",
            qualified=False,
            error_type=ValueError,
        )
        pinned_ref = (
            f"refs/kapso/execution-revisions/{self.run_id}/"
            f"node-{record.node_id}/revision-{record.execution_revision}"
        )
        git_command = self._require_git_command()
        pinned = git_command.run(
            workspace,
            (
                "for-each-ref",
                "--format=%(objectname)",
                pinned_ref,
            ),
            output_kind=CommandOutputKind.TEXT,
        )
        if pinned.returncode != 0:
            raise ValueError(
                pinned.stderr.decode("utf-8").strip() or "could not read revision pin"
            )
        commit = pinned.output.strip()
        if commit:
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("pinned execution revision commit is invalid")
            if attempt_commits and attempt_commits[0] != commit:
                raise ValueError("evaluation commit conflicts with pinned revision")
            return commit, pinned_ref
        branch_ref = f"refs/heads/{branch_name}"
        branch = git_command.run(
            workspace,
            (
                "for-each-ref",
                "--format=%(objectname)",
                branch_ref,
            ),
            output_kind=CommandOutputKind.TEXT,
        )
        if branch.returncode != 0:
            raise ValueError(
                branch.stderr.decode("utf-8").strip()
                or "could not read experiment branch"
            )
        branch_commit = branch.output.strip()
        if not branch_commit:
            if record.had_error and not attempt_commits:
                return None, ""
            raise ValueError("executed revision branch is missing")
        if not re.fullmatch(r"[0-9a-f]{40}", branch_commit):
            raise ValueError("experiment branch commit is invalid")
        commit = attempt_commits[0] if attempt_commits else branch_commit
        if attempt_commits and branch_commit != commit:
            raise ValueError("evaluation commit differs from experiment branch")
        verify = git_command.run(
            workspace,
            ("cat-file", "-e", f"{commit}^{{commit}}"),
            output_kind=CommandOutputKind.BINARY,
        )
        if verify.returncode != 0:
            raise ValueError("execution revision commit object is unavailable")
        create = git_command.run(
            workspace,
            ("update-ref", pinned_ref, commit, "0" * 40),
            output_kind=CommandOutputKind.TEXT,
        )
        if create.returncode != 0:
            raise ValueError(
                create.stderr.decode("utf-8").strip()
                or "could not pin execution revision"
            )
        return commit, pinned_ref

    def _require_git_command(self) -> BoundedGitCommand:
        if self._git_command is None:
            raise ValueError(
                "journaled Git evidence requires configured command bounds"
            )
        return self._git_command

    def get_top_experiments(self, k: int = 5) -> List[ExperimentRecord]:
        self._require_limit(k)
        eligible = [
            record
            for record in self.experiments
            if not record.had_error
            and record.evaluation_valid
            and record.normalized_utility is not None
        ]
        return sorted(
            eligible,
            key=lambda record: (record.normalized_utility, -record.node_id),
            reverse=True,
        )[:k]

    def get_recent_experiments(self, k: int = 5) -> List[ExperimentRecord]:
        self._require_limit(k)
        return self.experiments[-k:]

    def search_similar(self, query: str, k: int = 3) -> List[ExperimentRecord]:
        self._require_limit(k)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("experiment similarity query must be non-empty")
        embedded = [record for record in self.experiments if record.solution_embedding]
        if self._llm is None or not embedded:
            return self.get_recent_experiments(k)
        query_embedding = self._llm.create_embedding(query)
        ranked = sorted(
            embedded,
            key=lambda record: (
                cosine_similarity(query_embedding, record.solution_embedding),
                record.node_id,
            ),
            reverse=True,
        )
        return ranked[:k]

    def get_experiment_count(self) -> int:
        return len(self.experiments)

    def close(self) -> None:
        return None

    @staticmethod
    def _require_limit(k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("experiment retrieval limit must be positive")

    def _load(self) -> None:
        data = json.loads(
            self.path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
        if not isinstance(data, dict) or set(data) != {
            "schema",
            "run_id",
            "campaign_id",
            "revision",
            "objective_direction",
            "require_idea_links",
            "records",
        }:
            raise ValueError("experiment-history document fields are invalid")
        if data["schema"] != EXPERIMENT_HISTORY_SCHEMA:
            raise ValueError("experiment-history schema is incompatible")
        if data["objective_direction"] not in {"maximize", "minimize"}:
            raise ValueError("experiment-history objective direction is invalid")
        if not isinstance(data["require_idea_links"], bool):
            raise ValueError("experiment-history idea-link policy is invalid")
        if not isinstance(data["records"], list):
            raise ValueError("experiment-history records must be a list")
        require_identifier(data["run_id"], "experiment-history run_id")
        require_identifier(data["campaign_id"], "experiment-history campaign_id")
        if type(data["revision"]) is not int or data["revision"] < 0:
            raise ValueError("experiment-history revision must be non-negative")
        records = [ExperimentRecord.from_dict(item) for item in data["records"]]
        if [record.node_id for record in records] != list(range(len(records))):
            raise ValueError("experiment-history node ids must be contiguous")
        self.objective_direction = data["objective_direction"]
        self.require_idea_links = data["require_idea_links"]
        self.run_id = data["run_id"]
        self.campaign_id = data["campaign_id"]
        self.revision = data["revision"]
        self.experiments = records

    def _recover_from_journal(self) -> None:
        if self.revision_journal is None:
            raise ValueError("journal recovery requires a revision journal")
        events = self.revision_journal.read_events()
        if self.revision not in {len(events), len(events) - 1}:
            raise ValueError(
                "experiment history is not at a recoverable journal boundary"
            )
        projected_prefix: dict[int, ExperimentRecord] = {}
        for event in events[: self.revision]:
            projected_prefix[event.node_id] = ExperimentRecord.from_dict(
                to_json_value(event.projection)
            )
        expected_history = [
            projected_prefix[node_id] for node_id in sorted(projected_prefix)
        ]
        if self.experiments != expected_history:
            raise ValueError("experiment history conflicts with its journal prefix")
        terminal_by_node = {}
        for event in events:
            terminal_by_node[event.node_id] = ExperimentRecord.from_dict(
                to_json_value(event.projection)
            )
        terminal_records = [
            terminal_by_node[node_id] for node_id in sorted(terminal_by_node)
        ]
        if [record.node_id for record in terminal_records] != list(
            range(len(terminal_records))
        ):
            raise ValueError("journal terminal projections are not contiguous")
        for prior, terminal in zip(self.experiments, terminal_records):
            if prior.execution_revision > terminal.execution_revision:
                raise ValueError("experiment history revision exceeds journal terminal")
        if self.experiments != terminal_records or self.revision != len(events):
            self._save(terminal_records, len(events))
            self.experiments = terminal_records
            self.revision = len(events)

    def _refresh_journaled_authorities(self) -> None:
        expected_identity = (
            self.run_id,
            self.campaign_id,
            self.objective_direction,
            self.require_idea_links,
        )
        if self.path.exists():
            self._load()
        actual_identity = (
            self.run_id,
            self.campaign_id,
            self.objective_direction,
            self.require_idea_links,
        )
        if actual_identity != expected_identity:
            raise ValueError("experiment-history transaction identity changed")
        self._recover_from_journal()

    def _bind_transaction_lock(self, lock: Any) -> None:
        identity = canonical_json_bytes(
            {
                "campaign_id": self.campaign_id,
                "history_path": str(self.path.resolve()),
                "run_id": self.run_id,
            }
        )
        lock.seek(0)
        existing = lock.read()
        if existing and existing != identity:
            raise ValueError("journal is bound to another experiment history")
        if not existing:
            lock.write(identity)
            lock.flush()
            os.fsync(lock.fileno())

    def reconcile_revision_journal(self) -> None:
        """Require exact agreement with the journal terminal frontier."""
        if self.revision_journal is None:
            raise ValueError("experiment history has no revision journal")
        with self._transaction_lock_path.open("a+b") as lock:
            self._transaction_lock_path.chmod(0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._bind_transaction_lock(lock)
            self._refresh_journaled_authorities()
            events = self.revision_journal.read_events()
            terminal_by_node = {}
            for event in events:
                terminal_by_node[event.node_id] = ExperimentRecord.from_dict(
                    to_json_value(event.projection)
                )
            terminal = tuple(
                terminal_by_node[node_id] for node_id in sorted(terminal_by_node)
            )
            if tuple(self.experiments) != terminal or self.revision != len(events):
                raise ValueError("experiment history and revision journal diverged")
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _save(self, records: Iterable[ExperimentRecord], revision: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        persisted_records = list(records)
        document = {
            "schema": EXPERIMENT_HISTORY_SCHEMA,
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "revision": revision,
            "objective_direction": self.objective_direction,
            "require_idea_links": self.require_idea_links,
            "records": [record.to_dict() for record in persisted_records],
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(0o600)
        os.replace(temporary, self.path)
        self.path.chmod(0o600)
        directory_descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory_descriptor)
        os.close(directory_descriptor)


def load_store_from_env() -> ExperimentHistoryStore:
    """MCP process boundary: load paths and model routing from its launcher."""
    json_path = os.environ["EXPERIMENT_HISTORY_PATH"]
    embedding_model = os.environ.get("EXPERIMENT_EMBEDDING_MODEL")
    llm = LLMBackend(models={"embedding": embedding_model}) if embedding_model else None
    return ExperimentHistoryStore(json_path=json_path, llm=llm)


def format_experiments(experiments: Iterable[ExperimentRecord]) -> str:
    """Render complete executed content without exposing caller-owned metrics."""
    records = tuple(experiments)
    if not records:
        return "No experiments found."
    lines = []
    for record in records:
        status = (
            "FAILED"
            if record.had_error
            else (
                "INVALID EVALUATION"
                if not record.evaluation_valid
                else f"raw_score={record.raw_score}; utility={record.normalized_utility}"
            )
        )
        lines.append(f"""
## Experiment {record.node_id} ({status})

**Idea:** `{record.idea_id or 'not_applicable'}`

**Selection batch:** `{record.selection_batch_id or 'not_applicable'}`

**Parent node:** `{record.parent_node_id}`

**Fidelity:** `{record.build_fidelity}` build / `{record.eval_fidelity}` eval ({record.validation_tier})

**Solution:**
{record.solution}

**Feedback:**
{record.feedback}""")
        if record.technical_difficulties:
            lines.append(f"""
**Technical difficulties:**
{record.technical_difficulties}""")
    return "\n".join(lines)
