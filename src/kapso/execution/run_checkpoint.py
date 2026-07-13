"""Versioned, atomic persistence for resumable evolution campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class RunCheckpointError(RuntimeError):
    """Base class for checkpoint validation and persistence failures."""


class RunCheckpointMissingError(RunCheckpointError):
    """Raised when strict resume cannot find a checkpoint."""


class RunCheckpointCorruptError(RunCheckpointError):
    """Raised when checkpoint JSON is malformed or structurally invalid."""


class RunCheckpointIncompatibleError(RunCheckpointError):
    """Raised when a checkpoint does not match the requested campaign."""


class RunCheckpointCompletedError(RunCheckpointError):
    """Raised when attempting to resume a completed campaign."""


def goal_hash(goal: str) -> str:
    """Return a stable hash for an exact campaign goal."""
    return hashlib.sha256((goal or "").encode("utf-8")).hexdigest()


def config_fingerprint(config: Dict[str, Any]) -> str:
    """Return a stable hash for JSON-like strategy configuration."""
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class RunCheckpoint:
    """Complete state required to continue an evolution campaign."""

    schema_version: int
    strategy_type: str
    goal: str
    goal_hash: str
    config_fingerprint: str
    status: str
    completed_iterations: int
    cumulative_cost: float
    current_feedback: Optional[str]
    strategy_state: Dict[str, Any]

    SCHEMA_VERSION = 1
    VALID_STATUSES = {"running", "completed"}

    @classmethod
    def create(
        cls,
        *,
        strategy_type: str,
        goal: str,
        config_fingerprint: str,
        status: str,
        completed_iterations: int,
        cumulative_cost: float,
        current_feedback: Optional[str],
        strategy_state: Dict[str, Any],
    ) -> "RunCheckpoint":
        checkpoint = cls(
            schema_version=cls.SCHEMA_VERSION,
            strategy_type=strategy_type,
            goal=goal,
            goal_hash=goal_hash(goal),
            config_fingerprint=config_fingerprint,
            status=status,
            completed_iterations=completed_iterations,
            cumulative_cost=cumulative_cost,
            current_feedback=current_feedback,
            strategy_state=strategy_state,
        )
        checkpoint.validate_structure()
        return checkpoint

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunCheckpoint":
        if not isinstance(data, dict):
            raise RunCheckpointCorruptError(
                "Run checkpoint must contain a JSON object"
            )

        required = {
            "schema_version",
            "strategy_type",
            "goal",
            "goal_hash",
            "config_fingerprint",
            "status",
            "completed_iterations",
            "cumulative_cost",
            "current_feedback",
            "strategy_state",
        }
        missing = sorted(required - set(data))
        if missing:
            raise RunCheckpointCorruptError(
                f"Run checkpoint is missing fields: {', '.join(missing)}"
            )

        checkpoint = cls(**{key: data[key] for key in required})
        checkpoint.validate_structure()
        return checkpoint

    def validate_structure(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != self.SCHEMA_VERSION
        ):
            raise RunCheckpointIncompatibleError(
                "Unsupported run checkpoint schema version "
                f"{self.schema_version!r}; expected {self.SCHEMA_VERSION}"
            )
        if not isinstance(self.strategy_type, str) or not self.strategy_type:
            raise RunCheckpointCorruptError(
                "Run checkpoint strategy_type must be a non-empty string"
            )
        if not isinstance(self.goal, str):
            raise RunCheckpointCorruptError(
                "Run checkpoint goal must be a string"
            )
        if self.goal_hash != goal_hash(self.goal):
            raise RunCheckpointCorruptError(
                "Run checkpoint goal hash does not match its stored goal"
            )
        if (
            not isinstance(self.config_fingerprint, str)
            or not self.config_fingerprint
        ):
            raise RunCheckpointCorruptError(
                "Run checkpoint config_fingerprint must be a string"
            )
        if (
            not isinstance(self.status, str)
            or self.status not in self.VALID_STATUSES
        ):
            raise RunCheckpointCorruptError(
                "Run checkpoint status must be 'running' or 'completed'"
            )
        if (
            isinstance(self.completed_iterations, bool)
            or not isinstance(self.completed_iterations, int)
            or self.completed_iterations < 0
        ):
            raise RunCheckpointCorruptError(
                "Run checkpoint completed_iterations must be non-negative"
            )
        if (
            isinstance(self.cumulative_cost, bool)
            or not isinstance(self.cumulative_cost, (int, float))
            or not math.isfinite(float(self.cumulative_cost))
            or self.cumulative_cost < 0
        ):
            raise RunCheckpointCorruptError(
                "Run checkpoint cumulative_cost must be finite and non-negative"
            )
        if self.current_feedback is not None and not isinstance(
            self.current_feedback, str
        ):
            raise RunCheckpointCorruptError(
                "Run checkpoint current_feedback must be a string or null"
            )
        if not isinstance(self.strategy_state, dict):
            raise RunCheckpointCorruptError(
                "Run checkpoint strategy_state must be an object"
            )
        try:
            json.dumps(self.strategy_state, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RunCheckpointCorruptError(
                "Run checkpoint strategy_state must be JSON serializable"
            ) from exc

    def validate_resume(
        self,
        *,
        goal: str,
        strategy_type: str,
        config_fingerprint: str,
    ) -> None:
        if self.status == "completed":
            raise RunCheckpointCompletedError(
                "This evolution campaign is already completed and cannot be "
                "resumed without a future explicit override"
            )
        if self.goal_hash != goal_hash(goal) or self.goal != goal:
            raise RunCheckpointIncompatibleError(
                "Run checkpoint goal does not match the requested goal"
            )
        if self.strategy_type != strategy_type:
            raise RunCheckpointIncompatibleError(
                "Run checkpoint strategy mismatch: expected "
                f"{self.strategy_type!r}, requested {strategy_type!r}"
            )
        if self.config_fingerprint != config_fingerprint:
            raise RunCheckpointIncompatibleError(
                "Run checkpoint configuration does not match the requested run"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RunCheckpointStore:
    """Read and atomically replace ``.kapso/run_state.json``."""

    RELATIVE_PATH = Path(".kapso") / "run_state.json"
    LEGACY_RELATIVE_PATH = Path("checkpoint.pkl")

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.path = self.workspace_dir / self.RELATIVE_PATH
        self.legacy_path = self.workspace_dir / self.LEGACY_RELATIVE_PATH

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> RunCheckpoint:
        if not self.path.is_file():
            raise RunCheckpointMissingError(
                f"No run checkpoint found at {self.path}"
            )
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunCheckpointCorruptError(
                f"Could not read run checkpoint {self.path}: {exc}"
            ) from exc
        return RunCheckpoint.from_dict(data)

    def save(self, checkpoint: RunCheckpoint) -> None:
        checkpoint.validate_structure()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=".run_state.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    checkpoint.to_dict(),
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def mark_legacy_migrated(self) -> Path:
        destination = self.legacy_path.with_suffix(".pkl.migrated")
        os.replace(self.legacy_path, destination)
        return destination
