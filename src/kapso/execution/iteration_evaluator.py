"""Generic, observational evaluation of finalized experiment candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from kapso.execution.search_strategies.base import SearchNode


class IterationEvaluationError(RuntimeError):
    """Raised when an external candidate evaluation cannot be completed."""


class IterationEvaluationValidationError(IterationEvaluationError):
    """Raised when an evaluator returns malformed metrics or metadata."""


@dataclass(frozen=True)
class IterationEvaluationContext:
    """Isolated candidate context passed to an iteration evaluator.

    ``workspace_dir`` is a temporary detached Git worktree at ``git_ref``.
    ``node`` is a snapshot, so mutations cannot change Kapso's search state.
    """

    iteration: int
    goal: str
    workspace_dir: Path
    git_ref: str
    parent_ref: str
    node: "SearchNode"


@dataclass(frozen=True)
class IterationEvaluationResult:
    """Observational metrics returned by an iteration evaluator."""

    metrics: Mapping[str, float]
    primary_metric: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


IterationEvaluator = Callable[
    [IterationEvaluationContext],
    IterationEvaluationResult,
]


def normalize_failure_policy(policy: str) -> str:
    """Validate and normalize the evaluator failure policy."""
    if not isinstance(policy, str):
        raise ValueError(
            "iteration_evaluator_failure_policy must be 'record' or 'raise'"
        )
    normalized = policy.strip().lower()
    if normalized not in {"record", "raise"}:
        raise ValueError(
            "iteration_evaluator_failure_policy must be 'record' or 'raise'"
        )
    return normalized


def normalize_metrics(
    metrics: Mapping[str, Any],
    primary_metric: Optional[str],
) -> Tuple[Dict[str, float], Optional[str]]:
    """Return finite numeric metrics with a valid optional primary key."""
    if not isinstance(metrics, Mapping):
        raise IterationEvaluationValidationError(
            "Iteration evaluator metrics must be a mapping"
        )

    normalized: Dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise IterationEvaluationValidationError(
                "Iteration evaluator metric names must be non-empty strings"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise IterationEvaluationValidationError(
                f"Iteration evaluator metric {name!r} must be finite and "
                "numeric"
            )
        normalized[name] = float(value)

    if primary_metric is not None and (
        not isinstance(primary_metric, str)
        or primary_metric not in normalized
    ):
        raise IterationEvaluationValidationError(
            "Iteration evaluator primary_metric must name a returned metric"
        )
    return normalized, primary_metric


def normalize_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate, detach, and normalize JSON-compatible metadata."""
    if not isinstance(metadata, Mapping):
        raise IterationEvaluationValidationError(
            "Iteration evaluator metadata must be a mapping"
        )

    def validate_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise IterationEvaluationValidationError(
                    "Iteration evaluator metadata keys must be strings"
                )
            for nested in value.values():
                validate_keys(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                validate_keys(nested)

    validate_keys(metadata)
    try:
        encoded = json.dumps(dict(metadata), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise IterationEvaluationValidationError(
            "Iteration evaluator metadata must be JSON compatible"
        ) from exc
    return json.loads(encoded)


def normalize_result(
    result: IterationEvaluationResult,
) -> IterationEvaluationResult:
    """Validate and detach an evaluator result from caller-owned objects."""
    if not isinstance(result, IterationEvaluationResult):
        raise IterationEvaluationValidationError(
            "Iteration evaluator must return IterationEvaluationResult"
        )
    metrics, primary_metric = normalize_metrics(
        result.metrics,
        result.primary_metric,
    )
    metadata = normalize_metadata(result.metadata)
    return IterationEvaluationResult(
        metrics=metrics,
        primary_metric=primary_metric,
        metadata=metadata,
    )
