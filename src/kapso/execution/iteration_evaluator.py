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
    baseline_metrics: Mapping[str, float] = field(default_factory=dict)
    metric_directions: Mapping[str, str] = field(default_factory=dict)


IterationEvaluator = Callable[
    [IterationEvaluationContext],
    IterationEvaluationResult,
]


def compare_metrics(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    directions: Mapping[str, str],
) -> Dict[str, Any]:
    """Compare shared metrics, with explicit maximize/minimize semantics.

    Inputs must already be normalized. A missing baseline or direction is not
    evidence of improvement. Deltas are always candidate minus baseline.
    """
    comparisons = {}
    for name, value in candidate.items():
        if name not in baseline:
            continue
        delta = value - baseline[name]
        direction = directions.get(name)
        verdict = "unclassified"
        if direction is not None:
            improvement = delta if direction == "maximize" else -delta
            verdict = (
                "improved"
                if improvement > 0
                else "regressed" if improvement < 0 else "unchanged"
            )
        comparisons[name] = {
            "baseline": baseline[name],
            "candidate": value,
            "delta": delta,
            "direction": direction,
            "verdict": verdict,
        }
    return {
        "metrics": comparisons,
        "missing_baseline": sorted(set(candidate) - set(baseline)),
    }


def comparison_summary(metadata: Mapping[str, Any]) -> str:
    """Render a normalized comparison report; malformed reports raise."""
    if "baseline_comparison" not in metadata:
        return ""
    report = metadata["baseline_comparison"]
    parts = [
        f"{name}: {entry['candidate']:.6g} "
        f"vs {entry['baseline']:.6g} "
        f"(delta {entry['delta']:+.6g}, {entry['verdict']})"
        for name, entry in report["metrics"].items()
    ]
    if report["missing_baseline"]:
        parts.append("no baseline: " + ", ".join(report["missing_baseline"]))
    return "; ".join(parts)


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
    baseline, _ = normalize_metrics(result.baseline_metrics, None)
    unknown_baseline = set(baseline) - set(metrics)
    if unknown_baseline:
        raise IterationEvaluationValidationError(
            "baseline_metrics must name returned metrics: "
            + ", ".join(sorted(unknown_baseline))
        )
    directions = result.metric_directions
    if not isinstance(directions, Mapping) or any(
        name not in metrics
        or not isinstance(direction, str)
        or direction not in {"maximize", "minimize"}
        for name, direction in directions.items()
    ):
        raise IterationEvaluationValidationError(
            "metric_directions must map returned metrics "
            "to maximize or minimize"
        )
    comparison = (
        compare_metrics(metrics, baseline, directions) if baseline else None
    )
    # Reserved even without a baseline: the watch renderer reads this key as
    # a derived report, so caller metadata may never supply it. A result that
    # is already normalized carries exactly the report recomputed here.
    if "baseline_comparison" in metadata and (
        comparison is None or metadata["baseline_comparison"] != comparison
    ):
        raise IterationEvaluationValidationError(
            "baseline_comparison is reserved for the report Kapso derives "
            "from baseline_metrics"
        )
    if comparison is not None:
        metadata["baseline_comparison"] = comparison
        metadata = normalize_metadata(metadata)
    return IterationEvaluationResult(
        metrics=metrics,
        primary_metric=primary_metric,
        metadata=metadata,
        baseline_metrics=baseline,
        metric_directions=dict(directions),
    )
