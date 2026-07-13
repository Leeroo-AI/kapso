"""Versioned evaluation attempts and comparability-class selection.

The selection rules from the budget-aware experimentation design:

- every evaluation is an append-only ``EvaluationAttempt`` tagged with the
  evaluator version that produced it;
- attempts are comparable only within a ``ComparabilityClass``
  (evaluator_id x fidelity x params) — no score ever crosses an
  evaluator_id boundary;
- ``node.score`` is a *projection* of the node's attempts under the
  campaign's canonical class (replications aggregate as their mean), so the
  existing dumb selectors keep working;
- committed/final selection walks the tier order — FULL build with full
  eval > VALIDATE (full eval of a fast-built artifact) > probe-only —
  and missing evidence always compares worse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


FIDELITIES = {"fast", "full"}

# Tier order for committed/final selection. Higher is better evidence.
TIER_FULL = 2       # full-build artifact measured at full eval
TIER_VALIDATED = 1  # fast-built artifact measured at full eval
TIER_PROBE = 0      # fast-eval evidence only (or none)


def _require_finite(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class ComparabilityClass:
    """The identity a score is only meaningful within."""

    evaluator_id: str
    fidelity: str  # "fast" | "full"
    fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator_id, str) or not self.evaluator_id:
            raise ValueError("evaluator_id must be a non-empty string")
        if self.fidelity not in FIDELITIES:
            raise ValueError(
                f"fidelity must be one of {sorted(FIDELITIES)}"
            )
        _require_finite("fraction", self.fraction)
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")


@dataclass(frozen=True)
class EvaluationAttempt:
    """One measurement of one artifact by one evaluator version."""

    commit_sha: str
    evaluator_id: str
    fidelity: str
    fraction: float
    seed: int
    score: float
    duration_seconds: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.commit_sha, str) or not self.commit_sha:
            raise ValueError("commit_sha must be a non-empty string")
        # Reuse the class validation for the shared identity fields.
        ComparabilityClass(
            evaluator_id=self.evaluator_id,
            fidelity=self.fidelity,
            fraction=self.fraction,
            seed=self.seed,
        )
        _require_finite("score", self.score)
        if self.duration_seconds is not None:
            duration = _require_finite(
                "duration_seconds", self.duration_seconds
            )
            if duration < 0:
                raise ValueError("duration_seconds must be non-negative")

    @property
    def comparability_class(self) -> ComparabilityClass:
        return ComparabilityClass(
            evaluator_id=self.evaluator_id,
            fidelity=self.fidelity,
            fraction=self.fraction,
            seed=self.seed,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_sha": self.commit_sha,
            "evaluator_id": self.evaluator_id,
            "fidelity": self.fidelity,
            "fraction": self.fraction,
            "seed": self.seed,
            "score": self.score,
            "duration_seconds": self.duration_seconds,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationAttempt":
        if not isinstance(data, Mapping):
            raise ValueError("Evaluation attempt must be an object")
        return cls(
            commit_sha=data.get("commit_sha", ""),
            evaluator_id=data.get("evaluator_id", ""),
            fidelity=data.get("fidelity", ""),
            fraction=data.get("fraction", 0.0),
            seed=data.get("seed", 0),
            score=data.get("score", float("nan")),
            duration_seconds=data.get("duration_seconds"),
            metrics=dict(data.get("metrics", {})),
        )


def attempts_in_class(
    node: Any, comparability: ComparabilityClass
) -> List[EvaluationAttempt]:
    return [
        attempt
        for attempt in getattr(node, "evaluation_attempts", [])
        if attempt.comparability_class == comparability
    ]


def project_score(
    node: Any, comparability: ComparabilityClass
) -> Optional[float]:
    """The node's score under one ruler: mean of its replications there.

    None when the node was never measured in this class — and None never
    wins a ranking.
    """
    matching = attempts_in_class(node, comparability)
    if not matching:
        return None
    return sum(attempt.score for attempt in matching) / len(matching)


def evidence_tier(node: Any, evaluator_id: str) -> int:
    """Tier of the node's best evidence under one evaluator version."""
    full_eval_attempts = [
        attempt
        for attempt in getattr(node, "evaluation_attempts", [])
        if attempt.evaluator_id == evaluator_id
        and attempt.fidelity == "full"
    ]
    if not full_eval_attempts:
        return TIER_PROBE
    if getattr(node, "build_fidelity", "full") == "full":
        return TIER_FULL
    return TIER_VALIDATED


def full_eval_score(node: Any, evaluator_id: str) -> Optional[float]:
    """Mean full-eval score under one evaluator version (any fraction=1.0)."""
    matching = [
        attempt
        for attempt in getattr(node, "evaluation_attempts", [])
        if attempt.evaluator_id == evaluator_id
        and attempt.fidelity == "full"
    ]
    if not matching:
        return None
    return sum(attempt.score for attempt in matching) / len(matching)


def select_committed_candidate(
    nodes: Iterable[Any],
    *,
    evaluator_id: str,
    maximize: bool = True,
) -> Optional[Any]:
    """Tier-order walk for the committed/final decision.

    Only evidence under the canonical evaluator version counts; stale-class
    evidence ranks as missing. Within a tier, rank by mean full-eval score;
    probe-only nodes rank by nothing here — they cannot claim the committed
    slot at all.
    """
    eligible = [
        node
        for node in nodes
        if not getattr(node, "had_error", False)
        and getattr(node, "evaluation_valid", True)
        and evidence_tier(node, evaluator_id) > TIER_PROBE
    ]
    if not eligible:
        return None
    sign = 1.0 if maximize else -1.0
    return max(
        eligible,
        key=lambda node: (
            evidence_tier(node, evaluator_id),
            sign * full_eval_score(node, evaluator_id),
        ),
    )
