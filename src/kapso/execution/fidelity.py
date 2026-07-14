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


# =========================================================================
# The FidelityPolicy: deterministic PROBE / VALIDATE / FULL grants
# =========================================================================

PROFILE_PROBE = "probe"
PROFILE_VALIDATE = "validate"
PROFILE_FULL = "full"

FIDELITY_MODES = {"off", "on", "auto"}

_FIDELITY_BLOCK_KEYS = {
    "mode",
    "min_affordable_full_runs",
    "build",
    "eval",
    "committed_run_fraction",
    "promotion_margin",
    "max_full_runs",
    "max_full_evals",
    "calibration_min_pairs",
    "endgame_validate",
}


@dataclass(frozen=True)
class FidelitySpec:
    """Validated fidelity configuration from the budget block."""

    mode: str = "off"
    min_affordable_full_runs: int = 4
    build_fast_fraction: Optional[float] = None  # None: no fast build dial
    eval_fast_fraction: float = 0.15
    committed_run_fraction: float = 0.45
    promotion_margin: float = 0.02
    max_full_runs: int = 2
    max_full_evals: int = 3
    calibration_min_pairs: int = 3
    endgame_validate: bool = True

    def __post_init__(self) -> None:
        if self.mode not in FIDELITY_MODES:
            raise ValueError(
                f"fidelity mode must be one of {sorted(FIDELITY_MODES)}"
            )
        if self.build_fast_fraction is not None and not (
            0 < self.build_fast_fraction <= 1
        ):
            raise ValueError("build fast_fraction must be in (0, 1]")
        if not 0 < self.eval_fast_fraction <= 1:
            raise ValueError("eval fast_fraction must be in (0, 1]")
        if not 0 < self.committed_run_fraction < 1:
            raise ValueError("committed_run_fraction must be in (0, 1)")

    @classmethod
    def resolve(cls, block: Optional[Mapping[str, Any]]) -> "FidelitySpec":
        if not block:
            return cls()
        unknown = sorted(set(block) - _FIDELITY_BLOCK_KEYS)
        if unknown:
            raise ValueError(
                "Unknown fidelity config keys: " + ", ".join(unknown)
            )
        build_block = dict(block.get("build") or {})
        eval_block = dict(block.get("eval") or {})
        kwargs: Dict[str, Any] = {
            "mode": block.get("mode", "off"),
        }
        if "min_affordable_full_runs" in block:
            kwargs["min_affordable_full_runs"] = block[
                "min_affordable_full_runs"
            ]
        if "fast_fraction" in build_block:
            kwargs["build_fast_fraction"] = build_block["fast_fraction"]
        if "fast_fraction" in eval_block:
            kwargs["eval_fast_fraction"] = eval_block["fast_fraction"]
        for key in (
            "committed_run_fraction",
            "promotion_margin",
            "max_full_runs",
            "max_full_evals",
            "calibration_min_pairs",
            "endgame_validate",
        ):
            if key in block:
                kwargs[key] = block[key]
        return cls(**kwargs)


@dataclass(frozen=True)
class FidelityDecision:
    """One iteration's granted workload profile. Executive-issued."""

    profile: str  # probe | validate | full
    build_fidelity: str
    eval_fidelity: str
    eval_fraction: float
    target_node_id: Optional[int] = None  # validate/full promotion target
    reserve_run: bool = False
    deadline_seconds: Optional[float] = None  # eval-only run deadline
    reason: str = ""


FULL_PASSTHROUGH = FidelityDecision(
    profile=PROFILE_FULL,
    build_fidelity="full",
    eval_fidelity="full",
    eval_fraction=1.0,
    reason="fidelity off: every experiment runs full-size",
)


class FidelityPolicy:
    """Deterministic profile grants over measured state.

    Every counter is derived from node history, so a resumed campaign
    reaches identical decisions from the durable record — no hidden state.
    The LLM may request; only this arithmetic grants.
    """

    def __init__(
        self,
        *,
        spec: FidelitySpec,
        evaluator_id: str,
        subsample_seed: int,
        full_eval_upper_seconds: float,
        fast_eval_upper_seconds: float,
    ):
        self.spec = spec
        self.evaluator_id = evaluator_id
        self.subsample_seed = subsample_seed
        self.full_eval_upper_seconds = float(full_eval_upper_seconds)
        self.fast_eval_upper_seconds = float(fast_eval_upper_seconds)

    # -- derived arithmetic ------------------------------------------------

    def reserve_seconds(self, time_budget_seconds: float) -> float:
        """The escrowed committed slot: build cap + full eval, by
        construction committed_run_fraction of the budget."""
        return self.spec.committed_run_fraction * time_budget_seconds

    def full_champion(self, nodes: Iterable[Any]) -> Optional[Any]:
        """The committed candidate at FULL tier under the head evaluator."""
        committed = select_committed_candidate(
            nodes, evaluator_id=self.evaluator_id
        )
        if committed is not None and (
            evidence_tier(committed, self.evaluator_id) == TIER_FULL
        ):
            return committed
        return None

    def effective_reserve_seconds(
        self, time_budget_seconds: float, nodes: Iterable[Any]
    ) -> float:
        """The escrow, shrunk once a full-measured champion exists.

        Unchampioned: the full committed slot. Championed: only the
        contingency residual — re-securing an already-built artifact under
        a new evaluator head costs one full evaluation, not a build — and
        the freed difference flows back into the searchable window. The
        insurance stays exactly as large as the risk it still covers.
        """
        base = self.reserve_seconds(time_budget_seconds)
        if self.full_champion(nodes) is not None:
            return min(base, self.full_eval_upper_seconds)
        return base

    def build_cap_seconds(self, time_budget_seconds: float) -> float:
        return max(
            0.0,
            self.reserve_seconds(time_budget_seconds)
            - self.full_eval_upper_seconds,
        )

    def full_run_bound_seconds(self, time_budget_seconds: float) -> float:
        return self.reserve_seconds(time_budget_seconds)

    def enabled(self, time_budget_seconds: Optional[float]) -> bool:
        if self.spec.mode == "off" or time_budget_seconds is None:
            return False
        if self.spec.mode == "on":
            return True
        affordable_full_runs = time_budget_seconds / self.full_run_bound_seconds(
            time_budget_seconds
        )
        return affordable_full_runs < self.spec.min_affordable_full_runs

    # -- history-derived counters -------------------------------------------

    def _fast_class(self) -> ComparabilityClass:
        return ComparabilityClass(
            evaluator_id=self.evaluator_id,
            fidelity="fast",
            fraction=self.spec.eval_fast_fraction,
            seed=self.subsample_seed,
        )

    def full_evals_used(self, nodes: Iterable[Any]) -> int:
        return sum(
            1
            for node in nodes
            if getattr(node, "build_fidelity", "full") == "fast"
            and evidence_tier(node, self.evaluator_id) == TIER_VALIDATED
        )

    def full_runs_used(self, nodes: Iterable[Any]) -> int:
        return sum(
            1
            for node in nodes
            if evidence_tier(node, self.evaluator_id) == TIER_FULL
        )

    def calibration_pairs(self, nodes: Iterable[Any]) -> int:
        fast_class = self._fast_class()
        return sum(
            1
            for node in nodes
            if project_score(node, fast_class) is not None
            and full_eval_score(node, self.evaluator_id) is not None
        )

    def _fast_leader(self, nodes: List[Any]) -> Optional[Any]:
        fast_class = self._fast_class()
        scored = [
            (project_score(node, fast_class), node)
            for node in nodes
            if not getattr(node, "had_error", False)
            and getattr(node, "evaluation_valid", True)
            and project_score(node, fast_class) is not None
        ]
        if not scored:
            return None
        return max(scored, key=lambda pair: pair[0])[1]

    # -- the grant ladder ----------------------------------------------------

    def decide(
        self,
        *,
        nodes: List[Any],
        remaining_after_reserve: Optional[float],
        probe_estimate_seconds: float,
        reserve_run: bool = False,
    ) -> FidelityDecision:
        """Grant this iteration's profile. Pure arithmetic; no estimation
        beyond measured uppers and the declared fractions."""
        spec = self.spec
        build_fast = (
            spec.build_fast_fraction is not None
        )

        if reserve_run:
            committed = select_committed_candidate(
                nodes, evaluator_id=self.evaluator_id
            )
            target = committed or self._fast_leader(nodes)
            return FidelityDecision(
                profile=PROFILE_FULL,
                build_fidelity="full",
                eval_fidelity="full",
                eval_fraction=1.0,
                target_node_id=(
                    target.node_id if target is not None else None
                ),
                reserve_run=True,
                reason=(
                    "reserve run: the escrowed full-size attempt on the "
                    "best recipe"
                ),
            )

        fast_leader = self._fast_leader(nodes)
        committed = select_committed_candidate(
            nodes, evaluator_id=self.evaluator_id
        )
        leader_unvalidated = (
            fast_leader is not None
            and evidence_tier(fast_leader, self.evaluator_id) == TIER_PROBE
        )

        # VALIDATE: buy trust in the fast leader before committing to it.
        if (
            leader_unvalidated
            and remaining_after_reserve is not None
            and self.full_eval_upper_seconds <= remaining_after_reserve
            and self.full_evals_used(nodes) < spec.max_full_evals
        ):
            fast_class = self._fast_class()
            leader_score = project_score(fast_leader, fast_class)
            incumbent_score = (
                project_score(committed, fast_class)
                if committed is not None
                else None
            )
            clears_margin = incumbent_score is None or (
                leader_score - incumbent_score >= spec.promotion_margin
            )
            endgame = (
                spec.endgame_validate
                and remaining_after_reserve
                <= probe_estimate_seconds + self.full_eval_upper_seconds
            )
            if clears_margin or endgame:
                return FidelityDecision(
                    profile=PROFILE_VALIDATE,
                    build_fidelity=fast_leader.build_fidelity,
                    eval_fidelity="full",
                    eval_fraction=1.0,
                    target_node_id=fast_leader.node_id,
                    # The estimate gates admission (above); the deadline is
                    # the affordability window. Real artifacts carry real
                    # training cost a baseline-calibrated estimate cannot
                    # bound — a validate may run right up to the reserve.
                    deadline_seconds=remaining_after_reserve,
                    reason=(
                        "endgame validate: confirming the unvalidated fast "
                        "leader before the reserve decision"
                        if endgame
                        else "fast leader cleared the promotion margin"
                    ),
                )

        # FULL mid-campaign: only when the arithmetic affords it outside the
        # escrow, the pair table has earned trust, and slots remain.
        if (
            committed is not None
            and evidence_tier(committed, self.evaluator_id)
            == TIER_VALIDATED
            and remaining_after_reserve is not None
            and self._mid_campaign_full_affordable(remaining_after_reserve)
            and self.full_runs_used(nodes) < spec.max_full_runs
            and self.calibration_pairs(nodes) >= spec.calibration_min_pairs
        ):
            return FidelityDecision(
                profile=PROFILE_FULL,
                build_fidelity="full",
                eval_fidelity="full",
                eval_fraction=1.0,
                target_node_id=committed.node_id,
                reason=(
                    "mid-campaign full run: validated leader, calibrated "
                    "pairs, and the arithmetic affords it outside the escrow"
                ),
            )

        return FidelityDecision(
            profile=PROFILE_PROBE,
            build_fidelity="fast" if build_fast else "full",
            eval_fidelity="fast",
            eval_fraction=spec.eval_fast_fraction,
            reason="default probe: searching outside the escrow",
        )

    def _mid_campaign_full_affordable(
        self, remaining_after_reserve: float
    ) -> bool:
        # A mid-campaign full run must fit entirely outside the escrow.
        return (
            self.full_eval_upper_seconds * 2 <= remaining_after_reserve
        )
