"""Typed budget contracts for build campaigns.

Three pieces, per the budget-aware experimentation design:

- ``BudgetSpec``: the validated campaign budget declaration, resolved from
  explicit call arguments over the mode config's optional ``budget`` block.
- ``BudgetLedger``: the orchestrator-owned accounting core. Priors come from
  the resume checkpoint; the live slice is measured monotonically; attributed
  agent spend is read from node telemetry via a provider, and non-node actors
  (e.g. evaluation maintenance) record ``CostEntry`` rows directly.
- ``BudgetSnapshot``: the frozen per-iteration read model handed to search
  strategies. Strategies read it; only the orchestrator writes budget state.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from kapso.core.config import load_config


_BUDGET_BLOCK_KEYS = {
    # The experiment cap is the facade's (a solve() argument), the three
    # budgets below the handle's words (docs/plans/redesign/wp4-config.md).
    "experiments",
    "minutes",
    "usd",
    "reserve_minutes",
    "min_iteration_seconds",
    "min_agent_timeout_seconds",
    "judge_reserve_seconds",
    # Durable-clock heartbeat interval (consumed by the orchestrator, not
    # BudgetSpec): keeps elapsed_seconds and the spend fresh between boundary
    # checkpoints.
    "checkpoint_heartbeat_seconds",
    # How long `kapso build --stop` waits after SIGTERM before SIGKILL
    # (consumed by the verb, not BudgetSpec).
    "stop_grace_seconds",
    "stop_poll_seconds",
    # Validated separately by FidelitySpec.resolve.
    "fidelity",
}


# The budget's end (docs/plans/redesign/wp4-config.md). A set reserve is
# bounded by this fraction of the budget — structural, so a short budget
# keeps its search; the delivery's protected slice without a fidelity
# measurement is the final checkout and the records, in seconds; and the
# two stop details a budget's end records beside time and cost.
RESERVE_FRACTION_BOUND = 0.25
# The packaged budget block: the one home of the floors' defaults.
PACKAGED_BUDGET = load_config()["budget"]
DELIVERY_SECONDS = 30.0
ATTEMPT_WOULD_NOT_FIT = "attempt_would_not_fit"
FINISHING_CUT = "finishing_cut"


def admission_refusal(
    remaining_seconds: float, needed_seconds: float, last_round_seconds: Optional[float],
) -> str:
    """The prose of an admission refusal, for the log, the status file's
    note and the trace."""
    if last_round_seconds is not None and last_round_seconds >= needed_seconds:
        why = f"the last took {last_round_seconds / 60:.1f} min"
    else:
        why = f"the floor is {needed_seconds / 60:.1f} min"
    return f"the remaining {max(remaining_seconds, 0.0) / 60:.1f} min would not cover an attempt ({why})"


def _require_finite_non_negative(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class BudgetSpec:
    """Campaign budget declaration. Validated, JSON-round-trippable."""

    time_budget_seconds: Optional[float] = None
    cost_budget_usd: Optional[float] = None
    # The reserve kept out of the search for finishing. None derives it
    # each round from the obligations (the harness's verification minutes,
    # the judge's observed duration, the delivery's slice); a set value is
    # bounded by RESERVE_FRACTION_BOUND of the budget.
    finalization_reserve_seconds: Optional[float] = None
    # Enforcement floors, the packaged config's (`budget.min_iteration_seconds`,
    # `budget.min_agent_timeout_seconds`): below adapter startup plus one
    # tool round-trip an agent call cannot do useful work. A live campaign
    # resolves them from its own config; the defaults serve direct
    # construction.
    min_iteration_seconds: float = PACKAGED_BUDGET["min_iteration_seconds"]
    min_agent_timeout_seconds: float = PACKAGED_BUDGET["min_agent_timeout_seconds"]
    # What the judge is kept before one has been observed in this campaign
    # (`budget.judge_reserve_seconds`): the reserve a round keeps for it and
    # the obligation the finishing stages before it leave standing.
    judge_reserve_seconds: float = PACKAGED_BUDGET["judge_reserve_seconds"]

    def __post_init__(self) -> None:
        if self.time_budget_seconds is not None:
            _require_finite_non_negative(
                "time_budget_seconds", self.time_budget_seconds
            )
        if self.cost_budget_usd is not None:
            _require_finite_non_negative(
                "cost_budget_usd", self.cost_budget_usd
            )
        if self.finalization_reserve_seconds is not None:
            _require_finite_non_negative(
                "finalization_reserve_seconds", self.finalization_reserve_seconds
            )
        _require_finite_non_negative(
            "min_iteration_seconds", self.min_iteration_seconds
        )
        _require_finite_non_negative(
            "min_agent_timeout_seconds", self.min_agent_timeout_seconds
        )
        _require_finite_non_negative(
            "judge_reserve_seconds", self.judge_reserve_seconds
        )

    @property
    def is_unbudgeted(self) -> bool:
        return (
            self.time_budget_seconds is None
            and self.cost_budget_usd is None
        )

    @classmethod
    def resolve(
        cls,
        *,
        config_block: Optional[Mapping[str, Any]] = None,
        time_budget_minutes: Optional[float] = None,
        cost_budget: Optional[float] = None,
        finalization_reserve_minutes: Optional[float] = None,
    ) -> "BudgetSpec":
        """Explicit call arguments win over the mode config's budget block."""
        block = dict(config_block or {})
        unknown = sorted(set(block) - _BUDGET_BLOCK_KEYS)
        if unknown:
            raise ValueError(
                "Unknown budget config keys: " + ", ".join(unknown)
            )

        if time_budget_minutes is None:
            time_budget_minutes = block.get("minutes")
        if cost_budget is None:
            cost_budget = block.get("usd")
        if finalization_reserve_minutes is None:
            finalization_reserve_minutes = block.get("reserve_minutes")

        kwargs: Dict[str, Any] = {}
        if "min_iteration_seconds" in block:
            kwargs["min_iteration_seconds"] = block["min_iteration_seconds"]
        if "min_agent_timeout_seconds" in block:
            kwargs["min_agent_timeout_seconds"] = block[
                "min_agent_timeout_seconds"
            ]
        if "judge_reserve_seconds" in block:
            kwargs["judge_reserve_seconds"] = block["judge_reserve_seconds"]

        return cls(
            time_budget_seconds=(
                float(time_budget_minutes) * 60.0
                if time_budget_minutes is not None
                else None
            ),
            cost_budget_usd=(
                float(cost_budget) if cost_budget is not None else None
            ),
            finalization_reserve_seconds=(
                float(finalization_reserve_minutes) * 60.0
                if finalization_reserve_minutes is not None
                else None
            ),
            **kwargs,
        )

    def bounded_reserve_seconds(self) -> Optional[float]:
        """The set reserve, bounded by a quarter of the time budget; None
        when the reserve is derived from the obligations each round."""
        if self.finalization_reserve_seconds is None:
            return None
        return self.bound_reserve(self.finalization_reserve_seconds)

    def bound_reserve(self, seconds: float) -> float:
        """A reserve, set or derived, bounded by a quarter of the time
        budget: a short budget keeps its search (a verifying harness's
        minutes alone can exceed a short budget; the finishing sessions
        then take what their rooms leave)."""
        if self.time_budget_seconds is None:
            return seconds
        return min(seconds, RESERVE_FRACTION_BOUND * self.time_budget_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_budget_seconds": self.time_budget_seconds,
            "cost_budget_usd": self.cost_budget_usd,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "min_iteration_seconds": self.min_iteration_seconds,
            "min_agent_timeout_seconds": self.min_agent_timeout_seconds,
            "judge_reserve_seconds": self.judge_reserve_seconds,
        }


@dataclass(frozen=True)
class CostEntry:
    """One attributed spend record from a non-node actor."""

    component: str  # e.g. "evaluation_maintenance"
    cost_usd: float
    duration_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.strip():
            raise ValueError("CostEntry component must be a non-empty string")
        _require_finite_non_negative("cost_usd", self.cost_usd)
        _require_finite_non_negative("duration_seconds", self.duration_seconds)


class BudgetLedger:
    """Single owner of accumulated budget state.

    Strategies never hold this object: attributed agent spend flows in from
    node telemetry (the durable record) via ``phase_cost_provider``, meters
    are sampled live, and non-node actors append ``CostEntry`` rows through
    ``record()``. Reads, derived arithmetic, and enforcement stay with the
    orchestrator.
    """

    def __init__(
        self,
        *,
        prior_elapsed_seconds: float = 0.0,
        prior_cost_usd: float = 0.0,
        prior_cost_by_component: Optional[Mapping[str, float]] = None,
    ):
        self._prior_elapsed_seconds = _require_finite_non_negative(
            "prior_elapsed_seconds", prior_elapsed_seconds
        )
        self._prior_cost_usd = _require_finite_non_negative(
            "prior_cost_usd", prior_cost_usd
        )
        self._prior_cost_by_component = dict(prior_cost_by_component or {})
        self._clock_start_monotonic: Optional[float] = None
        self._meters: Dict[str, Callable[[], float]] = {}
        self._entries: List[CostEntry] = []
        self._phase_cost_provider: Callable[[], Dict[str, float]] = dict

    def start_clock(self) -> None:
        self._clock_start_monotonic = time.monotonic()

    def set_meter(self, name: str, read: Callable[[], float]) -> None:
        self._meters[name] = read

    def set_phase_cost_provider(
        self, provider: Callable[[], Dict[str, float]]
    ) -> None:
        self._phase_cost_provider = provider

    def record(self, entry: CostEntry) -> None:
        self._entries.append(entry)

    def elapsed_seconds(self) -> float:
        live = (
            time.monotonic() - self._clock_start_monotonic
            if self._clock_start_monotonic is not None
            else 0.0
        )
        return self._prior_elapsed_seconds + live

    def _live_components(self) -> Dict[str, float]:
        live = dict(self._phase_cost_provider())
        for name, read in self._meters.items():
            live[name] = live.get(name, 0.0) + read()
        for entry in self._entries:
            live[entry.component] = (
                live.get(entry.component, 0.0) + entry.cost_usd
            )
        return live

    def total_cost(self) -> float:
        return self._prior_cost_usd + sum(self._live_components().values())

    def cost_by_component(self) -> Dict[str, float]:
        components = dict(self._prior_cost_by_component)
        for name, value in self._live_components().items():
            components[name] = components.get(name, 0.0) + value
        return components


@dataclass(frozen=True)
class BudgetSnapshot:
    """Per-iteration read model handed to strategies. Never written by them."""

    iteration_index: int
    max_iterations: int
    elapsed_seconds: float
    cost_usd: float
    time_budget_seconds: Optional[float] = None
    cost_budget_usd: Optional[float] = None
    finalization_reserve_seconds: float = 0.0
    # Sourced from BudgetSpec at construction; the default (the packaged
    # config's) only serves direct construction in tests.
    min_agent_timeout_seconds: float = PACKAGED_BUDGET["min_agent_timeout_seconds"]
    judge_reserve_seconds: float = PACKAGED_BUDGET["judge_reserve_seconds"]
    # The delivery's protected slice this round: the fidelity measurement
    # when a maintainer measures the delivered artifact, the checkout else.
    delivery_seconds: float = DELIVERY_SECONDS

    def room(
        self,
        obligations_seconds: float = 0.0,
        *,
        finishing: bool = False,
        elapsed_since_snapshot: float = 0.0,
    ) -> Optional[float]:
        """The time a session launched now may take; None when the campaign
        has no time budget. A search session's room is the searchable
        remainder, the budget less the reserve; a finishing session's is
        the whole remainder less the obligations that follow it. The time
        burned since the snapshot is discounted, so sequential phases do
        not each clamp against the iteration-start remainder."""
        if self.time_budget_seconds is None:
            return None
        base = (
            self.remaining_seconds - obligations_seconds if finishing
            else self.remaining_after_reserve
        )
        return base - elapsed_since_snapshot

    def deadline(
        self, configured_seconds: Optional[float], room: Optional[float],
    ) -> Optional[float]:
        """A session's deadline: the smaller of its configured timeout and
        its room, never raised to the floor — a session with less room than
        `min_agent_timeout_seconds` is not launched at all. None when both
        are unbounded."""
        if room is None:
            return configured_seconds
        if configured_seconds is None:
            return room
        return min(configured_seconds, room)

    @property
    def remaining_seconds(self) -> Optional[float]:
        if self.time_budget_seconds is None:
            return None
        return self.time_budget_seconds - self.elapsed_seconds

    @property
    def remaining_after_reserve(self) -> Optional[float]:
        if self.remaining_seconds is None:
            return None
        return self.remaining_seconds - self.finalization_reserve_seconds

    @property
    def remaining_usd(self) -> Optional[float]:
        if self.cost_budget_usd is None:
            return None
        return self.cost_budget_usd - self.cost_usd

    @property
    def time_fraction(self) -> float:
        if self.time_budget_seconds is None or self.time_budget_seconds == 0:
            return 0.0
        return self.elapsed_seconds / self.time_budget_seconds

    @property
    def cost_fraction(self) -> float:
        if self.cost_budget_usd is None or self.cost_budget_usd == 0:
            return 0.0
        return self.cost_usd / self.cost_budget_usd

    @property
    def progress_percent(self) -> float:
        """Budget consumed, as a 0-100 percentage."""
        return (
            max(
                self.iteration_index / self.max_iterations
                if self.max_iterations > 0
                else 0.0,
                self.time_fraction,
                self.cost_fraction,
            )
            * 100.0
        )

    @property
    def exhausted(self) -> bool:
        return self.progress_percent >= 100.0
