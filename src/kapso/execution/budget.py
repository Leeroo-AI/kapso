"""Typed budget contracts for evolve campaigns.

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


_BUDGET_BLOCK_KEYS = {
    "time_budget_minutes",
    "cost_budget_usd",
    "finalization_reserve_minutes",
    "min_iteration_seconds",
    "min_agent_timeout_seconds",
}


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
    finalization_reserve_seconds: float = 0.0
    # Enforcement floors. Provisional values: below adapter startup plus one
    # tool round-trip an agent call cannot do useful work; revisit once
    # phase_telemetry provides measured call-duration distributions.
    min_iteration_seconds: float = 60.0
    min_agent_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.time_budget_seconds is not None:
            _require_finite_non_negative(
                "time_budget_seconds", self.time_budget_seconds
            )
        if self.cost_budget_usd is not None:
            _require_finite_non_negative(
                "cost_budget_usd", self.cost_budget_usd
            )
        _require_finite_non_negative(
            "finalization_reserve_seconds", self.finalization_reserve_seconds
        )
        _require_finite_non_negative(
            "min_iteration_seconds", self.min_iteration_seconds
        )
        _require_finite_non_negative(
            "min_agent_timeout_seconds", self.min_agent_timeout_seconds
        )
        if (
            self.time_budget_seconds is not None
            and self.finalization_reserve_seconds >= self.time_budget_seconds
        ):
            raise ValueError(
                "finalization_reserve_seconds must be smaller than "
                "time_budget_seconds"
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
            time_budget_minutes = block.get("time_budget_minutes")
        if cost_budget is None:
            cost_budget = block.get("cost_budget_usd")
        if finalization_reserve_minutes is None:
            finalization_reserve_minutes = block.get(
                "finalization_reserve_minutes"
            )

        kwargs: Dict[str, Any] = {}
        if "min_iteration_seconds" in block:
            kwargs["min_iteration_seconds"] = block["min_iteration_seconds"]
        if "min_agent_timeout_seconds" in block:
            kwargs["min_agent_timeout_seconds"] = block[
                "min_agent_timeout_seconds"
            ]

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
                else 0.0
            ),
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_budget_seconds": self.time_budget_seconds,
            "cost_budget_usd": self.cost_budget_usd,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "min_iteration_seconds": self.min_iteration_seconds,
            "min_agent_timeout_seconds": self.min_agent_timeout_seconds,
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
        """Identical arithmetic to the legacy budget_progress float."""
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
