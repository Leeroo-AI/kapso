"""Hermetic tests for the typed budget contracts (design doc, M3).

Pins: BudgetSpec validation and resolution precedence, BudgetSnapshot
arithmetic parity with the legacy budget_progress formula, BudgetLedger
composition, the fingerprint carve-out that makes budget top-up possible,
and the observe_budget hook reaching strategies each iteration.
"""

from pathlib import Path

import git
import pytest

from kapso.execution.budget import (
    BudgetLedger,
    BudgetSnapshot,
    BudgetSpec,
    CostEntry,
)
from kapso.execution.run_checkpoint import RunCheckpointStore

from tests.test_run_checkpoint import (
    _init_git_workspace,
    _orchestrator,
    _patch_orchestrator,
)


# =========================================================================
# BudgetSpec
# =========================================================================

def test_spec_rejects_invalid_values():
    with pytest.raises(ValueError, match="time_budget_seconds"):
        BudgetSpec(time_budget_seconds=-1)
    with pytest.raises(ValueError, match="cost_budget_usd"):
        BudgetSpec(cost_budget_usd=float("nan"))
    with pytest.raises(ValueError, match="finalization_reserve_seconds"):
        BudgetSpec(time_budget_seconds=600, finalization_reserve_seconds=600)


def test_spec_resolution_precedence_and_unknown_keys():
    block = {
        "time_budget_minutes": 120,
        "cost_budget_usd": 10.0,
        "finalization_reserve_minutes": 5,
        "min_agent_timeout_seconds": 30,
    }

    from_block = BudgetSpec.resolve(config_block=block)
    assert from_block.time_budget_seconds == 7200
    assert from_block.cost_budget_usd == 10.0
    assert from_block.finalization_reserve_seconds == 300
    assert from_block.min_agent_timeout_seconds == 30

    overridden = BudgetSpec.resolve(
        config_block=block,
        time_budget_minutes=30,
        finalization_reserve_minutes=1,
    )
    assert overridden.time_budget_seconds == 1800
    assert overridden.finalization_reserve_seconds == 60
    assert overridden.cost_budget_usd == 10.0  # block still fills the gap

    with pytest.raises(ValueError, match="Unknown budget config keys"):
        BudgetSpec.resolve(config_block={"time_budget_hours": 1})

    assert BudgetSpec.resolve(config_block=None).is_unbudgeted


# =========================================================================
# BudgetSnapshot
# =========================================================================

def test_snapshot_arithmetic_matches_legacy_progress_formula():
    snapshot = BudgetSnapshot(
        iteration_index=3,
        max_iterations=10,
        elapsed_seconds=1800,
        cost_usd=2.0,
        time_budget_seconds=3600,
        cost_budget_usd=10.0,
        finalization_reserve_seconds=600,
    )

    legacy = max(3 / 10, 1800 / 3600, 2.0 / 10.0) * 100
    assert snapshot.progress_percent == legacy
    assert snapshot.remaining_seconds == 1800
    assert snapshot.remaining_after_reserve == 1200
    assert snapshot.remaining_usd == 8.0
    assert not snapshot.exhausted

    unbudgeted = BudgetSnapshot(
        iteration_index=9,
        max_iterations=10,
        elapsed_seconds=1e9,
        cost_usd=1e9,
    )
    assert unbudgeted.remaining_seconds is None
    assert unbudgeted.remaining_after_reserve is None
    assert unbudgeted.remaining_usd is None
    assert unbudgeted.progress_percent == 90.0


# =========================================================================
# BudgetLedger
# =========================================================================

def test_ledger_composes_priors_meters_phases_and_entries():
    ledger = BudgetLedger(
        prior_elapsed_seconds=100.0,
        prior_cost_usd=1.0,
        prior_cost_by_component={"ideation": 0.6, "llm_backend": 0.4},
    )
    ledger.set_meter("llm_backend", lambda: 0.25)
    ledger.set_phase_cost_provider(lambda: {"ideation": 0.5})
    ledger.record(
        CostEntry(
            component="evaluation_maintenance",
            cost_usd=0.125,
            duration_seconds=30.0,
        )
    )

    assert ledger.total_cost() == pytest.approx(1.0 + 0.25 + 0.5 + 0.125)
    assert ledger.cost_by_component() == pytest.approx(
        {
            "ideation": 1.1,
            "llm_backend": 0.65,
            "evaluation_maintenance": 0.125,
        }
    )
    # Clock not started: only prior elapsed.
    assert ledger.elapsed_seconds() == 100.0
    ledger.start_clock()
    assert ledger.elapsed_seconds() >= 100.0

    with pytest.raises(ValueError, match="component"):
        CostEntry(component=" ", cost_usd=0.1, duration_seconds=1.0)


# =========================================================================
# Fingerprint carve-out + top-up + observe_budget (orchestrator-level)
# =========================================================================

def test_budget_block_is_excluded_from_config_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kapso.execution.orchestrator as orchestrator_module

    workspace_a = tmp_path / "a"
    workspace_b = tmp_path / "b"
    _init_git_workspace(workspace_a)
    _init_git_workspace(workspace_b)
    _patch_orchestrator(monkeypatch)

    without_budget = _orchestrator(workspace_a)

    monkeypatch.setattr(
        orchestrator_module,
        "load_mode_config",
        lambda config_path, mode: {
            "search_strategy": {"type": "generic", "params": {}},
            "budget": {"time_budget_minutes": 120},
        },
    )
    with_budget = _orchestrator(workspace_b)

    assert (
        with_budget.config_fingerprint == without_budget.config_fingerprint
    )
    assert with_budget._config_budget == {"time_budget_minutes": 120}


def test_budget_exhausted_campaign_resumes_with_a_bigger_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    first = _orchestrator(workspace).solve(
        experiment_max_iter=3,
        cost_budget=0.5,
    )
    assert first.stopped_reason == "budget_exhausted"
    assert RunCheckpointStore(str(workspace)).load().last_stop == "cost_budget"

    # The top-up: same campaign, bigger budget — fingerprint unchanged, the
    # durable ledger seeds from the checkpoint, the loop continues.
    resumed = _orchestrator(workspace, resume=True)
    result = resumed.solve(experiment_max_iter=1, cost_budget=50.0)

    assert result.cumulative_iterations == 2
    assert result.total_cost == 2.0


def test_strategy_receives_a_snapshot_every_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    orchestrator = _orchestrator(workspace)
    orchestrator.solve(
        experiment_max_iter=1,
        time_budget_minutes=60,
        finalization_reserve_minutes=10,
    )

    snapshot = orchestrator.search_strategy.budget_snapshot
    assert snapshot is not None
    assert snapshot.iteration_index == 0
    assert snapshot.max_iterations == 1
    assert snapshot.time_budget_seconds == 3600
    assert snapshot.finalization_reserve_seconds == 600
    assert snapshot.remaining_after_reserve is not None
