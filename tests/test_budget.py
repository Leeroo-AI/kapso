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
from tests.harness_support import fake_config

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
        BudgetSpec(time_budget_seconds=600, finalization_reserve_seconds=-1)
    # A set reserve is bounded by a quarter of the budget; none set is
    # derived each round (the budget's end).
    assert BudgetSpec(time_budget_seconds=600, finalization_reserve_seconds=600).bounded_reserve_seconds() == 150
    assert BudgetSpec(time_budget_seconds=600).bounded_reserve_seconds() is None
    assert BudgetSpec(finalization_reserve_seconds=90).bounded_reserve_seconds() == 90


def test_spec_resolution_precedence_and_unknown_keys():
    block = {
        "minutes": 120,
        "usd": 10.0,
        "reserve_minutes": 5,
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
        orchestrator_module, "load_config",
        lambda config_path=None: fake_config(budget={"minutes": 120}),
    )
    with_budget = _orchestrator(workspace_b)

    assert (
        with_budget.config_fingerprint == without_budget.config_fingerprint
    )
    assert with_budget._config_budget["minutes"] == 120


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


# =========================================================================
# M4: reserve gate, stop_detail, clamps, and the prompt block
# =========================================================================

def test_reserve_gate_refuses_admission_and_stays_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    # A 1-minute budget: the set reserve is bounded to a quarter (15s),
    # leaving 45s of searchable time, under the 60s iteration floor — no
    # iteration may be admitted, and the refusal is named.
    result = _orchestrator(workspace).solve(
        experiment_max_iter=3,
        time_budget_minutes=1,
        finalization_reserve_minutes=1.5,
    )

    assert result.stopped_reason == "budget_exhausted"
    assert result.stop_detail == "attempt_would_not_fit"
    assert result.iterations_run == 0

    checkpoint = RunCheckpointStore(str(workspace)).load()
    assert checkpoint.status == "running"
    assert checkpoint.last_stop == "attempt_would_not_fit"
    resumed = _orchestrator(workspace, resume=True)
    assert resumed.completed_iterations == 0


def test_cost_exhaustion_reports_stop_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    result = _orchestrator(workspace).solve(
        experiment_max_iter=3,
        cost_budget=0.5,
    )

    assert result.stopped_reason == "budget_exhausted"
    assert result.stop_detail == "cost_budget"


def test_snapshot_room_and_deadline_never_raised():
    budgeted = BudgetSnapshot(
        iteration_index=0,
        max_iterations=10,
        elapsed_seconds=3480,
        cost_usd=0.0,
        time_budget_seconds=3600,
        finalization_reserve_seconds=0.0,
    )
    assert budgeted.room() == 120 and budgeted.deadline(600, budgeted.room()) == 120
    # Sequential phases discount intra-iteration drift; the deadline is
    # the room, never raised to the floor (the predicate decides the launch).
    assert budgeted.deadline(600, budgeted.room(elapsed_since_snapshot=80)) == 40
    assert budgeted.deadline(600, budgeted.room(elapsed_since_snapshot=30)) == 90
    assert budgeted.deadline(None, budgeted.room()) == 120

    unbudgeted = BudgetSnapshot(iteration_index=0, max_iterations=10, elapsed_seconds=1e6, cost_usd=0.0)
    assert unbudgeted.room() is None
    assert unbudgeted.deadline(600, None) == 600 and unbudgeted.deadline(None, None) is None


def test_budget_status_block_renders_in_all_modes(tmp_path):
    from kapso.core.prompt_loader import load_prompt
    from kapso.execution.search_strategies.generic.strategy import (
        GenericSearch,
    )

    for template_name in (
        "execution/search_strategies/generic/prompts/ideation_claude_code.md",
        "execution/search_strategies/generic/prompts/"
        "implementation_claude_code.md",
    ):
        assert load_prompt(template_name).count("{{budget_status}}") == 1

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.harness = None
    strategy.workspace_dir = str(tmp_path)   # a prompt reads the campaign's notes from it
    strategy.experience_intro_path = None
    strategy.hosted = None
    strategy.iteration_count = 4

    strategy.budget_snapshot = None
    assert "no budget information" in strategy._render_budget_status()

    strategy.budget_snapshot = BudgetSnapshot(
        iteration_index=3,
        max_iterations=10,
        elapsed_seconds=0.0,
        cost_usd=0.0,
    )
    unbudgeted_text = strategy._render_budget_status()
    assert "Iteration 4 of 10." in unbudgeted_text
    assert "No time or cost budget is set" in unbudgeted_text

    strategy.budget_snapshot = BudgetSnapshot(
        iteration_index=3,
        max_iterations=10,
        elapsed_seconds=1800,
        cost_usd=2.5,
        time_budget_seconds=6000,
        cost_budget_usd=10.0,
        finalization_reserve_seconds=1200,
    )
    budgeted_text = strategy._render_budget_status()
    assert "Iteration 4 of 10." in budgeted_text
    assert "Elapsed 30 of 100 budgeted minutes." in budgeted_text
    assert "searchable time remaining: 50 minutes" in budgeted_text
    assert "Spent $2.50 of $10.00." in budgeted_text

    strategy.shared_artifacts_brief = "No shared-cache artifacts registered yet."
    rendered = strategy._build_ideation_prompt(
        problem="the problem",
        repo_memory_brief="memory",
    )
    assert "{{budget_status}}" not in rendered
    assert "Spent $2.50 of $10.00." in rendered
    assert "{{shared_artifacts_brief}}" not in rendered


def test_clamped_timeout_helper_uses_the_snapshot():
    import time as time_module

    from kapso.execution.search_strategies.generic.strategy import (
        GenericSearch,
    )

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.harness = None
    strategy.budget_snapshot = None
    strategy.budget_snapshot_monotonic = None
    assert strategy._clamped_timeout(600) == 600

    strategy.budget_snapshot = BudgetSnapshot(
        iteration_index=0,
        max_iterations=10,
        elapsed_seconds=0.0,
        cost_usd=0.0,
        time_budget_seconds=300,
        finalization_reserve_seconds=120,
    )
    assert strategy._clamped_timeout(600) == 180

    # The live-run overshoot this pins: ideation burned 100s of a 180s
    # remainder, so implementation gets ~80s — never the stale 180.
    strategy.budget_snapshot_monotonic = time_module.monotonic() - 100.0
    assert strategy._clamped_timeout(600) == pytest.approx(80, abs=2)
    # And never raised to the floor: 170s burned leaves 10s, which is the
    # deadline a session would get (the launch predicate refuses it).
    strategy.budget_snapshot_monotonic = time_module.monotonic() - 170.0
    assert strategy._clamped_timeout(600) == pytest.approx(10, abs=2)
