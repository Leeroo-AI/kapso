"""The budget's end (docs/plans/redesign/wp4-config.md, change 7): search
and finishing sessions clamped against their own remainders, the launch
predicate (no session under the floor), the unjudged attempt, admission
from the last round's duration, the reserve derived from the obligations
or bounded when set, and the two stop details said by name."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.execution.orchestrator as orchestrator_module
from kapso.harness import HarnessError
from kapso.execution.budget import (
    ATTEMPT_WOULD_NOT_FIT,
    DELIVERY_SECONDS,
    FINISHING_CUT,
    BudgetSnapshot,
    BudgetSpec,
    admission_refusal,
)
from kapso.execution.run_checkpoint import RunCheckpointStore
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from tests.harness_support import fake_config
from tests.test_run_checkpoint import _init_git_workspace, _orchestrator, _patch_orchestrator


def status_file(orchestrator) -> dict:
    return json.loads(Path(orchestrator.operation_status.path).read_text())


def snapshot(*, elapsed, budget=600.0, reserve=60.0, floor=30.0, delivery=10.0, judge=30.0):
    return BudgetSnapshot(
        iteration_index=0, max_iterations=5, elapsed_seconds=elapsed, cost_usd=0.0,
        time_budget_seconds=budget, finalization_reserve_seconds=reserve,
        min_agent_timeout_seconds=floor, judge_reserve_seconds=judge, delivery_seconds=delivery,
    )


def strategy_with(view, *, verifies=False, cost=False):
    s = GenericSearch.__new__(GenericSearch)
    s.budget_snapshot = view
    s.budget_snapshot_monotonic = None
    s.harness = SimpleNamespace(has_content=lambda module: verifies) if verifies else None
    s.cost_config = {"enabled": cost}
    s.finishing_cut = None
    return s


def test_search_and_finishing_rooms():
    """A search session gets the searchable remainder; a finishing one the
    whole remainder less what follows it — the evaluation and the
    verification at the floor, the judge at its reserve (here equal to
    the floor) — and the delivery's slice last."""
    s = strategy_with(snapshot(elapsed=500), verifies=True)
    # remaining 100, reserve 60: the search room is 40
    assert s._clamped_timeout(900) == 40
    # the judge: 100 - delivery 10 = 90; verification: 100 - judge 30 - 10 = 60
    assert s._finishing_deadline("judge", 900) == (90, None)
    assert s._finishing_deadline("verification", 900) == (60, None)
    # the reconstruction with no cost mode: verification + judge + delivery = 70
    assert s._finishing_deadline("reconstruction", None) == (30, None)
    # a configured timeout smaller than the room wins
    assert s._finishing_deadline("judge", 45) == (45, None)
    # without a harness verification, the verification is not an obligation
    plain = strategy_with(snapshot(elapsed=500))
    assert plain._finishing_deadline("reconstruction", None) == (60, None)
    # in cost mode the evaluation run is one
    costed = strategy_with(snapshot(elapsed=500), cost=True)
    assert costed._finishing_deadline("reconstruction", None) == (30, None)
    # The judge is kept its reserve, not the floor: a verification's room
    # ends where the judge's minutes begin (the release gate's hour: a
    # judge kept at the floor got 59 s and the attempt went unjudged).
    kept = strategy_with(snapshot(elapsed=500, judge=50), verifies=True)
    assert kept._finishing_deadline("verification", 900) == (40, None)
    assert kept._finishing_deadline("judge", 900) == (90, None)
    assert kept._finishing_deadline("reconstruction", None) == (None, "the budget left 10s for the reconstruction, under the 30s floor")


def test_a_session_under_the_floor_is_not_launched_and_says_why():
    s = strategy_with(snapshot(elapsed=565), verifies=True)
    # remaining 35: the judge's room is 25, under the 30s floor
    deadline, reason = s._finishing_deadline("judge", 900)
    assert deadline is None and reason == "the budget left 25s for the judge, under the 30s floor"
    # the verification's room is negative: said as 0s
    assert s._finishing_deadline("verification", 900)[1] == "the budget left 0s for the verification, under the 30s floor"


def test_an_unbudgeted_campaign_keeps_its_configured_timeouts():
    s = strategy_with(BudgetSnapshot(iteration_index=0, max_iterations=5, elapsed_seconds=1e6, cost_usd=0.0))
    assert s._clamped_timeout(600) == 600 and s._clamped_timeout(None) is None
    assert s._finishing_deadline("judge", 600) == (600, None)
    assert s._finishing_deadline("judge", None) == (None, None)
    none = strategy_with(None)
    assert none._clamped_timeout(600) == 600 and none._finishing_deadline("judge", 600) == (600, None)


def test_the_unjudged_attempt_keeps_its_score_and_is_neither_parent_nor_delivered():
    s = strategy_with(snapshot(elapsed=580))
    s.feedback_generator = SimpleNamespace(configured_timeout_seconds=600)
    node = SearchNode(node_id=0, branch_name="generic_exp_0", score=0.9, evaluation_valid=True)
    s._generate_feedback(node)
    assert node.unjudged == "the budget left no time for the judge (the budget left 10s for the judge, under the 30s floor)"
    assert node.score == 0.9
    assert s.finishing_cut == "node 0 unjudged: the budget left 10s for the judge, under the 30s floor"
    s.node_history = [node]
    s.registered_evaluator_id = ""
    s.parent_policy = "best"
    s.problem_handler = SimpleNamespace(maximize_scoring=True)
    assert s.get_best_experiment() is None
    assert s.get_deliverable_experiment() is None
    assert s._select_parent().branch_name == "main"
    judged = SearchNode(node_id=1, branch_name="generic_exp_1", score=0.5, evaluation_valid=True)
    s.node_history.append(judged)
    assert s.get_best_experiment() is judged and s.get_deliverable_experiment() is judged
    # The record carries the reason.
    assert SearchNode.from_dict(node.to_dict()).unjudged == node.unjudged


def test_the_reconstruction_and_the_implementation_follow_the_predicate(tmp_path):
    s = strategy_with(snapshot(elapsed=580))
    node = SearchNode(node_id=0, branch_name="generic_exp_0")
    s._ensure_technical_difficulties(node)
    assert node.technical_difficulties.startswith("(the reconstruction was not launched: the budget left")
    assert s.finishing_cut.startswith("the reconstruction for node 0 was not launched")
    # The implementation is a search session: under the floor it is not
    # launched either, and the lane records why.
    s.expansion_lane_env = None
    output, telemetry, recovered, suspended, error = s._implement(
        solution="s", problem="p", branch_name="generic_exp_0",
    )
    assert output.startswith("Implementation not launched: the budget left") and telemetry == {}
    assert recovered is None and suspended is None and "under the 30s floor" in error


def test_admission_refusal_prose():
    assert admission_refusal(480, 840, 840) == "the remaining 8.0 min would not cover an attempt (the last took 14.0 min)"
    assert admission_refusal(20, 60, None) == "the remaining 0.3 min would not cover an attempt (the floor is 1.0 min)"
    assert admission_refusal(-5, 60, 30) == "the remaining 0.0 min would not cover an attempt (the floor is 1.0 min)"


def test_admission_covers_the_last_round(tmp_path, monkeypatch):
    """With a last completed round of 14 min and 8 searchable minutes
    left, the next round is refused by name; without history the floor
    alone admits; the checkpoint carries the round for a resume."""
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "load_config", lambda config_path=None: fake_config(budget={"reserve_minutes": 2}))
    orchestrator = _orchestrator(workspace)
    strategy = orchestrator.search_strategy
    # The fake's rounds take no time: the floor alone admits them.
    first = orchestrator.solve(experiment_max_iter=1, time_budget_minutes=10)
    assert first.iterations_run == 1 and first.stopped_reason == "max_iterations"
    # A round that took 14 minutes of wall time cannot fit in 8.
    strategy.last_round_seconds = 14 * 60.0
    orchestrator._save_run_checkpoint(status="running")
    second = orchestrator.solve(experiment_max_iter=3, time_budget_minutes=10)
    assert second.stopped_reason == "budget_exhausted" and second.stop_detail == ATTEMPT_WOULD_NOT_FIT
    assert second.iterations_run == 0
    recent = status_file(orchestrator)["recent"]
    assert any("would not cover an attempt (the last took 14.0 min)" in line for line in recent)
    assert RunCheckpointStore(str(workspace)).load().last_stop == ATTEMPT_WOULD_NOT_FIT


def test_an_answered_continuation_is_admitted_whatever_the_remainder(tmp_path, monkeypatch):
    """The release gate's S5 (missing-key, the fourth Codex launch): the
    campaign paused on a request, the person answered within two minutes,
    and the resume was refused — "the remaining 16.3 min would not cover an
    attempt (the last took 18.2 min)", the last being the paused attempt's
    own elapsed time — so the reply never reached the session. A node the
    person has answered continues whatever the remainder; a new round after
    it is admitted on the real remainder as before."""
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator_module, "load_config", lambda config_path=None: fake_config(budget={"reserve_minutes": 2}))
    orchestrator = _orchestrator(workspace)
    strategy = orchestrator.search_strategy
    strategy.last_round_seconds = 14 * 60.0
    orchestrator._save_run_checkpoint(status="running")
    answered = {"pending": True}
    monkeypatch.setattr(strategy, "continues_an_answered_node", lambda: answered["pending"])
    original_run = strategy.run

    def run_then_settle(*args, **kwargs):
        answered["pending"] = False   # the continuation ran; the next admission is an ordinary round's
        return original_run(*args, **kwargs)
    monkeypatch.setattr(strategy, "run", run_then_settle)
    result = orchestrator.solve(experiment_max_iter=3, time_budget_minutes=10)
    assert result.iterations_run == 1, "the continuation ran although 8 searchable minutes could not cover a 14-minute round"
    assert result.stopped_reason == "budget_exhausted" and result.stop_detail == ATTEMPT_WOULD_NOT_FIT, "the round after it met the rule"
    recent = status_file(orchestrator)["recent"]
    assert any("would not cover an attempt (the last took 14.0 min)" in line for line in recent)


def test_a_finishing_cut_stops_the_campaign_by_name(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    orchestrator = _orchestrator(workspace)
    strategy = orchestrator.search_strategy
    original = strategy.run

    def run_and_cut(context, budget_progress=0.0):
        node = original(context, budget_progress)
        strategy.finishing_cut = "node 0 unjudged: the budget left 12s for the judge, under the 60s floor"
        return node

    strategy.run = run_and_cut
    result = orchestrator.solve(experiment_max_iter=3, time_budget_minutes=60)
    assert result.iterations_run == 1
    assert result.stopped_reason == "budget_exhausted" and result.stop_detail == FINISHING_CUT
    assert any("a finishing session was cut: node 0 unjudged" in line for line in status_file(orchestrator)["recent"])
    assert RunCheckpointStore(str(workspace)).load().last_stop == FINISHING_CUT


def test_the_reserve_is_derived_from_the_obligations_or_bounded_when_set(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    orchestrator = _orchestrator(workspace)
    strategy = orchestrator.search_strategy
    # Nothing set: a cold judge at the config's reserve (minutes, never the
    # floor: a judge kept at the floor got 59 s in the release gate's hour
    # and the attempt went unjudged) plus the delivery's slice; the
    # snapshot carries the same figure for the finishing stages' obligations.
    orchestrator.solve(experiment_max_iter=1, time_budget_minutes=60)
    spec = orchestrator.budget_spec
    assert spec.judge_reserve_seconds == 300 > spec.min_agent_timeout_seconds
    assert strategy.budget_snapshot.finalization_reserve_seconds == spec.judge_reserve_seconds + DELIVERY_SECONDS
    assert strategy.budget_snapshot.judge_reserve_seconds == spec.judge_reserve_seconds
    assert strategy.budget_snapshot.delivery_seconds == DELIVERY_SECONDS
    # The judge's observed duration replaces the reserve; a verifying
    # harness adds its minutes.
    strategy.node_history[-1].phase_telemetry["feedback"] = {"cost_usd": 0.0, "duration_seconds": 95.0}
    identity = {"name": "h", "version": "0.1", "hash": "0" * 64, "source": "/h"}
    orchestrator.harness = SimpleNamespace(
        name="h", version="0.1", has_content=lambda module: module == "verification",
        settings={"verification_minutes": 4}, ref_dict=lambda: dict(identity),
    )
    orchestrator.solve(experiment_max_iter=1, time_budget_minutes=60)
    assert strategy.budget_snapshot.finalization_reserve_seconds == 240 + 95 + DELIVERY_SECONDS
    assert status_file(orchestrator)["budget"]["reserve_min"] == round((240 + 95 + DELIVERY_SECONDS) / 60, 1)
    # The derived reserve is bounded like a set one: a quarter of a
    # ten-minute budget, not the 365 s the obligations add up to (the
    # proof run's ten-minute build, whose setup session got a negative
    # deadline).
    orchestrator.solve(experiment_max_iter=1, time_budget_minutes=10)
    assert strategy.budget_snapshot.finalization_reserve_seconds == 150.0
    # A set reserve wins, bounded by a quarter of the budget: 100 of 60 min is 15.
    orchestrator.harness = None
    orchestrator.solve(experiment_max_iter=1, time_budget_minutes=60, finalization_reserve_minutes=100)
    assert strategy.budget_snapshot.finalization_reserve_seconds == 15 * 60
    assert status_file(orchestrator)["budget"]["reserve_min"] == 15.0


def test_the_trace_budget_line_names_the_reserve_and_the_winning_budget():
    from kapso.execution.trace import _budget_source

    assert _budget_source(None) is None
    assert _budget_source({"brief": {"budget": {"minutes": None}}, "minutes": 40.0}) is None
    assert _budget_source({"brief": {"budget": {"minutes": 40.0}}, "minutes": 40.0}) is None
    assert _budget_source({"brief": {"budget": {"minutes": 75.0}}, "minutes": 40.0}) == "the launch's over the brief's 75.0 min"


def test_the_setup_session_is_refused_when_the_budget_leaves_it_no_room(tmp_path, monkeypatch):
    """The launch predicate reaches the harness setup session: a budget
    whose search room is under the floor stops the campaign with the
    reason before any session, instead of launching the setup with a
    negative deadline (the proof run's ten-minute build)."""
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    orchestrator = _orchestrator(workspace)
    identity = {"name": "h", "version": "0.1", "hash": "0" * 64, "source": "/h"}
    orchestrator.harness = SimpleNamespace(
        name="h", version="0.1", has_content=lambda module: module in ("setup", "verification"),
        settings={"verification_minutes": 10, "setup_minutes": 20}, ref_dict=lambda: dict(identity),
        label=lambda: "h 0.1 (000000000000)",
    )
    # One minute: the reserve is bounded to 15 s, the room 45 s, the floor 60 s.
    with pytest.raises(HarnessError, match=r"leaves 0\.[6-8] min for the harness setup session of h 0\.1 .*under the 60 s floor \(0\.2 min of the budget are kept for finishing\); give the campaign more minutes"):
        orchestrator.solve(experiment_max_iter=1, time_budget_minutes=1)
