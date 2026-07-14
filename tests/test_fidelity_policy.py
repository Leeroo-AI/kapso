"""Hermetic tests for the FidelityPolicy grant ladder and its wiring (M6b).

Every counter the policy reads is derived from node history, so the same
history always yields the same grants — resume-deterministic by
construction.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.execution.evaluation_maintainer.maintainer as maintainer_module
import kapso.execution.orchestrator as orchestrator_module
import kapso.execution.search_strategies.generic.strategy as strategy_module
from kapso.execution.budget import BudgetSpec
from kapso.execution.fidelity import (
    EvaluationAttempt,
    FULL_PASSTHROUGH,
    FidelityPolicy,
    FidelitySpec,
    PROFILE_FULL,
    PROFILE_PROBE,
    PROFILE_VALIDATE,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch

from tests.test_evaluation_maintainer_wiring import (
    ScriptedMaintainerAgent,
    manifest_stdout,
    patch_maintainer_environment,
    write_entrypoint,
)
from tests.test_evaluator_transition import fake_eval_subprocess
from tests.test_run_checkpoint import (
    _init_git_workspace,
    _orchestrator,
    _patch_orchestrator,
)


def attempt(*, fidelity="fast", fraction=0.15, score=0.5, evaluator="ev-1"):
    return EvaluationAttempt(
        commit_sha="sha",
        evaluator_id=evaluator,
        fidelity=fidelity,
        fraction=fraction,
        seed=1337,
        score=score,
    )


def probe_node(node_id, fast_score):
    node = SearchNode(
        node_id=node_id, build_fidelity="fast", eval_fidelity="fast"
    )
    node.evaluation_attempts = [attempt(score=fast_score)]
    return node


def validated_node(node_id, fast_score, full_score):
    node = probe_node(node_id, fast_score)
    node.evaluation_attempts.append(
        attempt(fidelity="full", fraction=1.0, score=full_score)
    )
    return node


def make_policy(
    full_eval_upper_seconds=100.0,
    fast_eval_upper_seconds=20.0,
    **spec_overrides,
):
    spec_kwargs = dict(
        mode="on",
        build_fast_fraction=0.10,
        eval_fast_fraction=0.15,
        committed_run_fraction=0.45,
        promotion_margin=0.02,
        max_full_runs=2,
        max_full_evals=3,
        calibration_min_pairs=2,
    )
    spec_kwargs.update(spec_overrides)
    spec = FidelitySpec(**spec_kwargs)

    # Provider stubs standing in for the strategy and maintainer: the
    # policy reads the evaluator head and timing uppers live, never
    # frozen copies.
    strategy_stub = SimpleNamespace(
        registered_evaluator_id="ev-1",
        registered_subsample_seed=1337,
    )

    def timing(fraction):
        upper = (
            full_eval_upper_seconds
            if abs(fraction - 1.0) < 1e-9
            else fast_eval_upper_seconds
        )
        return SimpleNamespace(
            expected_seconds=upper, upper_seconds=upper
        )

    maintainer_stub = SimpleNamespace(timing=timing)
    return FidelityPolicy(
        spec=spec, strategy=strategy_stub, maintainer=maintainer_stub
    )


# =========================================================================
# Spec and arithmetic
# =========================================================================

def test_spec_resolution_and_unknown_keys():
    spec = FidelitySpec.resolve(
        {
            "mode": "auto",
            "build": {"fast_fraction": 0.1},
            "eval": {"fast_fraction": 0.2},
            "committed_run_fraction": 0.5,
        }
    )
    assert spec.mode == "auto"
    assert spec.build_fast_fraction == 0.1
    assert spec.eval_fast_fraction == 0.2

    with pytest.raises(ValueError, match="Unknown fidelity config keys"):
        FidelitySpec.resolve({"fast_train_fraction": 0.1})

    assert FidelitySpec.resolve(None).mode == "off"
    # The budget block accepts the fidelity key untouched.
    assert BudgetSpec.resolve(
        config_block={"fidelity": {"mode": "on"}}
    ).is_unbudgeted


def timed_node(
    node_id,
    *,
    duration,
    build="full",
    eval_fidelity="fast",
    implementation_seconds=None,
):
    node = SearchNode(
        node_id=node_id, build_fidelity=build, eval_fidelity=eval_fidelity
    )
    node.duration_seconds = duration
    if implementation_seconds is not None:
        node.phase_telemetry = {
            "implementation": {"duration_seconds": implementation_seconds}
        }
    return node


def test_reserve_arithmetic():
    policy = make_policy()
    budget = 600 * 60.0

    assert policy.reserve_seconds(budget) == pytest.approx(0.45 * budget)
    assert policy.build_cap_seconds(budget) == pytest.approx(
        0.45 * budget - 100.0
    )
    # mode=on: always enabled with a time budget, never without one.
    assert policy.enabled(budget)
    assert not policy.enabled(None)


def test_full_run_price_layers():
    policy = make_policy()

    # L3: no telemetry -> unknown.
    assert policy.full_run_price_seconds([], 60.0) is None

    # L2, zero-build shape: probe iterations already build full; the
    # price is the probe mean plus the full-eval upper.
    probes = [
        timed_node(0, duration=500.0),
        timed_node(1, duration=700.0),
    ]
    assert policy.full_run_price_seconds(
        probes, 600.0
    ) == pytest.approx(600.0 + 100.0)

    # L2 with a build dial: the implementation phase scales by 1/fraction.
    dialed = [
        timed_node(
            0, duration=500.0, build="fast", implementation_seconds=200.0
        )
    ]
    assert policy.full_run_price_seconds(dialed, 500.0) == pytest.approx(
        500.0 + 100.0 + 200.0 * (1 / 0.10 - 1)
    )

    # L1: a measured full-profile run beats every extrapolation.
    full_run = timed_node(2, duration=600.0, eval_fidelity="full")
    assert policy.full_run_price_seconds(
        probes + [full_run], 600.0
    ) == pytest.approx(600.0)


def test_auto_enablement_prices_from_history():
    budget = 8 * 3600.0  # the motivating scenario: 8h budget

    auto = make_policy(mode="auto", min_affordable_full_runs=4)
    # Fresh campaign, unknown price -> conservative: tiers on.
    assert auto.enabled(budget)

    # History shows full runs cost ~10 min -> 48 affordable >= 4 -> off.
    cheap = [timed_node(0, duration=600.0, eval_fidelity="full")]
    assert not auto.enabled(budget, cheap, 500.0)

    # Expensive full runs (3h) -> 2.7 affordable < 4 -> tiers pay.
    costly = [timed_node(0, duration=3 * 3600.0, eval_fidelity="full")]
    assert auto.enabled(budget, costly, 500.0)

    # mode=on ignores pricing entirely.
    assert make_policy().enabled(budget, cheap, 500.0)


# =========================================================================
# The grant ladder
# =========================================================================

def test_default_grant_is_a_probe():
    decision = make_policy().decide(
        nodes=[], remaining_after_reserve=1000.0, probe_estimate_seconds=60.0
    )
    assert decision.profile == PROFILE_PROBE
    assert decision.build_fidelity == "fast"
    assert decision.eval_fidelity == "fast"
    assert decision.eval_fraction == 0.15


def test_unvalidated_leader_clearing_margin_gets_a_validate():
    nodes = [validated_node(0, 0.45, 0.40), probe_node(1, 0.48)]
    decision = make_policy().decide(
        nodes=nodes, remaining_after_reserve=1000.0,
        probe_estimate_seconds=60.0,
    )
    assert decision.profile == PROFILE_VALIDATE
    assert decision.target_node_id == 1
    # Admission was gated by the estimate; the granted deadline is the
    # affordability window (everything outside the reserve).
    assert decision.deadline_seconds == 1000.0

    # Below the margin: no validate, keep probing.
    close = [validated_node(0, 0.47, 0.40), probe_node(1, 0.48)]
    assert (
        make_policy().decide(
            nodes=close,
            remaining_after_reserve=1000.0,
            probe_estimate_seconds=60.0,
        ).profile
        == PROFILE_PROBE
    )


def test_validate_respects_cap_and_affordability():
    nodes = [probe_node(1, 0.48)]

    unaffordable = make_policy().decide(
        nodes=nodes, remaining_after_reserve=50.0, probe_estimate_seconds=60.0
    )
    assert unaffordable.profile == PROFILE_PROBE

    exhausted_cap = make_policy(max_full_evals=0).decide(
        nodes=nodes, remaining_after_reserve=1000.0,
        probe_estimate_seconds=60.0,
    )
    assert exhausted_cap.profile == PROFILE_PROBE


def test_endgame_validate_fires_near_the_gate_without_margin():
    # The leader does not clear the margin, but the searchable window has
    # shrunk to one probe plus one validate: confirm before the reserve.
    nodes = [validated_node(0, 0.47, 0.40), probe_node(1, 0.48)]
    decision = make_policy().decide(
        nodes=nodes, remaining_after_reserve=150.0, probe_estimate_seconds=60.0
    )
    assert decision.profile == PROFILE_VALIDATE
    assert "endgame" in decision.reason


def test_mid_campaign_full_is_calibration_gated():
    validated_leader = validated_node(0, 0.48, 0.42)
    validated_leader.duration_seconds = 400.0
    one_pair = [validated_leader]
    gated = make_policy().decide(
        nodes=one_pair, remaining_after_reserve=10000.0,
        probe_estimate_seconds=60.0,
    )
    assert gated.profile == PROFILE_PROBE  # pairs < calibration_min_pairs

    second_pair = validated_node(1, 0.44, 0.39)
    second_pair.duration_seconds = 400.0
    two_pairs = [validated_leader, second_pair]
    granted = make_policy().decide(
        nodes=two_pairs, remaining_after_reserve=10000.0,
        probe_estimate_seconds=60.0,
    )
    assert granted.profile == PROFILE_FULL
    assert granted.target_node_id == 0
    assert granted.reserve_run is False

    slots_gone = make_policy(max_full_runs=0).decide(
        nodes=two_pairs, remaining_after_reserve=10000.0,
        probe_estimate_seconds=60.0,
    )
    assert slots_gone.profile == PROFILE_PROBE


def test_reserve_run_targets_the_committed_candidate():
    nodes = [validated_node(0, 0.48, 0.42), probe_node(1, 0.50)]
    decision = make_policy().decide(
        nodes=nodes, remaining_after_reserve=10.0,
        probe_estimate_seconds=60.0, reserve_run=True,
    )
    assert decision.profile == PROFILE_FULL
    assert decision.reserve_run is True
    assert decision.target_node_id == 0


# =========================================================================
# Strategy execution of granted profiles
# =========================================================================

def test_validate_grant_short_circuits_and_appends_a_full_attempt(
    monkeypatch, tmp_path
):
    from contextlib import contextmanager

    target = probe_node(0, 0.48)
    target.branch_name = "generic_exp_0"

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.iteration_count = 0
    strategy.node_history = [target]
    strategy.registered_evaluator_id = "ev-1"
    strategy.registered_subsample_seed = 1337
    strategy.registered_data_manifest = {}
    workspace_root = tmp_path / "workspace_root"
    (workspace_root / "kapso_evaluation").mkdir(parents=True)
    (workspace_root / "kapso_evaluation" / "kapso_eval.py").write_text(
        "HEAD = 1\n"
    )
    strategy.workspace_dir = str(workspace_root)
    worktree = tmp_path / "frame_worktree"
    worktree.mkdir()

    class FakeWorkspace:
        repo = SimpleNamespace(
            commit=lambda branch: SimpleNamespace(hexsha="sha-full")
        )

        @contextmanager
        def materialize_ref(self, ref):
            yield str(worktree)

    strategy.workspace = FakeWorkspace()

    payload = {
        "fidelity": "full",
        "fraction": 1.0,
        "seed": 1337,
        "items": 100,
        "total_items": 100,
        "score": 0.41,
    }
    monkeypatch.setattr(
        strategy_module, "subprocess", fake_eval_subprocess(payload)
    )

    from kapso.execution.fidelity import FidelityDecision

    strategy.observe_fidelity(
        FidelityDecision(
            profile=PROFILE_VALIDATE,
            build_fidelity="fast",
            eval_fidelity="full",
            eval_fraction=1.0,
            target_node_id=0,
            deadline_seconds=100.0,
        )
    )
    returned = strategy.run("problem")

    assert returned is target
    full_attempts = [
        a for a in target.evaluation_attempts if a.fidelity == "full"
    ]
    assert len(full_attempts) == 1
    assert full_attempts[0].score == 0.41
    assert full_attempts[0].commit_sha == "sha-full"


# =========================================================================
# Orchestrator wiring
# =========================================================================

def test_fidelity_without_a_maintainer_fails_loud(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    monkeypatch.setattr(
        orchestrator_module,
        "load_mode_config",
        lambda config_path, mode: {
            "search_strategy": {"type": "generic", "params": {}},
            "budget": {"fidelity": {"mode": "on"}},
        },
    )

    with pytest.raises(ValueError, match="evaluation_maintainer"):
        _orchestrator(workspace)


def fidelity_mode_config(config_path, mode):
    return {
        "search_strategy": {"type": "generic", "params": {}},
        "evaluation_maintainer": {"type": "claude_code"},
        "budget": {
            "min_iteration_seconds": 90,
            "fidelity": {
                "mode": "on",
                "build": {"fast_fraction": 0.10},
                "eval": {"fast_fraction": 0.15},
                "committed_run_fraction": 0.45,
            },
        },
    }


def test_probe_grants_and_reserve_slot_reach_the_strategy(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", fidelity_mode_config
    )

    orchestrator = _orchestrator(workspace)
    orchestrator.solve(experiment_max_iter=1, time_budget_minutes=60)

    decisions = orchestrator.search_strategy.fidelity_decisions
    assert decisions[0].profile == PROFILE_PROBE
    snapshot = orchestrator.search_strategy.budget_snapshot
    assert snapshot.finalization_reserve_seconds == pytest.approx(
        0.45 * 3600
    )


def test_reserve_gate_executes_the_escrowed_full_run(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", fidelity_mode_config
    )

    orchestrator = _orchestrator(workspace)
    # 2-minute budget: reserve = 54s, searchable = 66s <= the 90s iteration
    # floor -> the gate trips immediately, and instead of stopping it grants
    # the one escrowed FULL run, then stops with the reserve honored.
    result = orchestrator.solve(experiment_max_iter=5, time_budget_minutes=2)

    decisions = orchestrator.search_strategy.fidelity_decisions
    assert len(decisions) == 1
    assert decisions[0].profile == PROFILE_FULL
    assert decisions[0].reserve_run is True
    assert result.iterations_run == 1
    assert result.stopped_reason == "budget_exhausted"
    assert result.stop_detail == "finalization_reserve"
    # The reserve run SPENDS the escrow — except the measurement's slice:
    # the campaign reserve is released (the live escrowed iteration was
    # once killed at the 60s floor) and the timing model's full-eval upper
    # remains as the residual reserve, so the build cannot starve the
    # frame measurement that follows it.
    full_upper = orchestrator.evaluation_maintainer.timing(1.0).upper_seconds
    reserve_snapshot = orchestrator.search_strategy.budget_snapshot
    assert reserve_snapshot.finalization_reserve_seconds == pytest.approx(
        full_upper
    )
    assert reserve_snapshot.finalization_reserve_seconds < 0.45 * 120
    # The escrowed measurement is kapso-owned: the reserve node's FULL
    # score comes from a frame run, not the agent's self-report (the live
    # reserve artifact did 0.9-class work whose self-report died with a
    # killed feedback call) — with its deadline floored at the estimate.
    reserve_measurement = orchestrator.search_strategy.bridge_calls[-1]
    assert reserve_measurement["fidelity"] == "full"
    assert reserve_measurement["fraction"] == 1.0
    assert reserve_measurement["deadline_seconds"] >= full_upper


def test_fidelity_off_grants_full_passthrough(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    orchestrator = _orchestrator(workspace)
    orchestrator.solve(experiment_max_iter=1)

    assert orchestrator.search_strategy.fidelity_decisions == [
        FULL_PASSTHROUGH
    ]


def test_frame_run_overrun_is_a_failed_attempt_not_a_crash(
    tmp_path, monkeypatch
):
    """The live campaign died on subprocess.TimeoutExpired when a real
    artifact's training outran a baseline-calibrated estimate. Overruns
    kill the process group and report None; they never raise.
    """
    from contextlib import contextmanager

    target = probe_node(0, 0.48)
    target.branch_name = "generic_exp_0"

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.node_history = [target]
    strategy.registered_evaluator_id = "ev-1"
    strategy.registered_subsample_seed = 1337
    strategy.registered_data_manifest = {}
    workspace_root = tmp_path / "workspace_root"
    (workspace_root / "kapso_evaluation").mkdir(parents=True)
    (workspace_root / "kapso_evaluation" / "kapso_eval.py").write_text(
        "HEAD = 1\n"
    )
    strategy.workspace_dir = str(workspace_root)
    worktree = tmp_path / "frame_worktree"
    worktree.mkdir()

    class FakeWorkspace:
        repo = SimpleNamespace(
            commit=lambda branch: SimpleNamespace(hexsha="sha-full")
        )

        @contextmanager
        def materialize_ref(self, ref):
            yield str(worktree)

    strategy.workspace = FakeWorkspace()

    class NeverEndingPopen:
        pid = 424242
        returncode = None

        def poll(self):
            return None

        def wait(self):
            return -15

    kills = []
    monkeypatch.setattr(
        strategy_module,
        "subprocess",
        SimpleNamespace(
            PIPE=-1,
            Popen=lambda *args, **kwargs: NeverEndingPopen(),
        ),
    )
    monkeypatch.setattr(
        strategy_module.os,
        "killpg",
        lambda pgid, sig: kills.append((pgid, sig)),
    )
    monkeypatch.setattr(
        strategy_module, "_FRAME_RUN_KILL_GRACE_SECONDS", 0.05
    )

    score = strategy._execute_registered_evaluation(
        target, fidelity="full", fraction=1.0, deadline_seconds=0.0
    )

    assert score is None
    assert not any(
        a.fidelity == "full" for a in target.evaluation_attempts
    )
    assert kills[0] == (424242, strategy_module.signal.SIGTERM)
    assert kills[-1] == (424242, strategy_module.signal.SIGKILL)



def test_frame_run_refuses_tampered_data(tmp_path, monkeypatch):
    """Frame runs (validate/bridge/reserve) never score a worktree whose
    protected inputs differ from the registered manifest — refusal happens
    before any subprocess spawns.
    """
    from contextlib import contextmanager

    from kapso.execution.evaluation_integrity import build_data_manifest

    honest = tmp_path / "honest"
    (honest / "data").mkdir(parents=True)
    (honest / "data" / "train.csv").write_text("PassengerId,y\n1,False\n")

    target = probe_node(0, 0.48)
    target.branch_name = "generic_exp_0"

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.node_history = [target]
    strategy.registered_evaluator_id = "ev-1"
    strategy.registered_subsample_seed = 1337
    strategy.registered_data_manifest = build_data_manifest(honest, ["data"])
    workspace_root = tmp_path / "workspace_root"
    (workspace_root / "kapso_evaluation").mkdir(parents=True)
    (workspace_root / "kapso_evaluation" / "kapso_eval.py").write_text(
        "HEAD = 1\n"
    )
    strategy.workspace_dir = str(workspace_root)

    rigged = tmp_path / "rigged"
    (rigged / "data").mkdir(parents=True)
    (rigged / "data" / "train.csv").write_text("PassengerId,y\n1,True\n")

    class FakeWorkspace:
        repo = SimpleNamespace(
            commit=lambda branch: SimpleNamespace(hexsha="sha-full")
        )

        @contextmanager
        def materialize_ref(self, ref):
            yield str(rigged)

    strategy.workspace = FakeWorkspace()

    def refuse_spawn(*args, **kwargs):
        raise AssertionError("subprocess must not spawn on tampered data")

    monkeypatch.setattr(
        strategy_module,
        "subprocess",
        SimpleNamespace(PIPE=-1, Popen=refuse_spawn),
    )

    score = strategy._execute_registered_evaluation(
        target, fidelity="full", fraction=1.0, deadline_seconds=None
    )

    assert score is None
    assert not any(
        attempt.fidelity == "full" for attempt in target.evaluation_attempts
    )


def test_effective_reserve_shrinks_once_a_champion_exists():
    """The escrow is insurance: once a full-measured champion exists, the
    residual risk is one re-evaluation (an evaluator transition can retire
    the measurement, never the artifact), so the reserve shrinks to the
    full-eval estimate and the difference returns to the search window.
    """
    policy = make_policy()
    budget = 3600.0
    base = 0.45 * budget

    assert policy.effective_reserve_seconds(budget, []) == pytest.approx(
        base
    )

    champion = SearchNode(node_id=0, build_fidelity="full")
    champion.evaluation_attempts = [
        attempt(fidelity="full", fraction=1.0, score=0.6)
    ]
    assert policy.effective_reserve_seconds(
        budget, [champion]
    ) == pytest.approx(100.0)  # make_policy's full_eval_upper_seconds

    # A flashy unvalidated probe is not a champion; neither is a
    # champion measured under a retired evaluator version.
    flashy = probe_node(1, 0.99)
    assert policy.effective_reserve_seconds(
        budget, [flashy]
    ) == pytest.approx(base)
    stale = SearchNode(node_id=2, build_fidelity="full")
    stale.evaluation_attempts = [
        attempt(
            fidelity="full", fraction=1.0, score=0.9, evaluator="ev-old"
        )
    ]
    assert policy.effective_reserve_seconds(
        budget, [stale]
    ) == pytest.approx(base)


def test_champion_shrink_returns_escrow_to_the_search_window(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", fidelity_mode_config
    )

    orchestrator = _orchestrator(workspace)
    strategy = orchestrator.search_strategy
    # Iteration 1 lands a full-measured champion; iteration 2's snapshot
    # must carry only the contingency residual as its reserve.
    strategy.champion_queue = [True]
    orchestrator.solve(experiment_max_iter=2, time_budget_minutes=60)

    full_upper = orchestrator.evaluation_maintainer.timing(1.0).upper_seconds
    final_snapshot = strategy.budget_snapshot
    assert final_snapshot.finalization_reserve_seconds == pytest.approx(
        min(0.45 * 3600, full_upper)
    )


def test_policy_tracks_the_live_evaluator_head_and_timing():
    """The policy reads providers, never frozen copies: after a change
    request re-registers the evaluator and re-calibrates timing, champion
    recognition and affordability must follow the new head immediately —
    a frozen policy kept judging under the retired ruler.
    """
    policy = make_policy()

    v1_champion = SearchNode(node_id=0, build_fidelity="full")
    v1_champion.evaluation_attempts = [
        attempt(fidelity="full", fraction=1.0, score=0.6)
    ]
    assert policy.full_champion([v1_champion]) is v1_champion

    # The transition: strategy adopts the new head. The v1 champion's
    # measurement is retired with it — and the escrow re-inflates.
    policy._strategy.registered_evaluator_id = "ev-2"
    assert policy.full_champion([v1_champion]) is None
    assert policy.effective_reserve_seconds(
        3600.0, [v1_champion]
    ) == pytest.approx(0.45 * 3600)

    # Re-calibration under v2 flows through the timing provider.
    policy._maintainer.timing = lambda fraction: SimpleNamespace(
        expected_seconds=300.0, upper_seconds=300.0
    )
    assert policy.full_eval_upper_seconds == 300.0


def test_mid_campaign_full_affordability_is_priced_from_history():
    policy = make_policy()
    # Ladder preconditions met: validated committed + 2 calibration pairs.
    nodes = [validated_node(0, 0.48, 0.42), validated_node(1, 0.44, 0.39)]
    for node in nodes:
        node.duration_seconds = 400.0

    # Price = probe 400 + eval upper 100 + build scale 0 (no impl
    # telemetry) = 500; a 10000s window affords it with 2x margin.
    granted = policy.decide(
        nodes=nodes, remaining_after_reserve=10000.0,
        probe_estimate_seconds=400.0,
    )
    assert granted.profile == PROFILE_FULL

    # The same window under the OLD eval-only rule (2x100=200 <= 900)
    # would still grant; the priced rule refuses a window a full run
    # would blow.
    refused = policy.decide(
        nodes=nodes, remaining_after_reserve=900.0,
        probe_estimate_seconds=400.0,
    )
    assert refused.profile == PROFILE_PROBE
