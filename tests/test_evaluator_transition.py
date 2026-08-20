"""Hermetic tests for the evaluator-transition state machine (M6c).

Pins: transitions are durable (pending checkpointed before the bridge,
anchored after), idempotent on resume, mechanical in their fallbacks, and
they close M5b's deferred gap — scores never silently span two evaluator
versions.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.execution.evaluation_maintainer.maintainer as maintainer_module
import kapso.execution.orchestrator as orchestrator_module
import kapso.execution.search_strategies.generic.registered_evaluation as registered_evaluation_module
from kapso.execution.run_checkpoint import RunCheckpoint, RunCheckpointStore
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch

from tests.test_evaluation_maintainer_wiring import (
    ScriptedMaintainerAgent,
    patch_maintainer_environment,
    write_entrypoint,
)
from tests.test_run_checkpoint import (
    _init_git_workspace,
    _orchestrator,
    _patch_orchestrator,
)


# =========================================================================
# Strategy-level: state round-trip and the bridge runner
# =========================================================================

def test_transition_state_round_trips_and_validates():
    source = GenericSearch.__new__(GenericSearch)
    source.node_history = []
    source.iteration_count = 0
    source.previous_errors = []
    source.parent_policy = "best"
    source.scores_evaluator_id = "ev-2"
    source.evaluator_transition = {
        "old_evaluator_id": "ev-1",
        "new_evaluator_id": "ev-2",
        "status": "anchored",
    }
    state = source.dump_state()

    restored = GenericSearch.__new__(GenericSearch)
    restored.parent_policy = "best"
    restored.load_state(state)
    assert restored.scores_evaluator_id == "ev-2"
    assert restored.evaluator_transition["status"] == "anchored"

    broken = dict(state)
    broken["evaluator_transition"] = {"status": "half-done"}
    with pytest.raises(ValueError, match="evaluator_transition"):
        fresh = GenericSearch.__new__(GenericSearch)
        fresh.parent_policy = "best"
        fresh.load_state(broken)


def test_multi_lane_checkpoint_restores_with_fewer_iterations_than_nodes():
    """One iteration legitimately spawns several lane nodes, so nodes
    routinely outnumber iterations. The old 'every node consumed an
    iteration' invariant rejected every multi-lane checkpoint as corrupt
    (first live hit: rel-event/user-ignore resume, 2026-08-09 — 4 lane
    nodes, iteration_count 1, RunCheckpointCorruptError)."""
    source = GenericSearch.__new__(GenericSearch)
    source.node_history = [
        SearchNode(node_id=i, branch_name=f"generic_exp_{i}")
        for i in range(4)
    ]
    source.iteration_count = 1
    source.previous_errors = []
    source.parent_policy = "best"
    source.scores_evaluator_id = "ev-1"
    source.evaluator_transition = None
    state = source.dump_state()

    restored = GenericSearch.__new__(GenericSearch)
    restored.parent_policy = "best"
    restored.load_state(state)
    assert restored.iteration_count == 1
    assert len(restored.node_history) == 4


class FakeEvalPopen:
    """A completed frame-run process for the strategy's Popen pattern."""

    def __init__(self, returncode: int = 0):
        self.pid = 99999
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


def fake_eval_subprocess(payload, returncode: int = 0):
    """A registered_evaluation.subprocess stand-in emitting one manifest line.

    Mirrors the live contract: the strategy hands Popen spooled FILES
    (never PIPE — an undrained pipe deadlocked a chatty evaluator live),
    so the fake writes the manifest line into the provided stdout file.
    """
    manifest_line = (
        f"{maintainer_module.MANIFEST_MARKER} {json.dumps(payload)}\n"
    )

    def popen(
        command, cwd, stdout=None, stderr=None, text=None,
        start_new_session=None,
    ):
        stdout.write(manifest_line)
        return FakeEvalPopen(returncode)

    return SimpleNamespace(PIPE=-1, Popen=popen)


def make_bridge_strategy(tmp_path, *, branches):
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.registered_evaluator_id = "ev-2"
    strategy.registered_subsample_seed = 1337
    strategy.registered_data_manifest = {}
    strategy.record_eval_duration = None
    # The registered head the frame run must overlay into worktrees.
    workspace_root = tmp_path / "workspace_root"
    (workspace_root / "kapso_evaluation").mkdir(parents=True)
    (workspace_root / "kapso_evaluation" / "kapso_eval.py").write_text(
        "REGISTERED_HEAD = True\n"
    )
    strategy.workspace_dir = str(workspace_root)
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    class FakeWorkspace:
        repo = SimpleNamespace(
            heads=[SimpleNamespace(name=name) for name in branches],
            commit=lambda branch: SimpleNamespace(hexsha=f"sha-{branch}"),
        )

        @contextmanager
        def materialize_ref(self, ref):
            yield str(worktree)

    strategy.workspace = FakeWorkspace()
    strategy.bridge_worktree = worktree
    return strategy


def test_bridge_skips_missing_artifacts_and_appends_on_success(
    tmp_path, monkeypatch
):
    strategy = make_bridge_strategy(tmp_path, branches=["generic_exp_1"])

    gone = SearchNode(node_id=0, branch_name="generic_exp_0")
    assert (
        strategy.run_bridge_evaluation(
            gone, fidelity="full", fraction=1.0, deadline_seconds=10
        )
        is False
    )
    assert gone.evaluation_attempts == []

    payload = {
        "fidelity": "full",
        "fraction": 1.0,
        "seed": 1337,
        "items": 100,
        "total_items": 100,
        "score": 0.37,
    }
    monkeypatch.setattr(
        registered_evaluation_module, "subprocess", fake_eval_subprocess(payload)
    )
    # The live requester arrived with evaluation_valid=False (the feedback
    # generator had voided its measurement under the defective evaluator);
    # a successful bridge is a fresh trustworthy measurement and restores it.
    alive = SearchNode(
        node_id=1, branch_name="generic_exp_1", evaluation_valid=False
    )
    assert (
        strategy.run_bridge_evaluation(
            alive, fidelity="full", fraction=1.0, deadline_seconds=10
        )
        is True
    )
    assert alive.evaluation_attempts[0].evaluator_id == "ev-2"
    assert alive.evaluation_attempts[0].score == 0.37
    assert alive.evaluation_valid is True
    # The frame run executed the REGISTERED head, not whatever evaluation
    # tree the branch carried (the live bridge labeled v2 ran a v1 tree).
    overlaid = (
        strategy.bridge_worktree / "kapso_evaluation" / "kapso_eval.py"
    )
    assert overlaid.read_text() == "REGISTERED_HEAD = True\n"


# =========================================================================
# Orchestrator-level: the durable state machine
# =========================================================================

def maintainer_mode_config(config_path, mode):
    return {
        "search_strategy": {"type": "generic", "params": {}},
        "evaluation_maintainer": {
            "type": "claude_code",
            "max_change_requests": 2,
        },
    }


def test_accepted_change_request_runs_the_full_transition(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", maintainer_mode_config
    )

    # One scripted agent serves both the setup and the CR call: the setup
    # call writes the entrypoint, the CR call edits it (the tree must
    # actually change for an accepted request to register).
    call_counter = {"count": 0}

    def setup_then_edit(root: Path) -> None:
        call_counter["count"] += 1
        write_entrypoint(root)
        if call_counter["count"] >= 2:
            (root / "kapso_evaluation" / "kapso_eval.py").write_text(
                "ENTRYPOINT = True\nFIXED = True\n"
            )

    patch_maintainer_environment(
        monkeypatch,
        ScriptedMaintainerAgent(
            setup_then_edit,
            output=(
                "<change_verdict>accept</change_verdict>"
                "<reason>confirmed wrapper defect</reason>"
            ),
        ),
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", maintainer_mode_config
    )
    orchestrator = _orchestrator(workspace)
    strategy = orchestrator.search_strategy
    strategy.next_agent_output = (
        "<evaluation_change_request>timeout bug"
        "</evaluation_change_request>"
    )

    orchestrator.solve(experiment_max_iter=1)

    # Fresh campaign anchored on v1 at start, then transitioned to v2.
    assert strategy.scores_evaluator_id == strategy.registered_evaluator_id
    assert strategy.evaluator_transition["status"] == "anchored"
    assert strategy.evaluator_transition["old_evaluator_id"] != (
        strategy.evaluator_transition["new_evaluator_id"]
    )
    # The bridge ran against the node under the new head at full fidelity
    # (fidelity is off, so the canonical class is full/1.0). Even an
    # unbudgeted campaign gets a BOUNDED affordability window — one
    # calibrated full-eval upper — never None (an unbounded window let a
    # deadlocked evaluator hold a campaign 6h, 2026-08-12).
    bridge_upper = orchestrator.evaluation_maintainer.timing(1.0).upper_seconds
    assert bridge_upper > 0
    assert strategy.bridge_calls == [
        {
            "node_id": 0,
            "fidelity": "full",
            "fraction": 1.0,
            "deadline_seconds": bridge_upper,
        }
    ]
    assert len(strategy.refreshed_classes) == 1
    assert (
        strategy.refreshed_classes[0].evaluator_id
        == strategy.registered_evaluator_id
    )

    checkpoint = RunCheckpointStore(str(workspace)).load()
    saved_transition = checkpoint.strategy_state["evaluator_transition"]
    assert saved_transition["status"] == "anchored"


def test_pending_transition_replays_idempotently_on_resume(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", maintainer_mode_config
    )
    _orchestrator(workspace).solve(experiment_max_iter=1)

    # Simulate a crash between registration and anchoring: rewrite the
    # checkpoint with a pending transition and stale score projections.
    store = RunCheckpointStore(str(workspace))
    checkpoint = store.load()
    state = dict(checkpoint.strategy_state)
    state["scores_evaluator_id"] = "stale-head"
    state["evaluator_transition"] = {
        "old_evaluator_id": "stale-head",
        "new_evaluator_id": "whatever-registered",
        "status": "pending",
    }
    store.save(
        RunCheckpoint.create(
            strategy_type=checkpoint.strategy_type,
            goal=checkpoint.goal,
            config_fingerprint=checkpoint.config_fingerprint,
            status="running",
            completed_iterations=checkpoint.completed_iterations,
            cumulative_cost=checkpoint.cumulative_cost,
            current_feedback=checkpoint.current_feedback,
            strategy_state=state,
            elapsed_seconds=checkpoint.elapsed_seconds,
            cost_by_component=checkpoint.cost_by_component,
        )
    )

    resumed = _orchestrator(workspace, resume=True)
    resumed.solve(experiment_max_iter=1)

    strategy = resumed.search_strategy
    assert strategy.evaluator_transition["status"] == "anchored"
    assert strategy.scores_evaluator_id == strategy.registered_evaluator_id
    assert len(strategy.bridge_calls) == 1

    final = RunCheckpointStore(str(workspace)).load()
    assert (
        final.strategy_state["evaluator_transition"]["status"] == "anchored"
    )


def test_failed_bridges_anchor_an_empty_frontier(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", maintainer_mode_config
    )
    _orchestrator(workspace).solve(experiment_max_iter=1)

    store = RunCheckpointStore(str(workspace))
    checkpoint = store.load()
    state = dict(checkpoint.strategy_state)
    state["scores_evaluator_id"] = "stale-head"
    store.save(
        RunCheckpoint.create(
            strategy_type=checkpoint.strategy_type,
            goal=checkpoint.goal,
            config_fingerprint=checkpoint.config_fingerprint,
            status="running",
            completed_iterations=checkpoint.completed_iterations,
            cumulative_cost=checkpoint.cumulative_cost,
            current_feedback=checkpoint.current_feedback,
            strategy_state=state,
            elapsed_seconds=checkpoint.elapsed_seconds,
            cost_by_component=checkpoint.cost_by_component,
        )
    )

    resumed = _orchestrator(workspace, resume=True)
    resumed.search_strategy.bridge_result = False
    resumed.solve(experiment_max_iter=1)

    strategy = resumed.search_strategy
    # Every bridge candidate failed: still anchored, frontier re-projected
    # (legitimately empty), never deadlocked.
    assert strategy.evaluator_transition["status"] == "anchored"
    assert len(strategy.refreshed_classes) == 1
    assert strategy.scores_evaluator_id == strategy.registered_evaluator_id


def test_accepted_request_bridges_the_requester_first(tmp_path, monkeypatch):
    """The CR filer's old score is unsound by the maintainer's own verdict
    (often None because of the very defect confirmed), so it must be
    bridged first — never ranked by the ruler that just got retired.
    """
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    call_counter = {"count": 0}

    def setup_then_edit(root: Path) -> None:
        call_counter["count"] += 1
        write_entrypoint(root)
        if call_counter["count"] >= 2:
            (root / "kapso_evaluation" / "kapso_eval.py").write_text(
                "ENTRYPOINT = True\nFIXED = True\n"
            )

    patch_maintainer_environment(
        monkeypatch,
        ScriptedMaintainerAgent(
            setup_then_edit,
            output=(
                "<change_verdict>accept</change_verdict>"
                "<reason>grader rejects every mixed submission</reason>"
            ),
        ),
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", maintainer_mode_config
    )
    orchestrator = _orchestrator(workspace)
    strategy = orchestrator.search_strategy
    # Iteration 1: healthy node, high score, no complaint. Iteration 2:
    # the requester — zeroed out by the defective evaluator.
    strategy.agent_output_queue = [
        "",
        "<evaluation_change_request>grader crashes on mixed labels"
        "</evaluation_change_request>",
    ]
    strategy.score_queue = [0.9, None]

    orchestrator.solve(experiment_max_iter=2)

    assert strategy.evaluator_transition["status"] == "anchored"
    # Old order would bridge node 0 (score 0.9) first; the requester wins.
    assert strategy.bridge_calls[0]["node_id"] == 1
    assert len(strategy.bridge_calls) == 1  # its bridge succeeded: done


def test_pending_priority_replays_first_on_resume(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", maintainer_mode_config
    )
    first = _orchestrator(workspace)
    # Node 0 scoreless, node 1 scored: without priority, replay would
    # bridge node 1 first.
    first.search_strategy.score_queue = [None, 0.5]
    first.solve(experiment_max_iter=2)

    store = RunCheckpointStore(str(workspace))
    checkpoint = store.load()
    state = dict(checkpoint.strategy_state)
    state["scores_evaluator_id"] = "stale-head"
    state["evaluator_transition"] = {
        "old_evaluator_id": "stale-head",
        "new_evaluator_id": "whatever-registered",
        "status": "pending",
        "priority_node_id": 0,
    }
    store.save(
        RunCheckpoint.create(
            strategy_type=checkpoint.strategy_type,
            goal=checkpoint.goal,
            config_fingerprint=checkpoint.config_fingerprint,
            status="running",
            completed_iterations=checkpoint.completed_iterations,
            cumulative_cost=checkpoint.cumulative_cost,
            current_feedback=checkpoint.current_feedback,
            strategy_state=state,
            elapsed_seconds=checkpoint.elapsed_seconds,
            cost_by_component=checkpoint.cost_by_component,
        )
    )

    resumed = _orchestrator(workspace, resume=True)
    resumed.solve(experiment_max_iter=1)

    strategy = resumed.search_strategy
    assert strategy.evaluator_transition["status"] == "anchored"
    assert strategy.bridge_calls[0]["node_id"] == 0


def test_unsound_measurement_bridges_but_tampering_never_does(
    tmp_path, monkeypatch
):
    """The live CR campaign's requester was evaluation_valid=False because
    the EVALUATION was defective — the old filter excluded it and the
    frontier restarted from baseline for no reason. Unsound measurements
    bridge (the artifact is fine); a non-empty integrity error (tampering)
    stays exclusionary.
    """
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    call_counter = {"count": 0}

    def setup_then_edit(root: Path) -> None:
        call_counter["count"] += 1
        write_entrypoint(root)
        if call_counter["count"] >= 2:
            (root / "kapso_evaluation" / "kapso_eval.py").write_text(
                "ENTRYPOINT = True\nFIXED = True\n"
            )

    patch_maintainer_environment(
        monkeypatch,
        ScriptedMaintainerAgent(
            setup_then_edit,
            output=(
                "<change_verdict>accept</change_verdict>"
                "<reason>defective guard confirmed</reason>"
            ),
        ),
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", maintainer_mode_config
    )
    orchestrator = _orchestrator(workspace)
    strategy = orchestrator.search_strategy
    # Iteration 1: a tampering node (integrity error). Iteration 2: the
    # requester — measurement unsound (valid False, clean integrity)
    # because the defective evaluator crashed on it.
    strategy.agent_output_queue = [
        "",
        "<evaluation_change_request>guard rejects every honest model"
        "</evaluation_change_request>",
    ]
    strategy.score_queue = [0.9, None]
    strategy.valid_queue = [False, False]
    strategy.integrity_queue = ["evaluation tree tampered", ""]

    orchestrator.solve(experiment_max_iter=2)

    assert strategy.evaluator_transition["status"] == "anchored"
    assert strategy.evaluator_transition["priority_node_id"] == 1
    # The tampering node (score 0.9) never bridges; the unsound-measurement
    # requester does, and first.
    assert [call["node_id"] for call in strategy.bridge_calls] == [1]


# =========================================================================
# Late-transition freeze: past the freeze fraction of a time budget, change
# requests are deferred — a ruler change near the deadline orphans final
# selection (2026-08-16: a transition at 92% budget left one eligible final
# from a 12h ladder).
# =========================================================================


def test_change_requests_freeze_late_in_a_time_budget(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint)
    )
    monkeypatch.setattr(
        orchestrator_module, "load_mode_config", maintainer_mode_config
    )
    orchestrator = _orchestrator(workspace)
    assert orchestrator._transition_freeze_fraction == 0.85

    orchestrator.budget_spec = SimpleNamespace(time_budget_seconds=3600.0)
    filings = []
    orchestrator.evaluation_maintainer.handle_change_request = (
        lambda request: filings.append(request)
        or SimpleNamespace(accepted=False, reason="filed", new_version=None)
    )
    candidate = SimpleNamespace(
        node_id=7,
        agent_output=(
            "<evaluation_change_request>late ruler change"
            "</evaluation_change_request>"
        ),
        evaluation_output="",
    )

    # Past the freeze fraction: the request is deferred, never filed.
    monkeypatch.setattr(
        type(orchestrator), "get_elapsed_seconds", lambda self: 3240.0
    )
    orchestrator._route_change_requests([candidate])
    assert filings == []
    assert orchestrator._change_requests_filed == 0

    # Before the freeze fraction the same request files normally.
    monkeypatch.setattr(
        type(orchestrator), "get_elapsed_seconds", lambda self: 1000.0
    )
    orchestrator._route_change_requests([candidate])
    assert len(filings) == 1
    assert orchestrator._change_requests_filed == 1

    # Without a time budget the freeze never engages.
    orchestrator.budget_spec = SimpleNamespace(time_budget_seconds=None)
    monkeypatch.setattr(
        type(orchestrator), "get_elapsed_seconds", lambda self: 10**9
    )
    orchestrator._route_change_requests([candidate])
    assert len(filings) == 2
