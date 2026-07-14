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
import kapso.execution.search_strategies.generic.strategy as strategy_module
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


class FakeEvalPopen:
    """A completed frame-run process for the strategy's Popen pattern."""

    def __init__(self, stdout: str, returncode: int = 0):
        self._stdout = stdout
        self.pid = 99999
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def communicate(self):
        return self._stdout, ""


def fake_eval_subprocess(payload, returncode: int = 0):
    """A strategy_module.subprocess stand-in emitting one manifest line."""
    manifest_line = (
        f"{maintainer_module.MANIFEST_MARKER} {json.dumps(payload)}\n"
    )

    def popen(
        command, cwd, stdout=None, stderr=None, text=None,
        start_new_session=None,
    ):
        return FakeEvalPopen(manifest_line, returncode)

    return SimpleNamespace(PIPE=-1, Popen=popen)


def make_bridge_strategy(tmp_path, *, branches):
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.registered_evaluator_id = "ev-2"
    strategy.registered_subsample_seed = 1337
    strategy.registered_data_manifest = {}

    class FakeWorkspace:
        repo = SimpleNamespace(
            heads=[SimpleNamespace(name=name) for name in branches],
            commit=lambda branch: SimpleNamespace(hexsha=f"sha-{branch}"),
        )

        @contextmanager
        def materialize_ref(self, ref):
            yield str(tmp_path)

    strategy.workspace = FakeWorkspace()
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
        strategy_module, "subprocess", fake_eval_subprocess(payload)
    )
    alive = SearchNode(node_id=1, branch_name="generic_exp_1")
    assert (
        strategy.run_bridge_evaluation(
            alive, fidelity="full", fraction=1.0, deadline_seconds=10
        )
        is True
    )
    assert alive.evaluation_attempts[0].evaluator_id == "ev-2"
    assert alive.evaluation_attempts[0].score == 0.37


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
    # (fidelity is off, so the canonical class is full/1.0); unbudgeted
    # campaign -> the affordability deadline is unbounded.
    assert strategy.bridge_calls == [
        {
            "node_id": 0,
            "fidelity": "full",
            "fraction": 1.0,
            "deadline_seconds": None,
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
