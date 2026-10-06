"""A platform budget refusal survives the real search loop and checkpoint."""

import json
import threading
from types import SimpleNamespace

import pytest

from kapso.execution.budget import BudgetLedger
from kapso.execution.cost_estimator import CostEstimate, normalize_cost_config
from kapso.execution.fidelity import FidelitySpec
from kapso.execution.inbox import file_requests, inbox_path, load_requests
from kapso.execution.memories.experiment_memory.store import ExperimentHistoryStore
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.run_checkpoint import RunCheckpointStore, config_fingerprint
from kapso.execution.search_strategies.generic.implementation import SuspendedSession

from test_cost_candidate_gate import candidate_strategy


@pytest.mark.parametrize("suspended_lane", [False, True], ids=["single-lane", "suspended-sibling"])
def test_campaign_budget_stop_persists_refusal_after_measured_spend(
    tmp_path, monkeypatch, suspended_lane,
):
    strategy, _, folders = candidate_strategy(tmp_path, count=3 if suspended_lane else 2)
    strategy.workspace.folders = {
        f"generic_exp_{node_id}": folder
        for node_id, folder in enumerate(folders.values())
    }
    strategy.workspace.workspace_dir = str(tmp_path)
    strategy.iteration_count = 0
    strategy.node_expansion_value = 2 if suspended_lane else 1
    strategy.inbox_settings = {"enabled": True, "path": inbox_path(tmp_path)}
    strategy.previous_errors = []
    strategy.scores_evaluator_id = ""
    strategy.evaluator_transition = None
    strategy.problem_handler = SimpleNamespace(
        maximize_scoring=True,
        honor_agent_stop=True,
        get_problem_context=lambda: "Improve the candidate",
        deliverable_ready_reserve_seconds=lambda: None,
    )
    policy = normalize_cost_config({
        "enabled": True,
        "estimator": "test_cost_campaign.estimate_candidate",
        "cap_per_candidate": 10.0,
        "campaign_budget": 4.0,
    })
    strategy.cost_config = policy

    # Agent-session boundaries supply the finalized local evaluator fixture.
    # SearchNode construction, admission, subprocess execution, ranking, and
    # checkpoint/history writes all run through the production lifecycle.
    monkeypatch.setattr(
        strategy, "_generate_solution",
        lambda problem, parent: (
            ["Ask for missing access", "Use a smaller plan"]
            if suspended_lane and strategy.node_history else ["Use a smaller plan"],
            [], {},
        ),
    )

    def implement(**kwargs):
        suspended = None
        if suspended_lane and kwargs["node_id"] == 1:
            (request_id, _), = file_requests(
                inbox_path(tmp_path), node=1, session="suspended-build", entries=[{
                    "key": "service-access",
                    "hit": "The candidate needs access to the service",
                    "tried": "Checked the available service permissions",
                    "fix": "Grant service access",
                    "next_steps": "Finish the candidate and evaluate it",
                }],
            )
            suspended = SuspendedSession([request_id], "suspended-build")
        return (
            "<code_changes_summary>Built candidate</code_changes_summary>"
            "<evaluation_script_path>kapso_evaluation/evaluate.py</evaluation_script_path>"
            "<technical_difficulties>No build difficulties</technical_difficulties>"
            "<score>null</score>",
            {}, None, suspended,
        )

    monkeypatch.setattr(strategy, "_implement", implement)
    # The user prohibits git commands; this is the worktree provider boundary.
    monkeypatch.setattr(strategy, "_get_code_diff", lambda branch, parent: "candidate changes")

    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.search_strategy = strategy
    orchestrator.problem_handler = strategy.problem_handler
    orchestrator.cost_config = policy
    orchestrator.cost_estimator = lambda context: CostEstimate(
        amount=2.0, unit="credits", basis="Candidate plan"
    )
    orchestrator.goal = "Improve the candidate"
    orchestrator.strategy_type = "generic"
    orchestrator.config_fingerprint = config_fingerprint({"cost": policy})
    orchestrator._config_budget = {}
    orchestrator.budget_ledger = BudgetLedger()
    orchestrator.fidelity_spec = FidelitySpec.resolve({"mode": "off"})
    orchestrator.evaluation_maintainer = None
    orchestrator.iteration_evaluator = None
    orchestrator.completed_iterations = 0
    orchestrator.current_feedback = None
    orchestrator.last_feedback_result = None
    orchestrator._checkpoint_lock = threading.Lock()
    orchestrator._heartbeat_stop = threading.Event()
    orchestrator.checkpoint_store = RunCheckpointStore(str(tmp_path))
    orchestrator.experiment_store = ExperimentHistoryStore(
        str(tmp_path / ".kapso" / "experiment_history.json"), cost_config=policy
    )
    strategy.before_candidate_evaluation = orchestrator._admit_candidate

    result = orchestrator.solve(experiment_max_iter=5)

    assert result.stopped_reason == "cost_budget_exhausted"
    assert result.stop_detail == "cost_budget_exhausted"
    assert result.iterations_run == 2
    assert result.cumulative_iterations == 2
    assert result.requests == []
    assert result.best_experiment.node_id == 0
    assert (tmp_path / "0" / "executed").read_text() == "yes"
    assert not (tmp_path / "1" / "executed").exists()
    checkpoint = orchestrator.checkpoint_store.load()
    assert checkpoint.last_stop == "cost_budget_exhausted"
    assert checkpoint.completed_iterations == 2
    assert checkpoint.status == "running"
    refused = checkpoint.strategy_state["node_history"][-1]
    assert refused["score"] is None
    assert refused["evaluation_valid"] is False
    assert refused["cost_refusal"]["campaign_cost_used"] == 3.0
    assert refused["cost_refusal"]["reason"] == "campaign_budget"
    history = json.loads((tmp_path / ".kapso" / "experiment_history.json").read_text())
    assert history[1]["cost_refusal"] == refused["cost_refusal"]
    assert len(history) == 2
    status = json.loads((tmp_path / ".kapso" / "status.json").read_text())
    assert status["state"] == "done"
    assert status["stopped_reason"] == "cost_budget_exhausted"
    assert status["stop_detail"] == "cost_budget_exhausted"
    assert not status.get("requests")
    if suspended_lane:
        suspended = checkpoint.strategy_state["node_history"][1]
        assert suspended["suspended"] is True
        assert suspended["request_ids"] == [1]
        assert suspended["score"] is None
        assert suspended["estimated_cost"] is None
        assert load_requests(inbox_path(tmp_path))[1].open is True
        assert refused["node_id"] == history[1]["node_id"] == 2
        assert not (tmp_path / "2" / "executed").exists()
