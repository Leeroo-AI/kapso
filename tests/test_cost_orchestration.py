"""Admission and durable cost policy contracts, without live providers."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from kapso.execution.cost_estimator import CostEstimate, normalize_cost_config
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.run_checkpoint import (
    RunCheckpoint,
    RunCheckpointIncompatibleError,
    RunCheckpointStore,
    config_fingerprint,
)
from kapso.execution.search_strategies.base import SearchNode


def checkpoint(cost_config, **overrides):
    arguments = dict(
        strategy_type="generic",
        goal="Improve throughput",
        config_fingerprint=config_fingerprint({"cost": cost_config}),
        cost_config=cost_config,
        status="running",
        completed_iterations=1,
        cumulative_cost=0.0,
        current_feedback="Try a cheaper candidate",
        strategy_state={"node_history": []},
    )
    arguments.update(overrides)
    return RunCheckpoint.create(**arguments)


@pytest.mark.parametrize("key,value", [("estimator", "other.estimate"), ("cap_per_candidate", 7)])
def test_resume_names_the_changed_cost_setting(tmp_path, key, value):
    original = {"enabled": True, "estimator": "campaign.estimate", "cap_per_candidate": 5}
    changed = {**original, key: value}
    store = RunCheckpointStore(str(tmp_path))
    store.save(checkpoint(original))

    with pytest.raises(RunCheckpointIncompatibleError, match=f"cost.{key}"):
        store.load().validate_resume(
            goal="Improve throughput",
            strategy_type="generic",
            config_fingerprint=config_fingerprint({"cost": changed}),
            cost_config=changed,
        )


def test_platform_budget_stop_survives_checkpoint(tmp_path):
    store = RunCheckpointStore(str(tmp_path))
    store.save(checkpoint({}, last_stop="cost_budget_exhausted"))
    assert store.load().last_stop == "cost_budget_exhausted"


def test_checkpoint_default_policy_matches_resolved_disabled_campaign():
    saved = RunCheckpoint.create(
        strategy_type="generic", goal="Improve throughput",
        config_fingerprint="same-run", status="running",
        completed_iterations=1, cumulative_cost=0.0,
        current_feedback=None, strategy_state={},
    )
    saved.validate_resume(
        goal="Improve throughput", strategy_type="generic",
        config_fingerprint="same-run", cost_config=normalize_cost_config(),
    )


def test_disabled_admission_does_not_require_an_estimator():
    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.cost_config = {"enabled": False}
    node = SearchNode(node_id=0, score=0.5)
    before = node.to_dict()
    assert orchestrator._admit_candidate(node)
    assert node.to_dict() == before


def admission_campaign(tmp_path, history, estimator, **cost_overrides):
    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.cost_config = normalize_cost_config({
        "enabled": True,
        "estimator": "campaign.estimate",
        "cap_per_candidate": 5,
        "campaign_budget": 10,
        **cost_overrides,
    })
    orchestrator.cost_estimator = estimator
    orchestrator._platform_budget_exhausted = False
    orchestrator.completed_iterations = 2
    orchestrator.goal = "Improve throughput"
    orchestrator.search_strategy = SimpleNamespace(
        workspace=SimpleNamespace(materialize_ref=lambda ref: nullcontext(tmp_path)),
        get_experiment_history=lambda: history,
    )
    return orchestrator


def test_cap_refuses_finalized_candidate_and_keeps_full_estimate(tmp_path):
    seen = []

    def estimate(context):
        seen.append(context)
        context.node.solution = "Mutation must not change live candidate"
        context.node.phase_telemetry["implementation"]["cost_usd"] = 999
        context.node.request_ids.append(999)
        return CostEstimate(6, "credits", "warehouse scan", {"rows": 1_000_000})

    node = SearchNode(node_id=2, branch_name="candidate-2", parent_branch_name="candidate-1", solution="Original")
    node.phase_telemetry = {"implementation": {"cost_usd": 1.0}}
    orchestrator = admission_campaign(tmp_path, [], estimate)
    assert not orchestrator._admit_candidate(node)
    assert node.score is None and not node.evaluation_valid
    assert node.cost_refusal["cap"] == 5
    assert node.estimated_cost["metadata"] == {"rows": 1_000_000}
    assert node.solution == "Original"
    assert node.phase_telemetry == {"implementation": {"cost_usd": 1.0}}
    assert node.request_ids == []
    assert "Candidate was not evaluated" in node.feedback
    assert seen[0].iteration == 3 and seen[0].git_ref == "candidate-2"
    assert seen[0].parent_ref == "candidate-1" and seen[0].workspace_dir == tmp_path
    assert not orchestrator._platform_budget_exhausted


def test_budget_uses_measured_cost_and_does_not_charge_refused_candidates(tmp_path):
    spent = SearchNode(node_id=0, estimated_cost={"amount": 9, "unit": "credits"}, measured_cost={"amount": 6, "unit": "credits"})
    refused = SearchNode(node_id=1, estimated_cost={"amount": 100, "unit": "credits"}, cost_refusal={"reason": "cap"})
    history = [spent, refused]
    orchestrator = admission_campaign(tmp_path, history, lambda context: CostEstimate(4, "credits", "plan"))
    admitted = SearchNode(node_id=2, branch_name="candidate-2")
    assert orchestrator._admit_candidate(admitted)
    history.append(admitted)
    blocked = SearchNode(node_id=3, branch_name="candidate-3")
    assert not orchestrator._admit_candidate(blocked)
    assert orchestrator._platform_budget_exhausted
    assert blocked.cost_refusal["reason"] == "campaign_budget"
    assert blocked.cost_refusal["campaign_cost_used"] == 10
    assert blocked.cost_refusal["cap"] == 10


def test_refused_candidate_never_reaches_external_evaluator():
    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.cost_config = {"enabled": True}

    def forbidden_evaluation(context):
        pytest.fail("Refused candidate reached evaluation")

    orchestrator.iteration_evaluator = forbidden_evaluation
    node = SearchNode(node_id=0, cost_refusal={"reason": "cap"})
    orchestrator._evaluate_candidates([node], iteration=1)
    assert node.metrics == {}


def test_invalid_candidate_cannot_bypass_admission_through_external_evaluation():
    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.cost_config = {"enabled": True}
    orchestrator.iteration_evaluator = lambda context: pytest.fail("Unadmitted evaluation")
    node = SearchNode(node_id=0, branch_name="invalid", evaluation_valid=False)
    orchestrator._evaluate_candidates([node], iteration=1)
    assert node.metrics == {}
