"""Candidate platform cost contracts, without a provider or a Git workspace."""

import json
from types import SimpleNamespace

import pytest

from kapso.execution.cost_estimator import (
    CostEstimate,
    campaign_cost_used,
    effective_cost,
    load_cost_estimator,
    normalize_cost_config,
    normalize_cost_estimate,
    selection_score,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies._template import MyStrategy


def estimate_fixture(context):
    return CostEstimate(3.0, "credits", "query plan", {"iteration": context.iteration})


def enabled_config(**overrides):
    return normalize_cost_config({
        "enabled": True,
        "estimator": f"{__name__}.estimate_fixture",
        "weight": 0.5,
        **overrides,
    })


@pytest.mark.parametrize("maximize, scores", [(True, [10, 10, 12]), (False, [10, 10, 8])])
def test_selection_prefers_lower_cost_and_better_score(maximize, scores):
    nodes = [
        SearchNode(node_id=0, score=scores[0], estimated_cost={"amount": 6, "unit": "credits"}),
        SearchNode(node_id=1, score=scores[1], estimated_cost={"amount": 2, "unit": "credits"}),
        SearchNode(node_id=2, score=scores[2], estimated_cost={"amount": 2, "unit": "credits"}),
    ]
    config = {"cost": enabled_config(), "maximize_scoring": maximize}
    ranked = sorted(nodes, key=lambda node: selection_score(node, config), reverse=True)
    assert [node.node_id for node in ranked] == [2, 1, 0]


def test_measured_replaces_estimated_in_rankings_and_budget():
    config = enabled_config()
    node = SearchNode(
        node_id=0, score=10,
        estimated_cost={"amount": 20, "unit": "credits"},
        measured_cost={"amount": 0, "unit": "credits"},
    )
    pending = SearchNode(node_id=1, estimated_cost={"amount": 4, "unit": "credits"})
    refused = SearchNode(node_id=2, estimated_cost={"amount": 90, "unit": "credits"},
                         cost_refusal={"reason": "cap_per_candidate"})
    assert effective_cost(node, config) == 0
    assert selection_score(node, {"cost": config, "maximize_scoring": True}) == 10
    assert campaign_cost_used([node, pending, refused], config) == 4


def test_suspended_build_has_not_spent_platform_budget():
    pending = SearchNode(node_id=0, score=0.9, suspended=True)
    assert campaign_cost_used([pending], enabled_config()) == 0


@pytest.mark.parametrize("maximize", [True, False])
def test_disabled_cost_keeps_existing_score_order_byte_identical(maximize):
    nodes = [SearchNode(node_id=index, score=score,
                        estimated_cost={"amount": 100 - index, "unit": "other"})
             for index, score in enumerate([0, -2, 3, 3])]
    config = {"cost": normalize_cost_config({"weight": 100}), "maximize_scoring": maximize}
    old = sorted(nodes, key=lambda node: node.score if maximize else -node.score, reverse=True)
    new = sorted(nodes, key=lambda node: selection_score(node, config), reverse=True)
    assert json.dumps([node.to_dict() for node in new]) == json.dumps([node.to_dict() for node in old])


def test_template_preserves_disabled_eligibility_and_excludes_invalid_when_enabled():
    valid = SearchNode(0, score=5, estimated_cost={"amount": 1, "unit": "credits"})
    invalid = SearchNode(1, score=10, evaluation_valid=False,
                         estimated_cost={"amount": 1, "unit": "credits"})
    strategy = SimpleNamespace(
        experiment_history=[valid, invalid],
        problem_handler=SimpleNamespace(maximize_scoring=True),
        cost_config=normalize_cost_config(),
    )
    assert MyStrategy.get_best_experiment(strategy) is invalid
    strategy.cost_config = enabled_config()
    assert MyStrategy.get_best_experiment(strategy) is valid


@pytest.mark.parametrize("maximize", [True, False])
def test_template_disabled_retains_missing_score_errors(maximize):
    strategy = SimpleNamespace(
        experiment_history=[SearchNode(0, score=None), SearchNode(1, score=1)],
        problem_handler=SimpleNamespace(maximize_scoring=maximize),
        cost_config=normalize_cost_config(),
    )
    with pytest.raises(TypeError):
        MyStrategy.get_best_experiment(strategy)
    with pytest.raises(TypeError):
        MyStrategy.get_experiment_history(strategy, best_last=True)


def test_omitted_cost_policy_uses_disabled_default():
    node = SearchNode(node_id=0, score=3)
    assert selection_score(node, {"cost": None, "maximize_scoring": False}) == -3


@pytest.mark.parametrize("overrides, message", [
    ({"enabled": True}, "estimator"),
    ({"enabled": "yes"}, "enabled"),
    ({"cap_per_candidate": -1}, "cap_per_candidate"),
    ({"campaign_budget": float("inf")}, "campaign_budget"),
    ({"weight": True}, "weight"),
    ({"unit": ""}, "unit"),
    ({"history_field": "cost_usd"}, "history_field"),
    ({"estimator": "unqualified"}, "estimator"),
    ({"cap_per_candiate": 2}, "cap_per_candiate"),
])
def test_cost_config_rejects_invalid_policy(overrides, message):
    with pytest.raises(ValueError, match=message):
        normalize_cost_config(overrides)


def test_estimator_loader_invokes_module_qualified_callable():
    estimator = load_cost_estimator(enabled_config())
    result = normalize_cost_estimate(estimator(SimpleNamespace(iteration=4)), "credits")
    assert result == {"amount": 3.0, "unit": "credits", "basis": "query plan", "metadata": {"iteration": 4}}
    with pytest.raises(ValueError, match="callable"):
        load_cost_estimator(enabled_config(estimator="math.pi"))


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf"), True, "3"])
def test_estimator_rejects_invalid_amount(amount):
    with pytest.raises(ValueError, match="amount"):
        normalize_cost_estimate(CostEstimate(amount, "credits", "plan"), "credits")


def test_cost_contract_rejects_mixed_units_and_invalid_metadata():
    with pytest.raises(ValueError, match="unit"):
        normalize_cost_estimate(CostEstimate(3, "hours", "plan"), "credits")
    with pytest.raises(ValueError, match="keys"):
        normalize_cost_estimate(CostEstimate(3, "credits", "plan", {"nested": {2: "bad"}}), "credits")
    with pytest.raises(ValueError):
        normalize_cost_estimate(CostEstimate(3, "credits", "plan", {"bad": float("nan")}), "credits")
    with pytest.raises(ValueError, match="unit"):
        effective_cost(SearchNode(0, measured_cost={"amount": 1, "unit": "hours"}), enabled_config())


def test_enabled_ranking_rejects_missing_cost_without_charging_unbuilt_candidates():
    config = enabled_config()
    with pytest.raises(ValueError, match="missing"):
        selection_score(SearchNode(0, score=2), {"cost": config, "maximize_scoring": True})
    failed_before_admission = SearchNode(1, had_error=True)
    failed_after_admission = SearchNode(2, had_error=True, estimated_cost={"amount": 3, "unit": "credits"})
    suspended = SearchNode(3, suspended=True)
    assert campaign_cost_used([failed_before_admission, failed_after_admission, suspended], config) == 3


def test_node_costs_and_refusal_round_trip_and_validate():
    estimate = {"amount": 12.0, "unit": "credits", "basis": "plan", "metadata": {"rows": 200}}
    refusal = {"estimate": estimate, "cap": 10.0, "reason": "cap_per_candidate", "campaign_cost_used": 5}
    node = SearchNode(0, estimated_cost=estimate, measured_cost={"amount": 4, "unit": "credits"}, cost_refusal=refusal)
    restored = SearchNode.from_dict(json.loads(json.dumps(node.to_dict())))
    assert restored.estimated_cost == estimate
    assert restored.measured_cost == {"amount": 4, "unit": "credits"}
    assert restored.cost_refusal == refusal
    with pytest.raises(ValueError, match="amount"):
        SearchNode.from_dict({"node_id": 0, "measured_cost": {"amount": -1, "unit": "credits"}})
