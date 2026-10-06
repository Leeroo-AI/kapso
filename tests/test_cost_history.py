"""Cost policy survives the history/MCP boundary without losing prompt content."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from kapso.execution.memories.experiment_memory.store import (
    ExperimentHistoryStore,
    format_experiments,
)
from kapso.gated_mcp.gates.experiment_history_gate import ExperimentHistoryGate


def cost_config(**overrides):
    return {
        "enabled": True,
        "estimator": "example.estimate_candidate",
        "cap_per_candidate": 20.0,
        "campaign_budget": 50.0,
        "unit": "credits",
        "weight": 1.0,
        "history_field": "estimated_cost",
        **overrides,
    }


def candidate(node_id, score, estimated, measured=None, refusal=None):
    return SimpleNamespace(
        node_id=node_id,
        solution=f"Candidate {node_id}",
        score=score,
        feedback="Use the measured result.",
        branch_name=f"candidate_{node_id}",
        had_error=False,
        error_message="",
        evaluation_valid=refusal is None,
        estimated_cost={
            "amount": estimated,
            "unit": "credits",
            "basis": "warehouse plan",
            "metadata": {"query": "full query " * 100},
        },
        measured_cost=(
            {"amount": measured, "unit": "credits"}
            if measured is not None
            else None
        ),
        cost_refusal=refusal,
    )


@pytest.mark.parametrize(
    ("maximize_scoring", "scores", "expected"),
    [(True, [10.0, 9.0], [1, 0]), (False, [1.0, 2.0], [1, 0])],
)
def test_reopened_history_keeps_cost_policy_and_problem_direction(
    tmp_path, maximize_scoring, scores, expected
):
    history_path = tmp_path / "history.json"
    store = ExperimentHistoryStore(
        str(history_path),
        cost_config=cost_config(),
        maximize_scoring=maximize_scoring,
    )
    store.add_experiment(candidate(0, scores[0], 1.0, measured=8.0))
    store.add_experiment(candidate(1, scores[1], 2.0))

    reopened = ExperimentHistoryStore(str(history_path))

    assert [record.node_id for record in reopened.get_top_experiments()] == expected
    assert reopened.experiments[0].measured_cost == {
        "amount": 8.0, "unit": "credits"
    }
    assert reopened.experiments[1].estimated_cost["basis"] == "warehouse plan"


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_top_experiments", {"k": 1}),
        ("get_recent_experiments", {"k": 1}),
        ("search_similar_experiments", {"query": "detailed query " * 30, "k": 1}),
    ],
)
def test_mcp_reports_whole_campaign_cost_once_above_subset(
    tmp_path, tool_name, arguments
):
    store = ExperimentHistoryStore(
        str(tmp_path / "history.json"), cost_config=cost_config()
    )
    store.add_experiment(candidate(0, 10.0, 10.0, measured=4.0))
    store.add_experiment(candidate(1, 9.0, 3.0))
    refusal = {
        "estimate": {
            "amount": 40.0,
            "unit": "credits",
            "basis": "warehouse plan",
            "metadata": {},
        },
        "cap": 20.0,
        "reason": "cap_per_candidate",
    }
    store.add_experiment(candidate(2, None, 40.0, refusal=refusal))
    gate = ExperimentHistoryGate()
    gate._store = ExperimentHistoryStore(store.json_path)

    response = asyncio.run(gate.handle_call(tool_name, arguments))
    rendered = response[0].text

    assert rendered.count("Campaign budget used: 7.0 / 50.0 credits") == 1
    assert rendered.index("Campaign budget used") < rendered.index("## Experiment")
    assert "estimated_cost" in rendered
    assert "measured_cost" in rendered
    assert "warehouse plan" in rendered
    assert "full query " * 100 in rendered
    if tool_name != "get_top_experiments":
        assert "**measured_cost:** null" in rendered
        assert "cost_refusal" in rendered
        assert '"cap": 20.0' in rendered
    if tool_name == "search_similar_experiments":
        assert arguments["query"] in rendered


def test_disabled_history_keeps_ranking_and_prompt_bytes(tmp_path):
    store = ExperimentHistoryStore(
        str(tmp_path / "history.json"), maximize_scoring=False
    )
    store.add_experiment(candidate(0, 3.0, 900.0))
    store.add_experiment(candidate(1, 1.0, 0.0))

    assert [record.node_id for record in store.get_top_experiments()] == [0, 1]
    assert format_experiments(store.experiments[:1]) == (
        "\n## Experiment 0 (score=3.0)\n\n**Solution:**\nCandidate 0"
        "\n\n**Feedback:**\nUse the measured result."
    )
    gate = ExperimentHistoryGate()
    gate._store = store
    response = asyncio.run(gate.handle_call("get_top_experiments", {"k": 1}))
    assert response[0].text == (
        "# Top 1 Experiments by Score\n\n"
        + format_experiments(store.experiments[:1])
    )


def test_history_preserves_complete_cost_refusal_and_prompt_content(tmp_path):
    store = ExperimentHistoryStore(
        str(tmp_path / "history.json"),
        cost_config=cost_config(history_field="measured_cost"),
    )
    refused = candidate(0, None, 40.0)
    refused.solution = "complete solution " * 500
    refused.feedback = "complete feedback " * 500
    refused.technical_difficulties = "complete difficulties " * 500
    refused.evaluation_valid = False
    refused.cost_refusal = {
        "estimate": refused.estimated_cost,
        "cap": 20.0,
        "reason": "full refusal basis " * 500,
    }
    store.add_experiment(refused)

    reopened = ExperimentHistoryStore(store.json_path)
    rendered = reopened.format_experiments(reopened.experiments)

    assert reopened.experiments[0].cost_refusal == refused.cost_refusal
    for full_text in (
        refused.solution,
        refused.feedback,
        refused.technical_difficulties,
        refused.cost_refusal["reason"],
    ):
        assert full_text in rendered
    assert rendered.index("**measured_cost:") < rendered.index("**estimated_cost:")
    assert "**measured_cost:** null" in rendered


def test_measured_zero_replaces_estimate_in_history_ranking_and_budget(tmp_path):
    store = ExperimentHistoryStore(
        str(tmp_path / "history.json"), cost_config=cost_config()
    )
    store.add_experiment(candidate(0, 10.0, 10.0, measured=0.0))
    store.add_experiment(candidate(1, 10.0, 1.0))

    assert [record.node_id for record in store.get_top_experiments()] == [0, 1]
    assert "Campaign budget used: 1.0 / 50.0 credits" in store.format_experiments(
        store.get_top_experiments(k=1)
    )


def test_corrupt_cost_context_fails_loudly(tmp_path):
    store = ExperimentHistoryStore(
        str(tmp_path / "history.json"), cost_config=cost_config()
    )
    store.add_experiment(candidate(0, 1.0, 1.0))
    (tmp_path / "history.cost.json").write_text("{malformed")

    with pytest.raises(json.JSONDecodeError):
        ExperimentHistoryStore(store.json_path)


def test_nonboolean_persisted_problem_direction_fails_loudly(tmp_path):
    history_path = tmp_path / "history.json"
    history_path.with_suffix(".cost.json").write_text(json.dumps({
        "cost": cost_config(),
        "maximize_scoring": "false",
    }))

    with pytest.raises(ValueError, match="maximize_scoring must be a boolean"):
        ExperimentHistoryStore(str(history_path))
