"""Contract tests for K-way node expansion in the generic strategy.

What must hold: the config validates strictly (K>1 needs the ensemble+
selector flow), the selector's ranked <solution_N> output parses with loud
degrade-to-fewer-lanes semantics, lane ids/branches are allocated up front,
K=1 keeps today's inline path (no executor), the barrier appends nodes in id
order, and the round representative is the best-scoring node carrying any
lane's stop verdict.
"""

from types import SimpleNamespace

import pytest

from kapso.core.prompt_loader import load_prompt
from kapso.execution.search_strategies.generic.strategy import (
    GenericSearch,
    normalize_node_expansion,
    parse_selected_solutions,
    render_lane_brief,
    validate_node_expansion_config,
)

MEMBERS = [
    {"cli": "codex", "model": "gpt-5.6-sol"},
    {"cli": "claude_code", "model": "claude-fable-5"},
]
SELECTOR = {"cli": "claude_code", "model": "claude-opus-4-8"}


def test_normalize_node_expansion():
    assert normalize_node_expansion({}) == (1, None)
    k, env = normalize_node_expansion(
        {
            "node_expansion_value": 2,
            "expansion_lane_env": [
                {"CUDA_VISIBLE_DEVICES": "0"},
                {"CUDA_VISIBLE_DEVICES": "1"},
            ],
        }
    )
    assert k == 2 and env[1]["CUDA_VISIBLE_DEVICES"] == "1"
    for bad in (0, 9, True, "2"):
        with pytest.raises(ValueError):
            normalize_node_expansion({"node_expansion_value": bad})
    with pytest.raises(ValueError, match="expansion_lane_env"):
        normalize_node_expansion(
            {"node_expansion_value": 2, "expansion_lane_env": [{"X": 1}]}
        )


def test_expansion_requires_ensemble_and_selector():
    validate_node_expansion_config(1, None, None)
    validate_node_expansion_config(2, MEMBERS, SELECTOR)
    with pytest.raises(ValueError, match="requires ideation_ensemble"):
        validate_node_expansion_config(2, None, None)
    with pytest.raises(ValueError, match="requires ideation_ensemble"):
        validate_node_expansion_config(2, MEMBERS, None)


def test_parse_selected_solutions():
    single = "<solution>only one</solution>"
    assert parse_selected_solutions(single, 1) == ["only one"]
    ranked = (
        "<selection_reasoning>...</selection_reasoning>"
        "<solution_1>alpha</solution_1><solution_2>beta</solution_2>"
    )
    assert parse_selected_solutions(ranked, 2) == ["alpha", "beta"]
    # Missing slot 2 -> loud degrade to one lane.
    assert parse_selected_solutions("<solution_1>alpha</solution_1>", 2) == ["alpha"]
    # No numbered tags -> legacy single tag still yields one lane.
    assert parse_selected_solutions(single, 2) == ["only one"]
    # Nothing parseable -> empty (caller ladder takes over).
    assert parse_selected_solutions("prose only", 2) == []


def make_stub(monkeypatch, expansion, scores, stop_flags):
    """GenericSearch.__new__ stub wired for _expand_round."""
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.node_expansion_value = expansion
    strategy.node_history = []
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)

    calls = {"lanes": [], "feedback": [], "recorded": []}

    def fake_lane(problem, solution, node_id, parent, decision,
                  sections, telemetry, started_at, lane_index):
        calls["lanes"].append((solution, node_id, lane_index))
        node = SimpleNamespace(
            node_id=node_id,
            score=None,
            should_stop=False,
            phase_telemetry={"implementation": {"cost_usd": 1.0}},
            evaluation_integrity_error=None,
        )
        return node

    def fake_integrity(node):
        return True

    def fake_feedback(node):
        idx = node.node_id - calls["lanes"][0][1]
        node.score = scores[idx]
        node.should_stop = stop_flags[idx]
        calls["feedback"].append(node.node_id)
        return node

    monkeypatch.setattr(strategy, "_run_expansion_lane", fake_lane)
    monkeypatch.setattr(strategy, "enforce_evaluation_integrity", fake_integrity)
    monkeypatch.setattr(strategy, "_generate_feedback", fake_feedback)
    monkeypatch.setattr(
        strategy, "_record_evaluation_attempt",
        lambda node: calls["recorded"].append(node.node_id),
    )
    return strategy, calls


def run_round(strategy, solutions):
    parent = SimpleNamespace(node_id=None, branch_name="main")
    return strategy._expand_round(
        problem="p", solutions=solutions, parent=parent, decision=None,
        ideation_sections=[], ideation_telemetry={},
        iteration_started_at="t0", iteration_started_monotonic=0.0,
    )


def test_k1_runs_inline_without_executor(monkeypatch):
    strategy, calls = make_stub(monkeypatch, 1, scores=[0.5], stop_flags=[False])
    import kapso.execution.search_strategies.generic.strategy as mod

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("executor must not be used for K=1")

    monkeypatch.setattr(mod, "ThreadPoolExecutor", Boom)
    node = run_round(strategy, ["only"])
    assert calls["lanes"] == [("only", 0, 0)]
    assert node.score == 0.5 and len(strategy.node_history) == 1


def test_k2_allocates_ids_appends_in_order_and_picks_winner(monkeypatch):
    strategy, calls = make_stub(
        monkeypatch, 2, scores=[0.41, 0.77], stop_flags=[False, False]
    )
    strategy.node_history = [SimpleNamespace(node_id=0)]  # one prior node
    node = run_round(strategy, ["exploit", "explore"])
    # Upfront allocation: ids 1 and 2, lane indices 0 and 1.
    assert calls["lanes"] == [("exploit", 1, 0), ("explore", 2, 1)]
    # Appended in id order after the barrier.
    appended = [n.node_id for n in strategy.node_history[1:]]
    assert appended == [1, 2]
    # Representative = best score; feedback ran for both, in order.
    assert node.node_id == 2 and node.score == 0.77
    assert calls["feedback"] == [1, 2] and calls["recorded"] == [1, 2]


def test_any_lane_stop_propagates_to_representative(monkeypatch):
    strategy, _ = make_stub(
        monkeypatch, 2, scores=[0.9, 0.2], stop_flags=[False, True]
    )
    node = run_round(strategy, ["a", "b"])
    # Winner is lane 0 by score, but lane 1's certified stop carries.
    assert node.score == 0.9 and node.should_stop is True


def test_scoreless_nodes_rank_last(monkeypatch):
    strategy, _ = make_stub(
        monkeypatch, 2, scores=[None, 0.1], stop_flags=[False, False]
    )
    node = run_round(strategy, ["a", "b"])
    assert node.score == 0.1


def test_expansion_addendum_template_contract():
    addendum = load_prompt(
        "execution/search_strategies/generic/prompts/"
        "ideation_selector_expansion_addendum.md"
    )
    assert "{{expansion_count}}" in addendum
    assert "COMPLEMENTARY" in addendum
    assert "<solution_1>" in addendum


def test_lane_brief_empty_env_renders_nothing():
    assert render_lane_brief(0, 2, None) == ""
    assert render_lane_brief(0, 2, {}) == ""


def test_lane_brief_multi_lane_announces_exclusive_assignment():
    brief = render_lane_brief(1, 2, {"CUDA_VISIBLE_DEVICES": "1"})
    # The fence must be visible: identity, the exact pin, the probe trap,
    # and shared-directory hygiene (first K=2 flight: a lane that probed
    # with nvidia-smi sharded onto its sibling's GPU and OOM'd it).
    assert "lane 1 of 2" in brief
    assert "`CUDA_VISIBLE_DEVICES=1`" in brief
    assert "nvidia-smi" in brief
    assert "DIFFERENT values" in brief
    assert "namespace" in brief


def test_lane_brief_single_lane_states_pins_without_sibling_talk():
    brief = render_lane_brief(0, 1, {"CUDA_VISIBLE_DEVICES": "0"})
    assert "`CUDA_VISIBLE_DEVICES=0`" in brief
    assert "sibling" not in brief


def test_implementation_prompt_injects_lane_brief(monkeypatch):
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.shared_artifacts_brief = "No shared-cache artifacts registered yet."
    monkeypatch.setattr(
        GenericSearch, "_render_budget_status", lambda self: "budget"
    )
    monkeypatch.setattr(
        GenericSearch, "_evaluation_instructions", lambda self: "eval"
    )
    kwargs = dict(
        solution="s",
        problem="p",
        branch_name="generic_exp_1",
        repo_memory_brief="",
        repo_memory_detail_access_instructions="",
        previous_errors="",
    )

    with_brief = strategy._build_implementation_prompt(
        **kwargs,
        lane_brief=render_lane_brief(1, 2, {"CUDA_VISIBLE_DEVICES": "1"}),
    )
    assert "lane 1 of 2" in with_brief
    assert "{{lane_brief}}" not in with_brief

    without_brief = strategy._build_implementation_prompt(**kwargs)
    assert "Parallel Lane Assignment" not in without_brief
    assert "{{lane_brief}}" not in without_brief
