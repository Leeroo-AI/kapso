"""Hermetic tests for per-iteration budget telemetry (design doc, M1).

Pins three contracts: SearchNode telemetry validation never zero-fills
unknowns, GenericSearch.run() attributes phase costs that sum to the node
total, and the FeedbackGenerator measures its own agent's spend as a
cumulative delta.
"""

from types import SimpleNamespace

import pytest

from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch
import kapso.execution.search_strategies.generic.feedback_generator.feedback_generator as feedback_module
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import (
    FeedbackGenerator,
)


# =========================================================================
# SearchNode validation
# =========================================================================

def test_from_dict_tolerates_absent_telemetry_fields():
    node = SearchNode.from_dict({"node_id": 0})

    assert node.duration_seconds is None
    assert node.cost_usd is None
    assert node.started_at == ""
    assert node.phase_telemetry == {}


@pytest.mark.parametrize("field_name", ["duration_seconds", "cost_usd"])
@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf"), True, "3"])
def test_from_dict_rejects_invalid_telemetry_numbers(field_name, bad_value):
    with pytest.raises(ValueError, match=field_name):
        SearchNode.from_dict({"node_id": 0, field_name: bad_value})


@pytest.mark.parametrize(
    "bad_telemetry",
    [
        "not a dict",
        {"ideation": "not a dict"},
        {"ideation": {"cost_usd": -0.5}},
        {"ideation": {"cost_usd": float("nan")}},
        {"ideation": {"cost_usd": True}},
    ],
)
def test_from_dict_rejects_invalid_phase_telemetry(bad_telemetry):
    with pytest.raises(ValueError, match="phase_telemetry"):
        SearchNode.from_dict({"node_id": 0, "phase_telemetry": bad_telemetry})


def test_telemetry_round_trips_through_to_dict():
    original = SearchNode(node_id=0)
    original.duration_seconds = 12.5
    original.cost_usd = 0.875
    original.started_at = "2026-07-13T00:00:00+00:00"
    original.phase_telemetry = {
        "ideation": {"cost_usd": 0.25, "duration_seconds": 4.0},
    }

    restored = SearchNode.from_dict(original.to_dict())

    assert restored.duration_seconds == 12.5
    assert restored.cost_usd == 0.875
    assert restored.started_at == "2026-07-13T00:00:00+00:00"
    assert restored.phase_telemetry == original.phase_telemetry


# =========================================================================
# GenericSearch.run() phase attribution
# =========================================================================

def test_run_sums_attributed_phase_costs_onto_the_node():
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.parent_policy = "baseline"
    strategy.registered_evaluator_id = ""
    strategy.fidelity_decision = None
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy.node_history = []
    strategy.iteration_count = 0
    strategy.workspace_dir = "/workspace"
    strategy._generate_solution = lambda problem, parent_branch: (
        "solution",
        [],
        {"cost_usd": 0.25, "duration_seconds": 1.0},
    )
    strategy._implement = lambda **kwargs: (
        "agent output",
        {"cost_usd": 0.5, "duration_seconds": 2.0},
    )
    strategy._get_code_diff = lambda branch_name, parent_branch: ""
    strategy._extract_agent_result = lambda output: {}
    strategy.enforce_evaluation_integrity = lambda node: True

    def fake_feedback(node):
        node.phase_telemetry["feedback"] = {
            "cost_usd": 0.125,
            "duration_seconds": 0.5,
        }
        return node

    strategy._generate_feedback = fake_feedback

    node = strategy.run("problem")

    assert node.phase_telemetry == {
        "ideation": {"cost_usd": 0.25, "duration_seconds": 1.0},
        "implementation": {"cost_usd": 0.5, "duration_seconds": 2.0},
        "feedback": {"cost_usd": 0.125, "duration_seconds": 0.5},
    }
    assert node.cost_usd == pytest.approx(0.875, abs=0.001)
    assert node.duration_seconds is not None and node.duration_seconds >= 0
    assert node.started_at.endswith("+00:00")
    # Telemetry must survive the durable representation.
    assert SearchNode.from_dict(node.to_dict()).cost_usd == node.cost_usd


# =========================================================================
# FeedbackGenerator spend measurement
# =========================================================================

class CountingAgent:
    """Agent stub whose cumulative cost advances by a known delta per call."""

    def __init__(self, delta):
        self.delta = delta
        self.cumulative = 1.0  # non-zero start: the delta is what must be used

    def initialize(self, workspace):
        pass

    def generate_code(self, prompt):
        self.cumulative += self.delta
        return SimpleNamespace(
            output=(
                "<stop>false</stop>"
                "<evaluation_valid>true</evaluation_valid>"
                "<score>0.9</score>"
                "<feedback>keep going</feedback>"
            )
        )

    def get_cumulative_cost(self):
        return self.cumulative


def test_feedback_generator_measures_its_call_as_a_cumulative_delta(
    monkeypatch, tmp_path
):
    agent = CountingAgent(delta=0.125)
    monkeypatch.setattr(
        feedback_module.CodingAgentFactory,
        "create",
        classmethod(lambda cls, config: agent),
    )
    generator = FeedbackGenerator(
        coding_agent_config=SimpleNamespace(agent_type="stub")
    )
    monkeypatch.setattr(
        generator, "_get_commit_message", lambda workspace_dir, branch: ""
    )

    result = generator.generate(
        goal="goal",
        idea="idea",
        code_changes_summary="summary",
        base_branch="main",
        head_branch="exp",
        evaluation_script_path="kapso_evaluation/evaluate.py",
        evaluation_result="score: 0.9",
        workspace_dir=str(tmp_path),
    )

    assert result.cost_usd == pytest.approx(0.125, abs=1e-9)
    assert result.duration_seconds is not None and result.duration_seconds >= 0
    assert result.to_dict()["cost_usd"] == result.cost_usd
