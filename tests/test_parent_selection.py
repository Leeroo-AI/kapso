"""Hermetic tests for generic-search parent selection and lineage."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.execution.coding_agents.adapters.claude_code_agent as claude_module
import kapso.gated_mcp as gated_mcp_module
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.base import SearchStrategy
from kapso.execution.search_strategies.generic.strategy import (
    GenericSearch,
    ParentSelection,
    normalize_parent_policy,
)


def strategy_with_history(policy="best", *, maximize=True):
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.parent_policy = policy
    strategy.registered_evaluator_id = ""
    strategy.problem_handler = SimpleNamespace(maximize_scoring=maximize)
    strategy.node_history = [
        SearchNode(node_id=0, branch_name="candidate-0", score=0.4),
        SearchNode(node_id=1, branch_name="candidate-1", score=0.8),
        SearchNode(
            node_id=2,
            branch_name="invalid-candidate",
            score=1.0,
            evaluation_valid=False,
        ),
        SearchNode(
            node_id=3,
            branch_name="failed-candidate",
            score=2.0,
            had_error=True,
        ),
    ]
    return strategy


@pytest.mark.parametrize("policy", [None, "", "BEST", "random", 1, True])
def test_invalid_parent_policy_is_rejected(policy):
    with pytest.raises(ValueError, match="parent_policy"):
        normalize_parent_policy(policy)


def test_invalid_policy_fails_before_workspace_initialization(monkeypatch):
    super_called = []

    def fake_super(*args, **kwargs):
        super_called.append(True)

    monkeypatch.setattr(SearchStrategy, "__init__", fake_super)
    config = SimpleNamespace(params={"parent_policy": "random"})

    with pytest.raises(ValueError, match="parent_policy"):
        GenericSearch(config)

    assert super_called == []


def test_best_policy_selects_one_valid_branch_and_node_pair():
    strategy = strategy_with_history()

    assert strategy._select_parent() == ParentSelection(
        branch_name="candidate-1",
        node_id=1,
    )

    strategy.problem_handler.maximize_scoring = False
    assert strategy._select_parent() == ParentSelection(
        branch_name="candidate-0",
        node_id=0,
    )


def test_baseline_policy_always_selects_main_without_a_parent_node():
    strategy = strategy_with_history("baseline")

    assert strategy._select_parent() == ParentSelection(
        branch_name="main",
        node_id=None,
    )


@pytest.mark.parametrize(
    ("policy", "expected_branch", "expected_node_id"),
    [
        ("best", "candidate-1", 1),
        ("baseline", "main", None),
    ],
)
def test_iteration_uses_the_same_selected_parent_everywhere(
    policy,
    expected_branch,
    expected_node_id,
):
    strategy = strategy_with_history(policy)
    strategy.iteration_count = 0
    strategy.workspace_dir = "/workspace"
    calls = {}

    def generate(problem, parent_branch):
        calls["ideation"] = parent_branch
        return "solution", [], {"cost_usd": 0.0, "duration_seconds": 0.0}

    def implement(**kwargs):
        calls["implementation"] = kwargs["parent_branch_name"]
        return "agent output", {"cost_usd": 0.0, "duration_seconds": 0.0}

    def code_diff(branch_name, parent_branch):
        calls["diff"] = parent_branch
        return "diff"

    strategy._generate_solution = generate
    strategy._implement = implement
    strategy._get_code_diff = code_diff
    strategy._extract_agent_result = lambda output: {}
    strategy.enforce_evaluation_integrity = lambda node: True
    strategy._generate_feedback = lambda node: node

    node = strategy.run("problem")

    assert node.parent_node_id == expected_node_id
    assert node.parent_branch_name == expected_branch
    assert calls == {
        "ideation": expected_branch,
        "implementation": expected_branch,
        "diff": expected_branch,
    }


def test_ideation_reads_from_a_detached_view_of_the_selected_ref(
    monkeypatch,
    tmp_path,
):
    selected_dir = str(tmp_path / "selected-parent")
    events = {}

    class FakeWorkspace:
        repo = object()

        @contextmanager
        def materialize_ref(self, ref):
            events["ref"] = ref
            Path(selected_dir).mkdir()
            yield selected_dir

    class FakeAgent:
        def __init__(self, config):
            events["config"] = config

        def initialize(self, workspace):
            events["agent_workspace"] = workspace

        def generate_code(self, prompt):
            return SimpleNamespace(
                success=True,
                output="<solution>selected solution</solution>",
                error=None,
            )

        def get_cumulative_cost(self):
            return 0.0

        def cleanup(self):
            events["cleaned"] = True

    def fake_mcp_config(**kwargs):
        events["mcp"] = kwargs
        return {}, ["mcp__repo_memory__get_repo_memory_summary"]

    monkeypatch.setattr(
        claude_module,
        "ClaudeCodeCodingAgent",
        FakeAgent,
    )
    monkeypatch.setattr(gated_mcp_module, "get_mcp_config", fake_mcp_config)
    monkeypatch.setattr(
        RepoMemoryManager,
        "load_from_git_branch",
        classmethod(lambda cls, repo, branch: {}),
    )
    monkeypatch.setattr(
        RepoMemoryManager,
        "render_summary_and_toc",
        classmethod(lambda cls, doc, max_chars=2500: "memory"),
    )

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace = FakeWorkspace()
    strategy.workspace_dir = str(tmp_path / "root-workspace")
    strategy.experiment_history_path = str(tmp_path / "history.json")
    strategy.ideation_gates = ["repo_memory"]
    strategy.gate_failure_policy = "warn"
    strategy.idea_generation_model = "model"
    strategy._claude_auth_settings = {"auth_mode": "oauth"}
    strategy.aws_region = "us-east-1"
    strategy.ideation_timeout = 10
    strategy.budget_snapshot = None
    strategy.iteration_count = 0

    solution, sections, telemetry = strategy._generate_solution(
        "problem",
        "candidate-7",
    )

    assert solution == "selected solution"
    assert sections == []
    assert telemetry["cost_usd"] == 0.0
    assert telemetry["duration_seconds"] >= 0
    assert events["ref"] == "candidate-7"
    assert events["agent_workspace"] == selected_dir
    assert events["mcp"]["repo_root"] == selected_dir
    assert events["mcp"]["experiment_history_path"] == str(
        (tmp_path / "history.json").resolve()
    )
    assert events["cleaned"] is True


def test_checkpoint_preserves_policy_and_validates_lineage():
    source = GenericSearch.__new__(GenericSearch)
    source.parent_policy = "baseline"
    source.node_history = [
        SearchNode(
            node_id=0,
            branch_name="candidate-0",
            parent_node_id=None,
            parent_branch_name="main",
        )
    ]
    source.iteration_count = 1
    source.previous_errors = []
    state = source.dump_state()

    restored = GenericSearch.__new__(GenericSearch)
    restored.parent_policy = "baseline"
    restored.load_state(state)
    assert restored.parent_policy == "baseline"

    restored = GenericSearch.__new__(GenericSearch)
    restored.parent_policy = "best"
    with pytest.raises(ValueError, match="does not match"):
        restored.load_state(state)

    invalid_lineage = dict(state)
    invalid_lineage["node_history"] = [
        SearchNode(
            node_id=0,
            branch_name="candidate-0",
            parent_node_id=None,
            parent_branch_name="candidate-9",
        ).to_dict()
    ]
    restored = GenericSearch.__new__(GenericSearch)
    restored.parent_policy = "baseline"
    with pytest.raises(ValueError, match="must be main"):
        restored.load_state(invalid_lineage)


def test_checkpoint_without_policy_uses_historical_best_default():
    source = GenericSearch.__new__(GenericSearch)
    source.node_history = []
    source.iteration_count = 0
    source.previous_errors = []
    state = source.dump_state()
    state.pop("parent_policy")

    restored = GenericSearch.__new__(GenericSearch)
    restored.load_state(state)

    assert restored.parent_policy == "best"
