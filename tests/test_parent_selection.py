"""Hermetic tests for generic-search parent selection and lineage."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from strategy_support import generic_params, packaged_hosts
from kapso.core.roles import resolve_role

import pytest



import kapso.execution.coding_agents.factory as factory_module
import kapso.execution.search_strategies.generic.ideation as ideation_module
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.base import SearchStrategy
from kapso.execution.cost_estimator import DEFAULT_COST_CONFIG
from kapso.execution.search_strategies.generic.strategy import (
    DEFAULT_PARENT_SAMPLING,
    GenericSearch,
    ParentSelection,
    normalize_parent_policy,
    normalize_parent_sampling,
    parent_weights,
)


# A plan long enough for the ideation pool's hygiene: a candidate under
# 80 characters is a skeleton, dropped before selection.
SELECTED_SOLUTION = ("selected solution: " + "a concrete codable step. " * 4).strip()


@pytest.fixture(autouse=True)
def stub_difficulties_fallback(monkeypatch):
    """Hermetic tests must never spawn the fallback difficulties session;
    fake agents emit no <technical_difficulties> tag, which triggers it."""
    monkeypatch.setattr(
        GenericSearch,
        "_ensure_technical_difficulties",
        lambda self, node: None,
    )


def strategy_with_history(policy="best", *, maximize=True):
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.harness = None
    strategy.parent_policy = policy
    strategy.registered_evaluator_id = ""
    strategy.fidelity_decision = None
    strategy.problem_handler = SimpleNamespace(maximize_scoring=maximize)
    strategy.node_expansion_value = 1
    strategy.expansion_lane_env = None
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
    config = SimpleNamespace(params=generic_params(parent_policy="random"), hosts=packaged_hosts())

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
    strategy._attempt_commit = lambda branch_name: "c" * 40
    calls = {}

    def generate(problem, parent_branch):
        calls["ideation"] = parent_branch
        return ["solution"], [], {"cost_usd": 0.0, "duration_seconds": 0.0}

    def implement(**kwargs):
        calls["implementation"] = kwargs["parent_branch_name"]
        return "agent output", {"cost_usd": 0.0, "duration_seconds": 0.0}, None, None, None

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
                output=f"<solution>{SELECTED_SOLUTION}</solution>",
                error=None,
                metadata={"tool_calls": []},
            )

        def get_cumulative_cost(self):
            return 0.0

        def cleanup(self):
            events["cleaned"] = True

    def fake_mcp_config(**kwargs):
        events["mcp"] = kwargs
        return {}, ["mcp__repo_memory__get_repo_memory_summary"]

    monkeypatch.setattr(
        factory_module.CodingAgentFactory, "create", classmethod(lambda cls, config: FakeAgent(config)),
    )
    monkeypatch.setattr(ideation_module, "get_mcp_config", fake_mcp_config)
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
    strategy.harness = None
    strategy.workspace = FakeWorkspace()
    strategy.workspace_dir = str(tmp_path / "root-workspace")
    strategy.experiment_history_path = str(tmp_path / "history.json")
    strategy.ideation_gates = ["repo_memory"]
    strategy.gate_failure_policy = "warn"
    strategy.bank_serving = None
    strategy.kg_index_path = None
    strategy.experience_intro_path = None
    strategy.hosted = None
    strategy.hosts = packaged_hosts()
    strategy.ideation_role = resolve_role(strategy.hosts, generic_params(ideation={"timeout": 10}), "ideation")
    strategy.shared_artifacts_brief = "No shared-cache artifacts registered yet."
    strategy.node_expansion_value = 1
    strategy.expansion_lane_env = None
    strategy.budget_snapshot = None
    strategy.iteration_count = 0
    strategy.env_strip = []
    strategy.env_defaults = {}
    strategy.ideation_ensemble = None
    strategy.ideation_selector = None
    strategy.ideation_lens_planner = None
    strategy.ideation_candidates_per_member = 1
    strategy.ensemble_time_split = None
    strategy.budget_snapshot_monotonic = None
    strategy.shared_cache_dir = None
    strategy.node_history = []
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy.llm = None

    solutions, sections, telemetry = strategy._generate_solution(
        "problem",
        "candidate-7",
    )

    assert solutions == [SELECTED_SOLUTION]
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
    source.harness = None
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
    source.scores_evaluator_id = ""
    source.evaluator_transition = None
    source.last_round_seconds = None
    state = source.dump_state()

    restored = GenericSearch.__new__(GenericSearch)
    restored.harness = None
    restored.parent_policy = "baseline"
    restored.load_state(state)
    assert restored.parent_policy == "baseline"

    restored = GenericSearch.__new__(GenericSearch)
    restored.harness = None
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
    restored.harness = None
    restored.parent_policy = "baseline"
    with pytest.raises(ValueError, match="must be main"):
        restored.load_state(invalid_lineage)


def test_checkpoint_without_policy_uses_historical_best_default():
    source = GenericSearch.__new__(GenericSearch)
    source.harness = None
    source.node_history = []
    source.iteration_count = 0
    source.previous_errors = []
    source.scores_evaluator_id = ""
    source.evaluator_transition = None
    source.last_round_seconds = None
    state = source.dump_state()
    state.pop("parent_policy")

    restored = GenericSearch.__new__(GenericSearch)
    restored.harness = None
    restored.load_state(state)

    assert restored.parent_policy == "best"


def test_best_policy_falls_back_to_latest_committed_work():
    strategy = strategy_with_history("best")
    strategy.node_history = [
        SearchNode(
            node_id=0,
            branch_name="generic_exp_0",
            evaluation_valid=False,
            code_diff="diff --git a/train.py b/train.py\n+import torch",
        ),
        SearchNode(
            node_id=1,
            branch_name="generic_exp_1",
            evaluation_valid=False,
            code_diff="diff --git a/train.py b/train.py\n+import trl",
        ),
    ]
    assert strategy._select_parent() == ParentSelection(
        branch_name="generic_exp_1", node_id=1
    )


def test_committed_fallback_excludes_errors_tampering_and_empty_diffs():
    strategy = strategy_with_history("best")
    strategy.node_history = [
        SearchNode(
            node_id=0,
            branch_name="errored",
            evaluation_valid=False,
            had_error=True,
            code_diff="+x",
        ),
        SearchNode(
            node_id=1,
            branch_name="tampered",
            evaluation_valid=False,
            evaluation_integrity_error="evaluator modified",
            code_diff="+x",
        ),
        SearchNode(
            node_id=2,
            branch_name="empty-diff",
            evaluation_valid=False,
            code_diff="   \n",
        ),
    ]
    assert strategy._select_parent() == ParentSelection(
        branch_name="main", node_id=None
    )


def test_valid_node_still_beats_later_committed_unevaluated_work():
    strategy = strategy_with_history("best")
    strategy.node_history = [
        SearchNode(node_id=0, branch_name="scored", score=0.1),
        SearchNode(
            node_id=1,
            branch_name="wip",
            evaluation_valid=False,
            code_diff="+x",
        ),
    ]
    assert strategy._select_parent() == ParentSelection(
        branch_name="scored", node_id=0
    )


def weighted_strategy(**sampling):
    strategy = strategy_with_history("weighted")
    strategy.parent_sampling = normalize_parent_sampling(sampling)
    return strategy


def test_parent_sampling_defaults_come_from_config_and_bad_values_raise():
    assert normalize_parent_sampling(None) == DEFAULT_PARENT_SAMPLING
    assert normalize_parent_sampling({"seed": 7})["seed"] == 7
    for bad in (
        {"temperature": 0},
        {"temperature": float("inf")},
        {"offspring_penalty": -1},
        {"seed": 1.5},
        {"seed": True},
        {"unknown": 1},
    ):
        with pytest.raises(ValueError, match="parent_sampling"):
            normalize_parent_sampling(bad)
    with pytest.raises(ValueError, match="parent_sampling"):
        normalize_parent_sampling([])


def test_weighted_policy_is_accepted():
    assert normalize_parent_policy("weighted") == "weighted"


def test_expanded_candidates_weigh_less_than_unexpanded_ones():
    assert parent_weights([0, 0], 1.0, 1.0)[0] > parent_weights([0, 0], 1.0, 1.0)[1]
    unexpanded, expanded = parent_weights([0], 1.0, 1.0)[0], parent_weights([3], 1.0, 1.0)[0]
    assert expanded == unexpanded / 4
    assert parent_weights([5], 1.0, 0.0) == parent_weights([0], 1.0, 0.0)


@pytest.mark.parametrize("maximize", [True, False])
def test_near_zero_temperature_without_penalty_matches_best(maximize):
    strategy = weighted_strategy(temperature=1e-9, offspring_penalty=0.0)
    strategy.problem_handler.maximize_scoring = maximize
    for seed in range(20):
        strategy.parent_sampling["seed"] = seed
        assert strategy._select_parent() == ParentSelection(
            branch_name="candidate-1" if maximize else "candidate-0",
            node_id=1 if maximize else 0,
        )


def test_weighted_never_picks_ineligible_candidates():
    strategy = weighted_strategy(temperature=100.0, offspring_penalty=0.0)
    strategy.node_history += [
        SearchNode(node_id=4, branch_name="unscored", score=None),
        SearchNode(
            node_id=5,
            branch_name="tampered",
            score=9.0,
            evaluation_valid=False,
            evaluation_integrity_error="evaluator modified",
        ),
        SearchNode(
            node_id=6,
            branch_name="refused",
            score=None,
            evaluation_valid=False,
            cost_refusal={"reason": "cap"},
        ),
    ]
    chosen = set()
    for seed in range(200):
        strategy.parent_sampling["seed"] = seed
        chosen.add(strategy._select_parent().branch_name)
    assert chosen == {"candidate-0", "candidate-1"}


def test_weighted_penalty_shifts_choice_away_from_expanded_parent():
    strategy = weighted_strategy(temperature=1.0, offspring_penalty=5.0)
    strategy.node_history += [
        SearchNode(
            node_id=4 + i,
            branch_name=f"child-{i}",
            parent_node_id=1,
            parent_branch_name="candidate-1",
            had_error=True,
        )
        for i in range(6)
    ]
    picks = []
    for seed in range(400):
        strategy.parent_sampling["seed"] = seed
        picks.append(strategy._select_parent().node_id)
    assert picks.count(0) > picks.count(1)


def test_weighted_choice_is_deterministic_and_varies_with_node_count():
    first = weighted_strategy(temperature=50.0, offspring_penalty=0.0, seed=3)
    second = weighted_strategy(temperature=50.0, offspring_penalty=0.0, seed=3)
    assert first._select_parent() == second._select_parent()

    seen = set()
    for extra in range(40):
        first.node_history.append(
            SearchNode(node_id=10 + extra, branch_name=f"x{extra}", had_error=True)
        )
        seen.add(first._select_parent().node_id)
    assert seen == {0, 1}


def test_weighted_before_any_scored_candidate_matches_best_fallback():
    strategy = weighted_strategy()
    strategy.node_history = [
        SearchNode(
            node_id=0,
            branch_name="wip",
            evaluation_valid=False,
            code_diff="+x",
        )
    ]
    assert strategy._select_parent() == ParentSelection(
        branch_name="wip", node_id=0
    )
    strategy.node_history = []
    assert strategy._select_parent() == ParentSelection(
        branch_name="main", node_id=None
    )


def test_weighted_ranks_with_cost_aware_selection():
    strategy = weighted_strategy(temperature=1e-9, offspring_penalty=0.0)
    strategy.cost_config = {**DEFAULT_COST_CONFIG, "enabled": True, "weight": 1.0}
    strategy.node_history = [
        SearchNode(node_id=0, branch_name="cheap", score=0.5, estimated_cost={"amount": 0.0, "unit": "credits"}),
        SearchNode(node_id=1, branch_name="costly", score=0.9, estimated_cost={"amount": 10.0, "unit": "credits"}),
    ]
    assert strategy._select_parent().branch_name == "cheap"
    assert strategy.get_best_experiment().branch_name == "cheap"
