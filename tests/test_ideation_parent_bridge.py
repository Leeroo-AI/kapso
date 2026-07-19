"""GenericSearch bridge tests for immutable parent and idea provenance."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import git

from kapso.execution.fidelity import FULL_PASSTHROUGH
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation import (
    CampaignAction,
    IdeaStatus,
    ParentPlan,
    ParentPlanKind,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from test_ideation_domain import BATCH_ID, generated_idea, resolved_parent


def commit(repo, root: Path, branch: str, content: str):
    repo.git.checkout(branch)
    (root / "solution.txt").write_text(content, encoding="utf-8")
    repo.git.add("solution.txt")
    repo.index.commit(f"commit {branch} {content}")
    return repo.head.commit.hexsha


def strategy_with_repo(tmp_path):
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test")
        config.set_value("user", "email", "test@example.com")
    (tmp_path / "solution.txt").write_text("baseline", encoding="utf-8")
    repo.git.add("solution.txt")
    repo.index.commit("baseline")
    repo.git.branch("-M", "main")
    repo.git.checkout("-b", "generic_exp_0")
    first_sha = commit(repo, tmp_path, "generic_exp_0", "first")
    repo.git.checkout("main")
    repo.git.checkout("-b", "generic_exp_1")
    second_sha = commit(repo, tmp_path, "generic_exp_1", "second")
    repo.git.checkout("main")
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace = SimpleNamespace(repo=repo)
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy.registered_evaluator_id = ""
    strategy.node_history = [
        SearchNode(node_id=0, branch_name="generic_exp_0", score=0.4),
        SearchNode(node_id=1, branch_name="generic_exp_1", score=0.8),
    ]
    return strategy, first_sha, second_sha


def test_parent_plans_freeze_baseline_best_specific_and_recovery_refs(tmp_path):
    strategy, first_sha, second_sha = strategy_with_repo(tmp_path)

    baseline = strategy._resolve_ideation_parent(
        ParentPlan(kind=ParentPlanKind.BASELINE)
    )
    best = strategy._resolve_ideation_parent(ParentPlan(kind=ParentPlanKind.BEST_VALID))
    specific = strategy._resolve_ideation_parent(
        ParentPlan(
            kind=ParentPlanKind.SPECIFIC_EXPERIMENT,
            experiment_node_id=0,
        )
    )
    recovery = strategy._resolve_ideation_parent(
        ParentPlan(
            kind=ParentPlanKind.RECOVER_BRANCH,
            experiment_node_id=1,
        )
    )

    assert baseline.node_id is None
    assert baseline.branch_name == "main"
    assert baseline.git_ref == strategy.workspace.repo.commit("main").hexsha
    assert (best.node_id, best.git_ref) == (1, second_sha)
    assert (specific.node_id, specific.git_ref) == (0, first_sha)
    assert (recovery.node_id, recovery.git_ref) == (1, second_sha)
    assert best.git_ref == best.materialized_ref == best.diff_base_ref


class FakeArchive:
    def __init__(self):
        self.revision = 7
        self.links = []

    def link_experiment(self, idea_id, node_id, batch_id, *, expected_revision):
        self.links.append((idea_id, node_id, batch_id, expected_revision))
        self.revision += 1


class FakeEngine:
    def __init__(self, result):
        self.result = result
        self.arguments = None

    def run(self, **kwargs):
        self.arguments = kwargs
        return self.result


def test_selected_idea_is_linked_before_implementation_and_uses_frozen_bases():
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.iteration_count = 0
    strategy.node_history = []
    strategy.workspace_dir = "/workspace"
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy.fidelity_decision = FULL_PASSTHROUGH
    strategy.ideation_campaign_id = "campaign-alpha"
    parent = replace(
        resolved_parent(),
        git_ref="frozen-sha",
        materialized_ref="frozen-sha",
        diff_base_ref="frozen-sha",
        feedback_base_ref="frozen-sha",
    )
    idea = replace(
        generated_idea(),
        resolved_parent=parent,
        status=IdeaStatus.SELECTED,
        selected_in_batch_id=BATCH_ID,
        selection_reason="Selected by the audited selector.",
    )
    telemetry = SimpleNamespace(
        coding_agent_duration_seconds=2.0,
        known_coding_agent_cost_usd=0.25,
        coding_agent_call_count=3,
        unpriced_coding_agent_call_count=2,
        embedding=None,
    )
    result = SimpleNamespace(
        action=CampaignAction.IDEATE,
        selected_idea=idea,
        resolved_parent=parent,
        batch_id=BATCH_ID,
        telemetry=telemetry,
    )
    engine = FakeEngine(result)
    archive = FakeArchive()
    events = []
    strategy._build_ideation_engine = lambda: engine
    strategy._ideation_experiment_inputs = lambda: ()
    strategy._ideation_capacity_view = lambda: object()
    strategy._resolve_ideation_parent = lambda plan: parent
    strategy._materialize_ideation_parent = lambda snapshot: None
    strategy._ensure_idea_archive = lambda: archive
    strategy._assert_parent_snapshot_current = lambda snapshot: events.append(
        ("parent_checked", snapshot.git_ref)
    )

    def implement(**kwargs):
        events.append(("implemented", kwargs["parent_branch_name"]))
        return "agent output", {"cost_usd": 0.5, "duration_seconds": 3.0}

    strategy._implement = implement
    strategy._get_code_diff = (
        lambda branch, base: events.append(("diff", base)) or "diff"
    )
    strategy._extract_agent_result = lambda output: {}
    strategy._ensure_technical_difficulties = lambda node: None
    strategy.enforce_evaluation_integrity = lambda node: True
    strategy._generate_feedback = lambda node: node
    strategy._record_evaluation_attempt = lambda node: None

    node = strategy.run("complete problem")

    assert archive.links == [(idea.idea_id, 0, BATCH_ID, 7)]
    assert events == [
        ("parent_checked", "frozen-sha"),
        ("implemented", parent.branch_name),
        ("diff", "frozen-sha"),
    ]
    assert node.idea_id == idea.idea_id
    assert node.selection_batch_id == BATCH_ID
    assert node.parent_node_id is None
    assert node.parent_branch_name == parent.branch_name
    assert node.implementation_base_ref == "frozen-sha"
    assert node.diff_base_ref == "frozen-sha"
    assert node.feedback_base_ref == "frozen-sha"
    assert node.phase_telemetry["ideation"]["cost_usd"] == 0.25
    assert strategy.node_history == [node]
