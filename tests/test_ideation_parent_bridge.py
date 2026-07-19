"""GenericSearch bridge tests for immutable parent and idea provenance."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import git
import pytest

from kapso.execution.budget import BudgetSnapshot
from kapso.execution.fidelity import FidelityDecision
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
    frozen_incumbent = strategy._resolve_ideation_parent(
        ParentPlan(kind=ParentPlanKind.BEST_VALID, experiment_node_id=0)
    )
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
    # Node 1 currently has the higher live score, but the frozen evidence
    # snapshot remains the sole BEST_VALID authority.
    assert (frozen_incumbent.node_id, frozen_incumbent.git_ref) == (0, first_sha)
    assert (specific.node_id, specific.git_ref) == (0, first_sha)
    assert (recovery.node_id, recovery.git_ref) == (1, second_sha)
    assert (
        frozen_incumbent.git_ref
        == frozen_incumbent.materialized_ref
        == frozen_incumbent.diff_base_ref
    )


def test_capacity_view_projects_the_budget_and_fidelity_grant_without_invention():
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.ideation_config = {
        "evidence": {
            "comparable_fidelity": "full",
            "comparable_fraction": 1.0,
        }
    }
    strategy.budget_snapshot = BudgetSnapshot(
        iteration_index=1,
        max_iterations=4,
        elapsed_seconds=10.0,
        cost_usd=0.0,
        time_budget_seconds=300.0,
        finalization_reserve_seconds=60.0,
    )
    strategy.fidelity_decision = FidelityDecision(
        profile="probe",
        build_fidelity="fast",
        eval_fidelity="fast",
        eval_fraction=0.25,
    )

    probe = strategy._ideation_capacity_view()

    assert probe.can_start_complete_action is True
    assert probe.can_run_granted_evaluation is True
    assert probe.can_run_comparable_evaluation is False
    assert probe.preserves_finalization_reserve is True

    strategy.fidelity_decision = FULL_PASSTHROUGH
    full = strategy._ideation_capacity_view()
    assert full.can_run_granted_evaluation is True
    assert full.can_run_comparable_evaluation is True


@pytest.mark.parametrize(
    ("node", "expected"),
    (
        (SearchNode(node_id=0), True),
        (SearchNode(node_id=0, had_error=True), False),
        (SearchNode(node_id=0, recoverable_error=True), False),
        (SearchNode(node_id=0, evaluation_valid=False), False),
    ),
)
def test_evidence_author_only_runs_after_a_valid_successful_evaluation(
    node,
    expected,
):
    assert GenericSearch._should_author_ideation_evidence(node) is expected


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
    strategy.active_batch_id = None
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
        strategy._last_implementation_success = True
        strategy._last_implementation_error = ""
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
    strategy._author_ideation_evidence = lambda node, problem: None

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


def test_recovery_reexecutes_same_node_and_preserves_all_attempt_telemetry(
    monkeypatch,
):
    strategy = GenericSearch.__new__(GenericSearch)
    parent = replace(
        resolved_parent(),
        node_id=0,
        branch_name="generic_exp_0",
        git_ref="failed-sha",
        materialized_ref="failed-sha",
        diff_base_ref="failed-sha",
        feedback_base_ref="failed-sha",
    )
    idea = replace(
        generated_idea(),
        resolved_parent=parent,
        status=IdeaStatus.IMPLEMENTING,
        selected_in_batch_id=BATCH_ID,
        selection_reason="Selected by the audited selector.",
        experiment_node_id=0,
    )
    existing = SearchNode(
        node_id=0,
        idea_id=idea.idea_id,
        selection_batch_id=BATCH_ID,
        solution=idea.proposal,
        branch_name="generic_exp_0",
        parent_branch_name="main",
        execution_revision=0,
        had_error=True,
        recoverable_error=True,
    )
    existing.duration_seconds = 7.0
    existing.implementation_base_ref = "failed-sha"
    existing.diff_base_ref = "original-baseline-sha"
    existing.feedback_base_ref = "original-baseline-sha"
    existing.phase_telemetry = {
        "ideation": {
            "cost_usd": 0.2,
            "duration_seconds": 2.0,
            "coding_agent_call_count": 2.0,
            "unpriced_coding_agent_call_count": 0.0,
        },
        "implementation": {"cost_usd": 0.4, "duration_seconds": 3.0},
        "feedback": {"cost_usd": 0.2, "duration_seconds": 1.0},
    }
    result = SimpleNamespace(
        action=CampaignAction.RECOVER,
        selected_idea=idea,
        resolved_parent=parent,
        batch_id=BATCH_ID,
        telemetry=SimpleNamespace(
            coding_agent_duration_seconds=1.0,
            known_coding_agent_cost_usd=0.3,
            coding_agent_call_count=1,
            unpriced_coding_agent_call_count=0,
            embedding=None,
        ),
    )
    strategy.iteration_count = 1
    strategy.node_history = [existing]
    strategy.workspace_dir = "/workspace"
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy.fidelity_decision = FULL_PASSTHROUGH
    strategy.ideation_campaign_id = "campaign-alpha"
    strategy.active_batch_id = None
    strategy._build_ideation_engine = lambda: FakeEngine(result)
    strategy._ideation_experiment_inputs = lambda: ()
    strategy._ideation_capacity_view = lambda: object()
    strategy._resolve_ideation_parent = lambda plan: parent
    strategy._materialize_ideation_parent = lambda snapshot: None
    strategy._assert_parent_snapshot_current = lambda snapshot: None

    def implement(**kwargs):
        strategy._last_implementation_success = True
        strategy._last_implementation_error = ""
        return "recovered", {"cost_usd": 0.1, "duration_seconds": 1.0}

    strategy._implement = implement
    diff_bases = []
    strategy._get_code_diff = (
        lambda branch, base: diff_bases.append((branch, base)) or "recovery diff"
    )
    strategy._extract_agent_result = lambda output: {}
    strategy._ensure_technical_difficulties = lambda node: None
    strategy.enforce_evaluation_integrity = lambda node: True
    strategy._generate_feedback = lambda node: node
    strategy._record_evaluation_attempt = lambda node: None
    strategy._author_ideation_evidence = lambda node, problem: None
    monotonic_values = iter((100.0, 104.0))
    monkeypatch.setattr(
        "kapso.execution.search_strategies.generic.strategy.time.monotonic",
        lambda: next(monotonic_values),
    )

    returned = strategy.run("complete problem")

    assert returned is existing
    assert returned.execution_revision == 1
    assert returned.had_error is False
    assert returned.recoverable_error is False
    assert returned.parent_branch_name == "main"
    assert returned.implementation_base_ref == "failed-sha"
    assert returned.diff_base_ref == "original-baseline-sha"
    assert returned.feedback_base_ref == "original-baseline-sha"
    assert diff_bases == [("generic_exp_0", "original-baseline-sha")]
    assert returned.phase_telemetry["ideation"] == {
        "cost_usd": pytest.approx(0.5),
        "duration_seconds": pytest.approx(3.0),
        "coding_agent_call_count": pytest.approx(3.0),
        "unpriced_coding_agent_call_count": pytest.approx(0.0),
    }
    assert returned.phase_telemetry["implementation"] == {
        "cost_usd": pytest.approx(0.5),
        "duration_seconds": pytest.approx(4.0),
    }
    assert returned.phase_telemetry["feedback"] == {
        "cost_usd": pytest.approx(0.2),
        "duration_seconds": pytest.approx(1.0),
    }
    assert returned.duration_seconds == pytest.approx(11.0)
    assert returned.cost_usd == pytest.approx(1.2)
    assert strategy.node_history == [existing]
