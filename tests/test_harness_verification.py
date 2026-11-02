"""The harness's verification of an attempt: between the evaluation and the
judge, in a worktree of the attempt's branch, with the diff and the
evaluation output at named paths; a void verdict clears the score,
excludes the node from parent selection and delivery, counts it as
rejected and reaches the judge, and the judge cannot resurrect it; a
valid verdict passes its summary to the judge and into the solution
record; a malformed verdict raises; no verdict is retried once, then the
attempt is unverified, kept as a parent and never delivered; a harness
without verification content runs no session."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional

import git
import pytest

import kapso.execution.search_strategies.generic.strategy as strategy_module
from kapso.core.config import load_config
from kapso.core.roles import SessionRole
from kapso.execution.budget import BudgetSnapshot
from kapso.execution.coding_agents.base import CodingAgentConfig, CodingResult
from kapso.execution.experiment_workspace.experiment_workspace import ExperimentWorkspace
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from kapso.handles import Solution
from kapso.harness import HarnessError, load_harness
from kapso.harness.verdict import is_deliverable, read_verdict, render_verification
from tests.handles_support import candidate_branch, git as run_git, git_workspace, make_solution

TRADING = Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading"


class FakeVerifier:
    """The verification session, scripted: writes (or does not write) the
    verdict file named in its prompt."""
    calls: List[dict] = []
    script: Callable[[Path, str], None] = staticmethod(lambda verdict_path, prompt: None)
    error: Optional[str] = None   # a session that dies instead of finishing
    killed: bool = False          # ... cut at its deadline, the turn unfinished

    def __init__(self, config):
        self.config = config
        self.workspace = None

    def initialize(self, workspace: str) -> None:
        self.workspace = Path(workspace)

    def generate_code(self, prompt: str, debug_mode: bool = False, timeout_seconds=None) -> CodingResult:
        verdict_path = Path(prompt.split("Write exactly one JSON file at `", 1)[1].split("`", 1)[0])
        FakeVerifier.calls.append({"workspace": self.workspace, "prompt": prompt, "config": self.config, "verdict_path": verdict_path})
        FakeVerifier.script(verdict_path, prompt)
        if FakeVerifier.error:
            metadata = {"deadline_exceeded": True, "completed_before_kill": False} if FakeVerifier.killed else {}
            return CodingResult(success=False, output="", error=FakeVerifier.error, metadata=metadata)
        return CodingResult(success=True, output="done")

    def get_cumulative_cost(self) -> float:
        return 0.1

    def cleanup(self) -> None:
        pass


class FakeJudge:
    def __init__(self):
        self.coding_agent_config = CodingAgentConfig(agent_type="claude_code", model="judge", debug_model="judge", agent_specific={})
        self.agent_type = "claude_code"
        self.configured_timeout_seconds = 60
        self.seen: List[dict] = []

    def attach_harness(self, staged):
        pass

    def generate(self, **kwargs):
        self.seen.append(kwargs)
        return SimpleNamespace(feedback="a solid attempt", evaluation_valid=True, score=0.9, stop=False, duration_seconds=1.0, cost_usd=0.0, cards_load_bearing=[], unjudged="")


class Workspace:
    """The strategy's workspace as verification uses it: the repository and
    a detached worktree of a ref."""

    def __init__(self, path: Path):
        self.workspace_dir = str(path)
        self.repo = git.Repo(path)

    def materialize_ref(self, ref: str):
        return ExperimentWorkspace.materialize_ref(self, ref)


def valid_verdict(verdict_path: Path, prompt: str) -> None:
    verdict_path.write_text(json.dumps({"verdict": "valid", "summary": "no look-ahead; costs charged on turnover", "findings": []}))


def void_verdict(verdict_path: Path, prompt: str) -> None:
    verdict_path.write_text(json.dumps({
        "verdict": "void", "summary": "a centred rolling window reads the future",
        "findings": [{"rule": "no look-ahead", "evidence": "strategy/signal.py:4 center=True"}],
    }))


@pytest.fixture
def harness(tmp_path):
    copy = tmp_path / "h" / "trading"
    shutil.copytree(TRADING, copy)
    return load_harness(str(copy), settings=dict(load_config()["harness"], store=str(tmp_path / "store")), say=lambda line: None)


@pytest.fixture
def strategy(tmp_path, harness, monkeypatch):
    FakeVerifier.calls = []
    FakeVerifier.script = staticmethod(valid_verdict)
    monkeypatch.setattr(strategy_module.CodingAgentFactory, "create", staticmethod(lambda config: FakeVerifier(config)))
    campaign = tmp_path / "campaign"
    git_workspace(campaign)
    (campaign / "kapso_task.md").write_text("# Task statement\n")
    candidate_branch(campaign, "generic_exp_0", "strategy.py", "signal = bars.rolling(20, center=True).mean()\n")
    # The engine commits its own bookkeeping on the branch too (repo memory).
    run_git(campaign, "checkout", "-q", "generic_exp_0")
    (campaign / ".kapso").mkdir(exist_ok=True)
    (campaign / ".kapso" / "repo_memory.json").write_text('{"generated_at": "later"}\n')
    run_git(campaign, "add", ".kapso/repo_memory.json")
    run_git(campaign, "commit", "-q", "-m", "engine bookkeeping")
    run_git(campaign, "checkout", "-q", "main")

    s = GenericSearch.__new__(GenericSearch)
    s.harness = harness
    s.implementation_role = SessionRole(name="implementation", cli="claude_code", model="implementer", effort=None, timeout=None)
    s.workspace_dir = str(campaign)
    s.workspace = Workspace(campaign)
    s.feedback_generator = FakeJudge()
    s.goal = "Raise the Sharpe above 1"
    s.budget_snapshot = None
    s.budget_snapshot_monotonic = None
    s.finishing_cut = None
    s.registered_evaluation_command = ""
    s.registered_evaluator_id = ""
    s.node_history = []
    s.iteration_count = 0
    s.env_strip = []
    s.env_defaults = {}
    s._clamped_timeout = lambda seconds: seconds
    s.problem_handler = SimpleNamespace(maximize_scoring=True)
    return s


def node_for(strategy, branch: str = "generic_exp_0", node_id: int = 0) -> SearchNode:
    return SearchNode(node_id=node_id, branch_name=branch, parent_branch_name="main", solution="# idea\n", workspace_dir=strategy.workspace_dir,
                      evaluation_output="sharpe: 1.4\nmax_drawdown: 0.1\n", score=1.4)


def test_a_valid_verdict_reaches_the_judge_and_the_record(strategy, tmp_path, harness):
    node = node_for(strategy)
    strategy._verify_attempt(node)
    strategy._generate_feedback(node)

    [call] = FakeVerifier.calls
    directory = tmp_path / "campaign" / ".kapso" / "verification" / "generic_exp_0"
    assert call["verdict_path"] == directory / "verdict.json"
    diff = (directory / "diff.patch").read_text()
    assert "center=True" in diff and "repo_memory" not in diff   # the attempt's changes, not the engine's
    assert (directory / "evaluation_output.txt").read_text() == "sharpe: 1.4\nmax_drawdown: 0.1\n"
    assert call["workspace"] != tmp_path / "campaign" and not call["workspace"].exists()   # a worktree, removed afterwards
    assert (harness.path / "instructions" / "verification.md").read_text().strip() in call["prompt"]
    assert "- lookahead-auditor — " in call["prompt"] and "`/lookahead-scan`" in call["prompt"]
    assert call["config"].agent_specific["timeout"] == 10 * 60 and call["config"].model == "implementer" and call["config"].agent_type == "claude_code"
    assert node.verification == {"verdict": "valid", "summary": "no look-ahead; costs charged on turnover", "findings": []}
    assert node.evaluation_valid is True and node.score == 0.9
    [judged] = strategy.feedback_generator.seen
    assert "Verdict: valid" in judged["verification"] and "no look-ahead; costs charged on turnover" in judged["verification"]
    assert node.phase_telemetry["verification"]["cost_usd"] == 0.1
    solution = make_solution(tmp_path / "campaign", verification=node.verification)
    assert "Verified:        valid — no look-ahead; costs charged on turnover" in solution.explain()
    solution.save()
    assert Solution.open(tmp_path / "campaign").verification == node.verification


def test_a_void_verdict_clears_the_score_and_the_judge_cannot_resurrect_it(strategy):
    FakeVerifier.script = staticmethod(void_verdict)
    node = node_for(strategy)
    strategy._verify_attempt(node)
    strategy._generate_feedback(node)   # the judge says valid with 0.9
    assert node.verification["verdict"] == "void"
    assert node.evaluation_valid is False and node.score is None and node.should_stop is False
    [judged] = strategy.feedback_generator.seen
    assert "Verdict: void" in judged["verification"]
    assert "- no look-ahead: strategy/signal.py:4 center=True" in judged["verification"]
    assert "A void verdict means this attempt is invalid" in judged["verification"]
    strategy.node_history = [node]
    assert strategy.get_best_experiment() is None and strategy.get_deliverable_experiment() is None
    assert sum(not n.evaluation_valid for n in strategy.get_experiment_history()) == 1

    # Not a parent either: the committed-work fallback skips a void.
    node.code_diff = "+signal = bars.rolling(20, center=True).mean()"
    strategy.parent_policy = "best"
    assert strategy._select_parent().branch_name == "main"

    # The void holds when the judge fails: nothing to deliver, no score.
    def judge_dies(**kwargs):
        raise RuntimeError("judge killed")
    again = node_for(strategy)
    strategy.feedback_generator.generate = judge_dies
    strategy._verify_attempt(again)
    strategy._generate_feedback(again)
    assert again.evaluation_valid is False and again.score is None
    assert "judge killed" in again.feedback
    assert is_deliverable(again.verification) is False and is_deliverable({"verdict": "unverified"}) is False
    assert is_deliverable({"verdict": "valid", "summary": "ok", "findings": []}) is True and is_deliverable(None) is True


def test_no_verdict_is_retried_once_then_unverified_and_never_delivered(strategy, tmp_path):
    FakeVerifier.script = staticmethod(lambda verdict_path, prompt: None)
    node = node_for(strategy)
    strategy._verify_attempt(node)
    strategy._generate_feedback(node)
    assert len(FakeVerifier.calls) == 2
    assert node.verification["verdict"] == "unverified"
    assert node.evaluation_valid is True and node.score == 0.9   # the score stands
    assert "verification is missing" in strategy.feedback_generator.seen[0]["verification"]

    # Unverified: a parent still, never the deliverable; a verified sibling is.
    candidate_branch(tmp_path / "campaign", "generic_exp_1", "other.py", "x = 1\n")
    FakeVerifier.script = staticmethod(valid_verdict)
    sibling = node_for(strategy, "generic_exp_1", node_id=1)
    strategy._verify_attempt(sibling)
    strategy._generate_feedback(sibling)
    sibling.score = 0.5
    strategy.node_history = [node, sibling]
    assert strategy.get_best_experiment() is node
    assert strategy.get_deliverable_experiment() is sibling
    strategy.node_history = [node]
    assert strategy.get_deliverable_experiment() is None


def test_a_session_that_dies_is_said_so_and_a_budget_clamp_ends_the_retries(strategy):
    """A killed or failed session is not one that chose silence: its error
    reaches the summary the judge and the record see. When the budget set
    the deadline, a second session would get less still and is not run."""
    FakeVerifier.error = "Deadline of 102.2s reached — terminating Claude Code"
    FakeVerifier.script = staticmethod(lambda verdict_path, prompt: None)
    node = node_for(strategy)
    strategy._verify_attempt(node)
    assert len(FakeVerifier.calls) == 2   # the configured time was given, so one retry
    assert node.verification["verdict"] == "unverified"
    assert node.verification["summary"] == "the verification session failed: Deadline of 102.2s reached — terminating Claude Code"

    FakeVerifier.calls = []
    # The budget's end: 200s remain, the judge's reserve (here 60s, a judge
    # already observed at that) and the delivery's slice follow, so the
    # session gets 110s of its 600 and no retry.
    strategy.budget_snapshot = BudgetSnapshot(
        iteration_index=0, max_iterations=3, elapsed_seconds=800, cost_usd=0.0, time_budget_seconds=1000,
        finalization_reserve_seconds=0.0, min_agent_timeout_seconds=60, judge_reserve_seconds=60, delivery_seconds=30,
    )
    FakeVerifier.killed = True
    clamped = node_for(strategy, "generic_exp_0", node_id=1)
    strategy._verify_attempt(clamped)
    assert len(FakeVerifier.calls) == 1
    assert FakeVerifier.calls[0]["config"].agent_specific["timeout"] == 110
    assert clamped.verification["summary"].startswith("the verification session failed: Deadline")
    assert strategy.finishing_cut.startswith("the verification of node 1 was cut")

    FakeVerifier.calls = []
    FakeVerifier.killed = False
    FakeVerifier.error = "the CLI crashed at startup"
    strategy.finishing_cut = None
    crashed = node_for(strategy, "generic_exp_0", node_id=3)
    strategy._verify_attempt(crashed)
    # Under the same clamp, a session that fails at once is retried: the
    # cut is the invocation's own outcome, not the clamp's arithmetic.
    assert len(FakeVerifier.calls) == 2 and strategy.finishing_cut is None
    FakeVerifier.error = None
    strategy.budget_snapshot = None
    strategy.finishing_cut = None

    FakeVerifier.calls = []
    silent = node_for(strategy, "generic_exp_0", node_id=2)
    strategy._verify_attempt(silent)
    assert len(FakeVerifier.calls) == 2 and silent.verification["summary"] == "the verification session wrote no verdict"


def test_a_malformed_verdict_raises(strategy):
    FakeVerifier.script = staticmethod(lambda verdict_path, prompt: verdict_path.write_text(json.dumps({"verdict": "void", "summary": "no evidence", "findings": []})))
    with pytest.raises(HarnessError, match="a void verdict needs at least one finding"):
        strategy._verify_attempt(node_for(strategy))
    FakeVerifier.script = staticmethod(lambda verdict_path, prompt: verdict_path.write_text(json.dumps({"verdict": "maybe", "summary": "?"})))
    with pytest.raises(HarnessError, match='verdict must be "valid" or "void"'):
        strategy._verify_attempt(node_for(strategy))
    FakeVerifier.script = staticmethod(lambda verdict_path, prompt: verdict_path.write_text("{not json"))
    with pytest.raises(json.JSONDecodeError):
        strategy._verify_attempt(node_for(strategy))


def test_a_harness_without_verification_content_runs_no_session(strategy, harness):
    (harness.path / "instructions" / "verification.md").unlink()
    manifest = harness.path / "harness.yaml"
    text = manifest.read_text()
    start, end = text.index("  verification:\n"), text.index("  feedback:\n")
    manifest.write_text(text[:start] + text[end:])
    settings = dict(load_config()["harness"], store=str(Path(strategy.workspace_dir).parent / "store"))
    strategy.harness = load_harness(str(harness.path), settings=settings, say=lambda line: None)
    node = node_for(strategy)
    strategy._verify_attempt(node)
    strategy._generate_feedback(node)
    assert FakeVerifier.calls == [] and node.verification is None
    assert strategy.feedback_generator.seen[0]["verification"] == ""
    assert node.evaluation_valid is True and node.score == 0.9


def test_the_verdict_contract_is_read_back_whole(tmp_path):
    path = tmp_path / "verdict.json"
    assert read_verdict(path)["verdict"] == "unverified"
    path.write_text(json.dumps({"verdict": "valid", "summary": "  fine  ", "findings": [{"rule": "r", "evidence": "e", "extra": 1}]}))
    assert read_verdict(path) == {"verdict": "valid", "summary": "fine", "findings": [{"rule": "r", "evidence": "e"}]}
    assert render_verification(None) == ""
