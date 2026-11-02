"""`Solution.open` reads the record a build wrote, from any process: the
three shapes a build returns (delivered, waiting on the person, nothing
succeeded), what `explain()` and `diff()` say for each, the registered
evaluator, and what `open()` says before the build has returned; and, with
a `branch`, one attempt of a recorded campaign as the trace shows it."""

import json
import subprocess
import sys

import pytest

from kapso.execution.evaluation_maintainer.registry import (
    EvaluationRegistry,
    EvaluatorVersion,
    TimingModel,
)
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import FeedbackResult
from kapso.execution.trace import read_trace
from kapso.handles import EvaluatorRef, Solution, SolutionError
from tests.handles_support import (
    candidate_branch,
    facade,
    fake_orchestrator,
    git,
    git_workspace,
    make_solution,
    recorded_campaign,
    solve_result,
)


def test_a_delivered_build_reads_back_with_its_candidate_score_and_diff(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    winner = candidate_branch(workspace, "generic_exp_2", "model.py", "improved = True\n")
    feedback = FeedbackResult(stop=True, evaluation_valid=True, feedback="AUC 0.9 beats the target", score=0.9)
    fake_orchestrator(
        monkeypatch, workspace,
        solve_result(final_feedback=feedback, stopped_reason="goal_achieved", total_cost=1.25),
        best_branch="generic_exp_2", score=0.9,
    )
    solution = facade().build("Raise AUC above 0.85", output=str(workspace))

    reopened = Solution.open(workspace)
    assert reopened == solution
    explained = reopened.explain()
    for line in (
        "Solution for:    Raise AUC above 0.85",
        f"Candidate:       generic_exp_2 @ {winner}",
        "Score:           0.9",
        "Goal achieved:   True",
        "Stopped:         goal_achieved",
        "Experiments:     1 this run, 1 in all",
        "Cost:            $1.250",
        "Feedback:\nAUC 0.9 beats the target",
    ):
        assert line in explained
    assert "+improved = True" in reopened.diff()


def test_a_build_waiting_on_the_person_keeps_its_requests(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    requests = [{"id": 1, "key": "env:OPENAI_API_KEY", "fix": "add it to .env"}]
    fake_orchestrator(monkeypatch, workspace, solve_result(stopped_reason="waiting_for_user", requests=requests))
    facade().build("g", output=str(workspace))

    reopened = Solution.open(workspace)
    assert reopened.stopped_reason == "waiting_for_user" and reopened.requests == requests
    assert reopened.branch is None and reopened.goal_achieved is False
    assert "Waiting on you:\n  #1  env:OPENAI_API_KEY" in reopened.explain()


def test_a_build_with_no_successful_candidate_has_no_branch_and_an_empty_diff(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    fake_orchestrator(monkeypatch, workspace, solve_result(stopped_reason="max_iterations", stop_detail=None))
    solution = facade().build("g", output=str(workspace))

    assert (solution.branch, solution.commit, solution.score) == (None, None, None)
    assert solution.diff() == ""
    assert "Candidate:       none succeeded" in solution.explain()
    assert Solution.open(workspace) == solution


def test_the_registered_evaluator_is_recorded(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    EvaluationRegistry(str(workspace)).register(EvaluatorVersion(
        evaluator_id="eval-1", version=1, provenance="provided", parent_evaluator=None,
        fidelity_support={}, timing=TimingModel(per_item_seconds=0.1, startup_seconds=1.0, total_items=10),
        created_at_iteration=0, reason="provided by the caller",
    ))
    fake_orchestrator(monkeypatch, workspace, solve_result(), score=0.8, best_branch="main")
    solution = facade().build("g", output=str(workspace))

    assert solution.evaluator == EvaluatorRef(version=1, evaluator_id="eval-1", provenance="provided")
    assert Solution.open(workspace).evaluator == solution.evaluator
    assert "Score:           0.8 (evaluator v1, provided)" in solution.explain()


def test_open_before_the_build_returned_names_watch(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".kapso").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="kapso watch"):
        Solution.open(campaign)
    with pytest.raises(FileNotFoundError, match="not a campaign directory"):
        Solution.open(tmp_path / "elsewhere")


def test_open_rejects_a_record_of_another_schema(tmp_path):
    record = tmp_path / "campaign" / ".kapso" / "solution.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"schema_version": 1}')
    with pytest.raises(ValueError, match="schema"):
        Solution.open(tmp_path / "campaign")


def test_deploy_refuses_nothing_to_deploy_and_a_workspace_moved_off_the_candidate(tmp_path, monkeypatch):
    """Deployment copies the directory as it is, so the facade checks that
    the directory is still at the delivered commit before anything runs."""
    workspace = tmp_path / "ws"
    base = git_workspace(workspace)
    winner = candidate_branch(workspace, "generic_exp_1", "model.py", "improved = True\n")
    fake_orchestrator(monkeypatch, workspace, solve_result(), best_branch="generic_exp_1", score=0.9)
    solution = facade().build("g", output=str(workspace))
    assert solution.commit == winner

    git(workspace, "checkout", "-q", "main")
    with pytest.raises(ValueError, match=f"checked out at {base[:12]}, not at the delivered candidate"):
        facade().deploy(solution)

    with pytest.raises(ValueError, match="nothing to deploy"):
        facade().deploy(make_solution(workspace, commit=None, branch=None))


def test_a_campaign_that_delivered_nothing_opens_its_attempt_as_the_trace_shows_it(tmp_path):
    """The real shape from the issue: a scored, valid attempt the budget
    cut before the judge read it, and nothing delivered."""
    campaign, commit = recorded_campaign(tmp_path, "churn-unjudged")
    delivered = Solution.open(campaign)
    assert delivered.commit is None and delivered.diff() == "" and delivered.chosen_attempt is None

    attempt = Solution.open(campaign, branch="generic_exp_0")
    [shown] = read_trace(campaign).attempts
    assert (attempt.branch, attempt.commit) == ("generic_exp_0", commit) == (shown["branch"], shown["commit"])
    assert attempt.score == shown["evaluation"]["score"] == 0.8454
    assert attempt.verification == shown["verification"]
    assert attempt.cost_usd == shown["cost_usd"]
    assert attempt.goal_achieved is False
    assert "+improved = True" in attempt.diff()
    # The campaign's facts stay the campaign's.
    assert attempt.brief == delivered.brief and attempt.stopped_reason == delivered.stopped_reason
    assert attempt.experiments_total == delivered.experiments_total
    assert attempt.chosen_attempt == {
        "score": 0.8454, "evaluation_valid": True, "evaluation_provenance": shown["evaluation"]["provenance"],
        "integrity_error": None, "unjudged": shown["unjudged"], "verification": None, "delivered_branch": None,
    }
    explained = attempt.explain()
    assert f"Candidate:       generic_exp_0 @ {commit}, chosen by you (the engine delivered nothing)" in explained
    assert "unjudged: the judge's session ended at the budget's deadline" in explained


def test_an_attempt_the_harness_voided_opens_with_its_verdict_and_the_judges_words(tmp_path):
    campaign, commit = recorded_campaign(tmp_path, "trading-void")
    attempt = Solution.open(campaign, branch="generic_exp_0")
    [shown] = read_trace(campaign).attempts
    assert attempt.commit == commit and attempt.score is None
    assert attempt.verification["verdict"] == "void" and attempt.verification == shown["verification"]
    assert attempt.feedback == (shown["judge"]["feedback"] or None)
    assert attempt.chosen_attempt["evaluation_valid"] is False
    assert "Verified:        void" in attempt.explain()


def test_the_attempt_carries_only_its_own_failed_sessions_not_the_campaigns(tmp_path):
    campaign, _commit = recorded_campaign(tmp_path, "trading")
    attempt = Solution.open(campaign, branch="generic_exp_0")
    assert {failure["label"] for failure in attempt.session_failures} == {"generic_exp_0", "verification/generic_exp_0"}
    assert {failure["label"] for failure in Solution.open(campaign).session_failures} > {"generic_exp_1"}


def test_an_attempt_with_no_recorded_cost_says_so_rather_than_crashing_explain(tmp_path):
    campaign, _commit = recorded_campaign(tmp_path, "churn-unjudged")
    state = campaign / ".kapso" / "run_state.json"
    checkpoint = json.loads(state.read_text(encoding="utf-8"))
    checkpoint["strategy_state"]["node_history"][0]["cost_usd"] = None
    state.write_text(json.dumps(checkpoint), encoding="utf-8")
    attempt = Solution.open(campaign, branch="generic_exp_0")
    assert attempt.cost_usd is None and "Cost:            not recorded" in attempt.explain()


def test_an_unknown_branch_is_refused_in_one_line_naming_the_attempts(tmp_path):
    campaign, _commit = recorded_campaign(tmp_path, "trading")
    with pytest.raises(SolutionError) as refused:
        Solution.open(campaign, branch="generic_exp_9")
    assert str(refused.value) == f"no attempt on branch generic_exp_9 in {campaign}; the attempts are: generic_exp_0, generic_exp_1"


def test_an_attempt_with_no_commit_is_refused_in_one_line_with_the_reason_from_its_sessions(tmp_path):
    campaign, _commit = recorded_campaign(tmp_path, "trading", branch="generic_exp_0", plant_commit=False)
    with pytest.raises(SolutionError, match="the attempt on generic_exp_1 has no commit") as refused:
        Solution.open(campaign, branch="generic_exp_1")
    assert "\n" not in str(refused.value)
    assert "generic_exp_1 ended timeout" in str(refused.value)
    with pytest.raises(SolutionError, match="the attempt on generic_exp_0 has no commit: rejected: the verification found ") as refused:
        Solution.open(campaign, branch="generic_exp_0")
    assert "\n" not in str(refused.value)


def test_open_with_a_branch_says_what_open_says_before_the_build_has_returned(tmp_path):
    campaign = tmp_path / "campaign"
    (campaign / ".kapso").mkdir(parents=True)
    with pytest.raises(SolutionError, match="kapso watch"):
        Solution.open(campaign, branch="generic_exp_0")


def test_a_chosen_attempt_is_never_written_over_the_engines_record(tmp_path):
    """`.kapso/solution.json` is the engine's word: what `kapso report`, the
    trace and a later `Solution.open` read as the deliverable."""
    campaign, _commit = recorded_campaign(tmp_path, "churn-unjudged")
    before = (campaign / ".kapso" / "solution.json").read_text(encoding="utf-8")
    with pytest.raises(SolutionError, match="not written over the engine's record"):
        Solution.open(campaign, branch="generic_exp_0").save()
    assert (campaign / ".kapso" / "solution.json").read_text(encoding="utf-8") == before


def test_the_handles_module_stays_light_until_a_branch_is_asked_for():
    """`from kapso import Solution` costs no engine import: the trace loads
    only when an attempt is opened."""
    code = "import sys, kapso.handles; print('kapso.execution.trace' in sys.modules, 'kapso.core' in sys.modules)"
    loaded = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.split()
    assert loaded == ["False", "False"]
