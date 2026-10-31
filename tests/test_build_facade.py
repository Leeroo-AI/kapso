"""`Kapso.build()`: the brief is the engine's goal, whole; budgets resolve
call over brief over config, and on a resume call over launch record over
brief over config; a resume reads the brief
back from the checkpoint; every path reaches the engine absolute; the
record a build writes has one pinned shape and `Solution.open` reads it
back."""

import json

import pytest

from kapso.execution.inbox import read_launch_record, write_launch_record
from kapso.execution.run_checkpoint import RunCheckpoint, RunCheckpointStore, config_fingerprint
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import FeedbackResult
from kapso.handles import Brief, Budget, Solution
from kapso.core.config import load_config
from tests.handles_support import (
    candidate_branch,
    facade,
    fake_orchestrator,
    git_workspace,
    solve_result,
)

RECORD_KEYS = [
    "schema_version", "brief", "branch", "commit", "base_commit", "score",
    "goal_achieved", "stopped_reason", "stop_detail", "experiments", "experiments_total",
    "cost_usd", "requests", "feedback", "evaluator", "harness", "experience", "verification",
    "rejected_candidates", "callback_metrics", "callback_primary_metric", "session_failures",
]


def test_a_sentence_becomes_a_brief_and_the_whole_brief_is_the_engines_goal(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    captured = fake_orchestrator(monkeypatch, workspace, solve_result())

    solution = facade().build("Raise AUC above 0.85", output=str(workspace))

    rendered = Brief(objective="Raise AUC above 0.85").render()
    assert captured["goal"] == rendered
    assert captured["handler"].problem_description == f"# Goal\n{rendered}"
    assert solution.brief == Brief(objective="Raise AUC above 0.85")
    assert solution.campaign == workspace.resolve()
    assert (workspace / ".kapso" / "solution.json").is_file()
    assert Solution.open(workspace) == solution


def test_a_large_brief_reaches_the_prompt_whole(tmp_path, monkeypatch):
    """Rule 6 at the prompt boundary: every line the engineer wrote is in
    the problem description the engine hands to the model, and the goal
    text parses back to the same brief."""
    paragraphs = [f"Paragraph {i}: " + ("detail " * 60).strip() for i in range(120)]
    objective = "\n\n".join(paragraphs) + (
        "\n\nThe failing test:\n"
        "    def test_contract():\n"
        "        assert predict(rows).shape == (len(rows),)"
    )
    assert len(objective) > 50_000
    brief = Brief(
        objective=objective,
        measured_by="evaluation/evaluate.py",
        rules=["the evaluator is never modified", "no fitting on the held-out rows"],
        budget=Budget(minutes=5),
    )
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    captured = fake_orchestrator(monkeypatch, workspace, solve_result())

    facade().build(brief, output=str(workspace))

    problem = captured["handler"].problem_description
    for line in objective.splitlines():
        assert line in problem
    assert "the evaluator is never modified" in problem
    assert "Measured by  evaluation/evaluate.py" in problem
    assert Brief.parse(captured["goal"]) == brief


def test_budget_precedence_is_the_call_then_the_brief_then_the_default(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    captured = fake_orchestrator(monkeypatch, workspace, solve_result())
    brief = Brief(objective="g", budget=Budget(minutes=30, experiments=4))

    facade().build(brief, output=str(workspace), budget=Budget(experiments=2))
    assert captured["solve"] == {
        "experiment_max_iter": 2, "time_budget_minutes": 30,
        "cost_budget": None, "finalization_reserve_minutes": None,
    }

    other = tmp_path / "other"
    git_workspace(other)
    captured = fake_orchestrator(monkeypatch, other, solve_result())
    facade().build("g", output=str(other))
    assert captured["solve"]["experiment_max_iter"] == load_config()["budget"]["experiments"]
    assert captured["solve"]["time_budget_minutes"] is None


def test_relative_paths_reach_the_engine_absolute(tmp_path, monkeypatch):
    """A session clone commits its run data by the workspace path from its
    own directory, so a relative output must not reach it (the first
    fresh-machine proof run crashed on `git add ../build/...`). The
    evaluation, data and a local repo are resolved the same way; a URL goes
    to the cloner as it is."""
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "campaigns" / "build"
    git_workspace(workspace)
    for name in ("evaluation", "data", "repo"):
        (tmp_path / name).mkdir()
    (tmp_path / "evaluation" / "evaluate.py").write_text("print('auc: 0.5')\n")
    captured = fake_orchestrator(monkeypatch, workspace, solve_result())

    solution = facade().build(
        "g", output="campaigns/build", evaluation="evaluation", data="data", repo="repo",
    )

    assert captured["workspace_dir"] == str(workspace.resolve())
    assert captured["eval_dir"] == str((tmp_path / "evaluation").resolve())
    assert captured["data_dir"] == str((tmp_path / "data").resolve())
    assert captured["initial_repo"] == str((tmp_path / "repo").resolve())
    assert solution.campaign == workspace.resolve()

    other = tmp_path / "campaigns" / "other"
    git_workspace(other)
    captured = fake_orchestrator(monkeypatch, other, solve_result())
    kapso = facade()
    kapso._clone_github_repo = lambda url: f"cloned:{url}"
    kapso.build("g", output="campaigns/other", repo="https://github.com/org/starter")
    assert captured["initial_repo"] == "cloned:https://github.com/org/starter"


def test_build_needs_a_brief_unless_resuming():
    with pytest.raises(ValueError, match="needs a brief"):
        facade().build()
    with pytest.raises(TypeError, match="string or a Brief"):
        facade().build(42)


def test_a_missing_directory_is_refused_before_anything_starts(tmp_path, monkeypatch):
    """The handler copied data only when it existed, so a typo in --data
    launched a campaign with no dataset and spent the budget before the
    cause showed; now it fails at the door, with the path."""
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    fake_orchestrator(monkeypatch, workspace, solve_result())
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "evaluate.py").write_text("print('auc: 0.5')\n")
    for keyword, path in (("data", tmp_path / "nowhere"), ("repo", tmp_path / "nowhere"),
                          ("evaluation", tmp_path / "nowhere")):
        with pytest.raises(FileNotFoundError, match=f"{keyword} directory not found"):
            facade().build("g", output=str(workspace), **{keyword: str(path)})
    assert not (workspace / ".kapso" / "solution.json").exists()


def test_every_fresh_campaign_writes_a_launch_record_with_the_brief(tmp_path, monkeypatch):
    """The record used to exist only with the inbox on; a resume needs it
    whatever the inbox setting, and it carries the brief."""
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    fake_orchestrator(monkeypatch, workspace, solve_result())   # the inbox off
    brief = Brief(objective="Raise AUC above 0.85", rules=["no peeking"], budget=Budget(minutes=30))

    facade().build(brief, output=str(workspace), budget=Budget(experiments=2))

    record = read_launch_record(workspace)
    assert record["brief"] == brief.to_dict()
    assert record["experiments"] == 2 and record["minutes"] == 30


def test_a_resume_with_the_inbox_off_keeps_the_briefs_budget(tmp_path, monkeypatch):
    """The checkpoint's brief is read before the budget merge, so a resume
    that passes nothing keeps the brief's minutes and dollars even when no
    launch record exists."""
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    brief = Brief(objective="g", budget=Budget(minutes=10, experiments=4, usd=2))
    RunCheckpointStore(str(workspace)).save(_checkpoint(brief.render()))
    captured = fake_orchestrator(monkeypatch, workspace, solve_result())

    facade().build(output=str(workspace), resume=True)

    assert captured["solve"] == {
        "experiment_max_iter": 4, "time_budget_minutes": 10,
        "cost_budget": 2, "finalization_reserve_minutes": None,
    }


def _checkpoint(goal: str) -> RunCheckpoint:
    return RunCheckpoint.create(
        strategy_type="generic", goal=goal, config_fingerprint=config_fingerprint({"m": 1}),
        status="running", completed_iterations=2, cumulative_cost=0.0,
        current_feedback=None, strategy_state={"node_history": []},
    )


def _launch_record(workspace, **values):
    record = {
        "brief": Brief(objective="recorded").to_dict(),
        "config_path": None, "host": None,
        "output_path": str(workspace), "experiments": 3, "minutes": 60.0,
        "usd": None, "reserve_minutes": None, "eval_dir": None, "data_dir": None,
        "resumable_from_inbox": True, "dotenv_path": "",
    }
    record.update(values)
    write_launch_record(workspace, record)


def test_a_resume_reads_the_brief_from_the_checkpoint_and_the_budget_from_the_record(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    # The brief carries its own budget: a bare resume must still run on the
    # launch's (the record's 60 minutes and 3 experiments), not the brief's
    # 45 and 5 — a campaign launched at 40 minutes resumed at its brief's 45
    # when the brief-merged figures were passed off as the caller's (the
    # release gate's S5 on Claude Code, 2026-10-30).
    brief = Brief(objective="Raise accuracy above 0.85", rules=["no peeking at the test set"], budget=Budget(minutes=45, experiments=5))
    RunCheckpointStore(str(workspace)).save(_checkpoint(brief.render()))
    _launch_record(workspace)
    captured = fake_orchestrator(monkeypatch, workspace, solve_result(cumulative_iterations=3))

    solution = facade().build(output=str(workspace), resume=True)

    assert captured["goal"] == brief.render() and captured["resume"] is True
    assert solution.brief == brief
    assert captured["solve"]["experiment_max_iter"] == 3
    assert captured["solve"]["time_budget_minutes"] == 60.0

    # A budget named on the resume wins over the record, field by field.
    captured = fake_orchestrator(monkeypatch, workspace, solve_result(cumulative_iterations=4))
    facade().build(output=str(workspace), resume=True, budget=Budget(minutes=90))
    assert captured["solve"]["experiment_max_iter"] == 3
    assert captured["solve"]["time_budget_minutes"] == 90


def test_the_record_has_the_pinned_shape_and_names_the_delivered_commit(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    base = git_workspace(workspace)
    winner = candidate_branch(workspace, "generic_exp_1", "model.py", "improved = True\n")
    feedback = FeedbackResult(stop=True, evaluation_valid=True, feedback="AUC 0.9 beats the target", score=0.9)
    fake_orchestrator(
        monkeypatch, workspace,
        solve_result(final_feedback=feedback, stopped_reason="goal_achieved", total_cost=1.25),
        best_branch="generic_exp_1", score=0.9,
    )

    solution = facade().build("g", output=str(workspace))

    record = json.loads((workspace / ".kapso" / "solution.json").read_text())
    assert list(record) == RECORD_KEYS
    assert record["schema_version"] == 5
    assert record["brief"] == Brief(objective="g").to_dict()
    assert (record["branch"], record["commit"], record["base_commit"]) == ("generic_exp_1", winner, base)
    assert (record["score"], record["goal_achieved"], record["stopped_reason"]) == (0.9, True, "goal_achieved")
    assert record["cost_usd"] == 1.25 and record["feedback"] == "AUC 0.9 beats the target"
    assert record["evaluator"] is None and record["harness"] is None and record["experience"] == []
    assert "+improved = True" in solution.diff()


def test_the_feedback_is_the_judges_words_on_the_delivered_candidate(tmp_path, monkeypatch):
    # The third proof run delivered node 0 (0.847) while node 1's judgment
    # failed at the budget's end; the record must carry node 0's feedback,
    # not the last node's failure text.
    workspace = tmp_path / "ws"
    git_workspace(workspace)
    candidate_branch(workspace, "generic_exp_0", "model.py", "blend = True\n")
    delivered = SearchNode(node_id=0, branch_name="generic_exp_0", feedback="A solid gain; blend the two models next.", score=0.8474)
    starved = SearchNode(node_id=1, branch_name="generic_exp_1", feedback="Feedback generation failed: the required tags were missing.", score=None)
    last = FeedbackResult(stop=False, evaluation_valid=True, feedback=starved.feedback, score=None)
    fake_orchestrator(
        monkeypatch, workspace,
        solve_result(final_feedback=last, stopped_reason="budget_exhausted", iterations_run=2, cumulative_iterations=2),
        best_branch="generic_exp_0", score=0.8474, history=[delivered, starved],
    )

    solution = facade().build("g", output=str(workspace))

    assert solution.feedback == delivered.feedback
    assert "Feedback:\nA solid gain" in solution.explain()


def test_build_detach_launches_the_cli_in_a_session_of_its_own_and_hands_back_a_launch(tmp_path, monkeypatch):
    # Every engineer of the release gate's S1 wrapped `kapso build --detach`
    # in a self-relaunching worker because build() blocks: the API offers the
    # same launch. The brief goes beside the output as a file the CLI reads,
    # the loaded harness and experience go as their flags, the budget's set
    # fields as theirs; the Launch names the pid, the log and the status.
    import kapso.kapso as facade_module
    calls = []
    monkeypatch.setattr(facade_module, "launch_detached",lambda argv, *, log_path, receipt_path: calls.append((list(argv), log_path, receipt_path)) or 4242)
    output = tmp_path / "campaign"
    brief = Brief(objective="Raise AUC above 0.85", rules=["no peeking"])
    launch = facade().build(brief, repo=str(tmp_path), output=str(output), budget=Budget(minutes=40, experiments=2), detach=True)
    [(argv, log_path, receipt_path)] = calls
    assert launch.pid == 4242 and launch.log == str(tmp_path / "campaign.log") and launch.status == str(output / ".kapso" / "status.json")
    assert log_path == tmp_path / "campaign.log" and receipt_path == tmp_path / "campaign.launch.json"
    brief_file = tmp_path / "campaign.brief.md"
    assert argv == ["build", "--output", str(output), "--brief", str(brief_file), "--repo", str(tmp_path), "--budget-minutes", "40", "--budget-experiments", "2"]
    assert Brief.read(brief_file) == brief, "the brief the child reads is the one given, whole"
    launch = facade().build(output=str(output), resume=True, detach=True)
    assert calls[-1][0] == ["build", "--output", str(output), "--resume"] and launch.pid == 4242
    with pytest.raises(ValueError, match="needs output"):
        facade().build("g", detach=True)
    with pytest.raises(ValueError, match="in-process callbacks"):
        facade().build("g", output=str(output), on_status=lambda status: None, detach=True)


def test_learn_detach_launches_the_cli_campaign_form_and_hands_back_a_launch(tmp_path, monkeypatch):
    import kapso.kapso as facade_module
    calls = []
    monkeypatch.setattr(facade_module, "launch_detached",lambda argv, *, log_path, receipt_path: calls.append((list(argv), log_path, receipt_path)) or 777)
    campaign = tmp_path / "campaign-prereq"
    launch = facade().learn(str(campaign), detach=True)
    [(argv, log_path, receipt_path)] = calls
    assert argv == ["learn", str(campaign)] and receipt_path is None and log_path == tmp_path / "campaign-prereq.learn.log"
    assert launch.pid == 777 and launch.log == str(log_path) and launch.status.endswith("status")
    with pytest.raises(ValueError, match="takes no on_status"):
        facade().learn(str(campaign), exam=False, detach=True)
