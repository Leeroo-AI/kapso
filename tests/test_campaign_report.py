"""`kapso report`: a campaign rendered as shareable Markdown.

What must hold: the report reads only what the campaign wrote (checkpoint,
status file, launch record, inbox, branches) and changes none of it; the
leaderboard and the deliverable follow the campaign's own ranking,
including minimize metrics and cost-aware selection; the lineage is drawn
from the checkpoint's node history; a missing checkpoint or a corrupt
state file raises.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kapso.cli import main
from kapso.core.config import load_config
from kapso.execution.campaign_report import render_campaign_report
from kapso.execution.inbox import (
    file_requests,
    inbox_path,
    record_reply,
    write_launch_record,
)
from kapso.execution.observability import BuildStatus
from kapso.execution.run_checkpoint import (
    RunCheckpoint,
    RunCheckpointCorruptError,
    RunCheckpointMissingError,
    RunCheckpointStore,
    config_fingerprint,
)
from kapso.execution.search_strategies.base import SearchNode

REQUEST = {
    "key": "env:OPENAI_API_KEY",
    "hit": "openai.AuthenticationError",
    "tried": "OPENAI_KEY unset too",
    "fix": "add OPENAI_API_KEY to .env",
    "next_steps": "embed, re-rank, evaluate",
}


def git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(workspace), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True, text=True, check=True,
    ).stdout


def commit_branch(workspace: Path, branch: str, parent: str, filename: str, text: str) -> None:
    git(workspace, "checkout", "-q", "-b", branch, parent)
    (workspace / filename).write_text(text)
    git(workspace, "add", filename)
    git(workspace, "commit", "-q", "-m", branch)


def campaign(tmp_path: Path, nodes, *, status="done", maximize=True, cost_config=None,
             scores_evaluator_id="") -> Path:
    workspace = tmp_path / "campaign"
    workspace.mkdir(parents=True)
    git(workspace, "init", "-q", "-b", "main")
    (workspace / "train.py").write_text("print('baseline')\n")
    git(workspace, "add", "train.py")
    git(workspace, "commit", "-q", "-m", "baseline")
    for node in nodes:
        parent = node.parent_branch_name
        commit_branch(workspace, node.branch_name, parent, f"{node.branch_name}.py", "x = 1\n")
    git(workspace, "checkout", "-q", "main")
    RunCheckpointStore(str(workspace)).save(RunCheckpoint.create(
        strategy_type="generic", goal="Raise accuracy\n\nKeep inference under 50ms",
        config_fingerprint=config_fingerprint({}), status="running",
        completed_iterations=len(nodes), cumulative_cost=1.5, current_feedback=None,
        strategy_state={
            "node_history": [node.to_dict() for node in nodes],
            "scores_evaluator_id": scores_evaluator_id,
        },
        elapsed_seconds=1800.0, cost_config=cost_config, maximize_scoring=maximize,
    ))
    write_launch_record(workspace, {"experiments": 10, "minutes": 60, "usd": 5.0})
    state = BuildStatus(workspace / ".kapso" / "status.json")
    state.update(iteration=len(nodes))
    if status == "done":
        state.done(stopped_reason="budget_exhausted", stop_detail="finalization_reserve")
    return workspace


def three_nodes():
    return [
        SearchNode(node_id=0, branch_name="generic_exp_0", parent_branch_name="main",
                   score=0.4, solution="# Core Idea\nlogistic regression\n"),
        SearchNode(node_id=1, branch_name="generic_exp_1", parent_node_id=0,
                   parent_branch_name="generic_exp_0", score=0.9,
                   solution="# Core Idea\ngradient boosting | tuned\n",
                   metrics={"auc": 0.91, "f1": 0.8}, primary_metric="auc"),
        SearchNode(node_id=2, branch_name="generic_exp_2", parent_branch_name="main",
                   had_error=True, solution="crashed idea"),
    ]


def leaderboard_nodes(report: str):
    section = report.split("## Leaderboard")[1].split("## Lineage")[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and line[2].isdigit()]
    return [int(row.split(" | ")[1]) for row in rows]


def test_finished_campaign_report_has_every_section(tmp_path):
    workspace = campaign(tmp_path, three_nodes())

    report = render_campaign_report(workspace, top=5)

    assert report.startswith("# Campaign report\n\n> Raise accuracy\n>\n> Keep inference under 50ms\n")
    assert "- **State:** done, stopped: budget_exhausted (finalization_reserve)" in report
    assert "- **Iterations:** 3 of 10" in report
    assert "- **Elapsed:** 30.0 of 60 min" in report
    assert "- **Model spend:** $1.50 of $5.00" in report
    assert "Platform cost" not in report
    assert (
        "- **Node:** 1\n- **Branch:** `generic_exp_1`\n- **Score:** 0.9\n"
        "- **Idea:** gradient boosting | tuned\n"
        "- **External metrics:** auc 0.91 (primary), f1 0.8"
    ) in report
    assert "| 1 | 1 | 0.9 | 0 | `generic_exp_1` | gradient boosting \\| tuned |" in report
    assert "| 2 | 0 | 0.4 | main | `generic_exp_0` | logistic regression |" in report
    assert "Top 2 of 2 scored candidates" in report
    assert (
        "## Lineage\n\n```text\n"
        "  ├─ generic_exp_0 (score=0.4, ok)\n"
        "  │ └─ generic_exp_1 (score=0.9, ok)\n"
        "  └─ generic_exp_2 (score=unscored, error)\n```"
    ) in report
    assert "### Node 1: `generic_exp_1` against `generic_exp_0`" in report
    assert "generic_exp_1.py | 1 +" in report
    assert "generic_exp_0.py" not in report.split("### Node 1")[1].split("### Node 0")[0]
    assert "## Human input\n\nNo requests." in report
    assert "- **Candidates:** 3\n- **Scored:** 2\n- **Failed:** 1" in report
    assert render_campaign_report(workspace, top=1).count("### Node") == 1


def test_running_campaign_reports_a_snapshot(tmp_path):
    workspace = campaign(tmp_path, [], status="running")

    report = render_campaign_report(workspace, top=5)

    assert "- **State:** running\n" in report
    assert "## Deliverable\n\nNo scored candidate yet." in report
    assert "## Leaderboard\n\nNo scored candidate yet." in report
    assert "## Lineage\n\nNo candidates yet." in report


def test_leaderboard_follows_minimize_and_cost_aware_ranking(tmp_path):
    def nodes(costs):
        return [
            SearchNode(node_id=i, branch_name=f"generic_exp_{i}", parent_branch_name="main",
                       score=score,
                       measured_cost=None if costs is None else {"amount": costs[i], "unit": "credits"})
            for i, score in enumerate([0.3, 0.1, 0.2])
        ]

    minimize = campaign(tmp_path / "minimize", nodes(None), maximize=False)
    assert leaderboard_nodes(render_campaign_report(minimize, top=5)) == [1, 2, 0]

    cost_config = {"enabled": True, "estimator": "pkg.estimate", "weight": 1.0}
    costed = campaign(tmp_path / "costed", nodes([0.25, 0.0, 0.0]), cost_config=cost_config)
    report = render_campaign_report(costed, top=5)
    assert leaderboard_nodes(report) == [2, 1, 0]
    assert "| Rank | Node | Score | Cost (credits) | Parent | Branch | Idea |" in report
    assert "| 3 | 0 | 0.3 | 0.25 | main |" in report
    assert "- **Platform cost:** 0.25 credits" in report


def test_deliverable_is_the_committed_candidate_not_the_fast_leader(tmp_path):
    full = {"commit_sha": "abc", "evaluator_id": "eval-1", "fidelity": "full",
            "fraction": 1.0, "seed": 0, "score": 0.7}
    nodes = [
        SearchNode(node_id=0, branch_name="generic_exp_0", parent_branch_name="main", score=0.95),
        SearchNode.from_dict({"node_id": 1, "branch_name": "generic_exp_1",
                              "parent_branch_name": "main", "score": 0.6,
                              "evaluation_attempts": [full]}),
    ]
    workspace = campaign(tmp_path, nodes, scores_evaluator_id="eval-1")

    report = render_campaign_report(workspace, top=5)

    assert "- **Node:** 1\n- **Branch:** `generic_exp_1`\n- **Score:** 0.7" in report
    assert leaderboard_nodes(report) == [0, 1]


def test_inbox_requests_and_their_replies(tmp_path):
    workspace = campaign(tmp_path, three_nodes(), status="running")
    path = inbox_path(workspace)
    file_requests(path, node=1, session="s", entries=[REQUEST, {**REQUEST, "key": "data/x.csv"}])
    record_reply(path, 1, "added it | in .env\nthanks")

    report = render_campaign_report(workspace, top=5)

    assert (
        "| Request | Node | Needs | Fix | State | Reply |\n|---|---|---|---|---|---|\n"
        "| #1 | 1 | `env:OPENAI_API_KEY` | add OPENAI_API_KEY to .env | answered "
        "| added it \\| in .env thanks |\n"
        "| #2 | 1 | `data/x.csv` | add OPENAI_API_KEY to .env | open | waiting |"
    ) in report


def test_report_leaves_the_campaign_untouched(tmp_path):
    workspace = campaign(tmp_path, three_nodes())

    # Every file, .git included: refs, HEAD and the index are files too.
    def snapshot():
        return {
            str(path.relative_to(workspace)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in sorted(workspace.rglob("*")) if path.is_file()
        }

    before = snapshot()
    render_campaign_report(workspace, top=5)
    assert snapshot() == before


def test_missing_checkpoint_and_corrupt_state_raise(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RunCheckpointMissingError):
        render_campaign_report(tmp_path / "empty", top=5)

    workspace = campaign(tmp_path, three_nodes())
    status_file = workspace / ".kapso" / "status.json"
    status_file.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        render_campaign_report(workspace, top=5)

    inbox_path(workspace).write_text("{not json\n")
    status_file.unlink()
    BuildStatus(status_file).update(iteration=3)
    with pytest.raises(json.JSONDecodeError):
        render_campaign_report(workspace, top=5)

    (workspace / ".kapso" / "run_state.json").write_text("{not json")
    with pytest.raises(RunCheckpointCorruptError):
        render_campaign_report(workspace, top=5)

    with pytest.raises(ValueError, match="top must be a positive integer"):
        render_campaign_report(workspace, top=0)


def test_cli_writes_the_file_and_reads_top_from_config(tmp_path, monkeypatch, capsys):
    nodes = [
        SearchNode(node_id=i, branch_name=f"generic_exp_{i}", parent_branch_name="main", score=i)
        for i in range(7)
    ]
    workspace = campaign(tmp_path, nodes)

    monkeypatch.setattr(sys, "argv", ["kapso", "report", str(workspace)])
    main()
    default_size = load_config()["report"]["leaderboard_size"]
    assert len(leaderboard_nodes(capsys.readouterr().out)) == default_size

    output = tmp_path / "report.md"
    monkeypatch.setattr(
        sys, "argv", ["kapso", "report", str(workspace), "--top", "2", "-o", str(output)]
    )
    main()
    assert capsys.readouterr().out == f"Report written to {output}\n"
    assert leaderboard_nodes(output.read_text()) == [6, 5]
