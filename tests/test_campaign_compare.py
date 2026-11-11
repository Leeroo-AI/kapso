"""`kapso compare`: two campaigns set side by side.

What must hold: a verdict follows the campaigns' own scoring direction and
delivered node; it is withheld when the scores do not mean the same thing
(opposite directions, different registered evaluators) and a different
goal is only a warning; the progress table shows where each search
improved; the command reads the campaigns and changes none of it; a
campaign with no checkpoint, or the same campaign twice, raises.
"""

import json
import subprocess
import sys

import pytest

from kapso.cli import main
from kapso.execution.campaign_compare import CampaignCompareError, compare_campaigns, render_comparison
from kapso.execution.run_checkpoint import RunCheckpointMissingError
from kapso.execution.search_strategies.base import SearchNode
from tests.test_campaign_report import campaign as report_campaign, three_nodes


def campaign(path, nodes, **options):
    """A campaign scored by a registered evaluator unless a test says otherwise."""
    return report_campaign(path, nodes, **{"scores_evaluator_id": "eval-1", **options})


def scored(*scores):
    return [
        SearchNode(node_id=i, branch_name=f"generic_exp_{i}", parent_branch_name="main", score=score)
        for i, score in enumerate(scores)
    ]


def test_higher_is_better_names_the_winner_and_shows_where_each_search_improved(tmp_path):
    a = campaign(tmp_path / "a", three_nodes())
    b = campaign(tmp_path / "b", scored(0.6, 0.5))

    comparison = compare_campaigns(a, b)
    report = render_comparison(comparison)

    assert comparison["verdict"]["winner"] == "a"
    assert "A delivers the better score by 0.3 (higher is better)." in report
    assert "Nothing found that makes the two scores incomparable." in report
    assert "| Delivered score | 0.9 | 0.6 |" in report
    assert "| Delivered branch | `generic_exp_1` | `generic_exp_0` |" in report
    assert "| Candidates | 3 | 2 |" in report
    assert "| Failed | 1 | 0 |" in report
    assert comparison["a"]["progress"] == [0.4, 0.9, 0.9]
    assert comparison["b"]["progress"] == [0.6, 0.6]
    assert report.split("| Candidate | A best | B best |")[1].strip().splitlines()[1:] == [
        "| 1 | 0.4 | 0.6 |",
        "| 2 | 0.9 | 0.6 |",
    ]


def test_lower_is_better_and_a_tie(tmp_path):
    lower = compare_campaigns(
        campaign(tmp_path / "a", scored(0.3, 0.1), maximize=False),
        campaign(tmp_path / "b", scored(0.2), maximize=False),
    )
    assert lower["verdict"]["winner"] == "a"
    assert lower["verdict"]["reason"] == "A delivers the better score by 0.1 (lower is better)"
    assert lower["a"]["progress"] == [0.3, 0.1]
    assert lower["b"]["progress"] == [0.2]

    tie = compare_campaigns(
        campaign(tmp_path / "c", scored(0.5)), campaign(tmp_path / "d", scored(0.5)),
    )
    assert tie["verdict"]["winner"] == "tie"


def test_scores_that_do_not_mean_the_same_thing_get_no_verdict(tmp_path):
    opposite = compare_campaigns(
        campaign(tmp_path / "a", scored(0.9)), campaign(tmp_path / "b", scored(0.1), maximize=False),
    )
    assert opposite["verdict"]["winner"] is None
    assert opposite["verdict"]["reason"].startswith("not comparable: the two campaigns score in opposite directions")

    evaluators = compare_campaigns(
        campaign(tmp_path / "c", scored(0.9), scores_evaluator_id="eval-1"),
        campaign(tmp_path / "d", scored(0.1), scores_evaluator_id="eval-2"),
    )
    assert evaluators["verdict"]["winner"] is None
    report = render_comparison(evaluators)
    assert "- **Blocks a verdict:** the two campaigns were scored by different evaluators (A: eval-1, B: eval-2)" in report
    assert "## Verdict\n\nNot comparable: " in report


def test_a_different_goal_or_no_registered_evaluator_warns_and_leaves_the_verdict(tmp_path):
    comparison = compare_campaigns(
        campaign(tmp_path / "a", scored(0.9), goal="Raise accuracy"),
        campaign(tmp_path / "b", scored(0.1), goal="Cut latency"),
    )

    assert comparison["comparability"] == {
        "blockers": [], "warnings": ["the two campaigns have different goals"],
    }
    assert comparison["verdict"]["winner"] == "a"
    assert "- **Worth knowing:** the two campaigns have different goals" in render_comparison(comparison)

    unregistered = compare_campaigns(
        campaign(tmp_path / "c", scored(0.9), scores_evaluator_id=""),
        campaign(tmp_path / "d", scored(0.1), scores_evaluator_id=""),
    )
    assert unregistered["comparability"] == {
        "blockers": [],
        "warnings": ["neither campaign registered an evaluator, so nothing shows the two scores were measured the same way"],
    }
    assert unregistered["verdict"]["winner"] == "a"


def test_a_campaign_with_nothing_scored_gets_no_verdict(tmp_path):
    comparison = compare_campaigns(
        campaign(tmp_path / "a", [], status="running"), campaign(tmp_path / "b", scored(0.5)),
    )
    report = render_comparison(comparison)

    assert comparison["verdict"] == {"winner": None, "reason": "no verdict: A has no scored candidate"}
    assert "| Delivered score | none | 0.5 |" in report
    assert "| 1 | - | 0.5 |" in report

    empty = render_comparison(compare_campaigns(
        campaign(tmp_path / "c", [], status="running"), campaign(tmp_path / "d", [], status="running"),
    ))
    assert "## Progress\n\nNo scored candidate in either campaign." in empty


def test_compare_leaves_both_campaigns_untouched(tmp_path):
    workspaces = [campaign(tmp_path / "a", three_nodes()), campaign(tmp_path / "b", scored(0.6))]

    def snapshot():
        return {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for workspace in workspaces
            for path in sorted(workspace.rglob("*")) if path.is_file()
        }

    before = snapshot()
    render_comparison(compare_campaigns(*workspaces))
    assert snapshot() == before


def test_a_missing_checkpoint_and_the_same_campaign_twice_raise(tmp_path):
    workspace = campaign(tmp_path / "a", scored(0.5))
    (tmp_path / "empty").mkdir()
    with pytest.raises(RunCheckpointMissingError):
        compare_campaigns(workspace, tmp_path / "empty")
    with pytest.raises(CampaignCompareError, match="two different campaigns"):
        compare_campaigns(workspace, workspace)


def test_naming_one_campaign_twice_is_one_line_at_the_command_line(tmp_path):
    workspace = campaign(tmp_path / "a", scored(0.5))

    run = subprocess.run(
        [sys.executable, "-m", "kapso.cli", "compare", str(workspace), str(workspace)],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )

    assert run.returncode == 1
    assert run.stderr.strip() == f"Error: compare needs two different campaigns, got {workspace} twice"
    assert run.stdout == ""


def test_cli_prints_markdown_json_or_writes_the_file(tmp_path, monkeypatch, capsys):
    a = campaign(tmp_path / "a", three_nodes())
    b = campaign(tmp_path / "b", scored(0.6))

    monkeypatch.setattr(sys, "argv", ["kapso", "compare", str(a), str(b)])
    main()
    assert capsys.readouterr().out.startswith("# Campaign comparison\n")

    monkeypatch.setattr(sys, "argv", ["kapso", "compare", str(a), str(b), "--json"])
    main()
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"]["winner"] == "a"
    assert data["b"]["delivered"] == {"node": 0, "branch": "generic_exp_0", "score": 0.6}

    output = tmp_path / "compare.md"
    monkeypatch.setattr(sys, "argv", ["kapso", "compare", str(a), str(b), "-o", str(output)])
    main()
    assert capsys.readouterr().out == f"Comparison written to {output}\n"
    assert output.read_text().startswith("# Campaign comparison\n")
