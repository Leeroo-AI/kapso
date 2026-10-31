"""`kapso build`: every flag the parser defines is accounted for and
reaches where it belongs (the facade call, the budget, the constructor,
the brief, or the detach handling), so a flag that stops reaching the
facade fails here; a brief file arrives as a `Brief`; a resume without a
brief passes none; and the errors a person can hit say what to do."""

import argparse
import json
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

import kapso.cli.build as build_cli
from kapso.cli import build_parser, main
from kapso.execution.inbox import launch_receipt_path
from kapso.handles import Brief, Budget

FACADE_KWARGS = {"evaluation", "data", "repo", "output", "resume", "coding_agent"}
BUDGET_FIELDS = {"budget_minutes": "minutes", "budget_experiments": "experiments",
                 "budget_usd": "usd", "budget_reserve_minutes": "reserve_minutes"}
CONSTRUCTOR = {"config"}
BRIEF = {"goal", "goal_file", "brief"}
DETACH = {"detach", "log"}
STOP = {"stop"}   # ends the campaign in --output; never a build() keyword


def _parser_dests():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_cli.add_parser(subparsers)
    return {action.dest for action in subparsers.choices["build"]._actions if action.dest != "help"}


def _mock_facade(monkeypatch):
    facade = Mock()
    constructor = Mock(return_value=facade)
    monkeypatch.setattr(build_cli, "Kapso", constructor)
    monkeypatch.setattr(build_cli, "_print_build_summary", lambda _: None)
    return constructor, facade


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["kapso", "build", *argv])
    main()


HARNESS = {"harness"}   # loaded on the facade before build(), never a build() keyword
EXPERIENCE = {"experience"}   # the same: each reference loaded before build()


def test_every_flag_of_the_parser_is_accounted_for():
    assert _parser_dests() == FACADE_KWARGS | set(BUDGET_FIELDS) | CONSTRUCTOR | BRIEF | DETACH | HARNESS | EXPERIENCE | STOP


def test_every_flag_reaches_where_it_belongs(monkeypatch):
    constructor, facade = _mock_facade(monkeypatch)
    _run(
        monkeypatch, "--goal", "g", "--evaluation", "ev", "--data", "da", "--repo", "re",
        "--output", "out", "--budget-experiments", "3", "--budget-minutes", "30", "--budget-usd", "5",
        "--budget-reserve-minutes", "2", "--resume", "--coding-agent", "codex",
        "--config", "c.yaml",
    )
    assert constructor.call_args.kwargs == {"config": "c.yaml"}
    args, kwargs = facade.build.call_args
    assert args == ("g",)
    assert kwargs == {
        "evaluation": "ev", "data": "da", "repo": "re", "output": "out",
        "budget": Budget(minutes=30, experiments=3, usd=5, reserve_minutes=2),
        "resume": True, "coding_agent": "codex",
    }


def test_a_brief_file_reaches_build_as_a_brief(monkeypatch, tmp_path):
    path = tmp_path / "kapso-brief.md"
    path.write_text("Objective    Raise AUC above 0.85\nRules\n  - no peeking\nBudget       minutes=5\n")
    _, facade = _mock_facade(monkeypatch)
    _run(monkeypatch, "--brief", str(path), "--output", "out")
    assert facade.build.call_args.args == (Brief.read(path),)
    assert facade.build.call_args.kwargs["budget"] == Budget()   # the brief's budget applies in the facade


def test_a_goal_file_is_plain_text(monkeypatch, tmp_path):
    path = tmp_path / "problem.txt"
    path.write_text("Raise AUC\nTarget: 0.9, by any means\n")
    _, facade = _mock_facade(monkeypatch)
    _run(monkeypatch, "--goal-file", str(path), "--output", "out")
    assert facade.build.call_args.args == ("Raise AUC\nTarget: 0.9, by any means\n",)


def test_a_resume_without_a_brief_passes_none(monkeypatch):
    _, facade = _mock_facade(monkeypatch)
    _run(monkeypatch, "--output", "out", "--resume")
    assert facade.build.call_args.args == (None,)
    assert facade.build.call_args.kwargs["resume"] is True


def test_no_brief_and_no_resume_exits_one(monkeypatch, capsys):
    _mock_facade(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "--output", "out")
    assert stop.value.code == 1
    assert "--goal, --goal-file or --brief is required" in capsys.readouterr().err


def test_a_missing_brief_exits_one(monkeypatch, capsys, tmp_path):
    _mock_facade(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "--brief", str(tmp_path / "none.md"), "--output", "out")
    assert stop.value.code == 1
    assert "brief not found" in capsys.readouterr().err


def test_abbreviated_flags_are_refused(monkeypatch, capsys):
    """argparse would accept `--det` for `--detach`; the child of a
    detached run strips only the full flag, so an abbreviation would
    detach again, forever."""
    _mock_facade(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "--goal", "g", "--output", "out", "--det")
    assert stop.value.code == 2
    assert "unrecognized arguments: --det" in capsys.readouterr().err


def test_a_log_inside_the_output_directory_is_refused(monkeypatch, capsys, tmp_path):
    """The log is created before the child starts; inside --output it
    would make the fresh campaign directory non-empty, which seeding
    refuses after the parent already reported a launch."""
    _mock_facade(monkeypatch)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "--goal", "g", "--output", "campaign", "--detach", "--log", "campaign/run.log")
    assert stop.value.code == 1
    assert "--log must be outside --output" in capsys.readouterr().err
    assert not (tmp_path / "campaign").exists()


def test_the_brief_forms_exclude_each_other(monkeypatch, capsys, tmp_path):
    path = tmp_path / "kapso-brief.md"
    path.write_text("Objective a\n")
    _mock_facade(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "--goal", "g", "--brief", str(path))
    assert stop.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_the_harness_flag_loads_before_the_build_and_wins_over_the_brief(monkeypatch, tmp_path):
    path = tmp_path / "kapso-brief.md"
    path.write_text("Objective    Raise AUC above 0.85\nHarness      ./harness/other\n")
    constructor, facade = _mock_facade(monkeypatch)
    facade.load_harness.return_value.missing_for.return_value = []
    _run(monkeypatch, "--brief", str(path), "--harness", "./harness/trading", "--output", "out")
    facade.load_harness.assert_called_once_with("./harness/trading")
    assert facade.method_calls[0][0] == "load_harness"   # loaded before build()
    assert facade.build.call_args.args == (Brief.read(path),)


def test_the_briefs_harness_line_loads_when_no_flag_is_given(monkeypatch, tmp_path):
    path = tmp_path / "kapso-brief.md"
    path.write_text("Objective    Raise AUC above 0.85\nHarness      ./harness/trading\n")
    _, facade = _mock_facade(monkeypatch)
    facade.load_harness.return_value.missing_for.return_value = []
    _run(monkeypatch, "--brief", str(path), "--output", "out")
    facade.load_harness.assert_called_once_with("./harness/trading")


def test_a_goal_sentence_loads_no_harness(monkeypatch):
    _, facade = _mock_facade(monkeypatch)
    _run(monkeypatch, "--goal", "g", "--output", "out")
    facade.load_harness.assert_not_called()
    facade.load_experience.assert_not_called()


def test_the_experience_flag_loads_each_source_before_the_build_and_wins_over_the_brief(monkeypatch, tmp_path):
    path = tmp_path / "kapso-brief.md"
    path.write_text("Objective    Raise AUC above 0.85\nExperience   leeroopedia\n")
    _, facade = _mock_facade(monkeypatch)
    _run(monkeypatch, "--brief", str(path), "--experience", "mine, org-x", "--output", "out")
    assert [call.args for call in facade.load_experience.call_args_list] == [("mine",), ("org-x",)]
    assert [name for name, _, _ in facade.method_calls[:2]] == ["load_experience", "load_experience"]
    assert facade.build.call_args.args == (Brief.read(path),)


def test_the_briefs_experience_line_loads_when_no_flag_is_given(monkeypatch, tmp_path):
    path = tmp_path / "kapso-brief.md"
    path.write_text("Objective    Raise AUC above 0.85\nExperience   mine, leeroopedia\n")
    _, facade = _mock_facade(monkeypatch)
    _run(monkeypatch, "--brief", str(path), "--output", "out")
    assert [call.args for call in facade.load_experience.call_args_list] == [("mine",), ("leeroopedia",)]


# ---------------------------------------------------------------- --stop

WRITER = """import subprocess, sys, time
from kapso.execution.observability import BuildStatus
status = BuildStatus(sys.argv[1], heartbeat_seconds=60)
status.phase("ideation")
agent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
open(sys.argv[2], "w").write(str(agent.pid))
time.sleep(60)
"""


def _writer(campaign):
    """A campaign's status writer in a session of its own, as a detached
    launch is, with a coding-agent session under it in a session of its
    own, as the adapters start them: the status file names the writer's
    pid, a file beside it the agent's. Beside the campaign, an override
    file with a one-second grace, so the stop's waits are short."""
    (campaign / ".kapso").mkdir(parents=True)
    (campaign.parent / "kapso-config.yaml").write_text("budget:\n  stop_grace_seconds: 1\n")
    status = campaign / ".kapso" / "status.json"
    agent_pid = campaign / ".kapso" / "agent.pid"
    child = subprocess.Popen([sys.executable, "-c", WRITER, str(status), str(agent_pid)], start_new_session=True)
    deadline = time.monotonic() + 20
    while not (status.is_file() and agent_pid.is_file()) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert status.is_file() and agent_pid.is_file()
    return child, status


def _alive(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid), "-o", "pid="], capture_output=True, text=True).stdout.strip() != ""


def test_stop_ends_the_writer_and_the_agent_session_under_it_and_marks_the_status(monkeypatch, tmp_path, capsys):
    campaign = tmp_path / "campaign"
    child, status = _writer(campaign)
    agent = int((campaign / ".kapso" / "agent.pid").read_text())
    assert _alive(agent)
    receipt = launch_receipt_path(campaign)
    receipt.write_text(json.dumps({"pid": child.pid, "log": "x", "argv": [], "started_at": "t"}))
    _run(monkeypatch, "--output", str(campaign), "--stop", "--config", str(campaign.parent / "kapso-config.yaml"))
    out = capsys.readouterr().out
    assert f"stopped the campaign at {campaign} (pid {child.pid})" in out
    assert f"resume:  kapso build --output {campaign} --resume --detach" in out
    deadline = time.monotonic() + 5
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert child.poll() is not None and not receipt.exists()
    deadline = time.monotonic() + 5
    while _alive(agent) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(agent), "the agent's own session outlived the stop"
    data = json.loads(status.read_text())
    assert data["state"] == "failed" and data["error"] == "stopped by the person" and data["stopped_reason"] == "stopped"
    assert data["finished_at"]
    # Stopped already: refused by state, nothing signalled.
    with pytest.raises(SystemExit) as refusal:
        _run(monkeypatch, "--output", str(campaign), "--stop", "--config", str(campaign.parent / "kapso-config.yaml"))
    assert refusal.value.code == 1 and "is failed; nothing to stop" in capsys.readouterr().err


def test_stop_refuses_other_flags_beside_it(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as refusal:
        _run(monkeypatch, "--output", str(tmp_path / "campaign"), "--stop", "--resume")
    assert refusal.value.code == 1 and "takes no other flag; drop --resume" in capsys.readouterr().err


def test_every_parser_refuses_an_abbreviated_flag():
    parser = build_parser()
    for argv in (["experience", "add", "--repo", "u", "--name", "n", "--det"], ["build", "--output", "x", "--sto"]):
        with pytest.raises(SystemExit) as refusal:
            parser.parse_args(argv)
        assert refusal.value.code == 2


def test_stop_refuses_a_dead_writer_and_names_the_resume(monkeypatch, tmp_path, capsys):
    campaign = tmp_path / "campaign"
    child, status = _writer(campaign)
    child.kill()
    child.wait()
    with pytest.raises(SystemExit) as refusal:
        _run(monkeypatch, "--output", str(campaign), "--stop")
    err = capsys.readouterr().err
    assert refusal.value.code == 1 and f"has no live writer (pid {child.pid}): it is dead, not running" in err
    assert f"kapso build --output {campaign} --resume --detach" in err
    assert json.loads(status.read_text())["state"] == "running"   # untouched: a resume reads it as before
    with pytest.raises(SystemExit) as refusal:
        _run(monkeypatch, "--output", str(tmp_path / "nowhere"), "--stop")
    assert refusal.value.code == 1 and "no status under" in capsys.readouterr().err
    with pytest.raises(SystemExit) as refusal:
        _run(monkeypatch, "--stop")
    assert refusal.value.code == 1 and "--stop needs --output" in capsys.readouterr().err


def test_the_files_beside_a_campaign_follow_a_bare_dot_to_the_directory_it_stands_for(tmp_path, monkeypatch):
    # `kapso watch .` crashed on the receipt's path: `.` has an empty name (found by the plugin in the full-disk record).
    from pathlib import Path
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    monkeypatch.chdir(campaign)
    assert launch_receipt_path(Path(".")) == tmp_path / "campaign.launch.json"
    assert build_cli._default_log(Path(".")) == tmp_path / "campaign.log"
    assert launch_receipt_path(campaign) == tmp_path / "campaign.launch.json"
