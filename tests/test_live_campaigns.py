"""`kapso watch` with no path (wp6-observability.md, change 8): the
campaigns the registry knows that are alive or waiting, and the learns
and ingests running under the status directory; `--all` for every one
with its state; the rule is the view's own (`starting` included, a stale
beat or a gone writer out, a campaign waiting on the person in); a
registered campaign with no status yet is told by its launch receipt;
`--follow` needs a campaign."""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.cli.watch as watch_module
from kapso.cli.watch import cmd_watch
import kapso.kapso as kapso_module
from kapso.execution.evaluation_integrity import EvaluationIntegrityError
from kapso.execution.inbox import launch_receipt_path, list_registered_campaigns, register_campaign
from kapso.execution import observability
from kapso.execution.observability import BuildStatus, LessonStatus, campaign_listing, render_listing
from tests.handles_support import facade
from tests.harness_support import fake_config, patch_engine


def campaign(tmp_path, name, *, state=None, stopped_reason=None, stale=False, dead=False, started=False):
    """A registered campaign with a status file in the state asked for."""
    workspace = tmp_path / name
    (workspace / ".kapso").mkdir(parents=True)
    status = BuildStatus(workspace / ".kapso" / "status.json", heartbeat_seconds=60, best={"score": 0.8, "node": 1})
    if not started:
        status.phase("ideation")
    if state == "done":
        status.done(stopped_reason=stopped_reason)
    elif state == "failed":
        status.failed("boom")
    data = json.loads((workspace / ".kapso" / "status.json").read_text())
    if stale:
        data["heartbeat_at"] = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    if dead:
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        data["pid"] = gone.pid
    (workspace / ".kapso" / "status.json").write_text(json.dumps(data))
    register_campaign(tmp_path / "registry.jsonl", workspace, f"{name} goal")
    return workspace


def test_the_listing_keeps_the_alive_and_the_waiting(tmp_path):
    registry = tmp_path / "registry.jsonl"
    status_dir = tmp_path / "status"
    starting = campaign(tmp_path, "starting", started=True)
    running = campaign(tmp_path, "running")
    stale = campaign(tmp_path, "stale", stale=True)
    dead = campaign(tmp_path, "dead", dead=True)
    done = campaign(tmp_path, "done", state="done", stopped_reason="max_iterations")
    waiting = campaign(tmp_path, "waiting", state="done", stopped_reason="waiting_for_user")
    failed = campaign(tmp_path, "failed", state="failed")
    gone = campaign(tmp_path, "gone")
    (gone / ".kapso" / "status.json").unlink()
    (gone / ".kapso").rmdir()
    gone.rmdir()
    # A campaign registered but not yet seeded: its receipt says it is starting.
    receipt_only = tmp_path / "receipt-only"
    receipt_only.mkdir()
    register_campaign(registry, receipt_only, "receipt goal")
    launch_receipt_path(receipt_only).write_text(json.dumps({"pid": os.getpid(), "log": "x.log", "argv": [], "started_at": "2026-10-25T18:00:00+00:00"}))
    # A learn running under the status directory.
    status_dir.mkdir()
    LessonStatus(status_dir / "learn-1.json", trajectory_id="c/1", heartbeat_seconds=60).phase("exam")

    listed = campaign_listing(registry, status_dir)
    by_path = {entry["path"]: entry for entry in listed}
    # `stale` stays: past its beats, but its writer is running (this process).
    assert set(by_path) == {str(starting), str(running), str(stale), str(waiting), str(receipt_only), str(status_dir / "learn-1.json")}
    assert by_path[str(starting)]["state"] == "starting" and by_path[str(running)]["phase"] == "ideation"
    assert by_path[str(receipt_only)]["state"] == "starting" and by_path[str(receipt_only)]["alive"] is True
    assert by_path[str(status_dir / "learn-1.json")]["operation"] == "learn"
    assert by_path[str(running)]["best"] == {"score": 0.8, "node": 1}
    # The registry's newest first, then the status directory.
    assert [entry["path"] for entry in listed][0] == str(receipt_only)

    everything = campaign_listing(registry, status_dir, everything=True)
    states = {entry["path"]: entry["state"] for entry in everything}
    assert states[str(stale)] == "running" and states[str(dead)] == "running"
    assert states[str(done)] == "done" and states[str(failed)] == "failed"
    assert str(gone) not in states
    assert {entry["alive"] for entry in everything if entry["path"] in (str(dead), str(done), str(failed))} == {False}

    text = render_listing(listed)
    assert "WAITING" in text and "RUNNING   build      ideation" in text and str(status_dir / "learn-1.json") in text


def test_watch_with_no_path_lists_and_follow_needs_a_campaign(tmp_path, monkeypatch, capsys):
    registry = tmp_path / "registry.jsonl"
    campaign(tmp_path, "running")
    monkeypatch.setattr(watch_module, "load_config", lambda path: fake_config(
        inbox={"registry": str(registry)}, learning={"status_dir": str(tmp_path / "none")},
    ))
    cmd_watch(SimpleNamespace(path=None, json=True, follow=False, all=False, config=None))
    [entry] = json.loads(capsys.readouterr().out)
    assert entry["path"] == str(tmp_path / "running") and entry["state"] == "running"
    cmd_watch(SimpleNamespace(path=None, json=False, follow=False, all=False, config=None))
    assert str(tmp_path / "running") in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cmd_watch(SimpleNamespace(path=None, json=False, follow=True, all=False, config=None))
    assert "needs a campaign to follow" in capsys.readouterr().err
    registry.unlink()
    cmd_watch(SimpleNamespace(path=None, json=False, follow=False, all=False, config=None))
    assert capsys.readouterr().out.startswith("nothing running or waiting")


def test_the_listing_says_dead_for_a_writer_that_is_gone(tmp_path):
    """The single-campaign screen says DEAD when the pid is gone; the
    listing said RUNNING from the file's own word (the pull-request
    review's finding)."""
    registry = tmp_path / "registry.jsonl"
    campaign(tmp_path, "dead", dead=True)
    campaign(tmp_path, "stale", stale=True)
    campaign(tmp_path, "live")
    lines = render_listing(campaign_listing(registry, tmp_path / "status", everything=True)).splitlines()
    states = {line.split()[-1].rsplit("/", 1)[-1]: line.split()[0] for line in lines}
    # `stale` is past its beats with its writer running (this test's process):
    # the single-campaign screen says STALLED, and a resume is refused.
    assert states == {"dead": "DEAD", "stale": "STALLED", "live": "RUNNING"}


def test_where_the_kernel_cannot_say_a_silent_writer_is_dead(tmp_path, monkeypatch):
    monkeypatch.setattr(observability, "process_start", lambda pid: None)
    registry = tmp_path / "registry.jsonl"
    campaign(tmp_path, "stale", stale=True)
    campaign(tmp_path, "live")
    lines = render_listing(campaign_listing(registry, tmp_path / "status", everything=True)).splitlines()
    states = {line.split()[-1].rsplit("/", 1)[-1]: line.split()[0] for line in lines}
    assert states == {"stale": "DEAD", "live": "RUNNING"}


def test_a_launch_refused_at_a_check_is_registered_all_the_same(tmp_path, monkeypatch):
    """The WP6 proof run's first launch died at the evaluation suite's
    refusal before the campaign directory existed, and `kapso watch --all`
    knew nothing of it: a launch with an output directory is registered
    before its checks, so the listing shows it stopped by its receipt."""
    patch_engine(monkeypatch)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "evaluate.py").write_text("print('score: 1')\n")
    (evaluation / "result.json").write_text("{}")
    workspace = tmp_path / "campaign"
    with pytest.raises(EvaluationIntegrityError, match="left by an earlier run"):
        facade().build("Improve support", evaluation=str(evaluation), output=str(workspace))
    assert not workspace.exists()
    registry = kapso_module.load_config()["inbox"]["registry"]
    [line] = Path(registry).read_text().splitlines()
    assert json.loads(line)["path"] == str(workspace.resolve()) and json.loads(line)["goal"] == "Improve support"
    # A foreground launch leaves no receipt and no directory: nothing to
    # list. A detached one's receipt, its pid gone, lists the campaign
    # stopped.
    assert list_registered_campaigns(registry) == []
    assert campaign_listing(registry, tmp_path / "status", everything=True) == []
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    launch_receipt_path(workspace).write_text(json.dumps({"pid": gone.pid, "log": str(tmp_path / "campaign.log"),
                                                          "argv": ["build"], "started_at": "2026-10-25T22:07:49+00:00"}))
    [listed] = campaign_listing(registry, tmp_path / "status", everything=True)
    assert listed["state"] == "stopped" and listed["path"] == str(workspace.resolve())

