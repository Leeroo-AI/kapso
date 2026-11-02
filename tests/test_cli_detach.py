"""`--detach`: `kapso build` and `kapso inbox reply` start the real run in a
session of its own and return at once with the pid, the log, the status
path and the watch line — the handoff every host (and the skill) needs,
in one flag instead of a nohup/setsid recipe per host."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

import kapso.cli.experience as experience_cli
import kapso.execution.launch as launch_module
from kapso.cli import main
from kapso.execution.inbox import LAUNCH_SCHEMA_VERSION, inbox_path, load_requests, write_launch_record
from kapso.execution.observability import BuildStatus
from kapso.experience import ExperienceReferenceError
from tests.test_inbox_cli import _paused_campaign


class _Popen:
    calls: List["_Popen"] = []

    def __init__(self, argv: List[str], **kwargs: Any):
        self.argv = argv
        self.kwargs: Dict[str, Any] = kwargs
        self.pid = 4242
        _Popen.calls.append(self)


@pytest.fixture
def popen(monkeypatch):
    _Popen.calls = []
    monkeypatch.setattr(launch_module.subprocess, "Popen", _Popen)
    return _Popen.calls


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["kapso", *argv])
    main()


def test_build_detach_reruns_itself_in_a_session_of_its_own(tmp_path, monkeypatch, popen, capsys):
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "build", "--goal", "g", "--output", "campaign", "--detach", "--budget-experiments", "2")

    [call] = popen
    assert call.argv == [
        sys.executable, "-m", "kapso.cli",
        "build", "--goal", "g", "--output", "campaign", "--budget-experiments", "2",
    ]
    assert call.kwargs["start_new_session"] is True
    assert call.kwargs["stdin"] is subprocess.DEVNULL
    assert call.kwargs["stderr"] is subprocess.STDOUT
    assert call.kwargs["cwd"] == os.getcwd()
    assert Path(call.kwargs["stdout"].name) == tmp_path / "campaign.log"
    assert (tmp_path / "campaign.log").is_file()

    campaign = (tmp_path / "campaign").resolve()
    out = capsys.readouterr().out
    assert "pid 4242" in out
    assert f"kapso watch {campaign} --follow" in out
    assert str(campaign / ".kapso" / "status.json") in out


def test_build_detach_needs_an_output_directory(tmp_path, monkeypatch, popen, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "build", "--goal", "g", "--detach")
    assert stop.value.code == 1
    assert "--detach needs --output" in capsys.readouterr().err
    assert popen == []


def test_build_detach_honours_log_and_strips_it_for_the_child(tmp_path, monkeypatch, popen):
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "build", "--goal", "g", "--output", "c", "--log", "run.log", "--detach")
    [call] = popen
    assert Path(call.kwargs["stdout"].name) == tmp_path / "run.log"
    assert "--log" not in call.argv and "--detach" not in call.argv


def test_reply_detach_resumes_in_a_session_of_its_own_and_records_nothing_here(tmp_path, monkeypatch, popen, capsys):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "reply", str(workspace), "1", "added the key", "--detach")

    [call] = popen
    assert call.argv == [
        sys.executable, "-m", "kapso.cli",
        "inbox", "reply", str(workspace.resolve()), "1", "added the key",
    ]
    assert call.kwargs["start_new_session"] is True
    assert Path(call.kwargs["stdout"].name) == tmp_path / "campaign.log"
    # The child records the reply and resumes; nothing changed here.
    assert load_requests(inbox_path(workspace))[1].open
    assert f"kapso watch {workspace.resolve()} --follow" in capsys.readouterr().out


def test_reply_detach_passes_yes_and_an_empty_note_means_done(tmp_path, monkeypatch, popen):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox", "reply", "--detach", "--yes")
    [call] = popen
    assert call.argv[-4:] == ["inbox", "reply", str(workspace.resolve()), "1"] or \
        call.argv[-5:] == ["inbox", "reply", str(workspace.resolve()), "1", "--yes"]
    assert call.argv[-1] == "--yes"


def test_reply_detach_refuses_a_credential_before_spawning(tmp_path, monkeypatch, popen, capsys):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "reply", "1", "sk-abcdefghijklmnopqrstuvwxyz0123", "--detach")
    assert "looks like a credential" in capsys.readouterr().out
    assert popen == []


def test_reply_detach_refuses_a_live_campaign(tmp_path, monkeypatch, popen, capsys):
    workspace = _paused_campaign(tmp_path)
    BuildStatus(workspace / ".kapso" / "status.json", heartbeat_seconds=60).update(iteration=1)
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "inbox", "reply", "1", "added", "--detach")
    assert stop.value.code == 1
    assert "is running" in capsys.readouterr().out
    assert popen == []


def test_build_detach_validates_the_harness_before_it_forks(tmp_path, monkeypatch, popen, capsys):
    """A missing prerequisite of the build's own modules stops the parent
    with the message, and no child is started to die in its log."""
    harness = tmp_path / "harness"
    shutil.copytree(Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading", harness)
    manifest = harness / "harness.yaml"
    # No requirements file: that check asks python3 in a subprocess, and
    # this test's Popen stub stands in for every process.
    manifest.write_text(manifest.read_text().replace(
        "environment:\n  python: requirements.txt\n",
        "environment:\n  commands: [kapso-test-missing-tool]\n",
    ))
    brief = tmp_path / "kapso-brief.md"
    brief.write_text(f"Objective    Raise the Sharpe above 1\nHarness      {harness}\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as caught:
        _run(monkeypatch, "build", "--brief", str(brief), "--output", "campaign", "--detach")
    assert caught.value.code == 1
    assert popen == []
    out = capsys.readouterr().out
    assert "cannot run a build on this machine; it needs:" in out
    assert "kapso-test-missing-tool (command): install kapso-test-missing-tool and put it on PATH" in out

    # A bare resume takes the harness from the launch record and checks it
    # the same way, before forking.
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    write_launch_record(campaign, {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "harness": {"name": "trading", "version": "0.1.0", "hash": "0" * 64, "source": str(harness)},
    })
    with pytest.raises(SystemExit) as caught:
        _run(monkeypatch, "build", "--output", "campaign", "--resume", "--detach")
    assert caught.value.code == 1 and popen == []
    assert "kapso-test-missing-tool (command)" in capsys.readouterr().out


def test_build_detach_loads_the_experience_before_it_forks(tmp_path, monkeypatch, popen, capsys):
    """A reference that names nothing stops the parent with the forms, and
    no child is started; a source that loads rides into the child's
    command line; a bare resume takes the record's list and refuses a
    source that is gone the same way."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ExperienceReferenceError):
        _run(monkeypatch, "build", "--goal", "g", "--output", "campaign", "--experience", "./nowhere/bank.git", "--detach")
    assert popen == []

    # The hosted gate loads on its key alone (a bank would ask git, which
    # this test's Popen stub stands in for).
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "k")
    _run(monkeypatch, "build", "--goal", "g", "--output", "campaign", "--experience", "leeroopedia", "--detach")
    [call] = popen
    assert call.argv[-2:] == ["--experience", "leeroopedia"]
    assert "Experience: leeroopedia" in capsys.readouterr().out

    campaign = tmp_path / "resumed"
    campaign.mkdir()
    write_launch_record(campaign, {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "experience": [{"reference": str(tmp_path / "gone.git"), "kind": "bank", "resolved": str(tmp_path / "gone.git"),
                        "head": "0" * 40, "provenance": "measured by others", "scope": "all"}],
    })
    with pytest.raises(ExperienceReferenceError):
        _run(monkeypatch, "build", "--output", "resumed", "--resume", "--detach")
    assert len(popen) == 1


def test_learn_detach_runs_the_crews_in_a_session_of_their_own(tmp_path, monkeypatch, popen, capsys):
    """Hours of crews: the same helper, the log beside the campaign, the
    handoff naming the learning status directory; no receipt (nothing
    reads one there)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "campaign").mkdir()
    _run(monkeypatch, "learn", "./campaign", "--harness", "./h", "--detach")
    [call] = popen
    assert call.argv == [sys.executable, "-m", "kapso.cli", "learn", "./campaign", "--harness", "./h"]
    assert call.kwargs["start_new_session"] is True
    assert Path(call.kwargs["stdout"].name) == tmp_path / "campaign.learn.log"
    out = capsys.readouterr().out
    assert "kapso learn started in a session of its own (pid 4242)" in out
    assert "follow:  kapso watch learning/status --follow" in out
    assert not list(tmp_path.glob("*.launch.json"))


def test_experience_add_detach_runs_an_ingest_in_a_session_of_its_own(tmp_path, monkeypatch, popen, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experience_cli, "experience_settings", lambda config: {"indexes": str(tmp_path / "indexes")})
    _run(monkeypatch, "experience", "add", "--repo", "https://github.com/org/x", "--name", "org-x", "--detach")
    [call] = popen
    assert call.argv == [sys.executable, "-m", "kapso.cli", "experience", "add", "--repo", "https://github.com/org/x", "--name", "org-x"]
    assert Path(call.kwargs["stdout"].name) == tmp_path / "indexes" / "org-x.log"
    assert "follow:  kapso watch learning/status --follow" in capsys.readouterr().out
    # An instant add has nothing to detach.
    with pytest.raises(SystemExit) as refusal:
        _run(monkeypatch, "experience", "add", "mine", "--detach")
    assert refusal.value.code == 1 and "--detach goes with --repo or --research" in capsys.readouterr().err
    assert len(popen) == 1
