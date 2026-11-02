"""The facade side of the inbox (design v4 §4.5–4.6, §10 scenarios 21–24).

What must hold: a fresh campaign writes the launch record and registers
itself, marking a run with a callback as not resumable from the inbox;
a resume writes neither; Kapso.inbox lists the open requests with the
idea line from the checkpoint; Kapso.reply refuses a live campaign,
records and waits while another request of the node is open, says so
for a non-resumable campaign, and otherwise resumes the campaign with the
launch arguments and the remaining iterations. A resume that fails or is
killed leaves the answer on file, listed as awaiting a resume, and a second
reply retries it.
"""

import json
import shlex
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import git
import pytest

from kapso.core.config import load_config
from tests.handles_support import facade
from tests.harness_support import fake_config
import kapso.kapso as kapso_module
from kapso.core.config import ConfigError
from kapso.execution.inbox import (
    file_requests,
    inbox_path,
    list_registered_campaigns,
    load_requests,
    read_launch_record,
    record_continued,
    record_reply,
    write_launch_record,
)
from kapso.execution.orchestrator import SolveResult
from kapso.execution.run_checkpoint import (
    RunCheckpoint,
    RunCheckpointCorruptError,
    RunCheckpointStore,
    config_fingerprint,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.handles import Brief, Budget
from kapso.kapso import Kapso

ENTRY = {
    "key": "env:OPENAI_API_KEY",
    "hit": "openai.AuthenticationError",
    "tried": "OPENAI_KEY unset too",
    "fix": "add OPENAI_API_KEY=sk-... to .env",
    "next_steps": "embed, re-rank, evaluate",
}


def _init_git_workspace(path: Path) -> None:
    path.mkdir(parents=True)
    repo = git.Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Inbox Test")
        config.set_value("user", "email", "inbox@example.com")
    path.joinpath("README.md").write_text("# Test\n")
    repo.git.add(["README.md"])
    repo.git.commit("-m", "initial")
    repo.git.branch("-M", "main")


def _facade(registry: Path) -> Kapso:
    """The facade stub with the inbox on and its registry in the test's
    directory (every launch registers the campaign)."""
    kapso = facade()
    kapso._config = fake_config(inbox={"enabled": True, "stop_grace_seconds": 120, "registry": str(registry)})
    return kapso


def _fake_orchestrator(workspace: Path, monkeypatch):
    class Strategy:
        def __init__(self):
            self.workspace = SimpleNamespace(workspace_dir=str(workspace))

        def get_experiment_history(self):
            return []

        def get_deliverable_experiment(self):
            return None

        def get_deliverable_score(self):
            return None

        def checkout_to_best_experiment_branch(self):
            return None

    class Orchestrator:
        def __init__(self, handler, **kwargs):
            self.search_strategy = Strategy()
            self.operation_status = SimpleNamespace(path="fake/.kapso/status.json")

        def solve(self, experiment_max_iter, time_budget_minutes=None, cost_budget=None,
                  finalization_reserve_minutes=None, on_status=None, finish=None):
            result = SolveResult(best_experiment=None, final_feedback=None,
                                 stopped_reason="max_iterations", iterations_run=1, total_cost=0.0)
            # the delivery runs inside solve(), as the real orchestrator runs it
            if finish is not None:
                finish(result)
            return result

    monkeypatch.setattr(kapso_module, "OrchestratorAgent", Orchestrator)


def test_a_fresh_campaign_writes_the_launch_record_and_registers(tmp_path, monkeypatch):
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    registry = tmp_path / "registry.jsonl"
    _fake_orchestrator(workspace, monkeypatch)

    _facade(registry).build(
        "Improve support\nmore detail", output=str(workspace),
        budget=Budget(experiments=7, minutes=120),
    )

    record = read_launch_record(workspace)
    assert record["experiments"] == 7 and record["minutes"] == 120
    assert record["output_path"] == str(workspace)
    assert record["host"] is None and record["resumable_from_inbox"] is True
    assert "dotenv_path" in record
    assert record["brief"]["objective"] == "Improve support\nmore detail"
    listed = list_registered_campaigns(registry)
    assert listed[0]["path"] == str(workspace.resolve()) and listed[0]["goal"] == "Improve support"


def test_the_launch_record_keeps_every_path_absolute(tmp_path, monkeypatch):
    """A reply or a resume may run from any directory — a host session's,
    not the launch's — so the record resolves what the launch typed."""
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "evaluate.py").write_text("print(1)\n")
    (tmp_path / "data").mkdir()
    _fake_orchestrator(workspace, monkeypatch)
    monkeypatch.chdir(tmp_path)

    _facade(tmp_path / "registry.jsonl").build("g", output="campaign", evaluation="eval", data="data")

    record = read_launch_record(workspace)
    assert record["eval_dir"] == str((tmp_path / "eval").resolve())
    assert record["data_dir"] == str((tmp_path / "data").resolve())
    assert record["output_path"] == str(workspace.resolve())
    assert record["config_path"] is None


def test_the_launch_record_resolves_a_relative_config_path(tmp_path, monkeypatch):
    """A relative `--config` typed at the launch made the reply fail after
    it had consumed the request (main, 2026-10-29): the record stores it
    resolved, as it stores the evaluation and data directories."""
    monkeypatch.chdir(tmp_path)
    write_launch_record(tmp_path / "campaign", {"config_path": "my.yaml", "eval_dir": None, "data_dir": "data"})
    record = read_launch_record(tmp_path / "campaign")
    assert record["config_path"] == str((tmp_path / "my.yaml").resolve())
    assert record["data_dir"] == str((tmp_path / "data").resolve()) and record["eval_dir"] is None


def test_resume_and_reply_load_the_campaigns_dotenv(tmp_path, monkeypatch):
    workspace = _paused_campaign(tmp_path)
    dotenv = tmp_path / "launch-dir" / ".env"
    record = read_launch_record(workspace)
    write_launch_record(workspace, {**{k: v for k, v in record.items() if k != "schema_version"},
                                    "eval_dir": None, "dotenv_path": str(dotenv)})
    loaded: List[str] = []
    monkeypatch.setattr(kapso_module, "load_dotenv", lambda path: loaded.append(path))

    # A reply, before the agent is even constructed (its knowledge index
    # may need a key), and a resume, before preflight.
    _status(workspace, "done")
    monkeypatch.setattr(Kapso, "__init__", lambda self, config=None, bank=None: None)
    monkeypatch.setattr(Kapso, "build", lambda self, **kwargs: None)
    Kapso.reply(str(workspace), 1, "added the key")
    assert loaded == [str(dotenv)]

    monkeypatch.undo()
    loaded.clear()
    monkeypatch.setattr(kapso_module, "load_dotenv", lambda path: loaded.append(path))
    monkeypatch.setattr(kapso_module, "run_preflight", lambda *args, **kwargs: None)
    _fake_orchestrator(workspace, monkeypatch)
    _facade(tmp_path / "registry.jsonl").build(output=str(workspace), resume=True)
    assert loaded == [str(dotenv)]


def test_a_callback_makes_the_campaign_not_resumable_from_the_inbox(tmp_path, monkeypatch):
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    _fake_orchestrator(workspace, monkeypatch)
    _facade(tmp_path / "registry.jsonl").build("g", output=str(workspace), iteration_evaluator=lambda context: None)
    assert read_launch_record(workspace)["resumable_from_inbox"] is False


def test_a_resume_writes_neither_record(tmp_path, monkeypatch):
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    registry = tmp_path / "registry.jsonl"
    _fake_orchestrator(workspace, monkeypatch)
    _facade(registry).build("g", output=str(workspace), resume=True)
    assert read_launch_record(workspace) is None and not registry.exists()


def _paused_campaign(tmp_path, *, requests=1, resumable=True, completed=2, max_iterations=5) -> Path:
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    entries = [ENTRY] + [{**ENTRY, "key": "data/x.csv"}] * (requests - 1)
    file_requests(inbox_path(workspace), node=3, session="s", entries=entries)
    node = SearchNode(node_id=3, branch_name="generic_exp_3", solution="# Core Idea\nre-rank with embeddings\n",
                      suspended=True, request_ids=list(range(1, requests + 1)), cli_session_id="cli-1")
    RunCheckpointStore(str(workspace)).save(RunCheckpoint.create(
        strategy_type="generic", goal=Brief(objective="Improve support").render(), config_fingerprint=config_fingerprint({"m": 1}),
        status="running", completed_iterations=completed, cumulative_cost=1.0, current_feedback=None,
        strategy_state={"node_history": [node.to_dict()]}, last_stop="waiting_for_user",
    ))
    write_launch_record(workspace, {
        "config_path": None, "host": "codex",
        "output_path": str(workspace), "experiments": max_iterations, "minutes": 90,
        "usd": None, "reserve_minutes": None, "eval_dir": "./eval", "data_dir": None,
        "resumable_from_inbox": resumable, "dotenv_path": "",
    })
    return workspace


def _patch_launch(workspace: Path, **fields: Any) -> None:
    path = workspace / ".kapso" / "launch.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


def _status(workspace: Path, state: str, heartbeat_age: float = 0.0) -> None:
    beat = datetime.now(timezone.utc).timestamp() - heartbeat_age
    stamp = datetime.fromtimestamp(beat, timezone.utc).isoformat(timespec="seconds")
    (workspace / ".kapso" / "status.json").write_text(json.dumps({
        "operation": "build", "phases": ["ideation"], "state": state, "pid": os.getpid(),
        "started_at": stamp, "heartbeat_at": stamp, "heartbeat_seconds": 60, "phase": None,
        "phase_started_at": None, "recent": [],
    }))


def test_inbox_lists_open_requests_with_the_idea_line(tmp_path):
    workspace = _paused_campaign(tmp_path, requests=2)
    requests = Kapso.inbox(str(workspace))
    assert [r.id for r in requests] == [1, 2]
    assert Kapso.inbox_ideas(str(workspace)) == {3: "re-rank with embeddings"}
    assert Kapso.inbox_ideas(str(tmp_path / "nowhere")) == {}


def test_reply_refuses_a_live_campaign(tmp_path):
    workspace = _paused_campaign(tmp_path)
    _status(workspace, "running")
    with pytest.raises(RuntimeError, match="is running"):
        Kapso.reply(str(workspace), 1, "added")
    assert load_requests(inbox_path(workspace))[1].open


def test_reply_records_and_waits_while_another_request_is_open(tmp_path, capsys):
    workspace = _paused_campaign(tmp_path, requests=2)
    _status(workspace, "done")
    assert Kapso.reply(str(workspace), 1, "added the key") is None
    assert load_requests(inbox_path(workspace))[1].reply == "added the key"
    assert "#2 still open, so node 3 waits" in capsys.readouterr().out


def test_reply_says_so_for_a_campaign_it_cannot_resume(tmp_path, capsys):
    workspace = _paused_campaign(tmp_path, resumable=False)
    assert Kapso.reply(str(workspace), 1) is None
    assert "resume it from your script" in capsys.readouterr().out
    assert load_requests(inbox_path(workspace))[1].reply == ""


def test_reply_resumes_with_the_launch_arguments(tmp_path, monkeypatch):
    workspace = _paused_campaign(tmp_path, completed=2, max_iterations=5)
    _status(workspace, "done")
    captured: Dict[str, Any] = {}
    sentinel = object()

    monkeypatch.setattr(Kapso, "__init__", lambda self, config=None, bank=None: captured.update(init=(config,)))
    monkeypatch.setattr(Kapso, "build", lambda self, **kwargs: captured.update(kwargs) or sentinel)

    assert Kapso.reply(str(workspace), 1, "added the key") is sentinel
    assert captured["init"] == (None,)
    # The reply names only what differs from the launch — the remaining
    # iterations — and resumes; build(resume=True) takes the goal from the
    # checkpoint and every other argument from the launch record itself.
    assert captured["resume"] is True
    assert captured["output"] == str(workspace.resolve())
    assert captured["budget"] == Budget(experiments=3)
    assert set(captured) == {"init", "resume", "output", "budget"}


@pytest.mark.parametrize("break_campaign, raised", [
    (lambda workspace: _patch_launch(workspace, config_path="/nowhere/config.yaml"), ConfigError),
    (lambda workspace: (workspace / ".kapso" / "launch.json").write_text('{"schema'), json.JSONDecodeError),
    (lambda workspace: (workspace / ".kapso" / "run_state.json").write_text('{"schema'), RunCheckpointCorruptError),
])
def test_a_resume_that_cannot_start_keeps_the_reply_retryable(tmp_path, break_campaign, raised):
    workspace = _paused_campaign(tmp_path)
    launch = (workspace / ".kapso" / "launch.json").read_text()
    state = (workspace / ".kapso" / "run_state.json").read_text()
    break_campaign(workspace)

    with pytest.raises(raised):
        Kapso.reply(str(workspace), 1, "added the key")

    request = load_requests(inbox_path(workspace))[1]
    assert request.state == "answered" and request.reply == "added the key"
    (workspace / ".kapso" / "launch.json").write_text(launch)
    (workspace / ".kapso" / "run_state.json").write_text(state)
    assert [r.id for r in Kapso.awaiting_resume(str(workspace))] == [1]


def _resume_stub(monkeypatch, *, raises=None) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    def build(self, **kwargs):
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return "resumed"

    monkeypatch.setattr(Kapso, "__init__", lambda self, config=None, bank=None: None)
    monkeypatch.setattr(Kapso, "build", build)
    return calls


def test_a_retry_keeps_the_reply_and_resumes(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    _resume_stub(monkeypatch, raises=RuntimeError("preflight: claude not on PATH"))
    with pytest.raises(RuntimeError, match="preflight"):
        Kapso.reply(str(workspace), 1, "added the key")
    assert f"kapso inbox reply {workspace.resolve()} 1" in capsys.readouterr().out

    calls = _resume_stub(monkeypatch)
    assert Kapso.reply(str(workspace), 1) == "resumed"
    assert Kapso.reply(str(workspace), 1, "added the key") == "resumed"
    assert len(calls) == 2
    assert load_requests(inbox_path(workspace))[1].reply == "added the key"
    assert sum(
        json.loads(line)["event"] == "replied"
        for line in inbox_path(workspace).read_text().splitlines()
    ) == 1


def test_the_printed_retry_line_survives_a_space_in_the_path(tmp_path, monkeypatch, capsys):
    root = tmp_path / "my work"
    root.mkdir()
    workspace = _paused_campaign(root)
    _resume_stub(monkeypatch, raises=RuntimeError("preflight: claude not on PATH"))
    with pytest.raises(RuntimeError, match="preflight"):
        Kapso.reply(str(workspace), 1, "added the key")

    printed = capsys.readouterr().out
    line = printed[printed.index("kapso inbox reply"):].splitlines()[0]
    assert shlex.split(line)[3:] == [str(workspace.resolve()), "1"]


def test_a_retry_with_a_new_note_replaces_the_reply(tmp_path, monkeypatch):
    workspace = _paused_campaign(tmp_path)
    _resume_stub(monkeypatch, raises=RuntimeError("no"))
    with pytest.raises(RuntimeError):
        Kapso.reply(str(workspace), 1, "added the key")
    _resume_stub(monkeypatch)
    assert Kapso.reply(str(workspace), 1, "go on without it") == "resumed"
    assert load_requests(inbox_path(workspace))[1].reply == "go on without it"


def test_a_kill_after_the_session_started_can_still_be_retried(tmp_path, monkeypatch):
    workspace = _paused_campaign(tmp_path)
    record_reply(inbox_path(workspace), 1, "added the key")
    record_continued(inbox_path(workspace), [1], node=3, session="s")
    calls = _resume_stub(monkeypatch)
    assert Kapso.reply(str(workspace), 1) == "resumed"
    assert len(calls) == 1
    # The session may have read that answer; a different one is refused.
    with pytest.raises(ValueError, match="handed to its session"):
        Kapso.reply(str(workspace), 1, "something else")


def test_replying_again_when_nothing_is_owed_never_runs_the_campaign(tmp_path, monkeypatch):
    workspace = _paused_campaign(tmp_path)
    record_reply(inbox_path(workspace), 1, "added the key")
    record_continued(inbox_path(workspace), [1], node=3, session="s")
    store = RunCheckpointStore(str(workspace))
    checkpoint = store.load()
    checkpoint.strategy_state["node_history"][0].update(suspended=False, request_ids=[])
    store.save(checkpoint)
    calls = _resume_stub(monkeypatch)

    assert Kapso.awaiting_resume(str(workspace)) == []
    with pytest.raises(ValueError, match="already answered"):
        Kapso.reply(str(workspace), 1)
    assert calls == []


def test_a_retry_refuses_a_live_campaign_and_it_awaits_nothing(tmp_path):
    workspace = _paused_campaign(tmp_path)
    record_reply(inbox_path(workspace), 1, "added the key")
    _status(workspace, "running")
    assert Kapso.awaiting_resume(str(workspace)) == []
    with pytest.raises(RuntimeError, match="is running"):
        Kapso.reply(str(workspace), 1)


def _two_replies(workspace: Path, monkeypatch, first_resume) -> List[Dict[str, Any]]:
    """The first reply is inside its resume, past the live check, when the
    second reply starts. Returns once both have finished."""
    calls: List[Dict[str, Any]] = []
    inside, release = threading.Event(), threading.Event()

    def build(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            inside.set()
            assert release.wait(10)
            first_resume()
        return "resumed"

    monkeypatch.setattr(Kapso, "__init__", lambda self, config=None, bank=None: None)
    monkeypatch.setattr(Kapso, "build", build)
    first = threading.Thread(target=Kapso.reply, args=(str(workspace), 1, "added the key"))
    second = threading.Thread(target=Kapso.reply, args=(str(workspace), 1))
    first.start()
    assert inside.wait(10)
    second.start()
    second.join(0.3)
    assert second.is_alive() and len(calls) == 1
    release.set()
    first.join(10)
    second.join(10)
    return calls


def test_two_replies_resume_the_campaign_once(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)

    def session_finished():
        record_continued(inbox_path(workspace), [1], node=3, session="s")
        store = RunCheckpointStore(str(workspace))
        checkpoint = store.load()
        checkpoint.strategy_state["node_history"][0].update(suspended=False, request_ids=[])
        store.save(checkpoint)

    assert len(_two_replies(workspace, monkeypatch, session_finished)) == 1
    assert "Another reply already resumed" in capsys.readouterr().out


def test_the_second_reply_takes_over_when_the_first_resume_never_started_a_session(tmp_path, monkeypatch):
    workspace = _paused_campaign(tmp_path)
    assert len(_two_replies(workspace, monkeypatch, lambda: None)) == 2


def test_reply_needs_the_campaign_directory_here(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a campaign directory"):
        Kapso.reply(str(tmp_path / "elsewhere"), 1, "x")
