"""The facade side of the inbox (design v4 §4.5–4.6, §10 scenarios 21–24).

What must hold: a fresh campaign writes the launch record and registers
itself, marking a run with a callback as not resumable from the inbox;
a resume writes neither; Kapso.inbox lists the open requests with the
idea line from the checkpoint; Kapso.reply refuses a live campaign,
records and waits while another request of the node is open, says so
for a non-resumable campaign, and otherwise resumes the campaign with the
launch arguments and the remaining iterations.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import git
import pytest

import kapso.kapso as kapso_module
from kapso.execution.inbox import (
    file_requests,
    inbox_path,
    list_registered_campaigns,
    load_requests,
    read_launch_record,
    write_launch_record,
)
from kapso.execution.orchestrator import SolveResult
from kapso.execution.run_checkpoint import RunCheckpoint, RunCheckpointStore, config_fingerprint
from kapso.execution.search_strategies.base import SearchNode
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


def _facade() -> Kapso:
    kapso = Kapso.__new__(Kapso)
    kapso.config_path = None
    kapso.knowledge_search = SimpleNamespace(is_enabled=lambda: False)
    kapso._config = {}
    kapso._bank_home = None
    kapso._kg_index_path = None
    return kapso


def _fake_orchestrator(workspace: Path, monkeypatch):
    class Strategy:
        def __init__(self):
            self.workspace = SimpleNamespace(workspace_dir=str(workspace))

        def get_experiment_history(self):
            return []

        def get_deliverable_score(self):
            return None

        def checkout_to_best_experiment_branch(self):
            return None

    class Orchestrator:
        def __init__(self, handler, **kwargs):
            self.search_strategy = Strategy()
            self.operation_status = SimpleNamespace(path="fake/.kapso/status.json")

        def solve(self, experiment_max_iter, time_budget_minutes=None, cost_budget=None,
                  finalization_reserve_minutes=None, on_status=None):
            return SolveResult(best_experiment=None, final_feedback=None,
                               stopped_reason="max_iterations", iterations_run=1, total_cost=0.0)

    monkeypatch.setattr(kapso_module, "OrchestratorAgent", Orchestrator)


def _inbox_on(monkeypatch, registry: Path):
    monkeypatch.setattr(
        kapso_module, "load_mode_config",
        lambda config_path, mode: {"inbox": {"enabled": True, "stop_grace_seconds": 120, "registry": str(registry)}},
    )


def test_a_fresh_campaign_writes_the_launch_record_and_registers(tmp_path, monkeypatch):
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    registry = tmp_path / "registry.jsonl"
    _fake_orchestrator(workspace, monkeypatch)
    _inbox_on(monkeypatch, registry)

    _facade().evolve(
        goal="Improve support\nmore detail", output_path=str(workspace), max_iterations=7,
        time_budget_minutes=120, eval_dir=None, mode="GENERIC", context=["a hint"],
    )

    record = read_launch_record(workspace)
    assert record["max_iterations"] == 7 and record["time_budget_minutes"] == 120
    assert record["mode"] == "GENERIC" and record["output_path"] == str(workspace)
    assert record["context"] == ["a hint"] and record["resumable_from_inbox"] is True
    assert "dotenv_path" in record
    listed = list_registered_campaigns(registry)
    assert listed[0]["path"] == str(workspace.resolve()) and listed[0]["goal"] == "Improve support"


def test_a_callback_makes_the_campaign_not_resumable_from_the_inbox(tmp_path, monkeypatch):
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    _fake_orchestrator(workspace, monkeypatch)
    _inbox_on(monkeypatch, tmp_path / "registry.jsonl")
    _facade().evolve(goal="g", output_path=str(workspace), iteration_evaluator=lambda context: None)
    assert read_launch_record(workspace)["resumable_from_inbox"] is False


def test_a_resume_writes_neither_record(tmp_path, monkeypatch):
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    registry = tmp_path / "registry.jsonl"
    _fake_orchestrator(workspace, monkeypatch)
    _inbox_on(monkeypatch, registry)
    _facade().evolve(goal="g", output_path=str(workspace), resume=True)
    assert read_launch_record(workspace) is None and not registry.exists()


def _paused_campaign(tmp_path, *, requests=1, resumable=True, completed=2, max_iterations=5) -> Path:
    workspace = tmp_path / "campaign"
    _init_git_workspace(workspace)
    entries = [ENTRY] + [{**ENTRY, "key": "data/x.csv"}] * (requests - 1)
    file_requests(inbox_path(workspace), node=3, session="s", entries=entries)
    node = SearchNode(node_id=3, branch_name="generic_exp_3", solution="# Core Idea\nre-rank with embeddings\n",
                      suspended=True, request_ids=list(range(1, requests + 1)), cli_session_id="cli-1")
    RunCheckpointStore(str(workspace)).save(RunCheckpoint.create(
        strategy_type="generic", goal="Improve support", config_fingerprint=config_fingerprint({"m": 1}),
        status="running", completed_iterations=completed, cumulative_cost=1.0, current_feedback=None,
        strategy_state={"node_history": [node.to_dict()]}, last_stop="waiting_for_user",
    ))
    write_launch_record(workspace, {
        "config_path": None, "kg_index": None, "mode": "GENERIC", "coding_agent": "codex",
        "output_path": str(workspace), "max_iterations": max_iterations, "time_budget_minutes": 90,
        "cost_budget": None, "finalization_reserve_minutes": None, "eval_dir": "./eval",
        "data_dir": None, "additional_context": "", "context": None, "serving_scope": None,
        "resumable_from_inbox": resumable, "dotenv_path": "",
    })
    return workspace


def _status(workspace: Path, state: str, heartbeat_age: float = 0.0) -> None:
    beat = datetime.now(timezone.utc).timestamp() - heartbeat_age
    stamp = datetime.fromtimestamp(beat, timezone.utc).isoformat(timespec="seconds")
    (workspace / ".kapso" / "status.json").write_text(json.dumps({
        "operation": "evolve", "phases": ["ideation"], "state": state, "pid": os.getpid(),
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

    monkeypatch.setattr(Kapso, "__init__", lambda self, config_path=None, kg_index=None, bank=None: captured.update(init=(config_path, kg_index)))
    monkeypatch.setattr(Kapso, "evolve", lambda self, **kwargs: captured.update(kwargs) or sentinel)

    assert Kapso.reply(str(workspace), 1, "added the key") is sentinel
    assert captured["init"] == (None, None)
    assert captured["resume"] is True
    assert captured["goal"] == "Improve support"
    assert captured["output_path"] == str(workspace.resolve())
    assert captured["max_iterations"] == 3
    assert captured["mode"] == "GENERIC" and captured["coding_agent"] == "codex"
    assert captured["eval_dir"] == "./eval" and captured["time_budget_minutes"] == 90


def test_reply_needs_the_campaign_directory_here(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a campaign directory"):
        Kapso.reply(str(tmp_path / "elsewhere"), 1, "x")
