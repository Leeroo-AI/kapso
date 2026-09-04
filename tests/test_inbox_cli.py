"""`kapso inbox` and `kapso inbox reply` (design v4 §4.6, §10 scenarios 21–26).

What must hold: inside a campaign `kapso inbox` lists its open requests
with the idea line and the reply hint (the id left out when one request
is open); from elsewhere a named directory or the registry is used;
`kapso inbox reply` resolves the campaign and the id the same way,
refuses a credential unless --yes, and prints the campaign's summary —
COMPLETED, or WAITING ON YOU when it asked again.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

import kapso.cli as cli_module
from kapso.cli import _print_evolve_summary, main
from kapso.execution.inbox import file_requests, inbox_path, register_campaign
from kapso.execution.run_checkpoint import RunCheckpoint, RunCheckpointStore, config_fingerprint
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.solution import SolutionResult
from kapso.kapso import Kapso

ENTRY = {
    "key": "env:OPENAI_API_KEY",
    "hit": "openai.AuthenticationError",
    "tried": "OPENAI_KEY unset too",
    "fix": "add OPENAI_API_KEY=sk-... to .env",
    "next_steps": "embed, re-rank, evaluate",
}


def _paused_campaign(root: Path, name: str = "campaign", *, requests: int = 1) -> Path:
    workspace = root / name
    workspace.mkdir(parents=True)
    entries = [ENTRY] + [{**ENTRY, "key": "data/x.csv"}] * (requests - 1)
    file_requests(inbox_path(workspace), node=3, session="s", entries=entries)
    node = SearchNode(
        node_id=3, branch_name="generic_exp_3", solution="# Core Idea\nre-rank with embeddings\n",
        suspended=True, request_ids=list(range(1, requests + 1)), cli_session_id="cli-1",
    )
    RunCheckpointStore(str(workspace)).save(RunCheckpoint.create(
        strategy_type="generic", goal="Improve support\nmore detail",
        config_fingerprint=config_fingerprint({"m": 1}), status="running", completed_iterations=2,
        cumulative_cost=1.0, current_feedback=None, strategy_state={"node_history": [node.to_dict()]},
        last_stop="waiting_for_user",
    ))
    return workspace


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["kapso", *argv])
    main()


def _capture_reply(monkeypatch, result=None) -> List[Any]:
    calls: List[Any] = []

    def reply(cls, campaign, request_id, note=""):
        calls.append((campaign, request_id, note))
        return result

    monkeypatch.setattr(Kapso, "reply", classmethod(reply))
    return calls


def test_inbox_inside_a_campaign_lists_its_requests(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox")
    out = capsys.readouterr().out
    assert out.startswith(f"{workspace.resolve()}  Improve support  waiting 1m")
    assert "#1  env:OPENAI_API_KEY" in out
    assert "for    node 3 · re-rank with embeddings" in out
    assert 'reply with   kapso inbox reply "…"' in out


def test_inbox_names_a_campaign_from_elsewhere(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path, requests=2)
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", str(workspace))
    out = capsys.readouterr().out
    assert "· 2 requests" in out and "#2  data/x.csv" in out
    assert 'reply with   kapso inbox reply 1 "…"' in out

    monkeypatch.setattr(Kapso, "inbox", staticmethod(lambda campaign: []))
    _run(monkeypatch, "inbox", str(workspace))
    assert capsys.readouterr().out.strip() == f"Nothing waiting on you in {workspace.resolve()}."


def test_inbox_from_elsewhere_reads_the_registry(tmp_path, monkeypatch, capsys):
    registry = tmp_path / "registry.jsonl"
    config = tmp_path / "config.yaml"
    config.write_text(
        "default_mode: GENERIC\nmodes:\n  GENERIC:\n    inbox:\n"
        f"      registry: {registry}\n"
    )
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "--config", str(config))
    assert capsys.readouterr().out.strip() == "Nothing waiting on you."

    quiet = tmp_path / "quiet"
    quiet.mkdir()
    register_campaign(registry, quiet, "Nothing asked")
    waiting = _paused_campaign(tmp_path, "waiting")
    register_campaign(registry, waiting, "Improve support")
    _run(monkeypatch, "inbox", "--config", str(config))
    out = capsys.readouterr().out
    assert str(waiting) in out and str(quiet) not in out
    assert "#1  env:OPENAI_API_KEY" in out


def test_reply_resolves_the_campaign_and_the_id(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    calls = _capture_reply(monkeypatch)

    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, "inbox", "reply", str(workspace), "1", "added the key")
    monkeypatch.chdir(workspace)
    _run(monkeypatch, "inbox", "reply", "added the key")
    _run(monkeypatch, "inbox", "reply")

    resolved = str(workspace.resolve())
    assert calls == [
        (resolved, 1, "added the key"),
        (resolved, 1, "added the key"),
        (resolved, 1, ""),
    ]
    assert capsys.readouterr().out == ""


def test_reply_needs_the_id_when_several_are_open(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path, requests=2)
    calls = _capture_reply(monkeypatch)
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "reply", "done")
    assert "which request?" in capsys.readouterr().out and calls == []

    _run(monkeypatch, "inbox", "reply", "2", "done")
    assert calls == [(str(workspace.resolve()), 2, "done")]


def test_reply_outside_a_campaign_must_name_it(tmp_path, monkeypatch, capsys):
    calls = _capture_reply(monkeypatch)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "reply", "1", "done")
    assert "not inside a campaign" in capsys.readouterr().out and calls == []


def test_reply_refuses_a_credential_without_yes(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    calls = _capture_reply(monkeypatch)
    monkeypatch.chdir(workspace)
    token = "sk-abcdefghijklmnopqrstuvwxyz0123"
    with pytest.raises(SystemExit):
        _run(monkeypatch, "inbox", "reply", "1", token)
    assert "looks like a credential" in capsys.readouterr().out and calls == []

    _run(monkeypatch, "inbox", "reply", "1", token, "--yes")
    assert calls == [(str(workspace.resolve()), 1, token)]


def _solution(workspace: Path, metadata: Dict[str, Any]) -> SolutionResult:
    return SolutionResult(goal="Improve support", code_path=str(workspace), metadata=metadata)


def test_reply_prints_the_campaign_summary(tmp_path, monkeypatch, capsys):
    workspace = _paused_campaign(tmp_path)
    monkeypatch.chdir(workspace)
    _capture_reply(monkeypatch, _solution(workspace, {"stopped_reason": "max_iterations", "cost": 2.5}))
    _run(monkeypatch, "inbox", "reply", "1", "added the key")
    out = capsys.readouterr().out
    assert "COMPLETED" in out and "Stopped reason: max_iterations" in out and "Cost: 2.5" in out


def test_the_summary_says_waiting_on_you_when_the_campaign_asked_again(tmp_path, capsys):
    workspace = tmp_path / "campaign"
    request = {"id": 2, "key": "env:OPENAI_API_KEY"}
    _print_evolve_summary(_solution(workspace, {
        "stopped_reason": "waiting_for_user", "requests": [request, {**request, "id": 3}],
    }))
    out = capsys.readouterr().out
    assert "WAITING ON YOU" in out and "COMPLETED" not in out
    assert f'Reply with:  kapso inbox reply 2 "…"   (inside {workspace})' in out
    assert f"Any time:    kapso inbox {workspace}" in out


def test_inbox_help_shows_both_forms(monkeypatch, capsys):
    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, "inbox", "--help")
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "reply [CAMPAIGN] [ID] [NOTE]" in out and "kapso inbox reply 1" in out
    assert cli_module.cmd_inbox.__doc__
