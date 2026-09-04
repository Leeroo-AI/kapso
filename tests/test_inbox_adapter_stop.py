"""The adapters end a session once it asks the person (design v4 §4.2).

What must hold, on both CLIs: a `requested` event for this session in
the inbox file arms the stop; a session that ends its own turn within the
grace is a clean stop; one that keeps working is ended after the grace,
which is not a deadline kill; either way the result is a success carrying
the request ids and the session id, never a retryable failure; a request
filed for another session is ignored; Claude is launched with the id
kapso minted and a different id in its init event is a wiring error;
Codex with capture_thread_id passes --json and records the thread id.
"""

import json
import os
import stat
import sys
import time

import pytest

import kapso.execution.coding_agents.adapters.claude_code_agent as claude_module
from kapso.execution.coding_agents.adapters.claude_code_agent import (
    ClaudeCodeCodingAgent,
)
from kapso.execution.coding_agents.adapters.codex_agent import CodexCodingAgent
from kapso.execution.coding_agents.base import CodingAgentConfig

GRACE = 1.0


def requested_line(session: str, request_id: int = 1) -> str:
    return json.dumps({
        "ts": "now", "event": "requested", "id": request_id, "node": 3,
        "session": session, "key": "env:X", "hit": "h", "tried": "t",
        "fix": "f", "next_steps": "n",
    })


# ---------------------------------------------------------------- Claude

CLAUDE_FAKE = r"""
import json, sys, time
inbox, mode, line, init_id = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
print(json.dumps({"type": "system", "subtype": "init", "session_id": init_id, "model": "m", "tools": []}), flush=True)
with open(inbox, "a") as f:
    f.write(line + "\n")
if mode == "sleep":
    time.sleep(30)
print(json.dumps({"type": "result", "result": "", "total_cost_usd": 0.25, "is_error": False, "usage": {}}), flush=True)
"""


def claude_agent(tmp_path, monkeypatch, inbox, *, session_id="sid-1"):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(claude_module.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(ClaudeCodeCodingAgent, "_verify_cli", lambda self: None)
    agent = ClaudeCodeCodingAgent(CodingAgentConfig(
        agent_type="claude_code", model="m", debug_model="m",
        agent_specific={
            "auth_mode": "api_key", "timeout": 30, "streaming": True,
            "session_id": session_id, "inbox_path": str(inbox),
            "inbox_stop_grace_seconds": GRACE,
        },
    ))
    agent.workspace = str(tmp_path)
    agent._auth_mode = "api_key"
    monkeypatch.setattr(agent, "_get_changed_files", lambda: [])
    monkeypatch.setattr(agent, "_get_env", lambda: None)
    return agent


def run_claude(agent, monkeypatch, inbox, mode, line, init_id="sid-1"):
    fake_cmd = [sys.executable, "-u", "-c", CLAUDE_FAKE, str(inbox), mode, line, init_id]
    monkeypatch.setattr(agent, "_build_command", lambda model, use_stream_json=False: fake_cmd)
    return agent._run_streaming("prompt", "m", agent._timeout)


def test_claude_clean_stop_after_asking(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox.jsonl"
    agent = claude_agent(tmp_path, monkeypatch, inbox)
    result = run_claude(agent, monkeypatch, inbox, "clean", requested_line("sid-1"))
    assert result.success and result.error is None
    meta = result.metadata
    assert meta["stopped_for_inbox"] is True
    assert meta["inbox_request_ids"] == [1]
    assert meta["inbox_killed"] is False
    assert meta["session_id"] == "sid-1" and meta["cli_session_id"] == "sid-1"
    assert "deadline_exceeded" not in meta
    assert agent.get_cumulative_cost() == 0.25


def test_claude_is_ended_after_the_grace_when_it_keeps_working(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox.jsonl"
    agent = claude_agent(tmp_path, monkeypatch, inbox)
    started = time.time()
    result = run_claude(agent, monkeypatch, inbox, "sleep", requested_line("sid-1"))
    assert time.time() - started < GRACE + 5
    assert result.success and result.error is None
    assert result.metadata["stopped_for_inbox"] is True
    assert result.metadata["inbox_killed"] is True
    assert "deadline_exceeded" not in result.metadata


def test_claude_ignores_requests_of_other_sessions(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox.jsonl"
    agent = claude_agent(tmp_path, monkeypatch, inbox)
    result = run_claude(agent, monkeypatch, inbox, "clean", requested_line("someone-else"))
    assert result.success
    assert "stopped_for_inbox" not in result.metadata
    assert result.metadata["cli_session_id"] == "sid-1"


def test_claude_init_session_id_mismatch_is_a_wiring_error(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox.jsonl"
    agent = claude_agent(tmp_path, monkeypatch, inbox)
    with pytest.raises(RuntimeError, match="reports session"):
        run_claude(agent, monkeypatch, inbox, "clean", requested_line("sid-1"), init_id="other")


def test_claude_launch_carries_the_minted_session_id(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox.jsonl"
    agent = claude_agent(tmp_path, monkeypatch, inbox)
    cmd = agent._build_command("m", use_stream_json=True)
    assert cmd[cmd.index("--session-id") + 1] == "sid-1"
    assert "--resume" not in cmd


def test_claude_inbox_requires_session_id_and_grace(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(claude_module.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(ClaudeCodeCodingAgent, "_verify_cli", lambda self: None)
    with pytest.raises(ValueError, match="requires session_id"):
        ClaudeCodeCodingAgent(CodingAgentConfig(
            agent_type="claude_code", model="m", debug_model="m",
            agent_specific={"auth_mode": "api_key", "inbox_path": str(tmp_path / "i")},
        ))
    with pytest.raises(ValueError, match="stop_grace_seconds"):
        ClaudeCodeCodingAgent(CodingAgentConfig(
            agent_type="claude_code", model="m", debug_model="m",
            agent_specific={"auth_mode": "api_key", "inbox_path": str(tmp_path / "i"), "session_id": "s"},
        ))


# ----------------------------------------------------------------- Codex

FAKE_CODEX = r"""#!/usr/bin/env bash
last=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --output-last-message) last="${args[i+1]}" ;;
  esac
done
cat > "${FAKE_STDIN_DUMP:-/dev/null}"
if [ -n "$FAKE_CODEX_ARGDUMP" ]; then printf '%s\n' "$*" > "$FAKE_CODEX_ARGDUMP"; fi
echo 'codex log line on stderr-merged stdout'
echo '{"type":"thread.started","thread_id":"thr-123"}'
if [ -n "$FAKE_INBOX_PATH" ]; then printf '%s\n' "$FAKE_REQUEST_LINE" >> "$FAKE_INBOX_PATH"; fi
case "$FAKE_CODEX_MODE" in
  sleep) exec sleep 30 ;;
  *) printf 'FINAL' > "$last" ;;
esac
"""


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "codex"
    script.write_text(FAKE_CODEX)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("FAKE_CODEX_MODE", raising=False)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


def codex_agent(workspace, inbox, *, session_id="sid-9", line=None, capture=True):
    agent = CodexCodingAgent(CodingAgentConfig(
        agent_type="codex", model="m1", debug_model="m1",
        agent_specific={
            "session_id": session_id, "inbox_path": str(inbox),
            "inbox_stop_grace_seconds": GRACE, "capture_thread_id": capture,
            "env_overrides": {
                "FAKE_INBOX_PATH": str(inbox),
                "FAKE_REQUEST_LINE": line or requested_line(session_id),
            },
        },
    ))
    agent.initialize(str(workspace))
    return agent


def test_codex_clean_stop_records_thread_id(tmp_path, fake_codex, monkeypatch):
    argdump = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_CODEX_ARGDUMP", str(argdump))
    inbox = tmp_path / "inbox.jsonl"
    result = codex_agent(fake_codex, inbox).generate_code("go")
    assert result.success and result.error is None
    meta = result.metadata
    assert meta["stopped_for_inbox"] is True
    assert meta["inbox_request_ids"] == [1]
    assert meta["inbox_killed"] is False
    assert meta["deadline_exceeded"] is False
    assert meta["session_id"] == "sid-9"
    assert meta["cli_session_id"] == "thr-123"
    assert "--json" in argdump.read_text()


def test_codex_is_ended_after_the_grace(tmp_path, fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "sleep")
    inbox = tmp_path / "inbox.jsonl"
    started = time.time()
    result = codex_agent(fake_codex, inbox).generate_code("go")
    # The grace, one poll, and the adapter's own SIGTERM-to-SIGKILL window.
    assert time.time() - started < GRACE + 1 + 5 + 3
    assert result.success and result.error is None
    assert result.metadata["inbox_killed"] is True
    assert result.metadata["deadline_exceeded"] is False


def test_codex_ignores_other_sessions_and_json_is_opt_in(tmp_path, fake_codex, monkeypatch):
    argdump = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_CODEX_ARGDUMP", str(argdump))
    inbox = tmp_path / "inbox.jsonl"
    agent = codex_agent(fake_codex, inbox, line=requested_line("someone-else"), capture=False)
    result = agent.generate_code("go")
    assert result.success and result.output == "FINAL"
    assert result.metadata["stopped_for_inbox"] is False
    assert result.metadata["cli_session_id"] is None
    assert "--json" not in argdump.read_text()
