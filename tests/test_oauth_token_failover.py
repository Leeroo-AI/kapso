"""Hermetic tests for the OAuth session-limit failover (claude-swap).

A Claude Max usage window exhausting mid-run starves the CLI with hard
429s (blocked runs 29-31 for ~1.8h; killed the official fable-5 SmolLM3
run). With recovery tokens configured, the adapter must mark the active
token exhausted, kill the CLI pid-only, and resume the SAME session on
the next healthy token. The end-to-end test drives a real subprocess
pair because the contract is process-level (kill, respawn, --resume).
"""

import json
import os
import sys
import time

import pytest

import kapso.execution.coding_agents.adapters.claude_code_agent as adapter_module
from kapso.execution.coding_agents.adapters.claude_code_agent import (
    ClaudeCodeCodingAgent,
    _OAUTH_EXHAUSTED_UNTIL,
    _parse_session_limit_reset,
    _select_oauth_token,
)
from kapso.execution.coding_agents.base import CodingAgentConfig


@pytest.fixture(autouse=True)
def clean_registry():
    _OAUTH_EXHAUSTED_UNTIL.clear()
    yield
    _OAUTH_EXHAUSTED_UNTIL.clear()


def test_reset_time_parser_contract():
    line = 'rate_limit_event: session limit · resets 8:10pm (UTC)'
    ts = _parse_session_limit_reset(line)
    assert ts is not None and ts > time.time()
    tm = time.gmtime(ts)
    assert (tm.tm_hour, tm.tm_min) == (20, 10)

    am = _parse_session_limit_reset("resets 12:05am (UTC)")
    assert am is not None and time.gmtime(am).tm_hour == 0

    assert _parse_session_limit_reset("no reset clock here") is None
    # A parsed time already in the past rolls to tomorrow (always future).
    past = _parse_session_limit_reset("resets 12:00am (UTC)")
    assert past is not None and past > time.time()


def test_token_selection_order_and_exhaustion():
    tokens = ["main", "rec1", "rec2"]
    assert _select_oauth_token(tokens) == "main"

    _OAUTH_EXHAUSTED_UNTIL["main"] = time.time() + 3600
    assert _select_oauth_token(tokens) == "rec1"

    _OAUTH_EXHAUSTED_UNTIL["rec1"] = time.time() + 7200
    assert _select_oauth_token(tokens) == "rec2"

    # All exhausted: pick the soonest-resetting one, not crash.
    _OAUTH_EXHAUSTED_UNTIL["rec2"] = time.time() + 900
    assert _select_oauth_token(tokens) == "rec2"

    # An expired exhaustion mark means the token is healthy again.
    _OAUTH_EXHAUSTED_UNTIL["main"] = time.time() - 1
    assert _select_oauth_token(tokens) == "main"


def make_oauth_agent(tmp_path, monkeypatch, base_env):
    monkeypatch.setattr(
        adapter_module.shutil, "which", lambda command: f"/usr/bin/{command}"
    )
    config = CodingAgentConfig(
        agent_type="claude_code",
        model="test-model",
        debug_model="test-model",
        agent_specific={
            "auth_mode": "oauth",
            "timeout": 300,
            "streaming": True,
        },
    )
    agent = ClaudeCodeCodingAgent(config)
    agent.workspace = str(tmp_path)
    monkeypatch.setattr(agent, "_get_changed_files", lambda: [])
    monkeypatch.setattr(agent, "_get_effective_env", lambda: dict(base_env))
    return agent


def test_get_env_resolves_one_token_and_strips_recovery_list(tmp_path, monkeypatch):
    agent = make_oauth_agent(
        tmp_path,
        monkeypatch,
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "main-tok",
            "CLAUDE_CODE_OAUTH_RECOVERY_TOKENS": "rec-tok-1,rec-tok-2",
        },
    )

    env = agent._get_env()

    # The child session sees exactly one token; the spare list never
    # reaches an agent environment (containment, like env_strip).
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "main-tok"
    assert "CLAUDE_CODE_OAUTH_RECOVERY_TOKENS" not in env
    assert agent._active_oauth_token == "main-tok"
    assert agent._standby_oauth_tokens == ["rec-tok-1", "rec-tok-2"]

    # Main exhausted -> the next spawn resolves the first recovery token.
    _OAUTH_EXHAUSTED_UNTIL["main-tok"] = time.time() + 3600
    env2 = agent._get_env()
    assert env2["CLAUDE_CODE_OAUTH_TOKEN"] == "rec-tok-1"
    assert agent._standby_oauth_tokens == ["main-tok", "rec-tok-2"]


def test_get_env_without_recovery_tokens_is_unchanged(tmp_path, monkeypatch):
    agent = make_oauth_agent(
        tmp_path, monkeypatch, {"CLAUDE_CODE_OAUTH_TOKEN": "solo-tok"}
    )
    env = agent._get_env()
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "solo-tok"
    assert agent._standby_oauth_tokens == []


def stream_line(payload):
    return f"print({json.dumps(json.dumps(payload))}, flush=True)"


# First spawn: announces its session id, then storms hard limit events and
# hangs (the starved-CLI shape). SIGTERM ends it (default python handling).
STARVED_SCRIPT = "\n".join(
    [
        "import os, time",
        stream_line(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "sess-original",
            }
        ),
        'tok = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")',
        'print(__import__("json").dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "token=" + tok}]}}), flush=True)',
        'limit = __import__("json").dumps({"type": "system", "subtype": "rate_limit_event", "message": "session limit · resets 8:10pm (UTC)"})',
        "for _ in range(3):",
        "    print(limit, flush=True)",
        "time.sleep(120)",
    ]
)

# Respawn: proves which token and how it was invoked, then finishes clean.
RESUMED_SCRIPT = "\n".join(
    [
        "import os, sys, json",
        'tok = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")',
        'print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "resumed-with=" + tok}]}}), flush=True)',
        'print(json.dumps({"type": "result", "result": "done resumed-with=" + tok, "total_cost_usd": 1.25, "is_error": False}), flush=True)',
    ]
)


def test_streaming_swaps_to_recovery_token_and_resumes(tmp_path, monkeypatch):
    agent = make_oauth_agent(
        tmp_path,
        monkeypatch,
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "main-tok",
            "CLAUDE_CODE_OAUTH_RECOVERY_TOKENS": "rec-tok-1",
        },
    )

    build_calls = []

    def fake_build(model, use_stream_json=False, resume_session_id=None):
        build_calls.append(resume_session_id)
        script = STARVED_SCRIPT if resume_session_id is None else RESUMED_SCRIPT
        return [sys.executable, "-u", "-c", script]

    monkeypatch.setattr(agent, "_build_command", fake_build)

    started = time.time()
    result = agent._run_streaming("original prompt", "test-model", 240)

    # Swap happened fast (no waiting out the storm), on the SAME session.
    assert time.time() - started < 60
    assert build_calls == [None, "sess-original"]

    # The starved token is registered exhausted until its advertised reset.
    assert _OAUTH_EXHAUSTED_UNTIL["main-tok"] > time.time()
    reset_tm = time.gmtime(_OAUTH_EXHAUSTED_UNTIL["main-tok"])
    assert (reset_tm.tm_hour, reset_tm.tm_min) == (20, 10)

    # The resumed process ran on the recovery token and completed.
    assert result.success is True
    assert "resumed-with=rec-tok-1" in result.output
    assert result.metadata["oauth_token_swaps"] == 1


def test_no_swap_without_standby_token(tmp_path, monkeypatch):
    """A limit storm with no healthy standby keeps today's behavior: the
    session rides it out (here: until the deadline path kills it)."""
    agent = make_oauth_agent(
        tmp_path, monkeypatch, {"CLAUDE_CODE_OAUTH_TOKEN": "solo-tok"}
    )
    monkeypatch.setattr(
        agent,
        "_build_command",
        lambda model, use_stream_json=False, resume_session_id=None: [
            sys.executable,
            "-u",
            "-c",
            STARVED_SCRIPT,
        ],
    )

    result = agent._run_streaming("prompt", "test-model", 6)

    assert result.metadata["oauth_token_swaps"] == 0
    assert "solo-tok" not in _OAUTH_EXHAUSTED_UNTIL
    assert result.metadata["deadline_exceeded"] is True
