"""Hermetic tests for streaming-path cost honesty and deadline enforcement.

The fake CLI is a real subprocess emitting stream-json events, because the
contracts under test are process-level: group kill, grace-window flush, and
cost capture on failure paths.
"""

import json
import sys
import time
from pathlib import Path

import pytest

from kapso.execution.coding_agents.adapters.claude_code_agent import (
    ClaudeCodeCodingAgent,
)
from kapso.execution.coding_agents.base import CodingAgentConfig


def make_agent(tmp_path, monkeypatch, *, timeout):
    config = CodingAgentConfig(
        agent_type="claude_code",
        model="test-model",
        debug_model="test-model",
        agent_specific={"timeout": timeout, "streaming": True},
    )
    agent = ClaudeCodeCodingAgent(config)
    agent.workspace = str(tmp_path)
    monkeypatch.setattr(agent, "_get_changed_files", lambda: [])
    monkeypatch.setattr(agent, "_get_env", lambda: None)
    return agent


def run_fake_cli(agent, monkeypatch, script, extra_args=()):
    fake_cmd = [sys.executable, "-u", "-c", script, *extra_args]
    monkeypatch.setattr(
        agent,
        "_build_command",
        lambda model, use_stream_json=False: fake_cmd,
    )
    return agent._run_streaming("prompt", "test-model", agent._timeout)


def stream_event(payload):
    return f"print({json.dumps(json.dumps(payload))}, flush=True)"


FAILURE_WITH_COST_SCRIPT = "\n".join(
    [
        stream_event(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "working"}]},
            }
        ),
        stream_event(
            {
                "type": "result",
                "result": "boom",
                "total_cost_usd": 1.25,
                "is_error": True,
            }
        ),
    ]
)

SUCCESS_WITH_COST_SCRIPT = "\n".join(
    [
        stream_event(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "working"}]},
            }
        ),
        stream_event(
            {
                "type": "result",
                "result": "done",
                "total_cost_usd": 0.75,
                "is_error": False,
            }
        ),
    ]
)

HANG_SCRIPT = "\n".join(
    [
        stream_event({"type": "assistant", "message": {"content": []}}),
        "import time",
        "time.sleep(60)",
    ]
)

SIGTERM_FLUSH_SCRIPT = "\n".join(
    [
        "import json, signal, sys, time",
        "def handler(signum, frame):",
        "    print(json.dumps({'type': 'result', 'result': 'terminated',"
        " 'total_cost_usd': 0.5, 'is_error': True}), flush=True)",
        "    sys.exit(1)",
        "signal.signal(signal.SIGTERM, handler)",
        stream_event({"type": "assistant", "message": {"content": []}}),
        "time.sleep(60)",
    ]
)

GRANDCHILD_SCRIPT = "\n".join(
    [
        "import json, subprocess, sys, time",
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(60)'])",
        "open(sys.argv[1], 'w').write(str(child.pid))",
        stream_event({"type": "assistant", "message": {"content": []}}),
        "time.sleep(60)",
    ]
)


def test_failed_call_reports_its_parsed_cost(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, timeout=30)
    result = run_fake_cli(agent, monkeypatch, FAILURE_WITH_COST_SCRIPT)

    assert result.success is False
    assert result.error == "boom"
    assert result.cost == 1.25
    assert agent.get_cumulative_cost() == 1.25


def test_successful_call_cost_recorded_once(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, timeout=30)
    result = run_fake_cli(agent, monkeypatch, SUCCESS_WITH_COST_SCRIPT)

    assert result.success is True
    assert result.cost == 0.75
    assert agent.get_cumulative_cost() == 0.75


def test_deadline_terminates_streaming_call(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, timeout=2)
    started = time.monotonic()
    result = run_fake_cli(agent, monkeypatch, HANG_SCRIPT)
    took = time.monotonic() - started

    assert result.success is False
    assert "deadline" in result.error
    assert result.metadata["deadline_exceeded"] is True
    assert result.metadata["elapsed_seconds"] >= 2
    assert took < 15


def test_deadline_grace_captures_flushed_cost(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, timeout=2)
    result = run_fake_cli(agent, monkeypatch, SIGTERM_FLUSH_SCRIPT)

    assert result.success is False
    assert result.metadata["deadline_exceeded"] is True
    assert result.cost == 0.5
    assert agent.get_cumulative_cost() == 0.5


def test_deadline_kill_takes_the_whole_process_group(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, timeout=2)
    pid_file = tmp_path / "grandchild.pid"
    result = run_fake_cli(
        agent, monkeypatch, GRANDCHILD_SCRIPT, extra_args=(str(pid_file),)
    )

    assert result.metadata["deadline_exceeded"] is True
    grandchild_pid = int(pid_file.read_text())
    for _ in range(50):
        if not Path(f"/proc/{grandchild_pid}").exists():
            break
        time.sleep(0.1)
    assert not Path(f"/proc/{grandchild_pid}").exists()


def test_generate_code_threads_the_per_call_timeout(tmp_path, monkeypatch):
    """Budget clamps reach individual calls on a long-lived agent through
    generate_code's timeout_seconds; the configured timeout stands when
    the caller passes nothing.
    """
    agent = make_agent(tmp_path, monkeypatch, timeout=300)
    agent.workspace = "/tmp/nowhere"
    seen = []
    monkeypatch.setattr(
        agent,
        "_build_command",
        lambda prompt, model, use_stream_json=False: ["true"],
    )
    monkeypatch.setattr(
        agent,
        "_run_streaming",
        lambda cmd, model, timeout_seconds: seen.append(timeout_seconds),
    )

    agent.generate_code("p", timeout_seconds=7.0)
    agent.generate_code("p")

    assert seen == [7.0, 300]


def test_stream_artifact_persists_raw_events(tmp_path, monkeypatch):
    """stream_artifact_path tees every raw stream-json line to disk —
    the per-session forensics that survive kills and feed the
    technical-difficulties fallback."""
    artifact = tmp_path / "forensics" / "stream.jsonl"
    config = CodingAgentConfig(
        agent_type="claude_code",
        model="test-model",
        debug_model="test-model",
        agent_specific={
            "timeout": 30,
            "streaming": True,
            "stream_artifact_path": str(artifact),
        },
    )
    agent = ClaudeCodeCodingAgent(config)
    agent.workspace = str(tmp_path)
    monkeypatch.setattr(agent, "_get_changed_files", lambda: [])
    monkeypatch.setattr(agent, "_get_env", lambda: None)

    run_fake_cli(agent, monkeypatch, SUCCESS_WITH_COST_SCRIPT)

    lines = artifact.read_text().strip().splitlines()
    assert len(lines) >= 2  # assistant event + result event
    assert any('"type": "result"' in ln or '"type":"result"' in ln for ln in lines)
