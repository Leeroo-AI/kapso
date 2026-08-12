"""Hermetic tests for the Codex coding-agent adapter.

A fake `codex` executable on PATH stands in for the real CLI: it records
argv and the child environment, honors --output-last-message, and switches
failure modes via FAKE_CODEX_MODE. Pins the adapter contract: env hygiene
(OPENAI_API_KEY passed through — CLI billing is pinned via config.toml's
preferred_auth_method — env_strip honored, env_defaults set-if-absent),
final-message-vs-stream output, deadline kill metadata, and fail-loud
error classification.
"""

import os
import stat
from pathlib import Path

import pytest

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.adapters.codex_agent import CodexCodingAgent

FAKE_CODEX = r"""#!/usr/bin/env bash
last=""
model=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --output-last-message) last="${args[i+1]}" ;;
    -m) model="${args[i+1]}" ;;
  esac
done
cat > /dev/null
if [ -n "$FAKE_CODEX_ENVDUMP" ]; then env > "$FAKE_CODEX_ENVDUMP"; fi
if [ -n "$FAKE_CODEX_ARGDUMP" ]; then printf '%s\n' "$*" > "$FAKE_CODEX_ARGDUMP"; fi
echo "stream line"
case "$FAKE_CODEX_MODE" in
  sleep) sleep 60 ;;
  fail) exit 3 ;;
  nomsg) exit 0 ;;
  *) printf 'FINAL[%s]' "$model" > "$last" ;;
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
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


def make_agent(workspace: Path, **agent_specific) -> CodexCodingAgent:
    config = CodingAgentConfig(
        agent_type="codex",
        model="m1",
        debug_model="dbg-model",
        agent_specific=agent_specific,
    )
    agent = CodexCodingAgent(config)
    agent.initialize(str(workspace))
    return agent


def test_success_env_hygiene_and_argv(tmp_path, fake_codex, monkeypatch):
    envdump = tmp_path / "env.txt"
    argdump = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_CODEX_ENVDUMP", str(envdump))
    monkeypatch.setenv("FAKE_CODEX_ARGDUMP", str(argdump))
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("STRIPME", "gone")
    monkeypatch.setenv("AMBIENT", "wins")

    agent = make_agent(
        fake_codex,
        effort="xhigh",
        env_strip=["STRIPME"],
        env_defaults={"AMBIENT": "loses", "FRESH": "applied"},
        env_overrides={"LANE": "0"},
    )
    result = agent.generate_code("do the thing")

    assert result.success and result.error is None
    assert result.output == "FINAL[m1]"
    assert result.cost == 0.0 and agent.get_cumulative_cost() == 0.0
    assert result.metadata["deadline_exceeded"] is False

    env = dict(
        line.split("=", 1) for line in envdump.read_text().splitlines() if "=" in line
    )
    # OPENAI_API_KEY passes through to the session: CLI billing is pinned
    # by preferred_auth_method in config.toml, and lane tooling needs the
    # key (absent-key sessions shipped 96-row hosted batches, 2026-08-12).
    assert env["OPENAI_API_KEY"] == "secret"
    assert "STRIPME" not in env
    assert env["AMBIENT"] == "wins"
    assert env["FRESH"] == "applied"
    assert env["LANE"] == "0"

    argv = argdump.read_text()
    assert "--sandbox danger-full-access" in argv
    assert 'model_reasoning_effort="xhigh"' in argv
    assert "--search" in argv


def test_deadline_kill_sets_metadata(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "sleep")
    agent = make_agent(fake_codex)
    result = agent.generate_code("slow", timeout_seconds=2)
    assert not result.success
    assert result.metadata["deadline_exceeded"] is True
    assert "deadline" in result.error


def test_nonzero_exit_surfaces_stream_tail(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "fail")
    agent = make_agent(fake_codex)
    result = agent.generate_code("boom")
    assert not result.success
    assert "code 3" in result.error
    assert "stream line" in result.output


def test_clean_exit_without_final_message_fails_loud(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "nomsg")
    agent = make_agent(fake_codex)
    result = agent.generate_code("quiet")
    assert not result.success
    assert "no final message" in result.error


def test_debug_mode_uses_debug_model_and_web_can_be_disabled(
    tmp_path, fake_codex, monkeypatch
):
    argdump = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_CODEX_ARGDUMP", str(argdump))
    agent = make_agent(fake_codex, web_search=False)
    result = agent.generate_code("dbg", debug_mode=True)
    assert result.success
    assert result.output == "FINAL[dbg-model]"
    assert "--search" not in argdump.read_text()


def test_stream_artifact_persists(tmp_path, fake_codex):
    artifact = tmp_path / "forensics" / "session.stream"
    agent = make_agent(fake_codex, stream_artifact_path=str(artifact))
    result = agent.generate_code("record me")
    assert result.success
    assert "stream line" in artifact.read_text()
    assert result.metadata["stream_path"] == str(artifact)


def test_generate_before_initialize_raises(fake_codex):
    config = CodingAgentConfig(
        agent_type="codex", model="m1", debug_model="m1", agent_specific={}
    )
    agent = CodexCodingAgent(config)
    with pytest.raises(RuntimeError, match="initialize"):
        agent.generate_code("x")


def test_mcp_servers_become_config_overrides(tmp_path, fake_codex, monkeypatch):
    argdump = tmp_path / "args.txt"
    monkeypatch.setenv("FAKE_CODEX_ARGDUMP", str(argdump))
    agent = make_agent(
        fake_codex,
        mcp_servers={
            "gated-knowledge": {
                "command": "/env/bin/python",
                "args": ["-m", "kapso.gated_mcp.server"],
                "cwd": "/repo",
                "env": {"MCP_ENABLED_GATES": "repo_memory"},
            }
        },
    )
    result = agent.generate_code("with mcp")
    assert result.success
    argv = argdump.read_text()
    assert 'mcp_servers.gated-knowledge.command="/env/bin/python"' in argv
    assert 'mcp_servers.gated-knowledge.args=["-m", "kapso.gated_mcp.server"]' in argv
    assert 'mcp_servers.gated-knowledge.cwd="/repo"' in argv
    assert 'mcp_servers.gated-knowledge.env={MCP_ENABLED_GATES = "repo_memory"}' in argv


def test_unsafe_mcp_server_name_rejected(fake_codex):
    with pytest.raises(ValueError, match="TOML-bare-key"):
        make_agent(fake_codex, mcp_servers={"bad name!": {"command": "x"}})


def test_streaming_tees_to_console_and_artifact(tmp_path, fake_codex, capfd):
    artifact = tmp_path / "s.stream"
    agent = make_agent(
        fake_codex, streaming=True, stream_artifact_path=str(artifact)
    )
    result = agent.generate_code("stream me")
    assert result.success
    assert "stream line" in artifact.read_text()
    assert "[codex] stream line" in capfd.readouterr().out


def test_buffered_by_default_no_console_output(tmp_path, fake_codex, capfd):
    artifact = tmp_path / "s.stream"
    agent = make_agent(fake_codex, stream_artifact_path=str(artifact))
    result = agent.generate_code("quiet stream")
    assert result.success
    assert "stream line" in artifact.read_text()
    assert "[codex]" not in capfd.readouterr().out
