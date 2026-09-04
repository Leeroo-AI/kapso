"""The continuation: an adapter resumes a stored session with one more
user message (design v4 §4.4–4.5).

What must hold: Claude resumes with `--resume <id>` and no `--session-id`,
keeping every launch flag, the follow-up on stdin; Codex resumes with
`codex exec <exec options> resume <thread> -`, the follow-up on stdin,
exec options before `resume`; an SDK adapter has no resume.
"""

import os
import stat
from types import SimpleNamespace

import pytest

import kapso.execution.coding_agents.adapters.claude_code_agent as claude_module
from kapso.execution.coding_agents.adapters.claude_code_agent import (
    ClaudeCodeCodingAgent,
)
from kapso.execution.coding_agents.adapters.codex_agent import CodexCodingAgent
from kapso.execution.coding_agents.base import (
    CodingAgentConfig,
    CodingAgentInterface,
    CodingResult,
)


def claude_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(claude_module.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(ClaudeCodeCodingAgent, "_verify_cli", lambda self: None)
    agent = ClaudeCodeCodingAgent(CodingAgentConfig(
        agent_type="claude_code", model="claude-opus-5", debug_model="claude-opus-5",
        agent_specific={
            "auth_mode": "api_key", "streaming": True, "session_id": "sid-1",
            "allowed_tools": ["Read", "mcp__gated-knowledge__request_from_user"],
            "effort": "max",
        },
    ))
    agent.workspace = str(tmp_path)
    agent._auth_mode = "api_key"
    agent._mcp_config_path = tmp_path / "mcp.json"
    agent._mcp_config_path.write_text("{}")
    monkeypatch.setattr(agent, "_get_changed_files", lambda: [])
    return agent


def test_claude_resume_command_keeps_launch_flags(monkeypatch, tmp_path):
    agent = claude_agent(monkeypatch, tmp_path)
    launch = agent._build_command("claude-opus-5", use_stream_json=True)
    resumed = agent._build_command("claude-opus-5", use_stream_json=True, resume_session_id="sid-1")
    assert resumed[resumed.index("--resume") + 1] == "sid-1"
    assert "--session-id" not in resumed
    for flag in ("--model", "--effort", "--allowedTools", "--disallowedTools", "--mcp-config", "--output-format"):
        assert launch[launch.index(flag) + 1] == resumed[resumed.index(flag) + 1]


def test_claude_resume_pipes_the_follow_up_on_stdin(monkeypatch, tmp_path):
    agent = claude_agent(monkeypatch, tmp_path)
    captured = {}

    class FakeStdin:
        def __init__(self):
            self.data = ""
            self.closed = False

        def write(self, text):
            self.data += text

        def close(self):
            self.closed = True

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = None
            self.stderr = None
            self.pid = 999

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, cwd, stdin, stdout, stderr, text, env, bufsize, start_new_session):
        captured.update(cmd=cmd)
        proc = FakeProc()
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(claude_module.subprocess, "Popen", fake_popen)
    result = agent.resume("sid-1", "Request #1 env:X — reply: added. Continue.")
    assert result.success
    assert captured["cmd"][captured["cmd"].index("--resume") + 1] == "sid-1"
    assert captured["proc"].stdin.data.startswith("Request #1")
    assert captured["proc"].stdin.closed is True
    assert all("Continue." not in part for part in captured["cmd"])
    assert result.metadata["cli_session_id"] == "sid-1"


FAKE_CODEX = r"""#!/usr/bin/env bash
last=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --output-last-message) last="${args[i+1]}" ;;
  esac
done
cat > "$FAKE_STDIN_DUMP"
printf '%s\n' "$*" > "$FAKE_CODEX_ARGDUMP"
printf 'RESUMED' > "$last"
"""


def test_codex_resume_command_and_stdin(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "codex"
    script.write_text(FAKE_CODEX)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    argdump, stdin_dump = tmp_path / "args.txt", tmp_path / "stdin.txt"
    monkeypatch.setenv("FAKE_CODEX_ARGDUMP", str(argdump))
    monkeypatch.setenv("FAKE_STDIN_DUMP", str(stdin_dump))
    agent = CodexCodingAgent(CodingAgentConfig(
        agent_type="codex", model="gpt-5.6-sol", debug_model="gpt-5.6-sol",
        agent_specific={"effort": "xhigh", "capture_thread_id": True},
    ))
    agent.initialize(str(tmp_path))
    result = agent.resume("thr-1", "Request #1 env:X — reply: added. Continue.")
    assert result.success and result.output == "RESUMED"
    argv = argdump.read_text().split()
    assert argv[argv.index("resume"):] == ["resume", "thr-1", "-"]
    assert argv.index("--output-last-message") < argv.index("resume")
    assert argv.index("--json") < argv.index("resume")
    assert argv.index("-m") < argv.index("resume") and argv[argv.index("-m") + 1] == "gpt-5.6-sol"
    assert stdin_dump.read_text().startswith("Request #1")


def test_sdk_adapters_cannot_resume():
    class Minimal(CodingAgentInterface):
        def initialize(self, workspace):
            pass

        def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
            return CodingResult(success=True, output="")

        def cleanup(self):
            pass

    agent = Minimal(CodingAgentConfig(agent_type="x", model="m", debug_model="m"))
    with pytest.raises(NotImplementedError, match="cannot resume"):
        agent.resume("id", "follow up")
