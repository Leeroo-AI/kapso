"""The prompt must never ride in the CLI's argv.

Run #8 lost two implementation sessions because the solution text lived in
the claude CLI's command line: the agent's own `pkill -f <word-from-plan>`
matched its ancestor and SIGTERMed the session. These pin the stdin
contract for both run paths and the argv-free command builder.
"""

import subprocess
from types import SimpleNamespace

import kapso.execution.coding_agents.adapters.claude_code_agent as claude_module
from kapso.execution.coding_agents.base import CodingAgentConfig


def make_agent(monkeypatch, streaming):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(claude_module.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(
        claude_module.ClaudeCodeCodingAgent,
        "_verify_cli",
        lambda self: None,
    )
    agent = claude_module.ClaudeCodeCodingAgent(
        CodingAgentConfig(
            agent_type="claude_code",
            model="claude-opus-4-8",
            debug_model="claude-opus-4-8",
            workspace="",
            agent_specific={"auth_mode": "api_key", "streaming": streaming},
        )
    )
    agent.workspace = "/tmp"
    agent._auth_mode = "api_key"
    return agent


def test_build_command_contains_no_prompt(monkeypatch):
    agent = make_agent(monkeypatch, streaming=False)
    cmd = agent._build_command("claude-opus-4-8", use_stream_json=True)
    assert cmd[:2] == ["claude", "-p"]
    joined = " ".join(cmd)
    assert "pkill" not in joined  # sanity: nothing content-like in argv
    # every element is a flag or a known value, never free text
    assert all(len(part) < 200 for part in cmd)


def test_buffered_run_pipes_prompt_via_stdin(monkeypatch):
    agent = make_agent(monkeypatch, streaming=False)
    captured = {}

    def fake_run(cmd, cwd, input, capture_output, text, timeout, env):
        captured.update(cmd=cmd, input=input)
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(claude_module.subprocess, "run", fake_run)
    result = agent.generate_code("SECRET PLAN mentioning vllm and pkill")
    assert result.success
    assert captured["input"] == "SECRET PLAN mentioning vllm and pkill"
    assert all("SECRET PLAN" not in part for part in captured["cmd"])


def test_streaming_run_pipes_prompt_via_stdin(monkeypatch):
    agent = make_agent(monkeypatch, streaming=True)
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
    result = agent.generate_code("SECRET PLAN mentioning vllm")
    assert captured["proc"].stdin.data == "SECRET PLAN mentioning vllm"
    assert captured["proc"].stdin.closed is True
    assert all("SECRET PLAN" not in part for part in captured["cmd"])
