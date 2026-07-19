"""Behavioral tests for the read-only coding-agent subprocess boundary."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.execution.search_strategies.generic.ideation.coding_agents import (
    CodingAgentInvocationError,
    CodingAgentRunnerSettings,
    SubprocessCodingAgentCallRunner,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CodingAgentCallRequest,
)


def install_executable(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def request(workspace: Path, cli: str) -> CodingAgentCallRequest:
    return CodingAgentCallRequest(
        operation_id="agent_call_" + "1" * 32,
        role="candidate",
        cli=cli,
        model="test-model",
        prompt="complete prompt\nwith a second line and no truncation",
        workspace=str(workspace),
        timeout_seconds=10,
        effort="high",
        allowed_tools=("Read", "WebSearch"),
    )


def runner(tmp_path: Path) -> SubprocessCodingAgentCallRunner:
    return SubprocessCodingAgentCallRunner(
        CodingAgentRunnerSettings(
            artifact_root=str((tmp_path / "artifacts").resolve()),
            termination_grace_seconds=1,
        )
    )


def test_codex_receives_full_prompt_on_stdin_without_embedding_key(
    tmp_path,
    monkeypatch,
):
    install_executable(
        tmp_path,
        "codex",
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys
pathlib.Path("codex_args.json").write_text(json.dumps(sys.argv[1:]))
pathlib.Path("codex_stdin.txt").write_text(sys.stdin.read())
pathlib.Path("codex_env.txt").write_text(str("OPENAI_API_KEY" in os.environ))
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"structured"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":7}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-agent")

    result = runner(tmp_path).run(request(tmp_path, "codex"), {"type": "object"})

    args = json.loads((tmp_path / "codex_args.json").read_text())
    assert (tmp_path / "codex_stdin.txt").read_text() == request(
        tmp_path, "codex"
    ).prompt
    assert request(tmp_path, "codex").prompt not in args
    assert (tmp_path / "codex_env.txt").read_text() == "False"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in args
    assert "--ignore-user-config" in args
    assert "--search" in args
    assert args[-1] == "-"
    assert json.loads(result.output) == {"proposal": "structured"}
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.cost_usd is None
    assert all(Path(path).is_file() for path in result.artifacts)
    assert Path(result.artifacts[0]).read_text() == request(tmp_path, "codex").prompt


def test_claude_receives_full_prompt_in_plan_mode_without_embedding_key(
    tmp_path,
    monkeypatch,
):
    install_executable(
        tmp_path,
        "claude",
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys
pathlib.Path("claude_args.json").write_text(json.dumps(sys.argv[1:]))
pathlib.Path("claude_stdin.txt").write_text(sys.stdin.read())
pathlib.Path("claude_env.txt").write_text(str("OPENAI_API_KEY" in os.environ))
print(json.dumps({
  "is_error": False,
  "structured_output": {"proposal": "structured"},
  "usage": {"input_tokens": 13, "output_tokens": 5},
  "total_cost_usd": 0.25
}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-agent")

    result = runner(tmp_path).run(
        request(tmp_path, "claude_code"),
        {"type": "object"},
    )

    args = json.loads((tmp_path / "claude_args.json").read_text())
    assert (tmp_path / "claude_stdin.txt").read_text() == request(
        tmp_path, "claude_code"
    ).prompt
    assert request(tmp_path, "claude_code").prompt not in args
    assert (tmp_path / "claude_env.txt").read_text() == "False"
    assert args[args.index("--permission-mode") + 1] == "plan"
    assert "--no-session-persistence" in args
    assert args[args.index("--tools") + 1] == "Read,WebSearch"
    assert json.loads(result.output) == {"proposal": "structured"}
    assert result.input_tokens == 13
    assert result.output_tokens == 5
    assert result.cost_usd == 0.25
    assert all(Path(path).is_file() for path in result.artifacts)


def test_completed_operation_is_reused_without_invoking_the_cli_again(
    tmp_path,
    monkeypatch,
):
    install_executable(
        tmp_path,
        "codex",
        """#!/usr/bin/env python3
import json
import pathlib
import sys
counter = pathlib.Path("invocations.txt")
counter.write_text(counter.read_text() + "x" if counter.exists() else "x")
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"structured"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":7}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    call_runner = runner(tmp_path)

    first = call_runner.run(request(tmp_path, "codex"), {"type": "object"})
    second = call_runner.run(request(tmp_path, "codex"), {"type": "object"})

    assert first == second
    assert (tmp_path / "invocations.txt").read_text() == "x"

    changed_model = replace(request(tmp_path, "codex"), model="different-model")
    with pytest.raises(CodingAgentInvocationError, match="identity was reused"):
        call_runner.run(changed_model, {"type": "object"})
    assert (tmp_path / "invocations.txt").read_text() == "x"


@pytest.mark.parametrize(
    ("name", "source", "expected_exception"),
    [
        ("codex", "#!/bin/sh\nexit 9\n", CodingAgentInvocationError),
        ("codex", "#!/bin/sh\nexit 0\n", CodingAgentInvocationError),
        ("codex", "#!/bin/sh\necho not-json\n", json.JSONDecodeError),
        ("claude", "#!/bin/sh\necho not-json\n", json.JSONDecodeError),
    ],
)
def test_failed_empty_and_malformed_agent_results_propagate(
    tmp_path,
    monkeypatch,
    name,
    source,
    expected_exception,
):
    install_executable(tmp_path, name, source)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    cli = "codex" if name == "codex" else "claude_code"

    with pytest.raises(expected_exception):
        runner(tmp_path).run(request(tmp_path, cli), {"type": "object"})

    artifact_directories = tuple((tmp_path / "artifacts").iterdir())
    assert len(artifact_directories) == 1
    assert (artifact_directories[0] / "prompt.txt").is_file()
    assert (artifact_directories[0] / "stdout.txt").is_file()
    assert (artifact_directories[0] / "stderr.txt").is_file()


def test_runner_rejects_non_absolute_artifact_root():
    with pytest.raises(ValueError, match="absolute"):
        CodingAgentRunnerSettings(
            artifact_root="relative/artifacts",
            termination_grace_seconds=1,
        )
