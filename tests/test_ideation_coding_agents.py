"""Behavioral tests for the read-only coding-agent subprocess boundary."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    CodingAgentInvocationError,
    CodingAgentRunnerSettings,
    SubprocessCodingAgentCallRunner,
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
    assert "--skip-git-repo-check" in args
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

    artifact_directories = tuple(
        path for path in (tmp_path / "artifacts").iterdir() if path.is_dir()
    )
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


def test_interrupted_operation_retries_with_exact_persisted_identity(
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
marker = pathlib.Path("first_attempt_failed")
if not marker.exists():
    marker.write_text("failed")
    print("first attempt", file=sys.stderr)
    sys.exit(9)
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"recovered"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":4,"output_tokens":2}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    call_runner = runner(tmp_path)

    with pytest.raises(CodingAgentInvocationError, match="status 9"):
        call_runner.run(request(tmp_path, "codex"), {"type": "object"})

    operation_directory = (
        tmp_path / "artifacts" / request(tmp_path, "codex").operation_id
    )
    prompt_before_retry = (operation_directory / "prompt.txt").read_bytes()
    result = call_runner.run(
        request(tmp_path, "codex"),
        {"type": "object"},
    )

    assert json.loads(result.output) == {"proposal": "recovered"}
    assert (operation_directory / "prompt.txt").read_bytes() == prompt_before_retry
    assert (operation_directory / "stderr.txt").read_text() == ""
    assert (operation_directory / "result.json").is_file()


def test_retry_recovers_atomic_input_temporary_file_left_by_crash(
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
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"recovered"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":4,"output_tokens":2}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    artifact_root = tmp_path / "artifacts"
    operation_directory = artifact_root / request(tmp_path, "codex").operation_id
    operation_directory.mkdir(parents=True)
    (operation_directory / "prompt.txt").write_text(request(tmp_path, "codex").prompt)
    temporary_schema = operation_directory / ".response_schema.json.tmp"
    temporary_schema.write_text("partial schema from terminated process")

    result = runner(tmp_path).run(
        request(tmp_path, "codex"),
        {"type": "object"},
    )

    assert json.loads(result.output) == {"proposal": "recovered"}
    assert not temporary_schema.exists()
    assert json.loads((operation_directory / "response_schema.json").read_text()) == {
        "type": "object"
    }
    assert (operation_directory / "invocation.json").is_file()


def test_concurrent_identical_operation_invokes_cli_once(tmp_path, monkeypatch):
    install_executable(
        tmp_path,
        "codex",
        """#!/usr/bin/env python3
import json
import pathlib
import sys
import time
counter = pathlib.Path("invocations.txt")
counter.write_text(counter.read_text() + "x" if counter.exists() else "x")
time.sleep(0.2)
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"one-call"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":6,"output_tokens":3}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    call_runner = runner(tmp_path)
    call_request = request(tmp_path, "codex")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(call_runner.run, call_request, {"type": "object"})
            for _ in range(2)
        )
        results = tuple(future.result() for future in futures)

    assert results[0] == results[1]
    assert (tmp_path / "invocations.txt").read_text() == "x"


def test_concurrent_conflicting_operation_allows_one_identity(tmp_path, monkeypatch):
    install_executable(
        tmp_path,
        "codex",
        """#!/usr/bin/env python3
import json
import pathlib
import sys
import time
counter = pathlib.Path("invocations.txt")
counter.write_text(counter.read_text() + "x" if counter.exists() else "x")
time.sleep(0.2)
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"winner"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":6,"output_tokens":3}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    call_runner = runner(tmp_path)
    first_request = request(tmp_path, "codex")
    second_request = replace(first_request, model="conflicting-model")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(call_runner.run, first_request, {"type": "object"}),
            pool.submit(call_runner.run, second_request, {"type": "object"}),
        )
        outcomes = tuple(
            future.result() if future.exception() is None else future.exception()
            for future in futures
        )

    assert sum(isinstance(value, CodingAgentCallResult) for value in outcomes) == 1
    errors = tuple(
        value for value in outcomes if isinstance(value, CodingAgentInvocationError)
    )
    assert len(errors) == 1
    assert "identity was reused" in str(errors[0])
    assert (tmp_path / "invocations.txt").read_text() == "x"


def test_operation_directory_symlink_is_rejected_without_touching_target(
    tmp_path,
    monkeypatch,
):
    install_executable(tmp_path, "codex", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (artifact_root / request(tmp_path, "codex").operation_id).symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(CodingAgentInvocationError, match="must be a directory"):
        runner(tmp_path).run(request(tmp_path, "codex"), {"type": "object"})

    assert tuple(outside.iterdir()) == ()


def test_symlinked_identity_artifact_is_rejected(tmp_path, monkeypatch):
    install_executable(tmp_path, "codex", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    artifact_root = tmp_path / "artifacts"
    operation_directory = artifact_root / request(tmp_path, "codex").operation_id
    operation_directory.mkdir(parents=True)
    outside = tmp_path / "outside-prompt.txt"
    outside.write_text(request(tmp_path, "codex").prompt)
    (operation_directory / "prompt.txt").symlink_to(outside)

    with pytest.raises(CodingAgentInvocationError, match="regular file"):
        runner(tmp_path).run(request(tmp_path, "codex"), {"type": "object"})

    assert outside.read_text() == request(tmp_path, "codex").prompt


def test_artifact_root_rejects_symlinked_parent_without_creating_target_child(
    tmp_path,
    monkeypatch,
):
    install_executable(tmp_path, "codex", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    outside = tmp_path / "outside-root"
    outside.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(outside, target_is_directory=True)
    call_runner = SubprocessCodingAgentCallRunner(
        CodingAgentRunnerSettings(
            artifact_root=str(linked_root / "agent-calls"),
            termination_grace_seconds=1,
        )
    )

    with pytest.raises(CodingAgentInvocationError, match="must not traverse"):
        call_runner.run(request(tmp_path, "codex"), {"type": "object"})

    assert not (outside / "agent-calls").exists()
