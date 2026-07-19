"""Fail-loud read-only coding-agent CLI boundary for ideation."""

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol, Tuple

from kapso.execution.search_strategies.generic.ideation.types import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    new_identifier,
)


class CodingAgentInvocationError(RuntimeError):
    """A coding-agent subprocess did not produce a successful result."""


class CodingAgentCallRunner(Protocol):
    def run(
        self,
        request: CodingAgentCallRequest,
        response_schema: Mapping[str, Any],
    ) -> CodingAgentCallResult:
        """Run one complete, structured, read-only agent invocation."""


@dataclass(frozen=True)
class CodingAgentRunnerSettings:
    artifact_root: str
    termination_grace_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_root, str) or not self.artifact_root.strip():
            raise ValueError("coding-agent artifact root must be non-empty")
        if not Path(self.artifact_root).is_absolute():
            raise ValueError("coding-agent artifact root must be absolute")
        if (
            isinstance(self.termination_grace_seconds, bool)
            or not isinstance(self.termination_grace_seconds, (int, float))
            or not math.isfinite(float(self.termination_grace_seconds))
            or self.termination_grace_seconds <= 0
        ):
            raise ValueError("coding-agent termination grace must be positive")


class SubprocessCodingAgentCallRunner:
    """Invoke Codex or Claude Code with schema-constrained final output."""

    def __init__(self, settings: CodingAgentRunnerSettings):
        self.settings = settings

    def run(
        self,
        request: CodingAgentCallRequest,
        response_schema: Mapping[str, Any],
    ) -> CodingAgentCallResult:
        workspace = Path(request.workspace)
        if not workspace.is_dir():
            raise ValueError("coding-agent workspace must be an existing directory")
        if shutil.which("timeout") is None:
            raise RuntimeError("GNU timeout is required for coding-agent deadlines")
        executable = "codex" if request.cli == "codex" else "claude"
        if shutil.which(executable) is None:
            raise RuntimeError(f"coding-agent CLI is not installed: {executable}")
        if not isinstance(response_schema, Mapping):
            raise ValueError("coding-agent response schema must be an object")
        supported_tools = (
            {"Read", "WebSearch"}
            if request.cli == "codex"
            else {"Read", "Glob", "Grep", "WebSearch"}
        )
        if not set(request.allowed_tools).issubset(supported_tools):
            raise ValueError("coding-agent request contains an unsupported tool")
        call_id = new_identifier("agent_call")
        artifact_directory = Path(self.settings.artifact_root) / call_id
        artifact_directory.mkdir(parents=True, exist_ok=False)
        prompt_path = artifact_directory / "prompt.txt"
        schema_path = artifact_directory / "response_schema.json"
        stdout_path = artifact_directory / "stdout.txt"
        stderr_path = artifact_directory / "stderr.txt"
        final_path = artifact_directory / "final.json"
        prompt_path.write_text(request.prompt, encoding="utf-8")
        schema_path.write_text(
            json.dumps(response_schema, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        command = self._command(request, schema_path, final_path)
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=workspace,
            input=request.prompt,
            text=True,
            capture_output=True,
            check=False,
        )
        duration = time.monotonic() - started
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise CodingAgentInvocationError(
                f"{request.cli} exited with status {completed.returncode}; "
                f"artifacts: {artifact_directory}"
            )
        if request.cli == "codex":
            output, input_tokens, output_tokens = self._parse_codex(
                completed.stdout,
                final_path,
            )
            cost_usd = None
        else:
            output, input_tokens, output_tokens, cost_usd = self._parse_claude(
                completed.stdout,
                final_path,
            )
        artifacts = tuple(
            str(path)
            for path in (
                prompt_path,
                schema_path,
                stdout_path,
                stderr_path,
                final_path,
            )
        )
        return CodingAgentCallResult(
            output=output,
            duration_seconds=duration,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            artifacts=artifacts,
        )

    def _command(
        self,
        request: CodingAgentCallRequest,
        schema_path: Path,
        final_path: Path,
    ) -> list[str]:
        deadline = f"{request.timeout_seconds}s"
        grace = f"{self.settings.termination_grace_seconds}s"
        prefix = [
            "timeout",
            "--signal=TERM",
            f"--kill-after={grace}",
            deadline,
            "env",
            "-u",
            "OPENAI_API_KEY",
        ]
        if request.cli == "codex":
            command = prefix + ["codex"]
            if "WebSearch" in request.allowed_tools:
                command.append("--search")
            command.extend(
                [
                    "--ask-for-approval",
                    "never",
                    "exec",
                    "--sandbox",
                    "read-only",
                    "--ephemeral",
                    "--ignore-user-config",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(final_path),
                    "--json",
                    "--color",
                    "never",
                    "--model",
                    request.model,
                ]
            )
            if request.effort is not None:
                command.extend(
                    ["--config", f'model_reasoning_effort="{request.effort}"']
                )
            command.append("-")
            return command
        schema = json.dumps(
            json.loads(schema_path.read_text(encoding="utf-8")),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        command = prefix + [
            "claude",
            "--print",
            "--permission-mode",
            "plan",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--model",
            request.model,
        ]
        if request.effort is not None:
            command.extend(["--effort", request.effort])
        command.extend(["--tools", ",".join(request.allowed_tools)])
        return command

    @staticmethod
    def _parse_codex(
        stdout: str,
        final_path: Path,
    ) -> Tuple[str, int | None, int | None]:
        if not stdout.strip():
            raise CodingAgentInvocationError("Codex returned an empty event stream")
        lines = stdout.splitlines()
        if any(not line.strip() for line in lines):
            raise CodingAgentInvocationError("Codex returned a blank JSONL event")
        events = tuple(json.loads(line) for line in lines)
        failures = tuple(
            event for event in events if event.get("type") in {"turn.failed", "error"}
        )
        if failures:
            raise CodingAgentInvocationError("Codex event stream contains a failure")
        completions = tuple(
            event for event in events if event.get("type") == "turn.completed"
        )
        if len(completions) != 1:
            raise CodingAgentInvocationError(
                "Codex event stream requires one completed turn"
            )
        if (
            not final_path.is_file()
            or not final_path.read_text(encoding="utf-8").strip()
        ):
            raise CodingAgentInvocationError(
                "Codex returned no final structured output"
            )
        output = final_path.read_text(encoding="utf-8")
        json.loads(output)
        usage = completions[0].get("usage")
        if not isinstance(usage, dict):
            raise CodingAgentInvocationError("Codex completion is missing usage")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        for value, name in (
            (input_tokens, "input tokens"),
            (output_tokens, "output tokens"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CodingAgentInvocationError(f"Codex {name} are invalid")
        return output, input_tokens, output_tokens

    @staticmethod
    def _parse_claude(
        stdout: str,
        final_path: Path,
    ) -> Tuple[str, int | None, int | None, float | None]:
        if not stdout.strip():
            raise CodingAgentInvocationError("Claude Code returned empty output")
        envelope = json.loads(stdout)
        if not isinstance(envelope, dict):
            raise CodingAgentInvocationError("Claude Code output must be an object")
        if envelope.get("is_error") is not False:
            raise CodingAgentInvocationError("Claude Code reported an error result")
        structured = envelope.get("structured_output")
        if not isinstance(structured, dict):
            raise CodingAgentInvocationError(
                "Claude Code returned no structured output"
            )
        output = json.dumps(
            structured,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        final_path.write_text(output + "\n", encoding="utf-8")
        usage = envelope.get("usage")
        if not isinstance(usage, dict):
            raise CodingAgentInvocationError("Claude Code result is missing usage")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        for value, name in (
            (input_tokens, "input tokens"),
            (output_tokens, "output tokens"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CodingAgentInvocationError(f"Claude Code {name} are invalid")
        cost_usd = envelope.get("total_cost_usd")
        if cost_usd is not None and (
            isinstance(cost_usd, bool)
            or not isinstance(cost_usd, (int, float))
            or not math.isfinite(float(cost_usd))
            or cost_usd < 0
        ):
            raise CodingAgentInvocationError("Claude Code cost is invalid")
        return output, input_tokens, output_tokens, cost_usd
