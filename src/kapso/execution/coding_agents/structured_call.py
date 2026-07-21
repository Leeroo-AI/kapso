"""Durable, fail-loud boundary for structured read-only coding-agent calls."""

import fcntl
import json
import math
import os
import re
import shutil
import stat
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

_OPERATION_IDENTIFIER_PATTERN = re.compile(r"^agent_call_[0-9a-f]{32}$")
_INPUT_FILENAMES = (
    "prompt.txt",
    "response_schema.json",
    "invocation.json",
)
_OUTPUT_FILENAMES = (
    "stdout.txt",
    "stderr.txt",
    "final.json",
)
_RESULT_FILENAME = "result.json"
_ARTIFACT_FILENAMES = _INPUT_FILENAMES + _OUTPUT_FILENAMES + (_RESULT_FILENAME,)


class CodingAgentInvocationError(RuntimeError):
    """A coding-agent operation is corrupt, conflicting, or unsuccessful."""


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, name)


def _require_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_nonnegative_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _require_exact_fields(
    payload: Any,
    expected: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = tuple(sorted(expected - set(payload)))
    unknown = tuple(sorted(set(payload) - expected))
    if missing or unknown:
        raise ValueError(
            f"{name} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return payload


def _require_unique_strings(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    strings = tuple(_require_nonempty_string(value, name) for value in values)
    if len(strings) != len(set(strings)):
        raise ValueError(f"{name} must not contain duplicates")
    return strings


@dataclass(frozen=True)
class CodingAgentCallRequest:
    """Complete immutable input to one structured coding-agent operation."""

    operation_id: str
    role: str
    cli: str
    model: str
    prompt: str
    workspace: str
    timeout_seconds: float
    effort: str | None = None
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.operation_id, str)
            or _OPERATION_IDENTIFIER_PATTERN.fullmatch(self.operation_id) is None
        ):
            raise ValueError(
                "coding-agent operation id must be agent_call_<32 lowercase hex>"
            )
        _require_nonempty_string(self.role, "coding-agent role")
        if self.cli not in {"codex", "claude_code"}:
            raise ValueError("coding-agent cli must be codex or claude_code")
        _require_nonempty_string(self.model, "coding-agent model")
        _require_nonempty_string(self.prompt, "coding-agent prompt")
        _require_nonempty_string(self.workspace, "coding-agent workspace")
        timeout = _require_nonnegative_number(
            self.timeout_seconds,
            "coding-agent timeout",
        )
        if timeout == 0:
            raise ValueError("coding-agent timeout must be greater than zero")
        _require_optional_string(self.effort, "coding-agent effort")
        object.__setattr__(
            self,
            "allowed_tools",
            _require_unique_strings(
                self.allowed_tools,
                "coding-agent allowed tools",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "role": self.role,
            "cli": self.cli,
            "model": self.model,
            "prompt": self.prompt,
            "workspace": self.workspace,
            "timeout_seconds": self.timeout_seconds,
            "effort": self.effort,
            "allowed_tools": list(self.allowed_tools),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodingAgentCallRequest":
        values = _require_exact_fields(
            payload,
            {
                "operation_id",
                "role",
                "cli",
                "model",
                "prompt",
                "workspace",
                "timeout_seconds",
                "effort",
                "allowed_tools",
            },
            "coding-agent request",
        )
        return cls(
            operation_id=values["operation_id"],
            role=values["role"],
            cli=values["cli"],
            model=values["model"],
            prompt=values["prompt"],
            workspace=values["workspace"],
            timeout_seconds=values["timeout_seconds"],
            effort=values["effort"],
            allowed_tools=values["allowed_tools"],
        )


@dataclass(frozen=True)
class CodingAgentCallResult:
    """Complete structured result and durable local artifact references."""

    output: str
    duration_seconds: float
    cost_usd: float | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise ValueError("coding-agent output must be a string")
        _require_nonnegative_number(
            self.duration_seconds,
            "coding-agent duration",
        )
        if self.cost_usd is not None:
            _require_nonnegative_number(self.cost_usd, "coding-agent cost")
        if self.input_tokens is not None:
            _require_nonnegative_integer(
                self.input_tokens,
                "coding-agent input tokens",
            )
        if self.output_tokens is not None:
            _require_nonnegative_integer(
                self.output_tokens,
                "coding-agent output tokens",
            )
        object.__setattr__(
            self,
            "artifacts",
            _require_unique_strings(self.artifacts, "coding-agent artifacts"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "artifacts": list(self.artifacts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodingAgentCallResult":
        values = _require_exact_fields(
            payload,
            {
                "output",
                "duration_seconds",
                "cost_usd",
                "input_tokens",
                "output_tokens",
                "artifacts",
            },
            "coding-agent result",
        )
        return cls(
            output=values["output"],
            duration_seconds=values["duration_seconds"],
            cost_usd=values["cost_usd"],
            input_tokens=values["input_tokens"],
            output_tokens=values["output_tokens"],
            artifacts=values["artifacts"],
        )


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
        artifact_root = Path(self.artifact_root)
        if not artifact_root.is_absolute():
            raise ValueError("coding-agent artifact root must be absolute")
        if str(artifact_root) != self.artifact_root or ".." in artifact_root.parts:
            raise ValueError("coding-agent artifact root must be normalized")
        if (
            isinstance(self.termination_grace_seconds, bool)
            or not isinstance(self.termination_grace_seconds, (int, float))
            or not math.isfinite(float(self.termination_grace_seconds))
            or self.termination_grace_seconds <= 0
        ):
            raise ValueError("coding-agent termination grace must be positive")


class SubprocessCodingAgentCallRunner:
    """Invoke Codex or Claude Code through a locked immutable operation identity."""

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
        schema_text = (
            json.dumps(response_schema, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        invocation_text = (
            json.dumps(
                {
                    "role": request.role,
                    "cli": request.cli,
                    "model": request.model,
                    "timeout_seconds": request.timeout_seconds,
                    "effort": request.effort,
                    "allowed_tools": list(request.allowed_tools),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        artifact_root = self._prepare_artifact_root()
        with ExitStack() as descriptors:
            root_descriptor = os.open(
                artifact_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            descriptors.callback(os.close, root_descriptor)
            lock_descriptor = os.open(
                request.operation_id + ".lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=root_descriptor,
            )
            lock_status = os.fstat(lock_descriptor)
            if not stat.S_ISREG(lock_status.st_mode):
                os.close(lock_descriptor)
                raise CodingAgentInvocationError(
                    "coding-agent operation lock must be a regular file"
                )
            lock_handle = os.fdopen(lock_descriptor, "r+b")
            descriptors.enter_context(lock_handle)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            operation_descriptor = self._open_operation_directory(
                root_descriptor,
                request.operation_id,
            )
            descriptors.callback(os.close, operation_descriptor)
            artifact_directory = artifact_root / request.operation_id
            return self._run_locked(
                request=request,
                schema_text=schema_text,
                invocation_text=invocation_text,
                operation_descriptor=operation_descriptor,
                artifact_directory=artifact_directory,
            )

    def _prepare_artifact_root(self) -> Path:
        artifact_root = Path(self.settings.artifact_root)
        self._validate_artifact_root_components(
            artifact_root,
            require_complete=False,
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._validate_artifact_root_components(
            artifact_root,
            require_complete=True,
        )
        return artifact_root

    @staticmethod
    def _validate_artifact_root_components(
        artifact_root: Path,
        *,
        require_complete: bool,
    ) -> None:
        with ExitStack() as descriptors:
            current_descriptor = os.open(
                artifact_root.anchor,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            descriptors.callback(os.close, current_descriptor)
            for component in artifact_root.parts[1:]:
                entries = set(os.listdir(current_descriptor))
                if component not in entries:
                    if require_complete:
                        raise CodingAgentInvocationError(
                            "coding-agent artifact root creation is incomplete"
                        )
                    return
                status = os.stat(
                    component,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(status.st_mode):
                    raise CodingAgentInvocationError(
                        "coding-agent artifact root must not traverse symlinks"
                    )
                current_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current_descriptor,
                )
                descriptors.callback(os.close, current_descriptor)

    @staticmethod
    def _open_operation_directory(
        root_descriptor: int,
        operation_id: str,
    ) -> int:
        root_entries = set(os.listdir(root_descriptor))
        if operation_id in root_entries:
            status = os.stat(
                operation_id,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(status.st_mode):
                raise CodingAgentInvocationError(
                    "coding-agent operation path must be a directory"
                )
        else:
            os.mkdir(operation_id, mode=0o700, dir_fd=root_descriptor)
            os.fsync(root_descriptor)
        return os.open(
            operation_id,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_descriptor,
        )

    def _run_locked(
        self,
        *,
        request: CodingAgentCallRequest,
        schema_text: str,
        invocation_text: str,
        operation_descriptor: int,
        artifact_directory: Path,
    ) -> CodingAgentCallResult:
        self._validate_and_recover_operation_directory(operation_descriptor)
        entries = set(os.listdir(operation_descriptor))
        result_exists = _RESULT_FILENAME in entries
        expected_inputs = {
            "prompt.txt": request.prompt,
            "response_schema.json": schema_text,
            "invocation.json": invocation_text,
        }
        for filename, expected_text in expected_inputs.items():
            if filename in entries:
                actual_text = self._read_regular_text(
                    operation_descriptor,
                    filename,
                )
                if actual_text != expected_text:
                    if result_exists:
                        raise CodingAgentInvocationError(
                            "coding-agent operation identity was reused with new input"
                        )
                    raise CodingAgentInvocationError(
                        f"coding-agent operation {filename} changed before retry"
                    )
            elif result_exists:
                raise CodingAgentInvocationError(
                    "completed coding-agent operation is missing identity input"
                )
            else:
                self._write_atomic_text(
                    operation_descriptor,
                    filename,
                    expected_text,
                )
        if result_exists:
            return self._read_cached_result(
                operation_descriptor,
                artifact_directory,
            )
        for filename in _OUTPUT_FILENAMES:
            if filename in set(os.listdir(operation_descriptor)):
                self._remove_regular_file(operation_descriptor, filename)
        schema_path = artifact_directory / "response_schema.json"
        final_path = artifact_directory / "final.json"
        command = self._command(request, schema_text, schema_path, final_path)
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=Path(request.workspace),
            input=request.prompt,
            text=True,
            capture_output=True,
            check=False,
        )
        duration = time.monotonic() - started
        self._write_atomic_text(
            operation_descriptor,
            "stdout.txt",
            completed.stdout,
        )
        self._write_atomic_text(
            operation_descriptor,
            "stderr.txt",
            completed.stderr,
        )
        if completed.returncode != 0:
            raise CodingAgentInvocationError(
                f"{request.cli} exited with status {completed.returncode}; "
                f"artifacts: {artifact_directory}"
            )
        if request.cli == "codex":
            output, input_tokens, output_tokens = self._parse_codex(
                completed.stdout,
                operation_descriptor,
            )
            cost_usd = None
        else:
            output, input_tokens, output_tokens, cost_usd = self._parse_claude(
                completed.stdout,
            )
            self._write_atomic_text(
                operation_descriptor,
                "final.json",
                output + "\n",
            )
        artifacts = self._artifact_paths(artifact_directory)
        result = CodingAgentCallResult(
            output=output,
            duration_seconds=duration,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            artifacts=artifacts,
        )
        self._write_atomic_text(
            operation_descriptor,
            _RESULT_FILENAME,
            json.dumps(result.to_dict(), sort_keys=True, allow_nan=False) + "\n",
        )
        return result

    @staticmethod
    def _validate_and_recover_operation_directory(descriptor: int) -> None:
        allowed = set(_ARTIFACT_FILENAMES)
        temporary = {f".{filename}.tmp" for filename in _ARTIFACT_FILENAMES}
        entries = set(os.listdir(descriptor))
        unknown = tuple(sorted(entries - allowed - temporary))
        if unknown:
            raise CodingAgentInvocationError(
                f"coding-agent operation directory has unknown entries: {unknown}"
            )
        for filename in sorted(entries & temporary):
            SubprocessCodingAgentCallRunner._remove_regular_file(
                descriptor,
                filename,
            )

    @staticmethod
    def _require_regular_file(descriptor: int, filename: str) -> None:
        status = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(status.st_mode):
            raise CodingAgentInvocationError(
                f"coding-agent artifact must be a regular file: {filename}"
            )

    @staticmethod
    def _read_regular_text(descriptor: int, filename: str) -> str:
        SubprocessCodingAgentCallRunner._require_regular_file(descriptor, filename)
        file_descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        with os.fdopen(file_descriptor, "r", encoding="utf-8") as handle:
            text = handle.read()
            os.fsync(handle.fileno())
            return text

    @staticmethod
    def _remove_regular_file(descriptor: int, filename: str) -> None:
        SubprocessCodingAgentCallRunner._require_regular_file(descriptor, filename)
        os.unlink(filename, dir_fd=descriptor)
        os.fsync(descriptor)

    @staticmethod
    def _write_atomic_text(descriptor: int, filename: str, text: str) -> None:
        temporary_name = f".{filename}.tmp"
        entries = set(os.listdir(descriptor))
        if temporary_name in entries:
            SubprocessCodingAgentCallRunner._remove_regular_file(
                descriptor,
                temporary_name,
            )
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        with os.fdopen(temporary_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )
        os.fsync(descriptor)

    def _read_cached_result(
        self,
        operation_descriptor: int,
        artifact_directory: Path,
    ) -> CodingAgentCallResult:
        for filename in _INPUT_FILENAMES + _OUTPUT_FILENAMES + (_RESULT_FILENAME,):
            self._require_regular_file(operation_descriptor, filename)
        result = CodingAgentCallResult.from_dict(
            json.loads(self._read_regular_text(operation_descriptor, _RESULT_FILENAME))
        )
        if result.artifacts != self._artifact_paths(artifact_directory):
            raise CodingAgentInvocationError(
                "cached coding-agent artifact references are invalid"
            )
        return result

    @staticmethod
    def _artifact_paths(artifact_directory: Path) -> tuple[str, ...]:
        return tuple(
            str(artifact_directory / filename)
            for filename in _INPUT_FILENAMES + _OUTPUT_FILENAMES
        )

    def _command(
        self,
        request: CodingAgentCallRequest,
        schema_text: str,
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
                    "--skip-git-repo-check",
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
            json.loads(schema_text),
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
        operation_descriptor: int,
    ) -> tuple[str, int | None, int | None]:
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
        entries = set(os.listdir(operation_descriptor))
        if "final.json" not in entries:
            raise CodingAgentInvocationError(
                "Codex returned no final structured output"
            )
        output = SubprocessCodingAgentCallRunner._read_regular_text(
            operation_descriptor,
            "final.json",
        )
        if not output.strip():
            raise CodingAgentInvocationError(
                "Codex returned no final structured output"
            )
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
    ) -> tuple[str, int | None, int | None, float | None]:
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
