"""Durable, fail-loud boundary for structured read-only coding-agent calls."""

import fcntl
import json
import math
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.knowledge.access import (
    PriorKnowledgeAccess,
    PriorKnowledgeAccessMaterialization,
)
from kapso.cross_run.agent_artifacts import (
    CODING_AGENT_ARTIFACT_FILENAMES as _ARTIFACT_FILENAMES,
    CODING_AGENT_INPUT_ARTIFACT_FILENAMES as _INPUT_FILENAMES,
    CODING_AGENT_OUTPUT_ARTIFACT_FILENAMES as _OUTPUT_FILENAMES,
    CODING_AGENT_RESULT_FILENAME as _RESULT_FILENAME,
)
from kapso.execution.coding_agents.credential_environment import (
    coding_agent_credential_environment,
)

_OPERATION_IDENTIFIER_PATTERN = re.compile(r"^agent_call_[0-9a-f]{32}$")
_EMPTY_MCP_AUDIT_DIGEST = tree_or_blob_digest(b"")
_CREDENTIAL_ENVIRONMENT_POLICY_VERSION = "kapso.coding_agent_credentials.v1"
_FILESYSTEM_POLICY_VERSION = "kapso.coding_agent_read.v1"
_MCP_AUDIT_POLICY_VERSION = "kapso.mcp_audit.v1"
_SENSITIVE_HOME_PATHS = (
    "~/.aws",
    "~/.azure",
    "~/.codex",
    "~/.config/gh",
    "~/.config/gcloud",
    "~/.docker",
    "~/.git-credentials",
    "~/.kube",
    "~/.netrc",
    "~/.ssh",
)


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


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise CodingAgentInvocationError(
                "prior-knowledge MCP audit contains a duplicate JSON key"
            )
        payload[key] = value
    return payload


def _coding_agent_workspace_path(value: str) -> Path:
    workspace_text = _require_nonempty_string(value, "coding-agent workspace")
    workspace = Path(workspace_text)
    if not workspace.is_absolute():
        raise ValueError("coding-agent workspace must be absolute")
    if str(workspace) != workspace_text or ".." in workspace.parts:
        raise ValueError("coding-agent workspace must be normalized")
    return workspace


def _validate_coding_agent_workspace(value: str) -> Path:
    workspace = _coding_agent_workspace_path(value)
    if not workspace.is_dir():
        raise ValueError("coding-agent workspace must be an existing directory")
    resolved_workspace = workspace.resolve(strict=True)
    if workspace != resolved_workspace:
        raise ValueError("coding-agent workspace must not traverse symlinks")
    user_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    forbidden_broad_roots = {user_home, *user_home.parents}
    if resolved_workspace in forbidden_broad_roots:
        raise ValueError("coding-agent workspace is broader than an allowed project")
    return workspace


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
    prior_knowledge: PriorKnowledgeAccessMaterialization | None = None

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
        _coding_agent_workspace_path(self.workspace)
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
        if self.prior_knowledge is not None and not isinstance(
            self.prior_knowledge,
            PriorKnowledgeAccessMaterialization,
        ):
            raise ValueError("coding-agent prior knowledge is invalid")

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
            "prior_knowledge": (
                None if self.prior_knowledge is None else self.prior_knowledge.to_dict()
            ),
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
                "prior_knowledge",
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
            prior_knowledge=(
                None
                if values["prior_knowledge"] is None
                else PriorKnowledgeAccessMaterialization.from_dict(
                    values["prior_knowledge"]
                )
            ),
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
    mcp_audit_digest: str = _EMPTY_MCP_AUDIT_DIGEST
    mcp_audit_event_count: int = 0

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
        if (
            not isinstance(self.mcp_audit_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.mcp_audit_digest) is None
        ):
            raise ValueError("coding-agent MCP audit digest is invalid")
        _require_nonnegative_integer(
            self.mcp_audit_event_count,
            "coding-agent MCP audit event count",
        )
        if (
            self.mcp_audit_event_count == 0
            and self.mcp_audit_digest != _EMPTY_MCP_AUDIT_DIGEST
        ):
            raise ValueError("empty coding-agent MCP audit has a non-empty digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "artifacts": list(self.artifacts),
            "mcp_audit_digest": self.mcp_audit_digest,
            "mcp_audit_event_count": self.mcp_audit_event_count,
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
                "mcp_audit_digest",
                "mcp_audit_event_count",
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
            mcp_audit_digest=values["mcp_audit_digest"],
            mcp_audit_event_count=values["mcp_audit_event_count"],
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
    sensitive_file_glob_scan_max_depth: int

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
        if (
            isinstance(self.sensitive_file_glob_scan_max_depth, bool)
            or not isinstance(self.sensitive_file_glob_scan_max_depth, int)
            or self.sensitive_file_glob_scan_max_depth <= 0
        ):
            raise ValueError(
                "coding-agent sensitive-file glob scan depth must be positive"
            )


class SubprocessCodingAgentCallRunner:
    """Invoke Codex or Claude Code through a locked immutable operation identity."""

    def __init__(self, settings: CodingAgentRunnerSettings):
        self.settings = settings

    def run(
        self,
        request: CodingAgentCallRequest,
        response_schema: Mapping[str, Any],
    ) -> CodingAgentCallResult:
        workspace = _validate_coding_agent_workspace(request.workspace)
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
                    "credential_environment_policy_version": (
                        _CREDENTIAL_ENVIRONMENT_POLICY_VERSION
                    ),
                    "filesystem_policy_version": _FILESYSTEM_POLICY_VERSION,
                    "mcp_audit_policy_version": _MCP_AUDIT_POLICY_VERSION,
                    "sensitive_file_glob_scan_max_depth": (
                        self.settings.sensitive_file_glob_scan_max_depth
                    ),
                    "prior_knowledge_snapshot_id": (
                        None
                        if request.prior_knowledge is None
                        else request.prior_knowledge.prior_knowledge_snapshot.prior_knowledge_snapshot_id
                    ),
                    "prior_knowledge_materialization_digest": (
                        None
                        if request.prior_knowledge is None
                        else request.prior_knowledge.materialization_digest
                    ),
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
            "prior_knowledge.json": self._prior_knowledge_text(request),
            "mcp_config.json": self._mcp_config_text(
                request,
                artifact_directory,
            ),
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
                request,
            )
        for filename in _OUTPUT_FILENAMES:
            if filename in set(os.listdir(operation_descriptor)):
                self._remove_regular_file(operation_descriptor, filename)
        self._write_atomic_text(
            operation_descriptor,
            "mcp_audit.jsonl",
            "",
        )
        schema_path = artifact_directory / "response_schema.json"
        final_path = artifact_directory / "final.json"
        mcp_config_path = artifact_directory / "mcp_config.json"
        command = self._command(
            request,
            schema_text,
            schema_path,
            final_path,
            mcp_config_path,
        )
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=Path(request.workspace),
            env=coding_agent_credential_environment(request.cli),
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
        audit_event_count, audit_digest = self._validate_mcp_audit(
            request,
            self._read_regular_text(operation_descriptor, "mcp_audit.jsonl"),
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
            mcp_audit_digest=audit_digest,
            mcp_audit_event_count=audit_event_count,
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
        request: CodingAgentCallRequest,
    ) -> CodingAgentCallResult:
        for filename in _INPUT_FILENAMES + _OUTPUT_FILENAMES + (_RESULT_FILENAME,):
            self._require_regular_file(operation_descriptor, filename)
        result = CodingAgentCallResult.from_dict(
            json.loads(self._read_regular_text(operation_descriptor, _RESULT_FILENAME))
        )
        audit_event_count, audit_digest = self._validate_mcp_audit(
            request,
            self._read_regular_text(operation_descriptor, "mcp_audit.jsonl"),
        )
        if (
            result.mcp_audit_digest != audit_digest
            or result.mcp_audit_event_count != audit_event_count
        ):
            raise CodingAgentInvocationError(
                "cached coding-agent MCP audit conflicts with completed result"
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

    @staticmethod
    def _prior_knowledge_text(request: CodingAgentCallRequest) -> str:
        if request.prior_knowledge is None:
            return "null\n"
        return request.prior_knowledge.to_json_bytes().decode("utf-8")

    @staticmethod
    def _mcp_server_config(
        request: CodingAgentCallRequest,
        artifact_directory: Path,
    ) -> dict[str, Any]:
        if request.prior_knowledge is None:
            raise ValueError("prior-knowledge MCP requires a materialization")
        python_path = Path(__file__).resolve().parents[3]
        if not (python_path / "kapso" / "gated_mcp").is_dir():
            raise ValueError("Kapso package root is missing for prior-knowledge MCP")
        materialization_path = artifact_directory / "prior_knowledge.json"
        audit_path = artifact_directory / "mcp_audit.jsonl"
        maximum_bytes = len(request.prior_knowledge.to_json_bytes())
        environment_executable = shutil.which("env")
        if environment_executable is None:
            raise ValueError("env executable is required for MCP isolation")
        return {
            "command": environment_executable,
            "args": [
                "-i",
                f"PYTHONPATH={python_path}",
                str(Path(sys.executable).resolve()),
                "-m",
                "kapso.gated_mcp.server",
                "--enabled-gates",
                "prior_knowledge",
                "--gate-failure-policy",
                "error",
                "--prior-knowledge-path",
                str(materialization_path),
                "--prior-knowledge-maximum-bytes",
                str(maximum_bytes),
                "--prior-knowledge-audit-path",
                str(audit_path),
                "--operation-id",
                request.operation_id,
            ],
        }

    def _mcp_config_text(
        self,
        request: CodingAgentCallRequest,
        artifact_directory: Path,
    ) -> str:
        servers = {}
        if request.prior_knowledge is not None:
            servers["prior_knowledge"] = self._mcp_server_config(
                request,
                artifact_directory,
            )
        return (
            json.dumps(
                {"mcpServers": servers},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

    @staticmethod
    def _validate_mcp_audit(
        request: CodingAgentCallRequest,
        audit_text: str,
    ) -> tuple[int, str]:
        if not audit_text:
            return 0, _EMPTY_MCP_AUDIT_DIGEST
        if request.prior_knowledge is None:
            raise CodingAgentInvocationError(
                "coding-agent call without prior knowledge produced an MCP audit"
            )
        if not audit_text.endswith("\n"):
            raise CodingAgentInvocationError(
                "prior-knowledge MCP audit has an incomplete final event"
            )
        lines = audit_text.splitlines()
        if any(not line.strip() for line in lines):
            raise CodingAgentInvocationError(
                "prior-knowledge MCP audit has a blank line"
            )
        access = PriorKnowledgeAccess(request.prior_knowledge)
        packet = access.packet
        member_ids = set(packet.selected_record_ids) | set(packet.proof_reference_ids)
        expected_fields = {
            "arguments",
            "operation_id",
            "prior_knowledge_snapshot_id",
            "response_digest",
            "returned_ids",
            "tool_name",
        }
        allowed_tools = {
            "list_prior_knowledge",
            "get_prior_knowledge_record",
        }
        for line in lines:
            event = json.loads(line, object_pairs_hook=_strict_json_object)
            if not isinstance(event, dict) or set(event) != expected_fields:
                raise CodingAgentInvocationError(
                    "prior-knowledge MCP audit fields are invalid"
                )
            if event["operation_id"] != request.operation_id:
                raise CodingAgentInvocationError(
                    "prior-knowledge MCP audit operation identity changed"
                )
            if (
                event["prior_knowledge_snapshot_id"]
                != packet.prior_knowledge_snapshot_id
            ):
                raise CodingAgentInvocationError(
                    "prior-knowledge MCP audit packet identity changed"
                )
            if event["tool_name"] not in allowed_tools:
                raise CodingAgentInvocationError(
                    "prior-knowledge MCP audit names an unknown tool"
                )
            arguments = event["arguments"]
            if not isinstance(arguments, dict):
                raise CodingAgentInvocationError(
                    "prior-knowledge MCP audit arguments are invalid"
                )
            returned_ids = event["returned_ids"]
            if (
                not isinstance(returned_ids, list)
                or len(returned_ids) != len(set(returned_ids))
                or not set(returned_ids).issubset(member_ids)
            ):
                raise CodingAgentInvocationError(
                    "prior-knowledge MCP audit returned IDs are invalid"
                )
            if event["tool_name"] == "list_prior_knowledge":
                if arguments or returned_ids != sorted(member_ids):
                    raise CodingAgentInvocationError(
                        "prior-knowledge list audit is inconsistent"
                    )
                response_payload = access.list_response_payload()
            else:
                if set(arguments) != {"record_id"}:
                    raise CodingAgentInvocationError(
                        "prior-knowledge get audit arguments are invalid"
                    )
                record_id = arguments["record_id"]
                if record_id not in member_ids or returned_ids != [record_id]:
                    raise CodingAgentInvocationError(
                        "prior-knowledge get audit is inconsistent"
                    )
                response_payload = access.record_response_payload(record_id)
            expected_response_digest = tree_or_blob_digest(
                canonical_json_bytes(response_payload)
            )
            if event["response_digest"] != expected_response_digest:
                raise CodingAgentInvocationError(
                    "prior-knowledge MCP audit response digest is inconsistent"
                )
            if canonical_json_bytes(event).decode("utf-8") != line:
                raise CodingAgentInvocationError(
                    "prior-knowledge MCP audit event is not canonical JSON"
                )
        return len(lines), tree_or_blob_digest(audit_text.encode("utf-8"))

    def _command(
        self,
        request: CodingAgentCallRequest,
        schema_text: str,
        schema_path: Path,
        final_path: Path,
        mcp_config_path: Path,
    ) -> list[str]:
        deadline = f"{request.timeout_seconds}s"
        grace = f"{self.settings.termination_grace_seconds}s"
        prefix = [
            "timeout",
            "--signal=TERM",
            f"--kill-after={grace}",
            deadline,
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
                    "--strict-config",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--cd",
                    request.workspace,
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
            command.extend(self._codex_permission_profile())
            if request.effort is not None:
                command.extend(
                    ["--config", f'model_reasoning_effort="{request.effort}"']
                )
            if request.prior_knowledge is not None:
                mcp_server = self._mcp_server_config(
                    request,
                    mcp_config_path.parent,
                )
                command.extend(
                    [
                        "--config",
                        f'mcp_servers.prior_knowledge.command={json.dumps(mcp_server["command"])}',
                        "--config",
                        "mcp_servers.prior_knowledge.args="
                        + json.dumps(mcp_server["args"], separators=(",", ":")),
                    ]
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
            "--safe-mode",
            "--setting-sources",
            "",
            "--exclude-dynamic-system-prompt-sections",
            "--settings",
            self._claude_security_settings(request, mcp_config_path.parent),
            "--permission-mode",
            "plan",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--model",
            request.model,
            "--disallowedTools",
            "Bash,Edit,Write,NotebookEdit",
        ]
        if request.effort is not None:
            command.extend(["--effort", request.effort])
        effective_tools = request.allowed_tools
        if request.prior_knowledge is not None:
            command.extend(
                [
                    "--mcp-config",
                    str(mcp_config_path),
                    "--strict-mcp-config",
                ]
            )
            effective_tools += (
                "mcp__prior_knowledge__list_prior_knowledge",
                "mcp__prior_knowledge__get_prior_knowledge_record",
            )
        command.extend(["--tools", ",".join(effective_tools)])
        return command

    def _codex_permission_profile(self) -> list[str]:
        profile = "kapso_ideation_read"
        denied_paths = ("/proc", *_SENSITIVE_HOME_PATHS)
        denied_entries = ",".join(f'{json.dumps(path)}="deny"' for path in denied_paths)
        filesystem = (
            "{"
            f"glob_scan_max_depth={self.settings.sensitive_file_glob_scan_max_depth},"
            '":minimal"="read",'
            '":workspace_roots"={"."="read","**/.env"="deny",'
            '"**/.env.*"="deny"},'
            f"{denied_entries}"
            "}"
        )
        overrides = (
            f'default_permissions="{profile}"',
            f"permissions={{{profile}={{filesystem={filesystem}}}}}",
        )
        return [item for override in overrides for item in ("--config", override)]

    @staticmethod
    def _claude_security_settings(
        request: CodingAgentCallRequest,
        artifact_directory: Path,
    ) -> str:
        denied_reads = [
            "Read(//proc/**)",
            "Read(**/.env)",
            "Read(**/.env.*)",
            *(f"Read({path}/**)" for path in _SENSITIVE_HOME_PATHS),
        ]
        return json.dumps(
            {
                "permissions": {"deny": denied_reads},
                "sandbox": {
                    "enabled": True,
                    "failIfUnavailable": True,
                    "filesystem": {
                        "denyRead": ["/"],
                        "allowRead": [
                            request.workspace,
                            str(artifact_directory),
                        ],
                    },
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

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
