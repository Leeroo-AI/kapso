"""Durable, fail-loud boundary for structured coding-agent calls."""

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

from jsonschema import Draft202012Validator

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.coding_agent_compatibility import (
    coding_agent_supported_tools,
)
from kapso.cross_run.contracts import CodingAgentWorkspaceDelta
from kapso.cross_run.knowledge.access import (
    PriorKnowledgeAccess,
    PriorKnowledgeAccessMaterialization,
)
from kapso.cross_run.agent_artifacts import (
    CODING_AGENT_INPUT_ARTIFACT_FILENAMES as _INPUT_FILENAMES,
    CODING_AGENT_RESULT_FILENAME as _RESULT_FILENAME,
    CODING_AGENT_WORKSPACE_DELTA_FILENAME,
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
    coding_agent_output_artifact_filenames,
    coding_agent_returned_artifact_filenames,
)
from kapso.execution.coding_agents.credential_environment import (
    coding_agent_credential_environment,
)
from kapso.execution.coding_agents.workspace_delta import (
    CodingAgentWorkspaceSnapshot,
    build_coding_agent_workspace_delta,
    inspect_coding_agent_workspace_descriptor,
    validate_coding_agent_workspace_delta,
)

_OPERATION_IDENTIFIER_PATTERN = re.compile(r"^agent_call_[0-9a-f]{32}$")
_EMPTY_MCP_AUDIT_DIGEST = tree_or_blob_digest(b"")
_CREDENTIAL_ENVIRONMENT_POLICY_VERSION = "kapso.coding_agent_credentials.v1"
_FILESYSTEM_POLICY_VERSION = "kapso.coding_agent_workspace.v4"
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


def coding_agent_response_schema_bytes(
    response_schema: Mapping[str, Any],
) -> bytes:
    if not isinstance(response_schema, Mapping):
        raise ValueError("coding-agent response schema must be an object")
    return (
        json.dumps(response_schema, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _claude_response_schema_argument(schema_text: str) -> str:
    schema = json.loads(schema_text)
    if not isinstance(schema, dict):
        raise ValueError("Claude response schema must be an object")
    dialect = schema.pop("$schema", None)
    if dialect not in {
        None,
        "https://json-schema.org/draft/2020-12/schema",
    }:
        raise ValueError("Claude response schema dialect is unsupported")
    return json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _codex_supports_response_schema(schema_text: str) -> bool:
    """Return whether every object uses Codex's closed structured-output shape."""

    schema = json.loads(schema_text)
    if not isinstance(schema, dict):
        raise ValueError("Codex response schema must be an object")

    def supports(node: Any) -> bool:
        if isinstance(node, list):
            return all(supports(item) for item in node)
        if not isinstance(node, dict):
            return True
        if node.get("type") == "object":
            properties = node.get("properties")
            required = node.get("required")
            if (
                not isinstance(properties, dict)
                or node.get("additionalProperties") is not False
                or not isinstance(required, list)
                or len(required) != len(set(required))
                or set(required) != set(properties)
            ):
                return False
        return all(supports(value) for value in node.values())

    return supports(schema)


def coding_agent_invocation_bytes(
    request: "CodingAgentCallRequest",
    *,
    sensitive_file_glob_scan_max_depth: int,
) -> bytes:
    _require_positive_integer(
        sensitive_file_glob_scan_max_depth,
        "coding-agent sensitive-file glob scan depth",
    )
    return (
        json.dumps(
            {
                "role": request.role,
                "cli": request.cli,
                "model": request.model,
                "timeout_seconds": request.timeout_seconds,
                "effort": request.effort,
                "allowed_tools": list(request.allowed_tools),
                "workspace_policy": request.workspace_policy.to_dict(),
                "credential_environment_policy_version": (
                    _CREDENTIAL_ENVIRONMENT_POLICY_VERSION
                ),
                "filesystem_policy_version": _FILESYSTEM_POLICY_VERSION,
                "mcp_audit_policy_version": _MCP_AUDIT_POLICY_VERSION,
                "sensitive_file_glob_scan_max_depth": (
                    sensitive_file_glob_scan_max_depth
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
    ).encode("utf-8")


def coding_agent_policy_versions() -> Mapping[str, str]:
    return {
        "credential_environment_policy_version": (
            _CREDENTIAL_ENVIRONMENT_POLICY_VERSION
        ),
        "filesystem_policy_version": _FILESYSTEM_POLICY_VERSION,
        "mcp_audit_policy_version": _MCP_AUDIT_POLICY_VERSION,
    }


def coding_agent_mcp_configuration_fingerprint(
    prior_knowledge: PriorKnowledgeAccessMaterialization | None,
) -> str:
    semantic_configuration = {
        "enabled": prior_knowledge is not None,
        "enabled_gates": (() if prior_knowledge is None else ("prior_knowledge",)),
        "gate_failure_policy": None if prior_knowledge is None else "error",
        "module": (
            None if prior_knowledge is None else "kapso.gated_mcp.prior_knowledge_cli"
        ),
        "prior_knowledge_materialization_digest": (
            None if prior_knowledge is None else prior_knowledge.materialization_digest
        ),
        "prior_knowledge_maximum_bytes": (
            None if prior_knowledge is None else len(prior_knowledge.to_json_bytes())
        ),
        "prior_knowledge_audit_maximum_bytes": (
            None if prior_knowledge is None else len(prior_knowledge.to_json_bytes())
        ),
    }
    return tree_or_blob_digest(canonical_json_bytes(semantic_configuration))


def coding_agent_mcp_configuration_bytes(
    request: "CodingAgentCallRequest",
    artifact_directory: Path,
) -> bytes:
    servers = {}
    if request.prior_knowledge is not None:
        servers["prior_knowledge"] = coding_agent_mcp_server_configuration(
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
    ).encode("utf-8")


def coding_agent_mcp_server_configuration(
    request: "CodingAgentCallRequest",
    artifact_directory: Path,
) -> dict[str, Any]:
    if request.prior_knowledge is None:
        raise ValueError("prior-knowledge MCP requires a materialization")
    python_path = Path(__file__).resolve().parents[3]
    if not (python_path / "kapso" / "gated_mcp").is_dir():
        raise ValueError("Kapso package root is missing for prior-knowledge MCP")
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
            "kapso.gated_mcp.prior_knowledge_cli",
            "--enabled-gates",
            "prior_knowledge",
            "--gate-failure-policy",
            "error",
            "--prior-knowledge-path",
            str(artifact_directory / "prior_knowledge.json"),
            "--prior-knowledge-maximum-bytes",
            str(len(request.prior_knowledge.to_json_bytes())),
            "--prior-knowledge-audit-path",
            str(artifact_directory / "mcp_audit.jsonl"),
            "--prior-knowledge-audit-maximum-bytes",
            str(len(request.prior_knowledge.to_json_bytes())),
            "--operation-id",
            request.operation_id,
        ],
    }


def validate_coding_agent_mcp_audit(
    *,
    operation_id: str,
    prior_knowledge: PriorKnowledgeAccessMaterialization | None,
    audit_text: str,
) -> tuple[int, str]:
    if not audit_text:
        return 0, _EMPTY_MCP_AUDIT_DIGEST
    if prior_knowledge is None:
        raise CodingAgentInvocationError(
            "coding-agent call without prior knowledge produced an MCP audit"
        )
    if not audit_text.endswith("\n"):
        raise CodingAgentInvocationError(
            "prior-knowledge MCP audit has an incomplete final event"
        )
    lines = audit_text.splitlines()
    if any(not line.strip() for line in lines):
        raise CodingAgentInvocationError("prior-knowledge MCP audit has a blank line")
    access = PriorKnowledgeAccess(prior_knowledge)
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
        if event["operation_id"] != operation_id:
            raise CodingAgentInvocationError(
                "prior-knowledge MCP audit operation identity changed"
            )
        if event["prior_knowledge_snapshot_id"] != (packet.prior_knowledge_snapshot_id):
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


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
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
class CodingAgentWorkspacePolicy:
    """Required authority and exact-tree limits for one agent workspace."""

    access: CodingAgentWorkspaceAccess
    expected_tree_hash: str | None
    maximum_entries: int | None
    maximum_bytes: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.access, CodingAgentWorkspaceAccess):
            raise ValueError("coding-agent workspace access is invalid")
        if self.access is CodingAgentWorkspaceAccess.READ_ONLY:
            if any(
                value is not None
                for value in (
                    self.expected_tree_hash,
                    self.maximum_entries,
                    self.maximum_bytes,
                )
            ):
                raise ValueError(
                    "read-only coding-agent workspace cannot declare edit limits"
                )
            return
        if (
            not isinstance(self.expected_tree_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.expected_tree_hash) is None
        ):
            raise ValueError(
                "editable coding-agent workspace requires an exact tree hash"
            )
        _require_positive_integer(
            self.maximum_entries,
            "coding-agent workspace maximum entries",
        )
        _require_positive_integer(
            self.maximum_bytes,
            "coding-agent workspace maximum bytes",
        )

    @classmethod
    def read_only(cls) -> "CodingAgentWorkspacePolicy":
        return cls(
            access=CodingAgentWorkspaceAccess.READ_ONLY,
            expected_tree_hash=None,
            maximum_entries=None,
            maximum_bytes=None,
        )

    @classmethod
    def edit_workspace(
        cls,
        *,
        expected_tree_hash: str,
        maximum_entries: int,
        maximum_bytes: int,
    ) -> "CodingAgentWorkspacePolicy":
        return cls(
            access=CodingAgentWorkspaceAccess.EDIT_WORKSPACE,
            expected_tree_hash=expected_tree_hash,
            maximum_entries=maximum_entries,
            maximum_bytes=maximum_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "access": self.access.value,
            "expected_tree_hash": self.expected_tree_hash,
            "maximum_entries": self.maximum_entries,
            "maximum_bytes": self.maximum_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodingAgentWorkspacePolicy":
        values = _require_exact_fields(
            payload,
            {
                "access",
                "expected_tree_hash",
                "maximum_entries",
                "maximum_bytes",
            },
            "coding-agent workspace policy",
        )
        return cls(
            access=CodingAgentWorkspaceAccess(values["access"]),
            expected_tree_hash=values["expected_tree_hash"],
            maximum_entries=values["maximum_entries"],
            maximum_bytes=values["maximum_bytes"],
        )


@dataclass(frozen=True)
class CodingAgentCallRequest:
    """Complete immutable input to one structured coding-agent operation."""

    operation_id: str
    role: str
    cli: str
    model: str
    prompt: str
    workspace: str
    workspace_policy: CodingAgentWorkspacePolicy
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
        if not isinstance(self.workspace_policy, CodingAgentWorkspacePolicy):
            raise ValueError("coding-agent workspace policy is invalid")
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
            "workspace_policy": self.workspace_policy.to_dict(),
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
                "workspace_policy",
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
            workspace_policy=CodingAgentWorkspacePolicy.from_dict(
                values["workspace_policy"]
            ),
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
    final_output_digest: str
    workspace_delta_digest: str | None
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
        if (
            not isinstance(self.final_output_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.final_output_digest) is None
        ):
            raise ValueError("coding-agent final output digest is invalid")
        if self.workspace_delta_digest is not None and (
            not isinstance(self.workspace_delta_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.workspace_delta_digest) is None
        ):
            raise ValueError("coding-agent workspace delta digest is invalid")
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
            "final_output_digest": self.final_output_digest,
            "workspace_delta_digest": self.workspace_delta_digest,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "artifacts": list(self.artifacts),
            "mcp_audit_digest": self.mcp_audit_digest,
            "mcp_audit_event_count": self.mcp_audit_event_count,
        }

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodingAgentCallResult":
        values = _require_exact_fields(
            payload,
            {
                "output",
                "duration_seconds",
                "cost_usd",
                "final_output_digest",
                "workspace_delta_digest",
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
            final_output_digest=values["final_output_digest"],
            workspace_delta_digest=values["workspace_delta_digest"],
            input_tokens=values["input_tokens"],
            output_tokens=values["output_tokens"],
            artifacts=values["artifacts"],
            mcp_audit_digest=values["mcp_audit_digest"],
            mcp_audit_event_count=values["mcp_audit_event_count"],
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "CodingAgentCallResult":
        if not isinstance(payload, bytes):
            raise ValueError("coding-agent result JSON must be bytes")
        return cls.from_dict(json.loads(payload))


class CodingAgentCallRunner(Protocol):
    def run(
        self,
        request: CodingAgentCallRequest,
        response_schema: Mapping[str, Any],
        *,
        workspace_authority_descriptor: int | None = None,
    ) -> CodingAgentCallResult:
        """Run one complete structured agent invocation."""


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


class _CodingAgentWorkspaceLease:
    """Exclusive lease for one private editable workspace path."""

    def __init__(
        self,
        workspace: Path,
        authority_descriptor: int | None,
    ):
        self.workspace = workspace
        self.authority_descriptor = authority_descriptor
        self.handle = None
        self.parent_descriptor = None
        self.parent_identity = None
        self.workspace_descriptor = None
        self.workspace_identity = None

    def __enter__(self):
        parent = self.workspace.parent
        parent_metadata = parent.stat(follow_symlinks=False)
        if (
            parent.is_symlink()
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            raise CodingAgentInvocationError(
                "editable coding-agent workspace parent must be private"
            )
        parent_identity = parent_metadata.st_dev, parent_metadata.st_ino
        lease_digest = tree_or_blob_digest(str(self.workspace).encode("utf-8"))[7:39]
        lease_name = f".kapso-workspace-{lease_digest}.lock"
        with ExitStack() as descriptors:
            parent_descriptor = os.open(
                parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptors.callback(os.close, parent_descriptor)
            opened_parent = os.fstat(parent_descriptor)
            if (opened_parent.st_dev, opened_parent.st_ino) != parent_identity:
                raise CodingAgentInvocationError(
                    "editable coding-agent workspace parent changed while opening"
                )
            workspace_metadata = os.stat(
                self.workspace.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            workspace_identity = (
                workspace_metadata.st_dev,
                workspace_metadata.st_ino,
            )
            workspace_descriptor = os.open(
                self.workspace.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_descriptor,
            )
            descriptors.callback(os.close, workspace_descriptor)
            opened_workspace = os.fstat(workspace_descriptor)
            if (
                not stat.S_ISDIR(workspace_metadata.st_mode)
                or (opened_workspace.st_dev, opened_workspace.st_ino)
                != workspace_identity
            ):
                raise CodingAgentInvocationError(
                    "editable coding-agent workspace changed while opening"
                )
            if self.authority_descriptor is not None:
                authority = os.fstat(self.authority_descriptor)
                if (
                    not stat.S_ISDIR(authority.st_mode)
                    or (authority.st_dev, authority.st_ino) != workspace_identity
                ):
                    raise CodingAgentInvocationError(
                        "editable coding-agent workspace differs from its authority"
                    )
            lease_descriptor = os.open(
                lease_name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=parent_descriptor,
            )
            descriptors.callback(os.close, lease_descriptor)
            handle = os.fdopen(lease_descriptor, "r+b", closefd=False)
            descriptors.enter_context(handle)
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
            ):
                raise CodingAgentInvocationError(
                    "coding-agent workspace lease must be a private independent file"
                )
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self.handle = handle
            self.parent_descriptor = parent_descriptor
            self.parent_identity = parent_identity
            self.workspace_descriptor = workspace_descriptor
            self.workspace_identity = workspace_identity
            descriptors.pop_all()
            return self

    def __exit__(self, exception_type, exception, traceback):
        with ExitStack() as descriptors:
            descriptors.callback(os.close, self.parent_descriptor)
            descriptors.callback(os.close, self.workspace_descriptor)
            descriptors.callback(os.close, self.handle.fileno())
            descriptors.enter_context(self.handle)
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            current_parent = self.workspace.parent.stat(follow_symlinks=False)
            current_workspace = os.stat(
                self.workspace.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                (current_parent.st_dev, current_parent.st_ino) != self.parent_identity
                or (current_workspace.st_dev, current_workspace.st_ino)
                != self.workspace_identity
                or (
                    os.fstat(self.workspace_descriptor).st_dev,
                    os.fstat(self.workspace_descriptor).st_ino,
                )
                != self.workspace_identity
            ):
                raise CodingAgentInvocationError(
                    "editable coding-agent workspace binding changed during use"
                )
        self.handle = None
        self.parent_descriptor = None
        self.parent_identity = None
        self.workspace_descriptor = None
        self.workspace_identity = None
        return False


class SubprocessCodingAgentCallRunner:
    """Invoke Codex or Claude Code through a locked immutable operation identity."""

    def __init__(self, settings: CodingAgentRunnerSettings):
        self.settings = settings

    def run(
        self,
        request: CodingAgentCallRequest,
        response_schema: Mapping[str, Any],
        *,
        workspace_authority_descriptor: int | None = None,
    ) -> CodingAgentCallResult:
        workspace = _validate_coding_agent_workspace(request.workspace)
        if shutil.which("timeout") is None:
            raise RuntimeError("GNU timeout is required for coding-agent deadlines")
        executable = "codex" if request.cli == "codex" else "claude"
        if shutil.which(executable) is None:
            raise RuntimeError(f"coding-agent CLI is not installed: {executable}")
        schema_bytes = coding_agent_response_schema_bytes(response_schema)
        supported_tools = coding_agent_supported_tools(
            request.cli,
            edit_workspace=(
                request.workspace_policy.access
                is CodingAgentWorkspaceAccess.EDIT_WORKSPACE
            ),
        )
        if not set(request.allowed_tools).issubset(supported_tools):
            raise ValueError("coding-agent request contains an unsupported tool")
        schema_text = schema_bytes.decode("utf-8")
        invocation_text = coding_agent_invocation_bytes(
            request,
            sensitive_file_glob_scan_max_depth=(
                self.settings.sensitive_file_glob_scan_max_depth
            ),
        ).decode("utf-8")
        artifact_root = self._prepare_artifact_root()
        with ExitStack() as descriptors:
            workspace_descriptor = None
            if request.workspace_policy.access is (
                CodingAgentWorkspaceAccess.EDIT_WORKSPACE
            ):
                workspace_lease = descriptors.enter_context(
                    _CodingAgentWorkspaceLease(
                        workspace,
                        workspace_authority_descriptor,
                    )
                )
                workspace_descriptor = workspace_lease.workspace_descriptor
            elif workspace_authority_descriptor is not None:
                raise CodingAgentInvocationError(
                    "read-only coding-agent call cannot receive edit authority"
                )
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
            if (
                not stat.S_ISREG(lock_status.st_mode)
                or lock_status.st_nlink != 1
                or lock_status.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
            ):
                os.close(lock_descriptor)
                raise CodingAgentInvocationError(
                    "coding-agent operation lock must be a private independent file"
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
                workspace_descriptor=workspace_descriptor,
            )

    def _prepare_artifact_root(self) -> Path:
        artifact_root = Path(self.settings.artifact_root)
        self._validate_artifact_root_components(
            artifact_root,
            require_complete=False,
        )
        artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
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
            if require_complete:
                final_status = os.fstat(current_descriptor)
                if final_status.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
                    raise CodingAgentInvocationError(
                        "coding-agent artifact root must be private"
                    )

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
            if not stat.S_ISDIR(status.st_mode) or status.st_mode & (
                0o077 | stat.S_ISUID | stat.S_ISGID
            ):
                raise CodingAgentInvocationError(
                    "coding-agent operation path must be a private directory"
                )
        else:
            os.mkdir(operation_id, mode=0o700, dir_fd=root_descriptor)
            os.fsync(root_descriptor)
        descriptor = os.open(
            operation_id,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_descriptor,
        )
        opened = os.fstat(descriptor)
        if opened.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
            os.close(descriptor)
            raise CodingAgentInvocationError(
                "coding-agent operation path must be a private directory"
            )
        return descriptor

    def _run_locked(
        self,
        *,
        request: CodingAgentCallRequest,
        schema_text: str,
        invocation_text: str,
        operation_descriptor: int,
        artifact_directory: Path,
        workspace_descriptor: int | None,
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
                workspace_descriptor,
            )
        baseline = self._inspect_editable_workspace(
            request,
            workspace_descriptor,
        )
        recoverable_outputs = {
            filename
            for access in CodingAgentWorkspaceAccess
            for filename in coding_agent_output_artifact_filenames(access)
        }
        for filename in recoverable_outputs:
            if filename in set(os.listdir(operation_descriptor)):
                self._remove_regular_file(operation_descriptor, filename)
        self._write_atomic_text(
            operation_descriptor,
            "mcp_audit.jsonl",
            "",
        )
        final_path = artifact_directory / "final.json"
        schema_path = artifact_directory / "response_schema.json"
        mcp_config_path = artifact_directory / "mcp_config.json"
        if request.cli == "codex":
            self._write_atomic_text(operation_descriptor, "final.json", "")
        command = self._command(
            request,
            schema_text,
            schema_path,
            final_path,
            mcp_config_path,
            workspace_descriptor,
        )
        execution_workspace = Path(request.workspace)
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=execution_workspace,
            env=coding_agent_credential_environment(request.cli),
            input=request.prompt,
            text=True,
            capture_output=True,
            check=False,
            pass_fds=(),
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
                output,
            )
        self._validate_response_output(schema_text, output)
        workspace_delta_digest = None
        if baseline is not None:
            edited = self._inspect_editable_workspace(
                request,
                workspace_descriptor,
                require_baseline=False,
            )
            if edited is None:
                raise CodingAgentInvocationError(
                    "editable coding-agent workspace produced no observation"
                )
            delta = build_coding_agent_workspace_delta(baseline, edited)
            validate_coding_agent_workspace_delta(baseline, delta)
            delta_payload = delta.to_json_bytes()
            self._write_atomic_text(
                operation_descriptor,
                CODING_AGENT_WORKSPACE_DELTA_FILENAME,
                delta_payload.decode("utf-8"),
            )
            workspace_delta_digest = tree_or_blob_digest(delta_payload)
        final_output = self._read_regular_text(
            operation_descriptor,
            "final.json",
        )
        if final_output != output:
            raise CodingAgentInvocationError(
                "coding-agent parsed output conflicts with final artifact"
            )
        final_output_digest = tree_or_blob_digest(final_output.encode("utf-8"))
        artifacts = self._artifact_paths(
            artifact_directory,
            request.workspace_policy.access,
        )
        result = CodingAgentCallResult(
            output=output,
            duration_seconds=duration,
            cost_usd=cost_usd,
            final_output_digest=final_output_digest,
            workspace_delta_digest=workspace_delta_digest,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            artifacts=artifacts,
            mcp_audit_digest=audit_digest,
            mcp_audit_event_count=audit_event_count,
        )
        self._write_atomic_text(
            operation_descriptor,
            _RESULT_FILENAME,
            result.to_json_bytes().decode("utf-8"),
        )
        return result

    @staticmethod
    def _validate_and_recover_operation_directory(
        descriptor: int,
    ) -> None:
        artifact_filenames = {
            filename
            for access in CodingAgentWorkspaceAccess
            for filename in coding_agent_artifact_filenames(access)
        }
        allowed = set(artifact_filenames)
        temporary = {f".{filename}.tmp" for filename in artifact_filenames}
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
    def _require_regular_file(descriptor: int, filename: str) -> os.stat_result:
        status = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or status.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            raise CodingAgentInvocationError(
                "coding-agent artifact must be a private independent file: "
                f"{filename}"
            )
        return status

    @staticmethod
    def _read_regular_text(descriptor: int, filename: str) -> str:
        status = SubprocessCodingAgentCallRunner._require_regular_file(
            descriptor, filename
        )
        file_descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=descriptor,
        )
        with os.fdopen(file_descriptor, "r", encoding="utf-8") as handle:
            opened = os.fstat(handle.fileno())
            if (
                (opened.st_dev, opened.st_ino) != (status.st_dev, status.st_ino)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
            ):
                raise CodingAgentInvocationError(
                    f"coding-agent artifact changed during read: {filename}"
                )
            text = handle.read()
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
        workspace_descriptor: int | None,
    ) -> CodingAgentCallResult:
        workspace_access = request.workspace_policy.access
        if set(os.listdir(operation_descriptor)) != set(
            coding_agent_artifact_filenames(workspace_access)
        ):
            raise CodingAgentInvocationError(
                "completed coding-agent operation has a conflicting artifact set"
            )
        for filename in coding_agent_artifact_filenames(workspace_access):
            self._require_regular_file(operation_descriptor, filename)
        result = CodingAgentCallResult.from_dict(
            json.loads(self._read_regular_text(operation_descriptor, _RESULT_FILENAME))
        )
        final_output = self._read_regular_text(
            operation_descriptor,
            "final.json",
        )
        final_output_digest = tree_or_blob_digest(final_output.encode("utf-8"))
        if result.final_output_digest != final_output_digest:
            raise CodingAgentInvocationError(
                "cached coding-agent final output conflicts with result"
            )
        if result.output != final_output:
            raise CodingAgentInvocationError(
                "cached coding-agent parsed output conflicts with final artifact"
            )
        response_schema = self._read_regular_text(
            operation_descriptor,
            "response_schema.json",
        )
        self._validate_response_output(response_schema, final_output)
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
        if workspace_access is CodingAgentWorkspaceAccess.EDIT_WORKSPACE:
            delta_payload = self._read_regular_text(
                operation_descriptor,
                CODING_AGENT_WORKSPACE_DELTA_FILENAME,
            ).encode("utf-8")
            if result.workspace_delta_digest != tree_or_blob_digest(delta_payload):
                raise CodingAgentInvocationError(
                    "cached coding-agent workspace delta conflicts with result"
                )
            delta = CodingAgentWorkspaceDelta.from_json_bytes(delta_payload)
            if delta.to_json_bytes() != delta_payload:
                raise CodingAgentInvocationError(
                    "cached coding-agent workspace delta is not canonical"
                )
            observed = self._inspect_editable_workspace(
                request,
                workspace_descriptor,
                require_baseline=False,
            )
            if observed is None:
                raise CodingAgentInvocationError(
                    "editable coding-agent workspace produced no observation"
                )
            if delta.baseline_tree_hash != request.workspace_policy.expected_tree_hash:
                raise CodingAgentInvocationError(
                    "cached coding-agent workspace delta names another baseline"
                )
            validate_coding_agent_workspace_delta(observed, delta)
        elif result.workspace_delta_digest is not None:
            raise CodingAgentInvocationError(
                "read-only coding-agent result names a workspace delta"
            )
        if result.artifacts != self._artifact_paths(
            artifact_directory,
            workspace_access,
        ):
            raise CodingAgentInvocationError(
                "cached coding-agent artifact references are invalid"
            )
        return result

    @staticmethod
    def _artifact_paths(
        artifact_directory: Path,
        workspace_access: CodingAgentWorkspaceAccess,
    ) -> tuple[str, ...]:
        return tuple(
            str(artifact_directory / filename)
            for filename in coding_agent_returned_artifact_filenames(workspace_access)
        )

    @staticmethod
    def _inspect_editable_workspace(
        request: CodingAgentCallRequest,
        workspace_descriptor: int | None,
        *,
        require_baseline: bool = True,
    ) -> CodingAgentWorkspaceSnapshot | None:
        policy = request.workspace_policy
        if policy.access is CodingAgentWorkspaceAccess.READ_ONLY:
            if workspace_descriptor is not None:
                raise CodingAgentInvocationError(
                    "read-only coding-agent workspace received edit authority"
                )
            return None
        if workspace_descriptor is None:
            raise CodingAgentInvocationError(
                "editable coding-agent workspace lacks a pinned descriptor"
            )
        observed = inspect_coding_agent_workspace_descriptor(
            workspace_descriptor,
            maximum_entries=policy.maximum_entries,
            maximum_bytes=policy.maximum_bytes,
        )
        if require_baseline and observed.tree_hash != policy.expected_tree_hash:
            raise CodingAgentInvocationError(
                "editable coding-agent workspace differs from its expected tree"
            )
        return observed

    @staticmethod
    def _prior_knowledge_text(request: CodingAgentCallRequest) -> str:
        if request.prior_knowledge is None:
            return "null\n"
        return request.prior_knowledge.to_json_bytes().decode("utf-8")

    def _mcp_config_text(
        self,
        request: CodingAgentCallRequest,
        artifact_directory: Path,
    ) -> str:
        return coding_agent_mcp_configuration_bytes(
            request,
            artifact_directory,
        ).decode("utf-8")

    @staticmethod
    def _validate_response_output(schema_text: str, output: str) -> None:
        schema = json.loads(schema_text)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(json.loads(output))

    @staticmethod
    def _validate_mcp_audit(
        request: CodingAgentCallRequest,
        audit_text: str,
    ) -> tuple[int, str]:
        return validate_coding_agent_mcp_audit(
            operation_id=request.operation_id,
            prior_knowledge=request.prior_knowledge,
            audit_text=audit_text,
        )

    def _command(
        self,
        request: CodingAgentCallRequest,
        schema_text: str,
        schema_path: Path,
        final_path: Path,
        mcp_config_path: Path,
        workspace_descriptor: int | None,
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
                    "--output-last-message",
                    str(final_path),
                    "--json",
                    "--color",
                    "never",
                    "--model",
                    request.model,
                ]
            )
            if _codex_supports_response_schema(schema_text):
                command.extend(["--output-schema", str(schema_path)])
            command.extend(
                self._codex_permission_profile(
                    request.workspace_policy.access,
                    request.workspace,
                )
            )
            if request.effort is not None:
                command.extend(
                    ["--config", f'model_reasoning_effort="{request.effort}"']
                )
            if request.prior_knowledge is not None:
                mcp_server = coding_agent_mcp_server_configuration(
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
        schema = _claude_response_schema_argument(schema_text)
        command = prefix + [
            "claude",
            "--print",
            "--safe-mode",
            "--setting-sources",
            "",
            "--exclude-dynamic-system-prompt-sections",
            "--settings",
            self._claude_security_settings(
                request,
                mcp_config_path.parent,
            ),
            "--permission-mode",
            (
                "acceptEdits"
                if request.workspace_policy.access
                is CodingAgentWorkspaceAccess.EDIT_WORKSPACE
                else "plan"
            ),
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--model",
            request.model,
            "--disallowedTools",
            (
                "Bash,NotebookEdit"
                if request.workspace_policy.access
                is CodingAgentWorkspaceAccess.EDIT_WORKSPACE
                else "Bash,Edit,Write,NotebookEdit"
            ),
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

    def _codex_permission_profile(
        self,
        workspace_access: CodingAgentWorkspaceAccess,
        workspace: str,
    ) -> list[str]:
        profile = (
            "kapso_workspace_edit"
            if workspace_access is CodingAgentWorkspaceAccess.EDIT_WORKSPACE
            else "kapso_ideation_read"
        )
        denied_paths = ("/proc", *_SENSITIVE_HOME_PATHS)
        denied_entries = ",".join(f'{json.dumps(path)}="deny"' for path in denied_paths)
        workspace_rules = (
            '"."="write","**/.git"="deny","**/.git/**"="deny",'
            '"**/.env"="deny","**/.env.*"="deny"'
            if workspace_access is CodingAgentWorkspaceAccess.EDIT_WORKSPACE
            else '"."="read","**/.env"="deny","**/.env.*"="deny"'
        )
        filesystem = (
            "{"
            f"glob_scan_max_depth={self.settings.sensitive_file_glob_scan_max_depth},"
            '":minimal"="read",'
            f'":workspace_roots"={{{workspace_rules}}},'
            f"{denied_entries}"
            "}"
        )
        overrides = (
            f'default_permissions="{profile}"',
            "permissions={"
            f"{profile}={{workspace_roots={{{json.dumps(workspace)}=true}},"
            f"filesystem={filesystem}}}"
            "}",
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
        denied_edits = (
            [
                "Edit(**/.git)",
                "Edit(**/.git/**)",
                "Edit(**/.env)",
                "Edit(**/.env.*)",
            ]
            if request.workspace_policy.access
            is CodingAgentWorkspaceAccess.EDIT_WORKSPACE
            else []
        )
        return json.dumps(
            {
                "permissions": {"deny": denied_reads + denied_edits},
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
