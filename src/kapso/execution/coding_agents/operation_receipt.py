"""Seal one completed coding-agent artifact directory into a typed receipt."""

from __future__ import annotations

import json
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.agent_artifacts import (
    CODING_AGENT_RESULT_FILENAME,
    CODING_AGENT_WORKSPACE_DELTA_FILENAME,
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
    coding_agent_returned_artifact_filenames,
)
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    CodingAgentOperationReceipt,
    CodingAgentWorkspaceDelta,
)
from kapso.cross_run.settings import CodingAgentSettings
from kapso.cross_run.knowledge.access import PriorKnowledgeAccessMaterialization
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    CodingAgentWorkspacePolicy,
    coding_agent_mcp_configuration_bytes,
    coding_agent_mcp_configuration_fingerprint,
    coding_agent_invocation_bytes,
    coding_agent_policy_versions,
    coding_agent_response_schema_bytes,
    validate_coding_agent_mcp_audit,
)


class CodingAgentOperationReceiptError(ValueError):
    """A completed coding-agent artifact closure is unsafe or inconsistent."""


@dataclass(frozen=True)
class SealedCodingAgentOperation:
    receipt: CodingAgentOperationReceipt
    final_output: str
    workspace_delta: CodingAgentWorkspaceDelta | None
    artifact_directory: Path
    artifact_bytes: Mapping[str, bytes]


@dataclass(frozen=True)
class VerifiedCodingAgentArtifactClosure:
    result: CodingAgentCallResult
    final_output: str
    workspace_delta: CodingAgentWorkspaceDelta | None
    invocation: Mapping[str, Any]
    mcp_configuration_fingerprint: str
    prior_knowledge: PriorKnowledgeAccessMaterialization | None


def seal_coding_agent_operation(
    *,
    request: CodingAgentCallRequest,
    response_schema: Mapping[str, Any],
    principal_id: str,
    agent: CodingAgentSettings,
    sensitive_file_glob_scan_max_depth: int,
    result: CodingAgentCallResult,
) -> SealedCodingAgentOperation:
    """Verify exact private artifacts and mint their immutable receipt."""

    if (
        request.cli != agent.cli
        or request.model != agent.model
        or request.timeout_seconds != agent.timeout_seconds
        or request.effort != agent.effort
        or request.allowed_tools != agent.allowed_tools
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent request differs from configured agent"
        )
    workspace_access = request.workspace_policy.access
    artifact_paths = tuple(Path(path) for path in result.artifacts)
    if not artifact_paths:
        raise CodingAgentOperationReceiptError(
            "coding-agent operation returned no artifacts"
        )
    if any(
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or ".." in path.parts
        for path in artifact_paths
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent artifact paths must be absolute and normalized"
        )
    directories = {path.parent for path in artifact_paths}
    names = {path.name for path in artifact_paths}
    if len(directories) != 1 or names != set(
        coding_agent_returned_artifact_filenames(workspace_access)
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent operation artifact set is invalid"
        )
    artifact_directory = next(iter(directories))
    if artifact_directory.name != request.operation_id:
        raise CodingAgentOperationReceiptError(
            "coding-agent artifact directory names another operation"
        )
    payloads = _read_private_artifact_directory(
        artifact_directory,
        coding_agent_artifact_filenames(workspace_access),
    )
    verified = verify_coding_agent_operation_artifacts(
        operation_id=request.operation_id,
        workspace_access=workspace_access,
        artifact_bytes=payloads,
    )
    if verified.result != result:
        raise CodingAgentOperationReceiptError(
            "coding-agent durable result differs from returned result"
        )
    if payloads["prompt.txt"] != request.prompt.encode("utf-8"):
        raise CodingAgentOperationReceiptError(
            "coding-agent prompt artifact differs from request"
        )
    if payloads["response_schema.json"] != coding_agent_response_schema_bytes(
        response_schema
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent schema artifact differs from request"
        )
    if payloads["invocation.json"] != coding_agent_invocation_bytes(
        request,
        sensitive_file_glob_scan_max_depth=sensitive_file_glob_scan_max_depth,
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent invocation artifact differs from request"
        )
    expected_prior_knowledge = (
        b"null\n"
        if request.prior_knowledge is None
        else request.prior_knowledge.to_json_bytes()
    )
    if payloads["prior_knowledge.json"] != expected_prior_knowledge:
        raise CodingAgentOperationReceiptError(
            "coding-agent prior-knowledge artifact differs from request"
        )
    if payloads["mcp_config.json"] != coding_agent_mcp_configuration_bytes(
        request,
        artifact_directory,
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent MCP configuration artifact differs from request"
        )
    checksums = {
        name: tree_or_blob_digest(payload) for name, payload in sorted(payloads.items())
    }
    receipt = CodingAgentOperationReceipt.mint(
        operation_id=request.operation_id,
        principal_id=principal_id,
        role=request.role,
        cli=agent.cli,
        model=agent.model,
        effort=agent.effort,
        workspace_access=workspace_access,
        artifact_checksums=checksums,
    )
    return SealedCodingAgentOperation(
        receipt=receipt,
        final_output=verified.final_output,
        workspace_delta=verified.workspace_delta,
        artifact_directory=artifact_directory,
        artifact_bytes=MappingProxyType(payloads),
    )


def verify_coding_agent_operation_artifacts(
    *,
    operation_id: str,
    workspace_access: CodingAgentWorkspaceAccess,
    artifact_bytes: Mapping[str, bytes],
) -> VerifiedCodingAgentArtifactClosure:
    expected_names = set(coding_agent_artifact_filenames(workspace_access))
    if set(artifact_bytes) != expected_names or any(
        not isinstance(payload, bytes) for payload in artifact_bytes.values()
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent artifact byte closure is invalid"
        )
    payloads = dict(artifact_bytes)
    result_payload = payloads[CODING_AGENT_RESULT_FILENAME]
    result = CodingAgentCallResult.from_json_bytes(result_payload)
    if result_payload != result.to_json_bytes():
        raise CodingAgentOperationReceiptError(
            "coding-agent durable result is not canonical"
        )
    result_paths = tuple(Path(path) for path in result.artifacts)
    if (
        any(
            not path.is_absolute()
            or path != Path(os.path.abspath(path))
            or ".." in path.parts
            or path.parent.name != operation_id
            for path in result_paths
        )
        or {path.name for path in result_paths}
        != set(coding_agent_returned_artifact_filenames(workspace_access))
        or len({path.parent for path in result_paths}) != 1
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent durable result artifact references are invalid"
        )
    final_payload = payloads["final.json"]
    if (
        final_payload != result.output.encode("utf-8")
        or tree_or_blob_digest(final_payload) != result.final_output_digest
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent final artifact differs from durable result"
        )
    workspace_delta = _workspace_delta(payloads, workspace_access, result)
    invocation = json.loads(payloads["invocation.json"])
    if not isinstance(invocation, dict) or payloads["invocation.json"] != (
        json.dumps(invocation, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8"):
        raise CodingAgentOperationReceiptError(
            "coding-agent invocation artifact is invalid"
        )
    expected_invocation_fields = {
        "allowed_tools",
        "cli",
        "credential_environment_policy_version",
        "effort",
        "filesystem_policy_version",
        "mcp_audit_policy_version",
        "model",
        "prior_knowledge_materialization_digest",
        "prior_knowledge_snapshot_id",
        "role",
        "sensitive_file_glob_scan_max_depth",
        "timeout_seconds",
        "workspace_policy",
    }
    if set(invocation) != expected_invocation_fields:
        raise CodingAgentOperationReceiptError(
            "coding-agent invocation fields are invalid"
        )
    workspace_policy = invocation["workspace_policy"]
    if not isinstance(workspace_policy, dict):
        raise CodingAgentOperationReceiptError(
            "coding-agent invocation workspace policy is invalid"
        )
    parsed_workspace_policy = CodingAgentWorkspacePolicy.from_dict(workspace_policy)
    if parsed_workspace_policy.access is not workspace_access:
        raise CodingAgentOperationReceiptError(
            "coding-agent invocation access differs from artifact closure"
        )
    expected_policy_versions = coding_agent_policy_versions()
    if any(
        invocation[field] != version
        for field, version in expected_policy_versions.items()
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent invocation names an unrecognized policy version"
        )
    schema = json.loads(payloads["response_schema.json"])
    if not isinstance(schema, dict) or payloads[
        "response_schema.json"
    ] != coding_agent_response_schema_bytes(schema):
        raise CodingAgentOperationReceiptError(
            "coding-agent response schema artifact is invalid"
        )
    prompt = payloads["prompt.txt"].decode("utf-8")
    if not prompt.strip():
        raise CodingAgentOperationReceiptError("coding-agent prompt artifact is empty")
    prior_knowledge = _validate_prior_knowledge_artifact(payloads, invocation)
    audit_payload = payloads["mcp_audit.jsonl"]
    audit_event_count, audit_digest = validate_coding_agent_mcp_audit(
        operation_id=operation_id,
        prior_knowledge=prior_knowledge,
        audit_text=audit_payload.decode("utf-8"),
    )
    if (
        audit_digest != result.mcp_audit_digest
        or audit_event_count != result.mcp_audit_event_count
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent audit artifact differs from durable result"
        )
    mcp_configuration_fingerprint = _validate_mcp_configuration(
        payloads["mcp_config.json"],
        operation_id=operation_id,
        prior_knowledge=prior_knowledge,
    )
    return VerifiedCodingAgentArtifactClosure(
        result=result,
        final_output=final_payload.decode("utf-8"),
        workspace_delta=workspace_delta,
        invocation=MappingProxyType(invocation),
        mcp_configuration_fingerprint=mcp_configuration_fingerprint,
        prior_knowledge=prior_knowledge,
    )


def _workspace_delta(
    payloads: dict[str, bytes],
    workspace_access: CodingAgentWorkspaceAccess,
    result: CodingAgentCallResult,
) -> CodingAgentWorkspaceDelta | None:
    if workspace_access is CodingAgentWorkspaceAccess.READ_ONLY:
        if result.workspace_delta_digest is not None:
            raise CodingAgentOperationReceiptError(
                "read-only coding-agent operation returned a workspace delta"
            )
        return None
    payload = payloads[CODING_AGENT_WORKSPACE_DELTA_FILENAME]
    if result.workspace_delta_digest != tree_or_blob_digest(payload):
        raise CodingAgentOperationReceiptError(
            "coding-agent workspace delta differs from returned result"
        )
    delta = CodingAgentWorkspaceDelta.from_json_bytes(payload)
    if payload != delta.to_json_bytes():
        raise CodingAgentOperationReceiptError(
            "coding-agent workspace delta is not canonical"
        )
    return delta


def _validate_prior_knowledge_artifact(
    payloads: dict[str, bytes],
    invocation: Mapping[str, Any],
) -> PriorKnowledgeAccessMaterialization | None:
    snapshot_id = invocation["prior_knowledge_snapshot_id"]
    materialization_digest = invocation["prior_knowledge_materialization_digest"]
    prior_payload = payloads["prior_knowledge.json"]
    if snapshot_id is None or materialization_digest is None:
        if (
            snapshot_id is not None
            or materialization_digest is not None
            or prior_payload != b"null\n"
        ):
            raise CodingAgentOperationReceiptError(
                "coding-agent empty prior-knowledge artifacts are inconsistent"
            )
        return None
    materialization = PriorKnowledgeAccessMaterialization.from_json_bytes(prior_payload)
    if (
        prior_payload != materialization.to_json_bytes()
        or snapshot_id
        != materialization.prior_knowledge_snapshot.prior_knowledge_snapshot_id
        or materialization_digest != materialization.materialization_digest
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent prior-knowledge artifacts are inconsistent"
        )
    return materialization


def _validate_mcp_configuration(
    payload: bytes,
    *,
    operation_id: str,
    prior_knowledge: PriorKnowledgeAccessMaterialization | None,
) -> str:
    configuration = json.loads(payload)
    if not isinstance(configuration, dict) or payload != (
        json.dumps(configuration, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8"):
        raise CodingAgentOperationReceiptError(
            "coding-agent MCP configuration artifact is invalid"
        )
    if set(configuration) != {"mcpServers"} or not isinstance(
        configuration["mcpServers"], dict
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent MCP configuration fields are invalid"
        )
    servers = configuration["mcpServers"]
    if prior_knowledge is None:
        if servers:
            raise CodingAgentOperationReceiptError(
                "coding-agent MCP configuration enables undeclared access"
            )
        return coding_agent_mcp_configuration_fingerprint(None)
    if set(servers) != {"prior_knowledge"}:
        raise CodingAgentOperationReceiptError(
            "coding-agent MCP configuration differs from prior-knowledge access"
        )
    server = servers["prior_knowledge"]
    if not isinstance(server, dict) or set(server) != {"args", "command"}:
        raise CodingAgentOperationReceiptError(
            "coding-agent prior-knowledge MCP server fields are invalid"
        )
    command = _normalized_absolute_path(
        server["command"],
        "coding-agent MCP command",
    )
    if command.name != "env":
        raise CodingAgentOperationReceiptError(
            "coding-agent prior-knowledge MCP command is not isolated"
        )
    arguments = server["args"]
    if (
        not isinstance(arguments, list)
        or len(arguments) != 17
        or any(not isinstance(argument, str) for argument in arguments)
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent prior-knowledge MCP arguments are invalid"
        )
    expected_fixed_arguments = {
        0: "-i",
        3: "-m",
        4: "kapso.gated_mcp.server",
        5: "--enabled-gates",
        6: "prior_knowledge",
        7: "--gate-failure-policy",
        8: "error",
        9: "--prior-knowledge-path",
        11: "--prior-knowledge-maximum-bytes",
        12: str(len(prior_knowledge.to_json_bytes())),
        13: "--prior-knowledge-audit-path",
        15: "--operation-id",
        16: operation_id,
    }
    if any(
        arguments[position] != expected
        for position, expected in expected_fixed_arguments.items()
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent prior-knowledge MCP authority is invalid"
        )
    if not arguments[1].startswith("PYTHONPATH="):
        raise CodingAgentOperationReceiptError(
            "coding-agent prior-knowledge MCP Python path is invalid"
        )
    _normalized_absolute_path(
        arguments[1].removeprefix("PYTHONPATH="),
        "coding-agent MCP Python path",
    )
    _normalized_absolute_path(arguments[2], "coding-agent MCP Python executable")
    prior_path = _normalized_absolute_path(
        arguments[10],
        "coding-agent prior-knowledge path",
    )
    audit_path = _normalized_absolute_path(
        arguments[14],
        "coding-agent prior-knowledge audit path",
    )
    if (
        prior_path.parent != audit_path.parent
        or prior_path.parent.name != operation_id
        or prior_path.name != "prior_knowledge.json"
        or audit_path.name != "mcp_audit.jsonl"
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent prior-knowledge MCP artifact paths are invalid"
        )
    return coding_agent_mcp_configuration_fingerprint(prior_knowledge)


def _normalized_absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise CodingAgentOperationReceiptError(f"{label} is invalid")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or ".." in path.parts
    ):
        raise CodingAgentOperationReceiptError(
            f"{label} must be absolute and normalized"
        )
    return path


def _read_private_artifact_directory(
    artifact_directory: Path,
    expected_names: tuple[str, ...],
) -> dict[str, bytes]:
    if artifact_directory.is_symlink() or not artifact_directory.is_dir():
        raise CodingAgentOperationReceiptError(
            "coding-agent artifact root must be a real directory"
        )
    with ExitStack() as descriptors:
        directory_descriptor = os.open(
            artifact_directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, directory_descriptor)
        directory_status = os.fstat(directory_descriptor)
        if directory_status.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
            raise CodingAgentOperationReceiptError(
                "coding-agent artifact root must be private"
            )
        if set(os.listdir(directory_descriptor)) != set(expected_names):
            raise CodingAgentOperationReceiptError(
                "coding-agent completed artifact closure is invalid"
            )
        payloads = {
            name: _read_private_artifact(directory_descriptor, name)
            for name in expected_names
        }
        current = os.stat(artifact_directory, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (
            directory_status.st_dev,
            directory_status.st_ino,
        ):
            raise CodingAgentOperationReceiptError(
                "coding-agent artifact root changed during sealing"
            )
    return payloads


def _read_private_artifact(directory_descriptor: int, name: str) -> bytes:
    expected = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(expected.st_mode)
        or expected.st_nlink != 1
        or expected.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
    ):
        raise CodingAgentOperationReceiptError(
            "coding-agent artifact must be a private independent file"
        )
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise CodingAgentOperationReceiptError(
                "coding-agent artifact changed while being opened"
            )
        payload = handle.read()
        completed = os.fstat(handle.fileno())
        if (
            completed.st_size != opened.st_size
            or completed.st_mtime_ns != opened.st_mtime_ns
            or completed.st_ctime_ns != opened.st_ctime_ns
            or len(payload) != opened.st_size
        ):
            raise CodingAgentOperationReceiptError(
                "coding-agent artifact changed while being read"
            )
    return payload
