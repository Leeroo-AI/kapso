"""Pure native-CLI commands and completion interpretation for run actions."""

from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    freeze_json,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentPriorKnowledgeAccessEvent,
    CodingAgentPriorKnowledgeAccessKind,
    CodingAgentRunActionRequest,
)
from kapso.cross_run.launch.run_action_coding_agent_schema import (
    validate_run_action_coding_agent_output,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_coding_agent_layout import (
    coding_agent_provider_environment,
    PROVIDER_FINAL_PATH,
    PROVIDER_MCP_CONFIGURATION_PATH,
    PROVIDER_RESPONSE_SCHEMA_PATH,
    PROVIDER_WORKSPACE_PATH,
    TEMPORARY_ROOT_PATH,
)

_CODEX_EXECUTABLE = "/usr/bin/codex"
_CLAUDE_EXECUTABLE = "/usr/local/bin/claude"
_CODEX_VERSION_OUTPUT = b"codex-cli 0.144.1\n"
_CLAUDE_VERSION_OUTPUT = b"2.1.220 (Claude Code)\n"
_PRIOR_KNOWLEDGE_MCP_EXECUTABLE = "/usr/local/bin/kapso-provider-python"
_PRIOR_KNOWLEDGE_MCP_MODULE = (
    "kapso.cross_run.launch.run_action_coding_agent_prior_knowledge_relay"
)
_PRIOR_KNOWLEDGE_SOCKET_NAMESPACE = "kapso-prior-knowledge"
_PRIOR_KNOWLEDGE_SESSION_NAMESPACE = b"kapso.prior_knowledge_session.v1\x00"
_PRIOR_KNOWLEDGE_TOOL_NAMES = (
    "get_prior_knowledge_record",
    "list_prior_knowledge",
)
_CODEX_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "item.started",
        "item.updated",
        "item.completed",
        "error",
    }
)
_CODEX_ITEM_TYPES = frozenset(
    {
        "agent_message",
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "reasoning",
        "todo_list",
        "web_search",
    }
)
_THREAD_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_CODEX_ITEM_ID_PATTERN = re.compile(r"^item_(?:0|[1-9][0-9]*)$")
_CODEX_WEB_SEARCH_CALL_ID_PATTERN = re.compile(
    r"^exec-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_NORMALIZED_DECIMAL_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)(?:[.][0-9]*[1-9])?$")


class RunActionCodingAgentCliError(RuntimeError):
    """A native coding-agent command or its completion evidence is invalid."""


@dataclass(frozen=True)
class CodingAgentCliPriorKnowledgeCall:
    """One provider-reported prior-knowledge call joined to MCP response bytes."""

    tool_name: str
    arguments: Mapping[str, Any]
    response_digest: str

    def __post_init__(self) -> None:
        if self.tool_name not in _PRIOR_KNOWLEDGE_TOOL_NAMES:
            raise RunActionCodingAgentCliError(
                "coding-agent prior-knowledge call names an unknown tool"
            )
        object.__setattr__(
            self,
            "arguments",
            freeze_json(self.arguments, "coding-agent prior-knowledge arguments"),
        )
        if (
            not isinstance(self.response_digest, str)
            or _DIGEST_PATTERN.fullmatch(self.response_digest) is None
        ):
            raise RunActionCodingAgentCliError(
                "coding-agent prior-knowledge response digest is invalid"
            )


@dataclass(frozen=True)
class CodingAgentCliOutcome:
    """Provider-neutral success evidence from one exact native CLI completion."""

    structured_output: Mapping[str, Any]
    input_tokens: int
    cached_input_tokens: int | None
    output_tokens: int
    reasoning_output_tokens: int | None
    cost_usd: str | None
    provider_event_stream_digest: str
    provider_diagnostic_stream_digest: str
    prior_knowledge_calls: tuple[CodingAgentCliPriorKnowledgeCall, ...] | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "structured_output",
            freeze_json(self.structured_output, "coding-agent CLI structured output"),
        )
        for value, name in (
            (self.input_tokens, "input tokens"),
            (self.output_tokens, "output tokens"),
        ):
            _require_nonnegative_integer(value, name)
        for value, name in (
            (self.cached_input_tokens, "cached input tokens"),
            (self.reasoning_output_tokens, "reasoning output tokens"),
        ):
            if value is not None:
                _require_nonnegative_integer(value, name)
        if (
            self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise RunActionCodingAgentCliError(
                "coding-agent cached input tokens exceed total input tokens"
            )
        if (
            self.reasoning_output_tokens is not None
            and self.reasoning_output_tokens > self.output_tokens
        ):
            raise RunActionCodingAgentCliError(
                "coding-agent reasoning tokens exceed total output tokens"
            )
        if self.cost_usd is not None and (
            not isinstance(self.cost_usd, str)
            or _NORMALIZED_DECIMAL_PATTERN.fullmatch(self.cost_usd) is None
        ):
            raise RunActionCodingAgentCliError(
                "coding-agent CLI cost is not normalized decimal text"
            )
        for value, name in (
            (self.provider_event_stream_digest, "provider event stream"),
            (self.provider_diagnostic_stream_digest, "provider diagnostic stream"),
        ):
            if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
                raise RunActionCodingAgentCliError(f"{name} digest is invalid")
        if self.prior_knowledge_calls is not None and (
            type(self.prior_knowledge_calls) is not tuple
            or any(
                type(call) is not CodingAgentCliPriorKnowledgeCall
                for call in self.prior_knowledge_calls
            )
        ):
            raise RunActionCodingAgentCliError(
                "coding-agent prior-knowledge call trace is invalid"
            )


def coding_agent_cli_command(
    request: CodingAgentRunActionRequest,
) -> tuple[str, ...]:
    """Return the sole argv admitted for the embedded native-CLI policy."""

    _require_request(request)
    command = (
        _codex_command(request)
        if request.interpretation_policy.cli == "codex"
        else _claude_command(request)
    )
    encoded_size = sum(len(argument.encode("utf-8")) + 1 for argument in command)
    if encoded_size > request.interpretation_policy.maximum_cli_argument_bytes:
        raise RunActionCodingAgentCliError(
            "coding-agent CLI argv exceeds its exact byte limit"
        )
    return command


def coding_agent_cli_workspace_path() -> str:
    """Return the fixed workspace path used by both native CLIs."""

    return PROVIDER_WORKSPACE_PATH


def coding_agent_cli_provider_environment(
    request: CodingAgentRunActionRequest,
) -> Mapping[str, str]:
    """Return the complete ambient authority admitted to either native CLI."""

    _require_request(request)
    return coding_agent_provider_environment(
        request.interpretation_policy.egress_relay_port
    )


def coding_agent_cli_temporary_path() -> str:
    """Return the sole writable support/candidate directory."""

    return TEMPORARY_ROOT_PATH


def coding_agent_cli_final_output_path(
    request: CodingAgentRunActionRequest,
) -> str | None:
    """Return Codex's scratch final path; Claude returns its final on stdout."""

    _require_request(request)
    return PROVIDER_FINAL_PATH if request.interpretation_policy.cli == "codex" else None


def coding_agent_cli_support_payloads(
    request: CodingAgentRunActionRequest,
) -> Mapping[str, bytes]:
    """Return every immutable fixed-path support file required by one call."""

    _require_request(request)
    payloads = (
        {PROVIDER_RESPONSE_SCHEMA_PATH: canonical_json_bytes(request.response_schema)}
        if request.interpretation_policy.cli == "codex"
        else {PROVIDER_MCP_CONFIGURATION_PATH: _claude_mcp_configuration(request)}
    )
    return MappingProxyType(payloads)


def coding_agent_cli_prior_knowledge_socket_name(
    request: CodingAgentRunActionRequest,
) -> str:
    """Return the request-bound abstract Unix socket name for one MCP session."""

    _require_request(request)
    if request.prior_knowledge is None:
        raise RunActionCodingAgentCliError(
            "prior-knowledge socket requires one materialization"
        )
    return f"{_PRIOR_KNOWLEDGE_SOCKET_NAMESPACE}.{request.operation_id}"


def coding_agent_cli_prior_knowledge_session_token(
    request: CodingAgentRunActionRequest,
) -> str:
    """Return the request-bound authentication token for one MCP session."""

    _require_request(request)
    if request.prior_knowledge is None:
        raise RunActionCodingAgentCliError(
            "prior-knowledge session requires one materialization"
        )
    return hashlib.sha256(
        _PRIOR_KNOWLEDGE_SESSION_NAMESPACE + request.to_json_bytes()
    ).hexdigest()


def coding_agent_cli_preflight_command(
    request: CodingAgentRunActionRequest,
) -> tuple[str, ...]:
    """Return the exact no-provider command that attests the pinned CLI ABI."""

    _require_request(request)
    executable = (
        _CODEX_EXECUTABLE
        if request.interpretation_policy.cli == "codex"
        else _CLAUDE_EXECUTABLE
    )
    return (executable, "--version")


def validate_coding_agent_cli_preflight(
    *,
    request: CodingAgentRunActionRequest,
    return_code: int,
    output_payload: bytes,
    diagnostic_payload: bytes,
) -> None:
    """Require one exact executable/version observation before model spend."""

    _require_request(request)
    expected = (
        _CODEX_VERSION_OUTPUT
        if request.interpretation_policy.cli == "codex"
        else _CLAUDE_VERSION_OUTPUT
    )
    if (
        type(return_code) is not int
        or return_code != 0
        or type(output_payload) is not bytes
        or output_payload != expected
        or type(diagnostic_payload) is not bytes
        or diagnostic_payload
    ):
        raise RunActionCodingAgentCliError(
            "coding-agent CLI executable/version preflight failed"
        )


def interpret_coding_agent_cli_completion(
    *,
    request: CodingAgentRunActionRequest,
    return_code: int,
    provider_output_payload: bytes,
    provider_diagnostic_payload: bytes,
    final_output_payload: bytes | None,
) -> CodingAgentCliOutcome:
    """Validate bounded raw CLI evidence and return one semantic success."""

    _require_request(request)
    if type(return_code) is not int:
        raise RunActionCodingAgentCliError("coding-agent CLI status is invalid")
    policy = request.interpretation_policy
    _require_bounded_payload(
        provider_output_payload,
        policy.maximum_provider_output_bytes,
        "provider output",
        allow_empty=return_code != 0,
    )
    _require_bounded_payload(
        provider_diagnostic_payload,
        policy.maximum_provider_diagnostic_bytes,
        "provider diagnostic",
        allow_empty=True,
    )
    if return_code != 0:
        evidence = (
            _codex_failure_evidence(provider_output_payload)
            if policy.cli == "codex"
            else tree_or_blob_digest(provider_output_payload)
        )
        failure = canonical_json_bytes(
            {
                "diagnostic": provider_diagnostic_payload.decode("utf-8"),
                "event_evidence": evidence,
            }
        ).decode("utf-8")
        raise RunActionCodingAgentCliError(
            f"coding-agent CLI did not exit with exact success: {failure}"
        )
    if policy.cli == "codex":
        if final_output_payload is None:
            raise RunActionCodingAgentCliError(
                "Codex completion is missing its final output"
            )
        _require_bounded_payload(
            final_output_payload,
            policy.maximum_raw_result_bytes,
            "Codex final output",
            allow_empty=False,
        )
        return _interpret_codex_completion(
            request,
            provider_output_payload,
            provider_diagnostic_payload,
            final_output_payload,
        )
    if final_output_payload is not None:
        raise RunActionCodingAgentCliError(
            "Claude completion cannot receive a separate final output"
        )
    return _interpret_claude_completion(
        request,
        provider_output_payload,
        provider_diagnostic_payload,
    )


def _codex_failure_evidence(provider_output_payload: bytes) -> str:
    if not provider_output_payload:
        return tree_or_blob_digest(provider_output_payload)
    if not provider_output_payload.endswith(b"\n"):
        raise RunActionCodingAgentCliError(
            "Codex failure event stream has an incomplete final event"
        )
    lines = provider_output_payload.splitlines()
    if any(not line.strip() for line in lines):
        raise RunActionCodingAgentCliError(
            "Codex failure event stream contains a blank event"
        )
    events = tuple(_require_event(_parse_codex_event(line)) for line in lines)
    failures = tuple(
        event for event in events if event.get("type") in {"error", "turn.failed"}
    )
    return (
        canonical_json_bytes(failures).decode("utf-8")
        if failures
        else tree_or_blob_digest(provider_output_payload)
    )


def validate_coding_agent_cli_prior_knowledge_trace(
    *,
    request: CodingAgentRunActionRequest,
    outcome: CodingAgentCliOutcome,
    accesses: tuple[CodingAgentPriorKnowledgeAccessEvent, ...],
) -> None:
    """Join Codex's typed MCP items to the ordered authenticated audit."""

    _require_request(request)
    if type(outcome) is not CodingAgentCliOutcome or type(accesses) is not tuple:
        raise RunActionCodingAgentCliError(
            "coding-agent prior-knowledge trace inputs are invalid"
        )
    if any(
        type(access) is not CodingAgentPriorKnowledgeAccessEvent for access in accesses
    ):
        raise RunActionCodingAgentCliError(
            "coding-agent prior-knowledge audit trace is invalid"
        )
    if request.interpretation_policy.cli == "claude_code":
        if outcome.prior_knowledge_calls is not None:
            raise RunActionCodingAgentCliError(
                "Claude completion unexpectedly contains a Codex MCP trace"
            )
        return
    expected_calls = tuple(
        CodingAgentCliPriorKnowledgeCall(
            tool_name=(
                "list_prior_knowledge"
                if access.access_kind is CodingAgentPriorKnowledgeAccessKind.LIST
                else "get_prior_knowledge_record"
            ),
            arguments=(
                {}
                if access.access_kind is CodingAgentPriorKnowledgeAccessKind.LIST
                else {"record_id": access.record_id}
            ),
            response_digest=access.response_digest,
        )
        for access in accesses
    )
    if outcome.prior_knowledge_calls != expected_calls:
        raise RunActionCodingAgentCliError(
            "Codex MCP event trace differs from the ordered prior-knowledge audit"
        )


def _codex_command(request: CodingAgentRunActionRequest) -> tuple[str, ...]:
    policy = request.interpretation_policy
    command = [
        _CODEX_EXECUTABLE,
        "--ask-for-approval",
        "never",
        "exec",
        "--strict-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--cd",
        PROVIDER_WORKSPACE_PATH,
        "--output-schema",
        PROVIDER_RESPONSE_SCHEMA_PATH,
        "--output-last-message",
        PROVIDER_FINAL_PATH,
        "--json",
        "--color",
        "never",
        "--model",
        policy.model,
        "--sandbox",
        (
            "workspace-write"
            if policy.workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
            else "read-only"
        ),
        "--config",
        f'model_reasoning_effort="{policy.effort}"',
        "--config",
        f'web_search="{"live" if policy.web_search_enabled else "disabled"}"',
        "--config",
        'shell_environment_policy.inherit="none"',
        "--config",
        "shell_environment_policy.ignore_default_excludes=false",
        "--config",
        (
            "shell_environment_policy.set="
            + "{"
            + ",".join(
                f"{key}={json.dumps(value)}"
                for key, value in sorted(
                    coding_agent_cli_provider_environment(request).items()
                )
            )
            + "}"
        ),
    ]
    if request.prior_knowledge is not None:
        mcp_arguments = _prior_knowledge_mcp_arguments(request)
        command.extend(
            [
                "--config",
                (
                    "mcp_servers.prior_knowledge.command="
                    + json.dumps(_PRIOR_KNOWLEDGE_MCP_EXECUTABLE)
                ),
                "--config",
                (
                    "mcp_servers.prior_knowledge.args="
                    + json.dumps(mcp_arguments, separators=(",", ":"))
                ),
                "--config",
                "mcp_servers.prior_knowledge.required=true",
                "--config",
                'mcp_servers.prior_knowledge.default_tools_approval_mode="approve"',
                "--config",
                (
                    "mcp_servers.prior_knowledge.enabled_tools="
                    + json.dumps(_PRIOR_KNOWLEDGE_TOOL_NAMES, separators=(",", ":"))
                ),
            ]
        )
    else:
        command.extend(["--config", "mcp_servers={}"])
    command.append("-")
    return tuple(command)


def _claude_mcp_configuration(request: CodingAgentRunActionRequest) -> bytes:
    servers = {}
    if request.prior_knowledge is not None:
        servers["prior_knowledge"] = {
            "command": _PRIOR_KNOWLEDGE_MCP_EXECUTABLE,
            "args": _prior_knowledge_mcp_arguments(request),
        }
    return canonical_json_bytes({"mcpServers": servers})


def _prior_knowledge_mcp_arguments(
    request: CodingAgentRunActionRequest,
) -> list[str]:
    if request.prior_knowledge is None:
        raise RunActionCodingAgentCliError(
            "prior-knowledge MCP arguments require one materialization"
        )
    return [
        "-m",
        _PRIOR_KNOWLEDGE_MCP_MODULE,
        "--socket-name",
        coding_agent_cli_prior_knowledge_socket_name(request),
        "--session-token",
        coding_agent_cli_prior_knowledge_session_token(request),
        "--chunk-size-bytes",
        str(request.interpretation_policy.prior_knowledge_relay_chunk_size_bytes),
        "--provider-user-id",
        str(request.interpretation_policy.provider_user_id),
        "--provider-group-id",
        str(request.interpretation_policy.provider_group_id),
        "--sidecar-user-id",
        str(request.interpretation_policy.supervisor_user_id),
        "--sidecar-group-id",
        str(request.interpretation_policy.supervisor_group_id),
    ]


def _claude_command(request: CodingAgentRunActionRequest) -> tuple[str, ...]:
    policy = request.interpretation_policy
    editing = policy.workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
    tools = ["Glob", "Grep", "Read"]
    if editing:
        tools.extend(["Edit", "Write"])
    if policy.web_search_enabled:
        tools.append("WebSearch")
    prior_knowledge_tools = [
        f"mcp__prior_knowledge__{tool_name}"
        for tool_name in _PRIOR_KNOWLEDGE_TOOL_NAMES
    ]
    if request.prior_knowledge is not None:
        tools.extend(prior_knowledge_tools)
    denied = [
        "Bash",
        "NotebookEdit",
        "Read(//proc/**)",
        "Read(/proc/**)",
        "Read(**/.env)",
        "Read(**/.env.*)",
    ]
    if not editing:
        denied.extend(["Edit", "Write"])
    else:
        denied.extend(["Edit(**/.git)", "Edit(**/.git/**)"])
    command = [
        _CLAUDE_EXECUTABLE,
        "--print",
        "--safe-mode",
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--settings",
        _claude_security_settings(editing),
        "--strict-mcp-config",
        "--mcp-config",
        PROVIDER_MCP_CONFIGURATION_PATH,
        "--permission-mode",
        "acceptEdits" if editing else "plan",
        "--no-session-persistence",
        "--input-format",
        "text",
        "--output-format",
        "json",
        "--json-schema",
        canonical_json_bytes(request.response_schema).decode("utf-8"),
        "--model",
        policy.model,
        "--effort",
        policy.effort,
        "--disallowedTools",
        ",".join(denied),
        "--tools",
        ",".join(tools),
    ]
    if request.prior_knowledge is not None:
        command.extend(["--allowedTools", ",".join(prior_knowledge_tools)])
    return tuple(command)


def _claude_security_settings(editing: bool) -> str:
    denied = [
        "Read(//proc/**)",
        "Read(/proc/**)",
        "Read(**/.env)",
        "Read(**/.env.*)",
    ]
    if editing:
        denied.extend(["Edit(**/.git)", "Edit(**/.git/**)"])
    return canonical_json_bytes(
        {
            "permissions": {"deny": denied},
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "filesystem": {
                    "denyRead": ["/"],
                    "allowRead": [PROVIDER_WORKSPACE_PATH],
                },
            },
        }
    ).decode("utf-8")


def _interpret_codex_completion(
    request: CodingAgentRunActionRequest,
    provider_output_payload: bytes,
    provider_diagnostic_payload: bytes,
    final_output_payload: bytes,
) -> CodingAgentCliOutcome:
    if not provider_output_payload.endswith(b"\n"):
        raise RunActionCodingAgentCliError(
            "Codex event stream has an incomplete final event"
        )
    lines = provider_output_payload.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise RunActionCodingAgentCliError(
            "Codex event stream is empty or contains a blank event"
        )
    events = tuple(_require_event(_parse_codex_event(line)) for line in lines)
    event_types = tuple(event["type"] for event in events)
    if any(event_type not in _CODEX_EVENT_TYPES for event_type in event_types):
        raise RunActionCodingAgentCliError(
            "Codex event stream contains an unknown event type"
        )
    if "turn.failed" in event_types or "error" in event_types:
        raise RunActionCodingAgentCliError(
            "Codex event stream contains a terminal failure"
        )
    if (
        event_types.count("thread.started") != 1
        or event_types[0] != "thread.started"
        or event_types.count("turn.started") != 1
        or event_types[1] != "turn.started"
        or event_types.count("turn.completed") != 1
        or event_types[-1] != "turn.completed"
    ):
        raise RunActionCodingAgentCliError(
            "Codex event stream lacks one exact ordered successful turn"
        )
    thread = events[0]
    if (
        set(thread) != {"type", "thread_id"}
        or not isinstance(thread["thread_id"], str)
        or _THREAD_ID_PATTERN.fullmatch(thread["thread_id"]) is None
        or set(events[1]) != {"type"}
    ):
        raise RunActionCodingAgentCliError(
            "Codex thread or turn-start evidence is invalid"
        )
    completion = events[-1]
    if set(completion) != {"type", "usage"}:
        raise RunActionCodingAgentCliError("Codex completion fields are invalid")
    usage = _require_mapping(completion["usage"], "Codex usage")
    expected_usage_fields = {
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    }
    if set(usage) != expected_usage_fields:
        raise RunActionCodingAgentCliError("Codex usage fields are invalid")
    input_tokens = _require_nonnegative_integer(
        usage["input_tokens"],
        "Codex input tokens",
    )
    cached_input_tokens = _require_nonnegative_integer(
        usage["cached_input_tokens"],
        "Codex cached input tokens",
    )
    output_tokens = _require_nonnegative_integer(
        usage["output_tokens"],
        "Codex output tokens",
    )
    reasoning_output_tokens = _require_nonnegative_integer(
        usage["reasoning_output_tokens"],
        "Codex reasoning output tokens",
    )
    if cached_input_tokens > input_tokens or reasoning_output_tokens > output_tokens:
        raise RunActionCodingAgentCliError(
            "Codex usage contains an impossible token decomposition"
        )
    completed_agent_messages = []
    prior_knowledge_calls = []
    item_states = {}
    next_item_number = 0
    for event in events[2:-1]:
        if event["type"].startswith("item."):
            if set(event) != {"type", "item"}:
                raise RunActionCodingAgentCliError(
                    "Codex item event fields are invalid"
                )
            item = _require_mapping(event["item"], "Codex item")
            if not isinstance(item.get("id"), str) or not isinstance(
                item.get("type"), str
            ):
                raise RunActionCodingAgentCliError("Codex item identity is invalid")
            if item["type"] not in _CODEX_ITEM_TYPES:
                raise RunActionCodingAgentCliError(
                    "Codex item type is unknown or reports a model reroute"
                )
            if (
                item["type"] == "web_search"
                and not request.interpretation_policy.web_search_enabled
            ):
                raise RunActionCodingAgentCliError(
                    "Codex used web search without request authority"
                )
            if (
                item["type"] == "file_change"
                and request.interpretation_policy.workspace_access
                is not RunFrontierWorkspaceAccess.EDIT_WORKSPACE
            ):
                raise RunActionCodingAgentCliError(
                    "Codex reported a file change without edit authority"
                )
            if item["type"] == "mcp_tool_call" and request.prior_knowledge is None:
                raise RunActionCodingAgentCliError(
                    "Codex used MCP without prior-knowledge authority"
                )
            item_id = item["id"]
            if _CODEX_ITEM_ID_PATTERN.fullmatch(item_id) is None:
                raise RunActionCodingAgentCliError("Codex item ID is invalid")
            prior_knowledge_call = _require_codex_item_fields(event["type"], item)
            lifecycle_identity = _codex_item_lifecycle_identity(item)
            existing_state = item_states.get(item_id)
            if existing_state is None:
                if item_id != f"item_{next_item_number}":
                    raise RunActionCodingAgentCliError(
                        "Codex item IDs are not exact contiguous wire identities"
                    )
                next_item_number += 1
                if event["type"] == "item.updated":
                    raise RunActionCodingAgentCliError(
                        "Codex item lifecycle starts with an update"
                    )
            elif (
                existing_state[0] != item["type"]
                or existing_state[1]
                or existing_state[2] != lifecycle_identity
                or event["type"] == "item.started"
            ):
                raise RunActionCodingAgentCliError(
                    "Codex item lifecycle changes identity or reopens a terminal item"
                )
            if event["type"] == "item.updated" and item["type"] != "todo_list":
                raise RunActionCodingAgentCliError(
                    "Codex updated an item type without an update lifecycle"
                )
            if prior_knowledge_call is not None:
                prior_knowledge_calls.append(prior_knowledge_call)
            item_states[item_id] = (
                item["type"],
                event["type"] == "item.completed",
                lifecycle_identity,
            )
            if event["type"] == "item.completed" and item["type"] == "agent_message":
                completed_agent_messages.append(item["text"])
        else:
            raise RunActionCodingAgentCliError(
                "Codex successful turn contains an out-of-order lifecycle event"
            )
    if any(
        not completed
        for _item_type, completed, _lifecycle_identity in item_states.values()
    ):
        raise RunActionCodingAgentCliError(
            "Codex successful turn contains an unterminated item"
        )
    if not completed_agent_messages:
        raise RunActionCodingAgentCliError(
            "Codex completion requires a final agent message"
        )
    structured_output = _require_structured_output(
        parse_json_bytes(final_output_payload),
        request,
    )
    structured_output_payload = canonical_json_bytes(structured_output)
    if (
        completed_agent_messages[-1].encode("utf-8") != structured_output_payload
        or final_output_payload != structured_output_payload
    ):
        raise RunActionCodingAgentCliError(
            "Codex final artifact differs from its completed agent message"
        )
    return CodingAgentCliOutcome(
        structured_output=structured_output,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        cost_usd=None,
        provider_event_stream_digest=tree_or_blob_digest(provider_output_payload),
        provider_diagnostic_stream_digest=tree_or_blob_digest(
            provider_diagnostic_payload
        ),
        prior_knowledge_calls=tuple(prior_knowledge_calls),
    )


def _interpret_claude_completion(
    request: CodingAgentRunActionRequest,
    provider_output_payload: bytes,
    provider_diagnostic_payload: bytes,
) -> CodingAgentCliOutcome:
    envelope = _require_mapping(
        parse_json_bytes(provider_output_payload),
        "Claude result",
    )
    decimal_envelope = _require_mapping(
        _parse_decimal_json(provider_output_payload),
        "Claude decimal result",
    )
    required_fields = {
        "is_error",
        "modelUsage",
        "num_turns",
        "permission_denials",
        "result",
        "session_id",
        "stop_reason",
        "structured_output",
        "subtype",
        "total_cost_usd",
        "type",
        "usage",
    }
    optional_fields = {
        "api_error_status",
        "duration_api_ms",
        "terminal_reason",
    }
    if (
        not required_fields.issubset(envelope)
        or not set(envelope).issubset(required_fields | optional_fields)
        or set(decimal_envelope) != set(envelope)
    ):
        raise RunActionCodingAgentCliError(
            "Claude success envelope fields differ from the pinned ABI"
        )
    if (
        envelope["type"] != "result"
        or envelope["subtype"] != "success"
        or envelope["is_error"] is not False
        or envelope["stop_reason"] != "tool_use"
        or (
            "terminal_reason" in envelope and envelope["terminal_reason"] != "completed"
        )
        or ("api_error_status" in envelope and envelope["api_error_status"] is not None)
        or not isinstance(envelope["session_id"], str)
        or _THREAD_ID_PATTERN.fullmatch(envelope["session_id"]) is None
        or _require_positive_integer(envelope["num_turns"], "Claude turn count") < 1
        or not isinstance(envelope["permission_denials"], list)
        or envelope["permission_denials"]
    ):
        raise RunActionCodingAgentCliError(
            "Claude result lacks one exact successful terminal state"
        )
    if "duration_api_ms" in envelope:
        _require_nonnegative_integer(
            envelope["duration_api_ms"],
            "Claude API duration milliseconds",
        )
    structured_output = _require_structured_output(
        envelope["structured_output"],
        request,
    )
    if not isinstance(envelope["result"], str) or envelope["result"].encode(
        "utf-8"
    ) != canonical_json_bytes(structured_output):
        raise RunActionCodingAgentCliError(
            "Claude result text differs from its structured output"
        )
    usage = _require_mapping(envelope["usage"], "Claude usage")
    if set(usage) != {
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "input_tokens",
        "output_tokens",
        "server_tool_use",
    }:
        raise RunActionCodingAgentCliError("Claude usage fields are invalid")
    input_tokens = _require_nonnegative_integer(
        usage.get("input_tokens"),
        "Claude input tokens",
    )
    cache_creation_input_tokens = _require_nonnegative_integer(
        usage.get("cache_creation_input_tokens"),
        "Claude cache-creation input tokens",
    )
    cache_read_input_tokens = _require_nonnegative_integer(
        usage.get("cache_read_input_tokens"),
        "Claude cache-read input tokens",
    )
    output_tokens = _require_nonnegative_integer(
        usage.get("output_tokens"),
        "Claude output tokens",
    )
    server_tool_use = _require_mapping(
        usage.get("server_tool_use"),
        "Claude server-tool usage",
    )
    if set(server_tool_use) != {"web_fetch_requests", "web_search_requests"}:
        raise RunActionCodingAgentCliError(
            "Claude server-tool usage fields are invalid"
        )
    web_search_requests = _require_nonnegative_integer(
        server_tool_use.get("web_search_requests"),
        "Claude web-search requests",
    )
    web_fetch_requests = _require_nonnegative_integer(
        server_tool_use.get("web_fetch_requests"),
        "Claude web-fetch requests",
    )
    if web_fetch_requests:
        raise RunActionCodingAgentCliError(
            "Claude used web fetch without native-tool authority"
        )
    if not request.interpretation_policy.web_search_enabled and web_search_requests:
        raise RunActionCodingAgentCliError(
            "Claude used web search without request authority"
        )
    total_cost = _require_nonnegative_decimal(
        decimal_envelope["total_cost_usd"],
        "Claude total cost",
    )
    model_usage = _require_mapping(
        decimal_envelope["modelUsage"],
        "Claude model usage",
    )
    if set(model_usage) != {request.interpretation_policy.model}:
        raise RunActionCodingAgentCliError(
            "Claude model usage differs from the sole requested model"
        )
    model_cost = Decimal(0)
    model_input_tokens = 0
    model_cache_creation_input_tokens = 0
    model_cache_read_input_tokens = 0
    model_output_tokens = 0
    model_web_search_requests = 0
    for model_name, model_evidence in model_usage.items():
        if not isinstance(model_name, str) or not model_name:
            raise RunActionCodingAgentCliError("Claude model identity is invalid")
        evidence = _require_mapping(model_evidence, "Claude model evidence")
        if set(evidence) != {
            "cacheCreationInputTokens",
            "cacheReadInputTokens",
            "contextWindow",
            "costUSD",
            "inputTokens",
            "maxOutputTokens",
            "outputTokens",
            "webSearchRequests",
        }:
            raise RunActionCodingAgentCliError(
                "Claude model-usage fields differ from the pinned ABI"
            )
        model_input_tokens += _require_nonnegative_integer(
            evidence["inputTokens"],
            "Claude model input tokens",
        )
        model_cache_creation_input_tokens += _require_nonnegative_integer(
            evidence["cacheCreationInputTokens"],
            "Claude model cache-creation input tokens",
        )
        model_cache_read_input_tokens += _require_nonnegative_integer(
            evidence["cacheReadInputTokens"],
            "Claude model cache-read input tokens",
        )
        model_output_tokens += _require_nonnegative_integer(
            evidence["outputTokens"],
            "Claude model output tokens",
        )
        model_web_search_requests += _require_nonnegative_integer(
            evidence["webSearchRequests"],
            "Claude model web-search requests",
        )
        _require_positive_integer(
            evidence["contextWindow"],
            "Claude model context window",
        )
        _require_positive_integer(
            evidence["maxOutputTokens"],
            "Claude model maximum output tokens",
        )
        model_cost += _require_nonnegative_decimal(
            evidence["costUSD"],
            "Claude model cost",
        )
    if (
        model_cost != total_cost
        or model_input_tokens != input_tokens
        or model_cache_creation_input_tokens != cache_creation_input_tokens
        or model_cache_read_input_tokens != cache_read_input_tokens
        or model_output_tokens != output_tokens
        or model_web_search_requests != web_search_requests
    ):
        raise RunActionCodingAgentCliError(
            "Claude totals differ from model-usage evidence"
        )
    return CodingAgentCliOutcome(
        structured_output=structured_output,
        input_tokens=(
            input_tokens + cache_creation_input_tokens + cache_read_input_tokens
        ),
        cached_input_tokens=cache_read_input_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=None,
        cost_usd=_normalize_decimal(total_cost),
        provider_event_stream_digest=tree_or_blob_digest(provider_output_payload),
        provider_diagnostic_stream_digest=tree_or_blob_digest(
            provider_diagnostic_payload
        ),
        prior_knowledge_calls=None,
    )


def _require_request(
    request: CodingAgentRunActionRequest,
) -> CodingAgentRunActionRequest:
    if type(request) is not CodingAgentRunActionRequest:
        raise RunActionCodingAgentCliError(
            "coding-agent CLI requires an exact run-action request"
        )
    request.require_policy(request.interpretation_policy)
    return request


def _require_bounded_payload(
    payload: bytes,
    maximum_bytes: int,
    name: str,
    *,
    allow_empty: bool,
) -> None:
    if (
        type(payload) is not bytes
        or (not allow_empty and not payload)
        or len(payload) > maximum_bytes
    ):
        raise RunActionCodingAgentCliError(f"{name} is empty, invalid, or oversized")


def _require_event(value: Any) -> Mapping[str, Any]:
    event = _require_mapping(value, "Codex event")
    if not isinstance(event.get("type"), str):
        raise RunActionCodingAgentCliError("Codex event type is invalid")
    return event


def _parse_codex_event(payload: bytes) -> Any:
    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_codex_object,
        parse_float=_parse_finite_provider_float,
        parse_constant=_reject_non_finite_decimal,
    )


def _strict_codex_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = tuple(key for key, _value in pairs)
    if keys == ("id", "type", "id", "query", "action") and pairs[1][1] == "web_search":
        return {
            "id": pairs[0][1],
            "type": pairs[1][1],
            "web_search_call_id": pairs[2][1],
            "query": pairs[3][1],
            "action": pairs[4][1],
        }
    result = {}
    for key, value in pairs:
        if key in result:
            raise RunActionCodingAgentCliError(
                f"duplicate Codex event object key: {key}"
            )
        result[key] = value
    return result


def _parse_finite_provider_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise RunActionCodingAgentCliError(f"non-finite provider JSON number: {token}")
    return value


def _require_codex_item_fields(
    event_type: str,
    item: Mapping[str, Any],
) -> CodingAgentCliPriorKnowledgeCall | None:
    item_type = item["type"]
    if item_type in {"agent_message", "reasoning"}:
        if (
            event_type != "item.completed"
            or set(item) != {"id", "text", "type"}
            or not isinstance(item["text"], str)
            or not item["text"].strip()
        ):
            raise RunActionCodingAgentCliError(
                f"Codex {item_type} item fields or lifecycle are invalid"
            )
        return None
    if item_type == "command_execution":
        _require_codex_command_item(event_type, item)
        return None
    if item_type == "file_change":
        _require_codex_file_change_item(event_type, item)
        return None
    if item_type == "mcp_tool_call":
        return _require_codex_mcp_item(event_type, item)
    if item_type == "web_search":
        _require_codex_web_search_item(event_type, item)
        return None
    if item_type == "todo_list":
        _require_codex_todo_item(event_type, item)
        return None
    raise RunActionCodingAgentCliError("Codex item type lacks a pinned schema")


def _codex_item_lifecycle_identity(item: Mapping[str, Any]) -> bytes | str | None:
    item_type = item["type"]
    if item_type == "command_execution":
        return item["command"]
    if item_type == "mcp_tool_call":
        return canonical_json_bytes(
            {
                "arguments": item["arguments"],
                "server": item["server"],
                "tool": item["tool"],
            }
        )
    if item_type == "web_search":
        return item["web_search_call_id"]
    return None


def _require_codex_command_item(
    event_type: str,
    item: Mapping[str, Any],
) -> None:
    if (
        set(item)
        != {
            "aggregated_output",
            "command",
            "exit_code",
            "id",
            "status",
            "type",
        }
        or not isinstance(item["command"], str)
        or not item["command"]
        or not isinstance(item["aggregated_output"], str)
        or (item["exit_code"] is not None and type(item["exit_code"]) is not int)
    ):
        raise RunActionCodingAgentCliError(
            "Codex command-execution item fields are invalid"
        )
    if event_type == "item.started":
        valid = (
            item["status"] == "in_progress"
            and item["exit_code"] is None
            and item["aggregated_output"] == ""
        )
    elif event_type == "item.completed":
        valid = (
            item["status"] == "completed"
            and item["exit_code"] == 0
            or item["status"] == "failed"
            and type(item["exit_code"]) is int
            and item["exit_code"] != 0
            or item["status"] == "declined"
            and item["exit_code"] is None
        )
    else:
        valid = False
    if not valid:
        raise RunActionCodingAgentCliError(
            "Codex command-execution lifecycle is invalid"
        )


def _require_codex_file_change_item(
    event_type: str,
    item: Mapping[str, Any],
) -> None:
    if (
        event_type != "item.completed"
        or set(item) != {"changes", "id", "status", "type"}
        or item["status"] not in {"completed", "failed"}
        or not isinstance(item["changes"], list)
    ):
        raise RunActionCodingAgentCliError(
            "Codex file-change item fields or lifecycle are invalid"
        )
    for change in item["changes"]:
        if not isinstance(change, Mapping) or set(change) != {"kind", "path"}:
            raise RunActionCodingAgentCliError(
                "Codex file-change entry fields are invalid"
            )
        path = change["path"]
        parsed_path = PurePosixPath(path) if isinstance(path, str) else None
        if (
            parsed_path is None
            or not path
            or parsed_path.is_absolute()
            or parsed_path.as_posix() != path
            or ".." in parsed_path.parts
            or parsed_path.parts[0] == ".git"
            or change["kind"] not in {"add", "delete", "update"}
        ):
            raise RunActionCodingAgentCliError(
                "Codex file-change entry is invalid or escapes source authority"
            )


def _require_codex_mcp_item(
    event_type: str,
    item: Mapping[str, Any],
) -> CodingAgentCliPriorKnowledgeCall | None:
    if (
        set(item)
        != {
            "arguments",
            "error",
            "id",
            "result",
            "server",
            "status",
            "tool",
            "type",
        }
        or item["server"] != "prior_knowledge"
        or item["tool"] not in _PRIOR_KNOWLEDGE_TOOL_NAMES
        or not isinstance(item["arguments"], Mapping)
    ):
        raise RunActionCodingAgentCliError(
            "Codex MCP item differs from the prior-knowledge authority"
        )
    if item["tool"] == "list_prior_knowledge":
        arguments_valid = not item["arguments"]
    else:
        arguments_valid = set(item["arguments"]) == {"record_id"} and isinstance(
            item["arguments"]["record_id"], str
        )
    if not arguments_valid:
        raise RunActionCodingAgentCliError("Codex MCP item arguments are invalid")
    if event_type == "item.started":
        valid = (
            item["status"] == "in_progress"
            and item["result"] is None
            and item["error"] is None
        )
        response_digest = None
    elif event_type == "item.completed":
        response_digest = _codex_mcp_response_digest(item["result"])
        valid = (
            item["status"] == "completed"
            and item["error"] is None
            and response_digest is not None
        )
    else:
        valid = False
    if not valid:
        result = item["result"]
        result_fields = (
            tuple(sorted(result))
            if isinstance(result, Mapping)
            else type(result).__name__
        )
        structured_content_type = (
            type(result.get("structured_content")).__name__
            if isinstance(result, Mapping)
            else None
        )
        raise RunActionCodingAgentCliError(
            "Codex MCP item did not complete through the exact success lifecycle: "
            f"event={event_type!r}, status={item['status']!r}, "
            f"error={item['error']!r}, result_fields={result_fields!r}, "
            f"structured_content_type={structured_content_type!r}"
        )
    if response_digest is None:
        return None
    return CodingAgentCliPriorKnowledgeCall(
        tool_name=item["tool"],
        arguments=item["arguments"],
        response_digest=response_digest,
    )


def _codex_mcp_response_digest(result: Any) -> str | None:
    if not isinstance(result, Mapping) or set(result) not in (
        {"content", "structured_content"},
        {"_meta", "content", "structured_content"},
    ):
        return None
    if result["structured_content"] is not None:
        return None
    content = result["content"]
    if not isinstance(content, list) or len(content) != 1:
        return None
    block = content[0]
    if (
        not isinstance(block, Mapping)
        or set(block) != {"text", "type"}
        or block["type"] != "text"
        or not isinstance(block["text"], str)
    ):
        return None
    response_payload = block["text"].encode("utf-8")
    if canonical_json_bytes(parse_json_bytes(response_payload)) != response_payload:
        return None
    return tree_or_blob_digest(response_payload)


def _require_codex_web_search_item(
    event_type: str,
    item: Mapping[str, Any],
) -> None:
    if (
        event_type not in {"item.started", "item.completed"}
        or set(item) != {"action", "id", "query", "type", "web_search_call_id"}
        or not isinstance(item["query"], str)
        or not isinstance(item["web_search_call_id"], str)
        or _CODEX_WEB_SEARCH_CALL_ID_PATTERN.fullmatch(item["web_search_call_id"])
        is None
        or not isinstance(item["action"], Mapping)
    ):
        raise RunActionCodingAgentCliError(
            "Codex web-search item fields or lifecycle are invalid"
        )
    action = item["action"]
    action_type = action.get("type")
    if action_type == "other":
        valid = set(action) == {"type"}
    elif action_type == "search":
        valid = set(action) in (
            {"query", "type"},
            {"queries", "type"},
            {"queries", "query", "type"},
        ) and (
            ("query" not in action or isinstance(action["query"], str))
            and (
                "queries" not in action
                or isinstance(action["queries"], list)
                and all(isinstance(query, str) for query in action["queries"])
            )
        )
    elif action_type == "open_page":
        valid = set(action) in ({"type"}, {"type", "url"}) and (
            "url" not in action or isinstance(action["url"], str)
        )
    elif action_type == "find_in_page":
        valid = (
            set(action).issubset({"pattern", "type", "url"})
            and set(action).issuperset({"type"})
            and all(isinstance(action[field], str) for field in set(action) - {"type"})
        )
    else:
        valid = False
    if not valid:
        raise RunActionCodingAgentCliError("Codex web-search action is invalid")


def _require_codex_todo_item(
    event_type: str,
    item: Mapping[str, Any],
) -> None:
    if (
        event_type not in {"item.started", "item.updated", "item.completed"}
        or set(item) != {"id", "items", "type"}
        or not isinstance(item["items"], list)
    ):
        raise RunActionCodingAgentCliError(
            "Codex to-do item fields or lifecycle are invalid"
        )
    for todo in item["items"]:
        if (
            not isinstance(todo, Mapping)
            or set(todo) != {"completed", "text"}
            or type(todo["completed"]) is not bool
            or not isinstance(todo["text"], str)
            or not todo["text"].strip()
        ):
            raise RunActionCodingAgentCliError("Codex to-do entry is invalid")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunActionCodingAgentCliError(f"{name} must be an object")
    return value


def _require_structured_output(
    value: Any,
    request: CodingAgentRunActionRequest,
) -> Mapping[str, Any]:
    output = _require_mapping(value, "coding-agent structured output")
    validate_run_action_coding_agent_output(request.response_schema, output)
    return MappingProxyType(dict(output))


def _require_nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RunActionCodingAgentCliError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RunActionCodingAgentCliError(f"{name} must be a positive integer")
    return value


def _strict_decimal_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RunActionCodingAgentCliError(
                f"duplicate provider JSON object key: {key}"
            )
        result[key] = value
    return result


def _reject_non_finite_decimal(token: str) -> None:
    raise RunActionCodingAgentCliError(f"non-finite provider JSON number: {token}")


def _parse_decimal_json(payload: bytes) -> Any:
    text = payload.decode("utf-8")
    return json.loads(
        text,
        object_pairs_hook=_strict_decimal_object,
        parse_float=Decimal,
        parse_constant=_reject_non_finite_decimal,
    )


def _require_nonnegative_decimal(value: Any, name: str) -> Decimal:
    if type(value) is int:
        decimal_value = Decimal(value)
    elif type(value) is Decimal:
        decimal_value = value
    else:
        raise RunActionCodingAgentCliError(f"{name} must be a JSON number")
    if not decimal_value.is_finite() or decimal_value < 0:
        raise RunActionCodingAgentCliError(
            f"{name} must be a finite non-negative number"
        )
    return decimal_value


def _normalize_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if not text or Decimal(text) == 0 else text


__all__ = [
    "CodingAgentCliPriorKnowledgeCall",
    "CodingAgentCliOutcome",
    "RunActionCodingAgentCliError",
    "coding_agent_cli_command",
    "coding_agent_cli_final_output_path",
    "coding_agent_cli_preflight_command",
    "coding_agent_cli_provider_environment",
    "coding_agent_cli_prior_knowledge_session_token",
    "coding_agent_cli_prior_knowledge_socket_name",
    "coding_agent_cli_support_payloads",
    "coding_agent_cli_temporary_path",
    "coding_agent_cli_workspace_path",
    "interpret_coding_agent_cli_completion",
    "validate_coding_agent_cli_preflight",
    "validate_coding_agent_cli_prior_knowledge_trace",
]
