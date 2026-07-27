"""Pure native-CLI commands and completion interpretation for run actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    freeze_json,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentRunActionRequest,
)
from kapso.cross_run.launch.run_action_coding_agent_schema import (
    validate_run_action_coding_agent_output,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)

_CODEX_EXECUTABLE = "/usr/bin/codex"
_CLAUDE_EXECUTABLE = "/usr/local/bin/claude"
_CODEX_VERSION_OUTPUT = b"codex-cli 0.144.1\n"
_CLAUDE_VERSION_OUTPUT = b"2.1.220 (Claude Code)\n"
_WORKSPACE_PATH = "/kapso/workspace"
_RESPONSE_SCHEMA_PATH = "/kapso/tmp/response.schema.json"
_PROVIDER_FINAL_PATH = "/kapso/tmp/provider.final.json"
_MCP_CONFIGURATION_PATH = "/kapso/tmp/mcp.config.json"
_PRIOR_KNOWLEDGE_PATH = "/kapso/tmp/prior_knowledge.json"
_PRIOR_KNOWLEDGE_AUDIT_PATH = "/kapso/tmp/prior_knowledge.audit.jsonl"
_PRIOR_KNOWLEDGE_MCP_EXECUTABLE = "/usr/local/bin/kapso-prior-knowledge-mcp"
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
_THREAD_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_NORMALIZED_DECIMAL_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)(?:[.][0-9]*[1-9])?$")


class RunActionCodingAgentCliError(RuntimeError):
    """A native coding-agent command or its completion evidence is invalid."""


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


def coding_agent_cli_command(
    request: CodingAgentRunActionRequest,
) -> tuple[str, ...]:
    """Return the sole argv admitted for the embedded native-CLI policy."""

    _require_request(request)
    if request.interpretation_policy.cli == "codex":
        return _codex_command(request)
    return _claude_command(request)


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
    if type(return_code) is not int or return_code != 0:
        raise RunActionCodingAgentCliError(
            "coding-agent CLI did not exit with exact success"
        )
    policy = request.interpretation_policy
    _require_bounded_payload(
        provider_output_payload,
        policy.maximum_provider_output_bytes,
        "provider output",
        allow_empty=False,
    )
    _require_bounded_payload(
        provider_diagnostic_payload,
        policy.maximum_provider_diagnostic_bytes,
        "provider diagnostic",
        allow_empty=True,
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
        _WORKSPACE_PATH,
        "--output-schema",
        _RESPONSE_SCHEMA_PATH,
        "--output-last-message",
        _PROVIDER_FINAL_PATH,
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
            '{HOME="/kapso/tmp/home",PATH="/usr/local/bin:/usr/bin:/bin"}'
        ),
    ]
    if request.prior_knowledge is not None:
        mcp_arguments = [
            "--prior-knowledge-path",
            _PRIOR_KNOWLEDGE_PATH,
            "--prior-knowledge-maximum-bytes",
            str(len(request.prior_knowledge.to_json_bytes())),
            "--prior-knowledge-audit-path",
            _PRIOR_KNOWLEDGE_AUDIT_PATH,
            "--operation-id",
            request.operation_id,
        ]
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
            ]
        )
    else:
        command.extend(["--config", "mcp_servers={}"])
    command.append("-")
    return tuple(command)


def _claude_command(request: CodingAgentRunActionRequest) -> tuple[str, ...]:
    policy = request.interpretation_policy
    editing = policy.workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
    tools = ["Glob", "Grep", "Read"]
    if editing:
        tools.extend(["Edit", "Write"])
    if policy.web_search_enabled:
        tools.append("WebSearch")
    prior_knowledge_tools = [
        "mcp__prior_knowledge__get_prior_knowledge_record",
        "mcp__prior_knowledge__list_prior_knowledge",
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
        _MCP_CONFIGURATION_PATH,
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
                    "allowRead": [_WORKSPACE_PATH],
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
    events = tuple(_require_event(parse_json_bytes(line)) for line in lines)
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
    completed_agent_messages = []
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
            if event["type"] == "item.completed" and item["type"] == "agent_message":
                if not isinstance(item.get("text"), str) or not item["text"].strip():
                    raise RunActionCodingAgentCliError(
                        "Codex completed agent message is invalid"
                    )
                completed_agent_messages.append(item["text"])
        else:
            raise RunActionCodingAgentCliError(
                "Codex successful turn contains an out-of-order lifecycle event"
            )
    if len(completed_agent_messages) != 1:
        raise RunActionCodingAgentCliError(
            "Codex completion requires one final agent message"
        )
    structured_output = _require_structured_output(
        parse_json_bytes(final_output_payload),
        request,
    )
    if parse_json_bytes(completed_agent_messages[0]) != structured_output:
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
    if not required_fields.issubset(envelope):
        raise RunActionCodingAgentCliError(
            "Claude success envelope is missing required evidence"
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
    structured_output = _require_structured_output(
        envelope["structured_output"],
        request,
    )
    if (
        not isinstance(envelope["result"], str)
        or parse_json_bytes(envelope["result"]) != structured_output
    ):
        raise RunActionCodingAgentCliError(
            "Claude result text differs from its structured output"
        )
    usage = _require_mapping(envelope["usage"], "Claude usage")
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
    if not model_usage:
        raise RunActionCodingAgentCliError("Claude model usage is empty")
    model_cost = Decimal(0)
    for model_name, model_evidence in model_usage.items():
        if not isinstance(model_name, str) or not model_name:
            raise RunActionCodingAgentCliError("Claude model identity is invalid")
        evidence = _require_mapping(model_evidence, "Claude model evidence")
        model_cost += _require_nonnegative_decimal(
            evidence.get("costUSD"),
            "Claude model cost",
        )
    if model_cost != total_cost:
        raise RunActionCodingAgentCliError(
            "Claude total cost differs from its model-usage evidence"
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
    "CodingAgentCliOutcome",
    "RunActionCodingAgentCliError",
    "coding_agent_cli_command",
    "coding_agent_cli_preflight_command",
    "interpret_coding_agent_cli_completion",
    "validate_coding_agent_cli_preflight",
]
