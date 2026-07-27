import json

import pytest

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.launch.run_action_coding_agent_cli import (
    RunActionCodingAgentCliError,
    coding_agent_cli_command,
    coding_agent_cli_preflight_command,
    interpret_coding_agent_cli_completion,
    validate_coding_agent_cli_preflight,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from test_run_action_coding_agent_contracts import (
    empty_prior_knowledge,
    interpretation_policy,
    run_action_request,
)

_THREAD_ID = "0199a213-81c0-7800-8aa1-bbab2a035a53"


def _codex_payload(output=None):
    structured = {"answer": "Use evidence."} if output is None else output
    text = canonical_json_bytes(structured).decode("utf-8")
    events = (
        {"type": "thread.started", "thread_id": _THREAD_ID},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "agent_message",
                "text": text,
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 80,
                "output_tokens": 20,
                "reasoning_output_tokens": 7,
            },
        },
    )
    return b"".join(canonical_json_bytes(event) + b"\n" for event in events)


def _claude_envelope(output=None):
    structured = {"answer": "Use evidence."} if output is None else output
    return {
        "is_error": False,
        "duration_api_ms": 100,
        "num_turns": 2,
        "stop_reason": "tool_use",
        "session_id": _THREAD_ID,
        "total_cost_usd": 0.125,
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 20,
            "output_tokens": 53,
            "server_tool_use": {
                "web_search_requests": 0,
                "web_fetch_requests": 0,
            },
        },
        "modelUsage": {
            "helper": {"costUSD": 0.005},
            "requested": {"costUSD": 0.12},
        },
        "permission_denials": [],
        "terminal_reason": "completed",
        "subtype": "success",
        "api_error_status": None,
        "result": canonical_json_bytes(structured).decode("utf-8"),
        "structured_output": structured,
        "type": "result",
    }


def test_codex_command_is_fixed_self_contained_and_disables_ambient_authority():
    request = run_action_request(
        interpretation_policy(web_search_enabled=False),
    )

    command = coding_agent_cli_command(request)

    assert command[0] == "/usr/bin/codex"
    assert command[-1] == "-"
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert command[command.index("--cd") + 1] == "/kapso/workspace"
    assert command[command.index("--output-schema") + 1] == (
        "/kapso/tmp/response.schema.json"
    )
    assert command[command.index("--output-last-message") + 1] == (
        "/kapso/tmp/provider.final.json"
    )
    assert 'web_search="disabled"' in command
    assert 'shell_environment_policy.inherit="none"' in command
    assert "mcp_servers={}" in command
    assert request.prompt not in command


def test_codex_command_binds_edit_web_and_prior_knowledge_authority():
    policy = interpretation_policy(
        workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        web_search_enabled=True,
    )
    request = run_action_request(
        policy,
        prior_knowledge=empty_prior_knowledge(),
        predecessor_digest=tree_or_blob_digest(b"predecessor"),
    )

    command = coding_agent_cli_command(request)

    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert 'web_search="live"' in command
    assert "mcp_servers.prior_knowledge.required=true" in command
    assert any(request.operation_id in value for value in command)
    assert "mcp_servers={}" not in command


@pytest.mark.parametrize(
    ("cli", "command", "version"),
    (
        ("codex", ("/usr/bin/codex", "--version"), b"codex-cli 0.144.1\n"),
        (
            "claude_code",
            ("/usr/local/bin/claude", "--version"),
            b"2.1.220 (Claude Code)\n",
        ),
    ),
)
def test_cli_preflight_requires_the_exact_pinned_executable_and_version(
    cli,
    command,
    version,
):
    request = run_action_request(
        interpretation_policy(
            cli=cli,
            web_search_enabled=False,
        )
    )

    assert coding_agent_cli_preflight_command(request) == command
    validate_coding_agent_cli_preflight(
        request=request,
        return_code=0,
        output_payload=version,
        diagnostic_payload=b"",
    )

    with pytest.raises(RunActionCodingAgentCliError, match="preflight failed"):
        validate_coding_agent_cli_preflight(
            request=request,
            return_code=0,
            output_payload=version.replace(b"0.144.1", b"0.143.0").replace(
                b"2.1.220",
                b"2.1.219",
            ),
            diagnostic_payload=b"",
        )


def test_claude_command_has_exact_tools_schema_and_no_prompt_argument():
    policy = interpretation_policy(
        cli="claude_code",
        web_search_enabled=False,
    )
    request = run_action_request(policy)

    command = coding_agent_cli_command(request)

    assert command[0] == "/usr/local/bin/claude"
    assert "--safe-mode" in command
    assert "--disable-slash-commands" in command
    assert "--strict-mcp-config" in command
    assert command[command.index("--permission-mode") + 1] == "plan"
    assert command[command.index("--tools") + 1] == "Glob,Grep,Read"
    assert "Bash" in command[command.index("--disallowedTools") + 1]
    assert command[command.index("--json-schema") + 1] == (
        canonical_json_bytes(request.response_schema).decode("utf-8")
    )
    assert request.prompt not in command


def test_claude_edit_command_adds_only_versioned_native_edit_tools():
    policy = interpretation_policy(
        cli="claude_code",
        workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        web_search_enabled=True,
    )
    request = run_action_request(
        policy,
        prior_knowledge=empty_prior_knowledge(),
        predecessor_digest=tree_or_blob_digest(b"predecessor"),
    )

    command = coding_agent_cli_command(request)
    tools = command[command.index("--tools") + 1].split(",")

    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    assert tools == [
        "Glob",
        "Grep",
        "Read",
        "Edit",
        "Write",
        "WebSearch",
        "mcp__prior_knowledge__get_prior_knowledge_record",
        "mcp__prior_knowledge__list_prior_knowledge",
    ]
    assert "Bash" not in tools
    assert command[command.index("--allowedTools") + 1].split(",") == [
        "mcp__prior_knowledge__get_prior_knowledge_record",
        "mcp__prior_knowledge__list_prior_knowledge",
    ]


def test_codex_completion_requires_closed_lifecycle_and_exact_final_join():
    request = run_action_request(interpretation_policy())
    final = canonical_json_bytes({"answer": "Use evidence."})

    outcome = interpret_coding_agent_cli_completion(
        request=request,
        return_code=0,
        provider_output_payload=_codex_payload(),
        provider_diagnostic_payload=b"diagnostic",
        final_output_payload=final,
    )

    assert outcome.structured_output == {"answer": "Use evidence."}
    assert outcome.input_tokens == 100
    assert outcome.cached_input_tokens == 80
    assert outcome.output_tokens == 20
    assert outcome.reasoning_output_tokens == 7
    assert outcome.cost_usd is None
    assert outcome.provider_event_stream_digest == tree_or_blob_digest(_codex_payload())
    assert outcome.provider_diagnostic_stream_digest == tree_or_blob_digest(
        b"diagnostic"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.rstrip(b"\n"), "incomplete final event"),
        (
            lambda payload: payload.replace(
                b'"type":"turn.completed"',
                b'"type":"turn.failed"',
            ),
            "terminal failure",
        ),
        (
            lambda payload: payload.replace(
                b'"type":"item.completed"',
                b'"type":"item.unknown"',
            ),
            "unknown event type",
        ),
    ),
)
def test_codex_completion_rejects_incomplete_failed_and_unknown_events(
    mutation,
    message,
):
    request = run_action_request(interpretation_policy())

    with pytest.raises(RunActionCodingAgentCliError, match=message):
        interpret_coding_agent_cli_completion(
            request=request,
            return_code=0,
            provider_output_payload=mutation(_codex_payload()),
            provider_diagnostic_payload=b"",
            final_output_payload=canonical_json_bytes({"answer": "Use evidence."}),
        )


def test_codex_completion_rejects_substituted_final_output():
    request = run_action_request(interpretation_policy())

    with pytest.raises(
        RunActionCodingAgentCliError,
        match="differs from its completed agent message",
    ):
        interpret_coding_agent_cli_completion(
            request=request,
            return_code=0,
            provider_output_payload=_codex_payload(),
            provider_diagnostic_payload=b"",
            final_output_payload=canonical_json_bytes({"answer": "Substituted"}),
        )


def test_claude_completion_reconciles_usage_and_exact_decimal_cost():
    policy = interpretation_policy(
        cli="claude_code",
        web_search_enabled=False,
    )
    request = run_action_request(policy)
    payload = json.dumps(
        _claude_envelope(),
        separators=(",", ":"),
    ).encode("utf-8")

    outcome = interpret_coding_agent_cli_completion(
        request=request,
        return_code=0,
        provider_output_payload=payload,
        provider_diagnostic_payload=b"",
        final_output_payload=None,
    )

    assert outcome.structured_output == {"answer": "Use evidence."}
    assert outcome.input_tokens == 122
    assert outcome.cached_input_tokens == 20
    assert outcome.output_tokens == 53
    assert outcome.reasoning_output_tokens is None
    assert outcome.cost_usd == "0.125"


def test_claude_completion_preserves_numeric_structured_output_for_schema_validation():
    request = run_action_request(
        interpretation_policy(
            cli="claude_code",
            web_search_enabled=False,
        ),
        response_schema={
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
            "additionalProperties": False,
        },
    )
    payload = json.dumps(
        _claude_envelope({"score": 1.25}),
        separators=(",", ":"),
    ).encode("utf-8")

    outcome = interpret_coding_agent_cli_completion(
        request=request,
        return_code=0,
        provider_output_payload=payload,
        provider_diagnostic_payload=b"",
        final_output_payload=None,
    )

    assert outcome.structured_output == {"score": 1.25}


def test_claude_completion_rejects_cost_permission_and_result_conflicts():
    request = run_action_request(
        interpretation_policy(
            cli="claude_code",
            web_search_enabled=False,
        )
    )
    cases = []
    cost_conflict = _claude_envelope()
    cost_conflict["total_cost_usd"] = 0.126
    cases.append((cost_conflict, "total cost differs"))
    denied = _claude_envelope()
    denied["permission_denials"] = [{"tool_name": "Read"}]
    cases.append((denied, "successful terminal state"))
    result_conflict = _claude_envelope()
    result_conflict["result"] = '{"answer":"Substituted"}'
    cases.append((result_conflict, "result text differs"))

    for envelope, message in cases:
        with pytest.raises(RunActionCodingAgentCliError, match=message):
            interpret_coding_agent_cli_completion(
                request=request,
                return_code=0,
                provider_output_payload=json.dumps(
                    envelope,
                    separators=(",", ":"),
                ).encode("utf-8"),
                provider_diagnostic_payload=b"",
                final_output_payload=None,
            )


@pytest.mark.parametrize("optional_field", ("api_error_status", "terminal_reason"))
def test_claude_completion_accepts_absent_optional_success_evidence(optional_field):
    request = run_action_request(
        interpretation_policy(
            cli="claude_code",
            web_search_enabled=False,
        )
    )
    envelope = _claude_envelope()
    del envelope[optional_field]

    outcome = interpret_coding_agent_cli_completion(
        request=request,
        return_code=0,
        provider_output_payload=json.dumps(envelope).encode("utf-8"),
        provider_diagnostic_payload=b"",
        final_output_payload=None,
    )

    assert outcome.structured_output == {"answer": "Use evidence."}


def test_claude_completion_rejects_unauthorized_web_search_and_duplicate_fields():
    request = run_action_request(
        interpretation_policy(
            cli="claude_code",
            web_search_enabled=False,
        )
    )
    searched = _claude_envelope()
    searched["usage"]["server_tool_use"]["web_search_requests"] = 1

    with pytest.raises(RunActionCodingAgentCliError, match="without request authority"):
        interpret_coding_agent_cli_completion(
            request=request,
            return_code=0,
            provider_output_payload=json.dumps(searched).encode("utf-8"),
            provider_diagnostic_payload=b"",
            final_output_payload=None,
        )

    fetched = _claude_envelope()
    fetched["usage"]["server_tool_use"]["web_fetch_requests"] = 1
    with pytest.raises(RunActionCodingAgentCliError, match="web fetch"):
        interpret_coding_agent_cli_completion(
            request=request,
            return_code=0,
            provider_output_payload=json.dumps(fetched).encode("utf-8"),
            provider_diagnostic_payload=b"",
            final_output_payload=None,
        )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        interpret_coding_agent_cli_completion(
            request=request,
            return_code=0,
            provider_output_payload=b'{"type":"result","type":"result"}',
            provider_diagnostic_payload=b"",
            final_output_payload=None,
        )


@pytest.mark.parametrize(
    ("return_code", "output", "diagnostic", "final", "message"),
    (
        (1, b"provider", b"", None, "exact success"),
        (0, b"", b"", None, "provider output"),
        (0, b"provider", b"", b"unexpected", "separate final output"),
    ),
)
def test_completion_rejects_process_and_provider_boundary_conflicts(
    return_code,
    output,
    diagnostic,
    final,
    message,
):
    request = run_action_request(
        interpretation_policy(
            cli="claude_code",
            web_search_enabled=False,
        )
    )

    with pytest.raises(RunActionCodingAgentCliError, match=message):
        interpret_coding_agent_cli_completion(
            request=request,
            return_code=return_code,
            provider_output_payload=output,
            provider_diagnostic_payload=diagnostic,
            final_output_payload=final,
        )


def test_completion_enforces_provider_stream_byte_bounds_before_parsing():
    request = run_action_request(interpretation_policy())

    with pytest.raises(RunActionCodingAgentCliError, match="provider output"):
        interpret_coding_agent_cli_completion(
            request=request,
            return_code=0,
            provider_output_payload=b"x"
            * (request.interpretation_policy.maximum_provider_output_bytes + 1),
            provider_diagnostic_payload=b"",
            final_output_payload=b"{}",
        )
