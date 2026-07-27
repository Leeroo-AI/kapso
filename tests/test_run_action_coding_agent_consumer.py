import json
import os
import stat
import subprocess
import sys
import time
from contextlib import ExitStack

import pytest

import kapso.cross_run.launch.run_action_coding_agent_consumer as consumer_module
from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.core.config import load_config
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.knowledge.access import PriorKnowledgeAccess
from kapso.cross_run.launch.run_action_coding_agent_consumer import (
    BoundedCodingAgentProcessCompletion,
    BoundedCodingAgentProcessRunner,
    NATIVE_CODING_AGENT_CONSUMER_ID,
    NATIVE_CODING_AGENT_CONSUMER_VERSION,
    RunActionCodingAgentConsumerError,
    consume_coding_agent_run_action,
)
from kapso.cross_run.launch.run_action_coding_agent_cli import (
    RunActionCodingAgentCliError,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentPriorKnowledgeAccessKind,
    RunActionCodingAgentContractError,
    read_canonical_coding_agent_result,
)
from kapso.cross_run.launch.run_action_coding_agent_scratch import (
    RunActionCodingAgentScratchError,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier_with_limits,
    inspect_run_workspace_source_tree,
)
from kapso.cross_run.launch.workspace import StarterWorkspaceBuilder
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_coding_agent_contracts import (
    empty_prior_knowledge,
    interpretation_policy,
    run_action_request,
)

_THREAD_ID = "0199a213-81c0-7800-8aa1-bbab2a035a53"


class ScriptedProcessRunner:
    def __init__(self, steps, *, git_workspace=None):
        self.steps = tuple(steps)
        self.git_workspace = git_workspace
        self.calls = []
        self.step_index = 0

    def run(
        self,
        command,
        *,
        stdin_payload,
        stdin_directory,
        working_directory,
        timeout_nanoseconds,
        termination_grace_nanoseconds,
        maximum_output_bytes,
        maximum_diagnostic_bytes,
        environment,
        inherited_descriptors,
    ):
        call = {
            "command": command,
            "stdin_payload": stdin_payload,
            "stdin_directory": stdin_directory,
            "working_directory": working_directory,
            "timeout_nanoseconds": timeout_nanoseconds,
            "termination_grace_nanoseconds": termination_grace_nanoseconds,
            "maximum_output_bytes": maximum_output_bytes,
            "maximum_diagnostic_bytes": maximum_diagnostic_bytes,
            "environment": environment,
            "inherited_descriptors": inherited_descriptors,
        }
        self.calls.append(call)
        if command[0] == "/usr/bin/git":
            if self.git_workspace is None:
                raise AssertionError("unexpected trusted Git command")
            translated = tuple(
                str(self.git_workspace) if value == "/kapso/workspace" else value
                for value in command
            )
            completed = subprocess.run(
                translated,
                cwd=self.git_workspace,
                input=stdin_payload,
                capture_output=True,
                check=False,
                env=dict(environment),
            )
            return _completion(
                output=completed.stdout,
                diagnostic=completed.stderr,
                return_code=completed.returncode,
            )
        step = self.steps[self.step_index]
        self.step_index += 1
        return step(call)


def _completion(output=b"", diagnostic=b"", return_code=0):
    return BoundedCodingAgentProcessCompletion(
        return_code=return_code,
        output_payload=output,
        diagnostic_payload=diagnostic,
    )


def _codex_event_payload(output=None, prior_response_payload=None):
    structured_output = {"answer": "Use evidence."} if output is None else output
    structured_text = canonical_json_bytes(structured_output).decode("utf-8")
    events = [
        {"type": "thread.started", "thread_id": _THREAD_ID},
        {"type": "turn.started"},
    ]
    if prior_response_payload is not None:
        events.extend(
            [
                {
                    "type": "item.started",
                    "item": {
                        "id": "item_0",
                        "type": "mcp_tool_call",
                        "server": "prior_knowledge",
                        "tool": "list_prior_knowledge",
                        "arguments": {},
                        "result": None,
                        "error": None,
                        "status": "in_progress",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item_0",
                        "type": "mcp_tool_call",
                        "server": "prior_knowledge",
                        "tool": "list_prior_knowledge",
                        "arguments": {},
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": canonical_json_bytes(
                                        prior_response_payload
                                    ).decode("utf-8"),
                                }
                            ],
                            "structured_content": None,
                        },
                        "error": None,
                        "status": "completed",
                    },
                },
            ]
        )
    events.extend(
        [
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1" if prior_response_payload is not None else "item_0",
                    "type": "agent_message",
                    "text": structured_text,
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
        ]
    )
    return b"".join(canonical_json_bytes(event) + b"\n" for event in events)


def _claude_payload(output=None):
    structured_output = {"answer": "Use evidence."} if output is None else output
    envelope = {
        "is_error": False,
        "duration_api_ms": 100,
        "num_turns": 1,
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
            "gpt-5.6": {
                "inputTokens": 2,
                "outputTokens": 53,
                "cacheReadInputTokens": 20,
                "cacheCreationInputTokens": 100,
                "webSearchRequests": 0,
                "costUSD": 0.125,
                "contextWindow": 200_000,
                "maxOutputTokens": 32_000,
            },
        },
        "permission_denials": [],
        "terminal_reason": "completed",
        "subtype": "success",
        "api_error_status": None,
        "result": canonical_json_bytes(structured_output).decode("utf-8"),
        "structured_output": structured_output,
        "type": "result",
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def _native_policy(**overrides):
    provider_groups = tuple(
        group_id for group_id in os.getgroups() if group_id != os.getegid()
    )
    if not provider_groups:
        raise AssertionError("consumer tests require one supplemental provider group")
    return interpretation_policy(
        consumer_id=NATIVE_CODING_AGENT_CONSUMER_ID,
        consumer_version=NATIVE_CODING_AGENT_CONSUMER_VERSION,
        supervisor_user_id=os.geteuid(),
        supervisor_group_id=os.getegid(),
        provider_user_id=os.geteuid() + 1,
        provider_group_id=provider_groups[0],
        **overrides,
    )


def _runtime_directories(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    source = workspace / "proposal.py"
    source_payload = b"score = 1\n"
    source.write_bytes(source_payload)
    source.chmod(0o600)
    settings = CrossRunSettings.from_dict(
        load_config("src/kapso/config.yaml")["cross_run"]
    )
    StarterWorkspaceBuilder(settings)._initialize_git_baseline(
        workspace,
        (
            SourceFileDescriptor(
                relative_path="proposal.py",
                digest=tree_or_blob_digest(source_payload),
                mode="100644",
                size=len(source_payload),
            ),
        ),
        {"proposal.py": source_payload},
    )
    temporary = tmp_path / "temporary"
    temporary.mkdir(mode=0o700)
    temporary.chmod(0o700)
    return workspace, source, temporary


def _open_runtime_descriptors(workspace, temporary, cleanup):
    workspace_descriptor = os.open(
        workspace,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    cleanup.callback(os.close, workspace_descriptor)
    temporary_descriptor = os.open(
        temporary,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    cleanup.callback(os.close, temporary_descriptor)
    return workspace_descriptor, temporary_descriptor


def _write_private(path, payload):
    path.write_bytes(payload)
    path.chmod(0o600)


def _write_provider_file(path, payload):
    path.write_bytes(payload)
    path.chmod(0o660)
    provider_group_id = next(
        group_id for group_id in os.getgroups() if group_id != os.getegid()
    )
    os.chown(path, -1, provider_group_id)


def _codex_provider_step(temporary, output=None):
    structured_output = {"answer": "Use evidence."} if output is None else output

    def run(_call):
        _write_provider_file(
            temporary / "provider-output" / "provider.final.json",
            canonical_json_bytes(structured_output),
        )
        return _completion(
            output=_codex_event_payload(structured_output),
            diagnostic=b"provider diagnostic",
        )

    return run


def test_consumer_executes_codex_read_only_and_publishes_one_canonical_result(
    tmp_path,
):
    workspace, _source, temporary = _runtime_directories(tmp_path)
    request = run_action_request(_native_policy(web_search_enabled=False))
    runner = ScriptedProcessRunner(
        (
            lambda _call: _completion(output=b"codex-cli 0.144.1\n"),
            _codex_provider_step(temporary),
        )
    )

    with ExitStack() as cleanup:
        workspace_descriptor, temporary_descriptor = _open_runtime_descriptors(
            workspace,
            temporary,
            cleanup,
        )
        published = consume_coding_agent_run_action(
            request_payload=request.to_json_bytes(),
            workspace_descriptor=workspace_descriptor,
            temporary_directory_descriptor=temporary_descriptor,
            process_runner=runner,
        )

    candidate = temporary / "result.candidate"
    result = read_canonical_coding_agent_result(candidate.read_bytes())
    result.validate_against(
        policy=request.interpretation_policy,
        request=request,
    )
    assert result.structured_output == {"answer": "Use evidence."}
    assert result.edited_source_tree_digest is None
    assert result.cost_usd is None
    assert result.prior_knowledge_accesses == ()
    assert published.content_digest == tree_or_blob_digest(candidate.read_bytes())
    assert stat.S_IMODE(candidate.stat().st_mode) == 0o600
    assert len(runner.calls) == 2
    assert runner.calls[0]["command"][runner.calls[0]["command"].index("--") + 1 :] == (
        "/usr/bin/codex",
        "--version",
    )
    assert runner.calls[0]["stdin_payload"] is None
    assert runner.calls[1]["stdin_payload"] == request.prompt.encode("utf-8")
    assert runner.calls[1]["stdin_directory"] == "/kapso/tmp"
    assert runner.calls[1]["working_directory"] == "/kapso/tmp/provider-workspace"
    assert request.prompt not in runner.calls[1]["command"]
    assert (
        runner.calls[0]["environment"]
        == runner.calls[1]["environment"]
        == {
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/kapso/tmp/provider-home",
            "LANG": "C",
            "LC_ALL": "C",
            "NO_COLOR": "1",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "TERM": "dumb",
            "TMPDIR": "/kapso/tmp/provider-home",
        }
    )
    assert len(runner.calls[0]["inherited_descriptors"]) == 4
    assert (
        runner.calls[0]["inherited_descriptors"]
        == runner.calls[1]["inherited_descriptors"]
    )


def test_fixed_path_main_bounds_request_before_parsing_self_declared_policy(
    tmp_path,
    monkeypatch,
):
    request_blob = tmp_path / "request.blob"
    _write_private(request_blob, b"untrusted")
    monkeypatch.setattr(consumer_module, "_REQUEST_PATH", str(request_blob))
    monkeypatch.setattr(
        sys,
        "argv",
        ["kapso-run-action-coding-agent-consumer", "--maximum-request-bytes", "8"],
    )

    with pytest.raises(
        RunActionCodingAgentConsumerError,
        match="request is not one complete bounded private file",
    ):
        consumer_module.main()


def test_explicit_frontier_accepts_a_valid_nested_workspace_branch(tmp_path):
    workspace, _source, _temporary = _runtime_directories(tmp_path)
    branch = "feature/scientific-improvement"
    subprocess.run(
        ("git", "-C", str(workspace), "branch", "-m", branch),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    with ExitStack() as cleanup:
        workspace_descriptor, _temporary_descriptor = _open_runtime_descriptors(
            workspace,
            _temporary,
            cleanup,
        )
        frontier = inspect_run_workspace_frontier_with_limits(
            workspace_descriptor,
            workspace_git_branch=branch,
            maximum_source_entries=10_000,
            maximum_source_bytes=1_073_741_824,
            maximum_git_entries=50_000,
            maximum_git_bytes=67_108_864,
            expected_commit_sha=None,
        )

    assert frontier.branch == branch


def test_consumer_executes_claude_edit_and_binds_the_observed_successor(tmp_path):
    workspace, source, temporary = _runtime_directories(tmp_path)
    with ExitStack() as cleanup:
        workspace_descriptor, temporary_descriptor = _open_runtime_descriptors(
            workspace,
            temporary,
            cleanup,
        )
        predecessor = inspect_run_workspace_source_tree(
            workspace_descriptor,
            maximum_entries=10_000,
            maximum_bytes=1_073_741_824,
        ).source_tree_digest

        def edit_workspace(_call):
            scratch_source = temporary / "provider-workspace" / source.name
            scratch_source.write_bytes(b"score = 2\n")
            scratch_source.chmod(0o660)
            return _completion(output=_claude_payload())

        request = run_action_request(
            _native_policy(
                cli="claude_code",
                workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
                web_search_enabled=False,
            ),
            predecessor_digest=predecessor,
        )
        runner = ScriptedProcessRunner(
            (
                lambda _call: _completion(output=b"2.1.220 (Claude Code)\n"),
                edit_workspace,
            ),
            git_workspace=workspace,
        )
        consume_coding_agent_run_action(
            request_payload=request.to_json_bytes(),
            workspace_descriptor=workspace_descriptor,
            temporary_directory_descriptor=temporary_descriptor,
            process_runner=runner,
        )
        successor = inspect_run_workspace_source_tree(
            workspace_descriptor,
            maximum_entries=10_000,
            maximum_bytes=1_073_741_824,
        ).source_tree_digest

    result = read_canonical_coding_agent_result(
        (temporary / "result.candidate").read_bytes()
    )
    result.validate_against(
        policy=request.interpretation_policy,
        request=request,
    )
    assert successor != predecessor
    assert result.edited_source_tree_digest == successor
    assert result.cost_usd == "0.125"
    command = runner.calls[0]["command"]
    assert command[command.index("--") + 1 :] == (
        "/usr/local/bin/claude",
        "--version",
    )
    for git_call in runner.calls[2:]:
        assert git_call["command"][1:5] == (
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
        )
        assert git_call["environment"]["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert git_call["environment"]["GIT_CONFIG_NOSYSTEM"] == "1"
        assert git_call["environment"]["GIT_TERMINAL_PROMPT"] == "0"
        assert git_call["environment"]["HOME"] == "/nonexistent"
        assert git_call["environment"]["XDG_CONFIG_HOME"] == "/nonexistent"


def test_consumer_translates_ordered_prior_knowledge_mcp_audit(tmp_path):
    workspace, _source, temporary = _runtime_directories(tmp_path)
    prior_knowledge = empty_prior_knowledge()
    request = run_action_request(
        _native_policy(web_search_enabled=False),
        prior_knowledge=prior_knowledge,
    )
    access = PriorKnowledgeAccess(prior_knowledge)
    event = {
        "arguments": {},
        "operation_id": request.operation_id,
        "prior_knowledge_snapshot_id": (
            prior_knowledge.prior_knowledge_snapshot.prior_knowledge_snapshot_id
        ),
        "response_digest": tree_or_blob_digest(
            canonical_json_bytes(access.list_response_payload())
        ),
        "returned_ids": [],
        "tool_name": "list_prior_knowledge",
    }

    def provider_with_audit(call):
        _write_provider_file(
            temporary / "provider-output" / "prior_knowledge.audit.jsonl",
            canonical_json_bytes(event) + b"\n",
        )
        _write_provider_file(
            temporary / "provider-output" / "provider.final.json",
            canonical_json_bytes({"answer": "Use evidence."}),
        )
        return _completion(
            output=_codex_event_payload(
                prior_response_payload=access.list_response_payload()
            ),
            diagnostic=b"provider diagnostic",
        )

    runner = ScriptedProcessRunner(
        (
            lambda _call: _completion(output=b"codex-cli 0.144.1\n"),
            provider_with_audit,
        )
    )
    with ExitStack() as cleanup:
        workspace_descriptor, temporary_descriptor = _open_runtime_descriptors(
            workspace,
            temporary,
            cleanup,
        )
        consume_coding_agent_run_action(
            request_payload=request.to_json_bytes(),
            workspace_descriptor=workspace_descriptor,
            temporary_directory_descriptor=temporary_descriptor,
            process_runner=runner,
        )

    result = read_canonical_coding_agent_result(
        (temporary / "result.candidate").read_bytes()
    )
    assert len(result.prior_knowledge_accesses) == 1
    assert (
        result.prior_knowledge_accesses[0].access_kind
        is CodingAgentPriorKnowledgeAccessKind.LIST
    )
    assert result.prior_knowledge_accesses[0].returned_record_ids == ()


def test_consumer_rejects_prior_knowledge_audit_substitution(tmp_path):
    workspace, _source, temporary = _runtime_directories(tmp_path)
    prior_knowledge = empty_prior_knowledge()
    request = run_action_request(
        _native_policy(web_search_enabled=False),
        prior_knowledge=prior_knowledge,
    )
    access = PriorKnowledgeAccess(prior_knowledge)
    event = {
        "arguments": {},
        "operation_id": request.operation_id,
        "prior_knowledge_snapshot_id": (
            prior_knowledge.prior_knowledge_snapshot.prior_knowledge_snapshot_id
        ),
        "response_digest": tree_or_blob_digest(
            canonical_json_bytes(access.list_response_payload())
        ),
        "returned_ids": [],
        "tool_name": "list_prior_knowledge",
    }

    def provider_with_substituted_audit(_call):
        _write_provider_file(
            temporary / "provider-output" / "prior_knowledge.audit.jsonl",
            canonical_json_bytes(event) + b"\n",
        )
        _write_provider_file(
            temporary / "provider-output" / "provider.final.json",
            canonical_json_bytes({"answer": "Use evidence."}),
        )
        return _completion(
            output=_codex_event_payload(
                prior_response_payload={"records": [], "substituted": True}
            )
        )

    runner = ScriptedProcessRunner(
        (
            lambda _call: _completion(output=b"codex-cli 0.144.1\n"),
            provider_with_substituted_audit,
        )
    )
    with ExitStack() as cleanup:
        workspace_descriptor, temporary_descriptor = _open_runtime_descriptors(
            workspace,
            temporary,
            cleanup,
        )
        with pytest.raises(RunActionCodingAgentCliError, match="ordered.*audit"):
            consume_coding_agent_run_action(
                request_payload=request.to_json_bytes(),
                workspace_descriptor=workspace_descriptor,
                temporary_directory_descriptor=temporary_descriptor,
                process_runner=runner,
            )

    assert not (temporary / "result.candidate").exists()


@pytest.mark.parametrize(
    ("failure_kind", "expected_exception", "message"),
    (
        ("preflight", RunActionCodingAgentCliError, "preflight failed"),
        (
            "preflight_state",
            RunActionCodingAgentConsumerError,
            "preflight left mutable provider state",
        ),
        ("provider", RunActionCodingAgentCliError, "exact success"),
        (
            "read_only_mutation",
            RunActionCodingAgentConsumerError,
            "changed disposable scratch",
        ),
        (
            "read_only_mode_mutation",
            RunActionCodingAgentScratchError,
            "inaccessible or unsafe",
        ),
        (
            "read_only_git_mode_mutation",
            RunActionCodingAgentConsumerError,
            "physical Git metadata changed",
        ),
        (
            "unchanged_edit",
            RunActionCodingAgentScratchError,
            "did not change the source tree",
        ),
        (
            "support_mutation",
            RunActionCodingAgentScratchError,
            "support inode changed",
        ),
    ),
)
def test_consumer_failures_never_publish_a_terminal_candidate(
    tmp_path,
    failure_kind,
    expected_exception,
    message,
):
    workspace, source, temporary = _runtime_directories(tmp_path)
    with ExitStack() as cleanup:
        workspace_descriptor, temporary_descriptor = _open_runtime_descriptors(
            workspace,
            temporary,
            cleanup,
        )
        predecessor = inspect_run_workspace_source_tree(
            workspace_descriptor,
            maximum_entries=10_000,
            maximum_bytes=1_073_741_824,
        ).source_tree_digest
        editing = failure_kind == "unchanged_edit"
        request = run_action_request(
            _native_policy(
                workspace_access=(
                    RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                    if editing
                    else RunFrontierWorkspaceAccess.READ_ONLY
                ),
                web_search_enabled=False,
            ),
            predecessor_digest=predecessor if editing else None,
        )

        def provider(_call):
            if failure_kind == "read_only_mutation":
                scratch_source = temporary / "provider-workspace" / source.name
                scratch_source.write_bytes(b"score = 3\n")
                scratch_source.chmod(0o660)
            if failure_kind == "read_only_mode_mutation":
                (temporary / "provider-workspace" / source.name).chmod(0o600)
            if failure_kind == "read_only_git_mode_mutation":
                (workspace / ".git" / "objects").chmod(0o500)
            if failure_kind == "support_mutation":
                support = temporary / "provider-support" / "response.schema.json"
                support.chmod(0o660)
                support.write_bytes(b'{"substituted":true}')
            _write_provider_file(
                temporary / "provider-output" / "provider.final.json",
                canonical_json_bytes({"answer": "Use evidence."}),
            )
            return _completion(
                output=_codex_event_payload(),
                return_code=1 if failure_kind == "provider" else 0,
            )

        def preflight(_call):
            if failure_kind == "preflight_state":
                _write_provider_file(
                    temporary / "provider-home" / "ambient.config",
                    b"untrusted",
                )
            return _completion(
                output=(
                    b"codex-cli 0.143.0\n"
                    if failure_kind == "preflight"
                    else b"codex-cli 0.144.1\n"
                )
            )

        runner = ScriptedProcessRunner(
            (
                preflight,
                provider,
            )
        )
        with pytest.raises(expected_exception, match=message):
            consume_coding_agent_run_action(
                request_payload=request.to_json_bytes(),
                workspace_descriptor=workspace_descriptor,
                temporary_directory_descriptor=temporary_descriptor,
                process_runner=runner,
            )

    assert not (temporary / "result.candidate").exists()
    if failure_kind in {"preflight", "preflight_state"}:
        assert len(runner.calls) == 1


def test_bounded_runner_passes_complete_stdin_from_writable_scratch(
    tmp_path,
):
    working_directory = tmp_path / "read-only-workspace"
    working_directory.mkdir(mode=0o700)
    working_directory.chmod(0o500)
    stdin_directory = tmp_path / "scratch"
    stdin_directory.mkdir(mode=0o700)
    stdin_directory.chmod(0o700)
    prompt = ("complete-unicode-prompt-αβγ\n" * 4_096).encode("utf-8")
    program = (
        "import os,sys;"
        "payload=sys.stdin.buffer.read();"
        "sys.stdout.buffer.write(str(len(payload)).encode());"
        "sys.stderr.buffer.write(os.getcwd().encode())"
    )

    completion = BoundedCodingAgentProcessRunner().run(
        (sys.executable, "-c", program),
        stdin_payload=prompt,
        stdin_directory=str(stdin_directory),
        working_directory=str(working_directory),
        timeout_nanoseconds=10_000_000_000,
        termination_grace_nanoseconds=1_000_000_000,
        maximum_output_bytes=1_024,
        maximum_diagnostic_bytes=4_096,
        environment=None,
        inherited_descriptors=(),
    )

    assert completion.return_code == 0
    assert completion.output_payload == str(len(prompt)).encode("ascii")
    assert completion.diagnostic_payload == str(working_directory).encode("utf-8")
    assert tuple(working_directory.iterdir()) == ()
    assert tuple(stdin_directory.iterdir()) == ()


def test_bounded_runner_accepts_exact_complete_dual_stream_limits(tmp_path):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir(mode=0o700)
    stdin_directory = tmp_path / "scratch"
    stdin_directory.mkdir(mode=0o700)
    output_payload = b"o" * 4_096
    diagnostic_payload = b"d" * 4_096
    program = (
        "import sys;"
        f"sys.stdout.buffer.write({output_payload!r});"
        "sys.stdout.buffer.flush();"
        f"sys.stderr.buffer.write({diagnostic_payload!r});"
        "sys.stderr.buffer.flush()"
    )

    completion = BoundedCodingAgentProcessRunner().run(
        (sys.executable, "-c", program),
        stdin_payload=None,
        stdin_directory=str(stdin_directory),
        working_directory=str(working_directory),
        timeout_nanoseconds=10_000_000_000,
        termination_grace_nanoseconds=1_000_000_000,
        maximum_output_bytes=len(output_payload),
        maximum_diagnostic_bytes=len(diagnostic_payload),
        environment=None,
        inherited_descriptors=(),
    )

    assert completion.output_payload == output_payload
    assert completion.diagnostic_payload == diagnostic_payload


def test_bounded_runner_rejects_diagnostic_overflow(tmp_path):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir(mode=0o700)
    stdin_directory = tmp_path / "scratch"
    stdin_directory.mkdir(mode=0o700)

    with pytest.raises(RunActionCodingAgentConsumerError, match="diagnostic exceeded"):
        BoundedCodingAgentProcessRunner().run(
            (sys.executable, "-c", 'import sys;sys.stderr.write("d"*9)'),
            stdin_payload=None,
            stdin_directory=str(stdin_directory),
            working_directory=str(working_directory),
            timeout_nanoseconds=10_000_000_000,
            termination_grace_nanoseconds=100_000_000,
            maximum_output_bytes=8,
            maximum_diagnostic_bytes=8,
            environment=None,
            inherited_descriptors=(),
        )


def test_bounded_runner_times_out_after_leader_closes_provider_streams(tmp_path):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir(mode=0o700)
    stdin_directory = tmp_path / "scratch"
    stdin_directory.mkdir(mode=0o700)
    started = time.monotonic()

    with pytest.raises(
        RunActionCodingAgentConsumerError,
        match="exceeded its exact timeout",
    ):
        BoundedCodingAgentProcessRunner().run(
            (
                sys.executable,
                "-c",
                "import os,time;os.close(1);os.close(2);time.sleep(30)",
            ),
            stdin_payload=None,
            stdin_directory=str(stdin_directory),
            working_directory=str(working_directory),
            timeout_nanoseconds=300_000_000,
            termination_grace_nanoseconds=200_000_000,
            maximum_output_bytes=8,
            maximum_diagnostic_bytes=8,
            environment=None,
            inherited_descriptors=(),
        )

    assert time.monotonic() - started < 2


def test_bounded_runner_kills_descendant_that_retains_provider_streams(tmp_path):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir(mode=0o700)
    stdin_directory = tmp_path / "scratch"
    stdin_directory.mkdir(mode=0o700)
    child_pid_path = working_directory / "child.pid"
    child_program = "import time;time.sleep(30)"
    leader_program = (
        "import pathlib,subprocess,sys;"
        f"child=subprocess.Popen([sys.executable,'-c',{child_program!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))"
    )
    started = time.monotonic()

    completion = BoundedCodingAgentProcessRunner().run(
        (sys.executable, "-c", leader_program),
        stdin_payload=None,
        stdin_directory=str(stdin_directory),
        working_directory=str(working_directory),
        timeout_nanoseconds=10_000_000_000,
        termination_grace_nanoseconds=200_000_000,
        maximum_output_bytes=8,
        maximum_diagnostic_bytes=8,
        environment=None,
        inherited_descriptors=(),
    )

    assert time.monotonic() - started < 2
    assert completion == _completion()
    child_pid = int(child_pid_path.read_text())
    child_status = subprocess.run(
        ("/bin/ps", "-o", "stat=", "-p", str(child_pid)),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    assert child_status.returncode == 1 or child_status.stdout.lstrip().startswith(b"Z")


def test_bounded_runner_kills_descendant_that_closes_provider_streams(tmp_path):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir(mode=0o700)
    stdin_directory = tmp_path / "scratch"
    stdin_directory.mkdir(mode=0o700)
    child_pid_path = working_directory / "child.pid"
    child_program = "import os,time;os.close(1);os.close(2);time.sleep(30)"
    leader_program = (
        "import pathlib,subprocess,sys;"
        f"child=subprocess.Popen([sys.executable,'-c',{child_program!r}]);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))"
    )

    completion = BoundedCodingAgentProcessRunner().run(
        (sys.executable, "-c", leader_program),
        stdin_payload=None,
        stdin_directory=str(stdin_directory),
        working_directory=str(working_directory),
        timeout_nanoseconds=10_000_000_000,
        termination_grace_nanoseconds=200_000_000,
        maximum_output_bytes=8,
        maximum_diagnostic_bytes=8,
        environment=None,
        inherited_descriptors=(),
    )

    assert completion == _completion()
    child_pid = int(child_pid_path.read_text())
    child_status = subprocess.run(
        ("/bin/ps", "-o", "stat=", "-p", str(child_pid)),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    assert child_status.returncode == 1 or child_status.stdout.lstrip().startswith(b"Z")


@pytest.mark.parametrize(
    ("program", "timeout_nanoseconds", "maximum_output_bytes", "message"),
    (
        (
            'import sys;sys.stdout.write("x"*1024)',
            10_000_000_000,
            8,
            "output exceeded",
        ),
        (
            "import time;time.sleep(30)",
            500_000_000,
            1_024,
            "exceeded its exact timeout",
        ),
    ),
)
def test_bounded_runner_kills_oversized_and_timed_out_processes(
    tmp_path,
    program,
    timeout_nanoseconds,
    maximum_output_bytes,
    message,
):
    working_directory = tmp_path / "workspace"
    working_directory.mkdir(mode=0o700)
    stdin_directory = tmp_path / "scratch"
    stdin_directory.mkdir(mode=0o700)

    with pytest.raises(RunActionCodingAgentConsumerError, match=message):
        BoundedCodingAgentProcessRunner().run(
            (sys.executable, "-c", program),
            stdin_payload=b"prompt",
            stdin_directory=str(stdin_directory),
            working_directory=str(working_directory),
            timeout_nanoseconds=timeout_nanoseconds,
            termination_grace_nanoseconds=100_000_000,
            maximum_output_bytes=maximum_output_bytes,
            maximum_diagnostic_bytes=1_024,
            environment=None,
            inherited_descriptors=(),
        )

    assert tuple(stdin_directory.iterdir()) == ()
