"""Behavioral tests for the read-only coding-agent subprocess boundary."""

import base64
import json
import os
import pwd
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from kapso.core.config import load_config
from kapso.cross_run.settings import CodingAgentSettings
from kapso.execution.coding_agents.operation_receipt import (
    CodingAgentOperationReceiptError,
    seal_coding_agent_operation,
    verify_coding_agent_operation_artifacts,
)
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    CodingAgentInvocationError,
    CodingAgentRunnerSettings,
    CodingAgentWorkspacePolicy,
    SubprocessCodingAgentCallRunner,
)
from kapso.execution.coding_agents.workspace_delta import (
    inspect_coding_agent_workspace,
    reconstruct_edited_workspace,
    validate_coding_agent_workspace_delta,
)
from kapso.execution.coding_agents.credential_environment import (
    coding_agent_credential_environment,
)
from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.contracts import (
    CodingAgentWorkspaceChangedFile,
    CodingAgentWorkspaceDelta,
    SourceFileDescriptor,
)
from kapso.cross_run.agent_artifacts import (
    CODING_AGENT_WORKSPACE_DELTA_FILENAME,
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
)
from kapso.cross_run.knowledge.access import PriorKnowledgeAccess
from test_prior_knowledge_gate import access_materialization

SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH = load_config("src/kapso/config.yaml")[
    "ideation_profiles"
]["DEFAULT"]["coding_agents"]["sensitive_file_glob_scan_max_depth"]
EXPERT_CONFIG = load_config("src/kapso/config.yaml")["cross_run"]["expert"]
WORKSPACE_ENTRY_LIMIT = EXPERT_CONFIG["candidate_entry_limit"]
WORKSPACE_BYTE_LIMIT = EXPERT_CONFIG["candidate_byte_limit"]


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
        workspace_policy=CodingAgentWorkspacePolicy.read_only(),
        timeout_seconds=10,
        effort="high",
        allowed_tools=("Read", "WebSearch"),
    )


def runner(tmp_path: Path) -> SubprocessCodingAgentCallRunner:
    return SubprocessCodingAgentCallRunner(
        CodingAgentRunnerSettings(
            artifact_root=str((tmp_path / "artifacts").resolve()),
            termination_grace_seconds=1,
            sensitive_file_glob_scan_max_depth=(SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH),
        )
    )


def editable_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    (workspace / "existing.txt").write_text("before", encoding="utf-8")
    (workspace / "deleted.txt").write_text("remove", encoding="utf-8")
    return workspace


def editable_request(workspace: Path, cli: str) -> CodingAgentCallRequest:
    baseline = inspect_coding_agent_workspace(
        workspace,
        maximum_entries=WORKSPACE_ENTRY_LIMIT,
        maximum_bytes=WORKSPACE_BYTE_LIMIT,
    )
    return replace(
        request(workspace, cli),
        allowed_tools=(
            ("Read",) if cli == "codex" else ("Edit", "Glob", "Grep", "Read", "Write")
        ),
        workspace_policy=CodingAgentWorkspacePolicy.edit_workspace(
            expected_tree_hash=baseline.tree_hash,
            maximum_entries=WORKSPACE_ENTRY_LIMIT,
            maximum_bytes=WORKSPACE_BYTE_LIMIT,
        ),
    )


def install_edit_executable(directory: Path, cli: str) -> None:
    edit_source = """
import pathlib
import sys
pathlib.Path("existing.txt").write_text("after")
pathlib.Path("created.py").write_text("print('created')\\n")
pathlib.Path("deleted.txt").unlink()
counter = pathlib.Path(__file__).with_name("edit-invocations.txt")
counter.write_text(counter.read_text() + "x" if counter.exists() else "x")
arguments = pathlib.Path(__file__).with_name(pathlib.Path(__file__).stem + "-edit-args.json")
arguments.write_text(__import__("json").dumps(sys.argv[1:]))
"""
    if cli == "codex":
        source = "#!/usr/bin/env python3\nimport json\nimport sys\n" + edit_source + """
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"changed_paths":["created.py","existing.txt"],"deleted_paths":["deleted.txt"]}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":2}}))
"""
        install_executable(directory, "codex", source)
        return
    source = "#!/usr/bin/env python3\nimport json\n" + edit_source + """
print(json.dumps({
  "is_error": False,
  "structured_output": {
    "changed_paths": ["created.py", "existing.txt"],
    "deleted_paths": ["deleted.txt"]
  },
  "usage": {"input_tokens": 3, "output_tokens": 2},
  "total_cost_usd": 0.1
}))
"""
    install_executable(directory, "claude", source)


@pytest.mark.parametrize(
    ("workspace", "error"),
    (("relative/workspace", "absolute"),),
)
def test_request_rejects_relative_agent_workspace(workspace, error):
    with pytest.raises(ValueError, match=error):
        CodingAgentCallRequest(
            operation_id="agent_call_" + "1" * 32,
            role="candidate",
            cli="codex",
            model="test-model",
            prompt="complete prompt",
            workspace=workspace,
            workspace_policy=CodingAgentWorkspacePolicy.read_only(),
            timeout_seconds=10,
        )


def test_workspace_policy_is_explicit_and_round_trips_exactly():
    read_only = CodingAgentWorkspacePolicy.read_only()
    editable = CodingAgentWorkspacePolicy.edit_workspace(
        expected_tree_hash="sha256:" + "1" * 64,
        maximum_entries=WORKSPACE_ENTRY_LIMIT,
        maximum_bytes=WORKSPACE_BYTE_LIMIT,
    )

    assert CodingAgentWorkspacePolicy.from_dict(read_only.to_dict()) == read_only
    assert CodingAgentWorkspacePolicy.from_dict(editable.to_dict()) == editable
    with pytest.raises(ValueError, match="cannot declare edit limits"):
        CodingAgentWorkspacePolicy(
            access=CodingAgentWorkspaceAccess.READ_ONLY,
            expected_tree_hash="sha256:" + "1" * 64,
            maximum_entries=None,
            maximum_bytes=None,
        )


@pytest.mark.parametrize(
    "workspace",
    ("/", pwd.getpwuid(os.getuid()).pw_dir),
)
def test_runner_rejects_broad_agent_workspace(tmp_path, workspace):
    call_request = CodingAgentCallRequest(
        operation_id="agent_call_" + "1" * 32,
        role="candidate",
        cli="codex",
        model="test-model",
        prompt="complete prompt",
        workspace=workspace,
        workspace_policy=CodingAgentWorkspacePolicy.read_only(),
        timeout_seconds=10,
    )

    with pytest.raises(ValueError, match="broader than an allowed project"):
        runner(tmp_path).run(call_request, {"type": "object"})


def test_request_rejects_non_normalized_agent_workspace(tmp_path):
    with pytest.raises(ValueError, match="normalized"):
        CodingAgentCallRequest(
            operation_id="agent_call_" + "1" * 32,
            role="candidate",
            cli="codex",
            model="test-model",
            prompt="complete prompt",
            workspace=str(tmp_path / "target" / ".." / "target"),
            workspace_policy=CodingAgentWorkspacePolicy.read_only(),
            timeout_seconds=10,
        )


def test_runner_rejects_symlinked_agent_workspace(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked-target"
    symlink.symlink_to(target, target_is_directory=True)
    call_request = CodingAgentCallRequest(
        operation_id="agent_call_" + "1" * 32,
        role="candidate",
        cli="codex",
        model="test-model",
        prompt="complete prompt",
        workspace=str(symlink),
        workspace_policy=CodingAgentWorkspacePolicy.read_only(),
        timeout_seconds=10,
    )

    with pytest.raises(ValueError, match="must not traverse symlinks"):
        runner(tmp_path).run(call_request, {"type": "object"})


def test_credential_broker_exposes_only_the_selected_cli_auth_family(monkeypatch):
    values = {
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "AWS_ACCESS_KEY_ID": "aws-id",
        "AWS_BEARER_TOKEN_BEDROCK": "aws-bearer-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "CODEX_HOME": "/tmp/codex-auth",
        "GH_TOKEN": "github-secret",
        "OPENAI_API_KEY": "embedding-secret",
        "PRIVATE_REGISTRY_TOKEN": "registry-secret",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    codex_environment = coding_agent_credential_environment("codex")
    claude_environment = coding_agent_credential_environment("claude_code")

    assert codex_environment["CODEX_HOME"] == values["CODEX_HOME"]
    assert not set(codex_environment).intersection(
        {
            "ANTHROPIC_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GH_TOKEN",
            "OPENAI_API_KEY",
            "PRIVATE_REGISTRY_TOKEN",
        }
    )
    assert claude_environment["ANTHROPIC_API_KEY"] == values["ANTHROPIC_API_KEY"]
    assert claude_environment["AWS_ACCESS_KEY_ID"] == values["AWS_ACCESS_KEY_ID"]
    assert (
        claude_environment["AWS_BEARER_TOKEN_BEDROCK"]
        == values["AWS_BEARER_TOKEN_BEDROCK"]
    )
    assert (
        claude_environment["AWS_SECRET_ACCESS_KEY"] == values["AWS_SECRET_ACCESS_KEY"]
    )
    assert not set(claude_environment).intersection(
        {"CODEX_HOME", "GH_TOKEN", "OPENAI_API_KEY", "PRIVATE_REGISTRY_TOKEN"}
    )


def test_codex_receives_full_prompt_and_closed_schema_without_embedding_key(
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

    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"proposal": {"type": "string"}},
        "required": ["proposal"],
    }
    result = runner(tmp_path).run(request(tmp_path, "codex"), response_schema)

    args = json.loads((tmp_path / "codex_args.json").read_text())
    assert (tmp_path / "codex_stdin.txt").read_text() == request(
        tmp_path, "codex"
    ).prompt
    assert request(tmp_path, "codex").prompt not in args
    assert (tmp_path / "codex_env.txt").read_text() == "False"
    assert "--sandbox" not in args
    assert "--strict-config" in args
    assert "--ephemeral" in args
    assert "--skip-git-repo-check" in args
    assert "--ignore-user-config" in args
    assert args[args.index("--output-schema") + 1] == str(
        tmp_path
        / "artifacts"
        / request(tmp_path, "codex").operation_id
        / "response_schema.json"
    )
    assert "--search" in args
    permission_overrides = [
        args[position + 1] for position, value in enumerate(args) if value == "--config"
    ]
    assert 'default_permissions="kapso_ideation_read"' in permission_overrides
    assert any('"/proc"="deny"' in value for value in permission_overrides)
    assert any('"~/.config/gh"="deny"' in value for value in permission_overrides)
    assert any('"**/.env"="deny"' in value for value in permission_overrides)
    assert args[-1] == "-"
    assert json.loads(result.output) == {"proposal": "structured"}
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.cost_usd is None
    assert all(Path(path).is_file() for path in result.artifacts)
    assert Path(result.artifacts[0]).read_text() == request(tmp_path, "codex").prompt


def test_codex_omits_provider_constraint_for_open_object_schema(
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
pathlib.Path("codex_args.json").write_text(json.dumps(sys.argv[1:]))
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"metadata":{"source":"fixture"}}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    runner(tmp_path).run(
        request(tmp_path, "codex"),
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "metadata": {"type": "object", "minProperties": 1},
            },
            "required": ["metadata"],
        },
    )

    args = json.loads((tmp_path / "codex_args.json").read_text())
    assert "--output-schema" not in args


def test_coding_agent_output_is_validated_against_the_durable_schema(
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
final_path.write_text('{"proposal":1}')
print(json.dumps({
  "type":"turn.completed",
  "usage":{"input_tokens":1,"output_tokens":1}
}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    with pytest.raises(ValidationError):
        runner(tmp_path).run(
            request(tmp_path, "codex"),
            {
                "type": "object",
                "required": ["proposal"],
                "properties": {"proposal": {"type": "string"}},
            },
        )


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
    assert "--bare" not in args
    assert "--safe-mode" in args
    assert args[args.index("--setting-sources") + 1] == ""
    settings_payload = json.loads(args[args.index("--settings") + 1])
    assert settings_payload["sandbox"]["enabled"] is True
    assert settings_payload["sandbox"]["failIfUnavailable"] is True
    assert settings_payload["sandbox"]["filesystem"]["denyRead"] == ["/"]
    assert str(tmp_path) in settings_payload["sandbox"]["filesystem"]["allowRead"]
    assert "Read(//proc/**)" in settings_payload["permissions"]["deny"]
    assert "Read(~/.config/gh/**)" in settings_payload["permissions"]["deny"]
    assert args[args.index("--disallowedTools") + 1] == ("Bash,Edit,Write,NotebookEdit")
    assert json.loads(args[args.index("--json-schema") + 1]) == {"type": "object"}
    assert "--no-session-persistence" in args
    assert args[args.index("--tools") + 1] == "Read,WebSearch"
    assert json.loads(result.output) == {"proposal": "structured"}
    assert result.input_tokens == 13
    assert result.output_tokens == 5
    assert result.cost_usd == 0.25
    assert all(Path(path).is_file() for path in result.artifacts)


def test_claude_transport_omits_the_locally_enforced_schema_dialect(
    tmp_path,
    monkeypatch,
):
    install_executable(
        tmp_path,
        "claude",
        """#!/usr/bin/env python3
import json
import pathlib
import sys
pathlib.Path("claude_args.json").write_text(json.dumps(sys.argv[1:]))
print(json.dumps({
  "is_error": False,
  "structured_output": {"proposal": "structured"},
  "usage": {"input_tokens": 1, "output_tokens": 1},
  "total_cost_usd": 0.01
}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    response_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
    }

    runner(tmp_path).run(request(tmp_path, "claude_code"), response_schema)

    args = json.loads((tmp_path / "claude_args.json").read_text())
    assert json.loads(args[args.index("--json-schema") + 1]) == {"type": "object"}
    stored_schema = json.loads(
        (
            tmp_path
            / "artifacts"
            / request(tmp_path, "claude_code").operation_id
            / "response_schema.json"
        ).read_text()
    )
    assert stored_schema == response_schema


@pytest.mark.parametrize("cli", ("codex", "claude_code"))
def test_edit_workspace_call_seals_exact_replayable_delta(
    tmp_path,
    monkeypatch,
    cli,
):
    install_edit_executable(tmp_path, cli)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    workspace = editable_workspace(tmp_path)
    baseline = inspect_coding_agent_workspace(
        workspace,
        maximum_entries=WORKSPACE_ENTRY_LIMIT,
        maximum_bytes=WORKSPACE_BYTE_LIMIT,
    )
    call_request = editable_request(workspace, cli)

    result = runner(tmp_path).run(call_request, {"type": "object"})
    if cli == "codex":
        arguments = json.loads((tmp_path / "codex-edit-args.json").read_text())
        assert arguments[arguments.index("--cd") + 1] == str(workspace)
    sealed = seal_coding_agent_operation(
        request=call_request,
        response_schema={"type": "object"},
        principal_id="candidate_proposer",
        agent=CodingAgentSettings(
            cli=call_request.cli,
            model=call_request.model,
            timeout_seconds=call_request.timeout_seconds,
            effort=call_request.effort,
            allowed_tools=tuple(sorted(call_request.allowed_tools)),
        ),
        sensitive_file_glob_scan_max_depth=(SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH),
        result=result,
    )

    delta_path = next(
        Path(path)
        for path in result.artifacts
        if Path(path).name == CODING_AGENT_WORKSPACE_DELTA_FILENAME
    )
    delta_payload = delta_path.read_bytes()
    delta = CodingAgentWorkspaceDelta.from_json_bytes(delta_payload)
    edited = inspect_coding_agent_workspace(
        workspace,
        maximum_entries=WORKSPACE_ENTRY_LIMIT,
        maximum_bytes=WORKSPACE_BYTE_LIMIT,
    )
    reconstructed = reconstruct_edited_workspace(baseline, delta)
    assert delta.to_json_bytes() == delta_payload
    assert sealed.workspace_delta == delta
    assert sealed.final_output == result.output
    assert dict(sealed.artifact_bytes)["workspace-delta.json"] == delta_payload
    with pytest.raises(
        CodingAgentOperationReceiptError,
        match="directory names another operation",
    ):
        seal_coding_agent_operation(
            request=replace(
                call_request,
                operation_id="agent_call_" + "f" * 32,
            ),
            response_schema={"type": "object"},
            principal_id="candidate_proposer",
            agent=CodingAgentSettings(
                cli=call_request.cli,
                model=call_request.model,
                timeout_seconds=call_request.timeout_seconds,
                effort=call_request.effort,
                allowed_tools=tuple(sorted(call_request.allowed_tools)),
            ),
            sensitive_file_glob_scan_max_depth=(SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH),
            result=result,
        )
    assert delta.changed_paths == ("created.py", "existing.txt")
    assert delta.deleted_paths == ("deleted.txt",)
    assert reconstructed == edited
    assert result.workspace_delta_digest == tree_or_blob_digest(delta_payload)
    invocation = json.loads(
        (delta_path.parent / "invocation.json").read_text(encoding="utf-8")
    )
    assert invocation["workspace_policy"] == call_request.workspace_policy.to_dict()
    executable = "codex" if cli == "codex" else "claude"
    arguments = json.loads(
        (tmp_path / f"{executable}-edit-args.json").read_text(encoding="utf-8")
    )
    if cli == "codex":
        permission_overrides = [
            arguments[position + 1]
            for position, value in enumerate(arguments)
            if value == "--config"
        ]
        assert 'default_permissions="kapso_workspace_edit"' in permission_overrides
        assert any(
            '":workspace_roots"={"."="write"' in value for value in permission_overrides
        )
        assert any('"**/.git/**"="deny"' in value for value in permission_overrides)
    else:
        assert arguments[arguments.index("--permission-mode") + 1] == "acceptEdits"
        assert arguments[arguments.index("--disallowedTools") + 1] == (
            "Bash,NotebookEdit"
        )
        assert arguments[arguments.index("--tools") + 1] == (
            "Edit,Glob,Grep,Read,Write"
        )
    (delta_path.parent / "mcp_config.json").write_text(
        json.dumps(
            {"mcpServers": {"undeclared": {}}},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        CodingAgentOperationReceiptError,
        match="enables undeclared access",
    ):
        seal_coding_agent_operation(
            request=call_request,
            response_schema={"type": "object"},
            principal_id="candidate_proposer",
            agent=CodingAgentSettings(
                cli=call_request.cli,
                model=call_request.model,
                timeout_seconds=call_request.timeout_seconds,
                effort=call_request.effort,
                allowed_tools=tuple(sorted(call_request.allowed_tools)),
            ),
            sensitive_file_glob_scan_max_depth=(SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH),
            result=result,
        )


def test_completed_edit_call_replays_from_fresh_exact_parent(tmp_path, monkeypatch):
    install_edit_executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    first_workspace = editable_workspace(tmp_path)
    call_request = editable_request(first_workspace, "codex")
    call_runner = runner(tmp_path)
    first = call_runner.run(call_request, {"type": "object"})
    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir(mode=0o700)
    (fresh_root / "existing.txt").write_text("before", encoding="utf-8")
    (fresh_root / "deleted.txt").write_text("remove", encoding="utf-8")

    replayed = call_runner.run(
        replace(call_request, workspace=str(fresh_root)),
        {"type": "object"},
    )

    assert replayed == first
    assert (tmp_path / "edit-invocations.txt").read_text(encoding="utf-8") == "x"
    assert tuple(sorted(path.name for path in fresh_root.iterdir())) == (
        "deleted.txt",
        "existing.txt",
    )


def test_distinct_operations_cannot_concurrently_edit_one_workspace(
    tmp_path,
    monkeypatch,
):
    install_edit_executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    workspace = editable_workspace(tmp_path)
    first_request = editable_request(workspace, "codex")
    second_request = replace(
        first_request,
        operation_id="agent_call_" + "2" * 32,
    )
    call_runner = runner(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(call_runner.run, first_request, {"type": "object"}),
            executor.submit(call_runner.run, second_request, {"type": "object"}),
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
    assert "expected tree" in str(errors[0])
    assert (tmp_path / "edit-invocations.txt").read_text(encoding="utf-8") == "x"


def test_edit_call_executes_and_seals_through_one_pinned_workspace_descriptor(
    tmp_path,
    monkeypatch,
):
    ready = tmp_path / "ready.pipe"
    resume = tmp_path / "resume.pipe"
    edited = tmp_path / "edited.pipe"
    restored = tmp_path / "restored.pipe"
    for pipe in (ready, resume, edited, restored):
        os.mkfifo(pipe)
    install_executable(
        tmp_path,
        "codex",
        f"""#!/usr/bin/env python3
import json
import pathlib
import sys
with pathlib.Path({str(ready)!r}).open("w") as handle:
    handle.write("ready")
with pathlib.Path({str(resume)!r}).open("r") as handle:
    handle.read()
pathlib.Path("existing.txt").write_text("after")
pathlib.Path("created.py").write_text("print('created')\\n")
pathlib.Path("deleted.txt").unlink()
with pathlib.Path({str(edited)!r}).open("w") as handle:
    handle.write("edited")
with pathlib.Path({str(restored)!r}).open("r") as handle:
    handle.read()
arguments = sys.argv[1:]
final_path = pathlib.Path(arguments[arguments.index("--output-last-message") + 1])
final_path.write_text('{{"changed_paths":["created.py","existing.txt"],"deleted_paths":["deleted.txt"]}}')
print(json.dumps({{"type":"turn.completed","usage":{{"input_tokens":3,"output_tokens":2}}}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    workspace_parent = tmp_path / "workspace-parent"
    workspace_parent.mkdir(mode=0o700)
    workspace = workspace_parent / "workspace"
    workspace.mkdir(mode=0o700)
    (workspace / "existing.txt").write_text("before", encoding="utf-8")
    (workspace / "deleted.txt").write_text("remove", encoding="utf-8")
    call_request = editable_request(workspace, "codex")
    moved_parent = tmp_path / "moved-workspace-parent"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            runner(tmp_path).run,
            call_request,
            {"type": "object"},
        )
        with ready.open("r") as handle:
            assert handle.read() == "ready"
        workspace_parent.rename(moved_parent)
        workspace_parent.mkdir(mode=0o700)
        replacement_workspace = workspace_parent / "workspace"
        replacement_workspace.mkdir(mode=0o700)
        (replacement_workspace / "substituted.py").write_bytes(b"substituted")
        with resume.open("w") as handle:
            handle.write("resume")
        with edited.open("r") as handle:
            assert handle.read() == "edited"
        (replacement_workspace / "substituted.py").unlink()
        replacement_workspace.rmdir()
        workspace_parent.rmdir()
        moved_parent.rename(workspace_parent)
        with restored.open("w") as handle:
            handle.write("restored")
        result = future.result()

    delta_path = next(
        Path(path)
        for path in result.artifacts
        if Path(path).name == CODING_AGENT_WORKSPACE_DELTA_FILENAME
    )
    delta = CodingAgentWorkspaceDelta.from_json_bytes(delta_path.read_bytes())
    assert delta.changed_paths == ("created.py", "existing.txt")
    assert delta.deleted_paths == ("deleted.txt",)
    assert not (workspace / "substituted.py").exists()


def test_edit_call_rejects_text_workspace_that_differs_from_outer_authority(
    tmp_path,
    monkeypatch,
):
    install_edit_executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    workspace = editable_workspace(tmp_path)
    call_request = editable_request(workspace, "codex")
    authority_descriptor = os.open(
        workspace,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    moved_workspace = tmp_path / "moved-workspace"
    workspace.rename(moved_workspace)
    workspace.mkdir(mode=0o700)
    (workspace / "existing.txt").write_text("before", encoding="utf-8")
    (workspace / "deleted.txt").write_text("remove", encoding="utf-8")

    with pytest.raises(CodingAgentInvocationError, match="differs from its authority"):
        runner(tmp_path).run(
            call_request,
            {"type": "object"},
            workspace_authority_descriptor=authority_descriptor,
        )

    os.close(authority_descriptor)
    (workspace / "existing.txt").unlink()
    (workspace / "deleted.txt").unlink()
    workspace.rmdir()
    moved_workspace.rename(workspace)
    assert not (tmp_path / "edit-invocations.txt").exists()


def test_incomplete_edit_artifacts_rerun_only_from_exact_parent(tmp_path, monkeypatch):
    install_edit_executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    first_workspace = editable_workspace(tmp_path)
    call_request = editable_request(first_workspace, "codex")
    call_runner = runner(tmp_path)
    result = call_runner.run(call_request, {"type": "object"})
    result_path = Path(result.artifacts[0]).parent / "result.json"
    result_path.unlink()

    with pytest.raises(CodingAgentInvocationError, match="expected tree"):
        call_runner.run(call_request, {"type": "object"})

    fresh_root = tmp_path / "fresh-after-crash"
    fresh_root.mkdir(mode=0o700)
    (fresh_root / "existing.txt").write_text("before", encoding="utf-8")
    (fresh_root / "deleted.txt").write_text("remove", encoding="utf-8")
    recovered = call_runner.run(
        replace(call_request, workspace=str(fresh_root)),
        {"type": "object"},
    )
    assert recovered.workspace_delta_digest is not None
    assert (tmp_path / "edit-invocations.txt").read_text(encoding="utf-8") == "xx"


def test_cached_edit_rejects_workspace_delta_tampering(tmp_path, monkeypatch):
    install_edit_executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    workspace = editable_workspace(tmp_path)
    call_request = editable_request(workspace, "codex")
    call_runner = runner(tmp_path)
    result = call_runner.run(call_request, {"type": "object"})
    delta_path = next(
        Path(path)
        for path in result.artifacts
        if Path(path).name == CODING_AGENT_WORKSPACE_DELTA_FILENAME
    )
    payload = json.loads(delta_path.read_text(encoding="utf-8"))
    payload["changed_files"][0]["content_base64"] = "dGFtcGVyZWQ="
    delta_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(CodingAgentInvocationError, match="delta conflicts"):
        call_runner.run(call_request, {"type": "object"})


def test_edit_call_rejects_wrong_parent_before_cli_invocation(tmp_path, monkeypatch):
    install_edit_executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    workspace = editable_workspace(tmp_path)
    call_request = replace(
        editable_request(workspace, "codex"),
        workspace_policy=CodingAgentWorkspacePolicy.edit_workspace(
            expected_tree_hash="sha256:" + "0" * 64,
            maximum_entries=WORKSPACE_ENTRY_LIMIT,
            maximum_bytes=WORKSPACE_BYTE_LIMIT,
        ),
    )

    with pytest.raises(CodingAgentInvocationError, match="expected tree"):
        runner(tmp_path).run(call_request, {"type": "object"})

    assert not (tmp_path / "edit-invocations.txt").exists()


def test_workspace_delta_rejects_file_descendant_collision(tmp_path):
    workspace = tmp_path / "empty-workspace"
    workspace.mkdir(mode=0o700)
    baseline = inspect_coding_agent_workspace(
        workspace,
        maximum_entries=WORKSPACE_ENTRY_LIMIT,
        maximum_bytes=WORKSPACE_BYTE_LIMIT,
    )
    content = b"collision"
    changes = tuple(
        CodingAgentWorkspaceChangedFile(
            before=None,
            after=SourceFileDescriptor(
                relative_path=path,
                digest=tree_or_blob_digest(content),
                mode="100644",
                size=len(content),
            ),
            content_base64=base64.b64encode(content).decode("ascii"),
        )
        for path in ("module", "module/child.py")
    )
    delta = CodingAgentWorkspaceDelta.mint(
        baseline_tree_hash=baseline.tree_hash,
        edited_tree_hash="sha256:" + "1" * 64,
        changed_files=changes,
        deleted_files=(),
    )

    with pytest.raises(ValueError, match="collides with a descendant"):
        validate_coding_agent_workspace_delta(baseline, delta)


@pytest.mark.parametrize(
    "unsafe_entry",
    ("symlink", "hardlink", "fifo", "socket", "oversized"),
)
def test_edit_call_rejects_unsafe_workspace_without_hanging(
    tmp_path,
    monkeypatch,
    unsafe_entry,
):
    install_edit_executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    workspace = editable_workspace(tmp_path)
    target = workspace / "unsafe"
    if unsafe_entry == "symlink":
        target.symlink_to(tmp_path / "outside")
    elif unsafe_entry == "hardlink":
        os.link(workspace / "existing.txt", target)
    elif unsafe_entry == "fifo":
        os.mkfifo(target, mode=0o600)
    elif unsafe_entry == "socket":
        endpoint = socket.socket(socket.AF_UNIX)
        endpoint.bind(str(target))
    else:
        target.write_bytes(b"too large")
    policy = CodingAgentWorkspacePolicy.edit_workspace(
        expected_tree_hash="sha256:" + "0" * 64,
        maximum_entries=WORKSPACE_ENTRY_LIMIT,
        maximum_bytes=(1 if unsafe_entry == "oversized" else WORKSPACE_BYTE_LIMIT),
    )
    call_request = replace(
        request(workspace, "codex"),
        allowed_tools=("Read",),
        workspace_policy=policy,
    )

    with pytest.raises(ValueError, match="workspace"):
        runner(tmp_path).run(call_request, {"type": "object"})

    if unsafe_entry == "socket":
        endpoint.close()
    assert not (tmp_path / "edit-invocations.txt").exists()


@pytest.mark.parametrize("cli", ("codex", "claude_code"))
def test_prior_packet_mount_is_explicit_auditable_and_credential_isolated(
    tmp_path,
    monkeypatch,
    cli,
):
    executable = "codex" if cli == "codex" else "claude"
    if cli == "codex":
        source = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys
pathlib.Path("prior_agent_args.json").write_text(json.dumps(sys.argv[1:]))
pathlib.Path("prior_agent_env.json").write_text(json.dumps(sorted(os.environ)))
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"structured"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":7}}))
"""
    else:
        source = """#!/usr/bin/env python3
import json
import os
import pathlib
import sys
pathlib.Path("prior_agent_args.json").write_text(json.dumps(sys.argv[1:]))
pathlib.Path("prior_agent_env.json").write_text(json.dumps(sorted(os.environ)))
print(json.dumps({
  "is_error": False,
  "structured_output": {"proposal": "structured"},
  "usage": {"input_tokens": 13, "output_tokens": 5},
  "total_cost_usd": 0.25
}))
"""
    install_executable(tmp_path, executable, source)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    secret_names = (
        "OPENAI_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "SSH_AUTH_SOCK",
        "GIT_ASKPASS",
        "DATABASE_URL",
        "PRIVATE_REGISTRY_TOKEN",
    )
    for secret_name in secret_names:
        monkeypatch.setenv(secret_name, "must-not-reach-agent")
    materialization = access_materialization()
    call_request = replace(
        request(tmp_path, cli),
        prior_knowledge=materialization,
    )

    result = runner(tmp_path).run(call_request, {"type": "object"})

    artifact_by_name = {Path(path).name: Path(path) for path in result.artifacts}
    persisted_packet = json.loads(
        artifact_by_name["prior_knowledge.json"].read_text(encoding="utf-8")
    )
    assert persisted_packet == materialization.to_dict()
    mcp_config = json.loads(
        artifact_by_name["mcp_config.json"].read_text(encoding="utf-8")
    )
    server = mcp_config["mcpServers"]["prior_knowledge"]
    assert Path(server["command"]).name == "env"
    assert server["args"][0] == "-i"
    assert any(argument.startswith("PYTHONPATH=") for argument in server["args"])
    assert "--enabled-gates" in server["args"]
    assert server["args"][server["args"].index("--enabled-gates") + 1] == (
        "prior_knowledge"
    )
    assert "--operation-id" in server["args"]
    assert server["args"][server["args"].index("--operation-id") + 1] == (
        call_request.operation_id
    )
    assert "--prior-knowledge-audit-maximum-bytes" in server["args"]
    assert server["args"][
        server["args"].index("--prior-knowledge-audit-maximum-bytes") + 1
    ] == str(len(materialization.to_json_bytes()))
    assert artifact_by_name["mcp_audit.jsonl"].read_text(encoding="utf-8") == ""
    agent_environment = set(
        json.loads((tmp_path / "prior_agent_env.json").read_text(encoding="utf-8"))
    )
    assert not agent_environment.intersection(secret_names)
    agent_arguments = json.loads(
        (tmp_path / "prior_agent_args.json").read_text(encoding="utf-8")
    )
    if cli == "codex":
        assert any(
            argument.startswith("mcp_servers.prior_knowledge.command=")
            for argument in agent_arguments
        )
    else:
        assert "--strict-mcp-config" in agent_arguments
        assert agent_arguments[agent_arguments.index("--mcp-config") + 1] == str(
            artifact_by_name["mcp_config.json"]
        )
        tools = agent_arguments[agent_arguments.index("--tools") + 1].split(",")
        assert "mcp__prior_knowledge__list_prior_knowledge" in tools
        assert "mcp__prior_knowledge__get_prior_knowledge_record" in tools
    sealed = seal_coding_agent_operation(
        request=call_request,
        response_schema={"type": "object"},
        principal_id="candidate_proposer",
        agent=CodingAgentSettings(
            cli=call_request.cli,
            model=call_request.model,
            timeout_seconds=call_request.timeout_seconds,
            effort=call_request.effort,
            allowed_tools=tuple(sorted(call_request.allowed_tools)),
        ),
        sensitive_file_glob_scan_max_depth=SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH,
        result=result,
    )
    assert sealed.receipt.operation_id == call_request.operation_id
    server["command"] = "/tmp/untrusted/env"
    server["args"][1] = "PYTHONPATH=/tmp/untrusted-package"
    server["args"][2] = "/tmp/untrusted-python"
    artifact_by_name["mcp_config.json"].write_text(
        json.dumps(mcp_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        CodingAgentOperationReceiptError,
        match="MCP configuration artifact differs from request",
    ):
        seal_coding_agent_operation(
            request=call_request,
            response_schema={"type": "object"},
            principal_id="candidate_proposer",
            agent=CodingAgentSettings(
                cli=call_request.cli,
                model=call_request.model,
                timeout_seconds=call_request.timeout_seconds,
                effort=call_request.effort,
                allowed_tools=tuple(sorted(call_request.allowed_tools)),
            ),
            sensitive_file_glob_scan_max_depth=SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH,
            result=result,
        )


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

    changed_access = replace(
        request(tmp_path, "codex"),
        workspace_policy=CodingAgentWorkspacePolicy.edit_workspace(
            expected_tree_hash="sha256:" + "0" * 64,
            maximum_entries=WORKSPACE_ENTRY_LIMIT,
            maximum_bytes=WORKSPACE_BYTE_LIMIT,
        ),
    )
    with pytest.raises(CodingAgentInvocationError, match="identity was reused"):
        call_runner.run(changed_access, {"type": "object"})

    changed_security_runner = SubprocessCodingAgentCallRunner(
        replace(
            call_runner.settings,
            sensitive_file_glob_scan_max_depth=(
                call_runner.settings.sensitive_file_glob_scan_max_depth + 1
            ),
        )
    )
    with pytest.raises(CodingAgentInvocationError, match="identity was reused"):
        changed_security_runner.run(
            request(tmp_path, "codex"),
            {"type": "object"},
        )


def test_cached_operation_rejects_semantically_corrupt_prior_audit(
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
final_path.write_text('{"proposal":"structured"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":7}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    materialization = access_materialization()
    call_request = replace(
        request(tmp_path, "codex"),
        prior_knowledge=materialization,
    )
    call_runner = runner(tmp_path)
    result = call_runner.run(call_request, {"type": "object"})
    artifacts = {Path(path).name: Path(path) for path in result.artifacts}
    selected_id = materialization.prior_knowledge_snapshot.selected_record_ids[0]
    corrupt_event = {
        "arguments": {},
        "operation_id": call_request.operation_id,
        "prior_knowledge_snapshot_id": (
            materialization.prior_knowledge_snapshot.prior_knowledge_snapshot_id
        ),
        "response_digest": "sha256:" + "0" * 64,
        "returned_ids": [selected_id],
        "tool_name": "get_prior_knowledge_record",
    }
    artifacts["mcp_audit.jsonl"].write_text(
        json.dumps(corrupt_event, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CodingAgentInvocationError, match="get audit arguments"):
        verify_coding_agent_operation_artifacts(
            operation_id=call_request.operation_id,
            workspace_access=CodingAgentWorkspaceAccess.READ_ONLY,
            artifact_bytes={
                name: (artifacts["mcp_audit.jsonl"].parent / name).read_bytes()
                for name in coding_agent_artifact_filenames(
                    CodingAgentWorkspaceAccess.READ_ONLY
                )
            },
        )

    with pytest.raises(CodingAgentInvocationError, match="get audit arguments"):
        call_runner.run(call_request, {"type": "object"})


def valid_list_audit(call_request):
    access = PriorKnowledgeAccess(call_request.prior_knowledge)
    member_ids = sorted(
        set(access.packet.selected_record_ids) | set(access.packet.proof_reference_ids)
    )
    event = {
        "arguments": {},
        "operation_id": call_request.operation_id,
        "prior_knowledge_snapshot_id": access.packet.prior_knowledge_snapshot_id,
        "response_digest": tree_or_blob_digest(
            canonical_json_bytes(access.list_response_payload())
        ),
        "returned_ids": member_ids,
        "tool_name": "list_prior_knowledge",
    }
    return canonical_json_bytes(event).decode("utf-8") + "\n"


def test_mcp_audit_requires_canonical_unique_json_and_reconstructible_response():
    call_request = replace(
        request(Path("/tmp"), "codex"),
        prior_knowledge=access_materialization(),
    )
    valid_audit = valid_list_audit(call_request)

    count, digest = SubprocessCodingAgentCallRunner._validate_mcp_audit(
        call_request,
        valid_audit,
    )

    assert count == 1
    assert digest == tree_or_blob_digest(valid_audit.encode("utf-8"))
    with pytest.raises(CodingAgentInvocationError, match="incomplete final event"):
        SubprocessCodingAgentCallRunner._validate_mcp_audit(
            call_request,
            valid_audit.rstrip("\n"),
        )
    event = json.loads(valid_audit)
    event["response_digest"] = "sha256:" + "0" * 64
    with pytest.raises(CodingAgentInvocationError, match="digest is inconsistent"):
        SubprocessCodingAgentCallRunner._validate_mcp_audit(
            call_request,
            canonical_json_bytes(event).decode("utf-8") + "\n",
        )
    with pytest.raises(CodingAgentInvocationError, match="not canonical JSON"):
        SubprocessCodingAgentCallRunner._validate_mcp_audit(
            call_request,
            json.dumps(json.loads(valid_audit), sort_keys=True) + "\n",
        )
    duplicate_key_audit = valid_audit.replace(
        '"arguments":{}',
        '"arguments":{},"arguments":{}',
    )
    with pytest.raises(CodingAgentInvocationError, match="duplicate JSON key"):
        SubprocessCodingAgentCallRunner._validate_mcp_audit(
            call_request,
            duplicate_key_audit,
        )


def test_cached_result_is_bound_to_the_exact_audit(tmp_path, monkeypatch):
    install_executable(
        tmp_path,
        "codex",
        """#!/usr/bin/env python3
import json
import pathlib
import sys
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"structured"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    call_runner = runner(tmp_path)
    call_request = request(tmp_path, "codex")
    result = call_runner.run(call_request, {"type": "object"})
    result_path = Path(result.artifacts[0]).parent / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["mcp_audit_event_count"] = 1
    result_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CodingAgentInvocationError, match="conflicts with completed"):
        call_runner.run(call_request, {"type": "object"})


def test_cached_result_is_bound_to_exact_final_artifact(tmp_path, monkeypatch):
    install_executable(
        tmp_path,
        "codex",
        """#!/usr/bin/env python3
import json
import pathlib
import sys
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"original"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    call_runner = runner(tmp_path)
    call_request = request(tmp_path, "codex")
    result = call_runner.run(call_request, {"type": "object"})
    final_path = next(
        Path(path) for path in result.artifacts if Path(path).name == "final.json"
    )
    final_path.write_text('{"proposal":"tampered"}', encoding="utf-8")

    with pytest.raises(CodingAgentInvocationError, match="final output conflicts"):
        call_runner.run(call_request, {"type": "object"})


def test_cached_result_output_is_bound_to_exact_final_artifact(tmp_path, monkeypatch):
    install_executable(
        tmp_path,
        "codex",
        """#!/usr/bin/env python3
import json
import pathlib
import sys
args = sys.argv[1:]
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text('{"proposal":"original"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    call_runner = runner(tmp_path)
    call_request = request(tmp_path, "codex")
    result = call_runner.run(call_request, {"type": "object"})
    result_path = Path(result.artifacts[0]).parent / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["output"] = '{"proposal":"tampered"}'
    result_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CodingAgentInvocationError, match="parsed output conflicts"):
        call_runner.run(call_request, {"type": "object"})


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
    assert (artifact_directories[0] / "mcp_audit.jsonl").is_file()


def test_runner_rejects_non_absolute_artifact_root():
    with pytest.raises(ValueError, match="absolute"):
        CodingAgentRunnerSettings(
            artifact_root="relative/artifacts",
            termination_grace_seconds=1,
            sensitive_file_glob_scan_max_depth=(SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH),
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

    with pytest.raises(CodingAgentInvocationError, match="private directory"):
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

    with pytest.raises(CodingAgentInvocationError, match="independent file"):
        runner(tmp_path).run(request(tmp_path, "codex"), {"type": "object"})

    assert outside.read_text() == request(tmp_path, "codex").prompt


@pytest.mark.parametrize("corruption", ("public_file", "hardlink", "public_directory"))
def test_completed_operation_rejects_nonprivate_artifacts(
    tmp_path,
    monkeypatch,
    corruption,
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
final_path.write_text('{"proposal":"original"}')
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}))
""",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    call_runner = runner(tmp_path)
    call_request = request(tmp_path, "codex")
    result = call_runner.run(call_request, {"type": "object"})
    operation = Path(result.artifacts[0]).parent
    final_path = operation / "final.json"
    if corruption == "public_file":
        final_path.chmod(0o644)
    elif corruption == "hardlink":
        os.link(final_path, tmp_path / "linked-final")
    else:
        operation.chmod(0o755)

    with pytest.raises(CodingAgentInvocationError, match="private"):
        call_runner.run(call_request, {"type": "object"})


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
            sensitive_file_glob_scan_max_depth=(SENSITIVE_FILE_GLOB_SCAN_MAX_DEPTH),
        )
    )

    with pytest.raises(CodingAgentInvocationError, match="must not traverse"):
        call_runner.run(request(tmp_path, "codex"), {"type": "object"})

    assert not (outside / "agent-calls").exists()
