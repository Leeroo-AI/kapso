from __future__ import annotations

import os
import subprocess
from contextlib import ExitStack
from dataclasses import replace

import pytest

from kapso.cross_run.launch.run_action_coding_agent_consumer import (
    NATIVE_CODING_AGENT_CONSUMER_ID,
    NATIVE_CODING_AGENT_CONSUMER_VERSION,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    read_canonical_coding_agent_result,
)
from kapso.cross_run.launch.run_action_contracts import RunFrontierWorkspaceAccess
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier_with_limits,
    inspect_run_workspace_source_tree,
)
from test_run_action_coding_agent_consumer import (
    _open_runtime_descriptors,
    _runtime_directories,
)
from test_run_action_coding_agent_contracts import (
    interpretation_policy,
    run_action_request,
)

_OFFLINE_IMAGE = "kapso/coding-agent-offline:m9"


@pytest.mark.parametrize(
    "workspace_access",
    (
        RunFrontierWorkspaceAccess.READ_ONLY,
        RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
    ),
)
def test_offline_coding_agent_runs_through_real_image(tmp_path, workspace_access):
    workspace, _source, temporary = _runtime_directories(tmp_path)
    input_directory = tmp_path / "input"
    input_directory.mkdir(mode=0o700)
    with ExitStack() as resources:
        workspace_descriptor, _temporary_descriptor = _open_runtime_descriptors(
            workspace,
            temporary,
            resources,
        )
        predecessor = inspect_run_workspace_source_tree(
            workspace_descriptor,
            maximum_entries=10_000,
            maximum_bytes=1_073_741_824,
        ).source_tree_digest
    policy = interpretation_policy(
        consumer_id=NATIVE_CODING_AGENT_CONSUMER_ID,
        consumer_version=NATIVE_CODING_AGENT_CONSUMER_VERSION,
        workspace_access=workspace_access,
        web_search_enabled=False,
    )
    request = run_action_request(
        policy,
        predecessor_digest=(
            predecessor
            if workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
            else None
        ),
    )
    if workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE:
        request = replace(request, prompt=f"OFFLINE_EDIT\n{request.prompt}")
    request_path = input_directory / "request.blob"
    request_path.write_bytes(request.to_json_bytes())
    request_path.chmod(0o600)
    request_bound = input_directory / "request.maximum_bytes"
    request_bound.write_text(
        f"{policy.maximum_request_bytes}\n",
        encoding="ascii",
    )
    request_bound.chmod(0o600)

    completed = subprocess.run(
        (
            "/usr/bin/docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "KILL",
            "--cap-add",
            "SETGID",
            "--cap-add",
            "SETPCAP",
            "--cap-add",
            "SETUID",
            "--group-add",
            "1001",
            "--security-opt",
            "apparmor=docker-default",
            "--security-opt",
            "no-new-privileges",
            "--security-opt",
            "seccomp=builtin",
            "--pids-limit",
            "128",
            "--init",
            "--user",
            "0:0",
            "--workdir",
            "/kapso/workspace",
            "--mount",
            f"type=bind,src={workspace},dst=/kapso/workspace",
            "--mount",
            f"type=bind,src={input_directory},dst=/kapso/input,readonly",
            "--mount",
            f"type=bind,src={temporary},dst=/kapso/tmp",
            _OFFLINE_IMAGE,
            "/usr/local/bin/kapso-coding-agent-supervisor",
            "--supervisor-user-id",
            str(policy.supervisor_user_id),
            "--supervisor-group-id",
            str(policy.supervisor_group_id),
            "--provider-user-id",
            str(policy.provider_user_id),
            "--provider-group-id",
            str(policy.provider_group_id),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    result = read_canonical_coding_agent_result(
        (temporary / "result.candidate").read_bytes()
    )
    result.validate_against(policy=policy, request=request)
    assert result.structured_output == {"answer": "offline boundary passed"}
    with ExitStack() as resources:
        workspace_descriptor, _temporary_descriptor = _open_runtime_descriptors(
            workspace,
            temporary,
            resources,
        )
        frontier = inspect_run_workspace_frontier_with_limits(
            workspace_descriptor,
            workspace_git_branch=policy.workspace_git_branch,
            maximum_source_entries=policy.maximum_workspace_entries,
            maximum_source_bytes=policy.maximum_workspace_bytes,
            maximum_git_entries=policy.maximum_workspace_git_entries,
            maximum_git_bytes=policy.maximum_workspace_git_bytes,
            expected_commit_sha=None,
        )
    if workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE:
        assert frontier.source_tree_digest != predecessor
        assert frontier.parent_commit_shas
        assert result.edited_source_tree_digest == frontier.source_tree_digest
    else:
        assert frontier.source_tree_digest == predecessor
        assert result.edited_source_tree_digest is None
    assert os.stat(temporary).st_gid == os.getegid()
