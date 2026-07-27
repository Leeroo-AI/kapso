from __future__ import annotations

import os
import stat
from contextlib import ExitStack

import pytest

from kapso.cross_run.launch.run_action_coding_agent_scratch import (
    prepare_coding_agent_scratch_layout,
    PROVIDER_SHARED_DIRECTORY_MODE,
    PROVIDER_SHARED_EXECUTABLE_MODE,
    PROVIDER_SHARED_FILE_MODE,
    RunActionCodingAgentScratchError,
    inspect_coding_agent_scratch_source_tree,
    require_coding_agent_scratch_support,
    sanitize_coding_agent_scratch_successor,
    share_coding_agent_scratch_source_tree,
)
from kapso.cross_run.launch.run_action_contracts import RunFrontierWorkspaceAccess
from kapso.cross_run.launch.workspace_frontier import (
    copy_run_workspace_source_tree,
    inspect_run_workspace_source_tree,
    plan_run_workspace_source_copy,
    replace_run_workspace_source_tree,
)
from test_launch_resolver import resolver_case
from test_run_action_workspace_copy import _open_destination, _open_source
from test_run_action_coding_agent_contracts import (
    interpretation_policy,
    run_action_request,
)
from test_run_state_publisher import publisher_case


def _prepare_detached_source(case, tmp_path, resources):
    settings = case["settings"]
    source_descriptor, source_frontier = _open_source(case, resources)
    plan = plan_run_workspace_source_copy(
        source_descriptor,
        expected=source_frontier,
        maximum_source_entries=settings.run_workspace_entry_limit,
        maximum_source_bytes=settings.run_workspace_size_bytes,
        maximum_git_entries=settings.run_workspace_git_entry_limit,
        maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
    )
    destination_path = tmp_path / "provider-workspace"
    destination_descriptor = _open_destination(destination_path, resources)
    detached = copy_run_workspace_source_tree(
        source_descriptor,
        destination_descriptor,
        plan=plan,
        maximum_source_entries=settings.run_workspace_entry_limit,
        maximum_source_bytes=settings.run_workspace_size_bytes,
        maximum_git_entries=settings.run_workspace_git_entry_limit,
        maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
    )
    return destination_path, destination_descriptor, detached


def test_detached_source_becomes_exact_group_shared_provider_authority(
    publisher_case,
    tmp_path,
):
    settings = publisher_case["settings"]
    with ExitStack() as resources:
        destination_path, destination_descriptor, detached = _prepare_detached_source(
            publisher_case, tmp_path, resources
        )

        shared = share_coding_agent_scratch_source_tree(
            destination_descriptor,
            expected_source=detached,
            supervisor_user_id=os.geteuid(),
            provider_user_id=os.geteuid() + 1,
            provider_group_id=os.getegid(),
            maximum_entries=settings.run_workspace_entry_limit,
            maximum_bytes=settings.run_workspace_size_bytes,
        )
        repeated = inspect_coding_agent_scratch_source_tree(
            destination_descriptor,
            supervisor_user_id=os.geteuid(),
            provider_user_id=os.geteuid() + 1,
            provider_group_id=os.getegid(),
            maximum_entries=settings.run_workspace_entry_limit,
            maximum_bytes=settings.run_workspace_size_bytes,
        )

    assert shared == repeated
    assert shared.source.source_tree_digest == detached.source_tree_digest
    assert stat.S_IMODE(destination_path.stat().st_mode) == (
        PROVIDER_SHARED_DIRECTORY_MODE
    )
    assert not (destination_path / ".git").exists()
    for path in destination_path.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        assert metadata.st_gid == os.getegid()
        if path.is_dir():
            assert stat.S_IMODE(metadata.st_mode) == PROVIDER_SHARED_DIRECTORY_MODE
        else:
            assert stat.S_IMODE(metadata.st_mode) in {
                PROVIDER_SHARED_FILE_MODE,
                PROVIDER_SHARED_EXECUTABLE_MODE,
            }


def test_shared_scratch_rejects_a_provider_inaccessible_file(
    publisher_case,
    tmp_path,
):
    settings = publisher_case["settings"]
    with ExitStack() as resources:
        destination_path, destination_descriptor, detached = _prepare_detached_source(
            publisher_case, tmp_path, resources
        )
        share_coding_agent_scratch_source_tree(
            destination_descriptor,
            expected_source=detached,
            supervisor_user_id=os.geteuid(),
            provider_user_id=os.geteuid() + 1,
            provider_group_id=os.getegid(),
            maximum_entries=settings.run_workspace_entry_limit,
            maximum_bytes=settings.run_workspace_size_bytes,
        )
        regular_file = next(
            path for path in destination_path.rglob("*") if path.is_file()
        )
        regular_file.chmod(0o600)

        with pytest.raises(
            RunActionCodingAgentScratchError,
            match="inaccessible or unsafe",
        ):
            inspect_coding_agent_scratch_source_tree(
                destination_descriptor,
                supervisor_user_id=os.geteuid(),
                provider_user_id=os.geteuid() + 1,
                provider_group_id=os.getegid(),
                maximum_entries=settings.run_workspace_entry_limit,
                maximum_bytes=settings.run_workspace_size_bytes,
            )


def test_shared_scratch_rejects_injected_git_authority(
    publisher_case,
    tmp_path,
):
    settings = publisher_case["settings"]
    with ExitStack() as resources:
        destination_path, destination_descriptor, detached = _prepare_detached_source(
            publisher_case, tmp_path, resources
        )
        share_coding_agent_scratch_source_tree(
            destination_descriptor,
            expected_source=detached,
            supervisor_user_id=os.geteuid(),
            provider_user_id=os.geteuid() + 1,
            provider_group_id=os.getegid(),
            maximum_entries=settings.run_workspace_entry_limit,
            maximum_bytes=settings.run_workspace_size_bytes,
        )
        injected = destination_path / ".git"
        injected.mkdir(mode=0o770)
        injected.chmod(PROVIDER_SHARED_DIRECTORY_MODE)

        with pytest.raises(
            RunActionCodingAgentScratchError,
            match="denied source path",
        ):
            inspect_coding_agent_scratch_source_tree(
                destination_descriptor,
                supervisor_user_id=os.geteuid(),
                provider_user_id=os.geteuid() + 1,
                provider_group_id=os.getegid(),
                maximum_entries=settings.run_workspace_entry_limit,
                maximum_bytes=settings.run_workspace_size_bytes,
            )


def test_provider_scratch_edit_is_sanitized_and_reprojected_without_git(
    publisher_case,
    tmp_path,
):
    provider_groups = tuple(
        group_id for group_id in os.getgroups() if group_id != os.getegid()
    )
    if not provider_groups:
        pytest.skip("scratch group transition requires one supplemental group")
    provider_group_id = provider_groups[0]
    settings = publisher_case["settings"]
    with ExitStack() as resources:
        workspace_descriptor, frontier = _open_source(publisher_case, resources)
        temporary_path = tmp_path / "scratch-root"
        temporary_descriptor = _open_destination(temporary_path, resources)
        policy = interpretation_policy(
            consumer_id="kapso.native_coding_agent_consumer",
            consumer_version="kapso.native_coding_agent_consumer.v1",
            workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
            web_search_enabled=False,
            workspace_git_branch=frontier.branch,
            supervisor_user_id=os.geteuid(),
            supervisor_group_id=os.getegid(),
            provider_user_id=os.geteuid() + 1,
            provider_group_id=provider_group_id,
        )
        request = run_action_request(
            policy,
            predecessor_digest=frontier.source_tree_digest,
        )
        layout = prepare_coding_agent_scratch_layout(
            trusted_workspace_descriptor=workspace_descriptor,
            temporary_root_descriptor=temporary_descriptor,
            trusted_frontier=frontier,
            request=request,
            support_payloads={
                "/kapso/tmp/provider-support/response.schema.json": b"{}"
            },
            resources=resources,
        )
        require_coding_agent_scratch_support(layout)
        scratch_file = next(
            path
            for path in (temporary_path / "provider-workspace").rglob("*")
            if path.is_file()
        )
        scratch_file.write_bytes(scratch_file.read_bytes() + b"# provider edit\n")
        scratch_file.chmod(PROVIDER_SHARED_FILE_MODE)
        os.chown(scratch_file, -1, provider_group_id)
        layout.restore_temporary_root()
        successor_descriptor, sanitized = sanitize_coding_agent_scratch_successor(
            layout,
            request=request,
            resources=resources,
        )
        replaced = replace_run_workspace_source_tree(
            workspace_descriptor,
            successor_descriptor,
            predecessor=frontier,
            maximum_source_entries=settings.run_workspace_entry_limit,
            maximum_source_bytes=settings.run_workspace_size_bytes,
            maximum_git_entries=settings.run_workspace_git_entry_limit,
            maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
        )
        observed = inspect_run_workspace_source_tree(
            workspace_descriptor,
            maximum_entries=settings.run_workspace_entry_limit,
            maximum_bytes=settings.run_workspace_size_bytes,
        )

    assert sanitized.source_tree_digest != frontier.source_tree_digest
    assert replaced == observed
    assert replaced.source_tree_digest == sanitized.source_tree_digest
    assert ".git" in os.listdir(publisher_case["active"].workspace)
