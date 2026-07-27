from __future__ import annotations

import os
import stat
from contextlib import ExitStack

import pytest

from kapso.cross_run.launch.run_action_coding_agent_scratch import (
    PROVIDER_SHARED_DIRECTORY_MODE,
    PROVIDER_SHARED_EXECUTABLE_MODE,
    PROVIDER_SHARED_FILE_MODE,
    RunActionCodingAgentScratchError,
    inspect_coding_agent_scratch_source_tree,
    share_coding_agent_scratch_source_tree,
)
from kapso.cross_run.launch.workspace_frontier import (
    copy_run_workspace_source_tree,
    plan_run_workspace_source_copy,
)
from test_launch_resolver import resolver_case
from test_run_action_workspace_copy import _open_destination, _open_source
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
