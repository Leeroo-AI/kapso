from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from dataclasses import replace

import pytest

from kapso.cross_run.launch import workspace_frontier as workspace_frontier_module
from kapso.cross_run.launch.workspace_frontier import (
    RunWorkspaceFrontierError,
    copy_run_workspace_frontier,
    copy_run_workspace_source_tree,
    inspect_detached_run_workspace_source_tree,
    inspect_run_workspace_frontier,
    inspect_run_workspace_source_tree,
    plan_run_workspace_frontier_copy,
    plan_run_workspace_source_copy,
)
from test_launch_resolver import resolver_case
from test_run_state_publisher import publisher_case


def _open_source(case, descriptors):
    source_descriptor, _identity = case["active"]._open_execution_workspace(descriptors)
    expected_commit = case[
        "checkpoint"
    ].safety_state.derivative_frontier.evidence.branch_heads[
        case["settings"].workspace_git_branch
    ]
    frontier = inspect_run_workspace_frontier(
        source_descriptor,
        settings=case["settings"],
        expected_commit_sha=expected_commit,
    )
    return source_descriptor, frontier


def _open_destination(path, descriptors):
    path.mkdir(mode=0o700)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, descriptor)
    return descriptor


def _source_file(case):
    return next(
        path
        for path in case["active"].workspace.rglob("*")
        if path.is_file() and path.stat().st_size > 0 and ".git" not in path.parts
    )


def _normalized_mode_substitution(path):
    mode = stat.S_IMODE(path.stat().st_mode)
    substituted_mode = {
        0o600: 0o644,
        0o644: 0o600,
        0o700: 0o755,
        0o755: 0o700,
    }[mode]
    path.chmod(substituted_mode)


def _same_size_content_substitution(path):
    payload = path.read_bytes()
    path.write_bytes(bytes((payload[0] ^ 0xFF,)) + payload[1:])


def test_source_only_inspection_matches_frontier_and_observes_uncommitted_edits(
    publisher_case,
):
    with ExitStack() as descriptors:
        source_descriptor, frontier = _open_source(
            publisher_case,
            descriptors,
        )
        source = inspect_run_workspace_source_tree(
            source_descriptor,
            maximum_entries=publisher_case["settings"].run_workspace_entry_limit,
            maximum_bytes=publisher_case["settings"].run_workspace_size_bytes,
        )
        source_file = _source_file(publisher_case)
        _same_size_content_substitution(source_file)
        edited = inspect_run_workspace_source_tree(
            source_descriptor,
            maximum_entries=publisher_case["settings"].run_workspace_entry_limit,
            maximum_bytes=publisher_case["settings"].run_workspace_size_bytes,
        )

    assert source.workspace_identity == frontier.workspace_identity
    assert source.source_tree_digest == frontier.source_tree_digest
    assert source.source_entry_count == frontier.source_entry_count
    assert source.source_size_bytes == frontier.source_size_bytes
    assert edited.workspace_identity == source.workspace_identity
    assert edited.source_tree_digest != source.source_tree_digest
    assert edited.source_entry_count == source.source_entry_count
    assert edited.source_size_bytes == source.source_size_bytes


@pytest.mark.parametrize(
    ("maximum_entries", "maximum_bytes", "message"),
    (
        (1, 2**63, "entry limit"),
        (2**63, 1, "byte limit"),
    ),
)
def test_source_only_inspection_enforces_exact_bounds(
    publisher_case,
    maximum_entries,
    maximum_bytes,
    message,
):
    with ExitStack() as descriptors:
        source_descriptor, _frontier = _open_source(
            publisher_case,
            descriptors,
        )
        with pytest.raises(RunWorkspaceFrontierError, match=message):
            inspect_run_workspace_source_tree(
                source_descriptor,
                maximum_entries=maximum_entries,
                maximum_bytes=maximum_bytes,
            )


def test_source_copy_excludes_git_and_reproves_the_detached_tree(
    publisher_case,
    tmp_path,
):
    destination_path = tmp_path / "detached-source"
    settings = publisher_case["settings"]
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_source_copy(
            source_descriptor,
            expected=source_frontier,
            maximum_source_entries=settings.run_workspace_entry_limit,
            maximum_source_bytes=settings.run_workspace_size_bytes,
            maximum_git_entries=settings.run_workspace_git_entry_limit,
            maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
        )
        destination_descriptor = _open_destination(destination_path, descriptors)

        destination_source = copy_run_workspace_source_tree(
            source_descriptor,
            destination_descriptor,
            plan=plan,
            maximum_source_entries=settings.run_workspace_entry_limit,
            maximum_source_bytes=settings.run_workspace_size_bytes,
            maximum_git_entries=settings.run_workspace_git_entry_limit,
            maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
        )

    assert destination_source.workspace_identity != source_frontier.workspace_identity
    assert destination_source.source_tree_digest == source_frontier.source_tree_digest
    assert destination_source.source_entry_count == source_frontier.source_entry_count
    assert destination_source.source_size_bytes == source_frontier.source_size_bytes
    assert plan.physical_entry_count == source_frontier.source_entry_count + 1
    assert plan.regular_file_size_bytes == source_frontier.source_size_bytes
    assert plan.allocated_size_bytes(4096) >= plan.regular_file_size_bytes
    assert not (destination_path / ".git").exists()


def test_detached_source_inspection_rejects_git_metadata(
    publisher_case,
    tmp_path,
):
    destination_path = tmp_path / "detached-source"
    settings = publisher_case["settings"]
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_source_copy(
            source_descriptor,
            expected=source_frontier,
            maximum_source_entries=settings.run_workspace_entry_limit,
            maximum_source_bytes=settings.run_workspace_size_bytes,
            maximum_git_entries=settings.run_workspace_git_entry_limit,
            maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
        )
        destination_descriptor = _open_destination(destination_path, descriptors)
        copy_run_workspace_source_tree(
            source_descriptor,
            destination_descriptor,
            plan=plan,
            maximum_source_entries=settings.run_workspace_entry_limit,
            maximum_source_bytes=settings.run_workspace_size_bytes,
            maximum_git_entries=settings.run_workspace_git_entry_limit,
            maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
        )
        (destination_path / ".git").mkdir(mode=0o700)

        with pytest.raises(RunWorkspaceFrontierError, match="denied source path"):
            inspect_detached_run_workspace_source_tree(
                destination_descriptor,
                maximum_entries=settings.run_workspace_entry_limit,
                maximum_bytes=settings.run_workspace_size_bytes,
            )


def test_source_copy_rejects_trusted_source_change_after_plan(
    publisher_case,
    tmp_path,
):
    destination_path = tmp_path / "detached-source"
    settings = publisher_case["settings"]
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_source_copy(
            source_descriptor,
            expected=source_frontier,
            maximum_source_entries=settings.run_workspace_entry_limit,
            maximum_source_bytes=settings.run_workspace_size_bytes,
            maximum_git_entries=settings.run_workspace_git_entry_limit,
            maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
        )
        _same_size_content_substitution(_source_file(publisher_case))
        destination_descriptor = _open_destination(destination_path, descriptors)

        with pytest.raises(
            RunWorkspaceFrontierError,
            match="differ from the checkpointed commit tree",
        ):
            copy_run_workspace_source_tree(
                source_descriptor,
                destination_descriptor,
                plan=plan,
                maximum_source_entries=settings.run_workspace_entry_limit,
                maximum_source_bytes=settings.run_workspace_size_bytes,
                maximum_git_entries=settings.run_workspace_git_entry_limit,
                maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
            )

    assert tuple(destination_path.iterdir()) == ()


def test_workspace_copy_includes_git_and_reproves_both_frontiers(
    publisher_case,
    tmp_path,
):
    destination_path = tmp_path / "copied-workspace"
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_frontier_copy(
            source_descriptor,
            settings=publisher_case["settings"],
            expected=source_frontier,
        )
        destination_descriptor = _open_destination(
            destination_path,
            descriptors,
        )

        destination_frontier = copy_run_workspace_frontier(
            source_descriptor,
            destination_descriptor,
            settings=publisher_case["settings"],
            plan=plan,
        )

    assert destination_frontier.workspace_identity != (
        source_frontier.workspace_identity
    )
    assert destination_frontier.source_tree_digest == source_frontier.source_tree_digest
    assert destination_frontier.git_closure_digest == source_frontier.git_closure_digest
    assert destination_frontier.commit_sha == source_frontier.commit_sha
    assert plan.physical_entry_count > source_frontier.source_entry_count
    assert plan.regular_file_size_bytes > source_frontier.source_size_bytes
    assert plan.allocated_size_bytes(4096) >= plan.regular_file_size_bytes
    assert (destination_path / ".git" / "HEAD").is_file()
    copied_directories = tuple(
        path for path in destination_path.rglob("*") if path.is_dir()
    )
    assert copied_directories
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o700 for path in copied_directories
    )


def test_workspace_copy_rejects_source_change_after_plan(
    publisher_case,
    tmp_path,
):
    destination_path = tmp_path / "copied-workspace"
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_frontier_copy(
            source_descriptor,
            settings=publisher_case["settings"],
            expected=source_frontier,
        )
        source_file = _source_file(publisher_case)
        source_file.write_bytes(source_file.read_bytes() + b"substitution")
        destination_descriptor = _open_destination(
            destination_path,
            descriptors,
        )

        with pytest.raises(
            RunWorkspaceFrontierError,
            match="differ from the checkpointed commit tree",
        ):
            copy_run_workspace_frontier(
                source_descriptor,
                destination_descriptor,
                settings=publisher_case["settings"],
                plan=plan,
            )

    assert tuple(destination_path.iterdir()) == ()


def test_workspace_copy_rejects_nonempty_destination_before_mutation(
    publisher_case,
    tmp_path,
):
    destination_path = tmp_path / "copied-workspace"
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_frontier_copy(
            source_descriptor,
            settings=publisher_case["settings"],
            expected=source_frontier,
        )
        destination_descriptor = _open_destination(
            destination_path,
            descriptors,
        )
        unexpected = destination_path / "unexpected"
        unexpected.write_bytes(b"present")

        with pytest.raises(
            RunWorkspaceFrontierError,
            match="endpoints differ",
        ):
            copy_run_workspace_frontier(
                source_descriptor,
                destination_descriptor,
                settings=publisher_case["settings"],
                plan=plan,
            )

    assert unexpected.read_bytes() == b"present"


@pytest.mark.parametrize(
    "late_mutation",
    ("source_mode", "destination_mode", "destination_content"),
)
def test_workspace_copy_final_physical_scan_rejects_late_mutation(
    publisher_case,
    tmp_path,
    monkeypatch,
    late_mutation,
):
    destination_path = tmp_path / "copied-workspace"
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_frontier_copy(
            source_descriptor,
            settings=publisher_case["settings"],
            expected=source_frontier,
        )
        destination_descriptor = _open_destination(
            destination_path,
            descriptors,
        )
        source_file = _source_file(publisher_case)
        relative_file = source_file.relative_to(publisher_case["active"].workspace)
        source_identity = (
            os.fstat(source_descriptor).st_dev,
            os.fstat(source_descriptor).st_ino,
        )
        destination_identity = (
            os.fstat(destination_descriptor).st_dev,
            os.fstat(destination_descriptor).st_ino,
        )
        source_observation_count = 0
        original_inspector = workspace_frontier_module.inspect_run_workspace_frontier

        def inspect_then_mutate(
            workspace_descriptor,
            *,
            settings,
            expected_commit_sha,
        ):
            nonlocal source_observation_count
            observed = original_inspector(
                workspace_descriptor,
                settings=settings,
                expected_commit_sha=expected_commit_sha,
            )
            metadata = os.fstat(workspace_descriptor)
            identity = (metadata.st_dev, metadata.st_ino)
            if identity == source_identity:
                source_observation_count += 1
                if late_mutation == "source_mode" and source_observation_count == 2:
                    _normalized_mode_substitution(source_file)
            elif identity == destination_identity:
                destination_file = destination_path / relative_file
                if late_mutation == "destination_mode":
                    _normalized_mode_substitution(destination_file)
                elif late_mutation == "destination_content":
                    _same_size_content_substitution(destination_file)
            return observed

        monkeypatch.setattr(
            workspace_frontier_module,
            "inspect_run_workspace_frontier",
            inspect_then_mutate,
        )

        with pytest.raises(
            RunWorkspaceFrontierError,
            match="metadata changed|unsafe metadata|differs from its stable",
        ):
            copy_run_workspace_frontier(
                source_descriptor,
                destination_descriptor,
                settings=publisher_case["settings"],
                plan=plan,
            )


@pytest.mark.parametrize("block_size_bytes", (False, 0, 3, -4096))
def test_workspace_copy_plan_rejects_invalid_allocation_units(
    publisher_case,
    block_size_bytes,
):
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_frontier_copy(
            source_descriptor,
            settings=publisher_case["settings"],
            expected=source_frontier,
        )

        with pytest.raises(
            RunWorkspaceFrontierError,
            match="allocation block size is invalid",
        ):
            plan.allocated_size_bytes(block_size_bytes)


def test_workspace_copy_plan_summary_cannot_be_reminted(
    publisher_case,
):
    with ExitStack() as descriptors:
        source_descriptor, source_frontier = _open_source(
            publisher_case,
            descriptors,
        )
        plan = plan_run_workspace_frontier_copy(
            source_descriptor,
            settings=publisher_case["settings"],
            expected=source_frontier,
        )

        invalid_changes = (
            {
                "regular_file_sizes": (0,) * plan.regular_file_count,
                "regular_file_size_bytes": 0,
            },
            {
                "directory_count": plan.directory_count + 1,
                "physical_entry_count": plan.physical_entry_count + 1,
            },
            {
                "regular_file_count": plan.regular_file_count + 1,
                "physical_entry_count": plan.physical_entry_count + 1,
                "regular_file_sizes": (*plan.regular_file_sizes, 0),
            },
        )
        for changes in invalid_changes:
            with pytest.raises(
                RunWorkspaceFrontierError,
                match="incomplete or unbounded",
            ):
                replace(plan, **changes)
