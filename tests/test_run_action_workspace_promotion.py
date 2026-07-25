from __future__ import annotations

import os
import subprocess
from contextlib import ExitStack

import pytest

import kapso.cross_run.launch.run_action_workspace_promotion as promotion_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_workspace_promotion import (
    RunActionWorkspacePromotionError,
    RunActionWorkspacePromoter,
)
from kapso.cross_run.launch.workspace_frontier import (
    copy_run_workspace_frontier,
    inspect_run_workspace_frontier,
    plan_run_workspace_frontier_copy,
    RunWorkspaceFrontierError,
)
from test_launch_resolver import resolver_case
from test_run_state_publisher import publisher_case


def _run_git(workspace, *arguments):
    process = subprocess.Popen(
        ["git", "-C", str(workspace), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0, stderr
    return stdout


def _copy_candidate(case, path):
    path.mkdir(mode=0o700)
    with ExitStack() as descriptors:
        source_descriptor, _identity = case["active"]._open_execution_workspace(
            descriptors
        )
        predecessor = inspect_run_workspace_frontier(
            source_descriptor,
            settings=case["settings"],
            expected_commit_sha=case[
                "checkpoint"
            ].safety_state.derivative_frontier.evidence.branch_heads[
                case["settings"].workspace_git_branch
            ],
        )
        plan = plan_run_workspace_frontier_copy(
            source_descriptor,
            settings=case["settings"],
            expected=predecessor,
        )
        candidate_descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, candidate_descriptor)
        copy_run_workspace_frontier(
            source_descriptor,
            candidate_descriptor,
            settings=case["settings"],
            plan=plan,
        )
    return predecessor


def _commit_candidate(path, filename, payload):
    (path / filename).write_text(payload, encoding="utf-8")
    _run_git(path, "add", "--", filename)
    _run_git(
        path,
        "-c",
        "user.name=Kapso Test",
        "-c",
        "user.email=kapso-test@example.invalid",
        "commit",
        "-m",
        f"Edit {filename}",
    )


def _stage(case, candidate_path, predecessor):
    promoter = RunActionWorkspacePromoter(
        active_workspace=case["active"],
        settings=case["settings"],
    )
    with ExitStack() as descriptors:
        descriptor = os.open(
            candidate_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, descriptor)
        promotion = promoter.stage(
            result_receipt_id=content_id(
                "run-action-result-receipt",
                {"fixture": "result"},
            ),
            prepared_workspace_proof_id=content_id(
                "run-action-prepared-workspace-proof",
                {"fixture": "workspace"},
            ),
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            candidate_descriptor=descriptor,
        )
    return promoter, promotion


def _inspect_public(case, expected_commit):
    with ExitStack() as descriptors:
        descriptor, _identity = case["active"]._open_execution_workspace(descriptors)
        return inspect_run_workspace_frontier(
            descriptor,
            settings=case["settings"],
            expected_commit_sha=expected_commit,
        )


def _reopen_case(case):
    active = case["active"]
    builder = active._closure.verifier
    run_root = active.run_root
    active.close()
    case["active"] = builder.reopen(run_root)
    return RunActionWorkspacePromoter(
        active_workspace=case["active"],
        settings=case["settings"],
    )


def _promote(case, promoter, **arguments):
    with ExitStack() as descriptors:
        lock_descriptor = promotion_module._lock_workspace(
            case["active"],
            descriptors,
        )
        return promoter._promote_decided(
            **arguments,
            workspace_lock_descriptor=lock_descriptor,
            _authority=(promotion_module._RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY),
        )


def _cleanup_accepted(case, promoter, **arguments):
    with ExitStack() as descriptors:
        lock_descriptor = promotion_module._lock_workspace(
            case["active"],
            descriptors,
        )
        return promoter._cleanup_accepted(
            **arguments,
            workspace_lock_descriptor=lock_descriptor,
            _authority=(promotion_module._RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY),
        )


def test_stage_keeps_public_predecessor_then_exchange_is_idempotent(
    publisher_case,
    tmp_path,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "promotion.txt", "candidate")

    promoter, promotion = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )

    with pytest.raises(
        RunActionWorkspacePromotionError,
        match="recovery inputs are invalid",
    ):
        _promote(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=content_id(
                "run-action-result-receipt",
                {"fixture": "foreign"},
            ),
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )
    assert (
        _inspect_public(
            publisher_case,
            predecessor.commit_sha,
        )
        == predecessor
    )
    promoted = _promote(
        publisher_case,
        promoter,
        predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
        promotion=promotion,
        result_receipt_id=promotion.result_receipt_id,
        prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
    )
    assert promoted == promotion.candidate_workspace.to_identity()
    assert _inspect_public(publisher_case, promoted.commit_sha) == promoted
    assert (
        _promote(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )
        == promoted
    )
    assert (
        _cleanup_accepted(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )
        == promoted
    )
    assert (
        _cleanup_accepted(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )
        == promoted
    )
    staging = (
        publisher_case["active"].run_root
        / publisher_case[
            "active"
        ].bootstrap_pin.installation_receipt.layout.run_action_workspace_staging_relative_path
    )
    assert tuple(staging.iterdir()) == ()


def test_stage_reopens_complete_candidate_left_before_decision(
    publisher_case,
    tmp_path,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")

    promoter, promotion = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )
    reopened_promoter, reopened = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )

    assert reopened == promotion
    promoted = _promote(
        publisher_case,
        reopened_promoter,
        predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
        promotion=reopened,
        result_receipt_id=reopened.result_receipt_id,
        prepared_workspace_proof_id=reopened.prepared_workspace_proof_id,
    )
    assert promoted == reopened.candidate_workspace.to_identity()


def test_stage_removes_interrupted_temporary_then_retries(
    publisher_case,
    tmp_path,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")
    staging = (
        publisher_case["active"].run_root
        / publisher_case[
            "active"
        ].bootstrap_pin.installation_receipt.layout.run_action_workspace_staging_relative_path
    )
    interrupted = staging / ".workspace-0123456789abcdef0123456789abcdef.tmp"
    interrupted.mkdir(mode=0o700)
    (interrupted / "partial").write_text("partial", encoding="utf-8")

    _promoter, promotion = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )

    assert tuple(path.name for path in staging.iterdir()) == ("workspace",)
    assert promotion.candidate_workspace.commit_sha != predecessor.commit_sha


def test_stage_capacity_rejection_leaves_no_candidate(
    publisher_case,
    tmp_path,
    monkeypatch,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")
    promoter = RunActionWorkspacePromoter(
        active_workspace=publisher_case["active"],
        settings=publisher_case["settings"],
    )
    descriptor = os.open(
        candidate_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    filesystem = os.fstatvfs(descriptor)
    low_capacity = os.statvfs_result(
        (
            filesystem.f_bsize,
            filesystem.f_frsize,
            filesystem.f_blocks,
            filesystem.f_bfree,
            0,
            filesystem.f_files,
            filesystem.f_ffree,
            0,
            filesystem.f_flag,
            filesystem.f_namemax,
        )
    )
    monkeypatch.setattr(
        promotion_module.os,
        "fstatvfs",
        lambda _descriptor: low_capacity,
    )

    with pytest.raises(
        RunActionWorkspacePromotionError,
        match="lacks byte or inode capacity",
    ):
        promoter.stage(
            result_receipt_id=content_id(
                "run-action-result-receipt",
                {"fixture": "capacity"},
            ),
            prepared_workspace_proof_id=content_id(
                "run-action-prepared-workspace-proof",
                {"fixture": "capacity"},
            ),
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            candidate_descriptor=descriptor,
        )
    os.close(descriptor)
    staging = (
        publisher_case["active"].run_root
        / publisher_case[
            "active"
        ].bootstrap_pin.installation_receipt.layout.run_action_workspace_staging_relative_path
    )
    assert tuple(staging.iterdir()) == ()


def test_staging_capacity_accepts_exact_candidate_allocation_boundary(
    publisher_case,
    tmp_path,
    monkeypatch,
):
    candidate_path = tmp_path / "candidate"
    _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")
    with ExitStack() as descriptors:
        descriptor = os.open(
            candidate_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, descriptor)
        candidate = inspect_run_workspace_frontier(
            descriptor,
            settings=publisher_case["settings"],
            expected_commit_sha=None,
        )
        plan = plan_run_workspace_frontier_copy(
            descriptor,
            settings=publisher_case["settings"],
            expected=candidate,
        )
        filesystem = os.fstatvfs(descriptor)
        block_count = plan.allocated_size_bytes(filesystem.f_frsize) // (
            filesystem.f_frsize
        )
        exact_capacity = os.statvfs_result(
            (
                filesystem.f_bsize,
                filesystem.f_frsize,
                filesystem.f_blocks,
                block_count,
                block_count,
                filesystem.f_files,
                plan.physical_entry_count,
                plan.physical_entry_count,
                filesystem.f_flag,
                filesystem.f_namemax,
            )
        )
        monkeypatch.setattr(
            promotion_module.os,
            "fstatvfs",
            lambda _descriptor: exact_capacity,
        )

        promotion_module._require_staging_capacity(
            descriptor,
            plan,
            publisher_case["settings"],
        )

        insufficient_capacity = os.statvfs_result(
            (
                exact_capacity.f_bsize,
                exact_capacity.f_frsize,
                exact_capacity.f_blocks,
                block_count - 1,
                block_count - 1,
                exact_capacity.f_files,
                exact_capacity.f_ffree,
                exact_capacity.f_favail,
                exact_capacity.f_flag,
                exact_capacity.f_namemax,
            )
        )
        monkeypatch.setattr(
            promotion_module.os,
            "fstatvfs",
            lambda _descriptor: insufficient_capacity,
        )
        with pytest.raises(
            RunActionWorkspacePromotionError,
            match="lacks byte or inode capacity",
        ):
            promotion_module._require_staging_capacity(
                descriptor,
                plan,
                publisher_case["settings"],
            )


def test_stage_rejects_unchanged_or_nondirect_candidate(
    publisher_case,
    tmp_path,
):
    unchanged_path = tmp_path / "unchanged"
    predecessor = _copy_candidate(publisher_case, unchanged_path)
    promoter = RunActionWorkspacePromoter(
        active_workspace=publisher_case["active"],
        settings=publisher_case["settings"],
    )
    unchanged_descriptor = os.open(
        unchanged_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with pytest.raises(
        RunActionWorkspacePromotionError,
        match="not one direct source successor",
    ):
        promoter.stage(
            result_receipt_id=content_id(
                "run-action-result-receipt",
                {"fixture": "unchanged"},
            ),
            prepared_workspace_proof_id=content_id(
                "run-action-prepared-workspace-proof",
                {"fixture": "unchanged"},
            ),
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            candidate_descriptor=unchanged_descriptor,
        )
    os.close(unchanged_descriptor)

    nondirect_path = tmp_path / "nondirect"
    _copy_candidate(publisher_case, nondirect_path)
    _commit_candidate(nondirect_path, "first.txt", "first")
    _commit_candidate(nondirect_path, "second.txt", "second")
    nondirect_descriptor = os.open(
        nondirect_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with pytest.raises(
        RunActionWorkspacePromotionError,
        match="not one direct source successor",
    ):
        promoter.stage(
            result_receipt_id=content_id(
                "run-action-result-receipt",
                {"fixture": "nondirect"},
            ),
            prepared_workspace_proof_id=content_id(
                "run-action-prepared-workspace-proof",
                {"fixture": "nondirect"},
            ),
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            candidate_descriptor=nondirect_descriptor,
        )
    os.close(nondirect_descriptor)


def test_promote_rejects_third_public_state_without_exchange(
    publisher_case,
    tmp_path,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")
    promoter, promotion = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )
    public_file = next(
        path
        for path in publisher_case["active"].workspace.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    public_file.write_bytes(public_file.read_bytes() + b"tampered")

    with pytest.raises(
        RunWorkspaceFrontierError,
        match="checkpointed commit tree",
    ):
        _promote(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )


def test_exchange_crash_reopens_as_exact_swapped_state(
    publisher_case,
    tmp_path,
    monkeypatch,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")
    promoter, promotion = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )
    original = promotion_module._rename_at

    def exchange_then_die(
        source_parent_descriptor,
        source_name,
        destination_parent_descriptor,
        destination_name,
        flags,
    ):
        original(
            source_parent_descriptor,
            source_name,
            destination_parent_descriptor,
            destination_name,
            flags,
        )
        if flags == promotion_module._RENAME_EXCHANGE:
            raise RuntimeError("injected death after exchange")

    monkeypatch.setattr(
        promotion_module,
        "_rename_at",
        exchange_then_die,
    )
    with pytest.raises(RuntimeError, match="after exchange"):
        _promote(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )
    monkeypatch.setattr(promotion_module, "_rename_at", original)

    promoter = _reopen_case(publisher_case)
    promoted = _promote(
        publisher_case,
        promoter,
        predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
        promotion=promotion,
        result_receipt_id=promotion.result_receipt_id,
        prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
    )
    assert _inspect_public(publisher_case, promoted.commit_sha) == promoted


def test_exchange_crash_after_first_parent_fsync_reopens_swapped(
    publisher_case,
    tmp_path,
    monkeypatch,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")
    promoter, promotion = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )
    original = promotion_module.os.fsync
    fsync_calls = 0

    def first_parent_fsync_then_die(descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        original(descriptor)
        if fsync_calls == 1:
            raise RuntimeError("injected death after first parent fsync")

    monkeypatch.setattr(
        promotion_module.os,
        "fsync",
        first_parent_fsync_then_die,
    )
    with pytest.raises(RuntimeError, match="first parent fsync"):
        _promote(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )
    monkeypatch.setattr(promotion_module.os, "fsync", original)

    promoter = _reopen_case(publisher_case)
    promoted = _promote(
        publisher_case,
        promoter,
        predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
        promotion=promotion,
        result_receipt_id=promotion.result_receipt_id,
        prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
    )
    assert _inspect_public(publisher_case, promoted.commit_sha) == promoted


def test_accepted_cleanup_crash_reopens_and_finishes(
    publisher_case,
    tmp_path,
    monkeypatch,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")
    promoter, promotion = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )
    _promote(
        publisher_case,
        promoter,
        predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
        promotion=promotion,
        result_receipt_id=promotion.result_receipt_id,
        prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
    )
    original = promotion_module.os.unlink
    removed_entries = 0

    def unlink_then_die(name, *, dir_fd):
        nonlocal removed_entries
        original(name, dir_fd=dir_fd)
        removed_entries += 1
        if removed_entries == 1:
            raise RuntimeError("injected death during accepted cleanup")

    monkeypatch.setattr(
        promotion_module.os,
        "unlink",
        unlink_then_die,
    )
    with pytest.raises(RuntimeError, match="during accepted cleanup"):
        _cleanup_accepted(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )
    monkeypatch.setattr(promotion_module.os, "unlink", original)

    promoter = _reopen_case(publisher_case)
    promoted = _cleanup_accepted(
        publisher_case,
        promoter,
        predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
        promotion=promotion,
        result_receipt_id=promotion.result_receipt_id,
        prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
    )
    assert _inspect_public(publisher_case, promoted.commit_sha) == promoted


def test_cleanup_rejects_same_device_bind_mount_boundary_before_removal(
    publisher_case,
    tmp_path,
    monkeypatch,
):
    candidate_path = tmp_path / "candidate"
    predecessor = _copy_candidate(publisher_case, candidate_path)
    _commit_candidate(candidate_path, "candidate.txt", "candidate")
    promoter, promotion = _stage(
        publisher_case,
        candidate_path,
        predecessor,
    )
    _promote(
        publisher_case,
        promoter,
        predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
        promotion=promotion,
        result_receipt_id=promotion.result_receipt_id,
        prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
    )
    original_mount_id = promotion_module.read_run_action_descriptor_mount_id

    def observe_same_device_bind_mount(descriptor):
        mount_id = original_mount_id(descriptor)
        descriptor_path = os.readlink(f"/proc/self/fd/{descriptor}")
        if descriptor_path.endswith("/workspace/.git"):
            return mount_id + 1
        return mount_id

    monkeypatch.setattr(
        promotion_module,
        "read_run_action_descriptor_mount_id",
        observe_same_device_bind_mount,
    )
    with pytest.raises(
        RunActionWorkspacePromotionError,
        match="mount boundary",
    ):
        _cleanup_accepted(
            publisher_case,
            promoter,
            predecessor=RunActionWorkspaceBinding.from_identity(predecessor),
            promotion=promotion,
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=promotion.prepared_workspace_proof_id,
        )

    assert (
        _inspect_public(
            publisher_case,
            promotion.candidate_workspace.commit_sha,
        )
        == promotion.candidate_workspace.to_identity()
    )
    staging = (
        publisher_case["active"].run_root
        / publisher_case[
            "active"
        ].bootstrap_pin.installation_receipt.layout.run_action_workspace_staging_relative_path
        / "workspace"
    )
    assert (staging / ".git").is_dir()
