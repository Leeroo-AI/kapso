"""Descriptor and ownership seams for blocked-workload runtime proof."""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from threading import Thread, get_ident
from types import SimpleNamespace

import pytest

import kapso.cross_run.launch.run_action_resolved_workload as workload_module
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_resolved_workload import (
    RunActionBlockedWorkloadLease,
    RunActionResolvedWorkloadError,
)
from kapso.cross_run.launch.run_action_recovery import (
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    RunActionProcessStatObservation,
    read_run_action_descriptor_mount_id,
)
from test_run_action_barrier_contracts import _resolved_graph
from test_run_action_supervisor_contracts import _remint_contract


def _ownership_only_lease() -> RunActionBlockedWorkloadLease:
    lease = object.__new__(RunActionBlockedWorkloadLease)
    lease._owner_process_id = os.getpid()
    lease._owner_thread_id = get_ident()
    lease._closed = False
    _issue_test_lease(lease)
    return lease


def _issue_test_lease(lease: RunActionBlockedWorkloadLease) -> None:
    with workload_module._BLOCKED_WORKLOAD_LEASE_LOCK:
        workload_module._ISSUED_BLOCKED_WORKLOAD_LEASES[id(lease)] = lease


def test_blocked_workload_lease_rejects_an_unissued_lookalike():
    lease = object.__new__(RunActionBlockedWorkloadLease)

    with pytest.raises(
        RunActionResolvedWorkloadError,
        match="unissued, closed, or foreign",
    ):
        lease.require_current()


def test_blocked_workload_lease_rejects_forked_and_cross_thread_use(monkeypatch):
    lease = _ownership_only_lease()
    owner_process_id = os.getpid()

    monkeypatch.setattr(workload_module.os, "getpid", lambda: owner_process_id + 1)
    with pytest.raises(
        RunActionResolvedWorkloadError,
        match="closed, forked, or on another thread",
    ):
        lease.require_current()

    monkeypatch.setattr(workload_module.os, "getpid", lambda: owner_process_id)
    failures = []

    def use_from_foreign_thread():
        with pytest.raises(
            RunActionResolvedWorkloadError,
            match="closed, forked, or on another thread",
        ):
            lease.require_current()
        failures.append(True)

    thread = Thread(target=use_from_foreign_thread)
    thread.start()
    thread.join()
    assert failures == [True]


def test_blocked_workload_lease_exposes_no_raw_descriptor():
    public_names = {
        name for name in dir(RunActionBlockedWorkloadLease) if not name.startswith("_")
    }

    assert public_names == {
        "activation_event",
        "close",
        "require_current",
        "resolved_workload_observation",
    }


def test_nofollow_container_path_rejects_intermediate_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "mount").symlink_to(outside, target_is_directory=True)
    root_descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as opened:
        opened.callback(os.close, root_descriptor)
        with ExitStack() as descriptors:
            with pytest.raises(OSError):
                workload_module._open_nofollow_container_path(
                    descriptors,
                    root_descriptor,
                    "/mount/child",
                    directory=True,
                )


def test_retained_root_detects_path_splice_while_original_fd_survives(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    mounted = root / "mounted"
    mounted.mkdir()
    root_descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    mounted_descriptor = os.open(
        mounted,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as opened:
        opened.callback(os.close, mounted_descriptor)
        opened.callback(os.close, root_descriptor)
        metadata = os.fstat(mounted_descriptor)
        retained = workload_module._RetainedResolvedRoot(
            destination="/mounted",
            descriptor=mounted_descriptor,
            metadata=workload_module._stable_metadata(metadata),
            mount_id=read_run_action_descriptor_mount_id(mounted_descriptor),
        )
        mounted.rename(root / "old-mounted")
        (root / "mounted").mkdir()

        with pytest.raises(
            RunActionResolvedWorkloadError,
            match="was replaced or spliced",
        ):
            workload_module._require_retained_root_current(
                root_descriptor,
                retained,
            )


def test_process_generation_admits_scheduler_state_transition():
    sleeping = RunActionProcessStatObservation(
        process_id=91,
        state="S",
        parent_process_id=1,
        start_time_ticks=10,
    )
    running = RunActionProcessStatObservation(
        process_id=91,
        state="R",
        parent_process_id=1,
        start_time_ticks=10,
    )

    assert workload_module._same_process_generation(running, sleeping)
    assert not workload_module._same_process_generation(
        RunActionProcessStatObservation(
            process_id=91,
            state="S",
            parent_process_id=1,
            start_time_ticks=11,
        ),
        sleeping,
    )


def test_running_occurrence_ignores_only_nonauthoritative_raw_digest():
    running = _resolved_graph().running_container_observation
    new_raw_snapshot = _remint_contract(
        running,
        complete_inspection_digest="sha256:" + "f" * 64,
    )
    restarted = _remint_contract(
        running,
        init_process_id=running.init_process_id + 1,
    )

    assert workload_module._same_running_container_occurrence(
        new_raw_snapshot,
        running,
    )
    assert not workload_module._same_running_container_occurrence(
        restarted,
        running,
    )


def test_committed_running_authority_rejects_state_and_token_splice():
    committed_running = _resolved_graph().running_container_observation
    current_running = _remint_contract(
        committed_running,
        complete_inspection_digest="sha256:" + "f" * 64,
    )
    running = RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
        observation_token=committed_running.complete_inspection_digest,
    )

    workload_module._require_committed_running_authority(
        running,
        committed_running,
        current_running,
    )
    with pytest.raises(
        RunActionResolvedWorkloadError,
        match="committed running observation",
    ):
        workload_module._require_committed_running_authority(
            RunActionCommittedSpawnObservation(
                state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
                observation_token=committed_running.complete_inspection_digest,
            ),
            committed_running,
            current_running,
        )
    with pytest.raises(
        RunActionResolvedWorkloadError,
        match="committed running observation",
    ):
        workload_module._require_committed_running_authority(
            RunActionCommittedSpawnObservation(
                state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
                observation_token="sha256:" + "0" * 64,
            ),
            committed_running,
            current_running,
        )


def test_require_current_reverse_check_detects_splice_during_logical_read(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root"
    root.mkdir()
    mounted = root / "mounted"
    mounted.mkdir()
    root_descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    mounted_descriptor = os.open(
        mounted,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as opened:
        opened.callback(os.close, mounted_descriptor)
        opened.callback(os.close, root_descriptor)
        metadata = os.fstat(mounted_descriptor)
        retained_root = workload_module._RetainedResolvedRoot(
            destination="/mounted",
            descriptor=mounted_descriptor,
            metadata=workload_module._stable_metadata(metadata),
            mount_id=read_run_action_descriptor_mount_id(mounted_descriptor),
        )
        init_process = SimpleNamespace(
            process_descriptor=31,
            root_descriptor=root_descriptor,
            stat_observation=SimpleNamespace(process_id=91),
        )
        wrapper_process = SimpleNamespace(
            stat_observation=SimpleNamespace(process_id=92),
        )
        running_container = SimpleNamespace(container_id="a" * 64)
        lease = object.__new__(RunActionBlockedWorkloadLease)
        lease._owner_process_id = os.getpid()
        lease._owner_thread_id = get_ident()
        lease._closed = False
        lease._proc_root_descriptor = 30
        lease._host_boot_id = "boot"
        lease._control_lease = SimpleNamespace(
            require_current=lambda: None,
            topology=RunActionControlDirectoryTopology.EMPTY,
        )
        lease._resolved_workload_observation = SimpleNamespace(
            running_container_observation=running_container,
        )
        lease._resource_manager = object()
        lease._preparation_allocation = object()
        lease._command = object()
        lease._volume_observation = object()
        lease._helper_evidence = object()
        lease._init_source_evidence = object()
        lease._docker_settings = object()
        lease._launch_settings = object()
        lease._process_snapshot_size_limit_bytes = 100
        lease._init_process = init_process
        lease._wrapper_process = wrapper_process
        lease._mount_info_snapshot = object()
        lease._retained_roots = (retained_root,)
        _issue_test_lease(lease)

        monkeypatch.setattr(
            workload_module,
            "read_run_action_host_boot_id",
            lambda _descriptor: "boot",
        )
        monkeypatch.setattr(
            workload_module,
            "_observe_running_container",
            lambda *_arguments: running_container,
        )
        monkeypatch.setattr(
            workload_module,
            "_same_running_container_occurrence",
            lambda *_arguments: True,
        )
        monkeypatch.setattr(
            workload_module,
            "_require_retained_process_current",
            lambda *_arguments: None,
        )
        monkeypatch.setattr(
            workload_module,
            "read_run_action_process_direct_child_from_descriptor",
            lambda *_arguments: 92,
        )
        monkeypatch.setattr(
            workload_module,
            "_read_mount_info_snapshot",
            lambda *_arguments: lease._mount_info_snapshot,
        )

        def splice_during_logical_read(*_arguments):
            mounted.rename(root / "old-mounted")
            (root / "mounted").mkdir()

        monkeypatch.setattr(
            workload_module,
            "_require_logical_mounts_current",
            splice_during_logical_read,
        )

        with pytest.raises(
            RunActionResolvedWorkloadError,
            match="was replaced or spliced",
        ):
            lease.require_current()


def test_directory_entries_are_exact_and_sorted(tmp_path):
    Path(tmp_path / "z").touch()
    Path(tmp_path / "a").touch()
    descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as opened:
        opened.callback(os.close, descriptor)
        assert workload_module._exact_directory_entries(descriptor) == ("a", "z")
