"""Descriptor-safe CAS persistence for exact cross-run checkpoints."""

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import kapso.cross_run.launch.checkpoint_store as checkpoint_store_module
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointHead,
    RunCheckpointStatus,
    RunCheckpointStop,
)
from kapso.cross_run.launch.checkpoint_store import (
    DurableRunCheckpoint,
    RunCheckpointStore,
    RunCheckpointStoreError,
)
from kapso.cross_run.launch.workspace import (
    ActiveLaunchWorkspace,
    LaunchWorkspaceError,
    StarterWorkspaceBuilder,
)
from test_launch_checkpoint_contracts import (
    _initial_checkpoint,
    _successor_safety,
)
from test_launch_resolver import resolver_case


def _active_launch(resolver_case, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    prepared = StarterWorkspaceBuilder(resolver_case["resolver"]._settings).build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="run-checkpoint-store",
        campaign_id="campaign-checkpoint-store",
    )
    return (
        prepared.require_builder_authority(),
        resolver_case["resolver"]._settings.launch,
    )


def _successor(initial, *, cost, stop=RunCheckpointStop.COST_BUDGET):
    pin = initial.safety_state.bootstrap_pin
    return RunCheckpoint.build(
        predecessor=initial,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=stop,
        completed_iterations=0,
        cumulative_cost=cost,
        elapsed_seconds=cost,
        cost_by_component={"implementation": cost},
        feedback_source=None,
        current_feedback=None,
        termination_decision=None,
        strategy_state=initial.strategy_state,
        safety_state=_successor_safety(pin, initial.safety_state),
    )


def _persist_initial(store, active):
    initial = _initial_checkpoint(active.bootstrap_pin)
    permit = store.issue_write_permit(None, initial)
    durable = store.compare_and_swap(permit, initial)
    return initial, durable


def test_checkpoint_store_requires_active_launch_authority(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)

    with pytest.raises(RunCheckpointStoreError, match="active launch"):
        RunCheckpointStore(object(), settings)
    cloned = replace(active)
    with pytest.raises(LaunchWorkspaceError, match="control authority"):
        RunCheckpointStore(cloned, settings)
    with pytest.raises(RunCheckpointStoreError, match="settings differ"):
        RunCheckpointStore(
            active,
            replace(
                settings,
                run_checkpoint_size_bytes=settings.run_checkpoint_size_bytes + 1,
            ),
        )
    assert type(active) is ActiveLaunchWorkspace


def test_checkpoint_store_publishes_genesis_successor_and_idempotent_retry(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, durable_initial = _persist_initial(store, active)

    assert type(durable_initial) is DurableRunCheckpoint
    assert durable_initial.require_current(store) == initial
    with pytest.raises(RunCheckpointStoreError, match="cloned"):
        replace(durable_initial).require_current(store)
    checkpoint_path = active.run_root / settings.run_checkpoint_path
    lock_path = active.run_root / settings.run_checkpoint_lock_path
    staging_path = active.run_root / settings.run_checkpoint_staging_path
    assert stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(staging_path.stat().st_mode) == 0o700

    successor = _successor(initial, cost=1.0)
    permit = store.issue_write_permit(initial.run_checkpoint_id, successor)
    durable_successor = store.compare_and_swap(permit, successor)
    assert durable_successor.require_current(store) == successor

    retry = store.issue_write_permit(initial.run_checkpoint_id, successor)
    assert store.compare_and_swap(retry, successor).checkpoint == successor


def test_checkpoint_store_rejects_consumed_stale_and_foreign_permits(
    resolver_case,
    tmp_path,
):
    first_active, settings = _active_launch(resolver_case, tmp_path / "first")
    first_store = RunCheckpointStore(first_active, settings)
    initial, _ = _persist_initial(first_store, first_active)
    successor = _successor(initial, cost=1.0)
    permit = first_store.issue_write_permit(initial.run_checkpoint_id, successor)
    with pytest.raises(RunCheckpointStoreError, match="cloned"):
        first_store.compare_and_swap(replace(permit), successor)
    first_store.compare_and_swap(permit, successor)

    with pytest.raises(RunCheckpointStoreError, match="consumed"):
        first_store.compare_and_swap(permit, successor)
    with pytest.raises(RunCheckpointStoreError, match="stale"):
        first_store.issue_write_permit(
            initial.run_checkpoint_id,
            _successor(initial, cost=2.0),
        )

    second_active, second_settings = _active_launch(
        resolver_case,
        tmp_path / "second",
    )
    second_store = RunCheckpointStore(second_active, second_settings)
    second_initial = _initial_checkpoint(second_active.bootstrap_pin)
    second_permit = second_store.issue_write_permit(None, second_initial)
    with pytest.raises(RunCheckpointStoreError, match="another candidate|foreign"):
        first_store.compare_and_swap(second_permit, successor)


def test_checkpoint_store_allows_only_one_concurrent_fork(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, _ = _persist_initial(store, active)
    first = _successor(initial, cost=1.0)
    second = _successor(
        initial,
        cost=2.0,
        stop=RunCheckpointStop.TIME_BUDGET,
    )
    first_permit = store.issue_write_permit(initial.run_checkpoint_id, first)
    second_permit = store.issue_write_permit(initial.run_checkpoint_id, second)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(store.compare_and_swap, first_permit, first),
            pool.submit(store.compare_and_swap, second_permit, second),
        )
    successes = tuple(
        future.result() for future in futures if future.exception() is None
    )
    failures = tuple(
        future.exception() for future in futures if future.exception() is not None
    )

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RunCheckpointStoreError)
    assert store.load().checkpoint == successes[0].checkpoint


@pytest.mark.parametrize(
    ("target", "mutation"),
    (
        ("checkpoint", "mode"),
        ("checkpoint", "hardlink"),
        ("lock", "fifo"),
        ("staging", "unexpected"),
        ("staging", "symlink"),
    ),
)
def test_checkpoint_store_rejects_unsafe_control_entries(
    resolver_case,
    tmp_path,
    target,
    mutation,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    _persist_initial(store, active)
    checkpoint_path = active.run_root / settings.run_checkpoint_path
    lock_path = active.run_root / settings.run_checkpoint_lock_path
    staging_path = active.run_root / settings.run_checkpoint_staging_path

    if target == "checkpoint" and mutation == "mode":
        checkpoint_path.chmod(0o600)
    elif target == "checkpoint":
        os.link(checkpoint_path, tmp_path / "checkpoint-hardlink")
    elif target == "lock":
        lock_path.unlink()
        os.mkfifo(lock_path, mode=0o600)
    elif mutation == "unexpected":
        (staging_path / "unexpected").write_bytes(b"unsafe")
        (staging_path / "unexpected").chmod(0o600)
    else:
        (
            staging_path / ("checkpoint-" + "a" * 64 + "-" + "b" * 32 + ".tmp")
        ).symlink_to(checkpoint_path)

    with pytest.raises((LaunchWorkspaceError, RunCheckpointStoreError, OSError)):
        store.load()


def test_checkpoint_store_rejects_noncanonical_and_oversize_bytes(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    _persist_initial(store, active)
    checkpoint_path = active.run_root / settings.run_checkpoint_path
    checkpoint_path.chmod(0o600)
    checkpoint_path.write_bytes(b'{"not":"a checkpoint"}')
    checkpoint_path.chmod(0o400)

    with pytest.raises((RunCheckpointStoreError, ValueError)):
        store.load()

    checkpoint_path.chmod(0o600)
    checkpoint_path.write_bytes(b"x" * (settings.run_checkpoint_size_bytes + 1))
    checkpoint_path.chmod(0o400)
    with pytest.raises((LaunchWorkspaceError, RunCheckpointStoreError)):
        store.load()


def test_checkpoint_store_rejects_rebound_parent_and_staging(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    _persist_initial(store, active)
    staging_path = active.run_root / settings.run_checkpoint_staging_path
    moved = staging_path.with_name("moved-staging")
    staging_path.rename(moved)
    staging_path.mkdir(mode=0o700)

    with pytest.raises((LaunchWorkspaceError, RunCheckpointStoreError)):
        store.load()

    moved.rmdir()


def test_checkpoint_store_cleans_safe_staging_after_pre_replace_failure(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, _ = _persist_initial(store, active)
    successor = _successor(initial, cost=1.0)
    permit = store.issue_write_permit(initial.run_checkpoint_id, successor)

    def fail_replace(*args, **kwargs):
        raise OSError("injected pre-replace failure")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_store_module.os, "replace", fail_replace)
        with pytest.raises(OSError, match="injected pre-replace"):
            store.compare_and_swap(permit, successor)

    assert store.load().checkpoint == initial
    staging = active.run_root / settings.run_checkpoint_staging_path
    assert tuple(staging.iterdir()) == ()


def test_checkpoint_store_retry_recovers_replace_before_parent_fsync(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, _ = _persist_initial(store, active)
    successor = _successor(initial, cost=1.0)
    permit = store.issue_write_permit(initial.run_checkpoint_id, successor)
    real_fsync = checkpoint_store_module.os.fsync
    calls = 0

    def fail_parent_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected post-replace failure")
        return real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_store_module.os, "fsync", fail_parent_fsync)
        with pytest.raises(OSError, match="injected post-replace"):
            store.compare_and_swap(permit, successor)

    assert store.load().checkpoint == successor
    retry = store.issue_write_permit(initial.run_checkpoint_id, successor)
    assert store.compare_and_swap(retry, successor).checkpoint == successor


@pytest.mark.parametrize("recovery_fsync_call", (1, 2))
def test_checkpoint_store_recovery_requires_both_rename_directories_durable(
    resolver_case,
    tmp_path,
    monkeypatch,
    recovery_fsync_call,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, _ = _persist_initial(store, active)
    successor = _successor(initial, cost=1.0)
    permit = store.issue_write_permit(initial.run_checkpoint_id, successor)
    real_fsync = checkpoint_store_module.os.fsync
    initial_calls = 0

    def fail_initial_parent_fsync(descriptor):
        nonlocal initial_calls
        initial_calls += 1
        if initial_calls == 2:
            raise OSError("injected initial rename durability failure")
        return real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(
            checkpoint_store_module.os,
            "fsync",
            fail_initial_parent_fsync,
        )
        with pytest.raises(OSError, match="initial rename durability"):
            store.compare_and_swap(permit, successor)

    recovery_calls = 0

    def fail_recovery_directory_fsync(descriptor):
        nonlocal recovery_calls
        recovery_calls += 1
        if recovery_calls == recovery_fsync_call:
            raise OSError("injected recovery directory durability failure")
        return real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(
            checkpoint_store_module.os,
            "fsync",
            fail_recovery_directory_fsync,
        )
        with pytest.raises(OSError, match="recovery directory durability"):
            store.load()

    assert store.load().checkpoint == successor


def test_checkpoint_store_reconciles_only_exact_checkpoint_ahead_of_journal(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, _ = _persist_initial(store, active)
    successor = _successor(initial, cost=1.0)
    permit = store.issue_write_permit(initial.run_checkpoint_id, successor)

    def fail_journal_append(*args, **kwargs):
        raise OSError("injected checkpoint-journal seam")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_append_head", fail_journal_append)
        with pytest.raises(OSError, match="checkpoint-journal seam"):
            store.compare_and_swap(permit, successor)

    assert store.load().checkpoint == successor


def test_checkpoint_store_rejects_deleted_durable_checkpoint(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    _persist_initial(store, active)
    checkpoint_path = active.run_root / settings.run_checkpoint_path
    checkpoint_path.unlink()

    with pytest.raises(RunCheckpointStoreError, match="absent checkpoint"):
        store.load()


def test_checkpoint_store_rejects_stale_canonical_checkpoint_substitution(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, _ = _persist_initial(store, active)
    checkpoint_path = active.run_root / settings.run_checkpoint_path
    initial_bytes = checkpoint_path.read_bytes()
    successor = _successor(initial, cost=1.0)
    permit = store.issue_write_permit(initial.run_checkpoint_id, successor)
    store.compare_and_swap(permit, successor)

    checkpoint_path.unlink()
    checkpoint_path.write_bytes(initial_bytes)
    checkpoint_path.chmod(0o400)

    with pytest.raises(RunCheckpointStoreError, match="rolled back"):
        store.load()
    with pytest.raises(RunCheckpointStoreError, match="rolled back"):
        RunCheckpointStore(active, settings)


def test_checkpoint_store_rejects_missing_or_replaced_journal_and_lock(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    _persist_initial(store, active)
    journal_path = active.run_root / settings.run_checkpoint_journal_path
    journal_bytes = journal_path.read_bytes()
    journal_path.rename(tmp_path / "prior-journal")

    with pytest.raises((LaunchWorkspaceError, OSError)):
        store.load()

    journal_path.write_bytes(journal_bytes)
    journal_path.chmod(0o600)
    with pytest.raises(LaunchWorkspaceError, match="control file changed"):
        store.load()

    lock_active, lock_settings = _active_launch(
        resolver_case,
        tmp_path / "lock",
    )
    lock_store = RunCheckpointStore(lock_active, lock_settings)
    _persist_initial(lock_store, lock_active)
    lock_path = lock_active.run_root / lock_settings.run_checkpoint_lock_path
    lock_path.rename(tmp_path / "prior-lock")
    lock_path.write_bytes(b"")
    lock_path.chmod(0o600)
    with pytest.raises(LaunchWorkspaceError, match="control file changed"):
        lock_store.load()


def test_checkpoint_store_repairs_only_exact_partial_journal_record(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, _ = _persist_initial(store, active)
    successor = _successor(initial, cost=1.0)
    permit = store.issue_write_permit(initial.run_checkpoint_id, successor)
    real_write = checkpoint_store_module.os.write
    calls = 0

    def partial_then_fail(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            prefix_size = max(1, len(payload) // 2)
            return real_write(descriptor, payload[:prefix_size])
        raise OSError("injected partial journal append")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_store_module.os, "write", partial_then_fail)
        with pytest.raises(OSError, match="partial journal append"):
            store.compare_and_swap(permit, successor)

    assert store.load().checkpoint == successor


def test_checkpoint_store_rejects_unrelated_partial_journal_tail(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    _persist_initial(store, active)
    journal_path = active.run_root / settings.run_checkpoint_journal_path
    with journal_path.open("ab") as handle:
        handle.write(b"unrelated-tail")
        handle.flush()
        os.fsync(handle.fileno())

    with pytest.raises(RunCheckpointStoreError, match="impossible incomplete"):
        store.load()


def test_checkpoint_store_rejects_full_journal_before_checkpoint_replace(
    resolver_case,
    tmp_path,
):
    active, settings = _active_launch(resolver_case, tmp_path)
    store = RunCheckpointStore(active, settings)
    initial, _ = _persist_initial(store, active)
    successor = _successor(initial, cost=1.0)
    journal_path = active.run_root / settings.run_checkpoint_journal_path
    checkpoint_path = active.run_root / settings.run_checkpoint_path
    prior_journal = journal_path.read_bytes()
    prior_checkpoint = checkpoint_path.read_bytes()
    successor_head = (
        RunCheckpointHead.initial(active.bootstrap_pin)
        .advance(initial)
        .advance(successor)
    )
    record_size = len(successor_head.to_json_bytes()) + 1
    store._settings = replace(
        settings,
        run_checkpoint_size_bytes=max(
            len(prior_checkpoint),
            len(successor.to_json_bytes()),
        )
        + 1,
        run_checkpoint_journal_size_bytes=len(prior_journal) + record_size - 1,
    )
    permit = store.issue_write_permit(initial.run_checkpoint_id, successor)

    with pytest.raises(RunCheckpointStoreError, match="cannot durably append"):
        store.compare_and_swap(permit, successor)

    assert checkpoint_path.read_bytes() == prior_checkpoint
    assert journal_path.read_bytes() == prior_journal
    assert store.load().checkpoint == initial
