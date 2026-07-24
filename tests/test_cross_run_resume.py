"""Descriptor-pinned restart and lifetime-runtime authority tests."""

from __future__ import annotations

import fcntl
import os
from copy import copy
from dataclasses import replace

import pytest

from kapso.cross_run.launch.run_state_publisher import RunStatePublisher
from kapso.cross_run.launch.workspace import (
    LaunchWorkspaceError,
    StarterWorkspaceBuilder,
)
from test_launch_resolver import resolver_case
from test_run_state_publisher import _genesis


def _build_active(resolver_case, run_root):
    settings = resolver_case["resolver"]._settings
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    prepared = StarterWorkspaceBuilder(settings).build(
        resolved,
        run_root,
        run_id="resume-run",
        campaign_id="resume-campaign",
    )
    return prepared.activate()


def test_reopen_uses_the_original_pin_and_reconciles_published_state(
    resolver_case,
    tmp_path,
) -> None:
    settings = resolver_case["resolver"]._settings
    active = _build_active(resolver_case, (tmp_path / "run").absolute())
    _projection, bundle, checkpoint = _genesis(active, resolver_case)
    publisher = RunStatePublisher(active, settings.launch)
    published = publisher.publish(
        publisher.issue_publication_permit(None, checkpoint, bundle),
        checkpoint,
        bundle,
    )
    run_root = active.run_root
    original_pin = active.bootstrap_pin
    active.close()

    resumed = StarterWorkspaceBuilder(settings).reopen(run_root)
    resumed_publisher = RunStatePublisher(resumed, settings.launch)
    frontier = resumed_publisher.load_reconciled()

    assert resumed.bootstrap_pin == original_pin
    assert frontier.require_current(resumed_publisher) == checkpoint
    assert frontier.projection == published.projection
    resumed.close()


def test_reopen_preserves_an_evolved_or_recovery_pending_workspace(
    resolver_case,
    tmp_path,
) -> None:
    settings = resolver_case["resolver"]._settings
    active = _build_active(resolver_case, (tmp_path / "run").absolute())
    source_descriptor = active.bootstrap_pin.launch_manifest.expert_source.extraction_receipt.source_tree_files[
        0
    ]
    source_path = active.workspace / source_descriptor.relative_path
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    run_root = active.run_root
    active.close()

    resumed = StarterWorkspaceBuilder(settings).reopen(run_root)
    resumed.require_control_authority()
    assert source_path.read_bytes().endswith(b"\n")
    resumed.close()


def test_runtime_lock_excludes_a_second_controller_until_close(
    resolver_case,
    tmp_path,
) -> None:
    settings = resolver_case["resolver"]._settings
    active = _build_active(resolver_case, (tmp_path / "run").absolute())

    with pytest.raises(BlockingIOError):
        StarterWorkspaceBuilder(settings).reopen(active.run_root)

    run_root = active.run_root
    active.close()
    resumed = StarterWorkspaceBuilder(settings).reopen(run_root)
    resumed.require_control_authority()
    resumed.close()


def test_cloned_runtime_cannot_act_or_close_the_issued_lease(
    resolver_case,
    tmp_path,
) -> None:
    active = _build_active(resolver_case, (tmp_path / "run").absolute())
    reconstructed_clone = replace(active)
    shallow_clone = copy(active)

    with pytest.raises(LaunchWorkspaceError, match="authority"):
        reconstructed_clone.require_control_authority()
    with pytest.raises(LaunchWorkspaceError, match="authority"):
        reconstructed_clone.close()
    with pytest.raises(LaunchWorkspaceError, match="authority"):
        shallow_clone.close()
    active.require_control_authority()
    active.close()


def test_runtime_retains_only_root_and_lock_descriptors(
    resolver_case,
    tmp_path,
) -> None:
    active = _build_active(resolver_case, (tmp_path / "run").absolute())

    assert active._lifecycle.descriptors is not None
    assert len(active._lifecycle.descriptors._exit_callbacks) == 2
    active.require_control_authority()
    active.close()


def test_activation_holds_the_runtime_lock_before_terminal_verification(
    resolver_case,
    tmp_path,
    monkeypatch,
) -> None:
    settings = resolver_case["resolver"]._settings
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    builder = StarterWorkspaceBuilder(settings)
    prepared = builder.build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="locked-verification-run",
        campaign_id="resume-campaign",
    )
    original_verify = builder._verify_published
    runtime_lock = (
        prepared.run_root
        / prepared.bootstrap_pin.installation_receipt.layout.run_runtime_lock_relative_path
    )

    def verify_while_locked(*arguments, **keywords):
        descriptor = os.open(
            runtime_lock,
            os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        with pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(descriptor)
        return original_verify(*arguments, **keywords)

    monkeypatch.setattr(builder, "_verify_published", verify_while_locked)
    active = prepared.activate()
    active.close()


def test_reopen_rejects_settings_and_noncanonical_pin_substitution(
    resolver_case,
    tmp_path,
) -> None:
    settings = resolver_case["resolver"]._settings
    active = _build_active(resolver_case, (tmp_path / "run").absolute())
    run_root = active.run_root
    pin_path = (
        run_root
        / active.bootstrap_pin.installation_receipt.layout.bootstrap_pin_relative_path
    )
    active.close()

    changed_launch = replace(
        settings.launch,
        run_checkpoint_size_bytes=settings.launch.run_checkpoint_size_bytes + 1,
    )
    with pytest.raises(LaunchWorkspaceError, match="configured launch authority"):
        StarterWorkspaceBuilder(replace(settings, launch=changed_launch)).reopen(
            run_root
        )

    pin_payload = pin_path.read_bytes()
    pin_path.chmod(0o644)
    pin_path.write_bytes(pin_payload + b"\n")
    pin_path.chmod(0o444)
    with pytest.raises(
        (LaunchWorkspaceError, ValueError),
        match="configured launch authority|canonical",
    ):
        StarterWorkspaceBuilder(settings).reopen(run_root)


def test_forked_child_cannot_use_or_release_parent_runtime_authority(
    resolver_case,
    tmp_path,
) -> None:
    settings = resolver_case["resolver"]._settings
    active = _build_active(resolver_case, (tmp_path / "run").absolute())
    read_descriptor, write_descriptor = os.pipe()
    child_process_id = os.fork()
    if child_process_id == 0:
        os.close(read_descriptor)
        with pytest.raises(LaunchWorkspaceError, match="authority"):
            active.require_control_authority()
        os.write(write_descriptor, b"invalid")
        os.close(write_descriptor)
        os._exit(0)

    os.close(write_descriptor)
    assert os.read(read_descriptor, len(b"invalid")) == b"invalid"
    os.close(read_descriptor)
    waited_process_id, status = os.waitpid(child_process_id, 0)
    assert waited_process_id == child_process_id
    assert os.waitstatus_to_exitcode(status) == 0
    with pytest.raises(BlockingIOError):
        StarterWorkspaceBuilder(settings).reopen(active.run_root)
    active.close()


def test_process_death_releases_the_runtime_lock_for_resume(
    resolver_case,
    tmp_path,
) -> None:
    settings = resolver_case["resolver"]._settings
    initial = _build_active(resolver_case, (tmp_path / "run").absolute())
    run_root = initial.run_root
    initial.close()
    ready_reader, ready_writer = os.pipe()
    exit_reader, exit_writer = os.pipe()
    child_process_id = os.fork()
    if child_process_id == 0:
        os.close(ready_reader)
        os.close(exit_writer)
        child_active = StarterWorkspaceBuilder(settings).reopen(run_root)
        os.write(ready_writer, b"ready")
        os.read(exit_reader, 1)
        child_active.require_control_authority()
        os._exit(29)

    os.close(ready_writer)
    os.close(exit_reader)
    assert os.read(ready_reader, len(b"ready")) == b"ready"
    os.close(ready_reader)
    with pytest.raises(BlockingIOError):
        StarterWorkspaceBuilder(settings).reopen(run_root)
    os.write(exit_writer, b"x")
    os.close(exit_writer)
    waited_process_id, status = os.waitpid(child_process_id, 0)
    assert waited_process_id == child_process_id
    assert os.waitstatus_to_exitcode(status) == 29

    resumed = StarterWorkspaceBuilder(settings).reopen(run_root)
    resumed.require_control_authority()
    resumed.close()
