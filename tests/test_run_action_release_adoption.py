"""Descriptor-level classification and adoption of durable workload releases."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.cross_run.launch.run_action_release_adoption as adoption_module
from kapso.core.config import load_config
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    RunActionReleaseAdoptionError,
    open_run_action_release_inspection,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionReleaseContractError,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    RunActionControlDirectoryLease,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_release_contracts import (
    _activation_event,
    _release_adoption_for_event,
    _resolved_for_security,
    _security_observation,
)

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


class _ReleaseMetadataOS:
    """Project real test inodes into the prepared contract's device namespace."""

    def __init__(
        self,
        *,
        original_inode: int,
        contract_device: int,
        contract_inode: int,
    ) -> None:
        self._original_inode = original_inode
        self._contract_device = contract_device
        self._contract_inode = contract_inode

    def __getattr__(self, name):
        return getattr(os, name)

    def fstat(self, descriptor):
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            return observed
        projected_inode = (
            self._contract_inode
            if observed.st_ino == self._original_inode
            else self._contract_inode + 1
        )
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_uid=observed.st_uid,
            st_gid=observed.st_gid,
            st_nlink=observed.st_nlink,
            st_size=observed.st_size,
            st_dev=self._contract_device,
            st_ino=projected_inode,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )


def _launch_settings():
    return CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).launch


def _release_case():
    security = _security_observation()
    resolved = _resolved_for_security(security)
    activation_event = _activation_event(resolved)
    adoption = _release_adoption_for_event(activation_event, security)
    return activation_event, adoption


def _control_lease(
    control_path: Path,
    *,
    topology: RunActionControlDirectoryTopology,
    require_current=None,
) -> RunActionControlDirectoryLease:
    descriptor = os.open(
        control_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    lease = object.__new__(RunActionControlDirectoryLease)
    lease._control_descriptor = descriptor
    lease._topology = topology
    lease.require_current = (
        (lambda: None) if require_current is None else require_current
    )
    lease.close = lambda: os.close(descriptor)
    return lease


def _write_release(path: Path, payload: bytes, owner_user_id: int, owner_group_id: int):
    path.write_bytes(payload)
    os.chown(path, owner_user_id, owner_group_id)
    path.chmod(0o400)


def _install_control_projection(monkeypatch, control_path, activation_event, lease):
    control = (
        activation_event.activation_revalidation_receipt.prepared_execution.control_directory
    )
    release_path = control_path / "release"
    release_metadata = os.stat(release_path, follow_symlinks=False)
    monkeypatch.setattr(
        adoption_module,
        "open_run_action_control_directory",
        lambda _prepared: lease,
    )
    monkeypatch.setattr(
        adoption_module,
        "read_run_action_descriptor_mount_id",
        lambda _descriptor, _byte_limit: control.mount_id,
    )
    monkeypatch.setattr(
        adoption_module,
        "os",
        _ReleaseMetadataOS(
            original_inode=release_metadata.st_ino,
            contract_device=control.device,
            contract_inode=control.inode + 1000,
        ),
    )


def test_empty_control_directory_is_classified_absent(tmp_path, monkeypatch):
    activation_event, _adoption = _release_case()
    lease = _control_lease(
        tmp_path,
        topology=RunActionControlDirectoryTopology.EMPTY,
    )
    monkeypatch.setattr(
        adoption_module,
        "open_run_action_control_directory",
        lambda _prepared: lease,
    )

    with open_run_action_release_inspection(
        activation_event=activation_event,
        launch_settings=_launch_settings(),
    ) as inspection:
        assert inspection.topology is RunActionControlDirectoryTopology.EMPTY
        with pytest.raises(
            RunActionReleaseAdoptionError,
            match="absent release inspection has no adoption",
        ):
            inspection.adoption


def test_process_snapshot_policy_mismatch_fails_before_control_inspection(
    monkeypatch,
):
    activation_event, _adoption = _release_case()
    settings = _launch_settings()
    mismatched_settings = replace(
        settings,
        run_action_process_snapshot_size_bytes=(
            settings.run_action_process_snapshot_size_bytes + 1
        ),
    )
    monkeypatch.setattr(
        adoption_module,
        "open_run_action_control_directory",
        lambda _prepared: pytest.fail(
            "control inspection preceded process-bound validation"
        ),
    )

    with pytest.raises(
        RunActionReleaseAdoptionError,
        match="policy differs from configured control bounds",
    ):
        open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=mismatched_settings,
        )


@pytest.mark.parametrize(
    "topology",
    (
        RunActionControlDirectoryTopology.RELEASED,
        RunActionControlDirectoryTopology.TIMED_OUT,
    ),
)
def test_canonical_release_is_adopted_from_its_retained_descriptor(
    tmp_path,
    monkeypatch,
    topology,
):
    activation_event, expected_adoption = _release_case()
    prepared = activation_event.activation_revalidation_receipt.prepared_execution
    release_path = tmp_path / "release"
    payload = expected_adoption.workload_release_receipt.to_json_bytes()
    _write_release(
        release_path,
        payload,
        prepared.runtime_volume_authority.owner_user_id,
        prepared.runtime_volume_authority.owner_group_id,
    )
    lease = _control_lease(
        tmp_path,
        topology=topology,
    )
    _install_control_projection(monkeypatch, tmp_path, activation_event, lease)

    with open_run_action_release_inspection(
        activation_event=activation_event,
        launch_settings=_launch_settings(),
    ) as inspection:
        assert inspection.topology is topology
        assert (
            inspection.adoption.workload_release_receipt
            == expected_adoption.workload_release_receipt
        )
        assert (
            inspection.adoption.release_inode == prepared.control_directory.inode + 1000
        )


def test_release_with_wrong_mode_fails_loud(tmp_path, monkeypatch):
    activation_event, expected_adoption = _release_case()
    prepared = activation_event.activation_revalidation_receipt.prepared_execution
    release_path = tmp_path / "release"
    _write_release(
        release_path,
        expected_adoption.workload_release_receipt.to_json_bytes(),
        prepared.runtime_volume_authority.owner_user_id,
        prepared.runtime_volume_authority.owner_group_id,
    )
    release_path.chmod(0o600)
    lease = _control_lease(
        tmp_path,
        topology=RunActionControlDirectoryTopology.RELEASED,
    )
    _install_control_projection(monkeypatch, tmp_path, activation_event, lease)

    with pytest.raises(
        RunActionReleaseAdoptionError,
        match="unsafe physical identity",
    ):
        open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=_launch_settings(),
        )


def test_release_for_another_event_fails_loud(tmp_path, monkeypatch):
    activation_event, expected_adoption = _release_case()
    prepared = activation_event.activation_revalidation_receipt.prepared_execution
    other_event = _activation_event(
        expected_adoption.workload_release_receipt.resolved_workload_observation,
        predecessor_label="other-event-four",
    )
    other_receipt = _release_adoption_for_event(
        other_event,
        _security_observation(),
    ).workload_release_receipt
    release_path = tmp_path / "release"
    _write_release(
        release_path,
        other_receipt.to_json_bytes(),
        prepared.runtime_volume_authority.owner_user_id,
        prepared.runtime_volume_authority.owner_group_id,
    )
    lease = _control_lease(
        tmp_path,
        topology=RunActionControlDirectoryTopology.RELEASED,
    )
    _install_control_projection(monkeypatch, tmp_path, activation_event, lease)

    with pytest.raises(
        RunActionReleaseContractError,
        match="identifies another activation event",
    ):
        open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=_launch_settings(),
        )


def test_malformed_release_bytes_fail_loud(tmp_path, monkeypatch):
    activation_event, _expected_adoption = _release_case()
    prepared = activation_event.activation_revalidation_receipt.prepared_execution
    release_path = tmp_path / "release"
    _write_release(
        release_path,
        b"not-json",
        prepared.runtime_volume_authority.owner_user_id,
        prepared.runtime_volume_authority.owner_group_id,
    )
    lease = _control_lease(
        tmp_path,
        topology=RunActionControlDirectoryTopology.RELEASED,
    )
    _install_control_projection(monkeypatch, tmp_path, activation_event, lease)

    with pytest.raises(json.JSONDecodeError):
        open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=_launch_settings(),
        )


def test_path_replacement_during_adoption_fails_loud(tmp_path, monkeypatch):
    activation_event, expected_adoption = _release_case()
    prepared = activation_event.activation_revalidation_receipt.prepared_execution
    release_path = tmp_path / "release"
    replacement_path = tmp_path / "replacement"
    payload = expected_adoption.workload_release_receipt.to_json_bytes()
    for path in (release_path, replacement_path):
        _write_release(
            path,
            payload,
            prepared.runtime_volume_authority.owner_user_id,
            prepared.runtime_volume_authority.owner_group_id,
        )
    require_count = {"value": 0}

    def require_current():
        require_count["value"] += 1
        if require_count["value"] == 4:
            replacement_path.replace(release_path)

    lease = _control_lease(
        tmp_path,
        topology=RunActionControlDirectoryTopology.RELEASED,
        require_current=require_current,
    )
    _install_control_projection(monkeypatch, tmp_path, activation_event, lease)

    with pytest.raises(
        RunActionReleaseAdoptionError,
        match="changed or was replaced",
    ):
        open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=_launch_settings(),
        )
