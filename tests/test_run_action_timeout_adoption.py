"""Descriptor-level adoption of an already-published timeout directive."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.cross_run.launch.run_action_timeout_adoption as timeout_adoption_module
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    open_run_action_release_inspection,
)
from kapso.cross_run.launch.run_action_timeout_adoption import (
    open_run_action_timeout_inspection,
    RunActionTimeoutAdoptionError,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionProviderTerminationReason,
)
from test_run_action_release_adoption import (
    _control_lease,
    _install_control_projection,
    _launch_settings,
    _release_case,
    _write_release,
)
from test_run_action_termination_contracts import (
    _remint,
    _termination_graph,
    _timeout_publication,
)


class _TimeoutMetadataOS:
    """Project one test timeout inode into the prepared control namespace."""

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


def _timeout_case(
    tmp_path: Path,
    monkeypatch,
    *,
    payload: bytes | None = None,
    payload_factory=None,
    mode: int = 0o400,
):
    activation_event, expected_adoption = _release_case()
    activation = activation_event.activation_revalidation_receipt
    prepared = activation.prepared_execution
    authority = prepared.runtime_volume_authority
    release_path = tmp_path / "release"
    _write_release(
        release_path,
        expected_adoption.workload_release_receipt.to_json_bytes(),
        authority.owner_user_id,
        authority.owner_group_id,
    )
    release_control_lease = _control_lease(
        tmp_path,
        topology=RunActionControlDirectoryTopology.TIMED_OUT,
    )
    _install_control_projection(
        monkeypatch,
        tmp_path,
        activation_event,
        release_control_lease,
    )
    release_inspection = open_run_action_release_inspection(
        activation_event=activation_event,
        launch_settings=_launch_settings(),
    )
    adoption = release_inspection.adoption
    expected_publication = _timeout_publication(activation, adoption)
    timeout_path = tmp_path / "timeout"
    if payload_factory is not None:
        payload = payload_factory(expected_publication).to_json_bytes()
    timeout_path.write_bytes(
        expected_publication.timeout_directive.to_json_bytes()
        if payload is None
        else payload
    )
    os.chown(timeout_path, authority.owner_user_id, authority.owner_group_id)
    timeout_path.chmod(mode)
    timeout_metadata = os.stat(timeout_path, follow_symlinks=False)
    control = prepared.control_directory
    monkeypatch.setattr(
        timeout_adoption_module,
        "open_run_action_release_inspection",
        lambda **_arguments: release_inspection,
    )
    monkeypatch.setattr(
        timeout_adoption_module,
        "read_run_action_descriptor_mount_id",
        lambda _descriptor, _byte_limit: control.mount_id,
    )
    monkeypatch.setattr(
        timeout_adoption_module,
        "os",
        _TimeoutMetadataOS(
            original_inode=timeout_metadata.st_ino,
            contract_device=control.device,
            contract_inode=expected_publication.timeout_inode,
        ),
    )
    return activation_event, adoption, expected_publication, timeout_path


def test_canonical_timeout_is_adopted_as_one_exact_publication(
    tmp_path,
    monkeypatch,
):
    activation_event, adoption, expected_publication, _timeout_path = _timeout_case(
        tmp_path,
        monkeypatch,
    )

    with open_run_action_timeout_inspection(
        activation_event=activation_event,
        launch_settings=_launch_settings(),
    ) as inspection:
        assert inspection.topology is RunActionControlDirectoryTopology.TIMED_OUT
        assert inspection.workload_release_adoption == adoption
        assert inspection.timeout_directive_publication == expected_publication
        inspection.require_current()


@pytest.mark.parametrize(
    ("payload", "mode", "error_type", "message"),
    (
        (b"not-json", 0o400, json.JSONDecodeError, None),
        (None, 0o600, RunActionTimeoutAdoptionError, "unsafe physical identity"),
        (b"", 0o400, RunActionTimeoutAdoptionError, "empty or exceeds"),
    ),
)
def test_malformed_timeout_publication_fails_loud(
    tmp_path,
    monkeypatch,
    payload,
    mode,
    error_type,
    message,
):
    activation_event, _adoption, _publication, _timeout_path = _timeout_case(
        tmp_path,
        monkeypatch,
        payload=payload,
        mode=mode,
    )

    with pytest.raises(error_type, match=message):
        open_run_action_timeout_inspection(
            activation_event=activation_event,
            launch_settings=_launch_settings(),
        )


def test_retained_timeout_rejects_path_replacement(tmp_path, monkeypatch):
    activation_event, _adoption, publication, timeout_path = _timeout_case(
        tmp_path,
        monkeypatch,
    )
    inspection = open_run_action_timeout_inspection(
        activation_event=activation_event,
        launch_settings=_launch_settings(),
    )
    replacement_path = tmp_path / "replacement"
    replacement_path.write_bytes(publication.timeout_directive.to_json_bytes())
    metadata = os.stat(timeout_path, follow_symlinks=False)
    os.chown(replacement_path, metadata.st_uid, metadata.st_gid)
    replacement_path.chmod(0o400)
    replacement_path.replace(timeout_path)

    with pytest.raises(
        RunActionTimeoutAdoptionError,
        match="changed or was replaced",
    ):
        inspection.require_current()
    inspection.close()


def test_over_bound_timeout_payload_fails_before_parsing(tmp_path, monkeypatch):
    settings = _launch_settings()
    activation_event, _adoption, _publication, _timeout_path = _timeout_case(
        tmp_path,
        monkeypatch,
        payload=b"x" * (settings.run_action_timeout_directive_size_bytes + 1),
    )

    with pytest.raises(RunActionTimeoutAdoptionError, match="exceeds"):
        open_run_action_timeout_inspection(
            activation_event=activation_event,
            launch_settings=settings,
        )


def test_canonical_timeout_for_another_occurrence_fails_loud(
    tmp_path,
    monkeypatch,
):
    foreign = _termination_graph(
        RunActionProviderTerminationReason.TIMEOUT
    ).timeout_directive_publication.timeout_directive
    activation_event, _adoption, _publication, _timeout_path = _timeout_case(
        tmp_path,
        monkeypatch,
        payload_factory=lambda expected: _remint(
            expected.timeout_directive,
            activation_event_id=foreign.activation_event_id,
        ),
    )

    with pytest.raises(
        RunActionTimeoutAdoptionError,
        match="differs from durable activation and release",
    ):
        open_run_action_timeout_inspection(
            activation_event=activation_event,
            launch_settings=_launch_settings(),
        )


def test_closed_timeout_inspection_cannot_be_reused(tmp_path, monkeypatch):
    activation_event, _adoption, _publication, _timeout_path = _timeout_case(
        tmp_path,
        monkeypatch,
    )
    inspection = open_run_action_timeout_inspection(
        activation_event=activation_event,
        launch_settings=_launch_settings(),
    )

    inspection.close()

    with pytest.raises(RunActionTimeoutAdoptionError, match="closed or foreign"):
        inspection.require_current()
