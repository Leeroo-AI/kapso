from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from dataclasses import replace

import pytest

import kapso.cross_run.docker.runtime as docker_runtime_module
import kapso.cross_run.launch.run_action_docker_inspect as docker_inspect_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import PinnedDockerRuntime
from kapso.cross_run.launch import run_action_runtime_volume as volume_module
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    target_command_from_main_projection,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    DockerRunActionEmptyVolumeObservation,
    DockerRunActionPreparedVolumeObservation,
    RunActionRuntimeVolumeError,
    _materialize_layout_at_descriptor,
    _open_exact_regular_file,
    _plan_runtime_volume_layout,
    _require_same_exact_regular_file,
    adopt_prepared_runtime_volume_layout,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_descriptor_mount_id as _read_run_action_descriptor_mount_id,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE,
    RunActionContainerLabel,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionPreparationAllocation,
    RunActionPreparationClaim,
    RunActionPreparedExecution,
    RunActionRuntimeVolumeSentinelEvidence,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
)
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_docker_inspect import _container_raw, _volume_raw
from test_run_action_docker_projection import _policy
from test_run_action_docker_resources import _InventoryDockerRunner
from test_launch_resolver import resolver_case
from test_run_action_supervisor_contracts import (
    _claim,
    _activation_revalidation_receipt,
    _fixture_content_id,
    _prepared_execution,
    _remint_contract,
    _result_capture_receipt,
    _spawn_commit,
    _terminal_observation,
    _volume_authority,
)
from test_run_state_publisher import publisher_case

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES = CrossRunSettings.from_dict(
    load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
).launch.run_action_process_snapshot_size_bytes
_GENERATION_NONCE = "9" * 32
_TEST_DOCKER_BYTES = b"prepared-layout adoption Docker"
_CREDENTIAL_LEASE_AUTHORITY_ID = _fixture_content_id(
    RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE,
    "credential lease",
)


def read_run_action_descriptor_mount_id(descriptor: int) -> int:
    return _read_run_action_descriptor_mount_id(
        descriptor,
        _RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES,
    )


@pytest.mark.parametrize(
    "mutation",
    ("entry", "path", "sentinel"),
)
def test_barrier_control_lease_retains_exact_empty_generation(
    layout_context,
    tmp_path,
    monkeypatch,
    mutation,
):
    settings, _claim_without_workspace, _authority, _empty = layout_context
    prepared, root_path, root_mount_id, root_metadata = _physical_barrier_control_case(
        tmp_path, settings
    )
    opened_roots = []

    def open_test_volume(descriptors, keeper):
        root = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        opened_roots.append(root)
        descriptors.callback(os.close, root)
        return volume_module._MountedRuntimeVolumeLease(
            process_descriptor=root,
            root_descriptor=root,
            keeper_container_id=keeper.container_id,
            keeper_process_id=keeper.process_id,
            process_start_time_ticks=keeper.process_start_time_ticks,
            process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
            root_mount_id=root_mount_id,
            root_device=root_metadata.st_dev,
            root_inode=root_metadata.st_ino,
            process_snapshot_size_limit_bytes=(
                keeper.issued_create_projection.execution_policy.supervisor_limits.process_snapshot_size_bytes
            ),
        )

    monkeypatch.setattr(
        volume_module,
        "_open_mounted_runtime_volume",
        open_test_volume,
    )
    monkeypatch.setattr(
        volume_module,
        "_require_same_mounted_runtime_volume",
        lambda _mounted, _keeper: None,
    )
    lease = volume_module.open_run_action_control_directory(prepared)
    assert "control_descriptor" not in type(lease).__dict__
    control_descriptor = lease._control_descriptor
    assert os.listdir(control_descriptor) == []
    owner_process_id = os.getpid()
    monkeypatch.setattr(
        volume_module.os,
        "getpid",
        lambda: owner_process_id + 1,
    )
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="belongs to another process",
    ):
        lease.require_current()
    monkeypatch.setattr(volume_module.os, "getpid", lambda: owner_process_id)
    lease.require_current()

    if mutation == "entry":
        (root_path / "control" / "unexpected").write_bytes(b"not a release")
    elif mutation == "path":
        (root_path / "control").rename(root_path / "detached-control")
        (root_path / "control").mkdir(mode=0o700)
    else:
        sentinel_path = root_path / ".kapso-generation"
        sentinel_path.unlink()
        sentinel_path.write_bytes(
            prepared.runtime_volume_authority.generation_nonce.encode("ascii")
        )
        sentinel_path.chmod(0o400)

    with pytest.raises(RunActionRuntimeVolumeError):
        lease.require_current()
    lease.close()
    with pytest.raises(RunActionRuntimeVolumeError, match="closed"):
        lease.require_current()
    with pytest.raises(OSError):
        os.fstat(control_descriptor)
    with pytest.raises(OSError):
        os.fstat(opened_roots[-1])


@pytest.mark.parametrize(
    ("entries", "expected_topology"),
    (
        ((), RunActionControlDirectoryTopology.EMPTY),
        (("release",), RunActionControlDirectoryTopology.RELEASED),
        (
            ("release", "timeout"),
            RunActionControlDirectoryTopology.TIMED_OUT,
        ),
        (("timeout",), None),
        (("release", "unexpected"), None),
    ),
)
def test_barrier_control_lease_admits_only_closed_semantic_topologies(
    layout_context,
    tmp_path,
    monkeypatch,
    entries,
    expected_topology,
):
    settings, _claim_without_workspace, _authority, _empty = layout_context
    prepared, root_path, root_mount_id, root_metadata = _physical_barrier_control_case(
        tmp_path,
        settings,
    )
    control_path = root_path / "control"
    for entry in entries:
        (control_path / entry).write_bytes(entry.encode("ascii"))

    def open_test_volume(descriptors, keeper):
        root = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, root)
        return volume_module._MountedRuntimeVolumeLease(
            process_descriptor=root,
            root_descriptor=root,
            keeper_container_id=keeper.container_id,
            keeper_process_id=keeper.process_id,
            process_start_time_ticks=keeper.process_start_time_ticks,
            process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
            root_mount_id=root_mount_id,
            root_device=root_metadata.st_dev,
            root_inode=root_metadata.st_ino,
            process_snapshot_size_limit_bytes=(
                keeper.issued_create_projection.execution_policy.supervisor_limits.process_snapshot_size_bytes
            ),
        )

    monkeypatch.setattr(
        volume_module,
        "_open_mounted_runtime_volume",
        open_test_volume,
    )
    monkeypatch.setattr(
        volume_module,
        "_require_same_mounted_runtime_volume",
        lambda _mounted, _keeper: None,
    )

    if expected_topology is None:
        with pytest.raises(
            RunActionRuntimeVolumeError,
            match="invalid semantic topology",
        ):
            volume_module.open_run_action_control_directory(prepared)
        return

    lease = volume_module.open_run_action_control_directory(prepared)
    assert lease.topology is expected_topology
    if expected_topology is RunActionControlDirectoryTopology.EMPTY:
        (control_path / "release").write_bytes(b"release")
    elif expected_topology is RunActionControlDirectoryTopology.RELEASED:
        (control_path / "timeout").write_bytes(b"timeout")
    else:
        (control_path / "timeout").unlink()
    with pytest.raises(RunActionRuntimeVolumeError):
        lease.require_current()
    lease.close()


def test_barrier_control_lease_rejects_substituted_prepared_inode(
    layout_context,
    tmp_path,
    monkeypatch,
):
    settings, _claim_without_workspace, _authority, _empty = layout_context
    prepared, root_path, root_mount_id, root_metadata = _physical_barrier_control_case(
        tmp_path, settings
    )
    control = _remint_contract(
        prepared.control_directory,
        inode=max(
            (
                prepared.runtime_volume_evidence.root_inode,
                prepared.runtime_volume_evidence.sentinel_evidence.inode,
                prepared.input_delivery_slot.inode,
                prepared.result_directory.inode,
                prepared.temporary_directory.inode,
                prepared.result_file.inode,
            )
        )
        + 1,
    )
    layout = _remint_contract(
        prepared.layout_proof,
        prepared_runtime_directory_ids=tuple(
            sorted(
                (
                    control.prepared_runtime_directory_id,
                    prepared.result_directory.prepared_runtime_directory_id,
                    prepared.temporary_directory.prepared_runtime_directory_id,
                )
            )
        ),
    )
    substituted = _remint_contract(
        prepared,
        control_directory=control,
        layout_proof=layout,
    )
    opened_roots = []

    def open_test_volume(descriptors, keeper):
        root = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        opened_roots.append(root)
        descriptors.callback(os.close, root)
        return volume_module._MountedRuntimeVolumeLease(
            process_descriptor=root,
            root_descriptor=root,
            keeper_container_id=keeper.container_id,
            keeper_process_id=keeper.process_id,
            process_start_time_ticks=keeper.process_start_time_ticks,
            process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
            root_mount_id=root_mount_id,
            root_device=root_metadata.st_dev,
            root_inode=root_metadata.st_ino,
            process_snapshot_size_limit_bytes=(
                keeper.issued_create_projection.execution_policy.supervisor_limits.process_snapshot_size_bytes
            ),
        )

    monkeypatch.setattr(
        volume_module,
        "_open_mounted_runtime_volume",
        open_test_volume,
    )
    monkeypatch.setattr(
        volume_module,
        "_require_same_mounted_runtime_volume",
        lambda _mounted, _keeper: None,
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="subpath is unsafe or substituted",
    ):
        volume_module.open_run_action_control_directory(substituted)
    with pytest.raises(OSError):
        os.fstat(opened_roots[-1])


def test_substituted_spawn_is_rejected_before_any_delivery_publication(
    tmp_path,
    monkeypatch,
):
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    policy = _policy(
        settings.docker,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))
    substituted_prepared = _prepared_execution(
        claim=prepared.preparation_claim,
        inode_offset=7,
    )
    substituted_spawn = _spawn_commit(substituted_prepared)
    authority = prepared.runtime_volume_authority
    volume = observe_runtime_volume(
        _volume_raw(authority, settings.docker),
        prepared.preparation_claim,
        authority,
        settings.docker,
    )
    input_final = tmp_path / prepared.input_delivery_slot.final_file_name
    credential_final = tmp_path / prepared.credential_delivery_slot.final_file_name

    def publish_if_called(slot, slot_directory_descriptor, payload):
        destination = input_final if slot.kind.value == "input" else credential_final
        destination.write_bytes(payload)
        raise AssertionError("delivery publication ran before spawn validation")

    monkeypatch.setattr(
        volume_module,
        "publish_or_adopt_run_action_delivery",
        publish_if_called,
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="activation spawn differs",
    ):
        volume_module.deliver_and_reobserve_runtime_volume_activation(
            prepared,
            substituted_spawn,
            volume,
            prepared.volume_keeper_evidence,
            request_payload=b"complete request",
            credential_payload=b"provider-token",
            credential_content_authority_id=_CREDENTIAL_LEASE_AUTHORITY_ID,
            workspace_descriptor=None,
            settings=settings.launch,
        )

    assert input_final.exists() is False
    assert credential_final.exists() is False


def test_unbounded_credential_authority_is_rejected_before_delivery(
    monkeypatch,
):
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    policy = _policy(
        settings.docker,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))
    spawn = _spawn_commit(prepared)
    authority = prepared.runtime_volume_authority
    volume = observe_runtime_volume(
        _volume_raw(authority, settings.docker),
        prepared.preparation_claim,
        authority,
        settings.docker,
    )

    def publish_if_called(*_arguments):
        raise AssertionError("delivery ran before credential authority validation")

    monkeypatch.setattr(
        volume_module,
        "publish_or_adopt_run_action_delivery",
        publish_if_called,
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="fixed lease content ID",
    ):
        volume_module.deliver_and_reobserve_runtime_volume_activation(
            prepared,
            spawn,
            volume,
            prepared.volume_keeper_evidence,
            request_payload=b"complete request",
            credential_payload=b"provider-token",
            credential_content_authority_id="unbounded.legacy.credential.authority",
            workspace_descriptor=None,
            settings=settings.launch,
        )


def test_terminal_workspace_source_requires_exact_prepared_volume_occurrence():
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    activation = _activation_revalidation_receipt(prepared, spawn)
    terminal = _terminal_observation(prepared, spawn)
    capture = _result_capture_receipt(
        prepared,
        activation,
        terminal,
        b'{"result":"complete"}',
    )

    volume_module._require_result_volume_occurrence(
        prepared,
        capture.reobserved_volume_evidence,
    )

    substituted = _prepared_execution(inode_offset=7).runtime_volume_evidence
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="prepared occurrence",
    ):
        volume_module._require_result_volume_occurrence(
            prepared,
            substituted,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "remove",
        "substitute",
        "workspace_timestamp",
        "workspace_path",
        "workspace_gid_observation",
    ),
)
def test_result_workspace_lease_retains_exact_event_6_sentinel(
    tmp_path,
    monkeypatch,
    mutation,
):
    prepared = _prepared_execution()
    authority = prepared.runtime_volume_authority
    keeper = prepared.volume_keeper_evidence
    root_path = tmp_path / "result-volume"
    root_path.mkdir(mode=0o700)
    workspace_path = root_path / "workspace"
    workspace_path.mkdir(mode=0o700)
    sentinel_path = root_path / ".kapso-generation"
    sentinel_payload = authority.generation_nonce.encode("ascii")
    sentinel_path.write_bytes(sentinel_payload)
    sentinel_path.chmod(0o400)
    descriptors = ExitStack()
    root_descriptor = os.open(
        root_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, root_descriptor)
    workspace_descriptor = os.open(
        "workspace",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_descriptor,
    )
    descriptors.callback(os.close, workspace_descriptor)
    root_metadata = os.fstat(root_descriptor)
    workspace_metadata = os.fstat(workspace_descriptor)
    root_mount_id = read_run_action_descriptor_mount_id(root_descriptor)
    sentinel_observation = volume_module._open_exact_regular_file(
        descriptors,
        root_descriptor,
        ".kapso-generation",
        expected_payload=sentinel_payload,
        expected_mode=0o400,
        authority=authority,
        root_mount_id=root_mount_id,
        root_device=root_metadata.st_dev,
        process_snapshot_size_limit_bytes=_RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES,
    )
    sentinel_metadata = sentinel_observation.metadata
    sentinel_evidence = RunActionRuntimeVolumeSentinelEvidence.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=".kapso-generation",
        file_type="regular",
        owner_user_id=sentinel_metadata.st_uid,
        owner_group_id=sentinel_metadata.st_gid,
        mode=stat.S_IMODE(sentinel_metadata.st_mode),
        link_count=sentinel_metadata.st_nlink,
        size_bytes=sentinel_metadata.st_size,
        content_digest=volume_module.tree_or_blob_digest(sentinel_payload),
        mount_id=sentinel_observation.mount_id,
        device=sentinel_metadata.st_dev,
        inode=sentinel_metadata.st_ino,
    )
    volume_module._require_exact_sentinel_observation(
        sentinel_observation,
        sentinel_evidence,
    )
    substituted_evidence = RunActionRuntimeVolumeSentinelEvidence.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=".kapso-generation",
        file_type="regular",
        owner_user_id=sentinel_metadata.st_uid,
        owner_group_id=sentinel_metadata.st_gid,
        mode=stat.S_IMODE(sentinel_metadata.st_mode),
        link_count=sentinel_metadata.st_nlink,
        size_bytes=sentinel_metadata.st_size,
        content_digest=volume_module.tree_or_blob_digest(sentinel_payload),
        mount_id=sentinel_observation.mount_id,
        device=sentinel_metadata.st_dev,
        inode=sentinel_metadata.st_ino + 1,
    )
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="sentinel was substituted",
    ):
        volume_module._require_exact_sentinel_observation(
            sentinel_observation,
            substituted_evidence,
        )
    mounted_volume = volume_module._MountedRuntimeVolumeLease(
        process_descriptor=root_descriptor,
        root_descriptor=root_descriptor,
        keeper_container_id=keeper.container_id,
        keeper_process_id=keeper.process_id,
        process_start_time_ticks=keeper.process_start_time_ticks,
        process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
        root_mount_id=root_mount_id,
        root_device=root_metadata.st_dev,
        root_inode=root_metadata.st_ino,
        process_snapshot_size_limit_bytes=(
            keeper.issued_create_projection.execution_policy.supervisor_limits.process_snapshot_size_bytes
        ),
    )
    monkeypatch.setattr(
        volume_module,
        "_require_same_mounted_runtime_volume",
        lambda _mounted, _keeper: None,
    )
    workspace_proof = _remint_contract(
        prepared.workspace_proof,
        owner_user_id=workspace_metadata.st_uid,
        owner_group_id=workspace_metadata.st_gid,
        root_mode=stat.S_IMODE(workspace_metadata.st_mode),
        mount_id=root_mount_id,
        device=workspace_metadata.st_dev,
        inode=workspace_metadata.st_ino,
    )
    workspace_metadata_identity = volume_module._stable_metadata(workspace_metadata)
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="lacks exact physical authority",
    ):
        volume_module.RunActionResultWorkspaceLease(
            descriptors=descriptors,
            mounted_volume=mounted_volume,
            keeper=keeper,
            sentinel_observation=sentinel_observation,
            workspace_descriptor=workspace_descriptor,
            workspace_proof=workspace_proof,
            workspace_metadata_identity=(
                *workspace_metadata_identity[:-1],
                workspace_metadata_identity[-1] + 1,
            ),
            _authority=volume_module._RESULT_WORKSPACE_LEASE_AUTHORITY,
        )
    lease = volume_module.RunActionResultWorkspaceLease(
        descriptors=descriptors,
        mounted_volume=mounted_volume,
        keeper=keeper,
        sentinel_observation=sentinel_observation,
        workspace_descriptor=workspace_descriptor,
        workspace_proof=workspace_proof,
        workspace_metadata_identity=workspace_metadata_identity,
        _authority=volume_module._RESULT_WORKSPACE_LEASE_AUTHORITY,
    )

    if mutation in {"remove", "substitute"}:
        sentinel_path.unlink()
        if mutation == "substitute":
            sentinel_path.write_bytes(sentinel_payload)
            sentinel_path.chmod(0o400)
            expected_error = RunActionRuntimeVolumeError
        else:
            expected_error = FileNotFoundError
    elif mutation == "workspace_timestamp":
        os.utime(
            workspace_path,
            ns=(
                workspace_metadata.st_atime_ns,
                workspace_metadata.st_mtime_ns + 1,
            ),
        )
        expected_error = RunActionRuntimeVolumeError
    elif mutation == "workspace_path":
        detached_workspace_path = root_path / "detached-workspace"
        workspace_path.rename(detached_workspace_path)
        workspace_path.mkdir(mode=0o700)
        expected_error = RunActionRuntimeVolumeError
    else:
        original_fstat = os.fstat

        def fstat_with_substituted_group(descriptor):
            observed = original_fstat(descriptor)
            if descriptor != workspace_descriptor:
                return observed
            values = list(observed)
            values[5] = observed.st_gid + 1
            return os.stat_result(values)

        monkeypatch.setattr(volume_module.os, "fstat", fstat_with_substituted_group)
        expected_error = RunActionRuntimeVolumeError
    with pytest.raises(expected_error):
        lease.require_current()
    lease.close()


def test_open_result_workspace_joins_live_sentinel_and_retains_its_path(
    layout_context,
    tmp_path,
    monkeypatch,
):
    settings, _claim_without_workspace, _authority, _empty = layout_context
    policy = _policy(
        settings.docker,
        workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))
    spawn = _spawn_commit(prepared)
    activation = _activation_revalidation_receipt(prepared, spawn)
    terminal = _terminal_observation(prepared, spawn)
    capture = _result_capture_receipt(
        prepared,
        activation,
        terminal,
        b'{"result":"complete"}',
    )
    authority = prepared.runtime_volume_authority
    root_path = tmp_path / "captured-result-volume"
    root_path.mkdir(mode=0o700)
    (root_path / "workspace").mkdir(mode=0o700)
    sentinel_path = root_path / ".kapso-generation"
    sentinel_payload = authority.generation_nonce.encode("ascii")
    sentinel_path.write_bytes(sentinel_payload)
    sentinel_path.chmod(0o400)
    with ExitStack() as observations:
        root_descriptor = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        observations.callback(os.close, root_descriptor)
        root_metadata = os.fstat(root_descriptor)
        root_mount_id = read_run_action_descriptor_mount_id(root_descriptor)
        sentinel_metadata = sentinel_path.stat(follow_symlinks=False)
    sentinel_evidence = RunActionRuntimeVolumeSentinelEvidence.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=".kapso-generation",
        file_type="regular",
        owner_user_id=sentinel_metadata.st_uid,
        owner_group_id=sentinel_metadata.st_gid,
        mode=stat.S_IMODE(sentinel_metadata.st_mode),
        link_count=sentinel_metadata.st_nlink,
        size_bytes=sentinel_metadata.st_size,
        content_digest=volume_module.tree_or_blob_digest(sentinel_payload),
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
        inode=sentinel_metadata.st_ino,
    )
    captured_volume = _remint_contract(
        capture.reobserved_volume_evidence,
        root_mount_id=root_mount_id,
        root_device=root_metadata.st_dev,
        root_inode=root_metadata.st_ino,
        sentinel_evidence=sentinel_evidence,
    )
    exact_capture = _remint_contract(
        capture,
        reobserved_volume_evidence=captured_volume,
        prepared_sentinel_evidence_id=(
            sentinel_evidence.runtime_volume_sentinel_evidence_id
        ),
        parent_mount_id=root_mount_id,
        parent_device=root_metadata.st_dev,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    workspace_metadata = (root_path / "workspace").stat(follow_symlinks=False)
    physical_input = _remint_contract(
        prepared.input_delivery_slot,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_result_directory = _remint_contract(
        prepared.result_directory,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_control_directory = _remint_contract(
        prepared.control_directory,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_temporary_directory = _remint_contract(
        prepared.temporary_directory,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_result_file = _remint_contract(
        prepared.result_file,
        prepared_parent_directory_id=(
            physical_result_directory.prepared_runtime_directory_id
        ),
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_workspace = _remint_contract(
        prepared.workspace_proof,
        owner_user_id=workspace_metadata.st_uid,
        owner_group_id=workspace_metadata.st_gid,
        root_mode=stat.S_IMODE(workspace_metadata.st_mode),
        mount_id=root_mount_id,
        device=workspace_metadata.st_dev,
        inode=workspace_metadata.st_ino,
    )
    physical_prepared_volume = _remint_contract(
        prepared.runtime_volume_evidence,
        root_mount_id=root_mount_id,
        root_device=root_metadata.st_dev,
        root_inode=root_metadata.st_ino,
        sentinel_evidence=sentinel_evidence,
    )
    physical_layout = _remint_contract(
        prepared.layout_proof,
        runtime_volume_evidence_id=(
            physical_prepared_volume.runtime_volume_evidence_id
        ),
        prepared_delivery_slot_ids=(physical_input.prepared_delivery_slot_id,),
        prepared_runtime_directory_ids=tuple(
            sorted(
                (
                    physical_control_directory.prepared_runtime_directory_id,
                    physical_result_directory.prepared_runtime_directory_id,
                    physical_temporary_directory.prepared_runtime_directory_id,
                )
            )
        ),
        prepared_result_file_id=physical_result_file.prepared_file_id,
        prepared_workspace_proof_id=(physical_workspace.prepared_workspace_proof_id),
    )
    physical_prepared = _remint_contract(
        prepared,
        runtime_volume_evidence=physical_prepared_volume,
        input_delivery_slot=physical_input,
        result_directory=physical_result_directory,
        control_directory=physical_control_directory,
        temporary_directory=physical_temporary_directory,
        result_file=physical_result_file,
        workspace_proof=physical_workspace,
        layout_proof=physical_layout,
    )
    physical_capture = _remint_contract(
        exact_capture,
        prepared_parent_authority_id=(
            physical_result_directory.prepared_runtime_directory_id
        ),
        prepared_file_id=physical_result_file.prepared_file_id,
    )

    opened_roots = []

    def open_test_volume(descriptors, keeper):
        root = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        opened_roots.append(root)
        descriptors.callback(os.close, root)
        return volume_module._MountedRuntimeVolumeLease(
            process_descriptor=root,
            root_descriptor=root,
            keeper_container_id=keeper.container_id,
            keeper_process_id=keeper.process_id,
            process_start_time_ticks=keeper.process_start_time_ticks,
            process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
            root_mount_id=root_mount_id,
            root_device=root_metadata.st_dev,
            root_inode=root_metadata.st_ino,
            process_snapshot_size_limit_bytes=(
                keeper.issued_create_projection.execution_policy.supervisor_limits.process_snapshot_size_bytes
            ),
        )

    monkeypatch.setattr(
        volume_module,
        "_open_mounted_runtime_volume",
        open_test_volume,
    )
    monkeypatch.setattr(
        volume_module,
        "_require_same_mounted_runtime_volume",
        lambda _mounted, _keeper: None,
    )
    substituted_capture_parent = _remint_contract(
        physical_capture,
        parent_inode=physical_capture.parent_inode + 1,
    )
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="differs from prepared result capture",
    ):
        volume_module.open_run_action_result_workspace(
            physical_prepared,
            substituted_capture_parent,
        )
    assert opened_roots == []
    substituted_workspace = _remint_contract(
        physical_prepared.workspace_proof,
        inode=max(
            (
                physical_prepared.runtime_volume_evidence.root_inode,
                physical_prepared.runtime_volume_evidence.sentinel_evidence.inode,
                physical_prepared.input_delivery_slot.inode,
                physical_prepared.result_directory.inode,
                physical_prepared.control_directory.inode,
                physical_prepared.temporary_directory.inode,
                physical_prepared.result_file.inode,
                physical_prepared.workspace_proof.inode,
            )
        )
        + 1,
    )
    substituted_layout = _remint_contract(
        physical_prepared.layout_proof,
        prepared_workspace_proof_id=(substituted_workspace.prepared_workspace_proof_id),
    )
    substituted_prepared = _remint_contract(
        physical_prepared,
        workspace_proof=substituted_workspace,
        layout_proof=substituted_layout,
    )
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="workspace differs from prepared proof",
    ):
        volume_module.open_run_action_result_workspace(
            substituted_prepared,
            physical_capture,
        )
    with pytest.raises(OSError):
        os.fstat(opened_roots[-1])
    retained_sentinel_path = root_path / ".kapso-generation.retained"
    os.link(sentinel_path, retained_sentinel_path)
    sentinel_path.unlink()
    sentinel_path.write_bytes(sentinel_payload)
    sentinel_path.chmod(0o400)
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="sentinel was substituted",
    ):
        volume_module.open_run_action_result_workspace(
            physical_prepared,
            physical_capture,
        )
    with pytest.raises(OSError):
        os.fstat(opened_roots[-1])
    sentinel_path.unlink()
    os.link(retained_sentinel_path, sentinel_path)
    retained_sentinel_path.unlink()

    original_require_current = (
        volume_module.RunActionResultWorkspaceLease.require_current
    )

    def reject_initial_lease_validation(_lease):
        raise RuntimeError("injected lease construction race")

    monkeypatch.setattr(
        volume_module.RunActionResultWorkspaceLease,
        "require_current",
        reject_initial_lease_validation,
    )
    with pytest.raises(RuntimeError, match="lease construction race"):
        volume_module.open_run_action_result_workspace(
            physical_prepared,
            physical_capture,
        )
    with pytest.raises(OSError):
        os.fstat(opened_roots[-1])
    monkeypatch.setattr(
        volume_module.RunActionResultWorkspaceLease,
        "require_current",
        original_require_current,
    )

    lease = volume_module.open_run_action_result_workspace(
        physical_prepared,
        physical_capture,
    )
    sentinel_path.unlink()
    sentinel_path.write_bytes(sentinel_payload)
    sentinel_path.chmod(0o400)
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="changed during exact observation",
    ):
        lease.require_current()
    lease.close()


def _physical_barrier_control_case(tmp_path, settings):
    policy = _policy(
        settings.docker,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))
    authority = prepared.runtime_volume_authority
    root_path = tmp_path / "barrier-control-volume"
    root_path.mkdir(mode=0o700)
    control_path = root_path / "control"
    control_path.mkdir(mode=0o700)
    sentinel_payload = authority.generation_nonce.encode("ascii")
    sentinel_path = root_path / ".kapso-generation"
    sentinel_path.write_bytes(sentinel_payload)
    sentinel_path.chmod(0o400)
    with ExitStack() as descriptors:
        root_descriptor = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, root_descriptor)
        root_metadata = os.fstat(root_descriptor)
        root_mount_id = read_run_action_descriptor_mount_id(root_descriptor)
        control_metadata = control_path.stat(follow_symlinks=False)
        sentinel_metadata = sentinel_path.stat(follow_symlinks=False)
    sentinel_evidence = RunActionRuntimeVolumeSentinelEvidence.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=".kapso-generation",
        file_type="regular",
        owner_user_id=sentinel_metadata.st_uid,
        owner_group_id=sentinel_metadata.st_gid,
        mode=stat.S_IMODE(sentinel_metadata.st_mode),
        link_count=sentinel_metadata.st_nlink,
        size_bytes=sentinel_metadata.st_size,
        content_digest=volume_module.tree_or_blob_digest(sentinel_payload),
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
        inode=sentinel_metadata.st_ino,
    )
    physical_volume = _remint_contract(
        prepared.runtime_volume_evidence,
        root_mount_id=root_mount_id,
        root_device=root_metadata.st_dev,
        root_inode=root_metadata.st_ino,
        sentinel_evidence=sentinel_evidence,
    )
    physical_input = _remint_contract(
        prepared.input_delivery_slot,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_control = _remint_contract(
        prepared.control_directory,
        owner_user_id=control_metadata.st_uid,
        owner_group_id=control_metadata.st_gid,
        mode=stat.S_IMODE(control_metadata.st_mode),
        mount_id=root_mount_id,
        device=control_metadata.st_dev,
        inode=control_metadata.st_ino,
    )
    physical_result_directory = _remint_contract(
        prepared.result_directory,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_temporary = _remint_contract(
        prepared.temporary_directory,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_result_file = _remint_contract(
        prepared.result_file,
        prepared_parent_directory_id=(
            physical_result_directory.prepared_runtime_directory_id
        ),
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
    )
    physical_layout = _remint_contract(
        prepared.layout_proof,
        runtime_volume_evidence_id=physical_volume.runtime_volume_evidence_id,
        prepared_delivery_slot_ids=(physical_input.prepared_delivery_slot_id,),
        prepared_runtime_directory_ids=tuple(
            sorted(
                (
                    physical_control.prepared_runtime_directory_id,
                    physical_result_directory.prepared_runtime_directory_id,
                    physical_temporary.prepared_runtime_directory_id,
                )
            )
        ),
        prepared_result_file_id=physical_result_file.prepared_file_id,
    )
    physical_prepared = _remint_contract(
        prepared,
        runtime_volume_evidence=physical_volume,
        input_delivery_slot=physical_input,
        control_directory=physical_control,
        result_directory=physical_result_directory,
        temporary_directory=physical_temporary,
        result_file=physical_result_file,
        layout_proof=physical_layout,
    )
    return physical_prepared, root_path, root_mount_id, root_metadata


def _physical_result_capture_case(
    tmp_path,
    settings,
    payload,
    *,
    result_size_limit=None,
):
    launch_settings = settings.launch
    policy = _policy(
        settings.docker,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    if result_size_limit is not None:
        launch_settings = replace(
            launch_settings,
            run_action_result_size_bytes=result_size_limit,
        )
        policy = _remint_contract(
            policy,
            supervisor_limits=_remint_contract(
                policy.supervisor_limits,
                result_size_bytes=result_size_limit,
            ),
        )
    prepared = _prepared_execution(claim=_claim(policy=policy))
    authority = prepared.runtime_volume_authority
    volume = observe_runtime_volume(
        _volume_raw(authority, settings.docker),
        prepared.preparation_claim,
        authority,
        settings.docker,
    )
    root_path = tmp_path / "captured-result"
    root_path.mkdir(mode=0o700)
    directory_paths = {
        name: root_path / name for name in ("control", "input", "result", "temporary")
    }
    for path in directory_paths.values():
        path.mkdir(mode=0o700)
    sentinel_payload = authority.generation_nonce.encode("ascii")
    sentinel_path = root_path / ".kapso-generation"
    sentinel_path.write_bytes(sentinel_payload)
    sentinel_path.chmod(0o400)
    result_path = directory_paths["result"] / "result.blob"
    result_path.write_bytes(payload)
    result_path.chmod(0o600)
    with ExitStack() as descriptors:
        root_descriptor = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, root_descriptor)
        root_metadata = os.fstat(root_descriptor)
        root_mount_id = read_run_action_descriptor_mount_id(root_descriptor)
    directory_metadata = {
        name: path.stat(follow_symlinks=False) for name, path in directory_paths.items()
    }
    sentinel_metadata = sentinel_path.stat(follow_symlinks=False)
    result_metadata = result_path.stat(follow_symlinks=False)
    sentinel_evidence = RunActionRuntimeVolumeSentinelEvidence.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=".kapso-generation",
        file_type="regular",
        owner_user_id=sentinel_metadata.st_uid,
        owner_group_id=sentinel_metadata.st_gid,
        mode=stat.S_IMODE(sentinel_metadata.st_mode),
        link_count=sentinel_metadata.st_nlink,
        size_bytes=sentinel_metadata.st_size,
        content_digest=volume_module.tree_or_blob_digest(sentinel_payload),
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
        inode=sentinel_metadata.st_ino,
    )
    physical_volume = _remint_contract(
        prepared.runtime_volume_evidence,
        docker_volume_occurrence_digest=volume.volume_occurrence_digest,
        root_mount_id=root_mount_id,
        root_device=root_metadata.st_dev,
        root_inode=root_metadata.st_ino,
        sentinel_evidence=sentinel_evidence,
    )
    physical_input = _remint_contract(
        prepared.input_delivery_slot,
        owner_user_id=directory_metadata["input"].st_uid,
        owner_group_id=directory_metadata["input"].st_gid,
        mode=stat.S_IMODE(directory_metadata["input"].st_mode),
        mount_id=root_mount_id,
        device=directory_metadata["input"].st_dev,
        inode=directory_metadata["input"].st_ino,
    )
    physical_directories = {
        name: _remint_contract(
            directory,
            owner_user_id=directory_metadata[name].st_uid,
            owner_group_id=directory_metadata[name].st_gid,
            mode=stat.S_IMODE(directory_metadata[name].st_mode),
            mount_id=root_mount_id,
            device=directory_metadata[name].st_dev,
            inode=directory_metadata[name].st_ino,
        )
        for name, directory in (
            ("control", prepared.control_directory),
            ("result", prepared.result_directory),
            ("temporary", prepared.temporary_directory),
        )
    }
    physical_result_file = _remint_contract(
        prepared.result_file,
        prepared_parent_directory_id=(
            physical_directories["result"].prepared_runtime_directory_id
        ),
        owner_user_id=result_metadata.st_uid,
        owner_group_id=result_metadata.st_gid,
        mode=stat.S_IMODE(result_metadata.st_mode),
        mount_id=root_mount_id,
        device=result_metadata.st_dev,
        inode=result_metadata.st_ino,
    )
    physical_layout = _remint_contract(
        prepared.layout_proof,
        runtime_volume_evidence_id=physical_volume.runtime_volume_evidence_id,
        prepared_delivery_slot_ids=(physical_input.prepared_delivery_slot_id,),
        prepared_runtime_directory_ids=tuple(
            sorted(
                directory.prepared_runtime_directory_id
                for directory in physical_directories.values()
            )
        ),
        prepared_result_file_id=physical_result_file.prepared_file_id,
    )
    physical_prepared = _remint_contract(
        prepared,
        runtime_volume_evidence=physical_volume,
        input_delivery_slot=physical_input,
        control_directory=physical_directories["control"],
        result_directory=physical_directories["result"],
        temporary_directory=physical_directories["temporary"],
        result_file=physical_result_file,
        layout_proof=physical_layout,
    )
    spawn = _spawn_commit(physical_prepared)
    terminal = _terminal_observation(physical_prepared, spawn)
    return (
        physical_prepared,
        terminal,
        volume,
        launch_settings,
        root_path,
        result_path,
        sentinel_path,
        root_mount_id,
        root_metadata,
    )


def _patch_physical_result_capture(
    monkeypatch,
    prepared,
    root_path,
    root_mount_id,
    root_metadata,
    payload_size,
):
    def open_test_volume(descriptors, keeper):
        root_descriptor = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, root_descriptor)
        return volume_module._MountedRuntimeVolumeLease(
            process_descriptor=root_descriptor,
            root_descriptor=root_descriptor,
            keeper_container_id=keeper.container_id,
            keeper_process_id=keeper.process_id,
            process_start_time_ticks=keeper.process_start_time_ticks,
            process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
            root_mount_id=root_mount_id,
            root_device=root_metadata.st_dev,
            root_inode=root_metadata.st_ino,
            process_snapshot_size_limit_bytes=(
                keeper.issued_create_projection.execution_policy.supervisor_limits.process_snapshot_size_bytes
            ),
        )

    prepared_volume = prepared.runtime_volume_evidence
    block_size = prepared_volume.allocation_block_size_bytes
    added_block_count = (payload_size + block_size - 1) // block_size
    used_block_count = prepared_volume.used_block_count + added_block_count
    available_block_count = prepared_volume.effective_block_count - used_block_count
    filesystem = os.statvfs_result(
        (
            block_size,
            block_size,
            prepared_volume.effective_block_count,
            available_block_count,
            available_block_count,
            prepared_volume.effective_inode_limit,
            prepared_volume.available_inode_count,
            prepared_volume.available_inode_count,
            0,
            255,
        )
    )
    monkeypatch.setattr(
        volume_module,
        "_open_mounted_runtime_volume",
        open_test_volume,
    )
    monkeypatch.setattr(
        volume_module,
        "_require_same_mounted_runtime_volume",
        lambda _mounted, _keeper: None,
    )
    monkeypatch.setattr(
        volume_module.os,
        "fstatvfs",
        lambda _descriptor: filesystem,
    )


def test_descriptor_result_capture_reads_only_the_original_bounded_inode(
    layout_context,
    tmp_path,
    monkeypatch,
):
    settings, _claim_without_workspace, _authority, _empty = layout_context
    payload = b'{"result":"descriptor-bound"}'
    (
        prepared,
        terminal,
        volume,
        launch_settings,
        root_path,
        _result_path,
        _sentinel_path,
        root_mount_id,
        root_metadata,
    ) = _physical_result_capture_case(tmp_path, settings, payload)
    _patch_physical_result_capture(
        monkeypatch,
        prepared,
        root_path,
        root_mount_id,
        root_metadata,
        len(payload),
    )

    receipt, captured_payload = volume_module.capture_run_action_result_file(
        prepared,
        terminal,
        volume,
        settings=launch_settings,
    )

    expected_added_blocks = (
        len(payload)
        + receipt.reobserved_volume_evidence.allocation_block_size_bytes
        - 1
    ) // receipt.reobserved_volume_evidence.allocation_block_size_bytes
    assert captured_payload == payload
    assert receipt.terminal_observation_id == terminal.terminal_observation_id
    assert receipt.prepared_parent_authority_id == (
        prepared.result_directory.prepared_runtime_directory_id
    )
    assert receipt.prepared_file_id == prepared.result_file.prepared_file_id
    assert receipt.inode == prepared.result_file.inode
    assert receipt.content_digest == volume_module.tree_or_blob_digest(payload)
    assert receipt.reobserved_volume_evidence.used_block_count == (
        prepared.runtime_volume_evidence.used_block_count + expected_added_blocks
    )


def test_descriptor_result_capture_preserves_an_exact_empty_original_inode(
    layout_context,
    tmp_path,
    monkeypatch,
):
    settings, _claim_without_workspace, _authority, _empty = layout_context
    payload = b""
    (
        prepared,
        terminal,
        volume,
        launch_settings,
        root_path,
        _result_path,
        _sentinel_path,
        root_mount_id,
        root_metadata,
    ) = _physical_result_capture_case(tmp_path, settings, payload)
    _patch_physical_result_capture(
        monkeypatch,
        prepared,
        root_path,
        root_mount_id,
        root_metadata,
        len(payload),
    )

    receipt, captured_payload = volume_module.capture_run_action_result_file(
        prepared,
        terminal,
        volume,
        settings=launch_settings,
    )

    assert captured_payload == b""
    assert receipt.size_bytes == 0
    assert receipt.content_digest == volume_module.tree_or_blob_digest(b"")
    assert receipt.inode == prepared.result_file.inode


@pytest.mark.parametrize(
    "mutation",
    (
        "replacement",
        "hard_link",
        "extra_result_entry",
        "mode",
        "sentinel",
        "fifo",
    ),
)
def test_descriptor_result_capture_rejects_unsafe_physical_files(
    layout_context,
    tmp_path,
    monkeypatch,
    mutation,
):
    settings, _claim_without_workspace, _authority, _empty = layout_context
    payload = b'{"result":"complete"}'
    (
        prepared,
        terminal,
        volume,
        launch_settings,
        root_path,
        result_path,
        sentinel_path,
        root_mount_id,
        root_metadata,
    ) = _physical_result_capture_case(tmp_path, settings, payload)
    if mutation == "replacement":
        os.link(result_path, root_path.parent / "detached-original-result")
        result_path.unlink()
        result_path.write_bytes(payload)
        result_path.chmod(0o600)
    elif mutation == "hard_link":
        os.link(result_path, root_path.parent / "detached-result-link")
    elif mutation == "extra_result_entry":
        (result_path.parent / "unexpected").write_bytes(b"unexpected")
    elif mutation == "mode":
        result_path.chmod(0o400)
    elif mutation == "sentinel":
        sentinel_payload = prepared.runtime_volume_authority.generation_nonce.encode(
            "ascii"
        )
        os.link(sentinel_path, root_path.parent / "detached-original-sentinel")
        sentinel_path.unlink()
        sentinel_path.write_bytes(sentinel_payload)
        sentinel_path.chmod(0o400)
    else:
        result_path.unlink()
        os.mkfifo(result_path, mode=0o600)
    observed_size = result_path.stat(follow_symlinks=False).st_size
    _patch_physical_result_capture(
        monkeypatch,
        prepared,
        root_path,
        root_mount_id,
        root_metadata,
        observed_size,
    )

    with pytest.raises(RunActionRuntimeVolumeError):
        volume_module.capture_run_action_result_file(
            prepared,
            terminal,
            volume,
            settings=launch_settings,
        )


def test_descriptor_result_capture_rejects_configured_limit_and_mid_read_change(
    layout_context,
    tmp_path,
    monkeypatch,
):
    settings, _claim_without_workspace, _authority, _empty = layout_context
    result_size_limit = 32
    payload = b"x" * (result_size_limit + 1)
    (
        prepared,
        terminal,
        volume,
        launch_settings,
        root_path,
        result_path,
        _sentinel_path,
        root_mount_id,
        root_metadata,
    ) = _physical_result_capture_case(
        tmp_path,
        settings,
        payload,
        result_size_limit=result_size_limit,
    )
    _patch_physical_result_capture(
        monkeypatch,
        prepared,
        root_path,
        root_mount_id,
        root_metadata,
        len(payload),
    )
    with pytest.raises(RunActionRuntimeVolumeError, match="oversized"):
        volume_module.capture_run_action_result_file(
            prepared,
            terminal,
            volume,
            settings=launch_settings,
        )

    result_path.write_bytes(b"stable")
    result_path.chmod(0o600)
    physical_metadata = result_path.stat(follow_symlinks=False)
    prepared_result = _remint_contract(
        prepared.result_file,
        inode=physical_metadata.st_ino,
    )
    prepared_layout = _remint_contract(
        prepared.layout_proof,
        prepared_result_file_id=prepared_result.prepared_file_id,
    )
    prepared = _remint_contract(
        prepared,
        result_file=prepared_result,
        layout_proof=prepared_layout,
    )
    spawn = _spawn_commit(prepared)
    terminal = _terminal_observation(prepared, spawn)
    _patch_physical_result_capture(
        monkeypatch,
        prepared,
        root_path,
        root_mount_id,
        root_metadata,
        len(b"stable"),
    )
    original_read = volume_module._read_bounded_descriptor_payload
    result_read_count = 0

    def mutate_after_first_result_read(descriptor, limit):
        nonlocal result_read_count
        observed = original_read(descriptor, limit)
        if observed == b"stable":
            result_read_count += 1
            if result_read_count == 1:
                result_path.write_bytes(b"mutated")
                result_path.chmod(0o600)
        return observed

    monkeypatch.setattr(
        volume_module,
        "_read_bounded_descriptor_payload",
        mutate_after_first_result_read,
    )
    with pytest.raises(RunActionRuntimeVolumeError, match="changed"):
        volume_module.capture_run_action_result_file(
            prepared,
            terminal,
            volume,
            settings=launch_settings,
        )


@pytest.fixture(scope="module")
def layout_context():
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    policy = _policy(
        settings.docker,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    claim = _claim(policy=policy)
    authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
    volume = observe_runtime_volume(
        _volume_raw(authority, settings.docker),
        claim,
        authority,
        settings.docker,
    )
    block_size = 4096
    effective_block_count = authority.size_limit_bytes // block_size
    empty = DockerRunActionEmptyVolumeObservation(
        runtime_volume_authority=authority,
        docker_volume_observation=volume,
        keeper_container_id="a" * 64,
        keeper_process_id=101,
        keeper_process_start_time_ticks=123456,
        process_cgroup_path=(f"/test.kapso.run_action.slice/docker-{'a' * 64}.scope"),
        mount_id=1232,
        device=os.makedev(0, 73),
        root_inode=71,
        filesystem_type="tmpfs",
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        root_mode=authority.root_mode,
        allocation_block_size_bytes=block_size,
        effective_block_count=effective_block_count,
        effective_size_bytes=authority.size_limit_bytes,
        effective_inode_limit=authority.inode_limit,
        used_block_count=0,
        used_size_bytes=0,
        used_inode_count=1,
        available_block_count=effective_block_count,
        available_size_bytes=authority.size_limit_bytes,
        available_inode_count=authority.inode_limit - 1,
        empty_entry_count=0,
        empty_size_bytes=0,
    )
    return settings, claim, authority, empty


def _open_empty_root(path, descriptors):
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, descriptor)
    return descriptor


def _physical_prepared_adoption_case(
    layout_context,
    tmp_path,
    monkeypatch,
    *,
    credential_mode=RunActionCredentialMode.NONE,
):
    settings, _claim_fixture, _authority_fixture, empty_fixture = layout_context
    docker_settings = replace(
        settings.docker,
        runtime_executable_digest=tree_or_blob_digest(_TEST_DOCKER_BYTES),
    )
    policy = _policy(
        docker_settings,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=credential_mode,
    )
    claim = _claim(policy=policy)
    authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
    volume = observe_runtime_volume(
        _volume_raw(authority, docker_settings),
        claim,
        authority,
        docker_settings,
    )
    empty = replace(
        empty_fixture,
        runtime_volume_authority=authority,
        docker_volume_observation=volume,
    )
    root_path = tmp_path / "runtime-volume"
    root_path.mkdir(mode=0o700)
    root_path.chmod(0o700)
    root_descriptor = os.open(
        root_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    root_metadata = os.fstat(root_descriptor)
    root_mount_id = read_run_action_descriptor_mount_id(root_descriptor)
    os.close(root_descriptor)
    contract_fixture = _prepared_execution(
        claim=claim,
        authority=authority,
        container_id="a" * 64,
    )
    keeper = contract_fixture.volume_keeper_evidence
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=claim,
        runtime_volume_authority=authority,
    )
    docker_runtime_root = tmp_path / "docker-runtime"
    docker_runtime_root.mkdir(mode=0o700)
    monkeypatch.setattr(
        docker_runtime_module,
        "read_verified_root_executable",
        lambda _path, _digest: _TEST_DOCKER_BYTES,
    )
    monkeypatch.setattr(
        docker_runtime_module,
        "_require_runtime_socket",
        lambda _path: None,
    )
    docker_runner = _InventoryDockerRunner(docker_settings)
    docker_runtime = PinnedDockerRuntime(
        trusted_root=docker_runtime_root.resolve(),
        settings=docker_settings,
        process_runner=docker_runner,
    )
    docker_runner.runtime = docker_runtime
    resource_manager = DockerRunActionResourceManager(docker_runtime)
    docker_runner.volumes[authority.volume_name] = _volume_raw(
        authority,
        docker_settings,
    )
    docker_runner.containers[keeper.container_id] = {
        "Config": {
            "Labels": {
                label.key: label.value
                for label in preparation_keeper_container_labels(claim)
            }
        },
        "Id": keeper.container_id,
        "Name": f"/{preparation_keeper_container_name(claim)}",
    }
    physical_empty = replace(
        empty,
        keeper_container_id=keeper.container_id,
        keeper_process_id=keeper.process_id,
        keeper_process_start_time_ticks=keeper.process_start_time_ticks,
        process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
        mount_id=root_mount_id,
        device=root_metadata.st_dev,
        root_inode=root_metadata.st_ino,
    )
    prepared_capacity = contract_fixture.runtime_volume_evidence

    def open_test_volume(descriptors, observed_keeper):
        assert observed_keeper == keeper
        opened_root = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, opened_root)
        return volume_module._MountedRuntimeVolumeLease(
            process_descriptor=opened_root,
            root_descriptor=opened_root,
            keeper_container_id=keeper.container_id,
            keeper_process_id=keeper.process_id,
            process_start_time_ticks=keeper.process_start_time_ticks,
            process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
            root_mount_id=root_mount_id,
            root_device=root_metadata.st_dev,
            root_inode=root_metadata.st_ino,
            process_snapshot_size_limit_bytes=(
                keeper.issued_create_projection.execution_policy.supervisor_limits.process_snapshot_size_bytes
            ),
        )

    def observe_filesystem(_descriptor):
        if tuple(root_path.iterdir()):
            return os.statvfs_result(
                (
                    prepared_capacity.allocation_block_size_bytes,
                    prepared_capacity.allocation_block_size_bytes,
                    prepared_capacity.effective_block_count,
                    prepared_capacity.available_block_count,
                    prepared_capacity.available_block_count,
                    prepared_capacity.effective_inode_limit,
                    prepared_capacity.available_inode_count,
                    prepared_capacity.available_inode_count,
                    0,
                    255,
                )
            )
        return os.statvfs_result(
            (
                physical_empty.allocation_block_size_bytes,
                physical_empty.allocation_block_size_bytes,
                physical_empty.effective_block_count,
                physical_empty.available_block_count,
                physical_empty.available_block_count,
                physical_empty.effective_inode_limit,
                physical_empty.available_inode_count,
                physical_empty.available_inode_count,
                0,
                255,
            )
        )

    monkeypatch.setattr(
        volume_module,
        "_open_mounted_runtime_volume",
        open_test_volume,
    )
    monkeypatch.setattr(
        volume_module,
        "_require_same_mounted_runtime_volume",
        lambda _mounted, _keeper: None,
    )
    monkeypatch.setattr(
        volume_module,
        "_require_mount_authority",
        lambda _mount, _metadata, _authority: None,
    )
    monkeypatch.setattr(
        volume_module,
        "_read_mount_info",
        lambda _process, _mount_id, destination, _byte_limit: volume_module._MountInfo(
            mount_id=root_mount_id,
            parent_mount_id=root_mount_id,
            device_major=os.major(root_metadata.st_dev),
            device_minor=os.minor(root_metadata.st_dev),
            mount_point=destination,
            mount_options=("nodev", "nosuid", "relatime", "rw"),
            optional_fields=(),
            filesystem_type="tmpfs",
            source="tmpfs",
            super_options=(
                "inode64",
                "noswap",
                "rw",
                f"gid={authority.owner_group_id}",
                f"mode={authority.root_mode:o}",
                f"nr_inodes={authority.inode_limit}",
                f"size={authority.size_limit_bytes}",
                f"uid={authority.owner_user_id}",
            ),
        ),
    )
    monkeypatch.setattr(volume_module.os, "fstatvfs", observe_filesystem)
    prepared = volume_module.materialize_runtime_volume_layout(
        claim,
        physical_empty,
        keeper,
        workspace_descriptor=None,
        settings=settings.launch,
    )
    return (
        allocation,
        resource_manager,
        keeper,
        prepared,
        root_path,
        docker_runner,
    )


def _replace_raw_container_identity(value, old_container_id, new_container_id):
    if type(value) is dict:
        return {
            key: _replace_raw_container_identity(
                item,
                old_container_id,
                new_container_id,
            )
            for key, item in value.items()
        }
    if type(value) is list:
        return [
            _replace_raw_container_identity(
                item,
                old_container_id,
                new_container_id,
            )
            for item in value
        ]
    if type(value) is str:
        return value.replace(old_container_id, new_container_id)
    return value


def _physical_selected_activation_case(
    layout_context,
    tmp_path,
    monkeypatch,
    *,
    credential_mode=RunActionCredentialMode.NONE,
):
    (
        allocation,
        resource_manager,
        keeper,
        prepared_volume,
        root_path,
        docker_runner,
    ) = _physical_prepared_adoption_case(
        layout_context,
        tmp_path,
        monkeypatch,
        credential_mode=credential_mode,
    )
    docker_settings = resource_manager.runtime_settings
    inert_container_evidence = _prepared_execution(
        claim=allocation.preparation_claim,
        authority=allocation.runtime_volume_authority,
        container_id="a" * 64,
    ).inert_container_evidence
    prepared = RunActionPreparedExecution.mint(
        preparation_claim=prepared_volume.preparation_claim,
        runtime_volume_authority=allocation.runtime_volume_authority,
        runtime_volume_evidence=prepared_volume.runtime_volume_evidence,
        volume_keeper_evidence=keeper,
        input_delivery_slot=prepared_volume.input_delivery_slot,
        result_directory=prepared_volume.result_directory,
        temporary_directory=prepared_volume.temporary_directory,
        control_directory=prepared_volume.control_directory,
        result_file=prepared_volume.result_file,
        credential_delivery_slot=prepared_volume.credential_delivery_slot,
        workspace_proof=prepared_volume.workspace_proof,
        layout_proof=prepared_volume.layout_proof,
        inert_container_evidence=inert_container_evidence,
    )
    projection = prepared.inert_container_evidence.issued_create_projection
    command = target_command_from_main_projection(projection)
    volume = observe_runtime_volume(
        _volume_raw(allocation.runtime_volume_authority, docker_settings),
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        docker_settings,
    )
    raw_keeper = _container_raw(
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        volume,
        command,
        docker_settings,
        keeper=True,
    )
    raw_keeper = _replace_raw_container_identity(
        raw_keeper,
        "b" * 64,
        keeper.container_id,
    )
    raw_keeper["State"]["Pid"] = keeper.process_id
    assert (
        raw_keeper["Id"],
        raw_keeper["Name"],
        raw_keeper["Config"]["Labels"],
    ) == (
        keeper.container_id,
        f"/{preparation_keeper_container_name(allocation.preparation_claim)}",
        {
            label.key: label.value
            for label in preparation_keeper_container_labels(
                allocation.preparation_claim
            )
        },
    )
    raw_main = _container_raw(
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        volume,
        command,
        docker_settings,
        keeper=False,
    )
    docker_runner.containers[keeper.container_id] = raw_keeper
    docker_runner.containers[prepared.inert_container_evidence.container_id] = raw_main
    monkeypatch.setattr(
        docker_inspect_module,
        "observe_mounted_keeper_helper",
        lambda _helper, *, container_id, process_id, process_snapshot_size_limit_bytes: (
            keeper.mounted_helper_evidence
            if (
                container_id == keeper.container_id
                and process_id == keeper.process_id
                and process_snapshot_size_limit_bytes > 0
            )
            else None
        ),
    )

    def observe_fixture_keeper(raw_inspection, *_arguments):
        assert raw_inspection["Id"] == keeper.container_id
        return keeper

    monkeypatch.setattr(
        volume_module,
        "observe_running_keeper",
        observe_fixture_keeper,
    )

    def observe_fixture_main(raw_inspection, *_arguments):
        assert raw_inspection["Id"] == inert_container_evidence.container_id
        if raw_inspection["State"]["Status"] != "created":
            raise RunActionRuntimeVolumeError("fixture main is no longer inert")
        return inert_container_evidence

    monkeypatch.setattr(
        volume_module,
        "observe_inert_main_container",
        observe_fixture_main,
    )
    assert (
        volume.volume_occurrence_digest
        == prepared.runtime_volume_evidence.docker_volume_occurrence_digest
    )
    payload = b"complete request"
    credential_payload = (
        b"provider-token"
        if credential_mode is RunActionCredentialMode.SUPERVISOR_FILE
        else None
    )
    prepared_volume = prepared.runtime_volume_evidence
    delivered_payloads = tuple(
        delivered_payload
        for delivered_payload in (payload, credential_payload)
        if delivered_payload is not None
    )
    delivered_block_count = sum(
        (len(delivered_payload) + prepared_volume.allocation_block_size_bytes - 1)
        // prepared_volume.allocation_block_size_bytes
        for delivered_payload in delivered_payloads
    )

    def observe_activated_filesystem(_descriptor):
        return os.statvfs_result(
            (
                prepared_volume.allocation_block_size_bytes,
                prepared_volume.allocation_block_size_bytes,
                prepared_volume.effective_block_count,
                prepared_volume.available_block_count - delivered_block_count,
                prepared_volume.available_block_count - delivered_block_count,
                prepared_volume.effective_inode_limit,
                prepared_volume.available_inode_count - len(delivered_payloads),
                prepared_volume.available_inode_count - len(delivered_payloads),
                0,
                255,
            )
        )

    monkeypatch.setattr(
        volume_module.os,
        "fstatvfs",
        observe_activated_filesystem,
    )
    spawn = _spawn_commit(prepared)
    activated = volume_module.deliver_and_reobserve_runtime_volume_activation(
        prepared,
        spawn,
        volume,
        keeper,
        request_payload=payload,
        credential_payload=credential_payload,
        credential_content_authority_id=(
            _CREDENTIAL_LEASE_AUTHORITY_ID if credential_payload is not None else None
        ),
        workspace_descriptor=None,
        settings=layout_context[0].launch,
    )
    selected = RunActionActivationRevalidationReceipt.mint(
        prepared_execution=prepared,
        spawn_commit=spawn,
        reobserved_volume_evidence=activated.reobserved_volume_evidence,
        reobserved_keeper_evidence=keeper,
        reobserved_container_evidence=prepared.inert_container_evidence,
        activated_workspace_observation=activated.activated_workspace_observation,
        activated_runtime_directory_observations=(
            activated.activated_runtime_directory_observations
        ),
        activated_sentinel_observation=(activated.activated_sentinel_observation),
        input_file_observation=activated.input_file_observation,
        result_file_observation=activated.result_file_observation,
        credential_file_observation=activated.credential_file_observation,
    )
    return (
        allocation,
        resource_manager,
        selected,
        root_path,
        docker_runner,
    )


def test_selected_activation_reopens_without_original_delivery_inputs(
    layout_context,
    tmp_path,
    monkeypatch,
):
    allocation, resource_manager, selected, root_path, _docker_runner = (
        _physical_selected_activation_case(
            layout_context,
            tmp_path,
            monkeypatch,
        )
    )
    snapshot = _runtime_layout_snapshot(root_path)

    with volume_module.open_selected_run_action_activation(
        allocation,
        selected,
        resource_manager,
        settings=layout_context[0].launch,
    ) as lease:
        assert lease.selected_receipt == selected
        assert lease.preparation_allocation == allocation
        assert (
            lease.inventory.main_container_id
            == selected.reobserved_container_evidence.container_id
        )
        lease.require_volume_current()

    assert _runtime_layout_snapshot(root_path) == snapshot


def test_selected_activation_lease_detects_same_inode_rewrite_after_reopen(
    layout_context,
    tmp_path,
    monkeypatch,
):
    allocation, resource_manager, selected, root_path, _docker_runner = (
        _physical_selected_activation_case(
            layout_context,
            tmp_path,
            monkeypatch,
        )
    )
    lease = volume_module.open_selected_run_action_activation(
        allocation,
        selected,
        resource_manager,
        settings=layout_context[0].launch,
    )
    input_path = root_path / "input" / "request.blob"
    input_path.chmod(0o600)
    input_path.write_bytes(b"changed request!")
    input_path.chmod(0o400)

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="changed during exact observation",
    ):
        lease.require_volume_current()

    lease.close()
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="closed or foreign",
    ):
        lease.require_current()


def test_selected_activation_reopens_credential_by_opaque_shape_without_reading_it(
    layout_context,
    tmp_path,
    monkeypatch,
):
    allocation, resource_manager, selected, root_path, _docker_runner = (
        _physical_selected_activation_case(
            layout_context,
            tmp_path,
            monkeypatch,
            credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
        )
    )
    credential_path = root_path / "credential" / "credentials"
    credential_path.chmod(0o600)
    credential_path.write_bytes(b"rotated-token!")
    credential_path.chmod(0o400)
    original_read = volume_module._read_bounded_descriptor_payload

    def reject_credential_read(descriptor, limit):
        descriptor_path = os.readlink(f"/proc/self/fd/{descriptor}")
        if descriptor_path.endswith("/credential/credentials"):
            raise AssertionError(
                "credential revalidation attempted to read secret bytes"
            )
        return original_read(descriptor, limit)

    monkeypatch.setattr(
        volume_module,
        "_read_bounded_descriptor_payload",
        reject_credential_read,
    )
    lease = volume_module.open_selected_run_action_activation(
        allocation,
        selected,
        resource_manager,
        settings=layout_context[0].launch,
    )
    assert selected.credential_file_observation is not None
    assert selected.credential_file_observation.content_digest is None
    credential_path.chmod(0o600)
    credential_path.write_bytes(b"another-token!")
    credential_path.chmod(0o400)

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="credential shape changed",
    ):
        lease.require_volume_current()

    lease.close()


def test_selected_activation_lease_survives_same_main_transition_to_running(
    layout_context,
    tmp_path,
    monkeypatch,
):
    allocation, resource_manager, selected, _root_path, docker_runner = (
        _physical_selected_activation_case(
            layout_context,
            tmp_path,
            monkeypatch,
        )
    )
    lease = volume_module.open_selected_run_action_activation(
        allocation,
        selected,
        resource_manager,
        settings=layout_context[0].launch,
    )
    main = docker_runner.containers[selected.reobserved_container_evidence.container_id]
    main["State"]["Status"] = "running"
    main["State"]["Running"] = True
    main["State"]["Pid"] = 4242
    main["State"]["StartedAt"] = "2026-07-25T00:00:01.123456789Z"

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="no longer inert",
    ):
        lease.require_current()

    lease.require_volume_current()
    lease.close()


def test_selected_activation_rejects_docker_change_after_descriptor_observation(
    layout_context,
    tmp_path,
    monkeypatch,
):
    allocation, resource_manager, selected, _root_path, docker_runner = (
        _physical_selected_activation_case(
            layout_context,
            tmp_path,
            monkeypatch,
        )
    )
    original_open = volume_module._open_selected_activation_descriptors

    def mutate_after_descriptor_observation(*arguments, **keywords):
        observation = original_open(*arguments, **keywords)
        docker_runner.volumes[allocation.runtime_volume_authority.volume_name][
            "CreatedAt"
        ] = "2026-07-25T00:00:01Z"
        return observation

    monkeypatch.setattr(
        volume_module,
        "_open_selected_activation_descriptors",
        mutate_after_descriptor_observation,
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="Docker occurrence changed during reopen",
    ):
        volume_module.open_selected_run_action_activation(
            allocation,
            selected,
            resource_manager,
            settings=layout_context[0].launch,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "input_content",
        "input_inode",
        "control_entry",
        "result_content",
        "temporary_entry",
        "main_occurrence",
        "volume_occurrence",
    ),
)
def test_selected_activation_rejects_changed_event_5_occurrence(
    layout_context,
    tmp_path,
    monkeypatch,
    mutation,
):
    allocation, resource_manager, selected, root_path, docker_runner = (
        _physical_selected_activation_case(
            layout_context,
            tmp_path,
            monkeypatch,
        )
    )
    if mutation == "input_content":
        input_path = root_path / "input" / "request.blob"
        input_path.chmod(0o600)
        input_path.write_bytes(b"changed request!")
        input_path.chmod(0o400)
    elif mutation == "input_inode":
        input_path = root_path / "input" / "request.blob"
        detached_input_path = root_path / "input" / "detached-request.blob"
        input_path.rename(detached_input_path)
        input_path.write_bytes(b"complete request")
        input_path.chmod(0o400)
        detached_input_path.unlink()
    elif mutation == "control_entry":
        (root_path / "control" / "release").write_bytes(b"release")
    elif mutation == "result_content":
        (root_path / "result" / "result.blob").write_bytes(b"unexpected")
    elif mutation == "temporary_entry":
        (root_path / "temporary" / "unexpected").write_bytes(b"unexpected")
    elif mutation == "main_occurrence":
        main = selected.reobserved_container_evidence.container_id
        docker_runner.containers["c" * 64] = docker_runner.containers.pop(main)
        docker_runner.containers["c" * 64]["Id"] = "c" * 64
    else:
        docker_runner.volumes[allocation.runtime_volume_authority.volume_name][
            "CreatedAt"
        ] = "2026-07-25T00:00:01Z"

    with pytest.raises(RunActionRuntimeVolumeError):
        volume_module.open_selected_run_action_activation(
            allocation,
            selected,
            resource_manager,
            settings=layout_context[0].launch,
        )


def _runtime_layout_snapshot(root_path):
    return tuple(
        (
            path.relative_to(root_path).as_posix(),
            stat.S_IFMT(path.stat(follow_symlinks=False).st_mode),
            stat.S_IMODE(path.stat(follow_symlinks=False).st_mode),
            None if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root_path.rglob("*"))
    )


def test_complete_prepared_layout_is_read_only_idempotently_adopted(
    layout_context,
    tmp_path,
    monkeypatch,
):
    (
        allocation,
        resource_manager,
        keeper,
        prepared,
        root_path,
        _docker_runner,
    ) = _physical_prepared_adoption_case(layout_context, tmp_path, monkeypatch)
    snapshot = _runtime_layout_snapshot(root_path)

    def reject_materialization(*_arguments, **_keywords):
        raise AssertionError("adoption attempted to materialize")

    monkeypatch.setattr(
        volume_module,
        "_materialize_layout_at_descriptor",
        reject_materialization,
    )
    first = adopt_prepared_runtime_volume_layout(
        allocation,
        resource_manager,
        keeper,
        settings=layout_context[0].launch,
    )
    second = adopt_prepared_runtime_volume_layout(
        allocation,
        resource_manager,
        keeper,
        settings=layout_context[0].launch,
    )

    assert first == prepared
    assert second == first
    assert _runtime_layout_snapshot(root_path) == snapshot


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_sentinel",
        "pending_sentinel",
        "extra_path",
        "wrong_sentinel",
    ),
)
def test_prepared_layout_adoption_rejects_partial_or_substituted_topology(
    layout_context,
    tmp_path,
    monkeypatch,
    mutation,
):
    (
        allocation,
        resource_manager,
        keeper,
        _prepared,
        root_path,
        _docker_runner,
    ) = _physical_prepared_adoption_case(layout_context, tmp_path, monkeypatch)
    sentinel_path = root_path / ".kapso-generation"
    if mutation == "missing_sentinel":
        sentinel_path.unlink()
    elif mutation == "pending_sentinel":
        sentinel_path.rename(
            root_path
            / f".kapso-generation.pending-{allocation.runtime_volume_authority.generation_nonce}"
        )
    elif mutation == "extra_path":
        (root_path / "unexpected").write_bytes(b"foreign")
    else:
        sentinel_path.chmod(0o600)
        sentinel_path.write_bytes(
            b"f" * len(allocation.runtime_volume_authority.generation_nonce)
        )
        sentinel_path.chmod(0o400)
    snapshot = _runtime_layout_snapshot(root_path)

    with pytest.raises(RunActionRuntimeVolumeError):
        adopt_prepared_runtime_volume_layout(
            allocation,
            resource_manager,
            keeper,
            settings=layout_context[0].launch,
        )

    assert _runtime_layout_snapshot(root_path) == snapshot


def test_prepared_layout_adoption_rejects_foreign_allocation(
    layout_context,
    tmp_path,
    monkeypatch,
):
    (
        allocation,
        resource_manager,
        keeper,
        _prepared,
        _root_path,
        _docker_runner,
    ) = _physical_prepared_adoption_case(layout_context, tmp_path, monkeypatch)
    foreign_allocation = RunActionPreparationAllocation.mint(
        preparation_claim=allocation.preparation_claim,
        runtime_volume_authority=_volume_authority(
            allocation.preparation_claim,
            nonce="8" * 32,
        ),
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="differs from its durable allocation",
    ):
        adopt_prepared_runtime_volume_layout(
            foreign_allocation,
            resource_manager,
            keeper,
            settings=layout_context[0].launch,
        )


def test_prepared_layout_adoption_rejects_docker_occurrence_change(
    layout_context,
    tmp_path,
    monkeypatch,
):
    (
        allocation,
        resource_manager,
        keeper,
        _prepared,
        root_path,
        docker_runner,
    ) = _physical_prepared_adoption_case(layout_context, tmp_path, monkeypatch)
    original_observe = volume_module._observe_prepared_layout_at_descriptor

    def mutate_docker_occurrence_after_physical_observation(*arguments, **keywords):
        observed = original_observe(*arguments, **keywords)
        docker_runner.volumes[allocation.runtime_volume_authority.volume_name][
            "CreatedAt"
        ] = "2026-07-25T00:00:01Z"
        return observed

    monkeypatch.setattr(
        volume_module,
        "_observe_prepared_layout_at_descriptor",
        mutate_docker_occurrence_after_physical_observation,
    )
    snapshot = _runtime_layout_snapshot(root_path)

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="Docker occurrence changed during adoption",
    ):
        adopt_prepared_runtime_volume_layout(
            allocation,
            resource_manager,
            keeper,
            settings=layout_context[0].launch,
        )

    assert _runtime_layout_snapshot(root_path) == snapshot


def test_descriptor_materializer_publishes_complete_layout_and_sentinel_last(
    layout_context,
    tmp_path,
):
    settings, claim, authority, empty = layout_context
    root_path = tmp_path / "runtime-volume"
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=None,
            settings=settings.launch,
        )

        workspace_frontier = _materialize_layout_at_descriptor(
            root_descriptor,
            claim=claim,
            authority=authority,
            plan=plan,
            workspace_descriptor=None,
            settings=settings.launch,
        )

    assert workspace_frontier is None
    assert tuple(sorted(path.name for path in root_path.iterdir())) == (
        ".kapso-generation",
        "control",
        "input",
        "result",
        "temporary",
    )
    assert (root_path / ".kapso-generation").read_bytes() == (
        authority.generation_nonce.encode("ascii")
    )
    assert stat.S_IMODE((root_path / ".kapso-generation").stat().st_mode) == 0o400
    assert tuple((root_path / "control").iterdir()) == ()
    assert tuple((root_path / "input").iterdir()) == ()
    assert (root_path / "result" / "result.blob").read_bytes() == b""
    assert tuple((root_path / "temporary").iterdir()) == ()
    assert all(
        stat.S_IMODE((root_path / name).stat().st_mode) == 0o700
        for name in ("control", "input", "result", "temporary")
    )
    assert stat.S_IMODE((root_path / "result" / "result.blob").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("failure_destination", "staging_present", "pending_present"),
    (
        ("control", True, False),
        ("input", True, False),
        ("result", True, False),
        ("temporary", True, False),
        (f".kapso-generation.pending-{_GENERATION_NONCE}", True, False),
        (".kapso-generation", False, True),
    ),
)
def test_descriptor_materializer_leaves_no_published_sentinel_before_final_rename(
    layout_context,
    tmp_path,
    monkeypatch,
    failure_destination,
    staging_present,
    pending_present,
):
    settings, claim, authority, empty = layout_context
    root_path = tmp_path / "runtime-volume"
    original_rename = volume_module._rename_no_replace

    def fail_final_sentinel(
        source_descriptor,
        source_name,
        destination_descriptor,
        destination_name,
    ):
        if destination_name == failure_destination:
            raise OSError("simulated publication crash")
        return original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(volume_module, "_rename_no_replace", fail_final_sentinel)
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=None,
            settings=settings.launch,
        )
        with pytest.raises(OSError, match="simulated publication crash"):
            _materialize_layout_at_descriptor(
                root_descriptor,
                claim=claim,
                authority=authority,
                plan=plan,
                workspace_descriptor=None,
                settings=settings.launch,
            )

    assert not (root_path / ".kapso-generation").exists()
    assert (
        root_path / f".kapso-generation.pending-{_GENERATION_NONCE}"
    ).exists() is pending_present
    assert (
        root_path / f".kapso-prepare-{_GENERATION_NONCE}"
    ).exists() is staging_present


def test_staging_removal_failure_cannot_publish_the_final_sentinel(
    layout_context,
    tmp_path,
    monkeypatch,
):
    settings, claim, authority, empty = layout_context
    root_path = tmp_path / "runtime-volume"
    original_rmdir = volume_module.os.rmdir

    def fail_staging_removal(path, *, dir_fd=None):
        if path == f".kapso-prepare-{_GENERATION_NONCE}":
            raise OSError("simulated staging-removal crash")
        return original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(volume_module.os, "rmdir", fail_staging_removal)
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=None,
            settings=settings.launch,
        )
        with pytest.raises(OSError, match="simulated staging-removal crash"):
            _materialize_layout_at_descriptor(
                root_descriptor,
                claim=claim,
                authority=authority,
                plan=plan,
                workspace_descriptor=None,
                settings=settings.launch,
            )

    assert not (root_path / ".kapso-generation").exists()
    assert (root_path / f".kapso-generation.pending-{_GENERATION_NONCE}").is_file()
    assert (root_path / f".kapso-prepare-{_GENERATION_NONCE}").is_dir()


@pytest.mark.parametrize("resource", ("bytes", "inodes"))
def test_layout_plan_requires_strict_peak_and_execution_headroom(
    layout_context,
    resource,
):
    settings, claim, _authority, empty = layout_context
    admitted = _plan_runtime_volume_layout(
        claim,
        empty,
        workspace_descriptor=None,
        settings=settings.launch,
    )
    limits = claim.execution_policy.docker_resource_limits
    if resource == "bytes":
        future_size_bytes = sum(
            volume_module._allocated_size_bytes(
                payload_size_limit_bytes,
                empty.allocation_block_size_bytes,
            )
            for payload_size_limit_bytes in (
                *(
                    slot_plan.payload_size_limit_bytes
                    for slot_plan in admitted.delivery_slot_plans
                ),
                admitted.result_file_plan.payload_size_limit_bytes,
                limits.runtime_temporary_reservation_size_bytes,
                claim.execution_policy.supervisor_limits.release_receipt_size_bytes,
                claim.execution_policy.supervisor_limits.timeout_directive_size_bytes,
            )
        )
        assert future_size_bytes == (
            volume_module._required_execution_headroom_size_bytes(
                claim,
                empty.allocation_block_size_bytes,
            )
        )
        exact_available_size = admitted.preparation_size_bytes + future_size_bytes
        exhausted = replace(
            empty,
            effective_block_count=(
                exact_available_size // empty.allocation_block_size_bytes
            ),
            effective_size_bytes=exact_available_size,
            available_block_count=(
                exact_available_size // empty.allocation_block_size_bytes
            ),
            available_size_bytes=exact_available_size,
        )
        admitted_boundary = replace(
            exhausted,
            effective_block_count=exhausted.effective_block_count + 1,
            effective_size_bytes=(
                exhausted.effective_size_bytes + empty.allocation_block_size_bytes
            ),
            available_block_count=exhausted.available_block_count + 1,
            available_size_bytes=(
                exhausted.available_size_bytes + empty.allocation_block_size_bytes
            ),
        )
    else:
        exact_available_inodes = (
            admitted.preparation_inode_count
            + len(admitted.delivery_slot_plans)
            + limits.runtime_temporary_reservation_inode_count
            + 2
        )
        exhausted = replace(
            empty,
            effective_inode_limit=exact_available_inodes + 1,
            available_inode_count=exact_available_inodes,
        )
        admitted_boundary = replace(
            exhausted,
            effective_inode_limit=exhausted.effective_inode_limit + 1,
            available_inode_count=exhausted.available_inode_count + 1,
        )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="lacks peak preparation and execution headroom",
    ):
        _plan_runtime_volume_layout(
            claim,
            exhausted,
            workspace_descriptor=None,
            settings=settings.launch,
        )
    assert (
        _plan_runtime_volume_layout(
            claim,
            admitted_boundary,
            workspace_descriptor=None,
            settings=settings.launch,
        ).preparation_inode_count
        == admitted.preparation_inode_count
    )


def test_timeout_directive_bound_reserves_exact_additional_volume_headroom(
    layout_context,
):
    _settings, claim, _authority, empty = layout_context
    block_size = empty.allocation_block_size_bytes
    limits = claim.execution_policy.supervisor_limits
    expanded_limits = _remint_contract(
        limits,
        timeout_directive_size_bytes=limits.timeout_directive_size_bytes + block_size,
    )
    expanded_policy = _remint_contract(
        claim.execution_policy,
        supervisor_limits=expanded_limits,
    )
    expanded_claim = _claim(policy=expanded_policy)

    assert volume_module._required_execution_headroom_size_bytes(
        expanded_claim,
        block_size,
    ) == (
        volume_module._required_execution_headroom_size_bytes(claim, block_size)
        + block_size
    )


def test_descriptor_materializer_rejects_nonempty_or_second_publication(
    layout_context,
    tmp_path,
):
    settings, claim, authority, empty = layout_context
    root_path = tmp_path / "runtime-volume"
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=None,
            settings=settings.launch,
        )
        (root_path / "substituted").write_bytes(b"foreign")
        with pytest.raises(RunActionRuntimeVolumeError, match="no longer empty"):
            _materialize_layout_at_descriptor(
                root_descriptor,
                claim=claim,
                authority=authority,
                plan=plan,
                workspace_descriptor=None,
                settings=settings.launch,
            )


def test_prepared_volume_aggregate_rejects_layout_splices():
    prepared = _prepared_execution()
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_delivery_slot=prepared.input_delivery_slot,
        result_directory=prepared.result_directory,
        control_directory=prepared.control_directory,
        result_file=prepared.result_file,
        temporary_directory=prepared.temporary_directory,
        credential_delivery_slot=prepared.credential_delivery_slot,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    substituted_layout = _prepared_execution(inode_offset=7).layout_proof

    assert observation.layout_proof.runtime_volume_evidence_id == (
        observation.runtime_volume_evidence.runtime_volume_evidence_id
    )
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(observation, layout_proof=substituted_layout)


@pytest.mark.parametrize(
    "mutation",
    (
        "directory_relative_paths",
        "logical_content_size_bytes",
        "logical_entry_count",
        "observed_used_size_bytes",
        "observed_used_inode_count",
    ),
)
def test_prepared_volume_aggregate_rejects_same_graph_layout_lies(mutation):
    prepared = _prepared_execution()
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_delivery_slot=prepared.input_delivery_slot,
        result_directory=prepared.result_directory,
        control_directory=prepared.control_directory,
        result_file=prepared.result_file,
        temporary_directory=prepared.temporary_directory,
        credential_delivery_slot=prepared.credential_delivery_slot,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    original_value = getattr(observation.layout_proof, mutation)
    substituted_value = (
        ("alien",) if mutation == "directory_relative_paths" else original_value + 1
    )
    substituted_layout = _remint_contract(
        observation.layout_proof,
        **{mutation: substituted_value},
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(observation, layout_proof=substituted_layout)


def test_prepared_volume_aggregate_rejects_claim_policy_authority_splice():
    prepared = _prepared_execution()
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_delivery_slot=prepared.input_delivery_slot,
        result_directory=prepared.result_directory,
        control_directory=prepared.control_directory,
        result_file=prepared.result_file,
        temporary_directory=prepared.temporary_directory,
        credential_delivery_slot=prepared.credential_delivery_slot,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    substituted_authority = _remint_contract(
        observation.runtime_volume_evidence.volume_authority,
        labels=tuple(
            RunActionContainerLabel(
                key=label.key,
                value=(
                    _fixture_content_id("run-action-reservation", "foreign")
                    if label.key == "com.kapso.run-action.reservation"
                    else label.value
                ),
            )
            for label in observation.runtime_volume_evidence.volume_authority.labels
        ),
    )
    substituted_sentinel = _remint_contract(
        observation.runtime_volume_evidence.sentinel_evidence,
        runtime_volume_authority_id=(substituted_authority.runtime_volume_authority_id),
    )
    substituted_evidence = _remint_contract(
        observation.runtime_volume_evidence,
        volume_authority=substituted_authority,
        observed_labels=substituted_authority.labels,
        sentinel_evidence=substituted_sentinel,
    )
    substituted_delivery_slots = tuple(
        _remint_contract(
            delivery_slot,
            runtime_volume_authority_id=(
                substituted_authority.runtime_volume_authority_id
            ),
        )
        for delivery_slot in (
            observation.input_delivery_slot,
            observation.credential_delivery_slot,
        )
        if delivery_slot is not None
    )
    substituted_runtime_directories = tuple(
        _remint_contract(
            runtime_directory,
            runtime_volume_authority_id=(
                substituted_authority.runtime_volume_authority_id
            ),
        )
        for runtime_directory in (
            observation.result_directory,
            observation.temporary_directory,
            observation.control_directory,
        )
    )
    substituted_result_file = _remint_contract(
        observation.result_file,
        runtime_volume_authority_id=(substituted_authority.runtime_volume_authority_id),
        prepared_parent_directory_id=(
            substituted_runtime_directories[0].prepared_runtime_directory_id
        ),
    )
    substituted_workspace = (
        None
        if observation.workspace_proof is None
        else _remint_contract(
            observation.workspace_proof,
            runtime_volume_authority_id=(
                substituted_authority.runtime_volume_authority_id
            ),
        )
    )
    substituted_layout = _remint_contract(
        observation.layout_proof,
        runtime_volume_authority_id=(substituted_authority.runtime_volume_authority_id),
        runtime_volume_evidence_id=(substituted_evidence.runtime_volume_evidence_id),
        prepared_delivery_slot_ids=tuple(
            sorted(
                delivery_slot.prepared_delivery_slot_id
                for delivery_slot in substituted_delivery_slots
            )
        ),
        prepared_runtime_directory_ids=tuple(
            sorted(
                directory.prepared_runtime_directory_id
                for directory in substituted_runtime_directories
            )
        ),
        prepared_result_file_id=substituted_result_file.prepared_file_id,
        prepared_workspace_proof_id=(
            None
            if substituted_workspace is None
            else substituted_workspace.prepared_workspace_proof_id
        ),
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(
            observation,
            runtime_volume_evidence=substituted_evidence,
            input_delivery_slot=substituted_delivery_slots[0],
            result_directory=substituted_runtime_directories[0],
            result_file=substituted_result_file,
            temporary_directory=substituted_runtime_directories[1],
            control_directory=substituted_runtime_directories[2],
            credential_delivery_slot=(
                None
                if len(substituted_delivery_slots) == 1
                else substituted_delivery_slots[1]
            ),
            workspace_proof=substituted_workspace,
            layout_proof=substituted_layout,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"generation_nonce": "f" * 32},
        {"owner_user_id": 1001},
        {"owner_group_id": 1001},
        {"payload_size_limit_bytes": 1},
    ),
)
def test_prepared_volume_aggregate_rejects_delivery_slot_authority_splices(changes):
    prepared = _prepared_execution()
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_delivery_slot=prepared.input_delivery_slot,
        result_directory=prepared.result_directory,
        control_directory=prepared.control_directory,
        result_file=prepared.result_file,
        temporary_directory=prepared.temporary_directory,
        credential_delivery_slot=prepared.credential_delivery_slot,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    substituted_input = _remint_contract(
        observation.input_delivery_slot,
        **changes,
    )
    substituted_layout = _remint_contract(
        observation.layout_proof,
        prepared_delivery_slot_ids=tuple(
            sorted(
                (
                    substituted_input.prepared_delivery_slot_id,
                    *(
                        ()
                        if observation.credential_delivery_slot is None
                        else (
                            observation.credential_delivery_slot.prepared_delivery_slot_id,
                        )
                    ),
                )
            )
        ),
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(
            observation,
            input_delivery_slot=substituted_input,
            layout_proof=substituted_layout,
        )


def test_prepared_volume_aggregate_rejects_workspace_authority_splice():
    prepared = _prepared_execution()
    assert prepared.workspace_proof is not None
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_delivery_slot=prepared.input_delivery_slot,
        result_directory=prepared.result_directory,
        control_directory=prepared.control_directory,
        result_file=prepared.result_file,
        temporary_directory=prepared.temporary_directory,
        credential_delivery_slot=prepared.credential_delivery_slot,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    substituted_workspace = _remint_contract(
        observation.workspace_proof,
        generation_nonce="f" * 32,
    )
    substituted_layout = _remint_contract(
        observation.layout_proof,
        prepared_workspace_proof_id=(substituted_workspace.prepared_workspace_proof_id),
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(
            observation,
            workspace_proof=substituted_workspace,
            layout_proof=substituted_layout,
        )


@pytest.mark.parametrize("mutation", ("sentinel_content", "prepared_file_mode"))
def test_exact_file_observation_detects_mutation_after_initial_read(
    layout_context,
    tmp_path,
    mutation,
):
    _settings, _claim, authority, _empty = layout_context
    root_path = tmp_path / "observation-root"
    file_path = root_path / "observed"
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        if mutation == "sentinel_content":
            expected_payload = authority.generation_nonce.encode("ascii")
            expected_mode = 0o400
        else:
            expected_payload = b""
            expected_mode = 0o600
        file_path.write_bytes(expected_payload)
        file_path.chmod(expected_mode)
        observation = _open_exact_regular_file(
            descriptors,
            root_descriptor,
            "observed",
            expected_payload=expected_payload,
            expected_mode=expected_mode,
            authority=authority,
            root_mount_id=read_run_action_descriptor_mount_id(root_descriptor),
            root_device=os.fstat(root_descriptor).st_dev,
            process_snapshot_size_limit_bytes=(_RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES),
        )

        if mutation == "sentinel_content":
            file_path.chmod(0o600)
            file_path.write_bytes(b"f" * len(expected_payload))
            file_path.chmod(expected_mode)
        else:
            file_path.chmod(0o400)

        with pytest.raises(
            RunActionRuntimeVolumeError,
            match="changed during exact observation",
        ):
            _require_same_exact_regular_file(observation)


def test_layout_materialization_copies_complete_workspace_and_git_closure(
    layout_context,
    publisher_case,
    tmp_path,
):
    settings, _claim_without_workspace, _authority, empty_without_workspace = (
        layout_context
    )
    policy = _policy(settings.docker)
    initial_claim = _claim(policy=policy)
    root_path = tmp_path / "runtime-volume"
    with ExitStack() as descriptors:
        workspace_descriptor, _workspace_identity = publisher_case[
            "active"
        ]._open_execution_workspace(descriptors)
        expected_commit = publisher_case[
            "checkpoint"
        ].safety_state.derivative_frontier.evidence.branch_heads[
            publisher_case["settings"].workspace_git_branch
        ]
        source_frontier = inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=publisher_case["settings"],
            expected_commit_sha=expected_commit,
        )
        workspace_binding = RunActionWorkspaceBinding.from_identity(source_frontier)
        frontier = _remint_contract(
            initial_claim.reservation.frontier,
            workspace_before=workspace_binding,
        )
        reservation = _remint_contract(
            initial_claim.reservation,
            frontier=frontier,
            exact_dependency_ids=tuple(
                sorted(
                    (
                        frontier.frontier_binding_id
                        if dependency_id
                        == initial_claim.reservation.frontier.frontier_binding_id
                        else dependency_id
                    )
                    for dependency_id in initial_claim.reservation.exact_dependency_ids
                )
            ),
        )
        claim = RunActionPreparationClaim.mint(
            reservation=reservation,
            execution_policy=policy,
        )
        authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
        volume = observe_runtime_volume(
            _volume_raw(authority, settings.docker),
            claim,
            authority,
            settings.docker,
        )
        empty = replace(
            empty_without_workspace,
            runtime_volume_authority=authority,
            docker_volume_observation=volume,
        )
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=workspace_descriptor,
            settings=publisher_case["settings"],
        )

        copied_frontier = _materialize_layout_at_descriptor(
            root_descriptor,
            claim=claim,
            authority=authority,
            plan=plan,
            workspace_descriptor=workspace_descriptor,
            settings=publisher_case["settings"],
        )

    assert copied_frontier.source_tree_digest == source_frontier.source_tree_digest
    assert copied_frontier.git_closure_digest == source_frontier.git_closure_digest
    assert plan.workspace_copy_plan.physical_entry_count > (
        source_frontier.source_entry_count
    )
    assert (root_path / "workspace" / ".git" / "HEAD").is_file()
    assert tuple((root_path / "credential").iterdir()) == ()
