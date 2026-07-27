"""Pure contracts for the post-start, pre-release workload barrier."""

from __future__ import annotations

import base64
import os
from dataclasses import fields
from pathlib import PurePosixPath

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import ContractValidationError
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierContractError,
    RunActionBarrierInitProcessObservation,
    RunActionBarrierRunningContainerObservation,
    RunActionBarrierWrapperProcessObservation,
    RunActionMountInfoObservation,
    RunActionMountInfoSnapshot,
    RunActionResolvedFileObservation,
    RunActionResolvedMountKind,
    RunActionResolvedMountRootObservation,
    RunActionResolvedWorkloadObservation,
    RunActionResolvedWorkspaceObservation,
    parse_run_action_mount_info_payload,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_DOCKER_INIT_DESTINATION,
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionCredentialMode,
    RunActionPreparedFileKind,
    RunActionPreparedMountAccess,
    RunActionPreparedRuntimeDirectoryKind,
    run_action_keeper_process_cgroup_path,
)
from test_run_action_supervisor_contracts import (
    _activation_revalidation_receipt,
    _claim,
    _execution_policy,
    _prepared_execution,
    _spawn_commit,
)

_HOST_BOOT_ID = "123e4567-e89b-42d3-a456-426614174000"
_MOUNT_NAMESPACE_DEVICE = 81
_MOUNT_NAMESPACE_INODE = 82
_PROCESS_ID_NAMESPACE_DEVICE = 83
_PROCESS_ID_NAMESPACE_INODE = 84
_INIT_PROCESS_ID = 4100
_WRAPPER_PROCESS_ID = 4101
_INIT_EXECUTABLE_MOUNT_ID = 5100
_HELPER_MOUNT_ID = 5101


def _resolved_graph(*, inode_offset=0, prepared=None, activation=None):
    prepared = (
        _prepared_execution(inode_offset=inode_offset) if prepared is None else prepared
    )
    spawn = _spawn_commit(prepared) if activation is None else activation.spawn_commit
    activation = (
        _activation_revalidation_receipt(prepared, spawn)
        if activation is None
        else activation
    )
    if activation.prepared_execution != prepared:
        raise AssertionError("resolved graph activation differs from its preparation")
    projection = prepared.inert_container_evidence.issued_create_projection
    cgroup_path = run_action_keeper_process_cgroup_path(
        prepared.preparation_claim.execution_policy,
        spawn.provider_execution_id,
    )
    running = RunActionBarrierRunningContainerObservation.mint(
        container_id=spawn.provider_execution_id,
        observed_inspect_projection=projection,
        complete_inspection_digest=tree_or_blob_digest(b"running inspection"),
        container_status="running",
        init_process_id=_INIT_PROCESS_ID + inode_offset,
        restart_count=0,
        started_at="2026-07-25T01:02:03.123456789Z",
        finished_at="0001-01-01T00:00:00Z",
        paused=False,
        restarting=False,
        dead=False,
        oom_killed=False,
        state_error="",
    )
    expected_mount_info = _mount_info_observations(activation, inode_offset)
    mount_info_snapshot = RunActionMountInfoSnapshot.from_raw_payload(
        _mount_info_raw_payload(expected_mount_info)
    )
    mount_info = mount_info_snapshot.records
    mount_info_by_point = {
        observation.mount_point: observation for observation in mount_info
    }
    container_root = mount_info_by_point["/"]
    init_source = projection.docker_init_source_evidence
    init = RunActionBarrierInitProcessObservation.mint(
        provider_execution_id=spawn.provider_execution_id,
        process_id=running.init_process_id,
        parent_process_id=4000 + inode_offset,
        process_start_time_ticks=6100 + inode_offset,
        process_state="S",
        process_cgroup_path=cgroup_path,
        mount_namespace_device=_MOUNT_NAMESPACE_DEVICE + inode_offset,
        mount_namespace_inode=_MOUNT_NAMESPACE_INODE + inode_offset,
        process_id_namespace_device=_PROCESS_ID_NAMESPACE_DEVICE + inode_offset,
        process_id_namespace_inode=_PROCESS_ID_NAMESPACE_INODE + inode_offset,
        command_line=(
            RUN_ACTION_DOCKER_INIT_DESTINATION,
            "--",
            projection.command_executable,
            *projection.command_arguments,
        ),
        root_mount_info_observation_id=container_root.mount_info_observation_id,
        root_mount_id=container_root.mount_id,
        root_device_major=container_root.device_major,
        root_device_minor=container_root.device_minor,
        root_device=os.makedev(
            container_root.device_major,
            container_root.device_minor,
        ),
        root_inode=6202 + inode_offset,
        executable_mount_id=_INIT_EXECUTABLE_MOUNT_ID + inode_offset,
        executable_device=init_source.device,
        executable_inode=init_source.inode,
        executable_digest=init_source.executable_digest,
    )
    helper = projection.supervisor_helper_evidence
    wrapper = RunActionBarrierWrapperProcessObservation.mint(
        provider_execution_id=spawn.provider_execution_id,
        init_process_observation_id=init.barrier_init_process_observation_id,
        process_id=_WRAPPER_PROCESS_ID + inode_offset,
        parent_process_id=init.process_id,
        process_start_time_ticks=6300 + inode_offset,
        process_state="S",
        process_cgroup_path=cgroup_path,
        mount_namespace_device=init.mount_namespace_device,
        mount_namespace_inode=init.mount_namespace_inode,
        process_id_namespace_device=init.process_id_namespace_device,
        process_id_namespace_inode=init.process_id_namespace_inode,
        command_line=(projection.command_executable, *projection.command_arguments),
        root_mount_info_observation_id=init.root_mount_info_observation_id,
        root_mount_id=init.root_mount_id,
        root_device_major=init.root_device_major,
        root_device_minor=init.root_device_minor,
        root_device=init.root_device,
        root_inode=init.root_inode,
        executable_mount_id=_HELPER_MOUNT_ID + inode_offset,
        executable_device=helper.device,
        executable_inode=helper.inode,
        executable_digest=helper.executable_digest,
    )
    roots = _resolved_mount_roots(activation, wrapper, init, mount_info_by_point)
    files = _resolved_files(activation, roots)
    workspace = _resolved_workspace(activation, roots)
    resolved = RunActionResolvedWorkloadObservation.mint(
        activation_revalidation_receipt=activation,
        host_boot_id=_HOST_BOOT_ID,
        running_container_observation=running,
        init_process_observation=init,
        wrapper_process_observation=wrapper,
        mount_info_snapshot=mount_info_snapshot,
        resolved_mount_root_observations=roots,
        resolved_file_observations=files,
        resolved_workspace_observation=workspace,
        control_entry_count=0,
        result_entry_count=0,
        temporary_entry_count=0,
        control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
    )
    return resolved


def _mount_info_observations(activation, inode_offset):
    projection = (
        activation.prepared_execution.inert_container_evidence.issued_create_projection
    )
    sources = _volume_root_sources(activation)
    init_source = projection.docker_init_source_evidence
    helper = projection.supervisor_helper_evidence
    observations = [
        _mount_info(
            mount_id=6200 + inode_offset,
            parent_mount_id=6199 + inode_offset,
            mount_point="/",
            access=RunActionPreparedMountAccess.READ_WRITE,
            filesystem_type="overlay",
            mount_source="overlay",
            device=os.makedev(0, 400 + inode_offset),
        ),
        _mount_info(
            mount_id=_INIT_EXECUTABLE_MOUNT_ID + inode_offset,
            parent_mount_id=6200 + inode_offset,
            mount_point=RUN_ACTION_DOCKER_INIT_DESTINATION,
            access=RunActionPreparedMountAccess.READ_ONLY,
            filesystem_type="tmpfs",
            mount_source="tmpfs",
            device=init_source.device,
        ),
        _mount_info(
            mount_id=_HELPER_MOUNT_ID + inode_offset,
            parent_mount_id=6200 + inode_offset,
            mount_point=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            access=RunActionPreparedMountAccess.READ_ONLY,
            filesystem_type="ext4",
            mount_source="/dev/root",
            device=helper.device,
        ),
        _mount_info(
            mount_id=8000 + inode_offset,
            parent_mount_id=6200 + inode_offset,
            mount_point="/escaped path",
            access=RunActionPreparedMountAccess.READ_ONLY,
            filesystem_type="tmpfs",
            mount_source="/source\tname",
            device=os.makedev(0, 500 + inode_offset),
        ),
    ]
    observations.extend(
        _mount_info(
            mount_id=7000 + position,
            parent_mount_id=6200 + inode_offset,
            mount_point=mount.container_destination,
            access=mount.container_access,
            filesystem_type="tmpfs",
            mount_source="tmpfs",
            device=sources[RunActionResolvedMountKind(mount.kind.value)]["device"],
        )
        for position, mount in enumerate(projection.mounts)
    )
    return tuple(sorted(observations, key=lambda observation: observation.mount_id))


def _mount_info_raw_payload(observations):
    return "".join(
        (
            f"{observation.mount_id} {observation.parent_mount_id} "
            f"{observation.device_major}:{observation.device_minor} "
            f"{_encode_mount_info_field(observation.mount_root)} "
            f"{_encode_mount_info_field(observation.mount_point)} "
            f"{','.join(_encode_mount_info_field(option) for option in observation.mount_options)}"
            f"{''.join(f' {_encode_mount_info_field(field)}' for field in observation.optional_fields)}"
            " - "
            f"{_encode_mount_info_field(observation.filesystem_type)} "
            f"{_encode_mount_info_field(observation.mount_source)} "
            f"{','.join(_encode_mount_info_field(option) for option in observation.super_options)}\n"
        )
        for observation in observations
    ).encode("latin-1")


def _encode_mount_info_field(value):
    escapes = {
        "\t": "\\011",
        "\n": "\\012",
        " ": "\\040",
        "\\": "\\134",
    }
    return "".join(escapes.get(character, character) for character in value)


def _mount_info(
    *,
    mount_id,
    parent_mount_id,
    mount_point,
    access,
    filesystem_type,
    mount_source,
    device,
):
    access_option = "ro" if access is RunActionPreparedMountAccess.READ_ONLY else "rw"
    return RunActionMountInfoObservation.mint(
        mount_id=mount_id,
        parent_mount_id=parent_mount_id,
        device_major=os.major(device),
        device_minor=os.minor(device),
        mount_root="/",
        mount_point=mount_point,
        mount_options=tuple(sorted(("nosuid", access_option))),
        optional_fields=(),
        filesystem_type=filesystem_type,
        mount_source=mount_source,
        super_options=(access_option,),
    )


def _resolved_mount_roots(activation, wrapper, init, mount_info_by_point):
    prepared = activation.prepared_execution
    projection = prepared.inert_container_evidence.issued_create_projection
    init_source = projection.docker_init_source_evidence
    helper = projection.supervisor_helper_evidence
    roots = [
        RunActionResolvedMountRootObservation.mint(
            kind=RunActionResolvedMountKind.DOCKER_INIT,
            source_authority_id=init_source.docker_init_source_evidence_id,
            container_destination=RUN_ACTION_DOCKER_INIT_DESTINATION,
            container_access=RunActionPreparedMountAccess.READ_ONLY,
            mount_info_observation_id=mount_info_by_point[
                RUN_ACTION_DOCKER_INIT_DESTINATION
            ].mount_info_observation_id,
            source_mount_id=init_source.mount_id,
            source_device=init_source.device,
            source_inode=init_source.inode,
            resolved_mount_id=init.executable_mount_id,
            resolved_device=init.executable_device,
            resolved_inode=init.executable_inode,
            mount_namespace_device=init.mount_namespace_device,
            mount_namespace_inode=init.mount_namespace_inode,
            file_type="regular",
            owner_user_id=init_source.owner_user_id,
            owner_group_id=init_source.owner_group_id,
            mode=init_source.mode,
        ),
        RunActionResolvedMountRootObservation.mint(
            kind=RunActionResolvedMountKind.SUPERVISOR_HELPER,
            source_authority_id=helper.supervisor_helper_evidence_id,
            container_destination=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            container_access=RunActionPreparedMountAccess.READ_ONLY,
            mount_info_observation_id=mount_info_by_point[
                RUN_ACTION_SUPERVISOR_HELPER_DESTINATION
            ].mount_info_observation_id,
            source_mount_id=helper.mount_id,
            source_device=helper.device,
            source_inode=helper.inode,
            resolved_mount_id=wrapper.executable_mount_id,
            resolved_device=wrapper.executable_device,
            resolved_inode=wrapper.executable_inode,
            mount_namespace_device=init.mount_namespace_device,
            mount_namespace_inode=init.mount_namespace_inode,
            file_type="regular",
            owner_user_id=helper.owner_user_id,
            owner_group_id=helper.owner_group_id,
            mode=helper.mode,
        ),
    ]
    sources = _volume_root_sources(activation)
    for position, mount in enumerate(projection.mounts):
        source = sources[RunActionResolvedMountKind(mount.kind.value)]
        roots.append(
            RunActionResolvedMountRootObservation.mint(
                kind=RunActionResolvedMountKind(mount.kind.value),
                source_authority_id=source["authority_id"],
                container_destination=mount.container_destination,
                container_access=mount.container_access,
                mount_info_observation_id=mount_info_by_point[
                    mount.container_destination
                ].mount_info_observation_id,
                source_mount_id=source["mount_id"],
                source_device=source["device"],
                source_inode=source["inode"],
                resolved_mount_id=7000 + position,
                resolved_device=source["device"],
                resolved_inode=source["inode"],
                mount_namespace_device=init.mount_namespace_device,
                mount_namespace_inode=init.mount_namespace_inode,
                file_type="directory",
                owner_user_id=source["owner_user_id"],
                owner_group_id=source["owner_group_id"],
                mode=source["mode"],
            )
        )
    return tuple(sorted(roots, key=lambda root: root.container_destination))


def _volume_root_sources(activation):
    prepared = activation.prepared_execution
    runtime_directories = {
        observed.kind: observed
        for observed in activation.activated_runtime_directory_observations
    }
    sources = {
        RunActionResolvedMountKind.INPUT: _source(
            prepared.input_delivery_slot.prepared_delivery_slot_id,
            activation.input_file_observation.parent_mount_id,
            activation.input_file_observation.parent_device,
            activation.input_file_observation.parent_inode,
            prepared.input_delivery_slot.owner_user_id,
            prepared.input_delivery_slot.owner_group_id,
            prepared.input_delivery_slot.mode,
        ),
        RunActionResolvedMountKind.RESULT: _runtime_directory_source(
            runtime_directories[RunActionPreparedRuntimeDirectoryKind.RESULT]
        ),
        RunActionResolvedMountKind.CONTROL: _runtime_directory_source(
            runtime_directories[RunActionPreparedRuntimeDirectoryKind.CONTROL]
        ),
        RunActionResolvedMountKind.TEMPORARY: _runtime_directory_source(
            runtime_directories[RunActionPreparedRuntimeDirectoryKind.TEMPORARY]
        ),
    }
    credential = activation.credential_file_observation
    if credential is not None:
        slot = prepared.credential_delivery_slot
        sources[RunActionResolvedMountKind.CREDENTIAL] = _source(
            slot.prepared_delivery_slot_id,
            credential.parent_mount_id,
            credential.parent_device,
            credential.parent_inode,
            slot.owner_user_id,
            slot.owner_group_id,
            slot.mode,
        )
    workspace = activation.activated_workspace_observation
    if workspace is not None:
        sources[RunActionResolvedMountKind.WORKSPACE] = _source(
            workspace.prepared_workspace_proof_id,
            workspace.mount_id,
            workspace.device,
            workspace.inode,
            workspace.owner_user_id,
            workspace.owner_group_id,
            workspace.root_mode,
        )
    return sources


def _runtime_directory_source(observation):
    return _source(
        observation.prepared_runtime_directory_id,
        observation.mount_id,
        observation.device,
        observation.inode,
        observation.owner_user_id,
        observation.owner_group_id,
        observation.mode,
    )


def _source(
    authority_id,
    mount_id,
    device,
    inode,
    owner_user_id,
    owner_group_id,
    mode,
):
    return {
        "authority_id": authority_id,
        "mount_id": mount_id,
        "device": device,
        "inode": inode,
        "owner_user_id": owner_user_id,
        "owner_group_id": owner_group_id,
        "mode": mode,
    }


def _resolved_files(activation, roots):
    roots_by_kind = {root.kind: root for root in roots}
    files = []
    for activated in (
        activation.input_file_observation,
        activation.credential_file_observation,
    ):
        if activated is None:
            continue
        root = roots_by_kind[RunActionResolvedMountKind(activated.kind.value)]
        files.append(
            RunActionResolvedFileObservation.mint(
                kind=activated.kind,
                activated_file_observation_id=(activated.activated_file_observation_id),
                resolved_mount_root_observation_id=(
                    root.resolved_mount_root_observation_id
                ),
                container_path=(
                    PurePosixPath(root.container_destination)
                    / PurePosixPath(activated.relative_path).name
                ).as_posix(),
                parent_entry_count=1,
                mount_id=root.resolved_mount_id,
                device=activated.device,
                inode=activated.inode,
                file_type=activated.file_type,
                owner_user_id=activated.owner_user_id,
                owner_group_id=activated.owner_group_id,
                mode=activated.mode,
                link_count=activated.link_count,
                size_bytes=activated.size_bytes,
                content_digest=activated.content_digest,
                content_authority_id=activated.content_authority_id,
            )
        )
    return tuple(sorted(files, key=lambda observed: observed.kind.value))


def _resolved_workspace(activation, roots):
    activated = activation.activated_workspace_observation
    if activated is None:
        return None
    root = {observation.kind: observation for observation in roots}[
        RunActionResolvedMountKind.WORKSPACE
    ]
    return RunActionResolvedWorkspaceObservation.mint(
        activated_workspace_observation_id=(
            activated.activated_workspace_observation_id
        ),
        resolved_mount_root_observation_id=(root.resolved_mount_root_observation_id),
        source_tree_digest=activated.source_tree_digest,
        git_closure_digest=activated.git_closure_digest,
        source_entry_count=activated.source_entry_count,
        source_size_bytes=activated.source_size_bytes,
    )


def _remint(contract, **changes):
    values = {
        field.name: getattr(contract, field.name)
        for field in fields(contract)
        if field.name != contract.IDENTITY_FIELD
    }
    values.update(changes)
    return type(contract).mint(**values)


def test_resolved_workload_graph_round_trips_with_exact_canonical_sets():
    resolved = _resolved_graph()

    assert (
        RunActionResolvedWorkloadObservation.from_json_bytes(resolved.to_json_bytes())
        == resolved
    )
    assert tuple(
        root.container_destination for root in resolved.resolved_mount_root_observations
    ) == tuple(
        sorted(
            root.container_destination
            for root in resolved.resolved_mount_root_observations
        )
    )
    assert tuple(
        observed.kind.value for observed in resolved.resolved_file_observations
    ) == ("credential", "input")
    result_root = {
        root.kind: root for root in resolved.resolved_mount_root_observations
    }[RunActionResolvedMountKind.RESULT]
    activated_result = {
        observed.kind: observed
        for observed in resolved.activation_revalidation_receipt.activated_runtime_directory_observations
    }[RunActionPreparedRuntimeDirectoryKind.RESULT]
    assert (
        result_root.source_authority_id,
        result_root.source_mount_id,
        result_root.source_device,
        result_root.source_inode,
        resolved.result_entry_count,
    ) == (
        activated_result.prepared_runtime_directory_id,
        activated_result.mount_id,
        activated_result.device,
        activated_result.inode,
        0,
    )
    assert (
        resolved.control_directory_topology is RunActionControlDirectoryTopology.EMPTY
    )
    assert "activation_event_id" not in resolved.to_dict()
    snapshot = resolved.mount_info_snapshot
    assert snapshot.raw_byte_length == len(snapshot.raw_payload)
    assert snapshot.raw_payload_digest == tree_or_blob_digest(snapshot.raw_payload)
    assert snapshot.records == parse_run_action_mount_info_payload(snapshot.raw_payload)
    assert "records" not in snapshot.to_dict()
    assert "raw_payload" not in snapshot.to_dict()
    escaped = next(
        record for record in snapshot.records if record.mount_point == "/escaped path"
    )
    assert escaped.mount_source == "/source\tname"
    assert {
        type(resolved.running_container_observation).CONTENT_NAMESPACE,
        type(resolved.init_process_observation).CONTENT_NAMESPACE,
        type(resolved.wrapper_process_observation).CONTENT_NAMESPACE,
        type(snapshot).CONTENT_NAMESPACE,
        type(snapshot.records[0]).CONTENT_NAMESPACE,
        type(resolved.resolved_mount_root_observations[0]).CONTENT_NAMESPACE,
        type(resolved.resolved_file_observations[0]).CONTENT_NAMESPACE,
        type(resolved.resolved_workspace_observation).CONTENT_NAMESPACE,
    } == {
        "run-action-barrier-running-container-observation",
        "run-action-barrier-init-process-observation",
        "run-action-barrier-wrapper-process-observation",
        "run-action-mount-info-snapshot",
        "run-action-mount-info-observation",
        "run-action-resolved-mount-root-observation",
        "run-action-resolved-file-observation",
        "run-action-resolved-workspace-observation",
    }


def test_resolved_workload_graph_admits_exact_workspace_and_credential_absence():
    policy = _execution_policy(
        kind=RunFrontierActionKind.EMBEDDING,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    resolved = _resolved_graph(
        prepared=_prepared_execution(claim=_claim(policy=policy))
    )

    assert resolved.resolved_workspace_observation is None
    assert {root.kind for root in resolved.resolved_mount_root_observations} == {
        RunActionResolvedMountKind.DOCKER_INIT,
        RunActionResolvedMountKind.SUPERVISOR_HELPER,
        RunActionResolvedMountKind.CONTROL,
        RunActionResolvedMountKind.INPUT,
        RunActionResolvedMountKind.RESULT,
        RunActionResolvedMountKind.TEMPORARY,
    }
    assert {observed.kind for observed in resolved.resolved_file_observations} == {
        RunActionPreparedFileKind.INPUT,
    }
    assert (
        RunActionResolvedWorkloadObservation.from_json_bytes(resolved.to_json_bytes())
        == resolved
    )


@pytest.mark.parametrize(
    ("target_name", "changes"),
    (
        ("init_process_observation", {"process_state": "T"}),
        (
            "running_container_observation",
            {"started_at": "0001-01-01T00:00:00Z"},
        ),
        ("init_process_observation", {"root_mount_id": 0}),
        ("init_process_observation", {"root_device_minor": 999}),
        ("wrapper_process_observation", {"parent_process_id": 0}),
        ("wrapper_process_observation", {"command_line": ("relative", "sh")}),
        (
            "resolved_file_observations",
            {"parent_entry_count": 2},
        ),
    ),
)
def test_leaf_contracts_reject_unsafe_process_mount_and_file_shapes(
    target_name,
    changes,
):
    resolved = _resolved_graph()
    target = getattr(resolved, target_name)
    if isinstance(target, tuple):
        target = target[0]

    with pytest.raises(RunActionBarrierContractError):
        _remint(target, **changes)


def test_resolved_mount_requires_the_actual_inode_to_join_event_5():
    root = _resolved_graph().resolved_mount_root_observations[0]

    with pytest.raises(RunActionBarrierContractError, match="substituted"):
        _remint(root, resolved_inode=root.resolved_inode + 1)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda resolved: {
            "activation_revalidation_receipt": (
                _resolved_graph(inode_offset=100).activation_revalidation_receipt
            )
        },
        lambda resolved: {
            "running_container_observation": _remint(
                resolved.running_container_observation,
                container_id="b" * 64,
            )
        },
        lambda resolved: {
            "init_process_observation": _remint(
                resolved.init_process_observation,
                executable_inode=resolved.init_process_observation.executable_inode + 1,
            )
        },
        lambda resolved: {
            "init_process_observation": _remint(
                resolved.init_process_observation,
                command_line=resolved.wrapper_process_observation.command_line,
            )
        },
        lambda resolved: {
            "init_process_observation": _remint(
                resolved.init_process_observation,
                parent_process_id=resolved.wrapper_process_observation.process_id,
            )
        },
        lambda resolved: {
            "wrapper_process_observation": _remint(
                resolved.wrapper_process_observation,
                process_cgroup_path=(
                    "/foreign.slice/"
                    f"docker-{resolved.wrapper_process_observation.provider_execution_id}.scope"
                ),
            )
        },
        lambda resolved: {
            "wrapper_process_observation": _remint(
                resolved.wrapper_process_observation,
                command_line=resolved.init_process_observation.command_line,
            )
        },
        lambda resolved: {
            "wrapper_process_observation": _remint(
                resolved.wrapper_process_observation,
                process_start_time_ticks=(
                    resolved.init_process_observation.process_start_time_ticks - 1
                ),
            )
        },
        lambda resolved: {
            "mount_info_snapshot": _resolved_graph(inode_offset=100).mount_info_snapshot
        },
        lambda resolved: {
            "resolved_mount_root_observations": tuple(
                reversed(resolved.resolved_mount_root_observations)
            )
        },
        lambda resolved: {
            "resolved_mount_root_observations": (
                resolved.resolved_mount_root_observations[:-1]
            )
        },
        lambda resolved: {
            "resolved_mount_root_observations": (
                _remint(
                    resolved.resolved_mount_root_observations[0],
                    source_authority_id=content_id(
                        "run-action-supervisor-helper-evidence",
                        {"foreign": True},
                    ),
                ),
                *resolved.resolved_mount_root_observations[1:],
            )
        },
        lambda resolved: {
            "resolved_file_observations": tuple(
                reversed(resolved.resolved_file_observations)
            )
        },
        lambda resolved: {
            "resolved_file_observations": (
                _remint(
                    resolved.resolved_file_observations[0],
                    activated_file_observation_id=content_id(
                        "run-action-activated-file-observation",
                        {"foreign": True},
                    ),
                ),
                *resolved.resolved_file_observations[1:],
            )
        },
        lambda resolved: {
            "resolved_workspace_observation": _remint(
                resolved.resolved_workspace_observation,
                source_tree_digest=tree_or_blob_digest(b"foreign workspace"),
            )
        },
        lambda resolved: {"control_entry_count": 1},
        lambda resolved: {"result_entry_count": 1},
        lambda resolved: {
            "control_directory_topology": RunActionControlDirectoryTopology.RELEASED
        },
    ),
)
def test_aggregate_rejects_cross_occurrence_splices_and_noncanonical_sets(mutate):
    resolved = _resolved_graph()

    with pytest.raises(
        RunActionBarrierContractError,
        match="activation receipt graph",
    ):
        _remint(resolved, **mutate(resolved))


@pytest.mark.parametrize("same_destination", (False, True))
def test_complete_mountinfo_rejects_nested_and_stacked_mount_overlays(
    same_destination,
):
    resolved = _resolved_graph()
    root = resolved.resolved_mount_root_observations[0]
    selected = {
        observation.mount_info_observation_id: observation
        for observation in resolved.mount_info_snapshot.records
    }[root.mount_info_observation_id]
    overlay = _remint(
        selected,
        mount_id=max(
            observation.mount_id for observation in resolved.mount_info_snapshot.records
        )
        + 1,
        parent_mount_id=selected.mount_id,
        mount_point=(
            root.container_destination
            if same_destination
            else f"{root.container_destination}/nested"
        ),
    )
    observations = tuple(
        sorted(
            (*resolved.mount_info_snapshot.records, overlay),
            key=lambda observation: observation.mount_id,
        )
    )
    snapshot = RunActionMountInfoSnapshot.from_raw_payload(
        _mount_info_raw_payload(observations)
    )

    with pytest.raises(
        RunActionBarrierContractError,
        match="activation receipt graph",
    ):
        _remint(
            resolved,
            mount_info_snapshot=snapshot,
        )


def test_mountinfo_snapshot_rejects_independent_raw_mutation():
    snapshot = _resolved_graph().mount_info_snapshot
    first_record = snapshot.records[0]
    mutated_raw = snapshot.raw_payload.replace(
        f"{first_record.mount_id} ".encode("ascii"),
        f"{first_record.mount_id + 100000} ".encode("ascii"),
        1,
    )

    with pytest.raises(RunActionBarrierContractError, match="full-EOF bytes"):
        _remint(
            snapshot,
            raw_payload_base64=base64.b64encode(mutated_raw).decode("ascii"),
            raw_payload_digest=tree_or_blob_digest(mutated_raw),
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw.removesuffix(b"\n"),
        lambda raw: raw.replace(b"\n", b"\n\n", 1),
        lambda raw: raw + raw.splitlines(keepends=True)[0],
        lambda raw: raw.replace(b"\\040", b"\\041", 1),
        lambda raw: raw.replace(b"nosuid,ro", b"nosuid,ro,ro", 1),
        lambda raw: b"0" + raw,
        lambda raw: raw.replace(b" ", b"  ", 1),
        lambda raw: raw.replace(b" ", b"\t", 1),
    ),
)
def test_mountinfo_parser_rejects_malformed_and_duplicate_payloads(mutate):
    raw_payload = _resolved_graph().mount_info_snapshot.raw_payload

    with pytest.raises(RunActionBarrierContractError):
        parse_run_action_mount_info_payload(mutate(raw_payload))


@pytest.mark.parametrize(
    "field",
    (
        "process_id",
        "parent_process_id",
        "process_start_time_ticks",
        "mount_namespace_device",
        "mount_namespace_inode",
        "process_id_namespace_device",
        "process_id_namespace_inode",
        "root_mount_id",
        "root_device_major",
        "root_device_minor",
        "root_device",
        "root_inode",
        "executable_mount_id",
        "executable_device",
        "executable_inode",
    ),
)
def test_process_observations_reject_values_above_unsigned_64(field):
    resolved = _resolved_graph()

    with pytest.raises(RunActionBarrierContractError):
        _remint(
            resolved.init_process_observation,
            **{field: RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER + 1},
        )


@pytest.mark.parametrize(
    "replacement",
    (
        str(RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER + 1).encode("ascii"),
        b"9" * 4301,
    ),
)
def test_mountinfo_parser_rejects_integer_above_unsigned_64_before_conversion(
    replacement,
):
    raw_payload = _resolved_graph().mount_info_snapshot.raw_payload
    first_mount_id = raw_payload.split(b" ", 1)[0]

    with pytest.raises(RunActionBarrierContractError, match="malformed"):
        parse_run_action_mount_info_payload(
            raw_payload.replace(
                first_mount_id,
                replacement,
                1,
            )
        )


@pytest.mark.parametrize(
    ("source_bytes", "expected_source"),
    (
        (b"\xff", "/source-\u00ff"),
        ("\u00e9".encode("utf-8"), "/source-\u00c3\u00a9"),
    ),
)
def test_mountinfo_snapshot_preserves_non_ascii_path_bytes_exactly(
    source_bytes,
    expected_source,
):
    raw_payload = b"1 1 0:1 / / rw - overlay /source-" + source_bytes + b" rw\n"

    snapshot = RunActionMountInfoSnapshot.from_raw_payload(raw_payload)

    assert snapshot.raw_payload == raw_payload
    assert snapshot.records[0].mount_source == expected_source
    round_tripped = RunActionMountInfoSnapshot.from_json_bytes(snapshot.to_json_bytes())
    assert round_tripped == snapshot
    assert round_tripped.raw_payload == raw_payload


def test_mountinfo_snapshot_accepts_kernel_hash_escape_in_mount_source():
    raw_payload = b"1 1 0:1 / / rw - overlay source\\043name rw\n"

    snapshot = RunActionMountInfoSnapshot.from_raw_payload(raw_payload)

    assert snapshot.raw_payload == raw_payload
    assert snapshot.records[0].mount_source == "source#name"
    round_tripped = RunActionMountInfoSnapshot.from_json_bytes(snapshot.to_json_bytes())
    assert round_tripped.raw_payload == raw_payload


def test_resolved_workload_rejects_mountinfo_above_pinned_snapshot_budget():
    policy = _execution_policy()
    limits = _remint(policy.supervisor_limits, process_snapshot_size_bytes=1)
    bounded_policy = _remint(policy, supervisor_limits=limits)

    with pytest.raises(
        RunActionBarrierContractError,
        match="activation receipt graph",
    ):
        _resolved_graph(
            prepared=_prepared_execution(claim=_claim(policy=bounded_policy))
        )


def test_mountinfo_access_is_derived_from_the_selected_namespace_record():
    resolved = _resolved_graph()
    read_only_root = next(
        root
        for root in resolved.resolved_mount_root_observations
        if root.container_access is RunActionPreparedMountAccess.READ_ONLY
    )
    records = []
    replacement_record = None
    for record in resolved.mount_info_snapshot.records:
        if record.mount_info_observation_id == read_only_root.mount_info_observation_id:
            replacement_record = _remint(
                record,
                mount_options=("nosuid", "rw"),
                super_options=("rw",),
            )
            records.append(replacement_record)
        else:
            records.append(record)
    observations = tuple(records)
    snapshot = RunActionMountInfoSnapshot.from_raw_payload(
        _mount_info_raw_payload(observations)
    )
    replacement_record = {record.mount_id: record for record in snapshot.records}[
        replacement_record.mount_id
    ]
    roots = tuple(
        (
            _remint(
                root,
                mount_info_observation_id=replacement_record.mount_info_observation_id,
            )
            if root == read_only_root
            else root
        )
        for root in resolved.resolved_mount_root_observations
    )

    with pytest.raises(
        RunActionBarrierContractError,
        match="activation receipt graph",
    ):
        _remint(
            resolved,
            mount_info_snapshot=snapshot,
            resolved_mount_root_observations=roots,
        )


def test_credential_observation_never_carries_a_secret_digest():
    resolved = _resolved_graph()
    credential = {
        observed.kind: observed for observed in resolved.resolved_file_observations
    }[RunActionPreparedFileKind.CREDENTIAL]

    with pytest.raises(RunActionBarrierContractError, match="incomplete"):
        _remint(
            credential,
            content_digest=tree_or_blob_digest(b"must not be recorded"),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"host_boot_id": "not-a-boot-id"},
        {"result_entry_count": 1},
        {"temporary_entry_count": 1},
    ),
)
def test_aggregate_rejects_invalid_boot_and_empty_directory_claims(changes):
    resolved = _resolved_graph()

    with pytest.raises(RunActionBarrierContractError):
        _remint(resolved, **changes)


def test_strict_reader_rejects_unknown_resolved_workload_fields():
    resolved = _resolved_graph()
    payload = resolved.to_dict()
    payload["unowned"] = True

    with pytest.raises(ContractValidationError, match="unknown"):
        RunActionResolvedWorkloadObservation.from_dict(payload)
