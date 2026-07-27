"""Pre-delivery canonical byte envelope for one future workload release."""

from __future__ import annotations

from dataclasses import fields
from pathlib import PurePosixPath
from typing import Any

from kapso.cross_run.canonical import canonical_json_bytes
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_activation_envelope import (
    activation_revalidation_receipt_wire_bound,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
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
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionCredentialValidityObservation,
    RunActionReleaseAuthorizationObservation,
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_DOCKER_INIT_DESTINATION,
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionCredentialMode,
    RunActionPreparedExecution,
    RunActionPreparedFileKind,
    RunActionPreparedMountAccess,
    RunActionPreparedMountKind,
    run_action_keeper_process_cgroup_path,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)

_MAXIMUM_PHYSICAL_INTEGER = RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
_MAXIMUM_DOCKER_TIMESTAMP = "9999-12-31T23:59:59.999999999Z"
_ZERO_DOCKER_TIMESTAMP = "0001-01-01T00:00:00Z"
_MAXIMUM_HOST_BOOT_ID = "ffffffff-ffff-4fff-afff-ffffffffffff"
_MAXIMUM_SHA256_DIGEST = f"sha256:{'f' * 64}"


class RunActionReleaseEnvelopeError(ValueError):
    """A future release receipt cannot be bounded from durable authority."""


def workload_release_receipt_size_bound(
    *,
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    required_security_observation: SecurityDenylistObservation,
) -> int:
    """Return a sound canonical byte bound for this future release receipt."""

    if (
        type(prepared_execution) is not RunActionPreparedExecution
        or type(spawn_commit) is not RunActionSpawnCommit
        or type(required_security_observation) is not SecurityDenylistObservation
    ):
        raise RunActionReleaseEnvelopeError(
            "release envelope requires exact prepared, spawn, and security authority"
        )
    reservation = prepared_execution.preparation_claim.reservation
    if (
        required_security_observation.matched_revocations
        or required_security_observation.observation_id
        != reservation.frontier.security_observation_id
    ):
        raise RunActionReleaseEnvelopeError(
            "release envelope security differs from the prepared reservation"
        )
    activation = activation_revalidation_receipt_wire_bound(
        prepared_execution,
        spawn_commit,
    )
    resolved = _resolved_workload_wire(
        prepared_execution,
        spawn_commit,
        activation,
    )
    authorization = _release_authorization_wire(
        activation,
        prepared_execution,
        required_security_observation,
    )
    receipt = _sealed_wire(
        RunActionWorkloadReleaseReceipt,
        workload_release_receipt_id=_content_identifier(
            RunActionWorkloadReleaseReceipt
        ),
        activation_event_id=_content_identifier(RunActionExecutionEvent),
        resolved_workload_observation=resolved,
        release_authorization_observation=authorization,
    )
    process_snapshot_size_bytes = (
        prepared_execution.preparation_claim.execution_policy.supervisor_limits.process_snapshot_size_bytes
    )
    return len(canonical_json_bytes(receipt)) + _mount_info_base64_size(
        process_snapshot_size_bytes
    )


def _resolved_workload_wire(
    prepared: RunActionPreparedExecution,
    spawn: RunActionSpawnCommit,
    activation: dict[str, Any],
) -> dict[str, Any]:
    projection = prepared.inert_container_evidence.issued_create_projection
    process_cgroup_path = run_action_keeper_process_cgroup_path(
        prepared.preparation_claim.execution_policy,
        spawn.provider_execution_id,
    )
    init_process = _process_wire(
        RunActionBarrierInitProcessObservation,
        provider_execution_id=spawn.provider_execution_id,
        init_process_observation_id=None,
        process_cgroup_path=process_cgroup_path,
        command_line=(
            RUN_ACTION_DOCKER_INIT_DESTINATION,
            "--",
            projection.command_executable,
            *projection.command_arguments,
        ),
        executable_digest=(projection.docker_init_source_evidence.executable_digest),
    )
    wrapper_process = _process_wire(
        RunActionBarrierWrapperProcessObservation,
        provider_execution_id=spawn.provider_execution_id,
        init_process_observation_id=init_process["barrier_init_process_observation_id"],
        process_cgroup_path=process_cgroup_path,
        command_line=(
            projection.command_executable,
            *projection.command_arguments,
        ),
        executable_digest=(projection.supervisor_helper_evidence.executable_digest),
    )
    roots = _resolved_mount_root_wires(prepared)
    files = _resolved_file_wires(prepared, activation)
    return _sealed_wire(
        RunActionResolvedWorkloadObservation,
        resolved_workload_observation_id=_content_identifier(
            RunActionResolvedWorkloadObservation
        ),
        activation_revalidation_receipt=activation,
        host_boot_id=_MAXIMUM_HOST_BOOT_ID,
        running_container_observation=_sealed_wire(
            RunActionBarrierRunningContainerObservation,
            barrier_running_container_observation_id=_content_identifier(
                RunActionBarrierRunningContainerObservation
            ),
            container_id=spawn.provider_execution_id,
            observed_inspect_projection=projection.to_dict(),
            complete_inspection_digest=_MAXIMUM_SHA256_DIGEST,
            container_status="running",
            init_process_id=_MAXIMUM_PHYSICAL_INTEGER,
            restart_count=0,
            started_at=_MAXIMUM_DOCKER_TIMESTAMP,
            finished_at=_ZERO_DOCKER_TIMESTAMP,
            paused=False,
            restarting=False,
            dead=False,
            oom_killed=False,
            state_error="",
        ),
        init_process_observation=init_process,
        wrapper_process_observation=wrapper_process,
        mount_info_snapshot=_mount_info_snapshot_wire(
            prepared.preparation_claim.execution_policy.supervisor_limits.process_snapshot_size_bytes
        ),
        resolved_mount_root_observations=roots,
        resolved_file_observations=files,
        resolved_workspace_observation=_resolved_workspace_wire(
            prepared,
            activation,
        ),
        control_entry_count=0,
        temporary_entry_count=0,
        control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
    )


def _process_wire(
    contract_type: type[
        RunActionBarrierInitProcessObservation
        | RunActionBarrierWrapperProcessObservation
    ],
    *,
    provider_execution_id: str,
    init_process_observation_id: str | None,
    process_cgroup_path: str,
    command_line: tuple[str, ...],
    executable_digest: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "provider_execution_id": provider_execution_id,
        "process_id": _MAXIMUM_PHYSICAL_INTEGER,
        "parent_process_id": _MAXIMUM_PHYSICAL_INTEGER,
        "process_start_time_ticks": _MAXIMUM_PHYSICAL_INTEGER,
        "process_state": "S",
        "process_cgroup_path": process_cgroup_path,
        "mount_namespace_device": _MAXIMUM_PHYSICAL_INTEGER,
        "mount_namespace_inode": _MAXIMUM_PHYSICAL_INTEGER,
        "process_id_namespace_device": _MAXIMUM_PHYSICAL_INTEGER,
        "process_id_namespace_inode": _MAXIMUM_PHYSICAL_INTEGER,
        "command_line": command_line,
        "root_mount_info_observation_id": _content_identifier(
            RunActionMountInfoObservation
        ),
        "root_mount_id": _MAXIMUM_PHYSICAL_INTEGER,
        "root_device_major": _MAXIMUM_PHYSICAL_INTEGER,
        "root_device_minor": _MAXIMUM_PHYSICAL_INTEGER,
        "root_device": _MAXIMUM_PHYSICAL_INTEGER,
        "root_inode": _MAXIMUM_PHYSICAL_INTEGER,
        "executable_mount_id": _MAXIMUM_PHYSICAL_INTEGER,
        "executable_device": _MAXIMUM_PHYSICAL_INTEGER,
        "executable_inode": _MAXIMUM_PHYSICAL_INTEGER,
        "executable_digest": executable_digest,
    }
    if contract_type is RunActionBarrierInitProcessObservation:
        values["barrier_init_process_observation_id"] = _content_identifier(
            RunActionBarrierInitProcessObservation
        )
    else:
        values["barrier_wrapper_process_observation_id"] = _content_identifier(
            RunActionBarrierWrapperProcessObservation
        )
        values["init_process_observation_id"] = init_process_observation_id
    return _sealed_wire(contract_type, **values)


def _mount_info_snapshot_wire(process_snapshot_size_bytes: int) -> dict[str, Any]:
    return _sealed_wire(
        RunActionMountInfoSnapshot,
        mount_info_snapshot_id=_content_identifier(RunActionMountInfoSnapshot),
        raw_payload_base64="",
        raw_byte_length=process_snapshot_size_bytes,
        raw_payload_digest=_MAXIMUM_SHA256_DIGEST,
    )


def _mount_info_base64_size(process_snapshot_size_bytes: int) -> int:
    return 4 * ((process_snapshot_size_bytes + 2) // 3)


def _resolved_mount_root_wires(
    prepared: RunActionPreparedExecution,
) -> tuple[dict[str, Any], ...]:
    projection = prepared.inert_container_evidence.issued_create_projection
    init_source = projection.docker_init_source_evidence
    helper_source = projection.supervisor_helper_evidence
    roots = [
        _resolved_mount_root_wire(
            kind=RunActionResolvedMountKind.DOCKER_INIT,
            source_authority_id=init_source.docker_init_source_evidence_id,
            container_destination=RUN_ACTION_DOCKER_INIT_DESTINATION,
            container_access=RunActionPreparedMountAccess.READ_ONLY,
            owner_user_id=init_source.owner_user_id,
            owner_group_id=init_source.owner_group_id,
            mode=init_source.mode,
            file_type="regular",
        ),
        _resolved_mount_root_wire(
            kind=RunActionResolvedMountKind.SUPERVISOR_HELPER,
            source_authority_id=helper_source.supervisor_helper_evidence_id,
            container_destination=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            container_access=RunActionPreparedMountAccess.READ_ONLY,
            owner_user_id=helper_source.owner_user_id,
            owner_group_id=helper_source.owner_group_id,
            mode=helper_source.mode,
            file_type="regular",
        ),
    ]
    for mount in projection.mounts:
        source = _prepared_mount_source(prepared, mount.kind)
        roots.append(
            _resolved_mount_root_wire(
                kind=RunActionResolvedMountKind(mount.kind.value),
                source_authority_id=source["authority_id"],
                container_destination=mount.container_destination,
                container_access=mount.container_access,
                owner_user_id=source["owner_user_id"],
                owner_group_id=source["owner_group_id"],
                mode=source["mode"],
                file_type="directory",
            )
        )
    return tuple(sorted(roots, key=lambda root: root["container_destination"]))


def _resolved_mount_root_wire(
    *,
    kind: RunActionResolvedMountKind,
    source_authority_id: str,
    container_destination: str,
    container_access: RunActionPreparedMountAccess,
    owner_user_id: int,
    owner_group_id: int,
    mode: int,
    file_type: str,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionResolvedMountRootObservation,
        resolved_mount_root_observation_id=_content_identifier(
            RunActionResolvedMountRootObservation
        ),
        kind=kind,
        source_authority_id=source_authority_id,
        container_destination=container_destination,
        container_access=container_access,
        mount_info_observation_id=_content_identifier(RunActionMountInfoObservation),
        source_mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        source_device=_MAXIMUM_PHYSICAL_INTEGER,
        source_inode=_MAXIMUM_PHYSICAL_INTEGER,
        resolved_mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        resolved_device=_MAXIMUM_PHYSICAL_INTEGER,
        resolved_inode=_MAXIMUM_PHYSICAL_INTEGER,
        mount_namespace_device=_MAXIMUM_PHYSICAL_INTEGER,
        mount_namespace_inode=_MAXIMUM_PHYSICAL_INTEGER,
        file_type=file_type,
        owner_user_id=owner_user_id,
        owner_group_id=owner_group_id,
        mode=mode,
    )


def _prepared_mount_source(
    prepared: RunActionPreparedExecution,
    kind: RunActionPreparedMountKind,
) -> dict[str, Any]:
    sources = {
        RunActionPreparedMountKind.INPUT: prepared.input_delivery_slot,
        RunActionPreparedMountKind.RESULT: prepared.result_directory,
        RunActionPreparedMountKind.CONTROL: prepared.control_directory,
        RunActionPreparedMountKind.TEMPORARY: prepared.temporary_directory,
        RunActionPreparedMountKind.CREDENTIAL: prepared.credential_delivery_slot,
        RunActionPreparedMountKind.WORKSPACE: prepared.workspace_proof,
    }
    source = sources[kind]
    if source is None:
        raise RunActionReleaseEnvelopeError(
            "release envelope prepared mount lacks its source authority"
        )
    identity_field = type(source).IDENTITY_FIELD
    if identity_field is None:
        raise RunActionReleaseEnvelopeError(
            "release envelope prepared mount source lacks an identity"
        )
    return {
        "authority_id": getattr(source, identity_field),
        "owner_user_id": source.owner_user_id,
        "owner_group_id": source.owner_group_id,
        "mode": (
            source.root_mode
            if kind is RunActionPreparedMountKind.WORKSPACE
            else source.mode
        ),
    }


def _resolved_file_wires(
    prepared: RunActionPreparedExecution,
    activation: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    projection = prepared.inert_container_evidence.issued_create_projection
    destinations = {
        RunActionPreparedFileKind(mount.kind.value): mount.container_destination
        for mount in projection.mounts
        if mount.kind
        in {
            RunActionPreparedMountKind.INPUT,
            RunActionPreparedMountKind.RESULT,
            RunActionPreparedMountKind.CREDENTIAL,
        }
    }
    activated_files = tuple(
        observed
        for observed in (
            activation["input_file_observation"],
            activation["result_file_observation"],
            activation["credential_file_observation"],
        )
        if observed is not None
    )
    files = []
    for observed in activated_files:
        kind = RunActionPreparedFileKind(observed["kind"])
        files.append(
            _sealed_wire(
                RunActionResolvedFileObservation,
                resolved_file_observation_id=_content_identifier(
                    RunActionResolvedFileObservation
                ),
                kind=kind,
                activated_file_observation_id=observed["activated_file_observation_id"],
                resolved_mount_root_observation_id=_content_identifier(
                    RunActionResolvedMountRootObservation
                ),
                container_path=(
                    PurePosixPath(destinations[kind])
                    / PurePosixPath(observed["relative_path"]).name
                ).as_posix(),
                parent_entry_count=1,
                mount_id=_MAXIMUM_PHYSICAL_INTEGER,
                device=_MAXIMUM_PHYSICAL_INTEGER,
                inode=_MAXIMUM_PHYSICAL_INTEGER,
                file_type=observed["file_type"],
                owner_user_id=observed["owner_user_id"],
                owner_group_id=observed["owner_group_id"],
                mode=observed["mode"],
                link_count=observed["link_count"],
                size_bytes=observed["size_bytes"],
                content_digest=observed["content_digest"],
                content_authority_id=observed["content_authority_id"],
            )
        )
    return tuple(sorted(files, key=lambda observed: observed["kind"].value))


def _resolved_workspace_wire(
    prepared: RunActionPreparedExecution,
    activation: dict[str, Any],
) -> dict[str, Any] | None:
    activated = activation["activated_workspace_observation"]
    if activated is None:
        return None
    proof = prepared.workspace_proof
    if proof is None:
        raise RunActionReleaseEnvelopeError(
            "release envelope activated workspace lacks prepared authority"
        )
    return _sealed_wire(
        RunActionResolvedWorkspaceObservation,
        resolved_workspace_observation_id=_content_identifier(
            RunActionResolvedWorkspaceObservation
        ),
        activated_workspace_observation_id=activated[
            "activated_workspace_observation_id"
        ],
        resolved_mount_root_observation_id=_content_identifier(
            RunActionResolvedMountRootObservation
        ),
        source_tree_digest=proof.observed_source_tree_digest,
        git_closure_digest=proof.observed_git_closure_digest,
        source_entry_count=proof.observed_source_entry_count,
        source_size_bytes=proof.observed_source_size_bytes,
    )


def _release_authorization_wire(
    activation: dict[str, Any],
    prepared: RunActionPreparedExecution,
    security: SecurityDenylistObservation,
) -> dict[str, Any]:
    credential = activation["credential_file_observation"]
    credential_required = (
        prepared.preparation_claim.execution_policy.credential_policy.mode
        is RunActionCredentialMode.SUPERVISOR_FILE
    )
    if (credential is not None) != credential_required:
        raise RunActionReleaseEnvelopeError(
            "release envelope credential topology is invalid"
        )
    credential_validity = (
        None
        if credential is None
        else _sealed_wire(
            RunActionCredentialValidityObservation,
            credential_validity_observation_id=_content_identifier(
                RunActionCredentialValidityObservation
            ),
            activated_credential_file_observation_id=credential[
                "activated_file_observation_id"
            ],
            credential_lease_authority_id=credential["content_authority_id"],
            observed_at_realtime_nanoseconds=_MAXIMUM_PHYSICAL_INTEGER,
            valid_until_realtime_nanoseconds=_MAXIMUM_PHYSICAL_INTEGER,
        )
    )
    return _sealed_wire(
        RunActionReleaseAuthorizationObservation,
        release_authorization_observation_id=_content_identifier(
            RunActionReleaseAuthorizationObservation
        ),
        security_observation=security.to_dict(),
        authorized_at_boottime_nanoseconds=_MAXIMUM_PHYSICAL_INTEGER,
        authorized_at_realtime_nanoseconds=_MAXIMUM_PHYSICAL_INTEGER,
        credential_validity_observation=credential_validity,
    )


def _content_identifier(contract_type: type[StrictContract]) -> str:
    namespace = contract_type.CONTENT_NAMESPACE
    if not isinstance(namespace, str) or contract_type.IDENTITY_FIELD is None:
        raise RunActionReleaseEnvelopeError(
            f"{contract_type.__name__} lacks a content identity"
        )
    return f"{namespace}:sha256:{'f' * 64}"


def _sealed_wire(
    contract_type: type[StrictContract],
    **values: Any,
) -> dict[str, Any]:
    expected = tuple(field.name for field in fields(contract_type))
    missing = tuple(sorted(set(expected) - set(values)))
    unknown = tuple(sorted(set(values) - set(expected)))
    if missing or unknown:
        raise RunActionReleaseEnvelopeError(
            f"{contract_type.__name__} envelope fields changed; "
            f"missing={missing}, unknown={unknown}"
        )
    return values


__all__ = [
    "RunActionReleaseEnvelopeError",
    "workload_release_receipt_size_bound",
]
