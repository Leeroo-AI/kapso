"""Pre-mutation canonical wire envelope for run-action event 3."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_docker_projection import (
    DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION,
    DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
    DockerRunActionCommand,
    main_barrier_command,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionCreateInspectProjection,
    DockerRunActionExecutionPolicy,
    DockerRunActionKeeperCreateInspectProjection,
    RUN_ACTION_BARRIER_PROTOCOL_VERSION,
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionCredentialMode,
    RunActionDockerInitSourceEvidence,
    RunActionInertContainerEvidence,
    RunActionMountedKeeperHelperEvidence,
    RunActionPreparationAllocation,
    RunActionPreparationClaim,
    RunActionPreparedDeliverySlot,
    RunActionPreparedExecution,
    RunActionPreparedFileKind,
    RunActionPreparedMountAccess,
    RunActionPreparedRuntimeDirectory,
    RunActionPreparedRuntimeDirectoryKind,
    RunActionPreparedWorkspaceProof,
    RunActionRuntimeVolumeAuthority,
    RunActionRuntimeVolumeEvidence,
    RunActionRuntimeVolumeLayoutProof,
    RunActionRuntimeVolumeSentinelEvidence,
    RunActionSupervisorHelperEvidence,
    RunActionVolumeKeeperEvidence,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_main_mounts,
    run_action_keeper_process_cgroup_path,
)
from kapso.cross_run.settings import DockerRuntimeSettings

_MAXIMUM_PHYSICAL_INTEGER = RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
_MAXIMUM_CONTAINER_ID = "f" * 64
_MAXIMUM_DIGEST = "sha256:" + "f" * 64
_ZERO_DOCKER_TIMESTAMP = "0001-01-01T00:00:00Z"


class RunActionPreparedEnvelopeError(ValueError):
    """Event 3 cannot be bounded from its exact pre-mutation authority."""


def prepared_execution_event_size_bound(
    *,
    preparation_allocation: RunActionPreparationAllocation,
    predecessor_event_id: str,
    command: DockerRunActionCommand,
    runtime_settings: DockerRuntimeSettings,
) -> int:
    """Return a sound canonical byte bound for the exact future event 3."""

    if (
        type(preparation_allocation) is not RunActionPreparationAllocation
        or type(command) is not DockerRunActionCommand
        or type(runtime_settings) is not DockerRuntimeSettings
    ):
        raise RunActionPreparedEnvelopeError(
            "prepared event envelope requires exact allocation, command, and runtime"
        )
    predecessor = require_content_id(
        predecessor_event_id,
        "prepared event predecessor",
    )
    if predecessor.split(":sha256:", 1)[0] != RunActionExecutionEvent.CONTENT_NAMESPACE:
        raise RunActionPreparedEnvelopeError(
            "prepared event envelope predecessor is not a run-action event"
        )
    claim = preparation_allocation.preparation_claim
    authority = preparation_allocation.runtime_volume_authority
    policy = claim.execution_policy
    if (
        command.command_template_id != policy.command_template_id
        or policy.projection_protocol_version
        != DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION
        or policy.raw_field_schema_id != DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID
        or policy.docker_runtime_settings_digest
        != tree_or_blob_digest(runtime_settings.to_json_bytes())
        or policy.supervisor_helper_source_path
        != runtime_settings.helper_executable_path
        or policy.supervisor_helper_executable_digest
        != runtime_settings.helper_executable_digest
        or policy.docker_init_source_path != runtime_settings.init_executable_path
        or policy.docker_init_executable_digest
        != runtime_settings.init_executable_digest
    ):
        raise RunActionPreparedEnvelopeError(
            "prepared event envelope differs from its Docker execution authority"
        )
    prepared = _prepared_execution_wire(
        preparation_allocation,
        command,
        runtime_settings,
    )
    event = _sealed_wire(
        RunActionExecutionEvent,
        event_id=_content_identifier(RunActionExecutionEvent),
        event_number=3,
        predecessor_event_id=predecessor,
        event_kind=RunActionExecutionEventKind.EXECUTION_PREPARED,
        reservation=claim.reservation.to_dict(),
        preparation_allocation=None,
        prepared_execution=prepared,
        spawn_commit=None,
        activation_revalidation_receipt=None,
        credential_retirement_intent=None,
        provider_termination_receipt=None,
        result_receipt=None,
        result_decision=None,
        acceptance=None,
        workspace_after=None,
    )
    return len(canonical_json_bytes(event))


def _prepared_execution_wire(
    allocation: RunActionPreparationAllocation,
    command: DockerRunActionCommand,
    runtime_settings: DockerRuntimeSettings,
) -> dict[str, Any]:
    claim = allocation.preparation_claim
    authority = allocation.runtime_volume_authority
    policy = claim.execution_policy
    helper = _supervisor_helper_wire(policy)
    init_source = _docker_init_source_wire(policy)
    keeper_projection = _keeper_projection_wire(
        policy,
        claim.preparation_claim_id,
        authority,
        helper,
        init_source,
    )
    keeper_cgroup_path = run_action_keeper_process_cgroup_path(
        policy,
        _MAXIMUM_CONTAINER_ID,
    )
    mounted_helper = _mounted_helper_wire(
        helper,
        keeper_cgroup_path,
    )
    keeper = _keeper_evidence_wire(
        claim,
        keeper_projection,
        mounted_helper,
    )
    sentinel = _sentinel_evidence_wire(authority)
    volume_evidence = _runtime_volume_evidence_wire(
        authority,
        keeper,
        keeper_cgroup_path,
        sentinel,
    )
    input_slot = _delivery_slot_wire(
        claim.preparation_claim_id,
        authority,
        kind=RunActionPreparedFileKind.INPUT,
        directory_relative_path="input",
        final_file_name="request.blob",
        payload_size_limit_bytes=claim.reservation.request_blob.size_bytes,
    )
    credential_slot = (
        None
        if policy.credential_policy.mode is RunActionCredentialMode.NONE
        else _delivery_slot_wire(
            claim.preparation_claim_id,
            authority,
            kind=RunActionPreparedFileKind.CREDENTIAL,
            directory_relative_path="credential",
            final_file_name="credentials",
            payload_size_limit_bytes=(
                policy.credential_policy.maximum_delivery_size_bytes
            ),
        )
    )
    result_directory = _runtime_directory_wire(
        claim.preparation_claim_id,
        authority,
        RunActionPreparedRuntimeDirectoryKind.RESULT,
    )
    temporary_directory = _runtime_directory_wire(
        claim.preparation_claim_id,
        authority,
        RunActionPreparedRuntimeDirectoryKind.TEMPORARY,
    )
    control_directory = _runtime_directory_wire(
        claim.preparation_claim_id,
        authority,
        RunActionPreparedRuntimeDirectoryKind.CONTROL,
    )
    workspace = claim.reservation.frontier.workspace_before
    workspace_proof = (
        None
        if workspace is None
        else _workspace_proof_wire(
            claim.preparation_claim_id,
            authority,
            workspace,
        )
    )
    layout = _layout_proof_wire(
        authority,
        volume_evidence,
        input_slot,
        credential_slot,
        result_directory,
        temporary_directory,
        control_directory,
        workspace_proof,
        workspace,
    )
    main_projection = _main_projection_wire(
        claim,
        policy,
        authority,
        command,
        runtime_settings,
        helper,
        init_source,
    )
    inert_main = _inert_main_wire(
        claim,
        policy,
        main_projection,
    )
    return _sealed_wire(
        RunActionPreparedExecution,
        prepared_execution_id=_content_identifier(RunActionPreparedExecution),
        preparation_claim=claim.to_dict(),
        runtime_volume_authority=authority.to_dict(),
        runtime_volume_evidence=volume_evidence,
        volume_keeper_evidence=keeper,
        input_delivery_slot=input_slot,
        result_directory=result_directory,
        temporary_directory=temporary_directory,
        control_directory=control_directory,
        credential_delivery_slot=credential_slot,
        workspace_proof=workspace_proof,
        layout_proof=layout,
        inert_container_evidence=inert_main,
    )


def _supervisor_helper_wire(
    policy: DockerRunActionExecutionPolicy,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionSupervisorHelperEvidence,
        supervisor_helper_evidence_id=_content_identifier(
            RunActionSupervisorHelperEvidence
        ),
        helper_authority_id=policy.supervisor_helper_executable_authority_id,
        source_path=policy.supervisor_helper_source_path,
        destination=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        mount_type="bind",
        mount_access=RunActionPreparedMountAccess.READ_ONLY,
        recursive_bind=False,
        file_type="regular",
        owner_user_id=0,
        owner_group_id=0,
        mode=0o755,
        link_count=1,
        file_format="elf",
        dynamic_dependency_count=0,
        elf_interpreter_present=False,
        executable_digest=policy.supervisor_helper_executable_digest,
        mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        device=_MAXIMUM_PHYSICAL_INTEGER,
        inode=_MAXIMUM_PHYSICAL_INTEGER,
    )


def _docker_init_source_wire(
    policy: DockerRunActionExecutionPolicy,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionDockerInitSourceEvidence,
        docker_init_source_evidence_id=_content_identifier(
            RunActionDockerInitSourceEvidence
        ),
        init_authority_id=policy.docker_init_executable_authority_id,
        source_path=policy.docker_init_source_path,
        file_type="regular",
        owner_user_id=0,
        owner_group_id=0,
        mode=0o755,
        link_count=1,
        file_format="elf",
        dynamic_dependency_count=0,
        elf_interpreter_present=False,
        executable_digest=policy.docker_init_executable_digest,
        mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        device=_MAXIMUM_PHYSICAL_INTEGER,
        inode=_MAXIMUM_PHYSICAL_INTEGER,
    )


def _keeper_projection_wire(
    policy: DockerRunActionExecutionPolicy,
    preparation_claim_id: str,
    authority: RunActionRuntimeVolumeAuthority,
    helper: dict[str, Any],
    init_source: dict[str, Any],
) -> dict[str, Any]:
    return _sealed_wire(
        DockerRunActionKeeperCreateInspectProjection,
        keeper_create_inspect_projection_id=_content_identifier(
            DockerRunActionKeeperCreateInspectProjection
        ),
        projection_protocol_version=policy.projection_protocol_version,
        raw_field_schema_id=policy.raw_field_schema_id,
        preparation_claim_id=preparation_claim_id,
        execution_policy=policy.to_dict(),
        volume_authority=authority.to_dict(),
        command_executable=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        command_arguments=("tail", "-f", "/dev/null"),
        helper_evidence=helper,
        docker_init_source_evidence=init_source,
        volume_mount_type="volume",
        volume_mount_destination=RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        volume_mount_access=RunActionPreparedMountAccess.READ_WRITE,
        network_mode="none",
        exact_mount_count=2,
        healthcheck_present=False,
        docker_socket_mounted=False,
        unclassified_raw_field_count=0,
        nonauthoritative_raw_field_count=_MAXIMUM_PHYSICAL_INTEGER,
    )


def _mounted_helper_wire(
    helper: dict[str, Any],
    keeper_cgroup_path: str,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionMountedKeeperHelperEvidence,
        mounted_keeper_helper_evidence_id=_content_identifier(
            RunActionMountedKeeperHelperEvidence
        ),
        source_helper_evidence=helper,
        container_id=_MAXIMUM_CONTAINER_ID,
        process_id=_MAXIMUM_PHYSICAL_INTEGER,
        process_start_time_ticks=_MAXIMUM_PHYSICAL_INTEGER,
        process_cgroup_path=keeper_cgroup_path,
        destination=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        device=_MAXIMUM_PHYSICAL_INTEGER,
        inode=_MAXIMUM_PHYSICAL_INTEGER,
        executable_digest=helper["executable_digest"],
    )


def _keeper_evidence_wire(
    claim: RunActionPreparationClaim,
    keeper_projection: dict[str, Any],
    mounted_helper: dict[str, Any],
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionVolumeKeeperEvidence,
        volume_keeper_evidence_id=_content_identifier(RunActionVolumeKeeperEvidence),
        preparation_claim_id=claim.preparation_claim_id,
        container_id=_MAXIMUM_CONTAINER_ID,
        container_name=preparation_keeper_container_name(claim),
        labels=_contract_tuple_wire(preparation_keeper_container_labels(claim)),
        issued_create_projection=keeper_projection,
        observed_inspect_projection=keeper_projection,
        mounted_helper_evidence=mounted_helper,
        container_status="running",
        process_id=_MAXIMUM_PHYSICAL_INTEGER,
        process_start_time_ticks=_MAXIMUM_PHYSICAL_INTEGER,
        restart_count=0,
        restart_policy_name="no",
        auto_remove=False,
    )


def _sentinel_evidence_wire(
    authority: RunActionRuntimeVolumeAuthority,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionRuntimeVolumeSentinelEvidence,
        runtime_volume_sentinel_evidence_id=_content_identifier(
            RunActionRuntimeVolumeSentinelEvidence
        ),
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=authority.sentinel_relative_path,
        file_type="regular",
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        mode=0o400,
        link_count=1,
        size_bytes=len(authority.generation_nonce),
        content_digest=_MAXIMUM_DIGEST,
        mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        device=_MAXIMUM_PHYSICAL_INTEGER,
        inode=_MAXIMUM_PHYSICAL_INTEGER,
    )


def _runtime_volume_evidence_wire(
    authority: RunActionRuntimeVolumeAuthority,
    keeper: dict[str, Any],
    keeper_cgroup_path: str,
    sentinel: dict[str, Any],
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionRuntimeVolumeEvidence,
        runtime_volume_evidence_id=_content_identifier(RunActionRuntimeVolumeEvidence),
        volume_authority=authority.to_dict(),
        docker_volume_occurrence_digest=_MAXIMUM_DIGEST,
        volume_keeper_evidence_id=keeper["volume_keeper_evidence_id"],
        keeper_container_id=_MAXIMUM_CONTAINER_ID,
        keeper_process_id=_MAXIMUM_PHYSICAL_INTEGER,
        keeper_process_start_time_ticks=_MAXIMUM_PHYSICAL_INTEGER,
        keeper_process_cgroup_path=keeper_cgroup_path,
        root_mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        root_device=_MAXIMUM_PHYSICAL_INTEGER,
        root_inode=_MAXIMUM_PHYSICAL_INTEGER,
        observed_volume_name=authority.volume_name,
        observed_labels=_contract_tuple_wire(authority.labels),
        observed_scope="local",
        observed_driver=authority.driver,
        observed_driver_options=authority.driver_options,
        observed_filesystem_type="tmpfs",
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        observed_owner_user_id=authority.owner_user_id,
        observed_owner_group_id=authority.owner_group_id,
        observed_root_mode=authority.root_mode,
        allocation_block_size_bytes=_MAXIMUM_PHYSICAL_INTEGER,
        effective_block_count=_MAXIMUM_PHYSICAL_INTEGER,
        effective_size_bytes=_MAXIMUM_PHYSICAL_INTEGER,
        effective_inode_limit=_MAXIMUM_PHYSICAL_INTEGER,
        used_block_count=_MAXIMUM_PHYSICAL_INTEGER,
        used_size_bytes=_MAXIMUM_PHYSICAL_INTEGER,
        used_inode_count=_MAXIMUM_PHYSICAL_INTEGER,
        available_block_count=_MAXIMUM_PHYSICAL_INTEGER,
        available_size_bytes=_MAXIMUM_PHYSICAL_INTEGER,
        available_inode_count=_MAXIMUM_PHYSICAL_INTEGER,
        sentinel_evidence=sentinel,
    )


def _delivery_slot_wire(
    preparation_claim_id: str,
    authority: RunActionRuntimeVolumeAuthority,
    *,
    kind: RunActionPreparedFileKind,
    directory_relative_path: str,
    final_file_name: str,
    payload_size_limit_bytes: int,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionPreparedDeliverySlot,
        prepared_delivery_slot_id=_content_identifier(RunActionPreparedDeliverySlot),
        preparation_claim_id=preparation_claim_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        kind=kind,
        directory_relative_path=directory_relative_path,
        final_file_name=final_file_name,
        directory_type="directory",
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        mode=0o700,
        observed_entry_count=0,
        payload_size_limit_bytes=payload_size_limit_bytes,
        mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        device=_MAXIMUM_PHYSICAL_INTEGER,
        inode=_MAXIMUM_PHYSICAL_INTEGER,
    )


def _runtime_directory_wire(
    preparation_claim_id: str,
    authority: RunActionRuntimeVolumeAuthority,
    kind: RunActionPreparedRuntimeDirectoryKind,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionPreparedRuntimeDirectory,
        prepared_runtime_directory_id=_content_identifier(
            RunActionPreparedRuntimeDirectory
        ),
        preparation_claim_id=preparation_claim_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        kind=kind,
        directory_relative_path=kind.value,
        directory_type="directory",
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        mode=0o700,
        observed_entry_count=0,
        mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        device=_MAXIMUM_PHYSICAL_INTEGER,
        inode=_MAXIMUM_PHYSICAL_INTEGER,
    )


def _workspace_proof_wire(
    preparation_claim_id: str,
    authority: RunActionRuntimeVolumeAuthority,
    workspace: RunActionWorkspaceBinding,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionPreparedWorkspaceProof,
        prepared_workspace_proof_id=_content_identifier(
            RunActionPreparedWorkspaceProof
        ),
        preparation_claim_id=preparation_claim_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        volume_subpath="workspace",
        workspace_binding=workspace.to_dict(),
        observed_source_tree_digest=workspace.source_tree_digest,
        observed_git_closure_digest=workspace.git_closure_digest,
        observed_source_entry_count=workspace.source_entry_count,
        observed_source_size_bytes=workspace.source_size_bytes,
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        root_mode=0o700,
        unexpected_entry_count=0,
        mount_id=_MAXIMUM_PHYSICAL_INTEGER,
        device=_MAXIMUM_PHYSICAL_INTEGER,
        inode=_MAXIMUM_PHYSICAL_INTEGER,
    )


def _layout_proof_wire(
    authority: RunActionRuntimeVolumeAuthority,
    volume_evidence: dict[str, Any],
    input_slot: dict[str, Any],
    credential_slot: dict[str, Any] | None,
    result_directory: dict[str, Any],
    temporary_directory: dict[str, Any],
    control_directory: dict[str, Any],
    workspace_proof: dict[str, Any] | None,
    workspace: RunActionWorkspaceBinding | None,
) -> dict[str, Any]:
    directories = tuple(
        sorted(
            (
                "control",
                "input",
                "result",
                "temporary",
                *(("credential",) if credential_slot is not None else ()),
                *(("workspace",) if workspace_proof is not None else ()),
            )
        )
    )
    delivery_slot_ids = (
        (input_slot["prepared_delivery_slot_id"],)
        if credential_slot is None
        else tuple(
            sorted(
                (
                    input_slot["prepared_delivery_slot_id"],
                    credential_slot["prepared_delivery_slot_id"],
                )
            )
        )
    )
    runtime_directory_ids = tuple(
        sorted(
            (
                result_directory["prepared_runtime_directory_id"],
                temporary_directory["prepared_runtime_directory_id"],
                control_directory["prepared_runtime_directory_id"],
            )
        )
    )
    workspace_size = 0 if workspace is None else workspace.source_size_bytes
    workspace_entries = 0 if workspace is None else workspace.source_entry_count
    return _sealed_wire(
        RunActionRuntimeVolumeLayoutProof,
        runtime_volume_layout_proof_id=_content_identifier(
            RunActionRuntimeVolumeLayoutProof
        ),
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        runtime_volume_evidence_id=volume_evidence["runtime_volume_evidence_id"],
        generation_nonce=authority.generation_nonce,
        empty_size_bytes=0,
        empty_entry_count=0,
        directory_relative_paths=directories,
        prepared_delivery_slot_ids=delivery_slot_ids,
        prepared_runtime_directory_ids=runtime_directory_ids,
        prepared_workspace_proof_id=(
            None
            if workspace_proof is None
            else workspace_proof["prepared_workspace_proof_id"]
        ),
        logical_content_size_bytes=len(authority.generation_nonce) + workspace_size,
        logical_entry_count=len(directories) + 1 + workspace_entries,
        observed_used_size_bytes=_MAXIMUM_PHYSICAL_INTEGER,
        observed_used_inode_count=_MAXIMUM_PHYSICAL_INTEGER,
        unexpected_entry_count=0,
    )


def _main_projection_wire(
    claim: RunActionPreparationClaim,
    policy: DockerRunActionExecutionPolicy,
    authority: RunActionRuntimeVolumeAuthority,
    command: DockerRunActionCommand,
    runtime_settings: DockerRuntimeSettings,
    helper: dict[str, Any],
    init_source: dict[str, Any],
) -> dict[str, Any]:
    executable, arguments = main_barrier_command(
        command,
        authority.generation_nonce,
        runtime_settings,
    )
    mounts = preparation_main_mounts(
        claim,
        authority,
    )
    return _sealed_wire(
        DockerRunActionCreateInspectProjection,
        create_inspect_projection_id=_content_identifier(
            DockerRunActionCreateInspectProjection
        ),
        projection_protocol_version=policy.projection_protocol_version,
        raw_field_schema_id=policy.raw_field_schema_id,
        execution_policy=policy.to_dict(),
        supervisor_helper_evidence=helper,
        docker_init_source_evidence=init_source,
        barrier_protocol_version=RUN_ACTION_BARRIER_PROTOCOL_VERSION,
        barrier_poll_interval_seconds=(
            runtime_settings.run_action_barrier_poll_interval_seconds
        ),
        barrier_generation_nonce=authority.generation_nonce,
        command_executable=executable,
        command_arguments=arguments,
        mounts=_contract_tuple_wire(mounts),
        exact_mount_count=len(mounts) + 1,
        unclassified_raw_field_count=0,
        nonauthoritative_raw_field_count=_MAXIMUM_PHYSICAL_INTEGER,
    )


def _inert_main_wire(
    claim: RunActionPreparationClaim,
    policy: DockerRunActionExecutionPolicy,
    projection: dict[str, Any],
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionInertContainerEvidence,
        inert_container_evidence_id=_content_identifier(
            RunActionInertContainerEvidence
        ),
        preparation_claim_id=claim.preparation_claim_id,
        container_id=_MAXIMUM_CONTAINER_ID,
        container_name=preparation_container_name(claim),
        labels=_contract_tuple_wire(preparation_container_labels(claim)),
        image_authority_id=policy.image_authority.image_authority_id,
        docker_runtime_settings_digest=policy.docker_runtime_settings_digest,
        issued_create_projection=projection,
        observed_inspect_projection=projection,
        container_status="created",
        process_id=0,
        restart_count=0,
        started_at=_ZERO_DOCKER_TIMESTAMP,
        finished_at=_ZERO_DOCKER_TIMESTAMP,
        restart_policy_name="no",
        auto_remove=False,
        network_mode="none",
        healthcheck_present=False,
        volume_plugin_mount_count=0,
        docker_socket_mounted=False,
    )


def _contract_tuple_wire(
    contracts: tuple[StrictContract, ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(contract.to_dict() for contract in contracts)


def _content_identifier(contract_type: type[StrictContract]) -> str:
    namespace = contract_type.CONTENT_NAMESPACE
    if not isinstance(namespace, str) or contract_type.IDENTITY_FIELD is None:
        raise RunActionPreparedEnvelopeError(
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
        raise RunActionPreparedEnvelopeError(
            f"{contract_type.__name__} envelope fields changed; "
            f"missing={missing}, unknown={unknown}"
        )
    return values


__all__ = [
    "RunActionPreparedEnvelopeError",
    "prepared_execution_event_size_bound",
]
