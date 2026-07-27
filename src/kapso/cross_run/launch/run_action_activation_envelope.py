"""Pre-delivery canonical wire envelope for run-action event 5."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from kapso.cross_run.canonical import canonical_json_bytes, require_content_id
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE,
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionActivatedFileObservation,
    RunActionActivatedRuntimeDirectoryObservation,
    RunActionActivatedSentinelObservation,
    RunActionActivatedWorkspaceObservation,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionPreparedDeliverySlot,
    RunActionPreparedExecution,
    RunActionPreparedFile,
    RunActionPreparedFileKind,
    RunActionPreparedRuntimeDirectory,
    RunActionPreparedWorkspaceProof,
    RunActionRuntimeVolumeEvidence,
)

_MAXIMUM_PHYSICAL_INTEGER = RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
_RUNTIME_VOLUME_EVIDENCE_FIELDS = (
    "runtime_volume_evidence_id",
    "volume_authority",
    "docker_volume_occurrence_digest",
    "volume_keeper_evidence_id",
    "keeper_container_id",
    "keeper_process_id",
    "keeper_process_start_time_ticks",
    "keeper_process_cgroup_path",
    "root_mount_id",
    "root_device",
    "root_inode",
    "observed_volume_name",
    "observed_labels",
    "observed_scope",
    "observed_driver",
    "observed_driver_options",
    "observed_filesystem_type",
    "observed_mount_flags",
    "observed_owner_user_id",
    "observed_owner_group_id",
    "observed_root_mode",
    "allocation_block_size_bytes",
    "effective_block_count",
    "effective_size_bytes",
    "effective_inode_limit",
    "used_block_count",
    "used_size_bytes",
    "used_inode_count",
    "available_block_count",
    "available_size_bytes",
    "available_inode_count",
    "sentinel_evidence",
)


class RunActionActivationEnvelopeError(ValueError):
    """Event 5 cannot be bounded from its exact pre-delivery authority."""


def activation_execution_event_size_bound(
    *,
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    predecessor_event_id: str,
) -> int:
    """Return a sound canonical byte bound for the exact future event 5."""

    if (
        type(prepared_execution) is not RunActionPreparedExecution
        or type(spawn_commit) is not RunActionSpawnCommit
    ):
        raise RunActionActivationEnvelopeError(
            "activation event envelope requires exact prepared and spawn authority"
        )
    predecessor = require_content_id(
        predecessor_event_id,
        "activation event predecessor",
    )
    if predecessor.split(":sha256:", 1)[0] != RunActionExecutionEvent.CONTENT_NAMESPACE:
        raise RunActionActivationEnvelopeError(
            "activation event envelope predecessor is not a run-action event"
        )
    _require_spawn_join(prepared_execution, spawn_commit)
    receipt = _activation_receipt_wire(prepared_execution, spawn_commit)
    event = _sealed_wire(
        RunActionExecutionEvent,
        event_id=_content_identifier(RunActionExecutionEvent),
        event_number=5,
        predecessor_event_id=predecessor,
        event_kind=RunActionExecutionEventKind.ACTIVATION_COMMITTED,
        reservation=prepared_execution.preparation_claim.reservation.to_dict(),
        preparation_allocation=None,
        prepared_execution=None,
        spawn_commit=None,
        activation_revalidation_receipt=receipt,
        provider_termination_receipt=None,
        result_receipt=None,
        result_decision=None,
        acceptance=None,
        workspace_after=None,
    )
    return len(canonical_json_bytes(event))


def _require_spawn_join(
    prepared: RunActionPreparedExecution,
    spawn: RunActionSpawnCommit,
) -> None:
    reservation = prepared.preparation_claim.reservation
    if (
        spawn.reservation_id != reservation.reservation_id
        or spawn.prepared_execution_id != prepared.prepared_execution_id
        or spawn.provider_execution_id != prepared.inert_container_evidence.container_id
        or spawn.boundary_identity != reservation.intent.boundary_identity
        or spawn.security_observation_id != reservation.frontier.security_observation_id
    ):
        raise RunActionActivationEnvelopeError(
            "activation event envelope spawn differs from prepared authority"
        )


def _activation_receipt_wire(
    prepared: RunActionPreparedExecution,
    spawn: RunActionSpawnCommit,
) -> dict[str, Any]:
    credential_slot = prepared.credential_delivery_slot
    credential_required = (
        prepared.preparation_claim.execution_policy.credential_policy.mode
        is RunActionCredentialMode.SUPERVISOR_FILE
    )
    if (credential_slot is not None) != credential_required:
        raise RunActionActivationEnvelopeError(
            "activation event envelope credential topology is invalid"
        )
    return _sealed_wire(
        RunActionActivationRevalidationReceipt,
        activation_revalidation_receipt_id=_content_identifier(
            RunActionActivationRevalidationReceipt
        ),
        prepared_execution=prepared.to_dict(),
        spawn_commit=spawn.to_dict(),
        reobserved_volume_evidence=_reobserved_volume_wire(
            prepared.runtime_volume_evidence
        ),
        reobserved_keeper_evidence=prepared.volume_keeper_evidence.to_dict(),
        reobserved_container_evidence=prepared.inert_container_evidence.to_dict(),
        activated_workspace_observation=_activated_workspace_wire(
            prepared.workspace_proof,
            spawn,
        ),
        activated_runtime_directory_observations=(
            _activated_runtime_directory_wire(
                prepared.control_directory,
                spawn,
            ),
            _activated_runtime_directory_wire(
                prepared.temporary_directory,
                spawn,
            ),
        ),
        activated_sentinel_observation=_activated_sentinel_wire(
            prepared,
            spawn,
        ),
        input_file_observation=_activated_delivery_file_wire(
            prepared.input_delivery_slot,
            spawn,
            size_bytes=(prepared.preparation_claim.reservation.request_blob.size_bytes),
            content_digest=prepared.preparation_claim.reservation.request_blob.digest,
            content_authority_id=(
                prepared.preparation_claim.reservation.request_blob.request_blob_id
            ),
        ),
        result_file_observation=_activated_result_file_wire(
            prepared.result_file,
            prepared.result_directory,
            spawn,
        ),
        credential_file_observation=(
            None
            if credential_slot is None
            else _activated_delivery_file_wire(
                credential_slot,
                spawn,
                size_bytes=credential_slot.payload_size_limit_bytes,
                content_digest=None,
                content_authority_id=_content_identifier_from_namespace(
                    RUN_ACTION_CREDENTIAL_LEASE_AUTHORITY_NAMESPACE
                ),
            )
        ),
    )


def _reobserved_volume_wire(
    prepared: RunActionRuntimeVolumeEvidence,
) -> dict[str, Any]:
    actual_fields = tuple(
        field.name for field in fields(RunActionRuntimeVolumeEvidence)
    )
    if actual_fields != _RUNTIME_VOLUME_EVIDENCE_FIELDS:
        raise RunActionActivationEnvelopeError(
            "RunActionRuntimeVolumeEvidence envelope fields changed"
        )
    wire = prepared.to_dict()
    for field_name in (
        "used_block_count",
        "used_size_bytes",
        "used_inode_count",
        "available_block_count",
        "available_size_bytes",
        "available_inode_count",
    ):
        wire[field_name] = _MAXIMUM_PHYSICAL_INTEGER
    wire["runtime_volume_evidence_id"] = _content_identifier(
        RunActionRuntimeVolumeEvidence
    )
    return wire


def _activated_workspace_wire(
    proof: RunActionPreparedWorkspaceProof | None,
    spawn: RunActionSpawnCommit,
) -> dict[str, Any] | None:
    if proof is None:
        return None
    return _sealed_wire(
        RunActionActivatedWorkspaceObservation,
        activated_workspace_observation_id=_content_identifier(
            RunActionActivatedWorkspaceObservation
        ),
        spawn_commit_id=spawn.spawn_commit_id,
        prepared_workspace_proof_id=proof.prepared_workspace_proof_id,
        runtime_volume_authority_id=proof.runtime_volume_authority_id,
        generation_nonce=proof.generation_nonce,
        source_tree_digest=proof.observed_source_tree_digest,
        git_closure_digest=proof.observed_git_closure_digest,
        source_entry_count=proof.observed_source_entry_count,
        source_size_bytes=proof.observed_source_size_bytes,
        owner_user_id=proof.owner_user_id,
        owner_group_id=proof.owner_group_id,
        root_mode=proof.root_mode,
        mount_id=proof.mount_id,
        device=proof.device,
        inode=proof.inode,
    )


def _activated_runtime_directory_wire(
    prepared: RunActionPreparedRuntimeDirectory,
    spawn: RunActionSpawnCommit,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionActivatedRuntimeDirectoryObservation,
        activated_runtime_directory_observation_id=_content_identifier(
            RunActionActivatedRuntimeDirectoryObservation
        ),
        spawn_commit_id=spawn.spawn_commit_id,
        prepared_runtime_directory_id=prepared.prepared_runtime_directory_id,
        runtime_volume_authority_id=prepared.runtime_volume_authority_id,
        generation_nonce=prepared.generation_nonce,
        kind=prepared.kind,
        directory_relative_path=prepared.directory_relative_path,
        directory_type=prepared.directory_type,
        owner_user_id=prepared.owner_user_id,
        owner_group_id=prepared.owner_group_id,
        mode=prepared.mode,
        observed_entry_count=0,
        mount_id=prepared.mount_id,
        device=prepared.device,
        inode=prepared.inode,
    )


def _activated_sentinel_wire(
    prepared: RunActionPreparedExecution,
    spawn: RunActionSpawnCommit,
) -> dict[str, Any]:
    sentinel = prepared.runtime_volume_evidence.sentinel_evidence
    return _sealed_wire(
        RunActionActivatedSentinelObservation,
        activated_sentinel_observation_id=_content_identifier(
            RunActionActivatedSentinelObservation
        ),
        spawn_commit_id=spawn.spawn_commit_id,
        prepared_sentinel_evidence_id=(sentinel.runtime_volume_sentinel_evidence_id),
        runtime_volume_authority_id=sentinel.runtime_volume_authority_id,
        generation_nonce=sentinel.generation_nonce,
        relative_path=sentinel.relative_path,
        file_type=sentinel.file_type,
        owner_user_id=sentinel.owner_user_id,
        owner_group_id=sentinel.owner_group_id,
        mode=sentinel.mode,
        link_count=sentinel.link_count,
        size_bytes=sentinel.size_bytes,
        content_digest=sentinel.content_digest,
        mount_id=sentinel.mount_id,
        device=sentinel.device,
        inode=sentinel.inode,
    )


def _activated_delivery_file_wire(
    prepared: RunActionPreparedDeliverySlot,
    spawn: RunActionSpawnCommit,
    *,
    size_bytes: int,
    content_digest: str | None,
    content_authority_id: str,
) -> dict[str, Any]:
    return _sealed_wire(
        RunActionActivatedFileObservation,
        activated_file_observation_id=_content_identifier(
            RunActionActivatedFileObservation
        ),
        spawn_commit_id=spawn.spawn_commit_id,
        prepared_parent_authority_id=prepared.prepared_delivery_slot_id,
        prepared_file_id=None,
        parent_mount_id=prepared.mount_id,
        parent_device=prepared.device,
        parent_inode=prepared.inode,
        runtime_volume_authority_id=prepared.runtime_volume_authority_id,
        generation_nonce=prepared.generation_nonce,
        kind=prepared.kind,
        relative_path=(
            f"{prepared.directory_relative_path}/{prepared.final_file_name}"
        ),
        file_type="regular",
        owner_user_id=prepared.owner_user_id,
        owner_group_id=prepared.owner_group_id,
        mode=0o400,
        link_count=1,
        size_bytes=size_bytes,
        mount_id=prepared.mount_id,
        device=prepared.device,
        inode=_MAXIMUM_PHYSICAL_INTEGER,
        content_digest=content_digest,
        content_authority_id=content_authority_id,
    )


def _activated_result_file_wire(
    prepared: RunActionPreparedFile,
    parent: RunActionPreparedRuntimeDirectory,
    spawn: RunActionSpawnCommit,
) -> dict[str, Any]:
    if prepared.kind is not RunActionPreparedFileKind.RESULT:
        raise RunActionActivationEnvelopeError(
            "activation event envelope result authority is invalid"
        )
    return _sealed_wire(
        RunActionActivatedFileObservation,
        activated_file_observation_id=_content_identifier(
            RunActionActivatedFileObservation
        ),
        spawn_commit_id=spawn.spawn_commit_id,
        prepared_parent_authority_id=parent.prepared_runtime_directory_id,
        prepared_file_id=prepared.prepared_file_id,
        parent_mount_id=parent.mount_id,
        parent_device=parent.device,
        parent_inode=parent.inode,
        runtime_volume_authority_id=prepared.runtime_volume_authority_id,
        generation_nonce=prepared.generation_nonce,
        kind=prepared.kind,
        relative_path=prepared.relative_path,
        file_type=prepared.file_type,
        owner_user_id=prepared.owner_user_id,
        owner_group_id=prepared.owner_group_id,
        mode=prepared.mode,
        link_count=prepared.link_count,
        size_bytes=prepared.size_bytes,
        mount_id=prepared.mount_id,
        device=prepared.device,
        inode=prepared.inode,
        content_digest=None,
        content_authority_id=None,
    )


def _content_identifier(contract_type: type[StrictContract]) -> str:
    namespace = contract_type.CONTENT_NAMESPACE
    if not isinstance(namespace, str) or contract_type.IDENTITY_FIELD is None:
        raise RunActionActivationEnvelopeError(
            f"{contract_type.__name__} lacks a content identity"
        )
    return _content_identifier_from_namespace(namespace)


def _content_identifier_from_namespace(namespace: str) -> str:
    return f"{namespace}:sha256:{'f' * 64}"


def _sealed_wire(
    contract_type: type[StrictContract],
    **values: Any,
) -> dict[str, Any]:
    expected = tuple(field.name for field in fields(contract_type))
    missing = tuple(sorted(set(expected) - set(values)))
    unknown = tuple(sorted(set(values) - set(expected)))
    if missing or unknown:
        raise RunActionActivationEnvelopeError(
            f"{contract_type.__name__} envelope fields changed; "
            f"missing={missing}, unknown={unknown}"
        )
    return values


__all__ = [
    "RunActionActivationEnvelopeError",
    "activation_execution_event_size_bound",
]
