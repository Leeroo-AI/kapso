"""Pre-delivery canonical wire envelope for credential-expiry event 7."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from kapso.cross_run.canonical import canonical_json_bytes
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_activation_envelope import (
    activation_revalidation_receipt_wire_bound,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_credential_contracts import (
    maximum_credential_retirement_intent,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionPreReleaseMainLossObservation,
    RunActionPreReleaseMainTerminalObservation,
    RunActionPreReleaseTerminalContainerObservation,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)

_MAXIMUM_DIGEST = f"sha256:{'f' * 64}"
_MAXIMUM_BOOT_ID = "ffffffff-ffff-4fff-afff-ffffffffffff"
_MAXIMUM_DOCKER_TIMESTAMP = "9999-12-31T23:59:59.999999999Z"


class RunActionCredentialExpiryEnvelopeError(ValueError):
    """Credential-expiry terminal evidence cannot be bounded before delivery."""


def credential_expiry_termination_event_size_bound(
    *,
    reservation: RunActionReservation,
    preparation_allocation: RunActionPreparationAllocation,
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
) -> int:
    """Return the widest canonical event-7 physical-evidence branch."""

    if (
        type(reservation) is not RunActionReservation
        or type(preparation_allocation) is not RunActionPreparationAllocation
        or type(prepared_execution) is not RunActionPreparedExecution
        or type(spawn_commit) is not RunActionSpawnCommit
        or preparation_allocation.preparation_claim
        != prepared_execution.preparation_claim
        or preparation_allocation.runtime_volume_authority
        != prepared_execution.runtime_volume_authority
        or reservation != preparation_allocation.preparation_claim.reservation
    ):
        raise RunActionCredentialExpiryEnvelopeError(
            "credential-expiry envelope lacks one exact prepared spawn"
        )
    activation = activation_revalidation_receipt_wire_bound(
        prepared_execution,
        spawn_commit,
    )
    intent = maximum_credential_retirement_intent(
        prepared_execution,
        spawn_commit,
    )
    loss = _loss_observation_wire(
        preparation_allocation,
        prepared_execution,
        spawn_commit,
        activation,
        intent.activation_event_id,
    )
    terminal = _terminal_observation_wire(
        preparation_allocation,
        prepared_execution,
        spawn_commit,
        activation,
        intent.activation_event_id,
    )
    return max(
        _termination_event_size(reservation, intent.to_dict(), loss=loss),
        _termination_event_size(reservation, intent.to_dict(), terminal=terminal),
    )


def _termination_event_size(
    reservation: RunActionReservation,
    intent: dict[str, Any],
    *,
    loss: dict[str, Any] | None = None,
    terminal: dict[str, Any] | None = None,
) -> int:
    receipt = _sealed_wire(
        RunActionProviderTerminationReceipt,
        provider_termination_receipt_id=_content_identifier(
            RunActionProviderTerminationReceipt
        ),
        disposition=RunActionProviderTerminationDisposition.INTERRUPTED,
        reason=RunActionProviderTerminationReason.CREDENTIAL_EXPIRED,
        activation_event_id=intent["activation_event_id"],
        workload_release_adoption=None,
        terminal_observation=terminal,
        timeout_directive_publication=None,
        pre_release_main_loss_observation=loss,
        credential_retirement_intent=intent,
    )
    event = _sealed_wire(
        RunActionExecutionEvent,
        event_id=_content_identifier(RunActionExecutionEvent),
        event_number=7,
        predecessor_event_id=_content_identifier(RunActionExecutionEvent),
        event_kind=RunActionExecutionEventKind.PROVIDER_TERMINATED,
        reservation=reservation.to_dict(),
        preparation_allocation=None,
        prepared_execution=None,
        spawn_commit=None,
        activation_revalidation_receipt=None,
        credential_retirement_intent=None,
        provider_termination_receipt=receipt,
        result_receipt=None,
        result_decision=None,
        acceptance=None,
        workspace_after=None,
    )
    return len(canonical_json_bytes(event))


def _loss_observation_wire(
    allocation: RunActionPreparationAllocation,
    prepared: RunActionPreparedExecution,
    spawn: RunActionSpawnCommit,
    activation: dict[str, Any],
    activation_event_id: str,
) -> dict[str, Any]:
    control = prepared.control_directory
    return _sealed_wire(
        RunActionPreReleaseMainLossObservation,
        pre_release_main_loss_observation_id=_content_identifier(
            RunActionPreReleaseMainLossObservation
        ),
        activation_event_id=activation_event_id,
        preparation_allocation=allocation.to_dict(),
        activation_revalidation_receipt=activation,
        host_boot_id=_MAXIMUM_BOOT_ID,
        observed_before_boottime_nanoseconds=(RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER),
        first_complete_inventory_digest=_MAXIMUM_DIGEST,
        reobserved_volume_evidence=activation["reobserved_volume_evidence"],
        reobserved_keeper_evidence=activation["reobserved_keeper_evidence"],
        second_complete_inventory_digest=_MAXIMUM_DIGEST,
        observed_after_boottime_nanoseconds=RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
        observed_runtime_volume_names=(prepared.runtime_volume_authority.volume_name,),
        observed_keeper_container_ids=(prepared.volume_keeper_evidence.container_id,),
        observed_main_container_ids=(),
        missing_provider_execution_id=spawn.provider_execution_id,
        control_mount_id=control.mount_id,
        control_device=control.device,
        control_inode=control.inode,
        control_entry_count=0,
        control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
    )


def _terminal_observation_wire(
    allocation: RunActionPreparationAllocation,
    prepared: RunActionPreparedExecution,
    spawn: RunActionSpawnCommit,
    activation: dict[str, Any],
    activation_event_id: str,
) -> dict[str, Any]:
    control = prepared.control_directory
    authority = prepared.runtime_volume_authority
    terminal = _sealed_wire(
        RunActionPreReleaseTerminalContainerObservation,
        pre_release_terminal_container_observation_id=_content_identifier(
            RunActionPreReleaseTerminalContainerObservation
        ),
        prepared_execution_id=prepared.prepared_execution_id,
        spawn_commit_id=spawn.spawn_commit_id,
        provider_execution_id=spawn.provider_execution_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        activation_revalidation_receipt_id=(
            activation["activation_revalidation_receipt_id"]
        ),
        observed_inspect_projection=(
            prepared.inert_container_evidence.issued_create_projection.to_dict()
        ),
        complete_inspection_digest=_MAXIMUM_DIGEST,
        container_status="exited",
        process_id=0,
        restart_count=0,
        paused=False,
        restarting=False,
        dead=False,
        started_at=_MAXIMUM_DOCKER_TIMESTAMP,
        finished_at=_MAXIMUM_DOCKER_TIMESTAMP,
        exit_code=255,
        oom_killed=False,
        state_error="",
    )
    return _sealed_wire(
        RunActionPreReleaseMainTerminalObservation,
        pre_release_main_terminal_observation_id=_content_identifier(
            RunActionPreReleaseMainTerminalObservation
        ),
        activation_event_id=activation_event_id,
        preparation_allocation=allocation.to_dict(),
        activation_revalidation_receipt=activation,
        host_boot_id=_MAXIMUM_BOOT_ID,
        observed_before_boottime_nanoseconds=(RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER),
        first_complete_inventory_digest=_MAXIMUM_DIGEST,
        reobserved_volume_evidence=activation["reobserved_volume_evidence"],
        reobserved_keeper_evidence=activation["reobserved_keeper_evidence"],
        terminal_container_observation=terminal,
        second_complete_inventory_digest=_MAXIMUM_DIGEST,
        observed_after_boottime_nanoseconds=RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
        observed_runtime_volume_names=(authority.volume_name,),
        observed_keeper_container_ids=(prepared.volume_keeper_evidence.container_id,),
        observed_main_container_ids=(spawn.provider_execution_id,),
        control_mount_id=control.mount_id,
        control_device=control.device,
        control_inode=control.inode,
        control_entry_count=0,
        control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
    )


def _content_identifier(contract_type: type[StrictContract]) -> str:
    namespace = contract_type.CONTENT_NAMESPACE
    if not isinstance(namespace, str) or contract_type.IDENTITY_FIELD is None:
        raise RunActionCredentialExpiryEnvelopeError(
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
        raise RunActionCredentialExpiryEnvelopeError(
            f"{contract_type.__name__} envelope fields changed; "
            f"missing={missing}, unknown={unknown}"
        )
    return values


__all__ = [
    "credential_expiry_termination_event_size_bound",
    "RunActionCredentialExpiryEnvelopeError",
]
