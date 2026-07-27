"""Pure graph join for one adopted release, terminal, and captured result."""

from __future__ import annotations

from kapso.cross_run.launch.run_action_contracts import RunFrontierWorkspaceAccess
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseAdoption,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionActivationRevalidationReceipt,
    RunActionResultCaptureReceipt,
    RunActionTerminalObservation,
    run_action_runtime_volume_occurrence_matches,
)


def run_action_terminal_result_evidence_matches(
    terminal: RunActionTerminalObservation,
    capture: RunActionResultCaptureReceipt,
    activation: RunActionActivationRevalidationReceipt,
    workload_release_adoption: RunActionWorkloadReleaseAdoption,
) -> bool:
    """Join capture to the exact adopted release and pre-start activation."""

    if (
        type(terminal) is not RunActionTerminalObservation
        or type(capture) is not RunActionResultCaptureReceipt
        or type(activation) is not RunActionActivationRevalidationReceipt
        or type(workload_release_adoption) is not RunActionWorkloadReleaseAdoption
    ):
        return False
    release = workload_release_adoption.workload_release_receipt
    resolved = release.resolved_workload_observation
    running = resolved.running_container_observation
    if resolved.activation_revalidation_receipt != activation:
        return False
    prepared = activation.prepared_execution
    spawn = activation.spawn_commit
    prepared_result_directory = prepared.result_directory
    activation_volume = activation.reobserved_volume_evidence
    capture_volume = capture.reobserved_volume_evidence
    result_allocation_size_bytes = (
        (capture.size_bytes + capture_volume.allocation_block_size_bytes - 1)
        // capture_volume.allocation_block_size_bytes
        * capture_volume.allocation_block_size_bytes
    )
    result_allocation_block_count = (
        result_allocation_size_bytes // capture_volume.allocation_block_size_bytes
    )
    return (
        terminal.activation_revalidation_receipt_id
        == activation.activation_revalidation_receipt_id
        and terminal.workload_release_adoption_id
        == workload_release_adoption.workload_release_adoption_id
        and terminal.exit_code == 0
        and terminal.oom_killed is False
        and terminal.prepared_execution_id == prepared.prepared_execution_id
        and terminal.spawn_commit_id == spawn.spawn_commit_id
        and terminal.provider_execution_id == spawn.provider_execution_id
        and terminal.provider_execution_id == running.container_id
        and terminal.runtime_volume_authority_id
        == prepared.runtime_volume_authority.runtime_volume_authority_id
        and terminal.generation_nonce
        == prepared.runtime_volume_authority.generation_nonce
        and terminal.observed_inspect_projection
        == prepared.inert_container_evidence.issued_create_projection
        and terminal.observed_inspect_projection == running.observed_inspect_projection
        and terminal.started_at == running.started_at
        and capture.terminal_observation_id == terminal.terminal_observation_id
        and capture.prepared_parent_authority_id
        == prepared_result_directory.prepared_runtime_directory_id
        and capture.parent_mount_id == prepared_result_directory.mount_id
        and capture.parent_device == prepared_result_directory.device
        and capture.parent_inode == prepared_result_directory.inode
        and capture.runtime_volume_authority_id
        == prepared.runtime_volume_authority.runtime_volume_authority_id
        and run_action_runtime_volume_occurrence_matches(
            capture_volume,
            prepared.runtime_volume_evidence,
        )
        and capture.prepared_sentinel_evidence_id
        == (
            prepared.runtime_volume_evidence.sentinel_evidence.runtime_volume_sentinel_evidence_id
        )
        and capture.generation_nonce
        == prepared.runtime_volume_authority.generation_nonce
        and capture.relative_path == "result/result.blob"
        and capture.file_type == "regular"
        and capture.owner_user_id == prepared_result_directory.owner_user_id
        and capture.owner_group_id == prepared_result_directory.owner_group_id
        and capture.mode == 0o600
        and capture.link_count == 1
        and capture.mount_id == prepared_result_directory.mount_id
        and capture.device == prepared_result_directory.device
        and capture.inode != prepared_result_directory.inode
        and capture.size_bytes
        <= prepared.preparation_claim.execution_policy.supervisor_limits.result_size_bytes
        and (
            prepared.preparation_claim.reservation.intent.workspace_access
            is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
            or (
                capture_volume.used_size_bytes
                >= activation_volume.used_size_bytes + 2 * result_allocation_size_bytes
                and capture_volume.used_block_count
                >= activation_volume.used_block_count
                + 2 * result_allocation_block_count
                and capture_volume.used_inode_count
                >= activation_volume.used_inode_count + 2
                and capture_volume.available_block_count
                <= activation_volume.available_block_count
                - 2 * result_allocation_block_count
                and capture_volume.available_size_bytes
                <= activation_volume.available_size_bytes
                - 2 * result_allocation_size_bytes
                and capture_volume.available_inode_count
                <= activation_volume.available_inode_count - 2
            )
        )
    )


__all__ = ["run_action_terminal_result_evidence_matches"]
