"""Trusted resolution of one exact naturally terminal released workload."""

from __future__ import annotations

import os
from contextlib import ExitStack

from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import observe_runtime_volume
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
    _RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionProviderResult,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    open_run_action_release_inspection,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    capture_run_action_result_file,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionResultCaptureReceipt,
    RunActionSupervisorHelperEvidence,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_host_boot_id,
)
from kapso.cross_run.launch.run_action_terminal_inspection import (
    inspect_run_action_terminal,
    reinspect_run_action_terminal,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings


class RunActionNaturalTerminalError(RuntimeError):
    """A natural terminal occurrence lacks one exact released outcome."""


def resolve_run_action_natural_terminal_once(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionContinuationOutcome:
    """Resolve result, OOM, nonzero exit, or exact empty result in one leaf."""

    if (
        type(capability) is not RunActionCommittedContinuationCapability
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(command) is not DockerRunActionCommand
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
        or resource_manager.runtime_settings != docker_settings
    ):
        raise RunActionNaturalTerminalError(
            "natural terminal inputs lack exact configured authority"
        )
    query = capability.query
    adoption = query.workload_release_adoption
    if (
        adoption is None
        or query.control_directory_topology
        is not RunActionControlDirectoryTopology.RELEASED
        or query.timeout_directive_publication is not None
    ):
        raise RunActionNaturalTerminalError(
            "natural terminal requires one exact released occurrence"
        )
    prepared = query.prepared_execution
    projection = prepared.inert_container_evidence.issued_create_projection
    if (
        command.command_template_id
        != prepared.preparation_claim.execution_policy.command_template_id
        or helper_evidence != projection.supervisor_helper_evidence
        or init_source_evidence != projection.docker_init_source_evidence
    ):
        raise RunActionNaturalTerminalError(
            "natural terminal inputs differ from durable execution authority"
        )
    retained_terminal = reinspect_run_action_terminal(
        capability=capability,
        resource_manager=resource_manager,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )
    with ExitStack() as descriptors:
        proc_root_descriptor = os.open(
            "/proc",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, proc_root_descriptor)
        release_inspection = open_run_action_release_inspection(
            activation_event=query.activation_event,
            launch_settings=launch_settings,
        )
        descriptors.callback(release_inspection.close)
        if (
            release_inspection.topology
            is not RunActionControlDirectoryTopology.RELEASED
            or release_inspection.adoption != adoption
            or read_run_action_host_boot_id(proc_root_descriptor)
            != adoption.workload_release_receipt.host_boot_id
        ):
            raise RunActionNaturalTerminalError(
                "natural terminal differs from its retained release occurrence"
            )
        inventory = resource_manager.observe(query.preparation_allocation)
        if (
            inventory.volume_inspection_digest is None
            or inventory.keeper_container_id
            != prepared.volume_keeper_evidence.container_id
            or inventory.main_container_id != query.spawn_commit.provider_execution_id
        ):
            raise RunActionNaturalTerminalError(
                "natural terminal lacks its exact Docker resource graph"
            )
        capture_receipt: RunActionResultCaptureReceipt | None = None
        result_payload: bytes | None = None
        if not retained_terminal.oom_killed and retained_terminal.exit_code == 0:
            volume = observe_runtime_volume(
                resource_manager.inspect_volume(inventory),
                prepared.preparation_claim,
                prepared.runtime_volume_authority,
                docker_settings,
            )
            capture_receipt, result_payload = capture_run_action_result_file(
                prepared,
                retained_terminal,
                volume,
                settings=launch_settings,
            )
        final_terminal = inspect_run_action_terminal(
            query=query,
            resource_manager=resource_manager,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
        )
        release_inspection.require_current()
        if (
            final_terminal != retained_terminal
            or resource_manager.observe(query.preparation_allocation) != inventory
            or read_run_action_host_boot_id(proc_root_descriptor)
            != adoption.workload_release_receipt.host_boot_id
        ):
            raise RunActionNaturalTerminalError(
                "natural terminal authority changed during resolution"
            )
        release_inspection.require_current()
        if type(result_payload) is bytes and result_payload:
            authority_query, authority_terminal = (
                RunActionCommittedContinuationCapability._take_result_capture_authority(
                    capability,
                    _authority=_RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
                )
            )
            if authority_query != query or authority_terminal != retained_terminal:
                raise RunActionNaturalTerminalError(
                    "natural result differs from its sealed terminal authority"
                )
            result = RunActionProviderResult(
                terminal_observation=retained_terminal,
                result_capture_receipt=capture_receipt,
                result_payload=result_payload,
            )
            RunActionCommittedContinuationCapability._complete_result_capture(
                capability,
                result,
                _authority=_RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
            )
            release_inspection.require_current()
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.RESULT_CAPTURED,
                result=result,
                provider_termination_receipt=None,
                timeout_directive_publication=None,
            )
        reason = _natural_termination_reason(
            retained_terminal.oom_killed,
            retained_terminal.exit_code,
            capture_receipt,
            result_payload,
        )
        authority_query, authority_terminal, loss_observation_id = (
            RunActionCommittedContinuationCapability._take_provider_termination_authority(
                capability,
                _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
            )
        )
        if (
            authority_query != query
            or authority_terminal != retained_terminal
            or loss_observation_id is not None
        ):
            raise RunActionNaturalTerminalError(
                "natural failure differs from its sealed terminal authority"
            )
        receipt = RunActionProviderTerminationReceipt.mint(
            disposition=RunActionProviderTerminationDisposition.FAILED,
            reason=reason,
            activation_event_id=query.activation_event.event_id,
            workload_release_adoption=adoption,
            terminal_observation=retained_terminal,
            timeout_directive_publication=None,
            empty_result_capture_receipt=(
                capture_receipt
                if reason is RunActionProviderTerminationReason.EMPTY_RESULT
                else None
            ),
            pre_release_main_loss_observation=None,
            credential_retirement_intent=None,
        )
        RunActionCommittedContinuationCapability._complete_provider_termination(
            capability,
            receipt,
            _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
        )
        release_inspection.require_current()
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PROVIDER_TERMINATED,
            result=None,
            provider_termination_receipt=receipt,
            timeout_directive_publication=None,
        )


def _natural_termination_reason(
    oom_killed: bool,
    exit_code: int,
    capture_receipt: RunActionResultCaptureReceipt | None,
    result_payload: bytes | None,
) -> RunActionProviderTerminationReason:
    if oom_killed:
        if capture_receipt is not None or result_payload is not None:
            raise RunActionNaturalTerminalError(
                "OOM termination unexpectedly consumed result authority"
            )
        return RunActionProviderTerminationReason.OOM
    if exit_code != 0:
        if capture_receipt is not None or result_payload is not None:
            raise RunActionNaturalTerminalError(
                "nonzero termination unexpectedly consumed result authority"
            )
        return RunActionProviderTerminationReason.NONZERO_EXIT
    if (
        type(capture_receipt) is not RunActionResultCaptureReceipt
        or result_payload != b""
    ):
        raise RunActionNaturalTerminalError(
            "zero exit lacks one exact nonempty or empty result"
        )
    return RunActionProviderTerminationReason.EMPTY_RESULT


__all__ = [
    "resolve_run_action_natural_terminal_once",
    "RunActionNaturalTerminalError",
]
