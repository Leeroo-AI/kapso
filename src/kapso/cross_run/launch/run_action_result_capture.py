"""Trusted descriptor capture for one exact terminal run-action result."""

from __future__ import annotations

import os
from contextlib import ExitStack

from kapso.cross_run.launch.run_action_docker_inspect import observe_runtime_volume
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
    RunActionCommittedContinuationCapability,
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
    RunActionSupervisorHelperEvidence,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_host_boot_id,
)
from kapso.cross_run.launch.run_action_terminal_inspection import (
    inspect_run_action_terminal,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings


class RunActionResultCaptureError(RuntimeError):
    """Terminal result bytes lack their exact retained physical authority."""


def capture_run_action_terminal_result(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionProviderResult:
    """Capture and register one exact result under the terminal capability."""

    if (
        type(capability) is not RunActionCommittedContinuationCapability
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(command) is not DockerRunActionCommand
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
    ):
        raise RunActionResultCaptureError(
            "result capture inputs lack exact configured authority"
        )
    query, retained_terminal = capability._take_result_capture_authority(
        _authority=_RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
    )
    adoption = query.workload_release_adoption
    if adoption is None:
        raise RunActionResultCaptureError(
            "result capture requires an adopted workload release"
        )
    prepared = query.prepared_execution
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
            raise RunActionResultCaptureError(
                "result capture differs from its retained release occurrence"
            )
        inventory = resource_manager.observe(query.preparation_allocation)
        if (
            inventory.volume_inspection_digest is None
            or inventory.keeper_container_id
            != prepared.volume_keeper_evidence.container_id
            or inventory.main_container_id != query.spawn_commit.provider_execution_id
        ):
            raise RunActionResultCaptureError(
                "result capture lacks its exact Docker resource graph"
            )
        first_terminal = inspect_run_action_terminal(
            query=query,
            resource_manager=resource_manager,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
        )
        if first_terminal != retained_terminal:
            raise RunActionResultCaptureError(
                "result capture terminal differs from its trusted reinspection"
            )
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
        second_terminal = inspect_run_action_terminal(
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
            second_terminal != retained_terminal
            or resource_manager.observe(query.preparation_allocation) != inventory
            or read_run_action_host_boot_id(proc_root_descriptor)
            != adoption.workload_release_receipt.host_boot_id
        ):
            raise RunActionResultCaptureError(
                "result capture authority changed during descriptor read"
            )
        release_inspection.require_current()
        result = RunActionProviderResult(
            terminal_observation=retained_terminal,
            result_capture_receipt=capture_receipt,
            result_payload=result_payload,
        )
        capability._complete_result_capture(
            result,
            _authority=_RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
        )
        return result


__all__ = [
    "RunActionResultCaptureError",
    "capture_run_action_terminal_result",
]
