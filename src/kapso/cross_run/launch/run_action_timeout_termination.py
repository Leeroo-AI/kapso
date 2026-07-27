"""Trusted terminal receipt for one exact durably timed-out occurrence."""

from __future__ import annotations

from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
    RunActionCommittedContinuationCapability,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionSupervisorHelperEvidence,
)
from kapso.cross_run.launch.run_action_terminal_inspection import (
    reinspect_run_action_terminal,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings


class RunActionTimeoutTerminationError(RuntimeError):
    """A terminal timeout lacks its retained durable and physical authority."""


def capture_run_action_timeout_termination(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionProviderTerminationReceipt:
    """Reinspect and register timeout precedence for one exact terminal."""

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
        raise RunActionTimeoutTerminationError(
            "timeout termination inputs lack exact configured authority"
        )
    terminal = reinspect_run_action_terminal(
        capability=capability,
        resource_manager=resource_manager,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )
    query, retained_terminal, loss_observation_id = (
        RunActionCommittedContinuationCapability._take_provider_termination_authority(
            capability,
            _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
        )
    )
    adoption = query.workload_release_adoption
    publication = query.timeout_directive_publication
    if (
        retained_terminal != terminal
        or loss_observation_id is not None
        or adoption is None
        or publication is None
    ):
        raise RunActionTimeoutTerminationError(
            "timeout termination lacks its exact retained evidence graph"
        )
    receipt = RunActionProviderTerminationReceipt.mint(
        disposition=RunActionProviderTerminationDisposition.INTERRUPTED,
        reason=RunActionProviderTerminationReason.TIMEOUT,
        activation_event_id=query.activation_event.event_id,
        workload_release_adoption=adoption,
        terminal_observation=terminal,
        timeout_directive_publication=publication,
        pre_release_main_loss_observation=None,
        credential_retirement_intent=None,
    )
    RunActionCommittedContinuationCapability._complete_provider_termination(
        capability,
        receipt,
        _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
    )
    return receipt


__all__ = [
    "capture_run_action_timeout_termination",
    "RunActionTimeoutTerminationError",
]
