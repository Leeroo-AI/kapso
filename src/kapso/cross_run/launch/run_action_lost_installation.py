"""Fenced terminal transition for an exact non-runnable Docker installation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import ExitStack
from threading import get_ident

from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_lost_installation_keeper,
    observe_pre_release_terminal_main_container,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    docker_run_action_resource_inventory_digest,
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionLostInstallationQuery,
    RunActionProviderTerminationPublicationFence,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionSupervisorHelperEvidence,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_host_boot_id,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_lost_installation_observation_token,
    RunActionLostInstallationObservation,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings


class RunActionLostInstallationError(RuntimeError):
    """A lost runtime installation is ambiguous, runnable, or changed."""


class _RunActionLostInstallationLease:
    """Thread-bound positive proof retained through terminal publication."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        proc_root_descriptor: int,
        query: RunActionLostInstallationQuery,
        observation: RunActionLostInstallationObservation,
        resource_manager: DockerRunActionResourceManager,
        command: DockerRunActionCommand,
        helper_evidence: RunActionSupervisorHelperEvidence,
        init_source_evidence: RunActionDockerInitSourceEvidence,
        docker_settings: DockerRuntimeSettings,
        launch_settings: LaunchSettings,
    ) -> None:
        self._descriptors = descriptors
        self._proc_root_descriptor = proc_root_descriptor
        self._query = query
        self._observation = observation
        self._resource_manager = resource_manager
        self._command = command
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self._launch_settings = launch_settings
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._closed = False
        self.require_current()

    def require_current(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionLostInstallationError(
                "lost-installation lease is closed or foreign"
            )
        observed = _capture_lost_installation_observation(
            query=self._query,
            proc_root_descriptor=self._proc_root_descriptor,
            resource_manager=self._resource_manager,
            command=self._command,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
            launch_settings=self._launch_settings,
        )
        if run_action_lost_installation_observation_token(
            observed
        ) != run_action_lost_installation_observation_token(self._observation):
            raise RunActionLostInstallationError(
                "lost runtime installation changed during publication"
            )

    def close(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionLostInstallationError(
                "lost-installation lease is already closed or foreign"
            )
        self._closed = True
        self._descriptors.close()


def inspect_run_action_lost_installation(
    *,
    query: RunActionLostInstallationQuery,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionContinuationOutcome | None:
    """Return a fenced interruption only when both exact processes are exited."""

    if (
        type(query) is not RunActionLostInstallationQuery
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(command) is not DockerRunActionCommand
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
        or resource_manager.runtime_settings != docker_settings
    ):
        raise RunActionLostInstallationError(
            "lost-installation inspection lacks exact configured authority"
        )
    inventory = resource_manager.observe(query.preparation_allocation)
    if not _inventory_is_complete(query, inventory):
        return None
    keeper_state = resource_manager.inspect_keeper(inventory).get("State")
    main_state = resource_manager.inspect_main(inventory).get("State")
    if not isinstance(keeper_state, Mapping) or not isinstance(main_state, Mapping):
        raise RunActionLostInstallationError(
            "lost-installation container state is malformed"
        )
    keeper_status = keeper_state.get("Status")
    main_status = main_state.get("Status")
    if not isinstance(keeper_status, str) or not isinstance(main_status, str):
        raise RunActionLostInstallationError(
            "lost-installation container lifecycle is malformed"
        )
    if keeper_status != "exited" or main_status != "exited":
        return None
    with ExitStack() as descriptors:
        proc_root_descriptor = os.open(
            "/proc",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, proc_root_descriptor)
        observation = _capture_lost_installation_observation(
            query=query,
            proc_root_descriptor=proc_root_descriptor,
            resource_manager=resource_manager,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
        )
        lease = _RunActionLostInstallationLease(
            descriptors=descriptors,
            proc_root_descriptor=proc_root_descriptor,
            query=query,
            observation=observation,
            resource_manager=resource_manager,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
        )
        lease._descriptors = descriptors.pop_all()
    credential_retirement_intent = query.credential_retirement_intent
    receipt = RunActionProviderTerminationReceipt.mint(
        disposition=RunActionProviderTerminationDisposition.INTERRUPTED,
        reason=(
            RunActionProviderTerminationReason.CREDENTIAL_EXPIRED
            if credential_retirement_intent is not None
            else RunActionProviderTerminationReason.RUNTIME_INSTALLATION_LOST
        ),
        activation_event_id=query.activation_event.event_id,
        workload_release_adoption=None,
        terminal_observation=observation,
        timeout_directive_publication=None,
        pre_release_main_loss_observation=None,
        credential_retirement_intent=credential_retirement_intent,
    )
    publication_fence = RunActionProviderTerminationPublicationFence(
        source=lease,
        _authority=_RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY,
    )
    return RunActionContinuationOutcome(
        state=RunActionContinuationState.PROVIDER_TERMINATED,
        result=None,
        provider_termination_receipt=receipt,
        timeout_directive_publication=None,
        provider_termination_publication_fence=publication_fence,
    )


def _capture_lost_installation_observation(
    *,
    query: RunActionLostInstallationQuery,
    proc_root_descriptor: int,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionLostInstallationObservation:
    prepared = query.activation_event.activation_revalidation_receipt.prepared_execution
    activation = query.activation_event.activation_revalidation_receipt
    first_boot_id = read_run_action_host_boot_id(proc_root_descriptor)
    first = resource_manager.observe(query.preparation_allocation)
    if not _inventory_is_complete(query, first):
        raise RunActionLostInstallationError(
            "lost installation lacks its exact volume, keeper, and main"
        )
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(first),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        docker_settings,
    )
    keeper = observe_lost_installation_keeper(
        resource_manager.inspect_keeper(first),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        helper_evidence,
        init_source_evidence,
        docker_settings,
    )
    main = observe_pre_release_terminal_main_container(
        resource_manager.inspect_main(first),
        activation,
        volume,
        command,
        helper_evidence,
        init_source_evidence,
        docker_settings,
        inspection_size_limit_bytes=(
            launch_settings.run_action_process_snapshot_size_bytes
        ),
    )
    second = resource_manager.observe(query.preparation_allocation)
    if (
        second != first
        or read_run_action_host_boot_id(proc_root_descriptor) != first_boot_id
        or volume.volume_occurrence_digest
        != prepared.runtime_volume_evidence.docker_volume_occurrence_digest
    ):
        raise RunActionLostInstallationError(
            "lost runtime installation changed during inspection"
        )
    return RunActionLostInstallationObservation.mint(
        activation_event_id=query.activation_event.event_id,
        preparation_allocation=query.preparation_allocation,
        activation_revalidation_receipt=activation,
        host_boot_id=first_boot_id,
        complete_inventory_digest=docker_run_action_resource_inventory_digest(first),
        docker_volume_occurrence_digest=volume.volume_occurrence_digest,
        keeper_observation=keeper,
        main_observation=main,
        observed_runtime_volume_names=(prepared.runtime_volume_authority.volume_name,),
        observed_keeper_container_ids=(prepared.volume_keeper_evidence.container_id,),
        observed_main_container_ids=(activation.spawn_commit.provider_execution_id,),
    )


def _inventory_is_complete(
    query: RunActionLostInstallationQuery,
    inventory: DockerRunActionResourceInventory,
) -> bool:
    activation = query.activation_event.activation_revalidation_receipt
    prepared = activation.prepared_execution
    return (
        type(inventory) is DockerRunActionResourceInventory
        and inventory.preparation_allocation == query.preparation_allocation
        and inventory.volume_inspection_digest is not None
        and inventory.keeper_container_id
        == prepared.volume_keeper_evidence.container_id
        and inventory.main_container_id == activation.spawn_commit.provider_execution_id
    )


__all__ = [
    "inspect_run_action_lost_installation",
    "RunActionLostInstallationError",
]
