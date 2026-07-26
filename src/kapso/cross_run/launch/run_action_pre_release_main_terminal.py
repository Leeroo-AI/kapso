"""Retained proof of one present, exited main before workload release."""

from __future__ import annotations

import os
from contextlib import ExitStack
from threading import get_ident

from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_pre_release_terminal_main_container,
)
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    docker_run_action_resource_inventory_digest,
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_pre_release_resources import (
    reobserve_pre_release_surviving_resources,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
    _RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnQuery,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionProviderTerminationPublicationFence,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    open_run_action_control_directory,
    RunActionControlDirectoryLease,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionSupervisorHelperEvidence,
    run_action_runtime_volume_occurrence_matches,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_host_boot_id,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_pre_release_main_terminal_observation_token,
    RunActionPreReleaseMainTerminalObservation,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings

_PRE_RELEASE_MAIN_TERMINAL_LEASE_AUTHORITY = object()


class RunActionPreReleaseMainTerminalError(RuntimeError):
    """A pre-release exited main lacks one retained positive proof."""


class _RunActionPreReleaseMainTerminalLease:
    """Thread-bound present-exited proof retained through event-6 publication."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        proc_root_descriptor: int,
        control_lease: RunActionControlDirectoryLease,
        query: RunActionCommittedSpawnQuery,
        observation: RunActionPreReleaseMainTerminalObservation,
        resource_manager: DockerRunActionResourceManager,
        command: DockerRunActionCommand,
        helper_evidence: RunActionSupervisorHelperEvidence,
        init_source_evidence: RunActionDockerInitSourceEvidence,
        docker_settings: DockerRuntimeSettings,
        launch_settings: LaunchSettings,
        _authority: object,
    ) -> None:
        if (
            type(descriptors) is not ExitStack
            or type(proc_root_descriptor) is not int
            or proc_root_descriptor < 0
            or type(control_lease) is not RunActionControlDirectoryLease
            or type(query) is not RunActionCommittedSpawnQuery
            or type(observation) is not RunActionPreReleaseMainTerminalObservation
            or type(resource_manager) is not DockerRunActionResourceManager
            or type(command) is not DockerRunActionCommand
            or type(helper_evidence) is not RunActionSupervisorHelperEvidence
            or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
            or type(docker_settings) is not DockerRuntimeSettings
            or type(launch_settings) is not LaunchSettings
            or resource_manager.runtime_settings != docker_settings
            or _authority is not _PRE_RELEASE_MAIN_TERMINAL_LEASE_AUTHORITY
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal lease lacks exact issued authority"
            )
        self._descriptors = descriptors
        self._proc_root_descriptor = proc_root_descriptor
        self._control_lease = control_lease
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

    @property
    def observation(self) -> RunActionPreReleaseMainTerminalObservation:
        self.require_current()
        return self._observation

    def require_current(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal lease is closed or foreign"
            )
        query = self._query
        prepared = query.prepared_execution
        activation = query.activation_revalidation_receipt
        self._control_lease.require_current()
        if (
            self._control_lease.topology is not RunActionControlDirectoryTopology.EMPTY
            or read_run_action_host_boot_id(self._proc_root_descriptor)
            != self._observation.host_boot_id
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal control occurrence changed"
            )
        inventory = self._resource_manager.observe(query.preparation_allocation)
        _require_pre_release_main_terminal_inventory(query, inventory)
        if (
            docker_run_action_resource_inventory_digest(inventory)
            != self._observation.first_complete_inventory_digest
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal inventory changed"
            )
        volume, keeper, volume_evidence = reobserve_pre_release_surviving_resources(
            query=query,
            inventory=inventory,
            resource_manager=self._resource_manager,
            control_lease=self._control_lease,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
        )
        terminal = observe_pre_release_terminal_main_container(
            self._resource_manager.inspect_main(inventory),
            activation,
            volume,
            self._command,
            self._helper_evidence,
            self._init_source_evidence,
            self._docker_settings,
            inspection_size_limit_bytes=(
                self._launch_settings.run_action_process_snapshot_size_bytes
            ),
        )
        if (
            keeper != activation.reobserved_keeper_evidence
            or not run_action_runtime_volume_occurrence_matches(
                volume_evidence,
                activation.reobserved_volume_evidence,
            )
            or terminal != self._observation.terminal_container_observation
            or volume.volume_name != prepared.runtime_volume_authority.volume_name
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal resources changed"
            )
        self._control_lease.require_current()
        if (
            self._resource_manager.observe(query.preparation_allocation) != inventory
            or read_run_action_host_boot_id(self._proc_root_descriptor)
            != self._observation.host_boot_id
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal authority changed during reinspection"
            )

    def __enter__(self) -> "_RunActionPreReleaseMainTerminalLease":
        self.require_current()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal lease is already closed or foreign"
            )
        self._closed = True
        self._descriptors.close()


def inspect_run_action_pre_release_main_terminal(
    *,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionPreReleaseMainTerminalObservation:
    """Classify stable present-exited main without retaining caller descriptors."""

    lease = _open_run_action_pre_release_main_terminal(
        query=query,
        resource_manager=resource_manager,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )
    with ExitStack() as cleanup:
        cleanup.callback(lease.close)
        return lease.observation


def _open_run_action_pre_release_main_terminal(
    *,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> _RunActionPreReleaseMainTerminalLease:
    """Retain stable proof that the event-5 main exited before release."""

    if (
        type(query) is not RunActionCommittedSpawnQuery
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(command) is not DockerRunActionCommand
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
        or resource_manager.runtime_settings != docker_settings
        or query.workload_release_adoption is not None
        or query.timeout_directive_publication is not None
        or query.control_directory_topology
        is not RunActionControlDirectoryTopology.EMPTY
    ):
        raise RunActionPreReleaseMainTerminalError(
            "pre-release terminal inspection lacks configured authority"
        )
    prepared = query.prepared_execution
    activation = query.activation_revalidation_receipt
    main_projection = prepared.inert_container_evidence.issued_create_projection
    keeper_projection = prepared.volume_keeper_evidence.issued_create_projection
    if (
        command.command_template_id
        != prepared.preparation_claim.execution_policy.command_template_id
        or helper_evidence != main_projection.supervisor_helper_evidence
        or init_source_evidence != main_projection.docker_init_source_evidence
        or helper_evidence != keeper_projection.helper_evidence
        or init_source_evidence != keeper_projection.docker_init_source_evidence
    ):
        raise RunActionPreReleaseMainTerminalError(
            "pre-release terminal inputs differ from durable prepared authority"
        )
    with ExitStack() as descriptors:
        proc_root_descriptor = os.open(
            "/proc",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, proc_root_descriptor)
        control_lease = open_run_action_control_directory(prepared)
        descriptors.callback(control_lease.close)
        if control_lease.topology is not RunActionControlDirectoryTopology.EMPTY:
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal requires the empty control topology"
            )
        clock = _SystemRunActionClock()
        observed_before = clock.boottime_nanoseconds()
        host_boot_id = read_run_action_host_boot_id(proc_root_descriptor)
        first_inventory = resource_manager.observe(query.preparation_allocation)
        _require_pre_release_main_terminal_inventory(query, first_inventory)
        volume, keeper, volume_evidence = reobserve_pre_release_surviving_resources(
            query=query,
            inventory=first_inventory,
            resource_manager=resource_manager,
            control_lease=control_lease,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
        )
        first_terminal = observe_pre_release_terminal_main_container(
            resource_manager.inspect_main(first_inventory),
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
        control_lease.require_current()
        if resource_manager.observe(query.preparation_allocation) != first_inventory:
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal inventory changed before second inspection"
            )
        second_terminal = observe_pre_release_terminal_main_container(
            resource_manager.inspect_main(first_inventory),
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
        second_inventory = resource_manager.observe(query.preparation_allocation)
        observed_after = clock.boottime_nanoseconds()
        control_lease.require_current()
        if (
            second_inventory != first_inventory
            or first_terminal != second_terminal
            or read_run_action_host_boot_id(proc_root_descriptor) != host_boot_id
            or keeper != activation.reobserved_keeper_evidence
            or not run_action_runtime_volume_occurrence_matches(
                volume_evidence,
                activation.reobserved_volume_evidence,
            )
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal changed during retained observation"
            )
        inventory_digest = docker_run_action_resource_inventory_digest(first_inventory)
        control = prepared.control_directory
        observation = RunActionPreReleaseMainTerminalObservation.mint(
            activation_event_id=query.activation_event.event_id,
            preparation_allocation=query.preparation_allocation,
            activation_revalidation_receipt=activation,
            host_boot_id=host_boot_id,
            observed_before_boottime_nanoseconds=observed_before,
            first_complete_inventory_digest=inventory_digest,
            reobserved_volume_evidence=volume_evidence,
            reobserved_keeper_evidence=keeper,
            terminal_container_observation=second_terminal,
            second_complete_inventory_digest=inventory_digest,
            observed_after_boottime_nanoseconds=observed_after,
            observed_runtime_volume_names=(
                prepared.runtime_volume_authority.volume_name,
            ),
            observed_keeper_container_ids=(
                prepared.volume_keeper_evidence.container_id,
            ),
            observed_main_container_ids=(query.spawn_commit.provider_execution_id,),
            control_mount_id=control.mount_id,
            control_device=control.device,
            control_inode=control.inode,
            control_entry_count=0,
            control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
        )
        lease = _RunActionPreReleaseMainTerminalLease(
            descriptors=descriptors,
            proc_root_descriptor=proc_root_descriptor,
            control_lease=control_lease,
            query=query,
            observation=observation,
            resource_manager=resource_manager,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
            _authority=_PRE_RELEASE_MAIN_TERMINAL_LEASE_AUTHORITY,
        )
        lease._descriptors = descriptors.pop_all()
    return lease


def capture_run_action_pre_release_main_terminal_termination(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionContinuationOutcome:
    """Reprove sealed present-exited state and transfer its event-6 fence."""

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
        raise RunActionPreReleaseMainTerminalError(
            "pre-release terminal capture lacks exact live authority"
        )
    query = capability.query
    terminal_lease = _open_run_action_pre_release_main_terminal(
        query=query,
        resource_manager=resource_manager,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )
    with ExitStack() as cleanup:
        cleanup.callback(terminal_lease.close)
        terminal_lease.require_current()
        observation = terminal_lease.observation
        observation_token = run_action_pre_release_main_terminal_observation_token(
            observation
        )
        if observation_token != capability.observation.observation_token:
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal differs from sealed classification"
            )
        authority_query, released_terminal, pre_release_observation_token = (
            capability._take_provider_termination_authority(
                _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
            )
        )
        if (
            authority_query is not query
            or released_terminal is not None
            or pre_release_observation_token != observation_token
        ):
            raise RunActionPreReleaseMainTerminalError(
                "pre-release terminal capability differs from retained lease"
            )
        receipt = RunActionProviderTerminationReceipt.mint(
            disposition=RunActionProviderTerminationDisposition.FAILED,
            reason=RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL,
            activation_event_id=query.activation_event.event_id,
            workload_release_adoption=None,
            terminal_observation=observation,
            timeout_directive_publication=None,
            empty_result_capture_receipt=None,
            pre_release_main_loss_observation=None,
        )
        terminal_lease.require_current()
        publication_fence = RunActionProviderTerminationPublicationFence(
            source=terminal_lease,
            _authority=(_RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY),
        )
        capability._complete_provider_termination(
            receipt,
            publication_fence,
            _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
        )
        cleanup.pop_all()
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PROVIDER_TERMINATED,
            result=None,
            provider_termination_receipt=receipt,
            timeout_directive_publication=None,
            provider_termination_publication_fence=publication_fence,
        )


def _require_pre_release_main_terminal_inventory(
    query: RunActionCommittedSpawnQuery,
    inventory: DockerRunActionResourceInventory,
) -> None:
    prepared = query.prepared_execution
    if (
        type(inventory) is not DockerRunActionResourceInventory
        or inventory.preparation_allocation != query.preparation_allocation
        or inventory.volume_inspection_digest is None
        or inventory.keeper_container_id != prepared.volume_keeper_evidence.container_id
        or inventory.main_container_id != query.spawn_commit.provider_execution_id
    ):
        raise RunActionPreReleaseMainTerminalError(
            "pre-release terminal lacks exact volume, keeper, and main"
        )


__all__ = [
    "capture_run_action_pre_release_main_terminal_termination",
    "inspect_run_action_pre_release_main_terminal",
    "RunActionPreReleaseMainTerminalError",
]
