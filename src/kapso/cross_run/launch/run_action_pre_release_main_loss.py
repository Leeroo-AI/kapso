"""Retained positive proof of one main-container loss before workload release."""

from __future__ import annotations

import os
from contextlib import ExitStack
from threading import get_ident

from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionVolumeObservation,
    observe_running_keeper,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    docker_run_action_resource_inventory_digest,
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
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
    RunActionRuntimeVolumeEvidence,
    RunActionSupervisorHelperEvidence,
    RunActionVolumeKeeperEvidence,
    run_action_runtime_volume_occurrence_matches,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_host_boot_id,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_pre_release_main_loss_observation_token,
    RunActionPreReleaseMainLossObservation,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from kapso.cross_run.settings import DockerRuntimeSettings

_PRE_RELEASE_MAIN_LOSS_LEASE_AUTHORITY = object()


class RunActionPreReleaseMainLossError(RuntimeError):
    """A pre-release main loss lacks one retained positive proof."""


class _RunActionPreReleaseMainLossLease:
    """Thread-bound physical proof for one classification or publication phase."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        proc_root_descriptor: int,
        control_lease: RunActionControlDirectoryLease,
        query: RunActionCommittedSpawnQuery,
        observation: RunActionPreReleaseMainLossObservation,
        resource_manager: DockerRunActionResourceManager,
        helper_evidence: RunActionSupervisorHelperEvidence,
        init_source_evidence: RunActionDockerInitSourceEvidence,
        docker_settings: DockerRuntimeSettings,
        _authority: object,
    ) -> None:
        if (
            type(descriptors) is not ExitStack
            or type(proc_root_descriptor) is not int
            or proc_root_descriptor < 0
            or type(control_lease) is not RunActionControlDirectoryLease
            or type(query) is not RunActionCommittedSpawnQuery
            or type(observation) is not RunActionPreReleaseMainLossObservation
            or type(resource_manager) is not DockerRunActionResourceManager
            or type(helper_evidence) is not RunActionSupervisorHelperEvidence
            or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
            or type(docker_settings) is not DockerRuntimeSettings
            or resource_manager.runtime_settings != docker_settings
            or _authority is not _PRE_RELEASE_MAIN_LOSS_LEASE_AUTHORITY
        ):
            raise RunActionPreReleaseMainLossError(
                "pre-release main-loss lease lacks exact issued authority"
            )
        self._descriptors = descriptors
        self._proc_root_descriptor = proc_root_descriptor
        self._control_lease = control_lease
        self._query = query
        self._observation = observation
        self._resource_manager = resource_manager
        self._helper_evidence = helper_evidence
        self._init_source_evidence = init_source_evidence
        self._docker_settings = docker_settings
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._closed = False
        self.require_current()

    @property
    def observation(self) -> RunActionPreReleaseMainLossObservation:
        self.require_current()
        return self._observation

    def require_current(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionPreReleaseMainLossError(
                "pre-release main-loss lease is closed or foreign"
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
            raise RunActionPreReleaseMainLossError(
                "pre-release main-loss control occurrence changed"
            )
        inventory = self._resource_manager.observe(query.preparation_allocation)
        _require_pre_release_main_loss_inventory(query, inventory)
        if (
            docker_run_action_resource_inventory_digest(inventory)
            != self._observation.first_complete_inventory_digest
        ):
            raise RunActionPreReleaseMainLossError(
                "pre-release main-loss inventory changed"
            )
        volume, keeper, volume_evidence = _reobserve_surviving_resources(
            query=query,
            inventory=inventory,
            resource_manager=self._resource_manager,
            control_lease=self._control_lease,
            helper_evidence=self._helper_evidence,
            init_source_evidence=self._init_source_evidence,
            docker_settings=self._docker_settings,
        )
        if (
            keeper != activation.reobserved_keeper_evidence
            or not run_action_runtime_volume_occurrence_matches(
                volume_evidence,
                activation.reobserved_volume_evidence,
            )
            or volume.volume_name != prepared.runtime_volume_authority.volume_name
        ):
            raise RunActionPreReleaseMainLossError(
                "pre-release main-loss surviving resources changed"
            )
        self._control_lease.require_current()
        if (
            self._resource_manager.observe(query.preparation_allocation) != inventory
            or read_run_action_host_boot_id(self._proc_root_descriptor)
            != self._observation.host_boot_id
        ):
            raise RunActionPreReleaseMainLossError(
                "pre-release main-loss authority changed during reinspection"
            )

    def __enter__(self) -> "_RunActionPreReleaseMainLossLease":
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
            raise RunActionPreReleaseMainLossError(
                "pre-release main-loss lease is already closed or foreign"
            )
        self._closed = True
        self._descriptors.close()


def inspect_run_action_pre_release_main_loss(
    *,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
) -> RunActionPreReleaseMainLossObservation:
    """Classify stable main loss without retaining caller-owned descriptors."""

    lease = _open_run_action_pre_release_main_loss(
        query=query,
        resource_manager=resource_manager,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
    )
    with ExitStack() as cleanup:
        cleanup.callback(lease.close)
        return lease.observation


def _open_run_action_pre_release_main_loss(
    *,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
) -> _RunActionPreReleaseMainLossLease:
    """Retain stable proof that main alone vanished before event-5 release."""

    if (
        type(query) is not RunActionCommittedSpawnQuery
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or resource_manager.runtime_settings != docker_settings
        or query.workload_release_adoption is not None
        or query.timeout_directive_publication is not None
        or query.control_directory_topology
        is not RunActionControlDirectoryTopology.EMPTY
    ):
        raise RunActionPreReleaseMainLossError(
            "pre-release main-loss inspection lacks exact configured authority"
        )
    prepared = query.prepared_execution
    activation = query.activation_revalidation_receipt
    projection = prepared.volume_keeper_evidence.issued_create_projection
    if (
        helper_evidence != projection.helper_evidence
        or init_source_evidence != projection.docker_init_source_evidence
    ):
        raise RunActionPreReleaseMainLossError(
            "pre-release main-loss inputs differ from durable keeper authority"
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
            raise RunActionPreReleaseMainLossError(
                "pre-release main loss requires the empty control topology"
            )
        clock = _SystemRunActionClock()
        observed_before = clock.boottime_nanoseconds()
        host_boot_id = read_run_action_host_boot_id(proc_root_descriptor)
        first_inventory = resource_manager.observe(query.preparation_allocation)
        _require_pre_release_main_loss_inventory(query, first_inventory)
        _volume, keeper, volume_evidence = _reobserve_surviving_resources(
            query=query,
            inventory=first_inventory,
            resource_manager=resource_manager,
            control_lease=control_lease,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
        )
        second_inventory = resource_manager.observe(query.preparation_allocation)
        observed_after = clock.boottime_nanoseconds()
        control_lease.require_current()
        if (
            second_inventory != first_inventory
            or read_run_action_host_boot_id(proc_root_descriptor) != host_boot_id
            or keeper != activation.reobserved_keeper_evidence
            or not run_action_runtime_volume_occurrence_matches(
                volume_evidence,
                activation.reobserved_volume_evidence,
            )
        ):
            raise RunActionPreReleaseMainLossError(
                "pre-release main loss changed during retained observation"
            )
        inventory_digest = docker_run_action_resource_inventory_digest(first_inventory)
        control = prepared.control_directory
        observation = RunActionPreReleaseMainLossObservation.mint(
            activation_event_id=query.activation_event.event_id,
            preparation_allocation=query.preparation_allocation,
            activation_revalidation_receipt=activation,
            host_boot_id=host_boot_id,
            observed_before_boottime_nanoseconds=observed_before,
            first_complete_inventory_digest=inventory_digest,
            reobserved_volume_evidence=volume_evidence,
            reobserved_keeper_evidence=keeper,
            second_complete_inventory_digest=inventory_digest,
            observed_after_boottime_nanoseconds=observed_after,
            observed_runtime_volume_names=(
                prepared.runtime_volume_authority.volume_name,
            ),
            observed_keeper_container_ids=(
                prepared.volume_keeper_evidence.container_id,
            ),
            observed_main_container_ids=(),
            missing_provider_execution_id=query.spawn_commit.provider_execution_id,
            control_mount_id=control.mount_id,
            control_device=control.device,
            control_inode=control.inode,
            control_entry_count=0,
            control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
        )
        lease = _RunActionPreReleaseMainLossLease(
            descriptors=descriptors,
            proc_root_descriptor=proc_root_descriptor,
            control_lease=control_lease,
            query=query,
            observation=observation,
            resource_manager=resource_manager,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
            _authority=_PRE_RELEASE_MAIN_LOSS_LEASE_AUTHORITY,
        )
        lease._descriptors = descriptors.pop_all()
    return lease


def capture_run_action_pre_release_main_loss_termination(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
) -> RunActionContinuationOutcome:
    """Reprove sealed loss and transfer its fence to terminal publication."""

    if (
        type(capability) is not RunActionCommittedContinuationCapability
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or resource_manager.runtime_settings != docker_settings
    ):
        raise RunActionPreReleaseMainLossError(
            "pre-release main-loss capture lacks exact live authority"
        )
    query = capability.query
    loss_lease = _open_run_action_pre_release_main_loss(
        query=query,
        resource_manager=resource_manager,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
    )
    with ExitStack() as cleanup:
        cleanup.callback(loss_lease.close)
        loss_lease.require_current()
        observation = loss_lease.observation
        observation_token = run_action_pre_release_main_loss_observation_token(
            observation
        )
        if observation_token != capability.observation.observation_token:
            raise RunActionPreReleaseMainLossError(
                "pre-release main loss differs from sealed classification"
            )
        authority_query, terminal, loss_observation_token = (
            capability._take_provider_termination_authority(
                _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
            )
        )
        if (
            authority_query is not query
            or terminal is not None
            or loss_observation_token != observation_token
        ):
            raise RunActionPreReleaseMainLossError(
                "pre-release main-loss capability differs from its retained lease"
            )
        receipt = RunActionProviderTerminationReceipt.mint(
            disposition=RunActionProviderTerminationDisposition.FAILED,
            reason=RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
            activation_event_id=query.activation_event.event_id,
            workload_release_adoption=None,
            terminal_observation=None,
            timeout_directive_publication=None,
            empty_result_capture_receipt=None,
            pre_release_main_loss_observation=observation,
        )
        loss_lease.require_current()
        publication_fence = RunActionProviderTerminationPublicationFence(
            source=loss_lease,
            _authority=(_RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY),
        )
        capability._complete_provider_termination(
            receipt,
            publication_fence,
            _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
        )
        cleanup.pop_all()
        outcome = RunActionContinuationOutcome(
            state=RunActionContinuationState.PROVIDER_TERMINATED,
            result=None,
            provider_termination_receipt=receipt,
            timeout_directive_publication=None,
            provider_termination_publication_fence=publication_fence,
        )
        return outcome


def _require_pre_release_main_loss_inventory(
    query: RunActionCommittedSpawnQuery,
    inventory: DockerRunActionResourceInventory,
) -> None:
    prepared = query.prepared_execution
    if (
        type(inventory) is not DockerRunActionResourceInventory
        or inventory.preparation_allocation != query.preparation_allocation
        or inventory.volume_inspection_digest is None
        or inventory.keeper_container_id != prepared.volume_keeper_evidence.container_id
        or inventory.main_container_id is not None
    ):
        raise RunActionPreReleaseMainLossError(
            "pre-release main loss lacks exactly volume and keeper"
        )


def _reobserve_surviving_resources(
    *,
    query: RunActionCommittedSpawnQuery,
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    control_lease: RunActionControlDirectoryLease,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
) -> tuple[
    DockerRunActionVolumeObservation,
    RunActionVolumeKeeperEvidence,
    RunActionRuntimeVolumeEvidence,
]:
    prepared = query.prepared_execution
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        docker_settings,
    )
    keeper = observe_running_keeper(
        resource_manager.inspect_keeper(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        helper_evidence,
        init_source_evidence,
        docker_settings,
    )
    volume_evidence = control_lease.reobserve_runtime_volume_evidence(
        volume,
        keeper,
    )
    return volume, keeper, volume_evidence


__all__ = [
    "capture_run_action_pre_release_main_loss_termination",
    "inspect_run_action_pre_release_main_loss",
    "RunActionPreReleaseMainLossError",
]
