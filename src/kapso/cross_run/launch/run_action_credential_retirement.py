"""Physical retirement of one pre-release main with an expired credential."""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from weakref import WeakKeyDictionary

from kapso.cross_run.docker.runtime import (
    _DOCKER_CLEANUP_EXCLUSION_ISSUANCE,
    _DOCKER_CLEANUP_REMOVE_AUTHORITY,
    _DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
    _docker_authorities_share_runtime,
    _docker_observation_and_cleanup_authorities_share_runtime,
    PinnedDockerCleanupAuthority,
    PinnedDockerContainmentAuthority,
    PinnedDockerRuntime,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierRunningContainerObservation,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionVolumeObservation,
    observe_running_barrier_main_container,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    _run_action_observation_authority,
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_main_start import _inert_observation_token
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_CREDENTIAL_RETIREMENT_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnQuery,
    RunActionCommittedSpawnState,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    open_selected_run_action_activation,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionSupervisorHelperEvidence,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_running_container_occurrence_matches,
)
from kapso.cross_run.launch.run_action_timeout_adoption import (
    open_run_action_timeout_inspection,
    RunActionTimeoutInspectionLease,
)
from kapso.cross_run.process import BoundedProcessResult
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings

_CREDENTIAL_RETIREMENT_MANAGER_LOCK = Lock()
_CREDENTIAL_RETIREMENT_MANAGER_AUTHORITIES: WeakKeyDictionary[
    DockerRunActionCredentialRetirementManager,
    _DockerRunActionCredentialRetirementAuthorities,
] = WeakKeyDictionary()


class RunActionCredentialRetirementError(RuntimeError):
    """Expired credential retirement lacks exact event-5 or Docker authority."""


@dataclass(frozen=True)
class _DockerRunActionCredentialRetirementAuthorities:
    containment: PinnedDockerContainmentAuthority
    cleanup: PinnedDockerCleanupAuthority

    def __post_init__(self) -> None:
        if (
            type(self.containment) is not PinnedDockerContainmentAuthority
            or type(self.cleanup) is not PinnedDockerCleanupAuthority
        ):
            raise RunActionCredentialRetirementError(
                "credential retirement manager authorities are malformed"
            )


class DockerRunActionCredentialRetirementManager:
    """Issue only the signal/remove projections needed to retire a main."""

    def __init__(self, runtime: PinnedDockerRuntime) -> None:
        if type(runtime) is not PinnedDockerRuntime:
            raise RunActionCredentialRetirementError(
                "credential retirement requires one pinned Docker runtime"
            )
        authorities = _DockerRunActionCredentialRetirementAuthorities(
            containment=runtime.issue_containment_authority(),
            cleanup=runtime.issue_cleanup_authority(),
        )
        with _CREDENTIAL_RETIREMENT_MANAGER_LOCK:
            if _CREDENTIAL_RETIREMENT_MANAGER_AUTHORITIES.get(self) is not None:
                raise RunActionCredentialRetirementError(
                    "credential retirement manager is already issued"
                )
            _CREDENTIAL_RETIREMENT_MANAGER_AUTHORITIES[self] = authorities
        self._owner_process_id = os.getpid()

    def _require_owner_process(self) -> None:
        if (
            not hasattr(self, "_owner_process_id")
            or type(self._owner_process_id) is not int
            or self._owner_process_id != os.getpid()
        ):
            raise RunActionCredentialRetirementError(
                "credential retirement manager is unissued, cloned, or foreign"
            )

    @property
    def runtime_settings(self) -> DockerRuntimeSettings:
        """Return settings from the exact issued Docker projections."""

        authorities = _credential_retirement_authorities(self)
        if authorities.containment.settings != authorities.cleanup.settings:
            raise RunActionCredentialRetirementError(
                "credential retirement manager authorities name different runtimes"
            )
        return authorities.containment.settings

    def __copy__(self):
        raise RunActionCredentialRetirementError(
            "credential retirement manager cannot be copied"
        )

    def __deepcopy__(self, memo):
        raise RunActionCredentialRetirementError(
            "credential retirement manager cannot be copied"
        )

    def __reduce__(self):
        raise RunActionCredentialRetirementError(
            "credential retirement manager cannot be serialized"
        )


def retire_run_action_expired_credential_once(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    retirement_manager: DockerRunActionCredentialRetirementManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> None:
    """Attempt one exact main retirement; fresh recovery proves the result."""

    if type(retirement_manager) is DockerRunActionCredentialRetirementManager:
        retirement_manager._require_owner_process()
    if (
        type(capability) is not RunActionCommittedContinuationCapability
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(retirement_manager) is not DockerRunActionCredentialRetirementManager
        or type(command) is not DockerRunActionCommand
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
        or resource_manager.runtime_settings != docker_settings
        or retirement_manager.runtime_settings != docker_settings
    ):
        raise RunActionCredentialRetirementError(
            "credential retirement inputs lack exact configured authority"
        )
    authorities = _credential_retirement_authorities(retirement_manager)
    observation_authority = _run_action_observation_authority(resource_manager)
    if not _docker_authorities_share_runtime(
        observation_authority,
        authorities.containment,
    ) or not _docker_observation_and_cleanup_authorities_share_runtime(
        observation_authority,
        authorities.cleanup,
    ):
        raise RunActionCredentialRetirementError(
            "credential retirement inputs name different Docker runtimes"
        )
    query = capability.query
    observation = capability.observation
    prepared = query.prepared_execution
    projection = prepared.inert_container_evidence.issued_create_projection
    if (
        query.credential_retirement_intent is None
        or query.control_directory_topology
        is not RunActionControlDirectoryTopology.EMPTY
        or query.workload_release_adoption is not None
        or query.timeout_directive_publication is not None
        or observation.state
        not in {
            RunActionCommittedSpawnState.INERT_CONTINUABLE,
            RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
        }
        or command.command_template_id
        != prepared.preparation_claim.execution_policy.command_template_id
        or helper_evidence != projection.supervisor_helper_evidence
        or init_source_evidence != projection.docker_init_source_evidence
    ):
        raise RunActionCredentialRetirementError(
            "credential retirement differs from expired event 5"
        )
    with open_run_action_timeout_inspection(
        activation_event=query.activation_event,
        launch_settings=launch_settings,
    ) as control_inspection:
        if (
            control_inspection.topology is not RunActionControlDirectoryTopology.EMPTY
            or control_inspection.workload_release_adoption is not None
            or control_inspection.timeout_directive_publication is not None
        ):
            raise RunActionCredentialRetirementError(
                "credential retirement lost the empty control topology"
            )
        if observation.state is RunActionCommittedSpawnState.INERT_CONTINUABLE:
            _retire_inert_main(
                capability=capability,
                query=query,
                resource_manager=resource_manager,
                cleanup_authority=authorities.cleanup,
                control_inspection=control_inspection,
                launch_settings=launch_settings,
            )
        else:
            _retire_running_main(
                capability=capability,
                query=query,
                resource_manager=resource_manager,
                containment_authority=authorities.containment,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=docker_settings,
                control_inspection=control_inspection,
            )


def _retire_inert_main(
    *,
    capability: RunActionCommittedContinuationCapability,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    cleanup_authority: PinnedDockerCleanupAuthority,
    control_inspection: RunActionTimeoutInspectionLease,
    launch_settings: LaunchSettings,
) -> None:
    with cleanup_authority._issue_exclusion_lease(
        _authority=_DOCKER_CLEANUP_EXCLUSION_ISSUANCE,
    ) as exclusion:
        with open_selected_run_action_activation(
            query.preparation_allocation,
            query.activation_revalidation_receipt,
            resource_manager,
            settings=launch_settings,
        ) as activation_lease:
            observation_token = _inert_observation_token(
                query,
                activation_lease.inventory,
            )
            control_inspection.require_current()
            activation_lease.require_current()
            exclusion.require_current()
            sealed_query, sealed_token, retirement_intent = (
                RunActionCommittedContinuationCapability._take_credential_retirement_authority(
                    capability,
                    observation_token,
                    control_inspection,
                    _authority=_RUN_ACTION_CREDENTIAL_RETIREMENT_AUTHORITY,
                )
            )
            if sealed_query != query or sealed_token != observation_token:
                raise RunActionCredentialRetirementError(
                    "credential retirement changed its inert event-5 seal"
                )
            control_inspection.require_current()
            activation_lease.require_current()
            exclusion.require_current()
            container_id = sealed_query.spawn_commit.provider_execution_id
            result = cleanup_authority._remove_stopped_container_once(
                container_id=container_id,
                exclusion_lease=exclusion,
                _authority=_DOCKER_CLEANUP_REMOVE_AUTHORITY,
            )
            _require_exact_retirement_command(
                result,
                ("container", "rm", container_id),
            )
            control_inspection.require_current()
            exclusion.require_current()
            RunActionCommittedContinuationCapability._complete_credential_retirement(
                capability,
                observation_token,
                retirement_intent,
                _authority=_RUN_ACTION_CREDENTIAL_RETIREMENT_AUTHORITY,
            )


def _retire_running_main(
    *,
    capability: RunActionCommittedContinuationCapability,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    containment_authority: PinnedDockerContainmentAuthority,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    control_inspection: RunActionTimeoutInspectionLease,
) -> None:
    inventory, _volume, running = _observe_exact_running_occurrence(
        query=query,
        expected_observation_token=capability.observation.observation_token,
        resource_manager=resource_manager,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
    )
    observation_token = running.complete_inspection_digest
    control_inspection.require_current()
    sealed_query, sealed_token, retirement_intent = (
        RunActionCommittedContinuationCapability._take_credential_retirement_authority(
            capability,
            observation_token,
            control_inspection,
            _authority=_RUN_ACTION_CREDENTIAL_RETIREMENT_AUTHORITY,
        )
    )
    if sealed_query != query or sealed_token != observation_token:
        raise RunActionCredentialRetirementError(
            "credential retirement changed its running event-5 seal"
        )
    current_inventory, _current_volume, current_running = (
        _observe_exact_running_occurrence(
            query=sealed_query,
            expected_observation_token=observation_token,
            resource_manager=resource_manager,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
        )
    )
    if (
        current_inventory != inventory
        or not run_action_running_container_occurrence_matches(
            current_running,
            running,
        )
        or current_running.complete_inspection_digest != observation_token
    ):
        raise RunActionCredentialRetirementError(
            "credential retirement running occurrence changed before signal"
        )
    control_inspection.require_current()
    containment_authority.require_live_authority()
    result = containment_authority._signal_container_once(
        container_id=running.container_id,
        signal_name="SIGKILL",
        _authority=_DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
    )
    _require_exact_retirement_command(
        result,
        (
            "container",
            "kill",
            "--signal",
            "SIGKILL",
            running.container_id,
        ),
    )
    control_inspection.require_current()
    RunActionCommittedContinuationCapability._complete_credential_retirement(
        capability,
        observation_token,
        retirement_intent,
        _authority=_RUN_ACTION_CREDENTIAL_RETIREMENT_AUTHORITY,
    )


def _observe_exact_running_occurrence(
    *,
    query: RunActionCommittedSpawnQuery,
    expected_observation_token: str,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
) -> tuple[
    DockerRunActionResourceInventory,
    DockerRunActionVolumeObservation,
    RunActionBarrierRunningContainerObservation,
]:
    prepared = query.prepared_execution
    if type(expected_observation_token) is not str or not expected_observation_token:
        raise RunActionCredentialRetirementError(
            "credential retirement running token is malformed"
        )
    inventory = resource_manager.observe(query.preparation_allocation)
    if (
        not inventory.volume_present
        or inventory.keeper_container_id != prepared.volume_keeper_evidence.container_id
        or inventory.main_container_id != query.spawn_commit.provider_execution_id
    ):
        raise RunActionCredentialRetirementError(
            "credential retirement lacks its exact Docker resource graph"
        )
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        docker_settings,
    )
    running = observe_running_barrier_main_container(
        resource_manager.inspect_main(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        command,
        helper_evidence,
        init_source_evidence,
        docker_settings,
    )
    if (
        running.container_id != query.spawn_commit.provider_execution_id
        or running.complete_inspection_digest != expected_observation_token
        or resource_manager.observe(query.preparation_allocation) != inventory
    ):
        raise RunActionCredentialRetirementError(
            "credential retirement running occurrence differs from event 5"
        )
    return inventory, volume, running


def _require_exact_retirement_command(
    result: BoundedProcessResult,
    expected_arguments: tuple[str, ...],
) -> None:
    if (
        type(result) is not BoundedProcessResult
        or type(expected_arguments) is not tuple
        or not expected_arguments
        or result.request.argv[-len(expected_arguments) :] != expected_arguments
    ):
        raise RunActionCredentialRetirementError(
            "credential retirement command result changed its exact request"
        )


def _credential_retirement_authorities(
    manager: DockerRunActionCredentialRetirementManager,
) -> _DockerRunActionCredentialRetirementAuthorities:
    if type(manager) is not DockerRunActionCredentialRetirementManager:
        raise RunActionCredentialRetirementError(
            "credential retirement manager is unissued or foreign"
        )
    manager._require_owner_process()
    with _CREDENTIAL_RETIREMENT_MANAGER_LOCK:
        authorities = _CREDENTIAL_RETIREMENT_MANAGER_AUTHORITIES.get(manager)
    if type(authorities) is not _DockerRunActionCredentialRetirementAuthorities:
        raise RunActionCredentialRetirementError(
            "credential retirement manager is unissued or foreign"
        )
    return authorities


__all__ = [
    "DockerRunActionCredentialRetirementManager",
    "retire_run_action_expired_credential_once",
    "RunActionCredentialRetirementError",
]
