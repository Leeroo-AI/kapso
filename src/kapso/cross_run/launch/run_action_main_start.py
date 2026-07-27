"""Token-sealed start of one durably activated run-action barrier."""

from __future__ import annotations

from threading import Lock
from weakref import WeakKeyDictionary

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    _DOCKER_START_CONTAINER_AUTHORITY,
    _DOCKER_START_EXCLUSION_ISSUANCE,
    _docker_observation_and_start_authorities_share_runtime,
    PinnedDockerRuntime,
    PinnedDockerStartAuthority,
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
)
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    _run_action_observation_authority,
    docker_run_action_resource_inventory_digest,
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_START_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
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
from kapso.cross_run.process import BoundedProcessOutcome, BoundedProcessResult
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings

_START_MANAGER_LOCK = Lock()
_START_MANAGER_AUTHORITIES: WeakKeyDictionary[
    DockerRunActionStartManager, PinnedDockerStartAuthority
] = WeakKeyDictionary()


class RunActionMainStartError(RuntimeError):
    """A main-container start lacks exact event-5 or Docker authority."""


class DockerRunActionStartManager:
    """Start-only Docker authority with no generic mutation surface."""

    def __init__(self, runtime: PinnedDockerRuntime) -> None:
        if type(runtime) is not PinnedDockerRuntime:
            raise RunActionMainStartError(
                "run-action start requires one pinned Docker runtime"
            )
        authority = runtime.issue_start_authority()
        with _START_MANAGER_LOCK:
            if _START_MANAGER_AUTHORITIES.get(self) is not None:
                raise RunActionMainStartError(
                    "run-action start manager is already issued"
                )
            _START_MANAGER_AUTHORITIES[self] = authority

    @property
    def runtime_settings(self) -> DockerRuntimeSettings:
        """Return settings from the exact issued start authority."""

        return _start_authority(self).settings


def inspect_run_action_inert_activation(
    *,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    launch_settings: LaunchSettings,
) -> RunActionCommittedSpawnObservation:
    """Classify one exact event-5 activation whose main remains never-started."""

    _require_inert_query(query, resource_manager, launch_settings)
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
        activation_lease.require_current()
    return RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.INERT_CONTINUABLE,
        observation_token=observation_token,
    )


def start_run_action_barrier_once(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    start_manager: DockerRunActionStartManager,
    command: DockerRunActionCommand,
    volume_observation: DockerRunActionVolumeObservation,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionBarrierRunningContainerObservation:
    """Consume one inert continuation and prove the same barrier is running."""

    if (
        type(capability) is not RunActionCommittedContinuationCapability
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(start_manager) is not DockerRunActionStartManager
        or type(command) is not DockerRunActionCommand
        or type(volume_observation) is not DockerRunActionVolumeObservation
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
        or resource_manager.runtime_settings != docker_settings
        or start_manager.runtime_settings != docker_settings
        or not _docker_observation_and_start_authorities_share_runtime(
            _run_action_observation_authority(resource_manager),
            _start_authority(start_manager),
        )
    ):
        raise RunActionMainStartError(
            "run-action start inputs lack one exact configured runtime"
        )
    query = capability.query
    _require_inert_query(query, resource_manager, launch_settings)
    prepared = query.prepared_execution
    projection = prepared.inert_container_evidence.issued_create_projection
    if (
        command.command_template_id
        != prepared.preparation_claim.execution_policy.command_template_id
        or volume_observation.volume_authority_id
        != prepared.runtime_volume_authority.runtime_volume_authority_id
        or helper_evidence != projection.supervisor_helper_evidence
        or init_source_evidence != projection.docker_init_source_evidence
    ):
        raise RunActionMainStartError(
            "run-action start inputs differ from durable event 5"
        )
    start_authority = _start_authority(start_manager)
    with start_authority._issue_exclusion_lease(
        _authority=_DOCKER_START_EXCLUSION_ISSUANCE,
    ) as exclusion:
        with open_selected_run_action_activation(
            query.preparation_allocation,
            query.activation_revalidation_receipt,
            resource_manager,
            settings=launch_settings,
        ) as activation_lease:
            inventory = activation_lease.inventory
            observation_token = _inert_observation_token(query, inventory)
            sealed_query, sealed_token = capability._take_start_authority(
                observation_token,
                _authority=_RUN_ACTION_START_AUTHORITY,
            )
            if sealed_query != query or sealed_token != observation_token:
                raise RunActionMainStartError(
                    "run-action start continuation changed its exact inert seal"
                )
            activation_lease.require_current()
            exclusion.require_current()
            container_id = query.spawn_commit.provider_execution_id
            result = start_authority._start_created_container_once(
                container_id=container_id,
                exclusion_lease=exclusion,
                _authority=_DOCKER_START_CONTAINER_AUTHORITY,
            )
            if (
                type(result) is not BoundedProcessResult
                or type(result.outcome) is not BoundedProcessOutcome
                or result.outcome is not BoundedProcessOutcome.COMPLETED
                or result.returncode != 0
                or result.request.argv[-3:] != ("container", "start", container_id)
                or result.stdout != f"{container_id}\n".encode("ascii")
                or result.stderr
                or result.stdout_bytes_observed != len(result.stdout)
                or result.stderr_bytes_observed != 0
            ):
                raise RunActionMainStartError(
                    "Docker start result is failed or ambiguous; fresh recovery is required"
                )
            activation_lease.require_volume_current()
            running = _observe_exact_running_barrier(
                query=query,
                expected_inventory=inventory,
                resource_manager=resource_manager,
                command=command,
                volume_observation=volume_observation,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=docker_settings,
            )
            activation_lease.require_volume_current()
            repeated = _observe_exact_running_barrier(
                query=query,
                expected_inventory=inventory,
                resource_manager=resource_manager,
                command=command,
                volume_observation=volume_observation,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=docker_settings,
            )
            if repeated != running:
                raise RunActionMainStartError(
                    "started run-action barrier changed during stable reinspection"
                )
            activation_lease.require_volume_current()
            exclusion.require_current()
            capability._complete_start(
                running,
                observation_token,
                _authority=_RUN_ACTION_START_AUTHORITY,
            )
            return running


def _require_inert_query(
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    launch_settings: LaunchSettings,
) -> None:
    if (
        type(query) is not RunActionCommittedSpawnQuery
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(launch_settings) is not LaunchSettings
        or query.control_directory_topology
        is not RunActionControlDirectoryTopology.EMPTY
        or query.workload_release_adoption is not None
        or query.timeout_directive_publication is not None
    ):
        raise RunActionMainStartError(
            "run-action inert continuation requires exact empty event 5"
        )


def _inert_observation_token(
    query: RunActionCommittedSpawnQuery,
    inventory: DockerRunActionResourceInventory,
) -> str:
    if (
        type(query) is not RunActionCommittedSpawnQuery
        or type(inventory) is not DockerRunActionResourceInventory
        or inventory.preparation_allocation != query.preparation_allocation
        or inventory.main_container_id != query.spawn_commit.provider_execution_id
        or inventory.keeper_container_id
        != query.prepared_execution.volume_keeper_evidence.container_id
        or not inventory.volume_present
    ):
        raise RunActionMainStartError(
            "run-action inert token lacks the exact event-5 inventory"
        )
    return tree_or_blob_digest(
        canonical_json_bytes(
            {
                "activation_event_id": query.activation_event.event_id,
                "activation_revalidation_receipt_id": (
                    query.activation_revalidation_receipt.activation_revalidation_receipt_id
                ),
                "control_directory_topology": (
                    RunActionControlDirectoryTopology.EMPTY.value
                ),
                "docker_resource_inventory_digest": (
                    docker_run_action_resource_inventory_digest(inventory)
                ),
                "inert_container_evidence_id": (
                    query.prepared_execution.inert_container_evidence.inert_container_evidence_id
                ),
                "preparation_allocation_id": (
                    query.preparation_allocation.preparation_allocation_id
                ),
                "provider_execution_id": query.spawn_commit.provider_execution_id,
                "spawn_commit_id": query.spawn_commit.spawn_commit_id,
            }
        )
    )


def _observe_exact_running_barrier(
    *,
    query: RunActionCommittedSpawnQuery,
    expected_inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    volume_observation: DockerRunActionVolumeObservation,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
) -> RunActionBarrierRunningContainerObservation:
    inventory = resource_manager.observe(query.preparation_allocation)
    if inventory != expected_inventory:
        raise RunActionMainStartError(
            "run-action Docker inventory changed across barrier start"
        )
    prepared = query.prepared_execution
    return observe_running_barrier_main_container(
        resource_manager.inspect_main(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume_observation,
        command,
        helper_evidence,
        init_source_evidence,
        docker_settings,
    )


def _start_authority(
    manager: DockerRunActionStartManager,
) -> PinnedDockerStartAuthority:
    with _START_MANAGER_LOCK:
        authority = _START_MANAGER_AUTHORITIES.get(manager)
    if (
        type(manager) is not DockerRunActionStartManager
        or type(authority) is not PinnedDockerStartAuthority
    ):
        raise RunActionMainStartError("run-action start manager is unissued or foreign")
    return authority


__all__ = [
    "DockerRunActionStartManager",
    "inspect_run_action_inert_activation",
    "RunActionMainStartError",
    "start_run_action_barrier_once",
]
