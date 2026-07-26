"""Shared exact reinspection of volume and keeper before workload release."""

from __future__ import annotations

from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionVolumeObservation,
    observe_running_keeper,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import RunActionCommittedSpawnQuery
from kapso.cross_run.launch.run_action_runtime_volume import (
    RunActionControlDirectoryLease,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionRuntimeVolumeEvidence,
    RunActionSupervisorHelperEvidence,
    RunActionVolumeKeeperEvidence,
)
from kapso.cross_run.settings import DockerRuntimeSettings


def reobserve_pre_release_surviving_resources(
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
    """Reprove the exact volume and keeper retained by an empty control tree."""

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


__all__ = ["reobserve_pre_release_surviving_resources"]
