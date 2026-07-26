"""Read-only terminal authority for one adopted run-action release occurrence."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any

from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_runtime_volume,
    observe_terminal_main_container,
)
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnQuery,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    RunActionReleasePresence,
    open_run_action_release_inspection,
)
from kapso.cross_run.launch.run_action_release_authority import (
    require_run_action_workload_release_receipt_matches_event,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionSupervisorHelperEvidence,
    RunActionTerminalObservation,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_host_boot_id,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings


class RunActionTerminalInspectionError(RuntimeError):
    """Terminal Docker state lacks one retained event-5 release authority."""


def inspect_run_action_terminal(
    *,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionTerminalObservation:
    """Observe one stable exited occurrence without continuation authority."""

    if type(query) is not RunActionCommittedSpawnQuery:
        raise RunActionTerminalInspectionError(
            "terminal inspection requires one exact committed query"
        )
    return _inspect_exact_terminal(
        query=query,
        resource_manager=resource_manager,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )


def reinspect_run_action_terminal(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionTerminalObservation:
    """Consume one sealed terminal capability and reproduce its exact digest."""

    if type(capability) is not RunActionCommittedContinuationCapability:
        raise RunActionTerminalInspectionError(
            "terminal reinspection requires one exact continuation capability"
        )
    query, observation_token = capability._take_terminal_inspection_authority(
        _authority=_RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY,
    )
    terminal = _inspect_exact_terminal(
        query=query,
        resource_manager=resource_manager,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )
    if terminal.complete_inspection_digest != observation_token:
        raise RunActionTerminalInspectionError(
            "terminal reinspection differs from its sealed observation"
        )
    capability._complete_terminal_inspection(
        terminal,
        _authority=_RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY,
    )
    return terminal


def _inspect_exact_terminal(
    *,
    query: RunActionCommittedSpawnQuery,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionTerminalObservation:
    if (
        type(resource_manager) is not DockerRunActionResourceManager
        or type(command) is not DockerRunActionCommand
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
    ):
        raise RunActionTerminalInspectionError(
            "terminal inspection inputs lack exact configured authority"
        )
    adoption = query.workload_release_adoption
    if adoption is None:
        raise RunActionTerminalInspectionError(
            "terminal inspection requires an adopted event-5 release"
        )
    require_run_action_workload_release_receipt_matches_event(
        adoption.workload_release_receipt,
        query.activation_event,
    )
    prepared = query.prepared_execution
    projection = prepared.inert_container_evidence.issued_create_projection
    if (
        command.command_template_id
        != prepared.preparation_claim.execution_policy.command_template_id
        or helper_evidence != projection.supervisor_helper_evidence
        or init_source_evidence != projection.docker_init_source_evidence
    ):
        raise RunActionTerminalInspectionError(
            "terminal inspection inputs differ from durable prepared authority"
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
            release_inspection.presence is not RunActionReleasePresence.PRESENT
            or release_inspection.adoption != adoption
            or read_run_action_host_boot_id(proc_root_descriptor)
            != adoption.workload_release_receipt.host_boot_id
        ):
            raise RunActionTerminalInspectionError(
                "terminal inspection differs from its retained release occurrence"
            )
        inventory = resource_manager.observe(query.preparation_allocation)
        if (
            inventory.volume_inspection_digest is None
            or inventory.keeper_container_id
            != prepared.volume_keeper_evidence.container_id
            or inventory.main_container_id != query.spawn_commit.provider_execution_id
        ):
            raise RunActionTerminalInspectionError(
                "terminal inspection lacks its exact Docker resource graph"
            )
        volume = observe_runtime_volume(
            resource_manager.inspect_volume(inventory),
            prepared.preparation_claim,
            prepared.runtime_volume_authority,
            docker_settings,
        )
        first_raw = resource_manager.inspect_main(inventory)
        first = observe_terminal_main_container(
            first_raw,
            query.activation_revalidation_receipt,
            adoption,
            volume,
            command,
            helper_evidence,
            init_source_evidence,
            docker_settings,
            inspection_size_limit_bytes=(
                launch_settings.run_action_process_snapshot_size_bytes
            ),
        )
        second_raw = resource_manager.inspect_main(inventory)
        second = observe_terminal_main_container(
            second_raw,
            query.activation_revalidation_receipt,
            adoption,
            volume,
            command,
            helper_evidence,
            init_source_evidence,
            docker_settings,
            inspection_size_limit_bytes=(
                launch_settings.run_action_process_snapshot_size_bytes
            ),
        )
        release_inspection.require_current()
        if first != second:
            changed_paths = _changed_paths(first_raw, second_raw)
            raise RunActionTerminalInspectionError(
                "terminal Docker snapshots changed during retained inspection at "
                + ",".join(changed_paths)
            )
        if resource_manager.observe(query.preparation_allocation) != inventory:
            raise RunActionTerminalInspectionError(
                "terminal Docker inventory changed during retained inspection"
            )
        if (
            read_run_action_host_boot_id(proc_root_descriptor)
            != adoption.workload_release_receipt.host_boot_id
        ):
            raise RunActionTerminalInspectionError(
                "host boot identity changed during retained terminal inspection"
            )
        release_inspection.require_current()
        return second


def _changed_paths(first: Any, second: Any, prefix: str = "") -> tuple[str, ...]:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        keys = set(first) | set(second)
        changed: list[str] = []
        for key in sorted(keys):
            path = key if not prefix else f"{prefix}.{key}"
            if key not in first or key not in second:
                changed.append(path)
            else:
                changed.extend(_changed_paths(first[key], second[key], path))
        return tuple(changed)
    if type(first) is list and type(second) is list:
        changed = []
        for index in range(max(len(first), len(second))):
            path = f"{prefix}[{index}]"
            if index >= len(first) or index >= len(second):
                changed.append(path)
            else:
                changed.extend(_changed_paths(first[index], second[index], path))
        return tuple(changed)
    return () if type(first) is type(second) and first == second else (prefix,)


__all__ = [
    "RunActionTerminalInspectionError",
    "inspect_run_action_terminal",
    "reinspect_run_action_terminal",
]
