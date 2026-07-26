"""Trusted at-least-once TERM/KILL containment for one durable timeout."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import ExitStack
from threading import Lock
from typing import Any
from weakref import WeakKeyDictionary

from kapso.cross_run.docker.runtime import (
    _DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
    _docker_authorities_share_runtime,
    PinnedDockerContainmentAuthority,
    PinnedDockerRuntime,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierRunningContainerObservation,
)
from kapso.cross_run.launch.run_action_containment_contracts import (
    RunActionTimeoutContainmentResult,
    RunActionTimeoutContainmentState,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_running_barrier_main_container,
    observe_runtime_volume,
    observe_terminal_main_container,
)
from kapso.cross_run.launch.run_action_docker_projection import DockerRunActionCommand
from kapso.cross_run.launch.run_action_docker_resources import (
    _run_action_observation_authority,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY,
    RunActionCommittedContinuationCapability,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionSupervisorHelperEvidence,
    RunActionTerminalObservation,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_host_boot_id,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_running_container_occurrence_matches,
    RunActionTimeoutDirectivePublicationReceipt,
)
from kapso.cross_run.launch.run_action_timeout_adoption import (
    open_run_action_timeout_inspection,
)
from kapso.cross_run.process import BoundedProcessOutcome, BoundedProcessResult
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings

_CONTAINMENT_MANAGER_LOCK = Lock()
_CONTAINMENT_MANAGER_AUTHORITIES: WeakKeyDictionary[
    DockerRunActionContainmentManager, PinnedDockerContainmentAuthority
] = WeakKeyDictionary()


class RunActionTimeoutContainmentError(RuntimeError):
    """The timed-out occurrence cannot be safely signaled or reobserved."""


class DockerRunActionContainmentManager:
    """Issued containment authority with no generic Docker mutation surface."""

    def __init__(self, runtime: PinnedDockerRuntime) -> None:
        if type(runtime) is not PinnedDockerRuntime:
            raise RunActionTimeoutContainmentError(
                "Docker run-action containment requires the pinned runtime"
            )
        authority = runtime.issue_containment_authority()
        with _CONTAINMENT_MANAGER_LOCK:
            if _CONTAINMENT_MANAGER_AUTHORITIES.get(self) is not None:
                raise RunActionTimeoutContainmentError(
                    "Docker run-action containment manager is already issued"
                )
            _CONTAINMENT_MANAGER_AUTHORITIES[self] = authority

    @property
    def runtime_settings(self) -> DockerRuntimeSettings:
        """Return settings from the exact issued containment authority."""

        return _containment_authority(self).settings


def contain_run_action_timeout_once(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    containment_manager: DockerRunActionContainmentManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionTimeoutContainmentResult:
    """Signal only the exact still-running occurrence selected by durable timeout."""

    if (
        type(capability) is not RunActionCommittedContinuationCapability
        or type(resource_manager) is not DockerRunActionResourceManager
        or type(containment_manager) is not DockerRunActionContainmentManager
        or type(command) is not DockerRunActionCommand
        or type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or type(docker_settings) is not DockerRuntimeSettings
        or type(launch_settings) is not LaunchSettings
        or resource_manager.runtime_settings != docker_settings
        or containment_manager.runtime_settings != docker_settings
        or not _docker_authorities_share_runtime(
            _run_action_observation_authority(resource_manager),
            _containment_authority(containment_manager),
        )
    ):
        raise RunActionTimeoutContainmentError(
            "timeout containment inputs lack one exact configured runtime"
        )
    query = capability.query
    observation_token = capability.observation.observation_token
    adoption = query.workload_release_adoption
    publication = query.timeout_directive_publication
    prepared = query.prepared_execution
    projection = prepared.inert_container_evidence.issued_create_projection
    if (
        query.control_directory_topology
        is not RunActionControlDirectoryTopology.TIMED_OUT
        or adoption is None
        or type(publication) is not RunActionTimeoutDirectivePublicationReceipt
        or command.command_template_id
        != prepared.preparation_claim.execution_policy.command_template_id
        or helper_evidence != projection.supervisor_helper_evidence
        or init_source_evidence != projection.docker_init_source_evidence
    ):
        raise RunActionTimeoutContainmentError(
            "timeout containment differs from durable prepared authority"
        )
    with ExitStack() as retained:
        proc_root_descriptor = os.open(
            "/proc",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        retained.callback(os.close, proc_root_descriptor)
        control_inspection = open_run_action_timeout_inspection(
            activation_event=query.activation_event,
            launch_settings=launch_settings,
        )
        retained.callback(control_inspection.close)
        if (
            control_inspection.topology
            is not RunActionControlDirectoryTopology.TIMED_OUT
            or control_inspection.workload_release_adoption != adoption
            or control_inspection.timeout_directive_publication != publication
            or read_run_action_host_boot_id(proc_root_descriptor)
            != publication.timeout_directive.host_boot_id
        ):
            raise RunActionTimeoutContainmentError(
                "timeout containment lost its retained timed-out occurrence"
            )
        authorization = capability._begin_timeout_containment(
            control_inspection,
            _authority=_RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY,
        )
        with authorization:
            containment_authority = _containment_authority(containment_manager)
            containment_authority.require_live_authority()
            inventory, volume, pre_signal = _observe_exact_pre_signal_occurrence(
                query=query,
                resource_manager=resource_manager,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=docker_settings,
                launch_settings=launch_settings,
            )
            pre_state, running, terminal = pre_signal
            if pre_state is RunActionTimeoutContainmentState.TERMINAL:
                return _complete_terminal_before_signal(
                    authorization=authorization,
                    terminal=terminal,
                    query=query,
                    resource_manager=resource_manager,
                    inventory=inventory,
                    control_inspection=control_inspection,
                    proc_root_descriptor=proc_root_descriptor,
                    host_boot_id=publication.timeout_directive.host_boot_id,
                )
            if (
                type(running) is not RunActionBarrierRunningContainerObservation
                or not run_action_running_container_occurrence_matches(
                    running,
                    publication.timeout_directive.running_container_observation,
                )
                or running.complete_inspection_digest != observation_token
            ):
                raise RunActionTimeoutContainmentError(
                    "timeout containment running occurrence differs from its seal"
                )
            control_inspection.require_current()
            final_state, final_running, final_terminal = _observe_post_signal_main(
                resource_manager.inspect_main(inventory),
                query=query,
                volume=volume,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=docker_settings,
                launch_settings=launch_settings,
            )
            if final_state is RunActionTimeoutContainmentState.TERMINAL:
                return _complete_terminal_before_signal(
                    authorization=authorization,
                    terminal=final_terminal,
                    query=query,
                    resource_manager=resource_manager,
                    inventory=inventory,
                    control_inspection=control_inspection,
                    proc_root_descriptor=proc_root_descriptor,
                    host_boot_id=publication.timeout_directive.host_boot_id,
                )
            if (
                type(final_running) is not RunActionBarrierRunningContainerObservation
                or not run_action_running_container_occurrence_matches(
                    final_running,
                    running,
                )
                or final_running.complete_inspection_digest != observation_token
                or resource_manager.observe(query.preparation_allocation) != inventory
                or read_run_action_host_boot_id(proc_root_descriptor)
                != publication.timeout_directive.host_boot_id
            ):
                raise RunActionTimeoutContainmentError(
                    "timeout containment lost its running occurrence before signal"
                )
            control_inspection.require_current()
            signal, selected_at = authorization._select_signal(
                final_running,
                publication.timeout_directive.host_boot_id,
                _authority=_RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY,
            )
            command_result = containment_authority._signal_container_once(
                container_id=final_running.container_id,
                signal_name=signal.value,
                _authority=_DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
            )
            post_state, post_running, post_terminal = _observe_stable_main_occurrence(
                query=query,
                resource_manager=resource_manager,
                inventory=inventory,
                volume=volume,
                command=command,
                helper_evidence=helper_evidence,
                init_source_evidence=init_source_evidence,
                docker_settings=docker_settings,
                launch_settings=launch_settings,
            )
            signal_dispatch_confirmed = _signal_dispatch_confirmed(
                command_result,
                final_running.container_id,
            )
            if (
                _signal_command_has_malformed_success(
                    command_result,
                    final_running.container_id,
                )
                or (
                    not signal_dispatch_confirmed
                    and post_state is RunActionTimeoutContainmentState.RUNNING
                )
                or resource_manager.observe(query.preparation_allocation) != inventory
                or read_run_action_host_boot_id(proc_root_descriptor)
                != publication.timeout_directive.host_boot_id
            ):
                raise RunActionTimeoutContainmentError(
                    "timeout containment signal was not safely resolved"
                )
            control_inspection.require_current()
            result = RunActionTimeoutContainmentResult(
                signal=signal,
                selected_at_boottime_nanoseconds=selected_at,
                signal_dispatch_confirmed=signal_dispatch_confirmed,
                state=post_state,
                running_observation=post_running,
                terminal_observation=post_terminal,
            )
            authorization._complete(
                result,
                _authority=_RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY,
            )
            control_inspection.require_current()
            return result


def _observe_exact_pre_signal_occurrence(
    *,
    query,
    resource_manager,
    command,
    helper_evidence,
    init_source_evidence,
    docker_settings,
    launch_settings,
):
    prepared = query.prepared_execution
    inventory = resource_manager.observe(query.preparation_allocation)
    if (
        inventory.volume_inspection_digest is None
        or inventory.keeper_container_id != prepared.volume_keeper_evidence.container_id
        or inventory.main_container_id != query.spawn_commit.provider_execution_id
    ):
        raise RunActionTimeoutContainmentError(
            "timeout containment lacks its exact Docker resource graph"
        )
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        docker_settings,
    )
    observation = _observe_stable_main_occurrence(
        query=query,
        resource_manager=resource_manager,
        inventory=inventory,
        volume=volume,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )
    if resource_manager.observe(query.preparation_allocation) != inventory:
        raise RunActionTimeoutContainmentError(
            "timeout containment resources changed during pre-signal inspection"
        )
    return inventory, volume, observation


def _observe_stable_main_occurrence(
    *,
    query,
    resource_manager,
    inventory,
    volume,
    command,
    helper_evidence,
    init_source_evidence,
    docker_settings,
    launch_settings,
):
    first = _observe_post_signal_main(
        resource_manager.inspect_main(inventory),
        query=query,
        volume=volume,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )
    second = _observe_post_signal_main(
        resource_manager.inspect_main(inventory),
        query=query,
        volume=volume,
        command=command,
        helper_evidence=helper_evidence,
        init_source_evidence=init_source_evidence,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )
    if first == second:
        return second
    if (
        first[0] is RunActionTimeoutContainmentState.RUNNING
        and second[0] is RunActionTimeoutContainmentState.TERMINAL
    ):
        third = _observe_post_signal_main(
            resource_manager.inspect_main(inventory),
            query=query,
            volume=volume,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
        )
        if second == third:
            return third
    raise RunActionTimeoutContainmentError(
        "timeout containment occurrence changed during stable inspection"
    )


def _complete_terminal_before_signal(
    *,
    authorization,
    terminal,
    query,
    resource_manager,
    inventory,
    control_inspection,
    proc_root_descriptor,
    host_boot_id,
):
    if (
        type(terminal) is not RunActionTerminalObservation
        or resource_manager.observe(query.preparation_allocation) != inventory
        or read_run_action_host_boot_id(proc_root_descriptor) != host_boot_id
    ):
        raise RunActionTimeoutContainmentError(
            "timeout containment terminal race lost its exact occurrence"
        )
    control_inspection.require_current()
    result = RunActionTimeoutContainmentResult(
        signal=None,
        selected_at_boottime_nanoseconds=None,
        signal_dispatch_confirmed=False,
        state=RunActionTimeoutContainmentState.TERMINAL,
        running_observation=None,
        terminal_observation=terminal,
    )
    authorization._complete(
        result,
        _authority=_RUN_ACTION_TIMEOUT_CONTAINMENT_AUTHORITY,
    )
    control_inspection.require_current()
    return result


def _observe_post_signal_main(
    raw_inspection: Mapping[str, Any],
    *,
    query,
    volume,
    command,
    helper_evidence,
    init_source_evidence,
    docker_settings,
    launch_settings,
):
    state = raw_inspection.get("State") if isinstance(raw_inspection, Mapping) else None
    if not isinstance(state, Mapping) or type(state.get("Running")) is not bool:
        raise RunActionTimeoutContainmentError(
            "timeout containment post-signal state is malformed"
        )
    if state["Running"]:
        running = observe_running_barrier_main_container(
            raw_inspection,
            query.prepared_execution.preparation_claim,
            query.prepared_execution.runtime_volume_authority,
            volume,
            command,
            helper_evidence,
            init_source_evidence,
            docker_settings,
        )
        return RunActionTimeoutContainmentState.RUNNING, running, None
    terminal = observe_terminal_main_container(
        raw_inspection,
        query.activation_revalidation_receipt,
        query.workload_release_adoption,
        volume,
        command,
        helper_evidence,
        init_source_evidence,
        docker_settings,
        inspection_size_limit_bytes=(
            launch_settings.run_action_process_snapshot_size_bytes
        ),
    )
    return RunActionTimeoutContainmentState.TERMINAL, None, terminal


def _signal_dispatch_confirmed(
    result: BoundedProcessResult,
    container_id: str,
) -> bool:
    return (
        type(result) is BoundedProcessResult
        and result.outcome is BoundedProcessOutcome.COMPLETED
        and result.returncode == 0
        and result.stdout == f"{container_id}\n".encode()
        and not result.stderr
    )


def _signal_command_has_malformed_success(
    result: BoundedProcessResult,
    container_id: str,
) -> bool:
    return type(result) is not BoundedProcessResult or (
        result.outcome is BoundedProcessOutcome.COMPLETED
        and result.returncode == 0
        and not _signal_dispatch_confirmed(result, container_id)
    )


def _containment_authority(
    manager: DockerRunActionContainmentManager,
) -> PinnedDockerContainmentAuthority:
    with _CONTAINMENT_MANAGER_LOCK:
        authority = _CONTAINMENT_MANAGER_AUTHORITIES.get(manager)
    if (
        type(manager) is not DockerRunActionContainmentManager
        or type(authority) is not PinnedDockerContainmentAuthority
    ):
        raise RunActionTimeoutContainmentError(
            "Docker run-action containment manager is unissued or foreign"
        )
    return authority


__all__ = [
    "contain_run_action_timeout_once",
    "DockerRunActionContainmentManager",
    "RunActionTimeoutContainmentError",
]
