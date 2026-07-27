"""Trusted publication of one durable timeout directive without signaling."""

from __future__ import annotations

import os
from contextlib import ExitStack

from kapso.cross_run.launch.run_action_atomic_publication import (
    open_run_action_anonymous_file,
)
from kapso.cross_run.launch.run_action_control_candidate import (
    _CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY,
    _RunActionControlFileTransition,
    _RunActionFrozenControlFileCandidate,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_running_barrier_main_container,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY,
    RunActionCommittedContinuationCapability,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionSupervisorHelperEvidence,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_host_boot_id,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_running_container_occurrence_matches,
    RunActionTimeoutDirectivePublicationReceipt,
)
from kapso.cross_run.launch.run_action_timeout_adoption import (
    _RUN_ACTION_TIMEOUT_PUBLICATION_AUTHORITY,
    open_run_action_timeout_inspection,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings

_ANONYMOUS_FILE_MODE = 0o600


class RunActionTimeoutPublicationError(RuntimeError):
    """A timeout directive or its exact running occurrence is unsafe."""


def publish_run_action_timeout_once(
    *,
    capability: RunActionCommittedContinuationCapability,
    resource_manager: DockerRunActionResourceManager,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    docker_settings: DockerRuntimeSettings,
    launch_settings: LaunchSettings,
) -> RunActionTimeoutDirectivePublicationReceipt | None:
    """Publish only the timeout fact; this function has no signal authority."""

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
        raise RunActionTimeoutPublicationError(
            "timeout publication inputs lack exact configured authority"
        )
    query = capability.query
    adoption = query.workload_release_adoption
    if (
        query.control_directory_topology
        is not RunActionControlDirectoryTopology.RELEASED
        or adoption is None
        or query.timeout_directive_publication is not None
    ):
        raise RunActionTimeoutPublicationError(
            "timeout publication requires one exact released query"
        )
    prepared = query.prepared_execution
    projection = prepared.inert_container_evidence.issued_create_projection
    if (
        command.command_template_id
        != prepared.preparation_claim.execution_policy.command_template_id
        or helper_evidence != projection.supervisor_helper_evidence
        or init_source_evidence != projection.docker_init_source_evidence
        or launch_settings.run_action_timeout_directive_size_bytes
        != prepared.preparation_claim.execution_policy.supervisor_limits.timeout_directive_size_bytes
    ):
        raise RunActionTimeoutPublicationError(
            "timeout publication inputs differ from durable prepared authority"
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
            is not RunActionControlDirectoryTopology.RELEASED
            or control_inspection.workload_release_adoption != adoption
            or control_inspection.timeout_directive_publication is not None
            or read_run_action_host_boot_id(proc_root_descriptor)
            != adoption.workload_release_receipt.host_boot_id
        ):
            raise RunActionTimeoutPublicationError(
                "timeout publication differs from its retained release occurrence"
            )
        authorization = (
            RunActionCommittedContinuationCapability._begin_timeout_publication(
                capability,
                control_inspection,
                _authority=_RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY,
            )
        )
        if authorization is None:
            return None
        with authorization:
            inventory = resource_manager.observe(query.preparation_allocation)
            if (
                inventory.volume_inspection_digest is None
                or inventory.keeper_container_id
                != prepared.volume_keeper_evidence.container_id
                or inventory.main_container_id
                != query.spawn_commit.provider_execution_id
            ):
                raise RunActionTimeoutPublicationError(
                    "timeout publication lacks its exact Docker resource graph"
                )
            volume = observe_runtime_volume(
                resource_manager.inspect_volume(inventory),
                prepared.preparation_claim,
                prepared.runtime_volume_authority,
                docker_settings,
            )
            first_running = observe_running_barrier_main_container(
                resource_manager.inspect_main(inventory),
                prepared.preparation_claim,
                prepared.runtime_volume_authority,
                volume,
                command,
                helper_evidence,
                init_source_evidence,
                docker_settings,
            )
            second_running = observe_running_barrier_main_container(
                resource_manager.inspect_main(inventory),
                prepared.preparation_claim,
                prepared.runtime_volume_authority,
                volume,
                command,
                helper_evidence,
                init_source_evidence,
                docker_settings,
            )
            released_running = (
                adoption.workload_release_receipt.resolved_workload_observation.running_container_observation
            )
            if (
                not run_action_running_container_occurrence_matches(
                    first_running,
                    second_running,
                )
                or not run_action_running_container_occurrence_matches(
                    second_running,
                    released_running,
                )
                or resource_manager.observe(query.preparation_allocation) != inventory
                or read_run_action_host_boot_id(proc_root_descriptor)
                != adoption.workload_release_receipt.host_boot_id
            ):
                raise RunActionTimeoutPublicationError(
                    "timeout publication running occurrence changed during inspection"
                )
            control_inspection.require_current()
            directive = authorization._mint_timeout_directive(
                second_running,
                adoption.workload_release_receipt.host_boot_id,
                _authority=_RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY,
            )
            control_descriptor, release_descriptor = (
                control_inspection._duplicate_timeout_publication_descriptors(
                    descriptors=retained,
                    _authority=_RUN_ACTION_TIMEOUT_PUBLICATION_AUTHORITY,
                )
            )
            anonymous_descriptor = open_run_action_anonymous_file(
                control_descriptor,
                _ANONYMOUS_FILE_MODE,
            )
            retained.callback(os.close, anonymous_descriptor)
            candidate = _RunActionFrozenControlFileCandidate(
                transition=_RunActionControlFileTransition.TIMEOUT,
                control_directory_descriptor=control_descriptor,
                anonymous_file_descriptor=anonymous_descriptor,
                predecessor_file_descriptor=release_descriptor,
                owner_user_id=prepared.control_directory.owner_user_id,
                owner_group_id=prepared.control_directory.owner_group_id,
                payload_size_limit_bytes=(
                    prepared.preparation_claim.execution_policy.supervisor_limits.timeout_directive_size_bytes
                ),
                process_snapshot_size_limit_bytes=(
                    prepared.preparation_claim.execution_policy.supervisor_limits.process_snapshot_size_bytes
                ),
                payload=directive.to_json_bytes(),
                _authority=_CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY,
            )
            retained.callback(candidate.close)
            final_running = observe_running_barrier_main_container(
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
                not run_action_running_container_occurrence_matches(
                    final_running,
                    second_running,
                )
                or resource_manager.observe(query.preparation_allocation) != inventory
                or read_run_action_host_boot_id(proc_root_descriptor)
                != adoption.workload_release_receipt.host_boot_id
            ):
                raise RunActionTimeoutPublicationError(
                    "timeout publication lost the running occurrence before link"
                )
            control_inspection.require_current()
            authorization._authorize_frozen_timeout_once(
                candidate=candidate,
                _authority=_RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY,
            )
            retained.pop_all().close()
            with open_run_action_timeout_inspection(
                activation_event=query.activation_event,
                launch_settings=launch_settings,
            ) as adopted_timeout:
                publication = adopted_timeout.timeout_directive_publication
                if (
                    adopted_timeout.topology
                    is not RunActionControlDirectoryTopology.TIMED_OUT
                    or adopted_timeout.workload_release_adoption != adoption
                    or type(publication)
                    is not RunActionTimeoutDirectivePublicationReceipt
                    or publication.timeout_directive != directive
                ):
                    raise RunActionTimeoutPublicationError(
                        "linked timeout was not adopted as its exact semantic transition"
                    )
                authorization._complete_timeout_publication(
                    adopted_timeout,
                    _authority=_RUN_ACTION_TIMEOUT_PUBLISHER_AUTHORITY,
                )
                adopted_timeout.require_current()
                return publication


__all__ = [
    "RunActionTimeoutPublicationError",
    "publish_run_action_timeout_once",
]
