"""Closed Docker observations for one issued run-action occurrence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierRunningContainerObservation,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION,
    DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
    DockerRunActionCommand,
    docker_run_action_raw_field_schema,
    main_barrier_command,
    target_command_from_barrier_invocation,
    volume_create_arguments,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseAdoption,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    observe_mounted_keeper_helper,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionCreateInspectProjection,
    DockerRunActionKeeperCreateInspectProjection,
    RUN_ACTION_BARRIER_PROTOCOL_VERSION,
    RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionActivationRevalidationReceipt,
    RunActionContainerLabel,
    RunActionDockerInitSourceEvidence,
    RunActionInertContainerEvidence,
    RunActionSupervisorHelperEvidence,
    RunActionPreparationClaim,
    RunActionPreparedMount,
    RunActionPreparedMountAccess,
    RunActionRuntimeVolumeAuthority,
    RunActionTerminalObservation,
    RunActionVolumeKeeperEvidence,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_main_mounts,
    runtime_volume_driver_options,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionPreReleaseTerminalContainerObservation,
)
from kapso.cross_run.settings import DockerRuntimeSettings

_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STORAGE_LAYER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VOLUME_AUTHORITY_ID_PATTERN = re.compile(
    r"^run-action-runtime-volume-authority:sha256:[0-9a-f]{64}$"
)
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T"
    r"([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.[0-9]+)?Z$"
)
_ZERO_DOCKER_TIMESTAMP = "0001-01-01T00:00:00Z"
_ORDER_NORMALIZED_RAW_FIELDS = (
    "Config.Env",
    "HostConfig.Mounts",
    "Mounts",
)
_RUNNING_CONTAINER_NONAUTHORITATIVE_RAW_FIELDS = (
    "HostnamePath",
    "HostsPath",
    "HostConfig.OomKillDisable",
    "NetworkSettings.Networks.none.EndpointID",
    "NetworkSettings.Networks.none.NetworkID",
    "NetworkSettings.SandboxID",
    "NetworkSettings.SandboxKey",
    "ResolvConfPath",
    "State.Pid",
    "State.StartedAt",
)
_MAIN_NONAUTHORITATIVE_RAW_FIELDS = (
    *_ORDER_NORMALIZED_RAW_FIELDS,
    "Created",
    "GraphDriver.Data.ID",
    "GraphDriver.Data.LowerDir",
    "GraphDriver.Data.MergedDir",
    "GraphDriver.Data.UpperDir",
    "GraphDriver.Data.WorkDir",
    "Id",
    *_RUNNING_CONTAINER_NONAUTHORITATIVE_RAW_FIELDS,
)
_KEEPER_NONAUTHORITATIVE_RAW_FIELDS = _MAIN_NONAUTHORITATIVE_RAW_FIELDS
_VOLUME_NONAUTHORITATIVE_RAW_FIELDS = ("CreatedAt", "Mountpoint")


class DockerRunActionInspectionError(ValueError):
    """A Docker observation differs from the complete issued authority."""


@dataclass(frozen=True)
class DockerRunActionVolumeObservation:
    """Closed daemon observation needed to bind container volume mounts."""

    volume_authority_id: str
    volume_occurrence_digest: str
    volume_name: str
    mountpoint: str
    created_at: str
    raw_field_schema_id: str
    unclassified_raw_field_count: int
    nonauthoritative_raw_field_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.volume_authority_id, str)
            or _VOLUME_AUTHORITY_ID_PATTERN.fullmatch(self.volume_authority_id) is None
            or not isinstance(self.volume_occurrence_digest, str)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                self.volume_occurrence_digest,
            )
            is None
            or not isinstance(self.volume_name, str)
            or not self.volume_name
            or not isinstance(self.mountpoint, str)
            or not self.mountpoint
            or not _is_utc_timestamp(self.created_at)
            or self.raw_field_schema_id != DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID
            or self.unclassified_raw_field_count != 0
            or self.nonauthoritative_raw_field_count
            != len(_VOLUME_NONAUTHORITATIVE_RAW_FIELDS)
        ):
            raise DockerRunActionInspectionError(
                "Docker runtime volume observation is incomplete"
            )


@dataclass(frozen=True)
class DockerRunActionInertKeeperObservation:
    """Closed pre-start keeper observation with no activation authority."""

    container_id: str
    issued_create_projection: DockerRunActionKeeperCreateInspectProjection
    observed_inspect_projection: DockerRunActionKeeperCreateInspectProjection

    def __post_init__(self) -> None:
        if (
            type(self.container_id) is not str
            or _CONTAINER_ID_PATTERN.fullmatch(self.container_id) is None
            or type(self.issued_create_projection)
            is not DockerRunActionKeeperCreateInspectProjection
            or type(self.observed_inspect_projection)
            is not DockerRunActionKeeperCreateInspectProjection
            or self.observed_inspect_projection != self.issued_create_projection
        ):
            raise DockerRunActionInspectionError(
                "inert keeper observation differs from its issued projection"
            )


class _DockerContainerLifecycle(str, Enum):
    """Exact lifecycle and role admitted by common container inspection."""

    CREATED_MAIN = "created_main"
    INERT_KEEPER = "inert_keeper"
    RUNNING_KEEPER = "running_keeper"
    RUNNING_MAIN = "running_main"
    EXITED_MAIN = "exited_main"


def observe_runtime_volume(
    raw_inspection: Mapping[str, Any],
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    settings: DockerRuntimeSettings,
) -> DockerRunActionVolumeObservation:
    """Parse one exact local tmpfs volume inspection."""

    volume_create_arguments(claim, authority, settings)
    raw = _require_mapping(raw_inspection, "Docker runtime volume inspection")
    _require_exact_fields(raw, "volume_inspect", "Docker runtime volume")
    options = _require_mapping(raw["Options"], "Docker runtime volume Options")
    _require_exact_fields(
        options,
        "volume_inspect_options",
        "Docker runtime volume Options",
    )
    labels = {label.key: label.value for label in authority.labels}
    expected_mountpoint = (
        PurePosixPath(settings.runtime_root_directory)
        / "volumes"
        / authority.volume_name
        / "_data"
    ).as_posix()
    if (
        raw["Name"] != authority.volume_name
        or raw["Labels"] != labels
        or raw["Scope"] != "local"
        or raw["Driver"] != authority.driver
        or options != _driver_option_mapping(runtime_volume_driver_options(authority))
        or raw["Mountpoint"] != expected_mountpoint
        or not _is_utc_timestamp(raw["CreatedAt"])
        or raw["CreatedAt"] == _ZERO_DOCKER_TIMESTAMP
    ):
        raise DockerRunActionInspectionError(
            "Docker runtime volume differs from issued authority"
        )
    occurrence_digest = tree_or_blob_digest(
        canonical_json_bytes(
            {
                "CreatedAt": raw["CreatedAt"],
                "Driver": raw["Driver"],
                "Labels": raw["Labels"],
                "Mountpoint": raw["Mountpoint"],
                "Name": raw["Name"],
                "Options": raw["Options"],
                "Scope": raw["Scope"],
            }
        )
    )
    return DockerRunActionVolumeObservation(
        volume_authority_id=authority.runtime_volume_authority_id,
        volume_occurrence_digest=occurrence_digest,
        volume_name=authority.volume_name,
        mountpoint=expected_mountpoint,
        created_at=raw["CreatedAt"],
        raw_field_schema_id=DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
        unclassified_raw_field_count=0,
        nonauthoritative_raw_field_count=len(_VOLUME_NONAUTHORITATIVE_RAW_FIELDS),
    )


def issued_main_projection(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
) -> DockerRunActionCreateInspectProjection:
    """Build the normalized main-container projection before allocation."""

    volume_create_arguments(claim, authority, settings)
    _require_command_policy(command, claim)
    _require_helper_evidence(helper_evidence, claim)
    _require_init_source_evidence(init_source_evidence, claim)
    barrier_executable, barrier_arguments = main_barrier_command(
        command,
        authority.generation_nonce,
        settings,
    )
    mounts = preparation_main_mounts(claim, authority)
    return DockerRunActionCreateInspectProjection.mint(
        projection_protocol_version=DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION,
        raw_field_schema_id=DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
        execution_policy=claim.execution_policy,
        supervisor_helper_evidence=helper_evidence,
        docker_init_source_evidence=init_source_evidence,
        barrier_protocol_version=RUN_ACTION_BARRIER_PROTOCOL_VERSION,
        barrier_poll_interval_seconds=(
            settings.run_action_barrier_poll_interval_seconds
        ),
        barrier_generation_nonce=authority.generation_nonce,
        command_executable=barrier_executable,
        command_arguments=barrier_arguments,
        mounts=mounts,
        exact_mount_count=len(mounts) + 1,
        unclassified_raw_field_count=0,
        nonauthoritative_raw_field_count=len(_MAIN_NONAUTHORITATIVE_RAW_FIELDS),
    )


def observe_inert_main_container(
    raw_inspection: Mapping[str, Any],
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
) -> RunActionInertContainerEvidence:
    """Parse one never-started main container and bind it to issuance."""

    issued = issued_main_projection(
        claim,
        authority,
        command,
        helper_evidence,
        init_source_evidence,
        settings,
    )
    _require_volume_observation(volume, authority, settings)
    raw = _require_mapping(raw_inspection, "Docker main container inspection")
    expected_labels = preparation_container_labels(claim)
    expected_mounts = preparation_main_mounts(claim, authority)
    barrier_executable, barrier_arguments = main_barrier_command(
        command,
        authority.generation_nonce,
        settings,
    )
    container_id = _require_common_container(
        raw,
        claim=claim,
        labels=expected_labels,
        container_name=preparation_container_name(claim),
        command_executable=barrier_executable,
        command_arguments=barrier_arguments,
        working_directory=claim.execution_policy.filesystem_policy.working_directory,
        host_config_mounts=_main_host_config_mounts(claim, expected_mounts),
        top_level_mounts=_main_top_level_mounts(claim, expected_mounts, volume),
        settings=settings,
        lifecycle=_DockerContainerLifecycle.CREATED_MAIN,
    )
    return RunActionInertContainerEvidence.mint(
        preparation_claim_id=claim.preparation_claim_id,
        container_id=container_id,
        container_name=preparation_container_name(claim),
        labels=expected_labels,
        image_authority_id=claim.execution_policy.image_authority.image_authority_id,
        docker_runtime_settings_digest=(
            claim.execution_policy.docker_runtime_settings_digest
        ),
        issued_create_projection=issued,
        observed_inspect_projection=issued,
        container_status="created",
        process_id=0,
        restart_count=0,
        started_at=_ZERO_DOCKER_TIMESTAMP,
        finished_at=_ZERO_DOCKER_TIMESTAMP,
        restart_policy_name="no",
        auto_remove=False,
        network_mode="none",
        healthcheck_present=False,
        volume_plugin_mount_count=0,
        docker_socket_mounted=False,
    )


def observe_allocation_inert_main_container(
    raw_inspection: Mapping[str, Any],
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
) -> RunActionInertContainerEvidence:
    """Authenticate and parse an allocation-stage main from durable policy."""

    raw = _require_mapping(
        raw_inspection,
        "Docker allocation-stage main inspection",
    )
    raw_arguments = raw["Args"]
    if type(raw_arguments) is not list or any(
        type(argument) is not str for argument in raw_arguments
    ):
        raise DockerRunActionInspectionError(
            "Docker allocation-stage main arguments are malformed"
        )
    command = target_command_from_barrier_invocation(
        raw["Path"],
        tuple(raw_arguments),
        claim.execution_policy,
    )
    return observe_inert_main_container(
        raw,
        claim,
        authority,
        volume,
        command,
        helper_evidence,
        init_source_evidence,
        settings,
    )


def observe_running_barrier_main_container(
    raw_inspection: Mapping[str, Any],
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
) -> RunActionBarrierRunningContainerObservation:
    """Parse a running main without claiming wrapper generation or resolved mounts."""

    issued = issued_main_projection(
        claim,
        authority,
        command,
        helper_evidence,
        init_source_evidence,
        settings,
    )
    _require_volume_observation(volume, authority, settings)
    raw, complete_inspection_payload, _raw_size_bytes = _snapshot_container_inspection(
        raw_inspection,
        "Docker running main inspection",
    )
    labels = preparation_container_labels(claim)
    mounts = preparation_main_mounts(claim, authority)
    barrier_executable, barrier_arguments = main_barrier_command(
        command,
        authority.generation_nonce,
        settings,
    )
    container_id = _require_common_container(
        raw,
        claim=claim,
        labels=labels,
        container_name=preparation_container_name(claim),
        command_executable=barrier_executable,
        command_arguments=barrier_arguments,
        working_directory=claim.execution_policy.filesystem_policy.working_directory,
        host_config_mounts=_main_host_config_mounts(claim, mounts),
        top_level_mounts=_main_top_level_mounts(claim, mounts, volume),
        settings=settings,
        lifecycle=_DockerContainerLifecycle.RUNNING_MAIN,
    )
    state = raw["State"]
    return RunActionBarrierRunningContainerObservation.mint(
        container_id=container_id,
        observed_inspect_projection=issued,
        complete_inspection_digest=tree_or_blob_digest(complete_inspection_payload),
        container_status=state["Status"],
        init_process_id=state["Pid"],
        restart_count=raw["RestartCount"],
        started_at=state["StartedAt"],
        finished_at=state["FinishedAt"],
        paused=state["Paused"],
        restarting=state["Restarting"],
        dead=state["Dead"],
        oom_killed=state["OOMKilled"],
        state_error=state["Error"],
    )


def observe_terminal_main_container(
    raw_inspection: Mapping[str, Any],
    activation: RunActionActivationRevalidationReceipt,
    workload_release_adoption: RunActionWorkloadReleaseAdoption,
    volume: DockerRunActionVolumeObservation,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
    *,
    inspection_size_limit_bytes: int,
) -> RunActionTerminalObservation:
    """Parse one exact exited main and bind it to its adopted event-5 release."""

    if (
        type(activation) is not RunActionActivationRevalidationReceipt
        or type(workload_release_adoption) is not RunActionWorkloadReleaseAdoption
        or type(inspection_size_limit_bytes) is not int
        or inspection_size_limit_bytes <= 0
    ):
        raise DockerRunActionInspectionError(
            "Docker terminal inspection lacks exact activation authority"
        )
    prepared = activation.prepared_execution
    authority = prepared.runtime_volume_authority
    issued, container_id, state, restart_count, complete_inspection_digest = (
        _observe_exited_main_container_snapshot(
            raw_inspection=raw_inspection,
            activation=activation,
            volume=volume,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            settings=settings,
            inspection_size_limit_bytes=inspection_size_limit_bytes,
        )
    )
    release_receipt = workload_release_adoption.workload_release_receipt
    released = release_receipt.resolved_workload_observation
    if (
        release_receipt.resolved_workload_observation.activation_revalidation_receipt
        != activation
        or container_id != activation.spawn_commit.provider_execution_id
        or container_id != prepared.inert_container_evidence.container_id
        or container_id != released.running_container_observation.container_id
        or state["StartedAt"] != released.running_container_observation.started_at
        or issued != prepared.inert_container_evidence.issued_create_projection
    ):
        raise DockerRunActionInspectionError(
            "Docker terminal main differs from its adopted released occurrence"
        )
    return RunActionTerminalObservation.mint(
        prepared_execution_id=prepared.prepared_execution_id,
        spawn_commit_id=activation.spawn_commit.spawn_commit_id,
        provider_execution_id=activation.spawn_commit.provider_execution_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        activation_revalidation_receipt_id=(
            activation.activation_revalidation_receipt_id
        ),
        workload_release_adoption_id=(
            workload_release_adoption.workload_release_adoption_id
        ),
        observed_inspect_projection=issued,
        complete_inspection_digest=complete_inspection_digest,
        container_status=state["Status"],
        process_id=state["Pid"],
        restart_count=restart_count,
        paused=state["Paused"],
        restarting=state["Restarting"],
        dead=state["Dead"],
        started_at=state["StartedAt"],
        finished_at=state["FinishedAt"],
        exit_code=state["ExitCode"],
        oom_killed=state["OOMKilled"],
        state_error=state["Error"],
    )


def observe_pre_release_terminal_main_container(
    raw_inspection: Mapping[str, Any],
    activation: RunActionActivationRevalidationReceipt,
    volume: DockerRunActionVolumeObservation,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
    *,
    inspection_size_limit_bytes: int,
) -> RunActionPreReleaseTerminalContainerObservation:
    """Parse one exact exited event-5 main without inventing release authority."""

    if (
        type(activation) is not RunActionActivationRevalidationReceipt
        or type(inspection_size_limit_bytes) is not int
        or inspection_size_limit_bytes <= 0
    ):
        raise DockerRunActionInspectionError(
            "Docker pre-release terminal inspection lacks activation authority"
        )
    prepared = activation.prepared_execution
    authority = prepared.runtime_volume_authority
    issued, container_id, state, restart_count, complete_inspection_digest = (
        _observe_exited_main_container_snapshot(
            raw_inspection=raw_inspection,
            activation=activation,
            volume=volume,
            command=command,
            helper_evidence=helper_evidence,
            init_source_evidence=init_source_evidence,
            settings=settings,
            inspection_size_limit_bytes=inspection_size_limit_bytes,
        )
    )
    if (
        container_id != activation.spawn_commit.provider_execution_id
        or container_id != prepared.inert_container_evidence.container_id
        or issued != prepared.inert_container_evidence.issued_create_projection
    ):
        raise DockerRunActionInspectionError(
            "Docker pre-release terminal main differs from event-5 authority"
        )
    return RunActionPreReleaseTerminalContainerObservation.mint(
        prepared_execution_id=prepared.prepared_execution_id,
        spawn_commit_id=activation.spawn_commit.spawn_commit_id,
        provider_execution_id=activation.spawn_commit.provider_execution_id,
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        activation_revalidation_receipt_id=(
            activation.activation_revalidation_receipt_id
        ),
        observed_inspect_projection=issued,
        complete_inspection_digest=complete_inspection_digest,
        container_status=state["Status"],
        process_id=state["Pid"],
        restart_count=restart_count,
        paused=state["Paused"],
        restarting=state["Restarting"],
        dead=state["Dead"],
        started_at=state["StartedAt"],
        finished_at=state["FinishedAt"],
        exit_code=state["ExitCode"],
        oom_killed=state["OOMKilled"],
        state_error=state["Error"],
    )


def _observe_exited_main_container_snapshot(
    *,
    raw_inspection: Mapping[str, Any],
    activation: RunActionActivationRevalidationReceipt,
    volume: DockerRunActionVolumeObservation,
    command: DockerRunActionCommand,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
    inspection_size_limit_bytes: int,
) -> tuple[
    DockerRunActionCreateInspectProjection,
    str,
    Mapping[str, Any],
    int,
    str,
]:
    prepared = activation.prepared_execution
    claim = prepared.preparation_claim
    authority = prepared.runtime_volume_authority
    issued = issued_main_projection(
        claim,
        authority,
        command,
        helper_evidence,
        init_source_evidence,
        settings,
    )
    _require_volume_observation(volume, authority, settings)
    raw, complete_inspection_payload, raw_size_bytes = _snapshot_container_inspection(
        raw_inspection,
        "Docker terminal main inspection",
    )
    if raw_size_bytes > inspection_size_limit_bytes:
        raise DockerRunActionInspectionError(
            "Docker terminal main inspection exceeds its configured bound"
        )
    labels = preparation_container_labels(claim)
    mounts = preparation_main_mounts(claim, authority)
    barrier_executable, barrier_arguments = main_barrier_command(
        command,
        authority.generation_nonce,
        settings,
    )
    container_id = _require_common_container(
        raw,
        claim=claim,
        labels=labels,
        container_name=preparation_container_name(claim),
        command_executable=barrier_executable,
        command_arguments=barrier_arguments,
        working_directory=claim.execution_policy.filesystem_policy.working_directory,
        host_config_mounts=_main_host_config_mounts(claim, mounts),
        top_level_mounts=_main_top_level_mounts(claim, mounts, volume),
        settings=settings,
        lifecycle=_DockerContainerLifecycle.EXITED_MAIN,
    )
    return (
        issued,
        container_id,
        _require_mapping(raw["State"], "Docker terminal main State"),
        raw["RestartCount"],
        tree_or_blob_digest(complete_inspection_payload),
    )


def reobserve_terminal_main_container_for_cleanup(
    raw_inspection: Mapping[str, Any],
    terminal: RunActionTerminalObservation,
) -> RunActionTerminalObservation:
    """Reprove the complete durable terminal snapshot before exact deletion."""

    if type(terminal) is not RunActionTerminalObservation:
        raise DockerRunActionInspectionError(
            "Docker terminal cleanup lacks durable terminal authority"
        )
    _require_reobserved_exited_main_container(
        raw_inspection=raw_inspection,
        complete_inspection_digest=terminal.complete_inspection_digest,
        provider_execution_id=terminal.provider_execution_id,
        container_status=terminal.container_status,
        process_id=terminal.process_id,
        restart_count=terminal.restart_count,
        paused=terminal.paused,
        restarting=terminal.restarting,
        dead=terminal.dead,
        started_at=terminal.started_at,
        finished_at=terminal.finished_at,
        exit_code=terminal.exit_code,
        oom_killed=terminal.oom_killed,
        state_error=terminal.state_error,
    )
    return terminal


def reobserve_pre_release_terminal_main_container_for_cleanup(
    raw_inspection: Mapping[str, Any],
    terminal: RunActionPreReleaseTerminalContainerObservation,
) -> RunActionPreReleaseTerminalContainerObservation:
    """Reprove one durable pre-release terminal snapshot before exact deletion."""

    if type(terminal) is not RunActionPreReleaseTerminalContainerObservation:
        raise DockerRunActionInspectionError(
            "Docker pre-release terminal cleanup lacks durable authority"
        )
    _require_reobserved_exited_main_container(
        raw_inspection=raw_inspection,
        complete_inspection_digest=terminal.complete_inspection_digest,
        provider_execution_id=terminal.provider_execution_id,
        container_status=terminal.container_status,
        process_id=terminal.process_id,
        restart_count=terminal.restart_count,
        paused=terminal.paused,
        restarting=terminal.restarting,
        dead=terminal.dead,
        started_at=terminal.started_at,
        finished_at=terminal.finished_at,
        exit_code=terminal.exit_code,
        oom_killed=terminal.oom_killed,
        state_error=terminal.state_error,
    )
    return terminal


def _require_reobserved_exited_main_container(
    *,
    raw_inspection: Mapping[str, Any],
    complete_inspection_digest: str,
    provider_execution_id: str,
    container_status: str,
    process_id: int,
    restart_count: int,
    paused: bool,
    restarting: bool,
    dead: bool,
    started_at: str,
    finished_at: str,
    exit_code: int,
    oom_killed: bool,
    state_error: str,
) -> None:
    raw, normalized_payload, _raw_size_bytes = _snapshot_container_inspection(
        raw_inspection,
        "Docker terminal cleanup main inspection",
    )
    state = _require_mapping(
        raw["State"],
        "Docker terminal cleanup main State",
    )
    if (
        tree_or_blob_digest(normalized_payload) != complete_inspection_digest
        or raw["Id"] != provider_execution_id
        or state["Status"] != container_status
        or state["Pid"] != process_id
        or raw["RestartCount"] != restart_count
        or state["Paused"] != paused
        or state["Restarting"] != restarting
        or state["Dead"] != dead
        or state["StartedAt"] != started_at
        or state["FinishedAt"] != finished_at
        or state["ExitCode"] != exit_code
        or state["OOMKilled"] != oom_killed
        or state["Error"] != state_error
    ):
        raise DockerRunActionInspectionError(
            "Docker terminal cleanup main differs from durable terminal authority"
        )


def issued_keeper_projection(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
) -> DockerRunActionKeeperCreateInspectProjection:
    """Build the normalized keeper projection before allocation."""

    volume_create_arguments(claim, authority, settings)
    _require_helper_evidence(helper_evidence, claim)
    _require_init_source_evidence(init_source_evidence, claim)
    return DockerRunActionKeeperCreateInspectProjection.mint(
        projection_protocol_version=DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION,
        raw_field_schema_id=DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
        preparation_claim_id=claim.preparation_claim_id,
        execution_policy=claim.execution_policy,
        volume_authority=authority,
        command_executable=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        command_arguments=("tail", "-f", "/dev/null"),
        helper_evidence=helper_evidence,
        docker_init_source_evidence=init_source_evidence,
        volume_mount_type="volume",
        volume_mount_destination=RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        volume_mount_access=RunActionPreparedMountAccess.READ_WRITE,
        network_mode="none",
        exact_mount_count=2,
        healthcheck_present=False,
        docker_socket_mounted=False,
        unclassified_raw_field_count=0,
        nonauthoritative_raw_field_count=len(_KEEPER_NONAUTHORITATIVE_RAW_FIELDS),
    )


def observe_running_keeper(
    raw_inspection: Mapping[str, Any],
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
) -> RunActionVolumeKeeperEvidence:
    """Parse one running, network-free runtime-volume keeper."""

    issued = issued_keeper_projection(
        claim,
        authority,
        helper_evidence,
        init_source_evidence,
        settings,
    )
    _require_volume_observation(volume, authority, settings)
    raw = _require_mapping(raw_inspection, "Docker keeper inspection")
    labels = preparation_keeper_container_labels(claim)
    container_id = _require_common_container(
        raw,
        claim=claim,
        labels=labels,
        container_name=preparation_keeper_container_name(claim),
        command_executable=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        command_arguments=("tail", "-f", "/dev/null"),
        working_directory=RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        host_config_mounts=_keeper_host_config_mounts(claim, authority),
        top_level_mounts=_keeper_top_level_mounts(
            claim,
            authority,
            volume,
        ),
        settings=settings,
        lifecycle=_DockerContainerLifecycle.RUNNING_KEEPER,
    )
    state = raw["State"]
    mounted_helper_evidence = observe_mounted_keeper_helper(
        helper_evidence,
        container_id=container_id,
        process_id=state["Pid"],
        process_snapshot_size_limit_bytes=(
            claim.execution_policy.supervisor_limits.process_snapshot_size_bytes
        ),
    )
    return RunActionVolumeKeeperEvidence.mint(
        preparation_claim_id=claim.preparation_claim_id,
        container_id=container_id,
        container_name=preparation_keeper_container_name(claim),
        labels=labels,
        issued_create_projection=issued,
        observed_inspect_projection=issued,
        mounted_helper_evidence=mounted_helper_evidence,
        container_status="running",
        process_id=state["Pid"],
        process_start_time_ticks=mounted_helper_evidence.process_start_time_ticks,
        restart_count=0,
        restart_policy_name="no",
        auto_remove=False,
    )


def observe_inert_keeper(
    raw_inspection: Mapping[str, Any],
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
) -> DockerRunActionInertKeeperObservation:
    """Parse one never-started keeper before issuing its sole start mutation."""

    issued = issued_keeper_projection(
        claim,
        authority,
        helper_evidence,
        init_source_evidence,
        settings,
    )
    _require_volume_observation(volume, authority, settings)
    raw = _require_mapping(raw_inspection, "Docker inert keeper inspection")
    labels = preparation_keeper_container_labels(claim)
    container_id = _require_common_container(
        raw,
        claim=claim,
        labels=labels,
        container_name=preparation_keeper_container_name(claim),
        command_executable=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        command_arguments=("tail", "-f", "/dev/null"),
        working_directory=RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        host_config_mounts=_keeper_host_config_mounts(claim, authority),
        top_level_mounts=_keeper_top_level_mounts(
            claim,
            authority,
            volume,
        ),
        settings=settings,
        lifecycle=_DockerContainerLifecycle.INERT_KEEPER,
    )
    return DockerRunActionInertKeeperObservation(
        container_id=container_id,
        issued_create_projection=issued,
        observed_inspect_projection=issued,
    )


def observe_allocation_keeper(
    raw_inspection: Mapping[str, Any],
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    helper_evidence: RunActionSupervisorHelperEvidence,
    init_source_evidence: RunActionDockerInitSourceEvidence,
    settings: DockerRuntimeSettings,
) -> DockerRunActionInertKeeperObservation | RunActionVolumeKeeperEvidence:
    """Parse only a never-started or exact running allocation-stage keeper."""

    raw = _require_mapping(
        raw_inspection,
        "Docker allocation-stage keeper inspection",
    )
    state = _require_mapping(
        raw["State"],
        "Docker allocation-stage keeper State",
    )
    status = state["Status"]
    if status == "created":
        return observe_inert_keeper(
            raw,
            claim,
            authority,
            volume,
            helper_evidence,
            init_source_evidence,
            settings,
        )
    if status == "running":
        return observe_running_keeper(
            raw,
            claim,
            authority,
            volume,
            helper_evidence,
            init_source_evidence,
            settings,
        )
    raise DockerRunActionInspectionError(
        "Docker allocation-stage keeper lifecycle is not removable"
    )


def _require_common_container(
    raw: Mapping[str, Any],
    *,
    claim: RunActionPreparationClaim,
    labels: tuple[RunActionContainerLabel, ...],
    container_name: str,
    command_executable: str,
    command_arguments: tuple[str, ...],
    working_directory: str,
    host_config_mounts: list[dict[str, Any]],
    top_level_mounts: list[dict[str, Any]],
    settings: DockerRuntimeSettings,
    lifecycle: _DockerContainerLifecycle,
) -> str:
    if type(lifecycle) is not _DockerContainerLifecycle:
        raise DockerRunActionInspectionError(
            "Docker container lifecycle mode is invalid"
        )
    _require_exact_fields(raw, "container_root", "Docker container")
    container_id = raw["Id"]
    if (
        not isinstance(container_id, str)
        or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
    ):
        raise DockerRunActionInspectionError("Docker container identity is malformed")
    policy = claim.execution_policy
    config = _require_mapping(raw["Config"], "Docker container Config")
    host_config = _require_mapping(
        raw["HostConfig"],
        "Docker container HostConfig",
    )
    state = _require_mapping(raw["State"], "Docker container State")
    graph_driver = _require_mapping(
        raw["GraphDriver"],
        "Docker container GraphDriver",
    )
    network_settings = _require_mapping(
        raw["NetworkSettings"],
        "Docker container NetworkSettings",
    )
    _require_exact_fields(config, "container_config", "Docker container Config")
    _require_exact_fields(
        host_config,
        "host_config",
        "Docker container HostConfig",
    )
    _require_exact_fields(state, "container_state", "Docker container State")
    _require_graph_driver(graph_driver, container_id, settings)
    is_running = lifecycle in {
        _DockerContainerLifecycle.RUNNING_KEEPER,
        _DockerContainerLifecycle.RUNNING_MAIN,
    }
    has_started = is_running or lifecycle is _DockerContainerLifecycle.EXITED_MAIN
    _require_network(network_settings, lifecycle=lifecycle)
    expected_config = _expected_container_config(
        claim,
        labels=labels,
        command_executable=command_executable,
        command_arguments=command_arguments,
        working_directory=working_directory,
    )
    observed_config = dict(config)
    observed_config["Env"] = _normalized_environment(config["Env"])
    expected_config["Env"] = _normalized_environment(expected_config["Env"])
    expected_host_config = _expected_host_config(
        claim,
        mounts=host_config_mounts,
        lifecycle=lifecycle,
    )
    observed_host_config = dict(host_config)
    observed_host_config["Mounts"] = _normalized_mounts(
        host_config["Mounts"],
        host_config=True,
    )
    expected_host_config["Mounts"] = _normalized_mounts(
        expected_host_config["Mounts"],
        host_config=True,
    )
    observed_top_level_mounts = _normalized_mounts(
        raw["Mounts"],
        host_config=False,
    )
    normalized_top_level_mounts = _normalized_mounts(
        top_level_mounts,
        host_config=False,
    )
    expected_daemon_paths = _daemon_managed_paths(
        container_id,
        settings,
        has_started=has_started,
    )
    expected_root_fields = {
        "AppArmorProfile": policy.sandbox_spec.apparmor_profile_id,
        "Args": list(command_arguments),
        "Driver": settings.runtime_storage_driver,
        "ExecIDs": None,
        "HostnamePath": expected_daemon_paths["HostnamePath"],
        "HostsPath": expected_daemon_paths["HostsPath"],
        "Image": policy.image_authority.image_config_digest,
        "LogPath": "",
        "MountLabel": "",
        "Mounts": normalized_top_level_mounts,
        "Name": f"/{container_name}",
        "Path": command_executable,
        "Platform": policy.image_authority.operating_system,
        "ProcessLabel": "",
        "ResolvConfPath": expected_daemon_paths["ResolvConfPath"],
        "RestartCount": 0,
    }
    if not _is_utc_timestamp(raw["Created"]) or raw["Created"] == (
        _ZERO_DOCKER_TIMESTAMP
    ):
        raise DockerRunActionInspectionError(
            "Docker container Created is not one valid nonzero timestamp"
        )
    _require_exact_value(observed_config, expected_config, "Config")
    _require_exact_value(
        observed_host_config,
        expected_host_config,
        "HostConfig",
    )
    for field_name, expected_value in expected_root_fields.items():
        observed_value = (
            observed_top_level_mounts if field_name == "Mounts" else raw[field_name]
        )
        _require_exact_value(observed_value, expected_value, field_name)
    _require_container_state(state, lifecycle=lifecycle)
    return container_id


def _expected_container_config(
    claim: RunActionPreparationClaim,
    *,
    labels: tuple[RunActionContainerLabel, ...],
    command_executable: str,
    command_arguments: tuple[str, ...],
    working_directory: str,
) -> dict[str, Any]:
    policy = claim.execution_policy
    return {
        "AttachStderr": True,
        "AttachStdin": False,
        "AttachStdout": True,
        "Cmd": list(command_arguments),
        "Domainname": "",
        "Entrypoint": [command_executable],
        "Env": [
            f"{variable.key}={variable.value}" for variable in policy.static_environment
        ],
        "Hostname": policy.hostname,
        "Image": policy.image_authority.image_reference,
        "Labels": {label.key: label.value for label in labels},
        "OpenStdin": False,
        "StdinOnce": False,
        "StopSignal": "SIGTERM",
        "StopTimeout": policy.supervisor_limits.termination_grace_seconds,
        "Tty": False,
        "User": f"{policy.user_id}:{policy.group_id}",
        "Volumes": None,
        "WorkingDir": working_directory,
    }


def _expected_host_config(
    claim: RunActionPreparationClaim,
    *,
    mounts: list[dict[str, Any]],
    lifecycle: _DockerContainerLifecycle,
) -> dict[str, Any]:
    if type(lifecycle) is not _DockerContainerLifecycle:
        raise DockerRunActionInspectionError(
            "Docker host configuration lifecycle mode is invalid"
        )
    policy = claim.execution_policy
    limits = policy.docker_resource_limits
    sandbox = policy.sandbox_spec
    provider_transition = lifecycle in {
        _DockerContainerLifecycle.CREATED_MAIN,
        _DockerContainerLifecycle.RUNNING_MAIN,
        _DockerContainerLifecycle.EXITED_MAIN,
    } and bool(sandbox.capability_additions)
    security_options = (
        sandbox.security_option_ids
        if provider_transition or not sandbox.capability_additions
        else (
            "apparmor:docker-default",
            "no-new-privileges",
            "seccomp:builtin",
        )
    )
    has_started = lifecycle in {
        _DockerContainerLifecycle.RUNNING_KEEPER,
        _DockerContainerLifecycle.RUNNING_MAIN,
        _DockerContainerLifecycle.EXITED_MAIN,
    }
    return {
        "AutoRemove": False,
        "Binds": None,
        "BlkioDeviceReadBps": [],
        "BlkioDeviceReadIOps": [],
        "BlkioDeviceWriteBps": [],
        "BlkioDeviceWriteIOps": [],
        "BlkioWeight": limits.block_io_weight,
        "BlkioWeightDevice": [],
        "CapAdd": (
            None
            if not provider_transition
            else [f"CAP_{value}" for value in sandbox.capability_additions]
        ),
        "CapDrop": ["ALL"],
        "Cgroup": "",
        "CgroupParent": sandbox.cgroup_parent_id,
        "CgroupnsMode": sandbox.cgroup_namespace_mode,
        "ConsoleSize": [0, 0],
        "ContainerIDFile": "",
        "CpuCount": 0,
        "CpuPercent": 0,
        "CpuPeriod": limits.cpu_period_microseconds,
        "CpuQuota": limits.cpu_quota_microseconds,
        "CpuRealtimePeriod": 0,
        "CpuRealtimeRuntime": 0,
        "CpuShares": limits.cpu_shares,
        "CpusetCpus": ",".join(str(value) for value in limits.cpuset_cpu_ids),
        "CpusetMems": ",".join(str(value) for value in limits.cpuset_memory_node_ids),
        "DeviceCgroupRules": None,
        "DeviceRequests": None,
        "Devices": [],
        "Dns": None,
        "DnsOptions": [],
        "DnsSearch": [],
        "ExtraHosts": None,
        "GroupAdd": (
            None
            if not provider_transition
            else [str(value) for value in sandbox.supplementary_group_ids]
        ),
        "IOMaximumBandwidth": 0,
        "IOMaximumIOps": 0,
        "Init": True,
        "IpcMode": sandbox.ipc_namespace_mode,
        "Isolation": "",
        "Links": None,
        "LogConfig": {"Config": {}, "Type": sandbox.log_driver},
        "MaskedPaths": list(sandbox.masked_system_paths),
        "Memory": limits.memory_size_bytes,
        "MemoryReservation": limits.memory_reservation_size_bytes,
        "MemorySwap": limits.memory_swap_size_bytes,
        "MemorySwappiness": None,
        "Mounts": mounts,
        "NanoCpus": 0,
        "NetworkMode": "none",
        "OomKillDisable": None if has_started else False,
        "OomScoreAdj": limits.oom_score_adjustment,
        "PidMode": "",
        "PidsLimit": limits.process_limit,
        "PortBindings": {},
        "Privileged": False,
        "PublishAllPorts": False,
        "ReadonlyPaths": list(sandbox.read_only_system_paths),
        "ReadonlyRootfs": True,
        "RestartPolicy": {"MaximumRetryCount": 0, "Name": "no"},
        "Runtime": sandbox.runtime_id,
        "SecurityOpt": list(security_options),
        "ShmSize": limits.shared_memory_size_bytes,
        "UTSMode": "",
        "Ulimits": [],
        "UsernsMode": "",
        "VolumeDriver": "",
        "VolumesFrom": None,
    }


def _main_host_config_mounts(
    claim: RunActionPreparationClaim,
    mounts: tuple[RunActionPreparedMount, ...],
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = [
        {
            "BindOptions": {
                "NonRecursive": True,
                "Propagation": "rprivate",
            },
            "ReadOnly": True,
            "Source": claim.execution_policy.supervisor_helper_source_path,
            "Target": RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            "Type": "bind",
        }
    ]
    for mount in mounts:
        observed: dict[str, Any] = {
            "Source": mount.volume_name,
            "Target": mount.container_destination,
            "Type": "volume",
            "VolumeOptions": {
                "DriverConfig": {},
                "NoCopy": True,
                "Subpath": mount.host_config_volume_subpath,
            },
        }
        if mount.container_access is RunActionPreparedMountAccess.READ_ONLY:
            observed["ReadOnly"] = True
        rendered.append(observed)
    return rendered


def _main_top_level_mounts(
    claim: RunActionPreparationClaim,
    mounts: tuple[RunActionPreparedMount, ...],
    volume: DockerRunActionVolumeObservation,
) -> list[dict[str, Any]]:
    return [
        {
            "Destination": RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            "Mode": "",
            "Propagation": "rprivate",
            "RW": False,
            "Source": claim.execution_policy.supervisor_helper_source_path,
            "Type": "bind",
        },
        *[
            {
                "Destination": mount.container_destination,
                "Driver": "local",
                "Mode": "z",
                "Name": mount.volume_name,
                "Propagation": "",
                "RW": (
                    mount.container_access is RunActionPreparedMountAccess.READ_WRITE
                ),
                "Source": volume.mountpoint,
                "Type": "volume",
            }
            for mount in mounts
        ],
    ]


def _keeper_host_config_mounts(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
) -> list[dict[str, Any]]:
    return [
        {
            "BindOptions": {
                "NonRecursive": True,
                "Propagation": "rprivate",
            },
            "ReadOnly": True,
            "Source": claim.execution_policy.supervisor_helper_source_path,
            "Target": RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            "Type": "bind",
        },
        {
            "Source": authority.volume_name,
            "Target": RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
            "Type": "volume",
            "VolumeOptions": {
                "DriverConfig": {},
                "NoCopy": True,
            },
        },
    ]


def _keeper_top_level_mounts(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
) -> list[dict[str, Any]]:
    return [
        {
            "Destination": RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
            "Mode": "",
            "Propagation": "rprivate",
            "RW": False,
            "Source": claim.execution_policy.supervisor_helper_source_path,
            "Type": "bind",
        },
        {
            "Destination": RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
            "Driver": "local",
            "Mode": "z",
            "Name": authority.volume_name,
            "Propagation": "",
            "RW": True,
            "Source": volume.mountpoint,
            "Type": "volume",
        },
    ]


def _require_graph_driver(
    graph_driver: Mapping[str, Any],
    container_id: str,
    settings: DockerRuntimeSettings,
) -> None:
    _require_exact_fields(
        graph_driver,
        "graph_driver",
        "Docker container GraphDriver",
    )
    data = _require_mapping(
        graph_driver["Data"],
        "Docker container GraphDriver.Data",
    )
    _require_exact_fields(
        data,
        "graph_driver_data",
        "Docker container GraphDriver.Data",
    )
    storage_root = (
        PurePosixPath(settings.runtime_root_directory) / settings.runtime_storage_driver
    )
    upper = _require_storage_path(data["UpperDir"], storage_root, "diff")
    storage_layer_id = upper.parent.name
    lower_paths = (
        data["LowerDir"].split(":") if isinstance(data["LowerDir"], str) else []
    )
    if (
        graph_driver["Name"] != settings.runtime_storage_driver
        or data["ID"] != container_id
        or _STORAGE_LAYER_ID_PATTERN.fullmatch(storage_layer_id) is None
        or data["MergedDir"] != (upper.parent / "merged").as_posix()
        or data["WorkDir"] != (upper.parent / "work").as_posix()
        or len(lower_paths) < 2
        or lower_paths[0]
        != (storage_root / f"{storage_layer_id}-init" / "diff").as_posix()
        or any(
            _require_storage_path(path, storage_root, "diff").parent.name.endswith(
                "-init"
            )
            for path in lower_paths[1:]
        )
        or any(
            _STORAGE_LAYER_ID_PATTERN.fullmatch(PurePosixPath(path).parent.name) is None
            for path in lower_paths[1:]
        )
    ):
        raise DockerRunActionInspectionError(
            "Docker container GraphDriver differs from closed overlay2 structure"
        )


def _require_storage_path(
    value: Any,
    storage_root: PurePosixPath,
    basename: str,
) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DockerRunActionInspectionError(
            "Docker container storage path is malformed"
        )
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or path.name != basename
        or path.parent.parent != storage_root
    ):
        raise DockerRunActionInspectionError(
            "Docker container storage path escapes Docker root"
        )
    return path


def _require_network(
    network_settings: Mapping[str, Any],
    *,
    lifecycle: _DockerContainerLifecycle,
) -> None:
    if type(lifecycle) is not _DockerContainerLifecycle:
        raise DockerRunActionInspectionError("Docker network lifecycle mode is invalid")
    _require_exact_fields(
        network_settings,
        "network_settings",
        "Docker container NetworkSettings",
    )
    networks = _require_mapping(
        network_settings["Networks"],
        "Docker container Networks",
    )
    if set(networks) != {"none"}:
        raise DockerRunActionInspectionError(
            "Docker container has an unexpected network"
        )
    network = _require_mapping(networks["none"], "Docker none network")
    _require_exact_fields(network, "network_none", "Docker none network")
    expected_defaults = {
        "Aliases": None,
        "DNSNames": None,
        "DriverOpts": None,
        "Gateway": "",
        "GlobalIPv6Address": "",
        "GlobalIPv6PrefixLen": 0,
        "GwPriority": 0,
        "IPAMConfig": None,
        "IPAddress": "",
        "IPPrefixLen": 0,
        "IPv6Gateway": "",
        "Links": None,
        "MacAddress": "",
    }
    if not _exact_value_matches(network_settings["Ports"], {}) or any(
        not _exact_value_matches(network[field], value)
        for field, value in expected_defaults.items()
    ):
        raise DockerRunActionInspectionError(
            "Docker none network exposes unexpected addressing or ports"
        )
    if lifecycle in {
        _DockerContainerLifecycle.RUNNING_KEEPER,
        _DockerContainerLifecycle.RUNNING_MAIN,
    }:
        sandbox_id = network_settings["SandboxID"]
        if (
            not isinstance(sandbox_id, str)
            or _CONTAINER_ID_PATTERN.fullmatch(sandbox_id) is None
            or network_settings["SandboxKey"]
            != f"/var/run/docker/netns/{sandbox_id[:12]}"
            or not isinstance(network["EndpointID"], str)
            or _CONTAINER_ID_PATTERN.fullmatch(network["EndpointID"]) is None
            or not isinstance(network["NetworkID"], str)
            or _CONTAINER_ID_PATTERN.fullmatch(network["NetworkID"]) is None
        ):
            raise DockerRunActionInspectionError(
                "Docker keeper none-network identity is malformed"
            )
    elif lifecycle is _DockerContainerLifecycle.EXITED_MAIN:
        if (
            network_settings["SandboxID"] != ""
            or network_settings["SandboxKey"] != ""
            or network["EndpointID"] != ""
            or not isinstance(network["NetworkID"], str)
            or _CONTAINER_ID_PATTERN.fullmatch(network["NetworkID"]) is None
        ):
            raise DockerRunActionInspectionError(
                "Docker exited container network residue is malformed"
            )
    elif (
        network_settings["SandboxID"] != ""
        or network_settings["SandboxKey"] != ""
        or network["EndpointID"] != ""
        or network["NetworkID"] != ""
    ):
        raise DockerRunActionInspectionError(
            "Docker inert container unexpectedly joined a network"
        )


def _normalized_mounts(
    value: Any,
    *,
    host_config: bool,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise DockerRunActionInspectionError("Docker container mounts are not a list")
    normalized: dict[str, Mapping[str, Any]] = {}
    for item in value:
        mount = _require_mapping(item, "Docker container mount")
        mount_type = mount.get("Type")
        destination_field = "Target" if host_config else "Destination"
        destination = mount.get(destination_field)
        if (
            mount_type not in {"bind", "volume"}
            or not isinstance(destination, str)
            or not destination
            or destination in normalized
        ):
            raise DockerRunActionInspectionError(
                "Docker container mount identity is malformed or duplicated"
            )
        if host_config and mount_type == "volume":
            mount_schema_name = (
                "host_config_read_only_volume_mount"
                if "ReadOnly" in mount
                else "host_config_read_write_volume_mount"
            )
            _require_exact_fields(
                mount,
                mount_schema_name,
                "Docker container mount",
            )
        else:
            mount_schema_name = (
                "host_config_bind_mount"
                if host_config
                else f"top_level_{mount_type}_mount"
            )
            _require_exact_fields(
                mount,
                mount_schema_name,
                "Docker container mount",
            )
        if host_config and mount_type == "volume":
            options = _require_mapping(
                mount["VolumeOptions"],
                "Docker container volume options",
            )
            options_schema_name = (
                "host_config_root_volume_options"
                if "Subpath" not in options
                else "host_config_subpath_volume_options"
            )
            _require_exact_fields(
                options,
                options_schema_name,
                "Docker container volume options",
            )
        if host_config and mount_type == "bind":
            bind_options = _require_mapping(
                mount["BindOptions"],
                "Docker container bind options",
            )
            _require_exact_fields(
                bind_options,
                "host_config_bind_options",
                "Docker container bind options",
            )
        normalized[destination] = mount
    return [normalized[destination] for destination in sorted(normalized)]


def _normalized_environment(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or any(
            not isinstance(assignment, str)
            or not assignment
            or "\x00" in assignment
            or "=" not in assignment
            for assignment in value
        )
        or len({assignment.split("=", 1)[0] for assignment in value}) != len(value)
    ):
        raise DockerRunActionInspectionError(
            "Docker container environment is malformed or duplicated"
        )
    return sorted(value)


def _snapshot_container_inspection(
    raw_inspection: Mapping[str, Any],
    name: str,
) -> tuple[Mapping[str, Any], bytes, int]:
    supplied = _require_mapping(raw_inspection, name)
    raw_payload = canonical_json_bytes(supplied)
    raw = _require_mapping(
        parse_json_bytes(raw_payload),
        f"{name} snapshot",
    )
    normalized = dict(raw)
    config = dict(_require_mapping(raw["Config"], f"{name} Config"))
    config["Env"] = _normalized_environment(config["Env"])
    normalized["Config"] = config
    host_config = dict(_require_mapping(raw["HostConfig"], f"{name} HostConfig"))
    host_config["Mounts"] = _normalized_mounts(
        host_config["Mounts"],
        host_config=True,
    )
    normalized["HostConfig"] = host_config
    normalized["Mounts"] = _normalized_mounts(
        raw["Mounts"],
        host_config=False,
    )
    normalized_payload = canonical_json_bytes(normalized)
    return (
        _require_mapping(
            parse_json_bytes(normalized_payload),
            f"{name} normalized snapshot",
        ),
        normalized_payload,
        len(raw_payload),
    )


def _require_container_state(
    state: Mapping[str, Any],
    *,
    lifecycle: _DockerContainerLifecycle,
) -> None:
    if type(lifecycle) is not _DockerContainerLifecycle:
        raise DockerRunActionInspectionError(
            "Docker container state lifecycle mode is invalid"
        )
    expected_common = {
        "Dead": False,
        "Error": "",
        "Paused": False,
        "Restarting": False,
    }
    if any(
        not _exact_value_matches(state[field], value)
        for field, value in expected_common.items()
    ):
        raise DockerRunActionInspectionError(
            "Docker container state differs from safe lifecycle"
        )
    if lifecycle in {
        _DockerContainerLifecycle.RUNNING_KEEPER,
        _DockerContainerLifecycle.RUNNING_MAIN,
    }:
        if (
            state["ExitCode"] != 0
            or state["FinishedAt"] != _ZERO_DOCKER_TIMESTAMP
            or state["OOMKilled"] is not False
            or state["Running"] is not True
            or state["Status"] != "running"
            or type(state["Pid"]) is not int
            or state["Pid"] <= 0
            or not _is_utc_timestamp(state["StartedAt"])
            or state["StartedAt"] == _ZERO_DOCKER_TIMESTAMP
        ):
            raise DockerRunActionInspectionError(
                "Docker container is not one stable running process"
            )
    elif lifecycle is _DockerContainerLifecycle.EXITED_MAIN:
        if (
            state["Running"] is not False
            or state["Status"] != "exited"
            or type(state["Pid"]) is not int
            or state["Pid"] != 0
            or not _is_utc_timestamp(state["StartedAt"])
            or state["StartedAt"] == _ZERO_DOCKER_TIMESTAMP
            or not _is_utc_timestamp(state["FinishedAt"])
            or state["FinishedAt"] == _ZERO_DOCKER_TIMESTAMP
            or type(state["ExitCode"]) is not int
            or not 0 <= state["ExitCode"] <= 255
            or type(state["OOMKilled"]) is not bool
        ):
            raise DockerRunActionInspectionError(
                "Docker container is not one stable exited process"
            )
    elif (
        state["Running"] is not False
        or state["Status"] != "created"
        or type(state["Pid"]) is not int
        or state["Pid"] != 0
        or state["StartedAt"] != _ZERO_DOCKER_TIMESTAMP
        or state["FinishedAt"] != _ZERO_DOCKER_TIMESTAMP
        or state["ExitCode"] != 0
        or state["OOMKilled"] is not False
    ):
        raise DockerRunActionInspectionError(
            "Docker container is not inert and never-started"
        )


def _daemon_managed_paths(
    container_id: str,
    settings: DockerRuntimeSettings,
    *,
    has_started: bool,
) -> dict[str, str]:
    if not has_started:
        return {
            "HostnamePath": "",
            "HostsPath": "",
            "ResolvConfPath": "",
        }
    container_root = (
        PurePosixPath(settings.runtime_root_directory) / "containers" / container_id
    )
    return {
        "HostnamePath": (container_root / "hostname").as_posix(),
        "HostsPath": (container_root / "hosts").as_posix(),
        "ResolvConfPath": (container_root / "resolv.conf").as_posix(),
    }


def _require_command_policy(
    command: DockerRunActionCommand,
    claim: RunActionPreparationClaim,
) -> None:
    if (
        type(command) is not DockerRunActionCommand
        or command.command_template_id != claim.execution_policy.command_template_id
    ):
        raise DockerRunActionInspectionError(
            "Docker command differs from execution policy template"
        )


def _require_helper_evidence(
    helper_evidence: RunActionSupervisorHelperEvidence,
    claim: RunActionPreparationClaim,
) -> None:
    policy = claim.execution_policy
    if (
        type(helper_evidence) is not RunActionSupervisorHelperEvidence
        or helper_evidence.helper_authority_id
        != policy.supervisor_helper_executable_authority_id
        or helper_evidence.source_path != policy.supervisor_helper_source_path
        or helper_evidence.executable_digest
        != policy.supervisor_helper_executable_digest
    ):
        raise DockerRunActionInspectionError(
            "Docker supervisor helper differs from execution policy"
        )


def _require_init_source_evidence(
    init_source_evidence: RunActionDockerInitSourceEvidence,
    claim: RunActionPreparationClaim,
) -> None:
    policy = claim.execution_policy
    if (
        type(init_source_evidence) is not RunActionDockerInitSourceEvidence
        or init_source_evidence.init_authority_id
        != policy.docker_init_executable_authority_id
        or init_source_evidence.source_path != policy.docker_init_source_path
        or init_source_evidence.executable_digest
        != policy.docker_init_executable_digest
    ):
        raise DockerRunActionInspectionError(
            "Docker init executable differs from execution policy"
        )


def _require_volume_observation(
    volume: DockerRunActionVolumeObservation,
    authority: RunActionRuntimeVolumeAuthority,
    settings: DockerRuntimeSettings,
) -> None:
    expected_mountpoint = (
        PurePosixPath(settings.runtime_root_directory)
        / "volumes"
        / authority.volume_name
        / "_data"
    ).as_posix()
    if (
        type(volume) is not DockerRunActionVolumeObservation
        or volume.volume_authority_id != authority.runtime_volume_authority_id
        or volume.volume_name != authority.volume_name
        or volume.mountpoint != expected_mountpoint
        or volume.raw_field_schema_id != DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID
        or volume.unclassified_raw_field_count != 0
        or volume.nonauthoritative_raw_field_count
        != len(_VOLUME_NONAUTHORITATIVE_RAW_FIELDS)
    ):
        raise DockerRunActionInspectionError(
            "Docker volume observation differs from container authority"
        )


def _driver_option_mapping(options: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for assignment in options:
        if "=" not in assignment:
            raise DockerRunActionInspectionError(
                "Docker volume option lacks an assignment"
            )
        key, value = assignment.split("=", 1)
        if not key or key in mapping:
            raise DockerRunActionInspectionError("Docker volume option is ambiguous")
        mapping[key] = value
    return mapping


def _require_exact_fields(
    value: Mapping[str, Any],
    schema_name: str,
    object_name: str,
) -> None:
    expected = set(docker_run_action_raw_field_schema()[schema_name])
    if set(value) != expected:
        raise DockerRunActionInspectionError(
            f"{object_name} has an unknown or missing raw field"
        )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DockerRunActionInspectionError(f"{name} is not an object")
    return value


def _require_exact_value(value: Any, expected: Any, name: str) -> None:
    if not _exact_value_matches(value, expected):
        differing_fields = (
            tuple(
                sorted(
                    key
                    for key in set(value) | set(expected)
                    if not _exact_value_matches(
                        value.get(key),
                        expected.get(key),
                    )
                )
            )
            if isinstance(value, Mapping) and isinstance(expected, Mapping)
            else ()
        )
        suffix = (
            "" if not differing_fields else f" at fields {','.join(differing_fields)}"
        )
        raise DockerRunActionInspectionError(
            f"Docker container {name} differs from issued create authority{suffix}"
        )


def _exact_value_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(_exact_value_matches(value[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            type(value) is list
            and len(value) == len(expected)
            and all(
                _exact_value_matches(observed_item, expected_item)
                for observed_item, expected_item in zip(value, expected)
            )
        )
    return type(value) is type(expected) and value == expected


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    match = _UTC_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        return False
    year, month, day, hour, minute, second = (
        int(component) for component in match.groups()
    )
    leap_year = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    month_lengths = (
        31,
        29 if leap_year else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    return (
        year > 0
        and 1 <= month <= len(month_lengths)
        and 1 <= day <= month_lengths[month - 1]
        and 0 <= hour < 24
        and 0 <= minute < 60
        and 0 <= second < 60
    )


__all__ = [
    "DockerRunActionInertKeeperObservation",
    "DockerRunActionInspectionError",
    "DockerRunActionVolumeObservation",
    "RunActionBarrierRunningContainerObservation",
    "issued_keeper_projection",
    "issued_main_projection",
    "observe_allocation_keeper",
    "observe_allocation_inert_main_container",
    "observe_inert_keeper",
    "observe_inert_main_container",
    "observe_running_barrier_main_container",
    "observe_running_keeper",
    "observe_runtime_volume",
    "observe_pre_release_terminal_main_container",
    "observe_terminal_main_container",
    "reobserve_pre_release_terminal_main_container_for_cleanup",
    "reobserve_terminal_main_container_for_cleanup",
]
