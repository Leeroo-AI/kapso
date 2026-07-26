"""Pure evidence contracts for one post-start, pre-release workload barrier."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import ClassVar

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionCreateInspectProjection,
    RUN_ACTION_DOCKER_INIT_DESTINATION,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionActivatedFileObservation,
    RunActionActivatedRuntimeDirectoryObservation,
    RunActionActivatedWorkspaceObservation,
    RunActionActivationRevalidationReceipt,
    RunActionPreparedFileKind,
    RunActionPreparedMountAccess,
    RunActionPreparedRuntimeDirectoryKind,
    run_action_keeper_process_cgroup_path,
)

_DOCKER_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})T"
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:[.](?P<fraction>[0-9]{1,9}))?Z$"
)
_BOOT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-" r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_ZERO_DOCKER_TIMESTAMP = "0001-01-01T00:00:00Z"
_RUNNING_PROCESS_STATES = {"R", "S"}


class RunActionBarrierContractError(ValueError):
    """Resolved barrier evidence is malformed, incomplete, or spliced."""


class RunActionResolvedMountKind(str, Enum):
    """Purpose of one actual mount root in the running main container."""

    DOCKER_INIT = "docker_init"
    SUPERVISOR_HELPER = "supervisor_helper"
    CONTROL = "control"
    CREDENTIAL = "credential"
    INPUT = "input"
    RESULT = "result"
    TEMPORARY = "temporary"
    WORKSPACE = "workspace"


@dataclass(frozen=True)
class RunActionBarrierRunningContainerObservation(StrictContract):
    """Closed Docker facts for one running main-container occurrence."""

    barrier_running_container_observation_id: str
    container_id: str
    observed_inspect_projection: DockerRunActionCreateInspectProjection
    complete_inspection_digest: str
    container_status: str
    init_process_id: int
    restart_count: int
    started_at: str
    finished_at: str
    paused: bool
    restarting: bool
    dead: bool
    oom_killed: bool
    state_error: str

    CONTENT_NAMESPACE: ClassVar[str] = (
        "run-action-barrier-running-container-observation"
    )
    IDENTITY_FIELD: ClassVar[str] = "barrier_running_container_observation_id"

    def _validate(self) -> None:
        if (
            _DOCKER_CONTAINER_ID_PATTERN.fullmatch(self.container_id) is None
            or type(self.observed_inspect_projection)
            is not DockerRunActionCreateInspectProjection
            or _SHA256_DIGEST_PATTERN.fullmatch(self.complete_inspection_digest) is None
            or self.container_status != "running"
            or type(self.init_process_id) is not int
            or self.init_process_id <= 0
            or type(self.restart_count) is not int
            or self.restart_count != 0
            or not _is_docker_timestamp(self.started_at)
            or self.started_at == _ZERO_DOCKER_TIMESTAMP
            or self.finished_at != _ZERO_DOCKER_TIMESTAMP
            or self.paused is not False
            or self.restarting is not False
            or self.dead is not False
            or self.oom_killed is not False
            or self.state_error != ""
        ):
            raise RunActionBarrierContractError(
                "barrier running container observation is not one stable occurrence"
            )


@dataclass(frozen=True)
class RunActionBarrierInitProcessObservation(StrictContract):
    """Linux generation and namespace identity of Docker's pinned init."""

    barrier_init_process_observation_id: str
    provider_execution_id: str
    process_id: int
    parent_process_id: int
    process_start_time_ticks: int
    process_state: str
    process_cgroup_path: str
    mount_namespace_device: int
    mount_namespace_inode: int
    process_id_namespace_device: int
    process_id_namespace_inode: int
    command_line: tuple[str, ...]
    root_mount_info_observation_id: str
    root_mount_id: int
    root_device_major: int
    root_device_minor: int
    root_device: int
    root_inode: int
    executable_mount_id: int
    executable_device: int
    executable_inode: int
    executable_digest: str

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-barrier-init-process-observation"
    IDENTITY_FIELD: ClassVar[str] = "barrier_init_process_observation_id"

    def _validate(self) -> None:
        _require_process_identity(
            provider_execution_id=self.provider_execution_id,
            process_id=self.process_id,
            parent_process_id=self.parent_process_id,
            process_start_time_ticks=self.process_start_time_ticks,
            process_state=self.process_state,
            process_cgroup_path=self.process_cgroup_path,
            mount_namespace_device=self.mount_namespace_device,
            mount_namespace_inode=self.mount_namespace_inode,
            process_id_namespace_device=self.process_id_namespace_device,
            process_id_namespace_inode=self.process_id_namespace_inode,
            command_line=self.command_line,
            executable_mount_id=self.executable_mount_id,
            executable_device=self.executable_device,
            executable_inode=self.executable_inode,
            executable_digest=self.executable_digest,
            name="barrier init",
        )
        _require_namespaced_content_id(
            self.root_mount_info_observation_id,
            RunActionMountInfoObservation.CONTENT_NAMESPACE,
            "barrier init root mountinfo",
        )
        if (
            any(
                type(value) is not int or value <= 0
                for value in (self.root_mount_id, self.root_device, self.root_inode)
            )
            or any(
                type(value) is not int or value < 0
                for value in (self.root_device_major, self.root_device_minor)
            )
            or (
                os.major(self.root_device),
                os.minor(self.root_device),
            )
            != (
                self.root_device_major,
                self.root_device_minor,
            )
        ):
            raise RunActionBarrierContractError("barrier init root identity is invalid")


@dataclass(frozen=True)
class RunActionBarrierWrapperProcessObservation(StrictContract):
    """Exact direct BusyBox child blocked on the positional release protocol."""

    barrier_wrapper_process_observation_id: str
    provider_execution_id: str
    init_process_observation_id: str
    process_id: int
    parent_process_id: int
    process_start_time_ticks: int
    process_state: str
    process_cgroup_path: str
    mount_namespace_device: int
    mount_namespace_inode: int
    process_id_namespace_device: int
    process_id_namespace_inode: int
    command_line: tuple[str, ...]
    root_mount_info_observation_id: str
    root_mount_id: int
    root_device_major: int
    root_device_minor: int
    root_device: int
    root_inode: int
    executable_mount_id: int
    executable_device: int
    executable_inode: int
    executable_digest: str

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-barrier-wrapper-process-observation"
    IDENTITY_FIELD: ClassVar[str] = "barrier_wrapper_process_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.init_process_observation_id,
            RunActionBarrierInitProcessObservation.CONTENT_NAMESPACE,
            "barrier wrapper init process",
        )
        _require_process_identity(
            provider_execution_id=self.provider_execution_id,
            process_id=self.process_id,
            parent_process_id=self.parent_process_id,
            process_start_time_ticks=self.process_start_time_ticks,
            process_state=self.process_state,
            process_cgroup_path=self.process_cgroup_path,
            mount_namespace_device=self.mount_namespace_device,
            mount_namespace_inode=self.mount_namespace_inode,
            process_id_namespace_device=self.process_id_namespace_device,
            process_id_namespace_inode=self.process_id_namespace_inode,
            command_line=self.command_line,
            executable_mount_id=self.executable_mount_id,
            executable_device=self.executable_device,
            executable_inode=self.executable_inode,
            executable_digest=self.executable_digest,
            name="barrier wrapper",
        )
        _require_namespaced_content_id(
            self.root_mount_info_observation_id,
            RunActionMountInfoObservation.CONTENT_NAMESPACE,
            "barrier wrapper root mountinfo",
        )
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.root_mount_id,
                    self.root_device,
                    self.root_inode,
                )
            )
            or any(
                type(value) is not int or value < 0
                for value in (self.root_device_major, self.root_device_minor)
            )
            or (
                os.major(self.root_device),
                os.minor(self.root_device),
            )
            != (
                self.root_device_major,
                self.root_device_minor,
            )
        ):
            raise RunActionBarrierContractError(
                "barrier wrapper root identity is invalid"
            )


@dataclass(frozen=True)
class RunActionMountInfoObservation(StrictContract):
    """One normalized record from the main process mount namespace."""

    mount_info_observation_id: str
    mount_id: int
    parent_mount_id: int
    device_major: int
    device_minor: int
    mount_root: str
    mount_point: str
    mount_options: tuple[str, ...]
    optional_fields: tuple[str, ...]
    filesystem_type: str
    mount_source: str
    super_options: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-mount-info-observation"
    IDENTITY_FIELD: ClassVar[str] = "mount_info_observation_id"

    def _validate(self) -> None:
        _require_mount_path(self.mount_root, "mountinfo root")
        _require_mount_path(self.mount_point, "mountinfo point")
        _require_mount_tokens(self.mount_options, "mountinfo mount options")
        _require_mount_tokens(
            self.optional_fields,
            "mountinfo optional fields",
            allow_empty=True,
        )
        _require_mount_tokens(self.super_options, "mountinfo super options")
        if (
            any(
                type(value) is not int or value <= 0
                for value in (self.mount_id, self.parent_mount_id)
            )
            or any(
                type(value) is not int or value < 0
                for value in (self.device_major, self.device_minor)
            )
            or ("ro" in self.mount_options) == ("rw" in self.mount_options)
            or not _is_mount_text(self.filesystem_type)
            or any(character.isspace() for character in self.filesystem_type)
            or not isinstance(self.mount_source, str)
            or not self.mount_source
            or not self.mount_source.isascii()
            or "\x00" in self.mount_source
        ):
            raise RunActionBarrierContractError(
                "mountinfo observation is incomplete or noncanonical"
            )


@dataclass(frozen=True)
class RunActionMountInfoSnapshot(StrictContract):
    """Exact full-EOF mountinfo payload and its parsed canonical records."""

    mount_info_snapshot_id: str
    raw_payload: str
    raw_byte_length: int
    raw_payload_digest: str
    records: tuple[RunActionMountInfoObservation, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-mount-info-snapshot"
    IDENTITY_FIELD: ClassVar[str] = "mount_info_snapshot_id"

    def _validate(self) -> None:
        if (
            not isinstance(self.raw_payload, str)
            or not self.raw_payload
            or not self.raw_payload.isascii()
            or "\x00" in self.raw_payload
            or "\r" in self.raw_payload
            or type(self.raw_byte_length) is not int
            or self.raw_byte_length != len(self.raw_payload.encode("ascii"))
            or self.raw_payload_digest
            != tree_or_blob_digest(self.raw_payload.encode("ascii"))
            or type(self.records) is not tuple
            or any(
                type(record) is not RunActionMountInfoObservation
                for record in self.records
            )
            or self.records != parse_run_action_mount_info_payload(self.raw_payload)
        ):
            raise RunActionBarrierContractError(
                "mountinfo snapshot differs from its exact full-EOF payload"
            )

    @classmethod
    def from_raw_payload(cls, raw_payload: str) -> "RunActionMountInfoSnapshot":
        """Parse and content-address one exact full-EOF mountinfo payload."""
        if not isinstance(raw_payload, str) or not raw_payload.isascii():
            raise RunActionBarrierContractError(
                "mountinfo raw payload must be exact ASCII text"
            )
        encoded = raw_payload.encode("ascii")
        return cls.mint(
            raw_payload=raw_payload,
            raw_byte_length=len(encoded),
            raw_payload_digest=tree_or_blob_digest(encoded),
            records=parse_run_action_mount_info_payload(raw_payload),
        )


@dataclass(frozen=True)
class RunActionResolvedMountRootObservation(StrictContract):
    """One actual container mount root joined to its event-5 source inode."""

    resolved_mount_root_observation_id: str
    kind: RunActionResolvedMountKind
    source_authority_id: str
    container_destination: str
    container_access: RunActionPreparedMountAccess
    mount_info_observation_id: str
    source_mount_id: int
    source_device: int
    source_inode: int
    resolved_mount_id: int
    resolved_device: int
    resolved_inode: int
    mount_namespace_device: int
    mount_namespace_inode: int
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-resolved-mount-root-observation"
    IDENTITY_FIELD: ClassVar[str] = "resolved_mount_root_observation_id"

    def _validate(self) -> None:
        require_content_id(
            self.source_authority_id,
            "resolved mount event-5 source authority",
        )
        _require_absolute_container_path(
            self.container_destination,
            "resolved mount destination",
        )
        _require_namespaced_content_id(
            self.mount_info_observation_id,
            RunActionMountInfoObservation.CONTENT_NAMESPACE,
            "resolved mount mountinfo",
        )
        expected_file_type = (
            "regular"
            if self.kind
            in {
                RunActionResolvedMountKind.DOCKER_INIT,
                RunActionResolvedMountKind.SUPERVISOR_HELPER,
            }
            else "directory"
        )
        if (
            type(self.kind) is not RunActionResolvedMountKind
            or type(self.container_access) is not RunActionPreparedMountAccess
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.source_mount_id,
                    self.source_device,
                    self.source_inode,
                    self.resolved_mount_id,
                    self.resolved_device,
                    self.resolved_inode,
                    self.mount_namespace_device,
                    self.mount_namespace_inode,
                )
            )
            or self.source_device != self.resolved_device
            or self.source_inode != self.resolved_inode
            or self.file_type != expected_file_type
            or type(self.owner_user_id) is not int
            or self.owner_user_id < 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id < 0
            or type(self.mode) is not int
            or not 0 < self.mode <= 0o777
            or (
                self.kind
                in {
                    RunActionResolvedMountKind.DOCKER_INIT,
                    RunActionResolvedMountKind.SUPERVISOR_HELPER,
                }
                and self.container_access is not RunActionPreparedMountAccess.READ_ONLY
            )
        ):
            raise RunActionBarrierContractError(
                "resolved mount root is unsafe, substituted, or overlaid"
            )


@dataclass(frozen=True)
class RunActionResolvedFileObservation(StrictContract):
    """One logical file reopened through its actual container mount."""

    resolved_file_observation_id: str
    kind: RunActionPreparedFileKind
    activated_file_observation_id: str
    resolved_mount_root_observation_id: str
    container_path: str
    parent_entry_count: int
    mount_id: int
    device: int
    inode: int
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    content_digest: str | None
    content_authority_id: str | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-resolved-file-observation"
    IDENTITY_FIELD: ClassVar[str] = "resolved_file_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.activated_file_observation_id,
            RunActionActivatedFileObservation.CONTENT_NAMESPACE,
            "resolved activated file",
        )
        _require_namespaced_content_id(
            self.resolved_mount_root_observation_id,
            RunActionResolvedMountRootObservation.CONTENT_NAMESPACE,
            "resolved file mount root",
        )
        _require_absolute_container_path(
            self.container_path,
            "resolved file container path",
        )
        if self.content_authority_id is not None:
            require_identifier(
                self.content_authority_id,
                "resolved file content authority",
            )
        if (
            type(self.kind) is not RunActionPreparedFileKind
            or type(self.parent_entry_count) is not int
            or self.parent_entry_count != 1
            or any(
                type(value) is not int or value <= 0
                for value in (self.mount_id, self.device, self.inode)
            )
            or self.file_type != "regular"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode
            != (0o600 if self.kind is RunActionPreparedFileKind.RESULT else 0o400)
            or type(self.link_count) is not int
            or self.link_count != 1
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
            or (
                self.content_digest is not None
                and _SHA256_DIGEST_PATTERN.fullmatch(self.content_digest) is None
            )
            or (
                self.kind is RunActionPreparedFileKind.INPUT
                and (
                    self.size_bytes <= 0
                    or self.content_digest is None
                    or self.content_authority_id is None
                )
            )
            or (
                self.kind is RunActionPreparedFileKind.RESULT
                and (
                    self.size_bytes != 0
                    or self.content_digest is not None
                    or self.content_authority_id is not None
                )
            )
            or (
                self.kind is RunActionPreparedFileKind.CREDENTIAL
                and (
                    self.size_bytes <= 0
                    or self.content_digest is not None
                    or self.content_authority_id is None
                )
            )
        ):
            raise RunActionBarrierContractError(
                "resolved file observation is unsafe or incomplete"
            )


@dataclass(frozen=True)
class RunActionResolvedWorkspaceObservation(StrictContract):
    """Fresh workspace semantics read through the resolved workspace mount."""

    resolved_workspace_observation_id: str
    activated_workspace_observation_id: str
    resolved_mount_root_observation_id: str
    source_tree_digest: str
    git_closure_digest: str
    source_entry_count: int
    source_size_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-resolved-workspace-observation"
    IDENTITY_FIELD: ClassVar[str] = "resolved_workspace_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.activated_workspace_observation_id,
            RunActionActivatedWorkspaceObservation.CONTENT_NAMESPACE,
            "resolved activated workspace",
        )
        _require_namespaced_content_id(
            self.resolved_mount_root_observation_id,
            RunActionResolvedMountRootObservation.CONTENT_NAMESPACE,
            "resolved workspace mount root",
        )
        if (
            _SHA256_DIGEST_PATTERN.fullmatch(self.source_tree_digest) is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.git_closure_digest) is None
            or type(self.source_entry_count) is not int
            or self.source_entry_count < 0
            or type(self.source_size_bytes) is not int
            or self.source_size_bytes < 0
        ):
            raise RunActionBarrierContractError(
                "resolved workspace observation is invalid"
            )


@dataclass(frozen=True)
class RunActionResolvedWorkloadObservation(StrictContract):
    """Complete blocked proof carrying the receipt a caller joins to event 5.

    The durable execution-event identity is deliberately not asserted here. The
    store integration must first read the typed activation event and pass its
    exact receipt into this graph.
    """

    resolved_workload_observation_id: str
    activation_revalidation_receipt: RunActionActivationRevalidationReceipt
    host_boot_id: str
    running_container_observation: RunActionBarrierRunningContainerObservation
    init_process_observation: RunActionBarrierInitProcessObservation
    wrapper_process_observation: RunActionBarrierWrapperProcessObservation
    mount_info_snapshot: RunActionMountInfoSnapshot
    resolved_mount_root_observations: tuple[RunActionResolvedMountRootObservation, ...]
    resolved_file_observations: tuple[RunActionResolvedFileObservation, ...]
    resolved_workspace_observation: RunActionResolvedWorkspaceObservation | None
    control_entry_count: int
    temporary_entry_count: int
    control_directory_topology: RunActionControlDirectoryTopology

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-resolved-workload-observation"
    IDENTITY_FIELD: ClassVar[str] = "resolved_workload_observation_id"

    def _validate(self) -> None:
        if (
            type(self.activation_revalidation_receipt)
            is not RunActionActivationRevalidationReceipt
            or type(self.running_container_observation)
            is not RunActionBarrierRunningContainerObservation
            or type(self.init_process_observation)
            is not RunActionBarrierInitProcessObservation
            or type(self.wrapper_process_observation)
            is not RunActionBarrierWrapperProcessObservation
            or type(self.mount_info_snapshot) is not RunActionMountInfoSnapshot
            or type(self.resolved_mount_root_observations) is not tuple
            or any(
                type(observation) is not RunActionResolvedMountRootObservation
                for observation in self.resolved_mount_root_observations
            )
            or type(self.resolved_file_observations) is not tuple
            or any(
                type(observation) is not RunActionResolvedFileObservation
                for observation in self.resolved_file_observations
            )
            or (
                self.resolved_workspace_observation is not None
                and type(self.resolved_workspace_observation)
                is not RunActionResolvedWorkspaceObservation
            )
            or _BOOT_ID_PATTERN.fullmatch(self.host_boot_id) is None
            or type(self.control_entry_count) is not int
            or self.control_entry_count != 0
            or type(self.temporary_entry_count) is not int
            or self.temporary_entry_count != 0
            or self.control_directory_topology
            is not RunActionControlDirectoryTopology.EMPTY
            or not _resolved_workload_graph_matches(self)
        ):
            raise RunActionBarrierContractError(
                "resolved workload observation differs from its activation receipt graph"
            )


def _resolved_workload_graph_matches(
    observation: RunActionResolvedWorkloadObservation,
) -> bool:
    activation = observation.activation_revalidation_receipt
    prepared = activation.prepared_execution
    spawn = activation.spawn_commit
    projection = prepared.inert_container_evidence.issued_create_projection
    running = observation.running_container_observation
    init = observation.init_process_observation
    wrapper = observation.wrapper_process_observation
    expected_cgroup_path = run_action_keeper_process_cgroup_path(
        prepared.preparation_claim.execution_policy,
        spawn.provider_execution_id,
    )
    if (
        running.container_id != spawn.provider_execution_id
        or running.observed_inspect_projection != projection
        or init.provider_execution_id != spawn.provider_execution_id
        or init.process_id != running.init_process_id
        or init.command_line
        != (
            RUN_ACTION_DOCKER_INIT_DESTINATION,
            "--",
            projection.command_executable,
            *projection.command_arguments,
        )
        or init.process_cgroup_path != expected_cgroup_path
        or init.executable_device != projection.docker_init_source_evidence.device
        or init.executable_inode != projection.docker_init_source_evidence.inode
        or init.executable_digest
        != projection.docker_init_source_evidence.executable_digest
        or wrapper.provider_execution_id != spawn.provider_execution_id
        or wrapper.init_process_observation_id
        != init.barrier_init_process_observation_id
        or init.parent_process_id == wrapper.process_id
        or wrapper.parent_process_id != init.process_id
        or wrapper.process_id == init.process_id
        or wrapper.process_start_time_ticks < init.process_start_time_ticks
        or wrapper.process_cgroup_path != expected_cgroup_path
        or (
            wrapper.mount_namespace_device,
            wrapper.mount_namespace_inode,
        )
        != (init.mount_namespace_device, init.mount_namespace_inode)
        or (
            wrapper.process_id_namespace_device,
            wrapper.process_id_namespace_inode,
        )
        != (
            init.process_id_namespace_device,
            init.process_id_namespace_inode,
        )
        or wrapper.root_mount_info_observation_id != init.root_mount_info_observation_id
        or (
            wrapper.root_mount_id,
            wrapper.root_device_major,
            wrapper.root_device_minor,
            wrapper.root_device,
            wrapper.root_inode,
        )
        != (
            init.root_mount_id,
            init.root_device_major,
            init.root_device_minor,
            init.root_device,
            init.root_inode,
        )
        or wrapper.command_line
        != (projection.command_executable, *projection.command_arguments)
        or wrapper.executable_device != projection.supervisor_helper_evidence.device
        or wrapper.executable_inode != projection.supervisor_helper_evidence.inode
        or wrapper.executable_digest
        != projection.supervisor_helper_evidence.executable_digest
    ):
        return False
    mount_info = observation.mount_info_snapshot.records
    if tuple(record.mount_id for record in mount_info) != tuple(
        sorted({record.mount_id for record in mount_info})
    ) or len({record.mount_info_observation_id for record in mount_info}) != len(
        mount_info
    ):
        return False
    mount_info_by_id = {
        record.mount_info_observation_id: record for record in mount_info
    }
    container_roots = tuple(
        record for record in mount_info if record.mount_point == "/"
    )
    if (
        len(container_roots) != 1
        or init.root_mount_info_observation_id not in mount_info_by_id
        or mount_info_by_id[init.root_mount_info_observation_id] != container_roots[0]
        or (
            container_roots[0].mount_id,
            container_roots[0].device_major,
            container_roots[0].device_minor,
        )
        != (
            init.root_mount_id,
            init.root_device_major,
            init.root_device_minor,
        )
    ):
        return False
    roots = observation.resolved_mount_root_observations
    if (
        not roots
        or tuple(root.container_destination for root in roots)
        != tuple(sorted({root.container_destination for root in roots}))
        or len({root.kind for root in roots}) != len(roots)
        or len({root.resolved_mount_id for root in roots}) != len(roots)
        or any(
            (root.mount_namespace_device, root.mount_namespace_inode)
            != (init.mount_namespace_device, init.mount_namespace_inode)
            for root in roots
        )
        or any(
            not _resolved_root_matches_mount_info(root, mount_info_by_id, mount_info)
            for root in roots
        )
    ):
        return False
    roots_by_kind = {root.kind: root for root in roots}
    init_root = roots_by_kind.get(RunActionResolvedMountKind.DOCKER_INIT)
    init_source = projection.docker_init_source_evidence
    if (
        init_root is None
        or init_root.source_authority_id != init_source.docker_init_source_evidence_id
        or init_root.container_destination != RUN_ACTION_DOCKER_INIT_DESTINATION
        or init_root.container_access is not RunActionPreparedMountAccess.READ_ONLY
        or (
            init_root.source_mount_id,
            init_root.source_device,
            init_root.source_inode,
        )
        != (init_source.mount_id, init_source.device, init_source.inode)
        or init_root.owner_user_id != init_source.owner_user_id
        or init_root.owner_group_id != init_source.owner_group_id
        or init_root.mode != init_source.mode
        or (
            init_root.resolved_mount_id,
            init_root.resolved_device,
            init_root.resolved_inode,
        )
        != (
            init.executable_mount_id,
            init.executable_device,
            init.executable_inode,
        )
    ):
        return False
    helper_root = roots_by_kind.get(RunActionResolvedMountKind.SUPERVISOR_HELPER)
    helper = projection.supervisor_helper_evidence
    if (
        helper_root is None
        or helper_root.source_authority_id != helper.supervisor_helper_evidence_id
        or helper_root.container_destination != RUN_ACTION_SUPERVISOR_HELPER_DESTINATION
        or helper_root.container_access is not RunActionPreparedMountAccess.READ_ONLY
        or (
            helper_root.source_mount_id,
            helper_root.source_device,
            helper_root.source_inode,
        )
        != (helper.mount_id, helper.device, helper.inode)
        or helper_root.owner_user_id != helper.owner_user_id
        or helper_root.owner_group_id != helper.owner_group_id
        or helper_root.mode != helper.mode
        or (
            helper_root.resolved_mount_id,
            helper_root.resolved_device,
            helper_root.resolved_inode,
        )
        != (
            wrapper.executable_mount_id,
            wrapper.executable_device,
            wrapper.executable_inode,
        )
    ):
        return False
    expected_volume_roots = _expected_volume_mount_roots(activation)
    expected_root_kinds = {
        RunActionResolvedMountKind.DOCKER_INIT,
        RunActionResolvedMountKind.SUPERVISOR_HELPER,
        *(expected_volume_roots),
    }
    if set(roots_by_kind) != expected_root_kinds:
        return False
    for prepared_mount in projection.mounts:
        kind = RunActionResolvedMountKind(prepared_mount.kind.value)
        root = roots_by_kind[kind]
        expected = expected_volume_roots[kind]
        if (
            root.source_authority_id != expected["source_authority_id"]
            or root.container_destination != prepared_mount.container_destination
            or root.container_access is not prepared_mount.container_access
            or (
                root.source_mount_id,
                root.source_device,
                root.source_inode,
            )
            != (
                expected["source_mount_id"],
                expected["source_device"],
                expected["source_inode"],
            )
            or root.owner_user_id != expected["owner_user_id"]
            or root.owner_group_id != expected["owner_group_id"]
            or root.mode != expected["mode"]
        ):
            return False
    if not _resolved_files_match_activation(
        observation.resolved_file_observations,
        activation,
        roots_by_kind,
    ):
        return False
    activated_workspace = activation.activated_workspace_observation
    resolved_workspace = observation.resolved_workspace_observation
    if activated_workspace is None:
        return resolved_workspace is None
    workspace_root = roots_by_kind.get(RunActionResolvedMountKind.WORKSPACE)
    return (
        type(resolved_workspace) is RunActionResolvedWorkspaceObservation
        and workspace_root is not None
        and resolved_workspace.activated_workspace_observation_id
        == activated_workspace.activated_workspace_observation_id
        and resolved_workspace.resolved_mount_root_observation_id
        == workspace_root.resolved_mount_root_observation_id
        and resolved_workspace.source_tree_digest
        == activated_workspace.source_tree_digest
        and resolved_workspace.git_closure_digest
        == activated_workspace.git_closure_digest
        and resolved_workspace.source_entry_count
        == activated_workspace.source_entry_count
        and resolved_workspace.source_size_bytes
        == activated_workspace.source_size_bytes
        and (
            workspace_root.source_mount_id,
            workspace_root.source_device,
            workspace_root.source_inode,
        )
        == (
            activated_workspace.mount_id,
            activated_workspace.device,
            activated_workspace.inode,
        )
    )


def _expected_volume_mount_roots(
    activation: RunActionActivationRevalidationReceipt,
) -> dict[RunActionResolvedMountKind, dict[str, int | str]]:
    prepared = activation.prepared_execution
    runtime_directories = {
        observed.kind: observed
        for observed in activation.activated_runtime_directory_observations
    }
    expected: dict[RunActionResolvedMountKind, dict[str, int | str]] = {
        RunActionResolvedMountKind.INPUT: _directory_source(
            prepared.input_delivery_slot.prepared_delivery_slot_id,
            activation.input_file_observation.parent_mount_id,
            activation.input_file_observation.parent_device,
            activation.input_file_observation.parent_inode,
            prepared.input_delivery_slot.owner_user_id,
            prepared.input_delivery_slot.owner_group_id,
            prepared.input_delivery_slot.mode,
        ),
        RunActionResolvedMountKind.RESULT: _directory_source(
            prepared.result_directory.prepared_runtime_directory_id,
            activation.result_file_observation.parent_mount_id,
            activation.result_file_observation.parent_device,
            activation.result_file_observation.parent_inode,
            prepared.result_directory.owner_user_id,
            prepared.result_directory.owner_group_id,
            prepared.result_directory.mode,
        ),
        RunActionResolvedMountKind.CONTROL: _activated_directory_source(
            runtime_directories[RunActionPreparedRuntimeDirectoryKind.CONTROL]
        ),
        RunActionResolvedMountKind.TEMPORARY: _activated_directory_source(
            runtime_directories[RunActionPreparedRuntimeDirectoryKind.TEMPORARY]
        ),
    }
    if activation.credential_file_observation is not None:
        credential_slot = prepared.credential_delivery_slot
        credential = activation.credential_file_observation
        expected[RunActionResolvedMountKind.CREDENTIAL] = _directory_source(
            credential_slot.prepared_delivery_slot_id,
            credential.parent_mount_id,
            credential.parent_device,
            credential.parent_inode,
            credential_slot.owner_user_id,
            credential_slot.owner_group_id,
            credential_slot.mode,
        )
    if activation.activated_workspace_observation is not None:
        workspace = activation.activated_workspace_observation
        expected[RunActionResolvedMountKind.WORKSPACE] = _directory_source(
            workspace.prepared_workspace_proof_id,
            workspace.mount_id,
            workspace.device,
            workspace.inode,
            workspace.owner_user_id,
            workspace.owner_group_id,
            workspace.root_mode,
        )
    return expected


def _activated_directory_source(
    observed: RunActionActivatedRuntimeDirectoryObservation,
) -> dict[str, int | str]:
    return _directory_source(
        observed.prepared_runtime_directory_id,
        observed.mount_id,
        observed.device,
        observed.inode,
        observed.owner_user_id,
        observed.owner_group_id,
        observed.mode,
    )


def _directory_source(
    source_authority_id: str,
    source_mount_id: int,
    source_device: int,
    source_inode: int,
    owner_user_id: int,
    owner_group_id: int,
    mode: int,
) -> dict[str, int | str]:
    return {
        "source_authority_id": source_authority_id,
        "source_mount_id": source_mount_id,
        "source_device": source_device,
        "source_inode": source_inode,
        "owner_user_id": owner_user_id,
        "owner_group_id": owner_group_id,
        "mode": mode,
    }


def _resolved_files_match_activation(
    resolved_files: tuple[RunActionResolvedFileObservation, ...],
    activation: RunActionActivationRevalidationReceipt,
    roots_by_kind: dict[
        RunActionResolvedMountKind,
        RunActionResolvedMountRootObservation,
    ],
) -> bool:
    expected_files = tuple(
        observed
        for observed in (
            activation.input_file_observation,
            activation.result_file_observation,
            activation.credential_file_observation,
        )
        if observed is not None
    )
    if tuple(file.kind.value for file in resolved_files) != tuple(
        sorted({file.kind.value for file in resolved_files})
    ) or {file.kind for file in resolved_files} != {
        file.kind for file in expected_files
    }:
        return False
    resolved_by_kind = {file.kind: file for file in resolved_files}
    for activated in expected_files:
        resolved = resolved_by_kind[activated.kind]
        mount_kind = RunActionResolvedMountKind(activated.kind.value)
        root = roots_by_kind[mount_kind]
        expected_path = (
            PurePosixPath(root.container_destination)
            / PurePosixPath(activated.relative_path).name
        ).as_posix()
        if (
            resolved.activated_file_observation_id
            != activated.activated_file_observation_id
            or resolved.resolved_mount_root_observation_id
            != root.resolved_mount_root_observation_id
            or resolved.container_path != expected_path
            or resolved.mount_id != root.resolved_mount_id
            or resolved.device != root.resolved_device
            or resolved.device != activated.device
            or resolved.inode != activated.inode
            or resolved.file_type != activated.file_type
            or resolved.owner_user_id != activated.owner_user_id
            or resolved.owner_group_id != activated.owner_group_id
            or resolved.mode != activated.mode
            or resolved.link_count != activated.link_count
            or resolved.size_bytes != activated.size_bytes
            or resolved.content_digest != activated.content_digest
            or resolved.content_authority_id != activated.content_authority_id
        ):
            return False
    return True


def parse_run_action_mount_info_payload(
    raw_payload: str,
) -> tuple[RunActionMountInfoObservation, ...]:
    """Parse one exact full-EOF Linux mountinfo payload."""
    if (
        not isinstance(raw_payload, str)
        or not raw_payload
        or not raw_payload.isascii()
        or not raw_payload.endswith("\n")
        or "\x00" in raw_payload
        or "\r" in raw_payload
    ):
        raise RunActionBarrierContractError(
            "mountinfo raw payload is malformed or incomplete"
        )
    lines = raw_payload.split("\n")
    if lines[-1] != "" or any(not line for line in lines[:-1]):
        raise RunActionBarrierContractError(
            "mountinfo raw payload has an invalid line boundary"
        )
    records = tuple(_parse_mount_info_line(line) for line in lines[:-1])
    if len({record.mount_id for record in records}) != len(records) or len(
        {record.mount_info_observation_id for record in records}
    ) != len(records):
        raise RunActionBarrierContractError(
            "mountinfo raw payload contains duplicate records"
        )
    return tuple(sorted(records, key=lambda record: record.mount_id))


def _parse_mount_info_line(line: str) -> RunActionMountInfoObservation:
    fields = line.split(" ")
    separators = tuple(
        position for position, field in enumerate(fields) if field == "-"
    )
    if (
        any(not field for field in fields)
        or len(separators) != 1
        or separators[0] < 6
        or len(fields) - separators[0] - 1 != 3
    ):
        raise RunActionBarrierContractError("mountinfo line shape is malformed")
    separator = separators[0]
    device_fields = fields[2].split(":")
    if len(device_fields) != 2:
        raise RunActionBarrierContractError("mountinfo device field is malformed")
    return RunActionMountInfoObservation.mint(
        mount_id=_parse_mount_info_integer(
            fields[0],
            "mountinfo mount ID",
            positive=True,
        ),
        parent_mount_id=_parse_mount_info_integer(
            fields[1],
            "mountinfo parent mount ID",
            positive=True,
        ),
        device_major=_parse_mount_info_integer(
            device_fields[0],
            "mountinfo device major",
            positive=False,
        ),
        device_minor=_parse_mount_info_integer(
            device_fields[1],
            "mountinfo device minor",
            positive=False,
        ),
        mount_root=_decode_mount_info_field(fields[3], "mountinfo root"),
        mount_point=_decode_mount_info_field(fields[4], "mountinfo point"),
        mount_options=_parse_mount_info_options(
            fields[5],
            "mountinfo mount options",
        ),
        optional_fields=_parse_mount_info_tokens(
            tuple(fields[6:separator]),
            "mountinfo optional fields",
            allow_empty=True,
        ),
        filesystem_type=_decode_mount_info_field(
            fields[separator + 1],
            "mountinfo filesystem type",
        ),
        mount_source=_decode_mount_info_field(
            fields[separator + 2],
            "mountinfo mount source",
        ),
        super_options=_parse_mount_info_options(
            fields[separator + 3],
            "mountinfo super options",
        ),
    )


def _parse_mount_info_integer(
    value: str,
    name: str,
    *,
    positive: bool,
) -> int:
    if not value.isdigit():
        raise RunActionBarrierContractError(f"{name} is malformed")
    parsed = int(value)
    if str(parsed) != value or (parsed <= 0 if positive else parsed < 0):
        raise RunActionBarrierContractError(f"{name} is noncanonical")
    return parsed


def _parse_mount_info_options(value: str, name: str) -> tuple[str, ...]:
    return _parse_mount_info_tokens(tuple(value.split(",")), name)


def _parse_mount_info_tokens(
    values: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if (not allow_empty and not values) or any(not value for value in values):
        raise RunActionBarrierContractError(f"{name} are malformed")
    decoded = tuple(_decode_mount_info_field(value, name) for value in values)
    if len(set(decoded)) != len(decoded):
        raise RunActionBarrierContractError(f"{name} contain duplicates")
    return tuple(sorted(decoded))


def _decode_mount_info_field(value: str, name: str) -> str:
    decoded: list[str] = []
    position = 0
    escapes = {
        "011": "\t",
        "012": "\n",
        "040": " ",
        "134": "\\",
    }
    while position < len(value):
        if value[position] != "\\":
            decoded.append(value[position])
            position += 1
            continue
        escape = value[position + 1 : position + 4]
        if len(escape) != 3 or escape not in escapes:
            raise RunActionBarrierContractError(
                f"{name} contains an invalid mountinfo escape"
            )
        decoded.append(escapes[escape])
        position += 4
    result = "".join(decoded)
    if not result or "\x00" in result:
        raise RunActionBarrierContractError(f"{name} decodes to an invalid value")
    return result


def _resolved_root_matches_mount_info(
    root: RunActionResolvedMountRootObservation,
    mount_info_by_id: dict[str, RunActionMountInfoObservation],
    mount_info: tuple[RunActionMountInfoObservation, ...],
) -> bool:
    selected = mount_info_by_id.get(root.mount_info_observation_id)
    expected_access_option = (
        "ro"
        if root.container_access is RunActionPreparedMountAccess.READ_ONLY
        else "rw"
    )
    destination = PurePosixPath(root.container_destination)
    return (
        selected is not None
        and selected.mount_id == root.resolved_mount_id
        and selected.mount_point == root.container_destination
        and selected.device_major == os.major(root.resolved_device)
        and selected.device_minor == os.minor(root.resolved_device)
        and expected_access_option in selected.mount_options
        and not any(
            record.mount_info_observation_id != root.mount_info_observation_id
            and (
                PurePosixPath(record.mount_point) == destination
                or destination in PurePosixPath(record.mount_point).parents
            )
            for record in mount_info
        )
    )


def _require_process_identity(
    *,
    provider_execution_id: str,
    process_id: int,
    parent_process_id: int,
    process_start_time_ticks: int,
    process_state: str,
    process_cgroup_path: str,
    mount_namespace_device: int,
    mount_namespace_inode: int,
    process_id_namespace_device: int,
    process_id_namespace_inode: int,
    command_line: tuple[str, ...],
    executable_mount_id: int,
    executable_device: int,
    executable_inode: int,
    executable_digest: str,
    name: str,
) -> None:
    if (
        _DOCKER_CONTAINER_ID_PATTERN.fullmatch(provider_execution_id) is None
        or type(process_id) is not int
        or process_id <= 0
        or type(parent_process_id) is not int
        or parent_process_id <= 0
        or parent_process_id == process_id
        or type(process_start_time_ticks) is not int
        or process_start_time_ticks <= 0
        or process_state not in _RUNNING_PROCESS_STATES
        or not isinstance(process_cgroup_path, str)
        or not process_cgroup_path.isascii()
        or "\x00" in process_cgroup_path
        or not process_cgroup_path.startswith("/")
        or PurePosixPath(process_cgroup_path).as_posix() != process_cgroup_path
        or ".." in PurePosixPath(process_cgroup_path).parts
        or not process_cgroup_path.endswith(f"/docker-{provider_execution_id}.scope")
        or any(
            type(value) is not int or value <= 0
            for value in (
                mount_namespace_device,
                mount_namespace_inode,
                process_id_namespace_device,
                process_id_namespace_inode,
                executable_mount_id,
                executable_device,
                executable_inode,
            )
        )
        or type(command_line) is not tuple
        or not command_line
        or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in command_line
        )
        or _SHA256_DIGEST_PATTERN.fullmatch(executable_digest) is None
    ):
        raise RunActionBarrierContractError(f"{name} process identity is invalid")
    _require_absolute_container_path(
        command_line[0],
        f"{name} executable argument",
    )


def _is_docker_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _DOCKER_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        return False
    parts = {
        name: int(match.group(name))
        for name in ("year", "month", "day", "hour", "minute", "second")
    }
    leap_year = parts["year"] % 4 == 0 and (
        parts["year"] % 100 != 0 or parts["year"] % 400 == 0
    )
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
        parts["year"] > 0
        and 1 <= parts["month"] <= 12
        and 1 <= parts["day"] <= month_lengths[parts["month"] - 1]
        and 0 <= parts["hour"] <= 23
        and 0 <= parts["minute"] <= 59
        and 0 <= parts["second"] <= 59
    )


def _is_mount_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and not any(character in value for character in ("\x00", "\n", "\r"))
    )


def _require_mount_tokens(
    values: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if (
        type(values) is not tuple
        or (not allow_empty and not values)
        or values != tuple(sorted(set(values)))
        or any(
            not _is_mount_text(value)
            or any(character.isspace() or character == "," for character in value)
            for value in values
        )
    ):
        raise RunActionBarrierContractError(f"{name} are invalid")


def _require_mount_path(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or not path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or "\x00" in value
    ):
        raise RunActionBarrierContractError(f"{name} is invalid")


def _require_namespaced_content_id(
    value: str,
    namespace: str,
    name: str,
) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionBarrierContractError(f"{name} uses another content namespace")


def _require_absolute_container_path(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or not path.is_absolute()
        or path == PurePosixPath("/")
        or path.as_posix() != value
        or ".." in path.parts
        or "\x00" in value
    ):
        raise RunActionBarrierContractError(f"{name} is invalid")


__all__ = [
    "RunActionBarrierContractError",
    "RunActionBarrierInitProcessObservation",
    "RunActionBarrierRunningContainerObservation",
    "RunActionBarrierWrapperProcessObservation",
    "RunActionMountInfoObservation",
    "RunActionMountInfoSnapshot",
    "RunActionResolvedFileObservation",
    "RunActionResolvedMountKind",
    "RunActionResolvedMountRootObservation",
    "RunActionResolvedWorkloadObservation",
    "RunActionResolvedWorkspaceObservation",
    "parse_run_action_mount_info_payload",
]
