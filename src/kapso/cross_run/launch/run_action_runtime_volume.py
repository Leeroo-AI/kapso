"""Descriptor-bound preparation and observation of one runtime-volume generation."""

from __future__ import annotations

import ctypes
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionVolumeObservation,
)
from kapso.cross_run.launch.run_action_keeper_helper import (
    read_run_action_descriptor_mount_id,
    read_run_action_process_cgroup_path_from_descriptor,
    read_run_action_process_start_time_from_descriptor,
)
from kapso.cross_run.launch.run_action_contracts import RunFrontierWorkspaceAccess
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    RunActionCredentialMode,
    RunActionPreparationClaim,
    RunActionPreparedExecution,
    RunActionPreparedFile,
    RunActionPreparedFileKind,
    RunActionPreparedWorkspaceProof,
    RunActionRuntimeVolumeAuthority,
    RunActionRuntimeVolumeEvidence,
    RunActionRuntimeVolumeLayoutProof,
    RunActionRuntimeVolumeSentinelEvidence,
    RunActionVolumeKeeperEvidence,
    issue_runtime_volume_authority,
    run_action_keeper_process_cgroup_path,
)
from kapso.cross_run.launch.workspace_frontier import (
    RunWorkspaceCopyPlan,
    RunWorkspaceFrontierIdentity,
    copy_run_workspace_frontier,
    inspect_run_workspace_frontier,
    plan_run_workspace_frontier_copy,
)
from kapso.cross_run.settings import LaunchSettings

_TMPFS_FILESYSTEM_TYPE = "tmpfs"
_MOUNT_OPTIONS = ("nodev", "nosuid", "relatime", "rw")
_SUPER_OPTION_FLAGS = ("inode64", "noswap", "rw")
_SUPER_OPTION_KEYS = ("gid", "mode", "nr_inodes", "size", "uid")
_OPTIONAL_MOUNT_FIELD_PATTERN = re.compile(r"^master:[1-9][0-9]*$")
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SIZE_OPTION_PATTERN = re.compile(r"^([1-9][0-9]*)([kmgt]?)$")
_RENAME_NOREPLACE = 1
_PREPARATION_STAGING_PREFIX = ".kapso-prepare-"
_PENDING_SENTINEL_PREFIX = ".kapso-generation.pending-"
_SENTINEL_NAME = ".kapso-generation"
_PREPARED_DIRECTORY_MODE = 0o700
_PREPARED_FILE_MODE = 0o600
_SENTINEL_MODE = 0o400
_SIZE_MULTIPLIERS = {
    "": 1,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
    "t": 1024**4,
}


class RunActionRuntimeVolumeError(RuntimeError):
    """The mounted runtime volume differs from its issued tmpfs authority."""


@dataclass(frozen=True)
class DockerRunActionPreparedVolumeObservation:
    """Complete prepared layout proven through one keeper process generation."""

    preparation_claim: RunActionPreparationClaim
    runtime_volume_evidence: RunActionRuntimeVolumeEvidence
    input_file: RunActionPreparedFile
    result_file: RunActionPreparedFile
    credential_file: RunActionPreparedFile | None
    workspace_proof: RunActionPreparedWorkspaceProof | None
    layout_proof: RunActionRuntimeVolumeLayoutProof

    def __post_init__(self) -> None:
        authority = (
            self.runtime_volume_evidence.volume_authority
            if type(self.runtime_volume_evidence) is RunActionRuntimeVolumeEvidence
            else None
        )
        prepared_files = tuple(
            prepared_file
            for prepared_file in (
                self.input_file,
                self.result_file,
                self.credential_file,
            )
            if prepared_file is not None
        )
        expected_file_plans = (
            _expected_file_plans(self.preparation_claim)
            if type(self.preparation_claim) is RunActionPreparationClaim
            else ()
        )
        expected_authority = (
            issue_runtime_volume_authority(
                self.preparation_claim,
                authority.generation_nonce,
            )
            if type(self.preparation_claim) is RunActionPreparationClaim
            and type(authority) is RunActionRuntimeVolumeAuthority
            else None
        )
        workspace_binding = (
            self.preparation_claim.reservation.frontier.workspace_before
            if type(self.preparation_claim) is RunActionPreparationClaim
            else None
        )
        expected_directory_names = (
            _expected_directory_names(self.preparation_claim)
            if type(self.preparation_claim) is RunActionPreparationClaim
            else ()
        )
        workspace_size_bytes = (
            0 if workspace_binding is None else workspace_binding.source_size_bytes
        )
        workspace_entry_count = (
            0 if workspace_binding is None else workspace_binding.source_entry_count
        )
        if (
            type(self.preparation_claim) is not RunActionPreparationClaim
            or type(self.runtime_volume_evidence) is not RunActionRuntimeVolumeEvidence
            or type(self.input_file) is not RunActionPreparedFile
            or type(self.result_file) is not RunActionPreparedFile
            or (
                self.credential_file is not None
                and type(self.credential_file) is not RunActionPreparedFile
            )
            or (
                self.workspace_proof is not None
                and type(self.workspace_proof) is not RunActionPreparedWorkspaceProof
            )
            or type(self.layout_proof) is not RunActionRuntimeVolumeLayoutProof
            or authority != expected_authority
            or len(prepared_files) != len(expected_file_plans)
            or any(
                (
                    prepared_file.preparation_claim_id
                    != self.preparation_claim.preparation_claim_id
                    or prepared_file.runtime_volume_authority_id
                    != authority.runtime_volume_authority_id
                    or prepared_file.generation_nonce != authority.generation_nonce
                    or prepared_file.kind is not expected_file_plan.kind
                    or prepared_file.relative_path != expected_file_plan.relative_path
                    or prepared_file.owner_user_id != authority.owner_user_id
                    or prepared_file.owner_group_id != authority.owner_group_id
                    or prepared_file.payload_size_limit_bytes
                    != expected_file_plan.payload_size_limit_bytes
                )
                for prepared_file, expected_file_plan in zip(
                    prepared_files,
                    expected_file_plans,
                    strict=True,
                )
            )
            or self.layout_proof.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or self.layout_proof.runtime_volume_evidence_id
            != self.runtime_volume_evidence.runtime_volume_evidence_id
            or self.layout_proof.generation_nonce != authority.generation_nonce
            or self.layout_proof.directory_relative_paths != expected_directory_names
            or self.layout_proof.prepared_file_ids
            != tuple(
                sorted(
                    prepared_file.prepared_file_id for prepared_file in prepared_files
                )
            )
            or self.layout_proof.prepared_workspace_proof_id
            != (
                None
                if self.workspace_proof is None
                else self.workspace_proof.prepared_workspace_proof_id
            )
            or self.layout_proof.logical_content_size_bytes
            != len(authority.generation_nonce) + workspace_size_bytes
            or self.layout_proof.logical_entry_count
            != (
                len(expected_directory_names)
                + len(prepared_files)
                + 1
                + workspace_entry_count
            )
            or self.layout_proof.observed_used_size_bytes
            != self.runtime_volume_evidence.used_size_bytes
            or self.layout_proof.observed_used_inode_count
            != self.runtime_volume_evidence.used_inode_count
            or (
                self.workspace_proof is not None
                and (
                    self.workspace_proof.preparation_claim_id
                    != self.preparation_claim.preparation_claim_id
                    or self.workspace_proof.runtime_volume_authority_id
                    != authority.runtime_volume_authority_id
                    or self.workspace_proof.generation_nonce
                    != authority.generation_nonce
                    or self.workspace_proof.workspace_binding != workspace_binding
                    or self.workspace_proof.owner_user_id != authority.owner_user_id
                    or self.workspace_proof.owner_group_id != authority.owner_group_id
                    or self.workspace_proof.root_mode != authority.root_mode
                )
            )
            or (self.workspace_proof is None) != (workspace_binding is None)
        ):
            raise RunActionRuntimeVolumeError(
                "prepared runtime volume observation is incomplete"
            )


@dataclass(frozen=True)
class DockerRunActionEmptyVolumeObservation:
    """Physical proof that the running keeper mounts one empty bounded tmpfs."""

    runtime_volume_authority: RunActionRuntimeVolumeAuthority
    docker_volume_observation: DockerRunActionVolumeObservation
    keeper_container_id: str
    keeper_process_id: int
    keeper_process_start_time_ticks: int
    process_cgroup_path: str
    mount_id: int
    device: int
    root_inode: int
    filesystem_type: str
    observed_mount_flags: tuple[str, ...]
    owner_user_id: int
    owner_group_id: int
    root_mode: int
    allocation_block_size_bytes: int
    effective_block_count: int
    effective_size_bytes: int
    effective_inode_limit: int
    used_block_count: int
    used_size_bytes: int
    used_inode_count: int
    available_block_count: int
    available_size_bytes: int
    available_inode_count: int
    empty_entry_count: int
    empty_size_bytes: int

    def __post_init__(self) -> None:
        cgroup_path = (
            PurePosixPath(self.process_cgroup_path)
            if type(self.process_cgroup_path) is str
            else None
        )
        if (
            type(self.runtime_volume_authority) is not RunActionRuntimeVolumeAuthority
            or type(self.docker_volume_observation)
            is not DockerRunActionVolumeObservation
            or self.docker_volume_observation.volume_authority_id
            != self.runtime_volume_authority.runtime_volume_authority_id
            or self.docker_volume_observation.volume_name
            != self.runtime_volume_authority.volume_name
            or type(self.keeper_container_id) is not str
            or _CONTAINER_ID_PATTERN.fullmatch(self.keeper_container_id) is None
            or type(self.keeper_process_id) is not int
            or self.keeper_process_id <= 0
            or type(self.keeper_process_start_time_ticks) is not int
            or self.keeper_process_start_time_ticks <= 0
            or cgroup_path is None
            or not self.process_cgroup_path.isascii()
            or "\x00" in self.process_cgroup_path
            or not cgroup_path.is_absolute()
            or cgroup_path.as_posix() != self.process_cgroup_path
            or ".." in cgroup_path.parts
            or not self.process_cgroup_path.endswith(
                f"/docker-{self.keeper_container_id}.scope"
            )
            or any(
                type(value) is not int or value <= 0
                for value in (self.mount_id, self.device, self.root_inode)
            )
            or self.filesystem_type != _TMPFS_FILESYSTEM_TYPE
            or self.observed_mount_flags != ("nodev", "nosuid", "noswap")
            or self.owner_user_id != self.runtime_volume_authority.owner_user_id
            or self.owner_group_id != self.runtime_volume_authority.owner_group_id
            or self.root_mode != self.runtime_volume_authority.root_mode
            or any(
                type(value) is not int
                for value in (
                    self.allocation_block_size_bytes,
                    self.effective_block_count,
                    self.effective_size_bytes,
                    self.effective_inode_limit,
                    self.used_block_count,
                    self.used_size_bytes,
                    self.used_inode_count,
                    self.available_block_count,
                    self.available_size_bytes,
                    self.available_inode_count,
                    self.empty_entry_count,
                    self.empty_size_bytes,
                )
            )
            or self.allocation_block_size_bytes <= 0
            or self.allocation_block_size_bytes & (self.allocation_block_size_bytes - 1)
            != 0
            or type(self.effective_block_count) is not int
            or self.effective_block_count <= 0
            or self.effective_size_bytes
            != self.effective_block_count * self.allocation_block_size_bytes
            or not 0
            < self.effective_size_bytes
            <= self.runtime_volume_authority.size_limit_bytes
            or not 0
            < self.effective_inode_limit
            <= self.runtime_volume_authority.inode_limit
            or self.used_block_count < 0
            or self.used_size_bytes < 0
            or self.used_inode_count < 0
            or self.used_block_count + self.available_block_count
            != self.effective_block_count
            or self.used_size_bytes
            != self.used_block_count * self.allocation_block_size_bytes
            or self.available_size_bytes
            != self.available_block_count * self.allocation_block_size_bytes
            or self.used_size_bytes + self.available_size_bytes
            != self.effective_size_bytes
            or self.used_inode_count + self.available_inode_count
            != self.effective_inode_limit
            or self.available_block_count <= 0
            or self.available_inode_count <= 0
            or self.empty_entry_count != 0
            or self.empty_size_bytes != 0
        ):
            raise RunActionRuntimeVolumeError(
                "empty runtime volume observation is incomplete or unsafe"
            )


@dataclass(frozen=True)
class _MountedRuntimeVolumeLease:
    process_descriptor: int
    root_descriptor: int
    keeper_container_id: str
    keeper_process_id: int
    process_start_time_ticks: int
    process_cgroup_path: str
    root_mount_id: int
    root_device: int
    root_inode: int


@dataclass(frozen=True)
class _PreparedLayoutObservation:
    root_mount_id: int
    root_device: int
    root_inode: int
    sentinel_metadata: os.stat_result
    sentinel_mount_id: int
    workspace_frontier: RunWorkspaceFrontierIdentity | None
    filesystem: os.statvfs_result


@dataclass(frozen=True)
class _ExactRegularFileObservation:
    descriptor: int
    parent_descriptor: int
    name: str
    metadata: os.stat_result
    mount_id: int
    payload: bytes


@dataclass(frozen=True)
class _PreparedFilePlan:
    kind: RunActionPreparedFileKind
    directory_name: str
    file_name: str
    relative_path: str
    payload_size_limit_bytes: int


@dataclass(frozen=True)
class _RuntimeVolumeLayoutPlan:
    directory_names: tuple[str, ...]
    file_plans: tuple[_PreparedFilePlan, ...]
    workspace_copy_plan: RunWorkspaceCopyPlan | None
    preparation_size_bytes: int
    preparation_inode_count: int


@dataclass(frozen=True)
class _MountInfo:
    mount_id: int
    parent_mount_id: int
    device_major: int
    device_minor: int
    mount_point: str
    mount_options: tuple[str, ...]
    optional_fields: tuple[str, ...]
    filesystem_type: str
    source: str
    super_options: tuple[str, ...]


def issue_fresh_runtime_volume_authority(
    claim: RunActionPreparationClaim,
) -> RunActionRuntimeVolumeAuthority:
    """Issue one unpredictable generation beneath a deterministic claim."""

    if type(claim) is not RunActionPreparationClaim:
        raise RunActionRuntimeVolumeError(
            "fresh runtime volume authority requires an exact preparation claim"
        )
    return issue_runtime_volume_authority(claim, secrets.token_hex(16))


def _open_mounted_runtime_volume(
    descriptors: ExitStack,
    keeper: RunActionVolumeKeeperEvidence,
) -> _MountedRuntimeVolumeLease:
    if (
        type(descriptors) is not ExitStack
        or type(keeper) is not RunActionVolumeKeeperEvidence
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume lease requires an exact keeper"
        )
    process_descriptor = os.open(
        f"/proc/{keeper.process_id}",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, process_descriptor)
    process_start_time_ticks = read_run_action_process_start_time_from_descriptor(
        process_descriptor,
        keeper.process_id,
    )
    process_cgroup_path = read_run_action_process_cgroup_path_from_descriptor(
        process_descriptor,
        keeper.container_id,
    )
    process_root_descriptor = os.open(
        "root",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    descriptors.callback(os.close, process_root_descriptor)
    root_descriptor = os.open(
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION.removeprefix("/"),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_root_descriptor,
    )
    descriptors.callback(os.close, root_descriptor)
    root_metadata = os.fstat(root_descriptor)
    lease = _MountedRuntimeVolumeLease(
        process_descriptor=process_descriptor,
        root_descriptor=root_descriptor,
        keeper_container_id=keeper.container_id,
        keeper_process_id=keeper.process_id,
        process_start_time_ticks=process_start_time_ticks,
        process_cgroup_path=process_cgroup_path,
        root_mount_id=read_run_action_descriptor_mount_id(root_descriptor),
        root_device=root_metadata.st_dev,
        root_inode=root_metadata.st_ino,
    )
    _require_same_mounted_runtime_volume(lease, keeper)
    return lease


def _require_same_mounted_runtime_volume(
    lease: _MountedRuntimeVolumeLease,
    keeper: RunActionVolumeKeeperEvidence,
) -> None:
    if (
        type(lease) is not _MountedRuntimeVolumeLease
        or type(keeper) is not RunActionVolumeKeeperEvidence
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume lease changed process or physical root"
        )
    root_metadata = os.fstat(lease.root_descriptor)
    if (
        keeper.container_id != lease.keeper_container_id
        or keeper.process_id != lease.keeper_process_id
        or keeper.process_start_time_ticks != lease.process_start_time_ticks
        or read_run_action_process_start_time_from_descriptor(
            lease.process_descriptor,
            lease.keeper_process_id,
        )
        != lease.process_start_time_ticks
        or read_run_action_process_cgroup_path_from_descriptor(
            lease.process_descriptor,
            lease.keeper_container_id,
        )
        != lease.process_cgroup_path
        or lease.process_cgroup_path
        != run_action_keeper_process_cgroup_path(
            keeper.issued_create_projection.execution_policy,
            keeper.container_id,
        )
        or read_run_action_descriptor_mount_id(lease.root_descriptor)
        != lease.root_mount_id
        or root_metadata.st_dev != lease.root_device
        or root_metadata.st_ino != lease.root_inode
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume lease changed process or physical root"
        )


def observe_empty_runtime_volume(
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
) -> DockerRunActionEmptyVolumeObservation:
    """Prove the keeper's mounted generation is empty before first publication."""

    if (
        type(authority) is not RunActionRuntimeVolumeAuthority
        or type(volume) is not DockerRunActionVolumeObservation
        or type(keeper) is not RunActionVolumeKeeperEvidence
        or volume.volume_authority_id != authority.runtime_volume_authority_id
        or keeper.issued_create_projection.volume_authority != authority
    ):
        raise RunActionRuntimeVolumeError(
            "empty runtime volume observation requires exact Docker authority"
        )
    with ExitStack() as descriptors:
        lease = _open_mounted_runtime_volume(descriptors, keeper)
        metadata_before = os.fstat(lease.root_descriptor)
        mount_info_before = _read_mount_info(
            lease.process_descriptor,
            lease.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        filesystem_before = os.fstatvfs(lease.root_descriptor)
        entries = tuple(sorted(os.listdir(lease.root_descriptor)))
        filesystem_after = os.fstatvfs(lease.root_descriptor)
        mount_info_after = _read_mount_info(
            lease.process_descriptor,
            lease.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        metadata_after = os.fstat(lease.root_descriptor)
        _require_same_mounted_runtime_volume(lease, keeper)
    if (
        _stable_metadata(metadata_after) != _stable_metadata(metadata_before)
        or mount_info_after != mount_info_before
        or _stable_filesystem(filesystem_after) != _stable_filesystem(filesystem_before)
        or entries
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume changed during empty-generation observation"
        )
    _require_mount_authority(
        mount_info_before,
        metadata_before,
        authority,
    )
    filesystem = filesystem_before
    if (
        filesystem.f_bsize != filesystem.f_frsize
        or filesystem.f_bfree != filesystem.f_bavail
        or filesystem.f_ffree != filesystem.f_favail
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume statvfs carries reserved or inconsistent capacity"
        )
    allocation_block_size_bytes = filesystem.f_frsize
    effective_block_count = filesystem.f_blocks
    available_block_count = filesystem.f_bfree
    used_block_count = effective_block_count - available_block_count
    effective_inode_limit = filesystem.f_files
    available_inode_count = filesystem.f_ffree
    used_inode_count = effective_inode_limit - available_inode_count
    return DockerRunActionEmptyVolumeObservation(
        runtime_volume_authority=authority,
        docker_volume_observation=volume,
        keeper_container_id=lease.keeper_container_id,
        keeper_process_id=lease.keeper_process_id,
        keeper_process_start_time_ticks=lease.process_start_time_ticks,
        process_cgroup_path=lease.process_cgroup_path,
        mount_id=lease.root_mount_id,
        device=metadata_before.st_dev,
        root_inode=metadata_before.st_ino,
        filesystem_type=mount_info_before.filesystem_type,
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        owner_user_id=metadata_before.st_uid,
        owner_group_id=metadata_before.st_gid,
        root_mode=stat.S_IMODE(metadata_before.st_mode),
        allocation_block_size_bytes=allocation_block_size_bytes,
        effective_block_count=effective_block_count,
        effective_size_bytes=effective_block_count * allocation_block_size_bytes,
        effective_inode_limit=effective_inode_limit,
        used_block_count=used_block_count,
        used_size_bytes=used_block_count * allocation_block_size_bytes,
        used_inode_count=used_inode_count,
        available_block_count=available_block_count,
        available_size_bytes=available_block_count * allocation_block_size_bytes,
        available_inode_count=available_inode_count,
        empty_entry_count=0,
        empty_size_bytes=0,
    )


def materialize_runtime_volume_layout(
    claim: RunActionPreparationClaim,
    empty_volume: DockerRunActionEmptyVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    *,
    workspace_descriptor: int | None,
    settings: LaunchSettings,
) -> DockerRunActionPreparedVolumeObservation:
    """Populate one proven-empty generation and publish its sentinel last."""

    if (
        type(claim) is not RunActionPreparationClaim
        or type(empty_volume) is not DockerRunActionEmptyVolumeObservation
        or type(keeper) is not RunActionVolumeKeeperEvidence
        or type(settings) is not LaunchSettings
        or empty_volume.runtime_volume_authority.preparation_claim_id
        != claim.preparation_claim_id
        or empty_volume.docker_volume_observation.volume_authority_id
        != empty_volume.runtime_volume_authority.runtime_volume_authority_id
        or keeper.preparation_claim_id != claim.preparation_claim_id
        or keeper.issued_create_projection.execution_policy != claim.execution_policy
        or keeper.issued_create_projection.volume_authority
        != empty_volume.runtime_volume_authority
        or (claim.execution_policy.user_id, claim.execution_policy.group_id)
        != (os.geteuid(), os.getegid())
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume materialization requires exact writable authority"
        )
    plan = _plan_runtime_volume_layout(
        claim,
        empty_volume,
        workspace_descriptor=workspace_descriptor,
        settings=settings,
    )
    with ExitStack() as descriptors:
        lease = _open_mounted_runtime_volume(descriptors, keeper)
        _require_empty_volume_lease(empty_volume, keeper, lease)
        workspace_frontier = _materialize_layout_at_descriptor(
            lease.root_descriptor,
            claim=claim,
            authority=empty_volume.runtime_volume_authority,
            plan=plan,
            workspace_descriptor=workspace_descriptor,
            settings=settings,
        )
        observed = _observe_prepared_layout_at_descriptor(
            lease,
            claim=claim,
            authority=empty_volume.runtime_volume_authority,
            keeper=keeper,
            workspace_frontier=workspace_frontier,
            settings=settings,
        )
        _require_same_mounted_runtime_volume(lease, keeper)
    return _mint_prepared_volume_observation(
        claim,
        empty_volume.runtime_volume_authority,
        empty_volume.docker_volume_observation,
        keeper,
        observed,
        empty_entry_count=empty_volume.empty_entry_count,
        empty_size_bytes=empty_volume.empty_size_bytes,
    )


def reobserve_runtime_volume_layout(
    prepared: RunActionPreparedExecution,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    *,
    settings: LaunchSettings,
) -> DockerRunActionPreparedVolumeObservation:
    """Reopen only one durably prepared generation without mutating it."""

    if (
        type(prepared) is not RunActionPreparedExecution
        or type(volume) is not DockerRunActionVolumeObservation
        or type(keeper) is not RunActionVolumeKeeperEvidence
        or type(settings) is not LaunchSettings
        or volume.volume_authority_id
        != prepared.runtime_volume_authority.runtime_volume_authority_id
        or volume.volume_name != prepared.runtime_volume_authority.volume_name
        or keeper != prepared.volume_keeper_evidence
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume reopen requires exact durable prepared authority"
        )
    with ExitStack() as descriptors:
        lease = _open_mounted_runtime_volume(descriptors, keeper)
        evidence = prepared.runtime_volume_evidence
        if (
            lease.root_mount_id != evidence.root_mount_id
            or lease.root_device != evidence.root_device
            or lease.root_inode != evidence.root_inode
        ):
            raise RunActionRuntimeVolumeError(
                "prepared runtime volume physical root was substituted"
            )
        observed = _observe_prepared_layout_at_descriptor(
            lease,
            claim=prepared.preparation_claim,
            authority=prepared.runtime_volume_authority,
            keeper=keeper,
            workspace_frontier=(
                None
                if prepared.workspace_proof is None
                else prepared.workspace_proof.workspace_binding.to_identity()
            ),
            settings=settings,
        )
        _require_same_mounted_runtime_volume(lease, keeper)
    reopened = _mint_prepared_volume_observation(
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        keeper,
        observed,
        empty_entry_count=prepared.layout_proof.empty_entry_count,
        empty_size_bytes=prepared.layout_proof.empty_size_bytes,
    )
    if reopened != DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_file=prepared.input_file,
        result_file=prepared.result_file,
        credential_file=prepared.credential_file,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    ):
        raise RunActionRuntimeVolumeError(
            "reopened runtime volume differs from durable prepared layout"
        )
    return reopened


def _plan_runtime_volume_layout(
    claim: RunActionPreparationClaim,
    empty_volume: DockerRunActionEmptyVolumeObservation,
    *,
    workspace_descriptor: int | None,
    settings: LaunchSettings,
) -> _RuntimeVolumeLayoutPlan:
    workspace_binding = claim.reservation.frontier.workspace_before
    workspace_access = claim.reservation.intent.workspace_access
    if workspace_access is RunFrontierWorkspaceAccess.NONE:
        if workspace_descriptor is not None or workspace_binding is not None:
            raise RunActionRuntimeVolumeError(
                "workspace-free layout carries workspace authority"
            )
        workspace_copy_plan = None
    else:
        if (
            type(workspace_descriptor) is not int
            or workspace_descriptor < 0
            or workspace_binding is None
        ):
            raise RunActionRuntimeVolumeError(
                "workspace layout lacks its exact source descriptor"
            )
        workspace_copy_plan = plan_run_workspace_frontier_copy(
            workspace_descriptor,
            settings=settings,
            expected=workspace_binding.to_identity(),
        )
    directory_names = _expected_directory_names(claim)
    file_plans = _expected_file_plans(claim)
    block_size = empty_volume.allocation_block_size_bytes
    nonworkspace_directory_count = len(directory_names) - (
        1 if workspace_copy_plan is not None else 0
    )
    workspace_size_bytes = (
        0
        if workspace_copy_plan is None
        else workspace_copy_plan.allocated_size_bytes(block_size)
    )
    workspace_inode_count = (
        0 if workspace_copy_plan is None else workspace_copy_plan.physical_entry_count
    )
    current_size_bytes = (
        block_size
        + nonworkspace_directory_count * block_size
        + workspace_size_bytes
        + _allocated_size_bytes(
            len(empty_volume.runtime_volume_authority.generation_nonce),
            block_size,
        )
    )
    limits = claim.execution_policy.docker_resource_limits
    future_size_bytes = sum(
        _allocated_size_bytes(file_plan.payload_size_limit_bytes, block_size)
        for file_plan in file_plans
    ) + _allocated_size_bytes(
        limits.runtime_temporary_reservation_size_bytes,
        block_size,
    )
    current_inode_count = (
        1 + nonworkspace_directory_count + workspace_inode_count + len(file_plans) + 1
    )
    if (
        current_size_bytes + future_size_bytes >= empty_volume.available_size_bytes
        or current_inode_count + limits.runtime_temporary_reservation_inode_count
        >= empty_volume.available_inode_count
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume lacks peak preparation and execution headroom"
        )
    return _RuntimeVolumeLayoutPlan(
        directory_names=directory_names,
        file_plans=file_plans,
        workspace_copy_plan=workspace_copy_plan,
        preparation_size_bytes=current_size_bytes,
        preparation_inode_count=current_inode_count,
    )


def _materialize_layout_at_descriptor(
    root_descriptor: int,
    *,
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    plan: _RuntimeVolumeLayoutPlan,
    workspace_descriptor: int | None,
    settings: LaunchSettings,
) -> RunWorkspaceFrontierIdentity | None:
    if tuple(sorted(os.listdir(root_descriptor))):
        raise RunActionRuntimeVolumeError(
            "runtime volume is no longer empty before materialization"
        )
    staging_name = _PREPARATION_STAGING_PREFIX + authority.generation_nonce
    pending_sentinel_name = _PENDING_SENTINEL_PREFIX + authority.generation_nonce
    with ExitStack() as descriptors:
        staging_descriptor = _create_private_directory(
            root_descriptor,
            staging_name,
            authority,
        )
        descriptors.callback(os.close, staging_descriptor)
        directory_descriptors = {}
        for directory_name in plan.directory_names:
            directory_descriptor = _create_private_directory(
                staging_descriptor,
                directory_name,
                authority,
            )
            descriptors.callback(os.close, directory_descriptor)
            directory_descriptors[directory_name] = directory_descriptor
        for file_plan in plan.file_plans:
            _create_empty_prepared_file(
                directory_descriptors[file_plan.directory_name],
                file_plan.file_name,
                authority,
            )
        workspace_frontier = None
        if plan.workspace_copy_plan is not None:
            if type(workspace_descriptor) is not int:
                raise RunActionRuntimeVolumeError(
                    "workspace copy lost its source descriptor"
                )
            workspace_frontier = copy_run_workspace_frontier(
                workspace_descriptor,
                directory_descriptors["workspace"],
                settings=settings,
                plan=plan.workspace_copy_plan,
            )
        _create_staged_sentinel(
            staging_descriptor,
            authority,
        )
        for directory_descriptor in directory_descriptors.values():
            os.fsync(directory_descriptor)
        os.fsync(staging_descriptor)
        for directory_name in plan.directory_names:
            _rename_no_replace(
                staging_descriptor,
                directory_name,
                root_descriptor,
                directory_name,
            )
        _rename_no_replace(
            staging_descriptor,
            _SENTINEL_NAME,
            root_descriptor,
            pending_sentinel_name,
        )
        os.fsync(root_descriptor)
        os.rmdir(staging_name, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
        if tuple(sorted(os.listdir(root_descriptor))) != tuple(
            sorted((*plan.directory_names, pending_sentinel_name))
        ):
            raise RunActionRuntimeVolumeError(
                "runtime volume publication contains an unexpected path"
            )
        _rename_no_replace(
            root_descriptor,
            pending_sentinel_name,
            root_descriptor,
            _SENTINEL_NAME,
        )
        os.fsync(root_descriptor)
    if tuple(sorted(os.listdir(root_descriptor))) != tuple(
        sorted((*plan.directory_names, _SENTINEL_NAME))
    ):
        raise RunActionRuntimeVolumeError(
            "published runtime volume layout is incomplete"
        )
    return workspace_frontier


def _create_private_directory(
    parent_descriptor: int,
    name: str,
    authority: RunActionRuntimeVolumeAuthority,
) -> int:
    os.mkdir(name, mode=_PREPARED_DIRECTORY_MODE, dir_fd=parent_descriptor)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    os.fchmod(descriptor, _PREPARED_DIRECTORY_MODE)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != authority.owner_user_id
        or metadata.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata.st_mode) != _PREPARED_DIRECTORY_MODE
        or metadata.st_dev <= 0
        or metadata.st_ino <= 0
    ):
        os.close(descriptor)
        raise RunActionRuntimeVolumeError(
            "runtime volume prepared directory has unsafe metadata"
        )
    return descriptor


def _create_empty_prepared_file(
    parent_descriptor: int,
    name: str,
    authority: RunActionRuntimeVolumeAuthority,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        _PREPARED_FILE_MODE,
        dir_fd=parent_descriptor,
    )
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), _PREPARED_FILE_MODE)
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != authority.owner_user_id
            or metadata.st_gid != authority.owner_group_id
            or stat.S_IMODE(metadata.st_mode) != _PREPARED_FILE_MODE
            or metadata.st_nlink != 1
            or metadata.st_size != 0
        ):
            raise RunActionRuntimeVolumeError(
                "runtime volume prepared file has unsafe metadata"
            )
        os.fsync(handle.fileno())
    os.fsync(parent_descriptor)


def _create_staged_sentinel(
    staging_descriptor: int,
    authority: RunActionRuntimeVolumeAuthority,
) -> None:
    payload = authority.generation_nonce.encode("ascii")
    descriptor = os.open(
        _SENTINEL_NAME,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        _SENTINEL_MODE,
        dir_fd=staging_descriptor,
    )
    with os.fdopen(descriptor, "wb", buffering=0) as handle:
        os.fchmod(handle.fileno(), _SENTINEL_MODE)
        written_size = 0
        while written_size < len(payload):
            written = handle.write(payload[written_size:])
            if written <= 0:
                raise RunActionRuntimeVolumeError(
                    "runtime volume sentinel write made no progress"
                )
            written_size += written
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != authority.owner_user_id
            or metadata.st_gid != authority.owner_group_id
            or stat.S_IMODE(metadata.st_mode) != _SENTINEL_MODE
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise RunActionRuntimeVolumeError(
                "runtime volume generation sentinel has unsafe metadata"
            )
        os.fsync(handle.fileno())


def _rename_no_replace(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise RunActionRuntimeVolumeError(
            "atomic no-replace runtime-volume publication is unavailable"
        )
    rename_at2 = libc.renameat2
    rename_at2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_at2.restype = ctypes.c_int
    result = rename_at2(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _allocated_size_bytes(size_bytes: int, block_size_bytes: int) -> int:
    return ((size_bytes + block_size_bytes - 1) // block_size_bytes) * block_size_bytes


def _require_empty_volume_lease(
    empty_volume: DockerRunActionEmptyVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    lease: _MountedRuntimeVolumeLease,
) -> None:
    metadata = os.fstat(lease.root_descriptor)
    mount_info = _read_mount_info(
        lease.process_descriptor,
        lease.root_mount_id,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )
    filesystem = os.fstatvfs(lease.root_descriptor)
    _require_mount_authority(
        mount_info,
        metadata,
        empty_volume.runtime_volume_authority,
    )
    _require_consistent_filesystem(filesystem)
    if (
        empty_volume.keeper_container_id != lease.keeper_container_id
        or empty_volume.keeper_process_id != lease.keeper_process_id
        or empty_volume.keeper_process_start_time_ticks
        != lease.process_start_time_ticks
        or empty_volume.process_cgroup_path != lease.process_cgroup_path
        or empty_volume.mount_id != lease.root_mount_id
        or empty_volume.device != lease.root_device
        or empty_volume.root_inode != lease.root_inode
        or empty_volume.keeper_container_id != keeper.container_id
        or _filesystem_capacity(filesystem)
        != (
            empty_volume.allocation_block_size_bytes,
            empty_volume.effective_block_count,
            empty_volume.effective_size_bytes,
            empty_volume.effective_inode_limit,
            empty_volume.used_block_count,
            empty_volume.used_size_bytes,
            empty_volume.used_inode_count,
            empty_volume.available_block_count,
            empty_volume.available_size_bytes,
            empty_volume.available_inode_count,
        )
        or tuple(sorted(os.listdir(lease.root_descriptor)))
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume no longer matches its empty-generation proof"
        )


def _observe_prepared_layout_at_descriptor(
    lease: _MountedRuntimeVolumeLease,
    *,
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    keeper: RunActionVolumeKeeperEvidence,
    workspace_frontier: RunWorkspaceFrontierIdentity | None,
    settings: LaunchSettings,
) -> _PreparedLayoutObservation:
    expected_directories = _expected_directory_names(claim)
    expected_root_entries = tuple(sorted((*expected_directories, _SENTINEL_NAME)))
    metadata_before = os.fstat(lease.root_descriptor)
    mount_info_before = _read_mount_info(
        lease.process_descriptor,
        lease.root_mount_id,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )
    filesystem_before = os.fstatvfs(lease.root_descriptor)
    _require_mount_authority(mount_info_before, metadata_before, authority)
    _require_consistent_filesystem(filesystem_before)
    if tuple(sorted(os.listdir(lease.root_descriptor))) != expected_root_entries:
        raise RunActionRuntimeVolumeError(
            "prepared runtime volume root topology is incomplete"
        )
    observed_identities = {
        (metadata_before.st_dev, metadata_before.st_ino),
    }
    directory_descriptors = {}
    directory_metadata_before = {}
    with ExitStack() as descriptors:
        for directory_name in expected_directories:
            directory_descriptor = os.open(
                directory_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=lease.root_descriptor,
            )
            descriptors.callback(os.close, directory_descriptor)
            directory_metadata = os.fstat(directory_descriptor)
            directory_mount_id = read_run_action_descriptor_mount_id(
                directory_descriptor
            )
            identity = (directory_metadata.st_dev, directory_metadata.st_ino)
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or directory_metadata.st_uid != authority.owner_user_id
                or directory_metadata.st_gid != authority.owner_group_id
                or stat.S_IMODE(directory_metadata.st_mode) != _PREPARED_DIRECTORY_MODE
                or directory_mount_id != lease.root_mount_id
                or directory_metadata.st_dev != lease.root_device
                or identity in observed_identities
            ):
                raise RunActionRuntimeVolumeError(
                    "prepared runtime volume directory is unsafe or substituted"
                )
            observed_identities.add(identity)
            directory_descriptors[directory_name] = directory_descriptor
            directory_metadata_before[directory_name] = _stable_metadata(
                directory_metadata
            )
        sentinel_observation = _open_exact_regular_file(
            descriptors,
            lease.root_descriptor,
            _SENTINEL_NAME,
            expected_payload=authority.generation_nonce.encode("ascii"),
            expected_mode=_SENTINEL_MODE,
            authority=authority,
            root_mount_id=lease.root_mount_id,
            root_device=lease.root_device,
        )
        sentinel_identity = (
            sentinel_observation.metadata.st_dev,
            sentinel_observation.metadata.st_ino,
        )
        if sentinel_identity in observed_identities:
            raise RunActionRuntimeVolumeError(
                "runtime volume sentinel repeats another layout inode"
            )
        observed_identities.add(sentinel_identity)
        file_plans = _expected_file_plans(claim)
        expected_children = {
            "input": ("request.blob",),
            "result": ("result.blob",),
            "temporary": (),
            **(
                {}
                if claim.execution_policy.credential_policy.mode
                is RunActionCredentialMode.NONE
                else {"credential": ("credentials",)}
            ),
        }
        for directory_name, child_names in expected_children.items():
            if tuple(sorted(os.listdir(directory_descriptors[directory_name]))) != (
                child_names
            ):
                raise RunActionRuntimeVolumeError(
                    "prepared runtime volume logical directory has extra paths"
                )
        prepared_file_observations = []
        for file_plan in file_plans:
            file_observation = _open_exact_regular_file(
                descriptors,
                directory_descriptors[file_plan.directory_name],
                file_plan.file_name,
                expected_payload=b"",
                expected_mode=_PREPARED_FILE_MODE,
                authority=authority,
                root_mount_id=lease.root_mount_id,
                root_device=lease.root_device,
            )
            identity = (
                file_observation.metadata.st_dev,
                file_observation.metadata.st_ino,
            )
            if identity in observed_identities:
                raise RunActionRuntimeVolumeError(
                    "prepared runtime volume file repeats another layout inode"
                )
            observed_identities.add(identity)
            prepared_file_observations.append(file_observation)
        observed_workspace_frontier = None
        if workspace_frontier is None:
            if "workspace" in directory_descriptors:
                raise RunActionRuntimeVolumeError(
                    "workspace-free runtime volume contains a workspace"
                )
        else:
            workspace_descriptor = directory_descriptors["workspace"]
            observed_workspace_frontier = inspect_run_workspace_frontier(
                workspace_descriptor,
                settings=settings,
                expected_commit_sha=workspace_frontier.commit_sha,
            )
            if not _same_workspace_semantics(
                observed_workspace_frontier,
                workspace_frontier,
            ):
                raise RunActionRuntimeVolumeError(
                    "prepared workspace differs from its durable frontier"
                )
            plan_run_workspace_frontier_copy(
                workspace_descriptor,
                settings=settings,
                expected=observed_workspace_frontier,
            )
        metadata_after = os.fstat(lease.root_descriptor)
        mount_info_after = _read_mount_info(
            lease.process_descriptor,
            lease.root_mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        filesystem_after = os.fstatvfs(lease.root_descriptor)
        _require_same_mounted_runtime_volume(lease, keeper)
        _require_same_exact_regular_file(sentinel_observation)
        for file_observation in prepared_file_observations:
            _require_same_exact_regular_file(file_observation)
        directory_metadata_after = {
            directory_name: _stable_metadata(os.fstat(directory_descriptor))
            for directory_name, directory_descriptor in directory_descriptors.items()
        }
    if (
        _root_metadata_identity(metadata_after)
        != _root_metadata_identity(metadata_before)
        or mount_info_after != mount_info_before
        or _stable_filesystem(filesystem_after) != _stable_filesystem(filesystem_before)
        or directory_metadata_after != directory_metadata_before
        or tuple(sorted(os.listdir(lease.root_descriptor))) != expected_root_entries
    ):
        raise RunActionRuntimeVolumeError(
            "prepared runtime volume changed during exact observation"
        )
    return _PreparedLayoutObservation(
        root_mount_id=lease.root_mount_id,
        root_device=lease.root_device,
        root_inode=lease.root_inode,
        sentinel_metadata=sentinel_observation.metadata,
        sentinel_mount_id=sentinel_observation.mount_id,
        workspace_frontier=observed_workspace_frontier,
        filesystem=filesystem_before,
    )


def _open_exact_regular_file(
    descriptors: ExitStack,
    parent_descriptor: int,
    name: str,
    *,
    expected_payload: bytes,
    expected_mode: int,
    authority: RunActionRuntimeVolumeAuthority,
    root_mount_id: int,
    root_device: int,
) -> _ExactRegularFileObservation:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata_before = os.fstat(descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(descriptor)
    payload = _read_bounded_descriptor_payload(
        descriptor,
        len(expected_payload) + 1,
    )
    metadata_after = os.fstat(descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(descriptor)
    if (
        _stable_metadata(metadata_after) != _stable_metadata(metadata_before)
        or mount_id_after != mount_id_before
        or not stat.S_ISREG(metadata_before.st_mode)
        or metadata_before.st_uid != authority.owner_user_id
        or metadata_before.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata_before.st_mode) != expected_mode
        or metadata_before.st_nlink != 1
        or metadata_before.st_size != len(expected_payload)
        or payload != expected_payload
        or mount_id_before != root_mount_id
        or metadata_before.st_dev != root_device
    ):
        raise RunActionRuntimeVolumeError(
            "prepared runtime volume file is unsafe or substituted"
        )
    return _ExactRegularFileObservation(
        descriptor=descriptor,
        parent_descriptor=parent_descriptor,
        name=name,
        metadata=metadata_before,
        mount_id=mount_id_before,
        payload=payload,
    )


def _require_same_exact_regular_file(
    observation: _ExactRegularFileObservation,
) -> None:
    os.lseek(observation.descriptor, 0, os.SEEK_SET)
    payload = _read_bounded_descriptor_payload(
        observation.descriptor,
        len(observation.payload) + 1,
    )
    metadata = os.fstat(observation.descriptor)
    mount_id = read_run_action_descriptor_mount_id(observation.descriptor)
    path_descriptor = os.open(
        observation.name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=observation.parent_descriptor,
    )
    with ExitStack() as path_descriptors:
        path_descriptors.callback(os.close, path_descriptor)
        path_metadata_before = os.fstat(path_descriptor)
        path_mount_id_before = read_run_action_descriptor_mount_id(path_descriptor)
        path_payload = _read_bounded_descriptor_payload(
            path_descriptor,
            len(observation.payload) + 1,
        )
        path_metadata_after = os.fstat(path_descriptor)
        path_mount_id_after = read_run_action_descriptor_mount_id(path_descriptor)
        if (
            payload != observation.payload
            or _stable_metadata(metadata) != _stable_metadata(observation.metadata)
            or mount_id != observation.mount_id
            or path_payload != observation.payload
            or _stable_metadata(path_metadata_before)
            != _stable_metadata(observation.metadata)
            or _stable_metadata(path_metadata_after)
            != _stable_metadata(path_metadata_before)
            or path_mount_id_before != observation.mount_id
            or path_mount_id_after != path_mount_id_before
        ):
            raise RunActionRuntimeVolumeError(
                "prepared runtime volume file changed during exact observation"
            )


def _read_bounded_descriptor_payload(descriptor: int, limit: int) -> bytes:
    chunks = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _mint_prepared_volume_observation(
    claim: RunActionPreparationClaim,
    authority: RunActionRuntimeVolumeAuthority,
    volume: DockerRunActionVolumeObservation,
    keeper: RunActionVolumeKeeperEvidence,
    observed: _PreparedLayoutObservation,
    *,
    empty_entry_count: int,
    empty_size_bytes: int,
) -> DockerRunActionPreparedVolumeObservation:
    file_plans = _expected_file_plans(claim)
    prepared_files = tuple(
        RunActionPreparedFile.mint(
            preparation_claim_id=claim.preparation_claim_id,
            runtime_volume_authority_id=authority.runtime_volume_authority_id,
            generation_nonce=authority.generation_nonce,
            kind=file_plan.kind,
            relative_path=file_plan.relative_path,
            file_type="regular",
            owner_user_id=authority.owner_user_id,
            owner_group_id=authority.owner_group_id,
            mode=_PREPARED_FILE_MODE,
            link_count=1,
            size_bytes=0,
            payload_size_limit_bytes=file_plan.payload_size_limit_bytes,
        )
        for file_plan in file_plans
    )
    prepared_by_kind = {
        prepared_file.kind: prepared_file for prepared_file in prepared_files
    }
    workspace_binding = claim.reservation.frontier.workspace_before
    if (workspace_binding is None) != (observed.workspace_frontier is None) or (
        workspace_binding is not None
        and not _same_workspace_semantics(
            observed.workspace_frontier,
            workspace_binding.to_identity(),
        )
    ):
        raise RunActionRuntimeVolumeError(
            "observed prepared workspace differs from its claim"
        )
    workspace_proof = (
        None
        if workspace_binding is None
        else RunActionPreparedWorkspaceProof.mint(
            preparation_claim_id=claim.preparation_claim_id,
            runtime_volume_authority_id=authority.runtime_volume_authority_id,
            generation_nonce=authority.generation_nonce,
            volume_subpath="workspace",
            workspace_binding=workspace_binding,
            observed_source_tree_digest=workspace_binding.source_tree_digest,
            observed_git_closure_digest=workspace_binding.git_closure_digest,
            observed_source_entry_count=workspace_binding.source_entry_count,
            observed_source_size_bytes=workspace_binding.source_size_bytes,
            owner_user_id=authority.owner_user_id,
            owner_group_id=authority.owner_group_id,
            root_mode=_PREPARED_DIRECTORY_MODE,
            unexpected_entry_count=0,
        )
    )
    sentinel_evidence = RunActionRuntimeVolumeSentinelEvidence.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        generation_nonce=authority.generation_nonce,
        relative_path=_SENTINEL_NAME,
        file_type="regular",
        owner_user_id=observed.sentinel_metadata.st_uid,
        owner_group_id=observed.sentinel_metadata.st_gid,
        mode=stat.S_IMODE(observed.sentinel_metadata.st_mode),
        link_count=observed.sentinel_metadata.st_nlink,
        size_bytes=observed.sentinel_metadata.st_size,
        content_digest=tree_or_blob_digest(authority.generation_nonce.encode("ascii")),
        mount_id=observed.sentinel_mount_id,
        device=observed.sentinel_metadata.st_dev,
        inode=observed.sentinel_metadata.st_ino,
    )
    filesystem = observed.filesystem
    _require_consistent_filesystem(filesystem)
    (
        allocation_block_size_bytes,
        effective_block_count,
        effective_size_bytes,
        effective_inode_limit,
        used_block_count,
        used_size_bytes,
        used_inode_count,
        available_block_count,
        available_size_bytes,
        available_inode_count,
    ) = _filesystem_capacity(filesystem)
    limits = claim.execution_policy.docker_resource_limits
    required_available_size_bytes = sum(
        _allocated_size_bytes(
            file_plan.payload_size_limit_bytes,
            allocation_block_size_bytes,
        )
        for file_plan in file_plans
    ) + _allocated_size_bytes(
        limits.runtime_temporary_reservation_size_bytes,
        allocation_block_size_bytes,
    )
    if (
        required_available_size_bytes >= available_size_bytes
        or limits.runtime_temporary_reservation_inode_count >= available_inode_count
    ):
        raise RunActionRuntimeVolumeError(
            "prepared runtime volume lacks positive execution headroom"
        )
    volume_evidence = RunActionRuntimeVolumeEvidence.mint(
        volume_authority=authority,
        volume_keeper_evidence_id=keeper.volume_keeper_evidence_id,
        keeper_container_id=keeper.container_id,
        keeper_process_id=keeper.process_id,
        keeper_process_start_time_ticks=keeper.process_start_time_ticks,
        keeper_process_cgroup_path=keeper.mounted_helper_evidence.process_cgroup_path,
        root_mount_id=observed.root_mount_id,
        root_device=observed.root_device,
        root_inode=observed.root_inode,
        observed_volume_name=volume.volume_name,
        observed_labels=authority.labels,
        observed_scope="local",
        observed_driver=authority.driver,
        observed_driver_options=authority.driver_options,
        observed_filesystem_type=_TMPFS_FILESYSTEM_TYPE,
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        observed_owner_user_id=authority.owner_user_id,
        observed_owner_group_id=authority.owner_group_id,
        observed_root_mode=authority.root_mode,
        allocation_block_size_bytes=allocation_block_size_bytes,
        effective_block_count=effective_block_count,
        effective_size_bytes=effective_size_bytes,
        effective_inode_limit=effective_inode_limit,
        used_block_count=used_block_count,
        used_size_bytes=used_size_bytes,
        used_inode_count=used_inode_count,
        available_block_count=available_block_count,
        available_size_bytes=available_size_bytes,
        available_inode_count=available_inode_count,
        sentinel_evidence=sentinel_evidence,
    )
    expected_directories = _expected_directory_names(claim)
    logical_workspace_size = (
        0 if workspace_binding is None else workspace_binding.source_size_bytes
    )
    logical_workspace_entries = (
        0 if workspace_binding is None else workspace_binding.source_entry_count
    )
    layout_proof = RunActionRuntimeVolumeLayoutProof.mint(
        runtime_volume_authority_id=authority.runtime_volume_authority_id,
        runtime_volume_evidence_id=volume_evidence.runtime_volume_evidence_id,
        generation_nonce=authority.generation_nonce,
        empty_size_bytes=empty_size_bytes,
        empty_entry_count=empty_entry_count,
        directory_relative_paths=expected_directories,
        prepared_file_ids=tuple(
            sorted(prepared_file.prepared_file_id for prepared_file in prepared_files)
        ),
        prepared_workspace_proof_id=(
            None
            if workspace_proof is None
            else workspace_proof.prepared_workspace_proof_id
        ),
        logical_content_size_bytes=(
            len(authority.generation_nonce) + logical_workspace_size
        ),
        logical_entry_count=(
            len(expected_directories)
            + len(prepared_files)
            + 1
            + logical_workspace_entries
        ),
        observed_used_size_bytes=used_size_bytes,
        observed_used_inode_count=used_inode_count,
        unexpected_entry_count=0,
    )
    return DockerRunActionPreparedVolumeObservation(
        preparation_claim=claim,
        runtime_volume_evidence=volume_evidence,
        input_file=prepared_by_kind[RunActionPreparedFileKind.INPUT],
        result_file=prepared_by_kind[RunActionPreparedFileKind.RESULT],
        credential_file=prepared_by_kind.get(RunActionPreparedFileKind.CREDENTIAL),
        workspace_proof=workspace_proof,
        layout_proof=layout_proof,
    )


def _expected_directory_names(claim: RunActionPreparationClaim) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                "input",
                "result",
                "temporary",
                *(
                    ()
                    if claim.execution_policy.credential_policy.mode
                    is RunActionCredentialMode.NONE
                    else ("credential",)
                ),
                *(
                    ()
                    if claim.reservation.frontier.workspace_before is None
                    else ("workspace",)
                ),
            )
        )
    )


def _expected_file_plans(
    claim: RunActionPreparationClaim,
) -> tuple[_PreparedFilePlan, ...]:
    credential_policy = claim.execution_policy.credential_policy
    return (
        _PreparedFilePlan(
            kind=RunActionPreparedFileKind.INPUT,
            directory_name="input",
            file_name="request.blob",
            relative_path="input/request.blob",
            payload_size_limit_bytes=claim.reservation.request_blob.size_bytes,
        ),
        _PreparedFilePlan(
            kind=RunActionPreparedFileKind.RESULT,
            directory_name="result",
            file_name="result.blob",
            relative_path="result/result.blob",
            payload_size_limit_bytes=(
                claim.execution_policy.supervisor_limits.result_size_bytes
            ),
        ),
        *(
            ()
            if credential_policy.mode is RunActionCredentialMode.NONE
            else (
                _PreparedFilePlan(
                    kind=RunActionPreparedFileKind.CREDENTIAL,
                    directory_name="credential",
                    file_name="credentials",
                    relative_path="credential/credentials",
                    payload_size_limit_bytes=(
                        credential_policy.maximum_delivery_size_bytes
                    ),
                ),
            )
        ),
    )


def _same_workspace_semantics(
    observed: RunWorkspaceFrontierIdentity,
    expected: RunWorkspaceFrontierIdentity,
) -> bool:
    return (
        observed.branch == expected.branch
        and observed.commit_sha == expected.commit_sha
        and observed.parent_commit_shas == expected.parent_commit_shas
        and observed.git_tree_sha == expected.git_tree_sha
        and observed.source_tree_digest == expected.source_tree_digest
        and observed.git_closure_digest == expected.git_closure_digest
        and observed.source_entry_count == expected.source_entry_count
        and observed.source_size_bytes == expected.source_size_bytes
    )


def _require_consistent_filesystem(filesystem: os.statvfs_result) -> None:
    if (
        filesystem.f_bsize != filesystem.f_frsize
        or filesystem.f_bfree != filesystem.f_bavail
        or filesystem.f_ffree != filesystem.f_favail
    ):
        raise RunActionRuntimeVolumeError(
            "runtime volume statvfs carries reserved or inconsistent capacity"
        )


def _filesystem_capacity(
    filesystem: os.statvfs_result,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    allocation_block_size_bytes = filesystem.f_frsize
    effective_block_count = filesystem.f_blocks
    available_block_count = filesystem.f_bfree
    used_block_count = effective_block_count - available_block_count
    effective_inode_limit = filesystem.f_files
    available_inode_count = filesystem.f_ffree
    used_inode_count = effective_inode_limit - available_inode_count
    return (
        allocation_block_size_bytes,
        effective_block_count,
        effective_block_count * allocation_block_size_bytes,
        effective_inode_limit,
        used_block_count,
        used_block_count * allocation_block_size_bytes,
        used_inode_count,
        available_block_count,
        available_block_count * allocation_block_size_bytes,
        available_inode_count,
    )


def _root_metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_ino,
        metadata.st_dev,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_mount_info(
    process_descriptor: int,
    mount_id: int,
    destination: str,
) -> _MountInfo:
    descriptor = os.open(
        "mountinfo",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read()
    return _parse_mount_info_payload(payload, mount_id, destination)


def _parse_mount_info_payload(
    payload: bytes,
    mount_id: int,
    destination: str,
) -> _MountInfo:
    if (
        type(payload) is not bytes
        or not payload
        or not payload.endswith(b"\n")
        or b"\x00" in payload
        or type(mount_id) is not int
        or mount_id <= 0
        or type(destination) is not str
        or destination != RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION
    ):
        raise RunActionRuntimeVolumeError("keeper mountinfo is malformed or incomplete")
    encoded_lines = payload.splitlines()
    matching_lines = tuple(
        line
        for line in encoded_lines
        if line.split(b" ", 1)[0] == str(mount_id).encode()
    )
    if len(matching_lines) != 1 or not matching_lines[0].isascii():
        raise RunActionRuntimeVolumeError(
            "keeper mountinfo lacks one runtime-volume mount"
        )
    line = matching_lines[0].decode("ascii")
    sections = line.split(" - ")
    if len(sections) != 2:
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mountinfo separator is malformed"
        )
    mount_fields = sections[0].split(" ")
    filesystem_fields = sections[1].split(" ")
    if (
        len(mount_fields) < 6
        or len(filesystem_fields) != 3
        or not mount_fields[0].isdigit()
        or not mount_fields[1].isdigit()
    ):
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mountinfo fields are malformed"
        )
    device_parts = mount_fields[2].split(":")
    if (
        len(device_parts) != 2
        or any(not part.isdigit() for part in device_parts)
        or mount_fields[3] != "/"
        or mount_fields[4] != destination
    ):
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mountinfo path or device is malformed"
        )
    return _MountInfo(
        mount_id=int(mount_fields[0]),
        parent_mount_id=int(mount_fields[1]),
        device_major=int(device_parts[0]),
        device_minor=int(device_parts[1]),
        mount_point=mount_fields[4],
        mount_options=_parse_options(mount_fields[5]),
        optional_fields=tuple(mount_fields[6:]),
        filesystem_type=filesystem_fields[0],
        source=filesystem_fields[1],
        super_options=_parse_options(filesystem_fields[2]),
    )


def _parse_options(value: str) -> tuple[str, ...]:
    options = tuple(value.split(","))
    if (
        not options
        or any(
            not option or any(character.isspace() for character in option)
            for option in options
        )
        or len(options) != len(set(options))
    ):
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mount options are malformed"
        )
    return tuple(sorted(options))


def _require_mount_authority(
    mount_info: _MountInfo,
    metadata: os.stat_result,
    authority: RunActionRuntimeVolumeAuthority,
) -> None:
    flag_options = tuple(
        option for option in mount_info.super_options if "=" not in option
    )
    keyed_option_pairs = tuple(
        option.split("=", 1) for option in mount_info.super_options if "=" in option
    )
    keyed_options = dict(keyed_option_pairs)
    if (
        mount_info.mount_options != _MOUNT_OPTIONS
        or any(
            _OPTIONAL_MOUNT_FIELD_PATTERN.fullmatch(field) is None
            for field in mount_info.optional_fields
        )
        or len(mount_info.optional_fields) > 1
        or mount_info.filesystem_type != _TMPFS_FILESYSTEM_TYPE
        or mount_info.source != _TMPFS_FILESYSTEM_TYPE
        or flag_options != _SUPER_OPTION_FLAGS
        or len(keyed_options) != len(keyed_option_pairs)
        or tuple(sorted(keyed_options)) != _SUPER_OPTION_KEYS
        or _parse_size_option(keyed_options["size"]) != authority.size_limit_bytes
        or keyed_options["nr_inodes"] != str(authority.inode_limit)
        or keyed_options["mode"] != f"{authority.root_mode:o}"
        or keyed_options["uid"] != str(authority.owner_user_id)
        or keyed_options["gid"] != str(authority.owner_group_id)
        or mount_info.device_major != os.major(metadata.st_dev)
        or mount_info.device_minor != os.minor(metadata.st_dev)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != authority.owner_user_id
        or metadata.st_gid != authority.owner_group_id
        or stat.S_IMODE(metadata.st_mode) != authority.root_mode
        or metadata.st_nlink < 2
    ):
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume mount differs from issued tmpfs authority"
        )


def _parse_size_option(value: str) -> int:
    match = _SIZE_OPTION_PATTERN.fullmatch(value)
    if match is None:
        raise RunActionRuntimeVolumeError(
            "keeper runtime-volume size option is malformed"
        )
    return int(match.group(1)) * _SIZE_MULTIPLIERS[match.group(2)]


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_ino,
        metadata.st_dev,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_filesystem(filesystem: os.statvfs_result) -> tuple[int, ...]:
    return (
        filesystem.f_bsize,
        filesystem.f_frsize,
        filesystem.f_blocks,
        filesystem.f_bfree,
        filesystem.f_bavail,
        filesystem.f_files,
        filesystem.f_ffree,
        filesystem.f_favail,
        filesystem.f_flag,
        filesystem.f_namemax,
    )


__all__ = [
    "DockerRunActionEmptyVolumeObservation",
    "DockerRunActionPreparedVolumeObservation",
    "RunActionRuntimeVolumeError",
    "issue_fresh_runtime_volume_authority",
    "materialize_runtime_volume_layout",
    "observe_empty_runtime_volume",
    "reobserve_runtime_volume_layout",
]
