"""Descriptor-bound preparation and observation of one runtime-volume generation."""

from __future__ import annotations

import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath

from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionVolumeObservation,
)
from kapso.cross_run.launch.run_action_keeper_helper import (
    read_run_action_descriptor_mount_id,
    read_run_action_process_cgroup_path_from_descriptor,
    read_run_action_process_start_time_from_descriptor,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    RunActionPreparationClaim,
    RunActionRuntimeVolumeAuthority,
    RunActionVolumeKeeperEvidence,
    issue_runtime_volume_authority,
    run_action_keeper_process_cgroup_path,
)

_TMPFS_FILESYSTEM_TYPE = "tmpfs"
_MOUNT_OPTIONS = ("nodev", "nosuid", "relatime", "rw")
_SUPER_OPTION_FLAGS = ("inode64", "noswap", "rw")
_SUPER_OPTION_KEYS = ("gid", "mode", "nr_inodes", "size", "uid")
_OPTIONAL_MOUNT_FIELD_PATTERN = re.compile(r"^master:[1-9][0-9]*$")
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SIZE_OPTION_PATTERN = re.compile(r"^([1-9][0-9]*)([kmgt]?)$")
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
    process_id = keeper.process_id
    container_id = keeper.container_id
    with ExitStack() as descriptors:
        process_descriptor = os.open(
            f"/proc/{process_id}",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, process_descriptor)
        process_start_time_before = read_run_action_process_start_time_from_descriptor(
            process_descriptor,
            process_id,
        )
        process_cgroup_path_before = (
            read_run_action_process_cgroup_path_from_descriptor(
                process_descriptor,
                container_id,
            )
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
        metadata_before = os.fstat(root_descriptor)
        mount_id = read_run_action_descriptor_mount_id(root_descriptor)
        mount_info_before = _read_mount_info(
            process_descriptor,
            mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        filesystem_before = os.fstatvfs(root_descriptor)
        entries = tuple(sorted(os.listdir(root_descriptor)))
        filesystem_after = os.fstatvfs(root_descriptor)
        mount_info_after = _read_mount_info(
            process_descriptor,
            mount_id,
            RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
        )
        metadata_after = os.fstat(root_descriptor)
        process_cgroup_path_after = read_run_action_process_cgroup_path_from_descriptor(
            process_descriptor,
            container_id,
        )
        process_start_time_after = read_run_action_process_start_time_from_descriptor(
            process_descriptor,
            process_id,
        )
    if (
        process_start_time_after != process_start_time_before
        or process_start_time_before != keeper.process_start_time_ticks
        or process_cgroup_path_after != process_cgroup_path_before
        or process_cgroup_path_before
        != run_action_keeper_process_cgroup_path(
            keeper.issued_create_projection.execution_policy,
            container_id,
        )
        or _stable_metadata(metadata_after) != _stable_metadata(metadata_before)
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
        keeper_container_id=container_id,
        keeper_process_id=process_id,
        keeper_process_start_time_ticks=process_start_time_before,
        process_cgroup_path=process_cgroup_path_before,
        mount_id=mount_id,
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
    "RunActionRuntimeVolumeError",
    "issue_fresh_runtime_volume_authority",
    "observe_empty_runtime_volume",
]
