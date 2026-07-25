"""Descriptor-bound evidence for the shared run-action supervisor helper."""

from __future__ import annotations

import os
import re
import stat
import struct
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionExecutionPolicy,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionSupervisorHelperEvidence,
    RunActionMountedKeeperHelperEvidence,
    RunActionPreparedMountAccess,
)

_ELF_MAGIC = b"\x7fELF"
_ELF_CLASS_32 = 1
_ELF_CLASS_64 = 2
_ELF_ENDIAN_LITTLE = 1
_ELF_ENDIAN_BIG = 2
_ELF_CURRENT_VERSION = 1
_ELF_PROGRAM_DYNAMIC = 2
_ELF_PROGRAM_INTERPRETER = 3
_ELF_32_HEADER_FORMAT = "HHIIIIIHHHHHH"
_ELF_64_HEADER_FORMAT = "HHIQQQIHHHHHH"
_ELF_32_PROGRAM_HEADER_SIZE = 32
_ELF_64_PROGRAM_HEADER_SIZE = 56
_ELF_IDENT_SIZE = 16
_FDINFO_MOUNT_ID_PREFIX = "mnt_id:\t"
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LIVE_PROCESS_STATES = ("D", "I", "P", "R", "S", "T", "t")


class RunActionSupervisorHelperError(ValueError):
    """The supervisor helper differs from its immutable static-code authority."""


def observe_supervisor_helper(
    policy: DockerRunActionExecutionPolicy,
) -> RunActionSupervisorHelperEvidence:
    """Read and prove one root-owned, singly-linked, static ELF executable."""

    if type(policy) is not DockerRunActionExecutionPolicy:
        raise RunActionSupervisorHelperError(
            "supervisor helper observation requires an exact execution policy"
        )
    path = Path(policy.supervisor_helper_source_path)
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise RunActionSupervisorHelperError(
            "supervisor helper path must be canonical and absolute"
        )
    metadata, mount_id = _observe_helper_path(
        path,
        policy.supervisor_helper_executable_digest,
    )
    return RunActionSupervisorHelperEvidence.mint(
        helper_authority_id=policy.supervisor_helper_executable_authority_id,
        source_path=policy.supervisor_helper_source_path,
        destination=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        mount_type="bind",
        mount_access=RunActionPreparedMountAccess.READ_ONLY,
        recursive_bind=False,
        file_type="regular",
        owner_user_id=metadata.st_uid,
        owner_group_id=metadata.st_gid,
        mode=stat.S_IMODE(metadata.st_mode),
        link_count=metadata.st_nlink,
        file_format="elf",
        dynamic_dependency_count=0,
        elf_interpreter_present=False,
        executable_digest=policy.supervisor_helper_executable_digest,
        mount_id=mount_id,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def observe_mounted_keeper_helper(
    source_evidence: RunActionSupervisorHelperEvidence,
    *,
    container_id: str,
    process_id: int,
) -> RunActionMountedKeeperHelperEvidence:
    """Prove the exact issued helper inode is mounted in the keeper process."""

    if (
        type(source_evidence) is not RunActionSupervisorHelperEvidence
        or type(container_id) is not str
        or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
        or type(process_id) is not int
        or process_id <= 0
    ):
        raise RunActionSupervisorHelperError(
            "mounted supervisor helper requires exact source and process identities"
        )
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
        mounted_descriptor = os.open(
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION.removeprefix("/"),
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=process_root_descriptor,
        )
        metadata, mount_id = _observe_helper_descriptor(
            mounted_descriptor,
            source_evidence.executable_digest,
        )
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
        or process_cgroup_path_after != process_cgroup_path_before
        or metadata.st_dev != source_evidence.device
        or metadata.st_ino != source_evidence.inode
        or mount_id == source_evidence.mount_id
    ):
        raise RunActionSupervisorHelperError(
            "mounted supervisor helper differs from its issued source inode"
        )
    return RunActionMountedKeeperHelperEvidence.mint(
        source_helper_evidence=source_evidence,
        container_id=container_id,
        process_id=process_id,
        process_start_time_ticks=process_start_time_before,
        process_cgroup_path=process_cgroup_path_before,
        destination=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        mount_id=mount_id,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        executable_digest=source_evidence.executable_digest,
    )


def _observe_helper_path(
    path: Path,
    expected_executable_digest: str,
) -> tuple[os.stat_result, int]:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    return _observe_helper_descriptor(descriptor, expected_executable_digest)


def _observe_helper_descriptor(
    descriptor: int,
    expected_executable_digest: str,
) -> tuple[os.stat_result, int]:
    with os.fdopen(descriptor, "rb") as handle:
        metadata_before = os.fstat(handle.fileno())
        mount_id_before = read_run_action_descriptor_mount_id(handle.fileno())
        _require_helper_metadata(metadata_before)
        payload = handle.read()
        metadata_after = os.fstat(handle.fileno())
        mount_id_after = read_run_action_descriptor_mount_id(handle.fileno())
    if (
        _stable_metadata(metadata_before) != _stable_metadata(metadata_after)
        or mount_id_before != mount_id_after
        or len(payload) != metadata_before.st_size
        or tree_or_blob_digest(payload) != expected_executable_digest
    ):
        raise RunActionSupervisorHelperError(
            "supervisor helper changed while proving its content"
        )
    _require_static_elf(payload)
    return metadata_before, mount_id_before


def read_run_action_process_start_time_from_descriptor(
    process_descriptor: int,
    process_id: int,
) -> int:
    """Read the Linux process-generation token through an open proc directory."""

    if (
        type(process_descriptor) is not int
        or process_descriptor < 0
        or type(process_id) is not int
        or process_id <= 0
    ):
        raise RunActionSupervisorHelperError("keeper process identity is malformed")
    descriptor = os.open(
        "stat",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read()
    prefix = f"{process_id} (".encode("ascii")
    command_end = payload.rfind(b") ")
    if (
        not payload.endswith(b"\n")
        or b"\x00" in payload
        or not payload.startswith(prefix)
        or command_end < len(prefix)
        or not payload.isascii()
    ):
        raise RunActionSupervisorHelperError("keeper process identity is malformed")
    fields = payload[command_end + len(b") ") :].decode("ascii").split()
    if (
        len(fields) < 20
        or fields[0] not in _LIVE_PROCESS_STATES
        or not fields[19].isdigit()
        or int(fields[19]) <= 0
    ):
        raise RunActionSupervisorHelperError(
            "keeper is not one live process generation"
        )
    return int(fields[19])


def read_run_action_process_cgroup_path_from_descriptor(
    process_descriptor: int,
    container_id: str,
) -> str:
    """Read one container cgroup through an already-open proc process directory."""

    if (
        type(process_descriptor) is not int
        or process_descriptor < 0
        or type(container_id) is not str
        or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
    ):
        raise RunActionSupervisorHelperError(
            "keeper cgroup read requires exact process and container identities"
        )
    descriptor = os.open(
        "cgroup",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read()
    return _parse_run_action_process_cgroup_path(payload, container_id)


def _parse_run_action_process_cgroup_path(
    payload: bytes,
    container_id: str,
) -> str:
    if (
        type(payload) is not bytes
        or type(container_id) is not str
        or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None
    ):
        raise RunActionSupervisorHelperError(
            "keeper cgroup parse requires exact payload and container identities"
        )
    lines = payload.splitlines()
    if (
        len(lines) != 1
        or not payload.endswith(b"\n")
        or not lines[0].startswith(b"0::")
        or len(lines[0]) <= len(b"0::")
        or b"\x00" in lines[0]
    ):
        raise RunActionSupervisorHelperError(
            "keeper process lacks one unified cgroup identity"
        )
    encoded_path = lines[0][len(b"0::") :]
    if not encoded_path.isascii():
        raise RunActionSupervisorHelperError(
            "keeper process cgroup identity is not ASCII"
        )
    process_cgroup_path = encoded_path.decode("ascii")
    parsed_path = PurePosixPath(process_cgroup_path)
    if (
        not parsed_path.is_absolute()
        or parsed_path.as_posix() != process_cgroup_path
        or ".." in parsed_path.parts
        or not process_cgroup_path.endswith(f"/docker-{container_id}.scope")
    ):
        raise RunActionSupervisorHelperError(
            "keeper process cgroup is not bound to the inspected container"
        )
    return process_cgroup_path


def _require_helper_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o755
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
    ):
        raise RunActionSupervisorHelperError(
            "supervisor helper is not immutable root-owned executable code"
        )


def read_run_action_descriptor_mount_id(descriptor: int) -> int:
    """Read the kernel mount identity for one already-open descriptor."""

    fdinfo_descriptor = os.open(
        f"/proc/self/fdinfo/{descriptor}",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(fdinfo_descriptor, "rb") as handle:
        payload = handle.read()
    prefix = _FDINFO_MOUNT_ID_PREFIX.encode("ascii")
    values = tuple(
        line.removeprefix(prefix)
        for line in payload.splitlines()
        if line.startswith(prefix)
    )
    if (
        len(values) != 1
        or not values[0]
        or not values[0].isdigit()
        or int(values[0]) <= 0
    ):
        raise RunActionSupervisorHelperError(
            "supervisor helper descriptor lacks one mount identity"
        )
    return int(values[0])


def _require_static_elf(payload: bytes) -> None:
    if (
        not isinstance(payload, bytes)
        or len(payload) < _ELF_IDENT_SIZE
        or payload[:4] != _ELF_MAGIC
        or payload[6] != _ELF_CURRENT_VERSION
    ):
        raise RunActionSupervisorHelperError(
            "supervisor helper is not a supported ELF executable"
        )
    byte_order = {
        _ELF_ENDIAN_LITTLE: "<",
        _ELF_ENDIAN_BIG: ">",
    }.get(payload[5])
    layout = {
        _ELF_CLASS_32: (
            _ELF_32_HEADER_FORMAT,
            _ELF_32_PROGRAM_HEADER_SIZE,
        ),
        _ELF_CLASS_64: (
            _ELF_64_HEADER_FORMAT,
            _ELF_64_PROGRAM_HEADER_SIZE,
        ),
    }.get(payload[4])
    if byte_order is None or layout is None:
        raise RunActionSupervisorHelperError(
            "supervisor helper uses an unsupported ELF encoding"
        )
    header_format, program_header_size = layout
    encoded_header_size = struct.calcsize(byte_order + header_format)
    header_size = _ELF_IDENT_SIZE + encoded_header_size
    if len(payload) < header_size:
        raise RunActionSupervisorHelperError(
            "supervisor helper ELF header is truncated"
        )
    header = struct.unpack(
        byte_order + header_format,
        payload[_ELF_IDENT_SIZE:header_size],
    )
    program_header_offset = header[4]
    encoded_header_size_field = header[7]
    program_header_entry_size = header[8]
    program_header_count = header[9]
    program_table_end = (
        program_header_offset + program_header_entry_size * program_header_count
    )
    if (
        encoded_header_size_field != header_size
        or program_header_entry_size != program_header_size
        or program_header_count <= 0
        or program_header_offset < header_size
        or program_table_end > len(payload)
    ):
        raise RunActionSupervisorHelperError(
            "supervisor helper ELF program table is malformed"
        )
    program_types = tuple(
        struct.unpack(
            byte_order + "I",
            payload[
                program_header_offset
                + position * program_header_entry_size : program_header_offset
                + position * program_header_entry_size
                + 4
            ],
        )[0]
        for position in range(program_header_count)
    )
    if (
        _ELF_PROGRAM_DYNAMIC in program_types
        or _ELF_PROGRAM_INTERPRETER in program_types
    ):
        raise RunActionSupervisorHelperError(
            "supervisor helper carries a dynamic loader or dependency table"
        )


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


__all__ = [
    "RunActionSupervisorHelperError",
    "observe_supervisor_helper",
    "observe_mounted_keeper_helper",
    "read_run_action_descriptor_mount_id",
    "read_run_action_process_cgroup_path_from_descriptor",
    "read_run_action_process_start_time_from_descriptor",
]
