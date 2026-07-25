"""Descriptor-bound evidence for the shared run-action supervisor helper."""

from __future__ import annotations

import os
import re
import stat
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionExecutionPolicy,
    RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
    RunActionDockerInitSourceEvidence,
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
_PROCESS_DESCRIPTOR_FILE_TYPES = {
    "exe": "regular",
    "ns/mnt": "regular",
    "ns/pid": "regular",
    "root": "directory",
}
_PROCESS_NAMESPACE_NAMES = ("mnt", "pid")


class RunActionSupervisorHelperError(ValueError):
    """The supervisor helper differs from its immutable static-code authority."""


@dataclass(frozen=True)
class RunActionProcessStatObservation:
    """One exact live Linux process generation read from proc stat."""

    process_id: int
    state: str
    parent_process_id: int
    start_time_ticks: int

    def __post_init__(self) -> None:
        if (
            type(self.process_id) is not int
            or self.process_id <= 0
            or type(self.state) is not str
            or self.state not in _LIVE_PROCESS_STATES
            or type(self.parent_process_id) is not int
            or self.parent_process_id < 0
            or type(self.start_time_ticks) is not int
            or self.start_time_ticks <= 0
        ):
            raise RunActionSupervisorHelperError(
                "run-action process stat observation is malformed"
            )


@dataclass(frozen=True)
class RunActionProcessDescriptorMetadata:
    """Stable physical identity of one descriptor opened through proc."""

    descriptor_name: str
    file_type: str
    mount_id: int
    device: int
    inode: int
    mode: int
    owner_user_id: int
    owner_group_id: int
    link_count: int
    size: int

    def __post_init__(self) -> None:
        if (
            type(self.descriptor_name) is not str
            or self.descriptor_name not in _PROCESS_DESCRIPTOR_FILE_TYPES
            or type(self.file_type) is not str
            or self.file_type != _PROCESS_DESCRIPTOR_FILE_TYPES[self.descriptor_name]
            or type(self.mount_id) is not int
            or self.mount_id <= 0
            or type(self.device) is not int
            or self.device < 0
            or type(self.inode) is not int
            or self.inode <= 0
            or type(self.mode) is not int
            or self.mode < 0
            or self.mode > 0o7777
            or type(self.owner_user_id) is not int
            or self.owner_user_id < 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id < 0
            or type(self.link_count) is not int
            or self.link_count <= 0
            or type(self.size) is not int
            or self.size < 0
        ):
            raise RunActionSupervisorHelperError(
                "run-action process descriptor metadata is malformed"
            )


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
    metadata, mount_id = _observe_static_executable_path(
        path,
        policy.supervisor_helper_executable_digest,
        "supervisor helper",
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


def observe_docker_init_source(
    policy: DockerRunActionExecutionPolicy,
) -> RunActionDockerInitSourceEvidence:
    """Read and prove the configured host Docker-init source executable."""

    if type(policy) is not DockerRunActionExecutionPolicy:
        raise RunActionSupervisorHelperError(
            "Docker init observation requires an exact execution policy"
        )
    path = Path(policy.docker_init_source_path)
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or path.resolve() != path
    ):
        raise RunActionSupervisorHelperError(
            "Docker init path must be canonical and absolute"
        )
    metadata, mount_id = _observe_static_executable_path(
        path,
        policy.docker_init_executable_digest,
        "Docker init executable",
    )
    return RunActionDockerInitSourceEvidence.mint(
        init_authority_id=policy.docker_init_executable_authority_id,
        source_path=policy.docker_init_source_path,
        file_type="regular",
        owner_user_id=metadata.st_uid,
        owner_group_id=metadata.st_gid,
        mode=stat.S_IMODE(metadata.st_mode),
        link_count=metadata.st_nlink,
        file_format="elf",
        dynamic_dependency_count=0,
        elf_interpreter_present=False,
        executable_digest=policy.docker_init_executable_digest,
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
        process_stat_before = read_run_action_process_stat_from_descriptor(
            process_descriptor,
            process_id,
        )
        process_cgroup_path_before = (
            read_run_action_process_cgroup_path_from_descriptor(
                process_descriptor,
                container_id,
            )
        )
        process_root_descriptor, _ = open_run_action_process_root_descriptor(
            descriptors,
            process_descriptor,
        )
        mounted_descriptor = os.open(
            RUN_ACTION_SUPERVISOR_HELPER_DESTINATION.removeprefix("/"),
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=process_root_descriptor,
        )
        metadata, mount_id = _observe_static_executable_descriptor(
            mounted_descriptor,
            source_evidence.executable_digest,
            "mounted supervisor helper",
        )
        process_cgroup_path_after = read_run_action_process_cgroup_path_from_descriptor(
            process_descriptor,
            container_id,
        )
        process_stat_after = read_run_action_process_stat_from_descriptor(
            process_descriptor,
            process_id,
        )
    if (
        process_stat_after.process_id != process_stat_before.process_id
        or process_stat_after.parent_process_id != process_stat_before.parent_process_id
        or process_stat_after.start_time_ticks != process_stat_before.start_time_ticks
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
        process_start_time_ticks=process_stat_before.start_time_ticks,
        process_cgroup_path=process_cgroup_path_before,
        destination=RUN_ACTION_SUPERVISOR_HELPER_DESTINATION,
        mount_id=mount_id,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        executable_digest=source_evidence.executable_digest,
    )


def _observe_static_executable_path(
    path: Path,
    expected_executable_digest: str,
    description: str,
) -> tuple[os.stat_result, int]:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    return _observe_static_executable_descriptor(
        descriptor,
        expected_executable_digest,
        description,
    )


def _observe_static_executable_descriptor(
    descriptor: int,
    expected_executable_digest: str,
    description: str,
) -> tuple[os.stat_result, int]:
    with os.fdopen(descriptor, "rb") as handle:
        metadata_before = os.fstat(handle.fileno())
        mount_id_before = read_run_action_descriptor_mount_id(handle.fileno())
        _require_static_executable_metadata(metadata_before, description)
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
            f"{description} changed while proving its content"
        )
    _require_static_elf(payload, description)
    return metadata_before, mount_id_before


def read_run_action_process_stat_from_descriptor(
    process_descriptor: int,
    process_id: int,
) -> RunActionProcessStatObservation:
    """Read one live Linux process observation through an open proc directory."""

    if (
        type(process_descriptor) is not int
        or process_descriptor < 0
        or type(process_id) is not int
        or process_id <= 0
    ):
        raise RunActionSupervisorHelperError("run-action process identity is malformed")
    descriptor = os.open(
        "stat",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read()
    return _parse_run_action_process_stat(payload, process_id)


def _parse_run_action_process_stat(
    payload: bytes,
    process_id: int,
) -> RunActionProcessStatObservation:
    if type(payload) is not bytes or type(process_id) is not int or process_id <= 0:
        raise RunActionSupervisorHelperError(
            "run-action process stat identity is malformed"
        )
    prefix = f"{process_id} (".encode("ascii")
    command_end = payload.rfind(b") ")
    if (
        not payload.endswith(b"\n")
        or b"\x00" in payload
        or b"\n" in payload[:-1]
        or b"\r" in payload
        or not payload.startswith(prefix)
        or command_end < len(prefix)
        or not payload.isascii()
    ):
        raise RunActionSupervisorHelperError(
            "run-action process stat identity is malformed"
        )
    fields = payload[command_end + len(b") ") : -len(b"\n")].split(b" ")
    if (
        len(fields) < 20
        or any(not field for field in fields)
        or len(fields[0]) != 1
        or fields[0].decode("ascii") not in _LIVE_PROCESS_STATES
        or not fields[1].isdigit()
        or not fields[19].isdigit()
        or int(fields[19]) <= 0
    ):
        raise RunActionSupervisorHelperError(
            "run-action process is not one live process generation"
        )
    return RunActionProcessStatObservation(
        process_id=process_id,
        state=fields[0].decode("ascii"),
        parent_process_id=int(fields[1]),
        start_time_ticks=int(fields[19]),
    )


def read_run_action_process_command_line_from_descriptor(
    process_descriptor: int,
) -> tuple[bytes, ...]:
    """Read the complete byte-exact argv through an open proc directory."""

    if type(process_descriptor) is not int or process_descriptor < 0:
        raise RunActionSupervisorHelperError(
            "run-action process command line requires an exact process descriptor"
        )
    descriptor = os.open(
        "cmdline",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read()
    return _parse_run_action_process_command_line(payload)


def _parse_run_action_process_command_line(
    payload: bytes,
) -> tuple[bytes, ...]:
    if type(payload) is not bytes or not payload or not payload.endswith(b"\x00"):
        raise RunActionSupervisorHelperError(
            "run-action process command line is not exact NUL-separated argv"
        )
    return tuple(payload[:-1].split(b"\x00"))


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
            "run-action cgroup read requires exact process and container identities"
        )
    descriptor = os.open(
        "cgroup",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read()
    return _parse_run_action_process_cgroup_path(payload, container_id)


def open_run_action_process_root_descriptor(
    descriptors: ExitStack,
    process_descriptor: int,
) -> tuple[int, RunActionProcessDescriptorMetadata]:
    """Open and identify the process root relative to its proc descriptor."""

    _require_process_descriptor_open_arguments(descriptors, process_descriptor)
    descriptor = os.open(
        "root",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    return descriptor, _observe_run_action_process_descriptor_metadata(
        descriptor,
        "root",
    )


def open_run_action_process_executable_descriptor(
    descriptors: ExitStack,
    process_descriptor: int,
) -> tuple[int, RunActionProcessDescriptorMetadata]:
    """Open and identify the process executable relative to its proc descriptor."""

    _require_process_descriptor_open_arguments(descriptors, process_descriptor)
    descriptor = os.open(
        "exe",
        os.O_RDONLY | os.O_CLOEXEC,
        dir_fd=process_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    return descriptor, _observe_run_action_process_descriptor_metadata(
        descriptor,
        "exe",
    )


def open_run_action_process_namespace_descriptor(
    descriptors: ExitStack,
    process_descriptor: int,
    namespace_name: str,
) -> tuple[int, RunActionProcessDescriptorMetadata]:
    """Open and identify one admitted namespace relative to a proc descriptor."""

    _require_process_descriptor_open_arguments(descriptors, process_descriptor)
    if (
        type(namespace_name) is not str
        or namespace_name not in _PROCESS_NAMESPACE_NAMES
    ):
        raise RunActionSupervisorHelperError(
            "run-action process namespace name is not admitted"
        )
    with ExitStack() as namespace_descriptors:
        namespace_directory_descriptor = os.open(
            "ns",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=process_descriptor,
        )
        namespace_descriptors.callback(os.close, namespace_directory_descriptor)
        descriptor = os.open(
            namespace_name,
            os.O_RDONLY | os.O_CLOEXEC,
            dir_fd=namespace_directory_descriptor,
        )
    descriptors.callback(os.close, descriptor)
    descriptor_name = f"ns/{namespace_name}"
    return descriptor, _observe_run_action_process_descriptor_metadata(
        descriptor,
        descriptor_name,
    )


def _require_process_descriptor_open_arguments(
    descriptors: ExitStack,
    process_descriptor: int,
) -> None:
    if (
        type(descriptors) is not ExitStack
        or type(process_descriptor) is not int
        or process_descriptor < 0
    ):
        raise RunActionSupervisorHelperError(
            "run-action process resource open requires exact descriptors"
        )


def _observe_run_action_process_descriptor_metadata(
    descriptor: int,
    descriptor_name: str,
) -> RunActionProcessDescriptorMetadata:
    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(descriptor_name) is not str
        or descriptor_name not in _PROCESS_DESCRIPTOR_FILE_TYPES
    ):
        raise RunActionSupervisorHelperError(
            "run-action process descriptor observation is malformed"
        )
    metadata_before = os.fstat(descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(descriptor)
    metadata_after = os.fstat(descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(descriptor)
    expected_file_type = _PROCESS_DESCRIPTOR_FILE_TYPES[descriptor_name]
    file_type = (
        "directory"
        if stat.S_ISDIR(metadata_before.st_mode)
        else "regular" if stat.S_ISREG(metadata_before.st_mode) else "unsupported"
    )
    if (
        _stable_metadata(metadata_before) != _stable_metadata(metadata_after)
        or mount_id_before != mount_id_after
        or file_type != expected_file_type
    ):
        raise RunActionSupervisorHelperError(
            "run-action process descriptor metadata changed or has the wrong type"
        )
    return RunActionProcessDescriptorMetadata(
        descriptor_name=descriptor_name,
        file_type=file_type,
        mount_id=mount_id_before,
        device=metadata_before.st_dev,
        inode=metadata_before.st_ino,
        mode=stat.S_IMODE(metadata_before.st_mode),
        owner_user_id=metadata_before.st_uid,
        owner_group_id=metadata_before.st_gid,
        link_count=metadata_before.st_nlink,
        size=metadata_before.st_size,
    )


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
            "run-action cgroup parse requires exact payload and container identities"
        )
    lines = payload.split(b"\n")
    if (
        len(lines) != 2
        or lines[1] != b""
        or not payload.endswith(b"\n")
        or not lines[0].startswith(b"0::")
        or len(lines[0]) <= len(b"0::")
        or b"\x00" in lines[0]
        or b"\r" in lines[0]
    ):
        raise RunActionSupervisorHelperError(
            "run-action process lacks one unified cgroup identity"
        )
    encoded_path = lines[0][len(b"0::") :]
    if not encoded_path.isascii():
        raise RunActionSupervisorHelperError(
            "run-action process cgroup identity is not ASCII"
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
            "run-action process cgroup is not bound to the inspected container"
        )
    return process_cgroup_path


def _require_static_executable_metadata(
    metadata: os.stat_result,
    description: str,
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o755
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
    ):
        raise RunActionSupervisorHelperError(
            f"{description} is not immutable root-owned executable code"
        )


def read_run_action_descriptor_mount_id(descriptor: int) -> int:
    """Read the kernel mount identity for one already-open descriptor."""

    if type(descriptor) is not int or descriptor < 0:
        raise RunActionSupervisorHelperError(
            "run-action descriptor mount identity requires an exact descriptor"
        )
    with ExitStack() as descriptors:
        proc_descriptor = os.open(
            "/proc",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, proc_descriptor)
        current_process_descriptor = os.open(
            str(os.getpid()),
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=proc_descriptor,
        )
        descriptors.callback(os.close, current_process_descriptor)
        fdinfo_directory_descriptor = os.open(
            "fdinfo",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=current_process_descriptor,
        )
        descriptors.callback(os.close, fdinfo_directory_descriptor)
        fdinfo_descriptor = os.open(
            str(descriptor),
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=fdinfo_directory_descriptor,
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


def _require_static_elf(payload: bytes, description: str) -> None:
    if (
        not isinstance(payload, bytes)
        or len(payload) < _ELF_IDENT_SIZE
        or payload[:4] != _ELF_MAGIC
        or payload[6] != _ELF_CURRENT_VERSION
    ):
        raise RunActionSupervisorHelperError(
            f"{description} is not a supported ELF executable"
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
            f"{description} uses an unsupported ELF encoding"
        )
    header_format, program_header_size = layout
    encoded_header_size = struct.calcsize(byte_order + header_format)
    header_size = _ELF_IDENT_SIZE + encoded_header_size
    if len(payload) < header_size:
        raise RunActionSupervisorHelperError(f"{description} ELF header is truncated")
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
            f"{description} ELF program table is malformed"
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
            f"{description} carries a dynamic loader or dependency table"
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
    "RunActionProcessDescriptorMetadata",
    "RunActionProcessStatObservation",
    "RunActionSupervisorHelperError",
    "observe_docker_init_source",
    "observe_supervisor_helper",
    "observe_mounted_keeper_helper",
    "open_run_action_process_executable_descriptor",
    "open_run_action_process_namespace_descriptor",
    "open_run_action_process_root_descriptor",
    "read_run_action_descriptor_mount_id",
    "read_run_action_process_cgroup_path_from_descriptor",
    "read_run_action_process_command_line_from_descriptor",
    "read_run_action_process_stat_from_descriptor",
]
