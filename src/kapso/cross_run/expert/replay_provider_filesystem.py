"""Safe host inputs and result snapshots for an isolated replay provider."""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayMatchedLegInvocation,
)
from kapso.cross_run.expert.replay_request import (
    VerifiedExpertSourceReplayCandidate,
    VerifiedExpertSourceReplayParent,
)

_INPUT_DIRECTORY_NAME = "input"
_EXPERT_DIRECTORY_NAME = "expert"
_ADAPTER_DIRECTORY_NAME = "adapter"
_TASK_DIRECTORY_NAME = "task"
_REQUEST_FILENAME = "request.json"
_RESULT_ARCHIVE_ROOT_NAME = "./"
_RESULT_ARCHIVE_PATH = "./result.json"
_TAR_BLOCK_SIZE = 512
_TAR_END_BLOCK_COUNT = 2
_TAR_ZERO_BLOCK = bytes(_TAR_BLOCK_SIZE)
_BUSYBOX_USTAR_SIGNATURE = b"ustar  \x00"


class SourceReplayProviderFilesystemError(ValueError):
    """A provider input tree or result snapshot violates its exact boundary."""


@dataclass(frozen=True)
class SourceReplayProviderInputLayout:
    """One immutable host input closure ready for container bind mounts."""

    trusted_root: Path
    workspace_root: Path

    def __post_init__(self) -> None:
        _require_direct_child(
            self.trusted_root, self.workspace_root, "provider workspace"
        )

    @property
    def input_root(self) -> Path:
        return self.workspace_root / _INPUT_DIRECTORY_NAME

    @property
    def expert_root(self) -> Path:
        return self.input_root / _EXPERT_DIRECTORY_NAME

    @property
    def adapter_root(self) -> Path:
        return self.input_root / _ADAPTER_DIRECTORY_NAME

    @property
    def task_root(self) -> Path:
        return self.input_root / _TASK_DIRECTORY_NAME

    @property
    def request_path(self) -> Path:
        return self.input_root / _REQUEST_FILENAME


def materialize_verified_byte_tree(
    *,
    trusted_root: Path,
    destination_root: Path,
    descriptors: tuple[SourceFileDescriptor, ...],
    source_contents: Mapping[str, bytes],
) -> Path:
    """Create one exact source tree as a new direct child of a trusted root."""

    _require_verified_byte_closure(descriptors, source_contents)
    _require_direct_child(trusted_root, destination_root, "byte-tree destination")
    with ExitStack() as descriptors_to_close:
        trusted_descriptor = _open_trusted_root(trusted_root, descriptors_to_close)
        destination_descriptor = _create_directory_at(
            trusted_descriptor,
            destination_root.name,
            descriptors_to_close,
        )
        _materialize_verified_byte_tree_at(
            destination_descriptor,
            descriptors,
            source_contents,
        )
        os.fsync(destination_descriptor)
    return destination_root


def materialize_source_replay_provider_inputs(
    *,
    invocation: ExpertSourceReplayMatchedLegInvocation,
    trusted_root: Path,
    workspace_root: Path,
) -> SourceReplayProviderInputLayout:
    """Materialize and freeze one matched leg's complete evaluator input closure."""

    if type(invocation) is not ExpertSourceReplayMatchedLegInvocation:
        raise SourceReplayProviderFilesystemError(
            "provider inputs require an exact matched-leg invocation"
        )
    _require_direct_child(trusted_root, workspace_root, "provider workspace")
    expert_descriptors, expert_contents = _expert_source_closure(invocation)
    adapter = invocation.materialized_case.task_adapter
    adapter_descriptors = adapter.evaluation_runtime_source_files
    adapter_contents = adapter.evaluation_runtime_source_contents
    _require_verified_byte_closure(expert_descriptors, expert_contents)
    _require_verified_byte_closure(adapter_descriptors, adapter_contents)
    for artifact in invocation.materialized_case.task_context.starting_artifacts:
        _require_verified_byte_closure(
            artifact.artifact.source_files,
            artifact.source_contents,
        )

    with ExitStack() as descriptors_to_close:
        trusted_descriptor = _open_trusted_root(trusted_root, descriptors_to_close)
        workspace_descriptor = _create_directory_at(
            trusted_descriptor,
            workspace_root.name,
            descriptors_to_close,
        )
        input_descriptor = _create_directory_at(
            workspace_descriptor,
            _INPUT_DIRECTORY_NAME,
            descriptors_to_close,
        )
        expert_descriptor = _create_directory_at(
            input_descriptor,
            _EXPERT_DIRECTORY_NAME,
            descriptors_to_close,
        )
        adapter_descriptor = _create_directory_at(
            input_descriptor,
            _ADAPTER_DIRECTORY_NAME,
            descriptors_to_close,
        )
        task_descriptor = _create_directory_at(
            input_descriptor,
            _TASK_DIRECTORY_NAME,
            descriptors_to_close,
        )
        _materialize_verified_byte_tree_at(
            expert_descriptor,
            expert_descriptors,
            expert_contents,
        )
        _materialize_verified_byte_tree_at(
            adapter_descriptor,
            adapter_descriptors,
            adapter_contents,
        )
        _materialize_task_artifacts_at(invocation, task_descriptor)
        _write_new_regular_file_at(
            input_descriptor,
            _REQUEST_FILENAME,
            invocation.task_evaluator_request.to_json_bytes(),
            0o600,
        )
        _freeze_input_directory(input_descriptor)
        os.fsync(workspace_descriptor)
    return SourceReplayProviderInputLayout(
        trusted_root=trusted_root,
        workspace_root=workspace_root,
    )


def cleanup_source_replay_provider_workspace(
    *,
    trusted_root: Path,
    workspace_root: Path,
) -> None:
    """Remove one owned provider workspace without following filesystem links."""

    _require_direct_child(trusted_root, workspace_root, "provider workspace")
    with ExitStack() as descriptors_to_close:
        trusted_descriptor = _open_trusted_root(trusted_root, descriptors_to_close)
        if not os.access(
            workspace_root.name,
            os.F_OK,
            dir_fd=trusted_descriptor,
            follow_symlinks=False,
        ):
            return
        workspace_metadata = os.stat(
            workspace_root.name,
            dir_fd=trusted_descriptor,
            follow_symlinks=False,
        )
        _require_cleanup_directory_metadata(
            workspace_metadata,
            "provider workspace must be a real owned directory",
        )
        workspace_identity = (
            workspace_metadata.st_dev,
            workspace_metadata.st_ino,
        )
        workspace_descriptor = _open_cleanup_directory_at(
            trusted_descriptor,
            workspace_root.name,
            workspace_identity,
            descriptors_to_close,
        )
        _validate_cleanup_directory_tree(workspace_descriptor)
        _remove_cleanup_directory_contents(workspace_descriptor)
        current_workspace = os.stat(
            workspace_root.name,
            dir_fd=trusted_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current_workspace.st_mode)
            or (current_workspace.st_dev, current_workspace.st_ino)
            != workspace_identity
        ):
            raise SourceReplayProviderFilesystemError(
                "provider workspace changed before removal"
            )
        os.rmdir(workspace_root.name, dir_fd=trusted_descriptor)
        os.fsync(trusted_descriptor)


def parse_source_replay_result_snapshot(
    snapshot: bytes,
    *,
    expected_owner_id: int,
    expected_group_id: int,
    maximum_result_bytes: int,
    maximum_snapshot_bytes: int,
) -> bytes:
    """Admit the one canonical result from a bounded BusyBox tar snapshot."""

    if not isinstance(snapshot, bytes):
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot must be bytes"
        )
    if (
        type(expected_owner_id) is not int
        or expected_owner_id < 0
        or type(expected_group_id) is not int
        or expected_group_id < 0
    ):
        raise SourceReplayProviderFilesystemError(
            "source replay result owner and group must be non-negative integers"
        )
    for value, name in (
        (maximum_result_bytes, "result byte bound"),
        (maximum_snapshot_bytes, "snapshot byte bound"),
    ):
        if type(value) is not int or value <= 0:
            raise SourceReplayProviderFilesystemError(
                f"source replay {name} must be a positive integer"
            )
    if len(snapshot) > maximum_snapshot_bytes:
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot exceeds its configured bound"
        )
    if len(snapshot) % _TAR_BLOCK_SIZE != 0:
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot is not block-aligned"
        )

    offset = 0
    observed_names: set[str] = set()
    result_payload = None
    while True:
        if offset + _TAR_BLOCK_SIZE > len(snapshot):
            raise SourceReplayProviderFilesystemError(
                "source replay result snapshot has no exact end marker"
            )
        header = snapshot[offset : offset + _TAR_BLOCK_SIZE]
        if header == _TAR_ZERO_BLOCK:
            end_offset = offset + (_TAR_BLOCK_SIZE * _TAR_END_BLOCK_COUNT)
            if (
                end_offset != len(snapshot)
                or snapshot[offset:end_offset] != _TAR_ZERO_BLOCK * _TAR_END_BLOCK_COUNT
            ):
                raise SourceReplayProviderFilesystemError(
                    "source replay result snapshot has ambiguous trailing bytes"
                )
            break
        entry = _parse_busybox_tar_header(header)
        name = entry.name
        if name in observed_names:
            raise SourceReplayProviderFilesystemError(
                "source replay result snapshot contains a duplicate entry"
            )
        observed_names.add(name)
        if name not in {_RESULT_ARCHIVE_ROOT_NAME, _RESULT_ARCHIVE_PATH}:
            raise SourceReplayProviderFilesystemError(
                "source replay result snapshot contains an extra entry"
            )
        if entry.owner_id != expected_owner_id or entry.group_id != expected_group_id:
            raise SourceReplayProviderFilesystemError(
                "source replay result snapshot has an unexpected owner or group"
            )
        if name == _RESULT_ARCHIVE_ROOT_NAME:
            if entry.type_flag != b"5" or entry.size != 0:
                raise SourceReplayProviderFilesystemError(
                    "source replay result snapshot root is not an empty directory"
                )
        else:
            if entry.type_flag != b"0":
                raise SourceReplayProviderFilesystemError(
                    "source replay result is not a regular file"
                )
            if entry.size > maximum_result_bytes:
                raise SourceReplayProviderFilesystemError(
                    "source replay result exceeds its configured byte bound"
                )

        offset += _TAR_BLOCK_SIZE
        padded_size = _tar_padded_size(entry.size)
        if offset + padded_size > len(snapshot):
            raise SourceReplayProviderFilesystemError(
                "source replay result snapshot contains a truncated entry"
            )
        entry_payload = snapshot[offset : offset + entry.size]
        padding = snapshot[offset + entry.size : offset + padded_size]
        if any(padding):
            raise SourceReplayProviderFilesystemError(
                "source replay result snapshot has non-zero entry padding"
            )
        if name == _RESULT_ARCHIVE_PATH:
            result_payload = entry_payload
        offset += padded_size

    if observed_names != {_RESULT_ARCHIVE_ROOT_NAME, _RESULT_ARCHIVE_PATH}:
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot lacks its exact result closure"
        )
    if result_payload is None:
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot lacks result bytes"
        )
    return result_payload


@dataclass(frozen=True)
class _BusyBoxTarEntry:
    name: str
    owner_id: int
    group_id: int
    size: int
    type_flag: bytes


def _parse_busybox_tar_header(header: bytes) -> _BusyBoxTarEntry:
    if len(header) != _TAR_BLOCK_SIZE:
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot header has the wrong size"
        )
    if header[257:265] != _BUSYBOX_USTAR_SIGNATURE:
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot is not the pinned BusyBox tar format"
        )
    type_flag = header[156:157]
    if type_flag not in {b"0", b"5"}:
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot contains a link or special entry"
        )
    if any(header[157:257]) or any(header[329:500]) or any(header[500:512]):
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot uses unsupported tar metadata"
        )
    expected_checksum = _parse_tar_checksum(header[148:156])
    observed_checksum = sum(header[:148]) + sum(b" " * 8) + sum(header[156:])
    if expected_checksum != observed_checksum:
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot header checksum is invalid"
        )
    name = _parse_tar_name(header[0:100])
    # Modes, timestamps, and textual names are never extracted or treated as
    # authority. Numeric ownership, exact paths, types, sizes, and bytes are.
    _parse_tar_octal(header[100:108], "mode")
    owner_id = _parse_tar_octal(header[108:116], "owner")
    group_id = _parse_tar_octal(header[116:124], "group")
    size = _parse_tar_octal(header[124:136], "size")
    _parse_tar_octal(header[136:148], "modification time")
    return _BusyBoxTarEntry(
        name=name,
        owner_id=owner_id,
        group_id=group_id,
        size=size,
        type_flag=type_flag,
    )


def _parse_tar_name(field: bytes) -> str:
    separator = field.find(b"\x00")
    if separator <= 0 or any(field[separator:]):
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot path is not canonical"
        )
    name_bytes = field[:separator]
    if any(byte < 32 or byte > 126 for byte in name_bytes):
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot path is not printable ASCII"
        )
    return name_bytes.decode("ascii")


def _parse_tar_octal(field: bytes, name: str) -> int:
    if (
        not field
        or field[-1:] != b"\x00"
        or any(byte not in b"01234567" for byte in field[:-1])
    ):
        raise SourceReplayProviderFilesystemError(
            f"source replay result snapshot {name} is not canonical octal"
        )
    return int(field[:-1], 8)


def _parse_tar_checksum(field: bytes) -> int:
    if (
        len(field) != 8
        or field[-2:] != b"\x00 "
        or any(byte not in b"01234567" for byte in field[:-2])
    ):
        raise SourceReplayProviderFilesystemError(
            "source replay result snapshot checksum is not canonical octal"
        )
    return int(field[:-2], 8)


def _tar_padded_size(size: int) -> int:
    return ((size + _TAR_BLOCK_SIZE - 1) // _TAR_BLOCK_SIZE) * _TAR_BLOCK_SIZE


def _expert_source_closure(
    invocation: ExpertSourceReplayMatchedLegInvocation,
) -> tuple[tuple[SourceFileDescriptor, ...], Mapping[str, bytes]]:
    source = invocation.expert_source
    if type(source) is VerifiedExpertSourceReplayCandidate:
        return source.source_tree.files, source.source_contents
    if type(source) is VerifiedExpertSourceReplayParent:
        return (
            source.parent_tree_receipt.source_extraction_receipt.source_tree_files,
            source.source_contents,
        )
    raise SourceReplayProviderFilesystemError(
        "provider invocation contains an unverified expert source"
    )


def _materialize_task_artifacts_at(
    invocation: ExpertSourceReplayMatchedLegInvocation,
    task_descriptor: int,
) -> None:
    context = invocation.materialized_case.task_context
    mounts_by_reference = {
        mount.starting_artifact_ref: mount.mount_path
        for mount in invocation.task_evaluator_request.starting_artifact_mounts
    }
    observed_mounts = {
        artifact.artifact.starting_artifact_ref: artifact.artifact.mount_path
        for artifact in context.starting_artifacts
    }
    if mounts_by_reference != observed_mounts:
        raise SourceReplayProviderFilesystemError(
            "provider task artifact mounts differ from the exact request"
        )
    for artifact in context.starting_artifacts:
        with ExitStack() as descriptors_to_close:
            mount_descriptor = _create_relative_directory_at(
                task_descriptor,
                PurePosixPath(artifact.artifact.mount_path).parts,
                descriptors_to_close,
            )
            _materialize_verified_byte_tree_at(
                mount_descriptor,
                artifact.artifact.source_files,
                artifact.source_contents,
            )
            os.fsync(mount_descriptor)
    os.fsync(task_descriptor)


def _require_verified_byte_closure(
    descriptors: tuple[SourceFileDescriptor, ...],
    source_contents: Mapping[str, bytes],
) -> None:
    if not isinstance(descriptors, tuple) or any(
        type(descriptor) is not SourceFileDescriptor for descriptor in descriptors
    ):
        raise SourceReplayProviderFilesystemError(
            "provider byte tree requires typed descriptor tuples"
        )
    paths = tuple(descriptor.relative_path for descriptor in descriptors)
    if paths != tuple(sorted(set(paths))):
        raise SourceReplayProviderFilesystemError(
            "provider byte tree paths must be sorted and unique"
        )
    normalized_paths = tuple(PurePosixPath(path) for path in paths)
    if any(
        left in right.parents or right in left.parents
        for position, left in enumerate(normalized_paths)
        for right in normalized_paths[position + 1 :]
    ):
        raise SourceReplayProviderFilesystemError(
            "provider byte tree contains a file-directory collision"
        )
    if not isinstance(source_contents, Mapping) or set(source_contents) != set(paths):
        raise SourceReplayProviderFilesystemError(
            "provider byte tree differs from its exact path closure"
        )
    for descriptor in descriptors:
        payload = source_contents[descriptor.relative_path]
        if (
            not isinstance(payload, bytes)
            or len(payload) != descriptor.size
            or tree_or_blob_digest(payload) != descriptor.digest
        ):
            raise SourceReplayProviderFilesystemError(
                "provider byte tree differs from its verified descriptors"
            )


def _materialize_verified_byte_tree_at(
    root_descriptor: int,
    descriptors: tuple[SourceFileDescriptor, ...],
    source_contents: Mapping[str, bytes],
) -> None:
    for descriptor in descriptors:
        relative_path = PurePosixPath(descriptor.relative_path)
        with ExitStack() as descriptors_to_close:
            parent_descriptor = _open_or_create_relative_directory_at(
                root_descriptor,
                relative_path.parts[:-1],
                descriptors_to_close,
            )
            mode = 0o755 if descriptor.mode == "100755" else 0o644
            _write_new_regular_file_at(
                parent_descriptor,
                relative_path.name,
                source_contents[descriptor.relative_path],
                mode,
            )
    os.fsync(root_descriptor)


def _open_trusted_root(path: Path, descriptors_to_close: ExitStack) -> int:
    if not isinstance(path, Path) or not path.is_absolute() or path.resolve() != path:
        raise SourceReplayProviderFilesystemError(
            "provider trusted root must be an absolute resolved path"
        )
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors_to_close.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise SourceReplayProviderFilesystemError(
            "provider trusted root must be an owner-private real directory"
        )
    return descriptor


def _require_direct_child(trusted_root: Path, child: Path, name: str) -> None:
    if (
        not isinstance(trusted_root, Path)
        or not isinstance(child, Path)
        or not trusted_root.is_absolute()
        or not child.is_absolute()
        or child != Path(os.path.abspath(child))
        or child.parent != trusted_root
        or child.name in {"", ".", ".."}
    ):
        raise SourceReplayProviderFilesystemError(
            f"{name} must be a direct child of its trusted root"
        )


def _create_directory_at(
    parent_descriptor: int,
    name: str,
    descriptors_to_close: ExitStack,
) -> int:
    os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors_to_close.callback(os.close, descriptor)
    os.fchmod(descriptor, 0o700)
    _require_owned_directory(descriptor)
    return descriptor


def _create_relative_directory_at(
    root_descriptor: int,
    parts: tuple[str, ...],
    descriptors_to_close: ExitStack,
) -> int:
    if not parts:
        raise SourceReplayProviderFilesystemError(
            "provider materialization root must not be empty"
        )
    parent_descriptor = _open_or_create_relative_directory_at(
        root_descriptor,
        parts[:-1],
        descriptors_to_close,
    )
    return _create_directory_at(
        parent_descriptor,
        parts[-1],
        descriptors_to_close,
    )


def _open_or_create_relative_directory_at(
    root_descriptor: int,
    parts: tuple[str, ...],
    descriptors_to_close: ExitStack,
) -> int:
    descriptor = root_descriptor
    for part in parts:
        exists = os.access(
            part,
            os.F_OK,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if not exists:
            os.mkdir(part, mode=0o700, dir_fd=descriptor)
            os.fsync(descriptor)
        child_descriptor = os.open(
            part,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=descriptor,
        )
        descriptors_to_close.callback(os.close, child_descriptor)
        if not exists:
            os.fchmod(child_descriptor, 0o700)
        _require_owned_directory(child_descriptor)
        descriptor = child_descriptor
    return descriptor


def _require_owned_directory(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise SourceReplayProviderFilesystemError(
            "provider materialization directory is not owner-private"
        )


def _write_new_regular_file_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    mode: int,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode,
        dir_fd=parent_descriptor,
    )
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), mode)
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
        ):
            raise SourceReplayProviderFilesystemError(
                "provider input is not an independent owned regular file"
            )
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.fsync(parent_descriptor)


def _freeze_input_directory(root_descriptor: int) -> None:
    with os.scandir(root_descriptor) as entries:
        ordered_entries = tuple(sorted(entries, key=lambda entry: entry.name))
    for entry in ordered_entries:
        metadata = entry.stat(follow_symlinks=False)
        if metadata.st_uid != os.geteuid():
            raise SourceReplayProviderFilesystemError(
                "provider input closure contains an unowned entry"
            )
        if stat.S_ISDIR(metadata.st_mode):
            child_descriptor = os.open(
                entry.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=root_descriptor,
            )
            with ExitStack() as descriptor_to_close:
                descriptor_to_close.callback(os.close, child_descriptor)
                _freeze_input_directory(child_descriptor)
                os.fchmod(child_descriptor, 0o555)
                os.fsync(child_descriptor)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            child_descriptor = os.open(
                entry.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=root_descriptor,
            )
            with os.fdopen(child_descriptor, "rb") as file_handle:
                exact_metadata = os.fstat(file_handle.fileno())
                if (
                    not stat.S_ISREG(exact_metadata.st_mode)
                    or exact_metadata.st_nlink != 1
                    or exact_metadata.st_uid != os.geteuid()
                ):
                    raise SourceReplayProviderFilesystemError(
                        "provider input closure changed while it was frozen"
                    )
                frozen_mode = 0o555 if exact_metadata.st_mode & stat.S_IXUSR else 0o444
                os.fchmod(file_handle.fileno(), frozen_mode)
                os.fsync(file_handle.fileno())
        else:
            raise SourceReplayProviderFilesystemError(
                "provider input closure contains a link or special entry"
            )
    os.fchmod(root_descriptor, 0o555)
    os.fsync(root_descriptor)


def _validate_cleanup_directory_tree(directory_descriptor: int) -> None:
    _require_cleanup_directory_metadata(
        os.fstat(directory_descriptor),
        "provider workspace contains an unowned directory",
    )
    with os.scandir(directory_descriptor) as entries:
        observed_entries = tuple(
            sorted(
                ((entry.name, entry.stat(follow_symlinks=False)) for entry in entries),
                key=lambda item: item[0],
            )
        )
    for name, expected in observed_entries:
        _require_cleanup_entry_identity(directory_descriptor, name, expected)
        if stat.S_ISDIR(expected.st_mode):
            _require_cleanup_directory_metadata(
                expected,
                "provider workspace contains an unowned directory",
            )
            with ExitStack() as descriptors_to_close:
                child_descriptor = _open_cleanup_directory_at(
                    directory_descriptor,
                    name,
                    (expected.st_dev, expected.st_ino),
                    descriptors_to_close,
                )
                _validate_cleanup_directory_tree(child_descriptor)
        elif (
            not stat.S_ISREG(expected.st_mode)
            or expected.st_nlink != 1
            or expected.st_uid != os.geteuid()
        ):
            raise SourceReplayProviderFilesystemError(
                "provider workspace contains a link, special, unowned, or linked entry"
            )
        else:
            _require_cleanup_regular_file_at(
                directory_descriptor,
                name,
                expected,
            )


def _remove_cleanup_directory_contents(directory_descriptor: int) -> None:
    os.fchmod(directory_descriptor, 0o700)
    with os.scandir(directory_descriptor) as entries:
        observed_entries = tuple(
            sorted(
                ((entry.name, entry.stat(follow_symlinks=False)) for entry in entries),
                key=lambda item: item[0],
            )
        )
    for name, expected in observed_entries:
        _require_cleanup_entry_identity(directory_descriptor, name, expected)
        if stat.S_ISDIR(expected.st_mode):
            _require_cleanup_directory_metadata(
                expected,
                "provider workspace contains an unowned directory",
            )
            with ExitStack() as descriptors_to_close:
                child_descriptor = _open_cleanup_directory_at(
                    directory_descriptor,
                    name,
                    (expected.st_dev, expected.st_ino),
                    descriptors_to_close,
                )
                _remove_cleanup_directory_contents(child_descriptor)
            _require_cleanup_entry_identity(directory_descriptor, name, expected)
            os.rmdir(name, dir_fd=directory_descriptor)
        elif (
            not stat.S_ISREG(expected.st_mode)
            or expected.st_nlink != 1
            or expected.st_uid != os.geteuid()
        ):
            raise SourceReplayProviderFilesystemError(
                "provider workspace contains a link, special, unowned, or linked entry"
            )
        else:
            _require_cleanup_regular_file_at(
                directory_descriptor,
                name,
                expected,
            )
            _require_cleanup_entry_identity(directory_descriptor, name, expected)
            os.unlink(name, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)


def _open_cleanup_directory_at(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    descriptors_to_close: ExitStack,
) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors_to_close.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    if (metadata.st_dev, metadata.st_ino) != expected_identity:
        raise SourceReplayProviderFilesystemError(
            "provider workspace directory changed while opening"
        )
    _require_cleanup_directory_metadata(
        metadata,
        "provider workspace contains an unowned directory",
    )
    return descriptor


def _require_cleanup_directory_metadata(metadata: os.stat_result, message: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise SourceReplayProviderFilesystemError(message)


def _require_cleanup_entry_identity(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        raise SourceReplayProviderFilesystemError(
            "provider workspace entry changed during cleanup"
        )


def _require_cleanup_regular_file_at(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    with ExitStack() as descriptors_to_close:
        descriptors_to_close.callback(os.close, descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise SourceReplayProviderFilesystemError(
                "provider workspace regular file changed during cleanup"
            )
