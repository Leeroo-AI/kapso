"""Atomic no-replace publication of one complete run-action result."""

from __future__ import annotations

import fcntl
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_atomic_publication import (
    link_run_action_anonymous_file_no_replace,
    open_run_action_anonymous_file,
    require_run_action_descriptor_payload,
    write_run_action_full_payload,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_descriptor_mount_id,
)

_RESULT_FILE_NAME = "result.blob"
_RESULT_FILE_MODE = 0o600


class RunActionResultPublicationError(RuntimeError):
    """The result directory or final publication is not exact."""


@dataclass(frozen=True)
class RunActionPublishedResultObservation:
    """Physical identity and content proof for one committed result inode."""

    relative_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    content_digest: str
    parent_mount_id: int
    parent_device: int
    parent_inode: int
    mount_id: int
    device: int
    inode: int


@dataclass(frozen=True)
class _ResultDirectoryAuthority:
    metadata: os.stat_result
    mount_id: int


@dataclass(frozen=True)
class _ExactPublishedResult:
    metadata: os.stat_result
    mount_id: int


def publish_or_adopt_run_action_result(
    result_directory_descriptor: int,
    payload: bytes,
    *,
    maximum_size_bytes: int,
    process_snapshot_size_limit_bytes: int,
) -> RunActionPublishedResultObservation:
    """Publish through O_TMPFILE or adopt the byte-exact committed result."""

    directory = _require_result_publication_inputs(
        result_directory_descriptor,
        payload,
        maximum_size_bytes,
        process_snapshot_size_limit_bytes,
    )
    with ExitStack() as publication_lock:
        fcntl.flock(result_directory_descriptor, fcntl.LOCK_EX)
        publication_lock.callback(
            fcntl.flock,
            result_directory_descriptor,
            fcntl.LOCK_UN,
        )
        directory = _require_result_publication_inputs(
            result_directory_descriptor,
            payload,
            maximum_size_bytes,
            process_snapshot_size_limit_bytes,
        )
        entries = _require_result_directory(
            result_directory_descriptor,
            directory,
            process_snapshot_size_limit_bytes,
        )
        if entries == (_RESULT_FILE_NAME,):
            published = _adopt_result(
                result_directory_descriptor,
                payload,
                directory,
                process_snapshot_size_limit_bytes,
            )
        elif entries:
            raise RunActionResultPublicationError(
                "result publication directory contains an unexpected entry"
            )
        else:
            published = _publish_result(
                result_directory_descriptor,
                payload,
                directory,
                process_snapshot_size_limit_bytes,
            )
        return _published_result_observation(directory, published, payload)


def _publish_result(
    result_directory_descriptor: int,
    payload: bytes,
    directory: _ResultDirectoryAuthority,
    process_snapshot_size_limit_bytes: int,
) -> _ExactPublishedResult:
    with ExitStack() as descriptors:
        descriptor = open_run_action_anonymous_file(
            result_directory_descriptor,
            _RESULT_FILE_MODE,
        )
        descriptors.callback(os.close, descriptor)
        initial = os.fstat(descriptor)
        if (
            initial.st_uid != directory.metadata.st_uid
            or initial.st_gid != directory.metadata.st_gid
        ):
            os.fchown(
                descriptor,
                directory.metadata.st_uid,
                directory.metadata.st_gid,
            )
        write_run_action_full_payload(descriptor, payload)
        require_run_action_descriptor_payload(descriptor, payload)
        os.fchmod(descriptor, _RESULT_FILE_MODE)
        os.fsync(descriptor)
        anonymous = _observe_result_descriptor(
            descriptor,
            payload,
            directory,
            process_snapshot_size_limit_bytes,
            expected_link_count=0,
        )
        link_run_action_anonymous_file_no_replace(
            descriptor,
            result_directory_descriptor,
            _RESULT_FILE_NAME,
        )
        linked = _observe_result_descriptor(
            descriptor,
            payload,
            directory,
            process_snapshot_size_limit_bytes,
            expected_link_count=1,
        )
        if _file_identity(linked) != _file_identity(anonymous):
            raise RunActionResultPublicationError(
                "linked result differs from its complete anonymous inode"
            )
        os.fsync(result_directory_descriptor)
        _require_result_directory(
            result_directory_descriptor,
            directory,
            process_snapshot_size_limit_bytes,
            expected_entries=(_RESULT_FILE_NAME,),
        )
        reopened = _observe_result_path(
            result_directory_descriptor,
            payload,
            directory,
            process_snapshot_size_limit_bytes,
        )
        if _stable_file(reopened) != _stable_file(linked):
            raise RunActionResultPublicationError(
                "published result path differs from its linked inode"
            )
        return reopened


def _adopt_result(
    result_directory_descriptor: int,
    payload: bytes,
    directory: _ResultDirectoryAuthority,
    process_snapshot_size_limit_bytes: int,
) -> _ExactPublishedResult:
    first = _observe_result_path(
        result_directory_descriptor,
        payload,
        directory,
        process_snapshot_size_limit_bytes,
    )
    with ExitStack() as descriptors:
        descriptor = _open_result_path(result_directory_descriptor)
        descriptors.callback(os.close, descriptor)
        os.fsync(descriptor)
        synchronized = _observe_result_descriptor(
            descriptor,
            payload,
            directory,
            process_snapshot_size_limit_bytes,
            expected_link_count=1,
        )
    os.fsync(result_directory_descriptor)
    _require_result_directory(
        result_directory_descriptor,
        directory,
        process_snapshot_size_limit_bytes,
        expected_entries=(_RESULT_FILE_NAME,),
    )
    second = _observe_result_path(
        result_directory_descriptor,
        payload,
        directory,
        process_snapshot_size_limit_bytes,
    )
    if _stable_file(first) != _stable_file(synchronized) or _stable_file(
        second
    ) != _stable_file(first):
        raise RunActionResultPublicationError(
            "adopted result changed during synchronization"
        )
    return second


def _require_result_publication_inputs(
    result_directory_descriptor: int,
    payload: bytes,
    maximum_size_bytes: int,
    process_snapshot_size_limit_bytes: int,
) -> _ResultDirectoryAuthority:
    if (
        type(result_directory_descriptor) is not int
        or result_directory_descriptor < 0
        or type(payload) is not bytes
        or not payload
        or type(maximum_size_bytes) is not int
        or not 0 < len(payload) <= maximum_size_bytes
        or type(process_snapshot_size_limit_bytes) is not int
        or process_snapshot_size_limit_bytes <= 0
    ):
        raise RunActionResultPublicationError(
            "result publication inputs are invalid or unbounded"
        )
    metadata_before = os.fstat(result_directory_descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(
        result_directory_descriptor,
        process_snapshot_size_limit_bytes,
    )
    metadata_after = os.fstat(result_directory_descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(
        result_directory_descriptor,
        process_snapshot_size_limit_bytes,
    )
    if (
        not stat.S_ISDIR(metadata_before.st_mode)
        or metadata_before.st_uid <= 0
        or metadata_before.st_gid <= 0
        or stat.S_IMODE(metadata_before.st_mode) != 0o700
        or metadata_before.st_nlink < 2
        or metadata_before.st_dev <= 0
        or metadata_before.st_ino <= 0
        or _stable_directory(metadata_after) != _stable_directory(metadata_before)
        or mount_id_after != mount_id_before
        or mount_id_before <= 0
    ):
        raise RunActionResultPublicationError(
            "result publication directory lacks exact private authority"
        )
    return _ResultDirectoryAuthority(
        metadata=metadata_before,
        mount_id=mount_id_before,
    )


def _require_result_directory(
    result_directory_descriptor: int,
    expected: _ResultDirectoryAuthority,
    process_snapshot_size_limit_bytes: int,
    *,
    expected_entries: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    metadata_before = os.fstat(result_directory_descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(
        result_directory_descriptor,
        process_snapshot_size_limit_bytes,
    )
    entries = tuple(sorted(os.listdir(result_directory_descriptor)))
    metadata_after = os.fstat(result_directory_descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(
        result_directory_descriptor,
        process_snapshot_size_limit_bytes,
    )
    if (
        _directory_identity(metadata_before) != _directory_identity(expected.metadata)
        or _stable_directory(metadata_after) != _stable_directory(metadata_before)
        or mount_id_before != expected.mount_id
        or mount_id_after != mount_id_before
        or (expected_entries is not None and entries != expected_entries)
    ):
        raise RunActionResultPublicationError(
            "result publication directory changed or has invalid topology"
        )
    return entries


def _observe_result_path(
    result_directory_descriptor: int,
    payload: bytes,
    directory: _ResultDirectoryAuthority,
    process_snapshot_size_limit_bytes: int,
) -> _ExactPublishedResult:
    descriptor = _open_result_path(result_directory_descriptor)
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        return _observe_result_descriptor(
            descriptor,
            payload,
            directory,
            process_snapshot_size_limit_bytes,
            expected_link_count=1,
        )


def _open_result_path(result_directory_descriptor: int) -> int:
    return os.open(
        _RESULT_FILE_NAME,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=result_directory_descriptor,
    )


def _observe_result_descriptor(
    descriptor: int,
    payload: bytes,
    directory: _ResultDirectoryAuthority,
    process_snapshot_size_limit_bytes: int,
    *,
    expected_link_count: int,
) -> _ExactPublishedResult:
    metadata_before = os.fstat(descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(
        descriptor,
        process_snapshot_size_limit_bytes,
    )
    require_run_action_descriptor_payload(descriptor, payload)
    metadata_after = os.fstat(descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(
        descriptor,
        process_snapshot_size_limit_bytes,
    )
    if (
        not stat.S_ISREG(metadata_before.st_mode)
        or metadata_before.st_uid != directory.metadata.st_uid
        or metadata_before.st_gid != directory.metadata.st_gid
        or stat.S_IMODE(metadata_before.st_mode) != _RESULT_FILE_MODE
        or metadata_before.st_nlink != expected_link_count
        or metadata_before.st_size != len(payload)
        or metadata_before.st_dev != directory.metadata.st_dev
        or metadata_before.st_ino == directory.metadata.st_ino
        or mount_id_before != directory.mount_id
        or _stable_file(metadata_after, mount_id_after)
        != _stable_file(metadata_before, mount_id_before)
    ):
        raise RunActionResultPublicationError(
            "published result inode is incomplete or substituted"
        )
    return _ExactPublishedResult(
        metadata=metadata_before,
        mount_id=mount_id_before,
    )


def _published_result_observation(
    directory: _ResultDirectoryAuthority,
    published: _ExactPublishedResult,
    payload: bytes,
) -> RunActionPublishedResultObservation:
    return RunActionPublishedResultObservation(
        relative_path="result/result.blob",
        file_type="regular",
        owner_user_id=published.metadata.st_uid,
        owner_group_id=published.metadata.st_gid,
        mode=stat.S_IMODE(published.metadata.st_mode),
        link_count=published.metadata.st_nlink,
        size_bytes=published.metadata.st_size,
        content_digest=tree_or_blob_digest(payload),
        parent_mount_id=directory.mount_id,
        parent_device=directory.metadata.st_dev,
        parent_inode=directory.metadata.st_ino,
        mount_id=published.mount_id,
        device=published.metadata.st_dev,
        inode=published.metadata.st_ino,
    )


def _stable_directory(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
    )


def _stable_file(
    observed: _ExactPublishedResult | os.stat_result,
    mount_id: int | None = None,
) -> tuple[int, ...]:
    if type(observed) is _ExactPublishedResult:
        metadata = observed.metadata
        exact_mount_id = observed.mount_id
    else:
        metadata = observed
        exact_mount_id = mount_id
    if type(metadata) is not os.stat_result or type(exact_mount_id) is not int:
        raise RunActionResultPublicationError(
            "published result observation is malformed"
        )
    return (
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        exact_mount_id,
    )


def _file_identity(observed: _ExactPublishedResult) -> tuple[int, int, int]:
    return (
        observed.mount_id,
        observed.metadata.st_dev,
        observed.metadata.st_ino,
    )


__all__ = [
    "publish_or_adopt_run_action_result",
    "RunActionPublishedResultObservation",
    "RunActionResultPublicationError",
]
