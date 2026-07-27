"""Crash-atomic publication of one coding-agent terminal result candidate."""

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

_RESULT_CANDIDATE_FILE_NAME = "result.candidate"
_RESULT_CANDIDATE_FILE_MODE = 0o600
_TEMPORARY_DIRECTORY_MODE = 0o700


class RunActionCodingAgentCandidateError(RuntimeError):
    """The temporary authority or candidate publication is not exact."""


@dataclass(frozen=True)
class CodingAgentPublishedCandidate:
    """Complete physical and content proof for the visible candidate inode."""

    size_bytes: int
    content_digest: str
    device: int
    inode: int

    def __post_init__(self) -> None:
        if (
            type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or not isinstance(self.content_digest, str)
            or not self.content_digest.startswith("sha256:")
            or type(self.device) is not int
            or self.device <= 0
            or type(self.inode) is not int
            or self.inode <= 0
        ):
            raise RunActionCodingAgentCandidateError(
                "coding-agent candidate observation is invalid"
            )


def publish_coding_agent_result_candidate(
    temporary_directory_descriptor: int,
    payload: bytes,
    *,
    maximum_size_bytes: int,
) -> CodingAgentPublishedCandidate:
    """Publish one complete candidate by anonymous inode and no-replace link."""

    _require_candidate_inputs(
        temporary_directory_descriptor,
        payload,
        maximum_size_bytes,
    )
    with ExitStack() as publication:
        fcntl.flock(temporary_directory_descriptor, fcntl.LOCK_EX)
        publication.callback(
            fcntl.flock,
            temporary_directory_descriptor,
            fcntl.LOCK_UN,
        )
        directory = _require_candidate_inputs(
            temporary_directory_descriptor,
            payload,
            maximum_size_bytes,
        )
        if _RESULT_CANDIDATE_FILE_NAME in os.listdir(temporary_directory_descriptor):
            raise RunActionCodingAgentCandidateError(
                "coding-agent result candidate already exists"
            )
        anonymous_descriptor = open_run_action_anonymous_file(
            temporary_directory_descriptor,
            _RESULT_CANDIDATE_FILE_MODE,
        )
        publication.callback(os.close, anonymous_descriptor)
        initial = os.fstat(anonymous_descriptor)
        if initial.st_uid != directory.st_uid or initial.st_gid != directory.st_gid:
            os.fchown(
                anonymous_descriptor,
                directory.st_uid,
                directory.st_gid,
            )
        write_run_action_full_payload(anonymous_descriptor, payload)
        require_run_action_descriptor_payload(anonymous_descriptor, payload)
        os.fchmod(anonymous_descriptor, _RESULT_CANDIDATE_FILE_MODE)
        os.fsync(anonymous_descriptor)
        anonymous = _require_candidate_descriptor(
            anonymous_descriptor,
            payload,
            directory,
            expected_link_count=0,
        )
        link_run_action_anonymous_file_no_replace(
            anonymous_descriptor,
            temporary_directory_descriptor,
            _RESULT_CANDIDATE_FILE_NAME,
        )
        linked = _require_candidate_descriptor(
            anonymous_descriptor,
            payload,
            directory,
            expected_link_count=1,
        )
        if _candidate_identity(linked) != _candidate_identity(anonymous):
            raise RunActionCodingAgentCandidateError(
                "linked candidate differs from its complete anonymous inode"
            )
        os.fsync(temporary_directory_descriptor)
        reopened_descriptor = os.open(
            _RESULT_CANDIDATE_FILE_NAME,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=temporary_directory_descriptor,
        )
        publication.callback(os.close, reopened_descriptor)
        reopened = _require_candidate_descriptor(
            reopened_descriptor,
            payload,
            directory,
            expected_link_count=1,
        )
        if _stable_candidate(reopened) != _stable_candidate(linked):
            raise RunActionCodingAgentCandidateError(
                "reopened candidate differs from its linked inode"
            )
        rebound = os.fstat(temporary_directory_descriptor)
        if _directory_identity(rebound) != _directory_identity(directory):
            raise RunActionCodingAgentCandidateError(
                "temporary directory changed during candidate publication"
            )
        return CodingAgentPublishedCandidate(
            size_bytes=reopened.st_size,
            content_digest=tree_or_blob_digest(payload),
            device=reopened.st_dev,
            inode=reopened.st_ino,
        )


def _require_candidate_inputs(
    temporary_directory_descriptor: int,
    payload: bytes,
    maximum_size_bytes: int,
) -> os.stat_result:
    if (
        type(temporary_directory_descriptor) is not int
        or temporary_directory_descriptor < 0
        or type(payload) is not bytes
        or not payload
        or type(maximum_size_bytes) is not int
        or not 0 < len(payload) <= maximum_size_bytes
    ):
        raise RunActionCodingAgentCandidateError(
            "coding-agent candidate inputs are invalid or unbounded"
        )
    metadata = os.fstat(temporary_directory_descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != os.getegid()
        or stat.S_IMODE(metadata.st_mode) != _TEMPORARY_DIRECTORY_MODE
        or metadata.st_nlink < 2
        or metadata.st_dev <= 0
        or metadata.st_ino <= 0
    ):
        raise RunActionCodingAgentCandidateError(
            "coding-agent temporary directory lacks private authority"
        )
    return metadata


def _require_candidate_descriptor(
    descriptor: int,
    payload: bytes,
    directory: os.stat_result,
    *,
    expected_link_count: int,
) -> os.stat_result:
    before = os.fstat(descriptor)
    require_run_action_descriptor_payload(descriptor, payload)
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != directory.st_uid
        or before.st_gid != directory.st_gid
        or stat.S_IMODE(before.st_mode) != _RESULT_CANDIDATE_FILE_MODE
        or before.st_nlink != expected_link_count
        or before.st_dev != directory.st_dev
        or before.st_ino <= 0
        or before.st_size != len(payload)
        or _stable_candidate(after) != _stable_candidate(before)
    ):
        raise RunActionCodingAgentCandidateError(
            "coding-agent candidate inode is invalid or changed"
        )
    return before


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
    )


def _candidate_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _stable_candidate(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


__all__ = [
    "CodingAgentPublishedCandidate",
    "RunActionCodingAgentCandidateError",
    "publish_coding_agent_result_candidate",
]
