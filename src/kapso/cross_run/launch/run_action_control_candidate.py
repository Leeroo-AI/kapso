"""Closed anonymous-file publication for the two legal control transitions."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from enum import Enum
from threading import get_ident, Lock
from weakref import WeakValueDictionary

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_atomic_publication import (
    link_run_action_anonymous_file_no_replace,
    require_run_action_descriptor_payload,
    write_run_action_full_payload,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_descriptor_mount_id,
)

_CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY = object()
_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY = object()
_ISSUED_CONTROL_FILE_CANDIDATES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_CONTROL_FILE_CANDIDATE_LOCK = Lock()
_CONTROL_DIRECTORY_MODE = 0o700
_CONTROL_FILE_MODE = 0o400


class RunActionControlCandidateError(RuntimeError):
    """A frozen control inode or its closed topology transition is unsafe."""


class _RunActionControlFileTransition(str, Enum):
    RELEASE = "release"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class _RunActionControlFileTransitionSpec:
    before: RunActionControlDirectoryTopology
    after: RunActionControlDirectoryTopology
    final_file_name: str
    predecessor_file_name: str | None


_CONTROL_FILE_TRANSITION_SPECS = {
    _RunActionControlFileTransition.RELEASE: _RunActionControlFileTransitionSpec(
        before=RunActionControlDirectoryTopology.EMPTY,
        after=RunActionControlDirectoryTopology.RELEASED,
        final_file_name="release",
        predecessor_file_name=None,
    ),
    _RunActionControlFileTransition.TIMEOUT: _RunActionControlFileTransitionSpec(
        before=RunActionControlDirectoryTopology.RELEASED,
        after=RunActionControlDirectoryTopology.TIMED_OUT,
        final_file_name="timeout",
        predecessor_file_name="release",
    ),
}


@dataclass(frozen=True)
class _RunActionLinkedControlFileEvidence:
    """Exact physical identity of the candidate after its durable link."""

    transition: _RunActionControlFileTransition
    final_file_name: str
    mount_id: int
    device: int
    inode: int
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    content_digest: str


class _RunActionFrozenControlFileCandidate:
    """One owner-bound candidate for a closed control topology transition."""

    def __init__(
        self,
        *,
        transition: _RunActionControlFileTransition,
        control_directory_descriptor: int,
        anonymous_file_descriptor: int,
        predecessor_file_descriptor: int | None,
        owner_user_id: int,
        owner_group_id: int,
        payload_size_limit_bytes: int,
        payload: bytes,
        _authority: object,
    ) -> None:
        requires_predecessor = transition is _RunActionControlFileTransition.TIMEOUT
        if (
            type(transition) is not _RunActionControlFileTransition
            or type(control_directory_descriptor) is not int
            or control_directory_descriptor < 0
            or type(anonymous_file_descriptor) is not int
            or anonymous_file_descriptor < 0
            or control_directory_descriptor == anonymous_file_descriptor
            or requires_predecessor != (type(predecessor_file_descriptor) is int)
            or (
                type(predecessor_file_descriptor) is int
                and (
                    predecessor_file_descriptor < 0
                    or predecessor_file_descriptor
                    in {
                        control_directory_descriptor,
                        anonymous_file_descriptor,
                    }
                )
            )
            or type(owner_user_id) is not int
            or owner_user_id <= 0
            or type(owner_group_id) is not int
            or owner_group_id <= 0
            or type(payload_size_limit_bytes) is not int
            or payload_size_limit_bytes <= 0
            or type(payload) is not bytes
            or not payload
            or len(payload) > payload_size_limit_bytes
            or _authority is not _CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY
        ):
            raise RunActionControlCandidateError(
                "control candidate requires one exact closed transition"
            )
        self._transition = transition
        self._spec = _CONTROL_FILE_TRANSITION_SPECS[transition]
        self._control_directory_descriptor = control_directory_descriptor
        self._anonymous_descriptor = anonymous_file_descriptor
        self._predecessor_descriptor = predecessor_file_descriptor
        self._owner_user_id = owner_user_id
        self._owner_group_id = owner_group_id
        self._payload_size_limit_bytes = payload_size_limit_bytes
        self._payload = payload
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._state = "staging"
        self._parent_identity = self._require_control_topology(self._spec.before)
        self._predecessor_identity = self._require_predecessor_file(None)
        initial_metadata = os.fstat(self._anonymous_descriptor)
        if (
            initial_metadata.st_uid != owner_user_id
            or initial_metadata.st_gid != owner_group_id
        ):
            os.fchown(
                self._anonymous_descriptor,
                owner_user_id,
                owner_group_id,
            )
        write_run_action_full_payload(self._anonymous_descriptor, payload)
        require_run_action_descriptor_payload(self._anonymous_descriptor, payload)
        os.fchmod(self._anonymous_descriptor, _CONTROL_FILE_MODE)
        os.fsync(self._anonymous_descriptor)
        self._frozen_identity = self._require_anonymous_file(link_count=0)
        self._state = "frozen"
        with _CONTROL_FILE_CANDIDATE_LOCK:
            if _ISSUED_CONTROL_FILE_CANDIDATES.get(id(self)) is not None:
                raise RunActionControlCandidateError(
                    "control candidate identity is already issued"
                )
            _ISSUED_CONTROL_FILE_CANDIDATES[id(self)] = self

    def _begin_publication(
        self,
        expected_payload: bytes,
        *,
        _authority: object,
    ) -> bytes:
        self._require_issued("frozen", _authority)
        if (
            type(expected_payload) is not bytes
            or expected_payload != self._payload
            or self._require_control_topology(self._spec.before)
            != self._parent_identity
            or self._require_predecessor_file(self._predecessor_identity)
            != self._predecessor_identity
            or self._require_anonymous_file(link_count=0) != self._frozen_identity
        ):
            raise RunActionControlCandidateError(
                "control candidate differs from its exact frozen payload"
            )
        self._state = "authorizing"
        return self._payload

    def _link_authorized_once(
        self,
        *,
        _authority: object,
    ) -> _RunActionLinkedControlFileEvidence:
        self._require_issued("authorizing", _authority)
        if (
            self._require_control_topology(self._spec.before) != self._parent_identity
            or self._require_predecessor_file(self._predecessor_identity)
            != self._predecessor_identity
            or self._require_anonymous_file(link_count=0) != self._frozen_identity
        ):
            raise RunActionControlCandidateError(
                "control candidate changed before its irreversible link"
            )
        self._state = "linking"
        link_run_action_anonymous_file_no_replace(
            self._anonymous_descriptor,
            self._control_directory_descriptor,
            self._spec.final_file_name,
        )
        self._state = "linked"
        linked_identity, evidence = self._require_linked_file()
        if linked_identity != self._frozen_identity:
            raise RunActionControlCandidateError(
                "linked control file differs from its frozen anonymous inode"
            )
        if (
            self._require_predecessor_file(self._predecessor_identity)
            != self._predecessor_identity
            or self._require_control_topology(self._spec.after) != self._parent_identity
        ):
            raise RunActionControlCandidateError(
                "control predecessor or topology changed across publication"
            )
        os.fsync(self._control_directory_descriptor)
        if (
            self._require_linked_file()[0] != linked_identity
            or self._require_predecessor_file(self._predecessor_identity)
            != self._predecessor_identity
            or self._require_control_topology(self._spec.after) != self._parent_identity
        ):
            raise RunActionControlCandidateError(
                "linked control file changed during durability proof"
            )
        self._state = "durable"
        return evidence

    def _require_issued(self, expected_state: str, _authority: object) -> None:
        with _CONTROL_FILE_CANDIDATE_LOCK:
            issued = _ISSUED_CONTROL_FILE_CANDIDATES.get(id(self))
        if (
            issued is not self
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
            or self._state != expected_state
            or _authority is not _CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY
        ):
            raise RunActionControlCandidateError(
                "control candidate is unissued, spent, or foreign"
            )

    def _require_control_topology(
        self,
        expected_topology: RunActionControlDirectoryTopology,
    ) -> tuple[int, ...]:
        before = os.fstat(self._control_directory_descriptor)
        mount_id_before = read_run_action_descriptor_mount_id(
            self._control_directory_descriptor
        )
        entries = tuple(sorted(os.listdir(self._control_directory_descriptor)))
        after = os.fstat(self._control_directory_descriptor)
        mount_id_after = read_run_action_descriptor_mount_id(
            self._control_directory_descriptor
        )
        identity = _directory_authority_identity(after, mount_id_after)
        if (
            type(expected_topology) is not RunActionControlDirectoryTopology
            or not stat.S_ISDIR(before.st_mode)
            or _stable_directory_snapshot(before, mount_id_before)
            != _stable_directory_snapshot(after, mount_id_after)
            or before.st_uid != self._owner_user_id
            or before.st_gid != self._owner_group_id
            or stat.S_IMODE(before.st_mode) != _CONTROL_DIRECTORY_MODE
            or mount_id_before <= 0
            or mount_id_after != mount_id_before
            or entries != expected_topology.entries
        ):
            raise RunActionControlCandidateError(
                "control directory topology is changed or unsafe"
            )
        return identity

    def _require_predecessor_file(
        self,
        expected_identity: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        predecessor_file_name = self._spec.predecessor_file_name
        if predecessor_file_name is None:
            if (
                self._predecessor_descriptor is not None
                or expected_identity is not None
            ):
                raise RunActionControlCandidateError(
                    "release transition gained a predecessor file"
                )
            return None
        retained_before = os.fstat(self._predecessor_descriptor)
        retained_mount_id_before = read_run_action_descriptor_mount_id(
            self._predecessor_descriptor
        )
        descriptor = os.open(
            predecessor_file_name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._control_directory_descriptor,
        )
        with os.fdopen(descriptor, "rb", buffering=0) as predecessor_file:
            path_metadata = os.fstat(predecessor_file.fileno())
            path_mount_id = read_run_action_descriptor_mount_id(
                predecessor_file.fileno()
            )
        retained_after = os.fstat(self._predecessor_descriptor)
        retained_mount_id_after = read_run_action_descriptor_mount_id(
            self._predecessor_descriptor
        )
        identity = _stable_predecessor_identity(
            retained_after,
            retained_mount_id_after,
        )
        control_metadata = os.fstat(self._control_directory_descriptor)
        if (
            not stat.S_ISREG(retained_before.st_mode)
            or _stable_predecessor_identity(
                retained_before,
                retained_mount_id_before,
            )
            != identity
            or _stable_predecessor_identity(path_metadata, path_mount_id) != identity
            or (expected_identity is not None and identity != expected_identity)
            or retained_before.st_uid != self._owner_user_id
            or retained_before.st_gid != self._owner_group_id
            or stat.S_IMODE(retained_before.st_mode) != _CONTROL_FILE_MODE
            or retained_before.st_nlink != 1
            or retained_after.st_nlink != 1
            or retained_before.st_size <= 0
            or retained_mount_id_before <= 0
            or retained_mount_id_after != retained_mount_id_before
            or retained_before.st_dev != control_metadata.st_dev
            or retained_before.st_ino == control_metadata.st_ino
        ):
            raise RunActionControlCandidateError(
                "timeout predecessor release is changed or unsafe"
            )
        return identity

    def _require_anonymous_file(self, *, link_count: int) -> tuple[int, ...]:
        before = os.fstat(self._anonymous_descriptor)
        mount_id_before = read_run_action_descriptor_mount_id(
            self._anonymous_descriptor
        )
        require_run_action_descriptor_payload(
            self._anonymous_descriptor,
            self._payload,
        )
        after = os.fstat(self._anonymous_descriptor)
        mount_id_after = read_run_action_descriptor_mount_id(self._anonymous_descriptor)
        identity = _publication_file_identity(after, mount_id_after)
        control_metadata = os.fstat(self._control_directory_descriptor)
        predecessor_inode = (
            None
            if self._predecessor_descriptor is None
            else os.fstat(self._predecessor_descriptor).st_ino
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or _publication_file_identity(before, mount_id_before) != identity
            or before.st_uid != self._owner_user_id
            or before.st_gid != self._owner_group_id
            or stat.S_IMODE(before.st_mode) != _CONTROL_FILE_MODE
            or before.st_nlink != link_count
            or after.st_nlink != link_count
            or before.st_size != len(self._payload)
            or mount_id_before <= 0
            or mount_id_after != mount_id_before
            or before.st_dev != control_metadata.st_dev
            or before.st_ino == control_metadata.st_ino
            or (predecessor_inode is not None and before.st_ino == predecessor_inode)
        ):
            raise RunActionControlCandidateError(
                "anonymous control inode is changed or unsafe"
            )
        return identity

    def _require_linked_file(
        self,
    ) -> tuple[tuple[int, ...], _RunActionLinkedControlFileEvidence]:
        if self._state not in {"linked", "durable"}:
            raise RunActionControlCandidateError(
                "control path cannot be observed before link"
            )
        retained = os.fstat(self._anonymous_descriptor)
        retained_mount_id = read_run_action_descriptor_mount_id(
            self._anonymous_descriptor
        )
        descriptor = os.open(
            self._spec.final_file_name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._control_directory_descriptor,
        )
        with os.fdopen(descriptor, "rb", buffering=0) as published_file:
            require_run_action_descriptor_payload(
                published_file.fileno(),
                self._payload,
            )
            path_metadata = os.fstat(published_file.fileno())
            path_mount_id = read_run_action_descriptor_mount_id(published_file.fileno())
            os.fsync(published_file.fileno())
        retained_identity = _publication_file_identity(
            retained,
            retained_mount_id,
        )
        path_identity = _publication_file_identity(path_metadata, path_mount_id)
        if (
            retained_identity != path_identity
            or retained.st_nlink != 1
            or path_metadata.st_nlink != 1
            or path_metadata.st_uid != self._owner_user_id
            or path_metadata.st_gid != self._owner_group_id
            or stat.S_IMODE(path_metadata.st_mode) != _CONTROL_FILE_MODE
        ):
            raise RunActionControlCandidateError(
                "control path differs from its retained linked inode"
            )
        return retained_identity, _RunActionLinkedControlFileEvidence(
            transition=self._transition,
            final_file_name=self._spec.final_file_name,
            mount_id=path_mount_id,
            device=path_metadata.st_dev,
            inode=path_metadata.st_ino,
            owner_user_id=path_metadata.st_uid,
            owner_group_id=path_metadata.st_gid,
            mode=stat.S_IMODE(path_metadata.st_mode),
            link_count=path_metadata.st_nlink,
            size_bytes=path_metadata.st_size,
            content_digest=tree_or_blob_digest(self._payload),
        )

    def close(self) -> None:
        if self._state == "closed":
            raise RunActionControlCandidateError("control candidate is already closed")
        with _CONTROL_FILE_CANDIDATE_LOCK:
            issued = _ISSUED_CONTROL_FILE_CANDIDATES.pop(id(self), None)
        if issued is not self:
            raise RunActionControlCandidateError(
                "control candidate issuance changed before close"
            )
        self._state = "closed"


def _directory_authority_identity(
    metadata: os.stat_result,
    mount_id: int,
) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_dev,
        metadata.st_ino,
        mount_id,
    )


def _stable_directory_snapshot(
    metadata: os.stat_result,
    mount_id: int,
) -> tuple[int, ...]:
    return (
        *_directory_authority_identity(metadata, mount_id),
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_predecessor_identity(
    metadata: os.stat_result,
    mount_id: int,
) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        mount_id,
    )


def _publication_file_identity(
    metadata: os.stat_result,
    mount_id: int,
) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        mount_id,
    )


__all__ = ["RunActionControlCandidateError"]
