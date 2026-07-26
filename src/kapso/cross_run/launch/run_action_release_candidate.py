"""Exact frozen inode that is the sole subject of final release authorization."""

from __future__ import annotations

import os
import stat
import time
from threading import get_ident, Lock
from weakref import WeakValueDictionary

from kapso.cross_run.launch.run_action_atomic_publication import (
    link_run_action_anonymous_file_no_replace,
    require_run_action_descriptor_payload,
    write_run_action_full_payload,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_descriptor_mount_id,
)

_RELEASE_CANDIDATE_ISSUANCE_AUTHORITY = object()
_RELEASE_CANDIDATE_AUTHORIZATION_AUTHORITY = object()
_ISSUED_RELEASE_CANDIDATES: WeakValueDictionary[int, object] = WeakValueDictionary()
_RELEASE_CANDIDATE_LOCK = Lock()
_RELEASE_FILE_MODE = 0o400
_RELEASE_FILE_NAME = "release"


class RunActionReleaseCandidateError(RuntimeError):
    """A frozen release inode or its exact publication transaction is unsafe."""


class _SystemRunActionReleaseClock:
    """Coordinator-issued Linux clocks that cannot be supplied by an adapter."""

    def boottime_nanoseconds(self) -> int:
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)

    def realtime_nanoseconds(self) -> int:
        return time.clock_gettime_ns(time.CLOCK_REALTIME)


class _RunActionFrozenReleaseCandidate:
    """Publisher-issued receipt inode with one fixed no-replace link operation."""

    def __init__(
        self,
        *,
        control_directory_descriptor: int,
        anonymous_file_descriptor: int,
        owner_user_id: int,
        owner_group_id: int,
        payload_size_limit_bytes: int,
        receipt: RunActionWorkloadReleaseReceipt,
        _authority: object,
    ) -> None:
        if (
            type(control_directory_descriptor) is not int
            or control_directory_descriptor < 0
            or type(anonymous_file_descriptor) is not int
            or anonymous_file_descriptor < 0
            or type(owner_user_id) is not int
            or owner_user_id <= 0
            or type(owner_group_id) is not int
            or owner_group_id <= 0
            or type(payload_size_limit_bytes) is not int
            or payload_size_limit_bytes <= 0
            or type(receipt) is not RunActionWorkloadReleaseReceipt
            or _authority is not _RELEASE_CANDIDATE_ISSUANCE_AUTHORITY
        ):
            raise RunActionReleaseCandidateError(
                "release candidate requires one exact bounded receipt transaction"
            )
        self._control_directory_descriptor = control_directory_descriptor
        self._anonymous_descriptor = anonymous_file_descriptor
        self._owner_user_id = owner_user_id
        self._owner_group_id = owner_group_id
        self._payload_size_limit_bytes = payload_size_limit_bytes
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._state = "staging"
        self._parent_identity = self._require_control_entries(())
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
        payload = receipt.to_json_bytes()
        if (
            not payload
            or len(payload) > payload_size_limit_bytes
            or RunActionWorkloadReleaseReceipt.from_json_bytes(payload) != receipt
        ):
            raise RunActionReleaseCandidateError(
                "release receipt bytes are noncanonical or oversized"
            )
        write_run_action_full_payload(self._anonymous_descriptor, payload)
        require_run_action_descriptor_payload(self._anonymous_descriptor, payload)
        os.fchmod(self._anonymous_descriptor, _RELEASE_FILE_MODE)
        os.fsync(self._anonymous_descriptor)
        self._receipt = receipt
        self._payload = payload
        self._frozen_identity = self._require_anonymous_file()
        self._state = "frozen"
        with _RELEASE_CANDIDATE_LOCK:
            if _ISSUED_RELEASE_CANDIDATES.get(id(self)) is not None:
                raise RunActionReleaseCandidateError(
                    "release candidate identity is already issued"
                )
            _ISSUED_RELEASE_CANDIDATES[id(self)] = self

    def _begin_authorization(
        self,
        expected_receipt: RunActionWorkloadReleaseReceipt,
        *,
        _authority: object,
    ) -> RunActionWorkloadReleaseReceipt:
        self._require_issued("frozen", _authority)
        if (
            type(expected_receipt) is not RunActionWorkloadReleaseReceipt
            or self._receipt != expected_receipt
            or self._require_control_entries(()) != self._parent_identity
            or self._require_anonymous_file() != self._frozen_identity
        ):
            raise RunActionReleaseCandidateError(
                "release candidate differs from its exact frozen receipt"
            )
        self._state = "authorizing"
        return self._receipt

    def _link_authorized_once(
        self,
        *,
        _authority: object,
    ) -> RunActionWorkloadReleaseReceipt:
        self._require_issued("authorizing", _authority)
        self._state = "linking"
        link_run_action_anonymous_file_no_replace(
            self._anonymous_descriptor,
            self._control_directory_descriptor,
            _RELEASE_FILE_NAME,
        )
        self._state = "linked"
        linked_identity = self._require_linked_file()
        if linked_identity != self._frozen_identity:
            raise RunActionReleaseCandidateError(
                "linked release differs from its frozen anonymous inode"
            )
        os.fsync(self._control_directory_descriptor)
        if (
            self._require_linked_file() != linked_identity
            or self._require_control_entries((_RELEASE_FILE_NAME,))
            != self._parent_identity
        ):
            raise RunActionReleaseCandidateError(
                "linked release changed during durability proof"
            )
        self._state = "durable"
        return self._receipt

    def _require_issued(self, expected_state: str, _authority: object) -> None:
        with _RELEASE_CANDIDATE_LOCK:
            issued = _ISSUED_RELEASE_CANDIDATES.get(id(self))
        if (
            issued is not self
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
            or self._state != expected_state
            or _authority is not _RELEASE_CANDIDATE_AUTHORIZATION_AUTHORITY
        ):
            raise RunActionReleaseCandidateError(
                "release candidate is unissued, spent, or foreign"
            )

    def _require_control_entries(
        self,
        expected_entries: tuple[str, ...],
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
            not stat.S_ISDIR(before.st_mode)
            or _stable_directory_snapshot(before, mount_id_before)
            != _stable_directory_snapshot(after, mount_id_after)
            or before.st_uid != self._owner_user_id
            or before.st_gid != self._owner_group_id
            or stat.S_IMODE(before.st_mode) != 0o700
            or mount_id_before <= 0
            or mount_id_after != mount_id_before
            or entries != expected_entries
        ):
            raise RunActionReleaseCandidateError(
                "release control directory is changed or unsafe"
            )
        return identity

    def _require_anonymous_file(self) -> tuple[int, ...]:
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
        identity = _stable_file_identity(after, mount_id_after)
        control_metadata = os.fstat(self._control_directory_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _stable_file_identity(before, mount_id_before) != identity
            or before.st_uid != self._owner_user_id
            or before.st_gid != self._owner_group_id
            or stat.S_IMODE(before.st_mode) != _RELEASE_FILE_MODE
            or before.st_nlink != 0
            or after.st_nlink != 0
            or before.st_size != len(self._payload)
            or mount_id_before <= 0
            or mount_id_after != mount_id_before
            or before.st_dev != control_metadata.st_dev
            or before.st_ino == control_metadata.st_ino
        ):
            raise RunActionReleaseCandidateError(
                "anonymous release inode is changed or unsafe"
            )
        return identity

    def _require_linked_file(self) -> tuple[int, ...]:
        if self._state not in {"linked", "durable"}:
            raise RunActionReleaseCandidateError(
                "release path cannot be observed before link"
            )
        retained = os.fstat(self._anonymous_descriptor)
        retained_mount_id = read_run_action_descriptor_mount_id(
            self._anonymous_descriptor
        )
        descriptor = os.open(
            _RELEASE_FILE_NAME,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._control_directory_descriptor,
        )
        with os.fdopen(descriptor, "rb", buffering=0) as release_file:
            require_run_action_descriptor_payload(
                release_file.fileno(),
                self._payload,
            )
            path_metadata = os.fstat(release_file.fileno())
            path_mount_id = read_run_action_descriptor_mount_id(release_file.fileno())
            os.fsync(release_file.fileno())
        retained_identity = _stable_file_identity(retained, retained_mount_id)
        path_identity = _stable_file_identity(path_metadata, path_mount_id)
        if (
            retained_identity != path_identity
            or retained.st_nlink != 1
            or path_metadata.st_nlink != 1
            or stat.S_IMODE(path_metadata.st_mode) != _RELEASE_FILE_MODE
        ):
            raise RunActionReleaseCandidateError(
                "release path differs from its retained linked inode"
            )
        return retained_identity

    def close(self) -> None:
        if self._state == "closed":
            raise RunActionReleaseCandidateError("release candidate is already closed")
        with _RELEASE_CANDIDATE_LOCK:
            issued = _ISSUED_RELEASE_CANDIDATES.pop(id(self), None)
        if issued is not self:
            raise RunActionReleaseCandidateError(
                "release candidate issuance changed before close"
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


def _stable_file_identity(
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


__all__ = ["RunActionReleaseCandidateError"]
