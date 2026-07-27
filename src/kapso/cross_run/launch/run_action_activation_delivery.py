"""Descriptor-only atomic publication of committed run-action payloads."""

from __future__ import annotations

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
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparedDeliverySlot,
    RunActionPreparedFileKind,
)

_ANONYMOUS_FILE_MODE = 0o600
_DELIVERED_FILE_MODE = 0o400
_DELIVERED_FILE_LEASE_AUTHORITY = object()


class RunActionActivationDeliveryError(RuntimeError):
    """A prepared delivery slot or published payload is not exact."""


@dataclass(frozen=True)
class RunActionDeliveredFilePhysicalObservation:
    """Non-secret physical identity of one atomically published payload."""

    prepared_delivery_slot_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    kind: RunActionPreparedFileKind
    relative_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    mount_id: int
    device: int
    inode: int
    content_digest: str | None


class RunActionDeliveredFileLease:
    """Process-bound ownership of one exact linked delivery descriptor."""

    def __init__(
        self,
        *,
        slot: RunActionPreparedDeliverySlot,
        slot_directory_descriptor: int,
        delivered_file_descriptor: int,
        physical_observation: RunActionDeliveredFilePhysicalObservation,
        exact_file: "_ExactPublishedFile",
        process_snapshot_size_limit_bytes: int,
        _authority: object,
    ) -> None:
        if (
            type(slot) is not RunActionPreparedDeliverySlot
            or type(slot_directory_descriptor) is not int
            or slot_directory_descriptor < 0
            or type(delivered_file_descriptor) is not int
            or delivered_file_descriptor < 0
            or type(physical_observation)
            is not RunActionDeliveredFilePhysicalObservation
            or type(exact_file) is not _ExactPublishedFile
            or type(process_snapshot_size_limit_bytes) is not int
            or process_snapshot_size_limit_bytes <= 0
            or _authority is not _DELIVERED_FILE_LEASE_AUTHORITY
        ):
            raise RunActionActivationDeliveryError(
                "delivered file lease lacks exact descriptor authority"
            )
        self._slot = slot
        self._slot_directory_descriptor = slot_directory_descriptor
        self._delivered_file_descriptor = delivered_file_descriptor
        self._physical_observation = physical_observation
        self._exact_file = exact_file
        self._process_snapshot_size_limit_bytes = process_snapshot_size_limit_bytes
        self._owner_process_id = os.getpid()
        self._closed = False
        self._require_retained_descriptor()

    @property
    def observation(self) -> RunActionDeliveredFilePhysicalObservation:
        self._require_retained_descriptor()
        return self._physical_observation

    def require_final_path(
        self,
        payload: bytes,
    ) -> RunActionDeliveredFilePhysicalObservation:
        """Join the late final pathname and bytes to the retained delivered inode."""

        _require_delivery_inputs(
            self._slot,
            self._slot_directory_descriptor,
            payload,
            self._process_snapshot_size_limit_bytes,
        )
        retained_before = self._observe_retained_descriptor(payload)
        _require_same_published_file(retained_before, self._exact_file)
        _require_exact_slot_directory(
            self._slot,
            self._slot_directory_descriptor,
            self._process_snapshot_size_limit_bytes,
            expected_entries=(self._slot.final_file_name,),
        )
        path_observation = _observe_exact_published_file(
            self._slot,
            self._slot_directory_descriptor,
            payload,
            self._process_snapshot_size_limit_bytes,
        )
        _require_same_published_file(path_observation, retained_before)
        _require_exact_slot_directory(
            self._slot,
            self._slot_directory_descriptor,
            self._process_snapshot_size_limit_bytes,
            expected_entries=(self._slot.final_file_name,),
        )
        retained_after = self._observe_retained_descriptor(payload)
        _require_same_published_file(retained_after, path_observation)
        return _physical_observation(
            self._slot,
            payload,
            retained_after,
        )

    def _observe_retained_descriptor(
        self,
        payload: bytes,
    ) -> "_ExactPublishedFile":
        self._require_retained_descriptor()
        return _observe_exact_published_descriptor(
            self._slot,
            self._delivered_file_descriptor,
            payload,
            self._process_snapshot_size_limit_bytes,
        )

    def _require_retained_descriptor(self) -> None:
        if self._owner_process_id != os.getpid() or self._closed:
            raise RunActionActivationDeliveryError(
                "delivered file lease is closed or belongs to another process"
            )
        metadata = os.fstat(self._delivered_file_descriptor)
        mount_id = read_run_action_descriptor_mount_id(
            self._delivered_file_descriptor,
            self._process_snapshot_size_limit_bytes,
        )
        if (
            _stable_file_metadata(metadata)
            != _stable_file_metadata(self._exact_file.metadata)
            or mount_id != self._exact_file.mount_id
        ):
            raise RunActionActivationDeliveryError(
                "retained delivered file descriptor changed physical state"
            )

    def __enter__(self) -> "RunActionDeliveredFileLease":
        self._require_retained_descriptor()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owner_process_id != os.getpid() or self._closed:
            raise RunActionActivationDeliveryError(
                "delivered file lease is already closed or foreign"
            )
        self._closed = True
        os.close(self._delivered_file_descriptor)


@dataclass(frozen=True)
class _ExactPublishedFile:
    metadata: os.stat_result
    mount_id: int


def publish_or_adopt_run_action_delivery(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    payload: bytes,
    process_snapshot_size_limit_bytes: int,
) -> RunActionDeliveredFileLease:
    """Publish once through O_TMPFILE or adopt the exact completed publication."""

    _require_delivery_inputs(
        slot,
        slot_directory_descriptor,
        payload,
        process_snapshot_size_limit_bytes,
    )
    entries = _require_exact_slot_directory(
        slot,
        slot_directory_descriptor,
        process_snapshot_size_limit_bytes,
    )
    if entries == (slot.final_file_name,):
        return _adopt_published_delivery(
            slot,
            slot_directory_descriptor,
            payload,
            process_snapshot_size_limit_bytes,
        )
    if entries:
        raise RunActionActivationDeliveryError(
            "prepared delivery slot contains an unexpected entry"
        )
    return _publish_anonymous_delivery(
        slot,
        slot_directory_descriptor,
        payload,
        process_snapshot_size_limit_bytes,
    )


def _publish_anonymous_delivery(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    payload: bytes,
    process_snapshot_size_limit_bytes: int,
) -> RunActionDeliveredFileLease:
    with ExitStack() as descriptors:
        descriptor = open_run_action_anonymous_file(
            slot_directory_descriptor,
            _ANONYMOUS_FILE_MODE,
        )
        descriptors.callback(os.close, descriptor)
        initial_metadata = os.fstat(descriptor)
        if (
            initial_metadata.st_uid != slot.owner_user_id
            or initial_metadata.st_gid != slot.owner_group_id
        ):
            os.fchown(
                descriptor,
                slot.owner_user_id,
                slot.owner_group_id,
            )
        write_run_action_full_payload(descriptor, payload)
        require_run_action_descriptor_payload(descriptor, payload)
        os.fchmod(descriptor, _DELIVERED_FILE_MODE)
        os.fsync(descriptor)
        anonymous_file_state = _require_exact_anonymous_file(
            slot,
            descriptor,
            payload,
            process_snapshot_size_limit_bytes,
        )
        link_run_action_anonymous_file_no_replace(
            descriptor,
            slot_directory_descriptor,
            slot.final_file_name,
        )
        linked_metadata = os.fstat(descriptor)
        linked_mount_id = read_run_action_descriptor_mount_id(
            descriptor,
            process_snapshot_size_limit_bytes,
        )
        if (
            _stable_file_identity(linked_metadata)
            != _stable_file_identity(anonymous_file_state.metadata)
            or linked_mount_id != anonymous_file_state.mount_id
            or linked_metadata.st_nlink != 1
        ):
            raise RunActionActivationDeliveryError(
                "linked delivery differs from its anonymous file"
            )
        os.fsync(slot_directory_descriptor)
        _require_exact_slot_directory(
            slot,
            slot_directory_descriptor,
            process_snapshot_size_limit_bytes,
            expected_entries=(slot.final_file_name,),
        )
        reopened = _observe_exact_published_file(
            slot,
            slot_directory_descriptor,
            payload,
            process_snapshot_size_limit_bytes,
        )
        _require_same_published_file(
            reopened,
            _ExactPublishedFile(
                metadata=linked_metadata,
                mount_id=linked_mount_id,
            ),
        )
        _require_exact_slot_directory(
            slot,
            slot_directory_descriptor,
            process_snapshot_size_limit_bytes,
            expected_entries=(slot.final_file_name,),
        )
        final_metadata = os.fstat(descriptor)
        final_mount_id = read_run_action_descriptor_mount_id(
            descriptor,
            process_snapshot_size_limit_bytes,
        )
        if (
            _stable_file_metadata(final_metadata)
            != _stable_file_metadata(reopened.metadata)
            or final_mount_id != reopened.mount_id
        ):
            raise RunActionActivationDeliveryError(
                "published delivery changed after path revalidation"
            )
        retained = _observe_exact_published_descriptor(
            slot,
            descriptor,
            payload,
            process_snapshot_size_limit_bytes,
        )
        _require_same_published_file(retained, reopened)
        lease = _mint_delivered_file_lease(
            slot,
            slot_directory_descriptor,
            descriptor,
            payload,
            retained,
            process_snapshot_size_limit_bytes,
        )
        descriptors.pop_all()
        return lease


def _adopt_published_delivery(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    payload: bytes,
    process_snapshot_size_limit_bytes: int,
) -> RunActionDeliveredFileLease:
    first = _observe_exact_published_file(
        slot,
        slot_directory_descriptor,
        payload,
        process_snapshot_size_limit_bytes,
    )
    synchronized = _synchronize_exact_published_file(
        slot,
        slot_directory_descriptor,
        payload,
        first,
        process_snapshot_size_limit_bytes,
    )
    _require_exact_slot_directory(
        slot,
        slot_directory_descriptor,
        process_snapshot_size_limit_bytes,
        expected_entries=(slot.final_file_name,),
    )
    second = _observe_exact_published_file(
        slot,
        slot_directory_descriptor,
        payload,
        process_snapshot_size_limit_bytes,
    )
    _require_same_published_file(second, first)
    _require_same_published_file(second, synchronized)
    _require_exact_slot_directory(
        slot,
        slot_directory_descriptor,
        process_snapshot_size_limit_bytes,
        expected_entries=(slot.final_file_name,),
    )
    with ExitStack() as descriptors:
        retained_descriptor = _open_published_file(
            slot,
            slot_directory_descriptor,
        )
        descriptors.callback(os.close, retained_descriptor)
        retained = _observe_exact_published_descriptor(
            slot,
            retained_descriptor,
            payload,
            process_snapshot_size_limit_bytes,
        )
        _require_same_published_file(retained, second)
        lease = _mint_delivered_file_lease(
            slot,
            slot_directory_descriptor,
            retained_descriptor,
            payload,
            retained,
            process_snapshot_size_limit_bytes,
        )
        descriptors.pop_all()
        return lease


def _synchronize_exact_published_file(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    payload: bytes,
    expected: _ExactPublishedFile,
    process_snapshot_size_limit_bytes: int,
) -> _ExactPublishedFile:
    descriptor = _open_published_file(
        slot,
        slot_directory_descriptor,
    )
    with os.fdopen(descriptor, "rb", buffering=0) as published_file:
        observed = _observe_exact_published_descriptor(
            slot,
            published_file.fileno(),
            payload,
            process_snapshot_size_limit_bytes,
        )
        _require_same_published_file(observed, expected)
        os.fsync(published_file.fileno())
        synchronized = _observe_exact_published_descriptor(
            slot,
            published_file.fileno(),
            payload,
            process_snapshot_size_limit_bytes,
        )
        _require_same_published_file(synchronized, observed)
        os.fsync(slot_directory_descriptor)
    return synchronized


def _require_delivery_inputs(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    payload: bytes,
    process_snapshot_size_limit_bytes: int,
) -> None:
    if (
        type(slot) is not RunActionPreparedDeliverySlot
        or type(slot_directory_descriptor) is not int
        or slot_directory_descriptor < 0
        or type(payload) is not bytes
        or not payload
        or len(payload) > slot.payload_size_limit_bytes
        or type(process_snapshot_size_limit_bytes) is not int
        or process_snapshot_size_limit_bytes <= 0
    ):
        raise RunActionActivationDeliveryError(
            "activation delivery requires one bounded nonempty payload and exact slot"
        )


def _require_exact_slot_directory(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    process_snapshot_size_limit_bytes: int,
    *,
    expected_entries: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    metadata_before = os.fstat(slot_directory_descriptor)
    mount_id_before = read_run_action_descriptor_mount_id(
        slot_directory_descriptor,
        process_snapshot_size_limit_bytes,
    )
    entries = tuple(sorted(os.listdir(slot_directory_descriptor)))
    metadata_after = os.fstat(slot_directory_descriptor)
    mount_id_after = read_run_action_descriptor_mount_id(
        slot_directory_descriptor,
        process_snapshot_size_limit_bytes,
    )
    if (
        not stat.S_ISDIR(metadata_before.st_mode)
        or _stable_directory_metadata(metadata_before)
        != _stable_directory_metadata(metadata_after)
        or metadata_before.st_uid != slot.owner_user_id
        or metadata_before.st_gid != slot.owner_group_id
        or stat.S_IMODE(metadata_before.st_mode) != slot.mode
        or mount_id_before != slot.mount_id
        or mount_id_after != mount_id_before
        or metadata_before.st_dev != slot.device
        or metadata_before.st_ino != slot.inode
    ):
        raise RunActionActivationDeliveryError(
            "delivery slot directory is unsafe, substituted, or noncanonical"
        )
    if expected_entries is not None and entries != tuple(sorted(expected_entries)):
        raise RunActionActivationDeliveryError(
            "delivery slot directory changed its exact final entry set"
        )
    if expected_entries is None and entries not in ((), (slot.final_file_name,)):
        raise RunActionActivationDeliveryError(
            "prepared delivery slot contains an unexpected entry"
        )
    return entries


def _require_exact_anonymous_file(
    slot: RunActionPreparedDeliverySlot,
    descriptor: int,
    payload: bytes,
    process_snapshot_size_limit_bytes: int,
) -> _ExactPublishedFile:
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
        or _stable_file_metadata(metadata_before)
        != _stable_file_metadata(metadata_after)
        or metadata_before.st_uid != slot.owner_user_id
        or metadata_before.st_gid != slot.owner_group_id
        or stat.S_IMODE(metadata_before.st_mode) != _DELIVERED_FILE_MODE
        or metadata_before.st_nlink != 0
        or metadata_before.st_size != len(payload)
        or mount_id_before != slot.mount_id
        or mount_id_after != mount_id_before
        or metadata_before.st_dev != slot.device
        or metadata_before.st_ino == slot.inode
    ):
        raise RunActionActivationDeliveryError(
            "anonymous delivery file is unsafe or substituted"
        )
    return _ExactPublishedFile(
        metadata=metadata_after,
        mount_id=mount_id_after,
    )


def _observe_exact_published_file(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    payload: bytes,
    process_snapshot_size_limit_bytes: int,
) -> _ExactPublishedFile:
    descriptor = _open_published_file(
        slot,
        slot_directory_descriptor,
    )
    with os.fdopen(descriptor, "rb", buffering=0) as published_file:
        return _observe_exact_published_descriptor(
            slot,
            published_file.fileno(),
            payload,
            process_snapshot_size_limit_bytes,
        )


def _open_published_file(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
) -> int:
    return os.open(
        slot.final_file_name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=slot_directory_descriptor,
    )


def _observe_exact_published_descriptor(
    slot: RunActionPreparedDeliverySlot,
    descriptor: int,
    payload: bytes,
    process_snapshot_size_limit_bytes: int,
) -> _ExactPublishedFile:
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
        or _stable_file_metadata(metadata_before)
        != _stable_file_metadata(metadata_after)
        or metadata_before.st_uid != slot.owner_user_id
        or metadata_before.st_gid != slot.owner_group_id
        or stat.S_IMODE(metadata_before.st_mode) != _DELIVERED_FILE_MODE
        or metadata_before.st_nlink != 1
        or metadata_before.st_size != len(payload)
        or mount_id_before != slot.mount_id
        or mount_id_after != mount_id_before
        or metadata_before.st_dev != slot.device
        or metadata_before.st_ino == slot.inode
    ):
        raise RunActionActivationDeliveryError(
            "published delivery file is unsafe or substituted"
        )
    return _ExactPublishedFile(
        metadata=metadata_after,
        mount_id=mount_id_after,
    )


def _require_same_published_file(
    observed: _ExactPublishedFile,
    expected: _ExactPublishedFile,
) -> None:
    if (
        _stable_file_metadata(observed.metadata)
        != _stable_file_metadata(expected.metadata)
        or observed.mount_id != expected.mount_id
    ):
        raise RunActionActivationDeliveryError(
            "published delivery physical state changed during exact observation"
        )


def _physical_observation(
    slot: RunActionPreparedDeliverySlot,
    payload: bytes,
    observed: _ExactPublishedFile,
) -> RunActionDeliveredFilePhysicalObservation:
    metadata = observed.metadata
    return RunActionDeliveredFilePhysicalObservation(
        prepared_delivery_slot_id=slot.prepared_delivery_slot_id,
        runtime_volume_authority_id=slot.runtime_volume_authority_id,
        generation_nonce=slot.generation_nonce,
        kind=slot.kind,
        relative_path=f"{slot.directory_relative_path}/{slot.final_file_name}",
        file_type="regular",
        owner_user_id=metadata.st_uid,
        owner_group_id=metadata.st_gid,
        mode=stat.S_IMODE(metadata.st_mode),
        link_count=metadata.st_nlink,
        size_bytes=metadata.st_size,
        mount_id=observed.mount_id,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        content_digest=(
            tree_or_blob_digest(payload)
            if slot.kind is RunActionPreparedFileKind.INPUT
            else None
        ),
    )


def _mint_delivered_file_lease(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    delivered_file_descriptor: int,
    payload: bytes,
    exact_file: _ExactPublishedFile,
    process_snapshot_size_limit_bytes: int,
) -> RunActionDeliveredFileLease:
    return RunActionDeliveredFileLease(
        slot=slot,
        slot_directory_descriptor=slot_directory_descriptor,
        delivered_file_descriptor=delivered_file_descriptor,
        physical_observation=_physical_observation(
            slot,
            payload,
            exact_file,
        ),
        exact_file=exact_file,
        process_snapshot_size_limit_bytes=process_snapshot_size_limit_bytes,
        _authority=_DELIVERED_FILE_LEASE_AUTHORITY,
    )


def _stable_directory_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_file_metadata(metadata: os.stat_result) -> tuple[int, ...]:
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
    )


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
    )


__all__ = [
    "RunActionActivationDeliveryError",
    "RunActionDeliveredFileLease",
    "RunActionDeliveredFilePhysicalObservation",
    "publish_or_adopt_run_action_delivery",
]
