"""Descriptor-bound adoption of an absent or already-published event-5 release."""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from threading import get_ident

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
)
from kapso.cross_run.launch.run_action_release_authority import (
    require_run_action_workload_release_receipt_matches_event,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseAdoption,
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    RunActionControlDirectoryLease,
    open_run_action_control_directory,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_descriptor_mount_id,
)
from kapso.cross_run.settings import LaunchSettings

_RELEASE_FILE_NAME = "release"
_RELEASE_FILE_MODE = 0o400
_RUN_ACTION_TIMEOUT_ADOPTION_AUTHORITY = object()
_RUN_ACTION_TIMEOUT_PUBLICATION_DESCRIPTOR_AUTHORITY = object()


class RunActionReleaseAdoptionError(RuntimeError):
    """The event-5 control directory or linked receipt is unsafe."""


class RunActionReleaseInspectionLease:
    """Owner-bound retained proof of one exact semantic control topology."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        control_lease: RunActionControlDirectoryLease,
        topology: RunActionControlDirectoryTopology,
        adoption: RunActionWorkloadReleaseAdoption | None,
        release_descriptor: int | None,
        release_identity: tuple[int, ...] | None,
        release_payload: bytes | None,
    ) -> None:
        if (
            type(descriptors) is not ExitStack
            or type(control_lease) is not RunActionControlDirectoryLease
            or type(topology) is not RunActionControlDirectoryTopology
            or (topology is not RunActionControlDirectoryTopology.EMPTY)
            != (type(adoption) is RunActionWorkloadReleaseAdoption)
            or (topology is not RunActionControlDirectoryTopology.EMPTY)
            != (type(release_descriptor) is int)
            or (topology is not RunActionControlDirectoryTopology.EMPTY)
            != (type(release_identity) is tuple)
            or (topology is not RunActionControlDirectoryTopology.EMPTY)
            != (type(release_payload) is bytes)
        ):
            raise RunActionReleaseAdoptionError(
                "release inspection lease lacks exact retained authority"
            )
        self._descriptors = descriptors
        self._control_lease = control_lease
        self._topology = topology
        self._adoption = adoption
        self._release_descriptor = release_descriptor
        self._release_identity = release_identity
        self._release_payload = release_payload
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._closed = False
        self.require_current()

    @property
    def topology(self) -> RunActionControlDirectoryTopology:
        self.require_current()
        return self._topology

    @property
    def adoption(self) -> RunActionWorkloadReleaseAdoption:
        self.require_current()
        if type(self._adoption) is not RunActionWorkloadReleaseAdoption:
            raise RunActionReleaseAdoptionError(
                "absent release inspection has no adoption"
            )
        return self._adoption

    def require_current(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionReleaseAdoptionError(
                "release inspection lease is closed or foreign"
            )
        self._control_lease.require_current()
        if self._control_lease.topology is not self._topology:
            raise RunActionReleaseAdoptionError(
                "control topology changed during retained release inspection"
            )
        if self._topology is RunActionControlDirectoryTopology.EMPTY:
            return
        before = os.fstat(self._release_descriptor)
        mount_id_before = read_run_action_descriptor_mount_id(self._release_descriptor)
        retained_payload_before = _read_complete_bounded_payload(
            self._release_descriptor,
            len(self._release_payload),
        )
        path_descriptor = os.open(
            _RELEASE_FILE_NAME,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._control_lease._control_descriptor,
        )
        with os.fdopen(path_descriptor, "rb", buffering=0) as path_file:
            path_payload = _read_complete_bounded_payload(
                path_file.fileno(),
                len(self._release_payload),
            )
            path_metadata = os.fstat(path_file.fileno())
            path_mount_id = read_run_action_descriptor_mount_id(path_file.fileno())
        retained_payload_after = _read_complete_bounded_payload(
            self._release_descriptor,
            len(self._release_payload),
        )
        after = os.fstat(self._release_descriptor)
        mount_id_after = read_run_action_descriptor_mount_id(self._release_descriptor)
        if (
            _release_identity(before, mount_id_before) != self._release_identity
            or _release_identity(path_metadata, path_mount_id) != self._release_identity
            or _release_identity(after, mount_id_after) != self._release_identity
            or retained_payload_before != self._release_payload
            or path_payload != self._release_payload
            or retained_payload_after != self._release_payload
        ):
            raise RunActionReleaseAdoptionError(
                "adopted release inode changed or was replaced"
            )
        self._control_lease.require_current()

    def _duplicate_timeout_control_descriptor(
        self,
        *,
        descriptors: ExitStack,
        _authority: object,
    ) -> int:
        self.require_current()
        if (
            self._topology is not RunActionControlDirectoryTopology.TIMED_OUT
            or type(descriptors) is not ExitStack
            or _authority is not _RUN_ACTION_TIMEOUT_ADOPTION_AUTHORITY
        ):
            raise RunActionReleaseAdoptionError(
                "timeout adoption lacks exact retained control authority"
            )
        descriptor = os.dup(self._control_lease._control_descriptor)
        descriptors.callback(os.close, descriptor)
        os.set_inheritable(descriptor, False)
        self.require_current()
        return descriptor

    def _duplicate_timeout_publication_descriptors(
        self,
        *,
        descriptors: ExitStack,
        _authority: object,
    ) -> tuple[int, int]:
        """Duplicate one retained RELEASED control and release occurrence."""

        self.require_current()
        if (
            self._topology is not RunActionControlDirectoryTopology.RELEASED
            or type(self._release_descriptor) is not int
            or type(descriptors) is not ExitStack
            or _authority is not _RUN_ACTION_TIMEOUT_PUBLICATION_DESCRIPTOR_AUTHORITY
        ):
            raise RunActionReleaseAdoptionError(
                "timeout publication lacks exact retained release descriptors"
            )
        control_descriptor = os.dup(self._control_lease._control_descriptor)
        descriptors.callback(os.close, control_descriptor)
        os.set_inheritable(control_descriptor, False)
        release_descriptor = os.dup(self._release_descriptor)
        descriptors.callback(os.close, release_descriptor)
        os.set_inheritable(release_descriptor, False)
        if control_descriptor == release_descriptor:
            raise RunActionReleaseAdoptionError(
                "timeout publication descriptors are not distinct"
            )
        self.require_current()
        return control_descriptor, release_descriptor

    def __enter__(self) -> "RunActionReleaseInspectionLease":
        self.require_current()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionReleaseAdoptionError(
                "release inspection lease is already closed or foreign"
            )
        self._closed = True
        self._descriptors.close()


def open_run_action_release_inspection(
    *,
    activation_event: RunActionExecutionEvent,
    launch_settings: LaunchSettings,
) -> RunActionReleaseInspectionLease:
    """Classify and retain the exact event-5 release inode through its keeper."""

    if (
        type(activation_event) is not RunActionExecutionEvent
        or activation_event.event_number != 5
        or activation_event.event_kind
        is not RunActionExecutionEventKind.ACTIVATION_COMMITTED
        or type(launch_settings) is not LaunchSettings
    ):
        raise RunActionReleaseAdoptionError(
            "release inspection requires one exact durable event 5"
        )
    prepared = activation_event.activation_revalidation_receipt.prepared_execution
    policy_bound = prepared.preparation_claim.execution_policy.supervisor_limits
    configured_bound = launch_settings.run_action_release_receipt_size_bytes
    if (
        policy_bound.release_receipt_size_bytes != configured_bound
        or policy_bound.timeout_directive_size_bytes
        != launch_settings.run_action_timeout_directive_size_bytes
    ):
        raise RunActionReleaseAdoptionError(
            "release inspection policy differs from configured control bounds"
        )
    with ExitStack() as descriptors:
        control_lease = open_run_action_control_directory(prepared)
        descriptors.callback(control_lease.close)
        topology = control_lease.topology
        if topology is RunActionControlDirectoryTopology.EMPTY:
            inspection = RunActionReleaseInspectionLease(
                descriptors=descriptors,
                control_lease=control_lease,
                topology=topology,
                adoption=None,
                release_descriptor=None,
                release_identity=None,
                release_payload=None,
            )
            inspection._descriptors = descriptors.pop_all()
            return inspection
        release_descriptor = os.open(
            _RELEASE_FILE_NAME,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=control_lease._control_descriptor,
        )
        descriptors.callback(os.close, release_descriptor)
        metadata_before = os.fstat(release_descriptor)
        mount_id_before = read_run_action_descriptor_mount_id(release_descriptor)
        payload = _read_complete_bounded_payload(
            release_descriptor,
            configured_bound,
        )
        metadata_after = os.fstat(release_descriptor)
        mount_id_after = read_run_action_descriptor_mount_id(release_descriptor)
        identity = _release_identity(metadata_after, mount_id_after)
        control = prepared.control_directory
        if (
            not stat.S_ISREG(metadata_before.st_mode)
            or identity != _release_identity(metadata_before, mount_id_before)
            or metadata_before.st_uid != control.owner_user_id
            or metadata_before.st_gid != control.owner_group_id
            or stat.S_IMODE(metadata_before.st_mode) != _RELEASE_FILE_MODE
            or metadata_before.st_nlink != 1
            or not 0 < metadata_before.st_size <= configured_bound
            or len(payload) != metadata_before.st_size
            or mount_id_before != control.mount_id
            or mount_id_after != mount_id_before
            or metadata_before.st_dev != control.device
            or metadata_before.st_ino == control.inode
        ):
            raise RunActionReleaseAdoptionError(
                "published release path has unsafe physical identity"
            )
        receipt = RunActionWorkloadReleaseReceipt.from_json_bytes(payload)
        if receipt.to_json_bytes() != payload:
            raise RunActionReleaseAdoptionError(
                "published release bytes are not canonical"
            )
        require_run_action_workload_release_receipt_matches_event(
            receipt,
            activation_event,
        )
        control_lease.require_current()
        os.fsync(release_descriptor)
        os.fsync(control_lease._control_descriptor)
        control_lease.require_current()
        adoption = RunActionWorkloadReleaseAdoption.mint(
            workload_release_receipt=receipt,
            control_mount_id=control.mount_id,
            control_device=control.device,
            control_inode=control.inode,
            owner_user_id=metadata_after.st_uid,
            owner_group_id=metadata_after.st_gid,
            mode=stat.S_IMODE(metadata_after.st_mode),
            link_count=metadata_after.st_nlink,
            size_bytes=metadata_after.st_size,
            content_digest=tree_or_blob_digest(payload),
            release_mount_id=mount_id_after,
            release_device=metadata_after.st_dev,
            release_inode=metadata_after.st_ino,
        )
        inspection = RunActionReleaseInspectionLease(
            descriptors=descriptors,
            control_lease=control_lease,
            topology=topology,
            adoption=adoption,
            release_descriptor=release_descriptor,
            release_identity=identity,
            release_payload=payload,
        )
        inspection._descriptors = descriptors.pop_all()
    return inspection


def _read_complete_bounded_payload(descriptor: int, size_limit: int) -> bytes:
    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(size_limit) is not int
        or size_limit <= 0
    ):
        raise RunActionReleaseAdoptionError(
            "release payload read lacks a positive exact bound"
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    remaining = size_limit + 1
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if not payload or len(payload) > size_limit:
        raise RunActionReleaseAdoptionError(
            "release payload is empty or exceeds its configured bound"
        )
    return payload


def _release_identity(
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


__all__ = [
    "RunActionReleaseAdoptionError",
    "RunActionReleaseInspectionLease",
    "open_run_action_release_inspection",
]
