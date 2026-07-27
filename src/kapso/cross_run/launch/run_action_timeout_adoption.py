"""Retained read-only adoption of an already-published timeout directive."""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from threading import get_ident

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    _RUN_ACTION_TIMEOUT_ADOPTION_AUTHORITY,
    _RUN_ACTION_TIMEOUT_PUBLICATION_DESCRIPTOR_AUTHORITY,
    open_run_action_release_inspection,
    RunActionReleaseInspectionLease,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseAdoption,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_descriptor_mount_id,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_timeout_publication_evidence_matches,
    RunActionTimeoutDirective,
    RunActionTimeoutDirectivePublicationReceipt,
)
from kapso.cross_run.settings import LaunchSettings

_TIMEOUT_FILE_NAME = "timeout"
_TIMEOUT_FILE_MODE = 0o400
_TIMEOUT_INSPECTION_AUTHORITY = object()
_RUN_ACTION_TIMEOUT_PUBLICATION_AUTHORITY = object()


class RunActionTimeoutAdoptionError(RuntimeError):
    """The semantic control topology or timeout inode is unsafe."""


class RunActionTimeoutInspectionLease:
    """Retain one exact empty, released, or timed-out control occurrence."""

    def __init__(
        self,
        *,
        descriptors: ExitStack,
        release_inspection: RunActionReleaseInspectionLease,
        activation_event: RunActionExecutionEvent,
        topology: RunActionControlDirectoryTopology,
        timeout_directive_publication: (
            RunActionTimeoutDirectivePublicationReceipt | None
        ),
        control_descriptor: int | None,
        timeout_descriptor: int | None,
        timeout_identity: tuple[int, ...] | None,
        timeout_payload: bytes | None,
        process_snapshot_size_limit_bytes: int,
        _authority: object,
    ) -> None:
        timed_out = topology is RunActionControlDirectoryTopology.TIMED_OUT
        if (
            type(descriptors) is not ExitStack
            or type(release_inspection) is not RunActionReleaseInspectionLease
            or type(activation_event) is not RunActionExecutionEvent
            or type(topology) is not RunActionControlDirectoryTopology
            or release_inspection.topology is not topology
            or timed_out
            != (
                type(timeout_directive_publication)
                is RunActionTimeoutDirectivePublicationReceipt
            )
            or timed_out != (type(control_descriptor) is int)
            or timed_out != (type(timeout_descriptor) is int)
            or timed_out != (type(timeout_identity) is tuple)
            or timed_out != (type(timeout_payload) is bytes)
            or type(process_snapshot_size_limit_bytes) is not int
            or process_snapshot_size_limit_bytes <= 0
            or _authority is not _TIMEOUT_INSPECTION_AUTHORITY
        ):
            raise RunActionTimeoutAdoptionError(
                "timeout inspection lacks exact retained authority"
            )
        if timed_out and (
            timeout_directive_publication.timeout_directive.to_json_bytes()
            != timeout_payload
            or not run_action_timeout_publication_evidence_matches(
                timeout_directive_publication,
                activation_event.event_id,
                activation_event.activation_revalidation_receipt,
                release_inspection.adoption,
            )
        ):
            raise RunActionTimeoutAdoptionError(
                "timeout inspection carries a spliced publication"
            )
        self._descriptors = descriptors
        self._release_inspection = release_inspection
        self._activation_event = activation_event
        self._topology = topology
        self._timeout_directive_publication = timeout_directive_publication
        self._control_descriptor = control_descriptor
        self._timeout_descriptor = timeout_descriptor
        self._timeout_identity = timeout_identity
        self._timeout_payload = timeout_payload
        self._process_snapshot_size_limit_bytes = process_snapshot_size_limit_bytes
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._closed = False
        self.require_current()

    @property
    def topology(self) -> RunActionControlDirectoryTopology:
        self.require_current()
        return self._topology

    @property
    def activation_event(self) -> RunActionExecutionEvent:
        self.require_current()
        return self._activation_event

    @property
    def workload_release_adoption(self) -> RunActionWorkloadReleaseAdoption | None:
        self.require_current()
        if self._topology is RunActionControlDirectoryTopology.EMPTY:
            return None
        return self._release_inspection.adoption

    @property
    def timeout_directive_publication(
        self,
    ) -> RunActionTimeoutDirectivePublicationReceipt | None:
        self.require_current()
        return self._timeout_directive_publication

    def require_current(self) -> None:
        if (
            self._closed
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
        ):
            raise RunActionTimeoutAdoptionError(
                "timeout inspection lease is closed or foreign"
            )
        self._release_inspection.require_current()
        if self._release_inspection.topology is not self._topology:
            raise RunActionTimeoutAdoptionError(
                "control topology changed during retained timeout inspection"
            )
        if self._topology is not RunActionControlDirectoryTopology.TIMED_OUT:
            return
        before = os.fstat(self._timeout_descriptor)
        mount_id_before = read_run_action_descriptor_mount_id(
            self._timeout_descriptor,
            self._process_snapshot_size_limit_bytes,
        )
        retained_payload_before = _read_complete_bounded_payload(
            self._timeout_descriptor,
            len(self._timeout_payload),
        )
        path_descriptor = os.open(
            _TIMEOUT_FILE_NAME,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._control_descriptor,
        )
        with os.fdopen(path_descriptor, "rb", buffering=0) as path_file:
            path_payload = _read_complete_bounded_payload(
                path_file.fileno(),
                len(self._timeout_payload),
            )
            path_metadata = os.fstat(path_file.fileno())
            path_mount_id = read_run_action_descriptor_mount_id(
                path_file.fileno(),
                self._process_snapshot_size_limit_bytes,
            )
        retained_payload_after = _read_complete_bounded_payload(
            self._timeout_descriptor,
            len(self._timeout_payload),
        )
        after = os.fstat(self._timeout_descriptor)
        mount_id_after = read_run_action_descriptor_mount_id(
            self._timeout_descriptor,
            self._process_snapshot_size_limit_bytes,
        )
        if (
            _timeout_identity(before, mount_id_before) != self._timeout_identity
            or _timeout_identity(path_metadata, path_mount_id) != self._timeout_identity
            or _timeout_identity(after, mount_id_after) != self._timeout_identity
            or retained_payload_before != self._timeout_payload
            or path_payload != self._timeout_payload
            or retained_payload_after != self._timeout_payload
        ):
            raise RunActionTimeoutAdoptionError(
                "adopted timeout inode changed or was replaced"
            )
        if (
            self._timeout_directive_publication.timeout_directive.to_json_bytes()
            != self._timeout_payload
            or not run_action_timeout_publication_evidence_matches(
                self._timeout_directive_publication,
                self._activation_event.event_id,
                self._activation_event.activation_revalidation_receipt,
                self._release_inspection.adoption,
            )
        ):
            raise RunActionTimeoutAdoptionError(
                "retained timeout publication is no longer exact"
            )
        self._release_inspection.require_current()

    def _duplicate_timeout_publication_descriptors(
        self,
        *,
        descriptors: ExitStack,
        _authority: object,
    ) -> tuple[int, int]:
        """Duplicate the retained released control and predecessor descriptors."""

        self.require_current()
        if (
            self._topology is not RunActionControlDirectoryTopology.RELEASED
            or self._timeout_directive_publication is not None
            or type(self._release_inspection._adoption)
            is not RunActionWorkloadReleaseAdoption
            or type(descriptors) is not ExitStack
            or _authority is not _RUN_ACTION_TIMEOUT_PUBLICATION_AUTHORITY
        ):
            raise RunActionTimeoutAdoptionError(
                "timeout publication lacks one exact retained release"
            )
        duplicated_control, duplicated_release = (
            self._release_inspection._duplicate_timeout_publication_descriptors(
                descriptors=descriptors,
                _authority=_RUN_ACTION_TIMEOUT_PUBLICATION_DESCRIPTOR_AUTHORITY,
            )
        )
        self.require_current()
        return duplicated_control, duplicated_release

    def __enter__(self) -> "RunActionTimeoutInspectionLease":
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
            raise RunActionTimeoutAdoptionError(
                "timeout inspection lease is already closed or foreign"
            )
        self._closed = True
        self._descriptors.close()


def open_run_action_timeout_inspection(
    *,
    activation_event: RunActionExecutionEvent,
    launch_settings: LaunchSettings,
) -> RunActionTimeoutInspectionLease:
    """Adopt the exact semantic control topology before adapter inspection."""

    if (
        type(activation_event) is not RunActionExecutionEvent
        or type(launch_settings) is not LaunchSettings
    ):
        raise RunActionTimeoutAdoptionError(
            "timeout inspection requires exact event and launch settings"
        )
    with ExitStack() as descriptors:
        release_inspection = open_run_action_release_inspection(
            activation_event=activation_event,
            launch_settings=launch_settings,
        )
        descriptors.callback(release_inspection.close)
        topology = release_inspection.topology
        if topology is not RunActionControlDirectoryTopology.TIMED_OUT:
            inspection = RunActionTimeoutInspectionLease(
                descriptors=descriptors,
                release_inspection=release_inspection,
                activation_event=activation_event,
                topology=topology,
                timeout_directive_publication=None,
                control_descriptor=None,
                timeout_descriptor=None,
                timeout_identity=None,
                timeout_payload=None,
                process_snapshot_size_limit_bytes=(
                    launch_settings.run_action_process_snapshot_size_bytes
                ),
                _authority=_TIMEOUT_INSPECTION_AUTHORITY,
            )
            inspection._descriptors = descriptors.pop_all()
            return inspection
        adoption = release_inspection.adoption
        activation = activation_event.activation_revalidation_receipt
        prepared = activation.prepared_execution
        control = prepared.control_directory
        authority = prepared.runtime_volume_authority
        timeout_size_bound = (
            prepared.preparation_claim.execution_policy.supervisor_limits.timeout_directive_size_bytes
        )
        if (
            timeout_size_bound
            != launch_settings.run_action_timeout_directive_size_bytes
        ):
            raise RunActionTimeoutAdoptionError(
                "timeout inspection policy differs from configured bound"
            )
        control_descriptor = release_inspection._duplicate_timeout_control_descriptor(
            descriptors=descriptors,
            _authority=_RUN_ACTION_TIMEOUT_ADOPTION_AUTHORITY,
        )
        timeout_descriptor = os.open(
            _TIMEOUT_FILE_NAME,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=control_descriptor,
        )
        descriptors.callback(os.close, timeout_descriptor)
        metadata_before = os.fstat(timeout_descriptor)
        mount_id_before = read_run_action_descriptor_mount_id(
            timeout_descriptor,
            launch_settings.run_action_process_snapshot_size_bytes,
        )
        payload = _read_complete_bounded_payload(
            timeout_descriptor,
            timeout_size_bound,
        )
        metadata_after = os.fstat(timeout_descriptor)
        mount_id_after = read_run_action_descriptor_mount_id(
            timeout_descriptor,
            launch_settings.run_action_process_snapshot_size_bytes,
        )
        identity = _timeout_identity(metadata_after, mount_id_after)
        if (
            not stat.S_ISREG(metadata_before.st_mode)
            or _timeout_identity(metadata_before, mount_id_before) != identity
            or metadata_before.st_uid != authority.owner_user_id
            or metadata_before.st_gid != authority.owner_group_id
            or stat.S_IMODE(metadata_before.st_mode) != _TIMEOUT_FILE_MODE
            or metadata_before.st_nlink != 1
            or not 0 < metadata_before.st_size <= timeout_size_bound
            or len(payload) != metadata_before.st_size
            or mount_id_before != control.mount_id
            or mount_id_after != mount_id_before
            or metadata_before.st_dev != control.device
            or len(
                {
                    control.inode,
                    adoption.release_inode,
                    metadata_before.st_ino,
                }
            )
            != 3
        ):
            raise RunActionTimeoutAdoptionError(
                "published timeout path has unsafe physical identity"
            )
        directive = RunActionTimeoutDirective.from_json_bytes(payload)
        if directive.to_json_bytes() != payload:
            raise RunActionTimeoutAdoptionError(
                "published timeout bytes are not canonical"
            )
        publication = RunActionTimeoutDirectivePublicationReceipt.mint(
            timeout_directive=directive,
            workload_release_adoption_id=adoption.workload_release_adoption_id,
            prepared_control_directory_id=control.prepared_runtime_directory_id,
            control_mount_id=control.mount_id,
            control_device=control.device,
            control_inode=control.inode,
            release_mount_id=adoption.release_mount_id,
            release_device=adoption.release_device,
            release_inode=adoption.release_inode,
            relative_path="control/timeout",
            file_type="regular",
            owner_user_id=metadata_after.st_uid,
            owner_group_id=metadata_after.st_gid,
            mode=stat.S_IMODE(metadata_after.st_mode),
            link_count=metadata_after.st_nlink,
            size_bytes=metadata_after.st_size,
            content_digest=tree_or_blob_digest(payload),
            timeout_mount_id=mount_id_after,
            timeout_device=metadata_after.st_dev,
            timeout_inode=metadata_after.st_ino,
        )
        if not run_action_timeout_publication_evidence_matches(
            publication,
            activation_event.event_id,
            activation,
            adoption,
        ):
            raise RunActionTimeoutAdoptionError(
                "published timeout differs from durable activation and release"
            )
        release_inspection.require_current()
        os.fsync(timeout_descriptor)
        os.fsync(control_descriptor)
        release_inspection.require_current()
        inspection = RunActionTimeoutInspectionLease(
            descriptors=descriptors,
            release_inspection=release_inspection,
            activation_event=activation_event,
            topology=topology,
            timeout_directive_publication=publication,
            control_descriptor=control_descriptor,
            timeout_descriptor=timeout_descriptor,
            timeout_identity=identity,
            timeout_payload=payload,
            process_snapshot_size_limit_bytes=(
                launch_settings.run_action_process_snapshot_size_bytes
            ),
            _authority=_TIMEOUT_INSPECTION_AUTHORITY,
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
        raise RunActionTimeoutAdoptionError(
            "timeout payload read lacks a positive exact bound"
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
        raise RunActionTimeoutAdoptionError(
            "timeout payload is empty or exceeds its configured bound"
        )
    return payload


def _timeout_identity(
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
    "RunActionTimeoutAdoptionError",
    "RunActionTimeoutInspectionLease",
    "open_run_action_timeout_inspection",
]
