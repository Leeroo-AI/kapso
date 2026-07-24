"""Private descriptor-safe mechanics for the run-state publisher."""

from __future__ import annotations

import fcntl
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kapso.cross_run.canonical import content_id
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointHead,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.settings import LaunchSettings

_STAGING_ENTRY_PATTERN = re.compile(r"^checkpoint-[0-9a-f]{64}-[0-9a-f]{32}[.]tmp$")


class RunCheckpointControlError(RuntimeError):
    """The protected checkpoint control plane is unsafe or inconsistent."""


@dataclass(frozen=True)
class _CheckpointFrontierInspection:
    """One locked, non-mutating observation of checkpoint and journal state."""

    head: RunCheckpointHead
    checkpoint: RunCheckpoint | None
    journal_size_bytes: int
    incomplete_tail: bytes
    journal_descriptor: int
    checkpoint_ahead: bool


class _RunCheckpointControl:
    """Private checkpoint mechanics used only under the publisher's shared lock."""

    def __init__(
        self,
        authority: ActiveLaunchWorkspace,
        settings: LaunchSettings,
    ) -> None:
        if type(authority) is not ActiveLaunchWorkspace:
            raise RunCheckpointControlError(
                "run checkpoint store requires active launch authority"
            )
        if type(settings) is not LaunchSettings:
            raise RunCheckpointControlError(
                "run checkpoint store requires exact launch settings"
            )
        authority.require_control_authority()
        if settings != authority._prepared._builder_verifier._settings.launch:
            raise RunCheckpointControlError(
                "run checkpoint settings differ from the active launch"
            )
        self._authority = authority
        self._settings = settings
        self._checkpoint_relative = _require_control_path(
            settings.run_checkpoint_path,
            "run checkpoint path",
        )
        self._journal_relative = _require_control_path(
            settings.run_checkpoint_journal_path,
            "run checkpoint journal path",
        )
        self._lock_relative = _require_control_path(
            settings.run_checkpoint_lock_path,
            "run checkpoint lock path",
        )
        self._staging_relative = _require_control_path(
            settings.run_checkpoint_staging_path,
            "run checkpoint staging path",
        )
        if not (
            self._checkpoint_relative.parent
            == self._journal_relative.parent
            == self._lock_relative.parent
            == self._staging_relative.parent
        ):
            raise RunCheckpointControlError(
                "run checkpoint controls require one shared parent"
            )
        self._control_parent_relative = self._checkpoint_relative.parent
        self._control_parent_identity: tuple[int, int] | None = None
        self._staging_identity: tuple[int, int] | None = None
        receipt = authority.bootstrap_pin.installation_receipt
        if receipt.launch_settings_id != content_id(
            "launch-settings",
            settings.to_dict(),
        ):
            raise RunCheckpointControlError(
                "run checkpoint settings differ from the bootstrap receipt"
            )
        self._journal_identity = (
            receipt.run_checkpoint_journal_device,
            receipt.run_checkpoint_journal_inode,
        )
        self._lock_identity = (
            receipt.run_checkpoint_lock_device,
            receipt.run_checkpoint_lock_inode,
        )
        with ExitStack() as descriptors:
            parent_descriptor = self._open_control_parent(descriptors)
            self._control_parent_identity = _directory_identity(
                parent_descriptor,
                "run checkpoint control parent",
            )
            self._ensure_staging_directory(parent_descriptor)
            self._open_locked(parent_descriptor, descriptors)
            self._clean_staging(parent_descriptor, descriptors)

    def _open_control_parent(self, descriptors: ExitStack) -> int:
        root_descriptor = _open_absolute_directory(
            self._authority.run_root,
            descriptors,
        )
        if (
            _directory_identity(root_descriptor, "active run root")
            != self._authority.published_root_identity
        ):
            raise RunCheckpointControlError(
                "active run root changed before checkpoint access"
            )
        descriptor = root_descriptor
        for name in self._control_parent_relative.parts:
            child_descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            descriptors.callback(os.close, child_descriptor)
            _require_private_directory(
                child_descriptor,
                "run checkpoint control parent",
            )
            descriptor = child_descriptor
        identity = _directory_identity(
            descriptor,
            "run checkpoint control parent",
        )
        if (
            self._control_parent_identity is not None
            and identity != self._control_parent_identity
        ):
            raise RunCheckpointControlError(
                "run checkpoint control parent was replaced"
            )
        return descriptor

    def _ensure_staging_directory(self, parent_descriptor: int) -> None:
        if not os.access(
            self._staging_relative.name,
            os.F_OK,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        ):
            os.mkdir(
                self._staging_relative.name,
                mode=0o700,
                dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        descriptor = os.open(
            self._staging_relative.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        with ExitStack() as descriptors:
            descriptors.callback(os.close, descriptor)
            _require_private_directory(
                descriptor,
                "run checkpoint staging directory",
            )
            identity = _directory_identity(
                descriptor,
                "run checkpoint staging directory",
            )
        if self._staging_identity is None:
            self._staging_identity = identity
        elif self._staging_identity != identity:
            raise RunCheckpointControlError(
                "run checkpoint staging directory was replaced"
            )

    def _open_staging(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> int:
        descriptor = os.open(
            self._staging_relative.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        _require_private_directory(
            descriptor,
            "run checkpoint staging directory",
        )
        if (
            _directory_identity(
                descriptor,
                "run checkpoint staging directory",
            )
            != self._staging_identity
        ):
            raise RunCheckpointControlError(
                "run checkpoint staging directory was replaced"
            )
        return descriptor

    def _open_locked(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> int:
        descriptor = os.open(
            self._lock_relative.name,
            os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != 0
            or (metadata.st_dev, metadata.st_ino) != self._lock_identity
        ):
            raise RunCheckpointControlError(
                "run checkpoint lock must be one owner-private file"
            )
        identity = (metadata.st_dev, metadata.st_ino)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        descriptors.callback(fcntl.flock, descriptor, fcntl.LOCK_UN)
        rebound = os.stat(
            self._lock_relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(rebound.st_mode)
            or (rebound.st_dev, rebound.st_ino) != identity
            or stat.S_IMODE(rebound.st_mode) != 0o600
            or rebound.st_uid != os.geteuid()
            or rebound.st_nlink != 1
            or rebound.st_size != 0
        ):
            raise RunCheckpointControlError(
                "run checkpoint lock changed while acquiring authority"
            )
        return descriptor

    def _clean_staging(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> None:
        staging_descriptor = self._open_staging(
            parent_descriptor,
            descriptors,
        )
        with os.scandir(staging_descriptor) as iterator:
            entries = tuple(iterator)
        if len(entries) > self._settings.run_checkpoint_staging_entry_limit:
            raise RunCheckpointControlError(
                "run checkpoint staging exceeds its entry bound"
            )
        for entry in entries:
            if _STAGING_ENTRY_PATTERN.fullmatch(entry.name) is None:
                raise RunCheckpointControlError(
                    "run checkpoint staging contains an unexpected entry"
                )
            metadata = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
                or metadata.st_size > self._settings.run_checkpoint_size_bytes
            ):
                raise RunCheckpointControlError(
                    "run checkpoint staging entry is unsafe"
                )
            os.unlink(entry.name, dir_fd=staging_descriptor)
        if entries:
            os.fsync(staging_descriptor)

    def _inspect_frontier(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> _CheckpointFrontierInspection:
        """Inspect checkpoint and journal without repairing any crash seam."""
        head, tail, complete_size, journal_descriptor = self._read_journal(
            parent_descriptor,
            descriptors,
        )
        checkpoint = self._read_checkpoint(parent_descriptor, descriptors)
        if checkpoint is None:
            if head.checkpoint is not None:
                raise RunCheckpointControlError(
                    "durable checkpoint journal names an absent checkpoint"
                )
            if tail:
                raise RunCheckpointControlError(
                    "fresh checkpoint journal has an incomplete record"
                )
            return _CheckpointFrontierInspection(
                head=head,
                checkpoint=None,
                journal_size_bytes=complete_size,
                incomplete_tail=b"",
                journal_descriptor=journal_descriptor,
                checkpoint_ahead=False,
            )
        if head.checkpoint == checkpoint:
            if tail:
                raise RunCheckpointControlError(
                    "checkpoint journal has an impossible incomplete record"
                )
            head.require_checkpoint(checkpoint)
            return _CheckpointFrontierInspection(
                head=head,
                checkpoint=checkpoint,
                journal_size_bytes=complete_size,
                incomplete_tail=b"",
                journal_descriptor=journal_descriptor,
                checkpoint_ahead=False,
            )
        expected_predecessor_id = (
            None if head.checkpoint is None else head.checkpoint.run_checkpoint_id
        )
        expected_sequence = (
            0 if head.checkpoint is None else head.checkpoint.checkpoint_sequence + 1
        )
        if (
            checkpoint.predecessor_checkpoint_id != expected_predecessor_id
            or checkpoint.checkpoint_sequence != expected_sequence
        ):
            raise RunCheckpointControlError(
                "run checkpoint was rolled back or skipped its journal frontier"
            )
        reconciled = head.advance(checkpoint)
        expected_record = reconciled.to_json_bytes() + b"\n"
        if tail and (
            len(tail) >= len(expected_record) or not expected_record.startswith(tail)
        ):
            raise RunCheckpointControlError(
                "checkpoint journal tail is not the exact recovery record prefix"
            )
        return _CheckpointFrontierInspection(
            head=head,
            checkpoint=checkpoint,
            journal_size_bytes=complete_size,
            incomplete_tail=tail,
            journal_descriptor=journal_descriptor,
            checkpoint_ahead=True,
        )

    def _recover_checkpoint_ahead(
        self,
        parent_descriptor: int,
        inspection: _CheckpointFrontierInspection,
        descriptors: ExitStack,
    ) -> _CheckpointFrontierInspection:
        """Repair only a bundle-validated adjacent checkpoint successor."""
        if (
            type(inspection) is not _CheckpointFrontierInspection
            or not inspection.checkpoint_ahead
            or inspection.checkpoint is None
        ):
            raise RunCheckpointControlError(
                "checkpoint recovery requires one adjacent successor"
            )
        observed = self._inspect_frontier(parent_descriptor, descriptors)
        if not _same_frontier(observed, inspection):
            raise RunCheckpointControlError("checkpoint frontier moved before recovery")
        reconciled = inspection.head.advance(inspection.checkpoint)
        expected_record = reconciled.to_json_bytes() + b"\n"
        staging_descriptor = self._open_staging(parent_descriptor, descriptors)
        os.fsync(parent_descriptor)
        os.fsync(staging_descriptor)
        if inspection.incomplete_tail:
            os.ftruncate(
                inspection.journal_descriptor,
                inspection.journal_size_bytes,
            )
            os.fsync(inspection.journal_descriptor)
        self._append_record(inspection.journal_descriptor, expected_record)
        durable = self._inspect_frontier(parent_descriptor, descriptors)
        if (
            durable.checkpoint_ahead
            or durable.head != reconciled
            or durable.checkpoint != inspection.checkpoint
            or durable.incomplete_tail
        ):
            raise RunCheckpointControlError(
                "reconciled checkpoint journal differs from its frontier"
            )
        durable.head.require_checkpoint(inspection.checkpoint)
        return durable

    def _commit_checkpoint(
        self,
        parent_descriptor: int,
        expected: _CheckpointFrontierInspection,
        candidate: RunCheckpoint,
        descriptors: ExitStack,
    ) -> _CheckpointFrontierInspection:
        """Replace a checkpoint and append its journal head under the caller's lock."""
        if (
            type(expected) is not _CheckpointFrontierInspection
            or expected.checkpoint_ahead
            or expected.incomplete_tail
            or type(candidate) is not RunCheckpoint
        ):
            raise RunCheckpointControlError(
                "checkpoint commit requires one reconciled predecessor"
            )
        candidate.require_bootstrap_pin(self._authority.bootstrap_pin)
        candidate.require_predecessor(expected.checkpoint)
        payload = candidate.to_json_bytes()
        if len(payload) > self._settings.run_checkpoint_size_bytes:
            raise RunCheckpointControlError(
                "run checkpoint exceeds its configured bound"
            )
        current = self._inspect_frontier(parent_descriptor, descriptors)
        if not _same_frontier(current, expected):
            raise RunCheckpointControlError(
                "run checkpoint compare-and-swap frontier moved"
            )
        successor_head = expected.head.advance(candidate)
        self._require_journal_append_capacity(
            parent_descriptor,
            expected.head,
            successor_head.to_json_bytes() + b"\n",
            descriptors,
        )
        self._write_checkpoint(
            parent_descriptor,
            candidate,
            payload,
            descriptors,
        )
        persisted = self._read_checkpoint(parent_descriptor, descriptors)
        if persisted != candidate:
            raise RunCheckpointControlError(
                "persisted run checkpoint differs from its candidate"
            )
        self._append_head(
            parent_descriptor,
            expected.head,
            successor_head,
            descriptors,
        )
        durable = self._inspect_frontier(parent_descriptor, descriptors)
        if (
            durable.checkpoint_ahead
            or durable.checkpoint != candidate
            or durable.head != successor_head
            or durable.incomplete_tail
        ):
            raise RunCheckpointControlError(
                "durable checkpoint and journal differ after commit"
            )
        return durable

    def _read_journal(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> tuple[RunCheckpointHead, bytes, int, int]:
        descriptor = os.open(
            self._journal_relative.name,
            os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        handle = descriptors.enter_context(os.fdopen(descriptor, "r+b", buffering=0))
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size == 0
            or metadata.st_size > self._settings.run_checkpoint_journal_size_bytes
            or (metadata.st_dev, metadata.st_ino) != self._journal_identity
        ):
            raise RunCheckpointControlError(
                "checkpoint journal must be one pinned, bounded private file"
            )
        identity = (metadata.st_dev, metadata.st_ino)
        handle.seek(0)
        payload = handle.read(self._settings.run_checkpoint_journal_size_bytes + 1)
        reopened = os.fstat(handle.fileno())
        rebound = os.stat(
            self._journal_relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            len(payload) > self._settings.run_checkpoint_journal_size_bytes
            or (
                reopened.st_dev,
                reopened.st_ino,
                reopened.st_size,
                stat.S_IMODE(reopened.st_mode),
            )
            != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                stat.S_IMODE(metadata.st_mode),
            )
            or (rebound.st_dev, rebound.st_ino) != identity
        ):
            raise RunCheckpointControlError("checkpoint journal changed while reading")
        complete_size = payload.rfind(b"\n") + 1
        if complete_size == 0:
            raise RunCheckpointControlError(
                "checkpoint journal lacks its complete initial record"
            )
        records = tuple(
            RunCheckpointHead.from_json_bytes(line)
            for line in payload[: complete_size - 1].split(b"\n")
        )
        if (
            not records
            or records[0] != RunCheckpointHead.initial(self._authority.bootstrap_pin)
            or any(
                record.to_json_bytes() != line
                for record, line in zip(
                    records,
                    payload[: complete_size - 1].split(b"\n"),
                    strict=True,
                )
            )
        ):
            raise RunCheckpointControlError(
                "checkpoint journal initial record or canonical bytes are invalid"
            )
        for previous, current in zip(records, records[1:], strict=False):
            if (
                current.checkpoint is None
                or previous.advance(current.checkpoint) != current
            ):
                raise RunCheckpointControlError(
                    "checkpoint journal lineage is not exact"
                )
        final_binding = os.stat(
            self._journal_relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (final_binding.st_dev, final_binding.st_ino) != identity:
            raise RunCheckpointControlError(
                "checkpoint journal was rebound while validating"
            )
        return records[-1], payload[complete_size:], complete_size, handle.fileno()

    def _read_checkpoint(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> RunCheckpoint | None:
        if not os.access(
            self._checkpoint_relative.name,
            os.F_OK,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        ):
            return None
        descriptor = os.open(
            self._checkpoint_relative.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        handle = descriptors.enter_context(os.fdopen(descriptor, "rb"))
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size > self._settings.run_checkpoint_size_bytes
        ):
            raise RunCheckpointControlError(
                "run checkpoint must be one bounded owner-private file"
            )
        payload = handle.read(self._settings.run_checkpoint_size_bytes + 1)
        reopened = os.fstat(handle.fileno())
        rebound = os.stat(
            self._checkpoint_relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            len(payload) > self._settings.run_checkpoint_size_bytes
            or (
                reopened.st_dev,
                reopened.st_ino,
                reopened.st_size,
                stat.S_IMODE(reopened.st_mode),
            )
            != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                stat.S_IMODE(metadata.st_mode),
            )
            or (rebound.st_dev, rebound.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise RunCheckpointControlError("run checkpoint changed while reading")
        checkpoint = RunCheckpoint.from_json_bytes(payload)
        if payload != checkpoint.to_json_bytes():
            raise RunCheckpointControlError("run checkpoint bytes are not canonical")
        checkpoint.require_bootstrap_pin(self._authority.bootstrap_pin)
        final_binding = os.stat(
            self._checkpoint_relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (final_binding.st_dev, final_binding.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise RunCheckpointControlError(
                "run checkpoint was rebound while validating"
            )
        return checkpoint

    def _append_head(
        self,
        parent_descriptor: int,
        expected: RunCheckpointHead,
        head: RunCheckpointHead,
        descriptors: ExitStack,
    ) -> None:
        if (
            type(expected) is not RunCheckpointHead
            or type(head) is not RunCheckpointHead
            or head.checkpoint is None
            or head.bootstrap_pin_id != self._authority.bootstrap_pin.bootstrap_pin_id
            or expected.advance(head.checkpoint) != head
        ):
            raise RunCheckpointControlError(
                "checkpoint journal record is not the exact successor"
            )
        current, tail, _, journal_descriptor = self._read_journal(
            parent_descriptor,
            descriptors,
        )
        if current != expected or tail:
            raise RunCheckpointControlError("checkpoint journal moved before append")
        self._append_record(
            journal_descriptor,
            head.to_json_bytes() + b"\n",
        )

    def _require_journal_append_capacity(
        self,
        parent_descriptor: int,
        expected: RunCheckpointHead,
        record: bytes,
        descriptors: ExitStack,
    ) -> None:
        current, tail, _, journal_descriptor = self._read_journal(
            parent_descriptor,
            descriptors,
        )
        if (
            current != expected
            or tail
            or not record.endswith(b"\n")
            or os.fstat(journal_descriptor).st_size + len(record)
            > self._settings.run_checkpoint_journal_size_bytes
        ):
            raise RunCheckpointControlError(
                "checkpoint journal cannot durably append the candidate"
            )

    def _append_record(self, journal_descriptor: int, record: bytes) -> None:
        metadata = os.fstat(journal_descriptor)
        if (
            not record.endswith(b"\n")
            or metadata.st_size + len(record)
            > self._settings.run_checkpoint_journal_size_bytes
        ):
            raise RunCheckpointControlError(
                "checkpoint journal record exceeds its configured bound"
            )
        remaining = memoryview(record)
        while remaining:
            written = os.write(journal_descriptor, remaining)
            if written <= 0:
                raise RunCheckpointControlError(
                    "checkpoint journal append made no progress"
                )
            remaining = remaining[written:]
        os.fsync(journal_descriptor)

    def _write_checkpoint(
        self,
        parent_descriptor: int,
        checkpoint: RunCheckpoint,
        payload: bytes,
        descriptors: ExitStack,
    ) -> None:
        staging_descriptor = self._open_staging(
            parent_descriptor,
            descriptors,
        )
        checkpoint_digest = checkpoint.run_checkpoint_id.rsplit(":", 1)[1]
        temporary_name = f"checkpoint-{checkpoint_digest}-{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=staging_descriptor,
        )
        handle = descriptors.enter_context(os.fdopen(descriptor, "wb"))
        handle.write(payload)
        handle.flush()
        os.fchmod(handle.fileno(), 0o400)
        os.fsync(handle.fileno())
        written = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(written.st_mode)
            or written.st_uid != os.geteuid()
            or written.st_nlink != 1
            or stat.S_IMODE(written.st_mode) != 0o400
            or written.st_size != len(payload)
        ):
            raise RunCheckpointControlError(
                "staged run checkpoint is not one exact private file"
            )
        os.replace(
            temporary_name,
            self._checkpoint_relative.name,
            src_dir_fd=staging_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        rebound = os.stat(
            self._checkpoint_relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (rebound.st_dev, rebound.st_ino) != (
            written.st_dev,
            written.st_ino,
        ):
            raise RunCheckpointControlError(
                "published run checkpoint differs from its staged inode"
            )
        os.fsync(parent_descriptor)
        os.fsync(staging_descriptor)


def _require_control_path(value: str, name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path == PurePosixPath(".")
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or len(path.parts) < 2
    ):
        raise RunCheckpointControlError(f"{name} is unsafe")
    return path


def _same_frontier(
    left: _CheckpointFrontierInspection,
    right: _CheckpointFrontierInspection,
) -> bool:
    return (
        left.head == right.head
        and left.checkpoint == right.checkpoint
        and left.journal_size_bytes == right.journal_size_bytes
        and left.incomplete_tail == right.incomplete_tail
        and left.checkpoint_ahead is right.checkpoint_ahead
    )


def _open_absolute_directory(path: Path, descriptors: ExitStack) -> int:
    normalized = Path(os.path.abspath(path))
    if not path.is_absolute() or path != normalized or normalized.parent == normalized:
        raise RunCheckpointControlError(
            "active run root must be absolute and normalized"
        )
    descriptor = os.open(
        normalized.anchor,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, descriptor)
    for name in normalized.parts[1:]:
        child_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=descriptor,
        )
        descriptors.callback(os.close, child_descriptor)
        descriptor = child_descriptor
    return descriptor


def _require_private_directory(descriptor: int, name: str) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RunCheckpointControlError(f"{name} must be owner-private")


def _directory_identity(descriptor: int, name: str) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RunCheckpointControlError(f"{name} must be a real directory")
    return metadata.st_dev, metadata.st_ino


__all__ = ["RunCheckpointControlError"]
