"""Owner-private compare-and-swap persistence for exact run checkpoints."""

from __future__ import annotations

import fcntl
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Lock
from weakref import WeakValueDictionary

from kapso.cross_run.canonical import content_id, require_content_id
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointHead,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.settings import LaunchSettings

_STAGING_ENTRY_PATTERN = re.compile(r"^checkpoint-[0-9a-f]{64}-[0-9a-f]{32}[.]tmp$")
_WRITE_PERMIT_AUTHORITY = object()
_DURABLE_CHECKPOINT_AUTHORITY = object()


class RunCheckpointStoreError(RuntimeError):
    """The protected checkpoint control plane is unsafe or inconsistent."""


@dataclass(frozen=True)
class DurableRunCheckpoint:
    """Unforgeable receipt that one exact checkpoint is the durable frontier."""

    checkpoint: RunCheckpoint
    journal_head_id: str
    journal_size_bytes: int
    run_root_identity: tuple[int, int]
    control_parent_identity: tuple[int, int]
    _store_identity: object = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.checkpoint) is not RunCheckpoint
            or type(self._store_identity) is not object
            or self._authority is not _DURABLE_CHECKPOINT_AUTHORITY
        ):
            raise RunCheckpointStoreError(
                "durable run checkpoint lacks live store authority"
            )
        _require_head_id(
            self.journal_head_id,
            "durable run checkpoint journal head",
        )
        if type(self.journal_size_bytes) is not int or self.journal_size_bytes <= 0:
            raise RunCheckpointStoreError(
                "durable run checkpoint journal position is invalid"
            )
        for identity, name in (
            (self.run_root_identity, "durable checkpoint run root"),
            (
                self.control_parent_identity,
                "durable checkpoint control parent",
            ),
        ):
            if (
                type(identity) is not tuple
                or len(identity) != 2
                or any(type(part) is not int or part < 0 for part in identity)
            ):
                raise RunCheckpointStoreError(f"{name} identity is invalid")

    @property
    def run_checkpoint_id(self) -> str:
        return self.checkpoint.run_checkpoint_id

    def require_current(self, store: "RunCheckpointStore") -> RunCheckpoint:
        if (
            type(store) is not RunCheckpointStore
            or store._store_identity is not self._store_identity
        ):
            raise RunCheckpointStoreError(
                "durable run checkpoint belongs to another store"
            )
        store._require_durable(self)
        current = store.load()
        if (
            current is None
            or current.checkpoint != self.checkpoint
            or current.journal_head_id != self.journal_head_id
            or current.journal_size_bytes != self.journal_size_bytes
        ):
            raise RunCheckpointStoreError("durable run checkpoint is no longer current")
        return current.checkpoint


@dataclass(frozen=True)
class RunCheckpointWritePermit:
    """One non-clonable authorization for one expected CAS frontier."""

    expected_checkpoint_id: str | None
    expected_journal_head_id: str
    expected_journal_size_bytes: int
    candidate_checkpoint_id: str
    _store_identity: object = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._authority is not _WRITE_PERMIT_AUTHORITY
            or type(self._store_identity) is not object
        ):
            raise RunCheckpointStoreError(
                "run checkpoint permit lacks live store authority"
            )
        if self.expected_checkpoint_id is not None:
            _require_checkpoint_id(
                self.expected_checkpoint_id,
                "run checkpoint permit expected checkpoint",
            )
        _require_head_id(
            self.expected_journal_head_id,
            "run checkpoint permit expected journal head",
        )
        if (
            type(self.expected_journal_size_bytes) is not int
            or self.expected_journal_size_bytes <= 0
        ):
            raise RunCheckpointStoreError(
                "run checkpoint permit journal position is invalid"
            )
        _require_checkpoint_id(
            self.candidate_checkpoint_id,
            "run checkpoint permit candidate",
        )


class RunCheckpointStore:
    """Descriptor-safe, bounded, canonical CAS over one run checkpoint."""

    def __init__(
        self,
        authority: ActiveLaunchWorkspace,
        settings: LaunchSettings,
    ) -> None:
        if type(authority) is not ActiveLaunchWorkspace:
            raise RunCheckpointStoreError(
                "run checkpoint store requires active launch authority"
            )
        if type(settings) is not LaunchSettings:
            raise RunCheckpointStoreError(
                "run checkpoint store requires exact launch settings"
            )
        authority.require_control_authority()
        if settings != authority._prepared._builder_verifier._settings.launch:
            raise RunCheckpointStoreError(
                "run checkpoint settings differ from the active launch"
            )
        self._authority = authority
        self._settings = settings
        self._store_identity = object()
        self._permit_lock = Lock()
        self._issued_permits: dict[int, RunCheckpointWritePermit] = {}
        self._durable_lock = Lock()
        self._issued_durable: WeakValueDictionary[int, DurableRunCheckpoint] = (
            WeakValueDictionary()
        )
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
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError(
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
            self._read_frontier(parent_descriptor, descriptors)

    def load(self) -> DurableRunCheckpoint | None:
        """Return the exact current frontier under the store lock."""

        head, checkpoint, journal_size = self._load_frontier()
        return (
            None
            if checkpoint is None
            else self._durable(
                checkpoint,
                head.run_checkpoint_head_id,
                journal_size,
            )
        )

    def _load_frontier(
        self,
    ) -> tuple[RunCheckpointHead, RunCheckpoint | None, int]:
        self._authority.require_control_authority()
        with ExitStack() as descriptors:
            parent_descriptor = self._open_control_parent(descriptors)
            self._open_locked(parent_descriptor, descriptors)
            self._clean_staging(parent_descriptor, descriptors)
            head, checkpoint, journal_size = self._read_frontier(
                parent_descriptor,
                descriptors,
            )
            self._authority.require_control_authority()
            return head, checkpoint, journal_size

    def issue_write_permit(
        self,
        expected_checkpoint_id: str | None,
        candidate: RunCheckpoint,
    ) -> RunCheckpointWritePermit:
        """Seal one expected frontier for a later CAS attempt."""

        if type(candidate) is not RunCheckpoint:
            raise RunCheckpointStoreError(
                "run checkpoint permit requires one exact candidate"
            )
        if expected_checkpoint_id is not None:
            _require_checkpoint_id(
                expected_checkpoint_id,
                "expected run checkpoint",
            )
        current_head, current_checkpoint, current_journal_size = self._load_frontier()
        current_id = (
            None if current_checkpoint is None else current_checkpoint.run_checkpoint_id
        )
        candidate.require_bootstrap_pin(self._authority.bootstrap_pin)
        if current_checkpoint == candidate:
            if expected_checkpoint_id != candidate.predecessor_checkpoint_id:
                raise RunCheckpointStoreError(
                    "idempotent checkpoint permit names another predecessor"
                )
        else:
            if current_id != expected_checkpoint_id:
                raise RunCheckpointStoreError(
                    "run checkpoint permit expected a stale frontier"
                )
            candidate.require_predecessor(current_checkpoint)
        permit = RunCheckpointWritePermit(
            expected_checkpoint_id=expected_checkpoint_id,
            expected_journal_head_id=current_head.run_checkpoint_head_id,
            expected_journal_size_bytes=current_journal_size,
            candidate_checkpoint_id=candidate.run_checkpoint_id,
            _store_identity=self._store_identity,
            _authority=_WRITE_PERMIT_AUTHORITY,
        )
        with self._permit_lock:
            self._issued_permits[id(permit)] = permit
        return permit

    def compare_and_swap(
        self,
        permit: RunCheckpointWritePermit,
        candidate: RunCheckpoint,
    ) -> DurableRunCheckpoint:
        """Consume one permit and atomically publish one exact successor."""

        if type(permit) is not RunCheckpointWritePermit:
            raise RunCheckpointStoreError(
                "run checkpoint CAS requires one exact write permit"
            )
        if type(candidate) is not RunCheckpoint:
            raise RunCheckpointStoreError(
                "run checkpoint CAS requires one exact checkpoint"
            )
        if permit.candidate_checkpoint_id != candidate.run_checkpoint_id:
            raise RunCheckpointStoreError(
                "run checkpoint permit authorizes another candidate"
            )
        self._consume_permit(permit)
        self._authority.require_control_authority()
        candidate.require_bootstrap_pin(self._authority.bootstrap_pin)
        payload = candidate.to_json_bytes()
        if len(payload) > self._settings.run_checkpoint_size_bytes:
            raise RunCheckpointStoreError("run checkpoint exceeds its configured bound")
        with ExitStack() as descriptors:
            parent_descriptor = self._open_control_parent(descriptors)
            self._open_locked(parent_descriptor, descriptors)
            self._clean_staging(parent_descriptor, descriptors)
            head, current, current_journal_size = self._read_frontier(
                parent_descriptor,
                descriptors,
            )
            current_id = None if current is None else current.run_checkpoint_id
            if current is not None and current == candidate:
                if (
                    permit.expected_checkpoint_id != candidate.predecessor_checkpoint_id
                    or permit.expected_journal_head_id != head.run_checkpoint_head_id
                    or permit.expected_journal_size_bytes != current_journal_size
                ):
                    raise RunCheckpointStoreError(
                        "idempotent checkpoint retry used another predecessor"
                    )
                self._authority.require_control_authority()
                return self._durable(
                    current,
                    head.run_checkpoint_head_id,
                    current_journal_size,
                )
            if (
                current_id != permit.expected_checkpoint_id
                or head.run_checkpoint_head_id != permit.expected_journal_head_id
                or current_journal_size != permit.expected_journal_size_bytes
            ):
                raise RunCheckpointStoreError(
                    "run checkpoint compare-and-swap frontier moved"
                )
            candidate.require_predecessor(current)
            successor_head = head.advance(candidate)
            self._require_journal_append_capacity(
                parent_descriptor,
                head,
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
                raise RunCheckpointStoreError(
                    "persisted run checkpoint differs from its CAS candidate"
                )
            self._append_head(
                parent_descriptor,
                head,
                successor_head,
                descriptors,
            )
            durable_head, tail, _, _ = self._read_journal(
                parent_descriptor,
                descriptors,
            )
            if tail:
                raise RunCheckpointStoreError(
                    "durable checkpoint journal has an incomplete tail"
                )
            durable_head.require_checkpoint(persisted)
            self._authority.require_control_authority()
            return self._durable(
                persisted,
                durable_head.run_checkpoint_head_id,
                os.stat(
                    self._journal_relative.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                ).st_size,
            )

    def _durable(
        self,
        checkpoint: RunCheckpoint,
        journal_head_id: str,
        journal_size_bytes: int,
    ) -> DurableRunCheckpoint:
        if self._control_parent_identity is None or self._staging_identity is None:
            raise RunCheckpointStoreError(
                "run checkpoint store has no pinned control authority"
            )
        durable = DurableRunCheckpoint(
            checkpoint=checkpoint,
            journal_head_id=journal_head_id,
            journal_size_bytes=journal_size_bytes,
            run_root_identity=self._authority.published_root_identity,
            control_parent_identity=self._control_parent_identity,
            _store_identity=self._store_identity,
            _authority=_DURABLE_CHECKPOINT_AUTHORITY,
        )
        with self._durable_lock:
            self._issued_durable[id(durable)] = durable
        return durable

    def _require_durable(self, durable: DurableRunCheckpoint) -> None:
        with self._durable_lock:
            issued = self._issued_durable.get(id(durable))
        if (
            issued is not durable
            or durable._authority is not _DURABLE_CHECKPOINT_AUTHORITY
            or durable._store_identity is not self._store_identity
        ):
            raise RunCheckpointStoreError("durable run checkpoint is cloned or foreign")

    def _consume_permit(self, permit: RunCheckpointWritePermit) -> None:
        with self._permit_lock:
            issued = self._issued_permits.pop(id(permit), None)
        if (
            issued is not permit
            or permit._authority is not _WRITE_PERMIT_AUTHORITY
            or permit._store_identity is not self._store_identity
        ):
            raise RunCheckpointStoreError(
                "run checkpoint write permit is cloned, foreign, or consumed"
            )

    def _open_control_parent(self, descriptors: ExitStack) -> int:
        root_descriptor = _open_absolute_directory(
            self._authority.run_root,
            descriptors,
        )
        if (
            _directory_identity(root_descriptor, "active run root")
            != self._authority.published_root_identity
        ):
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError("run checkpoint control parent was replaced")
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
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError(
                "run checkpoint staging exceeds its entry bound"
            )
        for entry in entries:
            if _STAGING_ENTRY_PATTERN.fullmatch(entry.name) is None:
                raise RunCheckpointStoreError(
                    "run checkpoint staging contains an unexpected entry"
                )
            metadata = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
            ):
                raise RunCheckpointStoreError("run checkpoint staging entry is unsafe")
            os.unlink(entry.name, dir_fd=staging_descriptor)
        if entries:
            os.fsync(staging_descriptor)

    def _read_frontier(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> tuple[RunCheckpointHead, RunCheckpoint | None, int]:
        head, tail, complete_size, journal_descriptor = self._read_journal(
            parent_descriptor,
            descriptors,
        )
        checkpoint = self._read_checkpoint(parent_descriptor, descriptors)
        if checkpoint is None:
            if head.checkpoint is not None:
                raise RunCheckpointStoreError(
                    "durable checkpoint journal names an absent checkpoint"
                )
            if tail:
                raise RunCheckpointStoreError(
                    "fresh checkpoint journal has an incomplete record"
                )
            return head, None, complete_size
        if head.checkpoint == checkpoint:
            if tail:
                raise RunCheckpointStoreError(
                    "checkpoint journal has an impossible incomplete record"
                )
            head.require_checkpoint(checkpoint)
            return head, checkpoint, complete_size
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
            raise RunCheckpointStoreError(
                "run checkpoint was rolled back or skipped its journal frontier"
            )
        reconciled = head.advance(checkpoint)
        expected_record = reconciled.to_json_bytes() + b"\n"
        staging_descriptor = self._open_staging(
            parent_descriptor,
            descriptors,
        )
        os.fsync(parent_descriptor)
        os.fsync(staging_descriptor)
        if tail:
            if len(tail) >= len(expected_record) or not expected_record.startswith(
                tail
            ):
                raise RunCheckpointStoreError(
                    "checkpoint journal tail is not the exact recovery record prefix"
                )
            os.ftruncate(journal_descriptor, complete_size)
            os.fsync(journal_descriptor)
        self._append_record(journal_descriptor, expected_record)
        durable_head, durable_tail, _, _ = self._read_journal(
            parent_descriptor,
            descriptors,
        )
        if durable_head != reconciled or durable_tail:
            raise RunCheckpointStoreError(
                "reconciled checkpoint journal differs from its frontier"
            )
        durable_head.require_checkpoint(checkpoint)
        return durable_head, checkpoint, os.fstat(journal_descriptor).st_size

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
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError("checkpoint journal changed while reading")
        complete_size = payload.rfind(b"\n") + 1
        if complete_size == 0:
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError(
                "checkpoint journal initial record or canonical bytes are invalid"
            )
        for previous, current in zip(records, records[1:], strict=False):
            if (
                current.checkpoint is None
                or previous.advance(current.checkpoint) != current
            ):
                raise RunCheckpointStoreError("checkpoint journal lineage is not exact")
        final_binding = os.stat(
            self._journal_relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (final_binding.st_dev, final_binding.st_ino) != identity:
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError("run checkpoint changed while reading")
        checkpoint = RunCheckpoint.from_json_bytes(payload)
        if payload != checkpoint.to_json_bytes():
            raise RunCheckpointStoreError("run checkpoint bytes are not canonical")
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
            raise RunCheckpointStoreError("run checkpoint was rebound while validating")
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
            raise RunCheckpointStoreError(
                "checkpoint journal record is not the exact successor"
            )
        current, tail, _, journal_descriptor = self._read_journal(
            parent_descriptor,
            descriptors,
        )
        if current != expected or tail:
            raise RunCheckpointStoreError("checkpoint journal moved before append")
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
            raise RunCheckpointStoreError(
                "checkpoint journal cannot durably append the candidate"
            )

    def _append_record(self, journal_descriptor: int, record: bytes) -> None:
        metadata = os.fstat(journal_descriptor)
        if (
            not record.endswith(b"\n")
            or metadata.st_size + len(record)
            > self._settings.run_checkpoint_journal_size_bytes
        ):
            raise RunCheckpointStoreError(
                "checkpoint journal record exceeds its configured bound"
            )
        remaining = memoryview(record)
        while remaining:
            written = os.write(journal_descriptor, remaining)
            if written <= 0:
                raise RunCheckpointStoreError(
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
            raise RunCheckpointStoreError(
                "staged run checkpoint is not one exact private file"
            )
        os.replace(
            temporary_name,
            self._checkpoint_relative.name,
            src_dir_fd=staging_descriptor,
            dst_dir_fd=parent_descriptor,
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
        raise RunCheckpointStoreError(f"{name} is unsafe")
    return path


def _require_checkpoint_id(value: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != RunCheckpoint.CONTENT_NAMESPACE:
        raise RunCheckpointStoreError(f"{name} uses the wrong namespace")


def _require_head_id(value: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != RunCheckpointHead.CONTENT_NAMESPACE:
        raise RunCheckpointStoreError(f"{name} uses the wrong namespace")


def _open_absolute_directory(path: Path, descriptors: ExitStack) -> int:
    normalized = Path(os.path.abspath(path))
    if not path.is_absolute() or path != normalized or normalized.parent == normalized:
        raise RunCheckpointStoreError("active run root must be absolute and normalized")
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
        raise RunCheckpointStoreError(f"{name} must be owner-private")


def _directory_identity(descriptor: int, name: str) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RunCheckpointStoreError(f"{name} must be a real directory")
    return metadata.st_dev, metadata.st_ino


__all__ = [
    "DurableRunCheckpoint",
    "RunCheckpointStore",
    "RunCheckpointStoreError",
    "RunCheckpointWritePermit",
]
