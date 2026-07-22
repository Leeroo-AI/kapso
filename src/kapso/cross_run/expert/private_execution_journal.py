"""Private create-only filesystem substrate for evaluator execution journals."""

from __future__ import annotations

import ctypes
import fcntl
import os
import re
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import StrictContract

_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_MAXIMUM_EVENT_NUMBER = (10**20) - 1
_RESERVATION_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LOCK_FILENAME_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.lock$"
)
_EVENT_FILENAME_PATTERN = re.compile(r"^(?P<number>[0-9]{20})\.json$")
_STAGING_FILENAME_PATTERN = re.compile(r"^\.(?:event|result)-[0-9a-f]{32}\.tmp$")
_RESULT_FILENAME_PATTERN = re.compile(r"^(?P<digest>[0-9a-f]{64})\.json$")


class ExecutionJournalStoreError(ValueError):
    """A private execution journal is unsafe, corrupt, or conflicting."""


@dataclass(frozen=True)
class ExecutionJournalResultBlob(StrictContract):
    digest: str
    size: int

    def _validate(self) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest) is None:
            raise ExecutionJournalStoreError(
                "execution journal result blob digest is invalid"
            )
        if type(self.size) is not int or self.size < 0:
            raise ExecutionJournalStoreError(
                "execution journal result blob size must be non-negative"
            )


@dataclass(frozen=True)
class NumberedExecutionJournalPayload:
    event_number: int
    payload: bytes

    def __post_init__(self) -> None:
        _require_event_number(self.event_number)
        if not isinstance(self.payload, bytes):
            raise ExecutionJournalStoreError(
                "execution journal event payload must be bytes"
            )


class ExecutionJournalFilesystem:
    """Own bounded private paths, locks, events, staging, and result blobs."""

    def __init__(
        self,
        root: Path,
        trusted_root: Path,
        *,
        maximum_event_size_bytes: int,
        maximum_result_size_bytes: int,
        maximum_staging_entry_count: int,
    ) -> None:
        if (
            not isinstance(trusted_root, Path)
            or not trusted_root.is_absolute()
            or trusted_root.resolve() != trusted_root
            or not trusted_root.is_dir()
        ):
            raise ExecutionJournalStoreError(
                "execution journal trusted root must be a resolved directory"
            )
        _validate_private_directory(trusted_root, "execution journal trusted root")
        if (
            not isinstance(root, Path)
            or not root.is_absolute()
            or root != Path(os.path.abspath(root))
            or root.parent != trusted_root
        ):
            raise ExecutionJournalStoreError(
                "execution journal must be a direct child of its trusted root"
            )
        for value, name in (
            (maximum_event_size_bytes, "event byte limit"),
            (maximum_result_size_bytes, "result byte limit"),
            (maximum_staging_entry_count, "staging entry limit"),
        ):
            if type(value) is not int or value <= 0:
                raise ExecutionJournalStoreError(
                    f"execution journal {name} must be a positive integer"
                )
        self.root = root
        self.trusted_root = trusted_root
        self.maximum_event_size_bytes = maximum_event_size_bytes
        self.maximum_result_size_bytes = maximum_result_size_bytes
        self.maximum_staging_entry_count = maximum_staging_entry_count
        self.lock_root = root / "locks"
        self.reservation_root = root / "reservations"
        self.initialization_lock_path = trusted_root / f".{root.name}.lock"
        with ExecutionJournalLock(self.initialization_lock_path):
            self._prepare_layout()

    def ensure_reservation_layout(self, reservation_digest: str) -> None:
        digest = _require_reservation_digest(reservation_digest)
        with ExecutionJournalLock(self.initialization_lock_path):
            reservation_root = self._reservation_path(digest)
            _ensure_private_directory(reservation_root, self.reservation_root)
            _ensure_private_directory(
                reservation_root / "events",
                reservation_root,
            )
            _ensure_private_directory(
                reservation_root / "staging",
                reservation_root,
            )
            _ensure_private_directory(
                reservation_root / "results",
                reservation_root,
            )

    def has_complete_reservation_layout(self, reservation_digest: str) -> bool:
        """Return false for absent/empty partial layout; reject unsafe partial state."""

        digest = _require_reservation_digest(reservation_digest)
        reservation_root = self._reservation_path(digest)
        with ExecutionJournalLock(self.initialization_lock_path):
            if not os.path.lexists(reservation_root):
                return False
            _validate_private_directory(
                reservation_root,
                "execution journal reservation",
            )
            required_directories = {"events", "staging", "results"}
            with os.scandir(reservation_root) as entries:
                observed_entries = tuple(entries)
            observed_names = {entry.name for entry in observed_entries}
            if not observed_names.issubset(required_directories):
                raise ExecutionJournalStoreError(
                    "partial execution journal layout has an unexpected entry"
                )
            for entry in observed_entries:
                if not entry.is_dir(follow_symlinks=False):
                    raise ExecutionJournalStoreError(
                        "partial execution journal layout contains a non-directory"
                    )
                _validate_private_directory(
                    Path(entry.path),
                    f"execution journal {entry.name}",
                )
            if observed_names == required_directories:
                return True
            for entry in observed_entries:
                with os.scandir(entry.path) as children:
                    if next(children, None) is not None:
                        raise ExecutionJournalStoreError(
                            "partial execution journal layout contains durable state"
                        )
            return False

    def reservation_lock(
        self,
        reservation_digest: str,
    ) -> ExecutionJournalLock:
        return ExecutionJournalLock(self.reservation_lock_path(reservation_digest))

    def lock(self, lock_filename: str) -> ExecutionJournalLock:
        if (
            not isinstance(lock_filename, str)
            or _LOCK_FILENAME_PATTERN.fullmatch(lock_filename) is None
        ):
            raise ExecutionJournalStoreError(
                "execution journal lock filename is invalid"
            )
        return ExecutionJournalLock(self.lock_root / lock_filename)

    def clean_staging(self, reservation_digest: str) -> None:
        digest = _require_reservation_digest(reservation_digest)
        staging_root = self._staging_path(digest)
        scanned_entries = []
        with os.scandir(staging_root) as entries:
            for entry in entries:
                scanned_entries.append(entry)
                if len(scanned_entries) > self.maximum_staging_entry_count:
                    raise ExecutionJournalStoreError(
                        "execution journal staging exceeds its configured bound"
                    )
        observed_entries = tuple(scanned_entries)
        for entry in observed_entries:
            if _STAGING_FILENAME_PATTERN.fullmatch(entry.name) is None:
                raise ExecutionJournalStoreError(
                    "execution journal staging contains an unexpected entry"
                )
            metadata = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
                or metadata.st_uid != os.geteuid()
            ):
                raise ExecutionJournalStoreError(
                    "execution journal staging entry is unsafe"
                )
        for entry in observed_entries:
            os.unlink(entry.path)
        if observed_entries:
            _fsync_directory(staging_root)

    def read_numbered_event_payloads(
        self,
        reservation_digest: str,
        maximum_count: int,
    ) -> tuple[NumberedExecutionJournalPayload, ...]:
        digest = _require_reservation_digest(reservation_digest)
        _require_nonnegative_count(maximum_count, "event count bound")
        scanned_entries = []
        with os.scandir(self._events_path(digest)) as entries:
            for entry in entries:
                scanned_entries.append(entry)
                if len(scanned_entries) > maximum_count:
                    raise ExecutionJournalStoreError(
                        "execution journal exceeds its structural event bound"
                    )
        numbered_payloads = []
        for entry in sorted(scanned_entries, key=lambda item: item.name):
            match = _EVENT_FILENAME_PATTERN.fullmatch(entry.name)
            if match is None:
                raise ExecutionJournalStoreError(
                    "execution journal contains an unexpected event entry"
                )
            numbered_payloads.append(
                NumberedExecutionJournalPayload(
                    event_number=int(match.group("number")),
                    payload=_read_private_file(
                        Path(entry.path),
                        required_mode=0o400,
                        name="execution journal event",
                        maximum_size_bytes=self.maximum_event_size_bytes,
                    ),
                )
            )
        return tuple(numbered_payloads)

    def publish_numbered_event(
        self,
        reservation_digest: str,
        event_number: int,
        payload: bytes,
    ) -> None:
        digest = _require_reservation_digest(reservation_digest)
        number = _require_event_number(event_number)
        _require_bounded_payload(
            payload,
            self.maximum_event_size_bytes,
            "execution journal event",
        )
        self._publish_create_only(
            staging_root=self._staging_path(digest),
            temporary_prefix="event",
            destination=self._events_path(digest) / f"{number:020d}.json",
            payload=payload,
            maximum_size_bytes=self.maximum_event_size_bytes,
            name="execution journal event",
        )

    def publish_result(
        self,
        reservation_digest: str,
        payload: bytes,
    ) -> ExecutionJournalResultBlob:
        digest = _require_reservation_digest(reservation_digest)
        _require_bounded_payload(
            payload,
            self.maximum_result_size_bytes,
            "execution journal result blob",
        )
        result_blob = ExecutionJournalResultBlob(
            digest=tree_or_blob_digest(payload),
            size=len(payload),
        )
        self._publish_create_only(
            staging_root=self._staging_path(digest),
            temporary_prefix="result",
            destination=self._result_path(digest, result_blob),
            payload=payload,
            maximum_size_bytes=self.maximum_result_size_bytes,
            name="execution journal result blob",
        )
        return result_blob

    def read_result(
        self,
        reservation_digest: str,
        result_blob: ExecutionJournalResultBlob,
    ) -> bytes:
        digest = _require_reservation_digest(reservation_digest)
        if type(result_blob) is not ExecutionJournalResultBlob:
            raise ExecutionJournalStoreError(
                "execution journal result reference is not exact"
            )
        payload = _read_private_file(
            self._result_path(digest, result_blob),
            required_mode=0o400,
            name="execution journal result blob",
            maximum_size_bytes=self.maximum_result_size_bytes,
        )
        if (
            len(payload) != result_blob.size
            or tree_or_blob_digest(payload) != result_blob.digest
        ):
            raise ExecutionJournalStoreError(
                "execution journal result blob differs from its descriptor"
            )
        return payload

    def validate_results(
        self,
        reservation_digest: str,
        maximum_count: int,
    ) -> None:
        digest = _require_reservation_digest(reservation_digest)
        _require_nonnegative_count(maximum_count, "result count bound")
        entry_count = 0
        with os.scandir(self._results_path(digest)) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > maximum_count:
                    raise ExecutionJournalStoreError(
                        "execution journal result store exceeds its structural bound"
                    )
                match = _RESULT_FILENAME_PATTERN.fullmatch(entry.name)
                if match is None:
                    raise ExecutionJournalStoreError(
                        "execution journal result store contains an unexpected entry"
                    )
                payload = _read_private_file(
                    Path(entry.path),
                    required_mode=0o400,
                    name="execution journal result blob",
                    maximum_size_bytes=self.maximum_result_size_bytes,
                )
                if tree_or_blob_digest(payload).removeprefix("sha256:") != match.group(
                    "digest"
                ):
                    raise ExecutionJournalStoreError(
                        "execution journal result filename differs from its payload"
                    )

    def reservation_lock_path(self, reservation_digest: str) -> Path:
        digest = _require_reservation_digest(reservation_digest)
        return self.lock_root / f"{digest}.lock"

    def reservation_path(self, reservation_digest: str) -> Path:
        return self._reservation_path(_require_reservation_digest(reservation_digest))

    def events_path(self, reservation_digest: str) -> Path:
        return self._events_path(_require_reservation_digest(reservation_digest))

    def staging_path(self, reservation_digest: str) -> Path:
        return self._staging_path(_require_reservation_digest(reservation_digest))

    def results_path(self, reservation_digest: str) -> Path:
        return self._results_path(_require_reservation_digest(reservation_digest))

    def event_path(self, reservation_digest: str, event_number: int) -> Path:
        digest = _require_reservation_digest(reservation_digest)
        number = _require_event_number(event_number)
        return self._events_path(digest) / f"{number:020d}.json"

    def result_path(
        self,
        reservation_digest: str,
        result_blob: ExecutionJournalResultBlob,
    ) -> Path:
        digest = _require_reservation_digest(reservation_digest)
        if type(result_blob) is not ExecutionJournalResultBlob:
            raise ExecutionJournalStoreError(
                "execution journal result reference is not exact"
            )
        return self._result_path(digest, result_blob)

    def _prepare_layout(self) -> None:
        _ensure_private_directory(self.root, self.trusted_root)
        _ensure_private_directory(self.lock_root, self.root)
        _ensure_private_directory(self.reservation_root, self.root)
        _validate_private_directory(self.root, "execution journal root")
        _validate_private_directory(self.lock_root, "execution journal locks")
        _validate_private_directory(
            self.reservation_root,
            "execution journal reservations",
        )

    def _publish_create_only(
        self,
        *,
        staging_root: Path,
        temporary_prefix: str,
        destination: Path,
        payload: bytes,
        maximum_size_bytes: int,
        name: str,
    ) -> None:
        temporary_path = (
            staging_root / f".{temporary_prefix}-{secrets.token_hex(16)}.tmp"
        )
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o400)
            os.fsync(handle.fileno())
        staged_payload = _read_private_file(
            temporary_path,
            required_mode=0o400,
            name=f"staged {name}",
            maximum_size_bytes=maximum_size_bytes,
        )
        if staged_payload != payload:
            raise ExecutionJournalStoreError(f"staged {name} differs from its payload")
        _rename_no_replace(temporary_path, destination)
        self._fsync_directory(destination.parent)
        self._fsync_directory(staging_root)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        _fsync_directory(path)

    def _reservation_path(self, reservation_digest: str) -> Path:
        return self.reservation_root / reservation_digest

    def _events_path(self, reservation_digest: str) -> Path:
        return self._reservation_path(reservation_digest) / "events"

    def _staging_path(self, reservation_digest: str) -> Path:
        return self._reservation_path(reservation_digest) / "staging"

    def _results_path(self, reservation_digest: str) -> Path:
        return self._reservation_path(reservation_digest) / "results"

    def _result_path(
        self,
        reservation_digest: str,
        result_blob: ExecutionJournalResultBlob,
    ) -> Path:
        digest = result_blob.digest.removeprefix("sha256:")
        return self._results_path(reservation_digest) / f"{digest}.json"


class ExecutionJournalLock:
    """Owner-private kernel lock whose runtime authority is process-local."""

    def __init__(self, path: Path) -> None:
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or path != Path(os.path.abspath(path))
        ):
            raise ExecutionJournalStoreError(
                "execution journal lock requires an absolute normalized path"
            )
        self.path = path
        self.handle = None
        self.acquired = False
        self.owner_process_id = None

    def __enter__(self) -> ExecutionJournalLock:
        if (
            self.handle is not None
            or self.acquired
            or self.owner_process_id is not None
        ):
            raise ExecutionJournalStoreError(
                "execution journal lock cannot be entered twice"
            )
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        self.handle = os.fdopen(descriptor, "r+b")
        metadata = os.fstat(self.handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
        ):
            self.handle.close()
            self.handle = None
            raise ExecutionJournalStoreError(
                "execution journal lock must be a private independent file"
            )
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        self.acquired = True
        self.owner_process_id = os.getpid()
        return self

    def require_acquired(self) -> None:
        if (
            not self.acquired
            or self.handle is None
            or self.owner_process_id != os.getpid()
        ):
            raise ExecutionJournalStoreError(
                "execution journal lock is not held by its creator process"
            )

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.require_acquired()
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.acquired = False
        self.owner_process_id = None
        self.handle.close()
        self.handle = None
        return False


def _require_reservation_digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or _RESERVATION_DIGEST_PATTERN.fullmatch(value) is None
    ):
        raise ExecutionJournalStoreError(
            "execution journal reservation digest must be 64 lowercase hex"
        )
    return value


def _require_event_number(value: int) -> int:
    if type(value) is not int or not 1 <= value <= _MAXIMUM_EVENT_NUMBER:
        raise ExecutionJournalStoreError(
            "execution journal event number is outside its filename range"
        )
    return value


def _require_nonnegative_count(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ExecutionJournalStoreError(
            f"execution journal {name} must be a non-negative integer"
        )


def _require_bounded_payload(
    payload: bytes, maximum_size_bytes: int, name: str
) -> None:
    if not isinstance(payload, bytes):
        raise ExecutionJournalStoreError(f"{name} must be bytes")
    if len(payload) > maximum_size_bytes:
        raise ExecutionJournalStoreError(f"{name} exceeds its configured bound")


def _ensure_private_directory(path: Path, parent: Path) -> None:
    if not os.path.lexists(path):
        os.mkdir(path, mode=0o700)
        _fsync_directory(parent)
    _validate_private_directory(path, "execution journal directory")


def _validate_private_directory(path: Path, name: str) -> None:
    metadata = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise ExecutionJournalStoreError(
            f"{name} must be an owner-private real directory"
        )


def _read_private_file(
    path: Path,
    *,
    required_mode: int,
    name: str,
    maximum_size_bytes: int,
) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != required_mode
            or metadata.st_uid != os.geteuid()
        ):
            raise ExecutionJournalStoreError(
                f"{name} must be a private independent regular file"
            )
        payload = handle.read(maximum_size_bytes + 1)
    if len(payload) > maximum_size_bytes:
        raise ExecutionJournalStoreError(f"{name} exceeds its configured bound")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    os.fsync(descriptor)
    os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise ExecutionJournalStoreError(
            "atomic no-replace execution journal publication is unavailable"
        )
    rename_at2 = libc.renameat2
    rename_at2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_at2.restype = ctypes.c_int
    result = rename_at2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "execution journal publication failed")
