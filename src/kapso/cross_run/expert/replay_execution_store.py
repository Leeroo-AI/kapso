"""Create-only local execution journal for expert source replay."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import (
    ExpertSourceReplayExecutionLegKind,
    ExpertSourceReplayExecutionRequest,
    ExpertSourceReplayExecutionReservation,
    StrictContract,
)
from kapso.cross_run.expert.replay_protocol import (
    TaskEvaluatorInvocationAllocation,
)

_EXECUTION_JOURNAL_SCHEMA_VERSION = "kapso.source_replay_execution_journal.v1"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_EVENT_FILENAME_PATTERN = re.compile(
    r"^(?P<number>[0-9]{20})-(?P<digest>[0-9a-f]{64})\.json$"
)
_STAGING_FILENAME_PATTERN = re.compile(r"^\.event-[0-9a-f]{32}\.tmp$")


class ExpertSourceReplayExecutionStoreError(ValueError):
    """The private execution journal is unsafe, corrupt, or conflicting."""


class SourceReplayExecutionJournalEventKind(str, Enum):
    INVOCATION_ALLOCATED = "invocation_allocated"


@dataclass(frozen=True)
class SourceReplayExecutionJournalEvent(StrictContract):
    event_id: str
    schema_version: str
    event_number: int
    predecessor_event_id: str | None
    event_kind: SourceReplayExecutionJournalEventKind
    reservation_id: str
    execution_request_id: str
    execution_case_id: str
    execution_leg_id: str
    invocation_allocation: TaskEvaluatorInvocationAllocation

    CONTENT_NAMESPACE: ClassVar[str] = "source-replay-execution-journal-event"
    IDENTITY_FIELD: ClassVar[str] = "event_id"

    def _validate(self) -> None:
        if self.schema_version != _EXECUTION_JOURNAL_SCHEMA_VERSION:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay execution journal schema is unsupported"
            )
        if type(self.event_number) is not int or self.event_number <= 0:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay execution event number must be positive"
            )
        if (self.predecessor_event_id is None) != (self.event_number == 1):
            raise ExpertSourceReplayExecutionStoreError(
                "only the first execution event may omit its predecessor"
            )
        if self.predecessor_event_id is not None:
            require_content_id(
                self.predecessor_event_id,
                "source replay execution predecessor_event_id",
            )
            if self.predecessor_event_id.split(":sha256:", 1)[0] != (
                "source-replay-execution-journal-event"
            ):
                raise ExpertSourceReplayExecutionStoreError(
                    "source replay execution predecessor uses the wrong namespace"
                )
        for value, namespace, name in (
            (
                self.reservation_id,
                "expert-source-replay-execution-reservation",
                "reservation_id",
            ),
            (
                self.execution_request_id,
                "expert-source-replay-execution-request",
                "execution_request_id",
            ),
            (
                self.execution_case_id,
                "expert-source-replay-execution-case",
                "execution_case_id",
            ),
            (
                self.execution_leg_id,
                "expert-source-replay-execution-leg",
                "execution_leg_id",
            ),
        ):
            require_content_id(value, f"source replay execution event {name}")
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayExecutionStoreError(
                    f"source replay execution event {name} uses the wrong namespace"
                )
        allocation = self.invocation_allocation
        if (
            allocation.reservation_id != self.reservation_id
            or allocation.execution_case_id != self.execution_case_id
            or allocation.execution_leg_id != self.execution_leg_id
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay invocation allocation differs from its event"
            )


def _validate_reservation_request(
    reservation: ExpertSourceReplayExecutionReservation,
    request: ExpertSourceReplayExecutionRequest,
) -> None:
    if not isinstance(
        reservation, ExpertSourceReplayExecutionReservation
    ) or not isinstance(request, ExpertSourceReplayExecutionRequest):
        raise ExpertSourceReplayExecutionStoreError(
            "execution journal requires typed reservation and request authority"
        )
    if (
        reservation.execution_request_id != request.execution_request_id
        or reservation.validation_attempt_id != request.validation_attempt_id
        or reservation.authorization_state_id != request.authorization_state_id
        or reservation.candidate_id != request.candidate_id
        or reservation.candidate_tree_hash != request.candidate_tree_hash
        or reservation.observed_parent_release_id != request.parent_release_id
    ):
        raise ExpertSourceReplayExecutionStoreError(
            "execution journal reservation differs from its request"
        )


def source_replay_execution_schedule(
    reservation: ExpertSourceReplayExecutionReservation,
    request: ExpertSourceReplayExecutionRequest,
) -> tuple[tuple[str, str], ...]:
    """Return the sole case/leg order authorized by paired protocol v1."""

    _validate_reservation_request(reservation, request)
    schedule = []
    for case in request.cases:
        legs = {
            ExpertSourceReplayExecutionLegKind.CONTROL_PARENT: case.control_leg,
            ExpertSourceReplayExecutionLegKind.CANDIDATE: case.candidate_leg,
        }
        schedule.extend(
            (case.execution_case_id, legs[leg_kind].execution_leg_id)
            for leg_kind in case.compute_binding.leg_order
        )
    return tuple(schedule)


def _new_invocation_nonce() -> str:
    return secrets.token_hex(16)


class SourceReplayInvocationAllocationPermit:
    """Runtime-only proof that one active locked session owns an allocation."""

    __slots__ = ("_session", "_event_id", "allocation")

    def __init__(
        self,
        session: _SourceReplayReservationSession,
        event: SourceReplayExecutionJournalEvent,
    ) -> None:
        self._session = session
        self._event_id = event.event_id
        self.allocation = event.invocation_allocation

    def require_current_allocation(
        self,
        execution_store: ExpertSourceReplayExecutionStore,
    ) -> TaskEvaluatorInvocationAllocation:
        self._session._require_live_store_lock(execution_store)
        if (
            self._session._allocation_permit is not self
            or not self._session._events
            or self._session._events[-1].event_id != self._event_id
            or self._session._events[-1].invocation_allocation != self.allocation
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay allocation permit is not current"
            )
        return self.allocation


class _SourceReplayReservationSession:
    """One exclusively locked reservation execution prefix."""

    def __init__(
        self,
        store: ExpertSourceReplayExecutionStore,
        reservation: ExpertSourceReplayExecutionReservation,
        request: ExpertSourceReplayExecutionRequest,
        events: tuple[SourceReplayExecutionJournalEvent, ...],
        execution_lock: _ExecutionStoreLock,
        factory_authority: object,
    ) -> None:
        if factory_authority is not store._session_factory_authority:
            raise ExpertSourceReplayExecutionStoreError(
                "execution session lacks canonical store authority"
            )
        self._store = store
        self.reservation = reservation
        self.request = request
        self._events = events
        self._execution_lock = execution_lock
        self._active = True
        self._allocation_permit = None

    @property
    def events(self) -> tuple[SourceReplayExecutionJournalEvent, ...]:
        self._require_active()
        return self._events

    def allocate_expected_leg(self) -> SourceReplayInvocationAllocationPermit:
        self._require_active()
        if self._events:
            if self._allocation_permit is None:
                self._allocation_permit = SourceReplayInvocationAllocationPermit(
                    self,
                    self._events[-1],
                )
            return self._allocation_permit
        execution_case_id, execution_leg_id = source_replay_execution_schedule(
            self.reservation,
            self.request,
        )[0]
        allocation = TaskEvaluatorInvocationAllocation(
            reservation_id=self.reservation.reservation_id,
            execution_case_id=execution_case_id,
            execution_leg_id=execution_leg_id,
            invocation_nonce=_new_invocation_nonce(),
        )
        event = SourceReplayExecutionJournalEvent.mint(
            schema_version=_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=1,
            predecessor_event_id=None,
            event_kind=SourceReplayExecutionJournalEventKind.INVOCATION_ALLOCATED,
            reservation_id=self.reservation.reservation_id,
            execution_request_id=self.request.execution_request_id,
            execution_case_id=execution_case_id,
            execution_leg_id=execution_leg_id,
            invocation_allocation=allocation,
        )
        self._store._publish_event(self.reservation.reservation_id, event)
        self._events = (event,)
        self._allocation_permit = SourceReplayInvocationAllocationPermit(
            self,
            event,
        )
        return self._allocation_permit

    def _require_active(self) -> None:
        if not self._active:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay reservation session is closed"
            )

    def _require_live_store_lock(
        self,
        execution_store: ExpertSourceReplayExecutionStore,
    ) -> None:
        self._require_active()
        if (
            execution_store is not self._store
            or not isinstance(self._execution_lock, _ExecutionStoreLock)
            or not self._execution_lock.acquired
            or self._execution_lock.handle is None
            or self._execution_lock.path
            != execution_store._lock_path(self.reservation.reservation_id)
            or execution_store._active_sessions.get(self.reservation.reservation_id)
            is not self
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay allocation permit lacks the canonical live store lock"
            )

    def _close(self) -> None:
        self._active = False
        self._allocation_permit = None


class _ReservationSessionContext:
    def __init__(
        self,
        store: ExpertSourceReplayExecutionStore,
        reservation: ExpertSourceReplayExecutionReservation,
        request: ExpertSourceReplayExecutionRequest,
    ) -> None:
        self.store = store
        self.reservation = reservation
        self.request = request
        self.stack = None
        self.session = None

    def __enter__(self) -> _SourceReplayReservationSession:
        self.store._prepare_reservation_layout(self.reservation.reservation_id)
        with ExitStack() as setup:
            execution_lock = setup.enter_context(
                _ExecutionStoreLock(
                    self.store._lock_path(self.reservation.reservation_id),
                )
            )
            self.store._clean_staging(self.reservation.reservation_id)
            events = self.store._read_events(
                self.reservation,
                self.request,
            )
            self.session = _SourceReplayReservationSession(
                self.store,
                self.reservation,
                self.request,
                events,
                execution_lock,
                self.store._session_factory_authority,
            )
            self.store._register_active_session(self.session)
            self.stack = setup.pop_all()
        return self.session

    def __exit__(self, exception_type, exception, traceback):
        self.store._unregister_active_session(self.session)
        self.session._close()
        self.session = None
        stack = self.stack
        self.stack = None
        return stack.__exit__(exception_type, exception, traceback)


class ExpertSourceReplayExecutionStore:
    """Own private, create-only, per-reservation execution event chains."""

    def __init__(self, root: Path, trusted_root: Path) -> None:
        if (
            not isinstance(trusted_root, Path)
            or not trusted_root.is_absolute()
            or trusted_root.resolve() != trusted_root
            or not trusted_root.is_dir()
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal trusted root must be a resolved directory"
            )
        if (
            not isinstance(root, Path)
            or not root.is_absolute()
            or root != Path(os.path.abspath(root))
            or root.parent != trusted_root
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal must be a direct child of its trusted root"
            )
        self.root = root
        self.trusted_root = trusted_root
        self.lock_root = root / "locks"
        self.reservation_root = root / "reservations"
        self.initialization_lock_path = trusted_root / f".{root.name}.lock"
        self._session_factory_authority = object()
        self._active_sessions = {}
        with _ExecutionStoreLock(self.initialization_lock_path):
            self._prepare_layout()

    def reservation_session(
        self,
        *,
        reservation: ExpertSourceReplayExecutionReservation,
        request: ExpertSourceReplayExecutionRequest,
    ) -> _ReservationSessionContext:
        _validate_reservation_request(reservation, request)
        return _ReservationSessionContext(self, reservation, request)

    def _register_active_session(
        self,
        session: _SourceReplayReservationSession,
    ) -> None:
        reservation_id = session.reservation.reservation_id
        if (
            session._store is not self
            or not session._execution_lock.acquired
            or reservation_id in self._active_sessions
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution store cannot register the reservation session"
            )
        self._active_sessions[reservation_id] = session

    def _unregister_active_session(
        self,
        session: _SourceReplayReservationSession,
    ) -> None:
        reservation_id = session.reservation.reservation_id
        if self._active_sessions.get(reservation_id) is not session:
            raise ExpertSourceReplayExecutionStoreError(
                "execution store reservation session registration changed"
            )
        del self._active_sessions[reservation_id]

    def _prepare_layout(self) -> None:
        self._ensure_private_directory(self.root, self.trusted_root)
        self._ensure_private_directory(self.lock_root, self.root)
        self._ensure_private_directory(self.reservation_root, self.root)
        self._validate_private_directory(self.root, "execution journal root")
        self._validate_private_directory(self.lock_root, "execution journal locks")
        self._validate_private_directory(
            self.reservation_root,
            "execution journal reservations",
        )

    def _prepare_reservation_layout(self, reservation_id: str) -> None:
        with _ExecutionStoreLock(self.initialization_lock_path):
            reservation_root = self._reservation_path(reservation_id)
            self._ensure_private_directory(reservation_root, self.reservation_root)
            self._ensure_private_directory(
                reservation_root / "events",
                reservation_root,
            )
            self._ensure_private_directory(
                reservation_root / "staging",
                reservation_root,
            )

    def _read_events(
        self,
        reservation: ExpertSourceReplayExecutionReservation,
        request: ExpertSourceReplayExecutionRequest,
    ) -> tuple[SourceReplayExecutionJournalEvent, ...]:
        events_root = self._events_path(reservation.reservation_id)
        entries = tuple(sorted(os.scandir(events_root), key=lambda entry: entry.name))
        parsed_entries = []
        seen_numbers = set()
        for entry in entries:
            match = _EVENT_FILENAME_PATTERN.fullmatch(entry.name)
            if match is None:
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal contains an unexpected event entry"
                )
            event_number = int(match.group("number"))
            if event_number in seen_numbers:
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal contains a forked event number"
                )
            seen_numbers.add(event_number)
            payload = self._read_private_file(
                Path(entry.path),
                required_mode=0o400,
                name="execution journal event",
            )
            event = SourceReplayExecutionJournalEvent.from_json_bytes(payload)
            if payload != event.to_json_bytes():
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal event is not canonical"
                )
            event_digest = event.event_id.split(":sha256:", 1)[1]
            if (
                event.event_number != event_number
                or match.group("digest") != event_digest
            ):
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal event filename differs from its identity"
                )
            parsed_entries.append(event)
        events = tuple(parsed_entries)
        self._validate_events(reservation, request, events)
        return events

    @staticmethod
    def _validate_events(
        reservation: ExpertSourceReplayExecutionReservation,
        request: ExpertSourceReplayExecutionRequest,
        events: tuple[SourceReplayExecutionJournalEvent, ...],
    ) -> None:
        _validate_reservation_request(reservation, request)
        if len(events) > 1:
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal contains an unsupported event suffix"
            )
        schedule = source_replay_execution_schedule(reservation, request)
        previous_event_id = None
        seen_nonces = set()
        seen_invocation_ids = set()
        for position, event in enumerate(events, start=1):
            expected_case_id, expected_leg_id = schedule[position - 1]
            allocation = event.invocation_allocation
            if (
                event.event_number != position
                or event.predecessor_event_id != previous_event_id
                or event.event_kind
                is not SourceReplayExecutionJournalEventKind.INVOCATION_ALLOCATED
                or event.reservation_id != reservation.reservation_id
                or event.execution_request_id != request.execution_request_id
                or event.execution_case_id != expected_case_id
                or event.execution_leg_id != expected_leg_id
                or allocation.invocation_nonce in seen_nonces
                or allocation.opaque_invocation_id in seen_invocation_ids
            ):
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal is not an exact authorized schedule prefix"
                )
            seen_nonces.add(allocation.invocation_nonce)
            seen_invocation_ids.add(allocation.opaque_invocation_id)
            previous_event_id = event.event_id

    def _publish_event(
        self,
        reservation_id: str,
        event: SourceReplayExecutionJournalEvent,
    ) -> None:
        staging_root = self._staging_path(reservation_id)
        temporary_path = staging_root / f".event-{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(event.to_json_bytes())
            handle.flush()
            os.fchmod(handle.fileno(), 0o400)
            os.fsync(handle.fileno())
        staged_payload = self._read_private_file(
            temporary_path,
            required_mode=0o400,
            name="staged execution journal event",
        )
        if staged_payload != event.to_json_bytes():
            raise ExpertSourceReplayExecutionStoreError(
                "staged execution event differs from canonical bytes"
            )
        destination = self._event_path(reservation_id, event)
        self._rename_no_replace(temporary_path, destination)
        self._fsync_directory(destination.parent)
        self._fsync_directory(staging_root)

    def _clean_staging(self, reservation_id: str) -> None:
        staging_root = self._staging_path(reservation_id)
        entries = tuple(os.scandir(staging_root))
        for entry in entries:
            if _STAGING_FILENAME_PATTERN.fullmatch(entry.name) is None:
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal staging contains an unexpected entry"
                )
            metadata = entry.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
            ):
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal staging entry is unsafe"
                )
            os.unlink(entry.path)
        if entries:
            self._fsync_directory(staging_root)

    def _event_path(
        self,
        reservation_id: str,
        event: SourceReplayExecutionJournalEvent,
    ) -> Path:
        digest = event.event_id.split(":sha256:", 1)[1]
        return self._events_path(reservation_id) / (
            f"{event.event_number:020d}-{digest}.json"
        )

    def _lock_path(self, reservation_id: str) -> Path:
        digest = self._reservation_digest(reservation_id)
        return self.lock_root / f"{digest}.lock"

    def _reservation_path(self, reservation_id: str) -> Path:
        return self.reservation_root / self._reservation_digest(reservation_id)

    def _events_path(self, reservation_id: str) -> Path:
        return self._reservation_path(reservation_id) / "events"

    def _staging_path(self, reservation_id: str) -> Path:
        return self._reservation_path(reservation_id) / "staging"

    @staticmethod
    def _reservation_digest(reservation_id: str) -> str:
        require_content_id(reservation_id, "source replay execution reservation_id")
        namespace, digest = reservation_id.split(":sha256:", 1)
        if namespace != "expert-source-replay-execution-reservation":
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal reservation uses the wrong namespace"
            )
        return digest

    @staticmethod
    def _ensure_private_directory(path: Path, parent: Path) -> None:
        if not os.path.lexists(path):
            os.mkdir(path, mode=0o700)
            ExpertSourceReplayExecutionStore._fsync_directory(parent)
        ExpertSourceReplayExecutionStore._validate_private_directory(
            path,
            "execution journal directory",
        )

    @staticmethod
    def _validate_private_directory(path: Path, name: str) -> None:
        metadata = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ExpertSourceReplayExecutionStoreError(
                f"{name} must be a private real directory"
            )

    @staticmethod
    def _read_private_file(path: Path, *, required_mode: int, name: str) -> bytes:
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
            ):
                raise ExpertSourceReplayExecutionStoreError(
                    f"{name} must be a private independent regular file"
                )
            return handle.read()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        os.fsync(descriptor)
        os.close(descriptor)

    @staticmethod
    def _rename_no_replace(source: Path, destination: Path) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(libc, "renameat2"):
            raise ExpertSourceReplayExecutionStoreError(
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
            raise OSError(
                error_number,
                "execution journal publication failed: "
                f"{errno.errorcode.get(error_number)}",
            )


class _ExecutionStoreLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None
        self.acquired = False

    def __enter__(self):
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
        ):
            self.handle.close()
            self.handle = None
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal lock must be a private independent file"
            )
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        self.acquired = True
        return self

    def __exit__(self, exception_type, exception, traceback):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.acquired = False
        self.handle.close()
        self.handle = None
        return False
