"""Create-only local execution journal for expert source replay."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import math
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertSourceReplayExecutionLegKind,
    ExpertSourceReplayExecutionRequest,
    ExpertSourceReplayExecutionReservation,
    StrictContract,
)
from kapso.cross_run.expert.replay_protocol import (
    TaskEvaluatorInvocationAllocation,
    TaskEvaluatorRequest,
    TaskEvaluatorResult,
    build_task_evaluator_request,
    parse_task_evaluator_result,
)
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplaySpawnAuthorityFence,
    source_replay_spawn_security_subject_ids,
    source_replay_task_adapter_trust_observations,
)
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayMatchedLegInvocation,
    ExpertSourceReplayProviderCompletion,
    ExpertSourceReplayExecutionProviderKey,
    ResolvedExpertSourceReplayExecutionCase,
    SourceReplayProviderExecutionHandle,
    expert_source_replay_execution_provider_key,
    source_replay_provider_execution_handle,
)
from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest
from kapso.cross_run.process import BoundedProcessOutcome
from kapso.cross_run.settings import ExpertValidationPolicySettings

_EXECUTION_JOURNAL_SCHEMA_VERSION = "kapso.source_replay_execution_journal.v2"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_EVENT_FILENAME_PATTERN = re.compile(r"^(?P<number>[0-9]{20})\.json$")
_STAGING_FILENAME_PATTERN = re.compile(r"^\.(?:event|result)-[0-9a-f]{32}\.tmp$")
_RESULT_FILENAME_PATTERN = re.compile(r"^(?P<digest>[0-9a-f]{64})\.json$")


class ExpertSourceReplayExecutionStoreError(ValueError):
    """The private execution journal is unsafe, corrupt, or conflicting."""


class SourceReplayExecutionJournalEventKind(str, Enum):
    INVOCATION_ALLOCATED = "invocation_allocated"
    SPAWN_COMMITTED = "spawn_committed"
    RESULT_RECEIVED = "result_received"
    RESULT_ACCEPTED = "result_accepted"


@dataclass(frozen=True)
class SourceReplayProcessObservation(StrictContract):
    outcome: BoundedProcessOutcome
    returncode: int
    stdout_bytes_observed: int
    stderr_bytes_observed: int
    duration_seconds: float

    def _validate(self) -> None:
        if type(self.returncode) is not int:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay process returncode must be an integer"
            )
        if (
            type(self.stdout_bytes_observed) is not int
            or self.stdout_bytes_observed < 0
            or type(self.stderr_bytes_observed) is not int
            or self.stderr_bytes_observed < 0
            or type(self.duration_seconds) is not float
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0.0
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay process observation is invalid"
            )


@dataclass(frozen=True)
class SourceReplayResultBlob(StrictContract):
    digest: str
    size: int

    def _validate(self) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest) is None:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay result blob digest is invalid"
            )
        if type(self.size) is not int or self.size < 0:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay result blob size must be non-negative"
            )


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
    spawn_authority_fence: SourceReplaySpawnAuthorityFence | None
    execution_provider_key: ExpertSourceReplayExecutionProviderKey | None
    provider_execution_handle: SourceReplayProviderExecutionHandle | None
    task_evaluator_request: TaskEvaluatorRequest | None
    aggregate_tolerance: float | None
    process_observation: SourceReplayProcessObservation | None
    result_blob: SourceReplayResultBlob | None
    task_evaluator_result: TaskEvaluatorResult | None

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
        allocated_shape = (
            self.spawn_authority_fence is None
            and self.execution_provider_key is None
            and self.provider_execution_handle is None
            and self.task_evaluator_request is None
            and self.aggregate_tolerance is None
            and self.process_observation is None
            and self.result_blob is None
            and self.task_evaluator_result is None
        )
        spawn_shape = (
            self.spawn_authority_fence is not None
            and self.execution_provider_key is not None
            and self.provider_execution_handle is not None
            and self.task_evaluator_request is not None
            and self.aggregate_tolerance is not None
            and self.process_observation is None
            and self.result_blob is None
            and self.task_evaluator_result is None
        )
        received_shape = (
            self.spawn_authority_fence is None
            and self.execution_provider_key is None
            and self.provider_execution_handle is None
            and self.task_evaluator_request is None
            and self.aggregate_tolerance is None
            and self.process_observation is not None
            and self.task_evaluator_result is None
        )
        accepted_shape = (
            self.spawn_authority_fence is None
            and self.execution_provider_key is None
            and self.provider_execution_handle is None
            and self.task_evaluator_request is None
            and self.aggregate_tolerance is None
            and self.process_observation is None
            and self.result_blob is None
            and self.task_evaluator_result is not None
        )
        expected_shape = {
            SourceReplayExecutionJournalEventKind.INVOCATION_ALLOCATED: allocated_shape,
            SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED: spawn_shape,
            SourceReplayExecutionJournalEventKind.RESULT_RECEIVED: received_shape,
            SourceReplayExecutionJournalEventKind.RESULT_ACCEPTED: accepted_shape,
        }[self.event_kind]
        if not expected_shape:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay execution event payload differs from its kind"
            )
        if self.aggregate_tolerance is not None and (
            type(self.aggregate_tolerance) is not float
            or not math.isfinite(self.aggregate_tolerance)
            or self.aggregate_tolerance < 0.0
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn aggregate tolerance is invalid"
            )
        if self.spawn_authority_fence is not None and (
            self.spawn_authority_fence.reservation_id != self.reservation_id
            or self.spawn_authority_fence.execution_request_id
            != self.execution_request_id
            or self.spawn_authority_fence.invocation_allocation != allocation
            or self.provider_execution_handle.dispatch_key
            != self.execution_provider_key
            or self.provider_execution_handle.invocation_allocation != allocation
            or self.task_evaluator_request.opaque_invocation_id
            != allocation.opaque_invocation_id
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn payload differs from its event"
            )
        if (
            self.process_observation is not None
            and (
                self.process_observation.outcome is not BoundedProcessOutcome.COMPLETED
                or self.process_observation.returncode != 0
            )
            and self.result_blob is not None
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay failed process cannot publish a result blob"
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


_SPAWN_AUTHORIZATION_SEAL = object()
_PROVIDER_COMPLETION_SEAL = object()


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


class SourceReplaySpawnAuthorizationPermit:
    """Runtime-only fresh authority bound to one live journal allocation."""

    __slots__ = (
        "_store",
        "_coordinator",
        "_allocation_permit",
        "_consumed",
        "_prepared_request",
        "resolved_case",
        "fence",
        "execution_provider_key",
        "provider_execution_handle",
        "task_evaluator_request",
        "aggregate_tolerance",
    )

    def __init__(
        self,
        seal: object,
        store: ExpertSourceReplayExecutionStore,
        coordinator: object,
        allocation_permit: SourceReplayInvocationAllocationPermit,
        prepared_request: PreparedExpertSourceReplayRequest,
        resolved_case: ResolvedExpertSourceReplayExecutionCase,
        fence: SourceReplaySpawnAuthorityFence,
        execution_provider_key: ExpertSourceReplayExecutionProviderKey,
        provider_execution_handle: SourceReplayProviderExecutionHandle,
        task_evaluator_request: TaskEvaluatorRequest,
        aggregate_tolerance: float,
    ) -> None:
        if seal is not _SPAWN_AUTHORIZATION_SEAL:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn authorization is not sealed"
            )
        self._store = store
        self._coordinator = coordinator
        self._allocation_permit = allocation_permit
        self._consumed = False
        self._prepared_request = prepared_request
        self.resolved_case = resolved_case
        self.fence = fence
        self.execution_provider_key = execution_provider_key
        self.provider_execution_handle = provider_execution_handle
        self.task_evaluator_request = task_evaluator_request
        self.aggregate_tolerance = aggregate_tolerance

    def _consume(
        self,
        session: _SourceReplayReservationSession,
    ) -> None:
        if self._consumed or session._store is not self._store:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn authorization is consumed or foreign"
            )
        allocation = self._allocation_permit.require_current_allocation(self._store)
        matching_cases = tuple(
            case
            for case in self._prepared_request.cases
            if case.request_case.execution_case_id == allocation.execution_case_id
        )
        if (
            session._allocation_permit is not self._allocation_permit
            or len(matching_cases) != 1
            or self.fence.invocation_allocation != allocation
            or self.task_evaluator_request.opaque_invocation_id
            != allocation.opaque_invocation_id
            or build_task_evaluator_request(matching_cases[0], allocation)
            != self.task_evaluator_request
            or expert_source_replay_execution_provider_key(matching_cases[0])
            != self.execution_provider_key
            or self.resolved_case.materialized_case != matching_cases[0]
            or self.resolved_case.dispatch_key != self.execution_provider_key
            or self.provider_execution_handle
            != source_replay_provider_execution_handle(
                self.execution_provider_key,
                allocation,
            )
            or source_replay_spawn_security_subject_ids(
                self._prepared_request,
                session.reservation,
                self.fence.current_release_observation,
                self.fence.task_adapter_trust_observations,
            )
            != self.fence.security_subject_ids
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn authorization differs from its live allocation"
            )
        self.resolved_case.require_current_provider_identity()
        self._consumed = True


class SourceReplaySpawnPermit:
    """One-shot journal-owned executor issued after the durable spawn boundary."""

    __slots__ = (
        "_session",
        "_spawn_event_id",
        "_execution_started",
        "_execution_guard",
        "_resolved_case",
        "_invocation",
    )

    def __init__(
        self,
        session: _SourceReplayReservationSession,
        event: SourceReplayExecutionJournalEvent,
        resolved_case: ResolvedExpertSourceReplayExecutionCase,
    ) -> None:
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_spawn_event_id", event.event_id)
        object.__setattr__(self, "_execution_started", False)
        object.__setattr__(self, "_execution_guard", Lock())
        object.__setattr__(self, "_resolved_case", resolved_case)
        object.__setattr__(
            self,
            "_invocation",
            ExpertSourceReplayMatchedLegInvocation(
                materialized_case=resolved_case.materialized_case,
                expert_source=resolved_case._expert_source_for(
                    event.invocation_allocation
                ),
                invocation_allocation=event.invocation_allocation,
                task_evaluator_request=event.task_evaluator_request,
                provider_handle=event.provider_execution_handle,
            ),
        )

    def __setattr__(self, name, value) -> None:
        raise ExpertSourceReplayExecutionStoreError(
            "source replay execution capability is immutable"
        )

    def execute(self) -> SourceReplaySealedLegCompletion:
        with self._execution_guard:
            self._require_current()
            if self._execution_started:
                raise ExpertSourceReplayExecutionStoreError(
                    "source replay spawn execution was already consumed"
                )
            object.__setattr__(self, "_execution_started", True)
        provider_completion = self._resolved_case._execute_leg(self._invocation)
        return self._session._seal_provider_completion(
            self,
            provider_completion,
        )

    def _require_current(self) -> None:
        self._session._require_live_store_lock(self._session._store)
        if (
            self._session._spawn_permit is not self
            or not self._session._events
            or self._session._events[-1].event_id != self._spawn_event_id
            or self._session._events[-1].event_kind
            is not SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn permit is not current"
            )


class SourceReplaySealedLegCompletion:
    """Runtime-only provider completion owned by one live journal session."""

    __slots__ = (
        "_session",
        "_spawn_permit",
        "_provider_completion",
    )

    def __init__(
        self,
        seal: object,
        session: _SourceReplayReservationSession,
        spawn_permit: SourceReplaySpawnPermit,
        provider_completion: ExpertSourceReplayProviderCompletion,
    ) -> None:
        if seal is not _PROVIDER_COMPLETION_SEAL:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay provider completion is not journal sealed"
            )
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_spawn_permit", spawn_permit)
        object.__setattr__(self, "_provider_completion", provider_completion)

    def __setattr__(self, name, value) -> None:
        raise ExpertSourceReplayExecutionStoreError(
            "source replay sealed provider completion is immutable"
        )

    def _consume(
        self,
        session: _SourceReplayReservationSession,
    ) -> ExpertSourceReplayProviderCompletion:
        if (
            self._session is not session
            or session._pending_completion is not self
            or session._spawn_permit is not self._spawn_permit
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay provider completion is consumed or foreign"
            )
        self._spawn_permit._require_current()
        session._pending_completion = None
        return self._provider_completion


class _SourceReplayReservationSession:
    """One exclusively locked reservation execution prefix."""

    def __init__(
        self,
        store: ExpertSourceReplayExecutionStore,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
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
        self.prepared_request = prepared_request
        self.request = prepared_request.request
        self._events = events
        self._execution_lock = execution_lock
        self._owner_process_id = os.getpid()
        self._active = True
        self._append_poisoned = False
        self._allocation_permit = None
        self._spawn_permit = None
        self._pending_completion = None

    @property
    def events(self) -> tuple[SourceReplayExecutionJournalEvent, ...]:
        self._require_active()
        return self._events

    def allocate_expected_leg(self) -> SourceReplayInvocationAllocationPermit:
        self._require_active()
        phase = len(self._events) % 4
        if phase == 1:
            if self._allocation_permit is None:
                self._allocation_permit = SourceReplayInvocationAllocationPermit(
                    self,
                    self._events[-1],
                )
            return self._allocation_permit
        if phase == 2:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn marker is permanently interrupted after reopen"
            )
        if phase == 3:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay received result must be accepted before another leg"
            )
        schedule = source_replay_execution_schedule(
            self.reservation,
            self.request,
        )
        schedule_position = len(self._events) // 4
        if schedule_position >= len(schedule):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay execution schedule is complete"
            )
        execution_case_id, execution_leg_id = schedule[schedule_position]
        allocation = TaskEvaluatorInvocationAllocation(
            reservation_id=self.reservation.reservation_id,
            execution_case_id=execution_case_id,
            execution_leg_id=execution_leg_id,
            invocation_nonce=_new_invocation_nonce(),
        )
        event = SourceReplayExecutionJournalEvent.mint(
            schema_version=_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=len(self._events) + 1,
            predecessor_event_id=(
                None if not self._events else self._events[-1].event_id
            ),
            event_kind=SourceReplayExecutionJournalEventKind.INVOCATION_ALLOCATED,
            reservation_id=self.reservation.reservation_id,
            execution_request_id=self.request.execution_request_id,
            execution_case_id=execution_case_id,
            execution_leg_id=execution_leg_id,
            invocation_allocation=allocation,
            spawn_authority_fence=None,
            execution_provider_key=None,
            provider_execution_handle=None,
            task_evaluator_request=None,
            aggregate_tolerance=None,
            process_observation=None,
            result_blob=None,
            task_evaluator_result=None,
        )
        self._poison_for_append()
        self._store._publish_event(self.reservation.reservation_id, event)
        self._events = (*self._events, event)
        self._append_poisoned = False
        self._allocation_permit = SourceReplayInvocationAllocationPermit(
            self,
            event,
        )
        return self._allocation_permit

    def _commit_spawn(
        self,
        authorization: SourceReplaySpawnAuthorizationPermit,
    ) -> SourceReplaySpawnPermit:
        self._require_active()
        if type(authorization) is not SourceReplaySpawnAuthorizationPermit:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn requires sealed fresh authorization"
            )
        if len(self._events) % 4 != 1 or self._events[-1].event_kind is not (
            SourceReplayExecutionJournalEventKind.INVOCATION_ALLOCATED
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay spawn requires the current allocation tail"
            )
        authorization._consume(self)
        allocation = self._events[-1].invocation_allocation
        event = SourceReplayExecutionJournalEvent.mint(
            schema_version=_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=len(self._events) + 1,
            predecessor_event_id=self._events[-1].event_id,
            event_kind=SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED,
            reservation_id=self.reservation.reservation_id,
            execution_request_id=self.request.execution_request_id,
            execution_case_id=allocation.execution_case_id,
            execution_leg_id=allocation.execution_leg_id,
            invocation_allocation=allocation,
            spawn_authority_fence=authorization.fence,
            execution_provider_key=authorization.execution_provider_key,
            provider_execution_handle=authorization.provider_execution_handle,
            task_evaluator_request=authorization.task_evaluator_request,
            aggregate_tolerance=authorization.aggregate_tolerance,
            process_observation=None,
            result_blob=None,
            task_evaluator_result=None,
        )
        self._poison_for_append()
        self._store._publish_event(self.reservation.reservation_id, event)
        self._events = (*self._events, event)
        self._append_poisoned = False
        self._spawn_permit = SourceReplaySpawnPermit(
            self,
            event,
            authorization.resolved_case,
        )
        return self._spawn_permit

    def _seal_provider_completion(
        self,
        spawn_permit: SourceReplaySpawnPermit,
        provider_completion: ExpertSourceReplayProviderCompletion,
    ) -> SourceReplaySealedLegCompletion:
        self._require_active()
        spawn_permit._require_current()
        if (
            type(provider_completion) is not ExpertSourceReplayProviderCompletion
            or self._pending_completion is not None
            or not spawn_permit._execution_started
            or provider_completion.provider_handle_id
            != spawn_permit._invocation.provider_handle.provider_handle_id
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay provider completion differs from its live spawn"
            )
        process_result = provider_completion.process_result
        result_payload = provider_completion.result_payload
        compute = (
            spawn_permit._resolved_case.materialized_case.request_case.compute_binding
        )
        process_request = process_result.request
        if (
            process_request.timeout_seconds != compute.leg_wall_time_limit_seconds
            or process_request.cleanup_timeout_seconds
            != compute.termination_grace_seconds
            or process_request.stdout_byte_limit != compute.stdout_byte_limit
            or process_request.stderr_byte_limit != compute.stderr_byte_limit
            or process_result.stdout_bytes_observed > compute.stdout_byte_limit
            or process_result.stderr_bytes_observed > compute.stderr_byte_limit
            or process_result.stdout_bytes_observed < len(process_result.stdout)
            or process_result.stderr_bytes_observed < len(process_result.stderr)
            or len(process_result.stdout) > compute.stdout_byte_limit
            or len(process_result.stderr) > compute.stderr_byte_limit
            or (
                result_payload is not None
                and len(result_payload)
                > min(
                    compute.output_byte_limit,
                    self._store.maximum_result_size_bytes,
                )
            )
            or (
                (
                    process_result.outcome is not BoundedProcessOutcome.COMPLETED
                    or process_result.returncode != 0
                )
                and result_payload is not None
            )
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay provider completion exceeds its exact compute authority"
            )
        sealed = SourceReplaySealedLegCompletion(
            _PROVIDER_COMPLETION_SEAL,
            self,
            spawn_permit,
            provider_completion,
        )
        self._pending_completion = sealed
        return sealed

    def record_result_received(
        self,
        completion: SourceReplaySealedLegCompletion,
    ) -> SourceReplayExecutionJournalEvent:
        self._require_active()
        if type(completion) is not SourceReplaySealedLegCompletion:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay result requires a journal-sealed provider completion"
            )
        provider_completion = completion._consume(self)
        process_result = provider_completion.process_result
        result_payload = provider_completion.result_payload
        self._poison_for_append()
        result_blob = (
            None
            if result_payload is None
            else self._store._publish_result_blob(
                self.reservation.reservation_id,
                result_payload,
            )
        )
        allocation = self._events[-1].invocation_allocation
        event = SourceReplayExecutionJournalEvent.mint(
            schema_version=_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=len(self._events) + 1,
            predecessor_event_id=self._events[-1].event_id,
            event_kind=SourceReplayExecutionJournalEventKind.RESULT_RECEIVED,
            reservation_id=self.reservation.reservation_id,
            execution_request_id=self.request.execution_request_id,
            execution_case_id=allocation.execution_case_id,
            execution_leg_id=allocation.execution_leg_id,
            invocation_allocation=allocation,
            spawn_authority_fence=None,
            execution_provider_key=None,
            provider_execution_handle=None,
            task_evaluator_request=None,
            aggregate_tolerance=None,
            process_observation=SourceReplayProcessObservation(
                outcome=process_result.outcome,
                returncode=process_result.returncode,
                stdout_bytes_observed=process_result.stdout_bytes_observed,
                stderr_bytes_observed=process_result.stderr_bytes_observed,
                duration_seconds=process_result.duration_seconds,
            ),
            result_blob=result_blob,
            task_evaluator_result=None,
        )
        self._store._publish_event(self.reservation.reservation_id, event)
        self._events = (*self._events, event)
        self._append_poisoned = False
        return event

    def accept_received_result(self) -> TaskEvaluatorResult:
        self._require_active()
        if len(self._events) % 4 != 3 or self._events[-1].event_kind is not (
            SourceReplayExecutionJournalEventKind.RESULT_RECEIVED
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay result acceptance requires a received-result tail"
            )
        received_event = self._events[-1]
        spawn_event = self._events[-2]
        if received_event.result_blob is None:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay process produced no acceptable result"
            )
        payload = self._store._read_result_blob(
            self.reservation.reservation_id,
            received_event.result_blob,
        )
        result = parse_task_evaluator_result(
            payload,
            spawn_event.task_evaluator_request,
            spawn_event.aggregate_tolerance,
        )
        allocation = received_event.invocation_allocation
        event = SourceReplayExecutionJournalEvent.mint(
            schema_version=_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=len(self._events) + 1,
            predecessor_event_id=received_event.event_id,
            event_kind=SourceReplayExecutionJournalEventKind.RESULT_ACCEPTED,
            reservation_id=self.reservation.reservation_id,
            execution_request_id=self.request.execution_request_id,
            execution_case_id=allocation.execution_case_id,
            execution_leg_id=allocation.execution_leg_id,
            invocation_allocation=allocation,
            spawn_authority_fence=None,
            execution_provider_key=None,
            provider_execution_handle=None,
            task_evaluator_request=None,
            aggregate_tolerance=None,
            process_observation=None,
            result_blob=None,
            task_evaluator_result=result,
        )
        self._poison_for_append()
        self._store._publish_event(self.reservation.reservation_id, event)
        self._events = (*self._events, event)
        self._append_poisoned = False
        return result

    def _require_active(self) -> None:
        if not self._active:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay reservation session is closed"
            )
        if self._append_poisoned:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay reservation session must reopen after append"
            )

    def _poison_for_append(self) -> None:
        self._require_active()
        self._append_poisoned = True
        self._allocation_permit = None
        self._spawn_permit = None
        self._pending_completion = None

    def _require_live_store_lock(
        self,
        execution_store: ExpertSourceReplayExecutionStore,
    ) -> None:
        self._require_active()
        if (
            execution_store is not self._store
            or os.getpid() != self._owner_process_id
            or not isinstance(self._execution_lock, _ExecutionStoreLock)
            or not self._execution_lock.acquired
            or self._execution_lock.handle is None
            or self._execution_lock.path
            != execution_store._lock_path(self.reservation.reservation_id)
            or execution_store._active_sessions.get(self.reservation.reservation_id)
            is not self
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay runtime authority lacks its creator process and canonical live store lock"
            )

    def _close(self) -> None:
        self._active = False
        self._append_poisoned = True
        self._allocation_permit = None
        self._spawn_permit = None
        self._pending_completion = None


class _ReservationSessionContext:
    def __init__(
        self,
        store: ExpertSourceReplayExecutionStore,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> None:
        self.store = store
        self.reservation = reservation
        self.prepared_request = prepared_request
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
                self.prepared_request,
            )
            self.session = _SourceReplayReservationSession(
                self.store,
                self.reservation,
                self.prepared_request,
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

    def __init__(
        self,
        root: Path,
        trusted_root: Path,
        policy_settings: ExpertValidationPolicySettings,
    ) -> None:
        if (
            not isinstance(trusted_root, Path)
            or not trusted_root.is_absolute()
            or trusted_root.resolve() != trusted_root
            or not trusted_root.is_dir()
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal trusted root must be a resolved directory"
            )
        trusted_root_metadata = os.stat(trusted_root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(trusted_root_metadata.st_mode)
            or stat.S_IMODE(trusted_root_metadata.st_mode) != 0o700
            or trusted_root_metadata.st_uid != os.geteuid()
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal trusted root must be owner-private"
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
        if not isinstance(policy_settings, ExpertValidationPolicySettings):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal requires canonical validation policy settings"
            )
        self.root = root
        self.trusted_root = trusted_root
        self.policy_settings = policy_settings
        self.maximum_event_size_bytes = (
            policy_settings.source_replay_journal_event_byte_limit
        )
        self.maximum_result_size_bytes = policy_settings.source_replay_result_byte_limit
        self.maximum_staging_entry_count = (
            policy_settings.source_replay_staging_entry_limit
        )
        self.lock_root = root / "locks"
        self.reservation_root = root / "reservations"
        self.initialization_lock_path = trusted_root / f".{root.name}.lock"
        self._session_factory_authority = object()
        self._spawn_authority_type = None
        self._active_sessions = {}
        with _ExecutionStoreLock(self.initialization_lock_path):
            self._prepare_layout()

    def reservation_session(
        self,
        *,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> _ReservationSessionContext:
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal requires its prepared byte authority"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            parent=prepared_request.parent,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        if (
            prepared.settings.policy != self.policy_settings
            or prepared.request.validation_policy_id
            != self.policy_settings.validation_policy().validation_policy_id
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal prepared authority uses another validation policy"
            )
        _validate_reservation_request(reservation, prepared.request)
        return _ReservationSessionContext(self, reservation, prepared)

    def _bind_spawn_authority(self, coordinator_type: type[object]) -> None:
        if (
            coordinator_type.__module__ != "kapso.cross_run.expert.replay_authority"
            or coordinator_type.__qualname__
            != "ExpertSourceReplayFreshAuthorityCoordinator"
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal spawn authority type is invalid"
            )
        if (
            self._spawn_authority_type is not None
            and self._spawn_authority_type is not coordinator_type
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal spawn authority is already bound"
            )
        self._spawn_authority_type = coordinator_type

    def _seal_spawn_authorization(
        self,
        *,
        coordinator: object,
        allocation_permit: SourceReplayInvocationAllocationPermit,
        prepared_request: PreparedExpertSourceReplayRequest,
        resolved_case: ResolvedExpertSourceReplayExecutionCase,
        fence: SourceReplaySpawnAuthorityFence,
        aggregate_tolerance: float,
    ) -> SourceReplaySpawnAuthorizationPermit:
        if type(coordinator) is not self._spawn_authority_type:
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal spawn authorization lacks its coordinator"
            )
        allocation = allocation_permit.require_current_allocation(self)
        session = allocation_permit._session
        reservation = session.reservation
        request = session.request
        if (
            not isinstance(prepared_request, PreparedExpertSourceReplayRequest)
            or type(resolved_case) is not ResolvedExpertSourceReplayExecutionCase
            or prepared_request != session.prepared_request
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal spawn resolution differs from its live session"
            )
        matching_cases = tuple(
            case
            for case in prepared_request.cases
            if case.request_case.execution_case_id == allocation.execution_case_id
        )
        if (
            len(matching_cases) != 1
            or resolved_case.materialized_case != matching_cases[0]
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal spawn resolution names another case"
            )
        resolved_case.require_exact_prepared_authority(prepared_request)
        resolved_case.require_current_provider_identity()
        execution_provider_key = expert_source_replay_execution_provider_key(
            matching_cases[0]
        )
        task_evaluator_request = build_task_evaluator_request(
            matching_cases[0],
            allocation,
        )
        provider_execution_handle = source_replay_provider_execution_handle(
            execution_provider_key,
            allocation,
        )
        if (
            not isinstance(fence, SourceReplaySpawnAuthorityFence)
            or fence.invocation_allocation != allocation
            or fence.reservation_id != reservation.reservation_id
            or fence.execution_request_id != request.execution_request_id
            or fence.authorization_transition_id
            != reservation.authorization_transition_id
            or fence.authorization_state_id != reservation.authorization_state_id
            or fence.candidate_id != reservation.candidate_id
            or prepared_request.request != request
            or task_evaluator_request.opaque_invocation_id
            != allocation.opaque_invocation_id
            or len(task_evaluator_request.to_json_bytes())
            > self.policy_settings.source_replay_task_request_byte_limit
            or aggregate_tolerance
            != self.policy_settings.source_replay_score_comparison_tolerance
            or not {
                reservation.reservation_id,
                *reservation.exact_dependency_ids,
                request.execution_request_id,
                *request.exact_dependency_ids,
            }.issubset(fence.security_subject_ids)
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal spawn authorization differs from journal authority"
            )
        return SourceReplaySpawnAuthorizationPermit(
            _SPAWN_AUTHORIZATION_SEAL,
            self,
            coordinator,
            allocation_permit,
            prepared_request,
            resolved_case,
            fence,
            execution_provider_key,
            provider_execution_handle,
            task_evaluator_request,
            aggregate_tolerance,
        )

    def _commit_spawn_authorization(
        self,
        *,
        coordinator: object,
        authorization: SourceReplaySpawnAuthorizationPermit,
    ) -> SourceReplaySpawnPermit:
        if (
            type(coordinator) is not self._spawn_authority_type
            or type(authorization) is not SourceReplaySpawnAuthorizationPermit
            or authorization._store is not self
            or authorization._coordinator is not coordinator
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal spawn commit lacks fresh coordinator authority"
            )
        return authorization._allocation_permit._session._commit_spawn(authorization)

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
            self._ensure_private_directory(
                reservation_root / "results",
                reservation_root,
            )

    def _read_events(
        self,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> tuple[SourceReplayExecutionJournalEvent, ...]:
        request = prepared_request.request
        events_root = self._events_path(reservation.reservation_id)
        maximum_event_count = 4 * len(
            source_replay_execution_schedule(reservation, request)
        )
        scanned_entries = []
        with os.scandir(events_root) as entries:
            for entry in entries:
                scanned_entries.append(entry)
                if len(scanned_entries) > maximum_event_count:
                    raise ExpertSourceReplayExecutionStoreError(
                        "execution journal exceeds its structural event bound"
                    )
        entries = tuple(sorted(scanned_entries, key=lambda entry: entry.name))
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
                maximum_size_bytes=self.maximum_event_size_bytes,
            )
            event = SourceReplayExecutionJournalEvent.from_json_bytes(payload)
            if payload != event.to_json_bytes():
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal event is not canonical"
                )
            if event.event_number != event_number:
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal event filename differs from its identity"
                )
            parsed_entries.append(event)
        events = tuple(parsed_entries)
        self._validate_result_entries(
            reservation.reservation_id,
            len(source_replay_execution_schedule(reservation, request)),
        )
        self._validate_events(reservation, prepared_request, events)
        return events

    def _validate_events(
        self,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
        events: tuple[SourceReplayExecutionJournalEvent, ...],
    ) -> None:
        request = prepared_request.request
        _validate_reservation_request(reservation, request)
        schedule = source_replay_execution_schedule(reservation, request)
        if len(events) > 4 * len(schedule):
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal contains an unsupported event suffix"
            )
        phases = (
            SourceReplayExecutionJournalEventKind.INVOCATION_ALLOCATED,
            SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED,
            SourceReplayExecutionJournalEventKind.RESULT_RECEIVED,
            SourceReplayExecutionJournalEventKind.RESULT_ACCEPTED,
        )
        previous_event_id = None
        seen_nonces = set()
        seen_invocation_ids = set()
        for position, event in enumerate(events, start=1):
            schedule_position = (position - 1) // 4
            phase_position = (position - 1) % 4
            expected_case_id, expected_leg_id = schedule[schedule_position]
            expected_kind = phases[phase_position]
            allocation = event.invocation_allocation
            allocation_event = events[schedule_position * 4]
            if (
                event.event_number != position
                or event.predecessor_event_id != previous_event_id
                or event.event_kind is not expected_kind
                or event.reservation_id != reservation.reservation_id
                or event.execution_request_id != request.execution_request_id
                or event.execution_case_id != expected_case_id
                or event.execution_leg_id != expected_leg_id
                or allocation != allocation_event.invocation_allocation
            ):
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal is not an exact authorized schedule prefix"
                )
            if phase_position == 0:
                if (
                    allocation.invocation_nonce in seen_nonces
                    or allocation.opaque_invocation_id in seen_invocation_ids
                ):
                    raise ExpertSourceReplayExecutionStoreError(
                        "execution journal reuses an invocation identity"
                    )
                seen_nonces.add(allocation.invocation_nonce)
                seen_invocation_ids.add(allocation.opaque_invocation_id)
            elif phase_position == 1:
                fence = event.spawn_authority_fence
                matching_cases = tuple(
                    case
                    for case in prepared_request.cases
                    if case.request_case.execution_case_id == expected_case_id
                )
                if (
                    len(matching_cases) != 1
                    or fence.authorization_transition_id
                    != reservation.authorization_transition_id
                    or fence.authorization_state_id
                    != reservation.authorization_state_id
                    or fence.candidate_id != reservation.candidate_id
                    or fence.scope_id
                    != prepared_request.parent.release_manifest.scope_id
                    or fence.expected_parent_release_id != request.parent_release_id
                    or fence.scope_contract_id != request.scope_contract_id
                    or fence.task_adapter_trust_observations
                    != source_replay_task_adapter_trust_observations(prepared_request)
                    or event.execution_provider_key
                    != expert_source_replay_execution_provider_key(matching_cases[0])
                    or event.provider_execution_handle
                    != source_replay_provider_execution_handle(
                        event.execution_provider_key,
                        allocation,
                    )
                    or event.task_evaluator_request
                    != build_task_evaluator_request(matching_cases[0], allocation)
                    or len(event.task_evaluator_request.to_json_bytes())
                    > self.policy_settings.source_replay_task_request_byte_limit
                    or event.aggregate_tolerance
                    != self.policy_settings.source_replay_score_comparison_tolerance
                    or source_replay_spawn_security_subject_ids(
                        prepared_request,
                        reservation,
                        fence.current_release_observation,
                        fence.task_adapter_trust_observations,
                    )
                    != fence.security_subject_ids
                ):
                    raise ExpertSourceReplayExecutionStoreError(
                        "execution journal spawn fence differs from its reservation"
                    )
            elif phase_position == 2:
                request_case = next(
                    case
                    for case in request.cases
                    if case.execution_case_id == expected_case_id
                )
                if (
                    event.process_observation.stdout_bytes_observed
                    > request_case.compute_binding.stdout_byte_limit
                    or event.process_observation.stderr_bytes_observed
                    > request_case.compute_binding.stderr_byte_limit
                    or (
                        event.result_blob is not None
                        and event.result_blob.size
                        > request_case.compute_binding.output_byte_limit
                    )
                ):
                    raise ExpertSourceReplayExecutionStoreError(
                        "execution journal result exceeds its persisted compute bounds"
                    )
                if event.result_blob is not None:
                    self._read_result_blob(
                        reservation.reservation_id,
                        event.result_blob,
                    )
            elif phase_position == 3:
                spawn_event = events[schedule_position * 4 + 1]
                received_event = events[schedule_position * 4 + 2]
                if received_event.result_blob is None:
                    raise ExpertSourceReplayExecutionStoreError(
                        "execution journal accepted a missing result blob"
                    )
                parsed = parse_task_evaluator_result(
                    self._read_result_blob(
                        reservation.reservation_id,
                        received_event.result_blob,
                    ),
                    spawn_event.task_evaluator_request,
                    spawn_event.aggregate_tolerance,
                )
                if parsed != event.task_evaluator_result:
                    raise ExpertSourceReplayExecutionStoreError(
                        "execution journal accepted result differs from its blob"
                    )
            previous_event_id = event.event_id

    def _publish_event(
        self,
        reservation_id: str,
        event: SourceReplayExecutionJournalEvent,
    ) -> None:
        payload = event.to_json_bytes()
        if len(payload) > self.maximum_event_size_bytes:
            raise ExpertSourceReplayExecutionStoreError(
                "execution journal event exceeds its configured bound"
            )
        staging_root = self._staging_path(reservation_id)
        temporary_path = staging_root / f".event-{secrets.token_hex(16)}.tmp"
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
        staged_payload = self._read_private_file(
            temporary_path,
            required_mode=0o400,
            name="staged execution journal event",
            maximum_size_bytes=self.maximum_event_size_bytes,
        )
        if staged_payload != payload:
            raise ExpertSourceReplayExecutionStoreError(
                "staged execution event differs from canonical bytes"
            )
        destination = self._event_path(reservation_id, event)
        self._rename_no_replace(temporary_path, destination)
        self._fsync_directory(destination.parent)
        self._fsync_directory(staging_root)

    def _clean_staging(self, reservation_id: str) -> None:
        staging_root = self._staging_path(reservation_id)
        scanned_entries = []
        with os.scandir(staging_root) as entries:
            for entry in entries:
                scanned_entries.append(entry)
                if len(scanned_entries) > self.maximum_staging_entry_count:
                    raise ExpertSourceReplayExecutionStoreError(
                        "execution journal staging exceeds its configured bound"
                    )
        entries = tuple(scanned_entries)
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
                or metadata.st_uid != os.geteuid()
            ):
                raise ExpertSourceReplayExecutionStoreError(
                    "execution journal staging entry is unsafe"
                )
            os.unlink(entry.path)
        if entries:
            self._fsync_directory(staging_root)

    def _publish_result_blob(
        self,
        reservation_id: str,
        payload: bytes,
    ) -> SourceReplayResultBlob:
        if len(payload) > self.maximum_result_size_bytes:
            raise ExpertSourceReplayExecutionStoreError(
                "source replay result blob exceeds its configured bound"
            )
        result_blob = SourceReplayResultBlob(
            digest=tree_or_blob_digest(payload),
            size=len(payload),
        )
        staging_root = self._staging_path(reservation_id)
        temporary_path = staging_root / f".result-{secrets.token_hex(16)}.tmp"
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
        staged_payload = self._read_private_file(
            temporary_path,
            required_mode=0o400,
            name="staged source replay result",
            maximum_size_bytes=self.maximum_result_size_bytes,
        )
        if staged_payload != payload:
            raise ExpertSourceReplayExecutionStoreError(
                "staged source replay result differs from its payload"
            )
        destination = self._result_path(reservation_id, result_blob)
        self._rename_no_replace(temporary_path, destination)
        self._fsync_directory(destination.parent)
        self._fsync_directory(staging_root)
        return result_blob

    def _read_result_blob(
        self,
        reservation_id: str,
        result_blob: SourceReplayResultBlob,
    ) -> bytes:
        payload = self._read_private_file(
            self._result_path(reservation_id, result_blob),
            required_mode=0o400,
            name="source replay result blob",
            maximum_size_bytes=self.maximum_result_size_bytes,
        )
        if (
            len(payload) != result_blob.size
            or tree_or_blob_digest(payload) != result_blob.digest
        ):
            raise ExpertSourceReplayExecutionStoreError(
                "source replay result blob differs from its descriptor"
            )
        return payload

    def _validate_result_entries(
        self,
        reservation_id: str,
        maximum_result_count: int,
    ) -> None:
        result_root = self._results_path(reservation_id)
        entry_count = 0
        with os.scandir(result_root) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > maximum_result_count:
                    raise ExpertSourceReplayExecutionStoreError(
                        "source replay result store exceeds its structural bound"
                    )
                match = _RESULT_FILENAME_PATTERN.fullmatch(entry.name)
                if match is None:
                    raise ExpertSourceReplayExecutionStoreError(
                        "source replay result store contains an unexpected entry"
                    )
                payload = self._read_private_file(
                    Path(entry.path),
                    required_mode=0o400,
                    name="source replay result blob",
                    maximum_size_bytes=self.maximum_result_size_bytes,
                )
                if tree_or_blob_digest(payload).removeprefix("sha256:") != match.group(
                    "digest"
                ):
                    raise ExpertSourceReplayExecutionStoreError(
                        "source replay result filename differs from its payload"
                    )

    def _event_path(
        self,
        reservation_id: str,
        event: SourceReplayExecutionJournalEvent,
    ) -> Path:
        return self._events_path(reservation_id) / f"{event.event_number:020d}.json"

    def _lock_path(self, reservation_id: str) -> Path:
        digest = self._reservation_digest(reservation_id)
        return self.lock_root / f"{digest}.lock"

    def _reservation_path(self, reservation_id: str) -> Path:
        return self.reservation_root / self._reservation_digest(reservation_id)

    def _events_path(self, reservation_id: str) -> Path:
        return self._reservation_path(reservation_id) / "events"

    def _staging_path(self, reservation_id: str) -> Path:
        return self._reservation_path(reservation_id) / "staging"

    def _results_path(self, reservation_id: str) -> Path:
        return self._reservation_path(reservation_id) / "results"

    def _result_path(
        self,
        reservation_id: str,
        result_blob: SourceReplayResultBlob,
    ) -> Path:
        digest = result_blob.digest.removeprefix("sha256:")
        return self._results_path(reservation_id) / f"{digest}.json"

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
            or metadata.st_uid != os.geteuid()
        ):
            raise ExpertSourceReplayExecutionStoreError(
                f"{name} must be a private real directory"
            )

    @staticmethod
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
                raise ExpertSourceReplayExecutionStoreError(
                    f"{name} must be a private independent regular file"
                )
            payload = handle.read(maximum_size_bytes + 1)
        if len(payload) > maximum_size_bytes:
            raise ExpertSourceReplayExecutionStoreError(
                f"{name} exceeds its configured bound"
            )
        return payload

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
            or metadata.st_uid != os.geteuid()
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
