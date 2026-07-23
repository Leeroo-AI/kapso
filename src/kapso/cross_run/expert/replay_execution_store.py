"""Create-only local execution journal for expert source replay."""

from __future__ import annotations

import math
import os
import secrets
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import (
    ExpertSourceReplayExecutionLegKind,
    ExpertSourceReplayExecutionRequest,
    ExpertSourceReplayExecutionReservation,
    StrictContract,
)
from kapso.cross_run.expert.replay_protocol import build_task_evaluator_request
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalFilesystem,
    ExecutionJournalLock,
    ExecutionJournalResultBlob,
    ExecutionJournalStoreError,
)
from kapso.cross_run.expert.replay_protocol_contracts import (
    ExpertSourceReplayInvocationAllocation,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TaskEvaluatorRequest,
    TaskEvaluatorResult,
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
    ExpertSourceReplayExecutionProviderRegistry,
    ResolvedExpertSourceReplayExecutionCase,
    SourceReplayProviderExecutionHandle,
    expert_source_replay_execution_provider_key,
    source_replay_provider_execution_handle,
)
from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    bounded_process_stream_observations_are_canonical,
    bounded_process_stream_observations_match_outcome,
    canonicalize_bounded_process_stream_observations,
)
from kapso.cross_run.settings import ExpertValidationPolicySettings

_EXECUTION_JOURNAL_SCHEMA_VERSION = "kapso.source_replay_execution_journal.v2"
_EXECUTION_JOURNAL_DIRECTORY_NAME = "source-replay-executions"


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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
                "source replay process observation is invalid"
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
    invocation_allocation: ExpertSourceReplayInvocationAllocation
    spawn_authority_fence: SourceReplaySpawnAuthorityFence | None
    execution_provider_key: ExpertSourceReplayExecutionProviderKey | None
    provider_execution_handle: SourceReplayProviderExecutionHandle | None
    task_evaluator_request: TaskEvaluatorRequest | None
    aggregate_tolerance: float | None
    process_observation: SourceReplayProcessObservation | None
    result_blob: ExecutionJournalResultBlob | None
    task_evaluator_result: TaskEvaluatorResult | None

    CONTENT_NAMESPACE: ClassVar[str] = "source-replay-execution-journal-event"
    IDENTITY_FIELD: ClassVar[str] = "event_id"

    def _validate(self) -> None:
        if self.schema_version != _EXECUTION_JOURNAL_SCHEMA_VERSION:
            raise ExecutionJournalStoreError(
                "source replay execution journal schema is unsupported"
            )
        if type(self.event_number) is not int or self.event_number <= 0:
            raise ExecutionJournalStoreError(
                "source replay execution event number must be positive"
            )
        if (self.predecessor_event_id is None) != (self.event_number == 1):
            raise ExecutionJournalStoreError(
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
                raise ExecutionJournalStoreError(
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
                raise ExecutionJournalStoreError(
                    f"source replay execution event {name} uses the wrong namespace"
                )
        allocation = self.invocation_allocation
        if (
            allocation.reservation_id != self.reservation_id
            or allocation.execution_case_id != self.execution_case_id
            or allocation.execution_leg_id != self.execution_leg_id
        ):
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
                "source replay execution event payload differs from its kind"
            )
        if self.aggregate_tolerance is not None and (
            type(self.aggregate_tolerance) is not float
            or not math.isfinite(self.aggregate_tolerance)
            or self.aggregate_tolerance < 0.0
        ):
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
                "source replay failed process cannot publish a result blob"
            )


def _validate_reservation_request(
    reservation: ExpertSourceReplayExecutionReservation,
    request: ExpertSourceReplayExecutionRequest,
) -> None:
    if not isinstance(
        reservation, ExpertSourceReplayExecutionReservation
    ) or not isinstance(request, ExpertSourceReplayExecutionRequest):
        raise ExecutionJournalStoreError(
            "execution journal requires typed reservation and request authority"
        )
    if (
        reservation.execution_request_id != request.execution_request_id
        or reservation.validation_attempt_id != request.validation_attempt_id
        or reservation.authorization_state_id != request.authorization_state_id
        or reservation.candidate_id != request.candidate_id
        or reservation.candidate_tree_hash != request.candidate_tree_hash
        or reservation.expected_current_release_id != request.source_base_release_id
    ):
        raise ExecutionJournalStoreError(
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
            ExpertSourceReplayExecutionLegKind.SOURCE_BASE_CONTROL: case.control_leg,
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
_COMPLETED_EXECUTION_SEAL = object()


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
    ) -> ExpertSourceReplayInvocationAllocation:
        self._session._require_live_store_lock(execution_store)
        if (
            self._session._allocation_permit is not self
            or not self._session._events
            or self._session._events[-1].event_id != self._event_id
            or self._session._events[-1].invocation_allocation != self.allocation
        ):
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
        raise ExecutionJournalStoreError(
            "source replay execution capability is immutable"
        )

    def execute(self) -> SourceReplaySealedLegCompletion:
        with self._execution_guard:
            self._require_current()
            if self._execution_started:
                raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
                "source replay provider completion is not journal sealed"
            )
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_spawn_permit", spawn_permit)
        object.__setattr__(self, "_provider_completion", provider_completion)

    def __setattr__(self, name, value) -> None:
        raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
                "source replay provider completion is consumed or foreign"
            )
        self._spawn_permit._require_current()
        session._pending_completion = None
        return self._provider_completion


class CompletedExpertSourceReplayExecution:
    """Detached runtime proof of one fully verified durable execution journal."""

    __slots__ = (
        "_execution_store",
        "_owner_process_id",
        "reservation",
        "prepared_request",
        "events",
    )

    def __init__(
        self,
        seal: object,
        execution_store: ExpertSourceReplayExecutionStore,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
        events: tuple[SourceReplayExecutionJournalEvent, ...],
    ) -> None:
        if seal is not _COMPLETED_EXECUTION_SEAL:
            raise ExecutionJournalStoreError(
                "completed source replay execution is not journal sealed"
            )
        object.__setattr__(self, "_execution_store", execution_store)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "reservation", reservation)
        object.__setattr__(self, "prepared_request", prepared_request)
        object.__setattr__(self, "events", events)

    def __setattr__(self, name, value) -> None:
        raise ExecutionJournalStoreError(
            "completed source replay execution is immutable"
        )

    def require_exact(
        self,
        execution_store: ExpertSourceReplayExecutionStore,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> tuple[SourceReplayExecutionJournalEvent, ...]:
        if (
            execution_store is not self._execution_store
            or os.getpid() != self._owner_process_id
            or reservation != self.reservation
            or prepared_request != self.prepared_request
        ):
            raise ExecutionJournalStoreError(
                "completed source replay execution differs from its journal authority"
            )
        return self.events


class _SourceReplayReservationSession:
    """One exclusively locked reservation execution prefix."""

    def __init__(
        self,
        store: ExpertSourceReplayExecutionStore,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
        events: tuple[SourceReplayExecutionJournalEvent, ...],
        execution_lock: ExecutionJournalLock,
        factory_authority: object,
    ) -> None:
        if factory_authority is not store._session_factory_authority:
            raise ExecutionJournalStoreError(
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

    def completed_execution(self) -> CompletedExpertSourceReplayExecution:
        self._require_live_store_lock(self._store)
        schedule = source_replay_execution_schedule(
            self.reservation,
            self.request,
        )
        durable_events = self._store._read_events(
            self.reservation,
            self.prepared_request,
        )
        if (
            durable_events != self._events
            or len(durable_events) != 4 * len(schedule)
            or durable_events[-1].event_kind
            is not SourceReplayExecutionJournalEventKind.RESULT_ACCEPTED
        ):
            raise ExecutionJournalStoreError(
                "source replay execution journal is incomplete"
            )
        return CompletedExpertSourceReplayExecution(
            _COMPLETED_EXECUTION_SEAL,
            self._store,
            self.reservation,
            self.prepared_request,
            durable_events,
        )

    def cleanup_interrupted_spawn(
        self,
        provider_registry: ExpertSourceReplayExecutionProviderRegistry,
    ) -> SourceReplayProviderExecutionHandle:
        self._require_active()
        if not isinstance(
            provider_registry,
            ExpertSourceReplayExecutionProviderRegistry,
        ):
            raise ExecutionJournalStoreError(
                "source replay interrupted cleanup requires its provider registry"
            )
        if (
            len(self._events) % 4 != 2
            or self._events[-1].event_kind
            is not SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED
            or self._spawn_permit is not None
        ):
            raise ExecutionJournalStoreError(
                "source replay cleanup requires a reopened interrupted spawn"
            )
        provider_handle = self._events[-1].provider_execution_handle
        if type(provider_handle) is not SourceReplayProviderExecutionHandle:
            raise ExecutionJournalStoreError(
                "source replay interrupted spawn has no provider handle"
            )
        provider_registry.cleanup_interrupted(provider_handle)
        return provider_handle

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
            raise ExecutionJournalStoreError(
                "source replay spawn marker is permanently interrupted after reopen"
            )
        if phase == 3:
            raise ExecutionJournalStoreError(
                "source replay received result must be accepted before another leg"
            )
        schedule = source_replay_execution_schedule(
            self.reservation,
            self.request,
        )
        schedule_position = len(self._events) // 4
        if schedule_position >= len(schedule):
            raise ExecutionJournalStoreError(
                "source replay execution schedule is complete"
            )
        execution_case_id, execution_leg_id = schedule[schedule_position]
        allocation = ExpertSourceReplayInvocationAllocation(
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
            raise ExecutionJournalStoreError(
                "source replay spawn requires sealed fresh authorization"
            )
        if len(self._events) % 4 != 1 or self._events[-1].event_kind is not (
            SourceReplayExecutionJournalEventKind.INVOCATION_ALLOCATED
        ):
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            or not bounded_process_stream_observations_match_outcome(
                outcome=process_result.outcome,
                stdout_bytes_observed=process_result.stdout_bytes_observed,
                stderr_bytes_observed=process_result.stderr_bytes_observed,
                stdout_byte_limit=compute.stdout_byte_limit,
                stderr_byte_limit=compute.stderr_byte_limit,
            )
            or process_result.stdout_bytes_observed < len(process_result.stdout)
            or process_result.stderr_bytes_observed < len(process_result.stderr)
            or len(process_result.stdout) > compute.stdout_byte_limit
            or len(process_result.stderr) > compute.stderr_byte_limit
            or (
                result_payload is not None
                and len(result_payload)
                > min(
                    compute.output_byte_limit,
                    self._store._filesystem.maximum_result_size_bytes,
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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
                "source replay result requires a journal-sealed provider completion"
            )
        provider_completion = completion._consume(self)
        process_result = provider_completion.process_result
        result_payload = provider_completion.result_payload
        compute = (
            completion._spawn_permit._resolved_case.materialized_case.request_case.compute_binding
        )
        stdout_bytes_observed, stderr_bytes_observed = (
            canonicalize_bounded_process_stream_observations(
                outcome=process_result.outcome,
                stdout_bytes_observed=process_result.stdout_bytes_observed,
                stderr_bytes_observed=process_result.stderr_bytes_observed,
                stdout_byte_limit=compute.stdout_byte_limit,
                stderr_byte_limit=compute.stderr_byte_limit,
            )
        )
        self._poison_for_append()
        result_blob = (
            None
            if result_payload is None
            else self._store._filesystem.publish_result(
                self._store._reservation_digest(self.reservation.reservation_id),
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
                stdout_bytes_observed=stdout_bytes_observed,
                stderr_bytes_observed=stderr_bytes_observed,
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
            raise ExecutionJournalStoreError(
                "source replay result acceptance requires a received-result tail"
            )
        received_event = self._events[-1]
        spawn_event = self._events[-2]
        if received_event.result_blob is None:
            raise ExecutionJournalStoreError(
                "source replay process produced no acceptable result"
            )
        payload = self._store._filesystem.read_result(
            self._store._reservation_digest(self.reservation.reservation_id),
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
            raise ExecutionJournalStoreError(
                "source replay reservation session is closed"
            )
        if self._append_poisoned:
            raise ExecutionJournalStoreError(
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
            or not isinstance(self._execution_lock, ExecutionJournalLock)
            or self._execution_lock.owner_process_id != os.getpid()
            or not self._execution_lock.acquired
            or self._execution_lock.handle is None
            or self._execution_lock.path
            != execution_store._filesystem.reservation_lock_path(
                execution_store._reservation_digest(self.reservation.reservation_id)
            )
            or execution_store._active_sessions.get(self.reservation.reservation_id)
            is not self
        ):
            raise ExecutionJournalStoreError(
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
        reservation_digest = self.store._reservation_digest(
            self.reservation.reservation_id
        )
        self.store._filesystem.ensure_reservation_layout(reservation_digest)
        with ExitStack() as setup:
            execution_lock = setup.enter_context(
                self.store._filesystem.reservation_lock(reservation_digest)
            )
            self.store._filesystem.clean_staging(reservation_digest)
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
        if not isinstance(policy_settings, ExpertValidationPolicySettings):
            raise ExecutionJournalStoreError(
                "execution journal requires canonical validation policy settings"
            )
        self._filesystem = ExecutionJournalFilesystem(
            root,
            trusted_root,
            maximum_event_size_bytes=(
                policy_settings.task_evaluation_journal_event_byte_limit
            ),
            maximum_result_size_bytes=(
                policy_settings.task_evaluation_result_byte_limit
            ),
            maximum_staging_entry_count=(
                policy_settings.task_evaluation_staging_entry_limit
            ),
        )
        self.root = root
        self.trusted_root = trusted_root
        self.policy_settings = policy_settings
        self._session_factory_authority = object()
        self._spawn_authority_type = None
        self._active_sessions = {}

    @staticmethod
    def canonical_root(validation_store_root: Path) -> Path:
        if not isinstance(validation_store_root, Path):
            raise ExecutionJournalStoreError(
                "execution journal canonical root requires a path"
            )
        return validation_store_root / _EXECUTION_JOURNAL_DIRECTORY_NAME

    def reservation_session(
        self,
        *,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> _ReservationSessionContext:
        prepared = self._require_prepared_authority(prepared_request)
        _validate_reservation_request(reservation, prepared.request)
        return _ReservationSessionContext(self, reservation, prepared)

    def stage_run_lock(self, candidate_id: str) -> ExecutionJournalLock:
        """Serialize one local candidate stage without granting validation authority."""

        require_content_id(candidate_id, "source replay stage candidate_id")
        namespace, digest = candidate_id.split(":sha256:", 1)
        if namespace != "expert-candidate":
            raise ExecutionJournalStoreError(
                "source replay stage lock requires an expert candidate"
            )
        return self._filesystem.lock(f"candidate-stage-{digest}.lock")

    def existing_reservation_events(
        self,
        *,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> tuple[SourceReplayExecutionJournalEvent, ...] | None:
        """Read an existing journal without creating its reservation layout."""

        prepared = self._require_prepared_authority(prepared_request)
        _validate_reservation_request(reservation, prepared.request)
        reservation_digest = self._reservation_digest(reservation.reservation_id)
        if not self._filesystem.has_complete_reservation_layout(reservation_digest):
            return None
        with self._filesystem.reservation_lock(reservation_digest):
            return self._read_events(reservation, prepared)

    def _require_prepared_authority(
        self,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> PreparedExpertSourceReplayRequest:
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExecutionJournalStoreError(
                "execution journal requires its prepared byte authority"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            source_base=prepared_request.source_base,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        if (
            prepared.settings.policy != self.policy_settings
            or prepared.request.validation_policy_id
            != self.policy_settings.validation_policy().validation_policy_id
        ):
            raise ExecutionJournalStoreError(
                "execution journal prepared authority uses another validation policy"
            )
        return prepared

    def _bind_spawn_authority(self, coordinator_type: type[object]) -> None:
        if (
            coordinator_type.__module__ != "kapso.cross_run.expert.replay_authority"
            or coordinator_type.__qualname__
            != "ExpertSourceReplayFreshAuthorityCoordinator"
        ):
            raise ExecutionJournalStoreError(
                "execution journal spawn authority type is invalid"
            )
        if (
            self._spawn_authority_type is not None
            and self._spawn_authority_type is not coordinator_type
        ):
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            > self.policy_settings.task_evaluation_task_request_byte_limit
            or aggregate_tolerance
            != self.policy_settings.task_evaluation_aggregate_tolerance
            or not {
                reservation.reservation_id,
                *reservation.exact_dependency_ids,
                request.execution_request_id,
                *request.exact_dependency_ids,
            }.issubset(fence.security_subject_ids)
        ):
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
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
            raise ExecutionJournalStoreError(
                "execution store cannot register the reservation session"
            )
        self._active_sessions[reservation_id] = session

    def _unregister_active_session(
        self,
        session: _SourceReplayReservationSession,
    ) -> None:
        reservation_id = session.reservation.reservation_id
        if self._active_sessions.get(reservation_id) is not session:
            raise ExecutionJournalStoreError(
                "execution store reservation session registration changed"
            )
        del self._active_sessions[reservation_id]

    def _read_events(
        self,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> tuple[SourceReplayExecutionJournalEvent, ...]:
        request = prepared_request.request
        reservation_digest = self._reservation_digest(reservation.reservation_id)
        maximum_event_count = 4 * len(
            source_replay_execution_schedule(reservation, request)
        )
        parsed_entries = []
        for numbered_payload in self._filesystem.read_numbered_event_payloads(
            reservation_digest,
            maximum_event_count,
        ):
            payload = numbered_payload.payload
            event = SourceReplayExecutionJournalEvent.from_json_bytes(payload)
            if payload != event.to_json_bytes():
                raise ExecutionJournalStoreError(
                    "execution journal event is not canonical"
                )
            if event.event_number != numbered_payload.event_number:
                raise ExecutionJournalStoreError(
                    "execution journal event filename differs from its identity"
                )
            parsed_entries.append(event)
        events = tuple(parsed_entries)
        self._filesystem.validate_results(
            reservation_digest,
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
            raise ExecutionJournalStoreError(
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
                raise ExecutionJournalStoreError(
                    "execution journal is not an exact authorized schedule prefix"
                )
            if phase_position == 0:
                if (
                    allocation.invocation_nonce in seen_nonces
                    or allocation.opaque_invocation_id in seen_invocation_ids
                ):
                    raise ExecutionJournalStoreError(
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
                    != prepared_request.source_base.release_manifest.scope_id
                    or fence.expected_current_release_id != request.source_base_release_id
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
                    > self.policy_settings.task_evaluation_task_request_byte_limit
                    or event.aggregate_tolerance
                    != self.policy_settings.task_evaluation_aggregate_tolerance
                    or source_replay_spawn_security_subject_ids(
                        prepared_request,
                        reservation,
                        fence.current_release_observation,
                        fence.task_adapter_trust_observations,
                    )
                    != fence.security_subject_ids
                ):
                    raise ExecutionJournalStoreError(
                        "execution journal spawn fence differs from its reservation"
                    )
            elif phase_position == 2:
                request_case = next(
                    case
                    for case in request.cases
                    if case.execution_case_id == expected_case_id
                )
                if not bounded_process_stream_observations_are_canonical(
                    outcome=event.process_observation.outcome,
                    stdout_bytes_observed=(
                        event.process_observation.stdout_bytes_observed
                    ),
                    stderr_bytes_observed=(
                        event.process_observation.stderr_bytes_observed
                    ),
                    stdout_byte_limit=(request_case.compute_binding.stdout_byte_limit),
                    stderr_byte_limit=(request_case.compute_binding.stderr_byte_limit),
                ) or (
                    event.result_blob is not None
                    and event.result_blob.size
                    > request_case.compute_binding.output_byte_limit
                ):
                    raise ExecutionJournalStoreError(
                        "execution journal result exceeds its persisted compute bounds"
                    )
                if event.result_blob is not None:
                    self._filesystem.read_result(
                        self._reservation_digest(reservation.reservation_id),
                        event.result_blob,
                    )
            elif phase_position == 3:
                spawn_event = events[schedule_position * 4 + 1]
                received_event = events[schedule_position * 4 + 2]
                if received_event.result_blob is None:
                    raise ExecutionJournalStoreError(
                        "execution journal accepted a missing result blob"
                    )
                parsed = parse_task_evaluator_result(
                    self._filesystem.read_result(
                        self._reservation_digest(reservation.reservation_id),
                        received_event.result_blob,
                    ),
                    spawn_event.task_evaluator_request,
                    spawn_event.aggregate_tolerance,
                )
                if parsed != event.task_evaluator_result:
                    raise ExecutionJournalStoreError(
                        "execution journal accepted result differs from its blob"
                    )
            previous_event_id = event.event_id

    def _publish_event(
        self,
        reservation_id: str,
        event: SourceReplayExecutionJournalEvent,
    ) -> None:
        self._filesystem.publish_numbered_event(
            self._reservation_digest(reservation_id),
            event.event_number,
            event.to_json_bytes(),
        )

    @staticmethod
    def _reservation_digest(reservation_id: str) -> str:
        require_content_id(reservation_id, "source replay execution reservation_id")
        namespace, digest = reservation_id.split(":sha256:", 1)
        if namespace != "expert-source-replay-execution-reservation":
            raise ExecutionJournalStoreError(
                "execution journal reservation uses the wrong namespace"
            )
        return digest
