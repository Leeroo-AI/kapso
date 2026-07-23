"""Create-only local execution journal for adapter-owned task evaluation."""

from __future__ import annotations

import os
import secrets
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalFilesystem,
    ExecutionJournalLock,
    ExecutionJournalResultBlob,
    ExecutionJournalStoreError,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationSpawnAuthorityFence,
)
from kapso.cross_run.expert.task_evaluation_authority_projection import (
    build_task_evaluation_spawn_authority_fence,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationInvocationAllocation,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    ResolvedTaskEvaluationCase,
    TaskEvaluationExecutionProviderRegistry,
    TaskEvaluationProviderCompletion,
    TaskEvaluationProviderExecutionHandle,
    task_evaluation_provider_execution_handle,
)
from kapso.cross_run.expert.task_evaluation_execution_journal import (
    TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION,
    TaskEvaluationExecutionJournalEvent,
    TaskEvaluationExecutionJournalEventKind,
    TaskEvaluationExecutionPrefixState,
    TaskEvaluationProcessObservation,
    task_evaluation_execution_schedule,
    validate_task_evaluation_execution_prefix,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_protocol import (
    build_task_evaluation_evaluator_request,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TaskEvaluatorResult,
    parse_task_evaluator_result,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    bounded_process_stream_observations_match_outcome,
    canonicalize_bounded_process_stream_observations,
)
from kapso.cross_run.settings import ExpertValidationPolicySettings

_TASK_EVALUATION_EXECUTION_DIRECTORY_NAME = "task-evaluation-executions"
_ALLOCATION_PERMIT_SEAL = object()
_SPAWN_AUTHORIZATION_SEAL = object()
_SPAWN_PERMIT_SEAL = object()
_PROVIDER_COMPLETION_SEAL = object()
_COMPLETED_EXECUTION_SEAL = object()


def _new_invocation_nonce() -> str:
    return secrets.token_hex(16)


class TaskEvaluationInvocationAllocationPermit:
    """Runtime-only ownership of the current durable invocation allocation."""

    __slots__ = ("_event_id", "_session", "allocation")

    def __init__(
        self,
        seal: object,
        session: _TaskEvaluationReservationSession,
        event: TaskEvaluationExecutionJournalEvent,
    ) -> None:
        if seal is not _ALLOCATION_PERMIT_SEAL:
            raise ExecutionJournalStoreError(
                "task evaluation allocation permit is not journal sealed"
            )
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_event_id", event.event_id)
        object.__setattr__(self, "allocation", event.invocation_allocation)

    def __setattr__(self, name, value) -> None:
        raise ExecutionJournalStoreError(
            "task evaluation allocation permit is immutable"
        )

    def require_current_allocation(
        self,
        execution_store: ExpertTaskEvaluationExecutionStore,
    ) -> TaskEvaluationInvocationAllocation:
        self._session._require_live_store_lock(execution_store)
        if (
            self._session._allocation_permit is not self
            or not self._session._events
            or self._session._events[-1].event_id != self._event_id
            or self._session._events[-1].invocation_allocation != self.allocation
            or self._session._events[-1].event_kind
            is not TaskEvaluationExecutionJournalEventKind.INVOCATION_ALLOCATED
        ):
            raise ExecutionJournalStoreError(
                "task evaluation allocation permit is not current"
            )
        return self.allocation

    def require_current_reservation_snapshot(
        self,
        execution_store: ExpertTaskEvaluationExecutionStore,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    ) -> ExpertTaskEvaluationReservationSnapshot:
        self.require_current_allocation(execution_store)
        if (
            type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot
            or reservation_snapshot != self._session.reservation_snapshot
        ):
            raise ExecutionJournalStoreError(
                "task evaluation reopened reservation differs from its live allocation"
            )
        return self._session.reservation_snapshot


class _TaskEvaluationSpawnAuthorizationPermit:
    """One live fresh-authority result awaiting durable spawn publication."""

    __slots__ = (
        "_allocation_permit",
        "_consumed",
        "_coordinator",
        "_prepared_request",
        "_store",
        "aggregate_tolerance",
        "execution_provider_key",
        "fence",
        "provider_execution_handle",
        "provider_registry",
        "resolved_case",
        "task_evaluator_request",
    )

    def __init__(
        self,
        seal: object,
        *,
        store: ExpertTaskEvaluationExecutionStore,
        coordinator: object,
        allocation_permit: TaskEvaluationInvocationAllocationPermit,
        prepared_request: PreparedTaskEvaluationRequest,
        provider_registry: TaskEvaluationExecutionProviderRegistry,
        resolved_case: ResolvedTaskEvaluationCase,
        fence: TaskEvaluationSpawnAuthorityFence,
        aggregate_tolerance: float,
    ) -> None:
        if seal is not _SPAWN_AUTHORIZATION_SEAL:
            raise ExecutionJournalStoreError(
                "task evaluation spawn authorization is not sealed"
            )
        allocation = allocation_permit.allocation
        provider_key = resolved_case.dispatch_key
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_allocation_permit", allocation_permit)
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "_prepared_request", prepared_request)
        object.__setattr__(self, "provider_registry", provider_registry)
        object.__setattr__(self, "resolved_case", resolved_case)
        object.__setattr__(self, "fence", fence)
        object.__setattr__(self, "execution_provider_key", provider_key)
        object.__setattr__(
            self,
            "provider_execution_handle",
            task_evaluation_provider_execution_handle(provider_key, allocation),
        )
        object.__setattr__(
            self,
            "task_evaluator_request",
            build_task_evaluation_evaluator_request(
                prepared_request,
                allocation_permit._session.reservation_snapshot,
                allocation,
            ),
        )
        object.__setattr__(self, "aggregate_tolerance", aggregate_tolerance)

    def __setattr__(self, name, value) -> None:
        raise ExecutionJournalStoreError(
            "task evaluation spawn authorization is immutable"
        )

    def _consume(self, session: _TaskEvaluationReservationSession) -> None:
        if self._consumed or session._store is not self._store:
            raise ExecutionJournalStoreError(
                "task evaluation spawn authorization is consumed or foreign"
            )
        allocation = self._allocation_permit.require_current_allocation(self._store)
        if (
            session._allocation_permit is not self._allocation_permit
            or session.prepared_request != self._prepared_request
        ):
            raise ExecutionJournalStoreError(
                "task evaluation spawn authorization differs from its live session"
            )
        self.provider_registry.require_exact_prepared_authority(self._prepared_request)
        resolved_case = self.provider_registry._resolved_case_for_allocation(
            prepared_request=self._prepared_request,
            reservation_snapshot=session.reservation_snapshot,
            invocation_allocation=allocation,
        )
        expected_fence = build_task_evaluation_spawn_authority_fence(
            prepared_request=self._prepared_request,
            reservation_snapshot=session.reservation_snapshot,
            invocation_allocation=allocation,
            stable_current_release_observation=(
                self.fence.stable_current_release_observation
            ),
            task_adapter_trust_observations=(
                self.fence.task_adapter_trust_observations
            ),
            security_denylist_observation=self.fence.security_denylist_observation,
        )
        expected_request = build_task_evaluation_evaluator_request(
            self._prepared_request,
            session.reservation_snapshot,
            allocation,
        )
        expected_handle = task_evaluation_provider_execution_handle(
            resolved_case.dispatch_key,
            allocation,
        )
        if (
            resolved_case is not self.resolved_case
            or self.fence != expected_fence
            or self.execution_provider_key != resolved_case.dispatch_key
            or self.provider_execution_handle != expected_handle
            or self.task_evaluator_request != expected_request
            or self.aggregate_tolerance
            != self._prepared_request.plan_join.settings.policy.task_evaluation_aggregate_tolerance
        ):
            raise ExecutionJournalStoreError(
                "task evaluation spawn authorization differs from exact authority"
            )
        resolved_case.require_current_provider_identity()
        object.__setattr__(self, "_consumed", True)


class TaskEvaluationSpawnPermit:
    """One-shot provider capability issued only after the spawn event is durable."""

    __slots__ = (
        "_execution_guard",
        "_execution_started",
        "_provider_registry",
        "_resolved_case",
        "_session",
        "_spawn_event_id",
    )

    def __init__(
        self,
        seal: object,
        session: _TaskEvaluationReservationSession,
        event: TaskEvaluationExecutionJournalEvent,
        provider_registry: TaskEvaluationExecutionProviderRegistry,
        resolved_case: ResolvedTaskEvaluationCase,
    ) -> None:
        if seal is not _SPAWN_PERMIT_SEAL:
            raise ExecutionJournalStoreError(
                "task evaluation spawn permit is not journal sealed"
            )
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_spawn_event_id", event.event_id)
        object.__setattr__(self, "_provider_registry", provider_registry)
        object.__setattr__(self, "_resolved_case", resolved_case)
        object.__setattr__(self, "_execution_started", False)
        object.__setattr__(self, "_execution_guard", Lock())

    def __setattr__(self, name, value) -> None:
        raise ExecutionJournalStoreError("task evaluation spawn permit is immutable")

    def execute(self) -> TaskEvaluationSealedLegCompletion:
        with self._execution_guard:
            self._require_current()
            if self._execution_started:
                raise ExecutionJournalStoreError(
                    "task evaluation spawn execution was already consumed"
                )
            object.__setattr__(self, "_execution_started", True)
        completion = self._provider_registry._execute_journal_leg(
            prepared_request=self._session.prepared_request,
            reservation_snapshot=self._session.reservation_snapshot,
            resolved_case=self._resolved_case,
            invocation_allocation=(self._session._events[-1].invocation_allocation),
        )
        return self._session._seal_provider_completion(self, completion)

    def _require_current(self) -> None:
        self._session._require_live_store_lock(self._session._store)
        if (
            self._session._spawn_permit is not self
            or not self._session._events
            or self._session._events[-1].event_id != self._spawn_event_id
            or self._session._events[-1].event_kind
            is not TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED
        ):
            raise ExecutionJournalStoreError(
                "task evaluation spawn permit is not current"
            )


class TaskEvaluationSealedLegCompletion:
    """Runtime-only provider completion owned by one live journal session."""

    __slots__ = ("_provider_completion", "_session", "_spawn_permit")

    def __init__(
        self,
        seal: object,
        session: _TaskEvaluationReservationSession,
        spawn_permit: TaskEvaluationSpawnPermit,
        provider_completion: TaskEvaluationProviderCompletion,
    ) -> None:
        if seal is not _PROVIDER_COMPLETION_SEAL:
            raise ExecutionJournalStoreError(
                "task evaluation provider completion is not journal sealed"
            )
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_spawn_permit", spawn_permit)
        object.__setattr__(self, "_provider_completion", provider_completion)

    def __setattr__(self, name, value) -> None:
        raise ExecutionJournalStoreError(
            "task evaluation sealed provider completion is immutable"
        )

    def _consume(
        self,
        session: _TaskEvaluationReservationSession,
    ) -> TaskEvaluationProviderCompletion:
        if (
            self._session is not session
            or session._pending_completion is not self
            or session._spawn_permit is not self._spawn_permit
        ):
            raise ExecutionJournalStoreError(
                "task evaluation provider completion is consumed or foreign"
            )
        self._spawn_permit._require_current()
        session._pending_completion = None
        return self._provider_completion


class CompletedTaskEvaluationExecution:
    """Detached process-local proof of one completely reduced durable journal."""

    __slots__ = (
        "_execution_store",
        "_owner_process_id",
        "events",
        "prepared_request",
        "reservation_snapshot",
    )

    def __init__(
        self,
        seal: object,
        execution_store: ExpertTaskEvaluationExecutionStore,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
        events: tuple[TaskEvaluationExecutionJournalEvent, ...],
    ) -> None:
        if seal is not _COMPLETED_EXECUTION_SEAL:
            raise ExecutionJournalStoreError(
                "completed task evaluation is not journal sealed"
            )
        object.__setattr__(self, "_execution_store", execution_store)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "reservation_snapshot", reservation_snapshot)
        object.__setattr__(self, "prepared_request", prepared_request)
        object.__setattr__(self, "events", events)

    def __setattr__(self, name, value) -> None:
        raise ExecutionJournalStoreError(
            "completed task evaluation execution is immutable"
        )

    def require_exact(
        self,
        execution_store: ExpertTaskEvaluationExecutionStore,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> tuple[TaskEvaluationExecutionJournalEvent, ...]:
        if (
            execution_store is not self._execution_store
            or os.getpid() != self._owner_process_id
            or reservation_snapshot != self.reservation_snapshot
            or prepared_request != self.prepared_request
        ):
            raise ExecutionJournalStoreError(
                "completed task evaluation differs from its journal authority"
            )
        return self.events


class _TaskEvaluationReservationSession:
    """One exclusively locked task-evaluation reservation prefix."""

    def __init__(
        self,
        store: ExpertTaskEvaluationExecutionStore,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
        prefix: TaskEvaluationExecutionPrefixState,
        execution_lock: ExecutionJournalLock,
        factory_authority: object,
    ) -> None:
        if factory_authority is not store._session_factory_authority:
            raise ExecutionJournalStoreError(
                "task evaluation session lacks canonical store authority"
            )
        self._store = store
        self.reservation_snapshot = reservation_snapshot
        self.prepared_request = prepared_request
        self._events = prefix.events
        self._schedule = prefix.schedule
        self._execution_lock = execution_lock
        self._owner_process_id = os.getpid()
        self._active = True
        self._append_poisoned = False
        self._allocation_permit = None
        self._spawn_permit = None
        self._pending_completion = None

    @property
    def events(self) -> tuple[TaskEvaluationExecutionJournalEvent, ...]:
        self._require_live_store_lock(self._store)
        return self._events

    def completed_execution(self) -> CompletedTaskEvaluationExecution:
        self._require_live_store_lock(self._store)
        durable_prefix = self._store._read_prefix(
            self.reservation_snapshot,
            self.prepared_request,
        )
        if (
            durable_prefix.events != self._events
            or durable_prefix.schedule != self._schedule
            or not durable_prefix.complete
            or not durable_prefix.events
            or durable_prefix.events[-1].event_kind
            is not TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED
        ):
            raise ExecutionJournalStoreError(
                "task evaluation execution journal is incomplete"
            )
        return CompletedTaskEvaluationExecution(
            _COMPLETED_EXECUTION_SEAL,
            self._store,
            self.reservation_snapshot,
            self.prepared_request,
            durable_prefix.events,
        )

    def cleanup_interrupted_spawn(
        self,
        provider_registry: TaskEvaluationExecutionProviderRegistry,
    ) -> TaskEvaluationProviderExecutionHandle:
        self._require_live_store_lock(self._store)
        if type(provider_registry) is not TaskEvaluationExecutionProviderRegistry:
            raise ExecutionJournalStoreError(
                "task evaluation cleanup requires its exact provider registry"
            )
        provider_registry.require_exact_prepared_authority(self.prepared_request)
        if (
            len(self._events) % 4 != 2
            or self._events[-1].event_kind
            is not TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED
            or self._spawn_permit is not None
        ):
            raise ExecutionJournalStoreError(
                "task evaluation cleanup requires a reopened interrupted spawn"
            )
        provider_handle = self._events[-1].provider_execution_handle
        if type(provider_handle) is not TaskEvaluationProviderExecutionHandle:
            raise ExecutionJournalStoreError(
                "task evaluation interrupted spawn has no exact provider handle"
            )
        provider_registry.cleanup_interrupted(provider_handle)
        return provider_handle

    def allocate_expected_leg(self) -> TaskEvaluationInvocationAllocationPermit:
        self._require_live_store_lock(self._store)
        phase = len(self._events) % 4
        if phase == 1:
            if self._allocation_permit is None:
                self._allocation_permit = TaskEvaluationInvocationAllocationPermit(
                    _ALLOCATION_PERMIT_SEAL,
                    self,
                    self._events[-1],
                )
            return self._allocation_permit
        if phase == 2:
            raise ExecutionJournalStoreError(
                "task evaluation spawn marker is permanently interrupted after reopen"
            )
        if phase == 3:
            raise ExecutionJournalStoreError(
                "task evaluation received result must be accepted before another leg"
            )
        schedule_position = len(self._events) // 4
        if schedule_position >= len(self._schedule):
            raise ExecutionJournalStoreError(
                "task evaluation execution schedule is complete"
            )
        evaluation_case_id, evaluation_leg_id = self._schedule[schedule_position]
        allocation = TaskEvaluationInvocationAllocation(
            reservation_id=self.reservation_snapshot.reservation.reservation_id,
            evaluation_case_id=evaluation_case_id,
            evaluation_leg_id=evaluation_leg_id,
            invocation_nonce=_new_invocation_nonce(),
        )
        event = TaskEvaluationExecutionJournalEvent.mint(
            schema_version=TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=len(self._events) + 1,
            predecessor_event_id=(
                None if not self._events else self._events[-1].event_id
            ),
            event_kind=(TaskEvaluationExecutionJournalEventKind.INVOCATION_ALLOCATED),
            request_id=self.reservation_snapshot.request.request_id,
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
        prefix = self._store._validate_prefix(
            self.reservation_snapshot,
            self.prepared_request,
            (*self._events, event),
        )
        self._poison_for_append()
        self._store._publish_event(
            self.reservation_snapshot.reservation.reservation_id,
            event,
        )
        self._events = prefix.events
        self._append_poisoned = False
        self._allocation_permit = TaskEvaluationInvocationAllocationPermit(
            _ALLOCATION_PERMIT_SEAL,
            self,
            event,
        )
        return self._allocation_permit

    def _commit_spawn(
        self,
        authorization: _TaskEvaluationSpawnAuthorizationPermit,
    ) -> TaskEvaluationSpawnPermit:
        self._require_live_store_lock(self._store)
        if type(authorization) is not _TaskEvaluationSpawnAuthorizationPermit:
            raise ExecutionJournalStoreError(
                "task evaluation spawn requires sealed fresh authorization"
            )
        if (
            len(self._events) % 4 != 1
            or self._events[-1].event_kind
            is not TaskEvaluationExecutionJournalEventKind.INVOCATION_ALLOCATED
        ):
            raise ExecutionJournalStoreError(
                "task evaluation spawn requires the current allocation tail"
            )
        authorization._consume(self)
        allocation = self._events[-1].invocation_allocation
        event = TaskEvaluationExecutionJournalEvent.mint(
            schema_version=TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=len(self._events) + 1,
            predecessor_event_id=self._events[-1].event_id,
            event_kind=TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED,
            request_id=self.reservation_snapshot.request.request_id,
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
        prefix = self._store._validate_prefix(
            self.reservation_snapshot,
            self.prepared_request,
            (*self._events, event),
        )
        spawn_permit = TaskEvaluationSpawnPermit(
            _SPAWN_PERMIT_SEAL,
            self,
            event,
            authorization.provider_registry,
            authorization.resolved_case,
        )
        self._poison_for_append()
        self._store._publish_event(
            self.reservation_snapshot.reservation.reservation_id,
            event,
        )
        self._events = prefix.events
        self._append_poisoned = False
        self._spawn_permit = spawn_permit
        return spawn_permit

    def _seal_provider_completion(
        self,
        spawn_permit: TaskEvaluationSpawnPermit,
        provider_completion: TaskEvaluationProviderCompletion,
    ) -> TaskEvaluationSealedLegCompletion:
        self._require_live_store_lock(self._store)
        spawn_permit._require_current()
        if (
            type(provider_completion) is not TaskEvaluationProviderCompletion
            or self._pending_completion is not None
            or not spawn_permit._execution_started
            or provider_completion.provider_handle_id
            != self._events[-1].provider_execution_handle.provider_handle_id
        ):
            raise ExecutionJournalStoreError(
                "task evaluation provider completion differs from its live spawn"
            )
        process_result = provider_completion.process_result
        result_payload = provider_completion.result_payload
        compute = spawn_permit._resolved_case.executable_case.compute_binding
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
                "task evaluation provider completion exceeds its exact compute authority"
            )
        sealed = TaskEvaluationSealedLegCompletion(
            _PROVIDER_COMPLETION_SEAL,
            self,
            spawn_permit,
            provider_completion,
        )
        self._pending_completion = sealed
        return sealed

    def record_result_received(
        self,
        completion: TaskEvaluationSealedLegCompletion,
    ) -> TaskEvaluationExecutionJournalEvent:
        self._require_live_store_lock(self._store)
        if type(completion) is not TaskEvaluationSealedLegCompletion:
            raise ExecutionJournalStoreError(
                "task evaluation result requires a journal-sealed completion"
            )
        provider_completion = completion._consume(self)
        process_result = provider_completion.process_result
        result_payload = provider_completion.result_payload
        compute = (
            completion._spawn_permit._resolved_case.executable_case.compute_binding
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
                self._store._reservation_digest(
                    self.reservation_snapshot.reservation.reservation_id
                ),
                result_payload,
            )
        )
        allocation = self._events[-1].invocation_allocation
        event = TaskEvaluationExecutionJournalEvent.mint(
            schema_version=TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=len(self._events) + 1,
            predecessor_event_id=self._events[-1].event_id,
            event_kind=TaskEvaluationExecutionJournalEventKind.RESULT_RECEIVED,
            request_id=self.reservation_snapshot.request.request_id,
            invocation_allocation=allocation,
            spawn_authority_fence=None,
            execution_provider_key=None,
            provider_execution_handle=None,
            task_evaluator_request=None,
            aggregate_tolerance=None,
            process_observation=TaskEvaluationProcessObservation(
                outcome=process_result.outcome,
                returncode=process_result.returncode,
                stdout_bytes_observed=stdout_bytes_observed,
                stderr_bytes_observed=stderr_bytes_observed,
                duration_seconds=process_result.duration_seconds,
            ),
            result_blob=result_blob,
            task_evaluator_result=None,
        )
        prefix = self._store._validate_prefix(
            self.reservation_snapshot,
            self.prepared_request,
            (*self._events, event),
        )
        self._store._publish_event(
            self.reservation_snapshot.reservation.reservation_id,
            event,
        )
        self._events = prefix.events
        self._append_poisoned = False
        return event

    def accept_received_result(self) -> TaskEvaluatorResult:
        self._require_live_store_lock(self._store)
        if (
            len(self._events) % 4 != 3
            or self._events[-1].event_kind
            is not TaskEvaluationExecutionJournalEventKind.RESULT_RECEIVED
        ):
            raise ExecutionJournalStoreError(
                "task evaluation acceptance requires a received-result tail"
            )
        received_event = self._events[-1]
        spawn_event = self._events[-2]
        if received_event.result_blob is None:
            raise ExecutionJournalStoreError(
                "task evaluation process produced no acceptable result"
            )
        payload = self._store._filesystem.read_result(
            self._store._reservation_digest(
                self.reservation_snapshot.reservation.reservation_id
            ),
            received_event.result_blob,
        )
        result = parse_task_evaluator_result(
            payload,
            spawn_event.task_evaluator_request,
            spawn_event.aggregate_tolerance,
        )
        allocation = received_event.invocation_allocation
        event = TaskEvaluationExecutionJournalEvent.mint(
            schema_version=TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION,
            event_number=len(self._events) + 1,
            predecessor_event_id=received_event.event_id,
            event_kind=TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED,
            request_id=self.reservation_snapshot.request.request_id,
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
        prefix = self._store._validate_prefix(
            self.reservation_snapshot,
            self.prepared_request,
            (*self._events, event),
        )
        self._poison_for_append()
        self._store._publish_event(
            self.reservation_snapshot.reservation.reservation_id,
            event,
        )
        self._events = prefix.events
        self._append_poisoned = False
        return result

    def _require_active(self) -> None:
        if not self._active:
            raise ExecutionJournalStoreError(
                "task evaluation reservation session is closed"
            )
        if self._append_poisoned:
            raise ExecutionJournalStoreError(
                "task evaluation reservation session must reopen after append"
            )

    def _poison_for_append(self) -> None:
        self._require_active()
        self._append_poisoned = True
        self._allocation_permit = None
        self._spawn_permit = None
        self._pending_completion = None

    def _require_live_store_lock(
        self,
        execution_store: ExpertTaskEvaluationExecutionStore,
    ) -> None:
        self._require_active()
        reservation_id = self.reservation_snapshot.reservation.reservation_id
        if (
            execution_store is not self._store
            or os.getpid() != self._owner_process_id
            or not isinstance(self._execution_lock, ExecutionJournalLock)
            or self._execution_lock.owner_process_id != os.getpid()
            or not self._execution_lock.acquired
            or self._execution_lock.handle is None
            or self._execution_lock.path
            != execution_store._filesystem.reservation_lock_path(
                execution_store._reservation_digest(reservation_id)
            )
            or execution_store._active_sessions.get(reservation_id) is not self
        ):
            raise ExecutionJournalStoreError(
                "task evaluation runtime authority lacks its creator process and canonical live store lock"
            )

    def _close(self) -> None:
        self._active = False
        self._append_poisoned = True
        self._allocation_permit = None
        self._spawn_permit = None
        self._pending_completion = None


class _TaskEvaluationReservationSessionContext:
    def __init__(
        self,
        store: ExpertTaskEvaluationExecutionStore,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> None:
        self.store = store
        self.reservation_snapshot = reservation_snapshot
        self.prepared_request = prepared_request
        self.stack = None
        self.session = None

    def __enter__(self) -> _TaskEvaluationReservationSession:
        if self.stack is not None or self.session is not None:
            raise ExecutionJournalStoreError(
                "task evaluation reservation context cannot be entered twice"
            )
        reservation_digest = self.store._reservation_digest(
            self.reservation_snapshot.reservation.reservation_id
        )
        self.store._filesystem.ensure_reservation_layout(reservation_digest)
        with ExitStack() as setup:
            execution_lock = setup.enter_context(
                self.store._filesystem.reservation_lock(reservation_digest)
            )
            self.store._filesystem.clean_staging(reservation_digest)
            prefix = self.store._read_prefix(
                self.reservation_snapshot,
                self.prepared_request,
            )
            self.session = _TaskEvaluationReservationSession(
                self.store,
                self.reservation_snapshot,
                self.prepared_request,
                prefix,
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


class ExpertTaskEvaluationExecutionStore:
    """Own private create-only execution chains for task-evaluation reservations."""

    def __init__(
        self,
        root: Path,
        trusted_root: Path,
        policy_settings: ExpertValidationPolicySettings,
    ) -> None:
        if type(policy_settings) is not ExpertValidationPolicySettings:
            raise ExecutionJournalStoreError(
                "task evaluation journal requires canonical policy settings"
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
                "task evaluation journal canonical root requires a path"
            )
        return validation_store_root / _TASK_EVALUATION_EXECUTION_DIRECTORY_NAME

    def reservation_session(
        self,
        *,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> _TaskEvaluationReservationSessionContext:
        prepared = self._require_prepared_authority(prepared_request)
        snapshot = self._require_reservation_authority(
            reservation_snapshot,
            prepared,
        )
        return _TaskEvaluationReservationSessionContext(self, snapshot, prepared)

    def existing_reservation_events(
        self,
        *,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> tuple[TaskEvaluationExecutionJournalEvent, ...] | None:
        """Read an existing journal without creating a reservation layout."""

        prepared = self._require_prepared_authority(prepared_request)
        snapshot = self._require_reservation_authority(
            reservation_snapshot,
            prepared,
        )
        reservation_digest = self._reservation_digest(
            snapshot.reservation.reservation_id
        )
        if not self._filesystem.has_complete_reservation_layout(reservation_digest):
            return None
        with self._filesystem.reservation_lock(reservation_digest):
            return self._read_prefix(snapshot, prepared).events

    def _require_prepared_authority(
        self,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> PreparedTaskEvaluationRequest:
        if type(prepared_request) is not PreparedTaskEvaluationRequest:
            raise ExecutionJournalStoreError(
                "task evaluation journal requires exact prepared byte authority"
            )
        prepared = PreparedTaskEvaluationRequest(
            plan_join=prepared_request.plan_join,
            stored_candidate=prepared_request.stored_candidate,
            candidate=prepared_request.candidate,
            source_base=prepared_request.source_base,
            current_release_observation=(prepared_request.current_release_observation),
            cases=prepared_request.cases,
        )
        policy = prepared.plan_join.settings.policy
        if (
            policy != self.policy_settings
            or prepared.plan_join.request.validation_policy_id
            != self.policy_settings.validation_policy().validation_policy_id
        ):
            raise ExecutionJournalStoreError(
                "task evaluation prepared authority uses another validation policy"
            )
        return prepared

    @staticmethod
    def _require_reservation_authority(
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> ExpertTaskEvaluationReservationSnapshot:
        if type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot:
            raise ExecutionJournalStoreError(
                "task evaluation journal requires exact reservation authority"
            )
        snapshot = ExpertTaskEvaluationReservationSnapshot(
            operation=reservation_snapshot.operation,
            reservation=reservation_snapshot.reservation,
            request=reservation_snapshot.request,
            current_release_observation=(
                reservation_snapshot.current_release_observation
            ),
            plan_reservation=reservation_snapshot.plan_reservation,
        )
        task_evaluation_execution_schedule(snapshot, prepared_request)
        return snapshot

    def _bind_spawn_authority(self, coordinator_type: type[object]) -> None:
        if (
            coordinator_type.__module__
            != "kapso.cross_run.expert.task_evaluation_authority"
            or coordinator_type.__qualname__
            != "TaskEvaluationFreshAuthorityCoordinator"
        ):
            raise ExecutionJournalStoreError(
                "task evaluation journal spawn authority type is invalid"
            )
        if (
            self._spawn_authority_type is not None
            and self._spawn_authority_type is not coordinator_type
        ):
            raise ExecutionJournalStoreError(
                "task evaluation journal spawn authority is already bound"
            )
        self._spawn_authority_type = coordinator_type

    def _seal_spawn_authorization(
        self,
        *,
        coordinator: object,
        allocation_permit: TaskEvaluationInvocationAllocationPermit,
        prepared_request: PreparedTaskEvaluationRequest,
        provider_registry: TaskEvaluationExecutionProviderRegistry,
        fence: TaskEvaluationSpawnAuthorityFence,
    ) -> _TaskEvaluationSpawnAuthorizationPermit:
        if type(coordinator) is not self._spawn_authority_type:
            raise ExecutionJournalStoreError(
                "task evaluation spawn authorization lacks its coordinator"
            )
        if (
            type(allocation_permit) is not TaskEvaluationInvocationAllocationPermit
            or type(provider_registry) is not TaskEvaluationExecutionProviderRegistry
        ):
            raise ExecutionJournalStoreError(
                "task evaluation spawn authorization lacks exact runtime authority"
            )
        allocation = allocation_permit.require_current_allocation(self)
        session = allocation_permit._session
        prepared = self._require_prepared_authority(prepared_request)
        if prepared != session.prepared_request:
            raise ExecutionJournalStoreError(
                "task evaluation spawn prepared authority differs from its session"
            )
        provider_registry.require_exact_prepared_authority(prepared)
        resolved_case = provider_registry._resolved_case_for_allocation(
            prepared_request=prepared,
            reservation_snapshot=session.reservation_snapshot,
            invocation_allocation=allocation,
        )
        if type(fence) is not TaskEvaluationSpawnAuthorityFence:
            raise ExecutionJournalStoreError("task evaluation spawn fence is not exact")
        expected_fence = build_task_evaluation_spawn_authority_fence(
            prepared_request=prepared,
            reservation_snapshot=session.reservation_snapshot,
            invocation_allocation=allocation,
            stable_current_release_observation=(
                fence.stable_current_release_observation
            ),
            task_adapter_trust_observations=fence.task_adapter_trust_observations,
            security_denylist_observation=fence.security_denylist_observation,
        )
        if fence != expected_fence:
            raise ExecutionJournalStoreError(
                "task evaluation spawn fence differs from exact authority"
            )
        return _TaskEvaluationSpawnAuthorizationPermit(
            _SPAWN_AUTHORIZATION_SEAL,
            store=self,
            coordinator=coordinator,
            allocation_permit=allocation_permit,
            prepared_request=prepared,
            provider_registry=provider_registry,
            resolved_case=resolved_case,
            fence=fence,
            aggregate_tolerance=(
                self.policy_settings.task_evaluation_aggregate_tolerance
            ),
        )

    def _commit_spawn_authorization(
        self,
        *,
        coordinator: object,
        authorization: _TaskEvaluationSpawnAuthorizationPermit,
    ) -> TaskEvaluationSpawnPermit:
        if (
            type(coordinator) is not self._spawn_authority_type
            or type(authorization) is not _TaskEvaluationSpawnAuthorizationPermit
            or authorization._store is not self
            or authorization._coordinator is not coordinator
        ):
            raise ExecutionJournalStoreError(
                "task evaluation spawn commit lacks fresh coordinator authority"
            )
        return authorization._allocation_permit._session._commit_spawn(authorization)

    def _register_active_session(
        self,
        session: _TaskEvaluationReservationSession,
    ) -> None:
        reservation_id = session.reservation_snapshot.reservation.reservation_id
        if (
            session._store is not self
            or not session._execution_lock.acquired
            or reservation_id in self._active_sessions
        ):
            raise ExecutionJournalStoreError(
                "task evaluation store cannot register the reservation session"
            )
        self._active_sessions[reservation_id] = session

    def _unregister_active_session(
        self,
        session: _TaskEvaluationReservationSession,
    ) -> None:
        reservation_id = session.reservation_snapshot.reservation.reservation_id
        if self._active_sessions.get(reservation_id) is not session:
            raise ExecutionJournalStoreError(
                "task evaluation reservation session registration changed"
            )
        del self._active_sessions[reservation_id]

    def _read_prefix(
        self,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> TaskEvaluationExecutionPrefixState:
        reservation_digest = self._reservation_digest(
            reservation_snapshot.reservation.reservation_id
        )
        schedule = task_evaluation_execution_schedule(
            reservation_snapshot,
            prepared_request,
        )
        parsed_events = []
        for numbered_payload in self._filesystem.read_numbered_event_payloads(
            reservation_digest,
            4 * len(schedule),
        ):
            payload = numbered_payload.payload
            event = TaskEvaluationExecutionJournalEvent.from_json_bytes(payload)
            if payload != event.to_json_bytes():
                raise ExecutionJournalStoreError(
                    "task evaluation journal event is not canonical"
                )
            if event.event_number != numbered_payload.event_number:
                raise ExecutionJournalStoreError(
                    "task evaluation event filename differs from its identity"
                )
            parsed_events.append(event)
        return self._validate_prefix(
            reservation_snapshot,
            prepared_request,
            tuple(parsed_events),
        )

    def _validate_prefix(
        self,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        prepared_request: PreparedTaskEvaluationRequest,
        events: tuple[TaskEvaluationExecutionJournalEvent, ...],
    ) -> TaskEvaluationExecutionPrefixState:
        reservation_digest = self._reservation_digest(
            reservation_snapshot.reservation.reservation_id
        )
        schedule = task_evaluation_execution_schedule(
            reservation_snapshot,
            prepared_request,
        )
        self._filesystem.validate_results(reservation_digest, len(schedule))
        result_blobs = tuple(
            sorted(
                {
                    event.result_blob
                    for event in events
                    if event.result_blob is not None
                },
                key=lambda blob: blob.digest,
            )
        )
        result_payloads = tuple(
            (
                result_blob,
                self._filesystem.read_result(reservation_digest, result_blob),
            )
            for result_blob in result_blobs
        )
        return validate_task_evaluation_execution_prefix(
            reservation_snapshot=reservation_snapshot,
            prepared_request=prepared_request,
            events=events,
            result_payloads=result_payloads,
        )

    def _publish_event(
        self,
        reservation_id: str,
        event: TaskEvaluationExecutionJournalEvent,
    ) -> None:
        self._filesystem.publish_numbered_event(
            self._reservation_digest(reservation_id),
            event.event_number,
            event.to_json_bytes(),
        )

    @staticmethod
    def _reservation_digest(reservation_id: str) -> str:
        require_content_id(reservation_id, "task evaluation reservation_id")
        namespace, digest = reservation_id.split(":sha256:", 1)
        if namespace != "task-evaluation-reservation":
            raise ExecutionJournalStoreError(
                "task evaluation journal reservation uses the wrong namespace"
            )
        return digest
