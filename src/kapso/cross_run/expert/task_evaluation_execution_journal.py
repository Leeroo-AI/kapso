"""Pure contracts and prefix reduction for task-evaluation execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, tree_or_blob_digest
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationSpawnAuthorityFence,
)
from kapso.cross_run.expert.task_evaluation_authority_projection import (
    build_task_evaluation_spawn_authority_fence,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationComputeBinding,
    TaskEvaluationInvocationAllocation,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    ExecutableTaskEvaluationCase,
    TaskEvaluationExecutionProviderKey,
    TaskEvaluationProviderExecutionHandle,
    project_prepared_task_evaluation_cases,
    task_evaluation_provider_execution_handle,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_protocol import (
    build_task_evaluation_evaluator_request,
)
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalResultBlob,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TaskEvaluatorRequest,
    TaskEvaluatorResult,
    parse_task_evaluator_result,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    bounded_process_stream_observations_are_canonical,
)

TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION = (
    "kapso.task_evaluation_execution_journal.v1"
)


class TaskEvaluationExecutionJournalError(ValueError):
    """A task-evaluation execution prefix is invalid or inconsistent."""


class TaskEvaluationExecutionJournalEventKind(str, Enum):
    INVOCATION_ALLOCATED = "invocation_allocated"
    SPAWN_COMMITTED = "spawn_committed"
    RESULT_RECEIVED = "result_received"
    RESULT_ACCEPTED = "result_accepted"


@dataclass(frozen=True)
class TaskEvaluationProcessObservation(StrictContract):
    outcome: BoundedProcessOutcome
    returncode: int
    stdout_bytes_observed: int
    stderr_bytes_observed: int
    duration_seconds: float

    def _validate(self) -> None:
        if type(self.returncode) is not int:
            raise TaskEvaluationExecutionJournalError(
                "task evaluation process returncode must be an integer"
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
            raise TaskEvaluationExecutionJournalError(
                "task evaluation process observation is invalid"
            )


@dataclass(frozen=True)
class TaskEvaluationExecutionJournalEvent(StrictContract):
    event_id: str
    schema_version: str
    event_number: int
    predecessor_event_id: str | None
    event_kind: TaskEvaluationExecutionJournalEventKind
    request_id: str
    invocation_allocation: TaskEvaluationInvocationAllocation
    spawn_authority_fence: TaskEvaluationSpawnAuthorityFence | None
    execution_provider_key: TaskEvaluationExecutionProviderKey | None
    provider_execution_handle: TaskEvaluationProviderExecutionHandle | None
    task_evaluator_request: TaskEvaluatorRequest | None
    aggregate_tolerance: float | None
    process_observation: TaskEvaluationProcessObservation | None
    result_blob: ExecutionJournalResultBlob | None
    task_evaluator_result: TaskEvaluatorResult | None

    CONTENT_NAMESPACE: ClassVar[str] = "task-evaluation-execution-journal-event"
    IDENTITY_FIELD: ClassVar[str] = "event_id"

    def _validate(self) -> None:
        if self.schema_version != TASK_EVALUATION_EXECUTION_JOURNAL_SCHEMA_VERSION:
            raise TaskEvaluationExecutionJournalError(
                "task evaluation execution journal schema is unsupported"
            )
        if type(self.event_number) is not int or self.event_number <= 0:
            raise TaskEvaluationExecutionJournalError(
                "task evaluation execution event number must be positive"
            )
        if (self.predecessor_event_id is None) != (self.event_number == 1):
            raise TaskEvaluationExecutionJournalError(
                "only the first task evaluation event may omit its predecessor"
            )
        if self.predecessor_event_id is not None:
            require_content_id(
                self.predecessor_event_id,
                "task evaluation execution predecessor",
            )
            if self.predecessor_event_id.split(":sha256:", 1)[0] != (
                self.CONTENT_NAMESPACE
            ):
                raise TaskEvaluationExecutionJournalError(
                    "task evaluation execution predecessor uses the wrong namespace"
                )
        require_content_id(self.request_id, "task evaluation execution request")
        if self.request_id.split(":sha256:", 1)[0] != "task-evaluation-request":
            raise TaskEvaluationExecutionJournalError(
                "task evaluation execution request uses the wrong namespace"
            )
        if type(self.invocation_allocation) is not TaskEvaluationInvocationAllocation:
            raise TaskEvaluationExecutionJournalError(
                "task evaluation execution event allocation is invalid"
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
            type(self.spawn_authority_fence) is TaskEvaluationSpawnAuthorityFence
            and type(self.execution_provider_key) is TaskEvaluationExecutionProviderKey
            and type(self.provider_execution_handle)
            is TaskEvaluationProviderExecutionHandle
            and type(self.task_evaluator_request) is TaskEvaluatorRequest
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
            and type(self.process_observation) is TaskEvaluationProcessObservation
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
            and type(self.task_evaluator_result) is TaskEvaluatorResult
        )
        if type(self.event_kind) is not TaskEvaluationExecutionJournalEventKind:
            raise TaskEvaluationExecutionJournalError(
                "task evaluation execution event kind is invalid"
            )
        expected_shape = {
            TaskEvaluationExecutionJournalEventKind.INVOCATION_ALLOCATED: (
                allocated_shape
            ),
            TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED: spawn_shape,
            TaskEvaluationExecutionJournalEventKind.RESULT_RECEIVED: received_shape,
            TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED: accepted_shape,
        }[self.event_kind]
        if not expected_shape:
            raise TaskEvaluationExecutionJournalError(
                "task evaluation execution payload differs from its event kind"
            )
        allocation = self.invocation_allocation
        if self.aggregate_tolerance is not None and (
            type(self.aggregate_tolerance) is not float
            or not math.isfinite(self.aggregate_tolerance)
            or self.aggregate_tolerance < 0.0
        ):
            raise TaskEvaluationExecutionJournalError(
                "task evaluation spawn aggregate tolerance is invalid"
            )
        if self.spawn_authority_fence is not None and (
            self.spawn_authority_fence.request_id != self.request_id
            or self.spawn_authority_fence.reservation_id != allocation.reservation_id
            or self.spawn_authority_fence.invocation_allocation != allocation
            or self.provider_execution_handle.dispatch_key
            != self.execution_provider_key
            or self.provider_execution_handle.invocation_allocation != allocation
            or self.task_evaluator_request.opaque_invocation_id
            != allocation.opaque_invocation_id
        ):
            raise TaskEvaluationExecutionJournalError(
                "task evaluation spawn payload differs from its allocation"
            )
        if (
            self.process_observation is not None
            and (
                self.process_observation.outcome is not BoundedProcessOutcome.COMPLETED
                or self.process_observation.returncode != 0
            )
            and self.result_blob is not None
        ):
            raise TaskEvaluationExecutionJournalError(
                "failed task evaluation process cannot publish a result blob"
            )
        if (
            self.task_evaluator_result is not None
            and self.task_evaluator_result.opaque_invocation_id
            != allocation.opaque_invocation_id
        ):
            raise TaskEvaluationExecutionJournalError(
                "accepted task evaluation result uses another invocation"
            )


@dataclass(frozen=True)
class TaskEvaluationExecutionPrefixState:
    """Non-authoritative state derived by pure validation of event data."""

    events: tuple[TaskEvaluationExecutionJournalEvent, ...]
    schedule: tuple[tuple[str, str], ...]

    @property
    def complete(self) -> bool:
        return len(self.events) == 4 * len(self.schedule)


def task_evaluation_execution_schedule(
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    prepared_request: PreparedTaskEvaluationRequest,
) -> tuple[tuple[str, str], ...]:
    """Return the sole case/leg order authorized by the persisted request."""

    _prepared, snapshot = _require_execution_authority(
        reservation_snapshot,
        prepared_request,
    )
    return _task_evaluation_execution_schedule(snapshot)


def _task_evaluation_execution_schedule(
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
) -> tuple[tuple[str, str], ...]:
    schedule = []
    for case in reservation_snapshot.request.cases:
        legs_by_kind = {leg.kind: leg for leg in case.legs}
        if len(legs_by_kind) != len(case.legs):
            raise TaskEvaluationExecutionJournalError(
                "task evaluation case contains duplicate semantic legs"
            )
        for leg_kind in case.compute_binding.leg_order:
            leg = legs_by_kind.get(leg_kind)
            if leg is None:
                raise TaskEvaluationExecutionJournalError(
                    "task evaluation compute schedule names an absent leg"
                )
            schedule.append((case.evaluation_case_id, leg.leg_id))
    return tuple(schedule)


def validate_task_evaluation_execution_prefix(
    *,
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    prepared_request: PreparedTaskEvaluationRequest,
    events: tuple[TaskEvaluationExecutionJournalEvent, ...],
    result_payloads: tuple[tuple[ExecutionJournalResultBlob, bytes], ...],
) -> TaskEvaluationExecutionPrefixState:
    """Validate an exact journal prefix without external or provider work."""

    prepared, snapshot = _require_execution_authority(
        reservation_snapshot,
        prepared_request,
    )
    schedule = _task_evaluation_execution_schedule(snapshot)
    if (
        type(events) is not tuple
        or any(
            type(event) is not TaskEvaluationExecutionJournalEvent for event in events
        )
        or len(events) > 4 * len(schedule)
    ):
        raise TaskEvaluationExecutionJournalError(
            "task evaluation journal contains an unsupported event suffix"
        )
    policy = prepared.plan_join.settings.policy
    if any(
        len(event.to_json_bytes()) > policy.task_evaluation_journal_event_byte_limit
        for event in events
    ):
        raise TaskEvaluationExecutionJournalError(
            "task evaluation journal event exceeds its configured bound"
        )
    payloads_by_blob = _validate_result_payloads(
        events,
        result_payloads,
        policy.task_evaluation_result_byte_limit,
    )
    phases = (
        TaskEvaluationExecutionJournalEventKind.INVOCATION_ALLOCATED,
        TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED,
        TaskEvaluationExecutionJournalEventKind.RESULT_RECEIVED,
        TaskEvaluationExecutionJournalEventKind.RESULT_ACCEPTED,
    )
    executable_cases = {
        case.evaluation_case_id: case
        for case in project_prepared_task_evaluation_cases(prepared)
    }
    request_cases = {case.evaluation_case_id: case for case in snapshot.request.cases}
    previous_event_id = None
    seen_nonces = set()
    seen_invocation_ids = set()
    for position, event in enumerate(events, start=1):
        schedule_position = (position - 1) // 4
        phase_position = (position - 1) % 4
        expected_case_id, expected_leg_id = schedule[schedule_position]
        allocation = event.invocation_allocation
        allocation_event = events[schedule_position * 4]
        if (
            event.event_number != position
            or event.predecessor_event_id != previous_event_id
            or event.event_kind is not phases[phase_position]
            or event.request_id != snapshot.request.request_id
            or allocation.reservation_id != snapshot.reservation.reservation_id
            or allocation.evaluation_case_id != expected_case_id
            or allocation.evaluation_leg_id != expected_leg_id
            or allocation != allocation_event.invocation_allocation
        ):
            raise TaskEvaluationExecutionJournalError(
                "task evaluation journal is not an exact authorized schedule prefix"
            )
        if phase_position == 0:
            if (
                allocation.invocation_nonce in seen_nonces
                or allocation.opaque_invocation_id in seen_invocation_ids
            ):
                raise TaskEvaluationExecutionJournalError(
                    "task evaluation journal reuses an invocation identity"
                )
            seen_nonces.add(allocation.invocation_nonce)
            seen_invocation_ids.add(allocation.opaque_invocation_id)
        elif phase_position == 1:
            _validate_spawn_event(
                event=event,
                prepared_request=prepared,
                reservation_snapshot=snapshot,
                executable_case=executable_cases[expected_case_id],
            )
        elif phase_position == 2:
            _validate_received_event(
                event,
                request_cases[expected_case_id].compute_binding,
                policy.task_evaluation_result_byte_limit,
            )
        else:
            spawn_event = events[schedule_position * 4 + 1]
            received_event = events[schedule_position * 4 + 2]
            if received_event.result_blob is None:
                raise TaskEvaluationExecutionJournalError(
                    "task evaluation journal accepted a missing result blob"
                )
            parsed = parse_task_evaluator_result(
                payloads_by_blob[received_event.result_blob],
                spawn_event.task_evaluator_request,
                spawn_event.aggregate_tolerance,
            )
            if parsed != event.task_evaluator_result:
                raise TaskEvaluationExecutionJournalError(
                    "accepted task evaluation result differs from its durable blob"
                )
        previous_event_id = event.event_id
    return TaskEvaluationExecutionPrefixState(events=events, schedule=schedule)


def _require_execution_authority(
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    prepared_request: PreparedTaskEvaluationRequest,
) -> tuple[PreparedTaskEvaluationRequest, ExpertTaskEvaluationReservationSnapshot]:
    if type(prepared_request) is not PreparedTaskEvaluationRequest:
        raise TaskEvaluationExecutionJournalError(
            "task evaluation journal requires exact prepared authority"
        )
    prepared = PreparedTaskEvaluationRequest(
        plan_join=prepared_request.plan_join,
        stored_candidate=prepared_request.stored_candidate,
        candidate=prepared_request.candidate,
        source_base=prepared_request.source_base,
        current_release_observation=prepared_request.current_release_observation,
        cases=prepared_request.cases,
    )
    if type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot:
        raise TaskEvaluationExecutionJournalError(
            "task evaluation journal requires exact reservation authority"
        )
    snapshot = ExpertTaskEvaluationReservationSnapshot(
        operation=reservation_snapshot.operation,
        reservation=reservation_snapshot.reservation,
        request=reservation_snapshot.request,
        current_release_observation=(reservation_snapshot.current_release_observation),
        plan_reservation=reservation_snapshot.plan_reservation,
    )
    if (
        snapshot.request != prepared.plan_join.request
        or snapshot.plan_reservation != prepared.plan_join.plan_reservation
    ):
        raise TaskEvaluationExecutionJournalError(
            "task evaluation journal reservation differs from prepared authority"
        )
    return prepared, snapshot


def _validate_result_payloads(
    events: tuple[TaskEvaluationExecutionJournalEvent, ...],
    result_payloads: tuple[tuple[ExecutionJournalResultBlob, bytes], ...],
    maximum_result_size_bytes: int,
) -> dict[ExecutionJournalResultBlob, bytes]:
    if type(result_payloads) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not ExecutionJournalResultBlob
        or not isinstance(item[1], bytes)
        for item in result_payloads
    ):
        raise TaskEvaluationExecutionJournalError(
            "task evaluation result payloads are not exact typed pairs"
        )
    descriptors = tuple(item[0] for item in result_payloads)
    if descriptors != tuple(sorted(set(descriptors), key=lambda blob: blob.digest)):
        raise TaskEvaluationExecutionJournalError(
            "task evaluation result payloads are not canonical"
        )
    referenced_descriptors = tuple(
        sorted(
            {event.result_blob for event in events if event.result_blob is not None},
            key=lambda blob: blob.digest,
        )
    )
    if descriptors != referenced_descriptors:
        raise TaskEvaluationExecutionJournalError(
            "task evaluation result payloads differ from referenced blobs"
        )
    payloads_by_blob = dict(result_payloads)
    for descriptor, payload in result_payloads:
        if (
            descriptor.size > maximum_result_size_bytes
            or len(payload) > maximum_result_size_bytes
        ):
            raise TaskEvaluationExecutionJournalError(
                "task evaluation result payload exceeds its configured bound"
            )
        if (
            len(payload) != descriptor.size
            or tree_or_blob_digest(payload) != descriptor.digest
        ):
            raise TaskEvaluationExecutionJournalError(
                "task evaluation result payload differs from its descriptor"
            )
    return payloads_by_blob


def _validate_spawn_event(
    *,
    event: TaskEvaluationExecutionJournalEvent,
    prepared_request: PreparedTaskEvaluationRequest,
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    executable_case: ExecutableTaskEvaluationCase,
) -> None:
    allocation = event.invocation_allocation
    expected_fence = build_task_evaluation_spawn_authority_fence(
        prepared_request=prepared_request,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
        stable_current_release_observation=(
            event.spawn_authority_fence.stable_current_release_observation
        ),
        task_adapter_trust_observations=(
            event.spawn_authority_fence.task_adapter_trust_observations
        ),
        security_denylist_observation=(
            event.spawn_authority_fence.security_denylist_observation
        ),
    )
    expected_provider_key = executable_case.provider_key
    expected_request = build_task_evaluation_evaluator_request(
        prepared_request,
        reservation_snapshot,
        allocation,
    )
    policy = prepared_request.plan_join.settings.policy
    if (
        event.spawn_authority_fence != expected_fence
        or event.execution_provider_key != expected_provider_key
        or event.provider_execution_handle
        != task_evaluation_provider_execution_handle(
            expected_provider_key,
            allocation,
        )
        or event.task_evaluator_request != expected_request
        or len(expected_request.to_json_bytes())
        > policy.task_evaluation_task_request_byte_limit
        or event.aggregate_tolerance != policy.task_evaluation_aggregate_tolerance
    ):
        raise TaskEvaluationExecutionJournalError(
            "task evaluation spawn differs from exact prepared authority"
        )


def _validate_received_event(
    event: TaskEvaluationExecutionJournalEvent,
    compute: TaskEvaluationComputeBinding,
    maximum_result_size_bytes: int,
) -> None:
    observation = event.process_observation
    if not bounded_process_stream_observations_are_canonical(
        outcome=observation.outcome,
        stdout_bytes_observed=observation.stdout_bytes_observed,
        stderr_bytes_observed=observation.stderr_bytes_observed,
        stdout_byte_limit=compute.stdout_byte_limit,
        stderr_byte_limit=compute.stderr_byte_limit,
    ) or (
        event.result_blob is not None
        and event.result_blob.size
        > min(compute.output_byte_limit, maximum_result_size_bytes)
    ):
        raise TaskEvaluationExecutionJournalError(
            "task evaluation result exceeds persisted compute bounds"
        )
