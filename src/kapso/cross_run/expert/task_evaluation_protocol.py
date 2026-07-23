"""Projection of one prepared matrix case into the blinded evaluator ABI."""

from __future__ import annotations

from kapso.cross_run.contracts import TaskAdapterReleaseMatrixCase
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationInvocationAllocation,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    MaterializedTaskEvaluationCase,
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TASK_EVALUATOR_PROTOCOL_VERSION,
    TaskEvaluatorProtocolError,
    TaskEvaluatorRequest,
    TaskEvaluatorStartingArtifactMount,
)


def build_task_evaluation_evaluator_request(
    prepared_request: PreparedTaskEvaluationRequest,
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    invocation_allocation: TaskEvaluationInvocationAllocation,
) -> TaskEvaluatorRequest:
    """Build one blinded request from an exact signed matrix case."""

    if type(prepared_request) is not PreparedTaskEvaluationRequest:
        raise TaskEvaluatorProtocolError(
            "task evaluator request requires one exact prepared matrix request"
        )
    if type(invocation_allocation) is not TaskEvaluationInvocationAllocation:
        raise TaskEvaluatorProtocolError(
            "task evaluator request requires one exact matrix allocation"
        )
    if type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot:
        raise TaskEvaluatorProtocolError(
            "task evaluator request requires one exact durable reservation"
        )
    prepared = PreparedTaskEvaluationRequest(
        plan_join=prepared_request.plan_join,
        stored_candidate=prepared_request.stored_candidate,
        candidate=prepared_request.candidate,
        source_base=prepared_request.source_base,
        current_release_observation=prepared_request.current_release_observation,
        cases=prepared_request.cases,
    )
    if (
        reservation_snapshot.request != prepared.plan_join.request
        or reservation_snapshot.plan_reservation != prepared.plan_join.plan_reservation
        or invocation_allocation.reservation_id
        != reservation_snapshot.reservation.reservation_id
    ):
        raise TaskEvaluatorProtocolError(
            "task evaluator allocation differs from its durable reservation"
        )
    materialized_case = _allocated_case(prepared, invocation_allocation)
    request_case = materialized_case.request_case
    adapter = materialized_case.adapter_runtime.manifest
    if adapter.task_evaluator.protocol_version != TASK_EVALUATOR_PROTOCOL_VERSION:
        raise TaskEvaluatorProtocolError(
            "task evaluator manifest protocol is unsupported"
        )
    signed_case = _signed_case(materialized_case)
    context = signed_case.task_context_binding
    consumed_dimension_ids = adapter.context_binding.consumed_dimension_ids
    return TaskEvaluatorRequest(
        protocol_version=TASK_EVALUATOR_PROTOCOL_VERSION,
        opaque_invocation_id=invocation_allocation.opaque_invocation_id,
        input_contract_fingerprint=context.input_contract_fingerprint,
        target_contract_fingerprint=context.target_contract_fingerprint,
        evaluation_fingerprints=signed_case.evaluation_fingerprints,
        context_dimensions={
            dimension_id: context.transfer_dimensions[dimension_id]
            for dimension_id in consumed_dimension_ids
        },
        starting_artifact_mounts=tuple(
            sorted(
                (
                    TaskEvaluatorStartingArtifactMount(
                        starting_artifact_ref=(artifact.artifact.starting_artifact_ref),
                        mount_path=artifact.artifact.mount_path,
                    )
                    for artifact in materialized_case.starting_artifacts
                ),
                key=lambda mount: mount.starting_artifact_ref,
            )
        ),
    )


def _allocated_case(
    prepared: PreparedTaskEvaluationRequest,
    allocation: TaskEvaluationInvocationAllocation,
) -> MaterializedTaskEvaluationCase:
    matches = tuple(
        materialized_case
        for materialized_case in prepared.cases
        if materialized_case.request_case.evaluation_case_id
        == allocation.evaluation_case_id
    )
    if len(matches) != 1 or allocation.evaluation_leg_id not in {
        leg.leg_id for leg in matches[0].request_case.legs
    }:
        raise TaskEvaluatorProtocolError(
            "task evaluator matrix allocation names another case or leg"
        )
    return matches[0]


def _signed_case(
    materialized_case: MaterializedTaskEvaluationCase,
) -> TaskAdapterReleaseMatrixCase:
    request_case = materialized_case.request_case
    matches = tuple(
        signed_case
        for signed_case in materialized_case.adapter.manifest.release_matrix_cases
        if signed_case.release_matrix_case_id == request_case.release_matrix_case_id
    )
    if len(matches) != 1:
        raise TaskEvaluatorProtocolError(
            "task evaluator matrix case lacks unique signed authority"
        )
    signed_case = matches[0]
    if (
        signed_case.evaluation_fingerprint_ids
        != request_case.evaluation_fingerprint_ids
        or signed_case.task_context_binding.task_context_binding_id
        != request_case.task_context_binding_id
        or signed_case.starting_artifact_ids != request_case.starting_artifact_ids
    ):
        raise TaskEvaluatorProtocolError(
            "task evaluator matrix case differs from its signed authority"
        )
    return signed_case
