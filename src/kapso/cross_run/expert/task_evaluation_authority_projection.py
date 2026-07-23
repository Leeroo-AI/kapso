"""Pure fresh-authority projections for one task-evaluation spawn."""

from __future__ import annotations

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationAuthorityError,
    TaskEvaluationCurrentReleaseObservation,
    TaskEvaluationSpawnAuthorityFence,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationCase,
    TaskEvaluationExpertLeg,
    TaskEvaluationInvocationAllocation,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)


def task_evaluation_adapter_trust_observations(
    prepared_request: PreparedTaskEvaluationRequest,
) -> tuple[TaskAdapterTrustObservation, ...]:
    prepared = _reconstruct_prepared_request(prepared_request)
    return tuple(
        sorted(
            (
                TaskAdapterTrustObservation.mint(
                    task_adapter_manifest_id=(
                        adapter.manifest.task_adapter_manifest_id
                    ),
                    verification_receipt_id=(
                        adapter.verification_receipt.verification_receipt_id
                    ),
                    verifier_id=adapter.verification_receipt.verifier_id,
                    verifier_version=adapter.verification_receipt.verifier_version,
                    dependency_ids=adapter.dependency_ids,
                )
                for adapter in prepared.adapters
            ),
            key=lambda observation: observation.observation_id,
        )
    )


def task_evaluation_allocation_case_leg(
    cases: tuple[TaskEvaluationCase, ...],
    invocation_allocation: TaskEvaluationInvocationAllocation,
) -> TaskEvaluationExpertLeg:
    if (
        type(cases) is not tuple
        or not cases
        or any(type(case) is not TaskEvaluationCase for case in cases)
        or type(invocation_allocation) is not TaskEvaluationInvocationAllocation
    ):
        raise TaskEvaluationAuthorityError(
            "task evaluation allocation lookup requires exact case authority"
        )
    matching_legs = tuple(
        leg
        for case in cases
        if case.evaluation_case_id == invocation_allocation.evaluation_case_id
        for leg in case.legs
        if leg.leg_id == invocation_allocation.evaluation_leg_id
    )
    if len(matching_legs) != 1:
        raise TaskEvaluationAuthorityError(
            "task evaluation spawn allocation names no exact reserved case leg"
        )
    return matching_legs[0]


def task_evaluation_spawn_security_subject_ids(
    *,
    prepared_request: PreparedTaskEvaluationRequest,
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    invocation_allocation: TaskEvaluationInvocationAllocation,
    stable_current_release_observation: TaskEvaluationCurrentReleaseObservation,
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...],
) -> tuple[str, ...]:
    prepared, snapshot = _require_spawn_inputs(
        prepared_request=prepared_request,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=invocation_allocation,
        stable_current_release_observation=stable_current_release_observation,
        task_adapter_trust_observations=task_adapter_trust_observations,
    )
    reservation = snapshot.reservation
    request = snapshot.request
    candidate = prepared.candidate.manifest
    current = stable_current_release_observation
    subject_ids = {
        reservation.reservation_id,
        *reservation.exact_dependency_ids,
        request.request_id,
        *request.exact_dependency_ids,
        invocation_allocation.evaluation_case_id,
        invocation_allocation.evaluation_leg_id,
        current.observation_id,
        *current.validation_closure_ids,
        *candidate.source_dependency_ids,
        *candidate.ancestor_candidate_ids,
        candidate.sanitation_report_id,
    }
    if current.publication_id is not None:
        subject_ids.add(current.publication_id)
    if prepared.source_base is not None:
        subject_ids.update(prepared.source_base.release_manifest.consumed_dependency_ids)
    for observation in task_adapter_trust_observations:
        subject_ids.update(
            {
                observation.observation_id,
                observation.task_adapter_manifest_id,
                observation.verification_receipt_id,
                observation.verifier_authority_subject_id,
                *observation.dependency_ids,
            }
        )
    ordered = tuple(sorted(subject_ids))
    if not ordered:
        raise TaskEvaluationAuthorityError(
            "task evaluation spawn security subjects are empty"
        )
    for subject_id in ordered:
        require_content_id(subject_id, "task evaluation spawn security subject")
    return ordered


def build_task_evaluation_spawn_authority_fence(
    *,
    prepared_request: PreparedTaskEvaluationRequest,
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    invocation_allocation: TaskEvaluationInvocationAllocation,
    stable_current_release_observation: TaskEvaluationCurrentReleaseObservation,
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...],
    security_denylist_observation: SecurityDenylistObservation,
) -> TaskEvaluationSpawnAuthorityFence:
    expected_subjects = task_evaluation_spawn_security_subject_ids(
        prepared_request=prepared_request,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=invocation_allocation,
        stable_current_release_observation=stable_current_release_observation,
        task_adapter_trust_observations=task_adapter_trust_observations,
    )
    request = reservation_snapshot.request
    denylist = security_denylist_observation
    if (
        type(denylist) is not SecurityDenylistObservation
        or denylist.scope_id != request.scope_id
        or denylist.scope_contract_id != request.scope_contract_id
        or denylist.checked_subject_ids != expected_subjects
        or denylist.matched_revocations
    ):
        raise TaskEvaluationAuthorityError(
            "task evaluation spawn denylist differs from exact security authority"
        )
    return TaskEvaluationSpawnAuthorityFence.mint(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        request_id=request.request_id,
        invocation_allocation=invocation_allocation,
        stable_current_release_observation=stable_current_release_observation,
        task_adapter_trust_observations=task_adapter_trust_observations,
        security_denylist_observation=denylist,
    )


def _require_spawn_inputs(
    *,
    prepared_request: PreparedTaskEvaluationRequest,
    reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
    invocation_allocation: TaskEvaluationInvocationAllocation,
    stable_current_release_observation: TaskEvaluationCurrentReleaseObservation,
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...],
) -> tuple[
    PreparedTaskEvaluationRequest,
    ExpertTaskEvaluationReservationSnapshot,
]:
    prepared = _reconstruct_prepared_request(prepared_request)
    if (
        type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot
        or reservation_snapshot.request != prepared.plan_join.request
        or type(invocation_allocation) is not TaskEvaluationInvocationAllocation
        or invocation_allocation.reservation_id
        != reservation_snapshot.reservation.reservation_id
        or type(stable_current_release_observation)
        is not TaskEvaluationCurrentReleaseObservation
        or stable_current_release_observation.scope_id
        != reservation_snapshot.request.scope_id
        or stable_current_release_observation.release_id
        != reservation_snapshot.reservation.observed_current_release_id
        or type(task_adapter_trust_observations) is not tuple
        or task_adapter_trust_observations
        != task_evaluation_adapter_trust_observations(prepared)
    ):
        raise TaskEvaluationAuthorityError(
            "task evaluation spawn inputs differ from reserved prepared authority"
        )
    task_evaluation_allocation_case_leg(
        reservation_snapshot.request.cases,
        invocation_allocation,
    )
    return prepared, reservation_snapshot


def _reconstruct_prepared_request(
    prepared_request: PreparedTaskEvaluationRequest,
) -> PreparedTaskEvaluationRequest:
    if type(prepared_request) is not PreparedTaskEvaluationRequest:
        raise TaskEvaluationAuthorityError(
            "task evaluation spawn requires exact prepared authority"
        )
    return PreparedTaskEvaluationRequest(
        plan_join=prepared_request.plan_join,
        stored_candidate=prepared_request.stored_candidate,
        candidate=prepared_request.candidate,
        source_base=prepared_request.source_base,
        current_release_observation=prepared_request.current_release_observation,
        cases=prepared_request.cases,
    )
