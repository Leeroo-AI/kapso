from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationContractError,
    TaskEvaluationInvocationAllocation,
    TaskEvaluationLegKind,
)
from kapso.cross_run.expert.task_evaluation_protocol import (
    build_task_evaluation_evaluator_request,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TASK_EVALUATOR_PROTOCOL_VERSION,
    TaskEvaluatorProtocolError,
    TaskEvaluatorRequest,
)
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
)
from test_expert_task_evaluation_preflight import (
    _CurrentAuthority,
    _coordinator,
    _current_observation,
)
from test_expert_task_evaluation_reservation import _parent_prepared


def _allocation(reservation_id, materialized_case, leg_id, nonce="1" * 32):
    return TaskEvaluationInvocationAllocation(
        reservation_id=reservation_id,
        evaluation_case_id=materialized_case.request_case.evaluation_case_id,
        evaluation_leg_id=leg_id,
        invocation_nonce=nonce,
    )


def test_matrix_case_projects_only_the_blinded_signed_task_authority(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    committed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    materialized_case = prepared.cases[0]
    request_case = materialized_case.request_case
    candidate_leg = next(
        leg for leg in request_case.legs if leg.kind is TaskEvaluationLegKind.CANDIDATE
    )
    allocation = _allocation(
        committed.reservation.reservation.reservation_id,
        materialized_case,
        candidate_leg.leg_id,
    )

    evaluator_request = build_task_evaluation_evaluator_request(
        prepared,
        committed.reservation,
        allocation,
    )
    signed_case = next(
        signed
        for signed in materialized_case.adapter.manifest.release_matrix_cases
        if signed.release_matrix_case_id == request_case.release_matrix_case_id
    )
    consumed_dimensions = (
        materialized_case.adapter.manifest.context_binding.consumed_dimension_ids
    )

    assert evaluator_request.protocol_version == TASK_EVALUATOR_PROTOCOL_VERSION
    assert evaluator_request.opaque_invocation_id == allocation.opaque_invocation_id
    assert evaluator_request.evaluation_fingerprints == (
        signed_case.evaluation_fingerprints
    )
    assert evaluator_request.context_dimensions == {
        dimension_id: signed_case.task_context_binding.transfer_dimensions[dimension_id]
        for dimension_id in consumed_dimensions
    }
    assert tuple(
        (mount.starting_artifact_ref, mount.mount_path)
        for mount in evaluator_request.starting_artifact_mounts
    ) == tuple(
        sorted(
            (
                artifact.artifact.starting_artifact_ref,
                artifact.artifact.mount_path,
            )
            for artifact in materialized_case.starting_artifacts
        )
    )
    assert TaskEvaluatorRequest.from_json_bytes(evaluator_request.to_json_bytes()) == (
        evaluator_request
    )


def test_bootstrap_matrix_projects_one_candidate_leg_without_parent(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan, _active_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    observation = _current_observation(prepared_plan)
    coordinator, _candidate_reader, source_base_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        source_base=None,
        current_authority=_CurrentAuthority((observation, observation)),
    )
    prepared = coordinator.build(plan_reservation)
    committed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    materialized_case = prepared.cases[0]
    assert tuple(leg.kind for leg in materialized_case.request_case.legs) == (
        TaskEvaluationLegKind.CANDIDATE,
    )
    allocation = _allocation(
        committed.reservation.reservation.reservation_id,
        materialized_case,
        materialized_case.request_case.legs[0].leg_id,
    )

    evaluator_request = build_task_evaluation_evaluator_request(
        prepared,
        committed.reservation,
        allocation,
    )

    assert evaluator_request.opaque_invocation_id == allocation.opaque_invocation_id
    assert source_base_provider.calls == []


def test_matrix_allocation_is_namespace_exact_and_content_bound(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    committed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    materialized_case = prepared.cases[0]
    leg = materialized_case.request_case.legs[0]
    allocation = _allocation(
        committed.reservation.reservation.reservation_id,
        materialized_case,
        leg.leg_id,
    )

    assert allocation.opaque_invocation_id.startswith("task_evaluation_invocation_")
    assert (
        len(allocation.opaque_invocation_id) == len("task_evaluation_invocation_") + 32
    )
    assert replace(allocation, invocation_nonce="2" * 32).opaque_invocation_id != (
        allocation.opaque_invocation_id
    )
    with pytest.raises(TaskEvaluationContractError, match="wrong namespace"):
        replace(
            allocation,
            reservation_id=content_id("foreign-reservation", {"foreign": True}),
        )
    with pytest.raises(TaskEvaluationContractError, match="128 random bits"):
        replace(allocation, invocation_nonce="predictable")


@pytest.mark.parametrize("foreign_subject", ("reservation", "case", "leg"))
def test_matrix_projection_rejects_a_foreign_allocation_subject(
    tmp_path,
    monkeypatch,
    foreign_subject,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    committed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    materialized_case = prepared.cases[0]
    allocation = _allocation(
        committed.reservation.reservation.reservation_id,
        materialized_case,
        materialized_case.request_case.legs[0].leg_id,
    )
    if foreign_subject == "reservation":
        allocation = replace(
            allocation,
            reservation_id=content_id(
                "task-evaluation-reservation",
                {"foreign": True},
            ),
        )
    elif foreign_subject == "case":
        allocation = replace(
            allocation,
            evaluation_case_id=content_id(
                "task-evaluation-case",
                {"foreign": True},
            ),
        )
    else:
        allocation = replace(
            allocation,
            evaluation_leg_id=content_id(
                "task-evaluation-leg",
                {"foreign": True},
            ),
        )

    expected_error = (
        "durable reservation"
        if foreign_subject == "reservation"
        else "another case or leg"
    )
    with pytest.raises(TaskEvaluatorProtocolError, match=expected_error):
        build_task_evaluation_evaluator_request(
            prepared,
            committed.reservation,
            allocation,
        )
