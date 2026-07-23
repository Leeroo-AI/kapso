from dataclasses import fields

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationAuthorityError,
    TaskEvaluationCurrentReleaseObservation,
    TaskEvaluationSpawnAuthorityFence,
)
from kapso.cross_run.expert.task_evaluation_authority_projection import (
    build_task_evaluation_spawn_authority_fence,
    task_evaluation_adapter_trust_observations,
    task_evaluation_allocation_case_leg,
    task_evaluation_spawn_security_subject_ids,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationCase,
    TaskEvaluationInvocationAllocation,
    TaskEvaluationLegKind,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)
from security_denylist_fixtures import matched_security_revocations
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
)
from test_expert_task_evaluation_preflight import (
    _CurrentAuthority,
    _coordinator,
    _current_observation,
)
from test_expert_task_evaluation_reservation import _parent_prepared


def _reserve(validation_store, snapshot, prepared):
    return validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    ).reservation


def _allocation(reservation_snapshot, prepared):
    case = prepared.plan_join.request.cases[0]
    return TaskEvaluationInvocationAllocation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        evaluation_case_id=case.evaluation_case_id,
        evaluation_leg_id=case.legs[0].leg_id,
        invocation_nonce="a" * 32,
    )


def _second_case_with_shared_legs(first_case):
    adapter_authority_id = content_id(
        "expert-release-matrix-adapter-authority",
        {"second": True},
    )
    provenance_binding_id = content_id(
        "expert-release-matrix-provenance-binding",
        {"second": True},
    )
    release_matrix_case_id = content_id(
        "task-adapter-release-matrix-case",
        {"second": True},
    )
    task_context_binding_id = content_id(
        "task-context-binding",
        {"second": True},
    )
    independence_group_id = content_id(
        "task-adapter-release-matrix-independence-group",
        {"second": True},
    )
    evaluation_cell_ids = tuple(
        sorted(
            content_id(
                "expert-release-matrix-evaluation-cell",
                {"position": position, "second": True},
            )
            for position, _fingerprint_id in enumerate(
                first_case.evaluation_fingerprint_ids
            )
        )
    )
    exact_dependency_ids = tuple(
        sorted(
            {
                adapter_authority_id,
                provenance_binding_id,
                release_matrix_case_id,
                task_context_binding_id,
                independence_group_id,
                *evaluation_cell_ids,
                *first_case.evaluation_fingerprint_ids,
                *first_case.starting_artifact_ids,
                first_case.compute_binding.compute_binding_id,
                *(leg.leg_id for leg in first_case.legs),
                *(
                    dependency_id
                    for leg in first_case.legs
                    for dependency_id in leg.exact_dependency_ids
                ),
            }
        )
    )
    return TaskEvaluationCase.mint(
        adapter_authority_id=adapter_authority_id,
        provenance_binding_id=provenance_binding_id,
        release_matrix_case_id=release_matrix_case_id,
        task_context_binding_id=task_context_binding_id,
        independence_group_id=independence_group_id,
        evaluation_cell_ids=evaluation_cell_ids,
        evaluation_fingerprint_ids=first_case.evaluation_fingerprint_ids,
        starting_artifact_ids=first_case.starting_artifact_ids,
        compute_binding=first_case.compute_binding,
        legs=first_case.legs,
        exact_dependency_ids=exact_dependency_ids,
    )


def _denylist(prepared, checked_subject_ids, *, matched_subject_ids=()):
    request = prepared.plan_join.request
    return SecurityDenylistObservation.mint(
        scope_id=request.scope_id,
        scope_contract_id=request.scope_contract_id,
        scope_repository_binding_hash=tree_or_blob_digest(b"scope binding"),
        snapshot_id=content_id("security-denylist-snapshot", {"generation": 7}),
        generation=7,
        publication_id=content_id(
            "github-publication",
            {"security_denylist_generation": 7},
        ),
        repository_full_name="Leeroo-AI/kapso-security",
        repository_node_id="security_repo_node",
        pointer_digest=tree_or_blob_digest(b"security CURRENT"),
        authority_commit_sha="b" * 40,
        release_attestation_ref="attestations/security-denylist",
        checked_subject_ids=checked_subject_ids,
        matched_revocations=matched_security_revocations(matched_subject_ids),
    )


def _moved_current(current):
    values = current.to_dict()
    values.pop("observation_id")
    values["default_branch_head_commit_sha"] = "c" * 40
    return TaskEvaluationCurrentReleaseObservation.mint(**values)


def _build_fence(prepared, reservation_snapshot, current):
    allocation = _allocation(reservation_snapshot, prepared)
    adapter_observations = task_evaluation_adapter_trust_observations(prepared)
    subject_ids = task_evaluation_spawn_security_subject_ids(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
        stable_current_release_observation=current,
        task_adapter_trust_observations=adapter_observations,
    )
    fence = build_task_evaluation_spawn_authority_fence(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
        stable_current_release_observation=current,
        task_adapter_trust_observations=adapter_observations,
        security_denylist_observation=_denylist(prepared, subject_ids),
    )
    return fence, subject_ids, adapter_observations


def _bootstrap_authority(tmp_path, monkeypatch):
    validation_store, snapshot, prepared_plan, _active_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    current = _current_observation(prepared_plan)
    coordinator, _candidate_reader, source_base_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        source_base=None,
        current_authority=_CurrentAuthority((current, current)),
    )
    prepared = coordinator.build(plan_reservation)
    reservation_snapshot = _reserve(validation_store, snapshot, prepared)
    return prepared, reservation_snapshot, source_base_provider


def test_parent_spawn_fence_binds_complete_exact_fresh_authority(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    reservation_snapshot = _reserve(validation_store, snapshot, prepared)
    moved_current = _moved_current(prepared.current_release_observation)

    fence, subject_ids, adapter_observations = _build_fence(
        prepared,
        reservation_snapshot,
        moved_current,
    )

    assert fence.stable_current_release_observation == moved_current
    assert moved_current != prepared.current_release_observation
    assert fence.security_subject_ids == subject_ids
    assert fence.task_adapter_trust_observations == adapter_observations
    assert set(reservation_snapshot.reservation.exact_dependency_ids).issubset(
        subject_ids
    )
    assert set(prepared.plan_join.request.exact_dependency_ids).issubset(subject_ids)
    assert set(prepared.source_base.release_manifest.consumed_dependency_ids).issubset(
        subject_ids
    )
    first_case = prepared.plan_join.request.cases[0]
    second_case = _second_case_with_shared_legs(first_case)
    first_candidate_leg = next(
        leg for leg in first_case.legs if leg.kind is TaskEvaluationLegKind.CANDIDATE
    )
    second_candidate_leg = next(
        leg for leg in second_case.legs if leg.kind is TaskEvaluationLegKind.CANDIDATE
    )
    assert first_candidate_leg.leg_id == second_candidate_leg.leg_id
    repeated_leg_allocation = TaskEvaluationInvocationAllocation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        evaluation_case_id=second_case.evaluation_case_id,
        evaluation_leg_id=first_candidate_leg.leg_id,
        invocation_nonce="c" * 32,
    )
    assert (
        task_evaluation_allocation_case_leg(
            (first_case, second_case),
            repeated_leg_allocation,
        )
        == second_candidate_leg
    )
    foreign_case_allocation = TaskEvaluationInvocationAllocation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        evaluation_case_id=content_id("task-evaluation-case", {"foreign": True}),
        evaluation_leg_id=first_candidate_leg.leg_id,
        invocation_nonce="d" * 32,
    )
    with pytest.raises(TaskEvaluationAuthorityError, match="case leg"):
        task_evaluation_allocation_case_leg(
            (first_case, second_case),
            foreign_case_allocation,
        )
    assert (
        TaskEvaluationSpawnAuthorityFence.from_json_bytes(fence.to_json_bytes())
        == fence
    )
    assert {field.name for field in fields(TaskEvaluationSpawnAuthorityFence)} == {
        "fence_id",
        "reservation_id",
        "request_id",
        "invocation_allocation",
        "stable_current_release_observation",
        "task_adapter_trust_observations",
        "security_denylist_observation",
    }


def test_bootstrap_spawn_fence_preserves_authenticated_absence(
    tmp_path,
    monkeypatch,
):
    prepared, reservation_snapshot, source_base_provider = _bootstrap_authority(
        tmp_path,
        monkeypatch,
    )

    fence, subject_ids, adapter_observations = _build_fence(
        prepared,
        reservation_snapshot,
        prepared.current_release_observation,
    )

    assert fence.stable_current_release_observation.release_id is None
    assert fence.stable_current_release_observation.publication_id is None
    assert fence.stable_current_release_observation.current_pointer_digest is None
    assert fence.task_adapter_trust_observations == adapter_observations
    assert fence.security_subject_ids == subject_ids
    assert source_base_provider.calls == []


def test_spawn_fence_rejects_nonexact_authority_inputs(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    reservation_snapshot = _reserve(validation_store, snapshot, prepared)
    current = prepared.current_release_observation
    allocation = _allocation(reservation_snapshot, prepared)
    adapter_observations = task_evaluation_adapter_trust_observations(prepared)
    subject_ids = task_evaluation_spawn_security_subject_ids(
        prepared_request=prepared,
        reservation_snapshot=reservation_snapshot,
        invocation_allocation=allocation,
        stable_current_release_observation=current,
        task_adapter_trust_observations=adapter_observations,
    )

    extra_subject_ids = tuple(
        sorted(
            {
                *subject_ids,
                content_id("unexpected-security-subject", {"extra": True}),
            }
        )
    )
    with pytest.raises(TaskEvaluationAuthorityError, match="exact security"):
        build_task_evaluation_spawn_authority_fence(
            prepared_request=prepared,
            reservation_snapshot=reservation_snapshot,
            invocation_allocation=allocation,
            stable_current_release_observation=current,
            task_adapter_trust_observations=adapter_observations,
            security_denylist_observation=_denylist(prepared, extra_subject_ids),
        )

    substituted_observation = TaskAdapterTrustObservation.mint(
        task_adapter_manifest_id=(adapter_observations[0].task_adapter_manifest_id),
        verification_receipt_id=adapter_observations[0].verification_receipt_id,
        verifier_id=adapter_observations[0].verifier_id,
        verifier_version="substituted_verifier_v2",
        dependency_ids=adapter_observations[0].dependency_ids,
    )
    substituted_observations = tuple(
        sorted(
            (substituted_observation, *adapter_observations[1:]),
            key=lambda observation: observation.observation_id,
        )
    )
    with pytest.raises(TaskEvaluationAuthorityError, match="prepared authority"):
        task_evaluation_spawn_security_subject_ids(
            prepared_request=prepared,
            reservation_snapshot=reservation_snapshot,
            invocation_allocation=allocation,
            stable_current_release_observation=current,
            task_adapter_trust_observations=substituted_observations,
        )

    foreign_allocation = TaskEvaluationInvocationAllocation(
        reservation_id=reservation_snapshot.reservation.reservation_id,
        evaluation_case_id=content_id("task-evaluation-case", {"foreign": True}),
        evaluation_leg_id=allocation.evaluation_leg_id,
        invocation_nonce="b" * 32,
    )
    with pytest.raises(TaskEvaluationAuthorityError, match="case leg"):
        task_evaluation_spawn_security_subject_ids(
            prepared_request=prepared,
            reservation_snapshot=reservation_snapshot,
            invocation_allocation=foreign_allocation,
            stable_current_release_observation=current,
            task_adapter_trust_observations=adapter_observations,
        )
