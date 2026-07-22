from dataclasses import replace

import pytest

from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationAdapterRuntime,
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationParent,
    materialize_task_evaluation_starting_artifacts,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    MaterializedTaskEvaluationCase,
    PreparedTaskEvaluationRequest,
    TaskEvaluationPreflightError,
    task_evaluation_materialization_usage,
)
from kapso.cross_run.expert.task_evaluation_request import (
    prepare_task_evaluation_request,
)
from kapso.cross_run.task_adapters import task_adapter_materialization_usage
from test_expert_candidate_workspace import released_workspace_fixture
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
    _release_matrix_fixture,
)


def _expert_sources(prepared_plan):
    stored = prepared_plan.stored_candidate
    candidate = VerifiedTaskEvaluationCandidate(
        manifest=stored.closure.manifest,
        commit_record=stored.commit_record,
        source_tree=stored.closure.candidate_tree,
        source_contents=stored.closure.candidate_contents,
    )
    packet = stored.closure.trigger_packet
    if packet.parent_release is None or packet.parent_tree_receipt is None:
        return candidate, None
    _released_packet, _materialized, parent_contents = released_workspace_fixture()
    return candidate, VerifiedTaskEvaluationParent(
        release_manifest=packet.parent_release,
        parent_tree_receipt=packet.parent_tree_receipt,
        source_contents=parent_contents,
    )


def _materialized_cases(plan_join, verified_adapters):
    plan = plan_join.plan_reservation.evaluation_plan
    provenances = {
        provenance.provenance_binding_id: provenance
        for provenance in plan.provenance_bindings
        if provenance.adapter_case is not None
    }
    authorities = {
        authority.adapter_authority_id: authority
        for authority in plan.adapter_authorities
    }
    adapters = {
        (
            adapter.manifest.task_adapter_manifest_id,
            adapter.verification_receipt.verification_receipt_id,
        ): adapter
        for adapter in verified_adapters
    }
    materialized = []
    for request_case in plan_join.request.cases:
        provenance = provenances[request_case.provenance_binding_id]
        authority = authorities[request_case.adapter_authority_id]
        adapter = adapters[
            (
                authority.task_adapter_manifest.task_adapter_manifest_id,
                authority.verification_receipt.verification_receipt_id,
            )
        ]
        assert provenance.adapter_case is not None
        materialized.append(
            MaterializedTaskEvaluationCase(
                request_case=request_case,
                adapter=adapter,
                adapter_runtime=(
                    VerifiedTaskEvaluationAdapterRuntime.from_verified_adapter(adapter)
                ),
                starting_artifacts=materialize_task_evaluation_starting_artifacts(
                    adapter=adapter,
                    signed_case=provenance.adapter_case,
                ),
            )
        )
    return tuple(materialized)


def _parent_preflight(
    tmp_path,
    monkeypatch,
    *,
    rotate_active_adapter=False,
    add_active_case=False,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        rotate_active_adapter=rotate_active_adapter,
        add_active_case=add_active_case,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    candidate, parent = _expert_sources(prepared_plan)
    plan_join = prepare_task_evaluation_request(
        plan_reservation=plan_reservation,
        settings=validation_store.settings,
        stored_candidate=prepared_plan.stored_candidate,
        candidate=candidate,
        parent=parent,
    )
    return (
        PreparedTaskEvaluationRequest(
            plan_join=plan_join,
            stored_candidate=prepared_plan.stored_candidate,
            candidate=candidate,
            parent=parent,
            cases=_materialized_cases(plan_join, prepared_plan.verified_adapters),
        ),
        prepared_plan,
    )


def test_prepared_request_keeps_only_executable_packages_and_counts_each_once(
    tmp_path,
    monkeypatch,
):
    prepared, plan_authority = _parent_preflight(
        tmp_path,
        monkeypatch,
        rotate_active_adapter=True,
    )
    adapter = prepared.adapters[0]
    adapter_usage = task_adapter_materialization_usage(
        source_file_sizes=tuple(
            descriptor.size
            for descriptor in adapter.source_extraction_receipt.source_tree_files
        ),
        source_archive_sizes=(len(adapter.source_archive),),
        proof_object_sizes=tuple(
            len(payload) for payload in adapter.proof_objects.values()
        ),
        publisher_verification_sizes=(len(adapter.publisher_verification),),
    )

    assert len(prepared.cases) == 1
    assert len(plan_authority.plan.adapter_authorities) == 2
    assert len(prepared.adapters) == 1
    assert prepared.parent is not None
    assert prepared.entry_count == (
        prepared.candidate.entry_count + prepared.parent.entry_count + adapter_usage[0]
    )
    assert prepared.byte_count == (
        prepared.candidate.byte_count + prepared.parent.byte_count + adapter_usage[1]
    )
    assert task_evaluation_materialization_usage(
        candidate=prepared.candidate,
        parent=prepared.parent,
        adapters=(adapter, adapter),
    ) == (prepared.entry_count, prepared.byte_count)


def test_prepared_request_rejects_subset_and_historical_adapter_substitution(
    tmp_path,
    monkeypatch,
):
    multi_case_root = tmp_path / "multi-case"
    rotated_root = tmp_path / "rotated"
    multi_case_root.mkdir()
    rotated_root.mkdir()
    prepared, _multi_case_plan = _parent_preflight(
        multi_case_root,
        monkeypatch,
        add_active_case=True,
    )

    with pytest.raises(
        TaskEvaluationPreflightError,
        match="case coverage is not exact",
    ):
        replace(prepared, cases=prepared.cases[:1])

    rotated, plan_authority = _parent_preflight(
        rotated_root,
        monkeypatch,
        rotate_active_adapter=True,
    )
    executable_adapter = rotated.adapters[0]
    historical_adapter = next(
        adapter
        for adapter in plan_authority.verified_adapters
        if adapter != executable_adapter
    )
    substituted_case = replace(
        rotated.cases[0],
        adapter=historical_adapter,
        adapter_runtime=(
            VerifiedTaskEvaluationAdapterRuntime.from_verified_adapter(
                historical_adapter
            )
        ),
    )
    with pytest.raises(
        TaskEvaluationPreflightError,
        match="differs from plan authority",
    ):
        replace(
            rotated,
            cases=(substituted_case,),
        )


def test_bootstrap_preflight_has_candidate_only_and_no_parent_usage(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan, _adapter_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    candidate, parent = _expert_sources(prepared_plan)
    assert parent is None
    plan_join = prepare_task_evaluation_request(
        plan_reservation=plan_reservation,
        settings=validation_store.settings,
        stored_candidate=prepared_plan.stored_candidate,
        candidate=candidate,
        parent=None,
    )

    prepared = PreparedTaskEvaluationRequest(
        plan_join=plan_join,
        stored_candidate=prepared_plan.stored_candidate,
        candidate=candidate,
        parent=None,
        cases=_materialized_cases(plan_join, prepared_plan.verified_adapters),
    )

    assert prepared.parent is None
    assert prepared.entry_count > prepared.candidate.entry_count
    assert all(len(case.request_case.legs) == 1 for case in prepared.cases)
