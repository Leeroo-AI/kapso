from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import ExpertCandidateCommitRecord
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixProvenanceKind,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationLimits,
    VerifiedTaskEvaluationAdapterRuntime,
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationParent,
    materialize_task_evaluation_starting_artifacts,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    MaterializedTaskEvaluationCase,
    PreparedTaskEvaluationRequest,
    TaskEvaluationPreflightCoordinator,
    TaskEvaluationPreflightError,
    task_evaluation_materialization_usage,
)
from kapso.cross_run.expert.task_evaluation_request import (
    prepare_task_evaluation_request,
)
from kapso.cross_run.task_adapters import task_adapter_materialization_usage
from test_expert_candidate_workspace import released_workspace_fixture
from test_expert_candidates import bootstrap_candidate_closure
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


def _current_observation(prepared_plan, *, head_commit_sha="a" * 40):
    packet = prepared_plan.stored_candidate.closure.trigger_packet
    release = packet.parent_release
    return TaskEvaluationCurrentReleaseObservation.mint(
        scope_id=packet.scope_contract.scope_id,
        release_id=None if release is None else release.release_id,
        publication_id=(
            None
            if release is None
            else content_id("github-publication", {"release": release.release_id})
        ),
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repo_node",
        default_branch_head_commit_sha=head_commit_sha,
        current_pointer_digest=(None if release is None else "sha256:" + "b" * 64),
        validation_closure_ids=(
            ()
            if release is None
            else (content_id("expert-validation", {"release": release.release_id}),)
        ),
    )


class _CandidateReader:
    def __init__(self, stored_candidate):
        self.stored_candidate = stored_candidate
        self.calls = []

    def read(self, candidate_id):
        self.calls.append(candidate_id)
        return self.stored_candidate


class _ParentProvider:
    def __init__(self, parent):
        self.parent = parent
        self.calls = []

    def materialize_exact(self, release_manifest, parent_tree_receipt, limits):
        self.calls.append((release_manifest, parent_tree_receipt, limits))
        if self.parent is None:
            raise AssertionError("bootstrap must not materialize a parent")
        return self.parent


class _ExactAdapterProvider:
    def __init__(self, adapters, *, callback=None):
        self.adapters = {
            (
                adapter.manifest.task_adapter_manifest_id,
                adapter.verification_receipt.verification_receipt_id,
            ): adapter
            for adapter in adapters
        }
        self.callback = callback
        self.calls = []

    def resolve_exact_bounded(
        self,
        *,
        task_adapter_manifest_id,
        verification_receipt_id,
        maximum_entries,
        maximum_bytes,
        timeout_seconds,
    ):
        self.calls.append(
            (
                task_adapter_manifest_id,
                verification_receipt_id,
                maximum_entries,
                maximum_bytes,
                timeout_seconds,
            )
        )
        if self.callback is not None:
            self.callback()
        return self.adapters[(task_adapter_manifest_id, verification_receipt_id)]


class _CurrentAuthority:
    def __init__(self, observations):
        self.observations = observations
        self.calls = []

    def observe_task_evaluation_current(self, scope_id):
        self.calls.append(scope_id)
        return self.observations[len(self.calls) - 1]


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def _coordinator(
    *,
    validation_store,
    prepared_plan,
    parent,
    current_authority,
    adapter_provider=None,
    clock=None,
):
    candidate_reader = _CandidateReader(prepared_plan.stored_candidate)
    parent_provider = _ParentProvider(parent)
    exact_adapter_provider = adapter_provider or _ExactAdapterProvider(
        prepared_plan.verified_adapters
    )
    coordinator = TaskEvaluationPreflightCoordinator(
        settings=validation_store.settings,
        plan_reservation_authority=validation_store,
        candidate_reader=candidate_reader,
        parent_provider=parent_provider,
        adapter_provider=exact_adapter_provider,
        current_release_authority=current_authority,
        monotonic_clock=clock or _Clock(),
    )
    return coordinator, candidate_reader, parent_provider, exact_adapter_provider


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
            current_release_observation=_current_observation(prepared_plan),
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
        current_release_observation=_current_observation(prepared_plan),
        cases=_materialized_cases(plan_join, prepared_plan.verified_adapters),
    )

    assert prepared.parent is None
    assert prepared.entry_count > prepared.candidate.entry_count
    assert all(len(case.request_case.legs) == 1 for case in prepared.cases)


def test_coordinator_materializes_exact_active_package_under_three_fences(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        rotate_active_adapter=True,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    _candidate, parent = _expert_sources(prepared_plan)
    observation = _current_observation(prepared_plan)
    current_authority = _CurrentAuthority((observation, observation))
    coordinator, candidate_reader, parent_provider, adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=parent,
        current_authority=current_authority,
    )

    prepared = coordinator.build(plan_reservation)

    executable_manifest_ids = {
        authority.task_adapter_manifest.task_adapter_manifest_id
        for authority in prepared_plan.plan.adapter_authorities
        if any(
            provenance.adapter_authority_id == authority.adapter_authority_id
            and provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
            for provenance in prepared_plan.plan.provenance_bindings
        )
    }
    assert candidate_reader.calls == [prepared_plan.plan.candidate_id]
    assert len(parent_provider.calls) == 1
    assert type(parent_provider.calls[0][2]) is TaskEvaluationMaterializationLimits
    assert current_authority.calls == [observation.scope_id, observation.scope_id]
    assert {call[0] for call in adapter_provider.calls} == executable_manifest_ids
    assert len(adapter_provider.calls) == len(executable_manifest_ids) == 1
    assert len(prepared_plan.plan.adapter_authorities) == 2
    assert prepared.current_release_observation == observation
    assert prepared.adapters == tuple(case.adapter for case in prepared.cases)


def test_bootstrap_coordinator_never_calls_parent_provider(
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
    current_authority = _CurrentAuthority((observation, observation))
    coordinator, _candidate_reader, parent_provider, adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=None,
        current_authority=current_authority,
    )

    prepared = coordinator.build(plan_reservation)

    assert prepared.parent is None
    assert parent_provider.calls == []
    assert len(adapter_provider.calls) == 1
    assert current_authority.calls == [observation.scope_id, observation.scope_id]


def test_coordinator_rejects_current_restoration_on_a_new_branch_head(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    _candidate, parent = _expert_sources(prepared_plan)
    before = _current_observation(prepared_plan, head_commit_sha="a" * 40)
    restored = _current_observation(prepared_plan, head_commit_sha="c" * 40)
    current_authority = _CurrentAuthority((before, restored))
    coordinator, _reader, _parent_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=parent,
        current_authority=current_authority,
    )

    with pytest.raises(TaskEvaluationPreflightError, match="changed"):
        coordinator.build(plan_reservation)


def test_coordinator_rejects_foreign_candidate_before_external_reads(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    foreign_closure = bootstrap_candidate_closure()
    foreign_candidate = StoredExpertCandidate(
        root=prepared_plan.stored_candidate.root,
        closure=foreign_closure,
        commit_record=ExpertCandidateCommitRecord.mint(
            candidate_id=foreign_closure.manifest.candidate_id,
            file_checksums={
                "candidate.json": tree_or_blob_digest(b"foreign-candidate")
            },
        ),
    )
    candidate_reader = _CandidateReader(foreign_candidate)
    _candidate, parent = _expert_sources(prepared_plan)
    parent_provider = _ParentProvider(parent)
    adapter_provider = _ExactAdapterProvider(prepared_plan.verified_adapters)
    observation = _current_observation(prepared_plan)
    current_authority = _CurrentAuthority((observation, observation))
    coordinator = TaskEvaluationPreflightCoordinator(
        settings=validation_store.settings,
        plan_reservation_authority=validation_store,
        candidate_reader=candidate_reader,
        parent_provider=parent_provider,
        adapter_provider=adapter_provider,
        current_release_authority=current_authority,
        monotonic_clock=_Clock(),
    )

    with pytest.raises(ValueError, match="candidate differs"):
        coordinator.build(plan_reservation)

    assert parent_provider.calls == []
    assert adapter_provider.calls == []
    assert current_authority.calls == []


def test_coordinator_rejects_adapter_package_substitution(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        rotate_active_adapter=True,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    _candidate, parent = _expert_sources(prepared_plan)
    observation = _current_observation(prepared_plan)
    current_authority = _CurrentAuthority((observation, observation))
    adapter_provider = _ExactAdapterProvider(prepared_plan.verified_adapters)
    executable_authority = next(
        authority
        for authority in prepared_plan.plan.adapter_authorities
        if any(
            provenance.adapter_authority_id == authority.adapter_authority_id
            and provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
            for provenance in prepared_plan.plan.provenance_bindings
        )
    )
    requested_key = (
        executable_authority.task_adapter_manifest.task_adapter_manifest_id,
        executable_authority.verification_receipt.verification_receipt_id,
    )
    adapter_provider.adapters[requested_key] = next(
        adapter
        for adapter in prepared_plan.verified_adapters
        if adapter.manifest.task_adapter_manifest_id != requested_key[0]
    )
    coordinator, _reader, _parent_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=parent,
        current_authority=current_authority,
        adapter_provider=adapter_provider,
    )

    with pytest.raises(TaskEvaluationPreflightError, match="reserved authority"):
        coordinator.build(plan_reservation)

    assert len(current_authority.calls) == 1


def test_coordinator_rejects_stale_plan_before_external_provider_calls(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    validation_store.reducer.current_release_provider.release_id = content_id(
        "expert-base-release",
        {"generation": "successor"},
    )
    validation_store.publish_current_release_authority_invalidation(
        candidate_id=prepared_plan.plan.candidate_id,
        expected_validation_state_id=snapshot.state.validation_state_id,
    )
    _candidate, parent = _expert_sources(prepared_plan)
    observation = _current_observation(prepared_plan)
    current_authority = _CurrentAuthority((observation, observation))
    coordinator, candidate_reader, parent_provider, adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=parent,
        current_authority=current_authority,
    )

    with pytest.raises(ValueError, match="head changed"):
        coordinator.build(plan_reservation)

    assert candidate_reader.calls == []
    assert parent_provider.calls == []
    assert adapter_provider.calls == []
    assert current_authority.calls == []


def test_coordinator_rejects_plan_head_advance_during_materialization(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    _candidate, parent = _expert_sources(prepared_plan)
    observation = _current_observation(prepared_plan)
    current_authority = _CurrentAuthority((observation, observation))

    def advance_validation_head():
        validation_store.reducer.current_release_provider.release_id = content_id(
            "expert-base-release",
            {"generation": "successor"},
        )
        validation_store.publish_current_release_authority_invalidation(
            candidate_id=prepared_plan.plan.candidate_id,
            expected_validation_state_id=snapshot.state.validation_state_id,
        )

    adapter_provider = _ExactAdapterProvider(
        prepared_plan.verified_adapters,
        callback=advance_validation_head,
    )
    coordinator, _reader, parent_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=parent,
        current_authority=current_authority,
        adapter_provider=adapter_provider,
    )

    with pytest.raises(ValueError, match="head changed"):
        coordinator.build(plan_reservation)

    assert len(parent_provider.calls) == 1
    assert len(adapter_provider.calls) == 1
    assert len(current_authority.calls) == 2


def test_coordinator_rejects_timeout_during_exact_package_resolution(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    _candidate, parent = _expert_sources(prepared_plan)
    observation = _current_observation(prepared_plan)
    current_authority = _CurrentAuthority((observation, observation))
    clock = _Clock()
    adapter_provider = _ExactAdapterProvider(
        prepared_plan.verified_adapters,
        callback=lambda: setattr(
            clock,
            "value",
            validation_store.settings.policy.task_evaluation_materialization_timeout_seconds,
        ),
    )
    coordinator, _reader, _parent_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=parent,
        current_authority=current_authority,
        adapter_provider=adapter_provider,
        clock=clock,
    )

    with pytest.raises(TaskEvaluationPreflightError, match="deadline expired"):
        coordinator.build(plan_reservation)

    assert len(current_authority.calls) == 1
