from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertSourceReplayExecutionLegKind,
    ExpertValidationStage,
)
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplayCurrentReleaseObservation,
    source_replay_task_adapter_trust_observations,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)
from kapso.cross_run.expert.replay_comparison import (
    build_expert_source_replay_paired_comparison_receipt,
)
from kapso.cross_run.expert.replay_decision import (
    decide_expert_source_replay_stage,
)
from kapso.cross_run.expert.replay_decision_contracts import (
    ExpertSourceReplayStageDecision,
)
from kapso.cross_run.expert.replay_execution_store import (
    SourceReplayExecutionJournalEventKind,
)
from kapso.cross_run.expert.replay_publication import (
    ExpertSourceReplayDecisionPublicationCoordinator,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayPublicationError,
    ExpertSourceReplayStageResultRecord,
    SourceReplayDecisionPublicationFence,
    _build_expert_source_replay_stage_result_record,
    source_replay_publication_security_subject_ids,
)
from kapso.cross_run.expert.validation_operation_contracts import (
    ExpertValidationOperation,
    ExpertValidationOperationKind,
)
from kapso.cross_run.expert.validation_store import (
    ExpertValidationStore,
    ExpertValidationStoreError,
)
from test_expert_replay_execution_store import _remint
from test_expert_source_replay_comparison import _complete_execution


class _PublicationCurrentAuthority:
    def __init__(self, observation, calls):
        self.observation = observation
        self.calls = calls

    def current_release_observation(self, scope_id):
        self.calls.append("current")
        assert scope_id == self.observation.scope_id
        return self.observation


class _PublicationDenylistAuthority:
    def __init__(self, template, calls):
        self.template = template
        self.calls = calls

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.calls.append("denylist")
        assert scope_id == self.template.scope_id
        assert scope_contract_id == self.template.scope_contract_id
        return _remint(
            self.template,
            checked_subject_ids=checked_subject_ids,
            denied_subject_ids=(),
        )


def _publication_coordinator(
    fixture,
    execution_store,
    result,
    calls,
    validation_store=None,
):
    return ExpertSourceReplayDecisionPublicationCoordinator(
        validation_store=(
            fixture.validation_store if validation_store is None else validation_store
        ),
        execution_store=execution_store,
        current_release_authority=_PublicationCurrentAuthority(
            result.publication_authority_fence.current_release_observation,
            calls,
        ),
        task_adapter_authority=fixture.adapter_provider,
        security_denylist_authority=_PublicationDenylistAuthority(
            result.publication_authority_fence.security_denylist_observation,
            calls,
        ),
    )


def _publication_evidence(tmp_path, aggregate_by_leg_kind=None):
    fixture, prepared, reservation, execution_store, completed = _complete_execution(
        tmp_path,
        aggregate_by_leg_kind=aggregate_by_leg_kind,
    )
    receipt = build_expert_source_replay_paired_comparison_receipt(
        completed_execution=completed,
        execution_store=execution_store,
        reservation=reservation,
        prepared_request=prepared,
    )
    decision = decide_expert_source_replay_stage(
        paired_comparison_receipt=receipt,
        prepared_request=prepared,
    )
    parent = prepared.parent.release_manifest
    current = SourceReplayCurrentReleaseObservation.mint(
        scope_id=parent.scope_id,
        release_id=parent.release_id,
        publication_id=content_id(
            "github-publication",
            {"final_publication": parent.release_id},
        ),
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repo_node",
        current_pointer_digest=tree_or_blob_digest(b"final CURRENT"),
        current_pointer_commit_sha="c" * 40,
        validation_closure_ids=(),
    )
    adapter_observations = source_replay_task_adapter_trust_observations(prepared)
    security_subjects = source_replay_publication_security_subject_ids(
        prepared_request=prepared,
        reservation=reservation,
        paired_comparison_receipt=receipt,
        stage_decision=decision,
        execution_events=completed.events,
        current_release_observation=current,
        task_adapter_trust_observations=adapter_observations,
    )
    denylist = SecurityDenylistObservation.mint(
        scope_id=parent.scope_id,
        scope_contract_id=prepared.request.scope_contract_id,
        scope_repository_binding_hash=tree_or_blob_digest(b"scope binding"),
        snapshot_id=content_id(
            "security-denylist-snapshot",
            {"generation": 8},
        ),
        generation=8,
        publication_id=content_id(
            "github-publication",
            {"security_denylist_generation": 8},
        ),
        repository_full_name="Leeroo-AI/kapso-security",
        repository_node_id="security_repo_node",
        pointer_digest=tree_or_blob_digest(b"security CURRENT 8"),
        authority_commit_sha="d" * 40,
        release_attestation_ref="attestations/security-denylist",
        checked_subject_ids=security_subjects,
        denied_subject_ids=(),
    )
    request = prepared.request
    fence = SourceReplayDecisionPublicationFence.mint(
        reservation_id=reservation.reservation_id,
        execution_request_id=request.execution_request_id,
        authorization_transition_id=reservation.authorization_transition_id,
        authorization_state_id=reservation.authorization_state_id,
        validation_attempt_id=reservation.validation_attempt_id,
        candidate_id=reservation.candidate_id,
        candidate_tree_hash=reservation.candidate_tree_hash,
        scope_id=parent.scope_id,
        scope_contract_id=request.scope_contract_id,
        expected_parent_release_id=request.parent_release_id,
        validation_policy_id=request.validation_policy_id,
        configuration_fingerprint=request.configuration_fingerprint,
        paired_comparison_receipt_id=receipt.paired_comparison_receipt_id,
        source_replay_stage_decision_id=decision.source_replay_stage_decision_id,
        outcome=decision.outcome,
        current_release_observation=current,
        task_adapter_trust_observations=adapter_observations,
        security_denylist_observation=denylist,
    )
    result = _build_expert_source_replay_stage_result_record(
        reservation=reservation,
        prepared_request=prepared,
        paired_comparison_receipt=receipt,
        stage_decision=decision,
        publication_authority_fence=fence,
    )
    return fixture, prepared, reservation, execution_store, completed, fence, result


def test_source_stage_result_is_self_contained_without_evaluator_evidence(tmp_path):
    _, _, _, _, _, fence, result = _publication_evidence(tmp_path)

    assert result.paired_comparison_receipt.paired_comparison_receipt_id in (
        result.exact_dependency_ids
    )
    assert result.stage_decision.source_replay_stage_decision_id in (
        result.exact_dependency_ids
    )
    assert fence.fence_id in result.exact_dependency_ids
    assert (
        ExpertSourceReplayStageResultRecord.from_json_bytes(result.to_json_bytes())
        == result
    )
    assert (
        SourceReplayDecisionPublicationFence.from_json_bytes(fence.to_json_bytes())
        == fence
    )
    assert b"invocation_allocation" not in fence.to_json_bytes()
    assert b"evaluator_run" not in result.to_json_bytes()
    assert b"attestation_envelope" not in result.to_json_bytes()


def test_final_security_closure_expands_every_spawn_fence_and_provider_handle(
    tmp_path,
):
    _, _, _, _, completed, fence, _ = _publication_evidence(tmp_path)
    spawn_events = tuple(
        event
        for event in completed.events
        if event.event_kind is SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED
    )

    assert len(spawn_events) == 2
    assert all(
        event.spawn_authority_fence.fence_id in fence.security_subject_ids
        and event.provider_execution_handle.provider_handle_id
        in fence.security_subject_ids
        and event.spawn_authority_fence.security_denylist_observation.snapshot_id
        in fence.security_subject_ids
        for event in spawn_events
    )
    assert all(
        observation.verifier_authority_subject_id in fence.security_subject_ids
        for event in spawn_events
        for observation in event.spawn_authority_fence.task_adapter_trust_observations
    )


def test_source_stage_result_rejects_extra_or_substituted_dependency(tmp_path):
    _, _, _, _, _, _, result = _publication_evidence(tmp_path)
    extra = content_id("unrelated-publication-input", {"value": 1})

    with pytest.raises(ExpertSourceReplayPublicationError, match="not exact"):
        _remint(
            result,
            exact_dependency_ids=tuple(sorted((*result.exact_dependency_ids, extra))),
        )

    with pytest.raises(ExpertSourceReplayPublicationError, match="inconsistent"):
        _remint(
            result,
            publication_authority_fence=_remint(
                result.publication_authority_fence,
                configuration_fingerprint=tree_or_blob_digest(b"other config"),
            ),
        )


def test_source_stage_result_requires_full_receipt_closure_in_final_denylist(
    tmp_path,
):
    _, prepared, reservation, _, _, fence, result = _publication_evidence(tmp_path)
    event_id = result.paired_comparison_receipt.execution_journal_event_ids[-1]
    narrowed_subjects = tuple(
        subject_id
        for subject_id in fence.security_subject_ids
        if subject_id != event_id
    )
    narrowed_denylist = _remint(
        fence.security_denylist_observation,
        checked_subject_ids=narrowed_subjects,
    )
    narrowed_fence = _remint(
        fence,
        security_denylist_observation=narrowed_denylist,
    )

    with pytest.raises(ExpertSourceReplayPublicationError, match="inconsistent"):
        _build_expert_source_replay_stage_result_record(
            reservation=reservation,
            prepared_request=prepared,
            paired_comparison_receipt=result.paired_comparison_receipt,
            stage_decision=result.stage_decision,
            publication_authority_fence=narrowed_fence,
        )


def test_source_stage_result_rejects_substituted_adapter_or_scientific_decision(
    tmp_path,
):
    _, prepared, reservation, _, _, fence, result = _publication_evidence(tmp_path)
    expected_observation = fence.task_adapter_trust_observations[0]
    substituted_observation = TaskAdapterTrustObservation.mint(
        task_adapter_manifest_id=expected_observation.task_adapter_manifest_id,
        verification_receipt_id=expected_observation.verification_receipt_id,
        verifier_id=expected_observation.verifier_id,
        verifier_version="substituted-verifier-version",
        dependency_ids=expected_observation.dependency_ids,
    )
    expanded_subjects = tuple(
        sorted(
            {
                *fence.security_subject_ids,
                substituted_observation.observation_id,
                substituted_observation.verifier_authority_subject_id,
            }
        )
    )
    substituted_fence = _remint(
        fence,
        task_adapter_trust_observations=(substituted_observation,),
        security_denylist_observation=_remint(
            fence.security_denylist_observation,
            checked_subject_ids=expanded_subjects,
        ),
    )

    with pytest.raises(ExpertSourceReplayPublicationError, match="prepared authority"):
        _build_expert_source_replay_stage_result_record(
            reservation=reservation,
            prepared_request=prepared,
            paired_comparison_receipt=result.paired_comparison_receipt,
            stage_decision=result.stage_decision,
            publication_authority_fence=substituted_fence,
        )


def test_source_stage_reducer_passes_and_preserves_a_typed_history_prefix(tmp_path):
    fixture, _, _, _, _, _, result = _publication_evidence(tmp_path)
    snapshot = fixture.validation_store.snapshot(result.candidate_id)
    assert snapshot is not None
    assert snapshot.latest_attempt is not None
    initial_count = len(snapshot.accepted_stage_results)

    after_source = fixture.validation_store.reducer.advance_source_replay_stage(
        state=snapshot.state,
        attempt=snapshot.latest_attempt,
        accepted_results=snapshot.accepted_stage_results,
        result=result,
    )

    assert after_source.promotion_state is ExpertPromotionState.VALIDATING
    assert len(after_source.accepted_stage_results) == initial_count + 1
    assert after_source.accepted_stage_results[-1].stage is (
        ExpertValidationStage.SOURCE_RUN_REPLAY
    )
    assert after_source.accepted_stage_results[-1].stage_result_record_id == (
        result.stage_result_record_id
    )
    fixture.validation_store.reducer._validate_accepted_history(
        state=after_source,
        attempt=snapshot.latest_attempt,
        accepted_results=(*snapshot.accepted_stage_results, result),
    )
    assert after_source.next_stage is ExpertValidationStage.AUTOMATED_REVIEW


def test_source_stage_reducer_failure_preserves_the_accepted_prefix(tmp_path):
    fixture, prepared, reservation, _, _, fence, result = _publication_evidence(
        tmp_path,
        aggregate_by_leg_kind={
            ExpertSourceReplayExecutionLegKind.CONTROL_PARENT: 0.8,
            ExpertSourceReplayExecutionLegKind.CANDIDATE: 0.0,
        },
    )
    snapshot = fixture.validation_store.snapshot(result.candidate_id)
    assert snapshot is not None
    assert snapshot.latest_attempt is not None
    assert result.outcome is ExpertEvaluatorOutcome.CANDIDATE_FAILED

    forged_decision = ExpertSourceReplayStageDecision.mint(
        paired_comparison_receipt_id=(
            result.paired_comparison_receipt.paired_comparison_receipt_id
        ),
        validation_policy_id=result.validation_policy_id,
        decision_policy_version=result.stage_decision.decision_policy_version,
        outcome=ExpertEvaluatorOutcome.PASSED,
        hard_regression_comparisons=(),
        paired_comparison_dependency_ids=(
            result.stage_decision.paired_comparison_dependency_ids
        ),
        exact_dependency_ids=result.stage_decision.exact_dependency_ids,
    )
    forged_subjects = tuple(
        sorted(
            {
                *fence.security_subject_ids,
                forged_decision.source_replay_stage_decision_id,
            }
        )
    )
    forged_fence = _remint(
        fence,
        source_replay_stage_decision_id=(
            forged_decision.source_replay_stage_decision_id
        ),
        outcome=ExpertEvaluatorOutcome.PASSED,
        security_denylist_observation=_remint(
            fence.security_denylist_observation,
            checked_subject_ids=forged_subjects,
        ),
    )
    with pytest.raises(ExpertSourceReplayPublicationError, match="prepared authority"):
        _build_expert_source_replay_stage_result_record(
            reservation=reservation,
            prepared_request=prepared,
            paired_comparison_receipt=result.paired_comparison_receipt,
            stage_decision=forged_decision,
            publication_authority_fence=forged_fence,
        )

    failed = fixture.validation_store.reducer.advance_source_replay_stage(
        state=snapshot.state,
        attempt=snapshot.latest_attempt,
        accepted_results=snapshot.accepted_stage_results,
        result=result,
    )

    assert failed.promotion_state is ExpertPromotionState.FAILED
    assert failed.accepted_stage_results == snapshot.state.accepted_stage_results
    assert failed.next_stage is None
    assert failed.terminal_evidence_ids == (result.stage_result_record_id,)
    assert failed.transition_evidence_id == result.stage_result_record_id


def test_validation_store_publishes_and_reopens_one_typed_source_result(tmp_path):
    (
        fixture,
        prepared,
        reservation,
        execution_store,
        completed,
        _,
        expected_result,
    ) = _publication_evidence(tmp_path)
    store = fixture.validation_store
    calls = []
    coordinator = _publication_coordinator(
        fixture,
        execution_store,
        expected_result,
        calls,
    )
    evaluator_result_root = store.object_root / "expert-evaluator-result-record"
    evaluator_result_names = tuple(
        sorted(path.name for path in evaluator_result_root.iterdir())
    )

    committed = coordinator.publish_completed(
        completed_execution=completed,
        reservation=reservation,
        prepared_request=prepared,
    )

    assert committed.replayed is False
    assert committed.stage_result == expected_result
    assert committed.snapshot.state.promotion_state is ExpertPromotionState.VALIDATING
    assert committed.snapshot.accepted_stage_results[-1] == expected_result
    assert calls == ["current", "denylist"]
    assert tuple(sorted(path.name for path in evaluator_result_root.iterdir())) == (
        evaluator_result_names
    )
    with store._lock(exclusive=False):
        operation = store._read_contract_unlocked(
            committed.snapshot.transition.operation_id,
            ExpertValidationOperation,
        )
    assert operation.operation_kind is (
        ExpertValidationOperationKind.SOURCE_REPLAY_STAGE_RESULT
    )
    assert operation.candidate_id == reservation.candidate_id
    assert operation.expected_transition_id == (reservation.authorization_transition_id)
    assert operation.request_record_id == reservation.reservation_id

    reopened = ExpertValidationStore(
        store.root,
        store.state_root,
        store.settings,
        store.reducer,
    )
    recovered = reopened.snapshot(reservation.candidate_id)
    replay_coordinator = _publication_coordinator(
        fixture,
        execution_store,
        expected_result,
        calls,
        validation_store=reopened,
    )
    replay_coordinator.current_release_authority.current_release_observation = (
        lambda _scope_id: pytest.fail("exact replay fetched CURRENT")
    )
    replayed = replay_coordinator.publish_completed(
        completed_execution=object(),
        reservation=reservation,
        prepared_request=prepared,
    )

    assert recovered == committed.snapshot
    assert recovered.accepted_stage_results[-1] == expected_result
    assert replayed is not None
    assert replayed.replayed is True
    assert replayed.stage_result == expected_result
    assert replayed.snapshot == committed.snapshot
    assert calls == ["current", "denylist"]


def test_validation_store_source_failure_preserves_prefix_and_terminalizes(tmp_path):
    (
        fixture,
        prepared,
        reservation,
        execution_store,
        completed,
        _,
        expected_result,
    ) = _publication_evidence(
        tmp_path,
        aggregate_by_leg_kind={
            ExpertSourceReplayExecutionLegKind.CONTROL_PARENT: 0.8,
            ExpertSourceReplayExecutionLegKind.CANDIDATE: 0.0,
        },
    )
    store = fixture.validation_store
    before = store.snapshot(reservation.candidate_id)
    assert before is not None
    coordinator = _publication_coordinator(
        fixture,
        execution_store,
        expected_result,
        [],
    )

    committed = coordinator.publish_completed(
        completed_execution=completed,
        reservation=reservation,
        prepared_request=prepared,
    )

    assert committed.snapshot.state.promotion_state is ExpertPromotionState.FAILED
    assert committed.snapshot.accepted_stage_results == before.accepted_stage_results
    assert committed.snapshot.state.accepted_stage_results == (
        before.state.accepted_stage_results
    )
    assert committed.snapshot.state.terminal_evidence_ids == (
        expected_result.stage_result_record_id,
    )


def test_validation_store_rejects_non_production_publication_registration(tmp_path):
    fixture, _, _, _, _, _, _ = _publication_evidence(tmp_path)

    with pytest.raises(
        ExpertValidationStoreError,
        match="another publication coordinator",
    ):
        fixture.validation_store._bind_source_replay_publication_authority(object())


def test_concurrent_source_publishers_converge_on_one_reservation_result(tmp_path):
    (
        fixture,
        prepared,
        reservation,
        execution_store,
        completed,
        _,
        expected_result,
    ) = _publication_evidence(tmp_path)
    coordinator = _publication_coordinator(
        fixture,
        execution_store,
        expected_result,
        [],
    )

    def publish(_position):
        return coordinator.publish_completed(
            completed_execution=completed,
            reservation=reservation,
            prepared_request=prepared,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(publish, range(2)))

    assert sum(not result.replayed for result in results) == 1
    assert len({result.snapshot.transition.transition_id for result in results}) == 1
    assert all(result.stage_result == expected_result for result in results)


def test_publication_rejects_changed_current_or_nonexact_denylist(tmp_path):
    (
        fixture,
        prepared,
        reservation,
        execution_store,
        completed,
        _,
        expected_result,
    ) = _publication_evidence(tmp_path)
    before = fixture.validation_store.snapshot(reservation.candidate_id)
    assert before is not None
    coordinator = _publication_coordinator(
        fixture,
        execution_store,
        expected_result,
        [],
    )
    current = expected_result.publication_authority_fence.current_release_observation
    coordinator.current_release_authority.observation = _remint(
        current,
        release_id=content_id("expert-base-release", {"changed": True}),
        publication_id=content_id("github-publication", {"changed": True}),
        current_pointer_digest=tree_or_blob_digest(b"changed CURRENT"),
        current_pointer_commit_sha="e" * 40,
    )

    with pytest.raises(ExpertSourceReplayPublicationError, match="current release"):
        coordinator.publish_completed(
            completed_execution=completed,
            reservation=reservation,
            prepared_request=prepared,
        )
    assert fixture.validation_store.snapshot(reservation.candidate_id) == before

    coordinator.current_release_authority.observation = current
    exact_observe = coordinator.security_denylist_authority.observe_exact

    def omit_one_subject(**request):
        observation = exact_observe(**request)
        return _remint(
            observation,
            checked_subject_ids=observation.checked_subject_ids[:-1],
        )

    coordinator.security_denylist_authority.observe_exact = omit_one_subject
    with pytest.raises(ExpertSourceReplayPublicationError, match="denylist"):
        coordinator.publish_completed(
            completed_execution=completed,
            reservation=reservation,
            prepared_request=prepared,
        )
    assert fixture.validation_store.snapshot(reservation.candidate_id) == before
