from __future__ import annotations

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplayCurrentReleaseObservation,
    SourceReplaySecurityDenylistObservation,
    SourceReplayTaskAdapterTrustObservation,
    source_replay_task_adapter_trust_observations,
)
from kapso.cross_run.expert.replay_comparison import (
    build_expert_source_replay_paired_comparison_receipt,
)
from kapso.cross_run.expert.replay_decision import (
    decide_expert_source_replay_stage,
)
from kapso.cross_run.expert.replay_decision_contracts import (
    ExpertSourceReplayDecisionError,
)
from kapso.cross_run.expert.replay_execution_store import (
    SourceReplayExecutionJournalEventKind,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayPublicationError,
    ExpertSourceReplayStageResultRecord,
    SourceReplayDecisionPublicationFence,
    _build_expert_source_replay_stage_result_record,
    source_replay_publication_security_subject_ids,
)
from test_expert_replay_execution_store import _remint
from test_expert_source_replay_comparison import _complete_execution


def _publication_evidence(tmp_path):
    fixture, prepared, reservation, execution_store, completed = _complete_execution(
        tmp_path
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
    denylist = SourceReplaySecurityDenylistObservation.mint(
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
    return fixture, prepared, reservation, completed, fence, result


def test_source_stage_result_is_self_contained_without_evaluator_evidence(tmp_path):
    _, _, _, _, fence, result = _publication_evidence(tmp_path)

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
    _, _, _, completed, fence, _ = _publication_evidence(tmp_path)
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
    _, _, _, _, _, result = _publication_evidence(tmp_path)
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
    _, prepared, reservation, _, fence, result = _publication_evidence(tmp_path)
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
    _, prepared, reservation, _, fence, result = _publication_evidence(tmp_path)
    expected_observation = fence.task_adapter_trust_observations[0]
    substituted_observation = SourceReplayTaskAdapterTrustObservation.mint(
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

    substituted_receipt = _remint(
        result.paired_comparison_receipt,
        aggregate_recomputation_tolerance=0.1,
    )
    with pytest.raises(ExpertSourceReplayDecisionError, match="tolerance"):
        _build_expert_source_replay_stage_result_record(
            reservation=reservation,
            prepared_request=prepared,
            paired_comparison_receipt=substituted_receipt,
            stage_decision=result.stage_decision,
            publication_authority_fence=fence,
        )
