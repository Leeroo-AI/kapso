from __future__ import annotations

import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest

import kapso.cross_run.expert.composition_source as composition_source_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import (
    ExpertCandidateCommitRecord,
    ExpertCandidateValidationState,
    ExpertPromotionState,
)
from kapso.cross_run.expert.composition_source import (
    ApprovedExpertCompositionSource,
    ExpertCompositionSourceError,
    ExpertCompositionSourceResolver,
    project_expert_composition_source_reference,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
)
from kapso.cross_run.expert.store import ExpertCandidateStoreError
from test_expert_publication_eligibility import (
    _coordinator,
    terminal_cases,
)


def _terminalize(case):
    matrix = case.matrix_commit
    return _coordinator(case).coordinator.publish(
        candidate_id=matrix.snapshot.state.candidate_id,
        release_matrix_stage_result_id=matrix.stage_result.stage_result_record_id,
    )


@pytest.fixture(scope="module")
def composition_source_cases(terminal_cases):
    return SimpleNamespace(
        parent=_terminalize(terminal_cases.parent_approved),
        bootstrap=_terminalize(terminal_cases.bootstrap_approved),
        retained=_terminalize(terminal_cases.retained),
        failed=_terminalize(terminal_cases.failed),
        corruption=_terminalize(terminal_cases.adversarial),
        parent_store=terminal_cases.parent_approved.validation_store,
        bootstrap_store=terminal_cases.bootstrap_approved.validation_store,
        retained_store=terminal_cases.retained.validation_store,
        failed_store=terminal_cases.failed.validation_store,
        corruption_store=terminal_cases.adversarial.validation_store,
    )


def test_resolver_projects_exact_terminal_approval_and_revalidates_freshness(
    composition_source_cases,
):
    case = composition_source_cases
    resolver = ExpertCompositionSourceResolver(case.parent_store)
    candidate_id = case.parent.snapshot.state.candidate_id

    capability = resolver.resolve(candidate_id)
    stored = case.parent_store.reducer.candidate_store.read(candidate_id)
    manifest = stored.closure.manifest
    reference = capability.source_reference

    assert type(capability) is ApprovedExpertCompositionSource
    assert capability.stored_candidate == stored
    assert capability.approval_snapshot == case.parent.snapshot
    assert capability.publication_eligibility_result == case.parent.stage_result
    assert reference == project_expert_composition_source_reference(stored)
    assert reference.candidate_id == manifest.candidate_id
    assert reference.candidate_commit_record_id == stored.commit_record.commit_record_id
    assert reference.scope_contract_id == manifest.scope_contract_id
    assert reference.parent_release_id == manifest.parent_release_id
    assert reference.parent_repository_map_id == manifest.parent_repository_map_ref
    assert reference.parent_tree_hash == manifest.parent_tree_hash
    assert reference.candidate_tree_hash == manifest.candidate_tree_hash
    assert reference.patch_id == stored.closure.patch.patch_id
    assert reference.patch_digest == manifest.patch_digest
    assert (
        reference.proposed_repository_map_id
        == stored.closure.repository_map.repository_map_id
    )
    assert reference.module_contract_ids == tuple(
        sorted(module.module_contract_id for module in stored.closure.module_contracts)
    )
    assert capability.security_subject_ids == tuple(
        sorted(set(capability.security_subject_ids))
    )
    assert {
        reference.source_reference_id,
        case.parent.snapshot.transition.transition_id,
        case.parent.snapshot.state.validation_state_id,
        case.parent.stage_result.stage_result_record_id,
        *reference.stable_authority_ids,
        *case.parent.stage_result.exact_dependency_ids,
    }.issubset(capability.security_subject_ids)
    assert resolver.require_current(capability) is None


def test_projection_rejects_a_substituted_candidate_commit(
    composition_source_cases,
):
    stored = composition_source_cases.parent_store.reducer.candidate_store.read(
        composition_source_cases.parent.snapshot.state.candidate_id
    )
    substituted_commit = ExpertCandidateCommitRecord.mint(
        candidate_id=content_id("expert-candidate", {"substituted": True}),
        file_checksums=stored.commit_record.file_checksums,
    )

    with pytest.raises(
        ExpertCompositionSourceError,
        match="inconsistent candidate closure",
    ):
        project_expert_composition_source_reference(
            replace(stored, commit_record=substituted_commit)
        )


@pytest.mark.parametrize(
    ("case_name", "expected_state"),
    (
        ("retained", ExpertPromotionState.PARETO_RETAINED),
        ("failed", ExpertPromotionState.FAILED),
    ),
)
def test_resolver_rejects_nonapproved_terminal_history(
    composition_source_cases,
    case_name,
    expected_state,
):
    result = getattr(composition_source_cases, case_name)
    store = getattr(composition_source_cases, f"{case_name}_store")
    assert result.snapshot.state.promotion_state is expected_state

    with pytest.raises(
        ExpertCompositionSourceError,
        match="complete typed approval history",
    ):
        ExpertCompositionSourceResolver(store).resolve(
            result.snapshot.state.candidate_id
        )


def test_resolver_rejects_approved_bootstrap_as_composition_source(
    composition_source_cases,
):
    result = composition_source_cases.bootstrap
    assert result.stage_result.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED

    with pytest.raises(
        ExpertCompositionSourceError,
        match="bootstrap candidate",
    ):
        ExpertCompositionSourceResolver(
            composition_source_cases.bootstrap_store
        ).resolve(result.snapshot.state.candidate_id)


def test_capability_is_immutable_unpicklable_and_resolver_process_bound(
    composition_source_cases,
    monkeypatch,
):
    resolver = ExpertCompositionSourceResolver(composition_source_cases.parent_store)
    capability = resolver.resolve(
        composition_source_cases.parent.snapshot.state.candidate_id
    )

    with pytest.raises(ExpertCompositionSourceError, match="resolver sealed"):
        ApprovedExpertCompositionSource(
            object(),
            resolver,
            stored_candidate=capability.stored_candidate,
            approval_snapshot=capability.approval_snapshot,
            publication_eligibility_result=(capability.publication_eligibility_result),
            source_reference=capability.source_reference,
        )
    with pytest.raises(ExpertCompositionSourceError, match="immutable"):
        capability.source_reference = capability.source_reference
    with pytest.raises(ExpertCompositionSourceError, match="resolver authority"):
        resolver.candidate_store = resolver.candidate_store
    with pytest.raises(ExpertCompositionSourceError, match="cannot be serialized"):
        pickle.dumps(capability)

    foreign_resolver = ExpertCompositionSourceResolver(
        composition_source_cases.parent_store
    )
    with pytest.raises(ExpertCompositionSourceError, match="foreign"):
        foreign_resolver.require_current(capability)

    owner_process_id = composition_source_module.os.getpid()
    monkeypatch.setattr(
        composition_source_module.os,
        "getpid",
        lambda: owner_process_id + 1,
    )
    with pytest.raises(ExpertCompositionSourceError, match="foreign"):
        capability.source_reference
    with pytest.raises(ExpertCompositionSourceError, match="foreign"):
        resolver.require_current(capability)


def test_capability_does_not_expose_mutable_candidate_or_approval_state(
    composition_source_cases,
):
    resolver = ExpertCompositionSourceResolver(composition_source_cases.parent_store)
    capability = resolver.resolve(
        composition_source_cases.parent.snapshot.state.candidate_id
    )
    exposed_candidate = capability.stored_candidate
    relative_path = sorted(exposed_candidate.closure.candidate_contents)[0]
    expected_payload = exposed_candidate.closure.candidate_contents[relative_path]

    with pytest.raises(TypeError):
        exposed_candidate.closure.candidate_contents[relative_path] = b"caller mutation"
    with pytest.raises(TypeError):
        exposed_candidate.closure.operation_artifacts["caller.txt"] = b"caller mutation"

    assert (
        capability.stored_candidate.closure.candidate_contents[relative_path]
        == expected_payload
    )
    assert "caller.txt" not in capability.stored_candidate.closure.operation_artifacts
    assert capability.approval_snapshot == composition_source_cases.parent.snapshot


def test_capability_rejects_a_changed_validation_head(
    composition_source_cases,
    monkeypatch,
):
    resolver = ExpertCompositionSourceResolver(composition_source_cases.parent_store)
    capability = resolver.resolve(
        composition_source_cases.parent.snapshot.state.candidate_id
    )
    approved_state = capability.approval_snapshot.state
    revoked_state = ExpertCandidateValidationState.mint(
        validation_attempt_id=approved_state.validation_attempt_id,
        candidate_id=approved_state.candidate_id,
        candidate_tree_hash=approved_state.candidate_tree_hash,
        predecessor_state_id=approved_state.validation_state_id,
        promotion_state=ExpertPromotionState.REVOKED,
        accepted_stage_results=approved_state.accepted_stage_results,
        next_stage=None,
        review_assertion_ids=approved_state.review_assertion_ids,
        terminal_evidence_ids=approved_state.terminal_evidence_ids,
        transition_evidence_id=approved_state.transition_evidence_id,
        reason="revoked_after_composition_source_resolution",
    )
    changed_snapshot = replace(
        capability.approval_snapshot,
        state=revoked_state,
    )
    monkeypatch.setattr(
        composition_source_cases.parent_store,
        "snapshot",
        lambda _candidate_id: changed_snapshot,
    )

    with pytest.raises(
        ExpertCompositionSourceError,
        match="not terminally approved",
    ):
        resolver.require_current(capability)


def test_capability_freshness_reopens_and_rejects_candidate_corruption(
    composition_source_cases,
):
    resolver = ExpertCompositionSourceResolver(
        composition_source_cases.corruption_store
    )
    candidate_id = composition_source_cases.corruption.snapshot.state.candidate_id
    capability = resolver.resolve(candidate_id)
    relative_path = sorted(capability.stored_candidate.closure.candidate_contents)[0]
    candidate_file = capability.stored_candidate.root / "source" / relative_path
    candidate_file.write_bytes(b"corrupted after source resolution")

    with pytest.raises(ExpertCandidateStoreError, match="checksum differs"):
        resolver.require_current(capability)
