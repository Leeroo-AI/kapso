from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import pytest

import kapso.cross_run.expert as expert_facade
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    CrossRunTaskBindingSettings,
    ExpertCandidateValidationState,
    ExpertPromotionState,
    ExpertValidationAttempt,
    ExpertValidationStage,
)
from kapso.cross_run.expert.composition import ExpertCompositionReducer
from kapso.cross_run.expert.composition_admission import (
    ExpertCompositionAdmissionCoordinator,
    ExpertCompositionAdmissionError,
    build_expert_composition_plan,
    compose_expert_active_task_bindings,
)
from kapso.cross_run.expert.composition_admission_contracts import (
    ExpertCompositionAdmissionContractError,
    ExpertCompositionAdmissionFence,
    composition_admission_security_subject_ids,
    validate_expert_composition_admission_fence,
)
from kapso.cross_run.expert.composition_admission_authority import (
    ExpertCompositionAdmissionAuthorityError,
)
from kapso.cross_run.expert.composition_base_provider import (
    CurrentExpertCompositionBase,
    GitHubExpertCompositionBaseProvider,
)
from kapso.cross_run.expert.composition_base import (
    expert_composition_base_security_subject_ids,
)
from kapso.cross_run.expert.composition_candidate import (
    project_deterministic_composition_candidate,
)
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplayCurrentReleaseObservation,
)
from kapso.cross_run.expert.promotion_authority import (
    publication_eligibility_candidate_security_subject_ids,
)
from kapso.cross_run.expert.store import (
    ExpertCandidateStore,
    ExpertCandidateStoreError,
    stored_candidate_admission_dependency_ids,
)
from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertValidationError,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from security_denylist_fixtures import matched_security_revocations
from test_expert_composition import reducer_case
from test_expert_publication_eligibility import terminal_cases


class _DenylistAuthority:
    def __init__(self):
        self.calls = []
        self.denied = False
        self.after_observe = lambda: None

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.calls.append(checked_subject_ids)
        observation = SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=tree_or_blob_digest(b"scope binding"),
            snapshot_id=content_id(
                "security-denylist-snapshot",
                {"composition_admission": True},
            ),
            generation=1,
            publication_id=content_id(
                "github-publication",
                {"composition_admission": True},
            ),
            repository_full_name="Leeroo-AI/kapso-security",
            repository_node_id="security_repository_node",
            pointer_digest=tree_or_blob_digest(b"security current"),
            authority_commit_sha="a" * 40,
            release_attestation_ref="attestations/composition-admission",
            checked_subject_ids=checked_subject_ids,
            matched_revocations=matched_security_revocations(
                (checked_subject_ids[0],) if self.denied else ()
            ),
        )
        self.after_observe()
        return observation


class _CurrentReleaseProvider:
    def __init__(self, scope_id, release_id):
        self.scope_id = scope_id
        self.release_id = release_id

    def current_release_id(self, scope_id):
        assert scope_id == self.scope_id
        return self.release_id


def test_composition_admission_facade_exports_are_explicit():
    expected = {
        "ExpertCompositionAdmissionContractError",
        "ExpertCompositionAdmissionCoordinator",
        "ExpertCompositionAdmissionDenylistAuthority",
        "ExpertCompositionAdmissionError",
        "ExpertCompositionAdmissionFence",
        "ExpertCompositionSourceAdmissionAuthority",
        "build_expert_composition_plan",
        "compose_expert_active_task_bindings",
        "composition_admission_security_subject_ids",
        "stored_candidate_admission_dependency_ids",
        "validate_expert_composition_admission_fence",
    }

    assert expected.issubset(expert_facade.__all__)
    assert all(hasattr(expert_facade, name) for name in expected)


def test_active_task_binding_union_preserves_every_source():
    first = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="post_training",
        task_adapter_id="posttrainbench",
    )
    second = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="tabular_prediction",
        task_adapter_id="relbench",
    )

    assert compose_expert_active_task_bindings(((second,), (first, second))) == (
        first,
        second,
    )


def _current_base_capability(case, provider):
    base = case.parent_base
    observation = SourceReplayCurrentReleaseObservation.mint(
        scope_id=base.scope_contract.scope_id,
        release_id=base.release_manifest.release_id,
        publication_id=content_id(
            "github-publication",
            {"release": base.release_manifest.release_id},
        ),
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repository_node",
        current_pointer_digest=tree_or_blob_digest(b"expert current"),
        current_pointer_commit_sha="b" * 40,
        validation_closure_ids=(
            content_id(
                "expert-validation-closure",
                {"release": base.release_manifest.release_id},
            ),
        ),
    )
    receipt = base.source_base_tree_receipt
    security_subject_ids = tuple(
        sorted(
            {
                base.reference.base_reference_id,
                *base.reference.stable_authority_ids,
                receipt.source_base_tree_receipt_id,
                receipt.source_extraction_receipt.extraction_receipt_id,
                observation.observation_id,
                observation.publication_id,
                *observation.validation_closure_ids,
                *base.release_manifest.consumed_dependency_ids,
            }
        )
    )
    capability = object.__new__(CurrentExpertCompositionBase)
    object.__setattr__(capability, "_provider", provider)
    object.__setattr__(capability, "_owner_process_id", os.getpid())
    object.__setattr__(capability, "_closure", base)
    object.__setattr__(capability, "_current_observation", observation)
    object.__setattr__(capability, "_resolved_current", object())
    object.__setattr__(capability, "_security_subject_ids", security_subject_ids)
    return capability, observation


def _base_provider(case, monkeypatch):
    settings = case.source._resolver.candidate_store.validator.settings
    provider = object.__new__(GitHubExpertCompositionBaseProvider)
    object.__setattr__(provider, "_resolver", object())
    object.__setattr__(provider, "_materializer", object())
    object.__setattr__(provider, "_settings", settings)
    capability, observation = _current_base_capability(case, provider)
    calls = []
    state = {"observation": observation}

    def resolve_current(bound, scope_contract):
        assert bound is provider
        assert scope_contract == case.parent_base.scope_contract
        calls.append("resolve")
        return capability

    def require_current(bound, current):
        assert bound is provider
        assert current is capability
        calls.append("current")
        return state["observation"]

    monkeypatch.setattr(
        GitHubExpertCompositionBaseProvider,
        "resolve_current",
        resolve_current,
    )
    monkeypatch.setattr(
        GitHubExpertCompositionBaseProvider,
        "require_current",
        require_current,
    )
    return provider, calls, state


def test_clean_composition_admission_is_fenced_atomic_and_reopenable(
    reducer_case,
    tmp_path,
    monkeypatch,
):
    case = reducer_case
    candidate_store = case.source._resolver.candidate_store
    base_provider, current_calls, current_state = _base_provider(case, monkeypatch)
    denylist = _DenylistAuthority()
    coordinator = ExpertCompositionAdmissionCoordinator(
        candidate_store=candidate_store,
        base_provider=base_provider,
        source_resolver=case.source._resolver,
        task_adapter_authority=(
            case.source._resolver.validation_store.reducer.task_adapter_provider
        ),
        security_denylist_authority=denylist,
    )

    stored = coordinator.compose_and_persist(
        scope_contract=case.parent_base.scope_contract,
        source_candidate_ids=(case.source.source_reference.candidate_id,),
    )
    reopened = candidate_store.read(stored.closure.manifest.candidate_id)
    fence = reopened.composition_admission_fence

    assert fence is not None
    assert reopened == stored
    assert current_calls == ["resolve", "current", "current"]
    assert len(denylist.calls) == 1
    assert fence.security_subject_ids == denylist.calls[0]
    assert fence.candidate_commit_record_id == stored.commit_record.commit_record_id
    assert "ADMISSION.json" not in stored.commit_record.file_checksums
    assert (stored.root / "ADMISSION.json").read_bytes() == fence.to_json_bytes()
    validate_expert_composition_admission_fence(
        fence=fence,
        closure=stored.closure,
        commit_record=stored.commit_record,
    )
    publication_subjects = set(
        publication_eligibility_candidate_security_subject_ids(stored)
    )
    assert {
        fence.admission_fence_id,
        fence.security_denylist_observation.observation_id,
        *fence.security_subject_ids,
    }.issubset(publication_subjects)
    reducer = case.source._resolver.validation_store.reducer
    current_release_provider = _CurrentReleaseProvider(
        case.parent_base.scope_contract.scope_id,
        case.parent_base.release_manifest.release_id,
    )
    eligibility = ExpertCandidateEligibilityEvaluator(
        reducer.settings,
        candidate_store,
        reducer.task_adapter_provider,
        current_release_provider,
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    assert set(stored_candidate_admission_dependency_ids(stored)).issubset(
        eligibility.decision.exact_dependency_ids
    )
    attempt = ExpertValidationAttempt.mint(
        candidate_id=eligibility.decision.candidate_id,
        candidate_tree_hash=eligibility.decision.candidate_tree_hash,
        candidate_commit_record_id=(eligibility.decision.candidate_commit_record_id),
        scope_contract_id=eligibility.decision.scope_contract_id,
        source_base_release_id=eligibility.decision.source_base_release_id,
        expected_current_release_id=(eligibility.decision.expected_current_release_id),
        recovery_plan_id=eligibility.decision.recovery_plan_id,
        eligibility_decision_id=eligibility.decision.eligibility_decision_id,
        validation_policy_id=eligibility.decision.validation_policy_id,
        configuration_fingerprint=eligibility.decision.configuration_fingerprint,
        validation_track=eligibility.decision.validation_track,
        attempt_number=1,
        predecessor_attempt_id=None,
        required_stages=(ExpertValidationStage.CONTRACT_SCHEMA,),
        configured_task_family_ids=(eligibility.decision.configured_task_family_ids),
        task_adapter_pins=eligibility.decision.task_adapter_pins,
        source_replay_selection=None,
        control_dependency_ids=eligibility.decision.control_dependency_ids,
        eligibility_dependency_ids=tuple(
            sorted(
                {
                    eligibility.decision.eligibility_decision_id,
                    *eligibility.decision.exact_dependency_ids,
                }
            )
        ),
    )
    state = ExpertCandidateValidationState.mint(
        validation_attempt_id=attempt.validation_attempt_id,
        candidate_id=attempt.candidate_id,
        candidate_tree_hash=attempt.candidate_tree_hash,
        predecessor_state_id=None,
        promotion_state=ExpertPromotionState.VALIDATING,
        accepted_stage_results=(),
        next_stage=ExpertValidationStage.CONTRACT_SCHEMA,
        review_assertion_ids=(),
        terminal_evidence_ids=(),
        transition_evidence_id=eligibility.decision.eligibility_decision_id,
        reason="validation_attempt_started",
    )
    original_observation = fence.current_release_observation
    refreshed_observation = SourceReplayCurrentReleaseObservation.mint(
        scope_id=original_observation.scope_id,
        release_id=original_observation.release_id,
        publication_id=original_observation.publication_id,
        repository_full_name=original_observation.repository_full_name,
        repository_node_id=original_observation.repository_node_id,
        current_pointer_digest=original_observation.current_pointer_digest,
        current_pointer_commit_sha="f" * 40,
        validation_closure_ids=original_observation.validation_closure_ids,
    )
    substituted_base_subjects = expert_composition_base_security_subject_ids(
        case.parent_base,
        refreshed_observation,
    )
    substituted_subjects = composition_admission_security_subject_ids(
        closure=stored.closure,
        commit_record=stored.commit_record,
        base_security_subject_ids=substituted_base_subjects,
        source_authorities=fence.source_authorities,
        current_release_observation=refreshed_observation,
        task_adapter_trust_observations=fence.task_adapter_trust_observations,
    )
    substituted_denylist = _DenylistAuthority().observe_exact(
        scope_id=fence.scope_id,
        scope_contract_id=fence.scope_contract_id,
        checked_subject_ids=substituted_subjects,
    )
    substituted_valid_fence = ExpertCompositionAdmissionFence.mint(
        candidate_id=fence.candidate_id,
        candidate_commit_record_id=fence.candidate_commit_record_id,
        candidate_tree_hash=fence.candidate_tree_hash,
        scope_id=fence.scope_id,
        scope_contract_id=fence.scope_contract_id,
        expected_current_release_id=fence.expected_current_release_id,
        composition_plan_id=fence.composition_plan_id,
        composition_materialization_id=fence.composition_materialization_id,
        base_reference_id=fence.base_reference_id,
        base_security_subject_ids=substituted_base_subjects,
        source_authorities=fence.source_authorities,
        current_release_observation=refreshed_observation,
        task_adapter_trust_observations=fence.task_adapter_trust_observations,
        security_denylist_observation=substituted_denylist,
    )
    validate_expert_composition_admission_fence(
        fence=substituted_valid_fence,
        closure=stored.closure,
        commit_record=stored.commit_record,
    )
    admission_path = stored.root / "ADMISSION.json"
    admission_path.write_bytes(substituted_valid_fence.to_json_bytes())
    with pytest.raises(
        ExpertValidationError,
        match="active attempt differs from its immutable candidate closure",
    ):
        reducer.invalidate_current_release_authority(
            state=state,
            attempt=attempt,
        )
    admission_path.write_bytes(fence.to_json_bytes())
    with pytest.raises(
        ExpertCandidateStoreError,
        match="exact composition admission authority",
    ):
        candidate_store._bind_composition_admission_authority(object())
    with pytest.raises(
        ExpertCandidateStoreError,
        match="foreign authority",
    ):
        candidate_store._commit_composition_admission(
            authority=coordinator._authority,
            admission=object(),
        )
    freshness = SimpleNamespace(approved_sources=(case.source,))
    with case.source._resolver.lease_current_approvals((case.source,)) as expired_lease:
        expired = candidate_store._seal_composition_admission(
            authority=coordinator._authority,
            approval_lease=expired_lease,
            closure=stored.closure,
            freshness_context=freshness,
        )
    with pytest.raises(
        ExpertCompositionAdmissionAuthorityError,
        match="inactive or foreign",
    ):
        candidate_store._commit_composition_admission(
            authority=coordinator._authority,
            admission=expired,
        )
    with case.source._resolver.lease_current_approvals(
        (case.source,)
    ) as approval_lease:
        sealed = candidate_store._seal_composition_admission(
            authority=coordinator._authority,
            approval_lease=approval_lease,
            closure=stored.closure,
            freshness_context=freshness,
        )
    with pytest.raises(
        ExpertCandidateStoreError,
        match="foreign authority",
    ):
        candidate_store._commit_composition_admission(
            authority=object(),
            admission=sealed,
        )

    current_state["observation"] = refreshed_observation
    with pytest.raises(
        ExpertCandidateStoreError,
        match="identity conflicts",
    ):
        coordinator.compose_and_persist(
            scope_contract=case.parent_base.scope_contract,
            source_candidate_ids=(case.source.source_reference.candidate_id,),
        )
    current_state["observation"] = original_observation
    moved_observation = SourceReplayCurrentReleaseObservation.mint(
        scope_id=original_observation.scope_id,
        release_id=original_observation.release_id,
        publication_id=original_observation.publication_id,
        repository_full_name=original_observation.repository_full_name,
        repository_node_id=original_observation.repository_node_id,
        current_pointer_digest=tree_or_blob_digest(b"moved expert current"),
        current_pointer_commit_sha="c" * 40,
        validation_closure_ids=original_observation.validation_closure_ids,
    )

    def move_current_after_denylist():
        current_state["observation"] = moved_observation

    denylist.after_observe = move_current_after_denylist
    with pytest.raises(
        ExpertCompositionAdmissionError,
        match="CURRENT changed",
    ):
        coordinator.compose_and_persist(
            scope_contract=case.parent_base.scope_contract,
            source_candidate_ids=(case.source.source_reference.candidate_id,),
        )
    current_state["observation"] = original_observation
    denylist.after_observe = lambda: None
    denylist.denied = True
    with pytest.raises(
        ExpertCompositionAdmissionError,
        match="denylist rejected",
    ):
        coordinator.compose_and_persist(
            scope_contract=case.parent_base.scope_contract,
            source_candidate_ids=(case.source.source_reference.candidate_id,),
        )
    assert candidate_store.read(stored.closure.manifest.candidate_id) == stored

    isolated_root = tmp_path / "isolated-candidates"
    isolated_store = ExpertCandidateStore(
        isolated_root,
        tmp_path,
        candidate_store.validator,
    )
    copied = isolated_store.object_root / stored.root.name
    shutil.copytree(stored.root, copied)
    admission_path = copied / "ADMISSION.json"
    admission_path.write_bytes(admission_path.read_bytes() + b"\n")
    with pytest.raises(
        ExpertCandidateStoreError,
        match="admission fence is not canonical",
    ):
        isolated_store.read(stored.closure.manifest.candidate_id)

    substituted_store = ExpertCandidateStore(
        tmp_path / "substituted-admission-candidates",
        tmp_path,
        candidate_store.validator,
    )
    substituted_copy = substituted_store.object_root / stored.root.name
    shutil.copytree(stored.root, substituted_copy)
    substituted_fence = ExpertCompositionAdmissionFence.mint(
        candidate_id=fence.candidate_id,
        candidate_commit_record_id=fence.candidate_commit_record_id,
        candidate_tree_hash=fence.candidate_tree_hash,
        scope_id=fence.scope_id,
        scope_contract_id=fence.scope_contract_id,
        expected_current_release_id=fence.expected_current_release_id,
        composition_plan_id=fence.composition_plan_id,
        composition_materialization_id=fence.composition_materialization_id,
        base_reference_id=fence.base_reference_id,
        base_security_subject_ids=tuple(
            sorted(
                {
                    *fence.base_security_subject_ids,
                    content_id("foreign-base-authority", {"substitution": True}),
                }
            )
        ),
        source_authorities=fence.source_authorities,
        current_release_observation=fence.current_release_observation,
        task_adapter_trust_observations=fence.task_adapter_trust_observations,
        security_denylist_observation=fence.security_denylist_observation,
    )
    (substituted_copy / "ADMISSION.json").write_bytes(substituted_fence.to_json_bytes())
    with pytest.raises(
        ExpertCompositionAdmissionContractError,
        match="base security closure is not exact",
    ):
        substituted_store.read(stored.closure.manifest.candidate_id)

    missing_store = ExpertCandidateStore(
        tmp_path / "missing-admission-candidates",
        tmp_path,
        candidate_store.validator,
    )
    missing_copy = missing_store.object_root / stored.root.name
    shutil.copytree(stored.root, missing_copy)
    (missing_copy / "ADMISSION.json").unlink()
    with pytest.raises(
        ExpertCandidateStoreError,
        match="lacks its admission fence",
    ):
        missing_store.read(stored.closure.manifest.candidate_id)


def test_composition_store_still_rejects_unfenced_generic_persistence(reducer_case):
    case = reducer_case
    source = case.source
    settings = source._resolver.candidate_store.validator.settings
    provider = object.__new__(GitHubExpertCompositionBaseProvider)
    object.__setattr__(provider, "_settings", settings)
    current_base, _ = _current_base_capability(case, provider)

    plan = build_expert_composition_plan(
        current_base=current_base,
        approved_sources=(source,),
        composition_policy_version=settings.composition_policy_version,
        composition_source_limit=settings.composition_source_limit,
        candidate_entry_limit=settings.candidate_entry_limit,
        candidate_byte_limit=settings.candidate_byte_limit,
    )
    reduction = ExpertCompositionReducer(
        candidate_entry_limit=settings.candidate_entry_limit,
        candidate_byte_limit=settings.candidate_byte_limit,
    ).reduce(
        plan=plan,
        current_base=case.parent_base,
        sources=(source.reduction_source,),
    )
    closure = project_deterministic_composition_candidate(
        reduction=reduction,
        current_base=case.parent_base,
        approved_sources=(source,),
        sanitizer=source._resolver.candidate_store.validator.sanitizer,
    )
    with pytest.raises(
        ExpertCandidateStoreError,
        match="sealed admission authority",
    ):
        source._resolver.candidate_store.persist(closure)
