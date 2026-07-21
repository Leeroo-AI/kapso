"""Policy-derived expert source replay selections."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertModuleContract,
    KnowledgeClaim,
    MissingReferenceError,
)
from kapso.cross_run.expert.replay import _derive_expert_source_replay_selection
from kapso.cross_run.expert.store import (
    ExpertCandidateCommitRecord,
    StoredExpertCandidate,
)
from kapso.cross_run.expert.triggers import (
    ExpertTriggerEvaluator,
    ExpertTriggerObservation,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.expert.validation import ExpertCandidateEligibilityEvaluator
from kapso.cross_run.settings import CrossRunSettings
from test_cross_run_retrieval import (
    analogical_context,
    outcome_episodes,
    source_fixture,
)
from test_expert_triggers import (
    clone_episode,
    configuration_fingerprint,
    expert_records,
    inspection_operation,
    supported_claim,
    trigger_packet,
    trigger_settings,
    verified_parent_tree_hash,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _candidate_id(label: str) -> str:
    return content_id("expert-candidate", {"label": label})


def _validation_policy():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).expert.validation


def _derive(
    candidate_id,
    packet,
    decision,
    candidate_modules,
    *,
    validation=None,
):
    selected_validation = validation or _validation_policy()
    stored_candidate = _stored_candidate(
        candidate_id,
        packet,
        decision,
        candidate_modules,
    )
    return _derive_expert_source_replay_selection(
        stored_candidate=stored_candidate,
        settings=selected_validation,
    )


def _stored_candidate(candidate_id, packet, decision, candidate_modules):
    commit_record = ExpertCandidateCommitRecord.mint(
        candidate_id=candidate_id,
        file_checksums={"candidate.json": tree_or_blob_digest(b"candidate")},
    )
    manifest = SimpleNamespace(
        candidate_id=candidate_id,
        candidate_tree_hash=tree_or_blob_digest(b"candidate-tree"),
        configuration_fingerprint=packet.configuration_fingerprint,
        trigger_evidence_packet_id=packet.evidence_packet_id,
        trigger_decision_id=decision.trigger_decision_id,
        module_contract_refs=tuple(
            sorted(module.module_contract_id for module in candidate_modules)
        ),
        scope_contract_id=packet.scope_contract.scope_contract_id,
        parent_release_id=packet.parent_release_id,
        change_kind=decision.change_kind,
        sanitation_report_id=content_id(
            "expert-candidate-sanitation",
            {"candidate_id": candidate_id},
        ),
    )
    return StoredExpertCandidate(
        root=Path("/test-candidate"),
        closure=SimpleNamespace(
            manifest=manifest,
            trigger_packet=packet,
            trigger_decision=decision,
            module_contracts=candidate_modules,
        ),
        commit_record=commit_record,
    )


class _CandidateReader:
    def __init__(self, stored_candidate):
        self.stored_candidate = stored_candidate

    def read(self, candidate_id):
        assert candidate_id == self.stored_candidate.closure.manifest.candidate_id
        return self.stored_candidate


class _AdapterProvider:
    def __init__(self, packet):
        binding = packet.active_task_bindings[0]
        manifest = SimpleNamespace(
            scope_contract_id=packet.scope_contract.scope_contract_id,
            task_family_id=binding.task_family_id,
            task_adapter_id=binding.task_adapter_id,
            task_adapter_manifest_id=content_id(
                "task-adapter-manifest",
                {"adapter": binding.task_adapter_id},
            ),
        )
        receipt = SimpleNamespace(
            verification_receipt_id=content_id(
                "task-adapter-verification",
                {"adapter": binding.task_adapter_id},
            )
        )
        self.adapter = SimpleNamespace(
            manifest=manifest,
            verification_receipt=receipt,
            dependency_ids=tuple(
                sorted(
                    (
                        manifest.task_adapter_manifest_id,
                        receipt.verification_receipt_id,
                    )
                )
            ),
        )

    def resolve_active(self, **_binding):
        return self.adapter

    def resolve_exact(self, **_pin):
        return self.adapter


class _CurrentReleaseProvider:
    def __init__(self, release_id):
        self.release_id = release_id

    def current_release_id(self, scope_id):
        assert scope_id == "ml_ai"
        return self.release_id


def _changed_module(
    module: ExpertModuleContract,
    *,
    supporting_episode_ids: tuple[str, ...] = (),
    known_failure_episode_ids: tuple[str, ...] = (),
) -> ExpertModuleContract:
    values = module.to_dict()
    values.pop("module_contract_id")
    values.update(
        {
            "version": f"v{int(module.version.removeprefix('v')) + 1}",
            "supporting_episode_ids": supporting_episode_ids,
            "known_failure_episode_ids": known_failure_episode_ids,
        }
    )
    return ExpertModuleContract.mint(**values)


def _observation(
    *,
    settings,
    kind,
    module_id,
    exact_evidence_ids,
    affected_paths=(),
):
    description = f"Canonical inspected {kind.value.replace('_', ' ')}."
    payload = {
        "affected_capability_ids": [module_id],
        "affected_paths": list(affected_paths),
        "configuration_fingerprint": configuration_fingerprint(settings),
        "description": description,
        "difficulty_evidence_signatures": {},
        "difficulty_signature": None,
        "exact_evidence_ids": list(exact_evidence_ids),
        "independent_lineage_ids": [],
        "inspection_policy_version": settings.inspection_policy_version,
        "kind": kind.value,
        "occurrence_count": 1,
        "parent_tree_hash": verified_parent_tree_hash(),
        "task_context_binding_ids": [],
    }
    final_output = json.dumps(payload, indent=2) + "\n"
    return ExpertTriggerObservation.mint(
        kind=kind,
        parent_tree_hash=verified_parent_tree_hash(),
        inspection_policy_version=settings.inspection_policy_version,
        configuration_fingerprint=configuration_fingerprint(settings),
        inspection_operation=inspection_operation(settings, final_output),
        inspection_final_output=final_output,
        difficulty_signature=None,
        difficulty_evidence_signatures={},
        description=description,
        affected_capability_ids=(module_id,),
        affected_paths=affected_paths,
        exact_evidence_ids=exact_evidence_ids,
        independent_lineage_ids=(),
        task_context_binding_ids=(),
        occurrence_count=1,
    )


def test_repeated_success_selects_causal_episodes_in_their_shared_exact_bundle():
    settings = trigger_settings()
    _, context, template, _, claim, _, _ = source_fixture()
    positive = outcome_episodes(template)[0]
    first = clone_episode(
        positive,
        name="replay-first",
        campaign_id="campaign-first",
        context=context,
    )
    second = clone_episode(
        positive,
        name="replay-second",
        campaign_id="campaign-second",
        context=analogical_context(context),
    )
    claim = supported_claim(claim, (first, second))
    packet = trigger_packet(
        settings=settings,
        episodes=(first, second),
        claims=(claim,),
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    validation = _validation_policy()
    result = _derive(
        _candidate_id("repeated-success"),
        packet,
        decision,
        packet.module_contracts,
        validation=validation,
    )

    assert result.reason_code == "selected"
    assert result.selection is not None
    assert result.selection.causal_episode_ids == tuple(
        sorted((first.episode_id, second.episode_id))
    )
    assert result.selection.coverage_episode_ids == ()
    assert tuple(case.source_bundle_id for case in result.selection.cases) == (
        first.source_bundle_id,
    )
    assert result.selection.cases[0].episode_ids == tuple(
        sorted((first.episode_id, second.episode_id))
    )
    assert claim.revision_id in result.selection.selection_evidence_ids
    assert result.selection.validation_policy_id == (
        validation.policy.validation_policy().validation_policy_id
    )

    altered_validation = replace(
        validation,
        policy=replace(
            validation.policy,
            source_replay_episode_limit=(
                validation.policy.source_replay_episode_limit + 1
            ),
        ),
    )
    altered = _derive(
        _candidate_id("repeated-success"),
        packet,
        decision,
        packet.module_contracts,
        validation=altered_validation,
    )
    assert altered.selection is not None
    assert altered.selection.source_replay_selection_id != (
        result.selection.source_replay_selection_id
    )
    with pytest.raises(MissingReferenceError, match="not exact"):
        replace(
            result.selection,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        *result.selection.exact_dependency_ids,
                        content_id("unrelated-proof", {"label": "forged"}),
                    }
                )
            ),
        )


def test_eligibility_enrolls_and_replays_the_exact_source_selection():
    trigger_policy = trigger_settings()
    _, context, template, _, claim, _, _ = source_fixture()
    positive = outcome_episodes(template)[0]
    first = clone_episode(
        positive,
        name="enrollment-first",
        campaign_id="enrollment-first-campaign",
        context=context,
    )
    second = clone_episode(
        positive,
        name="enrollment-second",
        campaign_id="enrollment-second-campaign",
        context=analogical_context(context),
    )
    claim = supported_claim(claim, (first, second))
    packet = trigger_packet(
        settings=trigger_policy,
        episodes=(first, second),
        claims=(claim,),
    )
    decision = ExpertTriggerEvaluator(trigger_policy).evaluate(packet)
    stored = _stored_candidate(
        _candidate_id("eligibility"),
        packet,
        decision,
        packet.module_contracts,
    )
    validation = _validation_policy()
    adapter_provider = _AdapterProvider(packet)
    unavailable = ExpertCandidateEligibilityEvaluator(
        validation,
        _CandidateReader(stored),
        adapter_provider,
        _CurrentReleaseProvider(packet.parent_release_id),
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    assert unavailable.decision.eligible is False
    assert unavailable.decision.reason_code == (
        "required_validation_infrastructure_unavailable"
    )
    assert unavailable.decision.source_replay_selection is None
    validation = replace(
        validation,
        policy=replace(
            validation.policy,
            sealed_canary_trust_root="test_sealed_canary_root",
        ),
    )
    evaluator = ExpertCandidateEligibilityEvaluator(
        validation,
        _CandidateReader(stored),
        adapter_provider,
        _CurrentReleaseProvider(packet.parent_release_id),
    )

    enrollment = evaluator.decide(candidate_id=stored.closure.manifest.candidate_id)
    replayed = evaluator.replay(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_pins=enrollment.decision.task_adapter_pins,
    )

    assert enrollment == replayed
    assert enrollment.decision.eligible is True
    assert enrollment.decision.source_replay_selection is not None
    assert (
        enrollment.decision.source_replay_selection.source_replay_selection_id
        in enrollment.decision.exact_dependency_ids
    )


def test_contradiction_claim_traversal_selects_only_contradicting_episodes():
    settings = trigger_settings()
    _, context, template, _, claim_template, _, _ = source_fixture()
    positive, _, negative, *_ = outcome_episodes(template)
    supporting = clone_episode(
        positive,
        name="supporting",
        campaign_id="supporting-campaign",
        context=context,
    )
    contradicting = clone_episode(
        negative,
        name="contradicting",
        campaign_id="contradicting-campaign",
        context=analogical_context(context),
    )
    claim = KnowledgeClaim.mint(
        claim_id=claim_template.claim_id,
        scope_contract_id=claim_template.scope_contract_id,
        statement=claim_template.statement,
        mechanism=claim_template.mechanism,
        applicability_predicates=claim_template.applicability_predicates,
        explicit_exclusions=claim_template.explicit_exclusions,
        supporting_episode_ids=(supporting.episode_id,),
        contradicting_episode_ids=(contradicting.episode_id,),
        proposal_provenance=claim_template.proposal_provenance,
        supersedes_revision_ids=(),
    )
    _, module, _, _ = expert_records()
    observation = _observation(
        settings=settings,
        kind=ExpertTriggerObservationKind.RELEASED_CAPABILITY_CONTRADICTION,
        module_id=module.module_id,
        exact_evidence_ids=tuple(sorted((claim.revision_id, supporting.episode_id))),
    )
    packet = trigger_packet(
        settings=settings,
        episodes=(supporting, contradicting),
        claims=(claim,),
        observations=(observation,),
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    result = _derive(
        _candidate_id("contradiction"),
        packet,
        decision,
        packet.module_contracts,
    )

    assert result.selection is not None
    assert result.selection.causal_episode_ids == (contradicting.episode_id,)
    assert supporting.episode_id not in result.selection.exact_dependency_ids


def test_changed_module_evidence_supplies_explicit_noncausal_replay_coverage():
    settings = trigger_settings()
    _, _, template, _, _, _, _ = source_fixture()
    episode = outcome_episodes(template)[0]
    packet_without_observation = trigger_packet(settings=settings, episodes=(episode,))
    module = packet_without_observation.module_contracts[0]
    observation = _observation(
        settings=settings,
        kind=ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        module_id=module.module_id,
        exact_evidence_ids=(
            packet_without_observation.repository_map.repository_map_id,
        ),
        affected_paths=("src/reproducible_execution/__init__.py",),
    )
    packet = trigger_packet(
        settings=settings,
        episodes=(episode,),
        observations=(observation,),
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    changed = _changed_module(
        packet.module_contracts[0],
        supporting_episode_ids=(episode.episode_id,),
    )

    result = _derive(
        _candidate_id("module-coverage"),
        packet,
        decision,
        (changed,),
    )

    assert result.selection is not None
    assert result.selection.causal_episode_ids == ()
    assert result.selection.coverage_episode_ids == (episode.episode_id,)
    assert result.selection.cases[0].episode_reason_codes[episode.episode_id] == (
        "changed_module_support",
    )
    assert changed.module_contract_id in result.selection.selection_evidence_ids


def test_noncausal_trigger_without_explicit_module_coverage_is_unavailable():
    settings = trigger_settings()
    packet_without_observation = trigger_packet(settings=settings)
    module = packet_without_observation.module_contracts[0]
    observation = _observation(
        settings=settings,
        kind=ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        module_id=module.module_id,
        exact_evidence_ids=(
            packet_without_observation.repository_map.repository_map_id,
        ),
        affected_paths=("src/reproducible_execution/__init__.py",),
    )
    packet = trigger_packet(settings=settings, observations=(observation,))
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    result = _derive(
        _candidate_id("no-replay-evidence"),
        packet,
        decision,
        packet.module_contracts,
    )

    assert result.selection is None
    assert result.reason_code == "source_replay_evidence_unavailable"

    stored = _stored_candidate(
        _candidate_id("no-replay-evidence"),
        packet,
        decision,
        packet.module_contracts,
    )
    enrollment = ExpertCandidateEligibilityEvaluator(
        _validation_policy(),
        _CandidateReader(stored),
        _AdapterProvider(packet),
        _CurrentReleaseProvider(packet.parent_release_id),
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    assert enrollment.decision.eligible is False
    assert enrollment.decision.required_stages == ()
    assert enrollment.decision.source_replay_selection is None
    assert enrollment.decision.reason_code == "source_replay_evidence_unavailable"


def test_replay_selection_limits_fail_enrollment_without_clipping_evidence():
    settings = trigger_settings()
    _, context, template, _, claim, _, _ = source_fixture()
    positive = outcome_episodes(template)[0]
    first = clone_episode(
        positive,
        name="limited-first",
        campaign_id="limited-first-campaign",
        context=context,
    )
    second = clone_episode(
        positive,
        name="limited-second",
        campaign_id="limited-second-campaign",
        context=analogical_context(context),
    )
    claim = supported_claim(claim, (first, second))
    packet = trigger_packet(
        settings=settings,
        episodes=(first, second),
        claims=(claim,),
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    validation = _validation_policy()
    bounded_validation = replace(
        validation,
        policy=replace(
            validation.policy,
            source_replay_episode_limit=1,
            sealed_canary_trust_root="test_sealed_canary_root",
        ),
    )
    result = _derive(
        _candidate_id("bounded"),
        packet,
        decision,
        packet.module_contracts,
        validation=bounded_validation,
    )

    assert result.selection is None
    assert result.reason_code == "source_replay_selection_limit_exceeded"

    stored = _stored_candidate(
        _candidate_id("bounded"),
        packet,
        decision,
        packet.module_contracts,
    )
    enrollment = ExpertCandidateEligibilityEvaluator(
        bounded_validation,
        _CandidateReader(stored),
        _AdapterProvider(packet),
        _CurrentReleaseProvider(packet.parent_release_id),
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    assert enrollment.decision.eligible is False
    assert enrollment.decision.source_replay_selection is None
    assert enrollment.decision.reason_code == ("source_replay_selection_limit_exceeded")
