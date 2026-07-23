"""Policy-derived expert source replay selections."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    ExpertCandidateCommitRecord,
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
    ExpertModuleContract,
    ExpertSourceTreeManifest,
    ExpertValidationTrack,
    KnowledgeClaim,
    MissingReferenceError,
    SourceFileDescriptor,
    TaskAdapterManifest,
)
from kapso.cross_run.expert.candidate_context import (
    project_agent_candidate_validation_context,
)
from kapso.cross_run.expert.replay import _derive_expert_source_replay_selection
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.triggers import (
    ExpertTriggerEvaluator,
    ExpertTriggerObservation,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.expert.validation import ExpertCandidateEligibilityEvaluator
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.task_adapters import VerifiedTaskAdapter
from test_cross_run_retrieval import (
    analogical_context,
    outcome_episodes,
    source_fixture,
)
from test_cross_run_contracts import build_records, verified_test_task_adapter
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
TEST_ORIGIN_PRINCIPAL_ID = "test_expert_generalizer"


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
    candidate_content = b"verified candidate source"
    candidate_file = SourceFileDescriptor(
        relative_path="src/expert.py",
        digest=tree_or_blob_digest(candidate_content),
        mode="100644",
        size=len(candidate_content),
    )
    candidate_tree = ExpertSourceTreeManifest.mint(
        tree_hash=source_tree_digest(
            {
                candidate_file.relative_path: (
                    candidate_file.digest,
                    candidate_file.mode,
                    candidate_file.size,
                )
            }
        ),
        files=(candidate_file,),
    )
    validation_context = project_agent_candidate_validation_context(
        packet=packet,
        decision=decision,
    )
    manifest = ExpertCandidateManifest.mint(
        candidate_tree_hash=candidate_tree.tree_hash,
        candidate_tree_ref=candidate_tree.source_tree_manifest_id,
        configuration_fingerprint=packet.configuration_fingerprint,
        derivation_kind=ExpertCandidateDerivationKind.AGENT_PROPOSAL,
        derivation_ref=content_id(
            "expert-agent-proposal-derivation",
            {"seed": candidate_id},
        ),
        validation_context_ref=validation_context.validation_context_id,
        module_contract_refs=tuple(
            sorted(module.module_contract_id for module in candidate_modules)
        ),
        scope_contract_id=packet.scope_contract.scope_contract_id,
        parent_release_id=packet.parent_release_id,
        parent_repository_map_ref=packet.repository_map.repository_map_id,
        parent_tree_hash=packet.parent_tree_hash,
        change_kind=decision.change_kind,
        patch_ref=content_id("expert-candidate-patch", {"seed": candidate_id}),
        patch_digest=tree_or_blob_digest(f"patch:{candidate_id}".encode()),
        proposed_repository_map_ref=packet.repository_map.repository_map_id,
        semantic_book_digest=tree_or_blob_digest(b"candidate-semantic-book"),
        source_dependency_ids=tuple(
            sorted((packet.evidence_packet_id, decision.trigger_decision_id))
        ),
        ancestor_candidate_ids=(),
        capability_lineage=(),
        sanitation_report_id=content_id(
            "expert-candidate-sanitation",
            {"seed": candidate_id},
        ),
    )
    commit_record = ExpertCandidateCommitRecord.mint(
        candidate_id=manifest.candidate_id,
        file_checksums={"candidate.json": tree_or_blob_digest(b"candidate")},
    )
    return StoredExpertCandidate(
        root=Path("/test-candidate"),
        closure=SimpleNamespace(
            manifest=manifest,
            validation_context=validation_context,
            validation_track=(
                ExpertValidationTrack.REPOSITORY_ARCHITECTURE
                if decision.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE
                else (
                    ExpertValidationTrack.MECHANICAL_GENERAL_FIX
                    if decision.reason_code == "mechanically_general_fix"
                    else ExpertValidationTrack.BEHAVIORAL_CAPABILITY
                )
            ),
            module_contracts=candidate_modules,
            candidate_tree=candidate_tree,
            candidate_contents={candidate_file.relative_path: candidate_content},
            parent_files=(
                ()
                if packet.parent_tree_receipt is None
                else packet.parent_tree_receipt.source_extraction_receipt.source_tree_files
            ),
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
    def __init__(self, packet, *, source_adapter=None, rotate_active=False):
        binding = packet.active_task_bindings[0]
        source_manifest_ids = {
            episode.artifact_environment.task_adapter_manifest_id
            for episode in packet.episodes
        }
        source_receipt_ids = {
            episode.artifact_environment.task_adapter_verification_receipt_id
            for episode in packet.episodes
        }
        assert len(source_manifest_ids) <= 1
        assert len(source_receipt_ids) <= 1
        if source_adapter is None:
            source_manifest = next(
                record
                for record in build_records()
                if isinstance(record, TaskAdapterManifest)
                and record.scope_contract_id == packet.scope_contract.scope_contract_id
                and record.task_family_id == binding.task_family_id
                and record.task_adapter_id == binding.task_adapter_id
            )
            source_adapter = verified_test_task_adapter(source_manifest)
        elif not isinstance(source_adapter, VerifiedTaskAdapter):
            raise TypeError("source replay fixture adapter must be verified")
        else:
            source_manifest = source_adapter.manifest
        assert (
            source_manifest.scope_contract_id == packet.scope_contract.scope_contract_id
        )
        assert source_manifest.task_family_id == binding.task_family_id
        assert source_manifest.task_adapter_id == binding.task_adapter_id
        assert not source_manifest_ids or source_manifest_ids == {
            source_adapter.manifest.task_adapter_manifest_id
        }
        assert not source_receipt_ids or source_receipt_ids == {
            source_adapter.verification_receipt.verification_receipt_id
        }
        self.exact_adapters = {
            (
                source_adapter.manifest.task_adapter_manifest_id,
                source_adapter.verification_receipt.verification_receipt_id,
            ): source_adapter
        }
        if rotate_active:
            active_values = source_manifest.to_dict()
            active_values.pop("task_adapter_manifest_id")
            active_values["validation_refs"] = tuple(
                sorted(
                    {
                        *source_manifest.validation_refs,
                        "validation.rotated_adapter",
                    }
                )
            )
            self.adapter = verified_test_task_adapter(
                TaskAdapterManifest.mint(**active_values),
                source_contents=source_adapter.source_contents,
            )
            self.exact_adapters[
                (
                    self.adapter.manifest.task_adapter_manifest_id,
                    self.adapter.verification_receipt.verification_receipt_id,
                )
            ] = self.adapter
        else:
            self.adapter = source_adapter
        self.timeouts_seen = []

    def resolve_active(self, **_binding):
        return self.adapter

    def resolve_exact(self, *, task_adapter_manifest_id, verification_receipt_id):
        return self.exact_adapters[(task_adapter_manifest_id, verification_receipt_id)]

    def resolve_exact_bounded(
        self,
        *,
        task_adapter_manifest_id,
        verification_receipt_id,
        maximum_entries,
        maximum_bytes,
        timeout_seconds,
    ):
        assert timeout_seconds > 0
        self.timeouts_seen.append(timeout_seconds)
        adapter = self.resolve_exact(
            task_adapter_manifest_id=task_adapter_manifest_id,
            verification_receipt_id=verification_receipt_id,
        )
        entry_count = (
            len(adapter.source_extraction_receipt.source_tree_files)
            + len(adapter.proof_objects)
            + 2
        )
        byte_count = (
            sum(
                descriptor.size
                for descriptor in adapter.source_extraction_receipt.source_tree_files
            )
            + len(adapter.source_archive)
            + sum(len(payload) for payload in adapter.proof_objects.values())
            + len(adapter.publisher_verification)
        )
        assert entry_count <= maximum_entries
        assert byte_count <= maximum_bytes
        return adapter


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
    validation_context = project_agent_candidate_validation_context(
        packet=packet,
        decision=decision,
    )
    assert claim.revision_id in (
        validation_context.replay_evidence.stable_dependency_ids
    )
    assert result.selection.validation_context_id == (
        validation_context.validation_context_id
    )
    assert result.selection.evidence_authority_ids == (
        validation_context.replay_evidence.evidence_authority_ids
    )
    assert {
        result.selection.validation_context_id,
        *result.selection.evidence_authority_ids,
    }.issubset(result.selection.exact_dependency_ids)
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
