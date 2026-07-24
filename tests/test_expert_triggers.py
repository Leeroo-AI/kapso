from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    CodingAgentOperationReceipt,
    CrossRunTaskBindingSettings,
    ExpertBaseReleaseManifest,
    ExpertReleaseLineage,
    ExpertCapabilityNode,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
    KnowledgeClaim,
    LineageEdge,
    LineageRelation,
    PublicationArtifactKind,
    RunBundle,
    SourceFileDescriptor,
    StrictContract,
    TaskAdapterBinding,
    TaskContextBinding,
    TaskFamilyDefinition,
    TransferEpisode,
)
from kapso.cross_run.record_contracts import SanitationReport
from kapso.cross_run.contracts import EMPTY_EXPERT_TREE_DIGEST
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertSourceBaseTreeReceipt,
    ExpertTriggerDecisionStore,
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvidencePacketBuilder,
    ExpertTriggerError,
    ExpertTriggerEvaluator,
    ExpertTriggerObservation,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.github.materializer import (
    SOURCE_ARCHIVE_EXTRACTOR_VERSION,
    CacheVerificationReceipt,
    SourceArchiveExtractionReceipt,
)
from kapso.cross_run.settings import CrossRunSettings, ExpertTriggerSettings
from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
)
from test_cross_run_contracts import build_records
from test_cross_run_retrieval import (
    analogical_context,
    environment_for_context,
    outcome_episodes,
    relbench_context,
    snapshot_and_index,
    source_fixture,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def digest(value: str) -> str:
    return tree_or_blob_digest(value.encode("utf-8"))


def trigger_settings(**updates: int) -> ExpertTriggerSettings:
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).expert.triggers
    return ExpertTriggerSettings.from_dict({**settings.to_dict(), **updates})


def configuration_fingerprint(settings: ExpertTriggerSettings) -> str:
    return tree_or_blob_digest(canonical_json_bytes(settings.to_dict()))


def source_base_tree_file() -> SourceFileDescriptor:
    content = b"verified source-base source"
    return SourceFileDescriptor(
        relative_path="src/expert.py",
        digest=tree_or_blob_digest(content),
        mode="100644",
        size=len(content),
    )


def verified_source_base_tree_hash() -> str:
    tree_file = source_base_tree_file()
    return source_tree_digest(
        {
            tree_file.relative_path: (
                tree_file.digest,
                tree_file.mode,
                tree_file.size,
            )
        }
    )


def inspection_operation(
    settings: ExpertTriggerSettings,
    final_output: str,
) -> CodingAgentOperationReceipt:
    final_result_digest = tree_or_blob_digest(final_output.encode("utf-8"))
    operation_suffix = digest("trigger-inspection").removeprefix("sha256:")[:32]
    return CodingAgentOperationReceipt.mint(
        operation_id=f"agent_call_{operation_suffix}",
        principal_id=settings.inspector_id,
        role=settings.inspector_role,
        cli="codex",
        model="gpt-5.6-sol",
        effort="xhigh",
        workspace_access=CodingAgentWorkspaceAccess.READ_ONLY,
        artifact_checksums={
            filename: (
                final_result_digest
                if filename == "final.json"
                else digest(f"inspection-{filename}")
            )
            for filename in coding_agent_artifact_filenames(
                CodingAgentWorkspaceAccess.READ_ONLY
            )
        },
    )


def task_binding(
    task_family_id: str,
    task_adapter_id: str,
) -> CrossRunTaskBindingSettings:
    return CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id=task_family_id,
        task_adapter_id=task_adapter_id,
    )


def expert_records():
    records = build_records()
    scope = next(
        record for record in records if isinstance(record, ExpertScopeContract)
    )
    module = next(
        record for record in records if isinstance(record, ExpertModuleContract)
    )
    repository_map = next(
        record for record in records if isinstance(record, ExpertRepositoryMap)
    )
    release = next(
        record for record in records if isinstance(record, ExpertBaseReleaseManifest)
    )
    return scope, module, repository_map, release


def clone_episode(
    episode: TransferEpisode,
    *,
    name: str,
    campaign_id: str,
    context: TaskContextBinding,
    parent_episode_ref: str | None = None,
) -> TransferEpisode:
    return TransferEpisode.mint(
        source={
            **episode.source,
            "campaign_id": campaign_id,
            "idea_id": f"idea-{name}",
            "node_id": f"node-{name}",
            "run_id": f"run-{name}",
        },
        source_bundle_id=episode.source_bundle_id,
        supersedes_projection_id=None,
        task_context_binding=context,
        artifact_environment=environment_for_context(
            episode.artifact_environment,
            context,
        ),
        proposal=f"Evaluate the {name} mechanism.",
        parent_episode_ref=parent_episode_ref,
        attempts=episode.attempts,
        terminal_attempt_revision=episode.terminal_attempt_revision,
        safe_observation_refs=episode.safe_observation_refs,
        sanitation_report_id=episode.sanitation_report_id,
        derivation_refs=episode.derivation_refs,
    )


def supported_claim(
    template: KnowledgeClaim,
    episodes: tuple[TransferEpisode, ...],
) -> KnowledgeClaim:
    return KnowledgeClaim.mint(
        claim_id=template.claim_id,
        scope_contract_id=template.scope_contract_id,
        statement=template.statement,
        mechanism=template.mechanism,
        applicability_predicates=template.applicability_predicates,
        explicit_exclusions=template.explicit_exclusions,
        supporting_episode_ids=tuple(
            sorted(episode.episode_id for episode in episodes)
        ),
        contradicting_episode_ids=(),
        proposal_provenance=template.proposal_provenance,
        supersedes_revision_ids=(),
    )


def trigger_packet(
    *,
    settings: ExpertTriggerSettings,
    episodes: tuple[TransferEpisode, ...] = (),
    claims: tuple[KnowledgeClaim, ...] = (),
    observations: tuple[ExpertTriggerObservation, ...] = (),
    bootstrap: bool = False,
    source_base_repository_map: ExpertRepositoryMap | None = None,
    source_base_module_contracts: tuple[ExpertModuleContract, ...] | None = None,
    source_base_release: ExpertBaseReleaseManifest | None = None,
    current_scope_contract: ExpertScopeContract | None = None,
    source_base_scope_contract: ExpertScopeContract | None = None,
    active_task_bindings: tuple[CrossRunTaskBindingSettings, ...] = (
        CrossRunTaskBindingSettings(
            scope_id="ml_ai",
            task_family_id="language_model_post_training",
            task_adapter_id="posttrain",
        ),
    ),
    knowledge_source_bundle: RunBundle | None = None,
    knowledge_sanitation_report: SanitationReport | None = None,
    knowledge_extra_facts: tuple[StrictContract, ...] = (),
    knowledge_projection_derivation_ids: tuple[str, ...] = (),
) -> ExpertTriggerEvidencePacket:
    scope, module, current_map, release = expert_records()
    current_scope = current_scope_contract or scope
    source_base_scope = source_base_scope_contract or scope
    selected_map = None if bootstrap else source_base_repository_map or current_map
    selected_modules = () if bootstrap else source_base_module_contracts or (module,)
    selected_release = None if bootstrap else source_base_release or release
    source_base_tree_hash = (
        EMPTY_EXPERT_TREE_DIGEST if bootstrap else verified_source_base_tree_hash()
    )
    source_base_tree_receipt = (
        None
        if bootstrap
        else ExpertSourceBaseTreeReceipt.mint(
            release_id=selected_release.release_id,
            cache_verification_receipt=CacheVerificationReceipt(
                artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
                artifact_id=selected_release.release_id,
                materialized_tree_digest=digest("full-materialized-package"),
                manifest_relative_path="expert.json",
                manifest_digest=digest("expert-manifest"),
                cache_tree_digest=digest("expert-cache"),
                asset_digests={
                    selected_release.source_archive_ref: selected_release.checksums[
                        selected_release.source_archive_ref
                    ]
                },
            ),
            source_extraction_receipt=SourceArchiveExtractionReceipt.mint(
                artifact_id=selected_release.release_id,
                source_archive_ref=selected_release.source_archive_ref,
                source_archive_digest=selected_release.checksums[
                    selected_release.source_archive_ref
                ],
                source_tree_hash=source_base_tree_hash,
                source_tree_files=(source_base_tree_file(),),
                extractor_version=SOURCE_ARCHIVE_EXTRACTOR_VERSION,
            ),
            source_base_tree_hash=source_base_tree_hash,
            repository_map_id=selected_map.repository_map_id,
            module_contract_ids=tuple(
                sorted(module.module_contract_id for module in selected_modules)
            ),
            materializer_version="kapso.expert_materializer.v1",
        )
    )
    package, _, _ = snapshot_and_index(
        (*episodes, *claims),
        extra_facts=knowledge_extra_facts,
        projection_derivation_ids=knowledge_projection_derivation_ids,
        source_bundle=knowledge_source_bundle,
        source_sanitation_report=knowledge_sanitation_report,
    )
    proof_ids = set(package.manifest.proof_dependency_closure_ids)
    return ExpertTriggerEvidencePacket.mint(
        knowledge_snapshot_manifest=package.manifest,
        knowledge_record_closure_digest=package.record_closure_digest,
        configuration_fingerprint=configuration_fingerprint(settings),
        scope_contract=current_scope,
        source_base_scope_contract=None if bootstrap else source_base_scope,
        source_base_release=selected_release,
        source_base_tree_receipt=source_base_tree_receipt,
        source_base_tree_hash=source_base_tree_hash,
        source_base_repository_map=selected_map,
        source_base_module_contracts=tuple(
            sorted(
                selected_modules,
                key=lambda contract: contract.module_contract_id,
            )
        ),
        episodes=tuple(sorted(episodes, key=lambda episode: episode.episode_id)),
        claims=tuple(sorted(claims, key=lambda claim: claim.revision_id)),
        trigger_observations=tuple(
            sorted(observations, key=lambda observation: observation.observation_id)
        ),
        active_task_bindings=active_task_bindings,
        proof_reference_ids=tuple(sorted(proof_ids)),
        recovery_barrier_basis_packet_id=None,
    )


def test_explicit_empty_parent_bootstraps_and_replays_exactly(tmp_path):
    settings = trigger_settings()
    packet = trigger_packet(settings=settings, bootstrap=True)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    assert decision.candidate_required is True
    assert decision.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE
    assert decision.reason_code == "empty_scope_bootstrap"

    store = ExpertTriggerDecisionStore(
        (tmp_path / "triggers").resolve(),
        tmp_path.resolve(),
        settings,
    )
    store.persist(packet, decision)
    store.persist(packet, decision)

    assert store.read_packet(packet.evidence_packet_id) == packet
    assert store.read_decision_for_packet(packet.evidence_packet_id) == decision


def test_bootstrap_requires_the_canonical_empty_tree_and_no_parent_topology():
    settings = trigger_settings()
    packet = trigger_packet(settings=settings, bootstrap=True)

    with pytest.raises(ExpertTriggerError, match="canonical empty source base"):
        replace(packet, source_base_tree_hash=digest("not-empty"))

    released = trigger_packet(settings=settings)
    with pytest.raises(ExpertTriggerError, match="released source base"):
        replace(released, source_base_tree_hash=EMPTY_EXPERT_TREE_DIGEST)
    different_content = b"different verified source-base source"
    different_file = SourceFileDescriptor(
        relative_path="src/expert.py",
        digest=tree_or_blob_digest(different_content),
        mode="100644",
        size=len(different_content),
    )
    different_tree_hash = source_tree_digest(
        {
            different_file.relative_path: (
                different_file.digest,
                different_file.mode,
                different_file.size,
            )
        }
    )
    mismatched_receipt = ExpertSourceBaseTreeReceipt.mint(
        release_id=released.source_base_release.release_id,
        cache_verification_receipt=replace(
            released.source_base_tree_receipt.cache_verification_receipt,
            cache_tree_digest=digest("different-cache-tree"),
        ),
        source_extraction_receipt=SourceArchiveExtractionReceipt.mint(
            artifact_id=released.source_base_release.release_id,
            source_archive_ref=released.source_base_release.source_archive_ref,
            source_archive_digest=released.source_base_release.checksums[
                released.source_base_release.source_archive_ref
            ],
            source_tree_hash=different_tree_hash,
            source_tree_files=(different_file,),
            extractor_version=SOURCE_ARCHIVE_EXTRACTOR_VERSION,
        ),
        source_base_tree_hash=different_tree_hash,
        repository_map_id=released.source_base_repository_map.repository_map_id,
        module_contract_ids=tuple(
            module.module_contract_id
            for module in released.source_base_module_contracts
        ),
        materializer_version="kapso.expert_materializer.v1",
    )
    with pytest.raises(ExpertTriggerError, match="tree receipt differs"):
        replace(released, source_base_tree_receipt=mismatched_receipt)


def test_snapshot_builder_uses_every_admitted_episode_not_a_retrieval_budget():
    settings = trigger_settings()
    _, context, episode, _, claim, _, _ = source_fixture()
    positive = outcome_episodes(episode)[0]
    other = clone_episode(
        positive,
        name="other",
        campaign_id=positive.source["campaign_id"],
        context=analogical_context(context),
    )
    claim = supported_claim(claim, (positive, other))
    package, _, _ = snapshot_and_index((positive, other, claim))

    packet = ExpertTriggerEvidencePacketBuilder(settings).build(
        knowledge_snapshot=package,
        scope_contract=package.prepared.scope_contract,
        source_base_scope_contract=None,
        source_base_release=None,
        source_base_tree_receipt=None,
        source_base_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        source_base_repository_map=None,
        source_base_module_contracts=(),
        active_task_bindings=(
            task_binding("language_model_post_training", "posttrain"),
        ),
    )

    assert tuple(item.episode_id for item in packet.episodes) == tuple(
        sorted((positive.episode_id, other.episode_id))
    )
    assert tuple(item.revision_id for item in packet.claims) == (claim.revision_id,)
    assert packet.knowledge_record_closure_digest == package.record_closure_digest


def test_repeated_success_requires_distinct_campaigns_and_transfer_contexts():
    settings = trigger_settings()
    _, context, episode, _, claim, _, _ = source_fixture()
    positive = outcome_episodes(episode)[0]
    first = clone_episode(
        positive,
        name="first",
        campaign_id="campaign-first",
        context=context,
    )
    second = clone_episode(
        positive,
        name="second",
        campaign_id="campaign-second",
        context=analogical_context(context),
    )
    claim = supported_claim(claim, (first, second))

    decision = ExpertTriggerEvaluator(settings).evaluate(
        trigger_packet(settings=settings, episodes=(first, second), claims=(claim,))
    )

    assert decision.candidate_required is True
    assert decision.change_kind is CandidateChangeKind.CAPABILITY
    assert decision.reason_code == "repeated_cross_context_success"
    assert len(decision.independent_lineage_ids) == 2
    assert len(decision.task_context_binding_ids) == 2


@pytest.mark.parametrize(
    ("same_campaign", "same_context"),
    ((True, False), (False, True)),
)
def test_repeated_success_rejects_cloned_lineage_or_unchanged_context(
    same_campaign,
    same_context,
):
    settings = trigger_settings()
    _, context, episode, _, claim, _, _ = source_fixture()
    positive = outcome_episodes(episode)[0]
    first = clone_episode(
        positive,
        name="first",
        campaign_id="campaign-first",
        context=context,
    )
    second = clone_episode(
        positive,
        name="second",
        campaign_id="campaign-first" if same_campaign else "campaign-second",
        context=context if same_context else analogical_context(context),
        parent_episode_ref=first.episode_id if same_campaign else None,
    )
    claim = supported_claim(claim, (first, second))

    decision = ExpertTriggerEvaluator(settings).evaluate(
        trigger_packet(settings=settings, episodes=(first, second), claims=(claim,))
    )

    assert decision.candidate_required is False
    assert decision.reason_code == "insufficient_evidence"


def test_repeated_difficulty_requires_a_typed_closed_observation():
    settings = trigger_settings()
    _, context, episode, _, _, _, _ = source_fixture()
    difficulty = outcome_episodes(episode)[3]
    first = clone_episode(
        difficulty,
        name="failure-first",
        campaign_id="campaign-first",
        context=context,
    )
    second = clone_episode(
        difficulty,
        name="failure-second",
        campaign_id="campaign-second",
        context=analogical_context(context),
    )
    source_base_tree_hash = verified_source_base_tree_hash()
    difficulty_signature = digest("shared-infrastructure-failure")
    exact_evidence_ids = tuple(sorted((first.episode_id, second.episode_id)))
    difficulty_evidence_signatures = {
        first.episode_id: difficulty_signature,
        second.episode_id: difficulty_signature,
    }
    task_context_binding_ids = tuple(
        sorted(
            (
                first.task_context_binding.task_context_binding_id,
                second.task_context_binding.task_context_binding_id,
            )
        )
    )
    description = "The same infrastructure boundary failed in independent contexts."
    inspection_payload = {
        "affected_capability_ids": (),
        "affected_paths": (),
        "configuration_fingerprint": configuration_fingerprint(settings),
        "description": description,
        "difficulty_evidence_signatures": difficulty_evidence_signatures,
        "difficulty_signature": difficulty_signature,
        "exact_evidence_ids": exact_evidence_ids,
        "independent_lineage_ids": (
            "ml_ai/campaign-first",
            "ml_ai/campaign-second",
        ),
        "inspection_policy_version": "kapso.expert_inspection.v1",
        "kind": "repeated_independent_difficulty",
        "occurrence_count": 2,
        "source_base_tree_hash": source_base_tree_hash,
        "task_context_binding_ids": task_context_binding_ids,
    }
    inspection_final_output = json.dumps(inspection_payload, indent=2) + "\n"
    operation = inspection_operation(settings, inspection_final_output)
    observation = ExpertTriggerObservation.mint(
        kind=ExpertTriggerObservationKind.REPEATED_INDEPENDENT_DIFFICULTY,
        source_base_tree_hash=source_base_tree_hash,
        inspection_policy_version="kapso.expert_inspection.v1",
        configuration_fingerprint=configuration_fingerprint(settings),
        inspection_operation=operation,
        inspection_final_output=inspection_final_output,
        difficulty_signature=difficulty_signature,
        difficulty_evidence_signatures=difficulty_evidence_signatures,
        description=description,
        affected_capability_ids=(),
        affected_paths=(),
        exact_evidence_ids=exact_evidence_ids,
        independent_lineage_ids=("ml_ai/campaign-first", "ml_ai/campaign-second"),
        task_context_binding_ids=task_context_binding_ids,
        occurrence_count=2,
    )

    without_observation = ExpertTriggerEvaluator(settings).evaluate(
        trigger_packet(settings=settings, episodes=(first, second))
    )
    with_observation = ExpertTriggerEvaluator(settings).evaluate(
        trigger_packet(
            settings=settings,
            episodes=(first, second),
            observations=(observation,),
        )
    )

    assert without_observation.candidate_required is False
    assert with_observation.candidate_required is True
    assert with_observation.reason_code == "repeated_independent_difficulty"

    with pytest.raises(ExpertTriggerError, match="result artifact"):
        replace(
            observation,
            inspection_final_output=observation.inspection_final_output + "\n",
        )
    with pytest.raises(ExpertTriggerError, match="fields differ"):
        replace(
            observation,
            difficulty_evidence_signatures={
                first.episode_id: observation.difficulty_signature,
                second.episode_id: digest("different-failure"),
            },
        )
    with pytest.raises(ExpertTriggerError, match="fields differ"):
        replace(observation, description="A different inspected conclusion.")

    authorized_operation = observation.inspection_operation
    unauthorized_operation = CodingAgentOperationReceipt.mint(
        operation_id=authorized_operation.operation_id,
        principal_id="rogue_trigger_inspector",
        role=authorized_operation.role,
        cli=authorized_operation.cli,
        model=authorized_operation.model,
        effort=authorized_operation.effort,
        workspace_access=authorized_operation.workspace_access,
        artifact_checksums=authorized_operation.artifact_checksums,
    )
    unauthorized_observation = ExpertTriggerObservation.mint(
        kind=observation.kind,
        source_base_tree_hash=observation.source_base_tree_hash,
        inspection_policy_version=observation.inspection_policy_version,
        configuration_fingerprint=observation.configuration_fingerprint,
        inspection_operation=unauthorized_operation,
        inspection_final_output=observation.inspection_final_output,
        difficulty_signature=observation.difficulty_signature,
        difficulty_evidence_signatures=observation.difficulty_evidence_signatures,
        description=observation.description,
        affected_capability_ids=observation.affected_capability_ids,
        affected_paths=observation.affected_paths,
        exact_evidence_ids=observation.exact_evidence_ids,
        independent_lineage_ids=observation.independent_lineage_ids,
        task_context_binding_ids=observation.task_context_binding_ids,
        occurrence_count=observation.occurrence_count,
    )
    unauthorized_packet = trigger_packet(
        settings=settings,
        episodes=(first, second),
        observations=(unauthorized_observation,),
    )
    with pytest.raises(ExpertTriggerError, match="inspection authority"):
        ExpertTriggerEvaluator(settings).evaluate(unauthorized_packet)


def test_uncovered_admitted_task_family_requests_an_architecture_candidate():
    settings = trigger_settings()
    scope, module, repository_map, release = expert_records()
    node = repository_map.capability_nodes[0]
    limited_map = ExpertRepositoryMap.mint(
        scope_contract_id=scope.scope_contract_id,
        capability_nodes=(
            ExpertCapabilityNode(
                capability_id=node.capability_id,
                module_contract_ref=node.module_contract_ref,
                owned_paths=node.owned_paths,
                task_family_bindings=("language_model_post_training",),
            ),
        ),
        dependency_edges=repository_map.dependency_edges,
        task_adapter_boundary=repository_map.task_adapter_boundary,
        validation_entrypoints=repository_map.validation_entrypoints,
        architecture_invariants=repository_map.architecture_invariants,
    )
    limited_release = ExpertBaseReleaseManifest.mint(
        scope_contract_id=release.scope_contract_id,
        scope_id=release.scope_id,
        lineage=release.lineage,
        candidate_id=release.candidate_id,
        candidate_commit_record_id=release.candidate_commit_record_id,
        candidate_tree_ref=release.candidate_tree_ref,
        candidate_tree_hash=release.candidate_tree_hash,
        candidate_derivation_ref=release.candidate_derivation_ref,
        candidate_validation_context_ref=release.candidate_validation_context_ref,
        candidate_patch_ref=release.candidate_patch_ref,
        candidate_sanitation_report_id=release.candidate_sanitation_report_id,
        candidate_ancestor_ids=release.candidate_ancestor_ids,
        candidate_source_dependency_ids=release.candidate_source_dependency_ids,
        candidate_consumed_expert_release_ids=(
            release.candidate_consumed_expert_release_ids
        ),
        repository_map_ref=limited_map.repository_map_id,
        module_contract_refs=release.module_contract_refs,
        module_versions=release.module_versions,
        semantic_book_digest=release.semantic_book_digest,
        validation_attempt_id=release.validation_attempt_id,
        approval_transition_id=release.approval_transition_id,
        approval_state_id=release.approval_state_id,
        publication_eligibility_result_id=(release.publication_eligibility_result_id),
        release_matrix_stage_result_id=release.release_matrix_stage_result_id,
        release_matrix_report_id=release.release_matrix_report_id,
        promotion_decision_id=release.promotion_decision_id,
        approval_assertion_ids=release.approval_assertion_ids,
        validation_policy_id=release.validation_policy_id,
        configuration_fingerprint=release.configuration_fingerprint,
        source_archive_ref=release.source_archive_ref,
        evidence_archive_ref=release.evidence_archive_ref,
        evidence_manifest_ref=release.evidence_manifest_ref,
        test_matrix_summary_ref=release.test_matrix_summary_ref,
        evidence_dependency_ids=release.evidence_dependency_ids,
        consumed_dependency_ids=tuple(
            sorted(
                {
                    *release.consumed_dependency_ids,
                    limited_map.repository_map_id,
                }
            )
        ),
        control_dependency_ids=release.control_dependency_ids,
        checksums=release.checksums,
    )

    inactive_packet = trigger_packet(
        settings=settings,
        source_base_repository_map=limited_map,
        source_base_module_contracts=(module,),
        source_base_release=limited_release,
    )
    assert (
        ExpertTriggerEvaluator(settings).evaluate(inactive_packet).candidate_required
        is False
    )

    packet = trigger_packet(
        settings=settings,
        source_base_repository_map=limited_map,
        source_base_module_contracts=(module,),
        source_base_release=limited_release,
        active_task_bindings=(
            task_binding("relational_tabular_prediction", "relbench"),
        ),
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    assert decision.candidate_required is True
    assert decision.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE
    assert decision.reason_code == "admitted_task_family_uncovered"


def test_attested_successor_scope_can_evolve_an_existing_release():
    settings = trigger_settings()
    source_base_scope, _, _, _ = expert_records()
    successor_scope = ExpertScopeContract.mint(
        scope_id=source_base_scope.scope_id,
        supersedes_scope_contract_id=source_base_scope.scope_contract_id,
        purpose=source_base_scope.purpose,
        explicit_non_goals=source_base_scope.explicit_non_goals,
        task_family_ontology=source_base_scope.task_family_ontology,
        task_family_lineage=source_base_scope.task_family_lineage,
        artifact_classes=("dataset", "feature_table", "model"),
        required_context_dimensions=source_base_scope.required_context_dimensions,
        context_dimension_schemas=source_base_scope.context_dimension_schemas,
        context_dimension_lineage=source_base_scope.context_dimension_lineage,
        task_adapter_contract=source_base_scope.task_adapter_contract,
        sanitation_policy_ref=source_base_scope.sanitation_policy_ref,
        validation_policy_ref=source_base_scope.validation_policy_ref,
        repository_architecture_constraints=source_base_scope.repository_architecture_constraints,
    )

    packet = trigger_packet(
        settings=settings,
        current_scope_contract=successor_scope,
        source_base_scope_contract=source_base_scope,
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    assert decision.candidate_required is True
    assert decision.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE
    assert decision.reason_code == "attested_scope_expansion"


def test_successor_family_rename_keeps_parent_evidence_historical():
    settings = trigger_settings()
    source_base_scope, _, _, _ = expert_records()
    successor_scope = ExpertScopeContract.mint(
        scope_id=source_base_scope.scope_id,
        supersedes_scope_contract_id=source_base_scope.scope_contract_id,
        purpose=source_base_scope.purpose,
        explicit_non_goals=source_base_scope.explicit_non_goals,
        task_family_ontology=(
            TaskFamilyDefinition(
                task_family_id="general_model_adaptation",
                capability_tags=("adaptation.training",),
            ),
        ),
        task_family_lineage=(
            LineageEdge(
                source_ids=("language_model_post_training",),
                target_ids=("general_model_adaptation",),
                relation=LineageRelation.RENAME,
            ),
        ),
        artifact_classes=source_base_scope.artifact_classes,
        required_context_dimensions=source_base_scope.required_context_dimensions,
        context_dimension_schemas=source_base_scope.context_dimension_schemas,
        context_dimension_lineage=source_base_scope.context_dimension_lineage,
        task_adapter_contract=(
            TaskAdapterBinding(
                task_family_id="general_model_adaptation",
                task_adapter_ids=("general_adapter",),
            ),
        ),
        sanitation_policy_ref=source_base_scope.sanitation_policy_ref,
        validation_policy_ref=source_base_scope.validation_policy_ref,
        repository_architecture_constraints=source_base_scope.repository_architecture_constraints,
    )
    _, context, episode, _, _, _, _ = source_fixture()
    historical = clone_episode(
        outcome_episodes(episode)[0],
        name="historical-relbench",
        campaign_id="historical-relbench-campaign",
        context=relbench_context(context),
    )

    packet = trigger_packet(
        settings=settings,
        episodes=(historical,),
        current_scope_contract=successor_scope,
        source_base_scope_contract=source_base_scope,
        active_task_bindings=(
            task_binding("general_model_adaptation", "general_adapter"),
        ),
    )
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    assert packet.active_task_family_ids == ("general_model_adaptation",)
    assert decision.candidate_required is True
    assert decision.reason_code == "attested_scope_expansion"

    malformed_scope = ExpertScopeContract.mint(
        scope_id=successor_scope.scope_id,
        supersedes_scope_contract_id=source_base_scope.scope_contract_id,
        purpose=successor_scope.purpose,
        explicit_non_goals=successor_scope.explicit_non_goals,
        task_family_ontology=successor_scope.task_family_ontology,
        task_family_lineage=(
            LineageEdge(
                source_ids=("nonexistent_parent_family",),
                target_ids=("general_model_adaptation",),
                relation=LineageRelation.RENAME,
            ),
        ),
        artifact_classes=successor_scope.artifact_classes,
        required_context_dimensions=successor_scope.required_context_dimensions,
        context_dimension_schemas=successor_scope.context_dimension_schemas,
        context_dimension_lineage=successor_scope.context_dimension_lineage,
        task_adapter_contract=successor_scope.task_adapter_contract,
        sanitation_policy_ref=successor_scope.sanitation_policy_ref,
        validation_policy_ref=successor_scope.validation_policy_ref,
        repository_architecture_constraints=successor_scope.repository_architecture_constraints,
    )
    with pytest.raises(ExpertTriggerError, match="lineage source.*source base"):
        trigger_packet(
            settings=settings,
            episodes=(historical,),
            current_scope_contract=malformed_scope,
            source_base_scope_contract=source_base_scope,
            active_task_bindings=(
                task_binding("general_model_adaptation", "general_adapter"),
            ),
        )
    malformed_context_scope = ExpertScopeContract.mint(
        **{
            key: value
            for key, value in successor_scope.to_dict().items()
            if key not in {"scope_contract_id", "context_dimension_lineage"}
        },
        context_dimension_lineage=(
            LineageEdge(
                source_ids=("nonexistent_parent_dimension",),
                target_ids=(successor_scope.context_dimension_schemas[0].dimension_id,),
                relation=LineageRelation.RENAME,
            ),
        ),
    )
    with pytest.raises(ExpertTriggerError, match="context lineage source.*source base"):
        trigger_packet(
            settings=settings,
            episodes=(historical,),
            current_scope_contract=malformed_context_scope,
            source_base_scope_contract=source_base_scope,
            active_task_bindings=(
                task_binding("general_model_adaptation", "general_adapter"),
            ),
        )


def test_trigger_packet_and_store_reject_noncanonical_or_tampered_state(tmp_path):
    settings = trigger_settings()
    _, _, episode, _, _, _, _ = source_fixture()
    first, second = outcome_episodes(episode)[:2]

    ordered = trigger_packet(settings=settings, episodes=(first, second))
    with pytest.raises(ExpertTriggerError, match="sorted"):
        replace(ordered, episodes=tuple(reversed(ordered.episodes)))

    packet = trigger_packet(settings=settings)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    store = ExpertTriggerDecisionStore(
        (tmp_path / "triggers").resolve(),
        tmp_path.resolve(),
        settings,
    )
    store.persist(packet, decision)
    packet_path = (
        store.packet_root / f"{packet.evidence_packet_id.rsplit(':', 1)[1]}.json"
    )
    packet_path.chmod(0o644)

    with pytest.raises(ExpertTriggerError, match="private regular file"):
        store.read_packet(packet.evidence_packet_id)

    symlink_target = tmp_path / "target"
    symlink_target.mkdir()
    symlink_root = tmp_path / "trigger-link"
    symlink_root.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(OSError):
        ExpertTriggerDecisionStore(
            symlink_root,
            tmp_path.resolve(),
            settings,
        )


def test_store_rejects_a_decision_for_another_packet(tmp_path):
    settings = trigger_settings()
    packet = trigger_packet(settings=settings)
    other_packet = trigger_packet(
        settings=settings,
        bootstrap=True,
    )
    other_decision = ExpertTriggerEvaluator(settings).evaluate(other_packet)

    store = ExpertTriggerDecisionStore(
        (tmp_path / "triggers").resolve(),
        tmp_path.resolve(),
        settings,
    )
    with pytest.raises(ExpertTriggerError, match="deterministic evaluation"):
        store.persist(packet, other_decision)


def test_store_recomputes_a_canonical_decision_on_read(tmp_path):
    settings = trigger_settings()
    packet = trigger_packet(settings=settings)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    assert decision.candidate_required is False
    store = ExpertTriggerDecisionStore(
        (tmp_path / "triggers").resolve(),
        tmp_path.resolve(),
        settings,
    )
    store.persist(packet, decision)
    forged = ExpertEvolutionTriggerDecision.mint(
        evidence_packet_id=packet.evidence_packet_id,
        knowledge_snapshot_id=packet.knowledge_snapshot_id,
        policy_version=settings.policy_version,
        configuration_fingerprint=packet.configuration_fingerprint,
        candidate_required=True,
        change_kind=CandidateChangeKind.CAPABILITY,
        reason_code="forged_candidate_authority",
        trigger_evidence_ids=(packet.knowledge_snapshot_id,),
        independent_lineage_ids=(),
        task_context_binding_ids=(),
        rationale="Canonical bytes must not replace deterministic authority.",
    )
    decision_path = (
        store.decision_root / f"{packet.evidence_packet_id.rsplit(':', 1)[1]}.json"
    )
    decision_path.write_bytes(forged.to_json_bytes())
    decision_path.chmod(0o600)

    with pytest.raises(ExpertTriggerError, match="atomic commit"):
        store.read_decision_for_packet(packet.evidence_packet_id)


def test_store_exposes_only_atomically_committed_packet_decision_pairs(tmp_path):
    settings = trigger_settings()
    packet = trigger_packet(settings=settings)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    store = ExpertTriggerDecisionStore(
        (tmp_path / "triggers").resolve(),
        tmp_path.resolve(),
        settings,
    )
    filename = f"{packet.evidence_packet_id.rsplit(':', 1)[1]}.json"
    conflicting_decision = store.decision_root / filename
    conflicting_decision.write_bytes(b"incomplete decision")
    conflicting_decision.chmod(0o600)

    with pytest.raises(ExpertTriggerError, match="conflicts"):
        store.persist(packet, decision)
    assert (store.packet_root / filename).is_file()
    with pytest.raises(FileNotFoundError):
        store.read_packet(packet.evidence_packet_id)
    with pytest.raises(FileNotFoundError):
        store.read_decision_for_packet(packet.evidence_packet_id)

    conflicting_decision.unlink()
    store.persist(packet, decision)
    assert store.read_packet(packet.evidence_packet_id) == packet
    assert store.read_decision_for_packet(packet.evidence_packet_id) == decision


def test_store_rejects_fifo_and_lock_link_tampering_without_blocking(tmp_path):
    settings = trigger_settings()
    packet = trigger_packet(settings=settings)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)

    fifo_store = ExpertTriggerDecisionStore(
        (tmp_path / "fifo-triggers").resolve(),
        tmp_path.resolve(),
        settings,
    )
    packet_filename = f"{packet.evidence_packet_id.rsplit(':', 1)[1]}.json"
    os.mkfifo(fifo_store.packet_root / packet_filename, mode=0o600)
    with pytest.raises(ExpertTriggerError, match="private regular file"):
        fifo_store.persist(packet, decision)

    link_target = tmp_path / "lock-target"
    link_target.write_bytes(b"untouched")
    link_target.chmod(0o600)
    linked_store = ExpertTriggerDecisionStore(
        (tmp_path / "linked-triggers").resolve(),
        tmp_path.resolve(),
        settings,
    )
    (linked_store.root / "trigger.lock").symlink_to(link_target)
    with pytest.raises(OSError):
        linked_store.persist(packet, decision)
    assert link_target.read_bytes() == b"untouched"


def test_store_anchors_child_directories_and_ignores_abandoned_temporaries(tmp_path):
    settings = trigger_settings()
    packet = trigger_packet(settings=settings)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    store = ExpertTriggerDecisionStore(
        (tmp_path / "triggers").resolve(),
        tmp_path.resolve(),
        settings,
    )
    abandoned = store.packet_root / ".abandoned.tmp"
    abandoned.write_bytes(b"partial")
    abandoned.chmod(0o600)
    store.persist(packet, decision)
    assert store.read_packet(packet.evidence_packet_id) == packet

    original_packets = tmp_path / "original-packets"
    store.packet_root.rename(original_packets)
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    store.packet_root.symlink_to(external, target_is_directory=True)

    with pytest.raises(OSError):
        store.persist(packet, decision)
    assert tuple(external.iterdir()) == ()


def test_store_rejects_a_broad_or_unowned_root(tmp_path):
    settings = trigger_settings()

    with pytest.raises(ExpertTriggerError, match="direct child"):
        ExpertTriggerDecisionStore(
            (tmp_path / "nested" / "triggers").resolve(),
            tmp_path.resolve(),
            settings,
        )
    with pytest.raises(ExpertTriggerError, match="authorized real directory"):
        ExpertTriggerDecisionStore(
            (tmp_path / "triggers").resolve(),
            tmp_path / "missing-state-root",
            settings,
        )
