from __future__ import annotations

from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.core.embeddings import EmbeddingRecord, complete_input_hash
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.record_contracts import BundleProjectionManifest, SanitationReport
from kapso.cross_run.catalog.store import CatalogGenerationManifest, CatalogInputDelta
from kapso.cross_run.contracts import (
    AdmissionState,
    ArtifactEnvironment,
    CatalogEntryState,
    ComparisonStatus,
    ContractValidationError,
    EffectUncertaintyMethod,
    EpisodeEvaluationStatus,
    ExpertScopeContract,
    ExecutionStatus,
    InterventionStructure,
    KnowledgeClaim,
    PriorIdea,
    PriorKnowledgeSnapshot,
    RelativeEffect,
    RunBundle,
    StrictContract,
    TaskContextBinding,
    TransferAttempt,
    TransferCompatibility,
    TransferEpisode,
)
from kapso.cross_run.knowledge.access import (
    PriorKnowledgeAccess,
    PriorKnowledgeAccessMaterialization,
)
from kapso.cross_run.knowledge.index import SnapshotSearchIndex
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackageBuilder
from kapso.cross_run.knowledge.retrieval import (
    CrossRunRetrievalError,
    CrossRunRetriever,
    PriorKnowledgeQuery,
)
from kapso.cross_run.settings import CrossRunSettings, RetrievalSettings
from test_cross_run_admission import sanitation_report
from test_cross_run_contracts import build_records

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def digest(value: str) -> str:
    return tree_or_blob_digest(value.encode("utf-8"))


def source_fixture():
    records = build_records()
    scope = next(
        record for record in records if isinstance(record, ExpertScopeContract)
    )
    context = next(
        record for record in records if isinstance(record, TaskContextBinding)
    )
    bundle = next(record for record in records if isinstance(record, RunBundle))
    raw_episode = next(
        record for record in records if isinstance(record, TransferEpisode)
    )
    raw_prior_idea = next(record for record in records if isinstance(record, PriorIdea))
    raw_claim = next(record for record in records if isinstance(record, KnowledgeClaim))
    report = sanitation_report(bundle.bundle_id, context)
    episode = TransferEpisode.mint(
        source=raw_episode.source,
        source_bundle_id=bundle.bundle_id,
        supersedes_projection_id=None,
        task_context_binding=context,
        artifact_environment=raw_episode.artifact_environment,
        proposal=raw_episode.proposal,
        parent_episode_ref=None,
        attempts=raw_episode.attempts,
        terminal_attempt_revision=raw_episode.terminal_attempt_revision,
        safe_observation_refs=raw_episode.safe_observation_refs,
        sanitation_report_id=report.report_id,
        derivation_refs=(bundle.bundle_id,),
    )
    prior_idea = PriorIdea.mint(
        source_bundle_id=bundle.bundle_id,
        supersedes_projection_id=None,
        source={
            **raw_prior_idea.source,
            "run_id": bundle.run_id,
            "campaign_id": bundle.campaign_id,
        },
        proposal=raw_prior_idea.proposal,
        descriptor=raw_prior_idea.descriptor,
        assumptions=raw_prior_idea.assumptions,
        source_status=raw_prior_idea.source_status,
        source_rationale=raw_prior_idea.source_rationale,
        source_evidence_refs=raw_prior_idea.source_evidence_refs,
        task_context_binding=context,
        sanitation_report_id=report.report_id,
    )
    claim = KnowledgeClaim.mint(
        claim_id=raw_claim.claim_id,
        scope_contract_id=scope.scope_contract_id,
        statement=raw_claim.statement,
        mechanism=raw_claim.mechanism,
        applicability_predicates=raw_claim.applicability_predicates,
        explicit_exclusions=raw_claim.explicit_exclusions,
        supporting_episode_ids=(episode.episode_id,),
        contradicting_episode_ids=(),
        proposal_provenance=raw_claim.proposal_provenance,
        supersedes_revision_ids=(),
    )
    return scope, context, episode, prior_idea, claim, bundle, report


def source_records():
    return source_fixture()[:5]


def analogical_context(context: TaskContextBinding) -> TaskContextBinding:
    return TaskContextBinding.mint(
        scope_contract_id=context.scope_contract_id,
        scope_id=context.scope_id,
        task_family_id=context.task_family_id,
        task_adapter_id=context.task_adapter_id,
        capability_tags=context.capability_tags,
        input_contract_fingerprint=digest("analog-input"),
        target_contract_fingerprint=context.target_contract_fingerprint,
        starting_artifact_refs=context.starting_artifact_refs,
        method_fingerprint=context.method_fingerprint,
        toolchain_fingerprint=context.toolchain_fingerprint,
        dependency_runtime_fingerprint=context.dependency_runtime_fingerprint,
        budget_hardware_envelope=context.budget_hardware_envelope,
        transfer_dimensions={
            **context.transfer_dimensions,
            "dataset_family": "other-instruction",
        },
    )


def relbench_context(context: TaskContextBinding) -> TaskContextBinding:
    return TaskContextBinding.mint(
        scope_contract_id=context.scope_contract_id,
        scope_id=context.scope_id,
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
        capability_tags=("relational.training",),
        input_contract_fingerprint=digest("rel-input"),
        target_contract_fingerprint=digest("rel-target"),
        starting_artifact_refs=(),
        method_fingerprint=digest("rel-method"),
        toolchain_fingerprint=digest("rel-toolchain"),
        dependency_runtime_fingerprint=digest("rel-runtime"),
        budget_hardware_envelope=context.budget_hardware_envelope,
        transfer_dimensions={
            "dataset_family": "relational",
            "runtime_family": context.transfer_dimensions["runtime_family"],
        },
    )


def episode_with_context(
    episode: TransferEpisode,
    context: TaskContextBinding,
    name: str,
) -> TransferEpisode:
    return TransferEpisode.mint(
        source={
            **episode.source,
            "node_id": f"node-{name}",
            "idea_id": f"idea-{name}",
        },
        source_bundle_id=episode.source_bundle_id,
        supersedes_projection_id=None,
        task_context_binding=context,
        artifact_environment=environment_for_context(
            episode.artifact_environment,
            context,
        ),
        proposal=f"Evaluate the {name} compatibility intervention.",
        parent_episode_ref=None,
        attempts=episode.attempts,
        terminal_attempt_revision=episode.terminal_attempt_revision,
        safe_observation_refs=episode.safe_observation_refs,
        sanitation_report_id=episode.sanitation_report_id,
        derivation_refs=episode.derivation_refs,
    )


def episode_with_proposal(
    episode: TransferEpisode,
    proposal: str,
) -> TransferEpisode:
    return TransferEpisode.mint(
        source=episode.source,
        source_bundle_id=episode.source_bundle_id,
        supersedes_projection_id=episode.supersedes_projection_id,
        task_context_binding=episode.task_context_binding,
        artifact_environment=episode.artifact_environment,
        proposal=proposal,
        parent_episode_ref=episode.parent_episode_ref,
        attempts=episode.attempts,
        terminal_attempt_revision=episode.terminal_attempt_revision,
        safe_observation_refs=episode.safe_observation_refs,
        sanitation_report_id=episode.sanitation_report_id,
        derivation_refs=episode.derivation_refs,
    )


def episode_with_attempt(
    episode: TransferEpisode,
    attempt: TransferAttempt,
    name: str,
    *,
    task_context: TaskContextBinding | None = None,
    proposal: str | None = None,
) -> TransferEpisode:
    selected_context = task_context or episode.task_context_binding
    return TransferEpisode.mint(
        source={
            **episode.source,
            "node_id": f"node-{name}",
            "idea_id": f"idea-{name}",
        },
        source_bundle_id=episode.source_bundle_id,
        supersedes_projection_id=None,
        task_context_binding=selected_context,
        artifact_environment=environment_for_context(
            episode.artifact_environment,
            selected_context,
        ),
        proposal=proposal or f"Evaluate the {name} outcome intervention.",
        parent_episode_ref=None,
        attempts=(attempt,),
        terminal_attempt_revision=0,
        safe_observation_refs=episode.safe_observation_refs,
        sanitation_report_id=episode.sanitation_report_id,
        derivation_refs=episode.derivation_refs,
    )


def environment_for_context(
    environment: ArtifactEnvironment,
    context: TaskContextBinding,
) -> ArtifactEnvironment:
    if set(environment.starting_artifact_content_ids) == set(
        context.starting_artifact_refs
    ):
        return environment
    values = environment.to_dict()
    values.pop("artifact_environment_id")
    values["starting_artifact_content_ids"] = {
        reference: content_id(
            "source-replay-starting-artifact",
            {"reference": reference},
        )
        for reference in context.starting_artifact_refs
    }
    return ArtifactEnvironment.mint(**values)


def outcome_episodes(episode: TransferEpisode) -> tuple[TransferEpisode, ...]:
    positive_attempt = episode.attempts[0]
    fingerprint = positive_attempt.evaluation_fingerprints[0]
    negative_value = positive_attempt.source_parent_effect.source_parent_value - 0.1
    negative_delta = (
        negative_value - positive_attempt.source_parent_effect.source_parent_value
    )
    negative_effect = RelativeEffect(
        evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
        metric_name=fingerprint.metric_name,
        objective_direction=fingerprint.objective_direction,
        candidate_value=negative_value,
        source_parent_value=positive_attempt.source_parent_effect.source_parent_value,
        raw_delta=negative_delta,
        normalized_delta=negative_delta,
        uncertainty=None,
        uncertainty_method=EffectUncertaintyMethod.UNAVAILABLE,
    )
    negative_attempt = replace_attempt(
        positive_attempt,
        measurements={fingerprint.metric_name: negative_value},
        source_parent_effect=negative_effect,
    )
    inconclusive_attempt = replace_attempt(
        positive_attempt,
        comparison_status=ComparisonStatus.INCONCLUSIVE,
        source_parent_effect=None,
    )
    frontier_attempt = TransferAttempt(
        execution_revision=0,
        captured_at=positive_attempt.captured_at,
        execution_status=ExecutionStatus.FAILED_TECHNICAL,
        evaluation_status=EpisodeEvaluationStatus.NOT_RUN,
        evaluation_fingerprints=(),
        score_of_record_fingerprint_id=None,
        comparison_status=ComparisonStatus.NOT_COMPARABLE,
        measurements={},
        source_parent_effect=None,
        intervention_ref=None,
        intervention_structure=InterventionStructure.UNDETERMINED,
        feedback=(),
        technical_difficulties=("The worker failed before evaluation.",),
        confounders=(),
    )
    return (
        episode_with_attempt(episode, positive_attempt, "positive"),
        episode_with_attempt(episode, negative_attempt, "negative"),
        episode_with_attempt(episode, inconclusive_attempt, "inconclusive"),
        episode_with_attempt(episode, frontier_attempt, "frontier"),
    )


def replace_attempt(attempt: TransferAttempt, **updates) -> TransferAttempt:
    fields = {
        "execution_revision": attempt.execution_revision,
        "captured_at": attempt.captured_at,
        "execution_status": attempt.execution_status,
        "evaluation_status": attempt.evaluation_status,
        "evaluation_fingerprints": attempt.evaluation_fingerprints,
        "score_of_record_fingerprint_id": attempt.score_of_record_fingerprint_id,
        "comparison_status": attempt.comparison_status,
        "measurements": attempt.measurements,
        "source_parent_effect": attempt.source_parent_effect,
        "intervention_ref": attempt.intervention_ref,
        "intervention_structure": attempt.intervention_structure,
        "feedback": attempt.feedback,
        "technical_difficulties": attempt.technical_difficulties,
        "confounders": attempt.confounders,
    }
    fields.update(updates)
    return TransferAttempt(**fields)


def retrieval_settings(**updates: int | float) -> RetrievalSettings:
    canonical = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).knowledge.retrieval
    return RetrievalSettings.from_dict({**canonical.to_dict(), **updates})


def snapshot_and_index(
    roots: tuple[TransferEpisode | PriorIdea | KnowledgeClaim, ...],
    settings: RetrievalSettings | None = None,
    extra_facts: tuple[StrictContract, ...] = (),
    projection_derivation_ids: tuple[str, ...] = (),
    source_bundle: RunBundle | None = None,
    source_sanitation_report: SanitationReport | None = None,
):
    scope, _, _, _, _, fixture_bundle, fixture_report = source_fixture()
    if (source_bundle is None) != (source_sanitation_report is None):
        raise ValueError(
            "source bundle and sanitation report must be supplied together"
        )
    bundle = fixture_bundle if source_bundle is None else source_bundle
    report = (
        fixture_report if source_sanitation_report is None else source_sanitation_report
    )
    configuration_fingerprint = digest("retrieval-catalog-config")
    states = tuple(
        CatalogEntryState.mint(
            subject_payload_id=(
                root.episode_id
                if isinstance(root, TransferEpisode)
                else (
                    root.prior_idea_id
                    if isinstance(root, PriorIdea)
                    else root.revision_id
                )
            ),
            catalog_generation=1,
            predecessor_state_id=None,
            configuration_fingerprint=configuration_fingerprint,
            admission_state=AdmissionState.ADMITTED,
            superseded_by_payload_ids=(),
            assertion_ids=(),
            revocation_ids=(),
            taint_source_ids=(),
        )
        for root in roots
    )
    projection = BundleProjectionManifest.mint(
        source_bundle_id=bundle.bundle_id,
        sanitation_report_id=report.report_id,
        episode_ids=tuple(
            sorted(
                root.episode_id for root in roots if isinstance(root, TransferEpisode)
            )
        ),
        prior_idea_ids=tuple(
            sorted(root.prior_idea_id for root in roots if isinstance(root, PriorIdea))
        ),
        derivation_object_ids=projection_derivation_ids,
    )
    facts = (*roots, bundle, report, projection, *extra_facts)
    objects_by_id = {getattr(record, record.IDENTITY_FIELD): record for record in facts}
    states_by_subject = {
        state.subject_payload_id: state.catalog_entry_state_id for state in states
    }
    objects_by_id.update({state.catalog_entry_state_id: state for state in states})
    fact_ids = tuple(sorted(getattr(record, record.IDENTITY_FIELD) for record in facts))
    input_delta = CatalogInputDelta.mint(
        scope_contract_id=scope.scope_contract_id,
        operation_id="retrieval-fixture-publication",
        configuration_fingerprint=configuration_fingerprint,
        added_object_ids=fact_ids,
        dependency_closure_ids=fact_ids,
    )
    objects_by_id[input_delta.input_delta_id] = input_delta
    generation = CatalogGenerationManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        generation_number=1,
        parent_generation_id=("catalog-generation:sha256:" + "0" * 64),
        configuration_fingerprint=configuration_fingerprint,
        fact_object_ids=fact_ids,
        derived_object_ids=tuple(
            sorted(state.catalog_entry_state_id for state in states)
        ),
        applied_input_delta_ids=(input_delta.input_delta_id,),
        bundle_frontier={
            f"{bundle.scope_id}/{bundle.run_id}/{bundle.campaign_id}": (
                bundle.bundle_id
            )
        },
        active_entry_state_ids=states_by_subject,
    )
    prepared = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        generation,
        lambda object_id: objects_by_id[object_id].to_json_bytes(),
    )
    index = SnapshotSearchIndex.build(prepared)
    retrieval_policy = settings or retrieval_settings()
    package = KnowledgeSnapshotPackageBuilder.finalize(
        prepared,
        parent_snapshot_ids=(),
        sanitation_policy_version="kapso.sanitation.v1",
        retrieval_policy_version="kapso.retrieval.v1",
        configuration_fingerprint=configuration_fingerprint,
        prompt_budget_policy=retrieval_policy.to_dict(),
        published_at="2026-07-21T16:00:00Z",
        publisher_attestation={"issuer": "retrieval-test"},
        search_files=index.files,
        embedding_sidecars=index.embedding_sidecars,
    )
    index.verify(package.manifest)
    return package, index, states


def query(context: TaskContextBinding, **updates) -> PriorKnowledgeQuery:
    fields = {
        "task_context_binding": context,
        "problem": "Improve representation parity reliably.",
        "current_gaps": ("Need cross-run evidence.",),
        "directive": "Retrieve relevant tested interventions.",
    }
    fields.update(updates)
    return PriorKnowledgeQuery(**fields)


def claim_supported_by(
    template: KnowledgeClaim,
    support: TransferEpisode,
    context: TaskContextBinding,
) -> KnowledgeClaim:
    return KnowledgeClaim.mint(
        claim_id=f"{template.claim_id}-cross-family",
        scope_contract_id=template.scope_contract_id,
        statement="The abstract intervention applies under the target predicates.",
        mechanism=template.mechanism,
        applicability_predicates=dict(context.transfer_dimensions),
        explicit_exclusions=template.explicit_exclusions,
        supporting_episode_ids=(support.episode_id,),
        contradicting_episode_ids=(),
        proposal_provenance=template.proposal_provenance,
        supersedes_revision_ids=(),
    )


def test_retrieval_separates_exact_and_analogical_and_excludes_other_task_family():
    _, context, episode, _, _ = source_records()
    analogical = episode_with_context(episode, analogical_context(context), "analog")
    incompatible = episode_with_context(episode, relbench_context(context), "rel")
    package, index, _ = snapshot_and_index((episode, analogical, incompatible))

    result = CrossRunRetriever(
        package,
        index,
        retrieval_settings(),
    ).retrieve(query(context))

    assert tuple(selection.compatibility for selection in result.selections) == (
        TransferCompatibility.EXACT_CONTEXT,
        TransferCompatibility.ANALOGICAL,
    )
    assert incompatible.episode_id not in (
        result.prior_knowledge_snapshot.selected_record_ids
    )


def test_effect_request_hard_filters_noncomparable_evaluation_identity():
    _, context, episode, _, _ = source_records()
    package, index, _ = snapshot_and_index((episode,))
    matching_id = episode.attempts[0].score_of_record_fingerprint_id

    matching = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context, effect_evaluation_fingerprint_ids=(matching_id,))
    )
    mismatching = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(
            context,
            effect_evaluation_fingerprint_ids=(
                "evaluation-fingerprint:sha256:" + "f" * 64,
            ),
        )
    )

    assert matching.prior_knowledge_snapshot.selected_record_ids == (
        episode.episode_id,
    )
    assert mismatching.prior_knowledge_snapshot.selected_record_ids == ()


def test_effect_request_excludes_frontier_without_a_score_fingerprint():
    _, context, episode, _, _ = source_records()
    frontier = outcome_episodes(episode)[3]
    package, index, _ = snapshot_and_index((frontier,))
    requested_id = episode.attempts[0].score_of_record_fingerprint_id

    result = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context, effect_evaluation_fingerprint_ids=(requested_id,))
    )

    assert result.prior_knowledge_snapshot.selected_record_ids == ()


def test_outcome_round_robin_prevents_positive_results_from_consuming_every_slot():
    _, context, episode, _, _ = source_records()
    roots = outcome_episodes(episode)
    package, index, _ = snapshot_and_index(roots)

    result = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context)
    )

    assert tuple(selection.outcome for selection in result.selections) == (
        "positive",
        "negative",
        "inconclusive",
        "frontier",
    )


def test_matching_cross_family_claim_is_analogical_and_uses_inconclusive_slot():
    _, context, episode, _, claim_template = source_records()
    cross_family_episode = episode_with_attempt(
        episode,
        episode.attempts[0],
        "cross-family",
        task_context=relbench_context(context),
    )
    claim = claim_supported_by(claim_template, cross_family_episode, context)
    package, index, _ = snapshot_and_index((cross_family_episode, claim))

    result = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context)
    )

    assert len(result.selections) == 1
    assert result.selections[0].record_id == claim.revision_id
    assert result.selections[0].compatibility is TransferCompatibility.ANALOGICAL
    assert result.selections[0].outcome == "inconclusive"


def test_prompt_budget_counts_cross_family_claim_proof_records():
    _, context, episode, _, claim_template = source_records()
    cross_family_episode = episode_with_attempt(
        episode,
        episode.attempts[0],
        "large-cross-family",
        task_context=relbench_context(context),
        proposal="Cross-family evidence " + "x" * 48000,
    )
    claim = claim_supported_by(claim_template, cross_family_episode, context)
    package, index, _ = snapshot_and_index((cross_family_episode, claim))

    result = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context)
    )

    assert result.prior_knowledge_snapshot.selected_record_ids == ()
    assert result.access_materialization.proof_records == ()


def test_retriever_rejects_runtime_policy_different_from_pinned_snapshot():
    _, _, episode, _, _ = source_records()
    package, index, _ = snapshot_and_index((episode,))

    with pytest.raises(
        CrossRunRetrievalError,
        match="retrieval settings differ",
    ):
        CrossRunRetriever(
            package,
            index,
            retrieval_settings(max_records_per_run=1),
        )


def test_retriever_rejects_in_memory_index_state_not_owned_by_sidecar_bytes():
    _, _, episode, _, _ = source_records()
    package, index, _ = snapshot_and_index((episode,))
    forged_metadata = {
        record_id: {
            **metadata,
            "outcome": "negative",
        }
        for record_id, metadata in index.metadata_by_id.items()
    }
    forged = replace(index, metadata_by_id=forged_metadata)

    with pytest.raises(
        CrossRunRetrievalError,
        match="in-memory search index differs",
    ):
        CrossRunRetriever(package, forged, retrieval_settings())


def test_selected_root_carries_only_its_query_proof_not_snapshot_audit_records():
    scope, context, episode, _, _ = source_records()
    package, index, states = snapshot_and_index((episode,))

    result = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context)
    )

    proof_ids = result.prior_knowledge_snapshot.proof_reference_ids
    assert episode.episode_id not in proof_ids
    assert states[0].catalog_entry_state_id in proof_ids
    assert scope.scope_contract_id in proof_ids
    assert package.prepared.catalog_generation_id not in proof_ids
    assert (
        package.prepared.catalog_generation.applied_input_delta_ids[0] not in proof_ids
    )
    assert (
        tuple(
            record["record_id"]
            for record in result.access_materialization.proof_records
        )
        == proof_ids
    )


def test_access_validates_a_materialization_with_a_real_applied_input_delta():
    _, context, episode, _, _ = source_records()
    package, _, _ = snapshot_and_index((episode,))
    input_delta_id = package.prepared.catalog_generation.applied_input_delta_ids[0]
    packet = PriorKnowledgeSnapshot.mint(
        source_snapshot_id=package.manifest.snapshot_id,
        query=query(context).packet_query(),
        retrieval_policy_version=package.manifest.retrieval_policy_version,
        task_context_binding_id=context.task_context_binding_id,
        selected_records=(),
        selected_record_ids=(),
        proof_reference_ids=(input_delta_id,),
        selection_metadata={},
        prompt_budget_policy=retrieval_settings().to_dict(),
        records_digest=tree_or_blob_digest(canonical_json_bytes(())),
    )
    materialization = PriorKnowledgeAccessMaterialization.mint(
        prior_knowledge_snapshot=packet,
        proof_records=(package.record_by_id(input_delta_id),),
    )

    access = PriorKnowledgeAccess(materialization)

    assert access.get_record(input_delta_id)["record_kind"] == "catalog-input-delta"


def test_projection_derivation_is_query_proof_but_unrelated_large_fact_is_not():
    _, context, episode, _, _, _, report = source_fixture()
    derivation = type(report).mint(
        schema=report.schema,
        capture_manifest_id=report.capture_manifest_id,
        scope_id=report.scope_id,
        task_family_id=report.task_family_id,
        policy_version=report.policy_version,
        policy_fingerprint=report.policy_fingerprint,
        scanner_version=report.scanner_version,
        status=report.status,
        findings=report.findings,
        excluded_paths=report.excluded_paths,
        taint_sources=report.taint_sources,
        admitted_refs={"derivation.json": digest("derivation")},
    )
    unrelated = type(report).mint(
        schema=report.schema,
        capture_manifest_id=report.capture_manifest_id,
        scope_id=report.scope_id,
        task_family_id=report.task_family_id,
        policy_version=report.policy_version,
        policy_fingerprint=report.policy_fingerprint,
        scanner_version=report.scanner_version,
        status=report.status,
        findings=report.findings,
        excluded_paths=report.excluded_paths,
        taint_sources=report.taint_sources,
        admitted_refs={f"unrelated/{'x' * 48000}.json": digest("unrelated")},
    )
    package, index, _ = snapshot_and_index(
        (episode,),
        extra_facts=(derivation, unrelated),
        projection_derivation_ids=(derivation.report_id,),
    )

    result = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context)
    )

    proof_ids = result.prior_knowledge_snapshot.proof_reference_ids
    assert result.prior_knowledge_snapshot.selected_record_ids == (episode.episode_id,)
    assert derivation.report_id in proof_ids
    assert unrelated.report_id not in proof_ids


def test_proof_references_exclude_every_selected_root_even_when_claim_uses_one():
    _, context, episode, _, claim = source_records()
    package, index, _ = snapshot_and_index((episode, claim))

    result = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context)
    )

    selected_ids = set(result.prior_knowledge_snapshot.selected_record_ids)
    proof_ids = set(result.prior_knowledge_snapshot.proof_reference_ids)
    assert selected_ids == {episode.episode_id, claim.revision_id}
    assert selected_ids.isdisjoint(proof_ids)


def test_run_diversity_cap_is_applied_after_deterministic_ranking():
    _, context, episode, prior_idea, _ = source_records()
    settings = retrieval_settings(max_records_per_run=1)
    package, index, _ = snapshot_and_index((episode, prior_idea), settings)

    result = CrossRunRetriever(
        package,
        index,
        settings,
    ).retrieve(query(context))

    assert result.prior_knowledge_snapshot.selected_record_ids == (episode.episode_id,)


def test_too_large_root_is_skipped_as_a_whole_without_partial_proof():
    _, context, episode, _, _ = source_records()
    oversized = episode_with_proposal(episode, "Intervention " + "x" * 48000)
    package, index, _ = snapshot_and_index((oversized,))

    result = CrossRunRetriever(
        package,
        index,
        retrieval_settings(),
    ).retrieve(query(context))

    assert result.prior_knowledge_snapshot.selected_records == ()
    assert result.prior_knowledge_snapshot.proof_reference_ids == ()
    assert result.access_materialization.proof_records == ()


def test_prompt_budget_counts_selection_metadata_with_proof_closed_records():
    _, context, episode, _, _ = source_records()
    package, index, _ = snapshot_and_index((episode,))
    baseline = CrossRunRetriever(package, index, retrieval_settings()).retrieve(
        query(context)
    )
    records = (
        *baseline.prior_knowledge_snapshot.selected_records,
        *baseline.access_materialization.proof_records,
    )
    record_only_budget = len(
        canonical_json_bytes(
            tuple(sorted(records, key=lambda record: record["record_id"]))
        )
    )
    settings = retrieval_settings(prompt_byte_budget=record_only_budget)
    constrained_package, constrained_index, _ = snapshot_and_index(
        (episode,),
        settings,
    )

    result = CrossRunRetriever(
        constrained_package,
        constrained_index,
        settings,
    ).retrieve(query(context))

    assert result.prior_knowledge_snapshot.selected_record_ids == ()


def test_prompt_budget_must_fit_even_the_explicit_empty_packet():
    _, context, episode, _, _ = source_records()
    settings = retrieval_settings(
        prompt_byte_budget=1,
        materialization_byte_budget=1,
    )
    package, index, _ = snapshot_and_index((episode,), settings)

    retriever = CrossRunRetriever(
        package,
        index,
        settings,
    )

    with pytest.raises(
        CrossRunRetrievalError,
        match="empty prior-knowledge packet",
    ):
        retriever.retrieve(query(context))


def test_same_query_over_same_pin_produces_identical_packet_and_materialization():
    _, context, episode, prior_idea, _ = source_records()
    package, index, _ = snapshot_and_index((episode, prior_idea))
    retriever = CrossRunRetriever(package, index, retrieval_settings())

    first = retriever.retrieve(query(context))
    second = retriever.retrieve(query(context))

    assert first.prior_knowledge_snapshot.to_json_bytes() == (
        second.prior_knowledge_snapshot.to_json_bytes()
    )
    assert first.access_materialization.to_json_bytes() == (
        second.access_materialization.to_json_bytes()
    )


def test_semantic_query_embedding_must_own_the_complete_query_text():
    _, context, _, _, _ = source_records()
    lexical_query = query(context)
    embedding = EmbeddingRecord(
        provider="openai",
        model="semantic-test",
        dimensions=2,
        canonicalizer_version="kapso.knowledge_embedding.v1",
        input_hash=complete_input_hash(lexical_query.lexical_text),
        vector=(1.0, 0.0),
    )

    bound = query(context, query_embedding=embedding)
    assert bound.packet_query()["semantic_query"] == {
        "embedding_space_id": embedding.embedding_space_id.value,
        "input_hash": embedding.input_hash,
        "query_vector_digest": tree_or_blob_digest(
            canonical_json_bytes(embedding.vector)
        ),
    }
    with pytest.raises(
        ContractValidationError,
        match="does not own the complete query text",
    ):
        query(context, query_embedding=replace(embedding, input_hash="f" * 64))
