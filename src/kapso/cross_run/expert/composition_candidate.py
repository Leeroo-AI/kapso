"""Deterministic projection of a clean composition into a candidate closure."""

from __future__ import annotations

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
)
from kapso.cross_run.expert.candidate_context import (
    ExpertCandidateValidationContext,
    candidate_consumed_expert_release_ids,
    compose_candidate_replay_evidence,
)
from kapso.cross_run.expert.candidate_derivations import (
    ExpertCompositionSourceProvenance,
    ExpertDeterministicCompositionDerivation,
    ExpertDeterministicCompositionDerivationRecord,
)
from kapso.cross_run.expert.candidates import ExpertCandidateClosure
from kapso.cross_run.expert.composition import (
    ExpertCompositionReduction,
)
from kapso.cross_run.expert.composition_base import (
    ExpertCompositionBaseClosure,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionDisposition,
)
from kapso.cross_run.expert.composition_source import (
    ApprovedExpertCompositionSource,
)
from kapso.cross_run.expert.sanitation import ExpertCandidateSanitizer


class ExpertCompositionCandidateError(ValueError):
    """A clean composition cannot be projected into one exact candidate."""


def project_deterministic_composition_candidate(
    *,
    reduction: ExpertCompositionReduction,
    current_base: ExpertCompositionBaseClosure,
    approved_sources: tuple[ApprovedExpertCompositionSource, ...],
    sanitizer: ExpertCandidateSanitizer,
) -> ExpertCandidateClosure:
    """Build the sole candidate identity implied by one clean reduction."""

    if (
        type(reduction) is not ExpertCompositionReduction
        or type(current_base) is not ExpertCompositionBaseClosure
        or type(approved_sources) is not tuple
        or any(
            type(source) is not ApprovedExpertCompositionSource
            for source in approved_sources
        )
        or type(sanitizer) is not ExpertCandidateSanitizer
    ):
        raise ExpertCompositionCandidateError(
            "composition candidate projection requires exact typed inputs"
        )
    assessment = reduction.assessment
    materialization = reduction.materialization
    plan = assessment.composition_plan
    if (
        assessment.disposition is not ExpertCompositionDisposition.CLEAN
        or materialization is None
        or current_base.reference != plan.current_base
        or current_base.scope_contract != plan.scope_contract
        or tuple(source.source_reference for source in approved_sources) != plan.sources
    ):
        raise ExpertCompositionCandidateError(
            "composition candidate requires its exact clean runtime closure"
        )
    source_provenance = tuple(
        ExpertCompositionSourceProvenance(
            candidate_manifest=source.stored_candidate.closure.manifest,
            candidate_commit_record=source.stored_candidate.commit_record,
            validation_context=source.stored_candidate.closure.validation_context,
            reduction_source=source.reduction_source,
            source_base_files=source.stored_candidate.closure.source_base_files,
            agent_derivation=source.stored_candidate.closure.derivation,
            sanitation_report=source.stored_candidate.closure.sanitation_report,
        )
        for source in approved_sources
    )
    replay_evidence = compose_candidate_replay_evidence(
        tuple(provenance.validation_context for provenance in source_provenance)
    )
    consumed_expert_release_ids = candidate_consumed_expert_release_ids(
        source_base_release_id=current_base.release_manifest.release_id,
        replay_evidence=replay_evidence,
        inherited_release_ids=tuple(
            sorted(
                {
                    release_id
                    for provenance in source_provenance
                    for release_id in (
                        provenance.candidate_manifest.consumed_expert_release_ids
                    )
                }
            )
        ),
    )
    context_dependencies = tuple(
        sorted(
            {
                plan.scope_contract.scope_contract_id,
                current_base.scope_contract.scope_contract_id,
                current_base.release_manifest.release_id,
                current_base.source_base_tree_receipt.source_base_tree_receipt_id,
                current_base.source_base_tree_receipt.source_extraction_receipt.extraction_receipt_id,
                current_base.repository_map.repository_map_id,
                *(
                    module.module_contract_id
                    for module in current_base.module_contracts
                ),
                replay_evidence.replay_evidence_id,
                *replay_evidence.stable_dependency_ids,
            }
        )
    )
    validation_context = ExpertCandidateValidationContext.mint(
        scope_contract=plan.scope_contract,
        source_base_scope_contract=current_base.scope_contract,
        source_base_release=current_base.release_manifest,
        source_base_tree_receipt=current_base.source_base_tree_receipt,
        source_base_tree_hash=current_base.reference.source_tree_hash,
        source_base_repository_map=current_base.repository_map,
        source_base_module_contracts=current_base.module_contracts,
        active_task_bindings=plan.active_task_bindings,
        replay_evidence=replay_evidence,
        stable_dependency_ids=context_dependencies,
    )
    source_validation_context_ids = {
        source.source_reference.candidate_id: (
            source.stored_candidate.closure.validation_context.validation_context_id
        )
        for source in approved_sources
    }
    source_origin_principal_ids = {
        source.source_reference.candidate_id: (
            source.stored_candidate.closure.origin_principal_ids
        )
        for source in approved_sources
    }
    derivation_dependencies = tuple(
        sorted(
            {
                plan.composition_plan_id,
                *plan.stable_authority_ids,
                *source_validation_context_ids.values(),
            }
        )
    )
    derivation_record = ExpertDeterministicCompositionDerivationRecord.mint(
        composition_materialization_id=materialization.materialization_id,
        source_validation_context_ids=source_validation_context_ids,
        source_origin_principal_ids=source_origin_principal_ids,
        source_dependency_ids=derivation_dependencies,
    )
    derivation = ExpertDeterministicCompositionDerivation(
        record=derivation_record,
        materialization=materialization,
        source_provenance=source_provenance,
        source_base_contents=current_base.source_contents,
    )
    sanitation = sanitizer.scan(
        plan.scope_contract.scope_contract_id,
        materialization.source_tree,
        reduction.source_contents,
    )
    manifest = ExpertCandidateManifest.mint(
        scope_contract_id=plan.scope_contract.scope_contract_id,
        change_kind=CandidateChangeKind.CAPABILITY,
        source_base_release_id=current_base.release_manifest.release_id,
        source_base_repository_map_ref=current_base.repository_map.repository_map_id,
        source_base_tree_hash=current_base.reference.source_tree_hash,
        consumed_expert_release_ids=consumed_expert_release_ids,
        derivation_kind=(ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION),
        derivation_ref=derivation_record.derivation_id,
        validation_context_ref=validation_context.validation_context_id,
        patch_ref=materialization.patch.patch_id,
        patch_digest=tree_or_blob_digest(materialization.patch.to_json_bytes()),
        candidate_tree_ref=materialization.source_tree.source_tree_manifest_id,
        candidate_tree_hash=materialization.source_tree.tree_hash,
        configuration_fingerprint=plan.configuration_fingerprint,
        module_contract_refs=tuple(
            sorted(
                module.module_contract_id for module in materialization.module_contracts
            )
        ),
        proposed_repository_map_ref=(materialization.repository_map.repository_map_id),
        semantic_book_digest=materialization.semantic_book_digest,
        source_dependency_ids=derivation_dependencies,
        ancestor_candidate_ids=derivation_record.ancestor_candidate_ids,
        capability_lineage=(),
        sanitation_report_id=sanitation.sanitation_report_id,
    )
    return ExpertCandidateClosure(
        manifest=manifest,
        validation_context=validation_context,
        patch=materialization.patch,
        candidate_tree=materialization.source_tree,
        source_base_files=materialization.source_base_tree.files,
        repository_map=materialization.repository_map,
        module_contracts=materialization.module_contracts,
        derivation=derivation,
        sanitation_report=sanitation,
        candidate_contents=reduction.source_contents,
    )
