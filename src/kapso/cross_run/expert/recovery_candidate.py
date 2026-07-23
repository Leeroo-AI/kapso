"""Deterministic projection of an authenticated historical recovery source."""

from __future__ import annotations

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
    ExpertRecoveryRestorePatch,
    ExpertSourceTreeManifest,
)
from kapso.cross_run.expert.candidate_context import (
    ExpertCandidateValidationContext,
    candidate_consumed_expert_release_ids,
    project_recovery_replay_evidence,
)
from kapso.cross_run.expert.candidate_derivations import (
    ExpertDeterministicRecoveryRestoreDerivation,
    ExpertDeterministicRecoveryRestoreDerivationRecord,
    RECOVERY_RESTORE_PRINCIPAL_ID,
)
from kapso.cross_run.expert.candidates import ExpertCandidateClosure
from kapso.cross_run.expert.composition_base import ExpertCompositionBaseClosure
from kapso.cross_run.expert.sanitation import ExpertCandidateSanitizer
from kapso.cross_run.expert.triggers import ExpertTriggerEvidencePacket


class ExpertRecoveryCandidateError(ValueError):
    """A historical recovery source cannot produce one exact restore candidate."""


def project_historical_recovery_candidate(
    *,
    base: ExpertCompositionBaseClosure,
    replay_basis_packet: ExpertTriggerEvidencePacket,
    sanitizer: ExpertCandidateSanitizer,
) -> ExpertCandidateClosure:
    """Build the only byte-identical candidate implied by a clean source."""

    if (
        type(base) is not ExpertCompositionBaseClosure
        or type(replay_basis_packet) is not ExpertTriggerEvidencePacket
        or type(sanitizer) is not ExpertCandidateSanitizer
    ):
        raise ExpertRecoveryCandidateError(
            "recovery candidate projection requires exact typed inputs"
        )
    if (
        replay_basis_packet.scope_contract != base.scope_contract
        or replay_basis_packet.source_base_release is None
    ):
        raise ExpertRecoveryCandidateError(
            "recovery replay basis must describe the same non-empty scope"
        )
    replay_evidence = project_recovery_replay_evidence(replay_basis_packet)
    context_dependencies = tuple(
        sorted(
            {
                base.scope_contract.scope_contract_id,
                base.release_manifest.release_id,
                base.source_base_tree_receipt.source_base_tree_receipt_id,
                base.source_base_tree_receipt.source_extraction_receipt.extraction_receipt_id,
                base.repository_map.repository_map_id,
                *(module.module_contract_id for module in base.module_contracts),
                replay_evidence.replay_evidence_id,
                *replay_evidence.stable_dependency_ids,
            }
        )
    )
    validation_context = ExpertCandidateValidationContext.mint(
        scope_contract=base.scope_contract,
        source_base_scope_contract=base.scope_contract,
        source_base_release=base.release_manifest,
        source_base_tree_receipt=base.source_base_tree_receipt,
        source_base_tree_hash=base.source_tree.source_tree_hash,
        source_base_repository_map=base.repository_map,
        source_base_module_contracts=base.module_contracts,
        active_task_bindings=replay_basis_packet.active_task_bindings,
        replay_evidence=replay_evidence,
        stable_dependency_ids=context_dependencies,
    )
    derivation_record = ExpertDeterministicRecoveryRestoreDerivationRecord.mint(
        replay_basis_packet_id=replay_basis_packet.evidence_packet_id,
        source_base_release_id=base.release_manifest.release_id,
        source_base_tree_receipt_id=(
            base.source_base_tree_receipt.source_base_tree_receipt_id
        ),
        origin_principal_ids=(RECOVERY_RESTORE_PRINCIPAL_ID,),
        source_dependency_ids=context_dependencies,
    )
    derivation = ExpertDeterministicRecoveryRestoreDerivation(
        record=derivation_record,
        replay_basis_packet=replay_basis_packet,
    )
    source_tree = ExpertSourceTreeManifest.mint(
        tree_hash=base.source_tree.source_tree_hash,
        files=base.source_files,
    )
    patch = ExpertRecoveryRestorePatch.mint(
        restored_release_id=base.release_manifest.release_id,
        source_base_tree_hash=base.source_tree.source_tree_hash,
        candidate_tree_hash=source_tree.tree_hash,
        changes=(),
    )
    sanitation = sanitizer.scan(
        base.scope_contract.scope_contract_id,
        source_tree,
        base.source_contents,
    )
    consumed_release_ids = candidate_consumed_expert_release_ids(
        source_base_release_id=base.release_manifest.release_id,
        replay_evidence=replay_evidence,
        inherited_release_ids=(),
    )
    manifest = ExpertCandidateManifest.mint(
        scope_contract_id=base.scope_contract.scope_contract_id,
        change_kind=CandidateChangeKind.CAPABILITY,
        source_base_release_id=base.release_manifest.release_id,
        source_base_repository_map_ref=base.repository_map.repository_map_id,
        source_base_tree_hash=base.source_tree.source_tree_hash,
        consumed_expert_release_ids=consumed_release_ids,
        derivation_kind=(ExpertCandidateDerivationKind.DETERMINISTIC_RECOVERY_RESTORE),
        derivation_ref=derivation_record.derivation_id,
        validation_context_ref=validation_context.validation_context_id,
        patch_ref=patch.patch_id,
        patch_digest=tree_or_blob_digest(patch.to_json_bytes()),
        candidate_tree_ref=source_tree.source_tree_manifest_id,
        candidate_tree_hash=source_tree.tree_hash,
        configuration_fingerprint=replay_basis_packet.configuration_fingerprint,
        module_contract_refs=tuple(
            sorted(module.module_contract_id for module in base.module_contracts)
        ),
        proposed_repository_map_ref=base.repository_map.repository_map_id,
        semantic_book_digest=base.release_manifest.semantic_book_digest,
        source_dependency_ids=context_dependencies,
        ancestor_candidate_ids=(),
        capability_lineage=(),
        sanitation_report_id=sanitation.sanitation_report_id,
    )
    return ExpertCandidateClosure(
        manifest=manifest,
        validation_context=validation_context,
        patch=patch,
        candidate_tree=source_tree,
        source_base_files=base.source_files,
        repository_map=base.repository_map,
        module_contracts=base.module_contracts,
        derivation=derivation,
        sanitation_report=sanitation,
        candidate_contents=base.source_contents,
    )


__all__ = [
    "ExpertRecoveryCandidateError",
    "RECOVERY_RESTORE_PRINCIPAL_ID",
    "project_historical_recovery_candidate",
]
