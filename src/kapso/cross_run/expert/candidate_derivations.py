"""Closed provenance records for expert candidate construction."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CodingAgentWorkspaceDelta,
    ExpertCandidateCommitRecord,
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
    ExpertCandidateOperationRecord,
    ExpertCandidateSanitationReport,
    SourceFileDescriptor,
    StrictContract,
)
from kapso.cross_run.expert.candidate_context import (
    ExpertCandidateValidationContext,
)
from kapso.cross_run.expert.proposal_contract import ExpertCandidateAncestorInput
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionMaterialization,
)
from kapso.cross_run.expert.composition import ExpertCompositionReductionSource
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertTriggerEvidencePacket,
)


class ExpertCandidateDerivationError(ValueError):
    """Candidate derivation provenance is incomplete or inconsistent."""


CANDIDATE_MANIFEST_PACKAGE_PATH = "candidate.json"
CANDIDATE_VALIDATION_CONTEXT_PACKAGE_PATH = "validation-context.json"
AGENT_DERIVATION_RECORD_PACKAGE_PATH = "derivations/agent/derivation.json"
COMPOSITION_DERIVATION_RECORD_PACKAGE_PATH = "derivations/composition/derivation.json"
RECOVERY_RESTORE_DERIVATION_RECORD_PACKAGE_PATH = (
    "derivations/recovery-restore/derivation.json"
)
RECOVERY_RESTORE_REPLAY_BASIS_PACKAGE_PATH = (
    "derivations/recovery-restore/replay-basis.json"
)
RECOVERY_RESTORE_PRINCIPAL_ID = "kapso_clean_forward_recovery"
CANDIDATE_PATCH_PACKAGE_PATH = "patch.json"
CANDIDATE_SOURCE_TREE_PACKAGE_PATH = "source-tree.json"
CANDIDATE_REPOSITORY_MAP_PACKAGE_PATH = "repository-map.json"
CANDIDATE_MODULE_PACKAGE_ROOT = "module-contracts"
CANDIDATE_SOURCE_PACKAGE_ROOT = "source"


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertCandidateDerivationError(f"{name} uses the wrong namespace")


@dataclass(frozen=True)
class ExpertAgentProposalDerivationRecord(StrictContract):
    """Stable identity of one complete coding-agent proposal derivation."""

    derivation_id: str
    trigger_evidence_packet_id: str
    trigger_decision_id: str
    operation_record_id: str
    workspace_delta_id: str
    ancestor_candidate_ids: tuple[str, ...]
    origin_principal_ids: tuple[str, ...]
    source_dependency_ids: tuple[str, ...]
    operation_artifact_checksums: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-agent-proposal-derivation"
    IDENTITY_FIELD: ClassVar[str] = "derivation_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.trigger_evidence_packet_id,
                "expert-trigger-evidence-packet",
                "agent derivation trigger packet",
            ),
            (
                self.trigger_decision_id,
                "expert-trigger-decision",
                "agent derivation trigger decision",
            ),
            (
                self.operation_record_id,
                "expert-candidate-operation",
                "agent derivation operation",
            ),
            (
                self.workspace_delta_id,
                "coding-agent-workspace-delta",
                "agent derivation workspace delta",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        for values, name in (
            (self.ancestor_candidate_ids, "agent derivation ancestors"),
            (self.source_dependency_ids, "agent derivation source dependencies"),
        ):
            if values != tuple(sorted(set(values))):
                raise ExpertCandidateDerivationError(
                    f"{name} must be sorted and unique"
                )
            for value in values:
                require_content_id(value, name)
        if not self.source_dependency_ids:
            raise ExpertCandidateDerivationError(
                "agent derivation source dependencies must not be empty"
            )
        if not self.origin_principal_ids or self.origin_principal_ids != tuple(
            sorted(set(self.origin_principal_ids))
        ):
            raise ExpertCandidateDerivationError(
                "agent derivation origin principals must be canonical and non-empty"
            )
        for principal_id in self.origin_principal_ids:
            require_identifier(principal_id, "agent derivation origin principal")
        if not self.operation_artifact_checksums:
            raise ExpertCandidateDerivationError(
                "agent derivation artifact closure must not be empty"
            )
        for name, digest in self.operation_artifact_checksums.items():
            if (
                not name
                or not isinstance(digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            ):
                raise ExpertCandidateDerivationError(
                    "agent derivation artifact checksum is invalid"
                )


@dataclass(frozen=True)
class ExpertAgentProposalDerivation:
    """Runtime closure for one agent-authored candidate derivation."""

    record: ExpertAgentProposalDerivationRecord
    trigger_packet: ExpertTriggerEvidencePacket
    trigger_decision: ExpertEvolutionTriggerDecision
    operation: ExpertCandidateOperationRecord
    workspace_delta: CodingAgentWorkspaceDelta
    operation_artifacts: Mapping[str, bytes]
    ancestor_inputs: tuple[ExpertCandidateAncestorInput, ...]

    def __post_init__(self) -> None:
        if (
            type(self.record) is not ExpertAgentProposalDerivationRecord
            or type(self.trigger_packet) is not ExpertTriggerEvidencePacket
            or type(self.trigger_decision) is not ExpertEvolutionTriggerDecision
            or type(self.operation) is not ExpertCandidateOperationRecord
            or type(self.workspace_delta) is not CodingAgentWorkspaceDelta
            or type(self.ancestor_inputs) is not tuple
            or any(
                type(ancestor) is not ExpertCandidateAncestorInput
                for ancestor in self.ancestor_inputs
            )
            or not isinstance(self.operation_artifacts, Mapping)
        ):
            raise ExpertCandidateDerivationError(
                "agent proposal derivation requires exact typed authorities"
            )
        frozen_artifacts = MappingProxyType(dict(self.operation_artifacts))
        object.__setattr__(self, "operation_artifacts", frozen_artifacts)
        expected_artifact_checksums = {
            name: tree_or_blob_digest(payload)
            for name, payload in sorted(frozen_artifacts.items())
        }
        expected_ancestor_ids = tuple(
            ancestor.manifest.candidate_id for ancestor in self.ancestor_inputs
        )
        if (
            self.record.trigger_evidence_packet_id
            != self.trigger_packet.evidence_packet_id
            or self.record.trigger_decision_id
            != self.trigger_decision.trigger_decision_id
            or self.record.operation_record_id != self.operation.operation_record_id
            or self.record.workspace_delta_id != self.workspace_delta.workspace_delta_id
            or self.record.ancestor_candidate_ids != expected_ancestor_ids
            or self.record.origin_principal_ids
            != (self.operation.proposer_authority.principal_id,)
            or dict(self.record.operation_artifact_checksums)
            != expected_artifact_checksums
        ):
            raise ExpertCandidateDerivationError(
                "agent proposal derivation record differs from its exact closure"
            )


@dataclass(frozen=True)
class ExpertDeterministicCompositionDerivationRecord(StrictContract):
    """Stable provenance of one mechanically composed candidate."""

    derivation_id: str
    composition_materialization_id: str
    source_validation_context_ids: Mapping[str, str]
    source_origin_principal_ids: Mapping[str, tuple[str, ...]]
    source_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-deterministic-composition-derivation"
    IDENTITY_FIELD: ClassVar[str] = "derivation_id"

    def _validate(self) -> None:
        object.__setattr__(
            self,
            "source_validation_context_ids",
            MappingProxyType(dict(self.source_validation_context_ids)),
        )
        object.__setattr__(
            self,
            "source_origin_principal_ids",
            MappingProxyType(dict(self.source_origin_principal_ids)),
        )
        _require_namespaced_id(
            self.composition_materialization_id,
            "expert-composition-materialization",
            "composition derivation materialization",
        )
        validation_context_keys = tuple(sorted(self.source_validation_context_ids))
        origin_keys = tuple(sorted(self.source_origin_principal_ids))
        if not validation_context_keys or validation_context_keys != origin_keys:
            raise ExpertCandidateDerivationError(
                "composition derivation source provenance keys must be exact"
            )
        for candidate_id in validation_context_keys:
            _require_namespaced_id(
                candidate_id,
                "expert-candidate",
                "composition derivation source candidate",
            )
            _require_namespaced_id(
                self.source_validation_context_ids[candidate_id],
                "expert-candidate-validation-context",
                "composition derivation source validation context",
            )
            principal_ids = self.source_origin_principal_ids[candidate_id]
            if not principal_ids or principal_ids != tuple(sorted(set(principal_ids))):
                raise ExpertCandidateDerivationError(
                    "composition derivation source principals must be canonical "
                    "and non-empty"
                )
            for principal_id in principal_ids:
                require_identifier(
                    principal_id,
                    "composition derivation source principal",
                )
        if not self.source_dependency_ids or self.source_dependency_ids != tuple(
            sorted(set(self.source_dependency_ids))
        ):
            raise ExpertCandidateDerivationError(
                "composition derivation source dependencies must be canonical "
                "and non-empty"
            )
        for dependency_id in self.source_dependency_ids:
            require_content_id(
                dependency_id,
                "composition derivation source dependency",
            )

    @property
    def ancestor_candidate_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.source_validation_context_ids))

    @property
    def origin_principal_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    principal_id
                    for principal_ids in self.source_origin_principal_ids.values()
                    for principal_id in principal_ids
                }
            )
        )


@dataclass(frozen=True)
class ExpertCompositionSourceProvenance:
    """Commit-authenticated stable records retained for one composition source."""

    candidate_manifest: ExpertCandidateManifest
    candidate_commit_record: ExpertCandidateCommitRecord
    validation_context: ExpertCandidateValidationContext
    reduction_source: ExpertCompositionReductionSource
    source_base_files: tuple[SourceFileDescriptor, ...]
    agent_derivation: ExpertAgentProposalDerivation
    sanitation_report: ExpertCandidateSanitationReport

    def __post_init__(self) -> None:
        if (
            type(self.candidate_manifest) is not ExpertCandidateManifest
            or type(self.candidate_commit_record) is not ExpertCandidateCommitRecord
            or type(self.validation_context) is not ExpertCandidateValidationContext
            or type(self.reduction_source) is not ExpertCompositionReductionSource
            or type(self.source_base_files) is not tuple
            or any(
                type(descriptor) is not SourceFileDescriptor
                for descriptor in self.source_base_files
            )
            or type(self.agent_derivation) is not ExpertAgentProposalDerivation
            or type(self.sanitation_report) is not ExpertCandidateSanitationReport
        ):
            raise ExpertCandidateDerivationError(
                "composition source provenance requires exact agent records"
            )
        manifest = self.candidate_manifest
        commit = self.candidate_commit_record
        reduction_source = self.reduction_source
        required_checksums = {
            CANDIDATE_MANIFEST_PACKAGE_PATH: tree_or_blob_digest(
                manifest.to_json_bytes()
            ),
            CANDIDATE_VALIDATION_CONTEXT_PACKAGE_PATH: tree_or_blob_digest(
                self.validation_context.to_json_bytes()
            ),
            AGENT_DERIVATION_RECORD_PACKAGE_PATH: tree_or_blob_digest(
                self.agent_derivation.record.to_json_bytes()
            ),
            CANDIDATE_PATCH_PACKAGE_PATH: tree_or_blob_digest(
                reduction_source.patch.to_json_bytes()
            ),
            CANDIDATE_SOURCE_TREE_PACKAGE_PATH: tree_or_blob_digest(
                reduction_source.candidate_tree.to_json_bytes()
            ),
            CANDIDATE_REPOSITORY_MAP_PACKAGE_PATH: tree_or_blob_digest(
                reduction_source.repository_map.to_json_bytes()
            ),
            **{
                f"{CANDIDATE_MODULE_PACKAGE_ROOT}/"
                f"{module.module_contract_id.rsplit(':', 1)[1]}.json": (
                    tree_or_blob_digest(module.to_json_bytes())
                )
                for module in reduction_source.module_contracts
            },
            **{
                f"{CANDIDATE_SOURCE_PACKAGE_ROOT}/{path}": tree_or_blob_digest(payload)
                for path, payload in reduction_source.candidate_contents.items()
            },
        }
        if (
            commit.candidate_id != manifest.candidate_id
            or manifest.validation_context_ref
            != self.validation_context.validation_context_id
            or manifest.derivation_kind
            is not ExpertCandidateDerivationKind.AGENT_PROPOSAL
            or manifest.derivation_ref != self.agent_derivation.record.derivation_id
            or self.agent_derivation.record.origin_principal_ids
            != self.reduction_source.source_reference.origin_principal_ids
            or self.reduction_source.validation_context != self.validation_context
            or self.reduction_source.source_reference.validation_context_ref
            != self.validation_context.validation_context_id
            or manifest.sanitation_report_id
            != self.sanitation_report.sanitation_report_id
            or reduction_source.source_reference.candidate_id != manifest.candidate_id
            or reduction_source.source_reference.change_kind is not manifest.change_kind
            or reduction_source.source_reference.derivation_ref
            != manifest.derivation_ref
            or reduction_source.source_reference.validation_context_ref
            != manifest.validation_context_ref
            or reduction_source.source_reference.origin_principal_ids
            != self.derivation_record.origin_principal_ids
            or reduction_source.source_reference.candidate_configuration_fingerprint
            != manifest.configuration_fingerprint
            or reduction_source.source_reference.source_base_release_id
            != manifest.source_base_release_id
            or reduction_source.source_reference.source_base_repository_map_id
            != manifest.source_base_repository_map_ref
            or reduction_source.source_reference.source_base_tree_hash
            != manifest.source_base_tree_hash
            or reduction_source.source_reference.candidate_tree_hash
            != manifest.candidate_tree_hash
            or reduction_source.candidate_tree.source_tree_manifest_id
            != manifest.candidate_tree_ref
            or reduction_source.source_reference.patch_id != manifest.patch_ref
            or reduction_source.source_reference.patch_digest != manifest.patch_digest
            or reduction_source.source_reference.proposed_repository_map_id
            != manifest.proposed_repository_map_ref
            or reduction_source.source_reference.module_contract_ids
            != manifest.module_contract_refs
            or any(
                commit.file_checksums.get(path) != digest
                for path, digest in required_checksums.items()
            )
        ):
            raise ExpertCandidateDerivationError(
                "composition source provenance differs from its candidate commit"
            )

    @property
    def candidate_id(self) -> str:
        return self.candidate_manifest.candidate_id

    @property
    def origin_principal_ids(self) -> tuple[str, ...]:
        return self.agent_derivation.record.origin_principal_ids

    @property
    def derivation_record(self) -> ExpertAgentProposalDerivationRecord:
        return self.agent_derivation.record


@dataclass(frozen=True)
class ExpertDeterministicCompositionDerivation:
    """Runtime closure for one deterministic composition derivation."""

    record: ExpertDeterministicCompositionDerivationRecord
    materialization: ExpertCompositionMaterialization
    source_provenance: tuple[ExpertCompositionSourceProvenance, ...]
    source_base_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if (
            type(self.record) is not ExpertDeterministicCompositionDerivationRecord
            or type(self.materialization) is not ExpertCompositionMaterialization
            or type(self.source_provenance) is not tuple
            or any(
                type(provenance) is not ExpertCompositionSourceProvenance
                for provenance in self.source_provenance
            )
            or not isinstance(self.source_base_contents, Mapping)
        ):
            raise ExpertCandidateDerivationError(
                "composition derivation requires exact typed authorities"
            )
        frozen_source_base_contents = MappingProxyType(dict(self.source_base_contents))
        object.__setattr__(self, "source_base_contents", frozen_source_base_contents)
        source_base_descriptors = {
            descriptor.relative_path: descriptor
            for descriptor in self.materialization.source_base_tree.files
        }
        if set(frozen_source_base_contents) != set(source_base_descriptors):
            raise ExpertCandidateDerivationError(
                "composition derivation source-base bytes differ from its tree closure"
            )
        for path, descriptor in source_base_descriptors.items():
            payload = frozen_source_base_contents[path]
            if (
                type(payload) is not bytes
                or len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.digest
            ):
                raise ExpertCandidateDerivationError(
                    f"composition derivation source-base bytes differ: {path}"
                )
        plan = self.materialization.composition_assessment.composition_plan
        source_candidate_ids = tuple(source.candidate_id for source in plan.sources)
        provenance_candidate_ids = tuple(
            provenance.candidate_id for provenance in self.source_provenance
        )
        source_context_ids = {
            provenance.candidate_id: provenance.validation_context.validation_context_id
            for provenance in self.source_provenance
        }
        source_origin_principal_ids = {
            provenance.candidate_id: provenance.origin_principal_ids
            for provenance in self.source_provenance
        }
        source_commit_ids = {
            provenance.candidate_id: provenance.candidate_commit_record.commit_record_id
            for provenance in self.source_provenance
        }
        planned_commit_ids = {
            source.candidate_id: source.candidate_commit_record_id
            for source in plan.sources
        }
        provenance_by_candidate = {
            provenance.candidate_id: provenance for provenance in self.source_provenance
        }
        source_references_match = all(
            source.derivation_kind is ExpertCandidateDerivationKind.AGENT_PROPOSAL
            and source.derivation_ref
            == provenance_by_candidate[
                source.candidate_id
            ].derivation_record.derivation_id
            and source.validation_context_ref
            == provenance_by_candidate[
                source.candidate_id
            ].validation_context.validation_context_id
            and source.origin_principal_ids
            == provenance_by_candidate[source.candidate_id].origin_principal_ids
            for source in plan.sources
        )
        expected_dependencies = {
            plan.composition_plan_id,
            *plan.stable_authority_ids,
            *self.record.source_validation_context_ids.values(),
        }
        if (
            self.record.composition_materialization_id
            != self.materialization.materialization_id
            or provenance_candidate_ids != source_candidate_ids
            or tuple(sorted(self.record.source_validation_context_ids))
            != source_candidate_ids
            or tuple(sorted(self.record.source_origin_principal_ids))
            != source_candidate_ids
            or dict(self.record.source_validation_context_ids) != source_context_ids
            or dict(self.record.source_origin_principal_ids)
            != source_origin_principal_ids
            or source_commit_ids != planned_commit_ids
            or not source_references_match
            or set(self.record.source_dependency_ids) != expected_dependencies
        ):
            raise ExpertCandidateDerivationError(
                "composition derivation record differs from its exact materialization"
            )


@dataclass(frozen=True)
class ExpertDeterministicRecoveryRestoreDerivationRecord(StrictContract):
    """Stable provenance of one byte-identical historical restore."""

    derivation_id: str
    replay_basis_packet_id: str
    source_base_release_id: str
    source_base_tree_receipt_id: str
    origin_principal_ids: tuple[str, ...]
    source_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = (
        "expert-deterministic-recovery-restore-derivation"
    )
    IDENTITY_FIELD: ClassVar[str] = "derivation_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.replay_basis_packet_id,
                "expert-trigger-evidence-packet",
                "recovery replay basis",
            ),
            (
                self.source_base_release_id,
                "expert-base-release",
                "recovery restore source release",
            ),
            (
                self.source_base_tree_receipt_id,
                "expert-source-base-tree-receipt",
                "recovery restore source receipt",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if not self.origin_principal_ids or self.origin_principal_ids != tuple(
            sorted(set(self.origin_principal_ids))
        ):
            raise ExpertCandidateDerivationError(
                "recovery restore origin principals must be canonical and non-empty"
            )
        for principal_id in self.origin_principal_ids:
            require_identifier(
                principal_id,
                "recovery restore origin principal",
            )
        if not self.source_dependency_ids or self.source_dependency_ids != tuple(
            sorted(set(self.source_dependency_ids))
        ):
            raise ExpertCandidateDerivationError(
                "recovery restore dependencies must be canonical and non-empty"
            )
        for dependency_id in self.source_dependency_ids:
            require_content_id(
                dependency_id,
                "recovery restore dependency",
            )


@dataclass(frozen=True)
class ExpertDeterministicRecoveryRestoreDerivation:
    """Runtime closure for one deterministic historical restore."""

    record: ExpertDeterministicRecoveryRestoreDerivationRecord
    replay_basis_packet: ExpertTriggerEvidencePacket

    def __post_init__(self) -> None:
        if (
            type(self.record) is not ExpertDeterministicRecoveryRestoreDerivationRecord
            or type(self.replay_basis_packet) is not ExpertTriggerEvidencePacket
            or self.record.replay_basis_packet_id
            != self.replay_basis_packet.evidence_packet_id
        ):
            raise ExpertCandidateDerivationError(
                "recovery restore derivation differs from its replay basis"
            )


ExpertCandidateDerivationRecord = (
    ExpertAgentProposalDerivationRecord
    | ExpertDeterministicCompositionDerivationRecord
    | ExpertDeterministicRecoveryRestoreDerivationRecord
)
ExpertCandidateDerivation = (
    ExpertAgentProposalDerivation
    | ExpertDeterministicCompositionDerivation
    | ExpertDeterministicRecoveryRestoreDerivation
)
