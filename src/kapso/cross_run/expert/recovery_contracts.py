"""Strict contracts for clean-forward expert recovery."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import ClassVar

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertBaseReleaseManifest,
    ExpertScopeContract,
    PublicationArtifactKind,
    StrictContract,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.github.materializer import CacheVerificationReceipt
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    GitHubArtifactActivationWitness,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)


class ExpertRecoveryContractError(ValueError):
    """A clean-forward recovery proof is incomplete or contradictory."""


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    required: bool = True,
) -> None:
    if (required and not values) or values != tuple(sorted(set(values))):
        raise ExpertRecoveryContractError(
            f"{name} must be sorted, unique, and contain every required value"
        )
    for value in values:
        require_content_id(value, name)


def recovery_assessment_dependency_ids(
    *,
    scope_contract_id: str,
    manifest: ExpertBaseReleaseManifest,
    pointer: CurrentArtifactPointer,
    witness: GitHubArtifactActivationWitness,
    security_observation: SecurityDenylistObservation,
    release_use_observation: ExpertReleaseUsePolicyObservation,
) -> tuple[str, ...]:
    """Project every content-addressed authority used by one assessment."""

    dependencies = {
        scope_contract_id,
        manifest.release_id,
        pointer.publication_record.publication_id,
        witness.witness_id,
        security_observation.observation_id,
        security_observation.snapshot_id,
        security_observation.publication_id,
        release_use_observation.observation_id,
        release_use_observation.knowledge_snapshot_id,
        release_use_observation.knowledge_publication_id,
        *manifest.consumed_dependency_ids,
        *manifest.control_dependency_ids,
    }
    for revocation in security_observation.matched_revocations:
        dependencies.add(revocation.revocation_id)
        dependencies.update(revocation.evidence_ids)
    for revocation in release_use_observation.matched_revocations:
        dependencies.add(revocation.revocation_id)
        dependencies.update(revocation.exact_evidence_refs)
    return tuple(sorted(dependencies))


@dataclass(frozen=True)
class ExpertRecoveryReleaseAssessment(StrictContract):
    """Fresh availability assessment of one authenticated historical release."""

    assessment_id: str
    sequence_index: int
    scope_contract_id: str
    manifest: ExpertBaseReleaseManifest
    publication_pointer: CurrentArtifactPointer
    activation_witness: GitHubArtifactActivationWitness
    cache_receipt: CacheVerificationReceipt
    security_subject_ids: tuple[str, ...]
    security_observation: SecurityDenylistObservation
    release_use_observation: ExpertReleaseUsePolicyObservation
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-recovery-release-assessment"
    IDENTITY_FIELD: ClassVar[str] = "assessment_id"

    def _validate(self) -> None:
        if type(self.sequence_index) is not int or self.sequence_index < 0:
            raise ExpertRecoveryContractError(
                "recovery assessment sequence index must be non-negative"
            )
        if (
            type(self.manifest) is not ExpertBaseReleaseManifest
            or type(self.publication_pointer) is not CurrentArtifactPointer
            or type(self.activation_witness) is not GitHubArtifactActivationWitness
            or type(self.cache_receipt) is not CacheVerificationReceipt
            or type(self.security_observation) is not SecurityDenylistObservation
            or type(self.release_use_observation)
            is not ExpertReleaseUsePolicyObservation
        ):
            raise ExpertRecoveryContractError(
                "recovery assessment requires exact authenticated records"
            )
        require_content_id(
            self.scope_contract_id,
            "recovery assessment scope contract",
        )
        manifest = self.manifest
        pointer = self.publication_pointer
        publication = pointer.publication_record
        witness = self.activation_witness
        security = self.security_observation
        release_use = self.release_use_observation
        expected_security_subjects = tuple(
            sorted(
                {
                    manifest.release_id,
                    publication.publication_id,
                    witness.witness_id,
                    *manifest.consumed_dependency_ids,
                }
            )
        )
        if (
            manifest.scope_contract_id != self.scope_contract_id
            or pointer.scope_id != manifest.scope_id
            or publication.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or publication.artifact_id != manifest.release_id
            or witness.scope_id != manifest.scope_id
            or witness.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or witness.artifact_id != manifest.release_id
            or witness.repository_full_name != publication.repository_full_name
            or witness.publication_intent_digest != pointer.publication_intent_digest
            or witness.current_pointer_digest
            != tree_or_blob_digest(pointer.to_json_bytes())
            or self.cache_receipt.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or self.cache_receipt.artifact_id != manifest.release_id
            or self.cache_receipt.materialized_tree_digest
            != pointer.materialized_tree_digest
            or self.cache_receipt.manifest_relative_path
            != pointer.manifest_relative_path
            or self.cache_receipt.manifest_digest != pointer.manifest_digest
            or dict(self.cache_receipt.asset_digests)
            != {asset.name: asset.sha256 for asset in publication.assets}
            or self.security_subject_ids != expected_security_subjects
            or security.scope_id != manifest.scope_id
            or security.scope_contract_id != self.scope_contract_id
            or security.scope_repository_binding_hash
            != witness.scope_repository_binding_hash
            or security.checked_subject_ids != expected_security_subjects
            or release_use.scope_id != manifest.scope_id
            or release_use.scope_contract_id != self.scope_contract_id
            or release_use.scope_repository_binding_hash
            != witness.scope_repository_binding_hash
            or release_use.checked_release_ids != (manifest.release_id,)
        ):
            raise ExpertRecoveryContractError(
                "recovery assessment authorities do not join the release exactly"
            )
        for revocation in release_use.matched_revocations:
            if (
                revocation.release_publication_id != publication.publication_id
                or revocation.release_activation_witness_id != witness.witness_id
            ):
                raise ExpertRecoveryContractError(
                    "release-use match does not join the authenticated activation"
                )
        _require_sorted_content_ids(
            self.security_subject_ids,
            "recovery assessment security subjects",
        )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "recovery assessment exact dependencies",
        )
        expected_dependencies = recovery_assessment_dependency_ids(
            scope_contract_id=self.scope_contract_id,
            manifest=manifest,
            pointer=pointer,
            witness=witness,
            security_observation=security,
            release_use_observation=release_use,
        )
        if self.exact_dependency_ids != expected_dependencies:
            raise ExpertRecoveryContractError(
                "recovery assessment dependency closure is not exact"
            )

    @property
    def release_id(self) -> str:
        return self.manifest.release_id

    @property
    def blocked(self) -> bool:
        return bool(
            self.security_observation.matched_revocations
            or self.release_use_observation.matched_revocations
        )


@dataclass(frozen=True)
class ExpertCleanForwardRecoveryPlan(StrictContract):
    """Authenticated rollback-as-forward source and activation ordering."""

    recovery_plan_id: str
    configuration_fingerprint: str
    scope_contract: ExpertScopeContract
    current_release_observation: TaskEvaluationCurrentReleaseObservation
    assessments: tuple[ExpertRecoveryReleaseAssessment, ...]
    source_base_release_id: str | None
    source_base_tree_hash: str
    source_base_repository_map_ref: str | None
    source_base_module_contract_refs: tuple[str, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-clean-forward-recovery-plan"
    IDENTITY_FIELD: ClassVar[str] = "recovery_plan_id"

    def _validate(self) -> None:
        if (
            type(self.scope_contract) is not ExpertScopeContract
            or type(self.current_release_observation)
            is not TaskEvaluationCurrentReleaseObservation
            or type(self.assessments) is not tuple
            or not self.assessments
            or any(
                type(assessment) is not ExpertRecoveryReleaseAssessment
                for assessment in self.assessments
            )
        ):
            raise ExpertRecoveryContractError(
                "recovery plan requires exact scope, CURRENT, and assessments"
            )
        scope = self.scope_contract
        current = self.current_release_observation
        if (
            not isinstance(self.configuration_fingerprint, str)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                self.configuration_fingerprint,
            )
            is None
        ):
            raise ExpertRecoveryContractError(
                "recovery plan configuration fingerprint is invalid"
            )
        indices = tuple(assessment.sequence_index for assessment in self.assessments)
        release_ids = tuple(assessment.release_id for assessment in self.assessments)
        if (
            indices != tuple(range(len(self.assessments)))
            or len(release_ids) != len(set(release_ids))
            or current.scope_id != scope.scope_id
            or current.release_id != release_ids[0]
            or current.publication_id
            != self.assessments[0].publication_pointer.publication_record.publication_id
            or current.repository_full_name
            != self.assessments[
                0
            ].publication_pointer.publication_record.repository_full_name
            or current.repository_node_id
            != self.assessments[
                0
            ].publication_pointer.publication_record.repository_node_id
            or current.current_pointer_digest
            != tree_or_blob_digest(
                self.assessments[0].publication_pointer.to_json_bytes()
            )
            or current.validation_closure_ids
            != self.assessments[0].publication_pointer.validation_closure_ids
            or current.default_branch_head_commit_sha
            != self.assessments[0].activation_witness.activation_commit_sha
            or any(
                assessment.scope_contract_id != scope.scope_contract_id
                or assessment.manifest.scope_id != scope.scope_id
                for assessment in self.assessments
            )
        ):
            raise ExpertRecoveryContractError(
                "recovery plan CURRENT or ordered assessment identity is inconsistent"
            )
        for newer, older in zip(
            self.assessments,
            self.assessments[1:],
            strict=False,
        ):
            if (
                newer.manifest.lineage.activation_predecessor_release_id
                != older.release_id
            ):
                raise ExpertRecoveryContractError(
                    "recovery plan historical activation chain is discontinuous"
                )
        if not self.assessments[0].blocked:
            raise ExpertRecoveryContractError(
                "recovery plan CURRENT is not blocked by either policy plane"
            )
        if any(not assessment.blocked for assessment in self.assessments[:-1]):
            raise ExpertRecoveryContractError(
                "recovery plan searched past an available historical release"
            )
        selected = self.assessments[-1]
        if self.source_base_release_id is None:
            if (
                not selected.blocked
                or selected.manifest.lineage.activation_predecessor_release_id
                is not None
                or self.source_base_tree_hash != EMPTY_EXPERT_TREE_DIGEST
                or self.source_base_repository_map_ref is not None
                or self.source_base_module_contract_refs
            ):
                raise ExpertRecoveryContractError(
                    "empty recovery source requires authenticated blocked lineage exhaustion"
                )
        elif (
            selected.blocked
            or len(self.assessments) < 2
            or self.source_base_release_id != selected.release_id
            or self.source_base_tree_hash != selected.manifest.candidate_tree_hash
            or self.source_base_repository_map_ref
            != selected.manifest.repository_map_ref
            or self.source_base_module_contract_refs
            != selected.manifest.module_contract_refs
        ):
            raise ExpertRecoveryContractError(
                "historical recovery source is not the first available predecessor"
            )
        for module_contract_ref in self.source_base_module_contract_refs:
            require_content_id(
                module_contract_ref,
                "recovery source module contract",
            )
        require_identifier(scope.scope_id, "recovery plan scope")
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "recovery plan exact dependencies",
        )
        expected_dependencies = {
            scope.scope_contract_id,
            current.observation_id,
            *current.validation_closure_ids,
            *(assessment.assessment_id for assessment in self.assessments),
            *(
                dependency_id
                for assessment in self.assessments
                for dependency_id in assessment.exact_dependency_ids
            ),
            *self.source_base_module_contract_refs,
        }
        if current.publication_id is not None:
            expected_dependencies.add(current.publication_id)
        if self.source_base_release_id is not None:
            expected_dependencies.add(self.source_base_release_id)
        if self.source_base_repository_map_ref is not None:
            expected_dependencies.add(self.source_base_repository_map_ref)
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertRecoveryContractError(
                "recovery plan dependency closure is not exact"
            )

    @property
    def activation_predecessor_release_id(self) -> str:
        return self.assessments[0].release_id

    @property
    def control_dependency_ids(self) -> tuple[str, ...]:
        """Policy/history proof retained without tainting the clean source."""

        consumed = {
            self.source_base_release_id,
            *self.source_base_module_contract_refs,
        }
        if self.source_base_repository_map_ref is not None:
            consumed.add(self.source_base_repository_map_ref)
        consumed.discard(None)
        return tuple(
            dependency_id
            for dependency_id in self.exact_dependency_ids
            if dependency_id not in consumed
        )


__all__ = [
    "ExpertCleanForwardRecoveryPlan",
    "ExpertRecoveryContractError",
    "ExpertRecoveryReleaseAssessment",
    "recovery_assessment_dependency_ids",
]
