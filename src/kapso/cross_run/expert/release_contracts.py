"""Content-addressed evidence and summaries for immutable expert releases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import (
    normalize_utc_timestamp,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertReleaseLineage,
    PublicationArtifactKind,
    StrictContract,
)
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    GitHubArtifactActivationWitness,
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ASSET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_MEDIA_TYPE_PATTERN = re.compile(
    r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+/[!#$%&'*+.^_`|~0-9A-Za-z-]+$"
)


class ExpertReleaseContractError(ValueError):
    """An expert release evidence contract is incomplete or contradictory."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertReleaseContractError(f"{name} uses the wrong namespace")


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ExpertReleaseContractError(f"{name} must be a sha256 digest")


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    required: bool = True,
) -> None:
    if (required and not values) or values != tuple(sorted(set(values))):
        raise ExpertReleaseContractError(
            f"{name} must be non-empty, sorted, and unique"
        )
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class ExpertReleaseMatrixSummary(StrictContract):
    """Public aggregate of the exact accepted release-matrix decision."""

    summary_id: str
    release_matrix_stage_result_id: str
    release_matrix_report_id: str
    promotion_decision_id: str
    mode: ExpertReleaseMatrixMode
    outcome: ExpertReleaseMatrixDecisionOutcome
    evaluation_cell_count: int
    provenance_count: int
    task_adapter_count: int

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-matrix-summary"
    IDENTITY_FIELD: ClassVar[str] = "summary_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.release_matrix_stage_result_id,
                "expert-release-matrix-stage-result",
                "matrix summary stage result",
            ),
            (
                self.release_matrix_report_id,
                "expert-release-matrix-report",
                "matrix summary report",
            ),
            (
                self.promotion_decision_id,
                "expert-release-matrix-promotion-decision",
                "matrix summary decision",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if self.outcome is not ExpertReleaseMatrixDecisionOutcome.APPROVED:
            raise ExpertReleaseContractError(
                "release matrix summary must represent an approved decision"
            )
        for value, name in (
            (self.evaluation_cell_count, "evaluation_cell_count"),
            (self.provenance_count, "provenance_count"),
            (self.task_adapter_count, "task_adapter_count"),
        ):
            if type(value) is not int or value <= 0:
                raise ExpertReleaseContractError(f"{name} must be positive")


@dataclass(frozen=True)
class ExpertReleaseEvidenceManifest(StrictContract):
    """Safe canonical record projection backing one scientific release identity."""

    evidence_manifest_id: str
    candidate_id: str
    candidate_commit_record_id: str
    candidate_tree_hash: str
    validation_attempt_id: str
    approval_transition_id: str
    approval_state_id: str
    publication_eligibility_result_id: str
    release_matrix_summary_id: str
    record_ids: tuple[str, ...]
    record_checksums: Mapping[str, str]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-evidence-manifest"
    IDENTITY_FIELD: ClassVar[str] = "evidence_manifest_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.candidate_id, "expert-candidate", "evidence candidate"),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "evidence candidate commit",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "evidence validation attempt",
            ),
            (
                self.approval_transition_id,
                "expert-validation-transition",
                "evidence approval transition",
            ),
            (
                self.approval_state_id,
                "expert-candidate-validation-state",
                "evidence approval state",
            ),
            (
                self.publication_eligibility_result_id,
                "expert-publication-eligibility-stage-result",
                "evidence publication result",
            ),
            (
                self.release_matrix_summary_id,
                "expert-release-matrix-summary",
                "evidence release matrix summary",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        _require_digest(self.candidate_tree_hash, "evidence candidate tree")
        _require_sorted_content_ids(self.record_ids, "evidence record IDs")
        if not self.record_checksums:
            raise ExpertReleaseContractError(
                "evidence manifest must checksum its projected records"
            )
        for relative_path, digest in self.record_checksums.items():
            path = PurePosixPath(relative_path)
            if (
                not relative_path
                or path.is_absolute()
                or path == PurePosixPath(".")
                or ".." in path.parts
                or path.as_posix() != relative_path
            ):
                raise ExpertReleaseContractError(
                    "evidence record checksum path is invalid"
                )
            _require_digest(digest, "evidence record checksum")
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "evidence exact dependencies",
        )
        required = {
            self.candidate_id,
            self.candidate_commit_record_id,
            self.validation_attempt_id,
            self.approval_transition_id,
            self.approval_state_id,
            self.publication_eligibility_result_id,
            self.release_matrix_summary_id,
            *self.record_ids,
        }
        if set(self.exact_dependency_ids) != required:
            raise ExpertReleaseContractError(
                "evidence manifest dependency closure is not exact"
            )


@dataclass(frozen=True)
class ExpertReleaseAssetDescriptor(StrictContract):
    """Stable identity of one immutable expert release asset."""

    name: str
    media_type: str
    size: int
    sha256: str

    def _validate(self) -> None:
        if (
            not isinstance(self.name, str)
            or _ASSET_NAME_PATTERN.fullmatch(self.name) is None
        ):
            raise ExpertReleaseContractError("release asset name is invalid")
        if (
            not isinstance(self.media_type, str)
            or _MEDIA_TYPE_PATTERN.fullmatch(self.media_type) is None
        ):
            raise ExpertReleaseContractError("release asset media type is invalid")
        if type(self.size) is not int or self.size <= 0:
            raise ExpertReleaseContractError("release asset size must be positive")
        _require_digest(self.sha256, "release asset digest")


@dataclass(frozen=True)
class ExpertReleasePublicationPlan(StrictContract):
    """Deterministic expert package and predecessor selected for publication."""

    publication_plan_id: str
    scope_contract_id: str
    scope_id: str
    release_id: str
    candidate_id: str
    candidate_tree_hash: str
    validation_attempt_id: str
    approval_transition_id: str
    approval_state_id: str
    publication_eligibility_result_id: str
    lineage: ExpertReleaseLineage
    current_release_observation: TaskEvaluationCurrentReleaseObservation
    activation_predecessor_pointer: CurrentArtifactPointer | None
    generation: int
    tag: str
    manifest_digest: str
    publication_source_tree_digest: str
    assets: tuple[ExpertReleaseAssetDescriptor, ...]
    manifest_consumed_dependency_ids: tuple[str, ...]
    manifest_control_dependency_ids: tuple[str, ...]
    validation_closure_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-publication-plan"
    IDENTITY_FIELD: ClassVar[str] = "publication_plan_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.scope_contract_id, "expert-scope-contract", "release plan scope"),
            (self.release_id, "expert-base-release", "release plan release"),
            (self.candidate_id, "expert-candidate", "release plan candidate"),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "release plan validation attempt",
            ),
            (
                self.approval_transition_id,
                "expert-validation-transition",
                "release plan approval transition",
            ),
            (
                self.approval_state_id,
                "expert-candidate-validation-state",
                "release plan approval state",
            ),
            (
                self.publication_eligibility_result_id,
                "expert-publication-eligibility-stage-result",
                "release plan publication result",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        require_identifier(self.scope_id, "release plan scope_id")
        _require_digest(self.candidate_tree_hash, "release plan candidate tree")
        _require_digest(self.manifest_digest, "release plan manifest")
        _require_digest(
            self.publication_source_tree_digest,
            "release plan publication source tree",
        )
        source_base_release_id = self.lineage.source_base_release_id
        activation_predecessor_release_id = (
            self.lineage.activation_predecessor_release_id
        )
        current = self.current_release_observation
        if (
            type(current) is not TaskEvaluationCurrentReleaseObservation
            or current.scope_id != self.scope_id
            or current.release_id != activation_predecessor_release_id
        ):
            raise ExpertReleaseContractError(
                "release plan CURRENT observation differs from its activation "
                "predecessor"
            )
        if (activation_predecessor_release_id is None) != (
            self.activation_predecessor_pointer is None
        ):
            raise ExpertReleaseContractError(
                "release plan activation predecessor pointer presence is "
                "inconsistent"
            )
        if self.activation_predecessor_pointer is not None:
            predecessor_record = self.activation_predecessor_pointer.publication_record
            if (
                self.activation_predecessor_pointer.scope_id != self.scope_id
                or predecessor_record.artifact_kind
                is not PublicationArtifactKind.EXPERT_BASE_RELEASE
                or predecessor_record.artifact_id != activation_predecessor_release_id
                or predecessor_record.publication_id != current.publication_id
                or predecessor_record.repository_full_name
                != current.repository_full_name
                or predecessor_record.repository_node_id != current.repository_node_id
                or tree_or_blob_digest(
                    self.activation_predecessor_pointer.to_json_bytes()
                )
                != current.current_pointer_digest
                or self.activation_predecessor_pointer.validation_closure_ids
                != current.validation_closure_ids
            ):
                raise ExpertReleaseContractError(
                    "release plan activation predecessor pointer differs from "
                    "CURRENT observation"
                )
        if type(self.generation) is not int or self.generation < 0:
            raise ExpertReleaseContractError(
                "release plan generation must be non-negative"
            )
        if (activation_predecessor_release_id is None) != (self.generation == 0):
            raise ExpertReleaseContractError(
                "release plan bootstrap and generation disagree"
            )
        release_digest = self.release_id.rsplit(":sha256:", 1)[1]
        if not isinstance(self.tag, str) or not self.tag.endswith(
            f"/E{self.generation:06d}-{release_digest}"
        ):
            raise ExpertReleaseContractError(
                "release plan tag differs from its generation or release identity"
            )
        if self.activation_predecessor_pointer is not None:
            predecessor_tag_match = re.fullmatch(
                r"(.*/E)([0-9]+)-([0-9a-f]{64})",
                predecessor_record.tag,
            )
            new_tag_match = re.fullmatch(
                r"(.*/E)([0-9]+)-([0-9a-f]{64})",
                self.tag,
            )
            if (
                predecessor_tag_match is None
                or new_tag_match is None
                or predecessor_tag_match.group(1) != new_tag_match.group(1)
                or int(predecessor_tag_match.group(2)) + 1 != self.generation
                or predecessor_tag_match.group(3)
                != predecessor_record.artifact_id.rsplit(":sha256:", 1)[1]
                or new_tag_match.group(3) != release_digest
            ):
                raise ExpertReleaseContractError(
                    "release plan generation is not the CURRENT successor"
                )
        asset_names = tuple(asset.name for asset in self.assets)
        if (
            not self.assets
            or any(
                type(asset) is not ExpertReleaseAssetDescriptor for asset in self.assets
            )
            or asset_names != tuple(sorted(set(asset_names)))
        ):
            raise ExpertReleaseContractError(
                "release plan assets must be non-empty, sorted, and unique"
            )
        _require_sorted_content_ids(
            self.manifest_consumed_dependency_ids,
            "release plan manifest consumed dependencies",
        )
        if source_base_release_id is not None and (
            source_base_release_id not in self.manifest_consumed_dependency_ids
        ):
            raise ExpertReleaseContractError(
                "release plan consumed dependencies omit its source base"
            )
        _require_sorted_content_ids(
            self.manifest_control_dependency_ids,
            "release plan manifest control dependencies",
            required=False,
        )
        if set(self.manifest_consumed_dependency_ids) & set(
            self.manifest_control_dependency_ids
        ):
            raise ExpertReleaseContractError(
                "release plan manifest dependency classes overlap"
            )
        if source_base_release_id == activation_predecessor_release_id:
            if self.manifest_control_dependency_ids:
                raise ExpertReleaseContractError(
                    "ordinary release plan cannot carry control dependencies"
                )
        else:
            recovery_plan_ids = {
                dependency_id
                for dependency_id in self.manifest_control_dependency_ids
                if dependency_id.split(":sha256:", 1)[0]
                == "expert-clean-forward-recovery-plan"
            }
            recovery_admission_ids = {
                dependency_id
                for dependency_id in self.manifest_control_dependency_ids
                if dependency_id.split(":sha256:", 1)[0]
                == "expert-recovery-candidate-admission"
            }
            if (
                activation_predecessor_release_id is None
                or activation_predecessor_release_id
                not in self.manifest_control_dependency_ids
                or source_base_release_id in self.manifest_control_dependency_ids
                or len(recovery_plan_ids) != 1
                or len(recovery_admission_ids) != 1
            ):
                raise ExpertReleaseContractError(
                    "recovery release plan dependency partition is invalid"
                )
        _require_sorted_content_ids(
            self.validation_closure_ids,
            "release plan validation closure",
        )
        if set(self.validation_closure_ids) != {
            self.release_id,
            *self.manifest_consumed_dependency_ids,
            *self.manifest_control_dependency_ids,
        }:
            raise ExpertReleaseContractError(
                "release plan validation closure is not exact"
            )


@dataclass(frozen=True)
class ExpertReleasePublicationIntent(StrictContract):
    """First-writer-wins timestamped reservation for one publication plan."""

    publication_intent_id: str
    publication_plan_id: str
    committed_at: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-publication-intent"
    IDENTITY_FIELD: ClassVar[str] = "publication_intent_id"

    def _validate(self) -> None:
        _require_namespaced_id(
            self.publication_plan_id,
            "expert-release-publication-plan",
            "release publication intent plan",
        )
        if normalize_utc_timestamp(self.committed_at, "committed_at") != (
            self.committed_at
        ):
            raise ExpertReleaseContractError(
                "release publication timestamp is not canonical"
            )


@dataclass(frozen=True)
class ExpertReleaseActivationReceipt(StrictContract):
    """Durable proof that one release won CURRENT at least once."""

    activation_receipt_id: str
    publication_intent_id: str
    publication_plan_id: str
    release_id: str
    candidate_id: str
    approval_transition_id: str
    approval_state_id: str
    planned_current_observation_id: str
    github_publication_intent: ArtifactPublicationIntent
    github_publication_pointer: CurrentArtifactPointer
    activation_witness: GitHubArtifactActivationWitness
    observed_current_release: TaskEvaluationCurrentReleaseObservation
    consumed_dependency_ids: tuple[str, ...]
    control_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-activation-receipt"
    IDENTITY_FIELD: ClassVar[str] = "activation_receipt_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.publication_intent_id,
                "expert-release-publication-intent",
                "activation receipt publication intent",
            ),
            (
                self.publication_plan_id,
                "expert-release-publication-plan",
                "activation receipt publication plan",
            ),
            (self.release_id, "expert-base-release", "activation receipt release"),
            (self.candidate_id, "expert-candidate", "activation receipt candidate"),
            (
                self.approval_transition_id,
                "expert-validation-transition",
                "activation receipt approval transition",
            ),
            (
                self.approval_state_id,
                "expert-candidate-validation-state",
                "activation receipt approval state",
            ),
            (
                self.planned_current_observation_id,
                "task-evaluation-current-release-observation",
                "activation receipt planned CURRENT",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        intent = self.github_publication_intent
        pointer = self.github_publication_pointer
        observed = self.observed_current_release
        witness = self.activation_witness
        if (
            type(intent) is not ArtifactPublicationIntent
            or type(pointer) is not CurrentArtifactPointer
            or not intent.binds(pointer)
            or intent.scope_id != observed.scope_id
            or intent.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or intent.artifact_id != self.release_id
            or intent.repository_full_name != observed.repository_full_name
            or pointer.publication_record.artifact_id != self.release_id
        ):
            raise ExpertReleaseContractError(
                "activation receipt GitHub publication is inconsistent"
            )
        if (
            type(witness) is not GitHubArtifactActivationWitness
            or witness.scope_id != intent.scope_id
            or witness.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or witness.artifact_id != self.release_id
            or witness.repository_full_name != intent.repository_full_name
            or witness.publication_intent_digest != intent.digest
            or witness.current_pointer_digest
            != tree_or_blob_digest(pointer.to_json_bytes())
        ):
            raise ExpertReleaseContractError(
                "activation receipt witness does not prove activation"
            )
        if observed.release_id == self.release_id and (
            observed.publication_id != pointer.publication_record.publication_id
            or observed.current_pointer_digest
            != tree_or_blob_digest(pointer.to_json_bytes())
            or observed.default_branch_head_commit_sha != witness.activation_commit_sha
        ):
            raise ExpertReleaseContractError(
                "active activation receipt differs from its CURRENT pointer"
            )
        if (
            observed.release_id != self.release_id
            and observed.publication_id == pointer.publication_record.publication_id
        ):
            raise ExpertReleaseContractError(
                "historical activation receipt reuses its publication identity"
            )
        for values, name in (
            (
                self.consumed_dependency_ids,
                "activation receipt consumed dependencies",
            ),
            (
                self.control_dependency_ids,
                "activation receipt control dependencies",
            ),
        ):
            _require_sorted_content_ids(values, name)
        if set(self.consumed_dependency_ids) & set(self.control_dependency_ids):
            raise ExpertReleaseContractError(
                "activation receipt dependency classes overlap"
            )
        required_consumed = {
            self.publication_intent_id,
            self.publication_plan_id,
            self.release_id,
            self.candidate_id,
            self.approval_transition_id,
            self.approval_state_id,
            pointer.publication_record.publication_id,
            witness.witness_id,
        }
        required = {
            *required_consumed,
            self.planned_current_observation_id,
            observed.observation_id,
            *intent.validation_closure_ids,
            *observed.validation_closure_ids,
        }
        if observed.release_id is not None:
            required.add(observed.release_id)
        if observed.publication_id is not None:
            required.add(observed.publication_id)
        required_control = {
            observed.observation_id,
            *observed.validation_closure_ids,
        }
        if observed.release_id is not None:
            required_control.add(observed.release_id)
        if observed.publication_id is not None:
            required_control.add(observed.publication_id)
        required_control.difference_update(required_consumed)
        required_control.difference_update(intent.validation_closure_ids)
        if (
            not required_consumed.issubset(self.consumed_dependency_ids)
            or not required_control.issubset(self.control_dependency_ids)
            or set(self.consumed_dependency_ids) | set(self.control_dependency_ids)
            != required
        ):
            raise ExpertReleaseContractError(
                "activation receipt categorized dependency closure is not exact"
            )


@dataclass(frozen=True)
class ExpertReleasePublicationStaleResolution(StrictContract):
    """Durable proof that a reserved publication lost its CURRENT authority."""

    stale_resolution_id: str
    publication_intent_id: str
    publication_plan_id: str
    release_id: str
    candidate_id: str
    approval_transition_id: str
    approval_state_id: str
    planned_current_observation_id: str
    observed_current_release: TaskEvaluationCurrentReleaseObservation
    observed_current_activation_witness: GitHubArtifactActivationWitness
    own_github_publication_intent: ArtifactPublicationIntent | None
    own_github_publication_pointer: CurrentArtifactPointer | None
    own_github_activation_preparation_commit_sha: str | None
    resolved_at: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-publication-stale-resolution"
    IDENTITY_FIELD: ClassVar[str] = "stale_resolution_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.publication_intent_id,
                "expert-release-publication-intent",
                "stale resolution intent",
            ),
            (
                self.publication_plan_id,
                "expert-release-publication-plan",
                "stale resolution plan",
            ),
            (self.release_id, "expert-base-release", "stale resolution release"),
            (self.candidate_id, "expert-candidate", "stale resolution candidate"),
            (
                self.approval_transition_id,
                "expert-validation-transition",
                "stale resolution approval transition",
            ),
            (
                self.approval_state_id,
                "expert-candidate-validation-state",
                "stale resolution approval state",
            ),
            (
                self.planned_current_observation_id,
                "task-evaluation-current-release-observation",
                "stale resolution planned CURRENT",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if type(self.observed_current_release) is not (
            TaskEvaluationCurrentReleaseObservation
        ):
            raise ExpertReleaseContractError(
                "stale resolution requires an exact CURRENT observation"
            )
        if self.observed_current_release.observation_id == (
            self.planned_current_observation_id
        ):
            raise ExpertReleaseContractError(
                "stale resolution must observe changed CURRENT authority"
            )
        if (
            self.observed_current_release.release_id is None
            or self.observed_current_release.release_id == self.release_id
        ):
            raise ExpertReleaseContractError(
                "stale resolution requires another active release"
            )
        own_intent = self.own_github_publication_intent
        own_pointer = self.own_github_publication_pointer
        preparation_commit = self.own_github_activation_preparation_commit_sha
        winner_witness = self.observed_current_activation_witness
        observed = self.observed_current_release
        if (
            type(winner_witness) is not GitHubArtifactActivationWitness
            or winner_witness.scope_id != observed.scope_id
            or winner_witness.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or winner_witness.artifact_id != observed.release_id
            or winner_witness.repository_full_name != observed.repository_full_name
            or winner_witness.activation_commit_sha
            != observed.default_branch_head_commit_sha
            or winner_witness.current_pointer_digest != observed.current_pointer_digest
        ):
            raise ExpertReleaseContractError(
                "stale resolution winner lacks its activation witness"
            )
        if own_pointer is not None and own_intent is None:
            raise ExpertReleaseContractError(
                "stale resolution own pointer lacks its GitHub publication intent"
            )
        if own_intent is not None and (
            type(own_intent) is not ArtifactPublicationIntent
            or own_intent.scope_id != self.observed_current_release.scope_id
            or own_intent.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or own_intent.artifact_id != self.release_id
            or own_intent.repository_full_name
            != self.observed_current_release.repository_full_name
        ):
            raise ExpertReleaseContractError(
                "stale resolution own GitHub intent is inconsistent"
            )
        if own_pointer is not None and (
            type(own_pointer) is not CurrentArtifactPointer
            or not own_intent.binds(own_pointer)
            or own_pointer.publication_record.artifact_id != self.release_id
        ):
            raise ExpertReleaseContractError(
                "stale resolution own GitHub pointer is inconsistent"
            )
        if preparation_commit is not None and (
            own_pointer is None
            or not re.fullmatch(r"[0-9a-f]{40}", preparation_commit)
            or preparation_commit == winner_witness.activation_commit_sha
        ):
            raise ExpertReleaseContractError(
                "stale resolution activation preparation is inconsistent"
            )
        normalize_utc_timestamp(self.resolved_at, "resolved_at")
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "stale resolution exact dependencies",
        )
        required = {
            self.publication_intent_id,
            self.publication_plan_id,
            self.release_id,
            self.candidate_id,
            self.approval_transition_id,
            self.approval_state_id,
            self.planned_current_observation_id,
            observed.observation_id,
            winner_witness.witness_id,
            *observed.validation_closure_ids,
        }
        if observed.release_id is not None:
            required.add(observed.release_id)
        if observed.publication_id is not None:
            required.add(observed.publication_id)
        if own_pointer is not None:
            required.add(own_pointer.publication_record.publication_id)
        if set(self.exact_dependency_ids) != required:
            raise ExpertReleaseContractError(
                "stale resolution dependency closure is not exact"
            )


__all__ = [
    "ExpertReleaseAssetDescriptor",
    "ExpertReleaseContractError",
    "ExpertReleaseActivationReceipt",
    "ExpertReleaseEvidenceManifest",
    "ExpertReleaseMatrixSummary",
    "ExpertReleasePublicationIntent",
    "ExpertReleasePublicationPlan",
    "ExpertReleasePublicationStaleResolution",
]
