"""Strict transaction contracts for cross-run launch resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    normalize_utc_timestamp,
    require_content_id,
    require_identifier,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    CrossRunTaskBindingSettings,
    ExpertBaseReleaseManifest,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    ScopeRepositorySettings,
    SourceFileDescriptor,
    StrictContract,
    TaskAdapterManifest,
    TaskContextBinding,
)
from kapso.cross_run.embedding_space import EmbeddingSpace
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.github.materializer import (
    CacheVerificationReceipt,
    SourceArchiveExtractionReceipt,
)
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    GitHubArtifactActivationWitness,
)
from kapso.cross_run.git_refs import require_git_ref_name
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.task_adapters import (
    TaskAdapterActivationRecord,
    TaskAdapterVerificationReceipt,
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_REPOSITORY_KEYS = {
    "expert_repository",
    "knowledge_repository",
    "repositories",
    "security_repository",
}


class LaunchContractError(ContractValidationError):
    """A launch request or resolved launch authority is not exact."""


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise LaunchContractError(f"{name} must be a sha256 digest")


def _require_sorted_unique(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise LaunchContractError(f"{name} must be sorted and unique")


def _require_relative_path(value: str, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise LaunchContractError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise LaunchContractError(f"{name} must be normalized and relative")
    return path


def _reject_repository_routing(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_REPOSITORY_KEYS & set(value)
        if forbidden:
            raise LaunchContractError(
                f"launch request cannot override repository routing at {path}: "
                f"{tuple(sorted(forbidden))}"
            )
        for key, child in value.items():
            _reject_repository_routing(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, child in enumerate(value):
            _reject_repository_routing(child, f"{path}[{position}]")


def expected_launch_source_composition_hash(
    *,
    expert_source_tree_hash: str,
    expert_repository_map: ExpertRepositoryMap,
    task_adapter: "LaunchTaskAdapterPin",
    starting_artifacts: "LaunchStartingArtifactMaterializationReceipt",
) -> str:
    """Hash the exact source identities and their workspace composition boundary."""

    _require_digest(expert_source_tree_hash, "expert source tree hash")
    if type(expert_repository_map) is not ExpertRepositoryMap:
        raise LaunchContractError("source composition requires one repository map")
    if type(task_adapter) is not LaunchTaskAdapterPin:
        raise LaunchContractError("source composition requires one task adapter pin")
    if type(starting_artifacts) is not LaunchStartingArtifactMaterializationReceipt:
        raise LaunchContractError(
            "source composition requires one starting-artifact receipt"
        )
    runtime_files = tuple(
        descriptor
        for descriptor in task_adapter.source_extraction_receipt.source_tree_files
        if PurePosixPath(descriptor.relative_path).parts[0] != "release_matrix_assets"
    )
    runtime_tree_hash = source_tree_digest(
        {
            descriptor.relative_path: (
                descriptor.digest,
                descriptor.mode,
                descriptor.size,
            )
            for descriptor in runtime_files
        }
    )
    return tree_or_blob_digest(
        canonical_json_bytes(
            {
                "expert_source_tree_hash": expert_source_tree_hash,
                "task_adapter_mount_path": (
                    expert_repository_map.task_adapter_boundary.adapter_mount_path
                ),
                "task_adapter_runtime_tree_hash": runtime_tree_hash,
                "starting_artifacts": tuple(
                    {
                        "starting_artifact_content_id": (
                            artifact.starting_artifact_content_id
                        ),
                        "starting_artifact_ref": artifact.starting_artifact_ref,
                        "mount_path": artifact.mount_path,
                        "materialized_tree_hash": artifact.materialized_tree_hash,
                    }
                    for artifact in starting_artifacts.starting_artifacts
                ),
            }
        )
    )


@dataclass(frozen=True)
class LaunchTaskContextRequest(StrictContract):
    """Task semantics known before the current scope contract is resolved."""

    task_context_request_id: str
    capability_tags: tuple[str, ...]
    input_contract_fingerprint: str
    target_contract_fingerprint: str
    starting_artifact_refs: tuple[str, ...]
    method_fingerprint: str
    toolchain_fingerprint: str
    dependency_runtime_fingerprint: str
    budget_hardware_envelope: Mapping[str, Any]
    transfer_dimensions: Mapping[str, Any]

    CONTENT_NAMESPACE: ClassVar[str] = "launch-task-context-request"
    IDENTITY_FIELD: ClassVar[str] = "task_context_request_id"

    def _validate(self) -> None:
        _require_sorted_unique(self.capability_tags, "launch capability_tags")
        for capability_tag in self.capability_tags:
            require_identifier(capability_tag, "launch capability tag")
        _require_sorted_unique(
            self.starting_artifact_refs,
            "launch starting_artifact_refs",
        )
        for artifact_ref in self.starting_artifact_refs:
            if not isinstance(artifact_ref, str) or not artifact_ref.strip():
                raise LaunchContractError(
                    "launch starting artifact reference must be non-empty text"
                )
        for value, name in (
            (self.input_contract_fingerprint, "input_contract_fingerprint"),
            (self.target_contract_fingerprint, "target_contract_fingerprint"),
            (self.method_fingerprint, "method_fingerprint"),
            (self.toolchain_fingerprint, "toolchain_fingerprint"),
            (
                self.dependency_runtime_fingerprint,
                "dependency_runtime_fingerprint",
            ),
        ):
            _require_digest(value, f"launch task context {name}")
        if not self.budget_hardware_envelope:
            raise LaunchContractError(
                "launch budget_hardware_envelope must not be empty"
            )
        _reject_repository_routing(
            self.budget_hardware_envelope,
            "task_context.budget_hardware_envelope",
        )
        _reject_repository_routing(
            self.transfer_dimensions,
            "task_context.transfer_dimensions",
        )

    def bind(
        self,
        *,
        binding: CrossRunTaskBindingSettings,
        scope_contract: ExpertScopeContract,
    ) -> TaskContextBinding:
        scope_contract.validate_binding(binding)
        task_context = TaskContextBinding.mint(
            scope_contract_id=scope_contract.scope_contract_id,
            scope_id=binding.scope_id,
            task_family_id=binding.task_family_id,
            task_adapter_id=binding.task_adapter_id,
            capability_tags=self.capability_tags,
            input_contract_fingerprint=self.input_contract_fingerprint,
            target_contract_fingerprint=self.target_contract_fingerprint,
            starting_artifact_refs=self.starting_artifact_refs,
            method_fingerprint=self.method_fingerprint,
            toolchain_fingerprint=self.toolchain_fingerprint,
            dependency_runtime_fingerprint=self.dependency_runtime_fingerprint,
            budget_hardware_envelope=self.budget_hardware_envelope,
            transfer_dimensions=self.transfer_dimensions,
        )
        task_context.validate_against(scope_contract)
        return task_context


@dataclass(frozen=True)
class LaunchRequest(StrictContract):
    """Complete repository-free request supplied to the trusted resolver."""

    launch_request_id: str
    binding: CrossRunTaskBindingSettings
    task_context_request: LaunchTaskContextRequest
    goal_digest: str
    starting_artifact_content_ids: Mapping[str, str]
    requested_coding_agent: str
    search_mode: str
    dependency_runtime_contract: Mapping[str, Any]
    budget_fidelity_envelope: Mapping[str, Any]
    configuration_fingerprint: str
    empty_scope_bootstrap_authorization_id: str | None

    CONTENT_NAMESPACE: ClassVar[str] = "launch-request"
    IDENTITY_FIELD: ClassVar[str] = "launch_request_id"

    def _validate(self) -> None:
        if type(self.binding) is not CrossRunTaskBindingSettings:
            raise LaunchContractError("launch binding uses another contract")
        if type(self.task_context_request) is not LaunchTaskContextRequest:
            raise LaunchContractError("launch task context uses another contract")
        for value, name in (
            (self.goal_digest, "goal_digest"),
            (self.configuration_fingerprint, "configuration_fingerprint"),
        ):
            _require_digest(value, f"launch request {name}")
        for value, name in (
            (self.requested_coding_agent, "requested_coding_agent"),
            (self.search_mode, "search_mode"),
        ):
            require_identifier(value, f"launch request {name}")
        if not self.dependency_runtime_contract or not self.budget_fidelity_envelope:
            raise LaunchContractError(
                "launch runtime and budget/fidelity envelopes must not be empty"
            )
        expected_refs = set(self.task_context_request.starting_artifact_refs)
        if set(self.starting_artifact_content_ids) != expected_refs:
            raise LaunchContractError(
                "launch starting artifacts differ from task context requirements"
            )
        artifact_ids = tuple(self.starting_artifact_content_ids.values())
        if len(artifact_ids) != len(set(artifact_ids)):
            raise LaunchContractError(
                "launch starting artifact references must name distinct content"
            )
        for artifact_ref, artifact_id in self.starting_artifact_content_ids.items():
            if not isinstance(artifact_ref, str) or not artifact_ref.strip():
                raise LaunchContractError(
                    "launch starting artifact key must be non-empty text"
                )
            require_content_id(artifact_id, "launch starting artifact content ID")
        if self.empty_scope_bootstrap_authorization_id is not None:
            require_content_id(
                self.empty_scope_bootstrap_authorization_id,
                "empty scope bootstrap authorization",
            )
            if (
                self.empty_scope_bootstrap_authorization_id.split(":sha256:", 1)[0]
                != "scope-bootstrap-authorization"
            ):
                raise LaunchContractError(
                    "empty scope bootstrap authorization uses the wrong namespace"
                )
        _reject_repository_routing(
            self.dependency_runtime_contract,
            "launch_request.dependency_runtime_contract",
        )
        _reject_repository_routing(
            self.budget_fidelity_envelope,
            "launch_request.budget_fidelity_envelope",
        )

    @property
    def request_hash(self) -> str:
        return tree_or_blob_digest(self.to_json_bytes())


@dataclass(frozen=True)
class LaunchStartingArtifact(StrictContract):
    """One content-addressed task input staged into the launch workspace."""

    starting_artifact_content_id: str
    starting_artifact_ref: str
    mount_path: str
    materialized_tree_hash: str
    source_files: tuple[SourceFileDescriptor, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "launch-starting-artifact"
    IDENTITY_FIELD: ClassVar[str] = "starting_artifact_content_id"

    def _validate(self) -> None:
        if not isinstance(self.starting_artifact_ref, str) or not (
            self.starting_artifact_ref.strip()
        ):
            raise LaunchContractError(
                "launch starting-artifact reference must be non-empty text"
            )
        mount_path = PurePosixPath(self.mount_path)
        if (
            not self.mount_path
            or mount_path.is_absolute()
            or ".." in mount_path.parts
            or mount_path == PurePosixPath(".")
            or mount_path.as_posix() != self.mount_path
        ):
            raise LaunchContractError(
                "launch starting-artifact mount must be normalized and relative"
            )
        _require_digest(
            self.materialized_tree_hash,
            "launch starting-artifact tree hash",
        )
        paths = tuple(descriptor.relative_path for descriptor in self.source_files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise LaunchContractError(
                "launch starting-artifact files must be non-empty, sorted, and unique"
            )
        source_paths = tuple(PurePosixPath(path) for path in paths)
        if any(
            source_path in other_path.parents
            for position, source_path in enumerate(source_paths)
            for other_path in source_paths[position + 1 :]
        ):
            raise LaunchContractError(
                "launch starting-artifact files contain a path collision"
            )
        expected_tree_hash = source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
                for descriptor in self.source_files
            }
        )
        if self.materialized_tree_hash != expected_tree_hash:
            raise LaunchContractError(
                "launch starting-artifact tree differs from its file closure"
            )


@dataclass(frozen=True)
class LaunchStartingArtifactMaterializationReceipt(StrictContract):
    """Exact provider receipt for the task-input bytes admitted into launch."""

    materialization_receipt_id: str
    task_context_binding_id: str
    starting_artifacts: tuple[LaunchStartingArtifact, ...]
    materializer_id: str
    materializer_version: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "launch-starting-artifact-materialization"
    IDENTITY_FIELD: ClassVar[str] = "materialization_receipt_id"

    def _validate(self) -> None:
        require_content_id(
            self.task_context_binding_id,
            "launch starting-artifact task_context_binding_id",
        )
        if self.task_context_binding_id.split(":sha256:", 1)[0] != (
            "task-context-binding"
        ):
            raise LaunchContractError(
                "launch starting-artifact receipt must name a TaskContextBinding"
            )
        artifact_ids = tuple(
            artifact.starting_artifact_content_id
            for artifact in self.starting_artifacts
        )
        if artifact_ids != tuple(sorted(set(artifact_ids))):
            raise LaunchContractError(
                "launch starting artifacts must be ID-sorted and unique"
            )
        artifact_refs = tuple(
            artifact.starting_artifact_ref for artifact in self.starting_artifacts
        )
        if len(artifact_refs) != len(set(artifact_refs)):
            raise LaunchContractError(
                "launch starting-artifact references must be unique"
            )
        mount_paths = tuple(
            PurePosixPath(artifact.mount_path) for artifact in self.starting_artifacts
        )
        if len(mount_paths) != len(set(mount_paths)) or any(
            left in right.parents or right in left.parents
            for position, left in enumerate(mount_paths)
            for right in mount_paths[position + 1 :]
        ):
            raise LaunchContractError("launch starting-artifact mount paths overlap")
        require_identifier(
            self.materializer_id,
            "launch starting-artifact materializer_id",
        )
        require_identifier(
            self.materializer_version,
            "launch starting-artifact materializer_version",
        )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "launch starting-artifact exact_dependency_ids",
        )
        if set(self.exact_dependency_ids) != {
            self.task_context_binding_id,
            *artifact_ids,
        }:
            raise LaunchContractError(
                "launch starting-artifact dependency closure is not exact"
            )


class LaunchCompatibilityAdmissionMode(str, Enum):
    """Structural admission algorithms selected by the configured policy version."""

    VERIFIED_CASE_NEW_ARTIFACT_CONTENT = "verified_case_new_artifact_content"


@dataclass(frozen=True)
class LaunchGitHubArtifactPin(StrictContract):
    """Exact immutable publication and verified local cache authority."""

    component_pin_id: str
    scope_id: str
    scope_repository_binding_hash: str
    pointer: CurrentArtifactPointer
    publication_intent: ArtifactPublicationIntent
    authority_commit_sha: str
    activation_witness: GitHubArtifactActivationWitness
    cache_receipt: CacheVerificationReceipt

    CONTENT_NAMESPACE: ClassVar[str] = "launch-github-artifact-pin"
    IDENTITY_FIELD: ClassVar[str] = "component_pin_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "launch component scope_id")
        for value, name in (
            (
                self.scope_repository_binding_hash,
                "scope_repository_binding_hash",
            ),
        ):
            _require_digest(value, f"launch component {name}")
        if _COMMIT_PATTERN.fullmatch(self.authority_commit_sha) is None:
            raise LaunchContractError(
                "launch component authority_commit_sha is invalid"
            )
        pointer = self.pointer
        intent = self.publication_intent
        publication = pointer.publication_record
        witness = self.activation_witness
        receipt = self.cache_receipt
        if (
            type(pointer) is not CurrentArtifactPointer
            or type(intent) is not ArtifactPublicationIntent
            or type(witness) is not GitHubArtifactActivationWitness
            or type(receipt) is not CacheVerificationReceipt
            or pointer.scope_id != self.scope_id
            or intent.scope_id != self.scope_id
            or not intent.binds(pointer)
            or witness.scope_id != self.scope_id
            or witness.scope_repository_binding_hash
            != self.scope_repository_binding_hash
            or witness.artifact_kind is not publication.artifact_kind
            or witness.artifact_id != publication.artifact_id
            or witness.repository_full_name != publication.repository_full_name
            or witness.publication_intent_digest != intent.digest
            or witness.current_pointer_digest != self.current_pointer_digest
            or receipt.artifact_kind is not publication.artifact_kind
            or receipt.artifact_id != publication.artifact_id
            or receipt.materialized_tree_digest != pointer.materialized_tree_digest
            or receipt.manifest_relative_path != pointer.manifest_relative_path
            or receipt.manifest_digest != pointer.manifest_digest
            or dict(receipt.asset_digests)
            != {asset.name: asset.sha256 for asset in publication.assets}
        ):
            raise LaunchContractError(
                "launch GitHub component authorities do not join exactly"
            )

    @property
    def artifact_kind(self) -> PublicationArtifactKind:
        return self.publication.artifact_kind

    @property
    def artifact_id(self) -> str:
        return self.publication.artifact_id

    @property
    def publication(self):
        return self.pointer.publication_record

    @property
    def publication_intent_digest(self) -> str:
        return self.publication_intent.digest

    @property
    def current_pointer_digest(self) -> str:
        return tree_or_blob_digest(self.pointer.to_json_bytes())

    @property
    def manifest_digest(self) -> str:
        return self.pointer.manifest_digest


@dataclass(frozen=True)
class LaunchExpertSourcePin(StrictContract):
    source_pin_id: str
    expert_release_id: str
    extraction_receipt: SourceArchiveExtractionReceipt

    CONTENT_NAMESPACE: ClassVar[str] = "launch-expert-source-pin"
    IDENTITY_FIELD: ClassVar[str] = "source_pin_id"

    def _validate(self) -> None:
        require_content_id(self.expert_release_id, "launch expert release")
        if (
            type(self.extraction_receipt) is not SourceArchiveExtractionReceipt
            or self.extraction_receipt.artifact_id != self.expert_release_id
        ):
            raise LaunchContractError(
                "launch expert source extraction names another release"
            )


@dataclass(frozen=True)
class LaunchTaskAdapterPin(StrictContract):
    adapter_pin_id: str
    activation: TaskAdapterActivationRecord
    manifest: TaskAdapterManifest
    verification_receipt: TaskAdapterVerificationReceipt
    source_extraction_receipt: SourceArchiveExtractionReceipt

    CONTENT_NAMESPACE: ClassVar[str] = "launch-task-adapter-pin"
    IDENTITY_FIELD: ClassVar[str] = "adapter_pin_id"

    def _validate(self) -> None:
        activation = self.activation
        manifest = self.manifest
        receipt = self.verification_receipt
        extraction = self.source_extraction_receipt
        if (
            type(activation) is not TaskAdapterActivationRecord
            or type(manifest) is not TaskAdapterManifest
            or type(receipt) is not TaskAdapterVerificationReceipt
            or type(extraction) is not SourceArchiveExtractionReceipt
            or activation.scope_contract_id != manifest.scope_contract_id
            or activation.task_family_id != manifest.task_family_id
            or activation.task_adapter_id != manifest.task_adapter_id
            or activation.task_adapter_manifest_id != manifest.task_adapter_manifest_id
            or activation.verification_receipt_id != receipt.verification_receipt_id
            or receipt.task_adapter_manifest_id != manifest.task_adapter_manifest_id
            or receipt.full_manifest_digest
            != tree_or_blob_digest(manifest.to_json_bytes())
            or receipt.publisher_attestation_digest
            != tree_or_blob_digest(canonical_json_bytes(manifest.publisher_attestation))
            or receipt.source_extraction_receipt_id != extraction.extraction_receipt_id
            or receipt.source_archive_ref != manifest.source_tree_ref
            or receipt.source_archive_ref != extraction.source_archive_ref
            or receipt.source_archive_digest != extraction.source_archive_digest
            or extraction.artifact_id != manifest.task_adapter_manifest_id
            or extraction.source_archive_ref != manifest.source_tree_ref
            or extraction.source_tree_hash != manifest.tree_hash
            or receipt.source_tree_hash != manifest.tree_hash
            or set(receipt.proof_object_digests)
            != {manifest.sanitation_report_id, *manifest.validation_refs}
        ):
            raise LaunchContractError(
                "launch task adapter authorities do not join exactly"
            )

    @property
    def dependency_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.verification_receipt.verification_receipt_id,
                    self.verification_receipt.source_extraction_receipt_id,
                    self.manifest.sanitation_report_id,
                    *self.verification_receipt.proof_object_ids,
                }
            )
        )


@dataclass(frozen=True)
class LaunchCompatibilityPolicy(StrictContract):
    """Content-addressed policy whose code version admits one launch tuple."""

    compatibility_policy_id: str
    policy_version: str
    admission_mode: LaunchCompatibilityAdmissionMode
    artifact_ttl_seconds: int

    CONTENT_NAMESPACE: ClassVar[str] = "launch-compatibility-policy"
    IDENTITY_FIELD: ClassVar[str] = "compatibility_policy_id"

    def _validate(self) -> None:
        require_identifier(self.policy_version, "launch compatibility policy version")
        if type(self.artifact_ttl_seconds) is not int or self.artifact_ttl_seconds < 1:
            raise LaunchContractError(
                "launch compatibility artifact TTL must be positive"
            )


@dataclass(frozen=True)
class LaunchCompatibilityReceipt(StrictContract):
    compatibility_receipt_id: str
    policy: LaunchCompatibilityPolicy
    launch_request_id: str
    task_context_binding_id: str
    scope_contract_id: str
    expert_component_pin_id: str
    expert_release_id: str
    knowledge_component_pin_id: str
    knowledge_snapshot_id: str
    task_adapter_pin_id: str
    task_adapter_manifest_id: str
    task_adapter_verification_receipt_id: str
    task_adapter_activation_id: str
    knowledge_embedding_space_id: str
    release_use_observation_id: str
    expert_validation_context_id: str
    expert_repository_map_id: str
    expert_module_contract_ids: tuple[str, ...]
    expert_release_matrix_stage_result_id: str
    expert_release_matrix_report_id: str
    expert_release_matrix_adapter_authority_id: str
    task_adapter_compatibility_case_ids: tuple[str, ...]
    starting_artifact_materialization_receipt_id: str
    starting_artifact_content_ids: tuple[str, ...]
    runtime_contract_digest: str
    source_composition_hash: str
    resolved_at: str
    compatible: bool
    reason_code: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "launch-compatibility-receipt"
    IDENTITY_FIELD: ClassVar[str] = "compatibility_receipt_id"

    def _validate(self) -> None:
        if type(self.policy) is not LaunchCompatibilityPolicy:
            raise LaunchContractError(
                "launch compatibility receipt uses another policy contract"
            )
        require_identifier(self.reason_code, "launch compatibility reason_code")
        for value, name in (
            (self.policy.compatibility_policy_id, "compatibility_policy_id"),
            (self.launch_request_id, "launch_request_id"),
            (self.task_context_binding_id, "task_context_binding_id"),
            (self.scope_contract_id, "scope_contract_id"),
            (self.expert_component_pin_id, "expert_component_pin_id"),
            (self.expert_release_id, "expert_release_id"),
            (self.knowledge_component_pin_id, "knowledge_component_pin_id"),
            (self.knowledge_snapshot_id, "knowledge_snapshot_id"),
            (self.task_adapter_pin_id, "task_adapter_pin_id"),
            (self.task_adapter_manifest_id, "task_adapter_manifest_id"),
            (
                self.task_adapter_verification_receipt_id,
                "task_adapter_verification_receipt_id",
            ),
            (self.task_adapter_activation_id, "task_adapter_activation_id"),
            (
                self.knowledge_embedding_space_id,
                "knowledge_embedding_space_id",
            ),
            (self.release_use_observation_id, "release_use_observation_id"),
            (self.expert_validation_context_id, "expert_validation_context_id"),
            (self.expert_repository_map_id, "expert_repository_map_id"),
            (
                self.expert_release_matrix_stage_result_id,
                "expert_release_matrix_stage_result_id",
            ),
            (
                self.expert_release_matrix_report_id,
                "expert_release_matrix_report_id",
            ),
            (
                self.expert_release_matrix_adapter_authority_id,
                "expert_release_matrix_adapter_authority_id",
            ),
            (
                self.starting_artifact_materialization_receipt_id,
                "starting_artifact_materialization_receipt_id",
            ),
        ):
            require_content_id(value, f"launch compatibility {name}")
        _require_sorted_unique(
            self.expert_module_contract_ids,
            "launch compatibility expert_module_contract_ids",
        )
        _require_sorted_unique(
            self.task_adapter_compatibility_case_ids,
            "launch compatibility task_adapter_compatibility_case_ids",
        )
        _require_sorted_unique(
            self.starting_artifact_content_ids,
            "launch compatibility starting_artifact_content_ids",
        )
        for value in (
            *self.expert_module_contract_ids,
            *self.task_adapter_compatibility_case_ids,
            *self.starting_artifact_content_ids,
        ):
            require_content_id(value, "launch compatibility dependency")
        _require_digest(
            self.runtime_contract_digest,
            "launch compatibility runtime_contract_digest",
        )
        _require_digest(
            self.source_composition_hash,
            "launch compatibility source_composition_hash",
        )
        normalize_utc_timestamp(self.resolved_at, "launch compatibility resolved_at")
        if type(self.compatible) is not bool or not self.compatible:
            raise LaunchContractError(
                "only an admitted compatible tuple can produce a launch receipt"
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "launch compatibility exact_dependency_ids",
        )
        required_dependencies = {
            self.policy.compatibility_policy_id,
            self.launch_request_id,
            self.task_context_binding_id,
            self.scope_contract_id,
            self.expert_component_pin_id,
            self.expert_release_id,
            self.knowledge_component_pin_id,
            self.knowledge_snapshot_id,
            self.task_adapter_pin_id,
            self.task_adapter_manifest_id,
            self.task_adapter_verification_receipt_id,
            self.task_adapter_activation_id,
            self.knowledge_embedding_space_id,
            self.release_use_observation_id,
            self.expert_validation_context_id,
            self.expert_repository_map_id,
            *self.expert_module_contract_ids,
            self.expert_release_matrix_stage_result_id,
            self.expert_release_matrix_report_id,
            self.expert_release_matrix_adapter_authority_id,
            *self.task_adapter_compatibility_case_ids,
            self.starting_artifact_materialization_receipt_id,
            *self.starting_artifact_content_ids,
        }
        if set(self.exact_dependency_ids) != required_dependencies:
            raise LaunchContractError(
                "launch compatibility dependency closure is not exact"
            )


def launch_security_subject_ids(
    *,
    launch_request: LaunchRequest,
    scope_contract: ExpertScopeContract,
    task_context_binding: TaskContextBinding,
    expert_component: LaunchGitHubArtifactPin,
    expert_manifest: ExpertBaseReleaseManifest,
    expert_source: LaunchExpertSourcePin,
    expert_repository_map: ExpertRepositoryMap,
    expert_module_contracts: tuple[ExpertModuleContract, ...],
    knowledge_component: LaunchGitHubArtifactPin,
    knowledge_manifest: KnowledgeSnapshotManifest,
    task_adapter: LaunchTaskAdapterPin,
    starting_artifacts: LaunchStartingArtifactMaterializationReceipt,
    knowledge_embedding_space: EmbeddingSpace,
    experiment_embedding_space: EmbeddingSpace,
    release_use_observation: ExpertReleaseUsePolicyObservation,
    compatibility_receipt: LaunchCompatibilityReceipt,
) -> tuple[str, ...]:
    """Return the exact content-addressed closure checked before execution."""

    return tuple(
        sorted(
            {
                launch_request.launch_request_id,
                scope_contract.scope_contract_id,
                task_context_binding.task_context_binding_id,
                expert_component.component_pin_id,
                expert_component.publication.publication_id,
                expert_component.activation_witness.witness_id,
                expert_manifest.release_id,
                expert_source.source_pin_id,
                expert_source.extraction_receipt.extraction_receipt_id,
                expert_repository_map.repository_map_id,
                *(
                    module_contract.module_contract_id
                    for module_contract in expert_module_contracts
                ),
                *expert_manifest.consumed_dependency_ids,
                knowledge_component.component_pin_id,
                knowledge_component.publication.publication_id,
                knowledge_component.activation_witness.witness_id,
                knowledge_manifest.snapshot_id,
                *knowledge_manifest.proof_dependency_closure_ids,
                task_adapter.adapter_pin_id,
                task_adapter.activation.activation_id,
                task_adapter.manifest.task_adapter_manifest_id,
                task_adapter.verification_receipt.verification_receipt_id,
                task_adapter.source_extraction_receipt.extraction_receipt_id,
                *task_adapter.dependency_ids,
                starting_artifacts.materialization_receipt_id,
                *starting_artifacts.exact_dependency_ids,
                knowledge_embedding_space.embedding_space_id,
                experiment_embedding_space.embedding_space_id,
                release_use_observation.observation_id,
                compatibility_receipt.compatibility_receipt_id,
                *compatibility_receipt.exact_dependency_ids,
            }
        )
    )


@dataclass(frozen=True)
class LaunchManifest(StrictContract):
    """Self-contained immutable authority for one scientific run."""

    launch_manifest_id: str
    launch_request: LaunchRequest
    launch_request_hash: str
    scope_contract: ExpertScopeContract
    task_context_binding: TaskContextBinding
    scope_repositories: ScopeRepositorySettings
    scope_repository_binding_hash: str
    configuration_fingerprint: str
    expert_component: LaunchGitHubArtifactPin
    expert_manifest: ExpertBaseReleaseManifest
    expert_source: LaunchExpertSourcePin
    expert_repository_map: ExpertRepositoryMap
    expert_module_contracts: tuple[ExpertModuleContract, ...]
    knowledge_component: LaunchGitHubArtifactPin
    knowledge_manifest: KnowledgeSnapshotManifest
    task_adapter: LaunchTaskAdapterPin
    starting_artifacts: LaunchStartingArtifactMaterializationReceipt
    knowledge_embedding_space: EmbeddingSpace
    experiment_embedding_space: EmbeddingSpace
    dependency_runtime_contract: Mapping[str, Any]
    sanitation_policy_version: str
    security_observation: SecurityDenylistObservation
    release_use_observation: ExpertReleaseUsePolicyObservation
    compatibility_receipt: LaunchCompatibilityReceipt
    expected_source_composition_hash: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "launch-manifest"
    IDENTITY_FIELD: ClassVar[str] = "launch_manifest_id"

    def _validate(self) -> None:
        if (
            type(self.launch_request) is not LaunchRequest
            or type(self.scope_contract) is not ExpertScopeContract
            or type(self.task_context_binding) is not TaskContextBinding
            or type(self.scope_repositories) is not ScopeRepositorySettings
            or type(self.expert_component) is not LaunchGitHubArtifactPin
            or type(self.expert_manifest) is not ExpertBaseReleaseManifest
            or type(self.expert_source) is not LaunchExpertSourcePin
            or type(self.expert_repository_map) is not ExpertRepositoryMap
            or any(
                type(module_contract) is not ExpertModuleContract
                for module_contract in self.expert_module_contracts
            )
            or type(self.knowledge_component) is not LaunchGitHubArtifactPin
            or type(self.knowledge_manifest) is not KnowledgeSnapshotManifest
            or type(self.task_adapter) is not LaunchTaskAdapterPin
            or type(self.starting_artifacts)
            is not LaunchStartingArtifactMaterializationReceipt
            or type(self.knowledge_embedding_space) is not EmbeddingSpace
            or type(self.experiment_embedding_space) is not EmbeddingSpace
            or type(self.security_observation) is not SecurityDenylistObservation
            or type(self.release_use_observation)
            is not ExpertReleaseUsePolicyObservation
            or type(self.compatibility_receipt) is not LaunchCompatibilityReceipt
        ):
            raise LaunchContractError("launch manifest uses an unrecognized authority")
        module_contract_ids = tuple(
            module_contract.module_contract_id
            for module_contract in self.expert_module_contracts
        )
        if module_contract_ids != tuple(sorted(set(module_contract_ids))):
            raise LaunchContractError(
                "launch expert module contracts must be sorted and unique"
            )
        for value, name in (
            (self.launch_request_hash, "launch_request_hash"),
            (
                self.scope_repository_binding_hash,
                "scope_repository_binding_hash",
            ),
            (self.configuration_fingerprint, "configuration_fingerprint"),
            (
                self.expected_source_composition_hash,
                "expected_source_composition_hash",
            ),
        ):
            _require_digest(value, f"launch manifest {name}")
        require_identifier(
            self.sanitation_policy_version,
            "launch sanitation policy version",
        )
        if (
            not self.dependency_runtime_contract
            or self.launch_request_hash != self.launch_request.request_hash
            or self.configuration_fingerprint
            != self.launch_request.configuration_fingerprint
            or self.dependency_runtime_contract
            != self.launch_request.dependency_runtime_contract
            or self.scope_repositories.scope_id != self.launch_request.binding.scope_id
            or self.scope_repositories.binding_fingerprint
            != self.scope_repository_binding_hash
        ):
            raise LaunchContractError(
                "launch request, runtime, or configuration is inconsistent"
            )
        binding = self.launch_request.binding
        self.scope_contract.validate_binding(binding)
        self.task_context_binding.validate_against(self.scope_contract)
        if self.task_context_binding != self.launch_request.task_context_request.bind(
            binding=binding,
            scope_contract=self.scope_contract,
        ):
            raise LaunchContractError(
                "launch task context differs from the bound request"
            )
        adapter_manifest = self.task_adapter.manifest
        materialized_artifact_ids = {
            artifact.starting_artifact_ref: (artifact.starting_artifact_content_id)
            for artifact in self.starting_artifacts.starting_artifacts
        }
        sidecar_spaces = {
            sidecar.embedding_space_id
            for sidecar in self.knowledge_manifest.embedding_sidecars
        }
        if (
            self.expert_component.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or self.expert_component.artifact_id != self.expert_manifest.release_id
            or self.expert_component.manifest_digest
            != tree_or_blob_digest(self.expert_manifest.to_json_bytes())
            or self.expert_source.expert_release_id != self.expert_manifest.release_id
            or self.expert_source.extraction_receipt.source_archive_ref
            != self.expert_manifest.source_archive_ref
            or self.expert_source.extraction_receipt.source_archive_digest
            != self.expert_manifest.checksums[self.expert_manifest.source_archive_ref]
            or self.expert_source.extraction_receipt.source_tree_hash
            != self.expert_manifest.candidate_tree_hash
            or self.expert_repository_map.repository_map_id
            != self.expert_manifest.repository_map_ref
            or self.expert_repository_map.scope_contract_id
            != self.scope_contract.scope_contract_id
            or module_contract_ids != self.expert_manifest.module_contract_refs
            or dict(self.expert_manifest.module_versions)
            != {
                module_contract.module_id: module_contract.version
                for module_contract in self.expert_module_contracts
            }
            or self.knowledge_component.artifact_kind
            is not PublicationArtifactKind.KNOWLEDGE_SNAPSHOT
            or self.knowledge_component.artifact_id
            != self.knowledge_manifest.snapshot_id
            or self.knowledge_component.manifest_digest
            != tree_or_blob_digest(self.knowledge_manifest.to_json_bytes())
            or self.expert_manifest.scope_id != binding.scope_id
            or self.expert_manifest.scope_contract_id
            != self.scope_contract.scope_contract_id
            or self.knowledge_manifest.scope_id != binding.scope_id
            or self.knowledge_manifest.scope_contract_id
            != self.scope_contract.scope_contract_id
            or adapter_manifest.scope_contract_id
            != self.scope_contract.scope_contract_id
            or adapter_manifest.task_family_id != binding.task_family_id
            or adapter_manifest.task_adapter_id != binding.task_adapter_id
            or self.starting_artifacts.task_context_binding_id
            != self.task_context_binding.task_context_binding_id
            or materialized_artifact_ids
            != dict(self.launch_request.starting_artifact_content_ids)
            or (
                sidecar_spaces
                and self.knowledge_embedding_space.embedding_space_id
                not in sidecar_spaces
            )
            or (not sidecar_spaces and self.knowledge_manifest.entry_state_refs)
            or self.sanitation_policy_version
            != self.knowledge_manifest.sanitation_policy_version
            or self.expert_component.scope_id != binding.scope_id
            or self.knowledge_component.scope_id != binding.scope_id
            or self.expert_component.scope_repository_binding_hash
            != self.scope_repository_binding_hash
            or self.knowledge_component.scope_repository_binding_hash
            != self.scope_repository_binding_hash
            or self.expert_component.publication.repository_full_name
            != self.scope_repositories.expert_repository
            or self.knowledge_component.publication.repository_full_name
            != self.scope_repositories.knowledge_repository
        ):
            raise LaunchContractError(
                "launch scientific components do not share one compatible scope"
            )
        release_use = self.release_use_observation
        security = self.security_observation
        if (
            release_use.scope_id != binding.scope_id
            or release_use.scope_contract_id != self.scope_contract.scope_contract_id
            or release_use.scope_repository_binding_hash
            != self.scope_repository_binding_hash
            or release_use.repository_full_name
            != self.scope_repositories.knowledge_repository
            or release_use.knowledge_snapshot_id != self.knowledge_manifest.snapshot_id
            or release_use.knowledge_publication_id
            != self.knowledge_component.publication.publication_id
            or release_use.checked_release_ids != (self.expert_manifest.release_id,)
            or release_use.matched_revocations
            or security.scope_id != binding.scope_id
            or security.scope_contract_id != self.scope_contract.scope_contract_id
            or security.scope_repository_binding_hash
            != self.scope_repository_binding_hash
            or security.repository_full_name
            != self.scope_repositories.security_repository
            or security.matched_revocations
        ):
            raise LaunchContractError(
                "launch security or release-use policy does not clear the tuple"
            )
        compatibility = self.compatibility_receipt
        if (
            compatibility.launch_request_id != self.launch_request.launch_request_id
            or compatibility.task_context_binding_id
            != self.task_context_binding.task_context_binding_id
            or compatibility.scope_contract_id != self.scope_contract.scope_contract_id
            or compatibility.expert_component_pin_id
            != self.expert_component.component_pin_id
            or compatibility.expert_release_id != self.expert_manifest.release_id
            or compatibility.knowledge_component_pin_id
            != self.knowledge_component.component_pin_id
            or compatibility.knowledge_snapshot_id
            != self.knowledge_manifest.snapshot_id
            or compatibility.task_adapter_pin_id != self.task_adapter.adapter_pin_id
            or compatibility.task_adapter_manifest_id
            != adapter_manifest.task_adapter_manifest_id
            or compatibility.task_adapter_verification_receipt_id
            != self.task_adapter.verification_receipt.verification_receipt_id
            or compatibility.task_adapter_activation_id
            != self.task_adapter.activation.activation_id
            or compatibility.knowledge_embedding_space_id
            != self.knowledge_embedding_space.embedding_space_id
            or compatibility.release_use_observation_id != release_use.observation_id
            or compatibility.expert_validation_context_id
            != self.expert_manifest.candidate_validation_context_ref
            or compatibility.expert_repository_map_id
            != self.expert_repository_map.repository_map_id
            or compatibility.expert_module_contract_ids != module_contract_ids
            or compatibility.expert_release_matrix_stage_result_id
            != self.expert_manifest.release_matrix_stage_result_id
            or compatibility.expert_release_matrix_report_id
            != self.expert_manifest.release_matrix_report_id
            or compatibility.starting_artifact_materialization_receipt_id
            != self.starting_artifacts.materialization_receipt_id
            or compatibility.starting_artifact_content_ids
            != tuple(sorted(self.launch_request.starting_artifact_content_ids.values()))
            or compatibility.runtime_contract_digest
            != tree_or_blob_digest(
                canonical_json_bytes(self.dependency_runtime_contract)
            )
            or compatibility.source_composition_hash
            != self.expected_source_composition_hash
            or self.expected_source_composition_hash
            != expected_launch_source_composition_hash(
                expert_source_tree_hash=self.expert_manifest.candidate_tree_hash,
                expert_repository_map=self.expert_repository_map,
                task_adapter=self.task_adapter,
                starting_artifacts=self.starting_artifacts,
            )
        ):
            raise LaunchContractError(
                "launch compatibility receipt names another tuple"
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "launch manifest exact_dependency_ids",
        )
        security_subjects = launch_security_subject_ids(
            launch_request=self.launch_request,
            scope_contract=self.scope_contract,
            task_context_binding=self.task_context_binding,
            expert_component=self.expert_component,
            expert_manifest=self.expert_manifest,
            expert_source=self.expert_source,
            expert_repository_map=self.expert_repository_map,
            expert_module_contracts=self.expert_module_contracts,
            knowledge_component=self.knowledge_component,
            knowledge_manifest=self.knowledge_manifest,
            task_adapter=self.task_adapter,
            starting_artifacts=self.starting_artifacts,
            knowledge_embedding_space=self.knowledge_embedding_space,
            experiment_embedding_space=self.experiment_embedding_space,
            release_use_observation=release_use,
            compatibility_receipt=compatibility,
        )
        required_dependencies = {*security_subjects, security.observation_id}
        if set(self.exact_dependency_ids) != required_dependencies:
            raise LaunchContractError("launch manifest dependency closure is not exact")
        if security.checked_subject_ids != security_subjects:
            raise LaunchContractError(
                "launch denylist observation differs from the exact dependency closure"
            )
        _reject_repository_routing(
            self.dependency_runtime_contract,
            "launch_manifest.dependency_runtime_contract",
        )

    @property
    def full_digest(self) -> str:
        return tree_or_blob_digest(self.to_json_bytes())


@dataclass(frozen=True)
class LaunchWorkspaceLayout(StrictContract):
    """Machine-independent paths inside one atomically published run root."""

    workspace_relative_path: str
    immutable_root_relative_path: str
    knowledge_snapshot_relative_path: str
    task_adapter_relative_path: str
    starting_artifacts_relative_path: str
    starting_artifact_roots: Mapping[str, str]
    launch_manifest_relative_path: str
    bootstrap_pin_relative_path: str
    run_checkpoint_relative_path: str
    run_checkpoint_journal_relative_path: str
    run_checkpoint_lock_relative_path: str
    run_checkpoint_staging_relative_path: str
    run_idea_archive_relative_path: str
    run_experiment_history_relative_path: str
    run_execution_journal_relative_path: str
    run_derived_state_store_relative_path: str
    run_derived_state_staging_relative_path: str
    run_action_store_relative_path: str
    run_action_ledger_relative_path: str
    run_runtime_lock_relative_path: str

    def _validate(self) -> None:
        workspace = _require_relative_path(
            self.workspace_relative_path,
            "launch workspace_relative_path",
        )
        immutable_root = _require_relative_path(
            self.immutable_root_relative_path,
            "launch immutable_root_relative_path",
        )
        immutable_children = (
            _require_relative_path(
                self.knowledge_snapshot_relative_path,
                "launch knowledge_snapshot_relative_path",
            ),
            _require_relative_path(
                self.task_adapter_relative_path,
                "launch task_adapter_relative_path",
            ),
            _require_relative_path(
                self.starting_artifacts_relative_path,
                "launch starting_artifacts_relative_path",
            ),
        )
        if any(immutable_root not in child.parents for child in immutable_children):
            raise LaunchContractError(
                "launch immutable components must be strict descendants of their root"
            )
        if any(
            left == right or left in right.parents or right in left.parents
            for position, left in enumerate(immutable_children)
            for right in immutable_children[position + 1 :]
        ):
            raise LaunchContractError(
                "launch immutable component roots must be prefix-disjoint"
            )
        if (
            workspace == immutable_root
            or workspace in immutable_root.parents
            or immutable_root in workspace.parents
        ):
            raise LaunchContractError(
                "launch workspace and immutable root must be prefix-disjoint"
            )
        control_paths = (
            _require_relative_path(
                self.launch_manifest_relative_path,
                "launch launch_manifest_relative_path",
            ),
            _require_relative_path(
                self.bootstrap_pin_relative_path,
                "launch bootstrap_pin_relative_path",
            ),
            _require_relative_path(
                self.run_checkpoint_relative_path,
                "launch run_checkpoint_relative_path",
            ),
            _require_relative_path(
                self.run_checkpoint_journal_relative_path,
                "launch run_checkpoint_journal_relative_path",
            ),
            _require_relative_path(
                self.run_checkpoint_lock_relative_path,
                "launch run_checkpoint_lock_relative_path",
            ),
            _require_relative_path(
                self.run_checkpoint_staging_relative_path,
                "launch run_checkpoint_staging_relative_path",
            ),
            _require_relative_path(
                self.run_idea_archive_relative_path,
                "launch run_idea_archive_relative_path",
            ),
            _require_relative_path(
                self.run_experiment_history_relative_path,
                "launch run_experiment_history_relative_path",
            ),
            _require_relative_path(
                self.run_execution_journal_relative_path,
                "launch run_execution_journal_relative_path",
            ),
            _require_relative_path(
                self.run_derived_state_store_relative_path,
                "launch run_derived_state_store_relative_path",
            ),
            _require_relative_path(
                self.run_derived_state_staging_relative_path,
                "launch run_derived_state_staging_relative_path",
            ),
            _require_relative_path(
                self.run_action_store_relative_path,
                "launch run_action_store_relative_path",
            ),
            _require_relative_path(
                self.run_action_ledger_relative_path,
                "launch run_action_ledger_relative_path",
            ),
            _require_relative_path(
                self.run_runtime_lock_relative_path,
                "launch run_runtime_lock_relative_path",
            ),
        )
        if any(path.parent != control_paths[2].parent for path in control_paths[3:]):
            raise LaunchContractError(
                "launch mutable run controls must share one parent"
            )
        materialized_roots = (workspace, immutable_root)
        if any(
            left == right or left in right.parents or right in left.parents
            for position, left in enumerate(control_paths)
            for right in control_paths[position + 1 :]
        ) or any(
            control == root or root in control.parents or control in root.parents
            for control in control_paths
            for root in materialized_roots
        ):
            raise LaunchContractError(
                "launch control files must be prefix-disjoint and outside "
                "materialized roots"
            )
        starting_root = immutable_children[2]
        materialized_roots: list[PurePosixPath] = []
        for artifact_id in sorted(self.starting_artifact_roots):
            require_content_id(artifact_id, "launch starting artifact ID")
            artifact_root = _require_relative_path(
                self.starting_artifact_roots[artifact_id],
                "launch starting artifact root",
            )
            if starting_root not in artifact_root.parents:
                raise LaunchContractError(
                    "launch starting artifact lies outside its read-only root"
                )
            materialized_roots.append(artifact_root)
        if len(materialized_roots) != len(set(materialized_roots)) or any(
            left in right.parents or right in left.parents
            for position, left in enumerate(materialized_roots)
            for right in materialized_roots[position + 1 :]
        ):
            raise LaunchContractError("launch starting-artifact roots overlap")


@dataclass(frozen=True)
class WorkspaceInstallationReceipt(StrictContract):
    """Exact durable installation derived from one complete launch manifest."""

    workspace_installation_receipt_id: str
    launch_manifest_id: str
    launch_manifest_full_digest: str
    run_id: str
    campaign_id: str
    layout: LaunchWorkspaceLayout
    expert_source_tree_hash: str
    knowledge_package_tree_hash: str
    task_adapter_runtime_tree_hash: str
    starting_artifact_materialization_receipt_id: str
    starting_artifact_tree_hashes: Mapping[str, str]
    expected_source_composition_hash: str
    workspace_git_branch: str
    workspace_baseline_commit_sha: str
    workspace_baseline_tree_sha: str
    workspace_git_index_digest: str
    workspace_git_object_ids: tuple[str, ...]
    launch_settings_id: str
    run_checkpoint_journal_device: int
    run_checkpoint_journal_inode: int
    run_checkpoint_lock_device: int
    run_checkpoint_lock_inode: int
    run_action_store_device: int
    run_action_store_inode: int
    run_action_registry_lock_device: int
    run_action_registry_lock_inode: int
    run_action_workspace_lock_device: int
    run_action_workspace_lock_inode: int
    run_runtime_lock_device: int
    run_runtime_lock_inode: int
    installer_id: str
    installer_version: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "workspace-installation-receipt"
    IDENTITY_FIELD: ClassVar[str] = "workspace_installation_receipt_id"

    def _validate(self) -> None:
        for value, name, namespace in (
            (self.launch_manifest_id, "launch_manifest_id", "launch-manifest"),
            (
                self.starting_artifact_materialization_receipt_id,
                "starting_artifact_materialization_receipt_id",
                "launch-starting-artifact-materialization",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise LaunchContractError(
                    f"workspace installation {name} uses the wrong namespace"
                )
        require_content_id(
            self.launch_settings_id,
            "workspace installation launch settings ID",
        )
        if self.launch_settings_id.split(":sha256:", 1)[0] != "launch-settings":
            raise LaunchContractError(
                "workspace installation launch settings uses the wrong namespace"
            )
        for value, name in (
            (
                self.run_checkpoint_journal_device,
                "run checkpoint journal device",
            ),
            (
                self.run_checkpoint_journal_inode,
                "run checkpoint journal inode",
            ),
            (self.run_checkpoint_lock_device, "run checkpoint lock device"),
            (self.run_checkpoint_lock_inode, "run checkpoint lock inode"),
            (self.run_action_store_device, "run action store device"),
            (self.run_action_store_inode, "run action store inode"),
            (
                self.run_action_registry_lock_device,
                "run action registry lock device",
            ),
            (
                self.run_action_registry_lock_inode,
                "run action registry lock inode",
            ),
            (
                self.run_action_workspace_lock_device,
                "run action workspace lock device",
            ),
            (
                self.run_action_workspace_lock_inode,
                "run action workspace lock inode",
            ),
            (self.run_runtime_lock_device, "run runtime lock device"),
            (self.run_runtime_lock_inode, "run runtime lock inode"),
        ):
            if type(value) is not int or value < 0:
                raise LaunchContractError(f"workspace installation {name} is invalid")
        if type(self.layout) is not LaunchWorkspaceLayout:
            raise LaunchContractError(
                "workspace installation requires one typed layout"
            )
        for value, name in (
            (self.launch_manifest_full_digest, "launch_manifest_full_digest"),
            (self.expert_source_tree_hash, "expert_source_tree_hash"),
            (self.knowledge_package_tree_hash, "knowledge_package_tree_hash"),
            (
                self.task_adapter_runtime_tree_hash,
                "task_adapter_runtime_tree_hash",
            ),
            (
                self.expected_source_composition_hash,
                "expected_source_composition_hash",
            ),
            (self.workspace_git_index_digest, "workspace_git_index_digest"),
        ):
            _require_digest(value, f"workspace installation {name}")
        for value, name in (
            (self.run_id, "run_id"),
            (self.campaign_id, "campaign_id"),
            (self.installer_id, "installer_id"),
            (self.installer_version, "installer_version"),
        ):
            require_identifier(value, f"workspace installation {name}")
        require_git_ref_name(
            f"refs/heads/{self.workspace_git_branch}",
            "workspace installation workspace_git_branch",
            qualified=True,
            error_type=LaunchContractError,
        )
        for value, name in (
            (self.workspace_baseline_commit_sha, "workspace_baseline_commit_sha"),
            (self.workspace_baseline_tree_sha, "workspace_baseline_tree_sha"),
        ):
            if _COMMIT_PATTERN.fullmatch(value) is None:
                raise LaunchContractError(
                    f"workspace installation {name} must be a Git object ID"
                )
        if (
            self.workspace_git_object_ids
            != tuple(sorted(set(self.workspace_git_object_ids)))
            or self.workspace_baseline_commit_sha not in self.workspace_git_object_ids
            or self.workspace_baseline_tree_sha not in self.workspace_git_object_ids
            or any(
                _COMMIT_PATTERN.fullmatch(object_id) is None
                for object_id in self.workspace_git_object_ids
            )
        ):
            raise LaunchContractError(
                "workspace installation Git object closure is not exact"
            )
        if set(self.starting_artifact_tree_hashes) != set(
            self.layout.starting_artifact_roots
        ):
            raise LaunchContractError(
                "workspace installation starting-artifact roots and hashes differ"
            )
        for artifact_id, tree_hash in self.starting_artifact_tree_hashes.items():
            require_content_id(
                artifact_id,
                "workspace installation starting artifact ID",
            )
            _require_digest(
                tree_hash,
                "workspace installation starting artifact tree hash",
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "workspace installation exact_dependency_ids",
        )
        if set(self.exact_dependency_ids) != {
            self.launch_manifest_id,
            self.launch_settings_id,
            self.starting_artifact_materialization_receipt_id,
        }:
            raise LaunchContractError(
                "workspace installation dependency closure is not exact"
            )


@dataclass(frozen=True)
class BootstrapPin(StrictContract):
    """Self-contained launch and installation authority published before spend."""

    bootstrap_pin_id: str
    launch_manifest: LaunchManifest
    launch_manifest_full_digest: str
    installation_receipt: WorkspaceInstallationReceipt
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "bootstrap-pin"
    IDENTITY_FIELD: ClassVar[str] = "bootstrap_pin_id"

    def _validate(self) -> None:
        if (
            type(self.launch_manifest) is not LaunchManifest
            or type(self.installation_receipt) is not WorkspaceInstallationReceipt
        ):
            raise LaunchContractError(
                "bootstrap pin requires typed launch and installation authorities"
            )
        expected_runtime_files = tuple(
            descriptor
            for descriptor in self.launch_manifest.task_adapter.source_extraction_receipt.source_tree_files
            if PurePosixPath(descriptor.relative_path).parts[0]
            != "release_matrix_assets"
        )
        expected_runtime_tree_hash = source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
                for descriptor in expected_runtime_files
            }
        )
        expected_starting_artifact_hashes = {
            artifact.starting_artifact_content_id: artifact.materialized_tree_hash
            for artifact in self.launch_manifest.starting_artifacts.starting_artifacts
        }
        if (
            self.launch_manifest_full_digest != self.launch_manifest.full_digest
            or self.installation_receipt.launch_manifest_id
            != self.launch_manifest.launch_manifest_id
            or self.installation_receipt.launch_manifest_full_digest
            != self.launch_manifest_full_digest
            or self.installation_receipt.expert_source_tree_hash
            != self.launch_manifest.expert_manifest.candidate_tree_hash
            or self.installation_receipt.starting_artifact_materialization_receipt_id
            != self.launch_manifest.starting_artifacts.materialization_receipt_id
            or self.installation_receipt.expected_source_composition_hash
            != self.launch_manifest.expected_source_composition_hash
            or self.installation_receipt.task_adapter_runtime_tree_hash
            != expected_runtime_tree_hash
            or self.installation_receipt.knowledge_package_tree_hash
            != self.launch_manifest.knowledge_component.cache_receipt.materialized_tree_digest
            or dict(self.installation_receipt.starting_artifact_tree_hashes)
            != expected_starting_artifact_hashes
        ):
            raise LaunchContractError(
                "bootstrap pin launch and installation authorities do not join"
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "bootstrap pin exact_dependency_ids",
        )
        if set(self.exact_dependency_ids) != {
            self.launch_manifest.launch_manifest_id,
            self.installation_receipt.workspace_installation_receipt_id,
        }:
            raise LaunchContractError("bootstrap pin dependency closure is not exact")


__all__ = [
    "BootstrapPin",
    "expected_launch_source_composition_hash",
    "launch_security_subject_ids",
    "LaunchCompatibilityAdmissionMode",
    "LaunchCompatibilityPolicy",
    "LaunchCompatibilityReceipt",
    "LaunchContractError",
    "LaunchExpertSourcePin",
    "LaunchGitHubArtifactPin",
    "LaunchManifest",
    "LaunchRequest",
    "LaunchStartingArtifact",
    "LaunchStartingArtifactMaterializationReceipt",
    "LaunchTaskAdapterPin",
    "LaunchTaskContextRequest",
    "LaunchWorkspaceLayout",
    "WorkspaceInstallationReceipt",
]
