"""Strict transaction contracts for cross-run launch resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    CrossRunTaskBindingSettings,
    ExpertBaseReleaseManifest,
    ExpertScopeContract,
    GitHubPublicationRecord,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    StrictContract,
    TaskAdapterManifest,
    TaskContextBinding,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.github.materializer import (
    CacheVerificationReceipt,
    SourceArchiveExtractionReceipt,
)
from kapso.cross_run.github.resolver import GitHubArtifactActivationWitness
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
class LaunchGitHubArtifactPin(StrictContract):
    """Exact immutable publication and verified local cache authority."""

    component_pin_id: str
    scope_id: str
    scope_repository_binding_hash: str
    publication: GitHubPublicationRecord
    publication_intent_digest: str
    current_pointer_digest: str
    authority_commit_sha: str
    activation_witness: GitHubArtifactActivationWitness
    cache_receipt: CacheVerificationReceipt
    manifest_digest: str

    CONTENT_NAMESPACE: ClassVar[str] = "launch-github-artifact-pin"
    IDENTITY_FIELD: ClassVar[str] = "component_pin_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "launch component scope_id")
        for value, name in (
            (
                self.scope_repository_binding_hash,
                "scope_repository_binding_hash",
            ),
            (self.publication_intent_digest, "publication_intent_digest"),
            (self.current_pointer_digest, "current_pointer_digest"),
            (self.manifest_digest, "manifest_digest"),
        ):
            _require_digest(value, f"launch component {name}")
        if _COMMIT_PATTERN.fullmatch(self.authority_commit_sha) is None:
            raise LaunchContractError(
                "launch component authority_commit_sha is invalid"
            )
        publication = self.publication
        witness = self.activation_witness
        receipt = self.cache_receipt
        if (
            type(publication) is not GitHubPublicationRecord
            or type(witness) is not GitHubArtifactActivationWitness
            or type(receipt) is not CacheVerificationReceipt
            or witness.scope_id != self.scope_id
            or witness.scope_repository_binding_hash
            != self.scope_repository_binding_hash
            or witness.artifact_kind is not publication.artifact_kind
            or witness.artifact_id != publication.artifact_id
            or witness.repository_full_name != publication.repository_full_name
            or witness.publication_intent_digest != self.publication_intent_digest
            or witness.current_pointer_digest != self.current_pointer_digest
            or receipt.artifact_kind is not publication.artifact_kind
            or receipt.artifact_id != publication.artifact_id
            or receipt.manifest_digest != self.manifest_digest
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
            or receipt.source_extraction_receipt_id != extraction.extraction_receipt_id
            or extraction.artifact_id != manifest.task_adapter_manifest_id
            or extraction.source_archive_ref != manifest.source_tree_ref
            or extraction.source_tree_hash != manifest.tree_hash
            or receipt.source_tree_hash != manifest.tree_hash
        ):
            raise LaunchContractError(
                "launch task adapter authorities do not join exactly"
            )


@dataclass(frozen=True)
class LaunchCompatibilityReceipt(StrictContract):
    compatibility_receipt_id: str
    policy_version: str
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
    embedding_space_id: str
    release_use_observation_id: str
    compatible: bool
    reason_code: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "launch-compatibility-receipt"
    IDENTITY_FIELD: ClassVar[str] = "compatibility_receipt_id"

    def _validate(self) -> None:
        require_identifier(self.policy_version, "launch compatibility policy")
        require_identifier(self.reason_code, "launch compatibility reason_code")
        for value, name in (
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
            (self.embedding_space_id, "embedding_space_id"),
            (self.release_use_observation_id, "release_use_observation_id"),
        ):
            require_content_id(value, f"launch compatibility {name}")
        if type(self.compatible) is not bool or not self.compatible:
            raise LaunchContractError(
                "only an admitted compatible tuple can produce a launch receipt"
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "launch compatibility exact_dependency_ids",
        )
        required_dependencies = {
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
            self.embedding_space_id,
            self.release_use_observation_id,
        }
        if set(self.exact_dependency_ids) != required_dependencies:
            raise LaunchContractError(
                "launch compatibility dependency closure is not exact"
            )


@dataclass(frozen=True)
class LaunchManifest(StrictContract):
    """Self-contained immutable authority for one scientific run."""

    launch_manifest_id: str
    launch_request: LaunchRequest
    launch_request_hash: str
    scope_contract: ExpertScopeContract
    task_context_binding: TaskContextBinding
    scope_repository_binding_hash: str
    configuration_fingerprint: str
    expert_component: LaunchGitHubArtifactPin
    expert_manifest: ExpertBaseReleaseManifest
    expert_source: LaunchExpertSourcePin
    knowledge_component: LaunchGitHubArtifactPin
    knowledge_manifest: KnowledgeSnapshotManifest
    task_adapter: LaunchTaskAdapterPin
    embedding_space_id: str
    dependency_runtime_contract: Mapping[str, Any]
    sanitation_policy_generation: int
    security_observation: SecurityDenylistObservation
    release_use_observation: ExpertReleaseUsePolicyObservation
    compatibility_receipt: LaunchCompatibilityReceipt
    expected_source_composition_hash: str
    publisher_attestation: Mapping[str, Any]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "launch-manifest"
    IDENTITY_FIELD: ClassVar[str] = "launch_manifest_id"
    CONTENT_EXCLUDED_FIELDS: ClassVar[tuple[str, ...]] = ("publisher_attestation",)

    def _validate(self) -> None:
        if (
            type(self.launch_request) is not LaunchRequest
            or type(self.scope_contract) is not ExpertScopeContract
            or type(self.task_context_binding) is not TaskContextBinding
            or type(self.expert_component) is not LaunchGitHubArtifactPin
            or type(self.expert_manifest) is not ExpertBaseReleaseManifest
            or type(self.expert_source) is not LaunchExpertSourcePin
            or type(self.knowledge_component) is not LaunchGitHubArtifactPin
            or type(self.knowledge_manifest) is not KnowledgeSnapshotManifest
            or type(self.task_adapter) is not LaunchTaskAdapterPin
            or type(self.security_observation) is not SecurityDenylistObservation
            or type(self.release_use_observation)
            is not ExpertReleaseUsePolicyObservation
            or type(self.compatibility_receipt) is not LaunchCompatibilityReceipt
        ):
            raise LaunchContractError("launch manifest uses an unrecognized authority")
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
        require_content_id(self.embedding_space_id, "launch embedding_space_id")
        if (
            type(self.sanitation_policy_generation) is not int
            or self.sanitation_policy_generation < 0
        ):
            raise LaunchContractError(
                "launch sanitation policy generation must be non-negative"
            )
        if (
            not self.dependency_runtime_contract
            or not self.publisher_attestation
            or self.launch_request_hash != self.launch_request.request_hash
            or self.configuration_fingerprint
            != self.launch_request.configuration_fingerprint
            or self.dependency_runtime_contract
            != self.launch_request.dependency_runtime_contract
        ):
            raise LaunchContractError(
                "launch request, runtime, configuration, or attestation is inconsistent"
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
            or self.embedding_space_id not in sidecar_spaces
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
            or release_use.knowledge_snapshot_id != self.knowledge_manifest.snapshot_id
            or release_use.knowledge_publication_id
            != self.knowledge_component.publication.publication_id
            or release_use.checked_release_ids != (self.expert_manifest.release_id,)
            or release_use.matched_revocations
            or security.scope_id != binding.scope_id
            or security.scope_contract_id != self.scope_contract.scope_contract_id
            or security.scope_repository_binding_hash
            != self.scope_repository_binding_hash
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
            or compatibility.embedding_space_id != self.embedding_space_id
            or compatibility.release_use_observation_id != release_use.observation_id
        ):
            raise LaunchContractError(
                "launch compatibility receipt names another tuple"
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "launch manifest exact_dependency_ids",
        )
        required_dependencies = {
            self.launch_request.launch_request_id,
            self.scope_contract.scope_contract_id,
            self.task_context_binding.task_context_binding_id,
            self.expert_component.component_pin_id,
            self.expert_component.publication.publication_id,
            self.expert_component.activation_witness.witness_id,
            self.expert_manifest.release_id,
            self.expert_source.source_pin_id,
            self.expert_source.extraction_receipt.extraction_receipt_id,
            self.knowledge_component.component_pin_id,
            self.knowledge_component.publication.publication_id,
            self.knowledge_component.activation_witness.witness_id,
            self.knowledge_manifest.snapshot_id,
            self.task_adapter.adapter_pin_id,
            self.task_adapter.activation.activation_id,
            adapter_manifest.task_adapter_manifest_id,
            self.task_adapter.verification_receipt.verification_receipt_id,
            self.task_adapter.source_extraction_receipt.extraction_receipt_id,
            self.embedding_space_id,
            security.observation_id,
            release_use.observation_id,
            compatibility.compatibility_receipt_id,
        }
        if set(self.exact_dependency_ids) != required_dependencies:
            raise LaunchContractError("launch manifest dependency closure is not exact")
        security_required_subjects = {
            self.scope_contract.scope_contract_id,
            self.expert_manifest.release_id,
            self.knowledge_manifest.snapshot_id,
            adapter_manifest.task_adapter_manifest_id,
            self.task_adapter.verification_receipt.verification_receipt_id,
            self.task_adapter.activation.activation_id,
            compatibility.compatibility_receipt_id,
        }
        if not security_required_subjects.issubset(security.checked_subject_ids):
            raise LaunchContractError(
                "launch denylist observation omits executable scientific subjects"
            )
        _reject_repository_routing(
            self.dependency_runtime_contract,
            "launch_manifest.dependency_runtime_contract",
        )

    @property
    def full_digest(self) -> str:
        return tree_or_blob_digest(self.to_json_bytes())


__all__ = [
    "LaunchCompatibilityReceipt",
    "LaunchContractError",
    "LaunchExpertSourcePin",
    "LaunchGitHubArtifactPin",
    "LaunchManifest",
    "LaunchRequest",
    "LaunchTaskAdapterPin",
    "LaunchTaskContextRequest",
]
