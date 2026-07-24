import pytest

from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CrossRunTaskBindingSettings,
    ExpertBaseReleaseManifest,
    ExpertScopeContract,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    ScopeRepositorySettings,
    SourceFileDescriptor,
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
from kapso.cross_run.launch.contracts import (
    LaunchCompatibilityReceipt,
    LaunchContractError,
    LaunchExpertSourcePin,
    LaunchGitHubArtifactPin,
    LaunchManifest,
    LaunchRequest,
    LaunchTaskAdapterPin,
    LaunchTaskContextRequest,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.task_adapters import TaskAdapterActivationRecord
from test_cross_run_contracts import build_records, digest, verified_test_task_adapter


def _record(record_type):
    return next(record for record in build_records() if type(record) is record_type)


def _component_pin(
    *,
    artifact_kind,
    artifact_id,
    repository,
    repository_node_id,
    manifest_digest,
    binding_fingerprint,
):
    asset = GitHubReleaseAsset(
        asset_id=f"asset-{artifact_kind.value}",
        name=f"{artifact_kind.value}.tar.zst",
        media_type="application/zstd",
        size=64,
        sha256=digest(f"{artifact_kind.value}-asset"),
    )
    publication = GitHubPublicationRecord.mint(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        repository_node_id=repository_node_id,
        repository_full_name=repository,
        commit_sha=(
            "1" if artifact_kind is PublicationArtifactKind.EXPERT_BASE_RELEASE else "2"
        )
        * 40,
        immutable_release_id=f"immutable-{artifact_kind.value}",
        tag=f"{artifact_kind.value}/fixture",
        assets=(asset,),
        release_attestation_ref=f"attestation/{artifact_kind.value}",
        published_at="2026-07-23T12:00:00Z",
        publisher_identity="leeroo-coder",
    )
    intent_digest = digest(f"{artifact_kind.value}-intent")
    pointer_digest = digest(f"{artifact_kind.value}-pointer")
    witness = GitHubArtifactActivationWitness.mint(
        scope_id="ml_ai",
        scope_repository_binding_hash=binding_fingerprint,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        repository_full_name=repository,
        activation_commit_sha=publication.commit_sha,
        publication_intent_digest=intent_digest,
        current_pointer_digest=pointer_digest,
    )
    receipt = CacheVerificationReceipt(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        materialized_tree_digest=digest(f"{artifact_kind.value}-materialized"),
        manifest_relative_path="manifest.json",
        manifest_digest=manifest_digest,
        cache_tree_digest=digest(f"{artifact_kind.value}-cache"),
        asset_digests={asset.name: asset.sha256},
    )
    return LaunchGitHubArtifactPin.mint(
        scope_id="ml_ai",
        scope_repository_binding_hash=binding_fingerprint,
        publication=publication,
        publication_intent_digest=intent_digest,
        current_pointer_digest=pointer_digest,
        authority_commit_sha=publication.commit_sha,
        activation_witness=witness,
        cache_receipt=receipt,
        manifest_digest=manifest_digest,
    )


def _launch_fixture(*, omit_compatibility_security_subject=False):
    repositories = _record(ScopeRepositorySettings)
    scope = _record(ExpertScopeContract)
    binding = _record(CrossRunTaskBindingSettings)
    adapter_manifest = _record(TaskAdapterManifest)
    verified_adapter = verified_test_task_adapter(adapter_manifest)
    source_payload = b"def launch():\n    return 'ready'\n"
    source_descriptor = SourceFileDescriptor(
        relative_path="src/launch.py",
        digest=tree_or_blob_digest(source_payload),
        mode="100644",
        size=len(source_payload),
    )
    source_tree_hash = source_tree_digest(
        {
            source_descriptor.relative_path: (
                source_descriptor.digest,
                source_descriptor.mode,
                source_descriptor.size,
            )
        }
    )
    source_archive_digest = digest("expert-source-archive")
    source_expert = _record(ExpertBaseReleaseManifest)
    expert_values = source_expert.to_dict()
    expert_values.pop("release_id")
    expert_values["candidate_tree_hash"] = source_tree_hash
    expert_values["checksums"] = {
        **source_expert.checksums,
        source_expert.source_archive_ref: source_archive_digest,
    }
    expert_manifest = ExpertBaseReleaseManifest.mint(**expert_values)
    knowledge_manifest = _record(KnowledgeSnapshotManifest)
    expert_component = _component_pin(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=expert_manifest.release_id,
        repository=repositories.expert_repository,
        repository_node_id="expert-repository-node",
        manifest_digest=tree_or_blob_digest(expert_manifest.to_json_bytes()),
        binding_fingerprint=repositories.binding_fingerprint,
    )
    knowledge_component = _component_pin(
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        artifact_id=knowledge_manifest.snapshot_id,
        repository=repositories.knowledge_repository,
        repository_node_id="knowledge-repository-node",
        manifest_digest=tree_or_blob_digest(knowledge_manifest.to_json_bytes()),
        binding_fingerprint=repositories.binding_fingerprint,
    )
    source_extraction = SourceArchiveExtractionReceipt.mint(
        artifact_id=expert_manifest.release_id,
        source_archive_ref=expert_manifest.source_archive_ref,
        source_archive_digest=source_archive_digest,
        source_tree_hash=source_tree_hash,
        source_tree_files=(source_descriptor,),
        extractor_version="kapso.source_archive_extractor.v1",
    )
    expert_source = LaunchExpertSourcePin.mint(
        expert_release_id=expert_manifest.release_id,
        extraction_receipt=source_extraction,
    )
    activation = TaskAdapterActivationRecord.mint(
        scope_contract_id=scope.scope_contract_id,
        task_family_id=binding.task_family_id,
        task_adapter_id=binding.task_adapter_id,
        task_adapter_manifest_id=adapter_manifest.task_adapter_manifest_id,
        verification_receipt_id=(
            verified_adapter.verification_receipt.verification_receipt_id
        ),
        predecessor_activation_id=None,
        authority_id="test_task_adapter_authority",
        authority_version="test.task_adapter_authority.v1",
        authority_envelope_digest=digest("adapter-activation-envelope"),
    )
    task_adapter = LaunchTaskAdapterPin.mint(
        activation=activation,
        manifest=adapter_manifest,
        verification_receipt=verified_adapter.verification_receipt,
        source_extraction_receipt=verified_adapter.source_extraction_receipt,
    )
    bound_context = _record(TaskContextBinding)
    task_context_request = LaunchTaskContextRequest.mint(
        capability_tags=bound_context.capability_tags,
        input_contract_fingerprint=bound_context.input_contract_fingerprint,
        target_contract_fingerprint=bound_context.target_contract_fingerprint,
        starting_artifact_refs=bound_context.starting_artifact_refs,
        method_fingerprint=bound_context.method_fingerprint,
        toolchain_fingerprint=bound_context.toolchain_fingerprint,
        dependency_runtime_fingerprint=(bound_context.dependency_runtime_fingerprint),
        budget_hardware_envelope=bound_context.budget_hardware_envelope,
        transfer_dimensions=bound_context.transfer_dimensions,
    )
    request = LaunchRequest.mint(
        binding=binding,
        task_context_request=task_context_request,
        goal_digest=digest("goal"),
        starting_artifact_content_ids={
            "artifact/base": content_id(
                "source-replay-starting-artifact",
                {"fixture": "base"},
            )
        },
        requested_coding_agent="codex",
        search_mode="generic",
        dependency_runtime_contract={"python": "3.13", "platform": "linux"},
        budget_fidelity_envelope={"fidelity": "full", "hours": 4},
        configuration_fingerprint=digest("launch-configuration"),
        empty_scope_bootstrap_authorization_id=None,
    )
    task_context = task_context_request.bind(
        binding=binding,
        scope_contract=scope,
    )
    embedding_space_id = knowledge_manifest.embedding_sidecars[0].embedding_space_id
    release_use = ExpertReleaseUsePolicyObservation.mint(
        scope_id=scope.scope_id,
        scope_contract_id=scope.scope_contract_id,
        scope_repository_binding_hash=repositories.binding_fingerprint,
        repository_full_name=repositories.knowledge_repository,
        repository_node_id=knowledge_component.publication.repository_node_id,
        knowledge_snapshot_id=knowledge_manifest.snapshot_id,
        catalog_generation=knowledge_manifest.catalog_generation,
        knowledge_publication_id=knowledge_component.publication.publication_id,
        current_pointer_digest=knowledge_component.current_pointer_digest,
        authority_commit_sha=knowledge_component.authority_commit_sha,
        release_attestation_ref=(
            knowledge_component.publication.release_attestation_ref
        ),
        checked_release_ids=(expert_manifest.release_id,),
        matched_revocations=(),
    )
    compatibility_dependencies = tuple(
        sorted(
            {
                request.launch_request_id,
                task_context.task_context_binding_id,
                scope.scope_contract_id,
                expert_component.component_pin_id,
                expert_manifest.release_id,
                knowledge_component.component_pin_id,
                knowledge_manifest.snapshot_id,
                task_adapter.adapter_pin_id,
                adapter_manifest.task_adapter_manifest_id,
                verified_adapter.verification_receipt.verification_receipt_id,
                activation.activation_id,
                embedding_space_id,
                release_use.observation_id,
            }
        )
    )
    compatibility = LaunchCompatibilityReceipt.mint(
        policy_version="kapso.launch_compatibility.v1",
        launch_request_id=request.launch_request_id,
        task_context_binding_id=task_context.task_context_binding_id,
        scope_contract_id=scope.scope_contract_id,
        expert_component_pin_id=expert_component.component_pin_id,
        expert_release_id=expert_manifest.release_id,
        knowledge_component_pin_id=knowledge_component.component_pin_id,
        knowledge_snapshot_id=knowledge_manifest.snapshot_id,
        task_adapter_pin_id=task_adapter.adapter_pin_id,
        task_adapter_manifest_id=adapter_manifest.task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(
            verified_adapter.verification_receipt.verification_receipt_id
        ),
        task_adapter_activation_id=activation.activation_id,
        embedding_space_id=embedding_space_id,
        release_use_observation_id=release_use.observation_id,
        compatible=True,
        reason_code="policy-admitted",
        exact_dependency_ids=compatibility_dependencies,
    )
    checked_subjects = {
        scope.scope_contract_id,
        expert_manifest.release_id,
        knowledge_manifest.snapshot_id,
        adapter_manifest.task_adapter_manifest_id,
        verified_adapter.verification_receipt.verification_receipt_id,
        activation.activation_id,
        compatibility.compatibility_receipt_id,
    }
    if omit_compatibility_security_subject:
        checked_subjects.remove(compatibility.compatibility_receipt_id)
    security = SecurityDenylistObservation.mint(
        scope_id=scope.scope_id,
        scope_contract_id=scope.scope_contract_id,
        scope_repository_binding_hash=repositories.binding_fingerprint,
        snapshot_id=content_id("security-denylist-snapshot", {"generation": 0}),
        generation=0,
        publication_id=content_id("github-publication", {"security": 0}),
        repository_full_name=repositories.security_repository,
        repository_node_id="security-repository-node",
        pointer_digest=digest("security-pointer"),
        authority_commit_sha="3" * 40,
        release_attestation_ref="attestation/security",
        checked_subject_ids=tuple(sorted(checked_subjects)),
        matched_revocations=(),
    )
    exact_dependencies = tuple(
        sorted(
            {
                request.launch_request_id,
                scope.scope_contract_id,
                task_context.task_context_binding_id,
                expert_component.component_pin_id,
                expert_component.publication.publication_id,
                expert_component.activation_witness.witness_id,
                expert_manifest.release_id,
                expert_source.source_pin_id,
                source_extraction.extraction_receipt_id,
                knowledge_component.component_pin_id,
                knowledge_component.publication.publication_id,
                knowledge_component.activation_witness.witness_id,
                knowledge_manifest.snapshot_id,
                task_adapter.adapter_pin_id,
                activation.activation_id,
                adapter_manifest.task_adapter_manifest_id,
                verified_adapter.verification_receipt.verification_receipt_id,
                verified_adapter.source_extraction_receipt.extraction_receipt_id,
                embedding_space_id,
                security.observation_id,
                release_use.observation_id,
                compatibility.compatibility_receipt_id,
            }
        )
    )
    values = {
        "launch_request": request,
        "launch_request_hash": request.request_hash,
        "scope_contract": scope,
        "task_context_binding": task_context,
        "scope_repository_binding_hash": repositories.binding_fingerprint,
        "configuration_fingerprint": request.configuration_fingerprint,
        "expert_component": expert_component,
        "expert_manifest": expert_manifest,
        "expert_source": expert_source,
        "knowledge_component": knowledge_component,
        "knowledge_manifest": knowledge_manifest,
        "task_adapter": task_adapter,
        "embedding_space_id": embedding_space_id,
        "dependency_runtime_contract": request.dependency_runtime_contract,
        "sanitation_policy_generation": 1,
        "security_observation": security,
        "release_use_observation": release_use,
        "compatibility_receipt": compatibility,
        "expected_source_composition_hash": digest("source-composition"),
        "publisher_attestation": {
            "issuer": "kapso-launcher",
            "signature": "verified",
        },
        "exact_dependency_ids": exact_dependencies,
    }
    return values


def test_launch_manifest_is_self_contained_and_round_trips():
    manifest = LaunchManifest.mint(**_launch_fixture())

    assert LaunchManifest.from_json_bytes(manifest.to_json_bytes()) == manifest
    assert manifest.task_context_binding.scope_id == "ml_ai"
    assert manifest.release_use_observation.matched_revocations == ()
    assert manifest.security_observation.matched_revocations == ()


def test_launch_request_rejects_repository_routing_injection():
    values = _launch_fixture()
    request = values["launch_request"]
    request_values = request.to_dict()
    request_values.pop("launch_request_id")
    request_values["dependency_runtime_contract"] = {
        "python": "3.13",
        "expert_repository": "attacker/other",
    }

    with pytest.raises(LaunchContractError, match="repository routing"):
        LaunchRequest.mint(**request_values)


def test_launch_manifest_requires_security_check_of_compatibility_authority():
    with pytest.raises(
        LaunchContractError,
        match="omits executable scientific subjects",
    ):
        LaunchManifest.mint(**_launch_fixture(omit_compatibility_security_subject=True))


def test_launch_compatibility_closure_is_exact():
    values = _launch_fixture()
    compatibility = values["compatibility_receipt"]
    compatibility_values = compatibility.to_dict()
    compatibility_values.pop("compatibility_receipt_id")
    compatibility_values["exact_dependency_ids"] = tuple(
        dependency_id
        for dependency_id in compatibility.exact_dependency_ids
        if dependency_id != compatibility.expert_release_id
    )

    with pytest.raises(LaunchContractError, match="closure is not exact"):
        LaunchCompatibilityReceipt.mint(**compatibility_values)


def test_launch_scientific_identity_excludes_attestation_but_pin_digest_does_not():
    manifest = LaunchManifest.mint(**_launch_fixture())
    values = manifest.to_dict()
    values.pop("launch_manifest_id")
    values["publisher_attestation"] = {
        "issuer": "kapso-launcher",
        "signature": "rotated",
    }
    rotated = LaunchManifest.mint(**values)

    assert rotated.launch_manifest_id == manifest.launch_manifest_id
    assert rotated.full_digest != manifest.full_digest


def test_task_adapter_pin_rejects_activation_package_splice():
    values = _launch_fixture()
    task_adapter = values["task_adapter"]
    activation = TaskAdapterActivationRecord.mint(
        scope_contract_id=task_adapter.activation.scope_contract_id,
        task_family_id=task_adapter.activation.task_family_id,
        task_adapter_id="other-adapter",
        task_adapter_manifest_id=task_adapter.activation.task_adapter_manifest_id,
        verification_receipt_id=task_adapter.activation.verification_receipt_id,
        predecessor_activation_id=None,
        authority_id=task_adapter.activation.authority_id,
        authority_version=task_adapter.activation.authority_version,
        authority_envelope_digest=(task_adapter.activation.authority_envelope_digest),
    )

    with pytest.raises(LaunchContractError, match="do not join"):
        LaunchTaskAdapterPin.mint(
            activation=activation,
            manifest=task_adapter.manifest,
            verification_receipt=task_adapter.verification_receipt,
            source_extraction_receipt=task_adapter.source_extraction_receipt,
        )
