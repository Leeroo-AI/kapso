from dataclasses import replace

import pytest

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CrossRunTaskBindingSettings,
    ExpertBaseReleaseManifest,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    ScopeRepositorySettings,
    SourceFileDescriptor,
    TaskAdapterManifest,
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
    PublicationAssetIntent,
    PublicationSourceFile,
)
from kapso.cross_run.launch.contracts import (
    LaunchCompatibilityAdmissionMode,
    LaunchCompatibilityPolicy,
    LaunchCompatibilityReceipt,
    LaunchContractError,
    LaunchExpertSourcePin,
    LaunchGitHubArtifactPin,
    LaunchManifest,
    LaunchRequest,
    LaunchStartingArtifact,
    LaunchStartingArtifactMaterializationReceipt,
    LaunchTaskAdapterPin,
    LaunchTaskContextRequest,
    expected_launch_source_composition_hash,
    launch_security_subject_ids,
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
        immutable_release_id="1",
        tag=f"{artifact_kind.value}/fixture",
        assets=(asset,),
        release_attestation_ref=f"attestation/{artifact_kind.value}",
        published_at="2026-07-23T12:00:00Z",
        publisher_identity="leeroo-coder",
    )
    source_file = PublicationSourceFile(
        relative_path="manifest.json",
        mode="100644",
        size=64,
        sha256=manifest_digest,
        git_blob_sha=(
            "4" if artifact_kind is PublicationArtifactKind.EXPERT_BASE_RELEASE else "5"
        )
        * 40,
    )
    asset_intent = PublicationAssetIntent(
        name=asset.name,
        media_type=asset.media_type,
        size=asset.size,
        sha256=asset.sha256,
    )
    materialized_tree_digest = digest(f"{artifact_kind.value}-materialized")
    intent = ArtifactPublicationIntent(
        scope_id="ml_ai",
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        repository_node_id=repository_node_id,
        repository_full_name=repository,
        expected_parent_sha="6" * 40,
        source_commit_sha=publication.commit_sha,
        source_tree_digest=source_tree_digest(
            {
                source_file.relative_path: (
                    source_file.sha256,
                    source_file.mode,
                    source_file.size,
                )
            }
        ),
        source_git_tree_sha="7" * 40,
        source_files=(source_file,),
        preserved_current=None,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path="manifest.json",
        manifest_digest=manifest_digest,
        tag=publication.tag,
        assets=(asset_intent,),
        validation_closure_ids=(artifact_id,),
        publisher_identity=publication.publisher_identity,
        committed_at=publication.published_at,
    )
    pointer = CurrentArtifactPointer(
        scope_id="ml_ai",
        publication_record=publication,
        publication_intent_digest=intent.digest,
        source_tree_digest=intent.source_tree_digest,
        source_git_tree_sha=intent.source_git_tree_sha,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path="manifest.json",
        manifest_digest=manifest_digest,
        validation_closure_ids=intent.validation_closure_ids,
    )
    pointer_digest = tree_or_blob_digest(pointer.to_json_bytes())
    witness = GitHubArtifactActivationWitness.mint(
        scope_id="ml_ai",
        scope_repository_binding_hash=binding_fingerprint,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        repository_full_name=repository,
        activation_commit_sha=publication.commit_sha,
        publication_intent_digest=intent.digest,
        current_pointer_digest=pointer_digest,
    )
    receipt = CacheVerificationReceipt(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path="manifest.json",
        manifest_digest=manifest_digest,
        cache_tree_digest=digest(f"{artifact_kind.value}-cache"),
        asset_digests={asset.name: asset.sha256},
    )
    return LaunchGitHubArtifactPin.mint(
        scope_id="ml_ai",
        scope_repository_binding_hash=binding_fingerprint,
        pointer=pointer,
        publication_intent=intent,
        authority_commit_sha=publication.commit_sha,
        activation_witness=witness,
        cache_receipt=receipt,
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
    expert_repository_map = _record(ExpertRepositoryMap)
    expert_module_contracts = tuple(
        sorted(
            (
                record
                for record in build_records()
                if type(record) is ExpertModuleContract
                and record.module_contract_id in source_expert.module_contract_refs
            ),
            key=lambda record: record.module_contract_id,
        )
    )
    expert_values = source_expert.to_dict()
    expert_values.pop("release_id")
    expert_values["candidate_tree_hash"] = source_tree_hash
    expert_values["checksums"] = {
        **source_expert.checksums,
        source_expert.source_archive_ref: source_archive_digest,
    }
    expert_manifest = ExpertBaseReleaseManifest.mint(**expert_values)
    knowledge_embedding_space = EmbeddingSpace.mint(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        canonicalizer_version="kapso.knowledge_embedding.v1",
    )
    source_knowledge_manifest = _record(KnowledgeSnapshotManifest)
    knowledge_values = source_knowledge_manifest.to_dict()
    knowledge_values.pop("snapshot_id")
    knowledge_values["embedding_sidecars"] = tuple(
        replace(
            sidecar,
            embedding_space_id=knowledge_embedding_space.embedding_space_id,
        )
        for sidecar in source_knowledge_manifest.embedding_sidecars
    )
    knowledge_manifest = KnowledgeSnapshotManifest.mint(**knowledge_values)
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
    adapter_case = adapter_manifest.release_matrix_cases[0]
    bound_context = adapter_case.task_context_binding
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
    starting_artifacts = tuple(
        sorted(
            (
                LaunchStartingArtifact.mint(
                    starting_artifact_ref=artifact.starting_artifact_ref,
                    mount_path=artifact.mount_path,
                    materialized_tree_hash=artifact.materialized_tree_hash,
                    source_files=artifact.source_files,
                )
                for artifact in adapter_case.starting_artifacts
            ),
            key=lambda artifact: artifact.starting_artifact_content_id,
        )
    )
    request = LaunchRequest.mint(
        binding=binding,
        task_context_request=task_context_request,
        goal_digest=digest("goal"),
        starting_artifact_content_ids={
            artifact.starting_artifact_ref: artifact.starting_artifact_content_id
            for artifact in starting_artifacts
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
    starting_artifact_receipt = LaunchStartingArtifactMaterializationReceipt.mint(
        task_context_binding_id=task_context.task_context_binding_id,
        starting_artifacts=starting_artifacts,
        materializer_id="launch_starting_artifact_materializer",
        materializer_version="kapso.launch_starting_artifact_materializer.v1",
        exact_dependency_ids=tuple(
            sorted(
                {
                    task_context.task_context_binding_id,
                    *(
                        artifact.starting_artifact_content_id
                        for artifact in starting_artifacts
                    ),
                }
            )
        ),
    )
    knowledge_embedding_space_id = knowledge_manifest.embedding_sidecars[
        0
    ].embedding_space_id
    experiment_embedding_space = EmbeddingSpace.mint(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        canonicalizer_version="kapso.idea_embedding.v1",
    )
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
    compatibility_policy = LaunchCompatibilityPolicy.mint(
        policy_version="kapso.launch_compatibility.v1",
        admission_mode=(
            LaunchCompatibilityAdmissionMode.VERIFIED_CASE_NEW_ARTIFACT_CONTENT
        ),
        artifact_ttl_seconds=604800,
    )
    matrix_adapter_authority_id = content_id(
        "expert-release-matrix-adapter-authority",
        {"fixture": "launch"},
    )
    compatibility_case_ids = (adapter_case.release_matrix_case_id,)
    starting_artifact_ids = tuple(
        sorted(request.starting_artifact_content_ids.values())
    )
    source_composition_hash = expected_launch_source_composition_hash(
        expert_source_tree_hash=expert_manifest.candidate_tree_hash,
        expert_repository_map=expert_repository_map,
        task_adapter=task_adapter,
        starting_artifacts=starting_artifact_receipt,
    )
    runtime_contract_digest = tree_or_blob_digest(
        canonical_json_bytes(request.dependency_runtime_contract)
    )
    compatibility_dependencies = tuple(
        sorted(
            {
                compatibility_policy.compatibility_policy_id,
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
                knowledge_embedding_space_id,
                release_use.observation_id,
                expert_manifest.candidate_validation_context_ref,
                expert_repository_map.repository_map_id,
                *(module.module_contract_id for module in expert_module_contracts),
                expert_manifest.release_matrix_stage_result_id,
                expert_manifest.release_matrix_report_id,
                matrix_adapter_authority_id,
                *compatibility_case_ids,
                starting_artifact_receipt.materialization_receipt_id,
                *starting_artifact_ids,
            }
        )
    )
    compatibility = LaunchCompatibilityReceipt.mint(
        policy=compatibility_policy,
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
        knowledge_embedding_space_id=knowledge_embedding_space_id,
        release_use_observation_id=release_use.observation_id,
        expert_validation_context_id=(expert_manifest.candidate_validation_context_ref),
        expert_repository_map_id=expert_repository_map.repository_map_id,
        expert_module_contract_ids=tuple(
            module.module_contract_id for module in expert_module_contracts
        ),
        expert_release_matrix_stage_result_id=(
            expert_manifest.release_matrix_stage_result_id
        ),
        expert_release_matrix_report_id=expert_manifest.release_matrix_report_id,
        expert_release_matrix_adapter_authority_id=matrix_adapter_authority_id,
        task_adapter_compatibility_case_ids=compatibility_case_ids,
        starting_artifact_materialization_receipt_id=(
            starting_artifact_receipt.materialization_receipt_id
        ),
        starting_artifact_content_ids=starting_artifact_ids,
        runtime_contract_digest=runtime_contract_digest,
        source_composition_hash=source_composition_hash,
        resolved_at="2026-07-23T12:30:00Z",
        compatible=True,
        reason_code="policy-admitted",
        exact_dependency_ids=compatibility_dependencies,
    )
    required_subjects = set(
        launch_security_subject_ids(
            launch_request=request,
            scope_contract=scope,
            task_context_binding=task_context,
            expert_component=expert_component,
            expert_manifest=expert_manifest,
            expert_source=expert_source,
            expert_repository_map=expert_repository_map,
            expert_module_contracts=expert_module_contracts,
            knowledge_component=knowledge_component,
            knowledge_manifest=knowledge_manifest,
            task_adapter=task_adapter,
            starting_artifacts=starting_artifact_receipt,
            knowledge_embedding_space=knowledge_embedding_space,
            experiment_embedding_space=experiment_embedding_space,
            release_use_observation=release_use,
            compatibility_receipt=compatibility,
        )
    )
    checked_subjects = set(required_subjects)
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
    exact_dependencies = tuple(sorted({*required_subjects, security.observation_id}))
    values = {
        "launch_request": request,
        "launch_request_hash": request.request_hash,
        "scope_contract": scope,
        "task_context_binding": task_context,
        "scope_repositories": repositories,
        "scope_repository_binding_hash": repositories.binding_fingerprint,
        "configuration_fingerprint": request.configuration_fingerprint,
        "expert_component": expert_component,
        "expert_manifest": expert_manifest,
        "expert_source": expert_source,
        "expert_repository_map": expert_repository_map,
        "expert_module_contracts": expert_module_contracts,
        "knowledge_component": knowledge_component,
        "knowledge_manifest": knowledge_manifest,
        "task_adapter": task_adapter,
        "starting_artifacts": starting_artifact_receipt,
        "knowledge_embedding_space": knowledge_embedding_space,
        "experiment_embedding_space": experiment_embedding_space,
        "dependency_runtime_contract": request.dependency_runtime_contract,
        "sanitation_policy_version": knowledge_manifest.sanitation_policy_version,
        "security_observation": security,
        "release_use_observation": release_use,
        "compatibility_receipt": compatibility,
        "expected_source_composition_hash": source_composition_hash,
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
        match="differs from the exact dependency closure",
    ):
        LaunchManifest.mint(**_launch_fixture(omit_compatibility_security_subject=True))


def test_launch_manifest_rejects_embedding_space_splices():
    values = _launch_fixture()
    values["knowledge_embedding_space"] = EmbeddingSpace.mint(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        canonicalizer_version="kapso.other_knowledge_embedding.v1",
    )
    with pytest.raises(LaunchContractError, match="compatible scope"):
        LaunchManifest.mint(**values)

    values = _launch_fixture()
    values["experiment_embedding_space"] = EmbeddingSpace.mint(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        canonicalizer_version="kapso.other_idea_embedding.v1",
    )
    with pytest.raises(LaunchContractError, match="dependency closure is not exact"):
        LaunchManifest.mint(**values)


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


def test_launch_manifest_full_digest_covers_every_serialized_field():
    manifest = LaunchManifest.mint(**_launch_fixture())

    assert manifest.full_digest == tree_or_blob_digest(manifest.to_json_bytes())


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


def test_github_component_pin_rejects_pointer_cache_splice():
    component = _launch_fixture()["expert_component"]
    mismatched_receipt = replace(
        component.cache_receipt,
        materialized_tree_digest=digest("other-materialized-tree"),
    )

    with pytest.raises(LaunchContractError, match="do not join"):
        LaunchGitHubArtifactPin.mint(
            scope_id=component.scope_id,
            scope_repository_binding_hash=(component.scope_repository_binding_hash),
            pointer=component.pointer,
            publication_intent=component.publication_intent,
            authority_commit_sha=component.authority_commit_sha,
            activation_witness=component.activation_witness,
            cache_receipt=mismatched_receipt,
        )
