import gc
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from weakref import ref

import pytest

from kapso.core.config import load_config
from kapso.core.embedding_contracts import EmbeddingSettings
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CrossRunTaskBindingSettings,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    ObjectiveDirection,
    PublicationArtifactKind,
    SourceFileDescriptor,
    TaskAdapterContextBinding,
    TaskAdapterManifest,
    TaskAdapterReleaseMatrixCase,
    TaskAdapterReleaseMatrixStartingArtifact,
    TaskContextBinding,
)
from kapso.cross_run.embedding_space import EmbeddingSpace
from kapso.cross_run.expert.release import (
    EXPERT_RELEASE_MANIFEST_PATH,
    ExpertReleaseAssembler,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.github.materializer import (
    CacheVerificationReceipt,
    ExpertReleaseSourceSnapshot,
    MaterializedArtifact,
    SourceArchiveExtractionReceipt,
)
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    GitHubArtifactActivationWitness,
    PublicationAssetIntent,
    PublicationSourceFile,
    RepositoryPolicyReport,
    ResolvedGitHubArtifact,
)
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackageBuilder
from kapso.cross_run.launch.contracts import (
    LaunchRequest,
    LaunchStartingArtifact,
    LaunchStartingArtifactMaterializationReceipt,
    LaunchTaskContextRequest,
)
from kapso.cross_run.launch.resolver import (
    ExpertLaunchEvidence,
    LaunchResolutionError,
    LaunchResolver,
    VerifiedLaunchStartingArtifact,
    VerifiedLaunchStartingArtifacts,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.task_adapters import (
    ActiveTaskAdapterBinding,
    TaskAdapterActivationRecord,
)
from test_cross_run_contracts import (
    build_records,
    digest,
    task_adapter_source,
    verified_test_task_adapter,
)
from test_expert_candidates import bootstrap_candidate_closure
from test_expert_release_assembly import _approved_bootstrap
from test_knowledge_snapshot_package import empty_generation
from task_adapter_matrix_fixtures import task_adapter_release_matrix_case

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
RESOLVED_AT = "2026-07-23T12:30:00Z"
PUBLISHED_AT = "2026-07-23T12:00:00Z"


class FixedClock:
    def now(self):
        return RESOLVED_AT


class FakeGitHubResolver:
    def __init__(self, currents, intents, witnesses):
        self.currents = currents
        self.intents = intents
        self.witnesses = witnesses
        self.resolve_counts = {
            PublicationArtifactKind.EXPERT_BASE_RELEASE: 0,
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT: 0,
        }
        self.changed_kind = None

    def resolve_current(self, scope_id, artifact_kind):
        assert scope_id == "ml_ai"
        self.resolve_counts[artifact_kind] += 1
        current = self.currents[artifact_kind]
        if (
            self.changed_kind is artifact_kind
            and self.resolve_counts[artifact_kind] > 1
        ):
            return replace(current, pointer_commit_sha="f" * 40)
        return current

    def read_artifact_intent(self, scope_id, artifact_kind, artifact_id):
        assert scope_id == "ml_ai"
        return self.intents[artifact_kind]

    def resolve_artifact_activation_witness(
        self,
        scope_id,
        artifact_kind,
        artifact_id,
        intent,
        pointer,
        *,
        allow_missing=False,
    ):
        assert not allow_missing
        assert scope_id == "ml_ai"
        assert artifact_id == pointer.publication_record.artifact_id
        assert intent == self.intents[artifact_kind]
        return self.witnesses[artifact_kind]


class FakeMaterializer:
    def __init__(
        self,
        artifacts,
        expert_source,
        expert_publication_files,
    ):
        self.artifacts = artifacts
        self.expert_source = expert_source
        self.expert_publication_files = expert_publication_files

    def materialize(self, resolved):
        return self.artifacts[resolved.pointer.publication_record.artifact_kind]

    def inspect_expert_release_source(
        self,
        materialized,
        *,
        maximum_entries,
        maximum_bytes,
    ):
        assert maximum_entries > 0
        assert maximum_bytes > 0
        return self.expert_source

    def read_verified_content_files(
        self,
        materialized,
        relative_paths,
        *,
        maximum_bytes,
    ):
        assert maximum_bytes > 0
        return MappingProxyType(
            {
                relative_path: self.expert_publication_files[relative_path][0]
                for relative_path in relative_paths
            }
        )


class FixedTaskAdapterProvider:
    def __init__(self, binding):
        self.binding = binding
        self.resolve_count = 0

    def resolve_active_binding(
        self,
        *,
        scope_contract_id,
        task_family_id,
        task_adapter_id,
    ):
        self.resolve_count += 1
        assert scope_contract_id == self.binding.activation.scope_contract_id
        assert task_family_id == self.binding.activation.task_family_id
        assert task_adapter_id == self.binding.activation.task_adapter_id
        return self.binding


class FixedStartingArtifactProvider:
    def __init__(self, verified):
        self.verified = verified
        self.resolve_count = 0

    def materialize_exact(
        self,
        *,
        task_context_binding,
        expected_artifact_content_ids,
        maximum_entries,
        maximum_bytes,
    ):
        self.resolve_count += 1
        assert expected_artifact_content_ids == {
            artifact.artifact.starting_artifact_ref: (
                artifact.artifact.starting_artifact_content_id
            )
            for artifact in self.verified.starting_artifacts
        }
        assert self.verified.entry_count <= maximum_entries
        assert self.verified.byte_count <= maximum_bytes
        receipt = LaunchStartingArtifactMaterializationReceipt.mint(
            task_context_binding_id=task_context_binding.task_context_binding_id,
            starting_artifacts=self.verified.receipt.starting_artifacts,
            materializer_id=self.verified.receipt.materializer_id,
            materializer_version=self.verified.receipt.materializer_version,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        task_context_binding.task_context_binding_id,
                        *(
                            item.artifact.starting_artifact_content_id
                            for item in self.verified.starting_artifacts
                        ),
                    }
                )
            ),
        )
        return VerifiedLaunchStartingArtifacts(
            receipt=receipt,
            starting_artifacts=self.verified.starting_artifacts,
        )


class FixedReleaseUseAuthority:
    def __init__(self, knowledge_current, knowledge_package, release_id):
        self.knowledge_current = knowledge_current
        self.knowledge_package = knowledge_package
        self.release_id = release_id
        self.resolve_count = 0

    def observe_exact(self, *, scope_contract, checked_release_ids):
        self.resolve_count += 1
        assert checked_release_ids == (self.release_id,)
        pointer = self.knowledge_current.pointer
        publication = pointer.publication_record
        return ExpertReleaseUsePolicyObservation.mint(
            scope_id=scope_contract.scope_id,
            scope_contract_id=scope_contract.scope_contract_id,
            scope_repository_binding_hash=(
                self.knowledge_current.repositories.binding_fingerprint
            ),
            repository_full_name=publication.repository_full_name,
            repository_node_id=publication.repository_node_id,
            knowledge_snapshot_id=self.knowledge_package.manifest.snapshot_id,
            catalog_generation=self.knowledge_package.manifest.catalog_generation,
            knowledge_publication_id=publication.publication_id,
            current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
            authority_commit_sha=self.knowledge_current.pointer_commit_sha,
            release_attestation_ref=publication.release_attestation_ref,
            checked_release_ids=checked_release_ids,
            matched_revocations=(),
        )


class RecordingSecurityAuthority:
    def __init__(self, settings):
        self.settings = settings
        self.checked_subject_ids = ()
        self.resolve_count = 0

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.resolve_count += 1
        self.checked_subject_ids = checked_subject_ids
        repositories = self.settings.scopes.resolve(scope_id)
        return SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=repositories.binding_fingerprint,
            snapshot_id=content_id(
                "security-denylist-snapshot",
                {"generation": 1},
            ),
            generation=1,
            publication_id=content_id(
                "github-publication",
                {"security-generation": 1},
            ),
            repository_full_name=repositories.security_repository,
            repository_node_id="security-repository-node",
            pointer_digest=digest("security-pointer"),
            authority_commit_sha="9" * 40,
            release_attestation_ref="attestation/security",
            checked_subject_ids=checked_subject_ids,
            matched_revocations=(),
        )


def _settings():
    return CrossRunSettings.from_dict(load_config(CANONICAL_CONFIG_PATH)["cross_run"])


def _experiment_embedding_space():
    config = load_config(CANONICAL_CONFIG_PATH)
    profile_name = config["modes"]["GENERIC"]["ideation_profile"]
    settings = EmbeddingSettings.from_dict(
        config["ideation_profiles"][profile_name]["embeddings"]
    )
    assert settings.enabled
    return EmbeddingSpace.mint(
        provider=settings.provider,
        model=settings.model,
        dimensions=settings.dimensions,
        canonicalizer_version=settings.canonicalizer_version,
    )


def _source_adapter_for_binding(binding, scope_contract_id):
    if binding.task_adapter_id == "posttrain":
        return None
    base_manifest = next(
        record
        for record in build_records()
        if type(record) is TaskAdapterManifest
        and record.scope_contract_id == scope_contract_id
    )
    base_adapter = verified_test_task_adapter(base_manifest)
    manifest_values = base_manifest.to_dict()
    manifest_values.pop("task_adapter_manifest_id")
    manifest_values.update(
        task_adapter_id=binding.task_adapter_id,
        task_family_id=binding.task_family_id,
        context_binding=TaskAdapterContextBinding(
            consumed_dimension_ids=("dataset_family", "runtime_family")
        ),
        release_matrix_cases=(
            task_adapter_release_matrix_case(
                scope_contract_id=scope_contract_id,
                scope_id=binding.scope_id,
                task_family_id=binding.task_family_id,
                task_adapter_id=binding.task_adapter_id,
                evaluator_fingerprint=(
                    base_manifest.task_evaluator.metric_comparison_bindings[
                        0
                    ].evaluator_fingerprint
                ),
                metric_directions=(("quality", ObjectiveDirection.MAXIMIZE),),
                transfer_dimensions={
                    "dataset_family": "relational_tabular",
                    "runtime_family": "pytorch",
                },
                label="relbench-system-scenario",
            ),
        ),
    )
    return verified_test_task_adapter(
        TaskAdapterManifest.mint(**manifest_values),
        source_contents=base_adapter.source_contents,
    )


def _resolved_artifact(
    *,
    settings,
    artifact_kind,
    artifact_id,
    manifest_digest,
    manifest_relative_path,
    commit_character,
    materialized_tree_digest=None,
):
    repositories = settings.scopes.resolve("ml_ai")
    repository = (
        repositories.expert_repository
        if artifact_kind is PublicationArtifactKind.EXPERT_BASE_RELEASE
        else repositories.knowledge_repository
    )
    asset = GitHubReleaseAsset(
        asset_id=f"{artifact_kind.value}-asset",
        name=f"{artifact_kind.value}.tar.zst",
        media_type="application/zstd",
        size=64,
        sha256=digest(f"{artifact_kind.value}-asset"),
    )
    publication = GitHubPublicationRecord.mint(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        repository_node_id=f"{artifact_kind.value}-repository-node",
        repository_full_name=repository,
        commit_sha=commit_character * 40,
        immutable_release_id=("1" if commit_character == "1" else "2"),
        tag=f"{artifact_kind.value}/fixture",
        assets=(asset,),
        release_attestation_ref=f"attestation/{artifact_kind.value}",
        published_at=PUBLISHED_AT,
        publisher_identity=settings.github.publisher_login,
    )
    source_file = PublicationSourceFile(
        relative_path=manifest_relative_path,
        mode="100644",
        size=64,
        sha256=manifest_digest,
        git_blob_sha=("3" if commit_character == "1" else "4") * 40,
    )
    asset_intent = PublicationAssetIntent(
        name=asset.name,
        media_type=asset.media_type,
        size=asset.size,
        sha256=asset.sha256,
    )
    if materialized_tree_digest is None:
        materialized_tree_digest = digest(f"{artifact_kind.value}-materialized")
    intent = ArtifactPublicationIntent(
        scope_id="ml_ai",
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        repository_node_id=publication.repository_node_id,
        repository_full_name=repository,
        expected_parent_sha="5" * 40,
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
        source_git_tree_sha="6" * 40,
        source_files=(source_file,),
        preserved_current=None,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path=manifest_relative_path,
        manifest_digest=manifest_digest,
        tag=publication.tag,
        assets=(asset_intent,),
        validation_closure_ids=(artifact_id,),
        publisher_identity=publication.publisher_identity,
        committed_at=PUBLISHED_AT,
    )
    pointer = CurrentArtifactPointer(
        scope_id="ml_ai",
        publication_record=publication,
        publication_intent_digest=intent.digest,
        source_tree_digest=intent.source_tree_digest,
        source_git_tree_sha=intent.source_git_tree_sha,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path=manifest_relative_path,
        manifest_digest=manifest_digest,
        validation_closure_ids=intent.validation_closure_ids,
    )
    current = ResolvedGitHubArtifact(
        repositories=repositories,
        pointer=pointer,
        policy=RepositoryPolicyReport(
            repository_full_name=repository,
            repository_node_id=publication.repository_node_id,
            private=True,
            default_branch=settings.github.default_branch,
            authenticated_actor=settings.github.publisher_login,
            write_access=True,
            immutable_releases=True,
        ),
        pointer_commit_sha=publication.commit_sha,
    )
    receipt = CacheVerificationReceipt(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        materialized_tree_digest=materialized_tree_digest,
        manifest_relative_path=manifest_relative_path,
        manifest_digest=manifest_digest,
        cache_tree_digest=digest(f"{artifact_kind.value}-cache"),
        asset_digests={asset.name: asset.sha256},
    )
    witness = GitHubArtifactActivationWitness.mint(
        scope_id="ml_ai",
        scope_repository_binding_hash=repositories.binding_fingerprint,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        repository_full_name=repository,
        activation_commit_sha=publication.commit_sha,
        publication_intent_digest=intent.digest,
        current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
    )
    return current, intent, receipt, witness


def build_resolver_case(
    tmp_path,
    monkeypatch,
    binding=None,
    *,
    expert_case=None,
    knowledge_package=None,
):
    settings = _settings()
    if expert_case is None:
        expert_fixture_root = tmp_path / "expert"
        expert_fixture_root.mkdir()
        candidate_closure = (
            None
            if binding is None
            else bootstrap_candidate_closure(active_task_bindings=(binding,))
        )
        source_adapter = (
            None
            if candidate_closure is None
            else _source_adapter_for_binding(
                binding,
                candidate_closure.validation_context.scope_contract.scope_contract_id,
            )
        )
        validation_store, matrix, approval, _ = _approved_bootstrap(
            expert_fixture_root,
            monkeypatch,
            candidate_closure=candidate_closure,
            source_adapter=source_adapter,
        )
        candidate_store = validation_store.reducer.candidate_store
        stored_candidate = candidate_store.read(approval.snapshot.state.candidate_id)
        assembler = ExpertReleaseAssembler(
            candidate_store=candidate_store,
            validation_store=validation_store,
            expert_settings=candidate_store.validator.settings,
            github_settings=settings.github,
        )
        expert_package = assembler.build(
            candidate_id=stored_candidate.closure.manifest.candidate_id
        )
        release_matrix_stage_result = matrix.stage_result
    else:
        stored_candidate = expert_case.stored_candidate
        expert_package = expert_case.expert_package
        release_matrix_stage_result = expert_case.release_matrix_stage_result
        source_adapter = expert_case.verified_adapter
    expert_descriptors = tuple(
        SourceFileDescriptor(
            relative_path=relative_path,
            digest=tree_or_blob_digest(payload),
            mode=mode,
            size=len(payload),
        )
        for relative_path, (payload, mode) in sorted(
            expert_package.source_files.items()
        )
    )
    expert_extraction = SourceArchiveExtractionReceipt.mint(
        artifact_id=expert_package.manifest.release_id,
        source_archive_ref=expert_package.manifest.source_archive_ref,
        source_archive_digest=tree_or_blob_digest(expert_package.source_archive),
        source_tree_hash=expert_package.manifest.candidate_tree_hash,
        source_tree_files=expert_descriptors,
        extractor_version="kapso.source_archive_extractor.v1",
    )
    expert_source = ExpertReleaseSourceSnapshot(
        release_manifest=expert_package.manifest,
        source_extraction_receipt=expert_extraction,
        source_contents={
            relative_path: payload
            for relative_path, (payload, _mode) in expert_package.source_files.items()
        },
    )

    scope_contract = stored_candidate.closure.validation_context.scope_contract
    if knowledge_package is None:
        prepared_knowledge = KnowledgeSnapshotPackageBuilder.prepare_empty(
            scope_contract,
            empty_generation(scope_contract),
        )
        knowledge_package = KnowledgeSnapshotPackageBuilder.finalize(
            prepared_knowledge,
            parent_snapshot_ids=(),
            sanitation_policy_version="kapso.sanitation.v1",
            retrieval_policy_version="kapso.retrieval.v1",
            configuration_fingerprint=digest("knowledge-config"),
            prompt_budget_policy={"maximum_records": 24},
            published_at=PUBLISHED_AT,
            publisher_attestation={"issuer": "test-publisher"},
        )
    knowledge_parent = tmp_path / "knowledge"
    knowledge_parent.mkdir()
    knowledge_content = knowledge_package.materialize(
        (knowledge_parent / "content").absolute()
    )

    expert_current, expert_intent, expert_receipt, expert_witness = _resolved_artifact(
        settings=settings,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=expert_package.manifest.release_id,
        manifest_digest=tree_or_blob_digest(expert_package.manifest.to_json_bytes()),
        manifest_relative_path=EXPERT_RELEASE_MANIFEST_PATH,
        commit_character="1",
    )
    knowledge_current, knowledge_intent, knowledge_receipt, knowledge_witness = (
        _resolved_artifact(
            settings=settings,
            artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            artifact_id=knowledge_package.manifest.snapshot_id,
            manifest_digest=tree_or_blob_digest(
                knowledge_package.manifest.to_json_bytes()
            ),
            manifest_relative_path="snapshot.json",
            commit_character="2",
            materialized_tree_digest=source_tree_digest(
                {
                    relative_path: (
                        tree_or_blob_digest(payload),
                        "100644",
                        len(payload),
                    )
                    for relative_path, payload in knowledge_package.files.items()
                }
            ),
        )
    )
    expert_artifact = MaterializedArtifact(
        root=(tmp_path / "expert-cache").absolute(),
        content=(tmp_path / "expert-cache" / "content").absolute(),
        assets=(tmp_path / "expert-cache" / "assets").absolute(),
        receipt=expert_receipt,
        reused=False,
    )
    knowledge_artifact = MaterializedArtifact(
        root=knowledge_parent.absolute(),
        content=knowledge_content,
        assets=(knowledge_parent / "assets").absolute(),
        receipt=knowledge_receipt,
        reused=False,
    )

    matrix_authority = next(
        authority
        for authority in release_matrix_stage_result.release_matrix_report.evaluation_plan.adapter_authorities
        if source_adapter is None
        or authority.task_adapter_manifest == source_adapter.manifest
    )
    verified_adapter = (
        verified_test_task_adapter(matrix_authority.task_adapter_manifest)
        if source_adapter is None
        else source_adapter
    )
    assert verified_adapter.manifest == matrix_authority.task_adapter_manifest
    assert verified_adapter.verification_receipt == (
        matrix_authority.verification_receipt
    )
    binding_settings = next(
        binding
        for binding in stored_candidate.closure.validation_context.active_task_bindings
        if binding.task_adapter_id == verified_adapter.manifest.task_adapter_id
    )
    activation = TaskAdapterActivationRecord.mint(
        scope_contract_id=scope_contract.scope_contract_id,
        task_family_id=binding_settings.task_family_id,
        task_adapter_id=binding_settings.task_adapter_id,
        task_adapter_manifest_id=verified_adapter.manifest.task_adapter_manifest_id,
        verification_receipt_id=(
            verified_adapter.verification_receipt.verification_receipt_id
        ),
        predecessor_activation_id=None,
        authority_id="test_task_adapter_authority",
        authority_version="test.task_adapter_authority.v1",
        authority_envelope_digest=digest("adapter-activation"),
    )
    active_adapter = ActiveTaskAdapterBinding(
        activation=activation,
        verified_adapter=verified_adapter,
    )
    adapter_case = verified_adapter.manifest.release_matrix_cases[0]
    case_context = adapter_case.task_context_binding
    runtime_contract = verified_adapter.manifest.runtime.to_dict()
    task_context_request = LaunchTaskContextRequest.mint(
        capability_tags=case_context.capability_tags,
        input_contract_fingerprint=case_context.input_contract_fingerprint,
        target_contract_fingerprint=case_context.target_contract_fingerprint,
        starting_artifact_refs=case_context.starting_artifact_refs,
        method_fingerprint=case_context.method_fingerprint,
        toolchain_fingerprint=case_context.toolchain_fingerprint,
        dependency_runtime_fingerprint=tree_or_blob_digest(
            canonical_json_bytes(runtime_contract)
        ),
        budget_hardware_envelope=case_context.budget_hardware_envelope,
        transfer_dimensions=case_context.transfer_dimensions,
    )
    verified_launch_artifacts = []
    for case_artifact in adapter_case.starting_artifacts:
        source_contents = {}
        source_descriptors = []
        for source_descriptor in case_artifact.source_files:
            payload = (
                f"new-task:{case_artifact.starting_artifact_ref}:"
                f"{source_descriptor.relative_path}"
            ).encode("utf-8")
            descriptor = SourceFileDescriptor(
                relative_path=source_descriptor.relative_path,
                digest=tree_or_blob_digest(payload),
                mode=source_descriptor.mode,
                size=len(payload),
            )
            source_descriptors.append(descriptor)
            source_contents[descriptor.relative_path] = payload
        launch_artifact = LaunchStartingArtifact.mint(
            starting_artifact_ref=case_artifact.starting_artifact_ref,
            mount_path=case_artifact.mount_path,
            materialized_tree_hash=source_tree_digest(
                {
                    descriptor.relative_path: (
                        descriptor.digest,
                        descriptor.mode,
                        descriptor.size,
                    )
                    for descriptor in source_descriptors
                }
            ),
            source_files=tuple(
                sorted(
                    source_descriptors,
                    key=lambda descriptor: descriptor.relative_path,
                )
            ),
        )
        verified_launch_artifacts.append(
            VerifiedLaunchStartingArtifact(
                artifact=launch_artifact,
                source_contents=source_contents,
            )
        )
    verified_launch_artifacts = tuple(
        sorted(
            verified_launch_artifacts,
            key=lambda item: item.artifact.starting_artifact_content_id,
        )
    )
    request = LaunchRequest.mint(
        binding=binding_settings,
        task_context_request=task_context_request,
        prompt_input_digest=digest("launch-prompt-input"),
        starting_artifact_content_ids={
            item.artifact.starting_artifact_ref: (
                item.artifact.starting_artifact_content_id
            )
            for item in verified_launch_artifacts
        },
        requested_coding_agent="codex",
        search_mode="generic",
        dependency_runtime_contract=runtime_contract,
        budget_fidelity_envelope={"fidelity": "full", "hours": 4},
        configuration_fingerprint=digest("launch-config"),
        empty_scope_bootstrap_authorization_id=None,
    )
    task_context = task_context_request.bind(
        binding=binding_settings,
        scope_contract=scope_contract,
    )
    starting_artifact_receipt = LaunchStartingArtifactMaterializationReceipt.mint(
        task_context_binding_id=task_context.task_context_binding_id,
        starting_artifacts=tuple(item.artifact for item in verified_launch_artifacts),
        materializer_id=settings.launch.starting_artifact_materializer_id,
        materializer_version=(settings.launch.starting_artifact_materializer_version),
        exact_dependency_ids=tuple(
            sorted(
                {
                    task_context.task_context_binding_id,
                    *(
                        item.artifact.starting_artifact_content_id
                        for item in verified_launch_artifacts
                    ),
                }
            )
        ),
    )
    verified_starting_artifacts = VerifiedLaunchStartingArtifacts(
        receipt=starting_artifact_receipt,
        starting_artifacts=verified_launch_artifacts,
    )

    currents = {
        PublicationArtifactKind.EXPERT_BASE_RELEASE: expert_current,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT: knowledge_current,
    }
    github = FakeGitHubResolver(
        currents,
        {
            PublicationArtifactKind.EXPERT_BASE_RELEASE: expert_intent,
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT: knowledge_intent,
        },
        {
            PublicationArtifactKind.EXPERT_BASE_RELEASE: expert_witness,
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT: knowledge_witness,
        },
    )
    materializer = FakeMaterializer(
        {
            PublicationArtifactKind.EXPERT_BASE_RELEASE: expert_artifact,
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT: knowledge_artifact,
        },
        expert_source,
        expert_package.publication_files,
    )
    task_adapters = FixedTaskAdapterProvider(active_adapter)
    starting_artifacts = FixedStartingArtifactProvider(verified_starting_artifacts)
    release_use = FixedReleaseUseAuthority(
        knowledge_current,
        knowledge_package,
        expert_package.manifest.release_id,
    )
    security = RecordingSecurityAuthority(settings)
    resolver = LaunchResolver(
        settings=settings,
        experiment_embedding_space=_experiment_embedding_space(),
        github_resolver=github,
        materializer=materializer,
        task_adapters=task_adapters,
        starting_artifacts=starting_artifacts,
        release_use_authority=release_use,
        security_authority=security,
        clock=FixedClock(),
    )
    return {
        "resolver": resolver,
        "request": request,
        "github": github,
        "task_adapters": task_adapters,
        "starting_artifacts": starting_artifacts,
        "release_use": release_use,
        "security": security,
        "expert_package": expert_package,
        "knowledge_package": knowledge_package,
        "evidence": ExpertLaunchEvidence(
            validation_context=stored_candidate.closure.validation_context,
            repository_map=stored_candidate.closure.repository_map,
            module_contracts=tuple(
                sorted(
                    stored_candidate.closure.module_contracts,
                    key=lambda module: module.module_contract_id,
                )
            ),
            release_matrix_stage_result=release_matrix_stage_result,
        ),
    }


@pytest.fixture
def resolver_case(tmp_path, monkeypatch):
    return build_resolver_case(tmp_path, monkeypatch)


def test_resolver_admits_exact_verified_release_matrix_interface(
    resolver_case,
):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])

    assert resolved.manifest.launch_request == resolver_case["request"]
    assert resolved.manifest.expert_manifest == resolver_case["expert_package"].manifest
    assert (
        resolved.manifest.knowledge_manifest
        == resolver_case["knowledge_package"].manifest
    )
    assert (
        resolved.manifest.knowledge_embedding_space.embedding_space_id
        != resolved.manifest.experiment_embedding_space.embedding_space_id
    )
    assert (
        resolved.manifest.knowledge_embedding_space.canonicalizer_version
        == "kapso.knowledge_embedding.v1"
    )
    assert (
        resolved.manifest.experiment_embedding_space.canonicalizer_version
        == "kapso.idea_embedding.v1"
    )
    assert (
        resolved.manifest.knowledge_embedding_space.provider,
        resolved.manifest.knowledge_embedding_space.model,
        resolved.manifest.knowledge_embedding_space.dimensions,
    ) == (
        resolved.manifest.experiment_embedding_space.provider,
        resolved.manifest.experiment_embedding_space.model,
        resolved.manifest.experiment_embedding_space.dimensions,
    )
    assert {
        resolved.manifest.knowledge_embedding_space.embedding_space_id,
        resolved.manifest.experiment_embedding_space.embedding_space_id,
    }.issubset(resolver_case["security"].checked_subject_ids)
    assert resolved.expert_evidence == resolver_case["evidence"]
    assert resolved.manifest.compatibility_receipt.task_adapter_compatibility_case_ids
    assert set(
        resolver_case["request"].starting_artifact_content_ids.values()
    ).issubset(resolver_case["security"].checked_subject_ids)
    assert resolver_case["github"].resolve_counts == {
        PublicationArtifactKind.EXPERT_BASE_RELEASE: 2,
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT: 2,
    }
    assert resolver_case["task_adapters"].resolve_count == 2
    assert resolver_case["release_use"].resolve_count == 1
    assert resolver_case["security"].resolve_count == 1
    assert resolver_case["starting_artifacts"].resolve_count == 1


def test_compatibility_admits_new_verified_bytes_at_exact_case_mount(
    resolver_case,
):
    active_binding = resolver_case["task_adapters"].binding
    base_manifest = active_binding.verified_adapter.manifest
    base_case = base_manifest.release_matrix_cases[0]
    artifact_ref = "artifact/new-task"
    package_root = "release_matrix_assets/new-task"
    matrix_payload = b"release matrix fixture"
    matrix_descriptor = SourceFileDescriptor(
        relative_path="input.bin",
        digest=tree_or_blob_digest(matrix_payload),
        mode="100644",
        size=len(matrix_payload),
    )
    matrix_artifact = TaskAdapterReleaseMatrixStartingArtifact.mint(
        starting_artifact_ref=artifact_ref,
        mount_path="inputs/task",
        package_source_root=package_root,
        materialized_tree_hash=source_tree_digest(
            {
                matrix_descriptor.relative_path: (
                    matrix_descriptor.digest,
                    matrix_descriptor.mode,
                    matrix_descriptor.size,
                )
            }
        ),
        source_files=(matrix_descriptor,),
    )
    context_values = base_case.task_context_binding.to_dict()
    context_values.pop("task_context_binding_id")
    context_values["starting_artifact_refs"] = (artifact_ref,)
    case_context = TaskContextBinding.mint(**context_values)
    case_values = base_case.to_dict()
    case_values.pop("release_matrix_case_id")
    case_values["task_context_binding"] = case_context
    case_values["starting_artifacts"] = (matrix_artifact,)
    release_case = TaskAdapterReleaseMatrixCase.mint(**case_values)
    source_contents = dict(active_binding.verified_adapter.source_contents)
    source_contents.update(
        {f"{package_root}/{matrix_descriptor.relative_path}": matrix_payload}
    )
    _, _, adapter_tree_hash = task_adapter_source(
        base_manifest.task_adapter_id,
        source_contents=source_contents,
    )
    manifest_values = base_manifest.to_dict()
    manifest_values.pop("task_adapter_manifest_id")
    manifest_values["release_matrix_cases"] = (release_case,)
    manifest_values["tree_hash"] = adapter_tree_hash
    manifest = type(base_manifest).mint(**manifest_values)
    verified_adapter = verified_test_task_adapter(
        manifest,
        source_contents=source_contents,
    )
    activation = TaskAdapterActivationRecord.mint(
        scope_contract_id=manifest.scope_contract_id,
        task_family_id=manifest.task_family_id,
        task_adapter_id=manifest.task_adapter_id,
        task_adapter_manifest_id=manifest.task_adapter_manifest_id,
        verification_receipt_id=(
            verified_adapter.verification_receipt.verification_receipt_id
        ),
        predecessor_activation_id=None,
        authority_id="test_task_adapter_authority",
        authority_version="test.task_adapter_authority.v1",
        authority_envelope_digest=digest("new-adapter-activation"),
    )
    adapter_binding = ActiveTaskAdapterBinding(
        activation=activation,
        verified_adapter=verified_adapter,
    )

    task_payload = b"new task bytes"
    task_descriptor = SourceFileDescriptor(
        relative_path="input.bin",
        digest=tree_or_blob_digest(task_payload),
        mode="100644",
        size=len(task_payload),
    )
    task_artifact = LaunchStartingArtifact.mint(
        starting_artifact_ref=artifact_ref,
        mount_path=matrix_artifact.mount_path,
        materialized_tree_hash=source_tree_digest(
            {
                task_descriptor.relative_path: (
                    task_descriptor.digest,
                    task_descriptor.mode,
                    task_descriptor.size,
                )
            }
        ),
        source_files=(task_descriptor,),
    )
    runtime_contract = manifest.runtime.to_dict()
    task_context_request = LaunchTaskContextRequest.mint(
        capability_tags=case_context.capability_tags,
        input_contract_fingerprint=case_context.input_contract_fingerprint,
        target_contract_fingerprint=case_context.target_contract_fingerprint,
        starting_artifact_refs=case_context.starting_artifact_refs,
        method_fingerprint=case_context.method_fingerprint,
        toolchain_fingerprint=case_context.toolchain_fingerprint,
        dependency_runtime_fingerprint=tree_or_blob_digest(
            canonical_json_bytes(runtime_contract)
        ),
        budget_hardware_envelope=case_context.budget_hardware_envelope,
        transfer_dimensions=case_context.transfer_dimensions,
    )
    request = LaunchRequest.mint(
        binding=resolver_case["request"].binding,
        task_context_request=task_context_request,
        prompt_input_digest=digest("new-artifact-prompt-input"),
        starting_artifact_content_ids={
            artifact_ref: task_artifact.starting_artifact_content_id
        },
        requested_coding_agent="codex",
        search_mode="generic",
        dependency_runtime_contract=runtime_contract,
        budget_fidelity_envelope={"fidelity": "full", "hours": 4},
        configuration_fingerprint=digest("new-artifact-config"),
        empty_scope_bootstrap_authorization_id=None,
    )
    task_context = task_context_request.bind(
        binding=request.binding,
        scope_contract=resolver_case["knowledge_package"].prepared.scope_contract,
    )
    receipt = LaunchStartingArtifactMaterializationReceipt.mint(
        task_context_binding_id=task_context.task_context_binding_id,
        starting_artifacts=(task_artifact,),
        materializer_id="launch_starting_artifact_materializer",
        materializer_version="kapso.launch_starting_artifact_materializer.v1",
        exact_dependency_ids=tuple(
            sorted(
                {
                    task_context.task_context_binding_id,
                    task_artifact.starting_artifact_content_id,
                }
            )
        ),
    )

    assert LaunchResolver._compatible_adapter_case_ids(
        request,
        adapter_binding,
        receipt,
    ) == (release_case.release_matrix_case_id,)

    wrong_mount_artifact = LaunchStartingArtifact.mint(
        starting_artifact_ref=artifact_ref,
        mount_path="inputs/other",
        materialized_tree_hash=task_artifact.materialized_tree_hash,
        source_files=task_artifact.source_files,
    )
    wrong_mount_request_values = request.to_dict()
    wrong_mount_request_values.pop("launch_request_id")
    wrong_mount_request_values["starting_artifact_content_ids"] = {
        artifact_ref: wrong_mount_artifact.starting_artifact_content_id
    }
    wrong_mount_request = LaunchRequest.mint(**wrong_mount_request_values)
    wrong_mount_receipt = LaunchStartingArtifactMaterializationReceipt.mint(
        task_context_binding_id=task_context.task_context_binding_id,
        starting_artifacts=(wrong_mount_artifact,),
        materializer_id="launch_starting_artifact_materializer",
        materializer_version="kapso.launch_starting_artifact_materializer.v1",
        exact_dependency_ids=tuple(
            sorted(
                {
                    task_context.task_context_binding_id,
                    wrong_mount_artifact.starting_artifact_content_id,
                }
            )
        ),
    )
    with pytest.raises(LaunchResolutionError, match="no compatible verified"):
        LaunchResolver._compatible_adapter_case_ids(
            wrong_mount_request,
            adapter_binding,
            wrong_mount_receipt,
        )

    other_ref = "artifact/other-task"
    other_ref_artifact = LaunchStartingArtifact.mint(
        starting_artifact_ref=other_ref,
        mount_path=task_artifact.mount_path,
        materialized_tree_hash=task_artifact.materialized_tree_hash,
        source_files=task_artifact.source_files,
    )
    other_context_values = task_context_request.to_dict()
    other_context_values.pop("task_context_request_id")
    other_context_values["starting_artifact_refs"] = (other_ref,)
    other_context_request = LaunchTaskContextRequest.mint(**other_context_values)
    other_request_values = request.to_dict()
    other_request_values.pop("launch_request_id")
    other_request_values["task_context_request"] = other_context_request
    other_request_values["starting_artifact_content_ids"] = {
        other_ref: other_ref_artifact.starting_artifact_content_id
    }
    other_request = LaunchRequest.mint(**other_request_values)
    other_task_context = other_context_request.bind(
        binding=request.binding,
        scope_contract=resolver_case["knowledge_package"].prepared.scope_contract,
    )
    other_ref_receipt = LaunchStartingArtifactMaterializationReceipt.mint(
        task_context_binding_id=other_task_context.task_context_binding_id,
        starting_artifacts=(other_ref_artifact,),
        materializer_id="launch_starting_artifact_materializer",
        materializer_version="kapso.launch_starting_artifact_materializer.v1",
        exact_dependency_ids=tuple(
            sorted(
                {
                    other_task_context.task_context_binding_id,
                    other_ref_artifact.starting_artifact_content_id,
                }
            )
        ),
    )
    with pytest.raises(LaunchResolutionError, match="no compatible verified"):
        LaunchResolver._compatible_adapter_case_ids(
            other_request,
            adapter_binding,
            other_ref_receipt,
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("method_fingerprint", digest("other-method")),
        ("toolchain_fingerprint", digest("other-toolchain")),
        ("input_contract_fingerprint", digest("other-input")),
        ("target_contract_fingerprint", digest("other-target")),
        ("budget_hardware_envelope", {"gpu": "other"}),
        ("capability_tags", ("other-capability",)),
        (
            "transfer_dimensions",
            {
                "dataset_family": "other",
                "runtime_family": "pytorch",
            },
        ),
    ),
)
def test_resolver_rejects_unverified_case_dimension(
    resolver_case,
    field_name,
    replacement,
):
    request_values = resolver_case["request"].to_dict()
    request_values.pop("launch_request_id")
    context_values = resolver_case["request"].task_context_request.to_dict()
    context_values.pop("task_context_request_id")
    context_values[field_name] = replacement
    request_values["task_context_request"] = LaunchTaskContextRequest.mint(
        **context_values
    )
    mismatched_request = LaunchRequest.mint(**request_values)

    with pytest.raises(LaunchResolutionError, match="no compatible verified"):
        resolver_case["resolver"].resolve(mismatched_request)


def test_verified_starting_artifact_rejects_content_id_without_exact_bytes():
    payload = b"verified task input"
    descriptor = SourceFileDescriptor(
        relative_path="input.bin",
        digest=tree_or_blob_digest(payload),
        mode="100644",
        size=len(payload),
    )
    artifact = LaunchStartingArtifact.mint(
        starting_artifact_ref="artifact/input",
        mount_path="inputs/task",
        materialized_tree_hash=source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
            }
        ),
        source_files=(descriptor,),
    )

    with pytest.raises(LaunchResolutionError, match="file descriptor"):
        VerifiedLaunchStartingArtifact(
            artifact=artifact,
            source_contents={"input.bin": payload + b"tampered"},
        )


def test_resolver_rejects_unconfigured_starting_artifact_materializer(
    resolver_case,
):
    provider = resolver_case["starting_artifacts"]
    receipt_values = provider.verified.receipt.to_dict()
    receipt_values.pop("materialization_receipt_id")
    receipt_values["materializer_version"] = "other.materializer.v1"
    provider.verified = VerifiedLaunchStartingArtifacts(
        receipt=LaunchStartingArtifactMaterializationReceipt.mint(**receipt_values),
        starting_artifacts=provider.verified.starting_artifacts,
    )

    with pytest.raises(LaunchResolutionError, match="differs from request or policy"):
        resolver_case["resolver"].resolve(resolver_case["request"])


def test_resolved_launch_rejects_forged_authority_and_evidence_splice(
    resolver_case,
):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])

    with pytest.raises(LaunchResolutionError, match="differ from"):
        replace(resolved, _resolver_authority=object())

    forged_clone = replace(resolved)
    with pytest.raises(LaunchResolutionError, match="live resolver authority"):
        forged_clone.require_resolver_authority()
    with ThreadPoolExecutor(max_workers=2) as executor:
        consume_futures = tuple(
            executor.submit(resolved.require_resolver_authority) for _ in range(2)
        )
    consume_exceptions = tuple(future.exception() for future in consume_futures)
    assert sum(exception is None for exception in consume_exceptions) == 1
    assert (
        sum(
            type(exception) is LaunchResolutionError for exception in consume_exceptions
        )
        == 1
    )

    validation_context = resolved.expert_evidence.validation_context
    if len(validation_context.active_task_bindings) > 1:
        alternate_bindings = validation_context.active_task_bindings[:-1]
    else:
        alternate_bindings = tuple(
            sorted(
                {
                    *validation_context.active_task_bindings,
                    CrossRunTaskBindingSettings(
                        scope_id=validation_context.scope_contract.scope_id,
                        task_family_id="relational_tabular_prediction",
                        task_adapter_id="relbench",
                    ),
                },
                key=lambda binding: (
                    binding.task_family_id,
                    binding.task_adapter_id,
                ),
            )
        )
    context_values = validation_context.to_dict()
    context_values.pop("validation_context_id")
    context_values["active_task_bindings"] = alternate_bindings
    alternate_context = type(validation_context).mint(**context_values)
    spliced_context_evidence = replace(
        resolved.expert_evidence,
        validation_context=alternate_context,
    )
    with pytest.raises(LaunchResolutionError, match="differ from"):
        replace(resolved, expert_evidence=spliced_context_evidence)

    stage = resolved.expert_evidence.release_matrix_stage_result
    report = stage.release_matrix_report
    task_evidence = report.task_execution_evidence
    assert task_evidence is not None
    alternate_transition_id = content_id(
        "expert-validation-transition",
        {"alternate": stage.authorization_transition_id},
    )
    transition_replacements = {
        stage.authorization_transition_id: alternate_transition_id
    }
    task_evidence_values = task_evidence.to_dict()
    task_evidence_values.pop("task_execution_evidence_id")
    for dependency_field in (
        "reservation_dependency_ids",
        "request_dependency_ids",
        "exact_dependency_ids",
    ):
        task_evidence_values[dependency_field] = tuple(
            sorted(
                transition_replacements.get(dependency_id, dependency_id)
                for dependency_id in task_evidence_values[dependency_field]
            )
        )
    alternate_task_evidence = type(task_evidence).mint(**task_evidence_values)
    report_replacements = {
        **transition_replacements,
        task_evidence.task_execution_evidence_id: (
            alternate_task_evidence.task_execution_evidence_id
        ),
    }
    report_values = report.to_dict()
    report_values.pop("release_matrix_report_id")
    report_values["task_execution_evidence"] = alternate_task_evidence
    report_values["exact_dependency_ids"] = tuple(
        sorted(
            report_replacements.get(dependency_id, dependency_id)
            for dependency_id in report.exact_dependency_ids
        )
    )
    alternate_report = type(report).mint(**report_values)
    stage_replacements = {
        **report_replacements,
        report.release_matrix_report_id: alternate_report.release_matrix_report_id,
    }
    stage_values = stage.to_dict()
    stage_values.pop("stage_result_record_id")
    stage_values["authorization_transition_id"] = alternate_transition_id
    stage_values["release_matrix_report"] = alternate_report
    stage_values["exact_dependency_ids"] = tuple(
        sorted(
            stage_replacements.get(dependency_id, dependency_id)
            for dependency_id in stage.exact_dependency_ids
        )
    )
    alternate_stage = type(stage).mint(**stage_values)
    spliced_stage_evidence = replace(
        resolved.expert_evidence,
        release_matrix_stage_result=alternate_stage,
    )
    with pytest.raises(LaunchResolutionError, match="differ from"):
        replace(resolved, expert_evidence=spliced_stage_evidence)

    with pytest.raises(LaunchResolutionError, match="typed evidence"):
        ExpertLaunchEvidence(
            validation_context=object(),
            repository_map=resolved.expert_evidence.repository_map,
            module_contracts=resolved.expert_evidence.module_contracts,
            release_matrix_stage_result=(
                resolved.expert_evidence.release_matrix_stage_result
            ),
        )


def test_abandoned_resolved_launch_is_not_retained(resolver_case):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    resolved_reference = ref(resolved)

    del resolved
    gc.collect()

    assert resolved_reference() is None


def test_resolver_rejects_runtime_contract_mismatch_before_policy_calls(
    resolver_case,
):
    request_values = resolver_case["request"].to_dict()
    request_values.pop("launch_request_id")
    request_values["dependency_runtime_contract"] = {
        **request_values["dependency_runtime_contract"],
        "architecture": "arm64",
    }
    mismatched_request = LaunchRequest.mint(**request_values)

    with pytest.raises(LaunchResolutionError, match="runtime contract differs"):
        resolver_case["resolver"].resolve(mismatched_request)

    assert resolver_case["release_use"].resolve_count == 0
    assert resolver_case["security"].resolve_count == 0


def test_resolver_rejects_current_change_during_resolution(resolver_case):
    resolver_case["github"].changed_kind = PublicationArtifactKind.KNOWLEDGE_SNAPSHOT

    with pytest.raises(LaunchResolutionError, match="changed during"):
        resolver_case["resolver"].resolve(resolver_case["request"])


def test_resolver_rejects_stale_release_before_materialization_policy(
    resolver_case,
):
    artifact_kind = PublicationArtifactKind.EXPERT_BASE_RELEASE
    current = resolver_case["github"].currents[artifact_kind]
    publication_values = current.pointer.publication_record.to_dict()
    publication_values.pop("publication_id")
    publication_values["published_at"] = "2026-07-01T00:00:00Z"
    stale_publication = GitHubPublicationRecord.mint(
        **publication_values,
    )
    resolver_case["github"].currents[artifact_kind] = replace(
        current,
        pointer=replace(
            current.pointer,
            publication_record=stale_publication,
        ),
    )

    with pytest.raises(LaunchResolutionError, match="freshness policy"):
        resolver_case["resolver"].resolve(resolver_case["request"])

    assert resolver_case["release_use"].resolve_count == 0
    assert resolver_case["security"].resolve_count == 0


def test_bootstrap_authorization_never_substitutes_for_missing_current(
    resolver_case,
):
    request_values = resolver_case["request"].to_dict()
    request_values.pop("launch_request_id")
    request_values["empty_scope_bootstrap_authorization_id"] = content_id(
        "scope-bootstrap-authorization",
        {"scope_id": "ml_ai"},
    )
    authorized_request = LaunchRequest.mint(**request_values)

    def missing_current(scope_id, artifact_kind):
        raise RuntimeError("CURRENT.json is missing")

    resolver_case["github"].resolve_current = missing_current
    with pytest.raises(RuntimeError, match="CURRENT.json is missing"):
        resolver_case["resolver"].resolve(authorized_request)
