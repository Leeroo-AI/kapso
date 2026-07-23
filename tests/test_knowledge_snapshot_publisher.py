import io
import tarfile
from dataclasses import replace

import pytest
import zstandard

import kapso.cross_run.knowledge.publisher as knowledge_publisher_module
from kapso.core.config import load_config
from kapso.core.embeddings import (
    EmbeddingBatch,
    EmbeddingRecord,
    EmbeddingSettings,
    EmbeddingTelemetry,
    complete_input_hash,
)
from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
)
from kapso.cross_run.github.publisher import PublicationTelemetry
from kapso.cross_run.github.command import GitHubCompareAndSwapError
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.knowledge.index import SnapshotSearchIndex
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackageBuilder
from kapso.cross_run.knowledge.publisher import (
    KnowledgeSnapshotPublicationError,
    KnowledgeSnapshotPublisher,
)
from kapso.cross_run.settings import CrossRunSettings
from test_knowledge_snapshot_package import (
    empty_generation,
    populated_generation,
    populated_generation_with_release_use_revocations,
    scope_contract,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
COMMITTED_AT = "2026-07-21T12:00:00Z"


def settings():
    return CrossRunSettings.from_dict(load_config(CANONICAL_CONFIG_PATH)["cross_run"])


def indexed_empty_package():
    scope = scope_contract()
    prepared = KnowledgeSnapshotPackageBuilder.prepare_empty(
        scope,
        empty_generation(scope),
    )
    search_index = SnapshotSearchIndex.build(prepared)
    return KnowledgeSnapshotPackageBuilder.finalize(
        prepared,
        parent_snapshot_ids=(),
        sanitation_policy_version="kapso.sanitation.v1",
        retrieval_policy_version="kapso.retrieval.v1",
        configuration_fingerprint=tree_or_blob_digest(b"knowledge-config"),
        prompt_budget_policy={"maximum_bytes": 48000, "maximum_records": 24},
        published_at=COMMITTED_AT,
        publisher_attestation={"issuer": "test-publisher"},
        search_files=search_index.files,
        embedding_sidecars=search_index.embedding_sidecars,
    )


class RecordingPublicationAuthority:
    def __init__(self):
        self.envelopes = []
        self.source_files = []
        self.asset_payloads = []

    def publish(self, envelope):
        source_files = {
            path.relative_to(envelope.source_tree).as_posix(): path.read_bytes()
            for path in sorted(envelope.source_tree.rglob("*"))
            if path.is_file()
        }
        asset_payloads = tuple(
            (asset.name, asset.path.read_bytes()) for asset in envelope.assets
        )
        self.envelopes.append(envelope)
        self.source_files.append(source_files)
        self.asset_payloads.append(asset_payloads)
        release_assets = tuple(
            GitHubReleaseAsset(
                asset_id=f"asset_{position}",
                name=asset.name,
                media_type=asset.media_type,
                size=asset.size,
                sha256=asset.sha256,
            )
            for position, asset in enumerate(envelope.assets)
        )
        publication_record = GitHubPublicationRecord.mint(
            artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            artifact_id=envelope.artifact_id,
            repository_node_id="repository_node",
            repository_full_name="Leeroo-AI/kapso-knowledge",
            commit_sha="b" * 40,
            immutable_release_id="release_1",
            tag=envelope.tag,
            assets=release_assets,
            release_attestation_ref="release-attestation.json",
            published_at=envelope.committed_at,
            publisher_identity="test_publisher",
        )
        return PublicationTelemetry(
            publication_record=publication_record,
            expected_parent_sha=envelope.expected_parent_sha,
            source_commit_sha="b" * 40,
            pointer_commit_sha="c" * 40,
            source_tree_digest=source_tree_digest(
                {
                    path: (tree_or_blob_digest(payload), "100644", len(payload))
                    for path, payload in source_files.items()
                }
            ),
            validation_closure_ids=envelope.validation_closure_ids,
            idempotent_replay=False,
        )


class ValidatingPublicationAuthority(RecordingPublicationAuthority):
    def __init__(self, github_settings, state_root):
        super().__init__()
        self._materializer = GitHubArtifactMaterializer(
            object(),
            github_settings,
            state_root,
        )
        self.materialized_tree_digest = None

    def publish(self, envelope):
        source_files = {
            path.relative_to(envelope.source_tree).as_posix(): (
                tree_or_blob_digest(path.read_bytes()),
                "100644",
            )
            for path in sorted(envelope.source_tree.rglob("*"))
            if path.is_file()
        }
        manifest_bytes = (
            envelope.source_tree / envelope.manifest_relative_path
        ).read_bytes()
        self.materialized_tree_digest = self._materializer.validate_local_package(
            artifact_kind=envelope.artifact_kind,
            artifact_id=envelope.artifact_id,
            manifest_relative_path=envelope.manifest_relative_path,
            manifest_digest=tree_or_blob_digest(manifest_bytes),
            assets=envelope.assets,
            source_files=source_files,
        )
        return super().publish(envelope)


class FailingPublicationAuthority:
    def publish(self, envelope):
        raise GitHubCompareAndSwapError("simulated CURRENT conflict")


class DeterministicEmbeddingProvider:
    def __init__(self, configured):
        self.settings = EmbeddingSettings(
            enabled=configured.enabled,
            provider=configured.provider,
            model=configured.model,
            dimensions=configured.dimensions,
            batch_size=configured.batch_size,
            timeout_seconds=configured.timeout_seconds,
            max_retries=configured.max_retries,
            canonicalizer_version=configured.canonicalizer_version,
        )
        self.inputs = ()

    def embed(self, texts):
        self.inputs = tuple(texts)
        vector = (1.0, *(0.0 for _ in range(self.settings.dimensions - 1)))
        return EmbeddingBatch(
            records=tuple(
                EmbeddingRecord(
                    provider=self.settings.provider,
                    model=self.settings.model,
                    dimensions=self.settings.dimensions,
                    canonicalizer_version=self.settings.canonicalizer_version,
                    input_hash=complete_input_hash(text),
                    vector=vector,
                )
                for text in self.inputs
            ),
            telemetry=EmbeddingTelemetry(
                provider=self.settings.provider,
                model=self.settings.model,
                call_count=1,
                input_tokens=10,
                duration_seconds=0.1,
                cost_usd=None,
            ),
        )


def unpack_assets(asset_payloads):
    extracted = {}
    decompressor = zstandard.ZstdDecompressor()
    for _, payload in asset_payloads:
        tar_bytes = decompressor.decompress(payload)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            for member in archive.getmembers():
                extracted[member.name] = archive.extractfile(member).read()
    return extracted


def test_publisher_keeps_indexes_out_of_git_and_release_contains_exact_package(
    tmp_path,
):
    package = indexed_empty_package()
    configured = settings()
    authority = ValidatingPublicationAuthority(
        configured.github,
        tmp_path / "materializer-state",
    )
    publisher = KnowledgeSnapshotPublisher(
        authority,
        configured.github,
        configured.knowledge,
    )

    result = publisher.publish(
        package,
        expected_parent_sha="a" * 40,
        expected_current_snapshot_id=None,
        committed_at=COMMITTED_AT,
        validation_closure_ids=(),
    )

    assert result.package is package
    assert all(not path.startswith("index/") for path in authority.source_files[0])
    assert "snapshot.json" in authority.source_files[0]
    assert unpack_assets(authority.asset_payloads[0]) == package.files
    assert authority.materialized_tree_digest is not None
    assert authority.envelopes[0].validation_closure_ids == tuple(
        sorted(
            (
                package.prepared.catalog_generation_id,
                package.manifest.scope_contract_id,
            )
        )
    )


def test_publisher_validates_the_release_use_revocation_projection():
    (
        scope,
        _,
        _,
        generation,
        objects,
        revocations,
    ) = populated_generation_with_release_use_revocations()
    parent_snapshot_id = content_id("knowledge-snapshot", {"parent": 1})
    prepared = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        generation,
        objects.__getitem__,
    )
    package = KnowledgeSnapshotPackageBuilder.finalize(
        prepared,
        parent_snapshot_ids=(parent_snapshot_id,),
        sanitation_policy_version="kapso.sanitation.v1",
        retrieval_policy_version="kapso.retrieval.v1",
        configuration_fingerprint=tree_or_blob_digest(b"knowledge-config"),
        prompt_budget_policy={"maximum_records": 24},
        published_at=COMMITTED_AT,
        publisher_attestation={"issuer": "test-publisher"},
    )
    configured = settings()
    authority = RecordingPublicationAuthority()
    publisher = KnowledgeSnapshotPublisher(
        authority,
        configured.github,
        configured.knowledge,
    )

    publisher.publish(
        package,
        expected_parent_sha="a" * 40,
        expected_current_snapshot_id=parent_snapshot_id,
        committed_at=COMMITTED_AT,
        validation_closure_ids=(),
    )

    expected_revocation_ids = {revocation.revocation_id for revocation in revocations}
    assert package.manifest.included_revocation_ids == ()
    assert expected_revocation_ids.issubset(
        authority.envelopes[0].validation_closure_ids
    )


def test_release_sharding_and_archive_bytes_are_deterministic():
    package = indexed_empty_package()
    configured = settings()
    probe = KnowledgeSnapshotPublisher(
        RecordingPublicationAuthority(),
        configured.github,
        configured.knowledge,
    )
    largest_single_file_archive = max(
        len(probe._archive(((path, payload),)))
        for path, payload in package.files.items()
    )
    bounded_github = replace(
        configured.github,
        release_asset_size_bytes=largest_single_file_archive,
    )
    first_authority = RecordingPublicationAuthority()
    second_authority = RecordingPublicationAuthority()
    first = KnowledgeSnapshotPublisher(
        first_authority,
        bounded_github,
        configured.knowledge,
    )
    second = KnowledgeSnapshotPublisher(
        second_authority,
        bounded_github,
        configured.knowledge,
    )

    publish_fields = {
        "expected_parent_sha": "a" * 40,
        "expected_current_snapshot_id": None,
        "committed_at": COMMITTED_AT,
        "validation_closure_ids": (),
    }
    first.publish(package, **publish_fields)
    second.publish(package, **publish_fields)

    assert len(first_authority.asset_payloads[0]) > 1
    assert first_authority.asset_payloads[0] == second_authority.asset_payloads[0]
    assert unpack_assets(first_authority.asset_payloads[0]) == package.files


def test_build_embeds_complete_canonical_roots_and_binds_vector_sidecar():
    scope, idea, _, generation, objects = populated_generation()
    configured = settings()
    embedding_provider = DeterministicEmbeddingProvider(configured.knowledge.embeddings)
    publisher = KnowledgeSnapshotPublisher(
        RecordingPublicationAuthority(),
        configured.github,
        configured.knowledge,
        embedding_provider,
    )

    result = publisher.build(
        scope,
        generation,
        objects.__getitem__,
        parent_snapshot_ids=(content_id("knowledge-snapshot", {"parent": 1}),),
        sanitation_policy_version="kapso.sanitation.v1",
        retrieval_policy_version="kapso.retrieval.v1",
        published_at=COMMITTED_AT,
        publisher_attestation={"issuer": "test-publisher"},
    )

    assert result.embedding_telemetry is not None
    assert len(embedding_provider.inputs) == 1
    assert idea.prior_idea_id in embedding_provider.inputs[0]
    assert len(result.package.manifest.embedding_sidecars) == 1
    assert (
        result.package.manifest.prompt_budget_policy["prompt_byte_budget"]
        == configured.knowledge.retrieval.prompt_byte_budget
    )


def test_populated_build_with_embeddings_disabled_never_constructs_a_provider(
    monkeypatch,
):
    scope, _, _, generation, objects = populated_generation()
    configured = settings()
    disabled_knowledge = replace(
        configured.knowledge,
        embeddings=replace(configured.knowledge.embeddings, enabled=False),
    )

    def reject_provider_construction(provider_settings):
        raise AssertionError("disabled build constructed an embedding provider")

    monkeypatch.setattr(
        knowledge_publisher_module,
        "OpenAIEmbeddingProvider",
        reject_provider_construction,
    )
    result = KnowledgeSnapshotPublisher(
        RecordingPublicationAuthority(),
        configured.github,
        disabled_knowledge,
    ).build(
        scope,
        generation,
        objects.__getitem__,
        parent_snapshot_ids=(content_id("knowledge-snapshot", {"parent": 1}),),
        sanitation_policy_version="kapso.sanitation.v1",
        retrieval_policy_version="kapso.retrieval.v1",
        published_at=COMMITTED_AT,
        publisher_attestation={"issuer": "test-publisher"},
    )

    assert result.embedding_telemetry is None
    assert result.package.manifest.embedding_sidecars == ()


def test_publication_rejects_scientific_parent_mismatch_before_m2():
    scope, _, _, generation, objects = populated_generation()
    configured = settings()
    expected_current = content_id("knowledge-snapshot", {"parent": 1})
    package = (
        KnowledgeSnapshotPublisher(
            RecordingPublicationAuthority(),
            configured.github,
            replace(
                configured.knowledge,
                embeddings=replace(configured.knowledge.embeddings, enabled=False),
            ),
        )
        .build(
            scope,
            generation,
            objects.__getitem__,
            parent_snapshot_ids=(expected_current,),
            sanitation_policy_version="kapso.sanitation.v1",
            retrieval_policy_version="kapso.retrieval.v1",
            published_at=COMMITTED_AT,
            publisher_attestation={"issuer": "test-publisher"},
        )
        .package
    )

    with pytest.raises(
        KnowledgeSnapshotPublicationError,
        match="resolved CURRENT identity",
    ):
        KnowledgeSnapshotPublisher(
            RecordingPublicationAuthority(),
            configured.github,
            configured.knowledge,
        ).publish(
            package,
            expected_parent_sha="a" * 40,
            expected_current_snapshot_id=content_id(
                "knowledge-snapshot",
                {"parent": 2},
            ),
            committed_at=COMMITTED_AT,
            validation_closure_ids=(),
        )


def test_populated_build_rejects_non_snapshot_parent_namespace():
    scope, _, _, generation, objects = populated_generation()
    configured = settings()
    publisher = KnowledgeSnapshotPublisher(
        RecordingPublicationAuthority(),
        configured.github,
        replace(
            configured.knowledge,
            embeddings=replace(configured.knowledge.embeddings, enabled=False),
        ),
    )

    with pytest.raises(
        KnowledgeSnapshotPublicationError,
        match="must identify a knowledge snapshot",
    ):
        publisher.build(
            scope,
            generation,
            objects.__getitem__,
            parent_snapshot_ids=(content_id("transfer-episode", {"parent": 1}),),
            sanitation_policy_version="kapso.sanitation.v1",
            retrieval_policy_version="kapso.retrieval.v1",
            published_at=COMMITTED_AT,
            publisher_attestation={"issuer": "test-publisher"},
        )


def test_publication_rejects_non_snapshot_current_namespace_before_m2():
    scope, _, _, generation, objects = populated_generation()
    configured = settings()
    package = (
        KnowledgeSnapshotPublisher(
            RecordingPublicationAuthority(),
            configured.github,
            replace(
                configured.knowledge,
                embeddings=replace(configured.knowledge.embeddings, enabled=False),
            ),
        )
        .build(
            scope,
            generation,
            objects.__getitem__,
            parent_snapshot_ids=(content_id("knowledge-snapshot", {"parent": 1}),),
            sanitation_policy_version="kapso.sanitation.v1",
            retrieval_policy_version="kapso.retrieval.v1",
            published_at=COMMITTED_AT,
            publisher_attestation={"issuer": "test-publisher"},
        )
        .package
    )

    with pytest.raises(
        KnowledgeSnapshotPublicationError,
        match="must identify a knowledge snapshot",
    ):
        KnowledgeSnapshotPublisher(
            RecordingPublicationAuthority(),
            configured.github,
            configured.knowledge,
        ).publish(
            package,
            expected_parent_sha="a" * 40,
            expected_current_snapshot_id=content_id(
                "transfer-episode",
                {"parent": 1},
            ),
            committed_at=COMMITTED_AT,
            validation_closure_ids=(),
        )


def test_m2_compare_and_swap_failure_propagates_without_fallback():
    package = indexed_empty_package()
    configured = settings()
    publisher = KnowledgeSnapshotPublisher(
        FailingPublicationAuthority(),
        configured.github,
        configured.knowledge,
    )

    with pytest.raises(GitHubCompareAndSwapError, match="simulated CURRENT conflict"):
        publisher.publish(
            package,
            expected_parent_sha="a" * 40,
            expected_current_snapshot_id=None,
            committed_at=COMMITTED_AT,
            validation_closure_ids=(),
        )
