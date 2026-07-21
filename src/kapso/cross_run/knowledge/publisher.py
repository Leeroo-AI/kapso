"""Deterministic release assembly for immutable knowledge snapshots."""

from __future__ import annotations

import io
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Protocol

import zstandard

from kapso.core.embeddings import (
    EmbeddingProvider,
    EmbeddingSettings as ProviderEmbeddingSettings,
    EmbeddingTelemetry,
    OpenAIEmbeddingProvider,
    complete_input_hash,
)
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.store import CatalogGenerationManifest
from kapso.cross_run.contracts import (
    ExpertScopeContract,
    PublicationArtifactKind,
)
from kapso.cross_run.github.publisher import (
    PublicationEnvelope,
    PublicationTelemetry,
    ReleaseAssetInput,
)
from kapso.cross_run.knowledge.index import (
    EmbeddingSpace,
    EmbeddingVector,
    EmbeddingVectorSet,
    SnapshotSearchIndex,
)
from kapso.cross_run.knowledge.package import (
    KnowledgeSnapshotPackage,
    KnowledgeSnapshotPackageBuilder,
)
from kapso.cross_run.settings import GitHubSettings, KnowledgeSettings

_KNOWLEDGE_MANIFEST_PATH = "snapshot.json"
_INDEX_DIRECTORY = "index"
_ARCHIVE_MEDIA_TYPE = "application/zstd"
_ARCHIVE_MODE = 0o644
_KNOWLEDGE_SNAPSHOT_ID_PREFIX = "knowledge-snapshot:sha256:"


class KnowledgeSnapshotPublicationError(ValueError):
    """A snapshot cannot be represented by the configured release bounds."""


def _require_knowledge_snapshot_id(value: object, name: str) -> str:
    snapshot_id = require_content_id(value, name)
    if not snapshot_id.startswith(_KNOWLEDGE_SNAPSHOT_ID_PREFIX):
        raise KnowledgeSnapshotPublicationError(
            f"{name} must identify a knowledge snapshot"
        )
    return snapshot_id


class KnowledgeReleasePublisher(Protocol):
    """The M2 publication authority consumed by the knowledge domain."""

    def publish(self, envelope: PublicationEnvelope) -> PublicationTelemetry: ...


@dataclass(frozen=True)
class KnowledgeSnapshotPublication:
    """Successful immutable publication and the exact scientific artifact."""

    package: KnowledgeSnapshotPackage
    telemetry: PublicationTelemetry

    def __post_init__(self) -> None:
        record = self.telemetry.publication_record
        if record.artifact_kind is not PublicationArtifactKind.KNOWLEDGE_SNAPSHOT:
            raise KnowledgeSnapshotPublicationError(
                "publication record is not a knowledge snapshot"
            )
        if record.artifact_id != self.package.manifest.snapshot_id:
            raise KnowledgeSnapshotPublicationError(
                "publication record names another snapshot"
            )


@dataclass(frozen=True)
class KnowledgeSnapshotBuild:
    """One verified package and separately attributable embedding telemetry."""

    package: KnowledgeSnapshotPackage
    embedding_telemetry: EmbeddingTelemetry | None


class KnowledgeSnapshotPublisher:
    """Package one verified snapshot and delegate its write-once GitHub transaction."""

    def __init__(
        self,
        publication_authority: KnowledgeReleasePublisher,
        github_settings: GitHubSettings,
        knowledge_settings: KnowledgeSettings,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._publication_authority = publication_authority
        self._github_settings = github_settings
        self._knowledge_settings = knowledge_settings
        self._embedding_provider = embedding_provider

    def build(
        self,
        scope_contract: ExpertScopeContract,
        catalog_generation: CatalogGenerationManifest,
        read_object_bytes: Callable[[str], bytes],
        *,
        parent_snapshot_ids: tuple[str, ...],
        sanitation_policy_version: str,
        retrieval_policy_version: str,
        published_at: str,
        publisher_attestation: Mapping[str, object],
    ) -> KnowledgeSnapshotBuild:
        """Rebuild canonical truth and every search sidecar from one exact generation."""

        if catalog_generation.generation_number == 0:
            if parent_snapshot_ids:
                raise KnowledgeSnapshotPublicationError(
                    "an EMPTY snapshot cannot name a scientific parent"
                )
        elif len(parent_snapshot_ids) != 1:
            raise KnowledgeSnapshotPublicationError(
                "a nonempty snapshot must name exactly one scientific parent"
            )
        else:
            _require_knowledge_snapshot_id(
                parent_snapshot_ids[0],
                "parent_snapshot_ids",
            )
        prepared = KnowledgeSnapshotPackageBuilder.prepare(
            scope_contract,
            catalog_generation,
            read_object_bytes,
        )
        vector_sets: tuple[EmbeddingVectorSet, ...] = ()
        telemetry: EmbeddingTelemetry | None = None
        if self._knowledge_settings.embeddings.enabled and prepared.retrieval_root_ids:
            expected_settings = self._provider_embedding_settings()
            provider = self._embedding_provider
            if provider is None:
                provider = OpenAIEmbeddingProvider(expected_settings)
            if provider.settings != expected_settings:
                raise KnowledgeSnapshotPublicationError(
                    "embedding provider settings differ from knowledge configuration"
                )
            source_texts = tuple(
                canonical_json_bytes(prepared.record_by_id(record_id)).decode("utf-8")
                for record_id in prepared.retrieval_root_ids
            )
            batch = provider.embed(source_texts)
            if len(batch.records) != len(prepared.retrieval_root_ids):
                raise KnowledgeSnapshotPublicationError(
                    "embedding provider returned an incomplete record closure"
                )
            if any(
                record.embedding_space_id != expected_settings.embedding_space_id
                or record.input_hash != complete_input_hash(source_text)
                for source_text, record in zip(source_texts, batch.records)
            ):
                raise KnowledgeSnapshotPublicationError(
                    "embedding result does not own its canonical source input"
                )
            space = EmbeddingSpace.mint(
                provider=expected_settings.provider,
                model=expected_settings.model,
                dimensions=expected_settings.dimensions,
                canonicalizer_version=expected_settings.canonicalizer_version,
            )
            if space.embedding_space_id != expected_settings.embedding_space_id.value:
                raise KnowledgeSnapshotPublicationError(
                    "persistent and provider embedding-space identities differ"
                )
            vectors = tuple(
                EmbeddingVector(
                    record_id=record_id,
                    input_digest=tree_or_blob_digest(source_text.encode("utf-8")),
                    values=record.vector,
                )
                for record_id, source_text, record in zip(
                    prepared.retrieval_root_ids,
                    source_texts,
                    batch.records,
                )
            )
            vector_sets = (EmbeddingVectorSet(space=space, vectors=vectors),)
            telemetry = batch.telemetry
        elif (
            not self._knowledge_settings.embeddings.enabled
            and self._embedding_provider is not None
        ):
            raise KnowledgeSnapshotPublicationError(
                "an embedding provider is invalid when embeddings are disabled or empty"
            )
        search_index = SnapshotSearchIndex.build(prepared, vector_sets)
        retrieval = self._knowledge_settings.retrieval
        package = KnowledgeSnapshotPackageBuilder.finalize(
            prepared,
            parent_snapshot_ids=parent_snapshot_ids,
            sanitation_policy_version=sanitation_policy_version,
            retrieval_policy_version=retrieval_policy_version,
            configuration_fingerprint=tree_or_blob_digest(
                canonical_json_bytes(self._knowledge_settings.to_dict())
            ),
            prompt_budget_policy=retrieval.to_dict(),
            published_at=published_at,
            publisher_attestation=publisher_attestation,
            search_files=search_index.files,
            embedding_sidecars=search_index.embedding_sidecars,
        )
        return KnowledgeSnapshotBuild(
            package=package,
            embedding_telemetry=telemetry,
        )

    def _provider_embedding_settings(self) -> ProviderEmbeddingSettings:
        configured = self._knowledge_settings.embeddings
        return ProviderEmbeddingSettings(
            enabled=configured.enabled,
            provider=configured.provider,
            model=configured.model,
            dimensions=configured.dimensions,
            batch_size=configured.batch_size,
            timeout_seconds=configured.timeout_seconds,
            max_retries=configured.max_retries,
            canonicalizer_version=configured.canonicalizer_version,
        )

    def publish(
        self,
        package: KnowledgeSnapshotPackage,
        *,
        expected_parent_sha: str,
        expected_current_snapshot_id: str | None,
        committed_at: str,
        validation_closure_ids: tuple[str, ...],
    ) -> KnowledgeSnapshotPublication:
        """Publish only bytes already closed by the verified snapshot manifest."""

        if not isinstance(package, KnowledgeSnapshotPackage):
            raise TypeError("package must be a KnowledgeSnapshotPackage")
        package.verify()
        if package.prepared.snapshot_kind == "EMPTY":
            if (
                expected_current_snapshot_id is not None
                or package.manifest.parent_snapshot_ids
            ):
                raise KnowledgeSnapshotPublicationError(
                    "an EMPTY publication cannot have a current scientific parent"
                )
        else:
            _require_knowledge_snapshot_id(
                expected_current_snapshot_id,
                "expected_current_snapshot_id",
            )
            if package.manifest.parent_snapshot_ids != (expected_current_snapshot_id,):
                raise KnowledgeSnapshotPublicationError(
                    "snapshot parent differs from resolved CURRENT identity"
                )
        for object_id in validation_closure_ids:
            require_content_id(object_id, "validation_closure_ids")
        complete_validation_closure = tuple(
            sorted(
                {
                    package.prepared.catalog_generation_id,
                    package.manifest.scope_contract_id,
                    *package.manifest.entry_state_refs,
                    *package.manifest.included_assertion_ids,
                    *package.manifest.included_revocation_ids,
                    *validation_closure_ids,
                }
            )
        )
        with tempfile.TemporaryDirectory(prefix="kapso-knowledge-release-") as root:
            release_root = Path(root)
            source_tree = release_root / "source"
            asset_root = release_root / "assets"
            source_tree.mkdir()
            asset_root.mkdir()
            self._write_source_tree(package, source_tree)
            assets = self._write_release_assets(package, asset_root)
            envelope = PublicationEnvelope(
                artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
                artifact_id=package.manifest.snapshot_id,
                scope_id=package.manifest.scope_id,
                expected_parent_sha=expected_parent_sha,
                source_tree=source_tree,
                manifest_relative_path=_KNOWLEDGE_MANIFEST_PATH,
                assets=assets,
                tag=(
                    self._github_settings.knowledge_tag_prefix
                    + package.manifest.snapshot_id.rsplit(":", 1)[1]
                ),
                committed_at=committed_at,
                validation_closure_ids=complete_validation_closure,
            )
            telemetry = self._publication_authority.publish(envelope)
        return KnowledgeSnapshotPublication(package=package, telemetry=telemetry)

    @staticmethod
    def _write_source_tree(
        package: KnowledgeSnapshotPackage,
        source_tree: Path,
    ) -> None:
        """Keep rebuildable indexes out of Git while retaining canonical truth."""

        source_files = {
            relative_path: payload
            for relative_path, payload in package.files.items()
            if PurePosixPath(relative_path).parts[0] != _INDEX_DIRECTORY
        }
        for relative_path, payload in sorted(source_files.items()):
            destination = source_tree / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                _ARCHIVE_MODE,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

    def _write_release_assets(
        self,
        package: KnowledgeSnapshotPackage,
        asset_root: Path,
    ) -> tuple[ReleaseAssetInput, ...]:
        shards = self._shard_files(package.files)
        if len(shards) > self._github_settings.release_asset_count_limit:
            raise KnowledgeSnapshotPublicationError(
                "knowledge package exceeds the configured release asset count"
            )
        assets: list[ReleaseAssetInput] = []
        total_size = 0
        width = len(str(len(shards) - 1))
        for position, payload in enumerate(shards):
            name = f"knowledge-snapshot-{position:0{width}d}.tar.zst"
            path = asset_root / name
            path.write_bytes(payload)
            size = len(payload)
            total_size += size
            assets.append(
                ReleaseAssetInput(
                    path=path,
                    name=name,
                    media_type=_ARCHIVE_MEDIA_TYPE,
                    size=size,
                    sha256=tree_or_blob_digest(payload),
                )
            )
        if total_size > self._github_settings.materialized_asset_size_bytes:
            raise KnowledgeSnapshotPublicationError(
                "knowledge release assets exceed the materialized-size bound"
            )
        return tuple(assets)

    def _shard_files(self, files: Mapping[str, bytes]) -> tuple[bytes, ...]:
        groups: list[tuple[tuple[str, bytes], ...]] = []
        current: tuple[tuple[str, bytes], ...] = ()
        for relative_path, payload in sorted(files.items()):
            candidate = (*current, (relative_path, payload))
            encoded = self._archive(candidate)
            if len(encoded) <= self._github_settings.release_asset_size_bytes:
                current = candidate
                continue
            if current:
                groups.append(current)
                current = ((relative_path, payload),)
                encoded = self._archive(current)
            if len(encoded) > self._github_settings.release_asset_size_bytes:
                raise KnowledgeSnapshotPublicationError(
                    f"package file cannot fit one release shard: {relative_path}"
                )
        if current:
            groups.append(current)
        if not groups:
            raise KnowledgeSnapshotPublicationError("knowledge package is empty")
        return tuple(self._archive(group) for group in groups)

    def _archive(self, files: tuple[tuple[str, bytes], ...]) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for relative_path, payload in files:
                member = tarfile.TarInfo(relative_path)
                member.size = len(payload)
                member.mode = _ARCHIVE_MODE
                member.mtime = 0
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                tar.addfile(member, io.BytesIO(payload))
        compression_parameters = zstandard.ZstdCompressionParameters.from_level(
            self._knowledge_settings.archive_compression_level,
            window_log=self._github_settings.zstd_window_size_bytes.bit_length() - 1,
            write_checksum=1,
            write_content_size=1,
            write_dict_id=0,
        )
        compressor = zstandard.ZstdCompressor(compression_params=compression_parameters)
        return compressor.compress(buffer.getvalue())


__all__ = [
    "KnowledgeSnapshotBuild",
    "KnowledgeSnapshotPublication",
    "KnowledgeSnapshotPublicationError",
    "KnowledgeSnapshotPublisher",
]
