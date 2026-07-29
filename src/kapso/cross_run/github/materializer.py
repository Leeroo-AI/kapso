"""Verified, content-addressed materialization of immutable release assets."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import stat
import tempfile
import ctypes
from contextlib import ExitStack
from dataclasses import dataclass
from io import DEFAULT_BUFFER_SIZE
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import BinaryIO, Mapping, Protocol

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    require_identifier,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    SecurityDenylistSnapshot,
    SourceFileDescriptor,
    StrictContract,
)
from kapso.cross_run.github.command import GitHubCommandClient
from kapso.cross_run.github.resolver import ResolvedGitHubArtifact
from kapso.cross_run.settings import GitHubSettings
from kapso.cross_run.source_archives import SourceArchiveExtractor

_RECEIPT_NAME = "VERIFIED.json"
_CACHE_LEASE_NAME = ".github-artifact-cache.lock"
_TRANSIENT_PREFIXES = (".staging-", ".pruning-", ".validation-")
SOURCE_ARCHIVE_EXTRACTOR_VERSION = "kapso.source_archive_extractor.v1"
_RENAME_NOREPLACE = 1


class MaterializationError(RuntimeError):
    """A release package or cache entry failed complete verification."""


class CacheCorruptionError(MaterializationError):
    """A committed content-addressed cache entry no longer verifies."""


@dataclass(frozen=True)
class CacheVerificationReceipt(StrictContract):
    artifact_kind: PublicationArtifactKind
    artifact_id: str
    materialized_tree_digest: str
    manifest_relative_path: str
    manifest_digest: str
    cache_tree_digest: str
    asset_digests: Mapping[str, str]

    def _validate(self) -> None:
        require_content_id(self.artifact_id, "artifact_id")
        manifest_path = PurePosixPath(self.manifest_relative_path)
        if (
            manifest_path.is_absolute()
            or ".." in manifest_path.parts
            or manifest_path.as_posix() != self.manifest_relative_path
        ):
            raise MaterializationError("receipt manifest path is invalid")
        for name in (
            "materialized_tree_digest",
            "manifest_digest",
            "cache_tree_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", value
            ):
                raise MaterializationError(f"receipt {name} must be a sha256 digest")
        if not self.asset_digests:
            raise MaterializationError("receipt asset digest closure is empty")
        for name, digest in self.asset_digests.items():
            if not name or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise MaterializationError("receipt asset digest is invalid")


@dataclass(frozen=True)
class SourceArchiveExtractionReceipt(StrictContract):
    """Exact source tree deterministically extracted from one verified asset."""

    extraction_receipt_id: str
    artifact_id: str
    source_archive_ref: str
    source_archive_digest: str
    source_tree_hash: str
    source_tree_files: tuple[SourceFileDescriptor, ...]
    extractor_version: str

    CONTENT_NAMESPACE = "source-archive-extraction-receipt"
    IDENTITY_FIELD = "extraction_receipt_id"

    def _validate(self) -> None:
        require_content_id(self.artifact_id, "source archive artifact_id")
        source_ref = PurePosixPath(self.source_archive_ref)
        if (
            not self.source_archive_ref
            or source_ref.is_absolute()
            or len(source_ref.parts) != 1
            or source_ref.as_posix() != self.source_archive_ref
            or not self.source_archive_ref.endswith((".tar", ".tar.zst"))
        ):
            raise MaterializationError("source archive reference is invalid")
        for value, name in (
            (self.source_archive_digest, "source archive digest"),
            (self.source_tree_hash, "source tree hash"),
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise MaterializationError(f"{name} is invalid")
        paths = tuple(item.relative_path for item in self.source_tree_files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise MaterializationError(
                "source tree files must be non-empty, sorted, and unique"
            )
        source_paths = tuple(PurePosixPath(path) for path in paths)
        if any(
            source_path in other_path.parents
            for position, source_path in enumerate(source_paths)
            for other_path in source_paths[position + 1 :]
        ):
            raise MaterializationError(
                "source tree files contain a file/directory collision"
            )
        expected_tree_hash = source_tree_digest(
            {
                item.relative_path: (item.digest, item.mode, item.size)
                for item in self.source_tree_files
            }
        )
        if self.source_tree_hash != expected_tree_hash:
            raise MaterializationError(
                "source tree hash differs from its exact file descriptor"
            )
        require_identifier(self.extractor_version, "source archive extractor_version")


@dataclass(frozen=True)
class MaterializedArtifact:
    root: Path
    content: Path
    assets: Path
    receipt: CacheVerificationReceipt
    reused: bool


@dataclass(frozen=True)
class ExpertReleaseSourceSnapshot:
    """Exact expert release manifest and source bytes read under one cache lease."""

    release_manifest: ExpertBaseReleaseManifest
    source_extraction_receipt: SourceArchiveExtractionReceipt
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if (
            type(self.release_manifest) is not ExpertBaseReleaseManifest
            or type(self.source_extraction_receipt)
            is not SourceArchiveExtractionReceipt
            or not isinstance(self.source_contents, Mapping)
        ):
            raise MaterializationError(
                "expert release source snapshot requires exact typed authorities"
            )
        frozen_contents = MappingProxyType(dict(self.source_contents))
        object.__setattr__(self, "source_contents", frozen_contents)
        manifest = self.release_manifest
        receipt = self.source_extraction_receipt
        descriptors = {
            descriptor.relative_path: descriptor
            for descriptor in receipt.source_tree_files
        }
        if (
            receipt.artifact_id != manifest.release_id
            or receipt.source_archive_ref != manifest.source_archive_ref
            or manifest.checksums.get(manifest.source_archive_ref)
            != receipt.source_archive_digest
            or set(frozen_contents) != set(descriptors)
        ):
            raise MaterializationError(
                "expert release source snapshot differs from its release closure"
            )
        for relative_path, descriptor in descriptors.items():
            payload = frozen_contents[relative_path]
            if (
                type(payload) is not bytes
                or len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.digest
            ):
                raise MaterializationError(
                    "expert release source snapshot bytes differ from its descriptor"
                )


class LocalReleaseAsset(Protocol):
    path: Path
    name: str
    size: int
    sha256: str


class GitHubArtifactMaterializer:
    """Download, verify, extract, and atomically expose one resolved release."""

    def __init__(
        self,
        client: GitHubCommandClient,
        settings: GitHubSettings,
        state_root: Path,
    ) -> None:
        self.client = client
        self.settings = settings
        self.source_archive_extractor = SourceArchiveExtractor(
            zstd_window_size_bytes=settings.zstd_window_size_bytes,
            error_type=MaterializationError,
        )
        cache_path = PurePosixPath(settings.cache_path)
        if cache_path.is_absolute():
            raise MaterializationError("GitHub cache path must be relative")
        self.state_root = state_root.absolute()
        self.cache_root = self.state_root / settings.cache_path
        self._validate_cache_ancestors()

    def validate_local_package(
        self,
        *,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
        manifest_relative_path: str,
        manifest_digest: str,
        assets: tuple[LocalReleaseAsset, ...],
        source_files: Mapping[str, tuple[str, str]],
    ) -> str:
        """Prove local release assets recreate the exact validated source files."""
        self._validate_asset_bounds(assets)
        with self._cache_lease():
            kind_directory = self._ensure_cache_kind_directory(artifact_kind)
            self._remove_abandoned_transient_entries(kind_directory)
            with ExitStack() as descriptor_stack:
                kind_descriptor = self._open_cache_kind_directory(
                    kind_directory,
                    artifact_kind,
                )
                descriptor_stack.callback(os.close, kind_descriptor)
                self._validate_open_cache_kind_directory(
                    kind_descriptor,
                    kind_directory,
                    artifact_kind,
                )
                with tempfile.TemporaryDirectory(
                    prefix=".validation-",
                    dir=self._descriptor_path(kind_descriptor),
                ) as staging_name:
                    return self._validate_local_package_staging(
                        Path(staging_name),
                        artifact_kind=artifact_kind,
                        artifact_id=artifact_id,
                        manifest_relative_path=manifest_relative_path,
                        manifest_digest=manifest_digest,
                        assets=assets,
                        source_files=source_files,
                    )

    def _validate_local_package_staging(
        self,
        staging: Path,
        *,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
        manifest_relative_path: str,
        manifest_digest: str,
        assets: tuple[LocalReleaseAsset, ...],
        source_files: Mapping[str, tuple[str, str]],
    ) -> str:
        assets_directory = staging / "assets"
        content_directory = staging / "content"
        assets_directory.mkdir()
        content_directory.mkdir()
        extracted_paths: dict[str, str] = {}
        extracted_bytes = 0
        extracted_entries = 0
        for asset in assets:
            if asset.size > self.settings.release_asset_size_bytes:
                raise MaterializationError(
                    "release asset exceeds configured size limit"
                )
            staged_asset = assets_directory / asset.name
            shutil.copyfile(asset.path, staged_asset)
            if staged_asset.stat().st_size != asset.size:
                raise MaterializationError("local release asset size mismatch")
            if self._file_digest(staged_asset) != asset.sha256:
                raise MaterializationError("local release asset digest mismatch")
            if self._is_archive(asset.name):
                added_bytes, added_entries = self._extract_archive(
                    staged_asset,
                    content_directory,
                    extracted_paths,
                    self.settings.materialized_asset_size_bytes - extracted_bytes,
                    self.settings.archive_entry_limit - extracted_entries,
                )
                extracted_bytes += added_bytes
                extracted_entries += added_entries
        manifest_path = content_directory / manifest_relative_path
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise MaterializationError("release package manifest is missing")
        manifest_bytes = self._read_control_file(
            manifest_path, "release package manifest"
        )
        if tree_or_blob_digest(manifest_bytes) != manifest_digest:
            raise MaterializationError("release package manifest digest mismatch")
        manifest = self._parse_manifest(artifact_kind, manifest_bytes)
        packaged_artifact_id = self._manifest_artifact_id(manifest)
        if packaged_artifact_id != artifact_id:
            raise MaterializationError("release package artifact identity mismatch")
        declared_content_paths = self._verify_manifest_checksums(
            manifest.checksums,
            content_directory,
            assets_directory,
            manifest_relative_path,
        )
        self._verify_non_archive_asset_closure(
            manifest.checksums,
            {asset.name: asset.sha256 for asset in assets},
        )
        if not set(source_files).issubset(declared_content_paths):
            raise MaterializationError(
                "release package omits part of the validated source closure"
            )
        self._verify_content_closure(content_directory, declared_content_paths)
        for relative_path, (expected_digest, expected_mode) in source_files.items():
            packaged_path = content_directory / relative_path
            if packaged_path.is_symlink() or not packaged_path.is_file():
                raise MaterializationError(
                    f"release package omits source file: {relative_path}"
                )
            if self._file_digest(packaged_path) != expected_digest:
                raise MaterializationError(
                    f"release package source digest mismatch: {relative_path}"
                )
            actual_mode = "100755" if packaged_path.stat().st_mode & 0o111 else "100644"
            if actual_mode != expected_mode:
                raise MaterializationError(
                    f"release package source mode mismatch: {relative_path}"
                )
        return self._source_tree_digest(content_directory, declared_content_paths)

    def materialize(self, resolved: ResolvedGitHubArtifact) -> MaterializedArtifact:
        with self._cache_lease():
            return self._materialize(resolved)

    def inspect_source_archive(
        self,
        materialized: MaterializedArtifact,
        source_archive_ref: str,
    ) -> SourceArchiveExtractionReceipt:
        """Re-extract one verified asset and attest its exact source-only tree."""
        with self._cache_lease():
            receipt, source_archive, source_digest = self._verified_source_archive(
                materialized,
                source_archive_ref,
            )
            kind_directory = materialized.root.parent
            with tempfile.TemporaryDirectory(
                prefix=".validation-source-",
                dir=kind_directory,
            ) as staging_name:
                source_tree = Path(staging_name) / "source"
                source_tree.mkdir()
                extraction_receipt = self._extract_source_tree(
                    receipt=receipt,
                    source_archive=source_archive,
                    source_archive_digest=source_digest,
                    destination=source_tree,
                )
                self._reverify_source_archive(
                    materialized,
                    receipt,
                    source_archive,
                    source_digest,
                )
                return extraction_receipt

    def inspect_expert_release_source(
        self,
        materialized: MaterializedArtifact,
        *,
        maximum_entries: int,
        maximum_bytes: int,
    ) -> ExpertReleaseSourceSnapshot:
        """Read one verified expert manifest and exact source tree atomically."""

        if (
            type(maximum_entries) is not int
            or maximum_entries < 1
            or type(maximum_bytes) is not int
            or maximum_bytes < 1
        ):
            raise MaterializationError(
                "expert source inspection bounds must be positive integers"
            )
        with self._cache_lease():
            _, manifest = self._inspect_expert_release_manifest_under_lease(
                materialized
            )
            (
                verified_receipt,
                source_archive,
                source_digest,
            ) = self._verified_source_archive(
                materialized,
                manifest.source_archive_ref,
            )
            kind_directory = materialized.root.parent
            with tempfile.TemporaryDirectory(
                prefix=".validation-expert-source-",
                dir=kind_directory,
            ) as staging_name:
                source_tree = Path(staging_name) / "source"
                source_tree.mkdir(mode=0o700)
                extraction_receipt = self._extract_source_tree(
                    receipt=verified_receipt,
                    source_archive=source_archive,
                    source_archive_digest=source_digest,
                    destination=source_tree,
                    maximum_entries=min(
                        maximum_entries,
                        self.settings.archive_entry_limit,
                    ),
                    maximum_bytes=min(
                        maximum_bytes,
                        self.settings.materialized_asset_size_bytes,
                    ),
                )
                source_contents = self._read_source_tree_contents(
                    source_tree,
                    extraction_receipt.source_tree_files,
                )
                self._reverify_source_archive(
                    materialized,
                    verified_receipt,
                    source_archive,
                    source_digest,
                )
                return ExpertReleaseSourceSnapshot(
                    release_manifest=manifest,
                    source_extraction_receipt=extraction_receipt,
                    source_contents=source_contents,
                )

    def inspect_expert_release_manifest(
        self,
        materialized: MaterializedArtifact,
    ) -> ExpertBaseReleaseManifest:
        """Read one canonical expert manifest without extracting its source."""

        with self._cache_lease():
            _, manifest = self._inspect_expert_release_manifest_under_lease(
                materialized
            )
            return manifest

    def read_verified_content_files(
        self,
        materialized: MaterializedArtifact,
        relative_paths: tuple[str, ...],
        *,
        maximum_bytes: int,
    ) -> Mapping[str, bytes]:
        """Read bounded authenticated records while the cache entry stays valid."""

        if type(materialized) is not MaterializedArtifact:
            raise MaterializationError(
                "verified content reads require one materialized artifact"
            )
        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or maximum_bytes > self.settings.materialized_asset_size_bytes
        ):
            raise MaterializationError(
                "verified content reads require a configured size limit"
            )
        if (
            not relative_paths
            or relative_paths != tuple(sorted(set(relative_paths)))
            or len(relative_paths) > self.settings.archive_entry_limit
        ):
            raise MaterializationError(
                "verified content paths must be non-empty, sorted, and bounded"
            )
        normalized_paths = tuple(PurePosixPath(path) for path in relative_paths)
        if any(
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() == "."
            or path.as_posix() != relative_path
            for path, relative_path in zip(normalized_paths, relative_paths)
        ):
            raise MaterializationError("verified content path is invalid")
        with self._cache_lease():
            expected_root = (
                self.cache_root
                / materialized.receipt.artifact_kind.value
                / materialized.receipt.artifact_id.rsplit(":", 1)[1]
            )
            if materialized.root != expected_root:
                raise CacheCorruptionError(
                    "materialized artifact is outside the authorized cache"
                )
            before = self._read_and_verify_receipt(
                materialized.root,
                materialized.receipt.artifact_kind,
            )
            if (
                before != materialized.receipt
                or materialized.content != materialized.root / "content"
                or materialized.assets != materialized.root / "assets"
            ):
                raise CacheCorruptionError(
                    "materialized artifact differs from its verified cache entry"
                )
            if before.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE:
                raise MaterializationError(
                    "verified control-record reads require an expert base release"
                )
            manifest_payload = self._read_control_file(
                materialized.content / before.manifest_relative_path,
                "expert release manifest",
                error_type=CacheCorruptionError,
            )
            manifest = ExpertBaseReleaseManifest.from_json_bytes(manifest_payload)
            if (
                manifest_payload != manifest.to_json_bytes()
                or tree_or_blob_digest(manifest_payload) != before.manifest_digest
                or manifest.release_id != before.artifact_id
            ):
                raise CacheCorruptionError(
                    "expert release manifest differs from its cache receipt"
                )
            missing_checksums = set(relative_paths) - set(manifest.checksums)
            if missing_checksums:
                raise CacheCorruptionError(
                    "verified content record lacks an authenticated manifest checksum"
                )
            with ExitStack() as descriptors:
                content_descriptor = os.open(
                    materialized.content,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                )
                descriptors.callback(os.close, content_descriptor)
                payloads = {}
                remaining_bytes = maximum_bytes
                for relative_path in relative_paths:
                    payload = self._read_relative_control_file(
                        content_descriptor,
                        PurePosixPath(relative_path),
                        f"verified content record {relative_path}",
                        maximum_bytes=remaining_bytes,
                    )
                    payloads[relative_path] = payload
                    remaining_bytes -= len(payload)
            if any(
                tree_or_blob_digest(payloads[relative_path])
                != manifest.checksums[relative_path]
                for relative_path in relative_paths
            ):
                raise CacheCorruptionError(
                    "verified content record differs from its manifest checksum"
                )
            after = self._read_and_verify_receipt(
                materialized.root,
                materialized.receipt.artifact_kind,
            )
            if after != before:
                raise CacheCorruptionError(
                    "materialized artifact changed during verified content read"
                )
            return MappingProxyType(payloads)

    def _read_relative_control_file(
        self,
        root_descriptor: int,
        relative_path: PurePosixPath,
        description: str,
        *,
        maximum_bytes: int,
    ) -> bytes:
        with ExitStack() as descriptors:
            directory_descriptor = os.dup(root_descriptor)
            descriptors.callback(os.close, directory_descriptor)
            for part in relative_path.parts[:-1]:
                child_descriptor = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_descriptor,
                )
                descriptors.callback(os.close, child_descriptor)
                directory_descriptor = child_descriptor
            file_descriptor = os.open(
                relative_path.parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                os.close(file_descriptor)
                raise CacheCorruptionError(f"{description} must be a regular file")
            if metadata.st_size > maximum_bytes:
                os.close(file_descriptor)
                raise CacheCorruptionError(
                    f"{description} exceeds configured control bound"
                )
            with os.fdopen(file_descriptor, "rb") as file_handle:
                payload = file_handle.read(maximum_bytes + 1)
            if len(payload) != metadata.st_size:
                raise CacheCorruptionError(
                    f"{description} changed during its verified read"
                )
            return payload

    def _inspect_expert_release_manifest_under_lease(
        self,
        materialized: MaterializedArtifact,
    ) -> tuple[CacheVerificationReceipt, ExpertBaseReleaseManifest]:
        if (
            type(materialized) is not MaterializedArtifact
            or materialized.receipt.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
        ):
            raise MaterializationError(
                "expert manifest inspection requires an expert base release"
            )
        expected_root = (
            self.cache_root
            / PublicationArtifactKind.EXPERT_BASE_RELEASE.value
            / materialized.receipt.artifact_id.rsplit(":", 1)[1]
        )
        if materialized.root != expected_root:
            raise CacheCorruptionError(
                "materialized expert release is outside the authorized cache"
            )
        receipt = self._read_and_verify_receipt(
            materialized.root,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
        )
        if (
            receipt != materialized.receipt
            or materialized.content != materialized.root / "content"
            or materialized.assets != materialized.root / "assets"
        ):
            raise CacheCorruptionError(
                "materialized expert release differs from its verified cache entry"
            )
        manifest_payload = self._read_control_file(
            materialized.content / receipt.manifest_relative_path,
            "expert release manifest",
            error_type=CacheCorruptionError,
        )
        manifest = ExpertBaseReleaseManifest.from_json_bytes(manifest_payload)
        if (
            manifest_payload != manifest.to_json_bytes()
            or tree_or_blob_digest(manifest_payload) != receipt.manifest_digest
            or manifest.release_id != receipt.artifact_id
        ):
            raise CacheCorruptionError(
                "expert release manifest differs from its cache receipt"
            )
        return receipt, manifest

    def extract_verified_source_archive(
        self,
        *,
        materialized: MaterializedArtifact,
        expected: SourceArchiveExtractionReceipt,
        destination: Path,
        destination_parent_descriptor: int,
    ) -> SourceArchiveExtractionReceipt:
        """Recreate one attested source tree in a fresh private destination."""

        destination_parent_identity = self._validate_source_destination(
            destination,
            destination_parent_descriptor,
        )
        with self._cache_lease():
            if (
                self._validate_source_destination(
                    destination,
                    destination_parent_descriptor,
                )
                != destination_parent_identity
            ):
                raise MaterializationError(
                    "source extraction parent changed before materialization"
                )
            receipt, source_archive, source_digest = self._verified_source_archive(
                materialized,
                expected.source_archive_ref,
            )
            if (
                expected.artifact_id != receipt.artifact_id
                or expected.source_archive_digest != source_digest
                or expected.extractor_version != SOURCE_ARCHIVE_EXTRACTOR_VERSION
            ):
                raise MaterializationError(
                    "expected source extraction differs from verified asset"
                )
            with tempfile.TemporaryDirectory(
                prefix=".source-materialization-",
                dir=self._descriptor_path(destination_parent_descriptor),
            ) as staging_name:
                staged_source = Path(staging_name) / "source"
                staged_source.mkdir(mode=0o700)
                observed = self._extract_source_tree(
                    receipt=receipt,
                    source_archive=source_archive,
                    source_archive_digest=source_digest,
                    destination=staged_source,
                )
                if observed != expected:
                    raise MaterializationError(
                        "extracted source tree differs from expected receipt"
                    )
                self._reverify_source_archive(
                    materialized,
                    receipt,
                    source_archive,
                    source_digest,
                )
                self._publish_source_tree(
                    staged_source,
                    destination,
                    destination_parent_descriptor,
                    destination_parent_identity,
                    expected,
                )
                return observed

    def _verified_source_archive(
        self,
        materialized: MaterializedArtifact,
        source_archive_ref: str,
    ) -> tuple[CacheVerificationReceipt, Path, str]:
        source_ref = PurePosixPath(source_archive_ref)
        if (
            not source_archive_ref
            or source_ref.is_absolute()
            or len(source_ref.parts) != 1
            or source_ref.as_posix() != source_archive_ref
        ):
            raise MaterializationError("source archive reference is invalid")
        if not self._is_archive(source_archive_ref):
            raise MaterializationError("source archive is not a supported archive")
        expected_root = (
            self.cache_root
            / materialized.receipt.artifact_kind.value
            / materialized.receipt.artifact_id.rsplit(":", 1)[1]
        )
        if materialized.root != expected_root:
            raise CacheCorruptionError(
                "materialized artifact is outside the authorized cache"
            )
        receipt = self._read_and_verify_receipt(
            materialized.root,
            materialized.receipt.artifact_kind,
        )
        if (
            receipt != materialized.receipt
            or materialized.content != materialized.root / "content"
            or materialized.assets != materialized.root / "assets"
        ):
            raise CacheCorruptionError(
                "materialized artifact differs from its verified cache entry"
            )
        source_digest = receipt.asset_digests.get(source_archive_ref)
        if source_digest is None:
            raise MaterializationError(
                "source archive is absent from the verified asset closure"
            )
        source_archive = materialized.assets / source_archive_ref
        if source_archive.is_symlink() or not source_archive.is_file():
            raise CacheCorruptionError("source archive asset is not regular")
        if self._file_digest(source_archive) != source_digest:
            raise CacheCorruptionError("source archive asset digest changed")
        return receipt, source_archive, source_digest

    def _extract_source_tree(
        self,
        *,
        receipt: CacheVerificationReceipt,
        source_archive: Path,
        source_archive_digest: str,
        destination: Path,
        maximum_entries: int | None = None,
        maximum_bytes: int | None = None,
    ) -> SourceArchiveExtractionReceipt:
        self._extract_archive(
            source_archive,
            destination,
            {},
            (
                self.settings.materialized_asset_size_bytes
                if maximum_bytes is None
                else maximum_bytes
            ),
            (
                self.settings.archive_entry_limit
                if maximum_entries is None
                else maximum_entries
            ),
        )
        source_files = self._source_tree_files(destination)
        return SourceArchiveExtractionReceipt.mint(
            artifact_id=receipt.artifact_id,
            source_archive_ref=source_archive.name,
            source_archive_digest=source_archive_digest,
            source_tree_hash=source_tree_digest(
                {
                    item.relative_path: (item.digest, item.mode, item.size)
                    for item in source_files
                }
            ),
            source_tree_files=source_files,
            extractor_version=SOURCE_ARCHIVE_EXTRACTOR_VERSION,
        )

    def _source_tree_files(
        self,
        source_tree: Path,
    ) -> tuple[SourceFileDescriptor, ...]:
        return self.source_archive_extractor.source_tree_files(source_tree)

    @staticmethod
    def _read_source_tree_contents(
        source_tree: Path,
        descriptors: tuple[SourceFileDescriptor, ...],
    ) -> Mapping[str, bytes]:
        contents: dict[str, bytes] = {}
        for descriptor in descriptors:
            source_path = source_tree / descriptor.relative_path
            if source_path.is_symlink() or not source_path.is_file():
                raise MaterializationError(
                    "expert source snapshot contains a non-regular file"
                )
            with source_path.open("rb") as source_handle:
                payload = source_handle.read(descriptor.size + 1)
            if (
                len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.digest
            ):
                raise MaterializationError(
                    "expert source snapshot changed during inspection"
                )
            contents[descriptor.relative_path] = payload
        return MappingProxyType(contents)

    def _reverify_source_archive(
        self,
        materialized: MaterializedArtifact,
        receipt: CacheVerificationReceipt,
        source_archive: Path,
        source_digest: str,
    ) -> None:
        if (
            self._file_digest(source_archive) != source_digest
            or self._read_and_verify_receipt(
                materialized.root,
                materialized.receipt.artifact_kind,
            )
            != receipt
        ):
            raise CacheCorruptionError(
                "verified source archive changed during extraction"
            )

    def _validate_source_destination(
        self,
        destination: Path,
        destination_parent_descriptor: int,
    ) -> tuple[int, int]:
        if (
            not destination.is_absolute()
            or destination != Path(os.path.abspath(destination))
            or destination == self.cache_root
            or self.cache_root in destination.parents
        ):
            raise MaterializationError(
                "source extraction destination must be absent and normalized"
            )
        parent = destination.parent
        if (
            parent in {Path("/"), Path.home()}
            or parent.is_symlink()
            or not parent.is_dir()
            or parent.resolve() != parent
        ):
            raise MaterializationError(
                "source extraction parent must be a normalized real directory"
            )
        metadata = parent.stat(follow_symlinks=False)
        opened_parent = os.fstat(destination_parent_descriptor)
        if not stat.S_ISDIR(opened_parent.st_mode) or (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            raise MaterializationError(
                "source extraction parent differs from its pinned descriptor"
            )
        if destination.name in os.listdir(
            destination_parent_descriptor
        ) or os.path.lexists(destination):
            raise MaterializationError(
                "source extraction destination must be absent and normalized"
            )
        if opened_parent.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
            raise MaterializationError("source extraction parent must be private")
        return opened_parent.st_dev, opened_parent.st_ino

    def _publish_source_tree(
        self,
        staged_source: Path,
        destination: Path,
        destination_parent_descriptor: int,
        expected_parent_identity: tuple[int, int],
        expected: SourceArchiveExtractionReceipt,
    ) -> None:
        with ExitStack() as descriptors:
            staging_parent_descriptor = os.open(
                staged_source.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptors.callback(os.close, staging_parent_descriptor)
            source_descriptor = os.open(
                staged_source.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=staging_parent_descriptor,
            )
            descriptors.callback(os.close, source_descriptor)
            source_metadata = os.fstat(source_descriptor)
            named_source_metadata = os.stat(
                staged_source.name,
                dir_fd=staging_parent_descriptor,
                follow_symlinks=False,
            )
            parent_metadata = os.fstat(destination_parent_descriptor)
            if (
                not stat.S_ISDIR(source_metadata.st_mode)
                or (source_metadata.st_dev, source_metadata.st_ino)
                != (named_source_metadata.st_dev, named_source_metadata.st_ino)
                or source_metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
                or destination.name in os.listdir(destination_parent_descriptor)
                or (parent_metadata.st_dev, parent_metadata.st_ino)
                != expected_parent_identity
                or parent_metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
            ):
                raise MaterializationError(
                    "source extraction staging or destination changed before publication"
                )
            source_files = self._source_tree_files(
                self._descriptor_path(source_descriptor)
            )
            observed_tree_hash = source_tree_digest(
                {
                    item.relative_path: (item.digest, item.mode, item.size)
                    for item in source_files
                }
            )
            named_source_metadata = os.stat(
                staged_source.name,
                dir_fd=staging_parent_descriptor,
                follow_symlinks=False,
            )
            if (
                source_files != expected.source_tree_files
                or observed_tree_hash != expected.source_tree_hash
                or (source_metadata.st_dev, source_metadata.st_ino)
                != (named_source_metadata.st_dev, named_source_metadata.st_ino)
            ):
                raise MaterializationError(
                    "source tree differs from expected receipt at publication"
                )
            libc = ctypes.CDLL(None, use_errno=True)
            if not hasattr(libc, "renameat2"):
                raise MaterializationError(
                    "atomic source-tree publication is unavailable"
                )
            rename_at2 = libc.renameat2
            rename_at2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            result = rename_at2(
                staging_parent_descriptor,
                os.fsencode(staged_source.name),
                destination_parent_descriptor,
                os.fsencode(destination.name),
                _RENAME_NOREPLACE,
            )
            if result != 0:
                error_number = ctypes.get_errno()
                raise MaterializationError(
                    "atomic source-tree publication failed: "
                    f"{os.strerror(error_number)}"
                )
            os.fsync(destination_parent_descriptor)

    def _materialize(self, resolved: ResolvedGitHubArtifact) -> MaterializedArtifact:
        self._validate_cache_ancestors()
        record = resolved.pointer.publication_record
        self._validate_asset_bounds(record.assets)
        canonical_kind_directory = self._ensure_cache_kind_directory(
            record.artifact_kind
        )
        self._remove_abandoned_transient_entries(canonical_kind_directory)
        with ExitStack() as descriptor_stack:
            kind_descriptor = self._open_cache_kind_directory(
                canonical_kind_directory,
                record.artifact_kind,
            )
            descriptor_stack.callback(os.close, kind_descriptor)
            self._validate_open_cache_kind_directory(
                kind_descriptor,
                canonical_kind_directory,
                record.artifact_kind,
            )
            anchored_kind_directory = self._descriptor_path(kind_descriptor)
            target_name = record.artifact_id.rsplit(":", 1)[1]
            canonical_target = canonical_kind_directory / target_name
            children = self._bounded_descriptor_children(
                kind_descriptor,
                self.settings.cache_entry_limit,
                "cache kind directory",
            )
            matching = tuple(path for path in children if path.name == target_name)
            if matching:
                target = matching[0]
                if target.is_symlink():
                    raise CacheCorruptionError("cache entry cannot be a symlink")
                receipt = self._verify_cache_entry(
                    target,
                    resolved,
                    record.artifact_kind,
                )
                self._validate_open_cache_kind_directory(
                    kind_descriptor,
                    canonical_kind_directory,
                    record.artifact_kind,
                )
                return MaterializedArtifact(
                    root=canonical_target,
                    content=canonical_target / "content",
                    assets=canonical_target / "assets",
                    receipt=receipt,
                    reused=True,
                )
            if len(children) >= self.settings.cache_entry_limit:
                raise MaterializationError(
                    "cache kind directory is at configured capacity"
                )
            with tempfile.TemporaryDirectory(
                prefix=".staging-", dir=anchored_kind_directory
            ) as staging_name:
                return self._materialize_staging(
                    resolved,
                    canonical_target,
                    Path(staging_name),
                    kind_descriptor,
                )

    def _materialize_staging(
        self,
        resolved: ResolvedGitHubArtifact,
        target: Path,
        staging: Path,
        kind_descriptor: int,
    ) -> MaterializedArtifact:
        record = resolved.pointer.publication_record
        assets_directory = staging / "assets"
        content_directory = staging / "content"
        assets_directory.mkdir()
        content_directory.mkdir()
        extracted_paths: dict[str, str] = {}
        extracted_bytes = 0
        extracted_entries = 0
        for asset in record.assets:
            if asset.size > self.settings.release_asset_size_bytes:
                raise MaterializationError(
                    "release asset exceeds configured size limit"
                )
            asset_path = assets_directory / asset.name
            downloaded = self.client.download_release_asset(
                record.repository_full_name,
                asset.asset_id,
                asset_path,
                asset.size,
            )
            if downloaded != asset_path or not asset_path.is_file():
                raise MaterializationError("release asset download path mismatch")
            if asset_path.stat().st_size != asset.size:
                raise MaterializationError("downloaded release asset size mismatch")
            if self._file_digest(asset_path) != asset.sha256:
                raise MaterializationError("downloaded release asset digest mismatch")
            if self._is_archive(asset.name):
                added_bytes, added_entries = self._extract_archive(
                    asset_path,
                    content_directory,
                    extracted_paths,
                    self.settings.materialized_asset_size_bytes - extracted_bytes,
                    self.settings.archive_entry_limit - extracted_entries,
                )
                extracted_bytes += added_bytes
                extracted_entries += added_entries
                if extracted_bytes > self.settings.materialized_asset_size_bytes:
                    raise MaterializationError(
                        "extracted release exceeds configured size limit"
                    )
                if extracted_entries > self.settings.archive_entry_limit:
                    raise MaterializationError(
                        "release exceeds configured archive entry limit"
                    )
        manifest_path = content_directory / resolved.pointer.manifest_relative_path
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise MaterializationError("materialized manifest is missing")
        manifest_bytes = self._read_control_file(manifest_path, "materialized manifest")
        if tree_or_blob_digest(manifest_bytes) != resolved.pointer.manifest_digest:
            raise MaterializationError("materialized manifest digest mismatch")
        manifest = self._parse_manifest(record.artifact_kind, manifest_bytes)
        manifest_artifact_id = self._manifest_artifact_id(manifest)
        if manifest_artifact_id != record.artifact_id:
            raise MaterializationError("materialized manifest identity mismatch")
        declared_content_paths = self._verify_manifest_checksums(
            manifest.checksums,
            content_directory,
            assets_directory,
            resolved.pointer.manifest_relative_path,
        )
        self._verify_non_archive_asset_closure(
            manifest.checksums,
            {asset.name: asset.sha256 for asset in record.assets},
        )
        self._verify_content_closure(content_directory, declared_content_paths)
        if (
            self._source_tree_digest(content_directory, declared_content_paths)
            != resolved.pointer.materialized_tree_digest
        ):
            raise MaterializationError(
                "materialized tree differs from published package descriptor"
            )
        cache_tree_digest = self._tree_digest(staging)
        receipt = CacheVerificationReceipt(
            artifact_kind=record.artifact_kind,
            artifact_id=record.artifact_id,
            materialized_tree_digest=resolved.pointer.materialized_tree_digest,
            manifest_relative_path=resolved.pointer.manifest_relative_path,
            manifest_digest=resolved.pointer.manifest_digest,
            cache_tree_digest=cache_tree_digest,
            asset_digests={asset.name: asset.sha256 for asset in record.assets},
        )
        receipt_path = staging / _RECEIPT_NAME
        receipt_path.write_bytes(receipt.to_json_bytes())
        self._flush_tree(staging)
        self._make_read_only(staging)
        canonical_kind_directory = target.parent
        self._validate_open_cache_kind_directory(
            kind_descriptor,
            canonical_kind_directory,
            record.artifact_kind,
        )
        os.replace(
            staging.name,
            target.name,
            src_dir_fd=kind_descriptor,
            dst_dir_fd=kind_descriptor,
        )
        os.fsync(kind_descriptor)
        self._validate_open_cache_kind_directory(
            kind_descriptor,
            canonical_kind_directory,
            record.artifact_kind,
        )
        installed_target = self._descriptor_path(kind_descriptor) / target.name
        if installed_target.is_symlink() or not installed_target.is_dir():
            raise CacheCorruptionError("installed cache entry is not canonical")
        return MaterializedArtifact(
            root=target,
            content=target / "content",
            assets=target / "assets",
            receipt=receipt,
            reused=False,
        )

    def inspect(self) -> tuple[CacheVerificationReceipt, ...]:
        with self._cache_lease():
            return self._inspect()

    def _inspect(self) -> tuple[CacheVerificationReceipt, ...]:
        self._validate_cache_ancestors()
        if not self.cache_root.exists():
            return ()
        receipts: list[CacheVerificationReceipt] = []
        for kind_directory in self._cache_kind_directories():
            self._remove_abandoned_transient_entries(kind_directory)
            for entry in self._bounded_children(
                kind_directory,
                self.settings.cache_entry_limit,
                "cache kind directory",
            ):
                if entry.is_symlink():
                    raise CacheCorruptionError("cache entry cannot be a symlink")
                if entry.name.startswith(_TRANSIENT_PREFIXES):
                    if not entry.is_dir():
                        raise CacheCorruptionError(
                            "cache transient entry must be a directory"
                        )
                    continue
                artifact_kind = PublicationArtifactKind(kind_directory.name)
                receipts.append(self._read_and_verify_receipt(entry, artifact_kind))
        return tuple(receipts)

    def prune(self, pinned_artifact_ids: tuple[str, ...]) -> tuple[str, ...]:
        with self._cache_lease():
            return self._prune(pinned_artifact_ids)

    def _prune(self, pinned_artifact_ids: tuple[str, ...]) -> tuple[str, ...]:
        self._validate_cache_ancestors()
        if len(pinned_artifact_ids) != len(set(pinned_artifact_ids)):
            raise MaterializationError("pinned artifact IDs must be unique")
        for artifact_id in pinned_artifact_ids:
            require_content_id(artifact_id, "pinned_artifact_ids")
        entries: list[tuple[float, Path, CacheVerificationReceipt]] = []
        if not self.cache_root.exists():
            return ()
        kind_directories = self._cache_kind_directories()
        for kind_directory in kind_directories:
            self._remove_abandoned_transient_entries(kind_directory)
            children = self._bounded_children(
                kind_directory,
                self.settings.cache_entry_limit,
                "cache kind directory",
            )
            for entry in children:
                if entry.is_symlink():
                    raise CacheCorruptionError("cache entry cannot be a symlink")
                if entry.name.startswith(_TRANSIENT_PREFIXES):
                    if entry.exists() and not entry.is_dir():
                        raise CacheCorruptionError(
                            "cache transient entry must be a directory"
                        )
                    continue
                receipt = self._read_and_verify_receipt(
                    entry,
                    PublicationArtifactKind(kind_directory.name),
                )
                entries.append((entry.stat().st_mtime, entry, receipt))
        retained = set(
            receipt.artifact_id
            for _, _, receipt in sorted(entries, reverse=True)[
                : self.settings.cache_retention_releases
            ]
        )
        retained.update(pinned_artifact_ids)
        removed: list[str] = []
        for _, entry, receipt in entries:
            if receipt.artifact_id in retained:
                continue
            tombstone = entry.with_name(f".pruning-{entry.name}")
            if tombstone.exists() or tombstone.is_symlink():
                raise CacheCorruptionError("cache pruning tombstone already exists")
            artifact_kind = receipt.artifact_kind
            canonical_kind_directory = self.cache_root / artifact_kind.value
            with ExitStack() as descriptor_stack:
                kind_descriptor = self._open_cache_kind_directory(
                    canonical_kind_directory,
                    artifact_kind,
                )
                descriptor_stack.callback(os.close, kind_descriptor)
                self._validate_open_cache_kind_directory(
                    kind_descriptor,
                    canonical_kind_directory,
                    artifact_kind,
                )
                os.replace(
                    entry.name,
                    tombstone.name,
                    src_dir_fd=kind_descriptor,
                    dst_dir_fd=kind_descriptor,
                )
                os.fsync(kind_descriptor)
                self._validate_open_cache_kind_directory(
                    kind_descriptor,
                    canonical_kind_directory,
                    artifact_kind,
                )
                anchored_tombstone = (
                    self._descriptor_path(kind_descriptor) / tombstone.name
                )
                self._delete_transient_directory(anchored_tombstone)
                os.fsync(kind_descriptor)
                self._validate_open_cache_kind_directory(
                    kind_descriptor,
                    canonical_kind_directory,
                    artifact_kind,
                )
            removed.append(receipt.artifact_id)
        return tuple(sorted(removed))

    def _validate_cache_ancestors(self) -> None:
        for path in reversed((self.cache_root, *self.cache_root.parents)):
            if path.is_symlink():
                raise CacheCorruptionError("cache path contains a symlinked ancestor")
            if path.exists() and not path.is_dir():
                raise CacheCorruptionError("cache path ancestor is not a directory")

    def _cache_lease(self) -> BinaryIO:
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._validate_cache_ancestors()
        lease_path = self.state_root / _CACHE_LEASE_NAME
        if lease_path.is_symlink():
            raise CacheCorruptionError("cache lease cannot be a symlink")
        descriptor = os.open(
            lease_path,
            os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise CacheCorruptionError("cache lease must be a private regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return os.fdopen(descriptor, "rb")

    def _ensure_cache_kind_directory(
        self, artifact_kind: PublicationArtifactKind
    ) -> Path:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._validate_cache_ancestors()
        kind_directory = self.cache_root / artifact_kind.value
        if kind_directory.is_symlink():
            raise CacheCorruptionError("cache kind directory cannot be a symlink")
        kind_directory.mkdir(exist_ok=True)
        self._validate_cache_kind_directory(kind_directory, artifact_kind)
        return kind_directory

    def _open_cache_kind_directory(
        self,
        kind_directory: Path,
        artifact_kind: PublicationArtifactKind,
    ) -> int:
        if kind_directory.is_symlink():
            raise CacheCorruptionError("cache kind directory cannot be a symlink")
        descriptor = os.open(
            kind_directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        return descriptor

    def _validate_open_cache_kind_directory(
        self,
        descriptor: int,
        kind_directory: Path,
        artifact_kind: PublicationArtifactKind,
    ) -> None:
        self._validate_cache_kind_directory(kind_directory, artifact_kind)
        opened = os.fstat(descriptor)
        observed = os.stat(kind_directory, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino)
        ):
            raise CacheCorruptionError("opened cache kind directory was replaced")

    def _descriptor_path(self, descriptor: int) -> Path:
        return Path("/proc/self/fd") / str(descriptor)

    def _validate_cache_kind_directory(
        self,
        kind_directory: Path,
        artifact_kind: PublicationArtifactKind,
    ) -> None:
        self._validate_cache_ancestors()
        expected = self.cache_root / artifact_kind.value
        if (
            kind_directory != expected
            or kind_directory.is_symlink()
            or not kind_directory.is_dir()
            or kind_directory.resolve() != kind_directory.absolute()
        ):
            raise CacheCorruptionError("cache kind directory is not contained")

    def _bounded_children(
        self,
        directory: Path,
        maximum_entries: int,
        description: str,
    ) -> tuple[Path, ...]:
        if maximum_entries < 0:
            raise CacheCorruptionError(f"{description} exceeds configured entry bound")
        if directory.is_symlink() or not directory.is_dir():
            raise CacheCorruptionError(f"{description} is not a safe directory")
        children: list[Path] = []
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if len(children) >= maximum_entries:
                    raise CacheCorruptionError(
                        f"{description} exceeds configured entry bound"
                    )
                children.append(Path(entry.path))
        return tuple(sorted(children))

    def _bounded_descriptor_children(
        self,
        descriptor: int,
        maximum_entries: int,
        description: str,
    ) -> tuple[Path, ...]:
        if maximum_entries < 0 or not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise CacheCorruptionError(f"{description} exceeds configured entry bound")
        children: list[Path] = []
        with os.scandir(descriptor) as iterator:
            for entry in iterator:
                if len(children) >= maximum_entries:
                    raise CacheCorruptionError(
                        f"{description} exceeds configured entry bound"
                    )
                children.append(self._descriptor_path(descriptor) / entry.name)
        return tuple(sorted(children))

    def _cache_kind_directories(self) -> tuple[Path, ...]:
        allowed = {kind.value for kind in PublicationArtifactKind}
        directories = self._bounded_children(
            self.cache_root,
            len(allowed),
            "cache root",
        )
        for directory in directories:
            if (
                directory.name not in allowed
                or directory.is_symlink()
                or not directory.is_dir()
            ):
                raise CacheCorruptionError("cache root contains an unexpected entry")
        return directories

    def _remove_abandoned_transient_entries(
        self,
        kind_directory: Path,
    ) -> None:
        artifact_kind = PublicationArtifactKind(kind_directory.name)
        with ExitStack() as descriptor_stack:
            kind_descriptor = self._open_cache_kind_directory(
                kind_directory,
                artifact_kind,
            )
            descriptor_stack.callback(os.close, kind_descriptor)
            self._validate_open_cache_kind_directory(
                kind_descriptor,
                kind_directory,
                artifact_kind,
            )
            while True:
                scanned_names: list[str] = []
                with os.scandir(kind_descriptor) as iterator:
                    for entry in iterator:
                        scanned_names.append(entry.name)
                        if len(scanned_names) > self.settings.cache_entry_limit:
                            break
                transient_names = sorted(
                    name
                    for name in scanned_names
                    if name.startswith(_TRANSIENT_PREFIXES)
                )
                if not transient_names:
                    if len(scanned_names) > self.settings.cache_entry_limit:
                        raise CacheCorruptionError(
                            "cache kind directory exceeds configured entry bound"
                        )
                    self._validate_open_cache_kind_directory(
                        kind_descriptor,
                        kind_directory,
                        artifact_kind,
                    )
                    break
                for transient_name in transient_names:
                    self._validate_open_cache_kind_directory(
                        kind_descriptor,
                        kind_directory,
                        artifact_kind,
                    )
                    anchored_entry = (
                        self._descriptor_path(kind_descriptor) / transient_name
                    )
                    self._delete_transient_directory(anchored_entry)
                    os.fsync(kind_descriptor)
                self._validate_open_cache_kind_directory(
                    kind_descriptor,
                    kind_directory,
                    artifact_kind,
                )

    def _delete_transient_directory(self, transient: Path) -> None:
        self._validate_removal_tree(transient)
        self._make_writable(transient)
        shutil.rmtree(transient)

    def _validate_removal_tree(self, root: Path) -> None:
        if root.is_symlink() or not root.is_dir():
            raise CacheCorruptionError("cache pruning target is not a safe directory")
        maximum_entries = (
            self.settings.archive_entry_limit
            + self.settings.release_asset_count_limit
            + 3
        )
        visited = 0
        pending = [root]
        while pending:
            directory = pending.pop()
            for path in self._bounded_children(
                directory,
                maximum_entries - visited,
                "cache pruning target",
            ):
                visited += 1
                if path.is_symlink():
                    raise CacheCorruptionError(
                        "cache pruning target contains a symlink"
                    )
                if path.is_dir():
                    pending.append(path)
                elif not path.is_file():
                    raise CacheCorruptionError(
                        "cache pruning target contains a special file"
                    )

    def _verify_cache_entry(
        self,
        target: Path,
        resolved: ResolvedGitHubArtifact,
        artifact_kind: PublicationArtifactKind,
    ) -> CacheVerificationReceipt:
        receipt = self._read_and_verify_receipt(target, artifact_kind)
        record = resolved.pointer.publication_record
        expected = {
            "artifact_kind": record.artifact_kind,
            "artifact_id": record.artifact_id,
            "materialized_tree_digest": resolved.pointer.materialized_tree_digest,
            "manifest_relative_path": resolved.pointer.manifest_relative_path,
            "manifest_digest": resolved.pointer.manifest_digest,
            "asset_digests": {asset.name: asset.sha256 for asset in record.assets},
        }
        for name, value in expected.items():
            if getattr(receipt, name) != value:
                raise CacheCorruptionError(f"cache receipt {name} mismatch")
        return receipt

    def _read_and_verify_receipt(
        self,
        target: Path,
        artifact_kind: PublicationArtifactKind,
    ) -> CacheVerificationReceipt:
        if not target.is_dir() or target.is_symlink():
            raise CacheCorruptionError("cache entry must be a directory")
        receipt_path = target / _RECEIPT_NAME
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise CacheCorruptionError("cache verification receipt is missing")
        receipt = CacheVerificationReceipt.from_json_bytes(
            self._read_control_file(
                receipt_path,
                "cache verification receipt",
                error_type=CacheCorruptionError,
            )
        )
        expected_entry_name = receipt.artifact_id.rsplit(":", 1)[1]
        if (
            receipt.artifact_kind is not artifact_kind
            or target.name != expected_entry_name
        ):
            raise CacheCorruptionError(
                "cache entry placement does not match its content identity"
            )
        self._validate_cache_bounds(target)
        if (
            self._tree_digest(target, ignore_root_receipt=True)
            != receipt.cache_tree_digest
        ):
            raise CacheCorruptionError("cache tree digest mismatch")
        self._verify_receipt_payloads(target, receipt)
        if target.stat().st_mode & 0o222:
            raise CacheCorruptionError("cache entry is not read-only")
        for path in target.rglob("*"):
            if path.stat().st_mode & 0o222:
                raise CacheCorruptionError("cache entry is not read-only")
        return receipt

    def _validate_cache_bounds(self, target: Path) -> None:
        expected_root_entries = {"assets", "content", _RECEIPT_NAME}
        root_entries = self._bounded_children(target, 4, "cache entry root")
        if {path.name for path in root_entries} != expected_root_entries:
            raise CacheCorruptionError("cache root closure is invalid")
        assets = target / "assets"
        content = target / "content"
        if (
            not assets.is_dir()
            or assets.is_symlink()
            or not content.is_dir()
            or content.is_symlink()
        ):
            raise CacheCorruptionError("cache directories are invalid")
        asset_entries = self._bounded_children(
            assets,
            self.settings.release_asset_count_limit,
            "cache assets",
        )
        asset_bytes = 0
        for path in asset_entries:
            if path.is_symlink() or not path.is_file():
                raise CacheCorruptionError("cache assets must be regular files")
            asset_bytes += path.stat().st_size
            if asset_bytes > self.settings.materialized_asset_size_bytes:
                raise CacheCorruptionError("cache assets exceed configured size bound")
        content_bytes = 0
        content_entries = 0
        pending = [content]
        while pending:
            directory = pending.pop()
            remaining = self.settings.archive_entry_limit - content_entries
            for path in self._bounded_children(
                directory,
                remaining,
                "cache content",
            ):
                content_entries += 1
                if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                    raise CacheCorruptionError(
                        "cache content must contain regular files and directories"
                    )
                if path.is_dir():
                    pending.append(path)
                else:
                    content_bytes += path.stat().st_size
                    if content_bytes > self.settings.materialized_asset_size_bytes:
                        raise CacheCorruptionError(
                            "cache content exceeds configured size bound"
                        )

    def _verify_receipt_payloads(
        self, target: Path, receipt: CacheVerificationReceipt
    ) -> None:
        assets = target / "assets"
        content = target / "content"
        if not assets.is_dir() or assets.is_symlink():
            raise CacheCorruptionError("cache asset directory is invalid")
        actual_assets: dict[str, Path] = {}
        for path in sorted(assets.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise CacheCorruptionError("cache assets must be regular files")
            actual_assets[path.name] = path
        if set(actual_assets) != set(receipt.asset_digests):
            raise CacheCorruptionError("cache asset closure mismatch")
        for name, expected_digest in receipt.asset_digests.items():
            if self._file_digest(actual_assets[name]) != expected_digest:
                raise CacheCorruptionError("cache asset digest mismatch")
        manifest_path = content / receipt.manifest_relative_path
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise CacheCorruptionError("cache manifest is missing")
        manifest_bytes = self._read_control_file(
            manifest_path,
            "cache manifest",
            error_type=CacheCorruptionError,
        )
        if tree_or_blob_digest(manifest_bytes) != receipt.manifest_digest:
            raise CacheCorruptionError("cache manifest digest mismatch")
        manifest = self._parse_manifest(receipt.artifact_kind, manifest_bytes)
        manifest_artifact_id = self._manifest_artifact_id(manifest)
        if manifest_artifact_id != receipt.artifact_id:
            raise CacheCorruptionError("cache manifest identity mismatch")
        declared_content_paths = self._verify_manifest_checksums(
            manifest.checksums,
            content,
            assets,
            receipt.manifest_relative_path,
        )
        self._verify_non_archive_asset_closure(
            manifest.checksums,
            receipt.asset_digests,
        )
        self._verify_content_closure(
            content, declared_content_paths, error_type=CacheCorruptionError
        )
        if (
            self._source_tree_digest(content, declared_content_paths)
            != receipt.materialized_tree_digest
        ):
            raise CacheCorruptionError(
                "cache content differs from verified package descriptor"
            )
        self._verify_reconstructed_content(actual_assets, content, target.parent)

    def _verify_reconstructed_content(
        self,
        actual_assets: Mapping[str, Path],
        content: Path,
        staging_parent: Path,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".staging-cache-verification-", dir=staging_parent
        ) as verification_name:
            self._verify_reconstructed_content_staging(
                actual_assets,
                content,
                Path(verification_name),
            )

    def _verify_reconstructed_content_staging(
        self,
        actual_assets: Mapping[str, Path],
        content: Path,
        verification_root: Path,
    ) -> None:
        reconstructed_content = verification_root / "content"
        reconstructed_content.mkdir()
        extracted_paths: dict[str, str] = {}
        extracted_bytes = 0
        extracted_entries = 0
        for asset_path in actual_assets.values():
            if not self._is_archive(asset_path.name):
                continue
            added_bytes, added_entries = self._extract_archive(
                asset_path,
                reconstructed_content,
                extracted_paths,
                self.settings.materialized_asset_size_bytes - extracted_bytes,
                self.settings.archive_entry_limit - extracted_entries,
            )
            extracted_bytes += added_bytes
            extracted_entries += added_entries
        if self._tree_digest(content) != self._tree_digest(reconstructed_content):
            raise CacheCorruptionError(
                "cache content does not reproduce from verified release assets"
            )

    def _is_archive(self, name: str) -> bool:
        return name.endswith((".tar", ".tar.zst"))

    def _extract_archive(
        self,
        archive: Path,
        destination: Path,
        extracted_paths: dict[str, str],
        maximum_bytes: int,
        maximum_entries: int,
    ) -> tuple[int, int]:
        return self.source_archive_extractor.extract(
            archive=archive,
            destination=destination,
            extracted_paths=extracted_paths,
            maximum_bytes=maximum_bytes,
            maximum_entries=maximum_entries,
        )

    def _file_digest(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            while True:
                chunk = file_handle.read(DEFAULT_BUFFER_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def _read_control_file(
        self,
        path: Path,
        description: str,
        *,
        error_type: type[MaterializationError] = MaterializationError,
    ) -> bytes:
        with path.open("rb") as file_handle:
            payload = file_handle.read(self.settings.control_blob_size_bytes + 1)
        if len(payload) > self.settings.control_blob_size_bytes:
            raise error_type(f"{description} exceeds configured control bound")
        return payload

    def _source_tree_digest(self, content: Path, paths: set[str]) -> str:
        files = {}
        for relative_path in sorted(paths):
            path = content / relative_path
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
            files[relative_path] = (
                self._file_digest(path),
                mode,
                path.stat().st_size,
            )
        return source_tree_digest(files)

    def _parse_manifest(
        self, artifact_kind: PublicationArtifactKind, payload: bytes
    ) -> (
        KnowledgeSnapshotManifest | ExpertBaseReleaseManifest | SecurityDenylistSnapshot
    ):
        if artifact_kind is PublicationArtifactKind.KNOWLEDGE_SNAPSHOT:
            return KnowledgeSnapshotManifest.from_json_bytes(payload)
        if artifact_kind is PublicationArtifactKind.EXPERT_BASE_RELEASE:
            return ExpertBaseReleaseManifest.from_json_bytes(payload)
        if artifact_kind is PublicationArtifactKind.SECURITY_DENYLIST:
            return SecurityDenylistSnapshot.from_json_bytes(payload)
        raise MaterializationError("materialized artifact kind is unsupported")

    @staticmethod
    def _manifest_artifact_id(
        manifest: (
            KnowledgeSnapshotManifest
            | ExpertBaseReleaseManifest
            | SecurityDenylistSnapshot
        ),
    ) -> str:
        if isinstance(manifest, ExpertBaseReleaseManifest):
            return manifest.release_id
        return manifest.snapshot_id

    def _validate_asset_bounds(self, assets: tuple[object, ...]) -> None:
        if not assets or len(assets) > self.settings.release_asset_count_limit:
            raise MaterializationError("release asset count exceeds configured limit")
        total_size = 0
        for asset in assets:
            size = getattr(asset, "size")
            if type(size) is not int or size < 1:
                raise MaterializationError("release asset size is invalid")
            if size > self.settings.release_asset_size_bytes:
                raise MaterializationError(
                    "release asset exceeds configured size limit"
                )
            total_size += size
            if total_size > self.settings.materialized_asset_size_bytes:
                raise MaterializationError(
                    "release asset closure exceeds configured size limit"
                )

    def _verify_manifest_checksums(
        self,
        checksums: Mapping[str, str],
        content: Path,
        assets: Path,
        manifest_relative_path: str,
    ) -> set[str]:
        content_paths = {manifest_relative_path}
        for relative_path, expected_digest in checksums.items():
            if relative_path == manifest_relative_path:
                continue
            content_path = content / relative_path
            asset_path = assets / relative_path
            if content_path.is_file() and not content_path.is_symlink():
                path = content_path
                content_paths.add(relative_path)
            elif asset_path.is_file() and not asset_path.is_symlink():
                path = asset_path
            else:
                raise MaterializationError(
                    f"manifest checksum path is missing: {relative_path}"
                )
            if self._file_digest(path) != expected_digest:
                raise MaterializationError(
                    f"manifest checksum mismatch: {relative_path}"
                )
        return content_paths

    def _verify_non_archive_asset_closure(
        self,
        checksums: Mapping[str, str],
        asset_digests: Mapping[str, str],
    ) -> None:
        for name, digest in asset_digests.items():
            if not self._is_archive(name) and checksums.get(name) != digest:
                raise MaterializationError(
                    f"non-archive release asset is outside manifest closure: {name}"
                )

    def _verify_content_closure(
        self,
        content: Path,
        declared_paths: set[str],
        *,
        error_type: type[MaterializationError] = MaterializationError,
    ) -> None:
        actual_files = {
            path.relative_to(content).as_posix()
            for path in content.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual_files != declared_paths:
            raise error_type("release package content is not closed")
        expected_directories: set[str] = set()
        for relative_path in declared_paths:
            path = PurePosixPath(relative_path)
            for parent in path.parents:
                if parent.as_posix() != ".":
                    expected_directories.add(parent.as_posix())
        actual_directories = {
            path.relative_to(content).as_posix()
            for path in content.rglob("*")
            if path.is_dir() and not path.is_symlink()
        }
        if actual_directories != expected_directories:
            raise error_type("release package directory closure is not exact")

    def _tree_digest(self, root: Path, *, ignore_root_receipt: bool = False) -> str:
        descriptor = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise CacheCorruptionError("cache tree contains a symlink")
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                descriptor.append({"kind": "directory", "path": relative})
                continue
            if not path.is_file():
                raise CacheCorruptionError("cache tree contains a special file")
            if ignore_root_receipt and path == root / _RECEIPT_NAME:
                continue
            descriptor.append(
                {
                    "digest": self._file_digest(path),
                    "executable": bool(path.stat().st_mode & 0o111),
                    "kind": "file",
                    "path": relative,
                    "size": path.stat().st_size,
                }
            )
        return tree_or_blob_digest(canonical_json_bytes(tuple(descriptor)))

    def _flush_tree(self, root: Path) -> None:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                with path.open("rb") as file_handle:
                    os.fsync(file_handle.fileno())
        for path in sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_dir()),
            key=lambda candidate: len(candidate.parts),
            reverse=True,
        ):
            self._flush_directory(path)
        self._flush_directory(root)

    def _flush_directory(self, path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
        os.close(descriptor)

    def _make_read_only(self, root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
            else:
                path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        root.chmod(0o555)

    def _make_writable(self, root: Path) -> None:
        root.chmod(0o755)
        for path in sorted(root.rglob("*")):
            path.chmod(0o755 if path.is_dir() else 0o644)
