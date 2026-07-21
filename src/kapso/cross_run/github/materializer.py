"""Verified, content-addressed materialization of immutable release assets."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import stat
import tarfile
import tempfile
import ctypes
from contextlib import ExitStack
from dataclasses import dataclass
from io import DEFAULT_BUFFER_SIZE
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Mapping, Protocol

import zstandard

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
    SourceFileDescriptor,
    StrictContract,
)
from kapso.cross_run.github.command import GitHubCommandClient
from kapso.cross_run.github.resolver import ResolvedGitHubArtifact
from kapso.cross_run.settings import GitHubSettings

_RECEIPT_NAME = "VERIFIED.json"
_CACHE_LEASE_NAME = ".github-artifact-cache.lock"
_TRANSIENT_PREFIXES = (".staging-", ".pruning-", ".validation-")
_TAR_BLOCK_SIZE = 512
_TAR_ZERO_BLOCK = b"\0" * _TAR_BLOCK_SIZE
_CANONICAL_TAR_TYPES = {b"\0", b"0", b"5"}
_ZSTD_MAX_BLOCK_SIZE = 128 * 1024
_ZSTD_MAX_FRAME_HEADER_SIZE = 18
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
        packaged_artifact_id = (
            manifest.snapshot_id
            if isinstance(manifest, KnowledgeSnapshotManifest)
            else manifest.release_id
        )
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

    def extract_verified_source_archive(
        self,
        *,
        materialized: MaterializedArtifact,
        expected: SourceArchiveExtractionReceipt,
        destination: Path,
    ) -> SourceArchiveExtractionReceipt:
        """Recreate one attested source tree in a fresh private destination."""

        destination_parent_identity = self._validate_source_destination(destination)
        with self._cache_lease():
            if (
                self._validate_source_destination(destination)
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
                dir=destination.parent,
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
    ) -> SourceArchiveExtractionReceipt:
        self._extract_archive(
            source_archive,
            destination,
            {},
            self.settings.materialized_asset_size_bytes,
            self.settings.archive_entry_limit,
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
        source_paths = tuple(sorted(source_tree.rglob("*")))
        relative_entries = {
            path: path.relative_to(source_tree).as_posix() for path in source_paths
        }
        for path, relative_path in relative_entries.items():
            self._safe_archive_path(relative_path, path.is_dir())
        if any(
            path.is_symlink() or (not path.is_file() and not path.is_dir())
            for path in source_paths
        ):
            raise MaterializationError(
                "extracted source tree contains an invalid entry"
            )
        if any(
            path.stat(follow_symlinks=False).st_nlink != 1
            for path in source_paths
            if path.is_file()
        ):
            raise MaterializationError(
                "extracted source tree files must be independent"
            )
        source_files = tuple(
            SourceFileDescriptor(
                relative_path=relative_entries[path],
                digest=self._file_digest(path),
                mode="100755" if path.stat().st_mode & 0o111 else "100644",
                size=path.stat().st_size,
            )
            for path in source_paths
            if path.is_file()
        )
        if not source_files:
            raise MaterializationError("source archive tree is empty")
        observed_directories = {
            relative_entries[path] for path in source_paths if path.is_dir()
        }
        implied_directories = {
            parent.as_posix()
            for source_file in source_files
            for parent in PurePosixPath(source_file.relative_path).parents
            if parent != PurePosixPath(".")
        }
        if observed_directories != implied_directories:
            raise MaterializationError(
                "source tree contains undeclared empty directories"
            )
        return source_files

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

    def _validate_source_destination(self, destination: Path) -> tuple[int, int]:
        if (
            not destination.is_absolute()
            or destination != Path(os.path.abspath(destination))
            or os.path.lexists(destination)
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
        if metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID):
            raise MaterializationError("source extraction parent must be private")
        return metadata.st_dev, metadata.st_ino

    def _publish_source_tree(
        self,
        staged_source: Path,
        destination: Path,
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
            parent_descriptor = os.open(
                destination.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptors.callback(os.close, parent_descriptor)
            source_metadata = os.fstat(source_descriptor)
            named_source_metadata = os.stat(
                staged_source.name,
                dir_fd=staging_parent_descriptor,
                follow_symlinks=False,
            )
            parent_metadata = os.fstat(parent_descriptor)
            if (
                not stat.S_ISDIR(source_metadata.st_mode)
                or (source_metadata.st_dev, source_metadata.st_ino)
                != (named_source_metadata.st_dev, named_source_metadata.st_ino)
                or source_metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
                or os.path.lexists(destination)
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
                parent_descriptor,
                os.fsencode(destination.name),
                _RENAME_NOREPLACE,
            )
            if result != 0:
                error_number = ctypes.get_errno()
                raise MaterializationError(
                    "atomic source-tree publication failed: "
                    f"{os.strerror(error_number)}"
                )
            os.fsync(parent_descriptor)

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
        manifest_artifact_id = (
            manifest.snapshot_id
            if isinstance(manifest, KnowledgeSnapshotManifest)
            else manifest.release_id
        )
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
        manifest_artifact_id = (
            manifest.snapshot_id
            if isinstance(manifest, KnowledgeSnapshotManifest)
            else manifest.release_id
        )
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
        tar_path = archive
        if archive.name.endswith(".tar.zst"):
            expected_content_size = self._validate_single_zstd_frame(
                archive,
                maximum_bytes,
            )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".decompressed-", suffix=".tar", dir=destination.parent
            )
            os.close(descriptor)
            tar_path = Path(temporary_name)
            with archive.open("rb") as compressed, tar_path.open("wb") as decompressed:
                decompressor = zstandard.ZstdDecompressor(
                    max_window_size=self.settings.zstd_window_size_bytes
                )
                with decompressor.stream_reader(compressed) as reader:
                    decompressed_bytes = 0
                    while True:
                        chunk = reader.read(DEFAULT_BUFFER_SIZE)
                        if not chunk:
                            break
                        decompressed_bytes += len(chunk)
                        if decompressed_bytes > maximum_bytes:
                            raise MaterializationError(
                                "decompressed archive exceeds configured size limit"
                            )
                        decompressed.write(chunk)
            if (
                expected_content_size is not None
                and decompressed_bytes != expected_content_size
            ):
                raise MaterializationError(
                    "decompressed archive size differs from its zstd frame"
                )
        result = self._extract_tar(
            tar_path,
            destination,
            extracted_paths,
            maximum_bytes,
            maximum_entries,
        )
        if tar_path != archive:
            tar_path.unlink()
        return result

    def _extract_tar(
        self,
        archive: Path,
        destination: Path,
        extracted_paths: dict[str, str],
        maximum_bytes: int,
        maximum_entries: int,
    ) -> tuple[int, int]:
        physical_members = self._validate_canonical_tar(
            archive,
            maximum_entries,
        )
        total_bytes = 0
        implicit_directories = 0
        with tarfile.open(archive, mode="r:") as package:
            for member in package:
                relative = self._safe_archive_path(member.name, member.isdir())
                if relative is None:
                    if not member.isdir():
                        raise MaterializationError(
                            "tar contains a hidden regular-file member"
                        )
                    continue
                if not member.isdir() and not member.isfile():
                    raise MaterializationError(
                        "tar links and special files are forbidden"
                    )
                implicit_directories += self._reserve_archive_path(
                    relative,
                    member.isdir(),
                    extracted_paths,
                )
                if physical_members + implicit_directories > maximum_entries:
                    raise MaterializationError("archive exceeds configured entry limit")
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total_bytes += member.size
                if total_bytes > maximum_bytes:
                    raise MaterializationError("archive exceeds configured size limit")
                source = package.extractfile(member)
                if source is None:
                    raise MaterializationError("tar file entry has no content")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
        return total_bytes, physical_members + implicit_directories

    def _validate_canonical_tar(self, archive: Path, maximum_entries: int) -> int:
        archive_size = archive.stat().st_size
        if archive_size % _TAR_BLOCK_SIZE != 0:
            raise MaterializationError("tar archive has an incomplete block")
        physical_members = 0
        terminated = False
        with archive.open("rb") as file_handle:
            while file_handle.tell() < archive_size:
                header = file_handle.read(_TAR_BLOCK_SIZE)
                if header == _TAR_ZERO_BLOCK:
                    second_zero = file_handle.read(_TAR_BLOCK_SIZE)
                    if second_zero != _TAR_ZERO_BLOCK:
                        raise MaterializationError(
                            "tar archive has an invalid end marker"
                        )
                    while file_handle.tell() < archive_size:
                        padding = file_handle.read(DEFAULT_BUFFER_SIZE)
                        if padding.strip(b"\0"):
                            raise MaterializationError(
                                "tar archive has data after its end marker"
                            )
                    terminated = True
                    break
                physical_members += 1
                if physical_members > maximum_entries:
                    raise MaterializationError("archive exceeds configured entry limit")
                member_type = header[156:157]
                if member_type not in _CANONICAL_TAR_TYPES:
                    raise MaterializationError(
                        "tar extension headers and special files are forbidden"
                    )
                raw_size = header[124:136].strip(b"\0 ")
                if raw_size and re.fullmatch(rb"[0-7]+", raw_size) is None:
                    raise MaterializationError("tar member size is not canonical octal")
                member_size = int(raw_size, 8) if raw_size else 0
                if member_type == b"5" and member_size != 0:
                    raise MaterializationError("tar directory contains payload bytes")
                padded_size = (
                    (member_size + _TAR_BLOCK_SIZE - 1) // _TAR_BLOCK_SIZE
                ) * _TAR_BLOCK_SIZE
                next_header = file_handle.tell() + padded_size
                if next_header > archive_size:
                    raise MaterializationError("tar member exceeds archive bounds")
                file_handle.seek(padded_size, os.SEEK_CUR)
        if not terminated:
            raise MaterializationError("tar archive has no canonical end marker")
        return physical_members

    def _validate_single_zstd_frame(
        self,
        archive: Path,
        maximum_bytes: int,
    ) -> int | None:
        archive_size = archive.stat().st_size
        with archive.open("rb") as file_handle:
            frame_header = file_handle.read(_ZSTD_MAX_FRAME_HEADER_SIZE)
            header_size = zstandard.frame_header_size(frame_header)
            parameters = zstandard.get_frame_parameters(frame_header)
            if parameters.window_size > self.settings.zstd_window_size_bytes:
                raise MaterializationError(
                    "zstd frame window exceeds configured size limit"
                )
            expected_content_size = (
                None
                if parameters.content_size == zstandard.CONTENTSIZE_UNKNOWN
                else parameters.content_size
            )
            if (
                expected_content_size is not None
                and expected_content_size > maximum_bytes
            ):
                raise MaterializationError(
                    "decompressed archive exceeds configured size limit"
                )
            file_handle.seek(header_size)
            last_block = False
            while not last_block:
                block_header = file_handle.read(3)
                if len(block_header) != 3:
                    raise MaterializationError("zstd frame has an incomplete block")
                descriptor = int.from_bytes(block_header, "little")
                last_block = bool(descriptor & 1)
                block_type = (descriptor >> 1) & 3
                block_size = descriptor >> 3
                if block_type == 3 or block_size > _ZSTD_MAX_BLOCK_SIZE:
                    raise MaterializationError("zstd frame block is invalid")
                stored_size = 1 if block_type == 1 else block_size
                next_block = file_handle.tell() + stored_size
                if next_block > archive_size:
                    raise MaterializationError(
                        "zstd frame block exceeds archive bounds"
                    )
                file_handle.seek(stored_size, os.SEEK_CUR)
            if parameters.has_checksum:
                checksum = file_handle.read(4)
                if len(checksum) != 4:
                    raise MaterializationError("zstd frame checksum is incomplete")
            if file_handle.tell() != archive_size:
                raise MaterializationError(
                    "zstd archive must contain exactly one canonical frame"
                )
        return expected_content_size

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

    def _safe_archive_path(self, name: str, is_directory: bool) -> str | None:
        normalized_name = name[:-1] if is_directory and name.endswith("/") else name
        if not normalized_name or normalized_name == ".":
            return None
        path = PurePosixPath(normalized_name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in normalized_name
            or path.as_posix() != normalized_name
            or ".git" in path.parts
            or normalized_name == ".gitmodules"
        ):
            raise MaterializationError("archive contains an unsafe path")
        return normalized_name

    def _reserve_archive_path(
        self,
        path: str,
        is_directory: bool,
        extracted_paths: dict[str, str],
    ) -> int:
        parts = PurePosixPath(path).parts
        implicit_directories = 0
        for depth in range(1, len(parts)):
            parent = PurePosixPath(*parts[:depth]).as_posix()
            state = extracted_paths.get(parent)
            if state == "file":
                raise MaterializationError(
                    "archive path descends through a regular file"
                )
            if state is None:
                extracted_paths[parent] = "implicit-directory"
                implicit_directories += 1
        state = extracted_paths.get(path)
        if is_directory and state == "implicit-directory":
            extracted_paths[path] = "directory"
            return implicit_directories
        if state is not None:
            raise MaterializationError("archive packages contain a duplicate path")
        extracted_paths[path] = "directory" if is_directory else "file"
        return implicit_directories

    def _parse_manifest(
        self, artifact_kind: PublicationArtifactKind, payload: bytes
    ) -> KnowledgeSnapshotManifest | ExpertBaseReleaseManifest:
        if artifact_kind is PublicationArtifactKind.KNOWLEDGE_SNAPSHOT:
            return KnowledgeSnapshotManifest.from_json_bytes(payload)
        return ExpertBaseReleaseManifest.from_json_bytes(payload)

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
