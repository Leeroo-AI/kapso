"""Immutable receipt-keyed task-adapter packages and authorized activation."""

from __future__ import annotations

import fcntl
import io
import os
import shutil
import stat
import tempfile
import time
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from typing import Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.github.materializer import (
    SOURCE_ARCHIVE_EXTRACTOR_VERSION,
    SourceArchiveExtractionReceipt,
)
from kapso.cross_run.contracts import TaskAdapterManifest
from kapso.cross_run.settings import TaskAdapterStoreSettings
from kapso.cross_run.source_archives import SourceArchiveExtractor
from kapso.cross_run.task_adapters import (
    TaskAdapterActivationRecord,
    TaskAdapterAuthority,
    TaskAdapterPackage,
    TaskAdapterVerificationReceipt,
    VerifiedTaskAdapter,
    task_adapter_materialization_usage,
)

_MANIFEST_NAME = "manifest.json"
_EXTRACTION_RECEIPT_NAME = "source-extraction.json"
_COMMIT_NAME = "COMMITTED.json"
_PUBLISHER_VERIFICATION_NAME = "publisher-verification.bin"
_LOCK_NAME = "task-adapters.lock"


class TaskAdapterStoreError(RuntimeError):
    """A package, activation, or store invariant failed."""


class TaskAdapterActivationConflict(TaskAdapterStoreError):
    """The expected active adapter generation changed."""


class TaskAdapterAuthorityRegistry:
    """Resolve configured active and historical verifier implementations."""

    def __init__(
        self,
        settings: TaskAdapterStoreSettings,
        authorities: tuple[TaskAdapterAuthority, ...],
    ) -> None:
        authority_map = {
            (authority.authority_id, authority.authority_version): authority
            for authority in authorities
        }
        configured_identities = {
            authority.identity for authority in settings.trusted_authorities
        }
        if len(authority_map) != len(authorities) or set(authority_map) != (
            configured_identities
        ):
            raise TaskAdapterStoreError(
                "task adapter authority implementations differ from configured trust"
            )
        self.settings = settings
        self.authorities = authority_map

    def active(self) -> TaskAdapterAuthority:
        return self.resolve(*self.settings.active_authority.identity)

    def resolve(
        self,
        authority_id: str,
        authority_version: str,
    ) -> TaskAdapterAuthority:
        authority = self.authorities.get((authority_id, authority_version))
        if authority is None:
            raise TaskAdapterStoreError(
                "task adapter authority version is untrusted or revoked"
            )
        return authority


class TaskAdapterPackageStore:
    """Publish and resolve complete adapter packages under one trust authority."""

    def __init__(
        self,
        state_path: Path,
        state_root: Path,
        settings: TaskAdapterStoreSettings,
        authority_registry: TaskAdapterAuthorityRegistry,
    ) -> None:
        self.state_path = state_path
        self.state_root = state_root
        self.settings = settings
        self.authority_registry = authority_registry
        self.extractor = SourceArchiveExtractor(
            zstd_window_size_bytes=settings.zstd_window_size_bytes,
            error_type=TaskAdapterStoreError,
        )
        self._validate_locations()
        if authority_registry.settings != settings:
            raise TaskAdapterStoreError(
                "task adapter authority registry differs from store configuration"
            )

    def publish(self, package: TaskAdapterPackage) -> VerifiedTaskAdapter:
        self._validate_package_bounds(package)
        with self._lease():
            self._prepare_locked()
            with tempfile.TemporaryDirectory(
                prefix=".staging-package-",
                dir=self.state_path / "packages",
            ) as staging_name:
                staged = self._stage_package(Path(staging_name), package)
                target = self._package_path(
                    staged.verification_receipt.verification_receipt_id
                )
                if target.exists() or target.is_symlink():
                    existing = self._read_package(target)
                    if existing != staged:
                        raise TaskAdapterStoreError(
                            "task adapter receipt collides with different package bytes"
                        )
                    return existing
                self._flush_tree(Path(staging_name))
                self._make_read_only(Path(staging_name))
                os.rename(Path(staging_name), target)
                self._fsync_directory(target.parent)
                return self._read_package(target)

    def read(self, verification_receipt_id: str) -> VerifiedTaskAdapter:
        require_content_id(verification_receipt_id, "verification_receipt_id")
        with self._lease():
            self._prepare_locked()
            return self._read_package(self._package_path(verification_receipt_id))

    def resolve_exact(
        self,
        *,
        task_adapter_manifest_id: str,
        verification_receipt_id: str,
    ) -> VerifiedTaskAdapter:
        require_content_id(task_adapter_manifest_id, "task_adapter_manifest_id")
        package = self.read(verification_receipt_id)
        if package.manifest.task_adapter_manifest_id != task_adapter_manifest_id:
            raise TaskAdapterStoreError(
                "task adapter exact pin names another scientific manifest"
            )
        return package

    def resolve_exact_bounded(
        self,
        *,
        task_adapter_manifest_id: str,
        verification_receipt_id: str,
        maximum_entries: int,
        maximum_bytes: int,
        timeout_seconds: int,
    ) -> VerifiedTaskAdapter:
        require_content_id(task_adapter_manifest_id, "task_adapter_manifest_id")
        require_content_id(verification_receipt_id, "verification_receipt_id")
        if (
            type(maximum_entries) is not int
            or maximum_entries <= 0
            or type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or type(timeout_seconds) is not int
            or timeout_seconds <= 0
        ):
            raise TaskAdapterStoreError(
                "bounded task adapter limits must be positive integers"
            )
        deadline = time.monotonic() + timeout_seconds
        with self._lease():
            self._require_materialization_deadline(deadline)
            self._prepare_locked()
            self._require_materialization_deadline(deadline)
            package_path = self._package_path(verification_receipt_id)
            self._validate_materialization_bounds(
                package_path,
                maximum_entries,
                maximum_bytes,
                deadline,
            )
            package = self._read_package(package_path, deadline=deadline)
            self._require_materialization_deadline(deadline)
        if package.manifest.task_adapter_manifest_id != task_adapter_manifest_id:
            raise TaskAdapterStoreError(
                "task adapter exact pin names another scientific manifest"
            )
        return package

    def activate(
        self,
        *,
        scope_contract_id: str,
        task_family_id: str,
        task_adapter_id: str,
        verification_receipt_id: str,
        expected_activation_id: str | None,
        authority_envelope: bytes,
    ) -> TaskAdapterActivationRecord:
        require_content_id(scope_contract_id, "scope_contract_id")
        require_content_id(verification_receipt_id, "verification_receipt_id")
        if expected_activation_id is not None:
            require_content_id(expected_activation_id, "expected_activation_id")
        if not isinstance(authority_envelope, bytes) or not authority_envelope:
            raise TaskAdapterStoreError("task adapter authority envelope is empty")
        with self._lease():
            self._prepare_locked()
            package = self._read_package(self._package_path(verification_receipt_id))
            active_authority = self.authority_registry.active()
            manifest = package.manifest
            if (
                manifest.scope_contract_id != scope_contract_id
                or manifest.task_family_id != task_family_id
                or manifest.task_adapter_id != task_adapter_id
            ):
                raise TaskAdapterStoreError(
                    "task adapter activation differs from its package binding"
                )
            pointer_path = self._active_pointer_path(
                scope_contract_id,
                task_family_id,
                task_adapter_id,
            )
            current = self._read_active_pointer(pointer_path)
            current_id = None if current is None else current.activation_id
            if current is not None:
                observed_current, current_envelope = self._read_activation(current_id)
                if observed_current != current:
                    raise TaskAdapterStoreError(
                        "task adapter active pointer differs from its activation"
                    )
                if (
                    current.scope_contract_id != scope_contract_id
                    or current.task_family_id != task_family_id
                    or current.task_adapter_id != task_adapter_id
                ):
                    raise TaskAdapterStoreError(
                        "task adapter active pointer names another logical binding"
                    )
                current_authority = self.authority_registry.resolve(
                    current.authority_id,
                    current.authority_version,
                )
                current_authority.verify_activation(
                    activation=current,
                    authority_envelope=current_envelope,
                )
            if current_id != expected_activation_id:
                if (
                    current is not None
                    and current.predecessor_activation_id == expected_activation_id
                    and current.scope_contract_id == scope_contract_id
                    and current.task_family_id == task_family_id
                    and current.task_adapter_id == task_adapter_id
                    and current.verification_receipt_id == verification_receipt_id
                    and current.authority_envelope_digest
                    == tree_or_blob_digest(authority_envelope)
                ):
                    return current
                raise TaskAdapterActivationConflict(
                    "task adapter active generation changed"
                )
            activation = TaskAdapterActivationRecord.mint(
                scope_contract_id=scope_contract_id,
                task_family_id=task_family_id,
                task_adapter_id=task_adapter_id,
                task_adapter_manifest_id=manifest.task_adapter_manifest_id,
                verification_receipt_id=verification_receipt_id,
                predecessor_activation_id=current_id,
                authority_id=active_authority.authority_id,
                authority_version=active_authority.authority_version,
                authority_envelope_digest=tree_or_blob_digest(authority_envelope),
            )
            active_authority.verify_activation(
                activation=activation,
                authority_envelope=authority_envelope,
            )
            self._persist_activation(activation, authority_envelope)
            self._write_active_pointer(pointer_path, activation)
            return activation

    def resolve_active(
        self,
        *,
        scope_contract_id: str,
        task_family_id: str,
        task_adapter_id: str,
    ) -> VerifiedTaskAdapter:
        require_content_id(scope_contract_id, "scope_contract_id")
        with self._lease():
            self._prepare_locked()
            pointer = self._read_active_pointer(
                self._active_pointer_path(
                    scope_contract_id,
                    task_family_id,
                    task_adapter_id,
                )
            )
            if pointer is None:
                raise TaskAdapterStoreError("task adapter binding is not active")
            activation, envelope = self._read_activation(pointer.activation_id)
            if activation != pointer:
                raise TaskAdapterStoreError(
                    "active task adapter pointer differs from immutable activation"
                )
            if (
                activation.scope_contract_id != scope_contract_id
                or activation.task_family_id != task_family_id
                or activation.task_adapter_id != task_adapter_id
            ):
                raise TaskAdapterStoreError(
                    "active task adapter pointer names another logical binding"
                )
            activation_authority = self.authority_registry.resolve(
                activation.authority_id,
                activation.authority_version,
            )
            activation_authority.verify_activation(
                activation=activation,
                authority_envelope=envelope,
            )
            package = self._read_package(
                self._package_path(activation.verification_receipt_id)
            )
            if (
                package.manifest.scope_contract_id != scope_contract_id
                or package.manifest.task_family_id != task_family_id
                or package.manifest.task_adapter_id != task_adapter_id
                or package.manifest.task_adapter_manifest_id
                != activation.task_adapter_manifest_id
            ):
                raise TaskAdapterStoreError(
                    "active task adapter package differs from its activation"
                )
            return package

    def _stage_package(
        self,
        staging: Path,
        package: TaskAdapterPackage,
    ) -> VerifiedTaskAdapter:
        manifest = package.manifest
        expected_proof_refs = {
            manifest.sanitation_report_id,
            *manifest.validation_refs,
        }
        if set(package.proof_objects) != expected_proof_refs:
            raise TaskAdapterStoreError(
                "task adapter package proof closure differs from its manifest"
            )
        source_archive_path = PurePosixPath(manifest.source_tree_ref)
        if (
            source_archive_path.is_absolute()
            or len(source_archive_path.parts) != 1
            or source_archive_path.as_posix() != manifest.source_tree_ref
            or not manifest.source_tree_ref.endswith((".tar", ".tar.zst"))
        ):
            raise TaskAdapterStoreError(
                "task adapter source reference must name one tar archive"
            )
        archive_directory = staging / "archive"
        source_directory = staging / "source"
        proof_directory = staging / "proofs"
        archive_directory.mkdir()
        source_directory.mkdir()
        proof_directory.mkdir()
        archive_path = archive_directory / manifest.source_tree_ref
        archive_path.write_bytes(package.source_archive)
        self.extractor.extract(
            archive=archive_path,
            destination=source_directory,
            extracted_paths={},
            maximum_bytes=self.settings.source_byte_limit,
            maximum_entries=self.settings.package_entry_limit,
        )
        source_files = self.extractor.source_tree_files(source_directory)
        observed_tree_hash = self.extractor.tree_hash(source_files)
        if observed_tree_hash != manifest.tree_hash:
            raise TaskAdapterStoreError(
                "task adapter source tree differs from its manifest"
            )
        source_archive_digest = tree_or_blob_digest(package.source_archive)
        extraction_receipt = SourceArchiveExtractionReceipt.mint(
            artifact_id=manifest.task_adapter_manifest_id,
            source_archive_ref=manifest.source_tree_ref,
            source_archive_digest=source_archive_digest,
            source_tree_hash=observed_tree_hash,
            source_tree_files=source_files,
            extractor_version=SOURCE_ARCHIVE_EXTRACTOR_VERSION,
        )
        active_authority = self.authority_registry.active()
        active_authority.verify_package(
            manifest=manifest,
            source_extraction_receipt=extraction_receipt,
            proof_objects=package.proof_objects,
            publisher_verification=package.publisher_verification,
        )
        proof_digests = {
            proof_ref: tree_or_blob_digest(payload)
            for proof_ref, payload in package.proof_objects.items()
        }
        receipt = TaskAdapterVerificationReceipt.mint(
            task_adapter_manifest_id=manifest.task_adapter_manifest_id,
            full_manifest_digest=tree_or_blob_digest(manifest.to_json_bytes()),
            publisher_attestation_digest=tree_or_blob_digest(
                canonical_json_bytes(manifest.publisher_attestation)
            ),
            source_extraction_receipt_id=extraction_receipt.extraction_receipt_id,
            source_archive_ref=manifest.source_tree_ref,
            source_archive_digest=source_archive_digest,
            source_tree_hash=observed_tree_hash,
            proof_object_digests=proof_digests,
            publisher_verification_digest=tree_or_blob_digest(
                package.publisher_verification
            ),
            verifier_id=active_authority.authority_id,
            verifier_version=active_authority.authority_version,
        )
        (staging / _MANIFEST_NAME).write_bytes(manifest.to_json_bytes())
        (staging / _EXTRACTION_RECEIPT_NAME).write_bytes(
            extraction_receipt.to_json_bytes()
        )
        (staging / _PUBLISHER_VERIFICATION_NAME).write_bytes(
            package.publisher_verification
        )
        for digest, payload in {
            proof_digests[proof_ref]: package.proof_objects[proof_ref]
            for proof_ref in expected_proof_refs
        }.items():
            (proof_directory / f"{digest[7:]}.bin").write_bytes(payload)
        (staging / _COMMIT_NAME).write_bytes(receipt.to_json_bytes())
        self._validate_stored_bounds(staging)
        return self._verified_from_parts(
            manifest=manifest,
            receipt=receipt,
            extraction_receipt=extraction_receipt,
            source_archive=package.source_archive,
            source_directory=source_directory,
            proof_objects=package.proof_objects,
            publisher_verification=package.publisher_verification,
        )

    def _read_package(
        self,
        package_path: Path,
        *,
        deadline: float | None = None,
    ) -> VerifiedTaskAdapter:
        self._require_materialization_deadline(deadline)
        if package_path.is_symlink() or not package_path.is_dir():
            raise TaskAdapterStoreError("task adapter package is missing or unsafe")
        self._validate_committed_permissions(package_path)
        self._validate_stored_bounds(package_path)
        self._require_exact_children(
            package_path,
            {
                _MANIFEST_NAME,
                _EXTRACTION_RECEIPT_NAME,
                _COMMIT_NAME,
                _PUBLISHER_VERIFICATION_NAME,
                "archive",
                "proofs",
                "source",
            },
        )
        manifest = self._read_manifest(
            package_path / _MANIFEST_NAME,
            deadline=deadline,
        )
        receipt = TaskAdapterVerificationReceipt.from_json_bytes(
            self._read_bounded(package_path / _COMMIT_NAME, deadline=deadline)
        )
        extraction_receipt = SourceArchiveExtractionReceipt.from_json_bytes(
            self._read_bounded(
                package_path / _EXTRACTION_RECEIPT_NAME,
                deadline=deadline,
            )
        )
        if self._package_path(receipt.verification_receipt_id) != package_path:
            raise TaskAdapterStoreError(
                "task adapter package receipt location is invalid"
            )
        package_authority = self.authority_registry.resolve(
            receipt.verifier_id,
            receipt.verifier_version,
        )
        archive_directory = package_path / "archive"
        self._require_exact_children(archive_directory, {manifest.source_tree_ref})
        source_archive = self._read_bounded(
            archive_directory / manifest.source_tree_ref,
            byte_limit=self.settings.package_byte_limit,
            deadline=deadline,
        )
        proof_directory = package_path / "proofs"
        expected_proof_names = {
            f"{digest[7:]}.bin" for digest in receipt.proof_object_digests.values()
        }
        self._require_exact_children(proof_directory, expected_proof_names)
        proof_objects = {
            proof_ref: self._read_bounded(
                proof_directory / f"{digest[7:]}.bin",
                deadline=deadline,
            )
            for proof_ref, digest in receipt.proof_object_digests.items()
        }
        publisher_verification = self._read_bounded(
            package_path / _PUBLISHER_VERIFICATION_NAME,
            deadline=deadline,
        )
        source_directory = package_path / "source"
        self._require_materialization_deadline(deadline)
        observed_source_files = self.extractor.source_tree_files(source_directory)
        self._require_materialization_deadline(deadline)
        if observed_source_files != extraction_receipt.source_tree_files:
            raise TaskAdapterStoreError(
                "stored task adapter source differs from extraction receipt"
            )
        with tempfile.TemporaryDirectory(
            prefix=".validation-source-",
            dir=self.state_path / "staging",
        ) as validation_name:
            validation_source = Path(validation_name) / "source"
            validation_source.mkdir()
            validation_archive = Path(validation_name) / manifest.source_tree_ref
            validation_archive.write_bytes(source_archive)
            self._require_materialization_deadline(deadline)
            self.extractor.extract(
                archive=validation_archive,
                destination=validation_source,
                extracted_paths={},
                maximum_bytes=self.settings.source_byte_limit,
                maximum_entries=self.settings.package_entry_limit,
            )
            self._require_materialization_deadline(deadline)
            if (
                self.extractor.source_tree_files(validation_source)
                != extraction_receipt.source_tree_files
            ):
                raise TaskAdapterStoreError(
                    "task adapter archive no longer reproduces its source tree"
                )
        self._require_materialization_deadline(deadline)
        package_authority.verify_package(
            manifest=manifest,
            source_extraction_receipt=extraction_receipt,
            proof_objects=proof_objects,
            publisher_verification=publisher_verification,
        )
        self._require_materialization_deadline(deadline)
        return self._verified_from_parts(
            manifest=manifest,
            receipt=receipt,
            extraction_receipt=extraction_receipt,
            source_archive=source_archive,
            source_directory=source_directory,
            proof_objects=proof_objects,
            publisher_verification=publisher_verification,
            deadline=deadline,
        )

    def _verified_from_parts(
        self,
        *,
        manifest: TaskAdapterManifest,
        receipt: TaskAdapterVerificationReceipt,
        extraction_receipt: SourceArchiveExtractionReceipt,
        source_archive: bytes,
        source_directory: Path,
        proof_objects: Mapping[str, bytes],
        publisher_verification: bytes,
        deadline: float | None = None,
    ) -> VerifiedTaskAdapter:
        source_contents = {
            descriptor.relative_path: self._read_bounded(
                source_directory / descriptor.relative_path,
                byte_limit=self.settings.source_byte_limit,
                deadline=deadline,
            )
            for descriptor in extraction_receipt.source_tree_files
        }
        self._require_materialization_deadline(deadline)
        return VerifiedTaskAdapter(
            manifest=manifest,
            verification_receipt=receipt,
            source_extraction_receipt=extraction_receipt,
            source_archive=source_archive,
            source_contents=source_contents,
            proof_objects=proof_objects,
            publisher_verification=publisher_verification,
        )

    def _persist_activation(
        self,
        activation: TaskAdapterActivationRecord,
        authority_envelope: bytes,
    ) -> None:
        target = self._activation_path(activation.activation_id)
        if target.exists() or target.is_symlink():
            observed, envelope = self._read_activation(activation.activation_id)
            if observed != activation or envelope != authority_envelope:
                raise TaskAdapterStoreError(
                    "task adapter activation ID collides with different bytes"
                )
            return
        with tempfile.TemporaryDirectory(
            prefix=".staging-activation-",
            dir=self.state_path / "activations",
        ) as staging_name:
            staging = Path(staging_name)
            (staging / "activation.json").write_bytes(activation.to_json_bytes())
            (staging / "authority-envelope.bin").write_bytes(authority_envelope)
            self._flush_tree(staging)
            self._make_read_only(staging)
            os.rename(staging, target)
            self._fsync_directory(target.parent)

    def _read_activation(
        self,
        activation_id: str,
    ) -> tuple[TaskAdapterActivationRecord, bytes]:
        target = self._activation_path(activation_id)
        if target.is_symlink() or not target.is_dir():
            raise TaskAdapterStoreError("task adapter activation is missing")
        self._validate_committed_permissions(target)
        self._require_exact_children(
            target,
            {"activation.json", "authority-envelope.bin"},
        )
        activation = TaskAdapterActivationRecord.from_json_bytes(
            self._read_bounded(target / "activation.json")
        )
        envelope = self._read_bounded(target / "authority-envelope.bin")
        if (
            activation.activation_id != activation_id
            or tree_or_blob_digest(envelope) != activation.authority_envelope_digest
        ):
            raise TaskAdapterStoreError("task adapter activation closure is invalid")
        return activation, envelope

    def _read_active_pointer(
        self,
        pointer_path: Path,
    ) -> TaskAdapterActivationRecord | None:
        if not pointer_path.exists() and not pointer_path.is_symlink():
            return None
        if pointer_path.is_symlink() or not pointer_path.is_file():
            raise TaskAdapterStoreError("task adapter active pointer is unsafe")
        return TaskAdapterActivationRecord.from_json_bytes(
            self._read_bounded(pointer_path)
        )

    def _write_active_pointer(
        self,
        pointer_path: Path,
        activation: TaskAdapterActivationRecord,
    ) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".active-",
            dir=pointer_path.parent,
        )
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(activation.to_json_bytes())
            file_handle.flush()
            os.fsync(file_handle.fileno())
        temporary_path = Path(temporary_name)
        temporary_path.chmod(0o600)
        os.replace(temporary_path, pointer_path)
        self._fsync_directory(pointer_path.parent)

    def _read_manifest(
        self,
        path: Path,
        *,
        deadline: float | None = None,
    ) -> TaskAdapterManifest:
        return TaskAdapterManifest.from_json_bytes(
            self._read_bounded(path, deadline=deadline)
        )

    def _validate_package_bounds(self, package: TaskAdapterPackage) -> None:
        entry_count = 2 + len(package.proof_objects)
        byte_count = (
            len(package.manifest.to_json_bytes())
            + len(package.source_archive)
            + len(package.publisher_verification)
            + sum(len(payload) for payload in package.proof_objects.values())
        )
        if entry_count > self.settings.package_entry_limit:
            raise TaskAdapterStoreError("task adapter package entry limit exceeded")
        if byte_count > self.settings.package_byte_limit:
            raise TaskAdapterStoreError("task adapter package byte limit exceeded")

    def _validate_stored_bounds(self, package_path: Path) -> None:
        paths = tuple(package_path.rglob("*"))
        if len(paths) > self.settings.package_entry_limit:
            raise TaskAdapterStoreError("task adapter package entry limit exceeded")
        package_bytes = 0
        source_bytes = 0
        source_root = package_path / "source"
        for path in paths:
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise TaskAdapterStoreError("task adapter package entry is unsafe")
            if not path.is_file():
                continue
            size = path.stat(follow_symlinks=False).st_size
            if source_root in path.parents:
                source_bytes += size
            else:
                package_bytes += size
        if package_bytes > self.settings.package_byte_limit:
            raise TaskAdapterStoreError("task adapter package byte limit exceeded")
        if source_bytes > self.settings.source_byte_limit:
            raise TaskAdapterStoreError("task adapter source byte limit exceeded")

    @staticmethod
    def _validate_materialization_bounds(
        package_path: Path,
        maximum_entries: int,
        maximum_bytes: int,
        deadline: float,
    ) -> None:
        if package_path.is_symlink() or not package_path.is_dir():
            raise TaskAdapterStoreError("task adapter package is missing or unsafe")
        paths = tuple(package_path.rglob("*"))
        if any(
            path.is_symlink() or (not path.is_file() and not path.is_dir())
            for path in paths
        ):
            raise TaskAdapterStoreError("task adapter package entry is unsafe")
        control_paths = {
            package_path / _MANIFEST_NAME,
            package_path / _EXTRACTION_RECEIPT_NAME,
            package_path / _COMMIT_NAME,
        }
        source_root = package_path / "source"
        archive_root = package_path / "archive"
        proof_root = package_path / "proofs"
        publisher_path = package_path / _PUBLISHER_VERIFICATION_NAME
        source_file_sizes = []
        source_archive_sizes = []
        proof_object_sizes = []
        publisher_verification_sizes = []
        for path in (item for item in paths if item.is_file()):
            TaskAdapterPackageStore._require_materialization_deadline(deadline)
            if path in control_paths:
                continue
            size = path.stat(follow_symlinks=False).st_size
            if source_root in path.parents:
                source_file_sizes.append(size)
            elif archive_root in path.parents:
                source_archive_sizes.append(size)
            elif proof_root in path.parents:
                proof_object_sizes.append(size)
            elif path == publisher_path:
                publisher_verification_sizes.append(size)
            else:
                raise TaskAdapterStoreError(
                    "task adapter materialization closure is not exact"
                )
        if len(source_archive_sizes) != 1 or len(publisher_verification_sizes) != 1:
            raise TaskAdapterStoreError(
                "task adapter materialization closure is not exact"
            )
        entries, materialized_bytes = task_adapter_materialization_usage(
            source_file_sizes=tuple(source_file_sizes),
            source_archive_sizes=tuple(source_archive_sizes),
            proof_object_sizes=tuple(proof_object_sizes),
            publisher_verification_sizes=tuple(publisher_verification_sizes),
        )
        if entries > maximum_entries or materialized_bytes > maximum_bytes:
            raise TaskAdapterStoreError(
                "task adapter package exceeds remaining replay materialization budget"
            )

    @staticmethod
    def _require_materialization_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskAdapterStoreError(
                "task adapter replay materialization deadline expired"
            )

    def _read_bounded(
        self,
        path: Path,
        *,
        byte_limit: int | None = None,
        deadline: float | None = None,
    ) -> bytes:
        self._require_materialization_deadline(deadline)
        limit = self.settings.package_byte_limit if byte_limit is None else byte_limit
        if path.is_symlink() or not path.is_file():
            raise TaskAdapterStoreError("task adapter package file is unsafe")
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise TaskAdapterStoreError(
                "task adapter package file must be an independent regular file"
            )
        payload_parts = []
        remaining = limit + 1
        with path.open("rb") as file_handle:
            while remaining > 0:
                self._require_materialization_deadline(deadline)
                payload_part = file_handle.read(min(remaining, io.DEFAULT_BUFFER_SIZE))
                if not payload_part:
                    break
                payload_parts.append(payload_part)
                remaining -= len(payload_part)
        self._require_materialization_deadline(deadline)
        payload = b"".join(payload_parts)
        if len(payload) > limit:
            raise TaskAdapterStoreError("task adapter package file exceeds bound")
        return payload

    def _require_exact_children(self, directory: Path, names: set[str]) -> None:
        if directory.is_symlink() or not directory.is_dir():
            raise TaskAdapterStoreError("task adapter package directory is unsafe")
        children = tuple(directory.iterdir())
        if len(children) > self.settings.package_entry_limit:
            raise TaskAdapterStoreError("task adapter package entry limit exceeded")
        if {child.name for child in children} != names:
            raise TaskAdapterStoreError("task adapter package closure is not exact")

    def _validate_locations(self) -> None:
        expected_state_path = self.state_root / self.settings.state_path
        if (
            not self.state_path.is_absolute()
            or not self.state_root.is_absolute()
            or self.state_path != self.state_path.absolute()
            or self.state_root != self.state_root.absolute()
            or self.state_path == self.state_root
            or self.state_root not in self.state_path.parents
            or self.state_path != expected_state_path
            or self.state_root in {Path("/"), Path.home()}
        ):
            raise TaskAdapterStoreError(
                "task adapter state path must be normalized below its state root"
            )
        for path in (self.state_root, *self.state_root.parents):
            if path.is_symlink():
                raise TaskAdapterStoreError(
                    "task adapter state path contains a symlinked ancestor"
                )

    def _ensure_directories(self) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (
            self.state_root.resolve() != self.state_root
            or self.state_path.resolve() != self.state_path
        ):
            raise TaskAdapterStoreError(
                "task adapter store contains a symlinked path component"
            )
        for name in ("packages", "activations", "active", "staging"):
            path = self.state_path / name
            if path.is_symlink():
                raise TaskAdapterStoreError(
                    "task adapter store directory cannot be a symlink"
                )
            path.mkdir(exist_ok=True, mode=0o700)
        for path in (
            self.state_path,
            *(
                self.state_path / name
                for name in ("packages", "activations", "active", "staging")
            ),
        ):
            metadata = path.stat(follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
                raise TaskAdapterStoreError(
                    "task adapter store directories must be private"
                )

    @staticmethod
    def _validate_committed_permissions(root: Path) -> None:
        root_metadata = root.stat(follow_symlinks=False)
        if stat.S_IMODE(root_metadata.st_mode) != 0o555:
            raise TaskAdapterStoreError(
                "committed task adapter directory is not read-only"
            )
        for path in root.rglob("*"):
            metadata = path.stat(follow_symlinks=False)
            mode = stat.S_IMODE(metadata.st_mode)
            relative_path = path.relative_to(root)
            source_file = (
                not path.is_dir()
                and relative_path.parts
                and relative_path.parts[0] == "source"
            )
            if path.is_dir():
                valid_mode = mode == 0o555
            elif source_file:
                valid_mode = mode in {0o444, 0o555}
            else:
                valid_mode = mode == 0o444
            if not valid_mode:
                raise TaskAdapterStoreError(
                    "committed task adapter entry is not read-only"
                )

    def _prepare_locked(self) -> None:
        self._ensure_directories()
        transient_directories = (
            (self.state_path / "packages", ".staging-package-"),
            (self.state_path / "activations", ".staging-activation-"),
            (self.state_path / "staging", ".validation-source-"),
        )
        for parent, prefix in transient_directories:
            for path in tuple(parent.iterdir()):
                if not path.name.startswith(prefix):
                    continue
                if path.is_symlink() or not path.is_dir():
                    raise TaskAdapterStoreError(
                        "task adapter transient directory is unsafe"
                    )
                shutil.rmtree(path)
        active_directory = self.state_path / "active"
        for path in tuple(active_directory.iterdir()):
            if not path.name.startswith(".active-"):
                continue
            if path.is_symlink() or not path.is_file():
                raise TaskAdapterStoreError("task adapter transient pointer is unsafe")
            path.unlink()

    def _lease(self) -> AbstractContextManager:
        self._ensure_directories()
        lock_path = self.state_path / _LOCK_NAME
        if lock_path.is_symlink():
            raise TaskAdapterStoreError("task adapter store lock is unsafe")
        descriptor = os.open(
            lock_path,
            os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise TaskAdapterStoreError(
                "task adapter store lock must be a regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return os.fdopen(descriptor, "rb")

    def _package_path(self, verification_receipt_id: str) -> Path:
        require_content_id(verification_receipt_id, "verification_receipt_id")
        return self.state_path / "packages" / verification_receipt_id.rsplit(":", 1)[1]

    def _activation_path(self, activation_id: str) -> Path:
        require_content_id(activation_id, "activation_id")
        return self.state_path / "activations" / activation_id.rsplit(":", 1)[1]

    def _active_pointer_path(
        self,
        scope_contract_id: str,
        task_family_id: str,
        task_adapter_id: str,
    ) -> Path:
        binding_id = content_id(
            "task-adapter-active-binding",
            {
                "scope_contract_id": scope_contract_id,
                "task_family_id": task_family_id,
                "task_adapter_id": task_adapter_id,
            },
        )
        return self.state_path / "active" / f"{binding_id.rsplit(':', 1)[1]}.json"

    @staticmethod
    def _flush_tree(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                with path.open("rb") as file_handle:
                    os.fsync(file_handle.fileno())
            elif path.is_dir():
                TaskAdapterPackageStore._fsync_directory(path)
        TaskAdapterPackageStore._fsync_directory(root)

    @staticmethod
    def _make_read_only(root: Path) -> None:
        for path in sorted(root.rglob("*"), reverse=True):
            executable = path.is_file() and path.stat().st_mode & 0o111
            path.chmod(0o555 if path.is_dir() or executable else 0o444)
        root.chmod(0o555)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        os.fsync(descriptor)
        os.close(descriptor)
