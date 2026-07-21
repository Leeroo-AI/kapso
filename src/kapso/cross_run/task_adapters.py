"""Typed trust boundary for shared task-adapter packages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import ClassVar, Mapping, Protocol

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    StrictContract,
    TaskAdapterManifest,
)
from kapso.cross_run.github.materializer import SourceArchiveExtractionReceipt


def task_adapter_binding_id(task_family_id: str, task_adapter_id: str) -> str:
    """Return the canonical identity shared by enrollment and execution."""

    require_identifier(task_family_id, "task_family_id")
    require_identifier(task_adapter_id, "task_adapter_id")
    return content_id(
        "task-adapter-binding",
        {
            "task_family_id": task_family_id,
            "task_adapter_id": task_adapter_id,
        },
    )


def task_adapter_materialization_usage(
    *,
    source_file_sizes: tuple[int, ...],
    source_archive_sizes: tuple[int, ...],
    proof_object_sizes: tuple[int, ...],
    publisher_verification_sizes: tuple[int, ...],
) -> tuple[int, int]:
    """Count the exact adapter payload closure acquired for replay."""

    size_groups = (
        source_file_sizes,
        source_archive_sizes,
        proof_object_sizes,
        publisher_verification_sizes,
    )
    if (
        len(source_archive_sizes) != 1
        or len(publisher_verification_sizes) != 1
        or any(
            type(size) is not int or size < 0 for sizes in size_groups for size in sizes
        )
    ):
        raise ContractValidationError(
            "task adapter materialization payload sizes are invalid"
        )
    return (
        sum(len(sizes) for sizes in size_groups),
        sum(size for sizes in size_groups for size in sizes),
    )


@dataclass(frozen=True)
class TaskAdapterVerificationReceipt(StrictContract):
    verification_receipt_id: str
    task_adapter_manifest_id: str
    full_manifest_digest: str
    publisher_attestation_digest: str
    source_extraction_receipt_id: str
    source_archive_ref: str
    source_archive_digest: str
    source_tree_hash: str
    proof_object_digests: Mapping[str, str]
    publisher_verification_digest: str
    verifier_id: str
    verifier_version: str

    CONTENT_NAMESPACE: ClassVar[str] = "task-adapter-verification-receipt"
    IDENTITY_FIELD: ClassVar[str] = "verification_receipt_id"

    def _validate(self) -> None:
        require_content_id(
            self.task_adapter_manifest_id,
            "task_adapter_manifest_id",
        )
        require_content_id(
            self.source_extraction_receipt_id,
            "source_extraction_receipt_id",
        )
        for value, name in (
            (self.full_manifest_digest, "full_manifest_digest"),
            (self.publisher_attestation_digest, "publisher_attestation_digest"),
            (self.source_archive_digest, "source_archive_digest"),
            (self.source_tree_hash, "source_tree_hash"),
            (
                self.publisher_verification_digest,
                "publisher_verification_digest",
            ),
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ContractValidationError(f"{name} must be a sha256 digest")
        source_archive_path = PurePosixPath(self.source_archive_ref)
        if (
            source_archive_path.is_absolute()
            or len(source_archive_path.parts) != 1
            or source_archive_path.as_posix() != self.source_archive_ref
            or not self.source_archive_ref.endswith((".tar", ".tar.zst"))
        ):
            raise ContractValidationError(
                "source_archive_ref must name one normalized tar archive"
            )
        if not self.proof_object_digests:
            raise ContractValidationError(
                "task adapter proof object closure must not be empty"
            )
        for proof_ref, digest in self.proof_object_digests.items():
            require_identifier(proof_ref, "proof_object_digests key")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                raise ContractValidationError(
                    "task adapter proof object digest is invalid"
                )
        require_identifier(self.verifier_id, "task adapter verifier_id")
        require_identifier(self.verifier_version, "task adapter verifier_version")

    @property
    def proof_object_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                content_id(
                    "task-adapter-proof-object",
                    {
                        "digest": digest,
                        "proof_ref": proof_ref,
                    },
                )
                for proof_ref, digest in self.proof_object_digests.items()
            )
        )


@dataclass(frozen=True)
class TaskAdapterPackage:
    manifest: TaskAdapterManifest
    source_archive: bytes
    proof_objects: Mapping[str, bytes]
    publisher_verification: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.source_archive, bytes) or not self.source_archive:
            raise ContractValidationError("task adapter source archive is empty")
        if not self.proof_objects or any(
            not isinstance(key, str) or not isinstance(value, bytes) or not value
            for key, value in self.proof_objects.items()
        ):
            raise ContractValidationError("task adapter proof objects are invalid")
        if (
            not isinstance(self.publisher_verification, bytes)
            or not self.publisher_verification
        ):
            raise ContractValidationError(
                "task adapter publisher verification is empty"
            )
        object.__setattr__(
            self,
            "proof_objects",
            MappingProxyType(dict(self.proof_objects)),
        )


@dataclass(frozen=True)
class TaskAdapterActivationRecord(StrictContract):
    activation_id: str
    scope_contract_id: str
    task_family_id: str
    task_adapter_id: str
    task_adapter_manifest_id: str
    verification_receipt_id: str
    predecessor_activation_id: str | None
    authority_id: str
    authority_version: str
    authority_envelope_digest: str

    CONTENT_NAMESPACE: ClassVar[str] = "task-adapter-activation"
    IDENTITY_FIELD: ClassVar[str] = "activation_id"

    def _validate(self) -> None:
        for value, name in (
            (self.scope_contract_id, "scope_contract_id"),
            (self.task_adapter_manifest_id, "task_adapter_manifest_id"),
            (self.verification_receipt_id, "verification_receipt_id"),
        ):
            require_content_id(value, name)
        if self.predecessor_activation_id is not None:
            require_content_id(
                self.predecessor_activation_id,
                "predecessor_activation_id",
            )
        for value, name in (
            (self.task_family_id, "task_family_id"),
            (self.task_adapter_id, "task_adapter_id"),
            (self.authority_id, "authority_id"),
            (self.authority_version, "authority_version"),
        ):
            require_identifier(value, name)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.authority_envelope_digest) is None:
            raise ContractValidationError(
                "authority_envelope_digest must be a sha256 digest"
            )


@dataclass(frozen=True)
class VerifiedTaskAdapter:
    manifest: TaskAdapterManifest
    verification_receipt: TaskAdapterVerificationReceipt
    source_extraction_receipt: SourceArchiveExtractionReceipt
    source_archive: bytes
    source_contents: Mapping[str, bytes]
    proof_objects: Mapping[str, bytes]
    publisher_verification: bytes

    def __post_init__(self) -> None:
        receipt = self.verification_receipt
        extraction = self.source_extraction_receipt
        expected_proof_refs = {
            self.manifest.sanitation_report_id,
            *self.manifest.validation_refs,
        }
        if (
            receipt.task_adapter_manifest_id != self.manifest.task_adapter_manifest_id
            or receipt.full_manifest_digest
            != tree_or_blob_digest(self.manifest.to_json_bytes())
            or receipt.publisher_attestation_digest
            != tree_or_blob_digest(
                canonical_json_bytes(self.manifest.publisher_attestation)
            )
            or receipt.source_archive_ref != self.manifest.source_tree_ref
            or receipt.source_tree_hash != self.manifest.tree_hash
            or set(receipt.proof_object_digests) != expected_proof_refs
            or receipt.source_extraction_receipt_id != extraction.extraction_receipt_id
            or extraction.artifact_id != self.manifest.task_adapter_manifest_id
            or extraction.source_archive_ref != self.manifest.source_tree_ref
            or extraction.source_archive_digest != receipt.source_archive_digest
            or extraction.source_tree_hash != self.manifest.tree_hash
            or tree_or_blob_digest(self.source_archive) != receipt.source_archive_digest
            or tree_or_blob_digest(self.publisher_verification)
            != receipt.publisher_verification_digest
        ):
            raise ContractValidationError(
                "task adapter verification receipt differs from its manifest"
            )
        expected_source_paths = {
            descriptor.relative_path for descriptor in extraction.source_tree_files
        }
        if set(self.source_contents) != expected_source_paths:
            raise ContractValidationError(
                "task adapter source contents differ from extraction receipt"
            )
        for descriptor in extraction.source_tree_files:
            payload = self.source_contents[descriptor.relative_path]
            if (
                not isinstance(payload, bytes)
                or len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.digest
            ):
                raise ContractValidationError(
                    "task adapter source content digest differs from extraction receipt"
                )
        if set(self.proof_objects) != expected_proof_refs or any(
            tree_or_blob_digest(self.proof_objects[proof_ref])
            != receipt.proof_object_digests[proof_ref]
            for proof_ref in expected_proof_refs
        ):
            raise ContractValidationError(
                "task adapter proof bytes differ from verification receipt"
            )
        object.__setattr__(
            self,
            "source_contents",
            MappingProxyType(dict(self.source_contents)),
        )
        object.__setattr__(
            self,
            "proof_objects",
            MappingProxyType(dict(self.proof_objects)),
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


class VerifiedTaskAdapterProvider(Protocol):
    """Resolve trusted active packages separately from exact replay pins."""

    def resolve_active(
        self,
        *,
        scope_contract_id: str,
        task_family_id: str,
        task_adapter_id: str,
    ) -> VerifiedTaskAdapter: ...

    def resolve_exact(
        self,
        *,
        task_adapter_manifest_id: str,
        verification_receipt_id: str,
    ) -> VerifiedTaskAdapter: ...


class TaskAdapterAuthority(Protocol):
    authority_id: str
    authority_version: str

    def verify_package(
        self,
        *,
        manifest: TaskAdapterManifest,
        source_extraction_receipt: SourceArchiveExtractionReceipt,
        proof_objects: Mapping[str, bytes],
        publisher_verification: bytes,
    ) -> None: ...

    def verify_activation(
        self,
        *,
        activation: TaskAdapterActivationRecord,
        authority_envelope: bytes,
    ) -> None: ...
