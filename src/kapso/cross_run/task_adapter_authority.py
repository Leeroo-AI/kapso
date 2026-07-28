"""Configured deterministic verification for trusted task-adapter packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from kapso.cross_run.canonical import parse_json_bytes, tree_or_blob_digest
from kapso.cross_run.github.materializer import SourceArchiveExtractionReceipt
from kapso.cross_run.settings import TaskAdapterAuthorityTrustSettings
from kapso.cross_run.task_adapters import (
    TaskAdapterActivationRecord,
    TaskAdapterManifest,
)


class TaskAdapterAuthorityError(ValueError):
    """A package or activation differs from its configured trust envelope."""


@dataclass(frozen=True)
class CanonicalTaskAdapterAuthority:
    """Verify the canonical proof and activation envelopes used by Kapso."""

    settings: TaskAdapterAuthorityTrustSettings

    def __post_init__(self) -> None:
        if type(self.settings) is not TaskAdapterAuthorityTrustSettings:
            raise TaskAdapterAuthorityError(
                "task-adapter authority requires exact trust settings"
            )

    @property
    def authority_id(self) -> str:
        return self.settings.authority_id

    @property
    def authority_version(self) -> str:
        return self.settings.authority_version

    def verify_package(
        self,
        *,
        manifest: TaskAdapterManifest,
        source_extraction_receipt: SourceArchiveExtractionReceipt,
        proof_objects: Mapping[str, bytes],
        publisher_verification: bytes,
    ) -> None:
        if (
            type(manifest) is not TaskAdapterManifest
            or type(source_extraction_receipt) is not SourceArchiveExtractionReceipt
            or not isinstance(proof_objects, Mapping)
            or type(publisher_verification) is not bytes
        ):
            raise TaskAdapterAuthorityError(
                "task-adapter package verification lacks exact inputs"
            )
        expected_proof_refs = {
            manifest.sanitation_report_id,
            *manifest.validation_refs,
        }
        if set(proof_objects) != expected_proof_refs:
            raise TaskAdapterAuthorityError(
                "task-adapter package differs from its proof closure"
            )
        for proof_ref, payload in proof_objects.items():
            if type(payload) is not bytes or parse_json_bytes(payload) != {
                "manifest_id": manifest.task_adapter_manifest_id,
                "outcome": "passed",
                "proof_ref": proof_ref,
                "tree_hash": manifest.tree_hash,
            }:
                raise TaskAdapterAuthorityError(
                    "task-adapter proof differs from its manifest"
                )
        expected_verification = {
            "archive_digest": source_extraction_receipt.source_archive_digest,
            "full_manifest_digest": tree_or_blob_digest(manifest.to_json_bytes()),
            "manifest_id": manifest.task_adapter_manifest_id,
            "proof_digests": {
                proof_ref: tree_or_blob_digest(payload)
                for proof_ref, payload in proof_objects.items()
            },
            "publisher_attestation": manifest.publisher_attestation,
            "tree_hash": manifest.tree_hash,
        }
        if parse_json_bytes(publisher_verification) != expected_verification:
            raise TaskAdapterAuthorityError(
                "task-adapter publisher verification differs from its package"
            )

    def verify_activation(
        self,
        *,
        activation: TaskAdapterActivationRecord,
        authority_envelope: bytes,
    ) -> None:
        if (
            type(activation) is not TaskAdapterActivationRecord
            or type(authority_envelope) is not bytes
            or parse_json_bytes(authority_envelope)
            != {
                "scope_contract_id": activation.scope_contract_id,
                "task_adapter_id": activation.task_adapter_id,
                "task_family_id": activation.task_family_id,
                "verification_receipt_id": activation.verification_receipt_id,
            }
        ):
            raise TaskAdapterAuthorityError(
                "task-adapter activation differs from its authority envelope"
            )


__all__ = [
    "CanonicalTaskAdapterAuthority",
    "TaskAdapterAuthorityError",
]
