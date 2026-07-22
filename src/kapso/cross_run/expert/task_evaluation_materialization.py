"""Exact byte authorities shared by task-evaluation producers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    ExpertCandidateManifest,
    ExpertSourceTreeManifest,
    SourceFileDescriptor,
    TaskAdapterManifest,
    TaskAdapterReleaseMatrixCase,
    TaskAdapterReleaseMatrixStartingArtifact,
)
from kapso.cross_run.expert.store import ExpertCandidateCommitRecord
from kapso.cross_run.expert.triggers import ExpertParentTreeReceipt
from kapso.cross_run.github.materializer import SourceArchiveExtractionReceipt
from kapso.cross_run.task_adapters import (
    TaskAdapterVerificationReceipt,
    VerifiedTaskAdapter,
)


class TaskEvaluationMaterializationError(ValueError):
    """Task-evaluation bytes differ from their immutable authority."""


@dataclass(frozen=True)
class TaskEvaluationMaterializationLimits:
    maximum_entries: int
    maximum_bytes: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.maximum_entries,
                self.maximum_bytes,
                self.timeout_seconds,
            )
        ):
            raise TaskEvaluationMaterializationError(
                "task-evaluation materialization limits must be positive integers"
            )


def _verified_source_contents(
    descriptors: tuple[SourceFileDescriptor, ...],
    source_contents: Mapping[str, bytes],
    label: str,
) -> Mapping[str, bytes]:
    snapshot = dict(source_contents)
    expected_paths = {descriptor.relative_path for descriptor in descriptors}
    if set(snapshot) != expected_paths:
        raise TaskEvaluationMaterializationError(
            f"{label} bytes differ from the exact path closure"
        )
    for descriptor in descriptors:
        payload = snapshot[descriptor.relative_path]
        if (
            not isinstance(payload, bytes)
            or len(payload) != descriptor.size
            or tree_or_blob_digest(payload) != descriptor.digest
        ):
            raise TaskEvaluationMaterializationError(
                f"{label} bytes differ from their descriptor"
            )
    return MappingProxyType(snapshot)


@dataclass(frozen=True)
class VerifiedTaskEvaluationCandidate:
    manifest: ExpertCandidateManifest
    commit_record: ExpertCandidateCommitRecord
    source_tree: ExpertSourceTreeManifest
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if (
            type(self.manifest) is not ExpertCandidateManifest
            or type(self.commit_record) is not ExpertCandidateCommitRecord
            or type(self.source_tree) is not ExpertSourceTreeManifest
        ):
            raise TaskEvaluationMaterializationError(
                "candidate task-evaluation source requires typed immutable authorities"
            )
        if (
            self.commit_record.candidate_id != self.manifest.candidate_id
            or self.manifest.candidate_tree_ref
            != self.source_tree.source_tree_manifest_id
            or self.manifest.candidate_tree_hash != self.source_tree.tree_hash
        ):
            raise TaskEvaluationMaterializationError(
                "candidate task-evaluation source differs from its immutable authority"
            )
        object.__setattr__(
            self,
            "source_contents",
            _verified_source_contents(
                self.source_tree.files,
                self.source_contents,
                "candidate task-evaluation source",
            ),
        )

    @property
    def entry_count(self) -> int:
        return len(self.source_tree.files)

    @property
    def byte_count(self) -> int:
        return sum(descriptor.size for descriptor in self.source_tree.files)


@dataclass(frozen=True)
class VerifiedTaskEvaluationParent:
    release_manifest: ExpertBaseReleaseManifest
    parent_tree_receipt: ExpertParentTreeReceipt
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if (
            type(self.release_manifest) is not ExpertBaseReleaseManifest
            or type(self.parent_tree_receipt) is not ExpertParentTreeReceipt
        ):
            raise TaskEvaluationMaterializationError(
                "parent task-evaluation source requires typed immutable authorities"
            )
        receipt = self.parent_tree_receipt
        extraction = receipt.source_extraction_receipt
        archive_ref = self.release_manifest.source_archive_ref
        archive_digest = self.release_manifest.checksums[archive_ref]
        if (
            receipt.release_id != self.release_manifest.release_id
            or receipt.parent_tree_hash != extraction.source_tree_hash
            or extraction.artifact_id != self.release_manifest.release_id
            or extraction.source_archive_ref != archive_ref
            or extraction.source_archive_digest != archive_digest
            or receipt.cache_verification_receipt.asset_digests.get(archive_ref)
            != archive_digest
        ):
            raise TaskEvaluationMaterializationError(
                "parent task-evaluation source differs from its immutable authority"
            )
        object.__setattr__(
            self,
            "source_contents",
            _verified_source_contents(
                extraction.source_tree_files,
                self.source_contents,
                "parent task-evaluation source",
            ),
        )

    @property
    def entry_count(self) -> int:
        return len(self.parent_tree_receipt.source_extraction_receipt.source_tree_files)

    @property
    def byte_count(self) -> int:
        return sum(
            descriptor.size
            for descriptor in (
                self.parent_tree_receipt.source_extraction_receipt.source_tree_files
            )
        )


@dataclass(frozen=True)
class VerifiedTaskEvaluationStartingArtifact:
    artifact: TaskAdapterReleaseMatrixStartingArtifact
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if type(self.artifact) is not TaskAdapterReleaseMatrixStartingArtifact:
            raise TaskEvaluationMaterializationError(
                "task-evaluation starting artifact requires signed case authority"
            )
        object.__setattr__(
            self,
            "source_contents",
            _verified_source_contents(
                self.artifact.source_files,
                self.source_contents,
                "task-evaluation starting artifact",
            ),
        )

    @property
    def entry_count(self) -> int:
        return len(self.artifact.source_files)

    @property
    def byte_count(self) -> int:
        return sum(descriptor.size for descriptor in self.artifact.source_files)


@dataclass(frozen=True)
class VerifiedTaskEvaluationAdapterRuntime:
    manifest: TaskAdapterManifest
    verification_receipt: TaskAdapterVerificationReceipt
    source_extraction_receipt: SourceArchiveExtractionReceipt
    source_files: tuple[SourceFileDescriptor, ...]
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if (
            type(self.manifest) is not TaskAdapterManifest
            or type(self.verification_receipt) is not TaskAdapterVerificationReceipt
            or type(self.source_extraction_receipt)
            is not SourceArchiveExtractionReceipt
        ):
            raise TaskEvaluationMaterializationError(
                "task-evaluation adapter runtime requires typed package authority"
            )
        receipt = self.verification_receipt
        extraction = self.source_extraction_receipt
        expected_proof_refs = {
            self.manifest.sanitation_report_id,
            *self.manifest.validation_refs,
        }
        expected_files = tuple(
            descriptor
            for descriptor in extraction.source_tree_files
            if PurePosixPath(descriptor.relative_path).parts[0]
            != "release_matrix_assets"
        )
        source_descriptors = {
            descriptor.relative_path: descriptor for descriptor in self.source_files
        }
        evaluator_descriptor = source_descriptors.get(
            self.manifest.task_evaluator.executable_path
        )
        runtime_lock_descriptor = source_descriptors.get(
            self.manifest.runtime.dependency_lock_path
        )
        if (
            receipt.task_adapter_manifest_id != self.manifest.task_adapter_manifest_id
            or receipt.full_manifest_digest
            != tree_or_blob_digest(self.manifest.to_json_bytes())
            or receipt.publisher_attestation_digest
            != tree_or_blob_digest(
                canonical_json_bytes(self.manifest.publisher_attestation)
            )
            or receipt.source_extraction_receipt_id != extraction.extraction_receipt_id
            or receipt.source_archive_ref != self.manifest.source_tree_ref
            or set(receipt.proof_object_digests) != expected_proof_refs
            or extraction.artifact_id != self.manifest.task_adapter_manifest_id
            or extraction.source_archive_ref != self.manifest.source_tree_ref
            or extraction.source_archive_digest != receipt.source_archive_digest
            or extraction.source_tree_hash != self.manifest.tree_hash
            or receipt.source_tree_hash != self.manifest.tree_hash
            or self.source_files != expected_files
            or evaluator_descriptor is None
            or evaluator_descriptor.mode != "100755"
            or runtime_lock_descriptor is None
        ):
            raise TaskEvaluationMaterializationError(
                "task-evaluation adapter runtime differs from its verified package"
            )
        object.__setattr__(
            self,
            "source_contents",
            _verified_source_contents(
                self.source_files,
                self.source_contents,
                "task-evaluation adapter runtime",
            ),
        )
        if (
            tree_or_blob_digest(
                self.source_contents[self.manifest.runtime.dependency_lock_path]
            )
            != self.manifest.runtime.dependency_lock_digest
        ):
            raise TaskEvaluationMaterializationError(
                "task-evaluation adapter runtime lock differs from its manifest"
            )

    @classmethod
    def from_verified_adapter(
        cls,
        adapter: VerifiedTaskAdapter,
    ) -> VerifiedTaskEvaluationAdapterRuntime:
        if type(adapter) is not VerifiedTaskAdapter:
            raise TaskEvaluationMaterializationError(
                "task-evaluation adapter runtime requires an exact verified package"
            )
        return cls(
            manifest=adapter.manifest,
            verification_receipt=adapter.verification_receipt,
            source_extraction_receipt=adapter.source_extraction_receipt,
            source_files=adapter.evaluation_runtime_source_files,
            source_contents=adapter.evaluation_runtime_source_contents,
        )

    @property
    def entry_count(self) -> int:
        return len(self.source_files)

    @property
    def byte_count(self) -> int:
        return sum(descriptor.size for descriptor in self.source_files)


def materialize_task_evaluation_starting_artifacts(
    *,
    adapter: VerifiedTaskAdapter,
    signed_case: TaskAdapterReleaseMatrixCase,
) -> tuple[VerifiedTaskEvaluationStartingArtifact, ...]:
    """Select only one signed case's fixture bytes from its verified package."""

    if (
        type(adapter) is not VerifiedTaskAdapter
        or type(signed_case) is not TaskAdapterReleaseMatrixCase
    ):
        raise TaskEvaluationMaterializationError(
            "task-evaluation artifacts require exact package and case authorities"
        )
    manifest_cases = {
        case.release_matrix_case_id: case
        for case in adapter.manifest.release_matrix_cases
    }
    if manifest_cases.get(signed_case.release_matrix_case_id) != signed_case:
        raise TaskEvaluationMaterializationError(
            "task-evaluation signed case differs from its verified package"
        )
    return tuple(
        VerifiedTaskEvaluationStartingArtifact(
            artifact=artifact,
            source_contents={
                descriptor.relative_path: adapter.source_contents[
                    (
                        PurePosixPath(artifact.package_source_root)
                        / descriptor.relative_path
                    ).as_posix()
                ]
                for descriptor in artifact.source_files
            },
        )
        for artifact in signed_case.starting_artifacts
    )
