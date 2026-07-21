"""Verified starting-artifact byte closures for faithful source replay."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    ContractValidationError,
    ExpertSourceReplayContextMaterializationReceipt,
    ExpertSourceReplayStartingArtifact,
    TaskContextBinding,
)


@dataclass(frozen=True)
class SourceReplayMaterializationLimits:
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
            raise ContractValidationError(
                "source replay materialization limits must be positive integers"
            )


@dataclass(frozen=True)
class VerifiedSourceReplayStartingArtifact:
    artifact: ExpertSourceReplayStartingArtifact
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ExpertSourceReplayStartingArtifact):
            raise ContractValidationError(
                "source replay starting artifact must be a typed contract"
            )
        expected_paths = {
            descriptor.relative_path for descriptor in self.artifact.source_files
        }
        if set(self.source_contents) != expected_paths:
            raise ContractValidationError(
                "source replay artifact contents differ from the exact descriptor"
            )
        for descriptor in self.artifact.source_files:
            payload = self.source_contents[descriptor.relative_path]
            if (
                not isinstance(payload, bytes)
                or len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.digest
            ):
                raise ContractValidationError(
                    "source replay artifact content differs from its descriptor"
                )
        object.__setattr__(
            self,
            "source_contents",
            MappingProxyType(dict(self.source_contents)),
        )


@dataclass(frozen=True)
class VerifiedSourceReplayContext:
    receipt: ExpertSourceReplayContextMaterializationReceipt
    starting_artifacts: tuple[VerifiedSourceReplayStartingArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.receipt,
            ExpertSourceReplayContextMaterializationReceipt,
        ):
            raise ContractValidationError(
                "source replay context requires a typed materialization receipt"
            )
        if any(
            not isinstance(item, VerifiedSourceReplayStartingArtifact)
            for item in self.starting_artifacts
        ):
            raise ContractValidationError(
                "source replay context contains an unverified artifact closure"
            )
        artifacts = tuple(item.artifact for item in self.starting_artifacts)
        if artifacts != self.receipt.starting_artifacts:
            raise ContractValidationError(
                "source replay artifact byte closures differ from the receipt"
            )

    @property
    def entry_count(self) -> int:
        return sum(len(item.artifact.source_files) for item in self.starting_artifacts)

    @property
    def byte_count(self) -> int:
        return sum(
            descriptor.size
            for item in self.starting_artifacts
            for descriptor in item.artifact.source_files
        )


class SourceReplayContextProvider(Protocol):
    """Resolve only exact captured starting artifacts under acquisition limits."""

    def materialize_exact(
        self,
        task_context_binding: TaskContextBinding,
        expected_artifact_content_ids: Mapping[str, str],
        limits: SourceReplayMaterializationLimits,
    ) -> VerifiedSourceReplayContext: ...
