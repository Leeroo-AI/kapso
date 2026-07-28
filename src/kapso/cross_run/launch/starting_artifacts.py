"""Bounded immutable starting-artifact bytes for launch resolution."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from kapso.cross_run.contracts import TaskContextBinding
from kapso.cross_run.launch.contracts import (
    LaunchStartingArtifactMaterializationReceipt,
)
from kapso.cross_run.launch.resolver import (
    VerifiedLaunchStartingArtifact,
    VerifiedLaunchStartingArtifacts,
)
from kapso.cross_run.settings import LaunchSettings


class LaunchStartingArtifactProviderError(ValueError):
    """Starting-artifact bytes differ from the requested immutable closure."""


class LaunchStartingArtifactSetProvider:
    """Expose one request-scoped, content-addressed artifact byte closure."""

    def __init__(
        self,
        artifacts: tuple[VerifiedLaunchStartingArtifact, ...],
        settings: LaunchSettings,
    ) -> None:
        if (
            type(artifacts) is not tuple
            or any(
                type(item) is not VerifiedLaunchStartingArtifact for item in artifacts
            )
            or type(settings) is not LaunchSettings
        ):
            raise LaunchStartingArtifactProviderError(
                "starting-artifact provider requires exact verified inputs"
            )
        artifact_ids = tuple(
            item.artifact.starting_artifact_content_id for item in artifacts
        )
        artifact_refs = tuple(item.artifact.starting_artifact_ref for item in artifacts)
        if artifact_ids != tuple(sorted(set(artifact_ids))) or len(
            artifact_refs
        ) != len(set(artifact_refs)):
            raise LaunchStartingArtifactProviderError(
                "starting-artifact provider inputs must be sorted and unique"
            )
        self._artifacts = artifacts
        self._settings = settings
        self._content_ids = MappingProxyType(
            {
                item.artifact.starting_artifact_ref: (
                    item.artifact.starting_artifact_content_id
                )
                for item in artifacts
            }
        )

    def materialize_exact(
        self,
        *,
        task_context_binding: TaskContextBinding,
        expected_artifact_content_ids: Mapping[str, str],
        maximum_entries: int,
        maximum_bytes: int,
    ) -> VerifiedLaunchStartingArtifacts:
        if (
            type(task_context_binding) is not TaskContextBinding
            or not isinstance(expected_artifact_content_ids, Mapping)
            or type(maximum_entries) is not int
            or maximum_entries <= 0
            or type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or maximum_entries != self._settings.starting_artifact_entry_limit
            or maximum_bytes != self._settings.starting_artifact_byte_limit
            or dict(expected_artifact_content_ids) != dict(self._content_ids)
            or set(task_context_binding.starting_artifact_refs)
            != set(self._content_ids)
        ):
            raise LaunchStartingArtifactProviderError(
                "starting-artifact request differs from its verified closure"
            )
        entry_count = sum(len(item.artifact.source_files) for item in self._artifacts)
        byte_count = sum(
            descriptor.size
            for item in self._artifacts
            for descriptor in item.artifact.source_files
        )
        if entry_count > maximum_entries or byte_count > maximum_bytes:
            raise LaunchStartingArtifactProviderError(
                "starting-artifact closure exceeds configured bounds"
            )
        receipt = LaunchStartingArtifactMaterializationReceipt.mint(
            task_context_binding_id=task_context_binding.task_context_binding_id,
            starting_artifacts=tuple(item.artifact for item in self._artifacts),
            materializer_id=self._settings.starting_artifact_materializer_id,
            materializer_version=self._settings.starting_artifact_materializer_version,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        task_context_binding.task_context_binding_id,
                        *(
                            item.artifact.starting_artifact_content_id
                            for item in self._artifacts
                        ),
                    }
                )
            ),
        )
        return VerifiedLaunchStartingArtifacts(
            receipt=receipt,
            starting_artifacts=self._artifacts,
        )


__all__ = [
    "LaunchStartingArtifactProviderError",
    "LaunchStartingArtifactSetProvider",
]
