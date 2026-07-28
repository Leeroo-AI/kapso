"""Bounded immutable starting-artifact bytes for launch resolution."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import SourceFileDescriptor, TaskContextBinding
from kapso.cross_run.launch.contracts import (
    LaunchStartingArtifact,
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

    @property
    def content_ids(self) -> Mapping[str, str]:
        """Return the exact immutable reference-to-content identity mapping."""

        return self._content_ids

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


def build_launch_starting_artifact_provider(
    *,
    sources: Mapping[str, tuple[Path, str]],
    settings: LaunchSettings,
) -> LaunchStartingArtifactSetProvider:
    """Seal caller-owned source directories into immutable launch artifacts."""

    if (
        not isinstance(sources, Mapping)
        or not sources
        or type(settings) is not LaunchSettings
        or any(
            not isinstance(artifact_ref, str)
            or not artifact_ref.strip()
            or type(source) is not tuple
            or len(source) != 2
            or not isinstance(source[0], Path)
            or not isinstance(source[1], str)
            for artifact_ref, source in sources.items()
        )
    ):
        raise LaunchStartingArtifactProviderError(
            "starting-artifact sources require exact paths and mount points"
        )
    artifacts = []
    total_entries = 0
    total_bytes = 0
    for artifact_ref in sorted(sources):
        source_root, mount_path = sources[artifact_ref]
        if (
            not source_root.is_absolute()
            or source_root.resolve() != source_root
            or not source_root.is_dir()
            or source_root.is_symlink()
        ):
            raise LaunchStartingArtifactProviderError(
                "starting-artifact source must be one real absolute directory"
            )
        source_contents, descriptors = _read_source_tree(
            source_root,
            maximum_entries=settings.starting_artifact_entry_limit - total_entries,
            maximum_bytes=settings.starting_artifact_byte_limit - total_bytes,
        )
        total_entries += len(descriptors)
        total_bytes += sum(descriptor.size for descriptor in descriptors)
        artifact = LaunchStartingArtifact.mint(
            starting_artifact_ref=artifact_ref,
            mount_path=mount_path,
            materialized_tree_hash=source_tree_digest(
                {
                    descriptor.relative_path: (
                        descriptor.digest,
                        descriptor.mode,
                        descriptor.size,
                    )
                    for descriptor in descriptors
                }
            ),
            source_files=descriptors,
        )
        artifacts.append(
            VerifiedLaunchStartingArtifact(
                artifact=artifact,
                source_contents=source_contents,
            )
        )
    return LaunchStartingArtifactSetProvider(
        tuple(
            sorted(
                artifacts,
                key=lambda item: item.artifact.starting_artifact_content_id,
            )
        ),
        settings,
    )


def _read_source_tree(
    root: Path,
    *,
    maximum_entries: int,
    maximum_bytes: int,
) -> tuple[dict[str, bytes], tuple[SourceFileDescriptor, ...]]:
    if maximum_entries <= 0 or maximum_bytes <= 0:
        raise LaunchStartingArtifactProviderError(
            "starting-artifact closure exceeds its configured bounds"
        )
    paths: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            ordered_entries = tuple(sorted(entries, key=lambda item: item.name))
        for entry in ordered_entries:
            if entry.is_symlink():
                raise LaunchStartingArtifactProviderError(
                    "starting-artifact source cannot contain symbolic links"
                )
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                paths.append(Path(entry.path))
            else:
                raise LaunchStartingArtifactProviderError(
                    "starting-artifact source contains a non-regular entry"
                )
    if not paths or len(paths) > maximum_entries:
        raise LaunchStartingArtifactProviderError(
            "starting-artifact source is empty or exceeds its entry bound"
        )
    contents: dict[str, bytes] = {}
    descriptors = []
    consumed_bytes = 0
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix()
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise LaunchStartingArtifactProviderError(
                    "starting-artifact file is not one independent regular file"
                )
            remaining = maximum_bytes - consumed_bytes
            payload = handle.read(remaining + 1)
        if len(payload) > remaining:
            raise LaunchStartingArtifactProviderError(
                "starting-artifact source exceeds its byte bound"
            )
        consumed_bytes += len(payload)
        contents[relative_path] = payload
        descriptors.append(
            SourceFileDescriptor(
                relative_path=relative_path,
                digest=tree_or_blob_digest(payload),
                mode=("100755" if metadata.st_mode & 0o111 else "100644"),
                size=len(payload),
            )
        )
    return contents, tuple(descriptors)


__all__ = [
    "build_launch_starting_artifact_provider",
    "LaunchStartingArtifactProviderError",
    "LaunchStartingArtifactSetProvider",
]
