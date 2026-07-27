"""Descriptor-safe proof that the run workspace equals one checkpointed Git head."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import struct
import zlib
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.git_refs import git_tree_shas, require_git_ref_name
from kapso.cross_run.launch.workspace import StarterWorkspaceBuilder
from kapso.cross_run.settings import LaunchSettings

_GIT_INDEX_ENTRY_HEADER = struct.Struct("!LLLLLLLLLL20sH")
_GIT_INDEX_HEADER = struct.Struct("!4sLL")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_GIT_IDENTITY_PATTERN = re.compile(
    rb"^(author|committer) [^\x00-\x1f\x7f<>]+ "
    rb"<[^\x00-\x20\x7f<>]+> [0-9]+ [+-][0-9]{4}$"
)


class RunWorkspaceFrontierError(RuntimeError):
    """The live source tree or Git authority differs from its checkpoint."""


@dataclass(frozen=True)
class RunWorkspaceSourceTreeIdentity:
    """Bounded source-only identity that deliberately excludes root Git metadata."""

    workspace_identity: tuple[int, int]
    source_tree_digest: str
    source_entry_count: int
    source_size_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.workspace_identity) is not tuple
            or len(self.workspace_identity) != 2
            or any(
                type(value) is not int or value <= 0
                for value in self.workspace_identity
            )
            or not isinstance(self.source_tree_digest, str)
            or not self.source_tree_digest.startswith("sha256:")
            or type(self.source_entry_count) is not int
            or self.source_entry_count <= 0
            or type(self.source_size_bytes) is not int
            or self.source_size_bytes < 0
        ):
            raise RunWorkspaceFrontierError(
                "run workspace source-tree identity is invalid"
            )


@dataclass(frozen=True)
class RunWorkspaceRegularTreeIdentity:
    """Bounded identity of every directory and regular file in a private tree."""

    root_identity: tuple[int, int]
    tree_digest: str
    entry_count: int
    regular_file_size_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.root_identity) is not tuple
            or len(self.root_identity) != 2
            or any(type(value) is not int or value <= 0 for value in self.root_identity)
            or not isinstance(self.tree_digest, str)
            or not self.tree_digest.startswith("sha256:")
            or type(self.entry_count) is not int
            or self.entry_count <= 0
            or type(self.regular_file_size_bytes) is not int
            or self.regular_file_size_bytes < 0
        ):
            raise RunWorkspaceFrontierError(
                "run workspace regular-tree identity is invalid"
            )


@dataclass(frozen=True)
class RunWorkspaceFrontierIdentity:
    """Exact clean source/Git identity observed through a pinned workspace fd."""

    workspace_identity: tuple[int, int]
    branch: str
    commit_sha: str
    parent_commit_shas: tuple[str, ...]
    git_tree_sha: str
    source_tree_digest: str
    git_closure_digest: str
    source_entry_count: int
    source_size_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.workspace_identity) is not tuple
            or len(self.workspace_identity) != 2
            or any(
                type(value) is not int or value < 0 for value in self.workspace_identity
            )
            or not self.branch
            or _GIT_SHA_PATTERN.fullmatch(self.commit_sha) is None
            or _GIT_SHA_PATTERN.fullmatch(self.git_tree_sha) is None
            or any(
                _GIT_SHA_PATTERN.fullmatch(parent) is None
                for parent in self.parent_commit_shas
            )
            or len(self.parent_commit_shas) != len(set(self.parent_commit_shas))
            or not self.source_tree_digest.startswith("sha256:")
            or not self.git_closure_digest.startswith("sha256:")
            or type(self.source_entry_count) is not int
            or self.source_entry_count <= 0
            or type(self.source_size_bytes) is not int
            or self.source_size_bytes < 0
        ):
            raise RunWorkspaceFrontierError(
                "run workspace frontier identity is invalid"
            )


@dataclass(frozen=True)
class _RunWorkspaceCopyEntry:
    name: str
    relative_path: str
    file_type: str
    mode: int
    size_bytes: int
    source_metadata: tuple[int, ...]
    children: tuple["_RunWorkspaceCopyEntry", ...]


@dataclass(frozen=True)
class RunWorkspaceCopyPlan:
    """Bounded physical source and Git topology prepared before destination writes."""

    source_frontier: RunWorkspaceFrontierIdentity
    source_root_metadata: tuple[int, ...]
    entries: tuple[_RunWorkspaceCopyEntry, ...]
    directory_count: int
    regular_file_count: int
    physical_entry_count: int
    regular_file_size_bytes: int
    regular_file_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.source_frontier) is not RunWorkspaceFrontierIdentity
            or len(self.source_root_metadata) != 9
            or any(type(value) is not int for value in self.source_root_metadata)
            or self.source_root_metadata[:2] != self.source_frontier.workspace_identity
            or not self.entries
            or any(type(entry) is not _RunWorkspaceCopyEntry for entry in self.entries)
        ):
            raise RunWorkspaceFrontierError(
                "run workspace copy plan is incomplete or unbounded"
            )
        (
            observed_directory_count,
            observed_regular_file_count,
            observed_regular_file_sizes,
        ) = _workspace_copy_entry_summary(
            self.entries,
            PurePosixPath("."),
            self.source_root_metadata[0],
        )
        if (
            type(self.directory_count) is not int
            or self.directory_count != observed_directory_count + 1
            or type(self.regular_file_count) is not int
            or self.regular_file_count != observed_regular_file_count
            or type(self.physical_entry_count) is not int
            or self.physical_entry_count
            != self.directory_count + self.regular_file_count
            or self.regular_file_count != len(self.regular_file_sizes)
            or type(self.regular_file_size_bytes) is not int
            or self.regular_file_size_bytes != sum(self.regular_file_sizes)
            or self.regular_file_sizes != observed_regular_file_sizes
            or any(
                type(size_bytes) is not int or size_bytes < 0
                for size_bytes in self.regular_file_sizes
            )
        ):
            raise RunWorkspaceFrontierError(
                "run workspace copy plan is incomplete or unbounded"
            )

    def allocated_size_bytes(self, block_size_bytes: int) -> int:
        """Conservatively reserve one block per directory and rounded file bytes."""

        if (
            type(block_size_bytes) is not int
            or block_size_bytes <= 0
            or block_size_bytes & (block_size_bytes - 1) != 0
        ):
            raise RunWorkspaceFrontierError(
                "run workspace copy allocation block size is invalid"
            )
        return self.directory_count * block_size_bytes + sum(
            ((size_bytes + block_size_bytes - 1) // block_size_bytes) * block_size_bytes
            for size_bytes in self.regular_file_sizes
        )


@dataclass(frozen=True)
class RunWorkspaceSourceCopyPlan:
    """Bounded physical source-only topology with root Git explicitly excluded."""

    source_frontier: RunWorkspaceFrontierIdentity
    source_root_metadata: tuple[int, ...]
    entries: tuple[_RunWorkspaceCopyEntry, ...]
    directory_count: int
    regular_file_count: int
    physical_entry_count: int
    regular_file_size_bytes: int
    regular_file_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.source_frontier) is not RunWorkspaceFrontierIdentity
            or len(self.source_root_metadata) != 9
            or any(type(value) is not int for value in self.source_root_metadata)
            or self.source_root_metadata[:2] != self.source_frontier.workspace_identity
            or not self.entries
            or any(type(entry) is not _RunWorkspaceCopyEntry for entry in self.entries)
            or any(entry.name == ".git" for entry in self.entries)
            or type(self.directory_count) is not int
            or self.directory_count <= 0
            or type(self.regular_file_count) is not int
            or self.regular_file_count <= 0
            or type(self.physical_entry_count) is not int
            or self.physical_entry_count <= 0
            or type(self.regular_file_size_bytes) is not int
            or self.regular_file_size_bytes < 0
            or type(self.regular_file_sizes) is not tuple
            or any(
                type(size_bytes) is not int or size_bytes < 0
                for size_bytes in self.regular_file_sizes
            )
        ):
            raise RunWorkspaceFrontierError(
                "run workspace source copy plan is incomplete or unsafe"
            )
        directory_count, regular_file_count, regular_file_sizes = (
            _workspace_copy_entry_summary(
                self.entries,
                PurePosixPath("."),
                self.source_root_metadata[0],
            )
        )
        if (
            self.directory_count != directory_count + 1
            or self.regular_file_count != regular_file_count
            or self.physical_entry_count
            != self.directory_count + self.regular_file_count
            or self.regular_file_sizes != regular_file_sizes
            or self.regular_file_size_bytes != sum(regular_file_sizes)
        ):
            raise RunWorkspaceFrontierError(
                "run workspace source copy plan bounds are inconsistent"
            )

    def allocated_size_bytes(self, block_size_bytes: int) -> int:
        """Conservatively reserve one block per directory and rounded file bytes."""

        if (
            type(block_size_bytes) is not int
            or block_size_bytes <= 0
            or block_size_bytes & (block_size_bytes - 1) != 0
        ):
            raise RunWorkspaceFrontierError(
                "run workspace source copy allocation block size is invalid"
            )
        return self.directory_count * block_size_bytes + sum(
            ((size_bytes + block_size_bytes - 1) // block_size_bytes) * block_size_bytes
            for size_bytes in self.regular_file_sizes
        )


def _workspace_copy_entry_summary(
    entries: tuple[_RunWorkspaceCopyEntry, ...],
    parent_path: PurePosixPath,
    source_device: int,
) -> tuple[int, int, tuple[int, ...]]:
    if (
        type(entries) is not tuple
        or any(type(entry) is not _RunWorkspaceCopyEntry for entry in entries)
        or tuple(entry.name for entry in entries)
        != tuple(sorted({entry.name for entry in entries}))
    ):
        raise RunWorkspaceFrontierError(
            "run workspace copy plan tree is incomplete or noncanonical"
        )
    directory_count = 0
    regular_file_count = 0
    regular_file_sizes = []
    for entry in entries:
        _require_copy_component(entry.name)
        expected_path = (
            PurePosixPath(entry.name)
            if parent_path == PurePosixPath(".")
            else parent_path / entry.name
        )
        if (
            type(entry.relative_path) is not str
            or entry.relative_path != expected_path.as_posix()
            or len(entry.source_metadata) != 9
            or any(type(value) is not int for value in entry.source_metadata)
            or entry.source_metadata[0] != source_device
            or type(entry.mode) is not int
            or type(entry.size_bytes) is not int
            or entry.size_bytes < 0
        ):
            raise RunWorkspaceFrontierError("run workspace copy plan entry is invalid")
        if entry.file_type == "directory":
            if (
                entry.mode != 0o700
                or entry.size_bytes != 0
                or not stat.S_ISDIR(entry.source_metadata[2])
            ):
                raise RunWorkspaceFrontierError(
                    "run workspace copy plan directory is invalid"
                )
            child_directories, child_files, child_sizes = _workspace_copy_entry_summary(
                entry.children,
                expected_path,
                source_device,
            )
            directory_count += child_directories + 1
            regular_file_count += child_files
            regular_file_sizes.extend(child_sizes)
        elif (
            entry.file_type == "regular"
            and stat.S_ISREG(entry.source_metadata[2])
            and entry.source_metadata[3] == 1
            and entry.source_metadata[6] == entry.size_bytes
            and stat.S_IMODE(entry.source_metadata[2]) == entry.mode
            and entry.mode in {0o400, 0o444, 0o600, 0o644, 0o700, 0o755}
            and not entry.children
        ):
            regular_file_count += 1
            regular_file_sizes.append(entry.size_bytes)
        else:
            raise RunWorkspaceFrontierError("run workspace copy plan file is invalid")
    return (
        directory_count,
        regular_file_count,
        tuple(regular_file_sizes),
    )


@dataclass
class _SourceScanState:
    entry_limit: int
    size_limit: int
    entry_count: int = 0
    size_bytes: int = 0

    def reserve_entry(self) -> None:
        self.entry_count += 1
        if self.entry_count > self.entry_limit:
            raise RunWorkspaceFrontierError(
                "run workspace exceeds its configured entry limit"
            )

    def reserve_file(self, size_bytes: int) -> None:
        self.size_bytes += size_bytes
        if self.size_bytes > self.size_limit:
            raise RunWorkspaceFrontierError(
                "run workspace exceeds its configured byte limit"
            )


@dataclass
class _GitClosureState:
    entry_limit: int
    size_limit: int
    entry_count: int = 0
    compressed_size_bytes: int = 0
    decoded_size_bytes: int = 0
    regular_files: dict[str, tuple[str, int, int]] = field(default_factory=dict)

    def reserve_entries(self, count: int) -> None:
        self.entry_count += count
        if self.entry_count > self.entry_limit:
            raise RunWorkspaceFrontierError(
                "run workspace Git metadata exceeds its configured entry limit"
            )

    def reserve_regular_file(
        self,
        relative_path: str,
        payload: bytes,
        permissions: int,
    ) -> None:
        if relative_path in self.regular_files:
            raise RunWorkspaceFrontierError(
                "run workspace Git metadata path is duplicated"
            )
        self.compressed_size_bytes += len(payload)
        if self.compressed_size_bytes > self.size_limit:
            raise RunWorkspaceFrontierError(
                "run workspace Git metadata exceeds its configured byte limit"
            )
        self.regular_files[relative_path] = (
            tree_or_blob_digest(payload),
            permissions,
            len(payload),
        )

    @property
    def closure_digest(self) -> str:
        return tree_or_blob_digest(
            canonical_json_bytes(
                {path: value for path, value in sorted(self.regular_files.items())}
            )
        )

    def reserve_decoded_object(self, payload: bytes) -> None:
        self.decoded_size_bytes += len(payload)
        if self.decoded_size_bytes > self.size_limit:
            raise RunWorkspaceFrontierError(
                "run workspace decoded Git objects exceed their configured byte limit"
            )


@dataclass
class _WorkspaceCopyScanState:
    entry_limit: int
    size_limit_bytes: int
    directory_count: int = 0
    regular_file_count: int = 0
    regular_file_size_bytes: int = 0
    regular_file_sizes: list[int] = field(default_factory=list)
    inode_identities: set[tuple[int, int]] = field(default_factory=set)

    def reserve_directory(self, metadata: os.stat_result) -> None:
        self._reserve_identity(metadata)
        self.directory_count += 1
        self._require_entry_limit()

    def reserve_regular_file(self, metadata: os.stat_result) -> None:
        self._reserve_identity(metadata)
        self.regular_file_count += 1
        self._require_entry_limit()
        self.regular_file_size_bytes += metadata.st_size
        if self.regular_file_size_bytes > self.size_limit_bytes:
            raise RunWorkspaceFrontierError(
                "run workspace physical copy exceeds its configured byte limits"
            )
        self.regular_file_sizes.append(metadata.st_size)

    def _reserve_identity(self, metadata: os.stat_result) -> None:
        identity = _metadata_identity(metadata)
        if identity in self.inode_identities:
            raise RunWorkspaceFrontierError(
                "run workspace physical copy contains a repeated inode"
            )
        self.inode_identities.add(identity)

    def _require_entry_limit(self) -> None:
        if self.directory_count + self.regular_file_count > self.entry_limit:
            raise RunWorkspaceFrontierError(
                "run workspace physical copy exceeds its configured entry limits"
            )


@dataclass(frozen=True)
class _GitObject:
    kind: str
    payload: bytes


def inspect_run_workspace_source_tree(
    workspace_descriptor: int,
    *,
    maximum_entries: int,
    maximum_bytes: int,
) -> RunWorkspaceSourceTreeIdentity:
    """Digest one bounded source tree while excluding exactly the root `.git`."""

    (
        workspace_metadata,
        state,
        descriptors_by_path,
        _blob_ids,
        _directory_modes,
        _regular_file_permissions,
        _physical_metadata,
    ) = _scan_run_workspace_regular_tree(
        workspace_descriptor,
        maximum_entries=maximum_entries,
        maximum_bytes=maximum_bytes,
        root_git_required=True,
        empty_directories_allowed=False,
        allowed_file_permissions=frozenset({0o600, 0o644, 0o700, 0o755}),
    )
    return RunWorkspaceSourceTreeIdentity(
        workspace_identity=(workspace_metadata.st_dev, workspace_metadata.st_ino),
        source_tree_digest=source_tree_digest(
            {
                path: (descriptor.digest, descriptor.mode, descriptor.size)
                for path, descriptor in descriptors_by_path.items()
            }
        ),
        source_entry_count=state.entry_count,
        source_size_bytes=state.size_bytes,
    )


def inspect_detached_run_workspace_source_tree(
    workspace_descriptor: int,
    *,
    maximum_entries: int,
    maximum_bytes: int,
) -> RunWorkspaceSourceTreeIdentity:
    """Digest one owner-private source tree that contains no root Git metadata."""

    (
        workspace_metadata,
        state,
        descriptors_by_path,
        _blob_ids,
        _directory_modes,
        _regular_file_permissions,
        _physical_metadata,
    ) = _scan_run_workspace_regular_tree(
        workspace_descriptor,
        maximum_entries=maximum_entries,
        maximum_bytes=maximum_bytes,
        root_git_required=False,
        empty_directories_allowed=False,
        allowed_file_permissions=frozenset({0o600, 0o644, 0o700, 0o755}),
    )
    return RunWorkspaceSourceTreeIdentity(
        workspace_identity=(workspace_metadata.st_dev, workspace_metadata.st_ino),
        source_tree_digest=source_tree_digest(
            {
                path: (descriptor.digest, descriptor.mode, descriptor.size)
                for path, descriptor in descriptors_by_path.items()
            }
        ),
        source_entry_count=state.entry_count,
        source_size_bytes=state.size_bytes,
    )


def inspect_run_workspace_regular_tree(
    directory_descriptor: int,
    *,
    maximum_entries: int,
    maximum_bytes: int,
) -> RunWorkspaceRegularTreeIdentity:
    """Digest a bounded private regular tree without a Git-root special case."""

    (
        root_metadata,
        state,
        descriptors_by_path,
        _blob_ids,
        directory_modes,
        regular_file_permissions,
        physical_metadata,
    ) = _scan_run_workspace_regular_tree(
        directory_descriptor,
        maximum_entries=maximum_entries,
        maximum_bytes=maximum_bytes,
        root_git_required=False,
        empty_directories_allowed=True,
        allowed_file_permissions=frozenset({0o400, 0o444, 0o600, 0o644, 0o700, 0o755}),
    )
    return _regular_tree_identity(
        root_metadata,
        state,
        descriptors_by_path,
        directory_modes,
        regular_file_permissions,
        physical_metadata,
    )


def inspect_run_workspace_source_regular_tree(
    workspace_descriptor: int,
    *,
    maximum_entries: int,
    maximum_bytes: int,
) -> RunWorkspaceRegularTreeIdentity:
    """Digest exact physical source authority while excluding root `.git`."""

    (
        root_metadata,
        state,
        descriptors_by_path,
        _blob_ids,
        directory_modes,
        regular_file_permissions,
        physical_metadata,
    ) = _scan_run_workspace_regular_tree(
        workspace_descriptor,
        maximum_entries=maximum_entries,
        maximum_bytes=maximum_bytes,
        root_git_required=True,
        empty_directories_allowed=True,
        allowed_file_permissions=frozenset({0o600, 0o644, 0o700, 0o755}),
    )
    return _regular_tree_identity(
        root_metadata,
        state,
        descriptors_by_path,
        directory_modes,
        regular_file_permissions,
        physical_metadata,
    )


def _regular_tree_identity(
    root_metadata: os.stat_result,
    state: _SourceScanState,
    descriptors_by_path: dict[str, SourceFileDescriptor],
    directory_modes: dict[str, int],
    regular_file_permissions: dict[str, int],
    physical_metadata: dict[str, tuple[int, ...]],
) -> RunWorkspaceRegularTreeIdentity:
    if set(directory_modes) | set(regular_file_permissions) != set(physical_metadata):
        raise RunWorkspaceFrontierError(
            "run workspace physical-tree metadata closure is inconsistent"
        )
    return RunWorkspaceRegularTreeIdentity(
        root_identity=(root_metadata.st_dev, root_metadata.st_ino),
        tree_digest=tree_or_blob_digest(
            canonical_json_bytes(
                {
                    "root": _copy_metadata_observation(root_metadata),
                    "directories": {
                        path: physical_metadata[path]
                        for path in sorted(directory_modes)
                    },
                    "regular_files": {
                        path: {
                            "content_digest": descriptor.digest,
                            "metadata": physical_metadata[path],
                        }
                        for path, descriptor in sorted(descriptors_by_path.items())
                    },
                }
            )
        ),
        entry_count=state.entry_count,
        regular_file_size_bytes=state.size_bytes,
    )


def _scan_run_workspace_regular_tree(
    workspace_descriptor: int,
    *,
    maximum_entries: int,
    maximum_bytes: int,
    root_git_required: bool,
    empty_directories_allowed: bool,
    allowed_file_permissions: frozenset[int],
) -> tuple[
    os.stat_result,
    _SourceScanState,
    dict[str, SourceFileDescriptor],
    dict[str, str],
    dict[str, int],
    dict[str, int],
    dict[str, tuple[int, ...]],
]:
    if (
        type(workspace_descriptor) is not int
        or workspace_descriptor < 0
        or type(maximum_entries) is not int
        or maximum_entries <= 0
        or type(maximum_bytes) is not int
        or maximum_bytes <= 0
    ):
        raise RunWorkspaceFrontierError(
            "run workspace source-tree inspection inputs are invalid"
        )
    workspace_metadata = os.fstat(workspace_descriptor)
    if (
        not stat.S_ISDIR(workspace_metadata.st_mode)
        or workspace_metadata.st_uid != os.geteuid()
        or workspace_metadata.st_gid != os.getegid()
        or stat.S_IMODE(workspace_metadata.st_mode) != 0o700
        or workspace_metadata.st_dev <= 0
        or workspace_metadata.st_ino <= 0
    ):
        raise RunWorkspaceFrontierError(
            "run workspace source-tree descriptor is not owner-private"
        )
    state = _SourceScanState(
        entry_limit=maximum_entries,
        size_limit=maximum_bytes,
    )
    (
        descriptors_by_path,
        blob_ids,
        directory_modes,
        regular_file_permissions,
        physical_metadata,
    ) = _scan_source_directory(
        workspace_descriptor,
        PurePosixPath("."),
        state,
        root=root_git_required,
        empty_directories_allowed=empty_directories_allowed,
        allowed_file_permissions=allowed_file_permissions,
    )
    if not descriptors_by_path:
        raise RunWorkspaceFrontierError("run workspace source tree is empty")
    rebound = os.fstat(workspace_descriptor)
    if _metadata_observation(rebound) != _metadata_observation(workspace_metadata):
        raise RunWorkspaceFrontierError(
            "run workspace source-tree descriptor changed during inspection"
        )
    return (
        workspace_metadata,
        state,
        descriptors_by_path,
        blob_ids,
        directory_modes,
        regular_file_permissions,
        physical_metadata,
    )


def inspect_run_workspace_frontier(
    workspace_descriptor: int,
    *,
    settings: LaunchSettings,
    expected_commit_sha: str | None,
) -> RunWorkspaceFrontierIdentity:
    """Reconcile clean working files, index, branch ref, and commit tree."""
    if type(settings) is not LaunchSettings:
        raise RunWorkspaceFrontierError(
            "run workspace inspection requires exact launch settings"
        )
    return inspect_run_workspace_frontier_with_limits(
        workspace_descriptor,
        workspace_git_branch=settings.workspace_git_branch,
        maximum_source_entries=settings.run_workspace_entry_limit,
        maximum_source_bytes=settings.run_workspace_size_bytes,
        maximum_git_entries=settings.run_workspace_git_entry_limit,
        maximum_git_bytes=settings.run_workspace_git_metadata_size_bytes,
        expected_commit_sha=expected_commit_sha,
    )


def inspect_run_workspace_frontier_with_limits(
    workspace_descriptor: int,
    *,
    workspace_git_branch: str,
    maximum_source_entries: int,
    maximum_source_bytes: int,
    maximum_git_entries: int,
    maximum_git_bytes: int,
    expected_commit_sha: str | None,
) -> RunWorkspaceFrontierIdentity:
    """Reconcile one clean Git frontier from explicit bounded authority."""

    if not isinstance(workspace_git_branch, str) or any(
        type(value) is not int or value <= 0
        for value in (
            maximum_source_entries,
            maximum_source_bytes,
            maximum_git_entries,
            maximum_git_bytes,
        )
    ):
        raise RunWorkspaceFrontierError(
            "run workspace frontier limits or branch are invalid"
        )
    require_git_ref_name(
        f"refs/heads/{workspace_git_branch}",
        "run workspace Git branch",
        qualified=True,
        error_type=RunWorkspaceFrontierError,
    )
    if expected_commit_sha is not None and (
        _GIT_SHA_PATTERN.fullmatch(expected_commit_sha) is None
    ):
        raise RunWorkspaceFrontierError("run workspace expected commit is invalid")
    workspace_metadata = os.fstat(workspace_descriptor)
    if (
        not stat.S_ISDIR(workspace_metadata.st_mode)
        or workspace_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(workspace_metadata.st_mode) != 0o700
    ):
        raise RunWorkspaceFrontierError("run workspace descriptor is not owner-private")
    state = _SourceScanState(
        entry_limit=maximum_source_entries,
        size_limit=maximum_source_bytes,
    )
    (
        descriptors_by_path,
        blob_ids,
        _directory_modes,
        _source_file_permissions,
        _physical_metadata,
    ) = _scan_source_directory(
        workspace_descriptor,
        PurePosixPath("."),
        state,
        root=True,
        empty_directories_allowed=False,
        allowed_file_permissions=frozenset({0o600, 0o644, 0o700, 0o755}),
    )
    if not descriptors_by_path:
        raise RunWorkspaceFrontierError("run workspace source tree is empty")
    git_tree_sha = git_tree_shas(
        {
            path: (blob_ids[path], descriptor.mode)
            for path, descriptor in descriptors_by_path.items()
        }
    )[""]
    with ExitStack() as descriptors:
        git_descriptor = _open_directory(
            workspace_descriptor,
            ".git",
            descriptors,
            "run workspace Git root",
        )
        git_state = _GitClosureState(
            entry_limit=maximum_git_entries,
            size_limit=maximum_git_bytes,
        )
        git_objects, branch_reference = _read_exact_git_closure(
            git_descriptor,
            workspace_git_branch=workspace_git_branch,
            maximum_git_bytes=maximum_git_bytes,
            state=git_state,
            descriptors=descriptors,
        )
        expected_head = f"ref: refs/heads/{workspace_git_branch}\n".encode("utf-8")
        head = _read_regular_file(
            git_descriptor,
            "HEAD",
            maximum_bytes=len(expected_head),
            allowed_modes={0o600, 0o644},
            name="run workspace Git HEAD",
        )
        git_state.reserve_regular_file(
            "HEAD",
            head,
            _regular_file_permissions(git_descriptor, "HEAD"),
        )
        if head != expected_head:
            raise RunWorkspaceFrontierError(
                "run workspace Git HEAD names another branch"
            )
        expected_config = StarterWorkspaceBuilder._git_config_bytes()
        config = _read_regular_file(
            git_descriptor,
            "config",
            maximum_bytes=len(expected_config),
            allowed_modes={0o600, 0o644},
            name="run workspace Git config",
        )
        git_state.reserve_regular_file(
            "config",
            config,
            _regular_file_permissions(git_descriptor, "config"),
        )
        if config != expected_config:
            raise RunWorkspaceFrontierError(
                "run workspace Git config changed after launch"
            )
        commit_sha = branch_reference.removesuffix(b"\n").decode("ascii")
        if (
            branch_reference != f"{commit_sha}\n".encode("ascii")
            or _GIT_SHA_PATTERN.fullmatch(commit_sha) is None
            or (expected_commit_sha is not None and commit_sha != expected_commit_sha)
        ):
            raise RunWorkspaceFrontierError(
                "run workspace branch head differs from its checkpoint"
            )
        commit_payload = _require_git_object(
            git_objects,
            commit_sha,
            expected_kind="commit",
        )
        commit_tree_sha, parent_commit_shas = _parse_commit(commit_payload)
        if commit_tree_sha != git_tree_sha:
            raise RunWorkspaceFrontierError(
                "run workspace files differ from the checkpointed commit tree"
            )
        index = _read_regular_file(
            git_descriptor,
            "index",
            maximum_bytes=maximum_git_bytes,
            allowed_modes={0o600, 0o644},
            name="run workspace Git index",
        )
        git_state.reserve_regular_file(
            "index",
            index,
            _regular_file_permissions(git_descriptor, "index"),
        )
        _require_index_matches_source(
            index,
            descriptors_by_path,
            blob_ids,
        )
        _require_reachable_git_object_closure(
            git_objects,
            commit_sha,
        )
    rebound = os.fstat(workspace_descriptor)
    if _metadata_observation(rebound) != _metadata_observation(workspace_metadata):
        raise RunWorkspaceFrontierError(
            "run workspace descriptor changed during reconciliation"
        )
    return RunWorkspaceFrontierIdentity(
        workspace_identity=(workspace_metadata.st_dev, workspace_metadata.st_ino),
        branch=workspace_git_branch,
        commit_sha=commit_sha,
        parent_commit_shas=parent_commit_shas,
        git_tree_sha=git_tree_sha,
        source_tree_digest=source_tree_digest(
            {
                path: (descriptor.digest, descriptor.mode, descriptor.size)
                for path, descriptor in descriptors_by_path.items()
            }
        ),
        git_closure_digest=git_state.closure_digest,
        source_entry_count=state.entry_count,
        source_size_bytes=state.size_bytes,
    )


def plan_run_workspace_frontier_copy(
    workspace_descriptor: int,
    *,
    settings: LaunchSettings,
    expected: RunWorkspaceFrontierIdentity,
) -> RunWorkspaceCopyPlan:
    """Prove and inventory the complete source and Git tree before any copy."""

    if (
        type(settings) is not LaunchSettings
        or type(expected) is not RunWorkspaceFrontierIdentity
    ):
        raise RunWorkspaceFrontierError(
            "run workspace copy plan requires exact settings and frontier"
        )
    observed_before = inspect_run_workspace_frontier(
        workspace_descriptor,
        settings=settings,
        expected_commit_sha=expected.commit_sha,
    )
    if observed_before != expected:
        raise RunWorkspaceFrontierError(
            "run workspace copy source differs from its durable frontier"
        )
    root_metadata = os.fstat(workspace_descriptor)
    state = _WorkspaceCopyScanState(
        entry_limit=(
            settings.run_workspace_entry_limit
            + settings.run_workspace_git_entry_limit
            + 2
        ),
        size_limit_bytes=(
            settings.run_workspace_size_bytes
            + settings.run_workspace_git_metadata_size_bytes
        ),
    )
    state.reserve_directory(root_metadata)
    entries = _plan_workspace_copy_directory(
        workspace_descriptor,
        PurePosixPath("."),
        root_metadata.st_dev,
        state,
    )
    observed_after = inspect_run_workspace_frontier(
        workspace_descriptor,
        settings=settings,
        expected_commit_sha=expected.commit_sha,
    )
    if observed_after != expected or _copy_metadata_observation(
        os.fstat(workspace_descriptor)
    ) != _copy_metadata_observation(root_metadata):
        raise RunWorkspaceFrontierError(
            "run workspace copy source changed during planning"
        )
    return RunWorkspaceCopyPlan(
        source_frontier=expected,
        source_root_metadata=_copy_metadata_observation(root_metadata),
        entries=entries,
        directory_count=state.directory_count,
        regular_file_count=state.regular_file_count,
        physical_entry_count=state.directory_count + state.regular_file_count,
        regular_file_size_bytes=state.regular_file_size_bytes,
        regular_file_sizes=tuple(state.regular_file_sizes),
    )


def plan_run_workspace_source_copy(
    workspace_descriptor: int,
    *,
    expected: RunWorkspaceFrontierIdentity,
    maximum_source_entries: int,
    maximum_source_bytes: int,
    maximum_git_entries: int,
    maximum_git_bytes: int,
) -> RunWorkspaceSourceCopyPlan:
    """Prove and inventory the source tree while excluding exactly root `.git`."""

    if type(expected) is not RunWorkspaceFrontierIdentity or any(
        type(value) is not int or value <= 0
        for value in (
            maximum_source_entries,
            maximum_source_bytes,
            maximum_git_entries,
            maximum_git_bytes,
        )
    ):
        raise RunWorkspaceFrontierError("run workspace source copy limits are invalid")
    observed_before = inspect_run_workspace_frontier_with_limits(
        workspace_descriptor,
        workspace_git_branch=expected.branch,
        maximum_source_entries=maximum_source_entries,
        maximum_source_bytes=maximum_source_bytes,
        maximum_git_entries=maximum_git_entries,
        maximum_git_bytes=maximum_git_bytes,
        expected_commit_sha=expected.commit_sha,
    )
    if observed_before != expected:
        raise RunWorkspaceFrontierError(
            "run workspace source copy differs from its durable frontier"
        )
    root_metadata = os.fstat(workspace_descriptor)
    state = _WorkspaceCopyScanState(
        entry_limit=maximum_source_entries + maximum_git_entries + 2,
        size_limit_bytes=maximum_source_bytes + maximum_git_bytes,
    )
    state.reserve_directory(root_metadata)
    complete_entries = _plan_workspace_copy_directory(
        workspace_descriptor,
        PurePosixPath("."),
        root_metadata.st_dev,
        state,
    )
    if sum(entry.name == ".git" for entry in complete_entries) != 1:
        raise RunWorkspaceFrontierError(
            "run workspace source copy lacks exact root Git exclusion"
        )
    entries = tuple(entry for entry in complete_entries if entry.name != ".git")
    directory_count, regular_file_count, regular_file_sizes = (
        _workspace_copy_entry_summary(
            entries,
            PurePosixPath("."),
            root_metadata.st_dev,
        )
    )
    observed_after = inspect_run_workspace_frontier_with_limits(
        workspace_descriptor,
        workspace_git_branch=expected.branch,
        maximum_source_entries=maximum_source_entries,
        maximum_source_bytes=maximum_source_bytes,
        maximum_git_entries=maximum_git_entries,
        maximum_git_bytes=maximum_git_bytes,
        expected_commit_sha=expected.commit_sha,
    )
    if observed_after != expected or _copy_metadata_observation(
        os.fstat(workspace_descriptor)
    ) != _copy_metadata_observation(root_metadata):
        raise RunWorkspaceFrontierError(
            "run workspace source changed during copy planning"
        )
    return RunWorkspaceSourceCopyPlan(
        source_frontier=expected,
        source_root_metadata=_copy_metadata_observation(root_metadata),
        entries=entries,
        directory_count=directory_count + 1,
        regular_file_count=regular_file_count,
        physical_entry_count=directory_count + regular_file_count + 1,
        regular_file_size_bytes=sum(regular_file_sizes),
        regular_file_sizes=regular_file_sizes,
    )


def copy_run_workspace_frontier(
    workspace_descriptor: int,
    destination_descriptor: int,
    *,
    settings: LaunchSettings,
    plan: RunWorkspaceCopyPlan,
) -> RunWorkspaceFrontierIdentity:
    """Copy one planned frontier and prove the destination equals its source."""

    if type(settings) is not LaunchSettings or type(plan) is not RunWorkspaceCopyPlan:
        raise RunWorkspaceFrontierError(
            "run workspace copy requires an exact physical plan"
        )
    observed_before = inspect_run_workspace_frontier(
        workspace_descriptor,
        settings=settings,
        expected_commit_sha=plan.source_frontier.commit_sha,
    )
    source_metadata = os.fstat(workspace_descriptor)
    destination_metadata = os.fstat(destination_descriptor)
    if (
        observed_before != plan.source_frontier
        or _copy_metadata_observation(source_metadata) != plan.source_root_metadata
        or not stat.S_ISDIR(destination_metadata.st_mode)
        or destination_metadata.st_uid != os.geteuid()
        or destination_metadata.st_gid != os.getegid()
        or stat.S_IMODE(destination_metadata.st_mode) != 0o700
        or tuple(os.listdir(destination_descriptor))
    ):
        raise RunWorkspaceFrontierError(
            "run workspace copy endpoints differ from the admitted plan"
        )
    _copy_workspace_directory_entries(
        workspace_descriptor,
        destination_descriptor,
        plan.entries,
        excluded_source_names=frozenset(),
    )
    os.fsync(destination_descriptor)
    source_physical_before = _require_workspace_copy_tree(
        workspace_descriptor,
        plan,
        source=True,
        excluded_root_names=frozenset(),
    )
    destination_physical_before = _require_workspace_copy_tree(
        destination_descriptor,
        plan,
        source=False,
        excluded_root_names=frozenset(),
    )
    observed_after = inspect_run_workspace_frontier(
        workspace_descriptor,
        settings=settings,
        expected_commit_sha=plan.source_frontier.commit_sha,
    )
    destination_frontier = inspect_run_workspace_frontier(
        destination_descriptor,
        settings=settings,
        expected_commit_sha=plan.source_frontier.commit_sha,
    )
    expected_destination = replace(
        plan.source_frontier,
        workspace_identity=destination_frontier.workspace_identity,
    )
    source_physical_after = _require_workspace_copy_tree(
        workspace_descriptor,
        plan,
        source=True,
        excluded_root_names=frozenset(),
    )
    destination_physical_after = _require_workspace_copy_tree(
        destination_descriptor,
        plan,
        source=False,
        excluded_root_names=frozenset(),
    )
    if (
        observed_after != plan.source_frontier
        or destination_frontier != expected_destination
        or source_physical_after != source_physical_before
        or destination_physical_after != destination_physical_before
        or _copy_metadata_observation(os.fstat(workspace_descriptor))
        != plan.source_root_metadata
        or tuple(sorted(os.listdir(destination_descriptor)))
        != tuple(entry.name for entry in plan.entries)
    ):
        raise RunWorkspaceFrontierError(
            "run workspace copy differs from its stable source frontier"
        )
    return destination_frontier


def copy_run_workspace_source_tree(
    workspace_descriptor: int,
    destination_descriptor: int,
    *,
    plan: RunWorkspaceSourceCopyPlan,
    maximum_source_entries: int,
    maximum_source_bytes: int,
    maximum_git_entries: int,
    maximum_git_bytes: int,
) -> RunWorkspaceSourceTreeIdentity:
    """Copy one planned source tree into an empty owner-private Git-free root."""

    if type(plan) is not RunWorkspaceSourceCopyPlan:
        raise RunWorkspaceFrontierError(
            "run workspace source copy requires its exact physical plan"
        )
    observed_before = inspect_run_workspace_frontier_with_limits(
        workspace_descriptor,
        workspace_git_branch=plan.source_frontier.branch,
        maximum_source_entries=maximum_source_entries,
        maximum_source_bytes=maximum_source_bytes,
        maximum_git_entries=maximum_git_entries,
        maximum_git_bytes=maximum_git_bytes,
        expected_commit_sha=plan.source_frontier.commit_sha,
    )
    source_metadata = os.fstat(workspace_descriptor)
    destination_metadata = os.fstat(destination_descriptor)
    if (
        observed_before != plan.source_frontier
        or _copy_metadata_observation(source_metadata) != plan.source_root_metadata
        or not stat.S_ISDIR(destination_metadata.st_mode)
        or destination_metadata.st_uid != os.geteuid()
        or destination_metadata.st_gid != os.getegid()
        or stat.S_IMODE(destination_metadata.st_mode) != 0o700
        or tuple(os.listdir(destination_descriptor))
    ):
        raise RunWorkspaceFrontierError(
            "run workspace source copy endpoints differ from the admitted plan"
        )
    _copy_workspace_directory_entries(
        workspace_descriptor,
        destination_descriptor,
        plan.entries,
        excluded_source_names=frozenset({".git"}),
    )
    os.fsync(destination_descriptor)
    source_physical_before = _require_workspace_copy_tree(
        workspace_descriptor,
        plan,
        source=True,
        excluded_root_names=frozenset({".git"}),
    )
    destination_physical_before = _require_workspace_copy_tree(
        destination_descriptor,
        plan,
        source=False,
        excluded_root_names=frozenset(),
    )
    destination_source_before = inspect_detached_run_workspace_source_tree(
        destination_descriptor,
        maximum_entries=maximum_source_entries,
        maximum_bytes=maximum_source_bytes,
    )
    observed_after = inspect_run_workspace_frontier_with_limits(
        workspace_descriptor,
        workspace_git_branch=plan.source_frontier.branch,
        maximum_source_entries=maximum_source_entries,
        maximum_source_bytes=maximum_source_bytes,
        maximum_git_entries=maximum_git_entries,
        maximum_git_bytes=maximum_git_bytes,
        expected_commit_sha=plan.source_frontier.commit_sha,
    )
    source_physical_after = _require_workspace_copy_tree(
        workspace_descriptor,
        plan,
        source=True,
        excluded_root_names=frozenset({".git"}),
    )
    destination_physical_after = _require_workspace_copy_tree(
        destination_descriptor,
        plan,
        source=False,
        excluded_root_names=frozenset(),
    )
    destination_source_after = inspect_detached_run_workspace_source_tree(
        destination_descriptor,
        maximum_entries=maximum_source_entries,
        maximum_bytes=maximum_source_bytes,
    )
    if (
        observed_after != plan.source_frontier
        or source_physical_after != source_physical_before
        or destination_physical_after != destination_physical_before
        or destination_source_after != destination_source_before
        or destination_source_after.source_tree_digest
        != plan.source_frontier.source_tree_digest
        or destination_source_after.source_entry_count
        != plan.source_frontier.source_entry_count
        or destination_source_after.source_size_bytes
        != plan.source_frontier.source_size_bytes
        or tuple(sorted(os.listdir(destination_descriptor)))
        != tuple(entry.name for entry in plan.entries)
    ):
        raise RunWorkspaceFrontierError(
            "run workspace source copy differs from its stable source frontier"
        )
    return destination_source_after


def replace_run_workspace_source_tree(
    workspace_descriptor: int,
    successor_descriptor: int,
    *,
    predecessor: RunWorkspaceFrontierIdentity,
    maximum_source_entries: int,
    maximum_source_bytes: int,
    maximum_git_entries: int,
    maximum_git_bytes: int,
) -> RunWorkspaceSourceTreeIdentity:
    """Replace only trusted source files from one sanitized private successor."""

    predecessor_plan = plan_run_workspace_source_copy(
        workspace_descriptor,
        expected=predecessor,
        maximum_source_entries=maximum_source_entries,
        maximum_source_bytes=maximum_source_bytes,
        maximum_git_entries=maximum_git_entries,
        maximum_git_bytes=maximum_git_bytes,
    )
    successor_before = inspect_detached_run_workspace_source_tree(
        successor_descriptor,
        maximum_entries=maximum_source_entries,
        maximum_bytes=maximum_source_bytes,
    )
    successor_root = os.fstat(successor_descriptor)
    state = _WorkspaceCopyScanState(
        entry_limit=maximum_source_entries + 1,
        size_limit_bytes=maximum_source_bytes,
    )
    state.reserve_directory(successor_root)
    successor_entries = _plan_workspace_copy_directory(
        successor_descriptor,
        PurePosixPath("."),
        successor_root.st_dev,
        state,
    )
    successor_after_plan = inspect_detached_run_workspace_source_tree(
        successor_descriptor,
        maximum_entries=maximum_source_entries,
        maximum_bytes=maximum_source_bytes,
    )
    if successor_after_plan != successor_before or _copy_metadata_observation(
        os.fstat(successor_descriptor)
    ) != _copy_metadata_observation(successor_root):
        raise RunWorkspaceFrontierError(
            "sanitized run workspace successor changed during planning"
        )
    observed_predecessor = inspect_run_workspace_frontier_with_limits(
        workspace_descriptor,
        workspace_git_branch=predecessor.branch,
        maximum_source_entries=maximum_source_entries,
        maximum_source_bytes=maximum_source_bytes,
        maximum_git_entries=maximum_git_entries,
        maximum_git_bytes=maximum_git_bytes,
        expected_commit_sha=predecessor.commit_sha,
    )
    if observed_predecessor != predecessor:
        raise RunWorkspaceFrontierError(
            "trusted run workspace changed before successor replacement"
        )
    _remove_workspace_directory_entries(
        workspace_descriptor,
        predecessor_plan.entries,
        excluded_names=frozenset({".git"}),
    )
    _copy_workspace_directory_entries(
        successor_descriptor,
        workspace_descriptor,
        successor_entries,
        excluded_source_names=frozenset(),
    )
    os.fsync(workspace_descriptor)
    trusted_successor = inspect_run_workspace_source_tree(
        workspace_descriptor,
        maximum_entries=maximum_source_entries,
        maximum_bytes=maximum_source_bytes,
    )
    successor_after_copy = inspect_detached_run_workspace_source_tree(
        successor_descriptor,
        maximum_entries=maximum_source_entries,
        maximum_bytes=maximum_source_bytes,
    )
    if (
        successor_after_copy != successor_before
        or trusted_successor.source_tree_digest != successor_before.source_tree_digest
        or trusted_successor.source_entry_count != successor_before.source_entry_count
        or trusted_successor.source_size_bytes != successor_before.source_size_bytes
        or set(os.listdir(workspace_descriptor))
        != {".git", *(entry.name for entry in successor_entries)}
    ):
        raise RunWorkspaceFrontierError(
            "trusted run workspace differs from its sanitized successor"
        )
    return trusted_successor


def _plan_workspace_copy_directory(
    directory_descriptor: int,
    relative_root: PurePosixPath,
    source_device: int,
    state: _WorkspaceCopyScanState,
) -> tuple[_RunWorkspaceCopyEntry, ...]:
    observed_entries = []
    with os.scandir(directory_descriptor) as iterator:
        for entry in iterator:
            _require_copy_component(entry.name)
            observed_entries.append(
                (
                    entry.name,
                    entry.stat(follow_symlinks=False),
                )
            )
    entries = []
    for name, expected in sorted(observed_entries, key=lambda item: item[0]):
        relative_path = (
            PurePosixPath(name)
            if relative_root == PurePosixPath(".")
            else relative_root / name
        )
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _copy_metadata_observation(current) != _copy_metadata_observation(expected)
            or expected.st_dev != source_device
            or expected.st_uid != os.geteuid()
            or expected.st_mode & (stat.S_ISUID | stat.S_ISGID)
        ):
            raise RunWorkspaceFrontierError(
                "run workspace copy entry changed or crossed a filesystem"
            )
        mode = stat.S_IMODE(expected.st_mode)
        if stat.S_ISDIR(expected.st_mode):
            if expected.st_mode & 0o022:
                raise RunWorkspaceFrontierError(
                    "run workspace copy directory is unsafe"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            with ExitStack() as descriptors:
                descriptors.callback(os.close, descriptor)
                if _copy_metadata_observation(os.fstat(descriptor)) != (
                    _copy_metadata_observation(expected)
                ):
                    raise RunWorkspaceFrontierError(
                        "run workspace copy directory changed while opening"
                    )
                state.reserve_directory(expected)
                children = _plan_workspace_copy_directory(
                    descriptor,
                    relative_path,
                    source_device,
                    state,
                )
                if _copy_metadata_observation(os.fstat(descriptor)) != (
                    _copy_metadata_observation(expected)
                ):
                    raise RunWorkspaceFrontierError(
                        "run workspace copy directory changed while scanning"
                    )
            entries.append(
                _RunWorkspaceCopyEntry(
                    name=name,
                    relative_path=relative_path.as_posix(),
                    file_type="directory",
                    mode=0o700,
                    size_bytes=0,
                    source_metadata=_copy_metadata_observation(expected),
                    children=children,
                )
            )
            continue
        if (
            not stat.S_ISREG(expected.st_mode)
            or expected.st_nlink != 1
            or mode not in {0o400, 0o444, 0o600, 0o644, 0o700, 0o755}
        ):
            raise RunWorkspaceFrontierError("run workspace copy file is unsafe")
        state.reserve_regular_file(expected)
        entries.append(
            _RunWorkspaceCopyEntry(
                name=name,
                relative_path=relative_path.as_posix(),
                file_type="regular",
                mode=mode,
                size_bytes=expected.st_size,
                source_metadata=_copy_metadata_observation(expected),
                children=(),
            )
        )
    rebound_names = tuple(sorted(os.listdir(directory_descriptor)))
    if rebound_names != tuple(entry.name for entry in entries):
        raise RunWorkspaceFrontierError(
            "run workspace copy directory changed while listing"
        )
    return tuple(entries)


def _remove_workspace_directory_entries(
    directory_descriptor: int,
    entries: tuple[_RunWorkspaceCopyEntry, ...],
    *,
    excluded_names: frozenset[str],
) -> None:
    observed_names = set(os.listdir(directory_descriptor))
    if observed_names & excluded_names != excluded_names or tuple(
        sorted(observed_names - excluded_names)
    ) != tuple(entry.name for entry in entries):
        raise RunWorkspaceFrontierError(
            "run workspace source topology changed before replacement"
        )
    for entry in entries:
        expected = os.stat(
            entry.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _copy_metadata_observation(expected) != entry.source_metadata:
            raise RunWorkspaceFrontierError(
                "run workspace source entry changed before replacement"
            )
        if entry.file_type == "directory":
            child_descriptor = os.open(
                entry.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            with ExitStack() as descriptors:
                descriptors.callback(os.close, child_descriptor)
                _remove_workspace_directory_entries(
                    child_descriptor,
                    entry.children,
                    excluded_names=frozenset(),
                )
                if os.fstat(child_descriptor).st_ino != expected.st_ino:
                    raise RunWorkspaceFrontierError(
                        "run workspace source directory changed during replacement"
                    )
            os.rmdir(entry.name, dir_fd=directory_descriptor)
        elif entry.file_type == "regular":
            os.unlink(entry.name, dir_fd=directory_descriptor)
        else:
            raise RunWorkspaceFrontierError(
                "run workspace replacement plan contains an unknown entry type"
            )
    if set(os.listdir(directory_descriptor)) != excluded_names:
        raise RunWorkspaceFrontierError(
            "run workspace source removal left unexpected authority"
        )


def _copy_workspace_directory_entries(
    source_descriptor: int,
    destination_descriptor: int,
    entries: tuple[_RunWorkspaceCopyEntry, ...],
    *,
    excluded_source_names: frozenset[str],
) -> None:
    source_names = set(os.listdir(source_descriptor))
    if (
        type(excluded_source_names) is not frozenset
        or source_names & excluded_source_names != excluded_source_names
        or tuple(sorted(source_names - excluded_source_names))
        != tuple(entry.name for entry in entries)
    ):
        raise RunWorkspaceFrontierError(
            "run workspace copy source topology changed before copying"
        )
    for entry in entries:
        expected = os.stat(
            entry.name,
            dir_fd=source_descriptor,
            follow_symlinks=False,
        )
        if _copy_metadata_observation(expected) != entry.source_metadata:
            raise RunWorkspaceFrontierError(
                "run workspace copy source entry changed before copying"
            )
        if entry.file_type == "directory":
            os.mkdir(entry.name, mode=0o700, dir_fd=destination_descriptor)
            source_child = os.open(
                entry.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=source_descriptor,
            )
            destination_child = os.open(
                entry.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=destination_descriptor,
            )
            with ExitStack() as descriptors:
                descriptors.callback(os.close, source_child)
                descriptors.callback(os.close, destination_child)
                if _copy_metadata_observation(os.fstat(source_child)) != (
                    entry.source_metadata
                ):
                    raise RunWorkspaceFrontierError(
                        "run workspace copy source directory changed while opening"
                    )
                _copy_workspace_directory_entries(
                    source_child,
                    destination_child,
                    entry.children,
                    excluded_source_names=frozenset(),
                )
                os.fchmod(destination_child, entry.mode)
                os.fsync(destination_child)
                _require_copied_directory(destination_child, entry.mode)
                if _copy_metadata_observation(os.fstat(source_child)) != (
                    entry.source_metadata
                ):
                    raise RunWorkspaceFrontierError(
                        "run workspace copy source directory changed while copying"
                    )
        elif entry.file_type == "regular":
            _copy_workspace_regular_file(
                source_descriptor,
                destination_descriptor,
                entry,
            )
        else:
            raise RunWorkspaceFrontierError(
                "run workspace copy plan contains an unknown entry type"
            )
        rebound = os.stat(
            entry.name,
            dir_fd=source_descriptor,
            follow_symlinks=False,
        )
        if _copy_metadata_observation(rebound) != entry.source_metadata:
            raise RunWorkspaceFrontierError(
                "run workspace copy source entry changed during copying"
            )
    rebound_names = set(os.listdir(source_descriptor))
    if rebound_names & excluded_source_names != excluded_source_names or tuple(
        sorted(rebound_names - excluded_source_names)
    ) != tuple(entry.name for entry in entries):
        raise RunWorkspaceFrontierError(
            "run workspace copy source topology changed during copying"
        )


def _copy_workspace_regular_file(
    source_parent_descriptor: int,
    destination_parent_descriptor: int,
    entry: _RunWorkspaceCopyEntry,
) -> None:
    source_descriptor = os.open(
        entry.name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=source_parent_descriptor,
    )
    destination_descriptor = os.open(
        entry.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=destination_parent_descriptor,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, source_descriptor)
        descriptors.callback(os.close, destination_descriptor)
        if _copy_metadata_observation(os.fstat(source_descriptor)) != (
            entry.source_metadata
        ):
            raise RunWorkspaceFrontierError(
                "run workspace copy source file changed while opening"
            )
        remaining_bytes = entry.size_bytes
        while remaining_bytes:
            written_bytes = os.sendfile(
                destination_descriptor,
                source_descriptor,
                None,
                remaining_bytes,
            )
            if written_bytes <= 0:
                raise RunWorkspaceFrontierError(
                    "run workspace copy ended before the planned file size"
                )
            remaining_bytes -= written_bytes
        os.fchmod(destination_descriptor, entry.mode)
        os.fsync(destination_descriptor)
        _require_copied_regular_file(
            destination_descriptor,
            entry.mode,
            entry.size_bytes,
        )
        if _copy_metadata_observation(os.fstat(source_descriptor)) != (
            entry.source_metadata
        ):
            raise RunWorkspaceFrontierError(
                "run workspace copy source file changed while reading"
            )


def _require_copied_directory(descriptor: int, mode: int) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != os.getegid()
        or stat.S_IMODE(metadata.st_mode) != mode
    ):
        raise RunWorkspaceFrontierError(
            "run workspace copied directory has unsafe metadata"
        )


def _require_copied_regular_file(
    descriptor: int,
    mode: int,
    size_bytes: int,
) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != os.getegid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != mode
        or metadata.st_size != size_bytes
    ):
        raise RunWorkspaceFrontierError("run workspace copied file has unsafe metadata")


def _require_workspace_copy_tree(
    root_descriptor: int,
    plan: RunWorkspaceCopyPlan | RunWorkspaceSourceCopyPlan,
    *,
    source: bool,
    excluded_root_names: frozenset[str],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    root_metadata = os.fstat(root_descriptor)
    if source:
        if _copy_metadata_observation(root_metadata) != plan.source_root_metadata:
            raise RunWorkspaceFrontierError(
                "run workspace copy source root changed after copying"
            )
    elif (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or root_metadata.st_gid != os.getegid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
    ):
        raise RunWorkspaceFrontierError(
            "run workspace copy destination root changed after copying"
        )
    identities = {_metadata_identity(root_metadata)}
    observations = [(".", _copy_metadata_observation(root_metadata))]
    observed_entry_count = _require_workspace_copy_entries(
        root_descriptor,
        plan.entries,
        source=source,
        excluded_names=excluded_root_names,
        root_device=root_metadata.st_dev,
        identities=identities,
        observations=observations,
    )
    if observed_entry_count + 1 != plan.physical_entry_count:
        raise RunWorkspaceFrontierError(
            "run workspace copy final physical entry count changed"
        )
    return tuple(observations)


def _require_workspace_copy_entries(
    directory_descriptor: int,
    entries: tuple[_RunWorkspaceCopyEntry, ...],
    *,
    source: bool,
    excluded_names: frozenset[str],
    root_device: int,
    identities: set[tuple[int, int]],
    observations: list[tuple[str, tuple[int, ...]]],
) -> int:
    observed_names = set(os.listdir(directory_descriptor))
    if (
        type(excluded_names) is not frozenset
        or observed_names & excluded_names != excluded_names
        or tuple(sorted(observed_names - excluded_names))
        != tuple(entry.name for entry in entries)
    ):
        raise RunWorkspaceFrontierError("run workspace copy final topology changed")
    observed_entry_count = 0
    for entry in entries:
        descriptor_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        if entry.file_type == "directory":
            descriptor_flags |= os.O_DIRECTORY
        else:
            descriptor_flags |= os.O_NONBLOCK
        descriptor = os.open(
            entry.name,
            descriptor_flags,
            dir_fd=directory_descriptor,
        )
        with ExitStack() as descriptors:
            descriptors.callback(os.close, descriptor)
            metadata = os.fstat(descriptor)
            identity = _metadata_identity(metadata)
            if identity in identities or metadata.st_dev != root_device:
                raise RunWorkspaceFrontierError(
                    "run workspace copy final tree repeats or crosses an inode"
                )
            identities.add(identity)
            observations.append(
                (
                    entry.relative_path,
                    _copy_metadata_observation(metadata),
                )
            )
            if source:
                if _copy_metadata_observation(metadata) != entry.source_metadata:
                    raise RunWorkspaceFrontierError(
                        "run workspace copy source metadata changed after copying"
                    )
            elif entry.file_type == "directory":
                _require_copied_directory(descriptor, entry.mode)
            else:
                _require_copied_regular_file(
                    descriptor,
                    entry.mode,
                    entry.size_bytes,
                )
            observed_entry_count += 1
            if entry.file_type == "directory":
                observed_entry_count += _require_workspace_copy_entries(
                    descriptor,
                    entry.children,
                    source=source,
                    excluded_names=frozenset(),
                    root_device=root_device,
                    identities=identities,
                    observations=observations,
                )
            elif entry.file_type != "regular" or entry.children:
                raise RunWorkspaceFrontierError(
                    "run workspace copy final plan contains an invalid entry"
                )
    return observed_entry_count


def _scan_source_directory(
    directory_descriptor: int,
    relative_root: PurePosixPath,
    state: _SourceScanState,
    *,
    root: bool,
    empty_directories_allowed: bool,
    allowed_file_permissions: frozenset[int],
) -> tuple[
    dict[str, SourceFileDescriptor],
    dict[str, str],
    dict[str, int],
    dict[str, int],
    dict[str, tuple[int, ...]],
]:
    observed_entries = []
    with os.scandir(directory_descriptor) as iterator:
        for entry in iterator:
            metadata = entry.stat(follow_symlinks=False)
            if not (root and entry.name == ".git"):
                state.reserve_entry()
            observed_entries.append((entry.name, metadata))
    entries = tuple(sorted(observed_entries, key=lambda item: item[0]))
    if root and sum(name == ".git" for name, _metadata in entries) != 1:
        raise RunWorkspaceFrontierError(
            "run workspace must contain one Git metadata directory"
        )
    source_files: dict[str, SourceFileDescriptor] = {}
    blob_ids: dict[str, str] = {}
    directory_modes: dict[str, int] = {}
    regular_file_permissions: dict[str, int] = {}
    physical_metadata: dict[str, tuple[int, ...]] = {}
    for name, expected in entries:
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _metadata_identity(current) != _metadata_identity(expected):
            raise RunWorkspaceFrontierError(
                "run workspace entry changed during inspection"
            )
        if root and name == ".git":
            if not stat.S_ISDIR(expected.st_mode):
                raise RunWorkspaceFrontierError(
                    "run workspace Git metadata is not a directory"
                )
            continue
        _require_source_component(name)
        relative_path = (
            PurePosixPath(name)
            if relative_root == PurePosixPath(".")
            else relative_root / name
        )
        if stat.S_ISDIR(expected.st_mode):
            if expected.st_uid != os.geteuid() or expected.st_mode & (
                0o022 | stat.S_ISUID | stat.S_ISGID
            ):
                raise RunWorkspaceFrontierError(
                    "run workspace source directory is unsafe"
                )
            with ExitStack() as child_descriptors:
                child_descriptor = _open_directory(
                    directory_descriptor,
                    name,
                    child_descriptors,
                    "run workspace source directory",
                )
                (
                    child_files,
                    child_blobs,
                    child_directory_modes,
                    child_regular_file_permissions,
                    child_physical_metadata,
                ) = _scan_source_directory(
                    child_descriptor,
                    relative_path,
                    state,
                    root=False,
                    empty_directories_allowed=empty_directories_allowed,
                    allowed_file_permissions=allowed_file_permissions,
                )
            if not child_files and not empty_directories_allowed:
                raise RunWorkspaceFrontierError(
                    "run workspace contains an untracked empty directory"
                )
            directory_modes[relative_path.as_posix()] = stat.S_IMODE(expected.st_mode)
            directory_modes.update(child_directory_modes)
            regular_file_permissions.update(child_regular_file_permissions)
            physical_metadata[relative_path.as_posix()] = _copy_metadata_observation(
                expected
            )
            physical_metadata.update(child_physical_metadata)
            source_files.update(child_files)
            blob_ids.update(child_blobs)
            continue
        descriptor, blob_id, permissions = _read_source_file(
            directory_descriptor,
            name,
            relative_path.as_posix(),
            expected,
            state,
            allowed_file_permissions,
        )
        if descriptor.relative_path in source_files:
            raise RunWorkspaceFrontierError("run workspace source path is duplicated")
        source_files[descriptor.relative_path] = descriptor
        blob_ids[descriptor.relative_path] = blob_id
        regular_file_permissions[descriptor.relative_path] = permissions
        physical_metadata[descriptor.relative_path] = _copy_metadata_observation(
            expected
        )
    return (
        source_files,
        blob_ids,
        directory_modes,
        regular_file_permissions,
        physical_metadata,
    )


def _read_source_file(
    directory_descriptor: int,
    name: str,
    relative_path: str,
    expected: os.stat_result,
    state: _SourceScanState,
    allowed_file_permissions: frozenset[int],
) -> tuple[SourceFileDescriptor, str, int]:
    permissions = stat.S_IMODE(expected.st_mode)
    if (
        not stat.S_ISREG(expected.st_mode)
        or expected.st_uid != os.geteuid()
        or expected.st_nlink != 1
        or permissions not in allowed_file_permissions
        or expected.st_mode & (stat.S_ISUID | stat.S_ISGID)
    ):
        raise RunWorkspaceFrontierError("run workspace source entry is unsafe")
    state.reserve_file(expected.st_size)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_descriptor,
    )
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha1.update(f"blob {expected.st_size}\0".encode("ascii"))
    observed_size = 0
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if _metadata_observation(opened) != _metadata_observation(expected):
            raise RunWorkspaceFrontierError(
                "run workspace source file changed while opening"
            )
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            observed_size += len(chunk)
            sha256.update(chunk)
            sha1.update(chunk)
        completed = os.fstat(handle.fileno())
    rebound = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if (
        observed_size != expected.st_size
        or _metadata_observation(completed) != _metadata_observation(expected)
        or _metadata_observation(rebound) != _metadata_observation(expected)
    ):
        raise RunWorkspaceFrontierError(
            "run workspace source file changed while reading"
        )
    mode = "100755" if permissions in {0o700, 0o755} else "100644"
    return (
        SourceFileDescriptor(
            relative_path=relative_path,
            digest=f"sha256:{sha256.hexdigest()}",
            mode=mode,
            size=observed_size,
        ),
        sha1.hexdigest(),
        permissions,
    )


def _open_directory(
    parent_descriptor: int,
    name: str,
    descriptors: ExitStack,
    description: str,
) -> int:
    expected = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(expected.st_mode)
        or expected.st_uid != os.geteuid()
        or expected.st_mode & (0o022 | stat.S_ISUID | stat.S_ISGID)
        or _metadata_observation(opened) != _metadata_observation(expected)
    ):
        raise RunWorkspaceFrontierError(f"{description} is unsafe")
    descriptors.callback(
        _require_directory_unchanged,
        parent_descriptor,
        name,
        descriptor,
        expected,
        description,
    )
    return descriptor


def _require_directory_unchanged(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    description: str,
) -> None:
    completed = os.fstat(descriptor)
    rebound = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if _metadata_observation(completed) != _metadata_observation(
        expected
    ) or _metadata_identity(rebound) != _metadata_identity(expected):
        raise RunWorkspaceFrontierError(f"{description} changed during reconciliation")


def _read_regular_file(
    parent_descriptor: int,
    filename: str,
    *,
    maximum_bytes: int,
    allowed_modes: set[int],
    name: str,
) -> bytes:
    expected = os.stat(
        filename,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    descriptor = os.open(
        filename,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(expected.st_mode)
            or expected.st_uid != os.geteuid()
            or expected.st_nlink != 1
            or stat.S_IMODE(expected.st_mode) not in allowed_modes
            or expected.st_size > maximum_bytes
            or _metadata_identity(opened) != _metadata_identity(expected)
        ):
            raise RunWorkspaceFrontierError(f"{name} is unsafe or oversized")
        payload = handle.read(maximum_bytes + 1)
        completed = os.fstat(handle.fileno())
    rebound = os.stat(
        filename,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        len(payload) > maximum_bytes
        or _metadata_observation(completed) != _metadata_observation(expected)
        or _metadata_observation(rebound) != _metadata_observation(expected)
    ):
        raise RunWorkspaceFrontierError(f"{name} changed while reading")
    return payload


def _regular_file_permissions(
    parent_descriptor: int,
    filename: str,
) -> int:
    metadata = os.stat(
        filename,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(metadata.st_mode):
        raise RunWorkspaceFrontierError(
            "run workspace Git metadata entry is not regular"
        )
    return stat.S_IMODE(metadata.st_mode)


def _read_exact_git_closure(
    git_descriptor: int,
    *,
    workspace_git_branch: str,
    maximum_git_bytes: int,
    state: _GitClosureState,
    descriptors: ExitStack,
) -> tuple[dict[str, _GitObject], bytes]:
    entries = _directory_entries(git_descriptor, state)
    allowed = {"HEAD", "config", "index", "objects", "refs", "COMMIT_EDITMSG"}
    if not {"HEAD", "config", "index", "objects", "refs"}.issubset(entries) or not set(
        entries
    ).issubset(allowed):
        raise RunWorkspaceFrontierError(
            "run workspace Git root contains unsupported control state"
        )
    if "COMMIT_EDITMSG" in entries:
        message = _read_regular_file(
            git_descriptor,
            "COMMIT_EDITMSG",
            maximum_bytes=maximum_git_bytes,
            allowed_modes={0o600, 0o644},
            name="run workspace Git commit message",
        )
        state.reserve_regular_file(
            "COMMIT_EDITMSG",
            message,
            _regular_file_permissions(git_descriptor, "COMMIT_EDITMSG"),
        )
    branch_reference = _read_exact_branch_reference(
        git_descriptor,
        workspace_git_branch,
        maximum_git_bytes,
        state,
        descriptors,
    )
    objects_descriptor = _open_directory(
        git_descriptor,
        "objects",
        descriptors,
        "run workspace Git objects",
    )
    objects = _read_loose_git_objects(
        objects_descriptor,
        maximum_git_bytes,
        state,
        descriptors,
    )
    return objects, branch_reference


def _read_exact_branch_reference(
    git_descriptor: int,
    branch: str,
    maximum_bytes: int,
    state: _GitClosureState,
    descriptors: ExitStack,
) -> bytes:
    parent_descriptor = _open_directory(
        git_descriptor,
        "refs",
        descriptors,
        "run workspace Git refs",
    )
    entries = _directory_entries(parent_descriptor, state)
    if set(entries) != {"heads"}:
        raise RunWorkspaceFrontierError(
            "run workspace Git refs contain unsupported references"
        )
    parent_descriptor = _open_directory(
        parent_descriptor,
        "heads",
        descriptors,
        "run workspace Git branch refs",
    )
    branch_parts = branch.split("/")
    for position, component in enumerate(branch_parts):
        entries = _directory_entries(parent_descriptor, state)
        if set(entries) != {component}:
            raise RunWorkspaceFrontierError(
                "run workspace Git branch refs are not exact"
            )
        if position < len(branch_parts) - 1:
            parent_descriptor = _open_directory(
                parent_descriptor,
                component,
                descriptors,
                "run workspace Git branch refs",
            )
    reference = _read_regular_file(
        parent_descriptor,
        branch_parts[-1],
        maximum_bytes=maximum_bytes,
        allowed_modes={0o600, 0o644},
        name="run workspace branch reference",
    )
    state.reserve_regular_file(
        f"refs/heads/{branch}",
        reference,
        _regular_file_permissions(
            parent_descriptor,
            branch_parts[-1],
        ),
    )
    return reference


def _read_loose_git_objects(
    objects_descriptor: int,
    maximum_bytes: int,
    state: _GitClosureState,
    descriptors: ExitStack,
) -> dict[str, _GitObject]:
    entries = _directory_entries(objects_descriptor, state)
    if "info" not in entries or "pack" not in entries:
        raise RunWorkspaceFrontierError("run workspace Git object store is incomplete")
    for fixed_directory in ("info", "pack"):
        descriptor = _open_directory(
            objects_descriptor,
            fixed_directory,
            descriptors,
            f"run workspace Git objects/{fixed_directory}",
        )
        fixed_entries = _directory_entries(descriptor, state)
        if fixed_entries:
            raise RunWorkspaceFrontierError(
                "run workspace Git object store is not loose and self-contained"
            )
    objects: dict[str, _GitObject] = {}
    loose_directories = tuple(name for name in entries if name not in {"info", "pack"})
    if any(re.fullmatch(r"[0-9a-f]{2}", name) is None for name in loose_directories):
        raise RunWorkspaceFrontierError(
            "run workspace Git object store contains an unsafe path"
        )
    for prefix in loose_directories:
        prefix_descriptor = _open_directory(
            objects_descriptor,
            prefix,
            descriptors,
            "run workspace loose Git object directory",
        )
        filenames = _directory_entries(prefix_descriptor, state)
        if not filenames or any(
            re.fullmatch(r"[0-9a-f]{38}", filename) is None for filename in filenames
        ):
            raise RunWorkspaceFrontierError(
                "run workspace loose Git object directory is invalid"
            )
        for filename in filenames:
            object_id = prefix + filename
            compressed = _read_regular_file(
                prefix_descriptor,
                filename,
                maximum_bytes=maximum_bytes,
                allowed_modes={0o400, 0o444, 0o600, 0o644},
                name="run workspace Git object",
            )
            state.reserve_regular_file(
                f"objects/{prefix}/{filename}",
                compressed,
                _regular_file_permissions(prefix_descriptor, filename),
            )
            if object_id in objects:
                raise RunWorkspaceFrontierError(
                    "run workspace Git object ID is duplicated"
                )
            objects[object_id] = _decode_git_object(
                object_id,
                compressed,
                state,
            )
    if not objects:
        raise RunWorkspaceFrontierError("run workspace Git object store is empty")
    return objects


def _decode_git_object(
    object_id: str,
    compressed: bytes,
    state: _GitClosureState,
) -> _GitObject:
    decompressor = zlib.decompressobj()
    object_bytes = decompressor.decompress(compressed, state.size_limit + 1)
    if (
        len(object_bytes) > state.size_limit
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or hashlib.sha1(object_bytes, usedforsecurity=False).hexdigest() != object_id
    ):
        raise RunWorkspaceFrontierError(
            "run workspace Git object is corrupt or oversized"
        )
    separator = object_bytes.find(b"\0")
    if separator <= 0:
        raise RunWorkspaceFrontierError("run workspace Git object lacks a header")
    header = object_bytes[:separator]
    payload = object_bytes[separator + 1 :]
    kind_bytes, separator, size_bytes = header.partition(b" ")
    if (
        not separator
        or kind_bytes not in {b"blob", b"commit", b"tree"}
        or not size_bytes.isdigit()
        or int(size_bytes) != len(payload)
    ):
        raise RunWorkspaceFrontierError(
            "run workspace Git object has an unsupported kind or size"
        )
    state.reserve_decoded_object(object_bytes)
    return _GitObject(
        kind=kind_bytes.decode("ascii"),
        payload=payload,
    )


def _require_git_object(
    objects: dict[str, _GitObject],
    object_id: str,
    *,
    expected_kind: str,
) -> bytes:
    value = objects.get(object_id)
    if value is None or value.kind != expected_kind:
        raise RunWorkspaceFrontierError(
            "run workspace Git lineage references a missing or wrong-kind object"
        )
    return value.payload


def _require_reachable_git_object_closure(
    objects: dict[str, _GitObject],
    commit_sha: str,
) -> None:
    pending = [("commit", commit_sha)]
    reachable = set()
    while pending:
        expected_kind, object_id = pending.pop()
        if object_id in reachable:
            continue
        payload = _require_git_object(
            objects,
            object_id,
            expected_kind=expected_kind,
        )
        reachable.add(object_id)
        if expected_kind == "commit":
            tree_sha, parent_shas = _parse_commit(payload)
            pending.append(("tree", tree_sha))
            pending.extend(("commit", parent_sha) for parent_sha in parent_shas)
        elif expected_kind == "tree":
            pending.extend(_parse_tree_edges(payload))
    if reachable != set(objects):
        raise RunWorkspaceFrontierError(
            "run workspace Git object store contains unreachable objects"
        )


def _parse_tree_edges(payload: bytes) -> tuple[tuple[str, str], ...]:
    position = 0
    edges = []
    sort_names = []
    names = set()
    while position < len(payload):
        separator = payload.find(b" ", position)
        terminator = payload.find(b"\0", separator + 1)
        if (
            separator <= position
            or terminator <= separator + 1
            or terminator + 21 > len(payload)
        ):
            raise RunWorkspaceFrontierError(
                "run workspace Git tree object is malformed"
            )
        mode = payload[position:separator]
        name = payload[separator + 1 : terminator].decode("utf-8")
        _require_source_component(name)
        if "/" in name or name in names:
            raise RunWorkspaceFrontierError(
                "run workspace Git tree contains an invalid path"
            )
        names.add(name)
        object_id = payload[terminator + 1 : terminator + 21].hex()
        if mode == b"40000":
            edges.append(("tree", object_id))
            sort_names.append(name.encode("utf-8") + b"/")
        elif mode in {b"100644", b"100755"}:
            edges.append(("blob", object_id))
            sort_names.append(name.encode("utf-8"))
        else:
            raise RunWorkspaceFrontierError(
                "run workspace Git tree contains an unsupported mode"
            )
        position = terminator + 21
    if sort_names != sorted(sort_names):
        raise RunWorkspaceFrontierError(
            "run workspace Git tree entries are not canonical"
        )
    return tuple(edges)


def _directory_entries(
    directory_descriptor: int,
    state: _GitClosureState,
) -> dict[str, os.stat_result]:
    entries = {}
    with os.scandir(directory_descriptor) as iterator:
        for entry in iterator:
            state.reserve_entries(1)
            entries[entry.name] = entry.stat(follow_symlinks=False)
    return entries


def _parse_commit(payload: bytes) -> tuple[str, tuple[str, ...]]:
    header, separator, _message = payload.partition(b"\n\n")
    lines = header.split(b"\n")
    if (
        not separator
        or not lines
        or not lines[0].startswith(b"tree ")
        or len(lines[0]) != 45
    ):
        raise RunWorkspaceFrontierError("run workspace Git commit structure is invalid")
    tree_sha = lines[0][5:].decode("ascii")
    position = 1
    parents = []
    while position < len(lines) and lines[position].startswith(b"parent "):
        if len(lines[position]) != 47:
            raise RunWorkspaceFrontierError(
                "run workspace Git commit parent is invalid"
            )
        parents.append(lines[position][7:].decode("ascii"))
        position += 1
    if (
        position + 2 != len(lines)
        or not lines[position].startswith(b"author ")
        or not lines[position + 1].startswith(b"committer ")
        or _GIT_IDENTITY_PATTERN.fullmatch(lines[position]) is None
        or _GIT_IDENTITY_PATTERN.fullmatch(lines[position + 1]) is None
        or _GIT_SHA_PATTERN.fullmatch(tree_sha) is None
        or any(_GIT_SHA_PATTERN.fullmatch(parent) is None for parent in parents)
        or len(parents) != len(set(parents))
    ):
        raise RunWorkspaceFrontierError(
            "run workspace Git commit references are invalid"
        )
    return tree_sha, tuple(parents)


def _require_index_matches_source(
    payload: bytes,
    descriptors_by_path: dict[str, SourceFileDescriptor],
    blob_ids: dict[str, str],
) -> None:
    if (
        len(payload) < _GIT_INDEX_HEADER.size + 20
        or hashlib.sha1(payload[:-20], usedforsecurity=False).digest() != payload[-20:]
    ):
        raise RunWorkspaceFrontierError("run workspace Git index checksum is invalid")
    signature, version, entry_count = _GIT_INDEX_HEADER.unpack_from(payload)
    if signature != b"DIRC" or version != 2 or entry_count != len(descriptors_by_path):
        raise RunWorkspaceFrontierError(
            "run workspace Git index header differs from its source tree"
        )
    position = _GIT_INDEX_HEADER.size
    observed: dict[str, tuple[str, str]] = {}
    previous_path_bytes = None
    for _entry_number in range(entry_count):
        entry_start = position
        if position + _GIT_INDEX_ENTRY_HEADER.size > len(payload) - 20:
            raise RunWorkspaceFrontierError(
                "run workspace Git index entry is truncated"
            )
        fields = _GIT_INDEX_ENTRY_HEADER.unpack_from(payload, position)
        position += _GIT_INDEX_ENTRY_HEADER.size
        mode = fields[6]
        object_id = fields[10].hex()
        flags = fields[11]
        if flags & 0xF000:
            raise RunWorkspaceFrontierError(
                "run workspace Git index contains unsupported flags"
            )
        declared_path_size = flags & 0x0FFF
        terminator = payload.find(b"\0", position, len(payload) - 20)
        if terminator < 0:
            raise RunWorkspaceFrontierError(
                "run workspace Git index path is unterminated"
            )
        path_bytes = payload[position:terminator]
        if (
            declared_path_size == 0x0FFF
            or declared_path_size != len(path_bytes)
            or (previous_path_bytes is not None and path_bytes <= previous_path_bytes)
        ):
            raise RunWorkspaceFrontierError(
                "run workspace Git index path order or length is invalid"
            )
        previous_path_bytes = path_bytes
        path = path_bytes.decode("utf-8")
        _require_source_path(path)
        position = entry_start + ((terminator + 1 - entry_start + 7) // 8) * 8
        if position > len(payload) - 20 or any(
            payload[index] != 0 for index in range(terminator + 1, position)
        ):
            raise RunWorkspaceFrontierError(
                "run workspace Git index padding is invalid"
            )
        normalized_mode = format(mode, "o")
        if path in observed or mode not in {0o100644, 0o100755}:
            raise RunWorkspaceFrontierError("run workspace Git index entry is invalid")
        observed[path] = (object_id, normalized_mode)
    expected = {
        path: (blob_ids[path], descriptor.mode)
        for path, descriptor in descriptors_by_path.items()
    }
    if observed != expected:
        raise RunWorkspaceFrontierError(
            "run workspace Git index differs from its clean source tree"
        )
    extension_signatures = set()
    while position < len(payload) - 20:
        if position + 8 > len(payload) - 20:
            raise RunWorkspaceFrontierError(
                "run workspace Git index extension is truncated"
            )
        signature = payload[position : position + 4]
        extension_size = struct.unpack_from("!L", payload, position + 4)[0]
        extension_end = position + 8 + extension_size
        if (
            signature != b"TREE"
            or signature in extension_signatures
            or extension_end > len(payload) - 20
        ):
            raise RunWorkspaceFrontierError(
                "run workspace Git index extension is unsupported"
            )
        extension_signatures.add(signature)
        position = extension_end
    if position != len(payload) - 20:
        raise RunWorkspaceFrontierError("run workspace Git index length is invalid")


def _require_source_component(name: str) -> None:
    if (
        name in {"", ".", "..", ".git", ".env"}
        or name.startswith(".env.")
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise RunWorkspaceFrontierError("run workspace contains a denied source path")


def _require_copy_component(name: str) -> None:
    if (
        type(name) is not str
        or name in {"", ".", ".."}
        or "/" in name
        or "\x00" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise RunWorkspaceFrontierError(
            "run workspace physical copy contains an unsafe path"
        )


def _require_source_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or ".git" in path.parts
    ):
        raise RunWorkspaceFrontierError(
            "run workspace Git index contains an unsafe source path"
        )
    for component in path.parts:
        _require_source_component(component)


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _metadata_observation(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        stat.S_IMODE(metadata.st_mode),
    )


def _copy_metadata_observation(
    metadata: os.stat_result,
) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


__all__ = [
    "copy_run_workspace_frontier",
    "copy_run_workspace_source_tree",
    "inspect_detached_run_workspace_source_tree",
    "inspect_run_workspace_frontier",
    "inspect_run_workspace_frontier_with_limits",
    "inspect_run_workspace_regular_tree",
    "inspect_run_workspace_source_regular_tree",
    "inspect_run_workspace_source_tree",
    "plan_run_workspace_frontier_copy",
    "plan_run_workspace_source_copy",
    "replace_run_workspace_source_tree",
    "RunWorkspaceCopyPlan",
    "RunWorkspaceSourceCopyPlan",
    "RunWorkspaceFrontierError",
    "RunWorkspaceFrontierIdentity",
    "RunWorkspaceRegularTreeIdentity",
    "RunWorkspaceSourceTreeIdentity",
]
