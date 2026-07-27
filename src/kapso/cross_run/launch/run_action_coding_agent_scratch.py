"""Group-shared `.git`-free scratch authority for coding-agent providers."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.launch.workspace_frontier import (
    RunWorkspaceRegularTreeIdentity,
    RunWorkspaceSourceTreeIdentity,
    inspect_detached_run_workspace_source_tree,
)

PROVIDER_SHARED_DIRECTORY_MODE = 0o2770
PROVIDER_SHARED_FILE_MODE = 0o660
PROVIDER_SHARED_EXECUTABLE_MODE = 0o770

_READ_CHUNK_SIZE_BYTES = 65_536


class RunActionCodingAgentScratchError(RuntimeError):
    """The provider scratch tree is unsafe, inaccessible, or unstable."""


@dataclass(frozen=True)
class CodingAgentScratchTreeIdentity:
    """Semantic and physical identity of one provider-readable source tree."""

    source: RunWorkspaceSourceTreeIdentity
    physical: RunWorkspaceRegularTreeIdentity

    def __post_init__(self) -> None:
        if (
            type(self.source) is not RunWorkspaceSourceTreeIdentity
            or type(self.physical) is not RunWorkspaceRegularTreeIdentity
            or self.source.workspace_identity != self.physical.root_identity
            or self.source.source_entry_count != self.physical.entry_count
            or self.source.source_size_bytes != self.physical.regular_file_size_bytes
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent scratch identity is inconsistent"
            )


@dataclass
class _ScratchScanState:
    maximum_entries: int
    maximum_bytes: int
    entry_count: int = 0
    regular_file_size_bytes: int = 0

    def reserve(self, size_bytes: int | None) -> None:
        self.entry_count += 1
        if self.entry_count > self.maximum_entries:
            raise RunActionCodingAgentScratchError(
                "coding-agent scratch exceeds its entry limit"
            )
        if size_bytes is not None:
            self.regular_file_size_bytes += size_bytes
            if self.regular_file_size_bytes > self.maximum_bytes:
                raise RunActionCodingAgentScratchError(
                    "coding-agent scratch exceeds its byte limit"
                )


def share_coding_agent_scratch_source_tree(
    workspace_descriptor: int,
    *,
    expected_source: RunWorkspaceSourceTreeIdentity,
    supervisor_user_id: int,
    provider_user_id: int,
    provider_group_id: int,
    maximum_entries: int,
    maximum_bytes: int,
) -> CodingAgentScratchTreeIdentity:
    """Convert a proved private detached copy into the provider group ABI."""

    if (
        type(expected_source) is not RunWorkspaceSourceTreeIdentity
        or os.geteuid() != supervisor_user_id
        or supervisor_user_id == provider_user_id
        or type(provider_group_id) is not int
        or provider_group_id <= 0
        or provider_group_id not in {os.getegid(), *os.getgroups()}
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch sharing identity is invalid"
        )
    observed_before = inspect_detached_run_workspace_source_tree(
        workspace_descriptor,
        maximum_entries=maximum_entries,
        maximum_bytes=maximum_bytes,
    )
    if observed_before != expected_source:
        raise RunActionCodingAgentScratchError(
            "coding-agent detached source differs before sharing"
        )
    root_before = os.fstat(workspace_descriptor)
    _share_private_directory_entries(
        workspace_descriptor,
        root_device=root_before.st_dev,
        provider_group_id=provider_group_id,
    )
    os.fchown(workspace_descriptor, -1, provider_group_id)
    os.fchmod(workspace_descriptor, PROVIDER_SHARED_DIRECTORY_MODE)
    os.fsync(workspace_descriptor)
    shared = inspect_coding_agent_scratch_source_tree(
        workspace_descriptor,
        supervisor_user_id=supervisor_user_id,
        provider_user_id=provider_user_id,
        provider_group_id=provider_group_id,
        maximum_entries=maximum_entries,
        maximum_bytes=maximum_bytes,
    )
    if (
        shared.source.source_tree_digest != expected_source.source_tree_digest
        or shared.source.source_entry_count != expected_source.source_entry_count
        or shared.source.source_size_bytes != expected_source.source_size_bytes
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent shared scratch differs from its detached source"
        )
    return shared


def inspect_coding_agent_scratch_source_tree(
    workspace_descriptor: int,
    *,
    supervisor_user_id: int,
    provider_user_id: int,
    provider_group_id: int,
    maximum_entries: int,
    maximum_bytes: int,
) -> CodingAgentScratchTreeIdentity:
    """Read a complete bounded provider tree only through safe shared metadata."""

    if (
        type(workspace_descriptor) is not int
        or workspace_descriptor < 0
        or any(
            type(value) is not int or value <= 0
            for value in (
                supervisor_user_id,
                provider_user_id,
                provider_group_id,
                maximum_entries,
                maximum_bytes,
            )
        )
        or supervisor_user_id == provider_user_id
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch inspection inputs are invalid"
        )
    root = os.fstat(workspace_descriptor)
    if (
        not stat.S_ISDIR(root.st_mode)
        or root.st_uid != supervisor_user_id
        or root.st_gid != provider_group_id
        or stat.S_IMODE(root.st_mode) != PROVIDER_SHARED_DIRECTORY_MODE
        or root.st_dev <= 0
        or root.st_ino <= 0
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch root is not supervisor-owned shared authority"
        )
    state = _ScratchScanState(maximum_entries, maximum_bytes)
    source_files: dict[str, SourceFileDescriptor] = {}
    physical_directories: dict[str, tuple[int, ...]] = {}
    physical_files: dict[str, dict[str, object]] = {}
    identities = {(root.st_dev, root.st_ino)}
    _scan_shared_directory(
        workspace_descriptor,
        relative_root=PurePosixPath("."),
        root_device=root.st_dev,
        allowed_user_ids=frozenset({supervisor_user_id, provider_user_id}),
        provider_group_id=provider_group_id,
        state=state,
        identities=identities,
        source_files=source_files,
        physical_directories=physical_directories,
        physical_files=physical_files,
    )
    if not source_files:
        raise RunActionCodingAgentScratchError("coding-agent scratch source is empty")
    root_after = os.fstat(workspace_descriptor)
    if _stable_metadata(root_after) != _stable_metadata(root):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch root changed during inspection"
        )
    source = RunWorkspaceSourceTreeIdentity(
        workspace_identity=(root.st_dev, root.st_ino),
        source_tree_digest=source_tree_digest(
            {
                path: (descriptor.digest, descriptor.mode, descriptor.size)
                for path, descriptor in source_files.items()
            }
        ),
        source_entry_count=state.entry_count,
        source_size_bytes=state.regular_file_size_bytes,
    )
    physical = RunWorkspaceRegularTreeIdentity(
        root_identity=(root.st_dev, root.st_ino),
        tree_digest=tree_or_blob_digest(
            canonical_json_bytes(
                {
                    "root": _stable_metadata(root),
                    "directories": {
                        path: physical_directories[path]
                        for path in sorted(physical_directories)
                    },
                    "regular_files": {
                        path: physical_files[path] for path in sorted(physical_files)
                    },
                }
            )
        ),
        entry_count=state.entry_count,
        regular_file_size_bytes=state.regular_file_size_bytes,
    )
    return CodingAgentScratchTreeIdentity(source=source, physical=physical)


def _share_private_directory_entries(
    directory_descriptor: int,
    *,
    root_device: int,
    provider_group_id: int,
) -> None:
    names = tuple(sorted(os.listdir(directory_descriptor)))
    for name in names:
        _require_source_component(name)
        expected = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            expected.st_dev != root_device
            or expected.st_uid != os.geteuid()
            or expected.st_gid != os.getegid()
            or expected.st_mode & (stat.S_ISUID | stat.S_ISGID)
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent private scratch entry is unsafe"
            )
        if stat.S_ISDIR(expected.st_mode):
            if expected.st_mode & 0o022:
                raise RunActionCodingAgentScratchError(
                    "coding-agent private scratch directory is writable by others"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            with ExitStack() as resources:
                resources.callback(os.close, descriptor)
                if _stable_metadata(os.fstat(descriptor)) != _stable_metadata(expected):
                    raise RunActionCodingAgentScratchError(
                        "coding-agent private scratch directory changed while opening"
                    )
                _share_private_directory_entries(
                    descriptor,
                    root_device=root_device,
                    provider_group_id=provider_group_id,
                )
                os.fchown(descriptor, -1, provider_group_id)
                os.fchmod(descriptor, PROVIDER_SHARED_DIRECTORY_MODE)
                os.fsync(descriptor)
            continue
        permissions = stat.S_IMODE(expected.st_mode)
        if (
            not stat.S_ISREG(expected.st_mode)
            or expected.st_nlink != 1
            or permissions not in {0o600, 0o644, 0o700, 0o755}
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent private scratch file is unsafe"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        with ExitStack() as resources:
            resources.callback(os.close, descriptor)
            if _stable_metadata(os.fstat(descriptor)) != _stable_metadata(expected):
                raise RunActionCodingAgentScratchError(
                    "coding-agent private scratch file changed while opening"
                )
            os.fchown(descriptor, -1, provider_group_id)
            os.fchmod(
                descriptor,
                (
                    PROVIDER_SHARED_EXECUTABLE_MODE
                    if permissions in {0o700, 0o755}
                    else PROVIDER_SHARED_FILE_MODE
                ),
            )
            os.fsync(descriptor)
    if tuple(sorted(os.listdir(directory_descriptor))) != names:
        raise RunActionCodingAgentScratchError(
            "coding-agent private scratch topology changed while sharing"
        )


def _scan_shared_directory(
    directory_descriptor: int,
    *,
    relative_root: PurePosixPath,
    root_device: int,
    allowed_user_ids: frozenset[int],
    provider_group_id: int,
    state: _ScratchScanState,
    identities: set[tuple[int, int]],
    source_files: dict[str, SourceFileDescriptor],
    physical_directories: dict[str, tuple[int, ...]],
    physical_files: dict[str, dict[str, object]],
) -> None:
    with os.scandir(directory_descriptor) as iterator:
        observed = tuple(
            sorted(
                (
                    entry.name,
                    entry.stat(follow_symlinks=False),
                )
                for entry in iterator
            )
        )
    for name, expected in observed:
        _require_source_component(name)
        current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        identity = (expected.st_dev, expected.st_ino)
        if (
            _stable_metadata(current) != _stable_metadata(expected)
            or expected.st_dev != root_device
            or identity in identities
            or expected.st_uid not in allowed_user_ids
            or expected.st_gid != provider_group_id
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent shared scratch entry identity is unsafe"
            )
        identities.add(identity)
        relative_path = (
            PurePosixPath(name)
            if relative_root == PurePosixPath(".")
            else relative_root / name
        )
        state.reserve(expected.st_size if stat.S_ISREG(expected.st_mode) else None)
        if stat.S_ISDIR(expected.st_mode):
            if stat.S_IMODE(expected.st_mode) != PROVIDER_SHARED_DIRECTORY_MODE:
                raise RunActionCodingAgentScratchError(
                    "coding-agent shared scratch directory is inaccessible"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            with ExitStack() as resources:
                resources.callback(os.close, descriptor)
                if _stable_metadata(os.fstat(descriptor)) != _stable_metadata(expected):
                    raise RunActionCodingAgentScratchError(
                        "coding-agent shared scratch directory changed while opening"
                    )
                child_count = state.entry_count
                _scan_shared_directory(
                    descriptor,
                    relative_root=relative_path,
                    root_device=root_device,
                    allowed_user_ids=allowed_user_ids,
                    provider_group_id=provider_group_id,
                    state=state,
                    identities=identities,
                    source_files=source_files,
                    physical_directories=physical_directories,
                    physical_files=physical_files,
                )
                if state.entry_count == child_count:
                    raise RunActionCodingAgentScratchError(
                        "coding-agent scratch contains an empty directory"
                    )
                completed = os.fstat(descriptor)
            rebound = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if _stable_metadata(completed) != _stable_metadata(
                expected
            ) or _stable_metadata(rebound) != _stable_metadata(expected):
                raise RunActionCodingAgentScratchError(
                    "coding-agent shared scratch directory changed while reading"
                )
            physical_directories[relative_path.as_posix()] = _stable_metadata(expected)
            continue
        permissions = stat.S_IMODE(expected.st_mode)
        if (
            not stat.S_ISREG(expected.st_mode)
            or expected.st_nlink != 1
            or permissions
            not in {PROVIDER_SHARED_FILE_MODE, PROVIDER_SHARED_EXECUTABLE_MODE}
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent shared scratch file is inaccessible or unsafe"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        sha256 = hashlib.sha256()
        observed_size = 0
        with os.fdopen(descriptor, "rb") as handle:
            if _stable_metadata(os.fstat(handle.fileno())) != _stable_metadata(
                expected
            ):
                raise RunActionCodingAgentScratchError(
                    "coding-agent shared scratch file changed while opening"
                )
            while True:
                chunk = handle.read(_READ_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                observed_size += len(chunk)
                sha256.update(chunk)
            completed = os.fstat(handle.fileno())
        rebound = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            observed_size != expected.st_size
            or _stable_metadata(completed) != _stable_metadata(expected)
            or _stable_metadata(rebound) != _stable_metadata(expected)
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent shared scratch file changed while reading"
            )
        path = relative_path.as_posix()
        source_files[path] = SourceFileDescriptor(
            relative_path=path,
            digest=f"sha256:{sha256.hexdigest()}",
            mode=(
                "100755" if permissions == PROVIDER_SHARED_EXECUTABLE_MODE else "100644"
            ),
            size=observed_size,
        )
        physical_files[path] = {
            "content_digest": source_files[path].digest,
            "metadata": _stable_metadata(expected),
        }
    if tuple(sorted(os.listdir(directory_descriptor))) != tuple(
        name for name, _metadata in observed
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent shared scratch topology changed during inspection"
        )


def _require_source_component(name: str) -> None:
    if (
        not isinstance(name, str)
        or name in {"", ".", "..", ".git", ".env"}
        or name.startswith(".env.")
        or "/" in name
        or "\x00" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch contains a denied source path"
        )


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
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
    "CodingAgentScratchTreeIdentity",
    "inspect_coding_agent_scratch_source_tree",
    "PROVIDER_SHARED_DIRECTORY_MODE",
    "PROVIDER_SHARED_EXECUTABLE_MODE",
    "PROVIDER_SHARED_FILE_MODE",
    "RunActionCodingAgentScratchError",
    "share_coding_agent_scratch_source_tree",
]
