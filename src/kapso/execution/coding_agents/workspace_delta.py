"""Pinned workspace observation and replayable coding-agent edit deltas."""

from __future__ import annotations

import base64
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    CodingAgentWorkspaceChangedFile,
    CodingAgentWorkspaceDelta,
    SourceFileDescriptor,
)


class CodingAgentWorkspaceError(ValueError):
    """A coding-agent workspace or durable delta is unsafe or inconsistent."""


@dataclass(frozen=True)
class CodingAgentWorkspaceFileSnapshot:
    descriptor: SourceFileDescriptor
    content: bytes


@dataclass(frozen=True)
class CodingAgentWorkspaceSnapshot:
    tree_hash: str
    files: tuple[CodingAgentWorkspaceFileSnapshot, ...]

    def by_path(self) -> dict[str, CodingAgentWorkspaceFileSnapshot]:
        return {file.descriptor.relative_path: file for file in self.files}


@dataclass
class _WorkspaceScanState:
    maximum_entries: int
    maximum_bytes: int
    observed_entries: int = 0
    observed_bytes: int = 0

    def observe_entry(self) -> None:
        self.observed_entries += 1
        if self.observed_entries > self.maximum_entries:
            raise CodingAgentWorkspaceError(
                "coding-agent workspace exceeds its entry limit"
            )

    def observe_file(self, size: int) -> None:
        self.observed_bytes += size
        if self.observed_bytes > self.maximum_bytes:
            raise CodingAgentWorkspaceError(
                "coding-agent workspace exceeds its byte limit"
            )


def inspect_coding_agent_workspace(
    root: Path,
    *,
    maximum_entries: int,
    maximum_bytes: int,
) -> CodingAgentWorkspaceSnapshot:
    """Read one exact regular-file tree through pinned descriptors."""

    if (
        not root.is_absolute()
        or root != Path(os.path.abspath(root))
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != root
    ):
        raise CodingAgentWorkspaceError(
            "coding-agent editable workspace must be a normalized real directory"
        )
    root_metadata = root.stat(follow_symlinks=False)
    with ExitStack() as descriptors:
        root_descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, root_descriptor)
        opened_root = os.fstat(root_descriptor)
        if (opened_root.st_dev, opened_root.st_ino) != (
            root_metadata.st_dev,
            root_metadata.st_ino,
        ):
            raise CodingAgentWorkspaceError(
                "coding-agent workspace root changed while opening"
            )
        snapshot = inspect_coding_agent_workspace_descriptor(
            root_descriptor,
            maximum_entries=maximum_entries,
            maximum_bytes=maximum_bytes,
        )
        current_root = os.stat(root, follow_symlinks=False)
        if (current_root.st_dev, current_root.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise CodingAgentWorkspaceError(
                "coding-agent workspace root changed during inspection"
            )
        return snapshot


def inspect_coding_agent_workspace_descriptor(
    root_descriptor: int,
    *,
    maximum_entries: int,
    maximum_bytes: int,
) -> CodingAgentWorkspaceSnapshot:
    """Read one exact tree from a caller-owned pinned directory descriptor."""

    if (
        isinstance(maximum_entries, bool)
        or not isinstance(maximum_entries, int)
        or maximum_entries <= 0
        or isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes <= 0
    ):
        raise CodingAgentWorkspaceError(
            "coding-agent workspace limits must be positive integers"
        )
    root_metadata = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_mode & (
        0o077 | stat.S_ISUID | stat.S_ISGID
    ):
        raise CodingAgentWorkspaceError(
            "coding-agent editable workspace descriptor must be private"
        )
    state = _WorkspaceScanState(maximum_entries, maximum_bytes)
    files = _read_workspace_directory(
        root_descriptor,
        PurePosixPath("."),
        state,
    )
    observed_root = os.fstat(root_descriptor)
    if _metadata_identity(observed_root) != _metadata_identity(root_metadata):
        raise CodingAgentWorkspaceError(
            "coding-agent workspace descriptor changed during inspection"
        )
    ordered_files = tuple(sorted(files, key=lambda file: file.descriptor.relative_path))
    tree_hash = _descriptor_tree_hash(
        {file.descriptor.relative_path: file.descriptor for file in ordered_files}
    )
    return CodingAgentWorkspaceSnapshot(tree_hash=tree_hash, files=ordered_files)


def build_coding_agent_workspace_delta(
    baseline: CodingAgentWorkspaceSnapshot,
    edited: CodingAgentWorkspaceSnapshot,
) -> CodingAgentWorkspaceDelta:
    baseline_files = baseline.by_path()
    edited_files = edited.by_path()
    changed_files = tuple(
        CodingAgentWorkspaceChangedFile(
            before=(
                None if path not in baseline_files else baseline_files[path].descriptor
            ),
            after=edited_files[path].descriptor,
            content_base64=base64.b64encode(edited_files[path].content).decode("ascii"),
        )
        for path in sorted(edited_files)
        if baseline_files.get(path) != edited_files[path]
    )
    deleted_files = tuple(
        baseline_files[path].descriptor
        for path in sorted(set(baseline_files) - set(edited_files))
    )
    return CodingAgentWorkspaceDelta.mint(
        baseline_tree_hash=baseline.tree_hash,
        edited_tree_hash=edited.tree_hash,
        changed_files=changed_files,
        deleted_files=deleted_files,
    )


def validate_coding_agent_workspace_delta(
    observed: CodingAgentWorkspaceSnapshot,
    delta: CodingAgentWorkspaceDelta,
) -> None:
    """Prove a delta against either its exact baseline or exact edited tree."""

    observed_files = {
        path: file.descriptor for path, file in observed.by_path().items()
    }
    if observed.tree_hash == delta.baseline_tree_hash:
        transformed = _apply_delta_descriptors(observed_files, delta)
        if _descriptor_tree_hash(transformed) != delta.edited_tree_hash:
            raise CodingAgentWorkspaceError(
                "coding-agent workspace delta does not reproduce its edited tree"
            )
        return
    if observed.tree_hash == delta.edited_tree_hash:
        transformed = _reverse_delta_descriptors(observed_files, delta)
        if _descriptor_tree_hash(transformed) != delta.baseline_tree_hash:
            raise CodingAgentWorkspaceError(
                "coding-agent workspace delta does not reproduce its baseline tree"
            )
        return
    raise CodingAgentWorkspaceError(
        "coding-agent workspace matches neither side of its durable delta"
    )


def reconstruct_edited_workspace(
    baseline: CodingAgentWorkspaceSnapshot,
    delta: CodingAgentWorkspaceDelta,
) -> CodingAgentWorkspaceSnapshot:
    """Reconstruct edited bytes from an exact baseline and durable delta."""

    if baseline.tree_hash != delta.baseline_tree_hash:
        raise CodingAgentWorkspaceError(
            "coding-agent delta cannot apply to another baseline"
        )
    files = baseline.by_path()
    for deleted in delta.deleted_files:
        observed = files.get(deleted.relative_path)
        if observed is None or observed.descriptor != deleted:
            raise CodingAgentWorkspaceError(
                "coding-agent deleted file differs from its baseline"
            )
        del files[deleted.relative_path]
    for change in delta.changed_files:
        observed = files.get(change.relative_path)
        if change.before is None:
            if observed is not None:
                raise CodingAgentWorkspaceError(
                    "coding-agent added file already exists in its baseline"
                )
        elif observed is None or observed.descriptor != change.before:
            raise CodingAgentWorkspaceError(
                "coding-agent changed file differs from its baseline"
            )
        files[change.relative_path] = CodingAgentWorkspaceFileSnapshot(
            descriptor=change.after,
            content=change.content,
        )
    reconstructed = CodingAgentWorkspaceSnapshot(
        tree_hash=_descriptor_tree_hash(
            {path: file.descriptor for path, file in files.items()}
        ),
        files=tuple(files[path] for path in sorted(files)),
    )
    if reconstructed.tree_hash != delta.edited_tree_hash:
        raise CodingAgentWorkspaceError(
            "coding-agent delta bytes do not reproduce its edited tree"
        )
    return reconstructed


def _read_workspace_directory(
    directory_descriptor: int,
    relative_root: PurePosixPath,
    state: _WorkspaceScanState,
) -> list[CodingAgentWorkspaceFileSnapshot]:
    with os.scandir(directory_descriptor) as iterator:
        entries = tuple(
            sorted(
                ((entry.name, entry.stat(follow_symlinks=False)) for entry in iterator),
                key=lambda item: item[0],
            )
        )
    files: list[CodingAgentWorkspaceFileSnapshot] = []
    for name, expected in entries:
        state.observe_entry()
        _validate_workspace_component(name)
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _metadata_identity(current) != _metadata_identity(expected):
            raise CodingAgentWorkspaceError(
                "coding-agent workspace entry changed during inspection"
            )
        relative_path = (
            PurePosixPath(name)
            if relative_root == PurePosixPath(".")
            else relative_root / name
        )
        if stat.S_ISDIR(expected.st_mode):
            if expected.st_mode & (0o022 | stat.S_ISUID | stat.S_ISGID):
                raise CodingAgentWorkspaceError(
                    "coding-agent workspace directory is writable outside its owner"
                )
            child_descriptor = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            with ExitStack() as child_descriptors:
                child_descriptors.callback(os.close, child_descriptor)
                opened = os.fstat(child_descriptor)
                if _metadata_identity(opened) != _metadata_identity(expected):
                    raise CodingAgentWorkspaceError(
                        "coding-agent workspace directory changed during inspection"
                    )
                files.extend(
                    _read_workspace_directory(
                        child_descriptor,
                        relative_path,
                        state,
                    )
                )
            current_child = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if _metadata_identity(current_child) != _metadata_identity(expected):
                raise CodingAgentWorkspaceError(
                    "coding-agent workspace directory changed during inspection"
                )
            continue
        files.append(
            _read_workspace_file(
                directory_descriptor,
                name,
                relative_path.as_posix(),
                expected,
                state,
            )
        )
    return files


def _read_workspace_file(
    directory_descriptor: int,
    name: str,
    relative_path: str,
    expected: os.stat_result,
    state: _WorkspaceScanState,
) -> CodingAgentWorkspaceFileSnapshot:
    permissions = stat.S_IMODE(expected.st_mode)
    if (
        not stat.S_ISREG(expected.st_mode)
        or expected.st_nlink != 1
        or permissions not in {0o600, 0o644, 0o700, 0o755}
        or expected.st_mode & (stat.S_ISUID | stat.S_ISGID)
    ):
        raise CodingAgentWorkspaceError(
            "coding-agent workspace entry is not an independent source file"
        )
    state.observe_file(expected.st_size)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if _metadata_identity(opened) != _metadata_identity(expected):
            raise CodingAgentWorkspaceError(
                "coding-agent workspace file changed during inspection"
            )
        content = handle.read()
        completed = os.fstat(handle.fileno())
        if (
            _metadata_observation(completed) != _metadata_observation(opened)
            or len(content) != opened.st_size
        ):
            raise CodingAgentWorkspaceError(
                "coding-agent workspace file changed while being read"
            )
    mode = "100755" if permissions in {0o700, 0o755} else "100644"
    return CodingAgentWorkspaceFileSnapshot(
        descriptor=SourceFileDescriptor(
            relative_path=relative_path,
            digest=tree_or_blob_digest(content),
            mode=mode,
            size=len(content),
        ),
        content=content,
    )


def _apply_delta_descriptors(
    baseline: dict[str, SourceFileDescriptor],
    delta: CodingAgentWorkspaceDelta,
) -> dict[str, SourceFileDescriptor]:
    transformed = dict(baseline)
    for deleted in delta.deleted_files:
        if transformed.get(deleted.relative_path) != deleted:
            raise CodingAgentWorkspaceError(
                "coding-agent deleted descriptor differs from its baseline"
            )
        del transformed[deleted.relative_path]
    for change in delta.changed_files:
        if transformed.get(change.relative_path) != change.before:
            raise CodingAgentWorkspaceError(
                "coding-agent changed descriptor differs from its baseline"
            )
        transformed[change.relative_path] = change.after
    return transformed


def _reverse_delta_descriptors(
    edited: dict[str, SourceFileDescriptor],
    delta: CodingAgentWorkspaceDelta,
) -> dict[str, SourceFileDescriptor]:
    transformed = dict(edited)
    for change in delta.changed_files:
        if transformed.get(change.relative_path) != change.after:
            raise CodingAgentWorkspaceError(
                "coding-agent changed descriptor differs from its edited tree"
            )
        if change.before is None:
            del transformed[change.relative_path]
        else:
            transformed[change.relative_path] = change.before
    for deleted in delta.deleted_files:
        if deleted.relative_path in transformed:
            raise CodingAgentWorkspaceError(
                "coding-agent deleted descriptor exists in its edited tree"
            )
        transformed[deleted.relative_path] = deleted
    return transformed


def _descriptor_tree_hash(files: dict[str, SourceFileDescriptor]) -> str:
    if not files:
        return EMPTY_EXPERT_TREE_DIGEST
    return source_tree_digest(
        {
            path: (descriptor.digest, descriptor.mode, descriptor.size)
            for path, descriptor in files.items()
        }
    )


def _validate_workspace_component(name: str) -> None:
    if (
        name in {"", ".", "..", ".git", ".env"}
        or name.startswith(".env.")
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise CodingAgentWorkspaceError("coding-agent workspace contains a denied path")


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _metadata_observation(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
