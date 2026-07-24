"""Descriptor-safe proof that the run workspace equals one checkpointed Git head."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import struct
import zlib
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.git_refs import git_tree_shas
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


@dataclass(frozen=True)
class _GitObject:
    kind: str
    payload: bytes


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
        entry_limit=settings.run_workspace_entry_limit,
        size_limit=settings.run_workspace_size_bytes,
    )
    descriptors_by_path, blob_ids = _scan_source_directory(
        workspace_descriptor,
        PurePosixPath("."),
        state,
        root=True,
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
            entry_limit=settings.run_workspace_git_entry_limit,
            size_limit=settings.run_workspace_git_metadata_size_bytes,
        )
        git_objects, branch_reference = _read_exact_git_closure(
            git_descriptor,
            settings=settings,
            state=git_state,
            descriptors=descriptors,
        )
        expected_head = f"ref: refs/heads/{settings.workspace_git_branch}\n".encode(
            "utf-8"
        )
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
            maximum_bytes=settings.run_workspace_git_metadata_size_bytes,
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
        branch=settings.workspace_git_branch,
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


def _scan_source_directory(
    directory_descriptor: int,
    relative_root: PurePosixPath,
    state: _SourceScanState,
    *,
    root: bool,
) -> tuple[dict[str, SourceFileDescriptor], dict[str, str]]:
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
                child_files, child_blobs = _scan_source_directory(
                    child_descriptor,
                    relative_path,
                    state,
                    root=False,
                )
            if not child_files:
                raise RunWorkspaceFrontierError(
                    "run workspace contains an untracked empty directory"
                )
            source_files.update(child_files)
            blob_ids.update(child_blobs)
            continue
        descriptor, blob_id = _read_source_file(
            directory_descriptor,
            name,
            relative_path.as_posix(),
            expected,
            state,
        )
        if descriptor.relative_path in source_files:
            raise RunWorkspaceFrontierError("run workspace source path is duplicated")
        source_files[descriptor.relative_path] = descriptor
        blob_ids[descriptor.relative_path] = blob_id
    return source_files, blob_ids


def _read_source_file(
    directory_descriptor: int,
    name: str,
    relative_path: str,
    expected: os.stat_result,
    state: _SourceScanState,
) -> tuple[SourceFileDescriptor, str]:
    permissions = stat.S_IMODE(expected.st_mode)
    if (
        not stat.S_ISREG(expected.st_mode)
        or expected.st_uid != os.geteuid()
        or expected.st_nlink != 1
        or permissions not in {0o600, 0o644, 0o700, 0o755}
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
    settings: LaunchSettings,
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
            maximum_bytes=settings.run_workspace_git_metadata_size_bytes,
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
        settings.workspace_git_branch,
        settings.run_workspace_git_metadata_size_bytes,
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
        settings.run_workspace_git_metadata_size_bytes,
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


__all__ = [
    "inspect_run_workspace_frontier",
    "RunWorkspaceFrontierError",
    "RunWorkspaceFrontierIdentity",
]
