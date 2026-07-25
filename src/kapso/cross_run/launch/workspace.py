"""Deterministic, atomic installation of one resolved cross-run launch."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import re
import stat
import struct
import tempfile
import zlib
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping
from weakref import WeakValueDictionary, finalize

from kapso.cross_run.canonical import (
    content_id,
    require_identifier,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.topology import (
    validate_expert_repository_topology,
    validate_expert_tree_ownership,
)
from kapso.cross_run.git_refs import git_object_sha, git_tree_shas
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackage
from kapso.cross_run.launch.checkpoint_contracts import RunCheckpointHead
from kapso.cross_run.launch.contracts import (
    BootstrapPin,
    LaunchManifest,
    LaunchWorkspaceLayout,
    WorkspaceInstallationReceipt,
    expected_launch_source_composition_hash,
)
from kapso.cross_run.launch.resolver import ResolvedLaunch
from kapso.cross_run.settings import CrossRunSettings, LaunchSettings

STARTER_WORKSPACE_INSTALLER_ID = "kapso_starter_workspace_builder"
STARTER_WORKSPACE_INSTALLER_VERSION = "kapso.starter_workspace_builder.v1"
_RENAME_NOREPLACE = 1
_GIT_INDEX_ENTRY_HEADER = struct.Struct("!LLLLLLLLLL20sH")
_GIT_INDEX_HEADER = struct.Struct("!4sLL")
_GIT_INDEX_PATH_LIMIT = 0xFFF
_GIT_COMMIT_TIME = "0 +0000"
_GIT_COMMIT_MESSAGE = b"Kapso launch baseline\n"
_RUN_CHECKPOINT_STAGING_PATTERN = re.compile(
    r"^checkpoint-[0-9a-f]{64}-[0-9a-f]{32}[.]tmp$"
)
_RUN_DERIVED_STATE_OBJECT_PATTERN = re.compile(r"^generation-[0-9a-f]{64}[.]bundle$")
_RUN_DERIVED_STATE_STAGING_PATTERN = re.compile(
    r"^generation-[0-9a-f]{64}-[0-9a-f]{32}[.]tmp$"
)
_RUN_ACTION_EVENT_PATTERN = re.compile(
    r"^operation-(?P<operation>[0-9a-f]{64})-event-[0-9]{4}[.]json$"
)
_RUN_ACTION_RESULT_PATTERN = re.compile(r"^result-[0-9a-f]{64}[.]blob$")
_RUN_ACTION_ACCEPTED_PATTERN = re.compile(r"^accepted-[0-9a-f]{64}[.]blob$")
_RUN_ACTION_INPUT_PATTERN = re.compile(r"^input-[0-9a-f]{64}[.]blob$")
_RUN_ACTION_STAGING_PATTERN = re.compile(
    r"^[.](?:accepted|event|input|result)-[0-9a-f]{32}[.]tmp$"
)
_RUN_ACTION_WORKSPACE_STAGING_PATTERN = re.compile(
    r"^(?:workspace|[.]workspace-[0-9a-f]{32}[.]tmp)$"
)


class _WorkspaceBuilderAuthority:
    pass


_WORKSPACE_BUILDER_AUTHORITY = _WorkspaceBuilderAuthority()
_ACTIVE_LAUNCH_AUTHORITY = object()
_ISSUED_PREPARED_WORKSPACES: WeakValueDictionary[int, object] = WeakValueDictionary()
_ISSUED_ACTIVE_WORKSPACES: WeakValueDictionary[int, object] = WeakValueDictionary()
_PREPARED_WORKSPACE_AUTHORITY_LOCK = Lock()


class LaunchWorkspaceError(RuntimeError):
    """A resolved launch could not become one exact durable workspace."""


def _required_layout_directory_paths(
    layout: LaunchWorkspaceLayout,
) -> tuple[PurePosixPath, ...]:
    required: set[PurePosixPath] = set()
    directory_roots = (
        PurePosixPath(layout.workspace_relative_path),
        PurePosixPath(layout.immutable_root_relative_path),
        PurePosixPath(layout.knowledge_snapshot_relative_path),
        PurePosixPath(layout.task_adapter_relative_path),
        PurePosixPath(layout.starting_artifacts_relative_path),
        PurePosixPath(layout.run_checkpoint_staging_relative_path),
        PurePosixPath(layout.run_derived_state_store_relative_path),
        PurePosixPath(layout.run_derived_state_staging_relative_path),
        PurePosixPath(layout.run_action_store_relative_path),
        PurePosixPath(layout.run_action_workspace_staging_relative_path),
        *(
            PurePosixPath(relative_path)
            for relative_path in layout.starting_artifact_roots.values()
        ),
    )
    control_paths = (
        PurePosixPath(layout.launch_manifest_relative_path),
        PurePosixPath(layout.bootstrap_pin_relative_path),
        PurePosixPath(layout.run_checkpoint_relative_path),
        PurePosixPath(layout.run_checkpoint_journal_relative_path),
        PurePosixPath(layout.run_checkpoint_lock_relative_path),
        PurePosixPath(layout.run_runtime_lock_relative_path),
        PurePosixPath(layout.run_idea_archive_relative_path),
        PurePosixPath(layout.run_experiment_history_relative_path),
        PurePosixPath(layout.run_execution_journal_relative_path),
        PurePosixPath(layout.run_action_ledger_relative_path),
    )
    for path in (*directory_roots, *control_paths):
        required.update(
            parent for parent in path.parents if parent != PurePosixPath(".")
        )
    required.update(directory_roots)
    return tuple(sorted(required, key=lambda path: (len(path.parts), path.as_posix())))


def _pinned_layout_directory_paths(
    layout: LaunchWorkspaceLayout,
) -> tuple[PurePosixPath, ...]:
    workspace_leaf = PurePosixPath(layout.workspace_relative_path)
    return tuple(
        path
        for path in _required_layout_directory_paths(layout)
        if path != workspace_leaf
    )


def _pinned_layout_directory_identities(
    layout: LaunchWorkspaceLayout,
    identities: Mapping[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    return {
        path.as_posix(): identities[path.as_posix()]
        for path in _pinned_layout_directory_paths(layout)
    }


def _require_inode_identity(
    value: tuple[int, int],
    name: str,
) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(part) is not int or part < 0 for part in value)
    ):
        raise LaunchWorkspaceError(f"{name} must be one inode identity")


def _open_real_root(
    path: Path,
    descriptors: ExitStack,
) -> int:
    metadata = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise LaunchWorkspaceError("published run root is unsafe")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, descriptor)
    reopened = os.fstat(descriptor)
    if (reopened.st_dev, reopened.st_ino) != (
        metadata.st_dev,
        metadata.st_ino,
    ):
        raise LaunchWorkspaceError("published run root changed while opening")
    return descriptor


def _open_layout_directories(
    root_descriptor: int,
    layout: LaunchWorkspaceLayout,
    descriptors: ExitStack,
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    opened: dict[str, int] = {}
    identities: dict[str, tuple[int, int]] = {}
    for path in _required_layout_directory_paths(layout):
        parent = path.parent
        parent_descriptor = (
            root_descriptor
            if parent == PurePosixPath(".")
            else opened[parent.as_posix()]
        )
        metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(metadata.st_mode):
            raise LaunchWorkspaceError(
                f"launch directory {path.as_posix()} is absent or unsafe"
            )
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        reopened = os.fstat(descriptor)
        identity = (reopened.st_dev, reopened.st_ino)
        if identity != (metadata.st_dev, metadata.st_ino):
            raise LaunchWorkspaceError(
                f"launch directory {path.as_posix()} changed while opening"
            )
        opened[path.as_posix()] = descriptor
        identities[path.as_posix()] = identity
    return opened, identities


def _require_owner_private_directory(
    descriptor: int,
    name: str,
) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise LaunchWorkspaceError(f"{name} is unsafe")
    return metadata.st_dev, metadata.st_ino


def _require_workspace_generation_directory(
    descriptor: int,
    receipt: WorkspaceInstallationReceipt,
    name: str,
) -> tuple[int, int]:
    identity = _require_owner_private_directory(descriptor, name)
    if identity[0] != receipt.run_action_workspace_staging_device:
        raise LaunchWorkspaceError(
            f"{name} is outside the workspace-promotion filesystem"
        )
    return identity


def _open_layout_file(
    root_descriptor: int,
    directory_descriptors: Mapping[str, int],
    relative_path: str,
) -> int:
    path = PurePosixPath(relative_path)
    parent_descriptor = (
        root_descriptor
        if path.parent == PurePosixPath(".")
        else directory_descriptors[path.parent.as_posix()]
    )
    metadata = os.stat(
        path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(metadata.st_mode):
        raise LaunchWorkspaceError(f"launch file {relative_path} is absent or unsafe")
    descriptor = os.open(
        path.name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    reopened = os.fstat(descriptor)
    if not stat.S_ISREG(reopened.st_mode) or (reopened.st_dev, reopened.st_ino) != (
        metadata.st_dev,
        metadata.st_ino,
    ):
        os.close(descriptor)
        raise LaunchWorkspaceError(f"launch file {relative_path} changed while opening")
    return descriptor


def _read_bounded_regular_file(
    path: Path,
    maximum_bytes: int,
    name: str,
    *,
    expected_size: int | None = None,
    expected_mode: int | None = None,
) -> bytes:
    metadata = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > maximum_bytes
        or (expected_size is not None and metadata.st_size != expected_size)
        or (
            expected_mode is not None
            and stat.S_IMODE(metadata.st_mode) != expected_mode
        )
    ):
        raise LaunchWorkspaceError(f"{name} is absent, oversized, or unsafe")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(descriptor, "rb") as handle:
        payload = handle.read(maximum_bytes + 1)
        reopened = os.fstat(handle.fileno())
    if len(payload) > maximum_bytes or (
        reopened.st_dev,
        reopened.st_ino,
        reopened.st_size,
        stat.S_IMODE(reopened.st_mode),
    ) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        stat.S_IMODE(metadata.st_mode),
    ):
        raise LaunchWorkspaceError(f"{name} changed while reading")
    return payload


@dataclass(frozen=True)
class _VerifiedWorkspaceClosure:
    """Exact local closure shared by fresh and resumed runtime authority."""

    run_root: Path
    workspace: Path
    bootstrap_pin: BootstrapPin
    published_root_identity: tuple[int, int]
    pinned_directory_identities: Mapping[str, tuple[int, int]]
    pinned_control_file_identities: Mapping[str, tuple[int, int]]
    verifier: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.verifier) is not StarterWorkspaceBuilder:
            raise LaunchWorkspaceError(
                "verified workspace closure requires its exact verifier"
            )
        object.__setattr__(
            self,
            "pinned_directory_identities",
            MappingProxyType(dict(self.pinned_directory_identities)),
        )
        object.__setattr__(
            self,
            "pinned_control_file_identities",
            MappingProxyType(dict(self.pinned_control_file_identities)),
        )


@dataclass
class _RuntimeDescriptorLease:
    """Process-local lifetime ownership of the root and runtime-lock descriptors."""

    root_descriptor: int
    runtime_lock_descriptor: int
    owner_process_id: int
    descriptors: ExitStack | None = None
    closed: bool = False

    def adopt(self, descriptors: ExitStack) -> None:
        if self.descriptors is not None or self.closed:
            raise LaunchWorkspaceError("runtime descriptor lease was already adopted")
        self.descriptors = descriptors

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.descriptors is not None:
            self.descriptors.close()
            self.descriptors = None


@dataclass(frozen=True)
class PreparedLaunchWorkspace:
    """Verified local paths returned only after atomic publication and reopen."""

    run_root: Path
    workspace: Path
    knowledge_snapshot: Path
    task_adapter: Path
    starting_artifacts: Mapping[str, Path]
    launch_manifest_path: Path
    bootstrap_pin_path: Path
    bootstrap_pin: BootstrapPin
    _builder_authority: object
    _published_root_identity: tuple[int, int]
    _pinned_directory_identities: Mapping[str, tuple[int, int]]
    _pinned_control_file_identities: Mapping[str, tuple[int, int]]
    _builder_verifier: object = field(repr=False, compare=False)
    _requires_initial_state: bool = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.bootstrap_pin) is not BootstrapPin
            or self._builder_authority is not _WORKSPACE_BUILDER_AUTHORITY
            or type(self._builder_verifier) is not StarterWorkspaceBuilder
            or type(self._requires_initial_state) is not bool
        ):
            raise LaunchWorkspaceError(
                "prepared launch requires live workspace-builder authority"
            )
        paths = (
            self.run_root,
            self.workspace,
            self.knowledge_snapshot,
            self.task_adapter,
            self.launch_manifest_path,
            self.bootstrap_pin_path,
            *self.starting_artifacts.values(),
        )
        if any(
            not path.is_absolute() or path != Path(os.path.abspath(path))
            for path in paths
        ):
            raise LaunchWorkspaceError(
                "prepared launch paths must be absolute and normalized"
            )
        layout = self.bootstrap_pin.installation_receipt.layout
        _require_inode_identity(
            self._published_root_identity,
            "prepared root identity",
        )
        expected_directory_paths = {
            path.as_posix() for path in _pinned_layout_directory_paths(layout)
        }
        expected_control_paths = {
            layout.launch_manifest_relative_path,
            layout.bootstrap_pin_relative_path,
            layout.run_checkpoint_journal_relative_path,
            layout.run_checkpoint_lock_relative_path,
            layout.run_runtime_lock_relative_path,
        }
        if (
            self.run_root.is_symlink()
            or not self.run_root.is_dir()
            or (
                self.run_root.stat(follow_symlinks=False).st_dev,
                self.run_root.stat(follow_symlinks=False).st_ino,
            )
            != self._published_root_identity
            or self.workspace != self.run_root / layout.workspace_relative_path
            or self.knowledge_snapshot
            != self.run_root / layout.knowledge_snapshot_relative_path
            or self.task_adapter != self.run_root / layout.task_adapter_relative_path
            or self.launch_manifest_path
            != self.run_root / layout.launch_manifest_relative_path
            or self.bootstrap_pin_path
            != self.run_root / layout.bootstrap_pin_relative_path
            or dict(self.starting_artifacts)
            != {
                artifact_id: self.run_root / relative_path
                for artifact_id, relative_path in layout.starting_artifact_roots.items()
            }
            or set(self._pinned_directory_identities) != expected_directory_paths
            or set(self._pinned_control_file_identities) != expected_control_paths
        ):
            raise LaunchWorkspaceError(
                "prepared launch paths differ from the pinned workspace layout"
            )
        object.__setattr__(
            self,
            "starting_artifacts",
            MappingProxyType(dict(self.starting_artifacts)),
        )
        for name, identities in (
            (
                "prepared directory identity",
                self._pinned_directory_identities.values(),
            ),
            (
                "prepared control-file identity",
                self._pinned_control_file_identities.values(),
            ),
        ):
            for identity in identities:
                _require_inode_identity(identity, name)
        object.__setattr__(
            self,
            "_pinned_directory_identities",
            MappingProxyType(dict(self._pinned_directory_identities)),
        )
        object.__setattr__(
            self,
            "_pinned_control_file_identities",
            MappingProxyType(dict(self._pinned_control_file_identities)),
        )

    def activate(self) -> "ActiveLaunchWorkspace":
        identity = id(self)
        with _PREPARED_WORKSPACE_AUTHORITY_LOCK:
            issued = _ISSUED_PREPARED_WORKSPACES.pop(identity, None)
        if (
            self._builder_authority is not _WORKSPACE_BUILDER_AUTHORITY
            or issued is not self
        ):
            raise LaunchWorkspaceError(
                "prepared launch lacks live workspace-builder authority"
            )
        return self._builder_verifier._activate_prepared(self)

    def _require_verified_identity(
        self,
        verified: "PreparedLaunchWorkspace",
    ) -> None:
        if (
            verified._published_root_identity != self._published_root_identity
            or dict(verified._pinned_directory_identities)
            != dict(self._pinned_directory_identities)
            or dict(verified._pinned_control_file_identities)
            != dict(self._pinned_control_file_identities)
            or verified._requires_initial_state != self._requires_initial_state
        ):
            raise LaunchWorkspaceError(
                "prepared launch filesystem changed before authority consumption"
            )

    def _require_filesystem_identity(self) -> None:
        layout = self.bootstrap_pin.installation_receipt.layout
        with ExitStack() as descriptors:
            root_descriptor = _open_real_root(self.run_root, descriptors)
            if (
                StarterWorkspaceBuilder._directory_identity(root_descriptor)
                != self._published_root_identity
            ):
                raise LaunchWorkspaceError(
                    "prepared run root no longer names its published inode"
                )
            opened, identities = _open_layout_directories(
                root_descriptor,
                layout,
                descriptors,
            )
            workspace_descriptor = opened[layout.workspace_relative_path]
            _require_workspace_generation_directory(
                workspace_descriptor,
                self.bootstrap_pin.installation_receipt,
                "prepared execution workspace",
            )
            if _pinned_layout_directory_identities(
                layout,
                identities,
            ) != dict(self._pinned_directory_identities):
                raise LaunchWorkspaceError(
                    "prepared launch directories changed after publication"
                )
            for (
                relative_path,
                expected_identity,
            ) in self._pinned_control_file_identities.items():
                descriptor = _open_layout_file(
                    root_descriptor,
                    opened,
                    relative_path,
                )
                descriptors.callback(os.close, descriptor)
                metadata = os.fstat(descriptor)
                if (metadata.st_dev, metadata.st_ino) != expected_identity:
                    raise LaunchWorkspaceError(
                        "prepared launch control file changed after publication"
                    )


def _close_active_workspace(
    identity: int,
    lifecycle: _RuntimeDescriptorLease,
) -> None:
    with _PREPARED_WORKSPACE_AUTHORITY_LOCK:
        _ISSUED_ACTIVE_WORKSPACES.pop(identity, None)
    lifecycle.close()


def _require_workspace_closure(
    closure: _VerifiedWorkspaceClosure,
    root_descriptor: int,
) -> None:
    if (
        StarterWorkspaceBuilder._directory_identity(root_descriptor)
        != closure.published_root_identity
    ):
        raise LaunchWorkspaceError("active run root no longer names its verified inode")
    layout = closure.bootstrap_pin.installation_receipt.layout
    with ExitStack() as descriptors:
        opened, identities = _open_layout_directories(
            root_descriptor,
            layout,
            descriptors,
        )
        _require_workspace_generation_directory(
            opened[layout.workspace_relative_path],
            closure.bootstrap_pin.installation_receipt,
            "active execution workspace",
        )
        if _pinned_layout_directory_identities(
            layout,
            identities,
        ) != dict(closure.pinned_directory_identities):
            raise LaunchWorkspaceError(
                "active launch directories changed after verification"
            )
        for (
            relative_path,
            expected_identity,
        ) in closure.pinned_control_file_identities.items():
            descriptor = _open_layout_file(
                root_descriptor,
                opened,
                relative_path,
            )
            descriptors.callback(os.close, descriptor)
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) != expected_identity:
                raise LaunchWorkspaceError(
                    "active launch control file changed after verification"
                )


@dataclass(frozen=True)
class ActiveLaunchWorkspace:
    """Process-bound runtime authority retaining the root and lifetime lock."""

    run_root: Path
    workspace: Path
    bootstrap_pin: BootstrapPin
    published_root_identity: tuple[int, int]
    _closure: _VerifiedWorkspaceClosure = field(repr=False, compare=False)
    _lifecycle: _RuntimeDescriptorLease = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)
    _finalizer: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self._closure) is not _VerifiedWorkspaceClosure
            or type(self._lifecycle) is not _RuntimeDescriptorLease
            or self._authority is not _ACTIVE_LAUNCH_AUTHORITY
            or self.run_root != self._closure.run_root
            or self.workspace != self._closure.workspace
            or self.bootstrap_pin != self._closure.bootstrap_pin
            or self.published_root_identity != self._closure.published_root_identity
            or self._lifecycle.owner_process_id != os.getpid()
            or self._lifecycle.closed
        ):
            raise LaunchWorkspaceError(
                "active launch requires consumed runtime authority"
            )
        _require_workspace_closure(
            self._closure,
            self._lifecycle.root_descriptor,
        )

    def __enter__(self) -> "ActiveLaunchWorkspace":
        self.require_control_authority()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        if not hasattr(self, "_finalizer"):
            raise LaunchWorkspaceError("active launch control authority is invalid")
        with _PREPARED_WORKSPACE_AUTHORITY_LOCK:
            issued = _ISSUED_ACTIVE_WORKSPACES.get(id(self))
        if issued is not self:
            raise LaunchWorkspaceError("active launch control authority is invalid")
        self._finalizer()

    def _require_live_authority(self) -> None:
        with _PREPARED_WORKSPACE_AUTHORITY_LOCK:
            issued = _ISSUED_ACTIVE_WORKSPACES.get(id(self))
        if (
            self._authority is not _ACTIVE_LAUNCH_AUTHORITY
            or issued is not self
            or self._lifecycle.closed
            or self._lifecycle.descriptors is None
            or self._lifecycle.owner_process_id != os.getpid()
        ):
            raise LaunchWorkspaceError("active launch control authority is invalid")
        root_metadata = os.fstat(self._lifecycle.root_descriptor)
        lock_metadata = os.fstat(self._lifecycle.runtime_lock_descriptor)
        layout = self.bootstrap_pin.installation_receipt.layout
        expected_lock_identity = self._closure.pinned_control_file_identities[
            layout.run_runtime_lock_relative_path
        ]
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (root_metadata.st_dev, root_metadata.st_ino)
            != self.published_root_identity
            or not stat.S_ISREG(lock_metadata.st_mode)
            or (lock_metadata.st_dev, lock_metadata.st_ino) != expected_lock_identity
        ):
            raise LaunchWorkspaceError(
                "active launch retained descriptors changed during use"
            )

    def require_control_authority(self) -> None:
        self._require_live_authority()
        with ExitStack() as descriptors:
            root_descriptor = self._open_run_root(descriptors)
            _require_workspace_closure(self._closure, root_descriptor)
            self._closure.verifier._verify_outer_run_root_closure(
                self._closure.verifier._descriptor_path(root_descriptor),
                self.bootstrap_pin.installation_receipt.layout,
                self.published_root_identity,
            )
        self._require_live_authority()

    def require_launch_settings(self, settings: object) -> None:
        if type(settings) is not LaunchSettings:
            raise LaunchWorkspaceError(
                "active launch settings require the exact settings type"
            )
        if self.bootstrap_pin.installation_receipt.launch_settings_id != content_id(
            "launch-settings",
            settings.to_dict(),
        ):
            raise LaunchWorkspaceError(
                "active launch settings differ from the bootstrap receipt"
            )
        self.require_control_authority()

    def _open_run_root(self, descriptors: ExitStack) -> int:
        """Duplicate the lifetime-pinned root without reopening public descendants."""
        self._require_live_authority()
        public_identity = StarterWorkspaceBuilder._path_directory_identity(
            self.run_root
        )
        if public_identity != self.published_root_identity:
            raise LaunchWorkspaceError(
                "active run-root pathname differs from its retained inode"
            )
        descriptor = os.dup(self._lifecycle.root_descriptor)
        os.set_inheritable(descriptor, False)
        descriptors.callback(os.close, descriptor)
        if (
            StarterWorkspaceBuilder._directory_identity(descriptor)
            != self.published_root_identity
        ):
            raise LaunchWorkspaceError(
                "active run root changed before descriptor duplication"
            )
        return descriptor

    def _open_workspace_path(
        self,
        descriptors: ExitStack,
    ) -> tuple[int, tuple[int, int]]:
        """Open the current workspace through pinned ancestors."""
        root_descriptor = self._open_run_root(descriptors)
        current_descriptor = root_descriptor
        current_path = PurePosixPath(".")
        layout = self.bootstrap_pin.installation_receipt.layout
        workspace_parts = PurePosixPath(layout.workspace_relative_path).parts
        for position, component in enumerate(workspace_parts):
            child_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current_descriptor,
            )
            descriptors.callback(os.close, child_descriptor)
            current_path = (
                PurePosixPath(component)
                if current_path == PurePosixPath(".")
                else current_path / component
            )
            metadata = os.fstat(child_descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise LaunchWorkspaceError(
                    "execution workspace path changed after launch"
                )
            if position == len(workspace_parts) - 1:
                _require_workspace_generation_directory(
                    child_descriptor,
                    self.bootstrap_pin.installation_receipt,
                    "execution workspace",
                )
            else:
                expected_identity = self._closure.pinned_directory_identities.get(
                    current_path.as_posix()
                )
                if (
                    expected_identity is None
                    or (
                        metadata.st_dev,
                        metadata.st_ino,
                    )
                    != expected_identity
                ):
                    raise LaunchWorkspaceError(
                        "execution workspace ancestor changed after launch"
                    )
            current_descriptor = child_descriptor
        identity = StarterWorkspaceBuilder._directory_identity(
            current_descriptor,
        )
        return current_descriptor, identity

    def _open_execution_workspace(
        self,
        descriptors: ExitStack,
    ) -> tuple[int, tuple[int, int]]:
        """Open the current replaceable workspace generation by descriptor."""
        current_descriptor, identity = self._open_workspace_path(descriptors)
        self.require_control_authority()
        with ExitStack() as verification_descriptors:
            _verified_descriptor, verified_identity = self._open_workspace_path(
                verification_descriptors
            )
            if verified_identity != identity:
                raise LaunchWorkspaceError(
                    "execution workspace changed while acquiring authority"
                )
        return current_descriptor, identity

    def _open_run_action_store(
        self,
        descriptors: ExitStack,
    ) -> tuple[int, tuple[int, int]]:
        """Open the receipt-pinned create-only action store by descriptor."""
        root_descriptor = self._open_run_root(descriptors)
        opened, identities = _open_layout_directories(
            root_descriptor,
            self.bootstrap_pin.installation_receipt.layout,
            descriptors,
        )
        relative_path = (
            self.bootstrap_pin.installation_receipt.layout.run_action_store_relative_path
        )
        descriptor = opened[relative_path]
        identity = identities[relative_path]
        receipt = self.bootstrap_pin.installation_receipt
        if identity != (
            receipt.run_action_store_device,
            receipt.run_action_store_inode,
        ):
            raise LaunchWorkspaceError(
                "active run action store differs from its receipt"
            )
        self.require_control_authority()
        return descriptor, identity

    def _open_run_action_workspace_staging(
        self,
        descriptors: ExitStack,
    ) -> tuple[int, tuple[int, int]]:
        """Open the receipt-pinned workspace-promotion staging root."""
        root_descriptor = self._open_run_root(descriptors)
        opened, identities = _open_layout_directories(
            root_descriptor,
            self.bootstrap_pin.installation_receipt.layout,
            descriptors,
        )
        relative_path = (
            self.bootstrap_pin.installation_receipt.layout.run_action_workspace_staging_relative_path
        )
        descriptor = opened[relative_path]
        identity = identities[relative_path]
        receipt = self.bootstrap_pin.installation_receipt
        if (
            identity
            != (
                receipt.run_action_workspace_staging_device,
                receipt.run_action_workspace_staging_inode,
            )
            or _require_owner_private_directory(
                descriptor,
                "active run action workspace staging root",
            )
            != identity
        ):
            raise LaunchWorkspaceError(
                "active run action workspace staging root differs from its receipt"
            )
        self.require_control_authority()
        return descriptor, identity

    def _require_execution_workspace(
        self,
        descriptor: int,
        identity: tuple[int, int],
    ) -> None:
        """Reprove that an execution lease still names the public workspace."""
        _require_inode_identity(identity, "active execution workspace")
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise LaunchWorkspaceError("active execution workspace changed during use")
        self.require_control_authority()
        with ExitStack() as descriptors:
            _current_descriptor, current_identity = self._open_workspace_path(
                descriptors
            )
            if current_identity != identity:
                raise LaunchWorkspaceError(
                    "active execution workspace is no longer public"
                )


def _issue_active_workspace(active: ActiveLaunchWorkspace) -> None:
    if (
        type(active) is not ActiveLaunchWorkspace
        or active._authority is not _ACTIVE_LAUNCH_AUTHORITY
    ):
        raise LaunchWorkspaceError(
            "only consumed workspace authority may activate a launch"
        )
    with _PREPARED_WORKSPACE_AUTHORITY_LOCK:
        if id(active) in _ISSUED_ACTIVE_WORKSPACES:
            raise LaunchWorkspaceError("active launch was already issued")
        _ISSUED_ACTIVE_WORKSPACES[id(active)] = active
    object.__setattr__(
        active,
        "_finalizer",
        finalize(
            active,
            _close_active_workspace,
            id(active),
            active._lifecycle,
        ),
    )


def _issue_prepared_workspace(
    prepared: PreparedLaunchWorkspace,
) -> None:
    if (
        type(prepared) is not PreparedLaunchWorkspace
        or prepared._builder_authority is not _WORKSPACE_BUILDER_AUTHORITY
    ):
        raise LaunchWorkspaceError(
            "only the workspace builder may issue a prepared launch"
        )
    prepared._require_filesystem_identity()
    with _PREPARED_WORKSPACE_AUTHORITY_LOCK:
        if id(prepared) in _ISSUED_PREPARED_WORKSPACES:
            raise LaunchWorkspaceError("prepared launch was already issued")
        _ISSUED_PREPARED_WORKSPACES[id(prepared)] = prepared


def _invalidate_inherited_workspace_authority() -> None:
    global _ISSUED_ACTIVE_WORKSPACES
    global _ISSUED_PREPARED_WORKSPACES
    global _PREPARED_WORKSPACE_AUTHORITY_LOCK

    for active in tuple(_ISSUED_ACTIVE_WORKSPACES.values()):
        active._lifecycle.close()
    _ISSUED_ACTIVE_WORKSPACES = WeakValueDictionary()
    _ISSUED_PREPARED_WORKSPACES = WeakValueDictionary()
    _PREPARED_WORKSPACE_AUTHORITY_LOCK = Lock()


os.register_at_fork(after_in_child=_invalidate_inherited_workspace_authority)


@dataclass(frozen=True)
class _GitLeaf:
    mode: str
    payload: bytes
    path: Path


class StarterWorkspaceBuilder:
    """Consume one resolved authority and atomically publish its complete run root."""

    def __init__(self, settings: CrossRunSettings) -> None:
        if type(settings) is not CrossRunSettings:
            raise LaunchWorkspaceError(
                "workspace builder requires exact cross-run settings"
            )
        self._settings = settings

    def reopen(self, run_root: Path) -> ActiveLaunchWorkspace:
        """Reconstruct one exclusive runtime from its durable local bootstrap pin."""
        with ExitStack() as descriptors:
            normalized_run_root, root_descriptor = self._open_existing_run_root(
                run_root,
                descriptors,
            )
            pin_payload = self._read_configured_bootstrap_pin(
                root_descriptor,
                descriptors,
            )
            pin = BootstrapPin.from_json_bytes(pin_payload)
            if (
                pin.to_json_bytes() != pin_payload
                or pin.installation_receipt.layout != self._layout(pin.launch_manifest)
                or pin.installation_receipt.launch_settings_id
                != content_id(
                    "launch-settings",
                    self._settings.launch.to_dict(),
                )
            ):
                raise LaunchWorkspaceError(
                    "reopened bootstrap pin differs from configured launch authority"
                )
            root_identity = self._directory_identity(root_descriptor)
        return self._activate_workspace(
            normalized_run_root,
            pin,
            expected_root_identity=root_identity,
            requires_initial_state=False,
            expected_prepared=None,
        )

    def _activate_prepared(
        self,
        prepared: PreparedLaunchWorkspace,
    ) -> ActiveLaunchWorkspace:
        if (
            type(prepared) is not PreparedLaunchWorkspace
            or prepared._builder_verifier is not self
        ):
            raise LaunchWorkspaceError(
                "workspace activation requires its exact prepared launch"
            )
        return self._activate_workspace(
            prepared.run_root,
            prepared.bootstrap_pin,
            expected_root_identity=prepared._published_root_identity,
            requires_initial_state=prepared._requires_initial_state,
            expected_prepared=prepared,
        )

    def _activate_workspace(
        self,
        run_root: Path,
        bootstrap_pin: BootstrapPin,
        *,
        expected_root_identity: tuple[int, int],
        requires_initial_state: bool,
        expected_prepared: PreparedLaunchWorkspace | None,
    ) -> ActiveLaunchWorkspace:
        with ExitStack() as descriptors:
            normalized_run_root, root_descriptor = self._open_existing_run_root(
                run_root,
                descriptors,
            )
            if (
                normalized_run_root != run_root
                or self._directory_identity(root_descriptor) != expected_root_identity
            ):
                raise LaunchWorkspaceError("run root changed before runtime activation")
            layout = bootstrap_pin.installation_receipt.layout
            opened_directories, _identities = _open_layout_directories(
                root_descriptor,
                layout,
                descriptors,
            )
            runtime_lock_path = PurePosixPath(layout.run_runtime_lock_relative_path)
            runtime_lock_parent = (
                root_descriptor
                if runtime_lock_path.parent == PurePosixPath(".")
                else opened_directories[runtime_lock_path.parent.as_posix()]
            )
            runtime_lock_descriptor = os.open(
                runtime_lock_path.name,
                os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=runtime_lock_parent,
            )
            descriptors.callback(os.close, runtime_lock_descriptor)
            runtime_lock_metadata = os.fstat(runtime_lock_descriptor)
            receipt = bootstrap_pin.installation_receipt
            if (
                not stat.S_ISREG(runtime_lock_metadata.st_mode)
                or runtime_lock_metadata.st_uid != os.geteuid()
                or runtime_lock_metadata.st_nlink != 1
                or runtime_lock_metadata.st_size != 0
                or stat.S_IMODE(runtime_lock_metadata.st_mode) != 0o600
                or (
                    runtime_lock_metadata.st_dev,
                    runtime_lock_metadata.st_ino,
                )
                != (
                    receipt.run_runtime_lock_device,
                    receipt.run_runtime_lock_inode,
                )
            ):
                raise LaunchWorkspaceError(
                    "runtime lock differs from its bootstrap receipt"
                )
            fcntl.flock(
                runtime_lock_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            verified = self._verify_published(
                self._descriptor_path(root_descriptor),
                bootstrap_pin,
                exposed_run_root=run_root,
                expected_root_identity=expected_root_identity,
                root_descriptor=root_descriptor,
                requires_initial_state=requires_initial_state,
            )
            if expected_prepared is not None:
                expected_prepared._require_verified_identity(verified)
            terminal_verified = StarterWorkspaceBuilder._verify_published(
                self,
                self._descriptor_path(root_descriptor),
                bootstrap_pin,
                exposed_run_root=run_root,
                expected_root_identity=expected_root_identity,
                root_descriptor=root_descriptor,
                requires_initial_state=requires_initial_state,
            )
            verified._require_verified_identity(terminal_verified)
            closure = _VerifiedWorkspaceClosure(
                run_root=terminal_verified.run_root,
                workspace=terminal_verified.workspace,
                bootstrap_pin=terminal_verified.bootstrap_pin,
                published_root_identity=(terminal_verified._published_root_identity),
                pinned_directory_identities=(
                    terminal_verified._pinned_directory_identities
                ),
                pinned_control_file_identities=(
                    terminal_verified._pinned_control_file_identities
                ),
                verifier=self,
            )
            with ExitStack() as retained_descriptors:
                retained_root_descriptor = os.dup(root_descriptor)
                os.set_inheritable(retained_root_descriptor, False)
                retained_descriptors.callback(
                    os.close,
                    retained_root_descriptor,
                )
                retained_runtime_lock_descriptor = os.dup(runtime_lock_descriptor)
                os.set_inheritable(
                    retained_runtime_lock_descriptor,
                    False,
                )
                retained_descriptors.callback(
                    os.close,
                    retained_runtime_lock_descriptor,
                )
                lifecycle = _RuntimeDescriptorLease(
                    root_descriptor=retained_root_descriptor,
                    runtime_lock_descriptor=(retained_runtime_lock_descriptor),
                    owner_process_id=os.getpid(),
                )
                active = ActiveLaunchWorkspace(
                    run_root=terminal_verified.run_root,
                    workspace=terminal_verified.workspace,
                    bootstrap_pin=terminal_verified.bootstrap_pin,
                    published_root_identity=(
                        terminal_verified._published_root_identity
                    ),
                    _closure=closure,
                    _lifecycle=lifecycle,
                    _authority=_ACTIVE_LAUNCH_AUTHORITY,
                )
                _issue_active_workspace(active)
                lifecycle.adopt(retained_descriptors.pop_all())
        active.require_control_authority()
        return active

    def build(
        self,
        resolved_launch: ResolvedLaunch,
        run_root: Path,
        *,
        run_id: str,
        campaign_id: str,
    ) -> PreparedLaunchWorkspace:
        if type(resolved_launch) is not ResolvedLaunch:
            raise LaunchWorkspaceError(
                "workspace builder requires one resolved launch authority"
            )
        require_identifier(run_id, "workspace run_id")
        require_identifier(campaign_id, "workspace campaign_id")
        normalized_run_root, parent_descriptor, parent_identity = (
            self._open_destination_parent(run_root)
        )
        with ExitStack() as descriptors:
            descriptors.callback(os.close, parent_descriptor)
            resolved_launch.require_resolver_authority()
            staging_name = Path(
                tempfile.mkdtemp(
                    prefix=".launch-staging-",
                    dir=self._descriptor_path(parent_descriptor),
                )
            ).name
            staging_descriptor = os.open(
                staging_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_descriptor,
            )
            descriptors.callback(os.close, staging_descriptor)
            staging_identity = self._directory_identity(staging_descriptor)
            staging = self._descriptor_path(staging_descriptor)
            manifest = resolved_launch.manifest
            layout = self._layout(manifest)
            paths = self._materialized_paths(staging, layout)

            self._validate_expert_source(resolved_launch)
            self._write_tree(
                paths["workspace"],
                manifest.expert_source.extraction_receipt.source_tree_files,
                resolved_launch.expert_source.source_contents,
                read_only=False,
                name="expert source",
            )
            knowledge_tree_hash = self._write_knowledge_snapshot(
                paths["knowledge_snapshot"],
                resolved_launch.knowledge_package,
            )
            adapter_runtime_tree_hash = self._write_task_adapter(
                paths["task_adapter"],
                resolved_launch,
            )
            self._write_starting_artifacts(
                staging,
                layout,
                resolved_launch,
            )
            self._make_directory_read_only(
                staging / layout.immutable_root_relative_path
            )
            expected_composition = expected_launch_source_composition_hash(
                expert_source_tree_hash=manifest.expert_manifest.candidate_tree_hash,
                expert_repository_map=manifest.expert_repository_map,
                task_adapter=manifest.task_adapter,
                starting_artifacts=manifest.starting_artifacts,
            )
            if expected_composition != manifest.expected_source_composition_hash:
                raise LaunchWorkspaceError(
                    "staged launch source composition differs from its manifest"
                )
            (
                baseline_tree_sha,
                baseline_commit_sha,
                git_index_digest,
                git_object_ids,
            ) = self._initialize_git_baseline(
                paths["workspace"],
                manifest.expert_source.extraction_receipt.source_tree_files,
                resolved_launch.expert_source.source_contents,
            )
            checkpoint_journal_path = (
                staging / layout.run_checkpoint_journal_relative_path
            )
            checkpoint_lock_path = staging / layout.run_checkpoint_lock_relative_path
            runtime_lock_path = staging / layout.run_runtime_lock_relative_path
            self._write_plain_file(
                checkpoint_journal_path,
                b"",
                mode=0o600,
            )
            self._write_plain_file(
                checkpoint_lock_path,
                b"",
                mode=0o600,
            )
            self._write_plain_file(
                runtime_lock_path,
                b"",
                mode=0o600,
            )
            action_store_path = staging / layout.run_action_store_relative_path
            action_store_path.mkdir(parents=True, mode=0o700)
            action_store_path.chmod(0o700)
            action_workspace_staging_path = (
                staging / layout.run_action_workspace_staging_relative_path
            )
            action_workspace_staging_path.mkdir(parents=True, mode=0o700)
            action_workspace_staging_path.chmod(0o700)
            action_registry_lock_path = action_store_path / "registry.lock"
            action_workspace_lock_path = action_store_path / "workspace.lock"
            self._write_plain_file(
                action_registry_lock_path,
                b"",
                mode=0o600,
            )
            self._write_plain_file(
                action_workspace_lock_path,
                b"",
                mode=0o600,
            )
            checkpoint_journal_metadata = checkpoint_journal_path.stat(
                follow_symlinks=False
            )
            checkpoint_lock_metadata = checkpoint_lock_path.stat(follow_symlinks=False)
            action_store_metadata = action_store_path.stat(follow_symlinks=False)
            action_workspace_staging_metadata = action_workspace_staging_path.stat(
                follow_symlinks=False
            )
            action_registry_lock_metadata = action_registry_lock_path.stat(
                follow_symlinks=False
            )
            action_workspace_lock_metadata = action_workspace_lock_path.stat(
                follow_symlinks=False
            )
            runtime_lock_metadata = runtime_lock_path.stat(follow_symlinks=False)
            launch_settings_id = content_id(
                "launch-settings",
                self._settings.launch.to_dict(),
            )

            installation = WorkspaceInstallationReceipt.mint(
                launch_manifest_id=manifest.launch_manifest_id,
                launch_manifest_full_digest=manifest.full_digest,
                run_id=run_id,
                campaign_id=campaign_id,
                layout=layout,
                expert_source_tree_hash=manifest.expert_manifest.candidate_tree_hash,
                knowledge_package_tree_hash=knowledge_tree_hash,
                task_adapter_runtime_tree_hash=adapter_runtime_tree_hash,
                starting_artifact_materialization_receipt_id=(
                    manifest.starting_artifacts.materialization_receipt_id
                ),
                starting_artifact_tree_hashes={
                    artifact.starting_artifact_content_id: (
                        artifact.materialized_tree_hash
                    )
                    for artifact in manifest.starting_artifacts.starting_artifacts
                },
                expected_source_composition_hash=expected_composition,
                workspace_git_branch=self._settings.launch.workspace_git_branch,
                workspace_baseline_commit_sha=baseline_commit_sha,
                workspace_baseline_tree_sha=baseline_tree_sha,
                workspace_git_index_digest=git_index_digest,
                workspace_git_object_ids=git_object_ids,
                launch_settings_id=launch_settings_id,
                run_checkpoint_journal_device=(checkpoint_journal_metadata.st_dev),
                run_checkpoint_journal_inode=(checkpoint_journal_metadata.st_ino),
                run_checkpoint_lock_device=checkpoint_lock_metadata.st_dev,
                run_checkpoint_lock_inode=checkpoint_lock_metadata.st_ino,
                run_action_store_device=action_store_metadata.st_dev,
                run_action_store_inode=action_store_metadata.st_ino,
                run_action_workspace_staging_device=(
                    action_workspace_staging_metadata.st_dev
                ),
                run_action_workspace_staging_inode=(
                    action_workspace_staging_metadata.st_ino
                ),
                run_action_registry_lock_device=(action_registry_lock_metadata.st_dev),
                run_action_registry_lock_inode=(action_registry_lock_metadata.st_ino),
                run_action_workspace_lock_device=(
                    action_workspace_lock_metadata.st_dev
                ),
                run_action_workspace_lock_inode=(action_workspace_lock_metadata.st_ino),
                run_runtime_lock_device=runtime_lock_metadata.st_dev,
                run_runtime_lock_inode=runtime_lock_metadata.st_ino,
                installer_id=STARTER_WORKSPACE_INSTALLER_ID,
                installer_version=STARTER_WORKSPACE_INSTALLER_VERSION,
                exact_dependency_ids=tuple(
                    sorted(
                        {
                            manifest.launch_manifest_id,
                            manifest.starting_artifacts.materialization_receipt_id,
                            launch_settings_id,
                        }
                    )
                ),
            )
            pin = BootstrapPin.mint(
                launch_manifest=manifest,
                launch_manifest_full_digest=manifest.full_digest,
                installation_receipt=installation,
                exact_dependency_ids=tuple(
                    sorted(
                        {
                            manifest.launch_manifest_id,
                            installation.workspace_installation_receipt_id,
                        }
                    )
                ),
            )
            manifest_bytes = manifest.to_json_bytes()
            pin_bytes = pin.to_json_bytes()
            self._require_control_file_bound(
                manifest_bytes,
                self._settings.launch.launch_manifest_size_bytes,
                "launch manifest",
            )
            self._require_control_file_bound(
                pin_bytes,
                self._settings.launch.bootstrap_pin_size_bytes,
                "bootstrap pin",
            )
            self._write_control_file(
                staging / layout.launch_manifest_relative_path,
                manifest_bytes,
            )
            self._write_control_file(
                staging / layout.bootstrap_pin_relative_path,
                pin_bytes,
            )
            initial_checkpoint_head = (
                RunCheckpointHead.initial(pin).to_json_bytes() + b"\n"
            )
            self._require_control_file_bound(
                initial_checkpoint_head,
                self._settings.launch.run_checkpoint_journal_size_bytes,
                "run checkpoint journal",
            )
            self._append_plain_file(
                checkpoint_journal_path,
                initial_checkpoint_head,
            )
            checkpoint_staging = staging / layout.run_checkpoint_staging_relative_path
            checkpoint_staging.mkdir(parents=True, mode=0o700)
            checkpoint_staging.chmod(0o700)
            for derived_state_directory in (
                layout.run_derived_state_store_relative_path,
                layout.run_derived_state_staging_relative_path,
            ):
                directory = staging / derived_state_directory
                directory.mkdir(parents=True, mode=0o700)
                directory.chmod(0o700)
            self._fsync_tree(staging)
            self._require_parent_identity(
                normalized_run_root.parent,
                parent_descriptor,
                parent_identity,
            )
            final_descriptor = self._rename_no_replace(
                parent_descriptor,
                staging_name,
                normalized_run_root.name,
                staging_identity,
            )
            descriptors.callback(os.close, final_descriptor)
            os.fsync(parent_descriptor)
            prepared = self._verify_published(
                self._descriptor_path(final_descriptor),
                pin,
                exposed_run_root=normalized_run_root,
                expected_root_identity=staging_identity,
                root_descriptor=final_descriptor,
                requires_initial_state=True,
            )
            self._require_published_identity(
                parent_descriptor,
                normalized_run_root.name,
                staging_identity,
            )
            _issue_prepared_workspace(prepared)
            return prepared

    def _layout(self, manifest: LaunchManifest) -> LaunchWorkspaceLayout:
        launch = self._settings.launch
        starting_root = PurePosixPath(launch.starting_artifacts_path)
        artifact_roots = {
            artifact.starting_artifact_content_id: (
                starting_root / artifact.starting_artifact_content_id.rsplit(":", 1)[1]
            ).as_posix()
            for artifact in manifest.starting_artifacts.starting_artifacts
        }
        return LaunchWorkspaceLayout(
            workspace_relative_path=launch.workspace_path,
            immutable_root_relative_path=launch.immutable_root_path,
            knowledge_snapshot_relative_path=launch.knowledge_snapshot_path,
            task_adapter_relative_path=launch.task_adapter_path,
            starting_artifacts_relative_path=launch.starting_artifacts_path,
            starting_artifact_roots=artifact_roots,
            launch_manifest_relative_path=launch.launch_manifest_path,
            bootstrap_pin_relative_path=launch.bootstrap_pin_path,
            run_checkpoint_relative_path=launch.run_checkpoint_path,
            run_checkpoint_journal_relative_path=(launch.run_checkpoint_journal_path),
            run_checkpoint_lock_relative_path=launch.run_checkpoint_lock_path,
            run_checkpoint_staging_relative_path=(launch.run_checkpoint_staging_path),
            run_idea_archive_relative_path=launch.run_idea_archive_path,
            run_experiment_history_relative_path=(launch.run_experiment_history_path),
            run_execution_journal_relative_path=launch.run_execution_journal_path,
            run_derived_state_store_relative_path=(launch.run_derived_state_store_path),
            run_derived_state_staging_relative_path=(
                launch.run_derived_state_staging_path
            ),
            run_action_store_relative_path=launch.run_action_store_path,
            run_action_workspace_staging_relative_path=(
                launch.run_action_workspace_staging_path
            ),
            run_action_ledger_relative_path=launch.run_action_ledger_path,
            run_runtime_lock_relative_path=launch.run_runtime_lock_path,
        )

    @staticmethod
    def _materialized_paths(
        root: Path,
        layout: LaunchWorkspaceLayout,
    ) -> dict[str, Path]:
        return {
            "workspace": root / layout.workspace_relative_path,
            "immutable_root": root / layout.immutable_root_relative_path,
            "knowledge_snapshot": root / layout.knowledge_snapshot_relative_path,
            "task_adapter": root / layout.task_adapter_relative_path,
            "starting_artifacts": root / layout.starting_artifacts_relative_path,
            "launch_manifest": root / layout.launch_manifest_relative_path,
            "bootstrap_pin": root / layout.bootstrap_pin_relative_path,
        }

    def _open_destination_parent(
        self,
        run_root: Path,
    ) -> tuple[Path, int, tuple[int, int]]:
        if not isinstance(run_root, Path):
            raise LaunchWorkspaceError("run root must be one pathlib.Path")
        normalized = Path(os.path.abspath(run_root))
        if not run_root.is_absolute() or run_root != normalized:
            raise LaunchWorkspaceError("run root must be absolute and normalized")
        if normalized.parent == normalized or len(normalized.parts) < 3:
            raise LaunchWorkspaceError("run root target is too broad")
        if os.path.lexists(normalized):
            raise LaunchWorkspaceError("fresh run root already exists")
        parent = normalized.parent
        if (
            parent.is_symlink()
            or not parent.is_dir()
            or parent.resolve() != parent.absolute()
        ):
            raise LaunchWorkspaceError("run root parent must be one real directory")
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        metadata = os.fstat(parent_descriptor)
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
            os.close(parent_descriptor)
            raise LaunchWorkspaceError(
                "run root parent must be owned by the current user and not writable "
                "by group or others"
            )
        if self._directory_identity(parent_descriptor) != (
            parent.stat(follow_symlinks=False).st_dev,
            parent.stat(follow_symlinks=False).st_ino,
        ):
            os.close(parent_descriptor)
            raise LaunchWorkspaceError(
                "run root parent changed while opening its authority"
            )
        return normalized, parent_descriptor, (metadata.st_dev, metadata.st_ino)

    def _open_existing_run_root(
        self,
        run_root: Path,
        descriptors: ExitStack,
    ) -> tuple[Path, int]:
        if not isinstance(run_root, Path):
            raise LaunchWorkspaceError("run root must be one pathlib.Path")
        normalized = Path(os.path.abspath(run_root))
        if (
            not run_root.is_absolute()
            or run_root != normalized
            or normalized.parent == normalized
            or len(normalized.parts) < 3
        ):
            raise LaunchWorkspaceError(
                "existing run root must be absolute, normalized, and narrow"
            )
        parent = normalized.parent
        if (
            parent.is_symlink()
            or not parent.is_dir()
            or parent.resolve() != parent.absolute()
        ):
            raise LaunchWorkspaceError(
                "existing run root parent must be one real directory"
            )
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, parent_descriptor)
        parent_metadata = os.fstat(parent_descriptor)
        if parent_metadata.st_uid != os.geteuid() or parent_metadata.st_mode & 0o022:
            raise LaunchWorkspaceError(
                "existing run root parent must be owned and private"
            )
        root_metadata = os.stat(
            normalized.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise LaunchWorkspaceError("existing run root is absent or unsafe")
        root_descriptor = os.open(
            normalized.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        descriptors.callback(os.close, root_descriptor)
        reopened = os.fstat(root_descriptor)
        if (
            reopened.st_uid != os.geteuid()
            or stat.S_IMODE(reopened.st_mode) != 0o700
            or (reopened.st_dev, reopened.st_ino)
            != (root_metadata.st_dev, root_metadata.st_ino)
        ):
            raise LaunchWorkspaceError(
                "existing run root changed while opening its authority"
            )
        return normalized, root_descriptor

    def _read_configured_bootstrap_pin(
        self,
        root_descriptor: int,
        descriptors: ExitStack,
    ) -> bytes:
        relative_path = PurePosixPath(self._settings.launch.bootstrap_pin_path)
        opened: dict[str, int] = {}
        parent_descriptor = root_descriptor
        current = PurePosixPath(".")
        for component in relative_path.parent.parts:
            child_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_descriptor,
            )
            descriptors.callback(os.close, child_descriptor)
            metadata = os.fstat(child_descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o022
            ):
                raise LaunchWorkspaceError("bootstrap pin parent is absent or unsafe")
            current = (
                PurePosixPath(component)
                if current == PurePosixPath(".")
                else current / component
            )
            opened[current.as_posix()] = child_descriptor
            parent_descriptor = child_descriptor
        payload, _identity = self._read_control_file(
            root_descriptor,
            opened,
            relative_path.as_posix(),
            self._settings.launch.bootstrap_pin_size_bytes,
        )
        return payload

    @staticmethod
    def _require_parent_identity(
        parent: Path,
        parent_descriptor: int,
        expected: tuple[int, int],
    ) -> None:
        metadata = parent.stat(follow_symlinks=False)
        if (
            parent.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected
            or StarterWorkspaceBuilder._directory_identity(parent_descriptor)
            != expected
        ):
            raise LaunchWorkspaceError(
                "run root parent identity changed during installation"
            )

    @staticmethod
    def _validate_expert_source(resolved_launch: ResolvedLaunch) -> None:
        manifest = resolved_launch.manifest
        descriptors = {
            descriptor.relative_path: descriptor
            for descriptor in manifest.expert_source.extraction_receipt.source_tree_files
        }
        contents = resolved_launch.expert_source.source_contents
        validate_expert_repository_topology(
            manifest.expert_repository_map,
            manifest.expert_module_contracts,
            validation_error_type=LaunchWorkspaceError,
        )
        validate_expert_tree_ownership(
            manifest.expert_repository_map,
            manifest.expert_module_contracts,
            descriptors,
            validation_error_type=LaunchWorkspaceError,
        )
        expected_book = compile_expert_semantic_book(
            manifest.scope_contract,
            manifest.expert_repository_map,
            manifest.expert_module_contracts,
        )
        expected_controls = {
            EXPERT_BOOK_PATH: expected_book,
            EXPERT_REPOSITORY_MAP_PATH: (
                manifest.expert_repository_map.to_json_bytes()
            ),
            **{
                expert_module_contract_path(module.module_contract_id): (
                    module.to_json_bytes()
                )
                for module in manifest.expert_module_contracts
            },
        }
        if (
            any(
                contents.get(path) != payload
                for path, payload in expected_controls.items()
            )
            or any(descriptors[path].mode != "100644" for path in expected_controls)
            or expert_semantic_book_digest(expected_book)
            != manifest.expert_manifest.semantic_book_digest
        ):
            raise LaunchWorkspaceError(
                "expert source generated controls differ from launch evidence"
            )

    def _write_knowledge_snapshot(
        self,
        destination: Path,
        package: KnowledgeSnapshotPackage,
    ) -> str:
        package.verify()
        if any(
            len(payload) > self._settings.launch.knowledge_snapshot_file_size_bytes
            for payload in package.files.values()
        ):
            raise LaunchWorkspaceError(
                "knowledge snapshot contains a file above its launch bound"
            )
        descriptors = tuple(
            SourceFileDescriptor(
                relative_path=relative_path,
                digest=tree_or_blob_digest(payload),
                mode="100644",
                size=len(payload),
            )
            for relative_path, payload in sorted(package.files.items())
        )
        self._write_tree(
            destination,
            descriptors,
            package.files,
            read_only=True,
            name="knowledge snapshot",
        )
        tree_hash = source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
                for descriptor in descriptors
            }
        )
        self._verify_knowledge_root(
            destination,
            package.manifest,
            tree_hash,
            require_read_only=True,
        )
        return tree_hash

    def _write_task_adapter(
        self,
        destination: Path,
        resolved_launch: ResolvedLaunch,
    ) -> str:
        adapter = resolved_launch.task_adapter_binding.verified_adapter
        descriptors = adapter.evaluation_runtime_source_files
        self._write_tree(
            destination,
            descriptors,
            adapter.evaluation_runtime_source_contents,
            read_only=True,
            name="task adapter",
        )
        runtime_tree_hash = source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
                for descriptor in descriptors
            }
        )
        expected_runtime_files = tuple(
            descriptor
            for descriptor in resolved_launch.manifest.task_adapter.source_extraction_receipt.source_tree_files
            if PurePosixPath(descriptor.relative_path).parts[0]
            != "release_matrix_assets"
        )
        if descriptors != expected_runtime_files:
            raise LaunchWorkspaceError(
                "staged task-adapter runtime differs from its launch pin"
            )
        return runtime_tree_hash

    def _write_starting_artifacts(
        self,
        staging: Path,
        layout: LaunchWorkspaceLayout,
        resolved_launch: ResolvedLaunch,
    ) -> None:
        base = staging / layout.starting_artifacts_relative_path
        base.mkdir(parents=True, mode=0o755)
        verified_by_id = {
            item.artifact.starting_artifact_content_id: item
            for item in resolved_launch.starting_artifacts.starting_artifacts
        }
        if set(verified_by_id) != set(layout.starting_artifact_roots):
            raise LaunchWorkspaceError(
                "starting-artifact layout differs from verified launch bytes"
            )
        for artifact_id in sorted(verified_by_id):
            item = verified_by_id[artifact_id]
            self._write_tree(
                staging / layout.starting_artifact_roots[artifact_id],
                item.artifact.source_files,
                item.source_contents,
                read_only=True,
                name="starting artifact",
            )
        self._make_directory_read_only(base)

    @staticmethod
    def _write_tree(
        destination: Path,
        descriptors: tuple[SourceFileDescriptor, ...],
        contents: Mapping[str, bytes],
        *,
        read_only: bool,
        name: str,
    ) -> None:
        descriptor_by_path = {
            descriptor.relative_path: descriptor for descriptor in descriptors
        }
        if len(descriptor_by_path) != len(descriptors) or set(contents) != set(
            descriptor_by_path
        ):
            raise LaunchWorkspaceError(f"{name} byte closure is not exact")
        if destination.exists() or destination.is_symlink():
            raise LaunchWorkspaceError(f"{name} destination already exists")
        destination.mkdir(
            parents=True,
            mode=0o755 if read_only else 0o700,
        )
        destination.chmod(0o755 if read_only else 0o700)
        for relative_path in sorted(descriptor_by_path):
            descriptor = descriptor_by_path[relative_path]
            path = StarterWorkspaceBuilder._safe_source_path(
                relative_path,
                f"{name} path",
            )
            payload = contents[relative_path]
            if (
                type(payload) is not bytes
                or len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.digest
                or descriptor.mode not in {"100644", "100755"}
            ):
                raise LaunchWorkspaceError(f"{name} bytes differ from their descriptor")
            output = destination / path
            output.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            file_descriptor = os.open(
                output,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o755 if descriptor.mode == "100755" else 0o644,
            )
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(
                    handle.fileno(),
                    0o755 if descriptor.mode == "100755" else 0o644,
                )
                os.fsync(handle.fileno())
            metadata = output.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != descriptor.size
            ):
                raise LaunchWorkspaceError(f"{name} produced an unsafe file")
        StarterWorkspaceBuilder._verify_tree(
            destination,
            descriptors,
            contents,
            name=name,
        )
        if read_only:
            StarterWorkspaceBuilder._make_tree_read_only(
                destination,
                descriptor_by_path,
            )

    @staticmethod
    def _safe_source_path(value: str, name: str) -> PurePosixPath:
        path = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or "\0" in value
            or path.is_absolute()
            or path == PurePosixPath(".")
            or ".." in path.parts
            or ".git" in path.parts
            or value == ".gitmodules"
            or path.as_posix() != value
        ):
            raise LaunchWorkspaceError(f"{name} is unsafe")
        return path

    @staticmethod
    def _verify_tree(
        root: Path,
        descriptors: tuple[SourceFileDescriptor, ...],
        contents: Mapping[str, bytes] | None,
        *,
        name: str,
        ignore_git_metadata: bool = False,
        require_read_only: bool = False,
        expected_root_identity: tuple[int, int] | None = None,
    ) -> None:
        root_metadata = root.stat()
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (expected_root_identity is None and root.is_symlink())
            or (
                expected_root_identity is not None
                and (root_metadata.st_dev, root_metadata.st_ino)
                != expected_root_identity
            )
        ):
            raise LaunchWorkspaceError(f"{name} root is absent or unsafe")
        expected = {descriptor.relative_path: descriptor for descriptor in descriptors}
        observed_files: dict[str, Path] = {}
        observed_directories: list[Path] = []
        for path in sorted(root.rglob("*")):
            relative_path = path.relative_to(root).as_posix()
            if ignore_git_metadata and PurePosixPath(relative_path).parts[0] == ".git":
                continue
            if path.is_symlink():
                raise LaunchWorkspaceError(f"{name} contains a symlink")
            metadata = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                observed_directories.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise LaunchWorkspaceError(f"{name} contains a special/shared file")
            observed_files[relative_path] = path
        if set(observed_files) != set(expected):
            raise LaunchWorkspaceError(f"{name} filesystem closure is not exact")
        for relative_path, path in observed_files.items():
            descriptor = expected[relative_path]
            expected_mode = (
                (0o555 if descriptor.mode == "100755" else 0o444)
                if require_read_only
                else (0o755 if descriptor.mode == "100755" else 0o644)
            )
            payload = _read_bounded_regular_file(
                path,
                descriptor.size,
                f"{name} file",
                expected_size=descriptor.size,
                expected_mode=expected_mode,
            )
            if (
                tree_or_blob_digest(payload) != descriptor.digest
                or (contents is not None and payload != contents[relative_path])
                or bool(path.stat(follow_symlinks=False).st_mode & stat.S_IXUSR)
                != (descriptor.mode == "100755")
                or (
                    require_read_only
                    and stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
                    != (0o555 if descriptor.mode == "100755" else 0o444)
                )
            ):
                raise LaunchWorkspaceError(
                    f"{name} file differs from its verified descriptor"
                )
        if require_read_only and (
            stat.S_IMODE(root_metadata.st_mode) != 0o555
            or any(
                stat.S_IMODE(directory.stat(follow_symlinks=False).st_mode) != 0o555
                for directory in observed_directories
            )
        ):
            raise LaunchWorkspaceError(f"{name} contains a writable directory")

    def _verify_knowledge_root(
        self,
        root: Path,
        manifest,
        expected_tree_hash: str,
        *,
        require_read_only: bool,
        expected_root_identity: tuple[int, int] | None = None,
    ) -> None:
        root_metadata = root.stat()
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (expected_root_identity is None and root.is_symlink())
            or (
                expected_root_identity is not None
                and (root_metadata.st_dev, root_metadata.st_ino)
                != expected_root_identity
            )
        ):
            raise LaunchWorkspaceError("knowledge snapshot root is absent or unsafe")
        expected_digests = {
            "snapshot.json": tree_or_blob_digest(manifest.to_json_bytes()),
            **dict(manifest.checksums),
        }
        observed: dict[str, tuple[str, str, int]] = {}
        directories: list[Path] = []
        for path in sorted(root.rglob("*")):
            relative_path = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise LaunchWorkspaceError("knowledge snapshot contains a symlink")
            metadata = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(path)
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (require_read_only and stat.S_IMODE(metadata.st_mode) != 0o444)
            ):
                raise LaunchWorkspaceError(
                    "knowledge snapshot contains a writable, special, or shared file"
                )
            payload = _read_bounded_regular_file(
                path,
                self._settings.launch.knowledge_snapshot_file_size_bytes,
                "knowledge snapshot file",
                expected_mode=0o444 if require_read_only else None,
            )
            digest = tree_or_blob_digest(payload)
            if expected_digests.get(relative_path) != digest:
                raise LaunchWorkspaceError(
                    "knowledge snapshot differs from its manifest checksum closure"
                )
            observed[relative_path] = (digest, "100644", len(payload))
        if set(observed) != set(expected_digests):
            raise LaunchWorkspaceError(
                "knowledge snapshot filesystem closure is not exact"
            )
        if source_tree_digest(observed) != expected_tree_hash or (
            require_read_only
            and (
                stat.S_IMODE(root_metadata.st_mode) != 0o555
                or any(
                    stat.S_IMODE(directory.stat(follow_symlinks=False).st_mode) != 0o555
                    for directory in directories
                )
            )
        ):
            raise LaunchWorkspaceError(
                "knowledge snapshot tree or immutable modes differ from its pin"
            )

    @staticmethod
    def _verify_immutable_root_closure(
        immutable_root: Path,
        layout: LaunchWorkspaceLayout,
        manifest: LaunchManifest,
        runtime_descriptors: tuple[SourceFileDescriptor, ...],
        *,
        expected_root_identity: tuple[int, int],
    ) -> None:
        root_path = PurePosixPath(layout.immutable_root_relative_path)
        expected_files: set[str] = set()
        expected_directories: set[str] = set()

        def add_component(
            component_relative_path: str,
            file_paths: tuple[str, ...],
        ) -> None:
            component = PurePosixPath(component_relative_path).relative_to(root_path)
            expected_directories.add(component.as_posix())
            for file_path in file_paths:
                relative_file = component / file_path
                expected_files.add(relative_file.as_posix())
                for parent in relative_file.parents:
                    if parent != PurePosixPath("."):
                        expected_directories.add(parent.as_posix())

        add_component(
            layout.knowledge_snapshot_relative_path,
            tuple(
                sorted(
                    {
                        "snapshot.json",
                        *manifest.knowledge_manifest.checksums,
                    }
                )
            ),
        )
        add_component(
            layout.task_adapter_relative_path,
            tuple(descriptor.relative_path for descriptor in runtime_descriptors),
        )
        starting_by_id = {
            artifact.starting_artifact_content_id: artifact
            for artifact in manifest.starting_artifacts.starting_artifacts
        }
        starting_root = PurePosixPath(
            layout.starting_artifacts_relative_path
        ).relative_to(root_path)
        expected_directories.add(starting_root.as_posix())
        for artifact_id, artifact in starting_by_id.items():
            add_component(
                layout.starting_artifact_roots[artifact_id],
                tuple(descriptor.relative_path for descriptor in artifact.source_files),
            )

        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        root_metadata = immutable_root.stat()
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (root_metadata.st_dev, root_metadata.st_ino) != expected_root_identity
            or stat.S_IMODE(root_metadata.st_mode) != 0o555
        ):
            raise LaunchWorkspaceError("immutable launch root is writable or unsafe")
        for path in immutable_root.rglob("*"):
            relative_path = path.relative_to(immutable_root).as_posix()
            if path.is_symlink():
                raise LaunchWorkspaceError("immutable launch root contains a symlink")
            metadata = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                if stat.S_IMODE(metadata.st_mode) != 0o555:
                    raise LaunchWorkspaceError(
                        "immutable launch root contains a writable directory"
                    )
                actual_directories.add(relative_path)
            elif (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink == 1
                and not metadata.st_mode & 0o222
            ):
                actual_files.add(relative_path)
            else:
                raise LaunchWorkspaceError(
                    "immutable launch root contains a writable, special, or shared file"
                )
        if actual_files != expected_files or actual_directories != expected_directories:
            raise LaunchWorkspaceError(
                "immutable launch filesystem closure is not exact"
            )

    @staticmethod
    def _make_tree_read_only(
        root: Path,
        descriptors: Mapping[str, SourceFileDescriptor],
    ) -> None:
        for relative_path, descriptor in descriptors.items():
            (root / relative_path).chmod(
                0o555 if descriptor.mode == "100755" else 0o444
            )
        for directory in sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        root.chmod(0o555)

    @staticmethod
    def _make_directory_read_only(root: Path) -> None:
        for directory in sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        root.chmod(0o555)

    def _initialize_git_baseline(
        self,
        workspace: Path,
        descriptors: tuple[SourceFileDescriptor, ...],
        contents: Mapping[str, bytes],
    ) -> tuple[str, str, str, tuple[str, ...]]:
        git_root = workspace / ".git"
        if git_root.exists() or git_root.is_symlink():
            raise LaunchWorkspaceError("expert source contains Git metadata")
        objects = git_root / "objects"
        references = git_root / "refs" / "heads"
        objects.mkdir(parents=True, mode=0o700)
        (objects / "info").mkdir(mode=0o700)
        (objects / "pack").mkdir(mode=0o700)
        references.mkdir(parents=True, mode=0o700)
        branch = self._settings.launch.workspace_git_branch
        author_name = self._settings.github.commit_author_name
        author_email = self._settings.github.commit_author_email
        if any(character in author_name for character in "\r\n<>"):
            raise LaunchWorkspaceError("Git baseline author name is unsafe")

        tree_sha, blob_ids = self._write_git_tree(
            git_root,
            descriptors,
            contents,
        )
        identity = f"{author_name} <{author_email}> {_GIT_COMMIT_TIME}".encode("utf-8")
        commit_payload = self._git_commit_payload(tree_sha, identity)
        commit_sha = self._write_git_object(git_root, "commit", commit_payload)
        self._write_plain_file(
            git_root / "HEAD",
            f"ref: refs/heads/{branch}\n".encode("utf-8"),
            mode=0o600,
        )
        self._write_plain_file(
            git_root / "config",
            self._git_config_bytes(),
            mode=0o600,
        )
        reference_path = git_root / "refs" / "heads" / branch
        reference_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._write_plain_file(
            reference_path,
            f"{commit_sha}\n".encode("ascii"),
            mode=0o600,
        )
        index_payload = self._git_index_payload(
            descriptors,
            blob_ids,
        )
        self._write_plain_file(git_root / "index", index_payload, mode=0o600)
        self._verify_git_object(git_root, tree_sha, "tree")
        self._verify_git_object(git_root, commit_sha, "commit")
        object_ids = tuple(
            sorted(
                path.parent.name + path.name
                for path in (git_root / "objects").glob("[0-9a-f][0-9a-f]/*")
                if path.is_file() and not path.is_symlink()
            )
        )
        return (
            tree_sha,
            commit_sha,
            tree_or_blob_digest(index_payload),
            object_ids,
        )

    def _write_git_tree(
        self,
        git_root: Path,
        descriptors: tuple[SourceFileDescriptor, ...],
        contents: Mapping[str, bytes],
    ) -> tuple[str, dict[str, str]]:
        root: dict[str, Any] = {}
        blob_ids: dict[str, str] = {}
        descriptor_by_path = {
            descriptor.relative_path: descriptor for descriptor in descriptors
        }
        for relative_path in sorted(descriptor_by_path):
            path = self._safe_source_path(relative_path, "Git source path")
            payload = contents[relative_path]
            blob_ids[relative_path] = self._write_git_object(
                git_root,
                "blob",
                payload,
            )
            cursor = root
            for part in path.parts[:-1]:
                child = cursor.setdefault(part, {})
                if not isinstance(child, dict):
                    raise LaunchWorkspaceError(
                        "Git source contains a file/directory collision"
                    )
                cursor = child
            name = path.parts[-1]
            if name in cursor:
                raise LaunchWorkspaceError("Git source path is duplicated")
            cursor[name] = _GitLeaf(
                mode=descriptor_by_path[relative_path].mode,
                payload=payload,
                path=Path(relative_path),
            )
        return self._write_git_tree_node(git_root, root), blob_ids

    def _write_git_tree_node(
        self,
        git_root: Path,
        node: Mapping[str, Any],
    ) -> str:
        entries: list[tuple[bytes, bytes]] = []
        for name, child in node.items():
            name_bytes = name.encode("utf-8")
            if isinstance(child, dict):
                object_id = self._write_git_tree_node(git_root, child)
                sort_key = name_bytes + b"/"
                entry = b"40000 " + name_bytes + b"\0" + bytes.fromhex(object_id)
            else:
                if type(child) is not _GitLeaf:
                    raise LaunchWorkspaceError("Git tree contains an invalid leaf")
                object_id = self._git_object_id("blob", child.payload)
                sort_key = name_bytes
                entry = child.mode.encode("ascii") + b" " + name_bytes
                entry += b"\0" + bytes.fromhex(object_id)
            entries.append((sort_key, entry))
        payload = b"".join(entry for _, entry in sorted(entries))
        return self._write_git_object(git_root, "tree", payload)

    @staticmethod
    def _git_index_payload(
        descriptors: tuple[SourceFileDescriptor, ...],
        blob_ids: Mapping[str, str],
    ) -> bytes:
        entries = []
        for descriptor in sorted(
            descriptors,
            key=lambda item: item.relative_path.encode("utf-8"),
        ):
            path_bytes = descriptor.relative_path.encode("utf-8")
            if len(path_bytes) > _GIT_INDEX_PATH_LIMIT:
                raise LaunchWorkspaceError(
                    "Git source path exceeds deterministic index format"
                )
            entry = _GIT_INDEX_ENTRY_HEADER.pack(
                0,
                0,
                0,
                0,
                0,
                0,
                int(descriptor.mode, 8),
                0,
                0,
                descriptor.size & 0xFFFFFFFF,
                bytes.fromhex(blob_ids[descriptor.relative_path]),
                len(path_bytes),
            )
            entry += path_bytes + b"\0"
            entry += b"\0" * (-len(entry) % 8)
            entries.append(entry)
        body = _GIT_INDEX_HEADER.pack(b"DIRC", 2, len(entries)) + b"".join(entries)
        return body + StarterWorkspaceBuilder._sha1(body)

    @staticmethod
    def _git_config_bytes() -> bytes:
        return (
            b"[core]\n"
            b"\trepositoryformatversion = 0\n"
            b"\tfilemode = true\n"
            b"\tbare = false\n"
            b"\tlogallrefupdates = false\n"
            b"\tautocrlf = false\n"
            b"\thooksPath = /dev/null\n"
            b"[commit]\n"
            b"\tgpgsign = false\n"
            b"[gc]\n"
            b"\tauto = 0\n"
            b"[i18n]\n"
            b"\tcommitEncoding = UTF-8\n"
        )

    @staticmethod
    def _git_commit_payload(tree_sha: str, identity: bytes) -> bytes:
        return (
            f"tree {tree_sha}\n".encode("ascii")
            + b"author "
            + identity
            + b"\ncommitter "
            + identity
            + b"\n\n"
            + _GIT_COMMIT_MESSAGE
        )

    @staticmethod
    def _git_object_id(kind: str, payload: bytes) -> str:
        object_bytes = f"{kind} {len(payload)}\0".encode("ascii") + payload
        return StarterWorkspaceBuilder._sha1(object_bytes).hex()

    def _write_git_object(
        self,
        git_root: Path,
        kind: str,
        payload: bytes,
    ) -> str:
        object_bytes = f"{kind} {len(payload)}\0".encode("ascii") + payload
        object_id = self._sha1(object_bytes).hex()
        object_path = git_root / "objects" / object_id[:2] / object_id[2:]
        object_path.parent.mkdir(exist_ok=True, mode=0o700)
        if object_path.exists():
            if self._read_git_object(git_root, object_id, kind) != payload:
                raise LaunchWorkspaceError("Git object ID collision")
            return object_id
        self._write_plain_file(
            object_path,
            zlib.compress(object_bytes),
            mode=0o444,
        )
        return object_id

    def _verify_git_object(
        self,
        git_root: Path,
        object_id: str,
        kind: str,
    ) -> None:
        self._read_git_object(git_root, object_id, kind)

    def _read_git_object(
        self,
        git_root: Path,
        object_id: str,
        kind: str,
    ) -> bytes:
        object_path = git_root / "objects" / object_id[:2] / object_id[2:]
        metadata = object_path.stat(follow_symlinks=False)
        maximum_object_bytes = (
            self._settings.github.source_tree_size_bytes
            + self._settings.github.git_tree_metadata_size_bytes
        )
        if (
            object_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > maximum_object_bytes
            or stat.S_IMODE(metadata.st_mode) != 0o444
        ):
            raise LaunchWorkspaceError("Git baseline object file is unsafe")
        descriptor = os.open(object_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(descriptor, "rb") as handle:
            compressed = handle.read(maximum_object_bytes + 1)
            reopened = os.fstat(handle.fileno())
        if len(compressed) > maximum_object_bytes or (
            reopened.st_dev,
            reopened.st_ino,
            reopened.st_size,
        ) != (metadata.st_dev, metadata.st_ino, metadata.st_size):
            raise LaunchWorkspaceError("Git baseline object changed while reading")
        decompressor = zlib.decompressobj()
        object_bytes = decompressor.decompress(
            compressed,
            maximum_object_bytes + 1,
        )
        if (
            len(object_bytes) > maximum_object_bytes
            or not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise LaunchWorkspaceError(
                "Git baseline object exceeds its configured bound"
            )
        prefix = f"{kind} ".encode("ascii")
        if (
            not object_bytes.startswith(prefix)
            or self._sha1(object_bytes).hex() != object_id
            or b"\0" not in object_bytes
        ):
            raise LaunchWorkspaceError("Git baseline object failed verification")
        header, payload = object_bytes.split(b"\0", 1)
        if header != f"{kind} {len(payload)}".encode("ascii"):
            raise LaunchWorkspaceError("Git baseline object header is invalid")
        return payload

    @staticmethod
    def _sha1(payload: bytes) -> bytes:
        return hashlib.sha1(payload, usedforsecurity=False).digest()

    @staticmethod
    def _write_control_file(path: Path, payload: bytes) -> None:
        StarterWorkspaceBuilder._write_plain_file(path, payload, mode=0o444)

    @staticmethod
    def _write_plain_file(path: Path, payload: bytes, *, mode: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())

    @staticmethod
    def _append_plain_file(path: Path, payload: bytes) -> None:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        with os.fdopen(descriptor, "ab", buffering=0) as handle:
            written = handle.write(payload)
            if written != len(payload):
                raise LaunchWorkspaceError("checkpoint journal append was incomplete")
            os.fsync(handle.fileno())

    def _verify_published(
        self,
        run_root: Path,
        expected_pin: BootstrapPin,
        *,
        requires_initial_state: bool,
        exposed_run_root: Path | None = None,
        expected_root_identity: tuple[int, int] | None = None,
        root_descriptor: int | None = None,
    ) -> PreparedLaunchWorkspace:
        if type(requires_initial_state) is not bool:
            raise LaunchWorkspaceError(
                "workspace verification mode must be an exact boolean"
            )
        layout = expected_pin.installation_receipt.layout
        with ExitStack() as descriptors:
            if root_descriptor is None:
                verified_root_descriptor = _open_real_root(run_root, descriptors)
            else:
                verified_root_descriptor = os.dup(root_descriptor)
                descriptors.callback(os.close, verified_root_descriptor)
            observed_root_identity = self._directory_identity(verified_root_descriptor)
            if (
                expected_root_identity is not None
                and observed_root_identity != expected_root_identity
            ):
                raise LaunchWorkspaceError(
                    "published run root differs from its staging inode"
                )
            opened_directories, directory_identities = _open_layout_directories(
                verified_root_descriptor,
                layout,
                descriptors,
            )
            pin_bytes, pin_identity = self._read_control_file(
                verified_root_descriptor,
                opened_directories,
                layout.bootstrap_pin_relative_path,
                self._settings.launch.bootstrap_pin_size_bytes,
            )
            manifest_bytes, manifest_identity = self._read_control_file(
                verified_root_descriptor,
                opened_directories,
                layout.launch_manifest_relative_path,
                self._settings.launch.launch_manifest_size_bytes,
            )
            pin = BootstrapPin.from_json_bytes(pin_bytes)
            manifest = LaunchManifest.from_json_bytes(manifest_bytes)
            if (
                pin != expected_pin
                or manifest != pin.launch_manifest
                or pin.to_json_bytes() != pin_bytes
                or manifest.to_json_bytes() != manifest_bytes
                or layout != self._layout(manifest)
                or tree_or_blob_digest(manifest_bytes)
                != pin.launch_manifest_full_digest
            ):
                raise LaunchWorkspaceError(
                    "published launch control files differ from their authority"
                )
            journal_descriptor = _open_layout_file(
                verified_root_descriptor,
                opened_directories,
                layout.run_checkpoint_journal_relative_path,
            )
            descriptors.callback(os.close, journal_descriptor)
            journal_metadata = os.fstat(journal_descriptor)
            receipt = pin.installation_receipt
            if (
                journal_metadata.st_uid != os.geteuid()
                or journal_metadata.st_nlink != 1
                or stat.S_IMODE(journal_metadata.st_mode) != 0o600
                or journal_metadata.st_size
                > self._settings.launch.run_checkpoint_journal_size_bytes
                or (
                    journal_metadata.st_dev,
                    journal_metadata.st_ino,
                )
                != (
                    receipt.run_checkpoint_journal_device,
                    receipt.run_checkpoint_journal_inode,
                )
            ):
                raise LaunchWorkspaceError("published checkpoint journal is unsafe")
            journal_payload = os.read(
                journal_descriptor,
                self._settings.launch.run_checkpoint_journal_size_bytes + 1,
            )
            journal_reopened = os.fstat(journal_descriptor)
            if (
                len(journal_payload)
                > self._settings.launch.run_checkpoint_journal_size_bytes
                or (
                    journal_reopened.st_dev,
                    journal_reopened.st_ino,
                    journal_reopened.st_size,
                    stat.S_IMODE(journal_reopened.st_mode),
                )
                != (
                    journal_metadata.st_dev,
                    journal_metadata.st_ino,
                    journal_metadata.st_size,
                    stat.S_IMODE(journal_metadata.st_mode),
                )
                or (
                    requires_initial_state
                    and journal_payload
                    != RunCheckpointHead.initial(pin).to_json_bytes() + b"\n"
                )
                or receipt.launch_settings_id
                != content_id("launch-settings", self._settings.launch.to_dict())
            ):
                raise LaunchWorkspaceError(
                    "published checkpoint journal differs from its authority"
                )
            lock_descriptor = _open_layout_file(
                verified_root_descriptor,
                opened_directories,
                layout.run_checkpoint_lock_relative_path,
            )
            descriptors.callback(os.close, lock_descriptor)
            lock_metadata = os.fstat(lock_descriptor)
            if (
                lock_metadata.st_uid != os.geteuid()
                or lock_metadata.st_nlink != 1
                or lock_metadata.st_size != 0
                or stat.S_IMODE(lock_metadata.st_mode) != 0o600
                or (lock_metadata.st_dev, lock_metadata.st_ino)
                != (
                    receipt.run_checkpoint_lock_device,
                    receipt.run_checkpoint_lock_inode,
                )
            ):
                raise LaunchWorkspaceError("published checkpoint lock is unsafe")
            lock_identity = (lock_metadata.st_dev, lock_metadata.st_ino)
            runtime_lock_descriptor = _open_layout_file(
                verified_root_descriptor,
                opened_directories,
                layout.run_runtime_lock_relative_path,
            )
            descriptors.callback(os.close, runtime_lock_descriptor)
            runtime_lock_metadata = os.fstat(runtime_lock_descriptor)
            if (
                runtime_lock_metadata.st_uid != os.geteuid()
                or runtime_lock_metadata.st_nlink != 1
                or runtime_lock_metadata.st_size != 0
                or stat.S_IMODE(runtime_lock_metadata.st_mode) != 0o600
                or (
                    runtime_lock_metadata.st_dev,
                    runtime_lock_metadata.st_ino,
                )
                != (
                    receipt.run_runtime_lock_device,
                    receipt.run_runtime_lock_inode,
                )
            ):
                raise LaunchWorkspaceError("published runtime lock is unsafe")
            runtime_lock_identity = (
                runtime_lock_metadata.st_dev,
                runtime_lock_metadata.st_ino,
            )
            action_store_identity = directory_identities[
                layout.run_action_store_relative_path
            ]
            if action_store_identity != (
                receipt.run_action_store_device,
                receipt.run_action_store_inode,
            ):
                raise LaunchWorkspaceError(
                    "published run action store differs from its authority"
                )
            action_workspace_staging_descriptor = opened_directories[
                layout.run_action_workspace_staging_relative_path
            ]
            action_workspace_staging_identity = directory_identities[
                layout.run_action_workspace_staging_relative_path
            ]
            if (
                action_workspace_staging_identity
                != (
                    receipt.run_action_workspace_staging_device,
                    receipt.run_action_workspace_staging_inode,
                )
                or _require_owner_private_directory(
                    action_workspace_staging_descriptor,
                    "published run action workspace staging root",
                )
                != action_workspace_staging_identity
            ):
                raise LaunchWorkspaceError(
                    "published run action workspace staging root differs "
                    "from its authority"
                )
            action_store_descriptor = opened_directories[
                layout.run_action_store_relative_path
            ]
            for (
                lock_name,
                expected_identity,
                description,
            ) in (
                (
                    "registry.lock",
                    (
                        receipt.run_action_registry_lock_device,
                        receipt.run_action_registry_lock_inode,
                    ),
                    "registry",
                ),
                (
                    "workspace.lock",
                    (
                        receipt.run_action_workspace_lock_device,
                        receipt.run_action_workspace_lock_inode,
                    ),
                    "workspace",
                ),
            ):
                action_lock_descriptor = os.open(
                    lock_name,
                    os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=action_store_descriptor,
                )
                descriptors.callback(os.close, action_lock_descriptor)
                action_lock_metadata = os.fstat(action_lock_descriptor)
                if (
                    not stat.S_ISREG(action_lock_metadata.st_mode)
                    or action_lock_metadata.st_uid != os.geteuid()
                    or action_lock_metadata.st_nlink != 1
                    or action_lock_metadata.st_size != 0
                    or stat.S_IMODE(action_lock_metadata.st_mode) != 0o600
                    or (
                        action_lock_metadata.st_dev,
                        action_lock_metadata.st_ino,
                    )
                    != expected_identity
                ):
                    raise LaunchWorkspaceError(
                        f"published run action {description} lock is unsafe"
                    )
            workspace_root = self._descriptor_path(
                opened_directories[layout.workspace_relative_path]
            )
            _require_workspace_generation_directory(
                opened_directories[layout.workspace_relative_path],
                receipt,
                "published execution workspace",
            )
            knowledge_root = self._descriptor_path(
                opened_directories[layout.knowledge_snapshot_relative_path]
            )
            task_adapter_root = self._descriptor_path(
                opened_directories[layout.task_adapter_relative_path]
            )
            immutable_root = self._descriptor_path(
                opened_directories[layout.immutable_root_relative_path]
            )
            expert_descriptors = (
                manifest.expert_source.extraction_receipt.source_tree_files
            )
            if requires_initial_state:
                self._verify_tree(
                    workspace_root,
                    expert_descriptors,
                    None,
                    name="published expert source",
                    ignore_git_metadata=True,
                    expected_root_identity=directory_identities[
                        layout.workspace_relative_path
                    ],
                )
                self._verify_git_baseline(
                    workspace_root,
                    pin.installation_receipt,
                    expert_descriptors,
                )
            self._verify_knowledge_root(
                knowledge_root,
                manifest.knowledge_manifest,
                pin.installation_receipt.knowledge_package_tree_hash,
                require_read_only=True,
                expected_root_identity=directory_identities[
                    layout.knowledge_snapshot_relative_path
                ],
            )
            runtime_descriptors = tuple(
                descriptor
                for descriptor in manifest.task_adapter.source_extraction_receipt.source_tree_files
                if PurePosixPath(descriptor.relative_path).parts[0]
                != "release_matrix_assets"
            )
            self._verify_tree(
                task_adapter_root,
                runtime_descriptors,
                None,
                name="published task adapter",
                require_read_only=True,
                expected_root_identity=directory_identities[
                    layout.task_adapter_relative_path
                ],
            )
            if (
                source_tree_digest(
                    {
                        descriptor.relative_path: (
                            descriptor.digest,
                            descriptor.mode,
                            descriptor.size,
                        )
                        for descriptor in runtime_descriptors
                    }
                )
                != pin.installation_receipt.task_adapter_runtime_tree_hash
            ):
                raise LaunchWorkspaceError(
                    "published task-adapter tree differs from its pin"
                )
            artifacts_by_id = {
                artifact.starting_artifact_content_id: artifact
                for artifact in manifest.starting_artifacts.starting_artifacts
            }
            for artifact_id, artifact in artifacts_by_id.items():
                artifact_relative_path = layout.starting_artifact_roots[artifact_id]
                self._verify_tree(
                    self._descriptor_path(opened_directories[artifact_relative_path]),
                    artifact.source_files,
                    None,
                    name="published starting artifact",
                    require_read_only=True,
                    expected_root_identity=directory_identities[artifact_relative_path],
                )
            self._verify_immutable_root_closure(
                immutable_root,
                layout,
                manifest,
                runtime_descriptors,
                expected_root_identity=directory_identities[
                    layout.immutable_root_relative_path
                ],
            )
            self._verify_outer_run_root_closure(
                self._descriptor_path(verified_root_descriptor),
                layout,
                observed_root_identity,
            )
            public_root = run_root if exposed_run_root is None else exposed_run_root
            if self._path_directory_identity(public_root) != observed_root_identity:
                raise LaunchWorkspaceError(
                    "published run-root pathname differs from its verified inode"
                )
            starting_paths = {
                artifact_id: public_root / relative_path
                for artifact_id, relative_path in layout.starting_artifact_roots.items()
            }
            return PreparedLaunchWorkspace(
                run_root=public_root,
                workspace=public_root / layout.workspace_relative_path,
                knowledge_snapshot=(
                    public_root / layout.knowledge_snapshot_relative_path
                ),
                task_adapter=public_root / layout.task_adapter_relative_path,
                starting_artifacts=starting_paths,
                launch_manifest_path=(
                    public_root / layout.launch_manifest_relative_path
                ),
                bootstrap_pin_path=(public_root / layout.bootstrap_pin_relative_path),
                bootstrap_pin=pin,
                _builder_authority=_WORKSPACE_BUILDER_AUTHORITY,
                _published_root_identity=observed_root_identity,
                _pinned_directory_identities=_pinned_layout_directory_identities(
                    layout,
                    directory_identities,
                ),
                _pinned_control_file_identities={
                    layout.launch_manifest_relative_path: manifest_identity,
                    layout.bootstrap_pin_relative_path: pin_identity,
                    layout.run_checkpoint_journal_relative_path: (
                        journal_metadata.st_dev,
                        journal_metadata.st_ino,
                    ),
                    layout.run_checkpoint_lock_relative_path: lock_identity,
                    layout.run_runtime_lock_relative_path: runtime_lock_identity,
                },
                _builder_verifier=self,
                _requires_initial_state=requires_initial_state,
            )

    def _verify_outer_run_root_closure(
        self,
        run_root: Path,
        layout: LaunchWorkspaceLayout,
        expected_root_identity: tuple[int, int],
    ) -> None:
        root_metadata = run_root.stat()
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (root_metadata.st_dev, root_metadata.st_ino) != expected_root_identity
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
            or root_metadata.st_uid != os.geteuid()
        ):
            raise LaunchWorkspaceError("published run-root envelope is unsafe")
        component_roots = (
            PurePosixPath(layout.workspace_relative_path),
            PurePosixPath(layout.immutable_root_relative_path),
        )
        component_ancestors = {
            parent
            for component_root in component_roots
            for parent in component_root.parents
            if parent != PurePosixPath(".")
        }
        immutable_control_files = {
            PurePosixPath(layout.launch_manifest_relative_path),
            PurePosixPath(layout.bootstrap_pin_relative_path),
        }
        checkpoint_file = PurePosixPath(layout.run_checkpoint_relative_path)
        checkpoint_journal = PurePosixPath(layout.run_checkpoint_journal_relative_path)
        checkpoint_lock = PurePosixPath(layout.run_checkpoint_lock_relative_path)
        runtime_lock = PurePosixPath(layout.run_runtime_lock_relative_path)
        checkpoint_staging = PurePosixPath(layout.run_checkpoint_staging_relative_path)
        projection_files = {
            PurePosixPath(layout.run_idea_archive_relative_path): (
                self._settings.launch.run_idea_archive_size_bytes
            ),
            PurePosixPath(layout.run_experiment_history_relative_path): (
                self._settings.launch.run_experiment_history_size_bytes
            ),
            PurePosixPath(layout.run_execution_journal_relative_path): (
                self._settings.launch.run_execution_journal_size_bytes
            ),
            PurePosixPath(layout.run_action_ledger_relative_path): (
                self._settings.launch.run_action_projection_size_bytes
            ),
        }
        derived_state_store = PurePosixPath(
            layout.run_derived_state_store_relative_path
        )
        derived_state_staging = PurePosixPath(
            layout.run_derived_state_staging_relative_path
        )
        action_store = PurePosixPath(layout.run_action_store_relative_path)
        action_workspace_staging = PurePosixPath(
            layout.run_action_workspace_staging_relative_path
        )
        action_workspace_staging_metadata = (
            run_root / action_workspace_staging.as_posix()
        ).stat(follow_symlinks=False)
        control_directories = {
            parent
            for control_file in (
                *immutable_control_files,
                checkpoint_file,
                checkpoint_journal,
                checkpoint_lock,
                runtime_lock,
                *projection_files,
            )
            for parent in control_file.parents
            if parent != PurePosixPath(".")
        }
        control_directories.update(
            {
                checkpoint_staging,
                derived_state_store,
                derived_state_staging,
                action_store,
                action_workspace_staging,
            }
        )
        envelope_directories = component_ancestors | control_directories
        staging_entry_count = 0
        derived_state_entry_count = 0
        derived_state_staging_entry_count = 0
        observed_checkpoint_journal = False
        observed_checkpoint_lock = False
        observed_runtime_lock = False
        action_store_entry_count = 0
        action_store_operation_digests = set()
        action_store_size_bytes = 0
        action_workspace_staging_roots = set()
        action_workspace_staging_inodes = set()
        action_workspace_staging_entry_count = 0
        action_workspace_staging_size_bytes = 0
        for path in run_root.rglob("*"):
            relative_path = PurePosixPath(path.relative_to(run_root).as_posix())
            if path.is_symlink():
                raise LaunchWorkspaceError("published run root contains a symlink")
            if any(
                relative_path == component or component in relative_path.parents
                for component in component_roots
            ):
                continue
            metadata = path.stat(follow_symlinks=False)
            if relative_path in envelope_directories:
                requires_private_mode = relative_path in {
                    checkpoint_file.parent,
                    checkpoint_staging,
                    derived_state_store,
                    derived_state_staging,
                    action_store,
                    action_workspace_staging,
                }
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or (
                        stat.S_IMODE(metadata.st_mode) != 0o700
                        if requires_private_mode
                        else metadata.st_mode & 0o022
                    )
                ):
                    raise LaunchWorkspaceError("published envelope directory is unsafe")
                continue
            if relative_path in immutable_control_files:
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o444
                ):
                    raise LaunchWorkspaceError("published control file is unsafe")
                continue
            if relative_path == checkpoint_file:
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o400
                    or metadata.st_size
                    > self._settings.launch.run_checkpoint_size_bytes
                ):
                    raise LaunchWorkspaceError(
                        "published run checkpoint file is unsafe"
                    )
                continue
            if relative_path == checkpoint_journal:
                observed_checkpoint_journal = True
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size
                    > self._settings.launch.run_checkpoint_journal_size_bytes
                ):
                    raise LaunchWorkspaceError(
                        "published run checkpoint journal is unsafe"
                    )
                continue
            if relative_path == checkpoint_lock:
                observed_checkpoint_lock = True
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size != 0
                ):
                    raise LaunchWorkspaceError(
                        "published run checkpoint lock is unsafe"
                    )
                continue
            if relative_path == runtime_lock:
                observed_runtime_lock = True
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size != 0
                ):
                    raise LaunchWorkspaceError("published runtime lock is unsafe")
                continue
            if relative_path in projection_files:
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o400
                    or metadata.st_size > projection_files[relative_path]
                ):
                    raise LaunchWorkspaceError(
                        "published run projection file is unsafe"
                    )
                continue
            if checkpoint_staging in relative_path.parents:
                staging_entry_count += 1
                if (
                    relative_path.parent != checkpoint_staging
                    or _RUN_CHECKPOINT_STAGING_PATTERN.fullmatch(relative_path.name)
                    is None
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
                    or metadata.st_size
                    > self._settings.launch.run_checkpoint_size_bytes
                ):
                    raise LaunchWorkspaceError(
                        "published run checkpoint staging entry is unsafe"
                    )
                continue
            if derived_state_store in relative_path.parents:
                derived_state_entry_count += 1
                if (
                    relative_path.parent != derived_state_store
                    or _RUN_DERIVED_STATE_OBJECT_PATTERN.fullmatch(relative_path.name)
                    is None
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o400
                    or metadata.st_size == 0
                    or metadata.st_size
                    > self._settings.launch.run_derived_generation_size_bytes
                ):
                    raise LaunchWorkspaceError(
                        "published derived-state object is unsafe"
                    )
                continue
            if derived_state_staging in relative_path.parents:
                derived_state_staging_entry_count += 1
                if (
                    relative_path.parent != derived_state_staging
                    or _RUN_DERIVED_STATE_STAGING_PATTERN.fullmatch(relative_path.name)
                    is None
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
                    or metadata.st_size
                    > self._settings.launch.run_derived_generation_size_bytes
                ):
                    raise LaunchWorkspaceError(
                        "published derived-state staging entry is unsafe"
                    )
                continue
            if action_workspace_staging in relative_path.parents:
                staged_relative_path = relative_path.relative_to(
                    action_workspace_staging
                )
                staged_root = staged_relative_path.parts[0]
                action_workspace_staging_roots.add(staged_root)
                action_workspace_staging_entry_count += 1
                inode_identity = (metadata.st_dev, metadata.st_ino)
                if (
                    _RUN_ACTION_WORKSPACE_STAGING_PATTERN.fullmatch(staged_root) is None
                    or metadata.st_dev != action_workspace_staging_metadata.st_dev
                    or metadata.st_uid != os.geteuid()
                    or inode_identity in action_workspace_staging_inodes
                    or metadata.st_mode & (stat.S_ISUID | stat.S_ISGID)
                    or (
                        stat.S_ISDIR(metadata.st_mode)
                        and stat.S_IMODE(metadata.st_mode) != 0o700
                    )
                    or (
                        stat.S_ISREG(metadata.st_mode)
                        and (
                            metadata.st_nlink != 1
                            or stat.S_IMODE(metadata.st_mode)
                            not in {0o400, 0o444, 0o600, 0o644, 0o700, 0o755}
                        )
                    )
                    or not (
                        stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
                    )
                    or (
                        len(staged_relative_path.parts) == 1
                        and not stat.S_ISDIR(metadata.st_mode)
                    )
                ):
                    raise LaunchWorkspaceError(
                        "published run action workspace staging entry is unsafe"
                    )
                action_workspace_staging_inodes.add(inode_identity)
                if stat.S_ISREG(metadata.st_mode):
                    action_workspace_staging_size_bytes += metadata.st_size
                continue
            if action_store in relative_path.parents:
                action_store_entry_count += 1
                is_fixed_lock = relative_path.name in {
                    "registry.lock",
                    "workspace.lock",
                }
                event_match = _RUN_ACTION_EVENT_PATTERN.fullmatch(relative_path.name)
                is_event = event_match is not None
                is_input = (
                    _RUN_ACTION_INPUT_PATTERN.fullmatch(relative_path.name) is not None
                )
                is_result = (
                    _RUN_ACTION_RESULT_PATTERN.fullmatch(relative_path.name) is not None
                )
                is_accepted = (
                    _RUN_ACTION_ACCEPTED_PATTERN.fullmatch(relative_path.name)
                    is not None
                )
                is_staging = (
                    _RUN_ACTION_STAGING_PATTERN.fullmatch(relative_path.name)
                    is not None
                )
                if event_match is not None:
                    action_store_operation_digests.add(event_match.group("operation"))
                action_store_size_bytes += metadata.st_size
                if (
                    relative_path.parent != action_store
                    or not any(
                        (
                            is_fixed_lock,
                            is_event,
                            is_input,
                            is_result,
                            is_accepted,
                            is_staging,
                        )
                    )
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or (
                        stat.S_IMODE(metadata.st_mode) != 0o600
                        if is_fixed_lock
                        else (
                            stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
                            if is_staging
                            else stat.S_IMODE(metadata.st_mode) != 0o400
                        )
                    )
                    or (is_fixed_lock and metadata.st_size != 0)
                    or (
                        is_event
                        and metadata.st_size
                        > self._settings.launch.run_action_event_size_bytes
                    )
                    or (
                        is_input
                        and metadata.st_size
                        > self._settings.launch.run_action_request_size_bytes
                    )
                    or (
                        (is_result or is_accepted)
                        and metadata.st_size
                        > self._settings.launch.run_action_result_size_bytes
                    )
                    or (
                        is_staging
                        and metadata.st_size
                        > max(
                            self._settings.launch.run_action_event_size_bytes,
                            self._settings.launch.run_action_request_size_bytes,
                            self._settings.launch.run_action_result_size_bytes,
                        )
                    )
                ):
                    raise LaunchWorkspaceError(
                        "published run action store entry is unsafe"
                    )
                continue
            raise LaunchWorkspaceError(
                "published run-root filesystem closure is not exact"
            )
        if (
            staging_entry_count
            > self._settings.launch.run_checkpoint_staging_entry_limit
        ):
            raise LaunchWorkspaceError(
                "published run checkpoint staging exceeds its entry bound"
            )
        if (
            derived_state_entry_count
            > self._settings.launch.run_derived_state_store_entry_limit
        ):
            raise LaunchWorkspaceError(
                "published derived-state store exceeds its entry bound"
            )
        if (
            derived_state_staging_entry_count
            > self._settings.launch.run_derived_state_staging_entry_limit
        ):
            raise LaunchWorkspaceError(
                "published derived-state staging exceeds its entry bound"
            )
        if (
            action_store_entry_count
            > self._settings.launch.run_action_store_entry_limit
        ):
            raise LaunchWorkspaceError(
                "published run action store exceeds its entry bound"
            )
        if (
            len(action_store_operation_digests)
            > self._settings.launch.run_action_operation_limit
            or action_store_size_bytes
            > self._settings.launch.run_action_store_size_bytes
        ):
            raise LaunchWorkspaceError(
                "published run action store exceeds its configured bounds"
            )
        if (
            len(action_workspace_staging_roots)
            > self._settings.launch.run_action_staging_entry_limit
            or action_workspace_staging_entry_count
            > (
                self._settings.launch.run_workspace_entry_limit
                + self._settings.launch.run_workspace_git_entry_limit
                + self._settings.launch.run_action_staging_entry_limit
            )
            or action_workspace_staging_size_bytes
            > (
                self._settings.launch.run_workspace_size_bytes
                + self._settings.launch.run_workspace_git_metadata_size_bytes
            )
        ):
            raise LaunchWorkspaceError(
                "published run action workspace staging exceeds its configured bounds"
            )
        if (
            not observed_checkpoint_journal
            or not observed_checkpoint_lock
            or not observed_runtime_lock
        ):
            raise LaunchWorkspaceError("published run control authority is incomplete")

    def _verify_git_baseline(
        self,
        workspace: Path,
        receipt: WorkspaceInstallationReceipt,
        descriptors: tuple[SourceFileDescriptor, ...],
    ) -> None:
        git_root = workspace / ".git"
        if (
            git_root.is_symlink()
            or not git_root.is_dir()
            or stat.S_IMODE(git_root.stat(follow_symlinks=False).st_mode) != 0o700
        ):
            raise LaunchWorkspaceError("published Git root is unsafe")
        reference = git_root / "refs" / "heads" / receipt.workspace_git_branch
        expected_head = f"ref: refs/heads/{receipt.workspace_git_branch}\n".encode(
            "utf-8"
        )
        expected_reference = f"{receipt.workspace_baseline_commit_sha}\n".encode(
            "ascii"
        )
        expected_config = self._git_config_bytes()
        head = _read_bounded_regular_file(
            git_root / "HEAD",
            len(expected_head),
            "Git HEAD",
            expected_size=len(expected_head),
            expected_mode=0o600,
        )
        reference_payload = _read_bounded_regular_file(
            reference,
            len(expected_reference),
            "Git branch reference",
            expected_size=len(expected_reference),
            expected_mode=0o600,
        )
        config_payload = _read_bounded_regular_file(
            git_root / "config",
            len(expected_config),
            "Git config",
            expected_size=len(expected_config),
            expected_mode=0o600,
        )
        if (
            head != expected_head
            or reference_payload != expected_reference
            or config_payload != expected_config
        ):
            raise LaunchWorkspaceError("published Git baseline metadata is unsafe")
        blob_ids = {
            descriptor.relative_path: git_object_sha(
                "blob",
                _read_bounded_regular_file(
                    workspace / descriptor.relative_path,
                    descriptor.size,
                    "Git source file",
                    expected_size=descriptor.size,
                    expected_mode=(0o755 if descriptor.mode == "100755" else 0o644),
                ),
            )
            for descriptor in descriptors
        }
        tree_ids = git_tree_shas(
            {
                descriptor.relative_path: (
                    blob_ids[descriptor.relative_path],
                    descriptor.mode,
                )
                for descriptor in descriptors
            }
        )
        if (
            tree_ids[""] != receipt.workspace_baseline_tree_sha
            or tuple(
                sorted(
                    {
                        *blob_ids.values(),
                        *tree_ids.values(),
                        receipt.workspace_baseline_commit_sha,
                    }
                )
            )
            != receipt.workspace_git_object_ids
        ):
            raise LaunchWorkspaceError(
                "published Git object closure differs from the expert source"
            )
        object_files = {
            path.parent.name + path.name: path
            for path in (git_root / "objects").glob("[0-9a-f][0-9a-f]/*")
            if path.is_file() and not path.is_symlink()
        }
        if set(object_files) != set(receipt.workspace_git_object_ids):
            raise LaunchWorkspaceError(
                "published Git loose-object closure is not exact"
            )
        for object_id in receipt.workspace_git_object_ids:
            kind = (
                "commit"
                if object_id == receipt.workspace_baseline_commit_sha
                else "tree" if object_id in tree_ids.values() else "blob"
            )
            self._verify_git_object(git_root, object_id, kind)
        commit_payload = self._read_git_object(
            git_root,
            receipt.workspace_baseline_commit_sha,
            "commit",
        )
        identity = (
            f"{self._settings.github.commit_author_name} "
            f"<{self._settings.github.commit_author_email}> {_GIT_COMMIT_TIME}"
        ).encode("utf-8")
        if commit_payload != self._git_commit_payload(
            receipt.workspace_baseline_tree_sha,
            identity,
        ):
            raise LaunchWorkspaceError(
                "published Git commit structure differs from the baseline protocol"
            )
        expected_index = self._git_index_payload(
            descriptors,
            blob_ids,
        )
        index_payload = _read_bounded_regular_file(
            git_root / "index",
            len(expected_index),
            "Git index",
            expected_size=len(expected_index),
            expected_mode=0o600,
        )
        if (
            index_payload != expected_index
            or tree_or_blob_digest(index_payload) != receipt.workspace_git_index_digest
        ):
            raise LaunchWorkspaceError(
                "published Git index differs from the deterministic baseline"
            )
        expected_control_files = {
            "HEAD",
            "config",
            "index",
            f"refs/heads/{receipt.workspace_git_branch}",
            *(
                f"objects/{object_id[:2]}/{object_id[2:]}"
                for object_id in receipt.workspace_git_object_ids
            ),
        }
        expected_directories = {
            "objects",
            "objects/info",
            "objects/pack",
            "refs",
            "refs/heads",
            *(
                f"objects/{object_id[:2]}"
                for object_id in receipt.workspace_git_object_ids
            ),
        }
        branch_path = PurePosixPath("refs/heads") / receipt.workspace_git_branch
        expected_directories.update(
            parent.as_posix()
            for parent in branch_path.parents
            if parent != PurePosixPath(".")
        )
        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        for path in git_root.rglob("*"):
            relative_path = path.relative_to(git_root).as_posix()
            if path.is_symlink():
                raise LaunchWorkspaceError("published Git metadata contains a symlink")
            metadata = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                if stat.S_IMODE(metadata.st_mode) != 0o700:
                    raise LaunchWorkspaceError(
                        "published Git metadata contains an unsafe directory mode"
                    )
                actual_directories.add(relative_path)
                continue
            expected_mode = 0o444 if relative_path.startswith("objects/") else 0o600
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != expected_mode
            ):
                raise LaunchWorkspaceError(
                    "published Git metadata contains unsafe file "
                    f"{relative_path} with mode {stat.S_IMODE(metadata.st_mode):o}"
                )
            actual_files.add(relative_path)
        if (
            actual_files != expected_control_files
            or actual_directories != expected_directories
        ):
            raise LaunchWorkspaceError(
                "published Git control-plane closure is not exact"
            )

    def _read_control_file(
        self,
        root_descriptor: int,
        directory_descriptors: Mapping[str, int],
        relative_path: str,
        maximum_bytes: int,
    ) -> tuple[bytes, tuple[int, int]]:
        descriptor = _open_layout_file(
            root_descriptor,
            directory_descriptors,
            relative_path,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > maximum_bytes
            or stat.S_IMODE(metadata.st_mode) != 0o444
        ):
            os.close(descriptor)
            raise LaunchWorkspaceError("launch control file exceeds its safe bounds")
        with os.fdopen(descriptor, "rb") as handle:
            payload = handle.read(maximum_bytes + 1)
            reopened = os.fstat(handle.fileno())
        if len(payload) > maximum_bytes or (
            reopened.st_dev,
            reopened.st_ino,
            reopened.st_size,
        ) != (metadata.st_dev, metadata.st_ino, metadata.st_size):
            raise LaunchWorkspaceError("launch control file changed while reading")
        return payload, (metadata.st_dev, metadata.st_ino)

    @staticmethod
    def _rename_no_replace(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
        expected_source_identity: tuple[int, int],
    ) -> int:
        source_metadata = os.stat(
            source_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(source_metadata.st_mode)
            or (source_metadata.st_dev, source_metadata.st_ino)
            != expected_source_identity
        ):
            raise LaunchWorkspaceError(
                "staging run-root identity changed before publication"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(libc, "renameat2"):
            raise LaunchWorkspaceError(
                "atomic no-replace run-root publication is unavailable"
            )
        rename_at2 = libc.renameat2
        rename_at2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_at2.restype = ctypes.c_int
        result = rename_at2(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise LaunchWorkspaceError("fresh run root already exists")
            raise OSError(
                error_number,
                os.strerror(error_number),
                destination_name,
            )
        final_descriptor = os.open(
            destination_name,
            os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        final_metadata = os.fstat(final_descriptor)
        if (
            not stat.S_ISDIR(final_metadata.st_mode)
            or (final_metadata.st_dev, final_metadata.st_ino)
            != expected_source_identity
        ):
            quarantine_result = rename_at2(
                parent_descriptor,
                os.fsencode(destination_name),
                parent_descriptor,
                os.fsencode(source_name),
                _RENAME_NOREPLACE,
            )
            os.fsync(parent_descriptor)
            os.close(final_descriptor)
            if quarantine_result != 0:
                raise LaunchWorkspaceError(
                    "substituted final run root could not be quarantined"
                )
            raise LaunchWorkspaceError(
                "published run-root identity differs from staging"
            )
        return final_descriptor

    @staticmethod
    def _descriptor_path(descriptor: int) -> Path:
        return Path(f"/proc/self/fd/{descriptor}")

    @staticmethod
    def _directory_identity(descriptor: int) -> tuple[int, int]:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise LaunchWorkspaceError(
                "workspace authority descriptor is not a directory"
            )
        return metadata.st_dev, metadata.st_ino

    @staticmethod
    def _path_directory_identity(path: Path) -> tuple[int, int]:
        metadata = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise LaunchWorkspaceError("published run root is unsafe")
        return metadata.st_dev, metadata.st_ino

    @staticmethod
    def _require_published_identity(
        parent_descriptor: int,
        published_name: str,
        expected: tuple[int, int],
    ) -> None:
        metadata = os.stat(
            published_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected
        ):
            raise LaunchWorkspaceError(
                "published run-root pathname no longer names its verified inode"
            )

    @staticmethod
    def _require_control_file_bound(
        payload: bytes,
        maximum_bytes: int,
        name: str,
    ) -> None:
        if len(payload) > maximum_bytes:
            raise LaunchWorkspaceError(f"{name} exceeds its configured launch bound")

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                os.fsync(descriptor)
                os.close(descriptor)
        for directory in sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            StarterWorkspaceBuilder._fsync_directory(directory)
        StarterWorkspaceBuilder._fsync_directory(root)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        os.fsync(descriptor)
        os.close(descriptor)


__all__ = [
    "ActiveLaunchWorkspace",
    "LaunchWorkspaceError",
    "PreparedLaunchWorkspace",
    "STARTER_WORKSPACE_INSTALLER_ID",
    "STARTER_WORKSPACE_INSTALLER_VERSION",
    "StarterWorkspaceBuilder",
]
