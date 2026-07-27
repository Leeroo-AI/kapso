"""Group-shared `.git`-free scratch authority for coding-agent providers."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.launch.workspace_frontier import (
    RunWorkspaceFrontierIdentity,
    RunWorkspaceRegularTreeIdentity,
    RunWorkspaceSourceTreeIdentity,
    copy_run_workspace_source_tree,
    inspect_detached_run_workspace_source_tree,
    plan_run_workspace_source_copy,
)
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CodingAgentRunActionRequest,
)
from kapso.cross_run.launch.run_action_coding_agent_layout import (
    PROVIDER_HOME_PATH,
    PROVIDER_OUTPUT_PATH,
    PROVIDER_SUPPORT_PATH,
    PROVIDER_WORKSPACE_PATH,
)

PROVIDER_SHARED_DIRECTORY_MODE = 0o2770
PROVIDER_SHARED_FILE_MODE = 0o660
PROVIDER_SHARED_EXECUTABLE_MODE = 0o770
PROVIDER_SUPPORT_DIRECTORY_MODE = 0o550
PROVIDER_SUPPORT_FILE_MODE = 0o440
PROVIDER_TEMPORARY_ROOT_MODE = 0o710

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


@dataclass(frozen=True)
class CodingAgentScratchSupportFile:
    """One retained immutable support file under the provider support root."""

    name: str
    descriptor: int
    payload: bytes
    metadata: tuple[int, ...]


@dataclass
class CodingAgentScratchLayout:
    """Live leaf descriptors and identities for one disposable provider call."""

    temporary_root_descriptor: int
    temporary_root_identity: tuple[int, int]
    supervisor_user_id: int
    supervisor_group_id: int
    provider_group_id: int
    workspace_descriptor: int
    home_descriptor: int
    output_descriptor: int
    support_descriptor: int
    baseline: CodingAgentScratchTreeIdentity
    support_files: tuple[CodingAgentScratchSupportFile, ...]
    temporary_root_restored: bool = False

    def restore_temporary_root(self) -> None:
        """Remove provider traversal authority before trusted result handling."""

        if self.temporary_root_restored:
            raise RunActionCodingAgentScratchError(
                "coding-agent temporary root was already restored"
            )
        metadata = os.fstat(self.temporary_root_descriptor)
        if (
            (metadata.st_dev, metadata.st_ino) != self.temporary_root_identity
            or metadata.st_uid != self.supervisor_user_id
            or metadata.st_gid != self.provider_group_id
            or stat.S_IMODE(metadata.st_mode) != PROVIDER_TEMPORARY_ROOT_MODE
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent temporary root changed before restoration"
            )
        os.fchown(
            self.temporary_root_descriptor,
            -1,
            self.supervisor_group_id,
        )
        os.fchmod(self.temporary_root_descriptor, 0o700)
        os.fsync(self.temporary_root_descriptor)
        restored = os.fstat(self.temporary_root_descriptor)
        if (
            (restored.st_dev, restored.st_ino) != self.temporary_root_identity
            or restored.st_uid != self.supervisor_user_id
            or restored.st_gid != self.supervisor_group_id
            or stat.S_IMODE(restored.st_mode) != 0o700
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent temporary root restoration is incomplete"
            )
        self.temporary_root_restored = True


def require_coding_agent_supervisor_identity(
    request: CodingAgentRunActionRequest,
) -> None:
    """Require the production supervisor's exact pre-provider identity."""

    if type(request) is not CodingAgentRunActionRequest:
        raise RunActionCodingAgentScratchError(
            "coding-agent supervisor request is invalid"
        )
    policy = request.interpretation_policy
    if (
        os.geteuid() != policy.supervisor_user_id
        or os.getegid() != policy.supervisor_group_id
        or tuple(sorted(os.getgroups()))
        != tuple(sorted({policy.supervisor_group_id, policy.provider_group_id}))
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent supervisor lacks its exact provider shared group"
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


def prepare_coding_agent_scratch_layout(
    *,
    trusted_workspace_descriptor: int,
    temporary_root_descriptor: int,
    trusted_frontier: RunWorkspaceFrontierIdentity,
    request: CodingAgentRunActionRequest,
    support_payloads: Mapping[str, bytes],
    resources: ExitStack,
) -> CodingAgentScratchLayout:
    """Build and retain the four exact provider leaf authorities."""

    if (
        type(trusted_frontier) is not RunWorkspaceFrontierIdentity
        or type(request) is not CodingAgentRunActionRequest
        or not isinstance(support_payloads, Mapping)
        or type(resources) is not ExitStack
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch preparation inputs are invalid"
        )
    policy = request.interpretation_policy
    if (
        os.geteuid() != policy.supervisor_user_id
        or os.getegid() != policy.supervisor_group_id
        or policy.provider_group_id not in os.getgroups()
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent supervisor cannot project its provider shared group"
        )
    temporary_metadata = os.fstat(temporary_root_descriptor)
    if (
        not stat.S_ISDIR(temporary_metadata.st_mode)
        or temporary_metadata.st_uid != policy.supervisor_user_id
        or temporary_metadata.st_gid != policy.supervisor_group_id
        or stat.S_IMODE(temporary_metadata.st_mode) != 0o700
        or tuple(os.listdir(temporary_root_descriptor))
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent temporary root is not empty private authority"
        )
    os.fchown(temporary_root_descriptor, -1, policy.provider_group_id)
    os.fchmod(temporary_root_descriptor, PROVIDER_TEMPORARY_ROOT_MODE)
    os.fsync(temporary_root_descriptor)
    temporary_shared = os.fstat(temporary_root_descriptor)
    if (
        (temporary_shared.st_dev, temporary_shared.st_ino)
        != (temporary_metadata.st_dev, temporary_metadata.st_ino)
        or temporary_shared.st_uid != policy.supervisor_user_id
        or temporary_shared.st_gid != policy.provider_group_id
        or stat.S_IMODE(temporary_shared.st_mode) != PROVIDER_TEMPORARY_ROOT_MODE
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent temporary root was not shared exactly"
        )
    workspace_descriptor = _create_leaf_directory(
        temporary_root_descriptor,
        PurePosixPath(PROVIDER_WORKSPACE_PATH).name,
        initial_mode=0o700,
        final_mode=0o700,
        group_id=policy.supervisor_group_id,
        resources=resources,
    )
    plan = plan_run_workspace_source_copy(
        trusted_workspace_descriptor,
        expected=trusted_frontier,
        maximum_source_entries=policy.maximum_workspace_entries,
        maximum_source_bytes=policy.maximum_workspace_bytes,
        maximum_git_entries=policy.maximum_workspace_git_entries,
        maximum_git_bytes=policy.maximum_workspace_git_bytes,
    )
    detached = copy_run_workspace_source_tree(
        trusted_workspace_descriptor,
        workspace_descriptor,
        plan=plan,
        maximum_source_entries=policy.maximum_workspace_entries,
        maximum_source_bytes=policy.maximum_workspace_bytes,
        maximum_git_entries=policy.maximum_workspace_git_entries,
        maximum_git_bytes=policy.maximum_workspace_git_bytes,
    )
    baseline = share_coding_agent_scratch_source_tree(
        workspace_descriptor,
        expected_source=detached,
        supervisor_user_id=policy.supervisor_user_id,
        provider_user_id=policy.provider_user_id,
        provider_group_id=policy.provider_group_id,
        maximum_entries=policy.maximum_workspace_entries,
        maximum_bytes=policy.maximum_workspace_bytes,
    )
    home_descriptor = _create_leaf_directory(
        temporary_root_descriptor,
        PurePosixPath(PROVIDER_HOME_PATH).name,
        initial_mode=0o700,
        final_mode=PROVIDER_SHARED_DIRECTORY_MODE,
        group_id=policy.provider_group_id,
        resources=resources,
    )
    output_descriptor = _create_leaf_directory(
        temporary_root_descriptor,
        PurePosixPath(PROVIDER_OUTPUT_PATH).name,
        initial_mode=0o700,
        final_mode=PROVIDER_SHARED_DIRECTORY_MODE,
        group_id=policy.provider_group_id,
        resources=resources,
    )
    support_descriptor = _create_leaf_directory(
        temporary_root_descriptor,
        PurePosixPath(PROVIDER_SUPPORT_PATH).name,
        initial_mode=0o700,
        final_mode=0o750,
        group_id=policy.provider_group_id,
        resources=resources,
    )
    support_files = _materialize_support_files(
        support_descriptor,
        support_payloads,
        provider_group_id=policy.provider_group_id,
        resources=resources,
    )
    os.fchmod(support_descriptor, PROVIDER_SUPPORT_DIRECTORY_MODE)
    os.fsync(support_descriptor)
    os.fsync(temporary_root_descriptor)
    identities = {
        (metadata.st_dev, metadata.st_ino)
        for metadata in (
            os.fstat(workspace_descriptor),
            os.fstat(home_descriptor),
            os.fstat(output_descriptor),
            os.fstat(support_descriptor),
        )
    }
    if len(identities) != 4:
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch leaf authorities overlap"
        )
    return CodingAgentScratchLayout(
        temporary_root_descriptor=temporary_root_descriptor,
        temporary_root_identity=(temporary_metadata.st_dev, temporary_metadata.st_ino),
        supervisor_user_id=policy.supervisor_user_id,
        supervisor_group_id=policy.supervisor_group_id,
        provider_group_id=policy.provider_group_id,
        workspace_descriptor=workspace_descriptor,
        home_descriptor=home_descriptor,
        output_descriptor=output_descriptor,
        support_descriptor=support_descriptor,
        baseline=baseline,
        support_files=support_files,
    )


def sanitize_coding_agent_scratch_successor(
    layout: CodingAgentScratchLayout,
    *,
    request: CodingAgentRunActionRequest,
    resources: ExitStack,
) -> tuple[int, RunWorkspaceSourceTreeIdentity]:
    """Copy one closed shared successor into fresh supervisor-private authority."""

    if (
        type(layout) is not CodingAgentScratchLayout
        or type(request) is not CodingAgentRunActionRequest
        or type(resources) is not ExitStack
        or not layout.temporary_root_restored
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch cannot be sanitized before provider closure"
        )
    policy = request.interpretation_policy
    observed_before = inspect_coding_agent_scratch_source_tree(
        layout.workspace_descriptor,
        supervisor_user_id=policy.supervisor_user_id,
        provider_user_id=policy.provider_user_id,
        provider_group_id=policy.provider_group_id,
        maximum_entries=policy.maximum_workspace_entries,
        maximum_bytes=policy.maximum_workspace_bytes,
    )
    if observed_before.source.source_tree_digest == (
        layout.baseline.source.source_tree_digest
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent edit did not change the source tree"
        )
    destination_descriptor = _create_leaf_directory(
        layout.temporary_root_descriptor,
        "accepted-source",
        initial_mode=0o700,
        final_mode=0o700,
        group_id=policy.supervisor_group_id,
        resources=resources,
    )
    source_root = os.fstat(layout.workspace_descriptor)
    _copy_shared_directory_to_private(
        layout.workspace_descriptor,
        destination_descriptor,
        root_device=source_root.st_dev,
        allowed_user_ids=frozenset(
            {policy.supervisor_user_id, policy.provider_user_id}
        ),
        provider_group_id=policy.provider_group_id,
    )
    os.fsync(destination_descriptor)
    sanitized = inspect_detached_run_workspace_source_tree(
        destination_descriptor,
        maximum_entries=policy.maximum_workspace_entries,
        maximum_bytes=policy.maximum_workspace_bytes,
    )
    observed_after = inspect_coding_agent_scratch_source_tree(
        layout.workspace_descriptor,
        supervisor_user_id=policy.supervisor_user_id,
        provider_user_id=policy.provider_user_id,
        provider_group_id=policy.provider_group_id,
        maximum_entries=policy.maximum_workspace_entries,
        maximum_bytes=policy.maximum_workspace_bytes,
    )
    if (
        observed_after != observed_before
        or sanitized.source_tree_digest != observed_before.source.source_tree_digest
        or sanitized.source_entry_count != observed_before.source.source_entry_count
        or sanitized.source_size_bytes != observed_before.source.source_size_bytes
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent sanitized successor differs from closed scratch"
        )
    return destination_descriptor, sanitized


def require_coding_agent_scratch_support(
    layout: CodingAgentScratchLayout,
) -> None:
    """Require every retained support inode and path to remain byte-identical."""

    if type(layout) is not CodingAgentScratchLayout or not layout.support_files:
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch support authority is absent"
        )
    if stat.S_IMODE(os.fstat(layout.support_descriptor).st_mode) != (
        PROVIDER_SUPPORT_DIRECTORY_MODE
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch support directory changed"
        )
    if tuple(sorted(os.listdir(layout.support_descriptor))) != tuple(
        file.name for file in layout.support_files
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch support topology changed"
        )
    for support_file in layout.support_files:
        if _stable_metadata(os.fstat(support_file.descriptor)) != support_file.metadata:
            raise RunActionCodingAgentScratchError(
                "coding-agent scratch support inode changed"
            )
        descriptor = os.open(
            support_file.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=layout.support_descriptor,
        )
        with os.fdopen(descriptor, "rb") as handle:
            payload = handle.read(len(support_file.payload) + 1)
            metadata = os.fstat(handle.fileno())
        if (
            payload != support_file.payload
            or _stable_metadata(metadata) != support_file.metadata
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent scratch support path was substituted"
            )


def _create_leaf_directory(
    parent_descriptor: int,
    name: str,
    *,
    initial_mode: int,
    final_mode: int,
    group_id: int,
    resources: ExitStack,
) -> int:
    if (
        not isinstance(name, str)
        or not name
        or PurePosixPath(name).name != name
        or name in {".", ".."}
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch leaf name is invalid"
        )
    os.mkdir(name, mode=initial_mode, dir_fd=parent_descriptor)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    resources.callback(os.close, descriptor)
    os.fchown(descriptor, -1, group_id)
    os.fchmod(descriptor, final_mode)
    os.fsync(descriptor)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != group_id
        or stat.S_IMODE(metadata.st_mode) != final_mode
        or tuple(os.listdir(descriptor))
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent scratch leaf was not created exactly"
        )
    return descriptor


def _materialize_support_files(
    support_descriptor: int,
    support_payloads: Mapping[str, bytes],
    *,
    provider_group_id: int,
    resources: ExitStack,
) -> tuple[CodingAgentScratchSupportFile, ...]:
    observed = []
    for absolute_path, payload in sorted(support_payloads.items()):
        path = PurePosixPath(absolute_path)
        if (
            path.parent.as_posix() != PROVIDER_SUPPORT_PATH
            or not path.name
            or type(payload) is not bytes
            or not payload
            or path.name in os.listdir(support_descriptor)
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent support projection is invalid"
            )
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=support_descriptor,
        )
        resources.callback(os.close, descriptor)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise RunActionCodingAgentScratchError(
                    "coding-agent support write made no progress"
                )
            remaining = remaining[written:]
        os.fchown(descriptor, -1, provider_group_id)
        os.fchmod(descriptor, PROVIDER_SUPPORT_FILE_MODE)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_gid != provider_group_id
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != PROVIDER_SUPPORT_FILE_MODE
            or metadata.st_size != len(payload)
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent support file publication is invalid"
            )
        observed.append(
            CodingAgentScratchSupportFile(
                name=path.name,
                descriptor=descriptor,
                payload=payload,
                metadata=_stable_metadata(metadata),
            )
        )
    if not observed:
        raise RunActionCodingAgentScratchError(
            "coding-agent support projection is empty"
        )
    return tuple(observed)


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


def _copy_shared_directory_to_private(
    source_descriptor: int,
    destination_descriptor: int,
    *,
    root_device: int,
    allowed_user_ids: frozenset[int],
    provider_group_id: int,
) -> None:
    with os.scandir(source_descriptor) as iterator:
        observed = tuple(
            sorted(
                (entry.name, entry.stat(follow_symlinks=False)) for entry in iterator
            )
        )
    for name, expected in observed:
        _require_source_component(name)
        current = os.stat(name, dir_fd=source_descriptor, follow_symlinks=False)
        if (
            _stable_metadata(current) != _stable_metadata(expected)
            or expected.st_dev != root_device
            or expected.st_uid not in allowed_user_ids
            or expected.st_gid != provider_group_id
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent shared scratch changed before sanitization"
            )
        permissions = stat.S_IMODE(expected.st_mode)
        if stat.S_ISDIR(expected.st_mode):
            if permissions != PROVIDER_SHARED_DIRECTORY_MODE:
                raise RunActionCodingAgentScratchError(
                    "coding-agent shared directory is unsafe during sanitization"
                )
            os.mkdir(name, mode=0o700, dir_fd=destination_descriptor)
            source_child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=source_descriptor,
            )
            destination_child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=destination_descriptor,
            )
            with ExitStack() as descriptors:
                descriptors.callback(os.close, source_child)
                descriptors.callback(os.close, destination_child)
                if _stable_metadata(os.fstat(source_child)) != _stable_metadata(
                    expected
                ):
                    raise RunActionCodingAgentScratchError(
                        "coding-agent shared directory changed while opening"
                    )
                _copy_shared_directory_to_private(
                    source_child,
                    destination_child,
                    root_device=root_device,
                    allowed_user_ids=allowed_user_ids,
                    provider_group_id=provider_group_id,
                )
                if not os.listdir(destination_child):
                    raise RunActionCodingAgentScratchError(
                        "coding-agent sanitized source contains an empty directory"
                    )
                os.fchmod(destination_child, 0o700)
                os.fsync(destination_child)
        elif (
            stat.S_ISREG(expected.st_mode)
            and expected.st_nlink == 1
            and permissions
            in {PROVIDER_SHARED_FILE_MODE, PROVIDER_SHARED_EXECUTABLE_MODE}
        ):
            _copy_shared_regular_file_to_private(
                source_descriptor,
                destination_descriptor,
                name=name,
                expected=expected,
                executable=permissions == PROVIDER_SHARED_EXECUTABLE_MODE,
            )
        else:
            raise RunActionCodingAgentScratchError(
                "coding-agent shared entry is unsafe during sanitization"
            )
        rebound = os.stat(name, dir_fd=source_descriptor, follow_symlinks=False)
        if _stable_metadata(rebound) != _stable_metadata(expected):
            raise RunActionCodingAgentScratchError(
                "coding-agent shared entry changed during sanitization"
            )
    if tuple(sorted(os.listdir(source_descriptor))) != tuple(
        name for name, _metadata in observed
    ):
        raise RunActionCodingAgentScratchError(
            "coding-agent shared topology changed during sanitization"
        )


def _copy_shared_regular_file_to_private(
    source_parent_descriptor: int,
    destination_parent_descriptor: int,
    *,
    name: str,
    expected: os.stat_result,
    executable: bool,
) -> None:
    source_descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=source_parent_descriptor,
    )
    destination_descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=destination_parent_descriptor,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, source_descriptor)
        descriptors.callback(os.close, destination_descriptor)
        if _stable_metadata(os.fstat(source_descriptor)) != _stable_metadata(expected):
            raise RunActionCodingAgentScratchError(
                "coding-agent shared file changed while opening"
            )
        remaining_bytes = expected.st_size
        while remaining_bytes:
            written_bytes = os.sendfile(
                destination_descriptor,
                source_descriptor,
                None,
                remaining_bytes,
            )
            if written_bytes <= 0:
                raise RunActionCodingAgentScratchError(
                    "coding-agent shared file ended during sanitization"
                )
            remaining_bytes -= written_bytes
        os.fchmod(destination_descriptor, 0o700 if executable else 0o600)
        os.fsync(destination_descriptor)
        copied = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(copied.st_mode)
            or copied.st_uid != os.geteuid()
            or copied.st_gid != os.getegid()
            or copied.st_nlink != 1
            or stat.S_IMODE(copied.st_mode) != (0o700 if executable else 0o600)
            or copied.st_size != expected.st_size
            or _stable_metadata(os.fstat(source_descriptor))
            != _stable_metadata(expected)
        ):
            raise RunActionCodingAgentScratchError(
                "coding-agent sanitized file is incomplete or unsafe"
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
    "CodingAgentScratchLayout",
    "CodingAgentScratchSupportFile",
    "CodingAgentScratchTreeIdentity",
    "inspect_coding_agent_scratch_source_tree",
    "prepare_coding_agent_scratch_layout",
    "PROVIDER_SHARED_DIRECTORY_MODE",
    "PROVIDER_SHARED_EXECUTABLE_MODE",
    "PROVIDER_SHARED_FILE_MODE",
    "require_coding_agent_scratch_support",
    "require_coding_agent_supervisor_identity",
    "RunActionCodingAgentScratchError",
    "sanitize_coding_agent_scratch_successor",
    "share_coding_agent_scratch_source_tree",
]
