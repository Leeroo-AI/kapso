"""Exact private workspaces for expert candidate proposal operations."""

from __future__ import annotations

import fcntl
import os
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType

from kapso.cross_run.contracts import EMPTY_EXPERT_TREE_DIGEST, SourceFileDescriptor
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_control_paths,
    expert_module_contract_path,
)
from kapso.cross_run.expert.candidates import ExpertCandidateValidationError
from kapso.cross_run.expert.topology import validate_expert_tree_ownership
from kapso.cross_run.expert.triggers import ExpertTriggerEvidencePacket
from kapso.cross_run.github.materializer import (
    GitHubArtifactMaterializer,
    MaterializedArtifact,
)
from kapso.cross_run.settings import ExpertSettings
from kapso.execution.coding_agents.workspace_delta import (
    CodingAgentWorkspaceSnapshot,
    inspect_coding_agent_workspace_descriptor,
)

_WORKSPACE_NAME_PREFIX = "workspace-"


class ExpertCandidateWorkspaceError(ValueError):
    """An expert candidate workspace is unsafe or differs from its source base."""


@dataclass(frozen=True)
class PreparedExpertCandidateWorkspace:
    path: Path
    source_base_tree_hash: str
    source_base_files: tuple[SourceFileDescriptor, ...]
    editable_snapshot: CodingAgentWorkspaceSnapshot


class ExpertCandidateWorkspaceLease:
    """Own one prepared workspace until candidate sealing finishes."""

    def __init__(
        self,
        prepared: PreparedExpertCandidateWorkspace,
        state_path: Path,
        state_descriptor: int,
        state_identity: tuple[int, int],
        root_descriptor: int,
        root_name: str,
        root_identity: tuple[int, int],
        workspace_descriptor: int,
        workspace_name: str,
        workspace_identity: tuple[int, int],
    ) -> None:
        self.prepared = prepared
        self._state_path = state_path
        self._state_descriptor = state_descriptor
        self._state_identity = state_identity
        self._root_descriptor = root_descriptor
        self._root_name = root_name
        self._root_identity = root_identity
        self._workspace_descriptor = workspace_descriptor
        self._workspace_name = workspace_name
        self._workspace_identity = workspace_identity
        self._entered = False
        self._closed = False

    def __enter__(self) -> PreparedExpertCandidateWorkspace:
        if self._entered or self._closed:
            raise ExpertCandidateWorkspaceError(
                "expert candidate workspace lease cannot be re-entered"
            )
        with ExitStack() as failed_entry:
            failed_entry.callback(self._abort_failed_entry)
            self._validate_binding()
            self._entered = True
            failed_entry.pop_all()
            return self.prepared

    def validate(self) -> None:
        """Prove the named workspace still resolves to every pinned inode."""

        if self._closed:
            raise ExpertCandidateWorkspaceError(
                "expert candidate workspace lease is already closed"
            )
        self._validate_binding()

    @property
    def workspace_authority_descriptor(self) -> int:
        """Return the active descriptor authority for one editable agent call."""

        if not self._entered or self._closed:
            raise ExpertCandidateWorkspaceError(
                "expert workspace authority requires an active lease"
            )
        return self._workspace_descriptor

    def _validate_binding(self) -> None:
        _require_directory_path_identity(
            self._state_path,
            self._state_identity,
            "expert workspace state root",
        )
        _require_opened_directory_identity(
            self._state_descriptor,
            self._state_identity,
            "expert workspace state descriptor",
        )
        _require_directory_entry_identity(
            self._state_descriptor,
            self._root_name,
            self._root_identity,
            "expert workspace root",
        )
        _require_opened_directory_identity(
            self._root_descriptor,
            self._root_identity,
            "expert workspace root descriptor",
        )
        if (
            _directory_identity_at(
                self._root_descriptor,
                self._workspace_name,
            )
            != self._workspace_identity
        ):
            raise ExpertCandidateWorkspaceError(
                "expert candidate workspace identity changed before use"
            )
        opened = os.fstat(self._workspace_descriptor)
        if (opened.st_dev, opened.st_ino) != self._workspace_identity:
            raise ExpertCandidateWorkspaceError(
                "expert candidate workspace descriptor changed before use"
            )

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            raise ExpertCandidateWorkspaceError(
                "expert candidate workspace lease is already closed"
            )
        self._closed = True
        self._release(validate_binding=True)

    def _abort_failed_entry(self) -> None:
        self._closed = True
        self._release(validate_binding=False)

    def _release(self, *, validate_binding: bool) -> None:
        with ExitStack() as descriptors:
            descriptors.callback(os.close, self._state_descriptor)
            descriptors.callback(os.close, self._root_descriptor)
            descriptors.callback(os.close, self._workspace_descriptor)
            descriptors.callback(
                _remove_untrusted_workspace,
                self._root_descriptor,
                self._workspace_descriptor,
                self._workspace_name,
                self._workspace_identity,
            )
            if validate_binding:
                self._validate_binding()


class ExpertCandidateWorkspaceManager:
    """Prepare exact empty or released-source-base workspaces under one private root."""

    def __init__(
        self,
        root: Path,
        state_root: Path,
        settings: ExpertSettings,
        materializer: GitHubArtifactMaterializer,
    ) -> None:
        self._validate_state_root(state_root)
        if (
            not root.is_absolute()
            or root != Path(os.path.abspath(root))
            or root.parent != state_root
            or root.name in {"", ".", ".."}
        ):
            raise ExpertCandidateWorkspaceError(
                "expert workspace root must be a direct normalized state child"
            )
        self.root = root
        self.state_root = state_root
        self.settings = settings
        self.materializer = materializer
        state_metadata = state_root.stat(follow_symlinks=False)
        self._state_identity = state_metadata.st_dev, state_metadata.st_ino
        with _WorkspaceInitializationLock(state_root / f".{root.name}.lock"):
            if os.path.lexists(root):
                self._validate_private_directory(root, "expert workspace root")
            else:
                os.mkdir(root, mode=0o700)
                self._fsync_directory(state_root)
            root_metadata = root.stat(follow_symlinks=False)
            self._root_identity = root_metadata.st_dev, root_metadata.st_ino

    def lease(
        self,
        *,
        trigger_packet: ExpertTriggerEvidencePacket,
        materialized_source_base: MaterializedArtifact | None,
    ) -> ExpertCandidateWorkspaceLease:
        workspace_name = _WORKSPACE_NAME_PREFIX + secrets.token_hex(16)
        workspace_path = self.root / workspace_name
        with ExitStack() as preparation:
            state_descriptor = os.open(
                self.state_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            preparation.callback(os.close, state_descriptor)
            self._validate_private_metadata(
                os.fstat(state_descriptor),
                "expert workspace state root",
            )
            _require_opened_directory_identity(
                state_descriptor,
                self._state_identity,
                "expert workspace state root",
            )
            _require_directory_path_identity(
                self.state_root,
                self._state_identity,
                "expert workspace state root",
            )
            parent_descriptor = os.open(
                self.root.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=state_descriptor,
            )
            preparation.callback(os.close, parent_descriptor)
            self._validate_private_metadata(
                os.fstat(parent_descriptor),
                "expert workspace root",
            )
            _require_opened_directory_identity(
                parent_descriptor,
                self._root_identity,
                "expert workspace root",
            )
            _require_directory_entry_identity(
                state_descriptor,
                self.root.name,
                self._root_identity,
                "expert workspace root",
            )
            construction_cleanup = _WorkspaceConstructionCleanup(
                parent_descriptor,
                workspace_name,
            )
            preparation.callback(construction_cleanup.remove)
            if trigger_packet.source_base_release is None:
                if materialized_source_base is not None:
                    raise ExpertCandidateWorkspaceError(
                        "bootstrap workspace cannot receive a materialized source base"
                    )
                os.mkdir(workspace_name, mode=0o700, dir_fd=parent_descriptor)
            else:
                self._materialize_released_parent(
                    trigger_packet,
                    materialized_source_base,
                    workspace_path,
                    parent_descriptor,
                )
            workspace_identity = _directory_identity_at(
                parent_descriptor,
                workspace_name,
            )
            construction_cleanup.expected_identity = workspace_identity
            workspace_descriptor = os.open(
                workspace_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_descriptor,
            )
            preparation.callback(os.close, workspace_descriptor)
            if (
                os.fstat(workspace_descriptor).st_dev,
                os.fstat(workspace_descriptor).st_ino,
            ) != workspace_identity:
                raise ExpertCandidateWorkspaceError(
                    "expert candidate workspace changed while opening"
                )
            prepared = self._prepare_workspace(
                trigger_packet,
                workspace_path,
                workspace_descriptor,
            )
            if (
                _directory_identity_at(
                    parent_descriptor,
                    workspace_name,
                )
                != workspace_identity
            ):
                raise ExpertCandidateWorkspaceError(
                    "expert candidate workspace changed during preparation"
                )
            _require_directory_path_identity(
                self.state_root,
                self._state_identity,
                "expert workspace state root",
            )
            _require_directory_entry_identity(
                state_descriptor,
                self.root.name,
                self._root_identity,
                "expert workspace root",
            )
            _require_opened_directory_identity(
                parent_descriptor,
                self._root_identity,
                "expert workspace root descriptor",
            )
            lease = ExpertCandidateWorkspaceLease(
                prepared,
                self.state_root,
                state_descriptor,
                self._state_identity,
                parent_descriptor,
                self.root.name,
                self._root_identity,
                workspace_descriptor,
                workspace_name,
                workspace_identity,
            )
            preparation.pop_all()
            return lease

    def _materialize_released_parent(
        self,
        packet: ExpertTriggerEvidencePacket,
        materialized_source_base: MaterializedArtifact | None,
        workspace_path: Path,
        workspace_parent_descriptor: int,
    ) -> None:
        if (
            materialized_source_base is None
            or packet.source_base_tree_receipt is None
            or materialized_source_base.receipt
            != packet.source_base_tree_receipt.cache_verification_receipt
        ):
            raise ExpertCandidateWorkspaceError(
                "released workspace requires its exact materialized source base"
            )
        observed = self.materializer.extract_verified_source_archive(
            materialized=materialized_source_base,
            expected=packet.source_base_tree_receipt.source_extraction_receipt,
            destination=workspace_path,
            destination_parent_descriptor=workspace_parent_descriptor,
        )
        if observed != packet.source_base_tree_receipt.source_extraction_receipt:
            raise ExpertCandidateWorkspaceError(
                "released workspace extraction differs from its trigger receipt"
            )

    def _prepare_workspace(
        self,
        packet: ExpertTriggerEvidencePacket,
        workspace_path: Path,
        workspace_descriptor: int,
    ) -> PreparedExpertCandidateWorkspace:
        full_snapshot = inspect_coding_agent_workspace_descriptor(
            workspace_descriptor,
            maximum_entries=self.settings.candidate_entry_limit,
            maximum_bytes=self.settings.candidate_byte_limit,
        )
        if packet.source_base_release is None:
            if full_snapshot.tree_hash != EMPTY_EXPERT_TREE_DIGEST:
                raise ExpertCandidateWorkspaceError(
                    "bootstrap workspace differs from the canonical empty tree"
                )
            return PreparedExpertCandidateWorkspace(
                path=workspace_path,
                source_base_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
                source_base_files=(),
                editable_snapshot=full_snapshot,
            )
        self._validate_released_parent(packet, full_snapshot)
        source_base_files = tuple(file.descriptor for file in full_snapshot.files)
        control_paths = set(expert_control_paths(packet.source_base_module_contracts))
        expected_editable_files = tuple(
            file.descriptor
            for file in full_snapshot.files
            if file.descriptor.relative_path not in control_paths
        )
        self._remove_generated_controls(workspace_descriptor, packet)
        editable_snapshot = inspect_coding_agent_workspace_descriptor(
            workspace_descriptor,
            maximum_entries=self.settings.candidate_entry_limit,
            maximum_bytes=self.settings.candidate_byte_limit,
        )
        if (
            tuple(file.descriptor for file in editable_snapshot.files)
            != expected_editable_files
        ):
            raise ExpertCandidateWorkspaceError(
                "released workspace changed outside generated controls"
            )
        return PreparedExpertCandidateWorkspace(
            path=workspace_path,
            source_base_tree_hash=packet.source_base_tree_hash,
            source_base_files=source_base_files,
            editable_snapshot=editable_snapshot,
        )

    @staticmethod
    def _validate_released_parent(
        packet: ExpertTriggerEvidencePacket,
        snapshot: CodingAgentWorkspaceSnapshot,
    ) -> None:
        if (
            packet.source_base_tree_receipt is None
            or packet.source_base_scope_contract is None
            or packet.source_base_repository_map is None
        ):
            raise ExpertCandidateWorkspaceError(
                "released workspace packet omits its source-base topology"
            )
        expected_extraction = packet.source_base_tree_receipt.source_extraction_receipt
        descriptors = tuple(file.descriptor for file in snapshot.files)
        if (
            descriptors != expected_extraction.source_tree_files
            or snapshot.tree_hash != expected_extraction.source_tree_hash
            or snapshot.tree_hash != packet.source_base_tree_hash
        ):
            raise ExpertCandidateWorkspaceError(
                "released workspace differs from its exact source receipt"
            )
        files = snapshot.by_path()
        controls = set(expert_control_paths(packet.source_base_module_contracts))
        control_root = PurePosixPath(EXPERT_REPOSITORY_MAP_PATH).parent
        observed_controls = {
            path
            for path in files
            if path == EXPERT_BOOK_PATH
            or PurePosixPath(path) == control_root
            or control_root in PurePosixPath(path).parents
        }
        if observed_controls != controls:
            raise ExpertCandidateWorkspaceError(
                "released workspace expert control closure is invalid"
            )
        expected_control_bytes = {
            EXPERT_BOOK_PATH: compile_expert_semantic_book(
                packet.source_base_scope_contract,
                packet.source_base_repository_map,
                packet.source_base_module_contracts,
            ),
            EXPERT_REPOSITORY_MAP_PATH: packet.source_base_repository_map.to_json_bytes(),
            **{
                expert_module_contract_path(module.module_contract_id): (
                    module.to_json_bytes()
                )
                for module in packet.source_base_module_contracts
            },
        }
        if any(
            files[path].content != payload
            for path, payload in expected_control_bytes.items()
        ):
            raise ExpertCandidateWorkspaceError(
                "released workspace control bytes differ from typed topology"
            )
        validate_expert_tree_ownership(
            packet.source_base_repository_map,
            packet.source_base_module_contracts,
            {path: file.descriptor for path, file in files.items()},
            validation_error_type=ExpertCandidateValidationError,
        )

    @staticmethod
    def _remove_generated_controls(
        workspace_descriptor: int,
        packet: ExpertTriggerEvidencePacket,
    ) -> None:
        expected_modules = {
            PurePosixPath(
                expert_module_contract_path(module.module_contract_id)
            ).name: module.to_json_bytes()
            for module in packet.source_base_module_contracts
        }
        with ExitStack() as descriptors:
            kapso_descriptor, kapso_identity = _open_real_directory_at(
                workspace_descriptor,
                ".kapso",
                descriptors,
                "expert control root",
            )
            expert_descriptor, expert_identity = _open_real_directory_at(
                kapso_descriptor,
                "expert",
                descriptors,
                "expert control directory",
            )
            modules_descriptor, modules_identity = _open_real_directory_at(
                expert_descriptor,
                "module-contracts",
                descriptors,
                "expert module-contract directory",
            )
            if set(os.listdir(workspace_descriptor)).intersection(
                {EXPERT_BOOK_PATH, ".kapso"}
            ) != {EXPERT_BOOK_PATH, ".kapso"}:
                raise ExpertCandidateWorkspaceError(
                    "released workspace control roots changed before removal"
                )
            if "expert" not in os.listdir(kapso_descriptor):
                raise ExpertCandidateWorkspaceError(
                    "released workspace expert control root changed before removal"
                )
            if set(os.listdir(expert_descriptor)) != {
                "repository-map.json",
                "module-contracts",
            }:
                raise ExpertCandidateWorkspaceError(
                    "released workspace expert control closure changed before removal"
                )
            if set(os.listdir(modules_descriptor)) != set(expected_modules):
                raise ExpertCandidateWorkspaceError(
                    "released workspace module controls changed before removal"
                )
            expected_book = compile_expert_semantic_book(
                packet.source_base_scope_contract,
                packet.source_base_repository_map,
                packet.source_base_module_contracts,
            )
            if (
                _read_regular_file_at(
                    workspace_descriptor,
                    EXPERT_BOOK_PATH,
                )
                != expected_book
            ):
                raise ExpertCandidateWorkspaceError(
                    "released workspace expert book changed before removal"
                )
            if (
                _read_regular_file_at(
                    expert_descriptor,
                    "repository-map.json",
                )
                != packet.source_base_repository_map.to_json_bytes()
            ):
                raise ExpertCandidateWorkspaceError(
                    "released workspace repository map changed before removal"
                )
            for module_name, expected_bytes in sorted(expected_modules.items()):
                if (
                    _read_regular_file_at(modules_descriptor, module_name)
                    != expected_bytes
                ):
                    raise ExpertCandidateWorkspaceError(
                        "released workspace module contract changed before removal"
                    )
            for module_name in sorted(expected_modules):
                os.unlink(module_name, dir_fd=modules_descriptor)
            os.fsync(modules_descriptor)
            os.unlink("repository-map.json", dir_fd=expert_descriptor)
            os.unlink(EXPERT_BOOK_PATH, dir_fd=workspace_descriptor)
            _remove_empty_pinned_directory(
                expert_descriptor,
                modules_descriptor,
                "module-contracts",
                modules_identity,
            )
            _remove_empty_pinned_directory(
                kapso_descriptor,
                expert_descriptor,
                "expert",
                expert_identity,
            )
            if not os.listdir(kapso_descriptor):
                _remove_empty_pinned_directory(
                    workspace_descriptor,
                    kapso_descriptor,
                    ".kapso",
                    kapso_identity,
                )
            os.fsync(workspace_descriptor)

    @staticmethod
    def _validate_state_root(path: Path) -> None:
        if (
            not path.is_absolute()
            or path != Path(os.path.abspath(path))
            or path in {Path("/"), Path.home()}
            or path.is_symlink()
            or not path.is_dir()
            or path.resolve() != path
        ):
            raise ExpertCandidateWorkspaceError(
                "expert workspace state root must be an authorized real directory"
            )
        ExpertCandidateWorkspaceManager._validate_private_directory(
            path,
            "expert workspace state root",
        )

    @staticmethod
    def _validate_private_directory(path: Path, name: str) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ExpertCandidateWorkspaceError(f"{name} must be a real directory")
        ExpertCandidateWorkspaceManager._validate_private_metadata(
            path.stat(follow_symlinks=False),
            name,
        )

    @staticmethod
    def _validate_private_metadata(metadata: os.stat_result, name: str) -> None:
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & (
            0o077 | stat.S_ISUID | stat.S_ISGID
        ):
            raise ExpertCandidateWorkspaceError(f"{name} must be a private directory")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        os.fsync(descriptor)
        os.close(descriptor)


class _WorkspaceInitializationLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> None:
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        ):
            os.close(descriptor)
            raise ExpertCandidateWorkspaceError(
                "expert workspace initialization lock is invalid"
            )
        self.handle = os.fdopen(descriptor, "r+b")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        return False


class _WorkspaceConstructionCleanup:
    """Remove a destination created before workspace preparation can finish."""

    def __init__(self, parent_descriptor: int, workspace_name: str) -> None:
        self.parent_descriptor = parent_descriptor
        self.workspace_name = workspace_name
        self.expected_identity: tuple[int, int] | None = None

    def remove(self) -> None:
        if self.workspace_name not in os.listdir(self.parent_descriptor):
            return
        observed_identity = _directory_identity_at(
            self.parent_descriptor,
            self.workspace_name,
        )
        if (
            self.expected_identity is not None
            and observed_identity != self.expected_identity
        ):
            raise ExpertCandidateWorkspaceError(
                "expert candidate workspace identity changed during construction"
            )
        with ExitStack() as descriptors:
            workspace_descriptor = _open_cleanup_directory_at(
                self.parent_descriptor,
                self.workspace_name,
                observed_identity,
                descriptors,
            )
            _remove_untrusted_workspace(
                self.parent_descriptor,
                workspace_descriptor,
                self.workspace_name,
                observed_identity,
            )


def _require_directory_path_identity(
    path: Path,
    expected_identity: tuple[int, int],
    purpose: str,
) -> None:
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise ExpertCandidateWorkspaceError(f"{purpose} identity changed")


def _require_opened_directory_identity(
    descriptor: int,
    expected_identity: tuple[int, int],
    purpose: str,
) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise ExpertCandidateWorkspaceError(f"{purpose} identity changed")


def _require_directory_entry_identity(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    purpose: str,
) -> None:
    metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & (0o077 | stat.S_ISUID | stat.S_ISGID)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise ExpertCandidateWorkspaceError(f"{purpose} identity changed")


def _directory_identity_at(parent_descriptor: int, name: str) -> tuple[int, int]:
    metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ExpertCandidateWorkspaceError(
            "expert candidate workspace is not a real directory"
        )
    return metadata.st_dev, metadata.st_ino


def _open_real_directory_at(
    parent_descriptor: int,
    name: str,
    descriptors: ExitStack,
    purpose: str,
) -> tuple[int, tuple[int, int]]:
    metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ExpertCandidateWorkspaceError(f"{purpose} must be a real directory")
    identity = metadata.st_dev, metadata.st_ino
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != identity:
        raise ExpertCandidateWorkspaceError(f"{purpose} changed while opening")
    return descriptor, identity


def _read_regular_file_at(parent_descriptor: int, name: str) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ExpertCandidateWorkspaceError(
                "released workspace control must be a single-linked regular file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()


def _remove_empty_pinned_directory(
    parent_descriptor: int,
    directory_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    opened = os.fstat(directory_descriptor)
    if (opened.st_dev, opened.st_ino) != expected_identity:
        raise ExpertCandidateWorkspaceError(
            "released workspace control descriptor changed before removal"
        )
    if os.listdir(directory_descriptor):
        raise ExpertCandidateWorkspaceError(
            "released workspace control directory is not empty"
        )
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != expected_identity
    ):
        raise ExpertCandidateWorkspaceError(
            "released workspace control directory changed before removal"
        )
    os.rmdir(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def _remove_untrusted_workspace(
    parent_descriptor: int,
    workspace_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != expected_identity
    ):
        raise ExpertCandidateWorkspaceError(
            "expert candidate workspace identity changed before cleanup"
        )
    opened = os.fstat(workspace_descriptor)
    if (opened.st_dev, opened.st_ino) != expected_identity:
        raise ExpertCandidateWorkspaceError(
            "expert candidate workspace descriptor changed before cleanup"
        )
    os.fchmod(workspace_descriptor, 0o700)
    _remove_untrusted_directory_contents(workspace_descriptor)
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != expected_identity:
        raise ExpertCandidateWorkspaceError(
            "expert candidate workspace changed during cleanup"
        )
    os.rmdir(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def _remove_untrusted_directory_contents(descriptor: int) -> None:
    os.fchmod(descriptor, 0o700)
    with os.scandir(descriptor) as iterator:
        entries = tuple(
            (entry.name, entry.stat(follow_symlinks=False)) for entry in iterator
        )
    for name, expected in entries:
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise ExpertCandidateWorkspaceError(
                "expert candidate workspace entry changed during cleanup"
            )
        if stat.S_ISDIR(expected.st_mode):
            with ExitStack() as child_descriptors:
                child_descriptor = _open_cleanup_directory_at(
                    descriptor,
                    name,
                    (expected.st_dev, expected.st_ino),
                    child_descriptors,
                )
                _remove_untrusted_directory_contents(child_descriptor)
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                raise ExpertCandidateWorkspaceError(
                    "expert candidate workspace directory changed before removal"
                )
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)


def _open_cleanup_directory_at(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    descriptors: ExitStack,
) -> int:
    path_descriptor = os.open(
        name,
        os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, path_descriptor)
    pinned = os.fstat(path_descriptor)
    if (pinned.st_dev, pinned.st_ino) != expected_identity:
        raise ExpertCandidateWorkspaceError(
            "expert candidate workspace directory changed while pinning cleanup"
        )
    pinned_path = Path("/proc/self/fd") / str(path_descriptor)
    os.chmod(pinned_path, 0o700)
    child_descriptor = os.open(
        pinned_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, child_descriptor)
    opened = os.fstat(child_descriptor)
    if (opened.st_dev, opened.st_ino) != expected_identity:
        raise ExpertCandidateWorkspaceError(
            "expert candidate workspace directory changed while opening cleanup"
        )
    os.fchmod(child_descriptor, 0o700)
    return child_descriptor
