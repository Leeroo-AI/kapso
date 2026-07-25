"""Atomic promotion of one isolated workspace generation."""

from __future__ import annotations

import ctypes
import fcntl
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_keeper_helper import (
    read_run_action_descriptor_mount_id,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.workspace import (
    ActiveLaunchWorkspace,
    LaunchWorkspaceError,
)
from kapso.cross_run.launch.workspace_frontier import (
    copy_run_workspace_frontier,
    inspect_run_workspace_frontier,
    plan_run_workspace_frontier_copy,
    RunWorkspaceFrontierIdentity,
)
from kapso.cross_run.settings import LaunchSettings

_STAGED_WORKSPACE_NAME = "workspace"
_STAGED_TEMPORARY_PATTERN = re.compile(r"^[.]workspace-[0-9a-f]{32}[.]tmp$")
_CLEANUP_REGULAR_MODES = frozenset({0o400, 0o444, 0o600, 0o644, 0o700, 0o755})
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY = object()


class RunActionWorkspacePromotionError(RuntimeError):
    """An isolated workspace cannot be staged or atomically promoted."""


@dataclass(frozen=True)
class RunActionWorkspacePromotion(StrictContract):
    """Event-7 evidence for one fully staged direct workspace successor."""

    workspace_promotion_id: str
    result_receipt_id: str
    prepared_workspace_proof_id: str
    candidate_workspace: RunActionWorkspaceBinding

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-workspace-promotion"
    IDENTITY_FIELD: ClassVar[str] = "workspace_promotion_id"

    def _validate(self) -> None:
        _require_namespaced_id(
            self.result_receipt_id,
            "run-action-result-receipt",
            "workspace promotion result receipt",
        )
        _require_namespaced_id(
            self.prepared_workspace_proof_id,
            "run-action-prepared-workspace-proof",
            "workspace promotion prepared workspace",
        )
        if type(self.candidate_workspace) is not RunActionWorkspaceBinding:
            raise RunActionWorkspacePromotionError(
                "workspace promotion candidate is invalid"
            )


@dataclass(frozen=True)
class _CleanupEntry:
    name: str
    identity: tuple[int, int]
    file_type: str
    mode: int
    size_bytes: int
    children: tuple["_CleanupEntry", ...]


@dataclass(frozen=True)
class _CleanupPlan:
    root: _CleanupEntry
    mount_id: int


@dataclass
class _CleanupScanState:
    entry_limit: int
    size_limit_bytes: int
    entry_count: int = 0
    size_bytes: int = 0
    inode_identities: set[tuple[int, int]] = field(default_factory=set)

    def reserve(self, metadata: os.stat_result) -> None:
        identity = metadata.st_dev, metadata.st_ino
        self.entry_count += 1
        if identity in self.inode_identities or self.entry_count > self.entry_limit:
            raise RunActionWorkspacePromotionError(
                "workspace staging cleanup tree is repeated or unbounded"
            )
        self.inode_identities.add(identity)
        if stat.S_ISREG(metadata.st_mode):
            self.size_bytes += metadata.st_size
            if self.size_bytes > self.size_limit_bytes:
                raise RunActionWorkspacePromotionError(
                    "workspace staging cleanup tree is unbounded"
                )


class RunActionWorkspacePromoter:
    """Stage and atomically exchange one complete workspace generation."""

    def __init__(
        self,
        *,
        active_workspace: ActiveLaunchWorkspace,
        settings: LaunchSettings,
    ) -> None:
        if (
            type(active_workspace) is not ActiveLaunchWorkspace
            or type(settings) is not LaunchSettings
        ):
            raise RunActionWorkspacePromotionError(
                "workspace promoter requires active launch settings"
            )
        active_workspace.require_launch_settings(settings)
        self._active_workspace = active_workspace
        self._settings = settings
        self._owner_process_id = os.getpid()

    def stage(
        self,
        *,
        result_receipt_id: str,
        prepared_workspace_proof_id: str,
        predecessor: RunActionWorkspaceBinding,
        candidate_descriptor: int,
        workspace_lock_descriptor: int | None = None,
        _authority: object | None = None,
    ) -> RunActionWorkspacePromotion:
        """Copy one clean direct successor without mutating the public workspace."""

        self._require_owner_process()
        _require_namespaced_id(
            result_receipt_id,
            "run-action-result-receipt",
            "workspace promotion result receipt",
        )
        _require_namespaced_id(
            prepared_workspace_proof_id,
            "run-action-prepared-workspace-proof",
            "workspace promotion prepared workspace",
        )
        if (
            type(predecessor) is not RunActionWorkspaceBinding
            or type(candidate_descriptor) is not int
            or candidate_descriptor < 0
        ):
            raise RunActionWorkspacePromotionError(
                "workspace staging inputs are invalid"
            )
        predecessor_identity = predecessor.to_identity()
        with ExitStack() as descriptors:
            if workspace_lock_descriptor is None:
                if _authority is not None:
                    raise RunActionWorkspacePromotionError(
                        "workspace staging authority is invalid"
                    )
                _lock_workspace(self._active_workspace, descriptors)
            else:
                if _authority is not _RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY:
                    raise RunActionWorkspacePromotionError(
                        "workspace staging lacks sealed recovery authority"
                    )
                _require_external_workspace_lock(
                    self._active_workspace,
                    workspace_lock_descriptor,
                    descriptors,
                )
            public_descriptor, public_identity = (
                self._active_workspace._open_execution_workspace(descriptors)
            )
            observed_predecessor = inspect_run_workspace_frontier(
                public_descriptor,
                settings=self._settings,
                expected_commit_sha=predecessor.commit_sha,
            )
            if (
                public_identity != predecessor_identity.workspace_identity
                or observed_predecessor != predecessor_identity
            ):
                raise RunActionWorkspacePromotionError(
                    "public workspace differs from the promotion predecessor"
                )
            candidate = inspect_run_workspace_frontier(
                candidate_descriptor,
                settings=self._settings,
                expected_commit_sha=None,
            )
            _require_direct_successor(predecessor_identity, candidate)
            plan = plan_run_workspace_frontier_copy(
                candidate_descriptor,
                settings=self._settings,
                expected=candidate,
            )
            staging_descriptor, staging_identity = (
                self._active_workspace._open_run_action_workspace_staging(descriptors)
            )
            if (
                staging_identity[0] != public_identity[0]
                or os.fstat(candidate_descriptor).st_dev
                == os.fstat(staging_descriptor).st_dev
                and os.fstat(candidate_descriptor).st_ino
                == os.fstat(staging_descriptor).st_ino
            ):
                raise RunActionWorkspacePromotionError(
                    "workspace staging endpoints are unsafe"
                )
            durable_staged = _reconcile_staging_before_decision(
                staging_descriptor,
                staging_identity[0],
                candidate,
                self._settings,
            )
            if durable_staged is not None:
                confirmed_public = inspect_run_workspace_frontier(
                    public_descriptor,
                    settings=self._settings,
                    expected_commit_sha=predecessor.commit_sha,
                )
                if confirmed_public != predecessor_identity:
                    raise RunActionWorkspacePromotionError(
                        "public workspace changed while reopening staged successor"
                    )
                self._active_workspace._require_execution_workspace(
                    public_descriptor,
                    public_identity,
                )
                return RunActionWorkspacePromotion.mint(
                    result_receipt_id=result_receipt_id,
                    prepared_workspace_proof_id=prepared_workspace_proof_id,
                    candidate_workspace=RunActionWorkspaceBinding.from_identity(
                        durable_staged
                    ),
                )
            _require_staging_capacity(
                staging_descriptor,
                plan,
                self._settings,
            )
            temporary_name = f".workspace-{secrets.token_hex(16)}.tmp"
            os.mkdir(
                temporary_name,
                mode=0o700,
                dir_fd=staging_descriptor,
            )
            temporary_descriptor = os.open(
                temporary_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=staging_descriptor,
            )
            descriptors.callback(os.close, temporary_descriptor)
            if os.fstat(temporary_descriptor).st_dev != staging_identity[0]:
                raise RunActionWorkspacePromotionError(
                    "workspace staging temporary crossed a filesystem"
                )
            staged = copy_run_workspace_frontier(
                candidate_descriptor,
                temporary_descriptor,
                settings=self._settings,
                plan=plan,
            )
            os.fsync(temporary_descriptor)
            os.fsync(staging_descriptor)
            _rename_at(
                staging_descriptor,
                temporary_name,
                staging_descriptor,
                _STAGED_WORKSPACE_NAME,
                _RENAME_NOREPLACE,
            )
            os.fsync(staging_descriptor)
            staged_descriptor = os.open(
                _STAGED_WORKSPACE_NAME,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=staging_descriptor,
            )
            descriptors.callback(os.close, staged_descriptor)
            durable_staged = inspect_run_workspace_frontier(
                staged_descriptor,
                settings=self._settings,
                expected_commit_sha=candidate.commit_sha,
            )
            expected_staged = RunWorkspaceFrontierIdentity(
                workspace_identity=durable_staged.workspace_identity,
                branch=candidate.branch,
                commit_sha=candidate.commit_sha,
                parent_commit_shas=candidate.parent_commit_shas,
                git_tree_sha=candidate.git_tree_sha,
                source_tree_digest=candidate.source_tree_digest,
                git_closure_digest=candidate.git_closure_digest,
                source_entry_count=candidate.source_entry_count,
                source_size_bytes=candidate.source_size_bytes,
            )
            if durable_staged != expected_staged or staged != expected_staged:
                raise RunActionWorkspacePromotionError(
                    "durable staged workspace differs from its isolated candidate"
                )
            confirmed_public = inspect_run_workspace_frontier(
                public_descriptor,
                settings=self._settings,
                expected_commit_sha=predecessor.commit_sha,
            )
            if confirmed_public != predecessor_identity:
                raise RunActionWorkspacePromotionError(
                    "public workspace changed while staging its successor"
                )
            self._active_workspace._require_execution_workspace(
                public_descriptor,
                public_identity,
            )
        return RunActionWorkspacePromotion.mint(
            result_receipt_id=result_receipt_id,
            prepared_workspace_proof_id=prepared_workspace_proof_id,
            candidate_workspace=RunActionWorkspaceBinding.from_identity(durable_staged),
        )

    def _promote_decided(
        self,
        *,
        predecessor: RunActionWorkspaceBinding,
        promotion: RunActionWorkspacePromotion,
        result_receipt_id: str,
        prepared_workspace_proof_id: str,
        workspace_lock_descriptor: int,
        _authority: object,
    ) -> RunWorkspaceFrontierIdentity:
        """Apply or reprove the sole atomic exchange authorized by event 7."""

        self._require_owner_process()
        if (
            type(predecessor) is not RunActionWorkspaceBinding
            or type(promotion) is not RunActionWorkspacePromotion
            or promotion.result_receipt_id != result_receipt_id
            or promotion.prepared_workspace_proof_id != prepared_workspace_proof_id
            or _authority is not _RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY
        ):
            raise RunActionWorkspacePromotionError(
                "workspace promotion recovery inputs are invalid"
            )
        before = predecessor.to_identity()
        candidate = promotion.candidate_workspace.to_identity()
        _require_direct_successor(before, candidate)
        with ExitStack() as descriptors:
            _require_external_workspace_lock(
                self._active_workspace,
                workspace_lock_descriptor,
                descriptors,
            )
            public_parent_descriptor, public_name = _open_public_workspace_parent(
                self._active_workspace,
                descriptors,
            )
            staging_descriptor, staging_identity = (
                self._active_workspace._open_run_action_workspace_staging(descriptors)
            )
            if os.fstat(public_parent_descriptor).st_dev != staging_identity[0]:
                raise RunActionWorkspacePromotionError(
                    "workspace exchange parents are on different filesystems"
                )
            public_descriptor = os.open(
                public_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=public_parent_descriptor,
            )
            descriptors.callback(os.close, public_descriptor)
            staged_descriptor = os.open(
                _STAGED_WORKSPACE_NAME,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=staging_descriptor,
            )
            descriptors.callback(os.close, staged_descriptor)
            public_identity = _directory_identity(public_descriptor)
            staged_identity = _directory_identity(staged_descriptor)
            if (
                public_identity == before.workspace_identity
                and staged_identity == candidate.workspace_identity
            ):
                _require_exact_frontier(
                    public_descriptor,
                    before,
                    self._settings,
                    "public promotion predecessor",
                )
                _require_exact_frontier(
                    staged_descriptor,
                    candidate,
                    self._settings,
                    "staged promotion candidate",
                )
                exchange_required = True
            elif (
                public_identity == candidate.workspace_identity
                and staged_identity == before.workspace_identity
            ):
                _require_exact_frontier(
                    public_descriptor,
                    candidate,
                    self._settings,
                    "public promoted candidate",
                )
                _require_exact_frontier(
                    staged_descriptor,
                    before,
                    self._settings,
                    "staged promotion predecessor",
                )
                exchange_required = False
            else:
                raise RunActionWorkspacePromotionError(
                    "workspace promotion state is neither before nor after exchange"
                )
            rebound_public_parent, rebound_public_name = _open_public_workspace_parent(
                self._active_workspace,
                descriptors,
            )
            rebound_staging, rebound_staging_identity = (
                self._active_workspace._open_run_action_workspace_staging(descriptors)
            )
            if (
                rebound_public_name != public_name
                or _directory_identity(rebound_public_parent)
                != _directory_identity(public_parent_descriptor)
                or rebound_staging_identity != staging_identity
            ):
                raise RunActionWorkspacePromotionError(
                    "workspace exchange parents detached before mutation"
                )
            _require_name_identity(
                rebound_public_parent,
                rebound_public_name,
                public_identity,
                "reachable public promotion generation",
            )
            _require_name_identity(
                rebound_staging,
                _STAGED_WORKSPACE_NAME,
                staged_identity,
                "reachable staged promotion generation",
            )
            if exchange_required:
                _rename_at(
                    rebound_public_parent,
                    rebound_public_name,
                    rebound_staging,
                    _STAGED_WORKSPACE_NAME,
                    _RENAME_EXCHANGE,
                )
            os.fsync(rebound_public_parent)
            os.fsync(rebound_staging)
        with ExitStack() as verification_descriptors:
            public_descriptor, public_identity = (
                self._active_workspace._open_execution_workspace(
                    verification_descriptors
                )
            )
            staging_descriptor, _identity = (
                self._active_workspace._open_run_action_workspace_staging(
                    verification_descriptors
                )
            )
            staged_descriptor = os.open(
                _STAGED_WORKSPACE_NAME,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=staging_descriptor,
            )
            verification_descriptors.callback(os.close, staged_descriptor)
            if (
                public_identity != candidate.workspace_identity
                or _directory_identity(staged_descriptor) != before.workspace_identity
            ):
                raise RunActionWorkspacePromotionError(
                    "workspace exchange did not publish its exact candidate"
                )
            _require_exact_frontier(
                public_descriptor,
                candidate,
                self._settings,
                "public promoted workspace",
            )
            _require_exact_frontier(
                staged_descriptor,
                before,
                self._settings,
                "staged retired workspace",
            )
        return candidate

    def _cleanup_accepted_if_owned(
        self,
        *,
        predecessor: RunActionWorkspaceBinding,
        promotion: RunActionWorkspacePromotion,
        result_receipt_id: str,
        prepared_workspace_proof_id: str,
        workspace_lock_descriptor: int,
        _authority: object,
    ) -> bool:
        """Clean only residue still owned by this accepted promotion."""

        self._require_owner_process()
        if (
            type(predecessor) is not RunActionWorkspaceBinding
            or type(promotion) is not RunActionWorkspacePromotion
            or promotion.result_receipt_id != result_receipt_id
            or promotion.prepared_workspace_proof_id != prepared_workspace_proof_id
            or _authority is not _RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY
        ):
            raise RunActionWorkspacePromotionError(
                "accepted workspace cleanup inputs are invalid"
            )
        before = predecessor.to_identity()
        candidate = promotion.candidate_workspace.to_identity()
        _require_direct_successor(before, candidate)
        with ExitStack() as descriptors:
            _require_external_workspace_lock(
                self._active_workspace,
                workspace_lock_descriptor,
                descriptors,
            )
            public_descriptor, public_identity = (
                self._active_workspace._open_execution_workspace(descriptors)
            )
            if public_identity != candidate.workspace_identity:
                return False
            _require_exact_frontier(
                public_descriptor,
                candidate,
                self._settings,
                "accepted public workspace",
            )
            staging_descriptor, _staging_identity = (
                self._active_workspace._open_run_action_workspace_staging(descriptors)
            )
            staged_names = tuple(sorted(os.listdir(staging_descriptor)))
            if not staged_names:
                return True
            if staged_names != (_STAGED_WORKSPACE_NAME,):
                return False
            staged_metadata = os.stat(
                _STAGED_WORKSPACE_NAME,
                dir_fd=staging_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(staged_metadata.st_mode)
                or (staged_metadata.st_dev, staged_metadata.st_ino)
                != before.workspace_identity
            ):
                return False
        self._cleanup_accepted(
            predecessor=predecessor,
            promotion=promotion,
            result_receipt_id=result_receipt_id,
            prepared_workspace_proof_id=prepared_workspace_proof_id,
            workspace_lock_descriptor=workspace_lock_descriptor,
            _authority=_authority,
        )
        return True

    def _cleanup_accepted(
        self,
        *,
        predecessor: RunActionWorkspaceBinding,
        promotion: RunActionWorkspacePromotion,
        result_receipt_id: str,
        prepared_workspace_proof_id: str,
        workspace_lock_descriptor: int,
        _authority: object,
    ) -> RunWorkspaceFrontierIdentity:
        """Remove the retired generation after event 8 is durably accepted."""

        self._require_owner_process()
        if (
            type(predecessor) is not RunActionWorkspaceBinding
            or type(promotion) is not RunActionWorkspacePromotion
            or promotion.result_receipt_id != result_receipt_id
            or promotion.prepared_workspace_proof_id != prepared_workspace_proof_id
            or _authority is not _RUN_ACTION_WORKSPACE_PROMOTION_AUTHORITY
        ):
            raise RunActionWorkspacePromotionError(
                "accepted workspace cleanup inputs are invalid"
            )
        before = predecessor.to_identity()
        candidate = promotion.candidate_workspace.to_identity()
        _require_direct_successor(before, candidate)
        with ExitStack() as descriptors:
            _require_external_workspace_lock(
                self._active_workspace,
                workspace_lock_descriptor,
                descriptors,
            )
            public_descriptor, public_identity = (
                self._active_workspace._open_execution_workspace(descriptors)
            )
            if public_identity != candidate.workspace_identity:
                raise RunActionWorkspacePromotionError(
                    "accepted workspace cleanup lacks its promoted public generation"
                )
            _require_exact_frontier(
                public_descriptor,
                candidate,
                self._settings,
                "accepted public workspace",
            )
            staging_descriptor, staging_identity = (
                self._active_workspace._open_run_action_workspace_staging(descriptors)
            )
            staged_names = tuple(sorted(os.listdir(staging_descriptor)))
            if staged_names:
                if staged_names != (_STAGED_WORKSPACE_NAME,):
                    raise RunActionWorkspacePromotionError(
                        "accepted workspace cleanup found an unauthorized staging state"
                    )
                cleanup_plan = _plan_staging_cleanup(
                    staging_descriptor,
                    _STAGED_WORKSPACE_NAME,
                    staging_identity[0],
                    self._settings,
                    expected_root_identity=before.workspace_identity,
                )
                _remove_staging_cleanup(
                    staging_descriptor,
                    cleanup_plan,
                )
                os.fsync(staging_descriptor)
            confirmed_public = inspect_run_workspace_frontier(
                public_descriptor,
                settings=self._settings,
                expected_commit_sha=candidate.commit_sha,
            )
            if confirmed_public != candidate or tuple(os.listdir(staging_descriptor)):
                raise RunActionWorkspacePromotionError(
                    "accepted workspace cleanup did not reach its terminal state"
                )
            self._active_workspace._require_execution_workspace(
                public_descriptor,
                public_identity,
            )
        return candidate

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise RunActionWorkspacePromotionError(
                "workspace promoter belongs to another process"
            )
        self._active_workspace.require_launch_settings(self._settings)


def _reconcile_staging_before_decision(
    staging_descriptor: int,
    staging_device: int,
    candidate: RunWorkspaceFrontierIdentity,
    settings: LaunchSettings,
) -> RunWorkspaceFrontierIdentity | None:
    staged_names = tuple(sorted(os.listdir(staging_descriptor)))
    if not staged_names:
        return None
    if len(staged_names) != 1:
        raise RunActionWorkspacePromotionError(
            "workspace staging contains more than one interrupted generation"
        )
    staged_name = staged_names[0]
    if _STAGED_TEMPORARY_PATTERN.fullmatch(staged_name) is not None:
        cleanup_plan = _plan_staging_cleanup(
            staging_descriptor,
            staged_name,
            staging_device,
            settings,
            expected_root_identity=None,
        )
        _remove_staging_cleanup(staging_descriptor, cleanup_plan)
        os.fsync(staging_descriptor)
        return None
    if staged_name != _STAGED_WORKSPACE_NAME:
        raise RunActionWorkspacePromotionError(
            "workspace staging contains an unauthorized generation"
        )
    with ExitStack() as descriptors:
        staged_descriptor = os.open(
            staged_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=staging_descriptor,
        )
        descriptors.callback(os.close, staged_descriptor)
        durable_staged = inspect_run_workspace_frontier(
            staged_descriptor,
            settings=settings,
            expected_commit_sha=candidate.commit_sha,
        )
        expected_staged = replace(
            candidate,
            workspace_identity=durable_staged.workspace_identity,
        )
        if (
            durable_staged != expected_staged
            or durable_staged.workspace_identity[0] != staging_device
        ):
            raise RunActionWorkspacePromotionError(
                "interrupted staged workspace differs from its isolated candidate"
            )
        _require_name_identity(
            staging_descriptor,
            staged_name,
            durable_staged.workspace_identity,
            "interrupted staged workspace",
        )
    return durable_staged


def _plan_staging_cleanup(
    staging_descriptor: int,
    name: str,
    staging_device: int,
    settings: LaunchSettings,
    *,
    expected_root_identity: tuple[int, int] | None,
) -> _CleanupPlan:
    staging_mount_id = read_run_action_descriptor_mount_id(staging_descriptor)
    metadata = os.stat(
        name,
        dir_fd=staging_descriptor,
        follow_symlinks=False,
    )
    _require_cleanup_directory_metadata(
        metadata,
        staging_device,
        expected_root_identity,
    )
    state = _CleanupScanState(
        entry_limit=(
            settings.run_workspace_entry_limit
            + settings.run_workspace_git_entry_limit
            + settings.run_action_staging_entry_limit
        ),
        size_limit_bytes=(
            settings.run_workspace_size_bytes
            + settings.run_workspace_git_metadata_size_bytes
        ),
    )
    state.reserve(metadata)
    with ExitStack() as descriptors:
        descriptor = _open_cleanup_directory(
            staging_descriptor,
            name,
            metadata.st_dev,
            metadata.st_ino,
            descriptors,
            staging_mount_id,
        )
        children = _plan_cleanup_directory(
            descriptor,
            staging_device,
            staging_mount_id,
            state,
        )
    return _CleanupPlan(
        root=_CleanupEntry(
            name=name,
            identity=(metadata.st_dev, metadata.st_ino),
            file_type="directory",
            mode=stat.S_IMODE(metadata.st_mode),
            size_bytes=0,
            children=children,
        ),
        mount_id=staging_mount_id,
    )


def _plan_cleanup_directory(
    descriptor: int,
    staging_device: int,
    staging_mount_id: int,
    state: _CleanupScanState,
) -> tuple[_CleanupEntry, ...]:
    with os.scandir(descriptor) as iterator:
        observed = tuple(
            sorted(
                (
                    entry.name,
                    entry.stat(follow_symlinks=False),
                )
                for entry in iterator
            )
        )
    planned = []
    for name, metadata in observed:
        if not name or "/" in name or name in {".", ".."}:
            raise RunActionWorkspacePromotionError(
                "workspace staging cleanup entry name is unsafe"
            )
        current = os.stat(
            name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        if _cleanup_metadata_observation(current) != _cleanup_metadata_observation(
            metadata
        ):
            raise RunActionWorkspacePromotionError(
                "workspace staging cleanup entry changed while scanning"
            )
        state.reserve(metadata)
        identity = metadata.st_dev, metadata.st_ino
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            _require_cleanup_directory_metadata(
                metadata,
                staging_device,
                expected_identity=None,
            )
            with ExitStack() as descriptors:
                child_descriptor = _open_cleanup_directory(
                    descriptor,
                    name,
                    metadata.st_dev,
                    metadata.st_ino,
                    descriptors,
                    staging_mount_id,
                )
                children = _plan_cleanup_directory(
                    child_descriptor,
                    staging_device,
                    staging_mount_id,
                    state,
                )
            planned.append(
                _CleanupEntry(
                    name=name,
                    identity=identity,
                    file_type="directory",
                    mode=mode,
                    size_bytes=0,
                    children=children,
                )
            )
        elif (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_dev == staging_device
            and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1
            and mode in _CLEANUP_REGULAR_MODES
            and not metadata.st_mode & (stat.S_ISUID | stat.S_ISGID)
        ):
            _require_cleanup_regular_file(
                descriptor,
                name,
                metadata,
            )
            planned.append(
                _CleanupEntry(
                    name=name,
                    identity=identity,
                    file_type="regular",
                    mode=mode,
                    size_bytes=metadata.st_size,
                    children=(),
                )
            )
        else:
            raise RunActionWorkspacePromotionError(
                "workspace staging cleanup tree contains an unsafe entry"
            )
    return tuple(planned)


def _remove_staging_cleanup(
    staging_descriptor: int,
    plan: _CleanupPlan,
) -> None:
    if read_run_action_descriptor_mount_id(staging_descriptor) != plan.mount_id:
        raise RunActionWorkspacePromotionError(
            "workspace staging cleanup mount changed before removal"
        )
    root = plan.root
    with ExitStack() as descriptors:
        root_descriptor = _open_cleanup_directory(
            staging_descriptor,
            root.name,
            root.identity[0],
            root.identity[1],
            descriptors,
            plan.mount_id,
        )
        _remove_cleanup_directory_contents(
            root_descriptor,
            root.children,
            plan.mount_id,
        )
        if tuple(os.listdir(root_descriptor)):
            raise RunActionWorkspacePromotionError(
                "workspace staging cleanup root is not empty"
            )
        _require_cleanup_entry_current(
            staging_descriptor,
            root,
        )
    os.rmdir(root.name, dir_fd=staging_descriptor)
    os.fsync(staging_descriptor)


def _remove_cleanup_directory_contents(
    descriptor: int,
    entries: tuple[_CleanupEntry, ...],
    staging_mount_id: int,
) -> None:
    if read_run_action_descriptor_mount_id(descriptor) != staging_mount_id:
        raise RunActionWorkspacePromotionError(
            "workspace staging cleanup crossed a mount boundary"
        )
    for entry in entries:
        _require_cleanup_entry_current(descriptor, entry)
        if entry.file_type == "directory":
            with ExitStack() as descriptors:
                child_descriptor = _open_cleanup_directory(
                    descriptor,
                    entry.name,
                    entry.identity[0],
                    entry.identity[1],
                    descriptors,
                    staging_mount_id,
                )
                _remove_cleanup_directory_contents(
                    child_descriptor,
                    entry.children,
                    staging_mount_id,
                )
                if tuple(os.listdir(child_descriptor)):
                    raise RunActionWorkspacePromotionError(
                        "workspace staging cleanup directory is not empty"
                    )
                _require_cleanup_entry_current(descriptor, entry)
            os.rmdir(entry.name, dir_fd=descriptor)
        elif entry.file_type == "regular":
            _require_cleanup_regular_entry(
                descriptor,
                entry,
            )
            _require_cleanup_entry_current(descriptor, entry)
            os.unlink(entry.name, dir_fd=descriptor)
        else:
            raise RunActionWorkspacePromotionError(
                "workspace staging cleanup plan is invalid"
            )
    os.fsync(descriptor)


def _open_cleanup_directory(
    parent_descriptor: int,
    name: str,
    expected_device: int,
    expected_inode: int,
    descriptors: ExitStack,
    expected_mount_id: int,
) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    _require_cleanup_directory_metadata(
        metadata,
        expected_device,
        expected_identity=(expected_device, expected_inode),
    )
    if read_run_action_descriptor_mount_id(descriptor) != expected_mount_id:
        raise RunActionWorkspacePromotionError(
            "workspace staging cleanup crossed a mount boundary"
        )
    return descriptor


def _require_cleanup_directory_metadata(
    metadata: os.stat_result,
    staging_device: int,
    expected_identity: tuple[int, int] | None,
) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_dev != staging_device
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_mode & (stat.S_ISUID | stat.S_ISGID)
        or (
            expected_identity is not None
            and (metadata.st_dev, metadata.st_ino) != expected_identity
        )
    ):
        raise RunActionWorkspacePromotionError(
            "workspace staging cleanup directory is unsafe"
        )


def _require_cleanup_entry_current(
    parent_descriptor: int,
    entry: _CleanupEntry,
) -> None:
    metadata = os.stat(
        entry.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        (metadata.st_dev, metadata.st_ino) != entry.identity
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != entry.mode
        or metadata.st_mode & (stat.S_ISUID | stat.S_ISGID)
        or (entry.file_type == "directory" and not stat.S_ISDIR(metadata.st_mode))
        or (
            entry.file_type == "regular"
            and (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != entry.size_bytes
            )
        )
    ):
        raise RunActionWorkspacePromotionError(
            "workspace staging cleanup entry changed before removal"
        )


def _require_cleanup_regular_file(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        observed = os.fstat(descriptor)
        if _cleanup_metadata_observation(observed) != _cleanup_metadata_observation(
            expected
        ):
            raise RunActionWorkspacePromotionError(
                "workspace staging cleanup file changed while opening"
            )


def _require_cleanup_regular_entry(
    parent_descriptor: int,
    entry: _CleanupEntry,
) -> None:
    metadata = os.stat(
        entry.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    _require_cleanup_regular_file(
        parent_descriptor,
        entry.name,
        metadata,
    )
    if (
        (metadata.st_dev, metadata.st_ino) != entry.identity
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != entry.mode
        or metadata.st_size != entry.size_bytes
    ):
        raise RunActionWorkspacePromotionError(
            "workspace staging cleanup file differs from its plan"
        )


def _cleanup_metadata_observation(metadata: os.stat_result) -> tuple[int, ...]:
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


def _require_direct_successor(
    predecessor: RunWorkspaceFrontierIdentity,
    candidate: RunWorkspaceFrontierIdentity,
) -> None:
    if (
        type(predecessor) is not RunWorkspaceFrontierIdentity
        or type(candidate) is not RunWorkspaceFrontierIdentity
        or candidate.workspace_identity == predecessor.workspace_identity
        or candidate.branch != predecessor.branch
        or candidate.commit_sha == predecessor.commit_sha
        or candidate.parent_commit_shas != (predecessor.commit_sha,)
        or candidate.git_tree_sha == predecessor.git_tree_sha
        or candidate.source_tree_digest == predecessor.source_tree_digest
    ):
        raise RunActionWorkspacePromotionError(
            "workspace promotion candidate is not one direct source successor"
        )


def _require_exact_frontier(
    descriptor: int,
    expected: RunWorkspaceFrontierIdentity,
    settings: LaunchSettings,
    name: str,
) -> None:
    observed = inspect_run_workspace_frontier(
        descriptor,
        settings=settings,
        expected_commit_sha=expected.commit_sha,
    )
    if observed != expected:
        raise RunActionWorkspacePromotionError(f"{name} differs from its evidence")


def _require_staging_capacity(
    staging_descriptor: int,
    plan,
    settings: LaunchSettings,
) -> None:
    filesystem = os.fstatvfs(staging_descriptor)
    block_size = filesystem.f_frsize
    candidate_allocation_size = plan.allocated_size_bytes(block_size)
    if (
        type(block_size) is not int
        or block_size <= 0
        or block_size & (block_size - 1) != 0
        or filesystem.f_bavail * block_size < candidate_allocation_size
        or filesystem.f_favail < plan.physical_entry_count
    ):
        raise RunActionWorkspacePromotionError(
            "workspace staging lacks byte or inode capacity"
        )


def _require_name_identity(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
    description: str,
) -> None:
    metadata = os.stat(
        name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise RunActionWorkspacePromotionError(
            f"{description} pathname changed before exchange"
        )


def _open_public_workspace_parent(
    active_workspace: ActiveLaunchWorkspace,
    descriptors: ExitStack,
) -> tuple[int, str]:
    layout = active_workspace.bootstrap_pin.installation_receipt.layout
    workspace_path = PurePosixPath(layout.workspace_relative_path)
    root_descriptor = active_workspace._open_run_root(descriptors)
    current_descriptor = root_descriptor
    current_path = PurePosixPath(".")
    for component in workspace_path.parent.parts:
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
        expected = active_workspace._closure.pinned_directory_identities.get(
            current_path.as_posix()
        )
        if expected is None or _directory_identity(child_descriptor) != expected:
            raise RunActionWorkspacePromotionError(
                "workspace promotion parent differs from its launch"
            )
        current_descriptor = child_descriptor
    return current_descriptor, workspace_path.name


def _directory_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RunActionWorkspacePromotionError(
            "workspace promotion directory is unsafe"
        )
    return metadata.st_dev, metadata.st_ino


def _lock_workspace(
    active_workspace: ActiveLaunchWorkspace,
    descriptors: ExitStack,
) -> int:
    store_descriptor, _identity = active_workspace._open_run_action_store(descriptors)
    descriptor = os.open(
        "workspace.lock",
        os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=store_descriptor,
    )
    descriptors.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    receipt = active_workspace.bootstrap_pin.installation_receipt
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_size != 0
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (metadata.st_dev, metadata.st_ino)
        != (
            receipt.run_action_workspace_lock_device,
            receipt.run_action_workspace_lock_inode,
        )
    ):
        raise RunActionWorkspacePromotionError(
            "workspace promotion lacks its receipt-pinned lock"
        )
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    rebound_store_descriptor, rebound_store_identity = (
        active_workspace._open_run_action_store(descriptors)
    )
    current = os.stat(
        "workspace.lock",
        dir_fd=rebound_store_descriptor,
        follow_symlinks=False,
    )
    if (
        rebound_store_identity
        != (
            receipt.run_action_store_device,
            receipt.run_action_store_inode,
        )
        or _cleanup_metadata_observation(current)
        != _cleanup_metadata_observation(metadata)
        or (current.st_dev, current.st_ino)
        != (
            receipt.run_action_workspace_lock_device,
            receipt.run_action_workspace_lock_inode,
        )
    ):
        raise RunActionWorkspacePromotionError(
            "workspace promotion lock detached while acquiring authority"
        )
    return descriptor


def _require_external_workspace_lock(
    active_workspace: ActiveLaunchWorkspace,
    descriptor: int,
    descriptors: ExitStack,
) -> None:
    if type(descriptor) is not int or descriptor < 0:
        raise RunActionWorkspacePromotionError(
            "workspace promotion external lock is invalid"
        )
    metadata = os.fstat(descriptor)
    receipt = active_workspace.bootstrap_pin.installation_receipt
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_size != 0
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (metadata.st_dev, metadata.st_ino)
        != (
            receipt.run_action_workspace_lock_device,
            receipt.run_action_workspace_lock_inode,
        )
    ):
        raise RunActionWorkspacePromotionError(
            "workspace promotion external lock differs from its receipt"
        )
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store_descriptor, store_identity = active_workspace._open_run_action_store(
        descriptors
    )
    current = os.stat(
        "workspace.lock",
        dir_fd=store_descriptor,
        follow_symlinks=False,
    )
    if store_identity != (
        receipt.run_action_store_device,
        receipt.run_action_store_inode,
    ) or _cleanup_metadata_observation(current) != _cleanup_metadata_observation(
        metadata
    ):
        raise RunActionWorkspacePromotionError(
            "workspace promotion external lock is no longer reachable"
        )


def _rename_at(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
    flags: int,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise RunActionWorkspacePromotionError("workspace promotion requires renameat2")
    result = libc.renameat2(
        ctypes.c_int(source_parent_descriptor),
        ctypes.c_char_p(os.fsencode(source_name)),
        ctypes.c_int(destination_parent_descriptor),
        ctypes.c_char_p(os.fsencode(destination_name)),
        ctypes.c_uint(flags),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _require_namespaced_id(
    value: str,
    namespace: str,
    name: str,
) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionWorkspacePromotionError(f"{name} uses the wrong namespace")


__all__ = [
    "RunActionWorkspacePromotion",
    "RunActionWorkspacePromotionError",
    "RunActionWorkspacePromoter",
]
