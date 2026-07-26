"""Single-lock publication and recovery for one reconciled run-state frontier."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from threading import Lock
from weakref import WeakValueDictionary

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointHead,
    RunCheckpointStatus,
)
from kapso.cross_run.launch.checkpoint_control import (
    _CheckpointFrontierInspection,
    _RunCheckpointControl,
)
from kapso.cross_run.launch.derived_state_bundle import RunDerivedStateBundle
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
    RunActionLedgerSnapshot,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionFrontierBinding,
    RunActionReservation,
    RunActionViewBinding,
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_STORE_AUTHORITY,
    RunActionExecutionStore,
    RunActionStoreInspection,
)
from kapso.cross_run.launch.resume_contracts import RunEligibilityDisposition
from kapso.cross_run.launch.run_state_projection import (
    ReconciledRunStateProjection,
)
from kapso.cross_run.launch.workspace import ActiveLaunchWorkspace
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from kapso.cross_run.settings import LaunchSettings

_GENERATION_OBJECT_PATTERN = re.compile(r"^generation-[0-9a-f]{64}[.]bundle$")
_GENERATION_STAGING_PATTERN = re.compile(
    r"^generation-[0-9a-f]{64}-[0-9a-f]{32}[.]tmp$"
)
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_RENAME_NOREPLACE = 1
_PUBLICATION_PERMIT_AUTHORITY = object()
_RECONCILED_FRONTIER_AUTHORITY = object()


class RunStatePublisherError(RuntimeError):
    """Run state cannot be proven or published as one exact frontier."""


@dataclass(frozen=True)
class RunStateViewIdentity:
    """Exact on-disk identity of one reconciled convenience view."""

    relative_path: str
    device: int
    inode: int
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        _require_relative_control_path(self.relative_path, "run-state view")
        if (
            type(self.device) is not int
            or self.device < 0
            or type(self.inode) is not int
            or self.inode < 0
            or _DIGEST_PATTERN.fullmatch(self.digest) is None
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
        ):
            raise RunStatePublisherError("run-state view identity is invalid")


@dataclass(frozen=True)
class RunStatePublicationPermit:
    """One non-clonable authorization bound to a candidate and observed frontier."""

    requested_predecessor_checkpoint_id: str | None
    observed_checkpoint_id: str | None
    expected_journal_head_id: str
    expected_journal_size_bytes: int
    candidate_checkpoint_id: str
    generation_id: str
    bundle_digest: str
    bundle_size_bytes: int
    _publisher_identity: object = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for value, name in (
            (
                self.requested_predecessor_checkpoint_id,
                "publication permit requested predecessor",
            ),
            (self.observed_checkpoint_id, "publication permit observed checkpoint"),
        ):
            if value is not None:
                _require_namespaced_id(value, RunCheckpoint.CONTENT_NAMESPACE, name)
        _require_namespaced_id(
            self.expected_journal_head_id,
            RunCheckpointHead.CONTENT_NAMESPACE,
            "publication permit journal head",
        )
        _require_namespaced_id(
            self.candidate_checkpoint_id,
            RunCheckpoint.CONTENT_NAMESPACE,
            "publication permit candidate",
        )
        require_content_id(self.generation_id, "publication permit generation")
        if (
            self.generation_id.split(":sha256:", 1)[0] != "run-derived-state-generation"
            or _DIGEST_PATTERN.fullmatch(self.bundle_digest) is None
            or type(self.expected_journal_size_bytes) is not int
            or self.expected_journal_size_bytes <= 0
            or type(self.bundle_size_bytes) is not int
            or self.bundle_size_bytes <= 0
            or type(self._publisher_identity) is not object
            or self._authority is not _PUBLICATION_PERMIT_AUTHORITY
        ):
            raise RunStatePublisherError("run-state publication permit is invalid")


@dataclass(frozen=True)
class ReconciledRunFrontier:
    """Live proof that checkpoint, bundle, journal, and views agree."""

    checkpoint: RunCheckpoint
    projection: ReconciledRunStateProjection
    bundle: RunDerivedStateBundle
    journal_head_id: str
    journal_size_bytes: int
    bundle_digest: str
    bundle_size_bytes: int
    checkpoint_identity: tuple[int, int]
    bundle_identity: tuple[int, int]
    view_identities: tuple[RunStateViewIdentity, ...]
    run_root_identity: tuple[int, int]
    control_parent_identity: tuple[int, int]
    object_store_identity: tuple[int, int]
    staging_identity: tuple[int, int]
    _publisher_identity: object = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.checkpoint) is not RunCheckpoint
            or type(self.projection) is not ReconciledRunStateProjection
            or type(self.bundle) is not RunDerivedStateBundle
            or self.bundle.generation != self.checkpoint.derived_state_generation
            or self.bundle_digest != self.bundle.digest
            or self.bundle_size_bytes != self.bundle.byte_size
            or type(self._publisher_identity) is not object
            or self._authority is not _RECONCILED_FRONTIER_AUTHORITY
        ):
            raise RunStatePublisherError(
                "reconciled run frontier lacks exact live authority"
            )
        _require_namespaced_id(
            self.journal_head_id,
            RunCheckpointHead.CONTENT_NAMESPACE,
            "reconciled frontier journal head",
        )
        if type(self.journal_size_bytes) is not int or self.journal_size_bytes <= 0:
            raise RunStatePublisherError(
                "reconciled frontier journal position is invalid"
            )
        for identity, name in (
            (self.checkpoint_identity, "checkpoint"),
            (self.bundle_identity, "bundle"),
            (self.run_root_identity, "run root"),
            (self.control_parent_identity, "control parent"),
            (self.object_store_identity, "object store"),
            (self.staging_identity, "staging"),
        ):
            _require_inode_identity(identity, f"reconciled frontier {name}")
        if (
            type(self.view_identities) is not tuple
            or any(
                type(identity) is not RunStateViewIdentity
                for identity in self.view_identities
            )
            or tuple(identity.relative_path for identity in self.view_identities)
            != tuple(
                sorted(identity.relative_path for identity in self.view_identities)
            )
        ):
            raise RunStatePublisherError(
                "reconciled frontier view identities are invalid"
            )

    @property
    def run_checkpoint_id(self) -> str:
        return self.checkpoint.run_checkpoint_id

    @property
    def generation_id(self) -> str:
        return self.bundle.generation.generation_id

    def require_current(
        self,
        publisher: "RunStatePublisher",
    ) -> RunCheckpoint:
        """Reprove this exact receipt without repairing any persistent state."""
        return publisher.require_current(self)


@dataclass(frozen=True)
class _ReconciledMaterial:
    inspection: _CheckpointFrontierInspection
    checkpoint: RunCheckpoint
    projection: ReconciledRunStateProjection
    bundle: RunDerivedStateBundle
    checkpoint_identity: tuple[int, int]
    bundle_identity: tuple[int, int]
    view_identities: tuple[RunStateViewIdentity, ...]


class RunStatePublisher:
    """Sole mutable authority for checkpoint-governed run state."""

    def __init__(
        self,
        authority: ActiveLaunchWorkspace,
        settings: LaunchSettings,
    ) -> None:
        if type(authority) is not ActiveLaunchWorkspace:
            raise RunStatePublisherError(
                "run-state publisher requires active launch authority"
            )
        if type(settings) is not LaunchSettings:
            raise RunStatePublisherError(
                "run-state publisher requires exact launch settings"
            )
        self._authority = authority
        self._settings = settings
        self._checkpoint = _RunCheckpointControl(authority, settings)
        self._action_store = RunActionExecutionStore(
            active_workspace=authority,
            settings=settings,
            _authority=_RUN_ACTION_STORE_AUTHORITY,
        )
        self._publisher_identity = object()
        self._permit_lock = Lock()
        self._issued_permits: dict[int, RunStatePublicationPermit] = {}
        self._receipt_lock = Lock()
        self._issued_receipts: WeakValueDictionary[int, ReconciledRunFrontier] = (
            WeakValueDictionary()
        )
        self._store_relative = _require_relative_control_path(
            settings.run_derived_state_store_path,
            "run derived-state object store",
        )
        self._staging_relative = _require_relative_control_path(
            settings.run_derived_state_staging_path,
            "run derived-state staging",
        )
        self._view_bound_by_path = {
            settings.run_action_ledger_path: settings.run_action_projection_size_bytes,
            settings.run_idea_archive_path: settings.run_idea_archive_size_bytes,
            settings.run_experiment_history_path: (
                settings.run_experiment_history_size_bytes
            ),
            settings.run_execution_journal_path: (
                settings.run_execution_journal_size_bytes
            ),
        }
        all_paths = (
            self._store_relative,
            self._staging_relative,
            *(
                _require_relative_control_path(path, "run-state view")
                for path in self._view_bound_by_path
            ),
        )
        if any(
            path.parent != self._checkpoint._control_parent_relative
            for path in all_paths
        ):
            raise RunStatePublisherError(
                "run-state publisher controls require one shared parent"
            )
        self._store_identity: tuple[int, int] | None = None
        self._staging_identity: tuple[int, int] | None = None
        with ExitStack() as descriptors:
            parent_descriptor = self._checkpoint._open_control_parent(descriptors)
            self._checkpoint._open_locked(parent_descriptor, descriptors)
            self._checkpoint._clean_staging(parent_descriptor, descriptors)
            store_descriptor = self._open_derived_directory(
                parent_descriptor,
                self._store_relative.name,
                None,
                "run derived-state object store",
                descriptors,
            )
            staging_descriptor = self._open_derived_directory(
                parent_descriptor,
                self._staging_relative.name,
                None,
                "run derived-state staging",
                descriptors,
            )
            self._store_identity = _directory_identity(
                store_descriptor,
                "run derived-state object store",
            )
            self._staging_identity = _directory_identity(
                staging_descriptor,
                "run derived-state staging",
            )
            self._clean_derived_staging(staging_descriptor)
            self._validate_store(store_descriptor)

    def issue_publication_permit(
        self,
        observed_frontier: ReconciledRunFrontier | None,
        candidate: RunCheckpoint,
        bundle: RunDerivedStateBundle,
    ) -> RunStatePublicationPermit:
        """Seal one reconciled frontier and exact candidate for later publication."""
        if observed_frontier is not None:
            self._require_issued_receipt(observed_frontier)
        projection = self._validate_candidate_bundle(candidate, bundle)
        with ExitStack() as descriptors:
            parent_descriptor = self._open_transaction(descriptors)
            current = self._load_locked(
                parent_descriptor,
                descriptors,
                repair=True,
            )
            observed_checkpoint = None if current is None else current.checkpoint
            observed_id = (
                None
                if observed_checkpoint is None
                else observed_checkpoint.run_checkpoint_id
            )
            if current is None:
                if observed_frontier is not None:
                    raise RunStatePublisherError(
                        "fresh publication received a non-fresh frontier"
                    )
            elif observed_frontier is None or not self._receipt_matches(
                observed_frontier, current
            ):
                raise RunStatePublisherError(
                    "run-state publication lacks the current reconciled frontier"
                )
            if observed_checkpoint == candidate:
                if (
                    current is None
                    or current.bundle != bundle
                    or bundle.generation.predecessor_checkpoint_head_id
                    != current.inspection.head.predecessor_head_id
                ):
                    raise RunStatePublisherError(
                        "idempotent publication names another predecessor or bundle"
                    )
            else:
                candidate.require_predecessor(observed_checkpoint)
                if bundle.generation.predecessor_checkpoint_head_id != (
                    RunCheckpointHead.initial(
                        self._authority.bootstrap_pin
                    ).run_checkpoint_head_id
                    if current is None
                    else current.inspection.head.run_checkpoint_head_id
                ):
                    raise RunStatePublisherError(
                        "candidate generation names another predecessor head"
                    )
                if current is not None:
                    projection.require_predecessor(current.projection)
            self._action_store.lock_workspace(
                RunFrontierWorkspaceAccess.READ_ONLY,
                descriptors,
            )
            self._require_action_publication_candidate(
                current,
                candidate,
                projection,
                self._action_store.inspect_locked(descriptors),
            )
            inspection = (
                self._fresh_inspection(parent_descriptor, descriptors)
                if current is None
                else current.inspection
            )
            permit = RunStatePublicationPermit(
                requested_predecessor_checkpoint_id=(
                    candidate.predecessor_checkpoint_id
                ),
                observed_checkpoint_id=observed_id,
                expected_journal_head_id=(inspection.head.run_checkpoint_head_id),
                expected_journal_size_bytes=inspection.journal_size_bytes,
                candidate_checkpoint_id=candidate.run_checkpoint_id,
                generation_id=bundle.generation.generation_id,
                bundle_digest=bundle.digest,
                bundle_size_bytes=bundle.byte_size,
                _publisher_identity=self._publisher_identity,
                _authority=_PUBLICATION_PERMIT_AUTHORITY,
            )
        with self._permit_lock:
            self._issued_permits[id(permit)] = permit
        return permit

    def publish(
        self,
        permit: RunStatePublicationPermit,
        candidate: RunCheckpoint,
        bundle: RunDerivedStateBundle,
    ) -> ReconciledRunFrontier:
        """Consume one permit and publish a complete reconciled generation."""
        self._consume_permit(permit)
        projection = self._validate_candidate_bundle(candidate, bundle)
        if (
            permit.candidate_checkpoint_id != candidate.run_checkpoint_id
            or permit.generation_id != bundle.generation.generation_id
            or permit.bundle_digest != bundle.digest
            or permit.bundle_size_bytes != bundle.byte_size
        ):
            raise RunStatePublisherError(
                "run-state publication permit authorizes another candidate"
            )
        with ExitStack() as descriptors:
            parent_descriptor = self._open_transaction(descriptors)
            current = self._load_locked(
                parent_descriptor,
                descriptors,
                repair=True,
            )
            inspection = (
                self._fresh_inspection(parent_descriptor, descriptors)
                if current is None
                else current.inspection
            )
            current_id = (
                None if current is None else current.checkpoint.run_checkpoint_id
            )
            if (
                current_id != permit.observed_checkpoint_id
                or inspection.head.run_checkpoint_head_id
                != permit.expected_journal_head_id
                or inspection.journal_size_bytes != permit.expected_journal_size_bytes
            ):
                raise RunStatePublisherError(
                    "run-state publication frontier moved after permit issuance"
                )
            self._action_store.lock_workspace(
                RunFrontierWorkspaceAccess.READ_ONLY,
                descriptors,
            )
            self._require_action_publication_candidate(
                current,
                candidate,
                projection,
                self._action_store.inspect_locked(descriptors),
            )
            if current is not None and current.checkpoint == candidate:
                if (
                    permit.requested_predecessor_checkpoint_id
                    != candidate.predecessor_checkpoint_id
                    or current.bundle != bundle
                ):
                    raise RunStatePublisherError(
                        "idempotent publication differs from its durable candidate"
                    )
                return self._mint_receipt(current)
            if current_id != permit.requested_predecessor_checkpoint_id:
                raise RunStatePublisherError(
                    "run-state publication predecessor changed"
                )
            candidate.require_predecessor(
                None if current is None else current.checkpoint
            )
            if current is not None:
                projection.require_predecessor(current.projection)
            successor_head = inspection.head.advance(candidate)
            self._checkpoint._require_journal_append_capacity(
                parent_descriptor,
                inspection.head,
                successor_head.to_json_bytes() + b"\n",
                descriptors,
            )
            self._require_store_capacity(
                parent_descriptor,
                bundle.object_name,
                descriptors,
            )
            self._publish_bundle(
                parent_descriptor,
                bundle,
                descriptors,
            )
            self._checkpoint._commit_checkpoint(
                parent_descriptor,
                inspection,
                candidate,
                descriptors,
            )
            self._promote_views(
                parent_descriptor,
                bundle,
                repair=True,
                descriptors=descriptors,
            )
            reconciled = self._load_locked(
                parent_descriptor,
                descriptors,
                repair=False,
            )
            if (
                reconciled is None
                or reconciled.checkpoint != candidate
                or reconciled.bundle != bundle
            ):
                raise RunStatePublisherError(
                    "published run-state frontier failed final reconciliation"
                )
            return self._mint_receipt(reconciled)

    def load_reconciled(self) -> ReconciledRunFrontier | None:
        """Recover safe crash seams, repair views, and return the current frontier."""
        with ExitStack() as descriptors:
            parent_descriptor = self._open_transaction(descriptors)
            material = self._load_locked(
                parent_descriptor,
                descriptors,
                repair=True,
            )
            return None if material is None else self._mint_receipt(material)

    def require_current(
        self,
        receipt: ReconciledRunFrontier,
    ) -> RunCheckpoint:
        """Verify one live receipt without repairing any persistent state."""
        self._require_issued_receipt(receipt)
        with ExitStack() as descriptors:
            return self._hold_current(receipt, descriptors)

    def action_ledger_snapshot(self) -> RunActionLedgerSnapshot:
        """Return the current durable ledger for candidate construction."""
        self._authority.require_control_authority()
        return self._action_store.snapshot()

    def _require_action_publication_candidate(
        self,
        current: _ReconciledMaterial | None,
        candidate: RunCheckpoint,
        projection: ReconciledRunStateProjection,
        inspection: RunActionStoreInspection,
    ) -> None:
        if (
            type(inspection) is not RunActionStoreInspection
            or projection.action_ledger != inspection.ledger
        ):
            raise RunStatePublisherError(
                "candidate action ledger differs from durable execution state"
            )
        terminal_kinds = {
            RunActionExecutionEventKind.RESULT_ACCEPTED,
            RunActionExecutionEventKind.CANCELLED,
            RunActionExecutionEventKind.FRONTIER_INVALIDATED,
        }
        if any(
            tail.tail_kind not in terminal_kinds
            for tail in inspection.ledger.operation_tails
        ):
            raise RunStatePublisherError(
                "candidate action ledger contains an unresolved execution"
            )
        if current is None:
            if inspection.ledger != RunActionLedgerSnapshot.empty():
                raise RunStatePublisherError("genesis action ledger must be empty")
            return
        ordered_new_operations = inspection.operations_since(
            current.projection.action_ledger,
        )
        if any(
            events[0].reservation.frontier.run_checkpoint_id
            != current.checkpoint.run_checkpoint_id
            for events in ordered_new_operations
        ):
            raise RunStatePublisherError(
                "candidate action ledger contains an execution from another frontier"
            )
        for events in ordered_new_operations:
            self._require_action_frontier_binding(
                current,
                events[0].reservation,
            )
        workspace_pairs = inspection.workspace_chain(
            ordered_new_operations,
        )
        workspace_changes = tuple(
            pair for pair in workspace_pairs if pair[0] != pair[1]
        )
        if len(workspace_changes) > 1:
            raise RunStatePublisherError(
                "one checkpoint cannot reconcile multiple workspace edits"
            )
        workspace_change = None if not workspace_changes else workspace_changes[0]
        if workspace_change is not None and (
            workspace_change[0] is None or workspace_change[1] is None
        ):
            raise RunStatePublisherError(
                "workspace-changing action lacks exact before and after frontiers"
            )
        self._require_candidate_workspace_evidence(
            current.checkpoint,
            candidate,
            workspace_change,
            None if not workspace_pairs else workspace_pairs[-1][1],
        )

    def _require_candidate_workspace_evidence(
        self,
        current_checkpoint: RunCheckpoint,
        candidate: RunCheckpoint,
        workspace_change: (
            tuple[RunActionWorkspaceBinding, RunActionWorkspaceBinding] | None
        ),
        final_workspace: RunActionWorkspaceBinding | None,
    ) -> None:
        branch = self._settings.workspace_git_branch
        current_evidence = current_checkpoint.safety_state.derivative_frontier.evidence
        candidate_evidence = candidate.safety_state.derivative_frontier.evidence
        current_commit = current_evidence.branch_heads.get(branch)
        if current_commit is None:
            raise RunStatePublisherError(
                "current checkpoint omits its configured workspace branch"
            )
        expected_commit = (
            current_commit if final_workspace is None else final_workspace.commit_sha
        )
        with ExitStack() as descriptors:
            workspace_descriptor, _identity = self._authority._open_execution_workspace(
                descriptors
            )
            observed = inspect_run_workspace_frontier(
                workspace_descriptor,
                settings=self._settings,
                expected_commit_sha=expected_commit,
            )
        if final_workspace is not None and observed != final_workspace.to_identity():
            raise RunStatePublisherError(
                "durable terminal workspace changed before publication"
            )
        current_ids = {
            advance.branch_advance_id for advance in current_evidence.branch_advances
        }
        new_branch_advances = tuple(
            advance
            for advance in candidate_evidence.branch_advances
            if advance.branch_advance_id not in current_ids and advance.branch == branch
        )
        if workspace_change is None:
            if (
                candidate_evidence.branch_heads.get(branch) != current_commit
                or new_branch_advances
            ):
                raise RunStatePublisherError(
                    "checkpoint changes workspace evidence without a durable edit"
                )
            return
        before, after = workspace_change
        if (
            before.branch != branch
            or before.commit_sha != current_commit
            or after.branch != branch
        ):
            raise RunStatePublisherError(
                "durable workspace edit differs from the checkpoint frontier"
            )
        terminal = tuple(
            advance
            for advance in current_evidence.branch_advances
            if advance.branch == branch and advance.commit_sha == before.commit_sha
        )
        predecessor_advance_id = (
            None
            if current_evidence.branch_origin_heads[branch] == before.commit_sha
            else terminal[0].branch_advance_id if len(terminal) == 1 else ""
        )
        if (
            candidate_evidence.branch_heads.get(branch) != after.commit_sha
            or len(new_branch_advances) != 1
            or new_branch_advances[0].predecessor_commit_sha != before.commit_sha
            or new_branch_advances[0].commit_sha != after.commit_sha
            or new_branch_advances[0].predecessor_branch_advance_id
            != predecessor_advance_id
            or new_branch_advances[0].authorization_safety_state_id
            != current_checkpoint.safety_state.safety_state_id
        ):
            raise RunStatePublisherError(
                "checkpoint does not exactly account for its durable workspace edit"
            )

    def _require_action_frontier_binding(
        self,
        current: _ReconciledMaterial,
        reservation: RunActionReservation,
    ) -> None:
        if type(reservation) is not RunActionReservation:
            raise RunStatePublisherError(
                "run action publication requires one exact reservation"
            )
        checkpoint = current.checkpoint
        safety = checkpoint.safety_state
        binding = reservation.frontier
        expected_views = tuple(
            RunActionViewBinding(
                relative_path=identity.relative_path,
                digest=identity.digest,
                size_bytes=identity.size_bytes,
            )
            for identity in current.view_identities
        )
        if (
            type(binding) is not RunActionFrontierBinding
            or checkpoint.status is not RunCheckpointStatus.ACTIVE
            or checkpoint.last_stop is not None
            or safety.disposition is RunEligibilityDisposition.SECURITY_BLOCKED
            or reservation.intent.boundary is not safety.boundary
            or binding.bootstrap_pin_id != safety.bootstrap_pin.bootstrap_pin_id
            or binding.run_checkpoint_id != checkpoint.run_checkpoint_id
            or binding.safety_state_id != safety.safety_state_id
            or binding.security_observation_id
            != safety.security_observation.observation_id
            or binding.generation_id
            != checkpoint.derived_state_generation.generation_id
            or binding.journal_head_id != current.inspection.head.run_checkpoint_head_id
            or binding.journal_size_bytes != current.inspection.journal_size_bytes
            or binding.bundle_digest != current.bundle.digest
            or binding.bundle_size_bytes != current.bundle.byte_size
            or binding.view_bindings != expected_views
        ):
            raise RunStatePublisherError(
                "run action reservation differs from the current frontier"
            )
        workspace_before = binding.workspace_before
        if workspace_before is None:
            return
        branch = self._settings.workspace_git_branch
        if (
            workspace_before.branch != branch
            or workspace_before.commit_sha
            != safety.derivative_frontier.evidence.branch_heads.get(branch)
        ):
            raise RunStatePublisherError(
                "run action workspace binding differs from the current frontier"
            )

    def _hold_current(
        self,
        receipt: ReconciledRunFrontier,
        descriptors: ExitStack,
    ) -> RunCheckpoint:
        """Hold a shared lock while proving one exact live frontier."""
        if type(descriptors) is not ExitStack:
            raise RunStatePublisherError(
                "shared frontier authority requires one descriptor stack"
            )
        self._require_issued_receipt(receipt)
        self._authority.require_control_authority()
        parent_descriptor = self._checkpoint._open_control_parent(descriptors)
        self._checkpoint._open_locked(
            parent_descriptor,
            descriptors,
            shared=True,
        )
        self._checkpoint._open_staging(parent_descriptor, descriptors)
        store_descriptor = self._open_store(parent_descriptor, descriptors)
        self._open_staging(parent_descriptor, descriptors)
        self._validate_store(store_descriptor)
        material = self._load_locked(
            parent_descriptor,
            descriptors,
            repair=False,
            clean_staging=False,
        )
        if material is None or not self._receipt_matches(receipt, material):
            raise RunStatePublisherError("reconciled run frontier is no longer current")
        self._authority.require_control_authority()
        return material.checkpoint

    def _open_transaction(
        self,
        descriptors: ExitStack,
        *,
        clean_staging: bool = True,
    ) -> int:
        self._authority.require_control_authority()
        parent_descriptor = self._checkpoint._open_control_parent(descriptors)
        self._checkpoint._open_locked(parent_descriptor, descriptors)
        checkpoint_staging_descriptor = self._checkpoint._open_staging(
            parent_descriptor,
            descriptors,
        )
        store_descriptor = self._open_store(parent_descriptor, descriptors)
        staging_descriptor = self._open_staging(parent_descriptor, descriptors)
        for descriptor in (
            parent_descriptor,
            checkpoint_staging_descriptor,
            store_descriptor,
            staging_descriptor,
        ):
            os.fsync(descriptor)
        if clean_staging:
            self._checkpoint._clean_staging(parent_descriptor, descriptors)
            self._clean_derived_staging(staging_descriptor)
        self._validate_store(store_descriptor)
        self._authority.require_control_authority()
        return parent_descriptor

    def _fresh_inspection(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> _CheckpointFrontierInspection:
        inspection = self._checkpoint._inspect_frontier(
            parent_descriptor,
            descriptors,
        )
        if inspection.checkpoint is not None or inspection.checkpoint_ahead:
            raise RunStatePublisherError(
                "fresh run-state frontier unexpectedly contains a checkpoint"
            )
        return inspection

    def _load_locked(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
        *,
        repair: bool,
        clean_staging: bool = True,
    ) -> _ReconciledMaterial | None:
        if clean_staging:
            self._checkpoint._clean_staging(parent_descriptor, descriptors)
            self._clean_derived_staging(
                self._open_staging(parent_descriptor, descriptors)
            )
        inspection = self._checkpoint._inspect_frontier(
            parent_descriptor,
            descriptors,
        )
        if inspection.checkpoint is None:
            self._require_no_fresh_views(parent_descriptor)
            return None
        bundle, bundle_identity = self._read_bundle(
            parent_descriptor,
            inspection.checkpoint.derived_state_generation.generation_id,
            descriptors,
        )
        projection = self._validate_candidate_bundle(
            inspection.checkpoint,
            bundle,
        )
        if inspection.checkpoint_ahead:
            if not repair:
                raise RunStatePublisherError("run checkpoint is ahead of its journal")
            inspection = self._checkpoint._recover_checkpoint_ahead(
                parent_descriptor,
                inspection,
                descriptors,
            )
        view_identities = self._promote_views(
            parent_descriptor,
            bundle,
            repair=repair,
            descriptors=descriptors,
        )
        checkpoint_identity = self._checkpoint_file_identity(parent_descriptor)
        self._authority.require_control_authority()
        return _ReconciledMaterial(
            inspection=inspection,
            checkpoint=inspection.checkpoint,
            projection=projection,
            bundle=bundle,
            checkpoint_identity=checkpoint_identity,
            bundle_identity=bundle_identity,
            view_identities=view_identities,
        )

    def _validate_candidate_bundle(
        self,
        candidate: RunCheckpoint,
        bundle: RunDerivedStateBundle,
    ) -> ReconciledRunStateProjection:
        if (
            type(candidate) is not RunCheckpoint
            or type(bundle) is not RunDerivedStateBundle
        ):
            raise RunStatePublisherError(
                "run-state publication requires exact checkpoint and bundle"
            )
        candidate.require_bootstrap_pin(self._authority.bootstrap_pin)
        if candidate.derived_state_generation != bundle.generation:
            raise RunStatePublisherError(
                "run checkpoint names another derived-state generation"
            )
        self._require_bundle_bounds(bundle)
        projection = ReconciledRunStateProjection.from_bundle(
            bundle,
            strategy_state=candidate.strategy_state,
            bootstrap_pin=self._authority.bootstrap_pin,
        )
        expected_paths = {
            binding.relative_path
            for binding in bundle.generation.run_state_layout.bindings
        }
        if not expected_paths.issubset(self._view_bound_by_path):
            raise RunStatePublisherError(
                "run-state bundle contains an unconfigured view"
            )
        return projection

    def _require_bundle_bounds(self, bundle: RunDerivedStateBundle) -> None:
        if (
            bundle.byte_size <= 0
            or bundle.byte_size > self._settings.run_derived_generation_size_bytes
        ):
            raise RunStatePublisherError(
                "run derived-state generation exceeds its configured bound"
            )
        for binding, payload in zip(
            bundle.generation.run_state_layout.bindings,
            bundle.payloads,
            strict=True,
        ):
            bound = self._view_bound_by_path.get(binding.relative_path)
            if bound is None or len(payload) > bound:
                raise RunStatePublisherError(
                    "run-state view exceeds its configured bound"
                )

    def _publish_bundle(
        self,
        parent_descriptor: int,
        bundle: RunDerivedStateBundle,
        descriptors: ExitStack,
    ) -> tuple[int, int]:
        store_descriptor = self._open_store(parent_descriptor, descriptors)
        staging_descriptor = self._open_staging(parent_descriptor, descriptors)
        digest = bundle.generation.generation_id.rsplit(":", 1)[1]
        temporary_name = f"generation-{digest}-{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=staging_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        written_size = 0
        for chunk in bundle.iter_bytes():
            _write_all(descriptor, chunk)
            written_size += len(chunk)
            if written_size > self._settings.run_derived_generation_size_bytes:
                raise RunStatePublisherError(
                    "staged run derived-state bundle exceeds its configured bound"
                )
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not _is_exact_private_file(
                metadata,
                mode=0o400,
                maximum_bytes=self._settings.run_derived_generation_size_bytes,
                allow_empty=False,
            )
            or metadata.st_size != bundle.byte_size
        ):
            raise RunStatePublisherError("staged run derived-state bundle is unsafe")
        os.lseek(descriptor, 0, os.SEEK_SET)
        read_descriptor = os.dup(descriptor)
        with os.fdopen(read_descriptor, "rb") as handle:
            staged = RunDerivedStateBundle.read_from(
                handle,
                maximum_bytes=self._settings.run_derived_generation_size_bytes,
            )
        if staged != bundle:
            raise RunStatePublisherError(
                "staged run derived-state bundle differs from its candidate"
            )
        published = _rename_file_no_replace(
            staging_descriptor,
            temporary_name,
            store_descriptor,
            bundle.object_name,
            (metadata.st_dev, metadata.st_ino),
        )
        if published:
            os.fsync(store_descriptor)
            os.fsync(staging_descriptor)
        else:
            existing, identity = self._read_bundle(
                parent_descriptor,
                bundle.generation.generation_id,
                descriptors,
            )
            if existing != bundle:
                raise RunStatePublisherError(
                    "existing generation object differs from its content name"
                )
            os.unlink(temporary_name, dir_fd=staging_descriptor)
            os.fsync(staging_descriptor)
            return identity
        persisted, identity = self._read_bundle(
            parent_descriptor,
            bundle.generation.generation_id,
            descriptors,
        )
        if persisted != bundle or identity != (metadata.st_dev, metadata.st_ino):
            raise RunStatePublisherError(
                "published generation object differs from its staged candidate"
            )
        return identity

    def _read_bundle(
        self,
        parent_descriptor: int,
        generation_id: str,
        descriptors: ExitStack,
    ) -> tuple[RunDerivedStateBundle, tuple[int, int]]:
        _require_namespaced_id(
            generation_id,
            "run-derived-state-generation",
            "retained generation",
        )
        object_name = f"generation-{generation_id.rsplit(':', 1)[1]}.bundle"
        if _GENERATION_OBJECT_PATTERN.fullmatch(object_name) is None:
            raise RunStatePublisherError("retained generation name is invalid")
        store_descriptor = self._open_store(parent_descriptor, descriptors)
        if not os.access(
            object_name,
            os.F_OK,
            dir_fd=store_descriptor,
            follow_symlinks=False,
        ):
            raise RunStatePublisherError(
                "referenced run derived-state bundle is absent"
            )
        descriptor = os.open(
            object_name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=store_descriptor,
        )
        handle = descriptors.enter_context(os.fdopen(descriptor, "rb"))
        metadata = os.fstat(handle.fileno())
        if not _is_exact_private_file(
            metadata,
            mode=0o400,
            maximum_bytes=self._settings.run_derived_generation_size_bytes,
            allow_empty=False,
        ):
            raise RunStatePublisherError(
                "retained generation object is not one bounded private file"
            )
        bundle = RunDerivedStateBundle.read_from(
            handle,
            maximum_bytes=self._settings.run_derived_generation_size_bytes,
        )
        reopened = os.fstat(handle.fileno())
        rebound = os.stat(
            object_name,
            dir_fd=store_descriptor,
            follow_symlinks=False,
        )
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            (
                reopened.st_dev,
                reopened.st_ino,
                reopened.st_size,
                stat.S_IMODE(reopened.st_mode),
            )
            != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                stat.S_IMODE(metadata.st_mode),
            )
            or (rebound.st_dev, rebound.st_ino) != identity
            or bundle.generation.generation_id != generation_id
        ):
            raise RunStatePublisherError(
                "retained generation object changed while reading"
            )
        return bundle, identity

    def _promote_views(
        self,
        parent_descriptor: int,
        bundle: RunDerivedStateBundle,
        *,
        repair: bool,
        descriptors: ExitStack,
    ) -> tuple[RunStateViewIdentity, ...]:
        payload_by_path = bundle.payload_by_relative_path()
        configured_idea_path = self._settings.run_idea_archive_path
        if configured_idea_path not in payload_by_path and os.access(
            PurePosixPath(configured_idea_path).name,
            os.F_OK,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        ):
            raise RunStatePublisherError(
                "tree run-state frontier contains a stray idea archive"
            )
        identities = []
        for relative_path in sorted(payload_by_path):
            payload = payload_by_path[relative_path]
            current = self._read_view(
                parent_descriptor,
                relative_path,
                allow_absent=True,
                descriptors=descriptors,
            )
            staged_identity = None
            if current is None or current[0] != payload:
                if not repair:
                    raise RunStatePublisherError(
                        "run-state view differs from its retained generation"
                    )
                staged_identity = self._replace_view(
                    parent_descriptor,
                    relative_path,
                    payload,
                    bundle.generation.generation_id,
                    descriptors,
                )
            exact = self._read_view(
                parent_descriptor,
                relative_path,
                allow_absent=False,
                descriptors=descriptors,
            )
            if exact is None or exact[0] != payload:
                raise RunStatePublisherError(
                    "promoted run-state view differs from its retained generation"
                )
            if (
                staged_identity is not None
                and (
                    exact[1].device,
                    exact[1].inode,
                )
                != staged_identity
            ):
                raise RunStatePublisherError(
                    "promoted run-state view differs from its staged inode"
                )
            identities.append(exact[1])
        return tuple(identities)

    def _replace_view(
        self,
        parent_descriptor: int,
        relative_path: str,
        payload: bytes,
        generation_id: str,
        descriptors: ExitStack,
    ) -> tuple[int, int]:
        staging_descriptor = self._open_staging(parent_descriptor, descriptors)
        generation_digest = generation_id.rsplit(":", 1)[1]
        temporary_name = f"generation-{generation_digest}-{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=staging_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        bound = self._view_bound_by_path[relative_path]
        if not _is_exact_private_file(
            metadata,
            mode=0o400,
            maximum_bytes=bound,
            allow_empty=True,
        ) or metadata.st_size != len(payload):
            raise RunStatePublisherError("staged run-state view is unsafe")
        os.replace(
            temporary_name,
            PurePosixPath(relative_path).name,
            src_dir_fd=staging_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
        os.fsync(staging_descriptor)
        return metadata.st_dev, metadata.st_ino

    def _read_view(
        self,
        parent_descriptor: int,
        relative_path: str,
        *,
        allow_absent: bool,
        descriptors: ExitStack,
    ) -> tuple[bytes, RunStateViewIdentity] | None:
        path = _require_relative_control_path(relative_path, "run-state view")
        if path.parent != self._checkpoint._control_parent_relative:
            raise RunStatePublisherError(
                "run-state view lies outside the control parent"
            )
        exists = os.access(
            path.name,
            os.F_OK,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not exists:
            if allow_absent:
                return None
            raise RunStatePublisherError("required run-state view is absent")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        handle = descriptors.enter_context(os.fdopen(descriptor, "rb"))
        metadata = os.fstat(handle.fileno())
        bound = self._view_bound_by_path[relative_path]
        if not _is_exact_private_file(
            metadata,
            mode=0o400,
            maximum_bytes=bound,
            allow_empty=True,
        ):
            raise RunStatePublisherError(
                "run-state view is not one bounded private file"
            )
        payload = handle.read(bound + 1)
        reopened = os.fstat(handle.fileno())
        rebound = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            len(payload) > bound
            or (
                reopened.st_dev,
                reopened.st_ino,
                reopened.st_size,
                stat.S_IMODE(reopened.st_mode),
            )
            != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                stat.S_IMODE(metadata.st_mode),
            )
            or (rebound.st_dev, rebound.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise RunStatePublisherError("run-state view changed while reading")
        identity = RunStateViewIdentity(
            relative_path=relative_path,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            digest=_sha256_digest(payload),
            size_bytes=len(payload),
        )
        return payload, identity

    def _require_no_fresh_views(self, parent_descriptor: int) -> None:
        for relative_path in self._view_bound_by_path:
            if os.access(
                PurePosixPath(relative_path).name,
                os.F_OK,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            ):
                raise RunStatePublisherError(
                    "fresh run contains a derived-state view without a checkpoint"
                )

    def _require_store_capacity(
        self,
        parent_descriptor: int,
        object_name: str,
        descriptors: ExitStack,
    ) -> None:
        store_descriptor = self._open_store(parent_descriptor, descriptors)
        entries = self._validate_store(store_descriptor)
        exists = os.access(
            object_name,
            os.F_OK,
            dir_fd=store_descriptor,
            follow_symlinks=False,
        )
        if (
            not exists
            and len(entries) >= self._settings.run_derived_state_store_entry_limit
        ):
            raise RunStatePublisherError(
                "run derived-state object store has no publication capacity"
            )

    def _validate_store(self, store_descriptor: int) -> tuple[str, ...]:
        with os.scandir(store_descriptor) as iterator:
            entries = tuple(iterator)
        if len(entries) > self._settings.run_derived_state_store_entry_limit:
            raise RunStatePublisherError(
                "run derived-state object store exceeds its entry bound"
            )
        names = []
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if _GENERATION_OBJECT_PATTERN.fullmatch(
                entry.name
            ) is None or not _is_exact_private_file(
                metadata,
                mode=0o400,
                maximum_bytes=(self._settings.run_derived_generation_size_bytes),
                allow_empty=False,
            ):
                raise RunStatePublisherError(
                    "run derived-state object store contains an unsafe entry"
                )
            names.append(entry.name)
        return tuple(sorted(names))

    def _clean_derived_staging(self, staging_descriptor: int) -> None:
        with os.scandir(staging_descriptor) as iterator:
            entries = tuple(iterator)
        if len(entries) > self._settings.run_derived_state_staging_entry_limit:
            raise RunStatePublisherError(
                "run derived-state staging exceeds its entry bound"
            )
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if (
                _GENERATION_STAGING_PATTERN.fullmatch(entry.name) is None
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
                or metadata.st_size > self._settings.run_derived_generation_size_bytes
            ):
                raise RunStatePublisherError(
                    "run derived-state staging contains an unsafe entry"
                )
            os.unlink(entry.name, dir_fd=staging_descriptor)
        if entries:
            os.fsync(staging_descriptor)

    def _open_store(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> int:
        return self._open_derived_directory(
            parent_descriptor,
            self._store_relative.name,
            self._store_identity,
            "run derived-state object store",
            descriptors,
        )

    def _open_staging(
        self,
        parent_descriptor: int,
        descriptors: ExitStack,
    ) -> int:
        return self._open_derived_directory(
            parent_descriptor,
            self._staging_relative.name,
            self._staging_identity,
            "run derived-state staging",
            descriptors,
        )

    @staticmethod
    def _open_derived_directory(
        parent_descriptor: int,
        name: str,
        expected_identity: tuple[int, int] | None,
        description: str,
        descriptors: ExitStack,
    ) -> int:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        descriptors.callback(os.close, descriptor)
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise RunStatePublisherError(f"{description} is unsafe or was replaced")
        rebound = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(rebound.st_mode)
            or (rebound.st_dev, rebound.st_ino) != identity
        ):
            raise RunStatePublisherError(f"{description} changed while opening")
        return descriptor

    def _checkpoint_file_identity(
        self,
        parent_descriptor: int,
    ) -> tuple[int, int]:
        metadata = os.stat(
            self._checkpoint._checkpoint_relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not _is_exact_private_file(
            metadata,
            mode=0o400,
            maximum_bytes=self._settings.run_checkpoint_size_bytes,
            allow_empty=False,
        ):
            raise RunStatePublisherError("reconciled run checkpoint file is unsafe")
        return metadata.st_dev, metadata.st_ino

    def _mint_receipt(
        self,
        material: _ReconciledMaterial,
    ) -> ReconciledRunFrontier:
        if (
            self._checkpoint._control_parent_identity is None
            or self._store_identity is None
            or self._staging_identity is None
        ):
            raise RunStatePublisherError(
                "run-state publisher lacks pinned directory authority"
            )
        receipt = ReconciledRunFrontier(
            checkpoint=material.checkpoint,
            projection=material.projection,
            bundle=material.bundle,
            journal_head_id=material.inspection.head.run_checkpoint_head_id,
            journal_size_bytes=material.inspection.journal_size_bytes,
            bundle_digest=material.bundle.digest,
            bundle_size_bytes=material.bundle.byte_size,
            checkpoint_identity=material.checkpoint_identity,
            bundle_identity=material.bundle_identity,
            view_identities=material.view_identities,
            run_root_identity=self._authority.published_root_identity,
            control_parent_identity=(self._checkpoint._control_parent_identity),
            object_store_identity=self._store_identity,
            staging_identity=self._staging_identity,
            _publisher_identity=self._publisher_identity,
            _authority=_RECONCILED_FRONTIER_AUTHORITY,
        )
        with self._receipt_lock:
            self._issued_receipts[id(receipt)] = receipt
        return receipt

    def _consume_permit(self, permit: RunStatePublicationPermit) -> None:
        if type(permit) is not RunStatePublicationPermit:
            raise RunStatePublisherError(
                "run-state publication requires one exact permit"
            )
        with self._permit_lock:
            issued = self._issued_permits.pop(id(permit), None)
        if (
            issued is not permit
            or permit._authority is not _PUBLICATION_PERMIT_AUTHORITY
            or permit._publisher_identity is not self._publisher_identity
        ):
            raise RunStatePublisherError(
                "run-state publication permit is cloned, foreign, or consumed"
            )

    def _require_issued_receipt(
        self,
        receipt: ReconciledRunFrontier,
    ) -> None:
        if type(receipt) is not ReconciledRunFrontier:
            raise RunStatePublisherError(
                "current-frontier check requires one exact receipt"
            )
        with self._receipt_lock:
            issued = self._issued_receipts.get(id(receipt))
        if (
            issued is not receipt
            or receipt._authority is not _RECONCILED_FRONTIER_AUTHORITY
            or receipt._publisher_identity is not self._publisher_identity
        ):
            raise RunStatePublisherError(
                "reconciled run frontier is cloned, foreign, or expired"
            )

    @staticmethod
    def _receipt_matches(
        receipt: ReconciledRunFrontier,
        material: _ReconciledMaterial,
    ) -> bool:
        return (
            receipt.checkpoint == material.checkpoint
            and receipt.projection == material.projection
            and receipt.bundle == material.bundle
            and receipt.journal_head_id
            == material.inspection.head.run_checkpoint_head_id
            and receipt.journal_size_bytes == material.inspection.journal_size_bytes
            and receipt.checkpoint_identity == material.checkpoint_identity
            and receipt.bundle_identity == material.bundle_identity
            and receipt.view_identities == material.view_identities
        )


def _require_relative_control_path(value: str, name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
        or len(path.parts) < 2
    ):
        raise RunStatePublisherError(f"{name} path is unsafe")
    return path


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunStatePublisherError(f"{name} uses the wrong namespace")


def _require_inode_identity(identity: tuple[int, int], name: str) -> None:
    if (
        type(identity) is not tuple
        or len(identity) != 2
        or any(type(part) is not int or part < 0 for part in identity)
    ):
        raise RunStatePublisherError(f"{name} identity is invalid")


def _directory_identity(descriptor: int, name: str) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RunStatePublisherError(f"{name} must be a real directory")
    return metadata.st_dev, metadata.st_ino


def _is_exact_private_file(
    metadata: os.stat_result,
    *,
    mode: int,
    maximum_bytes: int,
    allow_empty: bool,
) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == mode
        and (allow_empty or metadata.st_size > 0)
        and metadata.st_size <= maximum_bytes
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise RunStatePublisherError("run-state staged write made no progress")
        remaining = remaining[written:]


def _sha256_digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _rename_file_no_replace(
    source_directory: int,
    source_name: str,
    destination_directory: int,
    destination_name: str,
    expected_source_identity: tuple[int, int],
) -> bool:
    source = os.stat(
        source_name,
        dir_fd=source_directory,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(source.st_mode)
        or (source.st_dev, source.st_ino) != expected_source_identity
    ):
        raise RunStatePublisherError(
            "staged generation identity changed before publication"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise RunStatePublisherError(
            "atomic no-replace generation publication is unavailable"
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
        source_directory,
        os.fsencode(source_name),
        destination_directory,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        return False
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_name,
    )


__all__ = [
    "ReconciledRunFrontier",
    "RunStatePublicationPermit",
    "RunStatePublisher",
    "RunStatePublisherError",
    "RunStateViewIdentity",
]
