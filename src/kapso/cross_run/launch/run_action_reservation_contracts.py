"""Durable reservation contracts shared by the action store and supervisor."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_contracts import (
    RunActionContractError,
    RunActionIntent,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import RunActionLedgerSnapshot
from kapso.cross_run.launch.workspace_frontier import RunWorkspaceFrontierIdentity

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class RunActionReservationContractError(RunActionContractError):
    """A durable run-action reservation or dependency relation is invalid."""


@dataclass(frozen=True)
class RunActionViewBinding(StrictContract):
    """Serializable content identity of one checkpoint-owned mutable view."""

    relative_path: str
    digest: str
    size_bytes: int

    def _validate(self) -> None:
        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path
            or "\x00" in self.relative_path
            or _DIGEST_PATTERN.fullmatch(self.digest) is None
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
        ):
            raise RunActionReservationContractError(
                "run action view binding is invalid"
            )


@dataclass(frozen=True)
class RunActionWorkspaceBinding(StrictContract):
    """Serializable clean source/Git identity before or after one action."""

    workspace_device: int
    workspace_inode: int
    branch: str
    commit_sha: str
    parent_commit_shas: tuple[str, ...]
    git_tree_sha: str
    source_tree_digest: str
    git_closure_digest: str
    source_entry_count: int
    source_size_bytes: int

    def _validate(self) -> None:
        identity = RunWorkspaceFrontierIdentity(
            workspace_identity=(self.workspace_device, self.workspace_inode),
            branch=self.branch,
            commit_sha=self.commit_sha,
            parent_commit_shas=self.parent_commit_shas,
            git_tree_sha=self.git_tree_sha,
            source_tree_digest=self.source_tree_digest,
            git_closure_digest=self.git_closure_digest,
            source_entry_count=self.source_entry_count,
            source_size_bytes=self.source_size_bytes,
        )
        if identity.workspace_identity != (
            self.workspace_device,
            self.workspace_inode,
        ):
            raise RunActionReservationContractError(
                "run action workspace binding is invalid"
            )

    @classmethod
    def from_identity(
        cls,
        identity: RunWorkspaceFrontierIdentity,
    ) -> "RunActionWorkspaceBinding":
        if type(identity) is not RunWorkspaceFrontierIdentity:
            raise RunActionReservationContractError(
                "run action workspace binding requires one exact frontier"
            )
        return cls(
            workspace_device=identity.workspace_identity[0],
            workspace_inode=identity.workspace_identity[1],
            branch=identity.branch,
            commit_sha=identity.commit_sha,
            parent_commit_shas=identity.parent_commit_shas,
            git_tree_sha=identity.git_tree_sha,
            source_tree_digest=identity.source_tree_digest,
            git_closure_digest=identity.git_closure_digest,
            source_entry_count=identity.source_entry_count,
            source_size_bytes=identity.source_size_bytes,
        )

    def to_identity(self) -> RunWorkspaceFrontierIdentity:
        return RunWorkspaceFrontierIdentity(
            workspace_identity=(self.workspace_device, self.workspace_inode),
            branch=self.branch,
            commit_sha=self.commit_sha,
            parent_commit_shas=self.parent_commit_shas,
            git_tree_sha=self.git_tree_sha,
            source_tree_digest=self.source_tree_digest,
            git_closure_digest=self.git_closure_digest,
            source_entry_count=self.source_entry_count,
            source_size_bytes=self.source_size_bytes,
        )


@dataclass(frozen=True)
class RunActionFrontierBinding(StrictContract):
    """Complete durable identity of the reconciled frontier authorizing an action."""

    frontier_binding_id: str
    bootstrap_pin_id: str
    run_checkpoint_id: str
    safety_state_id: str
    security_observation_id: str
    generation_id: str
    journal_head_id: str
    journal_size_bytes: int
    bundle_digest: str
    bundle_size_bytes: int
    view_bindings: tuple[RunActionViewBinding, ...]
    workspace_before: RunActionWorkspaceBinding | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-frontier-binding"
    IDENTITY_FIELD: ClassVar[str] = "frontier_binding_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.bootstrap_pin_id, "bootstrap-pin", "bootstrap pin"),
            (self.run_checkpoint_id, "run-checkpoint", "checkpoint"),
            (self.safety_state_id, "run-safety-state", "safety state"),
            (
                self.security_observation_id,
                "security-denylist-observation",
                "security observation",
            ),
            (
                self.generation_id,
                "run-derived-state-generation",
                "derived generation",
            ),
            (self.journal_head_id, "run-checkpoint-head", "checkpoint head"),
        ):
            _require_namespaced_id(value, namespace, f"run action {name}")
        if (
            type(self.journal_size_bytes) is not int
            or self.journal_size_bytes <= 0
            or _DIGEST_PATTERN.fullmatch(self.bundle_digest) is None
            or type(self.bundle_size_bytes) is not int
            or self.bundle_size_bytes <= 0
            or any(
                type(binding) is not RunActionViewBinding
                for binding in self.view_bindings
            )
            or tuple(binding.relative_path for binding in self.view_bindings)
            != tuple(sorted({binding.relative_path for binding in self.view_bindings}))
            or (
                self.workspace_before is not None
                and type(self.workspace_before) is not RunActionWorkspaceBinding
            )
        ):
            raise RunActionReservationContractError(
                "run action frontier binding is invalid"
            )


@dataclass(frozen=True)
class RunActionRequestBlob(StrictContract):
    """Content descriptor for the complete untruncated provider request."""

    request_blob_id: str
    digest: str
    size_bytes: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-request-blob"
    IDENTITY_FIELD: ClassVar[str] = "request_blob_id"

    def _validate(self) -> None:
        if (
            _DIGEST_PATTERN.fullmatch(self.digest) is None
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
        ):
            raise RunActionReservationContractError(
                "run action request blob is invalid"
            )


@dataclass(frozen=True)
class RunActionReservation(StrictContract):
    """One operation identity durably reserved against one exact run frontier."""

    reservation_id: str
    intent: RunActionIntent
    frontier: RunActionFrontierBinding
    request_blob: RunActionRequestBlob
    predecessor_ledger_snapshot_id: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-reservation"
    IDENTITY_FIELD: ClassVar[str] = "reservation_id"

    def _validate(self) -> None:
        if (
            type(self.intent) is not RunActionIntent
            or type(self.frontier) is not RunActionFrontierBinding
            or type(self.request_blob) is not RunActionRequestBlob
        ):
            raise RunActionReservationContractError(
                "run action reservation requires intent and frontier"
            )
        _require_namespaced_id(
            self.predecessor_ledger_snapshot_id,
            RunActionLedgerSnapshot.CONTENT_NAMESPACE,
            "run action predecessor ledger",
        )
        expected = {
            self.intent.action_intent_id,
            self.frontier.frontier_binding_id,
            self.frontier.bootstrap_pin_id,
            self.frontier.run_checkpoint_id,
            self.frontier.safety_state_id,
            self.frontier.security_observation_id,
            self.frontier.generation_id,
            self.frontier.journal_head_id,
            self.request_blob.request_blob_id,
            self.predecessor_ledger_snapshot_id,
        }
        if (
            self.exact_dependency_ids != tuple(sorted(set(self.exact_dependency_ids)))
            or set(self.exact_dependency_ids) != expected
        ):
            raise RunActionReservationContractError(
                "run action reservation dependency closure is not exact"
            )
        if (self.intent.workspace_access is RunFrontierWorkspaceAccess.NONE) != (
            self.frontier.workspace_before is None
        ):
            raise RunActionReservationContractError(
                "run action reservation workspace authority differs from its intent"
            )
        if (
            self.request_blob.digest != self.intent.request_digest
            or self.request_blob.size_bytes != self.intent.request_size_bytes
        ):
            raise RunActionReservationContractError(
                "run action reservation request blob differs from its intent"
            )

    @classmethod
    def build(
        cls,
        *,
        intent: RunActionIntent,
        frontier: RunActionFrontierBinding,
        predecessor_ledger: RunActionLedgerSnapshot,
    ) -> "RunActionReservation":
        if (
            type(intent) is not RunActionIntent
            or type(frontier) is not RunActionFrontierBinding
            or type(predecessor_ledger) is not RunActionLedgerSnapshot
        ):
            raise RunActionReservationContractError(
                "run action reservation requires exact typed inputs"
            )
        request_blob = RunActionRequestBlob.mint(
            digest=intent.request_digest,
            size_bytes=intent.request_size_bytes,
        )
        return cls.mint(
            intent=intent,
            frontier=frontier,
            request_blob=request_blob,
            predecessor_ledger_snapshot_id=predecessor_ledger.ledger_snapshot_id,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        intent.action_intent_id,
                        frontier.frontier_binding_id,
                        frontier.bootstrap_pin_id,
                        frontier.run_checkpoint_id,
                        frontier.safety_state_id,
                        frontier.security_observation_id,
                        frontier.generation_id,
                        frontier.journal_head_id,
                        request_blob.request_blob_id,
                        predecessor_ledger.ledger_snapshot_id,
                    }
                )
            ),
        )


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionReservationContractError(f"{name} uses another namespace")


__all__ = [
    "RunActionFrontierBinding",
    "RunActionRequestBlob",
    "RunActionReservation",
    "RunActionReservationContractError",
    "RunActionViewBinding",
    "RunActionWorkspaceBinding",
]
