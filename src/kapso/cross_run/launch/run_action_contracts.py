"""Canonical request identities for run-scoped external actions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import require_identifier, tree_or_blob_digest
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class RunActionContractError(RuntimeError):
    """A run action request or capability shape is invalid."""


class RunFrontierActionKind(str, Enum):
    """External boundaries that may spend or execute untrusted code."""

    CODING_AGENT = "coding_agent"
    EMBEDDING = "embedding"
    EVALUATOR = "evaluator"


class RunFrontierWorkspaceAccess(str, Enum):
    """Workspace authority required by one external action."""

    NONE = "none"
    READ_ONLY = "read_only"
    EDIT_WORKSPACE = "edit_workspace"


_ALLOWED_ACTION_BOUNDARIES = {
    RunFrontierActionKind.CODING_AGENT: {
        RunSafetyBoundary.IDEATION,
        RunSafetyBoundary.IMPLEMENTATION,
        RunSafetyBoundary.EVALUATION,
    },
    RunFrontierActionKind.EMBEDDING: {RunSafetyBoundary.IDEATION},
    RunFrontierActionKind.EVALUATOR: {RunSafetyBoundary.EVALUATION},
}

_ALLOWED_ACTION_WORKSPACE_ACCESS = {
    (
        RunFrontierActionKind.CODING_AGENT,
        RunSafetyBoundary.IDEATION,
    ): RunFrontierWorkspaceAccess.READ_ONLY,
    (
        RunFrontierActionKind.CODING_AGENT,
        RunSafetyBoundary.IMPLEMENTATION,
    ): RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
    (
        RunFrontierActionKind.CODING_AGENT,
        RunSafetyBoundary.EVALUATION,
    ): RunFrontierWorkspaceAccess.READ_ONLY,
    (
        RunFrontierActionKind.EMBEDDING,
        RunSafetyBoundary.IDEATION,
    ): RunFrontierWorkspaceAccess.NONE,
    (
        RunFrontierActionKind.EVALUATOR,
        RunSafetyBoundary.EVALUATION,
    ): RunFrontierWorkspaceAccess.READ_ONLY,
}


@dataclass(frozen=True)
class RunActionIntent(StrictContract):
    """Content identity derived from one complete boundary request."""

    action_intent_id: str
    kind: RunFrontierActionKind
    boundary: RunSafetyBoundary
    operation_id: str
    request_digest: str
    request_size_bytes: int
    workspace_access: RunFrontierWorkspaceAccess

    CONTENT_NAMESPACE: ClassVar[str] = "run-frontier-action-intent"
    IDENTITY_FIELD: ClassVar[str] = "action_intent_id"

    def _validate(self) -> None:
        if (
            type(self.kind) is not RunFrontierActionKind
            or type(self.boundary) is not RunSafetyBoundary
            or type(self.workspace_access) is not RunFrontierWorkspaceAccess
        ):
            raise RunActionContractError(
                "run action intent uses an unrecognized enum"
            )
        require_identifier(self.operation_id, "run action operation_id")
        if (
            _DIGEST_PATTERN.fullmatch(self.request_digest) is None
            or type(self.request_size_bytes) is not int
            or self.request_size_bytes <= 0
        ):
            raise RunActionContractError(
                "run action intent request identity is invalid"
            )
        if self.boundary not in _ALLOWED_ACTION_BOUNDARIES[self.kind]:
            raise RunActionContractError(
                "run action kind is incompatible with its safety boundary"
            )
        if (
            _ALLOWED_ACTION_WORKSPACE_ACCESS.get((self.kind, self.boundary))
            is not self.workspace_access
        ):
            raise RunActionContractError(
                "run action kind is incompatible with workspace access"
            )

    @classmethod
    def from_request(
        cls,
        *,
        kind: RunFrontierActionKind,
        boundary: RunSafetyBoundary,
        operation_id: str,
        request_payload: bytes,
        workspace_access: RunFrontierWorkspaceAccess,
    ) -> "RunActionIntent":
        if type(request_payload) is not bytes or not request_payload:
            raise RunActionContractError(
                "run action request must be complete non-empty bytes"
            )
        return cls.mint(
            kind=kind,
            boundary=boundary,
            operation_id=operation_id,
            request_digest=tree_or_blob_digest(request_payload),
            request_size_bytes=len(request_payload),
            workspace_access=workspace_access,
        )


__all__ = [
    "RunActionContractError",
    "RunActionIntent",
    "RunFrontierActionKind",
    "RunFrontierWorkspaceAccess",
]
