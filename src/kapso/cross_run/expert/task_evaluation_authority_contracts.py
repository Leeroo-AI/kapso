"""Authenticated current-or-absent authority for task evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class TaskEvaluationAuthorityError(ValueError):
    """Fresh external task-evaluation authority is invalid or changed."""


@dataclass(frozen=True)
class TaskEvaluationCurrentReleaseObservation(StrictContract):
    observation_id: str
    scope_id: str
    release_id: str | None
    publication_id: str | None
    repository_full_name: str
    repository_node_id: str
    default_branch_head_commit_sha: str
    current_pointer_digest: str | None
    validation_closure_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-evaluation-current-release-observation"
    IDENTITY_FIELD: ClassVar[str] = "observation_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "task evaluation current scope")
        if _REPOSITORY_PATTERN.fullmatch(self.repository_full_name) is None:
            raise TaskEvaluationAuthorityError(
                "task evaluation current repository identity is invalid"
            )
        require_identifier(
            self.repository_node_id,
            "task evaluation current repository node",
        )
        if _COMMIT_PATTERN.fullmatch(self.default_branch_head_commit_sha) is None:
            raise TaskEvaluationAuthorityError(
                "task evaluation current branch head is invalid"
            )
        present = self.release_id is not None
        if present != (self.publication_id is not None) or present != (
            self.current_pointer_digest is not None
        ):
            raise TaskEvaluationAuthorityError(
                "task evaluation current release fields must appear together"
            )
        if present:
            require_content_id(self.release_id, "task evaluation current release")
            require_content_id(
                self.publication_id,
                "task evaluation current publication",
            )
            if (
                self.release_id.split(":sha256:", 1)[0] != "expert-base-release"
                or self.publication_id.split(":sha256:", 1)[0] != "github-publication"
                or _DIGEST_PATTERN.fullmatch(self.current_pointer_digest) is None
            ):
                raise TaskEvaluationAuthorityError(
                    "task evaluation current release authority is invalid"
                )
        elif self.validation_closure_ids:
            raise TaskEvaluationAuthorityError(
                "task evaluation absent current cannot name validation closure"
            )
        if self.validation_closure_ids != tuple(
            sorted(set(self.validation_closure_ids))
        ):
            raise TaskEvaluationAuthorityError(
                "task evaluation current validation closure is not canonical"
            )
        for closure_id in self.validation_closure_ids:
            require_content_id(
                closure_id,
                "task evaluation current validation closure",
            )
