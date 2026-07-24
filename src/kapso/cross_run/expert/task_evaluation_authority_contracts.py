"""Authenticated current-or-absent authority for task evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationInvocationAllocation,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)

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


@dataclass(frozen=True)
class TaskEvaluationSpawnAuthorityFence(StrictContract):
    fence_id: str
    reservation_id: str
    request_id: str
    invocation_allocation: TaskEvaluationInvocationAllocation
    stable_current_release_observation: TaskEvaluationCurrentReleaseObservation
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...]
    allowed_control_security_subject_ids: tuple[str, ...]
    security_denylist_observation: SecurityDenylistObservation

    CONTENT_NAMESPACE: ClassVar[str] = "task-evaluation-spawn-authority-fence"
    IDENTITY_FIELD: ClassVar[str] = "fence_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.reservation_id,
                "task-evaluation-reservation",
                "task evaluation spawn reservation",
            ),
            (
                self.request_id,
                "task-evaluation-request",
                "task evaluation spawn request",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise TaskEvaluationAuthorityError(f"{name} uses the wrong namespace")
        if self.invocation_allocation.reservation_id != self.reservation_id:
            raise TaskEvaluationAuthorityError(
                "task evaluation spawn allocation uses another reservation"
            )
        current = self.stable_current_release_observation
        observation_ids = tuple(
            observation.observation_id
            for observation in self.task_adapter_trust_observations
        )
        if not observation_ids or observation_ids != tuple(
            sorted(set(observation_ids))
        ):
            raise TaskEvaluationAuthorityError(
                "task evaluation spawn adapter observations are noncanonical"
            )
        denylist = self.security_denylist_observation
        if self.allowed_control_security_subject_ids != tuple(
            sorted(set(self.allowed_control_security_subject_ids))
        ):
            raise TaskEvaluationAuthorityError(
                "task evaluation allowed control security subjects are noncanonical"
            )
        for subject_id in self.allowed_control_security_subject_ids:
            require_content_id(
                subject_id,
                "task evaluation allowed control security subject",
            )
        if not set(self.allowed_control_security_subject_ids).issubset(
            denylist.checked_subject_ids
        ) or not set(denylist.matched_subject_ids).issubset(
            self.allowed_control_security_subject_ids
        ):
            raise TaskEvaluationAuthorityError(
                "task evaluation spawn security waiver exceeds exact control authority"
            )
        if denylist.scope_id != current.scope_id:
            raise TaskEvaluationAuthorityError(
                "task evaluation spawn denylist uses another scope authority"
            )
        required_subjects = {
            self.reservation_id,
            self.request_id,
            self.invocation_allocation.evaluation_case_id,
            self.invocation_allocation.evaluation_leg_id,
            current.observation_id,
            *current.validation_closure_ids,
        }
        if current.publication_id is not None:
            required_subjects.add(current.publication_id)
        for observation in self.task_adapter_trust_observations:
            required_subjects.update(
                {
                    observation.observation_id,
                    observation.task_adapter_manifest_id,
                    observation.verification_receipt_id,
                    observation.verifier_authority_subject_id,
                    *observation.dependency_ids,
                }
            )
        if not required_subjects.issubset(self.security_subject_ids):
            raise TaskEvaluationAuthorityError(
                "task evaluation spawn fence omits mandatory security subjects"
            )

    @property
    def security_subject_ids(self) -> tuple[str, ...]:
        return self.security_denylist_observation.checked_subject_ids
