"""Persistent fresh-authority contracts shared by replay coordination and journal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import (
    ExpertSourceReplayExecutionReservation,
    StrictContract,
)
from kapso.cross_run.expert.replay_protocol_contracts import (
    ExpertSourceReplayInvocationAllocation,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)

if TYPE_CHECKING:
    from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest

_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GITHUB_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ExpertSourceReplayFreshAuthorityError(ValueError):
    """A spawn lacks exact current external authority."""


def require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if (not values and not allow_empty) or values != tuple(sorted(set(values))):
        raise ExpertSourceReplayFreshAuthorityError(f"{name} must be sorted and unique")
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class SourceReplayCurrentReleaseObservation(StrictContract):
    observation_id: str
    scope_id: str
    release_id: str
    publication_id: str
    repository_full_name: str
    repository_node_id: str
    current_pointer_digest: str
    current_pointer_commit_sha: str
    validation_closure_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "source-replay-current-release-observation"
    IDENTITY_FIELD: ClassVar[str] = "observation_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "source replay current scope_id")
        for value, namespace, name in (
            (self.release_id, "expert-base-release", "release_id"),
            (self.publication_id, "github-publication", "publication_id"),
        ):
            require_content_id(value, f"source replay current {name}")
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayFreshAuthorityError(
                    f"source replay current {name} uses the wrong namespace"
                )
        if _GITHUB_REPOSITORY_PATTERN.fullmatch(self.repository_full_name) is None:
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay current repository identity is invalid"
            )
        require_identifier(
            self.repository_node_id,
            "source replay current repository_node_id",
        )
        if _SHA256_DIGEST_PATTERN.fullmatch(self.current_pointer_digest) is None:
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay current pointer digest is invalid"
            )
        if re.fullmatch(r"[0-9a-f]{40}", self.current_pointer_commit_sha) is None:
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay current pointer commit is invalid"
            )
        require_sorted_content_ids(
            self.validation_closure_ids,
            "source replay current validation closure",
            allow_empty=True,
        )


@dataclass(frozen=True)
class SourceReplaySpawnAuthorityFence(StrictContract):
    fence_id: str
    reservation_id: str
    execution_request_id: str
    authorization_transition_id: str
    authorization_state_id: str
    candidate_id: str
    scope_id: str
    scope_contract_id: str
    expected_current_release_id: str
    invocation_allocation: ExpertSourceReplayInvocationAllocation
    current_release_observation: SourceReplayCurrentReleaseObservation
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...]
    security_denylist_observation: SecurityDenylistObservation

    CONTENT_NAMESPACE: ClassVar[str] = "source-replay-spawn-authority-fence"
    IDENTITY_FIELD: ClassVar[str] = "fence_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "source replay spawn fence scope_id")
        for value, namespace, name in (
            (
                self.reservation_id,
                "expert-source-replay-execution-reservation",
                "reservation_id",
            ),
            (
                self.execution_request_id,
                "expert-source-replay-execution-request",
                "execution_request_id",
            ),
            (
                self.authorization_transition_id,
                "expert-validation-transition",
                "authorization_transition_id",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "authorization_state_id",
            ),
            (self.candidate_id, "expert-candidate", "candidate_id"),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "scope_contract_id",
            ),
            (
                self.expected_current_release_id,
                "expert-base-release",
                "expected_current_release_id",
            ),
        ):
            require_content_id(value, f"source replay spawn fence {name}")
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayFreshAuthorityError(
                    f"source replay spawn fence {name} uses the wrong namespace"
                )
        if self.invocation_allocation.reservation_id != self.reservation_id:
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn allocation uses another reservation"
            )
        current = self.current_release_observation
        if (
            current.scope_id != self.scope_id
            or current.release_id != self.expected_current_release_id
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn release differs from expected CURRENT"
            )
        observation_ids = tuple(
            observation.observation_id
            for observation in self.task_adapter_trust_observations
        )
        if not observation_ids or observation_ids != tuple(
            sorted(set(observation_ids))
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn adapter observations must be sorted and unique"
            )
        if self.security_denylist_observation.matched_revocations:
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn authority contains denied subjects"
            )
        denylist = self.security_denylist_observation
        if (
            denylist.scope_id != self.scope_id
            or denylist.scope_contract_id != self.scope_contract_id
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn denylist uses another scope authority"
            )
        required_security_subjects = {
            self.reservation_id,
            self.execution_request_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.candidate_id,
            self.scope_contract_id,
            self.expected_current_release_id,
            self.invocation_allocation.execution_case_id,
            self.invocation_allocation.execution_leg_id,
            current.observation_id,
            current.publication_id,
            *current.validation_closure_ids,
        }
        for observation in self.task_adapter_trust_observations:
            required_security_subjects.update(
                {
                    observation.observation_id,
                    observation.task_adapter_manifest_id,
                    observation.verification_receipt_id,
                    observation.verifier_authority_subject_id,
                    *observation.dependency_ids,
                }
            )
        if not required_security_subjects.issubset(self.security_subject_ids):
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn fence omits mandatory security subjects"
            )

    @property
    def security_subject_ids(self) -> tuple[str, ...]:
        return self.security_denylist_observation.checked_subject_ids


def source_replay_task_adapter_trust_observations(
    prepared: PreparedExpertSourceReplayRequest,
) -> tuple[TaskAdapterTrustObservation, ...]:
    adapters = {
        (
            item.task_adapter.manifest.task_adapter_manifest_id,
            item.task_adapter.verification_receipt.verification_receipt_id,
        ): item.task_adapter
        for item in prepared.cases
    }
    return tuple(
        sorted(
            (
                TaskAdapterTrustObservation.mint(
                    task_adapter_manifest_id=manifest_id,
                    verification_receipt_id=receipt_id,
                    verifier_id=adapter.verification_receipt.verifier_id,
                    verifier_version=adapter.verification_receipt.verifier_version,
                    dependency_ids=adapter.dependency_ids,
                )
                for (manifest_id, receipt_id), adapter in adapters.items()
            ),
            key=lambda observation: observation.observation_id,
        )
    )


def source_replay_spawn_security_subject_ids(
    prepared: PreparedExpertSourceReplayRequest,
    reservation: ExpertSourceReplayExecutionReservation,
    current: SourceReplayCurrentReleaseObservation,
    adapter_observations: tuple[TaskAdapterTrustObservation, ...],
) -> tuple[str, ...]:
    request = prepared.request
    candidate = prepared.candidate.manifest
    source_base_release = prepared.source_base.release_manifest
    subject_ids = {
        reservation.reservation_id,
        *reservation.exact_dependency_ids,
        request.execution_request_id,
        *request.exact_dependency_ids,
        current.observation_id,
        current.publication_id,
        *current.validation_closure_ids,
        *source_base_release.consumed_dependency_ids,
        *candidate.source_dependency_ids,
        *candidate.ancestor_candidate_ids,
        candidate.sanitation_report_id,
    }
    for observation in adapter_observations:
        subject_ids.update(
            {
                observation.task_adapter_manifest_id,
                observation.verification_receipt_id,
                observation.observation_id,
                observation.verifier_authority_subject_id,
                *observation.dependency_ids,
            }
        )
    ordered = tuple(sorted(subject_ids))
    require_sorted_content_ids(
        ordered,
        "source replay spawn security subjects",
    )
    return ordered
