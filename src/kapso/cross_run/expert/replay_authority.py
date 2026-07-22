"""Fresh external authority fencing for expert source-replay execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Protocol

from kapso.cross_run.canonical import (
    content_id,
    require_content_id,
    require_identifier,
)
from kapso.cross_run.contracts import (
    ExpertSourceReplayExecutionReservation,
    StrictContract,
)
from kapso.cross_run.expert.replay_protocol import (
    TaskEvaluatorInvocationAllocation,
)
from kapso.cross_run.expert.replay_execution_store import (
    ExpertSourceReplayExecutionStore,
    SourceReplayInvocationAllocationPermit,
)
from kapso.cross_run.expert.replay_request import (
    MaterializedExpertSourceReplayCase,
    PreparedExpertSourceReplayRequest,
)
from kapso.cross_run.expert.validation_store import (
    ExpertSourceReplayReservationSnapshot,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)

_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GITHUB_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ExpertSourceReplayFreshAuthorityError(ValueError):
    """A spawn lacks exact current external authority."""


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
        _require_sorted_content_ids(
            self.validation_closure_ids,
            "source replay current validation closure",
            allow_empty=True,
        )


@dataclass(frozen=True)
class SourceReplayTaskAdapterTrustObservation(StrictContract):
    observation_id: str
    task_adapter_manifest_id: str
    verification_receipt_id: str
    verifier_id: str
    verifier_version: str
    dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "source-replay-task-adapter-trust-observation"
    IDENTITY_FIELD: ClassVar[str] = "observation_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.task_adapter_manifest_id,
                "task-adapter-manifest",
                "task_adapter_manifest_id",
            ),
            (
                self.verification_receipt_id,
                "task-adapter-verification-receipt",
                "verification_receipt_id",
            ),
        ):
            require_content_id(value, f"source replay adapter trust {name}")
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayFreshAuthorityError(
                    f"source replay adapter trust {name} uses the wrong namespace"
                )
        for value, name in (
            (self.verifier_id, "verifier_id"),
            (self.verifier_version, "verifier_version"),
        ):
            require_identifier(value, f"source replay adapter trust {name}")
        _require_sorted_content_ids(
            self.dependency_ids,
            "source replay adapter trust dependencies",
        )
        if self.verification_receipt_id not in self.dependency_ids:
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay adapter trust omits its verification receipt"
            )

    @property
    def verifier_authority_subject_id(self) -> str:
        return content_id(
            "task-adapter-verifier-authority",
            {
                "verifier_id": self.verifier_id,
                "verifier_version": self.verifier_version,
            },
        )


@dataclass(frozen=True)
class SourceReplaySecurityDenylistObservation(StrictContract):
    observation_id: str
    snapshot_id: str
    generation: int
    checked_subject_ids: tuple[str, ...]
    denied_subject_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "source-replay-security-denylist-observation"
    IDENTITY_FIELD: ClassVar[str] = "observation_id"

    def _validate(self) -> None:
        require_content_id(self.snapshot_id, "source replay denylist snapshot_id")
        if self.snapshot_id.split(":sha256:", 1)[0] != "security-denylist-snapshot":
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay denylist snapshot uses the wrong namespace"
            )
        if type(self.generation) is not int or self.generation < 0:
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay denylist generation must be non-negative"
            )
        _require_sorted_content_ids(
            self.checked_subject_ids,
            "source replay denylist checked subjects",
        )
        _require_sorted_content_ids(
            self.denied_subject_ids,
            "source replay denylist denied subjects",
            allow_empty=True,
        )
        if not set(self.denied_subject_ids).issubset(self.checked_subject_ids):
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay denied subjects were not checked"
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
    expected_parent_release_id: str
    invocation_allocation: TaskEvaluatorInvocationAllocation
    security_subject_ids: tuple[str, ...]
    current_release_observation: SourceReplayCurrentReleaseObservation
    task_adapter_trust_observations: tuple[SourceReplayTaskAdapterTrustObservation, ...]
    security_denylist_observation: SourceReplaySecurityDenylistObservation

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
                self.expected_parent_release_id,
                "expert-base-release",
                "expected_parent_release_id",
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
        _require_sorted_content_ids(
            self.security_subject_ids,
            "source replay spawn security subjects",
        )
        if (
            self.security_subject_ids
            != self.security_denylist_observation.checked_subject_ids
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn security subjects differ from the denylist"
            )
        current = self.current_release_observation
        if (
            current.scope_id != self.scope_id
            or current.release_id != self.expected_parent_release_id
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn current release differs from its parent"
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
        if self.security_denylist_observation.denied_subject_ids:
            raise ExpertSourceReplayFreshAuthorityError(
                "source replay spawn authority contains denied subjects"
            )
        required_security_subjects = {
            self.reservation_id,
            self.execution_request_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.candidate_id,
            self.scope_contract_id,
            self.expected_parent_release_id,
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


class ExpertSourceReplayReservationAuthority(Protocol):
    """Reopen the exact current local reservation under a short shared lock."""

    def reopen_source_replay_reservation(
        self,
        *,
        reservation_id: str,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> ExpertSourceReplayReservationSnapshot: ...


class ExpertSourceReplayCurrentReleaseAuthority(Protocol):
    """Fetch and authenticate the live expert CURRENT without a stale fallback."""

    def current_release_observation(
        self,
        scope_id: str,
    ) -> SourceReplayCurrentReleaseObservation: ...


class ExpertSourceReplaySecurityDenylistAuthority(Protocol):
    """Authenticate current non-rollback denylist state for the exact subjects."""

    def observe_exact(
        self,
        checked_subject_ids: tuple[str, ...],
    ) -> SourceReplaySecurityDenylistObservation: ...


class ExpertSourceReplayFreshAuthorityCoordinator:
    """Build one moment-in-time spawn fence between exact local reopens."""

    def __init__(
        self,
        reservation_authority: ExpertSourceReplayReservationAuthority,
        execution_store: ExpertSourceReplayExecutionStore,
        current_release_authority: ExpertSourceReplayCurrentReleaseAuthority,
        task_adapter_authority: VerifiedTaskAdapterProvider,
        security_denylist_authority: ExpertSourceReplaySecurityDenylistAuthority,
    ) -> None:
        self.reservation_authority = reservation_authority
        if not isinstance(execution_store, ExpertSourceReplayExecutionStore):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn authority requires its canonical execution store"
            )
        self.execution_store = execution_store
        self.current_release_authority = current_release_authority
        self.task_adapter_authority = task_adapter_authority
        self.security_denylist_authority = security_denylist_authority

    def authorize_spawn(
        self,
        *,
        prepared_request: PreparedExpertSourceReplayRequest,
        reservation_id: str,
        invocation_permit: SourceReplayInvocationAllocationPermit,
    ) -> SourceReplaySpawnAuthorityFence:
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn authority requires a prepared request"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            parent=prepared_request.parent,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        if not isinstance(
            invocation_permit,
            SourceReplayInvocationAllocationPermit,
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn authority requires a live invocation allocation permit"
            )
        invocation_allocation = invocation_permit.require_current_allocation(
            self.execution_store
        )
        first = self.reservation_authority.reopen_source_replay_reservation(
            reservation_id=reservation_id,
            prepared_request=prepared,
        )
        request = prepared.request
        if (
            not isinstance(first, ExpertSourceReplayReservationSnapshot)
            or first.request != request
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn reopen differs from the prepared request"
            )
        reservation = first.reservation
        if invocation_allocation.reservation_id != reservation.reservation_id:
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn allocation differs from the reopened reservation"
            )
        materialized_case = _allocated_materialized_case(
            prepared,
            invocation_allocation,
        )
        scope_id = prepared.parent.release_manifest.scope_id
        if (
            prepared.parent.release_manifest.scope_contract_id
            != request.scope_contract_id
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn parent uses another scope contract"
            )
        current_observation = (
            self.current_release_authority.current_release_observation(scope_id)
        )
        if (
            not isinstance(
                current_observation,
                SourceReplayCurrentReleaseObservation,
            )
            or current_observation.scope_id != scope_id
            or current_observation.release_id != request.parent_release_id
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn current release differs from reserved authority"
            )
        adapter_observations = self._reverify_adapters(prepared)
        if not any(
            observation.task_adapter_manifest_id
            == materialized_case.task_adapter.manifest.task_adapter_manifest_id
            and observation.verification_receipt_id
            == materialized_case.task_adapter.verification_receipt.verification_receipt_id
            for observation in adapter_observations
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn allocated case lacks adapter trust authority"
            )
        checked_subject_ids = _spawn_security_subject_ids(
            prepared,
            reservation,
            current_observation,
            adapter_observations,
        )
        denylist_observation = self.security_denylist_authority.observe_exact(
            checked_subject_ids
        )
        if (
            not isinstance(
                denylist_observation,
                SourceReplaySecurityDenylistObservation,
            )
            or denylist_observation.checked_subject_ids != checked_subject_ids
            or denylist_observation.denied_subject_ids
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn denylist authority rejected the exact dependency closure"
            )
        second = self.reservation_authority.reopen_source_replay_reservation(
            reservation_id=reservation_id,
            prepared_request=prepared,
        )
        if (
            not isinstance(second, ExpertSourceReplayReservationSnapshot)
            or second != first
        ):
            raise ExpertSourceReplayFreshAuthorityError(
                "fresh spawn reservation changed during external checks"
            )
        return SourceReplaySpawnAuthorityFence.mint(
            reservation_id=reservation.reservation_id,
            execution_request_id=request.execution_request_id,
            authorization_transition_id=reservation.authorization_transition_id,
            authorization_state_id=reservation.authorization_state_id,
            candidate_id=reservation.candidate_id,
            scope_id=scope_id,
            scope_contract_id=request.scope_contract_id,
            expected_parent_release_id=request.parent_release_id,
            invocation_allocation=invocation_allocation,
            security_subject_ids=checked_subject_ids,
            current_release_observation=current_observation,
            task_adapter_trust_observations=adapter_observations,
            security_denylist_observation=denylist_observation,
        )

    def _reverify_adapters(
        self,
        prepared: PreparedExpertSourceReplayRequest,
    ) -> tuple[SourceReplayTaskAdapterTrustObservation, ...]:
        adapters = {
            (
                item.task_adapter.manifest.task_adapter_manifest_id,
                item.task_adapter.verification_receipt.verification_receipt_id,
            ): item.task_adapter
            for item in prepared.cases
        }
        observations = []
        for (manifest_id, receipt_id), expected in sorted(adapters.items()):
            observed = self.task_adapter_authority.resolve_exact(
                task_adapter_manifest_id=manifest_id,
                verification_receipt_id=receipt_id,
            )
            if not isinstance(observed, VerifiedTaskAdapter) or observed != expected:
                raise ExpertSourceReplayFreshAuthorityError(
                    "fresh spawn adapter differs from the prepared byte closure"
                )
            receipt = observed.verification_receipt
            observations.append(
                SourceReplayTaskAdapterTrustObservation.mint(
                    task_adapter_manifest_id=manifest_id,
                    verification_receipt_id=receipt_id,
                    verifier_id=receipt.verifier_id,
                    verifier_version=receipt.verifier_version,
                    dependency_ids=observed.dependency_ids,
                )
            )
        return tuple(sorted(observations, key=lambda item: item.observation_id))


def _allocated_materialized_case(
    prepared: PreparedExpertSourceReplayRequest,
    allocation: TaskEvaluatorInvocationAllocation,
) -> MaterializedExpertSourceReplayCase:
    matches = tuple(
        item
        for item in prepared.cases
        if item.request_case.execution_case_id == allocation.execution_case_id
    )
    if len(matches) != 1:
        raise ExpertSourceReplayFreshAuthorityError(
            "fresh spawn allocation names no unique prepared case"
        )
    request_case = matches[0].request_case
    if allocation.execution_leg_id not in {
        request_case.control_leg.execution_leg_id,
        request_case.candidate_leg.execution_leg_id,
    }:
        raise ExpertSourceReplayFreshAuthorityError(
            "fresh spawn allocation names no prepared leg"
        )
    return matches[0]


def _spawn_security_subject_ids(
    prepared: PreparedExpertSourceReplayRequest,
    reservation: ExpertSourceReplayExecutionReservation,
    current: SourceReplayCurrentReleaseObservation,
    adapter_observations: tuple[SourceReplayTaskAdapterTrustObservation, ...],
) -> tuple[str, ...]:
    request = prepared.request
    candidate = prepared.candidate.manifest
    parent = prepared.parent.release_manifest
    subject_ids = {
        reservation.reservation_id,
        *reservation.exact_dependency_ids,
        request.execution_request_id,
        *request.exact_dependency_ids,
        current.observation_id,
        current.publication_id,
        *current.validation_closure_ids,
        *parent.dependency_closure_ids,
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
    _require_sorted_content_ids(
        ordered,
        "source replay spawn security subjects",
    )
    return ordered


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if (not values and not allow_empty) or values != tuple(sorted(set(values))):
        raise ExpertSourceReplayFreshAuthorityError(f"{name} must be sorted and unique")
    for value in values:
        require_content_id(value, name)
