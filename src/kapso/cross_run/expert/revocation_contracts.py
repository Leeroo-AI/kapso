"""Authenticated emergency revocation evidence for immutable expert releases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import (
    normalize_utc_timestamp,
    require_content_id,
)
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    ExpertCandidateValidationState,
    ExpertPromotionState,
    ExpertValidationAttempt,
    MissingReferenceError,
    StrictContract,
)
from kapso.cross_run.expert.release_contracts import (
    ExpertReleaseActivationReceipt,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)


class ExpertReleaseRevocationContractError(ValueError):
    """An emergency release-revocation closure is incomplete or inconsistent."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertReleaseRevocationContractError(f"{name} uses the wrong namespace")


def expert_release_revocation_security_subject_ids(
    *,
    authorization_transition_id: str,
    released_state: ExpertCandidateValidationState,
    validation_attempt: ExpertValidationAttempt,
    activation_receipt: ExpertReleaseActivationReceipt,
    release_manifest: ExpertBaseReleaseManifest,
) -> tuple[str, ...]:
    """Project the exact proof-consumption closure of one activated release."""

    if (
        type(released_state) is not ExpertCandidateValidationState
        or type(validation_attempt) is not ExpertValidationAttempt
        or type(activation_receipt) is not ExpertReleaseActivationReceipt
        or type(release_manifest) is not ExpertBaseReleaseManifest
        or released_state.promotion_state is not ExpertPromotionState.RELEASED
        or released_state.validation_attempt_id
        != validation_attempt.validation_attempt_id
        or released_state.candidate_id != validation_attempt.candidate_id
        or released_state.candidate_tree_hash != validation_attempt.candidate_tree_hash
        or activation_receipt.candidate_id != validation_attempt.candidate_id
        or released_state.predecessor_state_id != activation_receipt.approval_state_id
        or activation_receipt.activation_receipt_id
        not in released_state.terminal_evidence_ids
        or release_manifest.release_id != activation_receipt.release_id
        or release_manifest.candidate_id != validation_attempt.candidate_id
        or release_manifest.candidate_tree_hash
        != validation_attempt.candidate_tree_hash
        or release_manifest.validation_attempt_id
        != validation_attempt.validation_attempt_id
    ):
        raise ExpertReleaseRevocationContractError(
            "release revocation inputs do not describe one activated release"
        )
    _require_namespaced_id(
        authorization_transition_id,
        "expert-validation-transition",
        "revocation authorization transition",
    )
    subjects = {
        authorization_transition_id,
        released_state.validation_state_id,
        validation_attempt.validation_attempt_id,
        activation_receipt.activation_receipt_id,
        activation_receipt.release_id,
        validation_attempt.candidate_id,
        *activation_receipt.consumed_dependency_ids,
        *release_manifest.consumed_dependency_ids,
    }
    return tuple(sorted(subjects))


@dataclass(frozen=True)
class ExpertReleaseRevocationReceipt(StrictContract):
    revocation_receipt_id: str
    release_id: str
    candidate_id: str
    candidate_tree_hash: str
    validation_attempt_id: str
    authorization_transition_id: str
    authorization_state_id: str
    activation_receipt_id: str
    security_denylist_observation: SecurityDenylistObservation
    revoked_at: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-release-revocation-receipt"
    IDENTITY_FIELD: ClassVar[str] = "revocation_receipt_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.release_id, "expert-base-release", "revoked release"),
            (self.candidate_id, "expert-candidate", "revoked candidate"),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "revocation validation attempt",
            ),
            (
                self.authorization_transition_id,
                "expert-validation-transition",
                "revocation authorization transition",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "revocation authorization state",
            ),
            (
                self.activation_receipt_id,
                "expert-release-activation-receipt",
                "revocation activation receipt",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.candidate_tree_hash) is None:
            raise ExpertReleaseRevocationContractError(
                "revocation candidate tree hash is invalid"
            )
        observation = self.security_denylist_observation
        if (
            type(observation) is not SecurityDenylistObservation
            or not observation.matched_revocations
            or self.release_id not in observation.checked_subject_ids
            or self.candidate_id not in observation.checked_subject_ids
            or self.activation_receipt_id not in observation.checked_subject_ids
        ):
            raise ExpertReleaseRevocationContractError(
                "revocation receipt lacks an exact emergency-denylist match"
            )
        normalize_utc_timestamp(self.revoked_at, "release revoked_at")
        if self.exact_dependency_ids != tuple(sorted(set(self.exact_dependency_ids))):
            raise ExpertReleaseRevocationContractError(
                "revocation dependencies must be sorted and unique"
            )
        for dependency_id in self.exact_dependency_ids:
            require_content_id(dependency_id, "revocation dependency")
        required = {
            self.release_id,
            self.candidate_id,
            self.validation_attempt_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.activation_receipt_id,
            observation.observation_id,
            observation.scope_contract_id,
            observation.snapshot_id,
            observation.publication_id,
            *observation.checked_subject_ids,
            *(
                revocation.revocation_id
                for revocation in observation.matched_revocations
            ),
            *(revocation.subject_id for revocation in observation.matched_revocations),
            *(
                evidence_id
                for revocation in observation.matched_revocations
                for evidence_id in revocation.evidence_ids
            ),
        }
        if set(self.exact_dependency_ids) != required:
            raise MissingReferenceError(
                "release revocation dependency closure is not exact"
            )
