"""Sealed fresh authority for terminal expert publication eligibility."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import (
    ExpertAcceptedStageResultRef,
    ExpertValidationStage,
    StrictContract,
)
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
    ExpertReleaseMatrixPromotionDecision,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExpertPublicationEligibilityContractError(ValueError):
    """Publication eligibility authority or evidence is inconsistent."""


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertPublicationEligibilityContractError(
            f"{name} uses the wrong namespace"
        )


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ExpertPublicationEligibilityContractError(
            f"{name} must be a sha256 digest"
        )


def _require_sorted_content_ids(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ExpertPublicationEligibilityContractError(
            f"{name} must be non-empty, sorted, and unique"
        )
    for value in values:
        require_content_id(value, name)


class ExpertCandidateReleaseUseOutcome(str, Enum):
    """Publication availability derived from one exact current policy read."""

    CLEARED = "cleared"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ExpertCandidateReleaseUseDecision(StrictContract):
    """One candidate's exact release-use eligibility under current policy."""

    release_use_decision_id: str
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    candidate_commit_record_id: str
    scope_contract_id: str
    scope_id: str
    release_matrix_stage_result_id: str
    promotion_decision_id: str
    policy_observation: ExpertReleaseUsePolicyObservation
    outcome: ExpertCandidateReleaseUseOutcome
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-release-use-decision"
    IDENTITY_FIELD: ClassVar[str] = "release_use_decision_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "release-use decision validation attempt",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "release-use decision candidate",
            ),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "release-use decision candidate commit",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "release-use decision scope contract",
            ),
            (
                self.release_matrix_stage_result_id,
                "expert-release-matrix-stage-result",
                "release-use decision release matrix result",
            ),
            (
                self.promotion_decision_id,
                "expert-release-matrix-promotion-decision",
                "release-use decision promotion decision",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        _require_digest(self.candidate_tree_hash, "release-use decision candidate tree")
        require_identifier(self.scope_id, "release-use decision scope")
        observation = self.policy_observation
        if (
            type(observation) is not ExpertReleaseUsePolicyObservation
            or observation.scope_id != self.scope_id
            or observation.scope_contract_id != self.scope_contract_id
        ):
            raise ExpertPublicationEligibilityContractError(
                "release-use decision policy observation uses another scope"
            )
        expected_outcome = (
            ExpertCandidateReleaseUseOutcome.BLOCKED
            if observation.matched_revocations
            else ExpertCandidateReleaseUseOutcome.CLEARED
        )
        if self.outcome is not expected_outcome:
            raise ExpertPublicationEligibilityContractError(
                "release-use decision outcome differs from current policy"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "release-use decision exact dependencies",
        )
        expected_dependencies = {
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.release_matrix_stage_result_id,
            self.promotion_decision_id,
            observation.observation_id,
            observation.knowledge_snapshot_id,
            observation.knowledge_publication_id,
            *observation.checked_release_ids,
        }
        for revocation in observation.matched_revocations:
            expected_dependencies.update(
                {
                    revocation.revocation_id,
                    revocation.release_id,
                    revocation.release_publication_id,
                    revocation.release_activation_witness_id,
                    *revocation.exact_evidence_refs,
                }
            )
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertPublicationEligibilityContractError(
                "release-use decision dependency closure is not exact"
            )


@dataclass(frozen=True)
class ExpertPublicationEligibilityAuthorityFence(StrictContract):
    """Fresh CURRENT, adapter, and denylist authority for an approved decision."""

    fence_id: str
    release_matrix_acceptance_transition_id: str
    release_matrix_acceptance_state_id: str
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    candidate_commit_record_id: str
    scope_contract_id: str
    scope_id: str
    expected_current_release_id: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    release_matrix_stage_result_id: str
    promotion_decision_id: str
    release_use_decision_id: str
    security_subject_ids: tuple[str, ...]
    current_release_observation: TaskEvaluationCurrentReleaseObservation
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...]
    security_denylist_observation: SecurityDenylistObservation

    CONTENT_NAMESPACE: ClassVar[str] = "expert-publication-eligibility-authority-fence"
    IDENTITY_FIELD: ClassVar[str] = "fence_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.release_matrix_acceptance_transition_id,
                "expert-validation-transition",
                "publication eligibility matrix acceptance transition",
            ),
            (
                self.release_matrix_acceptance_state_id,
                "expert-candidate-validation-state",
                "publication eligibility matrix acceptance state",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "publication eligibility validation attempt",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "publication eligibility candidate",
            ),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "publication eligibility candidate commit",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "publication eligibility scope contract",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "publication eligibility validation policy",
            ),
            (
                self.release_matrix_stage_result_id,
                "expert-release-matrix-stage-result",
                "publication eligibility release matrix stage result",
            ),
            (
                self.promotion_decision_id,
                "expert-release-matrix-promotion-decision",
                "publication eligibility promotion decision",
            ),
            (
                self.release_use_decision_id,
                "expert-candidate-release-use-decision",
                "publication eligibility release-use decision",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if self.expected_current_release_id is not None:
            _require_namespaced_id(
                self.expected_current_release_id,
                "expert-base-release",
                "publication eligibility expected CURRENT release",
            )
        require_identifier(self.scope_id, "publication eligibility scope")
        _require_digest(
            self.candidate_tree_hash,
            "publication eligibility candidate tree",
        )
        _require_digest(
            self.configuration_fingerprint,
            "publication eligibility configuration fingerprint",
        )
        _require_sorted_content_ids(
            self.security_subject_ids,
            "publication eligibility security subjects",
        )
        current = self.current_release_observation
        if (
            current.scope_id != self.scope_id
            or current.release_id != self.expected_current_release_id
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication eligibility CURRENT authority differs from expectation"
            )
        observation_ids = tuple(
            observation.observation_id
            for observation in self.task_adapter_trust_observations
        )
        if not observation_ids or observation_ids != tuple(
            sorted(set(observation_ids))
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication eligibility adapter observations must be canonical"
            )
        denylist = self.security_denylist_observation
        if (
            denylist.scope_id != self.scope_id
            or denylist.scope_contract_id != self.scope_contract_id
            or denylist.checked_subject_ids != self.security_subject_ids
            or denylist.matched_revocations
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication eligibility denylist observation differs from exact authority"
            )
        required_subjects = {
            self.release_matrix_acceptance_transition_id,
            self.release_matrix_acceptance_state_id,
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.validation_policy_id,
            self.release_matrix_stage_result_id,
            self.promotion_decision_id,
            current.observation_id,
            *current.validation_closure_ids,
        }
        if self.expected_current_release_id is not None:
            required_subjects.add(self.expected_current_release_id)
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
            raise ExpertPublicationEligibilityContractError(
                "publication eligibility security authority omits mandatory subjects"
            )

    @property
    def exact_dependency_ids(self) -> tuple[str, ...]:
        current = self.current_release_observation
        denylist = self.security_denylist_observation
        dependencies = {
            self.release_matrix_acceptance_transition_id,
            self.release_matrix_acceptance_state_id,
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.validation_policy_id,
            self.release_matrix_stage_result_id,
            self.promotion_decision_id,
            self.release_use_decision_id,
            *self.security_subject_ids,
            current.observation_id,
            *current.validation_closure_ids,
            denylist.observation_id,
            denylist.snapshot_id,
            denylist.publication_id,
        }
        if self.expected_current_release_id is not None:
            dependencies.add(self.expected_current_release_id)
        if current.publication_id is not None:
            dependencies.add(current.publication_id)
        for observation in self.task_adapter_trust_observations:
            dependencies.update(
                {
                    observation.observation_id,
                    observation.task_adapter_manifest_id,
                    observation.verification_receipt_id,
                    observation.verifier_authority_subject_id,
                    *observation.dependency_ids,
                }
            )
        return tuple(sorted(dependencies))


@dataclass(frozen=True)
class ExpertPublicationEligibilityStageResultRecord(StrictContract):
    """Terminal Pareto outcome sealed to the accepted release-matrix head."""

    stage_result_record_id: str
    release_matrix_acceptance_transition_id: str
    release_matrix_acceptance_state_id: str
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    candidate_commit_record_id: str
    scope_contract_id: str
    scope_id: str
    expected_current_release_id: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    accepted_stage_results: tuple[ExpertAcceptedStageResultRef, ...]
    promotion_decision: ExpertReleaseMatrixPromotionDecision
    release_use_decision: ExpertCandidateReleaseUseDecision | None
    publication_authority_fence: ExpertPublicationEligibilityAuthorityFence | None
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-publication-eligibility-stage-result"
    IDENTITY_FIELD: ClassVar[str] = "stage_result_record_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.release_matrix_acceptance_transition_id,
                "expert-validation-transition",
                "publication result matrix acceptance transition",
            ),
            (
                self.release_matrix_acceptance_state_id,
                "expert-candidate-validation-state",
                "publication result matrix acceptance state",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "publication result validation attempt",
            ),
            (
                self.candidate_id,
                "expert-candidate",
                "publication result candidate",
            ),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "publication result candidate commit",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "publication result scope contract",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "publication result validation policy",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if self.expected_current_release_id is not None:
            _require_namespaced_id(
                self.expected_current_release_id,
                "expert-base-release",
                "publication result expected CURRENT release",
            )
        require_identifier(self.scope_id, "publication result scope")
        _require_digest(
            self.candidate_tree_hash,
            "publication result candidate tree",
        )
        _require_digest(
            self.configuration_fingerprint,
            "publication result configuration fingerprint",
        )
        self._validate_accepted_release_matrix_prefix()
        decision = self.promotion_decision
        if (
            decision.validation_attempt_id != self.validation_attempt_id
            or decision.validation_policy_id != self.validation_policy_id
            or decision.configuration_fingerprint != self.configuration_fingerprint
            or self.accepted_stage_results[-1].stage_result_record_id
            != decision.release_matrix_stage_result_id
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication result decision differs from accepted matrix authority"
            )
        if (decision.mode is ExpertReleaseMatrixMode.BOOTSTRAP) != (
            self.expected_current_release_id is None
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication result CURRENT expectation differs from matrix mode"
            )
        fence = self.publication_authority_fence
        approved = decision.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
        release_use = self.release_use_decision
        if approved != (release_use is not None):
            raise ExpertPublicationEligibilityContractError(
                "only an approved decision requires release-use authority"
            )
        cleared = (
            release_use is not None
            and release_use.outcome is ExpertCandidateReleaseUseOutcome.CLEARED
        )
        if cleared != (fence is not None):
            raise ExpertPublicationEligibilityContractError(
                "only a release-use-cleared decision permits publication authority"
            )
        if release_use is not None:
            self._validate_release_use_decision(release_use)
        if fence is not None:
            self._validate_approved_fence(fence, release_use)
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "publication result exact dependencies",
        )
        expected_dependencies = {
            self.release_matrix_acceptance_transition_id,
            self.release_matrix_acceptance_state_id,
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.validation_policy_id,
            *(result.stage_result_record_id for result in self.accepted_stage_results),
            decision.promotion_decision_id,
            *decision.exact_dependency_ids,
        }
        if self.expected_current_release_id is not None:
            expected_dependencies.add(self.expected_current_release_id)
        if release_use is not None:
            expected_dependencies.update(
                {
                    release_use.release_use_decision_id,
                    *release_use.exact_dependency_ids,
                }
            )
        if fence is not None:
            expected_dependencies.update({fence.fence_id, *fence.exact_dependency_ids})
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ExpertPublicationEligibilityContractError(
                "publication result dependency closure is not exact"
            )

    def _validate_accepted_release_matrix_prefix(self) -> None:
        accepted = self.accepted_stage_results
        if not accepted or any(
            type(result) is not ExpertAcceptedStageResultRef for result in accepted
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication result requires an exact accepted stage prefix"
            )
        stages = tuple(result.stage for result in accepted)
        record_ids = tuple(result.stage_result_record_id for result in accepted)
        if (
            len(stages) != len(set(stages))
            or len(record_ids) != len(set(record_ids))
            or ExpertValidationStage.PUBLICATION_ELIGIBILITY in stages
            or stages[-1] is not ExpertValidationStage.RELEASE_MATRIX
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication result accepted prefix must end in one release matrix"
            )

    def _validate_release_use_decision(
        self,
        release_use: ExpertCandidateReleaseUseDecision,
    ) -> None:
        decision = self.promotion_decision
        if (
            type(release_use) is not ExpertCandidateReleaseUseDecision
            or release_use.validation_attempt_id != self.validation_attempt_id
            or release_use.candidate_id != self.candidate_id
            or release_use.candidate_tree_hash != self.candidate_tree_hash
            or release_use.candidate_commit_record_id != self.candidate_commit_record_id
            or release_use.scope_contract_id != self.scope_contract_id
            or release_use.scope_id != self.scope_id
            or release_use.release_matrix_stage_result_id
            != decision.release_matrix_stage_result_id
            or release_use.promotion_decision_id != decision.promotion_decision_id
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication result release-use decision differs from terminal authority"
            )

    def _validate_approved_fence(
        self,
        fence: ExpertPublicationEligibilityAuthorityFence,
        release_use: ExpertCandidateReleaseUseDecision,
    ) -> None:
        decision = self.promotion_decision
        if (
            type(fence) is not ExpertPublicationEligibilityAuthorityFence
            or fence.release_matrix_acceptance_transition_id
            != self.release_matrix_acceptance_transition_id
            or fence.release_matrix_acceptance_state_id
            != self.release_matrix_acceptance_state_id
            or fence.validation_attempt_id != self.validation_attempt_id
            or fence.candidate_id != self.candidate_id
            or fence.candidate_tree_hash != self.candidate_tree_hash
            or fence.candidate_commit_record_id != self.candidate_commit_record_id
            or fence.scope_contract_id != self.scope_contract_id
            or fence.scope_id != self.scope_id
            or fence.expected_current_release_id != self.expected_current_release_id
            or fence.validation_policy_id != self.validation_policy_id
            or fence.configuration_fingerprint != self.configuration_fingerprint
            or fence.release_matrix_stage_result_id
            != decision.release_matrix_stage_result_id
            or fence.promotion_decision_id != decision.promotion_decision_id
            or fence.release_use_decision_id != release_use.release_use_decision_id
        ):
            raise ExpertPublicationEligibilityContractError(
                "publication result fence differs from terminal authority"
            )
        required_security_subjects = {
            self.release_matrix_acceptance_transition_id,
            self.release_matrix_acceptance_state_id,
            self.validation_attempt_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.validation_policy_id,
            *(result.stage_result_record_id for result in self.accepted_stage_results),
            decision.promotion_decision_id,
            *decision.exact_dependency_ids,
        }
        if self.expected_current_release_id is not None:
            required_security_subjects.add(self.expected_current_release_id)
        if not required_security_subjects.issubset(fence.security_subject_ids):
            raise ExpertPublicationEligibilityContractError(
                "publication result fence omits exact decision authority"
            )

    @property
    def outcome(self) -> ExpertReleaseMatrixDecisionOutcome:
        return self.promotion_decision.outcome
