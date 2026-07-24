"""Fresh authority and one-shot publication for terminal expert promotion."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import (
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
    ExpertEvaluatorResultRecord,
    ExpertPromotionState,
    ExpertScopeContract,
    ExpertValidationAttempt,
)
from kapso.cross_run.expert.candidate_derivations import (
    ExpertAgentProposalDerivation,
    ExpertCompositionSourceProvenance,
    ExpertDeterministicCompositionDerivation,
    ExpertDeterministicRecoveryRestoreDerivation,
)
from kapso.cross_run.expert.composition_admission_contracts import (
    ExpertCompositionAdmissionFence,
)
from kapso.cross_run.expert.promotion import (
    decide_expert_release_matrix_promotion,
)
from kapso.cross_run.expert.promotion_authority_contracts import (
    ExpertCandidateReleaseUseDecision,
    ExpertCandidateReleaseUseOutcome,
    ExpertPublicationEligibilityAuthorityFence,
    ExpertPublicationEligibilityStageResultRecord,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
    ExpertReleaseMatrixPromotionDecision,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.expert.proposal_contract import ExpertCandidateAncestorInput
from kapso.cross_run.expert.recovery_candidate_contracts import (
    ExpertRecoveryCandidateAdmission,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayStageResultRecord,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewStageResultRecord,
)
from kapso.cross_run.expert.store import (
    StoredExpertCandidate,
    stored_candidate_admission_dependency_ids,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.validation_snapshots import (
    ExpertPublicationEligibilitySnapshot,
    ExpertValidationSnapshot,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    VerifiedTaskAdapterProvider,
)

if TYPE_CHECKING:
    from kapso.cross_run.expert.validation_store import (
        ExpertPublicationEligibilityStageCommitResult,
        ExpertValidationCommitResult,
        ExpertValidationStore,
    )


class ExpertPublicationEligibilityError(ValueError):
    """Terminal promotion lacks exact local or fresh external authority."""


class ExpertPublicationCurrentReleaseAuthority(Protocol):
    def observe_task_evaluation_current(
        self,
        scope_id: str,
    ) -> TaskEvaluationCurrentReleaseObservation: ...


class ExpertPublicationSecurityDenylistAuthority(Protocol):
    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SecurityDenylistObservation: ...


class ExpertPublicationReleaseUsePolicyAuthority(Protocol):
    def observe_exact(
        self,
        *,
        scope_contract: ExpertScopeContract,
        checked_release_ids: tuple[str, ...],
    ) -> ExpertReleaseUsePolicyObservation: ...


_PUBLICATION_ELIGIBILITY_EXECUTION_SEAL = object()


class ExpertPublicationEligibilityExecution:
    """One-shot process-local authority over one terminal promotion result."""

    __slots__ = (
        "_consumed",
        "_coordinator",
        "_owner_process_id",
        "_validation_store",
        "decision",
        "input_snapshot",
        "stage_result",
        "stored_candidate",
    )

    def __init__(
        self,
        seal: object,
        coordinator: ExpertPublicationEligibilityCoordinator,
        *,
        validation_store: ExpertValidationStore,
        input_snapshot: ExpertPublicationEligibilitySnapshot,
        stored_candidate: StoredExpertCandidate,
        decision: ExpertReleaseMatrixPromotionDecision,
        stage_result: ExpertPublicationEligibilityStageResultRecord,
    ) -> None:
        if seal is not _PUBLICATION_ELIGIBILITY_EXECUTION_SEAL:
            raise ExpertPublicationEligibilityError(
                "publication eligibility execution is not coordinator sealed"
            )
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_validation_store", validation_store)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_consumed", False)
        object.__setattr__(self, "input_snapshot", input_snapshot)
        object.__setattr__(self, "stored_candidate", stored_candidate)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "stage_result", stage_result)

    def __setattr__(self, name, value) -> None:
        raise ExpertPublicationEligibilityError(
            "publication eligibility execution is immutable"
        )

    def _require_bound(
        self,
        coordinator: object,
        validation_store: object,
    ) -> None:
        if (
            self._consumed
            or self._coordinator is not coordinator
            or self._validation_store is not validation_store
            or self._owner_process_id != os.getpid()
        ):
            raise ExpertPublicationEligibilityError(
                "publication eligibility execution is consumed or foreign"
            )

    def _consume(
        self,
        coordinator: object,
        validation_store: object,
    ) -> None:
        self._require_bound(coordinator, validation_store)
        object.__setattr__(self, "_consumed", True)


class ExpertPublicationEligibilityCoordinator:
    """Terminalize a Pareto decision under exact local and fresh remote authority."""

    def __init__(
        self,
        *,
        validation_store: ExpertValidationStore,
        current_release_authority: ExpertPublicationCurrentReleaseAuthority,
        task_adapter_authority: VerifiedTaskAdapterProvider,
        security_denylist_authority: ExpertPublicationSecurityDenylistAuthority,
        release_use_policy_authority: ExpertPublicationReleaseUsePolicyAuthority,
    ) -> None:
        reducer = validation_store.reducer
        if (
            reducer.current_release_provider is not current_release_authority
            or reducer.task_adapter_provider is not task_adapter_authority
        ):
            raise ExpertPublicationEligibilityError(
                "publication eligibility must share enrollment authorities"
            )
        self.validation_store = validation_store
        self.current_release_authority = current_release_authority
        self.task_adapter_authority = task_adapter_authority
        self.security_denylist_authority = security_denylist_authority
        self.release_use_policy_authority = release_use_policy_authority
        validation_store._bind_publication_eligibility_authority(self)

    def publish(
        self,
        *,
        candidate_id: str,
        release_matrix_stage_result_id: str,
    ) -> ExpertPublicationEligibilityStageCommitResult | ExpertValidationCommitResult:
        require_content_id(candidate_id, "publication eligibility candidate_id")
        require_content_id(
            release_matrix_stage_result_id,
            "publication eligibility release_matrix_stage_result_id",
        )
        replay, input_snapshot = (
            self.validation_store.reopen_or_replay_publication_eligibility(
                candidate_id=candidate_id,
                release_matrix_stage_result_id=release_matrix_stage_result_id,
            )
        )
        if replay is not None:
            return replay
        if input_snapshot is None:
            raise ExpertPublicationEligibilityError(
                "publication eligibility has no exact matrix input"
            )
        snapshot = input_snapshot.snapshot
        attempt = snapshot.latest_attempt
        if attempt is None:
            raise ExpertPublicationEligibilityError(
                "publication eligibility input has no validation attempt"
            )
        decision = decide_expert_release_matrix_promotion(
            stage_result=input_snapshot.release_matrix_stage_result,
            attempt=attempt,
            settings=self.validation_store.settings,
        )
        stored_candidate = self._reopen_candidate(input_snapshot)
        fence = None
        release_use_decision = None
        recovery_admission = publication_eligibility_recovery_admission(
            attempt=attempt,
            stored_candidate=stored_candidate,
        )
        allowed_control_security_subject_ids = (
            ()
            if recovery_admission is None
            else recovery_admission.allowed_control_security_subject_ids
        )
        if decision.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED:
            current_before = self._observe_current(stored_candidate)
            if current_before.release_id != attempt.expected_current_release_id:
                return self._invalidate_current_authority(input_snapshot)
            adapter_observations = self._reverify_adapters(input_snapshot)
            security_subject_ids = publication_eligibility_security_subject_ids(
                input_snapshot=input_snapshot,
                stored_candidate=stored_candidate,
                decision=decision,
                current_release_observation=current_before,
                task_adapter_trust_observations=adapter_observations,
            )
            denylist = self.security_denylist_authority.observe_exact(
                scope_id=stored_candidate.closure.validation_context.scope_id,
                scope_contract_id=attempt.scope_contract_id,
                checked_subject_ids=security_subject_ids,
            )
            if (
                type(denylist) is not SecurityDenylistObservation
                or denylist.scope_id
                != stored_candidate.closure.validation_context.scope_id
                or denylist.scope_contract_id != attempt.scope_contract_id
                or denylist.checked_subject_ids != security_subject_ids
                or not set(denylist.matched_subject_ids).issubset(
                    allowed_control_security_subject_ids
                )
            ):
                raise ExpertPublicationEligibilityError(
                    "publication eligibility denylist differs from exact authority"
                )
            manifest = stored_candidate.closure.manifest
            release_use_observation = self.release_use_policy_authority.observe_exact(
                scope_contract=(
                    stored_candidate.closure.validation_context.scope_contract
                ),
                checked_release_ids=manifest.consumed_expert_release_ids,
            )
            if (
                type(release_use_observation) is not ExpertReleaseUsePolicyObservation
                or release_use_observation.scope_id
                != stored_candidate.closure.validation_context.scope_id
                or release_use_observation.scope_contract_id
                != attempt.scope_contract_id
                or release_use_observation.checked_release_ids
                != manifest.consumed_expert_release_ids
            ):
                raise ExpertPublicationEligibilityError(
                    "publication eligibility release-use policy differs from candidate inputs"
                )
            current_after = self._observe_current(stored_candidate)
            if current_after.release_id != attempt.expected_current_release_id:
                return self._invalidate_current_authority(input_snapshot)
            if current_after != current_before:
                raise ExpertPublicationEligibilityError(
                    "publication eligibility CURRENT changed during external checks"
                )
            reopened = self.validation_store.reopen_publication_eligibility_snapshot(
                input_snapshot
            )
            if reopened != input_snapshot:
                raise ExpertPublicationEligibilityError(
                    "publication eligibility local authority changed during checks"
                )
            expected_security_subject_ids = (
                publication_eligibility_security_subject_ids(
                    input_snapshot=reopened,
                    stored_candidate=self._reopen_candidate(reopened),
                    decision=decision,
                    current_release_observation=current_before,
                    task_adapter_trust_observations=adapter_observations,
                )
            )
            if expected_security_subject_ids != security_subject_ids:
                raise ExpertPublicationEligibilityError(
                    "publication eligibility security projection changed"
                )
            release_use_decision = build_candidate_release_use_decision(
                input_snapshot=reopened,
                stored_candidate=stored_candidate,
                decision=decision,
                policy_observation=release_use_observation,
            )
            if release_use_decision.outcome is ExpertCandidateReleaseUseOutcome.CLEARED:
                fence = ExpertPublicationEligibilityAuthorityFence.mint(
                    release_matrix_acceptance_transition_id=(
                        snapshot.transition.transition_id
                    ),
                    release_matrix_acceptance_state_id=(
                        input_snapshot.snapshot.state.validation_state_id
                    ),
                    validation_attempt_id=attempt.validation_attempt_id,
                    candidate_id=attempt.candidate_id,
                    candidate_tree_hash=attempt.candidate_tree_hash,
                    candidate_commit_record_id=attempt.candidate_commit_record_id,
                    scope_contract_id=attempt.scope_contract_id,
                    scope_id=(stored_candidate.closure.validation_context.scope_id),
                    source_base_release_id=attempt.source_base_release_id,
                    expected_current_release_id=attempt.expected_current_release_id,
                    recovery_plan_id=attempt.recovery_plan_id,
                    control_dependency_ids=attempt.control_dependency_ids,
                    allowed_control_security_subject_ids=(
                        allowed_control_security_subject_ids
                    ),
                    validation_policy_id=attempt.validation_policy_id,
                    configuration_fingerprint=attempt.configuration_fingerprint,
                    release_matrix_stage_result_id=(
                        input_snapshot.release_matrix_stage_result.stage_result_record_id
                    ),
                    promotion_decision_id=decision.promotion_decision_id,
                    release_use_decision_id=(
                        release_use_decision.release_use_decision_id
                    ),
                    security_subject_ids=security_subject_ids,
                    current_release_observation=current_before,
                    task_adapter_trust_observations=adapter_observations,
                    security_denylist_observation=denylist,
                )
        stage_result = build_publication_eligibility_stage_result(
            input_snapshot=input_snapshot,
            stored_candidate=stored_candidate,
            decision=decision,
            release_use_decision=release_use_decision,
            publication_authority_fence=fence,
        )
        execution = ExpertPublicationEligibilityExecution(
            _PUBLICATION_ELIGIBILITY_EXECUTION_SEAL,
            self,
            validation_store=self.validation_store,
            input_snapshot=input_snapshot,
            stored_candidate=stored_candidate,
            decision=decision,
            stage_result=stage_result,
        )
        return self.validation_store.publish_publication_eligibility(execution)

    def _observe_current(
        self,
        stored_candidate: StoredExpertCandidate,
    ) -> TaskEvaluationCurrentReleaseObservation:
        scope_id = stored_candidate.closure.validation_context.scope_id
        observation = self.current_release_authority.observe_task_evaluation_current(
            scope_id
        )
        if (
            type(observation) is not TaskEvaluationCurrentReleaseObservation
            or observation.scope_id != scope_id
        ):
            raise ExpertPublicationEligibilityError(
                "publication eligibility CURRENT observation is invalid"
            )
        return observation

    def _reopen_candidate(
        self,
        input_snapshot: ExpertPublicationEligibilitySnapshot,
    ) -> StoredExpertCandidate:
        attempt = input_snapshot.snapshot.latest_attempt
        if attempt is None:
            raise ExpertPublicationEligibilityError(
                "publication eligibility candidate has no validation attempt"
            )
        stored = self.validation_store.reducer.candidate_store.read(
            attempt.candidate_id
        )
        manifest = stored.closure.manifest
        scope_contract = stored.closure.validation_context.scope_contract
        if (
            type(stored) is not StoredExpertCandidate
            or manifest.candidate_id != attempt.candidate_id
            or manifest.candidate_tree_hash != attempt.candidate_tree_hash
            or stored.commit_record.commit_record_id
            != attempt.candidate_commit_record_id
            or manifest.scope_contract_id != attempt.scope_contract_id
            or manifest.source_base_release_id != attempt.source_base_release_id
            or scope_contract.scope_contract_id != attempt.scope_contract_id
            or not set(stored_candidate_admission_dependency_ids(stored)).issubset(
                attempt.eligibility_dependency_ids
            )
        ):
            raise ExpertPublicationEligibilityError(
                "publication eligibility candidate differs from validation authority"
            )
        return stored

    def _reverify_adapters(
        self,
        input_snapshot: ExpertPublicationEligibilitySnapshot,
    ) -> tuple[TaskAdapterTrustObservation, ...]:
        observations = []
        plan = (
            input_snapshot.release_matrix_stage_result.release_matrix_report.evaluation_plan
        )
        for expected in plan.adapter_authorities:
            observed = self.task_adapter_authority.resolve_exact(
                task_adapter_manifest_id=(
                    expected.task_adapter_manifest.task_adapter_manifest_id
                ),
                verification_receipt_id=(
                    expected.verification_receipt.verification_receipt_id
                ),
            )
            if (
                type(observed) is not VerifiedTaskAdapter
                or observed.manifest != expected.task_adapter_manifest
                or observed.verification_receipt != expected.verification_receipt
                or observed.dependency_ids != expected.task_adapter_dependency_ids
            ):
                raise ExpertPublicationEligibilityError(
                    "publication eligibility adapter differs from matrix authority"
                )
            observations.append(
                TaskAdapterTrustObservation.mint(
                    task_adapter_manifest_id=(
                        observed.manifest.task_adapter_manifest_id
                    ),
                    verification_receipt_id=(
                        observed.verification_receipt.verification_receipt_id
                    ),
                    verifier_id=observed.verification_receipt.verifier_id,
                    verifier_version=observed.verification_receipt.verifier_version,
                    dependency_ids=observed.dependency_ids,
                )
            )
        ordered = tuple(sorted(observations, key=lambda item: item.observation_id))
        expected = publication_eligibility_task_adapter_trust_observations(
            input_snapshot
        )
        if ordered != expected:
            raise ExpertPublicationEligibilityError(
                "publication eligibility adapter observations are not exact"
            )
        return ordered

    def _invalidate_current_authority(
        self,
        input_snapshot: ExpertPublicationEligibilitySnapshot,
    ) -> ExpertValidationCommitResult:
        return self.validation_store.publish_current_release_authority_invalidation(
            candidate_id=input_snapshot.snapshot.state.candidate_id,
            expected_validation_state_id=(
                input_snapshot.snapshot.state.validation_state_id
            ),
        )


def publication_eligibility_task_adapter_trust_observations(
    input_snapshot: ExpertPublicationEligibilitySnapshot,
) -> tuple[TaskAdapterTrustObservation, ...]:
    """Derive the only adapter-observation set valid for the accepted matrix."""

    if type(input_snapshot) is not ExpertPublicationEligibilitySnapshot:
        raise ExpertPublicationEligibilityError(
            "publication eligibility adapter projection requires its exact snapshot"
        )
    plan = (
        input_snapshot.release_matrix_stage_result.release_matrix_report.evaluation_plan
    )
    return tuple(
        sorted(
            (
                TaskAdapterTrustObservation.mint(
                    task_adapter_manifest_id=(
                        authority.task_adapter_manifest.task_adapter_manifest_id
                    ),
                    verification_receipt_id=(
                        authority.verification_receipt.verification_receipt_id
                    ),
                    verifier_id=authority.verification_receipt.verifier_id,
                    verifier_version=authority.verification_receipt.verifier_version,
                    dependency_ids=authority.task_adapter_dependency_ids,
                )
                for authority in plan.adapter_authorities
            ),
            key=lambda observation: observation.observation_id,
        )
    )


def publication_eligibility_recovery_admission(
    *,
    attempt: ExpertValidationAttempt,
    stored_candidate: StoredExpertCandidate,
) -> ExpertRecoveryCandidateAdmission | None:
    """Reopen the exact durable recovery authority carried by an attempt."""

    if (
        type(attempt) is not ExpertValidationAttempt
        or type(stored_candidate) is not StoredExpertCandidate
    ):
        raise ExpertPublicationEligibilityError(
            "publication recovery authority requires exact durable inputs"
        )
    admission = stored_candidate.recovery_admission
    if admission is None:
        if (
            attempt.recovery_plan_id is not None
            or attempt.control_dependency_ids
            or attempt.expected_current_release_id != attempt.source_base_release_id
        ):
            raise ExpertPublicationEligibilityError(
                "publication attempt claims recovery without durable admission"
            )
        return None
    if type(admission) is not ExpertRecoveryCandidateAdmission:
        raise ExpertPublicationEligibilityError(
            "publication recovery admission uses another contract"
        )
    plan = admission.recovery_plan
    if (
        plan.recovery_plan_id != attempt.recovery_plan_id
        or plan.source_base_release_id != attempt.source_base_release_id
        or plan.activation_predecessor_release_id != attempt.expected_current_release_id
        or admission.candidate_id != attempt.candidate_id
        or admission.candidate_commit_record_id != attempt.candidate_commit_record_id
        or admission.control_dependency_ids != attempt.control_dependency_ids
        or not set(admission.dependency_ids).issubset(
            attempt.eligibility_dependency_ids
        )
    ):
        raise ExpertPublicationEligibilityError(
            "publication recovery admission differs from validation authority"
        )
    return admission


def publication_eligibility_security_subject_ids(
    *,
    input_snapshot: ExpertPublicationEligibilitySnapshot,
    stored_candidate: StoredExpertCandidate,
    decision: ExpertReleaseMatrixPromotionDecision,
    current_release_observation: TaskEvaluationCurrentReleaseObservation,
    task_adapter_trust_observations: tuple[TaskAdapterTrustObservation, ...],
) -> tuple[str, ...]:
    """Project the complete revocation closure checked before approval."""

    if (
        type(input_snapshot) is not ExpertPublicationEligibilitySnapshot
        or type(stored_candidate) is not StoredExpertCandidate
        or type(decision) is not ExpertReleaseMatrixPromotionDecision
        or type(current_release_observation)
        is not TaskEvaluationCurrentReleaseObservation
        or type(task_adapter_trust_observations) is not tuple
        or any(
            type(observation) is not TaskAdapterTrustObservation
            for observation in task_adapter_trust_observations
        )
    ):
        raise ExpertPublicationEligibilityError(
            "publication eligibility security projection requires exact authorities"
        )
    snapshot = input_snapshot.snapshot
    attempt = snapshot.latest_attempt
    if (
        attempt is None
        or input_snapshot.release_matrix_stage_result.stage_result_record_id
        != decision.release_matrix_stage_result_id
        or stored_candidate.closure.manifest.candidate_id != attempt.candidate_id
        or current_release_observation.scope_id
        != stored_candidate.closure.validation_context.scope_id
        or current_release_observation.release_id != attempt.expected_current_release_id
        or task_adapter_trust_observations
        != publication_eligibility_task_adapter_trust_observations(input_snapshot)
    ):
        raise ExpertPublicationEligibilityError(
            "publication eligibility security inputs do not share one authority"
        )
    recovery_admission = publication_eligibility_recovery_admission(
        attempt=attempt,
        stored_candidate=stored_candidate,
    )
    validation_context = stored_candidate.closure.validation_context
    candidate_security_subject_ids = (
        publication_eligibility_candidate_security_subject_ids(stored_candidate)
    )
    if recovery_admission is not None and set(
        recovery_admission.allowed_control_security_subject_ids
    ) & set(candidate_security_subject_ids):
        raise ExpertPublicationEligibilityError(
            "publication recovery waiver overlaps candidate scientific authority"
        )
    subjects = {
        snapshot.transition.transition_id,
        snapshot.state.validation_state_id,
        attempt.validation_attempt_id,
        *attempt.eligibility_dependency_ids,
        *attempt.control_dependency_ids,
        decision.promotion_decision_id,
        *decision.exact_dependency_ids,
        *candidate_security_subject_ids,
        current_release_observation.observation_id,
        *current_release_observation.validation_closure_ids,
    }
    if attempt.source_base_release_id is not None:
        subjects.add(attempt.source_base_release_id)
    if attempt.expected_current_release_id is not None:
        subjects.add(attempt.expected_current_release_id)
    if attempt.recovery_plan_id is not None:
        subjects.add(attempt.recovery_plan_id)
    source_base_release = validation_context.source_base_release
    if source_base_release is not None:
        subjects.update(source_base_release.consumed_dependency_ids)
    if current_release_observation.publication_id is not None:
        subjects.add(current_release_observation.publication_id)
    for result in snapshot.accepted_stage_results:
        if type(result) is ExpertEvaluatorResultRecord:
            subjects.update(
                {
                    result.evaluator_result_record_id,
                    result.evaluator_run.evaluator_run_id,
                    result.attestation_envelope.attestation.evaluator_attestation_id,
                    *result.evaluator_run.exact_input_ids,
                }
            )
        elif type(result) in {
            ExpertSourceReplayStageResultRecord,
            ExpertAutomatedReviewStageResultRecord,
            ExpertReleaseMatrixStageResultRecord,
        }:
            subjects.update(
                {
                    result.stage_result_record_id,
                    *result.exact_dependency_ids,
                }
            )
        else:
            raise ExpertPublicationEligibilityError(
                "publication eligibility accepted result type is unsupported"
            )
    for observation in task_adapter_trust_observations:
        subjects.update(
            {
                observation.observation_id,
                observation.task_adapter_manifest_id,
                observation.verification_receipt_id,
                observation.verifier_authority_subject_id,
                *observation.dependency_ids,
            }
        )
    if recovery_admission is not None:
        subjects.update(recovery_admission.allowed_control_security_subject_ids)
    ordered = tuple(sorted(subjects))
    for subject_id in ordered:
        require_content_id(subject_id, "publication eligibility security subject")
    return ordered


def publication_eligibility_candidate_security_subject_ids(
    stored_candidate: StoredExpertCandidate,
) -> tuple[str, ...]:
    """Project the complete derivation-neutral candidate revocation closure."""

    if type(stored_candidate) is not StoredExpertCandidate:
        raise ExpertPublicationEligibilityError(
            "candidate security projection requires one exact stored candidate"
        )
    closure = stored_candidate.closure
    manifest = closure.manifest
    if stored_candidate.commit_record.candidate_id != manifest.candidate_id:
        raise ExpertPublicationEligibilityError(
            "candidate security projection commit names another candidate"
        )
    subjects = _candidate_manifest_security_subject_ids(
        manifest=manifest,
        candidate_commit_record_id=stored_candidate.commit_record.commit_record_id,
        validation_context_dependency_ids=(
            closure.validation_context.stable_dependency_ids
        ),
    )
    source_base_release = closure.validation_context.source_base_release
    if source_base_release is not None:
        subjects.update(source_base_release.consumed_dependency_ids)
    derivation = closure.derivation
    if (
        manifest.derivation_kind
        in {
            ExpertCandidateDerivationKind.AGENT_PROPOSAL,
            ExpertCandidateDerivationKind.AGENT_RECOVERY_BOOTSTRAP,
        }
        and type(derivation) is ExpertAgentProposalDerivation
    ):
        if (
            stored_candidate.composition_admission_fence is not None
            or (
                manifest.derivation_kind is ExpertCandidateDerivationKind.AGENT_PROPOSAL
                and stored_candidate.recovery_admission is not None
            )
            or (
                manifest.derivation_kind
                is ExpertCandidateDerivationKind.AGENT_RECOVERY_BOOTSTRAP
                and type(stored_candidate.recovery_admission)
                is not ExpertRecoveryCandidateAdmission
            )
        ):
            raise ExpertPublicationEligibilityError(
                "agent candidate admission authority differs from its derivation"
            )
        subjects.update(_agent_derivation_security_subject_ids(derivation))
    elif (
        manifest.derivation_kind
        is ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION
        and type(derivation) is ExpertDeterministicCompositionDerivation
    ):
        admission_fence = stored_candidate.composition_admission_fence
        if type(admission_fence) is not ExpertCompositionAdmissionFence:
            raise ExpertPublicationEligibilityError(
                "composition candidate lacks its admission authority"
            )
        subjects.update(_composition_derivation_security_subject_ids(derivation))
        subjects.update(
            {
                admission_fence.admission_fence_id,
                admission_fence.security_denylist_observation.observation_id,
                admission_fence.security_denylist_observation.snapshot_id,
                admission_fence.security_denylist_observation.publication_id,
                *admission_fence.exact_dependency_ids,
                *admission_fence.security_subject_ids,
            }
        )
    elif (
        manifest.derivation_kind
        is ExpertCandidateDerivationKind.DETERMINISTIC_RECOVERY_RESTORE
        and type(derivation) is ExpertDeterministicRecoveryRestoreDerivation
    ):
        if (
            stored_candidate.composition_admission_fence is not None
            or type(stored_candidate.recovery_admission)
            is not ExpertRecoveryCandidateAdmission
        ):
            raise ExpertPublicationEligibilityError(
                "recovery restore lacks its exact admission authority"
            )
        subjects.update(_recovery_restore_security_subject_ids(derivation))
    else:
        raise ExpertPublicationEligibilityError(
            "candidate security projection does not recognize its derivation"
        )
    ordered = tuple(sorted(subjects))
    for subject_id in ordered:
        require_content_id(subject_id, "candidate security subject")
    return ordered


def _candidate_manifest_security_subject_ids(
    *,
    manifest: ExpertCandidateManifest,
    candidate_commit_record_id: str,
    validation_context_dependency_ids: tuple[str, ...],
) -> set[str]:
    return {
        manifest.candidate_id,
        candidate_commit_record_id,
        manifest.scope_contract_id,
        manifest.derivation_ref,
        manifest.validation_context_ref,
        manifest.patch_ref,
        manifest.candidate_tree_ref,
        manifest.proposed_repository_map_ref,
        manifest.sanitation_report_id,
        *manifest.module_contract_refs,
        *manifest.consumed_expert_release_ids,
        *manifest.source_dependency_ids,
        *manifest.ancestor_candidate_ids,
        *validation_context_dependency_ids,
    }


def _agent_derivation_security_subject_ids(
    derivation: ExpertAgentProposalDerivation,
) -> set[str]:
    if type(derivation) is not ExpertAgentProposalDerivation:
        raise ExpertPublicationEligibilityError(
            "agent security projection requires one direct agent derivation"
        )
    operation = derivation.operation
    return {
        derivation.record.derivation_id,
        derivation.record.trigger_evidence_packet_id,
        derivation.record.trigger_decision_id,
        *derivation.record.source_dependency_ids,
        operation.operation_record_id,
        operation.proposer_authority.authority_id,
        operation.operation_receipt.operation_receipt_id,
        operation.workspace_receipt.workspace_receipt_id,
        operation.workspace_delta_ref,
        derivation.workspace_delta.workspace_delta_id,
    }


def _recovery_restore_security_subject_ids(
    derivation: ExpertDeterministicRecoveryRestoreDerivation,
) -> set[str]:
    if type(derivation) is not ExpertDeterministicRecoveryRestoreDerivation:
        raise ExpertPublicationEligibilityError(
            "recovery security projection requires its exact derivation"
        )
    record = derivation.record
    return {
        record.derivation_id,
        record.replay_basis_packet_id,
        record.source_base_release_id,
        record.source_base_tree_receipt_id,
        *record.source_dependency_ids,
    }


def _composition_derivation_security_subject_ids(
    derivation: ExpertDeterministicCompositionDerivation,
) -> set[str]:
    if type(derivation) is not ExpertDeterministicCompositionDerivation:
        raise ExpertPublicationEligibilityError(
            "composition security projection requires its exact derivation"
        )
    materialization = derivation.materialization
    assessment = materialization.composition_assessment
    plan = assessment.composition_plan
    subjects = {
        derivation.record.derivation_id,
        derivation.record.composition_materialization_id,
        *derivation.record.source_validation_context_ids,
        *derivation.record.source_validation_context_ids.values(),
        *derivation.record.source_dependency_ids,
        materialization.materialization_id,
        *materialization.stable_authority_ids,
        assessment.assessment_id,
        *assessment.stable_authority_ids,
        plan.composition_plan_id,
        *plan.stable_authority_ids,
    }
    for provenance in derivation.source_provenance:
        subjects.update(_composition_source_security_subject_ids(provenance))
    return subjects


def _composition_source_security_subject_ids(
    provenance: ExpertCompositionSourceProvenance,
) -> set[str]:
    if (
        type(provenance) is not ExpertCompositionSourceProvenance
        or provenance.candidate_manifest.derivation_kind
        is not ExpertCandidateDerivationKind.AGENT_PROPOSAL
        or type(provenance.agent_derivation) is not ExpertAgentProposalDerivation
    ):
        raise ExpertPublicationEligibilityError(
            "composition security projection rejects unknown or nested sources"
        )
    source_reference = provenance.reduction_source.source_reference
    source_base_release = provenance.validation_context.source_base_release
    if source_base_release is None:
        raise ExpertPublicationEligibilityError(
            "composition security source lacks its source-base release"
        )
    subjects = _candidate_manifest_security_subject_ids(
        manifest=provenance.candidate_manifest,
        candidate_commit_record_id=(
            provenance.candidate_commit_record.commit_record_id
        ),
        validation_context_dependency_ids=(
            provenance.validation_context.stable_dependency_ids
        ),
    )
    subjects.update(
        {
            source_reference.source_reference_id,
            *source_reference.stable_authority_ids,
            *source_base_release.consumed_dependency_ids,
            *_agent_derivation_security_subject_ids(provenance.agent_derivation),
        }
    )
    for ancestor in provenance.agent_derivation.ancestor_inputs:
        subjects.update(_candidate_ancestor_security_subject_ids(ancestor))
    return subjects


def _candidate_ancestor_security_subject_ids(
    ancestor: ExpertCandidateAncestorInput,
) -> set[str]:
    if type(ancestor) is not ExpertCandidateAncestorInput:
        raise ExpertPublicationEligibilityError(
            "candidate ancestor security projection requires an exact input"
        )
    manifest = ancestor.manifest
    subjects = {
        ancestor.ancestor_input_id,
        ancestor.scope_contract.scope_contract_id,
        manifest.candidate_id,
        manifest.derivation_ref,
        manifest.validation_context_ref,
        manifest.patch_ref,
        manifest.candidate_tree_ref,
        manifest.proposed_repository_map_ref,
        manifest.sanitation_report_id,
        *manifest.module_contract_refs,
        *manifest.consumed_expert_release_ids,
        *manifest.source_dependency_ids,
        *manifest.ancestor_candidate_ids,
    }
    if manifest.source_base_release_id is not None:
        subjects.add(manifest.source_base_release_id)
    if manifest.source_base_repository_map_ref is not None:
        subjects.add(manifest.source_base_repository_map_ref)
    return subjects


def build_candidate_release_use_decision(
    *,
    input_snapshot: ExpertPublicationEligibilitySnapshot,
    stored_candidate: StoredExpertCandidate,
    decision: ExpertReleaseMatrixPromotionDecision,
    policy_observation: ExpertReleaseUsePolicyObservation,
) -> ExpertCandidateReleaseUseDecision:
    """Bind one current policy observation to the exact approved candidate."""

    snapshot = input_snapshot.snapshot
    attempt = snapshot.latest_attempt
    if (
        attempt is None
        or decision.outcome is not ExpertReleaseMatrixDecisionOutcome.APPROVED
    ):
        raise ExpertPublicationEligibilityError(
            "release-use decision differs from exact candidate authority"
        )
    return _build_candidate_release_use_decision(
        attempt=attempt,
        stored_candidate=stored_candidate,
        release_matrix_stage_result_id=(
            input_snapshot.release_matrix_stage_result.stage_result_record_id
        ),
        promotion_decision_id=decision.promotion_decision_id,
        policy_observation=policy_observation,
    )


def build_approved_candidate_release_use_decision(
    *,
    approval_snapshot: ExpertValidationSnapshot,
    stored_candidate: StoredExpertCandidate,
    policy_observation: ExpertReleaseUsePolicyObservation,
) -> ExpertCandidateReleaseUseDecision:
    """Bind a late current-policy read to one unchanged terminal approval."""

    attempt = approval_snapshot.latest_attempt
    publication_result = (
        None
        if not approval_snapshot.accepted_stage_results
        else approval_snapshot.accepted_stage_results[-1]
    )
    if (
        approval_snapshot.state.promotion_state is not ExpertPromotionState.APPROVED
        or attempt is None
        or type(publication_result) is not ExpertPublicationEligibilityStageResultRecord
        or publication_result.publication_authority_fence is None
        or publication_result.promotion_decision.outcome
        is not ExpertReleaseMatrixDecisionOutcome.APPROVED
    ):
        raise ExpertPublicationEligibilityError(
            "late release-use decision requires one exact terminal approval"
        )
    return _build_candidate_release_use_decision(
        attempt=attempt,
        stored_candidate=stored_candidate,
        release_matrix_stage_result_id=(
            publication_result.promotion_decision.release_matrix_stage_result_id
        ),
        promotion_decision_id=(
            publication_result.promotion_decision.promotion_decision_id
        ),
        policy_observation=policy_observation,
    )


def _build_candidate_release_use_decision(
    *,
    attempt: ExpertValidationAttempt,
    stored_candidate: StoredExpertCandidate,
    release_matrix_stage_result_id: str,
    promotion_decision_id: str,
    policy_observation: ExpertReleaseUsePolicyObservation,
) -> ExpertCandidateReleaseUseDecision:
    manifest = stored_candidate.closure.manifest
    if (
        manifest.candidate_id != attempt.candidate_id
        or manifest.candidate_tree_hash != attempt.candidate_tree_hash
        or stored_candidate.commit_record.commit_record_id
        != attempt.candidate_commit_record_id
        or policy_observation.scope_id
        != stored_candidate.closure.validation_context.scope_id
        or policy_observation.scope_contract_id != attempt.scope_contract_id
        or policy_observation.checked_release_ids
        != manifest.consumed_expert_release_ids
    ):
        raise ExpertPublicationEligibilityError(
            "release-use decision differs from exact candidate authority"
        )
    dependencies = {
        attempt.validation_attempt_id,
        attempt.candidate_id,
        attempt.candidate_commit_record_id,
        attempt.scope_contract_id,
        release_matrix_stage_result_id,
        promotion_decision_id,
        policy_observation.observation_id,
        policy_observation.knowledge_snapshot_id,
        policy_observation.knowledge_publication_id,
        *policy_observation.checked_release_ids,
    }
    for revocation in policy_observation.matched_revocations:
        dependencies.update(
            {
                revocation.revocation_id,
                revocation.release_id,
                revocation.release_publication_id,
                revocation.release_activation_witness_id,
                *revocation.exact_evidence_refs,
            }
        )
    outcome = (
        ExpertCandidateReleaseUseOutcome.BLOCKED
        if policy_observation.matched_revocations
        else ExpertCandidateReleaseUseOutcome.CLEARED
    )
    return ExpertCandidateReleaseUseDecision.mint(
        validation_attempt_id=attempt.validation_attempt_id,
        candidate_id=attempt.candidate_id,
        candidate_tree_hash=attempt.candidate_tree_hash,
        candidate_commit_record_id=attempt.candidate_commit_record_id,
        scope_contract_id=attempt.scope_contract_id,
        scope_id=stored_candidate.closure.validation_context.scope_id,
        release_matrix_stage_result_id=release_matrix_stage_result_id,
        promotion_decision_id=promotion_decision_id,
        policy_observation=policy_observation,
        outcome=outcome,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


def build_publication_eligibility_stage_result(
    *,
    input_snapshot: ExpertPublicationEligibilitySnapshot,
    stored_candidate: StoredExpertCandidate,
    decision: ExpertReleaseMatrixPromotionDecision,
    release_use_decision: ExpertCandidateReleaseUseDecision | None,
    publication_authority_fence: ExpertPublicationEligibilityAuthorityFence | None,
) -> ExpertPublicationEligibilityStageResultRecord:
    snapshot = input_snapshot.snapshot
    attempt = snapshot.latest_attempt
    if attempt is None:
        raise ExpertPublicationEligibilityError(
            "publication eligibility result has no validation attempt"
        )
    recovery_admission = publication_eligibility_recovery_admission(
        attempt=attempt,
        stored_candidate=stored_candidate,
    )
    allowed_control_security_subject_ids = (
        ()
        if recovery_admission is None
        else recovery_admission.allowed_control_security_subject_ids
    )
    scope_id = stored_candidate.closure.validation_context.scope_id
    dependencies = {
        snapshot.transition.transition_id,
        snapshot.state.validation_state_id,
        attempt.validation_attempt_id,
        attempt.candidate_id,
        attempt.candidate_commit_record_id,
        attempt.scope_contract_id,
        attempt.validation_policy_id,
        *(
            result.stage_result_record_id
            for result in snapshot.state.accepted_stage_results
        ),
        decision.promotion_decision_id,
        *decision.exact_dependency_ids,
        *attempt.control_dependency_ids,
        *allowed_control_security_subject_ids,
    }
    if attempt.source_base_release_id is not None:
        dependencies.add(attempt.source_base_release_id)
    if attempt.expected_current_release_id is not None:
        dependencies.add(attempt.expected_current_release_id)
    if attempt.recovery_plan_id is not None:
        dependencies.add(attempt.recovery_plan_id)
    if release_use_decision is not None:
        dependencies.update(
            {
                release_use_decision.release_use_decision_id,
                *release_use_decision.exact_dependency_ids,
            }
        )
    if publication_authority_fence is not None:
        dependencies.update(
            {
                publication_authority_fence.fence_id,
                *publication_authority_fence.exact_dependency_ids,
            }
        )
    return ExpertPublicationEligibilityStageResultRecord.mint(
        release_matrix_acceptance_transition_id=snapshot.transition.transition_id,
        release_matrix_acceptance_state_id=snapshot.state.validation_state_id,
        validation_attempt_id=attempt.validation_attempt_id,
        candidate_id=attempt.candidate_id,
        candidate_tree_hash=attempt.candidate_tree_hash,
        candidate_commit_record_id=attempt.candidate_commit_record_id,
        scope_contract_id=attempt.scope_contract_id,
        scope_id=scope_id,
        source_base_release_id=attempt.source_base_release_id,
        expected_current_release_id=attempt.expected_current_release_id,
        recovery_plan_id=attempt.recovery_plan_id,
        control_dependency_ids=attempt.control_dependency_ids,
        allowed_control_security_subject_ids=(allowed_control_security_subject_ids),
        validation_policy_id=attempt.validation_policy_id,
        configuration_fingerprint=attempt.configuration_fingerprint,
        accepted_stage_results=snapshot.state.accepted_stage_results,
        promotion_decision=decision,
        release_use_decision=release_use_decision,
        publication_authority_fence=publication_authority_fence,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )
