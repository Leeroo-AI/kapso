from __future__ import annotations

from dataclasses import fields

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertAcceptedStageResultRef,
    ExpertValidationStage,
    ExpertValidationTrack,
)
from kapso.cross_run.expert.promotion_authority_contracts import (
    ExpertCandidateReleaseUseDecision,
    ExpertCandidateReleaseUseOutcome,
    ExpertPublicationEligibilityAuthorityFence,
    ExpertPublicationEligibilityStageResultRecord,
)
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
    ExpertReleaseMatrixDecisionReason,
    ExpertReleaseMatrixPromotionDecision,
    ExpertReleaseMatrixReplicateAssessment,
    ExpertReleaseMatrixReplicateClassification,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
    TaskAdapterTrustObservation,
)
from security_denylist_fixtures import matched_security_revocations


def _id(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


def _remint(record, **changes):
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name != record.IDENTITY_FIELD
    }
    payload.update(changes)
    return type(record).mint(**payload)


def _decision(
    outcome: ExpertReleaseMatrixDecisionOutcome,
    *,
    mode: ExpertReleaseMatrixMode,
) -> ExpertReleaseMatrixPromotionDecision:
    attempt_id = _id("expert-validation-attempt", "attempt")
    policy_id = _id("expert-validation-policy", "policy")
    stage_result_id = _id("expert-release-matrix-stage-result", "matrix-stage")
    report_id = _id("expert-release-matrix-report", "matrix-report")
    operation_id = _id("expert-validation-operation", "plan-reservation")
    stage_dependencies = tuple(
        sorted(
            {
                attempt_id,
                policy_id,
                report_id,
                operation_id,
                _id("evaluation-fingerprint", "evaluation"),
            }
        )
    )
    attempt_dependencies = tuple(
        sorted(
            {
                _id("expert-candidate", "candidate"),
                _id("expert-candidate-commit", "candidate-commit"),
                _id("expert-scope-contract", "scope-contract"),
            }
        )
    )
    exact_dependencies = tuple(
        sorted(
            {
                stage_result_id,
                report_id,
                operation_id,
                attempt_id,
                policy_id,
                *stage_dependencies,
                *attempt_dependencies,
            }
        )
    )
    assessments: tuple[ExpertReleaseMatrixReplicateAssessment, ...]
    underpowered: tuple[str, ...]
    confirmed: tuple[str, ...]
    if mode is ExpertReleaseMatrixMode.BOOTSTRAP:
        if outcome is not ExpertReleaseMatrixDecisionOutcome.APPROVED:
            raise AssertionError("bootstrap fixture only supports approval")
        reason = ExpertReleaseMatrixDecisionReason.BOOTSTRAP_STANDALONE_COVERAGE
        track = ExpertValidationTrack.REPOSITORY_ARCHITECTURE
        assessments = ()
        underpowered = ()
        confirmed = ()
    elif outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED:
        reason = ExpertReleaseMatrixDecisionReason.CONFIRMED_BENEFIT
        track = ExpertValidationTrack.BEHAVIORAL_CAPABILITY
        assessments = (
            ExpertReleaseMatrixReplicateAssessment(
                evaluation_cell_id=_id(
                    "expert-release-matrix-evaluation-cell",
                    "quality",
                ),
                comparison_dimension_id="quality",
                replicate_id="replicate-1",
                normalized_effect=0.25,
                classification=ExpertReleaseMatrixReplicateClassification.GAIN,
                hard_regression=False,
            ),
        )
        underpowered = ()
        confirmed = ("quality",)
    elif outcome is ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED:
        reason = ExpertReleaseMatrixDecisionReason.UNDERPOWERED_EVIDENCE
        track = ExpertValidationTrack.BEHAVIORAL_CAPABILITY
        assessments = (
            ExpertReleaseMatrixReplicateAssessment(
                evaluation_cell_id=_id(
                    "expert-release-matrix-evaluation-cell",
                    "quality",
                ),
                comparison_dimension_id="quality",
                replicate_id="replicate-1",
                normalized_effect=0.0,
                classification=ExpertReleaseMatrixReplicateClassification.TIE,
                hard_regression=False,
            ),
        )
        underpowered = ("quality",)
        confirmed = ()
    else:
        reason = ExpertReleaseMatrixDecisionReason.NO_BENEFIT
        track = ExpertValidationTrack.BEHAVIORAL_CAPABILITY
        assessments = (
            ExpertReleaseMatrixReplicateAssessment(
                evaluation_cell_id=_id(
                    "expert-release-matrix-evaluation-cell",
                    "quality",
                ),
                comparison_dimension_id="quality",
                replicate_id="replicate-1",
                normalized_effect=0.0,
                classification=ExpertReleaseMatrixReplicateClassification.TIE,
                hard_regression=False,
            ),
        )
        underpowered = ()
        confirmed = ()
    return ExpertReleaseMatrixPromotionDecision.mint(
        release_matrix_stage_result_id=stage_result_id,
        release_matrix_report_id=report_id,
        plan_reservation_operation_id=operation_id,
        validation_attempt_id=attempt_id,
        validation_policy_id=policy_id,
        promotion_policy_version="promotion.v1",
        configuration_fingerprint=tree_or_blob_digest(b"configuration"),
        mode=mode,
        validation_track=track,
        outcome=outcome,
        reason=reason,
        replicate_assessments=assessments,
        underpowered_dimension_ids=underpowered,
        confirmed_benefit_dimension_ids=confirmed,
        release_matrix_stage_dependency_ids=stage_dependencies,
        attempt_dependency_ids=attempt_dependencies,
        exact_dependency_ids=exact_dependencies,
    )


def _trust_observations() -> tuple[TaskAdapterTrustObservation, ...]:
    observations = tuple(
        TaskAdapterTrustObservation.mint(
            task_adapter_manifest_id=_id(
                "task-adapter-manifest",
                f"adapter-{position}",
            ),
            verification_receipt_id=_id(
                "task-adapter-verification-receipt",
                f"receipt-{position}",
            ),
            verifier_id=f"verifier-{position}",
            verifier_version="v1",
            dependency_ids=tuple(
                sorted(
                    {
                        _id(
                            "task-adapter-verification-receipt",
                            f"receipt-{position}",
                        ),
                        _id("adapter-package", f"package-{position}"),
                    }
                )
            ),
        )
        for position in range(2)
    )
    return tuple(sorted(observations, key=lambda value: value.observation_id))


def _current_observation(
    *,
    expected_current_release_id: str | None,
) -> TaskEvaluationCurrentReleaseObservation:
    present = expected_current_release_id is not None
    return TaskEvaluationCurrentReleaseObservation.mint(
        scope_id="post-training",
        release_id=expected_current_release_id,
        publication_id=(
            _id("github-publication", "current-publication") if present else None
        ),
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="repository-node",
        default_branch_head_commit_sha="a" * 40,
        current_pointer_digest=(
            tree_or_blob_digest(b"current-pointer") if present else None
        ),
        validation_closure_ids=(
            (_id("expert-validation-transition", "released"),) if present else ()
        ),
    )


def _security_subject_ids(
    *,
    decision: ExpertReleaseMatrixPromotionDecision,
    accepted_stage_results: tuple[ExpertAcceptedStageResultRef, ...],
    expected_current_release_id: str | None,
    current: TaskEvaluationCurrentReleaseObservation,
    trust_observations: tuple[TaskAdapterTrustObservation, ...],
) -> tuple[str, ...]:
    subjects = {
        _id("expert-validation-transition", "matrix-accepted"),
        _id("expert-candidate-validation-state", "matrix-state"),
        decision.validation_attempt_id,
        _id("expert-candidate", "candidate"),
        _id("expert-candidate-commit", "candidate-commit"),
        _id("expert-scope-contract", "scope-contract"),
        decision.validation_policy_id,
        decision.release_matrix_stage_result_id,
        decision.promotion_decision_id,
        *decision.exact_dependency_ids,
        *(result.stage_result_record_id for result in accepted_stage_results),
        current.observation_id,
        *current.validation_closure_ids,
        _id("expert-candidate-source", "projected-candidate-closure"),
    }
    if expected_current_release_id is not None:
        subjects.add(expected_current_release_id)
    if current.publication_id is not None:
        subjects.add(current.publication_id)
    for observation in trust_observations:
        subjects.update(
            {
                observation.observation_id,
                observation.task_adapter_manifest_id,
                observation.verification_receipt_id,
                observation.verifier_authority_subject_id,
                *observation.dependency_ids,
            }
        )
    return tuple(sorted(subjects))


def _denylist(
    checked_subject_ids: tuple[str, ...],
    *,
    matched_subject_ids: tuple[str, ...] = (),
) -> SecurityDenylistObservation:
    return SecurityDenylistObservation.mint(
        scope_id="post-training",
        scope_contract_id=_id("expert-scope-contract", "scope-contract"),
        scope_repository_binding_hash=tree_or_blob_digest(b"repositories"),
        snapshot_id=_id("security-denylist-snapshot", "snapshot"),
        generation=3,
        publication_id=_id("github-publication", "denylist-publication"),
        repository_full_name="Leeroo-AI/kapso-knowledge",
        repository_node_id="denylist-repository-node",
        pointer_digest=tree_or_blob_digest(b"denylist-pointer"),
        authority_commit_sha="b" * 40,
        release_attestation_ref="refs/tags/security-v3",
        checked_subject_ids=checked_subject_ids,
        matched_revocations=matched_security_revocations(matched_subject_ids),
    )


def _approved_fence(
    *,
    decision: ExpertReleaseMatrixPromotionDecision,
    release_use_decision: ExpertCandidateReleaseUseDecision,
    accepted_stage_results: tuple[ExpertAcceptedStageResultRef, ...],
    source_base_release_id: str | None,
    expected_current_release_id: str | None,
    recovery_plan_id: str | None,
    control_dependency_ids: tuple[str, ...],
    allowed_control_security_subject_ids: tuple[str, ...],
    matched_subject_ids: tuple[str, ...] = (),
) -> ExpertPublicationEligibilityAuthorityFence:
    current = _current_observation(
        expected_current_release_id=expected_current_release_id
    )
    trust_observations = _trust_observations()
    security_subject_ids = _security_subject_ids(
        decision=decision,
        accepted_stage_results=accepted_stage_results,
        expected_current_release_id=expected_current_release_id,
        current=current,
        trust_observations=trust_observations,
    )
    security_subject_ids = tuple(
        sorted(
            {
                *security_subject_ids,
                *control_dependency_ids,
                *allowed_control_security_subject_ids,
                *(() if source_base_release_id is None else (source_base_release_id,)),
            }
        )
    )
    return ExpertPublicationEligibilityAuthorityFence.mint(
        release_matrix_acceptance_transition_id=_id(
            "expert-validation-transition",
            "matrix-accepted",
        ),
        release_matrix_acceptance_state_id=_id(
            "expert-candidate-validation-state",
            "matrix-state",
        ),
        validation_attempt_id=decision.validation_attempt_id,
        candidate_id=_id("expert-candidate", "candidate"),
        candidate_tree_hash=tree_or_blob_digest(b"candidate-tree"),
        candidate_commit_record_id=_id(
            "expert-candidate-commit",
            "candidate-commit",
        ),
        scope_contract_id=_id("expert-scope-contract", "scope-contract"),
        scope_id="post-training",
        source_base_release_id=source_base_release_id,
        expected_current_release_id=expected_current_release_id,
        recovery_plan_id=recovery_plan_id,
        control_dependency_ids=control_dependency_ids,
        allowed_control_security_subject_ids=(allowed_control_security_subject_ids),
        validation_policy_id=decision.validation_policy_id,
        configuration_fingerprint=decision.configuration_fingerprint,
        release_matrix_stage_result_id=decision.release_matrix_stage_result_id,
        promotion_decision_id=decision.promotion_decision_id,
        release_use_decision_id=release_use_decision.release_use_decision_id,
        security_subject_ids=security_subject_ids,
        current_release_observation=current,
        task_adapter_trust_observations=trust_observations,
        security_denylist_observation=_denylist(
            security_subject_ids,
            matched_subject_ids=matched_subject_ids,
        ),
    )


def _release_use_decision(
    *,
    decision: ExpertReleaseMatrixPromotionDecision,
    checked_release_ids: tuple[str, ...],
) -> ExpertCandidateReleaseUseDecision:
    observation = ExpertReleaseUsePolicyObservation.mint(
        scope_id="post-training",
        scope_contract_id=_id("expert-scope-contract", "scope-contract"),
        scope_repository_binding_hash=tree_or_blob_digest(b"repositories"),
        repository_full_name="Leeroo-AI/kapso-knowledge",
        repository_node_id="knowledge-repository-node",
        knowledge_snapshot_id=_id("knowledge-snapshot", "snapshot"),
        catalog_generation=3,
        knowledge_publication_id=_id("github-publication", "knowledge-publication"),
        current_pointer_digest=tree_or_blob_digest(b"knowledge-pointer"),
        authority_commit_sha="c" * 40,
        release_attestation_ref="refs/tags/knowledge-v3",
        checked_release_ids=checked_release_ids,
        matched_revocations=(),
    )
    dependencies = tuple(
        sorted(
            {
                decision.validation_attempt_id,
                _id("expert-candidate", "candidate"),
                _id("expert-candidate-commit", "candidate-commit"),
                _id("expert-scope-contract", "scope-contract"),
                decision.release_matrix_stage_result_id,
                decision.promotion_decision_id,
                observation.observation_id,
                observation.knowledge_snapshot_id,
                observation.knowledge_publication_id,
                *checked_release_ids,
            }
        )
    )
    return ExpertCandidateReleaseUseDecision.mint(
        validation_attempt_id=decision.validation_attempt_id,
        candidate_id=_id("expert-candidate", "candidate"),
        candidate_tree_hash=tree_or_blob_digest(b"candidate-tree"),
        candidate_commit_record_id=_id(
            "expert-candidate-commit",
            "candidate-commit",
        ),
        scope_contract_id=_id("expert-scope-contract", "scope-contract"),
        scope_id="post-training",
        release_matrix_stage_result_id=decision.release_matrix_stage_result_id,
        promotion_decision_id=decision.promotion_decision_id,
        policy_observation=observation,
        outcome=ExpertCandidateReleaseUseOutcome.CLEARED,
        exact_dependency_ids=dependencies,
    )


def _stage_result(
    outcome: ExpertReleaseMatrixDecisionOutcome,
    *,
    mode: ExpertReleaseMatrixMode = ExpertReleaseMatrixMode.CONTROL_COMPARISON,
    recovery_source_base: bool = True,
) -> ExpertPublicationEligibilityStageResultRecord:
    decision = _decision(outcome, mode=mode)
    accepted_stage_results = (
        ExpertAcceptedStageResultRef(
            stage=ExpertValidationStage.RELEASE_MATRIX,
            stage_result_record_id=decision.release_matrix_stage_result_id,
        ),
    )
    source_base_release_id = (
        None
        if mode is ExpertReleaseMatrixMode.BOOTSTRAP
        or (
            mode is ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
            and not recovery_source_base
        )
        else _id("expert-base-release", "source")
    )
    expected_current_release_id = source_base_release_id
    recovery_plan_id = None
    control_dependency_ids: tuple[str, ...] = ()
    allowed_control_security_subject_ids: tuple[str, ...] = ()
    if mode is ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY:
        expected_current_release_id = _id("expert-base-release", "blocked-barrier")
        recovery_plan_id = _id("expert-clean-forward-recovery-plan", "recovery")
        control_dependency_ids = tuple(
            sorted(
                {
                    expected_current_release_id,
                    recovery_plan_id,
                    _id("expert-recovery-candidate-admission", "admission"),
                }
            )
        )
        allowed_control_security_subject_ids = (expected_current_release_id,)
    release_use_decision = (
        _release_use_decision(
            decision=decision,
            checked_release_ids=(
                ()
                if expected_current_release_id is None
                else (expected_current_release_id,)
            ),
        )
        if outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
        else None
    )
    fence = (
        _approved_fence(
            decision=decision,
            release_use_decision=release_use_decision,
            accepted_stage_results=accepted_stage_results,
            source_base_release_id=source_base_release_id,
            expected_current_release_id=expected_current_release_id,
            recovery_plan_id=recovery_plan_id,
            control_dependency_ids=control_dependency_ids,
            allowed_control_security_subject_ids=(allowed_control_security_subject_ids),
            matched_subject_ids=(
                allowed_control_security_subject_ids
                if mode is ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
                else ()
            ),
        )
        if outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
        else None
    )
    dependencies = {
        _id("expert-validation-transition", "matrix-accepted"),
        _id("expert-candidate-validation-state", "matrix-state"),
        decision.validation_attempt_id,
        _id("expert-candidate", "candidate"),
        _id("expert-candidate-commit", "candidate-commit"),
        _id("expert-scope-contract", "scope-contract"),
        decision.validation_policy_id,
        *(result.stage_result_record_id for result in accepted_stage_results),
        decision.promotion_decision_id,
        *decision.exact_dependency_ids,
        *control_dependency_ids,
        *allowed_control_security_subject_ids,
    }
    if source_base_release_id is not None:
        dependencies.add(source_base_release_id)
    if expected_current_release_id is not None:
        dependencies.add(expected_current_release_id)
    if recovery_plan_id is not None:
        dependencies.add(recovery_plan_id)
    if release_use_decision is not None:
        dependencies.update(
            {
                release_use_decision.release_use_decision_id,
                *release_use_decision.exact_dependency_ids,
            }
        )
    if fence is not None:
        dependencies.update({fence.fence_id, *fence.exact_dependency_ids})
    return ExpertPublicationEligibilityStageResultRecord.mint(
        release_matrix_acceptance_transition_id=_id(
            "expert-validation-transition",
            "matrix-accepted",
        ),
        release_matrix_acceptance_state_id=_id(
            "expert-candidate-validation-state",
            "matrix-state",
        ),
        validation_attempt_id=decision.validation_attempt_id,
        candidate_id=_id("expert-candidate", "candidate"),
        candidate_tree_hash=tree_or_blob_digest(b"candidate-tree"),
        candidate_commit_record_id=_id(
            "expert-candidate-commit",
            "candidate-commit",
        ),
        scope_contract_id=_id("expert-scope-contract", "scope-contract"),
        scope_id="post-training",
        source_base_release_id=source_base_release_id,
        expected_current_release_id=expected_current_release_id,
        recovery_plan_id=recovery_plan_id,
        control_dependency_ids=control_dependency_ids,
        allowed_control_security_subject_ids=(allowed_control_security_subject_ids),
        validation_policy_id=decision.validation_policy_id,
        configuration_fingerprint=decision.configuration_fingerprint,
        accepted_stage_results=accepted_stage_results,
        promotion_decision=decision,
        release_use_decision=release_use_decision,
        publication_authority_fence=fence,
        exact_dependency_ids=tuple(sorted(dependencies)),
    )


@pytest.mark.parametrize(
    "outcome",
    (
        ExpertReleaseMatrixDecisionOutcome.FAILED,
        ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
        ExpertReleaseMatrixDecisionOutcome.APPROVED,
    ),
)
def test_terminal_record_accepts_every_pareto_outcome(outcome):
    result = _stage_result(outcome)

    assert result.outcome is outcome
    assert (result.publication_authority_fence is not None) == (
        outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
    )


def test_approved_record_roundtrips_with_exact_security_and_dependency_closures():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    fence = result.publication_authority_fence

    assert fence is not None
    assert (
        fence.security_denylist_observation.checked_subject_ids
        == fence.security_subject_ids
    )
    assert set(result.promotion_decision.exact_dependency_ids).issubset(
        fence.security_subject_ids
    )
    assert (
        ExpertPublicationEligibilityStageResultRecord.from_json_bytes(
            result.to_json_bytes()
        )
        == result
    )


def test_bootstrap_approval_binds_authenticated_current_absence():
    result = _stage_result(
        ExpertReleaseMatrixDecisionOutcome.APPROVED,
        mode=ExpertReleaseMatrixMode.BOOTSTRAP,
    )
    fence = result.publication_authority_fence

    assert result.expected_current_release_id is None
    assert fence is not None
    assert fence.current_release_observation.release_id is None
    assert fence.current_release_observation.publication_id is None


def test_recovery_approval_separates_scientific_source_from_current_control():
    result = _stage_result(
        ExpertReleaseMatrixDecisionOutcome.APPROVED,
        mode=ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY,
    )
    fence = result.publication_authority_fence

    assert fence is not None
    assert result.source_base_release_id != result.expected_current_release_id
    assert result.recovery_plan_id in result.control_dependency_ids
    assert result.expected_current_release_id in result.control_dependency_ids
    assert result.source_base_release_id not in result.control_dependency_ids
    assert result.allowed_control_security_subject_ids == (
        result.expected_current_release_id,
    )
    assert fence.security_denylist_observation.matched_subject_ids == (
        result.expected_current_release_id,
    )
    assert set(result.control_dependency_ids).issubset(fence.security_subject_ids)
    assert (
        fence.current_release_observation.release_id
        == result.expected_current_release_id
    )


def test_canonical_empty_recovery_approval_names_no_scientific_source():
    result = _stage_result(
        ExpertReleaseMatrixDecisionOutcome.APPROVED,
        mode=ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY,
        recovery_source_base=False,
    )
    fence = result.publication_authority_fence

    assert result.source_base_release_id is None
    assert result.expected_current_release_id is not None
    assert result.recovery_plan_id is not None
    assert fence is not None
    assert fence.source_base_release_id is None
    assert fence.expected_current_release_id == result.expected_current_release_id


def test_recovery_fence_rejects_revocation_of_scientific_source():
    result = _stage_result(
        ExpertReleaseMatrixDecisionOutcome.APPROVED,
        mode=ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY,
    )
    fence = result.publication_authority_fence
    assert fence is not None
    assert result.source_base_release_id is not None

    with pytest.raises(ValueError):
        _remint(
            fence,
            security_denylist_observation=_denylist(
                fence.security_subject_ids,
                matched_subject_ids=(result.source_base_release_id,),
            ),
        )


def test_recovery_fence_rejects_revocation_of_unwaived_control():
    result = _stage_result(
        ExpertReleaseMatrixDecisionOutcome.APPROVED,
        mode=ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY,
    )
    fence = result.publication_authority_fence
    assert fence is not None
    unwaived_control = next(
        dependency_id
        for dependency_id in result.control_dependency_ids
        if dependency_id not in result.allowed_control_security_subject_ids
    )

    with pytest.raises(ValueError):
        _remint(
            fence,
            security_denylist_observation=_denylist(
                fence.security_subject_ids,
                matched_subject_ids=(unwaived_control,),
            ),
        )


def test_fence_rejects_current_release_substitution():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    fence = result.publication_authority_fence
    assert fence is not None
    substituted_current = _current_observation(
        expected_current_release_id=_id("expert-base-release", "other-parent")
    )

    with pytest.raises(ValueError):
        _remint(fence, current_release_observation=substituted_current)


def test_fence_rejects_noncanonical_adapter_authority():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    fence = result.publication_authority_fence
    assert fence is not None

    with pytest.raises(ValueError):
        _remint(
            fence,
            task_adapter_trust_observations=tuple(
                reversed(fence.task_adapter_trust_observations)
            ),
        )


def test_fence_rejects_noncanonical_security_subject_projection():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    fence = result.publication_authority_fence
    assert fence is not None

    with pytest.raises(ValueError):
        _remint(
            fence,
            security_subject_ids=tuple(reversed(fence.security_subject_ids)),
        )


@pytest.mark.parametrize("checked_subject_change", ("missing", "extra"))
def test_fence_requires_denylist_checked_subject_equality(checked_subject_change):
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    fence = result.publication_authority_fence
    assert fence is not None
    if checked_subject_change == "missing":
        checked = fence.security_subject_ids[:-1]
    else:
        checked = tuple(
            sorted(
                {
                    *fence.security_subject_ids,
                    _id("expert-candidate-source", "unexpected"),
                }
            )
        )

    with pytest.raises(ValueError):
        _remint(fence, security_denylist_observation=_denylist(checked))


def test_fence_rejects_any_denied_security_subject():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    fence = result.publication_authority_fence
    assert fence is not None
    denied = (fence.security_subject_ids[0],)

    with pytest.raises(ValueError):
        _remint(
            fence,
            security_denylist_observation=_denylist(
                fence.security_subject_ids,
                matched_subject_ids=denied,
            ),
        )


def test_stage_rejects_approved_decision_without_fresh_fence():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)

    with pytest.raises(ValueError):
        _remint(result, publication_authority_fence=None)


@pytest.mark.parametrize(
    "outcome",
    (
        ExpertReleaseMatrixDecisionOutcome.FAILED,
        ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
    ),
)
def test_stage_prohibits_fresh_fence_for_nonapproved_outcome(outcome):
    result = _stage_result(outcome)
    approved = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)

    with pytest.raises(ValueError):
        _remint(
            result,
            publication_authority_fence=approved.publication_authority_fence,
        )


def test_stage_rejects_accepted_prefix_not_ending_in_decision_matrix():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.FAILED)
    wrong_matrix = ExpertAcceptedStageResultRef(
        stage=ExpertValidationStage.RELEASE_MATRIX,
        stage_result_record_id=_id(
            "expert-release-matrix-stage-result",
            "other-matrix",
        ),
    )

    with pytest.raises(ValueError):
        _remint(result, accepted_stage_results=(wrong_matrix,))


def test_stage_rejects_publication_eligibility_inside_accepted_prefix():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.FAILED)
    recursive_result = ExpertAcceptedStageResultRef(
        stage=ExpertValidationStage.PUBLICATION_ELIGIBILITY,
        stage_result_record_id=_id(
            "expert-publication-eligibility-stage-result",
            "recursive",
        ),
    )

    with pytest.raises(ValueError):
        _remint(
            result,
            accepted_stage_results=(
                recursive_result,
                *result.accepted_stage_results,
            ),
        )


def test_stage_rejects_candidate_commit_substitution_in_fence():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    fence = result.publication_authority_fence
    assert fence is not None
    other_commit = _id("expert-candidate-commit", "other-commit")
    subjects = tuple(sorted({*fence.security_subject_ids, other_commit}))
    substituted_fence = _remint(
        fence,
        candidate_commit_record_id=other_commit,
        security_subject_ids=subjects,
        security_denylist_observation=_denylist(subjects),
    )

    with pytest.raises(ValueError):
        _remint(result, publication_authority_fence=substituted_fence)


def test_stage_rejects_security_projection_omitting_decision_dependency():
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    fence = result.publication_authority_fence
    assert fence is not None
    omitted = next(
        dependency_id
        for dependency_id in result.promotion_decision.exact_dependency_ids
        if dependency_id
        not in {
            fence.release_matrix_stage_result_id,
            fence.validation_attempt_id,
            fence.validation_policy_id,
        }
    )
    subjects = tuple(
        subject_id for subject_id in fence.security_subject_ids if subject_id != omitted
    )
    reduced_fence = _remint(
        fence,
        security_subject_ids=subjects,
        security_denylist_observation=_denylist(subjects),
    )

    with pytest.raises(ValueError):
        _remint(result, publication_authority_fence=reduced_fence)


@pytest.mark.parametrize("dependency_change", ("missing", "extra", "reversed"))
def test_stage_requires_exact_canonical_dependency_closure(dependency_change):
    result = _stage_result(ExpertReleaseMatrixDecisionOutcome.APPROVED)
    if dependency_change == "missing":
        dependencies = result.exact_dependency_ids[:-1]
    elif dependency_change == "extra":
        dependencies = tuple(
            sorted(
                {
                    *result.exact_dependency_ids,
                    _id("expert-validation-transition", "unrelated"),
                }
            )
        )
    else:
        dependencies = tuple(reversed(result.exact_dependency_ids))

    with pytest.raises(ValueError):
        _remint(result, exact_dependency_ids=dependencies)
