import base64
import copy

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ContractValidationError,
    ExpertCandidateEligibilityDecision,
    ExpertCandidateValidationState,
    ExpertEvaluatorAttestation,
    ExpertEvaluatorAttestationEnvelope,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorRun,
    ExpertPromotionState,
    ExpertSealedCanaryAggregate,
    ExpertValidationAttempt,
    ExpertValidationStage,
    ExpertValidationTrack,
)
from kapso.cross_run.settings import (
    CrossRunConfigurationError,
    CrossRunSettings,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _content_id(label: str) -> str:
    return content_id("test-expert-validation", {"label": label})


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _validation_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).expert.validation


def _eligibility_decision(
    *,
    track: ExpertValidationTrack = ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
) -> ExpertCandidateEligibilityDecision:
    settings = _validation_settings()
    policy = settings.policy.validation_policy()
    adapter_bindings = {"family": _content_id("adapter")}
    stages = settings.policy.required_stages(
        track,
        tuple(adapter_bindings),
        has_parent_release=True,
    )
    return ExpertCandidateEligibilityDecision.mint(
        candidate_id=_content_id("candidate"),
        candidate_tree_hash=_digest("candidate-tree"),
        candidate_commit_record_id=_content_id("candidate-commit"),
        scope_contract_id=_content_id("scope"),
        parent_release_id=_content_id("parent-release"),
        validation_policy_id=policy.validation_policy_id,
        configuration_fingerprint=settings.configuration_fingerprint,
        eligible=True,
        validation_track=track,
        required_stages=stages,
        task_adapter_manifest_ids=adapter_bindings,
        exact_dependency_ids=tuple(
            sorted(
                {
                    _content_id("candidate-commit"),
                    _content_id("candidate"),
                    _content_id("adapter"),
                    _content_id("scope"),
                    _content_id("parent-release"),
                    policy.validation_policy_id,
                }
            )
        ),
        reason_code="eligible",
    )


def _attempt(
    decision: ExpertCandidateEligibilityDecision,
) -> ExpertValidationAttempt:
    return ExpertValidationAttempt.mint(
        candidate_id=decision.candidate_id,
        candidate_tree_hash=decision.candidate_tree_hash,
        candidate_commit_record_id=decision.candidate_commit_record_id,
        scope_contract_id=decision.scope_contract_id,
        parent_release_id=decision.parent_release_id,
        eligibility_decision_id=decision.eligibility_decision_id,
        validation_policy_id=decision.validation_policy_id,
        configuration_fingerprint=decision.configuration_fingerprint,
        validation_track=decision.validation_track,
        attempt_number=1,
        predecessor_attempt_id=None,
        required_stages=decision.required_stages,
        task_adapter_manifest_ids=decision.task_adapter_manifest_ids,
    )


def test_validation_policy_is_typed_and_independent_of_local_state_path():
    raw = load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    first = CrossRunSettings.from_dict(raw).expert.validation
    relocated = copy.deepcopy(raw)
    relocated["expert"]["validation"]["state_path"] = ".kapso/another_validation_store"
    second = CrossRunSettings.from_dict(relocated).expert.validation

    assert first.policy.validation_policy() == second.policy.validation_policy()
    assert first.configuration_fingerprint != second.configuration_fingerprint
    assert tuple(reviewer.agent.model for reviewer in first.policy.reviewers) == (
        "gpt-5.6-sol",
        "gpt-5.6-sol",
    )
    assert tuple(reviewer.agent.effort for reviewer in first.policy.reviewers) == (
        "xhigh",
        "xhigh",
    )


def test_stage_plan_is_track_parent_and_family_aware_and_fails_closed():
    policy = _validation_settings().policy
    bootstrap = policy.required_stages(
        ExpertValidationTrack.REPOSITORY_ARCHITECTURE,
        ("family",),
        has_parent_release=False,
    )
    mechanical = policy.required_stages(
        ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
        ("family",),
        has_parent_release=True,
    )
    behavioral = policy.required_stages(
        ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
        ("family", "second_family"),
        has_parent_release=True,
    )

    assert ExpertValidationStage.SOURCE_RUN_REPLAY not in bootstrap
    assert ExpertValidationStage.DEVELOPMENT_ANCHORS not in bootstrap
    assert ExpertValidationStage.SEALED_CANARY not in bootstrap
    assert policy.can_validate(
        ExpertValidationTrack.REPOSITORY_ARCHITECTURE,
        ("family",),
        has_parent_release=False,
    )
    assert ExpertValidationStage.SOURCE_RUN_REPLAY in mechanical
    assert ExpertValidationStage.SEALED_CANARY not in mechanical
    assert policy.can_validate(
        ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
        ("family",),
        has_parent_release=True,
    )
    assert ExpertValidationStage.CROSS_FAMILY_TRANSFER in behavioral
    assert ExpertValidationStage.SEALED_CANARY in behavioral
    assert not policy.can_validate(
        ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
        ("family", "second_family"),
        has_parent_release=True,
    )
    with pytest.raises(
        CrossRunConfigurationError,
        match="only repository architecture",
    ):
        policy.required_stages(
            ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
            ("family",),
            has_parent_release=False,
        )


def test_validation_attempt_binds_eligibility_policy_tree_and_adapters():
    decision = _eligibility_decision()
    attempt = _attempt(decision)

    assert attempt.eligibility_decision_id == decision.eligibility_decision_id
    assert attempt.validation_policy_id == decision.validation_policy_id
    assert attempt.candidate_tree_hash == decision.candidate_tree_hash
    assert attempt.task_adapter_manifest_ids == decision.task_adapter_manifest_ids
    assert attempt.required_stages[0] is ExpertValidationStage.CONTRACT_SCHEMA


def test_evaluator_run_carries_exact_outputs_and_signature_is_a_separate_envelope():
    decision = _eligibility_decision()
    attempt = _attempt(decision)
    output = b'{"passed":true}'
    arguments = {
        "validation_attempt_id": attempt.validation_attempt_id,
        "candidate_id": attempt.candidate_id,
        "candidate_tree_hash": attempt.candidate_tree_hash,
        "stage": attempt.required_stages[0],
        "evaluator_id": "expert_contract_evaluator",
        "evaluator_role": "expert_contract_evaluator",
        "evaluator_version": "kapso.expert_contract_evaluator.v1",
        "exact_input_ids": tuple(
            sorted(
                {
                    attempt.validation_attempt_id,
                    *attempt.task_adapter_manifest_ids.values(),
                }
            )
        ),
        "output_payloads_base64": {
            "result.json": base64.b64encode(output).decode("ascii")
        },
        "output_checksums": {"result.json": tree_or_blob_digest(output)},
        "measurements": {},
        "costs": {"compute_seconds": 1.0},
        "duration_seconds": 1.0,
        "outcome": ExpertEvaluatorOutcome.PASSED,
    }
    evaluator_run = ExpertEvaluatorRun.mint(**arguments)
    attestation = ExpertEvaluatorAttestation.mint(
        evaluator_run_id=evaluator_run.evaluator_run_id,
        issuer_id="expert_contract_evaluator",
        trust_root_id=None,
        predicate_digest=tree_or_blob_digest(evaluator_run.to_json_bytes()),
    )
    first = ExpertEvaluatorAttestationEnvelope(
        attestation=attestation,
        signature="first",
    )
    rotated = ExpertEvaluatorAttestationEnvelope(
        attestation=attestation,
        signature="rotated",
    )

    assert (
        first.attestation.evaluator_attestation_id
        == rotated.attestation.evaluator_attestation_id
    )
    assert first.signature != rotated.signature

    corrupt = dict(arguments)
    corrupt["output_checksums"] = {"result.json": _digest("wrong")}
    with pytest.raises(ContractValidationError, match="checksum differs"):
        ExpertEvaluatorRun.mint(
            **corrupt,
        )


def test_sealed_canary_persists_only_a_typed_aggregate():
    decision = _eligibility_decision(track=ExpertValidationTrack.BEHAVIORAL_CAPABILITY)
    attempt = _attempt(decision)
    aggregate = ExpertSealedCanaryAggregate(
        candidate_id=attempt.candidate_id,
        candidate_tree_hash=attempt.candidate_tree_hash,
        evaluator_version="kapso.expert_sealed_canary_evaluator.v1",
        evaluated_case_count=4,
        aggregate_measurements={"quality": -0.25},
    )
    aggregate_bytes = aggregate.to_json_bytes()
    arguments = {
        "validation_attempt_id": attempt.validation_attempt_id,
        "candidate_id": attempt.candidate_id,
        "candidate_tree_hash": attempt.candidate_tree_hash,
        "stage": ExpertValidationStage.SEALED_CANARY,
        "evaluator_id": "expert_sealed_canary_evaluator",
        "evaluator_role": "expert_sealed_canary_evaluator",
        "evaluator_version": aggregate.evaluator_version,
        "exact_input_ids": (attempt.validation_attempt_id,),
        "output_payloads_base64": {
            "aggregate.json": base64.b64encode(aggregate_bytes).decode("ascii")
        },
        "output_checksums": {"aggregate.json": tree_or_blob_digest(aggregate_bytes)},
        "measurements": {"quality": -0.25},
        "costs": {},
        "duration_seconds": 1.0,
        "outcome": ExpertEvaluatorOutcome.PASSED,
    }

    evaluator_run = ExpertEvaluatorRun.mint(**arguments)
    assert evaluator_run.measurements["quality"] == -0.25

    leaked = dict(arguments)
    leaked["output_payloads_base64"] = {
        **arguments["output_payloads_base64"],
        "cases.json": base64.b64encode(b"[]").decode("ascii"),
    }
    leaked["output_checksums"] = {
        **arguments["output_checksums"],
        "cases.json": tree_or_blob_digest(b"[]"),
    }
    with pytest.raises(ContractValidationError, match="only aggregate.json"):
        ExpertEvaluatorRun.mint(**leaked)


def test_ineligible_state_has_no_attempt_and_validating_state_names_next_stage():
    decision = _eligibility_decision()
    ineligible = ExpertCandidateValidationState.mint(
        validation_attempt_id=None,
        candidate_id=decision.candidate_id,
        candidate_tree_hash=decision.candidate_tree_hash,
        predecessor_state_id=None,
        promotion_state=ExpertPromotionState.INELIGIBLE,
        accepted_evaluator_evidence=(),
        next_stage=None,
        review_assertion_ids=(),
        terminal_evidence_ids=(decision.eligibility_decision_id,),
        transition_evidence_id=decision.eligibility_decision_id,
        reason="sealed validation infrastructure is unavailable",
    )
    attempt = _attempt(decision)
    validating = ExpertCandidateValidationState.mint(
        validation_attempt_id=attempt.validation_attempt_id,
        candidate_id=decision.candidate_id,
        candidate_tree_hash=decision.candidate_tree_hash,
        predecessor_state_id=ineligible.validation_state_id,
        promotion_state=ExpertPromotionState.VALIDATING,
        accepted_evaluator_evidence=(),
        next_stage=attempt.required_stages[0],
        review_assertion_ids=(),
        terminal_evidence_ids=(),
        transition_evidence_id=decision.eligibility_decision_id,
        reason="validation attempt started",
    )

    assert ineligible.validation_attempt_id is None
    assert validating.next_stage is ExpertValidationStage.CONTRACT_SCHEMA

    with pytest.raises(
        ContractValidationError,
        match="evaluated reviewed lineage",
    ):
        ExpertCandidateValidationState.mint(
            validation_attempt_id=attempt.validation_attempt_id,
            candidate_id=decision.candidate_id,
            candidate_tree_hash=decision.candidate_tree_hash,
            predecessor_state_id=None,
            promotion_state=ExpertPromotionState.APPROVED,
            accepted_evaluator_evidence=(),
            next_stage=None,
            review_assertion_ids=(),
            terminal_evidence_ids=(decision.eligibility_decision_id,),
            transition_evidence_id=decision.eligibility_decision_id,
            reason="invalid direct approval",
        )


def test_proposal_and_validation_authorities_must_be_disjoint():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["expert"]["validation"]["policy"]["evaluators"][0]["evaluator_id"] = raw[
        "expert"
    ]["architect_id"]

    with pytest.raises(CrossRunConfigurationError, match="must be disjoint"):
        CrossRunSettings.from_dict(raw)

    role_overlap = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    role_overlap["expert"]["validation"]["policy"]["evaluators"][0][
        "evaluator_role"
    ] = role_overlap["expert"]["architect_role"]
    with pytest.raises(CrossRunConfigurationError, match="roles must be disjoint"):
        CrossRunSettings.from_dict(role_overlap)

    internal_role_overlap = copy.deepcopy(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    internal_role_overlap["expert"]["validation"]["policy"]["reviewers"][0][
        "reviewer_role"
    ] = internal_role_overlap["expert"]["validation"]["policy"]["evaluators"][0][
        "evaluator_role"
    ]
    with pytest.raises(CrossRunConfigurationError, match="roles must be disjoint"):
        CrossRunSettings.from_dict(internal_role_overlap)
