import base64
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ContractValidationError,
    CrossRunTaskBindingSettings,
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
    TaskAdapterManifest,
)
from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertEvaluatorRunBuilder,
    ExpertValidationError,
    ExpertValidationPredecessor,
    ExpertValidationReducer,
    VerifiedTaskAdapter,
)
from kapso.cross_run.settings import (
    CrossRunConfigurationError,
    CrossRunSettings,
)
from test_expert_candidate_store import candidate_store
from test_expert_candidates import bootstrap_candidate_closure

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def _content_id(label: str) -> str:
    return content_id("test-expert-validation", {"label": label})


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _validation_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).expert.validation


class _AttestationVerifier:
    def verify(self, envelope):
        if envelope.signature != "test-signature":
            raise ExpertValidationError("invalid test signature")


class _TaskAdapterProvider:
    def __init__(self, *adapters):
        self.adapters = {
            adapter.task_adapter_manifest_id: VerifiedTaskAdapter(
                manifest=adapter,
                source_verification_receipt_id=_content_id(
                    f"adapter-source-verification-{adapter.task_adapter_id}"
                ),
            )
            for adapter in adapters
        }

    def resolve(self, manifest_id):
        return self.adapters[manifest_id]


class _CurrentReleaseProvider:
    def __init__(self, release_id):
        self.release_id = release_id

    def current_release_id(self, scope_id):
        assert scope_id == "ml_ai"
        return self.release_id


class _ValidationStateProvider:
    def __init__(self, predecessor=None):
        self.predecessor = predecessor

    def current(self, candidate_id):
        if self.predecessor is not None:
            assert self.predecessor.state.candidate_id == candidate_id
        return self.predecessor


def _eligibility_evaluator(settings, store, adapter, current_release_id=None):
    return ExpertCandidateEligibilityEvaluator(
        settings,
        store,
        _TaskAdapterProvider(adapter),
        _CurrentReleaseProvider(current_release_id),
    )


def _validation_reducer(
    settings,
    adapter,
    current_release_id=None,
    predecessor=None,
):
    return ExpertValidationReducer(
        settings,
        _AttestationVerifier(),
        _TaskAdapterProvider(adapter),
        _CurrentReleaseProvider(current_release_id),
        _ValidationStateProvider(predecessor),
    )


def _task_adapter(closure, position=0) -> TaskAdapterManifest:
    binding = closure.trigger_packet.active_task_bindings[position]
    return TaskAdapterManifest.mint(
        task_adapter_id=binding.task_adapter_id,
        scope_contract_id=closure.manifest.scope_contract_id,
        task_family_id=binding.task_family_id,
        publisher_attestation={"issuer": "test", "signature": "test"},
        task_evaluator_binding={"evaluator": "public"},
        context_dimension_binding={"dataset_family": "synthetic"},
        source_tree_ref="task-adapter.tar.zst",
        tree_hash=_digest(f"task-adapter-tree-{binding.task_adapter_id}"),
        dependency_runtime_contract={"python": ">=3.10"},
        sanitation_report_id=_content_id("task-adapter-sanitation"),
        validation_refs=("validation.adapter_smoke",),
    )


def _eligibility_decision(
    *,
    track: ExpertValidationTrack = ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
) -> ExpertCandidateEligibilityDecision:
    settings = _validation_settings()
    policy = settings.policy.validation_policy()
    adapter_bindings = {
        content_id(
            "task-adapter-binding",
            {"task_family_id": "family", "task_adapter_id": "adapter"},
        ): _content_id("adapter")
    }
    stages = settings.policy.required_stages(
        track,
        ("family",),
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
        configured_task_family_ids=("family",),
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
        configured_task_family_ids=decision.configured_task_family_ids,
        task_adapter_manifest_ids=decision.task_adapter_manifest_ids,
        eligibility_dependency_ids=tuple(
            sorted(
                {
                    decision.eligibility_decision_id,
                    *decision.exact_dependency_ids,
                }
            )
        ),
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


def test_bootstrap_enrollment_derives_architecture_track_and_exact_stage_plan(
    tmp_path,
):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    settings = _validation_settings()
    eligibility = _eligibility_evaluator(settings, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_manifest_ids=(adapter.task_adapter_manifest_id,),
    )
    started = _validation_reducer(settings, adapter).start(
        stored_candidate=stored,
        eligibility=eligibility,
    )

    assert eligibility.decision.eligible is True
    assert (
        eligibility.decision.validation_track
        is ExpertValidationTrack.REPOSITORY_ARCHITECTURE
    )
    assert ExpertValidationStage.SOURCE_RUN_REPLAY not in (
        eligibility.decision.required_stages
    )
    assert started.attempt is not None
    assert started.attempt.required_stages == eligibility.decision.required_stages
    assert _content_id("adapter-source-verification-posttrain") in (
        started.attempt.eligibility_dependency_ids
    )
    assert started.state.next_stage is ExpertValidationStage.CONTRACT_SCHEMA


def test_adapter_enrollment_requires_every_exact_trigger_binding(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    first_binding = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="family/branch",
        task_adapter_id="adapter",
    )
    second_binding = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="family",
        task_adapter_id="branch/adapter",
    )
    expanded_closure = SimpleNamespace(
        manifest=stored.closure.manifest,
        trigger_packet=SimpleNamespace(
            active_task_bindings=(first_binding, second_binding)
        ),
    )
    expanded_stored = SimpleNamespace(closure=expanded_closure)
    first_adapter = _task_adapter(expanded_closure)
    second_adapter = _task_adapter(expanded_closure, position=1)
    verified_first = VerifiedTaskAdapter(
        manifest=first_adapter,
        source_verification_receipt_id=_content_id("first-source-verification"),
    )
    verified_second = VerifiedTaskAdapter(
        manifest=second_adapter,
        source_verification_receipt_id=_content_id("second-source-verification"),
    )

    with pytest.raises(ExpertValidationError, match="trigger bindings"):
        ExpertCandidateEligibilityEvaluator._adapter_bindings(
            expanded_stored,
            (verified_first,),
        )

    bindings, verification_ids, task_family_ids = (
        ExpertCandidateEligibilityEvaluator._adapter_bindings(
            expanded_stored,
            (verified_first, verified_second),
        )
    )
    assert bindings == {
        content_id(
            "task-adapter-binding",
            {
                "task_family_id": first_binding.task_family_id,
                "task_adapter_id": first_binding.task_adapter_id,
            },
        ): first_adapter.task_adapter_manifest_id,
        content_id(
            "task-adapter-binding",
            {
                "task_family_id": second_binding.task_family_id,
                "task_adapter_id": second_binding.task_adapter_id,
            },
        ): second_adapter.task_adapter_manifest_id,
    }
    assert len(bindings) == 2
    assert verification_ids == tuple(
        sorted(
            (
                verified_first.source_verification_receipt_id,
                verified_second.source_verification_receipt_id,
            )
        )
    )
    assert task_family_ids == tuple(
        sorted((first_binding.task_family_id, second_binding.task_family_id))
    )


def test_stale_bootstrap_and_forged_track_are_ineligible_or_rejected(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    settings = _validation_settings()
    evaluator = _eligibility_evaluator(
        settings,
        store,
        adapter,
        _content_id("already-released"),
    )
    stale = evaluator.decide(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_manifest_ids=(adapter.task_adapter_manifest_id,),
    )

    assert stale.decision.eligible is False
    assert stale.decision.reason_code == "stale_parent_release"
    assert stale.decision.required_stages == ()

    valid = _eligibility_evaluator(settings, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_manifest_ids=(adapter.task_adapter_manifest_id,),
    )
    forged_decision = ExpertCandidateEligibilityDecision.mint(
        **{
            **{
                key: value
                for key, value in valid.decision.to_dict().items()
                if key not in {"eligibility_decision_id", "validation_track"}
            },
            "validation_track": ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
        }
    )
    with pytest.raises(ExpertValidationError, match="deterministic"):
        _validation_reducer(settings, adapter).start(
            stored_candidate=stored,
            eligibility=replace(valid, decision=forged_decision),
        )


def test_bounded_evaluator_result_advances_only_the_exact_next_stage(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    settings = _validation_settings()
    eligibility = _eligibility_evaluator(settings, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_manifest_ids=(adapter.task_adapter_manifest_id,),
    )
    started = _validation_reducer(settings, adapter).start(
        stored_candidate=stored,
        eligibility=eligibility,
    )
    assert started.attempt is not None
    builder = ExpertEvaluatorRunBuilder(settings)
    result = builder.build(
        attempt=started.attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        exact_additional_input_ids=(),
        output_payloads={"result.json": b'{"passed":true}'},
        measurements={},
        costs={"compute_seconds": 1.0},
        duration_seconds=1.0,
        outcome=ExpertEvaluatorOutcome.PASSED,
        signature="test-signature",
    )

    advanced = _validation_reducer(settings, adapter).advance(
        state=started.state,
        attempt=started.attempt,
        accepted_results=(),
        result=result,
    )

    assert len(advanced.accepted_evaluator_evidence) == 1
    assert (
        advanced.next_stage is ExpertValidationStage.IDENTITY_SECRETS_LICENSE_DEPENDENCY
    )
    second_result = builder.build(
        attempt=started.attempt,
        stage=ExpertValidationStage.IDENTITY_SECRETS_LICENSE_DEPENDENCY,
        exact_additional_input_ids=(),
        output_payloads={"result.json": b'{"passed":true}'},
        measurements={},
        costs={},
        duration_seconds=1.0,
        outcome=ExpertEvaluatorOutcome.PASSED,
        signature="test-signature",
    )
    with pytest.raises(ExpertValidationError, match="history is incomplete"):
        _validation_reducer(settings, adapter).advance(
            state=advanced,
            attempt=started.attempt,
            accepted_results=(),
            result=second_result,
        )
    twice_advanced = _validation_reducer(settings, adapter).advance(
        state=advanced,
        attempt=started.attempt,
        accepted_results=(result,),
        result=second_result,
    )
    assert (
        twice_advanced.next_stage is ExpertValidationStage.STATIC_UNIT_SECURITY_RESOURCE
    )

    invalid_signature = replace(
        result,
        attestation_envelope=replace(
            result.attestation_envelope,
            signature="invalid",
        ),
    )
    with pytest.raises(ExpertValidationError, match="invalid test signature"):
        _validation_reducer(settings, adapter).advance(
            state=started.state,
            attempt=started.attempt,
            accepted_results=(),
            result=invalid_signature,
        )

    out_of_order = builder.build(
        attempt=started.attempt,
        stage=ExpertValidationStage.STATIC_UNIT_SECURITY_RESOURCE,
        exact_additional_input_ids=(),
        output_payloads={"result.json": b'{"passed":true}'},
        measurements={},
        costs={},
        duration_seconds=1.0,
        outcome=ExpertEvaluatorOutcome.PASSED,
        signature="test-signature",
    )
    with pytest.raises(ExpertValidationError, match="out of order"):
        _validation_reducer(settings, adapter).advance(
            state=started.state,
            attempt=started.attempt,
            accepted_results=(),
            result=out_of_order,
        )


def test_failed_stage_is_terminal_and_a_retry_requires_a_new_attempt(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    settings = _validation_settings()
    eligibility = _eligibility_evaluator(settings, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_manifest_ids=(adapter.task_adapter_manifest_id,),
    )
    reducer = _validation_reducer(settings, adapter)
    started = reducer.start(
        stored_candidate=stored,
        eligibility=eligibility,
    )
    assert started.attempt is not None
    failed_result = ExpertEvaluatorRunBuilder(settings).build(
        attempt=started.attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        exact_additional_input_ids=(),
        output_payloads={"result.json": b'{"passed":false}'},
        measurements={},
        costs={},
        duration_seconds=1.0,
        outcome=ExpertEvaluatorOutcome.CANDIDATE_FAILED,
        signature="test-signature",
    )
    failed = reducer.advance(
        state=started.state,
        attempt=started.attempt,
        accepted_results=(),
        result=failed_result,
    )

    assert failed.promotion_state is ExpertPromotionState.FAILED
    assert failed.next_stage is None
    with pytest.raises(ExpertValidationError, match="active attempt"):
        reducer.advance(
            state=failed,
            attempt=started.attempt,
            accepted_results=(),
            result=failed_result,
        )

    retry_reducer = _validation_reducer(
        settings,
        adapter,
        predecessor=ExpertValidationPredecessor(
            latest_attempt=started.attempt,
            state=failed,
        ),
    )
    retry = retry_reducer.start(
        stored_candidate=stored,
        eligibility=eligibility,
    )
    assert retry.attempt is not None
    assert retry.attempt.attempt_number == 2
    assert retry.attempt.predecessor_attempt_id == started.attempt.validation_attempt_id
    assert retry.state.next_stage is ExpertValidationStage.CONTRACT_SCHEMA


def test_ineligible_state_does_not_reset_historical_attempt_lineage(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    settings = _validation_settings()
    eligible = _eligibility_evaluator(settings, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_manifest_ids=(adapter.task_adapter_manifest_id,),
    )
    initial = _validation_reducer(settings, adapter).start(
        stored_candidate=stored,
        eligibility=eligible,
    )
    assert initial.attempt is not None
    failed_result = ExpertEvaluatorRunBuilder(settings).build(
        attempt=initial.attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        exact_additional_input_ids=(),
        output_payloads={"result.json": b'{"passed":false}'},
        measurements={},
        costs={},
        duration_seconds=1.0,
        outcome=ExpertEvaluatorOutcome.CANDIDATE_FAILED,
        signature="test-signature",
    )
    failed = _validation_reducer(settings, adapter).advance(
        state=initial.state,
        attempt=initial.attempt,
        accepted_results=(),
        result=failed_result,
    )
    historical_attempt = ExpertValidationPredecessor(
        latest_attempt=initial.attempt,
        state=failed,
    )
    current_release_id = _content_id("temporarily-current-release")
    stale = _eligibility_evaluator(
        settings,
        store,
        adapter,
        current_release_id,
    ).decide(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_manifest_ids=(adapter.task_adapter_manifest_id,),
    )
    ineligible = _validation_reducer(
        settings,
        adapter,
        current_release_id=current_release_id,
        predecessor=historical_attempt,
    ).start(
        stored_candidate=stored,
        eligibility=stale,
    )
    assert ineligible.attempt is None

    retry = _validation_reducer(
        settings,
        adapter,
        predecessor=ExpertValidationPredecessor(
            latest_attempt=initial.attempt,
            state=ineligible.state,
        ),
    ).start(
        stored_candidate=stored,
        eligibility=eligible,
    )
    assert retry.attempt is not None
    assert retry.attempt.attempt_number == 2
    assert retry.attempt.predecessor_attempt_id == initial.attempt.validation_attempt_id


def test_evaluator_output_limits_are_enforced_before_result_identity(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    settings = _validation_settings()
    limited = replace(
        settings,
        policy=replace(settings.policy, artifact_byte_limit=1),
    )
    eligibility = _eligibility_evaluator(limited, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
        task_adapter_manifest_ids=(adapter.task_adapter_manifest_id,),
    )
    started = _validation_reducer(limited, adapter).start(
        stored_candidate=stored,
        eligibility=eligibility,
    )
    assert started.attempt is not None

    with pytest.raises(ExpertValidationError, match="byte limit"):
        ExpertEvaluatorRunBuilder(limited).build(
            attempt=started.attempt,
            stage=ExpertValidationStage.CONTRACT_SCHEMA,
            exact_additional_input_ids=(),
            output_payloads={"result.json": b"too large"},
            measurements={},
            costs={},
            duration_seconds=1.0,
            outcome=ExpertEvaluatorOutcome.PASSED,
            signature="test-signature",
        )
