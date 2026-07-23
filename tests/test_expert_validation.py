import base64
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    CrossRunContractError,
    CrossRunTaskBindingSettings,
    ExpertAcceptedStageResultRef,
    ExpertCandidateEligibilityDecision,
    ExpertCandidateValidationState,
    ExpertEvaluatorAttestation,
    ExpertEvaluatorAttestationEnvelope,
    ExpertEvaluatorOutcome,
    ExpertEvaluatorResultRecord,
    ExpertEvaluatorRun,
    ExpertPromotionState,
    ExpertSealedCanaryAggregate,
    ExpertSourceReplayAdapterPackagePin,
    ExpertSourceReplayCase,
    ExpertSourceReplaySelection,
    ExpertValidationAttempt,
    ExpertValidationStage,
    ExpertValidationTrack,
    ObjectiveDirection,
    SourceFileDescriptor,
    TaskAdapterContextBinding,
    TaskAdapterPackagePin,
    TaskAdapterManifest,
    TaskAdapterRuntimeContract,
    TaskEvaluatorBinding,
    TaskEvaluatorMetricComparisonBinding,
)
from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertEvaluatorRunBuilder,
    ExpertValidationError,
    ExpertValidationPredecessor,
    ExpertValidationReducer,
)
from kapso.cross_run.settings import (
    CrossRunConfigurationError,
    CrossRunSettings,
)
from kapso.cross_run.github.materializer import SourceArchiveExtractionReceipt
from kapso.cross_run.task_adapters import (
    TaskAdapterVerificationReceipt,
    VerifiedTaskAdapter,
)
from test_expert_candidate_store import candidate_store
from test_expert_candidates import bootstrap_candidate_closure
from test_expert_clean_recovery import _historical_candidate_system
from task_adapter_matrix_fixtures import task_adapter_release_matrix_case

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
TASK_ADAPTER_RUNTIME_LOCK = b"python==3.11.9\n"


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
        verified_adapters = tuple(_verified_adapter(adapter) for adapter in adapters)
        self.active_adapters = {
            (
                adapter.manifest.scope_contract_id,
                adapter.manifest.task_family_id,
                adapter.manifest.task_adapter_id,
            ): adapter
            for adapter in verified_adapters
        }
        self.exact_adapters = {
            (
                adapter.manifest.task_adapter_manifest_id,
                adapter.verification_receipt.verification_receipt_id,
            ): adapter
            for adapter in verified_adapters
        }

    def resolve_active(self, *, scope_contract_id, task_family_id, task_adapter_id):
        return self.active_adapters[
            (scope_contract_id, task_family_id, task_adapter_id)
        ]

    def resolve_exact(self, *, task_adapter_manifest_id, verification_receipt_id):
        return self.exact_adapters[(task_adapter_manifest_id, verification_receipt_id)]


class _CurrentReleaseProvider:
    def __init__(self, release_id):
        self.release_id = release_id

    def current_release_id(self, scope_id):
        assert scope_id == "ml_ai"
        return self.release_id


class _UnavailableCandidateReader:
    def read(self, candidate_id):
        raise AssertionError(f"unexpected candidate read: {candidate_id}")


class _CountingCandidateReader:
    def __init__(self, reader):
        self.reader = reader
        self.candidate_ids = []

    def read(self, candidate_id):
        self.candidate_ids.append(candidate_id)
        return self.reader.read(candidate_id)


class _StaticCandidateReader:
    def __init__(self, stored):
        self.stored = stored

    def read(self, candidate_id):
        assert self.stored.closure.manifest.candidate_id == candidate_id
        return self.stored


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
    candidate_store=None,
    current_release_id=None,
    predecessor=None,
):
    return ExpertValidationReducer(
        settings,
        candidate_store or _UnavailableCandidateReader(),
        _AttestationVerifier(),
        _TaskAdapterProvider(adapter),
        _CurrentReleaseProvider(current_release_id),
        _ValidationStateProvider(predecessor),
    )


def _task_adapter(closure, position=0) -> TaskAdapterManifest:
    binding = closure.validation_context.active_task_bindings[position]
    scope = closure.validation_context.scope_contract
    evaluator_fingerprint = _digest("source-evaluator")
    _, _, tree_hash = _adapter_source(binding.task_adapter_id)
    return TaskAdapterManifest.mint(
        task_adapter_id=binding.task_adapter_id,
        scope_contract_id=closure.manifest.scope_contract_id,
        task_family_id=binding.task_family_id,
        publisher_attestation={"issuer": "test", "signature": "test"},
        task_evaluator=TaskEvaluatorBinding(
            protocol_version="kapso.task_evaluator.v1",
            executable_path="adapter.py",
            supported_evaluator_fingerprints=(evaluator_fingerprint,),
            metric_comparison_bindings=(
                TaskEvaluatorMetricComparisonBinding(
                    evaluator_fingerprint=evaluator_fingerprint,
                    metric_name="accuracy",
                    objective_direction=ObjectiveDirection.MAXIMIZE,
                    comparison_dimension_id="quality",
                    comparison_scale=1.0,
                ),
            ),
        ),
        context_binding=TaskAdapterContextBinding(consumed_dimension_ids=()),
        release_matrix_cases=(
            task_adapter_release_matrix_case(
                scope_contract_id=scope.scope_contract_id,
                scope_id=scope.scope_id,
                task_family_id=binding.task_family_id,
                task_adapter_id=binding.task_adapter_id,
                evaluator_fingerprint=evaluator_fingerprint,
                metric_directions=(("accuracy", ObjectiveDirection.MAXIMIZE),),
                transfer_dimensions={
                    schema.dimension_id: "fixture"
                    for schema in scope.context_dimension_schemas
                },
                label=f"{binding.task_family_id}:{binding.task_adapter_id}",
            ),
        ),
        source_tree_ref="task-adapter.tar.zst",
        tree_hash=tree_hash,
        runtime=TaskAdapterRuntimeContract(
            runtime_protocol_version="kapso.task_adapter_runtime.v1",
            image_repository="registry.example/kapso/task-adapter-runtime",
            image_manifest_digest=_digest("task-adapter-runtime-image"),
            image_config_digest=_digest("task-adapter-runtime-config"),
            dependency_lock_path="requirements.lock",
            dependency_lock_digest=tree_or_blob_digest(TASK_ADAPTER_RUNTIME_LOCK),
            operating_system="linux",
            architecture="amd64",
            architecture_variant=None,
            environment={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        ),
        sanitation_report_id=_content_id("task-adapter-sanitation"),
        validation_refs=("validation.adapter_smoke",),
    )


def _adapter_source(
    task_adapter_id: str,
) -> tuple[dict[str, bytes], tuple[SourceFileDescriptor, ...], str]:
    source_contents = {
        "adapter.py": f"ADAPTER_ID = {task_adapter_id!r}\n".encode("utf-8"),
        "requirements.lock": TASK_ADAPTER_RUNTIME_LOCK,
    }
    source_files = tuple(
        SourceFileDescriptor(
            relative_path=path,
            digest=tree_or_blob_digest(payload),
            mode="100755" if path == "adapter.py" else "100644",
            size=len(payload),
        )
        for path, payload in sorted(source_contents.items())
    )
    tree_hash = source_tree_digest(
        {
            item.relative_path: (item.digest, item.mode, item.size)
            for item in source_files
        }
    )
    return source_contents, source_files, tree_hash


def _verified_adapter(adapter: TaskAdapterManifest) -> VerifiedTaskAdapter:
    proof_refs = {adapter.sanitation_report_id, *adapter.validation_refs}
    proof_objects = {
        proof_ref: f"proof:{proof_ref}".encode("utf-8") for proof_ref in proof_refs
    }
    source_contents, source_files, _ = _adapter_source(adapter.task_adapter_id)
    source_archive = f"archive:{adapter.task_adapter_id}".encode("utf-8")
    publisher_verification = f"publisher-verification:{adapter.task_adapter_id}".encode(
        "utf-8"
    )
    extraction_receipt = SourceArchiveExtractionReceipt.mint(
        artifact_id=adapter.task_adapter_manifest_id,
        source_archive_ref=adapter.source_tree_ref,
        source_archive_digest=tree_or_blob_digest(source_archive),
        source_tree_hash=adapter.tree_hash,
        source_tree_files=source_files,
        extractor_version="kapso.source_archive_extractor.v1",
    )
    receipt = TaskAdapterVerificationReceipt.mint(
        task_adapter_manifest_id=adapter.task_adapter_manifest_id,
        full_manifest_digest=tree_or_blob_digest(adapter.to_json_bytes()),
        publisher_attestation_digest=tree_or_blob_digest(
            canonical_json_bytes(adapter.publisher_attestation)
        ),
        source_extraction_receipt_id=extraction_receipt.extraction_receipt_id,
        source_archive_ref=adapter.source_tree_ref,
        source_archive_digest=tree_or_blob_digest(source_archive),
        source_tree_hash=adapter.tree_hash,
        proof_object_digests={
            proof_ref: tree_or_blob_digest(payload)
            for proof_ref, payload in proof_objects.items()
        },
        publisher_verification_digest=tree_or_blob_digest(publisher_verification),
        verifier_id="test_task_adapter_verifier",
        verifier_version="test.task_adapter_verifier.v1",
    )
    return VerifiedTaskAdapter(
        manifest=adapter,
        verification_receipt=receipt,
        source_extraction_receipt=extraction_receipt,
        source_archive=source_archive,
        source_contents=source_contents,
        proof_objects=proof_objects,
        publisher_verification=publisher_verification,
    )


def _eligibility_decision(
    *,
    track: ExpertValidationTrack = ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
) -> ExpertCandidateEligibilityDecision:
    settings = _validation_settings()
    policy = settings.policy.validation_policy()
    adapter_pin = TaskAdapterPackagePin(
        adapter_binding_id=content_id(
            "task-adapter-binding",
            {"task_family_id": "family", "task_adapter_id": "adapter"},
        ),
        task_adapter_manifest_id=_content_id("adapter"),
        verification_receipt_id=_content_id("adapter-verification"),
    )
    stages = settings.policy.required_stages(
        track,
        ("family",),
        has_source_base_release=True,
    )
    candidate_id = content_id("expert-candidate", {"label": "candidate"})
    candidate_commit_record_id = content_id(
        "expert-candidate-commit",
        {"label": "candidate-commit"},
    )
    episode_id = content_id("transfer-episode", {"label": "episode"})
    bundle_id = content_id("run-bundle", {"label": "bundle"})
    validation_context_id = content_id(
        "expert-candidate-validation-context",
        {"label": "validation-context"},
    )
    snapshot_id = content_id("knowledge-snapshot", {"label": "snapshot"})
    replay_evidence_id = content_id(
        "expert-candidate-replay-evidence",
        {"label": "replay-evidence"},
    )
    evidence_authority_ids = tuple(sorted((replay_evidence_id, snapshot_id)))
    source_adapter_manifest_id = content_id(
        "task-adapter-manifest",
        {"label": "source-adapter"},
    )
    source_adapter_receipt_id = content_id(
        "task-adapter-verification-receipt",
        {"label": "source-adapter-verification"},
    )
    source_adapter_pin = ExpertSourceReplayAdapterPackagePin.mint(
        scope_contract_id=content_id(
            "expert-scope-contract",
            {"label": "source-scope"},
        ),
        task_family_id="family",
        task_adapter_id="adapter",
        task_adapter_manifest_id=source_adapter_manifest_id,
        verification_receipt_id=source_adapter_receipt_id,
        episode_ids=(episode_id,),
    )
    selection_dependencies = tuple(
        sorted(
            {
                candidate_id,
                candidate_commit_record_id,
                validation_context_id,
                *evidence_authority_ids,
                policy.validation_policy_id,
                episode_id,
                bundle_id,
                source_adapter_pin.source_adapter_pin_id,
                source_adapter_manifest_id,
                source_adapter_receipt_id,
            }
        )
    )
    source_replay_selection = ExpertSourceReplaySelection.mint(
        candidate_id=candidate_id,
        candidate_tree_hash=_digest("candidate-tree"),
        candidate_commit_record_id=candidate_commit_record_id,
        validation_context_id=validation_context_id,
        evidence_authority_ids=evidence_authority_ids,
        validation_policy_id=policy.validation_policy_id,
        selection_policy_version=(
            settings.policy.source_replay_selection_policy_version
        ),
        configuration_fingerprint=settings.configuration_fingerprint,
        causal_episode_ids=(episode_id,),
        coverage_episode_ids=(),
        selection_evidence_ids=(episode_id,),
        cases=(
            ExpertSourceReplayCase(
                source_bundle_id=bundle_id,
                episode_ids=(episode_id,),
                episode_reason_codes={episode_id: ("causal_trigger_evidence",)},
            ),
        ),
        source_adapter_pins=(source_adapter_pin,),
        exact_dependency_ids=selection_dependencies,
    )
    return ExpertCandidateEligibilityDecision.mint(
        candidate_id=candidate_id,
        candidate_tree_hash=_digest("candidate-tree"),
        candidate_commit_record_id=candidate_commit_record_id,
        scope_contract_id=_content_id("scope"),
        source_base_release_id=_content_id("parent-release"),
        expected_current_release_id=_content_id("parent-release"),
        recovery_plan_id=None,
        validation_policy_id=policy.validation_policy_id,
        configuration_fingerprint=settings.configuration_fingerprint,
        eligible=True,
        validation_track=track,
        required_stages=stages,
        configured_task_family_ids=("family",),
        task_adapter_pins=(adapter_pin,),
        source_replay_selection=source_replay_selection,
        control_dependency_ids=(),
        exact_dependency_ids=tuple(
            sorted(
                {
                    candidate_commit_record_id,
                    candidate_id,
                    _content_id("adapter"),
                    _content_id("adapter-verification"),
                    _content_id("scope"),
                    _content_id("parent-release"),
                    policy.validation_policy_id,
                    source_replay_selection.source_replay_selection_id,
                    *source_replay_selection.exact_dependency_ids,
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
        source_base_release_id=decision.source_base_release_id,
        expected_current_release_id=decision.expected_current_release_id,
        recovery_plan_id=decision.recovery_plan_id,
        eligibility_decision_id=decision.eligibility_decision_id,
        validation_policy_id=decision.validation_policy_id,
        configuration_fingerprint=decision.configuration_fingerprint,
        validation_track=decision.validation_track,
        attempt_number=1,
        predecessor_attempt_id=None,
        required_stages=decision.required_stages,
        configured_task_family_ids=decision.configured_task_family_ids,
        task_adapter_pins=decision.task_adapter_pins,
        source_replay_selection=decision.source_replay_selection,
        control_dependency_ids=decision.control_dependency_ids,
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
        has_source_base_release=False,
    )
    mechanical = policy.required_stages(
        ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
        ("family",),
        has_source_base_release=True,
    )
    behavioral = policy.required_stages(
        ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
        ("family", "second_family"),
        has_source_base_release=True,
    )

    assert ExpertValidationStage.SOURCE_RUN_REPLAY not in bootstrap
    assert ExpertValidationStage.DEVELOPMENT_ANCHORS not in bootstrap
    assert ExpertValidationStage.SEALED_CANARY not in bootstrap
    assert policy.can_validate(
        ExpertValidationTrack.REPOSITORY_ARCHITECTURE,
        ("family",),
        has_source_base_release=False,
    )
    assert ExpertValidationStage.SOURCE_RUN_REPLAY in mechanical
    assert ExpertValidationStage.SEALED_CANARY not in mechanical
    assert policy.can_validate(
        ExpertValidationTrack.MECHANICAL_GENERAL_FIX,
        ("family",),
        has_source_base_release=True,
    )
    assert ExpertValidationStage.CROSS_FAMILY_TRANSFER in behavioral
    assert ExpertValidationStage.SEALED_CANARY in behavioral
    assert not policy.can_validate(
        ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
        ("family", "second_family"),
        has_source_base_release=True,
    )
    with pytest.raises(
        CrossRunConfigurationError,
        match="only repository architecture",
    ):
        policy.required_stages(
            ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
            ("family",),
            has_source_base_release=False,
        )


def test_validation_attempt_binds_eligibility_policy_tree_and_adapters():
    decision = _eligibility_decision()
    attempt = _attempt(decision)

    assert attempt.eligibility_decision_id == decision.eligibility_decision_id
    assert attempt.validation_policy_id == decision.validation_policy_id
    assert attempt.candidate_tree_hash == decision.candidate_tree_hash
    assert attempt.task_adapter_pins == decision.task_adapter_pins
    assert attempt.required_stages[0] is ExpertValidationStage.CONTRACT_SCHEMA


def test_ordinary_eligibility_and_attempt_reject_split_current_authority():
    decision = _eligibility_decision()
    decision_values = decision.to_dict()
    decision_values.pop("eligibility_decision_id")
    decision_values["expected_current_release_id"] = _content_id("other-current")
    with pytest.raises(
        ContractValidationError,
        match="ordinary eligibility must bind CURRENT",
    ):
        ExpertCandidateEligibilityDecision.mint(**decision_values)

    attempt = _attempt(decision)
    attempt_values = attempt.to_dict()
    attempt_values.pop("validation_attempt_id")
    attempt_values["expected_current_release_id"] = _content_id("other-current")
    with pytest.raises(
        ContractValidationError,
        match="ordinary validation attempt must bind CURRENT",
    ):
        ExpertValidationAttempt.mint(**attempt_values)


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
                    *(
                        pin.task_adapter_manifest_id
                        for pin in attempt.task_adapter_pins
                    ),
                    *(pin.verification_receipt_id for pin in attempt.task_adapter_pins),
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


def test_source_replay_execution_fails_closed_without_a_typed_receipt():
    settings = _validation_settings()
    attempt = _attempt(_eligibility_decision())
    selection = attempt.source_replay_selection
    assert selection is not None
    bundle_ids = tuple(case.source_bundle_id for case in selection.cases)
    builder = ExpertEvaluatorRunBuilder(settings)
    with pytest.raises(ExpertValidationError, match="typed execution receipt"):
        builder.build(
            attempt=attempt,
            stage=ExpertValidationStage.SOURCE_RUN_REPLAY,
            exact_additional_input_ids=bundle_ids,
            output_payloads={"result.json": b'{"passed":true}'},
            measurements={},
            costs={},
            duration_seconds=1.0,
            outcome=ExpertEvaluatorOutcome.PASSED,
            signature="test-signature",
        )

    evaluator = next(
        evaluator
        for evaluator in settings.policy.evaluators
        if evaluator.stage is ExpertValidationStage.SOURCE_RUN_REPLAY
    )
    payload = b'{"passed":true}'
    forged_run = ExpertEvaluatorRun.mint(
        validation_attempt_id=attempt.validation_attempt_id,
        candidate_id=attempt.candidate_id,
        candidate_tree_hash=attempt.candidate_tree_hash,
        stage=ExpertValidationStage.SOURCE_RUN_REPLAY,
        evaluator_id=evaluator.evaluator_id,
        evaluator_role=evaluator.evaluator_role,
        evaluator_version=evaluator.evaluator_version,
        exact_input_ids=(attempt.validation_attempt_id,),
        output_payloads_base64={
            "result.json": base64.b64encode(payload).decode("ascii")
        },
        output_checksums={"result.json": tree_or_blob_digest(payload)},
        measurements={},
        costs={},
        duration_seconds=1.0,
        outcome=ExpertEvaluatorOutcome.PASSED,
    )
    forged_attestation = ExpertEvaluatorAttestation.mint(
        evaluator_run_id=forged_run.evaluator_run_id,
        issuer_id=forged_run.evaluator_id,
        trust_root_id=None,
        predicate_digest=tree_or_blob_digest(forged_run.to_json_bytes()),
    )
    forged_result = ExpertEvaluatorResultRecord.mint(
        evaluator_run=forged_run,
        attestation_envelope=ExpertEvaluatorAttestationEnvelope(
            attestation=forged_attestation,
            signature="test-signature",
        ),
    )
    reducer = SimpleNamespace(settings=settings)
    with pytest.raises(ExpertValidationError, match="typed execution receipt"):
        ExpertValidationReducer._validate_result_closure(
            reducer,
            attempt,
            forged_result,
        )


def test_release_matrix_fails_closed_without_a_reserved_typed_stage_path():
    settings = _validation_settings()
    attempt = _attempt(_eligibility_decision())
    with pytest.raises(ExpertValidationError, match="typed stage path"):
        ExpertEvaluatorRunBuilder(settings).build(
            attempt=attempt,
            stage=ExpertValidationStage.RELEASE_MATRIX,
            exact_additional_input_ids=(_content_id("matrix-input"),),
            output_payloads={"result.json": b'{"passed":true}'},
            measurements={"quality": 1.0},
            costs={},
            duration_seconds=1.0,
            outcome=ExpertEvaluatorOutcome.PASSED,
            signature="test-signature",
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
        accepted_stage_results=(),
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
        accepted_stage_results=(),
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
            accepted_stage_results=(),
            next_stage=None,
            review_assertion_ids=(),
            terminal_evidence_ids=(decision.eligibility_decision_id,),
            transition_evidence_id=decision.eligibility_decision_id,
            reason="invalid direct approval",
        )


def test_accepted_stage_result_reference_is_typed_by_stage():
    source_result_id = content_id(
        "expert-source-replay-stage-result",
        {"source": True},
    )
    source_reference = ExpertAcceptedStageResultRef(
        stage=ExpertValidationStage.SOURCE_RUN_REPLAY,
        stage_result_record_id=source_result_id,
    )

    assert source_reference.stage_result_record_id == source_result_id
    with pytest.raises(ContractValidationError, match="wrong namespace"):
        ExpertAcceptedStageResultRef(
            stage=ExpertValidationStage.CONTRACT_SCHEMA,
            stage_result_record_id=source_result_id,
        )
    matrix_result_id = content_id(
        "expert-release-matrix-stage-result",
        {"matrix": True},
    )
    matrix_reference = ExpertAcceptedStageResultRef(
        stage=ExpertValidationStage.RELEASE_MATRIX,
        stage_result_record_id=matrix_result_id,
    )
    assert matrix_reference.stage_result_record_id == matrix_result_id
    with pytest.raises(ContractValidationError, match="wrong namespace"):
        ExpertAcceptedStageResultRef(
            stage=ExpertValidationStage.RELEASE_MATRIX,
            stage_result_record_id=content_id(
                "expert-evaluator-result-record",
                {"legacy-matrix": True},
            ),
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
    )
    reducer_candidate_reader = _CountingCandidateReader(store)
    started = _validation_reducer(
        settings,
        adapter,
        candidate_store=reducer_candidate_reader,
    ).start(
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
    assert reducer_candidate_reader.candidate_ids == [eligibility.decision.candidate_id]
    assert set(_verified_adapter(adapter).dependency_ids).issubset(
        started.attempt.eligibility_dependency_ids
    )
    assert started.state.next_stage is ExpertValidationStage.CONTRACT_SCHEMA


def test_recovery_enrollment_separates_scientific_source_from_current_authority(
    tmp_path,
):
    historical_system = _historical_candidate_system(tmp_path / "historical")
    historical_stored = historical_system.coordinator.restore_historical(
        scope_contract=historical_system.fixture.case.scope,
        replay_basis_packet=historical_system.replay_basis,
    )
    historical_admission = historical_stored.recovery_admission
    assert historical_admission is not None
    historical_adapter = _task_adapter(historical_stored.closure)
    settings = _validation_settings()
    historical_eligibility = _eligibility_evaluator(
        settings,
        historical_system.candidate_store,
        historical_adapter,
        historical_system.barrier.release_id,
    ).decide(candidate_id=historical_stored.closure.manifest.candidate_id)
    assert historical_eligibility.decision.source_base_release_id == (
        historical_system.selected.release_id
    )
    assert historical_eligibility.decision.expected_current_release_id == (
        historical_system.barrier.release_id
    )
    assert historical_eligibility.decision.recovery_plan_id == (
        historical_admission.recovery_plan.recovery_plan_id
    )
    assert historical_eligibility.decision.control_dependency_ids == (
        historical_admission.control_dependency_ids
    )
    assert (
        historical_system.selected.release_id
        not in historical_eligibility.decision.control_dependency_ids
    )

    system = _historical_candidate_system(
        tmp_path / "empty",
        empty_selection=True,
    )
    proposal = system.coordinator.bootstrap_empty(
        scope_contract=system.fixture.case.scope,
        replay_basis_packet=system.replay_basis,
    )
    stored = proposal.stored_candidate
    admission = stored.recovery_admission
    assert admission is not None
    adapter = _task_adapter(stored.closure)
    eligibility = _eligibility_evaluator(
        settings,
        system.candidate_store,
        adapter,
        system.barrier.release_id,
    ).decide(candidate_id=stored.closure.manifest.candidate_id)

    assert eligibility.decision.eligible is True
    assert eligibility.decision.source_base_release_id is None
    assert eligibility.decision.expected_current_release_id == system.barrier.release_id
    assert eligibility.decision.recovery_plan_id == (
        admission.recovery_plan.recovery_plan_id
    )
    assert (
        eligibility.decision.control_dependency_ids == admission.control_dependency_ids
    )
    started = _validation_reducer(
        settings,
        adapter,
        candidate_store=system.candidate_store,
        current_release_id=system.barrier.release_id,
    ).start(eligibility=eligibility)
    assert started.attempt is not None
    assert started.attempt.expected_current_release_id == (
        eligibility.decision.expected_current_release_id
    )
    assert started.attempt.recovery_plan_id == eligibility.decision.recovery_plan_id
    assert (
        started.attempt.control_dependency_ids
        == eligibility.decision.control_dependency_ids
    )

    stale = _eligibility_evaluator(
        settings,
        system.candidate_store,
        adapter,
        _content_id("moved-current"),
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    assert stale.decision.eligible is False
    assert stale.decision.reason_code == "recovery_barrier_not_current"

    missing_admission = replace(stored, recovery_admission=None)
    with pytest.raises(ExpertValidationError, match="lacks durable recovery admission"):
        _eligibility_evaluator(
            settings,
            _StaticCandidateReader(missing_admission),
            adapter,
            system.barrier.release_id,
        ).decide(candidate_id=stored.closure.manifest.candidate_id)


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
    source_scope = stored.closure.derivation.trigger_packet.scope_contract
    scope_values = source_scope.to_dict()
    scope_values.pop("scope_contract_id")
    family_type = type(source_scope.task_family_ontology[0])
    binding_type = type(source_scope.task_adapter_contract[0])
    scope_values["task_family_ontology"] = (
        family_type(
            task_family_id="family",
            capability_tags=("test.family",),
        ),
        family_type(
            task_family_id="family/branch",
            capability_tags=("test.family_branch",),
        ),
    )
    scope_values["task_adapter_contract"] = (
        binding_type(
            task_family_id="family",
            task_adapter_ids=("branch/adapter",),
        ),
        binding_type(
            task_family_id="family/branch",
            task_adapter_ids=("adapter",),
        ),
    )
    expanded_scope = type(source_scope).mint(**scope_values)
    expanded_closure = SimpleNamespace(
        manifest=SimpleNamespace(scope_contract_id=expanded_scope.scope_contract_id),
        validation_context=SimpleNamespace(
            active_task_bindings=(first_binding, second_binding),
            scope_contract=expanded_scope,
        ),
    )
    expanded_stored = SimpleNamespace(closure=expanded_closure)
    first_adapter = _task_adapter(expanded_closure)
    second_adapter = _task_adapter(expanded_closure, position=1)
    verified_first = _verified_adapter(first_adapter)
    verified_second = _verified_adapter(second_adapter)

    with pytest.raises(ExpertValidationError, match="unverified package"):
        ExpertCandidateEligibilityEvaluator._adapter_bindings(
            expanded_stored,
            (SimpleNamespace(manifest=first_adapter),),
        )
    with pytest.raises(ExpertValidationError, match="trigger bindings"):
        ExpertCandidateEligibilityEvaluator._adapter_bindings(
            expanded_stored,
            (verified_first,),
        )

    pins, verification_ids, task_family_ids = (
        ExpertCandidateEligibilityEvaluator._adapter_bindings(
            expanded_stored,
            (verified_first, verified_second),
        )
    )
    assert {pin.adapter_binding_id: pin.task_adapter_manifest_id for pin in pins} == {
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
    assert len(pins) == 2
    assert verification_ids == tuple(
        sorted({*verified_first.dependency_ids, *verified_second.dependency_ids})
    )
    assert task_family_ids == tuple(
        sorted((first_binding.task_family_id, second_binding.task_family_id))
    )


def test_adapter_context_allowlist_must_be_declared_by_the_exact_scope(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    known_dimension_id = stored.closure.derivation.trigger_packet.scope_contract.context_dimension_schemas[
        0
    ].dimension_id
    adapter_values = adapter.to_dict()
    adapter_values.pop("task_adapter_manifest_id")
    known_adapter = TaskAdapterManifest.mint(
        **{
            **adapter_values,
            "context_binding": TaskAdapterContextBinding(
                consumed_dimension_ids=(known_dimension_id,)
            ),
        }
    )

    pins, _, _ = ExpertCandidateEligibilityEvaluator._adapter_bindings(
        stored,
        (_verified_adapter(known_adapter),),
    )
    assert pins[0].task_adapter_manifest_id == known_adapter.task_adapter_manifest_id

    unknown_adapter = TaskAdapterManifest.mint(
        **{
            **adapter_values,
            "context_binding": TaskAdapterContextBinding(
                consumed_dimension_ids=("unknown_dimension",)
            ),
            "release_matrix_cases": (
                task_adapter_release_matrix_case(
                    scope_contract_id=adapter.scope_contract_id,
                    scope_id="ml_ai",
                    task_family_id=adapter.task_family_id,
                    task_adapter_id=adapter.task_adapter_id,
                    evaluator_fingerprint=_digest("source-evaluator"),
                    metric_directions=(("accuracy", ObjectiveDirection.MAXIMIZE),),
                    transfer_dimensions={"unknown_dimension": "fixture"},
                    label="unknown-dimension-adapter",
                ),
            ),
        }
    )
    with pytest.raises(ExpertValidationError, match="scope and trigger"):
        ExpertCandidateEligibilityEvaluator._adapter_bindings(
            stored,
            (_verified_adapter(unknown_adapter),),
        )


def test_adapter_release_matrix_cases_are_validated_against_the_exact_scope(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    base_case = adapter.release_matrix_cases[0]

    def adapter_with_context(**context_changes):
        context_values = base_case.task_context_binding.to_dict()
        context_values.pop("task_context_binding_id")
        context_values.update(context_changes)
        changed_case = type(base_case).mint(
            task_context_binding=type(base_case.task_context_binding).mint(
                **context_values
            ),
            independence_group=base_case.independence_group,
            evaluation_fingerprints=base_case.evaluation_fingerprints,
            starting_artifacts=base_case.starting_artifacts,
        )
        adapter_values = adapter.to_dict()
        adapter_values.pop("task_adapter_manifest_id")
        adapter_values["release_matrix_cases"] = (changed_case,)
        return TaskAdapterManifest.mint(**adapter_values)

    invalid_contexts = (
        ({"scope_id": "another_scope"}, "different scope"),
        (
            {"transfer_dimensions": {"dataset_family": "fixture"}},
            "missing=",
        ),
        (
            {
                "transfer_dimensions": {
                    "dataset_family": "fixture",
                    "runtime_family": "fixture",
                    "unknown_dimension": "fixture",
                }
            },
            "unknown=",
        ),
        (
            {
                "transfer_dimensions": {
                    "dataset_family": "fixture",
                    "runtime_family": 1,
                }
            },
            "must be a string",
        ),
    )
    for changes, message in invalid_contexts:
        with pytest.raises(CrossRunContractError, match=message):
            ExpertCandidateEligibilityEvaluator._adapter_bindings(
                stored,
                (_verified_adapter(adapter_with_context(**changes)),),
            )


def test_adapter_receipt_pins_attestation_without_changing_scientific_identity(
    tmp_path,
):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    rotated_adapter = replace(
        adapter,
        publisher_attestation={"issuer": "rotated", "signature": "new"},
    )

    original = _verified_adapter(adapter)
    rotated = _verified_adapter(rotated_adapter)

    assert (
        original.manifest.task_adapter_manifest_id
        == rotated.manifest.task_adapter_manifest_id
    )
    assert (
        original.verification_receipt.verification_receipt_id
        != rotated.verification_receipt.verification_receipt_id
    )
    assert (
        original.verification_receipt.publisher_attestation_digest
        != rotated.verification_receipt.publisher_attestation_digest
    )
    assert (
        original.verification_receipt.full_manifest_digest
        != rotated.verification_receipt.full_manifest_digest
    )


def test_reducer_replays_exact_adapter_pin_after_active_attestation_rotation(
    tmp_path,
):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    rotated_adapter = replace(
        adapter,
        publisher_attestation={"issuer": "rotated", "signature": "new"},
    )
    original = _verified_adapter(adapter)
    rotated = _verified_adapter(rotated_adapter)

    class _RotatingProvider:
        def __init__(self):
            self.active = original
            self.exact = {
                (
                    package.manifest.task_adapter_manifest_id,
                    package.verification_receipt.verification_receipt_id,
                ): package
                for package in (original, rotated)
            }

        def resolve_active(
            self,
            *,
            scope_contract_id,
            task_family_id,
            task_adapter_id,
        ):
            assert (
                scope_contract_id,
                task_family_id,
                task_adapter_id,
            ) == (
                adapter.scope_contract_id,
                adapter.task_family_id,
                adapter.task_adapter_id,
            )
            return self.active

        def resolve_exact(
            self,
            *,
            task_adapter_manifest_id,
            verification_receipt_id,
        ):
            return self.exact[(task_adapter_manifest_id, verification_receipt_id)]

    provider = _RotatingProvider()
    settings = _validation_settings()
    eligibility = ExpertCandidateEligibilityEvaluator(
        settings,
        store,
        provider,
        _CurrentReleaseProvider(None),
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    original_receipt_id = original.verification_receipt.verification_receipt_id
    assert eligibility.decision.task_adapter_pins[0].verification_receipt_id == (
        original_receipt_id
    )

    provider.active = rotated
    started = ExpertValidationReducer(
        settings,
        store,
        _AttestationVerifier(),
        provider,
        _CurrentReleaseProvider(None),
        _ValidationStateProvider(),
    ).start(eligibility=eligibility)

    assert started.attempt is not None
    assert started.attempt.task_adapter_pins == eligibility.decision.task_adapter_pins
    assert original_receipt_id in started.attempt.eligibility_dependency_ids


def test_adapter_receipt_must_match_the_full_manifest_and_proof_closure(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    verified = _verified_adapter(adapter)
    receipt_fields = {
        key: value
        for key, value in verified.verification_receipt.to_dict().items()
        if key != "verification_receipt_id"
    }

    with pytest.raises(ContractValidationError, match="differs from its manifest"):
        replace(
            verified,
            verification_receipt=TaskAdapterVerificationReceipt.mint(
                **{
                    **receipt_fields,
                    "full_manifest_digest": _digest("substituted-manifest"),
                }
            ),
        )
    with pytest.raises(ContractValidationError, match="normalized tar archive"):
        TaskAdapterVerificationReceipt.mint(
            **{
                **receipt_fields,
                "source_archive_ref": "../substituted.tar",
            }
        )
    with pytest.raises(ContractValidationError, match="differs from its manifest"):
        replace(
            verified,
            verification_receipt=TaskAdapterVerificationReceipt.mint(
                **{
                    **receipt_fields,
                    "proof_object_digests": {
                        adapter.sanitation_report_id: _digest("sanitation-only")
                    },
                }
            ),
        )


def test_active_adapter_resolution_cannot_redirect_a_trigger_binding(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    expected_adapter = _task_adapter(stored.closure)
    _, _, redirected_tree_hash = _adapter_source("redirected_adapter")
    redirected_adapter = TaskAdapterManifest.mint(
        task_adapter_id="redirected_adapter",
        scope_contract_id=expected_adapter.scope_contract_id,
        task_family_id=expected_adapter.task_family_id,
        publisher_attestation=expected_adapter.publisher_attestation,
        task_evaluator=expected_adapter.task_evaluator,
        context_binding=expected_adapter.context_binding,
        release_matrix_cases=(
            task_adapter_release_matrix_case(
                scope_contract_id=expected_adapter.scope_contract_id,
                scope_id=stored.closure.derivation.trigger_packet.scope_contract.scope_id,
                task_family_id=expected_adapter.task_family_id,
                task_adapter_id="redirected_adapter",
                evaluator_fingerprint=_digest("source-evaluator"),
                metric_directions=(("accuracy", ObjectiveDirection.MAXIMIZE),),
                transfer_dimensions={
                    schema.dimension_id: "fixture"
                    for schema in stored.closure.derivation.trigger_packet.scope_contract.context_dimension_schemas
                },
                label="redirected-adapter",
            ),
        ),
        source_tree_ref=expected_adapter.source_tree_ref,
        tree_hash=redirected_tree_hash,
        runtime=expected_adapter.runtime,
        sanitation_report_id=expected_adapter.sanitation_report_id,
        validation_refs=expected_adapter.validation_refs,
    )

    class _RedirectingProvider:
        def resolve_active(
            self,
            *,
            scope_contract_id,
            task_family_id,
            task_adapter_id,
        ):
            assert (
                scope_contract_id,
                task_family_id,
                task_adapter_id,
            ) == (
                expected_adapter.scope_contract_id,
                expected_adapter.task_family_id,
                expected_adapter.task_adapter_id,
            )
            return _verified_adapter(redirected_adapter)

        def resolve_exact(
            self,
            *,
            task_adapter_manifest_id,
            verification_receipt_id,
        ):
            raise AssertionError("exact replay resolution is not enrollment")

    evaluator = ExpertCandidateEligibilityEvaluator(
        _validation_settings(),
        store,
        _RedirectingProvider(),
        _CurrentReleaseProvider(None),
    )
    with pytest.raises(ExpertValidationError, match="trigger bindings"):
        evaluator.decide(candidate_id=stored.closure.manifest.candidate_id)


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
    )

    assert stale.decision.eligible is False
    assert stale.decision.reason_code == "source_base_not_current"
    assert stale.decision.required_stages == ()

    valid = _eligibility_evaluator(settings, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
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
        _validation_reducer(
            settings,
            adapter,
            candidate_store=store,
        ).start(
            eligibility=replace(valid, decision=forged_decision),
        )


def test_bounded_evaluator_result_advances_only_the_exact_next_stage(tmp_path):
    store = candidate_store(tmp_path)
    stored = store.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    settings = _validation_settings()
    eligibility = _eligibility_evaluator(settings, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
    )
    started = _validation_reducer(
        settings,
        adapter,
        candidate_store=store,
    ).start(
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

    advanced = _validation_reducer(settings, adapter).advance_evaluator_stage(
        state=started.state,
        attempt=started.attempt,
        accepted_results=(),
        result=result,
    )

    assert len(advanced.accepted_stage_results) == 1
    assert advanced.accepted_stage_results[0] == ExpertAcceptedStageResultRef(
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        stage_result_record_id=result.evaluator_result_record_id,
    )
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
        _validation_reducer(settings, adapter).advance_evaluator_stage(
            state=advanced,
            attempt=started.attempt,
            accepted_results=(),
            result=second_result,
        )
    twice_advanced = _validation_reducer(settings, adapter).advance_evaluator_stage(
        state=advanced,
        attempt=started.attempt,
        accepted_results=(result,),
        result=second_result,
    )
    assert (
        twice_advanced.next_stage is ExpertValidationStage.STATIC_UNIT_SECURITY_RESOURCE
    )

    invalid_signature = ExpertEvaluatorResultRecord.mint(
        evaluator_run=result.evaluator_run,
        attestation_envelope=replace(
            result.attestation_envelope,
            signature="invalid",
        ),
    )
    with pytest.raises(ExpertValidationError, match="invalid test signature"):
        _validation_reducer(settings, adapter).advance_evaluator_stage(
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
        _validation_reducer(settings, adapter).advance_evaluator_stage(
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
    )
    reducer = _validation_reducer(settings, adapter, candidate_store=store)
    started = reducer.start(
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
    failed = reducer.advance_evaluator_stage(
        state=started.state,
        attempt=started.attempt,
        accepted_results=(),
        result=failed_result,
    )

    assert failed.promotion_state is ExpertPromotionState.FAILED
    assert failed.next_stage is None
    with pytest.raises(ExpertValidationError, match="active attempt"):
        reducer.advance_evaluator_stage(
            state=failed,
            attempt=started.attempt,
            accepted_results=(),
            result=failed_result,
        )

    retry_reducer = _validation_reducer(
        settings,
        adapter,
        candidate_store=store,
        predecessor=ExpertValidationPredecessor(
            latest_attempt=started.attempt,
            state=failed,
        ),
    )
    retry = retry_reducer.start(
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
    )
    initial = _validation_reducer(
        settings,
        adapter,
        candidate_store=store,
    ).start(
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
    failed = _validation_reducer(settings, adapter).advance_evaluator_stage(
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
    )
    ineligible = _validation_reducer(
        settings,
        adapter,
        candidate_store=store,
        current_release_id=current_release_id,
        predecessor=historical_attempt,
    ).start(
        eligibility=stale,
    )
    assert ineligible.attempt is None

    retry = _validation_reducer(
        settings,
        adapter,
        candidate_store=store,
        predecessor=ExpertValidationPredecessor(
            latest_attempt=initial.attempt,
            state=ineligible.state,
        ),
    ).start(
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
        policy=replace(
            settings.policy,
            artifact_byte_limit=1,
            task_evaluation_result_byte_limit=1,
        ),
    )
    eligibility = _eligibility_evaluator(limited, store, adapter).decide(
        candidate_id=stored.closure.manifest.candidate_id,
    )
    started = _validation_reducer(
        limited,
        adapter,
        candidate_store=store,
    ).start(
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
