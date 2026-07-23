import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from pathlib import Path

import pytest

from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
)
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    ContractValidationError,
    ExpertAcceptedStageResultRef,
    ExpertCandidateCommitRecord,
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
    ExpertCandidateValidationState,
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertReviewDisposition,
    ExpertValidationAttempt,
    ExpertValidationStage,
    ExpertValidationTrack,
)
from kapso.cross_run.expert.review import (
    ExpertAutomatedReviewCoordinator,
    ExpertAutomatedReviewError,
    ExpertAutomatedReviewExecution,
    PreparedExpertAutomatedReviewPacket,
    build_expert_automated_review_stage_result,
    expert_candidate_review_derivation_evidence_ids,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionDisposition,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewAdjudication,
    ExpertAutomatedReviewOperationRecord,
    ExpertAutomatedReviewOutcome,
    ExpertAutomatedReviewPacket,
)
from kapso.cross_run.expert.review_stage import (
    ExpertAutomatedReviewStageOrchestrator,
)
from kapso.cross_run.expert.candidates import ExpertCandidateValidator
from kapso.cross_run.expert.candidate_derivations import (
    ExpertDeterministicCompositionDerivationRecord,
)
from kapso.cross_run.expert.proposal_contract import (
    mint_expert_candidate_ancestor_input,
)
from kapso.cross_run.expert.sanitation import ExpertCandidateSanitizer
from kapso.cross_run.expert.store import (
    ExpertCandidateStore,
    stored_candidate_admission_dependency_ids,
)
from kapso.cross_run.expert.validation_store import (
    ExpertValidationStore,
    ExpertValidationStoreError,
)
from test_expert_candidates import (
    bootstrap_candidate_closure,
    expert_settings,
    sanitation_settings,
)
from test_expert_composition_contracts import (
    _assessment,
    _materialization,
    _plan,
    _source_reference,
    composition_case,
)
from test_expert_clean_recovery import _historical_candidate_system
from test_expert_validation import (
    _eligibility_evaluator,
    _task_adapter,
    _validation_reducer,
    _validation_settings,
)
from test_expert_validation_store import _result


class AutomatedReviewProcess:
    def __init__(
        self,
        outputs: dict[str, dict],
    ):
        self.outputs = outputs
        self.calls = []

    def __call__(
        self,
        command,
        *,
        cwd,
        env,
        input,
        text,
        capture_output,
        check,
        pass_fds,
    ):
        assert text is True
        assert capture_output is True
        assert check is False
        assert pass_fds == ()
        assert env
        reviewer_roles = tuple(
            role for role in self.outputs if f'"reviewer_role":"{role}"' in input
        )
        assert len(reviewer_roles) == 1
        output = json.dumps(self.outputs[reviewer_roles[0]], sort_keys=True)
        final_path = Path(command[command.index("--output-last-message") + 1])
        final_path.write_text(output, encoding="utf-8")
        self.calls.append(
            {
                "command": tuple(command),
                "prompt": input,
                "workspace": Path(cwd),
            }
        )
        stdout = json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=stdout + "\n",
            stderr="",
        )


def _remint(record, **changes):
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name != record.IDENTITY_FIELD
    }
    payload.update(changes)
    return type(record).mint(**payload)


def _review_fixture(
    tmp_path,
    monkeypatch,
    outputs=None,
    artifact_byte_limit=None,
):
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    configured_expert_settings = expert_settings()
    if artifact_byte_limit is not None:
        configured_expert_settings = replace(
            configured_expert_settings,
            agent_artifact_byte_limit=artifact_byte_limit,
        )
    metadata_root = tmp_path / ".kapso"
    metadata_root.mkdir(mode=0o700)
    cross_run_root = metadata_root / "cross_run"
    cross_run_root.mkdir(mode=0o700)
    candidates = ExpertCandidateStore(
        (tmp_path / configured_expert_settings.candidate_path).resolve(),
        cross_run_root.resolve(),
        ExpertCandidateValidator(
            configured_expert_settings,
            sanitation_settings(),
        ),
    )
    stored = candidates.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    validation_settings = configured_expert_settings.validation
    eligibility = _eligibility_evaluator(
        validation_settings,
        candidates,
        adapter,
        None,
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    reducer = _validation_reducer(
        validation_settings,
        adapter,
        candidate_store=candidates,
    )
    store = ExpertValidationStore(
        (tmp_path / validation_settings.state_path).resolve(),
        cross_run_root.resolve(),
        validation_settings,
        reducer,
    )
    snapshot = store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    assert snapshot.latest_attempt is not None
    while snapshot.state.next_stage is not ExpertValidationStage.AUTOMATED_REVIEW:
        stage = snapshot.state.next_stage
        assert stage is not None
        result = _result(
            validation_settings,
            snapshot.latest_attempt,
            stage,
            ExpertEvaluatorOutcome.PASSED,
        )
        snapshot = store.publish_evaluator_result(
            candidate_id=snapshot.state.candidate_id,
            expected_transition_id=snapshot.transition.transition_id,
            result=result,
        ).snapshot
    configured_outputs = outputs or {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.CORE_ELIGIBLE.value,
            "judgment": validation_settings.policy.promotion.approval_judgment,
            "rationale": f"{reviewer.reviewer_id} found complete reusable evidence.",
        }
        for reviewer in validation_settings.policy.reviewers
    }
    runner = AutomatedReviewProcess(configured_outputs)
    monkeypatch.setattr(
        "kapso.execution.coding_agents.structured_call.subprocess.run",
        runner,
    )
    coordinator = ExpertAutomatedReviewCoordinator(
        configured_expert_settings,
        tmp_path.resolve(),
    )
    store._bind_automated_review_publication_authority(coordinator)
    prepared = coordinator.prepare(
        stored_candidate=stored,
        validation_attempt=snapshot.latest_attempt,
        authorization_transition_id=snapshot.transition.transition_id,
        authorization_state=snapshot.state,
        accepted_stage_results=snapshot.accepted_stage_results,
    )
    workspace = (tmp_path / "review-workspace").resolve()
    workspace.mkdir(mode=0o700)
    return coordinator, prepared, workspace, runner, snapshot, store


def _reopen_validation_store(store):
    return ExpertValidationStore(
        store.root,
        store.state_root,
        store.settings,
        store.reducer,
    )


def _composition_review_fixture(
    tmp_path,
    monkeypatch,
):
    review_root = tmp_path / "review"
    (
        coordinator,
        direct_prepared,
        workspace,
        runner,
        _snapshot,
        direct_store,
    ) = _review_fixture(review_root, monkeypatch)
    case = composition_case.__wrapped__()
    source = _source_reference(case.scope, case.module, label="review composition")
    plan = _plan(case.scope, case.base, (source,))
    assessment = _assessment(
        plan,
        ExpertCompositionDisposition.CLEAN,
        applicable=(source.source_reference_id,),
    )
    materialization = _materialization(case, assessment)
    source_context_ids = {
        source.candidate_id: source.validation_context_ref,
    }
    source_dependencies = tuple(
        sorted(
            {
                plan.composition_plan_id,
                *plan.stable_authority_ids,
                *source_context_ids.values(),
            }
        )
    )
    derivation_record = ExpertDeterministicCompositionDerivationRecord.mint(
        composition_materialization_id=materialization.materialization_id,
        source_validation_context_ids=source_context_ids,
        source_origin_principal_ids={
            source.candidate_id: source.origin_principal_ids,
        },
        source_dependency_ids=source_dependencies,
    )
    candidate_contents = dict(case.source_base_contents)
    candidate_contents["src/reproducible_execution/__init__.py"] = b"changed source"
    sanitation = ExpertCandidateSanitizer(sanitation_settings()).scan(
        case.scope.scope_contract_id,
        materialization.source_tree,
        candidate_contents,
    )
    manifest = ExpertCandidateManifest.mint(
        scope_contract_id=case.scope.scope_contract_id,
        change_kind=CandidateChangeKind.CAPABILITY,
        source_base_release_id=case.base.release_id,
        source_base_repository_map_ref=case.base.repository_map_id,
        source_base_tree_hash=case.base.source_tree_hash,
        consumed_expert_release_ids=(case.base.release_id,),
        derivation_kind=ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION,
        derivation_ref=derivation_record.derivation_id,
        validation_context_ref=source.validation_context_ref,
        patch_ref=materialization.patch.patch_id,
        patch_digest=tree_or_blob_digest(materialization.patch.to_json_bytes()),
        candidate_tree_ref=materialization.source_tree.source_tree_manifest_id,
        candidate_tree_hash=materialization.source_tree.tree_hash,
        configuration_fingerprint=plan.configuration_fingerprint,
        module_contract_refs=tuple(
            module.module_contract_id for module in materialization.module_contracts
        ),
        proposed_repository_map_ref=(materialization.repository_map.repository_map_id),
        semantic_book_digest=materialization.semantic_book_digest,
        source_dependency_ids=source_dependencies,
        ancestor_candidate_ids=(source.candidate_id,),
        capability_lineage=(),
        sanitation_report_id=sanitation.sanitation_report_id,
    )
    candidate_input = mint_expert_candidate_ancestor_input(
        manifest=manifest,
        scope_contract=case.scope,
        patch=materialization.patch,
        candidate_tree=materialization.source_tree,
        repository_map=materialization.repository_map,
        module_contracts=materialization.module_contracts,
        sanitation_report=sanitation,
        candidate_contents=candidate_contents,
    )
    commit = ExpertCandidateCommitRecord.mint(
        candidate_id=manifest.candidate_id,
        file_checksums={
            "candidate.json": tree_or_blob_digest(manifest.to_json_bytes()),
            "review-input.json": tree_or_blob_digest(candidate_input.to_json_bytes()),
        },
    )
    policy = coordinator.settings.validation.policy.validation_policy()
    eligibility_id = content_id(
        "expert-candidate-eligibility",
        {"candidate_id": manifest.candidate_id},
    )
    task_adapter_pins = direct_prepared.validation_attempt.task_adapter_pins
    eligibility_dependencies = {
        manifest.candidate_id,
        commit.commit_record_id,
        manifest.scope_contract_id,
        eligibility_id,
        policy.validation_policy_id,
        manifest.source_base_release_id,
        *(pin.task_adapter_manifest_id for pin in task_adapter_pins),
        *(pin.verification_receipt_id for pin in task_adapter_pins),
    }
    attempt = ExpertValidationAttempt.mint(
        candidate_id=manifest.candidate_id,
        candidate_tree_hash=manifest.candidate_tree_hash,
        candidate_commit_record_id=commit.commit_record_id,
        scope_contract_id=manifest.scope_contract_id,
        source_base_release_id=manifest.source_base_release_id,
        eligibility_decision_id=eligibility_id,
        validation_policy_id=policy.validation_policy_id,
        configuration_fingerprint=(
            coordinator.settings.validation.configuration_fingerprint
        ),
        validation_track=ExpertValidationTrack.BEHAVIORAL_CAPABILITY,
        attempt_number=1,
        predecessor_attempt_id=None,
        required_stages=(
            ExpertValidationStage.CONTRACT_SCHEMA,
            ExpertValidationStage.AUTOMATED_REVIEW,
        ),
        configured_task_family_ids=tuple(
            sorted({binding.task_family_id for binding in plan.active_task_bindings})
        ),
        task_adapter_pins=task_adapter_pins,
        source_replay_selection=None,
        eligibility_dependency_ids=tuple(sorted(eligibility_dependencies)),
    )
    accepted_result = _result(
        coordinator.settings.validation,
        attempt,
        ExpertValidationStage.CONTRACT_SCHEMA,
        ExpertEvaluatorOutcome.PASSED,
    )
    accepted_reference = ExpertAcceptedStageResultRef(
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        stage_result_record_id=accepted_result.evaluator_result_record_id,
    )
    authorization_state = ExpertCandidateValidationState.mint(
        validation_attempt_id=attempt.validation_attempt_id,
        candidate_id=attempt.candidate_id,
        candidate_tree_hash=attempt.candidate_tree_hash,
        predecessor_state_id=None,
        promotion_state=ExpertPromotionState.VALIDATING,
        accepted_stage_results=(accepted_reference,),
        next_stage=ExpertValidationStage.AUTOMATED_REVIEW,
        review_assertion_ids=(),
        terminal_evidence_ids=(),
        transition_evidence_id=accepted_result.evaluator_result_record_id,
        reason="composition_ready_for_automated_review",
    )
    derivation_evidence_ids = expert_candidate_review_derivation_evidence_ids(
        derivation_record=derivation_record,
        candidate_operation=None,
        composition_materialization=materialization,
        recovery_replay_basis=None,
    )
    authorization_transition_id = content_id(
        "expert-validation-transition",
        {"candidate_id": manifest.candidate_id},
    )
    packet_dependencies = {
        attempt.validation_attempt_id,
        authorization_transition_id,
        authorization_state.validation_state_id,
        manifest.candidate_id,
        commit.commit_record_id,
        candidate_input.ancestor_input_id,
        derivation_record.derivation_id,
        manifest.scope_contract_id,
        policy.validation_policy_id,
        manifest.source_base_release_id,
        accepted_result.evaluator_result_record_id,
        *derivation_evidence_ids,
    }
    packet = ExpertAutomatedReviewPacket.mint(
        validation_attempt_id=attempt.validation_attempt_id,
        authorization_transition_id=authorization_transition_id,
        authorization_state_id=authorization_state.validation_state_id,
        candidate_id=manifest.candidate_id,
        candidate_tree_hash=manifest.candidate_tree_hash,
        candidate_commit_record_id=commit.commit_record_id,
        candidate_input_id=candidate_input.ancestor_input_id,
        candidate_derivation_kind=manifest.derivation_kind,
        candidate_derivation_ref=manifest.derivation_ref,
        candidate_origin_principal_ids=derivation_record.origin_principal_ids,
        candidate_derivation_evidence_ids=derivation_evidence_ids,
        scope_contract_id=manifest.scope_contract_id,
        source_base_release_id=manifest.source_base_release_id,
        validation_policy_id=policy.validation_policy_id,
        configuration_fingerprint=attempt.configuration_fingerprint,
        agent_artifact_byte_limit=coordinator.settings.agent_artifact_byte_limit,
        accepted_stage_results=(accepted_reference,),
        exact_dependency_ids=tuple(sorted(packet_dependencies)),
    )
    prepared = PreparedExpertAutomatedReviewPacket(
        packet=packet,
        candidate_input=candidate_input,
        candidate_derivation_record=derivation_record,
        candidate_operation=None,
        composition_materialization=materialization,
        recovery_replay_basis=None,
        validation_attempt=attempt,
        authorization_state=authorization_state,
        validation_policy=policy,
        accepted_stage_results=(accepted_result,),
    )
    return (
        coordinator,
        prepared,
        workspace,
        runner,
        direct_prepared.candidate_operation,
        direct_store,
    )


def test_review_packet_is_complete_and_passes_only_unanimous_clean_review(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, workspace, runner, snapshot, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )

    execution = coordinator.execute(prepared, workspace=workspace)

    assert execution.stage_result.outcome is ExpertAutomatedReviewOutcome.PASSED
    assert execution.adjudication.outcome is ExpertAutomatedReviewOutcome.PASSED
    assert prepared.packet.candidate_derivation_kind is (
        ExpertCandidateDerivationKind.AGENT_PROPOSAL
    )
    assert prepared.candidate_operation is not None
    assert prepared.composition_materialization is None
    assert (
        prepared.packet.candidate_derivation_ref
        == prepared.candidate_derivation_record.derivation_id
    )
    assert len(execution.assertions) == len(
        coordinator.settings.validation.policy.reviewers
    )
    assert tuple(
        assertion.exact_evidence_ids for assertion in execution.assertions
    ) == (prepared.packet.evidence_ids,) * len(execution.assertions)
    assert all(
        operation.operation_receipt.workspace_access
        is CodingAgentWorkspaceAccess.READ_ONLY
        for operation in execution.operation_records
    )
    assert all(call["workspace"] == workspace for call in runner.calls)
    assert Path(coordinator.runner.settings.artifact_root) == (
        coordinator.workspace_root / coordinator.settings.agent_artifact_path
    )
    assert Path(coordinator.runner.settings.artifact_root).parent == store.root.parent
    source = prepared.candidate_input.candidate_contents_text["src/execution.py"]
    assert json.dumps(source)[1:-1] in runner.calls[0]["prompt"]
    assert (
        ExpertAutomatedReviewOperationRecord.from_json_bytes(
            execution.operation_records[0].to_json_bytes()
        )
        == execution.operation_records[0]
    )
    assert len(snapshot.accepted_stage_results) == len(
        prepared.packet.accepted_stage_results
    )


def test_composition_review_uses_materialization_without_fake_agent_proposal(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, workspace, runner, _source_operation, _store = (
        _composition_review_fixture(
            tmp_path,
            monkeypatch,
        )
    )

    execution = coordinator.execute(prepared, workspace=workspace)
    prompt_payload = coordinator._prompt_payload(
        prepared,
        coordinator.settings.validation.policy.reviewers[0],
    )
    derivation_payload = prompt_payload["candidate_derivation"]
    materialization = prepared.composition_materialization

    assert prepared.packet.candidate_derivation_kind is (
        ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION
    )
    assert (
        ExpertAutomatedReviewPacket.from_json_bytes(prepared.packet.to_json_bytes())
        == prepared.packet
    )
    assert prepared.candidate_operation is None
    assert materialization is not None
    assert derivation_payload["derivation_kind"] == "deterministic_composition"
    assert derivation_payload["composition_materialization"] == (
        materialization.to_dict()
    )
    assert "authoring_operation" not in derivation_payload
    assert "candidate_proposer" not in prompt_payload
    assert prepared.packet.candidate_origin_principal_ids == (
        prepared.candidate_derivation_record.origin_principal_ids
    )
    assert {
        source.candidate_commit_record_id
        for source in (materialization.composition_assessment.composition_plan.sources)
    }.issubset(prepared.packet.candidate_derivation_evidence_ids)
    assert execution.stage_result.outcome is ExpertAutomatedReviewOutcome.PASSED
    assert len(runner.calls) == len(coordinator.settings.validation.policy.reviewers)
    assert all("candidate_proposer" not in call["prompt"] for call in runner.calls)


def test_composition_review_rejects_agent_operation_substitution(
    tmp_path,
    monkeypatch,
):
    _coordinator, prepared, _workspace, _runner, source_operation, _store = (
        _composition_review_fixture(
            tmp_path,
            monkeypatch,
        )
    )
    assert source_operation is not None

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="composition review derivation evidence is inconsistent",
    ):
        replace(prepared, candidate_operation=source_operation)


def test_composition_review_rejects_source_authority_substitution(
    tmp_path,
    monkeypatch,
):
    _coordinator, prepared, _workspace, _runner, _source_operation, _store = (
        _composition_review_fixture(
            tmp_path,
            monkeypatch,
        )
    )
    record = prepared.candidate_derivation_record
    source_candidate_id = record.ancestor_candidate_ids[0]
    forged_record = _remint(
        record,
        source_origin_principal_ids={
            source_candidate_id: ("forged.origin.author",),
        },
    )

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="prepared automated review packet closure is inconsistent",
    ):
        replace(prepared, candidate_derivation_record=forged_record)


def test_composition_review_derivation_persists_reopens_and_detects_tampering(
    tmp_path,
    monkeypatch,
):
    _coordinator, prepared, _workspace, _runner, _source_operation, store = (
        _composition_review_fixture(
            tmp_path,
            monkeypatch,
        )
    )
    with store._lock(exclusive=True):
        store._write_automated_review_derivation_unlocked(prepared)

    reopened = _reopen_validation_store(store)
    with reopened._lock(exclusive=False):
        derivation_record, operation, materialization, recovery_replay_basis = (
            reopened._read_automated_review_derivation_unlocked(prepared.packet)
        )

    assert derivation_record == prepared.candidate_derivation_record
    assert operation is None
    assert materialization == prepared.composition_materialization
    assert recovery_replay_basis is None
    assert (
        replace(
            prepared,
            candidate_derivation_record=derivation_record,
            candidate_operation=operation,
            composition_materialization=materialization,
        )
        == prepared
    )

    assert materialization is not None
    materialization_path = reopened._object_path(
        materialization.materialization_id,
        create_namespace=False,
    )
    materialization_path.write_bytes(b"{}\n")
    with reopened._lock(exclusive=False):
        with pytest.raises(ContractValidationError):
            reopened._read_automated_review_derivation_unlocked(prepared.packet)


@pytest.mark.parametrize("empty_recovery", (False, True))
def test_recovery_review_derivation_persists_reopens_and_detects_tampering(
    tmp_path,
    monkeypatch,
    empty_recovery,
):
    (
        coordinator,
        direct_prepared,
        _workspace,
        _runner,
        _snapshot,
        store,
    ) = _review_fixture(tmp_path / "review", monkeypatch)
    recovery = _historical_candidate_system(
        tmp_path / "recovery",
        empty_selection=empty_recovery,
    )
    if empty_recovery:
        stored = recovery.coordinator.bootstrap_empty(
            scope_contract=recovery.fixture.case.scope,
            replay_basis_packet=recovery.replay_basis,
        ).stored_candidate
    else:
        stored = recovery.coordinator.restore_historical(
            scope_contract=recovery.fixture.case.scope,
            replay_basis_packet=recovery.replay_basis,
        )
    manifest = stored.closure.manifest
    commit = stored.commit_record
    policy = coordinator.settings.validation.policy.validation_policy()
    eligibility_id = content_id(
        "expert-candidate-eligibility",
        {"candidate_id": manifest.candidate_id, "recovery": True},
    )
    pins = direct_prepared.validation_attempt.task_adapter_pins
    eligibility_dependencies = {
        manifest.candidate_id,
        commit.commit_record_id,
        manifest.scope_contract_id,
        eligibility_id,
        policy.validation_policy_id,
        *(pin.task_adapter_manifest_id for pin in pins),
        *(pin.verification_receipt_id for pin in pins),
        *stored_candidate_admission_dependency_ids(stored),
    }
    if manifest.source_base_release_id is not None:
        eligibility_dependencies.add(manifest.source_base_release_id)
    attempt = ExpertValidationAttempt.mint(
        candidate_id=manifest.candidate_id,
        candidate_tree_hash=manifest.candidate_tree_hash,
        candidate_commit_record_id=commit.commit_record_id,
        scope_contract_id=manifest.scope_contract_id,
        source_base_release_id=manifest.source_base_release_id,
        eligibility_decision_id=eligibility_id,
        validation_policy_id=policy.validation_policy_id,
        configuration_fingerprint=(
            coordinator.settings.validation.configuration_fingerprint
        ),
        validation_track=(
            ExpertValidationTrack.REPOSITORY_ARCHITECTURE
            if empty_recovery
            else ExpertValidationTrack.BEHAVIORAL_CAPABILITY
        ),
        attempt_number=1,
        predecessor_attempt_id=None,
        required_stages=(
            ExpertValidationStage.CONTRACT_SCHEMA,
            ExpertValidationStage.AUTOMATED_REVIEW,
        ),
        configured_task_family_ids=tuple(
            sorted(
                {
                    binding.task_family_id
                    for binding in recovery.replay_basis.active_task_bindings
                }
            )
        ),
        task_adapter_pins=pins,
        source_replay_selection=None,
        eligibility_dependency_ids=tuple(sorted(eligibility_dependencies)),
    )
    accepted_result = _result(
        coordinator.settings.validation,
        attempt,
        ExpertValidationStage.CONTRACT_SCHEMA,
        ExpertEvaluatorOutcome.PASSED,
    )
    accepted_reference = ExpertAcceptedStageResultRef(
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        stage_result_record_id=accepted_result.evaluator_result_record_id,
    )
    authorization_state = ExpertCandidateValidationState.mint(
        validation_attempt_id=attempt.validation_attempt_id,
        candidate_id=attempt.candidate_id,
        candidate_tree_hash=attempt.candidate_tree_hash,
        predecessor_state_id=None,
        promotion_state=ExpertPromotionState.VALIDATING,
        accepted_stage_results=(accepted_reference,),
        next_stage=ExpertValidationStage.AUTOMATED_REVIEW,
        review_assertion_ids=(),
        terminal_evidence_ids=(),
        transition_evidence_id=accepted_result.evaluator_result_record_id,
        reason="recovery_ready_for_automated_review",
    )
    prepared = coordinator.prepare(
        stored_candidate=stored,
        validation_attempt=attempt,
        authorization_transition_id=content_id(
            "expert-validation-transition",
            {"candidate_id": manifest.candidate_id, "recovery": True},
        ),
        authorization_state=authorization_state,
        accepted_stage_results=(accepted_result,),
    )
    prompt_payload = coordinator._prompt_payload(
        prepared,
        coordinator.settings.validation.policy.reviewers[0],
    )
    assert prompt_payload["candidate_derivation"]["derivation_kind"] == (
        manifest.derivation_kind.value
    )
    if empty_recovery:
        assert "authoring_operation" in prompt_payload["candidate_derivation"]
        assert "replay_basis_packet" not in prompt_payload["candidate_derivation"]
    else:
        assert prompt_payload["candidate_derivation"]["replay_basis_packet"] == (
            recovery.replay_basis.to_dict()
        )
    with store._lock(exclusive=True):
        store._write_automated_review_derivation_unlocked(prepared)

    reopened = _reopen_validation_store(store)
    with reopened._lock(exclusive=False):
        derivation_record, operation, materialization, replay_basis = (
            reopened._read_automated_review_derivation_unlocked(prepared.packet)
        )
    assert derivation_record == prepared.candidate_derivation_record
    assert operation == prepared.candidate_operation
    assert materialization is None
    assert replay_basis == prepared.recovery_replay_basis
    assert (
        replace(
            prepared,
            candidate_derivation_record=derivation_record,
            candidate_operation=operation,
            composition_materialization=materialization,
            recovery_replay_basis=replay_basis,
        )
        == prepared
    )

    tamper_id = (
        operation.operation_record_id
        if operation is not None
        else recovery.replay_basis.evidence_packet_id
    )
    replay_path = reopened._object_path(
        tamper_id,
        create_namespace=False,
    )
    replay_path.write_bytes(b"{}\n")
    with reopened._lock(exclusive=False):
        with pytest.raises(ContractValidationError):
            reopened._read_automated_review_derivation_unlocked(prepared.packet)


def test_mixed_review_is_disputed_even_when_rejection_quorum_is_met(
    tmp_path,
    monkeypatch,
):
    validation_settings = _validation_settings()
    reviewers = validation_settings.policy.reviewers
    outputs = {
        reviewers[0].reviewer_role: {
            "disposition": ExpertReviewDisposition.CORE_ELIGIBLE.value,
            "judgment": validation_settings.policy.promotion.approval_judgment,
            "rationale": "Reusable and causally supported.",
        },
        reviewers[1].reviewer_role: {
            "disposition": ExpertReviewDisposition.TASK_SPECIFIC.value,
            "judgment": validation_settings.policy.promotion.rejection_judgment,
            "rationale": "Evidence does not establish transfer.",
        },
    }
    coordinator, prepared, workspace, _, _, _ = _review_fixture(
        tmp_path / "mixed",
        monkeypatch,
        outputs,
    )

    execution = coordinator.execute(prepared, workspace=workspace)

    assert execution.adjudication.outcome is ExpertAutomatedReviewOutcome.DISPUTED
    assert execution.stage_result.outcome is ExpertAutomatedReviewOutcome.DISPUTED


def test_clean_rejection_is_rejected(tmp_path, monkeypatch):
    validation_settings = _validation_settings()
    outputs = {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.UNSAFE_OR_SPECIALIZED.value,
            "judgment": validation_settings.policy.promotion.rejection_judgment,
            "rationale": "The candidate encodes a specialized unsafe shortcut.",
        }
        for reviewer in validation_settings.policy.reviewers
    }
    coordinator, prepared, workspace, _, _, _ = _review_fixture(
        tmp_path / "rejected",
        monkeypatch,
        outputs,
    )

    execution = coordinator.execute(prepared, workspace=workspace)

    assert execution.adjudication.outcome is ExpertAutomatedReviewOutcome.REJECTED
    assert execution.stage_result.outcome is ExpertAutomatedReviewOutcome.REJECTED


def test_review_rejects_judgment_disposition_conflict(tmp_path, monkeypatch):
    validation_settings = _validation_settings()
    outputs = {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.TASK_SPECIFIC.value,
            "judgment": validation_settings.policy.promotion.approval_judgment,
            "rationale": "Contradictory output.",
        }
        for reviewer in validation_settings.policy.reviewers
    }
    coordinator, prepared, workspace, _, _, _ = _review_fixture(
        tmp_path / "conflict",
        monkeypatch,
        outputs,
    )

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="judgment and disposition conflict",
    ):
        coordinator.execute(prepared, workspace=workspace)


def test_review_rejects_nonempty_or_shared_workspace(tmp_path, monkeypatch):
    coordinator, prepared, workspace, _, _, _ = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    (workspace / "unexpected.txt").write_text("not packet-only", encoding="utf-8")

    with pytest.raises(ExpertAutomatedReviewError, match="empty normalized"):
        coordinator.execute(prepared, workspace=workspace)

    (workspace / "unexpected.txt").unlink()
    workspace.chmod(0o755)
    with pytest.raises(ExpertAutomatedReviewError, match="must be private"):
        coordinator.execute(prepared, workspace=workspace)


def test_review_operation_artifact_tampering_fails_loud(tmp_path, monkeypatch):
    coordinator, prepared, workspace, _, _, _ = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    execution = coordinator.execute(prepared, workspace=workspace)
    operation = execution.operation_records[0]
    payloads = dict(operation.artifact_payloads_base64)
    payloads["prompt.txt"] = payloads["final.json"]

    with pytest.raises(ContractValidationError, match="artifacts differ"):
        replace(operation, artifact_payloads_base64=payloads)


def test_review_rejects_a_prompt_over_the_configured_artifact_limit(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, workspace, runner, _, _ = _review_fixture(
        tmp_path,
        monkeypatch,
        artifact_byte_limit=30_000,
    )

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="prompt exceeds the configured artifact limit",
    ):
        coordinator.execute(prepared, workspace=workspace)

    assert runner.calls == []


def test_review_rejects_an_artifact_closure_over_the_configured_limit(
    tmp_path,
    monkeypatch,
):
    validation_settings = _validation_settings()
    outputs = {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.CORE_ELIGIBLE.value,
            "judgment": validation_settings.policy.promotion.approval_judgment,
            "rationale": "x" * 10_000,
        }
        for reviewer in validation_settings.policy.reviewers
    }
    coordinator, prepared, workspace, runner, _, _ = _review_fixture(
        tmp_path,
        monkeypatch,
        outputs,
        artifact_byte_limit=40_000,
    )

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="artifacts exceed the configured limit",
    ):
        coordinator.execute(prepared, workspace=workspace)

    assert len(runner.calls) == 1


def test_review_execution_cannot_be_constructed_without_coordinator_authority(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, workspace, _, _, _ = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    execution = coordinator.execute(prepared, workspace=workspace)

    with pytest.raises(ExpertAutomatedReviewError, match="not coordinator sealed"):
        ExpertAutomatedReviewExecution(
            object(),
            coordinator,
            prepared_packet=prepared,
            assertions=execution.assertions,
            operation_records=execution.operation_records,
            adjudication=execution.adjudication,
            stage_result=execution.stage_result,
        )


def test_review_publication_requires_the_bound_coordinator(tmp_path, monkeypatch):
    coordinator, prepared, workspace, _, _, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    execution = coordinator.execute(prepared, workspace=workspace)
    unbound_store = _reopen_validation_store(store)

    with pytest.raises(
        ExpertValidationStoreError,
        match="lacks bound coordinator authority",
    ):
        unbound_store.publish_automated_review_stage(execution)


@pytest.mark.parametrize("substitution", ("runner", "run_method"))
def test_review_publication_rejects_a_substituted_synthetic_runner(
    tmp_path,
    monkeypatch,
    substitution,
):
    coordinator, prepared, workspace, _, _, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    execution = coordinator.execute(prepared, workspace=workspace)
    synthetic = AutomatedReviewProcess({})
    if substitution == "runner":
        coordinator.runner = synthetic
    else:
        coordinator.runner.run = synthetic

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="lacks configured CLI authority",
    ):
        store.publish_automated_review_stage(execution)


def test_review_publication_rechecks_bound_workspace_settings(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, workspace, _, _, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    execution = coordinator.execute(prepared, workspace=workspace)
    coordinator.settings = replace(
        coordinator.settings,
        agent_artifact_byte_limit=(coordinator.settings.agent_artifact_byte_limit + 1),
    )

    with pytest.raises(
        ExpertValidationStoreError,
        match="authority changed after binding",
    ):
        store.publish_automated_review_stage(execution)


def test_review_rejects_incomplete_or_out_of_order_accepted_evidence(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, _, _, snapshot, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    stored_candidate = store.reducer.candidate_store.read(prepared.packet.candidate_id)

    with pytest.raises(ExpertAutomatedReviewError, match="incomplete"):
        coordinator.prepare(
            stored_candidate=stored_candidate,
            validation_attempt=prepared.validation_attempt,
            authorization_transition_id=snapshot.transition.transition_id,
            authorization_state=snapshot.state,
            accepted_stage_results=prepared.accepted_stage_results[:-1],
        )

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="differs from its exact reference",
    ):
        replace(
            prepared,
            accepted_stage_results=tuple(reversed(prepared.accepted_stage_results)),
        )


def test_stage_result_builder_rejects_forged_pass_adjudication(
    tmp_path,
    monkeypatch,
):
    validation_settings = _validation_settings()
    outputs = {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.TASK_SPECIFIC.value,
            "judgment": validation_settings.policy.promotion.rejection_judgment,
            "rationale": "The evidence is task-specific.",
        }
        for reviewer in validation_settings.policy.reviewers
    }
    coordinator, prepared, workspace, _, _, _ = _review_fixture(
        tmp_path,
        monkeypatch,
        outputs,
    )
    execution = coordinator.execute(prepared, workspace=workspace)
    forged_payload = execution.adjudication.to_dict()
    del forged_payload["adjudication_id"]
    forged_payload["outcome"] = ExpertAutomatedReviewOutcome.PASSED.value
    forged = ExpertAutomatedReviewAdjudication.mint(**forged_payload)

    with pytest.raises(ExpertAutomatedReviewError, match="adjudication differs"):
        build_expert_automated_review_stage_result(
            prepared=prepared,
            assertions=execution.assertions,
            operation_records=execution.operation_records,
            adjudication=forged,
        )


def test_passed_review_publishes_reopens_and_replays_without_agent_work(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, workspace, runner, _, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    assert store.reopen_or_replay_automated_review(prepared.packet) is None
    execution = coordinator.execute(prepared, workspace=workspace)

    committed = store.publish_automated_review_stage(execution)
    reopened = _reopen_validation_store(store)
    snapshot = reopened.snapshot(prepared.packet.candidate_id)
    replayed = reopened.reopen_or_replay_automated_review(prepared.packet)

    assert committed.replayed is False
    assert snapshot == committed.snapshot
    assert snapshot is not None
    assert snapshot.state.promotion_state is ExpertPromotionState.VALIDATING
    assert snapshot.state.next_stage is ExpertValidationStage.RELEASE_MATRIX
    assert snapshot.state.review_assertion_ids == execution.stage_result.assertion_ids
    assert snapshot.accepted_stage_results[-1] == execution.stage_result
    assert replayed is not None
    assert replayed.replayed is True
    assert replayed.stage_result == execution.stage_result
    assert len(runner.calls) == len(store.settings.policy.reviewers)


@pytest.mark.parametrize(
    ("disposition", "expected_outcome", "expected_state"),
    (
        (
            ExpertReviewDisposition.TASK_SPECIFIC,
            ExpertAutomatedReviewOutcome.REJECTED,
            ExpertPromotionState.FAILED,
        ),
        (
            None,
            ExpertAutomatedReviewOutcome.DISPUTED,
            ExpertPromotionState.DISPUTED,
        ),
    ),
)
def test_nonpassing_review_publication_preserves_the_accepted_prefix(
    tmp_path,
    monkeypatch,
    disposition,
    expected_outcome,
    expected_state,
):
    settings = _validation_settings()
    reviewers = settings.policy.reviewers
    if disposition is None:
        outputs = {
            reviewers[0].reviewer_role: {
                "disposition": ExpertReviewDisposition.CORE_ELIGIBLE.value,
                "judgment": settings.policy.promotion.approval_judgment,
                "rationale": "Evidence supports a reusable change.",
            },
            reviewers[1].reviewer_role: {
                "disposition": ExpertReviewDisposition.TASK_SPECIFIC.value,
                "judgment": settings.policy.promotion.rejection_judgment,
                "rationale": "Transfer evidence remains insufficient.",
            },
        }
    else:
        outputs = {
            reviewer.reviewer_role: {
                "disposition": disposition.value,
                "judgment": settings.policy.promotion.rejection_judgment,
                "rationale": "The candidate is not core-eligible.",
            }
            for reviewer in reviewers
        }
    coordinator, prepared, workspace, _, before, store = _review_fixture(
        tmp_path,
        monkeypatch,
        outputs,
    )
    execution = coordinator.execute(prepared, workspace=workspace)

    committed = store.publish_automated_review_stage(execution)

    assert execution.stage_result.outcome is expected_outcome
    assert committed.snapshot.state.promotion_state is expected_state
    assert committed.snapshot.state.next_stage is None
    assert committed.snapshot.accepted_stage_results == before.accepted_stage_results
    assert committed.snapshot.transition.accepted_stage_result_record_ids == (
        before.transition.accepted_stage_result_record_ids
    )
    assert committed.snapshot.state.terminal_evidence_ids == (
        execution.stage_result.stage_result_record_id,
    )


def test_concurrent_identical_review_publications_commit_once(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, workspace, _, _, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    execution = coordinator.execute(prepared, workspace=workspace)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _position: store.publish_automated_review_stage(execution),
                range(2),
            )
        )

    assert sum(not result.replayed for result in results) == 1
    assert len({result.snapshot.transition.transition_id for result in results}) == 1


def test_review_publication_rederives_the_canonical_prompt(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, workspace, _, _, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    execution = coordinator.execute(prepared, workspace=workspace)
    original_template = coordinator.operation_template()
    monkeypatch.setattr(
        ExpertAutomatedReviewCoordinator,
        "operation_template",
        staticmethod(lambda: original_template + "\nUnrelated approval instruction."),
    )

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="canonical packet prompt",
    ):
        store.publish_automated_review_stage(execution)


def test_reopen_rejects_tampered_durable_review_operation(tmp_path, monkeypatch):
    coordinator, prepared, workspace, _, _, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    execution = coordinator.execute(prepared, workspace=workspace)
    store.publish_automated_review_stage(execution)
    operation = execution.operation_records[0]
    operation_path = store._object_path(
        operation.operation_record_id,
        create_namespace=False,
    )
    payload = json.loads(operation_path.read_bytes())
    payload["final_output"] = json.dumps(
        {
            "disposition": ExpertReviewDisposition.TASK_SPECIFIC.value,
            "judgment": store.settings.policy.promotion.rejection_judgment,
            "rationale": "tampered",
        },
        sort_keys=True,
    )
    operation_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ContractValidationError):
        store.snapshot(prepared.packet.candidate_id)


def test_review_stage_orchestrator_runs_end_to_end_and_then_does_no_work(
    tmp_path,
    monkeypatch,
):
    coordinator, prepared, _, runner, before, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    orchestrator = ExpertAutomatedReviewStageOrchestrator(
        coordinator=coordinator,
        candidate_store=store.reducer.candidate_store,
        validation_store=store,
    )
    assert before.latest_attempt is not None

    completed = orchestrator.run(before.latest_attempt)
    calls_after_completion = tuple(runner.calls)
    replayed = orchestrator.run(before.latest_attempt)

    assert completed.state.next_stage is ExpertValidationStage.RELEASE_MATRIX
    assert completed.accepted_stage_results[-1].stage_result_record_id == (
        completed.state.accepted_stage_results[-1].stage_result_record_id
    )
    assert replayed == completed
    assert tuple(runner.calls) == calls_after_completion
    assert prepared.packet.candidate_id == completed.state.candidate_id


def test_concurrent_review_stage_runs_invoke_each_reviewer_once(
    tmp_path,
    monkeypatch,
):
    coordinator, _, _, runner, before, store = _review_fixture(
        tmp_path,
        monkeypatch,
    )
    orchestrator = ExpertAutomatedReviewStageOrchestrator(
        coordinator=coordinator,
        candidate_store=store.reducer.candidate_store,
        validation_store=store,
    )
    assert before.latest_attempt is not None

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshots = tuple(
            executor.map(
                lambda _position: orchestrator.run(before.latest_attempt),
                range(2),
            )
        )

    assert snapshots[0] == snapshots[1]
    assert len(runner.calls) == len(store.settings.policy.reviewers)
