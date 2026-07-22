import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
)
from kapso.cross_run.canonical import canonical_json_bytes
from kapso.cross_run.contracts import (
    ContractValidationError,
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertReviewDisposition,
    ExpertValidationStage,
)
from kapso.cross_run.expert.review import (
    ExpertAutomatedReviewCoordinator,
    ExpertAutomatedReviewError,
    ExpertAutomatedReviewExecution,
    build_expert_automated_review_stage_result,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewAdjudication,
    ExpertAutomatedReviewOperationRecord,
    ExpertAutomatedReviewOutcome,
)
from kapso.cross_run.expert.review_stage import (
    ExpertAutomatedReviewStageOrchestrator,
)
from kapso.cross_run.expert.candidates import ExpertCandidateValidator
from kapso.cross_run.expert.store import ExpertCandidateStore
from kapso.cross_run.expert.validation_store import (
    ExpertValidationStore,
    ExpertValidationStoreError,
)
from test_expert_candidates import (
    bootstrap_candidate_closure,
    expert_settings,
    sanitation_settings,
)
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
            role
            for role in self.outputs
            if f'"reviewer_role":"{role}"' in input
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
        agent_artifact_byte_limit=(
            coordinator.settings.agent_artifact_byte_limit + 1
        ),
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
    stored_candidate = store.reducer.candidate_store.read(
        prepared.packet.candidate_id
    )

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
