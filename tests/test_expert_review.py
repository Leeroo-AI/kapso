import json
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
    coding_agent_returned_artifact_filenames,
)
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    ContractValidationError,
    ExpertEvaluatorOutcome,
    ExpertReviewDisposition,
    ExpertValidationStage,
)
from kapso.cross_run.expert.review import (
    ExpertAutomatedReviewCoordinator,
    ExpertAutomatedReviewError,
    build_expert_automated_review_stage_result,
)
from kapso.cross_run.expert.review_contracts import (
    ExpertAutomatedReviewAdjudication,
    ExpertAutomatedReviewOperationRecord,
    ExpertAutomatedReviewOutcome,
)
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallResult,
    coding_agent_invocation_bytes,
    coding_agent_mcp_configuration_bytes,
    coding_agent_response_schema_bytes,
)
from test_expert_validation import _validation_reducer, _validation_settings
from test_expert_validation_store import (
    _candidate_and_eligibility,
    _result,
    _validation_store,
)


class AutomatedReviewRunner:
    def __init__(
        self,
        artifact_root: Path,
        outputs: dict[str, dict],
        sensitive_file_glob_scan_max_depth: int,
    ):
        self.artifact_root = artifact_root
        self.outputs = outputs
        self.sensitive_file_glob_scan_max_depth = sensitive_file_glob_scan_max_depth
        self.calls = []

    def run(
        self,
        request,
        response_schema,
        *,
        workspace_authority_descriptor=None,
    ):
        assert workspace_authority_descriptor is None
        assert request.workspace_policy.access is CodingAgentWorkspaceAccess.READ_ONLY
        assert request.allowed_tools == ()
        assert request.prior_knowledge is None
        self.calls.append((request, response_schema))
        output = json.dumps(self.outputs[request.role], sort_keys=True)
        if not self.artifact_root.exists():
            self.artifact_root.mkdir(mode=0o700)
        artifact_directory = self.artifact_root / request.operation_id
        artifact_directory.mkdir(mode=0o700)
        artifacts = {
            "final.json": output.encode("utf-8"),
            "invocation.json": coding_agent_invocation_bytes(
                request,
                sensitive_file_glob_scan_max_depth=(
                    self.sensitive_file_glob_scan_max_depth
                ),
            ),
            "mcp_audit.jsonl": b"",
            "mcp_config.json": coding_agent_mcp_configuration_bytes(
                request,
                artifact_directory,
            ),
            "prior_knowledge.json": b"null\n",
            "prompt.txt": request.prompt.encode("utf-8"),
            "response_schema.json": coding_agent_response_schema_bytes(response_schema),
            "stderr.txt": b"",
            "stdout.txt": b"completed\n",
        }
        returned_paths = tuple(
            str(artifact_directory / name)
            for name in coding_agent_returned_artifact_filenames(
                CodingAgentWorkspaceAccess.READ_ONLY
            )
        )
        result = CodingAgentCallResult(
            output=output,
            duration_seconds=1.0,
            cost_usd=None,
            final_output_digest=tree_or_blob_digest(output.encode("utf-8")),
            workspace_delta_digest=None,
            input_tokens=1,
            output_tokens=1,
            artifacts=returned_paths,
        )
        artifacts["result.json"] = result.to_json_bytes()
        for name, payload in artifacts.items():
            path = artifact_directory / name
            path.write_bytes(payload)
            path.chmod(0o600)
        return result


def _review_fixture(tmp_path, outputs=None):
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    candidates, stored, adapter, validation_settings, eligibility = (
        _candidate_and_eligibility(tmp_path)
    )
    reducer = _validation_reducer(
        validation_settings,
        adapter,
        candidate_store=candidates,
    )
    store = _validation_store(tmp_path, validation_settings, reducer)
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
    expert_settings = candidates.validator.settings
    configured_outputs = outputs or {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.CORE_ELIGIBLE.value,
            "judgment": validation_settings.policy.promotion.approval_judgment,
            "rationale": f"{reviewer.reviewer_id} found complete reusable evidence.",
        }
        for reviewer in validation_settings.policy.reviewers
    }
    runner = AutomatedReviewRunner(
        tmp_path / "review-artifacts",
        configured_outputs,
        expert_settings.sensitive_file_glob_scan_max_depth,
    )
    coordinator = ExpertAutomatedReviewCoordinator(expert_settings, runner)
    prepared = coordinator.prepare(
        stored_candidate=stored,
        validation_attempt=snapshot.latest_attempt,
        authorization_transition_id=snapshot.transition.transition_id,
        authorization_state=snapshot.state,
        accepted_stage_results=snapshot.accepted_stage_results,
    )
    workspace = (tmp_path / "review-workspace").resolve()
    workspace.mkdir(mode=0o700)
    return coordinator, prepared, workspace, runner, snapshot


def test_review_packet_is_complete_and_passes_only_unanimous_clean_review(tmp_path):
    coordinator, prepared, workspace, runner, snapshot = _review_fixture(tmp_path)

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
    assert all(call[0].workspace == str(workspace) for call in runner.calls)
    assert all(call[0].allowed_tools == () for call in runner.calls)
    source = prepared.candidate_input.candidate_contents_text["src/execution.py"]
    assert json.dumps(source)[1:-1] in runner.calls[0][0].prompt
    assert (
        ExpertAutomatedReviewOperationRecord.from_json_bytes(
            execution.operation_records[0].to_json_bytes()
        )
        == execution.operation_records[0]
    )
    assert len(snapshot.accepted_stage_results) == len(
        prepared.packet.accepted_stage_results
    )


def test_mixed_review_is_disputed_even_when_rejection_quorum_is_met(tmp_path):
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
    coordinator, prepared, workspace, _, _ = _review_fixture(
        tmp_path / "mixed",
        outputs,
    )

    execution = coordinator.execute(prepared, workspace=workspace)

    assert execution.adjudication.outcome is ExpertAutomatedReviewOutcome.DISPUTED
    assert execution.stage_result.outcome is ExpertAutomatedReviewOutcome.DISPUTED


def test_clean_rejection_is_rejected(tmp_path):
    validation_settings = _validation_settings()
    outputs = {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.UNSAFE_OR_SPECIALIZED.value,
            "judgment": validation_settings.policy.promotion.rejection_judgment,
            "rationale": "The candidate encodes a specialized unsafe shortcut.",
        }
        for reviewer in validation_settings.policy.reviewers
    }
    coordinator, prepared, workspace, _, _ = _review_fixture(
        tmp_path / "rejected",
        outputs,
    )

    execution = coordinator.execute(prepared, workspace=workspace)

    assert execution.adjudication.outcome is ExpertAutomatedReviewOutcome.REJECTED
    assert execution.stage_result.outcome is ExpertAutomatedReviewOutcome.REJECTED


def test_review_rejects_judgment_disposition_conflict(tmp_path):
    validation_settings = _validation_settings()
    outputs = {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.TASK_SPECIFIC.value,
            "judgment": validation_settings.policy.promotion.approval_judgment,
            "rationale": "Contradictory output.",
        }
        for reviewer in validation_settings.policy.reviewers
    }
    coordinator, prepared, workspace, _, _ = _review_fixture(
        tmp_path / "conflict",
        outputs,
    )

    with pytest.raises(
        ExpertAutomatedReviewError,
        match="judgment and disposition conflict",
    ):
        coordinator.execute(prepared, workspace=workspace)


def test_review_rejects_nonempty_or_shared_workspace(tmp_path):
    coordinator, prepared, workspace, _, _ = _review_fixture(tmp_path)
    (workspace / "unexpected.txt").write_text("not packet-only", encoding="utf-8")

    with pytest.raises(ExpertAutomatedReviewError, match="empty normalized"):
        coordinator.execute(prepared, workspace=workspace)

    (workspace / "unexpected.txt").unlink()
    workspace.chmod(0o755)
    with pytest.raises(ExpertAutomatedReviewError, match="must be private"):
        coordinator.execute(prepared, workspace=workspace)


def test_review_operation_artifact_tampering_fails_loud(tmp_path):
    coordinator, prepared, workspace, _, _ = _review_fixture(tmp_path)
    execution = coordinator.execute(prepared, workspace=workspace)
    operation = execution.operation_records[0]
    payloads = dict(operation.artifact_payloads_base64)
    payloads["prompt.txt"] = payloads["final.json"]

    with pytest.raises(ContractValidationError, match="artifacts differ"):
        replace(operation, artifact_payloads_base64=payloads)


def test_review_rejects_incomplete_or_out_of_order_accepted_evidence(tmp_path):
    coordinator, prepared, _, _, snapshot = _review_fixture(tmp_path)
    stored = coordinator.settings
    assert stored is not None
    other_root = tmp_path / "other"
    other_root.mkdir(mode=0o700)

    with pytest.raises(ExpertAutomatedReviewError, match="incomplete"):
        coordinator.prepare(
            stored_candidate=_candidate_and_eligibility(other_root)[1],
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


def test_stage_result_builder_rejects_forged_pass_adjudication(tmp_path):
    validation_settings = _validation_settings()
    outputs = {
        reviewer.reviewer_role: {
            "disposition": ExpertReviewDisposition.TASK_SPECIFIC.value,
            "judgment": validation_settings.policy.promotion.rejection_judgment,
            "rationale": "The evidence is task-specific.",
        }
        for reviewer in validation_settings.policy.reviewers
    }
    coordinator, prepared, workspace, _, _ = _review_fixture(
        tmp_path,
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
