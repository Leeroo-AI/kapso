"""Replayable domain-neutral capture used by the production trust-path smoke."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.bundle import StoredRunBundle
from kapso.cross_run.capture.exporter import (
    BRANCH_SNAPSHOT_SCHEMA,
    CAPTURE_DESCRIPTOR_REF,
    CAPTURE_DESCRIPTOR_SCHEMA,
    BranchSnapshot,
    CaptureDescriptor,
)
from kapso.cross_run.capture.git_evidence import reconstruct_root_tree_sha
from kapso.cross_run.capture.sanitation import SANITATION_REPORT_REF
from kapso.cross_run.catalog.projector import ProjectionResult, RunBundleProjector
from kapso.cross_run.contracts import (
    ArtifactCompleteness,
    ArtifactEnvironment,
    CompletionState,
    EpisodeEvaluationStatus,
    EvaluationFingerprint,
    ExecutionStatus,
    ExpertScopeContract,
    RunBundle,
    TaskContextBinding,
)
from kapso.cross_run.git_refs import git_object_sha
from kapso.cross_run.record_contracts import (
    EXECUTION_REVISION_EVENT_SCHEMA,
    SANITATION_REPORT_SCHEMA,
    SANITATION_SCANNER_VERSION,
    ExecutionRevisionEvent,
    SanitationReport,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.execution.coding_agents.structured_call import CodingAgentCallResult
from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.memories.experiment_memory.record import (
    EXPERIMENT_HISTORY_SCHEMA,
    ExperimentRecord,
)
from kapso.execution.run_checkpoint import RunCheckpoint
from kapso.execution.search_strategies.generic.ideation.archive import (
    IDEA_ARCHIVE_SCHEMA,
    IdeaArchiveState,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    BatchStatus,
    CampaignAction,
    CampaignEvidenceSnapshot,
    CandidateAnalysis,
    CandidateDisposition,
    CandidateDispositionKind,
    EvaluationStatus,
    EvidenceSignal,
    IdeaBatch,
    IdeaDescriptor,
    IdeaOutcome,
    IdeaRecord,
    IdeaStatus,
    IdeationCapacityView,
    IdeationMode,
    ImplementationStatus,
    ObjectiveDirection,
    OperatorBrief,
    OperatorKind,
    ParentPlan,
    ParentPlanKind,
    PolicyDecision,
    PolicyReason,
    ResolvedParentSnapshot,
    SearchDirective,
    SelectionDecision,
    content_identifier,
)
from kapso.execution.search_strategies.generic.strategy import (
    GENERIC_SEARCH_STATE_SCHEMA,
)
from kapso.execution.search_strategies.node import SearchNode


_CAMPAIGN_ID = "production_smoke_campaign"
_CHECKPOINT_REF = "payload/checkpoint.json"
_EVENT_JOURNAL_REF = "payload/execution_events.jsonl"
_EXPERIMENT_HISTORY_REF = "payload/experiment_history.json"
_IDEA_ARCHIVE_REF = "payload/idea_archive.json"
_BRANCH_SNAPSHOT_REF = "payload/branches/00000000/00000000/manifest.json"
_BRANCH_SOURCE_REF = "payload/branches/00000000/00000000/files/bounded_contract.py"
_TECHNICAL_DIFFICULTY = (
    "The reusable semantic-parity boundary lacks a common preflight diagnostic "
    "for representation mismatches."
)


@dataclass(frozen=True)
class ProductionCapture:
    """One immutable raw bundle and its deterministic public projection."""

    stored_bundle: StoredRunBundle
    projection: ProjectionResult


@dataclass(frozen=True)
class _ProductionBranchEvidence:
    payloads: Mapping[str, bytes]
    artifact_refs: Mapping[str, str]
    event_artifact_refs: Mapping[str, str]


def build_production_capture(
    *,
    settings: CrossRunSettings,
    scope_contract: ExpertScopeContract,
    expert_base_release_id: str,
    task_adapter_manifest_id: str,
    task_adapter_verification_receipt_id: str,
    embedding_inputs: Sequence[str],
    committed_at: str,
    run_id: str,
    evaluation_fingerprint: EvaluationFingerprint,
    previous: ProjectionResult | None = None,
) -> ProductionCapture:
    """Build and mechanically project the production smoke's raw evidence."""

    if len(embedding_inputs) < 2 or any(
        not isinstance(value, str) or not value.strip()
        for value in embedding_inputs[:2]
    ):
        raise ValueError("production capture embedding inputs are incomplete")
    task_context = _task_context(scope_contract, embedding_inputs)
    environment = ArtifactEnvironment.mint(
        kapso_commit="0" * 40,
        expert_base_release_id=expert_base_release_id,
        task_adapter_manifest_id=task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(task_adapter_verification_receipt_id),
        starting_artifact_content_ids={},
        dependency_lock_hash=tree_or_blob_digest(b"transport-smoke-lock"),
    )
    if previous is not None and (
        previous.source_bundle.run_id != run_id
        or previous.source_bundle.campaign_id != _CAMPAIGN_ID
        or previous.source_bundle.scope_contract_id != scope_contract.scope_contract_id
    ):
        raise ValueError("production capture predecessor belongs to another run")

    archive, node, branch = _capture_frontier(
        tuple(embedding_inputs),
        committed_at,
        run_id,
        evaluation_fingerprint,
    )
    objective_direction = evaluation_fingerprint.objective_direction.value
    record = ExperimentRecord.from_node(node, objective_direction, True)
    event = ExecutionRevisionEvent.mint(
        schema=EXECUTION_REVISION_EVENT_SCHEMA,
        run_id=run_id,
        campaign_id=_CAMPAIGN_ID,
        node_id=node.node_id,
        execution_revision=node.execution_revision,
        idea_id=node.idea_id,
        selection_batch_id=node.selection_batch_id,
        parent_node_id=node.parent_node_id,
        started_at=committed_at,
        recorded_at=committed_at,
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=(
            evaluation_fingerprint.evaluator_fingerprint.removeprefix("sha256:"),
        ),
        measurements={**record.metrics, "raw_score": record.raw_score},
        feedback=node.feedback,
        technical_difficulties=node.technical_difficulties,
        artifact_refs=branch.event_artifact_refs,
        projection=record.to_dict(),
    )
    checkpoint = _checkpoint(
        archive=archive,
        node=node,
        configuration_fingerprint=settings.configuration_fingerprint,
    )
    core_payloads = {
        **branch.payloads,
        _CHECKPOINT_REF: canonical_json_bytes(checkpoint.__dict__),
        _EVENT_JOURNAL_REF: event.to_json_bytes() + b"\n",
        _EXPERIMENT_HISTORY_REF: canonical_json_bytes(
            {
                "schema": EXPERIMENT_HISTORY_SCHEMA,
                "run_id": run_id,
                "campaign_id": _CAMPAIGN_ID,
                "revision": 1,
                "objective_direction": objective_direction,
                "require_idea_links": True,
                "records": [record.to_dict()],
            }
        ),
        _IDEA_ARCHIVE_REF: canonical_json_bytes(archive.to_dict()),
    }
    capture_generation = (
        0 if previous is None else previous.source_bundle.capture_generation + 1
    )
    completeness = {
        "checkpoint": ArtifactCompleteness.PRESENT,
        "execution_event_journal": ArtifactCompleteness.PRESENT,
        "idea_archive": ArtifactCompleteness.PRESENT,
        "experiment_history": ArtifactCompleteness.PRESENT,
        "branch:0:0": ArtifactCompleteness.PRESENT,
    }
    artifact_refs = {
        "capture_descriptor": CAPTURE_DESCRIPTOR_REF,
        "checkpoint": _CHECKPOINT_REF,
        "execution_event_journal": _EVENT_JOURNAL_REF,
        "idea_archive": _IDEA_ARCHIVE_REF,
        "experiment_history": _EXPERIMENT_HISTORY_REF,
        **branch.artifact_refs,
    }
    descriptor = CaptureDescriptor(
        schema=CAPTURE_DESCRIPTOR_SCHEMA,
        scope_contract_id=scope_contract.scope_contract_id,
        scope_id=scope_contract.scope_id,
        run_id=run_id,
        campaign_id=_CAMPAIGN_ID,
        completion_state=CompletionState.STOPPED,
        capture_generation=capture_generation,
        started_at=committed_at,
        captured_at=committed_at,
        kapso_commit=environment.kapso_commit,
        launch_manifest_id=content_id(
            "launch-manifest",
            {"transport_smoke": "first-launch"},
        ),
        knowledge_snapshot_id=(
            content_id(
                "knowledge-snapshot",
                {"transport_smoke": "empty-snapshot"},
            )
            if previous is None
            else previous.source_bundle.knowledge_snapshot_id
        ),
        expert_base_release_id=expert_base_release_id,
        task_context_binding=task_context,
        artifact_environment=environment,
        evaluation_fingerprints=(evaluation_fingerprint,),
        artifact_completeness=completeness,
        artifact_refs=artifact_refs,
        branch_snapshot_refs=(_BRANCH_SNAPSHOT_REF,),
        run_log_refs=(),
    )
    admitted_payloads = {
        **core_payloads,
        CAPTURE_DESCRIPTOR_REF: descriptor.to_json_bytes(),
    }
    admitted_checksums = {
        path: tree_or_blob_digest(payload)
        for path, payload in admitted_payloads.items()
    }
    report = SanitationReport.mint(
        schema=SANITATION_REPORT_SCHEMA,
        capture_manifest_id=content_id(
            "capture-manifest",
            {
                "run_id": run_id,
                "capture_generation": capture_generation,
                "admitted_refs": admitted_checksums,
            },
        ),
        scope_id=scope_contract.scope_id,
        task_family_id=task_context.task_family_id,
        policy_version=settings.sanitation.policy_version,
        policy_fingerprint=tree_or_blob_digest(settings.sanitation.to_json_bytes()),
        scanner_version=SANITATION_SCANNER_VERSION,
        status="admitted",
        findings=(),
        excluded_paths=(),
        taint_sources=(),
        admitted_refs=admitted_checksums,
    )
    artifacts = {
        **admitted_payloads,
        SANITATION_REPORT_REF: report.to_json_bytes(),
    }
    checksums = {
        path: tree_or_blob_digest(payload) for path, payload in artifacts.items()
    }
    bundle = RunBundle.mint(
        scope_contract_id=scope_contract.scope_contract_id,
        scope_id=scope_contract.scope_id,
        run_id=run_id,
        campaign_id=_CAMPAIGN_ID,
        completion_state=CompletionState.STOPPED,
        capture_generation=capture_generation,
        supersedes_bundle_id=(
            None if previous is None else previous.source_bundle.bundle_id
        ),
        checkpoint_frontier=1,
        capture_watermarks={
            "branch_snapshot_count": 1,
            "checkpoint_completed_iterations": 1,
            "checkpoint_node_count": 1,
            "execution_journal_event_count": 1,
            "experiment_history_count": 1,
            "experiment_history_revision": 1,
            "idea_archive_revision": archive.revision,
            "strategy_iteration_count": 1,
        },
        configuration_fingerprint=settings.configuration_fingerprint,
        artifact_completeness=completeness,
        started_at=committed_at,
        captured_at=committed_at,
        kapso_commit=environment.kapso_commit,
        launch_manifest_id=descriptor.launch_manifest_id,
        knowledge_snapshot_id=descriptor.knowledge_snapshot_id,
        expert_base_release_id=expert_base_release_id,
        task_context_binding=task_context,
        artifact_environment=environment,
        capture_descriptor_ref=CAPTURE_DESCRIPTOR_REF,
        checkpoint_ref=_CHECKPOINT_REF,
        execution_event_journal_ref=_EVENT_JOURNAL_REF,
        idea_archive_ref=_IDEA_ARCHIVE_REF,
        experiment_history_ref=_EXPERIMENT_HISTORY_REF,
        sanitation_report_ref=SANITATION_REPORT_REF,
        branch_snapshot_refs=(_BRANCH_SNAPSHOT_REF,),
        run_log_refs=(),
        checksums=checksums,
    )
    stored = StoredRunBundle(manifest=bundle, artifacts=artifacts)
    projection = RunBundleProjector(
        settings.capture.score_comparison_tolerance
    ).project(stored, previous)
    return ProductionCapture(stored_bundle=stored, projection=projection)


def _task_context(
    scope_contract: ExpertScopeContract,
    embedding_inputs: Sequence[str],
) -> TaskContextBinding:
    context = TaskContextBinding.mint(
        scope_contract_id=scope_contract.scope_contract_id,
        scope_id=scope_contract.scope_id,
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
        capability_tags=("language.training",),
        input_contract_fingerprint=tree_or_blob_digest(
            embedding_inputs[0].encode("utf-8")
        ),
        target_contract_fingerprint=tree_or_blob_digest(
            embedding_inputs[1].encode("utf-8")
        ),
        starting_artifact_refs=(),
        method_fingerprint=tree_or_blob_digest(b"transport-smoke-method"),
        toolchain_fingerprint=tree_or_blob_digest(b"transport-smoke-toolchain"),
        dependency_runtime_fingerprint=tree_or_blob_digest(b"transport-smoke-runtime"),
        budget_hardware_envelope={"accelerator": "none", "hours": 1},
        transfer_dimensions={
            "dataset_family": "synthetic_public",
            "runtime_family": "python",
        },
    )
    context.validate_against(scope_contract)
    return context


def _capture_frontier(
    embedding_inputs: tuple[str, ...],
    committed_at: str,
    run_id: str,
    evaluation_fingerprint: EvaluationFingerprint,
) -> tuple[IdeaArchiveState, SearchNode, _ProductionBranchEvidence]:
    executed_idea_id = _typed_identifier("idea", "executed")
    deferred_idea_id = _typed_identifier("idea", "deferred")
    batch_id = _typed_identifier("batch", "batch")
    evidence_digest = _raw_digest("evidence")
    evidence_id = content_identifier("evidence_snapshot", evidence_digest)
    capacity_id = _typed_identifier("capacity_snapshot", "capacity")
    descriptor = IdeaDescriptor(
        approach_family="representation_validation",
        intervention_target="input_projection",
        mechanism="validate_semantic_parity_before_training",
        expected_effect="reduce_interface_regressions",
    )
    parent_plan = ParentPlan(kind=ParentPlanKind.BASELINE)
    resolved_parent = ResolvedParentSnapshot(
        node_id=None,
        branch_name="baseline",
        git_ref="a" * 40,
        materialized_ref="a" * 40,
        diff_base_ref="a" * 40,
        feedback_base_ref="a" * 40,
    )
    evidence = CampaignEvidenceSnapshot(
        snapshot_id=evidence_id,
        campaign_id=_CAMPAIGN_ID,
        objective_direction=ObjectiveDirection(
            evaluation_fingerprint.objective_direction.value
        ),
        generated_at=committed_at,
        content_hash=evidence_digest,
        experiments=(),
        claims=(),
        gaps=(),
        relevant_idea_ids=(),
        incumbent_node_id=None,
        latest_node_id=None,
        noise_floor=None,
        signals=(EvidenceSignal.NO_COMPARABLE_EXPERIMENT,),
    )
    capacity = IdeationCapacityView(
        capacity_snapshot_id=capacity_id,
        iteration_index=0,
        max_iterations=1,
        remaining_seconds=1.0,
        remaining_after_reserve_seconds=0.0,
        remaining_usd=1.0,
        fidelity_profile="full",
        build_fidelity="full",
        eval_fidelity="full",
        eval_fraction=1.0,
        target_node_id=None,
        reserve_run=False,
        deadline_seconds=1.0,
        can_start_complete_action=True,
        can_run_granted_evaluation=True,
        can_run_comparable_evaluation=True,
        preserves_finalization_reserve=True,
    )
    parent_brief = OperatorBrief(
        operator=OperatorKind.INDEPENDENT_DRAFT,
        rationale="Produce a transport hypothesis.",
        descriptor_target=descriptor,
        parent_plan=parent_plan,
    )
    directive = SearchDirective(
        decision=PolicyDecision(
            action=CampaignAction.IDEATE,
            mode=IdeationMode.BOOTSTRAP,
            reasons=(
                PolicyReason(
                    code="cold_start",
                    statement="No comparable experiment exists.",
                    evidence_refs=(evidence_id,),
                ),
            ),
        ),
        evidence_snapshot_id=evidence_id,
        capacity_snapshot_id=capacity_id,
        operator_briefs=(parent_brief, parent_brief),
        candidate_quota=2,
        repair_quota=0,
        validation_requirements=("full evaluator identity",),
        allowed_parent_plan_kinds=(ParentPlanKind.BASELINE,),
        terminal_constraints=("preserve finalization reserve",),
    )
    selection = SelectionDecision(
        selected_idea_id=executed_idea_id,
        fallback_idea_ids=(deferred_idea_id,),
        dispositions=(
            CandidateDisposition(
                idea_id=executed_idea_id,
                disposition=CandidateDispositionKind.SELECTED,
                reason="Highest expected diagnostic value.",
            ),
            CandidateDisposition(
                idea_id=deferred_idea_id,
                disposition=CandidateDispositionKind.DEFERRED,
                reason="Preserve an unexecuted follow-up.",
            ),
        ),
        diagnosis_audit=(),
        hard_rule_results=("schema valid",),
        gap_decisions=("no actionable gaps",),
        duplicate_overrides=(),
        decision_summary="Execute diagnostic while retaining one follow-up.",
        selection_artifacts=("selection.json",),
        expected_benefit=0.0,
        expected_cost=1.0,
    )
    generated_ids = (executed_idea_id, deferred_idea_id)
    batch = IdeaBatch(
        batch_id=batch_id,
        campaign_id=_CAMPAIGN_ID,
        iteration_index=0,
        context_hash=_raw_digest("context"),
        planning_archive_revision=0,
        problem_statement="Improve reusable semantic parity validation.",
        evidence_snapshot=evidence,
        capacity=capacity,
        directive=directive,
        resolved_parents=(resolved_parent, resolved_parent),
        created_at=committed_at,
        updated_at=committed_at,
        status=BatchStatus.COMPLETED,
        generated_idea_ids=generated_ids,
        generation_calls=tuple(
            _coding_agent_call(f'{{"idea_id":"{idea_id}"}}')
            for idea_id in generated_ids
        ),
        considered_idea_ids=generated_ids,
        analyses=tuple(
            CandidateAnalysis(idea_id=idea_id, eligible=True)
            for idea_id in generated_ids
        ),
        selection=selection,
        selection_call=_coding_agent_call(
            f'{{"selected_idea_id":"{executed_idea_id}"}}'
        ),
    )
    common_idea_fields = {
        "origin_batch_id": batch_id,
        "operator": OperatorKind.INDEPENDENT_DRAFT,
        "descriptor": descriptor,
        "parent_plan": parent_plan,
        "resolved_parent": resolved_parent,
        "evidence_refs": (evidence_id,),
        "directive_rationale": "Establish a reusable transport boundary.",
        "evaluation_method": "Run the canonical evaluator.",
        "resource_request": "One bounded diagnostic.",
        "created_at": committed_at,
        "expected_observations": ("Expose representation mismatch before training.",),
        "predicted_gain": 0.0,
        "predicted_cost": 1.0,
        "confidence": 0.8,
    }
    normalized_score = (
        1.0 if evaluation_fingerprint.objective_direction.value == "maximize" else -1.0
    )
    evaluated_idea = IdeaRecord(
        idea_id=executed_idea_id,
        proposal=(
            "Validate semantic parity before training through one reusable "
            "representation boundary."
        ),
        assumptions=("Representation mismatches are observable before training.",),
        status=IdeaStatus.EVALUATED,
        selected_in_batch_id=batch_id,
        selection_reason="Highest expected diagnostic value.",
        experiment_node_id=0,
        outcome=IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=normalized_score,
            validation_tier="full",
            actual_cost=0.0,
            actual_duration=1.0,
        ),
        **common_idea_fields,
    )
    deferred_idea = IdeaRecord(
        idea_id=deferred_idea_id,
        proposal=embedding_inputs[0],
        assumptions=(embedding_inputs[1],),
        status=IdeaStatus.DEFERRED,
        deferral_reason="The synthetic run stopped before executing this idea.",
        **common_idea_fields,
    )
    archive = IdeaArchiveState(
        schema=IDEA_ARCHIVE_SCHEMA,
        campaign_id=_CAMPAIGN_ID,
        revision=1,
        created_at=committed_at,
        updated_at=committed_at,
        batches=(batch,),
        ideas=(evaluated_idea, deferred_idea),
        claims=(),
        gaps=(),
    )
    source_payload = (
        b'"""Bounded semantic parity preflight."""\n\n'
        b"def validate_representation(value):\n"
        b"    return value is not None\n"
    )
    source_path = "bounded_contract.py"
    source_mode = "100644"
    source_blob_sha = git_object_sha("blob", source_payload)
    root_tree_sha = reconstruct_root_tree_sha(
        {source_path: (source_mode, source_blob_sha)}
    )
    commit_payload = (
        f"tree {root_tree_sha}\n"
        "author Kapso Production Smoke <kapso@example.invalid> 0 +0000\n"
        "committer Kapso Production Smoke <kapso@example.invalid> 0 +0000\n"
        "\n"
        "Validate representation parity\n"
    ).encode("utf-8")
    commit_sha = git_object_sha("commit", commit_payload)
    evaluator_id = evaluation_fingerprint.evaluator_fingerprint.removeprefix("sha256:")
    metric_name = evaluation_fingerprint.metric_name
    node = SearchNode(
        node_id=0,
        idea_id=executed_idea_id,
        selection_batch_id=batch_id,
        solution=evaluated_idea.proposal,
        branch_name="production_smoke_branch",
        feedback="",
        score=1.0,
        evaluation_valid=True,
        metrics={metric_name: 1.0},
        primary_metric=metric_name,
        duration_seconds=1.0,
        cost_usd=0.0,
        started_at=committed_at,
        evaluation_attempts=[
            EvaluationAttempt(
                commit_sha=commit_sha,
                evaluator_id=evaluator_id,
                fidelity=evaluation_fingerprint.fidelity,
                fraction=evaluation_fingerprint.fraction,
                seed=int(replicate_id.removeprefix("seed-")),
                score=1.0,
                duration_seconds=1.0,
                metrics={metric_name: 1.0},
            )
            for replicate_id in evaluation_fingerprint.seed_or_replicate_ids
        ],
        technical_difficulties=_TECHNICAL_DIFFICULTY,
    )
    revision_ref = f"refs/kapso/execution-revisions/{run_id}/node-0/revision-0"
    commit_ref = "payload/branches/00000000/00000000/commits/" f"{commit_sha}.txt"
    source_digest = tree_or_blob_digest(source_payload)
    branch = BranchSnapshot(
        schema=BRANCH_SNAPSHOT_SCHEMA,
        node_id=0,
        execution_revision=0,
        branch_name=node.branch_name,
        parent_branch_name="",
        revision_ref=revision_ref,
        commit_sha=commit_sha,
        implementation_base_ref="",
        diff_base_ref="",
        feedback_base_ref="",
        base_commit_shas={},
        evaluated_commit_shas=(commit_sha,),
        root_tree_sha=root_tree_sha,
        commit_objects=({"commit_sha": commit_sha, "payload_ref": commit_ref},),
        source_tree_digest=source_tree_digest(
            {source_path: (source_digest, source_mode, len(source_payload))}
        ),
        source_files=(
            {
                "git_blob_sha": source_blob_sha,
                "mode": source_mode,
                "payload_ref": _BRANCH_SOURCE_REF,
                "sha256": source_digest,
                "size": len(source_payload),
                "source_path": source_path,
            },
        ),
        excluded_files=(),
    )
    return (
        archive,
        node,
        _ProductionBranchEvidence(
            payloads={
                _BRANCH_SNAPSHOT_REF: branch.to_json_bytes(),
                _BRANCH_SOURCE_REF: source_payload,
                commit_ref: commit_payload,
            },
            artifact_refs={
                "branch:0:0": _BRANCH_SNAPSHOT_REF,
                "git_commit:00000000": commit_ref,
                "source:00000000": _BRANCH_SOURCE_REF,
            },
            event_artifact_refs={
                "branch": node.branch_name,
                "candidate_commit": commit_sha,
                "candidate_ref": revision_ref,
                **{
                    f"evaluation_commit_{position}": attempt.commit_sha
                    for position, attempt in enumerate(node.evaluation_attempts)
                },
            },
        ),
    )


def _checkpoint(
    *,
    archive: IdeaArchiveState,
    node: SearchNode,
    configuration_fingerprint: str,
) -> RunCheckpoint:
    return RunCheckpoint.create(
        strategy_type="generic",
        goal="Improve the complete task solution.",
        config_fingerprint=configuration_fingerprint,
        status="running",
        completed_iterations=1,
        cumulative_cost=0.0,
        current_feedback=node.feedback,
        strategy_state={
            "schema": GENERIC_SEARCH_STATE_SCHEMA,
            "campaign_id": _CAMPAIGN_ID,
            "idea_archive_schema": IDEA_ARCHIVE_SCHEMA,
            "idea_archive_snapshot": archive.to_dict(),
            "active_batch_id": None,
            "node_history": [node.to_dict()],
            "iteration_count": 1,
            "previous_errors": [node.error_message],
            "evaluation_integrity": {},
            "scores_evaluator_id": None,
            "evaluator_transition": None,
            "cross_run_identity": None,
        },
        elapsed_seconds=1.0,
        cost_by_component={},
        last_stop=None,
    )


def _coding_agent_call(output: str) -> CodingAgentCallResult:
    return CodingAgentCallResult(
        output=output,
        duration_seconds=1.0,
        cost_usd=0.0,
        final_output_digest=tree_or_blob_digest(output.encode("utf-8")),
        workspace_delta_digest=None,
        input_tokens=1,
        output_tokens=1,
    )


def _typed_identifier(prefix: str, seed: str) -> str:
    return content_identifier(prefix, _raw_digest(seed))


def _raw_digest(seed: str) -> str:
    return tree_or_blob_digest(seed.encode("utf-8")).removeprefix("sha256:")
