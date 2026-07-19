"""Behavioral contracts for evidence-directed ideation records."""

import math

import pytest

from kapso.execution.search_strategies.generic.ideation import (
    BatchStatus,
    CampaignAction,
    CampaignEvidenceSnapshot,
    CandidateAnalysis,
    CandidateDisposition,
    CandidateDispositionKind,
    ClaimKind,
    CodingAgentCallRequest,
    CodingAgentCallResult,
    DiagnosisAudit,
    EmbeddingRecord,
    EmbeddingTelemetry,
    EvaluationGap,
    EvaluationStatus,
    EvidenceClaim,
    EvidenceSignal,
    EvidenceStatus,
    ExperimentEvidence,
    GapState,
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
    ResurfacedIdea,
    SearchDirective,
    SelectionDecision,
    SimilarityMatch,
    content_identifier,
    require_batch_transition,
    require_gap_transition,
    require_idea_transition,
)

NOW = "2026-07-19T00:00:00+00:00"
IDEA_ID = "idea_" + "a" * 32
OTHER_IDEA_ID = "idea_" + "b" * 32
BATCH_ID = "batch_" + "c" * 32
CLAIM_ID = "claim_" + "d" * 32
GAP_ID = "gap_" + "e" * 32
EVIDENCE_DIGEST = "1" * 64
EVIDENCE_ID = content_identifier("evidence_snapshot", EVIDENCE_DIGEST)
CAPACITY_ID = "capacity_snapshot_" + "2" * 32


def descriptor() -> IdeaDescriptor:
    return IdeaDescriptor(
        approach_family="regularization",
        intervention_target="training loss",
        mechanism="stabilize gradients",
        expected_effect="improve validation utility",
    )


def parent_plan() -> ParentPlan:
    return ParentPlan(kind=ParentPlanKind.BASELINE)


def resolved_parent() -> ResolvedParentSnapshot:
    return ResolvedParentSnapshot(
        node_id=None,
        branch_name="baseline",
        git_ref="abc123",
        materialized_ref="abc123",
        diff_base_ref="abc123",
        feedback_base_ref="abc123",
    )


def policy() -> PolicyDecision:
    return PolicyDecision(
        action=CampaignAction.IDEATE,
        mode=IdeationMode.BOOTSTRAP,
        reasons=(
            PolicyReason(
                code="cold_start",
                statement="No completed experiment exists",
                evidence_refs=(EVIDENCE_ID,),
            ),
        ),
    )


def directive() -> SearchDirective:
    brief = OperatorBrief(
        operator=OperatorKind.INDEPENDENT_DRAFT,
        rationale="Establish a first measured hypothesis",
        descriptor_target=descriptor(),
        parent_plan=parent_plan(),
    )
    return SearchDirective(
        decision=policy(),
        evidence_snapshot_id=EVIDENCE_ID,
        capacity_snapshot_id=CAPACITY_ID,
        operator_briefs=(brief,),
        candidate_quota=1,
        repair_quota=1,
        validation_requirements=("full evaluator identity",),
        allowed_parent_plan_kinds=(ParentPlanKind.BASELINE,),
        terminal_constraints=("preserve finalization reserve",),
    )


def generated_idea(idea_id: str = IDEA_ID) -> IdeaRecord:
    return IdeaRecord(
        idea_id=idea_id,
        origin_batch_id=BATCH_ID,
        proposal="Add gradient clipping and measure validation utility.",
        operator=OperatorKind.INDEPENDENT_DRAFT,
        descriptor=descriptor(),
        parent_plan=parent_plan(),
        resolved_parent=resolved_parent(),
        assumptions=("gradient spikes are present",),
        evidence_refs=(EVIDENCE_ID,),
        directive_rationale="Establish a measurable baseline improvement.",
        evaluation_method="Run the canonical evaluator at full fidelity.",
        resource_request="One complete experiment.",
        created_at=NOW,
        expected_observations=("lower gradient variance",),
        predicted_gain=0.05,
        predicted_cost=1.0,
        confidence=0.7,
    )


def planned_batch() -> IdeaBatch:
    return IdeaBatch(
        batch_id=BATCH_ID,
        campaign_id="campaign-alpha",
        iteration_index=0,
        context_hash="3" * 64,
        evidence_snapshot_id=EVIDENCE_ID,
        directive=directive(),
        created_at=NOW,
        updated_at=NOW,
    )


def eligible_analysis() -> CandidateAnalysis:
    return CandidateAnalysis(idea_id=IDEA_ID, eligible=True)


def selection() -> SelectionDecision:
    return SelectionDecision(
        selected_idea_id=IDEA_ID,
        fallback_idea_ids=(),
        dispositions=(
            CandidateDisposition(
                idea_id=IDEA_ID,
                disposition=CandidateDispositionKind.SELECTED,
                reason="Best evidence-adjusted utility.",
            ),
        ),
        diagnosis_audit=(),
        hard_rule_results=("schema valid",),
        gap_decisions=("no actionable gaps",),
        duplicate_overrides=(),
        decision_summary="Select the only eligible bootstrap hypothesis.",
        selection_artifacts=("/tmp/selection.json",),
        expected_benefit=0.05,
        expected_cost=1.0,
    )


def test_every_record_round_trips_through_its_strict_parser():
    claim = EvidenceClaim(
        claim_id=CLAIM_ID,
        statement="No comparable experiment exists.",
        kind=ClaimKind.OBSERVATION,
        status=EvidenceStatus.INSUFFICIENT,
        source_refs=(),
        affected_idea_ids=(),
        affected_experiment_node_ids=(),
        updated_at=NOW,
    )
    gap = EvaluationGap(
        gap_id=GAP_ID,
        axis="validation stability",
        description="Variance across seeds is unknown.",
        state=GapState.OPEN,
        evidence_refs=(EVIDENCE_ID,),
        impact=0.8,
        uncertainty=0.9,
        estimated_cost=1.0,
        deferral_count=0,
        opened_at=NOW,
    )
    experiment = ExperimentEvidence(
        node_id=1,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        parent_node_id=0,
        proposal="Add gradient clipping.",
        raw_score=0.8,
        normalized_utility=0.8,
        evaluation_status=EvaluationStatus.VALID,
        implementation_status=ImplementationStatus.COMPLETED,
        evaluator_id="evaluator-v1",
        build_fidelity="full",
        eval_fidelity="full",
        eval_fraction=1.0,
        seed=7,
        comparable=True,
        feedback="Utility improved.",
        technical_difficulty=None,
        created_at=NOW,
    )
    snapshot = CampaignEvidenceSnapshot(
        snapshot_id=EVIDENCE_ID,
        campaign_id="campaign-alpha",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        generated_at=NOW,
        content_hash=EVIDENCE_DIGEST,
        experiments=(experiment,),
        claims=(claim,),
        gaps=(gap,),
        relevant_idea_ids=(IDEA_ID,),
        incumbent_node_id=1,
        latest_node_id=1,
        noise_floor=0.01,
        signals=(EvidenceSignal.CREDIBLE_IMPROVEMENT,),
    )
    capacity = IdeationCapacityView(
        capacity_snapshot_id=CAPACITY_ID,
        iteration_index=1,
        max_iterations=10,
        remaining_seconds=900.0,
        remaining_after_reserve_seconds=600.0,
        remaining_usd=5.0,
        fidelity_profile="full",
        build_fidelity="full",
        eval_fidelity="full",
        eval_fraction=1.0,
        target_node_id=None,
        reserve_run=False,
        deadline_seconds=300.0,
        can_start_complete_action=True,
        can_run_comparable_evaluation=True,
        preserves_finalization_reserve=True,
        opportunity_probe_required=False,
        opportunity_probe_admissible=False,
    )
    records = (
        CodingAgentCallRequest(
            role="candidate",
            cli="codex",
            model="gpt-5",
            prompt="Full prompt",
            workspace="/workspace",
            timeout_seconds=30.0,
            effort="high",
            allowed_tools=("Read",),
        ),
        CodingAgentCallResult(
            output='{"ideas": []}',
            duration_seconds=2.0,
            cost_usd=None,
            input_tokens=10,
            output_tokens=5,
            artifacts=("result.json",),
        ),
        EmbeddingTelemetry(
            provider="openai",
            model="text-embedding-3-small",
            call_count=1,
            input_tokens=10,
            duration_seconds=0.2,
        ),
        EmbeddingRecord(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=2,
            input_hash="4" * 64,
            vector=(0.1, 0.2),
        ),
        descriptor(),
        parent_plan(),
        resolved_parent(),
        directive().operator_briefs[0],
        policy().reasons[0],
        policy(),
        directive(),
        SimilarityMatch(idea_id=OTHER_IDEA_ID, similarity=0.4),
        eligible_analysis(),
        DiagnosisAudit(
            claim_id=CLAIM_ID,
            status=EvidenceStatus.INSUFFICIENT,
            evidence_refs=(),
        ),
        selection().dispositions[0],
        selection(),
        IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=0.1,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        ),
        generated_idea(),
        ResurfacedIdea(
            idea_id=OTHER_IDEA_ID,
            changed_conditions=("new comparable evidence",),
        ),
        planned_batch(),
        gap,
        claim,
        experiment,
        snapshot,
        capacity,
    )
    for record in records:
        restored = type(record).from_dict(record.to_dict())
        assert restored == record
        assert restored.to_dict() == record.to_dict()


def test_frozen_records_canonicalize_input_lists():
    evidence_refs = [EVIDENCE_ID]
    reason = PolicyReason(
        code="cold_start",
        statement="Start from baseline.",
        evidence_refs=evidence_refs,
    )
    evidence_refs.append("later mutation")
    assert reason.evidence_refs == (EVIDENCE_ID,)


def test_strict_parsers_reject_unknown_missing_and_string_lists():
    payload = policy().to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="fields are invalid"):
        PolicyDecision.from_dict(payload)

    payload = policy().to_dict()
    del payload["mode"]
    with pytest.raises(ValueError, match="missing=mode"):
        PolicyDecision.from_dict(payload)

    payload = policy().to_dict()
    payload["reasons"][0]["evidence_refs"] = "not-a-list"
    with pytest.raises(ValueError, match="list of strings"):
        PolicyDecision.from_dict(payload)


def test_typed_ids_and_finite_numbers_are_enforced():
    payload = generated_idea().to_dict()
    payload["idea_id"] = BATCH_ID
    with pytest.raises(ValueError, match="idea identifier prefix"):
        IdeaRecord.from_dict(payload)

    payload = generated_idea().to_dict()
    payload["predicted_gain"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        IdeaRecord.from_dict(payload)


def test_lifecycle_transition_tables_reject_skips_and_backwards_moves():
    require_batch_transition(BatchStatus.PLANNED, BatchStatus.GENERATED)
    require_idea_transition(IdeaStatus.DEFERRED, IdeaStatus.SELECTED)
    require_gap_transition(GapState.INCONCLUSIVE, GapState.CLOSED)

    with pytest.raises(ValueError, match="illegal batch transition"):
        require_batch_transition(BatchStatus.PLANNED, BatchStatus.SELECTED)
    with pytest.raises(ValueError, match="illegal idea transition"):
        require_idea_transition(IdeaStatus.EVALUATED, IdeaStatus.SELECTED)
    with pytest.raises(ValueError, match="illegal gap transition"):
        require_gap_transition(GapState.CLOSED, GapState.OPEN)


def test_finalize_directive_has_no_generation_work():
    final = SearchDirective(
        decision=PolicyDecision(
            action=CampaignAction.FINALIZE,
            mode=None,
            reasons=(
                PolicyReason(
                    code="terminal_capacity",
                    statement="No complete action fits.",
                    evidence_refs=(CAPACITY_ID,),
                ),
            ),
        ),
        evidence_snapshot_id=EVIDENCE_ID,
        capacity_snapshot_id=CAPACITY_ID,
        operator_briefs=(),
        candidate_quota=0,
        repair_quota=0,
        terminal_constraints=("deliver incumbent",),
    )
    assert final.decision.action == CampaignAction.FINALIZE

    payload = final.to_dict()
    payload["candidate_quota"] = 1
    with pytest.raises(ValueError, match="zero candidate quota"):
        SearchDirective.from_dict(payload)


def test_exact_duplicate_requires_a_material_change_to_remain_eligible():
    with pytest.raises(ValueError, match="materially changed"):
        CandidateAnalysis(
            idea_id=IDEA_ID,
            eligible=True,
            exact_duplicate_of=OTHER_IDEA_ID,
        )
    analysis = CandidateAnalysis(
        idea_id=IDEA_ID,
        eligible=True,
        exact_duplicate_of=OTHER_IDEA_ID,
        exact_duplicate_changed_conditions=("new evaluator version",),
    )
    assert analysis.eligible


def test_non_insufficient_claims_require_sources():
    with pytest.raises(ValueError, match="require sources"):
        EvidenceClaim(
            claim_id=CLAIM_ID,
            statement="The intervention improved utility.",
            kind=ClaimKind.OBSERVATION,
            status=EvidenceStatus.SUPPORTED,
            source_refs=(),
            affected_idea_ids=(),
            affected_experiment_node_ids=(),
            updated_at=NOW,
        )
