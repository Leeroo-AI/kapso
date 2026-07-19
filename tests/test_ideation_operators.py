"""Operator diversity, parent intent, gap reservation, and resurfacing tests."""

from dataclasses import replace

from kapso.execution.search_strategies.generic.ideation import (
    CandidateAnalysis,
    CandidateDisposition,
    CandidateDispositionKind,
    ClaimKind,
    EvaluationGap,
    EvidenceClaim,
    EvidenceStatus,
    GapState,
    IdeaArchive,
    IdeaDescriptor,
    IdeaStatus,
    IdeationMode,
    ObjectiveDirection,
    OperatorKind,
    ParentPlanKind,
    SelectionDecision,
    new_identifier,
)
from kapso.execution.search_strategies.generic.ideation.evidence import (
    CampaignEvidenceBuilder,
    ExperimentInput,
    rank_evaluation_gaps,
)
from kapso.execution.search_strategies.generic.ideation.operators import (
    OperatorSettings,
    find_resurfaceable_ideas,
    plan_search_directive,
)
from kapso.execution.search_strategies.generic.ideation.policy import choose_policy
from test_ideation_domain import (
    EVIDENCE_ID,
    NOW,
    generated_idea,
    planned_batch,
    selection,
)
from test_ideation_evidence import (
    CAMPAIGN_ID,
    add_completed_idea,
    capacity,
    evidence_settings,
    experiment,
    gap_settings,
)


def operator_settings(candidate_quota: int = 2) -> OperatorSettings:
    return OperatorSettings(
        candidate_quota=candidate_quota,
        repair_quota=1,
        reserve_gap_slot=True,
    )


def build_snapshot(archive, experiments, available_capacity=None):
    return CampaignEvidenceBuilder(evidence_settings()).build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=experiments,
        archive_state=archive.state,
        capacity=available_capacity or capacity(),
        generated_at=NOW,
    )


def test_actionable_gap_reserves_a_real_operator_slot(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    gap = EvaluationGap(
        gap_id=new_identifier("gap"),
        axis="seed stability",
        description="Seed stability is not measured.",
        state=GapState.OPEN,
        evidence_refs=(EVIDENCE_ID,),
        impact=0.9,
        uncertainty=0.9,
        estimated_cost=1.0,
        deferral_count=0,
        opened_at=NOW,
    )
    archive.add_gaps((gap,), expected_revision=0)
    snapshot = build_snapshot(archive, ())
    decision = choose_policy(snapshot, capacity())
    priorities = rank_evaluation_gaps(
        snapshot.gaps,
        evidence_confidence_by_gap={gap.gap_id: 0.8},
        uncertainty_reduction_by_gap={gap.gap_id: 0.8},
        as_of=NOW,
        settings=gap_settings(),
    )
    directive = plan_search_directive(
        decision,
        snapshot=snapshot,
        capacity=capacity(),
        archive_state=archive.state,
        gap_priorities=priorities,
        settings=operator_settings(),
    )

    assert decision.mode == IdeationMode.BOOTSTRAP
    assert directive.reserved_gap_id == gap.gap_id
    assert directive.operator_briefs[0].operator == OperatorKind.TARGET_GAP
    assert directive.operator_briefs[0].target_gap_id == gap.gap_id
    assert len({brief.descriptor_target for brief in directive.operator_briefs}) == 2


def test_same_family_plateau_allocates_a_mechanism_shift(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    experiments = []
    shared = IdeaDescriptor(
        approach_family="tree_ensemble",
        intervention_target="features",
        mechanism="feature_bagging",
        expected_effect="improve utility",
    )
    for node_id in range(1, 5):
        idea_id = new_identifier("idea")
        batch_id = new_identifier("batch")
        add_completed_idea(
            archive,
            idea_id=idea_id,
            batch_id=batch_id,
            node_id=node_id,
            descriptor=shared,
        )
        experiments.append(
            experiment(
                node_id=node_id,
                idea_id=idea_id,
                batch_id=batch_id,
                score=0.5,
                created_at=f"2026-07-19T00:0{node_id}:00+00:00",
            )
        )
    snapshot = build_snapshot(archive, tuple(experiments))
    decision = choose_policy(snapshot, capacity())
    directive = plan_search_directive(
        decision,
        snapshot=snapshot,
        capacity=capacity(),
        archive_state=archive.state,
        gap_priorities=(),
        settings=operator_settings(),
    )

    assert decision.mode == IdeationMode.EXPLORE
    assert directive.operator_briefs[0].operator == OperatorKind.MECHANISM_SHIFT
    assert directive.operator_briefs[0].descriptor_target.approach_family.startswith(
        "alternative_to_"
    )


def test_crossover_has_one_implementation_parent_and_explicit_sources(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    first_idea_id = new_identifier("idea")
    second_idea_id = new_identifier("idea")
    first_batch_id = new_identifier("batch")
    second_batch_id = new_identifier("batch")
    add_completed_idea(
        archive,
        idea_id=first_idea_id,
        batch_id=first_batch_id,
        node_id=1,
        descriptor=IdeaDescriptor(
            approach_family="linear",
            intervention_target="features",
            mechanism="sparse_interactions",
            expected_effect="improve utility",
        ),
    )
    add_completed_idea(
        archive,
        idea_id=second_idea_id,
        batch_id=second_batch_id,
        node_id=2,
        descriptor=IdeaDescriptor(
            approach_family="tree",
            intervention_target="features",
            mechanism="nonlinear_splits",
            expected_effect="improve utility",
        ),
    )
    archive.record_claims(
        (
            EvidenceClaim(
                claim_id=new_identifier("claim"),
                statement="Nonlinear splits improved comparable utility.",
                kind=ClaimKind.OBSERVATION,
                status=EvidenceStatus.SUPPORTED,
                source_refs=("experiment:2",),
                affected_idea_ids=(second_idea_id,),
                affected_experiment_node_ids=(2,),
                updated_at=NOW,
            ),
        ),
        expected_revision=archive.revision,
    )
    snapshot = build_snapshot(
        archive,
        (
            experiment(
                node_id=1,
                idea_id=first_idea_id,
                batch_id=first_batch_id,
                score=0.50,
            ),
            experiment(
                node_id=2,
                idea_id=second_idea_id,
                batch_id=second_batch_id,
                score=0.52,
                created_at="2026-07-19T00:01:00+00:00",
            ),
        ),
    )
    decision = choose_policy(snapshot, capacity())
    directive = plan_search_directive(
        decision,
        snapshot=snapshot,
        capacity=capacity(),
        archive_state=archive.state,
        gap_priorities=(),
        settings=operator_settings(candidate_quota=3),
    )
    crossover = next(
        brief
        for brief in directive.operator_briefs
        if brief.operator == OperatorKind.CROSSOVER
    )

    assert decision.mode == IdeationMode.EXPLOIT
    assert crossover.parent_plan.kind == ParentPlanKind.BEST_VALID
    assert crossover.parent_plan.experiment_node_id is None
    assert crossover.parent_plan.source_idea_ids == (first_idea_id,)
    assert crossover.parent_plan.source_experiment_node_ids == (1,)


def test_recovery_reuses_failed_branch_and_hypothesis(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    idea_id = new_identifier("idea")
    batch_id = new_identifier("batch")
    batch = replace(
        planned_batch(),
        batch_id=batch_id,
        iteration_index=1,
        context_hash="7" * 64,
    )
    idea = replace(generated_idea(idea_id), origin_batch_id=batch_id)
    archive.create_batch(batch, expected_revision=0)
    archive.add_ideas(batch_id, (idea,), expected_revision=1)
    archive.record_analysis(
        batch_id,
        CandidateAnalysis(idea_id=idea_id, eligible=True),
        expected_revision=2,
    )
    decision = replace(
        selection(),
        selected_idea_id=idea_id,
        dispositions=(
            CandidateDisposition(
                idea_id,
                CandidateDispositionKind.SELECTED,
                "Selected before technical failure.",
            ),
        ),
    )
    archive.record_selection(batch_id, decision, expected_revision=3)
    archive.link_experiment(idea_id, 1, batch_id, expected_revision=4)
    failed = ExperimentInput(
        node_id=1,
        idea_id=idea_id,
        selection_batch_id=batch_id,
        parent_node_id=0,
        proposal=idea.proposal,
        score=None,
        evaluation_valid=True,
        had_error=True,
        recoverable_error=True,
        build_fidelity="full",
        attempts=(),
        feedback="",
        technical_difficulty="Dependency installation timed out.",
        created_at=NOW,
    )
    snapshot = build_snapshot(archive, (failed,))
    policy = choose_policy(snapshot, capacity())
    directive = plan_search_directive(
        policy,
        snapshot=snapshot,
        capacity=capacity(),
        archive_state=archive.state,
        gap_priorities=(),
        settings=operator_settings(),
    )
    brief = directive.operator_briefs[0]

    assert policy.mode == IdeationMode.RECOVER
    assert brief.operator == OperatorKind.RECOVER
    assert brief.parent_plan.kind == ParentPlanKind.RECOVER_BRANCH
    assert brief.parent_plan.experiment_node_id == 1
    assert brief.descriptor_target == idea.descriptor


def test_verification_preserves_the_target_hypothesis(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    idea_id = new_identifier("idea")
    batch_id = new_identifier("batch")
    add_completed_idea(
        archive,
        idea_id=idea_id,
        batch_id=batch_id,
        node_id=1,
    )
    validation_capacity = capacity(
        fidelity_profile="validate",
        target_node_id=1,
    )
    snapshot = build_snapshot(
        archive,
        (
            experiment(
                node_id=1,
                idea_id=idea_id,
                batch_id=batch_id,
                score=0.5,
            ),
        ),
        validation_capacity,
    )
    policy = choose_policy(snapshot, validation_capacity)
    directive = plan_search_directive(
        policy,
        snapshot=snapshot,
        capacity=validation_capacity,
        archive_state=archive.state,
        gap_priorities=(),
        settings=operator_settings(),
    )
    brief = directive.operator_briefs[0]

    assert policy.mode == IdeationMode.VERIFY
    assert brief.operator == OperatorKind.VERIFY
    assert brief.parent_plan.experiment_node_id == 1
    assert brief.descriptor_target == archive.get_idea(idea_id).descriptor


def test_deferred_idea_resurfaces_only_after_a_relevant_condition_changes():
    idea = replace(
        generated_idea(),
        status=IdeaStatus.DEFERRED,
        deferral_reason="Insufficient comparable capacity.",
    )
    assert (
        find_resurfaceable_ideas(
            (idea,),
            changed_conditions_by_idea={},
            newly_resolved_gap_ids=(),
        )
        == ()
    )

    resurfaced = find_resurfaceable_ideas(
        (idea,),
        changed_conditions_by_idea={
            idea.idea_id: ("comparable capacity is now available",)
        },
        newly_resolved_gap_ids=(),
    )
    assert resurfaced[0].idea_id == idea.idea_id
    assert resurfaced[0].changed_conditions == ("comparable capacity is now available",)
