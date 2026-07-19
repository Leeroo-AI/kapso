"""Evidence normalization, comparability, signals, and gap priority tests."""

import statistics
from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.execution.search_strategies.generic.ideation import (
    CandidateAnalysis,
    CandidateDisposition,
    CandidateDispositionKind,
    ClaimKind,
    EvaluationGap,
    EvaluationStatus,
    EvidenceClaim,
    EvidenceSignal,
    EvidenceStatus,
    GapState,
    IdeaArchive,
    IdeaDescriptor,
    IdeaOutcome,
    IdeationCapacityView,
    ImplementationStatus,
    ObjectiveDirection,
    SelectionDecision,
    new_identifier,
)
from kapso.execution.search_strategies.generic.ideation.evidence import (
    CampaignEvidenceBuilder,
    EvaluationAttemptInput,
    EvidenceSettings,
    ExperimentInput,
    GapPrioritySettings,
    rank_evaluation_gaps,
)
from test_ideation_domain import (
    CAPACITY_ID,
    EVIDENCE_ID,
    NOW,
    analyzed_candidate,
    coding_agent_call,
    generated_idea,
    planned_batch,
    selection,
)

CAMPAIGN_ID = "campaign-alpha"


def capacity(**changes) -> IdeationCapacityView:
    values = {
        "capacity_snapshot_id": CAPACITY_ID,
        "iteration_index": 1,
        "max_iterations": 10,
        "remaining_seconds": 900.0,
        "remaining_after_reserve_seconds": 600.0,
        "remaining_usd": 5.0,
        "fidelity_profile": "full",
        "build_fidelity": "full",
        "eval_fidelity": "full",
        "eval_fraction": 1.0,
        "target_node_id": None,
        "reserve_run": False,
        "deadline_seconds": 300.0,
        "can_start_complete_action": True,
        "can_run_granted_evaluation": True,
        "can_run_comparable_evaluation": True,
        "preserves_finalization_reserve": True,
    }
    values.update(changes)
    return IdeationCapacityView(**values)


def evidence_settings() -> EvidenceSettings:
    return EvidenceSettings(
        evaluator_id="evaluator-v1",
        comparable_fidelity="full",
        comparable_fraction=1.0,
        comparable_seed=7,
        minimum_repeat_measurements=2,
        minimum_credible_delta=0.01,
        surprising_gain_multiplier=3.0,
        proxy_divergence_threshold=0.03,
        plateau_window=3,
        diversity_window=4,
        gap_debt_threshold=2,
    )


def gap_settings() -> GapPrioritySettings:
    return GapPrioritySettings(
        default_evidence_confidence=0.25,
        default_uncertainty_reduction=0.25,
        default_cost=1.0,
        minimum_cost=0.01,
    )


def add_completed_idea(
    archive: IdeaArchive,
    *,
    idea_id: str,
    batch_id: str,
    node_id: int,
    descriptor: IdeaDescriptor | None = None,
) -> None:
    batch = replace(
        planned_batch(),
        batch_id=batch_id,
        iteration_index=node_id,
        context_hash=f"{node_id:064x}",
    )
    idea = replace(
        generated_idea(idea_id),
        origin_batch_id=batch_id,
        descriptor=descriptor or generated_idea().descriptor,
    )
    archive.create_batch(batch, expected_revision=archive.revision)
    archive.add_ideas(
        batch_id,
        (idea,),
        generation_calls=(coding_agent_call(),),
        expected_revision=archive.revision,
    )
    archive.record_analyses(
        batch_id,
        (analyzed_candidate(CandidateAnalysis(idea_id=idea_id, eligible=True)),),
        expected_revision=archive.revision,
    )
    decision = replace(
        selection(),
        selected_idea_id=idea_id,
        dispositions=(
            CandidateDisposition(
                idea_id=idea_id,
                disposition=CandidateDispositionKind.SELECTED,
                reason="Best eligible evidence-adjusted utility.",
            ),
        ),
    )
    archive.record_selection(
        batch_id,
        decision,
        selection_call=coding_agent_call(),
        expected_revision=archive.revision,
    )
    archive.link_experiment(
        idea_id,
        node_id,
        batch_id,
        expected_revision=archive.revision,
    )
    archive.record_outcome(
        idea_id,
        IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=0.0,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        ),
        expected_revision=archive.revision,
    )


def experiment(
    *,
    node_id: int,
    idea_id: str,
    batch_id: str,
    score: float,
    attempt_scores: tuple[float, ...] | None = None,
    seed: int = 7,
    fidelity: str = "full",
    created_at: str = NOW,
) -> ExperimentInput:
    observed_scores = attempt_scores or (score,)
    return ExperimentInput(
        node_id=node_id,
        idea_id=idea_id,
        selection_batch_id=batch_id,
        parent_node_id=None if node_id == 1 else node_id - 1,
        proposal="Measure a typed intervention.",
        score=score,
        evaluation_valid=True,
        had_error=False,
        recoverable_error=False,
        build_fidelity="full",
        attempts=tuple(
            EvaluationAttemptInput(
                evaluator_id="evaluator-v1",
                fidelity=fidelity,
                fraction=1.0,
                seed=seed,
                score=attempt_score,
                duration_seconds=10.0,
            )
            for attempt_score in observed_scores
        ),
        feedback="Observed canonical utility.",
        technical_difficulty=None,
        created_at=created_at,
    )


def test_cold_start_is_content_addressed_and_explicitly_insufficient(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    builder = CampaignEvidenceBuilder(evidence_settings())
    first = builder.build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(),
        archive_state=archive.state,
        capacity=capacity(),
        generated_at=NOW,
    )
    second = builder.build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(),
        archive_state=archive.state,
        capacity=capacity(),
        generated_at="2026-07-19T00:01:00+00:00",
    )

    assert first.snapshot_id == second.snapshot_id
    assert first.content_hash == second.content_hash
    assert first.noise_floor is None
    assert EvidenceSignal.NO_COMPARABLE_EXPERIMENT in first.signals
    assert EvidenceSignal.PROVISIONAL_NOISE in first.signals


def test_maximize_and_minimize_histories_produce_same_normalized_trace(tmp_path):
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
    )
    add_completed_idea(
        archive,
        idea_id=second_idea_id,
        batch_id=second_batch_id,
        node_id=2,
    )
    builder = CampaignEvidenceBuilder(evidence_settings())
    maximize = builder.build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(
            experiment(
                node_id=1,
                idea_id=first_idea_id,
                batch_id=first_batch_id,
                score=0.4,
            ),
            experiment(
                node_id=2,
                idea_id=second_idea_id,
                batch_id=second_batch_id,
                score=0.6,
                created_at="2026-07-19T00:01:00+00:00",
            ),
        ),
        archive_state=archive.state,
        capacity=capacity(),
        generated_at=NOW,
    )
    minimize = builder.build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MINIMIZE,
        experiments=(
            experiment(
                node_id=1,
                idea_id=first_idea_id,
                batch_id=first_batch_id,
                score=-0.4,
            ),
            experiment(
                node_id=2,
                idea_id=second_idea_id,
                batch_id=second_batch_id,
                score=-0.6,
                created_at="2026-07-19T00:01:00+00:00",
            ),
        ),
        archive_state=archive.state,
        capacity=capacity(),
        generated_at=NOW,
    )

    assert [item.normalized_utility for item in maximize.experiments] == [
        item.normalized_utility for item in minimize.experiments
    ]
    assert maximize.incumbent_node_id == minimize.incumbent_node_id == 2


def test_score_presence_without_matching_comparability_metadata_is_not_comparable(
    tmp_path,
):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    idea_id = new_identifier("idea")
    batch_id = new_identifier("batch")
    add_completed_idea(
        archive,
        idea_id=idea_id,
        batch_id=batch_id,
        node_id=1,
    )
    snapshot = CampaignEvidenceBuilder(evidence_settings()).build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(
            experiment(
                node_id=1,
                idea_id=idea_id,
                batch_id=batch_id,
                score=0.8,
                seed=99,
            ),
        ),
        archive_state=archive.state,
        capacity=capacity(),
        generated_at=NOW,
    )

    assert not snapshot.experiments[0].comparable
    assert snapshot.incumbent_node_id is None
    assert EvidenceSignal.NO_COMPARABLE_EXPERIMENT in snapshot.signals


def test_only_supported_hypotheses_unlock_a_supported_lever(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    idea_id = new_identifier("idea")
    batch_id = new_identifier("batch")
    add_completed_idea(
        archive,
        idea_id=idea_id,
        batch_id=batch_id,
        node_id=1,
    )
    observation = EvidenceClaim(
        claim_id=new_identifier("claim"),
        statement="The registered score increased.",
        kind=ClaimKind.OBSERVATION,
        status=EvidenceStatus.SUPPORTED,
        source_refs=("experiment_node:1",),
        affected_idea_ids=(idea_id,),
        affected_experiment_node_ids=(1,),
        updated_at=NOW,
    )
    archive.record_claims((observation,), expected_revision=archive.revision)
    inputs = (
        experiment(
            node_id=1,
            idea_id=idea_id,
            batch_id=batch_id,
            score=0.6,
        ),
    )
    builder = CampaignEvidenceBuilder(evidence_settings())
    observation_snapshot = builder.build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=inputs,
        archive_state=archive.state,
        capacity=capacity(),
        generated_at=NOW,
    )

    assert EvidenceSignal.SUPPORTED_LEVER not in observation_snapshot.signals

    hypothesis = EvidenceClaim(
        claim_id=new_identifier("claim"),
        statement="The intervention mechanism caused the registered gain.",
        kind=ClaimKind.HYPOTHESIS,
        status=EvidenceStatus.SUPPORTED,
        source_refs=("experiment_node:1",),
        affected_idea_ids=(idea_id,),
        affected_experiment_node_ids=(1,),
        updated_at=NOW,
    )
    archive.record_claims((hypothesis,), expected_revision=archive.revision)
    hypothesis_snapshot = builder.build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=inputs,
        archive_state=archive.state,
        capacity=capacity(),
        generated_at=NOW,
    )

    assert EvidenceSignal.SUPPORTED_LEVER in hypothesis_snapshot.signals


def test_repeat_measurements_define_noise_and_proxy_divergence_is_typed(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    idea_id = new_identifier("idea")
    batch_id = new_identifier("batch")
    add_completed_idea(
        archive,
        idea_id=idea_id,
        batch_id=batch_id,
        node_id=1,
    )
    full = experiment(
        node_id=1,
        idea_id=idea_id,
        batch_id=batch_id,
        score=0.6,
        attempt_scores=(0.5, 0.7),
    )
    proxy = EvaluationAttemptInput(
        evaluator_id="evaluator-v1",
        fidelity="fast",
        fraction=1.0,
        seed=7,
        score=0.2,
        duration_seconds=2.0,
    )
    snapshot = CampaignEvidenceBuilder(evidence_settings()).build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(replace(full, attempts=(proxy, *full.attempts)),),
        archive_state=archive.state,
        capacity=capacity(),
        generated_at=NOW,
    )

    assert snapshot.noise_floor == pytest.approx(statistics.stdev((0.5, 0.7)))
    assert EvidenceSignal.PROXY_FULL_DIVERGENCE in snapshot.signals
    assert EvidenceSignal.PROVISIONAL_NOISE not in snapshot.signals


def test_gap_priority_uses_impact_before_age_and_records_defaults():
    old_low_impact = EvaluationGap(
        gap_id=new_identifier("gap"),
        axis="old axis",
        description="Old but low impact.",
        state=GapState.OPEN,
        evidence_refs=(EVIDENCE_ID,),
        impact=0.1,
        uncertainty=0.5,
        estimated_cost=None,
        deferral_count=5,
        opened_at="2026-07-01T00:00:00+00:00",
        last_considered_at=NOW,
    )
    new_high_impact = EvaluationGap(
        gap_id=new_identifier("gap"),
        axis="high impact axis",
        description="New and high impact.",
        state=GapState.OPEN,
        evidence_refs=(EVIDENCE_ID,),
        impact=0.9,
        uncertainty=0.8,
        estimated_cost=1.0,
        deferral_count=0,
        opened_at="2026-07-18T00:00:00+00:00",
    )
    ranked = rank_evaluation_gaps(
        (old_low_impact, new_high_impact),
        evidence_confidence_by_gap={new_high_impact.gap_id: 0.8},
        uncertainty_reduction_by_gap={new_high_impact.gap_id: 0.7},
        as_of=NOW,
        settings=gap_settings(),
    )

    assert ranked[0].gap_id == new_high_impact.gap_id
    assert "default cost" in ranked[1].assumptions
    assert ranked[1].deferral_count == 5


def test_shipped_modes_share_one_ideation_configuration_source():
    config = load_config("src/kapso/config.yaml")
    assert "ideation_defaults" not in config
    defaults = config["ideation_profiles"]["DEFAULT"]
    assert defaults["archive_path"] == ".kapso/idea_archive.json"
    assert config["modes"]["GENERIC"]["ideation_profile"] == "DEFAULT"
    assert config["modes"]["MINIMAL"]["ideation_profile"] == "DEFAULT"


def test_gap_deferral_increases_debt_without_changing_gap_state(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    gap = EvaluationGap(
        gap_id=new_identifier("gap"),
        axis="seed stability",
        description="Seed variance is unknown.",
        state=GapState.OPEN,
        evidence_refs=(EVIDENCE_ID,),
        impact=0.5,
        uncertainty=0.8,
        estimated_cost=1.0,
        deferral_count=0,
        opened_at=NOW,
    )
    archive.add_gaps((gap,), expected_revision=0)
    archive.defer_gap(
        gap.gap_id,
        "2026-07-19T00:01:00+00:00",
        expected_deferral_count=0,
        expected_revision=1,
    )
    archive.defer_gap(
        gap.gap_id,
        "2026-07-19T00:02:00+00:00",
        expected_deferral_count=1,
        expected_revision=2,
    )
    deferred = archive.list_gaps()[0]
    assert deferred.deferral_count == 2
    assert deferred.state == GapState.OPEN
