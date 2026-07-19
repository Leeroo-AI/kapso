"""Behavioral tests for explicit evaluator-authored ideation evidence."""

from dataclasses import replace

import pytest

from kapso.execution.search_strategies.generic.ideation import (
    CampaignEvidenceBuilder,
    CandidateAnalysis,
    CandidateDisposition,
    CandidateDispositionKind,
    EvaluationGap,
    EvaluationStatus,
    EvidenceSignal,
    GapState,
    IdeaArchive,
    IdeaOutcome,
    IdeaStatus,
    IdeationMode,
    ImplementationStatus,
    ObjectiveDirection,
    new_identifier,
)
from kapso.execution.search_strategies.generic.ideation.evaluator_evidence import (
    EVALUATOR_EVIDENCE_KEY,
    build_evaluator_evidence_writeback,
)
from kapso.execution.search_strategies.generic.ideation.policy import choose_policy
from test_ideation_domain import (
    EVIDENCE_ID,
    NOW,
    analyzed_candidate,
    coding_agent_call,
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
)


def add_implementing_idea(
    archive: IdeaArchive,
    *,
    idea_id: str,
    batch_id: str,
    node_id: int,
    target_gap_ids: tuple[str, ...] = (),
):
    batch = replace(
        planned_batch(),
        batch_id=batch_id,
        iteration_index=node_id,
        context_hash=f"{node_id:064x}",
        planning_archive_revision=archive.revision,
    )
    idea = replace(
        generated_idea(idea_id),
        origin_batch_id=batch_id,
        target_gap_ids=target_gap_ids,
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
    return archive.get_idea(idea_id)


def evidence_metadata(*, claims=(), open_gaps=(), targeted_gap_updates=()):
    return {
        EVALUATOR_EVIDENCE_KEY: {
            "claims": list(claims),
            "open_gaps": list(open_gaps),
            "targeted_gap_updates": list(targeted_gap_updates),
        }
    }


def supported_claim():
    return {
        "statement": "Gradient clipping caused the measured stability gain.",
        "kind": "hypothesis",
        "status": "supported",
        "source_refs": ["metric:validation_stability"],
    }


def open_gap():
    return {
        "axis": "seed stability",
        "description": "The gain has not been tested across seeds.",
        "evidence_refs": ["metric:validation_stability"],
        "impact": 0.8,
        "uncertainty": 0.9,
        "estimated_cost": 1.0,
    }


def test_metadata_without_ideation_evidence_produces_no_causal_writeback(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    idea = replace(
        generated_idea(),
        status=IdeaStatus.IMPLEMENTING,
        selected_in_batch_id=generated_idea().origin_batch_id,
        selection_reason="Selected for canonical evaluation.",
        experiment_node_id=1,
    )

    writeback = build_evaluator_evidence_writeback(
        {"suite": "canonical-v1", "score": 0.9},
        idea=idea,
        archive_state=archive.state,
        observed_at=NOW,
    )

    assert writeback.claim_updates == ()
    assert writeback.gap_updates == ()


@pytest.mark.parametrize(
    ("metadata", "observed_at"),
    (
        ({EVALUATOR_EVIDENCE_KEY: {"claims": []}}, NOW),
        (
            evidence_metadata(
                claims=(
                    {
                        **supported_claim(),
                        "status": "insufficient",
                    },
                )
            ),
            NOW,
        ),
        (evidence_metadata(), ""),
    ),
)
def test_present_malformed_ideation_evidence_fails_loudly(
    tmp_path,
    metadata,
    observed_at,
):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    idea = replace(
        generated_idea(),
        status=IdeaStatus.IMPLEMENTING,
        selected_in_batch_id=generated_idea().origin_batch_id,
        selection_reason="Selected for canonical evaluation.",
        experiment_node_id=1,
    )

    with pytest.raises(ValueError):
        build_evaluator_evidence_writeback(
            metadata,
            idea=idea,
            archive_state=archive.state,
            observed_at=observed_at,
        )


def test_supported_evaluator_claim_makes_next_policy_exploit(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    prior_idea_id = new_identifier("idea")
    prior_batch_id = new_identifier("batch")
    current_idea_id = new_identifier("idea")
    current_batch_id = new_identifier("batch")
    add_completed_idea(
        archive,
        idea_id=prior_idea_id,
        batch_id=prior_batch_id,
        node_id=1,
    )
    current_idea = add_implementing_idea(
        archive,
        idea_id=current_idea_id,
        batch_id=current_batch_id,
        node_id=2,
    )
    writeback = build_evaluator_evidence_writeback(
        evidence_metadata(claims=(supported_claim(),)),
        idea=current_idea,
        archive_state=archive.state,
        observed_at="2026-07-19T00:02:00+00:00",
    )
    repeated = build_evaluator_evidence_writeback(
        evidence_metadata(claims=(supported_claim(),)),
        idea=current_idea,
        archive_state=archive.state,
        observed_at="2026-07-19T00:02:00+00:00",
    )
    classified_outcome = writeback.apply_to_outcome(
        IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=0.02,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        )
    )
    state_with_evidence = replace(
        archive.state,
        claims=writeback.claim_updates,
    )

    assert repeated == writeback
    assert classified_outcome.supported_claim_ids == (
        writeback.claim_updates[0].claim_id,
    )
    snapshot = CampaignEvidenceBuilder(evidence_settings()).build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(
            experiment(
                node_id=1,
                idea_id=prior_idea_id,
                batch_id=prior_batch_id,
                score=0.5,
            ),
            experiment(
                node_id=2,
                idea_id=current_idea_id,
                batch_id=current_batch_id,
                score=0.52,
                created_at="2026-07-19T00:01:00+00:00",
            ),
        ),
        archive_state=state_with_evidence,
        capacity=capacity(),
        generated_at="2026-07-19T00:03:00+00:00",
    )

    assert EvidenceSignal.CREDIBLE_IMPROVEMENT in snapshot.signals
    assert EvidenceSignal.SUPPORTED_LEVER in snapshot.signals
    assert choose_policy(snapshot, capacity()).mode == IdeationMode.EXPLOIT


def test_evaluator_open_gap_becomes_gap_debt_after_explicit_deferrals(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    idea_id = new_identifier("idea")
    batch_id = new_identifier("batch")
    idea = add_implementing_idea(
        archive,
        idea_id=idea_id,
        batch_id=batch_id,
        node_id=1,
    )
    writeback = build_evaluator_evidence_writeback(
        evidence_metadata(open_gaps=(open_gap(),)),
        idea=idea,
        archive_state=archive.state,
        observed_at=NOW,
    )
    deferred_gap = replace(
        writeback.gap_updates[0],
        deferral_count=evidence_settings().gap_debt_threshold,
        last_considered_at="2026-07-19T00:01:00+00:00",
    )
    state_with_gap_debt = replace(archive.state, gaps=(deferred_gap,))

    snapshot = CampaignEvidenceBuilder(evidence_settings()).build(
        campaign_id=CAMPAIGN_ID,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(
            experiment(
                node_id=1,
                idea_id=idea_id,
                batch_id=batch_id,
                score=0.5,
            ),
        ),
        archive_state=state_with_gap_debt,
        capacity=capacity(),
        generated_at="2026-07-19T00:02:00+00:00",
    )

    assert EvidenceSignal.GAP_DEBT in snapshot.signals
    assert choose_policy(snapshot, capacity()).mode == IdeationMode.EXPLORE


def test_targeted_gap_update_is_mechanical_and_provenanced(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    gap = EvaluationGap(
        gap_id=new_identifier("gap"),
        axis="seed stability",
        description="Seed variance is unknown.",
        state=GapState.OPEN,
        evidence_refs=(EVIDENCE_ID,),
        impact=0.8,
        uncertainty=0.9,
        estimated_cost=1.0,
        deferral_count=0,
        opened_at=NOW,
    )
    archive.add_gaps((gap,), expected_revision=archive.revision)
    idea = add_implementing_idea(
        archive,
        idea_id=new_identifier("idea"),
        batch_id=new_identifier("batch"),
        node_id=1,
        target_gap_ids=(gap.gap_id,),
    )
    writeback = build_evaluator_evidence_writeback(
        evidence_metadata(
            targeted_gap_updates=(
                {
                    "gap_id": gap.gap_id,
                    "state": "closed",
                    "evidence_refs": ["metric:seed_variance"],
                    "closure_reason": "Canonical multi-seed evaluation resolved it.",
                },
            )
        ),
        idea=idea,
        archive_state=archive.state,
        observed_at="2026-07-19T00:02:00+00:00",
    )

    update = writeback.gap_updates[0]
    outcome = writeback.apply_to_outcome(
        IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=0.02,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        )
    )
    archive.record_outcome(
        idea.idea_id,
        outcome,
        gap_updates=writeback.gap_updates,
        expected_revision=archive.revision,
    )
    repeated = build_evaluator_evidence_writeback(
        evidence_metadata(
            targeted_gap_updates=(
                {
                    "gap_id": gap.gap_id,
                    "state": "closed",
                    "evidence_refs": ["metric:seed_variance"],
                    "closure_reason": "Canonical multi-seed evaluation resolved it.",
                },
            )
        ),
        idea=archive.get_idea(idea.idea_id),
        archive_state=archive.state,
        observed_at="2026-07-19T00:02:00+00:00",
    )
    assert update.state == GapState.CLOSED
    assert update.resolution_idea_id == idea.idea_id
    assert update.resolution_experiment_node_id == 1
    assert "experiment_node:1" in update.evidence_refs
    assert repeated.gap_updates == (update,)


def test_open_gap_outcome_replay_accepts_a_later_legal_resolution(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    opening_idea = add_implementing_idea(
        archive,
        idea_id=new_identifier("idea"),
        batch_id=new_identifier("batch"),
        node_id=1,
    )
    opening_metadata = evidence_metadata(open_gaps=(open_gap(),))
    opening_writeback = build_evaluator_evidence_writeback(
        opening_metadata,
        idea=opening_idea,
        archive_state=archive.state,
        observed_at=NOW,
    )
    opening_outcome = opening_writeback.apply_to_outcome(
        IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=0.02,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        )
    )
    archive.record_outcome(
        opening_idea.idea_id,
        opening_outcome,
        gap_updates=opening_writeback.gap_updates,
        expected_revision=archive.revision,
    )
    gap_id = opening_writeback.gap_updates[0].gap_id
    resolving_idea = add_implementing_idea(
        archive,
        idea_id=new_identifier("idea"),
        batch_id=new_identifier("batch"),
        node_id=2,
        target_gap_ids=(gap_id,),
    )
    resolution_writeback = build_evaluator_evidence_writeback(
        evidence_metadata(
            targeted_gap_updates=(
                {
                    "gap_id": gap_id,
                    "state": "closed",
                    "evidence_refs": ["metric:multi_seed_stability"],
                    "closure_reason": "Multi-seed evaluation resolved the gap.",
                },
            )
        ),
        idea=resolving_idea,
        archive_state=archive.state,
        observed_at="2026-07-19T00:02:00+00:00",
    )
    resolution_outcome = resolution_writeback.apply_to_outcome(
        IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=0.01,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        )
    )
    archive.record_outcome(
        resolving_idea.idea_id,
        resolution_outcome,
        gap_updates=resolution_writeback.gap_updates,
        expected_revision=archive.revision,
    )
    revision_after_resolution = archive.revision
    assert (
        archive.record_outcome(
            opening_idea.idea_id,
            opening_outcome,
            gap_updates=opening_writeback.gap_updates,
            expected_revision=revision_after_resolution,
        )
        == revision_after_resolution
    )
    assert (
        archive.add_gaps(
            opening_writeback.gap_updates,
            expected_revision=revision_after_resolution,
        )
        == revision_after_resolution
    )

    replayed = build_evaluator_evidence_writeback(
        opening_metadata,
        idea=archive.get_idea(opening_idea.idea_id),
        archive_state=archive.state,
        observed_at=NOW,
    )
    assert replayed.gap_updates[0].state == GapState.CLOSED
    assert (
        archive.record_outcome(
            opening_idea.idea_id,
            opening_outcome,
            gap_updates=replayed.gap_updates,
            expected_revision=revision_after_resolution,
        )
        == revision_after_resolution
    )
