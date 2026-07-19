"""Outcome projection tests for finalized linked experiments."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation import (
    CampaignEvidenceBuilder,
    EvaluationAttemptInput,
    EvaluationStatus,
    EvidenceSignal,
    ExperimentInput,
    GapState,
    IdeaStatus,
    IdeationMode,
    ImplementationStatus,
    ObjectiveDirection,
    build_idea_outcome,
)
from kapso.execution.search_strategies.generic.ideation.policy import choose_policy
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from test_ideation_domain import BATCH_ID, generated_idea
from test_ideation_engine import (
    CountingParents,
    PacketRunner,
    build_engine,
    run_engine,
)
from test_ideation_evidence import capacity, evidence_settings


def evaluation_attempt(
    score,
    *,
    evaluator_id="evaluator-v1",
    fidelity="full",
    fraction=1.0,
    seed=7,
):
    return EvaluationAttempt(
        commit_sha="a" * 40,
        evaluator_id=evaluator_id,
        fidelity=fidelity,
        fraction=fraction,
        seed=seed,
        score=score,
    )


def linked(node_id=0, *, parent=None):
    idea = generated_idea()
    if parent is not None:
        idea = replace(idea, resolved_parent=parent)
    idea = replace(
        idea,
        status=IdeaStatus.IMPLEMENTING,
        selected_in_batch_id=BATCH_ID,
        selection_reason="Selected by audited evidence.",
        experiment_node_id=node_id,
    )
    node = SearchNode(
        node_id=node_id,
        idea_id=idea.idea_id,
        selection_batch_id=BATCH_ID,
        solution=idea.proposal,
        score=0.7,
        build_fidelity="full",
        eval_fidelity="full",
        evaluation_attempts=[evaluation_attempt(0.7)],
        duration_seconds=5.0,
        cost_usd=0.3,
    )
    return idea, node


def test_valid_bootstrap_outcome_uses_objective_normalized_zero_basis():
    idea, node = linked()

    outcome = build_idea_outcome(
        node=node,
        idea=idea,
        nodes_by_id={0: node},
        objective_direction=ObjectiveDirection.MAXIMIZE,
    )

    assert outcome.evaluation_status == EvaluationStatus.VALID
    assert outcome.implementation_status == ImplementationStatus.COMPLETED
    assert outcome.normalized_delta == 0.7
    assert outcome.validation_tier == "full"


def test_valid_child_delta_uses_frozen_parent_and_minimize_direction():
    parent = SearchNode(
        node_id=0,
        score=0.6,
        evaluation_attempts=[evaluation_attempt(0.6)],
    )
    idea, child = linked(node_id=1)
    frozen_parent = replace(idea.resolved_parent, node_id=0)
    idea = replace(idea, resolved_parent=frozen_parent)
    child.score = 0.4
    child.evaluation_attempts = [evaluation_attempt(0.4)]

    outcome = build_idea_outcome(
        node=child,
        idea=idea,
        nodes_by_id={0: parent, 1: child},
        objective_direction=ObjectiveDirection.MINIMIZE,
    )

    assert outcome.normalized_delta == pytest.approx(0.2)


@pytest.mark.parametrize(
    "parent_attempt",
    [
        evaluation_attempt(0.6, evaluator_id="evaluator-v2"),
        evaluation_attempt(0.6, fidelity="fast"),
        evaluation_attempt(0.6, fraction=0.2),
        evaluation_attempt(0.6, seed=8),
    ],
    ids=["evaluator", "fidelity", "fraction", "seed"],
)
def test_incomparable_parent_measurement_makes_outcome_inconclusive(
    parent_attempt,
):
    parent = SearchNode(
        node_id=0,
        score=parent_attempt.score,
        eval_fidelity=parent_attempt.fidelity,
        evaluation_attempts=[parent_attempt],
    )
    idea, child = linked(node_id=1)
    idea = replace(
        idea,
        resolved_parent=replace(idea.resolved_parent, node_id=0),
    )

    outcome = build_idea_outcome(
        node=child,
        idea=idea,
        nodes_by_id={0: parent, 1: child},
        objective_direction=ObjectiveDirection.MAXIMIZE,
    )

    assert outcome.evaluation_status == EvaluationStatus.INCONCLUSIVE
    assert outcome.normalized_delta is None


def test_score_without_comparability_metadata_is_inconclusive():
    idea, node = linked()
    node.evaluation_attempts = []

    outcome = build_idea_outcome(
        node=node,
        idea=idea,
        nodes_by_id={0: node},
        objective_direction=ObjectiveDirection.MAXIMIZE,
    )

    assert outcome.evaluation_status == EvaluationStatus.INCONCLUSIVE
    assert outcome.normalized_delta is None


def test_invalid_and_inconclusive_results_have_no_hypothesis_delta():
    idea, invalid = linked()
    invalid.evaluation_valid = False
    invalid_outcome = build_idea_outcome(
        node=invalid,
        idea=idea,
        nodes_by_id={0: invalid},
        objective_direction=ObjectiveDirection.MAXIMIZE,
    )
    idea, inconclusive = linked()
    inconclusive.score = None
    inconclusive_outcome = build_idea_outcome(
        node=inconclusive,
        idea=idea,
        nodes_by_id={0: inconclusive},
        objective_direction=ObjectiveDirection.MAXIMIZE,
    )

    assert invalid_outcome.evaluation_status == EvaluationStatus.INVALID
    assert invalid_outcome.normalized_delta is None
    assert inconclusive_outcome.evaluation_status == EvaluationStatus.INCONCLUSIVE
    assert inconclusive_outcome.normalized_delta is None


def test_recoverable_failure_stays_open_without_terminal_outcome():
    idea, node = linked()
    node.score = None
    node.had_error = True
    node.recoverable_error = True
    node.evaluation_valid = False

    assert (
        build_idea_outcome(
            node=node,
            idea=idea,
            nodes_by_id={0: node},
            objective_direction=ObjectiveDirection.MAXIMIZE,
        )
        is None
    )


def test_generic_strategy_writes_final_outcome_to_the_linked_archive(tmp_path):
    archive, engine = build_engine(tmp_path, PacketRunner(tmp_path))
    selected = run_engine(engine, CountingParents(tmp_path))
    archive.link_experiment(
        selected.selected_idea.idea_id,
        0,
        selected.batch_id,
        expected_revision=archive.revision,
    )
    node = SearchNode(
        node_id=0,
        idea_id=selected.selected_idea.idea_id,
        selection_batch_id=selected.batch_id,
        solution=selected.selected_idea.proposal,
        score=0.8,
        evaluation_attempts=[evaluation_attempt(0.8)],
        duration_seconds=4.0,
        cost_usd=0.2,
    )
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.idea_archive = archive
    strategy.node_history = [node]
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy._ensure_idea_archive = lambda: archive

    strategy.record_finalized_idea_outcome(node)

    persisted = next(
        idea
        for idea in archive.state.ideas
        if idea.idea_id == selected.selected_idea.idea_id
    )
    assert persisted.status == IdeaStatus.EVALUATED
    assert persisted.outcome.normalized_delta == 0.8


def test_generic_strategy_writes_explicit_evaluator_evidence_for_next_policy(
    tmp_path,
):
    archive, engine = build_engine(tmp_path, PacketRunner(tmp_path))
    selected = run_engine(engine, CountingParents(tmp_path))
    archive.link_experiment(
        selected.selected_idea.idea_id,
        0,
        selected.batch_id,
        expected_revision=archive.revision,
    )
    node = SearchNode(
        node_id=0,
        idea_id=selected.selected_idea.idea_id,
        selection_batch_id=selected.batch_id,
        solution=selected.selected_idea.proposal,
        score=0.8,
        evaluation_attempts=[evaluation_attempt(0.8)],
        duration_seconds=4.0,
        cost_usd=0.2,
        started_at="2026-07-19T00:01:00+00:00",
        external_evaluation_metadata={
            "ideation_evidence": {
                "claims": [
                    {
                        "statement": "Gradient clipping caused the stability gain.",
                        "kind": "hypothesis",
                        "status": "supported",
                        "source_refs": ["metric:validation_stability"],
                    }
                ],
                "open_gaps": [
                    {
                        "axis": "seed stability",
                        "description": "The gain is untested across seeds.",
                        "evidence_refs": ["metric:validation_stability"],
                        "impact": 0.8,
                        "uncertainty": 0.9,
                        "estimated_cost": 1.0,
                    }
                ],
                "targeted_gap_updates": [],
            }
        },
    )
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.idea_archive = archive
    strategy.node_history = [node]
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy._ensure_idea_archive = lambda: archive

    strategy.record_finalized_idea_outcome(node)
    completed_revision = archive.revision
    strategy.record_finalized_idea_outcome(node)
    assert archive.revision == completed_revision
    claim = archive.state.claims[0]
    archive.record_claims(
        (
            replace(
                claim,
                source_refs=(*claim.source_refs, "metric:replicated_stability"),
                updated_at="2026-07-19T00:02:00+00:00",
            ),
        ),
        expected_revision=archive.revision,
    )
    gap = archive.list_gaps()[0]
    archive.defer_gap(
        gap.gap_id,
        "2026-07-19T00:03:00+00:00",
        expected_deferral_count=0,
        expected_revision=archive.revision,
    )
    archive.defer_gap(
        gap.gap_id,
        "2026-07-19T00:04:00+00:00",
        expected_deferral_count=1,
        expected_revision=archive.revision,
    )
    descendant_revision = archive.revision
    assert (
        archive.record_outcome(
            node.idea_id,
            archive.get_idea(node.idea_id).outcome,
            claim_updates=(claim,),
            gap_updates=(gap,),
            expected_revision=completed_revision,
        )
        == descendant_revision
    )
    strategy.record_finalized_idea_outcome(node)
    assert archive.revision == descendant_revision
    snapshot = CampaignEvidenceBuilder(evidence_settings()).build(
        campaign_id=archive.state.campaign_id,
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(
            ExperimentInput(
                node_id=0,
                idea_id=node.idea_id,
                selection_batch_id=node.selection_batch_id,
                parent_node_id=None,
                proposal=node.solution,
                score=0.8,
                evaluation_valid=True,
                had_error=False,
                recoverable_error=False,
                build_fidelity="full",
                attempts=(
                    EvaluationAttemptInput(
                        evaluator_id="evaluator-v1",
                        fidelity="full",
                        fraction=1.0,
                        seed=7,
                        score=0.8,
                        duration_seconds=4.0,
                    ),
                ),
                feedback="Observed canonical utility.",
                technical_difficulty=None,
                created_at=node.started_at,
            ),
        ),
        archive_state=archive.state,
        capacity=capacity(),
        generated_at="2026-07-19T00:05:00+00:00",
    )

    assert archive.state.claims[0].affected_experiment_node_ids == (0,)
    assert gap.state == GapState.OPEN
    assert EvidenceSignal.SUPPORTED_LEVER in snapshot.signals
    assert EvidenceSignal.GAP_DEBT in snapshot.signals
    decision = choose_policy(snapshot, capacity())
    assert decision.mode == IdeationMode.EXPLORE
    assert "gap_debt" in {reason.code for reason in decision.reasons}
