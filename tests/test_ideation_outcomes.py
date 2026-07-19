"""Outcome projection tests for finalized linked experiments."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation import (
    EvaluationStatus,
    IdeaStatus,
    ImplementationStatus,
    ObjectiveDirection,
    build_idea_outcome,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from test_ideation_domain import BATCH_ID, generated_idea
from test_ideation_engine import (
    CountingParents,
    PacketRunner,
    build_engine,
    run_engine,
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
    parent = SearchNode(node_id=0, score=0.6)
    idea, child = linked(node_id=1)
    frozen_parent = replace(idea.resolved_parent, node_id=0)
    idea = replace(idea, resolved_parent=frozen_parent)
    child.score = 0.4

    outcome = build_idea_outcome(
        node=child,
        idea=idea,
        nodes_by_id={0: parent, 1: child},
        objective_direction=ObjectiveDirection.MINIMIZE,
    )

    assert outcome.normalized_delta == pytest.approx(0.2)


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
