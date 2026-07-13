"""Hermetic tests for versioned attempts and tier-order selection (M6a).

Pins the comparability contract: scores never cross evaluator_id
boundaries, replications aggregate, projections keep the dumb selectors
correct, and the committed slot follows evidence tiers — never raw scores.
"""

from types import SimpleNamespace

import pytest

from kapso.execution.fidelity import (
    ComparabilityClass,
    EvaluationAttempt,
    TIER_FULL,
    TIER_PROBE,
    TIER_VALIDATED,
    evidence_tier,
    project_score,
    select_committed_candidate,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch


def attempt(
    *,
    evaluator_id="eval-v1",
    fidelity="fast",
    fraction=0.15,
    seed=1337,
    score=0.5,
    commit_sha="sha-1",
):
    return EvaluationAttempt(
        commit_sha=commit_sha,
        evaluator_id=evaluator_id,
        fidelity=fidelity,
        fraction=fraction,
        seed=seed,
        score=score,
    )


FAST_V1 = ComparabilityClass(
    evaluator_id="eval-v1", fidelity="fast", fraction=0.15, seed=1337
)


# =========================================================================
# Attempt and class contracts
# =========================================================================

def test_attempt_validation_and_round_trip():
    with pytest.raises(ValueError, match="fidelity"):
        attempt(fidelity="medium")
    with pytest.raises(ValueError, match="fraction"):
        attempt(fraction=0.0)
    with pytest.raises(ValueError, match="score"):
        attempt(score=float("nan"))
    with pytest.raises(ValueError, match="commit_sha"):
        attempt(commit_sha="")

    original = attempt(score=0.42)
    restored = EvaluationAttempt.from_dict(original.to_dict())
    assert restored == original
    assert restored.comparability_class == FAST_V1


def test_projection_is_class_isolated_and_aggregates_replications():
    node = SearchNode(node_id=0)
    node.evaluation_attempts = [
        attempt(score=0.4),
        attempt(score=0.6),  # replication in the same class -> mean
        attempt(evaluator_id="eval-v2", score=0.9),  # different ruler
        attempt(fidelity="full", fraction=1.0, score=0.99),  # different class
        attempt(seed=7, score=0.99),  # different item set
    ]

    assert project_score(node, FAST_V1) == pytest.approx(0.5)
    assert (
        project_score(
            node,
            ComparabilityClass(
                evaluator_id="eval-v3",
                fidelity="fast",
                fraction=0.15,
                seed=1337,
            ),
        )
        is None
    )


# =========================================================================
# Tier order
# =========================================================================

def test_evidence_tiers():
    probe_only = SearchNode(node_id=0, build_fidelity="fast")
    probe_only.evaluation_attempts = [attempt(score=0.9)]
    assert evidence_tier(probe_only, "eval-v1") == TIER_PROBE

    validated = SearchNode(node_id=1, build_fidelity="fast")
    validated.evaluation_attempts = [
        attempt(score=0.9),
        attempt(fidelity="full", fraction=1.0, score=0.4),
    ]
    assert evidence_tier(validated, "eval-v1") == TIER_VALIDATED

    full = SearchNode(node_id=2, build_fidelity="full")
    full.evaluation_attempts = [
        attempt(fidelity="full", fraction=1.0, score=0.43)
    ]
    assert evidence_tier(full, "eval-v1") == TIER_FULL
    # Under a different evaluator version, the same evidence is stale.
    assert evidence_tier(full, "eval-v2") == TIER_PROBE


def test_committed_selection_follows_tiers_not_raw_scores():
    flashy_probe = SearchNode(node_id=0, build_fidelity="fast")
    flashy_probe.evaluation_attempts = [attempt(score=0.99)]

    validated = SearchNode(node_id=1, build_fidelity="fast")
    validated.evaluation_attempts = [
        attempt(score=0.47),
        attempt(fidelity="full", fraction=1.0, score=0.45),
    ]

    full_lower_score = SearchNode(node_id=2, build_fidelity="full")
    full_lower_score.evaluation_attempts = [
        attempt(fidelity="full", fraction=1.0, score=0.43)
    ]

    stale_full = SearchNode(node_id=3, build_fidelity="full")
    stale_full.evaluation_attempts = [
        attempt(
            evaluator_id="eval-v0", fidelity="full", fraction=1.0, score=0.99
        )
    ]

    invalid = SearchNode(
        node_id=4, build_fidelity="full", evaluation_valid=False
    )
    invalid.evaluation_attempts = [
        attempt(fidelity="full", fraction=1.0, score=0.99)
    ]

    nodes = [flashy_probe, validated, full_lower_score, stale_full, invalid]

    # A FULL-tier node beats a VALIDATED node with a higher full score, and
    # both beat the flashy probe; stale and invalid evidence never competes.
    winner = select_committed_candidate(nodes, evaluator_id="eval-v1")
    assert winner is full_lower_score

    without_full = [flashy_probe, validated, stale_full]
    assert (
        select_committed_candidate(without_full, evaluator_id="eval-v1")
        is validated
    )

    assert (
        select_committed_candidate([flashy_probe], evaluator_id="eval-v1")
        is None
    )

    # Direction awareness within a tier.
    low = SearchNode(node_id=5, build_fidelity="full")
    low.evaluation_attempts = [
        attempt(fidelity="full", fraction=1.0, score=0.1)
    ]
    assert (
        select_committed_candidate(
            [full_lower_score, low], evaluator_id="eval-v1", maximize=False
        )
        is low
    )


# =========================================================================
# SearchNode serialization with attempts
# =========================================================================

def test_node_round_trips_fidelity_fields_and_attempts():
    node = SearchNode(node_id=0, build_fidelity="fast", eval_fidelity="fast")
    node.promoted_from = None
    node.evaluation_attempts = [attempt(score=0.4)]

    restored = SearchNode.from_dict(node.to_dict())

    assert restored.build_fidelity == "fast"
    assert restored.eval_fidelity == "fast"
    assert restored.evaluation_attempts == node.evaluation_attempts

    with pytest.raises(ValueError, match="build_fidelity"):
        SearchNode.from_dict({"node_id": 0, "build_fidelity": "medium"})
    with pytest.raises(ValueError, match="promoted_from"):
        SearchNode.from_dict({"node_id": 0, "promoted_from": -1})
    with pytest.raises(ValueError, match="evaluation_attempts"):
        SearchNode.from_dict({"node_id": 0, "evaluation_attempts": "nope"})


# =========================================================================
# GenericSearch integration
# =========================================================================

def make_strategy():
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.registered_evaluator_id = "eval-v1"
    strategy.registered_subsample_seed = 1337
    strategy.workspace = SimpleNamespace(
        repo=SimpleNamespace(
            commit=lambda branch: SimpleNamespace(hexsha=f"sha-{branch}")
        )
    )
    return strategy


def test_strategy_records_attempts_only_for_trustworthy_scores():
    strategy = make_strategy()

    node = SearchNode(node_id=0, branch_name="generic_exp_0", score=0.7)
    node.phase_telemetry = {"implementation": {"duration_seconds": 12.0}}
    strategy._record_evaluation_attempt(node)
    assert len(node.evaluation_attempts) == 1
    recorded = node.evaluation_attempts[0]
    assert recorded.commit_sha == "sha-generic_exp_0"
    assert recorded.evaluator_id == "eval-v1"
    assert recorded.fidelity == "full"
    assert recorded.duration_seconds == 12.0

    unscored = SearchNode(node_id=1, branch_name="generic_exp_1", score=None)
    strategy._record_evaluation_attempt(unscored)
    assert unscored.evaluation_attempts == []

    invalid = SearchNode(
        node_id=2,
        branch_name="generic_exp_2",
        score=0.9,
        evaluation_valid=False,
    )
    strategy._record_evaluation_attempt(invalid)
    assert invalid.evaluation_attempts == []

    strategy.registered_evaluator_id = ""
    unregistered = SearchNode(
        node_id=3, branch_name="generic_exp_3", score=0.9
    )
    strategy._record_evaluation_attempt(unregistered)
    assert unregistered.evaluation_attempts == []


def test_projection_refresh_moves_the_frontier_to_the_new_ruler():
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)

    old_leader = SearchNode(node_id=0, branch_name="a", score=0.9)
    old_leader.evaluation_attempts = [
        attempt(evaluator_id="eval-v1", score=0.9)
    ]
    anchored = SearchNode(node_id=1, branch_name="b", score=0.5)
    anchored.evaluation_attempts = [
        attempt(evaluator_id="eval-v1", score=0.5),
        attempt(evaluator_id="eval-v2", score=0.6),
    ]
    strategy.node_history = [old_leader, anchored]

    strategy.refresh_score_projections(
        ComparabilityClass(
            evaluator_id="eval-v2", fidelity="fast", fraction=0.15, seed=1337
        )
    )

    # Never measured under v2 -> None, and None never wins.
    assert old_leader.score is None
    assert anchored.score == 0.6
    assert strategy.get_best_experiment() is anchored
