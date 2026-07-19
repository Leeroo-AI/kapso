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
    full.evaluation_attempts = [attempt(fidelity="full", fraction=1.0, score=0.43)]
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
        attempt(evaluator_id="eval-v0", fidelity="full", fraction=1.0, score=0.99)
    ]

    invalid = SearchNode(node_id=4, build_fidelity="full", evaluation_valid=False)
    invalid.evaluation_attempts = [attempt(fidelity="full", fraction=1.0, score=0.99)]

    nodes = [flashy_probe, validated, full_lower_score, stale_full, invalid]

    # A FULL-tier node beats a VALIDATED node with a higher full score, and
    # both beat the flashy probe; stale and invalid evidence never competes.
    winner = select_committed_candidate(nodes, evaluator_id="eval-v1")
    assert winner is full_lower_score

    without_full = [flashy_probe, validated, stale_full]
    assert select_committed_candidate(without_full, evaluator_id="eval-v1") is validated

    assert select_committed_candidate([flashy_probe], evaluator_id="eval-v1") is None

    # Direction awareness within a tier.
    low = SearchNode(node_id=5, build_fidelity="full")
    low.evaluation_attempts = [attempt(fidelity="full", fraction=1.0, score=0.1)]
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
    strategy.registered_evaluation_command = "python kapso_evaluation/kapso_eval.py"
    strategy.registered_subsample_seed = 1337
    strategy.fidelity_decision = None
    strategy.workspace = SimpleNamespace(
        repo=SimpleNamespace(
            commit=lambda branch: SimpleNamespace(hexsha=f"sha-{branch}")
        )
    )
    return strategy


def test_strategy_records_attempts_only_for_trustworthy_scores():
    strategy = make_strategy()

    node = SearchNode(node_id=0, branch_name="generic_exp_0", score=0.99)
    node.evaluation_output = (
        'KAPSO_EVAL_MANIFEST {"fidelity":"full","fraction":1.0,'
        '"seed":1337,"items":40,"total_items":40,"score":0.7}'
    )
    node.phase_telemetry = {"implementation": {"duration_seconds": 12.0}}
    strategy._record_evaluation_attempt(node)
    assert len(node.evaluation_attempts) == 1
    recorded = node.evaluation_attempts[0]
    assert recorded.commit_sha == "sha-generic_exp_0"
    assert recorded.evaluator_id == "eval-v1"
    assert recorded.fidelity == "full"
    assert recorded.score == node.score == 0.7
    assert recorded.duration_seconds == 12.0

    unscored = SearchNode(node_id=1, branch_name="generic_exp_1", score=None)
    strategy._record_evaluation_attempt(unscored)
    assert unscored.evaluation_attempts == []

    self_report_only = SearchNode(
        node_id=4,
        branch_name="generic_exp_4",
        score=0.95,
        evaluation_output="implementation XML score without a manifest",
    )
    strategy._record_evaluation_attempt(self_report_only)
    assert self_report_only.score is None
    assert self_report_only.evaluation_attempts == []

    invalid = SearchNode(
        node_id=2,
        branch_name="generic_exp_2",
        score=0.9,
        evaluation_valid=False,
    )
    strategy._record_evaluation_attempt(invalid)
    assert invalid.evaluation_attempts == []

    strategy.registered_evaluator_id = ""
    unregistered = SearchNode(node_id=3, branch_name="generic_exp_3", score=0.9)
    strategy._record_evaluation_attempt(unregistered)
    assert unregistered.evaluation_attempts == []


def test_projection_refresh_moves_the_frontier_to_the_new_ruler():
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)

    old_leader = SearchNode(node_id=0, branch_name="a", score=0.9)
    old_leader.evaluation_attempts = [attempt(evaluator_id="eval-v1", score=0.9)]
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


def test_deliverable_prefers_committed_tier_over_fast_leader():
    """The live campaign checked out the unvalidated fast leader (probe
    0.87) over the TIER_FULL candidate (full 0.9). The deliverable slot
    follows the tier walk; parent selection alone stays on projections.
    """
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy.registered_evaluator_id = "eval-v1"

    full_tier = SearchNode(
        node_id=0,
        branch_name="generic_exp_0",
        score=0.49,
        build_fidelity="full",
    )
    full_tier.evaluation_attempts = [attempt(fidelity="full", fraction=1.0, score=0.9)]
    fast_leader = SearchNode(
        node_id=1,
        branch_name="generic_exp_1",
        score=0.87,
        build_fidelity="fast",
    )
    fast_leader.evaluation_attempts = [attempt(score=0.87)]
    strategy.node_history = [full_tier, fast_leader]

    assert strategy.get_deliverable_experiment() is full_tier
    # Reported score is the authoritative full-class measurement, not the
    # canonical (fast) projection stored on node.score.
    strategy.registered_subsample_seed = 1337
    assert strategy.get_deliverable_score() == 0.9

    # Without registered evidence the score leader stands.
    strategy.registered_evaluator_id = ""
    assert strategy.get_deliverable_experiment() is fast_leader
    assert strategy.get_deliverable_score() == 0.87


def test_manifest_line_is_the_score_of_record(tmp_path):
    """Two live nodes lost real measurements when the feedback call died:
    the manifest line the wrapper contractually prints is machine-readable,
    so the LLM judge is never the parser of record. Class-mismatched lines
    (custom fractions, wrong fidelity) are not the canonical measurement.
    """
    from kapso.execution.fidelity import FidelityDecision

    strategy = make_strategy()
    strategy.registered_evaluation_command = (
        "python kapso_evaluation/kapso_eval.py --fidelity fast "
        "--fraction 0.2 --seed 1337"
    )
    strategy.fidelity_decision = FidelityDecision(
        profile="probe",
        build_fidelity="fast",
        eval_fidelity="fast",
        eval_fraction=0.2,
    )

    def manifest_line(fraction, score):
        return (
            'KAPSO_EVAL_MANIFEST {"fidelity": "fast", '
            f'"fraction": {fraction}, "seed": 1337, "items": 40, '
            f'"total_items": 200, "score": {score}}}'
        )

    node = SearchNode(node_id=0, branch_name="generic_exp_0")
    # Several runs in-session: the LAST granted-class line wins.
    node.evaluation_output = "\n".join(
        [
            "exploring...",
            manifest_line(0.2, 0.61),
            "tuning...",
            manifest_line(0.2, 0.87),
        ]
    )
    assert strategy._manifest_score_of_record(node) == 0.87

    # A custom-fraction run is a different comparability class.
    node.evaluation_output = manifest_line(0.5, 0.91)
    assert strategy._manifest_score_of_record(node) is None

    # No manifest at all: nothing mechanical to say.
    node.evaluation_output = "the evaluation crashed before printing"
    assert strategy._manifest_score_of_record(node) is None

    # Unregistered mode: the manifest contract does not exist.
    strategy.registered_evaluation_command = ""
    node.evaluation_output = manifest_line(0.2, 0.9)
    assert strategy._manifest_score_of_record(node) is None


def test_malformed_manifest_line_raises(tmp_path):
    import pytest as pytest_module

    strategy = make_strategy()
    strategy.registered_evaluation_command = "python kapso_eval.py"
    strategy.fidelity_decision = None

    node = SearchNode(node_id=0, branch_name="generic_exp_0")
    node.evaluation_output = "KAPSO_EVAL_MANIFEST {not valid json"
    with pytest_module.raises(Exception):
        strategy._manifest_score_of_record(node)


# =========================================================================
# Best-node selection with unscored nodes (Arm-B regression, finding 13)
# =========================================================================


def test_minimize_unscored_valid_node_never_ranks_best():
    """A session that dies before its evaluation completes leaves a node
    with score=None and evaluation_valid=True (the default). On minimize
    metrics `None or 0` keyed as 0 and out-ranked every real negative key,
    so the unscored node became "best" for parent selection and history
    ranking. Both strategies must keep unscored nodes out of best/top."""
    from kapso.execution.search_strategies.benchmark_tree_search import (
        BenchmarkTreeSearch,
    )

    for strategy_cls in (GenericSearch, BenchmarkTreeSearch):
        strategy = strategy_cls.__new__(strategy_cls)
        strategy.problem_handler = SimpleNamespace(maximize_scoring=False)
        champion = SearchNode(node_id=5, branch_name="exp_5", score=2.6433)
        worse = SearchNode(node_id=7, branch_name="exp_7", score=2.6440)
        unscored = SearchNode(node_id=8, branch_name="exp_8", score=None)
        strategy.node_history = [champion, worse, unscored]

        assert strategy.get_best_experiment() is champion, strategy_cls.__name__
        history = strategy.get_experiment_history(best_last=True)
        assert history[-1] is champion, strategy_cls.__name__
        assert history[0] is unscored, strategy_cls.__name__


def test_all_unscored_history_yields_no_best_experiment():
    """With zero scored nodes there is no best experiment."""
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.problem_handler = SimpleNamespace(maximize_scoring=False)
    strategy.node_history = [
        SearchNode(node_id=0, branch_name="exp_0", score=None),
        SearchNode(node_id=1, branch_name="exp_1", score=None),
    ]

    assert strategy.get_best_experiment() is None
