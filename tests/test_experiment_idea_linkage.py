"""Strict experiment-memory projection and write-order tests."""

import json
from types import SimpleNamespace

import pytest

from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.memories.experiment_memory.store import (
    EXPERIMENT_HISTORY_SCHEMA,
    ExperimentHistoryStore,
    format_experiments,
)
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation import new_identifier
from test_ideation_domain import NOW


def node(node_id, score, *, maximize=True):
    idea_id = new_identifier("idea")
    batch_id = new_identifier("batch")
    attempt = EvaluationAttempt(
        commit_sha=f"commit-{node_id}",
        evaluator_id="evaluator-v1",
        fidelity="full",
        fraction=1.0,
        seed=7,
        score=score,
        duration_seconds=2.0,
    )
    return SearchNode(
        node_id=node_id,
        idea_id=idea_id,
        selection_batch_id=batch_id,
        solution=f"candidate {node_id}",
        branch_name=f"candidate-{node_id}",
        feedback="measured feedback",
        score=score,
        started_at=NOW,
        build_fidelity="full",
        eval_fidelity="full",
        evaluation_attempts=[attempt],
        duration_seconds=3.0,
        cost_usd=0.2,
        metrics={"private_metric": 0.99},
        primary_metric="private_metric",
    )


def test_strict_projection_round_trips_complete_idea_and_fidelity_lineage(tmp_path):
    path = tmp_path / "history.json"
    store = ExperimentHistoryStore(
        str(path),
        objective_direction="maximize",
        require_idea_links=True,
    )

    record = store.add_experiment(node(0, 0.4))
    reloaded = ExperimentHistoryStore(
        str(path),
        objective_direction="maximize",
        require_idea_links=True,
    )

    assert reloaded.experiments == [record]
    assert record.idea_id.startswith("idea_")
    assert record.selection_batch_id.startswith("batch_")
    assert record.normalized_utility == 0.4
    assert record.validation_tier == "full"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == EXPERIMENT_HISTORY_SCHEMA
    assert document["records"][0]["evaluation_attempts"][0]["seed"] == 7


def test_incompatible_legacy_list_fails_loudly(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="document fields"):
        ExperimentHistoryStore(
            str(path),
            objective_direction="maximize",
            require_idea_links=True,
        )


def test_minimize_ranking_uses_normalized_utility_and_invalid_records_do_not_rank(
    tmp_path,
):
    store = ExperimentHistoryStore(
        str(tmp_path / "history.json"),
        objective_direction="minimize",
        require_idea_links=True,
    )
    first = node(0, 0.8)
    second = node(1, 0.2)
    invalid = node(2, 0.01)
    invalid.evaluation_valid = False
    invalid.evaluation_attempts = []
    store.add_experiment(first)
    store.add_experiment(second)
    store.add_experiment(invalid)

    assert [item.node_id for item in store.get_top_experiments()] == [1, 0]
    assert store.experiments[2].raw_score is None


def test_recovery_replaces_one_node_only_at_the_next_execution_revision(tmp_path):
    store = ExperimentHistoryStore(
        str(tmp_path / "history.json"),
        objective_direction="maximize",
        require_idea_links=True,
    )
    failed = node(0, 0.4)
    failed.score = None
    failed.evaluation_attempts = []
    failed.had_error = True
    failed.recoverable_error = True
    failed.evaluation_valid = False
    failed.error_message = "repairable implementation failure"
    store.add_experiment(failed)

    recovered = node(0, 0.7)
    recovered.idea_id = failed.idea_id
    recovered.selection_batch_id = failed.selection_batch_id
    recovered.solution = failed.solution
    recovered.execution_revision = 1
    store.add_experiment(recovered)

    assert store.get_experiment_count() == 1
    assert store.experiments[0].execution_revision == 1
    assert store.experiments[0].raw_score == 0.7


def test_agent_render_includes_links_but_not_external_metrics(tmp_path):
    store = ExperimentHistoryStore(
        str(tmp_path / "history.json"),
        objective_direction="maximize",
        require_idea_links=True,
    )
    record = store.add_experiment(node(0, 0.4))

    rendered = format_experiments((record,))

    assert record.idea_id in rendered
    assert record.selection_batch_id in rendered
    assert "private_metric" not in rendered
    assert "0.99" not in rendered


def test_orchestrator_persists_experiment_before_idea_outcome():
    events = []
    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.experiment_store = SimpleNamespace(
        add_experiment=lambda candidate: events.append(
            ("experiment", candidate.node_id)
        )
    )
    orchestrator.search_strategy = SimpleNamespace(
        record_finalized_idea_outcome=lambda candidate: events.append(
            ("outcome", candidate.node_id)
        )
    )

    orchestrator._persist_finalized_candidates(
        [SearchNode(node_id=0), SearchNode(node_id=1)]
    )

    assert events == [
        ("experiment", 0),
        ("outcome", 0),
        ("experiment", 1),
        ("outcome", 1),
    ]


def test_reexecuted_node_is_a_finalized_candidate():
    recovered = SearchNode(node_id=0, execution_revision=1)
    assert OrchestratorAgent._new_candidates(
        {0: 0},
        [recovered],
        recovered,
    ) == [recovered]
