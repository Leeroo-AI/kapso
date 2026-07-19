"""Behavioral tests for strict embedding-backed experiment retrieval."""

from datetime import datetime, timezone

import pytest

from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.memories.experiment_memory.store import (
    ExperimentHistoryStore,
    cosine_similarity,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation import new_identifier


class StubEmbedder:
    """Deterministic embedding boundary with production-shaped behavior."""

    VOCAB = {
        "teacher distillation with vllm": [1.0, 0.0, 0.0],
        "dpo preference pairs": [0.0, 1.0, 0.0],
        "decoding config sweep": [0.0, 0.0, 1.0],
        "distill from a strong teacher": [0.9, 0.1, 0.0],
    }

    def __init__(self):
        self.calls = []

    def create_embedding(self, text, model=None):
        self.calls.append(text)
        return list(self.VOCAB[text])


def node(node_id, solution, score=1.0, evaluation_valid=True):
    attempts = []
    if evaluation_valid:
        attempts = [
            EvaluationAttempt(
                commit_sha=f"commit-{node_id}",
                evaluator_id="evaluator-v1",
                fidelity="full",
                fraction=1.0,
                seed=7,
                score=score,
                duration_seconds=1.0,
            )
        ]
    return SearchNode(
        node_id=node_id,
        idea_id=new_identifier("idea"),
        selection_batch_id=new_identifier("batch"),
        solution=solution,
        score=score if evaluation_valid else None,
        feedback="measured feedback",
        branch_name=f"candidate-{node_id}",
        started_at=datetime.now(timezone.utc).isoformat(),
        evaluation_valid=evaluation_valid,
        evaluation_attempts=attempts,
    )


def make_store(tmp_path, llm):
    return ExperimentHistoryStore(
        json_path=str(tmp_path / "history.json"),
        objective_direction="maximize",
        require_idea_links=True,
        llm=llm,
    )


def test_add_embeds_full_solution_once_and_persists_vector(tmp_path):
    llm = StubEmbedder()
    store = make_store(tmp_path, llm)

    store.add_experiment(node(0, "teacher distillation with vllm"))

    assert llm.calls == ["teacher distillation with vllm"]
    reloaded = make_store(tmp_path, llm=None)
    assert reloaded.experiments[0].solution_embedding == (1.0, 0.0, 0.0)


def test_idempotent_reconciliation_reuses_the_persisted_embedding(tmp_path):
    llm = StubEmbedder()
    store = make_store(tmp_path, llm)
    candidate = node(0, "teacher distillation with vllm")

    first = store.add_experiment(candidate)
    second = store.add_experiment(candidate)

    assert first == second
    assert llm.calls == ["teacher distillation with vllm"]


def test_search_ranks_all_embedded_experiments_by_cosine_similarity(tmp_path):
    llm = StubEmbedder()
    store = make_store(tmp_path, llm)
    store.add_experiment(node(0, "teacher distillation with vllm"))
    store.add_experiment(node(1, "dpo preference pairs"))
    store.add_experiment(node(2, "decoding config sweep"))

    results = store.search_similar("distill from a strong teacher", k=2)

    assert results[0].node_id == 0
    assert {result.node_id for result in results[1:]} <= {1, 2}


def test_invalid_evaluation_records_remain_discoverable_but_not_top_ranked(tmp_path):
    llm = StubEmbedder()
    store = make_store(tmp_path, llm)
    store.add_experiment(
        node(
            0,
            "teacher distillation with vllm",
            evaluation_valid=False,
        )
    )
    store.add_experiment(node(1, "dpo preference pairs", score=0.2))

    assert store.search_similar("distill from a strong teacher", k=1)[0].node_id == 0
    assert [record.node_id for record in store.get_top_experiments()] == [1]


def test_store_without_embedding_backend_is_explicitly_recency_only(tmp_path):
    store = make_store(tmp_path, llm=None)
    store.add_experiment(node(0, "teacher distillation with vllm"))
    store.add_experiment(node(1, "dpo preference pairs"))

    results = store.search_similar("distill from a strong teacher", k=1)

    assert [record.node_id for record in results] == [1]
    assert store.experiments[0].solution_embedding == ()


def test_corrupt_history_file_raises(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError):
        ExperimentHistoryStore(
            json_path=str(path),
            objective_direction="maximize",
            require_idea_links=True,
        )


def test_cosine_similarity_contract():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="dimensions differ"):
        cosine_similarity([1.0], [1.0, 2.0])
