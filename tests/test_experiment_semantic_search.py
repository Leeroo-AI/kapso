"""Hermetic tests for experiment-store semantic search (embedding-backed).

The provider boundary is the LLM backend's create_embedding; everything
below it is real store behavior: write-time embedding of the FULL
solution, JSON persistence of vectors, cosine ranking at query time, and
the documented recency degradation when no backend is configured.
"""

from types import SimpleNamespace

import pytest

from kapso.execution.memories.experiment_memory.store import (
    ExperimentHistoryStore,
    cosine_similarity,
)


class StubEmbedder:
    """Deterministic embedding backend: axis per known phrase."""

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
    return SimpleNamespace(
        node_id=node_id,
        solution=solution,
        score=score,
        feedback="fb",
        branch_name=f"b{node_id}",
        had_error=False,
        error_message="",
        technical_difficulties="",
        evaluation_valid=evaluation_valid,
    )


def make_store(tmp_path, llm):
    return ExperimentHistoryStore(
        json_path=str(tmp_path / "history.json"), llm=llm
    )


def test_add_embeds_full_solution_and_persists_vector(tmp_path):
    llm = StubEmbedder()
    store = make_store(tmp_path, llm)

    store.add_experiment(node(0, "teacher distillation with vllm"))

    assert llm.calls == ["teacher distillation with vllm"]  # full text, once
    reloaded = make_store(tmp_path, llm=None)
    assert reloaded.experiments[0].solution_embedding == [1.0, 0.0, 0.0]


def test_search_ranks_by_cosine_similarity(tmp_path):
    llm = StubEmbedder()
    store = make_store(tmp_path, llm)
    store.add_experiment(node(0, "teacher distillation with vllm"))
    store.add_experiment(node(1, "dpo preference pairs"))
    store.add_experiment(node(2, "decoding config sweep"))

    results = store.search_similar("distill from a strong teacher", k=2)

    assert [r.node_id for r in results] == [0, 1] or [
        r.node_id for r in results
    ] == [0, 2]
    assert results[0].node_id == 0  # the distillation record wins


def test_invalid_evaluation_records_stay_searchable(tmp_path):
    """New contract (supersedes the weaviate-era exclusion): every stored
    solution is discoverable via similarity — 'was this tried?' must
    surface invalid attempts too; renders mark them INVALID."""
    llm = StubEmbedder()
    store = make_store(tmp_path, llm)
    store.add_experiment(
        node(0, "teacher distillation with vllm", evaluation_valid=False)
    )

    results = store.search_similar("distill from a strong teacher", k=1)

    assert results[0].node_id == 0
    assert results[0].evaluation_valid is False


def test_no_backend_degrades_to_recent(tmp_path):
    store = make_store(tmp_path, llm=None)
    store.add_experiment(node(0, "teacher distillation with vllm"))
    store.add_experiment(node(1, "dpo preference pairs"))

    results = store.search_similar("anything", k=1)

    assert [r.node_id for r in results] == [1]  # recency, not similarity
    assert store.experiments[0].solution_embedding == []


def test_blank_solution_is_not_embedded(tmp_path):
    llm = StubEmbedder()
    store = make_store(tmp_path, llm)

    store.add_experiment(node(0, "   "))

    assert llm.calls == []
    assert store.experiments[0].solution_embedding == []


def test_corrupt_history_file_raises(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not json")

    with pytest.raises(ValueError):
        ExperimentHistoryStore(json_path=str(path))


def test_cosine_similarity_contract():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])
