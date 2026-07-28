"""Executed experiments use the shared embedding-provider contract."""

from __future__ import annotations

import pytest

from kapso.core.embedding_contracts import (
    EmbeddingBatch,
    EmbeddingRecord,
    EmbeddingSettings,
    EmbeddingTelemetry,
    complete_input_hash,
)
from kapso.cross_run.canonical import canonical_utc_now
from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.memories.experiment_memory.record import cosine_similarity
from kapso.execution.memories.experiment_memory.store import ExperimentHistoryStore
from kapso.execution.search_strategies.generic.ideation.types import new_identifier
from kapso.execution.search_strategies.node import SearchNode


class StubEmbeddingProvider:
    VOCAB = {
        "teacher distillation with vllm": (1.0, 0.0, 0.0),
        "dpo preference pairs": (0.0, 1.0, 0.0),
    }

    def __init__(self) -> None:
        self.settings = EmbeddingSettings(
            enabled=True,
            provider="openai",
            model="test-embedding-model",
            dimensions=3,
            batch_size=2,
            timeout_seconds=1,
            max_retries=0,
            canonicalizer_version="test.embedding.v1",
        )
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts) -> EmbeddingBatch:
        values = tuple(texts)
        self.calls.append(values)
        return EmbeddingBatch(
            records=tuple(
                EmbeddingRecord(
                    provider=self.settings.provider,
                    model=self.settings.model,
                    dimensions=self.settings.dimensions,
                    canonicalizer_version=self.settings.canonicalizer_version,
                    input_hash=complete_input_hash(text),
                    vector=self.VOCAB[text],
                )
                for text in values
            ),
            telemetry=EmbeddingTelemetry(
                provider=self.settings.provider,
                model=self.settings.model,
                call_count=1,
                input_tokens=0,
                duration_seconds=0,
                cost_usd=0,
            ),
        )


def node(node_id: int, solution: str) -> SearchNode:
    return SearchNode(
        node_id=node_id,
        idea_id=new_identifier("idea"),
        selection_batch_id=new_identifier("batch"),
        solution=solution,
        score=1.0,
        feedback="measured feedback",
        branch_name=f"candidate-{node_id}",
        started_at=canonical_utc_now(),
        evaluation_valid=True,
        evaluation_attempts=[
            EvaluationAttempt(
                commit_sha=f"commit-{node_id}",
                evaluator_id="evaluator-v1",
                fidelity="full",
                fraction=1.0,
                seed=7,
                score=1.0,
                duration_seconds=1.0,
            )
        ],
    )


def make_store(tmp_path, embedding_provider) -> ExperimentHistoryStore:
    return ExperimentHistoryStore(
        json_path=str(tmp_path / "history.json"),
        objective_direction="maximize",
        require_idea_links=True,
        embedding_provider=embedding_provider,
        run_id="run_test",
        campaign_id="campaign_test",
        journal_path=str(tmp_path / "execution_events.jsonl"),
    )


def test_add_embeds_full_solution_once_and_persists_vector(tmp_path) -> None:
    provider = StubEmbeddingProvider()
    store = make_store(tmp_path, provider)

    store.add_experiment(node(0, "teacher distillation with vllm"))

    assert provider.calls == [("teacher distillation with vllm",)]
    reloaded = make_store(tmp_path, None)
    assert reloaded.experiments[0].solution_embedding == (1.0, 0.0, 0.0)


def test_idempotent_reconciliation_reuses_persisted_embedding(tmp_path) -> None:
    provider = StubEmbeddingProvider()
    store = make_store(tmp_path, provider)
    candidate = node(0, "teacher distillation with vllm")

    first = store.add_experiment(candidate)
    second = store.add_experiment(candidate)

    assert first == second
    assert provider.calls == [("teacher distillation with vllm",)]


def test_direct_store_without_provider_records_no_embedding(tmp_path) -> None:
    store = make_store(tmp_path, None)
    store.add_experiment(node(0, "teacher distillation with vllm"))
    assert store.experiments[0].solution_embedding == ()


def test_corrupt_history_file_raises(tmp_path) -> None:
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError):
        ExperimentHistoryStore(
            json_path=str(path),
            objective_direction="maximize",
            require_idea_links=True,
        )


def test_cosine_similarity_contract() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="dimensions differ"):
        cosine_similarity([1.0], [1.0, 2.0])
