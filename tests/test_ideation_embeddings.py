"""Tests for the isolated OpenAI embedding boundary and local similarity."""

import hashlib
from types import SimpleNamespace

import pytest

from kapso.execution.search_strategies.generic.ideation.embeddings import (
    EmbeddingSettings,
    OpenAIEmbeddingProvider,
    canonical_idea_embedding_text,
    cosine_similarity,
    embedding_can_be_reused,
)
from kapso.execution.search_strategies.generic.ideation.types import EmbeddingRecord
from test_ideation_domain import generated_idea


class FakeEmbeddings:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.embeddings = FakeEmbeddings(response)


def settings(**changes) -> EmbeddingSettings:
    values = {
        "enabled": True,
        "model": "text-embedding-test",
        "dimensions": 3,
        "timeout_seconds": 4,
        "max_retries": 0,
    }
    values.update(changes)
    return EmbeddingSettings(**values)


def response(items=None, tokens=17):
    return SimpleNamespace(
        data=(
            items
            if items is not None
            else [SimpleNamespace(index=0, embedding=[1.0, 2.0, 3.0])]
        ),
        usage=SimpleNamespace(prompt_tokens=tokens),
    )


def test_provider_sends_complete_inputs_and_records_exact_metadata():
    client = FakeClient(
        response(
            [
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(settings(), client=client)
    texts = ("first full input", "second full input\nwith details")

    batch = provider.embed(texts)

    assert client.embeddings.calls == [
        {
            "model": "text-embedding-test",
            "dimensions": 3,
            "input": list(texts),
        }
    ]
    assert tuple(record.vector for record in batch.records) == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    assert (
        batch.records[1].input_hash
        == hashlib.sha256(texts[1].encode("utf-8")).hexdigest()
    )
    assert batch.telemetry.provider == "openai"
    assert batch.telemetry.call_count == 1
    assert batch.telemetry.input_tokens == 17


@pytest.mark.parametrize(
    "items",
    [
        [],
        [SimpleNamespace(index=2, embedding=[1.0, 2.0, 3.0])],
        [SimpleNamespace(index=0, embedding=[1.0, 2.0])],
    ],
)
def test_provider_rejects_count_index_and_dimension_mismatches(items):
    provider = OpenAIEmbeddingProvider(settings(), client=FakeClient(response(items)))
    with pytest.raises(ValueError):
        provider.embed(("full input",))


def test_provider_errors_propagate_without_degradation():
    class FailedEmbeddings:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    provider = OpenAIEmbeddingProvider(
        settings(),
        client=SimpleNamespace(embeddings=FailedEmbeddings()),
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        provider.embed(("full input",))


def test_embedding_cache_requires_exact_provider_model_dimensions_and_hash():
    text = "complete stable text"
    record = EmbeddingRecord(
        provider="openai",
        model="text-embedding-test",
        dimensions=3,
        input_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        vector=(1.0, 0.0, 0.0),
    )
    assert embedding_can_be_reused(record, text, settings())
    assert not embedding_can_be_reused(record, text + " changed", settings())
    assert not embedding_can_be_reused(
        record,
        text,
        settings(model="different-model"),
    )
    assert not embedding_can_be_reused(record, text, settings(enabled=False))


def test_cosine_rejects_incompatible_and_zero_vectors():
    left = EmbeddingRecord(
        provider="openai",
        model="model-a",
        dimensions=2,
        input_hash="1" * 64,
        vector=(1.0, 0.0),
    )
    right = EmbeddingRecord(
        provider="openai",
        model="model-a",
        dimensions=2,
        input_hash="2" * 64,
        vector=(0.0, 1.0),
    )
    assert cosine_similarity(left, right) == 0.0
    with pytest.raises(ValueError, match="compatible"):
        cosine_similarity(
            left,
            EmbeddingRecord(
                provider="openai",
                model="model-b",
                dimensions=2,
                input_hash="3" * 64,
                vector=(0.0, 1.0),
            ),
        )
    with pytest.raises(ValueError, match="non-zero"):
        cosine_similarity(
            left,
            EmbeddingRecord(
                provider="openai",
                model="model-a",
                dimensions=2,
                input_hash="4" * 64,
                vector=(0.0, 0.0),
            ),
        )


def test_canonical_idea_embedding_text_contains_all_semantic_fields():
    idea = generated_idea()
    text = canonical_idea_embedding_text(idea)
    assert idea.proposal in text
    assert idea.directive_rationale in text
    assert idea.evaluation_method in text
    assert idea.descriptor.mechanism in text
