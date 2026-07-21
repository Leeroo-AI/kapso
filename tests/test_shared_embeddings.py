"""Behavioral tests for the shared OpenAI embedding boundary."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from kapso.core.embeddings import (
    EmbeddingRecord,
    EmbeddingSettings,
    OpenAIEmbeddingProvider,
    complete_input_hash,
    cosine_similarity,
    embedding_can_be_reused,
)
from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import EmbeddingSidecar


class FakeEmbeddings:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, *responses):
        self.embeddings = FakeEmbeddings(responses)


def settings(**changes) -> EmbeddingSettings:
    values = {
        "enabled": True,
        "provider": "openai",
        "model": "text-embedding-test",
        "dimensions": 3,
        "batch_size": 2,
        "timeout_seconds": 4,
        "max_retries": 0,
        "canonicalizer_version": "kapso.test_embedding.v1",
    }
    values.update(changes)
    return EmbeddingSettings(**values)


def response(items, tokens=17):
    return SimpleNamespace(
        data=items,
        usage=SimpleNamespace(prompt_tokens=tokens),
    )


def test_provider_batches_complete_inputs_and_restores_response_order():
    client = FakeClient(
        response(
            [
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            ],
            tokens=11,
        ),
        response(
            [SimpleNamespace(index=0, embedding=[0.0, 0.0, 1.0])],
            tokens=7,
        ),
    )
    provider = OpenAIEmbeddingProvider(settings(), client=client)
    texts = (
        "first full input",
        "second full input\nwith details",
        "third complete input " + "x" * 10_000,
    )

    batch = provider.embed(texts)

    assert client.embeddings.calls == [
        {
            "model": "text-embedding-test",
            "dimensions": 3,
            "encoding_format": "float",
            "input": list(texts[:2]),
        },
        {
            "model": "text-embedding-test",
            "dimensions": 3,
            "encoding_format": "float",
            "input": [texts[2]],
        },
    ]
    assert tuple(record.vector for record in batch.records) == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert batch.records[2].input_hash == complete_input_hash(texts[2])
    assert batch.telemetry.call_count == 2
    assert batch.telemetry.input_tokens == 18
    assert batch.telemetry.duration_seconds >= 0
    assert batch.telemetry.cost_usd is None


@pytest.mark.parametrize(
    "items",
    [
        [],
        [SimpleNamespace(index=2, embedding=[1.0, 2.0, 3.0])],
        [SimpleNamespace(index=True, embedding=[1.0, 2.0, 3.0])],
        [SimpleNamespace(index=0, embedding=[1.0, 2.0])],
        [SimpleNamespace(index=0, embedding=[1.0, float("nan"), 3.0])],
    ],
)
def test_provider_rejects_malformed_count_index_and_vectors(items):
    provider = OpenAIEmbeddingProvider(
        settings(),
        client=FakeClient(response(items)),
    )
    with pytest.raises(ValueError):
        provider.embed(("full input",))


@pytest.mark.parametrize("tokens", [-1, True, 1.5, None])
def test_provider_rejects_malformed_usage(tokens):
    provider = OpenAIEmbeddingProvider(
        settings(),
        client=FakeClient(
            response(
                [SimpleNamespace(index=0, embedding=[1.0, 2.0, 3.0])],
                tokens=tokens,
            )
        ),
    )
    with pytest.raises(ValueError, match="prompt tokens"):
        provider.embed(("full input",))


def test_provider_errors_and_disabled_mode_fail_loud():
    class FailedEmbeddings:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    provider = OpenAIEmbeddingProvider(
        settings(),
        client=SimpleNamespace(embeddings=FailedEmbeddings()),
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        provider.embed(("full input",))
    with pytest.raises(ValueError, match="disabled"):
        OpenAIEmbeddingProvider(settings(enabled=False))


def test_embedding_space_identity_is_deterministic_and_sidecar_compatible():
    base = settings().embedding_space_id
    assert base == settings().embedding_space_id
    assert base != settings(model="different").embedding_space_id
    assert base != settings(dimensions=4).embedding_space_id
    assert (
        base
        != settings(canonicalizer_version="kapso.test_embedding.v2").embedding_space_id
    )
    assert base.value.startswith("embedding-space:sha256:")
    assert base.value == content_id("embedding-space", asdict(base))
    sidecar = EmbeddingSidecar(
        embedding_space_id=base.value,
        asset_ref="vectors/test.f32",
        checksum="sha256:" + "a" * 64,
    )
    assert sidecar.embedding_space_id == base.value


def test_embedding_reuse_requires_exact_space_and_complete_input_hash():
    text = "complete stable text"
    configured = settings()
    record = EmbeddingRecord(
        provider=configured.provider,
        model=configured.model,
        dimensions=configured.dimensions,
        canonicalizer_version=configured.canonicalizer_version,
        input_hash=complete_input_hash(text),
        vector=(1.0, 0.0, 0.0),
    )
    assert embedding_can_be_reused(record, text, configured)
    assert not embedding_can_be_reused(record, text + " changed", configured)
    assert not embedding_can_be_reused(
        record,
        text,
        replace(configured, canonicalizer_version="kapso.test_embedding.v2"),
    )
    assert not embedding_can_be_reused(
        record,
        text,
        replace(configured, enabled=False),
    )


def test_cosine_rejects_cross_space_and_zero_vectors():
    left = EmbeddingRecord(
        provider="openai",
        model="model-a",
        dimensions=2,
        canonicalizer_version="canonical.v1",
        input_hash="1" * 64,
        vector=(1.0, 0.0),
    )
    right = replace(left, input_hash="2" * 64, vector=(0.0, 1.0))
    assert cosine_similarity(left, right) == 0.0
    with pytest.raises(ValueError, match="compatible"):
        cosine_similarity(left, replace(right, canonicalizer_version="canonical.v2"))
    with pytest.raises(ValueError, match="non-zero"):
        cosine_similarity(left, replace(right, vector=(0.0, 0.0)))
