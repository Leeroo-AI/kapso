"""Dependency and ownership contracts for embeddings."""

import importlib.util
import subprocess
import sys

from kapso.core.embedding_contracts import (
    EmbeddingBatch,
    EmbeddingRecord,
    EmbeddingSettings,
    EmbeddingTelemetry,
    complete_input_hash,
)


def test_embedding_contracts_import_without_openai_provider_runtime():
    script = "\n".join(
        (
            "import sys",
            "sys.modules['openai'] = None",
            "sys.modules['kapso.core.embedding_provider'] = None",
            "import kapso.core.embedding_contracts as contracts",
            "assert not hasattr(contracts, 'OpenAIEmbeddingProvider')",
        )
    )

    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    assert completed.stderr == ""


def test_embedding_contracts_own_complete_provider_independent_batch():
    settings = EmbeddingSettings(
        enabled=True,
        provider="openai",
        model="text-embedding-3-small",
        dimensions=2,
        batch_size=4,
        timeout_seconds=30.0,
        max_retries=2,
        canonicalizer_version="kapso.idea_embedding.v1",
    )
    record = EmbeddingRecord(
        provider=settings.provider,
        model=settings.model,
        dimensions=settings.dimensions,
        canonicalizer_version=settings.canonicalizer_version,
        input_hash=complete_input_hash("complete input"),
        vector=(0.25, 0.75),
    )
    telemetry = EmbeddingTelemetry(
        provider=settings.provider,
        model=settings.model,
        call_count=1,
        input_tokens=2,
        duration_seconds=0.5,
        cost_usd=None,
    )

    batch = EmbeddingBatch(records=(record,), telemetry=telemetry)

    assert batch.records == (record,)
    assert record.embedding_space_id == settings.embedding_space_id
    assert importlib.util.find_spec("kapso.core.embeddings") is None
