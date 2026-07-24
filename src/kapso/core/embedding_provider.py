"""OpenAI provider boundary for dependency-pure embedding contracts."""

from __future__ import annotations

import time
from typing import Any, Iterable

from openai import OpenAI

from kapso.core.embedding_contracts import (
    EmbeddingBatch,
    EmbeddingRecord,
    EmbeddingSettings,
    EmbeddingTelemetry,
    complete_input_hash,
)


class OpenAIEmbeddingProvider:
    """Use only the official OpenAI embeddings endpoint."""

    def __init__(self, settings: EmbeddingSettings, client: Any = None):
        if not settings.enabled:
            raise ValueError("disabled embeddings must not construct a provider")
        self.settings = settings
        self.client = (
            client
            if client is not None
            else OpenAI(
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
                _strict_response_validation=True,
            )
        )

    def embed(self, texts: Iterable[str]) -> EmbeddingBatch:
        inputs = tuple(texts)
        if not inputs:
            raise ValueError("embedding input must not be empty")
        if not all(isinstance(text, str) and text for text in inputs):
            raise ValueError("embedding inputs must be non-empty strings")

        records: list[EmbeddingRecord] = []
        input_tokens = 0
        call_count = 0
        started = time.monotonic()
        for start in range(0, len(inputs), self.settings.batch_size):
            input_batch = inputs[start : start + self.settings.batch_size]
            response = self.client.embeddings.create(
                model=self.settings.model,
                dimensions=self.settings.dimensions,
                encoding_format="float",
                input=list(input_batch),
            )
            records.extend(self._records(input_batch, response))
            input_tokens += self._input_tokens(response)
            call_count += 1
        duration = time.monotonic() - started
        return EmbeddingBatch(
            records=tuple(records),
            telemetry=EmbeddingTelemetry(
                provider=self.settings.provider,
                model=self.settings.model,
                call_count=call_count,
                input_tokens=input_tokens,
                duration_seconds=duration,
                cost_usd=None,
            ),
        )

    def _records(self, inputs: tuple[str, ...], response: Any) -> list[EmbeddingRecord]:
        data = response.data
        if not isinstance(data, list) or len(data) != len(inputs):
            raise ValueError("embedding response count does not match input count")
        if not all(
            not isinstance(item.index, bool) and isinstance(item.index, int)
            for item in data
        ):
            raise ValueError("embedding response indices are invalid")
        ordered = tuple(sorted(data, key=lambda item: item.index))
        if tuple(item.index for item in ordered) != tuple(range(len(inputs))):
            raise ValueError("embedding response indices are invalid")
        return [
            EmbeddingRecord(
                provider=self.settings.provider,
                model=self.settings.model,
                dimensions=self.settings.dimensions,
                canonicalizer_version=self.settings.canonicalizer_version,
                input_hash=complete_input_hash(text),
                vector=item.embedding,
            )
            for text, item in zip(inputs, ordered)
        ]

    @staticmethod
    def _input_tokens(response: Any) -> int:
        prompt_tokens = response.usage.prompt_tokens
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
        ):
            raise ValueError(
                "embedding usage prompt tokens must be a non-negative integer"
            )
        return prompt_tokens


__all__ = ["OpenAIEmbeddingProvider"]
