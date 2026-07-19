"""Narrow OpenAI embeddings boundary and local similarity operations."""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Tuple

from openai import OpenAI

from kapso.execution.search_strategies.generic.ideation.types import (
    EmbeddingRecord,
    EmbeddingTelemetry,
    IdeaRecord,
)


@dataclass(frozen=True)
class EmbeddingSettings:
    enabled: bool
    model: str
    dimensions: int
    timeout_seconds: float
    max_retries: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("embedding enabled setting must be boolean")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("embedding model must be non-empty")
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions < 1
        ):
            raise ValueError("embedding dimensions must be positive")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("embedding timeout must be positive")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("embedding max retries must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmbeddingSettings":
        expected = {
            "enabled",
            "model",
            "dimensions",
            "timeout_seconds",
            "max_retries",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("embedding settings fields are invalid")
        return cls(**data)


@dataclass(frozen=True)
class EmbeddingBatch:
    records: Tuple[EmbeddingRecord, ...]
    telemetry: EmbeddingTelemetry


class EmbeddingProvider(Protocol):
    settings: EmbeddingSettings

    def embed(self, texts: Iterable[str]) -> EmbeddingBatch:
        """Embed complete input texts or raise the provider error."""


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
        started = time.monotonic()
        response = self.client.embeddings.create(
            model=self.settings.model,
            dimensions=self.settings.dimensions,
            encoding_format="float",
            input=list(inputs),
        )
        duration = time.monotonic() - started
        data = response.data
        if not isinstance(data, list) or len(data) != len(inputs):
            raise ValueError("embedding response count does not match input count")
        ordered = tuple(sorted(data, key=lambda item: item.index))
        if tuple(item.index for item in ordered) != tuple(range(len(inputs))):
            raise ValueError("embedding response indices are invalid")
        records = []
        for text, item in zip(inputs, ordered):
            vector = tuple(item.embedding)
            records.append(
                EmbeddingRecord(
                    provider="openai",
                    model=self.settings.model,
                    dimensions=self.settings.dimensions,
                    input_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    vector=vector,
                )
            )
        prompt_tokens = response.usage.prompt_tokens
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
        ):
            raise ValueError("embedding usage prompt tokens are invalid")
        return EmbeddingBatch(
            records=tuple(records),
            telemetry=EmbeddingTelemetry(
                provider="openai",
                model=self.settings.model,
                call_count=1,
                input_tokens=prompt_tokens,
                duration_seconds=duration,
            ),
        )


def canonical_idea_embedding_text(idea: IdeaRecord) -> str:
    """Return the complete stable representation used for idea similarity."""
    return json.dumps(
        {
            "proposal": idea.proposal,
            "descriptor": idea.descriptor.to_dict(),
            "assumptions": list(idea.assumptions),
            "evidence_refs": list(idea.evidence_refs),
            "directive_rationale": idea.directive_rationale,
            "claim_ids": list(idea.claim_ids),
            "resolves_claim_ids": list(idea.resolves_claim_ids),
            "evaluation_method": idea.evaluation_method,
            "expected_observations": list(idea.expected_observations),
            "resource_request": idea.resource_request,
            "claimed_nearest_idea_id": idea.claimed_nearest_idea_id,
            "claimed_nearest_experiment_node_id": (
                idea.claimed_nearest_experiment_node_id
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def embedding_can_be_reused(
    record: EmbeddingRecord,
    text: str,
    settings: EmbeddingSettings,
) -> bool:
    return (
        settings.enabled
        and record.provider == "openai"
        and record.model == settings.model
        and record.dimensions == settings.dimensions
        and record.input_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


def cosine_similarity(left: EmbeddingRecord, right: EmbeddingRecord) -> float:
    if (
        left.provider != right.provider
        or left.model != right.model
        or left.dimensions != right.dimensions
    ):
        raise ValueError("embedding records are not compatible")
    left_norm = math.sqrt(sum(value * value for value in left.vector))
    right_norm = math.sqrt(sum(value * value for value in right.vector))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity requires non-zero vectors")
    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left.vector, right.vector)
    ) / (left_norm * right_norm)
    if not math.isfinite(similarity):
        raise ValueError("cosine similarity is not finite")
    return similarity
