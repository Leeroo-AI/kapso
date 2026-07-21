"""Shared OpenAI embedding boundary and vector contracts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Protocol

from openai import OpenAI


def _require_nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_non_negative_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _require_exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"{name} fields are invalid")


@dataclass(frozen=True)
class EmbeddingSpaceId:
    """Exact semantic space in which vectors may be compared."""

    provider: str
    model: str
    dimensions: int
    canonicalizer_version: str

    def __post_init__(self) -> None:
        _require_nonempty_text(self.provider, "embedding provider")
        _require_nonempty_text(self.model, "embedding model")
        _require_positive_integer(self.dimensions, "embedding dimensions")
        _require_nonempty_text(
            self.canonicalizer_version, "embedding canonicalizer version"
        )

    @property
    def value(self) -> str:
        preimage = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(preimage).hexdigest()
        return f"embedding-space:sha256:{digest}"


@dataclass(frozen=True)
class EmbeddingSettings:
    enabled: bool
    provider: str
    model: str
    dimensions: int
    batch_size: int
    timeout_seconds: float
    max_retries: int
    canonicalizer_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("embedding enabled setting must be boolean")
        if self.provider != "openai":
            raise ValueError("only the OpenAI embedding provider is supported")
        _require_nonempty_text(self.model, "embedding model")
        _require_positive_integer(self.dimensions, "embedding dimensions")
        _require_positive_integer(self.batch_size, "embedding batch size")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("embedding timeout must be positive")
        _require_non_negative_integer(self.max_retries, "embedding max retries")
        _require_nonempty_text(
            self.canonicalizer_version, "embedding canonicalizer version"
        )

    @property
    def embedding_space_id(self) -> EmbeddingSpaceId:
        return EmbeddingSpaceId(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            canonicalizer_version=self.canonicalizer_version,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EmbeddingSettings:
        _require_exact_keys(
            data,
            {
                "enabled",
                "provider",
                "model",
                "dimensions",
                "batch_size",
                "timeout_seconds",
                "max_retries",
                "canonicalizer_version",
            },
            "embedding settings",
        )
        return cls(**data)


@dataclass(frozen=True)
class EmbeddingRecord:
    provider: str
    model: str
    dimensions: int
    canonicalizer_version: str
    input_hash: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        space = self.embedding_space_id
        if not isinstance(self.input_hash, str) or not _is_sha256(self.input_hash):
            raise ValueError("embedding input hash must be a SHA-256 digest")
        if not isinstance(self.vector, (list, tuple)):
            raise ValueError("embedding vector must be a list")
        vector = tuple(
            _require_finite_number(value, "embedding vector value")
            for value in self.vector
        )
        if len(vector) != space.dimensions:
            raise ValueError("embedding dimensions must match vector length")
        object.__setattr__(self, "vector", vector)

    @property
    def embedding_space_id(self) -> EmbeddingSpaceId:
        return EmbeddingSpaceId(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            canonicalizer_version=self.canonicalizer_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "canonicalizer_version": self.canonicalizer_version,
            "input_hash": self.input_hash,
            "vector": list(self.vector),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EmbeddingRecord:
        _require_exact_keys(
            data,
            {
                "provider",
                "model",
                "dimensions",
                "canonicalizer_version",
                "input_hash",
                "vector",
            },
            "embedding record",
        )
        return cls(**data)


@dataclass(frozen=True)
class EmbeddingTelemetry:
    provider: str
    model: str
    call_count: int
    input_tokens: int | None
    duration_seconds: float
    cost_usd: float | None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.provider, "embedding telemetry provider")
        _require_nonempty_text(self.model, "embedding telemetry model")
        _require_positive_integer(self.call_count, "embedding call count")
        if self.input_tokens is not None:
            _require_non_negative_integer(
                self.input_tokens, "embedding input token count"
            )
        _require_non_negative_number(self.duration_seconds, "embedding duration")
        if self.cost_usd is not None:
            _require_non_negative_number(self.cost_usd, "embedding cost")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EmbeddingTelemetry:
        _require_exact_keys(
            data,
            {
                "provider",
                "model",
                "call_count",
                "input_tokens",
                "duration_seconds",
                "cost_usd",
            },
            "embedding telemetry",
        )
        return cls(**data)


@dataclass(frozen=True)
class EmbeddingBatch:
    records: tuple[EmbeddingRecord, ...]
    telemetry: EmbeddingTelemetry

    def __post_init__(self) -> None:
        if not isinstance(self.records, (list, tuple)) or not self.records:
            raise ValueError("embedding batch records must be a non-empty list")
        records = tuple(self.records)
        if not all(isinstance(record, EmbeddingRecord) for record in records):
            raise ValueError("embedding batch contains an invalid record")
        if not isinstance(self.telemetry, EmbeddingTelemetry):
            raise ValueError("embedding batch telemetry is invalid")
        spaces = {record.embedding_space_id for record in records}
        if len(spaces) != 1:
            raise ValueError("embedding batch records must share one space")
        space = next(iter(spaces))
        if (
            self.telemetry.provider != space.provider
            or self.telemetry.model != space.model
        ):
            raise ValueError("embedding telemetry does not match record space")
        object.__setattr__(self, "records", records)


class EmbeddingProvider(Protocol):
    settings: EmbeddingSettings

    def embed(self, texts: Iterable[str]) -> EmbeddingBatch:
        """Embed every complete input text or raise the provider error."""


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
        return _require_non_negative_integer(
            prompt_tokens, "embedding usage prompt tokens"
        )


def complete_input_hash(text: str) -> str:
    if not isinstance(text, str) or not text:
        raise ValueError("embedding input must be a non-empty string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_can_be_reused(
    record: EmbeddingRecord,
    text: str,
    settings: EmbeddingSettings,
) -> bool:
    return (
        settings.enabled
        and record.embedding_space_id == settings.embedding_space_id
        and record.input_hash == complete_input_hash(text)
    )


def cosine_similarity(left: EmbeddingRecord, right: EmbeddingRecord) -> float:
    if left.embedding_space_id != right.embedding_space_id:
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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require_finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


__all__ = [
    "EmbeddingBatch",
    "EmbeddingProvider",
    "EmbeddingRecord",
    "EmbeddingSettings",
    "EmbeddingSpaceId",
    "EmbeddingTelemetry",
    "OpenAIEmbeddingProvider",
    "complete_input_hash",
    "cosine_similarity",
    "embedding_can_be_reused",
]
