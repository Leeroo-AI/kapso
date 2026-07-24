"""Dependency-pure, self-authenticating embedding-space authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_identifier
from kapso.cross_run.contracts import ContractValidationError, StrictContract


@dataclass(frozen=True)
class EmbeddingSpace(StrictContract):
    """Exact vector-comparison space, independent of any provider client."""

    embedding_space_id: str
    provider: str
    model: str
    dimensions: int
    canonicalizer_version: str

    CONTENT_NAMESPACE: ClassVar[str] = "embedding-space"
    IDENTITY_FIELD: ClassVar[str] = "embedding_space_id"

    def _validate(self) -> None:
        require_identifier(self.provider, "embedding provider")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ContractValidationError("embedding model must not be empty")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise ContractValidationError("embedding dimensions must be positive")
        require_identifier(self.canonicalizer_version, "canonicalizer_version")


__all__ = ["EmbeddingSpace"]
