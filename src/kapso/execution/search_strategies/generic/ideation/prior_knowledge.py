"""Pinned cross-run retrieval boundary for one ideation campaign."""

from __future__ import annotations

from dataclasses import dataclass

from kapso.core.embedding_contracts import EmbeddingProvider, EmbeddingTelemetry
from kapso.cross_run.canonical import canonical_json_bytes
from kapso.cross_run.knowledge.retrieval import (
    CrossRunRetrievalResult,
    CrossRunRetriever,
    PriorKnowledgeQuery,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CampaignEvidenceSnapshot,
    GapState,
    IdeationCrossRunIdentity,
    SearchDirective,
)


@dataclass(frozen=True)
class IdeationPriorKnowledgeResult:
    """Exact durable retrieval output plus optional query-vector telemetry."""

    retrieval: CrossRunRetrievalResult
    embedding_telemetry: EmbeddingTelemetry | None


@dataclass(frozen=True)
class IdeationCrossRunRuntime:
    """Verified runtime services corresponding to one pinned launch identity."""

    identity: IdeationCrossRunIdentity
    retriever: CrossRunRetriever
    query_embedding_provider: EmbeddingProvider | None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, IdeationCrossRunIdentity):
            raise TypeError("ideation cross-run identity is invalid")
        if not isinstance(self.retriever, CrossRunRetriever):
            raise TypeError("ideation cross-run retriever is invalid")
        if self.retriever.source_snapshot_id != self.identity.knowledge_snapshot_id:
            raise ValueError("ideation retriever owns another knowledge snapshot")
        spaces = self.retriever.semantic_embedding_space_ids
        if spaces and self.query_embedding_provider is None:
            raise ValueError("semantic knowledge index requires a query embedder")
        if spaces and self.identity.embedding_space_id not in spaces:
            raise ValueError("launch embedding space is absent from knowledge index")
        if self.query_embedding_provider is not None:
            provider_space = (
                self.query_embedding_provider.settings.embedding_space_id.value
            )
            if provider_space != self.identity.embedding_space_id:
                raise ValueError(
                    "ideation query embedder differs from the pinned launch space"
                )
            if provider_space not in spaces:
                raise ValueError(
                    "ideation query embedder is absent from the knowledge index"
                )

    def retrieve(
        self,
        *,
        problem_statement: str,
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
    ) -> IdeationPriorKnowledgeResult:
        """Retrieve only after local policy and directive planning are complete."""

        current_gaps = tuple(
            canonical_json_bytes(gap.to_dict()).decode("utf-8")
            for gap in evidence_snapshot.gaps
            if gap.state != GapState.CLOSED
        )
        directive_text = canonical_json_bytes(directive.to_dict()).decode("utf-8")
        query = PriorKnowledgeQuery(
            task_context_binding=self.identity.task_context_binding,
            problem=problem_statement,
            current_gaps=current_gaps,
            directive=directive_text,
            effect_evaluation_fingerprint_ids=(
                self.identity.effect_evaluation_fingerprint_ids
            ),
            active_exclusions=self.identity.active_exclusions,
        )
        telemetry = None
        if self.query_embedding_provider is not None:
            embedding_batch = self.query_embedding_provider.embed((query.lexical_text,))
            if len(embedding_batch.records) != 1:
                raise ValueError("query embedder returned an invalid batch size")
            telemetry = embedding_batch.telemetry
            query = PriorKnowledgeQuery(
                task_context_binding=query.task_context_binding,
                problem=query.problem,
                current_gaps=query.current_gaps,
                directive=query.directive,
                effect_evaluation_fingerprint_ids=(
                    query.effect_evaluation_fingerprint_ids
                ),
                active_exclusions=query.active_exclusions,
                query_embedding=embedding_batch.records[0],
            )
        retrieval = self.retriever.retrieve(query)
        materialization = retrieval.access_materialization
        packet = materialization.prior_knowledge_snapshot
        if packet.source_snapshot_id != self.identity.knowledge_snapshot_id:
            raise ValueError("retrieved prior packet uses another knowledge snapshot")
        if (
            packet.task_context_binding_id
            != self.identity.task_context_binding.task_context_binding_id
        ):
            raise ValueError("retrieved prior packet uses another task binding")
        return IdeationPriorKnowledgeResult(
            retrieval=retrieval,
            embedding_telemetry=telemetry,
        )

    def validate_persisted_batch_identity(
        self,
        identity: IdeationCrossRunIdentity | None,
    ) -> None:
        """Fail if resume attempts to use a batch from another pinned launch."""

        if identity != self.identity:
            raise ValueError("resume batch cross-run identity changed")
