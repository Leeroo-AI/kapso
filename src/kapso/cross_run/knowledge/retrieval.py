"""Compatibility-first deterministic retrieval from one pinned snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from kapso.core.embedding_contracts import EmbeddingRecord, complete_input_hash
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    AdmissionState,
    ContractValidationError,
    KnowledgeClaim,
    PriorKnowledgeSnapshot,
    TaskContextBinding,
    TransferCompatibility,
)
from kapso.cross_run.knowledge.access import PriorKnowledgeAccessMaterialization
from kapso.cross_run.knowledge.index import SnapshotSearchIndex
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackage
from kapso.cross_run.settings import RetrievalSettings

_OUTCOME_ORDER = ("positive", "negative", "inconclusive", "frontier")
_OUTCOMES = frozenset(_OUTCOME_ORDER)


class CrossRunRetrievalError(ValueError):
    """A retrieval request or its pinned scientific-memory input is invalid."""


@dataclass(frozen=True)
class PriorKnowledgeQuery:
    """Complete deterministic query context supplied by a trusted caller."""

    task_context_binding: TaskContextBinding
    problem: str
    current_gaps: tuple[str, ...]
    directive: str
    effect_evaluation_fingerprint_ids: tuple[str, ...] = ()
    active_exclusions: tuple[str, ...] = ()
    query_embedding: EmbeddingRecord | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.problem, "problem"),
            (self.directive, "directive"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"prior-knowledge {name} is empty")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.current_gaps
        ):
            raise ContractValidationError(
                "prior-knowledge current gaps must be non-empty text"
            )
        if self.effect_evaluation_fingerprint_ids != tuple(
            sorted(set(self.effect_evaluation_fingerprint_ids))
        ):
            raise ContractValidationError(
                "effect evaluation fingerprint IDs must be sorted and unique"
            )
        for fingerprint_id in self.effect_evaluation_fingerprint_ids:
            require_content_id(
                fingerprint_id,
                "effect evaluation fingerprint ID",
            )
        if self.active_exclusions != tuple(sorted(set(self.active_exclusions))):
            raise ContractValidationError("active exclusions must be sorted and unique")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.active_exclusions
        ):
            raise ContractValidationError(
                "active exclusions must be non-empty exact exclusion text"
            )
        if self.query_embedding is not None:
            if not isinstance(self.query_embedding, EmbeddingRecord):
                raise ContractValidationError(
                    "semantic query embedding must be an EmbeddingRecord"
                )
            if self.query_embedding.input_hash != complete_input_hash(
                self.lexical_text
            ):
                raise ContractValidationError(
                    "semantic query embedding does not own the complete query text"
                )

    @property
    def lexical_text(self) -> str:
        """Return every user-authored query component without clipping it."""

        return "\n".join((self.problem, *self.current_gaps, self.directive))

    def packet_query(self) -> Mapping[str, Any]:
        semantic_query = None
        if self.query_embedding is not None:
            semantic_query = {
                "embedding_space_id": self.query_embedding.embedding_space_id.value,
                "input_hash": self.query_embedding.input_hash,
                "query_vector_digest": tree_or_blob_digest(
                    canonical_json_bytes(self.query_embedding.vector)
                ),
            }
        return MappingProxyType(
            {
                "active_exclusions": self.active_exclusions,
                "current_gaps": self.current_gaps,
                "directive": self.directive,
                "effect_evaluation_fingerprint_ids": (
                    self.effect_evaluation_fingerprint_ids
                ),
                "problem": self.problem,
                "semantic_query": semantic_query,
                "task_context_binding": self.task_context_binding.to_dict(),
            }
        )


@dataclass(frozen=True)
class RetrievalSelection:
    """Auditable rank metadata for one selected complete root record."""

    record_id: str
    record_kind: str
    compatibility: TransferCompatibility
    outcome: str
    evidence_quality: int
    lexical_score: float
    retrieval_utility: float
    semantic_score: float
    recency: str
    proof_reference_ids: tuple[str, ...]


@dataclass(frozen=True)
class CrossRunRetrievalResult:
    """Persistable access material plus deterministic selection audit."""

    prior_knowledge_snapshot: PriorKnowledgeSnapshot
    access_materialization: PriorKnowledgeAccessMaterialization
    selections: tuple[RetrievalSelection, ...]


@dataclass(frozen=True)
class _Candidate:
    record_id: str
    record_kind: str
    compatibility: TransferCompatibility
    outcome: str
    evidence_quality: int
    lexical_score: float
    retrieval_utility: float
    semantic_score: float
    recency: str
    run_id: str
    approach_family: str
    lineage_id: str


class CrossRunRetriever:
    """Retrieve admitted roots without granting similarity admission authority."""

    def __init__(
        self,
        snapshot: KnowledgeSnapshotPackage,
        search_index: SnapshotSearchIndex,
        settings: RetrievalSettings,
    ) -> None:
        if not isinstance(snapshot, KnowledgeSnapshotPackage):
            raise TypeError("snapshot must be KnowledgeSnapshotPackage")
        if not isinstance(search_index, SnapshotSearchIndex):
            raise TypeError("search_index must be SnapshotSearchIndex")
        if not isinstance(settings, RetrievalSettings):
            raise TypeError("settings must be RetrievalSettings")
        snapshot.verify()
        trusted_index = SnapshotSearchIndex.open(snapshot.prepared, search_index.files)
        trusted_index.verify(snapshot.manifest)
        if search_index != trusted_index:
            raise CrossRunRetrievalError(
                "in-memory search index differs from its verified sidecar bytes"
            )
        if (
            trusted_index.record_closure_digest != snapshot.record_closure_digest
            or trusted_index.manifest.indexed_record_ids != snapshot.retrieval_root_ids
        ):
            raise CrossRunRetrievalError(
                "search index does not belong to the exact snapshot record closure"
            )
        if canonical_json_bytes(settings.to_dict()) != canonical_json_bytes(
            snapshot.manifest.prompt_budget_policy
        ):
            raise CrossRunRetrievalError(
                "retrieval settings differ from the pinned snapshot policy"
            )
        self._snapshot = snapshot
        self._index = trusted_index
        self._settings = settings
        self._records_by_id = MappingProxyType(
            {record["record_id"]: record for record in snapshot.record_envelopes}
        )

    @property
    def semantic_embedding_space_ids(self) -> tuple[str, ...]:
        """Return the exact semantic spaces published by the pinned index."""

        return tuple(
            sidecar.embedding_space.embedding_space_id
            for sidecar in self._index.manifest.vector_sidecars
        )

    @property
    def source_snapshot_id(self) -> str:
        """Return the immutable knowledge snapshot owned by this retriever."""

        return self._snapshot.manifest.snapshot_id

    def retrieve(self, query: PriorKnowledgeQuery) -> CrossRunRetrievalResult:
        """Return byte-budgeted roots and their complete recursive proof closure."""

        if not isinstance(query, PriorKnowledgeQuery):
            raise TypeError("query must be PriorKnowledgeQuery")
        query.task_context_binding.validate_against(
            self._snapshot.prepared.scope_contract
        )
        if (
            query.task_context_binding.scope_contract_id
            != self._snapshot.manifest.scope_contract_id
            or query.task_context_binding.scope_id != self._snapshot.manifest.scope_id
        ):
            raise CrossRunRetrievalError("query belongs to another knowledge scope")
        lexical_scores = self._index.lexical_scores(query.lexical_text)
        semantic_scores: Mapping[str, float] = MappingProxyType({})
        if query.query_embedding is not None:
            semantic_scores = self._index.semantic_scores(
                query.query_embedding.vector,
                query.query_embedding.embedding_space_id.value,
            )
        candidates = self._eligible_candidates(
            query,
            lexical_scores,
            semantic_scores,
        )
        selected: list[_Candidate] = []
        selected_closures: dict[str, tuple[str, ...]] = {}
        diversity_counts: dict[str, dict[str, int]] = {
            "run": {},
            "family": {},
            "lineage": {},
            "outcome": {},
            "type": {},
        }
        packet_query = query.packet_query()
        budget_policy = MappingProxyType(self._settings.to_dict())
        for candidate in candidates:
            if len(selected) >= self._settings.max_records:
                break
            if not self._within_diversity_caps(candidate, diversity_counts):
                continue
            closure = self._proof_closure(candidate.record_id)
            tentative = (*selected, candidate)
            tentative_closures = {
                **selected_closures,
                candidate.record_id: closure,
            }
            packet, materialization = self._materialize(
                tentative,
                tentative_closures,
                packet_query,
                budget_policy,
            )
            if (
                _proof_closed_record_bytes(materialization)
                > self._settings.prompt_byte_budget
                or len(materialization.to_json_bytes())
                > self._settings.materialization_byte_budget
            ):
                continue
            selected.append(candidate)
            selected_closures[candidate.record_id] = closure
            self._increment_diversity(candidate, diversity_counts)
        packet, materialization = self._materialize(
            tuple(selected),
            selected_closures,
            packet_query,
            budget_policy,
        )
        if (
            _proof_closed_record_bytes(materialization)
            > self._settings.prompt_byte_budget
        ):
            raise CrossRunRetrievalError(
                "empty prior-knowledge packet exceeds its prompt byte budget"
            )
        if (
            len(materialization.to_json_bytes())
            > self._settings.materialization_byte_budget
        ):
            raise CrossRunRetrievalError(
                "empty prior-knowledge materialization exceeds its byte budget"
            )
        selected_root_ids = set(packet.selected_record_ids)
        selections = tuple(
            RetrievalSelection(
                record_id=candidate.record_id,
                record_kind=candidate.record_kind,
                compatibility=candidate.compatibility,
                outcome=candidate.outcome,
                evidence_quality=candidate.evidence_quality,
                lexical_score=candidate.lexical_score,
                retrieval_utility=candidate.retrieval_utility,
                semantic_score=candidate.semantic_score,
                recency=candidate.recency,
                proof_reference_ids=tuple(
                    proof_id
                    for proof_id in selected_closures[candidate.record_id]
                    if proof_id not in selected_root_ids
                ),
            )
            for candidate in selected
        )
        return CrossRunRetrievalResult(
            prior_knowledge_snapshot=packet,
            access_materialization=materialization,
            selections=selections,
        )

    def _eligible_candidates(
        self,
        query: PriorKnowledgeQuery,
        lexical_scores: Mapping[str, float],
        semantic_scores: Mapping[str, float],
    ) -> tuple[_Candidate, ...]:
        candidates: list[_Candidate] = []
        for record_id in self._snapshot.retrieval_root_ids:
            metadata = self._index.metadata_by_id[record_id]
            if (
                metadata["trust_state"] != AdmissionState.ADMITTED.value
                or metadata["scope_contract_id"]
                != query.task_context_binding.scope_contract_id
                or metadata["scope_id"] != query.task_context_binding.scope_id
            ):
                continue
            envelope = self._records_by_id[record_id]
            compatibility = self._compatibility(query, envelope)
            if compatibility is TransferCompatibility.INCOMPATIBLE:
                continue
            if not self._evaluation_is_comparable(query, envelope):
                continue
            semantic_score = semantic_scores.get(record_id, 0.0)
            lexical_score = lexical_scores.get(record_id, 0.0)
            utility = lexical_score
            if query.query_embedding is not None:
                utility = (
                    self._settings.lexical_weight * lexical_score
                    + self._settings.semantic_weight * semantic_score
                )
            outcome = metadata["outcome"]
            if outcome not in _OUTCOMES:
                raise CrossRunRetrievalError("index contains an unknown outcome slot")
            lineage_ids = metadata["lineage_ids"]
            lineage_id = lineage_ids[0] if lineage_ids else record_id
            timestamps = metadata["timestamps"]
            recency = max(timestamps) if timestamps else ""
            candidates.append(
                _Candidate(
                    record_id=record_id,
                    record_kind=envelope["record_kind"],
                    compatibility=compatibility,
                    outcome=outcome,
                    evidence_quality=_evidence_quality(envelope),
                    lexical_score=lexical_score,
                    retrieval_utility=utility,
                    semantic_score=semantic_score,
                    recency=recency,
                    run_id=metadata["run_id"] or record_id,
                    approach_family=(
                        metadata["approach_family"]
                        or metadata["mechanism"]
                        or envelope["record_kind"]
                    ),
                    lineage_id=lineage_id,
                )
            )
        candidates.sort(key=lambda candidate: candidate.record_id)
        candidates.sort(key=lambda candidate: candidate.recency, reverse=True)
        candidates.sort(
            key=lambda candidate: candidate.retrieval_utility,
            reverse=True,
        )
        candidates.sort(
            key=lambda candidate: candidate.evidence_quality,
            reverse=True,
        )
        candidates.sort(
            key=lambda candidate: (
                0
                if candidate.compatibility is TransferCompatibility.EXACT_CONTEXT
                else 1
            )
        )
        return _round_robin_outcomes(tuple(candidates))

    def _compatibility(
        self,
        query: PriorKnowledgeQuery,
        envelope: Mapping[str, Any],
    ) -> TransferCompatibility:
        payload = envelope["payload"]
        if envelope["record_kind"] in {"prior-idea", "transfer-episode"}:
            source_context = TaskContextBinding.from_dict(
                payload["task_context_binding"]
            )
            source_context.validate_against(self._snapshot.prepared.scope_contract)
            if (
                source_context.task_family_id
                != query.task_context_binding.task_family_id
            ):
                return TransferCompatibility.INCOMPATIBLE
            return source_context.compatibility_with(query.task_context_binding)
        if envelope["record_kind"] != "knowledge-claim-revision":
            raise CrossRunRetrievalError("snapshot exposes an unknown root kind")
        claim = KnowledgeClaim.from_dict(payload)
        if any(
            query.task_context_binding.transfer_dimensions.get(dimension_id)
            != predicate_value
            for dimension_id, predicate_value in claim.applicability_predicates.items()
        ) or set(claim.explicit_exclusions) & set(query.active_exclusions):
            return TransferCompatibility.INCOMPATIBLE
        evidence_compatibilities = tuple(
            self._compatibility(
                query,
                self._records_by_id[episode_id],
            )
            for episode_id in (
                *claim.supporting_episode_ids,
                *claim.contradicting_episode_ids,
            )
        )
        if TransferCompatibility.EXACT_CONTEXT in evidence_compatibilities:
            return TransferCompatibility.EXACT_CONTEXT
        if TransferCompatibility.ANALOGICAL in evidence_compatibilities:
            return TransferCompatibility.ANALOGICAL
        return TransferCompatibility.ANALOGICAL

    def _evaluation_is_comparable(
        self,
        query: PriorKnowledgeQuery,
        envelope: Mapping[str, Any],
    ) -> bool:
        requested = set(query.effect_evaluation_fingerprint_ids)
        if not requested or envelope["record_kind"] == "prior-idea":
            return True
        if envelope["record_kind"] == "knowledge-claim-revision":
            claim = KnowledgeClaim.from_dict(envelope["payload"])
            return any(
                self._evaluation_is_comparable(
                    query,
                    self._records_by_id[episode_id],
                )
                for episode_id in (
                    *claim.supporting_episode_ids,
                    *claim.contradicting_episode_ids,
                )
            )
        terminal = envelope["payload"]["attempts"][
            envelope["payload"]["terminal_attempt_revision"]
        ]
        fingerprint_id = terminal["score_of_record_fingerprint_id"]
        if fingerprint_id is None:
            return False
        return fingerprint_id in requested

    def _proof_closure(self, root_id: str) -> tuple[str, ...]:
        pending = [root_id, self._snapshot.prepared.scope_contract_id]
        visited: set[str] = set()
        while pending:
            record_id = pending.pop()
            if record_id in visited:
                continue
            envelope = self._records_by_id.get(record_id)
            if envelope is None:
                raise CrossRunRetrievalError(
                    f"proof dependency is absent from snapshot: {record_id}"
                )
            visited.add(record_id)
            pending.extend(self._snapshot.prepared.proof_dependencies[record_id])
        return tuple(sorted(visited))

    def _within_diversity_caps(
        self,
        candidate: _Candidate,
        counts: Mapping[str, Mapping[str, int]],
    ) -> bool:
        checks = (
            ("run", candidate.run_id, self._settings.max_records_per_run),
            (
                "family",
                candidate.approach_family,
                self._settings.max_records_per_family,
            ),
            (
                "lineage",
                candidate.lineage_id,
                self._settings.max_records_per_lineage,
            ),
            ("outcome", candidate.outcome, self._settings.max_records_per_outcome),
            ("type", candidate.record_kind, self._settings.max_records_per_type),
        )
        return all(counts[group].get(key, 0) < limit for group, key, limit in checks)

    @staticmethod
    def _increment_diversity(
        candidate: _Candidate,
        counts: dict[str, dict[str, int]],
    ) -> None:
        for group, key in (
            ("run", candidate.run_id),
            ("family", candidate.approach_family),
            ("lineage", candidate.lineage_id),
            ("outcome", candidate.outcome),
            ("type", candidate.record_kind),
        ):
            counts[group][key] = counts[group].get(key, 0) + 1

    def _materialize(
        self,
        candidates: tuple[_Candidate, ...],
        closures: Mapping[str, tuple[str, ...]],
        packet_query: Mapping[str, Any],
        budget_policy: Mapping[str, Any],
    ) -> tuple[PriorKnowledgeSnapshot, PriorKnowledgeAccessMaterialization]:
        selected_ids = tuple(sorted(candidate.record_id for candidate in candidates))
        selected_records = tuple(
            self._records_by_id[record_id] for record_id in selected_ids
        )
        proof_ids = tuple(
            sorted(
                {
                    proof_id
                    for candidate in candidates
                    for proof_id in closures[candidate.record_id]
                }
                - set(selected_ids)
            )
        )
        selected_id_set = set(selected_ids)
        selection_metadata = {
            candidate.record_id: {
                "compatibility": candidate.compatibility.value,
                "evidence_quality": candidate.evidence_quality,
                "lexical_score": candidate.lexical_score,
                "outcome": candidate.outcome,
                "proof_reference_ids": tuple(
                    proof_id
                    for proof_id in closures[candidate.record_id]
                    if proof_id not in selected_id_set
                ),
                "rank": rank,
                "recency": candidate.recency,
                "retrieval_utility": candidate.retrieval_utility,
                "semantic_score": candidate.semantic_score,
            }
            for rank, candidate in enumerate(candidates)
        }
        packet = PriorKnowledgeSnapshot.mint(
            source_snapshot_id=self._snapshot.manifest.snapshot_id,
            query=packet_query,
            retrieval_policy_version=(self._snapshot.manifest.retrieval_policy_version),
            task_context_binding_id=(
                packet_query["task_context_binding"]["task_context_binding_id"]
            ),
            selected_records=selected_records,
            selected_record_ids=selected_ids,
            proof_reference_ids=proof_ids,
            selection_metadata=selection_metadata,
            prompt_budget_policy=budget_policy,
            records_digest=tree_or_blob_digest(canonical_json_bytes(selected_records)),
        )
        proof_records = tuple(self._records_by_id[proof_id] for proof_id in proof_ids)
        materialization = PriorKnowledgeAccessMaterialization.mint(
            prior_knowledge_snapshot=packet,
            proof_records=proof_records,
        )
        return packet, materialization


def _evidence_quality(envelope: Mapping[str, Any]) -> int:
    payload = envelope["payload"]
    if envelope["record_kind"] == "knowledge-claim-revision":
        return len(payload["supporting_episode_ids"]) + len(
            payload["contradicting_episode_ids"]
        )
    if envelope["record_kind"] == "prior-idea":
        return 0
    terminal = payload["attempts"][payload["terminal_attempt_revision"]]
    if (
        terminal["comparison_status"] == "comparable"
        and terminal["intervention_structure"] == "isolated_by_ablation"
    ):
        return 3
    if terminal["comparison_status"] == "comparable":
        return 2
    if terminal["execution_status"] == "completed":
        return 1
    return 0


def _round_robin_outcomes(candidates: tuple[_Candidate, ...]) -> tuple[_Candidate, ...]:
    ordered: list[_Candidate] = []
    for compatibility in (
        TransferCompatibility.EXACT_CONTEXT,
        TransferCompatibility.ANALOGICAL,
    ):
        buckets = {
            outcome: tuple(
                candidate
                for candidate in candidates
                if candidate.compatibility is compatibility
                and candidate.outcome == outcome
            )
            for outcome in _OUTCOME_ORDER
        }
        positions = {outcome: 0 for outcome in _OUTCOME_ORDER}
        while any(
            positions[outcome] < len(buckets[outcome]) for outcome in _OUTCOME_ORDER
        ):
            for outcome in _OUTCOME_ORDER:
                position = positions[outcome]
                if position < len(buckets[outcome]):
                    ordered.append(buckets[outcome][position])
                    positions[outcome] += 1
    if len(ordered) != len(candidates):
        raise CrossRunRetrievalError("candidate outcome partition is incomplete")
    return tuple(ordered)


def _proof_closed_record_bytes(
    materialization: PriorKnowledgeAccessMaterialization,
) -> int:
    records = (
        *materialization.prior_knowledge_snapshot.selected_records,
        *materialization.proof_records,
    )
    ordered = tuple(sorted(records, key=lambda record: record["record_id"]))
    return len(
        canonical_json_bytes(
            {
                "records": ordered,
                "selection_metadata": (
                    materialization.prior_knowledge_snapshot.selection_metadata
                ),
            }
        )
    )
