"""Deterministic hard-rule, evidence, descriptor, and similarity analysis."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from kapso.execution.search_strategies.generic.ideation.archive import (
    IdeaArchiveState,
)
from kapso.execution.search_strategies.generic.ideation.embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    canonical_idea_embedding_text,
    embedding_can_be_reused,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CampaignEvidenceSnapshot,
    CandidateAnalysis,
    EmbeddingRecord,
    EmbeddingTelemetry,
    EvidenceStatus,
    IdeaDescriptor,
    IdeaRecord,
    IdeationCapacityView,
    IdeationMode,
    ParentPlanKind,
    SearchDirective,
    SimilarityMatch,
)


@dataclass(frozen=True)
class AnalyzerSettings:
    semantic_similarity_threshold: float
    max_neighbors: int
    minimum_distinct_eligible: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.semantic_similarity_threshold, bool)
            or not isinstance(self.semantic_similarity_threshold, (int, float))
            or not math.isfinite(float(self.semantic_similarity_threshold))
            or not -1.0 <= self.semantic_similarity_threshold <= 1.0
        ):
            raise ValueError("semantic similarity threshold must be between -1 and 1")
        if (
            isinstance(self.max_neighbors, bool)
            or not isinstance(self.max_neighbors, int)
            or self.max_neighbors < 1
        ):
            raise ValueError("maximum semantic neighbors must be positive")
        if (
            isinstance(self.minimum_distinct_eligible, bool)
            or not isinstance(self.minimum_distinct_eligible, int)
            or self.minimum_distinct_eligible < 1
        ):
            raise ValueError("minimum distinct eligible candidates must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalyzerSettings":
        expected = {
            "semantic_similarity_threshold",
            "max_neighbors",
            "minimum_distinct_eligible",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("analyzer settings fields are invalid")
        return cls(**data)


@dataclass(frozen=True)
class AnalyzedCandidate:
    analysis: CandidateAnalysis
    descriptor: IdeaDescriptor
    embedding: EmbeddingRecord | None
    nearest_experiment_node_ids: Tuple[int, ...]
    similarity_flags: Tuple[str, ...]


@dataclass(frozen=True)
class AnalysisPoolResult:
    candidates: Tuple[AnalyzedCandidate, ...]
    embedding_telemetry: EmbeddingTelemetry | None


@dataclass(frozen=True)
class DiversityRepairRequest:
    reason: str
    eligible_descriptor_count: int
    missing_descriptor_targets: Tuple[IdeaDescriptor, ...]

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "reason": self.reason,
            "eligible_descriptor_count": self.eligible_descriptor_count,
            "missing_descriptor_targets": [
                descriptor.to_dict() for descriptor in self.missing_descriptor_targets
            ],
        }


class CandidateAnalyzer:
    """Analyze a complete pool; provider errors deliberately propagate."""

    def __init__(
        self,
        settings: AnalyzerSettings,
        embedding_provider: EmbeddingProvider | None,
    ):
        self.settings = settings
        self.embedding_provider = embedding_provider

    def analyze_pool(
        self,
        *,
        batch_id: str,
        candidates: Sequence[IdeaRecord],
        archive_state: IdeaArchiveState,
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
        capacity: IdeationCapacityView,
    ) -> AnalysisPoolResult:
        pool = tuple(candidates)
        if not pool:
            raise ValueError("candidate analysis pool must not be empty")
        candidate_ids = tuple(candidate.idea_id for candidate in pool)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate analysis pool ids must be unique")
        comparison = self._comparison_pool(pool, archive_state)
        embeddings, telemetry = self._embeddings(comparison)
        results = tuple(
            self._analyze_one(
                batch_id=batch_id,
                candidate=candidate,
                predecessors=tuple(archive_state.ideas) + pool[:index],
                comparison=comparison,
                embeddings=embeddings,
                evidence_snapshot=evidence_snapshot,
                directive=directive,
                capacity=capacity,
            )
            for index, candidate in enumerate(pool)
        )
        return AnalysisPoolResult(candidates=results, embedding_telemetry=telemetry)

    def repair_request(
        self,
        result: AnalysisPoolResult,
        directive: SearchDirective,
    ) -> DiversityRepairRequest | None:
        if directive.repair_quota == 0 or directive.decision.mode in {
            IdeationMode.VERIFY,
            IdeationMode.RECOVER,
        }:
            return None
        analysis_by_id = {
            candidate.analysis.idea_id: candidate.analysis
            for candidate in result.candidates
        }
        eligible_descriptors = {
            candidate.descriptor
            for candidate in result.candidates
            if analysis_by_id[candidate.analysis.idea_id].eligible
        }
        if len(eligible_descriptors) >= self.settings.minimum_distinct_eligible:
            return None
        missing = tuple(
            brief.descriptor_target
            for brief in directive.operator_briefs
            if brief.descriptor_target not in eligible_descriptors
        )
        return DiversityRepairRequest(
            reason="eligible candidates do not meet descriptor diversity minimum",
            eligible_descriptor_count=len(eligible_descriptors),
            missing_descriptor_targets=missing,
        )

    def _comparison_pool(
        self,
        candidates: Tuple[IdeaRecord, ...],
        archive_state: IdeaArchiveState,
    ) -> Tuple[IdeaRecord, ...]:
        ordered = []
        seen = set()
        for idea in candidates + tuple(archive_state.ideas):
            if idea.idea_id not in seen:
                seen.add(idea.idea_id)
                ordered.append(idea)
        return tuple(ordered)

    def _embeddings(
        self,
        ideas: Tuple[IdeaRecord, ...],
    ) -> Tuple[Mapping[str, EmbeddingRecord], EmbeddingTelemetry | None]:
        if self.embedding_provider is None:
            return {}, None
        settings = self.embedding_provider.settings
        records = {}
        missing_ideas = []
        missing_texts = []
        for idea in ideas:
            text = canonical_idea_embedding_text(idea)
            if idea.embedding is not None and embedding_can_be_reused(
                idea.embedding,
                text,
                settings,
            ):
                records[idea.idea_id] = idea.embedding
            else:
                missing_ideas.append(idea)
                missing_texts.append(text)
        if not missing_texts:
            return records, None
        batch = self.embedding_provider.embed(missing_texts)
        if len(batch.records) != len(missing_ideas):
            raise ValueError("embedding provider returned an invalid batch size")
        records.update(
            {idea.idea_id: record for idea, record in zip(missing_ideas, batch.records)}
        )
        return records, batch.telemetry

    def _analyze_one(
        self,
        *,
        batch_id: str,
        candidate: IdeaRecord,
        predecessors: Tuple[IdeaRecord, ...],
        comparison: Tuple[IdeaRecord, ...],
        embeddings: Mapping[str, EmbeddingRecord],
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
        capacity: IdeationCapacityView,
    ) -> AnalyzedCandidate:
        failures = []
        unsupported = []
        flags = []
        experiments_by_node = {
            experiment.node_id: experiment
            for experiment in evidence_snapshot.experiments
        }
        ideas_by_id = {idea.idea_id: idea for idea in comparison}
        claims_by_id = {claim.claim_id: claim for claim in evidence_snapshot.claims}
        gaps_by_id = {gap.gap_id: gap for gap in evidence_snapshot.gaps}
        valid_evidence_refs = self._valid_evidence_refs(evidence_snapshot)

        if candidate.parent_plan.kind not in directive.allowed_parent_plan_kinds:
            failures.append("parent_plan_not_allowed")
        if candidate.origin_batch_id == batch_id:
            matching_briefs = tuple(
                brief
                for brief in directive.operator_briefs
                if brief.operator == candidate.operator
                and brief.parent_plan == candidate.parent_plan
                and (() if brief.target_gap_id is None else (brief.target_gap_id,))
                == candidate.target_gap_ids
            )
            if not matching_briefs:
                failures.append("operator_brief_not_assigned")
            elif candidate.descriptor not in {
                brief.descriptor_target for brief in matching_briefs
            }:
                failures.append("operator_descriptor_mismatch")
            if not candidate.generation_artifacts or not all(
                Path(path).is_absolute() and Path(path).is_file()
                for path in candidate.generation_artifacts
            ):
                failures.append("generation_artifacts_invalid")
        self._validate_parent(
            candidate,
            evidence_snapshot,
            experiments_by_node,
            ideas_by_id,
            failures,
        )
        if any(
            reference not in valid_evidence_refs
            for reference in candidate.evidence_refs
        ):
            failures.append("evidence_reference_unknown")
        if not candidate.evidence_refs:
            failures.append("evidence_reference_required")
        if any(claim_id not in claims_by_id for claim_id in candidate.claim_ids):
            failures.append("claim_reference_unknown")
        for claim_id in candidate.claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim is not None and claim.status == EvidenceStatus.INSUFFICIENT:
                unsupported.append(claim_id)
            if (
                claim is not None
                and claim.status == EvidenceStatus.CONTRADICTED
                and claim_id not in candidate.resolves_claim_ids
            ):
                failures.append(f"contradicted_claim_not_resolved:{claim_id}")
        if any(gap_id not in gaps_by_id for gap_id in candidate.target_gap_ids):
            failures.append("target_gap_unknown")
        if not candidate.expected_observations:
            failures.append("expected_observation_required")
        if candidate.claimed_nearest_idea_id is not None and (
            candidate.claimed_nearest_idea_id not in ideas_by_id
        ):
            failures.append("claimed_nearest_idea_unknown")
        if candidate.claimed_nearest_experiment_node_id is not None and (
            candidate.claimed_nearest_experiment_node_id not in experiments_by_node
        ):
            failures.append("claimed_nearest_experiment_unknown")
        if not capacity.can_start_complete_action:
            failures.append("capacity_cannot_start_complete_action")
        if not capacity.can_run_comparable_evaluation:
            failures.append("capacity_cannot_run_comparable_evaluation")
        if not capacity.preserves_finalization_reserve:
            failures.append("capacity_does_not_preserve_finalization_reserve")
        if capacity.opportunity_probe_required and not (
            capacity.opportunity_probe_admissible
        ):
            failures.append("required_opportunity_probe_not_admissible")

        exact_duplicate = self._exact_duplicate(candidate, predecessors)
        changed_conditions = ()
        if exact_duplicate is not None:
            changed_conditions = self._changed_conditions(candidate, exact_duplicate)
            if not changed_conditions:
                failures.append("exact_duplicate_without_changed_conditions")

        descriptor_matches = tuple(
            idea.idea_id
            for idea in comparison
            if idea.idea_id != candidate.idea_id
            and idea.descriptor == candidate.descriptor
        )
        flags.extend(f"descriptor_match:{idea_id}" for idea_id in descriptor_matches)
        semantic_neighbors = self._semantic_neighbors(
            candidate,
            comparison,
            embeddings,
        )
        flags.extend(
            f"semantic_neighbor:{match.idea_id}" for match in semantic_neighbors
        )
        if (
            candidate.claimed_nearest_idea_id is not None
            and semantic_neighbors
            and candidate.claimed_nearest_idea_id != semantic_neighbors[0].idea_id
        ):
            flags.append("claimed_nearest_idea_mismatch")
        nearest_nodes = tuple(
            dict.fromkeys(
                idea.experiment_node_id
                for match in semantic_neighbors
                for idea in comparison
                if idea.idea_id == match.idea_id and idea.experiment_node_id is not None
            )
        )
        failures_tuple = tuple(dict.fromkeys(failures))
        analysis = CandidateAnalysis(
            idea_id=candidate.idea_id,
            eligible=not failures_tuple,
            hard_failures=failures_tuple,
            unsupported_claims=tuple(dict.fromkeys(unsupported)),
            exact_duplicate_of=(
                None if exact_duplicate is None else exact_duplicate.idea_id
            ),
            exact_duplicate_changed_conditions=changed_conditions,
            semantic_neighbors=semantic_neighbors,
        )
        return AnalyzedCandidate(
            analysis=analysis,
            descriptor=candidate.descriptor,
            embedding=embeddings.get(candidate.idea_id),
            nearest_experiment_node_ids=nearest_nodes,
            similarity_flags=tuple(dict.fromkeys(flags)),
        )

    @staticmethod
    def _valid_evidence_refs(
        snapshot: CampaignEvidenceSnapshot,
    ) -> set[str]:
        references = {snapshot.snapshot_id}
        for claim in snapshot.claims:
            references.add(claim.claim_id)
            references.update(claim.source_refs)
        for gap in snapshot.gaps:
            references.add(gap.gap_id)
            references.update(gap.evidence_refs)
        for experiment in snapshot.experiments:
            references.update(
                {
                    experiment.idea_id,
                    experiment.selection_batch_id,
                    f"experiment_node:{experiment.node_id}",
                }
            )
            if experiment.evaluator_id is not None:
                references.add(experiment.evaluator_id)
        return references

    @staticmethod
    def _validate_parent(
        candidate: IdeaRecord,
        snapshot: CampaignEvidenceSnapshot,
        experiments_by_node: Mapping[int, Any],
        ideas_by_id: Mapping[str, IdeaRecord],
        failures: list[str],
    ) -> None:
        plan = candidate.parent_plan
        if candidate.parent_idea_ids != plan.source_idea_ids:
            failures.append("parent_idea_provenance_mismatch")
        if any(idea_id not in ideas_by_id for idea_id in plan.source_idea_ids):
            failures.append("parent_idea_unknown")
        if any(
            node_id not in experiments_by_node
            for node_id in plan.source_experiment_node_ids
        ):
            failures.append("parent_source_experiment_unknown")
        if (
            plan.kind
            in {
                ParentPlanKind.SPECIFIC_EXPERIMENT,
                ParentPlanKind.RECOVER_BRANCH,
            }
            and candidate.resolved_parent.node_id != plan.experiment_node_id
        ):
            failures.append("resolved_parent_does_not_match_plan")
        if plan.kind == ParentPlanKind.BEST_VALID and (
            snapshot.incumbent_node_id is None
            or candidate.resolved_parent.node_id != snapshot.incumbent_node_id
        ):
            failures.append("resolved_parent_is_not_incumbent")
        expected_nodes = list(plan.source_experiment_node_ids)
        if plan.experiment_node_id is not None:
            expected_nodes.append(plan.experiment_node_id)
        expected_nodes.insert(0, candidate.resolved_parent.node_id)
        if candidate.parent_experiment_node_ids != tuple(dict.fromkeys(expected_nodes)):
            failures.append("parent_experiment_provenance_mismatch")

    @staticmethod
    def _exact_duplicate(
        candidate: IdeaRecord,
        predecessors: Tuple[IdeaRecord, ...],
    ) -> IdeaRecord | None:
        candidate_key = json.dumps(
            {
                "proposal": candidate.proposal,
                "descriptor": candidate.descriptor.to_dict(),
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        for idea in predecessors:
            if idea.idea_id == candidate.idea_id:
                continue
            key = json.dumps(
                {"proposal": idea.proposal, "descriptor": idea.descriptor.to_dict()},
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            if key == candidate_key:
                return idea
        return None

    @staticmethod
    def _changed_conditions(
        candidate: IdeaRecord,
        duplicate: IdeaRecord,
    ) -> Tuple[str, ...]:
        conditions = []
        if candidate.resolved_parent.node_id != duplicate.resolved_parent.node_id:
            conditions.append("resolved_parent_changed")
        if candidate.evaluation_method != duplicate.evaluation_method:
            conditions.append("evaluation_method_changed")
        if set(candidate.evidence_refs) - set(duplicate.evidence_refs):
            conditions.append("new_evidence_available")
        return tuple(conditions)

    def _semantic_neighbors(
        self,
        candidate: IdeaRecord,
        comparison: Tuple[IdeaRecord, ...],
        embeddings: Mapping[str, EmbeddingRecord],
    ) -> Tuple[SimilarityMatch, ...]:
        candidate_embedding = embeddings.get(candidate.idea_id)
        if candidate_embedding is None:
            return ()
        matches = []
        for idea in comparison:
            other = embeddings.get(idea.idea_id)
            if idea.idea_id != candidate.idea_id and other is not None:
                similarity = cosine_similarity(candidate_embedding, other)
                if similarity >= self.settings.semantic_similarity_threshold:
                    matches.append(
                        SimilarityMatch(
                            idea_id=idea.idea_id,
                            similarity=similarity,
                        )
                    )
        return tuple(
            sorted(
                matches,
                key=lambda match: (-match.similarity, match.idea_id),
            )[: self.settings.max_neighbors]
        )
