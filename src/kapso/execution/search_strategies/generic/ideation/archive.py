"""Strict, atomic persistence for evidence-directed ideation."""

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from kapso.execution.search_strategies.generic.ideation.types import (
    AnalyzedCandidate,
    BatchStatus,
    CandidateDispositionKind,
    CodingAgentCallResult,
    EmbeddingTelemetry,
    EvaluationGap,
    EvidenceClaim,
    EvidenceStatus,
    GapState,
    IdeaBatch,
    IdeaOutcome,
    IdeaRecord,
    IdeaStatus,
    ImplementationStatus,
    ResurfacedIdea,
    SelectionDecision,
    require_batch_transition,
    require_gap_transition,
    require_idea_transition,
    utc_now,
)

IDEA_ARCHIVE_SCHEMA = "kapso.ideation_archive.v3"


class IdeaArchiveError(RuntimeError):
    """Base error for idea-archive integrity failures."""


class ArchiveCorruptionError(IdeaArchiveError):
    """Persisted archive content violates the current strict schema."""


class ArchiveRevisionConflictError(IdeaArchiveError):
    """A mutation was based on a stale archive revision."""


class ArchiveIdentityConflictError(IdeaArchiveError):
    """An existing identifier was replayed with different content."""


class ArchiveLifecycleError(IdeaArchiveError):
    """A requested mutation violates the ideation lifecycle."""


class ArchiveMissingReferenceError(IdeaArchiveError):
    """A mutation references an absent archive record."""


class ArchiveLinkConflictError(IdeaArchiveError):
    """An idea-to-experiment link conflicts with an existing link."""


def _require_exact_keys(data: Any, expected: Iterable[str], name: str) -> None:
    if not isinstance(data, dict):
        raise ArchiveCorruptionError(f"{name} must be an object")
    expected_keys = set(expected)
    actual_keys = set(data)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        raise ArchiveCorruptionError(
            f"{name} has incompatible fields; missing={missing}; unknown={unknown}"
        )


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveCorruptionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ArchiveCorruptionError(f"non-finite JSON number: {value}")


def _replace_record(
    records: Tuple[Any, ...],
    identifier_name: str,
    replacement: Any,
) -> Tuple[Any, ...]:
    return tuple(
        (
            replacement
            if getattr(record, identifier_name) == getattr(replacement, identifier_name)
            else record
        )
        for record in records
    )


def _claim_is_compatible_descendant(
    original: EvidenceClaim,
    current: EvidenceClaim,
) -> bool:
    status_descendants = {
        EvidenceStatus.INSUFFICIENT: {
            EvidenceStatus.INSUFFICIENT,
            EvidenceStatus.SUPPORTED,
            EvidenceStatus.CONTRADICTED,
        },
        EvidenceStatus.SUPPORTED: {EvidenceStatus.SUPPORTED},
        EvidenceStatus.CONTRADICTED: {EvidenceStatus.CONTRADICTED},
    }
    return bool(
        original.claim_id == current.claim_id
        and original.statement == current.statement
        and original.kind == current.kind
        and current.status in status_descendants[original.status]
        and set(original.source_refs).issubset(current.source_refs)
        and set(original.affected_idea_ids).issubset(current.affected_idea_ids)
        and set(original.affected_experiment_node_ids).issubset(
            current.affected_experiment_node_ids
        )
        and datetime.fromisoformat(current.updated_at)
        >= datetime.fromisoformat(original.updated_at)
    )


def _gap_is_compatible_descendant(
    original: EvaluationGap,
    current: EvaluationGap,
) -> bool:
    state_descendants = {
        GapState.OPEN: {GapState.OPEN, GapState.INCONCLUSIVE, GapState.CLOSED},
        GapState.INCONCLUSIVE: {GapState.INCONCLUSIVE, GapState.CLOSED},
        GapState.CLOSED: {GapState.CLOSED},
    }
    if original.last_considered_at is not None and (
        current.last_considered_at is None
        or datetime.fromisoformat(current.last_considered_at)
        < datetime.fromisoformat(original.last_considered_at)
    ):
        return False
    if original.state == current.state and original.state != GapState.OPEN:
        same_resolution = (
            original.resolution_idea_id == current.resolution_idea_id
            and original.resolution_experiment_node_id
            == current.resolution_experiment_node_id
            and original.closure_reason == current.closure_reason
        )
    else:
        same_resolution = True
    return bool(
        original.gap_id == current.gap_id
        and original.axis == current.axis
        and original.description == current.description
        and original.impact == current.impact
        and original.uncertainty == current.uncertainty
        and original.estimated_cost == current.estimated_cost
        and original.opened_at == current.opened_at
        and set(original.evidence_refs).issubset(current.evidence_refs)
        and original.deferral_count <= current.deferral_count
        and current.state in state_descendants[original.state]
        and same_resolution
    )


@dataclass(frozen=True)
class IdeaArchiveState:
    schema: str
    campaign_id: str
    revision: int
    created_at: str
    updated_at: str
    batches: Tuple[IdeaBatch, ...]
    ideas: Tuple[IdeaRecord, ...]
    claims: Tuple[EvidenceClaim, ...]
    gaps: Tuple[EvaluationGap, ...]

    def __post_init__(self) -> None:
        if self.schema != IDEA_ARCHIVE_SCHEMA:
            raise ArchiveCorruptionError(
                f"unsupported idea archive schema: {self.schema!r}"
            )
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise ArchiveCorruptionError("archive campaign id must be non-empty")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ArchiveCorruptionError("archive revision must be non-negative")
        created = datetime.fromisoformat(self.created_at)
        updated = datetime.fromisoformat(self.updated_at)
        if created.utcoffset() is None or updated.utcoffset() is None:
            raise ArchiveCorruptionError("archive timestamps must include UTC offsets")
        if updated < created:
            raise ArchiveCorruptionError("archive update cannot precede creation")
        for records, record_type, name in (
            (self.batches, IdeaBatch, "batches"),
            (self.ideas, IdeaRecord, "ideas"),
            (self.claims, EvidenceClaim, "claims"),
            (self.gaps, EvaluationGap, "gaps"),
        ):
            if not isinstance(records, (list, tuple)) or not all(
                isinstance(record, record_type) for record in records
            ):
                raise ArchiveCorruptionError(f"archive {name} are invalid")
            object.__setattr__(self, name, tuple(records))
        self._validate_references()

    def _validate_references(self) -> None:
        batch_by_id = {batch.batch_id: batch for batch in self.batches}
        idea_by_id = {idea.idea_id: idea for idea in self.ideas}
        claim_by_id = {claim.claim_id: claim for claim in self.claims}
        gap_by_id = {gap.gap_id: gap for gap in self.gaps}
        for records, mapping, name in (
            (self.batches, batch_by_id, "batch"),
            (self.ideas, idea_by_id, "idea"),
            (self.claims, claim_by_id, "claim"),
            (self.gaps, gap_by_id, "gap"),
        ):
            if len(records) != len(mapping):
                raise ArchiveCorruptionError(f"duplicate archive {name} identifier")
        for batch in self.batches:
            if batch.campaign_id != self.campaign_id:
                raise ArchiveCorruptionError("batch campaign does not match archive")
            for idea_id in batch.generated_idea_ids + batch.considered_idea_ids:
                if idea_id not in idea_by_id:
                    raise ArchiveCorruptionError(
                        f"batch references missing idea: {idea_id}"
                    )
            for idea_id in batch.generated_idea_ids:
                if idea_by_id[idea_id].origin_batch_id != batch.batch_id:
                    raise ArchiveCorruptionError(
                        "generated idea does not point back to its origin batch"
                    )
        linked_nodes = set()
        for idea in self.ideas:
            origin = batch_by_id.get(idea.origin_batch_id)
            if origin is None or idea.idea_id not in origin.generated_idea_ids:
                raise ArchiveCorruptionError(
                    f"idea has invalid origin batch: {idea.idea_id}"
                )
            for parent_id in idea.parent_idea_ids:
                if parent_id not in idea_by_id:
                    raise ArchiveCorruptionError(
                        f"idea references missing parent idea: {parent_id}"
                    )
            if idea.exact_duplicate_of is not None and (
                idea.exact_duplicate_of not in idea_by_id
            ):
                raise ArchiveCorruptionError(
                    f"idea references missing duplicate: {idea.exact_duplicate_of}"
                )
            for claim_id in idea.claim_ids:
                if claim_id not in claim_by_id and idea.status not in {
                    IdeaStatus.GENERATED,
                    IdeaStatus.INVALID,
                    IdeaStatus.ABANDONED,
                }:
                    raise ArchiveCorruptionError(
                        f"idea references missing claim: {claim_id}"
                    )
            for gap_id in idea.target_gap_ids:
                if gap_id not in gap_by_id:
                    raise ArchiveCorruptionError(
                        f"idea references missing gap: {gap_id}"
                    )
            if idea.selected_in_batch_id is not None:
                selection_batch = batch_by_id.get(idea.selected_in_batch_id)
                if (
                    selection_batch is None
                    or selection_batch.selection is None
                    or selection_batch.selection.selected_idea_id != idea.idea_id
                ):
                    raise ArchiveCorruptionError(
                        "selected idea and selection batch are not reciprocal"
                    )
            if idea.experiment_node_id is not None:
                if idea.experiment_node_id in linked_nodes:
                    raise ArchiveCorruptionError(
                        "an experiment node is linked to more than one idea"
                    )
                linked_nodes.add(idea.experiment_node_id)
        for claim in self.claims:
            if not set(claim.affected_idea_ids).issubset(idea_by_id):
                raise ArchiveCorruptionError("claim references a missing idea")
        for gap in self.gaps:
            if gap.resolution_idea_id is not None:
                idea = idea_by_id.get(gap.resolution_idea_id)
                if (
                    idea is None
                    or idea.outcome is None
                    or idea.experiment_node_id != gap.resolution_experiment_node_id
                ):
                    raise ArchiveCorruptionError(
                        "gap resolution does not reference a completed outcome"
                    )
        self._validate_acyclic_idea_links(idea_by_id)

    @staticmethod
    def _validate_acyclic_idea_links(idea_by_id: Dict[str, IdeaRecord]) -> None:
        for root_id in idea_by_id:
            pending = list(idea_by_id[root_id].parent_idea_ids)
            visited = set()
            while pending:
                parent_id = pending.pop()
                if parent_id == root_id:
                    raise ArchiveCorruptionError("idea parent links contain a cycle")
                if parent_id not in visited:
                    visited.add(parent_id)
                    pending.extend(idea_by_id[parent_id].parent_idea_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "batches": [batch.to_dict() for batch in self.batches],
            "ideas": [idea.to_dict() for idea in self.ideas],
            "claims": [claim.to_dict() for claim in self.claims],
            "gaps": [gap.to_dict() for gap in self.gaps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaArchiveState":
        _require_exact_keys(
            data,
            {
                "schema",
                "campaign_id",
                "revision",
                "created_at",
                "updated_at",
                "batches",
                "ideas",
                "claims",
                "gaps",
            },
            "idea archive",
        )
        return cls(
            schema=data["schema"],
            campaign_id=data["campaign_id"],
            revision=data["revision"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            batches=tuple(IdeaBatch.from_dict(batch) for batch in data["batches"]),
            ideas=tuple(IdeaRecord.from_dict(idea) for idea in data["ideas"]),
            claims=tuple(EvidenceClaim.from_dict(claim) for claim in data["claims"]),
            gaps=tuple(EvaluationGap.from_dict(gap) for gap in data["gaps"]),
        )


class IdeaArchive:
    """Single-writer campaign archive with optimistic revision checks."""

    def __init__(self, path: Path, campaign_id: str):
        self.path = Path(path)
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise ValueError("campaign_id must be a non-empty string")
        self.campaign_id = campaign_id
        if self.path.exists():
            self._state = self._read_state()
            if self._state.campaign_id != campaign_id:
                raise ArchiveCorruptionError(
                    "archive campaign identity does not match requested campaign"
                )
        else:
            now = utc_now()
            self._state = IdeaArchiveState(
                schema=IDEA_ARCHIVE_SCHEMA,
                campaign_id=campaign_id,
                revision=0,
                created_at=now,
                updated_at=now,
                batches=(),
                ideas=(),
                claims=(),
                gaps=(),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write_atomic(self._state)

    @property
    def revision(self) -> int:
        return self._refresh().revision

    @property
    def state(self) -> IdeaArchiveState:
        return self._refresh()

    def _read_state(self) -> IdeaArchiveState:
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            raise ArchiveCorruptionError("idea archive is empty")
        data = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
        state = IdeaArchiveState.from_dict(data)
        if state.campaign_id != self.campaign_id:
            raise ArchiveCorruptionError("archive campaign identity changed")
        return state

    def _refresh(self) -> IdeaArchiveState:
        if self.path.exists():
            self._state = self._read_state()
        elif self._state.revision != 0:
            raise ArchiveCorruptionError("persisted idea archive disappeared")
        return self._state

    @staticmethod
    def _require_revision(state: IdeaArchiveState, expected_revision: int) -> None:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected revision must be a non-negative integer")
        if state.revision != expected_revision:
            raise ArchiveRevisionConflictError(
                f"expected revision {expected_revision}, found {state.revision}"
            )

    def _commit(
        self,
        proposed: IdeaArchiveState,
        expected_revision: int,
    ) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            current = self._read_state() if self.path.exists() else self._state
            self._require_revision(current, expected_revision)
            committed = replace(
                proposed,
                revision=expected_revision + 1,
                updated_at=utc_now(),
            )
            self._write_atomic(committed)
            self._state = committed
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return self._state.revision

    def _write_atomic(self, state: IdeaArchiveState) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                state.to_dict(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, self.path)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        os.fsync(directory_fd)
        os.close(directory_fd)

    @staticmethod
    def _batch_identity(batch: IdeaBatch) -> Tuple[Any, ...]:
        return (
            batch.batch_id,
            batch.campaign_id,
            batch.iteration_index,
            batch.context_hash,
            batch.planning_archive_revision,
            batch.problem_statement,
            batch.evidence_snapshot,
            batch.capacity,
            batch.directive,
            batch.resolved_parents,
            batch.created_at,
        )

    @staticmethod
    def _idea_generation_identity(idea: IdeaRecord) -> Tuple[Any, ...]:
        return (
            idea.idea_id,
            idea.origin_batch_id,
            idea.proposal,
            idea.operator,
            idea.descriptor,
            idea.parent_plan,
            idea.resolved_parent,
            idea.assumptions,
            idea.evidence_refs,
            idea.directive_rationale,
            idea.evaluation_method,
            idea.resource_request,
            idea.created_at,
            idea.parent_idea_ids,
            idea.parent_experiment_node_ids,
            idea.target_gap_ids,
            idea.claim_ids,
            idea.resolves_claim_ids,
            idea.expected_observations,
            idea.predicted_gain,
            idea.predicted_cost,
            idea.confidence,
            idea.claimed_nearest_idea_id,
            idea.claimed_nearest_experiment_node_id,
            idea.generation_artifacts,
        )

    @staticmethod
    def _find_batch(state: IdeaArchiveState, batch_id: str) -> IdeaBatch:
        matching = tuple(batch for batch in state.batches if batch.batch_id == batch_id)
        if not matching:
            raise ArchiveMissingReferenceError(f"unknown idea batch: {batch_id}")
        return matching[0]

    @staticmethod
    def _find_idea(state: IdeaArchiveState, idea_id: str) -> IdeaRecord:
        matching = tuple(idea for idea in state.ideas if idea.idea_id == idea_id)
        if not matching:
            raise ArchiveMissingReferenceError(f"unknown idea: {idea_id}")
        return matching[0]

    @staticmethod
    def _find_claim(state: IdeaArchiveState, claim_id: str) -> EvidenceClaim:
        matching = tuple(claim for claim in state.claims if claim.claim_id == claim_id)
        if not matching:
            raise ArchiveMissingReferenceError(f"unknown evidence claim: {claim_id}")
        return matching[0]

    @staticmethod
    def _find_gap(state: IdeaArchiveState, gap_id: str) -> EvaluationGap:
        matching = tuple(gap for gap in state.gaps if gap.gap_id == gap_id)
        if not matching:
            raise ArchiveMissingReferenceError(f"unknown evaluation gap: {gap_id}")
        return matching[0]

    def create_batch(self, batch: IdeaBatch, *, expected_revision: int) -> IdeaBatch:
        state = self._refresh()
        existing = tuple(
            item for item in state.batches if item.batch_id == batch.batch_id
        )
        if existing:
            if self._batch_identity(existing[0]) == self._batch_identity(batch):
                return existing[0]
            raise ArchiveIdentityConflictError(
                f"batch id already has different content: {batch.batch_id}"
            )
        self._require_revision(state, expected_revision)
        if batch.status != BatchStatus.PLANNED:
            raise ArchiveLifecycleError("new batches must be planned")
        if batch.campaign_id != self.campaign_id:
            raise ArchiveIdentityConflictError("batch campaign does not match archive")
        proposed = replace(state, batches=state.batches + (batch,))
        self._commit(proposed, expected_revision)
        return batch

    def add_ideas(
        self,
        batch_id: str,
        ideas: Iterable[IdeaRecord],
        *,
        generation_calls: Iterable[CodingAgentCallResult],
        resurfaced_ideas: Iterable[ResurfacedIdea] = (),
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        batch = self._find_batch(state, batch_id)
        generated = tuple(ideas)
        calls = tuple(generation_calls)
        resurfaced = tuple(resurfaced_ideas)
        if len(calls) != len(generated) or not all(
            isinstance(call, CodingAgentCallResult) for call in calls
        ):
            raise ValueError("each generated idea requires one coding-agent call")
        if not all(isinstance(item, ResurfacedIdea) for item in resurfaced):
            raise ValueError("resurfaced ideas must be typed records")
        resurfaced_ids = tuple(item.idea_id for item in resurfaced)
        generated_ids = tuple(idea.idea_id for idea in generated)
        considered_ids = generated_ids + resurfaced_ids
        if len(set(considered_ids)) != len(considered_ids):
            raise ArchiveIdentityConflictError("considered idea ids must be unique")
        idea_by_id = {idea.idea_id: idea for idea in state.ideas}
        if (
            batch.generated_idea_ids == generated_ids
            and batch.generation_calls == calls
            and batch.resurfaced_ideas == resurfaced
            and batch.considered_idea_ids == considered_ids
            and all(
                idea_id in idea_by_id
                and self._idea_generation_identity(idea_by_id[idea_id])
                == self._idea_generation_identity(idea)
                for idea_id, idea in zip(generated_ids, generated)
            )
        ):
            return state.revision
        self._require_revision(state, expected_revision)
        if batch.status != BatchStatus.PLANNED:
            raise ArchiveLifecycleError("ideas can only be added to a planned batch")
        if not considered_ids:
            raise ArchiveLifecycleError("a generated batch requires candidates")
        for idea in generated:
            if idea.origin_batch_id != batch_id or idea.status != IdeaStatus.GENERATED:
                raise ArchiveLifecycleError(
                    "new ideas must be generated in the target batch"
                )
            existing = idea_by_id.get(idea.idea_id)
            if existing is not None:
                raise ArchiveIdentityConflictError(
                    f"idea id already exists: {idea.idea_id}"
                )
        for idea_id in resurfaced_ids:
            idea = self._find_idea(state, idea_id)
            if idea.status != IdeaStatus.DEFERRED:
                raise ArchiveLifecycleError(
                    "only deferred ideas may be resurfaced directly"
                )
        require_batch_transition(batch.status, BatchStatus.GENERATED)
        next_batch = replace(
            batch,
            status=BatchStatus.GENERATED,
            generated_idea_ids=generated_ids,
            generation_calls=calls,
            resurfaced_ideas=resurfaced,
            considered_idea_ids=considered_ids,
            updated_at=utc_now(),
        )
        proposed = replace(
            state,
            batches=_replace_record(state.batches, "batch_id", next_batch),
            ideas=state.ideas + generated,
        )
        return self._commit(proposed, expected_revision)

    def record_analyses(
        self,
        batch_id: str,
        analyzed_candidates: Iterable[AnalyzedCandidate],
        *,
        embedding_telemetry: Optional[EmbeddingTelemetry] = None,
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        batch = self._find_batch(state, batch_id)
        analyzed = tuple(analyzed_candidates)
        if not analyzed or not all(
            isinstance(candidate, AnalyzedCandidate) for candidate in analyzed
        ):
            raise ValueError("candidate analyses must be complete typed records")
        if embedding_telemetry is not None and not isinstance(
            embedding_telemetry, EmbeddingTelemetry
        ):
            raise ValueError("analysis embedding telemetry is invalid")
        analysis_ids = tuple(candidate.analysis.idea_id for candidate in analyzed)
        if analysis_ids != batch.considered_idea_ids:
            raise ArchiveLifecycleError(
                "analysis pool must exactly match considered-idea order"
            )
        if batch.analyses:
            idea_by_id = {idea.idea_id: idea for idea in state.ideas}
            if (
                batch.analyses == tuple(candidate.analysis for candidate in analyzed)
                and batch.embedding_telemetry == embedding_telemetry
                and all(
                    idea_by_id[candidate.analysis.idea_id].embedding
                    == candidate.embedding
                    and idea_by_id[
                        candidate.analysis.idea_id
                    ].nearest_experiment_node_ids
                    == candidate.nearest_experiment_node_ids
                    and idea_by_id[candidate.analysis.idea_id].similarity_flags
                    == candidate.similarity_flags
                    for candidate in analyzed
                )
            ):
                return state.revision
            raise ArchiveIdentityConflictError("batch analysis pool already differs")
        self._require_revision(state, expected_revision)
        if batch.status != BatchStatus.GENERATED:
            raise ArchiveLifecycleError("analysis requires a generated batch")
        next_ideas = state.ideas
        for candidate in analyzed:
            analysis = candidate.analysis
            idea = self._find_idea(state, analysis.idea_id)
            if analysis.exact_duplicate_of is not None:
                self._find_idea(state, analysis.exact_duplicate_of)
            next_idea = replace(
                idea,
                exact_duplicate_of=analysis.exact_duplicate_of,
                embedding=candidate.embedding,
                nearest_experiment_node_ids=candidate.nearest_experiment_node_ids,
                similarity_flags=candidate.similarity_flags,
            )
            if not analysis.eligible:
                require_idea_transition(idea.status, IdeaStatus.INVALID)
                reason_parts = analysis.hard_failures or (
                    "candidate deemed ineligible",
                )
                next_idea = replace(
                    next_idea,
                    status=IdeaStatus.INVALID,
                    rejection_reason="; ".join(reason_parts),
                )
            next_ideas = _replace_record(next_ideas, "idea_id", next_idea)
        require_batch_transition(batch.status, BatchStatus.ANALYZED)
        next_batch = replace(
            batch,
            analyses=tuple(candidate.analysis for candidate in analyzed),
            embedding_telemetry=embedding_telemetry,
            status=BatchStatus.ANALYZED,
            updated_at=utc_now(),
        )
        proposed = replace(
            state,
            batches=_replace_record(state.batches, "batch_id", next_batch),
            ideas=next_ideas,
        )
        return self._commit(proposed, expected_revision)

    def add_repair_idea(
        self,
        batch_id: str,
        idea: IdeaRecord,
        *,
        generation_call: CodingAgentCallResult,
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        batch = self._find_batch(state, batch_id)
        if not isinstance(generation_call, CodingAgentCallResult):
            raise ValueError("repair idea requires a coding-agent call")
        existing = tuple(item for item in state.ideas if item.idea_id == idea.idea_id)
        if existing:
            if (
                len(batch.generated_idea_ids) == batch.directive.candidate_quota + 1
                and batch.generated_idea_ids[-1] == idea.idea_id
                and batch.generation_calls[-1] == generation_call
                and self._idea_generation_identity(existing[0])
                == self._idea_generation_identity(idea)
            ):
                return state.revision
            raise ArchiveIdentityConflictError(
                f"repair idea id already has different content: {idea.idea_id}"
            )
        self._require_revision(state, expected_revision)
        if batch.status != BatchStatus.GENERATED or batch.analyses:
            raise ArchiveLifecycleError(
                "repair candidates require a generated, unanalyzed batch"
            )
        if batch.directive.repair_quota != 1:
            raise ArchiveLifecycleError("batch does not authorize diversity repair")
        if len(batch.generated_idea_ids) != batch.directive.candidate_quota:
            raise ArchiveLifecycleError("batch already consumed its repair quota")
        if idea.origin_batch_id != batch_id or idea.status != IdeaStatus.GENERATED:
            raise ArchiveLifecycleError(
                "repair idea must be generated in the target batch"
            )
        next_batch = replace(
            batch,
            generated_idea_ids=batch.generated_idea_ids + (idea.idea_id,),
            generation_calls=batch.generation_calls + (generation_call,),
            considered_idea_ids=batch.considered_idea_ids + (idea.idea_id,),
            updated_at=utc_now(),
        )
        proposed = replace(
            state,
            batches=_replace_record(state.batches, "batch_id", next_batch),
            ideas=state.ideas + (idea,),
        )
        return self._commit(proposed, expected_revision)

    def record_selection(
        self,
        batch_id: str,
        decision: SelectionDecision,
        *,
        selection_call: CodingAgentCallResult,
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        batch = self._find_batch(state, batch_id)
        if not isinstance(selection_call, CodingAgentCallResult):
            raise ValueError("selection requires a coding-agent call")
        if batch.selection is not None:
            if batch.selection == decision and batch.selection_call == selection_call:
                return state.revision
            raise ArchiveIdentityConflictError("batch selection already differs")
        self._require_revision(state, expected_revision)
        if batch.status != BatchStatus.ANALYZED:
            raise ArchiveLifecycleError("selection requires complete analysis")
        selection_recorded_at = utc_now()
        dispositions = {
            disposition.idea_id: disposition for disposition in decision.dispositions
        }
        next_ideas = state.ideas
        for idea_id in batch.considered_idea_ids:
            idea = self._find_idea(state, idea_id)
            disposition = dispositions[idea_id]
            if disposition.disposition == CandidateDispositionKind.SELECTED:
                require_idea_transition(idea.status, IdeaStatus.SELECTED)
                next_idea = replace(
                    idea,
                    status=IdeaStatus.SELECTED,
                    selected_in_batch_id=batch_id,
                    selection_reason=disposition.reason,
                    deferral_reason=None,
                    rejection_reason=None,
                )
            elif disposition.disposition == CandidateDispositionKind.DEFERRED:
                if idea.status == IdeaStatus.GENERATED:
                    require_idea_transition(idea.status, IdeaStatus.DEFERRED)
                next_idea = replace(
                    idea,
                    status=IdeaStatus.DEFERRED,
                    deferral_reason=disposition.reason,
                    rejection_reason=None,
                )
            elif disposition.disposition == CandidateDispositionKind.REJECTED:
                require_idea_transition(idea.status, IdeaStatus.REJECTED)
                next_idea = replace(
                    idea,
                    status=IdeaStatus.REJECTED,
                    rejection_reason=disposition.reason,
                    deferral_reason=None,
                )
            else:
                if idea.status != IdeaStatus.INVALID:
                    raise ArchiveLifecycleError(
                        "invalid disposition requires failed candidate analysis"
                    )
                next_idea = replace(idea, rejection_reason=disposition.reason)
            next_ideas = _replace_record(next_ideas, "idea_id", next_idea)
        next_gaps = state.gaps
        reserved_gap_id = batch.directive.reserved_gap_id
        selected_idea = self._find_idea(state, decision.selected_idea_id)
        if (
            reserved_gap_id is not None
            and reserved_gap_id not in selected_idea.target_gap_ids
        ):
            reserved_gap = self._find_gap(state, reserved_gap_id)
            if reserved_gap.state != GapState.OPEN:
                raise ArchiveLifecycleError("only open reserved gaps may be deferred")
            next_gaps = _replace_record(
                next_gaps,
                "gap_id",
                replace(
                    reserved_gap,
                    deferral_count=reserved_gap.deferral_count + 1,
                    last_considered_at=selection_recorded_at,
                ),
            )
        require_batch_transition(batch.status, BatchStatus.SELECTED)
        next_batch = replace(
            batch,
            selection=decision,
            selection_call=selection_call,
            status=BatchStatus.SELECTED,
            updated_at=selection_recorded_at,
        )
        proposed = replace(
            state,
            batches=_replace_record(state.batches, "batch_id", next_batch),
            ideas=next_ideas,
            gaps=next_gaps,
        )
        return self._commit(proposed, expected_revision)

    def link_experiment(
        self,
        idea_id: str,
        node_id: int,
        selection_batch_id: str,
        *,
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        idea = self._find_idea(state, idea_id)
        batch = self._find_batch(state, selection_batch_id)
        if idea.experiment_node_id is not None:
            if (
                idea.experiment_node_id == node_id
                and idea.selected_in_batch_id == selection_batch_id
            ):
                return state.revision
            raise ArchiveLinkConflictError("idea is already linked to another node")
        self._require_revision(state, expected_revision)
        if (
            idea.status != IdeaStatus.SELECTED
            or idea.selected_in_batch_id != selection_batch_id
            or batch.status != BatchStatus.SELECTED
            or batch.selection is None
            or batch.selection.selected_idea_id != idea_id
        ):
            raise ArchiveLifecycleError("experiment link requires reciprocal selection")
        if any(item.experiment_node_id == node_id for item in state.ideas):
            raise ArchiveLinkConflictError("experiment node is already linked")
        require_idea_transition(idea.status, IdeaStatus.IMPLEMENTING)
        require_batch_transition(batch.status, BatchStatus.BRIDGED)
        next_idea = replace(
            idea,
            status=IdeaStatus.IMPLEMENTING,
            experiment_node_id=node_id,
        )
        next_batch = replace(
            batch,
            status=BatchStatus.BRIDGED,
            updated_at=utc_now(),
        )
        proposed = replace(
            state,
            batches=_replace_record(state.batches, "batch_id", next_batch),
            ideas=_replace_record(state.ideas, "idea_id", next_idea),
        )
        return self._commit(proposed, expected_revision)

    def record_outcome(
        self,
        idea_id: str,
        outcome: IdeaOutcome,
        *,
        claim_updates: Iterable[EvidenceClaim] = (),
        gap_updates: Iterable[EvaluationGap] = (),
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        idea = self._find_idea(state, idea_id)
        claim_changes = tuple(claim_updates)
        gap_changes = tuple(gap_updates)
        claim_change_ids = tuple(claim.claim_id for claim in claim_changes)
        gap_change_ids = tuple(gap.gap_id for gap in gap_changes)
        if len(set(claim_change_ids)) != len(claim_change_ids):
            raise ArchiveLifecycleError("outcome claim updates must be unique")
        if len(set(gap_change_ids)) != len(gap_change_ids):
            raise ArchiveLifecycleError("outcome gap updates must be unique")
        expected_claim_ids = set(outcome.supported_claim_ids) | set(
            outcome.contradicted_claim_ids
        )
        if set(claim_change_ids) != expected_claim_ids:
            raise ArchiveLifecycleError(
                "outcome claim updates must exactly cover classified claims"
            )
        if set(gap_change_ids) != set(outcome.gap_effects):
            raise ArchiveLifecycleError(
                "outcome gap updates must exactly cover gap effects"
            )
        if idea.outcome is not None:
            persisted_claim_changes = tuple(
                self._find_claim(state, claim.claim_id) for claim in claim_changes
            )
            persisted_gap_changes = tuple(
                self._find_gap(state, gap.gap_id) for gap in gap_changes
            )
            if (
                idea.outcome == outcome
                and all(
                    _claim_is_compatible_descendant(original, current)
                    for original, current in zip(
                        claim_changes,
                        persisted_claim_changes,
                    )
                )
                and all(
                    _gap_is_compatible_descendant(original, current)
                    for original, current in zip(
                        gap_changes,
                        persisted_gap_changes,
                    )
                )
            ):
                return state.revision
            raise ArchiveIdentityConflictError("idea outcome already differs")
        self._require_revision(state, expected_revision)
        if idea.status != IdeaStatus.IMPLEMENTING or idea.experiment_node_id is None:
            raise ArchiveLifecycleError("outcome requires an implementing idea")
        batch = self._find_batch(state, idea.selected_in_batch_id)
        if batch.status != BatchStatus.BRIDGED:
            raise ArchiveLifecycleError("outcome requires a bridged batch")
        update_by_claim = {claim.claim_id: claim for claim in claim_changes}
        existing_claims_by_id = {claim.claim_id: claim for claim in state.claims}
        next_claims = state.claims
        for claim_id, update in update_by_claim.items():
            expected_status = (
                EvidenceStatus.SUPPORTED
                if claim_id in outcome.supported_claim_ids
                else EvidenceStatus.CONTRADICTED
            )
            if update.status != expected_status:
                raise ArchiveLifecycleError("claim update conflicts with outcome")
            if idea_id not in update.affected_idea_ids or (
                idea.experiment_node_id not in update.affected_experiment_node_ids
            ):
                raise ArchiveIdentityConflictError(
                    "claim update lacks outcome provenance"
                )
            current = existing_claims_by_id.get(claim_id)
            if current is None:
                if update.affected_idea_ids != (idea_id,) or (
                    update.affected_experiment_node_ids != (idea.experiment_node_id,)
                ):
                    raise ArchiveIdentityConflictError(
                        "new claim must belong to the current outcome"
                    )
                next_claims += (update,)
            else:
                if (
                    current.statement != update.statement
                    or current.kind != update.kind
                    or current.status
                    not in {EvidenceStatus.INSUFFICIENT, update.status}
                    or not set(current.source_refs).issubset(update.source_refs)
                    or not set(current.affected_idea_ids).issubset(
                        update.affected_idea_ids
                    )
                    or not set(current.affected_experiment_node_ids).issubset(
                        update.affected_experiment_node_ids
                    )
                    or datetime.fromisoformat(update.updated_at)
                    <= datetime.fromisoformat(current.updated_at)
                ):
                    raise ArchiveIdentityConflictError("claim update changes identity")
                next_claims = _replace_record(next_claims, "claim_id", update)
        existing_gaps_by_id = {gap.gap_id: gap for gap in state.gaps}
        next_gaps = state.gaps
        for update in gap_changes:
            current = existing_gaps_by_id.get(update.gap_id)
            if current is None:
                if update.state != GapState.OPEN:
                    raise ArchiveLifecycleError("new outcome gaps must be open")
                if (
                    update.resolution_idea_id is not None
                    or update.resolution_experiment_node_id is not None
                    or f"experiment_node:{idea.experiment_node_id}"
                    not in update.evidence_refs
                ):
                    raise ArchiveIdentityConflictError(
                        "new gap lacks outcome provenance"
                    )
                next_gaps += (update,)
                continue
            if update.gap_id not in idea.target_gap_ids:
                raise ArchiveLifecycleError("outcome cannot resolve an untargeted gap")
            require_gap_transition(current.state, update.state)
            if (
                current.axis != update.axis
                or current.description != update.description
                or current.impact != update.impact
                or current.uncertainty != update.uncertainty
                or current.estimated_cost != update.estimated_cost
                or current.opened_at != update.opened_at
                or current.deferral_count != update.deferral_count
                or not set(current.evidence_refs).issubset(update.evidence_refs)
                or update.last_considered_at is None
                or (
                    current.last_considered_at is not None
                    and datetime.fromisoformat(update.last_considered_at)
                    <= datetime.fromisoformat(current.last_considered_at)
                )
                or update.resolution_idea_id != idea_id
                or update.resolution_experiment_node_id != idea.experiment_node_id
            ):
                raise ArchiveIdentityConflictError(
                    "gap update changes identity or lacks outcome provenance"
                )
            next_gaps = _replace_record(next_gaps, "gap_id", update)
        next_status = (
            IdeaStatus.FAILED_TECHNICAL
            if outcome.implementation_status == ImplementationStatus.FAILED_TECHNICAL
            else IdeaStatus.EVALUATED
        )
        require_idea_transition(idea.status, next_status)
        require_batch_transition(batch.status, BatchStatus.COMPLETED)
        next_idea = replace(idea, status=next_status, outcome=outcome)
        next_batch = replace(
            batch,
            status=BatchStatus.COMPLETED,
            updated_at=utc_now(),
        )
        proposed = replace(
            state,
            batches=_replace_record(state.batches, "batch_id", next_batch),
            ideas=_replace_record(state.ideas, "idea_id", next_idea),
            claims=next_claims,
            gaps=next_gaps,
        )
        return self._commit(proposed, expected_revision)

    def abandon_batch(
        self,
        batch_id: str,
        reason: str,
        *,
        expected_revision: int,
    ) -> int:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abandonment reason must be non-empty")
        state = self._refresh()
        batch = self._find_batch(state, batch_id)
        if batch.status == BatchStatus.ABANDONED:
            if batch.abandoned_reason == reason:
                return state.revision
            raise ArchiveIdentityConflictError("batch abandonment reason differs")
        self._require_revision(state, expected_revision)
        if batch.status in {BatchStatus.SELECTED, BatchStatus.BRIDGED}:
            raise ArchiveLifecycleError(
                "selected or bridged batches cannot be abandoned"
            )
        require_batch_transition(batch.status, BatchStatus.ABANDONED)
        next_ideas = state.ideas
        for idea_id in batch.generated_idea_ids:
            idea = self._find_idea(state, idea_id)
            if idea.status in {IdeaStatus.GENERATED, IdeaStatus.DEFERRED}:
                require_idea_transition(idea.status, IdeaStatus.ABANDONED)
                next_ideas = _replace_record(
                    next_ideas,
                    "idea_id",
                    replace(idea, status=IdeaStatus.ABANDONED),
                )
        next_batch = replace(
            batch,
            status=BatchStatus.ABANDONED,
            abandoned_reason=reason,
            updated_at=utc_now(),
        )
        proposed = replace(
            state,
            batches=_replace_record(state.batches, "batch_id", next_batch),
            ideas=next_ideas,
        )
        return self._commit(proposed, expected_revision)

    def record_claims(
        self,
        claims: Iterable[EvidenceClaim],
        *,
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        changes = tuple(claims)
        if not changes:
            return state.revision
        existing_by_id = {claim.claim_id: claim for claim in state.claims}
        if all(
            existing_by_id.get(claim.claim_id) is not None
            and _claim_is_compatible_descendant(
                claim,
                existing_by_id[claim.claim_id],
            )
            for claim in changes
        ):
            return state.revision
        self._require_revision(state, expected_revision)
        next_claims = state.claims
        for claim in changes:
            existing = existing_by_id.get(claim.claim_id)
            if existing is None:
                if not set(claim.affected_idea_ids).issubset(
                    idea.idea_id for idea in state.ideas
                ):
                    raise ArchiveMissingReferenceError("claim references missing idea")
                next_claims += (claim,)
                continue
            if (
                existing.statement != claim.statement
                or existing.kind != claim.kind
                or (
                    existing.status != claim.status
                    and existing.status != EvidenceStatus.INSUFFICIENT
                )
                or not set(existing.source_refs).issubset(claim.source_refs)
                or not set(existing.affected_idea_ids).issubset(claim.affected_idea_ids)
                or not set(existing.affected_experiment_node_ids).issubset(
                    claim.affected_experiment_node_ids
                )
                or datetime.fromisoformat(claim.updated_at)
                <= datetime.fromisoformat(existing.updated_at)
            ):
                raise ArchiveIdentityConflictError(
                    f"claim update conflicts with existing claim: {claim.claim_id}"
                )
            next_claims = _replace_record(next_claims, "claim_id", claim)
        proposed = replace(state, claims=next_claims)
        return self._commit(proposed, expected_revision)

    def add_gaps(
        self,
        gaps: Iterable[EvaluationGap],
        *,
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        additions = tuple(gaps)
        if not additions:
            return state.revision
        existing_by_id = {gap.gap_id: gap for gap in state.gaps}
        if all(
            existing_by_id.get(gap.gap_id) is not None
            and _gap_is_compatible_descendant(
                gap,
                existing_by_id[gap.gap_id],
            )
            for gap in additions
        ):
            return state.revision
        self._require_revision(state, expected_revision)
        for gap in additions:
            if gap.state != GapState.OPEN:
                raise ArchiveLifecycleError("new evaluation gaps must be open")
            if gap.gap_id in existing_by_id:
                raise ArchiveIdentityConflictError(
                    f"gap id already has different content: {gap.gap_id}"
                )
        proposed = replace(state, gaps=state.gaps + additions)
        return self._commit(proposed, expected_revision)

    def defer_gap(
        self,
        gap_id: str,
        considered_at: str,
        *,
        expected_deferral_count: int,
        expected_revision: int,
    ) -> int:
        state = self._refresh()
        gap = self._find_gap(state, gap_id)
        if gap.deferral_count == expected_deferral_count + 1 and (
            gap.last_considered_at == considered_at
        ):
            return state.revision
        self._require_revision(state, expected_revision)
        if gap.state != GapState.OPEN:
            raise ArchiveLifecycleError("only open gaps may accrue deferral debt")
        if gap.deferral_count != expected_deferral_count:
            raise ArchiveRevisionConflictError(
                "gap deferral count changed before update"
            )
        next_gap = replace(
            gap,
            deferral_count=gap.deferral_count + 1,
            last_considered_at=considered_at,
        )
        proposed = replace(
            state,
            gaps=_replace_record(state.gaps, "gap_id", next_gap),
        )
        return self._commit(proposed, expected_revision)

    def get_batch(self, batch_id: str) -> IdeaBatch:
        return self._find_batch(self._refresh(), batch_id)

    def get_idea(self, idea_id: str) -> IdeaRecord:
        return self._find_idea(self._refresh(), idea_id)

    def get_claim(self, claim_id: str) -> EvidenceClaim:
        return self._find_claim(self._refresh(), claim_id)

    def list_recent_ideas(self, limit: int) -> Tuple[IdeaRecord, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("idea limit must be a positive integer")
        ordered = sorted(
            self._refresh().ideas,
            key=lambda idea: (idea.created_at, idea.idea_id),
            reverse=True,
        )
        return tuple(ordered[:limit])

    def searchable_ideas(self) -> Tuple[IdeaRecord, ...]:
        excluded = {IdeaStatus.INVALID, IdeaStatus.ABANDONED}
        return tuple(
            idea for idea in self._refresh().ideas if idea.status not in excluded
        )

    def list_gaps(self, state: Optional[GapState] = None) -> Tuple[EvaluationGap, ...]:
        gaps = self._refresh().gaps
        if state is None:
            return gaps
        if not isinstance(state, GapState):
            raise ValueError("gap state filter is invalid")
        return tuple(gap for gap in gaps if gap.state == state)
