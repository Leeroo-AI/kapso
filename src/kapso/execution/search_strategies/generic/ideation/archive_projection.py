"""Pure canonical projections for transactional idea-archive persistence."""

from dataclasses import replace
from datetime import datetime
from typing import Iterable

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    normalize_utc_timestamp,
    parse_json_bytes,
)
from kapso.execution.search_strategies.generic.ideation.archive import (
    IDEA_ARCHIVE_SCHEMA,
    ArchiveCorruptionError,
    ArchiveIdentityConflictError,
    ArchiveLifecycleError,
    ArchiveMissingReferenceError,
    IdeaArchiveState,
    _claim_is_compatible_descendant,
    _gap_is_compatible_descendant,
    _replace_record,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    BatchStatus,
    EvaluationGap,
    EvidenceClaim,
    EvidenceStatus,
    GapState,
    IdeaOutcome,
    IdeaRecord,
    IdeaStatus,
    ImplementationStatus,
    require_batch_transition,
    require_gap_transition,
    require_idea_transition,
)


def encode_archive_state(state: IdeaArchiveState) -> bytes:
    """Encode one exact archive state without whitespace or a trailing newline."""

    if type(state) is not IdeaArchiveState:
        raise ArchiveCorruptionError(
            "canonical archive encoding requires one exact archive state"
        )
    return canonical_json_bytes(state.to_dict())


def decode_archive_state(payload: bytes) -> IdeaArchiveState:
    """Decode only the canonical byte representation of an archive state."""

    if type(payload) is not bytes:
        raise ArchiveCorruptionError("canonical archive payload must be bytes")
    parsed = parse_json_bytes(payload)
    state = IdeaArchiveState.from_dict(parsed)
    if payload != encode_archive_state(state):
        raise ArchiveCorruptionError("idea archive bytes are not canonical")
    return state


def build_archive_genesis(
    *,
    campaign_id: str,
    created_at: str,
) -> IdeaArchiveState:
    """Build an empty archive from one caller-supplied normalized instant."""

    normalized_created_at = normalize_utc_timestamp(
        created_at,
        "idea archive genesis created_at",
    )
    return IdeaArchiveState(
        schema=IDEA_ARCHIVE_SCHEMA,
        campaign_id=campaign_id,
        revision=0,
        created_at=normalized_created_at,
        updated_at=normalized_created_at,
        batches=(),
        ideas=(),
        claims=(),
        gaps=(),
    )


def _find_idea(state: IdeaArchiveState, idea_id: str) -> IdeaRecord:
    matching = tuple(idea for idea in state.ideas if idea.idea_id == idea_id)
    if not matching:
        raise ArchiveMissingReferenceError(f"unknown idea: {idea_id}")
    return matching[0]


def _find_batch(state: IdeaArchiveState, batch_id: str | None):
    matching = tuple(batch for batch in state.batches if batch.batch_id == batch_id)
    if not matching:
        raise ArchiveMissingReferenceError(f"unknown idea batch: {batch_id}")
    return matching[0]


def _find_claim(state: IdeaArchiveState, claim_id: str) -> EvidenceClaim:
    matching = tuple(claim for claim in state.claims if claim.claim_id == claim_id)
    if not matching:
        raise ArchiveMissingReferenceError(f"unknown evidence claim: {claim_id}")
    return matching[0]


def _find_gap(state: IdeaArchiveState, gap_id: str) -> EvaluationGap:
    matching = tuple(gap for gap in state.gaps if gap.gap_id == gap_id)
    if not matching:
        raise ArchiveMissingReferenceError(f"unknown evaluation gap: {gap_id}")
    return matching[0]


def _normalized_update_time(updated_at: str) -> str:
    return normalize_utc_timestamp(
        updated_at,
        "idea archive outcome updated_at",
    )


def _require_forward_update_time(
    state: IdeaArchiveState,
    normalized_updated_at: str,
) -> None:
    if datetime.fromisoformat(normalized_updated_at) < datetime.fromisoformat(
        state.updated_at
    ):
        raise ArchiveIdentityConflictError(
            "idea archive outcome update time precedes its frontier"
        )


def project_outcome(
    state: IdeaArchiveState,
    idea_id: str,
    outcome: IdeaOutcome,
    *,
    claim_updates: Iterable[EvidenceClaim] = (),
    gap_updates: Iterable[EvaluationGap] = (),
    updated_at: str,
) -> IdeaArchiveState:
    """Project one finalized idea outcome without filesystem or clock access."""

    if type(state) is not IdeaArchiveState:
        raise ArchiveCorruptionError(
            "outcome projection requires one exact archive state"
        )
    if type(outcome) is not IdeaOutcome:
        raise ArchiveLifecycleError("outcome projection requires one exact outcome")
    normalized_updated_at = _normalized_update_time(updated_at)
    idea = _find_idea(state, idea_id)
    claim_changes = tuple(claim_updates)
    gap_changes = tuple(gap_updates)
    if any(type(claim) is not EvidenceClaim for claim in claim_changes):
        raise ArchiveLifecycleError("outcome claim updates must be typed claims")
    if any(type(gap) is not EvaluationGap for gap in gap_changes):
        raise ArchiveLifecycleError("outcome gap updates must be typed gaps")
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
            _find_claim(state, claim.claim_id) for claim in claim_changes
        )
        persisted_gap_changes = tuple(
            _find_gap(state, gap.gap_id) for gap in gap_changes
        )
        if (
            idea.outcome == outcome
            and all(
                _claim_is_compatible_descendant(original, current)
                for original, current in zip(
                    claim_changes,
                    persisted_claim_changes,
                    strict=True,
                )
            )
            and all(
                _gap_is_compatible_descendant(original, current)
                for original, current in zip(
                    gap_changes,
                    persisted_gap_changes,
                    strict=True,
                )
            )
        ):
            return state
        raise ArchiveIdentityConflictError("idea outcome already differs")
    _require_forward_update_time(state, normalized_updated_at)
    if idea.status != IdeaStatus.IMPLEMENTING or idea.experiment_node_id is None:
        raise ArchiveLifecycleError("outcome requires an implementing idea")
    batch = _find_batch(state, idea.selected_in_batch_id)
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
            raise ArchiveIdentityConflictError("claim update lacks outcome provenance")
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
                or current.status not in {EvidenceStatus.INSUFFICIENT, update.status}
                or not set(current.source_refs).issubset(update.source_refs)
                or not set(current.affected_idea_ids).issubset(update.affected_idea_ids)
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
                raise ArchiveIdentityConflictError("new gap lacks outcome provenance")
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
        updated_at=normalized_updated_at,
    )
    return replace(
        state,
        revision=state.revision + 1,
        updated_at=normalized_updated_at,
        batches=_replace_record(state.batches, "batch_id", next_batch),
        ideas=_replace_record(state.ideas, "idea_id", next_idea),
        claims=next_claims,
        gaps=next_gaps,
    )


__all__ = [
    "build_archive_genesis",
    "decode_archive_state",
    "encode_archive_state",
    "project_outcome",
]
