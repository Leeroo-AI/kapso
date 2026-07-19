"""Mechanical evaluator-authored evidence write-back for completed ideas."""

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

from kapso.execution.search_strategies.generic.ideation.archive import IdeaArchiveState
from kapso.execution.search_strategies.generic.ideation.types import (
    ClaimKind,
    EvaluationGap,
    EvidenceClaim,
    EvidenceStatus,
    GapState,
    IdeaOutcome,
    IdeaRecord,
    content_identifier,
    require_gap_transition,
)

EVALUATOR_EVIDENCE_KEY = "ideation_evidence"


def _require_exact_keys(data: Any, expected: set[str], name: str) -> Dict[str, Any]:
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"{name} fields are invalid")
    return data


def _require_list(data: Any, name: str) -> list[Any]:
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    return data


def _require_strings(data: Any, name: str, *, nonempty: bool) -> Tuple[str, ...]:
    values = _require_list(data, name)
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    if nonempty and not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(values)


def _content_id(prefix: str, content: Dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return content_identifier(prefix, digest)


def _with_experiment_reference(
    references: Iterable[str],
    experiment_node_id: int,
) -> Tuple[str, ...]:
    result = list(references)
    experiment_reference = f"experiment_node:{experiment_node_id}"
    if experiment_reference not in result:
        result.append(experiment_reference)
    return tuple(result)


@dataclass(frozen=True)
class EvaluatorEvidenceWriteback:
    """Validated archive changes derived only from explicit evaluator metadata."""

    claim_updates: Tuple[EvidenceClaim, ...] = ()
    gap_updates: Tuple[EvaluationGap, ...] = ()

    def apply_to_outcome(self, outcome: IdeaOutcome) -> IdeaOutcome:
        supported = tuple(
            claim.claim_id
            for claim in self.claim_updates
            if claim.status == EvidenceStatus.SUPPORTED
        )
        contradicted = tuple(
            claim.claim_id
            for claim in self.claim_updates
            if claim.status == EvidenceStatus.CONTRADICTED
        )
        if set(outcome.supported_claim_ids) & set(contradicted) or set(
            outcome.contradicted_claim_ids
        ) & set(supported):
            raise ValueError("evaluator evidence conflicts with outcome claims")
        gap_effects = tuple(gap.gap_id for gap in self.gap_updates)
        return replace(
            outcome,
            supported_claim_ids=tuple(
                dict.fromkeys((*outcome.supported_claim_ids, *supported))
            ),
            contradicted_claim_ids=tuple(
                dict.fromkeys((*outcome.contradicted_claim_ids, *contradicted))
            ),
            gap_effects=tuple(dict.fromkeys((*outcome.gap_effects, *gap_effects))),
        )


def build_evaluator_evidence_writeback(
    metadata: Dict[str, Any],
    *,
    idea: IdeaRecord,
    archive_state: IdeaArchiveState,
    observed_at: str,
) -> EvaluatorEvidenceWriteback:
    """Parse the optional strict evaluator evidence object without score inference."""
    if not isinstance(metadata, dict):
        raise ValueError("external evaluation metadata must be an object")
    if EVALUATOR_EVIDENCE_KEY not in metadata:
        return EvaluatorEvidenceWriteback()
    if not isinstance(observed_at, str) or not observed_at.strip():
        raise ValueError("ideation evidence observed timestamp must be non-empty")
    observed = datetime.fromisoformat(observed_at)
    if observed.utcoffset() != timezone.utc.utcoffset(observed):
        raise ValueError("ideation evidence observed timestamp must be UTC")
    if idea.experiment_node_id is None:
        raise ValueError("evaluator evidence requires a linked experiment")
    evidence = _require_exact_keys(
        metadata[EVALUATOR_EVIDENCE_KEY],
        {"claims", "open_gaps", "targeted_gap_updates"},
        "ideation evidence",
    )
    experiment_node_id = idea.experiment_node_id
    claims_by_id = {claim.claim_id: claim for claim in archive_state.claims}
    gaps_by_id = {gap.gap_id: gap for gap in archive_state.gaps}
    claim_updates = tuple(
        _build_claim(
            item,
            idea_id=idea.idea_id,
            experiment_node_id=experiment_node_id,
            observed_at=observed_at,
            claims_by_id=claims_by_id,
        )
        for item in _require_list(evidence["claims"], "ideation evidence claims")
    )
    open_gaps = tuple(
        _build_open_gap(
            item,
            idea_id=idea.idea_id,
            experiment_node_id=experiment_node_id,
            observed_at=observed_at,
            gaps_by_id=gaps_by_id,
        )
        for item in _require_list(evidence["open_gaps"], "ideation evidence open gaps")
    )
    targeted_updates = tuple(
        _build_targeted_gap_update(
            item,
            idea=idea,
            gaps_by_id=gaps_by_id,
            experiment_node_id=experiment_node_id,
            observed_at=observed_at,
        )
        for item in _require_list(
            evidence["targeted_gap_updates"],
            "ideation evidence targeted gap updates",
        )
    )
    claim_ids = tuple(claim.claim_id for claim in claim_updates)
    gap_ids = tuple(gap.gap_id for gap in (*open_gaps, *targeted_updates))
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("ideation evidence claims must be unique")
    if len(set(gap_ids)) != len(gap_ids):
        raise ValueError("ideation evidence gaps must be unique")
    return EvaluatorEvidenceWriteback(
        claim_updates=claim_updates,
        gap_updates=(*open_gaps, *targeted_updates),
    )


def _build_claim(
    data: Any,
    *,
    idea_id: str,
    experiment_node_id: int,
    observed_at: str,
    claims_by_id: Dict[str, EvidenceClaim],
) -> EvidenceClaim:
    item = _require_exact_keys(
        data,
        {"statement", "kind", "status", "source_refs"},
        "ideation evidence claim",
    )
    if item["kind"] not in {kind.value for kind in ClaimKind}:
        raise ValueError("ideation evidence claim kind is invalid")
    if item["status"] not in {
        EvidenceStatus.SUPPORTED.value,
        EvidenceStatus.CONTRADICTED.value,
    }:
        raise ValueError("ideation evidence claim status is invalid")
    source_refs = _with_experiment_reference(
        _require_strings(item["source_refs"], "claim source refs", nonempty=True),
        experiment_node_id,
    )
    identity = {
        "idea_id": idea_id,
        "experiment_node_id": experiment_node_id,
        "statement": item["statement"],
        "kind": item["kind"],
        "status": item["status"],
        "source_refs": list(source_refs),
    }
    original = EvidenceClaim(
        claim_id=_content_id("claim", identity),
        statement=item["statement"],
        kind=ClaimKind(item["kind"]),
        status=EvidenceStatus(item["status"]),
        source_refs=source_refs,
        affected_idea_ids=(idea_id,),
        affected_experiment_node_ids=(experiment_node_id,),
        updated_at=observed_at,
    )
    current = claims_by_id.get(original.claim_id)
    if current is None:
        return original
    if (
        current.statement != original.statement
        or current.kind != original.kind
        or current.status != original.status
        or not set(original.source_refs).issubset(current.source_refs)
        or not set(original.affected_idea_ids).issubset(current.affected_idea_ids)
        or not set(original.affected_experiment_node_ids).issubset(
            current.affected_experiment_node_ids
        )
        or datetime.fromisoformat(current.updated_at)
        < datetime.fromisoformat(original.updated_at)
    ):
        raise ValueError("persisted evaluator claim is not a compatible descendant")
    return current


def _build_open_gap(
    data: Any,
    *,
    idea_id: str,
    experiment_node_id: int,
    observed_at: str,
    gaps_by_id: Dict[str, EvaluationGap],
) -> EvaluationGap:
    item = _require_exact_keys(
        data,
        {
            "axis",
            "description",
            "evidence_refs",
            "impact",
            "uncertainty",
            "estimated_cost",
        },
        "ideation evidence open gap",
    )
    evidence_refs = _with_experiment_reference(
        _require_strings(item["evidence_refs"], "gap evidence refs", nonempty=False),
        experiment_node_id,
    )
    identity = {
        "idea_id": idea_id,
        "experiment_node_id": experiment_node_id,
        **item,
        "evidence_refs": list(evidence_refs),
    }
    original = EvaluationGap(
        gap_id=_content_id("gap", identity),
        axis=item["axis"],
        description=item["description"],
        state=GapState.OPEN,
        evidence_refs=evidence_refs,
        impact=item["impact"],
        uncertainty=item["uncertainty"],
        estimated_cost=item["estimated_cost"],
        deferral_count=0,
        opened_at=observed_at,
    )
    current = gaps_by_id.get(original.gap_id)
    if current is None:
        return original
    if (
        current.axis != original.axis
        or current.description != original.description
        or current.impact != original.impact
        or current.uncertainty != original.uncertainty
        or current.estimated_cost != original.estimated_cost
        or current.opened_at != original.opened_at
        or not set(original.evidence_refs).issubset(current.evidence_refs)
        or current.deferral_count < original.deferral_count
        or current.state not in {GapState.OPEN, GapState.INCONCLUSIVE, GapState.CLOSED}
    ):
        raise ValueError("persisted evaluator gap is not a compatible descendant")
    return current


def _build_targeted_gap_update(
    data: Any,
    *,
    idea: IdeaRecord,
    gaps_by_id: Dict[str, EvaluationGap],
    experiment_node_id: int,
    observed_at: str,
) -> EvaluationGap:
    item = _require_exact_keys(
        data,
        {"gap_id", "state", "evidence_refs", "closure_reason"},
        "ideation evidence targeted gap update",
    )
    if item["gap_id"] not in gaps_by_id:
        raise ValueError("ideation evidence targets an unknown gap")
    if item["gap_id"] not in idea.target_gap_ids:
        raise ValueError("ideation evidence updates an untargeted gap")
    if item["state"] not in {
        GapState.INCONCLUSIVE.value,
        GapState.CLOSED.value,
    }:
        raise ValueError("targeted gap update state is invalid")
    current = gaps_by_id[item["gap_id"]]
    evidence_refs = _require_strings(
        item["evidence_refs"],
        "targeted gap evidence refs",
        nonempty=True,
    )
    target_state = GapState(item["state"])
    outcome_reference = f"experiment_node:{experiment_node_id}"
    if current.state == target_state:
        if (
            current.resolution_idea_id != idea.idea_id
            or current.resolution_experiment_node_id != experiment_node_id
            or current.last_considered_at != observed_at
            or current.closure_reason != item["closure_reason"]
            or not {*evidence_refs, outcome_reference}.issubset(current.evidence_refs)
        ):
            raise ValueError("persisted targeted gap evidence differs")
        return current
    if target_state == GapState.INCONCLUSIVE and current.state == GapState.CLOSED:
        if (
            current.last_considered_at is None
            or datetime.fromisoformat(current.last_considered_at)
            < datetime.fromisoformat(observed_at)
            or not {*evidence_refs, outcome_reference}.issubset(current.evidence_refs)
        ):
            raise ValueError("persisted targeted gap is not a compatible descendant")
        return current
    require_gap_transition(current.state, target_state)
    return replace(
        current,
        state=target_state,
        evidence_refs=tuple(
            dict.fromkeys(
                _with_experiment_reference(
                    (*current.evidence_refs, *evidence_refs),
                    experiment_node_id,
                )
            )
        ),
        last_considered_at=observed_at,
        closure_reason=item["closure_reason"],
        resolution_idea_id=idea.idea_id,
        resolution_experiment_node_id=experiment_node_id,
    )
