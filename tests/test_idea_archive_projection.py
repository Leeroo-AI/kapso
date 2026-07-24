"""Pure projection tests for transaction-ready idea-archive state."""

import subprocess
import sys
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import CanonicalizationError, canonical_json_bytes
from kapso.execution.search_strategies.generic.ideation.archive import (
    ArchiveCorruptionError,
    ArchiveIdentityConflictError,
    ArchiveLifecycleError,
    IdeaArchive,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    BatchStatus,
    ClaimKind,
    EvaluationGap,
    EvaluationStatus,
    EvidenceClaim,
    EvidenceStatus,
    GapState,
    IdeaOutcome,
    IdeaStatus,
    ImplementationStatus,
)
from kapso.execution.search_strategies.generic.ideation.archive_projection import (
    build_archive_genesis,
    decode_archive_state,
    encode_archive_state,
    project_outcome,
)
from test_ideation_domain import (
    BATCH_ID,
    CLAIM_ID,
    EVIDENCE_ID,
    GAP_ID,
    IDEA_ID,
    NOW,
    analyzed_candidate,
    coding_agent_call,
    eligible_analysis,
    generated_idea,
    planned_batch,
    selection,
)

CAMPAIGN_ID = "campaign-alpha"
OUTCOME_TIME = "2030-01-01T00:00:00Z"


def test_archive_projection_import_has_no_provider_or_registry_side_effects():
    import_script = """
import sys
import kapso.execution.search_strategies.generic.ideation.archive_projection
for forbidden_module in (
    "kapso.core.llm",
    "litellm",
    "kapso.execution.coding_agents.factory",
):
    if forbidden_module in sys.modules:
        raise RuntimeError(f"unexpected eager import: {forbidden_module}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", import_script],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def _outcome(**changes) -> IdeaOutcome:
    values = {
        "evaluation_status": EvaluationStatus.VALID,
        "implementation_status": ImplementationStatus.COMPLETED,
        "normalized_delta": 0.1,
        "validation_tier": "full",
        "actual_cost": 1.0,
        "actual_duration": 30.0,
    }
    values.update(changes)
    return IdeaOutcome(**values)


def _selected_archive(tmp_path, *, idea=None) -> IdeaArchive:
    archive = IdeaArchive(tmp_path / "idea_archive.json", CAMPAIGN_ID)
    archive.create_batch(planned_batch(), expected_revision=0)
    archive.add_ideas(
        BATCH_ID,
        (generated_idea() if idea is None else idea,),
        generation_calls=(coding_agent_call(),),
        expected_revision=1,
    )
    archive.record_analyses(
        BATCH_ID,
        (analyzed_candidate(eligible_analysis()),),
        expected_revision=2,
    )
    archive.record_selection(
        BATCH_ID,
        selection(),
        selection_call=coding_agent_call(),
        expected_revision=3,
    )
    return archive


def _bridged_archive(tmp_path, *, idea=None) -> IdeaArchive:
    archive = _selected_archive(tmp_path, idea=idea)
    archive.link_experiment(IDEA_ID, 1, BATCH_ID, expected_revision=4)
    return archive


def test_genesis_requires_explicit_normalized_time_and_is_canonical(tmp_path):
    state = build_archive_genesis(
        campaign_id=CAMPAIGN_ID,
        created_at=OUTCOME_TIME,
    )

    assert state.revision == 0
    assert state.created_at == OUTCOME_TIME
    assert state.updated_at == OUTCOME_TIME
    assert decode_archive_state(encode_archive_state(state)) == state
    assert tuple(tmp_path.iterdir()) == ()
    with pytest.raises(CanonicalizationError):
        build_archive_genesis(
            campaign_id=CAMPAIGN_ID,
            created_at="2030-01-01T00:00:00+00:00",
        )


def test_archive_projection_bytes_are_exactly_canonical(tmp_path):
    state = _bridged_archive(tmp_path).state

    payload = encode_archive_state(state)

    assert payload == canonical_json_bytes(state.to_dict())
    assert not payload.endswith(b"\n")
    assert decode_archive_state(payload) == state
    with pytest.raises(ArchiveCorruptionError, match="not canonical"):
        decode_archive_state(payload + b"\n")


def test_archive_projection_rejects_noncanonical_nested_timestamp(tmp_path):
    document = _bridged_archive(tmp_path).state.to_dict()
    document["batches"][0]["created_at"] = "2026-07-19T00:00:00+00:00"

    with pytest.raises(CanonicalizationError, match="UTC timestamp"):
        decode_archive_state(canonical_json_bytes(document))


def test_outcome_projection_is_pure_and_uses_explicit_time(tmp_path):
    archive = _bridged_archive(tmp_path)
    path = archive.path
    before_bytes = path.read_bytes()
    before_state = archive.state

    projected = project_outcome(
        before_state,
        IDEA_ID,
        _outcome(),
        updated_at=OUTCOME_TIME,
    )

    assert path.read_bytes() == before_bytes
    assert archive.state == before_state
    assert projected.revision == before_state.revision + 1
    assert projected.updated_at == OUTCOME_TIME
    batch = next(batch for batch in projected.batches if batch.batch_id == BATCH_ID)
    idea = next(idea for idea in projected.ideas if idea.idea_id == IDEA_ID)
    assert batch.status is BatchStatus.COMPLETED
    assert batch.updated_at == OUTCOME_TIME
    assert idea.status is IdeaStatus.EVALUATED
    assert idea.outcome == _outcome()


def test_outcome_projection_is_idempotent_and_rejects_conflicts(tmp_path):
    state = _bridged_archive(tmp_path).state
    outcome = _outcome()
    projected = project_outcome(
        state,
        IDEA_ID,
        outcome,
        updated_at=OUTCOME_TIME,
    )

    replayed = project_outcome(
        projected,
        IDEA_ID,
        outcome,
        updated_at="2029-01-01T00:01:00Z",
    )

    assert replayed is projected
    with pytest.raises(ArchiveIdentityConflictError, match="already differs"):
        project_outcome(
            projected,
            IDEA_ID,
            replace(outcome, normalized_delta=0.2),
            updated_at="2030-01-01T00:01:00Z",
        )


def test_outcome_projection_enforces_lifecycle_and_normalized_time(tmp_path):
    selected = _selected_archive(tmp_path).state
    with pytest.raises(ArchiveLifecycleError, match="implementing idea"):
        project_outcome(
            selected,
            IDEA_ID,
            _outcome(),
            updated_at=OUTCOME_TIME,
        )

    bridged = _bridged_archive(tmp_path / "bridged").state
    with pytest.raises(CanonicalizationError, match="UTC timestamp"):
        project_outcome(
            bridged,
            IDEA_ID,
            _outcome(),
            updated_at="2030-01-01T00:00:00+00:00",
        )
    with pytest.raises(ArchiveIdentityConflictError, match="precedes"):
        project_outcome(
            bridged,
            IDEA_ID,
            _outcome(),
            updated_at="2020-01-01T00:00:00Z",
        )


def test_projection_applies_claim_and_gap_transition_as_one_revision(tmp_path):
    claim = EvidenceClaim(
        claim_id=CLAIM_ID,
        statement="Gradient clipping improves utility.",
        kind=ClaimKind.HYPOTHESIS,
        status=EvidenceStatus.INSUFFICIENT,
        source_refs=(),
        affected_idea_ids=(),
        affected_experiment_node_ids=(),
        updated_at=NOW,
    )
    gap = EvaluationGap(
        gap_id=GAP_ID,
        axis="gradient stability",
        description="The gradient clipping effect is unknown.",
        state=GapState.OPEN,
        evidence_refs=(EVIDENCE_ID,),
        impact=0.8,
        uncertainty=0.9,
        estimated_cost=1.0,
        deferral_count=0,
        opened_at=NOW,
    )
    archive = IdeaArchive(tmp_path / "idea_archive.json", CAMPAIGN_ID)
    archive.record_claims((claim,), expected_revision=0)
    archive.add_gaps((gap,), expected_revision=1)
    archive.create_batch(planned_batch(), expected_revision=2)
    archive.add_ideas(
        BATCH_ID,
        (
            replace(
                generated_idea(),
                claim_ids=(CLAIM_ID,),
                target_gap_ids=(GAP_ID,),
            ),
        ),
        generation_calls=(coding_agent_call(),),
        expected_revision=3,
    )
    archive.record_analyses(
        BATCH_ID,
        (analyzed_candidate(eligible_analysis()),),
        expected_revision=4,
    )
    archive.record_selection(
        BATCH_ID,
        selection(),
        selection_call=coding_agent_call(),
        expected_revision=5,
    )
    archive.link_experiment(IDEA_ID, 1, BATCH_ID, expected_revision=6)
    state = archive.state
    outcome = _outcome(
        gap_effects=(GAP_ID,),
        supported_claim_ids=(CLAIM_ID,),
    )
    claim_update = replace(
        claim,
        status=EvidenceStatus.SUPPORTED,
        source_refs=("experiment:1",),
        affected_idea_ids=(IDEA_ID,),
        affected_experiment_node_ids=(1,),
        updated_at="2026-07-19T00:01:00Z",
    )
    gap_update = replace(
        gap,
        state=GapState.CLOSED,
        evidence_refs=(EVIDENCE_ID, "experiment:1"),
        last_considered_at="2026-07-19T00:01:00Z",
        closure_reason="Canonical evaluation resolved the uncertainty.",
        resolution_idea_id=IDEA_ID,
        resolution_experiment_node_id=1,
    )

    projected = project_outcome(
        state,
        IDEA_ID,
        outcome,
        claim_updates=(claim_update,),
        gap_updates=(gap_update,),
        updated_at=OUTCOME_TIME,
    )

    assert projected.revision == state.revision + 1
    assert projected.claims == (claim_update,)
    assert projected.gaps == (gap_update,)
    assert decode_archive_state(encode_archive_state(projected)) == projected
    with pytest.raises(ArchiveLifecycleError, match="exactly cover"):
        project_outcome(
            state,
            IDEA_ID,
            outcome,
            claim_updates=(),
            gap_updates=(gap_update,),
            updated_at=OUTCOME_TIME,
        )
    with pytest.raises(ArchiveIdentityConflictError, match="outcome provenance"):
        project_outcome(
            state,
            IDEA_ID,
            outcome,
            claim_updates=(
                replace(
                    claim_update,
                    affected_experiment_node_ids=(),
                ),
            ),
            gap_updates=(gap_update,),
            updated_at=OUTCOME_TIME,
        )


def test_technical_failure_projects_failed_status(tmp_path):
    state = _bridged_archive(tmp_path).state

    projected = project_outcome(
        state,
        IDEA_ID,
        _outcome(
            evaluation_status=EvaluationStatus.NOT_RUN,
            implementation_status=ImplementationStatus.FAILED_TECHNICAL,
            normalized_delta=None,
            validation_tier=None,
        ),
        updated_at=OUTCOME_TIME,
    )

    idea = next(idea for idea in projected.ideas if idea.idea_id == IDEA_ID)
    assert idea.status is IdeaStatus.FAILED_TECHNICAL
