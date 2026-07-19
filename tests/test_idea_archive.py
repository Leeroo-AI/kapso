"""Durability and lifecycle tests for the campaign-local idea archive."""

import json
from dataclasses import replace

import pytest

from kapso.execution.search_strategies.generic.ideation import (
    ArchiveCorruptionError,
    ArchiveIdentityConflictError,
    ArchiveLifecycleError,
    ArchiveLinkConflictError,
    ArchiveRevisionConflictError,
    BatchStatus,
    CandidateAnalysis,
    CandidateDisposition,
    CandidateDispositionKind,
    ClaimKind,
    EvaluationGap,
    EvaluationStatus,
    EvidenceClaim,
    EvidenceStatus,
    GapState,
    IdeaArchive,
    IdeaBatch,
    IdeaOutcome,
    IdeaStatus,
    ImplementationStatus,
    ResurfacedIdea,
    SelectionDecision,
    new_identifier,
)
from kapso.execution.search_strategies.generic.ideation import archive as archive_module
from test_ideation_domain import (
    BATCH_ID,
    CLAIM_ID,
    EVIDENCE_ID,
    GAP_ID,
    IDEA_ID,
    NOW,
    OTHER_IDEA_ID,
    directive,
    eligible_analysis,
    generated_idea,
    planned_batch,
    selection,
)

CAMPAIGN_ID = "campaign-alpha"


def run_selected_lifecycle(archive: IdeaArchive) -> IdeaOutcome:
    batch = planned_batch()
    idea = generated_idea()
    assert archive.create_batch(batch, expected_revision=0) == batch
    assert archive.add_ideas(BATCH_ID, (idea,), expected_revision=1) == 2
    assert (
        archive.record_analysis(
            BATCH_ID,
            eligible_analysis(),
            expected_revision=2,
        )
        == 3
    )
    assert (
        archive.record_selection(
            BATCH_ID,
            selection(),
            expected_revision=3,
        )
        == 4
    )
    assert (
        archive.link_experiment(
            IDEA_ID,
            1,
            BATCH_ID,
            expected_revision=4,
        )
        == 5
    )
    outcome = IdeaOutcome(
        evaluation_status=EvaluationStatus.VALID,
        implementation_status=ImplementationStatus.COMPLETED,
        normalized_delta=0.1,
        validation_tier="full",
        actual_cost=1.0,
        actual_duration=30.0,
    )
    assert (
        archive.record_outcome(
            IDEA_ID,
            outcome,
            expected_revision=5,
        )
        == 6
    )
    return outcome


def test_missing_archive_is_empty_and_full_lifecycle_round_trips(tmp_path):
    path = tmp_path / "idea_archive.json"
    archive = IdeaArchive(path, CAMPAIGN_ID)
    assert archive.revision == 0
    assert not path.exists()

    outcome = run_selected_lifecycle(archive)

    restored = IdeaArchive(path, CAMPAIGN_ID)
    assert restored.revision == 6
    assert restored.get_batch(BATCH_ID).status == BatchStatus.COMPLETED
    idea = restored.get_idea(IDEA_ID)
    assert idea.status == IdeaStatus.EVALUATED
    assert idea.experiment_node_id == 1
    assert idea.outcome == outcome


def test_every_identity_mutation_is_idempotent_after_later_progress(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    outcome = run_selected_lifecycle(archive)

    assert (
        archive.create_batch(planned_batch(), expected_revision=0).batch_id == BATCH_ID
    )
    assert (
        archive.add_ideas(
            BATCH_ID,
            (generated_idea(),),
            expected_revision=1,
        )
        == 6
    )
    assert (
        archive.record_analysis(
            BATCH_ID,
            eligible_analysis(),
            expected_revision=2,
        )
        == 6
    )
    assert (
        archive.record_selection(
            BATCH_ID,
            selection(),
            expected_revision=3,
        )
        == 6
    )
    assert (
        archive.link_experiment(
            IDEA_ID,
            1,
            BATCH_ID,
            expected_revision=4,
        )
        == 6
    )
    assert (
        archive.record_outcome(
            IDEA_ID,
            outcome,
            expected_revision=5,
        )
        == 6
    )


def test_conflicting_replay_and_stale_new_mutation_fail_loudly(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    archive.create_batch(planned_batch(), expected_revision=0)

    conflicting = replace(planned_batch(), context_hash="9" * 64)
    with pytest.raises(ArchiveIdentityConflictError, match="different content"):
        archive.create_batch(conflicting, expected_revision=0)

    second_batch = replace(
        planned_batch(),
        batch_id=new_identifier("batch"),
        iteration_index=1,
    )
    with pytest.raises(ArchiveRevisionConflictError, match="expected revision 0"):
        archive.create_batch(second_batch, expected_revision=0)


def test_atomic_replace_failure_preserves_prior_file_and_memory(tmp_path, monkeypatch):
    path = tmp_path / "ideas.json"
    archive = IdeaArchive(path, CAMPAIGN_ID)
    archive.create_batch(planned_batch(), expected_revision=0)
    prior_text = path.read_text(encoding="utf-8")

    def fail_replace(source, target):
        raise OSError(f"injected replace failure for {source} -> {target}")

    monkeypatch.setattr(archive_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        archive.add_ideas(
            BATCH_ID,
            (generated_idea(),),
            expected_revision=1,
        )

    assert archive._state.revision == 1
    assert path.read_text(encoding="utf-8") == prior_text
    assert IdeaArchive(path, CAMPAIGN_ID).revision == 1


@pytest.mark.parametrize(
    "content,error",
    (
        ("", "empty"),
        ('{"schema": NaN}', "non-finite"),
        ('{"schema": "one", "schema": "two"}', "duplicate JSON key"),
    ),
)
def test_corrupt_archive_never_degrades_to_empty(tmp_path, content, error):
    path = tmp_path / "ideas.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ArchiveCorruptionError, match=error):
        IdeaArchive(path, CAMPAIGN_ID)


def test_unknown_persisted_fields_are_rejected(tmp_path):
    path = tmp_path / "ideas.json"
    archive = IdeaArchive(path, CAMPAIGN_ID)
    archive.create_batch(planned_batch(), expected_revision=0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["obsolete"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArchiveCorruptionError, match="incompatible fields"):
        IdeaArchive(path, CAMPAIGN_ID)


def test_multi_batch_campaign_remains_strictly_loadable(tmp_path):
    path = tmp_path / "ideas.json"
    archive = IdeaArchive(path, CAMPAIGN_ID)
    batch_count = 40
    for iteration_index in range(batch_count):
        batch = replace(
            planned_batch(),
            batch_id=new_identifier("batch"),
            iteration_index=iteration_index,
            context_hash=f"{iteration_index:064x}",
        )
        archive.create_batch(batch, expected_revision=iteration_index)

    restored = IdeaArchive(path, CAMPAIGN_ID)
    assert restored.revision == batch_count
    assert len(restored.state.batches) == batch_count


def test_one_idea_cannot_link_to_two_experiment_nodes(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    archive.create_batch(planned_batch(), expected_revision=0)
    archive.add_ideas(BATCH_ID, (generated_idea(),), expected_revision=1)
    archive.record_analysis(BATCH_ID, eligible_analysis(), expected_revision=2)
    archive.record_selection(BATCH_ID, selection(), expected_revision=3)
    archive.link_experiment(IDEA_ID, 1, BATCH_ID, expected_revision=4)

    with pytest.raises(ArchiveLinkConflictError, match="another node"):
        archive.link_experiment(IDEA_ID, 2, BATCH_ID, expected_revision=5)


def test_deferred_idea_can_be_selected_later_without_changing_origin(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    first_batch = planned_batch()
    first = generated_idea()
    deferred = generated_idea(OTHER_IDEA_ID)
    archive.create_batch(first_batch, expected_revision=0)
    archive.add_ideas(BATCH_ID, (first, deferred), expected_revision=1)
    archive.record_analysis(
        BATCH_ID,
        CandidateAnalysis(idea_id=IDEA_ID, eligible=True),
        expected_revision=2,
    )
    archive.record_analysis(
        BATCH_ID,
        CandidateAnalysis(idea_id=OTHER_IDEA_ID, eligible=True),
        expected_revision=3,
    )
    first_decision = replace(
        selection(),
        fallback_idea_ids=(OTHER_IDEA_ID,),
        dispositions=(
            CandidateDisposition(
                IDEA_ID,
                CandidateDispositionKind.SELECTED,
                "Best current utility.",
            ),
            CandidateDisposition(
                OTHER_IDEA_ID,
                CandidateDispositionKind.DEFERRED,
                "Promising fallback.",
            ),
        ),
    )
    archive.record_selection(BATCH_ID, first_decision, expected_revision=4)
    archive.link_experiment(IDEA_ID, 1, BATCH_ID, expected_revision=5)
    archive.record_outcome(
        IDEA_ID,
        IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=0.01,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        ),
        expected_revision=6,
    )

    second_batch_id = new_identifier("batch")
    second_batch = replace(
        planned_batch(),
        batch_id=second_batch_id,
        iteration_index=1,
        context_hash="5" * 64,
    )
    new_idea_id = new_identifier("idea")
    new_idea = replace(
        generated_idea(new_idea_id),
        origin_batch_id=second_batch_id,
    )
    archive.create_batch(second_batch, expected_revision=7)
    archive.add_ideas(
        second_batch_id,
        (new_idea,),
        resurfaced_ideas=(
            ResurfacedIdea(
                idea_id=OTHER_IDEA_ID,
                changed_conditions=("new comparable evidence",),
            ),
        ),
        expected_revision=8,
    )
    archive.record_analysis(
        second_batch_id,
        CandidateAnalysis(idea_id=new_idea_id, eligible=True),
        expected_revision=9,
    )
    archive.record_analysis(
        second_batch_id,
        CandidateAnalysis(idea_id=OTHER_IDEA_ID, eligible=True),
        expected_revision=10,
    )
    second_decision = SelectionDecision(
        selected_idea_id=OTHER_IDEA_ID,
        fallback_idea_ids=(new_idea_id,),
        dispositions=(
            CandidateDisposition(
                OTHER_IDEA_ID,
                CandidateDispositionKind.SELECTED,
                "New evidence resolves its earlier uncertainty.",
            ),
            CandidateDisposition(
                new_idea_id,
                CandidateDispositionKind.DEFERRED,
                "Keep as fallback.",
            ),
        ),
        diagnosis_audit=(),
        hard_rule_results=("schema valid",),
        gap_decisions=("no reserved gap",),
        duplicate_overrides=(),
        decision_summary="Resurface the prior idea under changed evidence.",
        expected_benefit=0.1,
        expected_cost=1.0,
    )
    archive.record_selection(second_batch_id, second_decision, expected_revision=11)

    selected = archive.get_idea(OTHER_IDEA_ID)
    assert selected.origin_batch_id == BATCH_ID
    assert selected.selected_in_batch_id == second_batch_id
    assert selected.status == IdeaStatus.SELECTED


def test_outcome_atomically_updates_claims_and_targeted_gaps(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
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
    archive.record_claims((claim,), expected_revision=0)
    archive.add_gaps((gap,), expected_revision=1)
    archive.create_batch(planned_batch(), expected_revision=2)
    idea = replace(
        generated_idea(),
        claim_ids=(CLAIM_ID,),
        target_gap_ids=(GAP_ID,),
    )
    archive.add_ideas(BATCH_ID, (idea,), expected_revision=3)
    archive.record_analysis(BATCH_ID, eligible_analysis(), expected_revision=4)
    archive.record_selection(BATCH_ID, selection(), expected_revision=5)
    archive.link_experiment(IDEA_ID, 1, BATCH_ID, expected_revision=6)
    outcome = IdeaOutcome(
        evaluation_status=EvaluationStatus.VALID,
        implementation_status=ImplementationStatus.COMPLETED,
        normalized_delta=0.1,
        validation_tier="full",
        actual_cost=1.0,
        actual_duration=30.0,
        gap_effects=("closed gradient stability gap",),
        supported_claim_ids=(CLAIM_ID,),
    )
    claim_update = replace(
        claim,
        status=EvidenceStatus.SUPPORTED,
        source_refs=("experiment:1",),
        affected_idea_ids=(IDEA_ID,),
        affected_experiment_node_ids=(1,),
        updated_at="2026-07-19T00:01:00+00:00",
    )
    gap_update = replace(
        gap,
        state=GapState.CLOSED,
        evidence_refs=(EVIDENCE_ID, "experiment:1"),
        last_considered_at="2026-07-19T00:01:00+00:00",
        closure_reason="Canonical evaluation resolved the uncertainty.",
        resolution_idea_id=IDEA_ID,
        resolution_experiment_node_id=1,
    )
    assert (
        archive.record_outcome(
            IDEA_ID,
            outcome,
            claim_updates=(claim_update,),
            gap_updates=(gap_update,),
            expected_revision=7,
        )
        == 8
    )

    assert archive.get_claim(CLAIM_ID).status == EvidenceStatus.SUPPORTED
    assert archive.list_gaps()[0].state == GapState.CLOSED
    assert archive.get_batch(BATCH_ID).status == BatchStatus.COMPLETED


def test_one_repair_candidate_can_extend_a_generated_unanalyzed_batch(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    archive.create_batch(planned_batch(), expected_revision=0)
    archive.add_ideas(BATCH_ID, (generated_idea(),), expected_revision=1)
    with pytest.raises(ArchiveIdentityConflictError):
        archive.add_repair_idea(
            BATCH_ID,
            generated_idea(),
            expected_revision=2,
        )
    repair = replace(
        generated_idea(new_identifier("idea")),
        origin_batch_id=BATCH_ID,
        proposal="Repair missing descriptor coverage.",
    )

    assert archive.add_repair_idea(BATCH_ID, repair, expected_revision=2) == 3
    assert archive.add_repair_idea(BATCH_ID, repair, expected_revision=2) == 3
    batch = archive.get_batch(BATCH_ID)
    assert batch.status == BatchStatus.GENERATED
    assert batch.generated_idea_ids == (IDEA_ID, repair.idea_id)
    assert batch.considered_idea_ids == (IDEA_ID, repair.idea_id)

    with pytest.raises(ArchiveLifecycleError, match="consumed"):
        archive.add_repair_idea(
            BATCH_ID,
            replace(repair, idea_id=new_identifier("idea")),
            expected_revision=3,
        )


def test_unknown_model_claim_survives_only_as_an_invalid_candidate(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", CAMPAIGN_ID)
    archive.create_batch(planned_batch(), expected_revision=0)
    idea = replace(generated_idea(), claim_ids=(new_identifier("claim"),))
    archive.add_ideas(BATCH_ID, (idea,), expected_revision=1)
    analysis = CandidateAnalysis(
        idea_id=IDEA_ID,
        eligible=False,
        hard_failures=("claim_reference_unknown",),
    )
    archive.record_analysis(BATCH_ID, analysis, expected_revision=2)

    assert archive.get_idea(IDEA_ID).status == IdeaStatus.INVALID
    assert archive.get_idea(IDEA_ID).claim_ids == idea.claim_ids
