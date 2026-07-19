"""Behavioral tests for deterministic candidate analysis and repair gating."""

import hashlib
from dataclasses import replace

from kapso.execution.search_strategies.generic.ideation.analyzer import (
    AnalyzerSettings,
    CandidateAnalyzer,
)
from kapso.execution.search_strategies.generic.ideation.archive import IdeaArchive
from kapso.execution.search_strategies.generic.ideation.embeddings import (
    EmbeddingBatch,
    EmbeddingSettings,
)
from kapso.execution.search_strategies.generic.ideation.evidence import (
    CampaignEvidenceBuilder,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    EmbeddingRecord,
    EmbeddingTelemetry,
    EvidenceClaim,
    EvidenceStatus,
    ClaimKind,
    IdeaDescriptor,
    ObjectiveDirection,
    new_identifier,
)
from test_ideation_domain import (
    BATCH_ID,
    CLAIM_ID,
    EVIDENCE_ID,
    NOW,
    directive,
    generated_idea,
    planned_batch,
    resolved_parent,
)
from test_ideation_evidence import capacity, evidence_settings


def context(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")
    snapshot = CampaignEvidenceBuilder(evidence_settings()).build(
        campaign_id="campaign-alpha",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        experiments=(),
        archive_state=archive.state,
        capacity=capacity(),
        generated_at=NOW,
    )
    search_directive = replace(
        directive(),
        evidence_snapshot_id=snapshot.snapshot_id,
        decision=replace(
            directive().decision,
            reasons=(
                replace(
                    directive().decision.reasons[0],
                    evidence_refs=(snapshot.snapshot_id,),
                ),
            ),
        ),
    )
    return archive, snapshot, search_directive


def candidate(tmp_path, snapshot, search_directive, **changes):
    artifact = tmp_path / "agent-result.json"
    artifact.write_text("{}", encoding="utf-8")
    values = {
        "descriptor": search_directive.operator_briefs[0].descriptor_target,
        "evidence_refs": (snapshot.snapshot_id,),
        "parent_experiment_node_ids": (resolved_parent().node_id,),
        "generation_artifacts": (str(artifact.resolve()),),
    }
    values.update(changes)
    return replace(generated_idea(), **values)


def analyzer(provider=None, minimum=2):
    return CandidateAnalyzer(
        AnalyzerSettings(
            semantic_similarity_threshold=0.9,
            max_neighbors=3,
            minimum_distinct_eligible=minimum,
        ),
        provider,
    )


def analyze_one(
    tmp_path, *, candidate_changes=None, capacity_changes=None, provider=None
):
    archive, snapshot, search_directive = context(tmp_path)
    idea = candidate(
        tmp_path,
        snapshot,
        search_directive,
        **(candidate_changes or {}),
    )
    result = analyzer(provider).analyze_pool(
        batch_id=BATCH_ID,
        candidates=(idea,),
        archive_state=archive.state,
        evidence_snapshot=snapshot,
        directive=search_directive,
        capacity=capacity(**(capacity_changes or {})),
    )
    return result.candidates[0]


def test_valid_candidate_passes_hard_rules_and_records_no_synthetic_novelty(tmp_path):
    result = analyze_one(tmp_path)
    assert result.analysis.eligible
    assert result.analysis.hard_failures == ()
    assert result.analysis.semantic_neighbors == ()
    assert result.embedding is None


def test_descriptor_parent_evidence_artifact_and_capacity_rules_are_hard(tmp_path):
    bad_descriptor = IdeaDescriptor(
        approach_family="unassigned",
        intervention_target="other",
        mechanism="other",
        expected_effect="other",
    )
    result = analyze_one(
        tmp_path,
        candidate_changes={
            "descriptor": bad_descriptor,
            "evidence_refs": ("unknown-evidence",),
            "parent_experiment_node_ids": (),
            "generation_artifacts": (str((tmp_path / "missing").resolve()),),
        },
        capacity_changes={"can_run_comparable_evaluation": False},
    )
    assert not result.analysis.eligible
    assert set(result.analysis.hard_failures) >= {
        "operator_descriptor_mismatch",
        "evidence_reference_unknown",
        "parent_experiment_provenance_mismatch",
        "generation_artifacts_invalid",
        "capacity_cannot_run_comparable_evaluation",
    }


def test_contradicted_claim_requires_an_explicit_resolving_test(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    claim = EvidenceClaim(
        claim_id=CLAIM_ID,
        statement="The proposed mechanism does not improve utility.",
        kind=ClaimKind.HYPOTHESIS,
        status=EvidenceStatus.CONTRADICTED,
        source_refs=("evaluation:contradiction",),
        affected_idea_ids=(),
        affected_experiment_node_ids=(),
        updated_at=NOW,
    )
    snapshot = replace(snapshot, claims=(claim,))
    contradicted = candidate(
        tmp_path,
        snapshot,
        search_directive,
        evidence_refs=("evaluation:contradiction",),
        claim_ids=(CLAIM_ID,),
    )
    first = (
        analyzer()
        .analyze_pool(
            batch_id=BATCH_ID,
            candidates=(contradicted,),
            archive_state=archive.state,
            evidence_snapshot=snapshot,
            directive=search_directive,
            capacity=capacity(),
        )
        .candidates[0]
    )
    assert not first.analysis.eligible
    assert f"contradicted_claim_not_resolved:{CLAIM_ID}" in (
        first.analysis.hard_failures
    )

    resolving = replace(contradicted, resolves_claim_ids=(CLAIM_ID,))
    second = (
        analyzer()
        .analyze_pool(
            batch_id=BATCH_ID,
            candidates=(resolving,),
            archive_state=archive.state,
            evidence_snapshot=snapshot,
            directive=search_directive,
            capacity=capacity(),
        )
        .candidates[0]
    )
    assert second.analysis.eligible


def test_exact_duplicate_requires_derived_changed_condition(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    previous_batch_id = new_identifier("batch")
    previous_idea_id = new_identifier("idea")
    previous_batch = replace(
        planned_batch(),
        batch_id=previous_batch_id,
        iteration_index=1,
        context_hash="9" * 64,
    )
    previous = replace(
        generated_idea(previous_idea_id),
        origin_batch_id=previous_batch_id,
        descriptor=search_directive.operator_briefs[0].descriptor_target,
        evidence_refs=(snapshot.snapshot_id,),
        parent_experiment_node_ids=(resolved_parent().node_id,),
    )
    archive.create_batch(previous_batch, expected_revision=archive.revision)
    archive.add_ideas(
        previous_batch_id,
        (previous,),
        expected_revision=archive.revision,
    )
    duplicate = candidate(
        tmp_path,
        snapshot,
        search_directive,
        proposal=previous.proposal,
        evaluation_method=previous.evaluation_method,
    )
    first = (
        analyzer()
        .analyze_pool(
            batch_id=BATCH_ID,
            candidates=(duplicate,),
            archive_state=archive.state,
            evidence_snapshot=snapshot,
            directive=search_directive,
            capacity=capacity(),
        )
        .candidates[0]
    )
    assert not first.analysis.eligible
    assert first.analysis.exact_duplicate_of == previous_idea_id
    assert "exact_duplicate_without_changed_conditions" in first.analysis.hard_failures

    changed = replace(duplicate, evaluation_method="Run a new decisive ablation.")
    second = (
        analyzer()
        .analyze_pool(
            batch_id=BATCH_ID,
            candidates=(changed,),
            archive_state=archive.state,
            evidence_snapshot=snapshot,
            directive=search_directive,
            capacity=capacity(),
        )
        .candidates[0]
    )
    assert second.analysis.eligible
    assert second.analysis.exact_duplicate_changed_conditions == (
        "evaluation_method_changed",
    )


class UnitVectorProvider:
    def __init__(self):
        self.settings = EmbeddingSettings(
            enabled=True,
            model="semantic-test",
            dimensions=2,
            timeout_seconds=5,
            max_retries=0,
        )
        self.calls = []

    def embed(self, texts):
        complete = tuple(texts)
        self.calls.append(complete)
        return EmbeddingBatch(
            records=tuple(
                EmbeddingRecord(
                    provider="openai",
                    model=self.settings.model,
                    dimensions=2,
                    input_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    vector=(1.0, 0.0),
                )
                for text in complete
            ),
            telemetry=EmbeddingTelemetry(
                provider="openai",
                model=self.settings.model,
                call_count=1,
                input_tokens=20,
                duration_seconds=1,
            ),
        )


def test_semantic_neighbor_is_an_alarm_not_an_automatic_rejection(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    previous_batch_id = new_identifier("batch")
    previous_idea_id = new_identifier("idea")
    previous_batch = replace(
        planned_batch(),
        batch_id=previous_batch_id,
        iteration_index=1,
        context_hash="8" * 64,
    )
    previous = replace(
        generated_idea(previous_idea_id),
        origin_batch_id=previous_batch_id,
        descriptor=search_directive.operator_briefs[0].descriptor_target,
        evidence_refs=(snapshot.snapshot_id,),
        parent_experiment_node_ids=(resolved_parent().node_id,),
    )
    archive.create_batch(previous_batch, expected_revision=archive.revision)
    archive.add_ideas(
        previous_batch_id,
        (previous,),
        expected_revision=archive.revision,
    )
    novel_proposal = candidate(
        tmp_path,
        snapshot,
        search_directive,
        proposal="A materially different measured intervention.",
    )
    provider = UnitVectorProvider()
    pool = analyzer(provider).analyze_pool(
        batch_id=BATCH_ID,
        candidates=(novel_proposal,),
        archive_state=archive.state,
        evidence_snapshot=snapshot,
        directive=search_directive,
        capacity=capacity(),
    )
    result = pool.candidates[0]
    assert result.analysis.eligible
    assert result.analysis.semantic_neighbors[0].idea_id == previous_idea_id
    assert result.analysis.semantic_neighbors[0].similarity == 1.0
    assert f"semantic_neighbor:{previous_idea_id}" in result.similarity_flags
    assert result.embedding is not None
    assert pool.embedding_telemetry.call_count == 1


def test_analysis_metadata_is_persisted_atomically_with_eligibility(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    idea = candidate(tmp_path, snapshot, search_directive)
    batch = replace(
        planned_batch(),
        evidence_snapshot_id=snapshot.snapshot_id,
        directive=search_directive,
    )
    archive.create_batch(batch, expected_revision=0)
    archive.add_ideas(BATCH_ID, (idea,), expected_revision=1)
    analyzed = (
        analyzer(UnitVectorProvider())
        .analyze_pool(
            batch_id=BATCH_ID,
            candidates=(idea,),
            archive_state=archive.state,
            evidence_snapshot=snapshot,
            directive=search_directive,
            capacity=capacity(),
        )
        .candidates[0]
    )

    archive.record_analysis(
        BATCH_ID,
        analyzed.analysis,
        embedding=analyzed.embedding,
        nearest_experiment_node_ids=analyzed.nearest_experiment_node_ids,
        similarity_flags=analyzed.similarity_flags,
        expected_revision=2,
    )

    persisted = archive.get_idea(idea.idea_id)
    assert persisted.embedding == analyzed.embedding
    assert persisted.similarity_flags == analyzed.similarity_flags
    assert archive.get_batch(BATCH_ID).status.value == "analyzed"


def test_repair_is_requested_once_only_when_distinct_eligibility_is_low(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    idea = candidate(tmp_path, snapshot, search_directive)
    checker = analyzer(minimum=2)
    pool = checker.analyze_pool(
        batch_id=BATCH_ID,
        candidates=(idea,),
        archive_state=archive.state,
        evidence_snapshot=snapshot,
        directive=search_directive,
        capacity=capacity(),
    )
    repair = checker.repair_request(pool, search_directive)
    assert repair is not None
    assert repair.eligible_descriptor_count == 1
    assert (
        checker.repair_request(
            pool,
            replace(search_directive, repair_quota=0),
        )
        is None
    )
