"""Behavioral tests for the ideation orchestration boundary."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.execution.search_strategies.generic.ideation import (
    AnalyzerSettings,
    CampaignAction,
    CampaignEvidenceBuilder,
    CandidateAnalyzer,
    CandidateGenerator,
    CandidateGeneratorSettings,
    CandidateSelector,
    CodingAgentCallResult,
    EmbeddingBatch,
    EmbeddingRecord,
    EmbeddingSettings,
    EmbeddingTelemetry,
    ExperimentInput,
    GapPrioritySettings,
    GenerationMemberSettings,
    IdeaArchive,
    IdeaDescriptor,
    IdeaStatus,
    IdeationEngine,
    ObjectiveDirection,
    OperatorSettings,
    ParentPlanKind,
    ResolvedParentSnapshot,
)
from test_ideation_domain import NOW
from test_ideation_evidence import capacity, evidence_settings


def member():
    return GenerationMemberSettings(
        cli="codex",
        model="test-model",
        timeout_seconds=30,
        effort="high",
        allowed_tools=("Read",),
    )


class PacketRunner:
    def __init__(
        self, artifact_dir: Path, *, invalidate_first=False, fail_selector=False
    ):
        self.artifact_dir = artifact_dir
        self.invalidate_first = invalidate_first
        self.fail_selector = fail_selector
        self.roles = []

    def run(self, request, response_schema):
        self.roles.append(request.role)
        if request.role == "candidate_selector" and self.fail_selector:
            raise RuntimeError("selector failed")
        packet = json.loads(request.prompt.split("Mandatory packet:\n\n", 1)[1])
        if request.role == "evidence_author":
            code_diff_reference = next(
                reference
                for reference in packet["allowed_source_refs"]
                if reference.endswith(":code_diff")
            )
            feedback_reference = next(
                reference
                for reference in packet["allowed_source_refs"]
                if reference.endswith(":feedback")
            )
            output = {
                "claims": [
                    {
                        "statement": "The selected intervention caused the measured gain.",
                        "kind": "hypothesis",
                        "status": "supported",
                        "source_refs": [code_diff_reference, feedback_reference],
                    }
                ],
                "open_gaps": [],
                "targeted_gap_updates": [],
            }
        elif request.role == "candidate_selector":
            eligible_ids = tuple(
                item["idea_id"] for item in packet["eligible_candidates"]
            )
            output = {
                "selected_idea_id": eligible_ids[0],
                "fallback_idea_ids": list(eligible_ids[1:]),
                "rejected_ideas": [],
                "diagnosis_audit": [],
                "hard_rule_results": ["All deterministic rules passed."],
                "gap_decisions": ["No evaluation gap was displaced."],
                "duplicate_overrides": [],
                "decision_summary": "Select the strongest eligible intervention.",
                "expected_benefit": 0.1,
                "expected_cost": 1.0,
            }
        else:
            descriptor = packet["operator_brief"]["descriptor_target"]
            if self.invalidate_first and request.role == "candidate_0":
                descriptor = IdeaDescriptor(
                    approach_family="wrong_family",
                    intervention_target="wrong_target",
                    mechanism="wrong_mechanism",
                    expected_effect="wrong_effect",
                ).to_dict()
            output = {
                "proposal": f"Execute {request.role} as one measurable intervention.",
                "directive_rationale": "It follows the assigned evidence-backed operator.",
                "descriptor": descriptor,
                "assumptions": ["The intervention can be evaluated independently."],
                "evidence_refs": [packet["evidence_snapshot"]["snapshot_id"]],
                "claim_ids": [],
                "resolves_claim_ids": [],
                "expected_observations": ["Comparable utility changes."],
                "evaluation_method": "Run the canonical evaluator.",
                "resource_request": "One admitted experiment.",
                "predicted_gain": 0.02,
                "predicted_cost": 1.0,
                "confidence": 0.6,
                "claimed_nearest_idea_id": None,
                "claimed_nearest_experiment_node_id": None,
            }
        artifact = (self.artifact_dir / f"{request.role}.json").resolve()
        artifact.write_text(json.dumps(output), encoding="utf-8")
        return CodingAgentCallResult(
            output=json.dumps(output),
            duration_seconds=1,
            cost_usd=None,
            input_tokens=10,
            output_tokens=5,
            artifacts=(str(artifact),),
        )


class StoreAheadPacketRunner(PacketRunner):
    """Test double with the production runner's durable operation cache."""

    def __init__(self, artifact_dir: Path, *, fail_once_role: str):
        super().__init__(artifact_dir)
        self.fail_once_role = fail_once_role
        self.failed = False
        self.completed_by_operation = {}
        self.executions = []

    def run(self, request, response_schema):
        persisted = self.completed_by_operation.get(request.operation_id)
        if persisted is not None:
            return persisted
        self.executions.append(request.role)
        if request.role == self.fail_once_role and not self.failed:
            self.failed = True
            raise RuntimeError(f"{request.role} interrupted")
        result = super().run(request, response_schema)
        self.completed_by_operation[request.operation_id] = result
        return result


class CountingParents:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.materialized_refs = []

    def resolve(self, plan):
        node_id = plan.experiment_node_id
        if plan.kind == ParentPlanKind.BASELINE:
            node_id = None
        ref = "baseline-sha" if node_id is None else f"experiment-{node_id}-sha"
        return ResolvedParentSnapshot(
            node_id=node_id,
            branch_name="main" if node_id is None else f"experiment-{node_id}",
            git_ref=ref,
            materialized_ref=ref,
            diff_base_ref=ref,
            feedback_base_ref=ref,
        )

    @contextmanager
    def materialize(self, parent):
        self.materialized_refs.append(parent.materialized_ref)
        yield str(self.workspace)


class DeterministicEmbeddingProvider:
    def __init__(self):
        self.settings = EmbeddingSettings(
            enabled=True,
            model="test-embedding-model",
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
                input_tokens=23,
                duration_seconds=0.5,
            ),
        )


def build_engine(
    tmp_path,
    runner,
    *,
    minimum_distinct=2,
    embedding_provider=None,
):
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")
    generator_settings = CandidateGeneratorSettings(
        members=(member(), member()),
        repair_member=member(),
    )
    engine = IdeationEngine(
        archive=archive,
        evidence_builder=CampaignEvidenceBuilder(evidence_settings()),
        operator_settings=OperatorSettings(
            candidate_quota=2,
            repair_quota=1,
            reserve_gap_slot=True,
        ),
        gap_priority_settings=GapPrioritySettings(
            default_evidence_confidence=0.5,
            default_uncertainty_reduction=0.5,
            default_cost=1.0,
            minimum_cost=0.1,
        ),
        generator=CandidateGenerator(runner, generator_settings),
        analyzer=CandidateAnalyzer(
            AnalyzerSettings(
                semantic_similarity_threshold=0.92,
                max_neighbors=5,
                minimum_distinct_eligible=minimum_distinct,
            ),
            embedding_provider=embedding_provider,
        ),
        selector=CandidateSelector(runner, member()),
    )
    return archive, engine


def run_engine(engine, parents, **changes):
    arguments = {
        "campaign_id": "campaign-alpha",
        "iteration_index": 0,
        "problem_statement": "Improve the complete task solution.",
        "objective_direction": ObjectiveDirection.MAXIMIZE,
        "experiments": (),
        "capacity": capacity(),
        "selector_workspace": str(parents.workspace),
        "parent_resolver": parents.resolve,
        "parent_materializer": parents.materialize,
        "generated_at": NOW,
    }
    arguments.update(changes)
    return engine.run(**arguments)


def telemetry_totals(telemetry):
    return {
        "coding_agent_call_count": telemetry.coding_agent_call_count,
        "coding_agent_duration_seconds": telemetry.coding_agent_duration_seconds,
        "known_coding_agent_cost_usd": telemetry.known_coding_agent_cost_usd,
        "unpriced_coding_agent_call_count": (
            telemetry.unpriced_coding_agent_call_count
        ),
        "embedding": telemetry.embedding,
    }


def test_engine_persists_selection_and_materializes_each_parent_ref_once(tmp_path):
    runner = PacketRunner(tmp_path)
    archive, engine = build_engine(tmp_path, runner)
    parents = CountingParents(tmp_path)

    result = run_engine(engine, parents)

    assert result.action == CampaignAction.IDEATE
    assert result.selected_idea.status == IdeaStatus.SELECTED
    assert result.selection.selected_idea_id == result.selected_idea.idea_id
    assert result.archive_revision == archive.revision
    assert parents.materialized_refs == ["baseline-sha"]
    assert result.telemetry.coding_agent_call_count == 3
    persisted_batch = archive.state.batches[0]
    assert persisted_batch.selection == result.selection
    assert persisted_batch.status.value == "selected"


def test_engine_runs_exactly_one_repair_before_final_analysis(tmp_path):
    runner = PacketRunner(tmp_path, invalidate_first=True)
    archive, engine = build_engine(tmp_path, runner)
    parents = CountingParents(tmp_path)

    result = run_engine(engine, parents)

    assert runner.roles.count("diversity_repair") == 1
    assert len(archive.state.batches[0].generated_idea_ids) == 3
    assert len(archive.state.batches[0].analyses) == 3
    assert result.telemetry.coding_agent_call_count == 4


def test_selector_failure_leaves_analyzed_batch_without_fallback_winner(tmp_path):
    runner = PacketRunner(tmp_path, fail_selector=True)
    embedding_provider = DeterministicEmbeddingProvider()
    archive, engine = build_engine(
        tmp_path,
        runner,
        embedding_provider=embedding_provider,
    )
    parents = CountingParents(tmp_path)

    with pytest.raises(RuntimeError, match="selector failed"):
        run_engine(engine, parents)

    assert archive.state.batches[0].status.value == "analyzed"
    assert archive.state.batches[0].selection is None
    assert all(idea.status == IdeaStatus.GENERATED for idea in archive.state.ideas)

    runner.fail_selector = False
    resumed = run_engine(
        engine,
        parents,
        resume_batch_id=archive.state.batches[0].batch_id,
    )

    assert resumed.selected_idea is not None
    assert len(archive.state.batches) == 1
    assert runner.roles.count("candidate_0") == 1
    assert runner.roles.count("candidate_1") == 1
    assert runner.roles.count("candidate_selector") == 2
    assert resumed.telemetry.coding_agent_call_count == 3
    assert (
        resumed.telemetry.embedding
        == archive.get_batch(resumed.batch_id).embedding_telemetry
    )
    assert len(embedding_provider.calls) == 1
    reference_dir = tmp_path / "uninterrupted"
    reference_dir.mkdir()
    reference_provider = DeterministicEmbeddingProvider()
    _, reference_engine = build_engine(
        reference_dir,
        PacketRunner(reference_dir),
        embedding_provider=reference_provider,
    )
    reference = run_engine(reference_engine, CountingParents(reference_dir))
    assert telemetry_totals(resumed.telemetry) == telemetry_totals(reference.telemetry)


def test_generated_batch_resume_reuses_the_persisted_candidate_pool(
    tmp_path,
    monkeypatch,
):
    runner = PacketRunner(tmp_path)
    archive, engine = build_engine(tmp_path, runner)
    parents = CountingParents(tmp_path)
    original_analyze = engine.analyzer.analyze_pool

    def interrupt_analysis(**kwargs):
        raise RuntimeError("analysis interrupted")

    monkeypatch.setattr(engine.analyzer, "analyze_pool", interrupt_analysis)
    with pytest.raises(RuntimeError, match="analysis interrupted"):
        run_engine(engine, parents)

    batch = archive.state.batches[0]
    assert batch.status.value == "generated"
    generated_ids = batch.generated_idea_ids
    monkeypatch.setattr(engine.analyzer, "analyze_pool", original_analyze)

    resumed = run_engine(engine, parents, resume_batch_id=batch.batch_id)

    assert resumed.selected_idea is not None
    assert archive.state.batches[0].generated_idea_ids == generated_ids
    assert runner.roles.count("candidate_0") == 1
    assert runner.roles.count("candidate_1") == 1
    assert resumed.telemetry.coding_agent_call_count == 3


def test_planned_batch_resume_reuses_store_ahead_agent_result(tmp_path):
    runner = StoreAheadPacketRunner(tmp_path, fail_once_role="candidate_1")
    archive, engine = build_engine(tmp_path, runner)
    parents = CountingParents(tmp_path)

    with pytest.raises(RuntimeError, match="candidate_1 interrupted"):
        run_engine(engine, parents)

    batch = archive.state.batches[0]
    assert batch.status.value == "planned"
    assert batch.generated_idea_ids == ()
    assert runner.executions.count("candidate_0") == 1

    resumed = run_engine(engine, parents, resume_batch_id=batch.batch_id)

    assert resumed.selected_idea is not None
    assert runner.executions.count("candidate_0") == 1
    assert runner.executions.count("candidate_1") == 2
    assert resumed.telemetry.coding_agent_call_count == 3
    reference_dir = tmp_path / "planned-reference"
    reference_dir.mkdir()
    _, reference_engine = build_engine(
        reference_dir,
        PacketRunner(reference_dir),
    )
    reference = run_engine(reference_engine, CountingParents(reference_dir))
    assert telemetry_totals(resumed.telemetry) == telemetry_totals(reference.telemetry)


def test_resumed_phase_telemetry_matches_an_uninterrupted_transaction(
    tmp_path,
    monkeypatch,
):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference_runner = PacketRunner(reference_dir)
    _, reference_engine = build_engine(reference_dir, reference_runner)
    reference = run_engine(
        reference_engine,
        CountingParents(reference_dir),
    )

    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    resumed_runner = PacketRunner(resumed_dir)
    archive, resumed_engine = build_engine(resumed_dir, resumed_runner)
    parents = CountingParents(resumed_dir)
    original_analyze = resumed_engine.analyzer.analyze_pool

    def interrupt_analysis(**kwargs):
        raise RuntimeError("analysis interrupted")

    monkeypatch.setattr(resumed_engine.analyzer, "analyze_pool", interrupt_analysis)
    with pytest.raises(RuntimeError, match="analysis interrupted"):
        run_engine(resumed_engine, parents)
    batch = archive.state.batches[0]
    monkeypatch.setattr(resumed_engine.analyzer, "analyze_pool", original_analyze)

    resumed = run_engine(
        resumed_engine,
        parents,
        resume_batch_id=batch.batch_id,
    )

    assert telemetry_totals(resumed.telemetry) == telemetry_totals(reference.telemetry)


def test_failed_atomic_analysis_commit_cannot_double_count_resume_telemetry(
    tmp_path,
    monkeypatch,
):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference_provider = DeterministicEmbeddingProvider()
    _, reference_engine = build_engine(
        reference_dir,
        PacketRunner(reference_dir),
        embedding_provider=reference_provider,
    )
    reference = run_engine(reference_engine, CountingParents(reference_dir))

    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    resumed_provider = DeterministicEmbeddingProvider()
    archive, resumed_engine = build_engine(
        resumed_dir,
        PacketRunner(resumed_dir),
        embedding_provider=resumed_provider,
    )
    parents = CountingParents(resumed_dir)
    original_record_analyses = archive.record_analyses

    def interrupt_analysis_commit(*args, **kwargs):
        raise RuntimeError("analysis commit interrupted")

    monkeypatch.setattr(archive, "record_analyses", interrupt_analysis_commit)
    with pytest.raises(RuntimeError, match="analysis commit interrupted"):
        run_engine(resumed_engine, parents)
    batch = archive.state.batches[0]
    assert batch.status.value == "generated"
    assert batch.analyses == ()
    assert batch.embedding_telemetry is None
    monkeypatch.setattr(archive, "record_analyses", original_record_analyses)

    resumed = run_engine(
        resumed_engine,
        parents,
        resume_batch_id=batch.batch_id,
    )

    assert len(resumed_provider.calls) == 2
    assert telemetry_totals(resumed.telemetry) == telemetry_totals(reference.telemetry)


def test_finalize_makes_no_parent_or_agent_calls_and_creates_no_batch(tmp_path):
    runner = PacketRunner(tmp_path)
    archive, engine = build_engine(tmp_path, runner)
    parents = CountingParents(tmp_path)
    terminal_capacity = replace(capacity(), can_start_complete_action=False)

    result = run_engine(engine, parents, capacity=terminal_capacity)

    assert result.action == CampaignAction.FINALIZE
    assert result.batch_id is None
    assert archive.state.batches == ()
    assert parents.materialized_refs == []
    assert runner.roles == []


def test_recovery_reuses_original_idea_batch_node_and_makes_no_ai_call(tmp_path):
    runner = PacketRunner(tmp_path)
    archive, engine = build_engine(tmp_path, runner)
    parents = CountingParents(tmp_path)
    selected = run_engine(engine, parents)
    archive.link_experiment(
        selected.selected_idea.idea_id,
        0,
        selected.batch_id,
        expected_revision=archive.revision,
    )
    calls_before_recovery = tuple(runner.roles)
    failed = ExperimentInput(
        node_id=0,
        idea_id=selected.selected_idea.idea_id,
        selection_batch_id=selected.batch_id,
        parent_node_id=None,
        proposal=selected.selected_idea.proposal,
        score=None,
        evaluation_valid=False,
        had_error=True,
        recoverable_error=True,
        build_fidelity="full",
        attempts=(),
        feedback="Implementation stopped before evaluation.",
        technical_difficulty="A repairable implementation failure.",
        created_at=NOW,
    )

    recovered = run_engine(
        engine,
        parents,
        iteration_index=1,
        experiments=(failed,),
    )

    assert recovered.action == CampaignAction.RECOVER
    assert recovered.selected_idea.idea_id == selected.selected_idea.idea_id
    assert recovered.selected_idea.experiment_node_id == 0
    assert recovered.batch_id == selected.batch_id
    assert recovered.resolved_parent.node_id == 0
    assert tuple(runner.roles) == calls_before_recovery
    assert len(archive.state.batches) == 1
