"""Behavioral tests for the ideation orchestration boundary."""

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
        if request.role == "candidate_selector":
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


def build_engine(tmp_path, runner, *, minimum_distinct=2):
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
            embedding_provider=None,
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
    archive, engine = build_engine(tmp_path, runner)
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
