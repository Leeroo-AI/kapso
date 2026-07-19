"""Behavioral tests for independent structured candidate generation."""

import json
import threading
from dataclasses import replace

import pytest

from kapso.execution.search_strategies.generic.ideation.archive import IdeaArchive
from kapso.execution.search_strategies.generic.ideation.evidence import (
    CampaignEvidenceBuilder,
)
from kapso.execution.search_strategies.generic.ideation.generator import (
    CandidateGenerator,
    CandidateGeneratorSettings,
    GenerationMemberSettings,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CodingAgentCallResult,
    IdeaDescriptor,
    OperatorBrief,
    OperatorKind,
    ObjectiveDirection,
    ParentPlan,
    ParentPlanKind,
)
from test_ideation_domain import BATCH_ID, NOW, directive, resolved_parent
from test_ideation_evidence import capacity, evidence_settings


def payload(descriptor: IdeaDescriptor, proposal: str = "Measure one intervention"):
    return {
        "proposal": proposal,
        "directive_rationale": "This tests the assigned evidence-backed operator.",
        "descriptor": descriptor.to_dict(),
        "assumptions": ["The intervention is independently measurable."],
        "evidence_refs": [],
        "claim_ids": [],
        "resolves_claim_ids": [],
        "expected_observations": ["Comparable utility changes."],
        "evaluation_method": "Run the canonical evaluator once.",
        "resource_request": "One complete experiment.",
        "predicted_gain": 0.02,
        "predicted_cost": 1.0,
        "confidence": 0.6,
        "claimed_nearest_idea_id": None,
        "claimed_nearest_experiment_node_id": None,
    }


class FakeRunner:
    def __init__(self, outputs):
        self.outputs = outputs
        self.requests = []
        self.lock = threading.Lock()

    def run(self, request, response_schema):
        with self.lock:
            self.requests.append((request, response_schema))
        return CodingAgentCallResult(
            output=json.dumps(self.outputs[request.role]),
            duration_seconds=1,
            cost_usd=None,
            input_tokens=10,
            output_tokens=5,
            artifacts=(f"/tmp/{request.role}.json",),
        )


class FailedRunner:
    def run(self, request, response_schema):
        raise RuntimeError("agent failed")


def member(cli="codex") -> GenerationMemberSettings:
    return GenerationMemberSettings(
        cli=cli,
        model="test-model",
        timeout_seconds=30,
        effort="high",
        allowed_tools=("Read",),
    )


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


def test_every_member_receives_full_common_evidence_and_distinct_brief(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    second_descriptor = IdeaDescriptor(
        approach_family="orthogonal_family",
        intervention_target="dominant_failure",
        mechanism="mechanism_shift",
        expected_effect="escape_plateau",
    )
    second_brief = OperatorBrief(
        operator=OperatorKind.MECHANISM_SHIFT,
        rationale="Test an orthogonal mechanism.",
        descriptor_target=second_descriptor,
        parent_plan=ParentPlan(kind=ParentPlanKind.BASELINE),
    )
    search_directive = replace(
        search_directive,
        operator_briefs=(search_directive.operator_briefs[0], second_brief),
        candidate_quota=2,
    )
    outputs = {
        "candidate_0": payload(search_directive.operator_briefs[0].descriptor_target),
        "candidate_1": payload(second_descriptor, "Try an orthogonal mechanism"),
    }
    fake = FakeRunner(outputs)
    generator = CandidateGenerator(
        fake,
        CandidateGeneratorSettings(
            members=(member("codex"), member("claude_code")),
            repair_member=member(),
        ),
    )
    problem = "full problem beginning\n" + "important detail " * 200 + "full ending"

    generated = generator.generate(
        batch_id=BATCH_ID,
        problem_statement=problem,
        evidence_snapshot=snapshot,
        directive=search_directive,
        archive_state=archive.state,
        resolved_parents=(resolved_parent(), resolved_parent()),
        workspaces=(str(tmp_path), str(tmp_path)),
    )

    assert tuple(item.idea.operator for item in generated) == (
        OperatorKind.INDEPENDENT_DRAFT,
        OperatorKind.MECHANISM_SHIFT,
    )
    assert tuple(item.idea.descriptor for item in generated) == (
        search_directive.operator_briefs[0].descriptor_target,
        second_descriptor,
    )
    assert {request.role for request, _ in fake.requests} == {
        "candidate_0",
        "candidate_1",
    }
    for request, schema in fake.requests:
        packet = json.loads(request.prompt.split("Mandatory packet:\n\n", 1)[1])
        assert packet["problem_statement"] == problem
        assert packet["evidence_snapshot"]["snapshot_id"] == snapshot.snapshot_id
        assert packet["prior_ideas"] == []
        assert schema["additionalProperties"] is False
    assert generated[0].idea.generation_artifacts == ("/tmp/candidate_0.json",)


def test_malformed_structured_candidate_fails_without_salvage(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    malformed = payload(search_directive.operator_briefs[0].descriptor_target)
    malformed["unknown"] = "must not be accepted"
    generator = CandidateGenerator(
        FakeRunner({"candidate_0": malformed}),
        CandidateGeneratorSettings(
            members=(member(),),
            repair_member=member(),
        ),
    )

    with pytest.raises(ValueError, match="fields"):
        generator.generate(
            batch_id=BATCH_ID,
            problem_statement="full problem",
            evidence_snapshot=snapshot,
            directive=search_directive,
            archive_state=archive.state,
            resolved_parents=(resolved_parent(),),
            workspaces=(str(tmp_path),),
        )


def test_member_failure_propagates_and_batch_remains_unmodified(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    generator = CandidateGenerator(
        FailedRunner(),
        CandidateGeneratorSettings(
            members=(member(),),
            repair_member=member(),
        ),
    )
    with pytest.raises(RuntimeError, match="agent failed"):
        generator.generate(
            batch_id=BATCH_ID,
            problem_statement="full problem",
            evidence_snapshot=snapshot,
            directive=search_directive,
            archive_state=archive.state,
            resolved_parents=(resolved_parent(),),
            workspaces=(str(tmp_path),),
        )
    assert archive.state.batches == ()
    assert archive.state.ideas == ()


def test_repair_requires_explicit_single_call_authority(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    fake = FakeRunner(
        {
            "diversity_repair": payload(
                search_directive.operator_briefs[0].descriptor_target
            )
        }
    )
    generator = CandidateGenerator(
        fake,
        CandidateGeneratorSettings(
            members=(member(),),
            repair_member=member("claude_code"),
        ),
    )
    generated = generator.generate_repair(
        batch_id=BATCH_ID,
        problem_statement="full problem",
        evidence_snapshot=snapshot,
        directive=search_directive,
        archive_state=archive.state,
        operator_brief=search_directive.operator_briefs[0],
        resolved_parent=resolved_parent(),
        repair_request={"missing": "descriptor coverage"},
        workspace=str(tmp_path),
    )
    assert generated.idea.operator == OperatorKind.INDEPENDENT_DRAFT
    assert len(fake.requests) == 1
    assert (
        '"repair_request": {"missing": "descriptor coverage"}'
        in fake.requests[0][0].prompt
    )


def test_generation_settings_reject_non_list_tool_configuration():
    with pytest.raises(ValueError, match="allowed tools"):
        GenerationMemberSettings.from_dict(
            {
                "cli": "codex",
                "model": "test-model",
                "timeout_seconds": 30,
                "effort": "high",
                "allowed_tools": "Read",
            }
        )
