"""Behavioral tests for evidence-audited coding-agent selection."""

import json
from dataclasses import replace

import pytest

from kapso.execution.search_strategies.generic.ideation.generator import (
    GenerationMemberSettings,
)
from kapso.execution.search_strategies.generic.ideation.selector import (
    CandidateSelector,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CandidateAnalysis,
    CandidateDispositionKind,
    ClaimKind,
    CodingAgentCallResult,
    EvidenceClaim,
    EvidenceStatus,
    new_identifier,
)
from test_ideation_analyzer import candidate, context
from test_ideation_domain import BATCH_ID, CLAIM_ID, NOW


class FakeRunner:
    def __init__(self, output):
        self.output = output
        self.requests = []

    def run(self, request, response_schema):
        self.requests.append((request, response_schema))
        return CodingAgentCallResult(
            output=json.dumps(self.output),
            duration_seconds=1,
            cost_usd=None,
            input_tokens=10,
            output_tokens=8,
            artifacts=("/tmp/selector.json",),
        )


class FailedRunner:
    def run(self, request, response_schema):
        raise RuntimeError("selector failed")


def settings(cli="codex"):
    return GenerationMemberSettings(
        cli=cli,
        model="selector-model",
        timeout_seconds=30,
        effort="high",
        allowed_tools=("Read",),
    )


def output(selected_id, fallback_ids=(), rejected=(), audit=(), overrides=()):
    return {
        "selected_idea_id": selected_id,
        "fallback_idea_ids": list(fallback_ids),
        "rejected_ideas": list(rejected),
        "diagnosis_audit": list(audit),
        "hard_rule_results": ["All deterministic eligibility rules passed."],
        "gap_decisions": ["No reserved evaluation gap was displaced."],
        "duplicate_overrides": list(overrides),
        "decision_summary": "Best evidence-adjusted expected value.",
        "expected_benefit": 0.1,
        "expected_cost": 1.0,
    }


def test_selector_sees_only_eligible_candidate_content_and_covers_every_idea(
    tmp_path,
):
    archive, snapshot, search_directive = context(tmp_path)
    first = candidate(tmp_path, snapshot, search_directive)
    second = replace(
        candidate(tmp_path, snapshot, search_directive),
        idea_id=new_identifier("idea"),
        proposal="A second eligible intervention.",
    )
    invalid = replace(
        candidate(tmp_path, snapshot, search_directive),
        idea_id=new_identifier("idea"),
        proposal="An ineligible intervention.",
    )
    analyses = (
        CandidateAnalysis(idea_id=first.idea_id, eligible=True),
        CandidateAnalysis(idea_id=second.idea_id, eligible=True),
        CandidateAnalysis(
            idea_id=invalid.idea_id,
            eligible=False,
            hard_failures=("capacity_rule_failed",),
        ),
    )
    fake = FakeRunner(output(first.idea_id, (second.idea_id,)))
    selector = CandidateSelector(fake, settings("claude_code"))

    result = selector.select(
        batch_id=BATCH_ID,
        problem_statement="the complete problem statement",
        evidence_snapshot=snapshot,
        directive=search_directive,
        candidates=(first, second, invalid),
        analyses=analyses,
        workspace=str(tmp_path),
    )

    assert result.decision.selected_idea_id == first.idea_id
    assert result.decision.fallback_idea_ids == (second.idea_id,)
    assert tuple(
        disposition.disposition for disposition in result.decision.dispositions
    ) == (
        CandidateDispositionKind.SELECTED,
        CandidateDispositionKind.DEFERRED,
        CandidateDispositionKind.INVALID,
    )
    request, schema = fake.requests[0]
    packet = json.loads(request.prompt.split("Mandatory packet:\n\n", 1)[1])
    assert [item["idea_id"] for item in packet["eligible_candidates"]] == [
        first.idea_id,
        second.idea_id,
    ]
    assert invalid.idea_id not in {
        item["idea_id"] for item in packet["eligible_candidates"]
    }
    assert {item["idea_id"] for item in packet["candidate_analyses"]} == {
        first.idea_id,
        second.idea_id,
        invalid.idea_id,
    }
    assert request.cli == "claude_code"
    assert "`gap_decisions` entry" in request.prompt
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("unknown_position", ["selected", "fallback", "rejected"])
def test_selector_cannot_return_unknown_or_uncovered_eligible_id(
    tmp_path,
    unknown_position,
):
    archive, snapshot, search_directive = context(tmp_path)
    first = candidate(tmp_path, snapshot, search_directive)
    second = replace(first, idea_id=new_identifier("idea"), proposal="second")
    unknown = new_identifier("idea")
    selected = unknown if unknown_position == "selected" else first.idea_id
    fallbacks = (unknown,) if unknown_position == "fallback" else ()
    rejected = (
        ({"idea_id": unknown, "reason": "negative evidence"},)
        if unknown_position == "rejected"
        else ()
    )
    selector = CandidateSelector(
        FakeRunner(output(selected, fallbacks, rejected)),
        settings(),
    )
    with pytest.raises(ValueError, match="eligible pool"):
        selector.select(
            batch_id=BATCH_ID,
            problem_statement="problem",
            evidence_snapshot=snapshot,
            directive=search_directive,
            candidates=(first, second),
            analyses=(
                CandidateAnalysis(idea_id=first.idea_id, eligible=True),
                CandidateAnalysis(idea_id=second.idea_id, eligible=True),
            ),
            workspace=str(tmp_path),
        )


def test_selector_failure_propagates_without_a_fallback_winner(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    first = candidate(tmp_path, snapshot, search_directive)
    selector = CandidateSelector(FailedRunner(), settings())
    with pytest.raises(RuntimeError, match="selector failed"):
        selector.select(
            batch_id=BATCH_ID,
            problem_statement="problem",
            evidence_snapshot=snapshot,
            directive=search_directive,
            candidates=(first,),
            analyses=(CandidateAnalysis(idea_id=first.idea_id, eligible=True),),
            workspace=str(tmp_path),
        )


def test_selector_diagnosis_must_match_frozen_claim_status_and_sources(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    claim = EvidenceClaim(
        claim_id=CLAIM_ID,
        statement="The mechanism is supported.",
        kind=ClaimKind.HYPOTHESIS,
        status=EvidenceStatus.SUPPORTED,
        source_refs=("evaluation:support",),
        affected_idea_ids=(),
        affected_experiment_node_ids=(),
        updated_at=NOW,
    )
    snapshot = replace(snapshot, claims=(claim,))
    first = candidate(
        tmp_path,
        snapshot,
        search_directive,
        claim_ids=(CLAIM_ID,),
        evidence_refs=("evaluation:support",),
    )
    wrong_audit = (
        {
            "claim_id": CLAIM_ID,
            "status": EvidenceStatus.CONTRADICTED.value,
            "evidence_refs": ["evaluation:support"],
        },
    )
    selector = CandidateSelector(
        FakeRunner(output(first.idea_id, audit=wrong_audit)),
        settings(),
    )
    with pytest.raises(ValueError, match="status contradicts"):
        selector.select(
            batch_id=BATCH_ID,
            problem_statement="problem",
            evidence_snapshot=snapshot,
            directive=search_directive,
            candidates=(first,),
            analyses=(CandidateAnalysis(idea_id=first.idea_id, eligible=True),),
            workspace=str(tmp_path),
        )


def test_selector_diagnosis_cannot_add_unselected_claims(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    claim = EvidenceClaim(
        claim_id=CLAIM_ID,
        statement="A known claim that the selected candidate does not use.",
        kind=ClaimKind.HYPOTHESIS,
        status=EvidenceStatus.SUPPORTED,
        source_refs=("evaluation:support",),
        affected_idea_ids=(),
        affected_experiment_node_ids=(),
        updated_at=NOW,
    )
    snapshot = replace(snapshot, claims=(claim,))
    first = candidate(tmp_path, snapshot, search_directive)
    extra_audit = (
        {
            "claim_id": CLAIM_ID,
            "status": EvidenceStatus.SUPPORTED.value,
            "evidence_refs": ["evaluation:support"],
        },
    )
    selector = CandidateSelector(
        FakeRunner(output(first.idea_id, audit=extra_audit)),
        settings(),
    )

    with pytest.raises(ValueError, match="exactly cover selected claim ids"):
        selector.select(
            batch_id=BATCH_ID,
            problem_statement="problem",
            evidence_snapshot=snapshot,
            directive=search_directive,
            candidates=(first,),
            analyses=(CandidateAnalysis(idea_id=first.idea_id, eligible=True),),
            workspace=str(tmp_path),
        )


def test_selected_changed_duplicate_requires_explicit_override(tmp_path):
    archive, snapshot, search_directive = context(tmp_path)
    first = candidate(tmp_path, snapshot, search_directive)
    previous_id = new_identifier("idea")
    analysis = CandidateAnalysis(
        idea_id=first.idea_id,
        eligible=True,
        exact_duplicate_of=previous_id,
        exact_duplicate_changed_conditions=("evaluation_method_changed",),
    )
    selector = CandidateSelector(FakeRunner(output(first.idea_id)), settings())
    with pytest.raises(ValueError, match="explicit override"):
        selector.select(
            batch_id=BATCH_ID,
            problem_statement="problem",
            evidence_snapshot=snapshot,
            directive=search_directive,
            candidates=(first,),
            analyses=(analysis,),
            workspace=str(tmp_path),
        )
