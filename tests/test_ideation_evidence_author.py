"""Behavioral tests for built-in causal evidence authoring."""

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.iteration_evaluator import (
    IterationEvaluationError,
    IterationEvaluationResult,
)
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation import (
    EVIDENCE_AUTHOR_METADATA_KEY,
    EVIDENCE_AUTHOR_RESPONSE_SCHEMA,
    EVALUATOR_EVIDENCE_KEY,
    CodingAgentCallResult,
    EvidenceAuthor,
    GenerationMemberSettings,
    IdeaArchive,
    IdeaStatus,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from test_ideation_domain import BATCH_ID, NOW, generated_idea

COMMIT_SHA = "a" * 40
EVALUATOR_ID = "registered-evaluator-v1"


def member() -> GenerationMemberSettings:
    return GenerationMemberSettings(
        cli="codex",
        model="test-model",
        timeout_seconds=30,
        effort="high",
        allowed_tools=("Read",),
    )


def linked_idea(*, target_gap_ids=()):
    return replace(
        generated_idea(),
        status=IdeaStatus.IMPLEMENTING,
        selected_in_batch_id=BATCH_ID,
        selection_reason="Selected by the audited selector.",
        experiment_node_id=0,
        target_gap_ids=target_gap_ids,
    )


def evaluated_node(*, with_attempt=True) -> SearchNode:
    node = SearchNode(
        node_id=0,
        idea_id=generated_idea().idea_id,
        selection_batch_id=BATCH_ID,
        solution=generated_idea().proposal,
        branch_name="generic_exp_0",
        code_diff="diff --git a/train.py b/train.py\n+clip_grad_norm_(parameters, 1.0)\n",
        evaluation_output="registered evaluator: stability=0.91",
        feedback="Gradient variance fell while validation utility improved.",
        technical_difficulties="No implementation failure.",
        score=0.91,
        evaluation_valid=True,
        started_at=NOW,
    )
    if with_attempt:
        node.evaluation_attempts.append(
            EvaluationAttempt(
                commit_sha=COMMIT_SHA,
                evaluator_id=EVALUATOR_ID,
                fidelity="full",
                fraction=1.0,
                seed=17,
                score=0.91,
            )
        )
    return node


def source_refs(packet):
    references = packet["allowed_source_refs"]
    return [
        next(reference for reference in references if reference.endswith(":code_diff")),
        next(reference for reference in references if reference.endswith(":feedback")),
    ]


def supported_payload(packet):
    references = source_refs(packet)
    return {
        "claims": [
            {
                "statement": "Gradient clipping caused the measured stability gain.",
                "kind": "hypothesis",
                "status": "supported",
                "source_refs": references,
            }
        ],
        "open_gaps": [
            {
                "axis": "seed stability",
                "description": "The mechanism has not been replicated across seeds.",
                "evidence_refs": references,
                "impact": 0.8,
                "uncertainty": 0.9,
                "estimated_cost": 1.0,
            }
        ],
        "targeted_gap_updates": [],
    }


class PacketRunner:
    def __init__(self, output_builder):
        self.output_builder = output_builder
        self.requests = []

    def run(self, request, response_schema):
        packet = json.loads(request.prompt.split("Mandatory packet:\n\n", 1)[1])
        self.requests.append((request, response_schema, packet))
        output = self.output_builder(packet)
        return CodingAgentCallResult(
            output=json.dumps(output),
            duration_seconds=1.5,
            cost_usd=None,
            input_tokens=20,
            output_tokens=10,
            artifacts=("/tmp/evidence-author/final.json",),
        )


def author(runner):
    return EvidenceAuthor(runner, member())


def invoke(author_instance, archive, *, idea=None, node=None):
    return author_instance.author(
        problem_statement="full beginning\n"
        + "important detail " * 100
        + "full ending",
        idea=linked_idea() if idea is None else idea,
        node=evaluated_node() if node is None else node,
        archive_state=archive.state,
        workspace=str(archive.path.parent),
        current_commit_sha=COMMIT_SHA,
        evaluator_id=EVALUATOR_ID,
    )


def test_author_receives_full_provenance_and_returns_validated_metadata(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")
    runner = PacketRunner(supported_payload)
    result = invoke(author(runner), archive)
    request, schema, packet = runner.requests[0]

    assert packet["problem_statement"].endswith("full ending")
    assert "code_diff" not in packet["experiment"]
    assert "evaluation_output" not in packet["experiment"]
    assert "external_evaluation_metadata" not in packet["experiment"]
    assert packet["source_material"][source_refs(packet)[0]].endswith(
        "clip_grad_norm_(parameters, 1.0)\n"
    )
    assert packet["registered_evaluation_available"] is True
    assert schema == EVIDENCE_AUTHOR_RESPONSE_SCHEMA
    assert schema["additionalProperties"] is False
    assert request.role == "evidence_author"
    assert result.metadata[EVALUATOR_EVIDENCE_KEY] == result.evidence
    assert result.metadata[EVIDENCE_AUTHOR_METADATA_KEY] == {
        "operation_id": request.operation_id,
        "artifacts": ["/tmp/evidence-author/final.json"],
    }

    repeated = invoke(author(runner), archive)
    assert repeated.operation_id == result.operation_id


def test_empty_findings_are_valid_without_registered_evaluation(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")
    runner = PacketRunner(
        lambda _packet: {
            "claims": [],
            "open_gaps": [],
            "targeted_gap_updates": [],
        }
    )

    result = invoke(author(runner), archive, node=evaluated_node(with_attempt=False))

    assert result.evidence == {
        "claims": [],
        "open_gaps": [],
        "targeted_gap_updates": [],
    }
    assert runner.requests[0][2]["registered_evaluation_available"] is False


def test_unregistered_score_cannot_author_claims_or_gap_updates(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")
    runner = PacketRunner(supported_payload)

    with pytest.raises(ValueError, match="registered evaluation"):
        invoke(author(runner), archive, node=evaluated_node(with_attempt=False))


def test_open_gap_is_allowed_without_registered_evaluation(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")

    def open_gap_payload(packet):
        return {
            "claims": [],
            "open_gaps": [
                {
                    "axis": "evaluation coverage",
                    "description": "The implementation lacks multi-seed evaluation.",
                    "evidence_refs": source_refs(packet),
                    "impact": 0.7,
                    "uncertainty": 0.8,
                    "estimated_cost": 1.0,
                }
            ],
            "targeted_gap_updates": [],
        }

    result = invoke(
        author(PacketRunner(open_gap_payload)),
        archive,
        node=evaluated_node(with_attempt=False),
    )

    assert len(result.evidence["open_gaps"]) == 1


@pytest.mark.parametrize("violation", ("invented_ref", "missing_diff"))
def test_every_finding_requires_exact_diff_and_outcome_provenance(
    tmp_path,
    violation,
):
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")

    def invalid_payload(packet):
        payload = supported_payload(packet)
        if violation == "invented_ref":
            payload["claims"][0]["source_refs"].append("invented:source")
        else:
            payload["claims"][0]["source_refs"] = [source_refs(packet)[1]]
        return payload

    with pytest.raises(ValueError, match="allowed set|code diff"):
        invoke(author(PacketRunner(invalid_payload)), archive)


def test_targeted_gap_updates_must_name_a_gap_targeted_by_the_idea(tmp_path):
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")

    def invalid_target(packet):
        return {
            "claims": [],
            "open_gaps": [],
            "targeted_gap_updates": [
                {
                    "gap_id": "gap_" + "f" * 32,
                    "state": "closed",
                    "evidence_refs": source_refs(packet),
                    "closure_reason": "The registered evaluation resolved it.",
                }
            ],
        }

    with pytest.raises(ValueError, match="not targeted"):
        invoke(author(PacketRunner(invalid_target)), archive)


class MaterializingWorkspace:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def materialize_ref(self, _ref):
        yield self.path


def evaluator_orchestrator(tmp_path, evaluator):
    orchestrator = OrchestratorAgent.__new__(OrchestratorAgent)
    orchestrator.iteration_evaluator = evaluator
    orchestrator.iteration_evaluator_failure_policy = "raise"
    orchestrator.goal = "Improve the complete task solution."
    orchestrator.search_strategy = SimpleNamespace(
        workspace=MaterializingWorkspace(tmp_path)
    )
    return orchestrator


def authored_node() -> SearchNode:
    node = evaluated_node()
    node.external_evaluation_metadata = {
        EVALUATOR_EVIDENCE_KEY: {
            "claims": [],
            "open_gaps": [],
            "targeted_gap_updates": [],
        },
        EVIDENCE_AUTHOR_METADATA_KEY: {
            "operation_id": "agent_call_" + "a" * 32,
            "artifacts": ["/tmp/evidence-author/final.json"],
        },
    }
    return node


def test_external_metadata_preserves_evidence_unless_explicitly_overridden(tmp_path):
    node = authored_node()
    original_evidence = node.external_evaluation_metadata[EVALUATOR_EVIDENCE_KEY]
    orchestrator = evaluator_orchestrator(
        tmp_path,
        lambda _context: IterationEvaluationResult(
            metrics={"holdout": 0.9},
            metadata={"suite": "external-v1"},
        ),
    )

    orchestrator._evaluate_candidates([node], iteration=1)

    assert (
        node.external_evaluation_metadata[EVALUATOR_EVIDENCE_KEY] == original_evidence
    )
    assert node.external_evaluation_metadata[EVIDENCE_AUTHOR_METADATA_KEY][
        "operation_id"
    ].startswith("agent_call_")
    assert node.external_evaluation_metadata["suite"] == "external-v1"

    override = {
        "claims": [],
        "open_gaps": [
            {
                "axis": "external coverage",
                "description": "External evaluator found an uncovered slice.",
                "evidence_refs": ["external:slice"],
                "impact": 0.5,
                "uncertainty": 0.7,
                "estimated_cost": 1.0,
            }
        ],
        "targeted_gap_updates": [],
    }
    orchestrator.iteration_evaluator = lambda _context: IterationEvaluationResult(
        metrics={"holdout": 0.9},
        metadata={EVALUATOR_EVIDENCE_KEY: override},
    )

    orchestrator._evaluate_candidates([node], iteration=1)

    assert node.external_evaluation_metadata[EVALUATOR_EVIDENCE_KEY] == override
    assert EVIDENCE_AUTHOR_METADATA_KEY in node.external_evaluation_metadata


def test_external_evaluator_cannot_replace_author_provenance(tmp_path):
    node = authored_node()
    original_metadata = dict(node.external_evaluation_metadata)
    orchestrator = evaluator_orchestrator(
        tmp_path,
        lambda _context: IterationEvaluationResult(
            metrics={},
            metadata={EVIDENCE_AUTHOR_METADATA_KEY: {"forged": True}},
        ),
    )

    with pytest.raises(IterationEvaluationError, match="author provenance"):
        orchestrator._evaluate_candidates([node], iteration=1)

    assert node.external_evaluation_metadata == original_metadata


def test_external_evaluator_failure_preserves_built_in_evidence(tmp_path):
    node = authored_node()
    original_metadata = dict(node.external_evaluation_metadata)

    def failed_evaluator(_context):
        raise RuntimeError("external harness unavailable")

    orchestrator = evaluator_orchestrator(tmp_path, failed_evaluator)
    orchestrator.iteration_evaluator_failure_policy = "record"

    orchestrator._evaluate_candidates([node], iteration=1)

    assert node.external_evaluation_metadata == original_metadata
    assert "external harness unavailable" in node.external_evaluation_error


def test_evidence_author_telemetry_accumulates_across_same_node_recovery():
    node = SearchNode(node_id=0)
    call = CodingAgentCallResult(
        output="{}",
        duration_seconds=1.5,
        cost_usd=0.25,
        input_tokens=20,
        output_tokens=10,
        artifacts=("/tmp/evidence-author/final.json",),
    )

    GenericSearch._add_evidence_author_telemetry(node, call)
    GenericSearch._add_evidence_author_telemetry(node, call)

    assert node.phase_telemetry["evidence_author"] == {
        "cost_usd": 0.5,
        "duration_seconds": 3.0,
        "coding_agent_call_count": 2.0,
        "unpriced_coding_agent_call_count": 0.0,
        "input_tokens": 40.0,
        "output_tokens": 20.0,
    }
