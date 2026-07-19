"""Strict coding-agent authoring of causal ideation evidence."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation.archive import (
    IdeaArchiveState,
)
from kapso.execution.search_strategies.generic.ideation.coding_agents import (
    CodingAgentCallRunner,
)
from kapso.execution.search_strategies.generic.ideation.evaluator_evidence import (
    EVALUATOR_EVIDENCE_KEY,
    build_evaluator_evidence_writeback,
)
from kapso.execution.search_strategies.generic.ideation.generator import (
    GenerationMemberSettings,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    IdeaRecord,
    content_identifier,
)

EVIDENCE_AUTHOR_METADATA_KEY = "ideation_evidence_author"

_EVIDENCE_FIELDS = {"claims", "open_gaps", "targeted_gap_updates"}
_CLAIM_FIELDS = {"statement", "kind", "status", "source_refs"}
_OPEN_GAP_FIELDS = {
    "axis",
    "description",
    "evidence_refs",
    "impact",
    "uncertainty",
    "estimated_cost",
}
_TARGETED_GAP_UPDATE_FIELDS = {
    "gap_id",
    "state",
    "evidence_refs",
    "closure_reason",
}

EVIDENCE_AUTHOR_RESPONSE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_EVIDENCE_FIELDS),
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_CLAIM_FIELDS),
                "properties": {
                    "statement": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["observation", "hypothesis", "constraint"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["supported", "contradicted"],
                    },
                    "source_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "open_gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_OPEN_GAP_FIELDS),
                "properties": {
                    "axis": {"type": "string"},
                    "description": {"type": "string"},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "impact": {"type": "number"},
                    "uncertainty": {"type": "number"},
                    "estimated_cost": {"type": ["number", "null"]},
                },
            },
        },
        "targeted_gap_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_TARGETED_GAP_UPDATE_FIELDS),
                "properties": {
                    "gap_id": {"type": "string"},
                    "state": {
                        "type": "string",
                        "enum": ["inconclusive", "closed"],
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "closure_reason": {"type": ["string", "null"]},
                },
            },
        },
    },
}


def _require_exact_fields(data: Any, expected: set[str], name: str) -> Dict[str, Any]:
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"{name} fields are invalid")
    return data


def _require_list(data: Any, name: str) -> list[Any]:
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    return data


def _source_references(
    node: SearchNode,
) -> Tuple[Dict[str, str], str | None, Tuple[str, ...]]:
    references: Dict[str, str] = {}
    code_diff_reference = None
    if node.code_diff:
        code_diff_reference = f"experiment_node:{node.node_id}:code_diff"
        references[code_diff_reference] = node.code_diff
    outcome_references = []
    for field_name, content in (
        ("evaluation_output", node.evaluation_output),
        ("feedback", node.feedback),
    ):
        if content:
            reference = f"experiment_node:{node.node_id}:{field_name}"
            references[reference] = content
            outcome_references.append(reference)
    if node.technical_difficulties:
        reference = f"experiment_node:{node.node_id}:technical_difficulties"
        references[reference] = node.technical_difficulties
    return references, code_diff_reference, tuple(outcome_references)


def _has_registered_evaluation(
    node: SearchNode,
    *,
    current_commit_sha: str,
    evaluator_id: str,
) -> bool:
    if (
        node.had_error
        or not node.evaluation_valid
        or node.score is None
        or not evaluator_id
        or not node.evaluation_attempts
    ):
        return False
    latest = node.evaluation_attempts[-1]
    return (
        latest.commit_sha == current_commit_sha
        and latest.evaluator_id == evaluator_id
        and latest.fidelity == node.eval_fidelity
        and math.isclose(latest.score, node.score, rel_tol=0.0, abs_tol=1e-12)
    )


def _validate_provenance(
    evidence: Dict[str, Any],
    *,
    allowed_references: frozenset[str],
    code_diff_reference: str | None,
    outcome_references: Tuple[str, ...],
    target_gap_ids: Tuple[str, ...],
    registered_evaluation_available: bool,
) -> None:
    entries = []
    for item in _require_list(evidence["claims"], "evidence author claims"):
        entries.append(
            (
                _require_exact_fields(item, _CLAIM_FIELDS, "evidence author claim"),
                "source_refs",
            )
        )
    for item in _require_list(evidence["open_gaps"], "evidence author open gaps"):
        entries.append(
            (
                _require_exact_fields(
                    item,
                    _OPEN_GAP_FIELDS,
                    "evidence author open gap",
                ),
                "evidence_refs",
            )
        )
    targeted_updates = _require_list(
        evidence["targeted_gap_updates"],
        "evidence author targeted gap updates",
    )
    for item in targeted_updates:
        update = _require_exact_fields(
            item,
            _TARGETED_GAP_UPDATE_FIELDS,
            "evidence author targeted gap update",
        )
        if update["gap_id"] not in target_gap_ids:
            raise ValueError("evidence author updated a gap not targeted by the idea")
        entries.append((update, "evidence_refs"))
    if (evidence["claims"] or targeted_updates) and not registered_evaluation_available:
        raise ValueError(
            "causal claims and targeted gap updates require a registered evaluation"
        )
    for item, references_field in entries:
        references = _require_list(item[references_field], references_field)
        if not all(
            isinstance(reference, str) and reference for reference in references
        ):
            raise ValueError(
                "evidence author source references must be non-empty strings"
            )
        if len(set(references)) != len(references):
            raise ValueError("evidence author source references must be unique")
        if not set(references).issubset(allowed_references):
            raise ValueError("evidence author cited a source outside the allowed set")
        if code_diff_reference is None or code_diff_reference not in references:
            raise ValueError("authored evidence must cite the experiment code diff")
        if not set(references).intersection(outcome_references):
            raise ValueError(
                "authored evidence must cite evaluation output or evaluator feedback"
            )


@dataclass(frozen=True)
class EvidenceAuthorResult:
    """Validated evidence plus durable coding-agent provenance."""

    evidence: Dict[str, Any]
    operation_id: str
    call: CodingAgentCallResult

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            EVALUATOR_EVIDENCE_KEY: self.evidence,
            EVIDENCE_AUTHOR_METADATA_KEY: {
                "operation_id": self.operation_id,
                "artifacts": list(self.call.artifacts),
            },
        }


class EvidenceAuthor:
    """Author evidence from the complete implementation and evaluation record."""

    def __init__(
        self,
        runner: CodingAgentCallRunner,
        settings: GenerationMemberSettings,
        prompt_template: str | None = None,
    ):
        self.runner = runner
        self.settings = settings
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else (
                Path(__file__).parent.parent
                / "prompts"
                / "ideation_v3_evidence_author.md"
            ).read_text(encoding="utf-8")
        )
        if self.prompt_template.count("{{MANDATORY_PACKET}}") != 1:
            raise ValueError(
                "evidence author prompt requires one mandatory packet marker"
            )

    def author(
        self,
        *,
        problem_statement: str,
        idea: IdeaRecord,
        node: SearchNode,
        archive_state: IdeaArchiveState,
        workspace: str,
        current_commit_sha: str,
        evaluator_id: str,
    ) -> EvidenceAuthorResult:
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ValueError("evidence author problem statement must be non-empty")
        if (
            node.idea_id != idea.idea_id
            or node.selection_batch_id != idea.selected_in_batch_id
            or node.node_id != idea.experiment_node_id
        ):
            raise ValueError("evidence author node and idea provenance disagree")
        references, code_diff_reference, outcome_references = _source_references(node)
        registered_evaluation_available = _has_registered_evaluation(
            node,
            current_commit_sha=current_commit_sha,
            evaluator_id=evaluator_id,
        )
        operation_id = content_identifier(
            "agent_call",
            hashlib.sha256(
                (
                    f"{idea.idea_id}:{node.node_id}:"
                    f"{node.execution_revision}:evidence_author"
                ).encode("utf-8")
            ).hexdigest(),
        )
        experiment = node.to_dict()
        for source_field in (
            "agent_output",
            "code_diff",
            "evaluation_output",
            "feedback",
            "technical_difficulties",
            "external_evaluation_metadata",
        ):
            del experiment[source_field]
        packet = json.dumps(
            {
                "problem_statement": problem_statement,
                "idea": idea.to_dict(),
                "experiment": experiment,
                "registered_evaluation_available": registered_evaluation_available,
                "allowed_source_refs": sorted(references),
                "source_material": references,
                "allowed_target_gap_ids": list(idea.target_gap_ids),
                "archive_claims": [claim.to_dict() for claim in archive_state.claims],
                "archive_gaps": [gap.to_dict() for gap in archive_state.gaps],
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        prompt = self.prompt_template.replace("{{MANDATORY_PACKET}}", packet)
        call = self.runner.run(
            CodingAgentCallRequest(
                operation_id=operation_id,
                role="evidence_author",
                cli=self.settings.cli,
                model=self.settings.model,
                prompt=prompt,
                workspace=workspace,
                timeout_seconds=self.settings.timeout_seconds,
                effort=self.settings.effort,
                allowed_tools=self.settings.allowed_tools,
            ),
            EVIDENCE_AUTHOR_RESPONSE_SCHEMA,
        )
        parsed = _require_exact_fields(
            json.loads(call.output),
            _EVIDENCE_FIELDS,
            "evidence author result",
        )
        _validate_provenance(
            parsed,
            allowed_references=frozenset(references),
            code_diff_reference=code_diff_reference,
            outcome_references=outcome_references,
            target_gap_ids=idea.target_gap_ids,
            registered_evaluation_available=registered_evaluation_available,
        )
        build_evaluator_evidence_writeback(
            {EVALUATOR_EVIDENCE_KEY: parsed},
            idea=idea,
            archive_state=archive_state,
            observed_at=node.started_at,
        )
        return EvidenceAuthorResult(
            evidence=parsed,
            operation_id=operation_id,
            call=call,
        )
