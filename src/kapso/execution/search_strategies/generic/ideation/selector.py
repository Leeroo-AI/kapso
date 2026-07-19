"""Evidence-audited selection over the deterministic eligible candidate pool."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from kapso.execution.search_strategies.generic.ideation.coding_agents import (
    CodingAgentCallRunner,
)
from kapso.execution.search_strategies.generic.ideation.generator import (
    GenerationMemberSettings,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CampaignEvidenceSnapshot,
    CandidateAnalysis,
    CandidateDisposition,
    CandidateDispositionKind,
    CodingAgentCallRequest,
    CodingAgentCallResult,
    DiagnosisAudit,
    EvidenceStatus,
    IdeaRecord,
    SearchDirective,
    SelectionDecision,
)

_SELECTOR_FIELDS = {
    "selected_idea_id",
    "fallback_idea_ids",
    "rejected_ideas",
    "diagnosis_audit",
    "hard_rule_results",
    "gap_decisions",
    "duplicate_overrides",
    "decision_summary",
    "expected_benefit",
    "expected_cost",
}

SELECTOR_RESPONSE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_SELECTOR_FIELDS),
    "properties": {
        "selected_idea_id": {"type": "string", "minLength": 1},
        "fallback_idea_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "rejected_ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["idea_id", "reason"],
                "properties": {
                    "idea_id": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1},
                },
            },
        },
        "diagnosis_audit": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_id", "status", "evidence_refs"],
                "properties": {
                    "claim_id": {"type": "string", "minLength": 1},
                    "status": {
                        "type": "string",
                        "enum": [status.value for status in EvidenceStatus],
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                    },
                },
            },
        },
        "hard_rule_results": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "gap_decisions": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "duplicate_overrides": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "decision_summary": {"type": "string", "minLength": 1},
        "expected_benefit": {"type": "number"},
        "expected_cost": {"type": "number", "minimum": 0},
    },
}


@dataclass(frozen=True)
class SelectionResult:
    decision: SelectionDecision
    call: CodingAgentCallResult


class CandidateSelector:
    """Select only after deterministic rules have removed ineligible ideas."""

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
                Path(__file__).parent.parent / "prompts" / "ideation_v3_selector.md"
            ).read_text(encoding="utf-8")
        )
        if self.prompt_template.count("{{MANDATORY_PACKET}}") != 1:
            raise ValueError("selector prompt requires one mandatory packet marker")

    def select(
        self,
        *,
        problem_statement: str,
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
        candidates: Sequence[IdeaRecord],
        analyses: Sequence[CandidateAnalysis],
        workspace: str,
    ) -> SelectionResult:
        pool = tuple(candidates)
        analysis_pool = tuple(analyses)
        if not pool:
            raise ValueError("selector candidate pool must not be empty")
        candidate_by_id = {candidate.idea_id: candidate for candidate in pool}
        analysis_by_id = {analysis.idea_id: analysis for analysis in analysis_pool}
        if len(candidate_by_id) != len(pool) or len(analysis_by_id) != len(
            analysis_pool
        ):
            raise ValueError("selector candidate and analysis ids must be unique")
        if set(candidate_by_id) != set(analysis_by_id):
            raise ValueError("selector requires analysis for every candidate")
        eligible = tuple(
            candidate
            for candidate in pool
            if analysis_by_id[candidate.idea_id].eligible
        )
        if not eligible:
            raise ValueError("selector has no eligible candidates")
        prompt = self._prompt(
            problem_statement=problem_statement,
            evidence_snapshot=evidence_snapshot,
            directive=directive,
            eligible=eligible,
            analyses=analysis_pool,
        )
        call = self.runner.run(
            CodingAgentCallRequest(
                role="candidate_selector",
                cli=self.settings.cli,
                model=self.settings.model,
                prompt=prompt,
                workspace=workspace,
                timeout_seconds=self.settings.timeout_seconds,
                effort=self.settings.effort,
                allowed_tools=self.settings.allowed_tools,
            ),
            SELECTOR_RESPONSE_SCHEMA,
        )
        parsed = json.loads(call.output)
        if not isinstance(parsed, dict) or set(parsed) != _SELECTOR_FIELDS:
            raise ValueError("coding-agent selector fields are invalid")
        if not parsed["hard_rule_results"] or not parsed["gap_decisions"]:
            raise ValueError("selector must report hard rules and gap decisions")
        decision = self._decision(
            parsed=parsed,
            evidence_snapshot=evidence_snapshot,
            candidates=pool,
            analysis_by_id=analysis_by_id,
            selection_artifacts=call.artifacts,
        )
        return SelectionResult(decision=decision, call=call)

    def _decision(
        self,
        *,
        parsed: Mapping[str, Any],
        evidence_snapshot: CampaignEvidenceSnapshot,
        candidates: Tuple[IdeaRecord, ...],
        analysis_by_id: Mapping[str, CandidateAnalysis],
        selection_artifacts: Tuple[str, ...],
    ) -> SelectionDecision:
        eligible_ids = {
            idea_id for idea_id, analysis in analysis_by_id.items() if analysis.eligible
        }
        selected_id = parsed["selected_idea_id"]
        fallback_ids = tuple(parsed["fallback_idea_ids"])
        rejected = tuple(parsed["rejected_ideas"])
        if not all(
            isinstance(item, dict) and set(item) == {"idea_id", "reason"}
            for item in rejected
        ):
            raise ValueError("selector rejected idea fields are invalid")
        rejected_ids = tuple(item["idea_id"] for item in rejected)
        ranked_ids = (selected_id,) + fallback_ids + rejected_ids
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError("selector result contains duplicate idea ids")
        if set(ranked_ids) != eligible_ids:
            raise ValueError("selector result must cover exactly the eligible pool")

        candidate_by_id = {candidate.idea_id: candidate for candidate in candidates}
        diagnosis = tuple(
            DiagnosisAudit.from_dict(item) for item in parsed["diagnosis_audit"]
        )
        claims_by_id = {claim.claim_id: claim for claim in evidence_snapshot.claims}
        audit_by_id = {audit.claim_id: audit for audit in diagnosis}
        if len(audit_by_id) != len(diagnosis):
            raise ValueError("selector diagnosis audit ids must be unique")
        selected_claim_ids = set(candidate_by_id[selected_id].claim_ids)
        if not selected_claim_ids.issubset(audit_by_id):
            raise ValueError("selected material claims require diagnosis audit")
        for audit in diagnosis:
            claim = claims_by_id.get(audit.claim_id)
            if claim is None:
                raise ValueError("selector diagnosis audit references unknown claim")
            if audit.status != claim.status:
                raise ValueError("selector diagnosis status contradicts evidence")
            if not set(audit.evidence_refs).issubset(claim.source_refs):
                raise ValueError("selector diagnosis sources contradict evidence")
        if (
            analysis_by_id[selected_id].exact_duplicate_of is not None
            and not parsed["duplicate_overrides"]
        ):
            raise ValueError("selected duplicate requires an explicit override")

        rejected_reason_by_id = {item["idea_id"]: item["reason"] for item in rejected}
        fallback_rank = {
            idea_id: index + 1 for index, idea_id in enumerate(fallback_ids)
        }
        dispositions = []
        for candidate in candidates:
            analysis = analysis_by_id[candidate.idea_id]
            if not analysis.eligible:
                dispositions.append(
                    CandidateDisposition(
                        idea_id=candidate.idea_id,
                        disposition=CandidateDispositionKind.INVALID,
                        reason="; ".join(analysis.hard_failures),
                    )
                )
            elif candidate.idea_id == selected_id:
                dispositions.append(
                    CandidateDisposition(
                        idea_id=candidate.idea_id,
                        disposition=CandidateDispositionKind.SELECTED,
                        reason=parsed["decision_summary"],
                    )
                )
            elif candidate.idea_id in fallback_rank:
                dispositions.append(
                    CandidateDisposition(
                        idea_id=candidate.idea_id,
                        disposition=CandidateDispositionKind.DEFERRED,
                        reason=(
                            f"Fallback rank {fallback_rank[candidate.idea_id]}: "
                            f"{parsed['decision_summary']}"
                        ),
                    )
                )
            else:
                dispositions.append(
                    CandidateDisposition(
                        idea_id=candidate.idea_id,
                        disposition=CandidateDispositionKind.REJECTED,
                        reason=rejected_reason_by_id[candidate.idea_id],
                    )
                )
        return SelectionDecision(
            selected_idea_id=selected_id,
            fallback_idea_ids=fallback_ids,
            dispositions=tuple(dispositions),
            diagnosis_audit=diagnosis,
            hard_rule_results=tuple(parsed["hard_rule_results"]),
            gap_decisions=tuple(parsed["gap_decisions"]),
            duplicate_overrides=tuple(parsed["duplicate_overrides"]),
            decision_summary=parsed["decision_summary"],
            selection_artifacts=selection_artifacts,
            expected_benefit=parsed["expected_benefit"],
            expected_cost=parsed["expected_cost"],
        )

    def _prompt(
        self,
        *,
        problem_statement: str,
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
        eligible: Tuple[IdeaRecord, ...],
        analyses: Tuple[CandidateAnalysis, ...],
    ) -> str:
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ValueError("selector problem statement must be non-empty")
        packet = json.dumps(
            {
                "problem_statement": problem_statement,
                "evidence_snapshot": evidence_snapshot.to_dict(),
                "search_directive": directive.to_dict(),
                "eligible_candidates": [candidate.to_dict() for candidate in eligible],
                "candidate_analyses": [analysis.to_dict() for analysis in analyses],
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        return self.prompt_template.replace("{{MANDATORY_PACKET}}", packet)
