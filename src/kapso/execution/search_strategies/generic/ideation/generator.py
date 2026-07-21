"""Independent, schema-constrained candidate generation through coding agents."""

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from kapso.cross_run.knowledge.access import (
    PriorKnowledgeAccess,
    PriorKnowledgeAccessMaterialization,
)
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    CodingAgentCallRunner,
)
from kapso.execution.search_strategies.generic.ideation.archive import (
    IdeaArchiveState,
)
from kapso.execution.search_strategies.generic.ideation.evidence import (
    evidence_reference_ids,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    CampaignEvidenceSnapshot,
    IdeaDescriptor,
    IdeaRecord,
    OperatorBrief,
    ResolvedParentSnapshot,
    SearchDirective,
    content_identifier,
    new_identifier,
    utc_now,
)

_CANDIDATE_FIELDS = {
    "proposal",
    "directive_rationale",
    "descriptor",
    "assumptions",
    "evidence_refs",
    "claim_ids",
    "resolves_claim_ids",
    "expected_observations",
    "evaluation_method",
    "resource_request",
    "predicted_gain",
    "predicted_cost",
    "confidence",
    "claimed_nearest_idea_id",
    "claimed_nearest_experiment_node_id",
    "prior_knowledge_refs",
    "prior_adaptation_rationale",
}

CANDIDATE_RESPONSE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(_CANDIDATE_FIELDS),
    "properties": {
        "proposal": {"type": "string"},
        "directive_rationale": {"type": "string"},
        "descriptor": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "approach_family",
                "intervention_target",
                "mechanism",
                "expected_effect",
            ],
            "properties": {
                "approach_family": {"type": "string"},
                "intervention_target": {"type": "string"},
                "mechanism": {"type": "string"},
                "expected_effect": {"type": "string"},
            },
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
        },
        "claim_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "resolves_claim_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "expected_observations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evaluation_method": {"type": "string"},
        "resource_request": {"type": "string"},
        "predicted_gain": {"type": ["number", "null"]},
        "predicted_cost": {"type": ["number", "null"]},
        "confidence": {"type": ["number", "null"]},
        "claimed_nearest_idea_id": {"type": ["string", "null"]},
        "claimed_nearest_experiment_node_id": {
            "type": ["integer", "null"],
        },
        "prior_knowledge_refs": {
            "type": "array",
            "items": {"type": "string"},
        },
        "prior_adaptation_rationale": {"type": ["string", "null"]},
    },
}


@dataclass(frozen=True)
class GenerationMemberSettings:
    cli: str
    model: str
    timeout_seconds: float
    effort: str | None
    allowed_tools: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.cli not in {"codex", "claude_code"}:
            raise ValueError("generation member CLI must be codex or claude_code")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("generation member model must be non-empty")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("generation member timeout must be positive")
        if self.effort is not None and (
            not isinstance(self.effort, str) or not self.effort.strip()
        ):
            raise ValueError("generation member effort must be null or non-empty")
        if not isinstance(self.allowed_tools, (list, tuple)):
            raise ValueError("generation member tools must be an array")
        tools = tuple(self.allowed_tools)
        if any(not isinstance(tool, str) or not tool.strip() for tool in tools):
            raise ValueError("generation member tools must be non-empty strings")
        if len(tools) != len(set(tools)):
            raise ValueError("generation member tools must be unique")
        object.__setattr__(self, "allowed_tools", tools)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationMemberSettings":
        expected = {
            "cli",
            "model",
            "timeout_seconds",
            "effort",
            "allowed_tools",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("generation member settings fields are invalid")
        if not isinstance(data["allowed_tools"], list):
            raise ValueError("generation member allowed tools must be a list")
        return cls(
            cli=data["cli"],
            model=data["model"],
            timeout_seconds=data["timeout_seconds"],
            effort=data["effort"],
            allowed_tools=tuple(data["allowed_tools"]),
        )


@dataclass(frozen=True)
class CandidateGeneratorSettings:
    members: Tuple[GenerationMemberSettings, ...]
    repair_member: GenerationMemberSettings

    def __post_init__(self) -> None:
        if not self.members or not all(
            isinstance(member, GenerationMemberSettings) for member in self.members
        ):
            raise ValueError("candidate generator members are invalid")
        object.__setattr__(self, "members", tuple(self.members))
        if not isinstance(self.repair_member, GenerationMemberSettings):
            raise ValueError("candidate repair member is invalid")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateGeneratorSettings":
        if not isinstance(data, Mapping) or set(data) != {"members", "repair_member"}:
            raise ValueError("candidate generator settings fields are invalid")
        if not isinstance(data["members"], list):
            raise ValueError("candidate generator members must be a list")
        return cls(
            members=tuple(
                GenerationMemberSettings.from_dict(member) for member in data["members"]
            ),
            repair_member=GenerationMemberSettings.from_dict(data["repair_member"]),
        )


@dataclass(frozen=True)
class GeneratedCandidate:
    idea: IdeaRecord
    call: CodingAgentCallResult


class CandidateGenerator:
    """Generate independent candidates without mutating search or archive state."""

    def __init__(
        self,
        runner: CodingAgentCallRunner,
        settings: CandidateGeneratorSettings,
        prompt_template: str | None = None,
    ):
        self.runner = runner
        self.settings = settings
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else (
                Path(__file__).parent.parent / "prompts" / "ideation_v3_candidate.md"
            ).read_text(encoding="utf-8")
        )
        if self.prompt_template.count("{{MANDATORY_PACKET}}") != 1:
            raise ValueError("candidate prompt requires one mandatory packet marker")

    def generate(
        self,
        *,
        batch_id: str,
        problem_statement: str,
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
        archive_state: IdeaArchiveState,
        resolved_parents: Sequence[ResolvedParentSnapshot],
        workspaces: Sequence[str],
        prior_knowledge: PriorKnowledgeAccessMaterialization | None = None,
    ) -> Tuple[GeneratedCandidate, ...]:
        briefs = directive.operator_briefs
        parents = tuple(resolved_parents)
        member_count = directive.candidate_quota
        if member_count > len(self.settings.members):
            raise ValueError("candidate quota exceeds configured generator members")
        if len(briefs) != len(parents):
            raise ValueError("every operator brief requires one resolved parent")
        member_workspaces = tuple(workspaces)
        if len(briefs) != len(member_workspaces):
            raise ValueError("every operator brief requires one materialized workspace")
        tasks = tuple(
            (
                index,
                member,
                brief,
                parents[index],
                self._prompt(
                    batch_id=batch_id,
                    role=f"candidate_{index}",
                    problem_statement=problem_statement,
                    evidence_snapshot=evidence_snapshot,
                    directive=directive,
                    archive_state=archive_state,
                    operator_brief=brief,
                    resolved_parent=parents[index],
                    repair_request=None,
                    prior_knowledge=prior_knowledge,
                ),
                member_workspaces[index],
            )
            for index, (member, brief) in enumerate(
                zip(self.settings.members[:member_count], briefs)
            )
        )
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = tuple(
                executor.submit(
                    self._invoke,
                    batch_id,
                    f"candidate_{index}",
                    member,
                    brief,
                    parent,
                    prompt,
                    workspace,
                    prior_knowledge,
                )
                for index, member, brief, parent, prompt, workspace in tasks
            )
            return tuple(future.result() for future in futures)

    def generate_repair(
        self,
        *,
        batch_id: str,
        problem_statement: str,
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
        archive_state: IdeaArchiveState,
        operator_brief: OperatorBrief,
        resolved_parent: ResolvedParentSnapshot,
        repair_request: Mapping[str, Any],
        workspace: str,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None = None,
    ) -> GeneratedCandidate:
        if directive.repair_quota != 1:
            raise ValueError("search directive does not authorize a repair call")
        prompt = self._prompt(
            batch_id=batch_id,
            role="diversity_repair",
            problem_statement=problem_statement,
            evidence_snapshot=evidence_snapshot,
            directive=directive,
            archive_state=archive_state,
            operator_brief=operator_brief,
            resolved_parent=resolved_parent,
            repair_request=repair_request,
            prior_knowledge=prior_knowledge,
        )
        return self._invoke(
            batch_id,
            "diversity_repair",
            self.settings.repair_member,
            operator_brief,
            resolved_parent,
            prompt,
            workspace,
            prior_knowledge,
        )

    def _invoke(
        self,
        batch_id: str,
        role: str,
        settings: GenerationMemberSettings,
        brief: OperatorBrief,
        resolved_parent: ResolvedParentSnapshot,
        prompt: str,
        workspace: str,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None,
    ) -> GeneratedCandidate:
        call = self.runner.run(
            CodingAgentCallRequest(
                operation_id=content_identifier(
                    "agent_call",
                    hashlib.sha256(f"{batch_id}:{role}".encode("utf-8")).hexdigest(),
                ),
                role=role,
                cli=settings.cli,
                model=settings.model,
                prompt=prompt,
                workspace=workspace,
                timeout_seconds=settings.timeout_seconds,
                effort=settings.effort,
                allowed_tools=settings.allowed_tools,
                prior_knowledge=prior_knowledge,
            ),
            CANDIDATE_RESPONSE_SCHEMA,
        )
        parsed = json.loads(call.output)
        if not isinstance(parsed, dict) or set(parsed) != _CANDIDATE_FIELDS:
            raise ValueError("coding-agent candidate fields are invalid")
        descriptor = IdeaDescriptor.from_dict(parsed["descriptor"])
        prior_refs = tuple(parsed["prior_knowledge_refs"])
        allowed_prior_refs = set()
        if prior_knowledge is not None:
            allowed_prior_refs.update(
                record["record_id"]
                for record in PriorKnowledgeAccess(
                    prior_knowledge
                ).list_citable_records()
            )
        if not set(prior_refs).issubset(allowed_prior_refs):
            raise ValueError("candidate references prior knowledge outside its packet")
        source_nodes = list(brief.parent_plan.source_experiment_node_ids)
        if brief.parent_plan.experiment_node_id is not None:
            source_nodes.append(brief.parent_plan.experiment_node_id)
        resolved_nodes = (
            [] if resolved_parent.node_id is None else [resolved_parent.node_id]
        )
        parent_nodes = tuple(dict.fromkeys(resolved_nodes + source_nodes))
        target_gaps = () if brief.target_gap_id is None else (brief.target_gap_id,)
        idea = IdeaRecord(
            idea_id=new_identifier("idea"),
            origin_batch_id=batch_id,
            proposal=parsed["proposal"],
            operator=brief.operator,
            descriptor=descriptor,
            parent_plan=brief.parent_plan,
            resolved_parent=resolved_parent,
            assumptions=tuple(parsed["assumptions"]),
            evidence_refs=tuple(parsed["evidence_refs"]),
            directive_rationale=parsed["directive_rationale"],
            evaluation_method=parsed["evaluation_method"],
            resource_request=parsed["resource_request"],
            created_at=utc_now(),
            prior_knowledge_refs=prior_refs,
            prior_adaptation_rationale=parsed["prior_adaptation_rationale"],
            parent_idea_ids=brief.parent_plan.source_idea_ids,
            parent_experiment_node_ids=parent_nodes,
            target_gap_ids=target_gaps,
            claim_ids=tuple(parsed["claim_ids"]),
            resolves_claim_ids=tuple(parsed["resolves_claim_ids"]),
            expected_observations=tuple(parsed["expected_observations"]),
            predicted_gain=parsed["predicted_gain"],
            predicted_cost=parsed["predicted_cost"],
            confidence=parsed["confidence"],
            claimed_nearest_idea_id=parsed["claimed_nearest_idea_id"],
            claimed_nearest_experiment_node_id=parsed[
                "claimed_nearest_experiment_node_id"
            ],
            generation_artifacts=call.artifacts,
        )
        return GeneratedCandidate(idea=idea, call=call)

    def _prompt(
        self,
        *,
        batch_id: str,
        role: str,
        problem_statement: str,
        evidence_snapshot: CampaignEvidenceSnapshot,
        directive: SearchDirective,
        archive_state: IdeaArchiveState,
        operator_brief: OperatorBrief,
        resolved_parent: ResolvedParentSnapshot,
        repair_request: Mapping[str, Any] | None,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None,
    ) -> str:
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ValueError("candidate problem statement must be non-empty")
        local_current_run = {
            "evidence_snapshot": evidence_snapshot.to_dict(),
            "allowed_evidence_refs": sorted(evidence_reference_ids(evidence_snapshot)),
            "current_run_ideas": [idea.to_dict() for idea in archive_state.ideas],
            "archive_claims": [claim.to_dict() for claim in archive_state.claims],
            "archive_gaps": [gap.to_dict() for gap in archive_state.gaps],
        }
        foreign_prior_knowledge = {
            "authority": (
                "Advisory foreign knowledge only: never use these records as local "
                "parents, evidence, claims, incumbents, or gap resolutions. Cite "
                "them only through prior_knowledge_refs; exact reuse requires an "
                "adaptation or deliberate local-replication rationale."
            ),
            "materialization": (
                None if prior_knowledge is None else prior_knowledge.to_dict()
            ),
            "allowed_prior_knowledge_refs": (
                []
                if prior_knowledge is None
                else [
                    record["record_id"]
                    for record in PriorKnowledgeAccess(
                        prior_knowledge
                    ).list_citable_records()
                ]
            ),
        }
        packet = json.dumps(
            {
                "batch_id": batch_id,
                "role": role,
                "problem_statement": problem_statement,
                "search_directive": directive.to_dict(),
                "operator_brief": operator_brief.to_dict(),
                "resolved_parent": resolved_parent.to_dict(),
                "local_current_run": local_current_run,
                "foreign_prior_knowledge": foreign_prior_knowledge,
                "repair_request": repair_request,
            },
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        return self.prompt_template.replace("{{MANDATORY_PACKET}}", packet)
