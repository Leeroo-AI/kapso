"""Structured claim proposals over one immutable catalog packet."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kapso.cross_run.catalog.agent_operations import (
    build_catalog_agent_operation_receipt,
    validate_catalog_agent_workspace,
)
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CatalogEntryState,
    CodingAgentOperationReceipt,
    ExpertScopeContract,
    KnowledgeClaim,
    PriorIdea,
    ReviewAssertion,
    StrictContract,
    TransferEpisode,
)
from kapso.cross_run.record_contracts import (
    CatalogAgentOperationRecord,
    ClaimEvidenceClosure,
    catalog_agent_operation_id,
)
from kapso.cross_run.settings import CatalogSettings
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    CodingAgentCallRunner,
)

_PROMPT_TEMPLATE_PATH = Path(__file__).parents[1] / "prompts" / "claim_proposer.md"
_PROMPT_PACKET_MARKER = "CATALOG_PACKET_JSON"
_RELATIONSHIPS = {"contradiction", "not_applicable", "support"}


class ClaimProposalError(ValueError):
    """A proposal packet, coding-agent result, or claim is invalid."""


def _require_sorted_records(
    records: tuple[StrictContract, ...],
    identity_field: str,
    name: str,
) -> tuple[str, ...]:
    identities = tuple(getattr(record, identity_field) for record in records)
    if identities != tuple(sorted(set(identities))):
        raise ClaimProposalError(f"{name} must be sorted and uniquely identified")
    return identities


def _require_exact_object(
    value: Any,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise ClaimProposalError(f"{name} must be an object")
    missing = tuple(sorted(fields - set(value)))
    unknown = tuple(sorted(set(value) - fields))
    if missing or unknown:
        raise ClaimProposalError(
            f"{name} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimProposalError(f"{name} must be non-empty text")
    return value


def _require_text_array(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ClaimProposalError(f"{name} must be an array")
    values = tuple(_require_text(item, name) for item in value)
    if not values or len(values) != len(set(values)):
        raise ClaimProposalError(f"{name} must be non-empty and unique")
    return tuple(sorted(values))


@dataclass(frozen=True)
class ClaimProposalPacket(StrictContract):
    """Complete selected records from one exact catalog generation."""

    catalog_generation_id: str
    catalog_generation: int
    scope_contract: ExpertScopeContract
    episodes: tuple[TransferEpisode, ...]
    prior_ideas: tuple[PriorIdea, ...]
    existing_claims: tuple[KnowledgeClaim, ...]
    entry_states: tuple[CatalogEntryState, ...]
    review_assertions: tuple[ReviewAssertion, ...]
    operation_receipts: tuple[CodingAgentOperationReceipt, ...]
    proof_reference_ids: tuple[str, ...]

    def _validate(self) -> None:
        require_content_id(self.catalog_generation_id, "catalog_generation_id")
        if self.catalog_generation < 0:
            raise ClaimProposalError("catalog_generation must be non-negative")
        episode_ids = _require_sorted_records(
            self.episodes,
            "episode_id",
            "packet episodes",
        )
        prior_idea_ids = _require_sorted_records(
            self.prior_ideas,
            "prior_idea_id",
            "packet prior ideas",
        )
        claim_ids = _require_sorted_records(
            self.existing_claims,
            "revision_id",
            "packet claims",
        )
        state_ids = _require_sorted_records(
            self.entry_states,
            "catalog_entry_state_id",
            "packet entry states",
        )
        assertion_ids = _require_sorted_records(
            self.review_assertions,
            "assertion_id",
            "packet assertions",
        )
        receipt_ids = _require_sorted_records(
            self.operation_receipts,
            "operation_receipt_id",
            "packet operation receipts",
        )
        del state_ids, assertion_ids
        if self.proof_reference_ids != tuple(sorted(set(self.proof_reference_ids))):
            raise ClaimProposalError(
                "packet proof reference IDs must be sorted and unique"
            )
        for reference_id in self.proof_reference_ids:
            require_content_id(reference_id, "proof_reference_ids")
        scope_contract_id = self.scope_contract.scope_contract_id
        scope_id = self.scope_contract.scope_id
        for episode in self.episodes:
            if episode.task_context_binding.scope_contract_id != scope_contract_id:
                raise ClaimProposalError("packet episode uses another scope revision")
            episode.task_context_binding.validate_against(self.scope_contract)
            required = {
                episode.source_bundle_id,
                episode.sanitation_report_id,
                *episode.derivation_refs,
            }
            if not required.issubset(self.proof_reference_ids):
                raise ClaimProposalError("packet omits episode proof references")
        for prior_idea in self.prior_ideas:
            if (
                prior_idea.task_context_binding.scope_contract_id != scope_contract_id
                or prior_idea.task_context_binding.scope_id != scope_id
            ):
                raise ClaimProposalError("packet prior idea uses another scope")
            prior_idea.task_context_binding.validate_against(self.scope_contract)
            required = {
                prior_idea.source_bundle_id,
                prior_idea.sanitation_report_id,
            }
            if not required.issubset(self.proof_reference_ids):
                raise ClaimProposalError("packet omits prior-idea proof references")
        known_episode_ids = set(episode_ids)
        for claim in self.existing_claims:
            if claim.scope_contract_id != scope_contract_id:
                raise ClaimProposalError("packet claim uses another scope revision")
            claim_evidence = set(claim.supporting_episode_ids)
            claim_evidence.update(claim.contradicting_episode_ids)
            if not claim_evidence.issubset(known_episode_ids):
                raise ClaimProposalError("packet omits existing claim evidence")
        known_subject_ids = set(episode_ids) | set(prior_idea_ids) | set(claim_ids)
        if any(
            state.subject_payload_id not in known_subject_ids
            for state in self.entry_states
        ):
            raise ClaimProposalError("packet entry state has an unknown subject")
        known_receipt_ids = set(receipt_ids)
        known_assertion_evidence = set(self.proof_reference_ids) | known_subject_ids
        for assertion in self.review_assertions:
            if assertion.subject_id not in known_subject_ids:
                raise ClaimProposalError("packet assertion has an unknown subject")
            if assertion.review_operation_ref not in known_receipt_ids:
                raise ClaimProposalError("packet assertion operation is unresolved")
            if not set(assertion.exact_evidence_refs).issubset(
                known_assertion_evidence
            ):
                raise ClaimProposalError("packet assertion evidence is unresolved")

    @property
    def scientific_record_count(self) -> int:
        return len(self.episodes) + len(self.prior_ideas) + len(self.existing_claims)

    @property
    def packet_digest(self) -> str:
        return tree_or_blob_digest(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class ClaimProposalResult:
    claims: tuple[KnowledgeClaim, ...]
    claim_evidence_closures: tuple[ClaimEvidenceClosure, ...]
    operation_receipt: CodingAgentOperationReceipt
    operation_record: CatalogAgentOperationRecord
    call_result: CodingAgentCallResult
    packet_digest: str


@dataclass(frozen=True)
class _ParsedClaimProposal:
    statement: str
    mechanism: str
    applicability_predicates: Mapping[str, Any]
    explicit_exclusions: tuple[str, ...]
    supporting_episode_ids: tuple[str, ...]
    contradicting_episode_ids: tuple[str, ...]
    supersedes_revision_ids: tuple[str, ...]
    evidence_assessments: tuple[Mapping[str, str], ...]

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "statement": self.statement,
            "mechanism": self.mechanism,
            "applicability_predicates": self.applicability_predicates,
            "explicit_exclusions": self.explicit_exclusions,
            "supporting_episode_ids": self.supporting_episode_ids,
            "contradicting_episode_ids": self.contradicting_episode_ids,
            "supersedes_revision_ids": self.supersedes_revision_ids,
        }


class ClaimProposer:
    """Ask one configured coding agent to propose, but never admit, claims."""

    def __init__(
        self,
        settings: CatalogSettings,
        runner: CodingAgentCallRunner,
    ):
        self.settings = settings
        self.runner = runner

    def propose(
        self,
        packet: ClaimProposalPacket,
        workspace: Path,
    ) -> ClaimProposalResult:
        validate_catalog_agent_workspace(workspace)
        if self.settings.claim_proposer.allowed_tools:
            raise ClaimProposalError("claim proposer must not receive tools")
        if packet.scientific_record_count > self.settings.claim_packet_record_limit:
            raise ClaimProposalError(
                "claim packet exceeds the complete-record selection limit"
            )
        template = self.operation_template()
        schema = self.response_schema()
        prompt = template.replace(
            _PROMPT_PACKET_MARKER,
            canonical_json_bytes(packet.to_dict()).decode("utf-8"),
        )
        operation_preimage = self._operation_preimage(packet, template, schema)
        operation_id = catalog_agent_operation_id(operation_preimage)
        agent = self.settings.claim_proposer
        result = self.runner.run(
            CodingAgentCallRequest(
                operation_id=operation_id,
                role=self.settings.claim_proposer_role,
                cli=agent.cli,
                model=agent.model,
                prompt=prompt,
                workspace=str(workspace),
                timeout_seconds=agent.timeout_seconds,
                effort=agent.effort,
                allowed_tools=agent.allowed_tools,
            ),
            schema,
        )
        receipt, final_output = build_catalog_agent_operation_receipt(
            operation_id=operation_id,
            principal_id=self.settings.claim_proposer_id,
            role=self.settings.claim_proposer_role,
            agent=self.settings.claim_proposer,
            result=result,
        )
        claims, parsed_proposals = self._parse_claims(packet, receipt, final_output)
        evaluated_episode_ids = tuple(episode.episode_id for episode in packet.episodes)
        claim_evidence_closures = tuple(
            ClaimEvidenceClosure.mint(
                claim_revision_id=claim.revision_id,
                evaluated_episode_ids=evaluated_episode_ids,
                supporting_episode_ids=claim.supporting_episode_ids,
                contradicting_episode_ids=claim.contradicting_episode_ids,
                evidence_assessments=parsed_proposals[position].evidence_assessments,
                proposer_operation_receipt_id=receipt.operation_receipt_id,
                packet_digest=packet.packet_digest,
            )
            for position, claim in enumerate(claims)
        )
        produced_object_ids = tuple(
            sorted(
                {
                    *(claim.revision_id for claim in claims),
                    *(
                        closure.claim_evidence_closure_id
                        for closure in claim_evidence_closures
                    ),
                }
            )
        )
        operation_record = CatalogAgentOperationRecord.mint(
            operation_kind="claim_proposal",
            operation_receipt_id=receipt.operation_receipt_id,
            operation_preimage=operation_preimage,
            final_output=final_output,
            produced_object_ids=produced_object_ids,
        )
        return ClaimProposalResult(
            claims=claims,
            claim_evidence_closures=claim_evidence_closures,
            operation_receipt=receipt,
            operation_record=operation_record,
            call_result=result,
            packet_digest=packet.packet_digest,
        )

    def _operation_preimage(
        self,
        packet: ClaimProposalPacket,
        template: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "packet": packet.to_dict(),
            "template": template,
            "schema": schema,
            "catalog_configuration": self.settings.to_dict(),
        }

    @staticmethod
    def operation_template() -> str:
        template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        if template.count(_PROMPT_PACKET_MARKER) != 1:
            raise ClaimProposalError("claim proposer template marker is invalid")
        return template

    def _parse_claims(
        self,
        packet: ClaimProposalPacket,
        receipt: CodingAgentOperationReceipt,
        output: str,
    ) -> tuple[tuple[KnowledgeClaim, ...], tuple[_ParsedClaimProposal, ...]]:
        envelope = _require_exact_object(
            parse_json_bytes(output),
            {"claims"},
            "claim proposer output",
        )
        raw_claims = envelope["claims"]
        if not isinstance(raw_claims, list):
            raise ClaimProposalError("claim proposer claims must be an array")
        parsed = tuple(
            self._parse_claim(packet, raw, position)
            for position, raw in enumerate(raw_claims)
        )
        claim_keys = tuple(
            canonical_json_bytes(proposal.canonical_payload()) for proposal in parsed
        )
        if len(claim_keys) != len(set(claim_keys)):
            raise ClaimProposalError("claim proposer returned duplicate claims")
        ordered = tuple(
            proposal
            for _, proposal in sorted(zip(claim_keys, parsed), key=lambda item: item[0])
        )
        claims = tuple(
            self._mint_claim(packet, receipt, proposal, ordinal)
            for ordinal, proposal in enumerate(ordered)
        )
        return claims, ordered

    def _parse_claim(
        self,
        packet: ClaimProposalPacket,
        raw: Any,
        position: int,
    ) -> _ParsedClaimProposal:
        values = _require_exact_object(
            raw,
            {
                "applicability_predicates",
                "evidence_assessments",
                "explicit_exclusions",
                "mechanism",
                "statement",
                "supersedes_revision_ids",
            },
            f"claims[{position}]",
        )
        predicates = self._validate_predicates(
            packet.scope_contract,
            values["applicability_predicates"],
        )
        exclusions = _require_text_array(
            values["explicit_exclusions"],
            f"claims[{position}].explicit_exclusions",
        )
        supersedes = self._superseded_revisions(
            packet,
            values["supersedes_revision_ids"],
            position,
        )
        supporting, contradicting, assessments = self._classify_evidence(
            packet,
            values["evidence_assessments"],
            position,
        )
        return _ParsedClaimProposal(
            statement=_require_text(values["statement"], "claim statement"),
            mechanism=_require_text(values["mechanism"], "claim mechanism"),
            applicability_predicates=predicates,
            explicit_exclusions=exclusions,
            supporting_episode_ids=supporting,
            contradicting_episode_ids=contradicting,
            supersedes_revision_ids=supersedes,
            evidence_assessments=assessments,
        )

    @staticmethod
    def _mint_claim(
        packet: ClaimProposalPacket,
        receipt: CodingAgentOperationReceipt,
        proposal: _ParsedClaimProposal,
        ordinal: int,
    ) -> KnowledgeClaim:
        if proposal.supersedes_revision_ids:
            existing = {claim.revision_id: claim for claim in packet.existing_claims}
            lineage_ids = {
                existing[revision_id].claim_id
                for revision_id in proposal.supersedes_revision_ids
            }
            if len(lineage_ids) != 1:
                raise ClaimProposalError(
                    "one claim revision cannot merge unrelated claim lineages"
                )
            claim_id = next(iter(lineage_ids))
        else:
            claim_id = (
                "claim_"
                + content_id(
                    "claim-lineage",
                    {
                        "operation_receipt_id": receipt.operation_receipt_id,
                        "proposal_ordinal": ordinal,
                    },
                ).rsplit(":", 1)[1][:32]
            )
        return KnowledgeClaim.mint(
            claim_id=claim_id,
            scope_contract_id=packet.scope_contract.scope_contract_id,
            statement=proposal.statement,
            mechanism=proposal.mechanism,
            applicability_predicates=proposal.applicability_predicates,
            explicit_exclusions=proposal.explicit_exclusions,
            supporting_episode_ids=proposal.supporting_episode_ids,
            contradicting_episode_ids=proposal.contradicting_episode_ids,
            proposal_provenance={
                "operation_receipt_id": receipt.operation_receipt_id,
                "packet_digest": packet.packet_digest,
                "proposal_ordinal": ordinal,
            },
            supersedes_revision_ids=proposal.supersedes_revision_ids,
        )

    @staticmethod
    def _validate_predicates(
        scope_contract: ExpertScopeContract,
        value: Any,
    ) -> Mapping[str, Any]:
        if not isinstance(value, MappingABC) or not value:
            raise ClaimProposalError("claim applicability predicates must be non-empty")
        schemas = {
            schema.dimension_id: schema
            for schema in scope_contract.context_dimension_schemas
        }
        if not set(value).issubset(schemas):
            raise ClaimProposalError(
                "claim applicability uses an unregistered context dimension"
            )
        validated: dict[str, Any] = {}
        for dimension_id in sorted(value):
            predicate_value = value[dimension_id]
            if isinstance(predicate_value, list):
                predicate_value = tuple(predicate_value)
            schemas[dimension_id].validate_value(predicate_value)
            validated[dimension_id] = predicate_value
        return validated

    @staticmethod
    def _superseded_revisions(
        packet: ClaimProposalPacket,
        value: Any,
        position: int,
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ClaimProposalError(
                f"claims[{position}].supersedes_revision_ids must be an array"
            )
        revisions = tuple(value)
        if any(not isinstance(revision_id, str) for revision_id in revisions):
            raise ClaimProposalError("superseded revision ID must be text")
        if revisions != tuple(sorted(set(revisions))):
            raise ClaimProposalError(
                "superseded revision IDs must be sorted and unique"
            )
        known = {claim.revision_id for claim in packet.existing_claims}
        if not set(revisions).issubset(known):
            raise ClaimProposalError("claim supersedes an unknown revision")
        return revisions

    @staticmethod
    def _classify_evidence(
        packet: ClaimProposalPacket,
        value: Any,
        position: int,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[Mapping[str, str], ...],
    ]:
        if not isinstance(value, list):
            raise ClaimProposalError(
                f"claims[{position}].evidence_assessments must be an array"
            )
        relationships: dict[str, str] = {}
        rationales: dict[str, str] = {}
        for assessment_position, raw in enumerate(value):
            assessment = _require_exact_object(
                raw,
                {"episode_id", "rationale", "relationship"},
                f"claims[{position}].evidence_assessments[{assessment_position}]",
            )
            episode_id = _require_text(assessment["episode_id"], "episode_id")
            relationship = assessment["relationship"]
            if relationship not in _RELATIONSHIPS:
                raise ClaimProposalError("invalid evidence relationship")
            rationale = _require_text(assessment["rationale"], "evidence rationale")
            if episode_id in relationships:
                raise ClaimProposalError("episode was assessed more than once")
            relationships[episode_id] = relationship
            rationales[episode_id] = rationale
        expected = {episode.episode_id for episode in packet.episodes}
        if set(relationships) != expected:
            raise ClaimProposalError(
                "every packet episode must be assessed exactly once"
            )
        supporting = tuple(
            sorted(
                episode_id
                for episode_id, relationship in relationships.items()
                if relationship == "support"
            )
        )
        contradicting = tuple(
            sorted(
                episode_id
                for episode_id, relationship in relationships.items()
                if relationship == "contradiction"
            )
        )
        if not supporting and not contradicting:
            raise ClaimProposalError("claim has no supporting or contradicting episode")
        assessments = tuple(
            {
                "episode_id": episode_id,
                "relationship": relationships[episode_id],
                "rationale": rationales[episode_id],
            }
            for episode_id in sorted(relationships)
        )
        return supporting, contradicting, assessments

    @staticmethod
    def response_schema() -> Mapping[str, Any]:
        evidence_assessment = {
            "type": "object",
            "additionalProperties": False,
            "required": ["episode_id", "rationale", "relationship"],
            "properties": {
                "episode_id": {"type": "string", "minLength": 1},
                "rationale": {"type": "string", "minLength": 1},
                "relationship": {
                    "type": "string",
                    "enum": sorted(_RELATIONSHIPS),
                },
            },
        }
        claim = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "applicability_predicates",
                "evidence_assessments",
                "explicit_exclusions",
                "mechanism",
                "statement",
                "supersedes_revision_ids",
            ],
            "properties": {
                "applicability_predicates": {
                    "type": "object",
                    "minProperties": 1,
                },
                "evidence_assessments": {
                    "type": "array",
                    "items": evidence_assessment,
                },
                "explicit_exclusions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "mechanism": {"type": "string", "minLength": 1},
                "statement": {"type": "string", "minLength": 1},
                "supersedes_revision_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["claims"],
            "properties": {
                "claims": {
                    "type": "array",
                    "items": claim,
                }
            },
        }
