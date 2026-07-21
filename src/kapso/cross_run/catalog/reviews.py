"""Independent structured coding-agent reviews for catalog subjects."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.agent_operations import (
    CatalogAgentOperationRecord,
    build_catalog_agent_operation_receipt,
    catalog_agent_operation_id,
    validate_catalog_agent_workspace,
)
from kapso.cross_run.catalog.admission import ClaimEvidenceClosure
from kapso.cross_run.contracts import (
    CodingAgentOperationReceipt,
    ExpertScopeContract,
    KnowledgeClaim,
    PriorIdea,
    ReviewAssertion,
    StrictContract,
    TransferEpisode,
)
from kapso.cross_run.settings import CatalogReviewerSettings, CatalogSettings
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    CodingAgentCallRunner,
)

_PROMPT_TEMPLATE_PATH = Path(__file__).parents[1] / "prompts" / "catalog_reviewer.md"
_PROMPT_PACKET_MARKER = "REVIEW_PACKET_JSON"
_SUBJECT_TYPES = {
    "knowledge_claim": (KnowledgeClaim, "revision_id"),
    "prior_idea": (PriorIdea, "prior_idea_id"),
    "transfer_episode": (TransferEpisode, "episode_id"),
}


class CatalogReviewError(ValueError):
    """A review packet, reviewer result, or reviewer assignment is invalid."""


def _require_record_envelope(
    value: Any,
    name: str,
) -> tuple[str, str, StrictContract]:
    if not isinstance(value, MappingABC) or set(value) != {
        "payload",
        "record_id",
        "record_kind",
    }:
        raise CatalogReviewError(f"{name} must be a complete record envelope")
    record_id = value["record_id"]
    record_kind = value["record_kind"]
    payload = value["payload"]
    require_content_id(record_id, f"{name}.record_id")
    require_identifier(record_kind, f"{name}.record_kind")
    if not isinstance(payload, MappingABC) or not payload:
        raise CatalogReviewError(f"{name}.payload must be a complete object")
    if record_kind not in _SUBJECT_TYPES:
        raise CatalogReviewError(f"{name}.record_kind is unsupported")
    contract_type, identity_field = _SUBJECT_TYPES[record_kind]
    record = contract_type.from_dict(payload)
    if getattr(record, identity_field) != record_id:
        raise CatalogReviewError(f"{name} envelope identity mismatch")
    return record_id, record_kind, record


@dataclass(frozen=True)
class CatalogReviewPacket(StrictContract):
    """One subject plus every exact record needed for independent review."""

    catalog_generation_id: str
    catalog_generation: int
    scope_contract: ExpertScopeContract
    subject: Mapping[str, Any]
    evidence_records: tuple[Mapping[str, Any], ...]
    proposer_operation_receipt: CodingAgentOperationReceipt | None
    claim_evidence_closure: ClaimEvidenceClosure | None
    previous_assertions: tuple[ReviewAssertion, ...]
    previous_operation_receipts: tuple[CodingAgentOperationReceipt, ...]

    def _validate(self) -> None:
        require_content_id(self.catalog_generation_id, "catalog_generation_id")
        if self.catalog_generation < 0:
            raise CatalogReviewError("catalog_generation must be non-negative")
        subject_id, subject_kind, subject = _require_record_envelope(
            self.subject,
            "review subject",
        )
        evidence_ids: list[str] = []
        for position, record in enumerate(self.evidence_records):
            record_id, _, _ = _require_record_envelope(
                record,
                f"evidence_records[{position}]",
            )
            evidence_ids.append(record_id)
        if tuple(evidence_ids) != tuple(sorted(set(evidence_ids))):
            raise CatalogReviewError(
                "review evidence records must be sorted and unique"
            )
        assertion_ids = tuple(
            assertion.assertion_id for assertion in self.previous_assertions
        )
        if assertion_ids != tuple(sorted(set(assertion_ids))):
            raise CatalogReviewError(
                "previous assertions must be sorted and uniquely identified"
            )
        receipt_ids = tuple(
            receipt.operation_receipt_id for receipt in self.previous_operation_receipts
        )
        if receipt_ids != tuple(sorted(set(receipt_ids))):
            raise CatalogReviewError(
                "previous operation receipts must be sorted and unique"
            )
        known_receipts = set(receipt_ids)
        for assertion in self.previous_assertions:
            if assertion.subject_id != subject_id:
                raise CatalogReviewError("previous assertion names another subject")
            if assertion.review_operation_ref not in known_receipts:
                raise CatalogReviewError(
                    "previous assertion operation receipt is unresolved"
                )
        if isinstance(subject, KnowledgeClaim):
            if self.proposer_operation_receipt is None:
                raise CatalogReviewError("claim review requires proposer provenance")
            proposal_operation_id = subject.proposal_provenance.get(
                "operation_receipt_id"
            )
            if (
                proposal_operation_id
                != self.proposer_operation_receipt.operation_receipt_id
            ):
                raise CatalogReviewError("claim proposer receipt does not match")
            if self.claim_evidence_closure is None:
                raise CatalogReviewError("claim review requires evidence assessments")
            if (
                self.claim_evidence_closure.claim_revision_id != subject.revision_id
                or self.claim_evidence_closure.supporting_episode_ids
                != subject.supporting_episode_ids
                or self.claim_evidence_closure.contradicting_episode_ids
                != subject.contradicting_episode_ids
                or set(evidence_ids)
                != set(self.claim_evidence_closure.evaluated_episode_ids)
            ):
                raise CatalogReviewError("claim evidence assessment closure differs")
            if subject.scope_contract_id != self.scope_contract.scope_contract_id:
                raise CatalogReviewError("claim review uses another scope revision")
        elif (
            self.proposer_operation_receipt is not None
            or self.claim_evidence_closure is not None
        ):
            raise CatalogReviewError(
                "non-claim review cannot carry claim proposer provenance"
            )

    @property
    def subject_id(self) -> str:
        return self.subject["record_id"]

    @property
    def evidence_record_ids(self) -> tuple[str, ...]:
        return tuple(record["record_id"] for record in self.evidence_records)

    @property
    def packet_digest(self) -> str:
        return tree_or_blob_digest(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class CatalogReviewResult:
    assertion: ReviewAssertion
    operation_receipt: CodingAgentOperationReceipt
    operation_record: CatalogAgentOperationRecord
    call_result: CodingAgentCallResult


class CatalogReviewer:
    """Run one configured reviewer slot without granting admission authority."""

    def __init__(
        self,
        settings: CatalogSettings,
        runner: CodingAgentCallRunner,
    ):
        self.settings = settings
        self.runner = runner

    def review(
        self,
        packet: CatalogReviewPacket,
        reviewer: CatalogReviewerSettings,
        workspace: Path,
    ) -> CatalogReviewResult:
        validate_catalog_agent_workspace(workspace)
        configured = {
            candidate.reviewer_id: candidate for candidate in self.settings.reviewers
        }
        if configured.get(reviewer.reviewer_id) != reviewer:
            raise CatalogReviewError("reviewer slot is not configured")
        if reviewer.reviewer_id == self.settings.claim_proposer_id:
            raise CatalogReviewError("claim proposer cannot review its own output")
        if reviewer.agent.allowed_tools:
            raise CatalogReviewError("catalog reviewer must not receive tools")
        if len(packet.evidence_records) > self.settings.review_packet_record_limit:
            raise CatalogReviewError(
                "review packet exceeds the complete-record selection limit"
            )
        template = self.operation_template()
        schema = self.response_schema()
        prompt = template.replace(
            _PROMPT_PACKET_MARKER,
            canonical_json_bytes(packet.to_dict()).decode("utf-8"),
        )
        operation_preimage = {
            "packet": packet.to_dict(),
            "template": template,
            "schema": schema,
            "reviewer": reviewer.to_dict(),
            "catalog_configuration": self.settings.to_dict(),
        }
        operation_id = catalog_agent_operation_id(operation_preimage)
        result = self.runner.run(
            CodingAgentCallRequest(
                operation_id=operation_id,
                role=reviewer.reviewer_role,
                cli=reviewer.agent.cli,
                model=reviewer.agent.model,
                prompt=prompt,
                workspace=str(workspace),
                timeout_seconds=reviewer.agent.timeout_seconds,
                effort=reviewer.agent.effort,
                allowed_tools=reviewer.agent.allowed_tools,
            ),
            schema,
        )
        receipt, final_output = build_catalog_agent_operation_receipt(
            operation_id=operation_id,
            principal_id=reviewer.reviewer_id,
            role=reviewer.reviewer_role,
            agent=reviewer.agent,
            result=result,
        )
        assertion = self._parse_assertion(packet, reviewer, receipt, final_output)
        operation_record = CatalogAgentOperationRecord.mint(
            operation_kind="catalog_review",
            operation_receipt_id=receipt.operation_receipt_id,
            operation_preimage=operation_preimage,
            final_output=final_output,
            produced_object_ids=(assertion.assertion_id,),
        )
        return CatalogReviewResult(
            assertion=assertion,
            operation_receipt=receipt,
            operation_record=operation_record,
            call_result=result,
        )

    @staticmethod
    def operation_template() -> str:
        template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        if template.count(_PROMPT_PACKET_MARKER) != 1:
            raise CatalogReviewError("catalog reviewer template marker is invalid")
        return template

    def _parse_assertion(
        self,
        packet: CatalogReviewPacket,
        reviewer: CatalogReviewerSettings,
        receipt: CodingAgentOperationReceipt,
        output: str,
    ) -> ReviewAssertion:
        parsed = parse_json_bytes(output)
        fields = {
            "exact_evidence_refs",
            "judgment",
            "rationale",
            "supersedes_assertion_id",
        }
        if not isinstance(parsed, MappingABC) or set(parsed) != fields:
            raise CatalogReviewError("catalog reviewer output fields are invalid")
        judgment = parsed["judgment"]
        allowed_judgments = {
            self.settings.admission.approval_judgment,
            self.settings.admission.rejection_judgment,
        }
        if judgment not in allowed_judgments:
            raise CatalogReviewError("catalog reviewer judgment is invalid")
        rationale = parsed["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise CatalogReviewError("catalog reviewer rationale is required")
        evidence_refs = parsed["exact_evidence_refs"]
        if not isinstance(evidence_refs, list):
            raise CatalogReviewError("review evidence refs must be an array")
        evidence_refs = tuple(evidence_refs)
        if evidence_refs != packet.evidence_record_ids:
            raise CatalogReviewError("review must name every exact evidence record")
        expected_supersession = self._expected_supersession(packet, reviewer)
        if parsed["supersedes_assertion_id"] != expected_supersession:
            raise CatalogReviewError(
                "review assertion supersession does not match the active head"
            )
        return ReviewAssertion.mint(
            subject_id=packet.subject_id,
            reviewer_id=reviewer.reviewer_id,
            reviewer_role=reviewer.reviewer_role,
            rubric_version=reviewer.rubric_version,
            judgment=judgment,
            rationale=rationale,
            exact_evidence_refs=evidence_refs,
            supersedes_assertion_id=expected_supersession,
            review_operation_ref=receipt.operation_receipt_id,
        )

    @staticmethod
    def _expected_supersession(
        packet: CatalogReviewPacket,
        reviewer: CatalogReviewerSettings,
    ) -> str | None:
        assertions = tuple(
            assertion
            for assertion in packet.previous_assertions
            if assertion.reviewer_id == reviewer.reviewer_id
        )
        superseded = {
            assertion.supersedes_assertion_id
            for assertion in assertions
            if assertion.supersedes_assertion_id is not None
        }
        heads = tuple(
            assertion.assertion_id
            for assertion in assertions
            if assertion.assertion_id not in superseded
        )
        if len(heads) > 1:
            raise CatalogReviewError("reviewer assertion history has multiple heads")
        return heads[0] if heads else None

    def response_schema(self) -> Mapping[str, Any]:
        return self.response_schema_for(self.settings)

    @staticmethod
    def response_schema_for(settings: CatalogSettings) -> Mapping[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "exact_evidence_refs",
                "judgment",
                "rationale",
                "supersedes_assertion_id",
            ],
            "properties": {
                "exact_evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "judgment": {
                    "type": "string",
                    "enum": sorted(
                        {
                            settings.admission.approval_judgment,
                            settings.admission.rejection_judgment,
                        }
                    ),
                },
                "rationale": {"type": "string", "minLength": 1},
                "supersedes_assertion_id": {
                    "type": ["string", "null"],
                },
            },
        }
