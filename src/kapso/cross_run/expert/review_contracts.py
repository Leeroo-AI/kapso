"""Authenticated contracts for the automated expert-review stage."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping

from kapso.cross_run.agent_artifacts import CodingAgentWorkspaceAccess
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CodingAgentOperationReceipt,
    ContractValidationError,
    ExpertAcceptedStageResultRef,
    ExpertReviewDisposition,
    StrictContract,
)
from kapso.cross_run.settings import ExpertReviewerSettings
from kapso.execution.coding_agents.operation_receipt import (
    verify_coding_agent_operation_artifacts,
)

EXPERT_AUTOMATED_REVIEW_CONTRACT_VERSION = "kapso.expert_automated_review.v1"
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPERATION_PREIMAGE_FIELDS = {
    "input_artifact_checksums",
    "mcp_configuration_fingerprint",
    "review_contract_version",
    "review_packet_id",
    "reviewer",
    "sensitive_file_glob_scan_max_depth",
    "validation_configuration_fingerprint",
}
_INPUT_ARTIFACT_NAMES = {
    "invocation.json",
    "prior_knowledge.json",
    "prompt.txt",
    "response_schema.json",
}


class ExpertAutomatedReviewError(ValueError):
    """An automated-review fact or relation is invalid."""


class ExpertAutomatedReviewOutcome(str, Enum):
    PASSED = "passed"
    REJECTED = "rejected"
    DISPUTED = "disputed"


def _require_digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ContractValidationError(f"{name} must be a sha256 digest")


def _require_sorted_content_ids(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ContractValidationError(f"{name} must be non-empty, sorted, and unique")
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class ExpertAutomatedReviewPacket(StrictContract):
    """Immutable authority and evidence references for one review round."""

    review_packet_id: str
    validation_attempt_id: str
    authorization_transition_id: str
    authorization_state_id: str
    candidate_id: str
    candidate_tree_hash: str
    candidate_commit_record_id: str
    candidate_input_id: str
    proposer_operation_record_id: str
    trigger_evidence_packet_id: str
    trigger_decision_id: str
    scope_contract_id: str
    parent_release_id: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    accepted_stage_results: tuple[ExpertAcceptedStageResultRef, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-automated-review-packet"
    IDENTITY_FIELD: ClassVar[str] = "review_packet_id"

    def _validate(self) -> None:
        for value, name in (
            (self.validation_attempt_id, "review validation_attempt_id"),
            (self.authorization_transition_id, "review authorization_transition_id"),
            (self.authorization_state_id, "review authorization_state_id"),
            (self.candidate_id, "review candidate_id"),
            (self.candidate_commit_record_id, "review candidate_commit_record_id"),
            (self.candidate_input_id, "review candidate_input_id"),
            (
                self.proposer_operation_record_id,
                "review proposer_operation_record_id",
            ),
            (self.trigger_evidence_packet_id, "review trigger_evidence_packet_id"),
            (self.trigger_decision_id, "review trigger_decision_id"),
            (self.scope_contract_id, "review scope_contract_id"),
            (self.validation_policy_id, "review validation_policy_id"),
        ):
            require_content_id(value, name)
        if self.parent_release_id is not None:
            require_content_id(self.parent_release_id, "review parent_release_id")
        _require_digest(self.candidate_tree_hash, "review candidate_tree_hash")
        _require_digest(
            self.configuration_fingerprint,
            "review configuration_fingerprint",
        )
        if not self.accepted_stage_results:
            raise ContractValidationError(
                "automated review requires an accepted evaluator prefix"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "review exact_dependency_ids",
        )
        expected_dependencies = {
            self.validation_attempt_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.candidate_input_id,
            self.proposer_operation_record_id,
            self.trigger_evidence_packet_id,
            self.trigger_decision_id,
            self.scope_contract_id,
            self.validation_policy_id,
            *(result.stage_result_record_id for result in self.accepted_stage_results),
        }
        if self.parent_release_id is not None:
            expected_dependencies.add(self.parent_release_id)
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ContractValidationError(
                "automated review packet dependency closure is not exact"
            )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.exact_dependency_ids


@dataclass(frozen=True)
class ExpertAutomatedReviewAssertion(StrictContract):
    """One framework-owned assertion from one configured reviewer slot."""

    assertion_id: str
    review_packet_id: str
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    parent_release_id: str | None
    reviewer_id: str
    reviewer_role: str
    rubric_version: str
    judgment: str
    disposition: ExpertReviewDisposition
    rationale: str
    exact_evidence_ids: tuple[str, ...]
    review_operation_receipt_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-automated-review-assertion"
    IDENTITY_FIELD: ClassVar[str] = "assertion_id"

    def _validate(self) -> None:
        for value, name in (
            (self.review_packet_id, "assertion review_packet_id"),
            (self.validation_attempt_id, "assertion validation_attempt_id"),
            (self.candidate_id, "assertion candidate_id"),
            (
                self.review_operation_receipt_id,
                "assertion review_operation_receipt_id",
            ),
        ):
            require_content_id(value, name)
        if self.parent_release_id is not None:
            require_content_id(self.parent_release_id, "assertion parent_release_id")
        _require_digest(self.candidate_tree_hash, "assertion candidate_tree_hash")
        for value, name in (
            (self.reviewer_id, "assertion reviewer_id"),
            (self.reviewer_role, "assertion reviewer_role"),
            (self.rubric_version, "assertion rubric_version"),
            (self.judgment, "assertion judgment"),
        ):
            require_identifier(value, name)
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ContractValidationError("assertion rationale must be non-empty")
        _require_sorted_content_ids(
            self.exact_evidence_ids,
            "assertion exact_evidence_ids",
        )


@dataclass(frozen=True)
class ExpertAutomatedReviewOperationRecord(StrictContract):
    """Exact sealed coding-agent operation that produced one assertion."""

    operation_record_id: str
    review_packet_id: str
    operation_preimage: Mapping[str, Any]
    operation_receipt: CodingAgentOperationReceipt
    final_output: str
    artifact_payloads_base64: Mapping[str, str]
    produced_assertion_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-automated-review-operation"
    IDENTITY_FIELD: ClassVar[str] = "operation_record_id"

    def _validate(self) -> None:
        require_content_id(self.review_packet_id, "review operation packet ID")
        require_content_id(
            self.produced_assertion_id,
            "review operation produced assertion ID",
        )
        if set(self.operation_preimage) != _OPERATION_PREIMAGE_FIELDS:
            raise ContractValidationError(
                "automated review operation preimage fields are invalid"
            )
        if (
            self.operation_preimage["review_contract_version"]
            != EXPERT_AUTOMATED_REVIEW_CONTRACT_VERSION
            or self.operation_preimage["review_packet_id"] != self.review_packet_id
        ):
            raise ContractValidationError(
                "automated review operation preimage binding differs"
            )
        reviewer_payload = self.operation_preimage["reviewer"]
        if not isinstance(reviewer_payload, MappingABC):
            raise ContractValidationError("automated review reviewer is invalid")
        reviewer = ExpertReviewerSettings.from_dict(reviewer_payload)
        input_checksums = self.operation_preimage["input_artifact_checksums"]
        if (
            not isinstance(input_checksums, MappingABC)
            or set(input_checksums) != _INPUT_ARTIFACT_NAMES
        ):
            raise ContractValidationError(
                "automated review input checksums are invalid"
            )
        for name, digest in input_checksums.items():
            _require_digest(digest, f"automated review input checksum {name}")
        _require_digest(
            self.operation_preimage["mcp_configuration_fingerprint"],
            "automated review MCP configuration fingerprint",
        )
        _require_digest(
            self.operation_preimage["validation_configuration_fingerprint"],
            "automated review validation configuration fingerprint",
        )
        scan_depth = self.operation_preimage["sensitive_file_glob_scan_max_depth"]
        if type(scan_depth) is not int or scan_depth <= 0:
            raise ContractValidationError(
                "automated review sensitive scan depth must be positive"
            )
        expected_operation_id = (
            "agent_call_"
            + tree_or_blob_digest(canonical_json_bytes(self.operation_preimage))[7:39]
        )
        receipt = self.operation_receipt
        if (
            receipt.operation_id != expected_operation_id
            or receipt.principal_id != reviewer.reviewer_id
            or receipt.role != reviewer.reviewer_role
            or receipt.cli != reviewer.agent.cli
            or receipt.model != reviewer.agent.model
            or receipt.effort != reviewer.agent.effort
            or receipt.workspace_access is not CodingAgentWorkspaceAccess.READ_ONLY
        ):
            raise ContractValidationError(
                "automated review receipt differs from its reviewer authority"
            )
        payloads: dict[str, bytes] = {}
        for name, payload_base64 in self.artifact_payloads_base64.items():
            if not isinstance(payload_base64, str):
                raise ContractValidationError(
                    "automated review artifact payload must be base64 text"
                )
            payloads[name] = base64.b64decode(payload_base64, validate=True)
        checksums = {
            name: tree_or_blob_digest(payload)
            for name, payload in sorted(payloads.items())
        }
        if checksums != dict(receipt.artifact_checksums):
            raise ContractValidationError(
                "automated review artifacts differ from their receipt"
            )
        verified = verify_coding_agent_operation_artifacts(
            operation_id=receipt.operation_id,
            workspace_access=CodingAgentWorkspaceAccess.READ_ONLY,
            artifact_bytes=payloads,
        )
        invocation = verified.invocation
        if (
            verified.final_output != self.final_output
            or verified.workspace_delta is not None
            or verified.prior_knowledge is not None
            or invocation["role"] != reviewer.reviewer_role
            or invocation["cli"] != reviewer.agent.cli
            or invocation["model"] != reviewer.agent.model
            or invocation["effort"] != reviewer.agent.effort
            or invocation["timeout_seconds"] != reviewer.agent.timeout_seconds
            or invocation["allowed_tools"] != []
            or invocation["sensitive_file_glob_scan_max_depth"] != scan_depth
            or verified.mcp_configuration_fingerprint
            != self.operation_preimage["mcp_configuration_fingerprint"]
        ):
            raise ContractValidationError(
                "automated review artifacts differ from their operation preimage"
            )
        for name in _INPUT_ARTIFACT_NAMES:
            if tree_or_blob_digest(payloads[name]) != input_checksums[name]:
                raise ContractValidationError(
                    "automated review input artifact differs from its preimage"
                )
        output = parse_json_bytes(self.final_output)
        if not isinstance(output, MappingABC) or set(output) != {
            "disposition",
            "judgment",
            "rationale",
        }:
            raise ContractValidationError(
                "automated review final output fields are invalid"
            )


@dataclass(frozen=True)
class ExpertAutomatedReviewAdjudication(StrictContract):
    """Deterministic aggregate over one assertion per configured slot."""

    adjudication_id: str
    review_packet_id: str
    validation_policy_id: str
    promotion_policy_version: str
    assertion_ids: tuple[str, ...]
    approval_reviewer_ids: tuple[str, ...]
    rejection_reviewer_ids: tuple[str, ...]
    outcome: ExpertAutomatedReviewOutcome

    CONTENT_NAMESPACE: ClassVar[str] = "expert-automated-review-adjudication"
    IDENTITY_FIELD: ClassVar[str] = "adjudication_id"

    def _validate(self) -> None:
        require_content_id(self.review_packet_id, "adjudication review_packet_id")
        require_content_id(
            self.validation_policy_id,
            "adjudication validation_policy_id",
        )
        require_identifier(
            self.promotion_policy_version,
            "adjudication promotion_policy_version",
        )
        _require_sorted_content_ids(self.assertion_ids, "adjudication assertion_ids")
        for values, name in (
            (self.approval_reviewer_ids, "adjudication approval reviewers"),
            (self.rejection_reviewer_ids, "adjudication rejection reviewers"),
        ):
            if values != tuple(sorted(set(values))):
                raise ContractValidationError(f"{name} must be sorted and unique")
            for value in values:
                require_identifier(value, name)
        if set(self.approval_reviewer_ids) & set(self.rejection_reviewer_ids):
            raise ContractValidationError("one reviewer cannot both approve and reject")


@dataclass(frozen=True)
class ExpertAutomatedReviewStageResultRecord(StrictContract):
    """Complete reference closure published as the review-stage decision."""

    stage_result_record_id: str
    validation_attempt_id: str
    authorization_transition_id: str
    authorization_state_id: str
    candidate_id: str
    candidate_tree_hash: str
    scope_contract_id: str
    parent_release_id: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    review_packet_id: str
    assertion_ids: tuple[str, ...]
    operation_record_ids: tuple[str, ...]
    operation_receipt_ids: tuple[str, ...]
    adjudication_id: str
    outcome: ExpertAutomatedReviewOutcome
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-automated-review-stage-result"
    IDENTITY_FIELD: ClassVar[str] = "stage_result_record_id"

    def _validate(self) -> None:
        for value, name in (
            (self.validation_attempt_id, "review result validation_attempt_id"),
            (
                self.authorization_transition_id,
                "review result authorization_transition_id",
            ),
            (self.authorization_state_id, "review result authorization_state_id"),
            (self.candidate_id, "review result candidate_id"),
            (self.scope_contract_id, "review result scope_contract_id"),
            (self.validation_policy_id, "review result validation_policy_id"),
            (self.review_packet_id, "review result review_packet_id"),
            (self.adjudication_id, "review result adjudication_id"),
        ):
            require_content_id(value, name)
        if self.parent_release_id is not None:
            require_content_id(
                self.parent_release_id, "review result parent_release_id"
            )
        _require_digest(self.candidate_tree_hash, "review result candidate_tree_hash")
        _require_digest(
            self.configuration_fingerprint,
            "review result configuration_fingerprint",
        )
        for values, name in (
            (self.assertion_ids, "review result assertion_ids"),
            (self.operation_record_ids, "review result operation_record_ids"),
            (self.operation_receipt_ids, "review result operation_receipt_ids"),
        ):
            _require_sorted_content_ids(values, name)
        if not (
            len(self.assertion_ids)
            == len(self.operation_record_ids)
            == len(self.operation_receipt_ids)
        ):
            raise ContractValidationError(
                "review result reviewer closure cardinality differs"
            )
        _require_sorted_content_ids(
            self.exact_dependency_ids,
            "review result exact_dependency_ids",
        )
        expected_dependencies = {
            self.validation_attempt_id,
            self.authorization_transition_id,
            self.authorization_state_id,
            self.candidate_id,
            self.scope_contract_id,
            self.validation_policy_id,
            self.review_packet_id,
            self.adjudication_id,
            *self.assertion_ids,
            *self.operation_record_ids,
            *self.operation_receipt_ids,
        }
        if self.parent_release_id is not None:
            expected_dependencies.add(self.parent_release_id)
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise ContractValidationError(
                "automated review result dependency closure is not exact"
            )
