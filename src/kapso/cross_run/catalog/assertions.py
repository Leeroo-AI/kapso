"""Authenticated, append-only review assertion adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from kapso.cross_run.canonical import canonical_json_bytes, parse_json_bytes
from kapso.cross_run.catalog.agent_operations import CatalogAgentOperationRecord
from kapso.cross_run.contracts import (
    CodingAgentOperationReceipt,
    ContractValidationError,
    IdentityConflictError,
    MissingReferenceError,
    ReviewAssertion,
)
from kapso.cross_run.settings import CatalogReviewerSettings


class AssertionRegistryError(ValueError):
    """Review assertion facts cannot form a trusted adjudicated view."""


@dataclass(frozen=True)
class AssertionAdjudication:
    """Deterministic active-vote view for one reviewed subject."""

    subject_id: str
    assertion_ids: tuple[str, ...]
    active_current_assertion_ids: tuple[str, ...]
    historical_assertion_ids: tuple[str, ...]
    approval_reviewer_ids: tuple[str, ...]
    rejection_reviewer_ids: tuple[str, ...]
    conflicted_reviewer_ids: tuple[str, ...]
    approval_quorum_met: bool
    rejection_quorum_met: bool
    disputed: bool


class ReviewRegistry:
    """Validate reviewer authority and reduce explicit assertion supersession."""

    def __init__(
        self,
        reviewers: tuple[CatalogReviewerSettings, ...],
        *,
        approval_judgment: str,
        rejection_judgment: str,
        required_approvals: int,
        required_rejections: int,
        prohibited_principal_ids: tuple[str, ...] = (),
    ) -> None:
        reviewer_ids = tuple(reviewer.reviewer_id for reviewer in reviewers)
        if reviewer_ids != tuple(sorted(set(reviewer_ids))):
            raise AssertionRegistryError(
                "reviewer configuration must be sorted and unique"
            )
        if approval_judgment == rejection_judgment:
            raise AssertionRegistryError("approval and rejection judgments must differ")
        if required_approvals <= 0 or required_approvals > len(reviewers):
            raise AssertionRegistryError("approval quorum is invalid")
        if required_rejections <= 0 or required_rejections > len(reviewers):
            raise AssertionRegistryError("rejection quorum is invalid")
        prohibited = set(prohibited_principal_ids)
        overlap = prohibited & set(reviewer_ids)
        if overlap:
            raise IdentityConflictError(
                "claim proposer and reviewer principals must be distinct"
            )
        self._reviewers = {reviewer.reviewer_id: reviewer for reviewer in reviewers}
        self._approval_judgment = approval_judgment
        self._rejection_judgment = rejection_judgment
        self._required_approvals = required_approvals
        self._required_rejections = required_rejections

    def adjudicate(
        self,
        *,
        assertions: tuple[ReviewAssertion, ...],
        receipts: tuple[CodingAgentOperationReceipt, ...],
        operation_records: tuple[CatalogAgentOperationRecord, ...],
        known_object_ids: tuple[str, ...],
    ) -> Mapping[str, AssertionAdjudication]:
        """Validate all facts and return subject views independent of input order."""

        assertion_by_id = self._unique_assertions(assertions)
        receipt_by_id = self._unique_receipts(receipts)
        operations_by_assertion = self._review_operations(
            operation_records,
            assertion_by_id,
            receipt_by_id,
        )
        known_ids = set(known_object_ids)
        if len(known_ids) != len(known_object_ids):
            raise IdentityConflictError("known object IDs must be unique")
        for assertion_id in sorted(assertion_by_id):
            self._validate_assertion(
                assertion_by_id[assertion_id],
                assertion_by_id,
                receipt_by_id,
                operations_by_assertion,
                known_ids,
            )
        self._reject_supersession_cycles(assertion_by_id)
        subject_ids = tuple(
            sorted({assertion.subject_id for assertion in assertion_by_id.values()})
        )
        return MappingProxyType(
            {
                subject_id: self._adjudicate_subject(
                    subject_id,
                    assertion_by_id,
                )
                for subject_id in subject_ids
            }
        )

    @staticmethod
    def _unique_assertions(
        assertions: tuple[ReviewAssertion, ...],
    ) -> dict[str, ReviewAssertion]:
        by_id = {assertion.assertion_id: assertion for assertion in assertions}
        if len(by_id) != len(assertions):
            raise IdentityConflictError("review assertion IDs must be unique")
        return by_id

    @staticmethod
    def _unique_receipts(
        receipts: tuple[CodingAgentOperationReceipt, ...],
    ) -> dict[str, CodingAgentOperationReceipt]:
        by_id = {receipt.operation_receipt_id: receipt for receipt in receipts}
        if len(by_id) != len(receipts):
            raise IdentityConflictError("review operation receipt IDs must be unique")
        operation_ids = tuple(receipt.operation_id for receipt in receipts)
        if len(operation_ids) != len(set(operation_ids)):
            raise IdentityConflictError(
                "one coding-agent operation cannot authenticate multiple receipts"
            )
        return by_id

    def _validate_assertion(
        self,
        assertion: ReviewAssertion,
        assertions: Mapping[str, ReviewAssertion],
        receipts: Mapping[str, CodingAgentOperationReceipt],
        operations: Mapping[str, CatalogAgentOperationRecord],
        known_ids: set[str],
    ) -> None:
        reviewer = self._reviewers.get(assertion.reviewer_id)
        if reviewer is None:
            raise AssertionRegistryError("assertion uses an unconfigured reviewer")
        current_rubric = assertion.rubric_version == reviewer.rubric_version
        if current_rubric and assertion.reviewer_role != reviewer.reviewer_role:
            raise IdentityConflictError("assertion reviewer role is forged")
        if current_rubric and assertion.judgment not in {
            self._approval_judgment,
            self._rejection_judgment,
        }:
            raise ContractValidationError("assertion judgment is not configured")
        receipt = receipts.get(assertion.review_operation_ref)
        if receipt is None:
            raise MissingReferenceError("assertion review receipt is absent")
        if (
            receipt.principal_id != assertion.reviewer_id
            or receipt.role != assertion.reviewer_role
        ):
            raise IdentityConflictError(
                "assertion receipt does not match its configured reviewer slot"
            )
        operation = operations[assertion.assertion_id]
        if operation.operation_receipt_id != receipt.operation_receipt_id:
            raise IdentityConflictError(
                "assertion operation uses another reviewer receipt"
            )
        operation.validate_receipt(receipt)
        self._validate_operation_output(assertion, receipt, operation)
        if assertion.subject_id not in known_ids:
            raise MissingReferenceError("assertion subject is absent")
        missing_evidence = set(assertion.exact_evidence_refs) - known_ids
        if missing_evidence:
            raise MissingReferenceError(
                "assertion evidence closure is incomplete: "
                f"{tuple(sorted(missing_evidence))}"
            )
        predecessor_id = assertion.supersedes_assertion_id
        if predecessor_id is not None:
            predecessor = assertions.get(predecessor_id)
            if predecessor is None:
                raise MissingReferenceError("superseded assertion is absent")
            if (
                predecessor.reviewer_id != assertion.reviewer_id
                or predecessor.subject_id != assertion.subject_id
            ):
                raise IdentityConflictError(
                    "assertion can supersede only the same reviewer and subject"
                )
            if (
                not current_rubric
                and predecessor.rubric_version == reviewer.rubric_version
            ):
                raise ContractValidationError(
                    "a stale rubric cannot supersede a current-rubric assertion"
                )

    @staticmethod
    def _review_operations(
        operations: tuple[CatalogAgentOperationRecord, ...],
        assertions: Mapping[str, ReviewAssertion],
        receipts: Mapping[str, CodingAgentOperationReceipt],
    ) -> dict[str, CatalogAgentOperationRecord]:
        by_assertion: dict[str, CatalogAgentOperationRecord] = {}
        receipt_ids: list[str] = []
        for operation in operations:
            if operation.operation_kind != "catalog_review":
                raise AssertionRegistryError(
                    "non-review operation entered review registry"
                )
            if len(operation.produced_object_ids) != 1:
                raise AssertionRegistryError(
                    "one review operation must produce exactly one assertion"
                )
            assertion_id = operation.produced_object_ids[0]
            if assertion_id in by_assertion:
                raise IdentityConflictError(
                    "one assertion has multiple review operations"
                )
            if operation.operation_receipt_id not in receipts:
                raise MissingReferenceError("review operation receipt is absent")
            by_assertion[assertion_id] = operation
            receipt_ids.append(operation.operation_receipt_id)
        if set(by_assertion) != set(assertions):
            raise MissingReferenceError(
                "every assertion requires exactly one review operation"
            )
        if len(receipt_ids) != len(set(receipt_ids)):
            raise IdentityConflictError(
                "one review receipt cannot authenticate multiple assertions"
            )
        return by_assertion

    @staticmethod
    def _validate_operation_output(
        assertion: ReviewAssertion,
        receipt: CodingAgentOperationReceipt,
        operation: CatalogAgentOperationRecord,
    ) -> None:
        preimage = operation.operation_preimage
        reviewer_payload = preimage.get("reviewer")
        if not isinstance(reviewer_payload, Mapping):
            raise ContractValidationError("review operation reviewer is absent")
        historical_reviewer = CatalogReviewerSettings.from_dict(reviewer_payload)
        if (
            historical_reviewer.reviewer_id != assertion.reviewer_id
            or historical_reviewer.reviewer_role != assertion.reviewer_role
            or historical_reviewer.rubric_version != assertion.rubric_version
            or historical_reviewer.agent.cli != receipt.cli
            or historical_reviewer.agent.model != receipt.model
            or historical_reviewer.agent.effort != receipt.effort
        ):
            raise IdentityConflictError("review operation uses another reviewer slot")
        packet = operation.packet_payload
        subject = packet.get("subject")
        evidence_records = packet.get("evidence_records")
        if (
            not isinstance(subject, Mapping)
            or subject.get("record_id") != assertion.subject_id
            or not isinstance(evidence_records, (list, tuple))
        ):
            raise IdentityConflictError("review operation packet subject is forged")
        evidence_ids = tuple(
            record.get("record_id") if isinstance(record, Mapping) else None
            for record in evidence_records
        )
        if evidence_ids != assertion.exact_evidence_refs:
            raise IdentityConflictError("review operation packet evidence differs")
        output = parse_json_bytes(operation.final_output.encode("utf-8"))
        expected_fields = {
            "exact_evidence_refs",
            "judgment",
            "rationale",
            "supersedes_assertion_id",
        }
        if not isinstance(output, Mapping) or set(output) != expected_fields:
            raise ContractValidationError("review operation output fields are invalid")
        if (
            output["judgment"] != assertion.judgment
            or output["rationale"] != assertion.rationale
            or tuple(output["exact_evidence_refs"]) != assertion.exact_evidence_refs
            or output["supersedes_assertion_id"] != assertion.supersedes_assertion_id
        ):
            raise IdentityConflictError(
                "review assertion differs from the authenticated model output"
            )

    @staticmethod
    def _reject_supersession_cycles(
        assertions: Mapping[str, ReviewAssertion],
    ) -> None:
        for assertion_id in sorted(assertions):
            visited: set[str] = set()
            cursor: str | None = assertion_id
            while cursor is not None:
                if cursor in visited:
                    raise ContractValidationError(
                        "assertion supersession contains a cycle"
                    )
                visited.add(cursor)
                cursor = assertions[cursor].supersedes_assertion_id

    def _adjudicate_subject(
        self,
        subject_id: str,
        assertions: Mapping[str, ReviewAssertion],
    ) -> AssertionAdjudication:
        subject_assertions = tuple(
            assertion
            for assertion in assertions.values()
            if assertion.subject_id == subject_id
        )
        superseded_ids = {
            assertion.supersedes_assertion_id
            for assertion in subject_assertions
            if assertion.supersedes_assertion_id is not None
        }
        heads = tuple(
            assertion
            for assertion in subject_assertions
            if assertion.assertion_id not in superseded_ids
        )
        current_heads = tuple(
            assertion
            for assertion in heads
            if assertion.rubric_version
            == self._reviewers[assertion.reviewer_id].rubric_version
        )
        judgments_by_reviewer: dict[str, set[str]] = {}
        for assertion in current_heads:
            judgments_by_reviewer.setdefault(assertion.reviewer_id, set()).add(
                assertion.judgment
            )
        conflicted_reviewers = tuple(
            sorted(
                reviewer_id
                for reviewer_id, judgments in judgments_by_reviewer.items()
                if len(judgments) > 1
            )
        )
        approvals = tuple(
            sorted(
                reviewer_id
                for reviewer_id, judgments in judgments_by_reviewer.items()
                if judgments == {self._approval_judgment}
            )
        )
        rejections = tuple(
            sorted(
                reviewer_id
                for reviewer_id, judgments in judgments_by_reviewer.items()
                if judgments == {self._rejection_judgment}
            )
        )
        active_ids = tuple(
            sorted(assertion.assertion_id for assertion in current_heads)
        )
        all_ids = tuple(
            sorted(assertion.assertion_id for assertion in subject_assertions)
        )
        historical_ids = tuple(sorted(set(all_ids) - set(active_ids)))
        approval_quorum = len(approvals) >= self._required_approvals
        rejection_quorum = len(rejections) >= self._required_rejections
        disputed = bool(
            conflicted_reviewers or (approvals and rejections) or rejection_quorum
        )
        return AssertionAdjudication(
            subject_id=subject_id,
            assertion_ids=all_ids,
            active_current_assertion_ids=active_ids,
            historical_assertion_ids=historical_ids,
            approval_reviewer_ids=approvals,
            rejection_reviewer_ids=rejections,
            conflicted_reviewer_ids=conflicted_reviewers,
            approval_quorum_met=approval_quorum,
            rejection_quorum_met=rejection_quorum,
            disputed=disputed,
        )
