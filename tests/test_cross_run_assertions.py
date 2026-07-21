import json

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.catalog.agent_operations import (
    CatalogAgentOperationRecord,
    catalog_agent_operation_id,
)
from kapso.cross_run.catalog.assertions import ReviewRegistry
from kapso.cross_run.contracts import (
    CodingAgentOperationReceipt,
    IdentityConflictError,
    MissingReferenceError,
    ReviewAssertion,
)
from kapso.cross_run.settings import CrossRunSettings

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
ARTIFACT_FILENAMES = (
    "final.json",
    "invocation.json",
    "prompt.txt",
    "response_schema.json",
    "result.json",
    "stderr.txt",
    "stdout.txt",
)


def fixture_id(name):
    return content_id("assertion-test-fixture", {"name": name})


def digest(name):
    return tree_or_blob_digest(name.encode("utf-8"))


def settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).catalog


def review_facts(
    reviewer,
    name,
    subject_id,
    evidence_id,
    *,
    judgment="approve",
    rubric_version=None,
    supersedes=None,
    rationale="The exact evidence satisfies the scientific rubric.",
    principal_id=None,
):
    effective_rubric_version = rubric_version or reviewer.rubric_version
    output = {
        "judgment": judgment,
        "rationale": rationale,
        "exact_evidence_refs": [evidence_id],
        "supersedes_assertion_id": supersedes,
    }
    final_output = json.dumps(output, sort_keys=True)
    preimage = {
        "packet": {
            "subject": {"record_id": subject_id},
            "evidence_records": ({"record_id": evidence_id},),
        },
        "template": f"fixture-{name}",
        "schema": {"type": "object"},
        "reviewer": {
            **reviewer.to_dict(),
            "rubric_version": effective_rubric_version,
        },
    }
    operation_receipt = CodingAgentOperationReceipt.mint(
        operation_id=catalog_agent_operation_id(preimage),
        principal_id=principal_id or reviewer.reviewer_id,
        role=reviewer.reviewer_role,
        cli=reviewer.agent.cli,
        model=reviewer.agent.model,
        effort=reviewer.agent.effort,
        artifact_checksums={
            filename: (
                tree_or_blob_digest(final_output.encode("utf-8"))
                if filename == "final.json"
                else digest(f"{name}-{filename}")
            )
            for filename in ARTIFACT_FILENAMES
        },
    )
    review_assertion = ReviewAssertion.mint(
        subject_id=subject_id,
        reviewer_id=reviewer.reviewer_id,
        reviewer_role=reviewer.reviewer_role,
        rubric_version=effective_rubric_version,
        judgment=judgment,
        rationale=rationale,
        exact_evidence_refs=(evidence_id,),
        supersedes_assertion_id=supersedes,
        review_operation_ref=operation_receipt.operation_receipt_id,
    )
    operation_record = CatalogAgentOperationRecord.mint(
        operation_kind="catalog_review",
        operation_receipt_id=operation_receipt.operation_receipt_id,
        operation_preimage=preimage,
        final_output=final_output,
        produced_object_ids=(review_assertion.assertion_id,),
    )
    return review_assertion, operation_receipt, operation_record


def registry(catalog_settings):
    return ReviewRegistry(
        catalog_settings.reviewers,
        approval_judgment=catalog_settings.admission.approval_judgment,
        rejection_judgment=catalog_settings.admission.rejection_judgment,
        required_approvals=catalog_settings.admission.required_approvals,
        required_rejections=catalog_settings.admission.required_rejections,
        prohibited_principal_ids=(catalog_settings.claim_proposer_id,),
    )


def test_independent_current_rubric_votes_reach_quorum_deterministically():
    catalog_settings = settings()
    subject_id = fixture_id("subject")
    evidence_id = fixture_id("evidence")
    facts = tuple(
        review_facts(
            reviewer,
            f"approval-{position}",
            subject_id,
            evidence_id,
        )
        for position, reviewer in enumerate(catalog_settings.reviewers)
    )
    assertions = tuple(item[0] for item in facts)
    receipts = tuple(item[1] for item in facts)
    operations = tuple(item[2] for item in facts)

    forward = registry(catalog_settings).adjudicate(
        assertions=assertions,
        receipts=receipts,
        operation_records=operations,
        known_object_ids=(subject_id, evidence_id),
    )[subject_id]
    reversed_view = registry(catalog_settings).adjudicate(
        assertions=tuple(reversed(assertions)),
        receipts=tuple(reversed(receipts)),
        operation_records=tuple(reversed(operations)),
        known_object_ids=(evidence_id, subject_id),
    )[subject_id]

    assert forward == reversed_view
    assert forward.approval_quorum_met
    assert not forward.rejection_quorum_met
    assert not forward.disputed
    assert forward.approval_reviewer_ids == tuple(
        reviewer.reviewer_id for reviewer in catalog_settings.reviewers
    )


def test_stale_rubric_is_historical_and_explicit_supersession_selects_new_head():
    catalog_settings = settings()
    reviewer = catalog_settings.reviewers[0]
    subject_id = fixture_id("stale-subject")
    evidence_id = fixture_id("stale-evidence")
    stale, stale_receipt, stale_operation = review_facts(
        reviewer,
        "stale",
        subject_id,
        evidence_id,
        judgment="legacy_pass",
        rubric_version="kapso.catalog_review.previous",
    )
    current, current_receipt, current_operation = review_facts(
        reviewer,
        "current",
        subject_id,
        evidence_id,
        supersedes=stale.assertion_id,
    )

    view = registry(catalog_settings).adjudicate(
        assertions=(current, stale),
        receipts=(current_receipt, stale_receipt),
        operation_records=(current_operation, stale_operation),
        known_object_ids=(subject_id, evidence_id),
    )[subject_id]

    assert view.active_current_assertion_ids == (current.assertion_id,)
    assert view.historical_assertion_ids == (stale.assertion_id,)
    assert view.approval_reviewer_ids == (reviewer.reviewer_id,)
    assert not view.rejection_reviewer_ids


def test_conflicting_same_reviewer_branches_are_preserved_but_count_no_vote():
    catalog_settings = settings()
    reviewer = catalog_settings.reviewers[0]
    subject_id = fixture_id("branch-subject")
    evidence_id = fixture_id("branch-evidence")
    approval, approval_receipt, approval_operation = review_facts(
        reviewer,
        "branch-approval",
        subject_id,
        evidence_id,
    )
    rejection, rejection_receipt, rejection_operation = review_facts(
        reviewer,
        "branch-rejection",
        subject_id,
        evidence_id,
        judgment="reject",
    )

    view = registry(catalog_settings).adjudicate(
        assertions=(approval, rejection),
        receipts=(approval_receipt, rejection_receipt),
        operation_records=(approval_operation, rejection_operation),
        known_object_ids=(subject_id, evidence_id),
    )[subject_id]

    assert view.active_current_assertion_ids == tuple(
        sorted((approval.assertion_id, rejection.assertion_id))
    )
    assert view.conflicted_reviewer_ids == (reviewer.reviewer_id,)
    assert not view.approval_reviewer_ids
    assert not view.rejection_reviewer_ids
    assert view.disputed


def test_registry_rejects_forged_receipt_missing_evidence_and_cross_reviewer_lineage():
    catalog_settings = settings()
    first, second = catalog_settings.reviewers
    subject_id = fixture_id("invalid-subject")
    evidence_id = fixture_id("invalid-evidence")
    first_assertion, first_receipt, first_operation = review_facts(
        first,
        "first",
        subject_id,
        evidence_id,
    )
    cross_reviewer, second_receipt, second_operation = review_facts(
        second,
        "second",
        subject_id,
        evidence_id,
        supersedes=first_assertion.assertion_id,
    )

    with pytest.raises(MissingReferenceError):
        registry(catalog_settings).adjudicate(
            assertions=(first_assertion,),
            receipts=(first_receipt,),
            operation_records=(first_operation,),
            known_object_ids=(subject_id,),
        )
    with pytest.raises(IdentityConflictError):
        registry(catalog_settings).adjudicate(
            assertions=(first_assertion, cross_reviewer),
            receipts=(first_receipt, second_receipt),
            operation_records=(first_operation, second_operation),
            known_object_ids=(subject_id, evidence_id),
        )

    forged_assertion, forged_receipt, forged_operation = review_facts(
        first,
        "forged",
        subject_id,
        evidence_id,
        principal_id=second.reviewer_id,
    )
    with pytest.raises(IdentityConflictError):
        registry(catalog_settings).adjudicate(
            assertions=(forged_assertion,),
            receipts=(forged_receipt,),
            operation_records=(forged_operation,),
            known_object_ids=(subject_id, evidence_id),
        )


def test_claim_proposer_cannot_occupy_a_reviewer_slot():
    catalog_settings = settings()
    with pytest.raises(IdentityConflictError):
        ReviewRegistry(
            catalog_settings.reviewers,
            approval_judgment=catalog_settings.admission.approval_judgment,
            rejection_judgment=catalog_settings.admission.rejection_judgment,
            required_approvals=catalog_settings.admission.required_approvals,
            required_rejections=catalog_settings.admission.required_rejections,
            prohibited_principal_ids=(catalog_settings.reviewers[0].reviewer_id,),
        )


def test_authenticated_review_receipt_cannot_be_replayed_for_an_altered_assertion():
    catalog_settings = settings()
    reviewer = catalog_settings.reviewers[0]
    subject_id = fixture_id("replayed-subject")
    evidence_id = fixture_id("replayed-evidence")
    original, receipt, operation = review_facts(
        reviewer,
        "replayed-original",
        subject_id,
        evidence_id,
    )
    forged = ReviewAssertion.mint(
        subject_id=original.subject_id,
        reviewer_id=original.reviewer_id,
        reviewer_role=original.reviewer_role,
        rubric_version=original.rubric_version,
        judgment="reject",
        rationale="This text was never emitted by the authenticated reviewer.",
        exact_evidence_refs=original.exact_evidence_refs,
        supersedes_assertion_id=None,
        review_operation_ref=receipt.operation_receipt_id,
    )
    replayed_operation = CatalogAgentOperationRecord.mint(
        operation_kind=operation.operation_kind,
        operation_receipt_id=operation.operation_receipt_id,
        operation_preimage=operation.operation_preimage,
        final_output=operation.final_output,
        produced_object_ids=(forged.assertion_id,),
    )

    with pytest.raises(IdentityConflictError, match="authenticated model output"):
        registry(catalog_settings).adjudicate(
            assertions=(forged,),
            receipts=(receipt,),
            operation_records=(replayed_operation,),
            known_object_ids=(subject_id, evidence_id),
        )
