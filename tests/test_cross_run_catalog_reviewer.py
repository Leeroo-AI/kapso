from dataclasses import replace

import pytest

from kapso.cross_run.catalog.claims import ClaimProposalPacket, ClaimProposer
from kapso.cross_run.catalog.reviews import (
    CatalogReviewError,
    CatalogReviewer,
    CatalogReviewPacket,
)
from kapso.cross_run.contracts import ReviewAssertion, TransferEpisode
from test_cross_run_claim_proposer import (
    ArtifactFakeRunner,
    catalog_settings,
    packet,
    valid_output,
)
from test_cross_run_contracts import fixture_id


def proposed_claim(tmp_path):
    proposal_packet = packet()
    proposal_runner = ArtifactFakeRunner(
        tmp_path / "proposal-artifacts",
        valid_output(proposal_packet),
    )
    proposal_workspace = tmp_path / "proposal-workspace"
    proposal_workspace.mkdir()
    result = ClaimProposer(catalog_settings(), proposal_runner).propose(
        proposal_packet,
        proposal_workspace,
    )
    return proposal_packet, result


def review_packet(tmp_path, previous_assertions=(), previous_receipts=()):
    proposal_packet, proposal = proposed_claim(tmp_path)
    claim = proposal.claims[0]
    closure = proposal.claim_evidence_closures[0]
    evidence = tuple(
        {
            "record_id": episode.episode_id,
            "record_kind": "transfer_episode",
            "payload": episode.to_dict(),
        }
        for episode in proposal_packet.episodes
        if episode.episode_id in set(closure.evaluated_episode_ids)
    )
    return CatalogReviewPacket(
        catalog_generation_id=proposal_packet.catalog_generation_id,
        catalog_generation=proposal_packet.catalog_generation,
        scope_contract=proposal_packet.scope_contract,
        subject={
            "record_id": claim.revision_id,
            "record_kind": "knowledge_claim",
            "payload": claim.to_dict(),
        },
        evidence_records=evidence,
        proposer_operation_receipt=proposal.operation_receipt,
        claim_evidence_closure=closure,
        previous_assertions=tuple(
            sorted(previous_assertions, key=lambda assertion: assertion.assertion_id)
        ),
        previous_operation_receipts=tuple(
            sorted(
                previous_receipts,
                key=lambda receipt: receipt.operation_receipt_id,
            )
        ),
    )


def reviewer_output(review, supersedes=None):
    return {
        "judgment": "approve",
        "rationale": "The bounded mechanism is supported by the exact evidence.",
        "exact_evidence_refs": list(review.evidence_record_ids),
        "supersedes_assertion_id": supersedes,
    }


def test_catalog_reviewer_framework_assigns_identity_and_receipt(tmp_path):
    review = review_packet(tmp_path)
    settings = catalog_settings()
    reviewer = settings.reviewers[0]
    runner = ArtifactFakeRunner(
        tmp_path / "review-artifacts",
        reviewer_output(review),
    )
    workspace = tmp_path / "review-workspace"
    workspace.mkdir()

    result = CatalogReviewer(settings, runner).review(review, reviewer, workspace)

    assert result.assertion.subject_id == review.subject_id
    assert result.assertion.reviewer_id == reviewer.reviewer_id
    assert result.assertion.reviewer_role == reviewer.reviewer_role
    assert result.assertion.rubric_version == reviewer.rubric_version
    assert result.assertion.exact_evidence_refs == review.evidence_record_ids
    assert result.operation_receipt.principal_id == reviewer.reviewer_id
    assert result.assertion.review_operation_ref == (
        result.operation_receipt.operation_receipt_id
    )
    assert runner.requests[0].effort == "xhigh"
    assert runner.requests[0].allowed_tools == ()
    assert review.subject["payload"]["statement"] in runner.requests[0].prompt
    assert tuple(workspace.iterdir()) == ()


def test_claim_review_packet_exposes_not_applicable_episode_and_rationale(tmp_path):
    base_packet = packet()
    base_episode = base_packet.episodes[0]
    second_episode = TransferEpisode.mint(
        source={
            **dict(base_episode.source),
            "idea_id": "idea_not_applicable",
            "node_id": "node_not_applicable",
        },
        source_bundle_id=fixture_id("not-applicable-bundle"),
        supersedes_projection_id=None,
        task_context_binding=base_episode.task_context_binding,
        artifact_environment=base_episode.artifact_environment,
        proposal="Evaluate an unrelated intervention for closure completeness.",
        parent_episode_ref=None,
        attempts=base_episode.attempts,
        terminal_attempt_revision=base_episode.terminal_attempt_revision,
        safe_observation_refs=(),
        sanitation_report_id=fixture_id("not-applicable-sanitation"),
        derivation_refs=(fixture_id("not-applicable-derivation"),),
    )
    episodes = tuple(
        sorted((*base_packet.episodes, second_episode), key=lambda x: x.episode_id)
    )
    extended_packet = ClaimProposalPacket(
        catalog_generation_id=base_packet.catalog_generation_id,
        catalog_generation=base_packet.catalog_generation,
        scope_contract=base_packet.scope_contract,
        episodes=episodes,
        prior_ideas=base_packet.prior_ideas,
        existing_claims=base_packet.existing_claims,
        entry_states=base_packet.entry_states,
        review_assertions=base_packet.review_assertions,
        operation_receipts=base_packet.operation_receipts,
        proof_reference_ids=tuple(
            sorted(
                {
                    *base_packet.proof_reference_ids,
                    second_episode.source_bundle_id,
                    second_episode.sanitation_report_id,
                    *second_episode.derivation_refs,
                }
            )
        ),
    )
    output = valid_output(extended_packet)
    support_episode_id = output["claims"][0]["evidence_assessments"][0]["episode_id"]
    not_applicable_episode = next(
        episode for episode in episodes if episode.episode_id != support_episode_id
    )
    output["claims"][0]["evidence_assessments"].append(
        {
            "episode_id": not_applicable_episode.episode_id,
            "relationship": "not_applicable",
            "rationale": "This episode targets a different mechanism than the claim.",
        }
    )
    output["claims"][0]["evidence_assessments"].sort(
        key=lambda assessment: assessment["episode_id"]
    )
    proposal_runner = ArtifactFakeRunner(tmp_path / "proposal-artifacts", output)
    proposal_workspace = tmp_path / "proposal-workspace"
    proposal_workspace.mkdir()
    proposal = ClaimProposer(catalog_settings(), proposal_runner).propose(
        extended_packet,
        proposal_workspace,
    )
    claim = proposal.claims[0]
    closure = proposal.claim_evidence_closures[0]
    review = CatalogReviewPacket(
        catalog_generation_id=extended_packet.catalog_generation_id,
        catalog_generation=extended_packet.catalog_generation,
        scope_contract=extended_packet.scope_contract,
        subject={
            "record_id": claim.revision_id,
            "record_kind": "knowledge_claim",
            "payload": claim.to_dict(),
        },
        evidence_records=tuple(
            {
                "record_id": episode.episode_id,
                "record_kind": "transfer_episode",
                "payload": episode.to_dict(),
            }
            for episode in episodes
        ),
        proposer_operation_receipt=proposal.operation_receipt,
        claim_evidence_closure=closure,
        previous_assertions=(),
        previous_operation_receipts=(),
    )

    assert closure.not_applicable_episode_ids == (not_applicable_episode.episode_id,)
    assert review.evidence_record_ids == closure.evaluated_episode_ids
    assessment = next(
        value
        for value in review.claim_evidence_closure.evidence_assessments
        if value["episode_id"] == not_applicable_episode.episode_id
    )
    assert assessment["rationale"] == (
        "This episode targets a different mechanism than the claim."
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda output, review: output.__setitem__("reviewer_id", "forged"),
        lambda output, review: output.__setitem__("exact_evidence_refs", []),
        lambda output, review: output.__setitem__("judgment", "abstain"),
        lambda output, review: output.__setitem__("rationale", ""),
        lambda output, review: output.__setitem__(
            "supersedes_assertion_id", review.subject_id
        ),
    ],
)
def test_catalog_reviewer_rejects_forged_or_incomplete_output(
    tmp_path,
    mutate,
):
    review = review_packet(tmp_path)
    output = reviewer_output(review)
    mutate(output, review)
    settings = catalog_settings()
    runner = ArtifactFakeRunner(tmp_path / "review-artifacts", output)
    workspace = tmp_path / "review-workspace"
    workspace.mkdir()

    with pytest.raises((CatalogReviewError, ValueError)):
        CatalogReviewer(settings, runner).review(
            review,
            settings.reviewers[0],
            workspace,
        )


def test_catalog_reviewer_requires_configured_slot(tmp_path):
    review = review_packet(tmp_path)
    settings = catalog_settings()
    unknown = replace(settings.reviewers[0], reviewer_id="unknown_reviewer")
    runner = ArtifactFakeRunner(
        tmp_path / "review-artifacts",
        reviewer_output(review),
    )
    workspace = tmp_path / "review-workspace"
    workspace.mkdir()

    with pytest.raises(CatalogReviewError, match="not configured"):
        CatalogReviewer(settings, runner).review(review, unknown, workspace)
    assert runner.requests == []


def test_catalog_reviewer_supersedes_its_single_active_head(tmp_path):
    first_packet = review_packet(tmp_path)
    settings = catalog_settings()
    reviewer = settings.reviewers[0]
    first_runner = ArtifactFakeRunner(
        tmp_path / "first-review-artifacts",
        reviewer_output(first_packet),
    )
    first_workspace = tmp_path / "first-review-workspace"
    first_workspace.mkdir()
    first = CatalogReviewer(settings, first_runner).review(
        first_packet,
        reviewer,
        first_workspace,
    )
    second_packet = CatalogReviewPacket(
        **{
            **first_packet.to_dict(),
            "previous_assertions": (first.assertion,),
            "previous_operation_receipts": (first.operation_receipt,),
        }
    )
    settings = replace(
        settings,
        review_packet_record_limit=len(second_packet.evidence_records),
    )
    second_runner = ArtifactFakeRunner(
        tmp_path / "second-review-artifacts",
        reviewer_output(second_packet, first.assertion.assertion_id),
    )
    second_workspace = tmp_path / "second-review-workspace"
    second_workspace.mkdir()

    second = CatalogReviewer(settings, second_runner).review(
        second_packet,
        reviewer,
        second_workspace,
    )

    assert second.assertion.supersedes_assertion_id == first.assertion.assertion_id


def test_catalog_reviewer_refuses_ambiguous_same_reviewer_heads(tmp_path):
    base = review_packet(tmp_path)
    settings = catalog_settings()
    reviewer = settings.reviewers[0]
    first_runner = ArtifactFakeRunner(
        tmp_path / "first-review-artifacts",
        reviewer_output(base),
    )
    first_workspace = tmp_path / "first-review-workspace"
    first_workspace.mkdir()
    first = CatalogReviewer(settings, first_runner).review(
        base,
        reviewer,
        first_workspace,
    )
    competing = ReviewAssertion.mint(
        subject_id=first.assertion.subject_id,
        reviewer_id=first.assertion.reviewer_id,
        reviewer_role=first.assertion.reviewer_role,
        rubric_version=first.assertion.rubric_version,
        judgment="reject",
        rationale="A competing unsuperseded review head.",
        exact_evidence_refs=first.assertion.exact_evidence_refs,
        supersedes_assertion_id=None,
        review_operation_ref=first.operation_receipt.operation_receipt_id,
    )
    ambiguous = CatalogReviewPacket(
        **{
            **base.to_dict(),
            "previous_assertions": tuple(
                sorted(
                    (first.assertion, competing),
                    key=lambda assertion: assertion.assertion_id,
                )
            ),
            "previous_operation_receipts": (first.operation_receipt,),
        }
    )
    runner = ArtifactFakeRunner(
        tmp_path / "ambiguous-review-artifacts",
        reviewer_output(ambiguous),
    )
    workspace = tmp_path / "ambiguous-review-workspace"
    workspace.mkdir()

    with pytest.raises(CatalogReviewError, match="multiple heads"):
        CatalogReviewer(settings, runner).review(ambiguous, reviewer, workspace)
