"""Explicit authenticated Codex production check for M4 agent roles.

Run directly; the filename intentionally stays outside normal pytest discovery.
"""

from __future__ import annotations

from pathlib import Path

from kapso.cross_run.catalog.claims import ClaimProposer
from kapso.cross_run.catalog.reviews import CatalogReviewer, CatalogReviewPacket
from kapso.execution.coding_agents.structured_call import (
    CodingAgentRunnerSettings,
    SubprocessCodingAgentCallRunner,
)
from test_cross_run_claim_proposer import catalog_settings, packet


def _workspace_snapshot(workspace: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (str(path.relative_to(workspace)), path.read_bytes())
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    )


def _review_packet(proposal_packet, proposal) -> CatalogReviewPacket:
    claim = proposal.claims[0]
    closure = proposal.claim_evidence_closures[0]
    evidence_ids = set(closure.evaluated_episode_ids)
    return CatalogReviewPacket(
        catalog_generation_id=proposal_packet.catalog_generation_id,
        catalog_generation=proposal_packet.catalog_generation,
        scope_contract=proposal_packet.scope_contract,
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
            for episode in proposal_packet.episodes
            if episode.episode_id in evidence_ids
        ),
        proposer_operation_receipt=proposal.operation_receipt,
        claim_evidence_closure=closure,
        previous_assertions=(),
        previous_operation_receipts=(),
    )


def test_authenticated_codex_proposes_reviews_and_replays_once(tmp_path: Path) -> None:
    settings = catalog_settings()
    artifact_root = (tmp_path / "agent-artifacts").resolve()
    runner = SubprocessCodingAgentCallRunner(
        CodingAgentRunnerSettings(
            artifact_root=str(artifact_root),
            termination_grace_seconds=settings.termination_grace_seconds,
        )
    )
    workspace = (tmp_path / "empty-agent-workspace").resolve()
    workspace.mkdir()
    initial_workspace = _workspace_snapshot(workspace)
    proposal_packet = packet()
    proposer = ClaimProposer(settings, runner)

    proposal = proposer.propose(proposal_packet, workspace)
    proposal_result_path = (
        artifact_root / proposal.operation_receipt.operation_id / "result.json"
    )
    proposal_result_mtime = proposal_result_path.stat().st_mtime_ns
    replayed_proposal = proposer.propose(proposal_packet, workspace)

    assert proposal.claims
    assert replayed_proposal.claims == proposal.claims
    assert replayed_proposal.operation_receipt == proposal.operation_receipt
    assert proposal_result_path.stat().st_mtime_ns == proposal_result_mtime

    review_packet = _review_packet(proposal_packet, proposal)
    reviews = []
    for reviewer_slot in settings.reviewers:
        reviewer = CatalogReviewer(settings, runner)
        review = reviewer.review(review_packet, reviewer_slot, workspace)
        result_path = (
            artifact_root / review.operation_receipt.operation_id / "result.json"
        )
        result_mtime = result_path.stat().st_mtime_ns
        replayed_review = reviewer.review(review_packet, reviewer_slot, workspace)
        assert replayed_review.assertion == review.assertion
        assert replayed_review.operation_receipt == review.operation_receipt
        assert result_path.stat().st_mtime_ns == result_mtime
        reviews.append(review)

    assert len(reviews) == 2
    assert len({review.assertion.reviewer_id for review in reviews}) == 2
    assert all(
        review.assertion.subject_id == proposal.claims[0].revision_id
        for review in reviews
    )
    assert _workspace_snapshot(workspace) == initial_workspace
