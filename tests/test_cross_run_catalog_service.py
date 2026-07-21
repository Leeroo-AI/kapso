"""End-to-end M3 projection, catalog reduction, claim, and review flow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.catalog.claims import ClaimProposer
from kapso.cross_run.catalog.agent_operations import CatalogAgentOperationRecord
from kapso.cross_run.catalog.projector import RunBundleProjector
from kapso.cross_run.catalog.reducer import CatalogFactError
from kapso.cross_run.catalog.reviews import CatalogReviewer
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import (
    AdmissionState,
    CompletionState,
    ExpertScopeContract,
)
from cross_run_capture_fixtures import make_capture_fixture
from test_cross_run_catalog_reviewer import reviewer_output
from test_cross_run_claim_proposer import ArtifactFakeRunner, valid_output
from test_cross_run_contracts import build_records, fixture_id


def _scope_contract() -> ExpertScopeContract:
    return next(
        record for record in build_records() if isinstance(record, ExpertScopeContract)
    )


def _project_real_bundle(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    fixture = make_capture_fixture(run_root)
    pipeline = RunCapturePipeline(RunCaptureContext(fixture.request), fixture.settings)
    stored = pipeline.capture_if_due(CompletionState.STOPPED, force=True)
    assert stored is not None
    projection = RunBundleProjector(
        fixture.settings.capture.score_comparison_tolerance
    ).project(stored)
    return fixture, projection


def test_complete_catalog_flow_preserves_authority_boundaries(tmp_path: Path) -> None:
    fixture, projection = _project_real_bundle(tmp_path)
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        _scope_contract(),
        fixture.settings.catalog,
    )
    initial = catalog.store.read_current()

    projected = catalog.publish_projection(initial, projection).generation

    assert projected.generation_number == 1
    assert dict(projected.bundle_frontier) == {
        (
            f"{projection.source_bundle.scope_id}/"
            f"{projection.source_bundle.run_id}/"
            f"{projection.source_bundle.campaign_id}"
        ): projection.source_bundle.bundle_id
    }
    projected_view = catalog.read_generation(projected)
    assert {state.admission_state for state in projected_view.entry_states} == {
        AdmissionState.ADMITTED
    }
    assert projection.projection_manifest.projection_manifest_id in (
        projected.fact_object_ids
    )
    assert {event.event_id for event in projection.derivation_objects}.issubset(
        projected.fact_object_ids
    )

    proposal_packet = catalog.claim_proposal_packet(
        projected,
        episode_ids=tuple(
            episode.episode_id for episode in projected_view.facts.episodes
        ),
        prior_idea_ids=tuple(
            prior.prior_idea_id for prior in projected_view.facts.prior_ideas
        ),
        existing_claim_revision_ids=(),
    )
    proposal_runner = ArtifactFakeRunner(
        tmp_path / "proposal-artifacts",
        valid_output(proposal_packet),
    )
    proposal_workspace = (tmp_path / "proposal-workspace").resolve()
    proposal_workspace.mkdir()
    proposal = ClaimProposer(fixture.settings.catalog, proposal_runner).propose(
        proposal_packet,
        proposal_workspace,
    )
    proposed = catalog.publish_claim_proposal(
        projected,
        proposal_packet,
        proposal,
    ).generation

    assert proposed.generation_number == 2
    proposed_view = catalog.read_generation(proposed)
    claim_state = next(
        state
        for state in proposed_view.entry_states
        if state.subject_payload_id == proposal.claims[0].revision_id
    )
    assert claim_state.admission_state is AdmissionState.QUARANTINED
    assert proposal.operation_receipt.operation_receipt_id in proposed.fact_object_ids

    next_packet = catalog.claim_proposal_packet(
        proposed,
        episode_ids=tuple(
            episode.episode_id for episode in proposed_view.facts.episodes
        ),
        prior_idea_ids=(),
        existing_claim_revision_ids=(proposal.claims[0].revision_id,),
    )
    assert tuple(
        state.catalog_entry_state_id for state in next_packet.entry_states
    ) == tuple(
        sorted(state.catalog_entry_state_id for state in next_packet.entry_states)
    )

    review_packet = catalog.claim_review_packet(
        proposed,
        proposal.claims[0].revision_id,
    )
    reviews = []
    for position, reviewer_slot in enumerate(fixture.settings.catalog.reviewers):
        runner = ArtifactFakeRunner(
            tmp_path / f"review-artifacts-{position}",
            reviewer_output(review_packet),
        )
        workspace = (tmp_path / f"review-workspace-{position}").resolve()
        workspace.mkdir()
        reviews.append(
            CatalogReviewer(fixture.settings.catalog, runner).review(
                review_packet,
                reviewer_slot,
                workspace,
            )
        )
    reviewed = catalog.publish_reviews(
        proposed,
        review_packet,
        tuple(reviews),
    ).generation

    reviewed_view = catalog.read_generation(reviewed)
    reviewed_claim_state = next(
        state
        for state in reviewed_view.entry_states
        if state.subject_payload_id == proposal.claims[0].revision_id
    )
    assert reviewed_claim_state.assertion_ids == tuple(
        sorted(review.assertion.assertion_id for review in reviews)
    )
    assert reviewed_claim_state.admission_state is AdmissionState.QUARANTINED
    assert reviewed_claim_state.predecessor_state_id == (
        claim_state.catalog_entry_state_id
    )
    assert reviewed.generation_number == 3
    assert tuple(proposal_workspace.iterdir()) == ()
    assert all(
        tuple((tmp_path / f"review-workspace-{position}").iterdir()) == ()
        for position in range(len(reviews))
    )

    rereview_packet = catalog.claim_review_packet(
        reviewed,
        proposal.claims[0].revision_id,
    )
    assert rereview_packet.previous_assertions == tuple(
        sorted(
            (review.assertion for review in reviews),
            key=lambda assertion: assertion.assertion_id,
        )
    )
    rereviews = []
    for position, reviewer_slot in enumerate(fixture.settings.catalog.reviewers):
        runner = ArtifactFakeRunner(
            tmp_path / f"rereview-artifacts-{position}",
            reviewer_output(
                rereview_packet,
                reviews[position].assertion.assertion_id,
            ),
        )
        workspace = (tmp_path / f"rereview-workspace-{position}").resolve()
        workspace.mkdir()
        rereviews.append(
            CatalogReviewer(fixture.settings.catalog, runner).review(
                rereview_packet,
                reviewer_slot,
                workspace,
            )
        )
    rereviewed = catalog.publish_reviews(
        reviewed,
        rereview_packet,
        tuple(rereviews),
    ).generation
    next_rereview_packet = catalog.claim_review_packet(
        rereviewed,
        proposal.claims[0].revision_id,
    )

    assert next_rereview_packet.previous_assertions == tuple(
        sorted(
            (review.assertion for review in rereviews),
            key=lambda assertion: assertion.assertion_id,
        )
    )


def test_projection_manifest_rejects_missing_derivation_before_pointer_advance(
    tmp_path: Path,
) -> None:
    fixture, projection = _project_real_bundle(tmp_path)
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        _scope_contract(),
        fixture.settings.catalog,
    )
    initial = catalog.store.read_current()
    missing_event_id = projection.derivation_objects[0].event_id
    incomplete_facts = tuple(
        record
        for record in projection.catalog_facts
        if getattr(record, record.IDENTITY_FIELD) != missing_event_id
    )

    with pytest.raises(CatalogFactError, match="derivation event is absent"):
        catalog.publish(
            expected_generation=initial,
            operation_id="incomplete_projection",
            objects=incomplete_facts,
            dependency_closure_ids=tuple(
                sorted(
                    getattr(record, record.IDENTITY_FIELD)
                    for record in incomplete_facts
                )
            ),
        )

    assert catalog.store.read_current() == initial


def test_fact_order_cannot_change_catalog_generation_identity(tmp_path: Path) -> None:
    fixture, projection = _project_real_bundle(tmp_path)
    first = CrossRunCatalog(
        tmp_path / "catalog-a",
        _scope_contract(),
        fixture.settings.catalog,
    )
    second = CrossRunCatalog(
        tmp_path / "catalog-b",
        _scope_contract(),
        fixture.settings.catalog,
    )
    operation_id = "ordered_projection"
    facts = projection.catalog_facts
    fact_ids = tuple(sorted(getattr(record, record.IDENTITY_FIELD) for record in facts))

    first_result = first.publish(
        expected_generation=first.store.read_current(),
        operation_id=operation_id,
        objects=facts,
        dependency_closure_ids=fact_ids,
    )
    second_result = second.publish(
        expected_generation=second.store.read_current(),
        operation_id=operation_id,
        objects=tuple(reversed(facts)),
        dependency_closure_ids=tuple(reversed(fact_ids)),
    )

    assert first_result.generation == second_result.generation
    assert first_result.delta_manifest == second_result.delta_manifest


def test_reducer_rejects_forged_nested_agent_packet_payload(tmp_path: Path) -> None:
    fixture, projection = _project_real_bundle(tmp_path)
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        _scope_contract(),
        fixture.settings.catalog,
    )
    projected = catalog.publish_projection(
        catalog.store.read_current(),
        projection,
    ).generation
    projected_view = catalog.read_generation(projected)
    packet = catalog.claim_proposal_packet(
        projected,
        episode_ids=tuple(
            episode.episode_id for episode in projected_view.facts.episodes
        ),
        prior_idea_ids=(),
        existing_claim_revision_ids=(),
    )
    workspace = (tmp_path / "proposal-workspace").resolve()
    workspace.mkdir()
    proposal = ClaimProposer(
        fixture.settings.catalog,
        ArtifactFakeRunner(
            tmp_path / "proposal-artifacts",
            valid_output(packet),
        ),
    ).propose(packet, workspace)
    preimage = dict(proposal.operation_record.operation_preimage)
    packet_payload = dict(preimage["packet"])
    source_episode = packet.episodes[0]
    foreign_bundle_id = fixture_id("forged-packet-bundle")
    foreign_sanitation_id = fixture_id("forged-packet-sanitation")
    foreign_derivation_id = fixture_id("forged-packet-derivation")
    foreign_episode = type(source_episode).mint(
        source={
            **dict(source_episode.source),
            "idea_id": "idea_forged_packet",
            "node_id": "node_forged_packet",
        },
        source_bundle_id=foreign_bundle_id,
        supersedes_projection_id=None,
        task_context_binding=source_episode.task_context_binding,
        artifact_environment=source_episode.artifact_environment,
        proposal="Internally valid evidence that is absent from the catalog.",
        parent_episode_ref=None,
        attempts=source_episode.attempts,
        terminal_attempt_revision=source_episode.terminal_attempt_revision,
        safe_observation_refs=source_episode.safe_observation_refs,
        sanitation_report_id=foreign_sanitation_id,
        derivation_refs=tuple(sorted((foreign_bundle_id, foreign_derivation_id))),
    )
    packet_payload["episodes"] = [foreign_episode.to_dict()]
    packet_payload["entry_states"] = []
    packet_payload["proof_reference_ids"] = sorted(
        (foreign_bundle_id, foreign_derivation_id, foreign_sanitation_id)
    )
    preimage["packet"] = packet_payload
    forged_operation = CatalogAgentOperationRecord.mint(
        operation_kind=proposal.operation_record.operation_kind,
        operation_receipt_id=proposal.operation_record.operation_receipt_id,
        operation_preimage=preimage,
        final_output=proposal.operation_record.final_output,
        produced_object_ids=proposal.operation_record.produced_object_ids,
    )
    objects = (
        *proposal.claims,
        *proposal.claim_evidence_closures,
        proposal.operation_receipt,
        forged_operation,
    )

    with pytest.raises(CatalogFactError, match="episode packet record bytes differ"):
        catalog.publish(
            expected_generation=projected,
            operation_id="forged_nested_packet",
            objects=objects,
            dependency_closure_ids=tuple(
                sorted(
                    {
                        *projected.fact_object_ids,
                        *(getattr(record, record.IDENTITY_FIELD) for record in objects),
                    }
                )
            ),
        )

    assert catalog.store.read_current() == projected


def test_historical_agent_authority_survives_model_effort_and_rubric_rotation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture, projection = _project_real_bundle(tmp_path)
    scope = _scope_contract()
    root = tmp_path / "catalog"
    catalog = CrossRunCatalog(root, scope, fixture.settings.catalog)
    projected = catalog.publish_projection(
        catalog.store.read_current(),
        projection,
    ).generation
    projected_view = catalog.read_generation(projected)
    packet = catalog.claim_proposal_packet(
        projected,
        episode_ids=tuple(
            episode.episode_id for episode in projected_view.facts.episodes
        ),
        prior_idea_ids=(),
        existing_claim_revision_ids=(),
    )
    proposal_workspace = (tmp_path / "proposal-workspace").resolve()
    proposal_workspace.mkdir()
    proposal = ClaimProposer(
        fixture.settings.catalog,
        ArtifactFakeRunner(
            tmp_path / "proposal-artifacts",
            valid_output(packet),
        ),
    ).propose(packet, proposal_workspace)
    proposed = catalog.publish_claim_proposal(
        projected,
        packet,
        proposal,
    ).generation
    original = fixture.settings.catalog
    original_review_packet = catalog.claim_review_packet(
        proposed,
        proposal.claims[0].revision_id,
    )
    original_reviews = []
    for position, reviewer in enumerate(original.reviewers):
        workspace = (tmp_path / f"original-review-workspace-{position}").resolve()
        workspace.mkdir()
        original_reviews.append(
            CatalogReviewer(
                original,
                ArtifactFakeRunner(
                    tmp_path / f"original-review-artifacts-{position}",
                    reviewer_output(original_review_packet),
                ),
            ).review(original_review_packet, reviewer, workspace)
        )
    originally_reviewed = catalog.publish_reviews(
        proposed,
        original_review_packet,
        tuple(original_reviews),
    ).generation
    original_claim_schema = ClaimProposer.response_schema
    original_review_schema = CatalogReviewer.response_schema_for
    monkeypatch.setattr(
        ClaimProposer,
        "operation_template",
        staticmethod(lambda: "Evolved claim template\nCATALOG_PACKET_JSON\n"),
    )
    monkeypatch.setattr(
        CatalogReviewer,
        "operation_template",
        staticmethod(lambda: "Evolved review template\nREVIEW_PACKET_JSON\n"),
    )
    monkeypatch.setattr(
        ClaimProposer,
        "response_schema",
        staticmethod(
            lambda: {
                **original_claim_schema(),
                "$comment": "evolved claim schema",
            }
        ),
    )
    monkeypatch.setattr(
        CatalogReviewer,
        "response_schema_for",
        staticmethod(
            lambda settings: {
                **original_review_schema(settings),
                "$comment": "evolved review schema",
            }
        ),
    )
    rotated = replace(
        original,
        claim_proposer=replace(
            original.claim_proposer,
            model="rotated-proposer-model",
            effort="xhigh",
        ),
        reviewers=tuple(
            replace(
                reviewer,
                agent=replace(
                    reviewer.agent,
                    model="rotated-reviewer-model",
                    effort="high",
                ),
            )
            for reviewer in original.reviewers
        ),
    )
    rotated_catalog = CrossRunCatalog(root, scope, rotated)
    review_packet = rotated_catalog.claim_review_packet(
        originally_reviewed,
        proposal.claims[0].revision_id,
    )
    reviews = []
    for position, reviewer in enumerate(rotated.reviewers):
        workspace = (tmp_path / f"rotated-review-workspace-{position}").resolve()
        workspace.mkdir()
        reviews.append(
            CatalogReviewer(
                rotated,
                ArtifactFakeRunner(
                    tmp_path / f"rotated-review-artifacts-{position}",
                    reviewer_output(
                        review_packet,
                        next(
                            assertion.assertion_id
                            for assertion in review_packet.previous_assertions
                            if assertion.reviewer_id == reviewer.reviewer_id
                        ),
                    ),
                ),
            ).review(review_packet, reviewer, workspace)
        )

    reviewed = rotated_catalog.publish_reviews(
        originally_reviewed,
        review_packet,
        tuple(reviews),
    ).generation
    state = next(
        value
        for value in rotated_catalog.read_generation(reviewed).entry_states
        if value.subject_payload_id == proposal.claims[0].revision_id
    )

    assert state.assertion_ids == tuple(
        sorted(
            review.assertion.assertion_id for review in (*original_reviews, *reviews)
        )
    )


def test_catalog_rejects_unknown_fact_namespaces(tmp_path: Path) -> None:
    fixture, _ = _project_real_bundle(tmp_path)
    scope = _scope_contract()
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        scope,
        fixture.settings.catalog,
    )
    initial = catalog.store.read_current()

    with pytest.raises(CatalogFactError, match="unknown fact namespace"):
        catalog.publish(
            expected_generation=initial,
            operation_id="unknown_catalog_fact",
            objects=(scope,),
            dependency_closure_ids=(scope.scope_contract_id,),
        )

    assert catalog.store.read_current() == initial


def test_claim_operation_cannot_consume_a_fact_from_its_own_publication_delta(
    tmp_path: Path,
) -> None:
    fixture, projection = _project_real_bundle(tmp_path)
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        _scope_contract(),
        fixture.settings.catalog,
    )
    projected = catalog.publish_projection(
        catalog.store.read_current(),
        projection,
    ).generation
    view = catalog.read_generation(projected)
    base_packet = catalog.claim_proposal_packet(
        projected,
        episode_ids=tuple(episode.episode_id for episode in view.facts.episodes),
        prior_idea_ids=(),
        existing_claim_revision_ids=(),
    )
    first_workspace = (tmp_path / "first-proposal-workspace").resolve()
    first_workspace.mkdir()
    first = ClaimProposer(
        fixture.settings.catalog,
        ArtifactFakeRunner(
            tmp_path / "first-proposal-artifacts",
            valid_output(base_packet),
        ),
    ).propose(base_packet, first_workspace)
    future_fact_packet = type(base_packet)(
        catalog_generation_id=base_packet.catalog_generation_id,
        catalog_generation=base_packet.catalog_generation,
        scope_contract=base_packet.scope_contract,
        episodes=base_packet.episodes,
        prior_ideas=(),
        existing_claims=first.claims,
        entry_states=base_packet.entry_states,
        review_assertions=(),
        operation_receipts=(first.operation_receipt,),
        proof_reference_ids=base_packet.proof_reference_ids,
    )
    second_workspace = (tmp_path / "second-proposal-workspace").resolve()
    second_workspace.mkdir()
    second = ClaimProposer(
        fixture.settings.catalog,
        ArtifactFakeRunner(
            tmp_path / "second-proposal-artifacts",
            valid_output(future_fact_packet),
        ),
    ).propose(future_fact_packet, second_workspace)
    objects = (
        *first.claims,
        *first.claim_evidence_closures,
        first.operation_receipt,
        first.operation_record,
        *second.claims,
        *second.claim_evidence_closures,
        second.operation_receipt,
        second.operation_record,
    )

    with pytest.raises(CatalogFactError, match="future catalog facts"):
        catalog.publish(
            expected_generation=projected,
            operation_id="same_delta_dependency",
            objects=objects,
            dependency_closure_ids=tuple(
                sorted(
                    {
                        *projected.fact_object_ids,
                        *(getattr(record, record.IDENTITY_FIELD) for record in objects),
                    }
                )
            ),
        )

    assert catalog.store.read_current() == projected
