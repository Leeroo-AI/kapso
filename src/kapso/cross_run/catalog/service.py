"""High-level catalog publication and complete-record packet assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kapso.cross_run.canonical import content_id
from kapso.cross_run.catalog.claims import ClaimProposalPacket, ClaimProposalResult
from kapso.cross_run.catalog.projector import ProjectionResult
from kapso.cross_run.catalog.reducer import CatalogFactSet, CatalogGenerationReducer
from kapso.cross_run.catalog.reviews import CatalogReviewPacket, CatalogReviewResult
from kapso.cross_run.catalog.store import (
    CatalogCommitResult,
    CatalogGenerationManifest,
    CatalogInputDelta,
    CatalogStore,
)
from kapso.cross_run.contracts import (
    CatalogEntryState,
    ExpertScopeContract,
    KnowledgeClaim,
    ReviewAssertion,
    StrictContract,
)
from kapso.cross_run.settings import CatalogSettings


class CrossRunCatalogError(ValueError):
    """A catalog publication or packet request violates its exact closure."""


def _record_id(record: StrictContract) -> str:
    identity_field = record.IDENTITY_FIELD
    if identity_field is None:
        raise CrossRunCatalogError("catalog source fact has no content identity")
    return getattr(record, identity_field)


def _record_envelope(record: StrictContract, kind: str) -> dict[str, object]:
    return {
        "record_id": _record_id(record),
        "record_kind": kind,
        "payload": record.to_dict(),
    }


@dataclass(frozen=True)
class CatalogGenerationView:
    """One exact generation decoded into source facts and derived entry states."""

    generation: CatalogGenerationManifest
    facts: CatalogFactSet
    entry_states: tuple[CatalogEntryState, ...]


class CrossRunCatalog:
    """Publish proof-closed facts and build agent packets from exact generations."""

    def __init__(
        self,
        root: Path | str,
        scope_contract: ExpertScopeContract,
        settings: CatalogSettings,
    ) -> None:
        self.scope_contract = scope_contract
        self.settings = settings
        self.store = CatalogStore(root)
        self.reducer = CatalogGenerationReducer(settings, scope_contract)
        self.store.initialize(
            scope_contract_id=scope_contract.scope_contract_id,
            configuration_fingerprint=settings.configuration_fingerprint,
        )

    def read_current(self) -> CatalogGenerationView:
        return self.read_generation(self.store.read_current())

    def read_generation(
        self,
        generation: CatalogGenerationManifest,
    ) -> CatalogGenerationView:
        if generation.scope_contract_id != self.scope_contract.scope_contract_id:
            raise CrossRunCatalogError("catalog generation uses another scope contract")
        facts = CatalogFactSet.read_ids(
            generation.fact_object_ids,
            self.store.read_object_bytes,
        )
        states = tuple(
            CatalogEntryState.from_json_bytes(self.store.read_object_bytes(state_id))
            for _, state_id in sorted(generation.active_entry_state_ids.items())
        )
        return CatalogGenerationView(
            generation=generation,
            facts=facts,
            entry_states=states,
        )

    def publish(
        self,
        *,
        expected_generation: CatalogGenerationManifest,
        operation_id: str,
        objects: tuple[StrictContract, ...],
        dependency_closure_ids: tuple[str, ...],
    ) -> CatalogCommitResult:
        ordered_objects = tuple(sorted(objects, key=_record_id))
        object_ids = tuple(_record_id(record) for record in ordered_objects)
        dependencies = tuple(sorted(set(dependency_closure_ids) | set(object_ids)))
        delta = CatalogInputDelta.mint(
            scope_contract_id=self.scope_contract.scope_contract_id,
            operation_id=operation_id,
            configuration_fingerprint=self.settings.configuration_fingerprint,
            added_object_ids=object_ids,
            dependency_closure_ids=dependencies,
        )
        return self.store.publish(
            expected_generation_id=expected_generation.catalog_generation_id,
            expected_generation_number=expected_generation.generation_number,
            input_delta=delta,
            objects=ordered_objects,
            reducer=self.reducer,
        )

    def rebase(
        self,
        *,
        operation_id: str,
        objects: tuple[StrictContract, ...],
        dependency_closure_ids: tuple[str, ...],
    ) -> CatalogCommitResult:
        ordered_objects = tuple(sorted(objects, key=_record_id))
        object_ids = tuple(_record_id(record) for record in ordered_objects)
        delta = CatalogInputDelta.mint(
            scope_contract_id=self.scope_contract.scope_contract_id,
            operation_id=operation_id,
            configuration_fingerprint=self.settings.configuration_fingerprint,
            added_object_ids=object_ids,
            dependency_closure_ids=tuple(
                sorted(set(dependency_closure_ids) | set(object_ids))
            ),
        )
        return self.store.rebase(
            input_delta=delta,
            objects=ordered_objects,
            reducer=self.reducer,
        )

    def publish_projection(
        self,
        expected_generation: CatalogGenerationManifest,
        projection: ProjectionResult,
    ) -> CatalogCommitResult:
        facts = projection.catalog_facts
        fact_ids = tuple(sorted(_record_id(record) for record in facts))
        operation_id = content_id(
            "catalog-projection-operation",
            {
                "projection_manifest_id": projection.projection_manifest.projection_manifest_id
            },
        )
        return self.publish(
            expected_generation=expected_generation,
            operation_id=operation_id,
            objects=facts,
            dependency_closure_ids=fact_ids,
        )

    def claim_proposal_packet(
        self,
        generation: CatalogGenerationManifest,
        *,
        episode_ids: tuple[str, ...],
        prior_idea_ids: tuple[str, ...],
        existing_claim_revision_ids: tuple[str, ...],
    ) -> ClaimProposalPacket:
        view = self.read_generation(generation)
        facts = view.facts
        episodes_by_id = {episode.episode_id: episode for episode in facts.episodes}
        priors_by_id = {prior.prior_idea_id: prior for prior in facts.prior_ideas}
        claims_by_id = {claim.revision_id: claim for claim in facts.claims}
        if (
            set(episode_ids) - set(episodes_by_id)
            or set(prior_idea_ids) - set(priors_by_id)
            or set(existing_claim_revision_ids) - set(claims_by_id)
        ):
            raise CrossRunCatalogError("claim packet selection leaves the generation")
        episodes = tuple(episodes_by_id[record_id] for record_id in sorted(episode_ids))
        prior_ideas = tuple(
            priors_by_id[record_id] for record_id in sorted(prior_idea_ids)
        )
        existing_claims = tuple(
            claims_by_id[record_id] for record_id in sorted(existing_claim_revision_ids)
        )
        subject_ids = {
            *episode_ids,
            *prior_idea_ids,
            *existing_claim_revision_ids,
        }
        assertions = tuple(
            assertion
            for assertion in facts.assertions
            if assertion.subject_id in subject_ids
        )
        receipt_ids = {
            *(assertion.review_operation_ref for assertion in assertions),
            *(
                claim.proposal_provenance["operation_receipt_id"]
                for claim in existing_claims
            ),
        }
        receipts = tuple(
            receipt
            for receipt in facts.operation_receipts
            if receipt.operation_receipt_id in receipt_ids
        )
        entry_states = tuple(
            sorted(
                (
                    state
                    for state in view.entry_states
                    if state.subject_payload_id in subject_ids
                ),
                key=lambda state: state.catalog_entry_state_id,
            )
        )
        proof_reference_ids = {
            reference
            for episode in episodes
            for reference in (
                episode.source_bundle_id,
                episode.sanitation_report_id,
                *episode.derivation_refs,
            )
        }
        proof_reference_ids.update(
            reference
            for prior in prior_ideas
            for reference in (prior.source_bundle_id, prior.sanitation_report_id)
        )
        proof_reference_ids.update(
            reference
            for assertion in assertions
            for reference in assertion.exact_evidence_refs
        )
        return ClaimProposalPacket(
            catalog_generation_id=generation.catalog_generation_id,
            catalog_generation=generation.generation_number,
            scope_contract=self.scope_contract,
            episodes=episodes,
            prior_ideas=prior_ideas,
            existing_claims=existing_claims,
            entry_states=entry_states,
            review_assertions=assertions,
            operation_receipts=receipts,
            proof_reference_ids=tuple(sorted(proof_reference_ids)),
        )

    def publish_claim_proposal(
        self,
        expected_generation: CatalogGenerationManifest,
        packet: ClaimProposalPacket,
        proposal: ClaimProposalResult,
    ) -> CatalogCommitResult:
        if (
            packet.catalog_generation_id != expected_generation.catalog_generation_id
            or packet.packet_digest != proposal.packet_digest
        ):
            raise CrossRunCatalogError(
                "claim proposal does not bind the expected catalog"
            )
        objects: tuple[StrictContract, ...] = (
            *proposal.claims,
            *proposal.claim_evidence_closures,
            proposal.operation_receipt,
            proposal.operation_record,
        )
        dependencies = {
            *packet.proof_reference_ids,
            *(episode.episode_id for episode in packet.episodes),
            *(prior.prior_idea_id for prior in packet.prior_ideas),
            *(claim.revision_id for claim in packet.existing_claims),
            *(assertion.assertion_id for assertion in packet.review_assertions),
            *(receipt.operation_receipt_id for receipt in packet.operation_receipts),
            *(_record_id(record) for record in objects),
        }
        return self.publish(
            expected_generation=expected_generation,
            operation_id=content_id(
                "catalog-claim-publication-operation",
                {
                    "operation_receipt_id": proposal.operation_receipt.operation_receipt_id
                },
            ),
            objects=objects,
            dependency_closure_ids=tuple(sorted(dependencies)),
        )

    def claim_review_packet(
        self,
        generation: CatalogGenerationManifest,
        claim_revision_id: str,
    ) -> CatalogReviewPacket:
        view = self.read_generation(generation)
        facts = view.facts
        claims = {claim.revision_id: claim for claim in facts.claims}
        claim = claims.get(claim_revision_id)
        if claim is None:
            raise CrossRunCatalogError("claim revision is absent from the generation")
        episodes = {episode.episode_id: episode for episode in facts.episodes}
        closures = {
            closure.claim_revision_id: closure
            for closure in facts.claim_evidence_closures
        }
        closure = closures.get(claim_revision_id)
        if closure is None:
            raise CrossRunCatalogError("claim evidence closure is absent")
        evidence_ids = closure.evaluated_episode_ids
        if any(evidence_id not in episodes for evidence_id in evidence_ids):
            raise CrossRunCatalogError("claim evidence is absent from the generation")
        receipts = {
            receipt.operation_receipt_id: receipt
            for receipt in facts.operation_receipts
        }
        proposer_receipt_id = claim.proposal_provenance.get("operation_receipt_id")
        proposer_receipt = receipts.get(proposer_receipt_id)
        if proposer_receipt is None:
            raise CrossRunCatalogError("claim proposer receipt is absent")
        subject_assertions = tuple(
            assertion
            for assertion in facts.assertions
            if assertion.subject_id == claim_revision_id
        )
        previous_assertions = self._active_review_heads(subject_assertions)
        previous_receipt_ids = {
            assertion.review_operation_ref for assertion in previous_assertions
        }
        if any(receipt_id not in receipts for receipt_id in previous_receipt_ids):
            raise CrossRunCatalogError("previous review receipt is absent")
        return CatalogReviewPacket(
            catalog_generation_id=generation.catalog_generation_id,
            catalog_generation=generation.generation_number,
            scope_contract=self.scope_contract,
            subject=_record_envelope(claim, "knowledge_claim"),
            evidence_records=tuple(
                _record_envelope(episodes[evidence_id], "transfer_episode")
                for evidence_id in evidence_ids
            ),
            proposer_operation_receipt=proposer_receipt,
            claim_evidence_closure=closure,
            previous_assertions=previous_assertions,
            previous_operation_receipts=tuple(
                receipts[receipt_id] for receipt_id in sorted(previous_receipt_ids)
            ),
        )

    def _active_review_heads(
        self,
        assertions: tuple[ReviewAssertion, ...],
    ) -> tuple[ReviewAssertion, ...]:
        heads: list[ReviewAssertion] = []
        for reviewer in self.settings.reviewers:
            reviewer_assertions = tuple(
                assertion
                for assertion in assertions
                if assertion.reviewer_id == reviewer.reviewer_id
            )
            superseded_ids = {
                assertion.supersedes_assertion_id
                for assertion in reviewer_assertions
                if assertion.supersedes_assertion_id is not None
            }
            reviewer_heads = tuple(
                assertion
                for assertion in reviewer_assertions
                if assertion.assertion_id not in superseded_ids
            )
            if len(reviewer_heads) > 1:
                raise CrossRunCatalogError(
                    "review assertion history has multiple active heads"
                )
            heads.extend(reviewer_heads)
        return tuple(sorted(heads, key=lambda assertion: assertion.assertion_id))

    def publish_reviews(
        self,
        expected_generation: CatalogGenerationManifest,
        packet: CatalogReviewPacket,
        reviews: tuple[CatalogReviewResult, ...],
    ) -> CatalogCommitResult:
        if packet.catalog_generation_id != expected_generation.catalog_generation_id:
            raise CrossRunCatalogError("reviews do not bind the expected catalog")
        if not reviews:
            raise CrossRunCatalogError("review publication must not be empty")
        objects: tuple[StrictContract, ...] = (
            *(review.assertion for review in reviews),
            *(review.operation_receipt for review in reviews),
            *(review.operation_record for review in reviews),
        )
        dependencies = {
            packet.subject_id,
            *packet.evidence_record_ids,
            *(_record_id(record) for record in objects),
        }
        if packet.proposer_operation_receipt is not None:
            dependencies.add(packet.proposer_operation_receipt.operation_receipt_id)
        if packet.claim_evidence_closure is not None:
            dependencies.add(packet.claim_evidence_closure.claim_evidence_closure_id)
        dependencies.update(
            assertion.assertion_id for assertion in packet.previous_assertions
        )
        dependencies.update(
            receipt.operation_receipt_id
            for receipt in packet.previous_operation_receipts
        )
        return self.publish(
            expected_generation=expected_generation,
            operation_id=content_id(
                "catalog-review-publication-operation",
                {
                    "operation_receipt_ids": tuple(
                        sorted(
                            review.operation_receipt.operation_receipt_id
                            for review in reviews
                        )
                    )
                },
            ),
            objects=objects,
            dependency_closure_ids=tuple(sorted(dependencies)),
        )
