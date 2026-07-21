"""Deterministic catalog admission, revocation, and proof-taint reduction."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.agent_artifacts import CodingAgentWorkspaceAccess
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.assertions import (
    AssertionAdjudication,
    ReviewRegistry,
)
from kapso.cross_run.contracts import (
    AdmissionState,
    CatalogEntryState,
    CodingAgentOperationReceipt,
    ComparisonStatus,
    ContractValidationError,
    ExpertScopeContract,
    IdentityConflictError,
    InterventionStructure,
    KnowledgeClaim,
    MissingReferenceError,
    PriorIdea,
    ReviewAssertion,
    TransferEpisode,
)
from kapso.cross_run.record_contracts import (
    CatalogAgentOperationRecord,
    CatalogRevocation,
    CatalogTaint,
    ClaimEvidenceClosure,
    SanitationReport,
)
from kapso.cross_run.settings import CatalogSettings


class AdmissionReductionError(ValueError):
    """Catalog facts cannot be reduced into a valid generation."""


@dataclass(frozen=True)
class AdmissionReduction:
    """Complete deterministic state projection for one catalog generation."""

    states: tuple[CatalogEntryState, ...]
    assertion_views: Mapping[str, AssertionAdjudication]
    proof_dependencies: Mapping[str, tuple[str, ...]]


class AdmissionReducer:
    """Reduce immutable facts without score, recency, or similarity heuristics."""

    def __init__(
        self,
        settings: CatalogSettings,
        scope_contract: ExpertScopeContract,
    ) -> None:
        self._settings = settings
        self._scope_contract = scope_contract
        self._configuration_fingerprint = settings.configuration_fingerprint
        self._registry = ReviewRegistry(
            settings.reviewers,
            approval_judgment=settings.admission.approval_judgment,
            rejection_judgment=settings.admission.rejection_judgment,
            required_approvals=settings.admission.required_approvals,
            required_rejections=settings.admission.required_rejections,
            prohibited_principal_ids=(settings.claim_proposer_id,),
        )

    def reduce(
        self,
        *,
        catalog_generation: int,
        episodes: tuple[TransferEpisode, ...],
        prior_ideas: tuple[PriorIdea, ...],
        claims: tuple[KnowledgeClaim, ...],
        assertions: tuple[ReviewAssertion, ...],
        receipts: tuple[CodingAgentOperationReceipt, ...],
        operation_records: tuple[CatalogAgentOperationRecord, ...],
        claim_evidence_closures: tuple[ClaimEvidenceClosure, ...],
        sanitation_reports: tuple[SanitationReport, ...],
        proof_object_ids: tuple[str, ...],
        revocations: tuple[CatalogRevocation, ...] = (),
        taints: tuple[CatalogTaint, ...] = (),
        derivation_edges: Mapping[str, tuple[str, ...]] = MappingProxyType({}),
        predecessor_states: tuple[CatalogEntryState, ...] = (),
    ) -> AdmissionReduction:
        """Compute all active entry states from the complete grow-only fact set."""

        if catalog_generation <= 0:
            raise AdmissionReductionError("catalog generation must be positive")
        payloads = self._payload_map(episodes, prior_ideas, claims)
        proof_ids = self._proof_closure_ids(
            payloads,
            assertions,
            receipts,
            operation_records,
            claim_evidence_closures,
            sanitation_reports,
            revocations,
            taints,
            proof_object_ids,
        )
        admitted_sanitation = self._validate_sanitation_reports(
            sanitation_reports,
            episodes,
            prior_ideas,
        )
        assertion_views = self._registry.adjudicate(
            assertions=assertions,
            receipts=receipts,
            operation_records=tuple(
                operation
                for operation in operation_records
                if operation.operation_kind == "catalog_review"
            ),
            known_object_ids=tuple(sorted(proof_ids)),
        )
        self._validate_claim_operations(
            claims,
            claim_evidence_closures,
            receipts,
            operation_records,
            episodes,
        )
        dependencies = self._build_dependencies(
            episodes,
            prior_ideas,
            claims,
            assertions,
            operation_records,
            claim_evidence_closures,
            derivation_edges,
            proof_ids,
        )
        successor_ids = self._build_successor_index(
            episodes,
            prior_ideas,
            claims,
            payloads,
        )
        revocations_by_subject, taint_roots = self._taint_roots(
            revocations,
            taints,
            proof_ids,
        )
        propagated_taint = self._propagate_taint(dependencies, taint_roots)
        predecessors = self._predecessors(
            predecessor_states,
            payloads,
            catalog_generation,
        )
        evidence_closures = self._validate_claim_evidence_closures(
            claim_evidence_closures,
            claims,
            episodes,
            receipts,
        )
        self._validate_claim_review_evidence(
            assertions,
            evidence_closures,
            receipts,
        )
        projection_states: dict[str, AdmissionState] = {}
        for episode in episodes:
            base_state = self._projection_base_state(
                episode.sanitation_report_id,
                admitted_sanitation,
                assertion_views.get(episode.episode_id),
            )
            projection_states[episode.episode_id] = self._precedence_state(
                base_state,
                revocations_by_subject.get(episode.episode_id, ()),
                propagated_taint.get(episode.episode_id, ()),
                successor_ids.get(episode.episode_id, ()),
                assertion_views.get(episode.episode_id),
            )
        for prior_idea in prior_ideas:
            base_state = self._projection_base_state(
                prior_idea.sanitation_report_id,
                admitted_sanitation,
                assertion_views.get(prior_idea.prior_idea_id),
            )
            projection_states[prior_idea.prior_idea_id] = self._precedence_state(
                base_state,
                revocations_by_subject.get(prior_idea.prior_idea_id, ()),
                propagated_taint.get(prior_idea.prior_idea_id, ()),
                successor_ids.get(prior_idea.prior_idea_id, ()),
                assertion_views.get(prior_idea.prior_idea_id),
            )
        states: list[CatalogEntryState] = []
        for payload_id in sorted(payloads):
            payload = payloads[payload_id]
            assertion_view = assertion_views.get(payload_id)
            if isinstance(payload, KnowledgeClaim):
                base_state = self._claim_base_state(
                    payload,
                    episodes,
                    projection_states,
                    evidence_closures[payload.revision_id],
                    assertion_view,
                )
            else:
                base_state = projection_states[payload_id]
            direct_revocations = revocations_by_subject.get(payload_id, ())
            taint_sources = propagated_taint.get(payload_id, ())
            taint_sources = tuple(
                source_id
                for source_id in taint_sources
                if source_id not in direct_revocations
            )
            successors = successor_ids.get(payload_id, ())
            admission_state = self._precedence_state(
                base_state,
                direct_revocations,
                taint_sources,
                successors,
                assertion_view,
            )
            states.append(
                CatalogEntryState.mint(
                    subject_payload_id=payload_id,
                    catalog_generation=catalog_generation,
                    predecessor_state_id=(
                        predecessors[payload_id].catalog_entry_state_id
                        if payload_id in predecessors
                        else None
                    ),
                    configuration_fingerprint=self._configuration_fingerprint,
                    admission_state=admission_state,
                    superseded_by_payload_ids=successors,
                    assertion_ids=(
                        assertion_view.assertion_ids
                        if assertion_view is not None
                        else ()
                    ),
                    revocation_ids=direct_revocations,
                    taint_source_ids=taint_sources,
                )
            )
        return AdmissionReduction(
            states=tuple(states),
            assertion_views=assertion_views,
            proof_dependencies=MappingProxyType(dependencies),
        )

    @staticmethod
    def _payload_map(
        episodes: tuple[TransferEpisode, ...],
        prior_ideas: tuple[PriorIdea, ...],
        claims: tuple[KnowledgeClaim, ...],
    ) -> dict[str, TransferEpisode | PriorIdea | KnowledgeClaim]:
        payloads: dict[str, TransferEpisode | PriorIdea | KnowledgeClaim] = {}
        for payload in (*episodes, *prior_ideas, *claims):
            if isinstance(payload, TransferEpisode):
                payload_id = payload.episode_id
            elif isinstance(payload, PriorIdea):
                payload_id = payload.prior_idea_id
            else:
                payload_id = payload.revision_id
            if payload_id in payloads:
                raise IdentityConflictError("catalog payload IDs must be unique")
            payloads[payload_id] = payload
        return payloads

    @staticmethod
    def _proof_closure_ids(
        payloads: Mapping[str, TransferEpisode | PriorIdea | KnowledgeClaim],
        assertions: tuple[ReviewAssertion, ...],
        receipts: tuple[CodingAgentOperationReceipt, ...],
        operation_records: tuple[CatalogAgentOperationRecord, ...],
        claim_evidence_closures: tuple[ClaimEvidenceClosure, ...],
        sanitation_reports: tuple[SanitationReport, ...],
        revocations: tuple[CatalogRevocation, ...],
        taints: tuple[CatalogTaint, ...],
        proof_object_ids: tuple[str, ...],
    ) -> set[str]:
        identity_groups = (
            proof_object_ids,
            tuple(payloads),
            tuple(assertion.assertion_id for assertion in assertions),
            tuple(receipt.operation_receipt_id for receipt in receipts),
            tuple(operation.operation_record_id for operation in operation_records),
            tuple(
                closure.claim_evidence_closure_id for closure in claim_evidence_closures
            ),
            tuple(report.report_id for report in sanitation_reports),
            tuple(revocation.revocation_id for revocation in revocations),
            tuple(taint.taint_id for taint in taints),
        )
        if any(len(group) != len(set(group)) for group in identity_groups):
            raise IdentityConflictError(
                "each catalog proof object collection must contain unique IDs"
            )
        proof_ids = set().union(*identity_groups)
        for object_id in proof_ids:
            require_content_id(object_id, "catalog proof object ID")
        for revocation in revocations:
            if revocation.subject_id not in proof_ids:
                raise MissingReferenceError("revocation subject is absent")
            missing = set(revocation.exact_evidence_refs) - proof_ids
            if missing:
                raise MissingReferenceError("revocation evidence closure is incomplete")
        for taint in taints:
            if (
                taint.subject_id not in proof_ids
                or taint.source_subject_id not in proof_ids
            ):
                raise MissingReferenceError("taint subject or source is absent")
            missing = set(taint.exact_evidence_refs) - proof_ids
            if missing:
                raise MissingReferenceError("taint evidence closure is incomplete")
        return proof_ids

    def _validate_claim_operations(
        self,
        claims: tuple[KnowledgeClaim, ...],
        closures: tuple[ClaimEvidenceClosure, ...],
        receipts: tuple[CodingAgentOperationReceipt, ...],
        operations: tuple[CatalogAgentOperationRecord, ...],
        episodes: tuple[TransferEpisode, ...],
    ) -> None:
        receipt_by_id = {receipt.operation_receipt_id: receipt for receipt in receipts}
        operation_receipt_ids = tuple(
            operation.operation_receipt_id for operation in operations
        )
        if len(operation_receipt_ids) != len(set(operation_receipt_ids)):
            raise IdentityConflictError(
                "one coding-agent receipt has multiple operation records"
            )
        if set(operation_receipt_ids) != set(receipt_by_id):
            raise MissingReferenceError(
                "every coding-agent receipt requires one operation record"
            )
        claim_operations = tuple(
            operation
            for operation in operations
            if operation.operation_kind == "claim_proposal"
        )
        claim_by_id = {claim.revision_id: claim for claim in claims}
        closure_by_claim = {closure.claim_revision_id: closure for closure in closures}
        episode_by_id = {episode.episode_id: episode for episode in episodes}
        produced_claim_ids: set[str] = set()
        for operation in claim_operations:
            receipt = receipt_by_id[operation.operation_receipt_id]
            operation.validate_receipt(receipt)
            self._validate_proposer_receipt(receipt, operation)
            normalized_proposals = self._normalized_claim_output(
                operation.final_output,
                operation.packet_payload,
            )
            operation_claims = tuple(
                sorted(
                    (
                        claim
                        for claim in claims
                        if claim.proposal_provenance.get("operation_receipt_id")
                        == receipt.operation_receipt_id
                    ),
                    key=lambda claim: claim.proposal_provenance["proposal_ordinal"],
                )
            )
            if len(operation_claims) != len(normalized_proposals):
                raise MissingReferenceError(
                    "claim operation output count differs from catalog claims"
                )
            packet_digest = tree_or_blob_digest(
                canonical_json_bytes(operation.packet_payload)
            )
            packet_episode_ids = (
                tuple(
                    sorted(
                        item["episode_id"]
                        for item in normalized_proposals[0]["assessments"]
                    )
                )
                if normalized_proposals
                else tuple(
                    sorted(
                        episode["episode_id"]
                        for episode in operation.packet_payload.get("episodes", ())
                    )
                )
            )
            produced_ids: set[str] = set()
            for ordinal, (claim, proposal) in enumerate(
                zip(operation_claims, normalized_proposals)
            ):
                self._validate_claim_from_output(
                    claim,
                    closure_by_claim.get(claim.revision_id),
                    proposal,
                    ordinal,
                    receipt,
                    packet_digest,
                    packet_episode_ids,
                    claim_by_id,
                    episode_by_id,
                )
                produced_claim_ids.add(claim.revision_id)
                produced_ids.add(claim.revision_id)
                produced_ids.add(
                    closure_by_claim[claim.revision_id].claim_evidence_closure_id
                )
            if tuple(sorted(produced_ids)) != operation.produced_object_ids:
                raise IdentityConflictError(
                    "claim operation produced-object closure differs"
                )
        if produced_claim_ids != set(claim_by_id):
            raise MissingReferenceError(
                "every claim requires one authenticated proposal operation"
            )

    def _validate_proposer_receipt(
        self,
        receipt: CodingAgentOperationReceipt,
        operation: CatalogAgentOperationRecord,
    ) -> None:
        configuration = operation.operation_preimage.get("catalog_configuration")
        if not isinstance(configuration, Mapping):
            raise ContractValidationError(
                "claim operation catalog configuration is absent"
            )
        historical_settings = CatalogSettings.from_dict(configuration)
        proposer = historical_settings.claim_proposer
        if (
            receipt.principal_id != historical_settings.claim_proposer_id
            or receipt.role != historical_settings.claim_proposer_role
            or receipt.cli != proposer.cli
            or receipt.model != proposer.model
            or receipt.effort != proposer.effort
            or receipt.workspace_access is not CodingAgentWorkspaceAccess.READ_ONLY
        ):
            raise IdentityConflictError(
                "claim operation uses a forged proposer receipt"
            )

    def _normalized_claim_output(
        self,
        output_text: str,
        packet: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        output = parse_json_bytes(output_text.encode("utf-8"))
        if not isinstance(output, Mapping) or set(output) != {"claims"}:
            raise ContractValidationError("claim operation output fields are invalid")
        raw_claims = output["claims"]
        if not isinstance(raw_claims, list):
            raise ContractValidationError("claim operation claims must be an array")
        scope_payload = packet.get("scope_contract")
        if canonical_json_bytes(scope_payload) != canonical_json_bytes(
            self._scope_contract.to_dict()
        ):
            raise IdentityConflictError("claim operation uses another scope contract")
        packet_episodes = packet.get("episodes")
        if not isinstance(packet_episodes, (list, tuple)):
            raise ContractValidationError("claim operation packet episodes are invalid")
        expected_episode_ids = {
            episode["episode_id"]
            for episode in packet_episodes
            if isinstance(episode, Mapping) and "episode_id" in episode
        }
        if len(expected_episode_ids) != len(packet_episodes):
            raise ContractValidationError("claim operation packet episodes are invalid")
        normalized = tuple(
            self._normalize_raw_claim(raw, expected_episode_ids) for raw in raw_claims
        )
        keys = tuple(
            canonical_json_bytes(
                {key: value for key, value in proposal.items() if key != "assessments"}
            )
            for proposal in normalized
        )
        if len(keys) != len(set(keys)):
            raise ContractValidationError("claim operation returned duplicate claims")
        return tuple(
            proposal
            for _, proposal in sorted(zip(keys, normalized), key=lambda item: item[0])
        )

    def _normalize_raw_claim(
        self,
        raw: Any,
        expected_episode_ids: set[str],
    ) -> Mapping[str, Any]:
        fields = {
            "applicability_predicates",
            "evidence_assessments",
            "explicit_exclusions",
            "mechanism",
            "statement",
            "supersedes_revision_ids",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ContractValidationError("claim operation claim fields are invalid")
        for field in ("statement", "mechanism"):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise ContractValidationError(f"claim operation {field} is empty")
        exclusions = raw["explicit_exclusions"]
        supersedes = raw["supersedes_revision_ids"]
        if (
            not isinstance(exclusions, list)
            or not exclusions
            or any(
                not isinstance(value, str) or not value.strip() for value in exclusions
            )
            or len(exclusions) != len(set(exclusions))
            or not isinstance(supersedes, list)
            or any(not isinstance(value, str) for value in supersedes)
            or tuple(supersedes) != tuple(sorted(set(supersedes)))
        ):
            raise ContractValidationError("claim operation arrays are invalid")
        predicates = raw["applicability_predicates"]
        if not isinstance(predicates, Mapping) or not predicates:
            raise ContractValidationError("claim operation predicates are invalid")
        schemas = {
            schema.dimension_id: schema
            for schema in self._scope_contract.context_dimension_schemas
        }
        if not set(predicates).issubset(schemas):
            raise ContractValidationError("claim operation predicate is unregistered")
        for dimension_id, value in predicates.items():
            schemas[dimension_id].validate_value(value)
        assessments = raw["evidence_assessments"]
        if not isinstance(assessments, list):
            raise ContractValidationError("claim evidence assessments are invalid")
        normalized_assessments: list[Mapping[str, str]] = []
        assessment_ids: set[str] = set()
        for assessment in assessments:
            if not isinstance(assessment, Mapping) or set(assessment) != {
                "episode_id",
                "rationale",
                "relationship",
            }:
                raise ContractValidationError("claim evidence assessment is invalid")
            if any(
                not isinstance(value, str) or not value.strip()
                for value in assessment.values()
            ) or assessment["relationship"] not in {
                "support",
                "contradiction",
                "not_applicable",
            }:
                raise ContractValidationError("claim evidence assessment is invalid")
            if assessment["episode_id"] in assessment_ids:
                raise IdentityConflictError("claim episode was assessed more than once")
            assessment_ids.add(assessment["episode_id"])
            normalized_assessments.append(dict(assessment))
        if assessment_ids != expected_episode_ids:
            raise MissingReferenceError(
                "claim operation did not assess every packet episode"
            )
        normalized_assessments.sort(key=lambda item: item["episode_id"])
        supporting = tuple(
            item["episode_id"]
            for item in normalized_assessments
            if item["relationship"] == "support"
        )
        contradicting = tuple(
            item["episode_id"]
            for item in normalized_assessments
            if item["relationship"] == "contradiction"
        )
        if not supporting and not contradicting:
            raise ContractValidationError("claim operation has no classified evidence")
        return {
            "statement": raw["statement"],
            "mechanism": raw["mechanism"],
            "applicability_predicates": dict(sorted(predicates.items())),
            "explicit_exclusions": tuple(sorted(exclusions)),
            "supporting_episode_ids": supporting,
            "contradicting_episode_ids": contradicting,
            "supersedes_revision_ids": tuple(supersedes),
            "assessments": tuple(normalized_assessments),
        }

    def _validate_claim_from_output(
        self,
        claim: KnowledgeClaim,
        closure: ClaimEvidenceClosure | None,
        proposal: Mapping[str, Any],
        ordinal: int,
        receipt: CodingAgentOperationReceipt,
        packet_digest: str,
        packet_episode_ids: tuple[str, ...],
        claims_by_id: Mapping[str, KnowledgeClaim],
        episodes_by_id: Mapping[str, TransferEpisode],
    ) -> None:
        if closure is None:
            raise MissingReferenceError("claim evidence closure is absent")
        expected_fields = {
            "statement": claim.statement,
            "mechanism": claim.mechanism,
            "applicability_predicates": dict(claim.applicability_predicates),
            "explicit_exclusions": claim.explicit_exclusions,
            "supporting_episode_ids": claim.supporting_episode_ids,
            "contradicting_episode_ids": claim.contradicting_episode_ids,
            "supersedes_revision_ids": claim.supersedes_revision_ids,
        }
        if any(
            canonical_json_bytes(proposal[field]) != canonical_json_bytes(value)
            for field, value in expected_fields.items()
        ):
            raise IdentityConflictError(
                "catalog claim differs from authenticated model output"
            )
        if claim.proposal_provenance != {
            "operation_receipt_id": receipt.operation_receipt_id,
            "packet_digest": packet_digest,
            "proposal_ordinal": ordinal,
        }:
            raise IdentityConflictError("claim proposal provenance differs")
        if closure.packet_digest != packet_digest or (
            closure.proposer_operation_receipt_id != receipt.operation_receipt_id
        ):
            raise IdentityConflictError("claim evidence operation provenance differs")
        if closure.evaluated_episode_ids != packet_episode_ids or (
            closure.evidence_assessments != proposal["assessments"]
        ):
            raise IdentityConflictError("claim evidence assessments differ")
        if any(episode_id not in episodes_by_id for episode_id in packet_episode_ids):
            raise MissingReferenceError("claim packet episode is absent from catalog")
        if claim.supersedes_revision_ids:
            lineage_ids = {
                claims_by_id[revision_id].claim_id
                for revision_id in claim.supersedes_revision_ids
                if revision_id in claims_by_id
            }
            if len(lineage_ids) != 1 or claim.claim_id not in lineage_ids:
                raise IdentityConflictError("claim supersession lineage differs")
        else:
            expected_claim_id = (
                "claim_"
                + content_id(
                    "claim-lineage",
                    {
                        "operation_receipt_id": receipt.operation_receipt_id,
                        "proposal_ordinal": ordinal,
                    },
                ).rsplit(":", 1)[1][:32]
            )
            if claim.claim_id != expected_claim_id:
                raise IdentityConflictError("claim lineage identity is forged")

    @staticmethod
    def _validate_sanitation_reports(
        reports: tuple[SanitationReport, ...],
        episodes: tuple[TransferEpisode, ...],
        prior_ideas: tuple[PriorIdea, ...],
    ) -> set[str]:
        reports_by_id = {report.report_id: report for report in reports}
        if len(reports_by_id) != len(reports):
            raise IdentityConflictError("sanitation report IDs must be unique")
        for projection in (*episodes, *prior_ideas):
            report = reports_by_id.get(projection.sanitation_report_id)
            if report is None:
                raise MissingReferenceError(
                    "projection sanitation report fact is absent"
                )
            context = projection.task_context_binding
            if (
                report.scope_id != context.scope_id
                or report.task_family_id != context.task_family_id
            ):
                raise IdentityConflictError(
                    "projection sanitation report uses another task context"
                )
        return {report.report_id for report in reports if report.status == "admitted"}

    @staticmethod
    def _build_dependencies(
        episodes: tuple[TransferEpisode, ...],
        prior_ideas: tuple[PriorIdea, ...],
        claims: tuple[KnowledgeClaim, ...],
        assertions: tuple[ReviewAssertion, ...],
        operation_records: tuple[CatalogAgentOperationRecord, ...],
        claim_evidence_closures: tuple[ClaimEvidenceClosure, ...],
        derivation_edges: Mapping[str, tuple[str, ...]],
        proof_ids: set[str],
    ) -> dict[str, tuple[str, ...]]:
        dependencies: dict[str, set[str]] = {}
        for subject_id, refs in derivation_edges.items():
            require_content_id(subject_id, "derivation subject ID")
            if subject_id not in proof_ids:
                raise MissingReferenceError("derivation subject is absent")
            if len(refs) != len(set(refs)):
                raise IdentityConflictError("derivation refs must be unique")
            if set(refs) - proof_ids:
                raise MissingReferenceError("derivation dependency is absent")
            dependencies.setdefault(subject_id, set()).update(refs)
        for episode in episodes:
            dependencies.setdefault(episode.episode_id, set()).update(
                (
                    episode.source_bundle_id,
                    episode.sanitation_report_id,
                    *episode.derivation_refs,
                )
            )
        for prior_idea in prior_ideas:
            dependencies.setdefault(prior_idea.prior_idea_id, set()).update(
                (prior_idea.source_bundle_id, prior_idea.sanitation_report_id)
            )
        for claim in claims:
            dependencies.setdefault(claim.revision_id, set()).update(
                (*claim.supporting_episode_ids, *claim.contradicting_episode_ids)
            )
        for assertion in assertions:
            dependencies.setdefault(assertion.subject_id, set()).add(
                assertion.assertion_id
            )
            dependencies.setdefault(assertion.assertion_id, set()).update(
                (assertion.review_operation_ref, *assertion.exact_evidence_refs)
            )
        for operation in operation_records:
            dependencies.setdefault(operation.operation_record_id, set()).add(
                operation.operation_receipt_id
            )
            for produced_id in operation.produced_object_ids:
                dependencies.setdefault(produced_id, set()).add(
                    operation.operation_record_id
                )
        for closure in claim_evidence_closures:
            dependencies.setdefault(closure.claim_revision_id, set()).add(
                closure.claim_evidence_closure_id
            )
            dependencies.setdefault(
                closure.claim_evidence_closure_id,
                set(),
            ).update(
                (
                    closure.proposer_operation_receipt_id,
                    *closure.evaluated_episode_ids,
                )
            )
        for subject_id, refs in dependencies.items():
            missing = refs - proof_ids
            if missing:
                raise MissingReferenceError(
                    f"proof dependency closure is incomplete for {subject_id}"
                )
        return {
            subject_id: tuple(sorted(refs))
            for subject_id, refs in sorted(dependencies.items())
        }

    @staticmethod
    def _build_successor_index(
        episodes: tuple[TransferEpisode, ...],
        prior_ideas: tuple[PriorIdea, ...],
        claims: tuple[KnowledgeClaim, ...],
        payloads: Mapping[str, TransferEpisode | PriorIdea | KnowledgeClaim],
    ) -> dict[str, tuple[str, ...]]:
        successors: dict[str, set[str]] = {}
        projections = {
            **{episode.episode_id: episode for episode in episodes},
            **{prior.prior_idea_id: prior for prior in prior_ideas},
        }
        for payload_id, projection in projections.items():
            predecessor_id = projection.supersedes_projection_id
            if predecessor_id is None:
                continue
            predecessor = projections.get(predecessor_id)
            if predecessor is None:
                raise MissingReferenceError("superseded projection is absent")
            current_source = dict(projection.source)
            predecessor_source = dict(predecessor.source)
            current_source.pop("node_id", None)
            predecessor_source.pop("node_id", None)
            if current_source != predecessor_source:
                raise IdentityConflictError(
                    "projection supersession crosses source idea identity"
                )
            successors.setdefault(predecessor_id, set()).add(payload_id)
        claims_by_id = {claim.revision_id: claim for claim in claims}
        for claim in claims:
            for predecessor_id in claim.supersedes_revision_ids:
                predecessor = claims_by_id.get(predecessor_id)
                if predecessor is None:
                    raise MissingReferenceError("superseded claim revision is absent")
                if predecessor.claim_id != claim.claim_id:
                    raise IdentityConflictError(
                        "claim revision supersession crosses claim identity"
                    )
                successors.setdefault(predecessor_id, set()).add(claim.revision_id)
        if set(successors) - set(payloads):
            raise MissingReferenceError("supersession predecessor is absent")
        return {
            subject_id: tuple(sorted(subject_successors))
            for subject_id, subject_successors in sorted(successors.items())
        }

    @staticmethod
    def _taint_roots(
        revocations: tuple[CatalogRevocation, ...],
        taints: tuple[CatalogTaint, ...],
        proof_ids: set[str],
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, set[str]]]:
        revocation_ids = tuple(revocation.revocation_id for revocation in revocations)
        taint_ids = tuple(taint.taint_id for taint in taints)
        if len(revocation_ids) != len(set(revocation_ids)):
            raise IdentityConflictError("revocation IDs must be unique")
        if len(taint_ids) != len(set(taint_ids)):
            raise IdentityConflictError("taint IDs must be unique")
        direct_revocations: dict[str, set[str]] = {}
        roots: dict[str, set[str]] = {object_id: set() for object_id in proof_ids}
        for revocation in revocations:
            direct_revocations.setdefault(revocation.subject_id, set()).add(
                revocation.revocation_id
            )
            roots[revocation.subject_id].add(revocation.revocation_id)
        for taint in taints:
            roots[taint.subject_id].add(taint.taint_id)
        return (
            {
                subject_id: tuple(sorted(ids))
                for subject_id, ids in sorted(direct_revocations.items())
            },
            roots,
        )

    @staticmethod
    def _propagate_taint(
        dependencies: Mapping[str, tuple[str, ...]],
        roots: Mapping[str, set[str]],
    ) -> dict[str, tuple[str, ...]]:
        taint = {
            subject_id: set(source_ids) for subject_id, source_ids in roots.items()
        }
        changed = True
        while changed:
            changed = False
            for subject_id, dependency_ids in sorted(dependencies.items()):
                inherited = set().union(
                    *(
                        taint.get(dependency_id, set())
                        for dependency_id in dependency_ids
                    )
                )
                before = len(taint.setdefault(subject_id, set()))
                taint[subject_id].update(inherited)
                if len(taint[subject_id]) != before:
                    changed = True
        return {
            subject_id: tuple(sorted(source_ids))
            for subject_id, source_ids in sorted(taint.items())
            if source_ids
        }

    @staticmethod
    def _predecessors(
        predecessor_states: tuple[CatalogEntryState, ...],
        payloads: Mapping[str, TransferEpisode | PriorIdea | KnowledgeClaim],
        catalog_generation: int,
    ) -> dict[str, CatalogEntryState]:
        predecessors: dict[str, CatalogEntryState] = {}
        for state in predecessor_states:
            if state.subject_payload_id in predecessors:
                raise IdentityConflictError(
                    "multiple predecessor states name one subject"
                )
            if state.subject_payload_id not in payloads:
                raise MissingReferenceError(
                    "predecessor state subject is absent from catalog facts"
                )
            if state.catalog_generation != catalog_generation - 1:
                raise ContractValidationError(
                    "predecessor state must come from the immediately prior generation"
                )
            predecessors[state.subject_payload_id] = state
        return predecessors

    def _validate_claim_evidence_closures(
        self,
        closures: tuple[ClaimEvidenceClosure, ...],
        claims: tuple[KnowledgeClaim, ...],
        episodes: tuple[TransferEpisode, ...],
        receipts: tuple[CodingAgentOperationReceipt, ...],
    ) -> dict[str, ClaimEvidenceClosure]:
        claims_by_id = {claim.revision_id: claim for claim in claims}
        closures_by_claim = {closure.claim_revision_id: closure for closure in closures}
        if len(closures_by_claim) != len(closures):
            raise IdentityConflictError(
                "each claim must have exactly one evidence closure"
            )
        if set(closures_by_claim) != set(claims_by_id):
            raise MissingReferenceError(
                "every claim requires one complete evidence closure"
            )
        episode_ids = {episode.episode_id for episode in episodes}
        receipt_by_id = {receipt.operation_receipt_id: receipt for receipt in receipts}
        for revision_id, closure in closures_by_claim.items():
            if set(closure.evaluated_episode_ids) - episode_ids:
                raise MissingReferenceError(
                    "claim evidence closure references a missing episode"
                )
            claim = claims_by_id[revision_id]
            if (
                closure.supporting_episode_ids != claim.supporting_episode_ids
                or closure.contradicting_episode_ids != claim.contradicting_episode_ids
            ):
                raise ContractValidationError(
                    "claim evidence differs from its complete classification"
                )
            if set(claim.proposal_provenance) != {
                "operation_receipt_id",
                "packet_digest",
                "proposal_ordinal",
            }:
                raise ContractValidationError(
                    "claim proposal provenance fields are incomplete"
                )
            if (
                claim.proposal_provenance["operation_receipt_id"]
                != closure.proposer_operation_receipt_id
                or claim.proposal_provenance["packet_digest"] != closure.packet_digest
                or type(claim.proposal_provenance["proposal_ordinal"]) is not int
                or claim.proposal_provenance["proposal_ordinal"] < 0
            ):
                raise ContractValidationError(
                    "claim provenance does not match its evidence closure"
                )
            receipt = receipt_by_id.get(closure.proposer_operation_receipt_id)
            if receipt is None:
                raise MissingReferenceError(
                    "claim proposer operation receipt is absent"
                )
        return closures_by_claim

    @staticmethod
    def _validate_claim_review_evidence(
        assertions: tuple[ReviewAssertion, ...],
        evidence_closures: Mapping[str, ClaimEvidenceClosure],
        receipts: tuple[CodingAgentOperationReceipt, ...],
    ) -> None:
        receipts_by_id = {receipt.operation_receipt_id: receipt for receipt in receipts}
        for assertion in assertions:
            closure = evidence_closures.get(assertion.subject_id)
            if closure is None:
                continue
            if assertion.exact_evidence_refs != closure.evaluated_episode_ids:
                raise IdentityConflictError(
                    "claim review omits part of the evaluated evidence closure"
                )
            proposer_receipt = receipts_by_id[closure.proposer_operation_receipt_id]
            if assertion.reviewer_id == proposer_receipt.principal_id:
                raise IdentityConflictError(
                    "claim proposer cannot review its own historical output"
                )

    @staticmethod
    def _projection_base_state(
        sanitation_report_id: str,
        admitted_sanitation: set[str],
        assertion_view: AssertionAdjudication | None,
    ) -> AdmissionState:
        if assertion_view is not None and assertion_view.disputed:
            return AdmissionState.DISPUTED
        if sanitation_report_id in admitted_sanitation:
            return AdmissionState.ADMITTED
        return AdmissionState.QUARANTINED

    def _claim_base_state(
        self,
        claim: KnowledgeClaim,
        episodes: tuple[TransferEpisode, ...],
        projection_states: Mapping[str, AdmissionState],
        evidence_closure: ClaimEvidenceClosure,
        assertion_view: AssertionAdjudication | None,
    ) -> AdmissionState:
        if assertion_view is None:
            return AdmissionState.QUARANTINED
        if assertion_view.disputed:
            return AdmissionState.DISPUTED
        if not assertion_view.approval_quorum_met:
            return AdmissionState.QUARANTINED
        episode_by_id = {episode.episode_id: episode for episode in episodes}
        referenced_ids = (
            *claim.supporting_episode_ids,
            *claim.contradicting_episode_ids,
        )
        if any(
            episode_id not in episode_by_id
            or projection_states.get(episode_id) is not AdmissionState.ADMITTED
            for episode_id in referenced_ids
        ):
            return AdmissionState.QUARANTINED
        supporting = tuple(
            episode_by_id[episode_id] for episode_id in claim.supporting_episode_ids
        )
        if not supporting:
            return AdmissionState.QUARANTINED
        if any(
            episode.task_context_binding.scope_contract_id != claim.scope_contract_id
            for episode in (
                *supporting,
                *(episode_by_id[value] for value in claim.contradicting_episode_ids),
            )
        ):
            return AdmissionState.QUARANTINED
        if claim.scope_contract_id != self._scope_contract.scope_contract_id:
            return AdmissionState.QUARANTINED
        for episode in supporting:
            episode.task_context_binding.validate_against(self._scope_contract)
            for dimension_id, predicate_value in claim.applicability_predicates.items():
                context_value = episode.task_context_binding.transfer_dimensions.get(
                    dimension_id
                )
                if canonical_json_bytes(context_value) != canonical_json_bytes(
                    predicate_value
                ):
                    return AdmissionState.QUARANTINED
        if not set(referenced_ids).issubset(evidence_closure.evaluated_episode_ids):
            return AdmissionState.QUARANTINED
        if self._settings.admission.require_comparable_support and any(
            episode.attempts[episode.terminal_attempt_revision].comparison_status
            is not ComparisonStatus.COMPARABLE
            for episode in supporting
        ):
            return AdmissionState.QUARANTINED
        if self._settings.admission.require_isolated_support and any(
            episode.attempts[episode.terminal_attempt_revision].intervention_structure
            is not InterventionStructure.ISOLATED_BY_ABLATION
            for episode in supporting
        ):
            return AdmissionState.QUARANTINED
        independent_runs = {
            (episode.source["scope_id"], episode.source["run_id"])
            for episode in supporting
        }
        independent_contexts = {
            episode.task_context_binding.task_context_binding_id
            for episode in supporting
        }
        if (
            len(independent_runs) < self._settings.admission.minimum_supporting_runs
            or len(independent_contexts)
            < self._settings.admission.minimum_supporting_task_contexts
        ):
            return AdmissionState.QUARANTINED
        return AdmissionState.ADMITTED

    @staticmethod
    def _precedence_state(
        base_state: AdmissionState,
        direct_revocations: tuple[str, ...],
        taint_sources: tuple[str, ...],
        successors: tuple[str, ...],
        assertion_view: AssertionAdjudication | None,
    ) -> AdmissionState:
        if direct_revocations or taint_sources:
            return AdmissionState.REVOKED
        if successors:
            return AdmissionState.SUPERSEDED
        if assertion_view is not None and assertion_view.disputed:
            return AdmissionState.DISPUTED
        return base_state
