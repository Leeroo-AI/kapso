"""Complete deterministic reduction of grow-only catalog facts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import canonical_json_bytes
from kapso.cross_run.catalog.admission import AdmissionReducer
from kapso.cross_run.catalog.claims import ClaimProposalPacket, ClaimProposer
from kapso.cross_run.catalog.reviews import CatalogReviewer, CatalogReviewPacket
from kapso.cross_run.catalog.store import (
    CatalogInputDelta,
    CatalogReduction,
    CatalogReductionRequest,
    CatalogReducerError,
)
from kapso.cross_run.contracts import (
    CatalogEntryState,
    CodingAgentOperationReceipt,
    ExpertScopeContract,
    KnowledgeClaim,
    PriorIdea,
    ReviewAssertion,
    RunBundle,
    StrictContract,
    TransferEpisode,
)
from kapso.cross_run.record_registry import CATALOG_FACT_RECORD_TYPES
from kapso.cross_run.record_contracts import (
    BundleProjectionManifest,
    CatalogAgentOperationRecord,
    CatalogRevocation,
    CatalogTaint,
    ClaimEvidenceClosure,
    ExecutionRevisionEvent,
    SanitationReport,
)
from kapso.cross_run.settings import CatalogSettings


class CatalogFactError(CatalogReducerError):
    """The complete catalog fact set is not a valid proof-closed history."""


def _object_id(record: StrictContract) -> str:
    identity_field = record.IDENTITY_FIELD
    if identity_field is None:
        raise CatalogFactError("catalog fact has no content identity")
    return getattr(record, identity_field)


def _namespace(object_id: str) -> str:
    return object_id.split(":sha256:", 1)[0]


def _source_key(record: TransferEpisode | PriorIdea) -> tuple[str, str, str, str]:
    return (
        record.source["scope_id"],
        record.source["run_id"],
        record.source["campaign_id"],
        record.source["idea_id"],
    )


def _bundle_key(bundle: RunBundle) -> str:
    return f"{bundle.scope_id}/{bundle.run_id}/{bundle.campaign_id}"


@dataclass(frozen=True)
class CatalogFactSet:
    """Known typed views over the exact source fact closure."""

    bundles: tuple[RunBundle, ...]
    projection_manifests: tuple[BundleProjectionManifest, ...]
    derivation_events: tuple[ExecutionRevisionEvent, ...]
    sanitation_reports: tuple[SanitationReport, ...]
    episodes: tuple[TransferEpisode, ...]
    prior_ideas: tuple[PriorIdea, ...]
    claims: tuple[KnowledgeClaim, ...]
    assertions: tuple[ReviewAssertion, ...]
    operation_receipts: tuple[CodingAgentOperationReceipt, ...]
    agent_operation_records: tuple[CatalogAgentOperationRecord, ...]
    claim_evidence_closures: tuple[ClaimEvidenceClosure, ...]
    revocations: tuple[CatalogRevocation, ...]
    taints: tuple[CatalogTaint, ...]

    _FIELD_BY_TYPE: ClassVar[dict[type[StrictContract], str]] = {
        BundleProjectionManifest: "projection_manifests",
        CatalogRevocation: "revocations",
        CatalogTaint: "taints",
        ClaimEvidenceClosure: "claim_evidence_closures",
        CodingAgentOperationReceipt: "operation_receipts",
        CatalogAgentOperationRecord: "agent_operation_records",
        ExecutionRevisionEvent: "derivation_events",
        KnowledgeClaim: "claims",
        PriorIdea: "prior_ideas",
        ReviewAssertion: "assertions",
        RunBundle: "bundles",
        SanitationReport: "sanitation_reports",
        TransferEpisode: "episodes",
    }

    @classmethod
    def read(cls, request: CatalogReductionRequest) -> "CatalogFactSet":
        return cls.read_ids(request.fact_object_ids, request.read_object_bytes)

    @classmethod
    def read_ids(
        cls,
        fact_object_ids: tuple[str, ...],
        read_object_bytes: Callable[[str], bytes],
    ) -> "CatalogFactSet":
        grouped: dict[str, list[StrictContract]] = {
            field_name: [] for field_name in cls._FIELD_BY_TYPE.values()
        }
        for object_id in fact_object_ids:
            record_type = CATALOG_FACT_RECORD_TYPES.get(_namespace(object_id))
            if record_type is None:
                raise CatalogFactError("catalog contains an unknown fact namespace")
            record = record_type.from_json_bytes(read_object_bytes(object_id))
            if _object_id(record) != object_id:
                raise CatalogFactError("catalog fact bytes do not own their object ID")
            grouped[cls._FIELD_BY_TYPE[record_type]].append(record)
        return cls(
            **{
                field_name: tuple(sorted(records, key=_object_id))
                for field_name, records in grouped.items()
            }
        )


class CatalogGenerationReducer:
    """Derive one catalog generation from every immutable source fact."""

    def __init__(
        self,
        settings: CatalogSettings,
        scope_contract: ExpertScopeContract,
    ):
        self._settings = settings
        self._scope_contract = scope_contract
        self._admission = AdmissionReducer(settings, scope_contract)

    def __call__(self, request: CatalogReductionRequest) -> CatalogReduction:
        if (
            request.configuration_fingerprint
            != self._settings.configuration_fingerprint
        ):
            raise CatalogFactError(
                "catalog reduction settings do not match the input configuration"
            )
        facts = CatalogFactSet.read(request)
        self._validate_agent_operation_packets(request, facts)
        bundle_frontier = self._validate_projection_history(
            request.scope_contract_id,
            facts,
        )
        predecessor_states = tuple(
            request.parent_generation.active_entry_state_ids[subject_id]
            for subject_id in sorted(request.parent_generation.active_entry_state_ids)
        )
        predecessor_records = tuple(
            CatalogEntryState.from_json_bytes(request.read_object_bytes(state_id))
            for state_id in predecessor_states
        )
        reduced = self._admission.reduce(
            catalog_generation=request.generation_number,
            episodes=facts.episodes,
            prior_ideas=facts.prior_ideas,
            claims=facts.claims,
            assertions=facts.assertions,
            receipts=facts.operation_receipts,
            operation_records=facts.agent_operation_records,
            claim_evidence_closures=facts.claim_evidence_closures,
            sanitation_reports=facts.sanitation_reports,
            proof_object_ids=request.fact_object_ids,
            revocations=facts.revocations,
            taints=facts.taints,
            predecessor_states=predecessor_records,
        )
        return CatalogReduction(
            bundle_frontier=bundle_frontier,
            active_entry_state_ids={
                state.subject_payload_id: state.catalog_entry_state_id
                for state in reduced.states
            },
            derived_objects=reduced.states,
        )

    def _validate_agent_operation_packets(
        self,
        request: CatalogReductionRequest,
        facts: CatalogFactSet,
    ) -> None:
        episodes = {record.episode_id: record for record in facts.episodes}
        prior_ideas = {record.prior_idea_id: record for record in facts.prior_ideas}
        claims = {record.revision_id: record for record in facts.claims}
        assertions = {record.assertion_id: record for record in facts.assertions}
        receipts = {
            record.operation_receipt_id: record for record in facts.operation_receipts
        }
        closures = {
            record.claim_evidence_closure_id: record
            for record in facts.claim_evidence_closures
        }
        subjects: dict[str, StrictContract] = {
            **episodes,
            **prior_ideas,
            **claims,
        }
        delta_configurations: dict[str, set[str]] = defaultdict(set)
        for delta_id in request.applied_input_delta_ids:
            delta = CatalogInputDelta.from_json_bytes(
                request.read_object_bytes(delta_id)
            )
            for object_id in delta.added_object_ids:
                delta_configurations[object_id].add(delta.configuration_fingerprint)
        for operation in facts.agent_operation_records:
            is_new_operation = (
                operation.operation_record_id
                not in request.parent_generation.fact_object_ids
            )
            authority_settings = self._operation_authority_settings(
                operation,
                delta_configurations,
            )
            if operation.operation_kind == "claim_proposal":
                self._validate_claim_operation_packet(
                    operation,
                    authority_settings,
                    is_new_operation,
                    request,
                    episodes,
                    prior_ideas,
                    claims,
                    assertions,
                    receipts,
                )
            else:
                self._validate_review_operation_packet(
                    operation,
                    authority_settings,
                    is_new_operation,
                    request,
                    subjects,
                    assertions,
                    receipts,
                    closures,
                )

    @staticmethod
    def _operation_authority_settings(
        operation: CatalogAgentOperationRecord,
        delta_configurations: dict[str, set[str]],
    ) -> CatalogSettings:
        configuration = operation.operation_preimage.get("catalog_configuration")
        if not isinstance(configuration, Mapping):
            raise CatalogFactError("operation catalog configuration is absent")
        authority_settings = CatalogSettings.from_dict(configuration)
        recorded_fingerprints = delta_configurations.get(
            operation.operation_record_id,
            set(),
        )
        if recorded_fingerprints != {authority_settings.configuration_fingerprint}:
            raise CatalogFactError(
                "operation configuration does not match its publication delta"
            )
        return authority_settings

    def _validate_claim_operation_packet(
        self,
        operation: CatalogAgentOperationRecord,
        authority_settings: CatalogSettings,
        is_new_operation: bool,
        request: CatalogReductionRequest,
        episodes: dict[str, TransferEpisode],
        prior_ideas: dict[str, PriorIdea],
        claims: dict[str, KnowledgeClaim],
        assertions: dict[str, ReviewAssertion],
        receipts: dict[str, CodingAgentOperationReceipt],
    ) -> None:
        preimage = operation.operation_preimage
        if set(preimage) != {
            "catalog_configuration",
            "packet",
            "schema",
            "template",
        }:
            raise CatalogFactError("claim operation preimage fields are invalid")
        if canonical_json_bytes(preimage["catalog_configuration"]) != (
            canonical_json_bytes(authority_settings.to_dict())
        ):
            raise CatalogFactError("claim operation authority preimage differs")
        if is_new_operation and (
            preimage["template"] != ClaimProposer.operation_template()
            or canonical_json_bytes(preimage["schema"])
            != canonical_json_bytes(ClaimProposer.response_schema())
        ):
            raise CatalogFactError("claim operation template or schema differs")
        packet = ClaimProposalPacket.from_dict(operation.packet_payload)
        if packet.scope_contract != self._scope_contract:
            raise CatalogFactError("claim operation packet scope differs")
        if is_new_operation and (
            packet.catalog_generation_id
            != request.parent_generation.catalog_generation_id
            or packet.catalog_generation != request.parent_generation.generation_number
        ):
            raise CatalogFactError("claim operation packet generation differs")
        self._require_packet_records(packet.episodes, episodes, "episode")
        self._require_packet_records(packet.prior_ideas, prior_ideas, "prior idea")
        self._require_packet_records(packet.existing_claims, claims, "claim")
        self._require_packet_records(
            packet.review_assertions,
            assertions,
            "review assertion",
        )
        self._require_packet_records(
            packet.operation_receipts,
            receipts,
            "operation receipt",
        )
        if is_new_operation:
            packet_record_ids = {
                *(record.episode_id for record in packet.episodes),
                *(record.prior_idea_id for record in packet.prior_ideas),
                *(record.revision_id for record in packet.existing_claims),
                *(record.assertion_id for record in packet.review_assertions),
                *(record.operation_receipt_id for record in packet.operation_receipts),
            }
            if packet_record_ids - set(request.parent_generation.fact_object_ids):
                raise CatalogFactError("claim packet references future catalog facts")
        parent_state_ids = set(
            request.parent_generation.active_entry_state_ids.values()
        )
        for state in packet.entry_states:
            stored = CatalogEntryState.from_json_bytes(
                request.read_object_bytes(state.catalog_entry_state_id)
            )
            if stored != state:
                raise CatalogFactError("claim packet entry state bytes differ")
            if is_new_operation and (
                state.catalog_entry_state_id not in parent_state_ids
                or request.parent_generation.active_entry_state_ids.get(
                    state.subject_payload_id
                )
                != state.catalog_entry_state_id
            ):
                raise CatalogFactError("claim packet entry state is not active")
        allowed_proof_ids = (
            set(request.parent_generation.fact_object_ids)
            if is_new_operation
            else set(request.fact_object_ids)
        )
        if set(packet.proof_reference_ids) - allowed_proof_ids:
            raise CatalogFactError("claim packet proof reference is absent")

    def _validate_review_operation_packet(
        self,
        operation: CatalogAgentOperationRecord,
        authority_settings: CatalogSettings,
        is_new_operation: bool,
        request: CatalogReductionRequest,
        subjects: dict[str, StrictContract],
        assertions: dict[str, ReviewAssertion],
        receipts: dict[str, CodingAgentOperationReceipt],
        closures: dict[str, ClaimEvidenceClosure],
    ) -> None:
        preimage = operation.operation_preimage
        if set(preimage) != {
            "catalog_configuration",
            "packet",
            "reviewer",
            "schema",
            "template",
        }:
            raise CatalogFactError("review operation preimage fields are invalid")
        configured_reviewers = {
            reviewer.reviewer_id: reviewer for reviewer in authority_settings.reviewers
        }
        reviewer_payload = preimage["reviewer"]
        if not isinstance(reviewer_payload, Mapping):
            raise CatalogFactError("review operation reviewer is invalid")
        reviewer = configured_reviewers.get(reviewer_payload.get("reviewer_id"))
        if (
            reviewer is None
            or canonical_json_bytes(reviewer_payload)
            != canonical_json_bytes(reviewer.to_dict())
            or canonical_json_bytes(preimage["catalog_configuration"])
            != canonical_json_bytes(authority_settings.to_dict())
        ):
            raise CatalogFactError("review operation authority preimage differs")
        if is_new_operation and (
            preimage["template"] != CatalogReviewer.operation_template()
            or canonical_json_bytes(preimage["schema"])
            != canonical_json_bytes(
                CatalogReviewer.response_schema_for(authority_settings)
            )
        ):
            raise CatalogFactError("review operation template or schema differs")
        packet = CatalogReviewPacket.from_dict(operation.packet_payload)
        if packet.scope_contract != self._scope_contract:
            raise CatalogFactError("review operation packet scope differs")
        if is_new_operation and (
            packet.catalog_generation_id
            != request.parent_generation.catalog_generation_id
            or packet.catalog_generation != request.parent_generation.generation_number
        ):
            raise CatalogFactError("review operation packet generation differs")
        subject = subjects.get(packet.subject_id)
        if subject is None or canonical_json_bytes(packet.subject["payload"]) != (
            canonical_json_bytes(subject.to_dict())
        ):
            raise CatalogFactError("review operation subject bytes differ")
        for envelope in packet.evidence_records:
            evidence = subjects.get(envelope["record_id"])
            if evidence is None or canonical_json_bytes(envelope["payload"]) != (
                canonical_json_bytes(evidence.to_dict())
            ):
                raise CatalogFactError("review operation evidence bytes differ")
        self._require_packet_records(
            packet.previous_assertions,
            assertions,
            "previous review assertion",
        )
        self._require_packet_records(
            packet.previous_operation_receipts,
            receipts,
            "previous operation receipt",
        )
        if packet.proposer_operation_receipt is not None:
            self._require_packet_records(
                (packet.proposer_operation_receipt,),
                receipts,
                "claim proposer receipt",
            )
        if packet.claim_evidence_closure is not None:
            self._require_packet_records(
                (packet.claim_evidence_closure,),
                closures,
                "claim evidence closure",
            )
        if is_new_operation:
            packet_reference_ids = {
                packet.subject_id,
                *packet.evidence_record_ids,
                *(value.assertion_id for value in packet.previous_assertions),
                *(
                    value.operation_receipt_id
                    for value in packet.previous_operation_receipts
                ),
            }
            if packet.proposer_operation_receipt is not None:
                packet_reference_ids.add(
                    packet.proposer_operation_receipt.operation_receipt_id
                )
            if packet.claim_evidence_closure is not None:
                packet_reference_ids.add(
                    packet.claim_evidence_closure.claim_evidence_closure_id
                )
            if packet_reference_ids - set(request.parent_generation.fact_object_ids):
                raise CatalogFactError("review packet references future catalog facts")

    @staticmethod
    def _require_packet_records(
        records: tuple[StrictContract, ...],
        catalog_records: dict[str, StrictContract],
        name: str,
    ) -> None:
        for record in records:
            record_id = _object_id(record)
            if catalog_records.get(record_id) != record:
                raise CatalogFactError(f"{name} packet record bytes differ")

    @classmethod
    def _validate_projection_history(
        cls,
        scope_contract_id: str,
        facts: CatalogFactSet,
    ) -> dict[str, str]:
        bundles = {bundle.bundle_id: bundle for bundle in facts.bundles}
        manifests = {
            manifest.source_bundle_id: manifest
            for manifest in facts.projection_manifests
        }
        if len(bundles) != len(facts.bundles):
            raise CatalogFactError("bundle identities are not unique")
        if len(manifests) != len(facts.projection_manifests):
            raise CatalogFactError("one bundle has multiple projection manifests")
        if set(bundles) != set(manifests):
            raise CatalogFactError(
                "every bundle requires exactly one projection manifest"
            )
        if any(
            bundle.scope_contract_id != scope_contract_id for bundle in bundles.values()
        ):
            raise CatalogFactError("bundle leaves the catalog scope contract")
        if any(claim.scope_contract_id != scope_contract_id for claim in facts.claims):
            raise CatalogFactError("claim leaves the catalog scope contract")

        episodes = {record.episode_id: record for record in facts.episodes}
        prior_ideas = {record.prior_idea_id: record for record in facts.prior_ideas}
        events = {record.event_id: record for record in facts.derivation_events}
        reports = {record.report_id: record for record in facts.sanitation_reports}
        cls._require_unique_records(episodes, facts.episodes, "episode")
        cls._require_unique_records(prior_ideas, facts.prior_ideas, "prior idea")
        cls._require_unique_records(events, facts.derivation_events, "derivation event")
        cls._require_unique_records(
            reports, facts.sanitation_reports, "sanitation report"
        )

        assigned_episode_ids: list[str] = []
        assigned_prior_ids: list[str] = []
        assigned_event_ids: list[str] = []
        projections_by_bundle: dict[
            str, dict[tuple[str, str, str, str], TransferEpisode | PriorIdea]
        ] = {}
        for bundle_id, manifest in sorted(manifests.items()):
            if manifest.sanitation_report_id not in reports:
                raise CatalogFactError("projection sanitation report is absent")
            report = reports[manifest.sanitation_report_id]
            if report.status != "admitted":
                raise CatalogFactError("only admitted bundle projections may enter")
            bundle = bundles[bundle_id]
            if (
                report.scope_id != bundle.scope_id
                or report.task_family_id != bundle.task_context_binding.task_family_id
            ):
                raise CatalogFactError(
                    "projection sanitation report names another task context"
                )
            missing_episode_ids = set(manifest.episode_ids) - set(episodes)
            missing_prior_ids = set(manifest.prior_idea_ids) - set(prior_ideas)
            if missing_episode_ids or missing_prior_ids:
                raise CatalogFactError("projection record is absent")
            bundle_records: tuple[TransferEpisode | PriorIdea, ...] = (
                *(episodes[record_id] for record_id in manifest.episode_ids),
                *(prior_ideas[record_id] for record_id in manifest.prior_idea_ids),
            )
            if any(record.source_bundle_id != bundle_id for record in bundle_records):
                raise CatalogFactError("projection record names another bundle")
            if any(
                record.sanitation_report_id != manifest.sanitation_report_id
                for record in bundle_records
            ):
                raise CatalogFactError(
                    "projection record names another sanitation report"
                )
            if any(
                record.task_context_binding.scope_contract_id != scope_contract_id
                for record in bundle_records
            ):
                raise CatalogFactError("projection record leaves the scope contract")
            if any(
                record.task_context_binding != bundle.task_context_binding
                for record in bundle_records
            ):
                raise CatalogFactError(
                    "projection record changed the bundle task context"
                )
            if any(
                record.source["scope_id"] != bundle.scope_id
                or record.source["run_id"] != bundle.run_id
                or record.source["campaign_id"] != bundle.campaign_id
                for record in bundle_records
            ):
                raise CatalogFactError("projection record changed source identity")
            if any(
                event_id not in events for event_id in manifest.derivation_object_ids
            ):
                raise CatalogFactError("projection derivation event is absent")
            if any(
                events[event_id].run_id != bundle.run_id
                or events[event_id].campaign_id != bundle.campaign_id
                for event_id in manifest.derivation_object_ids
            ):
                raise CatalogFactError("projection event changed source identity")
            if any(
                bundle_id not in record.derivation_refs
                for record in bundle_records
                if isinstance(record, TransferEpisode)
            ):
                raise CatalogFactError("episode derivation omits its source bundle")
            expected_derivation_ids = {
                reference
                for record in bundle_records
                if isinstance(record, TransferEpisode)
                for reference in record.derivation_refs
                if reference != bundle_id
            }
            if expected_derivation_ids != set(manifest.derivation_object_ids):
                raise CatalogFactError("projection derivation closure is not exact")
            source_records = {_source_key(record): record for record in bundle_records}
            if len(source_records) != len(bundle_records):
                raise CatalogFactError("one source idea projected more than once")
            projections_by_bundle[bundle_id] = source_records
            assigned_episode_ids.extend(manifest.episode_ids)
            assigned_prior_ids.extend(manifest.prior_idea_ids)
            assigned_event_ids.extend(manifest.derivation_object_ids)

        cls._require_exact_assignment(assigned_episode_ids, set(episodes), "episodes")
        cls._require_exact_assignment(
            assigned_prior_ids, set(prior_ideas), "prior ideas"
        )
        if set(assigned_event_ids) != set(events):
            raise CatalogFactError("derivation events are not assigned")

        grouped: dict[str, list[RunBundle]] = defaultdict(list)
        for bundle in bundles.values():
            grouped[_bundle_key(bundle)].append(bundle)
        frontier: dict[str, str] = {}
        for logical_key, group in sorted(grouped.items()):
            chain = cls._bundle_chain(tuple(group))
            cls._validate_projection_supersession(chain, projections_by_bundle)
            frontier[logical_key] = chain[-1].bundle_id
        return frontier

    @staticmethod
    def _require_unique_records(
        records_by_id: dict[str, StrictContract],
        records: tuple[StrictContract, ...],
        name: str,
    ) -> None:
        if len(records_by_id) != len(records):
            raise CatalogFactError(f"{name} identities are not unique")

    @staticmethod
    def _require_exact_assignment(
        assigned_ids: list[str],
        expected_ids: set[str],
        name: str,
    ) -> None:
        if (
            len(assigned_ids) != len(set(assigned_ids))
            or set(assigned_ids) != expected_ids
        ):
            raise CatalogFactError(f"{name} are not assigned exactly once")

    @staticmethod
    def _bundle_chain(group: tuple[RunBundle, ...]) -> tuple[RunBundle, ...]:
        by_id = {bundle.bundle_id: bundle for bundle in group}
        roots = tuple(bundle for bundle in group if bundle.supersedes_bundle_id is None)
        if len(roots) != 1:
            raise CatalogFactError("bundle lineage must have exactly one root")
        children: dict[str, list[RunBundle]] = defaultdict(list)
        for bundle in group:
            if bundle.supersedes_bundle_id is not None:
                if bundle.supersedes_bundle_id not in by_id:
                    raise CatalogFactError("bundle supersession predecessor is absent")
                children[bundle.supersedes_bundle_id].append(bundle)
        chain = [roots[0]]
        while chain[-1].bundle_id in children:
            successors = children[chain[-1].bundle_id]
            if len(successors) != 1:
                raise CatalogFactError("bundle supersession lineage forked")
            successor = successors[0]
            CatalogGenerationReducer._validate_adjacent_bundles(chain[-1], successor)
            chain.append(successor)
        if len(chain) != len(group):
            raise CatalogFactError("bundle supersession lineage is disconnected")
        return tuple(chain)

    @staticmethod
    def _validate_adjacent_bundles(parent: RunBundle, child: RunBundle) -> None:
        stable_fields = (
            "scope_contract_id",
            "scope_id",
            "run_id",
            "campaign_id",
            "started_at",
            "kapso_commit",
            "launch_manifest_id",
            "knowledge_snapshot_id",
            "expert_base_release_id",
            "task_context_binding",
            "artifact_environment",
        )
        if child.supersedes_bundle_id != parent.bundle_id:
            raise CatalogFactError("bundle predecessor identity changed")
        if child.capture_generation != parent.capture_generation + 1:
            raise CatalogFactError("bundle capture generations are not contiguous")
        if child.checkpoint_frontier < parent.checkpoint_frontier:
            raise CatalogFactError("bundle checkpoint frontier moved backwards")
        if set(child.capture_watermarks) != set(parent.capture_watermarks) or any(
            child.capture_watermarks[name] < parent.capture_watermarks[name]
            for name in parent.capture_watermarks
        ):
            raise CatalogFactError("bundle capture watermarks moved backwards")
        if any(getattr(child, name) != getattr(parent, name) for name in stable_fields):
            raise CatalogFactError("bundle supersession changed stable run identity")

    @staticmethod
    def _validate_projection_supersession(
        chain: tuple[RunBundle, ...],
        projections_by_bundle: dict[
            str, dict[tuple[str, str, str, str], TransferEpisode | PriorIdea]
        ],
    ) -> None:
        prior_records: dict[tuple[str, str, str, str], TransferEpisode | PriorIdea] = {}
        for position, bundle in enumerate(chain):
            current_records = projections_by_bundle[bundle.bundle_id]
            if position == 0 and any(
                record.supersedes_projection_id is not None
                for record in current_records.values()
            ):
                raise CatalogFactError("root projection names a predecessor")
            if not set(prior_records).issubset(current_records):
                raise CatalogFactError("successor bundle dropped a source idea")
            for source_key, record in current_records.items():
                predecessor = prior_records.get(source_key)
                expected_predecessor_id = (
                    None if predecessor is None else _object_id(predecessor)
                )
                if record.supersedes_projection_id != expected_predecessor_id:
                    raise CatalogFactError(
                        "projection supersession does not match the bundle frontier"
                    )
                if isinstance(predecessor, TransferEpisode) and isinstance(
                    record, PriorIdea
                ):
                    raise CatalogFactError("an executed idea reverted to a prior idea")
            prior_records = current_records
