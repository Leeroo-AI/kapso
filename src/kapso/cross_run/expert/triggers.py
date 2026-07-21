"""Deterministic, persisted eligibility decisions for expert candidate calls."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Mapping

from kapso.cross_run.agent_artifacts import CodingAgentWorkspaceAccess
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    CodingAgentOperationReceipt,
    ComparisonStatus,
    CrossRunTaskBindingSettings,
    EpisodeEvaluationStatus,
    ExecutionStatus,
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertBaseReleaseManifest,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
    KnowledgeClaim,
    KnowledgeSnapshotManifest,
    MissingReferenceError,
    PublicationArtifactKind,
    StrictContract,
    TransferEpisode,
)
from kapso.cross_run.github.materializer import (
    CacheVerificationReceipt,
    SourceArchiveExtractionReceipt,
)
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackage
from kapso.cross_run.record_registry import parse_knowledge_record_payload
from kapso.cross_run.settings import ExpertTriggerSettings

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARCHITECTURE_OBSERVATION_KINDS = {
    "adapter_leakage",
    "contract_topology_mismatch",
    "exact_source_duplication",
    "semantic_navigation_failure",
}
_CAPABILITY_OBSERVATION_KINDS = {
    "mechanically_general_fix",
    "repeated_independent_difficulty",
    "released_capability_contradiction",
}
_RENAME_NOREPLACE = 1


class ExpertTriggerError(ValueError):
    """Expert trigger evidence or persistence is invalid."""


class ExpertTriggerObservationKind(str, Enum):
    EXACT_SOURCE_DUPLICATION = "exact_source_duplication"
    ADAPTER_LEAKAGE = "adapter_leakage"
    CONTRACT_TOPOLOGY_MISMATCH = "contract_topology_mismatch"
    SEMANTIC_NAVIGATION_FAILURE = "semantic_navigation_failure"
    MECHANICALLY_GENERAL_FIX = "mechanically_general_fix"
    REPEATED_INDEPENDENT_DIFFICULTY = "repeated_independent_difficulty"
    RELEASED_CAPABILITY_CONTRADICTION = "released_capability_contradiction"


@dataclass(frozen=True)
class ExpertTriggerObservation(StrictContract):
    """One typed exact-tree observation from a deterministic or reviewed inspector."""

    observation_id: str
    kind: ExpertTriggerObservationKind
    parent_tree_hash: str
    inspection_policy_version: str
    configuration_fingerprint: str
    inspection_operation: CodingAgentOperationReceipt
    inspection_final_output: str
    difficulty_signature: str | None
    difficulty_evidence_signatures: Mapping[str, str]
    description: str
    affected_capability_ids: tuple[str, ...]
    affected_paths: tuple[str, ...]
    exact_evidence_ids: tuple[str, ...]
    independent_lineage_ids: tuple[str, ...]
    task_context_binding_ids: tuple[str, ...]
    occurrence_count: int

    CONTENT_NAMESPACE = "expert-trigger-observation"
    IDENTITY_FIELD = "observation_id"

    def _validate(self) -> None:
        _require_digest(self.parent_tree_hash, "trigger observation parent tree hash")
        require_identifier(
            self.inspection_policy_version,
            "trigger observation inspection policy version",
        )
        _require_digest(
            self.configuration_fingerprint,
            "trigger observation configuration fingerprint",
        )
        if not isinstance(self.inspection_final_output, str) or not (
            self.inspection_final_output.strip()
        ):
            raise ExpertTriggerError("trigger inspection final output is empty")
        final_output_digest = tree_or_blob_digest(
            self.inspection_final_output.encode("utf-8")
        )
        if (
            self.inspection_operation.artifact_checksums["final.json"]
            != final_output_digest
            or self.inspection_operation.workspace_access
            is not CodingAgentWorkspaceAccess.READ_ONLY
        ):
            raise ExpertTriggerError(
                "trigger observation differs from its inspected result artifact"
            )
        inspected_payload = parse_json_bytes(
            self.inspection_final_output.encode("utf-8")
        )
        expected_payload = {
            "affected_capability_ids": self.affected_capability_ids,
            "affected_paths": self.affected_paths,
            "configuration_fingerprint": self.configuration_fingerprint,
            "description": self.description,
            "difficulty_evidence_signatures": self.difficulty_evidence_signatures,
            "difficulty_signature": self.difficulty_signature,
            "exact_evidence_ids": self.exact_evidence_ids,
            "independent_lineage_ids": self.independent_lineage_ids,
            "inspection_policy_version": self.inspection_policy_version,
            "kind": self.kind.value,
            "occurrence_count": self.occurrence_count,
            "parent_tree_hash": self.parent_tree_hash,
            "task_context_binding_ids": self.task_context_binding_ids,
        }
        if canonical_json_bytes(inspected_payload) != canonical_json_bytes(
            expected_payload
        ):
            raise ExpertTriggerError(
                "trigger observation fields differ from final inspection result"
            )
        if not self.description.strip():
            raise ExpertTriggerError("trigger observation description is empty")
        _require_sorted_identifiers(
            self.affected_capability_ids,
            "trigger observation affected capability IDs",
        )
        if self.affected_paths != tuple(sorted(set(self.affected_paths))):
            raise ExpertTriggerError(
                "trigger observation paths must be sorted and unique"
            )
        for relative_path in self.affected_paths:
            _require_relative_path(relative_path, "trigger observation path")
        _require_sorted_content_ids(
            self.exact_evidence_ids,
            "trigger observation exact evidence IDs",
            required=True,
        )
        for values, name in (
            (self.independent_lineage_ids, "independent_lineage_ids"),
            (self.task_context_binding_ids, "task_context_binding_ids"),
        ):
            if values != tuple(sorted(set(values))):
                raise ExpertTriggerError(f"{name} must be sorted and unique")
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ExpertTriggerError(f"{name} must contain non-empty text")
        if isinstance(self.occurrence_count, bool) or self.occurrence_count <= 0:
            raise ExpertTriggerError(
                "trigger observation occurrence count must be positive"
            )
        if self.kind is ExpertTriggerObservationKind.EXACT_SOURCE_DUPLICATION and (
            self.occurrence_count != len(self.affected_paths)
            or len(self.affected_capability_ids) < 2
        ):
            raise ExpertTriggerError(
                "exact-source duplication requires paths across capabilities"
            )
        if (
            self.kind is ExpertTriggerObservationKind.REPEATED_INDEPENDENT_DIFFICULTY
            and not self.independent_lineage_ids
        ):
            raise ExpertTriggerError(
                "repeated difficulty requires typed independent lineages"
            )
        if (
            self.kind is ExpertTriggerObservationKind.REPEATED_INDEPENDENT_DIFFICULTY
        ) != (self.difficulty_signature is not None):
            raise ExpertTriggerError(
                "only repeated difficulty requires a typed difficulty signature"
            )
        if self.difficulty_signature is not None:
            _require_digest(
                self.difficulty_signature,
                "trigger observation difficulty signature",
            )
            if set(self.difficulty_evidence_signatures) != set(
                self.exact_evidence_ids
            ) or set(self.difficulty_evidence_signatures.values()) != {
                self.difficulty_signature
            }:
                raise ExpertTriggerError(
                    "difficulty evidence must share the declared typed signature"
                )
        elif self.difficulty_evidence_signatures:
            raise ExpertTriggerError(
                "non-difficulty observation cannot carry difficulty signatures"
            )


@dataclass(frozen=True)
class ExpertParentTreeReceipt(StrictContract):
    """Verified binding from a released archive to one exact candidate parent tree."""

    parent_tree_receipt_id: str
    release_id: str
    cache_verification_receipt: CacheVerificationReceipt
    source_extraction_receipt: SourceArchiveExtractionReceipt
    parent_tree_hash: str
    repository_map_id: str
    module_contract_ids: tuple[str, ...]
    materializer_version: str

    CONTENT_NAMESPACE = "expert-parent-tree-receipt"
    IDENTITY_FIELD = "parent_tree_receipt_id"

    def _validate(self) -> None:
        require_content_id(self.release_id, "parent tree receipt release_id")
        require_content_id(
            self.repository_map_id,
            "parent tree receipt repository_map_id",
        )
        _require_digest(self.parent_tree_hash, "parent tree hash")
        if (
            self.cache_verification_receipt.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or self.cache_verification_receipt.artifact_id != self.release_id
        ):
            raise ExpertTriggerError(
                "parent tree cache receipt belongs to another release"
            )
        extraction = self.source_extraction_receipt
        if extraction.artifact_id != self.release_id:
            raise ExpertTriggerError(
                "parent source extraction belongs to another release"
            )
        if (
            self.cache_verification_receipt.asset_digests.get(
                extraction.source_archive_ref
            )
            != extraction.source_archive_digest
        ):
            raise ExpertTriggerError(
                "parent source extraction differs from the verified release asset"
            )
        if self.parent_tree_hash != extraction.source_tree_hash:
            raise ExpertTriggerError(
                "parent tree differs from the verified source extraction"
            )
        _require_sorted_content_ids(
            self.module_contract_ids,
            "parent tree receipt module contract IDs",
            required=True,
        )
        require_identifier(self.materializer_version, "materializer_version")


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ExpertTriggerError(f"{name} must be a sha256 digest")


def _require_relative_path(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or not value
        or path.is_absolute()
        or ".." in path.parts
        or path == PurePosixPath(".")
        or value != path.as_posix()
    ):
        raise ExpertTriggerError(f"{name} must be normalized and relative")


def _require_sorted_identifiers(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ExpertTriggerError(f"{name} must be sorted and unique")
    for value in values:
        require_identifier(value, name)


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    required: bool = False,
) -> None:
    if required and not values:
        raise ExpertTriggerError(f"{name} must not be empty")
    if values != tuple(sorted(set(values))):
        raise ExpertTriggerError(f"{name} must be sorted and unique")
    for value in values:
        require_content_id(value, name)


def _sorted_contract_ids(
    records: tuple[StrictContract, ...],
    identity_field: str,
    name: str,
) -> tuple[str, ...]:
    identities = tuple(getattr(record, identity_field) for record in records)
    if identities != tuple(sorted(set(identities))):
        raise ExpertTriggerError(f"{name} must be sorted and uniquely identified")
    return identities


def episode_lineage_id(
    episode: TransferEpisode,
    episodes_by_id: dict[str, TransferEpisode],
) -> str:
    root = episode
    seen = {episode.episode_id}
    while root.parent_episode_ref is not None:
        if root.parent_episode_ref in seen:
            raise ExpertTriggerError("episode lineage contains a cycle")
        seen.add(root.parent_episode_ref)
        parent = episodes_by_id.get(root.parent_episode_ref)
        if parent is None:
            raise MissingReferenceError(
                "trigger packet omits an episode lineage ancestor"
            )
        root = parent
    return "/".join(
        (
            root.source["scope_id"],
            root.source["campaign_id"],
        )
    )


def _transfer_context_signature(episode: TransferEpisode) -> str:
    binding = episode.task_context_binding
    return tree_or_blob_digest(
        canonical_json_bytes(
            {
                "input_contract_fingerprint": binding.input_contract_fingerprint,
                "starting_artifact_refs": binding.starting_artifact_refs,
                "target_contract_fingerprint": binding.target_contract_fingerprint,
                "task_family_id": binding.task_family_id,
                "transfer_dimensions": binding.transfer_dimensions,
            }
        )
    )


@dataclass(frozen=True)
class ExpertTriggerEvidencePacket(StrictContract):
    """Complete exhaustive evidence and parent topology for one decision."""

    evidence_packet_id: str
    knowledge_snapshot_manifest: KnowledgeSnapshotManifest
    knowledge_record_closure_digest: str
    configuration_fingerprint: str
    scope_contract: ExpertScopeContract
    parent_scope_contract: ExpertScopeContract | None
    parent_release: ExpertBaseReleaseManifest | None
    parent_tree_receipt: ExpertParentTreeReceipt | None
    parent_tree_hash: str
    repository_map: ExpertRepositoryMap | None
    module_contracts: tuple[ExpertModuleContract, ...]
    episodes: tuple[TransferEpisode, ...]
    claims: tuple[KnowledgeClaim, ...]
    trigger_observations: tuple[ExpertTriggerObservation, ...]
    active_task_bindings: tuple[CrossRunTaskBindingSettings, ...]
    proof_reference_ids: tuple[str, ...]

    CONTENT_NAMESPACE = "expert-trigger-evidence-packet"
    IDENTITY_FIELD = "evidence_packet_id"

    @property
    def knowledge_snapshot_id(self) -> str:
        return self.knowledge_snapshot_manifest.snapshot_id

    @property
    def parent_release_id(self) -> str | None:
        return None if self.parent_release is None else self.parent_release.release_id

    @property
    def active_task_family_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *(binding.task_family_id for binding in self.active_task_bindings),
                    *(
                        episode.task_context_binding.task_family_id
                        for episode in self.episodes
                        if episode.task_context_binding.scope_contract_id
                        == self.scope_contract.scope_contract_id
                    ),
                }
            )
        )

    def _validate(self) -> None:
        _require_digest(
            self.knowledge_record_closure_digest,
            "knowledge record closure digest",
        )
        _require_digest(self.configuration_fingerprint, "configuration fingerprint")
        _require_digest(self.parent_tree_hash, "expert trigger parent tree hash")
        module_ids = _sorted_contract_ids(
            self.module_contracts,
            "module_contract_id",
            "expert trigger module contracts",
        )
        episode_ids = _sorted_contract_ids(
            self.episodes,
            "episode_id",
            "expert trigger episodes",
        )
        claim_ids = _sorted_contract_ids(
            self.claims,
            "revision_id",
            "expert trigger claims",
        )
        observation_ids = _sorted_contract_ids(
            self.trigger_observations,
            "observation_id",
            "expert trigger observations",
        )
        if self.knowledge_snapshot_manifest.scope_id != self.scope_contract.scope_id:
            raise ExpertTriggerError("knowledge snapshot belongs to another scope")
        if self.parent_release is None:
            if (
                self.parent_tree_hash != EMPTY_EXPERT_TREE_DIGEST
                or self.parent_scope_contract is not None
                or self.parent_tree_receipt is not None
                or self.repository_map is not None
                or self.module_contracts
            ):
                raise ExpertTriggerError(
                    "bootstrap requires the explicit canonical empty parent"
                )
        else:
            if (
                self.parent_scope_contract is None
                or self.parent_tree_receipt is None
                or self.repository_map is None
            ):
                raise ExpertTriggerError(
                    "non-bootstrap trigger requires parent scope, tree receipt, and map"
                )
            if self.parent_tree_hash == EMPTY_EXPERT_TREE_DIGEST:
                raise ExpertTriggerError(
                    "a released parent cannot use the canonical empty tree"
                )
            if (
                self.parent_release.scope_contract_id
                != self.parent_scope_contract.scope_contract_id
                or self.parent_release.scope_id != self.parent_scope_contract.scope_id
                or self.parent_scope_contract.scope_id != self.scope_contract.scope_id
            ):
                raise ExpertTriggerError("parent release belongs to another scope")
            if (
                self.scope_contract.scope_contract_id
                != self.parent_scope_contract.scope_contract_id
                and self.scope_contract.supersedes_scope_contract_id
                != self.parent_scope_contract.scope_contract_id
            ):
                raise ExpertTriggerError(
                    "current scope must equal or directly supersede the parent scope"
                )
            if (
                self.scope_contract.scope_contract_id
                != self.parent_scope_contract.scope_contract_id
            ):
                parent_family_ids = {
                    family.task_family_id
                    for family in self.parent_scope_contract.task_family_ontology
                }
                parent_dimension_ids = {
                    schema.dimension_id
                    for schema in self.parent_scope_contract.context_dimension_schemas
                }
                if any(
                    not set(edge.source_ids).issubset(parent_family_ids)
                    for edge in self.scope_contract.task_family_lineage
                ):
                    raise ExpertTriggerError(
                        "successor task-family lineage source is absent from parent"
                    )
                if any(
                    not set(edge.source_ids).issubset(parent_dimension_ids)
                    for edge in self.scope_contract.context_dimension_lineage
                ):
                    raise ExpertTriggerError(
                        "successor context lineage source is absent from parent"
                    )
        evidence_scope_contract = (
            self.scope_contract
            if self.knowledge_snapshot_manifest.scope_contract_id
            == self.scope_contract.scope_contract_id
            else self.parent_scope_contract
        )
        if evidence_scope_contract is None or (
            self.knowledge_snapshot_manifest.scope_contract_id
            != evidence_scope_contract.scope_contract_id
        ):
            raise ExpertTriggerError(
                "knowledge snapshot scope is neither current nor parent"
            )
        if self.repository_map is not None:
            if (
                self.repository_map.scope_contract_id
                != self.parent_scope_contract.scope_contract_id
            ):
                raise ExpertTriggerError("repository map differs from parent scope")
            map_contract_ids = tuple(
                sorted(
                    node.module_contract_ref
                    for node in self.repository_map.capability_nodes
                )
            )
            if map_contract_ids != module_ids:
                raise MissingReferenceError(
                    "trigger packet module contracts differ from the repository map"
                )
            if self.parent_release is not None:
                if (
                    self.parent_release.repository_map_ref
                    != self.repository_map.repository_map_id
                ):
                    raise MissingReferenceError(
                        "parent release names another repository map"
                    )
                module_versions = {
                    contract.module_id: contract.version
                    for contract in self.module_contracts
                }
                if dict(self.parent_release.module_versions) != module_versions:
                    raise MissingReferenceError(
                        "parent release module versions differ from packet contracts"
                    )
                expected_archive_digest = self.parent_release.checksums[
                    self.parent_release.source_archive_ref
                ]
                if (
                    self.parent_tree_receipt.release_id
                    != self.parent_release.release_id
                    or self.parent_tree_receipt.cache_verification_receipt.asset_digests.get(
                        self.parent_release.source_archive_ref
                    )
                    != expected_archive_digest
                    or self.parent_tree_receipt.source_extraction_receipt.source_archive_ref
                    != self.parent_release.source_archive_ref
                    or self.parent_tree_receipt.source_extraction_receipt.source_archive_digest
                    != expected_archive_digest
                    or self.parent_tree_receipt.parent_tree_hash
                    != self.parent_tree_hash
                    or self.parent_tree_receipt.repository_map_id
                    != self.repository_map.repository_map_id
                    or self.parent_tree_receipt.module_contract_ids != module_ids
                ):
                    raise ExpertTriggerError(
                        "parent tree receipt differs from released topology"
                    )
        known_families = {
            family.task_family_id for family in self.scope_contract.task_family_ontology
        }
        binding_keys = tuple(
            (binding.task_family_id, binding.task_adapter_id)
            for binding in self.active_task_bindings
        )
        if not binding_keys or binding_keys != tuple(sorted(set(binding_keys))):
            raise ExpertTriggerError(
                "active task bindings must be non-empty, sorted, and unique"
            )
        for binding in self.active_task_bindings:
            self.scope_contract.validate_binding(binding)
        if not set(self.active_task_family_ids).issubset(known_families):
            raise ExpertTriggerError("active task family leaves the current scope")
        _require_sorted_content_ids(
            self.proof_reference_ids,
            "trigger proof reference IDs",
        )
        episodes_by_id = {episode.episode_id: episode for episode in self.episodes}
        for episode in self.episodes:
            binding = episode.task_context_binding
            if binding.scope_contract_id != evidence_scope_contract.scope_contract_id:
                raise ExpertTriggerError("trigger episode belongs to another scope")
            binding.validate_against(evidence_scope_contract)
            episode_lineage_id(episode, episodes_by_id)
            required_proof = {
                episode.source_bundle_id,
                episode.sanitation_report_id,
                *episode.derivation_refs,
            }
            if episode.supersedes_projection_id is not None:
                required_proof.add(episode.supersedes_projection_id)
            if not required_proof.issubset(self.proof_reference_ids):
                raise MissingReferenceError(
                    "trigger packet omits episode proof references"
                )
        admitted_episode_ids = set(
            self.knowledge_snapshot_manifest.admitted_episode_ids
        )
        episode_closure_ids = set(admitted_episode_ids)
        lineage_pending = list(sorted(admitted_episode_ids))
        while lineage_pending:
            episode_id = lineage_pending.pop()
            episode = episodes_by_id.get(episode_id)
            if episode is None:
                raise MissingReferenceError(
                    "trigger packet omits an admitted snapshot episode"
                )
            if (
                episode.parent_episode_ref is not None
                and episode.parent_episode_ref not in episode_closure_ids
            ):
                episode_closure_ids.add(episode.parent_episode_ref)
                lineage_pending.append(episode.parent_episode_ref)
        if episode_closure_ids != set(episode_ids):
            raise ExpertTriggerError(
                "trigger episodes differ from the snapshot lineage closure"
            )
        if set(claim_ids) != set(
            self.knowledge_snapshot_manifest.active_claim_revision_ids
        ):
            raise ExpertTriggerError(
                "trigger claims differ from the active snapshot claim set"
            )
        for claim in self.claims:
            if claim.scope_contract_id != evidence_scope_contract.scope_contract_id:
                raise ExpertTriggerError("trigger claim belongs to another scope")
            evidence_ids = set(claim.supporting_episode_ids)
            evidence_ids.update(claim.contradicting_episode_ids)
            if not evidence_ids.issubset(episode_ids):
                raise MissingReferenceError(
                    "trigger packet omits claim evidence episodes"
                )
            required_claim_proof = set(claim.supersedes_revision_ids)
            operation_receipt_id = claim.proposal_provenance.get("operation_receipt_id")
            if operation_receipt_id is not None:
                required_claim_proof.add(
                    require_content_id(
                        operation_receipt_id,
                        "claim proposal operation receipt",
                    )
                )
            if not required_claim_proof.issubset(self.proof_reference_ids):
                raise MissingReferenceError(
                    "trigger packet omits claim provenance references"
                )
        known_capability_ids = (
            set()
            if self.repository_map is None
            else {node.capability_id for node in self.repository_map.capability_nodes}
        )
        known_evidence_ids = {
            self.scope_contract.scope_contract_id,
            self.knowledge_snapshot_manifest.snapshot_id,
            *module_ids,
            *episode_ids,
            *claim_ids,
            *observation_ids,
            *(
                observation.inspection_operation.operation_receipt_id
                for observation in self.trigger_observations
            ),
            *self.proof_reference_ids,
        }
        if self.repository_map is not None:
            known_evidence_ids.add(self.repository_map.repository_map_id)
        if self.parent_release is not None:
            known_evidence_ids.add(self.parent_release.release_id)
        if self.parent_tree_receipt is not None:
            known_evidence_ids.add(self.parent_tree_receipt.parent_tree_receipt_id)
        if self.parent_scope_contract is not None:
            known_evidence_ids.add(self.parent_scope_contract.scope_contract_id)
        for observation in self.trigger_observations:
            if observation.parent_tree_hash != self.parent_tree_hash:
                raise ExpertTriggerError("trigger observation uses another parent tree")
            if not set(observation.affected_capability_ids).issubset(
                known_capability_ids
            ):
                raise MissingReferenceError(
                    "trigger observation names an unknown capability"
                )
            if not set(observation.exact_evidence_ids).issubset(known_evidence_ids):
                raise MissingReferenceError(
                    "trigger observation evidence leaves the packet closure"
                )
            if observation.configuration_fingerprint != self.configuration_fingerprint:
                raise ExpertTriggerError(
                    "trigger observation uses another configuration"
                )
            if (
                observation.inspection_operation.principal_id
                != self.trigger_observations[0].inspection_operation.principal_id
                or observation.inspection_operation.role
                != self.trigger_observations[0].inspection_operation.role
            ):
                raise ExpertTriggerError(
                    "trigger observations must share one inspection authority"
                )
            if (
                observation.kind
                is ExpertTriggerObservationKind.EXACT_SOURCE_DUPLICATION
            ):
                observed_owners: set[str] = set()
                for affected_path in observation.affected_paths:
                    path = PurePosixPath(affected_path)
                    owners = {
                        node.capability_id
                        for node in self.repository_map.capability_nodes
                        if any(
                            path == PurePosixPath(owned_path)
                            or PurePosixPath(owned_path) in path.parents
                            for owned_path in node.owned_paths
                        )
                    }
                    if len(owners) != 1:
                        raise ExpertTriggerError(
                            "duplicated path must have exactly one capability owner"
                        )
                    observed_owners.update(owners)
                if observed_owners != set(observation.affected_capability_ids):
                    raise ExpertTriggerError(
                        "duplication capabilities differ from affected path owners"
                    )
            if (
                observation.kind
                in {
                    ExpertTriggerObservationKind.ADAPTER_LEAKAGE,
                    ExpertTriggerObservationKind.CONTRACT_TOPOLOGY_MISMATCH,
                    ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
                    ExpertTriggerObservationKind.RELEASED_CAPABILITY_CONTRADICTION,
                    ExpertTriggerObservationKind.SEMANTIC_NAVIGATION_FAILURE,
                }
                and not observation.affected_capability_ids
            ):
                raise ExpertTriggerError(
                    "trigger observation must name an affected capability"
                )
            if (
                observation.kind
                is ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX
                and not observation.affected_paths
            ):
                raise ExpertTriggerError(
                    "mechanically general fix must name affected paths"
                )
            if (
                observation.kind
                is ExpertTriggerObservationKind.RELEASED_CAPABILITY_CONTRADICTION
            ):
                contradiction_claims = tuple(
                    claim
                    for claim in self.claims
                    if claim.revision_id in observation.exact_evidence_ids
                    and claim.contradicting_episode_ids
                )
                if not contradiction_claims:
                    raise ExpertTriggerError(
                        "released contradiction requires a contradicting claim"
                    )
            if (
                observation.kind
                is ExpertTriggerObservationKind.REPEATED_INDEPENDENT_DIFFICULTY
            ):
                evidence_episodes = tuple(
                    episodes_by_id[evidence_id]
                    for evidence_id in observation.exact_evidence_ids
                    if evidence_id in episodes_by_id
                )
                if len(evidence_episodes) != len(observation.exact_evidence_ids):
                    raise MissingReferenceError(
                        "repeated difficulty evidence must contain only episodes"
                    )
                if any(
                    not (
                        episode.attempts[
                            episode.terminal_attempt_revision
                        ].technical_difficulties
                        or episode.attempts[
                            episode.terminal_attempt_revision
                        ].execution_status
                        is ExecutionStatus.FAILED_TECHNICAL
                    )
                    for episode in evidence_episodes
                ):
                    raise ExpertTriggerError(
                        "repeated difficulty requires technical-failure evidence"
                    )
                expected_lineages = tuple(
                    sorted(
                        {
                            episode_lineage_id(episode, episodes_by_id)
                            for episode in evidence_episodes
                        }
                    )
                )
                expected_contexts = tuple(
                    sorted(
                        {
                            episode.task_context_binding.task_context_binding_id
                            for episode in evidence_episodes
                        }
                    )
                )
                if (
                    observation.independent_lineage_ids != expected_lineages
                    or observation.task_context_binding_ids != expected_contexts
                    or observation.occurrence_count != len(evidence_episodes)
                ):
                    raise ExpertTriggerError(
                        "repeated difficulty observation differs from episode closure"
                    )


class ExpertTriggerEvidencePacketBuilder:
    """Build an exhaustive trigger view from one verified knowledge snapshot."""

    def __init__(self, settings: ExpertTriggerSettings):
        self.settings = settings

    def build(
        self,
        *,
        knowledge_snapshot: KnowledgeSnapshotPackage,
        scope_contract: ExpertScopeContract,
        parent_scope_contract: ExpertScopeContract | None,
        parent_release: ExpertBaseReleaseManifest | None,
        parent_tree_receipt: ExpertParentTreeReceipt | None,
        parent_tree_hash: str,
        repository_map: ExpertRepositoryMap | None,
        module_contracts: tuple[ExpertModuleContract, ...],
        active_task_bindings: tuple[CrossRunTaskBindingSettings, ...],
        trigger_observations: tuple[ExpertTriggerObservation, ...] = (),
    ) -> ExpertTriggerEvidencePacket:
        knowledge_snapshot.verify()
        prepared = knowledge_snapshot.prepared
        episode_ids = set(prepared.admitted_episode_ids)
        pending = list(sorted(episode_ids))
        episodes_by_id: dict[str, TransferEpisode] = {}
        while pending:
            episode_id = pending.pop()
            envelope = knowledge_snapshot.record_by_id(episode_id)
            episode = parse_knowledge_record_payload(
                envelope["record_kind"],
                envelope["payload"],
            )
            if not isinstance(episode, TransferEpisode):
                raise ExpertTriggerError(
                    "knowledge snapshot admitted episode parsed incorrectly"
                )
            episodes_by_id[episode_id] = episode
            if (
                episode.parent_episode_ref is not None
                and episode.parent_episode_ref not in episode_ids
            ):
                episode_ids.add(episode.parent_episode_ref)
                pending.append(episode.parent_episode_ref)
        claims = []
        for claim_id in prepared.active_claim_revision_ids:
            envelope = knowledge_snapshot.record_by_id(claim_id)
            claim = parse_knowledge_record_payload(
                envelope["record_kind"],
                envelope["payload"],
            )
            if not isinstance(claim, KnowledgeClaim):
                raise ExpertTriggerError(
                    "knowledge snapshot active claim parsed incorrectly"
                )
            claims.append(claim)
        scientific_ids = {
            *episode_ids,
            *(claim.revision_id for claim in claims),
            prepared.scope_contract.scope_contract_id,
        }
        proof_closure = set(scientific_ids)
        proof_pending = list(sorted(scientific_ids))
        while proof_pending:
            record_id = proof_pending.pop()
            for dependency_id in prepared.proof_dependencies.get(record_id, ()):
                if dependency_id not in proof_closure:
                    proof_closure.add(dependency_id)
                    proof_pending.append(dependency_id)
        configuration_fingerprint = tree_or_blob_digest(
            canonical_json_bytes(self.settings.to_dict())
        )
        return ExpertTriggerEvidencePacket.mint(
            knowledge_snapshot_manifest=knowledge_snapshot.manifest,
            knowledge_record_closure_digest=prepared.record_closure_digest,
            configuration_fingerprint=configuration_fingerprint,
            scope_contract=scope_contract,
            parent_scope_contract=parent_scope_contract,
            parent_release=parent_release,
            parent_tree_receipt=parent_tree_receipt,
            parent_tree_hash=parent_tree_hash,
            repository_map=repository_map,
            module_contracts=tuple(
                sorted(
                    module_contracts,
                    key=lambda contract: contract.module_contract_id,
                )
            ),
            episodes=tuple(
                episodes_by_id[episode_id] for episode_id in sorted(episode_ids)
            ),
            claims=tuple(sorted(claims, key=lambda claim: claim.revision_id)),
            trigger_observations=tuple(
                sorted(
                    trigger_observations,
                    key=lambda observation: observation.observation_id,
                )
            ),
            active_task_bindings=tuple(
                sorted(
                    active_task_bindings,
                    key=lambda binding: (
                        binding.task_family_id,
                        binding.task_adapter_id,
                    ),
                )
            ),
            proof_reference_ids=tuple(sorted(proof_closure - scientific_ids)),
        )


@dataclass(frozen=True)
class ExpertEvolutionTriggerDecision(StrictContract):
    """Deterministic decision to skip or propose exactly one candidate class."""

    trigger_decision_id: str
    evidence_packet_id: str
    knowledge_snapshot_id: str
    policy_version: str
    configuration_fingerprint: str
    candidate_required: bool
    change_kind: CandidateChangeKind | None
    reason_code: str
    trigger_evidence_ids: tuple[str, ...]
    independent_lineage_ids: tuple[str, ...]
    task_context_binding_ids: tuple[str, ...]
    rationale: str

    CONTENT_NAMESPACE = "expert-trigger-decision"
    IDENTITY_FIELD = "trigger_decision_id"

    def _validate(self) -> None:
        require_content_id(self.evidence_packet_id, "evidence_packet_id")
        require_content_id(self.knowledge_snapshot_id, "knowledge_snapshot_id")
        require_identifier(self.policy_version, "trigger policy_version")
        _require_digest(
            self.configuration_fingerprint,
            "trigger configuration fingerprint",
        )
        require_identifier(self.reason_code, "trigger reason_code")
        if self.candidate_required != (self.change_kind is not None):
            raise ExpertTriggerError("candidate requirement and change kind must agree")
        _require_sorted_content_ids(
            self.trigger_evidence_ids,
            "trigger evidence IDs",
            required=True,
        )
        for values, name in (
            (self.independent_lineage_ids, "independent_lineage_ids"),
            (self.task_context_binding_ids, "task_context_binding_ids"),
        ):
            if values != tuple(sorted(set(values))):
                raise ExpertTriggerError(f"{name} must be sorted and unique")
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ExpertTriggerError(f"{name} must contain non-empty text")
        if not self.rationale.strip():
            raise ExpertTriggerError("trigger rationale must not be empty")


@dataclass(frozen=True)
class _ExpertTriggerCommitRecord(StrictContract):
    """Atomic visibility marker for one immutable packet/decision pair."""

    commit_record_id: str
    evidence_packet_id: str
    trigger_decision_id: str
    packet_digest: str
    decision_digest: str

    CONTENT_NAMESPACE = "expert-trigger-commit"
    IDENTITY_FIELD = "commit_record_id"

    def _validate(self) -> None:
        require_content_id(self.evidence_packet_id, "commit evidence_packet_id")
        require_content_id(self.trigger_decision_id, "commit trigger_decision_id")
        _require_digest(self.packet_digest, "commit packet digest")
        _require_digest(self.decision_digest, "commit decision digest")


@dataclass(frozen=True)
class _EligibleTrigger:
    change_kind: CandidateChangeKind
    reason_code: str
    evidence_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    rationale: str


class ExpertTriggerEvaluator:
    """Calculate one canonical trigger without model judgment."""

    def __init__(self, settings: ExpertTriggerSettings):
        self.settings = settings

    def evaluate(
        self,
        packet: ExpertTriggerEvidencePacket,
    ) -> ExpertEvolutionTriggerDecision:
        expected_fingerprint = tree_or_blob_digest(
            canonical_json_bytes(self.settings.to_dict())
        )
        if packet.configuration_fingerprint != expected_fingerprint:
            raise ExpertTriggerError(
                "trigger packet configuration differs from the evaluator"
            )
        for observation in packet.trigger_observations:
            operation = observation.inspection_operation
            if (
                observation.inspection_policy_version
                != self.settings.inspection_policy_version
                or operation.principal_id != self.settings.inspector_id
                or operation.role != self.settings.inspector_role
            ):
                raise ExpertTriggerError(
                    "trigger observation lacks configured inspection authority"
                )
        eligible = self._bootstrap(packet)
        if eligible is None:
            eligible = self._observation_trigger(packet)
        if eligible is None:
            eligible = self._scope_expansion(packet)
        if eligible is None:
            eligible = self._new_task_family(packet)
        if eligible is None:
            eligible = self._repeated_success(packet)
        if eligible is None:
            eligible = _EligibleTrigger(
                change_kind=CandidateChangeKind.CAPABILITY,
                reason_code="insufficient_evidence",
                evidence_ids=(packet.knowledge_snapshot_id,),
                lineage_ids=(),
                context_ids=(),
                rationale=(
                    "No configured independent capability or architecture trigger "
                    "threshold is satisfied."
                ),
            )
            candidate_required = False
            change_kind = None
        else:
            candidate_required = True
            change_kind = eligible.change_kind
        return ExpertEvolutionTriggerDecision.mint(
            evidence_packet_id=packet.evidence_packet_id,
            knowledge_snapshot_id=packet.knowledge_snapshot_id,
            policy_version=self.settings.policy_version,
            configuration_fingerprint=packet.configuration_fingerprint,
            candidate_required=candidate_required,
            change_kind=change_kind,
            reason_code=eligible.reason_code,
            trigger_evidence_ids=eligible.evidence_ids,
            independent_lineage_ids=eligible.lineage_ids,
            task_context_binding_ids=eligible.context_ids,
            rationale=eligible.rationale,
        )

    @staticmethod
    def _bootstrap(
        packet: ExpertTriggerEvidencePacket,
    ) -> _EligibleTrigger | None:
        if packet.parent_release_id is not None:
            return None
        return _EligibleTrigger(
            change_kind=CandidateChangeKind.REPOSITORY_ARCHITECTURE,
            reason_code="empty_scope_bootstrap",
            evidence_ids=tuple(
                sorted(
                    {
                        packet.scope_contract.scope_contract_id,
                        packet.knowledge_snapshot_id,
                    }
                )
            ),
            lineage_ids=(),
            context_ids=(),
            rationale=(
                "The scope has no expert release; propose the smallest architecture "
                "covering only admitted task families."
            ),
        )

    def _observation_trigger(
        self,
        packet: ExpertTriggerEvidencePacket,
    ) -> _EligibleTrigger | None:
        episodes_by_id = {episode.episode_id: episode for episode in packet.episodes}
        observations = tuple(
            observation
            for observation in packet.trigger_observations
            if observation.kind.value in _CAPABILITY_OBSERVATION_KINDS
            or (
                observation.kind.value in _ARCHITECTURE_OBSERVATION_KINDS
                and (
                    observation.kind
                    is not ExpertTriggerObservationKind.EXACT_SOURCE_DUPLICATION
                    or observation.occurrence_count
                    >= self.settings.minimum_duplicate_files
                )
            )
            if (
                observation.kind
                is not ExpertTriggerObservationKind.REPEATED_INDEPENDENT_DIFFICULTY
                or (
                    len(observation.independent_lineage_ids)
                    >= self.settings.minimum_failure_lineages
                    and len(
                        {
                            _transfer_context_signature(episodes_by_id[evidence_id])
                            for evidence_id in observation.exact_evidence_ids
                        }
                    )
                    >= self.settings.minimum_failure_contexts
                )
            )
        )
        if not observations:
            return None
        priority = {
            ExpertTriggerObservationKind.RELEASED_CAPABILITY_CONTRADICTION: 0,
            ExpertTriggerObservationKind.ADAPTER_LEAKAGE: 1,
            ExpertTriggerObservationKind.CONTRACT_TOPOLOGY_MISMATCH: 2,
            ExpertTriggerObservationKind.EXACT_SOURCE_DUPLICATION: 3,
            ExpertTriggerObservationKind.SEMANTIC_NAVIGATION_FAILURE: 4,
            ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX: 5,
            ExpertTriggerObservationKind.REPEATED_INDEPENDENT_DIFFICULTY: 6,
        }
        observation = sorted(
            observations,
            key=lambda item: (priority[item.kind], item.observation_id),
        )[0]
        change_kind = (
            CandidateChangeKind.REPOSITORY_ARCHITECTURE
            if observation.kind.value in _ARCHITECTURE_OBSERVATION_KINDS
            else CandidateChangeKind.CAPABILITY
        )
        return _EligibleTrigger(
            change_kind=change_kind,
            reason_code=observation.kind.value,
            evidence_ids=tuple(
                sorted({observation.observation_id, *observation.exact_evidence_ids})
            ),
            lineage_ids=observation.independent_lineage_ids,
            context_ids=observation.task_context_binding_ids,
            rationale=observation.description,
        )

    @staticmethod
    def _scope_expansion(
        packet: ExpertTriggerEvidencePacket,
    ) -> _EligibleTrigger | None:
        parent = packet.parent_scope_contract
        if (
            parent is None
            or parent.scope_contract_id == packet.scope_contract.scope_contract_id
        ):
            return None
        parent_families = {
            family.task_family_id for family in parent.task_family_ontology
        }
        active_added_families = tuple(
            sorted(set(packet.active_task_family_ids) - parent_families)
        )
        added_artifact_classes = tuple(
            sorted(
                set(packet.scope_contract.artifact_classes)
                - set(parent.artifact_classes)
            )
        )
        if not active_added_families and not added_artifact_classes:
            return None
        additions = tuple(
            (
                *(f"task-family:{family}" for family in active_added_families),
                *(f"artifact-class:{kind}" for kind in added_artifact_classes),
            )
        )
        return _EligibleTrigger(
            change_kind=CandidateChangeKind.REPOSITORY_ARCHITECTURE,
            reason_code="attested_scope_expansion",
            evidence_ids=tuple(
                sorted(
                    {
                        parent.scope_contract_id,
                        packet.scope_contract.scope_contract_id,
                        packet.knowledge_snapshot_id,
                    }
                )
            ),
            lineage_ids=(),
            context_ids=(),
            rationale=(
                "The attested successor scope activates repository additions: "
                f"{', '.join(additions)}."
            ),
        )

    @staticmethod
    def _new_task_family(
        packet: ExpertTriggerEvidencePacket,
    ) -> _EligibleTrigger | None:
        if packet.repository_map is None:
            return None
        bound = {
            family_id
            for node in packet.repository_map.capability_nodes
            for family_id in node.task_family_bindings
        }
        uncovered = tuple(sorted(set(packet.active_task_family_ids) - bound))
        if not uncovered:
            return None
        family_episodes = tuple(
            episode
            for episode in packet.episodes
            if episode.task_context_binding.task_family_id in uncovered
        )
        return _EligibleTrigger(
            change_kind=CandidateChangeKind.REPOSITORY_ARCHITECTURE,
            reason_code="admitted_task_family_uncovered",
            evidence_ids=tuple(
                sorted(
                    {
                        packet.scope_contract.scope_contract_id,
                        packet.knowledge_snapshot_id,
                        *(episode.episode_id for episode in family_episodes),
                    }
                )
            ),
            lineage_ids=(),
            context_ids=tuple(
                sorted(
                    {
                        episode.task_context_binding.task_context_binding_id
                        for episode in family_episodes
                    }
                )
            ),
            rationale=(
                "The admitted task family set contains no matching capability "
                f"boundary: {', '.join(uncovered)}."
            ),
        )

    def _repeated_success(
        self,
        packet: ExpertTriggerEvidencePacket,
    ) -> _EligibleTrigger | None:
        episodes_by_id = {episode.episode_id: episode for episode in packet.episodes}
        eligible = []
        for claim in packet.claims:
            successful = tuple(
                episodes_by_id[episode_id]
                for episode_id in claim.supporting_episode_ids
                if self._is_positive(episodes_by_id[episode_id])
            )
            lineages = {
                episode_lineage_id(episode, episodes_by_id) for episode in successful
            }
            context_signatures = {
                _transfer_context_signature(episode) for episode in successful
            }
            context_ids = {
                episode.task_context_binding.task_context_binding_id
                for episode in successful
            }
            if (
                len(lineages) >= self.settings.minimum_success_lineages
                and len(context_signatures) >= self.settings.minimum_success_contexts
            ):
                eligible.append(
                    (
                        claim,
                        successful,
                        tuple(sorted(lineages)),
                        tuple(sorted(context_ids)),
                        len(context_signatures),
                    )
                )
        if not eligible:
            return None
        claim, episodes, lineages, contexts, context_count = sorted(
            eligible,
            key=lambda item: (-len(item[2]), -item[4], item[0].revision_id),
        )[0]
        return _EligibleTrigger(
            change_kind=CandidateChangeKind.CAPABILITY,
            reason_code="repeated_cross_context_success",
            evidence_ids=tuple(
                sorted(
                    {claim.revision_id, *(episode.episode_id for episode in episodes)}
                )
            ),
            lineage_ids=lineages,
            context_ids=contexts,
            rationale=(
                "An admitted mechanism has positive comparable support across "
                f"{len(lineages)} independent lineages and {context_count} contexts."
            ),
        )

    @staticmethod
    def _is_positive(episode: TransferEpisode) -> bool:
        terminal = episode.attempts[episode.terminal_attempt_revision]
        return (
            terminal.execution_status is ExecutionStatus.COMPLETED
            and terminal.evaluation_status is EpisodeEvaluationStatus.VALID
            and terminal.comparison_status is ComparisonStatus.COMPARABLE
            and terminal.source_parent_effect is not None
            and terminal.source_parent_effect.normalized_delta > 0
        )


class ExpertTriggerDecisionStore:
    """Create-only packet and decision store with exact replay semantics."""

    def __init__(
        self,
        root: Path,
        state_root: Path,
        settings: ExpertTriggerSettings,
    ):
        self._validate_root(state_root, "expert trigger state root")
        if (
            not root.is_absolute()
            or ".." in root.parts
            or root != Path(os.path.abspath(root))
            or root.parent != state_root
        ):
            raise ExpertTriggerError(
                "expert trigger store must be a direct child of its state root"
            )
        self.root = root
        self.state_root = state_root
        self.settings = settings
        self.packet_root = root / "packets"
        self.decision_root = root / "decisions"
        self.commit_root = root / "commits"
        self._prepare_layout()

    def persist(
        self,
        packet: ExpertTriggerEvidencePacket,
        decision: ExpertEvolutionTriggerDecision,
    ) -> None:
        if decision != ExpertTriggerEvaluator(self.settings).evaluate(packet):
            raise ExpertTriggerError(
                "trigger decision differs from deterministic evaluation"
            )
        if (
            decision.evidence_packet_id != packet.evidence_packet_id
            or decision.knowledge_snapshot_id != packet.knowledge_snapshot_id
            or decision.configuration_fingerprint != packet.configuration_fingerprint
        ):
            raise ExpertTriggerError("trigger decision references another packet")
        packet_known_ids = {
            packet.scope_contract.scope_contract_id,
            packet.knowledge_snapshot_id,
            *(module.module_contract_id for module in packet.module_contracts),
            *(episode.episode_id for episode in packet.episodes),
            *(claim.revision_id for claim in packet.claims),
            *(
                observation.observation_id
                for observation in packet.trigger_observations
            ),
            *(
                observation.inspection_operation.operation_receipt_id
                for observation in packet.trigger_observations
            ),
            *packet.proof_reference_ids,
        }
        if packet.parent_release_id is not None:
            packet_known_ids.add(packet.parent_release_id)
        if packet.repository_map is not None:
            packet_known_ids.add(packet.repository_map.repository_map_id)
        if packet.parent_scope_contract is not None:
            packet_known_ids.add(packet.parent_scope_contract.scope_contract_id)
        if packet.parent_tree_receipt is not None:
            packet_known_ids.add(packet.parent_tree_receipt.parent_tree_receipt_id)
        if not set(decision.trigger_evidence_ids).issubset(packet_known_ids):
            raise MissingReferenceError(
                "trigger decision evidence leaves its persisted packet"
            )
        packet_bytes = packet.to_json_bytes()
        decision_bytes = decision.to_json_bytes()
        commit_record = _ExpertTriggerCommitRecord.mint(
            evidence_packet_id=packet.evidence_packet_id,
            trigger_decision_id=decision.trigger_decision_id,
            packet_digest=tree_or_blob_digest(packet_bytes),
            decision_digest=tree_or_blob_digest(decision_bytes),
        )
        with ExitStack() as stack:
            root_descriptor = self._open_directory(self.root)
            stack.callback(os.close, root_descriptor)
            lock_descriptor = os.open(
                "trigger.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=root_descriptor,
            )
            lock = stack.enter_context(os.fdopen(lock_descriptor, "r+b"))
            self._validate_private_regular(
                os.fstat(lock.fileno()),
                "expert trigger lock",
            )
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            packet_descriptor = self._open_child_directory(
                root_descriptor,
                "packets",
            )
            stack.callback(os.close, packet_descriptor)
            decision_descriptor = self._open_child_directory(
                root_descriptor,
                "decisions",
            )
            stack.callback(os.close, decision_descriptor)
            commit_descriptor = self._open_child_directory(
                root_descriptor,
                "commits",
            )
            stack.callback(os.close, commit_descriptor)
            self._write_once_at(
                packet_descriptor,
                self._filename(packet.evidence_packet_id),
                packet_bytes,
            )
            self._write_once_at(
                decision_descriptor,
                self._filename(packet.evidence_packet_id),
                decision_bytes,
            )
            os.fsync(packet_descriptor)
            os.fsync(decision_descriptor)
            self._write_once_at(
                commit_descriptor,
                self._filename(packet.evidence_packet_id),
                commit_record.to_json_bytes(),
            )
            os.fsync(commit_descriptor)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def read_packet(self, evidence_packet_id: str) -> ExpertTriggerEvidencePacket:
        require_content_id(evidence_packet_id, "evidence_packet_id")
        packet, _ = self._read_committed_pair(evidence_packet_id)
        return packet

    def read_decision_for_packet(
        self,
        evidence_packet_id: str,
    ) -> ExpertEvolutionTriggerDecision:
        require_content_id(evidence_packet_id, "evidence_packet_id")
        packet, decision = self._read_committed_pair(evidence_packet_id)
        if decision != ExpertTriggerEvaluator(self.settings).evaluate(packet):
            raise ExpertTriggerError(
                "stored trigger decision differs from deterministic evaluation"
            )
        return decision

    def _read_committed_pair(
        self,
        evidence_packet_id: str,
    ) -> tuple[ExpertTriggerEvidencePacket, ExpertEvolutionTriggerDecision]:
        with ExitStack() as stack:
            root_descriptor = self._open_directory(self.root)
            stack.callback(os.close, root_descriptor)
            commit_descriptor = self._open_child_directory(
                root_descriptor,
                "commits",
            )
            stack.callback(os.close, commit_descriptor)
            packet_descriptor = self._open_child_directory(
                root_descriptor,
                "packets",
            )
            stack.callback(os.close, packet_descriptor)
            decision_descriptor = self._open_child_directory(
                root_descriptor,
                "decisions",
            )
            stack.callback(os.close, decision_descriptor)
            filename = self._filename(evidence_packet_id)
            first_commit_payload = self._read_regular_at(
                commit_descriptor,
                filename,
            )
            packet_payload = self._read_regular_at(
                packet_descriptor,
                filename,
            )
            decision_payload = self._read_regular_at(
                decision_descriptor,
                filename,
            )
            second_commit_payload = self._read_regular_at(
                commit_descriptor,
                filename,
            )
        if first_commit_payload != second_commit_payload:
            raise ExpertTriggerError("trigger commit changed during read")
        commit_record = _ExpertTriggerCommitRecord.from_json_bytes(first_commit_payload)
        packet = ExpertTriggerEvidencePacket.from_json_bytes(packet_payload)
        decision = ExpertEvolutionTriggerDecision.from_json_bytes(decision_payload)
        if (
            commit_record.evidence_packet_id != evidence_packet_id
            or packet.evidence_packet_id != evidence_packet_id
            or decision.evidence_packet_id != evidence_packet_id
            or commit_record.trigger_decision_id != decision.trigger_decision_id
            or commit_record.packet_digest != tree_or_blob_digest(packet_payload)
            or commit_record.decision_digest != tree_or_blob_digest(decision_payload)
        ):
            raise ExpertTriggerError(
                "stored trigger pair differs from its atomic commit"
            )
        return packet, decision

    def _prepare_layout(self) -> None:
        if os.path.lexists(self.root):
            root_descriptor = self._open_directory(self.root)
            os.close(root_descriptor)
        else:
            os.mkdir(self.root, mode=0o700)
            self._fsync_directory(self.state_root)
        with ExitStack() as stack:
            root_descriptor = self._open_directory(self.root)
            stack.callback(os.close, root_descriptor)
            existing = set(os.listdir(root_descriptor))
            for name in ("packets", "decisions", "commits"):
                if name not in existing:
                    os.mkdir(name, mode=0o700, dir_fd=root_descriptor)
                child_descriptor = self._open_child_directory(
                    root_descriptor,
                    name,
                )
                os.close(child_descriptor)
            os.fsync(root_descriptor)

    @staticmethod
    def _validate_root(path: Path, name: str) -> None:
        if (
            not path.is_absolute()
            or path != Path(os.path.abspath(path))
            or path in {Path("/"), Path.home(), Path.cwd()}
            or path.is_symlink()
            or not path.is_dir()
            or path.resolve() != path
        ):
            raise ExpertTriggerError(f"{name} must be an authorized real directory")

    @classmethod
    def _open_directory(cls, path: Path) -> int:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        cls._validate_private_directory(
            os.fstat(descriptor),
            "expert trigger directory",
        )
        return descriptor

    @classmethod
    def _open_child_directory(cls, parent_descriptor: int, name: str) -> int:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        cls._validate_private_directory(
            os.fstat(descriptor),
            f"expert trigger {name} directory",
        )
        return descriptor

    @staticmethod
    def _validate_private_directory(metadata: os.stat_result, name: str) -> None:
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ExpertTriggerError(f"{name} must be a private directory")

    @staticmethod
    def _validate_private_regular(metadata: os.stat_result, name: str) -> None:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
        ):
            raise ExpertTriggerError(f"{name} is not a private regular file")

    @staticmethod
    def _filename(content_identifier: str) -> str:
        return content_identifier.rsplit(":", 1)[1] + ".json"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.fsync(descriptor)
        os.close(descriptor)

    @staticmethod
    def _read_regular_at(directory_descriptor: int, filename: str) -> bytes:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            ExpertTriggerDecisionStore._validate_private_regular(
                metadata,
                "expert trigger object",
            )
            return handle.read()

    @classmethod
    def _write_once_at(
        cls,
        directory_descriptor: int,
        filename: str,
        payload: bytes,
    ) -> None:
        if filename in os.listdir(directory_descriptor):
            if cls._read_regular_at(directory_descriptor, filename) != payload:
                raise ExpertTriggerError(
                    "expert trigger object conflicts with persisted bytes"
                )
            return
        temporary_filename = f".{filename}.{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary_filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        cls._rename_no_replace(
            directory_descriptor,
            temporary_filename,
            filename,
        )

    @staticmethod
    def _rename_no_replace(
        directory_descriptor: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(libc, "renameat2"):
            raise ExpertTriggerError(
                "atomic no-replace trigger publication is unavailable"
            )
        rename_at2 = libc.renameat2
        rename_at2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_at2.restype = ctypes.c_int
        result = rename_at2(
            directory_descriptor,
            os.fsencode(source_name),
            directory_descriptor,
            os.fsencode(destination_name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                f"trigger publication failed: {errno.errorcode.get(error_number)}",
            )
