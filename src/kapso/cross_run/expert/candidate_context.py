"""Origin-neutral scientific and validation context for expert candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
)
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    CrossRunTaskBindingSettings,
    ExpertBaseReleaseManifest,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
    KnowledgeClaim,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    StrictContract,
    TransferEpisode,
)
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertParentTreeReceipt,
    ExpertTriggerEvidencePacket,
    episode_lineage_id,
)


class ExpertCandidateContextError(ValueError):
    """A candidate scientific context is incomplete or contradictory."""


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    required: bool = False,
) -> None:
    if (required and not values) or values != tuple(sorted(set(values))):
        raise ExpertCandidateContextError(f"{name} must be sorted and unique")
    for value in values:
        require_content_id(value, name)


@dataclass(frozen=True)
class ExpertCandidateReplayEvidence(StrictContract):
    """Stable episode basis from which active validation policy selects replays."""

    replay_evidence_id: str
    knowledge_snapshot_manifests: tuple[KnowledgeSnapshotManifest, ...]
    scope_contracts: tuple[ExpertScopeContract, ...]
    episodes: tuple[TransferEpisode, ...]
    causal_episode_ids: tuple[str, ...]
    causal_episode_reason_codes: Mapping[str, tuple[str, ...]]
    evidence_authority_ids: tuple[str, ...]
    proof_reference_ids: tuple[str, ...]
    stable_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-replay-evidence"
    IDENTITY_FIELD: ClassVar[str] = "replay_evidence_id"

    def _validate(self) -> None:
        if (
            type(self.knowledge_snapshot_manifests) is not tuple
            or any(
                type(manifest) is not KnowledgeSnapshotManifest
                for manifest in self.knowledge_snapshot_manifests
            )
            or type(self.scope_contracts) is not tuple
            or any(
                type(scope_contract) is not ExpertScopeContract
                for scope_contract in self.scope_contracts
            )
            or type(self.episodes) is not tuple
            or any(type(episode) is not TransferEpisode for episode in self.episodes)
        ):
            raise ExpertCandidateContextError(
                "candidate replay evidence requires exact typed records"
            )
        snapshot_ids = tuple(
            manifest.snapshot_id for manifest in self.knowledge_snapshot_manifests
        )
        scope_contract_ids = tuple(
            scope_contract.scope_contract_id for scope_contract in self.scope_contracts
        )
        episode_ids = tuple(episode.episode_id for episode in self.episodes)
        if (
            not snapshot_ids
            or snapshot_ids != tuple(sorted(set(snapshot_ids)))
            or not scope_contract_ids
            or scope_contract_ids != tuple(sorted(set(scope_contract_ids)))
            or episode_ids != tuple(sorted(set(episode_ids)))
        ):
            raise ExpertCandidateContextError(
                "candidate replay snapshots and episodes must be canonical"
            )
        scopes_by_id = {
            scope_contract.scope_contract_id: scope_contract
            for scope_contract in self.scope_contracts
        }
        if set(scope_contract_ids) != {
            manifest.scope_contract_id for manifest in self.knowledge_snapshot_manifests
        } or any(
            manifest.scope_id != scopes_by_id[manifest.scope_contract_id].scope_id
            for manifest in self.knowledge_snapshot_manifests
        ):
            raise ExpertCandidateContextError(
                "candidate replay snapshots differ from their exact scopes"
            )
        _require_sorted_content_ids(
            self.causal_episode_ids,
            "candidate causal episodes",
        )
        if not set(self.causal_episode_ids).issubset(episode_ids) or set(
            self.causal_episode_reason_codes
        ) != set(self.causal_episode_ids):
            raise ExpertCandidateContextError(
                "candidate causal episode reasons differ from the episode closure"
            )
        for reasons in self.causal_episode_reason_codes.values():
            if not reasons or reasons != tuple(sorted(set(reasons))):
                raise ExpertCandidateContextError(
                    "candidate causal episode reasons must be canonical"
                )
            for reason in reasons:
                require_identifier(reason, "candidate causal episode reason")
        _require_sorted_content_ids(
            self.evidence_authority_ids,
            "candidate replay evidence authorities",
            required=True,
        )
        if not set(snapshot_ids).issubset(self.evidence_authority_ids):
            raise ExpertCandidateContextError(
                "candidate replay authorities omit a knowledge snapshot"
            )
        _require_sorted_content_ids(
            self.proof_reference_ids,
            "candidate replay proof references",
        )
        _require_sorted_content_ids(
            self.stable_dependency_ids,
            "candidate replay stable dependencies",
            required=True,
        )
        expected_dependencies = {
            *snapshot_ids,
            *scope_contract_ids,
            *episode_ids,
            *self.evidence_authority_ids,
            *self.proof_reference_ids,
            *(episode.source_bundle_id for episode in self.episodes),
        }
        if set(self.stable_dependency_ids) != expected_dependencies:
            raise ExpertCandidateContextError(
                "candidate replay stable dependency closure is not exact"
            )
        included_bundle_ids = {
            bundle_id
            for manifest in self.knowledge_snapshot_manifests
            for bundle_id in manifest.included_bundle_ids
        }
        episode_bundle_ids = {episode.source_bundle_id for episode in self.episodes}
        if not episode_bundle_ids.issubset(included_bundle_ids) or not (
            episode_bundle_ids.issubset(self.proof_reference_ids)
        ):
            raise ExpertCandidateContextError(
                "candidate replay bundles leave snapshot proof authority"
            )
        episodes_by_id = {episode.episode_id: episode for episode in self.episodes}
        admitted_lineage_ids: set[str] = set()
        for manifest in self.knowledge_snapshot_manifests:
            manifest_lineage_ids = set(manifest.admitted_episode_ids)
            lineage_pending = list(sorted(manifest.admitted_episode_ids))
            while lineage_pending:
                episode_id = lineage_pending.pop()
                episode = episodes_by_id.get(episode_id)
                if episode is None:
                    raise ExpertCandidateContextError(
                        "candidate replay omits an admitted snapshot episode"
                    )
                if (
                    episode.parent_episode_ref is not None
                    and episode.parent_episode_ref not in manifest_lineage_ids
                ):
                    manifest_lineage_ids.add(episode.parent_episode_ref)
                    lineage_pending.append(episode.parent_episode_ref)
            for episode_id in manifest_lineage_ids:
                episode = episodes_by_id[episode_id]
                required_manifest_proof = {
                    episode.episode_id,
                    episode.source_bundle_id,
                    episode.sanitation_report_id,
                    *episode.derivation_refs,
                }
                if episode.parent_episode_ref is not None:
                    required_manifest_proof.add(episode.parent_episode_ref)
                if episode.supersedes_projection_id is not None:
                    required_manifest_proof.add(episode.supersedes_projection_id)
                if (
                    episode.task_context_binding.scope_contract_id
                    != manifest.scope_contract_id
                    or episode.source_bundle_id not in manifest.included_bundle_ids
                    or not required_manifest_proof.issubset(
                        manifest.proof_dependency_closure_ids
                    )
                ):
                    raise ExpertCandidateContextError(
                        "candidate replay snapshot does not independently authorize "
                        "its admitted episode lineage"
                    )
            admitted_lineage_ids.update(manifest_lineage_ids)
        if admitted_lineage_ids != set(episode_ids):
            raise ExpertCandidateContextError(
                "candidate replay episodes differ from admitted snapshot lineage"
            )
        for episode in self.episodes:
            binding = episode.task_context_binding
            scope_contract = scopes_by_id.get(binding.scope_contract_id)
            if scope_contract is None:
                raise ExpertCandidateContextError(
                    "candidate replay episode leaves its snapshot scopes"
                )
            binding.validate_against(scope_contract)
            episode_lineage_id(episode, episodes_by_id)
            required_proof = {
                episode.source_bundle_id,
                episode.sanitation_report_id,
                *episode.derivation_refs,
            }
            if episode.supersedes_projection_id is not None:
                required_proof.add(episode.supersedes_projection_id)
            if not required_proof.issubset(self.proof_reference_ids):
                raise ExpertCandidateContextError(
                    "candidate replay omits episode proof references"
                )


@dataclass(frozen=True)
class ExpertCandidateValidationContext(StrictContract):
    """Stable origin-neutral context consumed by the validation cascade."""

    validation_context_id: str
    scope_contract: ExpertScopeContract
    parent_scope_contract: ExpertScopeContract | None
    parent_release: ExpertBaseReleaseManifest | None
    parent_tree_receipt: ExpertParentTreeReceipt | None
    parent_tree_hash: str
    parent_repository_map: ExpertRepositoryMap | None
    parent_module_contracts: tuple[ExpertModuleContract, ...]
    active_task_bindings: tuple[CrossRunTaskBindingSettings, ...]
    replay_evidence: ExpertCandidateReplayEvidence
    stable_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-validation-context"
    IDENTITY_FIELD: ClassVar[str] = "validation_context_id"

    def _validate(self) -> None:
        if (
            type(self.scope_contract) is not ExpertScopeContract
            or type(self.parent_module_contracts) is not tuple
            or any(
                type(module) is not ExpertModuleContract
                for module in self.parent_module_contracts
            )
            or type(self.active_task_bindings) is not tuple
            or any(
                type(binding) is not CrossRunTaskBindingSettings
                for binding in self.active_task_bindings
            )
            or type(self.replay_evidence) is not ExpertCandidateReplayEvidence
        ):
            raise ExpertCandidateContextError(
                "candidate validation context requires exact typed records"
            )
        if self.parent_release is None:
            if (
                self.parent_tree_hash != EMPTY_EXPERT_TREE_DIGEST
                or self.parent_scope_contract is not None
                or self.parent_tree_receipt is not None
                or self.parent_repository_map is not None
                or self.parent_module_contracts
            ):
                raise ExpertCandidateContextError(
                    "bootstrap validation context must have an explicit empty parent"
                )
        elif (
            type(self.parent_release) is not ExpertBaseReleaseManifest
            or type(self.parent_scope_contract) is not ExpertScopeContract
            or type(self.parent_tree_receipt) is not ExpertParentTreeReceipt
            or type(self.parent_repository_map) is not ExpertRepositoryMap
            or self.parent_release.scope_contract_id
            != self.parent_scope_contract.scope_contract_id
            or self.parent_release.scope_id != self.parent_scope_contract.scope_id
            or self.parent_scope_contract.scope_id != self.scope_contract.scope_id
            or self.parent_release.release_id != self.parent_tree_receipt.release_id
            or self.parent_release.repository_map_ref
            != self.parent_repository_map.repository_map_id
            or self.parent_tree_hash != self.parent_tree_receipt.parent_tree_hash
            or self.parent_repository_map.scope_contract_id
            != self.parent_scope_contract.scope_contract_id
            or dict(self.parent_release.module_versions)
            != {
                module.module_id: module.version
                for module in self.parent_module_contracts
            }
        ):
            raise ExpertCandidateContextError(
                "released validation context parent authority is inconsistent"
            )
        if self.parent_release is not None:
            if self.parent_tree_hash == EMPTY_EXPERT_TREE_DIGEST:
                raise ExpertCandidateContextError(
                    "released validation context cannot use the empty tree"
                )
            if (
                self.scope_contract.scope_contract_id
                != self.parent_scope_contract.scope_contract_id
                and self.scope_contract.supersedes_scope_contract_id
                != self.parent_scope_contract.scope_contract_id
            ):
                raise ExpertCandidateContextError(
                    "candidate scope must equal or directly supersede its parent"
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
                ) or any(
                    not set(edge.source_ids).issubset(parent_dimension_ids)
                    for edge in self.scope_contract.context_dimension_lineage
                ):
                    raise ExpertCandidateContextError(
                        "candidate scope lineage leaves its direct parent"
                    )
            module_contract_ids = tuple(
                sorted(
                    module.module_contract_id for module in self.parent_module_contracts
                )
            )
            map_contract_ids = tuple(
                sorted(
                    node.module_contract_ref
                    for node in self.parent_repository_map.capability_nodes
                )
            )
            receipt = self.parent_tree_receipt
            cache_receipt = receipt.cache_verification_receipt
            extraction_receipt = receipt.source_extraction_receipt
            archive_ref = self.parent_release.source_archive_ref
            archive_digest = self.parent_release.checksums[archive_ref]
            if (
                map_contract_ids != module_contract_ids
                or receipt.repository_map_id
                != self.parent_repository_map.repository_map_id
                or receipt.module_contract_ids != module_contract_ids
                or cache_receipt.artifact_kind
                is not PublicationArtifactKind.EXPERT_BASE_RELEASE
                or cache_receipt.artifact_id != self.parent_release.release_id
                or cache_receipt.asset_digests.get(archive_ref) != archive_digest
                or extraction_receipt.artifact_id != self.parent_release.release_id
                or extraction_receipt.source_archive_ref != archive_ref
                or extraction_receipt.source_archive_digest != archive_digest
                or extraction_receipt.source_tree_hash != self.parent_tree_hash
            ):
                raise ExpertCandidateContextError(
                    "candidate parent release closure is inconsistent"
                )
        binding_keys = tuple(
            (binding.task_family_id, binding.task_adapter_id)
            for binding in self.active_task_bindings
        )
        if not binding_keys or binding_keys != tuple(sorted(set(binding_keys))):
            raise ExpertCandidateContextError(
                "candidate active task bindings must be canonical and non-empty"
            )
        for binding in self.active_task_bindings:
            self.scope_contract.validate_binding(binding)
        allowed_replay_scope_contract_ids = {self.scope_contract.scope_contract_id}
        if self.parent_scope_contract is not None:
            allowed_replay_scope_contract_ids.add(
                self.parent_scope_contract.scope_contract_id
            )
        if any(
            replay_scope.scope_id != self.scope_contract.scope_id
            or replay_scope.scope_contract_id not in allowed_replay_scope_contract_ids
            for replay_scope in self.replay_evidence.scope_contracts
        ):
            raise ExpertCandidateContextError(
                "candidate replay scope leaves current or parent scope authority"
            )
        _require_sorted_content_ids(
            self.stable_dependency_ids,
            "candidate validation stable dependencies",
            required=True,
        )
        expected_dependencies = {
            self.scope_contract.scope_contract_id,
            self.replay_evidence.replay_evidence_id,
            *self.replay_evidence.stable_dependency_ids,
        }
        if self.parent_release is not None:
            expected_dependencies.update(
                {
                    self.parent_scope_contract.scope_contract_id,
                    self.parent_release.release_id,
                    self.parent_tree_receipt.parent_tree_receipt_id,
                    self.parent_tree_receipt.source_extraction_receipt.extraction_receipt_id,
                    self.parent_repository_map.repository_map_id,
                    *(
                        module.module_contract_id
                        for module in self.parent_module_contracts
                    ),
                }
            )
        if set(self.stable_dependency_ids) != expected_dependencies:
            raise ExpertCandidateContextError(
                "candidate validation stable dependency closure is not exact"
            )

    @property
    def scope_id(self) -> str:
        return self.scope_contract.scope_id

    @property
    def active_task_family_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({binding.task_family_id for binding in self.active_task_bindings})
        )


def project_agent_candidate_validation_context(
    *,
    packet: ExpertTriggerEvidencePacket,
    decision: ExpertEvolutionTriggerDecision,
) -> ExpertCandidateValidationContext:
    """Project the generic context exactly from one authorized agent proposal."""

    if decision.evidence_packet_id != packet.evidence_packet_id:
        raise ExpertCandidateContextError(
            "agent context decision differs from its trigger packet"
        )
    episodes_by_id = {episode.episode_id: episode for episode in packet.episodes}
    claims_by_id: dict[str, KnowledgeClaim] = {
        claim.revision_id: claim for claim in packet.claims
    }
    observations_by_id = {
        observation.observation_id: observation
        for observation in packet.trigger_observations
    }
    episodes_by_bundle: dict[str, tuple[TransferEpisode, ...]] = {}
    for bundle_id in sorted({episode.source_bundle_id for episode in packet.episodes}):
        episodes_by_bundle[bundle_id] = tuple(
            episode
            for episode in packet.episodes
            if episode.source_bundle_id == bundle_id
        )
    contradiction = decision.reason_code == "released_capability_contradiction"
    pending = [
        (evidence_id, "decision") for evidence_id in decision.trigger_evidence_ids
    ]
    processed: set[tuple[str, str]] = set()
    causal_episode_ids: set[str] = set()
    reason_codes: dict[str, set[str]] = {}
    while pending:
        evidence_id, role = pending.pop(0)
        if (evidence_id, role) in processed:
            continue
        processed.add((evidence_id, role))
        namespace = evidence_id.split(":sha256:", 1)[0]
        if namespace == "transfer-episode":
            if evidence_id not in episodes_by_id:
                raise ExpertCandidateContextError(
                    "agent causal episode is absent from its trigger packet"
                )
            if not contradiction or role == "contradicting_claim":
                causal_episode_ids.add(evidence_id)
                reason_codes.setdefault(evidence_id, set()).add(
                    "causal_contradiction"
                    if role == "contradicting_claim"
                    else "causal_trigger_evidence"
                )
        elif namespace == "expert-trigger-observation":
            observation = observations_by_id.get(evidence_id)
            if observation is None:
                raise ExpertCandidateContextError(
                    "agent causal observation is absent from its trigger packet"
                )
            pending.extend(
                (child, "observation") for child in observation.exact_evidence_ids
            )
        elif namespace == "knowledge-claim-revision":
            claim = claims_by_id.get(evidence_id)
            if claim is None:
                raise ExpertCandidateContextError(
                    "agent causal claim is absent from its trigger packet"
                )
            if contradiction:
                pending.extend(
                    (episode_id, "contradicting_claim")
                    for episode_id in claim.contradicting_episode_ids
                )
        elif namespace == "run-bundle":
            bundle_episodes = episodes_by_bundle.get(evidence_id, ())
            if not bundle_episodes:
                raise ExpertCandidateContextError(
                    "agent causal bundle has no trigger-packet episode"
                )
            if not contradiction or role == "contradicting_claim":
                for episode in bundle_episodes:
                    causal_episode_ids.add(episode.episode_id)
                    reason_codes.setdefault(episode.episode_id, set()).add(
                        "causal_trigger_bundle"
                    )
    causal_episodes = tuple(
        episodes_by_id[episode_id] for episode_id in sorted(causal_episode_ids)
    )
    if decision.task_context_binding_ids and {
        episode.task_context_binding.task_context_binding_id
        for episode in causal_episodes
    } != set(decision.task_context_binding_ids):
        raise ExpertCandidateContextError(
            "agent causal contexts differ from its trigger decision"
        )
    if decision.independent_lineage_ids and {
        episode_lineage_id(episode, episodes_by_id) for episode in causal_episodes
    } != set(decision.independent_lineage_ids):
        raise ExpertCandidateContextError(
            "agent causal lineages differ from its trigger decision"
        )
    evidence_scope_contract = (
        packet.scope_contract
        if packet.knowledge_snapshot_manifest.scope_contract_id
        == packet.scope_contract.scope_contract_id
        else packet.parent_scope_contract
    )
    evidence_authority_ids = tuple(
        sorted(
            {
                packet.knowledge_snapshot_manifest.snapshot_id,
                packet.evidence_packet_id,
                decision.trigger_decision_id,
            }
        )
    )
    replay_dependencies = tuple(
        sorted(
            {
                packet.knowledge_snapshot_manifest.snapshot_id,
                evidence_scope_contract.scope_contract_id,
                *(episode.episode_id for episode in packet.episodes),
                *(episode.source_bundle_id for episode in packet.episodes),
                *evidence_authority_ids,
                *packet.proof_reference_ids,
            }
        )
    )
    replay_evidence = ExpertCandidateReplayEvidence.mint(
        knowledge_snapshot_manifests=(packet.knowledge_snapshot_manifest,),
        scope_contracts=(evidence_scope_contract,),
        episodes=packet.episodes,
        causal_episode_ids=tuple(sorted(causal_episode_ids)),
        causal_episode_reason_codes={
            episode_id: tuple(sorted(reasons))
            for episode_id, reasons in sorted(reason_codes.items())
        },
        evidence_authority_ids=evidence_authority_ids,
        proof_reference_ids=packet.proof_reference_ids,
        stable_dependency_ids=replay_dependencies,
    )
    context_dependencies = {
        packet.scope_contract.scope_contract_id,
        replay_evidence.replay_evidence_id,
        *replay_evidence.stable_dependency_ids,
    }
    if packet.parent_release is not None:
        context_dependencies.update(
            {
                packet.parent_scope_contract.scope_contract_id,
                packet.parent_release.release_id,
                packet.parent_tree_receipt.parent_tree_receipt_id,
                packet.parent_tree_receipt.source_extraction_receipt.extraction_receipt_id,
                packet.repository_map.repository_map_id,
                *(module.module_contract_id for module in packet.module_contracts),
            }
        )
    return ExpertCandidateValidationContext.mint(
        scope_contract=packet.scope_contract,
        parent_scope_contract=packet.parent_scope_contract,
        parent_release=packet.parent_release,
        parent_tree_receipt=packet.parent_tree_receipt,
        parent_tree_hash=packet.parent_tree_hash,
        parent_repository_map=packet.repository_map,
        parent_module_contracts=packet.module_contracts,
        active_task_bindings=packet.active_task_bindings,
        replay_evidence=replay_evidence,
        stable_dependency_ids=tuple(sorted(context_dependencies)),
    )
