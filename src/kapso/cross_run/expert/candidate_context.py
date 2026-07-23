"""Origin-neutral scientific and validation context for expert candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping, TypeVar

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
    ExpertSourceBaseTreeReceipt,
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


def candidate_consumed_expert_release_ids(
    *,
    source_base_release_id: str | None,
    replay_evidence: ExpertCandidateReplayEvidence,
    inherited_release_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Project the transitive releases whose code or run outputs shaped a candidate."""

    if (
        type(replay_evidence) is not ExpertCandidateReplayEvidence
        or type(inherited_release_ids) is not tuple
    ):
        raise ExpertCandidateContextError(
            "candidate release-use lineage requires exact typed inputs"
        )
    release_ids = set(inherited_release_ids)
    if source_base_release_id is not None:
        release_ids.add(source_base_release_id)
    release_ids.update(
        episode.artifact_environment.expert_base_release_id
        for episode in replay_evidence.episodes
    )
    ordered = tuple(sorted(release_ids))
    for release_id in ordered:
        require_content_id(release_id, "candidate consumed expert release")
        if release_id.split(":sha256:", 1)[0] != "expert-base-release":
            raise ExpertCandidateContextError(
                "candidate consumed expert release uses the wrong namespace"
            )
    return ordered


@dataclass(frozen=True)
class ExpertCandidateValidationContext(StrictContract):
    """Stable origin-neutral context consumed by the validation cascade."""

    validation_context_id: str
    scope_contract: ExpertScopeContract
    source_base_scope_contract: ExpertScopeContract | None
    source_base_release: ExpertBaseReleaseManifest | None
    source_base_tree_receipt: ExpertSourceBaseTreeReceipt | None
    source_base_tree_hash: str
    source_base_repository_map: ExpertRepositoryMap | None
    source_base_module_contracts: tuple[ExpertModuleContract, ...]
    active_task_bindings: tuple[CrossRunTaskBindingSettings, ...]
    replay_evidence: ExpertCandidateReplayEvidence
    stable_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-validation-context"
    IDENTITY_FIELD: ClassVar[str] = "validation_context_id"

    def _validate(self) -> None:
        if (
            type(self.scope_contract) is not ExpertScopeContract
            or type(self.source_base_module_contracts) is not tuple
            or any(
                type(module) is not ExpertModuleContract
                for module in self.source_base_module_contracts
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
        if self.source_base_release is None:
            if (
                self.source_base_tree_hash != EMPTY_EXPERT_TREE_DIGEST
                or self.source_base_scope_contract is not None
                or self.source_base_tree_receipt is not None
                or self.source_base_repository_map is not None
                or self.source_base_module_contracts
            ):
                raise ExpertCandidateContextError(
                    "bootstrap validation context must have an explicit empty source base"
                )
        elif (
            type(self.source_base_release) is not ExpertBaseReleaseManifest
            or type(self.source_base_scope_contract) is not ExpertScopeContract
            or type(self.source_base_tree_receipt) is not ExpertSourceBaseTreeReceipt
            or type(self.source_base_repository_map) is not ExpertRepositoryMap
            or self.source_base_release.scope_contract_id
            != self.source_base_scope_contract.scope_contract_id
            or self.source_base_release.scope_id
            != self.source_base_scope_contract.scope_id
            or self.source_base_scope_contract.scope_id != self.scope_contract.scope_id
            or self.source_base_release.release_id
            != self.source_base_tree_receipt.release_id
            or self.source_base_release.repository_map_ref
            != self.source_base_repository_map.repository_map_id
            or self.source_base_tree_hash
            != self.source_base_tree_receipt.source_base_tree_hash
            or self.source_base_repository_map.scope_contract_id
            != self.source_base_scope_contract.scope_contract_id
            or dict(self.source_base_release.module_versions)
            != {
                module.module_id: module.version
                for module in self.source_base_module_contracts
            }
        ):
            raise ExpertCandidateContextError(
                "released validation context source-base authority is inconsistent"
            )
        if self.source_base_release is not None:
            if self.source_base_tree_hash == EMPTY_EXPERT_TREE_DIGEST:
                raise ExpertCandidateContextError(
                    "released validation context cannot use the empty tree"
                )
            if (
                self.scope_contract.scope_contract_id
                != self.source_base_scope_contract.scope_contract_id
                and self.scope_contract.supersedes_scope_contract_id
                != self.source_base_scope_contract.scope_contract_id
            ):
                raise ExpertCandidateContextError(
                    "candidate scope must equal or directly supersede its parent"
                )
            if (
                self.scope_contract.scope_contract_id
                != self.source_base_scope_contract.scope_contract_id
            ):
                parent_family_ids = {
                    family.task_family_id
                    for family in self.source_base_scope_contract.task_family_ontology
                }
                parent_dimension_ids = {
                    schema.dimension_id
                    for schema in self.source_base_scope_contract.context_dimension_schemas
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
                    module.module_contract_id
                    for module in self.source_base_module_contracts
                )
            )
            map_contract_ids = tuple(
                sorted(
                    node.module_contract_ref
                    for node in self.source_base_repository_map.capability_nodes
                )
            )
            receipt = self.source_base_tree_receipt
            cache_receipt = receipt.cache_verification_receipt
            extraction_receipt = receipt.source_extraction_receipt
            archive_ref = self.source_base_release.source_archive_ref
            archive_digest = self.source_base_release.checksums[archive_ref]
            if (
                map_contract_ids != module_contract_ids
                or receipt.repository_map_id
                != self.source_base_repository_map.repository_map_id
                or receipt.module_contract_ids != module_contract_ids
                or cache_receipt.artifact_kind
                is not PublicationArtifactKind.EXPERT_BASE_RELEASE
                or cache_receipt.artifact_id != self.source_base_release.release_id
                or cache_receipt.asset_digests.get(archive_ref) != archive_digest
                or extraction_receipt.artifact_id != self.source_base_release.release_id
                or extraction_receipt.source_archive_ref != archive_ref
                or extraction_receipt.source_archive_digest != archive_digest
                or extraction_receipt.source_tree_hash != self.source_base_tree_hash
            ):
                raise ExpertCandidateContextError(
                    "candidate source-base release closure is inconsistent"
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
        if self.source_base_scope_contract is not None:
            allowed_replay_scope_contract_ids.add(
                self.source_base_scope_contract.scope_contract_id
            )
        if any(
            replay_scope.scope_id != self.scope_contract.scope_id
            or replay_scope.scope_contract_id not in allowed_replay_scope_contract_ids
            for replay_scope in self.replay_evidence.scope_contracts
        ):
            raise ExpertCandidateContextError(
                "candidate replay scope leaves current or source-base scope authority"
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
        if self.source_base_release is not None:
            expected_dependencies.update(
                {
                    self.source_base_scope_contract.scope_contract_id,
                    self.source_base_release.release_id,
                    self.source_base_tree_receipt.source_base_tree_receipt_id,
                    self.source_base_tree_receipt.source_extraction_receipt.extraction_receipt_id,
                    self.source_base_repository_map.repository_map_id,
                    *(
                        module.module_contract_id
                        for module in self.source_base_module_contracts
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


_ReplayRecord = TypeVar(
    "_ReplayRecord",
    KnowledgeSnapshotManifest,
    ExpertScopeContract,
    TransferEpisode,
)


def _exact_replay_record_union(
    records: tuple[_ReplayRecord, ...],
    *,
    identity_field: str,
    name: str,
) -> tuple[_ReplayRecord, ...]:
    by_id: dict[str, _ReplayRecord] = {}
    for record in records:
        record_id = getattr(record, identity_field)
        existing = by_id.get(record_id)
        if existing is not None and existing != record:
            raise ExpertCandidateContextError(
                f"candidate replay {name} reuse one ID with unequal content"
            )
        by_id[record_id] = record
    return tuple(by_id[record_id] for record_id in sorted(by_id))


def compose_candidate_replay_evidence(
    source_contexts: tuple[ExpertCandidateValidationContext, ...],
) -> ExpertCandidateReplayEvidence:
    """Union source replay closures without trimming or weakening provenance."""

    if not source_contexts or any(
        type(context) is not ExpertCandidateValidationContext
        for context in source_contexts
    ):
        raise ExpertCandidateContextError(
            "composed replay requires exact non-empty source contexts"
        )
    snapshots = _exact_replay_record_union(
        tuple(
            manifest
            for context in source_contexts
            for manifest in context.replay_evidence.knowledge_snapshot_manifests
        ),
        identity_field="snapshot_id",
        name="knowledge snapshots",
    )
    scope_contracts = _exact_replay_record_union(
        tuple(
            scope_contract
            for context in source_contexts
            for scope_contract in context.replay_evidence.scope_contracts
        ),
        identity_field="scope_contract_id",
        name="scope contracts",
    )
    episodes = _exact_replay_record_union(
        tuple(
            episode
            for context in source_contexts
            for episode in context.replay_evidence.episodes
        ),
        identity_field="episode_id",
        name="episodes",
    )
    reason_codes: dict[str, set[str]] = {}
    for context in source_contexts:
        for (
            episode_id,
            reasons,
        ) in context.replay_evidence.causal_episode_reason_codes.items():
            reason_codes.setdefault(episode_id, set()).update(reasons)
    causal_episode_ids = tuple(
        sorted(
            {
                episode_id
                for context in source_contexts
                for episode_id in context.replay_evidence.causal_episode_ids
            }
        )
    )
    evidence_authority_ids = tuple(
        sorted(
            {
                *(context.validation_context_id for context in source_contexts),
                *(
                    authority_id
                    for context in source_contexts
                    for authority_id in (context.replay_evidence.evidence_authority_ids)
                ),
            }
        )
    )
    proof_reference_ids = tuple(
        sorted(
            {
                proof_id
                for context in source_contexts
                for proof_id in context.replay_evidence.proof_reference_ids
            }
        )
    )
    stable_dependency_ids = tuple(
        sorted(
            {
                *(manifest.snapshot_id for manifest in snapshots),
                *(scope.scope_contract_id for scope in scope_contracts),
                *(episode.episode_id for episode in episodes),
                *(episode.source_bundle_id for episode in episodes),
                *evidence_authority_ids,
                *proof_reference_ids,
            }
        )
    )
    return ExpertCandidateReplayEvidence.mint(
        knowledge_snapshot_manifests=snapshots,
        scope_contracts=scope_contracts,
        episodes=episodes,
        causal_episode_ids=causal_episode_ids,
        causal_episode_reason_codes={
            episode_id: tuple(sorted(reason_codes[episode_id]))
            for episode_id in causal_episode_ids
        },
        evidence_authority_ids=evidence_authority_ids,
        proof_reference_ids=proof_reference_ids,
        stable_dependency_ids=stable_dependency_ids,
    )


def project_recovery_replay_evidence(
    packet: ExpertTriggerEvidencePacket,
) -> ExpertCandidateReplayEvidence:
    """Treat every current packet episode as applicable recovery evidence."""

    if type(packet) is not ExpertTriggerEvidencePacket:
        raise ExpertCandidateContextError(
            "recovery replay projection requires one exact trigger packet"
        )
    evidence_scope_contract = (
        packet.scope_contract
        if packet.knowledge_snapshot_manifest.scope_contract_id
        == packet.scope_contract.scope_contract_id
        else packet.source_base_scope_contract
    )
    if evidence_scope_contract is None:
        raise ExpertCandidateContextError(
            "recovery replay packet has no knowledge scope authority"
        )
    episode_ids = tuple(episode.episode_id for episode in packet.episodes)
    evidence_authority_ids = tuple(
        sorted(
            {
                packet.knowledge_snapshot_manifest.snapshot_id,
                packet.evidence_packet_id,
            }
        )
    )
    dependencies = tuple(
        sorted(
            {
                packet.knowledge_snapshot_manifest.snapshot_id,
                evidence_scope_contract.scope_contract_id,
                *episode_ids,
                *(episode.source_bundle_id for episode in packet.episodes),
                *evidence_authority_ids,
                *packet.proof_reference_ids,
            }
        )
    )
    return ExpertCandidateReplayEvidence.mint(
        knowledge_snapshot_manifests=(packet.knowledge_snapshot_manifest,),
        scope_contracts=(evidence_scope_contract,),
        episodes=packet.episodes,
        causal_episode_ids=episode_ids,
        causal_episode_reason_codes={
            episode_id: ("clean_forward_recovery",) for episode_id in episode_ids
        },
        evidence_authority_ids=evidence_authority_ids,
        proof_reference_ids=packet.proof_reference_ids,
        stable_dependency_ids=dependencies,
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
        else packet.source_base_scope_contract
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
    if packet.source_base_release is not None:
        context_dependencies.update(
            {
                packet.source_base_scope_contract.scope_contract_id,
                packet.source_base_release.release_id,
                packet.source_base_tree_receipt.source_base_tree_receipt_id,
                packet.source_base_tree_receipt.source_extraction_receipt.extraction_receipt_id,
                packet.source_base_repository_map.repository_map_id,
                *(
                    module.module_contract_id
                    for module in packet.source_base_module_contracts
                ),
            }
        )
    return ExpertCandidateValidationContext.mint(
        scope_contract=packet.scope_contract,
        source_base_scope_contract=packet.source_base_scope_contract,
        source_base_release=packet.source_base_release,
        source_base_tree_receipt=packet.source_base_tree_receipt,
        source_base_tree_hash=packet.source_base_tree_hash,
        source_base_repository_map=packet.source_base_repository_map,
        source_base_module_contracts=packet.source_base_module_contracts,
        active_task_bindings=packet.active_task_bindings,
        replay_evidence=replay_evidence,
        stable_dependency_ids=tuple(sorted(context_dependencies)),
    )
