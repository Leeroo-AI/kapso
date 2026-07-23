"""Deterministic prompt, output, and ancestor contracts for expert proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertCandidateManifest,
    ExpertCandidateOperationKind,
    ExpertCandidatePatch,
    ExpertCandidatePatchChange,
    ExpertCandidateSanitationReport,
    ExpertCapabilityLineage,
    ExpertCapabilityNode,
    ExpertDependencyEdge,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertRecoveryRestorePatch,
    ExpertScopeContract,
    ExpertSourceTreeManifest,
    ExpertTaskAdapterBoundary,
    StrictContract,
    TransferEpisode,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertTriggerEvidencePacket,
)
from kapso.cross_run.knowledge.access import PriorKnowledgeAccessMaterialization
from kapso.cross_run.record_registry import parse_knowledge_record_payload

EXPERT_PROPOSAL_CONTRACT_VERSION = "kapso.expert_proposal.v1"
EXPERT_PROPOSAL_PACKET_MARKER = "EXPERT_PROPOSAL_PACKET_JSON"

_PROMPT_ROOT = Path(__file__).parents[1] / "prompts"
_PROMPT_PATHS = {
    ExpertCandidateOperationKind.BOOTSTRAP: _PROMPT_ROOT / "expert_repo_bootstrap.md",
    ExpertCandidateOperationKind.RECOVERY_BOOTSTRAP: (
        _PROMPT_ROOT / "expert_repo_bootstrap.md"
    ),
    ExpertCandidateOperationKind.RESTRUCTURE: (
        _PROMPT_ROOT / "expert_repo_restructure.md"
    ),
    ExpertCandidateOperationKind.GENERALIZE: (
        _PROMPT_ROOT / "expert_capability_generalization.md"
    ),
}


class ExpertProposalContractError(ValueError):
    """An expert proposal input or structured output is invalid."""


def _require_nonempty_text(value: str, name: str) -> None:
    if not value.strip():
        raise ExpertProposalContractError(f"{name} must not be empty")


def _require_sorted_unique(values: tuple[str, ...], name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ExpertProposalContractError(f"{name} must be sorted and unique")


@dataclass(frozen=True)
class ExpertModuleProposal(StrictContract):
    """Complete semantic module contract without framework-owned identity."""

    module_id: str
    version: str
    purpose: str
    problem_signals: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    preconditions: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    dependency_capability_ids: tuple[str, ...]
    incompatible_capability_ids: tuple[str, ...]
    resource_bounds: Mapping[str, Any]
    dependency_license_manifest: Mapping[str, Any]
    supporting_episode_ids: tuple[str, ...]
    known_failure_episode_ids: tuple[str, ...]
    entrypoint_refs: tuple[str, ...]
    test_refs: tuple[str, ...]
    replay_refs: tuple[str, ...]

    def mint_contract(self) -> ExpertModuleContract:
        return ExpertModuleContract.mint(**self.to_dict())


@dataclass(frozen=True)
class ExpertCapabilityTopologyProposal(StrictContract):
    """One semantic capability's path and task-family ownership."""

    capability_id: str
    owned_paths: tuple[str, ...]
    task_family_bindings: tuple[str, ...]


@dataclass(frozen=True)
class ExpertRepositoryTopologyProposal(StrictContract):
    """Complete repository topology without derived references or edges."""

    capability_nodes: tuple[ExpertCapabilityTopologyProposal, ...]
    task_adapter_boundary: ExpertTaskAdapterBoundary
    validation_entrypoints: tuple[str, ...]
    architecture_invariants: tuple[str, ...]


@dataclass(frozen=True)
class ExpertArchitectProposal(StrictContract):
    """Full desired repository state emitted by bootstrap/restructure roles."""

    summary: str
    changed_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]
    repository_topology: ExpertRepositoryTopologyProposal
    module_contracts: tuple[ExpertModuleProposal, ...]
    capability_lineage: tuple[ExpertCapabilityLineage, ...]

    def _validate(self) -> None:
        _require_nonempty_text(self.summary, "architect proposal summary")
        _require_sorted_unique(self.changed_paths, "architect changed_paths")
        _require_sorted_unique(self.deleted_paths, "architect deleted_paths")
        module_ids = tuple(module.module_id for module in self.module_contracts)
        if not module_ids or module_ids != tuple(sorted(set(module_ids))):
            raise ExpertProposalContractError(
                "architect module contracts must be non-empty, sorted, and unique"
            )
        capability_ids = tuple(
            node.capability_id for node in self.repository_topology.capability_nodes
        )
        if capability_ids != tuple(sorted(set(capability_ids))):
            raise ExpertProposalContractError(
                "architect capability nodes must be non-empty, sorted, and unique"
            )
        lineage_keys = tuple(
            (
                lineage.relation.value,
                lineage.source_capability_ids,
                lineage.target_capability_ids,
            )
            for lineage in self.capability_lineage
        )
        if lineage_keys != tuple(sorted(set(lineage_keys))):
            raise ExpertProposalContractError(
                "architect capability lineage must be sorted and unique"
            )


@dataclass(frozen=True)
class ExpertGeneralizerProposal(StrictContract):
    """Smallest complete set of module contracts changed by generalization."""

    summary: str
    changed_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]
    changed_module_contracts: tuple[ExpertModuleProposal, ...]

    def _validate(self) -> None:
        _require_nonempty_text(self.summary, "generalizer proposal summary")
        _require_sorted_unique(self.changed_paths, "generalizer changed_paths")
        _require_sorted_unique(self.deleted_paths, "generalizer deleted_paths")
        module_ids = tuple(module.module_id for module in self.changed_module_contracts)
        if not module_ids or module_ids != tuple(sorted(set(module_ids))):
            raise ExpertProposalContractError(
                "changed module contracts must be non-empty, sorted, and unique"
            )


ExpertStructuredProposal = ExpertArchitectProposal | ExpertGeneralizerProposal


@dataclass(frozen=True)
class ExpertCandidateAncestorInput(StrictContract):
    """Self-contained reusable source input from one persisted candidate."""

    ancestor_input_id: str
    manifest: ExpertCandidateManifest
    scope_contract: ExpertScopeContract
    patch: ExpertCandidatePatch | ExpertRecoveryRestorePatch
    candidate_tree: ExpertSourceTreeManifest
    repository_map: ExpertRepositoryMap
    module_contracts: tuple[ExpertModuleContract, ...]
    sanitation_report: ExpertCandidateSanitationReport
    candidate_contents_text: Mapping[str, str]

    CONTENT_NAMESPACE = "expert-candidate-ancestor-input"
    IDENTITY_FIELD = "ancestor_input_id"

    def _validate(self) -> None:
        module_refs = tuple(
            sorted(module.module_contract_id for module in self.module_contracts)
        )
        module_ids = tuple(module.module_id for module in self.module_contracts)
        if module_ids != tuple(sorted(set(module_ids))):
            raise ExpertProposalContractError(
                "ancestor modules must be sorted and uniquely identified"
            )
        if (
            self.manifest.scope_contract_id != self.scope_contract.scope_contract_id
            or self.manifest.patch_ref != self.patch.patch_id
            or self.manifest.patch_digest
            != tree_or_blob_digest(self.patch.to_json_bytes())
            or self.manifest.candidate_tree_ref
            != self.candidate_tree.source_tree_manifest_id
            or self.manifest.candidate_tree_hash != self.candidate_tree.tree_hash
            or self.manifest.proposed_repository_map_ref
            != self.repository_map.repository_map_id
            or self.manifest.module_contract_refs != module_refs
            or self.manifest.sanitation_report_id
            != self.sanitation_report.sanitation_report_id
            or self.patch.candidate_tree_hash != self.candidate_tree.tree_hash
        ):
            raise ExpertProposalContractError(
                "ancestor materialization differs from its candidate manifest"
            )
        contents = self.candidate_contents()
        descriptors = {file.relative_path: file for file in self.candidate_tree.files}
        if set(contents) != set(descriptors):
            raise ExpertProposalContractError(
                "ancestor source bytes differ from its exact tree closure"
            )
        for path, payload in contents.items():
            descriptor = descriptors[path]
            if (
                tree_or_blob_digest(payload) != descriptor.digest
                or len(payload) != descriptor.size
            ):
                raise ExpertProposalContractError(
                    f"ancestor source differs from its descriptor: {path}"
                )
        source_base_descriptors = dict(descriptors)
        for change in self.patch.changes:
            if descriptors.get(change.relative_path) != change.after:
                raise ExpertProposalContractError(
                    "ancestor patch differs from its candidate tree"
                )
            if change.before is None:
                source_base_descriptors.pop(change.relative_path, None)
            else:
                source_base_descriptors[change.relative_path] = change.before
        source_base_tree_hash = (
            EMPTY_EXPERT_TREE_DIGEST
            if not source_base_descriptors
            else source_tree_digest(
                {
                    path: (descriptor.digest, descriptor.mode, descriptor.size)
                    for path, descriptor in source_base_descriptors.items()
                }
            )
        )
        expected_changes = tuple(
            ExpertCandidatePatchChange(
                relative_path=path,
                before=source_base_descriptors.get(path),
                after=descriptors.get(path),
            )
            for path in sorted(set(source_base_descriptors) | set(descriptors))
            if source_base_descriptors.get(path) != descriptors.get(path)
        )
        if (
            self.patch.source_base_tree_hash != self.manifest.source_base_tree_hash
            or self.patch.source_base_tree_hash != source_base_tree_hash
            or self.patch.changes != expected_changes
        ):
            raise ExpertProposalContractError(
                "ancestor patch does not transform its exact source-base tree"
            )
        self._validate_generated_controls(contents)

    def candidate_contents(self) -> dict[str, bytes]:
        return {
            path: payload.encode("utf-8")
            for path, payload in self.candidate_contents_text.items()
        }

    def _validate_generated_controls(self, contents: Mapping[str, bytes]) -> None:
        expected_book = compile_expert_semantic_book(
            self.scope_contract,
            self.repository_map,
            self.module_contracts,
        )
        if (
            contents.get(EXPERT_BOOK_PATH) != expected_book
            or self.manifest.semantic_book_digest
            != expert_semantic_book_digest(expected_book)
            or contents.get(EXPERT_REPOSITORY_MAP_PATH)
            != self.repository_map.to_json_bytes()
        ):
            raise ExpertProposalContractError(
                "ancestor generated book or repository map differs"
            )
        for module in self.module_contracts:
            if contents.get(expert_module_contract_path(module.module_contract_id)) != (
                module.to_json_bytes()
            ):
                raise ExpertProposalContractError(
                    "ancestor generated module contract differs"
                )


def mint_expert_candidate_ancestor_input(
    *,
    manifest: ExpertCandidateManifest,
    scope_contract: ExpertScopeContract,
    patch: ExpertCandidatePatch | ExpertRecoveryRestorePatch,
    candidate_tree: ExpertSourceTreeManifest,
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
    sanitation_report: ExpertCandidateSanitationReport,
    candidate_contents: Mapping[str, bytes],
) -> ExpertCandidateAncestorInput:
    return ExpertCandidateAncestorInput.mint(
        manifest=manifest,
        scope_contract=scope_contract,
        patch=patch,
        candidate_tree=candidate_tree,
        repository_map=repository_map,
        module_contracts=module_contracts,
        sanitation_report=sanitation_report,
        candidate_contents_text={
            path: payload.decode("utf-8")
            for path, payload in sorted(candidate_contents.items())
        },
    )


def expert_candidate_operation_kind(
    packet: ExpertTriggerEvidencePacket,
    decision: ExpertEvolutionTriggerDecision,
) -> ExpertCandidateOperationKind:
    if not decision.candidate_required or decision.change_kind is None:
        raise ExpertProposalContractError(
            "expert proposal requires an affirmative trigger decision"
        )
    if packet.recovery_barrier_basis_packet_id is not None:
        return ExpertCandidateOperationKind.RECOVERY_BOOTSTRAP
    if packet.source_base_release is None:
        return ExpertCandidateOperationKind.BOOTSTRAP
    if decision.change_kind.value == "repository_architecture":
        return ExpertCandidateOperationKind.RESTRUCTURE
    return ExpertCandidateOperationKind.GENERALIZE


def validate_expert_prior_knowledge(
    packet: ExpertTriggerEvidencePacket,
    prior_knowledge: PriorKnowledgeAccessMaterialization | None,
) -> None:
    if prior_knowledge is None:
        return
    snapshot = prior_knowledge.prior_knowledge_snapshot
    selected_ids = set(snapshot.selected_record_ids)
    allowed_selected_ids = {
        *packet.knowledge_snapshot_manifest.admitted_episode_ids,
        *packet.knowledge_snapshot_manifest.admitted_prior_idea_ids,
        *packet.knowledge_snapshot_manifest.active_claim_revision_ids,
    }
    allowed_proof_ids = {
        *packet.proof_reference_ids,
        *allowed_selected_ids,
    }
    if (
        snapshot.source_snapshot_id != packet.knowledge_snapshot_id
        or not selected_ids.issubset(allowed_selected_ids)
        or not set(snapshot.proof_reference_ids).issubset(allowed_proof_ids)
    ):
        raise ExpertProposalContractError(
            "expert prior knowledge leaves its trigger evidence closure"
        )


def expert_candidate_source_dependency_ids(
    packet: ExpertTriggerEvidencePacket,
    decision: ExpertEvolutionTriggerDecision,
    prior_knowledge: PriorKnowledgeAccessMaterialization | None,
) -> tuple[str, ...]:
    """Return every content-addressed input that can influence a proposal."""

    dependencies = {
        packet.evidence_packet_id,
        decision.trigger_decision_id,
        *decision.trigger_evidence_ids,
    }
    if prior_knowledge is not None:
        snapshot = prior_knowledge.prior_knowledge_snapshot
        dependencies.update(
            {
                snapshot.prior_knowledge_snapshot_id,
                snapshot.source_snapshot_id,
                *snapshot.selected_record_ids,
                *snapshot.proof_reference_ids,
            }
        )
    return tuple(sorted(dependencies))


def expert_candidate_prior_knowledge_release_ids(
    prior_knowledge: PriorKnowledgeAccessMaterialization | None,
) -> tuple[str, ...]:
    """Project expert releases from every prior episode visible to the proposer."""

    if prior_knowledge is None:
        return ()
    snapshot = prior_knowledge.prior_knowledge_snapshot
    envelopes = (*snapshot.selected_records, *prior_knowledge.proof_records)
    episodes = tuple(
        parse_knowledge_record_payload(
            envelope["record_kind"],
            envelope["payload"],
        )
        for envelope in envelopes
        if envelope["record_kind"] == "transfer-episode"
    )
    if any(type(episode) is not TransferEpisode for episode in episodes):
        raise ExpertProposalContractError(
            "prior-knowledge episode uses another record type"
        )
    return tuple(
        sorted(
            {
                episode.artifact_environment.expert_base_release_id
                for episode in episodes
            }
        )
    )


def build_expert_proposal_packet(
    *,
    packet: ExpertTriggerEvidencePacket,
    decision: ExpertEvolutionTriggerDecision,
    operation_kind: ExpertCandidateOperationKind,
    editable_input_tree_hash: str,
    maximum_entries: int,
    maximum_bytes: int,
    ancestor_inputs: tuple[ExpertCandidateAncestorInput, ...],
) -> Mapping[str, Any]:
    return {
        "proposal_contract_version": EXPERT_PROPOSAL_CONTRACT_VERSION,
        "operation_kind": operation_kind.value,
        "trigger_packet": packet.to_dict(),
        "trigger_decision": decision.to_dict(),
        "editable_input_tree_hash": editable_input_tree_hash,
        "workspace_limits": {
            "maximum_entries": maximum_entries,
            "maximum_bytes": maximum_bytes,
        },
        "generated_control_paths": {
            "book": EXPERT_BOOK_PATH,
            "repository_map": EXPERT_REPOSITORY_MAP_PATH,
            "module_contract_root": ".kapso/expert/module-contracts",
        },
        "ancestor_inputs": tuple(ancestor.to_dict() for ancestor in ancestor_inputs),
    }


def build_expert_proposal_prompt(
    operation_kind: ExpertCandidateOperationKind,
    proposal_packet: Mapping[str, Any],
) -> str:
    template_path = _PROMPT_PATHS[operation_kind]
    template = template_path.read_text(encoding="utf-8")
    if template.count(EXPERT_PROPOSAL_PACKET_MARKER) != 1:
        raise ExpertProposalContractError("expert proposal template marker is invalid")
    return template.replace(
        EXPERT_PROPOSAL_PACKET_MARKER,
        canonical_json_bytes(proposal_packet).decode("utf-8"),
    )


def expert_proposal_packet_digest(proposal_packet: Mapping[str, Any]) -> str:
    return tree_or_blob_digest(canonical_json_bytes(proposal_packet))


def parse_expert_proposal(
    operation_kind: ExpertCandidateOperationKind,
    final_output: str,
) -> ExpertStructuredProposal:
    payload = parse_json_bytes(final_output)
    if operation_kind is ExpertCandidateOperationKind.GENERALIZE:
        return ExpertGeneralizerProposal.from_dict(payload)
    return ExpertArchitectProposal.from_dict(payload)


def derive_expert_proposal_topology(
    *,
    packet: ExpertTriggerEvidencePacket,
    operation_kind: ExpertCandidateOperationKind,
    proposal: ExpertStructuredProposal,
) -> tuple[
    ExpertRepositoryMap,
    tuple[ExpertModuleContract, ...],
    tuple[ExpertCapabilityLineage, ...],
]:
    if operation_kind is ExpertCandidateOperationKind.GENERALIZE:
        if not isinstance(proposal, ExpertGeneralizerProposal):
            raise ExpertProposalContractError(
                "generalization received an architect proposal"
            )
        return _derive_generalized_topology(packet, proposal)
    if not isinstance(proposal, ExpertArchitectProposal):
        raise ExpertProposalContractError(
            "architecture operation received a generalizer proposal"
        )
    return _derive_architect_topology(packet, operation_kind, proposal)


def _derive_architect_topology(
    packet: ExpertTriggerEvidencePacket,
    operation_kind: ExpertCandidateOperationKind,
    proposal: ExpertArchitectProposal,
) -> tuple[
    ExpertRepositoryMap,
    tuple[ExpertModuleContract, ...],
    tuple[ExpertCapabilityLineage, ...],
]:
    modules = tuple(module.mint_contract() for module in proposal.module_contracts)
    modules_by_id = {module.module_id: module for module in modules}
    topology = proposal.repository_topology
    if set(modules_by_id) != {node.capability_id for node in topology.capability_nodes}:
        raise ExpertProposalContractError(
            "architect module contracts and capability nodes are not a bijection"
        )
    nodes = tuple(
        ExpertCapabilityNode(
            capability_id=node.capability_id,
            module_contract_ref=modules_by_id[node.capability_id].module_contract_id,
            owned_paths=node.owned_paths,
            task_family_bindings=node.task_family_bindings,
        )
        for node in topology.capability_nodes
    )
    edges = tuple(
        sorted(
            (
                ExpertDependencyEdge(
                    source_capability_id=module.module_id,
                    target_capability_id=dependency_id,
                )
                for module in modules
                for dependency_id in module.dependency_capability_ids
            ),
            key=lambda edge: (
                edge.source_capability_id,
                edge.target_capability_id,
            ),
        )
    )
    if (
        operation_kind
        in {
            ExpertCandidateOperationKind.BOOTSTRAP,
            ExpertCandidateOperationKind.RECOVERY_BOOTSTRAP,
        }
        and proposal.capability_lineage
    ):
        raise ExpertProposalContractError(
            "bootstrap proposal cannot declare capability lineage"
        )
    repository_map = ExpertRepositoryMap.mint(
        scope_contract_id=packet.scope_contract.scope_contract_id,
        capability_nodes=nodes,
        dependency_edges=edges,
        task_adapter_boundary=topology.task_adapter_boundary,
        validation_entrypoints=topology.validation_entrypoints,
        architecture_invariants=topology.architecture_invariants,
    )
    if operation_kind is ExpertCandidateOperationKind.RESTRUCTURE:
        _validate_restructure_authority(
            packet,
            repository_map,
            modules,
            proposal,
        )
    return repository_map, modules, proposal.capability_lineage


def _validate_restructure_authority(
    packet: ExpertTriggerEvidencePacket,
    repository_map: ExpertRepositoryMap,
    modules: tuple[ExpertModuleContract, ...],
    proposal: ExpertArchitectProposal,
) -> None:
    if packet.source_base_repository_map is None:
        raise ExpertProposalContractError(
            "restructure requires a released source-base repository map"
        )
    source_base_modules = {
        module.module_id: module for module in packet.source_base_module_contracts
    }
    current_modules = {module.module_id: module for module in modules}
    for module_id in sorted(set(source_base_modules) & set(current_modules)):
        source_base_module = source_base_modules[module_id]
        current = current_modules[module_id]
        if source_base_module.module_contract_id != current.module_contract_id:
            _validate_preserved_module_change(
                source_base_module,
                current,
                allow_path_reference_replacement=True,
            )
            _validate_restructure_path_references(source_base_module, current, proposal)
    if _repository_architecture_signature(
        packet.source_base_repository_map,
        packet.source_base_module_contracts,
    ) == _repository_architecture_signature(repository_map, modules):
        raise ExpertProposalContractError(
            "restructure must change repository structure or path interfaces"
        )


def _validate_restructure_path_references(
    source_base_module: ExpertModuleContract,
    changed: ExpertModuleContract,
    proposal: ExpertArchitectProposal,
) -> None:
    for field_name in ("entrypoint_refs", "test_refs", "replay_refs"):
        source_base_refs = set(getattr(source_base_module, field_name))
        changed_refs = set(getattr(changed, field_name))
        removed_refs = source_base_refs - changed_refs
        added_refs = changed_refs - source_base_refs
        if not removed_refs:
            continue
        if (
            not removed_refs.issubset(proposal.deleted_paths)
            or not added_refs.issubset(proposal.changed_paths)
            or len(added_refs) < len(removed_refs)
        ):
            raise ExpertProposalContractError(
                "restructure path-reference replacement lacks an exact source move"
            )


def _repository_architecture_signature(
    repository_map: ExpertRepositoryMap,
    modules: tuple[ExpertModuleContract, ...],
) -> tuple[Any, ...]:
    return (
        tuple(
            (
                node.capability_id,
                node.owned_paths,
                node.task_family_bindings,
            )
            for node in repository_map.capability_nodes
        ),
        repository_map.dependency_edges,
        repository_map.task_adapter_boundary,
        repository_map.validation_entrypoints,
        repository_map.architecture_invariants,
        tuple(
            (
                module.module_id,
                module.entrypoint_refs,
                module.test_refs,
                module.replay_refs,
            )
            for module in sorted(modules, key=lambda item: item.module_id)
        ),
    )


def _derive_generalized_topology(
    packet: ExpertTriggerEvidencePacket,
    proposal: ExpertGeneralizerProposal,
) -> tuple[
    ExpertRepositoryMap,
    tuple[ExpertModuleContract, ...],
    tuple[ExpertCapabilityLineage, ...],
]:
    if packet.source_base_repository_map is None:
        raise ExpertProposalContractError(
            "generalization requires a released source-base repository map"
        )
    source_base_modules = {
        module.module_id: module for module in packet.source_base_module_contracts
    }
    changed_modules = {
        module.module_id: module.mint_contract()
        for module in proposal.changed_module_contracts
    }
    if not set(changed_modules).issubset(source_base_modules):
        raise ExpertProposalContractError(
            "generalization changes an unknown capability"
        )
    if any(
        changed_modules[module_id].module_contract_id
        == source_base_modules[module_id].module_contract_id
        for module_id in changed_modules
    ):
        raise ExpertProposalContractError(
            "generalization names an unchanged module contract"
        )
    for module_id, changed_module in changed_modules.items():
        _validate_preserved_module_change(
            source_base_modules[module_id],
            changed_module,
            allow_path_reference_replacement=False,
        )
    modules = tuple(
        changed_modules.get(module_id, source_base_modules[module_id])
        for module_id in sorted(source_base_modules)
    )
    module_refs = {module.module_id: module.module_contract_id for module in modules}
    source_base_map = packet.source_base_repository_map
    repository_map = ExpertRepositoryMap.mint(
        scope_contract_id=packet.scope_contract.scope_contract_id,
        capability_nodes=tuple(
            ExpertCapabilityNode(
                capability_id=node.capability_id,
                module_contract_ref=module_refs[node.capability_id],
                owned_paths=node.owned_paths,
                task_family_bindings=node.task_family_bindings,
            )
            for node in source_base_map.capability_nodes
        ),
        dependency_edges=source_base_map.dependency_edges,
        task_adapter_boundary=source_base_map.task_adapter_boundary,
        validation_entrypoints=source_base_map.validation_entrypoints,
        architecture_invariants=source_base_map.architecture_invariants,
    )
    return repository_map, modules, ()


def _validate_preserved_module_change(
    source_base_module: ExpertModuleContract,
    changed: ExpertModuleContract,
    *,
    allow_path_reference_replacement: bool,
) -> None:
    source_base_version = source_base_module.version[1:]
    changed_version = changed.version[1:]
    if len(changed_version) < len(source_base_version) or (
        len(changed_version) == len(source_base_version)
        and changed_version <= source_base_version
    ):
        raise ExpertProposalContractError(
            "generalization must advance the changed module version"
        )
    exact_fields = (
        "purpose",
        "dependency_capability_ids",
        "incompatible_capability_ids",
        "resource_bounds",
    )
    if any(
        getattr(changed, name) != getattr(source_base_module, name)
        for name in exact_fields
    ):
        raise ExpertProposalContractError(
            "generalization changes a fixed module safety envelope"
        )
    monotonic_fields = [
        "problem_signals",
        "inputs",
        "outputs",
        "preconditions",
        "incompatibilities",
        "supporting_episode_ids",
        "known_failure_episode_ids",
    ]
    if not allow_path_reference_replacement:
        monotonic_fields.extend(("entrypoint_refs", "test_refs", "replay_refs"))
    if any(
        not set(getattr(source_base_module, name)).issubset(getattr(changed, name))
        for name in monotonic_fields
    ):
        raise ExpertProposalContractError(
            "generalization removes accumulated module safety or provenance"
        )
    if any(
        key not in changed.dependency_license_manifest
        or changed.dependency_license_manifest[key]
        != source_base_module.dependency_license_manifest[key]
        for key in source_base_module.dependency_license_manifest
    ):
        raise ExpertProposalContractError(
            "generalization removes or rewrites dependency license metadata"
        )


def expert_proposal_response_schema(
    operation_kind: ExpertCandidateOperationKind,
) -> Mapping[str, Any]:
    string_array = {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
    }
    module = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "dependency_capability_ids",
            "dependency_license_manifest",
            "entrypoint_refs",
            "incompatibilities",
            "incompatible_capability_ids",
            "inputs",
            "known_failure_episode_ids",
            "module_id",
            "outputs",
            "preconditions",
            "problem_signals",
            "purpose",
            "replay_refs",
            "resource_bounds",
            "supporting_episode_ids",
            "test_refs",
            "version",
        ],
        "properties": {
            "module_id": {"type": "string", "minLength": 1},
            "version": {"type": "string", "pattern": "^v[1-9][0-9]*$"},
            "purpose": {"type": "string", "minLength": 1},
            "problem_signals": string_array,
            "inputs": string_array,
            "outputs": string_array,
            "preconditions": string_array,
            "incompatibilities": string_array,
            "dependency_capability_ids": string_array,
            "incompatible_capability_ids": string_array,
            "resource_bounds": {"type": "object", "minProperties": 1},
            "dependency_license_manifest": {
                "type": "object",
                "minProperties": 1,
            },
            "supporting_episode_ids": string_array,
            "known_failure_episode_ids": string_array,
            "entrypoint_refs": string_array,
            "test_refs": string_array,
            "replay_refs": string_array,
        },
    }
    common_properties = {
        "summary": {"type": "string", "minLength": 1},
        "changed_paths": string_array,
        "deleted_paths": string_array,
    }
    if operation_kind is ExpertCandidateOperationKind.GENERALIZE:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "changed_module_contracts",
                "changed_paths",
                "deleted_paths",
                "summary",
            ],
            "properties": {
                **common_properties,
                "changed_module_contracts": {
                    "type": "array",
                    "minItems": 1,
                    "items": module,
                },
            },
        }
    capability_node = {
        "type": "object",
        "additionalProperties": False,
        "required": ["capability_id", "owned_paths", "task_family_bindings"],
        "properties": {
            "capability_id": {"type": "string", "minLength": 1},
            "owned_paths": string_array,
            "task_family_bindings": string_array,
        },
    }
    adapter_boundary = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "adapter_mount_path",
            "inputs",
            "interface_entrypoint_refs",
            "invariants",
            "outputs",
        ],
        "properties": {
            "adapter_mount_path": {"type": "string", "minLength": 1},
            "interface_entrypoint_refs": string_array,
            "inputs": string_array,
            "outputs": string_array,
            "invariants": string_array,
        },
    }
    topology = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "architecture_invariants",
            "capability_nodes",
            "task_adapter_boundary",
            "validation_entrypoints",
        ],
        "properties": {
            "capability_nodes": {
                "type": "array",
                "minItems": 1,
                "items": capability_node,
            },
            "task_adapter_boundary": adapter_boundary,
            "validation_entrypoints": string_array,
            "architecture_invariants": string_array,
        },
    }
    lineage = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "evidence_ids",
            "relation",
            "source_capability_ids",
            "target_capability_ids",
        ],
        "properties": {
            "source_capability_ids": string_array,
            "target_capability_ids": string_array,
            "relation": {
                "type": "string",
                "enum": ["merge", "rename", "retire", "split"],
            },
            "evidence_ids": string_array,
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "capability_lineage",
            "changed_paths",
            "deleted_paths",
            "module_contracts",
            "repository_topology",
            "summary",
        ],
        "properties": {
            **common_properties,
            "repository_topology": topology,
            "module_contracts": {
                "type": "array",
                "minItems": 1,
                "items": module,
            },
            "capability_lineage": {"type": "array", "items": lineage},
        },
    }


def expert_candidate_control_namespace(path: str) -> bool:
    source_path = PurePosixPath(path)
    control_root = PurePosixPath(EXPERT_REPOSITORY_MAP_PATH).parent
    return (
        path == EXPERT_BOOK_PATH
        or source_path == control_root
        or control_root in source_path.parents
    )
