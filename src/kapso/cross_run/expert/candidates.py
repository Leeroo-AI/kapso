"""Aggregate validation for immutable expert candidate closures."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from kapso.cross_run.agent_artifacts import (
    CODING_AGENT_WORKSPACE_DELTA_FILENAME,
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
)
from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    CandidateChangeKind,
    CodingAgentWorkspaceChangedFile,
    CodingAgentWorkspaceDelta,
    ExpertCandidateManifest,
    ExpertCandidateOperationKind,
    ExpertCandidateOperationRecord,
    ExpertCandidatePatch,
    ExpertCandidatePatchChange,
    ExpertCandidateSanitationReport,
    ExpertCandidateSanitationStatus,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertSourceTreeManifest,
    SourceFileDescriptor,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_control_paths,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.sanitation import ExpertCandidateSanitizer
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvaluator,
)
from kapso.cross_run.settings import (
    CodingAgentSettings,
    ExpertSettings,
    SanitationSettings,
)
from kapso.execution.coding_agents.operation_receipt import (
    verify_coding_agent_operation_artifacts,
)


class ExpertCandidateValidationError(ValueError):
    """An expert candidate closure is internally inconsistent."""


@dataclass(frozen=True)
class ExpertCandidateClosure:
    manifest: ExpertCandidateManifest
    trigger_packet: ExpertTriggerEvidencePacket
    trigger_decision: ExpertEvolutionTriggerDecision
    patch: ExpertCandidatePatch
    candidate_tree: ExpertSourceTreeManifest
    parent_files: tuple[SourceFileDescriptor, ...]
    repository_map: ExpertRepositoryMap
    module_contracts: tuple[ExpertModuleContract, ...]
    operation: ExpertCandidateOperationRecord
    sanitation_report: ExpertCandidateSanitationReport
    candidate_contents: Mapping[str, bytes]
    workspace_delta: CodingAgentWorkspaceDelta
    operation_artifacts: Mapping[str, bytes]
    ancestor_candidates: tuple[ExpertCandidateManifest, ...] = ()


class ExpertCandidateValidator:
    """Join every candidate record into one exact, authority-safe closure."""

    def __init__(
        self,
        settings: ExpertSettings,
        sanitation_settings: SanitationSettings,
    ):
        self.settings = settings
        self.sanitizer = ExpertCandidateSanitizer(sanitation_settings)

    def validate(self, closure: ExpertCandidateClosure) -> bytes:
        self._validate_trigger_and_parent(closure)
        candidate_files = self._validate_candidate_tree(closure)
        parent_files = self._validate_parent_tree(closure)
        self._validate_operation(closure)
        self._validate_operation_tree_delta(closure, parent_files, candidate_files)
        self._validate_patch(closure, parent_files, candidate_files)
        self._validate_sanitation(closure)
        modules = self._validate_topology(closure, candidate_files)
        self._validate_evidence_and_lineage(closure, modules)
        self._validate_ancestors(closure)
        book = self._validate_control_files_and_book(closure, modules)
        return book

    def _validate_trigger_and_parent(self, closure: ExpertCandidateClosure) -> None:
        manifest = closure.manifest
        packet = closure.trigger_packet
        decision = closure.trigger_decision
        expected_decision = ExpertTriggerEvaluator(self.settings.triggers).evaluate(
            packet
        )
        if decision != expected_decision:
            raise ExpertCandidateValidationError(
                "candidate trigger decision differs from deterministic policy"
            )
        if (
            manifest.trigger_evidence_packet_id != packet.evidence_packet_id
            or manifest.trigger_decision_id != decision.trigger_decision_id
            or decision.evidence_packet_id != packet.evidence_packet_id
            or manifest.scope_contract_id != packet.scope_contract.scope_contract_id
            or manifest.configuration_fingerprint != packet.configuration_fingerprint
            or decision.configuration_fingerprint != packet.configuration_fingerprint
        ):
            raise ExpertCandidateValidationError(
                "candidate trigger, scope, or configuration binding differs"
            )
        if not decision.candidate_required or decision.change_kind is not (
            manifest.change_kind
        ):
            raise ExpertCandidateValidationError(
                "trigger decision does not authorize this candidate class"
            )
        expected_dependencies = tuple(
            sorted(
                {
                    packet.evidence_packet_id,
                    decision.trigger_decision_id,
                    *decision.trigger_evidence_ids,
                }
            )
        )
        if manifest.source_dependency_ids != expected_dependencies:
            raise ExpertCandidateValidationError(
                "candidate source dependency closure differs from its trigger"
            )
        if packet.parent_release is None:
            if (
                manifest.parent_release_id is not None
                or manifest.parent_repository_map_ref is not None
                or manifest.parent_tree_hash != EMPTY_EXPERT_TREE_DIGEST
                or closure.parent_files
                or manifest.change_kind
                is not CandidateChangeKind.REPOSITORY_ARCHITECTURE
                or decision.reason_code != "empty_scope_bootstrap"
            ):
                raise ExpertCandidateValidationError(
                    "bootstrap candidate differs from the explicit empty parent"
                )
        elif (
            manifest.parent_release_id != packet.parent_release.release_id
            or manifest.parent_repository_map_ref
            != packet.repository_map.repository_map_id
            or manifest.parent_tree_hash != packet.parent_tree_hash
        ):
            raise ExpertCandidateValidationError(
                "candidate parent differs from the verified trigger parent"
            )

    def _validate_operation(self, closure: ExpertCandidateClosure) -> None:
        manifest = closure.manifest
        operation = closure.operation
        if (
            manifest.proposer_operation_record_id != operation.operation_record_id
            or operation.trigger_decision_id != manifest.trigger_decision_id
            or operation.trigger_evidence_packet_id
            != manifest.trigger_evidence_packet_id
            or operation.parent_tree_hash != manifest.parent_tree_hash
            or operation.ancestor_candidate_ids != manifest.ancestor_candidate_ids
            or operation.configuration_fingerprint != manifest.configuration_fingerprint
        ):
            raise ExpertCandidateValidationError(
                "candidate operation differs from the manifest authority"
            )
        if manifest.parent_release_id is None:
            expected_kind = ExpertCandidateOperationKind.BOOTSTRAP
        elif manifest.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE:
            expected_kind = ExpertCandidateOperationKind.RESTRUCTURE
        else:
            expected_kind = ExpertCandidateOperationKind.GENERALIZE
        if operation.operation_kind is not expected_kind:
            raise ExpertCandidateValidationError(
                "candidate operation kind differs from the authorized change"
            )
        if expected_kind is ExpertCandidateOperationKind.GENERALIZE:
            self._validate_agent_authority(
                operation,
                self.settings.generalizer_id,
                self.settings.generalizer_role,
                self.settings.generalizer,
            )
        else:
            self._validate_agent_authority(
                operation,
                self.settings.architect_id,
                self.settings.architect_role,
                self.settings.architect,
            )
        self._validate_operation_artifacts(closure)

    def _validate_operation_artifacts(
        self,
        closure: ExpertCandidateClosure,
    ) -> None:
        operation = closure.operation
        receipt = operation.operation_receipt
        expected_names = set(
            coding_agent_artifact_filenames(CodingAgentWorkspaceAccess.EDIT_WORKSPACE)
        )
        if set(closure.operation_artifacts) != expected_names:
            raise ExpertCandidateValidationError(
                "candidate operation artifact closure is incomplete"
            )
        if sum(len(payload) for payload in closure.operation_artifacts.values()) > (
            self.settings.agent_artifact_byte_limit
        ):
            raise ExpertCandidateValidationError(
                "candidate operation artifacts exceed the configured limit"
            )
        for name, payload in closure.operation_artifacts.items():
            if not isinstance(payload, bytes):
                raise ExpertCandidateValidationError(
                    "candidate operation artifact values must be bytes"
                )
            if receipt.artifact_checksums.get(name) != tree_or_blob_digest(payload):
                raise ExpertCandidateValidationError(
                    f"candidate operation artifact checksum differs: {name}"
                )
        input_checksums = operation.operation_preimage["input_artifact_checksums"]
        if any(
            input_checksums[name]
            != tree_or_blob_digest(closure.operation_artifacts[name])
            for name in input_checksums
        ):
            raise ExpertCandidateValidationError(
                "candidate operation preimage differs from its input artifacts"
            )
        verified = verify_coding_agent_operation_artifacts(
            operation_id=receipt.operation_id,
            workspace_access=receipt.workspace_access,
            artifact_bytes=closure.operation_artifacts,
        )
        if operation.operation_preimage["mcp_configuration_fingerprint"] != (
            verified.mcp_configuration_fingerprint
        ):
            raise ExpertCandidateValidationError(
                "candidate operation preimage differs from its MCP configuration"
            )
        invocation = verified.invocation
        expected_agent = (
            self.settings.generalizer
            if operation.operation_kind is ExpertCandidateOperationKind.GENERALIZE
            else self.settings.architect
        )
        workspace_policy = invocation["workspace_policy"]
        if (
            invocation["role"] != receipt.role
            or invocation["cli"] != receipt.cli
            or invocation["model"] != receipt.model
            or invocation["effort"] != receipt.effort
            or invocation["timeout_seconds"] != expected_agent.timeout_seconds
            or tuple(invocation["allowed_tools"]) != expected_agent.allowed_tools
            or invocation["sensitive_file_glob_scan_max_depth"]
            != self.settings.sensitive_file_glob_scan_max_depth
            or workspace_policy["access"]
            != CodingAgentWorkspaceAccess.EDIT_WORKSPACE.value
            or workspace_policy["maximum_entries"]
            != self.settings.candidate_entry_limit
            or workspace_policy["maximum_bytes"] != self.settings.candidate_byte_limit
            or workspace_policy["expected_tree_hash"]
            != closure.workspace_delta.baseline_tree_hash
        ):
            raise ExpertCandidateValidationError(
                "candidate invocation differs from configured proposer authority"
            )
        delta_payload = closure.operation_artifacts[
            CODING_AGENT_WORKSPACE_DELTA_FILENAME
        ]
        if (
            operation.workspace_delta_ref != closure.workspace_delta.workspace_delta_id
            or operation.workspace_delta_digest != tree_or_blob_digest(delta_payload)
            or delta_payload != closure.workspace_delta.to_json_bytes()
        ):
            raise ExpertCandidateValidationError(
                "candidate workspace delta differs from its durable artifact"
            )
        if (
            verified.workspace_delta != closure.workspace_delta
            or verified.final_output != operation.final_output
        ):
            raise ExpertCandidateValidationError(
                "candidate result differs from its durable output or delta"
            )

    @staticmethod
    def _validate_agent_authority(
        operation: ExpertCandidateOperationRecord,
        principal_id: str,
        role: str,
        agent: CodingAgentSettings,
    ) -> None:
        receipt = operation.operation_receipt
        if (
            receipt.principal_id != principal_id
            or receipt.role != role
            or receipt.cli != agent.cli
            or receipt.model != agent.model
            or receipt.effort != agent.effort
            or receipt.workspace_access is not CodingAgentWorkspaceAccess.EDIT_WORKSPACE
        ):
            raise ExpertCandidateValidationError(
                "candidate operation lacks configured proposer authority"
            )

    def _validate_candidate_tree(
        self,
        closure: ExpertCandidateClosure,
    ) -> dict[str, SourceFileDescriptor]:
        manifest = closure.manifest
        tree = closure.candidate_tree
        if (
            manifest.candidate_tree_ref != tree.source_tree_manifest_id
            or manifest.candidate_tree_hash != tree.tree_hash
        ):
            raise ExpertCandidateValidationError(
                "candidate manifest references another source tree"
            )
        files = {file.relative_path: file for file in tree.files}
        if (
            len(files) > self.settings.candidate_entry_limit
            or sum(file.size for file in tree.files)
            > self.settings.candidate_byte_limit
        ):
            raise ExpertCandidateValidationError(
                "candidate tree exceeds configured aggregate limits"
            )
        if set(closure.candidate_contents) != set(files):
            raise ExpertCandidateValidationError(
                "candidate bytes differ from the exact tree path closure"
            )
        for path, descriptor in files.items():
            payload = closure.candidate_contents[path]
            if not isinstance(payload, bytes):
                raise ExpertCandidateValidationError(
                    "candidate content values must be bytes"
                )
            if (
                tree_or_blob_digest(payload) != descriptor.digest
                or len(payload) != descriptor.size
            ):
                raise ExpertCandidateValidationError(
                    f"candidate bytes differ from descriptor: {path}"
                )
        return files

    @staticmethod
    def _validate_parent_tree(
        closure: ExpertCandidateClosure,
    ) -> dict[str, SourceFileDescriptor]:
        paths = tuple(file.relative_path for file in closure.parent_files)
        if paths != tuple(sorted(set(paths))):
            raise ExpertCandidateValidationError(
                "parent files must be sorted and uniquely identified"
            )
        observed_hash = (
            EMPTY_EXPERT_TREE_DIGEST
            if not closure.parent_files
            else source_tree_digest(
                {
                    file.relative_path: (file.digest, file.mode, file.size)
                    for file in closure.parent_files
                }
            )
        )
        if observed_hash != closure.manifest.parent_tree_hash:
            raise ExpertCandidateValidationError(
                "parent file descriptor differs from the verified parent tree"
            )
        if closure.trigger_packet.parent_tree_receipt is not None and (
            closure.trigger_packet.parent_tree_receipt.source_extraction_receipt.source_tree_files
            != closure.parent_files
        ):
            raise ExpertCandidateValidationError(
                "parent files differ from the source extraction receipt"
            )
        return {file.relative_path: file for file in closure.parent_files}

    @staticmethod
    def _validate_patch(
        closure: ExpertCandidateClosure,
        parent_files: Mapping[str, SourceFileDescriptor],
        candidate_files: Mapping[str, SourceFileDescriptor],
    ) -> None:
        manifest = closure.manifest
        patch = closure.patch
        if (
            manifest.patch_ref != patch.patch_id
            or manifest.patch_digest != tree_or_blob_digest(patch.to_json_bytes())
            or patch.parent_tree_hash != manifest.parent_tree_hash
            or patch.candidate_tree_hash != manifest.candidate_tree_hash
        ):
            raise ExpertCandidateValidationError(
                "candidate patch identity or tree binding differs"
            )
        expected_changes = tuple(
            ExpertCandidatePatchChange(
                relative_path=path,
                before=parent_files.get(path),
                after=candidate_files.get(path),
            )
            for path in sorted(set(parent_files) | set(candidate_files))
            if parent_files.get(path) != candidate_files.get(path)
        )
        if patch.changes != expected_changes:
            raise ExpertCandidateValidationError(
                "candidate patch does not transform parent into candidate tree"
            )

    @staticmethod
    def _validate_operation_tree_delta(
        closure: ExpertCandidateClosure,
        parent_files: Mapping[str, SourceFileDescriptor],
        candidate_files: Mapping[str, SourceFileDescriptor],
    ) -> None:
        parent_control = set(
            expert_control_paths(closure.trigger_packet.module_contracts)
        )
        candidate_control = set(expert_control_paths(closure.module_contracts))
        editable_parent = {
            path: descriptor
            for path, descriptor in parent_files.items()
            if path not in parent_control
        }
        editable_candidate = {
            path: descriptor
            for path, descriptor in candidate_files.items()
            if path not in candidate_control
        }
        if not editable_candidate:
            raise ExpertCandidateValidationError(
                "candidate operation produced no editable source tree"
            )
        edited_tree_hash = source_tree_digest(
            {
                path: (descriptor.digest, descriptor.mode, descriptor.size)
                for path, descriptor in editable_candidate.items()
            }
        )
        changed_paths = tuple(
            path
            for path in sorted(editable_candidate)
            if editable_parent.get(path) != editable_candidate[path]
        )
        deleted_paths = tuple(
            path for path in sorted(editable_parent) if path not in editable_candidate
        )
        operation = closure.operation
        workspace_receipt = operation.workspace_receipt
        editable_parent_tree_hash = (
            EMPTY_EXPERT_TREE_DIGEST
            if not editable_parent
            else source_tree_digest(
                {
                    path: (descriptor.digest, descriptor.mode, descriptor.size)
                    for path, descriptor in editable_parent.items()
                }
            )
        )
        expected_delta = CodingAgentWorkspaceDelta.mint(
            baseline_tree_hash=editable_parent_tree_hash,
            edited_tree_hash=edited_tree_hash,
            changed_files=tuple(
                CodingAgentWorkspaceChangedFile(
                    before=editable_parent.get(path),
                    after=editable_candidate[path],
                    content_base64=base64.b64encode(
                        closure.candidate_contents[path]
                    ).decode("ascii"),
                )
                for path in changed_paths
            ),
            deleted_files=tuple(editable_parent[path] for path in deleted_paths),
        )
        if (
            closure.workspace_delta != expected_delta
            or workspace_receipt.editable_parent_tree_hash != editable_parent_tree_hash
            or workspace_receipt.edited_tree_hash != edited_tree_hash
            or workspace_receipt.changed_paths != changed_paths
            or workspace_receipt.deleted_paths != deleted_paths
        ):
            raise ExpertCandidateValidationError(
                "candidate operation receipt differs from the edited tree delta"
            )

    def _validate_sanitation(self, closure: ExpertCandidateClosure) -> None:
        manifest = closure.manifest
        report = closure.sanitation_report
        if (
            closure.trigger_packet.scope_contract.sanitation_policy_ref
            != self.sanitizer.settings.policy_version
        ):
            raise ExpertCandidateValidationError(
                "candidate sanitation configuration differs from scope policy"
            )
        expected_report = self.sanitizer.scan(
            manifest.scope_contract_id,
            closure.candidate_tree,
            closure.candidate_contents,
        )
        if (
            manifest.sanitation_report_id != report.sanitation_report_id
            or report != expected_report
            or report.status is not ExpertCandidateSanitationStatus.ADMITTED
        ):
            raise ExpertCandidateValidationError(
                "candidate sanitation does not admit the exact candidate tree"
            )

    @staticmethod
    def _validate_topology(
        closure: ExpertCandidateClosure,
        candidate_files: Mapping[str, SourceFileDescriptor],
    ) -> dict[str, ExpertModuleContract]:
        manifest = closure.manifest
        repository_map = closure.repository_map
        packet = closure.trigger_packet
        module_contract_ids = tuple(
            sorted(module.module_contract_id for module in closure.module_contracts)
        )
        if len(module_contract_ids) != len(set(module_contract_ids)):
            raise ExpertCandidateValidationError(
                "candidate module contracts must be uniquely identified"
            )
        semantic_module_ids = tuple(
            module.module_id for module in closure.module_contracts
        )
        if semantic_module_ids != tuple(sorted(set(semantic_module_ids))):
            raise ExpertCandidateValidationError(
                "candidate semantic module IDs must be sorted and unique"
            )
        if (
            manifest.module_contract_refs != module_contract_ids
            or manifest.proposed_repository_map_ref != repository_map.repository_map_id
            or repository_map.scope_contract_id != manifest.scope_contract_id
        ):
            raise ExpertCandidateValidationError(
                "candidate map or module references differ from the manifest"
            )
        modules = {module.module_id: module for module in closure.module_contracts}
        nodes = {node.capability_id: node for node in repository_map.capability_nodes}
        if set(modules) != set(nodes) or any(
            nodes[module_id].module_contract_ref
            != modules[module_id].module_contract_id
            for module_id in modules
        ):
            raise ExpertCandidateValidationError(
                "candidate capability nodes and modules are not a bijection"
            )
        outgoing = {
            capability_id: tuple(
                sorted(
                    edge.target_capability_id
                    for edge in repository_map.dependency_edges
                    if edge.source_capability_id == capability_id
                )
            )
            for capability_id in nodes
        }
        if any(
            module.dependency_capability_ids != outgoing[module.module_id]
            for module in closure.module_contracts
        ):
            raise ExpertCandidateValidationError(
                "module dependencies differ from repository map edges"
            )
        for module in closure.module_contracts:
            if not set(module.incompatible_capability_ids).issubset(modules):
                raise ExpertCandidateValidationError(
                    "module incompatibility references an unknown capability"
                )
            for incompatible_id in module.incompatible_capability_ids:
                if module.module_id not in (
                    modules[incompatible_id].incompatible_capability_ids
                ):
                    raise ExpertCandidateValidationError(
                        "module incompatibility must be symmetric"
                    )
        known_families = {
            family.task_family_id
            for family in packet.scope_contract.task_family_ontology
        }
        bound_families = {
            family_id
            for node in repository_map.capability_nodes
            for family_id in node.task_family_bindings
        }
        if not bound_families.issubset(known_families) or not set(
            packet.active_task_family_ids
        ).issubset(bound_families):
            raise ExpertCandidateValidationError(
                "candidate task-family coverage differs from active scope"
            )
        ExpertCandidateValidator._validate_tree_ownership(
            repository_map,
            closure.module_contracts,
            candidate_files,
        )
        if manifest.change_kind is CandidateChangeKind.CAPABILITY:
            ExpertCandidateValidator._validate_capability_change_boundary(closure)
        return modules

    @staticmethod
    def _validate_tree_ownership(
        repository_map: ExpertRepositoryMap,
        modules: tuple[ExpertModuleContract, ...],
        candidate_files: Mapping[str, SourceFileDescriptor],
    ) -> None:
        control_paths = set(expert_control_paths(modules))
        if not control_paths.issubset(candidate_files):
            raise ExpertCandidateValidationError(
                "candidate tree omits generated expert control files"
            )
        control_root = PurePosixPath(EXPERT_REPOSITORY_MAP_PATH).parent
        book_path = PurePosixPath(EXPERT_BOOK_PATH)
        for path in candidate_files:
            source_path = PurePosixPath(path)
            if (
                source_path == book_path
                or source_path == control_root
                or control_root in source_path.parents
            ) and path not in control_paths:
                raise ExpertCandidateValidationError(
                    f"candidate tree contains undeclared expert control file: {path}"
                )
        mount = PurePosixPath(repository_map.task_adapter_boundary.adapter_mount_path)
        owned_roots = {
            node.capability_id: tuple(PurePosixPath(path) for path in node.owned_paths)
            for node in repository_map.capability_nodes
        }
        for roots in owned_roots.values():
            if any(
                root == book_path
                or root == control_root
                or root in control_root.parents
                or control_root in root.parents
                for root in roots
            ):
                raise ExpertCandidateValidationError(
                    "capability ownership overlaps generated expert controls"
                )
            if any(
                root == mount or root in mount.parents or mount in root.parents
                for root in roots
            ):
                raise ExpertCandidateValidationError(
                    "task-adapter mount overlaps expert-owned source"
                )
        if any(
            PurePosixPath(path) == mount or mount in PurePosixPath(path).parents
            for path in candidate_files
        ):
            raise ExpertCandidateValidationError(
                "candidate tree contains the external task adapter"
            )
        owners_by_path: dict[str, str] = {}
        for path in candidate_files:
            if path in control_paths:
                continue
            source_path = PurePosixPath(path)
            owners = tuple(
                capability_id
                for capability_id, roots in owned_roots.items()
                if any(
                    root == source_path or root in source_path.parents for root in roots
                )
            )
            if len(owners) != 1:
                raise ExpertCandidateValidationError(
                    f"candidate source path needs exactly one owner: {path}"
                )
            owners_by_path[path] = owners[0]
        for capability_id, roots in owned_roots.items():
            for root in roots:
                if not any(
                    owner == capability_id
                    and (
                        root == PurePosixPath(path)
                        or root in PurePosixPath(path).parents
                    )
                    for path, owner in owners_by_path.items()
                ):
                    raise ExpertCandidateValidationError(
                        f"candidate owned root is empty: {root.as_posix()}"
                    )
        paths = set(candidate_files)
        module_by_id = {module.module_id: module for module in modules}
        for capability_id, module in module_by_id.items():
            for path in (
                *module.entrypoint_refs,
                *module.test_refs,
                *module.replay_refs,
            ):
                if path not in paths or owners_by_path.get(path) != capability_id:
                    raise ExpertCandidateValidationError(
                        f"module path is missing or foreign-owned: {path}"
                    )
        for path in (
            *repository_map.validation_entrypoints,
            *repository_map.task_adapter_boundary.interface_entrypoint_refs,
        ):
            if path not in paths or path not in owners_by_path:
                raise ExpertCandidateValidationError(
                    f"repository entrypoint is missing or unowned: {path}"
                )

    @staticmethod
    def _validate_capability_change_boundary(
        closure: ExpertCandidateClosure,
    ) -> None:
        parent_map = closure.trigger_packet.repository_map
        if parent_map is None:
            raise ExpertCandidateValidationError(
                "capability candidate requires a released parent topology"
            )
        current_map = closure.repository_map
        parent_nodes = {
            node.capability_id: node for node in parent_map.capability_nodes
        }
        current_nodes = {
            node.capability_id: node for node in current_map.capability_nodes
        }
        if (
            set(parent_nodes) != set(current_nodes)
            or any(
                (
                    parent_nodes[capability_id].owned_paths,
                    parent_nodes[capability_id].task_family_bindings,
                )
                != (
                    current_nodes[capability_id].owned_paths,
                    current_nodes[capability_id].task_family_bindings,
                )
                for capability_id in current_nodes
            )
            or parent_map.dependency_edges != current_map.dependency_edges
            or parent_map.task_adapter_boundary != current_map.task_adapter_boundary
            or parent_map.validation_entrypoints != current_map.validation_entrypoints
            or parent_map.architecture_invariants != current_map.architecture_invariants
            or closure.manifest.capability_lineage
        ):
            raise ExpertCandidateValidationError(
                "capability candidate cannot change repository topology"
            )
        parent_contracts = {
            module.module_id: module
            for module in closure.trigger_packet.module_contracts
        }
        current_contracts = {
            module.module_id: module for module in closure.module_contracts
        }
        changed_capabilities = {
            module_id
            for module_id in current_contracts
            if current_contracts[module_id].module_contract_id
            != parent_contracts[module_id].module_contract_id
        }
        if not changed_capabilities:
            raise ExpertCandidateValidationError(
                "capability candidate changes no module contract"
            )
        for module_id in changed_capabilities:
            if (
                current_contracts[module_id].version
                == parent_contracts[module_id].version
            ):
                raise ExpertCandidateValidationError(
                    "changed capability contract must advance its version"
                )
        current_control = set(expert_control_paths(closure.module_contracts))
        parent_control = set(
            expert_control_paths(closure.trigger_packet.module_contracts)
        )
        for change in closure.patch.changes:
            if change.relative_path in current_control | parent_control:
                continue
            owners = {
                capability_id
                for capability_id, node in current_nodes.items()
                if any(
                    PurePosixPath(root) == PurePosixPath(change.relative_path)
                    or PurePosixPath(root)
                    in PurePosixPath(change.relative_path).parents
                    for root in node.owned_paths
                )
            }
            if not owners or not owners.issubset(changed_capabilities):
                raise ExpertCandidateValidationError(
                    "capability candidate edits an unchanged capability"
                )

    @staticmethod
    def _validate_evidence_and_lineage(
        closure: ExpertCandidateClosure,
        modules: Mapping[str, ExpertModuleContract],
    ) -> None:
        packet = closure.trigger_packet
        known_evidence = {
            *(episode.episode_id for episode in packet.episodes),
            *(claim.revision_id for claim in packet.claims),
            *(
                observation.observation_id
                for observation in packet.trigger_observations
            ),
            *packet.proof_reference_ids,
            *closure.trigger_decision.trigger_evidence_ids,
        }
        episode_ids = {episode.episode_id for episode in packet.episodes}
        for module in modules.values():
            if not (
                set(module.supporting_episode_ids)
                | set(module.known_failure_episode_ids)
            ).issubset(episode_ids):
                raise ExpertCandidateValidationError(
                    "module evidence leaves the trigger episode closure"
                )
        parent_capabilities = (
            set()
            if packet.repository_map is None
            else {node.capability_id for node in packet.repository_map.capability_nodes}
        )
        candidate_capabilities = set(modules)
        removed = parent_capabilities - candidate_capabilities
        new = candidate_capabilities - parent_capabilities
        lineage_sources: list[str] = []
        lineage_targets: list[str] = []
        for lineage in closure.manifest.capability_lineage:
            if (
                not set(lineage.source_capability_ids).issubset(removed)
                or not set(lineage.target_capability_ids).issubset(new)
                or not set(lineage.evidence_ids).issubset(known_evidence)
            ):
                raise ExpertCandidateValidationError(
                    "candidate capability lineage leaves its topology or evidence"
                )
            lineage_sources.extend(lineage.source_capability_ids)
            lineage_targets.extend(lineage.target_capability_ids)
        if (
            set(lineage_sources) != removed
            or len(lineage_sources) != len(set(lineage_sources))
            or len(lineage_targets) != len(set(lineage_targets))
        ):
            raise ExpertCandidateValidationError(
                "candidate capability lineage is incomplete or ambiguous"
            )
        parent_contracts = {
            module.module_id: module for module in packet.module_contracts
        }
        for capability_id in parent_capabilities & candidate_capabilities:
            parent = parent_contracts[capability_id]
            current = modules[capability_id]
            if (
                parent.module_contract_id != current.module_contract_id
                and parent.version == current.version
            ):
                raise ExpertCandidateValidationError(
                    "preserved capability contract change must advance its version"
                )

    def _validate_ancestors(self, closure: ExpertCandidateClosure) -> None:
        manifest = closure.manifest
        ancestor_ids = tuple(
            ancestor.candidate_id for ancestor in closure.ancestor_candidates
        )
        if ancestor_ids != manifest.ancestor_candidate_ids:
            raise ExpertCandidateValidationError(
                "candidate ancestor closure differs from its manifest"
            )
        if len(ancestor_ids) > self.settings.triggers.maximum_ancestor_candidates:
            raise ExpertCandidateValidationError(
                "candidate ancestor closure exceeds configured limit"
            )
        if any(
            ancestor.scope_contract_id != manifest.scope_contract_id
            or ancestor.parent_tree_hash != manifest.parent_tree_hash
            or ancestor.candidate_id == manifest.candidate_id
            for ancestor in closure.ancestor_candidates
        ):
            raise ExpertCandidateValidationError(
                "candidate ancestor is incompatible with this proposal"
            )

    @staticmethod
    def _validate_control_files_and_book(
        closure: ExpertCandidateClosure,
        modules: Mapping[str, ExpertModuleContract],
    ) -> bytes:
        contents = closure.candidate_contents
        repository_map = closure.repository_map
        expected_book = compile_expert_semantic_book(
            closure.trigger_packet.scope_contract,
            repository_map,
            closure.module_contracts,
        )
        if (
            contents[EXPERT_BOOK_PATH] != expected_book
            or closure.manifest.semantic_book_digest
            != expert_semantic_book_digest(expected_book)
            or contents[EXPERT_REPOSITORY_MAP_PATH] != repository_map.to_json_bytes()
        ):
            raise ExpertCandidateValidationError(
                "candidate semantic book or repository map control file differs"
            )
        for module in modules.values():
            path = expert_module_contract_path(module.module_contract_id)
            if contents[path] != module.to_json_bytes():
                raise ExpertCandidateValidationError(
                    "candidate module control file differs from its contract"
                )
        return expected_book
