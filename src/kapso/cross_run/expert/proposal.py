"""Shared coding-agent execution and deterministic expert candidate sealing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from kapso.cross_run.agent_artifacts import CodingAgentWorkspaceAccess
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
    ExpertCandidateOperationKind,
    ExpertCandidateOperationRecord,
    ExpertCandidatePatch,
    ExpertCandidatePatchChange,
    ExpertCandidateWorkspaceReceipt,
    ExpertModuleContract,
    ExpertProposerAuthority,
    ExpertRepositoryMap,
    ExpertSourceTreeManifest,
    SourceFileDescriptor,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.candidates import ExpertCandidateClosure
from kapso.cross_run.expert.candidate_context import (
    project_agent_candidate_validation_context,
)
from kapso.cross_run.expert.candidate_derivations import (
    ExpertAgentProposalDerivation,
    ExpertAgentProposalDerivationRecord,
)
from kapso.cross_run.expert.proposal_contract import (
    EXPERT_PROPOSAL_CONTRACT_VERSION,
    ExpertCandidateAncestorInput,
    ExpertProposalContractError,
    build_expert_proposal_packet,
    build_expert_proposal_prompt,
    derive_expert_proposal_topology,
    expert_candidate_source_dependency_ids,
    expert_candidate_control_namespace,
    expert_candidate_operation_kind,
    expert_proposal_packet_digest,
    expert_proposal_response_schema,
    mint_expert_candidate_ancestor_input,
    parse_expert_proposal,
    validate_expert_prior_knowledge,
)
from kapso.cross_run.expert.store import ExpertCandidateStore, StoredExpertCandidate
from kapso.cross_run.expert.triggers import (
    ExpertEvolutionTriggerDecision,
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvaluator,
)
from kapso.cross_run.expert.workspace import (
    ExpertCandidateWorkspaceManager,
    PreparedExpertCandidateWorkspace,
)
from kapso.cross_run.github.materializer import MaterializedArtifact
from kapso.cross_run.knowledge.access import PriorKnowledgeAccessMaterialization
from kapso.cross_run.settings import (
    CodingAgentSettings,
    ExpertSettings,
)
from kapso.execution.coding_agents.operation_receipt import (
    SealedCodingAgentOperation,
    seal_coding_agent_operation,
)
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    CodingAgentCallRunner,
    CodingAgentWorkspacePolicy,
    coding_agent_invocation_bytes,
    coding_agent_mcp_configuration_fingerprint,
    coding_agent_response_schema_bytes,
)
from kapso.execution.coding_agents.workspace_delta import (
    CodingAgentWorkspaceSnapshot,
    inspect_coding_agent_workspace_descriptor,
    reconstruct_edited_workspace,
)

_PROVISIONAL_OPERATION_ID = "agent_call_" + "0" * 32


@dataclass(frozen=True)
class ExpertCandidateProposalResult:
    """One quarantined stored candidate and its live call result."""

    stored_candidate: StoredExpertCandidate
    call_result: CodingAgentCallResult


class ExpertCandidateProposalEngine:
    """Run one authorized proposer and seal its exact candidate closure."""

    def __init__(
        self,
        *,
        settings: ExpertSettings,
        runner: CodingAgentCallRunner,
        workspace_manager: ExpertCandidateWorkspaceManager,
        candidate_store: ExpertCandidateStore,
    ) -> None:
        if (
            workspace_manager.settings != settings
            or candidate_store.validator.settings != settings
        ):
            raise ExpertProposalContractError(
                "expert proposal components use different settings"
            )
        self.settings = settings
        self.runner = runner
        self.workspace_manager = workspace_manager
        self.candidate_store = candidate_store

    def propose_architecture(
        self,
        *,
        packet: ExpertTriggerEvidencePacket,
        decision: ExpertEvolutionTriggerDecision,
        materialized_parent: MaterializedArtifact | None,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None,
        ancestor_candidate_ids: tuple[str, ...],
    ) -> ExpertCandidateProposalResult:
        return self._propose(
            packet=packet,
            decision=decision,
            materialized_parent=materialized_parent,
            prior_knowledge=prior_knowledge,
            ancestor_candidate_ids=ancestor_candidate_ids,
            allowed_operation_kinds=(
                ExpertCandidateOperationKind.BOOTSTRAP,
                ExpertCandidateOperationKind.RESTRUCTURE,
            ),
        )

    def propose_generalization(
        self,
        *,
        packet: ExpertTriggerEvidencePacket,
        decision: ExpertEvolutionTriggerDecision,
        materialized_parent: MaterializedArtifact,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None,
        ancestor_candidate_ids: tuple[str, ...],
    ) -> ExpertCandidateProposalResult:
        return self._propose(
            packet=packet,
            decision=decision,
            materialized_parent=materialized_parent,
            prior_knowledge=prior_knowledge,
            ancestor_candidate_ids=ancestor_candidate_ids,
            allowed_operation_kinds=(ExpertCandidateOperationKind.GENERALIZE,),
        )

    def _propose(
        self,
        *,
        packet: ExpertTriggerEvidencePacket,
        decision: ExpertEvolutionTriggerDecision,
        materialized_parent: MaterializedArtifact | None,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None,
        ancestor_candidate_ids: tuple[str, ...],
        allowed_operation_kinds: tuple[ExpertCandidateOperationKind, ...],
    ) -> ExpertCandidateProposalResult:
        expected_decision = ExpertTriggerEvaluator(self.settings.triggers).evaluate(
            packet
        )
        if decision != expected_decision:
            raise ExpertProposalContractError(
                "expert proposal decision differs from deterministic policy"
            )
        operation_kind = expert_candidate_operation_kind(packet, decision)
        if operation_kind not in allowed_operation_kinds:
            raise ExpertProposalContractError(
                "expert proposer role cannot perform this operation kind"
            )
        validate_expert_prior_knowledge(packet, prior_knowledge)
        ancestor_inputs = self._ancestor_inputs(packet, ancestor_candidate_ids)
        lease = self.workspace_manager.lease(
            trigger_packet=packet,
            materialized_parent=materialized_parent,
        )
        with lease as prepared:
            proposal_packet = build_expert_proposal_packet(
                packet=packet,
                decision=decision,
                operation_kind=operation_kind,
                editable_parent_tree_hash=prepared.editable_snapshot.tree_hash,
                maximum_entries=self.settings.candidate_entry_limit,
                maximum_bytes=self.settings.candidate_byte_limit,
                ancestor_inputs=ancestor_inputs,
            )
            response_schema = expert_proposal_response_schema(operation_kind)
            prompt = build_expert_proposal_prompt(operation_kind, proposal_packet)
            if len(prompt.encode("utf-8")) > self.settings.agent_artifact_byte_limit:
                raise ExpertProposalContractError(
                    "expert proposal prompt exceeds the configured artifact limit"
                )
            (
                request,
                operation_preimage,
                proposer_authority,
                agent,
            ) = self._request(
                packet=packet,
                decision=decision,
                operation_kind=operation_kind,
                prepared=prepared,
                prompt=prompt,
                response_schema=response_schema,
                prior_knowledge=prior_knowledge,
                ancestor_inputs=ancestor_inputs,
                proposal_packet=proposal_packet,
            )
            call_result = self.runner.run(
                request,
                response_schema,
                workspace_authority_descriptor=(lease.workspace_authority_descriptor),
            )
            lease.validate()
            sealed = seal_coding_agent_operation(
                request=request,
                response_schema=response_schema,
                principal_id=proposer_authority.principal_id,
                agent=agent,
                sensitive_file_glob_scan_max_depth=(
                    self.settings.sensitive_file_glob_scan_max_depth
                ),
                result=call_result,
            )
            closure = self._assemble_closure(
                packet=packet,
                decision=decision,
                operation_kind=operation_kind,
                prepared=prepared,
                operation_preimage=operation_preimage,
                proposer_authority=proposer_authority,
                sealed=sealed,
                ancestor_inputs=ancestor_inputs,
                prior_knowledge=prior_knowledge,
            )
            live_snapshot = inspect_coding_agent_workspace_descriptor(
                lease.workspace_authority_descriptor,
                maximum_entries=self.settings.candidate_entry_limit,
                maximum_bytes=self.settings.candidate_byte_limit,
            )
            expected_snapshot = reconstruct_edited_workspace(
                prepared.editable_snapshot,
                closure.derivation.workspace_delta,
            )
            if live_snapshot != expected_snapshot:
                raise ExpertProposalContractError(
                    "expert workspace differs from its durable agent delta"
                )
            lease.validate()
        stored = self.candidate_store.persist(closure)
        return ExpertCandidateProposalResult(
            stored_candidate=stored,
            call_result=call_result,
        )

    def _ancestor_inputs(
        self,
        packet: ExpertTriggerEvidencePacket,
        candidate_ids: tuple[str, ...],
    ) -> tuple[ExpertCandidateAncestorInput, ...]:
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ExpertProposalContractError(
                "expert ancestor candidate IDs must be sorted and unique"
            )
        if len(candidate_ids) > self.settings.triggers.maximum_ancestor_candidates:
            raise ExpertProposalContractError(
                "expert ancestor candidate selection exceeds its configured limit"
            )
        stored_candidates = tuple(
            self.candidate_store.read(candidate_id) for candidate_id in candidate_ids
        )
        inputs = tuple(
            mint_expert_candidate_ancestor_input(
                manifest=stored.closure.manifest,
                scope_contract=stored.closure.validation_context.scope_contract,
                patch=stored.closure.patch,
                candidate_tree=stored.closure.candidate_tree,
                repository_map=stored.closure.repository_map,
                module_contracts=stored.closure.module_contracts,
                sanitation_report=stored.closure.sanitation_report,
                candidate_contents=stored.closure.candidate_contents,
            )
            for stored in stored_candidates
        )
        if any(
            ancestor.manifest.scope_contract_id
            != packet.scope_contract.scope_contract_id
            or ancestor.manifest.parent_tree_hash != packet.parent_tree_hash
            for ancestor in inputs
        ):
            raise ExpertProposalContractError(
                "expert ancestor candidate differs from the proposal authority"
            )
        return inputs

    def _request(
        self,
        *,
        packet: ExpertTriggerEvidencePacket,
        decision: ExpertEvolutionTriggerDecision,
        operation_kind: ExpertCandidateOperationKind,
        prepared: PreparedExpertCandidateWorkspace,
        prompt: str,
        response_schema: Mapping,
        prior_knowledge: PriorKnowledgeAccessMaterialization | None,
        ancestor_inputs: tuple[ExpertCandidateAncestorInput, ...],
        proposal_packet: Mapping,
    ) -> tuple[
        CodingAgentCallRequest,
        Mapping,
        ExpertProposerAuthority,
        CodingAgentSettings,
    ]:
        if operation_kind is ExpertCandidateOperationKind.GENERALIZE:
            principal_id = self.settings.generalizer_id
            role = self.settings.generalizer_role
            agent = self.settings.generalizer
        else:
            principal_id = self.settings.architect_id
            role = self.settings.architect_role
            agent = self.settings.architect
        proposer_authority = ExpertProposerAuthority.mint(
            principal_id=principal_id,
            role=role,
            cli=agent.cli,
            model=agent.model,
            effort=agent.effort,
            timeout_seconds=agent.timeout_seconds,
            allowed_tools=agent.allowed_tools,
            workspace_access=CodingAgentWorkspaceAccess.EDIT_WORKSPACE,
            workspace_maximum_entries=self.settings.candidate_entry_limit,
            workspace_maximum_bytes=self.settings.candidate_byte_limit,
            sensitive_file_glob_scan_max_depth=(
                self.settings.sensitive_file_glob_scan_max_depth
            ),
        )
        workspace_policy = CodingAgentWorkspacePolicy.edit_workspace(
            expected_tree_hash=prepared.editable_snapshot.tree_hash,
            maximum_entries=self.settings.candidate_entry_limit,
            maximum_bytes=self.settings.candidate_byte_limit,
        )
        provisional_request = CodingAgentCallRequest(
            operation_id=_PROVISIONAL_OPERATION_ID,
            role=role,
            cli=agent.cli,
            model=agent.model,
            prompt=prompt,
            workspace=str(prepared.path),
            workspace_policy=workspace_policy,
            timeout_seconds=agent.timeout_seconds,
            effort=agent.effort,
            allowed_tools=agent.allowed_tools,
            prior_knowledge=prior_knowledge,
        )
        input_artifacts = {
            "invocation.json": coding_agent_invocation_bytes(
                provisional_request,
                sensitive_file_glob_scan_max_depth=(
                    self.settings.sensitive_file_glob_scan_max_depth
                ),
            ),
            "prior_knowledge.json": (
                b"null\n"
                if prior_knowledge is None
                else prior_knowledge.to_json_bytes()
            ),
            "prompt.txt": prompt.encode("utf-8"),
            "response_schema.json": coding_agent_response_schema_bytes(response_schema),
        }
        operation_preimage = {
            "ancestor_candidate_ids": tuple(
                ancestor.manifest.candidate_id for ancestor in ancestor_inputs
            ),
            "configuration_fingerprint": packet.configuration_fingerprint,
            "input_artifact_checksums": {
                name: tree_or_blob_digest(payload)
                for name, payload in sorted(input_artifacts.items())
            },
            "mcp_configuration_fingerprint": (
                coding_agent_mcp_configuration_fingerprint(prior_knowledge)
            ),
            "operation_kind": operation_kind.value,
            "parent_tree_hash": prepared.parent_tree_hash,
            "principal_id": principal_id,
            "proposer_authority_id": proposer_authority.authority_id,
            "proposal_contract_version": EXPERT_PROPOSAL_CONTRACT_VERSION,
            "proposal_packet_digest": expert_proposal_packet_digest(proposal_packet),
            "trigger_decision_id": decision.trigger_decision_id,
            "trigger_evidence_packet_id": packet.evidence_packet_id,
        }
        operation_id = (
            "agent_call_"
            + tree_or_blob_digest(canonical_json_bytes(operation_preimage))[7:39]
        )
        request = replace(provisional_request, operation_id=operation_id)
        if input_artifacts["invocation.json"] != coding_agent_invocation_bytes(
            request,
            sensitive_file_glob_scan_max_depth=(
                self.settings.sensitive_file_glob_scan_max_depth
            ),
        ):
            raise ExpertProposalContractError(
                "expert operation identity changes its invocation preimage"
            )
        return request, operation_preimage, proposer_authority, agent

    def _assemble_closure(
        self,
        *,
        packet: ExpertTriggerEvidencePacket,
        decision: ExpertEvolutionTriggerDecision,
        operation_kind: ExpertCandidateOperationKind,
        prepared: PreparedExpertCandidateWorkspace,
        operation_preimage: Mapping,
        proposer_authority: ExpertProposerAuthority,
        sealed: SealedCodingAgentOperation,
        ancestor_inputs: tuple[ExpertCandidateAncestorInput, ...],
        prior_knowledge: PriorKnowledgeAccessMaterialization | None,
    ) -> ExpertCandidateClosure:
        workspace_delta = sealed.workspace_delta
        if workspace_delta is None:
            raise ExpertProposalContractError(
                "expert proposer did not return an editable workspace delta"
            )
        changed_paths = tuple(
            change.relative_path for change in workspace_delta.changed_files
        )
        deleted_paths = tuple(
            file.relative_path for file in workspace_delta.deleted_files
        )
        proposal = parse_expert_proposal(operation_kind, sealed.final_output)
        if (
            proposal.changed_paths != changed_paths
            or proposal.deleted_paths != deleted_paths
        ):
            raise ExpertProposalContractError(
                "expert proposal path declarations differ from its workspace delta"
            )
        edited_snapshot = reconstruct_edited_workspace(
            prepared.editable_snapshot,
            workspace_delta,
        )
        if any(
            expert_candidate_control_namespace(file.descriptor.relative_path)
            for file in edited_snapshot.files
        ):
            raise ExpertProposalContractError(
                "expert proposer authored a generated control path"
            )
        repository_map, modules, lineage = derive_expert_proposal_topology(
            packet=packet,
            operation_kind=operation_kind,
            proposal=proposal,
        )
        candidate_tree, candidate_contents, book = self._candidate_tree(
            packet,
            edited_snapshot,
            repository_map,
            modules,
        )
        parent_files = {file.relative_path: file for file in prepared.parent_files}
        candidate_files = {file.relative_path: file for file in candidate_tree.files}
        patch = ExpertCandidatePatch.mint(
            parent_tree_hash=prepared.parent_tree_hash,
            candidate_tree_hash=candidate_tree.tree_hash,
            changes=tuple(
                ExpertCandidatePatchChange(
                    relative_path=path,
                    before=parent_files.get(path),
                    after=candidate_files.get(path),
                )
                for path in sorted(set(parent_files) | set(candidate_files))
                if parent_files.get(path) != candidate_files.get(path)
            ),
        )
        workspace_receipt = ExpertCandidateWorkspaceReceipt.mint(
            operation_receipt_id=sealed.receipt.operation_receipt_id,
            operation_id=sealed.receipt.operation_id,
            parent_tree_hash=prepared.parent_tree_hash,
            editable_parent_tree_hash=prepared.editable_snapshot.tree_hash,
            edited_tree_hash=edited_snapshot.tree_hash,
            changed_paths=changed_paths,
            deleted_paths=deleted_paths,
        )
        delta_payload = workspace_delta.to_json_bytes()
        operation = ExpertCandidateOperationRecord.mint(
            operation_kind=operation_kind,
            trigger_decision_id=decision.trigger_decision_id,
            trigger_evidence_packet_id=packet.evidence_packet_id,
            parent_tree_hash=prepared.parent_tree_hash,
            ancestor_candidate_ids=tuple(
                ancestor.manifest.candidate_id for ancestor in ancestor_inputs
            ),
            configuration_fingerprint=packet.configuration_fingerprint,
            proposer_authority=proposer_authority,
            operation_preimage=operation_preimage,
            operation_receipt=sealed.receipt,
            workspace_receipt=workspace_receipt,
            workspace_delta_ref=workspace_delta.workspace_delta_id,
            workspace_delta_digest=tree_or_blob_digest(delta_payload),
            final_output=sealed.final_output,
        )
        sanitation = self.candidate_store.validator.sanitizer.scan(
            packet.scope_contract.scope_contract_id,
            candidate_tree,
            candidate_contents,
        )
        source_dependencies = expert_candidate_source_dependency_ids(
            packet,
            decision,
            prior_knowledge,
        )
        change_kind = (
            CandidateChangeKind.CAPABILITY
            if operation_kind is ExpertCandidateOperationKind.GENERALIZE
            else CandidateChangeKind.REPOSITORY_ARCHITECTURE
        )
        validation_context = project_agent_candidate_validation_context(
            packet=packet,
            decision=decision,
        )
        derivation_record = ExpertAgentProposalDerivationRecord.mint(
            trigger_evidence_packet_id=packet.evidence_packet_id,
            trigger_decision_id=decision.trigger_decision_id,
            operation_record_id=operation.operation_record_id,
            workspace_delta_id=workspace_delta.workspace_delta_id,
            ancestor_candidate_ids=tuple(
                ancestor.manifest.candidate_id for ancestor in ancestor_inputs
            ),
            origin_principal_ids=(operation.proposer_authority.principal_id,),
            source_dependency_ids=source_dependencies,
            operation_artifact_checksums={
                name: tree_or_blob_digest(payload)
                for name, payload in sorted(sealed.artifact_bytes.items())
            },
        )
        derivation = ExpertAgentProposalDerivation(
            record=derivation_record,
            trigger_packet=packet,
            trigger_decision=decision,
            operation=operation,
            workspace_delta=workspace_delta,
            operation_artifacts=sealed.artifact_bytes,
            ancestor_inputs=ancestor_inputs,
        )
        manifest = ExpertCandidateManifest.mint(
            scope_contract_id=packet.scope_contract.scope_contract_id,
            change_kind=change_kind,
            parent_release_id=(
                None
                if packet.parent_release is None
                else packet.parent_release.release_id
            ),
            parent_repository_map_ref=(
                None
                if packet.repository_map is None
                else packet.repository_map.repository_map_id
            ),
            parent_tree_hash=prepared.parent_tree_hash,
            derivation_kind=ExpertCandidateDerivationKind.AGENT_PROPOSAL,
            derivation_ref=derivation_record.derivation_id,
            validation_context_ref=validation_context.validation_context_id,
            patch_ref=patch.patch_id,
            patch_digest=tree_or_blob_digest(patch.to_json_bytes()),
            candidate_tree_ref=candidate_tree.source_tree_manifest_id,
            candidate_tree_hash=candidate_tree.tree_hash,
            configuration_fingerprint=packet.configuration_fingerprint,
            module_contract_refs=tuple(
                sorted(module.module_contract_id for module in modules)
            ),
            proposed_repository_map_ref=repository_map.repository_map_id,
            semantic_book_digest=expert_semantic_book_digest(book),
            source_dependency_ids=source_dependencies,
            ancestor_candidate_ids=tuple(
                ancestor.manifest.candidate_id for ancestor in ancestor_inputs
            ),
            capability_lineage=lineage,
            sanitation_report_id=sanitation.sanitation_report_id,
        )
        return ExpertCandidateClosure(
            manifest=manifest,
            validation_context=validation_context,
            patch=patch,
            candidate_tree=candidate_tree,
            parent_files=prepared.parent_files,
            repository_map=repository_map,
            module_contracts=modules,
            derivation=derivation,
            sanitation_report=sanitation,
            candidate_contents=candidate_contents,
        )

    def _candidate_tree(
        self,
        packet: ExpertTriggerEvidencePacket,
        edited_snapshot: CodingAgentWorkspaceSnapshot,
        repository_map: ExpertRepositoryMap,
        modules: tuple[ExpertModuleContract, ...],
    ) -> tuple[ExpertSourceTreeManifest, dict[str, bytes], bytes]:
        book = compile_expert_semantic_book(
            packet.scope_contract,
            repository_map,
            modules,
        )
        contents = {
            file.descriptor.relative_path: file.content
            for file in edited_snapshot.files
        }
        controls = {
            EXPERT_BOOK_PATH: book,
            EXPERT_REPOSITORY_MAP_PATH: repository_map.to_json_bytes(),
            **{
                expert_module_contract_path(module.module_contract_id): (
                    module.to_json_bytes()
                )
                for module in modules
            },
        }
        contents.update(controls)
        descriptors = tuple(
            SourceFileDescriptor(
                relative_path=path,
                digest=tree_or_blob_digest(payload),
                mode=(
                    next(
                        file.descriptor.mode
                        for file in edited_snapshot.files
                        if file.descriptor.relative_path == path
                    )
                    if path not in controls
                    else "100644"
                ),
                size=len(payload),
            )
            for path, payload in sorted(contents.items())
        )
        tree_hash = source_tree_digest(
            {
                file.relative_path: (file.digest, file.mode, file.size)
                for file in descriptors
            }
        )
        return (
            ExpertSourceTreeManifest.mint(tree_hash=tree_hash, files=descriptors),
            contents,
            book,
        )
