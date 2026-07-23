from __future__ import annotations

import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    CandidateChangeKind,
    CodingAgentOperationReceipt,
    CodingAgentWorkspaceChangedFile,
    CodingAgentWorkspaceDelta,
    ContractValidationError,
    ExpertCandidateDerivationKind,
    ExpertCandidateManifest,
    ExpertCandidateOperationKind,
    ExpertCandidateOperationRecord,
    ExpertCandidatePatch,
    ExpertCandidatePatchChange,
    ExpertCandidateSanitationReport,
    ExpertCandidateSanitationStatus,
    ExpertCandidateWorkspaceReceipt,
    ExpertCapabilityNode,
    ExpertModuleContract,
    ExpertProposerAuthority,
    ExpertRepositoryMap,
    ExpertSourceTreeManifest,
    ExpertTaskAdapterBoundary,
    SourceFileDescriptor,
)
from kapso.cross_run.expert import (
    ExpertCandidateClosure,
    ExpertCandidateValidationError,
    ExpertCandidateValidator,
    ExpertCandidateSanitizer,
    ExpertTriggerEvaluator,
    ExpertEvolutionTriggerDecision,
    compile_expert_semantic_book,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    expert_module_contract_path,
)
from kapso.cross_run.expert.candidate_context import (
    project_agent_candidate_validation_context,
)
from kapso.cross_run.expert.candidate_derivations import (
    ExpertAgentProposalDerivation,
    ExpertAgentProposalDerivationRecord,
)
from kapso.cross_run.expert.proposal_contract import (
    EXPERT_PROPOSAL_CONTRACT_VERSION,
    build_expert_proposal_packet,
    build_expert_proposal_prompt,
    expert_proposal_packet_digest,
    expert_proposal_response_schema,
)
from kapso.cross_run.expert.topology import (
    ExpertTopologyValidationError,
    validate_expert_repository_topology,
    validate_expert_tree_ownership,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
    coding_agent_returned_artifact_filenames,
)
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallRequest,
    CodingAgentCallResult,
    CodingAgentWorkspacePolicy,
    coding_agent_mcp_configuration_fingerprint,
    coding_agent_invocation_bytes,
    coding_agent_response_schema_bytes,
)
from kapso.execution.coding_agents.operation_receipt import (
    verify_coding_agent_operation_artifacts,
)
from test_expert_triggers import trigger_packet, trigger_settings

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def digest(value: str) -> str:
    return tree_or_blob_digest(value.encode("utf-8"))


def expert_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).expert


def sanitation_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).sanitation


def file_descriptors(contents: dict[str, bytes]) -> tuple[SourceFileDescriptor, ...]:
    return tuple(
        SourceFileDescriptor(
            relative_path=path,
            digest=tree_or_blob_digest(contents[path]),
            mode="100644",
            size=len(contents[path]),
        )
        for path in sorted(contents)
    )


def descriptor_tree_hash(files: tuple[SourceFileDescriptor, ...]) -> str:
    return source_tree_digest(
        {file.relative_path: (file.digest, file.mode, file.size) for file in files}
    )


def bootstrap_candidate_closure(
    *,
    missing_test_ref: bool = False,
    unowned_file: bool = False,
    manual_book: bool = False,
    rejected_sanitation: bool = False,
    forged_patch_tree_hash: bool = False,
    incomplete_patch: bool = False,
    foreign_trigger: bool = False,
    forged_workspace_tree: bool = False,
    workspace_access: CodingAgentWorkspaceAccess = (
        CodingAgentWorkspaceAccess.EDIT_WORKSPACE
    ),
) -> ExpertCandidateClosure:
    settings = trigger_settings()
    packet = trigger_packet(settings=settings, bootstrap=True)
    decision = ExpertTriggerEvaluator(settings).evaluate(packet)
    test_ref = "tests/missing.py" if missing_test_ref else "tests/test_execution.py"
    module = ExpertModuleContract.mint(
        module_id="shared.execution",
        version="v1",
        purpose="Execute a task through one provenance-bound interface.",
        problem_signals=("Task implementations duplicate execution control.",),
        inputs=(),
        outputs=("validated artifact",),
        preconditions=(),
        incompatibilities=(),
        dependency_capability_ids=(),
        incompatible_capability_ids=(),
        resource_bounds={"concurrency": 1},
        dependency_license_manifest={"license": "MIT"},
        supporting_episode_ids=(),
        known_failure_episode_ids=(),
        entrypoint_refs=("src/execution.py",),
        test_refs=(test_ref,),
        replay_refs=(),
    )
    repository_map = ExpertRepositoryMap.mint(
        scope_contract_id=packet.scope_contract.scope_contract_id,
        capability_nodes=(
            ExpertCapabilityNode(
                capability_id=module.module_id,
                module_contract_ref=module.module_contract_id,
                owned_paths=("src/execution.py", "tests/test_execution.py"),
                task_family_bindings=("language_model_post_training",),
            ),
        ),
        dependency_edges=(),
        task_adapter_boundary=ExpertTaskAdapterBoundary(
            adapter_mount_path=".kapso/task-adapter",
            interface_entrypoint_refs=("src/execution.py",),
            inputs=("task contract",),
            outputs=("validated artifact",),
            invariants=("The task adapter remains external and read-only.",),
        ),
        validation_entrypoints=("tests/test_execution.py",),
        architecture_invariants=("No task identity appears in generic defaults.",),
    )
    expected_book = compile_expert_semantic_book(
        packet.scope_contract,
        repository_map,
        (module,),
    )
    editable_contents = {
        "src/execution.py": b"def execute(task):\n    return task.run()\n",
        "tests/test_execution.py": b"def test_execute():\n    assert True\n",
    }
    if unowned_file:
        editable_contents["notes.txt"] = b"unowned"
    contents = {
        **editable_contents,
        EXPERT_BOOK_PATH: b"# manual\n" if manual_book else expected_book,
        EXPERT_REPOSITORY_MAP_PATH: repository_map.to_json_bytes(),
        expert_module_contract_path(module.module_contract_id): module.to_json_bytes(),
    }
    files = file_descriptors(contents)
    candidate_tree_hash = descriptor_tree_hash(files)
    candidate_tree = ExpertSourceTreeManifest.mint(
        tree_hash=candidate_tree_hash,
        files=files,
    )
    patch_tree_hash = (
        digest("forged-patch-candidate-tree")
        if forged_patch_tree_hash
        else candidate_tree_hash
    )
    patch = ExpertCandidatePatch.mint(
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        candidate_tree_hash=patch_tree_hash,
        changes=tuple(
            ExpertCandidatePatchChange(
                relative_path=file.relative_path,
                before=None,
                after=file,
            )
            for file in (files[:-1] if incomplete_patch else files)
        ),
    )
    configured = expert_settings()
    module_proposal = module.to_dict()
    del module_proposal["module_contract_id"]
    final_output = (
        json.dumps(
            {
                "capability_lineage": (),
                "changed_paths": tuple(sorted(editable_contents)),
                "deleted_paths": (),
                "module_contracts": (module_proposal,),
                "repository_topology": {
                    "architecture_invariants": (
                        "No task identity appears in generic defaults.",
                    ),
                    "capability_nodes": (
                        {
                            "capability_id": module.module_id,
                            "owned_paths": (
                                "src/execution.py",
                                "tests/test_execution.py",
                            ),
                            "task_family_bindings": ("language_model_post_training",),
                        },
                    ),
                    "task_adapter_boundary": (
                        repository_map.task_adapter_boundary.to_dict()
                    ),
                    "validation_entrypoints": ("tests/test_execution.py",),
                },
                "summary": "Bootstrapped the smallest shared execution capability.",
            },
            indent=2,
        )
        + "\n"
    )
    edited_files = file_descriptors(editable_contents)
    edited_tree_hash = descriptor_tree_hash(edited_files)
    declared_changed_paths = tuple(sorted(editable_contents))
    workspace_delta = CodingAgentWorkspaceDelta.mint(
        baseline_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        edited_tree_hash=edited_tree_hash,
        changed_files=tuple(
            CodingAgentWorkspaceChangedFile(
                before=None,
                after=file,
                content_base64=base64.b64encode(
                    editable_contents[file.relative_path]
                ).decode("ascii"),
            )
            for file in edited_files
        ),
        deleted_files=(),
    )
    workspace_policy = (
        CodingAgentWorkspacePolicy.edit_workspace(
            expected_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
            maximum_entries=configured.candidate_entry_limit,
            maximum_bytes=configured.candidate_byte_limit,
        )
        if workspace_access is CodingAgentWorkspaceAccess.EDIT_WORKSPACE
        else CodingAgentWorkspacePolicy.read_only()
    )
    proposal_packet = build_expert_proposal_packet(
        packet=packet,
        decision=decision,
        operation_kind=ExpertCandidateOperationKind.BOOTSTRAP,
        editable_parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        maximum_entries=configured.candidate_entry_limit,
        maximum_bytes=configured.candidate_byte_limit,
        ancestor_inputs=(),
    )
    prompt = build_expert_proposal_prompt(
        ExpertCandidateOperationKind.BOOTSTRAP,
        proposal_packet,
    )
    provisional_request = CodingAgentCallRequest(
        operation_id="agent_call_" + "0" * 32,
        role=configured.architect_role,
        cli=configured.architect.cli,
        model=configured.architect.model,
        prompt=prompt,
        workspace="/tmp/kapso-candidate-fixture",
        workspace_policy=workspace_policy,
        timeout_seconds=configured.architect.timeout_seconds,
        effort=configured.architect.effort,
        allowed_tools=configured.architect.allowed_tools,
    )
    response_schema = expert_proposal_response_schema(
        ExpertCandidateOperationKind.BOOTSTRAP
    )
    input_artifacts = {
        "invocation.json": coding_agent_invocation_bytes(
            provisional_request,
            sensitive_file_glob_scan_max_depth=(
                configured.sensitive_file_glob_scan_max_depth
            ),
        ),
        "prior_knowledge.json": b"null\n",
        "prompt.txt": provisional_request.prompt.encode("utf-8"),
        "response_schema.json": coding_agent_response_schema_bytes(response_schema),
    }
    proposer_authority = ExpertProposerAuthority.mint(
        principal_id=configured.architect_id,
        role=configured.architect_role,
        cli=configured.architect.cli,
        model=configured.architect.model,
        effort=configured.architect.effort,
        timeout_seconds=configured.architect.timeout_seconds,
        allowed_tools=configured.architect.allowed_tools,
        workspace_access=CodingAgentWorkspaceAccess.EDIT_WORKSPACE,
        workspace_maximum_entries=configured.candidate_entry_limit,
        workspace_maximum_bytes=configured.candidate_byte_limit,
        sensitive_file_glob_scan_max_depth=(
            configured.sensitive_file_glob_scan_max_depth
        ),
    )
    operation_preimage = {
        "ancestor_candidate_ids": (),
        "configuration_fingerprint": packet.configuration_fingerprint,
        "input_artifact_checksums": {
            name: tree_or_blob_digest(payload)
            for name, payload in sorted(input_artifacts.items())
        },
        "mcp_configuration_fingerprint": (
            coding_agent_mcp_configuration_fingerprint(None)
        ),
        "operation_kind": ExpertCandidateOperationKind.BOOTSTRAP.value,
        "parent_tree_hash": EMPTY_EXPERT_TREE_DIGEST,
        "principal_id": configured.architect_id,
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
    call_request = replace(provisional_request, operation_id=operation_id)
    artifact_directory = f"/tmp/kapso-agent-artifacts/{operation_id}"
    returned_names = coding_agent_returned_artifact_filenames(workspace_access)
    call_result = CodingAgentCallResult(
        output=final_output,
        duration_seconds=1.0,
        cost_usd=None,
        final_output_digest=tree_or_blob_digest(final_output.encode("utf-8")),
        workspace_delta_digest=(
            tree_or_blob_digest(workspace_delta.to_json_bytes())
            if workspace_access is CodingAgentWorkspaceAccess.EDIT_WORKSPACE
            else None
        ),
        input_tokens=1,
        output_tokens=1,
        artifacts=tuple(f"{artifact_directory}/{name}" for name in returned_names),
    )
    operation_artifacts = {
        **input_artifacts,
        "mcp_config.json": (
            json.dumps({"mcpServers": {}}, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "stdout.txt": b"completed\n",
        "stderr.txt": b"",
        "final.json": final_output.encode("utf-8"),
        "mcp_audit.jsonl": b"",
        "result.json": call_result.to_json_bytes(),
    }
    if workspace_access is CodingAgentWorkspaceAccess.EDIT_WORKSPACE:
        operation_artifacts["workspace-delta.json"] = workspace_delta.to_json_bytes()
    assert input_artifacts["invocation.json"] == coding_agent_invocation_bytes(
        call_request,
        sensitive_file_glob_scan_max_depth=(
            configured.sensitive_file_glob_scan_max_depth
        ),
    )
    operation_receipt = CodingAgentOperationReceipt.mint(
        operation_id=operation_id,
        principal_id=configured.architect_id,
        role=configured.architect_role,
        cli=configured.architect.cli,
        model=configured.architect.model,
        effort=configured.architect.effort,
        workspace_access=workspace_access,
        artifact_checksums={
            filename: tree_or_blob_digest(payload)
            for filename, payload in operation_artifacts.items()
        },
    )
    workspace_receipt = ExpertCandidateWorkspaceReceipt.mint(
        operation_receipt_id=operation_receipt.operation_receipt_id,
        operation_id=operation_id,
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        editable_parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        edited_tree_hash=(
            digest("forged-workspace-tree")
            if forged_workspace_tree
            else edited_tree_hash
        ),
        changed_paths=declared_changed_paths,
        deleted_paths=(),
    )
    operation = ExpertCandidateOperationRecord.mint(
        operation_kind=ExpertCandidateOperationKind.BOOTSTRAP,
        trigger_decision_id=decision.trigger_decision_id,
        trigger_evidence_packet_id=packet.evidence_packet_id,
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        ancestor_candidate_ids=(),
        configuration_fingerprint=packet.configuration_fingerprint,
        proposer_authority=proposer_authority,
        operation_preimage=operation_preimage,
        operation_receipt=operation_receipt,
        workspace_receipt=workspace_receipt,
        workspace_delta_ref=workspace_delta.workspace_delta_id,
        workspace_delta_digest=tree_or_blob_digest(workspace_delta.to_json_bytes()),
        final_output=final_output,
    )
    sanitation = ExpertCandidateSanitizer(sanitation_settings()).scan(
        packet.scope_contract.scope_contract_id,
        candidate_tree,
        contents,
    )
    if rejected_sanitation:
        sanitation = ExpertCandidateSanitationReport.mint(
            scope_contract_id=packet.scope_contract.scope_contract_id,
            candidate_tree_hash=candidate_tree_hash,
            policy_version="attacker.policy.v1",
            policy_fingerprint=digest("attacker-policy"),
            scanner_version="attacker.scanner.v1",
            status=ExpertCandidateSanitationStatus.ADMITTED,
            scanned_files=files,
            findings=(),
        )
    source_dependencies = tuple(
        sorted(
            {
                packet.evidence_packet_id,
                decision.trigger_decision_id,
                *decision.trigger_evidence_ids,
            }
        )
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
        ancestor_candidate_ids=(),
        origin_principal_ids=(operation.proposer_authority.principal_id,),
        source_dependency_ids=source_dependencies,
        operation_artifact_checksums={
            name: tree_or_blob_digest(payload)
            for name, payload in sorted(operation_artifacts.items())
        },
    )
    derivation = ExpertAgentProposalDerivation(
        record=derivation_record,
        trigger_packet=packet,
        trigger_decision=decision,
        operation=operation,
        workspace_delta=workspace_delta,
        operation_artifacts=operation_artifacts,
        ancestor_inputs=(),
    )
    manifest = ExpertCandidateManifest.mint(
        scope_contract_id=packet.scope_contract.scope_contract_id,
        change_kind=CandidateChangeKind.REPOSITORY_ARCHITECTURE,
        parent_release_id=None,
        parent_repository_map_ref=None,
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        derivation_kind=ExpertCandidateDerivationKind.AGENT_PROPOSAL,
        derivation_ref=(
            content_id(
                "expert-agent-proposal-derivation",
                {"foreign": "trigger"},
            )
            if foreign_trigger
            else derivation_record.derivation_id
        ),
        validation_context_ref=validation_context.validation_context_id,
        patch_ref=patch.patch_id,
        patch_digest=tree_or_blob_digest(patch.to_json_bytes()),
        candidate_tree_ref=candidate_tree.source_tree_manifest_id,
        candidate_tree_hash=candidate_tree_hash,
        configuration_fingerprint=packet.configuration_fingerprint,
        module_contract_refs=(module.module_contract_id,),
        proposed_repository_map_ref=repository_map.repository_map_id,
        semantic_book_digest=expert_semantic_book_digest(expected_book),
        source_dependency_ids=source_dependencies,
        ancestor_candidate_ids=(),
        capability_lineage=(),
        sanitation_report_id=sanitation.sanitation_report_id,
    )
    return ExpertCandidateClosure(
        manifest=manifest,
        validation_context=validation_context,
        patch=patch,
        candidate_tree=candidate_tree,
        parent_files=(),
        repository_map=repository_map,
        module_contracts=(module,),
        derivation=derivation,
        sanitation_report=sanitation,
        candidate_contents=contents,
    )


def replace_agent_derivation(closure, **changes):
    current = closure.derivation
    packet = changes.get("trigger_packet", current.trigger_packet)
    decision = changes.get("trigger_decision", current.trigger_decision)
    operation = changes.get("operation", current.operation)
    workspace_delta = changes.get("workspace_delta", current.workspace_delta)
    operation_artifacts = changes.get(
        "operation_artifacts",
        current.operation_artifacts,
    )
    ancestor_inputs = changes.get("ancestor_inputs", current.ancestor_inputs)
    record = ExpertAgentProposalDerivationRecord.mint(
        trigger_evidence_packet_id=packet.evidence_packet_id,
        trigger_decision_id=decision.trigger_decision_id,
        operation_record_id=operation.operation_record_id,
        workspace_delta_id=workspace_delta.workspace_delta_id,
        ancestor_candidate_ids=tuple(
            ancestor.manifest.candidate_id for ancestor in ancestor_inputs
        ),
        origin_principal_ids=(operation.proposer_authority.principal_id,),
        source_dependency_ids=current.record.source_dependency_ids,
        operation_artifact_checksums={
            name: tree_or_blob_digest(payload)
            for name, payload in sorted(operation_artifacts.items())
        },
    )
    derivation = ExpertAgentProposalDerivation(
        record=record,
        trigger_packet=packet,
        trigger_decision=decision,
        operation=operation,
        workspace_delta=workspace_delta,
        operation_artifacts=operation_artifacts,
        ancestor_inputs=ancestor_inputs,
    )
    manifest_payload = closure.manifest.to_dict()
    manifest_payload.pop("candidate_id")
    manifest_payload["derivation_ref"] = record.derivation_id
    return replace(
        closure,
        manifest=ExpertCandidateManifest.mint(**manifest_payload),
        derivation=derivation,
    )


def test_bootstrap_candidate_closure_is_complete_and_deterministic():
    closure = bootstrap_candidate_closure()

    book = ExpertCandidateValidator(expert_settings(), sanitation_settings()).validate(
        closure
    )

    assert book == closure.candidate_contents[EXPERT_BOOK_PATH]
    assert tree_or_blob_digest(book) == closure.manifest.semantic_book_digest


def test_candidate_rejects_agent_owned_generated_control_namespace():
    closure = bootstrap_candidate_closure()
    node = closure.repository_map.capability_nodes[0]
    repository_map = ExpertRepositoryMap.mint(
        scope_contract_id=closure.repository_map.scope_contract_id,
        capability_nodes=(
            ExpertCapabilityNode(
                capability_id=node.capability_id,
                module_contract_ref=node.module_contract_ref,
                owned_paths=tuple(
                    sorted((*node.owned_paths, ".kapso/expert/agent-note.txt"))
                ),
                task_family_bindings=node.task_family_bindings,
            ),
        ),
        dependency_edges=closure.repository_map.dependency_edges,
        task_adapter_boundary=closure.repository_map.task_adapter_boundary,
        validation_entrypoints=closure.repository_map.validation_entrypoints,
        architecture_invariants=closure.repository_map.architecture_invariants,
    )
    foreign_control = SourceFileDescriptor(
        relative_path=".kapso/expert/agent-note.txt",
        digest=digest("agent-authored-control"),
        mode="100644",
        size=len("agent-authored-control"),
    )
    candidate_files = {
        **{file.relative_path: file for file in closure.candidate_tree.files},
        foreign_control.relative_path: foreign_control,
    }

    with pytest.raises(
        ExpertCandidateValidationError,
        match="undeclared expert control",
    ):
        validate_expert_tree_ownership(
            repository_map,
            closure.module_contracts,
            candidate_files,
            validation_error_type=ExpertCandidateValidationError,
        )


def test_public_topology_validator_returns_exact_capability_index():
    closure = bootstrap_candidate_closure()

    modules = validate_expert_repository_topology(
        closure.repository_map,
        closure.module_contracts,
    )

    assert modules == {
        closure.module_contracts[0].module_id: closure.module_contracts[0]
    }


def test_public_topology_validator_uses_domain_neutral_error():
    closure = bootstrap_candidate_closure()
    node = closure.repository_map.capability_nodes[0]
    mismatched_map = ExpertRepositoryMap.mint(
        scope_contract_id=closure.repository_map.scope_contract_id,
        capability_nodes=(
            ExpertCapabilityNode(
                capability_id=node.capability_id,
                module_contract_ref=content_id(
                    "test-expert-module-contract",
                    {"label": "mismatched"},
                ),
                owned_paths=node.owned_paths,
                task_family_bindings=node.task_family_bindings,
            ),
        ),
        dependency_edges=closure.repository_map.dependency_edges,
        task_adapter_boundary=closure.repository_map.task_adapter_boundary,
        validation_entrypoints=closure.repository_map.validation_entrypoints,
        architecture_invariants=closure.repository_map.architecture_invariants,
    )

    with pytest.raises(
        ExpertTopologyValidationError,
        match="capability nodes and modules are not a bijection",
    ):
        validate_expert_repository_topology(
            mismatched_map,
            closure.module_contracts,
        )


def test_candidate_rejects_duplicate_semantic_module_ids():
    closure = bootstrap_candidate_closure()
    module = closure.module_contracts[0]
    duplicate = ExpertModuleContract.mint(
        **{
            key: value
            for key, value in module.to_dict().items()
            if key not in {"module_contract_id", "purpose"}
        },
        purpose="A second contract illegally claiming the same capability.",
    )
    duplicate_closure = replace(
        closure,
        module_contracts=(module, duplicate),
    )

    with pytest.raises(
        ExpertCandidateValidationError,
        match="semantic topology differs from the agent proposal",
    ):
        ExpertCandidateValidator(expert_settings(), sanitation_settings()).validate(
            duplicate_closure
        )


def test_candidate_recomputes_trigger_decision_authority():
    closure = bootstrap_candidate_closure()
    decision_payload = closure.derivation.trigger_decision.to_dict()
    decision_payload.pop("trigger_decision_id")
    decision_payload["knowledge_snapshot_id"] = content_id(
        "fixture", {"foreign": "snapshot"}
    )
    decision_payload["policy_version"] = "attacker.trigger.policy"
    forged_decision = ExpertEvolutionTriggerDecision.mint(**decision_payload)

    with pytest.raises(
        ExpertCandidateValidationError,
        match="differs from deterministic policy",
    ):
        ExpertCandidateValidator(expert_settings(), sanitation_settings()).validate(
            replace_agent_derivation(closure, trigger_decision=forged_decision)
        )


def test_candidate_operation_requires_edit_workspace_authority():
    with pytest.raises(
        ContractValidationError,
        match="workspace delta differs from its operation receipt",
    ):
        bootstrap_candidate_closure(
            workspace_access=CodingAgentWorkspaceAccess.READ_ONLY
        )


def test_candidate_requires_scope_sanitation_policy_resolution():
    closure = bootstrap_candidate_closure()
    different_policy = replace(
        sanitation_settings(),
        policy_version="different.sanitation.policy",
    )

    with pytest.raises(
        ExpertCandidateValidationError,
        match="differs from scope policy",
    ):
        ExpertCandidateValidator(expert_settings(), different_policy).validate(closure)


def test_candidate_requires_exact_self_contained_operation_artifacts():
    closure = bootstrap_candidate_closure()
    artifacts = dict(closure.derivation.operation_artifacts)
    artifacts["prompt.txt"] += b"tampered"

    with pytest.raises(
        ExpertCandidateValidationError,
        match="artifact checksum differs",
    ):
        ExpertCandidateValidator(expert_settings(), sanitation_settings()).validate(
            replace_agent_derivation(closure, operation_artifacts=artifacts)
        )


def test_candidate_requires_durable_workspace_delta_bytes():
    closure = bootstrap_candidate_closure()
    artifacts = dict(closure.derivation.operation_artifacts)
    artifacts["workspace-delta.json"] += b"\n"

    with pytest.raises(
        ExpertCandidateValidationError,
        match="artifact checksum differs",
    ):
        ExpertCandidateValidator(expert_settings(), sanitation_settings()).validate(
            replace_agent_derivation(closure, operation_artifacts=artifacts)
        )


def test_embedded_operation_artifact_verifier_rejects_invalid_result_contract():
    closure = bootstrap_candidate_closure()
    artifacts = dict(closure.derivation.operation_artifacts)
    artifacts["result.json"] = b"{}\n"

    with pytest.raises(ValueError, match="fields mismatch"):
        verify_coding_agent_operation_artifacts(
            operation_id=closure.derivation.operation.operation_receipt.operation_id,
            workspace_access=CodingAgentWorkspaceAccess.EDIT_WORKSPACE,
            artifact_bytes=artifacts,
        )


def test_embedded_operation_artifact_verifier_rejects_unknown_policy_version():
    closure = bootstrap_candidate_closure()
    artifacts = dict(closure.derivation.operation_artifacts)
    invocation = json.loads(artifacts["invocation.json"])
    invocation["mcp_audit_policy_version"] = "kapso.mcp_audit.unknown"
    artifacts["invocation.json"] = (
        json.dumps(invocation, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    with pytest.raises(ValueError, match="unrecognized policy version"):
        verify_coding_agent_operation_artifacts(
            operation_id=closure.derivation.operation.operation_receipt.operation_id,
            workspace_access=CodingAgentWorkspaceAccess.EDIT_WORKSPACE,
            artifact_bytes=artifacts,
        )


@pytest.mark.parametrize(
    "limited_settings",
    (
        replace(expert_settings(), candidate_entry_limit=1),
        replace(expert_settings(), candidate_byte_limit=1),
        replace(expert_settings(), agent_artifact_byte_limit=1),
    ),
)
def test_candidate_enforces_aggregate_source_and_artifact_limits(limited_settings):
    closure = bootstrap_candidate_closure()

    with pytest.raises(
        ExpertCandidateValidationError,
        match="aggregate limits|operation artifacts exceed",
    ):
        ExpertCandidateValidator(limited_settings, sanitation_settings()).validate(
            closure
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"forged_patch_tree_hash": True}, "patch identity or tree binding"),
        ({"incomplete_patch": True}, "does not transform"),
        ({"foreign_trigger": True}, "trigger, scope, or configuration"),
        ({"rejected_sanitation": True}, "sanitation does not admit"),
        ({"missing_test_ref": True}, "module path is missing"),
        ({"unowned_file": True}, "needs exactly one owner"),
        ({"manual_book": True}, "semantic book"),
        ({"forged_workspace_tree": True}, "edited tree delta"),
    ),
)
def test_candidate_aggregate_rejects_inconsistent_closures(updates, message):
    closure = bootstrap_candidate_closure(**updates)

    with pytest.raises(ExpertCandidateValidationError, match=message):
        ExpertCandidateValidator(expert_settings(), sanitation_settings()).validate(
            closure
        )


def two_capability_boundary_fixture(*, changed_ids, edited_capability_id):
    closure = bootstrap_candidate_closure()
    parent_first = closure.module_contracts[0]
    second_payload = parent_first.to_dict()
    del second_payload["module_contract_id"]
    second_payload.update(
        {
            "module_id": "shared.other",
            "purpose": "Provide another independent capability.",
            "entrypoint_refs": ("src/other.py",),
            "test_refs": ("src/other_test.py",),
            "replay_refs": (),
        }
    )
    parent_second = ExpertModuleContract.mint(**second_payload)
    parent_modules = (parent_first, parent_second)
    current_modules = []
    for module in parent_modules:
        if module.module_id not in changed_ids:
            current_modules.append(module)
            continue
        payload = module.to_dict()
        del payload["module_contract_id"]
        payload["version"] = "v2"
        payload["problem_signals"] = tuple(
            sorted((*payload["problem_signals"], "A newly observed difficulty."))
        )
        current_modules.append(ExpertModuleContract.mint(**payload))
    current_modules = tuple(current_modules)
    parent_nodes = (
        ExpertCapabilityNode(
            capability_id=parent_first.module_id,
            module_contract_ref=parent_first.module_contract_id,
            owned_paths=("src/execution.py", "tests/test_execution.py"),
            task_family_bindings=("language_model_post_training",),
        ),
        ExpertCapabilityNode(
            capability_id=parent_second.module_id,
            module_contract_ref=parent_second.module_contract_id,
            owned_paths=("src/other.py", "src/other_test.py"),
            task_family_bindings=("language_model_post_training",),
        ),
    )
    current_by_id = {module.module_id: module for module in current_modules}
    current_nodes = tuple(
        replace(
            node,
            module_contract_ref=current_by_id[node.capability_id].module_contract_id,
        )
        for node in parent_nodes
    )
    map_fields = {
        "scope_contract_id": closure.repository_map.scope_contract_id,
        "dependency_edges": (),
        "task_adapter_boundary": closure.repository_map.task_adapter_boundary,
        "validation_entrypoints": closure.repository_map.validation_entrypoints,
        "architecture_invariants": closure.repository_map.architecture_invariants,
    }
    parent_map = ExpertRepositoryMap.mint(
        capability_nodes=parent_nodes,
        **map_fields,
    )
    current_map = ExpertRepositoryMap.mint(
        capability_nodes=current_nodes,
        **map_fields,
    )
    edited_path = (
        "src/execution.py"
        if edited_capability_id == parent_first.module_id
        else "src/other.py"
    )
    before = SourceFileDescriptor(
        relative_path=edited_path,
        digest=digest("before"),
        mode="100644",
        size=6,
    )
    after = replace(before, digest=digest("after"), size=5)
    patch = ExpertCandidatePatch.mint(
        parent_tree_hash=digest("parent-two-capability-tree"),
        candidate_tree_hash=digest("candidate-two-capability-tree"),
        changes=(
            ExpertCandidatePatchChange(
                relative_path=edited_path,
                before=before,
                after=after,
            ),
        ),
    )
    return SimpleNamespace(
        validation_context=SimpleNamespace(
            parent_repository_map=parent_map,
            parent_module_contracts=parent_modules,
        ),
        derivation=SimpleNamespace(
            trigger_packet=SimpleNamespace(trigger_observations=()),
            trigger_decision=SimpleNamespace(trigger_evidence_ids=()),
        ),
        repository_map=current_map,
        module_contracts=current_modules,
        manifest=SimpleNamespace(capability_lineage=()),
        patch=patch,
    )


def test_capability_change_requires_an_owned_edit_for_every_changed_contract():
    closure = two_capability_boundary_fixture(
        changed_ids={"shared.execution", "shared.other"},
        edited_capability_id="shared.execution",
    )

    with pytest.raises(
        ExpertCandidateValidationError,
        match="must own an edited or deleted source path",
    ):
        ExpertCandidateValidator._validate_capability_change_boundary(closure)


def test_capability_change_stays_within_triggering_observation_scope():
    closure = two_capability_boundary_fixture(
        changed_ids={"shared.other"},
        edited_capability_id="shared.other",
    )
    closure.derivation.trigger_packet.trigger_observations = (
        SimpleNamespace(
            observation_id="selected-observation",
            affected_capability_ids=("shared.execution",),
        ),
    )
    closure.derivation.trigger_decision.trigger_evidence_ids = ("selected-observation",)

    with pytest.raises(
        ExpertCandidateValidationError,
        match="leaves the triggering observation scope",
    ):
        ExpertCandidateValidator._validate_capability_change_boundary(closure)
