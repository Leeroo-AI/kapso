from __future__ import annotations

import json
from dataclasses import replace

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
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.agent_artifacts import (
    CODING_AGENT_ARTIFACT_FILENAMES,
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
    final_output = (
        json.dumps(
            {
                "changed_paths": tuple(sorted(editable_contents)),
                "deleted_paths": (),
                "summary": "Bootstrapped the smallest shared execution capability.",
            },
            indent=2,
        )
        + "\n"
    )
    operation_preimage = {
        "ancestor_candidate_ids": (),
        "configuration_fingerprint": packet.configuration_fingerprint,
        "operation_kind": ExpertCandidateOperationKind.BOOTSTRAP.value,
        "parent_tree_hash": EMPTY_EXPERT_TREE_DIGEST,
        "trigger_decision_id": decision.trigger_decision_id,
        "trigger_evidence_packet_id": packet.evidence_packet_id,
    }
    operation_id = (
        "agent_call_"
        + tree_or_blob_digest(canonical_json_bytes(operation_preimage))[7:39]
    )
    edited_files = file_descriptors(editable_contents)
    edited_tree_hash = descriptor_tree_hash(edited_files)
    declared_changed_paths = tuple(sorted(editable_contents))
    operation_receipt = CodingAgentOperationReceipt.mint(
        operation_id=operation_id,
        principal_id=configured.architect_id,
        role=configured.architect_role,
        cli=configured.architect.cli,
        model=configured.architect.model,
        effort=configured.architect.effort,
        artifact_checksums={
            filename: (
                tree_or_blob_digest(final_output.encode("utf-8"))
                if filename == "final.json"
                else digest(f"bootstrap-{filename}")
            )
            for filename in CODING_AGENT_ARTIFACT_FILENAMES
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
        operation_preimage=operation_preimage,
        operation_receipt=operation_receipt,
        workspace_receipt=workspace_receipt,
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
    manifest = ExpertCandidateManifest.mint(
        scope_contract_id=packet.scope_contract.scope_contract_id,
        change_kind=CandidateChangeKind.REPOSITORY_ARCHITECTURE,
        parent_release_id=None,
        parent_repository_map_ref=None,
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        trigger_decision_id=(
            content_id("fixture", {"foreign": "trigger"})
            if foreign_trigger
            else decision.trigger_decision_id
        ),
        trigger_evidence_packet_id=packet.evidence_packet_id,
        patch_ref=patch.patch_id,
        patch_digest=tree_or_blob_digest(patch.to_json_bytes()),
        candidate_tree_ref=candidate_tree.source_tree_manifest_id,
        candidate_tree_hash=candidate_tree_hash,
        configuration_fingerprint=packet.configuration_fingerprint,
        module_contract_refs=(module.module_contract_id,),
        proposed_repository_map_ref=repository_map.repository_map_id,
        semantic_book_digest=expert_semantic_book_digest(expected_book),
        proposer_operation_record_id=operation.operation_record_id,
        source_dependency_ids=source_dependencies,
        ancestor_candidate_ids=(),
        capability_lineage=(),
        sanitation_report_id=sanitation.sanitation_report_id,
    )
    return ExpertCandidateClosure(
        manifest=manifest,
        trigger_packet=packet,
        trigger_decision=decision,
        patch=patch,
        candidate_tree=candidate_tree,
        parent_files=(),
        repository_map=repository_map,
        module_contracts=(module,),
        operation=operation,
        sanitation_report=sanitation,
        candidate_contents=contents,
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
        ExpertCandidateValidator._validate_tree_ownership(
            repository_map,
            closure.module_contracts,
            candidate_files,
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
        match="semantic module IDs must be sorted and unique",
    ):
        ExpertCandidateValidator(expert_settings(), sanitation_settings()).validate(
            duplicate_closure
        )


def test_candidate_recomputes_trigger_decision_authority():
    closure = bootstrap_candidate_closure()
    decision_payload = closure.trigger_decision.to_dict()
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
            replace(closure, trigger_decision=forged_decision)
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
