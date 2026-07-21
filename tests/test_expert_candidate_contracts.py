from dataclasses import replace

import pytest

from kapso.cross_run.canonical import (
    CanonicalizationError,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    EMPTY_EXPERT_TREE_DIGEST,
    CandidateChangeKind,
    ContractValidationError,
    ExpertCandidateManifest,
    ExpertCandidateOperationRecord,
    ExpertCandidatePatch,
    ExpertCandidateSanitationFinding,
    ExpertCandidateSanitationReport,
    ExpertCandidateSanitationStatus,
    ExpertBaseReleaseManifest,
    ExpertCapabilityLineage,
    ExpertCapabilityLineageRelation,
    ExpertCapabilityNode,
    ExpertDependencyEdge,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertSanitationSeverity,
    ExpertSourceTreeManifest,
    IdentityConflictError,
    SourceFileDescriptor,
)
from test_cross_run_contracts import build_records


def digest(value: str) -> str:
    return tree_or_blob_digest(value.encode("utf-8"))


def record(record_type):
    return next(item for item in build_records() if isinstance(item, record_type))


def test_module_contract_allows_honest_empty_conditions_but_requires_value():
    module = record(ExpertModuleContract)
    minimal = ExpertModuleContract.mint(
        **{
            key: value
            for key, value in module.to_dict().items()
            if key
            not in {
                "module_contract_id",
                "inputs",
                "preconditions",
                "incompatibilities",
                "replay_refs",
            }
        },
        inputs=(),
        preconditions=(),
        incompatibilities=(),
        replay_refs=(),
    )

    assert minimal.inputs == ()
    assert minimal.replay_refs == ()
    with pytest.raises(ContractValidationError, match="problem_signals"):
        ExpertModuleContract.mint(
            **{
                key: value
                for key, value in minimal.to_dict().items()
                if key not in {"module_contract_id", "problem_signals"}
            },
            problem_signals=(),
        )


def test_repository_map_rejects_prefix_ownership_and_noncanonical_edges():
    repository_map = record(ExpertRepositoryMap)
    module_ref = repository_map.capability_nodes[0].module_contract_ref
    prefix_nodes = (
        ExpertCapabilityNode(
            capability_id="capability.parent",
            module_contract_ref=module_ref,
            owned_paths=("src/shared",),
            task_family_bindings=(),
        ),
        ExpertCapabilityNode(
            capability_id="capability.child",
            module_contract_ref=module_ref,
            owned_paths=("src/shared/child",),
            task_family_bindings=(),
        ),
    )
    prefix_nodes = tuple(sorted(prefix_nodes, key=lambda node: node.capability_id))

    with pytest.raises(IdentityConflictError, match="prefix-disjoint"):
        ExpertRepositoryMap.mint(
            scope_contract_id=repository_map.scope_contract_id,
            capability_nodes=prefix_nodes,
            dependency_edges=(),
            task_adapter_boundary=repository_map.task_adapter_boundary,
            validation_entrypoints=repository_map.validation_entrypoints,
            architecture_invariants=repository_map.architecture_invariants,
        )

    disjoint_nodes = tuple(
        ExpertCapabilityNode(
            capability_id=capability_id,
            module_contract_ref=module_ref,
            owned_paths=(f"src/{capability_id[-1]}",),
            task_family_bindings=(),
        )
        for capability_id in ("capability.a", "capability.b", "capability.c")
    )
    with pytest.raises(ContractValidationError, match="edges must be sorted"):
        ExpertRepositoryMap.mint(
            scope_contract_id=repository_map.scope_contract_id,
            capability_nodes=disjoint_nodes,
            dependency_edges=(
                ExpertDependencyEdge("capability.b", "capability.c"),
                ExpertDependencyEdge("capability.a", "capability.c"),
            ),
            task_adapter_boundary=repository_map.task_adapter_boundary,
            validation_entrypoints=repository_map.validation_entrypoints,
            architecture_invariants=repository_map.architecture_invariants,
        )


def test_exact_tree_patch_and_operation_records_reject_substitution():
    source_tree = record(ExpertSourceTreeManifest)
    patch = record(ExpertCandidatePatch)
    operation = record(ExpertCandidateOperationRecord)

    with pytest.raises(ContractValidationError, match="tree hash differs"):
        replace(source_tree, tree_hash=digest("substituted-tree"))
    with pytest.raises(ContractValidationError, match="non-empty"):
        replace(patch, changes=())
    with pytest.raises(ContractValidationError, match="final output differs"):
        replace(operation, final_output=operation.final_output + "\n")
    with pytest.raises(ContractValidationError, match="preimage differs"):
        replace(operation, operation_preimage={"different": "input"})
    with pytest.raises(CanonicalizationError):
        replace(
            operation.workspace_receipt,
            edited_tree_hash=digest("substituted-edit-tree"),
        )


def test_source_tree_rejects_file_directory_collision():
    files = (
        SourceFileDescriptor(
            relative_path="src/node",
            digest=digest("node-file"),
            mode="100644",
            size=len("node-file"),
        ),
        SourceFileDescriptor(
            relative_path="src/node/child.py",
            digest=digest("child-file"),
            mode="100644",
            size=len("child-file"),
        ),
    )
    tree_hash = source_tree_digest(
        {file.relative_path: (file.digest, file.mode, file.size) for file in files}
    )

    with pytest.raises(ContractValidationError, match="file/directory collision"):
        ExpertSourceTreeManifest.mint(tree_hash=tree_hash, files=files)


def test_candidate_parent_is_optional_only_as_one_complete_pair():
    candidate = record(ExpertCandidateManifest)

    with pytest.raises(ContractValidationError, match="must appear together"):
        replace(candidate, parent_release_id=content_id("fixture", {"parent": 1}))
    assert candidate.parent_release_id is None

    release = record(ExpertBaseReleaseManifest)
    repository_map = record(ExpertRepositoryMap)
    non_bootstrap = ExpertCandidateManifest.mint(
        **{
            key: value
            for key, value in candidate.to_dict().items()
            if key
            not in {
                "candidate_id",
                "change_kind",
                "parent_release_id",
                "parent_repository_map_ref",
                "parent_tree_hash",
            }
        },
        change_kind=CandidateChangeKind.CAPABILITY,
        parent_release_id=release.release_id,
        parent_repository_map_ref=repository_map.repository_map_id,
        parent_tree_hash=digest("released-parent-tree"),
    )
    assert non_bootstrap.parent_release_id == release.release_id

    with pytest.raises(ContractValidationError, match="bootstrap"):
        ExpertCandidateManifest.mint(
            **{
                key: value
                for key, value in candidate.to_dict().items()
                if key not in {"candidate_id", "change_kind"}
            },
            change_kind=CandidateChangeKind.CAPABILITY,
        )

    legacy_payload = candidate.to_dict()
    legacy_payload["validation_attempt_refs"] = ("validation/legacy",)
    with pytest.raises(ContractValidationError, match="unknown"):
        ExpertCandidateManifest.from_dict(legacy_payload)


def test_sanitation_status_and_capability_retirement_are_typed():
    report = record(ExpertCandidateSanitationReport)
    blocking = ExpertCandidateSanitationFinding(
        code="credential_material",
        relative_path=report.scanned_files[0].relative_path,
        evidence_digest=digest("finding"),
        severity=ExpertSanitationSeverity.BLOCKING,
    )
    with pytest.raises(ContractValidationError, match="blocking findings"):
        ExpertCandidateSanitationReport.mint(
            scope_contract_id=report.scope_contract_id,
            candidate_tree_hash=report.candidate_tree_hash,
            policy_version=report.policy_version,
            policy_fingerprint=report.policy_fingerprint,
            scanner_version=report.scanner_version,
            status=ExpertCandidateSanitationStatus.ADMITTED,
            scanned_files=report.scanned_files,
            findings=(blocking,),
        )
    foreign_finding = replace(
        blocking,
        relative_path="src/not-scanned.py",
        severity=ExpertSanitationSeverity.WARNING,
    )
    with pytest.raises(ContractValidationError, match="unscanned file"):
        ExpertCandidateSanitationReport.mint(
            scope_contract_id=report.scope_contract_id,
            candidate_tree_hash=report.candidate_tree_hash,
            policy_version=report.policy_version,
            policy_fingerprint=report.policy_fingerprint,
            scanner_version=report.scanner_version,
            status=ExpertCandidateSanitationStatus.ADMITTED,
            scanned_files=report.scanned_files,
            findings=(foreign_finding,),
        )

    evidence_id = content_id("fixture", {"lineage": "retirement"})
    retired = ExpertCapabilityLineage(
        source_capability_ids=("capability.old",),
        target_capability_ids=(),
        relation=ExpertCapabilityLineageRelation.RETIRE,
        evidence_ids=(evidence_id,),
    )
    assert retired.target_capability_ids == ()
    with pytest.raises(ContractValidationError, match="rename.*cardinality"):
        ExpertCapabilityLineage(
            source_capability_ids=("capability.old",),
            target_capability_ids=(),
            relation=ExpertCapabilityLineageRelation.RENAME,
            evidence_ids=(evidence_id,),
        )


def test_content_identity_still_detects_valid_shape_mutation():
    candidate = record(ExpertCandidateManifest)

    with pytest.raises(CanonicalizationError):
        replace(candidate, semantic_book_digest=digest("different-book"))
