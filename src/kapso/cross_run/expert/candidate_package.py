"""Canonical package projection shared by candidate validation and storage."""

from __future__ import annotations

from typing import Mapping

from kapso.cross_run.canonical import canonical_json_bytes
from kapso.cross_run.contracts import (
    ExpertCandidateManifest,
    ExpertCandidatePatch,
    ExpertCandidateSanitationReport,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertSourceTreeManifest,
    SourceFileDescriptor,
    StrictContract,
)
from kapso.cross_run.expert.candidate_context import (
    ExpertCandidateValidationContext,
)
from kapso.cross_run.expert.candidate_derivations import (
    AGENT_DERIVATION_RECORD_PACKAGE_PATH,
    CANDIDATE_MANIFEST_PACKAGE_PATH,
    CANDIDATE_MODULE_PACKAGE_ROOT,
    CANDIDATE_PATCH_PACKAGE_PATH,
    CANDIDATE_REPOSITORY_MAP_PACKAGE_PATH,
    CANDIDATE_SOURCE_PACKAGE_ROOT,
    CANDIDATE_SOURCE_TREE_PACKAGE_PATH,
    CANDIDATE_VALIDATION_CONTEXT_PACKAGE_PATH,
    ExpertAgentProposalDerivation,
)

SOURCE_BASE_FILES_PACKAGE_PATH = "source-base-files.json"
SANITATION_REPORT_PACKAGE_PATH = "sanitation.json"
AGENT_DERIVATION_PACKAGE_ROOT = "derivations/agent"
AGENT_TRIGGER_PACKET_PACKAGE_PATH = (
    f"{AGENT_DERIVATION_PACKAGE_ROOT}/trigger-packet.json"
)
AGENT_TRIGGER_DECISION_PACKAGE_PATH = (
    f"{AGENT_DERIVATION_PACKAGE_ROOT}/trigger-decision.json"
)
AGENT_OPERATION_PACKAGE_PATH = f"{AGENT_DERIVATION_PACKAGE_ROOT}/operation.json"
AGENT_WORKSPACE_DELTA_PACKAGE_PATH = (
    f"{AGENT_DERIVATION_PACKAGE_ROOT}/workspace-delta.json"
)
AGENT_ANCESTORS_PACKAGE_PATH = f"{AGENT_DERIVATION_PACKAGE_ROOT}/ancestors.json"
AGENT_ARTIFACT_PACKAGE_ROOT = f"{AGENT_DERIVATION_PACKAGE_ROOT}/artifacts"


def contract_tuple_package_bytes(contracts: tuple[StrictContract, ...]) -> bytes:
    return canonical_json_bytes(tuple(contract.to_dict() for contract in contracts))


def direct_agent_candidate_package_files(
    *,
    manifest: ExpertCandidateManifest,
    validation_context: ExpertCandidateValidationContext,
    patch: ExpertCandidatePatch,
    candidate_tree: ExpertSourceTreeManifest,
    source_base_files: tuple[SourceFileDescriptor, ...],
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
    derivation: ExpertAgentProposalDerivation,
    sanitation_report: ExpertCandidateSanitationReport,
    candidate_contents: Mapping[str, bytes],
) -> dict[str, bytes]:
    """Project the exact create-only package for one direct agent candidate."""

    files = {
        CANDIDATE_MANIFEST_PACKAGE_PATH: manifest.to_json_bytes(),
        CANDIDATE_VALIDATION_CONTEXT_PACKAGE_PATH: validation_context.to_json_bytes(),
        CANDIDATE_PATCH_PACKAGE_PATH: patch.to_json_bytes(),
        CANDIDATE_SOURCE_TREE_PACKAGE_PATH: candidate_tree.to_json_bytes(),
        SOURCE_BASE_FILES_PACKAGE_PATH: contract_tuple_package_bytes(source_base_files),
        CANDIDATE_REPOSITORY_MAP_PACKAGE_PATH: repository_map.to_json_bytes(),
        SANITATION_REPORT_PACKAGE_PATH: sanitation_report.to_json_bytes(),
        AGENT_DERIVATION_RECORD_PACKAGE_PATH: derivation.record.to_json_bytes(),
        AGENT_TRIGGER_PACKET_PACKAGE_PATH: derivation.trigger_packet.to_json_bytes(),
        AGENT_TRIGGER_DECISION_PACKAGE_PATH: derivation.trigger_decision.to_json_bytes(),
        AGENT_OPERATION_PACKAGE_PATH: derivation.operation.to_json_bytes(),
        AGENT_WORKSPACE_DELTA_PACKAGE_PATH: derivation.workspace_delta.to_json_bytes(),
        AGENT_ANCESTORS_PACKAGE_PATH: contract_tuple_package_bytes(
            derivation.ancestor_inputs
        ),
    }
    for module in module_contracts:
        digest = module.module_contract_id.rsplit(":", 1)[1]
        files[f"{CANDIDATE_MODULE_PACKAGE_ROOT}/{digest}.json"] = module.to_json_bytes()
    for path, payload in candidate_contents.items():
        files[f"{CANDIDATE_SOURCE_PACKAGE_ROOT}/{path}"] = payload
    for name, payload in derivation.operation_artifacts.items():
        files[f"{AGENT_ARTIFACT_PACKAGE_ROOT}/{name}"] = payload
    return dict(sorted(files.items()))
