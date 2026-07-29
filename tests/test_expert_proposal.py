from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
    coding_agent_returned_artifact_filenames,
)
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContractValidationError,
    ExpertCandidateOperationKind,
    ExpertModuleContract,
    PriorKnowledgeSnapshot,
)
from kapso.cross_run.expert.architect import ExpertRepositoryArchitect
from kapso.cross_run.expert.candidates import (
    ExpertCandidateValidationError,
    ExpertCandidateValidator,
)
from kapso.cross_run.expert.generalizer import ExpertCapabilityGeneralizer
from kapso.cross_run.expert.proposal import ExpertCandidateProposalEngine
from kapso.cross_run.expert.store import (
    ExpertCandidateStore,
    ExpertCandidateStoreError,
)
from kapso.cross_run.expert.triggers import (
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvaluator,
    ExpertTriggerObservation,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.expert.workspace import (
    ExpertCandidateWorkspaceError,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
)
from kapso.cross_run.expert.proposal_contract import (
    EXPERT_PROPOSAL_CONTRACT_VERSION,
    ExpertModuleProposal,
    ExpertProposalContractError,
    _repository_architecture_signature,
    build_expert_proposal_prompt,
    derive_expert_proposal_topology,
    expert_candidate_prior_knowledge_release_ids,
    parse_expert_proposal,
)
from kapso.cross_run.expert.workspace import ExpertCandidateWorkspaceManager
from kapso.cross_run.knowledge.access import PriorKnowledgeAccessMaterialization
from kapso.execution.coding_agents.structured_call import (
    CodingAgentCallResult,
    coding_agent_invocation_bytes,
    coding_agent_mcp_configuration_bytes,
    coding_agent_response_schema_bytes,
)
from kapso.execution.coding_agents.workspace_delta import (
    build_coding_agent_workspace_delta,
    inspect_coding_agent_workspace,
)
from test_expert_candidates import expert_settings, sanitation_settings
from test_expert_triggers import trigger_packet, trigger_settings
from test_expert_triggers import configuration_fingerprint, inspection_operation
from test_expert_candidate_workspace import (
    FixtureSourceMaterializer,
    released_workspace_fixture,
)
from test_cross_run_retrieval import source_fixture


@pytest.mark.parametrize("operation_kind", tuple(ExpertCandidateOperationKind))
def test_expert_proposal_prompts_require_enforceable_resource_bounds(operation_kind):
    prompt = build_expert_proposal_prompt(operation_kind, {})
    normalized_prompt = " ".join(prompt.split())

    assert "resource_bounds` as an enforceable safety contract" in normalized_prompt
    assert "including its type and string contents" in normalized_prompt
    assert "must not leak implementation exceptions" in normalized_prompt
    assert "explicit finite depth and cardinality limits" in normalized_prompt
    assert "Never claim asymptotic bounds lower" in normalized_prompt


def test_generalization_prompt_names_the_fixed_safety_envelope():
    prompt = build_expert_proposal_prompt(
        ExpertCandidateOperationKind.GENERALIZE,
        {},
    )
    normalized_prompt = " ".join(prompt.split())

    assert (
        "copy `purpose`, `dependency_capability_ids`, "
        "`incompatible_capability_ids`, and `resource_bounds` exactly"
        in normalized_prompt
    )
    assert (
        "you may add justified values, but never remove or rewrite" in normalized_prompt
    )


class UnusedSourceMaterializer:
    def extract_verified_source_archive(self, **kwargs):
        raise AssertionError("bootstrap must not materialize a released source base")


class RootSubstitutingWorkspaceLease:
    def __init__(self, lease, workspace_root: Path):
        self.lease = lease
        self.workspace_root = workspace_root

    def __enter__(self):
        return self.lease.__enter__()

    def __exit__(self, exception_type, exception, traceback):
        displaced = self.workspace_root.with_name(
            self.workspace_root.name + "-displaced"
        )
        os.rename(self.workspace_root, displaced)
        os.mkdir(self.workspace_root, mode=0o700)
        return self.lease.__exit__(exception_type, exception, traceback)

    @property
    def workspace_authority_descriptor(self):
        return self.lease.workspace_authority_descriptor

    def validate(self):
        self.lease.validate()


class RootSubstitutingWorkspaceManager(ExpertCandidateWorkspaceManager):
    def lease(self, **kwargs):
        return RootSubstitutingWorkspaceLease(
            super().lease(**kwargs),
            self.root,
        )


class BootstrapProposalRunner:
    def __init__(
        self,
        artifact_root: Path,
        output: str,
        source: dict[str, bytes],
        deleted_paths: tuple[str, ...] = (),
    ):
        self.artifact_root = artifact_root
        self.output = output
        self.source = source
        self.deleted_paths = deleted_paths
        self.calls = []

    def run(
        self,
        request,
        response_schema,
        *,
        workspace_authority_descriptor=None,
    ):
        self.calls.append((request, response_schema, workspace_authority_descriptor))
        assert workspace_authority_descriptor is not None
        settings = expert_settings()
        workspace = Path(request.workspace)
        baseline = inspect_coding_agent_workspace(
            workspace,
            maximum_entries=settings.candidate_entry_limit,
            maximum_bytes=settings.candidate_byte_limit,
        )
        for relative_path in self.deleted_paths:
            (workspace / relative_path).unlink()
        for relative_path, payload in self.source.items():
            output = workspace / relative_path
            output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            output.write_bytes(payload)
            output.chmod(0o644)
        edited = inspect_coding_agent_workspace(
            workspace,
            maximum_entries=settings.candidate_entry_limit,
            maximum_bytes=settings.candidate_byte_limit,
        )
        delta = build_coding_agent_workspace_delta(baseline, edited)
        if not self.artifact_root.exists():
            self.artifact_root.mkdir(mode=0o700)
        artifact_directory = self.artifact_root / request.operation_id
        artifact_directory.mkdir(mode=0o700)
        artifacts = {
            "final.json": self.output.encode("utf-8"),
            "invocation.json": coding_agent_invocation_bytes(
                request,
                sensitive_file_glob_scan_max_depth=(
                    settings.sensitive_file_glob_scan_max_depth
                ),
            ),
            "mcp_audit.jsonl": b"",
            "mcp_config.json": coding_agent_mcp_configuration_bytes(
                request,
                artifact_directory,
            ),
            "prior_knowledge.json": (
                b"null\n"
                if request.prior_knowledge is None
                else request.prior_knowledge.to_json_bytes()
            ),
            "prompt.txt": request.prompt.encode("utf-8"),
            "response_schema.json": coding_agent_response_schema_bytes(response_schema),
            "stderr.txt": b"",
            "stdout.txt": b"completed\n",
            "workspace-delta.json": delta.to_json_bytes(),
        }
        returned_paths = tuple(
            str(artifact_directory / name)
            for name in coding_agent_returned_artifact_filenames(
                CodingAgentWorkspaceAccess.EDIT_WORKSPACE
            )
        )
        result = CodingAgentCallResult(
            output=self.output,
            duration_seconds=1.0,
            cost_usd=None,
            final_output_digest=tree_or_blob_digest(self.output.encode("utf-8")),
            workspace_delta_digest=tree_or_blob_digest(delta.to_json_bytes()),
            input_tokens=1,
            output_tokens=1,
            artifacts=returned_paths,
        )
        artifacts["result.json"] = result.to_json_bytes()
        for name, payload in artifacts.items():
            path = artifact_directory / name
            path.write_bytes(payload)
            path.chmod(0o600)
        return result


def bootstrap_output() -> str:
    return (
        json.dumps(
            {
                "capability_lineage": [],
                "changed_paths": [
                    "src/execution.py",
                    "tests/test_execution.py",
                ],
                "deleted_paths": [],
                "module_contracts": [
                    {
                        "dependency_capability_ids": [],
                        "dependency_license_manifest": {"license": "MIT"},
                        "entrypoint_refs": ["src/execution.py"],
                        "incompatibilities": [],
                        "incompatible_capability_ids": [],
                        "inputs": [],
                        "known_failure_episode_ids": [],
                        "module_id": "shared.execution",
                        "outputs": ["validated artifact"],
                        "preconditions": [],
                        "problem_signals": [
                            "Task implementations duplicate execution control."
                        ],
                        "purpose": (
                            "Execute a task through one provenance-bound interface."
                        ),
                        "replay_refs": [],
                        "resource_bounds": {"concurrency": 1},
                        "supporting_episode_ids": [],
                        "test_refs": ["tests/test_execution.py"],
                        "version": "v1",
                    }
                ],
                "repository_topology": {
                    "architecture_invariants": [
                        "No task identity appears in generic defaults."
                    ],
                    "capability_nodes": [
                        {
                            "capability_id": "shared.execution",
                            "owned_paths": [
                                "src/execution.py",
                                "tests/test_execution.py",
                            ],
                            "task_family_bindings": ["language_model_post_training"],
                        }
                    ],
                    "task_adapter_boundary": {
                        "adapter_mount_path": ".kapso/task-adapter",
                        "inputs": ["task contract"],
                        "interface_entrypoint_refs": ["src/execution.py"],
                        "invariants": [
                            "The task adapter remains external and read-only."
                        ],
                        "outputs": ["validated artifact"],
                    },
                    "validation_entrypoints": ["tests/test_execution.py"],
                },
                "summary": "Bootstrapped the smallest shared execution capability.",
            },
            sort_keys=True,
        )
        + "\n"
    )


def proposal_system(
    tmp_path,
    output=None,
    settings=None,
    workspace_manager_type=ExpertCandidateWorkspaceManager,
):
    tmp_path.chmod(0o700)
    configured = expert_settings() if settings is None else settings
    validator = ExpertCandidateValidator(configured, sanitation_settings())
    store = ExpertCandidateStore(tmp_path / "candidates", tmp_path, validator)
    manager = workspace_manager_type(
        tmp_path / "workspaces",
        tmp_path,
        configured,
        UnusedSourceMaterializer(),
    )
    source = {
        "src/execution.py": b"def execute(task):\n    return task.run()\n",
        "tests/test_execution.py": b"def test_execute():\n    assert True\n",
    }
    runner = BootstrapProposalRunner(
        tmp_path / "agent-artifacts",
        bootstrap_output() if output is None else output,
        source,
    )
    engine = ExpertCandidateProposalEngine(
        settings=configured,
        runner=runner,
        workspace_manager=manager,
        candidate_store=store,
    )
    return ExpertRepositoryArchitect(engine), store, runner, source


def test_architect_bootstrap_seals_and_reopens_exact_candidate(tmp_path):
    architect, store, runner, source = proposal_system(tmp_path)
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    result = architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=None,
    )
    closure = result.stored_candidate.closure
    reopened = store.read(closure.manifest.candidate_id)

    assert reopened == result.stored_candidate
    assert closure.derivation.operation.operation_kind.value == "bootstrap"
    assert closure.derivation.operation.operation_preimage[
        "proposal_contract_version"
    ] == (EXPERT_PROPOSAL_CONTRACT_VERSION)
    assert closure.derivation.operation.operation_preimage["principal_id"] == (
        closure.derivation.operation.operation_receipt.principal_id
    )
    assert closure.candidate_contents[EXPERT_BOOK_PATH].startswith(
        b"# Expert Repository\n"
    )
    assert closure.candidate_contents[EXPERT_REPOSITORY_MAP_PATH] == (
        closure.repository_map.to_json_bytes()
    )
    assert {path: closure.candidate_contents[path] for path in source} == source
    assert len(runner.calls) == 1
    assert tuple((tmp_path / "workspaces").iterdir()) == ()


def test_architect_tracks_every_model_visible_episode_release(tmp_path):
    _, _, episode, _, _, _, _ = source_fixture()
    architect, store, _, _ = proposal_system(tmp_path)
    packet = trigger_packet(
        settings=trigger_settings(),
        episodes=(episode,),
        bootstrap=True,
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    result = architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=None,
    )
    closure = result.stored_candidate.closure

    assert episode.episode_id not in (
        closure.validation_context.replay_evidence.causal_episode_ids
    )
    assert closure.manifest.consumed_expert_release_ids == (
        episode.artifact_environment.expert_base_release_id,
    )
    assert store.read(closure.manifest.candidate_id) == result.stored_candidate


def test_architect_principal_rotation_changes_operation_identity(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir(mode=0o700)
    second_root.mkdir(mode=0o700)
    configured = expert_settings()
    rotated = replace(configured, architect_id="expert-architect-rotated")
    first_architect, _, _, _ = proposal_system(first_root, settings=configured)
    second_architect, _, _, _ = proposal_system(second_root, settings=rotated)
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    first = first_architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=None,
    ).stored_candidate.closure.derivation.operation
    second = second_architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=None,
    ).stored_candidate.closure.derivation.operation

    assert first.operation_receipt.operation_id != second.operation_receipt.operation_id
    assert first.operation_preimage["principal_id"] == configured.architect_id
    assert second.operation_preimage["principal_id"] == rotated.architect_id


def test_historical_candidate_reopens_after_principal_rotation(tmp_path):
    architect, _, _, _ = proposal_system(tmp_path)
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)
    candidate = architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=None,
    ).stored_candidate
    configured = expert_settings()
    rotated = replace(
        configured,
        architect_id="expert-architect-rotated",
        candidate_entry_limit=configured.candidate_entry_limit + 1,
        candidate_byte_limit=configured.candidate_byte_limit + 1,
    )
    reopened_store = ExpertCandidateStore(
        tmp_path / "candidates",
        tmp_path,
        ExpertCandidateValidator(rotated, sanitation_settings()),
    )

    assert reopened_store.read(candidate.closure.manifest.candidate_id) == candidate


def test_failed_workspace_lease_close_persists_no_candidate(tmp_path):
    architect, store, _, _ = proposal_system(
        tmp_path,
        workspace_manager_type=RootSubstitutingWorkspaceManager,
    )
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    with pytest.raises(
        ExpertCandidateWorkspaceError,
        match="workspace root",
    ):
        architect.propose(
            packet=packet,
            decision=decision,
            materialized_source_base=None,
        )

    assert tuple(store.object_root.iterdir()) == ()


def test_architect_path_declaration_mismatch_leaves_no_candidate(tmp_path):
    payload = json.loads(bootstrap_output())
    payload["changed_paths"] = ["src/execution.py"]
    output = json.dumps(payload, sort_keys=True) + "\n"
    architect, store, runner, _ = proposal_system(tmp_path, output)
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    with pytest.raises(
        ExpertProposalContractError,
        match="path declarations differ",
    ):
        architect.propose(
            packet=packet,
            decision=decision,
            materialized_source_base=None,
        )

    assert len(runner.calls) == 1
    assert tuple(store.object_root.iterdir()) == ()
    assert tuple((tmp_path / "workspaces").iterdir()) == ()


def test_bootstrap_rejects_speculative_inactive_task_family(tmp_path):
    payload = json.loads(bootstrap_output())
    payload["repository_topology"]["capability_nodes"][0]["task_family_bindings"] = [
        "language_model_post_training",
        "relational_tabular_prediction",
    ]
    output = json.dumps(payload, sort_keys=True) + "\n"
    architect, store, runner, _ = proposal_system(tmp_path, output)
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    with pytest.raises(
        ExpertCandidateValidationError,
        match="speculates beyond active task families",
    ):
        architect.propose(
            packet=packet,
            decision=decision,
            materialized_source_base=None,
        )

    assert len(runner.calls) == 1
    assert tuple(store.object_root.iterdir()) == ()
    assert tuple((tmp_path / "workspaces").iterdir()) == ()


def test_architect_persists_exact_ancestor_source_input(tmp_path):
    _, _, episode, _, _, _, _ = source_fixture()
    architect, store, runner, _ = proposal_system(tmp_path)
    packet = trigger_packet(
        settings=trigger_settings(),
        episodes=(episode,),
        bootstrap=True,
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)
    first = architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=None,
    ).stored_candidate

    second = architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=None,
        ancestor_candidate_ids=(first.closure.manifest.candidate_id,),
    ).stored_candidate
    ancestor = second.closure.derivation.ancestor_inputs[0]

    assert second.closure.manifest.candidate_id != first.closure.manifest.candidate_id
    assert ancestor.manifest == first.closure.manifest
    assert "workspace_delta" not in ancestor.to_dict()
    assert ancestor.candidate_contents() == first.closure.candidate_contents
    assert ancestor.candidate_contents_text["src/execution.py"].startswith(
        "def execute"
    )
    assert "candidate_contents_base64" not in ancestor.to_dict()
    assert second.closure.manifest.consumed_expert_release_ids == (
        episode.artifact_environment.expert_base_release_id,
    )
    assert store.read(second.closure.manifest.candidate_id) == second
    assert len(runner.calls) == 2

    isolated_root = tmp_path / "isolated"
    isolated_root.mkdir(mode=0o700)
    isolated_store = ExpertCandidateStore(
        isolated_root / "candidates",
        isolated_root,
        ExpertCandidateValidator(expert_settings(), sanitation_settings()),
    )
    with pytest.raises(ExpertCandidateStoreError, match="does not exist"):
        isolated_store.persist(second.closure)


def test_architect_rejects_foreign_prior_snapshot_before_agent_call(tmp_path):
    architect, store, runner, _ = proposal_system(tmp_path)
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)
    prior_packet = PriorKnowledgeSnapshot.mint(
        source_snapshot_id=content_id(
            "knowledge-snapshot",
            {"fixture": "foreign"},
        ),
        query={"problem": "foreign knowledge"},
        retrieval_policy_version="kapso.retrieval.v1",
        task_context_binding_id=content_id(
            "task-context-binding",
            {"fixture": "foreign"},
        ),
        selected_records=(),
        selected_record_ids=(),
        proof_reference_ids=(),
        selection_metadata={},
        prompt_budget_policy={"maximum_bytes": 1},
        records_digest=tree_or_blob_digest(canonical_json_bytes(())),
    )
    prior_knowledge = PriorKnowledgeAccessMaterialization.mint(
        prior_knowledge_snapshot=prior_packet,
        proof_records=(),
    )

    with pytest.raises(
        ExpertProposalContractError,
        match="prior knowledge leaves its trigger evidence closure",
    ):
        architect.propose(
            packet=packet,
            decision=decision,
            materialized_source_base=None,
            prior_knowledge=prior_knowledge,
        )

    assert runner.calls == []
    assert tuple(store.object_root.iterdir()) == ()
    assert tuple((tmp_path / "workspaces").iterdir()) == ()


def test_architect_binds_every_model_visible_knowledge_record_as_dependency(tmp_path):
    _, context, episode, _, claim, _, _ = source_fixture()
    packet = trigger_packet(
        settings=trigger_settings(),
        episodes=(episode,),
        claims=(claim,),
        bootstrap=True,
    )
    selected_record = {
        "record_id": claim.revision_id,
        "record_kind": claim.CONTENT_NAMESPACE,
        "payload": claim.to_dict(),
    }
    proof_record = {
        "record_id": episode.episode_id,
        "record_kind": episode.CONTENT_NAMESPACE,
        "payload": episode.to_dict(),
    }
    prior_snapshot = PriorKnowledgeSnapshot.mint(
        source_snapshot_id=packet.knowledge_snapshot_id,
        query={"problem": "Generalize reproducible execution."},
        retrieval_policy_version="kapso.retrieval.v1",
        task_context_binding_id=context.task_context_binding_id,
        selected_records=(selected_record,),
        selected_record_ids=(claim.revision_id,),
        proof_reference_ids=(episode.episode_id,),
        selection_metadata={
            claim.revision_id: {
                "compatibility": "exact_context",
                "evidence_quality": 1,
                "lexical_score": 1.0,
                "outcome": "positive",
                "proof_reference_ids": (episode.episode_id,),
                "rank": 0,
                "recency": "",
                "retrieval_utility": 1.0,
                "semantic_score": 0.0,
            }
        },
        prompt_budget_policy={"maximum_records": 1},
        records_digest=tree_or_blob_digest(canonical_json_bytes((selected_record,))),
    )
    prior_knowledge = PriorKnowledgeAccessMaterialization.mint(
        prior_knowledge_snapshot=prior_snapshot,
        proof_records=(proof_record,),
    )
    assert expert_candidate_prior_knowledge_release_ids(prior_knowledge) == (
        episode.artifact_environment.expert_base_release_id,
    )
    architect, store, runner, _ = proposal_system(tmp_path)
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    result = architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=None,
        prior_knowledge=prior_knowledge,
    )
    dependencies = set(result.stored_candidate.closure.manifest.source_dependency_ids)

    assert {
        prior_snapshot.prior_knowledge_snapshot_id,
        prior_snapshot.source_snapshot_id,
        claim.revision_id,
        episode.episode_id,
    }.issubset(dependencies)
    assert store.read(result.stored_candidate.closure.manifest.candidate_id) == (
        result.stored_candidate
    )
    assert len(runner.calls) == 1


def released_observation_packet(kind, description):
    packet, materialized, contents = released_workspace_fixture()
    settings = trigger_settings()
    module = packet.source_base_module_contracts[0]
    inspection_payload = {
        "affected_capability_ids": [module.module_id],
        "affected_paths": ["src/reproducible_execution/__init__.py"],
        "configuration_fingerprint": configuration_fingerprint(settings),
        "description": description,
        "difficulty_evidence_signatures": {},
        "difficulty_signature": None,
        "exact_evidence_ids": [packet.source_base_repository_map.repository_map_id],
        "independent_lineage_ids": [],
        "inspection_policy_version": settings.inspection_policy_version,
        "kind": kind.value,
        "occurrence_count": 1,
        "source_base_tree_hash": packet.source_base_tree_hash,
        "task_context_binding_ids": [],
    }
    inspection_final_output = json.dumps(inspection_payload, indent=2) + "\n"
    observation = ExpertTriggerObservation.mint(
        kind=kind,
        source_base_tree_hash=packet.source_base_tree_hash,
        inspection_policy_version=settings.inspection_policy_version,
        configuration_fingerprint=configuration_fingerprint(settings),
        inspection_operation=inspection_operation(
            settings,
            inspection_final_output,
        ),
        inspection_final_output=inspection_final_output,
        difficulty_signature=None,
        difficulty_evidence_signatures={},
        description=description,
        affected_capability_ids=(module.module_id,),
        affected_paths=("src/reproducible_execution/__init__.py",),
        exact_evidence_ids=(packet.source_base_repository_map.repository_map_id,),
        independent_lineage_ids=(),
        task_context_binding_ids=(),
        occurrence_count=1,
    )
    triggered_packet = ExpertTriggerEvidencePacket.mint(
        knowledge_snapshot_manifest=packet.knowledge_snapshot_manifest,
        knowledge_record_closure_digest=packet.knowledge_record_closure_digest,
        configuration_fingerprint=packet.configuration_fingerprint,
        scope_contract=packet.scope_contract,
        source_base_scope_contract=packet.source_base_scope_contract,
        source_base_release=packet.source_base_release,
        source_base_tree_receipt=packet.source_base_tree_receipt,
        source_base_tree_hash=packet.source_base_tree_hash,
        source_base_repository_map=packet.source_base_repository_map,
        source_base_module_contracts=packet.source_base_module_contracts,
        episodes=packet.episodes,
        claims=packet.claims,
        trigger_observations=(observation,),
        active_task_bindings=packet.active_task_bindings,
        proof_reference_ids=packet.proof_reference_ids,
        recovery_barrier_basis_packet_id=None,
    )
    return triggered_packet, materialized, contents


def generalization_packet():
    return released_observation_packet(
        ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        "A provenance field can be added without changing topology.",
    )


def generalizer_output(packet) -> str:
    module = packet.source_base_module_contracts[0].to_dict()
    del module["module_contract_id"]
    module["version"] = "v2"
    return (
        json.dumps(
            {
                "changed_module_contracts": [module],
                "changed_paths": ["src/reproducible_execution/__init__.py"],
                "deleted_paths": [],
                "summary": "Added provenance capture to reusable execution.",
            },
            sort_keys=True,
        )
        + "\n"
    )


def test_generalizer_rejects_declared_but_unchanged_module_contract():
    packet, _, _ = generalization_packet()
    module = packet.source_base_module_contracts[0].to_dict()
    del module["module_contract_id"]
    output = (
        json.dumps(
            {
                "changed_module_contracts": [module],
                "changed_paths": ["src/reproducible_execution/__init__.py"],
                "deleted_paths": [],
                "summary": "Restated an unchanged module.",
            },
            sort_keys=True,
        )
        + "\n"
    )
    proposal = parse_expert_proposal(
        ExpertCandidateOperationKind.GENERALIZE,
        output,
    )

    with pytest.raises(
        ExpertProposalContractError,
        match="unchanged module contract",
    ):
        derive_expert_proposal_topology(
            packet=packet,
            operation_kind=ExpertCandidateOperationKind.GENERALIZE,
            proposal=proposal,
        )


def test_expert_module_version_rejects_non_monotonic_format():
    packet, _, _ = generalization_packet()
    module = packet.source_base_module_contracts[0].to_dict()
    del module["module_contract_id"]
    module["version"] = "v0"
    output = json.dumps(
        {
            "changed_module_contracts": [module],
            "changed_paths": ["src/reproducible_execution/__init__.py"],
            "deleted_paths": [],
            "summary": "Attempted a non-monotonic version.",
        },
        sort_keys=True,
    )

    with pytest.raises(
        ContractValidationError,
        match="expert module version must be a positive v-prefixed integer",
    ):
        proposal = parse_expert_proposal(
            ExpertCandidateOperationKind.GENERALIZE,
            output,
        )
        derive_expert_proposal_topology(
            packet=packet,
            operation_kind=ExpertCandidateOperationKind.GENERALIZE,
            proposal=proposal,
        )


def test_module_proposal_rejects_episode_as_both_support_and_failure():
    packet, _, _ = generalization_packet()
    module = packet.source_base_module_contracts[0].to_dict()
    del module["module_contract_id"]
    episode_id = "transfer-episode:sha256:" + "a" * 64
    module["supporting_episode_ids"] = [episode_id]
    module["known_failure_episode_ids"] = [episode_id]

    with pytest.raises(
        ExpertProposalContractError,
        match="supporting and failure evidence must be disjoint",
    ):
        ExpertModuleProposal.from_dict(module)


def test_generalizer_compares_unbounded_versions_without_integer_conversion():
    packet, _, _ = generalization_packet()
    module = packet.source_base_module_contracts[0].to_dict()
    del module["module_contract_id"]
    module["version"] = "v" + "9" * 5_000
    output = json.dumps(
        {
            "changed_module_contracts": [module],
            "changed_paths": ["src/reproducible_execution/__init__.py"],
            "deleted_paths": [],
            "summary": "Advanced a large canonical module version.",
        },
        sort_keys=True,
    )
    proposal = parse_expert_proposal(
        ExpertCandidateOperationKind.GENERALIZE,
        output,
    )

    _, modules, _ = derive_expert_proposal_topology(
        packet=packet,
        operation_kind=ExpertCandidateOperationKind.GENERALIZE,
        proposal=proposal,
    )

    assert modules[0].version == module["version"]


def test_repository_architecture_signature_ignores_module_input_order():
    packet, _, _ = generalization_packet()
    first = packet.source_base_module_contracts[0]
    second_payload = first.to_dict()
    del second_payload["module_contract_id"]
    second_payload.update(
        {
            "module_id": "shared.secondary",
            "entrypoint_refs": ("src/secondary.py",),
            "test_refs": ("tests/test_secondary.py",),
            "replay_refs": (),
        }
    )
    second = ExpertModuleContract.mint(**second_payload)

    assert _repository_architecture_signature(
        packet.source_base_repository_map,
        (first, second),
    ) == _repository_architecture_signature(
        packet.source_base_repository_map,
        (second, first),
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("preconditions", [], "removes accumulated module safety"),
        (
            "resource_bounds",
            {"maximum_workers": 32},
            "changes a fixed module safety envelope",
        ),
        (
            "dependency_license_manifest",
            {"license": "Apache-2.0"},
            "rewrites dependency license metadata",
        ),
    ),
)
def test_generalizer_cannot_weaken_accumulated_module_contract(
    field,
    replacement,
    message,
):
    packet, _, _ = generalization_packet()
    module = packet.source_base_module_contracts[0].to_dict()
    del module["module_contract_id"]
    module["version"] = "v2"
    module[field] = replacement
    output = json.dumps(
        {
            "changed_module_contracts": [module],
            "changed_paths": ["src/reproducible_execution/__init__.py"],
            "deleted_paths": [],
            "summary": "Attempted to weaken an accumulated module contract.",
        },
        sort_keys=True,
    )
    proposal = parse_expert_proposal(
        ExpertCandidateOperationKind.GENERALIZE,
        output,
    )

    with pytest.raises(ExpertProposalContractError, match=message):
        derive_expert_proposal_topology(
            packet=packet,
            operation_kind=ExpertCandidateOperationKind.GENERALIZE,
            proposal=proposal,
        )


def test_generalizer_preserves_topology_and_replaces_changed_contract(tmp_path):
    tmp_path.chmod(0o700)
    packet, materialized, source_base_contents = generalization_packet()
    settings = expert_settings()
    validator = ExpertCandidateValidator(settings, sanitation_settings())
    store = ExpertCandidateStore(tmp_path / "candidates", tmp_path, validator)
    manager = ExpertCandidateWorkspaceManager(
        tmp_path / "workspaces",
        tmp_path,
        settings,
        FixtureSourceMaterializer(source_base_contents),
    )
    runner = BootstrapProposalRunner(
        tmp_path / "agent-artifacts",
        generalizer_output(packet),
        {
            "src/reproducible_execution/__init__.py": (
                b"def execute(task):\n    return task.run_with_provenance()\n"
            )
        },
    )
    generalizer = ExpertCapabilityGeneralizer(
        ExpertCandidateProposalEngine(
            settings=settings,
            runner=runner,
            workspace_manager=manager,
            candidate_store=store,
        )
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    result = generalizer.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=materialized,
    )
    closure = result.stored_candidate.closure

    assert closure.derivation.operation.operation_kind.value == "generalize"
    assert closure.module_contracts[0].version == "v2"
    assert closure.repository_map.capability_nodes[0].owned_paths == (
        packet.source_base_repository_map.capability_nodes[0].owned_paths
    )
    assert closure.repository_map.dependency_edges == (
        packet.source_base_repository_map.dependency_edges
    )
    assert closure.manifest.capability_lineage == ()
    assert store.read(closure.manifest.candidate_id) == result.stored_candidate


def restructure_output(packet) -> str:
    module = packet.source_base_module_contracts[0].to_dict()
    del module["module_contract_id"]
    module["entrypoint_refs"] = ["src/execution/__init__.py"]
    module["version"] = "v2"
    node = packet.source_base_repository_map.capability_nodes[0]
    adapter = packet.source_base_repository_map.task_adapter_boundary.to_dict()
    adapter["interface_entrypoint_refs"] = ["src/execution/__init__.py"]
    return (
        json.dumps(
            {
                "capability_lineage": [],
                "changed_paths": ["src/execution/__init__.py"],
                "deleted_paths": [
                    "src/reproducible_execution/__init__.py",
                ],
                "module_contracts": [module],
                "repository_topology": {
                    "architecture_invariants": (
                        packet.source_base_repository_map.architecture_invariants
                    ),
                    "capability_nodes": [
                        {
                            "capability_id": node.capability_id,
                            "owned_paths": ["src/execution", "tests"],
                            "task_family_bindings": node.task_family_bindings,
                        }
                    ],
                    "task_adapter_boundary": adapter,
                    "validation_entrypoints": (
                        packet.source_base_repository_map.validation_entrypoints
                    ),
                },
                "summary": "Moved the capability without changing its identity.",
            },
            sort_keys=True,
        )
        + "\n"
    )


def unchanged_restructure_output(packet) -> str:
    module = packet.source_base_module_contracts[0].to_dict()
    del module["module_contract_id"]
    module["version"] = "v2"
    module["problem_signals"] = sorted(
        [*module["problem_signals"], "Execution needs clearer provenance."]
    )
    source_base_map = packet.source_base_repository_map
    return (
        json.dumps(
            {
                "capability_lineage": [],
                "changed_paths": ["src/reproducible_execution/__init__.py"],
                "deleted_paths": [],
                "module_contracts": [module],
                "repository_topology": {
                    "architecture_invariants": source_base_map.architecture_invariants,
                    "capability_nodes": [
                        {
                            "capability_id": node.capability_id,
                            "owned_paths": node.owned_paths,
                            "task_family_bindings": node.task_family_bindings,
                        }
                        for node in source_base_map.capability_nodes
                    ],
                    "task_adapter_boundary": (
                        source_base_map.task_adapter_boundary.to_dict()
                    ),
                    "validation_entrypoints": source_base_map.validation_entrypoints,
                },
                "summary": "Attempted a capability-only architecture proposal.",
            },
            sort_keys=True,
        )
        + "\n"
    )


def test_architect_restructure_requires_real_structural_delta(tmp_path):
    tmp_path.chmod(0o700)
    packet, materialized, source_base_contents = released_observation_packet(
        ExpertTriggerObservationKind.CONTRACT_TOPOLOGY_MISMATCH,
        "The capability contract and physical path no longer agree.",
    )
    settings = expert_settings()
    store = ExpertCandidateStore(
        tmp_path / "candidates",
        tmp_path,
        ExpertCandidateValidator(settings, sanitation_settings()),
    )
    manager = ExpertCandidateWorkspaceManager(
        tmp_path / "workspaces",
        tmp_path,
        settings,
        FixtureSourceMaterializer(source_base_contents),
    )
    runner = BootstrapProposalRunner(
        tmp_path / "agent-artifacts",
        unchanged_restructure_output(packet),
        {
            "src/reproducible_execution/__init__.py": (
                b"def execute(task):\n    return task.run_with_provenance()\n"
            )
        },
    )
    architect = ExpertRepositoryArchitect(
        ExpertCandidateProposalEngine(
            settings=settings,
            runner=runner,
            workspace_manager=manager,
            candidate_store=store,
        )
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    with pytest.raises(
        ExpertProposalContractError,
        match="must change repository structure",
    ):
        architect.propose(
            packet=packet,
            decision=decision,
            materialized_source_base=materialized,
        )

    assert tuple(store.object_root.iterdir()) == ()


def test_architect_restructure_cannot_weaken_preserved_capability(tmp_path):
    tmp_path.chmod(0o700)
    packet, materialized, source_base_contents = released_observation_packet(
        ExpertTriggerObservationKind.CONTRACT_TOPOLOGY_MISMATCH,
        "The capability contract and physical path no longer agree.",
    )
    payload = json.loads(restructure_output(packet))
    payload["module_contracts"][0]["preconditions"] = []
    payload["module_contracts"][0]["resource_bounds"] = {"maximum_workers": 999_999}
    payload["module_contracts"][0]["dependency_license_manifest"] = {
        "license": "UNKNOWN"
    }
    settings = expert_settings()
    store = ExpertCandidateStore(
        tmp_path / "candidates",
        tmp_path,
        ExpertCandidateValidator(settings, sanitation_settings()),
    )
    manager = ExpertCandidateWorkspaceManager(
        tmp_path / "workspaces",
        tmp_path,
        settings,
        FixtureSourceMaterializer(source_base_contents),
    )
    runner = BootstrapProposalRunner(
        tmp_path / "agent-artifacts",
        json.dumps(payload, sort_keys=True) + "\n",
        {"src/execution/__init__.py": b"def execute(task):\n    return task.run()\n"},
        deleted_paths=("src/reproducible_execution/__init__.py",),
    )
    architect = ExpertRepositoryArchitect(
        ExpertCandidateProposalEngine(
            settings=settings,
            runner=runner,
            workspace_manager=manager,
            candidate_store=store,
        )
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    with pytest.raises(
        ExpertProposalContractError,
        match="fixed module safety envelope",
    ):
        architect.propose(
            packet=packet,
            decision=decision,
            materialized_source_base=materialized,
        )

    assert tuple(store.object_root.iterdir()) == ()


def test_architect_restructure_cannot_erase_replay_provenance(tmp_path):
    tmp_path.chmod(0o700)
    packet, materialized, source_base_contents = released_observation_packet(
        ExpertTriggerObservationKind.CONTRACT_TOPOLOGY_MISMATCH,
        "The capability contract and physical path no longer agree.",
    )
    payload = json.loads(restructure_output(packet))
    payload["module_contracts"][0]["replay_refs"] = []
    settings = expert_settings()
    store = ExpertCandidateStore(
        tmp_path / "candidates",
        tmp_path,
        ExpertCandidateValidator(settings, sanitation_settings()),
    )
    manager = ExpertCandidateWorkspaceManager(
        tmp_path / "workspaces",
        tmp_path,
        settings,
        FixtureSourceMaterializer(source_base_contents),
    )
    runner = BootstrapProposalRunner(
        tmp_path / "agent-artifacts",
        json.dumps(payload, sort_keys=True) + "\n",
        {"src/execution/__init__.py": b"def execute(task):\n    return task.run()\n"},
        deleted_paths=("src/reproducible_execution/__init__.py",),
    )
    architect = ExpertRepositoryArchitect(
        ExpertCandidateProposalEngine(
            settings=settings,
            runner=runner,
            workspace_manager=manager,
            candidate_store=store,
        )
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    with pytest.raises(
        ExpertProposalContractError,
        match="path-reference replacement lacks an exact source move",
    ):
        architect.propose(
            packet=packet,
            decision=decision,
            materialized_source_base=materialized,
        )

    assert tuple(store.object_root.iterdir()) == ()


def test_architect_restructure_preserves_capability_identity_on_path_move(tmp_path):
    tmp_path.chmod(0o700)
    packet, materialized, source_base_contents = released_observation_packet(
        ExpertTriggerObservationKind.CONTRACT_TOPOLOGY_MISMATCH,
        "The capability contract and physical path no longer agree.",
    )
    settings = expert_settings()
    validator = ExpertCandidateValidator(settings, sanitation_settings())
    store = ExpertCandidateStore(tmp_path / "candidates", tmp_path, validator)
    manager = ExpertCandidateWorkspaceManager(
        tmp_path / "workspaces",
        tmp_path,
        settings,
        FixtureSourceMaterializer(source_base_contents),
    )
    runner = BootstrapProposalRunner(
        tmp_path / "agent-artifacts",
        restructure_output(packet),
        {"src/execution/__init__.py": (b"def execute(task):\n    return task.run()\n")},
        deleted_paths=("src/reproducible_execution/__init__.py",),
    )
    architect = ExpertRepositoryArchitect(
        ExpertCandidateProposalEngine(
            settings=settings,
            runner=runner,
            workspace_manager=manager,
            candidate_store=store,
        )
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)

    result = architect.propose(
        packet=packet,
        decision=decision,
        materialized_source_base=materialized,
    )
    closure = result.stored_candidate.closure

    assert closure.derivation.operation.operation_kind.value == "restructure"
    assert closure.repository_map.capability_nodes[0].capability_id == (
        packet.source_base_repository_map.capability_nodes[0].capability_id
    )
    assert closure.repository_map.capability_nodes[0].owned_paths == (
        "src/execution",
        "tests",
    )
    assert closure.manifest.capability_lineage == ()
    assert store.read(closure.manifest.candidate_id) == result.stored_candidate
