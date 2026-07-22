import base64
from dataclasses import replace
from typing import Mapping

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    AdmissionState,
    ArtifactCompleteness,
    ArtifactEnvironment,
    BootstrapPin,
    BundleArtifactRef,
    CandidateChangeKind,
    CaptureManifest,
    CatalogEntryState,
    CodingAgentOperationReceipt,
    CodingAgentWorkspaceChangedFile,
    CodingAgentWorkspaceDelta,
    ComparisonStatus,
    CompletionState,
    ContextDimensionSchema,
    ContextValueType,
    ContractValidationError,
    CrossRunTaskBindingSettings,
    EmbeddingSidecar,
    EpisodeEvaluationStatus,
    EvaluationFingerprint,
    EffectUncertaintyMethod,
    EMPTY_EXPERT_TREE_DIGEST,
    ExecutionStatus,
    ExpertBaseReleaseManifest,
    ExpertCandidateOperationKind,
    ExpertCandidateOperationRecord,
    ExpertCandidateManifest,
    ExpertCandidatePatch,
    ExpertCandidatePatchChange,
    ExpertCandidateSanitationReport,
    ExpertCandidateSanitationStatus,
    ExpertCandidateWorkspaceReceipt,
    ExpertCapabilityNode,
    ExpertDependencyEdge,
    ExpertModuleContract,
    ExpertProposerAuthority,
    ExpertRepositoryMap,
    ExpertScopeContract,
    ExpertSourceReplayComputeBinding,
    ExpertSourceReplayExecutionLegKind,
    ExpertSourceReplayStartingArtifact,
    ExpertSourceTreeManifest,
    ExpertTaskAdapterBoundary,
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    IdentityConflictError,
    IncompatibleArtifactError,
    InterventionStructure,
    KnowledgeClaim,
    KnowledgeSnapshotManifest,
    LaunchManifest,
    LineageEdge,
    LineageRelation,
    MissingReferenceError,
    ObjectiveDirection,
    PriorIdea,
    PriorIdeaStatus,
    PriorKnowledgeSnapshot,
    PublicationArtifactKind,
    ReviewAssertion,
    RelativeEffect,
    RunBundle,
    ScopeRepositorySettings,
    SourceFileDescriptor,
    TaskAdapterBinding,
    TaskAdapterContextBinding,
    TaskAdapterManifest,
    TaskAdapterRuntimeContract,
    TaskContextBinding,
    TaskEvaluatorBinding,
    TaskEvaluatorMetricComparisonBinding,
    TaskFamilyDefinition,
    TransferAttempt,
    TransferCompatibility,
    TransferEpisode,
    expert_source_replay_matched_compute_digest,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.github.materializer import SourceArchiveExtractionReceipt
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.task_adapters import (
    TaskAdapterVerificationReceipt,
    VerifiedTaskAdapter,
)
from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
TASK_ADAPTER_RUNTIME_LOCK = b"python==3.11.9\n"


def fixture_id(name):
    return content_id("fixture", {"name": name})


def digest(name):
    return tree_or_blob_digest(name.encode("utf-8"))


def task_adapter_source(task_adapter_id, *, source_contents=None):
    selected_source_contents = (
        {
            "adapter.py": f"ADAPTER_ID = {task_adapter_id!r}\n".encode("utf-8"),
            "requirements.lock": TASK_ADAPTER_RUNTIME_LOCK,
        }
        if source_contents is None
        else dict(source_contents)
    )
    source_files = tuple(
        SourceFileDescriptor(
            relative_path=path,
            digest=tree_or_blob_digest(payload),
            mode="100755" if path == "adapter.py" else "100644",
            size=len(payload),
        )
        for path, payload in sorted(selected_source_contents.items())
    )
    tree_hash = source_tree_digest(
        {
            item.relative_path: (item.digest, item.mode, item.size)
            for item in source_files
        }
    )
    return selected_source_contents, source_files, tree_hash


def verified_test_task_adapter(adapter, *, source_contents=None):
    proof_refs = {adapter.sanitation_report_id, *adapter.validation_refs}
    proof_objects = {
        proof_ref: f"proof:{proof_ref}".encode("utf-8") for proof_ref in proof_refs
    }
    source_contents, source_files, source_tree_hash = task_adapter_source(
        adapter.task_adapter_id,
        source_contents=source_contents,
    )
    assert source_tree_hash == adapter.tree_hash
    source_archive = f"archive:{adapter.task_adapter_id}".encode("utf-8")
    publisher_verification = f"publisher-verification:{adapter.task_adapter_id}".encode(
        "utf-8"
    )
    extraction_receipt = SourceArchiveExtractionReceipt.mint(
        artifact_id=adapter.task_adapter_manifest_id,
        source_archive_ref=adapter.source_tree_ref,
        source_archive_digest=tree_or_blob_digest(source_archive),
        source_tree_hash=adapter.tree_hash,
        source_tree_files=source_files,
        extractor_version="kapso.source_archive_extractor.v1",
    )
    verification_receipt = TaskAdapterVerificationReceipt.mint(
        task_adapter_manifest_id=adapter.task_adapter_manifest_id,
        full_manifest_digest=tree_or_blob_digest(adapter.to_json_bytes()),
        publisher_attestation_digest=tree_or_blob_digest(
            canonical_json_bytes(adapter.publisher_attestation)
        ),
        source_extraction_receipt_id=extraction_receipt.extraction_receipt_id,
        source_archive_ref=adapter.source_tree_ref,
        source_archive_digest=tree_or_blob_digest(source_archive),
        source_tree_hash=adapter.tree_hash,
        proof_object_digests={
            proof_ref: tree_or_blob_digest(payload)
            for proof_ref, payload in proof_objects.items()
        },
        publisher_verification_digest=tree_or_blob_digest(publisher_verification),
        verifier_id="test_task_adapter_verifier",
        verifier_version="test.task_adapter_verifier.v1",
    )
    return VerifiedTaskAdapter(
        manifest=adapter,
        verification_receipt=verification_receipt,
        source_extraction_receipt=extraction_receipt,
        source_archive=source_archive,
        source_contents=source_contents,
        proof_objects=proof_objects,
        publisher_verification=publisher_verification,
    )


def operation_receipt(name):
    operation_suffix = digest(name).removeprefix("sha256:")[:32]
    return CodingAgentOperationReceipt.mint(
        operation_id=f"agent_call_{operation_suffix}",
        principal_id=f"reviewer-{name}",
        role="independent_reviewer",
        cli="codex",
        model="gpt-5.6-sol",
        effort="xhigh",
        workspace_access=CodingAgentWorkspaceAccess.READ_ONLY,
        artifact_checksums={
            filename: digest(f"{name}-{filename}")
            for filename in coding_agent_artifact_filenames(
                CodingAgentWorkspaceAccess.READ_ONLY
            )
        },
    )


def assertion(subject_id, name, receipt):
    return ReviewAssertion.mint(
        subject_id=subject_id,
        reviewer_id=f"reviewer-{name}",
        reviewer_role="independent_reviewer",
        rubric_version="rubric-v1",
        judgment="approve",
        rationale="The exact evidence satisfies the configured rubric.",
        exact_evidence_refs=(fixture_id(f"{name}-evidence"),),
        supersedes_assertion_id=None,
        review_operation_ref=receipt.operation_receipt_id,
    )


def build_records(
    *,
    task_adapter_runtime: TaskAdapterRuntimeContract | None = None,
    task_adapter_source_contents: Mapping[str, bytes] | None = None,
    task_evaluator: TaskEvaluatorBinding | None = None,
):
    task_families = (
        TaskFamilyDefinition(
            task_family_id="language_model_post_training",
            capability_tags=("language.training",),
        ),
        TaskFamilyDefinition(
            task_family_id="relational_tabular_prediction",
            capability_tags=("relational.training",),
        ),
    )
    dimensions = (
        ContextDimensionSchema(
            dimension_id="dataset_family",
            value_type=ContextValueType.STRING,
            required=True,
        ),
        ContextDimensionSchema(
            dimension_id="runtime_family",
            value_type=ContextValueType.STRING,
            required=True,
        ),
    )
    adapter_bindings = (
        TaskAdapterBinding(
            task_family_id="language_model_post_training",
            task_adapter_ids=("posttrain",),
        ),
        TaskAdapterBinding(
            task_family_id="relational_tabular_prediction",
            task_adapter_ids=("relbench",),
        ),
    )
    scope = ExpertScopeContract.mint(
        scope_id="ml_ai",
        supersedes_scope_contract_id=None,
        purpose="Share validated machine-learning capabilities.",
        explicit_non_goals=("No benchmark-to-recipe decision tree.",),
        task_family_ontology=task_families,
        task_family_lineage=(),
        artifact_classes=("dataset", "model"),
        required_context_dimensions=("dataset_family", "runtime_family"),
        context_dimension_schemas=dimensions,
        context_dimension_lineage=(),
        task_adapter_contract=adapter_bindings,
        sanitation_policy_ref="kapso.sanitation.v1",
        validation_policy_ref="kapso.validation.v1",
        repository_architecture_constraints=("Keep adapters read-only.",),
    )
    repository_settings = ScopeRepositorySettings(
        scope_id="ml_ai",
        expert_repository="Leeroo-AI/kapso-expert",
        knowledge_repository="Leeroo-AI/kapso-knowledge",
        security_repository="Leeroo-AI/kapso-security",
    )
    task_binding = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
    )
    context = TaskContextBinding.mint(
        scope_contract_id=scope.scope_contract_id,
        scope_id="ml_ai",
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
        capability_tags=("language.training",),
        input_contract_fingerprint=digest("input"),
        target_contract_fingerprint=digest("target"),
        starting_artifact_refs=("artifact/base",),
        method_fingerprint=digest("method"),
        toolchain_fingerprint=digest("toolchain"),
        dependency_runtime_fingerprint=digest("runtime"),
        budget_hardware_envelope={"accelerator": "H100", "hours": 4},
        transfer_dimensions={
            "dataset_family": "instruction",
            "runtime_family": "pytorch",
        },
    )
    selected_source_contents, _, task_adapter_tree_hash = task_adapter_source(
        "posttrain",
        source_contents=task_adapter_source_contents,
    )
    selected_runtime = (
        TaskAdapterRuntimeContract(
            runtime_protocol_version="kapso.task_adapter_runtime.v1",
            image_repository="registry.example/kapso/task-adapter-runtime",
            image_manifest_digest=digest("task-adapter-runtime-image"),
            image_config_digest=digest("task-adapter-runtime-config"),
            dependency_lock_path="requirements.lock",
            dependency_lock_digest=tree_or_blob_digest(TASK_ADAPTER_RUNTIME_LOCK),
            operating_system="linux",
            architecture="amd64",
            architecture_variant=None,
            environment={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        )
        if task_adapter_runtime is None
        else task_adapter_runtime
    )
    task_adapter = TaskAdapterManifest.mint(
        task_adapter_id="posttrain",
        scope_contract_id=scope.scope_contract_id,
        task_family_id="language_model_post_training",
        publisher_attestation={"issuer": "test-publisher", "signature": "adapter"},
        task_evaluator=(
            TaskEvaluatorBinding(
                protocol_version="kapso.task_evaluator.v1",
                executable_path="adapter.py",
                supported_evaluator_fingerprints=(digest("evaluator"),),
                metric_comparison_bindings=(
                    TaskEvaluatorMetricComparisonBinding(
                        evaluator_fingerprint=digest("evaluator"),
                        metric_name="quality",
                        objective_direction=ObjectiveDirection.MAXIMIZE,
                        comparison_dimension_id="quality",
                        comparison_scale=1.0,
                    ),
                ),
            )
            if task_evaluator is None
            else task_evaluator
        ),
        context_binding=TaskAdapterContextBinding(
            consumed_dimension_ids=("dataset_family", "runtime_family"),
        ),
        source_tree_ref="task-adapter.tar.zst",
        tree_hash=task_adapter_tree_hash,
        runtime=selected_runtime,
        sanitation_report_id=fixture_id("adapter-sanitation"),
        validation_refs=("validation/adapter-smoke",),
    )
    verified_adapter = verified_test_task_adapter(
        task_adapter,
        source_contents=selected_source_contents,
    )
    evaluation = EvaluationFingerprint.mint(
        benchmark_id="posttrain",
        dataset_version="v1",
        split_version="public-v1",
        evaluator_fingerprint=digest("evaluator"),
        metric_name="quality",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        fidelity="full",
        fraction=1.0,
        seed_or_replicate_ids=("seed-1",),
        aggregation_protocol="arithmetic-mean",
        judge_version=None,
    )
    bootstrap_review_operation = operation_receipt("bootstrap")
    bootstrap_assertion = assertion(
        fixture_id("bootstrap-candidate"),
        "bootstrap",
        bootstrap_review_operation,
    )
    module = ExpertModuleContract.mint(
        module_id="shared.reproducible_execution",
        version="v1",
        purpose="Produce provenance-bound resumable execution.",
        problem_signals=("Interrupted work loses reproducibility.",),
        inputs=("run contract",),
        outputs=("resumable artifact",),
        preconditions=("writable workspace",),
        incompatibilities=("untracked mutable input",),
        dependency_capability_ids=(),
        incompatible_capability_ids=(),
        resource_bounds={"maximum_workers": 1},
        dependency_license_manifest={"license": "MIT"},
        supporting_episode_ids=(),
        known_failure_episode_ids=(),
        entrypoint_refs=("src/reproducible_execution/__init__.py",),
        test_refs=("tests/test_resume.py",),
        replay_refs=("tests/replay_resume.py",),
    )
    capability_node = ExpertCapabilityNode(
        capability_id="shared.reproducible_execution",
        module_contract_ref=module.module_contract_id,
        owned_paths=("src/reproducible_execution", "tests"),
        task_family_bindings=(
            "language_model_post_training",
            "relational_tabular_prediction",
        ),
    )
    repository_map = ExpertRepositoryMap.mint(
        scope_contract_id=scope.scope_contract_id,
        capability_nodes=(capability_node,),
        dependency_edges=(),
        task_adapter_boundary=ExpertTaskAdapterBoundary(
            adapter_mount_path=".kapso/task-adapter",
            interface_entrypoint_refs=("src/reproducible_execution/__init__.py",),
            inputs=("task contract",),
            outputs=("validated artifact",),
            invariants=("The adapter remains external and read-only.",),
        ),
        validation_entrypoints=("tests/test_resume.py",),
        architecture_invariants=("No task identity defaults.",),
    )
    expert_release = ExpertBaseReleaseManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        scope_id="ml_ai",
        parent_release_ids=(),
        repository_map_ref=repository_map.repository_map_id,
        module_versions={module.module_id: module.version},
        semantic_book_digest=digest("book"),
        configuration_fingerprint=digest("expert-config"),
        source_archive_ref="expert.tar.zst",
        dependency_closure_ids=tuple(
            sorted((repository_map.repository_map_id, bootstrap_assertion.assertion_id))
        ),
        checksums={"expert.tar.zst": digest("expert-archive")},
        test_matrix_results={"fresh_task": "passed"},
        approval_assertion_ids=(bootstrap_assertion.assertion_id,),
        contamination_scanner_version="scanner-v1",
        dependency_lock_hash=digest("lock"),
        compatibility_envelope={"python": ">=3.10"},
        publisher_attestation={"issuer": "test-publisher", "signature": "expert"},
    )
    starting_artifact_payload = b"starting artifact:artifact/base"
    starting_artifact_file = SourceFileDescriptor(
        relative_path="artifact.bin",
        digest=tree_or_blob_digest(starting_artifact_payload),
        mode="100644",
        size=len(starting_artifact_payload),
    )
    starting_artifact = ExpertSourceReplayStartingArtifact.mint(
        starting_artifact_ref="artifact/base",
        mount_path="inputs/base",
        materialized_tree_hash=source_tree_digest(
            {
                starting_artifact_file.relative_path: (
                    starting_artifact_file.digest,
                    starting_artifact_file.mode,
                    starting_artifact_file.size,
                )
            }
        ),
        source_files=(starting_artifact_file,),
    )
    artifact_environment = ArtifactEnvironment.mint(
        kapso_commit="0" * 40,
        expert_base_release_id=expert_release.release_id,
        task_adapter_manifest_id=task_adapter.task_adapter_manifest_id,
        task_adapter_verification_receipt_id=(
            verified_adapter.verification_receipt.verification_receipt_id
        ),
        starting_artifact_content_ids={
            "artifact/base": starting_artifact.starting_artifact_content_id
        },
        dependency_lock_hash=digest("lock"),
    )
    initial_snapshot_id = fixture_id("empty-snapshot")
    security_denylist_snapshot_id = content_id(
        "security-denylist-snapshot",
        {"generation": 1},
    )
    launch = LaunchManifest.mint(
        launch_request_hash=digest("launch-request"),
        scope_id="ml_ai",
        scope_contract_id=scope.scope_contract_id,
        scope_repository_binding_hash=repository_settings.binding_fingerprint,
        configuration_fingerprint=digest("launch-config"),
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
        knowledge_snapshot_id=initial_snapshot_id,
        knowledge_publication_ref=fixture_id("empty-snapshot-publication"),
        expert_base_release_id=expert_release.release_id,
        expert_publication_ref=fixture_id("expert-publication"),
        embedding_space_id=fixture_id("embedding-space"),
        dependency_runtime_contract={"python": ">=3.10"},
        sanitation_policy_generation=1,
        security_denylist_snapshot_id=security_denylist_snapshot_id,
        security_denylist_generation=1,
        expected_source_composition_hash=digest("workspace"),
        publisher_attestation={"issuer": "test-publisher", "signature": "launch"},
    )
    capture = CaptureManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        scope_id="ml_ai",
        run_id="run-1",
        campaign_id="campaign-1",
        capture_generation=0,
        supersedes_capture_manifest_id=None,
        checkpoint_frontier=1,
        capture_watermarks={"events": 1},
        configuration_fingerprint=digest("capture-config"),
        artifact_refs={"checkpoint": "checkpoint.json"},
        checksums={"checkpoint.json": digest("checkpoint")},
        captured_at="2026-07-20T13:00:00Z",
    )
    run_bundle = RunBundle.mint(
        scope_contract_id=scope.scope_contract_id,
        scope_id="ml_ai",
        run_id="run-1",
        campaign_id="campaign-1",
        completion_state=CompletionState.COMPLETE,
        capture_generation=0,
        supersedes_bundle_id=None,
        checkpoint_frontier=1,
        capture_watermarks={"events": 1},
        configuration_fingerprint=digest("capture-config"),
        artifact_completeness={
            "checkpoint": ArtifactCompleteness.PRESENT,
            "logs": ArtifactCompleteness.PRESENT,
        },
        started_at="2026-07-20T12:00:00Z",
        captured_at="2026-07-20T13:00:00Z",
        kapso_commit="0" * 40,
        launch_manifest_id=launch.launch_manifest_id,
        knowledge_snapshot_id=initial_snapshot_id,
        expert_base_release_id=expert_release.release_id,
        task_context_binding=context,
        artifact_environment=artifact_environment,
        capture_descriptor_ref="capture_descriptor.json",
        checkpoint_ref="checkpoint.json",
        execution_event_journal_ref="events.jsonl",
        idea_archive_ref="idea_archive.json",
        experiment_history_ref="experiment_history.json",
        sanitation_report_ref="sanitation_report.json",
        branch_snapshot_refs=("branches/node-1.tar.zst",),
        run_log_refs=("logs/run.log",),
        checksums={
            "branches/node-1.tar.zst": digest("branch"),
            "capture_descriptor.json": digest("descriptor"),
            "checkpoint.json": digest("checkpoint"),
            "events.jsonl": digest("events"),
            "experiment_history.json": digest("history"),
            "idea_archive.json": digest("ideas"),
            "logs/run.log": digest("log"),
            "sanitation_report.json": digest("sanitation"),
        },
    )
    intervention_ref = BundleArtifactRef(
        relative_path="branches/node-1.tar.zst",
        checksum=digest("branch"),
    )
    relative_effect = RelativeEffect(
        evaluation_fingerprint_id=evaluation.evaluation_fingerprint_id,
        metric_name="quality",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        candidate_value=0.8,
        source_parent_value=0.7,
        raw_delta=0.8 - 0.7,
        normalized_delta=0.8 - 0.7,
        uncertainty=None,
        uncertainty_method=EffectUncertaintyMethod.UNAVAILABLE,
    )
    attempt = TransferAttempt(
        execution_revision=0,
        captured_at="2026-07-20T12:55:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluation_fingerprints=(evaluation,),
        score_of_record_fingerprint_id=evaluation.evaluation_fingerprint_id,
        comparison_status=ComparisonStatus.COMPARABLE,
        measurements={"quality": 0.8},
        source_parent_effect=relative_effect,
        intervention_ref=intervention_ref,
        intervention_structure=InterventionStructure.ISOLATED_BY_ABLATION,
        feedback=("Improved validation quality.",),
        technical_difficulties=(),
        confounders=(),
    )
    episode = TransferEpisode.mint(
        source={
            "scope_id": "ml_ai",
            "run_id": "run-1",
            "campaign_id": "campaign-1",
            "node_id": "node-1",
            "idea_id": "idea-1",
            "batch_id": "batch-1",
        },
        source_bundle_id=run_bundle.bundle_id,
        supersedes_projection_id=None,
        task_context_binding=context,
        artifact_environment=artifact_environment,
        proposal="Validate representation parity before training.",
        parent_episode_ref=None,
        attempts=(attempt,),
        terminal_attempt_revision=0,
        safe_observation_refs=(),
        sanitation_report_id=fixture_id("episode-sanitation"),
        derivation_refs=(run_bundle.bundle_id,),
    )
    prior_idea = PriorIdea.mint(
        source_bundle_id=run_bundle.bundle_id,
        supersedes_projection_id=None,
        source={
            "scope_id": "ml_ai",
            "run_id": "run-1",
            "campaign_id": "campaign-1",
            "batch_id": "batch-1",
            "idea_id": "idea-2",
        },
        proposal="Explore a different packing policy.",
        descriptor={
            "approach_family": "packing",
            "expected_effect": "reduce padding",
            "intervention_target": "batch construction",
            "mechanism": "length grouping",
        },
        assumptions=("Examples have varied lengths.",),
        source_status=PriorIdeaStatus.DEFERRED,
        source_rationale="The run budget ended.",
        source_evidence_refs=("local-evidence-1",),
        task_context_binding=context,
        sanitation_report_id=fixture_id("prior-sanitation"),
    )
    claim_review_operation = operation_receipt("claim")
    claim_assertion = assertion(
        episode.episode_id,
        "claim",
        claim_review_operation,
    )
    claim = KnowledgeClaim.mint(
        claim_id="claim-template-parity",
        scope_contract_id=scope.scope_contract_id,
        statement="Representation parity reduces formatting regressions.",
        mechanism="The same rendering contract removes train/inference drift.",
        applicability_predicates={"dataset_family": "instruction"},
        explicit_exclusions=("Completion-only datasets are untested.",),
        supporting_episode_ids=(episode.episode_id,),
        contradicting_episode_ids=(),
        proposal_provenance={"operation": "codex-cli", "source": episode.episode_id},
        supersedes_revision_ids=(),
    )
    catalog_state = CatalogEntryState.mint(
        subject_payload_id=claim.revision_id,
        catalog_generation=1,
        predecessor_state_id=None,
        configuration_fingerprint=digest("catalog-config"),
        admission_state=AdmissionState.ADMITTED,
        superseded_by_payload_ids=(),
        assertion_ids=(claim_assertion.assertion_id,),
        revocation_ids=(),
        taint_source_ids=(),
    )
    proof_ids = tuple(
        sorted(
            (
                episode.episode_id,
                prior_idea.prior_idea_id,
                claim.revision_id,
                catalog_state.catalog_entry_state_id,
                claim_assertion.assertion_id,
            )
        )
    )
    sidecar = EmbeddingSidecar(
        embedding_space_id=fixture_id("embedding-space"),
        asset_ref="knowledge-search.tar.zst",
        checksum=digest("search-sidecar"),
    )
    snapshot = KnowledgeSnapshotManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        scope_id="ml_ai",
        parent_snapshot_ids=(initial_snapshot_id,),
        included_bundle_ids=(run_bundle.bundle_id,),
        admitted_episode_ids=(episode.episode_id,),
        admitted_prior_idea_ids=(prior_idea.prior_idea_id,),
        active_claim_revision_ids=(claim.revision_id,),
        catalog_generation=1,
        configuration_fingerprint=digest("knowledge-config"),
        entry_state_refs=(catalog_state.catalog_entry_state_id,),
        included_assertion_ids=(claim_assertion.assertion_id,),
        included_revocation_ids=(),
        proof_dependency_closure_ids=proof_ids,
        sanitation_policy_version="kapso.sanitation.v1",
        retrieval_policy_version="kapso.retrieval.v1",
        embedding_sidecars=(sidecar,),
        prompt_budget_policy={"maximum_records": 24},
        checksums={
            "knowledge-search.tar.zst": digest("search-sidecar"),
            "snapshot.json": digest("snapshot"),
        },
        published_at="2026-07-20T14:00:00Z",
        publisher_attestation={"issuer": "test-publisher", "signature": "snapshot"},
    )
    selected_records = (
        {
            "record_id": claim.revision_id,
            "record_kind": "knowledge_claim",
            "payload": claim.to_dict(),
        },
    )
    prior_snapshot = PriorKnowledgeSnapshot.mint(
        source_snapshot_id=snapshot.snapshot_id,
        query={"problem": "Improve post-training reliability."},
        retrieval_policy_version="kapso.retrieval.v1",
        task_context_binding_id=context.task_context_binding_id,
        selected_records=selected_records,
        selected_record_ids=(claim.revision_id,),
        proof_reference_ids=(episode.episode_id,),
        selection_metadata={
            claim.revision_id: {
                "compatibility": "exact_context",
                "evidence_quality": 1,
                "lexical_score": 1.0,
                "outcome": "inconclusive",
                "proof_reference_ids": (episode.episode_id,),
                "rank": 0,
                "recency": "",
                "retrieval_utility": 1.0,
                "semantic_score": 0.0,
            }
        },
        prompt_budget_policy={"maximum_records": 24},
        records_digest=tree_or_blob_digest(canonical_json_bytes(selected_records)),
    )
    candidate_source = b"def resume():\n    return 'reproducible'\n"
    candidate_test = b"def test_resume():\n    assert True\n"
    candidate_replay = b"def replay():\n    return 'replayed'\n"
    candidate_book = compile_expert_semantic_book(scope, repository_map, (module,))
    candidate_contents = {
        "src/reproducible_execution/__init__.py": candidate_source,
        "tests/replay_resume.py": candidate_replay,
        "tests/test_resume.py": candidate_test,
        EXPERT_BOOK_PATH: candidate_book,
        EXPERT_REPOSITORY_MAP_PATH: repository_map.to_json_bytes(),
        expert_module_contract_path(module.module_contract_id): module.to_json_bytes(),
    }
    candidate_files = tuple(
        SourceFileDescriptor(
            relative_path=path,
            digest=tree_or_blob_digest(candidate_contents[path]),
            mode="100644",
            size=len(candidate_contents[path]),
        )
        for path in sorted(candidate_contents)
    )
    candidate_tree_hash = source_tree_digest(
        {
            file.relative_path: (file.digest, file.mode, file.size)
            for file in candidate_files
        }
    )
    candidate_tree = ExpertSourceTreeManifest.mint(
        tree_hash=candidate_tree_hash,
        files=candidate_files,
    )
    candidate_declared_paths = (
        "src/reproducible_execution/__init__.py",
        "tests/replay_resume.py",
        "tests/test_resume.py",
    )
    candidate_editable_tree_hash = source_tree_digest(
        {
            file.relative_path: (file.digest, file.mode, file.size)
            for file in candidate_files
            if file.relative_path in set(candidate_declared_paths)
        }
    )
    candidate_patch = ExpertCandidatePatch.mint(
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        candidate_tree_hash=candidate_tree_hash,
        changes=tuple(
            ExpertCandidatePatchChange(
                relative_path=file.relative_path,
                before=None,
                after=file,
            )
            for file in candidate_files
        ),
    )
    trigger_decision_id = fixture_id("expert-trigger-decision")
    trigger_packet_id = fixture_id("expert-trigger-packet")
    candidate_proposer_authority = ExpertProposerAuthority.mint(
        principal_id="expert-architect",
        role="expert_architect",
        cli="claude_code",
        model="fable",
        effort="xhigh",
        timeout_seconds=600,
        allowed_tools=("Edit", "Read"),
        workspace_access=CodingAgentWorkspaceAccess.EDIT_WORKSPACE,
        workspace_maximum_entries=1000,
        workspace_maximum_bytes=1_000_000,
        sensitive_file_glob_scan_max_depth=4,
    )
    candidate_operation_preimage = {
        "ancestor_candidate_ids": (),
        "configuration_fingerprint": digest("expert-validation-config"),
        "input_artifact_checksums": {
            name: digest(f"candidate-{name}")
            for name in (
                "invocation.json",
                "prior_knowledge.json",
                "prompt.txt",
                "response_schema.json",
            )
        },
        "mcp_configuration_fingerprint": digest("candidate-mcp-configuration"),
        "operation_kind": ExpertCandidateOperationKind.BOOTSTRAP.value,
        "parent_tree_hash": EMPTY_EXPERT_TREE_DIGEST,
        "principal_id": "expert-architect",
        "proposer_authority_id": candidate_proposer_authority.authority_id,
        "proposal_contract_version": "kapso.expert_proposal.v1",
        "proposal_packet_digest": digest("candidate-proposal-packet"),
        "trigger_decision_id": trigger_decision_id,
        "trigger_evidence_packet_id": trigger_packet_id,
    }
    candidate_final_output = (
        canonical_json_bytes(
            {
                "changed_paths": candidate_declared_paths,
                "deleted_paths": (),
                "summary": "Added reproducible resume support.",
            }
        ).decode("utf-8")
        + "\n"
    )
    candidate_operation_id = (
        "agent_call_"
        + tree_or_blob_digest(canonical_json_bytes(candidate_operation_preimage))[7:39]
    )
    candidate_workspace_delta = CodingAgentWorkspaceDelta.mint(
        baseline_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        edited_tree_hash=candidate_editable_tree_hash,
        changed_files=tuple(
            CodingAgentWorkspaceChangedFile(
                before=None,
                after=file,
                content_base64=base64.b64encode(
                    candidate_contents[file.relative_path]
                ).decode("ascii"),
            )
            for file in candidate_files
            if file.relative_path in set(candidate_declared_paths)
        ),
        deleted_files=(),
    )
    candidate_operation_receipt = CodingAgentOperationReceipt.mint(
        operation_id=candidate_operation_id,
        principal_id="expert-architect",
        role="expert_architect",
        cli="claude_code",
        model="fable",
        effort="xhigh",
        workspace_access=CodingAgentWorkspaceAccess.EDIT_WORKSPACE,
        artifact_checksums={
            filename: (
                tree_or_blob_digest(candidate_final_output.encode("utf-8"))
                if filename == "final.json"
                else (
                    tree_or_blob_digest(candidate_workspace_delta.to_json_bytes())
                    if filename == "workspace-delta.json"
                    else digest(f"candidate-{filename}")
                )
            )
            for filename in coding_agent_artifact_filenames(
                CodingAgentWorkspaceAccess.EDIT_WORKSPACE
            )
        },
    )
    candidate_workspace_receipt = ExpertCandidateWorkspaceReceipt.mint(
        operation_receipt_id=candidate_operation_receipt.operation_receipt_id,
        operation_id=candidate_operation_id,
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        editable_parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        edited_tree_hash=candidate_editable_tree_hash,
        changed_paths=candidate_declared_paths,
        deleted_paths=(),
    )
    candidate_operation = ExpertCandidateOperationRecord.mint(
        operation_kind=ExpertCandidateOperationKind.BOOTSTRAP,
        trigger_decision_id=trigger_decision_id,
        trigger_evidence_packet_id=trigger_packet_id,
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        ancestor_candidate_ids=(),
        configuration_fingerprint=digest("expert-validation-config"),
        proposer_authority=candidate_proposer_authority,
        operation_preimage=candidate_operation_preimage,
        operation_receipt=candidate_operation_receipt,
        workspace_receipt=candidate_workspace_receipt,
        workspace_delta_ref=candidate_workspace_delta.workspace_delta_id,
        workspace_delta_digest=tree_or_blob_digest(
            candidate_workspace_delta.to_json_bytes()
        ),
        final_output=candidate_final_output,
    )
    candidate_sanitation = ExpertCandidateSanitationReport.mint(
        scope_contract_id=scope.scope_contract_id,
        candidate_tree_hash=candidate_tree_hash,
        policy_version="kapso.expert_candidate_sanitation.v1",
        policy_fingerprint=digest("candidate-sanitation-policy"),
        scanner_version="kapso.expert_candidate_scanner.v1",
        status=ExpertCandidateSanitationStatus.ADMITTED,
        scanned_files=candidate_files,
        findings=(),
    )
    candidate = ExpertCandidateManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        change_kind=CandidateChangeKind.REPOSITORY_ARCHITECTURE,
        parent_release_id=None,
        parent_repository_map_ref=None,
        parent_tree_hash=EMPTY_EXPERT_TREE_DIGEST,
        trigger_decision_id=trigger_decision_id,
        trigger_evidence_packet_id=trigger_packet_id,
        patch_ref=candidate_patch.patch_id,
        patch_digest=tree_or_blob_digest(candidate_patch.to_json_bytes()),
        candidate_tree_ref=candidate_tree.source_tree_manifest_id,
        candidate_tree_hash=candidate_tree_hash,
        configuration_fingerprint=digest("expert-validation-config"),
        module_contract_refs=(module.module_contract_id,),
        proposed_repository_map_ref=repository_map.repository_map_id,
        semantic_book_digest=expert_semantic_book_digest(candidate_book),
        proposer_operation_record_id=candidate_operation.operation_record_id,
        source_dependency_ids=tuple(
            sorted((claim.revision_id, trigger_decision_id, trigger_packet_id))
        ),
        ancestor_candidate_ids=(),
        capability_lineage=(),
        sanitation_report_id=candidate_sanitation.sanitation_report_id,
    )
    publication_asset = GitHubReleaseAsset(
        asset_id="asset-1",
        name="knowledge-snapshot.tar.zst",
        media_type="application/zstd",
        size=1024,
        sha256=digest("release-asset"),
    )
    publication = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        artifact_id=snapshot.snapshot_id,
        repository_node_id="repository-node-1",
        repository_full_name="Leeroo-AI/kapso-knowledge",
        commit_sha="1" * 40,
        immutable_release_id="release-1",
        tag="knowledge/S000001",
        assets=(publication_asset,),
        release_attestation_ref="attestations/release-1",
        published_at="2026-07-20T14:05:00Z",
        publisher_identity="leeroo-coder",
    )
    bootstrap = BootstrapPin.mint(
        launch_manifest_id=launch.launch_manifest_id,
        launch_request_hash=launch.launch_request_hash,
        scope_id="ml_ai",
        scope_contract_id=scope.scope_contract_id,
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
        knowledge_snapshot_id=initial_snapshot_id,
        expert_base_release_id=expert_release.release_id,
        task_adapter_manifest_id=task_adapter.task_adapter_manifest_id,
        security_denylist_snapshot_id=security_denylist_snapshot_id,
        security_denylist_generation=1,
        workspace_tree_hash=digest("workspace"),
        created_at="2026-07-20T12:00:00Z",
    )
    dependency_edge = ExpertDependencyEdge(
        source_capability_id="language.template_parity",
        target_capability_id="shared.reproducible_execution",
    )
    lineage = LineageEdge(
        source_ids=("old-capability",),
        target_ids=("new-capability",),
        relation=LineageRelation.RENAME,
    )
    return (
        repository_settings,
        task_binding,
        *task_families,
        *dimensions,
        *adapter_bindings,
        scope,
        context,
        evaluation,
        bootstrap_review_operation,
        bootstrap_assertion,
        module,
        capability_node,
        dependency_edge,
        repository_map,
        expert_release,
        artifact_environment,
        task_adapter,
        launch,
        capture,
        run_bundle,
        attempt,
        episode,
        prior_idea,
        claim_review_operation,
        claim_assertion,
        claim,
        catalog_state,
        sidecar,
        snapshot,
        prior_snapshot,
        candidate_tree,
        candidate_patch,
        candidate_workspace_receipt,
        candidate_operation,
        candidate_sanitation,
        candidate,
        publication_asset,
        publication,
        bootstrap,
        lineage,
    )


def test_every_contract_round_trips_through_canonical_json():
    for record in build_records():
        restored = type(record).from_json_bytes(record.to_json_bytes())
        assert restored == record, type(record).__name__
        assert restored.to_json_bytes() == record.to_json_bytes()


def test_scope_requires_named_policy_authorities():
    scope = next(
        record for record in build_records() if isinstance(record, ExpertScopeContract)
    )

    with pytest.raises(CanonicalizationError, match="sanitation_policy_ref"):
        ExpertScopeContract.mint(
            **{
                key: value
                for key, value in scope.to_dict().items()
                if key not in {"scope_contract_id", "sanitation_policy_ref"}
            },
            sanitation_policy_ref="",
        )


def test_nested_contract_has_stable_golden_bytes_and_identity():
    attempt = next(
        record for record in build_records() if isinstance(record, TransferAttempt)
    )

    assert attempt.to_json_bytes() == (
        b'{"captured_at":"2026-07-20T12:55:00Z","comparison_status":"comparable",'
        b'"confounders":[],"evaluation_fingerprints":[{"aggregation_protocol":'
        b'"arithmetic-mean","benchmark_id":"posttrain","dataset_version":"v1",'
        b'"evaluation_fingerprint_id":"evaluation-fingerprint:sha256:'
        b'746d9917d4b68f1398a01b3c41cd97f367b9e69c0a170654de99375abbfec5de",'
        b'"evaluator_fingerprint":"sha256:'
        b'27f6343f980c4c0ab821d9c72d211d0eb76b0853825cefd80476a05e43cab27e",'
        b'"fidelity":"full","fraction":1.0,"judge_version":null,"metric_name":'
        b'"quality","objective_direction":"maximize","seed_or_replicate_ids":'
        b'["seed-1"],"split_version":"public-v1"}],"evaluation_status":"valid",'
        b'"execution_revision":0,"execution_status":"completed","feedback":'
        b'["Improved validation quality."],"intervention_ref":{"checksum":'
        b'"sha256:f38c764c8aa00b6578f4254a4dc6d9b50f88fa926e270ea7859bd1b707cd8662",'
        b'"relative_path":"branches/node-1.tar.zst"},"intervention_structure":'
        b'"isolated_by_ablation","measurements":{"quality":0.8},'
        b'"score_of_record_fingerprint_id":"evaluation-fingerprint:sha256:'
        b'746d9917d4b68f1398a01b3c41cd97f367b9e69c0a170654de99375abbfec5de",'
        b'"source_parent_effect":{"candidate_value":0.8,'
        b'"evaluation_fingerprint_id":"evaluation-fingerprint:sha256:'
        b'746d9917d4b68f1398a01b3c41cd97f367b9e69c0a170654de99375abbfec5de",'
        b'"metric_name":"quality","normalized_delta":0.10000000000000009,'
        b'"objective_direction":"maximize","raw_delta":0.10000000000000009,'
        b'"source_parent_value":0.7,"uncertainty":null,'
        b'"uncertainty_method":"unavailable"},'
        b'"technical_difficulties":[]}'
    )
    assert content_id("transfer-attempt-fixture", attempt) == (
        "transfer-attempt-fixture:sha256:"
        "021c65bc04cbc3abf2132aa51dcddeb3a9f9691ddb86c6840794df5232855040"
    )


def test_contracts_reject_missing_unknown_and_wrongly_typed_fields():
    evaluation = next(
        record
        for record in build_records()
        if isinstance(record, EvaluationFingerprint)
    )
    payload = evaluation.to_dict()

    missing = dict(payload)
    missing.pop("metric_name")
    with pytest.raises(ContractValidationError):
        EvaluationFingerprint.from_dict(missing)

    unknown = dict(payload)
    unknown["model_name"] = "domain-specific-leak"
    with pytest.raises(ContractValidationError):
        EvaluationFingerprint.from_dict(unknown)

    boolean_integer = dict(payload)
    boolean_integer["fraction"] = True
    with pytest.raises(ContractValidationError):
        EvaluationFingerprint.from_dict(boolean_integer)


def test_task_adapter_manifest_has_one_typed_scientific_contract():
    manifest = next(
        record for record in build_records() if isinstance(record, TaskAdapterManifest)
    )
    payload = manifest.to_dict()

    assert set(payload) >= {"task_evaluator", "context_binding", "runtime"}
    assert not {
        "task_evaluator_binding",
        "context_dimension_binding",
        "dependency_runtime_contract",
    } & set(payload)
    comparison_binding = manifest.task_evaluator.metric_comparison_bindings[0]
    assert comparison_binding.comparison_dimension_id == "quality"
    assert comparison_binding.comparison_scale == 1.0

    legacy_payload = dict(payload)
    legacy_payload["task_evaluator_binding"] = legacy_payload.pop("task_evaluator")
    with pytest.raises(ContractValidationError, match="fields"):
        TaskAdapterManifest.from_dict(legacy_payload)

    with pytest.raises(ContractValidationError, match="normalized relative path"):
        replace(
            manifest,
            task_evaluator=replace(
                manifest.task_evaluator,
                executable_path="../adapter.py",
            ),
        )
    with pytest.raises(ContractValidationError, match="finite positive"):
        replace(comparison_binding, comparison_scale=0.0)
    with pytest.raises(ContractValidationError, match="cover every supported"):
        replace(manifest.task_evaluator, metric_comparison_bindings=())
    with pytest.raises(ContractValidationError, match="cover every supported"):
        replace(
            manifest.task_evaluator,
            metric_comparison_bindings=(
                replace(
                    comparison_binding,
                    evaluator_fingerprint=digest("unsupported-evaluator"),
                ),
            ),
        )
    with pytest.raises(ContractValidationError, match="image_repository"):
        replace(
            manifest,
            runtime=replace(manifest.runtime, image_repository="runtime:latest"),
        )
    with pytest.raises(ContractValidationError, match="explicit registry"):
        replace(
            manifest,
            runtime=replace(
                manifest.runtime,
                image_repository="namespace/runtime",
            ),
        )
    with pytest.raises(ContractValidationError, match="image_manifest_digest"):
        replace(
            manifest,
            runtime=replace(
                manifest.runtime,
                image_manifest_digest="sha256:not-a-manifest-digest",
            ),
        )
    assert manifest.runtime.image_reference == (
        f"{manifest.runtime.image_repository}@{manifest.runtime.image_manifest_digest}"
    )
    assert dict(manifest.runtime.environment) == {
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    with pytest.raises(ContractValidationError, match="non-empty PATH"):
        replace(
            manifest,
            runtime=replace(
                manifest.runtime,
                environment={"LANG": "C.UTF-8"},
            ),
        )
    with pytest.raises(ContractValidationError, match="key-sorted"):
        replace(
            manifest,
            runtime=replace(
                manifest.runtime,
                environment={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            ),
        )
    for secret_key in (
        "API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "DOCKER_AUTH_CONFIG",
        "GITHUB_PAT",
        "NETRC",
    ):
        with pytest.raises(ContractValidationError, match="non-secret"):
            replace(
                manifest,
                runtime=replace(
                    manifest.runtime,
                    environment={secret_key: "forbidden"},
                ),
            )
    for unsafe_value in (
        "contains\ncontrol",
        "contains\u0085control",
        "contains\u202econtrol",
        "contains\u2028control",
        "contains\ud800control",
    ):
        with pytest.raises(ContractValidationError, match="non-secret"):
            replace(
                manifest,
                runtime=replace(
                    manifest.runtime,
                    environment={"LANG": unsafe_value},
                ),
            )
    legacy_runtime = manifest.runtime.to_dict()
    legacy_runtime.pop("image_repository")
    legacy_runtime.pop("image_manifest_digest")
    legacy_runtime.pop("image_config_digest")
    legacy_runtime["image_digest"] = digest("legacy-ambiguous-image")
    with pytest.raises(ContractValidationError, match="fields"):
        TaskAdapterRuntimeContract.from_dict(legacy_runtime)
    with pytest.raises(ContractValidationError, match="non-empty, sorted, and unique"):
        replace(
            manifest.task_evaluator,
            supported_evaluator_fingerprints=(),
        )
    assert TaskAdapterContextBinding(consumed_dimension_ids=()).to_dict() == {
        "consumed_dimension_ids": []
    }

    relbench_values = manifest.to_dict()
    relbench_values.pop("task_adapter_manifest_id")
    relbench_values.update(
        task_adapter_id="relbench",
        task_family_id="relational_tabular_prediction",
        context_binding=TaskAdapterContextBinding(
            consumed_dimension_ids=("dataset_family",)
        ),
    )
    relbench_manifest = TaskAdapterManifest.mint(**relbench_values)
    assert TaskAdapterManifest.from_json_bytes(relbench_manifest.to_json_bytes()) == (
        relbench_manifest
    )
    assert relbench_manifest.runtime == manifest.runtime


def test_matched_compute_digest_binds_every_shared_scientific_input():
    fingerprint_id = content_id("evaluation-fingerprint", {"name": "score"})
    inputs = {
        "bundle_lineage_ids": (content_id("run-bundle", {"generation": 0}),),
        "projection_manifest_id": content_id("projection", {"name": "p"}),
        "episode_id": content_id("episode", {"name": "e"}),
        "source_execution_revision": 0,
        "source_evaluation_fingerprint_ids": (fingerprint_id,),
        "source_score_of_record_fingerprint_id": fingerprint_id,
        "task_context_binding_id": content_id("context", {"name": "c"}),
        "context_materialization_receipt_id": content_id(
            "context-receipt", {"name": "r"}
        ),
        "starting_artifact_content_ids": (content_id("artifact", {"name": "a"}),),
        "task_adapter_manifest_id": content_id("adapter", {"name": "m"}),
        "verification_receipt_id": content_id("verification", {"name": "v"}),
        "task_adapter_source_tree_hash": digest("adapter-tree"),
        "task_evaluator_digest": digest("evaluator-contract"),
        "task_adapter_runtime_digest": digest("runtime-contract"),
        "task_adapter_context_binding_digest": digest("context-contract"),
        "compute_binding_id": content_id("compute-binding", {"name": "limits"}),
    }
    replacements = {
        "bundle_lineage_ids": (content_id("run-bundle", {"generation": 1}),),
        "projection_manifest_id": content_id("projection", {"name": "changed"}),
        "episode_id": content_id("episode", {"name": "changed"}),
        "source_execution_revision": 1,
        "source_evaluation_fingerprint_ids": (
            content_id("evaluation-fingerprint", {"name": "diagnostic"}),
            fingerprint_id,
        ),
        "source_score_of_record_fingerprint_id": content_id(
            "evaluation-fingerprint", {"name": "other-score"}
        ),
        "task_context_binding_id": content_id("context", {"name": "changed"}),
        "context_materialization_receipt_id": content_id(
            "context-receipt", {"name": "changed"}
        ),
        "starting_artifact_content_ids": (content_id("artifact", {"name": "changed"}),),
        "task_adapter_manifest_id": content_id("adapter", {"name": "changed"}),
        "verification_receipt_id": content_id("verification", {"name": "changed"}),
        "task_adapter_source_tree_hash": digest("changed-adapter-tree"),
        "task_evaluator_digest": digest("changed-evaluator-contract"),
        "task_adapter_runtime_digest": digest("changed-runtime-contract"),
        "task_adapter_context_binding_digest": digest("changed-context-contract"),
        "compute_binding_id": content_id("compute-binding", {"name": "changed-limits"}),
    }
    baseline = expert_source_replay_matched_compute_digest(**inputs)

    for field_name, changed_value in replacements.items():
        changed_inputs = {**inputs, field_name: changed_value}
        assert (
            expert_source_replay_matched_compute_digest(**changed_inputs) != baseline
        ), field_name


def _source_replay_compute_binding():
    return ExpertSourceReplayComputeBinding.mint(
        paired_execution_protocol_version="kapso.paired-execution.v1",
        execution_provider_id="docker",
        execution_provider_version="docker-provider-v1",
        execution_provider_settings_digest=digest("docker-provider-settings-v1"),
        sandbox_policy_version="offline-readonly-v1",
        leg_wall_time_limit_seconds=600,
        termination_grace_seconds=10,
        cpu_millicore_limit=8000,
        memory_byte_limit=32 * 1024**3,
        shared_memory_byte_limit=1024**3,
        process_limit=4096,
        open_file_limit=4096,
        writable_inode_limit=10_000,
        writable_storage_byte_limit=4 * 1024**3,
        output_entry_limit=1000,
        output_byte_limit=1024**3,
        stdout_byte_limit=16 * 1024**2,
        stderr_byte_limit=16 * 1024**2,
        accelerator_class_id=None,
        accelerator_count=0,
        leg_order=(
            ExpertSourceReplayExecutionLegKind.CONTROL_PARENT,
            ExpertSourceReplayExecutionLegKind.CANDIDATE,
        ),
    )


def test_source_replay_compute_binding_is_canonical_and_every_field_is_bound():
    binding = _source_replay_compute_binding()
    payload = binding.to_dict()
    replacements = {
        "paired_execution_protocol_version": "kapso.paired-execution.v2",
        "execution_provider_id": "isolated-docker",
        "execution_provider_version": "docker-provider-v2",
        "execution_provider_settings_digest": digest("docker-provider-settings-v2"),
        "sandbox_policy_version": "offline-readonly-v2",
        "leg_wall_time_limit_seconds": 601,
        "termination_grace_seconds": 11,
        "cpu_millicore_limit": 8001,
        "memory_byte_limit": binding.memory_byte_limit + 1,
        "shared_memory_byte_limit": binding.shared_memory_byte_limit + 1,
        "process_limit": 4097,
        "open_file_limit": 4097,
        "writable_inode_limit": 10_001,
        "writable_storage_byte_limit": binding.writable_storage_byte_limit + 1,
        "output_entry_limit": 1001,
        "output_byte_limit": binding.output_byte_limit + 1,
        "stdout_byte_limit": binding.stdout_byte_limit + 1,
        "stderr_byte_limit": binding.stderr_byte_limit + 1,
        "accelerator_class_id": "h100",
        "accelerator_count": 1,
        "leg_order": (
            ExpertSourceReplayExecutionLegKind.CANDIDATE.value,
            ExpertSourceReplayExecutionLegKind.CONTROL_PARENT.value,
        ),
    }

    assert ExpertSourceReplayComputeBinding.from_json_bytes(
        binding.to_json_bytes()
    ) == (binding)
    for field_name, changed_value in replacements.items():
        changed_payload = {
            **payload,
            field_name: changed_value,
        }
        changed_payload.pop("compute_binding_id")
        if field_name in {"accelerator_class_id", "accelerator_count"}:
            changed_payload.update(
                accelerator_class_id="h100",
                accelerator_count=1,
            )
        changed = ExpertSourceReplayComputeBinding.mint(**changed_payload)
        assert changed.compute_binding_id != binding.compute_binding_id, field_name


@pytest.mark.parametrize(
    ("field_name", "changed_value", "error"),
    (
        ("cpu_millicore_limit", True, "must be an integer"),
        ("memory_byte_limit", 0, "positive integer"),
        ("accelerator_count", True, "must be an integer"),
        ("accelerator_count", 1, "present together"),
        ("termination_grace_seconds", 601, "internally inconsistent"),
        ("shared_memory_byte_limit", 32 * 1024**3 + 1, "internally inconsistent"),
        ("output_entry_limit", 10_001, "internally inconsistent"),
        ("output_byte_limit", 4 * 1024**3 + 1, "internally inconsistent"),
        (
            "leg_order",
            ("control_parent", "control_parent"),
            "both legs exactly once",
        ),
    ),
)
def test_source_replay_compute_binding_rejects_invalid_envelopes(
    field_name,
    changed_value,
    error,
):
    payload = _source_replay_compute_binding().to_dict()
    payload.pop("compute_binding_id")
    payload[field_name] = changed_value

    with pytest.raises(ContractValidationError, match=error):
        ExpertSourceReplayComputeBinding.mint(**payload)


def test_content_mutation_is_detected_but_attestation_rotation_preserves_identity():
    release = next(
        record
        for record in build_records()
        if isinstance(record, ExpertBaseReleaseManifest)
    )
    rotated = replace(
        release,
        publisher_attestation={"issuer": "rotated", "signature": "new"},
    )
    assert rotated.release_id == release.release_id

    with pytest.raises(CanonicalizationError):
        replace(release, semantic_book_digest=digest("changed-book"))
    with pytest.raises(ContractValidationError, match="supported release asset"):
        replace(release, source_archive_ref="nested/expert.tar.zst")


def test_scope_validates_both_families_without_domain_conditionals():
    records = build_records()
    scope = next(
        record for record in records if isinstance(record, ExpertScopeContract)
    )
    posttrain_context = next(
        record for record in records if isinstance(record, TaskContextBinding)
    )
    posttrain_context.validate_against(scope)
    relbench_context = TaskContextBinding.mint(
        scope_contract_id=scope.scope_contract_id,
        scope_id="ml_ai",
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
        capability_tags=("relational.training",),
        input_contract_fingerprint=digest("rel-input"),
        target_contract_fingerprint=digest("rel-target"),
        starting_artifact_refs=("artifact/rel-base",),
        method_fingerprint=digest("rel-method"),
        toolchain_fingerprint=digest("rel-tools"),
        dependency_runtime_fingerprint=digest("rel-runtime"),
        budget_hardware_envelope={"accelerator": "H100"},
        transfer_dimensions={
            "dataset_family": "relational",
            "runtime_family": "pytorch",
        },
    )
    relbench_context.validate_against(scope)

    assert posttrain_context.compatibility_with(posttrain_context) is (
        TransferCompatibility.EXACT_CONTEXT
    )
    assert posttrain_context.compatibility_with(relbench_context) is (
        TransferCompatibility.ANALOGICAL
    )


def test_both_task_families_resolve_one_scope_without_erasing_binding_identity():
    records = build_records()
    scope = next(
        record for record in records if isinstance(record, ExpertScopeContract)
    )
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    posttrain = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="language_model_post_training",
        task_adapter_id="posttrain",
    )
    relbench = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
    )

    posttrain_repositories = settings.resolve_binding(posttrain, scope)
    relbench_repositories = settings.resolve_binding(relbench, scope)

    assert posttrain_repositories == relbench_repositories
    assert posttrain.task_family_id != relbench.task_family_id
    assert posttrain.task_adapter_id != relbench.task_adapter_id


def test_run_bundle_orders_mixed_precision_timestamps_by_instant():
    bundle = next(record for record in build_records() if isinstance(record, RunBundle))
    payload = {
        key: value
        for key, value in bundle.to_dict().items()
        if key not in {"bundle_id", "started_at", "captured_at"}
    }

    RunBundle.mint(
        **payload,
        started_at="2026-07-20T12:00:00Z",
        captured_at="2026-07-20T12:00:00.100000Z",
    )
    with pytest.raises(ContractValidationError):
        RunBundle.mint(
            **payload,
            started_at="2026-07-20T12:00:00.100000Z",
            captured_at="2026-07-20T12:00:00Z",
        )


def test_composite_records_reject_conflicting_scope_identities():
    records = build_records()
    context = next(
        record for record in records if isinstance(record, TaskContextBinding)
    )
    bundle = next(record for record in records if isinstance(record, RunBundle))
    episode = next(record for record in records if isinstance(record, TransferEpisode))
    other_context = TaskContextBinding.mint(
        **{
            key: value
            for key, value in context.to_dict().items()
            if key not in {"task_context_binding_id", "scope_id"}
        },
        scope_id="other_scope",
    )

    with pytest.raises(IncompatibleArtifactError):
        RunBundle.mint(
            **{
                key: value
                for key, value in bundle.to_dict().items()
                if key not in {"bundle_id", "task_context_binding"}
            },
            task_context_binding=other_context,
        )

    with pytest.raises(IncompatibleArtifactError):
        TransferEpisode.mint(
            **{
                key: value
                for key, value in episode.to_dict().items()
                if key not in {"episode_id", "source"}
            },
            source={**episode.source, "scope_id": "other_scope"},
        )


def test_context_rejects_unregistered_dimensions_and_wrong_scope_revision():
    records = build_records()
    scope = next(
        record for record in records if isinstance(record, ExpertScopeContract)
    )
    context = next(
        record for record in records if isinstance(record, TaskContextBinding)
    )
    unknown_dimension = TaskContextBinding.mint(
        **{
            key: value
            for key, value in context.to_dict().items()
            if key != "task_context_binding_id" and key != "transfer_dimensions"
        },
        transfer_dimensions={
            **context.transfer_dimensions,
            "model_name": "forbidden-core-dimension",
        },
    )
    with pytest.raises(ContractValidationError):
        unknown_dimension.validate_against(scope)

    other_scope = ExpertScopeContract.mint(
        **{
            key: value
            for key, value in scope.to_dict().items()
            if key not in {"scope_contract_id", "purpose"}
        },
        purpose="A different immutable revision.",
    )
    with pytest.raises(IncompatibleArtifactError):
        context.validate_against(other_scope)


def test_evaluation_comparability_is_independent_from_context_transfer():
    records = build_records()
    evaluation = next(
        record for record in records if isinstance(record, EvaluationFingerprint)
    )
    context = next(
        record for record in records if isinstance(record, TaskContextBinding)
    )
    changed_evaluation = EvaluationFingerprint.mint(
        **{
            key: value
            for key, value in evaluation.to_dict().items()
            if key not in {"evaluation_fingerprint_id", "split_version"}
        },
        split_version="public-v2",
    )
    changed_context = TaskContextBinding.mint(
        **{
            key: value
            for key, value in context.to_dict().items()
            if key not in {"task_context_binding_id", "budget_hardware_envelope"}
        },
        budget_hardware_envelope={"accelerator": "A100", "hours": 4},
    )

    assert not evaluation.comparable_with(changed_evaluation)
    assert (
        context.compatibility_with(changed_context) is TransferCompatibility.ANALOGICAL
    )


def test_gap_or_duplicate_attempt_revisions_fail_loud():
    attempt = next(
        record for record in build_records() if isinstance(record, TransferAttempt)
    )
    episode = next(
        record for record in build_records() if isinstance(record, TransferEpisode)
    )
    second = replace(attempt, execution_revision=2)
    with pytest.raises(ContractValidationError):
        TransferEpisode.mint(
            **{
                key: value
                for key, value in episode.to_dict().items()
                if key not in {"episode_id", "attempts", "terminal_attempt_revision"}
            },
            attempts=(attempt, replace(second, execution_revision=3)),
            terminal_attempt_revision=3,
        )


def test_attempt_cannot_present_invalid_or_unbound_evaluation_as_comparable():
    attempt = next(
        record for record in build_records() if isinstance(record, TransferAttempt)
    )

    with pytest.raises(ContractValidationError):
        replace(attempt, evaluation_status=EpisodeEvaluationStatus.INVALID)
    with pytest.raises(ContractValidationError):
        replace(attempt, evaluation_fingerprints=())


def test_attempt_effect_is_bound_to_the_score_measurement():
    attempt = next(
        record for record in build_records() if isinstance(record, TransferAttempt)
    )

    with pytest.raises(ContractValidationError):
        replace(attempt, measurements={"quality": 0.9})
    with pytest.raises(ContractValidationError):
        replace(
            attempt,
            source_parent_effect=replace(
                attempt.source_parent_effect,
                metric_name="other_metric",
            ),
        )
    with pytest.raises(ContractValidationError):
        replace(
            attempt,
            evaluation_status=EpisodeEvaluationStatus.NOT_RUN,
            evaluation_fingerprints=(),
            score_of_record_fingerprint_id=None,
            comparison_status=ComparisonStatus.NOT_COMPARABLE,
            source_parent_effect=None,
        )


def test_prior_idea_may_preserve_an_empty_source_assumption_set():
    prior_idea = next(
        record for record in build_records() if isinstance(record, PriorIdea)
    )

    empty_assumptions = PriorIdea.mint(
        **{
            key: value
            for key, value in prior_idea.to_dict().items()
            if key not in {"prior_idea_id", "assumptions"}
        },
        assumptions=(),
    )

    assert empty_assumptions.assumptions == ()


def test_catalog_state_preserves_revocation_precedence_over_supersession():
    state = next(
        record for record in build_records() if isinstance(record, CatalogEntryState)
    )
    successor_id = fixture_id("successor-payload")
    taint_id = fixture_id("taint-source")

    revoked = CatalogEntryState.mint(
        subject_payload_id=state.subject_payload_id,
        catalog_generation=2,
        predecessor_state_id=state.catalog_entry_state_id,
        configuration_fingerprint=state.configuration_fingerprint,
        admission_state=AdmissionState.REVOKED,
        superseded_by_payload_ids=(successor_id,),
        assertion_ids=state.assertion_ids,
        revocation_ids=(),
        taint_source_ids=(taint_id,),
    )
    assert revoked.superseded_by_payload_ids == (successor_id,)

    with pytest.raises(ContractValidationError):
        replace(revoked, admission_state=AdmissionState.SUPERSEDED)
    with pytest.raises(ContractValidationError):
        replace(state, superseded_by_payload_ids=(successor_id,))


def test_prose_tuple_fields_reject_blank_elements():
    records = build_records()
    cases = (
        (
            next(
                record for record in records if isinstance(record, ExpertScopeContract)
            ),
            "explicit_non_goals",
        ),
        (
            next(record for record in records if isinstance(record, TransferAttempt)),
            "feedback",
        ),
        (
            next(record for record in records if isinstance(record, PriorIdea)),
            "assumptions",
        ),
        (
            next(record for record in records if isinstance(record, KnowledgeClaim)),
            "explicit_exclusions",
        ),
        (
            next(
                record for record in records if isinstance(record, ExpertModuleContract)
            ),
            "inputs",
        ),
        (
            next(
                record for record in records if isinstance(record, ExpertRepositoryMap)
            ),
            "architecture_invariants",
        ),
    )

    for record, field_name in cases:
        with pytest.raises(ContractValidationError):
            replace(record, **{field_name: (" ",)})


def test_repository_aliasing_and_invalid_lineage_are_rejected():
    with pytest.raises(IdentityConflictError):
        ScopeRepositorySettings(
            scope_id="ml_ai",
            expert_repository="Leeroo-AI/same",
            knowledge_repository="Leeroo-AI/same",
            security_repository="Leeroo-AI/security",
        )
    with pytest.raises(ContractValidationError):
        LineageEdge(
            source_ids=("one", "two"),
            target_ids=("three", "four"),
            relation=LineageRelation.MERGE,
        )


def test_unknown_family_and_adapter_bindings_fail_against_scope_contract():
    scope = next(
        record for record in build_records() if isinstance(record, ExpertScopeContract)
    )
    with pytest.raises(IncompatibleArtifactError):
        scope.validate_binding(
            CrossRunTaskBindingSettings(
                scope_id="ml_ai",
                task_family_id="unknown_family",
                task_adapter_id="unknown_adapter",
            )
        )
    with pytest.raises(IncompatibleArtifactError):
        scope.validate_binding(
            CrossRunTaskBindingSettings(
                scope_id="ml_ai",
                task_family_id="language_model_post_training",
                task_adapter_id="relbench",
            )
        )


def test_prior_packet_rejects_ids_only_placeholder_records():
    prior_snapshot = next(
        record
        for record in build_records()
        if isinstance(record, PriorKnowledgeSnapshot)
    )
    with pytest.raises(ContractValidationError):
        PriorKnowledgeSnapshot.mint(
            **{
                key: value
                for key, value in prior_snapshot.to_dict().items()
                if key
                not in {
                    "prior_knowledge_snapshot_id",
                    "selected_records",
                    "records_digest",
                }
            },
            selected_records=({"record_id": prior_snapshot.selected_record_ids[0]},),
            records_digest=tree_or_blob_digest(
                canonical_json_bytes(
                    ({"record_id": prior_snapshot.selected_record_ids[0]},)
                )
            ),
        )


def test_prior_packet_requires_complete_selection_metadata_for_every_record():
    prior_snapshot = next(
        record
        for record in build_records()
        if isinstance(record, PriorKnowledgeSnapshot)
    )
    fields = {
        key: value
        for key, value in prior_snapshot.to_dict().items()
        if key not in {"prior_knowledge_snapshot_id", "selection_metadata"}
    }

    with pytest.raises(
        ContractValidationError,
        match="keyed by every selected record exactly",
    ):
        PriorKnowledgeSnapshot.mint(**fields, selection_metadata={})


@pytest.mark.parametrize("evidence_mode", ["overlap", "absent"])
def test_knowledge_claim_requires_unambiguous_evidence(evidence_mode):
    claim = next(
        record for record in build_records() if isinstance(record, KnowledgeClaim)
    )
    payload = {
        key: value
        for key, value in claim.to_dict().items()
        if key
        not in {
            "revision_id",
            "supporting_episode_ids",
            "contradicting_episode_ids",
        }
    }
    support = claim.supporting_episode_ids if evidence_mode == "overlap" else ()
    contradiction = claim.supporting_episode_ids if evidence_mode == "overlap" else ()

    with pytest.raises(ContractValidationError):
        KnowledgeClaim.mint(
            **payload,
            supporting_episode_ids=support,
            contradicting_episode_ids=contradiction,
        )


def test_knowledge_snapshot_rejects_incomplete_proof_closure():
    snapshot = next(
        record
        for record in build_records()
        if isinstance(record, KnowledgeSnapshotManifest)
    )
    payload = {
        key: value
        for key, value in snapshot.to_dict().items()
        if key not in {"snapshot_id", "proof_dependency_closure_ids"}
    }

    with pytest.raises(MissingReferenceError):
        KnowledgeSnapshotManifest.mint(
            **payload,
            proof_dependency_closure_ids=snapshot.proof_dependency_closure_ids[1:],
        )
