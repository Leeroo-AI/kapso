from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    AdmissionState,
    ArtifactCompleteness,
    ArtifactEnvironment,
    BootstrapPin,
    CandidateChangeKind,
    CaptureManifest,
    CatalogEntryState,
    ClaimState,
    ComparisonStatus,
    CompletionState,
    ContextDimensionSchema,
    ContextValueType,
    ContractValidationError,
    CrossRunTaskBindingSettings,
    EmbeddingSidecar,
    EpisodeEvaluationStatus,
    EvaluationFingerprint,
    ExecutionStatus,
    ExpertBaseReleaseManifest,
    ExpertCandidateManifest,
    ExpertCapabilityNode,
    ExpertDependencyEdge,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
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
    RunBundle,
    ScopeRepositorySettings,
    TaskAdapterBinding,
    TaskAdapterManifest,
    TaskContextBinding,
    TaskFamilyDefinition,
    TransferAttempt,
    TransferCompatibility,
    TransferEpisode,
)
from kapso.cross_run.settings import CrossRunSettings

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def fixture_id(name):
    return content_id("fixture", {"name": name})


def digest(name):
    return tree_or_blob_digest(name.encode("utf-8"))


def assertion(subject_id, name):
    return ReviewAssertion.mint(
        subject_id=subject_id,
        reviewer_id=f"reviewer-{name}",
        reviewer_role="independent_reviewer",
        rubric_version="rubric-v1",
        judgment="approve",
        rationale="The exact evidence satisfies the configured rubric.",
        exact_evidence_refs=(fixture_id(f"{name}-evidence"),),
        created_at="2026-07-20T12:00:00Z",
        supersedes_assertion_id=None,
        reviewer_attestation={"issuer": "test-trust-root", "signature": name},
    )


def build_records():
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
    bootstrap_assertion = assertion(fixture_id("bootstrap-candidate"), "bootstrap")
    module = ExpertModuleContract.mint(
        module_id="shared.reproducible_execution",
        version="v1",
        purpose="Produce provenance-bound resumable execution.",
        inputs=("run contract",),
        outputs=("resumable artifact",),
        preconditions=("writable workspace",),
        incompatibilities=("untracked mutable input",),
        resource_bounds={"maximum_workers": 1},
        dependency_license_manifest={"license": "MIT"},
        supporting_episode_ids=(),
        known_failure_episode_ids=(),
        test_refs=("tests/test_resume.py",),
        replay_refs=("tests/replay_resume.py",),
    )
    capability_node = ExpertCapabilityNode(
        capability_id="shared.reproducible_execution",
        module_contract_ref=module.module_contract_id,
        owned_paths=("src/reproducible_execution",),
        task_family_bindings=(
            "language_model_post_training",
            "relational_tabular_prediction",
        ),
    )
    repository_map = ExpertRepositoryMap.mint(
        scope_contract_id=scope.scope_contract_id,
        capability_nodes=(capability_node,),
        dependency_edges=(),
        task_adapter_boundary={"mode": "read_only"},
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
        source_archive_ref="release-assets/expert.tar.zst",
        dependency_closure_ids=tuple(
            sorted((repository_map.repository_map_id, bootstrap_assertion.assertion_id))
        ),
        checksums={"release-assets/expert.tar.zst": digest("expert-archive")},
        test_matrix_results={"fresh_task": "passed"},
        approval_assertion_ids=(bootstrap_assertion.assertion_id,),
        contamination_scanner_version="scanner-v1",
        dependency_lock_hash=digest("lock"),
        compatibility_envelope={"python": ">=3.10"},
        publisher_attestation={"issuer": "test-publisher", "signature": "expert"},
    )
    artifact_environment = ArtifactEnvironment.mint(
        kapso_commit="0" * 40,
        expert_base_release_id=expert_release.release_id,
        task_adapter_hash=digest("adapter"),
        dependency_lock_hash=digest("lock"),
    )
    task_adapter = TaskAdapterManifest.mint(
        task_adapter_id="posttrain",
        scope_contract_id=scope.scope_contract_id,
        task_family_id="language_model_post_training",
        publisher_attestation={"issuer": "test-publisher", "signature": "adapter"},
        task_evaluator_binding={"evaluator": "public"},
        context_dimension_binding={
            "dataset_family": "instruction",
            "runtime_family": "pytorch",
        },
        source_tree_ref="task-adapter.tar.zst",
        tree_hash=digest("adapter-tree"),
        dependency_runtime_contract={"python": ">=3.10"},
        sanitation_report_id=fixture_id("adapter-sanitation"),
        validation_refs=("validation/adapter-smoke",),
    )
    initial_snapshot_id = fixture_id("empty-snapshot")
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
        security_denylist_generation=1,
        expected_source_composition_hash=digest("workspace"),
        publisher_attestation={"issuer": "test-publisher", "signature": "launch"},
    )
    capture = CaptureManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        scope_id="ml_ai",
        run_id="run-1",
        campaign_id="campaign-1",
        capture_generation=1,
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
        capture_generation=1,
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
        checkpoint_ref="checkpoint.json",
        execution_event_journal_ref="events.jsonl",
        idea_archive_ref="idea_archive.json",
        experiment_history_ref="experiment_history.json",
        branch_snapshot_refs=("branches/node-1.tar.zst",),
        run_log_refs=("logs/run.log",),
        checksums={
            "branches/node-1.tar.zst": digest("branch"),
            "checkpoint.json": digest("checkpoint"),
            "events.jsonl": digest("events"),
            "experiment_history.json": digest("history"),
            "idea_archive.json": digest("ideas"),
            "logs/run.log": digest("log"),
        },
    )
    attempt = TransferAttempt(
        execution_revision=1,
        captured_at="2026-07-20T12:55:00Z",
        evaluation_fingerprint=evaluation,
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        comparison_status=ComparisonStatus.COMPARABLE,
        measurements={"quality": 0.8},
        source_parent_effect={"delta": 0.1, "uncertainty": 0.02},
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
        supersedes_episode_id=None,
        task_context_binding=context,
        artifact_environment=artifact_environment,
        proposal="Validate representation parity before training.",
        intervention_ref="artifacts/node-1.patch",
        intervention_structure=InterventionStructure.ISOLATED_BY_ABLATION,
        parent_episode_ref=None,
        attempts=(attempt,),
        terminal_attempt_revision=1,
        safe_observation_refs=("observation/parity",),
        sanitation_report_id=fixture_id("episode-sanitation"),
        derivation_refs=(run_bundle.bundle_id,),
    )
    prior_idea = PriorIdea.mint(
        source_bundle_id=run_bundle.bundle_id,
        source={
            "campaign_id": "campaign-1",
            "batch_id": "batch-1",
            "idea_id": "idea-2",
        },
        proposal="Explore a different packing policy.",
        descriptor="Packing-policy exploration",
        assumptions=("Examples have varied lengths.",),
        source_status=PriorIdeaStatus.DEFERRED,
        source_rationale="The run budget ended.",
        evidence_refs=(episode.episode_id,),
        task_context_binding=context,
        sanitation_report_id=fixture_id("prior-sanitation"),
    )
    claim_assertion = assertion(episode.episode_id, "claim")
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
        state=ClaimState.PROVISIONAL,
        review_assertion_ids=(claim_assertion.assertion_id,),
        supersedes_claim_ids=(),
    )
    catalog_state = CatalogEntryState.mint(
        subject_payload_id=claim.revision_id,
        catalog_generation=1,
        configuration_fingerprint=digest("catalog-config"),
        admission_state=AdmissionState.ADMITTED,
        assertion_ids=(claim_assertion.assertion_id,),
        revocation_ids=(),
        taint_source_ids=(),
        publisher_attestation={"issuer": "test-publisher", "signature": "catalog"},
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
        prompt_budget_policy={"maximum_records": 24},
        records_digest=tree_or_blob_digest(canonical_json_bytes(selected_records)),
    )
    candidate = ExpertCandidateManifest.mint(
        scope_contract_id=scope.scope_contract_id,
        change_kind=CandidateChangeKind.CAPABILITY,
        parent_release_id=expert_release.release_id,
        parent_tree_hash=digest("parent-tree"),
        trigger="Repeated representation drift across independent contexts.",
        trigger_evidence_ids=(episode.episode_id,),
        patch_ref="candidates/template-parity.patch",
        candidate_tree_hash=digest("candidate-tree"),
        configuration_fingerprint=digest("expert-validation-config"),
        module_contract_refs=(module.module_contract_id,),
        proposed_repository_map_ref=repository_map.repository_map_id,
        proposer_operation={"cli": "codex", "model": "configured"},
        source_dependency_ids=(claim.revision_id,),
        ancestor_candidate_ids=(),
        capability_lineage=(),
        validation_attempt_refs=("validation/candidate-1",),
        sanitation_report_id=fixture_id("candidate-sanitation"),
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
        claim_assertion,
        claim,
        catalog_state,
        sidecar,
        snapshot,
        prior_snapshot,
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


def test_nested_contract_has_stable_golden_bytes_and_identity():
    attempt = next(
        record for record in build_records() if isinstance(record, TransferAttempt)
    )

    assert attempt.to_json_bytes() == (
        b'{"captured_at":"2026-07-20T12:55:00Z","comparison_status":"comparable",'
        b'"confounders":[],"evaluation_fingerprint":{"aggregation_protocol":'
        b'"arithmetic-mean","benchmark_id":"posttrain","dataset_version":"v1",'
        b'"evaluation_fingerprint_id":"evaluation-fingerprint:sha256:'
        b'746d9917d4b68f1398a01b3c41cd97f367b9e69c0a170654de99375abbfec5de",'
        b'"evaluator_fingerprint":"sha256:'
        b'27f6343f980c4c0ab821d9c72d211d0eb76b0853825cefd80476a05e43cab27e",'
        b'"fidelity":"full","fraction":1.0,"judge_version":null,"metric_name":'
        b'"quality","objective_direction":"maximize","seed_or_replicate_ids":'
        b'["seed-1"],"split_version":"public-v1"},"evaluation_status":"valid",'
        b'"execution_revision":1,"execution_status":"completed","feedback":'
        b'["Improved validation quality."],"measurements":{"quality":0.8},'
        b'"source_parent_effect":{"delta":0.1,"uncertainty":0.02},'
        b'"technical_difficulties":[]}'
    )
    assert content_id("transfer-attempt-fixture", attempt) == (
        "transfer-attempt-fixture:sha256:"
        "7bf501999a228f2cf9e0f971b9425a8f1793cbeb3836132749eb77288d4485a1"
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
        replace(attempt, evaluation_fingerprint=None)


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
