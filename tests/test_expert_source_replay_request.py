from __future__ import annotations

import itertools
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.catalog.lineage import (
    RunBundleLineageError,
    VerifiedRunBundleLineage,
)
from kapso.cross_run.catalog.projector import RunBundleProjector
from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.contracts import (
    CompletionState,
    ContractValidationError,
    ExpertCandidateValidationState,
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertSourceReplayExecutionLegKind,
    ExpertSourceReplayContextMaterializationReceipt,
    ExpertSourceReplayStartingArtifact,
    ExpertValidationStage,
    MissingReferenceError,
    ObjectiveDirection,
    SourceFileDescriptor,
    TaskAdapterManifest,
)
from kapso.cross_run.expert.replay_context import (
    VerifiedSourceReplayContext,
    VerifiedSourceReplayStartingArtifact,
)
from kapso.cross_run.expert.replay_request import (
    ExpertSourceReplayPreflightCoordinator,
    ExpertSourceReplayRequestError,
    VerifiedExpertSourceReplayParent,
    _source_replay_compute_bindings,
)
from kapso.cross_run.expert.triggers import (
    ExpertTriggerEvaluator,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertEvaluatorRunBuilder,
    ExpertValidationPredecessor,
    ExpertValidationReducer,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStore
from cross_run_capture_fixtures import make_capture_fixture
from test_cross_run_contracts import build_records, verified_test_task_adapter
from test_expert_source_replay import (
    _AdapterProvider,
    _CandidateReader,
    _CurrentReleaseProvider,
    _candidate_id,
    _changed_module,
    _observation,
    _stored_candidate,
    _validation_policy,
)
from test_expert_triggers import trigger_packet, trigger_settings
from test_expert_validation import _AttestationVerifier, _ValidationStateProvider

REQUEST_FIXTURE_SEQUENCE = itertools.count()


class _BundleProvider:
    def __init__(self, stored_bundles, projection):
        self.lineage = VerifiedRunBundleLineage(
            bundles=stored_bundles,
            tip_projection=projection,
        )
        self.bundle_ids = []
        self.timeouts_seen = []

    def resolve_exact_bounded(
        self,
        bundle_id,
        *,
        maximum_entries,
        maximum_bytes,
        timeout_seconds,
        retained_bundles,
    ):
        self.bundle_ids.append(bundle_id)
        self.timeouts_seen.append(timeout_seconds)
        assert bundle_id == self.lineage.bundle_ids[-1]
        assert maximum_entries > 0
        assert maximum_bytes > 0
        assert all(
            retained_bundle.manifest.bundle_id == retained_id
            for retained_id, retained_bundle in retained_bundles.items()
        )
        return self.lineage


class _ContextProvider:
    def __init__(self, settings, context, *, input_fingerprint=None):
        self.settings = settings
        self.input_fingerprint = input_fingerprint
        self.verified = None
        self.context_ids = []
        self.limits_seen = []

    def materialize_exact(self, context, expected_artifact_content_ids, limits):
        self.context_ids.append(context.task_context_binding_id)
        self.limits_seen.append(limits)
        artifacts = []
        for reference in context.starting_artifact_refs:
            payload = f"starting artifact:{reference}".encode("utf-8")
            descriptor = SourceFileDescriptor(
                relative_path="artifact.bin",
                digest=tree_or_blob_digest(payload),
                mode="100644",
                size=len(payload),
            )
            artifact = ExpertSourceReplayStartingArtifact.mint(
                starting_artifact_ref=reference,
                mount_path=f"inputs/{reference.rsplit('/', 1)[-1]}",
                materialized_tree_hash=source_tree_digest(
                    {
                        descriptor.relative_path: (
                            descriptor.digest,
                            descriptor.mode,
                            descriptor.size,
                        )
                    }
                ),
                source_files=(descriptor,),
            )
            assert (
                expected_artifact_content_ids[reference]
                == artifact.starting_artifact_content_id
            )
            artifacts.append(
                VerifiedSourceReplayStartingArtifact(
                    artifact=artifact,
                    source_contents={"artifact.bin": payload},
                )
            )
        ordered_artifacts = tuple(
            sorted(
                artifacts,
                key=lambda item: item.artifact.starting_artifact_content_id,
            )
        )
        receipt = ExpertSourceReplayContextMaterializationReceipt.mint(
            task_context_binding_id=context.task_context_binding_id,
            input_contract_fingerprint=(
                self.input_fingerprint or context.input_contract_fingerprint
            ),
            target_contract_fingerprint=context.target_contract_fingerprint,
            starting_artifacts=tuple(item.artifact for item in ordered_artifacts),
            materializer_id=(
                self.settings.policy.source_replay_context_materializer_id
            ),
            materializer_version=(
                self.settings.policy.source_replay_context_materializer_version
            ),
        )
        self.verified = VerifiedSourceReplayContext(
            receipt=receipt,
            starting_artifacts=ordered_artifacts,
        )
        return self.verified


class _ParentProvider:
    def __init__(self):
        self.materializations = []

    def materialize_exact(self, release, parent_tree_receipt, limits):
        self.materializations.append((release.release_id, limits))
        return VerifiedExpertSourceReplayParent(
            release_manifest=release,
            parent_tree_receipt=parent_tree_receipt,
            source_contents={"src/expert.py": b"verified parent source"},
        )


def _request_fixture(
    tmp_path,
    *,
    rotate_active_adapter=False,
    bundle_generations=1,
    evaluator_fingerprint=None,
    evaluation_evidence=None,
    contract_records=None,
    source_adapter=None,
    validation_settings=None,
    candidate_first_execution=False,
):
    if (contract_records is None) != (source_adapter is None):
        raise ValueError(
            "source replay fixture records and adapter must be supplied together"
        )
    fixture_root = tmp_path / f"request-{next(REQUEST_FIXTURE_SEQUENCE)}"
    fixture_root.mkdir()
    capture_fixture = make_capture_fixture(
        fixture_root,
        contract_records=contract_records,
        evaluator_fingerprint=evaluator_fingerprint,
        evaluation_evidence=evaluation_evidence,
    )
    capture_pipeline = RunCapturePipeline(
        RunCaptureContext(capture_fixture.request),
        capture_fixture.settings,
    )
    stored_bundles = []
    for capture_generation in range(bundle_generations):
        if capture_generation > 0:
            capture_fixture.strategy.previous_errors.append(
                f"source replay generation {capture_generation} observation"
            )
            capture_fixture.save_checkpoint("running")
        stored_bundle = capture_pipeline.capture_if_due(
            CompletionState.STOPPED,
            force=True,
        )
        assert stored_bundle is not None
        stored_bundles.append(stored_bundle)
    bundle_projector = RunBundleProjector(
        capture_fixture.settings.capture.score_comparison_tolerance
    )
    projection = None
    projections = []
    for stored_bundle in stored_bundles:
        projection = bundle_projector.project(stored_bundle, previous=projection)
        projections.append(projection)
    assert projection is not None
    episode = projection.episodes[0]
    knowledge_facts_by_id = {}
    for historical_projection in projections[:-1]:
        historical_facts = (
            historical_projection.source_bundle,
            historical_projection.sanitation_report,
            *historical_projection.derivation_objects,
            *historical_projection.episodes,
        )
        knowledge_facts_by_id.update(
            {getattr(fact, fact.IDENTITY_FIELD): fact for fact in historical_facts}
        )
    knowledge_facts_by_id.update(
        {event.event_id: event for event in projection.derivation_objects}
    )
    knowledge_extra_facts = tuple(
        knowledge_facts_by_id[fact_id] for fact_id in sorted(knowledge_facts_by_id)
    )
    knowledge_projection_derivation_ids = tuple(
        event.event_id for event in projection.derivation_objects
    )
    trigger_policy = trigger_settings()
    packet_without_observation = trigger_packet(
        settings=trigger_policy,
        episodes=(episode,),
        knowledge_source_bundle=projection.source_bundle,
        knowledge_sanitation_report=projection.sanitation_report,
        knowledge_extra_facts=knowledge_extra_facts,
        knowledge_projection_derivation_ids=knowledge_projection_derivation_ids,
    )
    module = packet_without_observation.module_contracts[0]
    observation = _observation(
        settings=trigger_policy,
        kind=ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        module_id=module.module_id,
        exact_evidence_ids=(
            packet_without_observation.repository_map.repository_map_id,
        ),
        affected_paths=("src/reproducible_execution/__init__.py",),
    )
    packet = trigger_packet(
        settings=trigger_policy,
        episodes=(episode,),
        observations=(observation,),
        knowledge_source_bundle=projection.source_bundle,
        knowledge_sanitation_report=projection.sanitation_report,
        knowledge_extra_facts=knowledge_extra_facts,
        knowledge_projection_derivation_ids=knowledge_projection_derivation_ids,
    )
    decision = ExpertTriggerEvaluator(trigger_policy).evaluate(packet)
    changed_module = _changed_module(
        packet.module_contracts[0],
        supporting_episode_ids=(episode.episode_id,),
    )
    stored = _stored_candidate(
        _candidate_id("request"),
        packet,
        decision,
        (changed_module,),
    )
    settings = (
        _validation_policy() if validation_settings is None else validation_settings
    )
    if candidate_first_execution:
        base_protocol_version = (
            settings.policy.source_replay_paired_execution_protocol_version
        )
        protocol_ordinal = 0
        while (
            _source_replay_compute_bindings(settings, (episode.episode_id,))[
                episode.episode_id
            ].leg_order[0]
            is not ExpertSourceReplayExecutionLegKind.CANDIDATE
        ):
            settings = replace(
                settings,
                policy=replace(
                    settings.policy,
                    source_replay_paired_execution_protocol_version=(
                        f"{base_protocol_version}.candidate-first-{protocol_ordinal}"
                    ),
                ),
            )
            protocol_ordinal += 1
    adapter_provider = _AdapterProvider(
        packet,
        source_adapter=source_adapter,
        rotate_active=rotate_active_adapter,
    )
    current_release_provider = _CurrentReleaseProvider(packet.parent_release_id)
    eligibility = ExpertCandidateEligibilityEvaluator(
        settings,
        _CandidateReader(stored),
        adapter_provider,
        current_release_provider,
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    assert eligibility.decision.eligible is True
    reducer = ExpertValidationReducer(
        settings,
        _CandidateReader(stored),
        _AttestationVerifier(),
        adapter_provider,
        current_release_provider,
        _ValidationStateProvider(),
    )
    validation_state_root = fixture_root / "validation-state"
    validation_state_root.mkdir(mode=0o700)
    validation_store = ExpertValidationStore(
        (validation_state_root / "validation").resolve(),
        validation_state_root.resolve(),
        settings,
        reducer,
    )
    snapshot = validation_store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    assert snapshot.latest_attempt is not None
    while snapshot.state.next_stage is not ExpertValidationStage.SOURCE_RUN_REPLAY:
        stage = snapshot.state.next_stage
        assert stage is not None
        result = ExpertEvaluatorRunBuilder(settings).build(
            attempt=snapshot.latest_attempt,
            stage=stage,
            exact_additional_input_ids=(),
            output_payloads={"result.json": b'{"completed":true}'},
            measurements={},
            costs={},
            duration_seconds=1.0,
            outcome=ExpertEvaluatorOutcome.PASSED,
            signature="test-signature",
        )
        snapshot = validation_store.publish_evaluator_result(
            candidate_id=snapshot.state.candidate_id,
            expected_transition_id=snapshot.transition.transition_id,
            result=result,
        ).snapshot
    attempt = snapshot.latest_attempt
    state = snapshot.state
    bundle_provider = _BundleProvider(tuple(stored_bundles), projection)
    context_provider = _ContextProvider(
        settings,
        episode.task_context_binding,
    )
    parent_provider = _ParentProvider()
    coordinator = ExpertSourceReplayPreflightCoordinator(
        settings,
        _CandidateReader(stored),
        validation_store,
        current_release_provider,
        parent_provider,
        bundle_provider,
        adapter_provider,
        context_provider,
        time.monotonic,
    )
    return SimpleNamespace(
        coordinator=coordinator,
        validation_store=validation_store,
        stored=stored,
        settings=settings,
        eligibility=eligibility,
        attempt=attempt,
        state=state,
        episode=episode,
        adapter_provider=adapter_provider,
        bundle_provider=bundle_provider,
        context_provider=context_provider,
        parent_provider=parent_provider,
        current_release_provider=current_release_provider,
    )


def _prepared(fixture):
    result = fixture.coordinator.build(fixture.attempt)
    assert result.invalidated_state is None
    assert result.prepared_request is not None
    return result.prepared_request


@pytest.mark.parametrize("episode_count", (2, 3, 32))
def test_compute_schedule_is_balanced_and_input_order_invariant(episode_count):
    episode_ids = tuple(
        content_id("transfer-episode", {"position": position})
        for position in range(episode_count)
    )
    settings = _validation_policy()

    schedule = _source_replay_compute_bindings(settings, episode_ids)
    reversed_schedule = _source_replay_compute_bindings(
        settings,
        tuple(reversed(episode_ids)),
    )
    ordered_bindings = tuple(schedule[episode_id] for episode_id in sorted(episode_ids))
    control_first_count = sum(
        binding.leg_order[0] is ExpertSourceReplayExecutionLegKind.CONTROL_PARENT
        for binding in ordered_bindings
    )

    assert schedule == reversed_schedule
    assert abs(control_first_count - (episode_count - control_first_count)) <= 1
    assert all(
        left.leg_order != right.leg_order
        for left, right in itertools.pairwise(ordered_bindings)
    )


def test_request_materializes_exact_candidate_bundle_episode_adapter_and_context(
    tmp_path,
):
    fixture = _request_fixture(tmp_path)

    prepared = _prepared(fixture)

    assert prepared.candidate.manifest == fixture.stored.closure.manifest
    assert prepared.authorization_state == fixture.state
    assert len(prepared.cases) == 1
    request_case = prepared.request.cases[0]
    assert request_case.episode_id == fixture.episode.episode_id
    assert request_case.source_execution_revision == (
        fixture.episode.terminal_attempt_revision
    )
    assert request_case.task_adapter_manifest_id == (
        fixture.episode.artifact_environment.task_adapter_manifest_id
    )
    assert request_case.context_materialization_receipt_id == (
        fixture.context_provider.verified.receipt.context_materialization_receipt_id
    )
    assert fixture.bundle_provider.bundle_ids == [fixture.episode.source_bundle_id]
    assert fixture.context_provider.context_ids == [
        fixture.episode.task_context_binding.task_context_binding_id
    ]
    assert fixture.context_provider.limits_seen[0].maximum_bytes < (
        fixture.settings.policy.source_replay_materialization_byte_limit
    )
    assert request_case.control_leg.expert_artifact_id == (
        prepared.parent.release_manifest.release_id
    )
    assert request_case.candidate_leg.expert_artifact_id == (
        prepared.candidate.manifest.candidate_id
    )
    compute_binding = request_case.compute_binding
    source_replay_evaluator = next(
        evaluator
        for evaluator in fixture.settings.policy.evaluators
        if evaluator.stage is ExpertValidationStage.SOURCE_RUN_REPLAY
    )
    assert compute_binding.leg_wall_time_limit_seconds == (
        source_replay_evaluator.timeout_seconds
    )
    assert compute_binding.cpu_millicore_limit == (
        fixture.settings.policy.source_replay_cpu_millicore_limit
    )
    assert compute_binding.output_entry_limit == (
        fixture.settings.policy.artifact_entry_limit
    )
    assert set(compute_binding.leg_order) == set(ExpertSourceReplayExecutionLegKind)
    assert compute_binding.compute_binding_id in request_case.exact_dependency_ids
    assert set(prepared.request.exact_dependency_ids) == {
        prepared.request.validation_attempt_id,
        prepared.request.authorization_state_id,
        prepared.request.source_replay_selection_id,
        prepared.request.candidate_id,
        prepared.request.candidate_commit_record_id,
        prepared.request.scope_contract_id,
        prepared.request.parent_release_id,
        prepared.request.parent_tree_receipt_id,
        prepared.request.parent_source_extraction_receipt_id,
        prepared.request.candidate_source_tree_manifest_id,
        prepared.request.validation_policy_id,
        *prepared.request.attempt_dependency_ids,
        request_case.execution_case_id,
        *request_case.exact_dependency_ids,
    }
    with pytest.raises(MissingReferenceError, match="not exact"):
        replace(
            prepared.request,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        *prepared.request.exact_dependency_ids,
                        content_id("unrelated-proof", {"label": "extra"}),
                    }
                )
            ),
        )


@pytest.mark.parametrize(
    ("binding_update", "message"),
    (
        ({"metric_name": "other_metric"}, "metric comparison authority"),
        ({"comparison_dimension_id": "unknown_dimension"}, "central promotion"),
        (
            {
                "objective_direction": ObjectiveDirection.MINIMIZE,
                "comparison_dimension_id": "cost",
            },
            "metric comparison authority",
        ),
    ),
)
def test_request_rejects_invalid_metric_comparison_binding(
    tmp_path,
    binding_update,
    message,
):
    records = build_records()
    manifest = next(
        record for record in records if isinstance(record, TaskAdapterManifest)
    )
    binding = manifest.task_evaluator.metric_comparison_bindings[0]
    changed_evaluator = replace(
        manifest.task_evaluator,
        metric_comparison_bindings=(replace(binding, **binding_update),),
    )
    changed_records = build_records(task_evaluator=changed_evaluator)
    changed_manifest = next(
        record for record in changed_records if isinstance(record, TaskAdapterManifest)
    )
    fixture = _request_fixture(
        tmp_path,
        contract_records=changed_records,
        source_adapter=verified_test_task_adapter(changed_manifest),
    )

    with pytest.raises(ExpertSourceReplayRequestError, match=message):
        _prepared(fixture)


def test_request_rejects_central_metric_direction_conflict(tmp_path):
    settings = _validation_policy()
    changed_dimensions = tuple(
        (
            replace(dimension, direction=ObjectiveDirection.MINIMIZE)
            if dimension.dimension_id == "quality"
            else dimension
        )
        for dimension in settings.policy.promotion.pareto_dimensions
    )
    changed_settings = replace(
        settings,
        policy=replace(
            settings.policy,
            promotion=replace(
                settings.policy.promotion,
                pareto_dimensions=changed_dimensions,
            ),
        ),
    )
    fixture = _request_fixture(
        tmp_path,
        validation_settings=changed_settings,
    )

    with pytest.raises(ExpertSourceReplayRequestError, match="central promotion"):
        _prepared(fixture)


def test_request_rejects_a_source_fingerprint_from_another_evaluator(tmp_path):
    fixture = _request_fixture(
        tmp_path,
        evaluator_fingerprint=tree_or_blob_digest(b"another-evaluator"),
    )

    with pytest.raises(ExpertSourceReplayRequestError, match="exact task evaluator"):
        fixture.coordinator.build(fixture.attempt)


def test_aggregate_limit_counts_candidate_parent_adapter_bundle_and_context(tmp_path):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    adapters = tuple(
        {
            item.task_adapter.verification_receipt.verification_receipt_id: (
                item.task_adapter
            )
            for item in prepared.cases
        }.values()
    )
    lineages = tuple(
        {
            item.bundle_lineage.bundle_ids: item.bundle_lineage
            for item in prepared.cases
        }.values()
    )
    contexts = tuple(
        {
            item.task_context.receipt.context_materialization_receipt_id: (
                item.task_context
            )
            for item in prepared.cases
        }.values()
    )
    entry_count, byte_count = fixture.coordinator._materialization_usage(
        candidate=prepared.candidate,
        parent=prepared.parent,
        adapters=adapters,
        lineages=lineages,
        contexts=contexts,
    )
    limited_settings = replace(
        fixture.settings,
        policy=replace(
            fixture.settings.policy,
            source_replay_materialization_entry_limit=entry_count,
            source_replay_materialization_byte_limit=byte_count - 1,
        ),
    )
    limited_coordinator = ExpertSourceReplayPreflightCoordinator(
        limited_settings,
        fixture.coordinator.candidate_store,
        fixture.coordinator.validation_authority,
        fixture.coordinator.current_release_provider,
        fixture.coordinator.parent_provider,
        fixture.coordinator.bundle_provider,
        fixture.coordinator.task_adapter_provider,
        fixture.coordinator.task_context_provider,
        time.monotonic,
    )

    with pytest.raises(ExpertSourceReplayRequestError, match="aggregate"):
        limited_coordinator._check_materialization_totals(
            candidate=prepared.candidate,
            parent=prepared.parent,
            adapters=adapters,
            lineages=lineages,
            contexts=contexts,
        )
    with pytest.raises(
        ExpertSourceReplayRequestError, match="differs from its request"
    ):
        replace(prepared, settings=limited_settings)
    looser_settings = replace(
        fixture.settings,
        policy=replace(
            fixture.settings.policy,
            source_replay_materialization_byte_limit=byte_count + 1,
        ),
    )
    with pytest.raises(
        ExpertSourceReplayRequestError, match="differs from its request"
    ):
        replace(prepared, settings=looser_settings)


def test_aggregate_limit_deduplicates_starting_artifact_content_across_contexts(
    tmp_path,
):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    replay_case = prepared.cases[0]
    original_context = replay_case.task_context
    original_receipt = original_context.receipt
    second_receipt = ExpertSourceReplayContextMaterializationReceipt.mint(
        task_context_binding_id=content_id(
            "task-context-binding",
            {"label": "second-context-with-shared-artifact"},
        ),
        input_contract_fingerprint=original_receipt.input_contract_fingerprint,
        target_contract_fingerprint=original_receipt.target_contract_fingerprint,
        starting_artifacts=original_receipt.starting_artifacts,
        materializer_id=original_receipt.materializer_id,
        materializer_version=original_receipt.materializer_version,
    )
    second_context = VerifiedSourceReplayContext(
        receipt=second_receipt,
        starting_artifacts=original_context.starting_artifacts,
    )
    arguments = {
        "candidate": prepared.candidate,
        "parent": prepared.parent,
        "adapters": (replay_case.task_adapter,),
        "lineages": (replay_case.bundle_lineage,),
    }

    one_context_usage = fixture.coordinator._materialization_usage(
        **arguments,
        contexts=(original_context,),
    )
    shared_artifact_usage = fixture.coordinator._materialization_usage(
        **arguments,
        contexts=(original_context, second_context),
    )

    assert shared_artifact_usage == one_context_usage


def test_preflight_uses_one_decreasing_materialization_deadline(tmp_path):
    fixture = _request_fixture(tmp_path)

    class _AdvancingClock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            observed = self.value
            self.value += 1.0
            return observed

    clock = _AdvancingClock()
    fixture.coordinator.monotonic_clock = clock
    _prepared(fixture)
    parent_timeout = fixture.parent_provider.materializations[0][1].timeout_seconds
    adapter_timeout = fixture.adapter_provider.timeouts_seen[0]
    bundle_timeout = fixture.bundle_provider.timeouts_seen[0]
    context_timeout = fixture.context_provider.limits_seen[0].timeout_seconds

    assert parent_timeout > adapter_timeout > bundle_timeout > context_timeout

    expired = _request_fixture(tmp_path)

    class _ControlledClock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            return self.value

    controlled_clock = _ControlledClock()
    original_parent_provider = expired.parent_provider

    class _ExpiringParentProvider:
        def materialize_exact(self, release, parent_tree_receipt, limits):
            parent = original_parent_provider.materialize_exact(
                release,
                parent_tree_receipt,
                limits,
            )
            controlled_clock.value = float(
                expired.settings.policy.source_replay_materialization_timeout_seconds
            )
            return parent

    expired.coordinator.parent_provider = _ExpiringParentProvider()
    expired.coordinator.monotonic_clock = controlled_clock
    with pytest.raises(ExpertSourceReplayRequestError, match="deadline expired"):
        expired.coordinator.build(expired.attempt)


def test_preflight_retains_replays_and_counts_the_complete_bundle_lineage(tmp_path):
    fixture = _request_fixture(tmp_path, bundle_generations=2)
    prepared = _prepared(fixture)
    lineage = prepared.cases[0].bundle_lineage

    assert len(lineage.bundles) == 2
    _, total_bytes = fixture.coordinator._materialization_usage(
        candidate=prepared.candidate,
        parent=prepared.parent,
        adapters=(prepared.cases[0].task_adapter,),
        lineages=(lineage,),
        contexts=(prepared.cases[0].task_context,),
    )
    non_bundle_bytes = (
        prepared.candidate.byte_count
        + prepared.parent.byte_count
        + sum(
            descriptor.size
            for descriptor in prepared.cases[
                0
            ].task_adapter.source_extraction_receipt.source_tree_files
        )
        + len(prepared.cases[0].task_adapter.source_archive)
        + sum(
            len(payload)
            for payload in prepared.cases[0].task_adapter.proof_objects.values()
        )
        + len(prepared.cases[0].task_adapter.publisher_verification)
        + prepared.cases[0].task_context.byte_count
    )
    expected_bundle_bytes = sum(
        len(payload)
        for bundle in lineage.bundles
        for payload in bundle.artifacts.values()
    )
    assert total_bytes - non_bundle_bytes == expected_bundle_bytes
    assert expected_bundle_bytes > sum(
        len(payload) for payload in lineage.tip_bundle.artifacts.values()
    )
    assert (
        fixture.coordinator._materialization_usage(
            candidate=prepared.candidate,
            parent=prepared.parent,
            adapters=(prepared.cases[0].task_adapter,),
            lineages=(lineage, lineage),
            contexts=(prepared.cases[0].task_context,),
        )[1]
        == total_bytes
    )

    with pytest.raises(RunBundleLineageError, match="generation zero"):
        VerifiedRunBundleLineage(
            bundles=(lineage.tip_bundle,),
            tip_projection=lineage.tip_projection,
        )
    with pytest.raises(RunBundleLineageError, match="cycle"):
        VerifiedRunBundleLineage(
            bundles=(
                lineage.bundles[0],
                lineage.bundles[0],
                lineage.bundles[1],
            ),
            tip_projection=lineage.tip_projection,
        )


def test_request_rejects_execution_before_source_replay_is_current(tmp_path):
    fixture = _request_fixture(tmp_path)
    early_state_values = fixture.state.to_dict()
    early_state_values.pop("validation_state_id")
    early_state_values["next_stage"] = ExpertValidationStage.CONTRACT_SCHEMA
    early_state = ExpertCandidateValidationState.mint(
        **early_state_values,
    )

    class _EarlyValidationAuthority:
        def current(self, candidate_id):
            assert candidate_id == fixture.attempt.candidate_id
            return ExpertValidationPredecessor(
                latest_attempt=fixture.attempt,
                state=early_state,
            )

        def publish_parent_authority_invalidation(self, **_request):
            raise AssertionError("early validation cannot invalidate parent authority")

    fixture.coordinator.validation_authority = _EarlyValidationAuthority()

    with pytest.raises(ExpertSourceReplayRequestError, match="current authorized"):
        fixture.coordinator.build(fixture.attempt)


def test_verified_context_rejects_a_shallow_artifact_wrapper(tmp_path):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    verified_context = fixture.context_provider.verified

    with pytest.raises(ContractValidationError, match="unverified artifact closure"):
        VerifiedSourceReplayContext(
            receipt=verified_context.receipt,
            starting_artifacts=(
                SimpleNamespace(
                    artifact=verified_context.starting_artifacts[0].artifact,
                    source_contents=(
                        verified_context.starting_artifacts[0].source_contents
                    ),
                ),
            ),
        )
    with pytest.raises(ExpertSourceReplayRequestError, match="unverified authority"):
        replace(
            prepared.cases[0],
            task_adapter=SimpleNamespace(
                manifest=prepared.cases[0].task_adapter.manifest,
            ),
        )


def test_request_rejects_projection_adapter_and_context_substitution(tmp_path):
    unverified_lineage = _request_fixture(tmp_path)
    verified = unverified_lineage.bundle_provider.lineage
    unverified_lineage.bundle_provider.lineage = SimpleNamespace(
        bundle_ids=verified.bundle_ids,
        tip_bundle=verified.tip_bundle,
        tip_projection=verified.tip_projection,
    )
    with pytest.raises(ExpertSourceReplayRequestError, match="root-to-tip"):
        unverified_lineage.coordinator.build(unverified_lineage.attempt)

    manifest_only_bundle = _request_fixture(tmp_path)
    verified = manifest_only_bundle.bundle_provider.lineage
    manifest_only_bundle.bundle_provider.lineage = VerifiedRunBundleLineage(
        bundles=(SimpleNamespace(manifest=verified.tip_bundle.manifest),),
        tip_projection=verified.tip_projection,
    )
    with pytest.raises(ExpertSourceReplayRequestError, match="root-to-tip"):
        manifest_only_bundle.coordinator.build(manifest_only_bundle.attempt)

    shallow_adapter = _request_fixture(tmp_path)
    verified_adapter = shallow_adapter.adapter_provider.adapter

    class _ShallowAdapterProvider:
        def resolve_exact_bounded(self, **_request):
            return SimpleNamespace(
                manifest=verified_adapter.manifest,
                verification_receipt=verified_adapter.verification_receipt,
                dependency_ids=verified_adapter.dependency_ids,
            )

    shallow_adapter.coordinator.task_adapter_provider = _ShallowAdapterProvider()
    with pytest.raises(ExpertSourceReplayRequestError, match="verified package"):
        shallow_adapter.coordinator.build(shallow_adapter.attempt)

    missing_episode = _request_fixture(tmp_path)
    missing_episode.bundle_provider.lineage = replace(
        missing_episode.bundle_provider.lineage,
        tip_projection=replace(
            missing_episode.bundle_provider.lineage.tip_projection,
            episodes=(),
            derivation_objects=(),
        ),
    )
    prepared_from_bytes = _prepared(missing_episode)
    assert prepared_from_bytes.cases[0].episode == missing_episode.episode

    wrong_adapter = _request_fixture(tmp_path)
    historical_key = next(iter(wrong_adapter.adapter_provider.exact_adapters))
    substituted_manifest_values = (
        wrong_adapter.adapter_provider.adapter.manifest.to_dict()
    )
    substituted_manifest_values.pop("task_adapter_manifest_id")
    substituted_manifest_values["validation_refs"] = tuple(
        sorted(
            {
                *substituted_manifest_values["validation_refs"],
                "validation.substituted_adapter",
            }
        )
    )
    wrong_adapter.adapter_provider.exact_adapters[historical_key] = (
        verified_test_task_adapter(
            TaskAdapterManifest.mint(**substituted_manifest_values)
        )
    )
    with pytest.raises(ExpertSourceReplayRequestError, match="selection pin"):
        wrong_adapter.coordinator.build(wrong_adapter.attempt)

    wrong_context = _request_fixture(tmp_path)
    wrong_context.coordinator.task_context_provider = _ContextProvider(
        wrong_context.settings,
        wrong_context.episode.task_context_binding,
        input_fingerprint=tree_or_blob_digest(b"substituted-input-contract"),
    )
    with pytest.raises(ExpertSourceReplayRequestError, match="materialization"):
        wrong_context.coordinator.build(wrong_context.attempt)

    substituted_artifact = _request_fixture(tmp_path)

    class _SubstitutingArtifactProvider(_ContextProvider):
        def materialize_exact(self, context, expected_artifact_content_ids, limits):
            verified_context = super().materialize_exact(
                context,
                expected_artifact_content_ids,
                limits,
            )
            verified_artifact = verified_context.starting_artifacts[0]
            artifact_values = verified_artifact.artifact.to_dict()
            artifact_values.pop("starting_artifact_content_id")
            artifact_values["mount_path"] = "inputs/substituted"
            artifact = ExpertSourceReplayStartingArtifact.mint(**artifact_values)
            receipt = ExpertSourceReplayContextMaterializationReceipt.mint(
                task_context_binding_id=context.task_context_binding_id,
                input_contract_fingerprint=context.input_contract_fingerprint,
                target_contract_fingerprint=context.target_contract_fingerprint,
                starting_artifacts=(artifact,),
                materializer_id=(
                    self.settings.policy.source_replay_context_materializer_id
                ),
                materializer_version=(
                    self.settings.policy.source_replay_context_materializer_version
                ),
            )
            return VerifiedSourceReplayContext(
                receipt=receipt,
                starting_artifacts=(
                    VerifiedSourceReplayStartingArtifact(
                        artifact=artifact,
                        source_contents=verified_artifact.source_contents,
                    ),
                ),
            )

    substituted_artifact.coordinator.task_context_provider = (
        _SubstitutingArtifactProvider(
            substituted_artifact.settings,
            substituted_artifact.episode.task_context_binding,
        )
    )
    with pytest.raises(ExpertSourceReplayRequestError, match="captured authority"):
        substituted_artifact.coordinator.build(substituted_artifact.attempt)


def test_request_uses_historical_adapter_after_active_package_rotation(tmp_path):
    fixture = _request_fixture(tmp_path, rotate_active_adapter=True)

    prepared = _prepared(fixture)

    enrolled_active_manifest_id = fixture.attempt.task_adapter_pins[
        0
    ].task_adapter_manifest_id
    historical_manifest_id = (
        fixture.episode.artifact_environment.task_adapter_manifest_id
    )
    assert enrolled_active_manifest_id != historical_manifest_id
    assert prepared.request.cases[0].task_adapter_manifest_id == historical_manifest_id
    assert prepared.cases[0].task_adapter.manifest.task_adapter_manifest_id == (
        historical_manifest_id
    )


def test_request_rechecks_current_parent_after_materialization(tmp_path):
    fixture = _request_fixture(tmp_path)

    class _RotatingCurrentRelease:
        def __init__(self):
            self.calls = 0

        def current_release_id(self, scope_id):
            assert scope_id == "ml_ai"
            self.calls += 1
            if self.calls == 1:
                return fixture.attempt.parent_release_id
            return content_id("expert-base-release", {"label": "advanced"})

    rotating_current = _RotatingCurrentRelease()
    fixture.coordinator.current_release_provider = rotating_current
    fixture.validation_store.reducer.current_release_provider = rotating_current

    result = fixture.coordinator.build(fixture.attempt)

    assert result.prepared_request is None
    assert result.invalidated_state is not None
    assert result.invalidated_state.promotion_state is ExpertPromotionState.FAILED
    assert fixture.validation_store.current(fixture.attempt.candidate_id).state == (
        result.invalidated_state
    )


def test_materialized_case_rejects_a_reminted_request_field(tmp_path):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    materialized_case = prepared.cases[0]
    case_values = materialized_case.request_case.to_dict()
    case_values.pop("execution_case_id")
    case_values["matched_compute_binding_digest"] = tree_or_blob_digest(
        b"substituted-matched-compute"
    )
    with pytest.raises(ContractValidationError, match="matched-compute digest"):
        type(materialized_case.request_case).mint(**case_values)


def test_prepared_request_rejects_a_self_consistent_forged_selection_reason(
    tmp_path,
):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    materialized = prepared.cases[0]
    forged_selection_case = replace(
        materialized.selection_case,
        episode_reason_codes={
            materialized.episode.episode_id: ("forged_reason",),
        },
    )
    request_case_values = materialized.request_case.to_dict()
    request_case_values.pop("execution_case_id")
    request_case_values["episode_reason_codes"] = ("forged_reason",)
    forged_request_case = type(materialized.request_case).mint(**request_case_values)
    forged_materialized = replace(
        materialized,
        selection_case=forged_selection_case,
        request_case=forged_request_case,
    )
    request_values = prepared.request.to_dict()
    request_values.pop("execution_request_id")
    request_values["cases"] = (forged_request_case,)
    dependencies = set(prepared.request.exact_dependency_ids)
    dependencies.remove(materialized.request_case.execution_case_id)
    dependencies.add(forged_request_case.execution_case_id)
    request_values["exact_dependency_ids"] = tuple(sorted(dependencies))
    forged_request = type(prepared.request).mint(**request_values)

    with pytest.raises(
        ExpertSourceReplayRequestError, match="differs from its request"
    ):
        replace(
            prepared,
            request=forged_request,
            cases=(forged_materialized,),
        )


def test_aggregate_request_rejects_a_self_consistent_forged_candidate_leg(tmp_path):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    original_case = prepared.request.cases[0]
    original_leg = original_case.candidate_leg
    forged_candidate_id = content_id("expert-candidate", {"label": "forged"})
    forged_commit_id = content_id(
        "expert-candidate-commit",
        {"label": "forged"},
    )
    forged_leg = type(original_leg).mint(
        kind=original_leg.kind,
        expert_artifact_id=forged_candidate_id,
        expert_source_receipt_id=forged_commit_id,
        expert_tree_hash=tree_or_blob_digest(b"forged-candidate-tree"),
        exact_dependency_ids=tuple(sorted((forged_candidate_id, forged_commit_id))),
    )
    case_values = original_case.to_dict()
    case_values.pop("execution_case_id")
    case_values["candidate_leg"] = forged_leg
    case_dependencies = set(original_case.exact_dependency_ids)
    case_dependencies.difference_update(
        {
            original_leg.execution_leg_id,
            *original_leg.exact_dependency_ids,
        }
    )
    case_dependencies.update(
        {forged_leg.execution_leg_id, *forged_leg.exact_dependency_ids}
    )
    case_values["exact_dependency_ids"] = tuple(sorted(case_dependencies))
    forged_case = type(original_case).mint(**case_values)
    request_values = prepared.request.to_dict()
    request_values.pop("execution_request_id")
    request_values["cases"] = (forged_case,)
    request_dependencies = set(prepared.request.exact_dependency_ids)
    request_dependencies.difference_update(
        {original_case.execution_case_id, *original_case.exact_dependency_ids}
    )
    request_dependencies.update(
        {forged_case.execution_case_id, *forged_case.exact_dependency_ids}
    )
    request_values["exact_dependency_ids"] = tuple(sorted(request_dependencies))

    with pytest.raises(ContractValidationError, match="aggregate authority"):
        type(prepared.request).mint(**request_values)
