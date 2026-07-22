from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import time
from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertValidationStage,
)
from kapso.cross_run.expert import (
    ExpertCandidateProposalEngine,
    ExpertCandidateStore,
    ExpertCandidateValidator,
    ExpertCandidateWorkspaceManager,
    ExpertCapabilityGeneralizer,
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvaluator,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.expert.promotion_plan import (
    ExpertReleaseMatrixPlanError,
    PreparedExpertReleaseMatrixPlan,
    _canonical_verified_adapters,
    derive_expert_release_matrix_plan,
)
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayExecutionProviderRegistry,
    expert_source_replay_execution_provider_key,
)
from kapso.cross_run.expert.replay_execution_store import (
    ExpertSourceReplayExecutionStore,
)
from kapso.cross_run.expert.replay_publication import (
    ExpertSourceReplayDecisionPublicationCoordinator,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayStageResultRecord,
)
from kapso.cross_run.expert.replay_request import (
    ExpertSourceReplayPreflightCoordinator,
    VerifiedExpertSourceReplayParent,
)
from kapso.cross_run.expert.replay_stage import ExpertSourceReplayStageOrchestrator
from kapso.cross_run.expert.review import ExpertAutomatedReviewCoordinator
from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertEvaluatorRunBuilder,
    ExpertValidationReducer,
)
from kapso.cross_run.expert.validation_store import (
    ExpertValidationStore,
)
from test_expert_candidate_workspace import FixtureSourceMaterializer
from test_expert_candidates import (
    expert_settings,
    sanitation_settings,
)
from test_expert_proposal import (
    BootstrapProposalRunner,
    generalizer_output,
    released_observation_packet,
)
from test_expert_replay_execution_store import _MatchedLegProvider, _coordinator
from test_expert_replay_stage import _RegistryFactory
from test_expert_review import AutomatedReviewProcess
from test_expert_source_replay import (
    _AdapterProvider,
    _CurrentReleaseProvider,
    _validation_policy,
)
from test_expert_source_replay_request import (
    _prepared,
    _request_fixture,
)
from test_expert_triggers import trigger_settings
from test_expert_validation import (
    _AttestationVerifier,
    _ValidationStateProvider,
)


def _quality_only_validation_settings():
    settings = _validation_policy()
    quality_dimension = next(
        dimension
        for dimension in settings.policy.promotion.pareto_dimensions
        if dimension.dimension_id == "quality"
    )
    return replace(
        settings,
        policy=replace(
            settings.policy,
            promotion=replace(
                settings.policy.promotion,
                pareto_dimensions=(quality_dimension,),
            ),
        ),
    )


class _ReplayParentProvider:
    def __init__(self, contents):
        self.contents = contents

    def materialize_exact(self, release, parent_tree_receipt, _limits):
        return VerifiedExpertSourceReplayParent(
            release_manifest=release,
            parent_tree_receipt=parent_tree_receipt,
            source_contents=self.contents,
        )


class _ParentChangingDuringAdapterResolution:
    def __init__(self, expected_release_id, changed_release_id):
        self.expected_release_id = expected_release_id
        self.changed_release_id = changed_release_id
        self.observations = 0

    def current_release_id(self, _scope_id):
        self.observations += 1
        if self.observations == 1:
            return self.expected_release_id
        return self.changed_release_id


def _release_matrix_fixture(tmp_path, monkeypatch):
    settings = _quality_only_validation_settings()
    source_fixture = _request_fixture(
        tmp_path,
        validation_settings=settings,
    )
    source_packet = source_fixture.stored.closure.trigger_packet
    released_packet, materialized_parent, parent_contents = released_observation_packet(
        ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        "A provenance field can be added without changing topology.",
    )
    packet = ExpertTriggerEvidencePacket.mint(
        knowledge_snapshot_manifest=source_packet.knowledge_snapshot_manifest,
        knowledge_record_closure_digest=(source_packet.knowledge_record_closure_digest),
        configuration_fingerprint=source_packet.configuration_fingerprint,
        scope_contract=released_packet.scope_contract,
        parent_scope_contract=released_packet.parent_scope_contract,
        parent_release=released_packet.parent_release,
        parent_tree_receipt=released_packet.parent_tree_receipt,
        parent_tree_hash=released_packet.parent_tree_hash,
        repository_map=released_packet.repository_map,
        module_contracts=released_packet.module_contracts,
        episodes=source_packet.episodes,
        claims=source_packet.claims,
        trigger_observations=released_packet.trigger_observations,
        active_task_bindings=source_packet.active_task_bindings,
        proof_reference_ids=source_packet.proof_reference_ids,
    )
    configured_expert_settings = replace(
        expert_settings(),
        validation=settings,
    )
    workspace_root = (tmp_path / "release-matrix-system").resolve()
    workspace_root.mkdir(mode=0o700)
    cross_run_root = (workspace_root / ".kapso" / "cross_run").resolve()
    cross_run_root.mkdir(mode=0o700, parents=True)
    candidate_store = ExpertCandidateStore(
        (workspace_root / configured_expert_settings.candidate_path).resolve(),
        cross_run_root,
        ExpertCandidateValidator(
            configured_expert_settings,
            sanitation_settings(),
        ),
    )
    proposal_workspace_root = (cross_run_root / "proposal-workspaces").resolve()
    proposal_payload = json.loads(generalizer_output(packet))
    proposal_payload["changed_module_contracts"][0]["supporting_episode_ids"] = [
        source_packet.episodes[0].episode_id
    ]
    proposal_runner = BootstrapProposalRunner(
        (workspace_root / configured_expert_settings.agent_artifact_path).resolve(),
        json.dumps(proposal_payload, sort_keys=True) + "\n",
        {
            "src/reproducible_execution/__init__.py": (
                b"def execute(task):\n    return task.run_with_provenance()\n"
            )
        },
    )
    proposal_engine = ExpertCandidateProposalEngine(
        settings=configured_expert_settings,
        runner=proposal_runner,
        workspace_manager=ExpertCandidateWorkspaceManager(
            proposal_workspace_root,
            cross_run_root,
            configured_expert_settings,
            FixtureSourceMaterializer(parent_contents),
        ),
        candidate_store=candidate_store,
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)
    stored_candidate = (
        ExpertCapabilityGeneralizer(proposal_engine)
        .propose(
            packet=packet,
            decision=decision,
            materialized_parent=materialized_parent,
        )
        .stored_candidate
    )
    adapter_provider = _AdapterProvider(packet)
    current_release_provider = _CurrentReleaseProvider(packet.parent_release_id)
    eligibility = ExpertCandidateEligibilityEvaluator(
        settings,
        candidate_store,
        adapter_provider,
        current_release_provider,
    ).decide(candidate_id=stored_candidate.closure.manifest.candidate_id)
    reducer = ExpertValidationReducer(
        settings,
        candidate_store,
        _AttestationVerifier(),
        adapter_provider,
        current_release_provider,
        _ValidationStateProvider(),
    )
    validation_store = ExpertValidationStore(
        (workspace_root / settings.state_path).resolve(),
        cross_run_root,
        settings,
        reducer,
    )
    validation_snapshot = validation_store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    assert validation_snapshot.latest_attempt is not None
    while (
        validation_snapshot.state.next_stage
        is not ExpertValidationStage.SOURCE_RUN_REPLAY
    ):
        stage = validation_snapshot.state.next_stage
        assert stage is not None
        result = ExpertEvaluatorRunBuilder(settings).build(
            attempt=validation_snapshot.latest_attempt,
            stage=stage,
            exact_additional_input_ids=(),
            output_payloads={"result.json": b'{"completed":true}'},
            measurements={},
            costs={},
            duration_seconds=1.0,
            outcome=ExpertEvaluatorOutcome.PASSED,
            signature="test-signature",
        )
        validation_snapshot = validation_store.publish_evaluator_result(
            candidate_id=validation_snapshot.state.candidate_id,
            expected_transition_id=validation_snapshot.transition.transition_id,
            result=result,
        ).snapshot
        assert validation_snapshot.latest_attempt is not None
    source_preflight = ExpertSourceReplayPreflightCoordinator(
        settings,
        candidate_store,
        validation_store,
        current_release_provider,
        _ReplayParentProvider(parent_contents),
        source_fixture.bundle_provider,
        adapter_provider,
        source_fixture.context_provider,
        time.monotonic,
    )
    source_prepared_result = source_preflight.build(validation_snapshot.latest_attempt)
    assert source_prepared_result.prepared_request is not None
    source_prepared = source_prepared_result.prepared_request
    execution_store = ExpertSourceReplayExecutionStore(
        (validation_store.root / "source-replay-executions").resolve(),
        validation_store.root,
        settings.policy,
    )
    provider = _MatchedLegProvider(
        validation_store.root,
        expert_source_replay_execution_provider_key(source_prepared.cases[0]),
    )
    registry = ExpertSourceReplayExecutionProviderRegistry((provider,))
    registry_factory = _RegistryFactory(source_prepared, registry)
    spawn_coordinator = _coordinator(
        SimpleNamespace(
            validation_store=validation_store,
            adapter_provider=adapter_provider,
        ),
        source_prepared,
        execution_store,
    )
    publication_coordinator = ExpertSourceReplayDecisionPublicationCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
        current_release_authority=spawn_coordinator.current_release_authority,
        task_adapter_authority=adapter_provider,
        security_denylist_authority=spawn_coordinator.security_denylist_authority,
    )
    source_orchestrator = ExpertSourceReplayStageOrchestrator(
        validation_store=validation_store,
        preflight_coordinator=source_preflight,
        execution_store=execution_store,
        provider_registry_factory=registry_factory,
        spawn_authority_coordinator=spawn_coordinator,
        publication_coordinator=publication_coordinator,
    )
    source_snapshot = source_orchestrator.run(validation_snapshot.latest_attempt)

    outputs = {
        reviewer.reviewer_role: {
            "disposition": "core_eligible",
            "judgment": settings.policy.promotion.approval_judgment,
            "rationale": f"{reviewer.reviewer_id} accepted the exact evidence.",
        }
        for reviewer in settings.policy.reviewers
    }
    runner = AutomatedReviewProcess(outputs)
    monkeypatch.setattr(
        "kapso.execution.coding_agents.structured_call.subprocess.run",
        runner,
    )
    review_coordinator = ExpertAutomatedReviewCoordinator(
        configured_expert_settings,
        workspace_root,
    )
    validation_store._bind_automated_review_publication_authority(review_coordinator)
    assert source_snapshot.latest_attempt is not None
    prepared_review = review_coordinator.prepare(
        stored_candidate=stored_candidate,
        validation_attempt=source_snapshot.latest_attempt,
        authorization_transition_id=source_snapshot.transition.transition_id,
        authorization_state=source_snapshot.state,
        accepted_stage_results=source_snapshot.accepted_stage_results,
    )
    review_workspace = (tmp_path / "release-matrix-review").resolve()
    review_workspace.mkdir(mode=0o700)
    review_execution = review_coordinator.execute(
        prepared_review,
        workspace=review_workspace,
    )
    snapshot = validation_store.publish_automated_review_stage(
        review_execution
    ).snapshot
    assert snapshot.latest_attempt is not None
    verified_adapters = tuple(
        {
            (
                case.task_adapter.manifest.task_adapter_manifest_id,
                case.task_adapter.verification_receipt.verification_receipt_id,
            ): case.task_adapter
            for case in source_prepared.cases
        }.values()
    )
    prepared_plan = derive_expert_release_matrix_plan(
        state=snapshot.state,
        attempt=snapshot.latest_attempt,
        accepted_stage_results=snapshot.accepted_stage_results,
        source_replay_request=source_prepared.request,
        stored_candidate=stored_candidate,
        verified_adapters=verified_adapters,
        validation_policy=settings.policy.validation_policy(),
        validation_settings=settings,
    )
    return validation_store, snapshot, prepared_plan


def test_plan_derivation_uses_every_accepted_source_case_and_fingerprint(
    tmp_path,
    monkeypatch,
):
    _, snapshot, prepared = _release_matrix_fixture(tmp_path, monkeypatch)
    source_result = next(
        result
        for result in snapshot.accepted_stage_results
        if type(result) is ExpertSourceReplayStageResultRecord
    )
    request_case_ids = {
        case.execution_case_id for case in prepared.source_replay_request.cases
    }
    comparison_case_ids = {
        case.execution_case_id
        for case in source_result.paired_comparison_receipt.case_comparisons
    }

    assert (
        {
            provenance.source_execution_case_id
            for provenance in prepared.plan.provenance_bindings
        }
        == request_case_ids
        == comparison_case_ids
    )
    assert {
        cell.evaluation_fingerprint.evaluation_fingerprint_id
        for cell in prepared.plan.evaluation_cells
    } == {
        fingerprint_id
        for case in prepared.source_replay_request.cases
        for fingerprint_id in case.source_evaluation_fingerprint_ids
    }


def test_release_matrix_plan_reservation_is_atomic_reopenable_and_unchanged(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )

    committed = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared,
    )
    replayed = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared,
    )
    reopened_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened = reopened_store.reopen_release_matrix_plan_reservation(
        evaluation_plan_id=prepared.plan.evaluation_plan_id,
        prepared_plan=prepared,
    )

    assert committed.replayed is False
    assert replayed.replayed is True
    assert committed.reservation.snapshot == snapshot
    assert validation_store.snapshot(prepared.plan.candidate_id) == snapshot
    assert replayed.reservation == committed.reservation
    assert reopened == committed.reservation
    assert reopened.operation.request_record_id == prepared.plan.evaluation_plan_id
    assert (
        reopened.operation.expected_transition_id == snapshot.transition.transition_id
    )


def test_concurrent_identical_release_matrix_plans_bind_once(tmp_path, monkeypatch):
    validation_store, snapshot, prepared = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )

    def reserve(_position):
        return validation_store.reserve_release_matrix_plan(
            expected_transition_id=snapshot.transition.transition_id,
            prepared_plan=prepared,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(reserve, range(8)))

    assert sum(not result.replayed for result in results) == 1
    assert len({result.reservation.operation.operation_id for result in results}) == 1


def test_release_matrix_reservation_rejects_a_changed_parent_authority(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )
    validation_store.reducer.current_release_provider.release_id = content_id(
        "expert-base-release",
        {"changed": "before-plan-admission"},
    )

    with pytest.raises(ExpertReleaseMatrixPlanError, match="parent authority changed"):
        validation_store.reserve_release_matrix_plan(
            expected_transition_id=snapshot.transition.transition_id,
            prepared_plan=prepared,
        )

    assert validation_store.snapshot(prepared.plan.candidate_id) == snapshot


def test_release_matrix_reservation_rechecks_parent_after_adapter_resolution(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
    )
    validation_store.reducer.current_release_provider = (
        _ParentChangingDuringAdapterResolution(
            prepared.plan.parent_release_id,
            content_id(
                "expert-base-release",
                {"changed": "during-adapter-resolution"},
            ),
        )
    )

    with pytest.raises(
        ExpertReleaseMatrixPlanError,
        match="parent authority changed during adapter resolution",
    ):
        validation_store.reserve_release_matrix_plan(
            expected_transition_id=snapshot.transition.transition_id,
            prepared_plan=prepared,
        )

    assert validation_store.snapshot(prepared.plan.candidate_id) == snapshot


def test_verified_adapter_packages_have_one_canonical_runtime_order(tmp_path):
    fixture = _request_fixture(tmp_path, rotate_active_adapter=True)
    adapters = tuple(reversed(tuple(fixture.adapter_provider.exact_adapters.values())))

    canonical = _canonical_verified_adapters(adapters)

    assert tuple(
        (
            adapter.manifest.task_adapter_manifest_id,
            adapter.verification_receipt.verification_receipt_id,
        )
        for adapter in canonical
    ) == tuple(
        sorted(
            (
                adapter.manifest.task_adapter_manifest_id,
                adapter.verification_receipt.verification_receipt_id,
            )
            for adapter in adapters
        )
    )
    with pytest.raises(ExpertReleaseMatrixPlanError, match="not unique"):
        _canonical_verified_adapters((canonical[0], canonical[0]))


def test_prepared_plan_rejects_an_unrelated_source_request(tmp_path, monkeypatch):
    _, _, prepared = _release_matrix_fixture(tmp_path, monkeypatch)
    other_fixture = _request_fixture(
        tmp_path,
        validation_settings=_quality_only_validation_settings(),
    )
    unrelated_request = _prepared(other_fixture).request

    with pytest.raises(ExpertReleaseMatrixPlanError, match="source authority"):
        PreparedExpertReleaseMatrixPlan(
            plan=prepared.plan,
            state=prepared.state,
            attempt=prepared.attempt,
            accepted_stage_results=prepared.accepted_stage_results,
            source_replay_request=unrelated_request,
            stored_candidate=prepared.stored_candidate,
            verified_adapters=prepared.verified_adapters,
            validation_policy=prepared.validation_policy,
            validation_settings=prepared.validation_settings,
        )
