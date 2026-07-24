from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import shutil
import time
from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertValidationStage,
    TaskAdapterManifest,
)
from kapso.cross_run.expert.candidates import ExpertCandidateValidator
from kapso.cross_run.expert.generalizer import ExpertCapabilityGeneralizer
from kapso.cross_run.expert.proposal import ExpertCandidateProposalEngine
from kapso.cross_run.expert.store import ExpertCandidateStore
from kapso.cross_run.expert.triggers import (
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvaluator,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.expert.workspace import ExpertCandidateWorkspaceManager
from kapso.cross_run.expert.promotion_plan import (
    ExpertReleaseMatrixPlanError,
    PreparedExpertReleaseMatrixPlan,
    _canonical_verified_adapters,
    derive_expert_release_matrix_plan,
)
from kapso.cross_run.expert.promotion_evidence import (
    derive_expert_release_matrix_source_rows,
)
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixMode,
    ExpertReleaseMatrixProvenanceKind,
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
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationSourceBase,
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
    bootstrap_candidate_closure,
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
from test_cross_run_contracts import verified_test_task_adapter
from test_expert_validation import (
    _AttestationVerifier,
    _ValidationStateProvider,
)
from task_adapter_matrix_fixtures import task_adapter_release_matrix_case


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

    def materialize_exact(self, release, source_base_tree_receipt, _limits):
        return VerifiedTaskEvaluationSourceBase(
            release_manifest=release,
            source_base_tree_receipt=source_base_tree_receipt,
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


def _release_matrix_fixture(
    tmp_path,
    monkeypatch,
    *,
    rotate_active_adapter=False,
    add_active_case=False,
    include_source_evidence_authority=False,
):
    settings = _quality_only_validation_settings()
    source_fixture = _request_fixture(
        tmp_path,
        validation_settings=settings,
    )
    source_packet = source_fixture.packet
    released_packet, materialized_source_base, source_base_contents = (
        released_observation_packet(
            ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
            "A provenance field can be added without changing topology.",
        )
    )
    packet = ExpertTriggerEvidencePacket.mint(
        knowledge_snapshot_manifest=source_packet.knowledge_snapshot_manifest,
        knowledge_record_closure_digest=(source_packet.knowledge_record_closure_digest),
        configuration_fingerprint=source_packet.configuration_fingerprint,
        scope_contract=released_packet.scope_contract,
        source_base_scope_contract=released_packet.source_base_scope_contract,
        source_base_release=released_packet.source_base_release,
        source_base_tree_receipt=released_packet.source_base_tree_receipt,
        source_base_tree_hash=released_packet.source_base_tree_hash,
        source_base_repository_map=released_packet.source_base_repository_map,
        source_base_module_contracts=released_packet.source_base_module_contracts,
        episodes=source_packet.episodes,
        claims=source_packet.claims,
        trigger_observations=released_packet.trigger_observations,
        active_task_bindings=source_packet.active_task_bindings,
        proof_reference_ids=source_packet.proof_reference_ids,
        recovery_barrier_basis_packet_id=None,
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
            FixtureSourceMaterializer(source_base_contents),
        ),
        candidate_store=candidate_store,
    )
    decision = ExpertTriggerEvaluator(trigger_settings()).evaluate(packet)
    stored_candidate = (
        ExpertCapabilityGeneralizer(proposal_engine)
        .propose(
            packet=packet,
            decision=decision,
            materialized_source_base=materialized_source_base,
        )
        .stored_candidate
    )
    adapter_provider = _AdapterProvider(
        packet,
        rotate_active=rotate_active_adapter,
    )
    if add_active_case:
        active_manifest = adapter_provider.adapter.manifest
        comparison_bindings = active_manifest.task_evaluator.metric_comparison_bindings
        additional_case = task_adapter_release_matrix_case(
            scope_contract_id=active_manifest.scope_contract_id,
            scope_id=packet.scope_contract.scope_id,
            task_family_id=active_manifest.task_family_id,
            task_adapter_id=active_manifest.task_adapter_id,
            evaluator_fingerprint=comparison_bindings[0].evaluator_fingerprint,
            metric_directions=tuple(
                (binding.metric_name, binding.objective_direction)
                for binding in comparison_bindings
            ),
            transfer_dimensions=dict(
                active_manifest.release_matrix_cases[
                    0
                ].task_context_binding.transfer_dimensions
            ),
            label="additional-active-case",
        )
        active_values = active_manifest.to_dict()
        active_values.pop("task_adapter_manifest_id")
        active_values["release_matrix_cases"] = tuple(
            sorted(
                (*active_manifest.release_matrix_cases, additional_case),
                key=lambda case: case.release_matrix_case_id,
            )
        )
        adapter_provider.adapter = verified_test_task_adapter(
            TaskAdapterManifest.mint(**active_values),
            source_contents=adapter_provider.adapter.source_contents,
        )
        adapter_provider.exact_adapters[
            (
                adapter_provider.adapter.manifest.task_adapter_manifest_id,
                adapter_provider.adapter.verification_receipt.verification_receipt_id,
            )
        ] = adapter_provider.adapter
    current_release_provider = _CurrentReleaseProvider(packet.source_base_release_id)
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
        _ReplayParentProvider(source_base_contents),
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
    verified_adapters = tuple(adapter_provider.exact_adapters.values())
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
    if include_source_evidence_authority:
        return (
            validation_store,
            snapshot,
            prepared_plan,
            execution_store,
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

    source_provenances = tuple(
        provenance
        for provenance in prepared.plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
    )
    adapter_provenances = tuple(
        provenance
        for provenance in prepared.plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
    )
    assert (
        {provenance.source_execution_case_id for provenance in source_provenances}
        == request_case_ids
        == comparison_case_ids
    )
    assert {provenance.provenance_case_id for provenance in adapter_provenances} == {
        case.release_matrix_case_id
        for authority in prepared.plan.adapter_authorities
        if authority.task_adapter_pin in prepared.attempt.task_adapter_pins
        for case in authority.task_adapter_manifest.release_matrix_cases
    }
    assert {
        cell.evaluation_fingerprint.evaluation_fingerprint_id
        for cell in prepared.plan.evaluation_cells
        if cell.provenance_binding_id
        in {provenance.provenance_binding_id for provenance in source_provenances}
    } == {
        fingerprint_id
        for case in prepared.source_replay_request.cases
        for fingerprint_id in case.source_evaluation_fingerprint_ids
    }


def test_historical_source_package_cases_are_not_planned_as_active_cases(
    tmp_path,
    monkeypatch,
):
    _, _, prepared = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        rotate_active_adapter=True,
    )
    authority_by_id = {
        authority.adapter_authority_id: authority
        for authority in prepared.plan.adapter_authorities
    }
    active_manifest_ids = {
        pin.task_adapter_manifest_id for pin in prepared.attempt.task_adapter_pins
    }
    adapter_authority_manifest_ids = {
        authority_by_id[
            provenance.adapter_authority_id
        ].task_adapter_manifest.task_adapter_manifest_id
        for provenance in prepared.plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
    }
    source_authority_manifest_ids = {
        authority_by_id[
            provenance.adapter_authority_id
        ].task_adapter_manifest.task_adapter_manifest_id
        for provenance in prepared.plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
    }

    assert adapter_authority_manifest_ids == active_manifest_ids
    assert source_authority_manifest_ids.isdisjoint(active_manifest_ids)
    assert len(prepared.plan.adapter_authorities) == 2


def test_plan_enumerates_every_case_from_the_active_signed_manifest(
    tmp_path,
    monkeypatch,
):
    _, _, prepared = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        add_active_case=True,
    )
    active_manifest_ids = {
        pin.task_adapter_manifest_id for pin in prepared.attempt.task_adapter_pins
    }
    active_authorities = tuple(
        authority
        for authority in prepared.plan.adapter_authorities
        if authority.task_adapter_manifest.task_adapter_manifest_id
        in active_manifest_ids
    )
    adapter_provenances = tuple(
        provenance
        for provenance in prepared.plan.provenance_bindings
        if provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
    )

    assert len(active_authorities) == 1
    assert {provenance.provenance_case_id for provenance in adapter_provenances} == {
        case.release_matrix_case_id
        for case in active_authorities[0].task_adapter_manifest.release_matrix_cases
    }
    assert len(adapter_provenances) == 2


def _bootstrap_release_matrix_fixture(
    tmp_path,
    monkeypatch,
):
    settings = _quality_only_validation_settings()
    configured_expert_settings = replace(
        expert_settings(),
        validation=settings,
    )
    workspace_root = (tmp_path / "bootstrap-system").resolve()
    workspace_root.mkdir(mode=0o700)
    cross_run_root = (workspace_root / ".kapso" / "cross_run").resolve()
    cross_run_root.mkdir(mode=0o700, parents=True)
    candidates = ExpertCandidateStore(
        (workspace_root / configured_expert_settings.candidate_path).resolve(),
        cross_run_root,
        ExpertCandidateValidator(
            configured_expert_settings,
            sanitation_settings(),
        ),
    )
    stored_candidate = candidates.persist(bootstrap_candidate_closure())
    packet = stored_candidate.closure.derivation.trigger_packet
    adapter_provider = _AdapterProvider(packet)
    current_release_provider = _CurrentReleaseProvider(None)
    eligibility = ExpertCandidateEligibilityEvaluator(
        settings,
        candidates,
        adapter_provider,
        current_release_provider,
    ).decide(candidate_id=stored_candidate.closure.manifest.candidate_id)
    reducer = ExpertValidationReducer(
        settings,
        candidates,
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
    snapshot = validation_store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    assert snapshot.latest_attempt is not None
    while snapshot.state.next_stage is not ExpertValidationStage.AUTOMATED_REVIEW:
        stage = snapshot.state.next_stage
        assert stage is not None
        assert stage not in {
            ExpertValidationStage.SOURCE_RUN_REPLAY,
            ExpertValidationStage.RELEASE_MATRIX,
        }
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
        assert snapshot.latest_attempt is not None
    outputs = {
        reviewer.reviewer_role: {
            "disposition": "core_eligible",
            "judgment": settings.policy.promotion.approval_judgment,
            "rationale": f"{reviewer.reviewer_id} accepted bootstrap evidence.",
        }
        for reviewer in settings.policy.reviewers
    }
    monkeypatch.setattr(
        "kapso.execution.coding_agents.structured_call.subprocess.run",
        AutomatedReviewProcess(outputs),
    )
    review_coordinator = ExpertAutomatedReviewCoordinator(
        configured_expert_settings,
        workspace_root,
    )
    validation_store._bind_automated_review_publication_authority(review_coordinator)
    prepared_review = review_coordinator.prepare(
        stored_candidate=stored_candidate,
        validation_attempt=snapshot.latest_attempt,
        authorization_transition_id=snapshot.transition.transition_id,
        authorization_state=snapshot.state,
        accepted_stage_results=snapshot.accepted_stage_results,
    )
    review_workspace = (tmp_path / "bootstrap-review").resolve()
    review_workspace.mkdir(mode=0o700)
    review_execution = review_coordinator.execute(
        prepared_review,
        workspace=review_workspace,
    )
    snapshot = validation_store.publish_automated_review_stage(
        review_execution
    ).snapshot
    assert snapshot.latest_attempt is not None
    prepared = derive_expert_release_matrix_plan(
        state=snapshot.state,
        attempt=snapshot.latest_attempt,
        accepted_stage_results=snapshot.accepted_stage_results,
        source_replay_request=None,
        stored_candidate=stored_candidate,
        verified_adapters=(adapter_provider.adapter,),
        validation_policy=settings.policy.validation_policy(),
        validation_settings=settings,
    )
    return validation_store, snapshot, prepared, adapter_provider


def test_bootstrap_plan_reserves_and_reopens_without_source_replay(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, adapter_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    committed = validation_store.reserve_release_matrix_plan(
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

    assert prepared.plan.mode is ExpertReleaseMatrixMode.BOOTSTRAP
    assert prepared.source_replay_request is None
    assert all(
        provenance.provenance_kind is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
        for provenance in prepared.plan.provenance_bindings
    )
    assert {
        provenance.provenance_case_id
        for provenance in prepared.plan.provenance_bindings
    } == {
        case.release_matrix_case_id
        for case in adapter_provider.adapter.manifest.release_matrix_cases
    }
    assert committed.reservation.evaluation_plan == prepared.plan
    assert reopened == committed.reservation
    with pytest.raises(ValueError, match="one accepted source result"):
        reopened_store.reopen_release_matrix_source_evidence(
            plan_reservation=reopened,
        )


def test_bootstrap_reservation_rejects_a_release_appearing_during_admission(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, _adapter_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    validation_store.reducer.current_release_provider = (
        _ParentChangingDuringAdapterResolution(
            None,
            content_id(
                "expert-base-release",
                {"appeared": "during-bootstrap-admission"},
            ),
        )
    )

    with pytest.raises(
        ExpertReleaseMatrixPlanError,
        match="source-base authority changed during adapter resolution",
    ):
        validation_store.reserve_release_matrix_plan(
            expected_transition_id=snapshot.transition.transition_id,
            prepared_plan=prepared,
        )

    assert validation_store.snapshot(prepared.plan.candidate_id) == snapshot


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


def test_accepted_source_evidence_reopens_after_restart_without_execution_journal(
    tmp_path,
    monkeypatch,
):
    (
        validation_store,
        snapshot,
        prepared,
        execution_store,
    ) = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        include_source_evidence_authority=True,
    )
    committed = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared,
    )
    shutil.rmtree(execution_store.root)
    monkeypatch.setattr(
        _MatchedLegProvider,
        "execute_leg",
        lambda *_args, **_kwargs: pytest.fail(
            "accepted source evidence must not rerun its provider"
        ),
    )
    reopened_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened_plan = reopened_store.reopen_release_matrix_plan_reservation(
        evaluation_plan_id=prepared.plan.evaluation_plan_id,
        prepared_plan=prepared,
    )
    evidence = reopened_store.reopen_release_matrix_source_evidence(
        plan_reservation=reopened_plan,
    )
    rows = derive_expert_release_matrix_source_rows(
        validation_store=reopened_store,
        plan_reservation=reopened_plan,
    )
    source_cells = tuple(
        cell
        for cell in prepared.plan.evaluation_cells
        if next(
            provenance
            for provenance in prepared.plan.provenance_bindings
            if provenance.provenance_binding_id == cell.provenance_binding_id
        ).provenance_kind
        is ExpertReleaseMatrixProvenanceKind.SOURCE_REPLAY
    )
    case_comparisons = {
        comparison.execution_case_id: comparison
        for comparison in evidence.stage_result.paired_comparison_receipt.case_comparisons
    }
    provenances = {
        provenance.provenance_binding_id: provenance
        for provenance in prepared.plan.provenance_bindings
    }

    assert evidence.plan_reservation == committed.reservation
    assert tuple(row.evaluation_cell_id for row in rows) == tuple(
        cell.evaluation_cell_id for cell in source_cells
    )
    for cell, row in zip(source_cells, rows, strict=True):
        comparison = case_comparisons[
            provenances[cell.provenance_binding_id].source_execution_case_id
        ]
        fingerprint_comparison = next(
            item
            for item in comparison.fingerprint_comparisons
            if item.evaluation_fingerprint.evaluation_fingerprint_id
            == cell.evaluation_fingerprint.evaluation_fingerprint_id
        )
        assert row.candidate_observation_event_id == (
            comparison.candidate_result_accepted_event_id
        )
        assert row.control_observation_event_id == (
            comparison.control_result_accepted_event_id
        )
        assert row.candidate_replicate_values == (
            fingerprint_comparison.candidate_result.replicate_values
        )
        assert row.control_replicate_values == (
            fingerprint_comparison.control_result.replicate_values
        )
    accepted_object_ids = (
        evidence.request.execution_request_id,
        evidence.reservation.reservation_id,
        evidence.stage_result.stage_result_record_id,
        evidence.stage_result.paired_comparison_receipt.paired_comparison_receipt_id,
    )
    for identity in accepted_object_ids:
        namespace, digest = identity.split(":sha256:", 1)
        object_path = reopened_store.object_root / namespace / f"{digest}.json"
        payload = object_path.read_bytes()
        object_path.unlink()
        with pytest.raises(ValueError):
            reopened_store.reopen_release_matrix_source_evidence(
                plan_reservation=reopened_plan,
            )
        object_path.write_bytes(payload)
        object_path.chmod(0o600)


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

    with pytest.raises(
        ExpertReleaseMatrixPlanError, match="source-base authority changed"
    ):
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
            prepared.plan.source_base_release_id,
            content_id(
                "expert-base-release",
                {"changed": "during-adapter-resolution"},
            ),
        )
    )

    with pytest.raises(
        ExpertReleaseMatrixPlanError,
        match="source-base authority changed during adapter resolution",
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
