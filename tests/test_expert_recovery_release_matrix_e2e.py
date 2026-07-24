"""End-to-end clean-forward recovery through release-matrix decision."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertCandidateDerivationKind,
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertSealedCanaryAggregate,
    ExpertValidationStage,
)
from kapso.cross_run.expert.promotion import (
    decide_expert_release_matrix_promotion,
)
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixMode,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
    ExpertReleaseMatrixDecisionReason,
)
from kapso.cross_run.expert.promotion_plan import (
    derive_expert_release_matrix_plan,
)
from kapso.cross_run.expert.promotion_stage import (
    ExpertReleaseMatrixStageCoordinator,
)
from kapso.cross_run.expert.candidates import ExpertCandidateValidator
from kapso.cross_run.expert.replay_authority import (
    ExpertSourceReplayFreshAuthorityCoordinator,
    SourceReplayCurrentReleaseObservation,
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
from kapso.cross_run.expert.replay_request import (
    ExpertSourceReplayPreflightCoordinator,
)
from kapso.cross_run.expert.replay_stage import (
    ExpertSourceReplayStageOrchestrator,
)
from kapso.cross_run.expert.review import (
    ExpertAutomatedReviewCoordinator,
)
from kapso.cross_run.expert.store import ExpertCandidateStore
from kapso.cross_run.expert.task_evaluation_authority import (
    TaskEvaluationAuthorityError,
    TaskEvaluationFreshAuthorityCoordinator,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderRegistry,
    project_prepared_task_evaluation_cases,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    ExpertTaskEvaluationExecutionStore,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationLegKind,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationSourceBase,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    TaskEvaluationPreflightCoordinator,
)
from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertEvaluatorRunBuilder,
    ExpertValidationReducer,
)
from kapso.cross_run.expert.validation_store import (
    ExpertValidationStore,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from security_denylist_fixtures import matched_security_revocations
from test_expert_clean_recovery import (
    _historical_candidate_system,
)
from test_expert_candidates import sanitation_settings
from test_cross_run_retrieval import source_fixture
from test_expert_promotion_evidence import (
    _complete_execution,
)
from test_expert_release_matrix_reservation import (
    _quality_only_validation_settings,
)
from test_expert_replay_execution_store import (
    _MatchedLegProvider,
)
from test_expert_replay_stage import (
    _RegistryFactory,
)
from test_expert_review import AutomatedReviewProcess
from test_expert_source_replay import (
    _AdapterProvider,
    _CurrentReleaseProvider,
)
from test_expert_source_replay_request import (
    _request_fixture,
)
from test_expert_task_evaluation_execution_store import (
    _AdapterAuthority as _TaskAdapterAuthority,
)
from test_expert_task_evaluation_execution_store import (
    _CurrentAuthority as _TaskCurrentAuthority,
)
from test_expert_task_evaluation_execution_store import (
    _Provider as _TaskProvider,
)
from test_expert_task_evaluation_preflight import (
    _CandidateReader as _TaskCandidateReader,
)
from test_expert_task_evaluation_preflight import (
    _Clock,
    _CurrentAuthority as _TaskPreflightCurrentAuthority,
    _ExactAdapterProvider,
    _ParentProvider as _TaskParentProvider,
)
from test_expert_validation import (
    _AttestationVerifier,
    _ValidationStateProvider,
)


class _ReplayCurrentAuthority:
    def __init__(self, observation):
        self.observation = observation

    def current_release_observation(self, scope_id):
        assert scope_id == self.observation.scope_id
        return self.observation


class _ExactDenylistAuthority:
    def __init__(self, matched_subject_ids=()):
        self.matched_subject_ids = tuple(sorted(matched_subject_ids))
        self.checked_subject_ids = ()

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.checked_subject_ids = checked_subject_ids
        return SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=tree_or_blob_digest(b"scope binding"),
            snapshot_id=content_id(
                "security-denylist-snapshot",
                {"matched": self.matched_subject_ids},
            ),
            generation=1,
            publication_id=content_id(
                "github-publication",
                {"security": self.matched_subject_ids},
            ),
            repository_full_name="Leeroo-AI/kapso-security",
            repository_node_id="security_repo_node",
            pointer_digest=tree_or_blob_digest(b"security CURRENT"),
            authority_commit_sha="b" * 40,
            release_attestation_ref="attestations/security-denylist",
            checked_subject_ids=checked_subject_ids,
            matched_revocations=matched_security_revocations(self.matched_subject_ids),
        )


def _advance_to_stage(validation_store, snapshot, settings, target_stage):
    while snapshot.state.next_stage is not target_stage:
        stage = snapshot.state.next_stage
        assert stage is not None
        external_evidence_stages = {
            ExpertValidationStage.DEVELOPMENT_ANCHORS,
            ExpertValidationStage.CROSS_FAMILY_TRANSFER,
            ExpertValidationStage.SEALED_CANARY,
        }
        measurements = {}
        output_payloads = {"result.json": b'{"completed":true}'}
        if stage is ExpertValidationStage.SEALED_CANARY:
            measurements = {"quality": 1.0}
            evaluator_version = next(
                evaluator.evaluator_version
                for evaluator in settings.policy.evaluators
                if evaluator.stage is stage
            )
            aggregate = ExpertSealedCanaryAggregate(
                candidate_id=snapshot.latest_attempt.candidate_id,
                candidate_tree_hash=snapshot.latest_attempt.candidate_tree_hash,
                evaluator_version=evaluator_version,
                evaluated_case_count=1,
                aggregate_measurements=measurements,
            )
            output_payloads = {"aggregate.json": aggregate.to_json_bytes()}
        result = ExpertEvaluatorRunBuilder(settings).build(
            attempt=snapshot.latest_attempt,
            stage=stage,
            exact_additional_input_ids=(
                (
                    content_id(
                        "expert-evaluator-external-evidence",
                        {"stage": stage.value},
                    ),
                )
                if stage in external_evidence_stages
                else ()
            ),
            output_payloads=output_payloads,
            measurements=measurements,
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
    return snapshot


def _recovery_validation_settings():
    settings = _quality_only_validation_settings()
    return replace(
        settings,
        state_path="validation",
        policy=replace(
            settings.policy,
            sealed_canary_trust_root="test_sealed_canary_trust_root",
            promotion=replace(
                settings.policy.promotion,
                minimum_replicates_per_cell=1,
                minimum_distinct_context_lineage_pairs=1,
            ),
        ),
    )


def _reopen_candidate_store_for_validation(recovery, settings):
    workspace_root = recovery.candidate_store.state_root
    configured_expert_settings = replace(
        recovery.settings.expert,
        candidate_path="candidates",
        validation=settings,
    )
    candidate_store = ExpertCandidateStore(
        recovery.candidate_store.root,
        workspace_root,
        ExpertCandidateValidator(
            configured_expert_settings,
            sanitation_settings(),
        ),
    )
    return workspace_root, configured_expert_settings, candidate_store


def _publish_accepted_review(
    *,
    monkeypatch,
    tmp_path,
    label,
    validation_store,
    snapshot,
    stored_candidate,
    configured_expert_settings,
    workspace_root,
    settings,
):
    reviewer_outputs = {
        reviewer.reviewer_role: {
            "disposition": "core_eligible",
            "judgment": settings.policy.promotion.approval_judgment,
            "rationale": f"{reviewer.reviewer_id} accepted recovery evidence.",
        }
        for reviewer in settings.policy.reviewers
    }
    monkeypatch.setattr(
        "kapso.execution.coding_agents.structured_call.subprocess.run",
        AutomatedReviewProcess(reviewer_outputs),
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
    review_workspace = (tmp_path / label).resolve()
    review_workspace.mkdir(mode=0o700)
    review_execution = review_coordinator.execute(
        prepared_review,
        workspace=review_workspace,
    )
    return validation_store.publish_automated_review_stage(review_execution).snapshot


def _execute_and_publish_recovery_matrix(
    *,
    validation_store,
    prepared_task,
    reservation_snapshot,
    current_observation,
    scientific_subject_id,
    allowed_barrier_subject_id,
):
    execution_store = ExpertTaskEvaluationExecutionStore(
        ExpertTaskEvaluationExecutionStore.canonical_root(
            validation_store.root
        ).resolve(),
        validation_store.root,
        prepared_task.plan_join.settings.policy,
    )
    provider_keys = tuple(
        sorted(
            {
                case.provider_key
                for case in project_prepared_task_evaluation_cases(prepared_task)
            },
            key=lambda provider_key: provider_key.identity,
        )
    )
    providers = tuple(
        _TaskProvider(validation_store.root, provider_key)
        for provider_key in provider_keys
    )
    registry = TaskEvaluationExecutionProviderRegistry(
        prepared_task,
        providers,
    )
    scientific_denylist = _ExactDenylistAuthority((scientific_subject_id,))
    scientific_rejection_authority = TaskEvaluationFreshAuthorityCoordinator(
        reservation_authority=validation_store,
        execution_store=execution_store,
        current_release_authority=_TaskCurrentAuthority(current_observation),
        task_adapter_authority=_TaskAdapterAuthority(prepared_task),
        security_denylist_authority=scientific_denylist,
    )
    with execution_store.reservation_session(
        reservation_snapshot=reservation_snapshot,
        prepared_request=prepared_task,
    ) as task_session:
        allocation_permit = task_session.allocate_expected_leg()
        with pytest.raises(
            TaskEvaluationAuthorityError,
            match="denylist differs from exact security authority",
        ):
            scientific_rejection_authority.commit_spawn(
                prepared_request=prepared_task,
                reservation_id=reservation_snapshot.reservation.reservation_id,
                invocation_permit=allocation_permit,
                provider_registry=registry,
            )
    assert scientific_subject_id in scientific_denylist.checked_subject_ids

    execution_store = ExpertTaskEvaluationExecutionStore(
        execution_store.root,
        execution_store.trusted_root,
        execution_store.policy_settings,
    )
    allowed_denylist = _ExactDenylistAuthority((allowed_barrier_subject_id,))
    task_authority = TaskEvaluationFreshAuthorityCoordinator(
        reservation_authority=validation_store,
        execution_store=execution_store,
        current_release_authority=_TaskCurrentAuthority(current_observation),
        task_adapter_authority=_TaskAdapterAuthority(prepared_task),
        security_denylist_authority=allowed_denylist,
    )
    completed_task = _complete_execution(
        prepared=prepared_task,
        reservation_snapshot=reservation_snapshot,
        execution_store=execution_store,
        registry=registry,
        authority_coordinator=task_authority,
    )
    assert allowed_barrier_subject_id in allowed_denylist.checked_subject_ids
    return ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    ).publish_completed(
        completed_execution=completed_task,
        reservation_snapshot=reservation_snapshot,
        prepared_request=prepared_task,
    )


def _historical_recovery_matrix_case(
    tmp_path,
    monkeypatch,
    *,
    chain_length=2,
):
    source_fixture_root = tmp_path / "source-fixture"
    source_fixture_root.mkdir()
    source_fixture = _request_fixture(source_fixture_root)
    recovery = _historical_candidate_system(
        tmp_path / "recovery",
        chain_length=chain_length,
    )
    replay_basis = type(source_fixture.packet).mint(
        knowledge_snapshot_manifest=(source_fixture.packet.knowledge_snapshot_manifest),
        knowledge_record_closure_digest=(
            source_fixture.packet.knowledge_record_closure_digest
        ),
        configuration_fingerprint=recovery.replay_basis.configuration_fingerprint,
        scope_contract=recovery.fixture.case.scope,
        source_base_scope_contract=recovery.fixture.case.scope,
        source_base_release=recovery.barrier,
        source_base_tree_receipt=recovery.barrier_receipt,
        source_base_tree_hash=recovery.barrier_receipt.source_base_tree_hash,
        source_base_repository_map=recovery.fixture.case.repository_map,
        source_base_module_contracts=recovery.fixture.case.modules,
        episodes=source_fixture.packet.episodes,
        claims=source_fixture.packet.claims,
        trigger_observations=(),
        active_task_bindings=source_fixture.packet.active_task_bindings,
        proof_reference_ids=source_fixture.packet.proof_reference_ids,
        recovery_barrier_basis_packet_id=None,
    )
    recovery.fixture.security.blocked_release_ids.add(recovery.barrier.release_id)
    stored_candidate = recovery.coordinator.restore_historical(
        scope_contract=recovery.fixture.case.scope,
        replay_basis_packet=replay_basis,
    )
    admission = stored_candidate.recovery_admission
    assert admission is not None

    settings = _recovery_validation_settings()
    workspace_root, configured_expert_settings, candidate_store = (
        _reopen_candidate_store_for_validation(recovery, settings)
    )
    assert (
        candidate_store.read(stored_candidate.closure.manifest.candidate_id)
        == stored_candidate
    )
    adapter_provider = _AdapterProvider(
        replay_basis,
        source_adapter=source_fixture.adapter_provider.adapter,
    )
    current_provider = _CurrentReleaseProvider(recovery.barrier.release_id)
    eligibility = ExpertCandidateEligibilityEvaluator(
        settings,
        candidate_store,
        adapter_provider,
        current_provider,
    ).decide(candidate_id=stored_candidate.closure.manifest.candidate_id)
    assert eligibility.decision.eligible is True
    assert eligibility.decision.source_base_release_id == recovery.selected.release_id
    assert (
        eligibility.decision.expected_current_release_id == recovery.barrier.release_id
    )
    assert eligibility.decision.recovery_plan_id == (
        admission.recovery_plan.recovery_plan_id
    )

    reducer = ExpertValidationReducer(
        settings,
        candidate_store,
        _AttestationVerifier(),
        adapter_provider,
        current_provider,
        _ValidationStateProvider(),
    )
    validation_store = ExpertValidationStore(
        (workspace_root / settings.state_path).resolve(),
        workspace_root,
        settings,
        reducer,
    )
    snapshot = validation_store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    snapshot = _advance_to_stage(
        validation_store,
        snapshot,
        settings,
        ExpertValidationStage.SOURCE_RUN_REPLAY,
    )
    attempt = snapshot.latest_attempt
    assert attempt is not None
    recovery_context = stored_candidate.closure.validation_context
    selected_source = VerifiedTaskEvaluationSourceBase(
        release_manifest=recovery_context.source_base_release,
        source_base_tree_receipt=recovery_context.source_base_tree_receipt,
        source_contents=recovery.fixture.case.source_contents,
    )

    source_preflight = ExpertSourceReplayPreflightCoordinator(
        settings,
        candidate_store,
        validation_store,
        current_provider,
        _TaskParentProvider(selected_source),
        source_fixture.bundle_provider,
        adapter_provider,
        source_fixture.context_provider,
        time.monotonic,
    )
    prepared_source_result = source_preflight.build(attempt)
    assert prepared_source_result.invalidated_state is None
    prepared_source = prepared_source_result.prepared_request
    assert prepared_source is not None
    assert prepared_source.request.expected_current_release_id == (
        recovery.barrier.release_id
    )
    assert prepared_source.request.allowed_control_security_subject_ids == (
        recovery.barrier.release_id,
    )

    source_execution_store = ExpertSourceReplayExecutionStore(
        (validation_store.root / "source-replay-executions").resolve(),
        validation_store.root,
        settings.policy,
    )
    source_provider = _MatchedLegProvider(
        validation_store.root,
        expert_source_replay_execution_provider_key(prepared_source.cases[0]),
    )
    source_registry = ExpertSourceReplayExecutionProviderRegistry((source_provider,))
    source_current_observation = SourceReplayCurrentReleaseObservation.mint(
        scope_id=recovery.fixture.case.scope.scope_id,
        release_id=recovery.barrier.release_id,
        publication_id=recovery.fixture.remotes[
            0
        ].pointer.publication_record.publication_id,
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repo_node",
        current_pointer_digest=tree_or_blob_digest(b"recovery barrier CURRENT"),
        current_pointer_commit_sha="a" * 40,
        validation_closure_ids=(),
    )
    replay_current_authority = _ReplayCurrentAuthority(source_current_observation)
    replay_denylist = _ExactDenylistAuthority((recovery.barrier.release_id,))
    source_spawn = ExpertSourceReplayFreshAuthorityCoordinator(
        validation_store,
        source_execution_store,
        replay_current_authority,
        adapter_provider,
        replay_denylist,
    )
    source_publication = ExpertSourceReplayDecisionPublicationCoordinator(
        validation_store=validation_store,
        execution_store=source_execution_store,
        current_release_authority=replay_current_authority,
        task_adapter_authority=adapter_provider,
        security_denylist_authority=replay_denylist,
    )
    source_snapshot = ExpertSourceReplayStageOrchestrator(
        validation_store=validation_store,
        preflight_coordinator=source_preflight,
        execution_store=source_execution_store,
        provider_registry_factory=_RegistryFactory(
            prepared_source,
            source_registry,
        ),
        spawn_authority_coordinator=source_spawn,
        publication_coordinator=source_publication,
    ).run(attempt)
    source_snapshot = _advance_to_stage(
        validation_store,
        source_snapshot,
        settings,
        ExpertValidationStage.AUTOMATED_REVIEW,
    )
    assert source_snapshot.state.next_stage is ExpertValidationStage.AUTOMATED_REVIEW
    assert len(source_provider.invocations) == 2

    matrix_snapshot = _publish_accepted_review(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        label="historical-review",
        validation_store=validation_store,
        snapshot=source_snapshot,
        stored_candidate=stored_candidate,
        configured_expert_settings=configured_expert_settings,
        workspace_root=workspace_root,
        settings=settings,
    )
    assert matrix_snapshot.state.next_stage is ExpertValidationStage.RELEASE_MATRIX
    assert matrix_snapshot.latest_attempt is not None

    prepared_plan = derive_expert_release_matrix_plan(
        state=matrix_snapshot.state,
        attempt=matrix_snapshot.latest_attempt,
        accepted_stage_results=matrix_snapshot.accepted_stage_results,
        source_replay_request=prepared_source.request,
        stored_candidate=stored_candidate,
        verified_adapters=tuple(adapter_provider.exact_adapters.values()),
        validation_policy=settings.policy.validation_policy(),
        validation_settings=settings,
    )
    assert prepared_plan.plan.mode is ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
    assert prepared_plan.plan.source_base_release_id == recovery.selected.release_id
    assert prepared_plan.plan.expected_current_release_id == recovery.barrier.release_id
    plan_commit = validation_store.reserve_release_matrix_plan(
        expected_transition_id=matrix_snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    )
    reopened_validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened_plan = reopened_validation_store.reopen_release_matrix_plan_reservation(
        evaluation_plan_id=prepared_plan.plan.evaluation_plan_id,
        prepared_plan=prepared_plan,
    )
    assert reopened_plan == plan_commit.reservation

    task_preflight = TaskEvaluationPreflightCoordinator(
        settings=settings,
        plan_reservation_authority=reopened_validation_store,
        candidate_reader=_TaskCandidateReader(stored_candidate),
        source_base_provider=_TaskParentProvider(selected_source),
        adapter_provider=_ExactAdapterProvider(prepared_plan.verified_adapters),
        current_release_authority=_TaskPreflightCurrentAuthority(
            (
                recovery.fixture.current.observation,
                recovery.fixture.current.observation,
            )
        ),
        monotonic_clock=_Clock(),
    )
    prepared_task = task_preflight.build(reopened_plan)
    assert prepared_task.plan_join.request.mode is (
        ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
    )
    assert all(len(case.request_case.legs) == 2 for case in prepared_task.cases)
    task_reservation = reopened_validation_store.reserve_task_evaluation(
        expected_transition_id=matrix_snapshot.transition.transition_id,
        prepared_request=prepared_task,
    ).reservation
    restarted_validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened_task_reservation = (
        restarted_validation_store.reopen_task_evaluation_reservation(
            reservation_id=task_reservation.reservation.reservation_id,
            prepared_request=prepared_task,
        )
    )
    assert reopened_task_reservation == task_reservation

    committed_stage = _execute_and_publish_recovery_matrix(
        validation_store=restarted_validation_store,
        prepared_task=prepared_task,
        reservation_snapshot=reopened_task_reservation,
        current_observation=recovery.fixture.current.observation,
        scientific_subject_id=recovery.selected.release_id,
        allowed_barrier_subject_id=recovery.barrier.release_id,
    )
    report = committed_stage.stage_result.release_matrix_report
    assert report.mode is ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
    assert all(row.control_replicate_values is not None for row in report.evidence_rows)
    assert (
        committed_stage.snapshot.state.next_stage
        is ExpertValidationStage.PUBLICATION_ELIGIBILITY
    )

    decision = decide_expert_release_matrix_promotion(
        stage_result=committed_stage.stage_result,
        attempt=matrix_snapshot.latest_attempt,
        settings=settings,
    )
    assert decision.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
    assert decision.reason is ExpertReleaseMatrixDecisionReason.RECOVERY_NON_REGRESSION
    assert committed_stage.snapshot.state.promotion_state is (
        ExpertPromotionState.VALIDATING
    )
    return SimpleNamespace(
        recovery=recovery,
        stored_candidate=stored_candidate,
        admission=admission,
        settings=settings,
        configured_expert_settings=configured_expert_settings,
        candidate_store=candidate_store,
        validation_store=restarted_validation_store,
        matrix_snapshot=matrix_snapshot,
        prepared_task=prepared_task,
        committed_stage=committed_stage,
        decision=decision,
    )


def test_historical_clean_forward_recovery_reaches_durable_matrix_decision(
    tmp_path,
    monkeypatch,
):
    _historical_recovery_matrix_case(tmp_path, monkeypatch)


def _canonical_empty_recovery_matrix_case(
    tmp_path,
    monkeypatch,
):
    _scope, _context, episode, _prior, _claim, _bundle, _report = source_fixture()
    recovery = _historical_candidate_system(
        tmp_path / "empty-recovery",
        empty_selection=True,
        episodes=(episode,),
    )
    recovery.fixture.security.blocked_release_ids.add(recovery.barrier.release_id)
    stored_candidate = recovery.coordinator.bootstrap_empty(
        scope_contract=recovery.fixture.case.scope,
        replay_basis_packet=recovery.replay_basis,
    ).stored_candidate
    admission = stored_candidate.recovery_admission
    assert admission is not None
    assert stored_candidate.closure.manifest.derivation_kind is (
        ExpertCandidateDerivationKind.AGENT_RECOVERY_BOOTSTRAP
    )
    assert stored_candidate.closure.manifest.source_base_release_id is None
    assert admission.recovery_plan.source_base_release_id is None
    assert admission.allowed_control_security_subject_ids == (
        recovery.barrier.release_id,
    )

    settings = _recovery_validation_settings()
    workspace_root, configured_expert_settings, candidate_store = (
        _reopen_candidate_store_for_validation(recovery, settings)
    )
    assert (
        candidate_store.read(stored_candidate.closure.manifest.candidate_id)
        == stored_candidate
    )
    adapter_provider = _AdapterProvider(recovery.replay_basis)
    current_provider = _CurrentReleaseProvider(recovery.barrier.release_id)
    eligibility = ExpertCandidateEligibilityEvaluator(
        settings,
        candidate_store,
        adapter_provider,
        current_provider,
    ).decide(candidate_id=stored_candidate.closure.manifest.candidate_id)
    assert eligibility.decision.eligible is True
    assert eligibility.decision.source_base_release_id is None
    assert eligibility.decision.expected_current_release_id == (
        recovery.barrier.release_id
    )
    assert eligibility.decision.source_replay_selection is None

    reducer = ExpertValidationReducer(
        settings,
        candidate_store,
        _AttestationVerifier(),
        adapter_provider,
        current_provider,
        _ValidationStateProvider(),
    )
    validation_store = ExpertValidationStore(
        (workspace_root / settings.state_path).resolve(),
        workspace_root,
        settings,
        reducer,
    )
    snapshot = validation_store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    snapshot = _advance_to_stage(
        validation_store,
        snapshot,
        settings,
        ExpertValidationStage.AUTOMATED_REVIEW,
    )
    assert ExpertValidationStage.SOURCE_RUN_REPLAY not in (
        snapshot.latest_attempt.required_stages
    )

    matrix_snapshot = _publish_accepted_review(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        label="empty-review",
        validation_store=validation_store,
        snapshot=snapshot,
        stored_candidate=stored_candidate,
        configured_expert_settings=configured_expert_settings,
        workspace_root=workspace_root,
        settings=settings,
    )
    assert matrix_snapshot.state.next_stage is ExpertValidationStage.RELEASE_MATRIX
    assert matrix_snapshot.latest_attempt is not None

    prepared_plan = derive_expert_release_matrix_plan(
        state=matrix_snapshot.state,
        attempt=matrix_snapshot.latest_attempt,
        accepted_stage_results=matrix_snapshot.accepted_stage_results,
        source_replay_request=None,
        stored_candidate=stored_candidate,
        verified_adapters=tuple(adapter_provider.exact_adapters.values()),
        validation_policy=settings.policy.validation_policy(),
        validation_settings=settings,
    )
    plan = prepared_plan.plan
    assert plan.mode is ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
    assert plan.source_base_release_id is None
    assert plan.expected_current_release_id == recovery.barrier.release_id
    assert prepared_plan.source_replay_request is None
    plan_commit = validation_store.reserve_release_matrix_plan(
        expected_transition_id=matrix_snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    )
    reopened_validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened_plan = reopened_validation_store.reopen_release_matrix_plan_reservation(
        evaluation_plan_id=plan.evaluation_plan_id,
        prepared_plan=prepared_plan,
    )
    assert reopened_plan == plan_commit.reservation

    source_base_provider = _TaskParentProvider(None)
    task_preflight = TaskEvaluationPreflightCoordinator(
        settings=settings,
        plan_reservation_authority=reopened_validation_store,
        candidate_reader=_TaskCandidateReader(stored_candidate),
        source_base_provider=source_base_provider,
        adapter_provider=_ExactAdapterProvider(prepared_plan.verified_adapters),
        current_release_authority=_TaskPreflightCurrentAuthority(
            (
                recovery.fixture.current.observation,
                recovery.fixture.current.observation,
            )
        ),
        monotonic_clock=_Clock(),
    )
    prepared_task = task_preflight.build(reopened_plan)
    request = prepared_task.plan_join.request
    assert request.mode is ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
    assert request.source_base_release_id is None
    assert request.expected_current_release_id == recovery.barrier.release_id
    assert request.allowed_control_security_subject_ids == (
        recovery.barrier.release_id,
    )
    assert source_base_provider.calls == []
    assert all(
        tuple(leg.kind for leg in case.request_case.legs)
        == (TaskEvaluationLegKind.CANDIDATE,)
        for case in prepared_task.cases
    )
    task_reservation = reopened_validation_store.reserve_task_evaluation(
        expected_transition_id=matrix_snapshot.transition.transition_id,
        prepared_request=prepared_task,
    ).reservation
    restarted_validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened_task_reservation = (
        restarted_validation_store.reopen_task_evaluation_reservation(
            reservation_id=task_reservation.reservation.reservation_id,
            prepared_request=prepared_task,
        )
    )
    assert reopened_task_reservation == task_reservation
    assert (
        reopened_task_reservation.reservation.observed_current_release_id
        == recovery.barrier.release_id
    )

    committed_stage = _execute_and_publish_recovery_matrix(
        validation_store=restarted_validation_store,
        prepared_task=prepared_task,
        reservation_snapshot=reopened_task_reservation,
        current_observation=recovery.fixture.current.observation,
        scientific_subject_id=stored_candidate.closure.manifest.candidate_id,
        allowed_barrier_subject_id=recovery.barrier.release_id,
    )
    report = committed_stage.stage_result.release_matrix_report
    assert report.mode is ExpertReleaseMatrixMode.CLEAN_FORWARD_RECOVERY
    assert report.source_base_release_id is None
    assert all(row.control_replicate_values is None for row in report.evidence_rows)
    assert (
        committed_stage.snapshot.state.next_stage
        is ExpertValidationStage.PUBLICATION_ELIGIBILITY
    )

    decision = decide_expert_release_matrix_promotion(
        stage_result=committed_stage.stage_result,
        attempt=matrix_snapshot.latest_attempt,
        settings=settings,
    )
    assert decision.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
    assert decision.reason is (
        ExpertReleaseMatrixDecisionReason.RECOVERY_STANDALONE_COVERAGE
    )
    return SimpleNamespace(
        recovery=recovery,
        stored_candidate=stored_candidate,
        admission=admission,
        settings=settings,
        configured_expert_settings=configured_expert_settings,
        candidate_store=candidate_store,
        validation_store=restarted_validation_store,
        matrix_snapshot=matrix_snapshot,
        prepared_task=prepared_task,
        committed_stage=committed_stage,
        decision=decision,
    )


def test_canonical_empty_recovery_reaches_durable_standalone_decision(
    tmp_path,
    monkeypatch,
):
    _canonical_empty_recovery_matrix_case(tmp_path, monkeypatch)
