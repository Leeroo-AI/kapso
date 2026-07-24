"""End-to-end source replay for an authenticated historical recovery candidate."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertValidationStage,
)
from kapso.cross_run.expert.replay_authority import (
    ExpertSourceReplayFreshAuthorityCoordinator,
)
from kapso.cross_run.expert.replay_authority_contracts import (
    ExpertSourceReplayFreshAuthorityError,
    SourceReplayCurrentReleaseObservation,
)
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayExecutionProviderRegistry,
    expert_source_replay_execution_provider_key,
)
from kapso.cross_run.expert.replay_execution_store import (
    ExpertSourceReplayExecutionStore,
    SourceReplayExecutionJournalEventKind,
    source_replay_execution_schedule,
)
from kapso.cross_run.expert.replay_publication import (
    ExpertSourceReplayDecisionPublicationCoordinator,
)
from kapso.cross_run.expert.replay_publication_contracts import (
    ExpertSourceReplayPublicationError,
)
from kapso.cross_run.expert.replay_request import (
    ExpertSourceReplayPreflightCoordinator,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationSourceBase,
)
from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertEvaluatorRunBuilder,
    ExpertValidationReducer,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStore
from kapso.cross_run.security_authority_contracts import SecurityDenylistObservation
from security_denylist_fixtures import matched_security_revocations
from test_expert_clean_recovery import _historical_candidate_system, _remint
from test_expert_replay_execution_store import _MatchedLegProvider
from test_expert_source_replay import _CandidateReader, _CurrentReleaseProvider
from test_expert_source_replay_request import _request_fixture
from test_expert_triggers import trigger_packet
from test_expert_validation import _AttestationVerifier, _ValidationStateProvider


class _HistoricalSourceBaseProvider:
    def __init__(self, source_contents):
        self.source_contents = source_contents

    def materialize_exact(self, release, source_base_tree_receipt, limits):
        assert limits.maximum_entries > 0
        assert limits.maximum_bytes > 0
        return VerifiedTaskEvaluationSourceBase(
            release_manifest=release,
            source_base_tree_receipt=source_base_tree_receipt,
            source_contents=self.source_contents,
        )


class _CurrentReleaseAuthority:
    def __init__(self, observation):
        self.observation = observation

    def current_release_observation(self, scope_id):
        assert scope_id == self.observation.scope_id
        return self.observation


class _MutableSecurityDenylistAuthority:
    def __init__(self):
        self.matched_subject_ids = ()
        self.observations = []

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        assert set(self.matched_subject_ids).issubset(checked_subject_ids)
        observation = SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=tree_or_blob_digest(
                b"historical recovery scope binding"
            ),
            snapshot_id=content_id(
                "security-denylist-snapshot",
                {"matched_subject_ids": self.matched_subject_ids},
            ),
            generation=len(self.observations) + 1,
            publication_id=content_id(
                "github-publication",
                {
                    "generation": len(self.observations) + 1,
                    "matched_subject_ids": self.matched_subject_ids,
                },
            ),
            repository_full_name="Leeroo-AI/kapso-security",
            repository_node_id="security_repository_node",
            pointer_digest=tree_or_blob_digest(
                f"security CURRENT {len(self.observations) + 1}".encode("utf-8")
            ),
            authority_commit_sha=f"{len(self.observations) + 1:040x}",
            release_attestation_ref="attestations/security-denylist",
            checked_subject_ids=checked_subject_ids,
            matched_revocations=matched_security_revocations(
                self.matched_subject_ids
            ),
        )
        self.observations.append(observation)
        return observation


def _advance_to_source_replay(validation_store, snapshot, settings):
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
    return snapshot


def _resolved_case(prepared, registry, execution_case_id):
    return next(
        resolved
        for resolved in registry.resolve_all(prepared)
        if resolved.materialized_case.request_case.execution_case_id
        == execution_case_id
    )


def test_historical_recovery_source_replay_survives_restarts_and_fences_revocations(
    tmp_path,
):
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    capture = _request_fixture(capture_root)
    recovery = _historical_candidate_system(tmp_path / "recovery")
    projection = capture.bundle_provider.lineage.tip_projection
    replay_basis = trigger_packet(
        settings=recovery.settings.expert.triggers,
        source_base_repository_map=recovery.fixture.case.repository_map,
        source_base_module_contracts=recovery.fixture.case.modules,
        source_base_release=recovery.barrier,
        current_scope_contract=recovery.fixture.case.scope,
        source_base_scope_contract=recovery.fixture.case.scope,
        episodes=(capture.episode,),
        knowledge_source_bundle=projection.source_bundle,
        knowledge_sanitation_report=projection.sanitation_report,
        knowledge_extra_facts=projection.derivation_objects,
        knowledge_projection_derivation_ids=tuple(
            event.event_id for event in projection.derivation_objects
        ),
    )
    replay_basis = _remint(
        replay_basis,
        source_base_tree_receipt=recovery.barrier_receipt,
        source_base_tree_hash=recovery.barrier_receipt.source_base_tree_hash,
    )
    recovery.fixture.security.blocked_release_ids.add(recovery.barrier.release_id)
    stored = recovery.coordinator.restore_historical(
        scope_contract=recovery.fixture.case.scope,
        replay_basis_packet=replay_basis,
    )
    admission = stored.recovery_admission
    assert admission is not None
    assert stored.closure.manifest.source_base_release_id == (
        recovery.selected.release_id
    )
    assert admission.recovery_plan.activation_predecessor_release_id == (
        recovery.barrier.release_id
    )
    assert admission.allowed_control_security_subject_ids == (
        recovery.barrier.release_id,
    )

    settings = replace(
        capture.settings,
        policy=replace(
            capture.settings.policy,
            sealed_canary_trust_root="test_sealed_canary_root",
        ),
    )
    candidate_reader = _CandidateReader(stored)
    current_release_provider = _CurrentReleaseProvider(recovery.barrier.release_id)
    eligibility = ExpertCandidateEligibilityEvaluator(
        settings,
        candidate_reader,
        capture.adapter_provider,
        current_release_provider,
    ).decide(candidate_id=stored.closure.manifest.candidate_id)
    assert eligibility.decision.eligible is True
    assert eligibility.decision.source_base_release_id == recovery.selected.release_id
    assert eligibility.decision.expected_current_release_id == (
        recovery.barrier.release_id
    )
    assert eligibility.decision.recovery_plan_id == (
        admission.recovery_plan.recovery_plan_id
    )

    reducer = ExpertValidationReducer(
        settings,
        candidate_reader,
        _AttestationVerifier(),
        capture.adapter_provider,
        current_release_provider,
        _ValidationStateProvider(),
    )
    state_root = tmp_path / "validation-state"
    state_root.mkdir(mode=0o700)
    validation_store = ExpertValidationStore(
        (state_root / "validation").resolve(),
        state_root.resolve(),
        settings,
        reducer,
    )
    started = validation_store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    source_replay_snapshot = _advance_to_source_replay(
        validation_store,
        started,
        settings,
    )
    attempt = source_replay_snapshot.latest_attempt
    assert attempt is not None

    preflight = ExpertSourceReplayPreflightCoordinator(
        settings,
        candidate_reader,
        validation_store,
        current_release_provider,
        _HistoricalSourceBaseProvider(recovery.fixture.case.source_contents),
        capture.bundle_provider,
        capture.adapter_provider,
        capture.context_provider,
        time.monotonic,
    )
    prepared_result = preflight.build(attempt)
    assert prepared_result.invalidated_state is None
    prepared = prepared_result.prepared_request
    assert prepared is not None
    request = prepared.request
    assert request.source_base_release_id == recovery.selected.release_id
    assert request.expected_current_release_id == recovery.barrier.release_id
    assert request.allowed_control_security_subject_ids == (
        recovery.barrier.release_id,
    )
    reservation = validation_store.reserve_source_replay(
        expected_transition_id=source_replay_snapshot.transition.transition_id,
        prepared_request=prepared,
    ).reservation

    restarted_validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        settings,
        reducer,
    )
    reopened = restarted_validation_store.reopen_source_replay_reservation(
        reservation_id=reservation.reservation_id,
        prepared_request=prepared,
    )
    assert reopened.reservation == reservation
    assert reopened.snapshot == source_replay_snapshot

    execution_store = ExpertSourceReplayExecutionStore(
        ExpertSourceReplayExecutionStore.canonical_root(
            restarted_validation_store.root
        ),
        restarted_validation_store.root,
        settings.policy,
    )
    current_observation = SourceReplayCurrentReleaseObservation.mint(
        scope_id=recovery.selected.scope_id,
        release_id=recovery.barrier.release_id,
        publication_id=content_id(
            "github-publication",
            {"release_id": recovery.barrier.release_id},
        ),
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repository_node",
        current_pointer_digest=tree_or_blob_digest(b"blocked recovery CURRENT"),
        current_pointer_commit_sha="a" * 40,
        validation_closure_ids=(),
    )
    current_authority = _CurrentReleaseAuthority(current_observation)
    denylist_authority = _MutableSecurityDenylistAuthority()
    spawn_coordinator = ExpertSourceReplayFreshAuthorityCoordinator(
        restarted_validation_store,
        execution_store,
        current_authority,
        capture.adapter_provider,
        denylist_authority,
    )
    provider = _MatchedLegProvider(
        restarted_validation_store.root,
        expert_source_replay_execution_provider_key(prepared.cases[0]),
    )
    registry = ExpertSourceReplayExecutionProviderRegistry((provider,))
    schedule = source_replay_execution_schedule(reservation, request)

    with execution_store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        first_permit = session.allocate_expected_leg()
        first_allocation = first_permit.require_current_allocation(execution_store)
        first_resolved = _resolved_case(
            prepared,
            registry,
            first_allocation.execution_case_id,
        )
        denylist_authority.matched_subject_ids = (recovery.selected.release_id,)
        with pytest.raises(
            ExpertSourceReplayFreshAuthorityError,
            match="denylist authority rejected",
        ):
            spawn_coordinator.commit_spawn(
                prepared_request=prepared,
                reservation_id=reservation.reservation_id,
                invocation_permit=first_permit,
                resolved_case=first_resolved,
            )
        assert tuple(event.event_kind for event in session.events) == (
            SourceReplayExecutionJournalEventKind.INVOCATION_ALLOCATED,
        )

        denylist_authority.matched_subject_ids = (recovery.barrier.release_id,)
        first_execution = spawn_coordinator.commit_spawn(
            prepared_request=prepared,
            reservation_id=reservation.reservation_id,
            invocation_permit=first_permit,
            resolved_case=first_resolved,
        )
        session.record_result_received(first_execution.execute())
        session.accept_received_result()
        for _execution_case_id, _execution_leg_id in schedule[1:]:
            permit = session.allocate_expected_leg()
            allocation = permit.require_current_allocation(execution_store)
            execution = spawn_coordinator.commit_spawn(
                prepared_request=prepared,
                reservation_id=reservation.reservation_id,
                invocation_permit=permit,
                resolved_case=_resolved_case(
                    prepared,
                    registry,
                    allocation.execution_case_id,
                ),
            )
            session.record_result_received(execution.execute())
            session.accept_received_result()
        completed = session.completed_execution()

    spawn_fences = tuple(
        event.spawn_authority_fence
        for event in completed.events
        if event.event_kind is SourceReplayExecutionJournalEventKind.SPAWN_COMMITTED
    )
    assert len(spawn_fences) == len(schedule)
    assert all(fence is not None for fence in spawn_fences)
    assert all(
        fence.security_denylist_observation.matched_subject_ids
        == (recovery.barrier.release_id,)
        for fence in spawn_fences
        if fence is not None
    )

    final_validation_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        settings,
        reducer,
    )
    final_execution_store = ExpertSourceReplayExecutionStore(
        ExpertSourceReplayExecutionStore.canonical_root(
            final_validation_store.root
        ),
        final_validation_store.root,
        settings.policy,
    )
    with final_execution_store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as restarted_session:
        reopened_completed = restarted_session.completed_execution()
    assert reopened_completed.events == completed.events

    publication_coordinator = ExpertSourceReplayDecisionPublicationCoordinator(
        validation_store=final_validation_store,
        execution_store=final_execution_store,
        current_release_authority=current_authority,
        task_adapter_authority=capture.adapter_provider,
        security_denylist_authority=denylist_authority,
    )
    denylist_authority.matched_subject_ids = (recovery.selected.release_id,)
    with pytest.raises(
        ExpertSourceReplayPublicationError,
        match="denylist rejected",
    ):
        publication_coordinator.publish_completed(
            completed_execution=reopened_completed,
            reservation=reservation,
            prepared_request=prepared,
        )
    assert final_validation_store.snapshot(request.candidate_id) == (
        source_replay_snapshot
    )

    denylist_authority.matched_subject_ids = (recovery.barrier.release_id,)
    published = publication_coordinator.publish_completed(
        completed_execution=reopened_completed,
        reservation=reservation,
        prepared_request=prepared,
    )

    assert published.replayed is False
    assert published.snapshot.state.promotion_state is ExpertPromotionState.VALIDATING
    source_replay_position = attempt.required_stages.index(
        ExpertValidationStage.SOURCE_RUN_REPLAY
    )
    assert published.snapshot.state.next_stage is (
        attempt.required_stages[source_replay_position + 1]
    )
    assert published.snapshot.accepted_stage_results[-1] == published.stage_result
    assert (
        published.stage_result.publication_authority_fence
        .security_denylist_observation.matched_subject_ids
        == (recovery.barrier.release_id,)
    )
    reopened_final = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        settings,
        reducer,
    ).snapshot(request.candidate_id)
    assert reopened_final == published.snapshot
