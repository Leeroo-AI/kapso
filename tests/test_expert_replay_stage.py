from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kapso.cross_run.contracts import (
    ExpertPromotionState,
    ExpertValidationStage,
)
from kapso.cross_run.canonical import content_id
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
from kapso.cross_run.expert.replay_stage import (
    ExpertSourceReplayPermanentlyInterruptedError,
    ExpertSourceReplayStageError,
    ExpertSourceReplayStageOrchestrator,
)
from kapso.cross_run.expert.validation_store import (
    ExpertValidationCompareAndSwapError,
)
from test_expert_replay_execution_store import (
    _MatchedLegProvider,
    _coordinator,
)
from test_expert_source_replay_request import _prepared, _request_fixture


class _RegistryFactory:
    def __init__(self, prepared, registry):
        self.prepared = prepared
        self.registry = registry
        self.calls = 0

    def __call__(self, prepared):
        assert prepared == self.prepared
        self.calls += 1
        return self.registry


def _stage_fixture(tmp_path):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    execution_store = ExpertSourceReplayExecutionStore(
        (fixture.validation_store.root / "source-replay-executions").resolve(),
        fixture.validation_store.root,
        prepared.settings.policy,
    )
    provider = _MatchedLegProvider(
        fixture.validation_store.root,
        expert_source_replay_execution_provider_key(prepared.cases[0]),
    )
    registry = ExpertSourceReplayExecutionProviderRegistry((provider,))
    registry_factory = _RegistryFactory(prepared, registry)
    spawn_coordinator = _coordinator(
        fixture,
        prepared,
        execution_store,
    )
    publication_coordinator = ExpertSourceReplayDecisionPublicationCoordinator(
        validation_store=fixture.validation_store,
        execution_store=execution_store,
        current_release_authority=(spawn_coordinator.current_release_authority),
        task_adapter_authority=fixture.adapter_provider,
        security_denylist_authority=(spawn_coordinator.security_denylist_authority),
    )
    orchestrator = ExpertSourceReplayStageOrchestrator(
        validation_store=fixture.validation_store,
        preflight_coordinator=fixture.coordinator,
        execution_store=execution_store,
        provider_registry_factory=registry_factory,
        spawn_authority_coordinator=spawn_coordinator,
        publication_coordinator=publication_coordinator,
    )
    return SimpleNamespace(
        fixture=fixture,
        prepared=prepared,
        execution_store=execution_store,
        provider=provider,
        registry=registry,
        registry_factory=registry_factory,
        spawn_coordinator=spawn_coordinator,
        orchestrator=orchestrator,
    )


def _reserve(stage_fixture):
    snapshot = stage_fixture.fixture.validation_store.snapshot(
        stage_fixture.prepared.request.candidate_id
    )
    assert snapshot is not None
    return stage_fixture.fixture.validation_store.reserve_source_replay(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=stage_fixture.prepared,
    ).reservation


def _resolved_case(stage_fixture, allocation):
    return next(
        item
        for item in stage_fixture.registry.resolve_all(stage_fixture.prepared)
        if item.materialized_case.request_case.execution_case_id
        == allocation.execution_case_id
    )


def test_stage_orchestrator_runs_and_publishes_one_typed_source_result(tmp_path):
    stage = _stage_fixture(tmp_path)

    snapshot = stage.orchestrator.run(stage.fixture.attempt)

    assert snapshot.state.promotion_state is ExpertPromotionState.VALIDATING
    assert snapshot.state.next_stage is ExpertValidationStage.AUTOMATED_REVIEW
    assert len(stage.provider.invocations) == 2
    assert stage.registry_factory.calls == 1
    assert isinstance(
        snapshot.accepted_stage_results[-1],
        ExpertSourceReplayStageResultRecord,
    )
    assert snapshot.accepted_stage_results[-1].outcome.value == "passed"

    replayed = stage.orchestrator.run(stage.fixture.attempt)

    assert replayed == snapshot
    assert len(stage.provider.invocations) == 2
    assert stage.registry_factory.calls == 1


def test_stage_orchestrator_resumes_allocated_and_received_journal_tails(tmp_path):
    allocated = _stage_fixture(tmp_path)
    allocated_reservation = _reserve(allocated)
    with allocated.execution_store.reservation_session(
        reservation=allocated_reservation,
        prepared_request=allocated.prepared,
    ) as session:
        original_allocation = (
            session.allocate_expected_leg().require_current_allocation(
                allocated.execution_store
            )
        )

    allocated_snapshot = allocated.orchestrator.run(allocated.fixture.attempt)

    assert allocated_snapshot.state.next_stage is (
        ExpertValidationStage.AUTOMATED_REVIEW
    )
    assert len(allocated.provider.invocations) == 2
    assert (
        allocated.provider.invocations[0].invocation_allocation == original_allocation
    )

    received = _stage_fixture(tmp_path)
    received_reservation = _reserve(received)
    with received.execution_store.reservation_session(
        reservation=received_reservation,
        prepared_request=received.prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        allocation = allocation_permit.require_current_allocation(
            received.execution_store
        )
        execution = received.spawn_coordinator.commit_spawn(
            prepared_request=received.prepared,
            reservation_id=received_reservation.reservation_id,
            invocation_permit=allocation_permit,
            resolved_case=_resolved_case(received, allocation),
        )
        session.record_result_received(execution.execute())
    first_invocation_id = received.provider.invocations[0].invocation_allocation

    received_snapshot = received.orchestrator.run(received.fixture.attempt)

    assert received_snapshot.state.next_stage is ExpertValidationStage.AUTOMATED_REVIEW
    assert len(received.provider.invocations) == 2
    assert received.provider.invocations[0].invocation_allocation == first_invocation_id


def test_stage_orchestrator_cleans_but_never_reexecutes_reopened_spawn(tmp_path):
    stage = _stage_fixture(tmp_path)
    reservation = _reserve(stage)
    with stage.execution_store.reservation_session(
        reservation=reservation,
        prepared_request=stage.prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        allocation = allocation_permit.require_current_allocation(stage.execution_store)
        stage.spawn_coordinator.commit_spawn(
            prepared_request=stage.prepared,
            reservation_id=reservation.reservation_id,
            invocation_permit=allocation_permit,
            resolved_case=_resolved_case(stage, allocation),
        )

    with pytest.raises(
        ExpertSourceReplayPermanentlyInterruptedError,
        match="permanently interrupted",
    ):
        stage.orchestrator.run(stage.fixture.attempt)
    with pytest.raises(
        ExpertSourceReplayPermanentlyInterruptedError,
        match="permanently interrupted",
    ):
        stage.orchestrator.run(stage.fixture.attempt)

    assert stage.provider.invocations == []
    assert len(stage.provider.interrupted_cleanup_handles) == 2
    assert (
        stage.provider.interrupted_cleanup_handles[0]
        == stage.provider.interrupted_cleanup_handles[1]
    )


def test_stage_orchestrator_returns_parent_invalidation_before_provider_work(
    tmp_path,
):
    stage = _stage_fixture(tmp_path)

    class _RotatingCurrentRelease:
        def __init__(self):
            self.calls = 0

        def current_release_id(self, scope_id):
            assert scope_id == "ml_ai"
            self.calls += 1
            if self.calls == 1:
                return stage.fixture.attempt.parent_release_id
            return content_id("expert-base-release", {"label": "advanced"})

    rotating_current = _RotatingCurrentRelease()
    stage.fixture.coordinator.current_release_provider = rotating_current
    stage.fixture.validation_store.reducer.current_release_provider = rotating_current

    snapshot = stage.orchestrator.run(stage.fixture.attempt)

    assert snapshot.state.promotion_state is ExpertPromotionState.FAILED
    assert snapshot.state.reason == "validation_parent_release_changed"
    assert stage.registry_factory.calls == 0
    assert stage.provider.invocations == []


def test_concurrent_stage_orchestrators_converge_without_duplicate_execution(
    tmp_path,
):
    stage = _stage_fixture(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshots = tuple(
            executor.map(
                stage.orchestrator.run,
                (stage.fixture.attempt, stage.fixture.attempt),
            )
        )

    assert snapshots[0] == snapshots[1]
    assert snapshots[0].state.next_stage is ExpertValidationStage.AUTOMATED_REVIEW
    assert len(stage.provider.invocations) == 2
    assert stage.registry_factory.calls == 1


def test_stage_orchestrator_finishes_complete_tail_without_provider_bootstrap(
    tmp_path,
):
    stage = _stage_fixture(tmp_path)
    reservation = _reserve(stage)
    with stage.execution_store.reservation_session(
        reservation=reservation,
        prepared_request=stage.prepared,
    ) as session:
        while len(session.events) < len(stage.prepared.request.cases) * 8:
            allocation_permit = session.allocate_expected_leg()
            allocation = allocation_permit.require_current_allocation(
                stage.execution_store
            )
            execution = stage.spawn_coordinator.commit_spawn(
                prepared_request=stage.prepared,
                reservation_id=reservation.reservation_id,
                invocation_permit=allocation_permit,
                resolved_case=_resolved_case(stage, allocation),
            )
            session.record_result_received(execution.execute())
            session.accept_received_result()
        session.completed_execution()

    stage.orchestrator.provider_registry_factory = lambda _prepared: pytest.fail(
        "complete journal requested provider bootstrap"
    )

    snapshot = stage.orchestrator.run(stage.fixture.attempt)

    assert snapshot.state.next_stage is ExpertValidationStage.AUTOMATED_REVIEW
    assert len(stage.provider.invocations) == 2


def test_stage_orchestrator_accepts_final_received_tail_without_provider_bootstrap(
    tmp_path,
):
    stage = _stage_fixture(tmp_path)
    reservation = _reserve(stage)
    with stage.execution_store.reservation_session(
        reservation=reservation,
        prepared_request=stage.prepared,
    ) as session:
        event_limit = len(stage.prepared.request.cases) * 8
        while len(session.events) < event_limit:
            allocation_permit = session.allocate_expected_leg()
            allocation = allocation_permit.require_current_allocation(
                stage.execution_store
            )
            execution = stage.spawn_coordinator.commit_spawn(
                prepared_request=stage.prepared,
                reservation_id=reservation.reservation_id,
                invocation_permit=allocation_permit,
                resolved_case=_resolved_case(stage, allocation),
            )
            session.record_result_received(execution.execute())
            if len(session.events) + 1 < event_limit:
                session.accept_received_result()
            else:
                break

    stage.orchestrator.provider_registry_factory = lambda _prepared: pytest.fail(
        "final received result requested provider bootstrap"
    )

    snapshot = stage.orchestrator.run(stage.fixture.attempt)

    assert snapshot.state.next_stage is ExpertValidationStage.AUTOMATED_REVIEW
    assert len(stage.provider.invocations) == 2


def test_reservation_replay_crosses_only_its_current_publication_head(tmp_path):
    stage = _stage_fixture(tmp_path)
    authorization = stage.fixture.validation_store.snapshot(
        stage.fixture.attempt.candidate_id
    )
    assert authorization is not None
    source_snapshot = stage.orchestrator.run(stage.fixture.attempt)

    replayed = stage.fixture.validation_store.reserve_source_replay(
        expected_transition_id=authorization.transition.transition_id,
        prepared_request=stage.prepared,
    )
    assert replayed.replayed is True

    stage.fixture.current_release_provider.release_id = content_id(
        "expert-base-release",
        {"label": "advanced-after-source-publication"},
    )
    invalidated = stage.fixture.validation_store.publish_parent_authority_invalidation(
        candidate_id=stage.fixture.attempt.candidate_id,
        expected_validation_state_id=source_snapshot.state.validation_state_id,
    ).snapshot

    with pytest.raises(ExpertValidationCompareAndSwapError, match="head changed"):
        stage.fixture.validation_store.reserve_source_replay(
            expected_transition_id=authorization.transition.transition_id,
            prepared_request=stage.prepared,
        )
    assert invalidated.state.reason == "validation_parent_release_changed"


def test_stage_orchestrator_rejects_mismatched_policy_wiring_before_run(tmp_path):
    stage = _stage_fixture(tmp_path)
    changed_policy = replace(
        stage.prepared.settings.policy,
        source_replay_score_comparison_tolerance=(
            stage.prepared.settings.policy.source_replay_score_comparison_tolerance / 2
        ),
    )
    alternate_execution_store = ExpertSourceReplayExecutionStore(
        (stage.fixture.validation_store.root / "alternate-source-replay").resolve(),
        stage.fixture.validation_store.root,
        stage.prepared.settings.policy,
    )

    with pytest.raises(ExpertSourceReplayStageError, match="validation authority"):
        ExpertSourceReplayStageOrchestrator(
            validation_store=stage.fixture.validation_store,
            preflight_coordinator=stage.fixture.coordinator,
            execution_store=alternate_execution_store,
            provider_registry_factory=stage.registry_factory,
            spawn_authority_coordinator=stage.spawn_coordinator,
            publication_coordinator=stage.orchestrator.publication_coordinator,
        )

    mismatched_execution_store = ExpertSourceReplayExecutionStore(
        stage.execution_store.root,
        stage.fixture.validation_store.root,
        changed_policy,
    )

    with pytest.raises(ExpertSourceReplayStageError, match="validation authority"):
        ExpertSourceReplayStageOrchestrator(
            validation_store=stage.fixture.validation_store,
            preflight_coordinator=stage.fixture.coordinator,
            execution_store=mismatched_execution_store,
            provider_registry_factory=stage.registry_factory,
            spawn_authority_coordinator=stage.spawn_coordinator,
            publication_coordinator=stage.orchestrator.publication_coordinator,
        )

    stage.fixture.coordinator.settings = replace(
        stage.fixture.settings,
        policy=changed_policy,
    )
    with pytest.raises(ExpertSourceReplayStageError, match="preflight"):
        ExpertSourceReplayStageOrchestrator(
            validation_store=stage.fixture.validation_store,
            preflight_coordinator=stage.fixture.coordinator,
            execution_store=stage.execution_store,
            provider_registry_factory=stage.registry_factory,
            spawn_authority_coordinator=stage.spawn_coordinator,
            publication_coordinator=stage.orchestrator.publication_coordinator,
        )


def test_stage_orchestrator_repairs_partial_reservation_layout_after_crash(tmp_path):
    stage = _stage_fixture(tmp_path)
    reservation = _reserve(stage)
    partial_root = stage.execution_store._reservation_path(reservation.reservation_id)
    partial_root.mkdir(mode=0o700)

    snapshot = stage.orchestrator.run(stage.fixture.attempt)

    assert snapshot.state.next_stage is ExpertValidationStage.AUTOMATED_REVIEW
    assert {path.name for path in partial_root.iterdir()} == {
        "events",
        "results",
        "staging",
    }
    assert len(stage.provider.invocations) == 2
