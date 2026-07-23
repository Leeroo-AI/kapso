"""Production composition root for one expert source-replay stage."""

from __future__ import annotations

from typing import Callable

from kapso.cross_run.contracts import (
    ExpertPromotionState,
    ExpertSourceReplayExecutionReservation,
    ExpertValidationAttempt,
    ExpertValidationStage,
)
from kapso.cross_run.expert.replay_authority import (
    ExpertSourceReplayFreshAuthorityCoordinator,
)
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayExecutionProviderRegistry,
    ResolvedExpertSourceReplayExecutionCase,
)
from kapso.cross_run.expert.replay_execution_store import (
    CompletedExpertSourceReplayExecution,
    ExpertSourceReplayExecutionStore,
    source_replay_execution_schedule,
)
from kapso.cross_run.expert.replay_publication import (
    ExpertSourceReplayDecisionPublicationCoordinator,
)
from kapso.cross_run.expert.replay_request import (
    ExpertSourceReplayPreflightCoordinator,
    PreparedExpertSourceReplayRequest,
)
from kapso.cross_run.expert.validation_snapshots import ExpertValidationSnapshot
from kapso.cross_run.expert.validation_store import ExpertValidationStore


class ExpertSourceReplayStageError(ValueError):
    """The source-replay stage cannot continue from its durable authority."""


class ExpertSourceReplayPermanentlyInterruptedError(ExpertSourceReplayStageError):
    """A durable spawn marker forbids re-executing its scientific invocation."""


SourceReplayProviderRegistryFactory = Callable[
    [PreparedExpertSourceReplayRequest],
    ExpertSourceReplayExecutionProviderRegistry,
]


class ExpertSourceReplayStageOrchestrator:
    """Run or resume one source-replay stage through its canonical authorities."""

    def __init__(
        self,
        *,
        validation_store: ExpertValidationStore,
        preflight_coordinator: ExpertSourceReplayPreflightCoordinator,
        execution_store: ExpertSourceReplayExecutionStore,
        provider_registry_factory: SourceReplayProviderRegistryFactory,
        spawn_authority_coordinator: ExpertSourceReplayFreshAuthorityCoordinator,
        publication_coordinator: ExpertSourceReplayDecisionPublicationCoordinator,
    ) -> None:
        if not isinstance(validation_store, ExpertValidationStore):
            raise ExpertSourceReplayStageError(
                "source replay orchestration requires its canonical validation store"
            )
        if (
            not isinstance(
                preflight_coordinator,
                ExpertSourceReplayPreflightCoordinator,
            )
            or preflight_coordinator.validation_authority is not validation_store
            or preflight_coordinator.settings != validation_store.settings
        ):
            raise ExpertSourceReplayStageError(
                "source replay preflight must use the orchestrator validation store"
            )
        if (
            not isinstance(execution_store, ExpertSourceReplayExecutionStore)
            or execution_store.trusted_root != validation_store.root
            or execution_store.root
            != ExpertSourceReplayExecutionStore.canonical_root(validation_store.root)
            or execution_store.policy_settings != validation_store.settings.policy
        ):
            raise ExpertSourceReplayStageError(
                "source replay execution store differs from validation authority"
            )
        if not callable(provider_registry_factory):
            raise ExpertSourceReplayStageError(
                "source replay orchestration requires a provider registry factory"
            )
        if (
            type(spawn_authority_coordinator)
            is not ExpertSourceReplayFreshAuthorityCoordinator
            or spawn_authority_coordinator.reservation_authority is not validation_store
            or spawn_authority_coordinator.execution_store is not execution_store
        ):
            raise ExpertSourceReplayStageError(
                "source replay spawn authority differs from orchestrator stores"
            )
        if (
            type(publication_coordinator)
            is not ExpertSourceReplayDecisionPublicationCoordinator
            or publication_coordinator.validation_store is not validation_store
            or publication_coordinator.execution_store is not execution_store
        ):
            raise ExpertSourceReplayStageError(
                "source replay publication authority differs from orchestrator stores"
            )
        self.validation_store = validation_store
        self.preflight_coordinator = preflight_coordinator
        self.execution_store = execution_store
        self.provider_registry_factory = provider_registry_factory
        self.spawn_authority_coordinator = spawn_authority_coordinator
        self.publication_coordinator = publication_coordinator

    def run(self, attempt: ExpertValidationAttempt) -> ExpertValidationSnapshot:
        """Run a fresh stage or resume its exact durable journal prefix."""

        if not isinstance(attempt, ExpertValidationAttempt):
            raise ExpertSourceReplayStageError(
                "source replay orchestration requires a validation attempt"
            )
        with self.execution_store.stage_run_lock(attempt.candidate_id):
            return self._run_locked(attempt)

    def _run_locked(
        self,
        attempt: ExpertValidationAttempt,
    ) -> ExpertValidationSnapshot:
        initial = self.validation_store.snapshot(attempt.candidate_id)
        if initial is None or initial.latest_attempt != attempt:
            raise ExpertSourceReplayStageError(
                "source replay attempt is not the current validation attempt"
            )
        if _source_replay_is_resolved(initial, attempt):
            return initial
        expected_transition_id = initial.transition.transition_id
        preflight = self.preflight_coordinator.build(attempt)
        if preflight.invalidated_state is not None:
            invalidated = self.validation_store.snapshot(attempt.candidate_id)
            if (
                invalidated is None
                or invalidated.latest_attempt != attempt
                or invalidated.state != preflight.invalidated_state
            ):
                raise ExpertSourceReplayStageError(
                    "source replay invalidation differs from validation authority"
                )
            return invalidated
        prepared = preflight.prepared_request
        if prepared is None:
            raise ExpertSourceReplayStageError(
                "source replay preflight produced no executable authority"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared.request,
            settings=prepared.settings,
            attempt=prepared.attempt,
            selection=prepared.selection,
            candidate=prepared.candidate,
            source_base=prepared.source_base,
            authorization_state=prepared.authorization_state,
            cases=prepared.cases,
        )
        if prepared.attempt != attempt:
            raise ExpertSourceReplayStageError(
                "source replay preflight returned another validation attempt"
            )
        reservation_commit = self.validation_store.existing_source_replay_reservation(
            expected_transition_id=expected_transition_id,
            prepared_request=prepared,
        )
        provider_registry = None
        resolved_by_case_id = None
        if reservation_commit is None or self._existing_tail_requires_provider(
            reservation_commit.reservation,
            prepared,
        ):
            provider_registry = self.provider_registry_factory(prepared)
            if not isinstance(
                provider_registry,
                ExpertSourceReplayExecutionProviderRegistry,
            ):
                raise ExpertSourceReplayStageError(
                    "source replay provider factory returned a noncanonical registry"
                )
            resolved_cases = provider_registry.resolve_all(prepared)
            resolved_by_case_id = _resolved_cases_by_id(prepared, resolved_cases)
        if reservation_commit is None:
            reservation_commit = self.validation_store.reserve_source_replay(
                expected_transition_id=expected_transition_id,
                prepared_request=prepared,
            )
        completed_execution = self._execute_or_resume(
            prepared=prepared,
            reservation=reservation_commit.reservation,
            provider_registry=provider_registry,
            resolved_by_case_id=resolved_by_case_id,
        )
        published = self.publication_coordinator.publish_completed(
            completed_execution=completed_execution,
            reservation=reservation_commit.reservation,
            prepared_request=prepared,
        )
        return published.snapshot

    def _existing_tail_requires_provider(
        self,
        reservation: ExpertSourceReplayExecutionReservation,
        prepared: PreparedExpertSourceReplayRequest,
    ) -> bool:
        events = self.execution_store.existing_reservation_events(
            reservation=reservation,
            prepared_request=prepared,
        )
        if events is None:
            return True
        event_limit = 4 * len(
            source_replay_execution_schedule(reservation, prepared.request)
        )
        return len(events) != event_limit and not (
            len(events) + 1 == event_limit and len(events) % 4 == 3
        )

    def _execute_or_resume(
        self,
        *,
        prepared: PreparedExpertSourceReplayRequest,
        reservation: ExpertSourceReplayExecutionReservation,
        provider_registry: ExpertSourceReplayExecutionProviderRegistry | None,
        resolved_by_case_id: dict[str, ResolvedExpertSourceReplayExecutionCase] | None,
    ) -> CompletedExpertSourceReplayExecution:
        schedule = source_replay_execution_schedule(reservation, prepared.request)
        with self.execution_store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ) as session:
            while len(session.events) < len(schedule) * 4:
                phase = len(session.events) % 4
                if phase == 2:
                    if provider_registry is None:
                        raise ExpertSourceReplayStageError(
                            "interrupted source replay lacks its cleanup provider"
                        )
                    provider_handle = session.cleanup_interrupted_spawn(
                        provider_registry
                    )
                    raise ExpertSourceReplayPermanentlyInterruptedError(
                        "source replay invocation is permanently interrupted after "
                        f"spawn commit: {provider_handle.provider_handle_id}"
                    )
                if phase == 3:
                    session.accept_received_result()
                    continue
                allocation_permit = session.allocate_expected_leg()
                allocation = allocation_permit.require_current_allocation(
                    self.execution_store
                )
                if resolved_by_case_id is None:
                    raise ExpertSourceReplayStageError(
                        "executable source replay lacks provider resolution"
                    )
                resolved_case = resolved_by_case_id.get(allocation.execution_case_id)
                if resolved_case is None:
                    raise ExpertSourceReplayStageError(
                        "source replay allocation has no pre-resolved provider"
                    )
                execution = self.spawn_authority_coordinator.commit_spawn(
                    prepared_request=prepared,
                    reservation_id=reservation.reservation_id,
                    invocation_permit=allocation_permit,
                    resolved_case=resolved_case,
                )
                completion = execution.execute()
                session.record_result_received(completion)
                session.accept_received_result()
            return session.completed_execution()


def _resolved_cases_by_id(
    prepared: PreparedExpertSourceReplayRequest,
    resolved_cases: tuple[ResolvedExpertSourceReplayExecutionCase, ...],
) -> dict[str, ResolvedExpertSourceReplayExecutionCase]:
    resolved_by_case_id = {
        item.materialized_case.request_case.execution_case_id: item
        for item in resolved_cases
    }
    expected_case_ids = {item.request_case.execution_case_id for item in prepared.cases}
    if (
        len(resolved_cases) != len(resolved_by_case_id)
        or set(resolved_by_case_id) != expected_case_ids
    ):
        raise ExpertSourceReplayStageError(
            "source replay provider resolution does not cover every case exactly"
        )
    return resolved_by_case_id


def _source_replay_is_resolved(
    snapshot: ExpertValidationSnapshot,
    attempt: ExpertValidationAttempt,
) -> bool:
    if snapshot.latest_attempt != attempt:
        return False
    if (
        snapshot.state.promotion_state is ExpertPromotionState.FAILED
        and snapshot.state.reason == "validation_current_release_authority_changed"
    ):
        return True
    if any(
        item.stage is ExpertValidationStage.SOURCE_RUN_REPLAY
        for item in snapshot.state.accepted_stage_results
    ):
        return True
    result_id = snapshot.transition.transition_stage_result_record_id
    return (
        snapshot.state.promotion_state is ExpertPromotionState.FAILED
        and result_id is not None
        and result_id.startswith("expert-source-replay-stage-result:sha256:")
    )
