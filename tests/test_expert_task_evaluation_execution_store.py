from concurrent.futures import ThreadPoolExecutor

import pytest

import kapso.cross_run.expert.task_evaluation_execution_store as store_module
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalStoreError,
)
from kapso.cross_run.expert.task_evaluation_authority import (
    TaskEvaluationFreshAuthorityCoordinator,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderRegistry,
    TaskEvaluationProviderCompletion,
    project_prepared_task_evaluation_cases,
)
from kapso.cross_run.expert.task_evaluation_execution_journal import (
    TaskEvaluationExecutionJournalEventKind,
    task_evaluation_execution_schedule,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    CompletedTaskEvaluationExecution,
    ExpertTaskEvaluationExecutionStore,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TaskEvaluatorFingerprintResult,
    TaskEvaluatorResult,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessResult,
)
from test_expert_task_evaluation_authority import _denylist
from test_expert_task_evaluation_reservation import _parent_prepared


class _CurrentAuthority:
    def __init__(self, observation):
        self.observation = observation
        self.calls = []

    def observe_task_evaluation_current(self, scope_id):
        self.calls.append(scope_id)
        assert scope_id == self.observation.scope_id
        return self.observation


class _AdapterAuthority:
    def __init__(self, prepared):
        self.adapters = {
            (
                adapter.manifest.task_adapter_manifest_id,
                adapter.verification_receipt.verification_receipt_id,
            ): adapter
            for adapter in prepared.adapters
        }
        self.calls = []

    def resolve_exact(
        self,
        *,
        task_adapter_manifest_id,
        verification_receipt_id,
    ):
        key = task_adapter_manifest_id, verification_receipt_id
        self.calls.append(key)
        return self.adapters[key]


class _DenylistAuthority:
    def __init__(self, prepared):
        self.prepared = prepared
        self.calls = []

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.calls.append((scope_id, scope_contract_id, checked_subject_ids))
        return _denylist(self.prepared, checked_subject_ids)


class _Provider:
    def __init__(
        self,
        trusted_root,
        dispatch_key,
        *,
        result_payload=None,
        outcome=BoundedProcessOutcome.COMPLETED,
        returncode=0,
    ):
        self.trusted_root = trusted_root
        self.dispatch_key = dispatch_key
        self.result_payload = result_payload
        self.outcome = outcome
        self.returncode = returncode
        self.support_calls = []
        self.execution_calls = []
        self.cleanup_calls = []

    def require_supported_execution(self, requirements):
        assert requirements.dispatch_key == self.dispatch_key
        self.support_calls.append(requirements)

    def execute_leg(self, invocation):
        self.execution_calls.append(invocation)
        requirements = invocation.execution_requirements
        payload = self.result_payload
        if payload is None and self.outcome is BoundedProcessOutcome.COMPLETED:
            payload = _result(invocation.task_evaluator_request).to_json_bytes()
        stdout_bytes_observed = (
            requirements.stdout_byte_limit + 17
            if self.outcome is BoundedProcessOutcome.STDOUT_LIMIT_EXCEEDED
            else 0
        )
        stderr_bytes_observed = (
            requirements.stderr_byte_limit + 17
            if self.outcome is BoundedProcessOutcome.STDERR_LIMIT_EXCEEDED
            else 0
        )
        return TaskEvaluationProviderCompletion(
            provider_handle_id=invocation.provider_handle.provider_handle_id,
            process_result=BoundedProcessResult(
                request=BoundedProcessRequest(
                    argv=("true",),
                    trusted_root=self.trusted_root,
                    cwd=self.trusted_root,
                    timeout_seconds=requirements.leg_wall_time_limit_seconds,
                    cleanup_timeout_seconds=requirements.termination_grace_seconds,
                    stdout_byte_limit=requirements.stdout_byte_limit,
                    stderr_byte_limit=requirements.stderr_byte_limit,
                    environment={},
                ),
                outcome=self.outcome,
                returncode=self.returncode,
                stdout=b"",
                stderr=b"",
                stdout_bytes_observed=stdout_bytes_observed,
                stderr_bytes_observed=stderr_bytes_observed,
                duration_seconds=0.5,
            ),
            result_payload=payload,
        )

    def cleanup_interrupted(self, provider_handle):
        self.cleanup_calls.append(provider_handle)


def _result(task_evaluator_request):
    return TaskEvaluatorResult(
        protocol_version=task_evaluator_request.protocol_version,
        opaque_invocation_id=task_evaluator_request.opaque_invocation_id,
        fingerprint_results=tuple(
            TaskEvaluatorFingerprintResult(
                evaluation_fingerprint_id=fingerprint.evaluation_fingerprint_id,
                aggregate_value=1.0,
                replicate_values={
                    replicate_id: 1.0
                    for replicate_id in fingerprint.seed_or_replicate_ids
                },
            )
            for fingerprint in task_evaluator_request.evaluation_fingerprints
        ),
    )


def _fixture(tmp_path, monkeypatch, *, provider_kwargs=None):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    reservation_snapshot = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    ).reservation
    store = ExpertTaskEvaluationExecutionStore(
        ExpertTaskEvaluationExecutionStore.canonical_root(
            validation_store.root
        ).resolve(),
        validation_store.root,
        prepared.plan_join.settings.policy,
    )
    executable_cases = project_prepared_task_evaluation_cases(prepared)
    keys = tuple(
        sorted(
            {case.provider_key for case in executable_cases},
            key=lambda key: key.identity,
        )
    )
    providers = tuple(
        _Provider(
            validation_store.root,
            key,
            **({} if provider_kwargs is None else provider_kwargs),
        )
        for key in keys
    )
    registry = TaskEvaluationExecutionProviderRegistry(prepared, providers)
    current_authority = _CurrentAuthority(prepared.current_release_observation)
    adapter_authority = _AdapterAuthority(prepared)
    denylist_authority = _DenylistAuthority(prepared)
    coordinator = TaskEvaluationFreshAuthorityCoordinator(
        reservation_authority=validation_store,
        execution_store=store,
        current_release_authority=current_authority,
        task_adapter_authority=adapter_authority,
        security_denylist_authority=denylist_authority,
    )
    return (
        prepared,
        reservation_snapshot,
        store,
        registry,
        providers,
        coordinator,
    )


def _commit_spawn(
    session,
    prepared,
    reservation_snapshot,
    registry,
    coordinator,
):
    allocation_permit = session.allocate_expected_leg()
    return coordinator.commit_spawn(
        prepared_request=prepared,
        reservation_id=reservation_snapshot.reservation.reservation_id,
        invocation_permit=allocation_permit,
        provider_registry=registry,
    )


def _event_path(store, reservation_snapshot, event_number):
    return store._filesystem.event_path(
        store._reservation_digest(reservation_snapshot.reservation.reservation_id),
        event_number,
    )


def test_allocation_restarts_with_the_same_nonce_and_schedule_position(
    tmp_path,
    monkeypatch,
):
    prepared, reservation, store, *_runtime = _fixture(tmp_path, monkeypatch)
    foreign_store = ExpertTaskEvaluationExecutionStore(
        store.root,
        store.trusted_root,
        store.policy_settings,
    )
    nonce_calls = []

    def fixed_nonce():
        nonce_calls.append(True)
        return "0123456789abcdef0123456789abcdef"

    monkeypatch.setattr(store_module, "_new_invocation_nonce", fixed_nonce)
    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        allocation = permit.require_current_allocation(store)
        assert session.allocate_expected_leg() is permit
        with pytest.raises(ExecutionJournalStoreError, match="canonical live store"):
            permit.require_current_allocation(foreign_store)

    with pytest.raises(ExecutionJournalStoreError, match="closed"):
        permit.require_current_allocation(store)

    with foreign_store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as recovered_session:
        recovered = recovered_session.allocate_expected_leg()
        assert recovered.require_current_allocation(foreign_store) == allocation

    assert nonce_calls == [True]
    assert (
        allocation.evaluation_case_id,
        allocation.evaluation_leg_id,
    ) == task_evaluation_execution_schedule(reservation, prepared)[0]


def test_complete_execution_is_provider_once_and_mints_exact_capability(
    tmp_path,
    monkeypatch,
):
    prepared, reservation, store, registry, providers, coordinator = _fixture(
        tmp_path,
        monkeypatch,
    )
    schedule = task_evaluation_execution_schedule(reservation, prepared)

    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        while len(session.events) < 4 * len(schedule):
            spawn = _commit_spawn(
                session,
                prepared,
                reservation,
                registry,
                coordinator,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = tuple(executor.submit(spawn.execute) for _ in range(2))
            exceptions = tuple(future.exception() for future in futures)
            assert sum(exception is None for exception in exceptions) == 1
            assert (
                sum(
                    isinstance(exception, ExecutionJournalStoreError)
                    for exception in exceptions
                )
                == 1
            )
            completion = next(
                future.result()
                for future, exception in zip(futures, exceptions, strict=True)
                if exception is None
            )
            session.record_result_received(completion)
            session.accept_received_result()
        completed = session.completed_execution()

    assert type(completed) is CompletedTaskEvaluationExecution
    assert completed.require_exact(store, reservation, prepared) == completed.events
    assert len(completed.events) == 4 * len(schedule)
    assert sum(len(provider.execution_calls) for provider in providers) == len(schedule)
    foreign_store = ExpertTaskEvaluationExecutionStore(
        store.root,
        store.trusted_root,
        store.policy_settings,
    )
    with pytest.raises(ExecutionJournalStoreError, match="journal authority"):
        completed.require_exact(foreign_store, reservation, prepared)
    with pytest.raises(ExecutionJournalStoreError, match="closed"):
        spawn.execute()


def test_received_result_restarts_and_accepts_without_provider_work(
    tmp_path,
    monkeypatch,
):
    prepared, reservation, store, registry, providers, coordinator = _fixture(
        tmp_path,
        monkeypatch,
    )
    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        spawn = _commit_spawn(
            session,
            prepared,
            reservation,
            registry,
            coordinator,
        )
        received = session.record_result_received(spawn.execute())

    call_count = sum(len(provider.execution_calls) for provider in providers)
    recovered_store = ExpertTaskEvaluationExecutionStore(
        store.root,
        store.trusted_root,
        store.policy_settings,
    )
    with recovered_store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as recovered_session:
        result = recovered_session.accept_received_result()
        assert recovered_session.events[-2] == received
        assert recovered_session.events[-1].task_evaluator_result == result

    assert sum(len(provider.execution_calls) for provider in providers) == call_count


def test_malformed_result_is_durable_reopenable_and_nonadvancing(
    tmp_path,
    monkeypatch,
):
    prepared, reservation, store, registry, providers, coordinator = _fixture(
        tmp_path,
        monkeypatch,
        provider_kwargs={"result_payload": b"malformed evaluator result"},
    )
    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        spawn = _commit_spawn(
            session,
            prepared,
            reservation,
            registry,
            coordinator,
        )
        session.record_result_received(spawn.execute())

    recovered_store = ExpertTaskEvaluationExecutionStore(
        store.root,
        store.trusted_root,
        store.policy_settings,
    )
    with recovered_store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as recovered_session:
        assert len(recovered_session.events) == 3
        with pytest.raises(ValueError):
            recovered_session.accept_received_result()
        with pytest.raises(ExecutionJournalStoreError, match="must be accepted"):
            recovered_session.allocate_expected_leg()

    assert sum(len(provider.execution_calls) for provider in providers) == 1


def test_technical_result_without_blob_is_durable_and_terminal(
    tmp_path,
    monkeypatch,
):
    prepared, reservation, store, registry, providers, coordinator = _fixture(
        tmp_path,
        monkeypatch,
        provider_kwargs={
            "outcome": BoundedProcessOutcome.TIMED_OUT,
            "returncode": -15,
        },
    )
    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        spawn = _commit_spawn(
            session,
            prepared,
            reservation,
            registry,
            coordinator,
        )
        received = session.record_result_received(spawn.execute())

    assert received.result_blob is None
    assert received.process_observation.outcome is BoundedProcessOutcome.TIMED_OUT
    recovered_store = ExpertTaskEvaluationExecutionStore(
        store.root,
        store.trusted_root,
        store.policy_settings,
    )
    with recovered_store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as recovered_session:
        with pytest.raises(ExecutionJournalStoreError, match="no acceptable result"):
            recovered_session.accept_received_result()
        assert len(recovered_session.events) == 3

    assert sum(len(provider.execution_calls) for provider in providers) == 1


@pytest.mark.parametrize(
    ("outcome", "observation_field", "limit_field"),
    (
        (
            BoundedProcessOutcome.STDOUT_LIMIT_EXCEEDED,
            "stdout_bytes_observed",
            "stdout_byte_limit",
        ),
        (
            BoundedProcessOutcome.STDERR_LIMIT_EXCEEDED,
            "stderr_bytes_observed",
            "stderr_byte_limit",
        ),
    ),
)
def test_stream_limit_result_is_saturated_durable_and_terminal(
    tmp_path,
    monkeypatch,
    outcome,
    observation_field,
    limit_field,
):
    prepared, reservation, store, registry, providers, coordinator = _fixture(
        tmp_path,
        monkeypatch,
        provider_kwargs={"outcome": outcome, "returncode": -15},
    )
    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        spawn = _commit_spawn(
            session,
            prepared,
            reservation,
            registry,
            coordinator,
        )
        received = session.record_result_received(spawn.execute())

    compute = project_prepared_task_evaluation_cases(prepared)[0].compute_binding
    assert received.result_blob is None
    assert received.process_observation.outcome is outcome
    assert getattr(received.process_observation, observation_field) == (
        getattr(compute, limit_field) + 1
    )
    recovered_store = ExpertTaskEvaluationExecutionStore(
        store.root,
        store.trusted_root,
        store.policy_settings,
    )
    with recovered_store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as recovered_session:
        assert recovered_session.events[-1] == received
        with pytest.raises(ExecutionJournalStoreError, match="no acceptable result"):
            recovered_session.accept_received_result()

    assert sum(len(provider.execution_calls) for provider in providers) == 1


def test_reopened_spawn_never_executes_and_ignores_bounded_orphan_result(
    tmp_path,
    monkeypatch,
):
    prepared, reservation, store, registry, providers, coordinator = _fixture(
        tmp_path,
        monkeypatch,
    )
    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        spawn = _commit_spawn(
            session,
            prepared,
            reservation,
            registry,
            coordinator,
        )
        store._filesystem.publish_result(
            store._reservation_digest(reservation.reservation.reservation_id),
            b"unreferenced crash result",
        )

    recovered_store = ExpertTaskEvaluationExecutionStore(
        store.root,
        store.trusted_root,
        store.policy_settings,
    )
    with recovered_store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as recovered_session:
        assert recovered_session.events[-1].event_kind is (
            TaskEvaluationExecutionJournalEventKind.SPAWN_COMMITTED
        )
        with pytest.raises(ExecutionJournalStoreError, match="interrupted"):
            recovered_session.allocate_expected_leg()
        first_handle = recovered_session.cleanup_interrupted_spawn(registry)
        second_handle = recovered_session.cleanup_interrupted_spawn(registry)

    assert first_handle == second_handle
    assert sum(len(provider.execution_calls) for provider in providers) == 0
    assert sum(len(provider.cleanup_calls) for provider in providers) == 2
    with pytest.raises(ExecutionJournalStoreError, match="closed"):
        spawn.execute()


def test_post_rename_failure_poison_requires_reopen_and_uses_durable_event(
    tmp_path,
    monkeypatch,
):
    prepared, reservation, store, *_runtime = _fixture(tmp_path, monkeypatch)
    reservation_digest = store._reservation_digest(
        reservation.reservation.reservation_id
    )
    events_path = store._filesystem.events_path(reservation_digest)
    original_fsync = store._filesystem._fsync_directory

    def fail_after_event_rename(path):
        original_fsync(path)
        if path == events_path:
            raise OSError("simulated event-directory fsync failure")

    monkeypatch.setattr(
        store._filesystem,
        "_fsync_directory",
        fail_after_event_rename,
    )
    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        with pytest.raises(OSError, match="simulated"):
            session.allocate_expected_leg()
        with pytest.raises(ExecutionJournalStoreError, match="must reopen"):
            _ = session.events

    monkeypatch.setattr(store._filesystem, "_fsync_directory", original_fsync)
    recovered_store = ExpertTaskEvaluationExecutionStore(
        store.root,
        store.trusted_root,
        store.policy_settings,
    )
    with recovered_store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as recovered_session:
        assert len(recovered_session.events) == 1
        recovered_session.allocate_expected_leg()


def test_noncanonical_durable_event_fails_before_session_authority(
    tmp_path,
    monkeypatch,
):
    prepared, reservation, store, *_runtime = _fixture(tmp_path, monkeypatch)
    with store.reservation_session(
        reservation_snapshot=reservation,
        prepared_request=prepared,
    ) as session:
        session.allocate_expected_leg()

    event_path = _event_path(store, reservation, 1)
    payload = event_path.read_bytes()
    event_path.chmod(0o600)
    event_path.write_bytes(payload + b"\n")
    event_path.chmod(0o400)

    with pytest.raises(ExecutionJournalStoreError, match="not canonical"):
        with store.reservation_session(
            reservation_snapshot=reservation,
            prepared_request=prepared,
        ):
            raise AssertionError("corrupt journal cannot yield a session")
