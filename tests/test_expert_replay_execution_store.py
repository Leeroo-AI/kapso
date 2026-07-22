from __future__ import annotations

import fcntl
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from multiprocessing import get_context

import pytest

import kapso.cross_run.expert.replay_execution_store as execution_store_module
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import expert_source_replay_matched_compute_digest
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayExecutionProviderRegistry,
    ExpertSourceReplayProviderCompletion,
    expert_source_replay_execution_provider_key,
    source_replay_provider_execution_handle,
)
from kapso.cross_run.expert.replay_authority import (
    ExpertSourceReplayFreshAuthorityCoordinator,
    SourceReplayCurrentReleaseObservation,
    SourceReplaySecurityDenylistObservation,
)
from kapso.cross_run.expert.replay_execution_store import (
    ExpertSourceReplayExecutionStore,
    ExpertSourceReplayExecutionStoreError,
    SourceReplayExecutionJournalEvent,
    SourceReplaySealedLegCompletion,
    source_replay_execution_schedule,
)
from kapso.cross_run.expert.replay_protocol import TaskEvaluatorInvocationAllocation
from kapso.cross_run.expert.replay_protocol import (
    TaskEvaluatorFingerprintResult,
    TaskEvaluatorResult,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessResult,
)
from test_expert_source_replay_request import _prepared, _request_fixture


def _journal_fixture(tmp_path):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    snapshot = fixture.validation_store.snapshot(prepared.request.candidate_id)
    assert snapshot is not None
    committed = fixture.validation_store.reserve_source_replay(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    store = ExpertSourceReplayExecutionStore(
        (fixture.validation_store.root / "source-replay-executions").resolve(),
        fixture.validation_store.root,
        prepared.settings.policy,
    )
    return fixture, prepared, committed.reservation, store


def _allocate_in_process(store, reservation, prepared, result_queue):
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation = session.allocate_expected_leg().require_current_allocation(store)
        result_queue.put(allocation.to_json_bytes())


def _reject_inherited_execution_in_process(execution, result_queue):
    with pytest.raises(
        ExpertSourceReplayExecutionStoreError,
        match="creator process",
    ):
        execution.execute()
    result_queue.put("rejected")


def _remint(record, **changes):
    values = {field.name: getattr(record, field.name) for field in fields(record)}
    values.pop(record.IDENTITY_FIELD)
    values.update(changes)
    return type(record).mint(**values)


def _replace_published_event(store, reservation_id, event):
    event_path = store._event_path(reservation_id, event)
    event_path.chmod(0o600)
    event_path.write_bytes(event.to_json_bytes())
    event_path.chmod(0o400)


class _CurrentReleaseAuthority:
    def __init__(self, observation):
        self.observation = observation

    def current_release_observation(self, scope_id):
        assert scope_id == self.observation.scope_id
        return self.observation


class _SecurityDenylistAuthority:
    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        return SourceReplaySecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=tree_or_blob_digest(b"scope binding"),
            snapshot_id=content_id("security-denylist-snapshot", {"generation": 1}),
            generation=1,
            publication_id=content_id(
                "github-publication",
                {"security_generation": 1},
            ),
            repository_full_name="Leeroo-AI/kapso-security",
            repository_node_id="security_repo_node",
            pointer_digest=tree_or_blob_digest(b"security CURRENT"),
            authority_commit_sha="b" * 40,
            release_attestation_ref="attestations/security-denylist",
            checked_subject_ids=checked_subject_ids,
            denied_subject_ids=(),
        )


def _coordinator(fixture, prepared, store):
    parent = prepared.parent.release_manifest
    current = SourceReplayCurrentReleaseObservation.mint(
        scope_id=parent.scope_id,
        release_id=parent.release_id,
        publication_id=content_id(
            "github-publication",
            {"release_id": parent.release_id},
        ),
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repo_node",
        current_pointer_digest=tree_or_blob_digest(b"CURRENT"),
        current_pointer_commit_sha="a" * 40,
        validation_closure_ids=(),
    )
    return ExpertSourceReplayFreshAuthorityCoordinator(
        fixture.validation_store,
        store,
        _CurrentReleaseAuthority(current),
        fixture.adapter_provider,
        _SecurityDenylistAuthority(),
    )


def _commit_spawn(
    fixture,
    prepared,
    reservation,
    store,
    permit,
    provider=None,
):
    allocation = permit.require_current_allocation(store)
    materialized_case = next(
        case
        for case in prepared.cases
        if case.request_case.execution_case_id == allocation.execution_case_id
    )
    execution_provider = provider or _MatchedLegProvider(
        fixture.validation_store.root,
        expert_source_replay_execution_provider_key(materialized_case),
    )
    resolved_case = next(
        resolved
        for resolved in ExpertSourceReplayExecutionProviderRegistry(
            (execution_provider,)
        ).resolve_all(prepared)
        if resolved.materialized_case == materialized_case
    )
    execution = _coordinator(fixture, prepared, store).commit_spawn(
        prepared_request=prepared,
        reservation_id=reservation.reservation_id,
        invocation_permit=permit,
        resolved_case=resolved_case,
    )
    return execution, execution_provider


def _result_payload(task_evaluator_request):
    result = TaskEvaluatorResult(
        protocol_version=task_evaluator_request.protocol_version,
        opaque_invocation_id=task_evaluator_request.opaque_invocation_id,
        fingerprint_results=tuple(
            TaskEvaluatorFingerprintResult(
                evaluation_fingerprint_id=(fingerprint.evaluation_fingerprint_id),
                aggregate_value=1.0,
                replicate_values={
                    replicate_id: 1.0
                    for replicate_id in fingerprint.seed_or_replicate_ids
                },
            )
            for fingerprint in task_evaluator_request.evaluation_fingerprints
        ),
    )
    return result.to_json_bytes(), result


def _process_result(
    tmp_path,
    outcome=BoundedProcessOutcome.COMPLETED,
    returncode=0,
    stdout_bytes_observed=0,
    stderr_bytes_observed=0,
    compute_binding=None,
):
    request = BoundedProcessRequest(
        argv=("true",),
        trusted_root=tmp_path.resolve(),
        cwd=tmp_path.resolve(),
        timeout_seconds=(
            1
            if compute_binding is None
            else compute_binding.leg_wall_time_limit_seconds
        ),
        cleanup_timeout_seconds=(
            1 if compute_binding is None else compute_binding.termination_grace_seconds
        ),
        stdout_byte_limit=(
            1 if compute_binding is None else compute_binding.stdout_byte_limit
        ),
        stderr_byte_limit=(
            1 if compute_binding is None else compute_binding.stderr_byte_limit
        ),
        environment={},
    )
    return BoundedProcessResult(
        request=request,
        outcome=outcome,
        returncode=returncode,
        stdout=b"",
        stderr=b"",
        stdout_bytes_observed=stdout_bytes_observed,
        stderr_bytes_observed=stderr_bytes_observed,
        duration_seconds=0.5,
    )


_VALID_RESULT_PAYLOAD = object()


class _MatchedLegProvider:
    def __init__(
        self,
        trusted_root,
        dispatch_key,
        *,
        outcome=BoundedProcessOutcome.COMPLETED,
        returncode=0,
        result_payload=_VALID_RESULT_PAYLOAD,
        stdout_bytes_observed=0,
        stderr_bytes_observed=0,
    ):
        self.trusted_root = trusted_root
        self.dispatch_key = dispatch_key
        self.outcome = outcome
        self.returncode = returncode
        self.result_payload = result_payload
        self.stdout_bytes_observed = stdout_bytes_observed
        self.stderr_bytes_observed = stderr_bytes_observed
        self.invocations = []

    def execute_leg(self, invocation):
        self.invocations.append(invocation)
        payload = (
            _result_payload(invocation.task_evaluator_request)[0]
            if self.result_payload is _VALID_RESULT_PAYLOAD
            else self.result_payload
        )
        return ExpertSourceReplayProviderCompletion(
            provider_handle_id=invocation.provider_handle.provider_handle_id,
            process_result=_process_result(
                self.trusted_root,
                outcome=self.outcome,
                returncode=self.returncode,
                stdout_bytes_observed=self.stdout_bytes_observed,
                stderr_bytes_observed=self.stderr_bytes_observed,
                compute_binding=(
                    invocation.materialized_case.request_case.compute_binding
                ),
            ),
            result_payload=payload,
        )


class _FailingMatchedLegProvider(_MatchedLegProvider):
    def execute_leg(self, invocation):
        self.invocations.append(invocation)
        raise RuntimeError("provider execution failed")


class _MutatingMatchedLegProvider(_MatchedLegProvider):
    def execute_leg(self, invocation):
        completion = super().execute_leg(invocation)
        self.dispatch_key = replace(
            self.dispatch_key,
            execution_provider_version=(
                f"{self.dispatch_key.execution_provider_version}.changed"
            ),
        )
        return completion


class _ProcessCountingMatchedLegProvider(_MatchedLegProvider):
    def __init__(self, *args, process_call_count, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_call_count = process_call_count

    def execute_leg(self, invocation):
        with self.process_call_count.get_lock():
            self.process_call_count.value += 1
        return super().execute_leg(invocation)


def _two_case_authority(prepared, reservation):
    first_case = prepared.request.cases[0]
    second_episode_id = content_id(
        "transfer-episode",
        {"fixture": "execution-journal-second-case"},
    )
    second_compute_binding = _remint(
        first_case.compute_binding,
        leg_order=tuple(reversed(first_case.compute_binding.leg_order)),
    )
    second_dependencies = set(first_case.exact_dependency_ids)
    second_dependencies.remove(first_case.episode_id)
    second_dependencies.remove(first_case.compute_binding.compute_binding_id)
    second_dependencies.update(
        {second_episode_id, second_compute_binding.compute_binding_id}
    )
    second_case = _remint(
        first_case,
        episode_id=second_episode_id,
        compute_binding=second_compute_binding,
        matched_compute_binding_digest=expert_source_replay_matched_compute_digest(
            bundle_lineage_ids=first_case.bundle_lineage_ids,
            projection_manifest_id=first_case.projection_manifest_id,
            episode_id=second_episode_id,
            source_execution_revision=first_case.source_execution_revision,
            source_evaluation_fingerprint_ids=(
                first_case.source_evaluation_fingerprint_ids
            ),
            source_score_of_record_fingerprint_id=(
                first_case.source_score_of_record_fingerprint_id
            ),
            task_context_binding_id=first_case.task_context_binding_id,
            context_materialization_receipt_id=(
                first_case.context_materialization_receipt_id
            ),
            starting_artifact_content_ids=first_case.starting_artifact_content_ids,
            task_adapter_manifest_id=first_case.task_adapter_manifest_id,
            verification_receipt_id=first_case.verification_receipt_id,
            task_adapter_source_tree_hash=first_case.task_adapter_source_tree_hash,
            task_evaluator_digest=first_case.task_evaluator_digest,
            task_adapter_runtime_digest=first_case.task_adapter_runtime_digest,
            task_adapter_context_binding_digest=(
                first_case.task_adapter_context_binding_digest
            ),
            compute_binding_id=second_compute_binding.compute_binding_id,
        ),
        exact_dependency_ids=tuple(sorted(second_dependencies)),
    )
    cases = tuple(sorted((first_case, second_case), key=lambda case: case.episode_id))
    request_dependencies = set(prepared.request.exact_dependency_ids)
    request_dependencies.update(
        {
            second_case.execution_case_id,
            *second_case.exact_dependency_ids,
        }
    )
    request = _remint(
        prepared.request,
        cases=cases,
        exact_dependency_ids=tuple(sorted(request_dependencies)),
    )
    reservation_dependencies = set(reservation.exact_dependency_ids)
    reservation_dependencies.remove(reservation.execution_request_id)
    reservation_dependencies.add(request.execution_request_id)
    updated_reservation = _remint(
        reservation,
        execution_request_id=request.execution_request_id,
        exact_dependency_ids=tuple(sorted(reservation_dependencies)),
    )
    return updated_reservation, request


def test_allocation_is_create_only_and_replays_exactly_after_restart(
    tmp_path,
    monkeypatch,
):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    nonce_calls = []

    def fixed_nonce():
        nonce_calls.append(True)
        return "0123456789abcdef0123456789abcdef"

    monkeypatch.setattr(execution_store_module, "_new_invocation_nonce", fixed_nonce)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        assert session.allocate_expected_leg() is permit
        allocation = permit.require_current_allocation(store)
        events = session.events

    recovered_store = ExpertSourceReplayExecutionStore(
        store.root,
        store.trusted_root,
        prepared.settings.policy,
    )
    with recovered_store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered_session:
        recovered_permit = recovered_session.allocate_expected_leg()
        recovered = recovered_permit.require_current_allocation(recovered_store)

    schedule = source_replay_execution_schedule(reservation, prepared.request)
    assert nonce_calls == [True]
    assert recovered == allocation
    assert recovered_permit is not permit
    assert (allocation.execution_case_id, allocation.execution_leg_id) == schedule[0]
    assert len(events) == 1
    assert events[0].invocation_allocation == allocation
    assert (
        events[0].to_json_bytes()
        == type(events[0]).from_json_bytes(events[0].to_json_bytes()).to_json_bytes()
    )


def test_schedule_is_derived_from_the_request_compute_binding(tmp_path):
    _, prepared, reservation, _ = _journal_fixture(tmp_path)
    reservation, request = _two_case_authority(prepared, reservation)

    expected_schedule = []
    for case in request.cases:
        legs_by_kind = {
            case.control_leg.kind: case.control_leg.execution_leg_id,
            case.candidate_leg.kind: case.candidate_leg.execution_leg_id,
        }
        expected_schedule.extend(
            (case.execution_case_id, legs_by_kind[leg_kind])
            for leg_kind in case.compute_binding.leg_order
        )

    assert source_replay_execution_schedule(
        reservation,
        request,
    ) == tuple(expected_schedule)


def test_event_file_is_private_canonical_and_session_cannot_escape_its_lock(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        event = session.events[0]

    event_entries = tuple(os.scandir(store._events_path(reservation.reservation_id)))
    assert len(event_entries) == 1
    assert event_entries[0].name == "00000000000000000001.json"
    metadata = event_entries[0].stat(follow_symlinks=False)
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o400
    assert metadata.st_nlink == 1
    event_path = store._events_path(reservation.reservation_id) / event_entries[0].name
    assert event_path.read_bytes() == event.to_json_bytes()
    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="closed"):
        permit.require_current_allocation(store)


def test_concurrent_sessions_allocate_one_nonce_for_the_exact_leg(
    tmp_path,
    monkeypatch,
):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    nonce_calls = []

    def counted_nonce():
        nonce_calls.append(len(nonce_calls))
        return f"{len(nonce_calls):032x}"

    monkeypatch.setattr(execution_store_module, "_new_invocation_nonce", counted_nonce)

    def allocate(_position):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ) as session:
            return session.allocate_expected_leg().require_current_allocation(store)

    with ThreadPoolExecutor(max_workers=8) as executor:
        allocations = tuple(executor.map(allocate, range(16)))

    assert nonce_calls == [0]
    assert len(set(allocations)) == 1


def test_process_race_allocates_one_create_only_ordinal(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    process_context = get_context("fork")
    result_queue = process_context.Queue()
    processes = tuple(
        process_context.Process(
            target=_allocate_in_process,
            args=(store, reservation, prepared, result_queue),
        )
        for _ in range(4)
    )

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    assert all(process.exitcode == 0 for process in processes)
    allocations = tuple(result_queue.get() for _ in processes)
    result_queue.close()
    result_queue.join_thread()
    assert len(set(allocations)) == 1
    assert len(tuple(store._events_path(reservation.reservation_id).iterdir())) == 1


def test_durable_spawn_result_acceptance_advances_the_exact_schedule(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    schedule = source_replay_execution_schedule(reservation, prepared.request)

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        execution, provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
        )
        completion = execution.execute()
        first_case = prepared.cases[0].request_case
        expected_expert_source = (
            prepared.parent
            if schedule[0][1] == first_case.control_leg.execution_leg_id
            else prepared.candidate
        )
        expected_result = _result_payload(
            provider.invocations[0].task_evaluator_request
        )[1]
        received = session.record_result_received(completion)
        accepted_result = session.accept_received_result()
        next_allocation = session.allocate_expected_leg().require_current_allocation(
            store
        )

        assert tuple(event.event_kind.value for event in session.events[:4]) == (
            "invocation_allocated",
            "spawn_committed",
            "result_received",
            "result_accepted",
        )
        assert received.result_blob is not None
        assert provider.invocations[0].expert_source == expected_expert_source
        assert accepted_result == expected_result
        assert (
            next_allocation.execution_case_id,
            next_allocation.execution_leg_id,
        ) == schedule[1]


def test_concurrent_execution_capability_invokes_the_provider_once(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        execution, provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = tuple(executor.submit(execution.execute) for _ in range(8))
        successful = tuple(future for future in futures if future.exception() is None)
        failures = tuple(future.exception() for future in futures if future.exception())

        assert len(successful) == 1
        assert len(failures) == 7
        assert all("already consumed" in str(failure) for failure in failures)
        assert len(provider.invocations) == 1
        session.record_result_received(successful[0].result())


def test_forked_execution_capability_is_rejected_outside_its_creator_process(
    tmp_path,
):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    process_context = get_context("fork")
    process_call_count = process_context.Value("i", 0)
    result_queue = process_context.Queue()
    provider = _ProcessCountingMatchedLegProvider(
        fixture.validation_store.root,
        expert_source_replay_execution_provider_key(prepared.cases[0]),
        process_call_count=process_call_count,
    )
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
            provider,
        )
        child = process_context.Process(
            target=_reject_inherited_execution_in_process,
            args=(execution, result_queue),
        )
        child.start()
        child.join()

        assert child.exitcode == 0
        assert result_queue.get() == "rejected"
        assert process_call_count.value == 0
        session.record_result_received(execution.execute())
        assert process_call_count.value == 1

    result_queue.close()
    result_queue.join_thread()


def test_provider_exception_consumes_execution_capability(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    provider = _FailingMatchedLegProvider(
        fixture.validation_store.root,
        expert_source_replay_execution_provider_key(prepared.cases[0]),
    )
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
            provider,
        )
        with pytest.raises(RuntimeError, match="provider execution failed"):
            execution.execute()
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="already consumed",
        ):
            execution.execute()

    assert len(provider.invocations) == 1


def test_provider_identity_change_after_execution_cannot_seal_a_result(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    provider = _MutatingMatchedLegProvider(
        fixture.validation_store.root,
        expert_source_replay_execution_provider_key(prepared.cases[0]),
    )
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
            provider,
        )
        with pytest.raises(ValueError, match="identity changed"):
            execution.execute()
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="already consumed",
        ):
            execution.execute()

    assert len(provider.invocations) == 1


def test_provider_identity_change_after_spawn_burns_capability_without_a_call(
    tmp_path,
):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        execution, provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
        )
        provider.dispatch_key = replace(
            provider.dispatch_key,
            execution_provider_version=(
                f"{provider.dispatch_key.execution_provider_version}.changed"
            ),
        )
        with pytest.raises(ValueError, match="identity changed"):
            execution.execute()
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="already consumed",
        ):
            execution.execute()

    assert provider.invocations == []


def test_complete_schedule_never_reallocates_or_reuses_an_invocation(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    schedule = source_replay_execution_schedule(reservation, prepared.request)
    observed_allocations = []
    observed_expert_sources = []

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        for expected_case_id, expected_leg_id in schedule:
            allocation_permit = session.allocate_expected_leg()
            allocation = allocation_permit.require_current_allocation(store)
            observed_allocations.append(allocation)
            assert (allocation.execution_case_id, allocation.execution_leg_id) == (
                expected_case_id,
                expected_leg_id,
            )
            execution, provider = _commit_spawn(
                fixture,
                prepared,
                reservation,
                store,
                allocation_permit,
            )
            session.record_result_received(execution.execute())
            observed_expert_sources.append(provider.invocations[0].expert_source)
            session.accept_received_result()

        with pytest.raises(ExpertSourceReplayExecutionStoreError, match="complete"):
            session.allocate_expected_leg()
        assert len(session.events) == 4 * len(schedule)

    assert len({item.invocation_nonce for item in observed_allocations}) == len(
        schedule
    )
    assert len({item.opaque_invocation_id for item in observed_allocations}) == len(
        schedule
    )
    cases_by_id = {
        item.request_case.execution_case_id: item.request_case
        for item in prepared.cases
    }
    assert tuple(observed_expert_sources) == tuple(
        (
            prepared.parent
            if execution_leg_id
            == cases_by_id[execution_case_id].control_leg.execution_leg_id
            else prepared.candidate
        )
        for execution_case_id, execution_leg_id in schedule
    )

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        with pytest.raises(ExpertSourceReplayExecutionStoreError, match="complete"):
            recovered.allocate_expected_leg()


def test_reopened_spawn_marker_is_permanently_interrupted(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            permit,
        )

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="permanently interrupted",
        ):
            recovered.allocate_expected_leg()


def test_received_result_is_accepted_after_restart_without_a_spawn_permit(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        execution, provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
        )
        completion = execution.execute()
        expected = _result_payload(provider.invocations[0].task_evaluator_request)[1]
        session.record_result_received(completion)

    recovered_store = ExpertSourceReplayExecutionStore(
        store.root,
        store.trusted_root,
        prepared.settings.policy,
    )
    with recovered_store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        assert recovered.accept_received_result() == expected
        assert len(recovered.events) == 4


def test_technical_completion_is_durable_and_cannot_advance(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        provider = _MatchedLegProvider(
            fixture.validation_store.root,
            expert_source_replay_execution_provider_key(prepared.cases[0]),
            outcome=BoundedProcessOutcome.TIMED_OUT,
            returncode=-15,
            result_payload=None,
        )
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
            provider,
        )
        received = session.record_result_received(execution.execute())

        assert received.process_observation.outcome is BoundedProcessOutcome.TIMED_OUT
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="no acceptable result",
        ):
            session.accept_received_result()
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="must be accepted",
        ):
            session.allocate_expected_leg()


def test_result_and_process_observation_bounds_consume_the_spawn(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        provider = _MatchedLegProvider(
            fixture.validation_store.root,
            expert_source_replay_execution_provider_key(prepared.cases[0]),
            stdout_bytes_observed=(
                prepared.request.cases[0].compute_binding.stdout_byte_limit + 1
            ),
        )
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
            provider,
        )
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="compute authority",
        ):
            execution.execute()
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="already consumed",
        ):
            execution.execute()


def test_oversized_result_is_rejected_before_blob_publication(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        provider = _MatchedLegProvider(
            fixture.validation_store.root,
            expert_source_replay_execution_provider_key(prepared.cases[0]),
            result_payload=b"x"
            * (prepared.settings.policy.source_replay_result_byte_limit + 1),
        )
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
            provider,
        )
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="compute authority",
        ):
            execution.execute()
        assert tuple(store._results_path(reservation.reservation_id).iterdir()) == ()


def test_malformed_durable_result_cannot_advance_or_repeat_the_spawn(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        provider = _MatchedLegProvider(
            fixture.validation_store.root,
            expert_source_replay_execution_provider_key(prepared.cases[0]),
            result_payload=b"not canonical task-evaluator JSON",
        )
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
            provider,
        )
        session.record_result_received(execution.execute())
        with pytest.raises(ValueError):
            session.accept_received_result()
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="must be accepted",
        ):
            session.allocate_expected_leg()

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        with pytest.raises(ValueError):
            recovered.accept_received_result()


def test_raw_results_and_requests_cannot_cross_runtime_boundaries(
    tmp_path,
):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
        )
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="journal-sealed",
        ):
            session.record_result_received(_process_result(tmp_path))
        assert not hasattr(execution, "claim_execution")
        assert not hasattr(session, "commit_spawn")
        with pytest.raises(ExpertSourceReplayExecutionStoreError, match="immutable"):
            execution._execution_started = False
        completion = execution.execute()
        with pytest.raises(ExpertSourceReplayExecutionStoreError, match="immutable"):
            completion._provider_completion = None
        session.record_result_received(completion)
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="consumed or foreign",
        ):
            session.record_result_received(completion)


def test_provider_completion_requires_the_journal_factory_seal():
    with pytest.raises(
        ExpertSourceReplayExecutionStoreError,
        match="not journal sealed",
    ):
        SourceReplaySealedLegCompletion(object(), None, None, None)


def test_sealed_completion_cannot_cross_a_reopened_session(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
        )
        completion = execution.execute()

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="consumed or foreign",
        ):
            recovered.record_result_received(completion)


def test_spawn_publication_failure_never_yields_execution_authority(
    tmp_path,
    monkeypatch,
):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    original_publish = store._publish_event

    def fail_spawn(reservation_id, event):
        if event.event_kind.value == "spawn_committed":
            raise OSError("spawn publication interrupted")
        return original_publish(reservation_id, event)

    monkeypatch.setattr(store, "_publish_event", fail_spawn)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        allocation = allocation_permit.require_current_allocation(store)
        provider = _MatchedLegProvider(
            fixture.validation_store.root,
            expert_source_replay_execution_provider_key(prepared.cases[0]),
        )
        with pytest.raises(OSError, match="spawn publication interrupted"):
            _commit_spawn(
                fixture,
                prepared,
                reservation,
                store,
                allocation_permit,
                provider,
            )
        assert provider.invocations == []
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="must reopen",
        ):
            _ = session.events

    monkeypatch.setattr(store, "_publish_event", original_publish)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        assert (
            recovered.allocate_expected_leg().require_current_allocation(store)
            == allocation
        )


@pytest.mark.parametrize(
    "append_kind",
    (
        "allocation",
        "spawn",
        "result_blob",
        "result_received",
        "result_accepted",
    ),
)
def test_post_rename_append_failure_requires_reopen_without_fork(
    tmp_path,
    monkeypatch,
    append_kind,
):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = None
        completion = None
        if append_kind != "allocation":
            allocation_permit = session.allocate_expected_leg()
        if append_kind in {"result_blob", "result_received", "result_accepted"}:
            execution, _provider = _commit_spawn(
                fixture,
                prepared,
                reservation,
                store,
                allocation_permit,
            )
            completion = execution.execute()
        if append_kind == "result_accepted":
            session.record_result_received(completion)
        original_fsync = store._fsync_directory
        failure_directory = (
            store._results_path(reservation.reservation_id)
            if append_kind == "result_blob"
            else store._events_path(reservation.reservation_id)
        )

        def publish_then_fail(path):
            original_fsync(path)
            if path == failure_directory:
                raise OSError("post-rename fsync response lost")

        monkeypatch.setattr(store, "_fsync_directory", publish_then_fail)
        with pytest.raises(OSError, match="response lost"):
            if append_kind == "allocation":
                session.allocate_expected_leg()
            elif append_kind == "spawn":
                _commit_spawn(
                    fixture,
                    prepared,
                    reservation,
                    store,
                    allocation_permit,
                )
            elif append_kind in {"result_blob", "result_received"}:
                session.record_result_received(completion)
            else:
                session.accept_received_result()
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="must reopen",
        ):
            _ = session.events

    monkeypatch.setattr(store, "_fsync_directory", original_fsync)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        if append_kind == "allocation":
            assert len(recovered.events) == 1
            recovered.allocate_expected_leg()
        elif append_kind in {"spawn", "result_blob"}:
            assert len(recovered.events) == 2
            with pytest.raises(
                ExpertSourceReplayExecutionStoreError,
                match="permanently interrupted",
            ):
                recovered.allocate_expected_leg()
        elif append_kind == "result_received":
            assert len(recovered.events) == 3
            recovered.accept_received_result()
        else:
            assert len(recovered.events) == 4
            recovered.allocate_expected_leg()


def test_result_publication_failure_leaves_an_unrepeatable_spawn_tail(
    tmp_path,
    monkeypatch,
):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    original_publish = store._publish_event
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
        )
        completion = execution.execute()

        def fail_result(reservation_id, event):
            if event.event_kind.value == "result_received":
                raise OSError("result publication interrupted")
            return original_publish(reservation_id, event)

        monkeypatch.setattr(store, "_publish_event", fail_result)
        with pytest.raises(OSError, match="result publication interrupted"):
            session.record_result_received(completion)

    monkeypatch.setattr(store, "_publish_event", original_publish)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        with pytest.raises(
            ExpertSourceReplayExecutionStoreError,
            match="permanently interrupted",
        ):
            recovered.allocate_expected_leg()


def test_journal_rejects_a_reservation_request_substitution(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    _, other_prepared, _, _ = _journal_fixture(tmp_path)

    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="differs"):
        store.reservation_session(
            reservation=reservation,
            prepared_request=other_prepared,
        )

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        assert session.events == ()


@pytest.mark.parametrize(
    "substitution",
    ("aggregate_tolerance", "provider_key", "task_request"),
)
def test_reopen_rederives_persisted_spawn_authority(tmp_path, substitution):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
        )
        spawn_event = session.events[-1]

    changes = {}
    if substitution == "aggregate_tolerance":
        changes["aggregate_tolerance"] = spawn_event.aggregate_tolerance + 1.0
    elif substitution == "provider_key":
        changed_key = replace(
            spawn_event.execution_provider_key,
            execution_provider_version=(
                f"{spawn_event.execution_provider_key.execution_provider_version}.other"
            ),
        )
        changes.update(
            execution_provider_key=changed_key,
            provider_execution_handle=source_replay_provider_execution_handle(
                changed_key,
                spawn_event.invocation_allocation,
            ),
        )
    else:
        changes["task_evaluator_request"] = replace(
            spawn_event.task_evaluator_request,
            input_contract_fingerprint=tree_or_blob_digest(b"substituted input"),
        )
    substituted = _remint(spawn_event, **changes)
    _replace_published_event(store, reservation.reservation_id, substituted)

    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="spawn fence"):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ):
            pass


def test_reopen_rejects_a_self_consistent_adapter_authority_substitution(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
        )
        spawn_event = session.events[-1]

    fence = spawn_event.spawn_authority_fence
    original_observation = fence.task_adapter_trust_observations[0]
    changed_observation = _remint(
        original_observation,
        verifier_version=f"{original_observation.verifier_version}.other",
    )
    checked_subjects = set(fence.security_subject_ids)
    checked_subjects.remove(original_observation.observation_id)
    checked_subjects.remove(original_observation.verifier_authority_subject_id)
    checked_subjects.update(
        {
            changed_observation.observation_id,
            changed_observation.verifier_authority_subject_id,
        }
    )
    changed_denylist = _remint(
        fence.security_denylist_observation,
        checked_subject_ids=tuple(sorted(checked_subjects)),
    )
    changed_fence = _remint(
        fence,
        task_adapter_trust_observations=(changed_observation,),
        security_denylist_observation=changed_denylist,
    )
    substituted = _remint(
        spawn_event,
        spawn_authority_fence=changed_fence,
    )
    _replace_published_event(store, reservation.reservation_id, substituted)

    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="spawn fence"):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ):
            pass


def test_reopen_rejects_persisted_process_observations_over_compute_bounds(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            session.allocate_expected_leg(),
        )
        session.record_result_received(execution.execute())
        received_event = session.events[-1]

    compute = prepared.request.cases[0].compute_binding
    changed_observation = replace(
        received_event.process_observation,
        stdout_bytes_observed=compute.stdout_byte_limit + 1,
    )
    substituted = _remint(
        received_event,
        process_observation=changed_observation,
    )
    _replace_published_event(store, reservation.reservation_id, substituted)

    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="compute bounds"):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ):
            pass


def test_journal_store_rejects_a_different_prepared_policy(tmp_path):
    fixture, prepared, reservation, _store = _journal_fixture(tmp_path)
    changed_policy = replace(
        prepared.settings.policy,
        source_replay_stdout_byte_limit=(
            prepared.settings.policy.source_replay_stdout_byte_limit - 1
        ),
    )
    store = ExpertSourceReplayExecutionStore(
        (fixture.validation_store.root / "changed-policy-executions").resolve(),
        fixture.validation_store.root,
        changed_policy,
    )

    with pytest.raises(
        ExpertSourceReplayExecutionStoreError,
        match="another validation policy",
    ):
        store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        )


@pytest.mark.parametrize(
    "corruption",
    ("mode", "noncanonical", "hardlink", "fifo"),
)
def test_journal_fails_loud_for_corrupt_published_events(tmp_path, corruption):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        session.allocate_expected_leg()
    event_path = next(store._events_path(reservation.reservation_id).iterdir())

    if corruption == "mode":
        event_path.chmod(0o600)
    elif corruption == "noncanonical":
        event_path.chmod(0o600)
        with event_path.open("ab") as handle:
            handle.write(b"\n")
        event_path.chmod(0o400)
    elif corruption == "hardlink":
        os.link(event_path, event_path.with_name(event_path.name + ".hardlink"))
    else:
        event_path.unlink()
        os.mkfifo(event_path, mode=0o400)

    with pytest.raises(ExpertSourceReplayExecutionStoreError):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ):
            pass
    lock_descriptor = os.open(
        store._lock_path(reservation.reservation_id),
        os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(lock_descriptor, "r+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def test_event_ordinal_is_the_create_only_publication_slot(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        session.allocate_expected_leg()
    event_path = next(store._events_path(reservation.reservation_id).iterdir())
    original_event = SourceReplayExecutionJournalEvent.from_json_bytes(
        event_path.read_bytes()
    )
    forked_allocation = TaskEvaluatorInvocationAllocation(
        reservation_id=original_event.reservation_id,
        execution_case_id=original_event.execution_case_id,
        execution_leg_id=original_event.execution_leg_id,
        invocation_nonce="f" * 32,
    )
    if forked_allocation == original_event.invocation_allocation:
        forked_allocation = TaskEvaluatorInvocationAllocation(
            reservation_id=original_event.reservation_id,
            execution_case_id=original_event.execution_case_id,
            execution_leg_id=original_event.execution_leg_id,
            invocation_nonce="0" * 32,
        )
    forked_event = SourceReplayExecutionJournalEvent.mint(
        schema_version=original_event.schema_version,
        event_number=original_event.event_number,
        predecessor_event_id=original_event.predecessor_event_id,
        event_kind=original_event.event_kind,
        reservation_id=original_event.reservation_id,
        execution_request_id=original_event.execution_request_id,
        execution_case_id=original_event.execution_case_id,
        execution_leg_id=original_event.execution_leg_id,
        invocation_allocation=forked_allocation,
        spawn_authority_fence=None,
        execution_provider_key=None,
        provider_execution_handle=None,
        task_evaluator_request=None,
        aggregate_tolerance=None,
        process_observation=None,
        result_blob=None,
        task_evaluator_result=None,
    )
    fork_staging_path = store._staging_path(reservation.reservation_id) / (
        f".event-{'f' * 32}.tmp"
    )
    fork_staging_path.write_bytes(forked_event.to_json_bytes())
    fork_staging_path.chmod(0o400)

    with pytest.raises(OSError):
        store._rename_no_replace(
            fork_staging_path,
            store._event_path(reservation.reservation_id, forked_event),
        )
    assert event_path.read_bytes() == original_event.to_json_bytes()
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as recovered:
        assert recovered.events == (original_event,)


def test_journal_removes_only_validated_orphan_staging_files(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ):
        pass
    staging_root = store._staging_path(reservation.reservation_id)
    orphan = staging_root / f".event-{'a' * 32}.tmp"
    orphan.write_bytes(b"orphan")
    orphan.chmod(0o600)

    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ):
        pass
    assert not orphan.exists()

    unexpected = staging_root / "untrusted-entry"
    unexpected.write_bytes(b"unsafe")
    unexpected.chmod(0o600)
    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="unexpected"):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ):
            pass


def test_journal_bounds_event_reads_and_staging_enumeration(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        session.allocate_expected_leg()
    event_path = next(store._events_path(reservation.reservation_id).iterdir())
    event_path.chmod(0o600)
    event_path.write_bytes(b"x" * (store.maximum_event_size_bytes + 1))
    event_path.chmod(0o400)

    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="configured bound"):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ):
            pass

    event_path.unlink()
    staging_root = store._staging_path(reservation.reservation_id)
    for token in ("a", "b"):
        path = staging_root / f".event-{token * 32}.tmp"
        path.write_bytes(b"orphan")
        path.chmod(0o600)
    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="staging.*bound"):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ):
            pass


def test_result_blob_corruption_and_untrusted_root_fail_loud(tmp_path):
    fixture, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        prepared_request=prepared,
    ) as session:
        allocation_permit = session.allocate_expected_leg()
        execution, _provider = _commit_spawn(
            fixture,
            prepared,
            reservation,
            store,
            allocation_permit,
        )
        received = session.record_result_received(execution.execute())
    result_path = store._result_path(reservation.reservation_id, received.result_blob)
    result_path.chmod(0o600)

    with pytest.raises(ExpertSourceReplayExecutionStoreError):
        with store.reservation_session(
            reservation=reservation,
            prepared_request=prepared,
        ):
            pass

    trusted_root = fixture.validation_store.root
    trusted_root.chmod(0o777)
    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="owner-private"):
        ExpertSourceReplayExecutionStore(
            (trusted_root / "untrusted-executions").resolve(),
            trusted_root,
            prepared.settings.policy,
        )
