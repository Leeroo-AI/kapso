from __future__ import annotations

import fcntl
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import pytest

import kapso.cross_run.expert.replay_execution_store as execution_store_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import expert_source_replay_matched_compute_digest
from kapso.cross_run.expert.replay_execution_store import (
    ExpertSourceReplayExecutionStore,
    ExpertSourceReplayExecutionStoreError,
    SourceReplayExecutionJournalEvent,
    source_replay_execution_schedule,
)
from kapso.cross_run.expert.replay_protocol import TaskEvaluatorInvocationAllocation
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
    )
    return fixture, prepared, committed.reservation, store


def _remint(record, **changes):
    values = {field.name: getattr(record, field.name) for field in fields(record)}
    values.pop(record.IDENTITY_FIELD)
    values.update(changes)
    return type(record).mint(**values)


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
        request=prepared.request,
    ) as session:
        permit = session.allocate_expected_leg()
        assert session.allocate_expected_leg() is permit
        allocation = permit.require_current_allocation(store)
        events = session.events

    recovered_store = ExpertSourceReplayExecutionStore(store.root, store.trusted_root)
    with recovered_store.reservation_session(
        reservation=reservation,
        request=prepared.request,
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
        request=prepared.request,
    ) as session:
        permit = session.allocate_expected_leg()
        event = session.events[0]

    event_entries = tuple(os.scandir(store._events_path(reservation.reservation_id)))
    assert len(event_entries) == 1
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
            request=prepared.request,
        ) as session:
            return session.allocate_expected_leg().require_current_allocation(store)

    with ThreadPoolExecutor(max_workers=8) as executor:
        allocations = tuple(executor.map(allocate, range(16)))

    assert nonce_calls == [0]
    assert len(set(allocations)) == 1


def test_journal_rejects_a_reservation_request_substitution(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    _, other_prepared, _, _ = _journal_fixture(tmp_path)

    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="differs"):
        store.reservation_session(
            reservation=reservation,
            request=other_prepared.request,
        )

    with store.reservation_session(
        reservation=reservation,
        request=prepared.request,
    ) as session:
        assert session.events == ()


@pytest.mark.parametrize(
    "corruption",
    ("mode", "noncanonical", "hardlink", "fifo"),
)
def test_journal_fails_loud_for_corrupt_published_events(tmp_path, corruption):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        request=prepared.request,
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
            request=prepared.request,
        ):
            pass
    lock_descriptor = os.open(
        store._lock_path(reservation.reservation_id),
        os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(lock_descriptor, "r+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def test_journal_detects_a_forked_event_ordinal(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        request=prepared.request,
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
    )
    fork_path = store._event_path(reservation.reservation_id, forked_event)
    fork_path.write_bytes(forked_event.to_json_bytes())
    fork_path.chmod(0o400)

    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="forked"):
        with store.reservation_session(
            reservation=reservation,
            request=prepared.request,
        ):
            pass


def test_journal_removes_only_validated_orphan_staging_files(tmp_path):
    _, prepared, reservation, store = _journal_fixture(tmp_path)
    with store.reservation_session(
        reservation=reservation,
        request=prepared.request,
    ):
        pass
    staging_root = store._staging_path(reservation.reservation_id)
    orphan = staging_root / f".event-{'a' * 32}.tmp"
    orphan.write_bytes(b"orphan")
    orphan.chmod(0o600)

    with store.reservation_session(
        reservation=reservation,
        request=prepared.request,
    ):
        pass
    assert not orphan.exists()

    unexpected = staging_root / "untrusted-entry"
    unexpected.write_bytes(b"unsafe")
    unexpected.chmod(0o600)
    with pytest.raises(ExpertSourceReplayExecutionStoreError, match="unexpected"):
        with store.reservation_session(
            reservation=reservation,
            request=prepared.request,
        ):
            pass
