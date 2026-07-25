"""Crash-durable create-only execution state for run-scoped actions."""

from __future__ import annotations

import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from threading import Barrier, Event

import pytest

import kapso.cross_run.launch.run_action_store as run_action_store_module
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.launch.run_action_contracts import RunActionIntent
from kapso.cross_run.launch.run_action_gate import (
    bind_run_action_frontier,
    RunFrontierActionError,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import RunActionLedgerError
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_MUTATION_AUTHORITY,
    _RUN_ACTION_STORE_AUTHORITY,
    RunActionExecutionEventKind,
    RunActionExecutionStore,
    RunActionExecutionEvent,
    RunActionAcceptance,
    RunActionResultDisposition,
    RunActionStoreError,
    RunActionTerminalReason,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationClaim,
)
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from test_run_frontier_action_gate import (
    _accept_action,
    _action_case,
    _boundary_identity,
    _claim_action,
    _commit_workspace_edit,
    _issue_implementation_agent,
)
from test_launch_resolver import resolver_case
from test_run_state_publisher import publisher_case
from test_run_action_supervisor_contracts import (
    _execution_policy,
    _prepared_execution,
)


def _open_store(active, settings):
    return RunActionExecutionStore(
        active_workspace=active,
        settings=settings,
        _authority=_RUN_ACTION_STORE_AUTHORITY,
    )


def test_reservation_contracts_are_not_reexported_by_action_store():
    for name in (
        "RunActionFrontierBinding",
        "RunActionRequestBlob",
        "RunActionReservation",
        "RunActionSpawnCommit",
        "RunActionViewBinding",
        "RunActionWorkspaceBinding",
    ):
        assert not hasattr(run_action_store_module, name)


def _open_session(store, reservation):
    return store._session(
        reservation,
        _authority=_RUN_ACTION_MUTATION_AUTHORITY,
    )


def _prepare_session(session, *, container_id=None, inode_offset=None):
    policy = _execution_policy(
        kind=session.reservation.intent.kind,
        workspace_access=session.reservation.intent.workspace_access,
    )
    claim = session.claim_preparation(policy)
    operation_digest = hashlib.sha256(
        session.reservation.intent.operation_id.encode("utf-8")
    ).hexdigest()
    prepared = _prepared_execution(
        claim=claim,
        container_id=operation_digest if container_id is None else container_id,
        inode_offset=(
            int(operation_digest[:8], 16) if inode_offset is None else inode_offset
        ),
    )
    session.commit_prepared_execution(prepared)
    return prepared


def _execution_event(
    *,
    reservation,
    event_number,
    predecessor_event_id,
    event_kind,
    preparation_claim=None,
    prepared_execution=None,
    spawn_commit=None,
):
    return RunActionExecutionEvent.mint(
        event_number=event_number,
        predecessor_event_id=predecessor_event_id,
        event_kind=event_kind,
        reservation=reservation,
        preparation_claim=preparation_claim,
        prepared_execution=prepared_execution,
        spawn_commit=spawn_commit,
        result_receipt=None,
        acceptance=None,
        terminal_reason=None,
        workspace_after=None,
    )


def _reserve_concurrently(active, settings, reservation, request_payload, start):
    store = _open_store(active, settings)
    start.wait()
    with _open_session(store, reservation) as session:
        session.reserve(request_payload)


def _acquire_workspace_lock_concurrently(
    active,
    settings,
    start,
    acquired,
):
    store = _open_store(active, settings)
    start.wait()
    with ExitStack() as descriptors:
        store.lock_workspace(
            RunFrontierWorkspaceAccess.READ_ONLY,
            descriptors,
        )
        acquired.set()


def _reserved_action(
    case,
    *,
    operation_id="durable_agent_0123456789abcdef",
    frontier=None,
    workspace=None,
):
    if frontier is None:
        _publisher, frontier, _security, _gate = _action_case(case)
    request_payload = b'{"prompt":"complete, untruncated request"}'
    intent = RunActionIntent.from_request(
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id=operation_id,
        request_payload=request_payload,
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
    )
    if workspace is None:
        with ExitStack() as descriptors:
            workspace_descriptor, _identity = case["active"]._open_execution_workspace(
                descriptors
            )
            workspace = inspect_run_workspace_frontier(
                workspace_descriptor,
                settings=case["settings"],
                expected_commit_sha=frontier.checkpoint.safety_state.derivative_frontier.evidence.branch_heads[
                    case["settings"].workspace_git_branch
                ],
            )
    binding = bind_run_action_frontier(frontier, workspace)
    predecessor_ledger = _open_store(
        case["active"],
        case["settings"],
    ).snapshot()
    reservation = RunActionReservation.build(
        intent=intent,
        frontier=binding,
        predecessor_ledger=predecessor_ledger,
    )
    return frontier, request_payload, reservation, workspace


def test_action_store_reopens_complete_request_result_and_terminal_prefix(
    publisher_case,
):
    frontier, request_payload, reservation, workspace = _reserved_action(publisher_case)
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(store, reservation) as session:
        reserved = session.reserve(request_payload)
        assert reserved.event_kind is RunActionExecutionEventKind.INTENT_RESERVED
        with pytest.raises(RunActionStoreError, match="unavailable before spawn"):
            session.read_request()
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=session.reservation.intent.boundary_identity,
        )
        assert spawn.boundary_identity == session.reservation.intent.boundary_identity
        assert session.read_request() == request_payload
        raw_result = b'{"provider_response":"complete"}'
        result = session.record_result(
            spawn_commit=spawn,
            result_payload=raw_result,
        )
        assert session.read_result(result) == raw_result
        accepted_result = b'{"proposal":"complete"}'
        acceptance = session.accept_result(
            result_receipt=result,
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=accepted_result,
            workspace_after=workspace,
        )
        assert session.read_accepted_result(acceptance) == accepted_result
        assert acceptance.workspace_after.to_identity() == workspace

    snapshot = store.snapshot()
    assert snapshot.event_count == 6
    assert snapshot.operation_tails[0].tail_kind is (
        RunActionExecutionEventKind.RESULT_ACCEPTED
    )

    reopened = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    assert reopened.snapshot() == snapshot
    with _open_session(reopened, reservation) as session:
        assert session.read_request() == request_payload
        assert session.read_result(session.events[4].result_receipt) == raw_result
        assert (
            session.read_accepted_result(session.events[5].acceptance)
            == accepted_result
        )


def test_action_store_rejects_legacy_and_cross_authority_event_splices(
    publisher_case,
):
    frontier, request_payload, reservation, workspace = _reserved_action(
        publisher_case,
        operation_id="splice_target_action_0123456789abcdef",
    )
    _frontier, _alternate_payload, alternate_reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id="splice_source_action_0123456789abcdef",
        frontier=frontier,
        workspace=workspace,
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(store, reservation) as session:
        session.reserve(request_payload)
        durable_prepared = _prepare_session(session)
        session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )
        durable_events = session.events

    policy = _execution_policy(
        kind=alternate_reservation.intent.kind,
        workspace_access=alternate_reservation.intent.workspace_access,
    )
    foreign_claim = RunActionPreparationClaim.mint(
        reservation=alternate_reservation,
        execution_policy=policy,
    )
    foreign_prepared = _prepared_execution(
        claim=foreign_claim,
        container_id="e" * 64,
        inode_offset=701,
    )
    alternate_prepared = _prepared_execution(
        claim=durable_events[1].preparation_claim,
        container_id="f" * 64,
        inode_offset=1701,
    )
    legacy_spawn = _execution_event(
        reservation=reservation,
        event_number=2,
        predecessor_event_id=durable_events[0].event_id,
        event_kind=RunActionExecutionEventKind.SPAWN_COMMITTED,
        spawn_commit=durable_events[3].spawn_commit,
    )
    spliced_claim = _execution_event(
        reservation=reservation,
        event_number=2,
        predecessor_event_id=durable_events[0].event_id,
        event_kind=RunActionExecutionEventKind.PREPARATION_CLAIMED,
        preparation_claim=foreign_claim,
    )
    spliced_prepared = _execution_event(
        reservation=reservation,
        event_number=3,
        predecessor_event_id=durable_events[1].event_id,
        event_kind=RunActionExecutionEventKind.EXECUTION_PREPARED,
        prepared_execution=foreign_prepared,
    )
    alternate_prepared_event = _execution_event(
        reservation=reservation,
        event_number=3,
        predecessor_event_id=durable_events[1].event_id,
        event_kind=RunActionExecutionEventKind.EXECUTION_PREPARED,
        prepared_execution=alternate_prepared,
    )
    spliced_spawn = _execution_event(
        reservation=reservation,
        event_number=4,
        predecessor_event_id=alternate_prepared_event.event_id,
        event_kind=RunActionExecutionEventKind.SPAWN_COMMITTED,
        spawn_commit=durable_events[3].spawn_commit,
    )

    invalid_prefixes = (
        ((durable_events[0], legacy_spawn), "changed identity"),
        ((durable_events[0], spliced_claim), "differs from its reservation"),
        (
            (durable_events[0], durable_events[1], spliced_prepared),
            "differs from its claim",
        ),
        (
            (
                durable_events[0],
                durable_events[1],
                alternate_prepared_event,
                spliced_spawn,
            ),
            "spawn differs from its reservation",
        ),
    )
    for prefix, message in invalid_prefixes:
        with pytest.raises(RunActionStoreError, match=message):
            run_action_store_module._validate_event_prefix(prefix)
    assert durable_prepared == durable_events[2].prepared_execution


def test_action_store_rejects_wrong_spawn_security_before_publication(
    publisher_case,
):
    _frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id="wrong_spawn_security_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    with _open_session(store, reservation) as session:
        session.reserve(request_payload)
        _prepare_session(session)
        with pytest.raises(RunActionStoreError, match="spawn security"):
            session.commit_spawn(
                security_observation_id=content_id(
                    "security-denylist-observation",
                    {"wrong": True},
                ),
                boundary_identity=reservation.intent.boundary_identity,
            )
        assert len(session.events) == 3

    operation_digest = hashlib.sha256(
        reservation.intent.operation_id.encode("utf-8")
    ).hexdigest()
    assert not (store_path / f"operation-{operation_digest}-event-0004.json").exists()


@pytest.mark.parametrize("constrained_resource", ("entry", "byte"))
def test_action_reservation_requires_complete_lifecycle_capacity_without_strands(
    publisher_case,
    constrained_resource,
):
    _frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id=f"{constrained_resource}_capacity_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    constrained_settings = replace(publisher_case["settings"])
    with _open_session(store, reservation) as session:
        intent_event = session._event(
            RunActionExecutionEventKind.INTENT_RESERVED,
        )
        if constrained_resource == "entry":
            projected_entry_count = len(tuple(store_path.iterdir())) + 2 + 5 + 2
            object.__setattr__(
                constrained_settings,
                "run_action_store_entry_limit",
                projected_entry_count - 1,
            )
        else:
            current_size_bytes = sum(
                path.stat().st_size for path in store_path.iterdir()
            )
            projected_size_bytes = (
                current_size_bytes
                + len(request_payload)
                + len(intent_event.to_json_bytes())
                + 5 * constrained_settings.run_action_event_size_bytes
                + 2 * constrained_settings.run_action_result_size_bytes
            )
            object.__setattr__(
                constrained_settings,
                "run_action_store_size_bytes",
                projected_size_bytes - 1,
            )
        object.__setattr__(store, "_settings", constrained_settings)
        with pytest.raises(RunActionStoreError, match="lacks capacity"):
            session.reserve(request_payload)
        assert session.events == ()

    assert {path.name for path in store_path.iterdir()} == {
        "registry.lock",
        "workspace.lock",
    }


def test_action_store_rejects_reminted_failed_edit_with_changed_workspace(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"complete before tamper"}'
    permit = _issue_implementation_agent(
        gate,
        frontier,
        "reminted_failed_edit",
        payload,
    )
    with gate.hold(permit, payload) as lease:
        _claim_action(gate, lease)
        _commit_workspace_edit(
            publisher_case,
            "reminted-failed-edit.txt",
            "complete\n",
        )
        _accept_action(gate, lease)

    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    events = store.inspect().events_for(permit.intent.operation_id)
    original = events[-1]
    acceptance = RunActionAcceptance.mint(
        result_receipt_id=original.acceptance.result_receipt_id,
        disposition=RunActionResultDisposition.FAILED,
        accepted_result_blob=original.acceptance.accepted_result_blob,
        workspace_after=original.acceptance.workspace_after,
    )
    tampered = RunActionExecutionEvent.mint(
        event_number=original.event_number,
        predecessor_event_id=original.predecessor_event_id,
        event_kind=original.event_kind,
        reservation=original.reservation,
        preparation_claim=None,
        prepared_execution=None,
        spawn_commit=None,
        result_receipt=None,
        acceptance=acceptance,
        terminal_reason=None,
        workspace_after=None,
    )
    operation_digest = tree_or_blob_digest(
        permit.intent.operation_id.encode("utf-8")
    ).removeprefix("sha256:")
    event_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
        / f"operation-{operation_digest}-event-0006.json"
    )
    event_path.chmod(0o600)
    event_path.write_bytes(tampered.to_json_bytes())
    event_path.chmod(0o400)
    orphan_payload = b'{"must_survive_rejected_reopen":true}'
    orphan_path = event_path.parent / (
        "accepted-"
        f"{tree_or_blob_digest(orphan_payload).removeprefix('sha256:')}.blob"
    )
    orphan_path.write_bytes(orphan_payload)
    orphan_path.chmod(0o400)

    with pytest.raises(RunActionStoreError, match="failed editing action"):
        _open_store(
            publisher_case["active"],
            publisher_case["settings"],
        )
    assert orphan_path.read_bytes() == orphan_payload


def test_action_store_requires_sealed_construction_and_mutation(
    publisher_case,
):
    with pytest.raises(RunActionStoreError, match="active launch settings"):
        RunActionExecutionStore(
            active_workspace=publisher_case["active"],
            settings=publisher_case["settings"],
            _authority=object(),
        )
    _frontier, _payload, reservation, _workspace = _reserved_action(publisher_case)
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with pytest.raises(RunActionStoreError, match="sealed mutation"):
        store._session(reservation, _authority=object())


def test_action_session_registration_failure_closes_pinned_descriptors(
    publisher_case,
    monkeypatch,
):
    _frontier, _payload, reservation, _workspace = _reserved_action(publisher_case)
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    opened_descriptors = []
    original_open_store = store._open_store

    def capture_open_store(descriptors):
        opened = original_open_store(descriptors)
        opened_descriptors.append(opened[0])
        return opened

    def reject_registration(_session):
        raise RunActionStoreError("injected registration conflict")

    monkeypatch.setattr(store, "_open_store", capture_open_store)
    monkeypatch.setattr(store, "_register_session", reject_registration)

    with pytest.raises(RunActionStoreError, match="injected registration"):
        with _open_session(store, reservation):
            raise AssertionError("unreachable")

    assert opened_descriptors
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_action_store_durably_rejects_duplicate_operation(
    publisher_case,
):
    _frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(store, reservation) as session:
        session.reserve(request_payload)
    with _open_session(store, reservation) as session:
        with pytest.raises(RunActionStoreError, match="already reserved"):
            session.reserve(request_payload)

    assert store.snapshot().event_count == 1


def test_action_store_cancel_and_interrupted_prefixes_are_terminal(
    publisher_case,
):
    frontier, request_payload, reservation, workspace = _reserved_action(
        publisher_case,
        operation_id="cancelled_action_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(store, reservation) as session:
        session.reserve(request_payload)
        session.cancel(RunActionTerminalReason.STALE_FRONTIER)
        with pytest.raises(RunActionStoreError, match="terminal reason"):
            replace(
                session.events[-1],
                terminal_reason=RunActionTerminalReason.PROVIDER_FAILED,
            )
        with pytest.raises(RunActionStoreError, match="requires a"):
            session.commit_spawn(
                security_observation_id=(
                    frontier.checkpoint.safety_state.security_observation.observation_id
                ),
                boundary_identity=session.reservation.intent.boundary_identity,
            )

    (
        second_frontier,
        second_payload,
        second_reservation,
        second_workspace,
    ) = _reserved_action(
        publisher_case,
        operation_id="interrupted_action_0123456789abcdef",
        frontier=frontier,
        workspace=workspace,
    )
    with _open_session(store, second_reservation) as session:
        session.reserve(second_payload)
        _prepare_session(session)
        session.commit_spawn(
            security_observation_id=(
                second_frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=session.reservation.intent.boundary_identity,
        )
        session.interrupt(
            reason=RunActionTerminalReason.PROVIDER_INTERRUPTED,
            workspace_after=second_workspace,
        )
    tails = {tail.operation_id: tail for tail in store.snapshot().operation_tails}
    assert tails[reservation.intent.operation_id].tail_kind is (
        RunActionExecutionEventKind.CANCELLED
    )
    assert tails[second_reservation.intent.operation_id].tail_kind is (
        RunActionExecutionEventKind.INTERRUPTED
    )
    assert workspace == second_workspace


def test_action_store_rejects_request_substitution_and_result_corruption(
    publisher_case,
):
    _frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(store, reservation) as session:
        with pytest.raises(RunActionStoreError, match="complete request"):
            session.reserve(request_payload + b" altered")
        session.reserve(request_payload)
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(reservation.frontier.security_observation_id),
            boundary_identity=session.reservation.intent.boundary_identity,
        )
        result = session.record_result(
            spawn_commit=spawn,
            result_payload=b'{"result":"durable"}',
        )
    result_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
        / f"result-{result.result_blob.digest.removeprefix('sha256:')}.blob"
    )
    result_path.chmod(0o600)
    result_path.write_bytes(b'{"result":"corrupt"}')
    result_path.chmod(0o400)
    with pytest.raises(RunActionStoreError):
        store.snapshot()


def test_action_ledger_snapshot_is_canonical_and_content_addressed(
    publisher_case,
):
    _frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    assert store.snapshot().event_count == 0
    with _open_session(store, reservation) as session:
        session.reserve(request_payload)
    snapshot = store.snapshot()
    assert snapshot == type(snapshot).from_json_bytes(snapshot.to_json_bytes())
    assert tree_or_blob_digest(snapshot.to_json_bytes()).startswith("sha256:")
    with pytest.raises(RunActionLedgerError, match="tail is invalid"):
        replace(
            snapshot.operation_tails[0],
            tail_kind=RunActionExecutionEventKind.RESULT_ACCEPTED,
        )
    with pytest.raises(RunActionLedgerError, match="changed or extended"):
        type(snapshot).empty().require_predecessor(snapshot)


def test_action_binding_rejects_workspace_outside_checkpoint_frontier(
    publisher_case,
):
    frontier, _payload, _reservation, workspace = _reserved_action(publisher_case)
    with pytest.raises(RunFrontierActionError, match="checkpoint branch frontier"):
        bind_run_action_frontier(
            frontier,
            replace(workspace, commit_sha="f" * 40),
        )


def test_action_store_cleans_crash_staging_before_reopen(
    publisher_case,
):
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    staged_paths = tuple(
        store_path / f".accepted-{position:032x}.tmp"
        for position in range(
            publisher_case["settings"].run_action_staging_entry_limit + 1
        )
    )
    for staged_path in staged_paths:
        staged_path.write_bytes(b'{"accepted":"crash-window"}')
        staged_path.chmod(0o400)

    reopened = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )

    assert not any(staged_path.exists() for staged_path in staged_paths)
    assert reopened.snapshot().event_count == 0


def test_action_store_reclaims_unreferenced_final_blobs(
    publisher_case,
):
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    orphan_paths = []
    for kind, payload in (
        ("input", b'{"orphaned":"request"}'),
        ("result", b'{"orphaned":"raw-result"}'),
        ("accepted", b'{"orphaned":"accepted-result"}'),
    ):
        digest = tree_or_blob_digest(payload).removeprefix("sha256:")
        path = store_path / f"{kind}-{digest}.blob"
        path.write_bytes(payload)
        path.chmod(0o400)
        orphan_paths.append(path)

    reopened = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )

    assert not any(path.exists() for path in orphan_paths)
    assert reopened.snapshot().event_count == 0


def test_action_store_rejects_reused_prepared_container_identity(
    publisher_case,
):
    frontier, request_payload, first, workspace = _reserved_action(
        publisher_case,
        operation_id="first_provider_action_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    shared_container_id = "d" * 64
    with _open_session(store, first) as session:
        session.reserve(request_payload)
        _prepare_session(session, container_id=shared_container_id)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=session.reservation.intent.boundary_identity,
        )
        result = session.record_result(
            spawn_commit=spawn,
            result_payload=b'{"result":"first"}',
        )
        session.accept_result(
            result_receipt=result,
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=b'{"accepted":"first"}',
            workspace_after=workspace,
        )
    _frontier, second_payload, second, _workspace = _reserved_action(
        publisher_case,
        operation_id="second_provider_action_0123456789abcdef",
        frontier=frontier,
        workspace=workspace,
    )
    with _open_session(store, second) as session:
        session.reserve(second_payload)
        policy = _execution_policy(
            kind=session.reservation.intent.kind,
            workspace_access=session.reservation.intent.workspace_access,
        )
        claim = session.claim_preparation(policy)
        operation_digest = hashlib.sha256(
            session.reservation.intent.operation_id.encode("utf-8")
        ).hexdigest()
        prepared = _prepared_execution(
            claim=claim,
            container_id=shared_container_id,
            inode_offset=int(operation_digest[:8], 16),
        )
        with pytest.raises(RunActionStoreError, match="authority was reused"):
            session.commit_prepared_execution(
                prepared,
            )


def test_action_store_rejects_reused_prepared_filesystem_authority(
    publisher_case,
):
    frontier, request_payload, first, workspace = _reserved_action(
        publisher_case,
        operation_id="first_slot_action_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    shared_inode_offset = 4701
    with _open_session(store, first) as session:
        session.reserve(request_payload)
        _prepare_session(session, inode_offset=shared_inode_offset)
        spawn = session.commit_spawn(
            security_observation_id=first.frontier.security_observation_id,
            boundary_identity=first.intent.boundary_identity,
        )
        result = session.record_result(
            spawn_commit=spawn,
            result_payload=b'{"result":"first slot authority"}',
        )
        session.accept_result(
            result_receipt=result,
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=b'{"accepted":"first slot authority"}',
            workspace_after=workspace,
        )

    _frontier, second_payload, second, _workspace = _reserved_action(
        publisher_case,
        operation_id="second_slot_action_0123456789abcdef",
        frontier=frontier,
        workspace=workspace,
    )
    with _open_session(store, second) as session:
        session.reserve(second_payload)
        policy = _execution_policy(
            kind=second.intent.kind,
            workspace_access=second.intent.workspace_access,
        )
        claim = session.claim_preparation(policy)
        candidate = _prepared_execution(
            claim=claim,
            container_id="c" * 64,
            inode_offset=shared_inode_offset,
        )
        with pytest.raises(RunActionStoreError, match="authority was reused"):
            session.commit_prepared_execution(candidate)


def test_action_store_concurrent_reservation_is_create_once(
    publisher_case,
):
    _frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case
    )
    start = Barrier(3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(
                _reserve_concurrently,
                publisher_case["active"],
                publisher_case["settings"],
                reservation,
                request_payload,
                start,
            )
            for _position in range(2)
        )
        start.wait()
    outcomes = tuple(
        future.exception() if future.exception() is not None else future.result()
        for future in futures
    )

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, RunActionStoreError) for outcome in outcomes) == 1
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    assert store.snapshot().event_count == 1


def test_action_store_concurrent_distinct_reservations_share_one_floor(
    publisher_case,
):
    frontier, request_payload, first, workspace = _reserved_action(
        publisher_case,
        operation_id="first_floor_action_0123456789abcdef",
    )
    _frontier, second_payload, second, _workspace = _reserved_action(
        publisher_case,
        operation_id="second_floor_action_0123456789abcdef",
        frontier=frontier,
        workspace=workspace,
    )
    start = Barrier(3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(
                _reserve_concurrently,
                publisher_case["active"],
                publisher_case["settings"],
                first,
                request_payload,
                start,
            ),
            pool.submit(
                _reserve_concurrently,
                publisher_case["active"],
                publisher_case["settings"],
                second,
                second_payload,
                start,
            ),
        )
        start.wait()
    outcomes = tuple(
        future.exception() if future.exception() is not None else future.result()
        for future in futures
    )

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, RunActionStoreError) for outcome in outcomes) == 1
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    assert store.snapshot().event_count == 1
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    assert {
        path.name for path in store_path.iterdir() if path.name.endswith(".lock")
    } == {"registry.lock", "workspace.lock"}


def test_action_store_rejects_event_sequence_gap(
    publisher_case,
):
    _frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(store, reservation) as session:
        session.reserve(request_payload)
    operation_digest = tree_or_blob_digest(
        reservation.intent.operation_id.encode("utf-8")
    ).removeprefix("sha256:")
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    (store_path / f"operation-{operation_digest}-event-0001.json").rename(
        store_path / f"operation-{operation_digest}-event-0002.json"
    )

    with pytest.raises(RunActionStoreError, match="sequence has a gap"):
        store.snapshot()


def test_action_store_rejects_fixed_lock_inode_substitution(
    publisher_case,
    tmp_path,
):
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    lock_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
        / "registry.lock"
    )
    lock_path.rename(tmp_path / "original-registry.lock")
    lock_path.touch(mode=0o600)

    with pytest.raises(RunActionStoreError, match="differs from its receipt"):
        store.snapshot()


def test_action_store_workspace_lock_excludes_concurrent_sessions(
    publisher_case,
):
    start = Event()
    acquired = Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _acquire_workspace_lock_concurrently,
            publisher_case["active"],
            publisher_case["settings"],
            start,
            acquired,
        )
        store = _open_store(
            publisher_case["active"],
            publisher_case["settings"],
        )
        with ExitStack() as descriptors:
            store.lock_workspace(
                RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
                descriptors,
            )
            start.set()
            assert not acquired.wait(0.1)
        assert acquired.wait(5)
        assert future.result() is None
