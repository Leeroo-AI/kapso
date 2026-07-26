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
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import ContractValidationError
from kapso.cross_run.launch.run_action_contracts import RunActionIntent
from kapso.cross_run.launch.run_action_gate import (
    bind_run_action_frontier,
    RunFrontierActionError,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionLedgerError,
    RunActionLedgerSnapshot,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_RECOVERY_AUTHORITY,
    _RUN_ACTION_RESERVATION_AUTHORITY,
    _RUN_ACTION_STORE_AUTHORITY,
    RunActionExecutionEventKind,
    RunActionExecutionStore,
    RunActionExecutionEvent,
    RunActionAcceptance,
    RunActionResultDecision,
    RunActionResultDisposition,
    RunActionStoreInspection,
    RunActionStoreError,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationAllocation,
    RunActionPreparationClaim,
    issue_runtime_volume_authority,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from kapso.cross_run.launch.run_action_workspace_promotion import (
    RunActionWorkspacePromotion,
)
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from test_run_frontier_action_gate import (
    _action_case,
    _boundary_identity,
    _commit_workspace_edit,
    _reserve_implementation_agent,
    _reserve_ideation_agent,
    _successor_at_boundary,
)
from test_launch_resolver import resolver_case
from test_run_state_publisher import publisher_case
from test_run_action_release_contracts import _release_adoption_for_event
from test_run_action_termination_contracts import (
    _pre_release_loss,
    _termination_graph,
)
from test_run_action_supervisor_contracts import (
    _activation_revalidation_receipt,
    _execution_policy,
    _prepared_execution,
    _remint_contract,
    _result_capture_receipt,
    _spawn_commit,
    _terminal_observation,
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


def test_reservation_authority_cannot_open_an_execution_session():
    assert not hasattr(RunActionExecutionStore, "_session")
    assert not hasattr(
        run_action_store_module._RunActionExecutionSession,
        "reserve",
    )
    assert not hasattr(
        run_action_store_module._RunActionExecutionSession,
        "accept_result",
    )
    assert not hasattr(
        run_action_store_module._RunActionExecutionSession,
        "read_accepted_result",
    )
    assert not hasattr(
        run_action_store_module,
        "_RUN_ACTION_MUTATION_AUTHORITY",
    )


def test_store_owns_fresh_preparation_allocation_entropy():
    claim = _prepared_execution().preparation_claim

    first = run_action_store_module._issue_preparation_allocation(claim)
    second = run_action_store_module._issue_preparation_allocation(claim)

    assert first.preparation_claim == claim
    assert second.preparation_claim == claim
    assert (
        first.runtime_volume_authority.generation_nonce
        != second.runtime_volume_authority.generation_nonce
    )
    assert first.runtime_volume_authority.labels != (
        second.runtime_volume_authority.labels
    )
    assert not hasattr(
        run_action_store_module,
        "issue_fresh_preparation_allocation",
    )


def _open_session(store, reservation):
    return store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    )


def _reserve_action(store, reservation, request_payload):
    return store._reserve_action(
        reservation,
        request_payload,
        _authority=_RUN_ACTION_RESERVATION_AUTHORITY,
    )


def _prepare_session(session, *, container_id=None, inode_offset=None):
    policy = _execution_policy(
        kind=session.reservation.intent.kind,
        workspace_access=session.reservation.intent.workspace_access,
    )
    allocation = session.allocate_preparation(policy)
    operation_digest = hashlib.sha256(
        session.reservation.intent.operation_id.encode("utf-8")
    ).hexdigest()
    prepared = _prepared_execution(
        claim=allocation.preparation_claim,
        authority=allocation.runtime_volume_authority,
        container_id=operation_digest if container_id is None else container_id,
        inode_offset=(
            int(operation_digest[:8], 16) if inode_offset is None else inode_offset
        ),
    )
    session.commit_prepared_execution(prepared)
    return prepared


def _record_captured_result(session, spawn, payload, security_observation):
    prepared = session.events[2].prepared_execution
    activation = _activation_revalidation_receipt(prepared, spawn)
    if session.events[-1].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED:
        session.commit_activation(activation)
    workload_release_adoption = _release_adoption_for_event(
        session.events[4],
        security_observation,
    )
    terminal = _terminal_observation(
        prepared,
        spawn,
        workload_release_adoption,
    )
    capture = _result_capture_receipt(
        prepared,
        activation,
        terminal,
        payload,
    )
    return session.record_result(
        spawn_commit=spawn,
        workload_release_adoption=workload_release_adoption,
        terminal_observation=terminal,
        result_capture_receipt=capture,
        result_payload=payload,
    )


def _released_provider_termination(session, spawn, security_observation):
    prepared = session.events[2].prepared_execution
    activation = _activation_revalidation_receipt(prepared, spawn)
    if session.events[-1].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED:
        session.commit_activation(activation)
    adoption = _release_adoption_for_event(
        session.events[4],
        security_observation,
    )
    terminal = _remint_contract(
        _terminal_observation(prepared, spawn, adoption),
        exit_code=137,
        oom_killed=True,
    )
    return RunActionProviderTerminationReceipt.mint(
        disposition=RunActionProviderTerminationDisposition.FAILED,
        reason=RunActionProviderTerminationReason.OOM,
        activation_event_id=session.events[4].event_id,
        workload_release_adoption=adoption,
        terminal_observation=terminal,
        timeout_directive_publication=None,
        empty_result_capture_receipt=None,
        pre_release_main_loss_observation=None,
    )


def _pre_release_provider_termination(session):
    activation = session.events[4].activation_revalidation_receipt
    loss = _pre_release_loss(activation, session.events[4].event_id)
    return RunActionProviderTerminationReceipt.mint(
        disposition=RunActionProviderTerminationDisposition.FAILED,
        reason=RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
        activation_event_id=session.events[4].event_id,
        workload_release_adoption=None,
        terminal_observation=None,
        timeout_directive_publication=None,
        empty_result_capture_receipt=None,
        pre_release_main_loss_observation=loss,
    )


def _execution_event(
    *,
    reservation,
    event_number,
    predecessor_event_id,
    event_kind,
    preparation_allocation=None,
    prepared_execution=None,
    spawn_commit=None,
    activation_revalidation_receipt=None,
    provider_termination_receipt=None,
    result_receipt=None,
    result_decision=None,
    acceptance=None,
    workspace_after=None,
):
    return RunActionExecutionEvent.mint(
        event_number=event_number,
        predecessor_event_id=predecessor_event_id,
        event_kind=event_kind,
        reservation=reservation,
        preparation_allocation=preparation_allocation,
        prepared_execution=prepared_execution,
        spawn_commit=spawn_commit,
        activation_revalidation_receipt=activation_revalidation_receipt,
        provider_termination_receipt=provider_termination_receipt,
        result_receipt=result_receipt,
        result_decision=result_decision,
        acceptance=acceptance,
        workspace_after=workspace_after,
    )


def _reserve_concurrently(active, settings, reservation, request_payload, start):
    store = _open_store(active, settings)
    start.wait()
    _reserve_action(store, reservation, request_payload)


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
    reserved = _reserve_action(store, reservation, request_payload)
    assert reserved.event_kind is RunActionExecutionEventKind.INTENT_RESERVED
    with _open_session(store, reservation) as session:
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
        result = _record_captured_result(
            session,
            spawn,
            raw_result,
            frontier.checkpoint.safety_state.security_observation,
        )
        assert session.read_result(result) == raw_result
        accepted_result = b'{"proposal":"complete"}'
        decision = session.decide_result(
            result_interpreter_identity=(
                reservation.intent.boundary_identity.result_interpreter_identity
            ),
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=accepted_result,
            workspace_promotion=None,
        )
        acceptance = session.accept_decision(
            workspace_after=workspace,
        )
        assert session.read_decided_result(decision) == accepted_result
        assert acceptance.workspace_after.to_identity() == workspace

    snapshot = store.snapshot()
    assert snapshot.event_count == 8
    assert snapshot.operation_tails[0].tail_kind is (
        RunActionExecutionEventKind.RESULT_ACCEPTED
    )
    with pytest.raises(RunActionLedgerError, match="tail is invalid"):
        replace(
            snapshot.operation_tails[0],
            event_ids=snapshot.operation_tails[0].event_ids[:7],
        )

    reopened = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    assert reopened.snapshot() == snapshot
    with _open_session(reopened, reservation) as session:
        assert session.read_request() == request_payload
        assert session.read_result(session.events[5].result_receipt) == raw_result
        assert (
            session.read_decided_result(session.events[6].result_decision)
            == accepted_result
        )


def test_provider_termination_is_terminal_without_result_blobs(
    publisher_case,
):
    frontier, request_payload, reservation, workspace = _reserved_action(
        publisher_case,
        operation_id="provider_terminated_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=reservation.intent.boundary_identity,
        )
        receipt = _released_provider_termination(
            session,
            spawn,
            frontier.checkpoint.safety_state.security_observation,
        )
        assert session.terminate_provider(receipt) == receipt
        event = session.events[-1]
        assert event.event_number == 6
        assert event.predecessor_event_id == session.events[4].event_id
        assert event.event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED
        assert event.provider_termination_receipt == receipt
        assert event.result_receipt is None
        with pytest.raises(RunActionStoreError, match="result_received tail"):
            session.decide_result(
                result_interpreter_identity=(
                    reservation.intent.boundary_identity.result_interpreter_identity
                ),
                disposition=RunActionResultDisposition.FAILED,
                accepted_result_payload=b'{"must":"not publish"}',
                workspace_promotion=None,
            )
        with pytest.raises(RunActionStoreError, match="activation_committed tail"):
            session.terminate_provider(receipt)

    snapshot = store.snapshot()
    assert snapshot.event_count == 6
    assert snapshot.operation_tails[0].tail_kind is (
        RunActionExecutionEventKind.PROVIDER_TERMINATED
    )
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    assert not tuple(store_path.glob("result-*.blob"))
    assert not tuple(store_path.glob("accepted-*.blob"))
    extended_tail = replace(
        snapshot.operation_tails[0],
        event_ids=(
            *snapshot.operation_tails[0].event_ids,
            content_id(
                "run-action-execution-event",
                {"forbidden": "terminal extension"},
            ),
        ),
        tail_kind=RunActionExecutionEventKind.RESULT_DECIDED,
    )
    extended = RunActionLedgerSnapshot.build((extended_tail,))
    with pytest.raises(RunActionLedgerError, match="terminal predecessor"):
        extended.require_predecessor(snapshot)

    reopened = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(reopened, reservation) as session:
        assert session.events[-1] == event
        assert session.read_request() == request_payload
    assert RunActionStoreInspection.workspace_chain(
        (reopened.inspect().events_for(reservation.intent.operation_id),)
    ) == (
        (
            RunActionWorkspaceBinding.from_identity(workspace),
            RunActionWorkspaceBinding.from_identity(workspace),
        ),
    )


def test_provider_termination_rejects_event_position_and_graph_splices(
    publisher_case,
):
    frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id="provider_termination_splice_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=reservation.intent.boundary_identity,
        )
        activation = _activation_revalidation_receipt(
            session.events[2].prepared_execution,
            spawn,
        )
        session.commit_activation(activation)
        exact = _pre_release_provider_termination(session)
        with pytest.raises(
            RunActionStoreError,
            match="terminal kind differs from its event position",
        ):
            _execution_event(
                reservation=reservation,
                event_number=5,
                predecessor_event_id=session.events[3].event_id,
                event_kind=RunActionExecutionEventKind.PROVIDER_TERMINATED,
                provider_termination_receipt=exact,
            )
        with pytest.raises(
            RunActionStoreError,
            match="terminal kind differs from its event position",
        ):
            _execution_event(
                reservation=reservation,
                event_number=7,
                predecessor_event_id=content_id(
                    "run-action-execution-event",
                    {"fixture": "event six"},
                ),
                event_kind=RunActionExecutionEventKind.PROVIDER_TERMINATED,
                provider_termination_receipt=exact,
            )
        with pytest.raises(
            RunActionStoreError,
            match="durable events 2 and 5",
        ):
            session.terminate_provider(
                _termination_graph(RunActionProviderTerminationReason.OOM)
            )
        foreign_event_id = content_id(
            "run-action-execution-event",
            {"fixture": "foreign activation"},
        )
        foreign_loss = _pre_release_loss(activation, foreign_event_id)
        foreign_loss_receipt = RunActionProviderTerminationReceipt.mint(
            disposition=RunActionProviderTerminationDisposition.FAILED,
            reason=RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
            activation_event_id=foreign_event_id,
            workload_release_adoption=None,
            terminal_observation=None,
            timeout_directive_publication=None,
            empty_result_capture_receipt=None,
            pre_release_main_loss_observation=foreign_loss,
        )
        with pytest.raises(
            RunActionStoreError,
            match="durable events 2 and 5",
        ):
            session.terminate_provider(foreign_loss_receipt)
        foreign_graph = _termination_graph(
            RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
        )
        foreign_loss_at_current_event = _remint_contract(
            foreign_graph.pre_release_main_loss_observation,
            activation_event_id=session.events[4].event_id,
        )
        foreign_graph_at_current_event = RunActionProviderTerminationReceipt.mint(
            disposition=RunActionProviderTerminationDisposition.FAILED,
            reason=RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
            activation_event_id=session.events[4].event_id,
            workload_release_adoption=None,
            terminal_observation=None,
            timeout_directive_publication=None,
            empty_result_capture_receipt=None,
            pre_release_main_loss_observation=foreign_loss_at_current_event,
        )
        with pytest.raises(
            RunActionStoreError,
            match="durable events 2 and 5",
        ):
            session.terminate_provider(foreign_graph_at_current_event)
        session.terminate_provider(exact)


@pytest.mark.parametrize("termination_event_committed", (False, True))
def test_provider_termination_publication_recovers_exactly_across_crash(
    publisher_case,
    monkeypatch,
    termination_event_committed,
):
    publication_state = "committed" if termination_event_committed else "absent"
    frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id=f"termination_publication_{publication_state}_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )
        receipt = _released_provider_termination(
            session,
            spawn,
            frontier.checkpoint.safety_state.security_observation,
        )

    publish_event_locked = store._publish_event_locked

    def interrupt_termination_publication(store_descriptor, operation_id, event):
        if (
            event.event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED
            and not termination_event_committed
        ):
            raise RuntimeError("injected death before provider termination event")
        publish_event_locked(store_descriptor, operation_id, event)
        if event.event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED:
            raise RuntimeError("injected death after provider termination event")

    monkeypatch.setattr(
        store,
        "_publish_event_locked",
        interrupt_termination_publication,
    )
    with pytest.raises(RuntimeError, match="provider termination event"):
        with _open_session(store, reservation) as session:
            session.terminate_provider(receipt)
    monkeypatch.setattr(store, "_publish_event_locked", publish_event_locked)

    reopened = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    events = reopened.inspect().events_for(reservation.intent.operation_id)
    expected_tail = (
        RunActionExecutionEventKind.PROVIDER_TERMINATED
        if termination_event_committed
        else RunActionExecutionEventKind.ACTIVATION_COMMITTED
    )
    assert events[-1].event_kind is expected_tail
    assert len(events) == (6 if termination_event_committed else 5)
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    assert not tuple(store_path.glob("result-*.blob"))
    assert not tuple(store_path.glob("accepted-*.blob"))
    with _open_session(reopened, reservation) as session:
        if termination_event_committed:
            with pytest.raises(
                RunActionStoreError,
                match="activation_committed tail",
            ):
                session.terminate_provider(receipt)
        else:
            session.terminate_provider(receipt)
            assert session.events[-1].event_kind is (
                RunActionExecutionEventKind.PROVIDER_TERMINATED
            )


@pytest.mark.parametrize("mutation", ("old", "unknown"))
def test_store_rejects_old_or_unknown_provider_termination_event_json(
    publisher_case,
    mutation,
):
    frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id=f"termination_{mutation}_json_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=reservation.intent.boundary_identity,
        )
        receipt = _released_provider_termination(
            session,
            spawn,
            frontier.checkpoint.safety_state.security_observation,
        )
        session.terminate_provider(receipt)
        event = session.events[-1]
    payload = event.to_dict()
    if mutation == "old":
        del payload["provider_termination_receipt"]
    else:
        payload["legacy_provider_interruption"] = "timeout"
    event_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
        / store._event_name(reservation.intent.operation_id, 6)
    )
    event_path.chmod(0o600)
    event_path.write_bytes(canonical_json_bytes(payload))
    event_path.chmod(0o400)

    with pytest.raises(ContractValidationError, match="fields mismatch"):
        _open_store(
            publisher_case["active"],
            publisher_case["settings"],
        )


def test_gate_allows_the_next_action_after_provider_termination(
    publisher_case,
):
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    reservation = _reserve_ideation_agent(
        gate,
        frontier,
        b'{"prompt":"provider will fail"}',
    )
    with _open_session(gate._action_store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )
        receipt = _released_provider_termination(
            session,
            spawn,
            frontier.checkpoint.safety_state.security_observation,
        )
        session.terminate_provider(receipt)

    terminated_ledger = gate._action_store.snapshot()
    next_reservation = gate.reserve(
        frontier,
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="after_provider_termination_0123456789abcdef",
        request_payload=b'{"prompt":"next action"}',
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
    )

    assert terminated_ledger.event_count == 6
    assert (
        next_reservation.predecessor_ledger_snapshot_id
        == terminated_ledger.ledger_snapshot_id
    )


def test_publisher_reconciles_provider_termination_as_unchanged_terminal(
    publisher_case,
):
    publisher, frontier, _security, gate = _action_case(publisher_case)
    reservation = _reserve_ideation_agent(
        gate,
        frontier,
        b'{"prompt":"provider will terminate"}',
    )
    with _open_session(gate._action_store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )
        receipt = _released_provider_termination(
            session,
            spawn,
            frontier.checkpoint.safety_state.security_observation,
        )
        session.terminate_provider(receipt)
    bundle, checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        frontier,
        RunSafetyBoundary.IDEATION,
    )

    published = publisher.publish(
        publisher.issue_publication_permit(frontier, checkpoint, bundle),
        checkpoint,
        bundle,
    )

    assert published.projection.action_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.PROVIDER_TERMINATED
    )


def test_action_store_requires_exact_decision_before_acceptance(
    publisher_case,
) -> None:
    frontier, request_payload, reservation, workspace = _reserved_action(
        publisher_case,
        operation_id="decision_boundary_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=reservation.intent.boundary_identity,
        )
        result = _record_captured_result(
            session,
            spawn,
            b'{"provider_response":"complete"}',
            frontier.checkpoint.safety_state.security_observation,
        )
        with pytest.raises(RunActionStoreError, match="result_decided tail"):
            session.accept_decision(workspace_after=workspace)
        foreign_interpreter = _boundary_identity(
            RunFrontierActionKind.EMBEDDING
        ).result_interpreter_identity
        with pytest.raises(RunActionStoreError, match="differs"):
            session.decide_result(
                result_interpreter_identity=foreign_interpreter,
                disposition=RunActionResultDisposition.SUCCEEDED,
                accepted_result_payload=b'{"accepted":"foreign"}',
                workspace_promotion=None,
            )
        decision = session.decide_result(
            result_interpreter_identity=(
                reservation.intent.boundary_identity.result_interpreter_identity
            ),
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=b'{"accepted":"exact"}',
            workspace_promotion=None,
        )
        assert session.events[-1].event_kind is (
            RunActionExecutionEventKind.RESULT_DECIDED
        )
        acceptance = session.accept_decision(workspace_after=workspace)
        events = session.events

    assert decision.result_receipt_id == result.result_receipt_id
    assert acceptance.result_decision_id == decision.result_decision_id
    assert events[-1].event_number == 8
    foreign_decision = RunActionResultDecision.mint(
        result_receipt_id=result.result_receipt_id,
        result_interpreter_identity_id=(
            foreign_interpreter.result_interpreter_identity_id
        ),
        disposition=decision.disposition,
        accepted_result_blob=decision.accepted_result_blob,
        workspace_promotion=None,
    )
    foreign_decision_event = _execution_event(
        reservation=reservation,
        event_number=7,
        predecessor_event_id=events[5].event_id,
        event_kind=RunActionExecutionEventKind.RESULT_DECIDED,
        result_decision=foreign_decision,
    )
    with pytest.raises(RunActionStoreError, match="decision differs"):
        run_action_store_module._validate_event_prefix(
            (*events[:6], foreign_decision_event)
        )
    foreign_acceptance = RunActionAcceptance.mint(
        result_decision_id=foreign_decision.result_decision_id,
        workspace_after=acceptance.workspace_after,
    )
    foreign_acceptance_event = _execution_event(
        reservation=reservation,
        event_number=8,
        predecessor_event_id=events[6].event_id,
        event_kind=RunActionExecutionEventKind.RESULT_ACCEPTED,
        acceptance=foreign_acceptance,
    )
    with pytest.raises(RunActionStoreError, match="acceptance differs"):
        run_action_store_module._validate_event_prefix(
            (*events[:7], foreign_acceptance_event)
        )


def test_action_store_rejects_terminal_and_capture_occurrence_splices(
    publisher_case,
):
    frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id="terminal_capture_splice_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        prepared = _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=session.reservation.intent.boundary_identity,
        )
        payload = b'{"provider_response":"complete"}'
        activation = _activation_revalidation_receipt(prepared, spawn)
        session.commit_activation(activation)
        workload_release_adoption = _release_adoption_for_event(
            session.events[4],
            frontier.checkpoint.safety_state.security_observation,
        )
        foreign_prepared = _prepared_execution(
            claim=prepared.preparation_claim,
            inode_offset=8181,
        )
        foreign_spawn = _spawn_commit(
            foreign_prepared,
            invocation_nonce="2" * 32,
        )
        foreign_activation = _activation_revalidation_receipt(
            foreign_prepared,
            foreign_spawn,
        )
        foreign_terminal = _terminal_observation(
            foreign_prepared,
            foreign_spawn,
        )
        foreign_capture = _result_capture_receipt(
            foreign_prepared,
            foreign_activation,
            foreign_terminal,
            payload,
        )
        with pytest.raises(
            RunActionStoreError,
            match="result differs from its durable spawn",
        ):
            session.record_result(
                spawn_commit=spawn,
                workload_release_adoption=workload_release_adoption,
                terminal_observation=foreign_terminal,
                result_capture_receipt=foreign_capture,
                result_payload=payload,
            )

        terminal = _terminal_observation(
            prepared,
            spawn,
            workload_release_adoption,
        )
        capture = _result_capture_receipt(
            prepared,
            activation,
            terminal,
            payload,
        )
        for unsuccessful_terminal in (
            _remint_contract(terminal, exit_code=1),
            _remint_contract(terminal, oom_killed=True),
        ):
            unsuccessful_capture = _remint_contract(
                capture,
                terminal_observation_id=(unsuccessful_terminal.terminal_observation_id),
            )
            with pytest.raises(
                RunActionStoreError,
                match="result differs from its spawn",
            ):
                session.record_result(
                    spawn_commit=spawn,
                    workload_release_adoption=workload_release_adoption,
                    terminal_observation=unsuccessful_terminal,
                    result_capture_receipt=unsuccessful_capture,
                    result_payload=payload,
                )
        substituted_volume = _remint_contract(
            prepared.runtime_volume_evidence,
            root_inode=prepared.runtime_volume_evidence.root_inode + 1,
        )
        substituted_capture = _remint_contract(
            capture,
            reobserved_volume_evidence=substituted_volume,
        )
        with pytest.raises(RunActionStoreError, match="result differs from its spawn"):
            session.record_result(
                spawn_commit=spawn,
                workload_release_adoption=workload_release_adoption,
                terminal_observation=terminal,
                result_capture_receipt=substituted_capture,
                result_payload=payload,
            )
        backwards_usage_capture = _remint_contract(
            capture,
            reobserved_volume_evidence=activation.reobserved_volume_evidence,
        )
        with pytest.raises(RunActionStoreError, match="result differs from its spawn"):
            session.record_result(
                spawn_commit=spawn,
                workload_release_adoption=workload_release_adoption,
                terminal_observation=terminal,
                result_capture_receipt=backwards_usage_capture,
                result_payload=payload,
            )
        substituted_result_inode = _remint_contract(
            capture,
            inode=capture.inode + 1,
        )
        with pytest.raises(RunActionStoreError, match="result differs from its spawn"):
            session.record_result(
                spawn_commit=spawn,
                workload_release_adoption=workload_release_adoption,
                terminal_observation=terminal,
                result_capture_receipt=substituted_result_inode,
                result_payload=payload,
            )
        substituted_result_parent = _remint_contract(
            capture,
            parent_inode=capture.parent_inode + 1,
        )
        with pytest.raises(RunActionStoreError, match="result differs from its spawn"):
            session.record_result(
                spawn_commit=spawn,
                workload_release_adoption=workload_release_adoption,
                terminal_observation=terminal,
                result_capture_receipt=substituted_result_parent,
                result_payload=payload,
            )


def test_activation_selection_survives_ambiguous_event_publication(
    publisher_case,
    monkeypatch,
) -> None:
    frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id="ambiguous_activation_publication_01234567",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    publish_event_locked = store._publish_event_locked

    def publish_then_interrupt(store_descriptor, operation_id, event):
        publish_event_locked(store_descriptor, operation_id, event)
        if event.event_kind is RunActionExecutionEventKind.ACTIVATION_COMMITTED:
            raise RuntimeError("injected death after activation event publication")

    monkeypatch.setattr(
        store,
        "_publish_event_locked",
        publish_then_interrupt,
    )
    _reserve_action(store, reservation, request_payload)
    with pytest.raises(RuntimeError, match="after activation event"):
        with _open_session(store, reservation) as session:
            prepared = _prepare_session(session)
            spawn = session.commit_spawn(
                security_observation_id=(
                    frontier.checkpoint.safety_state.security_observation.observation_id
                ),
                boundary_identity=reservation.intent.boundary_identity,
            )
            session.commit_activation(_activation_revalidation_receipt(prepared, spawn))

    events = store.inspect().events_for(reservation.intent.operation_id)
    assert len(events) == 5
    assert events[-1].event_kind is (RunActionExecutionEventKind.ACTIVATION_COMMITTED)
    assert events[-1].activation_revalidation_receipt.prepared_execution == prepared
    assert events[-1].activation_revalidation_receipt.spawn_commit == spawn


def test_preparation_allocation_survives_ambiguous_event_publication(
    publisher_case,
    monkeypatch,
) -> None:
    _frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id="ambiguous_allocation_publication_01234567",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    publish_event_locked = store._publish_event_locked

    def publish_then_interrupt(store_descriptor, operation_id, event):
        publish_event_locked(store_descriptor, operation_id, event)
        if event.event_kind is RunActionExecutionEventKind.PREPARATION_ALLOCATED:
            raise RuntimeError("injected death after allocation event publication")

    monkeypatch.setattr(
        store,
        "_publish_event_locked",
        publish_then_interrupt,
    )
    _reserve_action(store, reservation, request_payload)
    with pytest.raises(RuntimeError, match="after allocation event"):
        with _open_session(store, reservation) as session:
            session.allocate_preparation(
                _execution_policy(
                    kind=reservation.intent.kind,
                    workspace_access=reservation.intent.workspace_access,
                )
            )

    events = store.inspect().events_for(reservation.intent.operation_id)
    assert len(events) == 2
    assert events[-1].event_kind is RunActionExecutionEventKind.PREPARATION_ALLOCATED
    allocation = events[-1].preparation_allocation
    assert allocation.preparation_claim.reservation == reservation
    assert (
        allocation.runtime_volume_authority.preparation_claim_id
        == allocation.preparation_claim.preparation_claim_id
    )


@pytest.mark.parametrize("decision_event_committed", (False, True))
def test_result_decision_blob_and_event_recover_as_one_durable_boundary(
    publisher_case,
    monkeypatch,
    decision_event_committed,
) -> None:
    publication_state = "committed" if decision_event_committed else "absent"
    frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id=f"decision_publication_{publication_state}_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=reservation.intent.boundary_identity,
        )
        _record_captured_result(
            session,
            spawn,
            b'{"provider_response":"complete"}',
            frontier.checkpoint.safety_state.security_observation,
        )

    publish_event_locked = store._publish_event_locked

    def interrupt_decision_publication(store_descriptor, operation_id, event):
        if (
            event.event_kind is RunActionExecutionEventKind.RESULT_DECIDED
            and not decision_event_committed
        ):
            raise RuntimeError("injected death before result decision event")
        publish_event_locked(store_descriptor, operation_id, event)
        if event.event_kind is RunActionExecutionEventKind.RESULT_DECIDED:
            raise RuntimeError("injected death after result decision event")

    monkeypatch.setattr(
        store,
        "_publish_event_locked",
        interrupt_decision_publication,
    )
    with pytest.raises(RuntimeError, match="result decision event"):
        with _open_session(store, reservation) as session:
            session.decide_result(
                result_interpreter_identity=(
                    reservation.intent.boundary_identity.result_interpreter_identity
                ),
                disposition=RunActionResultDisposition.SUCCEEDED,
                accepted_result_payload=b'{"accepted":"durable decision"}',
                workspace_promotion=None,
            )

    reopened = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    events = reopened.inspect().events_for(reservation.intent.operation_id)
    expected_tail = (
        RunActionExecutionEventKind.RESULT_DECIDED
        if decision_event_committed
        else RunActionExecutionEventKind.RESULT_RECEIVED
    )
    assert events[-1].event_kind is expected_tail
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    accepted_blobs = tuple(store_path.glob("accepted-*.blob"))
    assert len(accepted_blobs) == int(decision_event_committed)


def test_allocated_frontier_invalidation_is_terminal_without_request_authority(
    publisher_case,
) -> None:
    _frontier, request_payload, reservation, workspace = _reserved_action(
        publisher_case,
        operation_id="claimed_resource_loss_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        session.allocate_preparation(
            _execution_policy(
                kind=reservation.intent.kind,
                workspace_access=reservation.intent.workspace_access,
            )
        )
        session.invalidate_frontier()
        assert len(session.events) == 3
        terminal = session.events[-1]
        assert terminal.workspace_after.to_identity() == workspace
        with pytest.raises(RunActionStoreError, match="unavailable before spawn"):
            session.read_request()

    reopened = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    tail = reopened.snapshot().operation_tails[-1]
    assert tail.event_count == 3
    assert tail.tail_kind is RunActionExecutionEventKind.FRONTIER_INVALIDATED


def test_prepared_frontier_invalidation_is_terminal_and_cannot_extend(
    publisher_case,
) -> None:
    frontier, request_payload, reservation, _workspace = _reserved_action(
        publisher_case,
        operation_id="prepared_frontier_loss_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        session.invalidate_frontier()
        assert len(session.events) == 4
        with pytest.raises(RunActionStoreError, match="requires a"):
            session.commit_spawn(
                security_observation_id=(
                    frontier.checkpoint.safety_state.security_observation.observation_id
                ),
                boundary_identity=reservation.intent.boundary_identity,
            )
        with pytest.raises(RunActionStoreError, match="unavailable before spawn"):
            session.read_request()

    assert store.snapshot().operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.FRONTIER_INVALIDATED
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
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
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
    foreign_allocation = RunActionPreparationAllocation.mint(
        preparation_claim=foreign_claim,
        runtime_volume_authority=issue_runtime_volume_authority(
            foreign_claim,
            "c" * 32,
        ),
    )
    foreign_prepared = _prepared_execution(
        claim=foreign_claim,
        authority=foreign_allocation.runtime_volume_authority,
        container_id="e" * 64,
        inode_offset=701,
    )
    alternate_prepared = _prepared_execution(
        claim=durable_events[1].preparation_allocation.preparation_claim,
        authority=(durable_events[1].preparation_allocation.runtime_volume_authority),
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
    spliced_allocation = _execution_event(
        reservation=reservation,
        event_number=2,
        predecessor_event_id=durable_events[0].event_id,
        event_kind=RunActionExecutionEventKind.PREPARATION_ALLOCATED,
        preparation_allocation=foreign_allocation,
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
    missing_workspace_invalidation = _execution_event(
        reservation=reservation,
        event_number=3,
        predecessor_event_id=durable_events[1].event_id,
        event_kind=RunActionExecutionEventKind.FRONTIER_INVALIDATED,
    )

    invalid_prefixes = (
        ((durable_events[0], legacy_spawn), "changed identity"),
        ((durable_events[0], spliced_allocation), "differs from its reservation"),
        (
            (durable_events[0], durable_events[1], spliced_prepared),
            "differs from its allocation",
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
        (
            (
                durable_events[0],
                durable_events[1],
                missing_workspace_invalidation,
            ),
            "exact unchanged workspace",
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
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
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
    intent_event = _execution_event(
        reservation=reservation,
        event_number=1,
        predecessor_event_id=None,
        event_kind=RunActionExecutionEventKind.INTENT_RESERVED,
    )
    if constrained_resource == "entry":
        projected_capacity = len(tuple(store_path.iterdir())) + 2 + 7 + 2
        object.__setattr__(
            constrained_settings,
            "run_action_store_entry_limit",
            projected_capacity - 1,
        )
    else:
        current_size_bytes = sum(path.stat().st_size for path in store_path.iterdir())
        projected_capacity = (
            current_size_bytes
            + len(request_payload)
            + len(intent_event.to_json_bytes())
            + 7 * constrained_settings.run_action_event_size_bytes
            + 2 * constrained_settings.run_action_result_size_bytes
        )
        object.__setattr__(
            constrained_settings,
            "run_action_store_size_bytes",
            projected_capacity - 1,
        )
    object.__setattr__(store, "_settings", constrained_settings)
    with pytest.raises(RunActionStoreError, match="lacks capacity"):
        _reserve_action(store, reservation, request_payload)

    assert {path.name for path in store_path.iterdir()} == {
        "registry.lock",
        "workspace.lock",
    }
    setting_name = (
        "run_action_store_entry_limit"
        if constrained_resource == "entry"
        else "run_action_store_size_bytes"
    )
    object.__setattr__(constrained_settings, setting_name, projected_capacity)
    reserved = _reserve_action(store, reservation, request_payload)
    assert reserved.event_kind is RunActionExecutionEventKind.INTENT_RESERVED


def test_action_store_rejects_reminted_failed_edit_with_changed_workspace(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"complete before tamper"}'
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "reminted_failed_edit",
        payload,
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )
        _record_captured_result(
            session,
            spawn,
            b'{"provider_result":"complete"}',
            frontier.checkpoint.safety_state.security_observation,
        )
        session.decide_result(
            result_interpreter_identity=(
                reservation.intent.boundary_identity.result_interpreter_identity
            ),
            disposition=RunActionResultDisposition.FAILED,
            accepted_result_payload=b'{"accepted_result":"failed"}',
            workspace_promotion=None,
        )
        session.accept_decision(
            workspace_after=reservation.frontier.workspace_before.to_identity(),
        )

    _commit_workspace_edit(
        publisher_case,
        "reminted-failed-edit.txt",
        "complete\n",
    )
    with ExitStack() as descriptors:
        workspace_descriptor, _identity = publisher_case[
            "active"
        ]._open_execution_workspace(descriptors)
        changed_workspace = inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=publisher_case["settings"],
            expected_commit_sha=None,
        )
    events = store.inspect().events_for(reservation.intent.operation_id)
    original = events[-1]
    acceptance = RunActionAcceptance.mint(
        result_decision_id=original.acceptance.result_decision_id,
        workspace_after=RunActionWorkspaceBinding.from_identity(changed_workspace),
    )
    tampered = RunActionExecutionEvent.mint(
        event_number=original.event_number,
        predecessor_event_id=original.predecessor_event_id,
        event_kind=original.event_kind,
        reservation=original.reservation,
        preparation_allocation=None,
        prepared_execution=None,
        spawn_commit=None,
        activation_revalidation_receipt=None,
        provider_termination_receipt=None,
        result_receipt=None,
        result_decision=None,
        acceptance=acceptance,
        workspace_after=None,
    )
    operation_digest = tree_or_blob_digest(
        reservation.intent.operation_id.encode("utf-8")
    ).removeprefix("sha256:")
    event_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
        / f"operation-{operation_digest}-event-0008.json"
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

    with pytest.raises(RunActionStoreError, match="changed its workspace"):
        _open_store(
            publisher_case["active"],
            publisher_case["settings"],
        )
    assert orphan_path.read_bytes() == orphan_payload


def test_successful_edit_decision_owns_exact_workspace_promotion(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "promotion_required",
        b'{"implementation":"isolated success"}',
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with _open_session(store, reservation) as session:
        prepared = _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )
        result_receipt = _record_captured_result(
            session,
            spawn,
            b'{"provider_result":"successful edit"}',
            frontier.checkpoint.safety_state.security_observation,
        )
        before = reservation.frontier.workspace_before
        candidate_identity = replace(
            before.to_identity(),
            workspace_identity=(
                before.workspace_device,
                before.workspace_inode + 1,
            ),
            commit_sha=("a" * 40 if before.commit_sha != "a" * 40 else "b" * 40),
            parent_commit_shas=(before.commit_sha,),
            git_tree_sha=("c" * 40 if before.git_tree_sha != "c" * 40 else "d" * 40),
            source_tree_digest=(
                "sha256:" + "1" * 64
                if before.source_tree_digest != "sha256:" + "1" * 64
                else "sha256:" + "2" * 64
            ),
            source_entry_count=before.source_entry_count + 1,
            source_size_bytes=before.source_size_bytes + 1,
        )
        with pytest.raises(
            RunActionStoreError,
            match="decision promotion",
        ):
            session.decide_result(
                result_interpreter_identity=(
                    reservation.intent.boundary_identity.result_interpreter_identity
                ),
                disposition=RunActionResultDisposition.SUCCEEDED,
                accepted_result_payload=b'{"accepted_result":"missing promotion"}',
                workspace_promotion=None,
            )
        promotion = RunActionWorkspacePromotion.mint(
            result_receipt_id=result_receipt.result_receipt_id,
            prepared_workspace_proof_id=(
                prepared.workspace_proof.prepared_workspace_proof_id
            ),
            candidate_workspace=RunActionWorkspaceBinding.from_identity(
                candidate_identity
            ),
        )
        foreign_result_promotion = RunActionWorkspacePromotion.mint(
            result_receipt_id=content_id(
                "run-action-result-receipt",
                {"foreign": "result"},
            ),
            prepared_workspace_proof_id=(promotion.prepared_workspace_proof_id),
            candidate_workspace=promotion.candidate_workspace,
        )
        with pytest.raises(
            RunActionStoreError,
            match="durable execution",
        ):
            session.decide_result(
                result_interpreter_identity=(
                    reservation.intent.boundary_identity.result_interpreter_identity
                ),
                disposition=RunActionResultDisposition.SUCCEEDED,
                accepted_result_payload=b'{"accepted_result":"foreign result"}',
                workspace_promotion=foreign_result_promotion,
            )
        foreign_proof_promotion = RunActionWorkspacePromotion.mint(
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=content_id(
                "run-action-prepared-workspace-proof",
                {"foreign": "proof"},
            ),
            candidate_workspace=promotion.candidate_workspace,
        )
        with pytest.raises(
            RunActionStoreError,
            match="durable execution",
        ):
            session.decide_result(
                result_interpreter_identity=(
                    reservation.intent.boundary_identity.result_interpreter_identity
                ),
                disposition=RunActionResultDisposition.SUCCEEDED,
                accepted_result_payload=b'{"accepted_result":"foreign proof"}',
                workspace_promotion=foreign_proof_promotion,
            )
        nondirect_promotion = RunActionWorkspacePromotion.mint(
            result_receipt_id=promotion.result_receipt_id,
            prepared_workspace_proof_id=(promotion.prepared_workspace_proof_id),
            candidate_workspace=RunActionWorkspaceBinding.from_identity(
                replace(
                    candidate_identity,
                    parent_commit_shas=("f" * 40,),
                )
            ),
        )
        with pytest.raises(
            RunActionStoreError,
            match="direct source successor",
        ):
            session.decide_result(
                result_interpreter_identity=(
                    reservation.intent.boundary_identity.result_interpreter_identity
                ),
                disposition=RunActionResultDisposition.SUCCEEDED,
                accepted_result_payload=b'{"accepted_result":"nondirect"}',
                workspace_promotion=nondirect_promotion,
            )
        decision = session.decide_result(
            result_interpreter_identity=(
                reservation.intent.boundary_identity.result_interpreter_identity
            ),
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=b'{"accepted_result":"successful edit"}',
            workspace_promotion=promotion,
        )
        with pytest.raises(
            RunActionStoreError,
            match="promoted workspace",
        ):
            session.accept_decision(
                workspace_after=before.to_identity(),
            )
        acceptance = session.accept_decision(
            workspace_after=candidate_identity,
        )
        assert decision.workspace_promotion == promotion
        assert acceptance.workspace_after == promotion.candidate_workspace


def test_action_store_requires_sealed_construction_and_mutation(
    publisher_case,
):
    with pytest.raises(RunActionStoreError, match="active launch settings"):
        RunActionExecutionStore(
            active_workspace=publisher_case["active"],
            settings=publisher_case["settings"],
            _authority=object(),
        )
    _frontier, payload, reservation, _workspace = _reserved_action(publisher_case)
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with pytest.raises(RunActionStoreError, match="sealed reservation"):
        store._reserve_action(
            reservation,
            payload,
            _authority=object(),
        )
    with pytest.raises(RunActionStoreError, match="sealed recovery"):
        store._recovery_session(reservation, _authority=object())


def test_recovery_session_registration_failure_closes_pinned_descriptors(
    publisher_case,
    monkeypatch,
):
    _frontier, payload, reservation, _workspace = _reserved_action(publisher_case)
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, reservation, payload)
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
    _reserve_action(store, reservation, request_payload)
    with pytest.raises(RunActionStoreError, match="predecessor ledger moved"):
        _reserve_action(store, reservation, request_payload)

    assert store.snapshot().event_count == 1


def test_action_store_cancel_and_frontier_invalidation_are_terminal(
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
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        session.cancel()
        with pytest.raises(RunActionStoreError, match="terminal kind"):
            replace(
                session.events[-1],
                event_kind=RunActionExecutionEventKind.FRONTIER_INVALIDATED,
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
    _reserve_action(store, second_reservation, second_payload)
    with _open_session(store, second_reservation) as session:
        session.allocate_preparation(
            _execution_policy(
                kind=second_reservation.intent.kind,
                workspace_access=second_reservation.intent.workspace_access,
            )
        )
        session.invalidate_frontier()
    tails = {tail.operation_id: tail for tail in store.snapshot().operation_tails}
    assert tails[reservation.intent.operation_id].tail_kind is (
        RunActionExecutionEventKind.CANCELLED
    )
    assert tails[second_reservation.intent.operation_id].tail_kind is (
        RunActionExecutionEventKind.FRONTIER_INVALIDATED
    )
    assert workspace == second_workspace


@pytest.mark.parametrize("result_kind", ("result", "accepted"))
def test_action_store_rejects_request_substitution_and_result_corruption(
    publisher_case,
    result_kind,
):
    _frontier, request_payload, reservation, workspace = _reserved_action(
        publisher_case
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    with pytest.raises(RunActionStoreError, match="complete request"):
        _reserve_action(store, reservation, request_payload + b" altered")
    _reserve_action(store, reservation, request_payload)
    with _open_session(store, reservation) as session:
        _prepare_session(session)
        spawn = session.commit_spawn(
            security_observation_id=(reservation.frontier.security_observation_id),
            boundary_identity=session.reservation.intent.boundary_identity,
        )
        result = _record_captured_result(
            session,
            spawn,
            b'{"result":"durable"}',
            _frontier.checkpoint.safety_state.security_observation,
        )
        result_blob = result.result_blob
        if result_kind == "accepted":
            decision = session.decide_result(
                result_interpreter_identity=(
                    reservation.intent.boundary_identity.result_interpreter_identity
                ),
                disposition=RunActionResultDisposition.SUCCEEDED,
                accepted_result_payload=b'{"accepted":"durable"}',
                workspace_promotion=None,
            )
            session.accept_decision(workspace_after=workspace)
            result_blob = decision.accepted_result_blob
    result_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
        / f"{result_kind}-{result_blob.digest.removeprefix('sha256:')}.blob"
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
    _reserve_action(store, reservation, request_payload)
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
    _reserve_action(store, first, request_payload)
    with _open_session(store, first) as session:
        _prepare_session(session, container_id=shared_container_id)
        spawn = session.commit_spawn(
            security_observation_id=(
                frontier.checkpoint.safety_state.security_observation.observation_id
            ),
            boundary_identity=session.reservation.intent.boundary_identity,
        )
        result = _record_captured_result(
            session,
            spawn,
            b'{"result":"first"}',
            frontier.checkpoint.safety_state.security_observation,
        )
        session.decide_result(
            result_interpreter_identity=(
                first.intent.boundary_identity.result_interpreter_identity
            ),
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=b'{"accepted":"first"}',
            workspace_promotion=None,
        )
        session.accept_decision(
            workspace_after=workspace,
        )
    _frontier, second_payload, second, _workspace = _reserved_action(
        publisher_case,
        operation_id="second_provider_action_0123456789abcdef",
        frontier=frontier,
        workspace=workspace,
    )
    _reserve_action(store, second, second_payload)
    with _open_session(store, second) as session:
        policy = _execution_policy(
            kind=session.reservation.intent.kind,
            workspace_access=session.reservation.intent.workspace_access,
        )
        allocation = session.allocate_preparation(policy)
        operation_digest = hashlib.sha256(
            session.reservation.intent.operation_id.encode("utf-8")
        ).hexdigest()
        prepared = _prepared_execution(
            claim=allocation.preparation_claim,
            authority=allocation.runtime_volume_authority,
            container_id=shared_container_id,
            inode_offset=int(operation_digest[:8], 16),
        )
        with pytest.raises(RunActionStoreError, match="authority was reused"):
            session.commit_prepared_execution(
                prepared,
            )


def test_action_store_rejects_reused_runtime_volume_generation_authority(
    publisher_case,
    monkeypatch,
):
    frontier, request_payload, first, workspace = _reserved_action(
        publisher_case,
        operation_id="first_slot_action_0123456789abcdef",
    )
    store = _open_store(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _reserve_action(store, first, request_payload)
    with _open_session(store, first) as session:
        allocation = session.allocate_preparation(
            _execution_policy(
                kind=first.intent.kind,
                workspace_access=first.intent.workspace_access,
            )
        )
        first_nonce = allocation.runtime_volume_authority.generation_nonce
        session.invalidate_frontier()

    _frontier, second_payload, second, _workspace = _reserved_action(
        publisher_case,
        operation_id="second_slot_action_0123456789abcdef",
        frontier=frontier,
        workspace=workspace,
    )
    _reserve_action(store, second, second_payload)
    with _open_session(store, second) as session:
        policy = _execution_policy(
            kind=second.intent.kind,
            workspace_access=second.intent.workspace_access,
        )

        def allocation_with_reused_nonce(claim):
            return RunActionPreparationAllocation.mint(
                preparation_claim=claim,
                runtime_volume_authority=issue_runtime_volume_authority(
                    claim,
                    first_nonce,
                ),
            )

        monkeypatch.setattr(
            run_action_store_module,
            "_issue_preparation_allocation",
            allocation_with_reused_nonce,
        )
        with pytest.raises(RunActionStoreError, match="authority was reused"):
            session.allocate_preparation(policy)


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
    _reserve_action(store, reservation, request_payload)
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
