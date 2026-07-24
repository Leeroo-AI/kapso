"""Crash-seam tests for fail-closed durable run-action recovery."""

from __future__ import annotations

import os
from copy import copy

import pytest

from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_gate import RunFrontierActionGate
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_RECOVERY_ADAPTER_REGISTRY_AUTHORITY,
    RunActionAdapterAcceptance,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionPreparedSpawn,
    RunActionProviderResult,
    RunActionRecoveryAdapterRegistry,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_store import (
    RunActionFrontierBinding,
    RunActionReservation,
    RunActionResultDisposition,
    RunActionStoreError,
    RunActionViewBinding,
)
from kapso.cross_run.launch.run_state_publisher import RunStatePublisher
from kapso.cross_run.launch.workspace import StarterWorkspaceBuilder
from kapso.cross_run.launch.workspace_frontier import RunWorkspaceFrontierError
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import _security_observation
from test_run_frontier_action_gate import (
    _action_case,
    _boundary_identity,
    _claim_action,
    _commit_workspace_edit,
    _complete_action,
    _issue_ideation_agent,
    _issue_implementation_agent,
    _StaticSecurityAuthority,
)
from test_run_state_publisher import publisher_case


class _FakeRecoveryAdapter:
    def __init__(
        self,
        boundary_identity,
        *,
        observation_state=RunActionCommittedSpawnState.UNKNOWN,
        fail_fresh_start=False,
        reattach_result=True,
    ) -> None:
        self.boundary_identity = boundary_identity
        self.observation_state = observation_state
        self.fail_fresh_start = fail_fresh_start
        self.reattach_result = reattach_result
        self.prepare_calls = []
        self.start_calls = []
        self.start_workspace_descriptors = []
        self.inspect_calls = []
        self.reattach_calls = []
        self.accept_calls = []

    def prepare_fresh(self, reservation):
        self.prepare_calls.append(reservation)
        return RunActionPreparedSpawn(
            provider_execution_id=(f"recovered_{reservation.intent.operation_id}"),
            boundary_identity=self.boundary_identity,
        )

    def start_once(self, capability):
        self.start_calls.append(capability)
        workspace_descriptor = capability.workspace_descriptor
        self.start_workspace_descriptors.append(workspace_descriptor)
        if workspace_descriptor is not None:
            os.fstat(workspace_descriptor)
        if self.fail_fresh_start:
            raise RuntimeError("injected death after durable spawn commit")
        return RunActionProviderResult(result_payload=b'{"provider":"fresh-complete"}')

    def inspect_committed(self, query):
        self.inspect_calls.append(query)
        if self.observation_state is RunActionCommittedSpawnState.RESULT_AVAILABLE:
            return RunActionCommittedSpawnObservation(
                state=self.observation_state,
                result=RunActionProviderResult(
                    result_payload=b'{"provider":"recovered-complete"}'
                ),
                reattach_token=None,
            )
        if self.observation_state is RunActionCommittedSpawnState.RUNNING_REATTACHABLE:
            return RunActionCommittedSpawnObservation(
                state=self.observation_state,
                result=None,
                reattach_token="test.reattach.token",
            )
        return RunActionCommittedSpawnObservation(
            state=self.observation_state,
            result=None,
            reattach_token=None,
        )

    def reattach(self, query, observation):
        self.reattach_calls.append((query, observation))
        if not self.reattach_result:
            return None
        return RunActionProviderResult(
            result_payload=b'{"provider":"reattached-complete"}'
        )

    def accept_result(
        self,
        *,
        request_payload,
        result_payload,
        workspace_before,
        workspace_after,
    ):
        self.accept_calls.append(
            (
                request_payload,
                result_payload,
                workspace_before,
                workspace_after,
            )
        )
        return RunActionAdapterAcceptance(
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=b'{"accepted":"deterministic"}',
        )


def _recovery_registry(*adapters) -> RunActionRecoveryAdapterRegistry:
    return RunActionRecoveryAdapterRegistry(
        tuple(adapters),
        _authority=_RUN_ACTION_RECOVERY_ADAPTER_REGISTRY_AUTHORITY,
    )


def _recovery_coordinator(gate, *adapters):
    return gate.recovery_coordinator(_recovery_registry(*adapters))


class _NondeterministicAcceptanceAdapter(_FakeRecoveryAdapter):
    def accept_result(self, **arguments):
        acceptance = super().accept_result(**arguments)
        if len(self.accept_calls) == 1:
            return acceptance
        return RunActionAdapterAcceptance(
            disposition=acceptance.disposition,
            accepted_result_payload=b'{"accepted":"changed"}',
        )


class _SecurityAdvancingPrepareAdapter(_FakeRecoveryAdapter):
    def __init__(self, boundary_identity, advance_security) -> None:
        super().__init__(boundary_identity)
        self.advance_security = advance_security

    def prepare_fresh(self, reservation):
        prepared = super().prepare_fresh(reservation)
        self.advance_security()
        return prepared


class _WorkspaceMutatingAcceptanceAdapter(_FakeRecoveryAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.retained_workspace_descriptor = None

    def start_once(self, capability):
        self.retained_workspace_descriptor = os.dup(capability.workspace_descriptor)
        return super().start_once(capability)

    def accept_result(self, **arguments):
        acceptance = super().accept_result(**arguments)
        if len(self.accept_calls) == 2:
            descriptor = os.open(
                "acceptance-mutation.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=self.retained_workspace_descriptor,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"mutated during acceptance")
        return acceptance


def _reserved_case(case):
    _publisher, frontier, _security, gate = _action_case(case)
    payload = b'{"prompt":"recover-completely"}'
    permit = _issue_ideation_agent(gate, frontier, payload)
    return frontier, gate, permit, payload


def _leave_spawn_committed(gate, permit, payload) -> None:
    with pytest.raises(RuntimeError, match="after spawn"):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            raise RuntimeError("injected death after spawn")


def _leave_result_received(gate, permit, payload) -> bytes:
    raw_result = b'{"provider":"durable-result"}'
    with pytest.raises(RuntimeError, match="after result"):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            gate.record_result(lease, result_payload=raw_result)
            raise RuntimeError("injected death after result")
    return raw_result


def _advance_security(case, gate, frontier) -> None:
    required = frontier.checkpoint.safety_state.security_observation
    pin = case["active"].bootstrap_pin
    gate._security_authority.observation = _security_observation(
        pin,
        required.checked_subject_ids,
        generation_offset=(
            required.generation
            - pin.launch_manifest.security_observation.generation
            + 1
        ),
    )


def test_reserved_action_recovers_through_one_fresh_spawn(
    publisher_case,
) -> None:
    frontier, gate, _permit, _payload = _reserved_case(publisher_case)
    adapter = _FakeRecoveryAdapter(
        _boundary_identity(RunFrontierActionKind.CODING_AGENT)
    )
    coordinator = _recovery_coordinator(gate, adapter)

    plan = coordinator.inspect(frontier)
    report = coordinator.recover(frontier)

    assert plan.pending_operation_id == plan.ordered_operation_ids[-1]
    assert report.is_complete
    assert report.unresolved_operation_id is None
    assert len(adapter.prepare_calls) == 1
    assert adapter.prepare_calls[0].intent.operation_id == (
        report.recovered_operations[-1].operation_id
    )
    assert len(adapter.start_calls) == 1
    assert not adapter.inspect_calls
    assert report.recovered_operations[-1].accepted_result_payload == (
        b'{"accepted":"deterministic"}'
    )


def test_fresh_spawn_capability_is_spent_and_clone_fork_invalid(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _FakeRecoveryAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    capability = adapter.start_calls[0]
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        capability.request_payload
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        adapter.start_once(capability)
    assert len(adapter.start_workspace_descriptors) == 1
    cloned = copy(capability)
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        cloned.request_payload
    read_descriptor, write_descriptor = os.pipe()
    child_process_id = os.fork()
    if child_process_id == 0:
        os.close(read_descriptor)
        with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
            capability.request_payload
        os.write(write_descriptor, b"invalid")
        os._exit(0)
    os.close(write_descriptor)
    assert os.read(read_descriptor, len(b"invalid")) == b"invalid"
    os.close(read_descriptor)
    waited_process_id, status = os.waitpid(child_process_id, 0)
    assert waited_process_id == child_process_id
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.parametrize(
    ("state", "expected_reattach_calls", "expected_terminal"),
    (
        (
            RunActionCommittedSpawnState.RESULT_AVAILABLE,
            0,
            RunActionExecutionEventKind.RESULT_ACCEPTED,
        ),
        (
            RunActionCommittedSpawnState.RUNNING_REATTACHABLE,
            1,
            RunActionExecutionEventKind.RESULT_ACCEPTED,
        ),
        (
            RunActionCommittedSpawnState.PROVEN_QUIESCENT_WITHOUT_RESULT,
            0,
            RunActionExecutionEventKind.INTERRUPTED,
        ),
    ),
)
def test_committed_spawn_is_queried_and_never_freshly_replayed(
    publisher_case,
    state,
    expected_reattach_calls,
    expected_terminal,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_spawn_committed(gate, permit, payload)
    adapter = _FakeRecoveryAdapter(
        permit.intent.boundary_identity,
        observation_state=state,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert not adapter.prepare_calls
    assert not adapter.start_calls
    assert len(adapter.inspect_calls) == 1
    assert not hasattr(adapter.inspect_calls[0], "request_payload")
    assert len(adapter.reattach_calls) == expected_reattach_calls
    assert report.recovered_operations[-1].events[-1].event_kind is (expected_terminal)


def test_quiescent_edit_recovers_only_one_clean_direct_successor(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"recover edit"}'
    permit = _issue_implementation_agent(
        gate,
        frontier,
        "recovery",
        payload,
    )
    with pytest.raises(RuntimeError, match="after clean edit"):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            commit_sha = _commit_workspace_edit(
                publisher_case,
                "recovered.py",
                "RECOVERED = True\n",
            )
            raise RuntimeError("injected death after clean edit")
    adapter = _FakeRecoveryAdapter(
        permit.intent.boundary_identity,
        observation_state=(
            RunActionCommittedSpawnState.PROVEN_QUIESCENT_WITHOUT_RESULT
        ),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    terminal = report.recovered_operations[-1].events[-1]
    assert report.is_complete
    assert terminal.event_kind is RunActionExecutionEventKind.INTERRUPTED
    assert terminal.workspace_after.commit_sha == commit_sha


@pytest.mark.parametrize("workspace_state", ("dirty", "multiple_commits"))
def test_ambiguous_edit_workspace_remains_nonterminal(
    publisher_case,
    workspace_state,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"ambiguous edit"}'
    permit = _issue_implementation_agent(
        gate,
        frontier,
        workspace_state,
        payload,
    )
    with pytest.raises(RuntimeError, match="after ambiguous edit"):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            if workspace_state == "dirty":
                (publisher_case["active"].workspace / "uncommitted.py").write_text(
                    "DIRTY = True\n", encoding="utf-8"
                )
            else:
                _commit_workspace_edit(
                    publisher_case,
                    "first.py",
                    "FIRST = True\n",
                )
                _commit_workspace_edit(
                    publisher_case,
                    "second.py",
                    "SECOND = True\n",
                )
            raise RuntimeError("injected death after ambiguous edit")
    adapter = _FakeRecoveryAdapter(
        permit.intent.boundary_identity,
        observation_state=(
            RunActionCommittedSpawnState.PROVEN_QUIESCENT_WITHOUT_RESULT
        ),
    )
    expected_error = (
        RunWorkspaceFrontierError if workspace_state == "dirty" else RunActionStoreError
    )

    with pytest.raises(expected_error):
        _recovery_coordinator(gate, adapter).recover(frontier)

    assert (
        _recovery_coordinator(gate, adapter).inspect(frontier).pending_operation_id
        == permit.intent.operation_id
    )


def test_unknown_committed_spawn_remains_unresolved_without_replay(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_spawn_committed(gate, permit, payload)
    adapter = _FakeRecoveryAdapter(
        permit.intent.boundary_identity,
        observation_state=RunActionCommittedSpawnState.UNKNOWN,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert report.unresolved_operation_id == permit.intent.operation_id
    assert report.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.SPAWN_COMMITTED
    )
    assert not adapter.prepare_calls
    assert not adapter.start_calls
    assert not adapter.reattach_calls


def test_reattachment_without_a_result_remains_unresolved(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_spawn_committed(gate, permit, payload)
    adapter = _FakeRecoveryAdapter(
        permit.intent.boundary_identity,
        observation_state=(RunActionCommittedSpawnState.RUNNING_REATTACHABLE),
        reattach_result=False,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(adapter.reattach_calls) == 1
    assert not adapter.start_calls


def test_security_advance_prevents_committed_spawn_reattachment(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_spawn_committed(gate, permit, payload)
    _advance_security(publisher_case, gate, frontier)
    adapter = _FakeRecoveryAdapter(
        permit.intent.boundary_identity,
        observation_state=(RunActionCommittedSpawnState.RUNNING_REATTACHABLE),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(adapter.inspect_calls) == 1
    assert not adapter.reattach_calls
    assert not adapter.start_calls


def test_received_result_runs_only_deterministic_local_acceptance(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    raw_result = _leave_result_received(gate, permit, payload)
    adapter = _FakeRecoveryAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert not adapter.prepare_calls
    assert not adapter.start_calls
    assert not adapter.inspect_calls
    assert not adapter.reattach_calls
    assert len(adapter.accept_calls) == 2
    assert adapter.accept_calls[0][0] == payload
    assert adapter.accept_calls[0][1] == raw_result


def test_nondeterministic_local_acceptance_never_becomes_durable(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_result_received(gate, permit, payload)
    adapter = _NondeterministicAcceptanceAdapter(permit.intent.boundary_identity)
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RunActionRecoveryError, match="nondeterministic"):
        coordinator.recover(frontier)

    plan = coordinator.inspect(frontier)
    assert plan.pending_operation_id == permit.intent.operation_id
    assert plan.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.RESULT_RECEIVED
    )


def test_workspace_mutation_during_acceptance_never_becomes_terminal(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _WorkspaceMutatingAcceptanceAdapter(
        permit.intent.boundary_identity,
    )
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RunWorkspaceFrontierError):
        coordinator.recover(frontier)

    os.close(adapter.retained_workspace_descriptor)
    plan = coordinator.inspect(frontier)
    assert plan.pending_operation_id == permit.intent.operation_id
    assert plan.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.RESULT_RECEIVED
    )


def test_terminal_replay_reads_accepted_bytes_without_adapter_use(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    with gate.hold(permit, payload) as lease:
        _complete_action(gate, lease)
    adapter = _FakeRecoveryAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert report.recovered_operations[0].accepted_result_payload == (
        b'{"accepted_result":"complete"}'
    )
    assert not adapter.prepare_calls
    assert not adapter.start_calls
    assert not adapter.inspect_calls
    assert not adapter.accept_calls


def test_fresh_spawn_crash_is_reopened_only_as_committed_work(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    crashing = _FakeRecoveryAdapter(
        permit.intent.boundary_identity,
        fail_fresh_start=True,
    )
    coordinator = _recovery_coordinator(gate, crashing)

    with pytest.raises(RuntimeError, match="durable spawn"):
        coordinator.recover(frontier)

    assert len(crashing.start_calls) == 1
    crashing.fail_fresh_start = False
    crashing.observation_state = RunActionCommittedSpawnState.RESULT_AVAILABLE
    report = coordinator.recover(frontier)
    assert report.is_complete
    assert len(crashing.start_calls) == 1
    assert len(crashing.inspect_calls) == 1


def test_security_advance_cancels_unspawned_reservation(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    _advance_security(publisher_case, gate, frontier)
    adapter = _FakeRecoveryAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert report.recovered_operations[-1].events[-1].event_kind is (
        RunActionExecutionEventKind.CANCELLED
    )
    assert not adapter.prepare_calls


def test_security_advance_during_prepare_prevents_spawn_commit(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _SecurityAdvancingPrepareAdapter(
        permit.intent.boundary_identity,
        lambda: _advance_security(publisher_case, gate, frontier),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert len(adapter.prepare_calls) == 1
    assert not adapter.start_calls
    assert report.recovered_operations[-1].events[-1].event_kind is (
        RunActionExecutionEventKind.CANCELLED
    )


def test_recovery_rejects_adapter_identity_substitution(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_result_received(gate, permit, payload)
    substituted = _FakeRecoveryAdapter(
        _boundary_identity(RunFrontierActionKind.EMBEDDING)
    )

    coordinator = _recovery_coordinator(gate, substituted)
    with pytest.raises(RunActionRecoveryError, match="absent or substituted"):
        coordinator.recover(frontier)

    assert (
        coordinator.inspect(frontier).pending_operation_id == permit.intent.operation_id
    )


def test_recovery_rejects_same_identity_adapter_object_substitution(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    original = _FakeRecoveryAdapter(permit.intent.boundary_identity)
    substituted = _FakeRecoveryAdapter(permit.intent.boundary_identity)
    adapter_registry = _recovery_registry(original)
    coordinator = gate.recovery_coordinator(adapter_registry)
    adapter_registry._adapters = (substituted,)

    with pytest.raises(RunActionRecoveryError, match="altered"):
        coordinator.recover(frontier)

    assert not original.prepare_calls
    assert not substituted.prepare_calls


def test_recovery_rejects_substituted_frontier_view_binding(
    publisher_case,
) -> None:
    frontier, gate, _permit, _payload = _reserved_case(publisher_case)
    original = (
        gate._publisher._action_store.inspect()
        .operations_since(frontier.projection.action_ledger)[0][0]
        .reservation
    )
    first_view, *remaining_views = original.frontier.view_bindings
    substituted_frontier = RunActionFrontierBinding.mint(
        bootstrap_pin_id=original.frontier.bootstrap_pin_id,
        run_checkpoint_id=original.frontier.run_checkpoint_id,
        safety_state_id=original.frontier.safety_state_id,
        security_observation_id=original.frontier.security_observation_id,
        generation_id=original.frontier.generation_id,
        journal_head_id=original.frontier.journal_head_id,
        journal_size_bytes=original.frontier.journal_size_bytes,
        bundle_digest=original.frontier.bundle_digest,
        bundle_size_bytes=original.frontier.bundle_size_bytes,
        view_bindings=(
            RunActionViewBinding(
                relative_path=first_view.relative_path,
                digest=f"sha256:{'0' * 64}",
                size_bytes=first_view.size_bytes,
            ),
            *remaining_views,
        ),
        workspace_before=original.frontier.workspace_before,
    )
    substituted_reservation = RunActionReservation.build(
        intent=original.intent,
        frontier=substituted_frontier,
        predecessor_ledger=frontier.projection.action_ledger,
    )
    adapter = _FakeRecoveryAdapter(original.intent.boundary_identity)

    with pytest.raises(RunActionRecoveryError, match="current frontier"):
        _recovery_coordinator(gate, adapter)._require_reservation_frontier(
            frontier,
            substituted_reservation,
        )


def test_recovery_coordinator_clone_and_forked_copy_are_invalid(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _FakeRecoveryAdapter(permit.intent.boundary_identity)
    adapter_registry = _recovery_registry(adapter)
    coordinator = gate.recovery_coordinator(adapter_registry)
    cloned = copy(coordinator)
    cloned_registry = copy(adapter_registry)

    with pytest.raises(RunActionRecoveryError, match="cloned"):
        cloned.inspect(frontier)
    with pytest.raises(RunActionRecoveryError, match="cloned"):
        gate.recovery_coordinator(cloned_registry)
    read_descriptor, write_descriptor = os.pipe()
    child_process_id = os.fork()
    if child_process_id == 0:
        os.close(read_descriptor)
        with pytest.raises(RunActionRecoveryError, match="foreign"):
            coordinator.inspect(frontier)
        with pytest.raises(RunActionRecoveryError, match="foreign"):
            adapter_registry.resolve(adapter.boundary_identity)
        os.write(write_descriptor, b"invalid")
        os._exit(0)
    os.close(write_descriptor)
    assert os.read(read_descriptor, len(b"invalid")) == b"invalid"
    os.close(read_descriptor)
    waited_process_id, status = os.waitpid(child_process_id, 0)
    assert waited_process_id == child_process_id
    assert os.waitstatus_to_exitcode(status) == 0


def test_result_received_recovers_after_full_runtime_restart(
    resolver_case,
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    raw_result = _leave_result_received(gate, permit, payload)
    run_root = publisher_case["active"].run_root
    publisher_case["active"].close()

    settings = resolver_case["resolver"]._settings
    active = StarterWorkspaceBuilder(settings).reopen(run_root)
    publisher = RunStatePublisher(active, settings.launch)
    reopened_frontier = publisher.load_reconciled()
    security = _StaticSecurityAuthority(
        reopened_frontier.checkpoint.safety_state.security_observation
    )
    reopened_gate = RunFrontierActionGate(
        active_workspace=active,
        publisher=publisher,
        security_authority=security,
    )
    adapter = _FakeRecoveryAdapter(permit.intent.boundary_identity)
    report = _recovery_coordinator(reopened_gate, adapter).recover(reopened_frontier)

    assert report.is_complete
    assert adapter.accept_calls[0][1] == raw_result
    assert not adapter.inspect_calls
    active.close()


def test_fresh_embedding_recovery_never_receives_workspace_descriptor(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    payload = b'{"embedding":["complete input"]}'
    permit = gate.issue(
        frontier,
        kind=RunFrontierActionKind.EMBEDDING,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="embedding_recovery_0123456789abcdef",
        request_payload=payload,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        boundary_identity=_boundary_identity(RunFrontierActionKind.EMBEDDING),
    )
    adapter = _FakeRecoveryAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert adapter.start_workspace_descriptors == [None]
