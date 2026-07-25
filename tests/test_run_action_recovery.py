"""Crash-seam tests for fail-closed durable run-action recovery."""

from __future__ import annotations

import os
import hashlib
from copy import copy
from dataclasses import replace

import pytest

import kapso.cross_run.launch.run_action_recovery as run_action_recovery_module
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpointStatus,
    RunCheckpointStop,
)
from kapso.cross_run.launch.resume_contracts import (
    RunEligibilityDisposition,
    RunSafetyBoundary,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionIntent,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_gate import RunFrontierActionGate
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_RECOVERY_IMPLEMENTATION_REGISTRY_AUTHORITY,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnQuery,
    RunActionCommittedSpawnState,
    RunActionInterpretedResult,
    RunActionPreparationMode,
    RunActionPreparationObservation,
    RunActionPreparationOrigin,
    RunActionPreparationState,
    RunActionProviderResult,
    RunActionRecoveryError,
    RunActionRecoveryImplementation,
    RunActionRecoveryImplementationRegistry,
    RunActionUnactivatedSpawnQuery,
    RunActionUnactivatedSpawnObservation,
    RunActionUnactivatedSpawnState,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionFrontierBinding,
    RunActionReservation,
    RunActionViewBinding,
)
from kapso.cross_run.launch.run_action_store import (
    RunActionExecutionEvent,
    RunActionResultDisposition,
    RunActionStoreError,
    RunActionTerminalReason,
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
    _record_result,
    _StaticSecurityAuthority,
)
from test_run_state_publisher import publisher_case
from test_run_action_supervisor_contracts import (
    _activation_revalidation_receipt,
    _execution_policy,
    _prepared_execution,
    _remint_contract,
    _result_capture_receipt,
    _spawn_commit,
    _terminal_observation,
)


class _FakeResultInterpreter:
    def __init__(
        self,
        result_interpreter_identity,
        *,
        disposition=RunActionResultDisposition.SUCCEEDED,
    ) -> None:
        self.result_interpreter_identity = result_interpreter_identity
        self.disposition = disposition
        self.interpret_calls = []

    def interpret(
        self,
        *,
        request_payload,
        result_payload,
    ):
        self.interpret_calls.append((request_payload, result_payload))
        return RunActionInterpretedResult(
            disposition=self.disposition,
            accepted_result_payload=b'{"accepted":"deterministic"}',
        )


class _FakeExecutionAdapter:
    def __init__(
        self,
        boundary_identity,
        *,
        observation_state=RunActionCommittedSpawnState.UNKNOWN,
        fail_activation=False,
        reattach_result=True,
        disposition=RunActionResultDisposition.SUCCEEDED,
    ) -> None:
        self.execution_lifecycle_identity = (
            boundary_identity.execution_lifecycle_identity
        )
        matching_policies = tuple(
            policy
            for workspace_access in RunFrontierWorkspaceAccess
            if (
                policy := _execution_policy(
                    kind=boundary_identity.kind,
                    workspace_access=workspace_access,
                )
            ).docker_execution_policy_id
            == self.execution_lifecycle_identity.execution_policy_id
        )
        if len(matching_policies) != 1:
            raise AssertionError("test boundary lacks one execution policy")
        self.execution_policy = matching_policies[0]
        self.result_interpreter = _FakeResultInterpreter(
            boundary_identity.result_interpreter_identity,
            disposition=disposition,
        )
        self.observation_state = observation_state
        self.fail_activation = fail_activation
        self.reattach_result = reattach_result
        self.prepare_calls = []
        self.preparation_capabilities = []
        self.prepare_allocations = []
        self.prepared_bound_allocations = []
        self.prepare_modes = []
        self.stage_calls = []
        self.start_calls = []
        self.start_workspace_descriptors = []
        self.inspect_calls = []
        self.reattach_calls = []

    @staticmethod
    def _prepared_for_allocation(allocation):
        claim = allocation.preparation_claim
        operation_digest = hashlib.sha256(
            claim.reservation.intent.operation_id.encode("utf-8")
        ).hexdigest()
        inode_offset = (
            int(
                allocation.runtime_volume_authority.generation_nonce,
                16,
            )
            - 1
        )
        return _prepared_execution(
            claim=claim,
            container_id=operation_digest,
            inode_offset=inode_offset,
        )

    @staticmethod
    def _provider_result(prepared, spawn, activation, payload):
        terminal = _terminal_observation(prepared, spawn)
        return RunActionProviderResult(
            terminal_observation=terminal,
            result_capture_receipt=_result_capture_receipt(
                prepared,
                activation,
                terminal,
                payload,
            ),
            result_payload=payload,
        )

    def prepared_event_size_bound(
        self,
        *,
        preparation_allocation,
        predecessor_event_id,
    ):
        self.prepared_bound_allocations.append(preparation_allocation)
        prepared = self._prepared_for_allocation(preparation_allocation)
        event = RunActionExecutionEvent.mint(
            event_number=3,
            predecessor_event_id=predecessor_event_id,
            event_kind=RunActionExecutionEventKind.EXECUTION_PREPARED,
            reservation=preparation_allocation.preparation_claim.reservation,
            preparation_allocation=None,
            prepared_execution=prepared,
            spawn_commit=None,
            activation_revalidation_receipt=None,
            result_receipt=None,
            acceptance=None,
            terminal_reason=None,
            workspace_after=None,
        )
        return len(event.to_json_bytes())

    def activation_event_size_bound(
        self,
        *,
        prepared_execution,
        spawn_commit,
        predecessor_event_id,
    ):
        activation = _activation_revalidation_receipt(
            prepared_execution,
            spawn_commit,
        )
        event = RunActionExecutionEvent.mint(
            event_number=5,
            predecessor_event_id=predecessor_event_id,
            event_kind=RunActionExecutionEventKind.ACTIVATION_COMMITTED,
            reservation=prepared_execution.preparation_claim.reservation,
            preparation_allocation=None,
            prepared_execution=None,
            spawn_commit=None,
            activation_revalidation_receipt=activation,
            result_receipt=None,
            acceptance=None,
            terminal_reason=None,
            workspace_after=None,
        )
        return len(event.to_json_bytes())

    def prepare(self, capability):
        self.preparation_capabilities.append(capability)
        allocation = capability.preparation_allocation
        claim = allocation.preparation_claim
        mode = capability.mode
        durable = capability.durable_prepared_execution
        workspace_descriptor = capability.workspace_descriptor
        workspace_source_path = capability.workspace_source_path
        if workspace_descriptor is not None:
            os.fstat(workspace_descriptor)
            assert workspace_source_path is not None
            assert os.stat(workspace_source_path, follow_symlinks=False).st_ino == (
                os.fstat(workspace_descriptor).st_ino
            )
        self.prepare_calls.append(claim.reservation)
        self.prepare_allocations.append(allocation)
        self.prepare_modes.append(mode)
        prepared = (
            durable
            if mode is RunActionPreparationMode.REVALIDATE_PREPARED
            else self._prepared_for_allocation(allocation)
        )
        origin = {
            RunActionPreparationMode.CREATE_ALLOCATED: (
                RunActionPreparationOrigin.NEWLY_MATERIALIZED
            ),
            RunActionPreparationMode.REOPEN_ALLOCATED: (
                RunActionPreparationOrigin.REOPENED_ALLOCATION
            ),
            RunActionPreparationMode.REVALIDATE_PREPARED: (
                RunActionPreparationOrigin.REVALIDATED_PREPARED
            ),
        }[mode]
        return RunActionPreparationObservation(
            state=RunActionPreparationState.EXACT_PREPARED,
            prepared_execution=prepared,
            origin=origin,
        )

    def stage_activation(self, capability):
        self.stage_calls.append(capability)
        if self.fail_activation:
            raise RuntimeError("injected death after durable spawn commit")
        return _activation_revalidation_receipt(
            capability.prepared_execution,
            capability.spawn_commit,
        )

    def activate_committed_once(self, capability):
        self.start_calls.append(capability)
        workspace_descriptor = capability.workspace_descriptor
        self.start_workspace_descriptors.append(workspace_descriptor)
        if workspace_descriptor is not None:
            os.fstat(workspace_descriptor)
        activation = capability.activation_revalidation_receipt
        return self._provider_result(
            capability.prepared_execution,
            capability.spawn_commit,
            activation,
            b'{"provider":"fresh-complete"}',
        )

    def inspect_unactivated(self, query):
        self.inspect_calls.append(query)
        state = (
            RunActionUnactivatedSpawnState.INERT_ACTIVATABLE
            if self.observation_state is RunActionCommittedSpawnState.INERT_ACTIVATABLE
            else RunActionUnactivatedSpawnState.UNKNOWN
        )
        return RunActionUnactivatedSpawnObservation(state=state)

    def inspect_committed(self, query):
        self.inspect_calls.append(query)
        if self.observation_state is RunActionCommittedSpawnState.RESULT_AVAILABLE:
            return RunActionCommittedSpawnObservation(
                state=self.observation_state,
                result=self._provider_result(
                    query.prepared_execution,
                    query.spawn_commit,
                    query.activation_revalidation_receipt,
                    b'{"provider":"recovered-complete"}',
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
        return self._provider_result(
            query.prepared_execution,
            query.spawn_commit,
            query.activation_revalidation_receipt,
            b'{"provider":"reattached-complete"}',
        )


def _recovery_registry(
    *execution_adapters,
) -> RunActionRecoveryImplementationRegistry:
    return RunActionRecoveryImplementationRegistry(
        tuple(
            RunActionRecoveryImplementation(
                boundary_identity=RunActionBoundaryIdentity.mint(
                    kind=execution_adapter.execution_lifecycle_identity.kind,
                    execution_lifecycle_identity=(
                        execution_adapter.execution_lifecycle_identity
                    ),
                    result_interpreter_identity=(
                        execution_adapter.result_interpreter.result_interpreter_identity
                    ),
                ),
                execution_adapter=execution_adapter,
                result_interpreter=execution_adapter.result_interpreter,
            )
            for execution_adapter in execution_adapters
        ),
        _authority=_RUN_ACTION_RECOVERY_IMPLEMENTATION_REGISTRY_AUTHORITY,
    )


def _recovery_coordinator(gate, *execution_adapters):
    return gate.recovery_coordinator(_recovery_registry(*execution_adapters))


class _NondeterministicResultInterpreter(_FakeResultInterpreter):
    def interpret(self, **arguments):
        interpreted = super().interpret(**arguments)
        if len(self.interpret_calls) == 1:
            return interpreted
        return RunActionInterpretedResult(
            disposition=interpreted.disposition,
            accepted_result_payload=b'{"accepted":"changed"}',
        )


class _SecurityAdvancingPrepareAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, advance_security) -> None:
        super().__init__(boundary_identity)
        self.advance_security = advance_security

    def prepare(self, capability):
        observation = super().prepare(capability)
        self.advance_security()
        return observation


class _ActivePreparationCapabilityAuditAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.active_clone_and_fork_rejected = False

    def prepare(self, capability):
        cloned = copy(capability)
        with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
            cloned.preparation_allocation
        read_descriptor, write_descriptor = os.pipe()
        child_process_id = os.fork()
        if child_process_id == 0:
            os.close(read_descriptor)
            with pytest.raises(
                RunActionRecoveryError,
                match="not in its one invocation",
            ):
                capability.preparation_allocation
            os.write(write_descriptor, b"invalid")
            os._exit(0)
        os.close(write_descriptor)
        assert os.read(read_descriptor, len(b"invalid")) == b"invalid"
        os.close(read_descriptor)
        waited_process_id, status = os.waitpid(child_process_id, 0)
        assert waited_process_id == child_process_id
        assert os.waitstatus_to_exitcode(status) == 0
        self.active_clone_and_fork_rejected = True
        return super().prepare(capability)


class _SecurityAdvancingStageAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, advance_security) -> None:
        super().__init__(boundary_identity)
        self.advance_security = advance_security

    def stage_activation(self, capability):
        activation = super().stage_activation(capability)
        self.advance_security()
        return activation


class _AdvancingOnCallSecurityAuthority:
    def __init__(self, current, advanced, advance_on_call) -> None:
        self.current = current
        self.advanced = advanced
        self.advance_on_call = advance_on_call
        self.call_count = 0

    def observe_exact_descendant_of(self, **_arguments):
        self.call_count += 1
        return (
            self.advanced if self.call_count >= self.advance_on_call else self.current
        )


class _CrashAfterPreparationAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.crash_after_preparation = True

    def prepare(self, capability):
        observation = super().prepare(capability)
        if self.crash_after_preparation:
            raise RuntimeError("injected death after provider preparation")
        return observation


class _CrashBeforeProviderStartAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.crash_before_provider_start = True
        self.activation_attempts = []

    def activate_committed_once(self, capability):
        self.activation_attempts.append(
            (capability.prepared_execution, capability.spawn_commit)
        )
        if self.crash_before_provider_start:
            raise RuntimeError("injected death before provider start")
        return super().activate_committed_once(capability)


class _OversizedPreparationEnvelopeAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, oversized_event_size_bytes) -> None:
        super().__init__(boundary_identity)
        self.oversized_event_size_bytes = oversized_event_size_bytes

    def prepared_event_size_bound(self, **_arguments):
        return self.oversized_event_size_bytes


class _OversizedActivationEnvelopeAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, oversized_event_size_bytes) -> None:
        super().__init__(boundary_identity)
        self.oversized_event_size_bytes = oversized_event_size_bytes

    def activation_event_size_bound(self, **_arguments):
        return self.oversized_event_size_bytes


class _NonExactPreparationAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, state, modes) -> None:
        super().__init__(boundary_identity)
        self.preparation_state = state
        self.nonexact_modes = set(modes)

    def prepare(self, capability):
        mode = capability.mode
        if mode not in self.nonexact_modes:
            return super().prepare(capability)
        allocation = capability.preparation_allocation
        claim = allocation.preparation_claim
        if capability.workspace_descriptor is not None:
            os.fstat(capability.workspace_descriptor)
        self.prepare_calls.append(claim.reservation)
        self.prepare_allocations.append(allocation)
        self.prepare_modes.append(mode)
        return RunActionPreparationObservation(
            state=self.preparation_state,
            prepared_execution=None,
            origin=None,
        )


class _ReplacementAllocationAdapter(_FakeExecutionAdapter):
    def prepare(self, capability):
        allocation = capability.preparation_allocation
        claim = allocation.preparation_claim
        replacement_nonce = (
            "0" * 31 + "2"
            if allocation.runtime_volume_authority.generation_nonce == "0" * 31 + "1"
            else "0" * 31 + "1"
        )
        self.prepare_calls.append(claim.reservation)
        self.prepare_allocations.append(allocation)
        self.prepare_modes.append(capability.mode)
        self.replacement_prepared = _prepared_execution(
            claim=claim,
            inode_offset=int(replacement_nonce, 16) - 1,
        )
        return RunActionPreparationObservation(
            state=RunActionPreparationState.EXACT_PREPARED,
            prepared_execution=self.replacement_prepared,
            origin=RunActionPreparationOrigin.NEWLY_MATERIALIZED,
        )


class _AccessGuardedExecutionAdapter(_FakeExecutionAdapter):
    _GUARDED_NAMES = frozenset(
        {
            "execution_lifecycle_identity",
            "execution_policy",
            "activation_event_size_bound",
            "activate_committed_once",
            "inspect_committed",
            "inspect_unactivated",
            "prepared_event_size_bound",
            "prepare",
            "reattach",
            "stage_activation",
        }
    )

    def __init__(self, boundary_identity) -> None:
        self.reject_execution_access = False
        super().__init__(boundary_identity)

    def __getattribute__(self, name):
        if name in type(self)._GUARDED_NAMES and object.__getattribute__(
            self, "reject_execution_access"
        ):
            raise AssertionError("result recovery accessed the execution adapter")
        return super().__getattribute__(name)


class _WorkspaceMutatingResultInterpreter(_FakeResultInterpreter):
    def __init__(self, result_interpreter_identity) -> None:
        super().__init__(result_interpreter_identity)
        self.retained_workspace_descriptor = None

    def interpret(self, **arguments):
        interpreted = super().interpret(**arguments)
        if len(self.interpret_calls) == 2:
            descriptor = os.open(
                "interpretation-mutation.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=self.retained_workspace_descriptor,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"mutated during interpretation")
        return interpreted


class _WorkspaceRetainingExecutionAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.result_interpreter = _WorkspaceMutatingResultInterpreter(
            boundary_identity.result_interpreter_identity
        )

    def activate_committed_once(self, capability):
        self.result_interpreter.retained_workspace_descriptor = os.dup(
            capability.workspace_descriptor
        )
        return super().activate_committed_once(capability)


def test_recovery_implementation_requires_distinct_lifecycle_and_interpreter() -> None:
    boundary_identity = _boundary_identity(RunFrontierActionKind.CODING_AGENT)
    combined = _FakeExecutionAdapter(boundary_identity)
    combined.result_interpreter_identity = boundary_identity.result_interpreter_identity

    with pytest.raises(RunActionRecoveryError, match="differs from its boundary"):
        RunActionRecoveryImplementation(
            boundary_identity=boundary_identity,
            execution_adapter=combined,
            result_interpreter=combined,
        )


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


def _commit_test_activation(gate, lease):
    prepared = lease._session.events[2].prepared_execution
    activation = _activation_revalidation_receipt(
        prepared,
        lease._spawn_commit,
    )
    gate.commit_activation(
        lease,
        activation_revalidation_receipt=activation,
    )
    return activation


def _leave_activation_committed(gate, permit, payload) -> None:
    with pytest.raises(RuntimeError, match="after activation"):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            _commit_test_activation(gate, lease)
            raise RuntimeError("injected death after activation")


def _leave_result_received(gate, permit, payload) -> bytes:
    raw_result = b'{"provider":"durable-result"}'
    with pytest.raises(RuntimeError, match="after result"):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            _record_result(gate, lease, raw_result)
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


def test_provider_result_rejects_payload_or_terminal_capture_splices():
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    activation = _activation_revalidation_receipt(prepared, spawn)
    result = _FakeExecutionAdapter._provider_result(
        prepared,
        spawn,
        activation,
        b'{"provider":"complete"}',
    )

    with pytest.raises(
        RunActionRecoveryError,
        match="terminal capture evidence",
    ):
        replace(result, result_payload=b'{"provider":"changed"}')
    foreign_prepared = _prepared_execution(inode_offset=9)
    foreign_spawn = _spawn_commit(
        foreign_prepared,
        invocation_nonce="2" * 32,
    )
    foreign_terminal = _terminal_observation(
        foreign_prepared,
        foreign_spawn,
    )
    with pytest.raises(
        RunActionRecoveryError,
        match="terminal capture evidence",
    ):
        replace(result, terminal_observation=foreign_terminal)


def test_committed_spawn_query_rejects_security_boundary_splice():
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    security_observation_id = spawn.security_observation_id
    substituted_security = (
        f"{security_observation_id.rsplit(':sha256:', 1)[0]}:sha256:" f"{'f' * 64}"
    )
    substituted_spawn = _remint_contract(
        spawn,
        security_observation_id=substituted_security,
    )

    with pytest.raises(RunActionRecoveryError, match="exact durable identity"):
        RunActionUnactivatedSpawnQuery(
            prepared_execution=prepared,
            spawn_commit=substituted_spawn,
        )


def test_reserved_action_recovers_through_one_preparation_and_activation(
    publisher_case,
) -> None:
    frontier, gate, _permit, _payload = _reserved_case(publisher_case)
    adapter = _FakeExecutionAdapter(
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


def test_preparation_capability_is_spent_and_clone_fork_invalid(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _ActivePreparationCapabilityAuditAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert adapter.active_clone_and_fork_rejected
    capability = adapter.preparation_capabilities[0]
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        capability.preparation_allocation
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        adapter.prepare(capability)
    cloned = copy(capability)
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        cloned.preparation_allocation
    read_descriptor, write_descriptor = os.pipe()
    child_process_id = os.fork()
    if child_process_id == 0:
        os.close(read_descriptor)
        with pytest.raises(
            RunActionRecoveryError,
            match="not in its one invocation",
        ):
            capability.preparation_allocation
        os.write(write_descriptor, b"invalid")
        os._exit(0)
    os.close(write_descriptor)
    assert os.read(read_descriptor, len(b"invalid")) == b"invalid"
    os.close(read_descriptor)
    waited_process_id, status = os.waitpid(child_process_id, 0)
    assert waited_process_id == child_process_id
    assert os.waitstatus_to_exitcode(status) == 0


def test_activation_capability_is_spent_and_clone_fork_invalid(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    capability = adapter.stage_calls[0]
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        capability.request_payload
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        adapter.stage_activation(capability)
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
    committed_capability = adapter.start_calls[0]
    with pytest.raises(RunActionRecoveryError, match="one invocation"):
        committed_capability.activation_event
    with pytest.raises(RunActionRecoveryError, match="one invocation"):
        adapter.activate_committed_once(committed_capability)


def test_legacy_direct_spawn_interfaces_are_removed() -> None:
    for name in (
        "RunActionPreparedSpawn",
        "RunActionFreshSpawnCapability",
        "activate_once",
        "prepare_fresh",
    ):
        assert not hasattr(run_action_recovery_module, name)
    assert not hasattr(RunFrontierActionGate, "claim")
    assert not hasattr(RunFrontierActionGate, "claim_preparation")
    assert not hasattr(
        RunActionExecutionEventKind,
        "PREPARATION_CLAIMED",
    )
    assert "preparation_claim" not in RunActionExecutionEvent.__dataclass_fields__
    assert not hasattr(
        run_action_recovery_module.RunActionPreparationCapability,
        "preparation_claim",
    )
    for name in ("ALLOCATE_ONCE", "REOPEN_CLAIM"):
        assert not hasattr(RunActionPreparationMode, name)
    for name in (
        "NEW_ALLOCATION",
        "REOPENED_CLAIM",
        "ALLOCATED_AFTER_PROVEN_ABSENCE",
    ):
        assert not hasattr(RunActionPreparationOrigin, name)


def test_allocation_crash_recovery_reuses_exact_durable_authority(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    allocation_crash = _CrashAfterPreparationAdapter(permit.intent.boundary_identity)
    coordinator = _recovery_coordinator(gate, allocation_crash)

    with pytest.raises(RuntimeError, match="provider preparation"):
        coordinator.recover(frontier)
    durable_allocation = (
        gate._action_store.inspect()
        .events_for(permit.intent.operation_id)[-1]
        .preparation_allocation
    )
    assert (
        gate._action_store.inspect()
        .events_for(permit.intent.operation_id)[-1]
        .event_kind
        is RunActionExecutionEventKind.PREPARATION_ALLOCATED
    )

    allocation_crash.crash_after_preparation = False
    assert coordinator.recover(frontier).is_complete
    assert allocation_crash.prepare_modes == [
        RunActionPreparationMode.CREATE_ALLOCATED,
        RunActionPreparationMode.REOPEN_ALLOCATED,
    ]
    assert allocation_crash.prepare_allocations == [
        durable_allocation,
        durable_allocation,
    ]
    assert allocation_crash.prepared_bound_allocations == [
        durable_allocation,
        durable_allocation,
        durable_allocation,
        durable_allocation,
    ]
    assert (
        allocation_crash.prepare_allocations[0].preparation_allocation_id
        == allocation_crash.prepare_allocations[1].preparation_allocation_id
    )
    assert (
        allocation_crash.prepare_allocations[0].runtime_volume_authority
        == allocation_crash.prepare_allocations[1].runtime_volume_authority
    )
    assert (
        allocation_crash.prepare_allocations[
            0
        ].runtime_volume_authority.generation_nonce
        == allocation_crash.prepare_allocations[
            1
        ].runtime_volume_authority.generation_nonce
    )
    prepared = (
        gate._action_store.inspect()
        .events_for(permit.intent.operation_id)[2]
        .prepared_execution
    )
    assert (
        prepared.runtime_volume_authority == durable_allocation.runtime_volume_authority
    )


def test_preparation_rejects_replacement_allocation_nonce(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _ReplacementAllocationAdapter(permit.intent.boundary_identity)

    with pytest.raises(RunActionRecoveryError, match="another prepared execution"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    allocation = (
        gate._action_store.inspect()
        .events_for(permit.intent.operation_id)[-1]
        .preparation_allocation
    )
    assert allocation is not None
    assert adapter.prepare_allocations == [allocation]
    assert (
        adapter.replacement_prepared.runtime_volume_authority
        != allocation.runtime_volume_authority
    )


def test_prepared_crash_recovery_uses_revalidation_mode(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    prepared_crash = _FakeExecutionAdapter(
        permit.intent.boundary_identity,
    )
    coordinator = _recovery_coordinator(gate, prepared_crash)
    security_check_count = 0

    def crash_after_prepared(_frontier):
        nonlocal security_check_count
        security_check_count += 1
        if security_check_count == 3:
            raise RuntimeError("injected death after prepared commit")
        return True

    coordinator._security_is_current = crash_after_prepared
    with pytest.raises(RuntimeError, match="after prepared commit"):
        coordinator.recover(frontier)
    assert (
        gate._action_store.inspect()
        .events_for(permit.intent.operation_id)[-1]
        .event_kind
        is RunActionExecutionEventKind.EXECUTION_PREPARED
    )

    coordinator._security_is_current = lambda _frontier: True
    assert coordinator.recover(frontier).is_complete
    assert prepared_crash.prepare_modes == [
        RunActionPreparationMode.CREATE_ALLOCATED,
        RunActionPreparationMode.REVALIDATE_PREPARED,
    ]


def test_allocated_resource_loss_terminally_closes_without_provider_authority(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _NonExactPreparationAdapter(
        permit.intent.boundary_identity,
        RunActionPreparationState.PROVEN_RESOURCE_LOST,
        {RunActionPreparationMode.CREATE_ALLOCATED},
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    terminal = report.recovered_operations[-1].events[-1]
    assert report.is_complete
    assert terminal.event_number == 3
    assert terminal.event_kind is RunActionExecutionEventKind.INTERRUPTED
    assert terminal.terminal_reason is (
        RunActionTerminalReason.SUPERVISOR_RESOURCE_LOST_BEFORE_SPAWN
    )
    assert not adapter.start_calls
    assert not adapter.inspect_calls


def test_prepared_resource_loss_interrupts_without_recreation(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _NonExactPreparationAdapter(
        permit.intent.boundary_identity,
        RunActionPreparationState.PROVEN_RESOURCE_LOST,
        {RunActionPreparationMode.REVALIDATE_PREPARED},
    )
    coordinator = _recovery_coordinator(gate, adapter)
    security_check_count = 0

    def crash_after_prepared(_frontier):
        nonlocal security_check_count
        security_check_count += 1
        if security_check_count == 3:
            raise RuntimeError("injected death after prepared commit")
        return True

    coordinator._security_is_current = crash_after_prepared
    with pytest.raises(RuntimeError, match="after prepared commit"):
        coordinator.recover(frontier)
    durable_prepared = (
        gate._action_store.inspect()
        .events_for(permit.intent.operation_id)[2]
        .prepared_execution
    )
    coordinator._security_is_current = lambda _frontier: True

    report = coordinator.recover(frontier)

    terminal = report.recovered_operations[-1].events[-1]
    assert report.is_complete
    assert terminal.event_number == 4
    assert terminal.terminal_reason is (
        RunActionTerminalReason.SUPERVISOR_RESOURCE_LOST_BEFORE_SPAWN
    )
    assert len(adapter.prepare_calls) == 2
    assert adapter.prepare_modes == [
        RunActionPreparationMode.CREATE_ALLOCATED,
        RunActionPreparationMode.REVALIDATE_PREPARED,
    ]
    assert (
        report.recovered_operations[-1].events[2].prepared_execution == durable_prepared
    )
    assert not adapter.start_calls


def test_unknown_preparation_remains_retryable_and_request_inaccessible(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _NonExactPreparationAdapter(
        permit.intent.boundary_identity,
        RunActionPreparationState.UNKNOWN,
        {
            RunActionPreparationMode.CREATE_ALLOCATED,
            RunActionPreparationMode.REOPEN_ALLOCATED,
        },
    )
    coordinator = _recovery_coordinator(gate, adapter)

    first = coordinator.recover(frontier)
    second = coordinator.recover(frontier)

    assert not first.is_complete
    assert not second.is_complete
    events = gate._action_store.inspect().events_for(permit.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.PREPARATION_ALLOCATED
    assert adapter.prepare_modes == [
        RunActionPreparationMode.CREATE_ALLOCATED,
        RunActionPreparationMode.REOPEN_ALLOCATED,
    ]
    assert not adapter.start_calls


def test_preparation_envelope_rejects_oversize_before_materialization(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _OversizedPreparationEnvelopeAdapter(
        permit.intent.boundary_identity,
        publisher_case["settings"].run_action_event_size_bytes + 1,
    )

    with pytest.raises(RunActionRecoveryError, match="event envelope"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(permit.intent.operation_id)
    assert events[-1].event_kind is (RunActionExecutionEventKind.PREPARATION_ALLOCATED)
    assert not adapter.prepare_calls
    assert not adapter.start_calls


def test_activation_envelope_rejects_oversize_before_delivery(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _OversizedActivationEnvelopeAdapter(
        permit.intent.boundary_identity,
        publisher_case["settings"].run_action_event_size_bytes + 1,
    )

    with pytest.raises(RunActionRecoveryError, match="activation event envelope"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(permit.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED
    assert not adapter.stage_calls
    assert not adapter.start_calls


def test_committed_inert_recovery_activates_same_spawn_without_repreparing(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _CrashBeforeProviderStartAdapter(
        permit.intent.boundary_identity,
    )
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RuntimeError, match="before provider start"):
        coordinator.recover(frontier)

    durable_prefix = gate._action_store.inspect().events_for(permit.intent.operation_id)
    assert durable_prefix[-1].event_kind is (
        RunActionExecutionEventKind.ACTIVATION_COMMITTED
    )
    durable_prepared = durable_prefix[2].prepared_execution
    durable_spawn = durable_prefix[3].spawn_commit

    adapter.crash_before_provider_start = False
    adapter.observation_state = RunActionCommittedSpawnState.INERT_ACTIVATABLE
    report = coordinator.recover(frontier)

    assert report.is_complete
    assert len(adapter.prepare_calls) == 1
    assert len(adapter.start_calls) == 1
    assert len(adapter.activation_attempts) == 2
    assert adapter.activation_attempts == [
        (durable_prepared, durable_spawn),
        (durable_prepared, durable_spawn),
    ]
    assert len(adapter.inspect_calls) == 1
    terminal_events = report.recovered_operations[-1].events
    assert terminal_events[:5] == durable_prefix


def test_failed_edit_recovery_terminates_with_unchanged_workspace(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"provider failure"}'
    permit = _issue_implementation_agent(
        gate,
        frontier,
        "failed_recovery",
        payload,
    )
    adapter = _FakeExecutionAdapter(
        permit.intent.boundary_identity,
        disposition=RunActionResultDisposition.FAILED,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    terminal = report.recovered_operations[-1].events[-1]
    assert report.is_complete
    assert terminal.acceptance.disposition is RunActionResultDisposition.FAILED
    assert (
        terminal.acceptance.workspace_after
        == terminal.reservation.frontier.workspace_before
    )


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
def test_committed_spawn_is_queried_and_never_reactivated(
    publisher_case,
    state,
    expected_reattach_calls,
    expected_terminal,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_activation_committed(gate, permit, payload)
    adapter = _FakeExecutionAdapter(
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
            _commit_test_activation(gate, lease)
            commit_sha = _commit_workspace_edit(
                publisher_case,
                "recovered.py",
                "RECOVERED = True\n",
            )
            raise RuntimeError("injected death after clean edit")
    adapter = _FakeExecutionAdapter(
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
            _commit_test_activation(gate, lease)
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
    adapter = _FakeExecutionAdapter(
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
    adapter = _FakeExecutionAdapter(
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
    _leave_activation_committed(gate, permit, payload)
    adapter = _FakeExecutionAdapter(
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
    _leave_activation_committed(gate, permit, payload)
    _advance_security(publisher_case, gate, frontier)
    adapter = _FakeExecutionAdapter(
        permit.intent.boundary_identity,
        observation_state=(RunActionCommittedSpawnState.RUNNING_REATTACHABLE),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(adapter.inspect_calls) == 1
    assert not adapter.reattach_calls
    assert not adapter.start_calls


def test_received_result_runs_only_pure_local_interpretation(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    raw_result = _leave_result_received(gate, permit, payload)
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert not adapter.prepare_calls
    assert not adapter.start_calls
    assert not adapter.inspect_calls
    assert not adapter.reattach_calls
    assert len(adapter.result_interpreter.interpret_calls) == 2
    assert adapter.result_interpreter.interpret_calls[0] == (payload, raw_result)


def test_received_result_does_not_access_execution_adapter(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    raw_result = _leave_result_received(gate, permit, payload)
    execution_adapter = _AccessGuardedExecutionAdapter(permit.intent.boundary_identity)
    coordinator = _recovery_coordinator(gate, execution_adapter)
    execution_adapter.reject_execution_access = True

    report = coordinator.recover(frontier)

    assert report.is_complete
    assert execution_adapter.result_interpreter.interpret_calls[0] == (
        payload,
        raw_result,
    )


def test_nondeterministic_local_interpretation_never_becomes_durable(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_result_received(gate, permit, payload)
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)
    adapter.result_interpreter = _NondeterministicResultInterpreter(
        permit.intent.boundary_identity.result_interpreter_identity
    )
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RunActionRecoveryError, match="nondeterministic"):
        coordinator.recover(frontier)

    plan = coordinator.inspect(frontier)
    assert plan.pending_operation_id == permit.intent.operation_id
    assert plan.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.RESULT_RECEIVED
    )


def test_workspace_mutation_during_interpretation_never_becomes_terminal(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _WorkspaceRetainingExecutionAdapter(
        permit.intent.boundary_identity,
    )
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RunWorkspaceFrontierError):
        coordinator.recover(frontier)

    os.close(adapter.result_interpreter.retained_workspace_descriptor)
    plan = coordinator.inspect(frontier)
    assert plan.pending_operation_id == permit.intent.operation_id
    assert plan.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.RESULT_RECEIVED
    )


def test_terminal_replay_reads_accepted_bytes_without_implementation_use(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    with gate.hold(permit, payload) as lease:
        _complete_action(gate, lease)
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert report.recovered_operations[0].accepted_result_payload == (
        b'{"accepted_result":"complete"}'
    )
    assert not adapter.prepare_calls
    assert not adapter.start_calls
    assert not adapter.inspect_calls
    assert not adapter.result_interpreter.interpret_calls


def test_activation_staging_crash_reopens_only_as_unactivated_spawn(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    crashing = _FakeExecutionAdapter(
        permit.intent.boundary_identity,
        fail_activation=True,
    )
    coordinator = _recovery_coordinator(gate, crashing)

    with pytest.raises(RuntimeError, match="durable spawn"):
        coordinator.recover(frontier)

    assert len(crashing.stage_calls) == 1
    assert not crashing.start_calls
    crashing.fail_activation = False
    crashing.observation_state = RunActionCommittedSpawnState.INERT_ACTIVATABLE
    report = coordinator.recover(frontier)
    assert report.is_complete
    assert len(crashing.start_calls) == 1
    assert len(crashing.inspect_calls) == 1


def test_security_advance_cancels_unspawned_reservation(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    _advance_security(publisher_case, gate, frontier)
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert report.recovered_operations[-1].events[-1].event_kind is (
        RunActionExecutionEventKind.CANCELLED
    )
    assert not adapter.prepare_calls


def test_security_advance_after_preparation_terminally_invalidates_frontier(
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
    terminal = gate._action_store.inspect().events_for(permit.intent.operation_id)[-1]
    assert terminal.event_kind is RunActionExecutionEventKind.INTERRUPTED
    assert terminal.terminal_reason is (
        RunActionTerminalReason.FRONTIER_INVALIDATED_BEFORE_SPAWN
    )
    assert terminal.workspace_after == terminal.reservation.frontier.workspace_before


def test_security_advance_during_allocation_fsync_prevents_materialization(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    required = frontier.checkpoint.safety_state.security_observation
    pin = publisher_case["active"].bootstrap_pin
    advanced = _security_observation(
        pin,
        required.checked_subject_ids,
        generation_offset=(
            required.generation
            - pin.launch_manifest.security_observation.generation
            + 1
        ),
    )
    authority = _AdvancingOnCallSecurityAuthority(
        required,
        advanced,
        advance_on_call=2,
    )
    gate._security_authority = authority
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert authority.call_count == 2
    assert not adapter.prepare_calls
    events = gate._action_store.inspect().events_for(permit.intent.operation_id)
    assert events[1].event_kind is RunActionExecutionEventKind.PREPARATION_ALLOCATED
    assert events[1].preparation_allocation is not None
    assert events[-1].event_kind is RunActionExecutionEventKind.INTERRUPTED
    assert events[-1].terminal_reason is (
        RunActionTerminalReason.FRONTIER_INVALIDATED_BEFORE_SPAWN
    )


def test_security_advance_during_activation_staging_never_starts_provider(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _SecurityAdvancingStageAdapter(
        permit.intent.boundary_identity,
        lambda: _advance_security(publisher_case, gate, frontier),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(adapter.stage_calls) == 1
    assert not adapter.start_calls
    events = gate._action_store.inspect().events_for(permit.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED


def test_security_advance_after_activation_commit_never_starts_provider(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    required = frontier.checkpoint.safety_state.security_observation
    pin = publisher_case["active"].bootstrap_pin
    advanced = _security_observation(
        pin,
        required.checked_subject_ids,
        generation_offset=(
            required.generation
            - pin.launch_manifest.security_observation.generation
            + 1
        ),
    )
    authority = _AdvancingOnCallSecurityAuthority(
        required,
        advanced,
        advance_on_call=5,
    )
    gate._security_authority = authority
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert authority.call_count == 5
    assert len(adapter.stage_calls) == 1
    assert not adapter.start_calls
    events = gate._action_store.inspect().events_for(permit.intent.operation_id)
    assert events[-1].event_kind is (RunActionExecutionEventKind.ACTIVATION_COMMITTED)


def test_workspace_advance_after_claim_terminally_invalidates_frontier(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _CrashAfterPreparationAdapter(permit.intent.boundary_identity)
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RuntimeError, match="provider preparation"):
        coordinator.recover(frontier)
    _commit_workspace_edit(
        publisher_case,
        "external-frontier.py",
        "EXTERNAL_FRONTIER = True\n",
    )
    adapter.crash_after_preparation = False

    report = coordinator.recover(frontier)

    terminal = report.recovered_operations[-1].events[-1]
    assert report.is_complete
    assert terminal.event_kind is RunActionExecutionEventKind.INTERRUPTED
    assert terminal.terminal_reason is (
        RunActionTerminalReason.FRONTIER_INVALIDATED_BEFORE_SPAWN
    )
    assert terminal.workspace_after == terminal.reservation.frontier.workspace_before
    assert len(adapter.prepare_calls) == 1
    assert not adapter.start_calls


def test_recovery_rejects_boundary_identity_substitution(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_result_received(gate, permit, payload)
    substituted = _FakeExecutionAdapter(
        _boundary_identity(RunFrontierActionKind.EMBEDDING)
    )

    coordinator = _recovery_coordinator(gate, substituted)
    with pytest.raises(RunActionRecoveryError, match="absent or substituted"):
        coordinator.recover(frontier)

    assert (
        coordinator.inspect(frontier).pending_operation_id == permit.intent.operation_id
    )


def test_recovery_rejects_same_identity_execution_object_substitution(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    original = _FakeExecutionAdapter(permit.intent.boundary_identity)
    substituted = _FakeExecutionAdapter(permit.intent.boundary_identity)
    implementation_registry = _recovery_registry(original)
    coordinator = gate.recovery_coordinator(implementation_registry)
    original_implementation = implementation_registry._implementations[0]
    implementation_registry._implementations = (
        RunActionRecoveryImplementation(
            boundary_identity=original_implementation.boundary_identity,
            execution_adapter=substituted,
            result_interpreter=original_implementation.result_interpreter,
        ),
    )

    with pytest.raises(RunActionRecoveryError, match="altered"):
        coordinator.recover(frontier)

    assert not original.prepare_calls
    assert not substituted.prepare_calls


def test_recovery_rejects_same_identity_interpreter_object_substitution(
    publisher_case,
) -> None:
    frontier, gate, permit, payload = _reserved_case(publisher_case)
    _leave_result_received(gate, permit, payload)
    execution_adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)
    implementation_registry = _recovery_registry(execution_adapter)
    coordinator = gate.recovery_coordinator(implementation_registry)
    implementation = implementation_registry._implementations[0]
    substituted = _FakeResultInterpreter(
        permit.intent.boundary_identity.result_interpreter_identity
    )
    object.__setattr__(implementation, "result_interpreter", substituted)

    with pytest.raises(RunActionRecoveryError, match="interpreter.*substituted"):
        coordinator.recover(frontier)

    assert not execution_adapter.prepare_calls
    assert not execution_adapter.inspect_calls
    assert not substituted.interpret_calls


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
    adapter = _FakeExecutionAdapter(original.intent.boundary_identity)

    with pytest.raises(RunActionRecoveryError, match="current frontier"):
        _recovery_coordinator(gate, adapter)._require_reservation_frontier(
            frontier,
            substituted_reservation,
        )


def test_recovery_rejects_exact_frontier_at_another_safety_boundary(
    publisher_case,
) -> None:
    frontier, gate, _permit, payload = _reserved_case(publisher_case)
    original = (
        gate._publisher._action_store.inspect()
        .operations_since(frontier.projection.action_ledger)[0][0]
        .reservation
    )
    substituted_intent = RunActionIntent.from_request(
        kind=original.intent.kind,
        boundary=RunSafetyBoundary.EVALUATION,
        operation_id=original.intent.operation_id,
        request_payload=payload,
        workspace_access=original.intent.workspace_access,
        boundary_identity=original.intent.boundary_identity,
    )
    substituted_reservation = RunActionReservation.build(
        intent=substituted_intent,
        frontier=original.frontier,
        predecessor_ledger=frontier.projection.action_ledger,
    )
    adapter = _FakeExecutionAdapter(original.intent.boundary_identity)

    with pytest.raises(RunActionRecoveryError, match="current frontier"):
        _recovery_coordinator(gate, adapter)._require_reservation_frontier(
            frontier,
            substituted_reservation,
        )


@pytest.mark.parametrize(
    "ineligible_state",
    ("stopped", "completed", "security_blocked"),
)
def test_recovery_rejects_a_nonactionable_current_frontier(
    publisher_case,
    ineligible_state,
) -> None:
    frontier, gate, _permit, _payload = _reserved_case(publisher_case)
    reservation = (
        gate._publisher._action_store.inspect()
        .operations_since(frontier.projection.action_ledger)[0][0]
        .reservation
    )
    checkpoint = object.__new__(type(frontier.checkpoint))
    for field_name, value in vars(frontier.checkpoint).items():
        object.__setattr__(checkpoint, field_name, value)
    if ineligible_state == "stopped":
        object.__setattr__(
            checkpoint,
            "last_stop",
            RunCheckpointStop.COST_BUDGET,
        )
    elif ineligible_state == "completed":
        object.__setattr__(
            checkpoint,
            "status",
            RunCheckpointStatus.COMPLETED,
        )
    else:
        safety_state = object.__new__(type(checkpoint.safety_state))
        for field_name, value in vars(checkpoint.safety_state).items():
            object.__setattr__(safety_state, field_name, value)
        object.__setattr__(
            safety_state,
            "disposition",
            RunEligibilityDisposition.SECURITY_BLOCKED,
        )
        object.__setattr__(checkpoint, "safety_state", safety_state)
    ineligible_frontier = object.__new__(type(frontier))
    for field_name, value in vars(frontier).items():
        object.__setattr__(ineligible_frontier, field_name, value)
    object.__setattr__(ineligible_frontier, "checkpoint", checkpoint)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    with pytest.raises(RunActionRecoveryError, match="current frontier"):
        _recovery_coordinator(gate, adapter)._require_reservation_frontier(
            ineligible_frontier,
            reservation,
        )


def test_recovery_coordinator_clone_and_forked_copy_are_invalid(
    publisher_case,
) -> None:
    frontier, gate, permit, _payload = _reserved_case(publisher_case)
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)
    implementation_registry = _recovery_registry(adapter)
    coordinator = gate.recovery_coordinator(implementation_registry)
    cloned = copy(coordinator)
    cloned_registry = copy(implementation_registry)

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
            implementation_registry.resolve_execution(permit.intent.boundary_identity)
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
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)
    report = _recovery_coordinator(reopened_gate, adapter).recover(reopened_frontier)

    assert report.is_complete
    assert adapter.result_interpreter.interpret_calls[0][1] == raw_result
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
    adapter = _FakeExecutionAdapter(permit.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert adapter.start_workspace_descriptors == [None]
