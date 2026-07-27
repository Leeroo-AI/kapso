"""Crash-seam tests for fail-closed durable run-action recovery."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
from copy import copy
from contextlib import ExitStack
from dataclasses import replace

import pytest

import kapso.cross_run.launch.run_action_recovery as run_action_recovery_module
import kapso.cross_run.launch.run_action_workspace_promotion as promotion_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpointStatus,
    RunCheckpointStop,
)
from kapso.cross_run.launch.resume_contracts import (
    RunBranchAdvance,
    RunDerivativeFrontier,
    RunEligibilityDisposition,
    RunSafetyBoundary,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionIntent,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierRunningContainerObservation,
)
from kapso.cross_run.launch.run_action_gate import RunFrontierActionGate
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_RECOVERY_IMPLEMENTATION_REGISTRY_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnQuery,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionInterpretedResult,
    RunActionPreparationMode,
    RunActionPreparationObservation,
    RunActionPreparationOrigin,
    RunActionPreparationState,
    RunActionProviderResult,
    RunActionRecoveredOperation,
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
    _RUN_ACTION_RECOVERY_AUTHORITY,
    RunActionExecutionEvent,
    RunActionResultDisposition,
    RunActionStoreError,
)
from kapso.cross_run.launch.run_action_prepared_envelope import (
    prepared_execution_event_size_bound,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    issue_runtime_volume_authority,
    RunActionPreparationAllocation,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_pre_release_main_loss_observation_token,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from kapso.cross_run.launch.run_state_publisher import (
    RunStatePublisher,
    RunStatePublisherError,
)
from kapso.cross_run.launch.workspace import StarterWorkspaceBuilder
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
    RunWorkspaceFrontierError,
)
from kapso.cross_run.settings import CrossRunSettings
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import (
    _remint_evidence,
    _security_observation,
)
from test_run_frontier_action_gate import (
    _action_case,
    _boundary_identity,
    _commit_workspace_edit,
    _reserve_ideation_agent,
    _reserve_implementation_agent,
    _run_git,
    _static_resource_finalization_authority,
    _successor_at_boundary,
    _StaticSecurityAuthority,
)
from test_run_state_publisher import publisher_case
from test_run_action_release_contracts import _release_adoption_for_event
from test_run_action_supervisor_contracts import (
    _activation_revalidation_receipt,
    _boundary,
    _execution_policy,
    _prepared_execution,
    _remint_contract,
    _result_capture_receipt,
    _spawn_commit,
    _terminal_observation,
)
from test_run_action_docker_projection import _policy as _docker_projection_policy
from test_run_action_termination_contracts import (
    _pre_release_loss,
    _termination_graph,
    _timeout_publication,
)

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


class _AbsentReleaseInspection:
    topology = RunActionControlDirectoryTopology.EMPTY
    workload_release_adoption = None
    timeout_directive_publication = None

    def require_current(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False


class _PresentReleaseInspection:
    topology = RunActionControlDirectoryTopology.RELEASED
    timeout_directive_publication = None

    def __init__(self, adoption):
        self.adoption = adoption
        self.workload_release_adoption = adoption

    def require_current(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False


class _TimedOutReleaseInspection(_PresentReleaseInspection):
    topology = RunActionControlDirectoryTopology.TIMED_OUT

    def __init__(self, adoption, timeout_directive_publication):
        super().__init__(adoption)
        self.timeout_directive_publication = timeout_directive_publication


class _ChangingAbsentReleaseInspection(_AbsentReleaseInspection):
    def __init__(self, change_at_check) -> None:
        self.change_at_check = change_at_check
        self.current_checks = 0

    def require_current(self):
        self.current_checks += 1
        if self.current_checks == self.change_at_check:
            raise RuntimeError("release presence changed during retained inspection")


class _ChangingPresentReleaseInspection(_PresentReleaseInspection):
    def __init__(self, adoption, change_at_check) -> None:
        super().__init__(adoption)
        self.change_at_check = change_at_check
        self.current_checks = 0

    def require_current(self):
        self.current_checks += 1
        if self.current_checks == self.change_at_check:
            raise RuntimeError("release adoption changed during retained inspection")


@pytest.fixture(autouse=True)
def _synthetic_release_inspection(monkeypatch):
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_release_inspection",
        lambda **_arguments: _AbsentReleaseInspection(),
    )
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_timeout_inspection",
        lambda **_arguments: _AbsentReleaseInspection(),
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


class _ObservationTokenString(str):
    pass


class _FakeResultWorkspaceLease:
    def __init__(self, path) -> None:
        self._path = path
        self._descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        self._identity = (
            os.fstat(self._descriptor).st_dev,
            os.fstat(self._descriptor).st_ino,
        )
        self._closed = False

    @property
    def workspace_descriptor(self):
        self.require_current()
        return self._descriptor

    def require_current(self):
        if self._closed:
            raise AssertionError("fake result workspace lease is closed")
        current = os.stat(self._path, follow_symlinks=False)
        assert (current.st_dev, current.st_ino) == self._identity
        assert (
            os.fstat(self._descriptor).st_dev,
            os.fstat(self._descriptor).st_ino,
        ) == self._identity

    def __enter__(self):
        self.require_current()
        return self

    def __exit__(self, *_arguments):
        os.close(self._descriptor)
        self._closed = True


def _isolated_edit_candidate(
    case,
    tmp_path,
    *,
    candidate_name="isolated-candidate",
    edit_name="isolated-edit.py",
):
    candidate = tmp_path / candidate_name
    shutil.copytree(case["active"].workspace, candidate)
    candidate.chmod(0o700)
    (candidate / edit_name).write_text(
        "ISOLATED_EDIT = True\n",
        encoding="utf-8",
    )
    _run_git(candidate, "add", "--", edit_name)
    _run_git(
        candidate,
        "-c",
        "user.name=Kapso Test",
        "-c",
        "user.email=kapso-test@example.invalid",
        "commit",
        "-m",
        "Apply isolated edit",
    )
    return candidate


class _FakeExecutionAdapter:
    def __init__(
        self,
        boundary_identity,
        *,
        observation_state=RunActionCommittedSpawnState.INERT_CONTINUABLE,
        fail_activation=False,
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
        self.prepare_calls = []
        self.preparation_capabilities = []
        self.prepare_allocations = []
        self.prepared_bound_allocations = []
        self.prepare_modes = []
        self.stage_calls = []
        self.continuation_calls = []
        self.inspect_calls = []
        self.security_observation = None

    @staticmethod
    def _prepared_for_allocation(allocation):
        claim = allocation.preparation_claim
        operation_digest = hashlib.sha256(
            claim.reservation.intent.operation_id.encode("utf-8")
        ).hexdigest()
        inode_offset = int(
            hashlib.sha256(
                allocation.runtime_volume_authority.generation_nonce.encode("ascii")
            ).hexdigest()[:12],
            16,
        )
        return _prepared_execution(
            claim=claim,
            authority=allocation.runtime_volume_authority,
            container_id=operation_digest,
            inode_offset=inode_offset,
        )

    @staticmethod
    def _provider_result(
        prepared,
        spawn,
        activation,
        payload,
        workload_release_adoption=None,
    ):
        terminal = _terminal_observation(
            prepared,
            spawn,
            workload_release_adoption,
        )
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
            provider_termination_receipt=None,
            result_receipt=None,
            result_decision=None,
            acceptance=None,
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
            provider_termination_receipt=None,
            result_receipt=None,
            result_decision=None,
            acceptance=None,
            workspace_after=None,
        )
        return len(event.to_json_bytes())

    def release_receipt_size_bound(self, *, reservation):
        assert (
            reservation.intent.boundary_identity.execution_lifecycle_identity
            == self.execution_lifecycle_identity
        )
        return self.execution_policy.supervisor_limits.release_receipt_size_bytes

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

    def continue_committed_once(self, capability):
        self.continuation_calls.append(capability)
        if (
            capability.observation.state
            is RunActionCommittedSpawnState.INERT_CONTINUABLE
        ):
            observation_token = capability.observation.observation_token
            query, sealed_token = capability._take_start_authority(
                observation_token,
                _authority=run_action_recovery_module._RUN_ACTION_START_AUTHORITY,
            )
            assert sealed_token == observation_token
            running = RunActionBarrierRunningContainerObservation.mint(
                container_id=query.spawn_commit.provider_execution_id,
                observed_inspect_projection=(
                    query.prepared_execution.inert_container_evidence.issued_create_projection
                ),
                complete_inspection_digest=tree_or_blob_digest(
                    b"fake blocked run-action barrier"
                ),
                container_status="running",
                init_process_id=4242,
                restart_count=0,
                started_at="2026-07-25T01:02:03.123456789Z",
                finished_at="0001-01-01T00:00:00Z",
                paused=False,
                restarting=False,
                dead=False,
                oom_killed=False,
                state_error="",
            )
            capability._complete_start(
                running,
                sealed_token,
                _authority=run_action_recovery_module._RUN_ACTION_START_AUTHORITY,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=None,
            )
        if (
            capability.observation.state
            is RunActionCommittedSpawnState.RUNNING_CONTINUABLE
        ):
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=None,
            )
        raise AssertionError(
            "terminal fake must use the trusted terminal and result-capture leaves"
        )

    def inspect_unactivated(self, query):
        self.inspect_calls.append(query)
        state = (
            RunActionUnactivatedSpawnState.INERT_ACTIVATABLE
            if self.observation_state is RunActionCommittedSpawnState.INERT_CONTINUABLE
            else RunActionUnactivatedSpawnState.UNKNOWN
        )
        return RunActionUnactivatedSpawnObservation(state=state)

    def inspect_committed(self, query):
        self.inspect_calls.append(query)
        token = (
            hashlib.sha256(
                (
                    f"{query.spawn_commit.spawn_commit_id}:"
                    f"{self.observation_state.value}:{len(self.inspect_calls)}"
                ).encode("utf-8")
            ).hexdigest()
            if self.observation_state is not RunActionCommittedSpawnState.UNKNOWN
            else None
        )
        return RunActionCommittedSpawnObservation(
            state=self.observation_state,
            observation_token=None if token is None else f"sha256:{token}",
        )


class _PublicationLockAuditAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, checkpoint_lock_path) -> None:
        super().__init__(boundary_identity)
        self.checkpoint_lock_path = checkpoint_lock_path
        self.exclusive_lock_rejections = 0

    def continue_committed_once(self, capability):
        with self.checkpoint_lock_path.open("r+b", buffering=0) as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        self.exclusive_lock_rejections += 1
        return super().continue_committed_once(capability)


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
    for execution_adapter in execution_adapters:
        execution_adapter.security_observation = gate._security_authority.observation
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


class _ActiveContinuationCapabilityAuditAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.active_clone_and_fork_rejected = False

    def continue_committed_once(self, capability):
        cloned = copy(capability)
        with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
            cloned.observation
        read_descriptor, write_descriptor = os.pipe()
        child_process_id = os.fork()
        if child_process_id == 0:
            os.close(read_descriptor)
            with pytest.raises(
                RunActionRecoveryError,
                match="not in its one invocation",
            ):
                capability.observation
            os.write(write_descriptor, b"invalid")
            os._exit(0)
        os.close(write_descriptor)
        assert os.read(read_descriptor, len(b"invalid")) == b"invalid"
        os.close(read_descriptor)
        waited_process_id, status = os.waitpid(child_process_id, 0)
        assert waited_process_id == child_process_id
        assert os.waitstatus_to_exitcode(status) == 0
        self.active_clone_and_fork_rejected = True
        return super().continue_committed_once(capability)


class _TokenSealingExecutionAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.sealed_query = None
        self.sealed_observation = None
        self.token_revalidated = False

    def inspect_committed(self, query):
        observation = super().inspect_committed(query)
        self.sealed_query = query
        self.sealed_observation = observation
        return observation

    def continue_committed_once(self, capability):
        assert capability.query is self.sealed_query
        assert capability.observation is self.sealed_observation
        assert (
            capability.observation.observation_token
            == self.sealed_observation.observation_token
        )
        self.token_revalidated = True
        return super().continue_committed_once(capability)


class _PendingContinuationAdapter(_FakeExecutionAdapter):
    def continue_committed_once(self, capability):
        self.continuation_calls.append(capability)
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PENDING,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        )


class _PreReleasePublicationFenceSource:
    def __init__(self, change_at_check=None) -> None:
        self.change_at_check = change_at_check
        self.current_checks = 0
        self.closed = False

    def require_current(self):
        if self.closed:
            raise AssertionError("test pre-release publication fence is closed")
        self.current_checks += 1
        if self.current_checks == self.change_at_check:
            raise RuntimeError("pre-release main absence changed")

    def close(self):
        if self.closed:
            raise AssertionError("test pre-release publication fence closed twice")
        self.closed = True


class _TrustedPreReleaseTerminationAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, *, fence_change_at_check=None) -> None:
        super().__init__(
            boundary_identity,
            observation_state=(
                RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE
            ),
        )
        self.loss_observation = None
        self.termination_receipt = None
        self.publication_fence_source = _PreReleasePublicationFenceSource(
            fence_change_at_check
        )

    def inspect_committed(self, query):
        self.inspect_calls.append(query)
        self.loss_observation = _pre_release_loss(
            query.activation_revalidation_receipt,
            query.activation_event.event_id,
        )
        return RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
            observation_token=run_action_pre_release_main_loss_observation_token(
                self.loss_observation
            ),
        )

    def continue_committed_once(self, capability):
        self.continuation_calls.append(capability)
        query, terminal, loss_observation_id = (
            capability._take_provider_termination_authority(
                _authority=(
                    run_action_recovery_module._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY
                ),
            )
        )
        assert terminal is None
        assert (
            query.activation_event.event_id == self.loss_observation.activation_event_id
        )
        assert (
            loss_observation_id
            == run_action_pre_release_main_loss_observation_token(self.loss_observation)
        )
        self.termination_receipt = RunActionProviderTerminationReceipt.mint(
            disposition=RunActionProviderTerminationDisposition.FAILED,
            reason=RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
            activation_event_id=query.activation_event.event_id,
            workload_release_adoption=None,
            terminal_observation=None,
            timeout_directive_publication=None,
            empty_result_capture_receipt=None,
            pre_release_main_loss_observation=self.loss_observation,
        )
        publication_fence = run_action_recovery_module.RunActionProviderTerminationPublicationFence(
            source=self.publication_fence_source,
            _authority=(
                run_action_recovery_module._RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY
            ),
        )
        capability._complete_provider_termination(
            self.termination_receipt,
            publication_fence,
            _authority=(
                run_action_recovery_module._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY
            ),
        )
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PROVIDER_TERMINATED,
            result=None,
            provider_termination_receipt=self.termination_receipt,
            timeout_directive_publication=None,
            provider_termination_publication_fence=publication_fence,
        )


class _TrustedReleasedTerminationAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(
            boundary_identity,
            observation_state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
        )
        self.terminal = None
        self.termination_receipt = None

    def inspect_committed(self, query):
        self.inspect_calls.append(query)
        self.terminal = _remint_contract(
            _terminal_observation(
                query.prepared_execution,
                query.spawn_commit,
                query.workload_release_adoption,
            ),
            exit_code=137,
            oom_killed=True,
        )
        return RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            observation_token=self.terminal.complete_inspection_digest,
        )

    def continue_committed_once(self, capability):
        self.continuation_calls.append(capability)
        query, observation_token = capability._take_terminal_inspection_authority(
            _authority=(
                run_action_recovery_module._RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY
            ),
        )
        assert observation_token == self.terminal.complete_inspection_digest
        capability._complete_terminal_inspection(
            self.terminal,
            _authority=(
                run_action_recovery_module._RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY
            ),
        )
        termination_query, retained_terminal, loss_observation_id = (
            capability._take_provider_termination_authority(
                _authority=(
                    run_action_recovery_module._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY
                ),
            )
        )
        assert termination_query == query
        assert retained_terminal == self.terminal
        assert loss_observation_id is None
        self.termination_receipt = RunActionProviderTerminationReceipt.mint(
            disposition=RunActionProviderTerminationDisposition.FAILED,
            reason=RunActionProviderTerminationReason.OOM,
            activation_event_id=query.activation_event.event_id,
            workload_release_adoption=query.workload_release_adoption,
            terminal_observation=self.terminal,
            timeout_directive_publication=None,
            empty_result_capture_receipt=None,
            pre_release_main_loss_observation=None,
        )
        capability._complete_provider_termination(
            self.termination_receipt,
            _authority=(
                run_action_recovery_module._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY
            ),
        )
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PROVIDER_TERMINATED,
            result=None,
            provider_termination_receipt=self.termination_receipt,
            timeout_directive_publication=None,
        )


class _TrustedTimeoutTerminationAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(
            boundary_identity,
            observation_state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
        )
        self.terminal = None
        self.termination_receipt = None

    def inspect_committed(self, query):
        self.inspect_calls.append(query)
        assert (
            query.control_directory_topology
            is RunActionControlDirectoryTopology.TIMED_OUT
        )
        assert query.timeout_directive_publication is not None
        self.terminal = _terminal_observation(
            query.prepared_execution,
            query.spawn_commit,
            query.workload_release_adoption,
        )
        return RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            observation_token=self.terminal.complete_inspection_digest,
        )

    def continue_committed_once(self, capability):
        self.continuation_calls.append(capability)
        query, observation_token = capability._take_terminal_inspection_authority(
            _authority=(
                run_action_recovery_module._RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY
            ),
        )
        assert observation_token == self.terminal.complete_inspection_digest
        capability._complete_terminal_inspection(
            self.terminal,
            _authority=(
                run_action_recovery_module._RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY
            ),
        )
        termination_query, retained_terminal, loss_observation_id = (
            capability._take_provider_termination_authority(
                _authority=(
                    run_action_recovery_module._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY
                ),
            )
        )
        assert termination_query == query
        assert retained_terminal == self.terminal
        assert loss_observation_id is None
        self.termination_receipt = RunActionProviderTerminationReceipt.mint(
            disposition=RunActionProviderTerminationDisposition.INTERRUPTED,
            reason=RunActionProviderTerminationReason.TIMEOUT,
            activation_event_id=query.activation_event.event_id,
            workload_release_adoption=query.workload_release_adoption,
            terminal_observation=self.terminal,
            timeout_directive_publication=query.timeout_directive_publication,
            empty_result_capture_receipt=None,
            pre_release_main_loss_observation=None,
        )
        capability._complete_provider_termination(
            self.termination_receipt,
            _authority=(
                run_action_recovery_module._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY
            ),
        )
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PROVIDER_TERMINATED,
            result=None,
            provider_termination_receipt=self.termination_receipt,
            timeout_directive_publication=None,
        )


class _SecurityAdvancingStageAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, advance_security) -> None:
        super().__init__(boundary_identity)
        self.advance_security = advance_security

    def stage_activation(self, capability):
        activation = super().stage_activation(capability)
        self.advance_security()
        return activation


class _AdoptionAwarePendingAdapter(_FakeExecutionAdapter):
    def continue_committed_once(self, capability):
        self.observed_release_adoption = capability.workload_release_adoption
        self.continuation_calls.append(capability)
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PENDING,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        )


class _AdvancingOnCallSecurityAuthority:
    def __init__(self, current, advanced, advance_on_call) -> None:
        self.current = current
        self.advanced = advanced
        self.advance_on_call = advance_on_call
        self.call_count = 0

    @property
    def observation(self):
        return (
            self.advanced if self.call_count >= self.advance_on_call else self.current
        )

    def observe_exact_descendant_of(self, **_arguments):
        self.call_count += 1
        return (
            self.advanced if self.call_count >= self.advance_on_call else self.current
        )


class _WorkspaceMutatingContinuationAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, workspace_path) -> None:
        super().__init__(boundary_identity)
        self.workspace_path = workspace_path

    def continue_committed_once(self, capability):
        outcome = super().continue_committed_once(capability)
        (self.workspace_path / "continuation-mutation.txt").write_text(
            "mutated by execution adapter\n",
            encoding="utf-8",
        )
        return outcome


class _CrashAfterPreparationAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.crash_after_preparation = True

    def prepare(self, capability):
        observation = super().prepare(capability)
        if self.crash_after_preparation:
            raise RuntimeError("injected death after provider preparation")
        return observation


class _CrashBeforeProviderContinuationAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity) -> None:
        super().__init__(boundary_identity)
        self.crash_before_provider_start = True
        self.continuation_attempts = []

    def continue_committed_once(self, capability):
        self.continuation_attempts.append(
            (capability.prepared_execution, capability.spawn_commit)
        )
        if self.crash_before_provider_start:
            raise RuntimeError("injected death before provider start")
        return super().continue_committed_once(capability)


class _OversizedPreparationEnvelopeAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, oversized_event_size_bytes) -> None:
        super().__init__(boundary_identity)
        self.oversized_event_size_bytes = oversized_event_size_bytes

    def prepared_event_size_bound(self, **_arguments):
        return self.oversized_event_size_bytes


class _ProductionPreparationEnvelopeAuditAdapter(_FakeExecutionAdapter):
    def __init__(
        self,
        boundary_identity,
        execution_policy,
        command,
        runtime_settings,
    ) -> None:
        super().__init__(
            _boundary_identity(
                boundary_identity.kind,
                execution_policy.filesystem_policy.workspace_access,
            )
        )
        self.execution_lifecycle_identity = (
            boundary_identity.execution_lifecycle_identity
        )
        self.execution_policy = execution_policy
        self.result_interpreter = _FakeResultInterpreter(
            boundary_identity.result_interpreter_identity
        )
        self.command = command
        self.runtime_settings = runtime_settings
        self.prepared_envelope_calls = []
        self.materialization_attempts = []

    def prepared_event_size_bound(
        self,
        *,
        preparation_allocation,
        predecessor_event_id,
    ):
        self.prepared_envelope_calls.append(
            (preparation_allocation, predecessor_event_id)
        )
        return prepared_execution_event_size_bound(
            preparation_allocation=preparation_allocation,
            predecessor_event_id=predecessor_event_id,
            command=self.command,
            runtime_settings=self.runtime_settings,
        )

    def prepare(self, capability):
        self.materialization_attempts.append(capability)
        raise AssertionError("production envelope admitted provider materialization")


class _OversizedActivationEnvelopeAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, oversized_event_size_bytes) -> None:
        super().__init__(boundary_identity)
        self.oversized_event_size_bytes = oversized_event_size_bytes

    def activation_event_size_bound(self, **_arguments):
        return self.oversized_event_size_bytes


class _InvalidReleaseEnvelopeAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, release_receipt_size_bound) -> None:
        super().__init__(boundary_identity)
        self.invalid_release_receipt_size_bound = release_receipt_size_bound

    def release_receipt_size_bound(self, **_arguments):
        return self.invalid_release_receipt_size_bound


class _ActivationCrowdingReleaseEnvelopeAdapter(_FakeExecutionAdapter):
    def __init__(self, boundary_identity, activation_event_size_bound) -> None:
        super().__init__(boundary_identity)
        self.crowding_activation_event_size_bound = activation_event_size_bound

    def activation_event_size_bound(self, **_arguments):
        return self.crowding_activation_event_size_bound


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
            "continue_committed_once",
            "inspect_committed",
            "inspect_unactivated",
            "prepared_event_size_bound",
            "release_receipt_size_bound",
            "prepare",
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


class _AccessGuardedResultInterpreter(_FakeResultInterpreter):
    def __init__(self, result_interpreter_identity) -> None:
        self.reject_interpreter_access = False
        super().__init__(result_interpreter_identity)

    def __getattribute__(self, name):
        if name in {
            "interpret",
            "result_interpreter_identity",
        } and object.__getattribute__(
            self,
            "reject_interpreter_access",
        ):
            raise AssertionError("decided-result recovery accessed the interpreter")
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
    reservation = _reserve_ideation_agent(gate, frontier, payload)
    return frontier, gate, reservation, payload


def _append_spawn_committed(gate, reservation):
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
        allocation = session.allocate_preparation(adapter.execution_policy)
        prepared = adapter._prepared_for_allocation(allocation)
        session.commit_prepared_execution(prepared)
        return session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )


def _append_activation_committed(gate, reservation):
    spawn = _append_spawn_committed(gate, reservation)
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        prepared = session.events[2].prepared_execution
        activation = _activation_revalidation_receipt(prepared, spawn)
        session.commit_activation(activation)
        return activation


def _append_result_received(gate, reservation) -> bytes:
    raw_result = b'{"provider":"durable-result"}'
    activation = _append_activation_committed(gate, reservation)
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        prepared = session.events[2].prepared_execution
        spawn = session.events[3].spawn_commit
        workload_release_adoption = _release_adoption_for_event(
            session.events[4],
            gate._security_authority.observation,
        )
        result = _FakeExecutionAdapter._provider_result(
            prepared,
            spawn,
            activation,
            raw_result,
            workload_release_adoption,
        )
        session.record_result(
            spawn_commit=spawn,
            workload_release_adoption=workload_release_adoption,
            terminal_observation=result.terminal_observation,
            result_capture_receipt=result.result_capture_receipt,
            result_payload=result.result_payload,
        )
    return raw_result


def _append_provider_terminated(gate, reservation) -> None:
    activation = _append_activation_committed(gate, reservation)
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        prepared = session.events[2].prepared_execution
        spawn = session.events[3].spawn_commit
        adoption = _release_adoption_for_event(
            session.events[4],
            gate._security_authority.observation,
        )
        terminal = _remint_contract(
            _terminal_observation(prepared, spawn, adoption),
            exit_code=137,
            oom_killed=True,
        )
        session.terminate_provider(
            RunActionProviderTerminationReceipt.mint(
                disposition=RunActionProviderTerminationDisposition.FAILED,
                reason=RunActionProviderTerminationReason.OOM,
                activation_event_id=session.events[4].event_id,
                workload_release_adoption=adoption,
                terminal_observation=terminal,
                timeout_directive_publication=None,
                empty_result_capture_receipt=None,
                pre_release_main_loss_observation=None,
            )
        )
        assert activation == session.events[4].activation_revalidation_receipt


def _append_result_decided(gate, reservation) -> bytes:
    raw_result = _append_result_received(gate, reservation)
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        session.decide_result(
            result_interpreter_identity=(
                reservation.intent.boundary_identity.result_interpreter_identity
            ),
            disposition=RunActionResultDisposition.SUCCEEDED,
            accepted_result_payload=b'{"accepted_result":"complete"}',
            workspace_promotion=None,
        )
    return raw_result


def _append_result_accepted(gate, reservation) -> bytes:
    raw_result = _append_result_decided(gate, reservation)
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        workspace_binding = reservation.frontier.workspace_before
        session.accept_decision(
            workspace_after=(
                None if workspace_binding is None else workspace_binding.to_identity()
            ),
        )
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
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _FakeExecutionAdapter(
        _boundary_identity(RunFrontierActionKind.CODING_AGENT)
    )
    coordinator = _recovery_coordinator(gate, adapter)

    plan = coordinator.inspect(frontier)
    report = coordinator.recover(frontier)

    assert plan.pending_operation_id == plan.ordered_operation_ids[-1]
    assert not report.is_complete
    assert report.unresolved_operation_id == reservation.intent.operation_id
    assert len(adapter.prepare_calls) == 1
    assert adapter.prepare_calls[0].intent.operation_id == (
        reservation.intent.operation_id
    )
    assert len(adapter.continuation_calls) == 1
    assert len(adapter.inspect_calls) == 1
    assert report.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.ACTIVATION_COMMITTED
    )


def test_preparation_capability_is_spent_and_clone_fork_invalid(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _ActivePreparationCapabilityAuditAdapter(
        reservation.intent.boundary_identity
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
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
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    capability = adapter.stage_calls[0]
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        capability.request_payload
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        adapter.stage_activation(capability)
    assert not hasattr(adapter.continuation_calls[0], "workspace_descriptor")
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
    committed_capability = adapter.continuation_calls[0]
    with pytest.raises(RunActionRecoveryError, match="one invocation"):
        committed_capability.activation_event
    with pytest.raises(RunActionRecoveryError, match="one invocation"):
        adapter.continue_committed_once(committed_capability)


def test_committed_continuation_capability_is_active_only_once(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _ActiveContinuationCapabilityAuditAdapter(
        reservation.intent.boundary_identity
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert adapter.active_clone_and_fork_rejected
    capability = adapter.continuation_calls[0]
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        capability.observation
    with pytest.raises(RunActionRecoveryError, match="not in its one invocation"):
        adapter.continue_committed_once(capability)


def test_committed_inspection_carries_only_state_and_token() -> None:
    assert set(RunActionCommittedSpawnObservation.__dataclass_fields__) == {
        "state",
        "observation_token",
    }
    with pytest.raises(RunActionRecoveryError, match="payload differs"):
        RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            observation_token=None,
        )
    with pytest.raises(RunActionRecoveryError, match="token is invalid"):
        RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            observation_token="test.result.bytes",
        )
    for token in (
        1,
        _ObservationTokenString(f"sha256:{'f' * 64}"),
    ):
        with pytest.raises(RunActionRecoveryError, match="token is invalid"):
            RunActionCommittedSpawnObservation(
                state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
                observation_token=token,
            )


def test_committed_query_rejects_allocation_and_activation_splices(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    allocation = events[1].preparation_allocation
    replacement_nonce = (
        "e" * 32
        if allocation.runtime_volume_authority.generation_nonce == "f" * 32
        else "f" * 32
    )
    replacement_allocation = RunActionPreparationAllocation.mint(
        preparation_claim=allocation.preparation_claim,
        runtime_volume_authority=issue_runtime_volume_authority(
            allocation.preparation_claim,
            replacement_nonce,
        ),
    )

    with pytest.raises(RunActionRecoveryError, match="durable activation"):
        RunActionCommittedSpawnQuery(
            preparation_allocation=replacement_allocation,
            activation_event=events[4],
            workload_release_adoption=None,
            timeout_directive_publication=None,
        )

    foreign_prepared = _prepared_execution(inode_offset=11)
    foreign_spawn = _spawn_commit(
        foreign_prepared,
        invocation_nonce="d" * 32,
    )
    spliced_event = _remint_contract(
        events[4],
        activation_revalidation_receipt=_activation_revalidation_receipt(
            foreign_prepared,
            foreign_spawn,
        ),
    )
    with pytest.raises(RunActionRecoveryError, match="durable activation"):
        RunActionCommittedSpawnQuery(
            preparation_allocation=allocation,
            activation_event=spliced_event,
            workload_release_adoption=None,
            timeout_directive_publication=None,
        )


def test_committed_query_derives_exact_empty_released_and_timed_out_topology(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    allocation = events[1].preparation_allocation
    activation_event = events[4]
    adoption = _release_adoption_for_event(
        activation_event,
        frontier.checkpoint.safety_state.security_observation,
    )
    publication = _timeout_publication(
        activation_event.activation_revalidation_receipt,
        adoption,
    )

    empty = RunActionCommittedSpawnQuery(
        preparation_allocation=allocation,
        activation_event=activation_event,
        workload_release_adoption=None,
        timeout_directive_publication=None,
    )
    released = RunActionCommittedSpawnQuery(
        preparation_allocation=allocation,
        activation_event=activation_event,
        workload_release_adoption=adoption,
        timeout_directive_publication=None,
    )
    timed_out = RunActionCommittedSpawnQuery(
        preparation_allocation=allocation,
        activation_event=activation_event,
        workload_release_adoption=adoption,
        timeout_directive_publication=publication,
    )

    assert empty.control_directory_topology is RunActionControlDirectoryTopology.EMPTY
    assert (
        released.control_directory_topology
        is RunActionControlDirectoryTopology.RELEASED
    )
    assert (
        timed_out.control_directory_topology
        is RunActionControlDirectoryTopology.TIMED_OUT
    )
    with pytest.raises(RunActionRecoveryError, match="control topology"):
        RunActionCommittedSpawnQuery(
            preparation_allocation=allocation,
            activation_event=activation_event,
            workload_release_adoption=None,
            timeout_directive_publication=publication,
        )
    foreign_publication = _termination_graph(
        RunActionProviderTerminationReason.TIMEOUT
    ).timeout_directive_publication
    with pytest.raises(RunActionRecoveryError, match="control topology"):
        RunActionCommittedSpawnQuery(
            preparation_allocation=allocation,
            activation_event=activation_event,
            workload_release_adoption=adoption,
            timeout_directive_publication=foreign_publication,
        )


def test_committed_continuation_reuses_exact_inspection_seal(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _TokenSealingExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert adapter.token_revalidated


def test_recovery_contract_has_no_proof_free_terminal_states() -> None:
    assert set(RunActionPreparationState) == {
        RunActionPreparationState.EXACT_PREPARED,
        RunActionPreparationState.UNKNOWN,
    }
    assert set(RunActionCommittedSpawnState) == {
        RunActionCommittedSpawnState.INERT_CONTINUABLE,
        RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
        RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
        RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
        RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE,
        RunActionCommittedSpawnState.UNKNOWN,
    }
    assert set(RunActionContinuationState) == {
        RunActionContinuationState.PENDING,
        RunActionContinuationState.TIMEOUT_PUBLISHED,
        RunActionContinuationState.RESULT_CAPTURED,
        RunActionContinuationState.PROVIDER_TERMINATED,
    }


def test_terminal_continuation_cannot_skip_trusted_reinspection(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    adapter = _PendingContinuationAdapter(
        reservation.intent.boundary_identity,
        observation_state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
    )

    with pytest.raises(
        RunActionRecoveryError,
        match="terminal continuation lacks its trusted reinspection",
    ):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.ACTIVATION_COMMITTED


def test_legacy_direct_spawn_interfaces_are_removed() -> None:
    for name in (
        "RunActionPreparedSpawn",
        "RunActionFreshSpawnCapability",
        "activate_once",
        "prepare_fresh",
        "RunActionCommittedActivationCapability",
    ):
        assert not hasattr(run_action_recovery_module, name)
    assert not hasattr(
        run_action_recovery_module.RunActionExecutionAdapter,
        "activate_committed_once",
    )
    assert not hasattr(
        run_action_recovery_module.RunActionExecutionAdapter,
        "reattach",
    )
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


def test_recovery_callback_holds_shared_publication_lock(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    checkpoint_lock_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_checkpoint_lock_path
    )
    adapter = _PublicationLockAuditAdapter(
        reservation.intent.boundary_identity,
        checkpoint_lock_path,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert adapter.exclusive_lock_rejections == 1
    with checkpoint_lock_path.open("r+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_allocation_crash_recovery_reuses_exact_durable_authority(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    allocation_crash = _CrashAfterPreparationAdapter(
        reservation.intent.boundary_identity
    )
    coordinator = _recovery_coordinator(gate, allocation_crash)

    with pytest.raises(RuntimeError, match="provider preparation"):
        coordinator.recover(frontier)
    durable_allocation = (
        gate._action_store.inspect()
        .events_for(reservation.intent.operation_id)[-1]
        .preparation_allocation
    )
    assert (
        gate._action_store.inspect()
        .events_for(reservation.intent.operation_id)[-1]
        .event_kind
        is RunActionExecutionEventKind.PREPARATION_ALLOCATED
    )

    allocation_crash.crash_after_preparation = False
    assert not coordinator.recover(frontier).is_complete
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
        .events_for(reservation.intent.operation_id)[2]
        .prepared_execution
    )
    assert (
        prepared.runtime_volume_authority == durable_allocation.runtime_volume_authority
    )


def test_preparation_rejects_replacement_allocation_nonce(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _ReplacementAllocationAdapter(reservation.intent.boundary_identity)

    with pytest.raises(RunActionRecoveryError, match="another prepared execution"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    allocation = (
        gate._action_store.inspect()
        .events_for(reservation.intent.operation_id)[-1]
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
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    prepared_crash = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
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
        .events_for(reservation.intent.operation_id)[-1]
        .event_kind
        is RunActionExecutionEventKind.EXECUTION_PREPARED
    )

    coordinator._security_is_current = lambda _frontier: True
    assert not coordinator.recover(frontier).is_complete
    assert prepared_crash.prepare_modes == [
        RunActionPreparationMode.CREATE_ALLOCATED,
        RunActionPreparationMode.REVALIDATE_PREPARED,
    ]


def test_unknown_preparation_remains_retryable_and_request_inaccessible(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _NonExactPreparationAdapter(
        reservation.intent.boundary_identity,
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
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.PREPARATION_ALLOCATED
    assert adapter.prepare_modes == [
        RunActionPreparationMode.CREATE_ALLOCATED,
        RunActionPreparationMode.REOPEN_ALLOCATED,
    ]
    assert not adapter.continuation_calls


def test_preparation_envelope_rejects_oversize_before_materialization(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _OversizedPreparationEnvelopeAdapter(
        reservation.intent.boundary_identity,
        publisher_case["settings"].run_action_event_size_bytes + 1,
    )

    with pytest.raises(RunActionRecoveryError, match="event envelope"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is (RunActionExecutionEventKind.PREPARATION_ALLOCATED)
    assert not adapter.prepare_calls
    assert not adapter.continuation_calls


def test_production_preparation_envelope_rejects_one_byte_before_mutation(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    docker_settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker
    command = DockerRunActionCommand.build(
        entrypoint="/bin/tool",
        arguments=("default",),
    )
    execution_policy = _docker_projection_policy(
        docker_settings,
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        command_template_id=command.command_template_id,
    )
    boundary_identity = _boundary(
        RunFrontierActionKind.CODING_AGENT,
        execution_policy_id=execution_policy.docker_execution_policy_id,
    )
    reservation = gate.reserve(
        frontier,
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="agent_call_fedcba9876543210fedcba9876543210",
        request_payload=b'{"prompt":"production envelope"}',
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        boundary_identity=boundary_identity,
    )
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        allocation = session.allocate_preparation(execution_policy)
        predecessor_event_id = session.events[-1].event_id
    durable_events = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )
    bound = prepared_execution_event_size_bound(
        preparation_allocation=allocation,
        predecessor_event_id=predecessor_event_id,
        command=command,
        runtime_settings=docker_settings,
    )
    assert max(len(event.to_json_bytes()) for event in durable_events) < bound
    adapter = _ProductionPreparationEnvelopeAuditAdapter(
        boundary_identity,
        execution_policy,
        command,
        docker_settings,
    )
    coordinator = _recovery_coordinator(gate, adapter)
    object.__setattr__(
        publisher_case["settings"],
        "run_action_event_size_bytes",
        bound - 1,
    )

    with pytest.raises(
        RunActionRecoveryError,
        match="preparation event envelope",
    ):
        coordinator.recover(frontier)

    final_events = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )
    assert tuple(event.event_kind for event in final_events) == (
        RunActionExecutionEventKind.INTENT_RESERVED,
        RunActionExecutionEventKind.PREPARATION_ALLOCATED,
    )
    assert adapter.prepared_envelope_calls == [
        (allocation, predecessor_event_id),
        (allocation, predecessor_event_id),
    ]
    assert not adapter.materialization_attempts
    assert not adapter.prepare_calls
    assert not adapter.stage_calls
    assert not adapter.continuation_calls


def test_activation_envelope_rejects_oversize_before_delivery(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _OversizedActivationEnvelopeAdapter(
        reservation.intent.boundary_identity,
        publisher_case["settings"].run_action_event_size_bytes + 1,
    )

    with pytest.raises(RunActionRecoveryError, match="activation event envelope"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED
    assert not adapter.stage_calls
    assert not adapter.continuation_calls


@pytest.mark.parametrize("case", ("oversized", "snapshot_equal"))
def test_release_receipt_envelope_rejects_before_allocation(
    publisher_case,
    case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    settings = publisher_case["settings"]
    invalid_bound = (
        settings.run_action_release_receipt_size_bytes + 1
        if case == "oversized"
        else settings.run_action_process_snapshot_size_bytes
    )
    adapter = _InvalidReleaseEnvelopeAdapter(
        reservation.intent.boundary_identity,
        invalid_bound,
    )

    with pytest.raises(RunActionRecoveryError, match="release-receipt envelope"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.INTENT_RESERVED
    assert not adapter.prepare_calls
    assert not adapter.stage_calls
    assert not adapter.continuation_calls


def test_execution_envelope_rejects_timeout_policy_config_mismatch(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    settings = publisher_case["settings"]
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    coordinator = _recovery_coordinator(gate, adapter)
    gate._publisher._settings = replace(
        settings,
        run_action_timeout_directive_size_bytes=(
            settings.run_action_timeout_directive_size_bytes
            + settings.run_action_process_snapshot_size_bytes
        ),
    )

    with pytest.raises(RunActionRecoveryError, match="timeout envelope"):
        coordinator._release_receipt_size_bound(adapter, reservation)
    assert coordinator.inspect(frontier).pending_operation_id == (
        reservation.intent.operation_id
    )


def test_activation_bound_must_leave_resolved_release_envelope(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    settings = publisher_case["settings"]
    adapter = _ActivationCrowdingReleaseEnvelopeAdapter(
        reservation.intent.boundary_identity,
        (
            settings.run_action_release_receipt_size_bytes
            - settings.run_action_process_snapshot_size_bytes
        ),
    )

    with pytest.raises(
        RunActionRecoveryError,
        match="cannot fit the release-receipt envelope",
    ):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED
    assert not adapter.stage_calls
    assert not adapter.continuation_calls


def test_committed_inert_recovery_activates_same_spawn_without_repreparing(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _CrashBeforeProviderContinuationAdapter(
        reservation.intent.boundary_identity,
    )
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RuntimeError, match="before provider start"):
        coordinator.recover(frontier)

    durable_prefix = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )
    assert durable_prefix[-1].event_kind is (
        RunActionExecutionEventKind.ACTIVATION_COMMITTED
    )
    durable_prepared = durable_prefix[2].prepared_execution
    durable_spawn = durable_prefix[3].spawn_commit

    adapter.crash_before_provider_start = False
    adapter.observation_state = RunActionCommittedSpawnState.INERT_CONTINUABLE
    report = coordinator.recover(frontier)

    assert not report.is_complete
    assert len(adapter.prepare_calls) == 1
    assert len(adapter.continuation_calls) == 1
    assert len(adapter.continuation_attempts) == 2
    assert adapter.continuation_attempts == [
        (durable_prepared, durable_spawn),
        (durable_prepared, durable_spawn),
    ]
    assert len(adapter.inspect_calls) == 2
    unresolved_events = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )
    assert unresolved_events == durable_prefix


def test_failed_edit_recovery_terminates_with_unchanged_workspace(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"provider failure"}'
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "failed_recovery",
        payload,
    )
    adapter = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
        disposition=RunActionResultDisposition.FAILED,
    )
    _append_result_received(gate, reservation)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    events = report.recovered_operations[-1].events
    terminal = events[-1]
    assert report.is_complete
    assert events[-2].result_decision.disposition is RunActionResultDisposition.FAILED
    assert (
        terminal.acceptance.workspace_after
        == terminal.reservation.frontier.workspace_before
    )


def test_successful_edit_recovery_promotes_isolated_result_workspace(
    publisher_case,
    tmp_path,
    monkeypatch,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "successful_edit_promotion",
        b'{"implementation":"isolated success"}',
    )
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    candidate = _isolated_edit_candidate(publisher_case, tmp_path)
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_result_workspace",
        lambda _prepared, _capture: _FakeResultWorkspaceLease(candidate),
    )
    _append_result_received(gate, reservation)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    events = report.recovered_operations[-1].events
    assert report.is_complete
    assert events[-2].event_kind is RunActionExecutionEventKind.RESULT_DECIDED
    assert events[-1].event_kind is RunActionExecutionEventKind.RESULT_ACCEPTED
    promotion = events[-2].result_decision.workspace_promotion
    assert promotion is not None
    assert events[-1].acceptance.workspace_after == promotion.candidate_workspace
    assert len(adapter.result_interpreter.interpret_calls) == 2
    with ExitStack() as descriptors:
        workspace_descriptor, _identity = publisher_case[
            "active"
        ]._open_execution_workspace(descriptors)
        observed_workspace = inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=publisher_case["settings"],
            expected_commit_sha=promotion.candidate_workspace.commit_sha,
        )
    assert promotion.candidate_workspace.to_identity() == observed_workspace
    staging = (
        publisher_case["active"].run_root
        / publisher_case[
            "active"
        ].bootstrap_pin.installation_receipt.layout.run_action_workspace_staging_relative_path
    )
    assert tuple(staging.iterdir()) == ()


def test_durable_edit_decision_recovers_without_provider_or_interpreter(
    publisher_case,
    tmp_path,
    monkeypatch,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "durable_edit_decision",
        b'{"implementation":"durable decision"}',
    )
    adapter = _AccessGuardedExecutionAdapter(reservation.intent.boundary_identity)
    guarded_interpreter = _AccessGuardedResultInterpreter(
        reservation.intent.boundary_identity.result_interpreter_identity
    )
    adapter.result_interpreter = guarded_interpreter
    candidate = _isolated_edit_candidate(publisher_case, tmp_path)
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_result_workspace",
        lambda _prepared, _capture: _FakeResultWorkspaceLease(candidate),
    )
    _append_result_received(gate, reservation)
    original = gate._action_store._publish_result_event

    def publish_decision_then_die(*arguments, **keywords):
        original(*arguments, **keywords)
        if keywords["event"].event_kind is RunActionExecutionEventKind.RESULT_DECIDED:
            raise RuntimeError("injected death after durable edit decision")

    monkeypatch.setattr(
        gate._action_store,
        "_publish_result_event",
        publish_decision_then_die,
    )
    coordinator = _recovery_coordinator(gate, adapter)
    with pytest.raises(RuntimeError, match="durable edit decision"):
        coordinator.recover(frontier)
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.RESULT_DECIDED
    assert events[-1].result_decision.workspace_promotion is not None
    monkeypatch.setattr(
        gate._action_store,
        "_publish_result_event",
        original,
    )
    adapter.reject_execution_access = True
    guarded_interpreter.reject_interpreter_access = True

    report = coordinator.recover(frontier)

    assert report.is_complete
    assert report.recovered_operations[-1].events[-1].event_kind is (
        RunActionExecutionEventKind.RESULT_ACCEPTED
    )


def test_exchanged_edit_decision_recovers_before_acceptance_without_stale_inspection(
    publisher_case,
    tmp_path,
    monkeypatch,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "exchanged_edit_decision",
        b'{"implementation":"exchange then restart"}',
    )
    adapter = _AccessGuardedExecutionAdapter(reservation.intent.boundary_identity)
    guarded_interpreter = _AccessGuardedResultInterpreter(
        reservation.intent.boundary_identity.result_interpreter_identity
    )
    adapter.result_interpreter = guarded_interpreter
    candidate = _isolated_edit_candidate(publisher_case, tmp_path)
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_result_workspace",
        lambda _prepared, _capture: _FakeResultWorkspaceLease(candidate),
    )
    _append_result_received(gate, reservation)
    coordinator = _recovery_coordinator(gate, adapter)
    original_promote = coordinator._workspace_promoter._promote_decided

    def promote_then_die(**arguments):
        promoted = original_promote(**arguments)
        assert promoted.commit_sha != reservation.frontier.workspace_before.commit_sha
        raise RuntimeError("injected death after workspace exchange")

    monkeypatch.setattr(
        coordinator._workspace_promoter,
        "_promote_decided",
        promote_then_die,
    )
    with pytest.raises(RuntimeError, match="after workspace exchange"):
        coordinator.recover(frontier)
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    promotion = events[-1].result_decision.workspace_promotion
    assert events[-1].event_kind is RunActionExecutionEventKind.RESULT_DECIDED
    assert promotion is not None

    restarted = _recovery_coordinator(gate, adapter)
    adapter.reject_execution_access = True
    guarded_interpreter.reject_interpreter_access = True
    report = restarted.recover(frontier)

    assert report.is_complete
    terminal_events = report.recovered_operations[-1].events
    assert terminal_events[-1].event_kind is (
        RunActionExecutionEventKind.RESULT_ACCEPTED
    )
    assert terminal_events[-1].acceptance.workspace_after == (
        promotion.candidate_workspace
    )


def test_accepted_edit_cleanup_retries_from_durable_acceptance(
    publisher_case,
    tmp_path,
    monkeypatch,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "accepted_cleanup_retry",
        b'{"implementation":"cleanup retry"}',
    )
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    candidate = _isolated_edit_candidate(publisher_case, tmp_path)
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_result_workspace",
        lambda _prepared, _capture: _FakeResultWorkspaceLease(candidate),
    )
    _append_result_received(gate, reservation)
    original_unlink = promotion_module.os.unlink
    removed_entries = 0

    def unlink_then_die(name, *, dir_fd):
        nonlocal removed_entries
        original_unlink(name, dir_fd=dir_fd)
        removed_entries += 1
        if removed_entries == 1:
            raise RuntimeError("injected death after durable event 8")

    coordinator = _recovery_coordinator(gate, adapter)
    original_cleanup = coordinator._workspace_promoter._cleanup_accepted

    def cleanup_with_crash(**arguments):
        monkeypatch.setattr(
            promotion_module.os,
            "unlink",
            unlink_then_die,
        )
        return original_cleanup(**arguments)

    monkeypatch.setattr(
        coordinator._workspace_promoter,
        "_cleanup_accepted",
        cleanup_with_crash,
    )
    with pytest.raises(RuntimeError, match="durable event 8"):
        coordinator.recover(frontier)
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.RESULT_ACCEPTED
    monkeypatch.setattr(promotion_module.os, "unlink", original_unlink)
    monkeypatch.setattr(
        coordinator._workspace_promoter,
        "_cleanup_accepted",
        original_cleanup,
    )

    report = coordinator.recover(frontier)

    assert report.is_complete
    staging = (
        publisher_case["active"].run_root
        / publisher_case[
            "active"
        ].bootstrap_pin.installation_receipt.layout.run_action_workspace_staging_relative_path
    )
    assert tuple(staging.iterdir()) == ()


def test_projected_acceptance_cleanup_precedes_new_pending_edit_after_restart(
    resolver_case,
    publisher_case,
    tmp_path,
    monkeypatch,
) -> None:
    publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "projected_cleanup_restart",
        b'{"implementation":"projected cleanup"}',
    )
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    candidate = _isolated_edit_candidate(publisher_case, tmp_path)
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_result_workspace",
        lambda _prepared, _capture: _FakeResultWorkspaceLease(candidate),
    )
    _append_result_received(gate, reservation)
    coordinator = _recovery_coordinator(gate, adapter)

    def stop_before_cleanup(**_arguments):
        raise RuntimeError("injected death before accepted cleanup")

    original_cleanup = coordinator._workspace_promoter._cleanup_accepted
    monkeypatch.setattr(
        coordinator._workspace_promoter,
        "_cleanup_accepted",
        stop_before_cleanup,
    )
    with pytest.raises(RuntimeError, match="before accepted cleanup"):
        coordinator.recover(frontier)
    monkeypatch.setattr(
        coordinator._workspace_promoter,
        "_cleanup_accepted",
        original_cleanup,
    )
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    promotion = events[-2].result_decision.workspace_promotion
    current_evidence = frontier.checkpoint.safety_state.derivative_frontier.evidence
    branch = publisher_case["settings"].workspace_git_branch
    prior_advances = tuple(
        advance
        for advance in current_evidence.branch_advances
        if (
            advance.branch == branch
            and advance.commit_sha == reservation.frontier.workspace_before.commit_sha
        )
    )
    predecessor_advance_id = (
        None
        if current_evidence.branch_origin_heads[branch]
        == reservation.frontier.workspace_before.commit_sha
        else prior_advances[0].branch_advance_id
    )
    branch_advance = RunBranchAdvance.build(
        branch=branch,
        predecessor_commit_sha=reservation.frontier.workspace_before.commit_sha,
        commit_sha=promotion.candidate_workspace.commit_sha,
        predecessor_branch_advance_id=predecessor_advance_id,
        authorization_safety_state_id=(
            frontier.checkpoint.safety_state.safety_state_id
        ),
    )
    advanced_evidence = _remint_evidence(
        current_evidence,
        branch_heads={
            **current_evidence.branch_heads,
            branch: promotion.candidate_workspace.commit_sha,
        },
        branch_advances=(
            *current_evidence.branch_advances,
            branch_advance,
        ),
    )
    advanced_frontier = RunDerivativeFrontier.build(
        launch_subject_ids=(
            frontier.checkpoint.safety_state.derivative_frontier.launch_subject_ids
        ),
        evidence=advanced_evidence,
        derivatives=(frontier.checkpoint.safety_state.derivative_frontier.derivatives),
    )
    successor_bundle, successor_checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        frontier,
        RunSafetyBoundary.IMPLEMENTATION,
        derivative_frontier=advanced_frontier,
    )
    published = publisher.publish(
        publisher.issue_publication_permit(
            frontier,
            successor_checkpoint,
            successor_bundle,
        ),
        successor_checkpoint,
        successor_bundle,
    )
    run_root = publisher_case["active"].run_root
    publisher_case["active"].close()

    settings = resolver_case["resolver"]._settings
    active = StarterWorkspaceBuilder(settings).reopen(run_root)
    reopened_publisher = RunStatePublisher(active, settings.launch)
    reopened_frontier = reopened_publisher.load_reconciled()
    assert reopened_frontier.run_checkpoint_id == published.run_checkpoint_id
    security = _StaticSecurityAuthority(
        reopened_frontier.checkpoint.safety_state.security_observation
    )
    reopened_gate = RunFrontierActionGate(
        active_workspace=active,
        publisher=reopened_publisher,
        security_authority=security,
        credential_validity_authority=None,
        resource_finalization_authority=(
            _static_resource_finalization_authority(reopened_publisher)
        ),
    )
    publisher_case["active"] = active
    pending = _reserve_implementation_agent(
        reopened_gate,
        reopened_frontier,
        "pending_after_projected_cleanup",
        b'{"implementation":"edit after projected cleanup"}',
    )
    next_candidate = _isolated_edit_candidate(
        publisher_case,
        tmp_path,
        candidate_name="next-isolated-candidate",
        edit_name="next-isolated-edit.py",
    )
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_result_workspace",
        lambda _prepared, _capture: _FakeResultWorkspaceLease(next_candidate),
    )
    reopened_adapter = _AccessGuardedExecutionAdapter(pending.intent.boundary_identity)
    guarded_interpreter = _AccessGuardedResultInterpreter(
        pending.intent.boundary_identity.result_interpreter_identity
    )
    reopened_adapter.result_interpreter = guarded_interpreter
    reopened_coordinator = _recovery_coordinator(
        reopened_gate,
        reopened_adapter,
    )
    _append_result_received(reopened_gate, pending)
    original_publish = reopened_gate._action_store._publish_result_event

    def publish_second_decision_then_die(*arguments, **keywords):
        original_publish(*arguments, **keywords)
        if keywords["event"].event_kind is RunActionExecutionEventKind.RESULT_DECIDED:
            raise RuntimeError("injected death after second durable edit decision")

    monkeypatch.setattr(
        reopened_gate._action_store,
        "_publish_result_event",
        publish_second_decision_then_die,
    )
    with pytest.raises(RuntimeError, match="second durable edit decision"):
        reopened_coordinator.recover(reopened_frontier)
    second_events = reopened_gate._action_store.inspect().events_for(
        pending.intent.operation_id
    )
    assert second_events[-1].event_kind is RunActionExecutionEventKind.RESULT_DECIDED
    assert second_events[-1].result_decision.workspace_promotion is not None
    monkeypatch.setattr(
        reopened_gate._action_store,
        "_publish_result_event",
        original_publish,
    )
    restarted_coordinator = _recovery_coordinator(
        reopened_gate,
        reopened_adapter,
    )
    prior_cleanup_calls = 0

    def reject_prior_cleanup(**_arguments):
        nonlocal prior_cleanup_calls
        prior_cleanup_calls += 1
        raise AssertionError("durable event 7 was mistaken for prior cleanup residue")

    monkeypatch.setattr(
        restarted_coordinator._workspace_promoter,
        "_cleanup_accepted_if_owned",
        reject_prior_cleanup,
    )
    reopened_adapter.reject_execution_access = True
    guarded_interpreter.reject_interpreter_access = True

    report = restarted_coordinator.recover(reopened_frontier)
    assert report.is_complete
    terminal_events = report.recovered_operations[-1].events
    assert terminal_events[0].reservation.intent.operation_id == (
        pending.intent.operation_id
    )
    assert terminal_events[-1].event_kind is (
        RunActionExecutionEventKind.RESULT_ACCEPTED
    )
    assert terminal_events[-2].result_decision.workspace_promotion is not None
    assert not reopened_adapter.prepare_calls
    assert not reopened_adapter.continuation_calls
    assert len(guarded_interpreter.interpret_calls) == 2
    assert prior_cleanup_calls == 0
    staging = (
        active.run_root
        / active.bootstrap_pin.installation_receipt.layout.run_action_workspace_staging_relative_path
    )
    assert tuple(staging.iterdir()) == ()
    active.close()


def test_running_committed_spawn_is_queried_and_never_reactivated(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    adapter = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
        observation_state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert not adapter.prepare_calls
    assert len(adapter.inspect_calls) == 1
    assert not hasattr(adapter.inspect_calls[0], "request_payload")
    assert len(adapter.continuation_calls) == 1
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.ACTIVATION_COMMITTED


def test_running_edit_rejects_a_host_workspace_successor(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"recover edit"}'
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "recovery",
        payload,
    )
    _append_activation_committed(gate, reservation)
    commit_sha = _commit_workspace_edit(
        publisher_case,
        "recovered.py",
        "RECOVERED = True\n",
    )
    adapter = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
        observation_state=(RunActionCommittedSpawnState.RUNNING_CONTINUABLE),
    )

    with pytest.raises(
        RunWorkspaceFrontierError,
        match="branch head differs",
    ):
        _recovery_coordinator(gate, adapter).recover(frontier)

    assert commit_sha != reservation.frontier.workspace_before.commit_sha


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
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        workspace_state,
        payload,
    )
    _append_activation_committed(gate, reservation)
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
    adapter = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
        observation_state=(RunActionCommittedSpawnState.RUNNING_CONTINUABLE),
    )
    with pytest.raises(RunWorkspaceFrontierError):
        _recovery_coordinator(gate, adapter).recover(frontier)

    assert (
        _recovery_coordinator(gate, adapter).inspect(frontier).pending_operation_id
        == reservation.intent.operation_id
    )


def test_unknown_committed_spawn_remains_unresolved_without_replay(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    adapter = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
        observation_state=RunActionCommittedSpawnState.UNKNOWN,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert report.unresolved_operation_id == reservation.intent.operation_id
    assert report.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.ACTIVATION_COMMITTED
    )
    assert not adapter.prepare_calls
    assert len(adapter.inspect_calls) == 1
    assert not adapter.continuation_calls


def test_continuation_without_a_result_remains_unresolved(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    adapter = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
        observation_state=(RunActionCommittedSpawnState.RUNNING_CONTINUABLE),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(adapter.continuation_calls) == 1


def test_security_advance_prevents_committed_spawn_continuation(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    _advance_security(publisher_case, gate, frontier)
    adapter = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
        observation_state=(RunActionCommittedSpawnState.RUNNING_CONTINUABLE),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(adapter.inspect_calls) == 1
    assert not adapter.continuation_calls


def test_published_release_is_adopted_after_security_advance(
    publisher_case,
    monkeypatch,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    activation_event = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )[4]
    adoption = _release_adoption_for_event(
        activation_event,
        frontier.checkpoint.safety_state.security_observation,
    )
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_timeout_inspection",
        lambda **_arguments: _PresentReleaseInspection(adoption),
    )
    _advance_security(publisher_case, gate, frontier)
    adapter = _AdoptionAwarePendingAdapter(
        reservation.intent.boundary_identity,
        observation_state=(RunActionCommittedSpawnState.RUNNING_CONTINUABLE),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert adapter.observed_release_adoption == adoption
    assert len(adapter.continuation_calls) == 1


def test_timeout_outcome_is_independently_readopted_before_recovery_returns(
    publisher_case,
    monkeypatch,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    activation_event = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )[4]
    adoption = _release_adoption_for_event(
        activation_event,
        gate._security_authority.observation,
    )
    publication = _timeout_publication(
        activation_event.activation_revalidation_receipt,
        adoption,
    )
    inspections = []

    def inspect_transition(**_arguments):
        inspection = (
            _PresentReleaseInspection(adoption)
            if not inspections
            else _TimedOutReleaseInspection(adoption, publication)
        )
        inspections.append(inspection)
        return inspection

    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_timeout_inspection",
        inspect_transition,
    )
    monkeypatch.setattr(
        RunActionCommittedContinuationCapability,
        "_invoke_once",
        lambda _capability, _adapter: RunActionContinuationOutcome(
            state=RunActionContinuationState.TIMEOUT_PUBLISHED,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=publication,
        ),
    )
    adapter = _AdoptionAwarePendingAdapter(
        reservation.intent.boundary_identity,
        observation_state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(inspections) == 2
    assert inspections[0].topology is RunActionControlDirectoryTopology.RELEASED
    assert inspections[1].topology is RunActionControlDirectoryTopology.TIMED_OUT
    assert (
        gate._action_store.inspect()
        .events_for(reservation.intent.operation_id)[-1]
        .event_kind
        is RunActionExecutionEventKind.ACTIVATION_COMMITTED
    )


@pytest.mark.parametrize(
    "fresh_adoption",
    (
        "still_released",
        "substituted_publication",
    ),
)
def test_timeout_outcome_rejects_missing_or_substituted_fresh_adoption(
    publisher_case,
    monkeypatch,
    fresh_adoption,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    activation_event = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )[4]
    adoption = _release_adoption_for_event(
        activation_event,
        gate._security_authority.observation,
    )
    publication = _timeout_publication(
        activation_event.activation_revalidation_receipt,
        adoption,
    )
    substituted_publication = _remint_contract(
        publication,
        timeout_inode=publication.timeout_inode + 1,
    )
    inspections = []

    def inspect_transition(**_arguments):
        if not inspections:
            inspection = _PresentReleaseInspection(adoption)
        elif fresh_adoption == "still_released":
            inspection = _PresentReleaseInspection(adoption)
        else:
            inspection = _TimedOutReleaseInspection(
                adoption,
                substituted_publication,
            )
        inspections.append(inspection)
        return inspection

    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_timeout_inspection",
        inspect_transition,
    )
    monkeypatch.setattr(
        RunActionCommittedContinuationCapability,
        "_invoke_once",
        lambda _capability, _adapter: RunActionContinuationOutcome(
            state=RunActionContinuationState.TIMEOUT_PUBLISHED,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=publication,
        ),
    )
    adapter = _AdoptionAwarePendingAdapter(
        reservation.intent.boundary_identity,
        observation_state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
    )

    with pytest.raises(
        RunActionRecoveryError,
        match="differs from fresh recovery adoption",
    ):
        _recovery_coordinator(gate, adapter).recover(frontier)

    assert len(inspections) == 2
    assert (
        gate._action_store.inspect()
        .events_for(reservation.intent.operation_id)[-1]
        .event_kind
        is RunActionExecutionEventKind.ACTIVATION_COMMITTED
    )


def test_workspace_mutation_during_continuation_never_records_result(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _WorkspaceMutatingContinuationAdapter(
        reservation.intent.boundary_identity,
        publisher_case["active"].workspace,
    )
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RunWorkspaceFrontierError):
        coordinator.recover(frontier)

    plan = coordinator.inspect(frontier)
    assert plan.pending_operation_id == reservation.intent.operation_id
    assert plan.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.ACTIVATION_COMMITTED
    )


def test_received_result_runs_only_pure_local_interpretation(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    raw_result = _append_result_received(gate, reservation)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert not adapter.prepare_calls
    assert not adapter.continuation_calls
    assert not adapter.inspect_calls
    assert not adapter.continuation_calls
    assert len(adapter.result_interpreter.interpret_calls) == 2
    assert adapter.result_interpreter.interpret_calls[0] == (payload, raw_result)


def test_received_result_does_not_access_execution_adapter(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    raw_result = _append_result_received(gate, reservation)
    execution_adapter = _AccessGuardedExecutionAdapter(
        reservation.intent.boundary_identity
    )
    coordinator = _recovery_coordinator(gate, execution_adapter)
    execution_adapter.reject_execution_access = True

    report = coordinator.recover(frontier)

    assert report.is_complete
    assert execution_adapter.result_interpreter.interpret_calls[0] == (
        payload,
        raw_result,
    )


def test_decided_result_accepts_without_adapter_or_interpreter_access(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_decided(gate, reservation)
    execution_adapter = _AccessGuardedExecutionAdapter(
        reservation.intent.boundary_identity
    )
    guarded_interpreter = _AccessGuardedResultInterpreter(
        reservation.intent.boundary_identity.result_interpreter_identity
    )
    execution_adapter.result_interpreter = guarded_interpreter
    coordinator = _recovery_coordinator(gate, execution_adapter)
    execution_adapter.reject_execution_access = True
    guarded_interpreter.reject_interpreter_access = True

    report = coordinator.recover(frontier)

    assert report.is_complete
    events = report.recovered_operations[-1].events
    assert events[-2].event_kind is RunActionExecutionEventKind.RESULT_DECIDED
    assert events[-1].event_kind is RunActionExecutionEventKind.RESULT_ACCEPTED
    assert report.recovered_operations[-1].accepted_result_payload == (
        b'{"accepted_result":"complete"}'
    )


def test_decided_result_remains_nonterminal_for_publication(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_decided(gate, reservation)
    publisher = gate._publisher
    successor_bundle, successor_checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        frontier,
        RunSafetyBoundary.IMPLEMENTATION,
    )

    with pytest.raises(RunStatePublisherError, match="unresolved execution"):
        publisher.issue_publication_permit(
            frontier,
            successor_checkpoint,
            successor_bundle,
        )


def test_nondeterministic_local_interpretation_never_becomes_durable(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    _append_result_received(gate, reservation)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    adapter.result_interpreter = _NondeterministicResultInterpreter(
        reservation.intent.boundary_identity.result_interpreter_identity
    )
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RunActionRecoveryError, match="nondeterministic"):
        coordinator.recover(frontier)

    plan = coordinator.inspect(frontier)
    assert plan.pending_operation_id == reservation.intent.operation_id
    assert plan.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.RESULT_RECEIVED
    )


def test_workspace_mutation_during_interpretation_never_becomes_terminal(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_received(gate, reservation)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    adapter.result_interpreter = _WorkspaceMutatingResultInterpreter(
        reservation.intent.boundary_identity.result_interpreter_identity
    )
    adapter.result_interpreter.retained_workspace_descriptor = os.open(
        publisher_case["active"].workspace,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    coordinator = _recovery_coordinator(gate, adapter)

    with pytest.raises(RunWorkspaceFrontierError):
        coordinator.recover(frontier)

    os.close(adapter.result_interpreter.retained_workspace_descriptor)
    plan = coordinator.inspect(frontier)
    assert plan.pending_operation_id == reservation.intent.operation_id
    assert plan.live_ledger.operation_tails[-1].tail_kind is (
        RunActionExecutionEventKind.RESULT_RECEIVED
    )


def test_terminal_replay_reads_accepted_bytes_without_implementation_use(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_accepted(gate, reservation)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert report.recovered_operations[0].accepted_result_payload == (
        b'{"accepted_result":"complete"}'
    )
    assert not adapter.prepare_calls
    assert not adapter.continuation_calls
    assert not adapter.inspect_calls
    assert not adapter.result_interpreter.interpret_calls


def test_provider_termination_replays_without_implementation_use(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_provider_terminated(gate, reservation)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert report.recovered_operations[0].events[-1].event_kind is (
        RunActionExecutionEventKind.PROVIDER_TERMINATED
    )
    assert report.recovered_operations[0].accepted_result_payload is None
    assert not adapter.prepare_calls
    assert not adapter.continuation_calls
    assert not adapter.inspect_calls
    assert not adapter.result_interpreter.interpret_calls


def test_terminal_resources_block_recovery_report(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_accepted(gate, reservation)
    driver = gate._resource_finalization_authority._require_current()[2]
    driver.block_finalization = True

    with pytest.raises(RuntimeError, match="terminal resources remain"):
        _recovery_coordinator(
            gate,
            _FakeExecutionAdapter(reservation.intent.boundary_identity),
        ).recover(frontier)

    assert driver.finalized_operation_ids == [reservation.intent.operation_id]


def test_terminal_resources_are_reproved_at_both_publication_phases(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_accepted(gate, reservation)
    publisher = gate._publisher
    successor_bundle, successor_checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        frontier,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    driver = gate._resource_finalization_authority._require_current()[2]
    driver.block_absence = True

    with pytest.raises(RuntimeError, match="terminal resources remain"):
        publisher.issue_publication_permit(
            frontier,
            successor_checkpoint,
            successor_bundle,
        )

    driver.block_absence = False
    permit = publisher.issue_publication_permit(
        frontier,
        successor_checkpoint,
        successor_bundle,
    )
    driver.block_absence = True
    with pytest.raises(RuntimeError, match="terminal resources remain"):
        publisher.publish(
            permit,
            successor_checkpoint,
            successor_bundle,
        )

    assert driver.absence_checked_operation_ids == [
        reservation.intent.operation_id,
        reservation.intent.operation_id,
        reservation.intent.operation_id,
    ]


def test_terminal_resources_block_the_next_reservation(
    publisher_case,
) -> None:
    publisher, frontier, _security, gate = _action_case(publisher_case)
    reservation = _reserve_ideation_agent(gate, frontier)
    _append_result_accepted(gate, reservation)
    before = publisher.action_ledger_snapshot()
    driver = gate._resource_finalization_authority._require_current()[2]
    driver.block_absence = True

    with pytest.raises(RuntimeError, match="terminal resources remain"):
        _reserve_ideation_agent(gate, frontier)

    assert publisher.action_ledger_snapshot() == before
    assert driver.absence_checked_operation_ids == [reservation.intent.operation_id]


def test_registered_pre_release_loss_persists_as_terminal_event(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    adapter = _TrustedPreReleaseTerminationAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert report.is_complete
    assert len(events) == 6
    assert events[-1].event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED
    assert events[-1].provider_termination_receipt == adapter.termination_receipt
    assert adapter.publication_fence_source.closed


def test_pre_release_loss_fence_change_blocks_terminal_event(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    adapter = _TrustedPreReleaseTerminationAdapter(
        reservation.intent.boundary_identity,
        fence_change_at_check=4,
    )

    with pytest.raises(RuntimeError, match="main absence changed"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert len(events) == 5
    assert events[-1].event_kind is (RunActionExecutionEventKind.ACTIVATION_COMMITTED)
    assert adapter.publication_fence_source.closed


def test_existing_timeout_is_adopted_once_persisted_and_replayed(
    publisher_case,
    monkeypatch,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    activation_event = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )[4]
    adoption = _release_adoption_for_event(
        activation_event,
        gate._security_authority.observation,
    )
    publication = _timeout_publication(
        activation_event.activation_revalidation_receipt,
        adoption,
    )
    inspections = []

    def inspect_existing_timeout(**_arguments):
        inspection = _TimedOutReleaseInspection(adoption, publication)
        inspections.append(inspection)
        return inspection

    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_timeout_inspection",
        inspect_existing_timeout,
    )
    adapter = _TrustedTimeoutTerminationAdapter(
        reservation.intent.boundary_identity,
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert report.is_complete
    assert len(events) == 6
    assert len(inspections) == 2
    assert adapter.inspect_calls[0].timeout_directive_publication == publication
    assert events[-1].provider_termination_receipt == adapter.termination_receipt
    assert (
        events[-1].provider_termination_receipt.reason
        is RunActionProviderTerminationReason.TIMEOUT
    )

    guarded = _AccessGuardedExecutionAdapter(reservation.intent.boundary_identity)
    restarted = _recovery_coordinator(gate, guarded)
    guarded.reject_execution_access = True
    replay = restarted.recover(frontier)

    assert replay.is_complete
    assert replay.recovered_operations[-1].events[-1] == events[-1]
    assert not guarded.inspect_calls
    assert not guarded.continuation_calls
    assert len(inspections) == 2


@pytest.mark.parametrize("publish_before_interrupt", (False, True))
def test_provider_termination_has_exact_crash_restart_semantics(
    publisher_case,
    monkeypatch,
    publish_before_interrupt,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    adapter = _TrustedPreReleaseTerminationAdapter(reservation.intent.boundary_identity)
    coordinator = _recovery_coordinator(gate, adapter)
    store = gate._action_store
    publish_event_locked = store._publish_event_locked

    def interrupt_provider_termination(store_descriptor, operation_id, event):
        if (
            event.event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED
            and not publish_before_interrupt
        ):
            raise RuntimeError("injected death before provider termination event")
        publish_event_locked(store_descriptor, operation_id, event)
        if event.event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED:
            raise RuntimeError("injected death after provider termination event")

    monkeypatch.setattr(
        store,
        "_publish_event_locked",
        interrupt_provider_termination,
    )
    with pytest.raises(RuntimeError, match="provider termination event"):
        coordinator.recover(frontier)

    events = store.inspect().events_for(reservation.intent.operation_id)
    assert len(events) == (6 if publish_before_interrupt else 5)
    monkeypatch.setattr(store, "_publish_event_locked", publish_event_locked)
    if publish_before_interrupt:
        guarded = _AccessGuardedExecutionAdapter(reservation.intent.boundary_identity)
        restarted = _recovery_coordinator(gate, guarded)
        guarded.reject_execution_access = True

        report = restarted.recover(frontier)

        assert report.is_complete
        assert report.recovered_operations[-1].events[-1].event_kind is (
            RunActionExecutionEventKind.PROVIDER_TERMINATED
        )


@pytest.mark.parametrize(
    ("change_at_check", "expected_event_count"),
    (
        (1, 5),
        (2, 6),
    ),
)
def test_provider_termination_retains_exact_release_absence_during_publication(
    publisher_case,
    monkeypatch,
    change_at_check,
    expected_event_count,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    adapter = _TrustedPreReleaseTerminationAdapter(reservation.intent.boundary_identity)
    inspections = iter(
        (
            _AbsentReleaseInspection(),
            _ChangingAbsentReleaseInspection(change_at_check),
        )
    )
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_timeout_inspection",
        lambda **_arguments: next(inspections),
    )

    with pytest.raises(RuntimeError, match="release presence changed"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert len(events) == expected_event_count


@pytest.mark.parametrize(
    ("change_at_check", "expected_event_count"),
    (
        (1, 5),
        (2, 6),
    ),
)
def test_provider_termination_retains_exact_release_adoption_during_publication(
    publisher_case,
    monkeypatch,
    change_at_check,
    expected_event_count,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_activation_committed(gate, reservation)
    activation_event = gate._action_store.inspect().events_for(
        reservation.intent.operation_id
    )[4]
    adoption = _release_adoption_for_event(
        activation_event,
        gate._security_authority.observation,
    )
    adapter = _TrustedReleasedTerminationAdapter(reservation.intent.boundary_identity)
    inspections = iter(
        (
            _PresentReleaseInspection(adoption),
            _ChangingPresentReleaseInspection(adoption, change_at_check),
        )
    )
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_timeout_inspection",
        lambda **_arguments: next(inspections),
    )

    with pytest.raises(RuntimeError, match="release adoption changed"):
        _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert len(events) == expected_event_count


def test_normal_adapter_without_a_physical_termination_leaf_remains_pending(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert not report.is_complete
    assert len(adapter.continuation_calls) == 1
    assert events[-1].event_kind is RunActionExecutionEventKind.ACTIVATION_COMMITTED


def test_recovered_operation_rejects_malformed_accepted_prefix(
    publisher_case,
) -> None:
    _frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_accepted(gate, reservation)
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        accepted_event = session.events[-1]

    with pytest.raises(
        RunActionRecoveryError,
        match="recovered run action operation is invalid",
    ):
        RunActionRecoveredOperation(
            events=(accepted_event,),
            accepted_result_payload=b'{"accepted_result":"complete"}',
        )


def test_activation_staging_crash_reopens_only_as_unactivated_spawn(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    crashing = _FakeExecutionAdapter(
        reservation.intent.boundary_identity,
        fail_activation=True,
    )
    coordinator = _recovery_coordinator(gate, crashing)

    with pytest.raises(RuntimeError, match="durable spawn"):
        coordinator.recover(frontier)

    assert len(crashing.stage_calls) == 1
    assert not crashing.continuation_calls
    crashing.fail_activation = False
    crashing.observation_state = RunActionCommittedSpawnState.INERT_CONTINUABLE
    report = coordinator.recover(frontier)
    assert not report.is_complete
    assert len(crashing.continuation_calls) == 1
    assert len(crashing.inspect_calls) == 2


def test_security_advance_cancels_unspawned_reservation(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _advance_security(publisher_case, gate, frontier)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert report.recovered_operations[-1].events[-1].event_kind is (
        RunActionExecutionEventKind.CANCELLED
    )
    assert not adapter.prepare_calls


def test_security_advance_after_preparation_terminally_invalidates_frontier(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _SecurityAdvancingPrepareAdapter(
        reservation.intent.boundary_identity,
        lambda: _advance_security(publisher_case, gate, frontier),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert len(adapter.prepare_calls) == 1
    assert not adapter.continuation_calls
    terminal = gate._action_store.inspect().events_for(reservation.intent.operation_id)[
        -1
    ]
    assert terminal.event_kind is RunActionExecutionEventKind.FRONTIER_INVALIDATED
    assert terminal.workspace_after == terminal.reservation.frontier.workspace_before


def test_security_advance_during_allocation_fsync_prevents_materialization(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
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
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert report.is_complete
    assert authority.call_count == 2
    assert not adapter.prepare_calls
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[1].event_kind is RunActionExecutionEventKind.PREPARATION_ALLOCATED
    assert events[1].preparation_allocation is not None
    assert events[-1].event_kind is RunActionExecutionEventKind.FRONTIER_INVALIDATED


def test_security_advance_during_activation_staging_never_starts_provider(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _SecurityAdvancingStageAdapter(
        reservation.intent.boundary_identity,
        lambda: _advance_security(publisher_case, gate, frontier),
    )

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(adapter.stage_calls) == 1
    assert not adapter.continuation_calls
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED


def test_security_advance_after_activation_commit_never_starts_provider(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
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
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert authority.call_count == 5
    assert len(adapter.stage_calls) == 1
    assert not adapter.continuation_calls
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert events[-1].event_kind is (RunActionExecutionEventKind.ACTIVATION_COMMITTED)


def test_workspace_advance_after_preparation_remains_unresolved(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
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
    _commit_workspace_edit(
        publisher_case,
        "external-frontier.py",
        "EXTERNAL_FRONTIER = True\n",
    )
    coordinator._security_is_current = lambda _frontier: True

    report = coordinator.recover(frontier)

    tail = gate._action_store.inspect().events_for(reservation.intent.operation_id)[-1]
    assert not report.is_complete
    assert report.unresolved_operation_id == reservation.intent.operation_id
    assert tail.event_kind is RunActionExecutionEventKind.EXECUTION_PREPARED
    assert tail.workspace_after is None
    assert len(adapter.prepare_calls) == 1
    assert not adapter.continuation_calls


def test_recovery_rejects_boundary_identity_substitution(
    publisher_case,
) -> None:
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    _append_result_received(gate, reservation)
    substituted = _FakeExecutionAdapter(
        _boundary_identity(RunFrontierActionKind.EMBEDDING)
    )

    coordinator = _recovery_coordinator(gate, substituted)
    with pytest.raises(RunActionRecoveryError, match="absent or substituted"):
        coordinator.recover(frontier)

    assert (
        coordinator.inspect(frontier).pending_operation_id
        == reservation.intent.operation_id
    )


def test_recovery_rejects_same_identity_execution_object_substitution(
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    original = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    substituted = _FakeExecutionAdapter(reservation.intent.boundary_identity)
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
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    _append_result_received(gate, reservation)
    execution_adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    implementation_registry = _recovery_registry(execution_adapter)
    coordinator = gate.recovery_coordinator(implementation_registry)
    implementation = implementation_registry._implementations[0]
    substituted = _FakeResultInterpreter(
        reservation.intent.boundary_identity.result_interpreter_identity
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
    frontier, gate, _reservation, _payload = _reserved_case(publisher_case)
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
    frontier, gate, _reservation, payload = _reserved_case(publisher_case)
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
    frontier, gate, _reservation, _payload = _reserved_case(publisher_case)
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
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
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
            implementation_registry.resolve_execution(
                reservation.intent.boundary_identity
            )
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
    frontier, gate, reservation, payload = _reserved_case(publisher_case)
    raw_result = _append_result_received(gate, reservation)
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
        credential_validity_authority=None,
        resource_finalization_authority=(
            _static_resource_finalization_authority(publisher)
        ),
    )
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    report = _recovery_coordinator(reopened_gate, adapter).recover(reopened_frontier)

    assert report.is_complete
    assert adapter.result_interpreter.interpret_calls[0][1] == raw_result
    assert not adapter.inspect_calls
    active.close()


def test_result_decided_recovers_after_full_runtime_restart_without_implementation(
    resolver_case,
    publisher_case,
) -> None:
    frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_decided(gate, reservation)
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
        credential_validity_authority=None,
        resource_finalization_authority=(
            _static_resource_finalization_authority(publisher)
        ),
    )
    adapter = _AccessGuardedExecutionAdapter(reservation.intent.boundary_identity)
    guarded_interpreter = _AccessGuardedResultInterpreter(
        reservation.intent.boundary_identity.result_interpreter_identity
    )
    adapter.result_interpreter = guarded_interpreter
    coordinator = _recovery_coordinator(reopened_gate, adapter)
    adapter.reject_execution_access = True
    guarded_interpreter.reject_interpreter_access = True

    report = coordinator.recover(reopened_frontier)

    assert report.is_complete
    assert report.recovered_operations[-1].accepted_result_payload == (
        b'{"accepted_result":"complete"}'
    )
    active.close()


def test_fresh_embedding_recovery_never_receives_workspace_descriptor(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    payload = b'{"embedding":["complete input"]}'
    reservation = gate.reserve(
        frontier,
        kind=RunFrontierActionKind.EMBEDDING,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="embedding_recovery_0123456789abcdef",
        request_payload=payload,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        boundary_identity=_boundary_identity(RunFrontierActionKind.EMBEDDING),
    )
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)

    report = _recovery_coordinator(gate, adapter).recover(frontier)

    assert not report.is_complete
    assert len(adapter.continuation_calls) == 1
    assert not hasattr(adapter.continuation_calls[0], "workspace_descriptor")
