"""Physical event-5 retirement for an expired supervisor credential."""

from __future__ import annotations

import copy
from dataclasses import replace
import os
from pathlib import Path
import pickle
import signal
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from test_run_action_main_start import (
    _ActivationLease,
    _capability,
    _case,
    _credential_broker_registry,
    _StartAdapter,
)
from test_run_action_supervisor_contracts import _remint_contract

import kapso.cross_run.launch.run_action_credential_retirement as retirement
import kapso.cross_run.launch.run_action_recovery as recovery
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_credential_contracts import (
    RunActionCredentialRetirementIntent,
)
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_recovery import (
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionContinuationState,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    run_action_credential_lease_request,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessResult,
)


class _ControlInspection:
    def __init__(self, activation_event, *, change_at_check=None):
        self.activation_event = activation_event
        self.topology = RunActionControlDirectoryTopology.EMPTY
        self.workload_release_adoption = None
        self.timeout_directive_publication = None
        self.change_at_check = change_at_check
        self.current_checks = 0
        self.closed = False

    def require_current(self):
        if self.closed:
            raise AssertionError("test control inspection is closed")
        self.current_checks += 1
        if self.current_checks == self.change_at_check:
            raise RuntimeError("empty control topology changed")

    def __enter__(self):
        self.require_current()
        return self

    def __exit__(self, *_arguments):
        self.closed = True


class _CleanupExclusion:
    def __init__(self, authority):
        self.authority = authority
        self.closed = False
        self.current_checks = 0

    def require_current(self):
        if self.closed or not self.authority.exclusion_active:
            raise AssertionError("test cleanup exclusion is closed")
        self.current_checks += 1

    def __enter__(self):
        self.authority.exclusion_active = True
        self.require_current()
        return self

    def __exit__(self, *_arguments):
        self.closed = True
        self.authority.exclusion_active = False


class _CleanupAuthority:
    def __init__(self, docker_settings, result):
        self.settings = docker_settings
        self.result = result
        self.calls = []
        self.exclusion_active = False
        self.exclusions = []

    def _issue_exclusion_lease(self, *, _authority):
        assert _authority is retirement._DOCKER_CLEANUP_EXCLUSION_ISSUANCE
        exclusion = _CleanupExclusion(self)
        self.exclusions.append(exclusion)
        return exclusion

    def _remove_stopped_container_once(
        self,
        *,
        container_id,
        exclusion_lease,
        _authority,
    ):
        assert _authority is retirement._DOCKER_CLEANUP_REMOVE_AUTHORITY
        exclusion_lease.require_current()
        self.calls.append(container_id)
        return self.result


class _ContainmentAuthority:
    def __init__(self, docker_settings, result):
        self.settings = docker_settings
        self.result = result
        self.calls = []
        self.live_checks = 0

    def require_live_authority(self):
        self.live_checks += 1

    def _signal_container_once(
        self,
        *,
        container_id,
        signal_name,
        _authority,
    ):
        assert _authority is retirement._DOCKER_CONTAINMENT_SIGNAL_AUTHORITY
        self.calls.append((container_id, signal_name))
        return self.result


def _process_result(
    case,
    arguments,
    *,
    outcome=BoundedProcessOutcome.COMPLETED,
    returncode=0,
):
    trusted_root = Path.cwd().resolve()
    request = BoundedProcessRequest(
        argv=("docker", *arguments),
        trusted_root=trusted_root,
        cwd=trusted_root,
        timeout_seconds=case.docker_settings.command_timeout_seconds,
        cleanup_timeout_seconds=case.docker_settings.cleanup_timeout_seconds,
        stdout_byte_limit=case.docker_settings.command_output_byte_limit,
        stderr_byte_limit=case.docker_settings.command_output_byte_limit,
        environment={},
    )
    return BoundedProcessResult(
        request=request,
        outcome=outcome,
        returncode=returncode,
        stdout=b"",
        stderr=b"",
        stdout_bytes_observed=0,
        stderr_bytes_observed=0,
        duration_seconds=0.0,
    )


def _expired_capability(case, *, observation=None):
    credential_registry, credential_backend = _credential_broker_registry()
    request = run_action_credential_lease_request(
        case.query.prepared_execution,
        case.query.spawn_commit,
    )
    credential_backend._lease_expiries[request.credential_lease_request_id] = 1
    credential_observation = recovery._observe_pre_release_credential_state(
        case.query.activation_revalidation_receipt,
        credential_registry,
        _SystemRunActionClock(),
    )
    case.query = replace(
        case.query,
        credential_retirement_intent=RunActionCredentialRetirementIntent.mint(
            activation_event_id=case.query.activation_event.event_id,
            pre_release_credential_observation_id=(
                credential_observation.pre_release_credential_observation_id
            ),
            credential_lease_status=credential_observation.credential_lease_status,
            observed_before_realtime_nanoseconds=(
                credential_observation.observed_before_realtime_nanoseconds
            ),
            observed_after_realtime_nanoseconds=(
                credential_observation.observed_after_realtime_nanoseconds
            ),
            required_valid_until_realtime_nanoseconds=(
                credential_observation.required_valid_until_realtime_nanoseconds
            ),
        ),
    )
    return (
        _capability(
            case,
            observation=observation,
            credential_broker_registry=credential_registry,
        ),
        credential_backend,
    )


def _wire_retirement(
    monkeypatch,
    case,
    *,
    cleanup_result,
    containment_result,
    control_inspection=None,
):
    cleanup = _CleanupAuthority(case.docker_settings, cleanup_result)
    containment = _ContainmentAuthority(
        case.docker_settings,
        containment_result,
    )
    manager = object.__new__(retirement.DockerRunActionCredentialRetirementManager)
    manager._owner_process_id = os.getpid()
    selected_control = (
        _ControlInspection(case.query.activation_event)
        if control_inspection is None
        else control_inspection
    )
    monkeypatch.setattr(
        recovery,
        "RunActionTimeoutInspectionLease",
        _ControlInspection,
    )
    monkeypatch.setattr(
        retirement,
        "open_run_action_timeout_inspection",
        lambda **_arguments: selected_control,
    )
    monkeypatch.setattr(
        retirement.DockerRunActionCredentialRetirementManager,
        "runtime_settings",
        property(lambda _self: case.docker_settings),
    )
    monkeypatch.setattr(
        retirement,
        "_credential_retirement_authorities",
        lambda _manager: SimpleNamespace(
            containment=containment,
            cleanup=cleanup,
        ),
    )
    monkeypatch.setattr(
        retirement,
        "_run_action_observation_authority",
        lambda _manager: object(),
    )
    monkeypatch.setattr(
        retirement,
        "_docker_authorities_share_runtime",
        lambda _observation, _containment: True,
    )
    monkeypatch.setattr(
        retirement,
        "_docker_observation_and_cleanup_authorities_share_runtime",
        lambda _observation, _cleanup: True,
    )
    return SimpleNamespace(
        manager=manager,
        cleanup=cleanup,
        containment=containment,
        control=selected_control,
    )


def _retire(case, wired, capability):
    retirement.retire_run_action_expired_credential_once(
        capability=capability,
        resource_manager=case.resource_manager,
        retirement_manager=wired.manager,
        command=case.command,
        helper_evidence=case.helper,
        init_source_evidence=case.init,
        docker_settings=case.docker_settings,
        launch_settings=case.launch_settings,
    )


def _wire_running_observations(monkeypatch, case):
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "observe",
        lambda _self, _allocation: case.inventory,
    )
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "inspect_volume",
        lambda _self, _inventory: {},
    )
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "inspect_main",
        lambda _self, _inventory: {},
    )
    monkeypatch.setattr(
        retirement,
        "observe_runtime_volume",
        lambda *_arguments: case.volume,
    )
    monkeypatch.setattr(
        retirement,
        "observe_running_barrier_main_container",
        lambda *_arguments: case.running,
    )


def test_expired_inert_credential_removes_only_exact_created_main(monkeypatch):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    removed = _process_result(case, ("container", "rm", container_id))
    unused_signal = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=removed,
        containment_result=unused_signal,
    )
    activation_lease = _ActivationLease(case.inventory)
    monkeypatch.setattr(
        retirement,
        "open_selected_run_action_activation",
        lambda *_arguments, **_keywords: activation_lease,
    )
    capability, credential_backend = _expired_capability(case)

    outcome = capability._invoke_once(
        _StartAdapter(lambda active: _retire(case, wired, active))
    )

    assert outcome.state is RunActionContinuationState.PENDING
    assert wired.cleanup.calls == [container_id]
    assert wired.containment.calls == []
    assert activation_lease.current_checks >= 2
    assert wired.cleanup.exclusions[0].current_checks >= 3
    assert wired.cleanup.exclusions[0].closed
    assert wired.control.closed
    assert len(credential_backend.status_calls) == 1


def test_expired_running_credential_kills_only_stable_barrier(monkeypatch):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    removed = _process_result(case, ("container", "rm", container_id))
    killed = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=removed,
        containment_result=killed,
    )
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "observe",
        lambda _self, _allocation: case.inventory,
    )
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "inspect_volume",
        lambda _self, _inventory: {},
    )
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "inspect_main",
        lambda _self, _inventory: {},
    )
    monkeypatch.setattr(
        retirement,
        "observe_runtime_volume",
        lambda *_arguments: case.volume,
    )
    monkeypatch.setattr(
        retirement,
        "observe_running_barrier_main_container",
        lambda *_arguments: case.running,
    )
    observation = RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
        observation_token=case.running.complete_inspection_digest,
    )
    capability, credential_backend = _expired_capability(
        case,
        observation=observation,
    )

    outcome = capability._invoke_once(
        _StartAdapter(lambda active: _retire(case, wired, active))
    )

    assert outcome.state is RunActionContinuationState.PENDING
    assert wired.cleanup.calls == []
    assert wired.containment.calls == [(container_id, "SIGKILL")]
    assert wired.containment.live_checks == 1
    assert wired.control.closed
    assert len(credential_backend.status_calls) == 1


def test_running_occurrence_change_before_signal_blocks_retirement(monkeypatch):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    removed = _process_result(case, ("container", "rm", container_id))
    killed = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=removed,
        containment_result=killed,
    )
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "observe",
        lambda _self, _allocation: case.inventory,
    )
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "inspect_volume",
        lambda _self, _inventory: {},
    )
    monkeypatch.setattr(
        retirement.DockerRunActionResourceManager,
        "inspect_main",
        lambda _self, _inventory: {},
    )
    monkeypatch.setattr(
        retirement,
        "observe_runtime_volume",
        lambda *_arguments: case.volume,
    )
    changed_running = _remint_contract(
        case.running,
        complete_inspection_digest="sha256:" + "f" * 64,
    )
    running_observations = iter((case.running, changed_running))
    monkeypatch.setattr(
        retirement,
        "observe_running_barrier_main_container",
        lambda *_arguments: next(running_observations),
    )
    observation = RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
        observation_token=case.running.complete_inspection_digest,
    )
    capability, _credential_backend = _expired_capability(
        case,
        observation=observation,
    )

    with pytest.raises(
        retirement.RunActionCredentialRetirementError,
        match="differs from event 5",
    ):
        capability._invoke_once(
            _StartAdapter(lambda active: _retire(case, wired, active))
        )

    assert wired.cleanup.calls == []
    assert wired.containment.calls == []


def test_failed_retirement_command_is_only_an_attempt_and_requires_recovery(
    monkeypatch,
):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    timed_out_remove = _process_result(
        case,
        ("container", "rm", container_id),
        outcome=BoundedProcessOutcome.TIMED_OUT,
        returncode=-9,
    )
    unused_signal = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=timed_out_remove,
        containment_result=unused_signal,
    )
    monkeypatch.setattr(
        retirement,
        "open_selected_run_action_activation",
        lambda *_arguments, **_keywords: _ActivationLease(case.inventory),
    )
    capability, _credential_backend = _expired_capability(case)

    outcome = capability._invoke_once(
        _StartAdapter(lambda active: _retire(case, wired, active))
    )

    assert outcome.state is RunActionContinuationState.PENDING
    assert wired.cleanup.calls == [container_id]

    retry_wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=timed_out_remove,
        containment_result=unused_signal,
    )
    monkeypatch.setattr(
        retirement,
        "open_selected_run_action_activation",
        lambda *_arguments, **_keywords: _ActivationLease(case.inventory),
    )
    retry_capability, _credential_backend = _expired_capability(case)

    retry_outcome = retry_capability._invoke_once(
        _StartAdapter(lambda active: _retire(case, retry_wired, active))
    )

    assert retry_outcome.state is RunActionContinuationState.PENDING
    assert retry_wired.cleanup.calls == [container_id]


def test_ambiguous_running_retirement_is_only_an_attempt(monkeypatch):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    unused_remove = _process_result(case, ("container", "rm", container_id))
    timed_out_kill = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
        outcome=BoundedProcessOutcome.TIMED_OUT,
        returncode=-9,
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=unused_remove,
        containment_result=timed_out_kill,
    )
    _wire_running_observations(monkeypatch, case)
    observation = RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
        observation_token=case.running.complete_inspection_digest,
    )
    capability, _credential_backend = _expired_capability(
        case,
        observation=observation,
    )

    outcome = capability._invoke_once(
        _StartAdapter(lambda active: _retire(case, wired, active))
    )

    assert outcome.state is RunActionContinuationState.PENDING
    assert wired.cleanup.calls == []
    assert wired.containment.calls == [(container_id, "SIGKILL")]

    retry_wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=unused_remove,
        containment_result=timed_out_kill,
    )
    _wire_running_observations(monkeypatch, case)
    retry_capability, _credential_backend = _expired_capability(
        case,
        observation=observation,
    )

    retry_outcome = retry_capability._invoke_once(
        _StartAdapter(lambda active: _retire(case, retry_wired, active))
    )

    assert retry_outcome.state is RunActionContinuationState.PENDING
    assert retry_wired.cleanup.calls == []
    assert retry_wired.containment.calls == [(container_id, "SIGKILL")]


@pytest.mark.parametrize(
    ("change_at_check", "signal_expected"),
    ((3, False), (4, True)),
)
def test_running_control_change_fences_signal_and_completion(
    monkeypatch,
    change_at_check,
    signal_expected,
):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    unused_remove = _process_result(case, ("container", "rm", container_id))
    killed = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    control = _ControlInspection(
        case.query.activation_event,
        change_at_check=change_at_check,
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=unused_remove,
        containment_result=killed,
        control_inspection=control,
    )
    _wire_running_observations(monkeypatch, case)
    observation = RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
        observation_token=case.running.complete_inspection_digest,
    )
    capability, _credential_backend = _expired_capability(
        case,
        observation=observation,
    )

    with pytest.raises(RuntimeError, match="control topology changed"):
        capability._invoke_once(
            _StartAdapter(lambda active: _retire(case, wired, active))
        )

    assert wired.cleanup.calls == []
    assert wired.containment.calls == (
        [(container_id, "SIGKILL")] if signal_expected else []
    )


def test_control_change_after_retirement_command_prevents_completion(monkeypatch):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    removed = _process_result(case, ("container", "rm", container_id))
    unused_signal = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    control = _ControlInspection(
        case.query.activation_event,
        change_at_check=4,
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=removed,
        containment_result=unused_signal,
        control_inspection=control,
    )
    monkeypatch.setattr(
        retirement,
        "open_selected_run_action_activation",
        lambda *_arguments, **_keywords: _ActivationLease(case.inventory),
    )
    capability, _credential_backend = _expired_capability(case)

    with pytest.raises(RuntimeError, match="control topology changed"):
        capability._invoke_once(
            _StartAdapter(lambda active: _retire(case, wired, active))
        )

    assert wired.cleanup.calls == [container_id]
    assert wired.containment.calls == []


def test_control_change_blocks_expired_credential_mutation(monkeypatch):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    removed = _process_result(case, ("container", "rm", container_id))
    unused_signal = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    control = _ControlInspection(
        case.query.activation_event,
        change_at_check=2,
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=removed,
        containment_result=unused_signal,
        control_inspection=control,
    )
    monkeypatch.setattr(
        retirement,
        "open_selected_run_action_activation",
        lambda *_arguments, **_keywords: _ActivationLease(case.inventory),
    )
    capability, _credential_backend = _expired_capability(case)

    with pytest.raises(RuntimeError, match="control topology changed"):
        capability._invoke_once(
            _StartAdapter(lambda active: _retire(case, wired, active))
        )

    assert wired.cleanup.calls == []
    assert wired.containment.calls == []


def test_valid_credential_cannot_use_retirement_authority(monkeypatch):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    removed = _process_result(case, ("container", "rm", container_id))
    unused_signal = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=removed,
        containment_result=unused_signal,
    )
    monkeypatch.setattr(
        retirement,
        "open_selected_run_action_activation",
        lambda *_arguments, **_keywords: _ActivationLease(case.inventory),
    )
    capability = _capability(case)

    with pytest.raises(
        retirement.RunActionCredentialRetirementError,
        match="differs from expired event 5",
    ):
        capability._invoke_once(
            _StartAdapter(lambda active: _retire(case, wired, active))
        )

    assert wired.cleanup.calls == []
    assert wired.containment.calls == []


def test_retirement_manager_from_another_runtime_is_rejected(monkeypatch):
    case = _case(monkeypatch)
    container_id = case.query.spawn_commit.provider_execution_id
    removed = _process_result(case, ("container", "rm", container_id))
    unused_signal = _process_result(
        case,
        ("container", "kill", "--signal", "SIGKILL", container_id),
    )
    wired = _wire_retirement(
        monkeypatch,
        case,
        cleanup_result=removed,
        containment_result=unused_signal,
    )
    monkeypatch.setattr(
        retirement,
        "_docker_authorities_share_runtime",
        lambda _observation, _containment: False,
    )
    capability, _credential_backend = _expired_capability(case)

    with pytest.raises(
        retirement.RunActionCredentialRetirementError,
        match="different Docker runtimes",
    ):
        capability._invoke_once(
            _StartAdapter(lambda active: _retire(case, wired, active))
        )

    assert wired.cleanup.calls == []
    assert wired.containment.calls == []


def test_retirement_requires_issued_exact_manager():
    manager = object.__new__(retirement.DockerRunActionCredentialRetirementManager)

    with pytest.raises(
        retirement.RunActionCredentialRetirementError,
        match="unissued",
    ):
        retirement._credential_retirement_authorities(manager)

    with pytest.raises(
        retirement.RunActionCredentialRetirementError,
        match="one pinned Docker runtime",
    ):
        retirement.DockerRunActionCredentialRetirementManager(object())


def test_retirement_manager_rejects_copy_and_serialization():
    manager = object.__new__(retirement.DockerRunActionCredentialRetirementManager)

    with pytest.raises(
        retirement.RunActionCredentialRetirementError,
        match="cannot be copied",
    ):
        copy.copy(manager)
    with pytest.raises(
        retirement.RunActionCredentialRetirementError,
        match="cannot be copied",
    ):
        copy.deepcopy(manager)
    with pytest.raises(
        retirement.RunActionCredentialRetirementError,
        match="cannot be serialized",
    ):
        pickle.dumps(manager)


def test_retirement_manager_rejects_fork_before_inherited_lock_access():
    manager = object.__new__(retirement.DockerRunActionCredentialRetirementManager)
    manager._owner_process_id = os.getpid()
    lock_held = Event()
    release_lock = Event()

    def hold_manager_lock():
        with retirement._CREDENTIAL_RETIREMENT_MANAGER_LOCK:
            lock_held.set()
            assert release_lock.wait(timeout=5)

    holder = Thread(target=hold_manager_lock)
    holder.start()
    assert lock_held.wait(timeout=5)

    child = os.fork()
    if child == 0:
        signal.alarm(5)
        with pytest.raises(
            retirement.RunActionCredentialRetirementError,
            match="foreign",
        ):
            retirement._credential_retirement_authorities(manager)
        signal.alarm(0)
        os._exit(37)

    release_lock.set()
    holder.join()
    _child_pid, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 37
