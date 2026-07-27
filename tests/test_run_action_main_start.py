"""Sealed event-5 authority for starting the run-action barrier."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.cross_run.launch.run_action_main_start as main_start
import kapso.cross_run.launch.run_action_resolved_workload as resolved_workload
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_docker_inspect import observe_runtime_volume
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnQuery,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_resolved_workload import (
    RunActionResolvedWorkloadError,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessResult,
)
from test_run_action_docker_inspect import _volume_raw
from test_run_action_release_contracts import (
    _security_observation as _release_security_observation,
)
from test_run_action_terminal_inspection import (
    _configured_settings,
    _inspection_context,
    _SecurityAuthority,
)


class _ActivationLease:
    def __init__(self, inventory):
        self.inventory = inventory
        self.current_checks = 0
        self.volume_checks = 0
        self.closed = False

    def require_current(self):
        if self.closed:
            raise AssertionError("test activation lease is closed")
        self.current_checks += 1

    def require_volume_current(self):
        if self.closed:
            raise AssertionError("test activation lease is closed")
        self.volume_checks += 1

    def __enter__(self):
        self.require_current()
        return self

    def __exit__(self, *_arguments):
        self.closed = True


class _StartAuthority:
    def __init__(self, docker_settings, result):
        self.settings = docker_settings
        self.result = result
        self.calls = []
        self.exclusion_active = False

    def _issue_exclusion_lease(self, *, _authority):
        if _authority is not main_start._DOCKER_START_EXCLUSION_ISSUANCE:
            raise AssertionError("test start exclusion received the wrong authority")
        return _StartExclusion(self)

    def _start_created_container_once(
        self,
        *,
        container_id,
        exclusion_lease,
        _authority,
    ):
        if _authority is not main_start._DOCKER_START_CONTAINER_AUTHORITY:
            raise AssertionError("test start received the wrong private authority")
        exclusion_lease.require_current()
        self.calls.append(container_id)
        return self.result


class _StartExclusion:
    def __init__(self, start_authority):
        self.start_authority = start_authority
        self.closed = False
        self.current_checks = 0

    def require_current(self):
        if self.closed or not self.start_authority.exclusion_active:
            raise AssertionError("test start exclusion is closed")
        self.current_checks += 1

    def __enter__(self):
        self.start_authority.exclusion_active = True
        self.require_current()
        return self

    def __exit__(self, *_arguments):
        self.closed = True
        self.start_authority.exclusion_active = False


class _StartAdapter:
    def __init__(self, invoke):
        self._invoke = invoke

    def continue_committed_once(self, capability):
        self._invoke(capability)
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PENDING,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        )


def _case(monkeypatch, *, result_outcome=BoundedProcessOutcome.COMPLETED):
    docker_settings, launch_settings = _configured_settings()
    released_query, inventory, _raw, command, helper, init = _inspection_context(
        docker_settings
    )
    query = RunActionCommittedSpawnQuery(
        preparation_allocation=released_query.preparation_allocation,
        activation_event=released_query.activation_event,
        workload_release_adoption=None,
        timeout_directive_publication=None,
    )
    resource_manager = object.__new__(main_start.DockerRunActionResourceManager)
    start_manager = object.__new__(main_start.DockerRunActionStartManager)
    monkeypatch.setattr(
        main_start.DockerRunActionResourceManager,
        "runtime_settings",
        property(lambda _self: docker_settings),
    )
    monkeypatch.setattr(
        main_start.DockerRunActionStartManager,
        "runtime_settings",
        property(lambda _self: docker_settings),
    )
    leases = []

    def open_activation(*_arguments, **_keywords):
        if leases and not start_authority.exclusion_active:
            raise AssertionError(
                "event-5 start revalidation ran outside mutation exclusion"
            )
        lease = _ActivationLease(inventory)
        leases.append(lease)
        return lease

    monkeypatch.setattr(
        main_start,
        "open_selected_run_action_activation",
        open_activation,
    )
    monkeypatch.setattr(
        main_start,
        "_docker_observation_and_start_authorities_share_runtime",
        lambda _observation, _start: True,
    )
    monkeypatch.setattr(
        main_start,
        "_run_action_observation_authority",
        lambda _manager: object(),
    )
    container_id = query.spawn_commit.provider_execution_id
    trusted_root = Path.cwd().resolve()
    request = BoundedProcessRequest(
        argv=("docker", "container", "start", container_id),
        trusted_root=trusted_root,
        cwd=trusted_root,
        timeout_seconds=docker_settings.command_timeout_seconds,
        cleanup_timeout_seconds=docker_settings.cleanup_timeout_seconds,
        stdout_byte_limit=docker_settings.command_output_byte_limit,
        stderr_byte_limit=docker_settings.command_output_byte_limit,
        environment={},
    )
    result = BoundedProcessResult(
        request=request,
        outcome=result_outcome,
        returncode=0,
        stdout=f"{container_id}\n".encode("ascii"),
        stderr=b"",
        stdout_bytes_observed=len(container_id) + 1,
        stderr_bytes_observed=0,
        duration_seconds=0.0,
    )
    start_authority = _StartAuthority(docker_settings, result)
    monkeypatch.setattr(
        main_start,
        "_start_authority",
        lambda _manager: start_authority,
    )
    running = (
        released_query.workload_release_adoption.workload_release_receipt.resolved_workload_observation.running_container_observation
    )
    monkeypatch.setattr(
        main_start,
        "_observe_exact_running_barrier",
        lambda **_arguments: running,
    )
    authority = query.prepared_execution.runtime_volume_authority
    volume = observe_runtime_volume(
        _volume_raw(authority, docker_settings),
        query.prepared_execution.preparation_claim,
        authority,
        docker_settings,
    )
    observation = main_start.inspect_run_action_inert_activation(
        query=query,
        resource_manager=resource_manager,
        launch_settings=launch_settings,
    )
    return SimpleNamespace(
        docker_settings=docker_settings,
        launch_settings=launch_settings,
        query=query,
        inventory=inventory,
        resource_manager=resource_manager,
        start_manager=start_manager,
        start_authority=start_authority,
        leases=leases,
        command=command,
        volume=volume,
        helper=helper,
        init=init,
        running=running,
        observation=observation,
    )


def _capability(case, observation=None):
    return RunActionCommittedContinuationCapability(
        query=case.query,
        observation=case.observation if observation is None else observation,
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=_SystemRunActionClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )


def _start(case, capability):
    return main_start.start_run_action_barrier_once(
        capability=capability,
        resource_manager=case.resource_manager,
        start_manager=case.start_manager,
        command=case.command,
        volume_observation=case.volume,
        helper_evidence=case.helper,
        init_source_evidence=case.init,
        docker_settings=case.docker_settings,
        launch_settings=case.launch_settings,
    )


def test_blocked_workload_process_bound_mismatch_fails_before_proc_inspection(
    monkeypatch,
):
    case = _case(monkeypatch)
    capability = _capability(case)
    mismatched_settings = replace(
        case.launch_settings,
        run_action_process_snapshot_size_bytes=(
            case.launch_settings.run_action_process_snapshot_size_bytes + 1
        ),
    )
    monkeypatch.setattr(
        resolved_workload.os,
        "open",
        lambda *_arguments, **_keywords: pytest.fail(
            "proc inspection preceded process-bound validation"
        ),
    )

    def open_with_mismatched_bound(active_capability):
        resolved_workload.open_run_action_blocked_workload(
            active_capability,
            committed_running_observation=case.running,
            resource_manager=case.resource_manager,
            preparation_allocation=case.query.preparation_allocation,
            command=case.command,
            volume_observation=case.volume,
            helper_evidence=case.helper,
            init_source_evidence=case.init,
            docker_settings=case.docker_settings,
            launch_settings=mismatched_settings,
        )

    with pytest.raises(
        RunActionResolvedWorkloadError,
        match="inputs differ from exact durable event 5",
    ):
        capability._invoke_once(
            _StartAdapter(open_with_mismatched_bound),
        )


def test_inert_inspection_and_start_bind_one_exact_event_5(monkeypatch):
    case = _case(monkeypatch)
    capability = _capability(case)
    observed = []

    outcome = capability._invoke_once(
        _StartAdapter(lambda active: observed.append(_start(case, active)))
    )

    assert outcome.state is RunActionContinuationState.PENDING
    assert observed == [case.running]
    assert case.start_authority.calls == [case.query.spawn_commit.provider_execution_id]
    assert len(case.leases) == 2
    assert case.leases[0].current_checks >= 2
    assert case.leases[1].current_checks >= 2
    assert case.leases[1].volume_checks == 3
    assert all(lease.closed for lease in case.leases)
    with pytest.raises(
        RunActionRecoveryError,
        match="spent, cloned, or foreign",
    ):
        capability._invoke_once(_StartAdapter(lambda _active: None))


def test_wrong_inert_token_rejects_before_docker_start(monkeypatch):
    case = _case(monkeypatch)
    wrong = RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.INERT_CONTINUABLE,
        observation_token="sha256:" + "f" * 64,
    )
    capability = _capability(case, wrong)

    with pytest.raises(
        RunActionRecoveryError,
        match="lacks exact live event-5 authority",
    ):
        capability._invoke_once(_StartAdapter(lambda active: _start(case, active)))

    assert case.start_authority.calls == []


def test_start_rejects_a_manager_from_another_runtime(monkeypatch):
    case = _case(monkeypatch)
    capability = _capability(case)
    monkeypatch.setattr(
        main_start,
        "_docker_observation_and_start_authorities_share_runtime",
        lambda _observation, _start: False,
    )

    with pytest.raises(
        main_start.RunActionMainStartError,
        match="one exact configured runtime",
    ):
        capability._invoke_once(_StartAdapter(lambda active: _start(case, active)))

    assert case.start_authority.calls == []


def test_inert_pending_without_registered_start_is_rejected(monkeypatch):
    case = _case(monkeypatch)
    capability = _capability(case)

    with pytest.raises(
        RunActionRecoveryError,
        match="inert continuation lacks exact start authority",
    ):
        capability._invoke_once(_StartAdapter(lambda _active: None))

    assert case.start_authority.calls == []


def test_final_volume_revalidation_precedes_start_completion(monkeypatch):
    case = _case(monkeypatch)
    capability = _capability(case)

    def reject_third_volume_check(lease):
        if lease.closed:
            raise AssertionError("test activation lease is closed")
        lease.volume_checks += 1
        if lease.volume_checks == 3:
            raise RuntimeError("final retained volume changed")

    monkeypatch.setattr(
        _ActivationLease,
        "require_volume_current",
        reject_third_volume_check,
    )

    with pytest.raises(RuntimeError, match="final retained volume changed"):
        capability._invoke_once(_StartAdapter(lambda active: _start(case, active)))

    assert case.start_authority.calls == [case.query.spawn_commit.provider_execution_id]
    with pytest.raises(
        RunActionRecoveryError,
        match="spent, cloned, or foreign",
    ):
        capability._invoke_once(_StartAdapter(lambda _active: None))


def test_ambiguous_start_burns_capability_and_requires_fresh_recovery(monkeypatch):
    case = _case(monkeypatch, result_outcome=BoundedProcessOutcome.TIMED_OUT)
    capability = _capability(case)

    with pytest.raises(
        main_start.RunActionMainStartError,
        match="failed or ambiguous",
    ):
        capability._invoke_once(_StartAdapter(lambda active: _start(case, active)))

    assert case.start_authority.calls == [case.query.spawn_commit.provider_execution_id]
    with pytest.raises(
        RunActionRecoveryError,
        match="spent, cloned, or foreign",
    ):
        capability._invoke_once(_StartAdapter(lambda _active: None))

    case.start_authority.result = replace(
        case.start_authority.result,
        outcome=BoundedProcessOutcome.COMPLETED,
    )
    fresh_capability = _capability(case)
    outcome = fresh_capability._invoke_once(
        _StartAdapter(lambda active: _start(case, active))
    )
    assert outcome.state is RunActionContinuationState.PENDING
    assert case.start_authority.calls == [
        case.query.spawn_commit.provider_execution_id,
        case.query.spawn_commit.provider_execution_id,
    ]


def test_noninert_continuation_cannot_take_start_authority(monkeypatch):
    case = _case(monkeypatch)
    running_observation = RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
        observation_token=case.running.complete_inspection_digest,
    )
    capability = _capability(case, running_observation)

    with pytest.raises(
        RunActionRecoveryError,
        match="lacks exact live event-5 authority",
    ):
        capability._invoke_once(_StartAdapter(lambda active: _start(case, active)))

    assert case.start_authority.calls == []
