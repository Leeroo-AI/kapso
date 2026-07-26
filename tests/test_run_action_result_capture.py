from __future__ import annotations

import copy

import pytest

import kapso.cross_run.launch.run_action_docker_inspect as docker_inspect
import kapso.cross_run.launch.run_action_result_capture as result_capture
import kapso.cross_run.launch.run_action_terminal_inspection as terminal_inspection
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    _RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
    _RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
    _RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionProviderResult,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_release_candidate import (
    _SystemRunActionReleaseClock,
)
from test_run_action_docker_inspect import _volume_raw
from test_run_action_supervisor_contracts import (
    _result_capture_receipt,
    _terminal_observation,
)
from test_run_action_terminal_inspection import (
    _configured_settings,
    _inspection_context,
    _patch_physical_inspection,
    _SecurityAuthority,
)
from test_run_action_release_contracts import (
    _security_observation as _release_security_observation,
)


class _ReleaseInspection:
    def __init__(self, topology, adoption):
        self.topology = topology
        self.adoption = adoption
        self.workload_release_adoption = adoption
        self.timeout_directive_publication = None
        self.current_checks = 0
        self.closed = False

    def require_current(self):
        if self.closed:
            raise AssertionError("test release inspection is closed")
        self.current_checks += 1

    def close(self):
        if self.closed:
            raise AssertionError("test release inspection closed twice")
        self.closed = True


def _capture_case(
    monkeypatch,
    clock_samples,
    *,
    observation_state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
):
    docker_settings, launch_settings = _configured_settings()
    query, inventory, raw, command, helper, init = _inspection_context(docker_settings)
    adoption = query.workload_release_adoption
    terminal_release = _ReleaseInspection(
        RunActionControlDirectoryTopology.RELEASED,
        adoption,
    )
    manager, remaining_main_payloads = _patch_physical_inspection(
        monkeypatch,
        inventory=inventory,
        volume_raw=_volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
        main_inspections=(copy.deepcopy(raw), copy.deepcopy(raw)),
        control_inspection=terminal_release,
        host_boot_id=adoption.workload_release_receipt.host_boot_id,
    )
    _normalized, normalized_payload, _raw_size_bytes = (
        docker_inspect._snapshot_container_inspection(
            raw,
            "test result capture",
        )
    )
    release_clock = _SystemRunActionReleaseClock()
    clock_values = iter(clock_samples)
    release_clock.boottime_nanoseconds = lambda: next(clock_values)
    capability = RunActionCommittedContinuationCapability(
        query=query,
        observation=RunActionCommittedSpawnObservation(
            state=observation_state,
            observation_token=tree_or_blob_digest(normalized_payload),
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=release_clock,
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )
    outer_release = _ReleaseInspection(
        RunActionControlDirectoryTopology.RELEASED,
        adoption,
    )
    monkeypatch.setattr(
        result_capture,
        "open_run_action_release_inspection",
        lambda **_arguments: outer_release,
    )
    monkeypatch.setattr(
        result_capture,
        "read_run_action_host_boot_id",
        lambda _descriptor: adoption.workload_release_receipt.host_boot_id,
    )
    return (
        capability,
        query,
        manager,
        raw,
        command,
        helper,
        init,
        docker_settings,
        launch_settings,
        outer_release,
        remaining_main_payloads,
    )


def _reinspect_terminal(
    capability,
    manager,
    command,
    helper,
    init,
    docker_settings,
    launch_settings,
):
    return terminal_inspection.reinspect_run_action_terminal(
        capability=capability,
        resource_manager=manager,
        command=command,
        helper_evidence=helper,
        init_source_evidence=init,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )


def test_trusted_result_capture_registers_the_exact_provider_result(monkeypatch):
    deadline = _inspection_context(_configured_settings()[0])[
        0
    ].workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    (
        capability,
        query,
        manager,
        _raw,
        command,
        helper,
        init,
        docker_settings,
        launch_settings,
        outer_release,
        remaining_main_payloads,
    ) = _capture_case(monkeypatch, (deadline - 1, deadline + 1))
    payload = b'{"captured":"exact"}'

    class _TrustedCaptureAdapter:
        def __init__(self):
            self.terminal = None
            self.result = None

        def continue_committed_once(self, active_capability):
            self.terminal = _reinspect_terminal(
                active_capability,
                manager,
                command,
                helper,
                init,
                docker_settings,
                launch_settings,
            )
            capture_receipt = _result_capture_receipt(
                query.prepared_execution,
                query.activation_revalidation_receipt,
                self.terminal,
                payload,
            )
            monkeypatch.setattr(
                result_capture,
                "inspect_run_action_terminal",
                lambda **_arguments: self.terminal,
            )
            monkeypatch.setattr(
                result_capture,
                "capture_run_action_result_file",
                lambda *_arguments, **_keywords: (capture_receipt, payload),
            )
            self.result = result_capture.capture_run_action_terminal_result(
                capability=active_capability,
                resource_manager=manager,
                command=command,
                helper_evidence=helper,
                init_source_evidence=init,
                docker_settings=docker_settings,
                launch_settings=launch_settings,
            )
            with pytest.raises(
                RunActionRecoveryError,
                match="termination registration lacks exact live authority",
            ):
                active_capability._take_provider_termination_authority(
                    _authority=_RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
                )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.RESULT_CAPTURED,
                result=self.result,
                provider_termination_receipt=None,
            )

    adapter = _TrustedCaptureAdapter()
    outcome = capability._invoke_once(adapter)

    assert outcome.result == adapter.result
    assert outcome.result.result_payload == payload
    assert outer_release.current_checks == 2
    assert outer_release.closed
    assert not remaining_main_payloads


def test_fabricated_result_cannot_bypass_the_trusted_capture_leaf(monkeypatch):
    deadline = _inspection_context(_configured_settings()[0])[
        0
    ].workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    (
        capability,
        query,
        manager,
        _raw,
        command,
        helper,
        init,
        docker_settings,
        launch_settings,
        _outer_release,
        _remaining_main_payloads,
    ) = _capture_case(monkeypatch, (deadline - 1,))
    payload = b'{"fabricated":"typed"}'

    class _FabricatingAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            terminal = _reinspect_terminal(
                active_capability,
                manager,
                command,
                helper,
                init,
                docker_settings,
                launch_settings,
            )
            result = RunActionProviderResult(
                terminal_observation=terminal,
                result_capture_receipt=_result_capture_receipt(
                    query.prepared_execution,
                    query.activation_revalidation_receipt,
                    terminal,
                    payload,
                ),
                result_payload=payload,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.RESULT_CAPTURED,
                result=result,
                provider_termination_receipt=None,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="terminal continuation lacks its trusted outcome registration",
    ):
        capability._invoke_once(_FabricatingAdapter())


@pytest.mark.parametrize(
    "observation_state",
    (
        RunActionCommittedSpawnState.INERT_CONTINUABLE,
        RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
    ),
)
def test_nonterminal_observation_cannot_return_a_fabricated_result(
    monkeypatch,
    observation_state,
):
    deadline = _inspection_context(_configured_settings()[0])[
        0
    ].workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    (
        capability,
        query,
        _manager,
        _raw,
        _command,
        _helper,
        _init,
        _docker_settings,
        _launch_settings,
        _outer_release,
        _remaining_main_payloads,
    ) = _capture_case(
        monkeypatch,
        (deadline - 1,),
        observation_state=observation_state,
    )
    payload = b'{"fabricated":"nonterminal"}'
    prepared = query.prepared_execution
    forged_terminal = _terminal_observation(
        prepared,
        query.spawn_commit,
        query.workload_release_adoption,
    )
    forged_result = RunActionProviderResult(
        terminal_observation=forged_terminal,
        result_capture_receipt=_result_capture_receipt(
            prepared,
            query.activation_revalidation_receipt,
            forged_terminal,
            payload,
        ),
        result_payload=payload,
    )

    class _FabricatingNonterminalAdapter:
        @staticmethod
        def continue_committed_once(_active_capability):
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.RESULT_CAPTURED,
                result=forged_result,
                provider_termination_receipt=None,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="nonterminal continuation consumed terminal outcome authority",
    ):
        capability._invoke_once(_FabricatingNonterminalAdapter())


@pytest.mark.parametrize("capture_outcome", ("substitute", "pending"))
def test_adapter_cannot_discard_a_trusted_capture(monkeypatch, capture_outcome):
    deadline = _inspection_context(_configured_settings()[0])[
        0
    ].workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    (
        capability,
        query,
        manager,
        _raw,
        command,
        helper,
        init,
        docker_settings,
        launch_settings,
        _outer_release,
        _remaining_main_payloads,
    ) = _capture_case(monkeypatch, (deadline - 1,))
    captured_payload = b'{"captured":"trusted"}'
    substituted_payload = b'{"captured":"substituted"}'

    class _DiscardingAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            terminal = _reinspect_terminal(
                active_capability,
                manager,
                command,
                helper,
                init,
                docker_settings,
                launch_settings,
            )
            capture_receipt = _result_capture_receipt(
                query.prepared_execution,
                query.activation_revalidation_receipt,
                terminal,
                captured_payload,
            )
            monkeypatch.setattr(
                result_capture,
                "inspect_run_action_terminal",
                lambda **_arguments: terminal,
            )
            monkeypatch.setattr(
                result_capture,
                "capture_run_action_result_file",
                lambda *_arguments, **_keywords: (
                    capture_receipt,
                    captured_payload,
                ),
            )
            result_capture.capture_run_action_terminal_result(
                capability=active_capability,
                resource_manager=manager,
                command=command,
                helper_evidence=helper,
                init_source_evidence=init,
                docker_settings=docker_settings,
                launch_settings=launch_settings,
            )
            if capture_outcome == "pending":
                return RunActionContinuationOutcome(
                    state=RunActionContinuationState.PENDING,
                    result=None,
                    provider_termination_receipt=None,
                )
            substituted = RunActionProviderResult(
                terminal_observation=terminal,
                result_capture_receipt=_result_capture_receipt(
                    query.prepared_execution,
                    query.activation_revalidation_receipt,
                    terminal,
                    substituted_payload,
                ),
                result_payload=substituted_payload,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.RESULT_CAPTURED,
                result=substituted,
                provider_termination_receipt=None,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="terminal continuation lacks its trusted outcome registration",
    ):
        capability._invoke_once(_DiscardingAdapter())


def test_result_capture_must_start_by_the_original_execution_deadline(monkeypatch):
    query = _inspection_context(_configured_settings()[0])[0]
    deadline = (
        query.workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    )
    (
        capability,
        query,
        manager,
        _raw,
        command,
        helper,
        init,
        docker_settings,
        launch_settings,
        _outer_release,
        _remaining_main_payloads,
    ) = _capture_case(monkeypatch, (deadline + 1,))
    payload = b'{"captured":"deadline"}'

    class _DeadlineAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            terminal = _reinspect_terminal(
                active_capability,
                manager,
                command,
                helper,
                init,
                docker_settings,
                launch_settings,
            )
            monkeypatch.setattr(
                result_capture,
                "inspect_run_action_terminal",
                lambda **_arguments: terminal,
            )
            monkeypatch.setattr(
                result_capture,
                "capture_run_action_result_file",
                lambda *_arguments, **_keywords: (
                    _result_capture_receipt(
                        query.prepared_execution,
                        query.activation_revalidation_receipt,
                        terminal,
                        payload,
                    ),
                    payload,
                ),
            )
            result = result_capture.capture_run_action_terminal_result(
                capability=active_capability,
                resource_manager=manager,
                command=command,
                helper_evidence=helper,
                init_source_evidence=init,
                docker_settings=docker_settings,
                launch_settings=launch_settings,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.RESULT_CAPTURED,
                result=result,
                provider_termination_receipt=None,
            )

    with pytest.raises(RunActionRecoveryError, match="started outside"):
        capability._invoke_once(_DeadlineAdapter())


def test_timed_out_topology_cannot_consume_result_capture_authority():
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    terminal = _terminal_observation(
        query.prepared_execution,
        query.spawn_commit,
        query.workload_release_adoption,
    )
    capability = RunActionCommittedContinuationCapability(
        query=query,
        observation=RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            observation_token=terminal.complete_inspection_digest,
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=_SystemRunActionReleaseClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )

    class _TimedOutAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            active_capability._take_terminal_inspection_authority(
                _authority=_RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY,
            )
            active_capability._complete_terminal_inspection(
                terminal,
                _authority=_RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY,
            )
            with pytest.raises(
                RunActionRecoveryError,
                match="result capture lacks exact live terminal authority",
            ):
                active_capability._take_result_capture_authority(
                    _authority=_RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
                )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
                provider_termination_receipt=None,
            )

    capability._invoke_once(_TimedOutAdapter())
