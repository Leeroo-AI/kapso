"""Unified natural terminal resolution under one retained release."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

import kapso.cross_run.launch.run_action_docker_inspect as docker_inspect
import kapso.cross_run.launch.run_action_natural_terminal as natural_terminal
import kapso.cross_run.launch.run_action_terminal_inspection as terminal_inspection
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionProviderResult,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from test_run_action_docker_inspect import _volume_raw
from test_run_action_release_contracts import (
    _security_observation as _release_security_observation,
)
from test_run_action_supervisor_contracts import _result_capture_receipt
from test_run_action_terminal_inspection import (
    _configured_settings,
    _inspection_context,
    _patch_physical_inspection,
    _SecurityAuthority,
)


class _ReleaseInspection:
    def __init__(self, topology, adoption, timeout_directive_publication=None):
        self.topology = topology
        self.adoption = adoption
        self.workload_release_adoption = adoption
        self.timeout_directive_publication = timeout_directive_publication
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


def _case(
    monkeypatch,
    *,
    exit_code=0,
    oom_killed=False,
    timed_out=False,
):
    docker_settings, launch_settings = _configured_settings()
    query, inventory, raw, command, helper, init = _inspection_context(
        docker_settings,
        timed_out=timed_out,
    )
    raw["State"]["ExitCode"] = exit_code
    raw["State"]["OOMKilled"] = oom_killed
    first_control_inspection = _ReleaseInspection(
        query.control_directory_topology,
        query.workload_release_adoption,
        query.timeout_directive_publication,
    )
    manager, remaining_main_payloads = _patch_physical_inspection(
        monkeypatch,
        docker_settings=docker_settings,
        inventory=inventory,
        volume_raw=_volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
        main_inspections=tuple(copy.deepcopy(raw) for _copy_number in range(4)),
        control_inspection=first_control_inspection,
        host_boot_id=(
            query.workload_release_adoption.workload_release_receipt.host_boot_id
        ),
    )
    control_inspections = [first_control_inspection]

    def open_control_inspection(**_arguments):
        if not control_inspections[-1].closed:
            return control_inspections[-1]
        inspection = _ReleaseInspection(
            query.control_directory_topology,
            query.workload_release_adoption,
            query.timeout_directive_publication,
        )
        control_inspections.append(inspection)
        return inspection

    monkeypatch.setattr(
        terminal_inspection,
        "open_run_action_timeout_inspection",
        open_control_inspection,
    )
    _normalized, normalized_payload, _raw_size_bytes = (
        docker_inspect._snapshot_container_inspection(
            raw,
            "test natural terminal",
        )
    )
    capability = RunActionCommittedContinuationCapability(
        query=query,
        observation=RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            observation_token=tree_or_blob_digest(normalized_payload),
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=_SystemRunActionClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )
    outer_release = _ReleaseInspection(
        query.control_directory_topology,
        query.workload_release_adoption,
        query.timeout_directive_publication,
    )
    monkeypatch.setattr(
        natural_terminal,
        "open_run_action_release_inspection",
        lambda **_arguments: outer_release,
    )
    monkeypatch.setattr(
        natural_terminal,
        "read_run_action_host_boot_id",
        lambda _descriptor: (
            query.workload_release_adoption.workload_release_receipt.host_boot_id
        ),
    )
    return SimpleNamespace(
        capability=capability,
        query=query,
        manager=manager,
        command=command,
        helper=helper,
        init=init,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
        outer_release=outer_release,
        control_inspections=control_inspections,
        remaining_main_payloads=remaining_main_payloads,
    )


def _resolve(case, monkeypatch, result_payload):
    capture_calls = []

    def capture(prepared, terminal, _volume, *, settings):
        capture_calls.append((prepared, terminal, settings))
        return (
            _result_capture_receipt(
                prepared,
                case.query.activation_revalidation_receipt,
                terminal,
                result_payload,
            ),
            result_payload,
        )

    monkeypatch.setattr(
        natural_terminal,
        "capture_run_action_result_file",
        capture,
    )

    class _ResolutionAdapter:
        def __init__(self):
            self.outcome = None

        def continue_committed_once(self, capability):
            self.outcome = natural_terminal.resolve_run_action_natural_terminal_once(
                capability=capability,
                resource_manager=case.manager,
                command=case.command,
                helper_evidence=case.helper,
                init_source_evidence=case.init,
                docker_settings=case.docker_settings,
                launch_settings=case.launch_settings,
            )
            return self.outcome

    adapter = _ResolutionAdapter()
    outcome = case.capability._invoke_once(adapter)
    assert outcome is adapter.outcome
    return outcome, capture_calls


def test_nonempty_zero_exit_registers_one_exact_provider_result(monkeypatch):
    case = _case(monkeypatch)
    payload = b'{"captured":"exact"}'

    outcome, capture_calls = _resolve(case, monkeypatch, payload)

    assert outcome.state is RunActionContinuationState.RESULT_CAPTURED
    assert type(outcome.result) is RunActionProviderResult
    assert outcome.result.result_payload == payload
    assert len(capture_calls) == 1
    assert case.outer_release.current_checks == 3
    assert case.outer_release.closed
    assert not case.remaining_main_payloads


def test_empty_zero_exit_registers_exact_empty_result_failure(monkeypatch):
    case = _case(monkeypatch)

    outcome, capture_calls = _resolve(case, monkeypatch, b"")

    assert outcome.state is RunActionContinuationState.PROVIDER_TERMINATED
    assert type(outcome.provider_termination_receipt) is (
        RunActionProviderTerminationReceipt
    )
    assert outcome.provider_termination_receipt.reason is (
        RunActionProviderTerminationReason.EMPTY_RESULT
    )
    assert (
        outcome.provider_termination_receipt.empty_result_capture_receipt.size_bytes
        == 0
    )
    assert len(capture_calls) == 1
    assert not case.remaining_main_payloads


@pytest.mark.parametrize(
    ("exit_code", "oom_killed", "expected_reason"),
    (
        (137, True, RunActionProviderTerminationReason.OOM),
        (0, True, RunActionProviderTerminationReason.OOM),
        (9, False, RunActionProviderTerminationReason.NONZERO_EXIT),
    ),
)
def test_terminal_failure_precedence_never_reads_result(
    monkeypatch,
    exit_code,
    oom_killed,
    expected_reason,
):
    case = _case(
        monkeypatch,
        exit_code=exit_code,
        oom_killed=oom_killed,
    )

    outcome, capture_calls = _resolve(
        case,
        monkeypatch,
        b"must not be read",
    )

    assert outcome.state is RunActionContinuationState.PROVIDER_TERMINATED
    assert outcome.provider_termination_receipt.reason is expected_reason
    assert outcome.provider_termination_receipt.empty_result_capture_receipt is None
    assert capture_calls == []
    assert not case.remaining_main_payloads


def test_timed_out_terminal_cannot_enter_natural_resolution(monkeypatch):
    case = _case(monkeypatch, timed_out=True)

    class _TimedOutAdapter:
        @staticmethod
        def continue_committed_once(capability):
            natural_terminal.resolve_run_action_natural_terminal_once(
                capability=capability,
                resource_manager=case.manager,
                command=case.command,
                helper_evidence=case.helper,
                init_source_evidence=case.init,
                docker_settings=case.docker_settings,
                launch_settings=case.launch_settings,
            )
            raise AssertionError("timed-out terminal entered natural resolution")

    with pytest.raises(
        natural_terminal.RunActionNaturalTerminalError,
        match="exact released occurrence",
    ):
        case.capability._invoke_once(_TimedOutAdapter())
    assert case.outer_release.current_checks == 0
    assert len(case.remaining_main_payloads) == 4


def test_settings_substitution_is_rejected_before_terminal_authority(monkeypatch):
    case = _case(monkeypatch)
    foreign_settings = replace(
        case.docker_settings,
        runtime_socket_path="/run/foreign-docker.sock",
    )

    class _ForeignSettingsAdapter:
        @staticmethod
        def continue_committed_once(capability):
            natural_terminal.resolve_run_action_natural_terminal_once(
                capability=capability,
                resource_manager=case.manager,
                command=case.command,
                helper_evidence=case.helper,
                init_source_evidence=case.init,
                docker_settings=foreign_settings,
                launch_settings=case.launch_settings,
            )
            raise AssertionError("foreign settings entered natural resolution")

    with pytest.raises(
        natural_terminal.RunActionNaturalTerminalError,
        match="configured authority",
    ):
        case.capability._invoke_once(_ForeignSettingsAdapter())
    assert case.outer_release.current_checks == 0
    assert len(case.remaining_main_payloads) == 4


def test_adapter_cannot_discard_a_trusted_natural_resolution(monkeypatch):
    case = _case(monkeypatch)
    payload = b'{"captured":"trusted"}'

    def capture(prepared, terminal, _volume, *, settings):
        return (
            _result_capture_receipt(
                prepared,
                case.query.activation_revalidation_receipt,
                terminal,
                payload,
            ),
            payload,
        )

    monkeypatch.setattr(
        natural_terminal,
        "capture_run_action_result_file",
        capture,
    )

    class _DiscardingAdapter:
        @staticmethod
        def continue_committed_once(capability):
            natural_terminal.resolve_run_action_natural_terminal_once(
                capability=capability,
                resource_manager=case.manager,
                command=case.command,
                helper_evidence=case.helper,
                init_source_evidence=case.init,
                docker_settings=case.docker_settings,
                launch_settings=case.launch_settings,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=None,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="trusted outcome registration",
    ):
        case.capability._invoke_once(_DiscardingAdapter())
