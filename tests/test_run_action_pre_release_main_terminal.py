"""Retained present-exited authority for a main before workload release."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_run_frontier_action_gate import _credential_broker_registry

import kapso.cross_run.launch.run_action_pre_release_main_terminal as main_terminal
import kapso.cross_run.launch.run_action_pre_release_resources as pre_release_resources
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnQuery,
    RunActionCommittedSpawnState,
    RunActionContinuationState,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    RunActionControlDirectoryLease,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_pre_release_main_terminal_observation_token,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
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

_HOST_BOOT_ID = "123e4567-e89b-42d3-a456-426614174000"


def _case(monkeypatch):
    docker_settings, launch_settings = _configured_settings()
    released_query, inventory, raw, command, helper, init = _inspection_context(
        docker_settings
    )
    query = RunActionCommittedSpawnQuery(
        preparation_allocation=released_query.preparation_allocation,
        activation_event=released_query.activation_event,
        credential_retirement_intent=None,
        workload_release_adoption=None,
        timeout_directive_publication=None,
    )
    state = {
        "inventory": inventory,
        "main_raw": raw,
        "topology": RunActionControlDirectoryTopology.EMPTY,
        "control_checks": 0,
        "control_leases": [],
    }
    manager = object.__new__(main_terminal.DockerRunActionResourceManager)
    activation = query.activation_revalidation_receipt

    monkeypatch.setattr(
        main_terminal.DockerRunActionResourceManager,
        "runtime_settings",
        property(lambda _self: docker_settings),
    )
    monkeypatch.setattr(
        main_terminal.DockerRunActionResourceManager,
        "observe",
        lambda _self, _allocation: state["inventory"],
    )
    monkeypatch.setattr(
        main_terminal.DockerRunActionResourceManager,
        "inspect_volume",
        lambda _self, _inventory: _volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
    )
    monkeypatch.setattr(
        main_terminal.DockerRunActionResourceManager,
        "inspect_keeper",
        lambda _self, _inventory: {},
    )
    monkeypatch.setattr(
        main_terminal.DockerRunActionResourceManager,
        "inspect_main",
        lambda _self, _inventory: copy.deepcopy(state["main_raw"]),
    )

    def require_control_current(lease):
        if lease._test_closed:
            raise AssertionError("test control lease is closed")
        if state["topology"] is not RunActionControlDirectoryTopology.EMPTY:
            raise main_terminal.RunActionPreReleaseMainTerminalError(
                "test control topology changed"
            )
        state["control_checks"] += 1

    def close_control(lease):
        if lease._test_closed:
            raise AssertionError("test control lease closed twice")
        lease._test_closed = True

    def reobserve_volume(lease, _volume, keeper):
        require_control_current(lease)
        assert keeper == activation.reobserved_keeper_evidence
        return activation.reobserved_volume_evidence

    monkeypatch.setattr(
        RunActionControlDirectoryLease,
        "require_current",
        require_control_current,
    )
    monkeypatch.setattr(
        RunActionControlDirectoryLease,
        "topology",
        property(lambda _self: state["topology"]),
    )
    monkeypatch.setattr(
        RunActionControlDirectoryLease,
        "close",
        close_control,
    )
    monkeypatch.setattr(
        RunActionControlDirectoryLease,
        "reobserve_runtime_volume_evidence",
        reobserve_volume,
    )
    monkeypatch.setattr(
        main_terminal,
        "open_run_action_control_directory",
        lambda _prepared: _new_control_lease(state),
    )
    monkeypatch.setattr(
        pre_release_resources,
        "observe_running_keeper",
        lambda *_arguments: activation.reobserved_keeper_evidence,
    )
    monkeypatch.setattr(
        main_terminal,
        "read_run_action_host_boot_id",
        lambda _descriptor: _HOST_BOOT_ID,
    )
    next_clock_value = [80_000_000_000]

    def read_clock(_self):
        value = next_clock_value[0]
        next_clock_value[0] += 1
        return value

    monkeypatch.setattr(
        _SystemRunActionClock,
        "boottime_nanoseconds",
        read_clock,
    )
    return SimpleNamespace(
        query=query,
        manager=manager,
        command=command,
        helper=helper,
        init=init,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
        state=state,
        inventory=inventory,
    )


def _new_control_lease(state):
    lease = object.__new__(RunActionControlDirectoryLease)
    lease._test_closed = False
    state["control_leases"].append(lease)
    return lease


def _inspect(case):
    return main_terminal.inspect_run_action_pre_release_main_terminal(
        query=case.query,
        resource_manager=case.manager,
        command=case.command,
        helper_evidence=case.helper,
        init_source_evidence=case.init,
        docker_settings=case.docker_settings,
        launch_settings=case.launch_settings,
    )


def _capability(case, observation):
    return RunActionCommittedContinuationCapability(
        query=case.query,
        observation=RunActionCommittedSpawnObservation(
            state=(RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE),
            observation_token=(
                run_action_pre_release_main_terminal_observation_token(observation)
            ),
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_broker_registry=_credential_broker_registry()[0],
        release_clock=_SystemRunActionClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )


def _capture(case, capability):
    class _TerminalAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            return (
                main_terminal.capture_run_action_pre_release_main_terminal_termination(
                    capability=active_capability,
                    resource_manager=case.manager,
                    command=case.command,
                    helper_evidence=case.helper,
                    init_source_evidence=case.init,
                    docker_settings=case.docker_settings,
                    launch_settings=case.launch_settings,
                )
            )

    return capability._invoke_once(_TerminalAdapter())


def test_stable_pre_release_terminal_is_captured_as_provider_failure(monkeypatch):
    case = _case(monkeypatch)
    observation = _inspect(case)

    outcome = _capture(case, _capability(case, observation))

    receipt = outcome.provider_termination_receipt
    assert outcome.state is RunActionContinuationState.PROVIDER_TERMINATED
    assert receipt.disposition is RunActionProviderTerminationDisposition.FAILED
    assert (
        receipt.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL
    )
    assert receipt.workload_release_adoption is None
    assert receipt.pre_release_main_loss_observation is None
    assert run_action_pre_release_main_terminal_observation_token(
        receipt.terminal_observation
    ) == run_action_pre_release_main_terminal_observation_token(observation)
    assert len(case.state["control_leases"]) == 2
    assert case.state["control_leases"][0]._test_closed
    assert not case.state["control_leases"][1]._test_closed
    outcome.provider_termination_publication_fence.require_current()
    outcome.provider_termination_publication_fence.close()
    assert all(lease._test_closed for lease in case.state["control_leases"])


def test_present_to_absent_transition_burns_capture_without_event6(monkeypatch):
    case = _case(monkeypatch)
    observation = _inspect(case)
    case.state["inventory"] = replace(case.inventory, main_container_id=None)

    with pytest.raises(
        main_terminal.RunActionPreReleaseMainTerminalError,
        match="volume, keeper, and main",
    ):
        _capture(case, _capability(case, observation))
    assert all(lease._test_closed for lease in case.state["control_leases"])


def test_terminal_mutation_rejects_sealed_capture(monkeypatch):
    case = _case(monkeypatch)
    observation = _inspect(case)
    changed = copy.deepcopy(case.state["main_raw"])
    changed["State"]["FinishedAt"] = "2026-07-25T01:02:05.123456789Z"
    case.state["main_raw"] = changed

    with pytest.raises(
        main_terminal.RunActionPreReleaseMainTerminalError,
        match="differs from sealed classification",
    ):
        _capture(case, _capability(case, observation))
    assert all(lease._test_closed for lease in case.state["control_leases"])


def test_release_appearance_invalidates_pre_release_terminal(monkeypatch):
    case = _case(monkeypatch)
    observation = _inspect(case)
    case.state["topology"] = RunActionControlDirectoryTopology.RELEASED

    with pytest.raises(
        main_terminal.RunActionPreReleaseMainTerminalError,
        match="empty control topology",
    ):
        _capture(case, _capability(case, observation))
    assert all(lease._test_closed for lease in case.state["control_leases"])


def test_keeper_or_main_substitution_rejects_classification(monkeypatch):
    case = _case(monkeypatch)
    case.state["inventory"] = replace(
        case.inventory,
        keeper_container_id="f" * 64,
    )
    with pytest.raises(
        main_terminal.RunActionPreReleaseMainTerminalError,
        match="volume, keeper, and main",
    ):
        _inspect(case)

    case.state["inventory"] = replace(
        case.inventory,
        main_container_id="e" * 64,
    )
    with pytest.raises(
        main_terminal.RunActionPreReleaseMainTerminalError,
        match="volume, keeper, and main",
    ):
        _inspect(case)


def test_publication_fence_cannot_cross_threads(monkeypatch):
    case = _case(monkeypatch)
    observation = _inspect(case)
    outcome = _capture(case, _capability(case, observation))
    publication_fence = outcome.provider_termination_publication_fence

    with ThreadPoolExecutor(max_workers=1) as executor:
        foreign_check = executor.submit(publication_fence.require_current)
        with pytest.raises(
            RunActionRecoveryError,
            match="closed or foreign",
        ):
            foreign_check.result()

    publication_fence.require_current()
    publication_fence.close()


def test_foreign_settings_reject_before_control_authority(monkeypatch):
    case = _case(monkeypatch)
    foreign_settings = replace(
        case.docker_settings,
        runtime_socket_path="/run/foreign-docker.sock",
    )

    with pytest.raises(
        main_terminal.RunActionPreReleaseMainTerminalError,
        match="configured authority",
    ):
        main_terminal.inspect_run_action_pre_release_main_terminal(
            query=case.query,
            resource_manager=case.manager,
            command=case.command,
            helper_evidence=case.helper,
            init_source_evidence=case.init,
            docker_settings=foreign_settings,
            launch_settings=case.launch_settings,
        )
    assert case.state["control_checks"] == 0
    assert case.state["control_leases"] == []
