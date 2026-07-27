"""Retained physical authority for a main loss before workload release."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_run_frontier_action_gate import _credential_broker_registry

import kapso.cross_run.launch.run_action_pre_release_main_loss as main_loss
import kapso.cross_run.launch.run_action_pre_release_resources as pre_release_resources
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
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
    run_action_pre_release_main_loss_observation_token,
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
    docker_settings, _launch_settings = _configured_settings()
    released_query, inventory, _raw, _command, helper, init = _inspection_context(
        docker_settings
    )
    query = RunActionCommittedSpawnQuery(
        preparation_allocation=released_query.preparation_allocation,
        activation_event=released_query.activation_event,
        credential_retirement_intent=None,
        workload_release_adoption=None,
        timeout_directive_publication=None,
    )
    surviving_inventory = replace(inventory, main_container_id=None)
    state = {
        "inventory": surviving_inventory,
        "control_checks": 0,
        "control_leases": [],
    }
    manager = object.__new__(main_loss.DockerRunActionResourceManager)
    activation = query.activation_revalidation_receipt

    monkeypatch.setattr(
        main_loss.DockerRunActionResourceManager,
        "runtime_settings",
        property(lambda _self: docker_settings),
    )
    monkeypatch.setattr(
        main_loss.DockerRunActionResourceManager,
        "observe",
        lambda _self, _allocation: state["inventory"],
    )
    monkeypatch.setattr(
        main_loss.DockerRunActionResourceManager,
        "inspect_volume",
        lambda _self, _inventory: _volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
    )
    monkeypatch.setattr(
        main_loss.DockerRunActionResourceManager,
        "inspect_keeper",
        lambda _self, _inventory: {},
    )

    def require_control_current(_self):
        if _self._test_closed:
            raise AssertionError("test control lease is closed")
        state["control_checks"] += 1

    def close_control(_self):
        if _self._test_closed:
            raise AssertionError("test control lease closed twice")
        _self._test_closed = True

    def reobserve_volume(_self, _volume, keeper):
        require_control_current(_self)
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
        property(lambda _self: query.control_directory_topology),
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
        main_loss,
        "open_run_action_control_directory",
        lambda _prepared: _new_control_lease(state),
    )
    monkeypatch.setattr(
        pre_release_resources,
        "observe_running_keeper",
        lambda *_arguments: activation.reobserved_keeper_evidence,
    )
    monkeypatch.setattr(
        main_loss,
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
        helper=helper,
        init=init,
        docker_settings=docker_settings,
        state=state,
        surviving_inventory=surviving_inventory,
    )


def _new_control_lease(state):
    lease = object.__new__(RunActionControlDirectoryLease)
    lease._test_closed = False
    state["control_leases"].append(lease)
    return lease


def _inspect(case):
    return main_loss.inspect_run_action_pre_release_main_loss(
        query=case.query,
        resource_manager=case.manager,
        helper_evidence=case.helper,
        init_source_evidence=case.init,
        docker_settings=case.docker_settings,
    )


def _capability(case, observation):
    return RunActionCommittedContinuationCapability(
        query=case.query,
        observation=RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
            observation_token=run_action_pre_release_main_loss_observation_token(
                observation
            ),
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_broker_registry=_credential_broker_registry()[0],
        release_clock=_SystemRunActionClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )


def _capture(case, capability):
    class _LossAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            return main_loss.capture_run_action_pre_release_main_loss_termination(
                capability=active_capability,
                resource_manager=case.manager,
                helper_evidence=case.helper,
                init_source_evidence=case.init,
                docker_settings=case.docker_settings,
            )

    return capability._invoke_once(_LossAdapter())


def test_stable_main_loss_is_captured_once_as_provider_failure(monkeypatch):
    case = _case(monkeypatch)
    observation = _inspect(case)
    capability = _capability(case, observation)

    outcome = _capture(case, capability)

    receipt = outcome.provider_termination_receipt
    assert outcome.state is RunActionContinuationState.PROVIDER_TERMINATED
    assert receipt.disposition is RunActionProviderTerminationDisposition.FAILED
    assert receipt.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
    assert receipt.pre_release_main_loss_observation.observed_main_container_ids == ()
    assert len(case.state["control_leases"]) == 2
    assert case.state["control_leases"][0]._test_closed
    assert not case.state["control_leases"][1]._test_closed
    outcome.provider_termination_publication_fence.require_current()
    outcome.provider_termination_publication_fence.close()
    assert all(lease._test_closed for lease in case.state["control_leases"])


def test_main_reappearance_burns_lease_without_registering_termination(monkeypatch):
    case = _case(monkeypatch)
    observation = _inspect(case)
    capability = _capability(case, observation)
    case.state["inventory"] = replace(
        case.surviving_inventory,
        main_container_id=case.query.spawn_commit.provider_execution_id,
    )

    with pytest.raises(
        main_loss.RunActionPreReleaseMainLossError,
        match="exactly volume and keeper",
    ):
        _capture(case, capability)
    assert all(lease._test_closed for lease in case.state["control_leases"])


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


def test_classification_closes_descriptors_before_coordinator_checks(monkeypatch):
    case = _case(monkeypatch)

    first_observation = _inspect(case)
    second_observation = _inspect(case)

    assert run_action_pre_release_main_loss_observation_token(
        first_observation
    ) == run_action_pre_release_main_loss_observation_token(second_observation)
    assert all(lease._test_closed for lease in case.state["control_leases"])


def test_surviving_resource_substitution_rejects_loss_classification(monkeypatch):
    case = _case(monkeypatch)
    case.state["inventory"] = replace(
        case.surviving_inventory,
        keeper_container_id="f" * 64,
    )

    with pytest.raises(
        main_loss.RunActionPreReleaseMainLossError,
        match="exactly volume and keeper",
    ):
        _inspect(case)
    assert all(lease._test_closed for lease in case.state["control_leases"])


def test_foreign_settings_reject_before_opening_control_authority(monkeypatch):
    case = _case(monkeypatch)
    foreign_settings = replace(
        case.docker_settings,
        runtime_socket_path="/run/foreign-docker.sock",
    )

    with pytest.raises(
        main_loss.RunActionPreReleaseMainLossError,
        match="configured authority",
    ):
        main_loss.inspect_run_action_pre_release_main_loss(
            query=case.query,
            resource_manager=case.manager,
            helper_evidence=case.helper,
            init_source_evidence=case.init,
            docker_settings=foreign_settings,
        )
    assert case.state["control_checks"] == 0
    assert case.state["control_leases"] == []
