from __future__ import annotations

import copy

import pytest

import kapso.cross_run.launch.run_action_docker_inspect as docker_inspect
import kapso.cross_run.launch.run_action_terminal_inspection as terminal_inspection
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
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
from kapso.cross_run.launch.run_action_release_candidate import (
    _SystemRunActionReleaseClock,
)
from kapso.cross_run.launch.run_action_release_adoption import (
    RunActionReleasePresence,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationAllocation,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_barrier_contracts import _resolved_graph
from test_run_action_docker_inspect import (
    _MAIN_CONTAINER_ID,
    _terminal_context,
    _terminal_main_raw,
    _volume_raw,
)
from test_run_action_release_contracts import _activation_event
from test_run_action_release_contracts import (
    _security_observation as _release_security_observation,
)

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


class _ReleaseInspection:
    def __init__(self, presence, adoption):
        self.presence = presence
        self.adoption = adoption
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


class _SecurityAuthority:
    def observe_exact_descendant_of(self, **_arguments):
        raise AssertionError("terminal reinspection must not reauthorize security")


def _inspection_context(docker_settings):
    (
        prepared,
        activation,
        adoption,
        volume,
        command,
        helper,
        init,
    ) = _terminal_context(docker_settings)
    activation_event = _activation_event(
        _resolved_graph(prepared=prepared, activation=activation)
    )
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_authority=prepared.runtime_volume_authority,
    )
    query = RunActionCommittedSpawnQuery(
        preparation_allocation=allocation,
        activation_event=activation_event,
        workload_release_adoption=adoption,
    )
    inventory = DockerRunActionResourceInventory(
        preparation_allocation=allocation,
        volume_inspection_digest="sha256:" + "1" * 64,
        keeper_container_id=prepared.volume_keeper_evidence.container_id,
        main_container_id=_MAIN_CONTAINER_ID,
    )
    raw = _terminal_main_raw(
        prepared,
        adoption,
        volume,
        command,
        docker_settings,
    )
    return query, inventory, raw, command, helper, init


def _configured_settings():
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    return settings.docker, settings.launch


def _patch_physical_inspection(
    monkeypatch,
    *,
    inventory,
    volume_raw,
    main_inspections,
    release_inspection,
    host_boot_id,
):
    main_payloads = list(main_inspections)
    manager = object.__new__(DockerRunActionResourceManager)
    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "observe",
        lambda _self, _allocation: inventory,
    )
    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "inspect_volume",
        lambda _self, _inventory: copy.deepcopy(volume_raw),
    )

    def inspect_main(_self, _inventory):
        if not main_payloads:
            raise AssertionError("terminal inspection read the main too many times")
        return main_payloads.pop(0)

    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "inspect_main",
        inspect_main,
    )
    monkeypatch.setattr(
        terminal_inspection,
        "open_run_action_release_inspection",
        lambda **_arguments: release_inspection,
    )
    monkeypatch.setattr(
        terminal_inspection,
        "read_run_action_host_boot_id",
        lambda _descriptor: host_boot_id,
    )
    return manager, main_payloads


def test_terminal_inspection_retains_release_and_requires_two_equal_snapshots(
    monkeypatch,
):
    docker_settings, launch_settings = _configured_settings()
    query, inventory, raw, command, helper, init = _inspection_context(docker_settings)
    adoption = query.workload_release_adoption
    release_inspection = _ReleaseInspection(
        RunActionReleasePresence.PRESENT,
        adoption,
    )
    manager, remaining_payloads = _patch_physical_inspection(
        monkeypatch,
        inventory=inventory,
        volume_raw=_volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
        main_inspections=(copy.deepcopy(raw), copy.deepcopy(raw)),
        release_inspection=release_inspection,
        host_boot_id=adoption.workload_release_receipt.host_boot_id,
    )

    terminal = terminal_inspection.inspect_run_action_terminal(
        query=query,
        resource_manager=manager,
        command=command,
        helper_evidence=helper,
        init_source_evidence=init,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
    )

    assert terminal.workload_release_adoption_id == (
        adoption.workload_release_adoption_id
    )
    assert not remaining_payloads
    assert release_inspection.current_checks == 2
    assert release_inspection.closed


def test_terminal_inspection_rejects_absent_release_or_changing_terminal(
    monkeypatch,
):
    docker_settings, launch_settings = _configured_settings()
    query, inventory, raw, command, helper, init = _inspection_context(docker_settings)
    adoption = query.workload_release_adoption
    absent = _ReleaseInspection(RunActionReleasePresence.ABSENT, None)
    manager, _remaining_payloads = _patch_physical_inspection(
        monkeypatch,
        inventory=inventory,
        volume_raw=_volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
        main_inspections=(),
        release_inspection=absent,
        host_boot_id=adoption.workload_release_receipt.host_boot_id,
    )
    with pytest.raises(
        terminal_inspection.RunActionTerminalInspectionError,
        match="retained release",
    ):
        terminal_inspection.inspect_run_action_terminal(
            query=query,
            resource_manager=manager,
            command=command,
            helper_evidence=helper,
            init_source_evidence=init,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
        )
    changed = copy.deepcopy(raw)
    changed["State"]["FinishedAt"] = "2026-07-25T01:02:05.123456789Z"
    present = _ReleaseInspection(RunActionReleasePresence.PRESENT, adoption)
    manager, _remaining_payloads = _patch_physical_inspection(
        monkeypatch,
        inventory=inventory,
        volume_raw=_volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
        main_inspections=(copy.deepcopy(raw), changed),
        release_inspection=present,
        host_boot_id=adoption.workload_release_receipt.host_boot_id,
    )
    with pytest.raises(
        terminal_inspection.RunActionTerminalInspectionError,
        match="changed",
    ):
        terminal_inspection.inspect_run_action_terminal(
            query=query,
            resource_manager=manager,
            command=command,
            helper_evidence=helper,
            init_source_evidence=init,
            docker_settings=docker_settings,
            launch_settings=launch_settings,
        )


def test_terminal_reinspection_consumes_one_capability_and_seals_the_digest(
    monkeypatch,
):
    docker_settings, launch_settings = _configured_settings()
    query, inventory, raw, command, helper, init = _inspection_context(docker_settings)
    adoption = query.workload_release_adoption
    release_inspection = _ReleaseInspection(
        RunActionReleasePresence.PRESENT,
        adoption,
    )
    manager, remaining_payloads = _patch_physical_inspection(
        monkeypatch,
        inventory=inventory,
        volume_raw=_volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
        main_inspections=(copy.deepcopy(raw), copy.deepcopy(raw)),
        release_inspection=release_inspection,
        host_boot_id=adoption.workload_release_receipt.host_boot_id,
    )
    _normalized, normalized_payload, _raw_size_bytes = (
        docker_inspect._snapshot_container_inspection(
            raw,
            "test terminal inspection",
        )
    )
    observation = RunActionCommittedSpawnObservation(
        state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
        observation_token=tree_or_blob_digest(normalized_payload),
    )
    capability = RunActionCommittedContinuationCapability(
        query=query,
        observation=observation,
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=_SystemRunActionReleaseClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )

    class _TerminalReinspectionAdapter:
        def __init__(self):
            self.terminal = None

        def continue_committed_once(self, active_capability):
            self.terminal = terminal_inspection.reinspect_run_action_terminal(
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
                match="terminal reinspection lacks exact live continuation authority",
            ):
                terminal_inspection.reinspect_run_action_terminal(
                    capability=active_capability,
                    resource_manager=manager,
                    command=command,
                    helper_evidence=helper,
                    init_source_evidence=init,
                    docker_settings=docker_settings,
                    launch_settings=launch_settings,
                )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
            )

    adapter = _TerminalReinspectionAdapter()
    outcome = capability._invoke_once(adapter)
    with pytest.raises(RunActionRecoveryError, match="spent, cloned, or foreign"):
        capability._invoke_once(adapter)

    assert outcome.state is RunActionContinuationState.PENDING
    assert adapter.terminal.complete_inspection_digest == observation.observation_token
    assert not remaining_payloads


def test_terminal_continuation_rejects_an_adapter_that_skips_trusted_reinspection():
    docker_settings, _launch_settings = _configured_settings()
    query, _inventory, raw, _command, _helper, _init = _inspection_context(
        docker_settings
    )
    _normalized, normalized_payload, _raw_size_bytes = (
        docker_inspect._snapshot_container_inspection(
            raw,
            "test terminal inspection",
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
        release_clock=_SystemRunActionReleaseClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )

    class _BypassingAdapter:
        @staticmethod
        def continue_committed_once(_capability):
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="terminal continuation lacks its trusted reinspection",
    ):
        capability._invoke_once(_BypassingAdapter())
