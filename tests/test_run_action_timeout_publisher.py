"""Sealed timeout publication without signal or termination authority."""

from __future__ import annotations

import copy
import os
import stat
from dataclasses import replace
from types import SimpleNamespace

import pytest

import kapso.cross_run.launch.run_action_control_candidate as control_candidate_module
import kapso.cross_run.launch.run_action_release_adoption as release_adoption_module
import kapso.cross_run.launch.run_action_timeout_adoption as timeout_adoption_module
import kapso.cross_run.launch.run_action_timeout_publisher as timeout_publisher
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_control_candidate import (
    _RunActionControlFileTransition,
    _RunActionFrozenControlFileCandidate,
    _RunActionLinkedControlFileEvidence,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    RunActionControlDirectoryLease,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionProviderTerminationReason,
    RunActionTimeoutDirective,
    RunActionTimeoutDirectivePublicationReceipt,
)
from kapso.cross_run.launch.run_action_timeout_adoption import (
    RunActionTimeoutInspectionLease,
)
from test_run_action_docker_inspect import (
    _running_main_raw,
    _volume_raw,
)
from test_run_action_release_contracts import (
    _security_observation as _release_security_observation,
)
from test_run_action_terminal_inspection import (
    _configured_settings,
    _inspection_context,
)
from test_run_action_termination_contracts import _termination_graph


class _SecurityAuthority:
    def observe_exact_descendant_of(self, **_arguments):
        raise AssertionError("timeout publication must not reauthorize release")


class _ProjectedControlOS:
    """Project one real control tree into the prepared contract namespace."""

    def __init__(
        self,
        *,
        release_inode: int,
        real_release_inode: int,
        control_device: int,
        control_inode: int,
    ) -> None:
        self._release_inode = release_inode
        self._real_release_inode = real_release_inode
        self._control_device = control_device
        self._control_inode = control_inode

    def __getattr__(self, name):
        return getattr(os, name)

    def fstat(self, descriptor):
        observed = os.fstat(descriptor)
        if stat.S_ISDIR(observed.st_mode):
            projected_inode = self._control_inode
        elif stat.S_ISREG(observed.st_mode):
            projected_inode = (
                self._release_inode
                if observed.st_ino == self._real_release_inode
                else self._release_inode + 1
            )
        else:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_uid=observed.st_uid,
            st_gid=observed.st_gid,
            st_nlink=observed.st_nlink,
            st_size=observed.st_size,
            st_dev=self._control_device,
            st_ino=projected_inode,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )


def _physical_control_lease(control_path):
    entries = tuple(sorted(path.name for path in control_path.iterdir()))
    topologies = {
        topology.entries: topology for topology in RunActionControlDirectoryTopology
    }
    if entries not in topologies:
        raise AssertionError("test control directory has an invalid topology")
    topology = topologies[entries]
    descriptor = os.open(
        control_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    lease = object.__new__(RunActionControlDirectoryLease)
    lease._control_descriptor = descriptor
    lease._topology = topology
    closed = {"value": False}

    def require_current():
        if closed["value"]:
            raise AssertionError("test control lease is closed")
        current_entries = tuple(sorted(os.listdir(lease._control_descriptor)))
        if current_entries != topology.entries:
            raise AssertionError("test control topology changed")

    def close():
        if closed["value"]:
            raise AssertionError("test control lease closed twice")
        closed["value"] = True
        os.close(descriptor)

    lease.require_current = require_current
    lease.close = close
    return lease


def _timeout_inspection(topology, adoption, publication=None):
    inspection = object.__new__(RunActionTimeoutInspectionLease)
    inspection._topology = topology
    inspection._timeout_directive_publication = publication
    inspection._release_inspection = SimpleNamespace(adoption=adoption)
    inspection.current_checks = 0
    inspection.closed = False

    def require_current():
        if inspection.closed:
            raise AssertionError("timeout inspection is closed")
        inspection.current_checks += 1

    def close():
        if inspection.closed:
            raise AssertionError("timeout inspection closed twice")
        inspection.closed = True

    def duplicate(*, descriptors, _authority):
        control_descriptor = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
        descriptors.callback(os.close, control_descriptor)
        release_descriptor = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
        descriptors.callback(os.close, release_descriptor)
        return control_descriptor, release_descriptor

    inspection.require_current = require_current
    inspection.close = close
    inspection._duplicate_timeout_publication_descriptors = duplicate
    return inspection


def _publication_for(query, directive):
    adoption = query.workload_release_adoption
    prepared = query.prepared_execution
    control = prepared.control_directory
    authority = prepared.runtime_volume_authority
    payload = directive.to_json_bytes()
    return RunActionTimeoutDirectivePublicationReceipt.mint(
        timeout_directive=directive,
        workload_release_adoption_id=adoption.workload_release_adoption_id,
        prepared_control_directory_id=control.prepared_runtime_directory_id,
        control_mount_id=control.mount_id,
        control_device=control.device,
        control_inode=control.inode,
        release_mount_id=adoption.release_mount_id,
        release_device=adoption.release_device,
        release_inode=adoption.release_inode,
        relative_path="control/timeout",
        file_type="regular",
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        mode=0o400,
        link_count=1,
        size_bytes=len(payload),
        content_digest=tree_or_blob_digest(payload),
        timeout_mount_id=control.mount_id,
        timeout_device=control.device,
        timeout_inode=adoption.release_inode + 1,
    )


def _case(
    monkeypatch,
    clock_samples,
    *,
    adoption_crash=False,
    final_running_changed=False,
):
    docker_settings, launch_settings = _configured_settings()
    query, inventory, _terminal_raw, command, helper, init = _inspection_context(
        docker_settings
    )
    adoption = query.workload_release_adoption
    prepared = query.prepared_execution
    volume_raw = _volume_raw(prepared.runtime_volume_authority, docker_settings)
    volume = observe_runtime_volume(
        volume_raw,
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        docker_settings,
    )
    running_raw = _running_main_raw(
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        command,
        docker_settings,
    )
    released_running = (
        adoption.workload_release_receipt.resolved_workload_observation.running_container_observation
    )
    running_raw["State"]["Pid"] = released_running.init_process_id
    running_raw["State"]["StartedAt"] = released_running.started_at
    manager = object.__new__(DockerRunActionResourceManager)
    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "runtime_settings",
        property(lambda _self: docker_settings),
    )
    provider_calls = []
    main_payloads = [
        copy.deepcopy(running_raw),
        copy.deepcopy(running_raw),
        copy.deepcopy(running_raw),
    ]
    if final_running_changed:
        main_payloads[-1]["State"]["StartedAt"] = "2026-07-25T01:02:04.123456789Z"

    def observe(_self, _allocation):
        provider_calls.append("observe")
        return inventory

    def inspect_volume(_self, _inventory):
        provider_calls.append("inspect_volume")
        return copy.deepcopy(volume_raw)

    def inspect_main(_self, _inventory):
        provider_calls.append("inspect_main")
        if not main_payloads:
            raise AssertionError("timeout publisher inspected main too often")
        return main_payloads.pop(0)

    monkeypatch.setattr(DockerRunActionResourceManager, "observe", observe)
    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "inspect_volume",
        inspect_volume,
    )
    monkeypatch.setattr(DockerRunActionResourceManager, "inspect_main", inspect_main)
    shared = {
        "publication": None,
        "link_count": 0,
        "prepare_count": 0,
        "clock_count": 0,
    }
    released_inspections = []
    adopted_inspections = []

    def open_inspection(**_arguments):
        if shared["publication"] is None:
            inspection = _timeout_inspection(
                RunActionControlDirectoryTopology.RELEASED,
                adoption,
            )
            released_inspections.append(inspection)
            return inspection
        if adoption_crash:
            raise RuntimeError("injected crash after timeout link")
        inspection = _timeout_inspection(
            RunActionControlDirectoryTopology.TIMED_OUT,
            adoption,
            shared["publication"],
        )
        adopted_inspections.append(inspection)
        return inspection

    monkeypatch.setattr(
        timeout_publisher,
        "open_run_action_timeout_inspection",
        open_inspection,
    )
    monkeypatch.setattr(
        timeout_publisher,
        "read_run_action_host_boot_id",
        lambda _descriptor: adoption.workload_release_receipt.host_boot_id,
    )
    monkeypatch.setattr(
        timeout_publisher,
        "open_run_action_anonymous_file",
        lambda _descriptor, _mode: os.open(
            "/dev/null",
            os.O_RDONLY | os.O_CLOEXEC,
        ),
    )

    def candidate_init(candidate, **arguments):
        candidate._payload = arguments["payload"]
        candidate._transition = arguments["transition"]

    def begin_publication(candidate, expected_payload, *, _authority):
        if expected_payload != candidate._payload:
            raise AssertionError("candidate payload was substituted")
        return candidate._payload

    def prepare_link(_candidate, *, _authority):
        shared["prepare_count"] += 1
        return None

    def link_once(candidate, *, _authority):
        directive = RunActionTimeoutDirective.from_json_bytes(candidate._payload)
        publication = _publication_for(query, directive)
        shared["publication"] = publication
        shared["link_count"] += 1
        return _RunActionLinkedControlFileEvidence(
            transition=_RunActionControlFileTransition.TIMEOUT,
            final_file_name="timeout",
            mount_id=publication.timeout_mount_id,
            device=publication.timeout_device,
            inode=publication.timeout_inode,
            owner_user_id=publication.owner_user_id,
            owner_group_id=publication.owner_group_id,
            mode=publication.mode,
            link_count=publication.link_count,
            size_bytes=publication.size_bytes,
            content_digest=publication.content_digest,
        )

    monkeypatch.setattr(
        _RunActionFrozenControlFileCandidate, "__init__", candidate_init
    )
    monkeypatch.setattr(
        _RunActionFrozenControlFileCandidate,
        "_begin_publication",
        begin_publication,
    )
    monkeypatch.setattr(
        _RunActionFrozenControlFileCandidate,
        "_prepare_authorized_link_once",
        prepare_link,
    )
    monkeypatch.setattr(
        _RunActionFrozenControlFileCandidate,
        "_link_prepared_once",
        link_once,
    )
    monkeypatch.setattr(
        _RunActionFrozenControlFileCandidate,
        "close",
        lambda _candidate: None,
    )
    clock = _SystemRunActionClock()
    samples = iter(clock_samples)

    def sample_boottime():
        shared["clock_count"] += 1
        if shared["clock_count"] == 3 and shared["prepare_count"] != 1:
            raise AssertionError("final timeout clock preceded candidate preparation")
        return next(samples)

    clock.boottime_nanoseconds = sample_boottime
    capability = RunActionCommittedContinuationCapability(
        query=query,
        observation=RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            observation_token=(
                adoption.workload_release_receipt.resolved_workload_observation.running_container_observation.complete_inspection_digest
            ),
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=clock,
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
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
        provider_calls=provider_calls,
        main_payloads=main_payloads,
        shared=shared,
        released_inspections=released_inspections,
        adopted_inspections=adopted_inspections,
    )


def _publish(case, capability):
    return timeout_publisher.publish_run_action_timeout_once(
        capability=capability,
        resource_manager=case.manager,
        command=case.command,
        helper_evidence=case.helper,
        init_source_evidence=case.init,
        docker_settings=case.docker_settings,
        launch_settings=case.launch_settings,
    )


def test_physical_timeout_link_is_freshly_adopted_end_to_end(
    tmp_path,
    monkeypatch,
):
    docker_settings, launch_settings = _configured_settings()
    query, inventory, _terminal_raw, command, helper, init = _inspection_context(
        docker_settings
    )
    adoption = query.workload_release_adoption
    release = adoption.workload_release_receipt
    prepared = query.prepared_execution
    control = prepared.control_directory
    authority = prepared.runtime_volume_authority
    control_path = tmp_path / "control"
    control_path.mkdir(mode=0o700)
    os.chown(control_path, control.owner_user_id, control.owner_group_id)
    release_path = control_path / "release"
    release_path.write_bytes(release.to_json_bytes())
    os.chown(release_path, authority.owner_user_id, authority.owner_group_id)
    release_path.chmod(0o400)
    projected_os = _ProjectedControlOS(
        release_inode=adoption.release_inode,
        real_release_inode=release_path.stat().st_ino,
        control_device=control.device,
        control_inode=control.inode,
    )
    monkeypatch.setattr(
        release_adoption_module,
        "open_run_action_control_directory",
        lambda _prepared: _physical_control_lease(control_path),
    )
    for module in (
        release_adoption_module,
        timeout_adoption_module,
        control_candidate_module,
    ):
        monkeypatch.setattr(
            module,
            "read_run_action_descriptor_mount_id",
            lambda _descriptor, _byte_limit: control.mount_id,
        )
        monkeypatch.setattr(module, "os", projected_os)
    monkeypatch.setattr(
        timeout_publisher,
        "read_run_action_host_boot_id",
        lambda _descriptor: release.host_boot_id,
    )

    volume_raw = _volume_raw(authority, docker_settings)
    volume = observe_runtime_volume(
        volume_raw,
        prepared.preparation_claim,
        authority,
        docker_settings,
    )
    running_raw = _running_main_raw(
        prepared.preparation_claim,
        authority,
        volume,
        command,
        docker_settings,
    )
    released_running = (
        release.resolved_workload_observation.running_container_observation
    )
    running_raw["State"]["Pid"] = released_running.init_process_id
    running_raw["State"]["StartedAt"] = released_running.started_at
    main_payloads = [
        copy.deepcopy(running_raw),
        copy.deepcopy(running_raw),
        copy.deepcopy(running_raw),
    ]
    manager = object.__new__(DockerRunActionResourceManager)
    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "runtime_settings",
        property(lambda _self: docker_settings),
    )
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
            raise AssertionError("physical timeout path inspected main too often")
        return main_payloads.pop(0)

    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "inspect_main",
        inspect_main,
    )
    clock = _SystemRunActionClock()
    samples = iter(
        (
            release.execution_deadline_boottime_nanoseconds,
            release.execution_deadline_boottime_nanoseconds + 1,
            release.execution_deadline_boottime_nanoseconds + 2,
        )
    )
    clock.boottime_nanoseconds = lambda: next(samples)
    capability = RunActionCommittedContinuationCapability(
        query=query,
        observation=RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            observation_token=released_running.complete_inspection_digest,
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=clock,
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )

    class _PhysicalTimeoutAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            publication = timeout_publisher.publish_run_action_timeout_once(
                capability=active_capability,
                resource_manager=manager,
                command=command,
                helper_evidence=helper,
                init_source_evidence=init,
                docker_settings=docker_settings,
                launch_settings=launch_settings,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.TIMEOUT_PUBLISHED,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=publication,
            )

    outcome = capability._invoke_once(_PhysicalTimeoutAdapter())

    timeout_path = control_path / "timeout"
    assert timeout_path.read_bytes() == (
        outcome.timeout_directive_publication.timeout_directive.to_json_bytes()
    )
    assert tuple(sorted(path.name for path in control_path.iterdir())) == (
        "release",
        "timeout",
    )
    with timeout_adoption_module.open_run_action_timeout_inspection(
        activation_event=query.activation_event,
        launch_settings=launch_settings,
    ) as inspection:
        assert inspection.topology is RunActionControlDirectoryTopology.TIMED_OUT
        assert (
            inspection.timeout_directive_publication
            == outcome.timeout_directive_publication
        )
        inspection.require_current()
    assert not main_payloads


def test_exact_deadline_publishes_and_registers_without_provider_mutation(monkeypatch):
    query = _inspection_context(_configured_settings()[0])[0]
    release = query.workload_release_adoption.workload_release_receipt
    case = _case(
        monkeypatch,
        (
            release.execution_deadline_boottime_nanoseconds,
            release.execution_deadline_boottime_nanoseconds + 1,
            release.execution_deadline_boottime_nanoseconds + 2,
        ),
    )

    class _TimeoutAdapter:
        @staticmethod
        def continue_committed_once(capability):
            publication = _publish(case, capability)
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.TIMEOUT_PUBLISHED,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=publication,
            )

    outcome = case.capability._invoke_once(_TimeoutAdapter())

    assert outcome.timeout_directive_publication == case.shared["publication"]
    assert case.shared["link_count"] == 1
    assert case.provider_calls == [
        "observe",
        "inspect_volume",
        "inspect_main",
        "inspect_main",
        "observe",
        "inspect_main",
        "observe",
    ]
    assert not case.main_payloads
    assert len(case.adopted_inspections) == 1


def test_not_due_check_is_pending_and_one_shot(monkeypatch):
    query = _inspection_context(_configured_settings()[0])[0]
    deadline = (
        query.workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    )
    case = _case(monkeypatch, (deadline - 1,))

    class _NotDueAdapter:
        @staticmethod
        def continue_committed_once(capability):
            assert _publish(case, capability) is None
            with pytest.raises(
                RunActionRecoveryError,
                match="lacks exact live released authority",
            ):
                _publish(case, capability)
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=None,
            )

    outcome = case.capability._invoke_once(_NotDueAdapter())

    assert outcome.state is RunActionContinuationState.PENDING
    assert case.provider_calls == []
    assert case.shared["link_count"] == 0


def test_publisher_rejects_docker_settings_not_joined_to_its_manager(monkeypatch):
    case = _case(monkeypatch, ())
    foreign_settings = replace(
        case.docker_settings,
        runtime_socket_path="/run/foreign-docker.sock",
    )

    with pytest.raises(
        timeout_publisher.RunActionTimeoutPublicationError,
        match="configured authority",
    ):
        timeout_publisher.publish_run_action_timeout_once(
            capability=case.capability,
            resource_manager=case.manager,
            command=case.command,
            helper_evidence=case.helper,
            init_source_evidence=case.init,
            docker_settings=foreign_settings,
            launch_settings=case.launch_settings,
        )

    assert case.provider_calls == []
    assert case.shared["clock_count"] == 0
    assert case.shared["link_count"] == 0


def test_post_containment_resume_publishes_for_immediate_kill(monkeypatch):
    query = _inspection_context(_configured_settings()[0])[0]
    containment = (
        query.workload_release_adoption.workload_release_receipt.containment_deadline_boottime_nanoseconds
    )
    case = _case(
        monkeypatch,
        (
            containment + 1,
            containment + 2,
            containment + 3,
        ),
    )

    class _LateAdapter:
        @staticmethod
        def continue_committed_once(capability):
            publication = _publish(case, capability)
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.TIMEOUT_PUBLISHED,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=publication,
            )

    outcome = case.capability._invoke_once(_LateAdapter())

    assert outcome.state is RunActionContinuationState.TIMEOUT_PUBLISHED
    assert case.shared["link_count"] == 1
    assert (
        outcome.timeout_directive_publication.timeout_directive.observed_before_boottime_nanoseconds
        > containment
    )


def test_final_authorization_clock_regression_prevents_link(monkeypatch):
    query = _inspection_context(_configured_settings()[0])[0]
    release = query.workload_release_adoption.workload_release_receipt
    case = _case(
        monkeypatch,
        (
            release.execution_deadline_boottime_nanoseconds,
            release.execution_deadline_boottime_nanoseconds + 2,
            release.execution_deadline_boottime_nanoseconds + 1,
        ),
    )

    class _RegressedClockAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _publish(case, capability)

    with pytest.raises(
        RunActionRecoveryError,
        match="clock regressed",
    ):
        case.capability._invoke_once(_RegressedClockAdapter())
    assert case.shared["prepare_count"] == 1
    assert case.shared["link_count"] == 0
    assert case.shared["publication"] is None


def test_crash_between_link_and_adoption_leaves_only_the_durable_fact(monkeypatch):
    query = _inspection_context(_configured_settings()[0])[0]
    deadline = (
        query.workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    )
    case = _case(
        monkeypatch,
        (deadline, deadline + 1, deadline + 2),
        adoption_crash=True,
    )

    class _CrashAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _publish(case, capability)

    with pytest.raises(RuntimeError, match="after timeout link"):
        case.capability._invoke_once(_CrashAdapter())
    assert case.shared["link_count"] == 1
    assert (
        type(case.shared["publication"]) is RunActionTimeoutDirectivePublicationReceipt
    )
    assert all(
        "signal" not in call and "stop" not in call for call in case.provider_calls
    )


def test_final_running_substitution_prevents_the_irreversible_link(monkeypatch):
    query = _inspection_context(_configured_settings()[0])[0]
    deadline = (
        query.workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    )
    case = _case(
        monkeypatch,
        (deadline, deadline + 1),
        final_running_changed=True,
    )

    class _ChangedAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _publish(case, capability)

    with pytest.raises(
        timeout_publisher.RunActionTimeoutPublicationError,
        match="lost the running occurrence",
    ):
        case.capability._invoke_once(_ChangedAdapter())
    assert case.shared["link_count"] == 0


def test_already_timed_out_running_occurrence_rejects_noop_pending():
    docker_settings = _configured_settings()[0]
    query = _inspection_context(docker_settings, timed_out=True)[0]
    release = query.workload_release_adoption.workload_release_receipt
    capability = RunActionCommittedContinuationCapability(
        query=query,
        observation=RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            observation_token=(
                release.resolved_workload_observation.running_container_observation.complete_inspection_digest
            ),
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=_SystemRunActionClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )

    class _PendingContainmentAdapter:
        @staticmethod
        def continue_committed_once(_capability):
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=None,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="nonterminal continuation consumed terminal outcome authority",
    ):
        capability._invoke_once(_PendingContainmentAdapter())


def test_timeout_outcome_cannot_substitute_the_registered_publication(monkeypatch):
    query = _inspection_context(_configured_settings()[0])[0]
    deadline = (
        query.workload_release_adoption.workload_release_receipt.execution_deadline_boottime_nanoseconds
    )
    case = _case(monkeypatch, (deadline, deadline + 1, deadline + 2))
    foreign_publication = _termination_graph(
        RunActionProviderTerminationReason.TIMEOUT
    ).timeout_directive_publication

    class _SubstitutionAdapter:
        @staticmethod
        def continue_committed_once(capability):
            _publish(case, capability)
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.TIMEOUT_PUBLISHED,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=foreign_publication,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="nonterminal continuation consumed terminal outcome authority",
    ):
        case.capability._invoke_once(_SubstitutionAdapter())
