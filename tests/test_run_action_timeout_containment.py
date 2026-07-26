"""Exact TERM/KILL containment and terminal timeout registration."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import kapso.cross_run.docker.runtime as runtime_module
import kapso.cross_run.launch.run_action_docker_inspect as docker_inspect
import kapso.cross_run.launch.run_action_timeout_containment as containment_module
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import PinnedDockerRuntime
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_containment_contracts import (
    RunActionTimeoutContainmentSignal,
    RunActionTimeoutContainmentState,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_running_barrier_main_container,
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
)
from kapso.cross_run.launch.run_action_timeout_containment import (
    contain_run_action_timeout_once,
    DockerRunActionContainmentManager,
    RunActionTimeoutContainmentError,
)
from kapso.cross_run.launch.run_action_timeout_termination import (
    capture_run_action_timeout_termination,
)
from kapso.cross_run.process import BoundedProcessOutcome, BoundedProcessResult
from test_cross_run_docker_runtime import _info, _version
from test_run_action_docker_inspect import (
    _running_main_raw,
    _volume_raw,
)
from test_run_action_release_contracts import (
    _security_observation as _release_security_observation,
)
from test_run_action_terminal_inspection import (
    _configured_settings,
    _ControlInspection,
    _inspection_context,
    _patch_physical_inspection,
    _SecurityAuthority,
)
from test_run_action_timeout_publisher import _timeout_inspection

_TEST_DOCKER_BYTES = b"run-action timeout containment Docker"


class _ContainmentDockerRunner:
    def __init__(
        self,
        settings,
        *,
        signal_failure=False,
        malformed_signal_success=False,
    ):
        self.settings = settings
        self.signal_failure = signal_failure
        self.malformed_signal_success = malformed_signal_success
        self.requests = []
        self.signal_commands = []

    def run(self, request):
        self.requests.append(request)
        arguments = request.argv[5:]
        if arguments == ("version", "--format", "{{json .}}"):
            stdout = _json_line(_version(self.settings))
            returncode = 0
            stderr = b""
        elif arguments == ("info", "--format", "{{json .}}"):
            stdout = _json_line(_info(self.settings))
            returncode = 0
            stderr = b""
        elif (
            len(arguments) == 5
            and arguments[:3] == ("container", "kill", "--signal")
            and arguments[3] in {"SIGTERM", "SIGKILL"}
        ):
            self.signal_commands.append(arguments)
            if self.signal_failure:
                stdout = b""
                returncode = 1
                stderr = b"container was already terminal\n"
            elif self.malformed_signal_success:
                stdout = b"unexpected successful response\n"
                returncode = 0
                stderr = b""
            else:
                stdout = f"{arguments[-1]}\n".encode()
                returncode = 0
                stderr = b""
        else:
            raise AssertionError(f"unexpected Docker command: {arguments}")
        return BoundedProcessResult(
            request=request,
            outcome=BoundedProcessOutcome.COMPLETED,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_bytes_observed=len(stdout),
            stderr_bytes_observed=len(stderr),
            duration_seconds=0.0,
        )


def _json_line(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def _case(
    tmp_path,
    monkeypatch,
    selected_at,
    *,
    pre_signal_terminal=False,
    transition_before_signal=False,
    post_signal_terminal=False,
    signal_failure=False,
    malformed_signal_success=False,
):
    docker_settings, launch_settings = _configured_settings()
    docker_settings = replace(
        docker_settings,
        runtime_executable_digest=tree_or_blob_digest(_TEST_DOCKER_BYTES),
    )

    def read_executable(_path, expected_digest):
        assert expected_digest == docker_settings.runtime_executable_digest
        return _TEST_DOCKER_BYTES

    monkeypatch.setattr(
        runtime_module,
        "read_verified_root_executable",
        read_executable,
    )
    monkeypatch.setattr(runtime_module, "_require_runtime_socket", lambda _path: None)
    if not tmp_path.exists():
        tmp_path.mkdir(mode=0o700)
    tmp_path.chmod(0o700)
    runner = _ContainmentDockerRunner(
        docker_settings,
        signal_failure=signal_failure,
        malformed_signal_success=malformed_signal_success,
    )
    runtime = PinnedDockerRuntime(
        trusted_root=tmp_path.resolve(),
        settings=docker_settings,
        process_runner=runner,
    )
    resource_manager = DockerRunActionResourceManager(runtime)
    containment_manager = DockerRunActionContainmentManager(runtime)
    query, inventory, terminal_raw, command, helper, init = _inspection_context(
        docker_settings,
        timed_out=True,
    )
    prepared = query.prepared_execution
    adoption = query.workload_release_adoption
    publication = query.timeout_directive_publication
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
    running = observe_running_barrier_main_container(
        running_raw,
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        command,
        helper,
        init,
        docker_settings,
    )
    if pre_signal_terminal:
        main_payloads = [copy.deepcopy(terminal_raw), copy.deepcopy(terminal_raw)]
    elif transition_before_signal:
        main_payloads = [
            copy.deepcopy(running_raw),
            copy.deepcopy(terminal_raw),
            copy.deepcopy(terminal_raw),
        ]
    elif post_signal_terminal:
        main_payloads = [
            copy.deepcopy(running_raw),
            copy.deepcopy(running_raw),
            copy.deepcopy(running_raw),
            copy.deepcopy(terminal_raw),
            copy.deepcopy(terminal_raw),
        ]
    else:
        main_payloads = [
            copy.deepcopy(running_raw),
            copy.deepcopy(running_raw),
            copy.deepcopy(running_raw),
            copy.deepcopy(running_raw),
            copy.deepcopy(running_raw),
        ]
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
            raise AssertionError("containment inspected main too many times")
        return main_payloads.pop(0)

    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "inspect_main",
        inspect_main,
    )
    control_inspection = _timeout_inspection(
        RunActionControlDirectoryTopology.TIMED_OUT,
        adoption,
        publication,
    )
    monkeypatch.setattr(
        containment_module,
        "open_run_action_timeout_inspection",
        lambda **_arguments: control_inspection,
    )
    monkeypatch.setattr(
        containment_module,
        "read_run_action_host_boot_id",
        lambda _descriptor: publication.timeout_directive.host_boot_id,
    )
    clock = _SystemRunActionClock()
    clock.boottime_nanoseconds = lambda: selected_at
    capability = RunActionCommittedContinuationCapability(
        query=query,
        observation=RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            observation_token=running.complete_inspection_digest,
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
        inventory=inventory,
        resource_manager=resource_manager,
        containment_manager=containment_manager,
        command=command,
        helper=helper,
        init=init,
        docker_settings=docker_settings,
        launch_settings=launch_settings,
        runner=runner,
        main_payloads=main_payloads,
        control_inspection=control_inspection,
        runtime=runtime,
    )


def _contain(case, capability):
    return contain_run_action_timeout_once(
        capability=capability,
        resource_manager=case.resource_manager,
        containment_manager=case.containment_manager,
        command=case.command,
        helper_evidence=case.helper,
        init_source_evidence=case.init,
        docker_settings=case.docker_settings,
        launch_settings=case.launch_settings,
    )


def _invoke_pending(case):
    class _ContainmentAdapter:
        def __init__(self):
            self.result = None

        def continue_committed_once(self, capability):
            self.result = _contain(case, capability)
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=None,
            )

    adapter = _ContainmentAdapter()
    outcome = case.capability._invoke_once(adapter)
    return outcome, adapter.result


@pytest.mark.parametrize(
    ("clock_offset", "expected_signal"),
    (
        (-1, RunActionTimeoutContainmentSignal.TERMINATE),
        (0, RunActionTimeoutContainmentSignal.KILL),
        (1, RunActionTimeoutContainmentSignal.KILL),
    ),
)
def test_absolute_containment_deadline_selects_term_or_kill(
    tmp_path,
    monkeypatch,
    clock_offset,
    expected_signal,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    case = _case(tmp_path, monkeypatch, deadline + clock_offset)

    outcome, result = _invoke_pending(case)

    assert outcome.state is RunActionContinuationState.PENDING
    assert result.signal is expected_signal
    assert result.signal_dispatch_confirmed
    assert result.state is RunActionTimeoutContainmentState.RUNNING
    assert case.runner.signal_commands == [
        (
            "container",
            "kill",
            "--signal",
            expected_signal.value,
            case.query.spawn_commit.provider_execution_id,
        )
    ]
    assert not case.main_payloads


def test_fresh_recovery_reissues_term_before_the_same_absolute_deadline(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    signals = []

    for attempt_name in ("first", "second"):
        case = _case(
            tmp_path / attempt_name,
            monkeypatch,
            deadline - 1,
        )
        _outcome, result = _invoke_pending(case)
        signals.append(result.signal)

    assert signals == [
        RunActionTimeoutContainmentSignal.TERMINATE,
        RunActionTimeoutContainmentSignal.TERMINATE,
    ]


def test_crash_after_signal_reissues_kill_on_fresh_recovery(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    crashed_case = _case(tmp_path / "crashed", monkeypatch, deadline)
    original_observer = containment_module._observe_stable_main_occurrence
    observation_count = {"value": 0}

    def crash_after_signal(**arguments):
        observation_count["value"] += 1
        if observation_count["value"] == 2:
            raise RuntimeError("injected crash after containment signal")
        return original_observer(**arguments)

    monkeypatch.setattr(
        containment_module,
        "_observe_stable_main_occurrence",
        crash_after_signal,
    )

    class _CrashingAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _contain(crashed_case, capability)

    with pytest.raises(RuntimeError, match="injected crash"):
        crashed_case.capability._invoke_once(_CrashingAdapter())
    assert len(crashed_case.runner.signal_commands) == 1

    monkeypatch.setattr(
        containment_module,
        "_observe_stable_main_occurrence",
        original_observer,
    )
    recovered_case = _case(tmp_path / "recovered", monkeypatch, deadline)
    _outcome, result = _invoke_pending(recovered_case)

    assert result.signal is RunActionTimeoutContainmentSignal.KILL
    assert len(recovered_case.runner.signal_commands) == 1


@pytest.mark.parametrize(
    "transition_before_signal",
    (False, True),
)
def test_natural_terminal_before_signal_completes_without_mutation(
    tmp_path,
    monkeypatch,
    transition_before_signal,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    case = _case(
        tmp_path,
        monkeypatch,
        deadline,
        pre_signal_terminal=not transition_before_signal,
        transition_before_signal=transition_before_signal,
    )

    outcome, result = _invoke_pending(case)

    assert outcome.state is RunActionContinuationState.PENDING
    assert result.signal is None
    assert not result.signal_dispatch_confirmed
    assert result.state is RunActionTimeoutContainmentState.TERMINAL
    assert case.runner.signal_commands == []
    assert not case.main_payloads


def test_ambiguous_signal_is_admitted_only_after_stable_exact_terminal(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    terminal_case = _case(
        tmp_path / "terminal",
        monkeypatch,
        deadline,
        post_signal_terminal=True,
        signal_failure=True,
    )

    _outcome, terminal_result = _invoke_pending(terminal_case)

    assert terminal_result.signal is RunActionTimeoutContainmentSignal.KILL
    assert not terminal_result.signal_dispatch_confirmed
    assert terminal_result.state is RunActionTimeoutContainmentState.TERMINAL


def test_ambiguous_signal_with_still_running_occurrence_fails_loud(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    case = _case(
        tmp_path,
        monkeypatch,
        deadline,
        signal_failure=True,
    )

    class _FailingAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _contain(case, capability)

    with pytest.raises(
        RunActionTimeoutContainmentError,
        match="not safely resolved",
    ):
        case.capability._invoke_once(_FailingAdapter())


def test_malformed_successful_signal_response_fails_loud(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    case = _case(
        tmp_path,
        monkeypatch,
        deadline,
        malformed_signal_success=True,
    )

    class _MalformedSuccessAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _contain(case, capability)

    with pytest.raises(
        RunActionTimeoutContainmentError,
        match="not safely resolved",
    ):
        case.capability._invoke_once(_MalformedSuccessAdapter())
    assert len(case.runner.signal_commands) == 1


def test_post_signal_inventory_substitution_fails_loud(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    case = _case(tmp_path, monkeypatch, deadline)
    substituted_inventory = replace(
        case.inventory,
        main_container_id=(
            "f" * 64 if case.inventory.main_container_id != "f" * 64 else "e" * 64
        ),
    )
    monkeypatch.setattr(
        DockerRunActionResourceManager,
        "observe",
        lambda _self, _allocation: (
            substituted_inventory if case.runner.signal_commands else case.inventory
        ),
    )

    class _SubstitutedInventoryAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _contain(case, capability)

    with pytest.raises(
        RunActionTimeoutContainmentError,
        match="not safely resolved",
    ):
        case.capability._invoke_once(_SubstitutedInventoryAdapter())


def test_post_signal_host_boot_substitution_fails_loud(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    case = _case(tmp_path, monkeypatch, deadline)
    retained_boot_id = (
        case.query.timeout_directive_publication.timeout_directive.host_boot_id
    )
    monkeypatch.setattr(
        containment_module,
        "read_run_action_host_boot_id",
        lambda _descriptor: (
            "changed-host-boot" if case.runner.signal_commands else retained_boot_id
        ),
    )

    class _SubstitutedBootAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _contain(case, capability)

    with pytest.raises(
        RunActionTimeoutContainmentError,
        match="not safely resolved",
    ):
        case.capability._invoke_once(_SubstitutedBootAdapter())


def test_post_signal_control_topology_substitution_fails_loud(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    case = _case(tmp_path, monkeypatch, deadline)
    original_require_current = case.control_inspection.require_current

    def require_retained_timeout():
        if case.runner.signal_commands:
            raise AssertionError("injected post-signal control topology change")
        original_require_current()

    case.control_inspection.require_current = require_retained_timeout

    class _SubstitutedTopologyAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return _contain(case, capability)

    with pytest.raises(
        AssertionError,
        match="post-signal control topology change",
    ):
        case.capability._invoke_once(_SubstitutedTopologyAdapter())


def test_same_settings_foreign_runtime_cannot_split_observation_and_signal(
    tmp_path,
    monkeypatch,
):
    query = _inspection_context(_configured_settings()[0], timed_out=True)[0]
    deadline = (
        query.timeout_directive_publication.timeout_directive.containment_deadline_boottime_nanoseconds
    )
    case = _case(tmp_path / "primary", monkeypatch, deadline)
    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir()
    foreign_root.chmod(0o700)
    foreign_runner = _ContainmentDockerRunner(case.docker_settings)
    foreign_runtime = PinnedDockerRuntime(
        trusted_root=foreign_root.resolve(),
        settings=case.docker_settings,
        process_runner=foreign_runner,
    )
    foreign_manager = DockerRunActionContainmentManager(foreign_runtime)

    class _ForeignAdapter:
        @staticmethod
        def continue_committed_once(capability):
            return contain_run_action_timeout_once(
                capability=capability,
                resource_manager=case.resource_manager,
                containment_manager=foreign_manager,
                command=case.command,
                helper_evidence=case.helper,
                init_source_evidence=case.init,
                docker_settings=case.docker_settings,
                launch_settings=case.launch_settings,
            )

    with pytest.raises(
        RunActionTimeoutContainmentError,
        match="one exact configured runtime",
    ):
        case.capability._invoke_once(_ForeignAdapter())
    assert foreign_runner.signal_commands == []


def test_terminal_timeout_leaf_registers_exact_provider_termination(monkeypatch):
    docker_settings, launch_settings = _configured_settings()
    query, inventory, raw, command, helper, init = _inspection_context(
        docker_settings,
        timed_out=True,
    )
    adoption = query.workload_release_adoption
    control_inspection = _ControlInspection(
        RunActionControlDirectoryTopology.TIMED_OUT,
        adoption,
        query.timeout_directive_publication,
    )
    manager, remaining_payloads = _patch_physical_inspection(
        monkeypatch,
        docker_settings=docker_settings,
        inventory=inventory,
        volume_raw=_volume_raw(
            query.prepared_execution.runtime_volume_authority,
            docker_settings,
        ),
        main_inspections=(copy.deepcopy(raw), copy.deepcopy(raw)),
        control_inspection=control_inspection,
        host_boot_id=adoption.workload_release_receipt.host_boot_id,
    )
    _normalized, normalized_payload, _raw_size_bytes = (
        docker_inspect._snapshot_container_inspection(
            raw,
            "test timeout terminal",
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

    class _TimeoutTerminationAdapter:
        def __init__(self):
            self.receipt = None

        def continue_committed_once(self, active_capability):
            self.receipt = capture_run_action_timeout_termination(
                capability=active_capability,
                resource_manager=manager,
                command=command,
                helper_evidence=helper,
                init_source_evidence=init,
                docker_settings=docker_settings,
                launch_settings=launch_settings,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PROVIDER_TERMINATED,
                result=None,
                provider_termination_receipt=self.receipt,
                timeout_directive_publication=None,
            )

    adapter = _TimeoutTerminationAdapter()
    outcome = capability._invoke_once(adapter)

    assert outcome.provider_termination_receipt == adapter.receipt
    assert adapter.receipt.timeout_directive_publication == (
        query.timeout_directive_publication
    )
    assert not remaining_payloads
