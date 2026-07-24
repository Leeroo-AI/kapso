from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

import kapso.cross_run.docker.runtime as runtime_module
import kapso.cross_run.expert.replay_docker_provider as provider_module
from kapso.cross_run.docker.runtime import PinnedDockerRuntime
from kapso.cross_run.expert.replay_docker_provider import (
    SourceReplayDockerExecutionProvider,
)
from kapso.cross_run.expert.task_evaluation_docker_provider import (
    TaskEvaluationDockerSandboxError,
    task_adapter_docker_image_authority,
)
from kapso.cross_run.expert.replay_execution import (
    expert_source_replay_execution_provider_key,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessResult,
)
from test_expert_task_evaluation_docker_resources import _StatefulDockerRunner
from test_cross_run_docker_runtime import _image, _json_line
from test_expert_task_evaluation_provider_filesystem import (
    _RESULT_PAYLOAD,
    _matched_invocation,
    _valid_result_snapshot,
)
from test_expert_source_replay_request import _prepared, _request_fixture


class _ProviderDockerRunner(_StatefulDockerRunner):
    def __init__(self, runtime_settings, provider_settings, adapter_runtime, compute):
        super().__init__(runtime_settings)
        self.provider_settings = provider_settings
        self.adapter_runtime = adapter_runtime
        self.compute = compute
        self.attach_outcome = BoundedProcessOutcome.COMPLETED
        self.attach_returncode = 0
        self.attach_oom_killed = False
        self.mutate_evaluator = None
        self.mutate_evaluator_after_attach = None
        self.mutate_image = None

    def run(self, request):
        self.requests.append(request)
        arguments = request.argv[5:]
        outcome = BoundedProcessOutcome.COMPLETED
        returncode = 0
        stderr = b""
        if arguments[:3] == ("container", "start", "--attach"):
            container = self._container_by_id(arguments[3])
            container["HostConfig"]["OomKillDisable"] = None
            container["NetworkSettings"]["Networks"]["none"]["NetworkID"] = "c" * 64
            outcome = self.attach_outcome
            returncode = self.attach_returncode
            if outcome is BoundedProcessOutcome.COMPLETED:
                container["State"].update(
                    {
                        "ExitCode": returncode,
                        "OOMKilled": self.attach_oom_killed,
                        "Pid": 0,
                        "Running": False,
                        "Status": "exited",
                    }
                )
            else:
                if self.attach_oom_killed:
                    container["State"].update(
                        {
                            "ExitCode": 137,
                            "OOMKilled": True,
                            "Pid": 0,
                            "Running": False,
                            "Status": "exited",
                        }
                    )
                else:
                    container["State"].update(
                        {
                            "Pid": 1234,
                            "Running": True,
                            "Status": "running",
                        }
                    )
            if container["State"]["Status"] == "running":
                container["NetworkSettings"]["SandboxID"] = "d" * 64
                container["NetworkSettings"][
                    "SandboxKey"
                ] = f"/var/run/docker/netns/{'d' * 12}"
                container["NetworkSettings"]["Networks"]["none"]["EndpointID"] = (
                    "e" * 64
                )
            if self.mutate_evaluator_after_attach is not None:
                self.mutate_evaluator_after_attach(container)
            stdout = b"evaluator output\n"
        elif arguments[:2] == ("container", "exec"):
            stdout = _valid_result_snapshot(self.provider_settings.container_user_id)
        else:
            stdout = self._dispatch(arguments)
        return BoundedProcessResult(
            request=request,
            outcome=outcome,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_bytes_observed=len(stdout),
            stderr_bytes_observed=len(stderr),
            duration_seconds=0.0,
        )

    def _dispatch(self, arguments):
        if arguments[:4] == (
            "image",
            "inspect",
            "--format",
            "{{json .}}",
        ):
            image = _image(
                task_adapter_docker_image_authority(self.adapter_runtime),
                tuple(sorted(self.adapter_runtime.environment.items())),
            )
            if self.mutate_image is not None:
                self.mutate_image(image)
            return _json_line(image)
        if arguments[:2] == ("container", "create"):
            return self._create_container(arguments)
        if arguments[:2] == ("container", "stop"):
            container = self._container_by_id(arguments[-1])
            container["State"].update(
                {
                    "ExitCode": 137,
                    "Pid": 0,
                    "Running": False,
                    "Status": "exited",
                }
            )
            return f"{arguments[-1]}\n".encode()
        if arguments[:2] == ("container", "start"):
            container = self._container_by_id(arguments[2])
            container["HostConfig"]["OomKillDisable"] = None
            container["NetworkSettings"]["SandboxID"] = "d" * 64
            container["NetworkSettings"][
                "SandboxKey"
            ] = f"/var/run/docker/netns/{'d' * 12}"
            container["NetworkSettings"]["Networks"]["none"]["NetworkID"] = "c" * 64
            container["NetworkSettings"]["Networks"]["none"]["EndpointID"] = "e" * 64
            container["State"].update(
                {
                    "Pid": 1234,
                    "Running": True,
                    "Status": "running",
                }
            )
            return f"{arguments[2]}\n".encode()
        return super()._dispatch(arguments)

    def _create_container(self, arguments):
        name = _flag_value(arguments, "--name")
        role = next(
            value.split("=", 1)[1]
            for value in _flag_values(arguments, "--label")
            if value.startswith("io.kapso.task-evaluation.role=")
        )
        labels = {
            value.split("=", 1)[0]: value.split("=", 1)[1]
            for value in _flag_values(arguments, "--label")
        }
        image_position = arguments.index(self.adapter_runtime.image_reference)
        command = list(arguments[image_position + 1 :])
        entrypoint = _flag_value(arguments, "--entrypoint")
        mounts = [_inspect_mount(value) for value in _flag_values(arguments, "--mount")]
        container_id = ("a" if role == "evaluator" else "b") * 64
        cpu_quota = (
            self.compute.cpu_millicore_limit
            * self.provider_settings.cpu_period_microseconds
            // 1000
        )
        environment = {}
        for assignment in _flag_values(arguments, "--env"):
            key, value = assignment.split("=", 1)
            environment[key] = value
        payload = {
            "Args": command,
            "Config": {
                "Cmd": command or None,
                "Entrypoint": [entrypoint],
                "Env": [f"{key}={value}" for key, value in environment.items()],
                "Hostname": "kapso-task-evaluation",
                "Image": self.adapter_runtime.image_reference,
                "Labels": labels,
                "StopTimeout": self.compute.termination_grace_seconds,
                "User": (
                    f"{self.provider_settings.container_user_id}:"
                    f"{self.provider_settings.container_group_id}"
                ),
                "WorkingDir": _flag_value(arguments, "--workdir"),
            },
            "HostConfig": {
                "AutoRemove": False,
                "Binds": None,
                "CapAdd": None,
                "CapDrop": ["ALL"],
                "Cgroup": "",
                "CgroupnsMode": "private",
                "CpuPeriod": self.provider_settings.cpu_period_microseconds,
                "CpuQuota": cpu_quota,
                "DeviceRequests": None,
                "DeviceCgroupRules": None,
                "Devices": [],
                "Dns": None,
                "DnsOptions": [],
                "DnsSearch": [],
                "ExtraHosts": None,
                "GroupAdd": None,
                "IpcMode": "private",
                "Links": None,
                "LogConfig": {"Config": {}, "Type": "none"},
                "Memory": self.compute.memory_byte_limit,
                "MemorySwap": self.compute.memory_byte_limit,
                "NetworkMode": "none",
                "OomKillDisable": False,
                "PidMode": "",
                "PidsLimit": self.compute.process_limit,
                "PortBindings": {},
                "Privileged": False,
                "PublishAllPorts": False,
                "ReadonlyRootfs": True,
                "RestartPolicy": {"MaximumRetryCount": 0, "Name": "no"},
                "Runtime": self.settings.runtime_default_runtime,
                "SecurityOpt": ["no-new-privileges", "seccomp=builtin"],
                "ShmSize": self.compute.shared_memory_byte_limit,
                "UTSMode": "",
                "Ulimits": [
                    {
                        "Hard": self.compute.open_file_limit,
                        "Name": "nofile",
                        "Soft": self.compute.open_file_limit,
                    }
                ],
                "UsernsMode": "",
                "VolumeDriver": "",
                "VolumesFrom": None,
            },
            "Id": container_id,
            "Image": self.adapter_runtime.image_config_digest,
            "Mounts": mounts,
            "Name": f"/{name}",
            "NetworkSettings": {
                "SandboxID": "",
                "SandboxKey": "",
                "Ports": {},
                "Networks": {
                    "none": {
                        "IPAMConfig": None,
                        "Links": None,
                        "Aliases": None,
                        "DriverOpts": None,
                        "GwPriority": 0,
                        "NetworkID": "",
                        "EndpointID": "",
                        "Gateway": "",
                        "IPAddress": "",
                        "MacAddress": "",
                        "IPPrefixLen": 0,
                        "IPv6Gateway": "",
                        "GlobalIPv6Address": "",
                        "GlobalIPv6PrefixLen": 0,
                        "DNSNames": None,
                    }
                },
            },
            "Path": entrypoint,
            "RestartCount": 0,
            "State": {
                "Dead": False,
                "Error": "",
                "ExitCode": 0,
                "OOMKilled": False,
                "Paused": False,
                "Pid": 0,
                "Restarting": False,
                "Running": False,
                "Status": "created",
            },
        }
        if role == "evaluator" and self.mutate_evaluator is not None:
            self.mutate_evaluator(payload)
        self.containers[name] = payload
        return f"{container_id}\n".encode()

    def _container_by_id(self, container_id):
        return next(
            payload
            for payload in self.containers.values()
            if payload["Id"] == container_id
        )


def _flag_values(arguments, flag):
    return tuple(
        arguments[position + 1]
        for position, value in enumerate(arguments[:-1])
        if value == flag
    )


def _flag_value(arguments, flag):
    values = _flag_values(arguments, flag)
    assert len(values) == 1
    return values[0]


def _inspect_mount(specification):
    fields = specification.split(",")
    values = {
        field.split("=", 1)[0]: field.split("=", 1)[1]
        for field in fields
        if "=" in field
    }
    if values["type"] == "bind":
        return {
            "Destination": values["dst"],
            "Propagation": values["bind-propagation"],
            "RW": False,
            "Source": values["src"],
            "Type": "bind",
        }
    return {
        "Destination": values["dst"],
        "Driver": "local",
        "Name": values["src"],
        "Propagation": "",
        "RW": "readonly" not in fields,
        "Type": "volume",
    }


@pytest.fixture(scope="module")
def prepared_replay_request(tmp_path_factory):
    return _prepared(
        _request_fixture(tmp_path_factory.mktemp("expert-replay-docker-provider"))
    )


@pytest.fixture
def provider(tmp_path, monkeypatch, prepared_replay_request):
    invocation = _matched_invocation(prepared_replay_request, "candidate_leg")
    case = invocation.materialized_case
    compute = case.request_case.compute_binding
    settings = prepared_replay_request.settings.task_evaluation_provider
    runtime_settings = settings.runtime
    adapter_runtime = case.task_adapter.manifest.runtime
    tmp_path.chmod(0o700)
    docker_path = tmp_path / "docker"
    docker_path.write_bytes(b"docker")
    docker_path.chmod(0o500)
    docker_config_root = tmp_path / "config"
    docker_config_root.mkdir(mode=0o700)
    runner = _ProviderDockerRunner(
        runtime_settings,
        settings,
        adapter_runtime,
        compute,
    )
    runtime = object.__new__(PinnedDockerRuntime)
    runtime._trusted_root = tmp_path.resolve()
    runtime._settings = runtime_settings
    runtime._process_runner = runner
    runtime._docker_path = docker_path
    runtime._docker_digest = runtime_settings.runtime_executable_digest
    runtime._docker_config_root = docker_config_root
    runtime._environment = MappingProxyType(
        {
            "DOCKER_API_VERSION": runtime_settings.runtime_api_version,
            "DOCKER_CONFIG": str(docker_config_root),
            "HOME": str(tmp_path),
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    monkeypatch.setattr(
        runtime_module,
        "read_verified_private_executable",
        lambda _path: runtime_settings.runtime_executable_digest,
    )
    monkeypatch.setattr(runtime_module, "_require_runtime_socket", lambda _path: None)

    def materialize_helper(self, identity):
        helper_root = identity.workspace_root / "provider"
        helper_root.mkdir(mode=0o700)
        helper_path = helper_root / "busybox"
        helper_path.write_bytes(b"busybox")
        helper_path.chmod(0o555)
        helper_root.chmod(0o555)
        return helper_root

    monkeypatch.setattr(
        SourceReplayDockerExecutionProvider,
        "_materialize_helper",
        materialize_helper,
    )
    execution_provider = SourceReplayDockerExecutionProvider(
        dispatch_key=expert_source_replay_execution_provider_key(case),
        provider_settings=settings,
        policy_settings=prepared_replay_request.settings.policy,
        runtime=runtime,
    )
    return execution_provider, runner, invocation


def test_provider_runs_exact_isolated_lifecycle_and_cleans_everything(provider):
    execution_provider, runner, invocation = provider

    completion = execution_provider.execute_leg(invocation)

    assert (
        completion.provider_handle_id == invocation.provider_handle.provider_handle_id
    )
    assert completion.process_result.outcome is BoundedProcessOutcome.COMPLETED
    assert completion.result_payload == _RESULT_PAYLOAD
    assert runner.containers == {}
    assert runner.volumes == {}
    assert tuple(execution_provider._runtime.trusted_root.glob("replay-*")) == ()
    commands = tuple(request.argv[5:] for request in runner.requests)
    attach_position = next(
        position
        for position, command in enumerate(commands)
        if command[:3] == ("container", "start", "--attach")
    )
    evaluator_remove_position = next(
        position
        for position, command in enumerate(commands)
        if command[:2] == ("container", "rm") and command[-1] == "a" * 64
    )
    snapshot_position = next(
        position
        for position, command in enumerate(commands)
        if command[:2] == ("container", "exec")
    )
    assert attach_position < evaluator_remove_position < snapshot_position
    evaluator_create = next(
        command
        for command in commands
        if command[:2] == ("container", "create")
        and "io.kapso.task-evaluation.role=evaluator" in command
    )
    assert "--pull" in evaluator_create
    assert evaluator_create[evaluator_create.index("--pull") + 1] == "never"
    assert evaluator_create[evaluator_create.index("--network") + 1] == "none"
    assert "--read-only" in evaluator_create
    assert "--oom-kill-disable=false" in evaluator_create
    assert evaluator_create[evaluator_create.index("--cap-drop") + 1] == "ALL"
    assert _flag_values(evaluator_create, "--env") == (
        "HOME=/kapso/home",
        "HOSTNAME=kapso-task-evaluation",
        "LANG=C.UTF-8",
        "PATH=/usr/bin:/bin",
    )


def test_provider_returns_no_payload_for_a_bounded_attach_failure(provider):
    execution_provider, runner, invocation = provider
    runner.attach_outcome = BoundedProcessOutcome.TIMED_OUT
    runner.attach_returncode = -9

    completion = execution_provider.execute_leg(invocation)

    assert completion.process_result.outcome is BoundedProcessOutcome.TIMED_OUT
    assert completion.result_payload is None
    assert runner.containers == {}
    assert runner.volumes == {}
    assert not any(
        request.argv[5:7] == ("container", "exec") for request in runner.requests
    )
    stop_command = next(
        request.argv[5:]
        for request in runner.requests
        if request.argv[5:7] == ("container", "stop")
    )
    assert stop_command[2:5] == (
        "--time",
        str(
            invocation.materialized_case.request_case.compute_binding.termination_grace_seconds
        ),
        "a" * 64,
    )


@pytest.mark.parametrize(
    "outcome",
    (
        BoundedProcessOutcome.COMPLETED,
        BoundedProcessOutcome.TIMED_OUT,
    ),
)
def test_provider_records_oom_as_an_unsuccessful_bounded_completion(
    provider,
    outcome,
):
    execution_provider, runner, invocation = provider
    runner.attach_outcome = outcome
    runner.attach_returncode = 137 if outcome is BoundedProcessOutcome.COMPLETED else -9
    runner.attach_oom_killed = True

    completion = execution_provider.execute_leg(invocation)

    assert completion.process_result.outcome is outcome
    assert completion.result_payload is None
    assert runner.containers == {}
    assert runner.volumes == {}


def test_provider_rejects_a_weakened_created_container_before_start(provider):
    execution_provider, runner, invocation = provider
    runner.mutate_evaluator = lambda payload: payload["HostConfig"].__setitem__(
        "ReadonlyRootfs", False
    )

    with pytest.raises(TaskEvaluationDockerSandboxError, match="sandbox authority"):
        execution_provider.execute_leg(invocation)

    assert runner.containers == {}
    assert runner.volumes == {}
    assert not any(
        request.argv[5:8] == ("container", "start", "--attach")
        for request in runner.requests
    )


def test_provider_accepts_environment_order_chosen_by_daemon(provider):
    execution_provider, runner, invocation = provider
    runner.mutate_evaluator = lambda payload: payload["Config"]["Env"].reverse()

    completion = execution_provider.execute_leg(invocation)

    assert completion.result_payload is not None


def test_provider_rejects_duplicate_environment_authority_before_start(provider):
    execution_provider, runner, invocation = provider
    runner.mutate_evaluator = lambda payload: payload["Config"]["Env"].append(
        payload["Config"]["Env"][0]
    )

    with pytest.raises(TaskEvaluationDockerSandboxError, match="sandbox authority"):
        execution_provider.execute_leg(invocation)

    assert not any(
        request.argv[5:8] == ("container", "start", "--attach")
        for request in runner.requests
    )


def test_provider_rejects_compute_authority_mutated_during_execution(provider):
    execution_provider, runner, invocation = provider
    runner.mutate_evaluator_after_attach = lambda payload: payload[
        "HostConfig"
    ].__setitem__("Memory", runner.compute.memory_byte_limit + 1)

    with pytest.raises(
        TaskEvaluationDockerSandboxError,
        match="owned resources changed",
    ):
        execution_provider.execute_leg(invocation)

    assert runner.containers == {}
    assert runner.volumes == {}


def test_provider_rejects_network_attached_during_execution(provider):
    execution_provider, runner, invocation = provider
    runner.mutate_evaluator_after_attach = lambda payload: payload["NetworkSettings"][
        "Networks"
    ].__setitem__(
        "bridge",
        dict(payload["NetworkSettings"]["Networks"]["none"]),
    )

    with pytest.raises(
        TaskEvaluationDockerSandboxError,
        match="owned resources changed",
    ):
        execution_provider.execute_leg(invocation)

    assert runner.containers == {}
    assert runner.volumes == {}


def test_interrupted_cleanup_never_starts_or_executes(provider):
    execution_provider, runner, invocation = provider
    identity = execution_provider._resources.identity(
        invocation.provider_handle.provider_handle_id
    )
    identity.workspace_root.mkdir(mode=0o700)
    runner.containers[identity.evaluator_name] = {
        "Config": {"Labels": dict(identity.labels_for("evaluator"))},
        "Id": "a" * 64,
        "Name": f"/{identity.evaluator_name}",
    }
    runner.volumes[identity.volume_name] = {
        "Driver": "local",
        "Labels": dict(identity.labels_for("volume")),
        "Name": identity.volume_name,
        "Options": {},
        "Scope": "local",
    }

    execution_provider.cleanup_interrupted(invocation.provider_handle)
    execution_provider.cleanup_interrupted(invocation.provider_handle)

    assert runner.containers == {}
    assert runner.volumes == {}
    assert not identity.workspace_root.exists()
    assert not any(
        request.argv[5:7] in {("container", "start"), ("container", "exec")}
        for request in runner.requests
    )
