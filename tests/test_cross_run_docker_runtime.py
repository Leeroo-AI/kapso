from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import kapso.cross_run.docker.runtime as runtime_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    DockerImageAuthority,
    PinnedDockerRuntime,
    PinnedDockerRuntimeError,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessResult,
)
from kapso.cross_run.settings import CrossRunSettings

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_TEST_EXECUTABLE_BYTES = b"private pinned Docker test executable"


class _ScriptedProcessRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        output = self.outputs.pop(0)
        return BoundedProcessResult(
            request=request,
            outcome=output.get("outcome", BoundedProcessOutcome.COMPLETED),
            returncode=output.get("returncode", 0),
            stdout=output.get("stdout", b""),
            stderr=output.get("stderr", b""),
            stdout_bytes_observed=len(output.get("stdout", b"")),
            stderr_bytes_observed=len(output.get("stderr", b"")),
            duration_seconds=0.0,
        )


@pytest.fixture(scope="module")
def provider_settings():
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker
    return replace(
        settings,
        runtime_executable_digest=tree_or_blob_digest(_TEST_EXECUTABLE_BYTES),
    )


@pytest.fixture(autouse=True)
def isolated_local_authority(monkeypatch, provider_settings):
    def read_executable(_path, expected_digest):
        if expected_digest != provider_settings.runtime_executable_digest:
            raise PinnedDockerRuntimeError(
                "Docker authority executable differs from its pinned digest"
            )
        return _TEST_EXECUTABLE_BYTES

    monkeypatch.setattr(
        runtime_module, "read_verified_root_executable", read_executable
    )
    monkeypatch.setattr(runtime_module, "_require_runtime_socket", lambda _path: None)


def _json_line(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def _version(settings):
    return {
        "Client": {
            "ApiVersion": settings.runtime_api_version,
            "Version": settings.runtime_server_version,
        },
        "Server": {
            "ApiVersion": settings.runtime_api_version,
            "Os": settings.runtime_host_operating_system,
            "Version": settings.runtime_server_version,
        },
    }


def _info(settings):
    return {
        "Architecture": settings.runtime_host_architecture,
        "CgroupDriver": settings.runtime_cgroup_driver,
        "CgroupVersion": settings.runtime_cgroup_version,
        "CpuCfsPeriod": True,
        "CpuCfsQuota": True,
        "DefaultRuntime": settings.runtime_default_runtime,
        "Driver": settings.runtime_storage_driver,
        "DockerRootDir": settings.runtime_root_directory,
        "MemoryLimit": True,
        "OSType": settings.runtime_host_operating_system,
        "PidsLimit": True,
        "Plugins": {"Network": ["null"], "Volume": ["local"]},
        "Runtimes": {settings.runtime_default_runtime: {}},
        "SecurityOptions": list(settings.required_security_options),
        "ServerVersion": settings.runtime_server_version,
        "SwapLimit": True,
    }


def _runtime_contract():
    return DockerImageAuthority.mint(
        image_reference=(
            "registry.example/kapso/replay-runtime@" + tree_or_blob_digest(b"manifest")
        ),
        image_config_digest=tree_or_blob_digest(b"config"),
        operating_system="linux",
        architecture="amd64",
        architecture_variant=None,
    )


def _image(runtime, environment=()):
    return {
        "Architecture": runtime.architecture,
        "Config": {
            "Cmd": None,
            "Entrypoint": None,
            "Env": [f"{key}={value}" for key, value in environment],
            "Healthcheck": None,
            "Volumes": None,
        },
        "Id": runtime.image_config_digest,
        "Os": runtime.operating_system,
        "RepoDigests": [runtime.image_reference],
        "Variant": "",
    }


def _fresh_image_outputs(settings, image):
    return (
        {"stdout": _json_line(_version(settings))},
        {"stdout": _json_line(_info(settings))},
        {"stdout": _json_line(image)},
    )


def _make_runtime(tmp_path, settings, additional_outputs=()):
    tmp_path.chmod(0o700)
    runner = _ScriptedProcessRunner(
        [
            {"stdout": _json_line(_version(settings))},
            {"stdout": _json_line(_info(settings))},
            *additional_outputs,
        ]
    )
    runtime = PinnedDockerRuntime(
        trusted_root=tmp_path.resolve(),
        settings=settings,
        process_runner=runner,
    )
    return runtime, runner


def _construct_runtime_after_signal(trusted_root, settings, start_signal):
    start_signal.wait()
    PinnedDockerRuntime(
        trusted_root=trusted_root,
        settings=settings,
        process_runner=_ScriptedProcessRunner([]),
    )


def _hold_docker_mutation(
    trusted_root,
    settings,
    entered,
    release,
):
    class _MutationHoldingRunner:
        def __init__(self):
            self.call_count = 0

        def run(self, request):
            self.call_count += 1
            if self.call_count == 1:
                stdout = _json_line(_version(settings))
            elif self.call_count == 2:
                stdout = _json_line(_info(settings))
            else:
                entered.set()
                release.wait(settings.command_timeout_seconds)
                stdout = b""
            return BoundedProcessResult(
                request=request,
                outcome=BoundedProcessOutcome.COMPLETED,
                returncode=0,
                stdout=stdout,
                stderr=b"",
                stdout_bytes_observed=len(stdout),
                stderr_bytes_observed=0,
                duration_seconds=0.0,
            )

    runtime = PinnedDockerRuntime(
        trusted_root=trusted_root,
        settings=settings,
        process_runner=_MutationHoldingRunner(),
    )
    runtime.run_control(("container", "start", "a" * 64))


def _announce_runtime_ready(
    trusted_root,
    settings,
    ready,
):
    runtime = PinnedDockerRuntime(
        trusted_root=trusted_root,
        settings=settings,
        process_runner=_ScriptedProcessRunner(
            [
                {"stdout": _json_line(_version(settings))},
                {"stdout": _json_line(_info(settings))},
                {"stdout": b""},
            ]
        ),
    )
    runtime.run_control(("container", "start", "b" * 64))
    ready.set()


class _AttachedStartHoldingRunner:
    def __init__(self, settings, entered, release):
        self.settings = settings
        self.entered = entered
        self.release = release
        self.requests = []
        self.started = False

    def run(self, request):
        self.requests.append(request)
        arguments = request.argv[5:]
        if arguments == ("version", "--format", "{{json .}}"):
            stdout = _json_line(_version(self.settings))
        elif arguments == ("info", "--format", "{{json .}}"):
            stdout = _json_line(_info(self.settings))
        elif arguments[:3] == ("container", "start", "--attach"):
            self.started = True
            self.entered.set()
            self.release.wait(self.settings.command_timeout_seconds)
            stdout = b""
        elif arguments[:4] == (
            "container",
            "inspect",
            "--format",
            "{{json .}}",
        ):
            stdout = _json_line(
                {
                    "Id": arguments[4],
                    "State": {
                        "Status": "running" if self.started else "created",
                    },
                }
            )
        else:
            raise AssertionError(f"unexpected Docker arguments: {arguments}")
        return BoundedProcessResult(
            request=request,
            outcome=BoundedProcessOutcome.COMPLETED,
            returncode=0,
            stdout=stdout,
            stderr=b"",
            stdout_bytes_observed=len(stdout),
            stderr_bytes_observed=0,
            duration_seconds=0.0,
        )


def test_runtime_executes_privately_pinned_cli_with_no_inherited_environment(
    tmp_path,
    provider_settings,
):
    runtime, runner = _make_runtime(tmp_path, provider_settings)
    settings_digest_suffix = tree_or_blob_digest(
        provider_settings.to_json_bytes()
    ).removeprefix("sha256:")
    executable_digest_suffix = provider_settings.runtime_executable_digest.removeprefix(
        "sha256:"
    )
    docker_path = tmp_path / "authority" / f"docker-{executable_digest_suffix}"
    docker_config_path = tmp_path / f"docker-config-{settings_digest_suffix}"

    assert runtime.settings is provider_settings
    assert len(runner.requests) == 2
    for request in runner.requests:
        assert request.argv[:5] == (
            str(docker_path),
            "--host",
            f"unix://{provider_settings.runtime_socket_path}",
            "--config",
            str(docker_config_path),
        )
        assert dict(request.environment) == {
            "DOCKER_API_VERSION": provider_settings.runtime_api_version,
            "DOCKER_CONFIG": str(docker_config_path),
            "HOME": str(tmp_path),
            "LANG": "C",
            "LC_ALL": "C",
        }
    assert runner.requests[0].argv[-3:] == (
        "version",
        "--format",
        "{{json .}}",
    )
    assert runner.requests[1].argv[-3:] == (
        "info",
        "--format",
        "{{json .}}",
    )
    assert docker_path.stat().st_mode & 0o777 == 0o500
    assert (docker_config_path / "config.json").read_bytes() == b'{"auths":{}}\n'


def test_containment_authority_has_only_closed_term_kill_surface(
    tmp_path,
    monkeypatch,
    provider_settings,
):
    runtime, runner = _make_runtime(
        tmp_path,
        provider_settings,
        additional_outputs=(
            {"stdout": _json_line(_version(provider_settings))},
            {"stdout": _json_line(_info(provider_settings))},
        ),
    )
    authority = runtime.issue_containment_authority()
    request_count = len(runner.requests)

    assert authority.settings == provider_settings
    assert not hasattr(authority, "run_control")
    assert not hasattr(authority, "run_bounded")
    with pytest.raises(
        PinnedDockerRuntimeError,
        match="closed authority",
    ):
        authority._signal_container_once(
            container_id="a" * 64,
            signal_name="SIGHUP",
            _authority=runtime_module._DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
        )
    with pytest.raises(
        PinnedDockerRuntimeError,
        match="closed authority",
    ):
        authority._signal_container_once(
            container_id="short-id",
            signal_name="SIGKILL",
            _authority=runtime_module._DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
        )
    assert len(runner.requests) == request_count
    owner_process_id = authority._owner_process_id
    monkeypatch.setattr(
        runtime_module.os,
        "getpid",
        lambda: owner_process_id + 1,
    )
    with pytest.raises(
        PinnedDockerRuntimeError,
        match="unissued or foreign",
    ):
        authority.settings


def test_cleanup_authority_has_only_closed_exact_removal_surface(
    tmp_path,
    monkeypatch,
    provider_settings,
):
    main_id = "a" * 64
    keeper_id = "b" * 64
    volume_name = "kapso-cleanup-volume"
    runtime, runner = _make_runtime(
        tmp_path,
        provider_settings,
        additional_outputs=(
            {"stdout": _json_line(_version(provider_settings))},
            {"stdout": _json_line(_info(provider_settings))},
            {"stdout": f"{main_id}\n".encode("ascii")},
            {"stdout": f"{keeper_id}\n".encode("ascii")},
            {"stdout": f"{volume_name}\n".encode("ascii")},
        ),
    )
    authority = runtime.issue_cleanup_authority()
    request_count = len(runner.requests)

    assert authority.settings == provider_settings
    assert not hasattr(authority, "run_control")
    assert not hasattr(authority, "run_bounded")
    with pytest.raises(PinnedDockerRuntimeError, match="closed authority"):
        authority._remove_stopped_container_once(
            container_id="short-id",
            exclusion_lease=None,
            _authority=runtime_module._DOCKER_CLEANUP_REMOVE_AUTHORITY,
        )
    with pytest.raises(PinnedDockerRuntimeError, match="closed authority"):
        authority._remove_volume_once(
            volume_name="../foreign",
            exclusion_lease=None,
            _authority=runtime_module._DOCKER_CLEANUP_REMOVE_AUTHORITY,
        )
    with pytest.raises(PinnedDockerRuntimeError, match="closed authority"):
        authority._remove_stopped_container_once(
            container_id=main_id,
            exclusion_lease=None,
            _authority=runtime_module._DOCKER_CLEANUP_REMOVE_AUTHORITY,
        )
    assert len(runner.requests) == request_count

    with authority._issue_exclusion_lease(
        _authority=runtime_module._DOCKER_CLEANUP_EXCLUSION_ISSUANCE,
    ) as exclusion:
        authority._remove_stopped_container_once(
            container_id=main_id,
            exclusion_lease=exclusion,
            _authority=runtime_module._DOCKER_CLEANUP_REMOVE_AUTHORITY,
        )
        authority._remove_running_keeper_once(
            container_id=keeper_id,
            exclusion_lease=exclusion,
            _authority=runtime_module._DOCKER_CLEANUP_REMOVE_AUTHORITY,
        )
        authority._remove_volume_once(
            volume_name=volume_name,
            exclusion_lease=exclusion,
            _authority=runtime_module._DOCKER_CLEANUP_REMOVE_AUTHORITY,
        )
    assert tuple(request.argv[5:] for request in runner.requests[-3:]) == (
        ("container", "rm", main_id),
        ("container", "rm", "--force", keeper_id),
        ("volume", "rm", volume_name),
    )

    owner_process_id = authority._owner_process_id
    monkeypatch.setattr(
        runtime_module.os,
        "getpid",
        lambda: owner_process_id + 1,
    )
    with pytest.raises(
        PinnedDockerRuntimeError,
        match="unissued or foreign",
    ):
        authority.settings


def test_independent_processes_share_one_daemon_mutation_lock(
    tmp_path,
    provider_settings,
):
    context = multiprocessing.get_context("fork")
    holder_root = (tmp_path / "holder").resolve()
    contender_root = (tmp_path / "contender").resolve()
    holder_root.mkdir(mode=0o700)
    contender_root.mkdir(mode=0o700)
    entered = context.Event()
    release = context.Event()
    ready = context.Event()
    holder = context.Process(
        target=_hold_docker_mutation,
        args=(holder_root, provider_settings, entered, release),
    )
    contender = context.Process(
        target=_announce_runtime_ready,
        args=(contender_root, provider_settings, ready),
    )

    holder.start()
    assert entered.wait(provider_settings.command_timeout_seconds)
    contender.start()
    contender_was_blocked = not ready.wait(
        provider_settings.run_action_barrier_poll_interval_seconds
    )
    release.set()
    holder.join(provider_settings.command_timeout_seconds)
    contender.join(provider_settings.command_timeout_seconds)

    assert contender_was_blocked
    assert ready.is_set()
    assert holder.exitcode == 0
    assert contender.exitcode == 0


def test_same_thread_recursive_daemon_mutation_fails_loud(
    tmp_path,
    provider_settings,
):
    runtime, runner = _make_runtime(
        tmp_path,
        provider_settings,
        ({"stdout": b""},),
    )

    with runtime_module._open_docker_mutation_lease(
        runtime,
        timeout_seconds=provider_settings.command_timeout_seconds,
    ):
        with pytest.raises(
            PinnedDockerRuntimeError,
            match="cannot be acquired recursively",
        ):
            runtime.run_control(("container", "start", "c" * 64))

    assert len(runner.requests) == 2


def test_corrupt_mutation_lease_closes_before_failing_loud(
    tmp_path,
    provider_settings,
    monkeypatch,
):
    runtime, _runner = _make_runtime(tmp_path, provider_settings)
    lease = runtime_module._open_docker_mutation_lease(
        runtime,
        timeout_seconds=provider_settings.command_timeout_seconds,
    )
    original_fstat = runtime_module.os.fstat
    corrupt = True

    def observed_fstat(descriptor):
        metadata = original_fstat(descriptor)
        if descriptor != lease._descriptor or not corrupt:
            return metadata
        fields = list(metadata)
        fields[6] = 1
        return runtime_module.os.stat_result(fields)

    monkeypatch.setattr(runtime_module.os, "fstat", observed_fstat)

    with pytest.raises(
        PinnedDockerRuntimeError,
        match="changed while retained",
    ):
        lease.close()

    assert lease._closed
    assert lease._owner_key not in runtime_module._DOCKER_MUTATION_LEASE_OWNERS
    corrupt = False
    with runtime_module._open_docker_mutation_lease(
        runtime,
        timeout_seconds=provider_settings.command_timeout_seconds,
    ):
        pass


def test_containment_never_waits_behind_an_unrelated_daemon_mutation(
    tmp_path,
    provider_settings,
):
    context = multiprocessing.get_context("fork")
    holder_root = (tmp_path / "holder").resolve()
    contender_root = (tmp_path / "contender").resolve()
    holder_root.mkdir(mode=0o700)
    contender_root.mkdir(mode=0o700)
    entered = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_docker_mutation,
        args=(holder_root, provider_settings, entered, release),
    )
    holder.start()
    assert entered.wait(provider_settings.command_timeout_seconds)
    runtime, runner = _make_runtime(
        contender_root,
        provider_settings,
        (
            {"stdout": _json_line(_version(provider_settings))},
            {"stdout": _json_line(_info(provider_settings))},
            {"stdout": b""},
        ),
    )
    containment = runtime.issue_containment_authority()

    signal_result = containment._signal_container_once(
        container_id="d" * 64,
        signal_name="SIGKILL",
        _authority=runtime_module._DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
    )

    release.set()
    holder.join(provider_settings.command_timeout_seconds)
    assert holder.exitcode == 0
    assert signal_result.outcome is BoundedProcessOutcome.COMPLETED
    assert runner.requests[-1].argv[5:] == (
        "container",
        "kill",
        "--signal",
        "SIGKILL",
        "d" * 64,
    )


def test_attached_execution_releases_daemon_lock_after_start_transition(
    tmp_path,
    provider_settings,
):
    context = multiprocessing.get_context("fork")
    attached_root = (tmp_path / "attached").resolve()
    contender_root = (tmp_path / "contender").resolve()
    attached_root.mkdir(mode=0o700)
    contender_root.mkdir(mode=0o700)
    entered = context.Event()
    release = context.Event()
    attached_runner = _AttachedStartHoldingRunner(
        provider_settings,
        entered,
        release,
    )
    attached_runtime = PinnedDockerRuntime(
        trusted_root=attached_root,
        settings=provider_settings,
        process_runner=attached_runner,
    )
    contender_runtime, contender_runner = _make_runtime(
        contender_root,
        provider_settings,
        ({"stdout": b""},),
    )
    container_id = "e" * 64

    with ThreadPoolExecutor(max_workers=1) as execution:
        attached = execution.submit(
            attached_runtime.run_bounded,
            ("container", "start", "--attach", container_id),
            timeout_seconds=provider_settings.command_timeout_seconds,
            cleanup_timeout_seconds=provider_settings.cleanup_timeout_seconds,
            stdout_byte_limit=provider_settings.command_output_byte_limit,
            stderr_byte_limit=provider_settings.command_output_byte_limit,
        )
        assert entered.wait(provider_settings.command_timeout_seconds)
        contender_runtime.run_control(("container", "start", "f" * 64))
        assert not attached.done()
        release.set()
        assert attached.result().outcome is BoundedProcessOutcome.COMPLETED

    assert contender_runner.requests[-1].argv[5:] == (
        "container",
        "start",
        "f" * 64,
    )


def test_attached_execution_requires_created_occurrence_before_submit(
    tmp_path,
    provider_settings,
):
    container_id = "0" * 64
    runtime, runner = _make_runtime(
        tmp_path,
        provider_settings,
        (
            {
                "stdout": _json_line(
                    {
                        "Id": container_id,
                        "State": {"Status": "exited"},
                    }
                )
            },
        ),
    )

    with pytest.raises(
        PinnedDockerRuntimeError,
        match="lacks an exact created occurrence",
    ):
        runtime.run_bounded(
            ("container", "start", "--attach", container_id),
            timeout_seconds=provider_settings.command_timeout_seconds,
            cleanup_timeout_seconds=provider_settings.cleanup_timeout_seconds,
            stdout_byte_limit=provider_settings.command_output_byte_limit,
            stderr_byte_limit=provider_settings.command_output_byte_limit,
        )

    assert not any(
        request.argv[5:8] == ("container", "start", "--attach")
        for request in runner.requests
    )


def test_independent_process_runtimes_serialize_authority_publication(
    tmp_path,
    monkeypatch,
    provider_settings,
):
    tmp_path.chmod(0o700)
    process_context = multiprocessing.get_context("fork")
    concurrent_calls = process_context.Array("i", (0, 0), lock=True)
    start_signal = process_context.Event()
    original_ensure_private_directory = runtime_module._ensure_private_directory

    def observe_private_directory(path, parent):
        with concurrent_calls.get_lock():
            concurrent_calls[0] += 1
            concurrent_calls[1] = max(concurrent_calls[1], concurrent_calls[0])
        time.sleep(0.05)
        original_ensure_private_directory(path, parent)
        with concurrent_calls.get_lock():
            concurrent_calls[0] -= 1

    monkeypatch.setattr(
        runtime_module,
        "_ensure_private_directory",
        observe_private_directory,
    )
    monkeypatch.setattr(
        runtime_module.PinnedDockerRuntime,
        "require_live_authority",
        lambda _runtime: None,
    )
    processes = tuple(
        process_context.Process(
            target=_construct_runtime_after_signal,
            args=(tmp_path.resolve(), provider_settings, start_signal),
        )
        for _process_number in range(2)
    )
    for process in processes:
        process.start()
    start_signal.set()
    for process in processes:
        process.join(10)

    assert tuple(process.exitcode for process in processes) == (0, 0)
    assert tuple(concurrent_calls) == (0, 1)


def test_runtime_client_version_is_bound_by_bytes_not_server_version(
    tmp_path,
    provider_settings,
):
    version = _version(provider_settings)
    version["Client"]["Version"] = "different-content-pinned-client"
    tmp_path.chmod(0o700)
    runner = _ScriptedProcessRunner(
        [
            {"stdout": _json_line(version)},
            {"stdout": _json_line(_info(provider_settings))},
        ]
    )

    PinnedDockerRuntime(
        trusted_root=tmp_path.resolve(),
        settings=provider_settings,
        process_runner=runner,
    )


@pytest.mark.parametrize(
    ("target", "field_name", "invalid_value"),
    (
        ("version_server", "Version", "different"),
        ("version_server", "ApiVersion", "different"),
        ("info", "Architecture", "different"),
        ("info", "Driver", "different"),
        ("info", "DockerRootDir", "/another/root"),
        ("info", "CgroupDriver", "cgroupfs"),
        ("info", "CgroupVersion", "different"),
        ("info", "MemoryLimit", False),
        ("info", "SwapLimit", False),
        ("info", "PidsLimit", False),
        (
            "info",
            "SecurityOptions",
            ["name=apparmor", "name=cgroupns", "name=extra"],
        ),
        ("info", "Runtimes", {}),
    ),
)
def test_runtime_rejects_changed_daemon_authority(
    tmp_path,
    provider_settings,
    target,
    field_name,
    invalid_value,
):
    version = _version(provider_settings)
    info = _info(provider_settings)
    if target == "version_server":
        version["Server"][field_name] = invalid_value
    else:
        info[field_name] = invalid_value
    tmp_path.chmod(0o700)
    runner = _ScriptedProcessRunner(
        [
            {"stdout": _json_line(version)},
            {"stdout": _json_line(info)},
        ]
    )

    with pytest.raises(PinnedDockerRuntimeError, match="daemon differs"):
        PinnedDockerRuntime(
            trusted_root=tmp_path.resolve(),
            settings=provider_settings,
            process_runner=runner,
        )


def test_runtime_rejects_ambiguous_or_failed_control_output(
    tmp_path,
    provider_settings,
):
    tmp_path.chmod(0o700)
    runner = _ScriptedProcessRunner(
        [
            {"stdout": _json_line(_version(provider_settings))},
            {
                "stdout": _json_line(_info(provider_settings)),
                "stderr": b"warning\n",
            },
        ]
    )

    with pytest.raises(PinnedDockerRuntimeError, match="failed"):
        PinnedDockerRuntime(
            trusted_root=tmp_path.resolve(),
            settings=provider_settings,
            process_runner=runner,
        )


def test_runtime_rejects_duplicate_json_keys(tmp_path, provider_settings):
    tmp_path.chmod(0o700)
    runner = _ScriptedProcessRunner(
        [
            {"stdout": b'{"Client":{},"Client":{},"Server":{}}\n'},
        ]
    )

    with pytest.raises(PinnedDockerRuntimeError, match="duplicate key"):
        PinnedDockerRuntime(
            trusted_root=tmp_path.resolve(),
            settings=provider_settings,
            process_runner=runner,
        )


def test_runtime_rejects_nonstandard_json_constants(tmp_path, provider_settings):
    tmp_path.chmod(0o700)
    runner = _ScriptedProcessRunner(
        [
            {"stdout": b'{"Client":NaN,"Server":{}}\n'},
        ]
    )

    with pytest.raises(PinnedDockerRuntimeError, match="nonstandard constant"):
        PinnedDockerRuntime(
            trusted_root=tmp_path.resolve(),
            settings=provider_settings,
            process_runner=runner,
        )


def test_runtime_accepts_only_the_exact_local_image_identity(
    tmp_path,
    provider_settings,
):
    runtime_contract = _runtime_contract()
    runtime, runner = _make_runtime(
        tmp_path,
        provider_settings,
        _fresh_image_outputs(provider_settings, _image(runtime_contract)),
    )

    inspected = runtime.inspect_exact_image(runtime_contract)

    assert inspected == _image(runtime_contract)
    assert len(runner.requests) == 5
    assert runner.requests[-1].argv[-5:] == (
        "image",
        "inspect",
        "--format",
        "{{json .}}",
        runtime_contract.image_reference,
    )


@pytest.mark.parametrize(
    "image_reference",
    (
        "registry.example/kapso/replay-runtime:latest",
        "--help@sha256:" + "a" * 64,
        "registry.example/UPPER/replay@sha256:" + "a" * 64,
        "unqualified/replay@sha256:" + "a" * 64,
        "registry.example:0/kapso/replay@sha256:" + "a" * 64,
        "registry.example:999999/kapso/replay@sha256:" + "a" * 64,
    ),
)
def test_image_authority_requires_a_canonical_registry_qualified_digest_reference(
    image_reference,
):
    runtime_contract = _runtime_contract()

    with pytest.raises(PinnedDockerRuntimeError, match="image authority is invalid"):
        replace(
            runtime_contract,
            image_reference=image_reference,
        )


@pytest.mark.parametrize(
    "repository",
    (
        "registry.example/kapso/replay-runtime",
        "localhost:5000/kapso/replay-runtime",
        "127.0.0.1:5000/kapso/replay-runtime",
        "registry:5000/kapso/replay-runtime",
    ),
)
def test_image_authority_accepts_canonical_registry_qualified_repositories(
    repository,
):
    runtime_contract = _runtime_contract()
    authority = DockerImageAuthority.mint(
        image_reference=f"{repository}@sha256:" + "a" * 64,
        image_config_digest=runtime_contract.image_config_digest,
        operating_system=runtime_contract.operating_system,
        architecture=runtime_contract.architecture,
        architecture_variant=runtime_contract.architecture_variant,
    )

    assert authority.image_reference.startswith(repository)


def test_image_authority_round_trips_with_verified_content_identity():
    authority = _runtime_contract()

    assert DockerImageAuthority.from_json_bytes(authority.to_json_bytes()) == authority
    assert authority.image_authority_id.startswith("docker-image-authority:sha256:")


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("Id", "sha256:" + "f" * 64),
        ("RepoDigests", []),
        ("Architecture", "arm64"),
        ("Variant", "v8"),
    ),
)
def test_runtime_rejects_changed_image_identity(
    tmp_path,
    provider_settings,
    field_name,
    invalid_value,
):
    runtime_contract = _runtime_contract()
    image = _image(runtime_contract)
    image[field_name] = invalid_value
    runtime, _ = _make_runtime(
        tmp_path,
        provider_settings,
        _fresh_image_outputs(provider_settings, image),
    )

    with pytest.raises(PinnedDockerRuntimeError, match="image differs"):
        runtime.inspect_exact_image(runtime_contract)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("operating_system", " "),
        ("architecture", "amd 64"),
        ("architecture_variant", "-v8"),
    ),
)
def test_image_authority_requires_canonical_platform_identifiers(
    field_name,
    value,
):
    with pytest.raises(PinnedDockerRuntimeError, match="image authority is invalid"):
        replace(_runtime_contract(), **{field_name: value})


def test_runtime_returns_image_configuration_without_imposing_consumer_policy(
    tmp_path,
    provider_settings,
):
    authority = _runtime_contract()
    image = _image(authority)
    image["Config"] = {
        "Cmd": ["run"],
        "Entrypoint": ["/usr/bin/tool"],
        "Env": ["PURPOSE=generic"],
        "Healthcheck": {"Test": ["NONE"]},
        "Volumes": {"/data": {}},
    }
    runtime, _ = _make_runtime(
        tmp_path,
        provider_settings,
        _fresh_image_outputs(provider_settings, image),
    )

    assert runtime.inspect_exact_image(authority) == image


def test_runtime_revalidates_pinned_cli_before_each_command(
    tmp_path,
    provider_settings,
):
    runtime, _ = _make_runtime(tmp_path, provider_settings)
    pinned_path = next((tmp_path / "authority").iterdir())
    pinned_path.chmod(0o700)

    with pytest.raises(PinnedDockerRuntimeError, match="private Docker executable"):
        runtime.require_live_authority()


def test_runtime_rejects_provider_settings_with_an_unpinned_cli_digest(
    tmp_path,
    provider_settings,
):
    tmp_path.chmod(0o700)
    runner = _ScriptedProcessRunner([])
    changed = replace(
        provider_settings,
        runtime_executable_digest="sha256:" + "f" * 64,
    )

    with pytest.raises(PinnedDockerRuntimeError, match="pinned digest"):
        PinnedDockerRuntime(
            trusted_root=tmp_path.resolve(),
            settings=changed,
            process_runner=runner,
        )
