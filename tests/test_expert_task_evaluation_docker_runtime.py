from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import replace

import pytest

import kapso.cross_run.expert.task_evaluation_docker_runtime as runtime_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import TaskAdapterRuntimeContract
from kapso.cross_run.expert.task_evaluation_docker_runtime import (
    TaskEvaluationDockerRuntime,
    TaskEvaluationDockerRuntimeError,
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
    ).expert.validation.task_evaluation_provider
    return replace(
        settings,
        runtime_executable_digest=tree_or_blob_digest(_TEST_EXECUTABLE_BYTES),
    )


@pytest.fixture(autouse=True)
def isolated_local_authority(monkeypatch, provider_settings):
    def read_executable(_path, expected_digest):
        if expected_digest != provider_settings.runtime_executable_digest:
            raise TaskEvaluationDockerRuntimeError(
                "task evaluation authority executable differs from its pinned digest"
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
        "CgroupVersion": settings.runtime_cgroup_version,
        "CpuCfsPeriod": True,
        "CpuCfsQuota": True,
        "DefaultRuntime": settings.runtime_default_runtime,
        "Driver": settings.runtime_storage_driver,
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
    return TaskAdapterRuntimeContract(
        runtime_protocol_version="kapso.task_adapter_runtime.v1",
        image_repository="registry.example/kapso/replay-runtime",
        image_manifest_digest=tree_or_blob_digest(b"manifest"),
        image_config_digest=tree_or_blob_digest(b"config"),
        dependency_lock_path="requirements.lock",
        dependency_lock_digest=tree_or_blob_digest(b"lock"),
        operating_system="linux",
        architecture="amd64",
        architecture_variant=None,
        environment={"LANG": "C", "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
    )


def _image(runtime):
    return {
        "Architecture": runtime.architecture,
        "Config": {
            "Cmd": None,
            "Entrypoint": None,
            "Env": [f"{key}={value}" for key, value in runtime.environment.items()],
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
    runtime = TaskEvaluationDockerRuntime(
        trusted_root=tmp_path.resolve(),
        settings=settings,
        process_runner=runner,
    )
    return runtime, runner


def _construct_runtime_after_signal(trusted_root, settings, start_signal):
    start_signal.wait()
    TaskEvaluationDockerRuntime(
        trusted_root=trusted_root,
        settings=settings,
        process_runner=_ScriptedProcessRunner([]),
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
        runtime_module.TaskEvaluationDockerRuntime,
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

    TaskEvaluationDockerRuntime(
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

    with pytest.raises(TaskEvaluationDockerRuntimeError, match="daemon differs"):
        TaskEvaluationDockerRuntime(
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

    with pytest.raises(TaskEvaluationDockerRuntimeError, match="failed"):
        TaskEvaluationDockerRuntime(
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

    with pytest.raises(TaskEvaluationDockerRuntimeError, match="duplicate key"):
        TaskEvaluationDockerRuntime(
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

    with pytest.raises(TaskEvaluationDockerRuntimeError, match="nonstandard constant"):
        TaskEvaluationDockerRuntime(
            trusted_root=tmp_path.resolve(),
            settings=provider_settings,
            process_runner=runner,
        )


def test_runtime_accepts_only_the_exact_local_image(tmp_path, provider_settings):
    runtime_contract = _runtime_contract()
    runtime, runner = _make_runtime(
        tmp_path,
        provider_settings,
        _fresh_image_outputs(provider_settings, _image(runtime_contract)),
    )

    runtime.require_exact_image(runtime_contract)

    assert len(runner.requests) == 5
    assert runner.requests[-1].argv[-5:] == (
        "image",
        "inspect",
        "--format",
        "{{json .}}",
        runtime_contract.image_reference,
    )


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    (
        (("Id",), "sha256:" + "f" * 64),
        (("RepoDigests",), []),
        (("Architecture",), "arm64"),
        (("Variant",), "v8"),
        (("Config", "Env"), ["LANG=C", "EXTRA=value"]),
        (("Config", "Cmd"), ["inherited"]),
        (("Config", "Entrypoint"), ["/inherited-entrypoint"]),
        (("Config", "Volumes"), {"/kapso/writable": {}}),
        (("Config", "Healthcheck"), {"Test": ["NONE"]}),
    ),
)
def test_runtime_rejects_changed_image_authority(
    tmp_path,
    provider_settings,
    field_path,
    invalid_value,
):
    runtime_contract = _runtime_contract()
    image = _image(runtime_contract)
    target = image
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = invalid_value
    runtime, _ = _make_runtime(
        tmp_path,
        provider_settings,
        _fresh_image_outputs(provider_settings, image),
    )

    with pytest.raises(TaskEvaluationDockerRuntimeError, match="image differs"):
        runtime.require_exact_image(runtime_contract)


def test_runtime_revalidates_pinned_cli_before_each_command(
    tmp_path,
    provider_settings,
):
    runtime, _ = _make_runtime(tmp_path, provider_settings)
    pinned_path = next((tmp_path / "authority").iterdir())
    pinned_path.chmod(0o700)

    with pytest.raises(TaskEvaluationDockerRuntimeError, match="private executable"):
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

    with pytest.raises(TaskEvaluationDockerRuntimeError, match="pinned digest"):
        TaskEvaluationDockerRuntime(
            trusted_root=tmp_path.resolve(),
            settings=changed,
            process_runner=runner,
        )
