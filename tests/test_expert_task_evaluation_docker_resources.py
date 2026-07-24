from __future__ import annotations

import json
from dataclasses import replace

import pytest

import kapso.cross_run.docker.runtime as runtime_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import PinnedDockerRuntime
from kapso.cross_run.expert.task_evaluation_docker_resources import (
    TaskEvaluationDockerResourceError,
    TaskEvaluationDockerResourceManager,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessResult,
)
from kapso.cross_run.settings import CrossRunSettings
from test_cross_run_docker_runtime import _info, _json_line, _version
from test_expert_source_replay_request import _prepared, _request_fixture
from test_expert_task_evaluation_provider_filesystem import _matched_invocation

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_TEST_DOCKER_BYTES = b"resource manager Docker executable"


class _StatefulDockerRunner:
    def __init__(self, settings):
        self.settings = settings
        self.requests = []
        self.containers = {}
        self.volumes = {}

    def run(self, request):
        self.requests.append(request)
        arguments = request.argv[5:]
        stdout = self._dispatch(arguments)
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

    def _dispatch(self, arguments):
        if arguments == ("version", "--format", "{{json .}}"):
            return _json_line(_version(self.settings))
        if arguments == ("info", "--format", "{{json .}}"):
            return _json_line(_info(self.settings))
        if arguments[:3] == ("container", "ls", "--all"):
            name = arguments[5].removeprefix("name=^/").removesuffix("$")
            return b"" if name not in self.containers else _json_line(name)
        if arguments[:4] == (
            "container",
            "inspect",
            "--format",
            "{{json .}}",
        ):
            return _json_line(self.containers[arguments[4]])
        if arguments[:2] == ("container", "rm"):
            container_id = arguments[-1]
            name = next(
                name
                for name, payload in self.containers.items()
                if payload["Id"] == container_id
            )
            del self.containers[name]
            return f"{container_id}\n".encode()
        if arguments[:2] == ("volume", "ls"):
            name = arguments[3].removeprefix("name=^").removesuffix("$")
            return b"" if name not in self.volumes else _json_line(name)
        if arguments[:4] == (
            "volume",
            "inspect",
            "--format",
            "{{json .}}",
        ):
            return _json_line(self.volumes[arguments[4]])
        if arguments[:2] == ("volume", "create"):
            name = arguments[-1]
            labels = {
                value.split("=", 1)[0]: value.split("=", 1)[1]
                for position, value in enumerate(arguments)
                if arguments[position - 1] == "--label"
            }
            options = {
                value.split("=", 1)[0]: value.split("=", 1)[1]
                for position, value in enumerate(arguments)
                if arguments[position - 1] == "--opt"
            }
            self.volumes[name] = {
                "Driver": "local",
                "Labels": labels,
                "Name": name,
                "Options": options,
                "Scope": "local",
            }
            return f"{name}\n".encode()
        if arguments[:2] == ("volume", "rm"):
            name = arguments[-1]
            del self.volumes[name]
            return f"{name}\n".encode()
        raise AssertionError(f"unexpected Docker arguments: {arguments}")


@pytest.fixture
def resources(tmp_path, monkeypatch):
    provider_settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).expert.validation.task_evaluation_provider
    runtime_settings = replace(
        provider_settings.runtime,
        runtime_executable_digest=tree_or_blob_digest(_TEST_DOCKER_BYTES),
    )
    provider_settings = replace(provider_settings, runtime=runtime_settings)
    monkeypatch.setattr(
        runtime_module,
        "read_verified_root_executable",
        lambda _path, _digest: _TEST_DOCKER_BYTES,
    )
    monkeypatch.setattr(runtime_module, "_require_runtime_socket", lambda _path: None)
    tmp_path.chmod(0o700)
    runner = _StatefulDockerRunner(runtime_settings)
    runtime = PinnedDockerRuntime(
        trusted_root=tmp_path.resolve(),
        settings=runtime_settings,
        process_runner=runner,
    )
    manager = TaskEvaluationDockerResourceManager(runtime, provider_settings)
    return manager, runner, provider_settings


@pytest.fixture
def replay_invocation(tmp_path):
    prepared = _prepared(_request_fixture(tmp_path))
    return _matched_invocation(prepared, "candidate_leg")


def _container(name, container_id, labels):
    return {
        "Config": {"Labels": dict(labels)},
        "Id": container_id,
        "Name": f"/{name}",
    }


def _volume(name, labels, options=None):
    return {
        "Driver": "local",
        "Labels": dict(labels),
        "Name": name,
        "Options": {} if options is None else options,
        "Scope": "local",
    }


def test_resource_identity_uses_the_full_unpredictable_handle_digest(
    resources,
    replay_invocation,
):
    manager, _, _ = resources
    identity = manager.identity(replay_invocation.provider_handle.provider_handle_id)
    suffix = replay_invocation.provider_handle.provider_handle_id.rsplit(":", 1)[-1]

    assert (
        identity.workspace_root == manager.runtime.trusted_root / f"execution-{suffix}"
    )
    assert identity.evaluator_name.endswith(suffix)
    assert identity.keeper_name.endswith(suffix)
    assert identity.volume_name.endswith(suffix)
    assert identity.labels_for("evaluator") == {
        "io.kapso.task-evaluation.handle": (
            replay_invocation.provider_handle.provider_handle_id
        ),
        "io.kapso.task-evaluation.role": "evaluator",
    }


def test_writable_volume_is_fresh_exact_labelled_tmpfs(
    resources,
    replay_invocation,
):
    manager, runner, provider_settings = resources
    identity = manager.require_absent(
        replay_invocation.provider_handle.provider_handle_id
    )
    compute = replay_invocation.materialized_case.request_case.compute_binding

    observation = manager.create_writable_volume(
        identity,
        writable_storage_byte_limit=compute.writable_storage_byte_limit,
        writable_inode_limit=compute.writable_inode_limit,
    )

    assert observation.name == identity.volume_name
    assert observation.payload["Options"] == {
        "device": "tmpfs",
        "o": (
            f"uid={provider_settings.container_user_id},"
            f"gid={provider_settings.container_group_id},"
            "mode=0700,"
            f"size={compute.writable_storage_byte_limit},"
            f"nr_inodes={compute.writable_inode_limit},"
            "nosuid,nodev,noexec"
        ),
        "type": "tmpfs",
    }
    create_arguments = next(
        request.argv[5:]
        for request in runner.requests
        if request.argv[5:7] == ("volume", "create")
    )
    assert create_arguments[-1] == identity.volume_name

    with pytest.raises(TaskEvaluationDockerResourceError, match="not fresh"):
        manager.create_writable_volume(
            identity,
            writable_storage_byte_limit=compute.writable_storage_byte_limit,
            writable_inode_limit=compute.writable_inode_limit,
        )


def test_cleanup_removes_only_prevalidated_ids_and_is_repeatable(
    resources,
    replay_invocation,
):
    manager, runner, _ = resources
    identity = manager.identity(replay_invocation.provider_handle.provider_handle_id)
    evaluator_id = "a" * 64
    keeper_id = "b" * 64
    runner.containers[identity.evaluator_name] = _container(
        identity.evaluator_name,
        evaluator_id,
        identity.labels_for("evaluator"),
    )
    runner.containers[identity.keeper_name] = _container(
        identity.keeper_name,
        keeper_id,
        identity.labels_for("keeper"),
    )
    runner.volumes[identity.volume_name] = _volume(
        identity.volume_name,
        identity.labels_for("volume"),
    )

    assert (
        manager.cleanup_daemon_resources(
            replay_invocation.provider_handle.provider_handle_id
        )
        == identity
    )
    assert (
        manager.cleanup_daemon_resources(
            replay_invocation.provider_handle.provider_handle_id
        )
        == identity
    )

    mutations = tuple(
        request.argv[5:]
        for request in runner.requests
        if request.argv[5:7]
        in {
            ("container", "rm"),
            ("volume", "rm"),
        }
    )
    assert mutations == (
        ("container", "rm", "--force", "--volumes", evaluator_id),
        ("container", "rm", "--force", "--volumes", keeper_id),
        ("volume", "rm", identity.volume_name),
    )
    assert not any(
        arguments[0] in {"start", "exec"}
        or arguments[:2] in {("container", "start"), ("container", "exec")}
        for arguments in (request.argv[5:] for request in runner.requests)
    )


def test_cleanup_rejects_substituted_labels_before_any_removal(
    resources,
    replay_invocation,
):
    manager, runner, _ = resources
    identity = manager.identity(replay_invocation.provider_handle.provider_handle_id)
    runner.containers[identity.evaluator_name] = _container(
        identity.evaluator_name,
        "a" * 64,
        {
            "io.kapso.task-evaluation.handle": "substituted",
            "io.kapso.task-evaluation.role": "evaluator",
        },
    )

    with pytest.raises(TaskEvaluationDockerResourceError, match="handle labels"):
        manager.cleanup_daemon_resources(
            replay_invocation.provider_handle.provider_handle_id
        )

    assert identity.evaluator_name in runner.containers
    assert not any(
        request.argv[5:7] in {("container", "rm"), ("volume", "rm")}
        for request in runner.requests
    )


def test_ambiguous_resource_lookup_fails_loud(resources, replay_invocation):
    manager, runner, _ = resources
    identity = manager.identity(replay_invocation.provider_handle.provider_handle_id)
    original_dispatch = runner._dispatch

    def ambiguous(arguments):
        if arguments[:3] == ("container", "ls", "--all"):
            return json.dumps(identity.evaluator_name).encode() + b"\nextra\n"
        return original_dispatch(arguments)

    runner._dispatch = ambiguous

    with pytest.raises(TaskEvaluationDockerResourceError, match="ambiguous"):
        manager.observe(identity)


def test_container_removal_normalizes_only_mount_order(
    resources,
    replay_invocation,
):
    manager, runner, _ = resources
    identity = manager.identity(replay_invocation.provider_handle.provider_handle_id)
    evaluator_id = "a" * 64
    runner.containers[identity.evaluator_name] = {
        **_container(
            identity.evaluator_name,
            evaluator_id,
            identity.labels_for("evaluator"),
        ),
        "Mounts": [
            {
                "Destination": "/kapso/input",
                "RW": False,
                "Type": "bind",
            },
            {
                "Destination": "/kapso/writable",
                "RW": True,
                "Type": "volume",
            },
        ],
    }
    observation = manager.observe(identity)[0]
    runner.containers[identity.evaluator_name]["Mounts"].reverse()

    manager.remove_container(identity, observation)

    assert identity.evaluator_name not in runner.containers


def test_container_removal_rejects_changed_mount_authority(
    resources,
    replay_invocation,
):
    manager, runner, _ = resources
    identity = manager.identity(replay_invocation.provider_handle.provider_handle_id)
    runner.containers[identity.evaluator_name] = {
        **_container(
            identity.evaluator_name,
            "a" * 64,
            identity.labels_for("evaluator"),
        ),
        "Mounts": [
            {
                "Destination": "/kapso/input",
                "RW": False,
                "Type": "bind",
            }
        ],
    }
    observation = manager.observe(identity)[0]
    runner.containers[identity.evaluator_name]["Mounts"][0]["RW"] = True

    with pytest.raises(TaskEvaluationDockerResourceError, match="changed"):
        manager.remove_container(identity, observation)

    assert identity.evaluator_name in runner.containers
