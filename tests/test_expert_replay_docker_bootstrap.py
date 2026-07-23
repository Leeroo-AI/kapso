from __future__ import annotations

import multiprocessing
import stat
import time
from dataclasses import replace

import pytest

import kapso.cross_run.expert.replay_docker_bootstrap as bootstrap_module
from kapso.cross_run.expert.replay_docker_bootstrap import (
    SourceReplayDockerBootstrapError,
    build_source_replay_docker_provider_registry,
)
from kapso.cross_run.expert.replay_docker_provider import (
    SourceReplayDockerExecutionProvider,
    SourceReplayDockerProviderError,
)
from kapso.cross_run.expert.task_evaluation_docker_runtime import (
    TaskEvaluationDockerRuntime,
)
from kapso.cross_run.expert.replay_execution import (
    expert_source_replay_execution_provider_key,
)
from test_expert_source_replay_request import _prepared, _request_fixture


@pytest.fixture(scope="module")
def prepared_replay_request(tmp_path_factory):
    return _prepared(
        _request_fixture(tmp_path_factory.mktemp("expert-replay-docker-bootstrap"))
    )


@pytest.fixture(scope="module")
def other_prepared_replay_request(tmp_path_factory):
    return _prepared(
        _request_fixture(
            tmp_path_factory.mktemp("other-expert-replay-docker-bootstrap")
        )
    )


def _runtime(trusted_root, settings):
    runtime = object.__new__(TaskEvaluationDockerRuntime)
    runtime._trusted_root = trusted_root
    runtime._settings = settings
    return runtime


def _build_registry_after_signal(prepared_request, workspace_root, start_signal):
    start_signal.wait()
    build_source_replay_docker_provider_registry(
        prepared_request=prepared_request,
        workspace_root=workspace_root,
    )


def test_bootstrap_binds_request_keys_to_one_historical_runtime(
    tmp_path,
    monkeypatch,
    prepared_replay_request,
):
    tmp_path.chmod(0o700)
    provider_settings = prepared_replay_request.settings.task_evaluation_provider
    configured_root = (tmp_path / provider_settings.workspace_path).resolve()
    runtime = _runtime(configured_root, provider_settings)
    constructions = []
    assert not configured_root.exists()

    def create_runtime(*, trusted_root, settings):
        constructions.append((trusted_root, settings))
        return runtime

    monkeypatch.setattr(
        bootstrap_module.TaskEvaluationDockerRuntime,
        "create",
        create_runtime,
    )

    registry = build_source_replay_docker_provider_registry(
        prepared_request=prepared_replay_request,
        workspace_root=tmp_path.resolve(),
    )
    resolved_cases = registry.resolve_all(prepared_replay_request)

    assert constructions == []
    current_path = tmp_path
    for part in provider_settings.workspace_path.split("/"):
        current_path /= part
        metadata = current_path.stat()
        assert stat.S_ISDIR(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o700
    assert tuple(case.materialized_case for case in resolved_cases) == (
        prepared_replay_request.cases
    )
    runtime_authorities = {
        id(case._provider._runtime_authority) for case in resolved_cases
    }
    assert len(runtime_authorities) == 1
    runtime_authority = resolved_cases[0]._provider._runtime_authority
    assert runtime_authority.trusted_root == configured_root
    assert runtime_authority.get() is runtime
    assert runtime_authority.get() is runtime
    assert constructions == [(configured_root, provider_settings)]
    assert all(
        case.dispatch_key
        == expert_source_replay_execution_provider_key(case.materialized_case)
        for case in resolved_cases
    )


def test_independent_process_registries_serialize_clean_root_initialization(
    tmp_path,
    monkeypatch,
    prepared_replay_request,
):
    tmp_path.chmod(0o700)
    process_context = multiprocessing.get_context("fork")
    concurrent_calls = process_context.Array("i", (0, 0), lock=True)
    start_signal = process_context.Event()
    original_child_exists = bootstrap_module._configured_child_exists

    def observe_child_exists(parent_descriptor, name):
        with concurrent_calls.get_lock():
            concurrent_calls[0] += 1
            concurrent_calls[1] = max(concurrent_calls[1], concurrent_calls[0])
        time.sleep(0.05)
        exists = original_child_exists(parent_descriptor, name)
        with concurrent_calls.get_lock():
            concurrent_calls[0] -= 1
        return exists

    monkeypatch.setattr(
        bootstrap_module,
        "_configured_child_exists",
        observe_child_exists,
    )
    processes = tuple(
        process_context.Process(
            target=_build_registry_after_signal,
            args=(prepared_replay_request, tmp_path.resolve(), start_signal),
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


def test_registry_rejects_another_prepared_request_without_docker(
    tmp_path,
    monkeypatch,
    prepared_replay_request,
    other_prepared_replay_request,
):
    tmp_path.chmod(0o700)

    def reject_runtime_construction(**_arguments):
        raise AssertionError("request-bound resolution contacted Docker")

    monkeypatch.setattr(
        bootstrap_module.TaskEvaluationDockerRuntime,
        "create",
        reject_runtime_construction,
    )
    registry = build_source_replay_docker_provider_registry(
        prepared_request=prepared_replay_request,
        workspace_root=tmp_path.resolve(),
    )
    assert expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    ) == expert_source_replay_execution_provider_key(
        other_prepared_replay_request.cases[0]
    )

    with pytest.raises(SourceReplayDockerBootstrapError, match="another prepared"):
        registry.resolve_all(other_prepared_replay_request)


@pytest.mark.parametrize(
    "field_name",
    (
        "paired_execution_protocol_version",
        "execution_provider_id",
        "execution_provider_version",
        "execution_provider_settings_digest",
        "sandbox_policy_version",
        "task_adapter_runtime_protocol_version",
        "task_evaluator_protocol_version",
    ),
)
def test_bootstrap_rejects_every_unsupported_key_before_runtime_construction(
    tmp_path,
    monkeypatch,
    prepared_replay_request,
    field_name,
):
    tmp_path.chmod(0o700)
    supported_key = expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    )
    unsupported_value = (
        "sha256:" + "f" * 64
        if field_name == "execution_provider_settings_digest"
        else f"{getattr(supported_key, field_name)}.future"
    )
    unsupported_key = replace(supported_key, **{field_name: unsupported_value})
    monkeypatch.setattr(
        bootstrap_module,
        "expert_source_replay_execution_provider_key",
        lambda _case: unsupported_key,
    )

    def reject_runtime_construction(**_arguments):
        raise AssertionError("unsupported protocol contacted Docker")

    monkeypatch.setattr(
        bootstrap_module.TaskEvaluationDockerRuntime,
        "create",
        reject_runtime_construction,
    )

    with pytest.raises(SourceReplayDockerBootstrapError, match="unsupported key"):
        build_source_replay_docker_provider_registry(
            prepared_request=prepared_replay_request,
            workspace_root=tmp_path.resolve(),
        )


def test_provider_direct_construction_rejects_unsupported_protocol(
    tmp_path,
    prepared_replay_request,
):
    tmp_path.chmod(0o700)
    settings = prepared_replay_request.settings
    supported_key = expert_source_replay_execution_provider_key(
        prepared_replay_request.cases[0]
    )
    runtime = _runtime(tmp_path.resolve(), settings.task_evaluation_provider)

    with pytest.raises(
        SourceReplayDockerProviderError, match="implementation authority"
    ):
        SourceReplayDockerExecutionProvider(
            dispatch_key=replace(
                supported_key,
                task_evaluator_protocol_version=(
                    f"{supported_key.task_evaluator_protocol_version}.future"
                ),
            ),
            provider_settings=settings.task_evaluation_provider,
            policy_settings=settings.policy,
            runtime=runtime,
        )


def test_bootstrap_rejects_unprepared_input_before_runtime_construction(
    tmp_path,
    monkeypatch,
):
    tmp_path.chmod(0o700)

    def reject_runtime_construction(**_arguments):
        raise AssertionError("invalid request contacted Docker")

    monkeypatch.setattr(
        bootstrap_module.TaskEvaluationDockerRuntime,
        "create",
        reject_runtime_construction,
    )

    with pytest.raises(SourceReplayDockerBootstrapError, match="prepared request"):
        build_source_replay_docker_provider_registry(
            prepared_request=object(),
            workspace_root=tmp_path.resolve(),
        )


def test_bootstrap_rejects_a_symlinked_configured_workspace_before_runtime(
    tmp_path,
    monkeypatch,
    prepared_replay_request,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(mode=0o700)
    outside_root = tmp_path / "outside"
    outside_root.mkdir(mode=0o700)
    (workspace_root / ".kapso").symlink_to(outside_root, target_is_directory=True)

    def reject_runtime_construction(**_arguments):
        raise AssertionError("unsafe workspace contacted Docker")

    monkeypatch.setattr(
        bootstrap_module.TaskEvaluationDockerRuntime,
        "create",
        reject_runtime_construction,
    )

    with pytest.raises(SourceReplayDockerBootstrapError, match="contains a symlink"):
        build_source_replay_docker_provider_registry(
            prepared_request=prepared_replay_request,
            workspace_root=workspace_root.resolve(),
        )
