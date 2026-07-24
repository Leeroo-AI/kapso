from dataclasses import replace

import pytest

import kapso.cross_run.expert.task_evaluation_docker_bootstrap as bootstrap_module
from kapso.cross_run.docker.runtime import PinnedDockerRuntime
from kapso.cross_run.expert.task_evaluation_docker_bootstrap import (
    TaskEvaluationDockerBootstrapError,
    TaskEvaluationDockerWorkspaceError,
    build_task_evaluation_docker_provider_registry,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    project_prepared_task_evaluation_cases,
)
from test_expert_task_evaluation_reservation import _parent_prepared


@pytest.fixture(scope="module")
def prepared_task_evaluation(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("task-evaluation-docker-bootstrap")
    yield _parent_prepared(root, monkeypatch)[2]
    monkeypatch.undo()


def _runtime(trusted_root, settings):
    runtime = object.__new__(PinnedDockerRuntime)
    runtime._trusted_root = trusted_root
    runtime._settings = settings
    return runtime


def test_bootstrap_resolves_the_complete_request_before_lazy_runtime_creation(
    tmp_path,
    monkeypatch,
    prepared_task_evaluation,
):
    tmp_path.chmod(0o700)
    settings = prepared_task_evaluation.plan_join.settings
    provider_settings = settings.task_evaluation_provider
    trusted_root = (tmp_path / provider_settings.workspace_path).resolve()
    runtime = _runtime(trusted_root, provider_settings.runtime)
    constructions = []

    def create_runtime(*, trusted_root, settings):
        constructions.append((trusted_root, settings))
        return runtime

    monkeypatch.setattr(
        bootstrap_module.PinnedDockerRuntime,
        "create",
        create_runtime,
    )

    registry = build_task_evaluation_docker_provider_registry(
        prepared_request=prepared_task_evaluation,
        workspace_root=tmp_path.resolve(),
    )
    resolved_cases = registry.resolve_all()

    assert constructions == []
    assert tuple(case.executable_case for case in resolved_cases) == (
        project_prepared_task_evaluation_cases(prepared_task_evaluation)
    )
    runtime_authorities = {
        id(case._provider._runtime_authority) for case in resolved_cases
    }
    assert len(runtime_authorities) == 1
    runtime_authority = resolved_cases[0]._provider._runtime_authority
    assert runtime_authority.get() is runtime
    assert runtime_authority.get() is runtime
    assert constructions == [(trusted_root, provider_settings.runtime)]


@pytest.mark.parametrize(
    "field_name",
    (
        "execution_protocol_version",
        "execution_provider_id",
        "execution_provider_version",
        "execution_provider_settings_digest",
        "sandbox_policy_version",
        "task_adapter_runtime_protocol_version",
        "task_evaluator_protocol_version",
    ),
)
def test_bootstrap_rejects_every_implementation_key_dimension_before_filesystem(
    tmp_path,
    monkeypatch,
    prepared_task_evaluation,
    field_name,
):
    tmp_path.chmod(0o700)
    executable_case = project_prepared_task_evaluation_cases(prepared_task_evaluation)[
        0
    ]
    supported_key = executable_case.provider_key
    unsupported_value = (
        "sha256:" + "f" * 64
        if field_name == "execution_provider_settings_digest"
        else f"{getattr(supported_key, field_name)}.future"
    )
    unsupported_key = replace(supported_key, **{field_name: unsupported_value})

    class UnsupportedCase:
        provider_key = unsupported_key

    monkeypatch.setattr(
        bootstrap_module,
        "project_prepared_task_evaluation_cases",
        lambda _prepared: (UnsupportedCase(),),
    )
    configured_root = (
        tmp_path
        / prepared_task_evaluation.plan_join.settings.task_evaluation_provider.workspace_path
    )

    with pytest.raises(TaskEvaluationDockerBootstrapError, match="unsupported key"):
        build_task_evaluation_docker_provider_registry(
            prepared_request=prepared_task_evaluation,
            workspace_root=tmp_path.resolve(),
        )

    assert not configured_root.exists()


def test_bootstrap_registry_rejects_foreign_prepared_authority_without_docker(
    tmp_path,
    monkeypatch,
    prepared_task_evaluation,
):
    tmp_path.chmod(0o700)
    registry = build_task_evaluation_docker_provider_registry(
        prepared_request=prepared_task_evaluation,
        workspace_root=tmp_path.resolve(),
    )
    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir(mode=0o700)
    foreign_prepared = _parent_prepared(foreign_root, monkeypatch)[2]

    def reject_runtime_construction(**_arguments):
        raise AssertionError("prepared-authority comparison contacted Docker")

    monkeypatch.setattr(
        bootstrap_module.PinnedDockerRuntime,
        "create",
        reject_runtime_construction,
    )

    with pytest.raises(ValueError, match="prepared authority"):
        registry.require_exact_prepared_authority(foreign_prepared)


def test_bootstrap_rejects_symlinked_configured_root_without_docker(
    tmp_path,
    monkeypatch,
    prepared_task_evaluation,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(mode=0o700)
    outside_root = tmp_path / "outside"
    outside_root.mkdir(mode=0o700)
    (workspace_root / ".kapso").symlink_to(outside_root, target_is_directory=True)

    def reject_runtime_construction(**_arguments):
        raise AssertionError("unsafe workspace contacted Docker")

    monkeypatch.setattr(
        bootstrap_module.PinnedDockerRuntime,
        "create",
        reject_runtime_construction,
    )

    with pytest.raises(TaskEvaluationDockerWorkspaceError, match="contains a symlink"):
        build_task_evaluation_docker_provider_registry(
            prepared_request=prepared_task_evaluation,
            workspace_root=workspace_root.resolve(),
        )
