from dataclasses import replace
from types import MappingProxyType

import pytest

import kapso.cross_run.expert.task_evaluation_docker_runtime as runtime_module
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationInvocationAllocation,
)
from kapso.cross_run.expert.task_evaluation_docker_provider import (
    TaskEvaluationDockerExecutionProvider,
    TaskEvaluationDockerProviderError,
    TaskEvaluationDockerSandboxCompletion,
    TaskEvaluationDockerSandboxInvocation,
)
from kapso.cross_run.expert.task_evaluation_docker_runtime import (
    TaskEvaluationDockerRuntime,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderRegistry,
    TaskEvaluationProviderCompletion,
    project_prepared_task_evaluation_cases,
)
from kapso.cross_run.process import (
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessResult,
)
from test_expert_replay_docker_provider import _ProviderDockerRunner
from test_expert_task_evaluation_provider_filesystem import _RESULT_PAYLOAD
from test_expert_task_evaluation_reservation import _parent_prepared


class _InvocationCaptureProvider:
    def __init__(self, dispatch_key, result_root):
        self.dispatch_key = dispatch_key
        self.result_root = result_root
        self.requirements = []
        self.invocations = []

    def require_supported_execution(self, requirements):
        self.requirements.append(requirements)

    def execute_leg(self, invocation):
        self.invocations.append(invocation)
        request = BoundedProcessRequest(
            argv=("true",),
            trusted_root=self.result_root,
            cwd=self.result_root,
            timeout_seconds=1,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1,
            stderr_byte_limit=1,
            environment={},
        )
        return TaskEvaluationProviderCompletion(
            provider_handle_id=invocation.provider_handle.provider_handle_id,
            process_result=BoundedProcessResult(
                request=request,
                outcome=BoundedProcessOutcome.COMPLETED,
                returncode=0,
                stdout=b"",
                stderr=b"",
                stdout_bytes_observed=0,
                stderr_bytes_observed=0,
                duration_seconds=0.0,
            ),
            result_payload=b"captured",
        )

    def cleanup_interrupted(self, _provider_handle):
        raise AssertionError("invocation capture must not clean resources")


@pytest.fixture(scope="module")
def task_evaluation_authority(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("task-evaluation-docker-provider-authority")
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        root,
        monkeypatch,
    )
    reservation = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    ).reservation
    executable_case = project_prepared_task_evaluation_cases(prepared)[0]
    capture = _InvocationCaptureProvider(
        executable_case.provider_key,
        root.resolve(),
    )
    registry = TaskEvaluationExecutionProviderRegistry(prepared, (capture,))
    allocation = TaskEvaluationInvocationAllocation(
        reservation_id=reservation.reservation.reservation_id,
        evaluation_case_id=executable_case.evaluation_case_id,
        evaluation_leg_id=executable_case.legs[0].authority.leg_id,
        invocation_nonce="a" * 32,
    )
    resolved_case = registry._resolved_case_for_allocation(
        prepared_request=prepared,
        reservation_snapshot=reservation,
        invocation_allocation=allocation,
    )
    registry._execute_journal_leg(
        prepared_request=prepared,
        reservation_snapshot=reservation,
        resolved_case=resolved_case,
        invocation_allocation=allocation,
    )
    assert len(capture.requirements) == 1
    assert len(capture.invocations) == 1
    yield prepared, capture.requirements[0], capture.invocations[0]
    monkeypatch.undo()


def _runtime(tmp_path, monkeypatch, prepared, requirements):
    provider_settings = prepared.plan_join.settings.task_evaluation_provider
    adapter_runtime = requirements.runtime_contract
    tmp_path.chmod(0o700)
    docker_path = tmp_path / "docker"
    docker_path.write_bytes(b"docker")
    docker_path.chmod(0o500)
    docker_config_root = tmp_path / "config"
    docker_config_root.mkdir(mode=0o700)
    runner = _ProviderDockerRunner(
        provider_settings,
        adapter_runtime,
        requirements,
    )
    runtime = object.__new__(TaskEvaluationDockerRuntime)
    runtime._trusted_root = tmp_path.resolve()
    runtime._settings = provider_settings
    runtime._process_runner = runner
    runtime._docker_path = docker_path
    runtime._docker_digest = provider_settings.runtime_executable_digest
    runtime._docker_config_root = docker_config_root
    runtime._environment = MappingProxyType(
        {
            "DOCKER_API_VERSION": provider_settings.runtime_api_version,
            "DOCKER_CONFIG": str(docker_config_root),
            "HOME": str(tmp_path),
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    monkeypatch.setattr(
        runtime_module,
        "read_verified_private_executable",
        lambda _path: provider_settings.runtime_executable_digest,
    )
    monkeypatch.setattr(runtime_module, "_require_runtime_socket", lambda _path: None)
    return runtime, runner


def _provider(tmp_path, monkeypatch, task_evaluation_authority):
    prepared, requirements, invocation = task_evaluation_authority
    runtime, runner = _runtime(tmp_path, monkeypatch, prepared, requirements)

    def materialize_helper(self, identity):
        helper_root = identity.workspace_root / "provider"
        helper_root.mkdir(mode=0o700)
        helper_path = helper_root / "busybox"
        helper_path.write_bytes(b"busybox")
        helper_path.chmod(0o555)
        helper_root.chmod(0o555)
        return helper_root

    monkeypatch.setattr(
        TaskEvaluationDockerExecutionProvider,
        "_materialize_helper",
        materialize_helper,
    )
    provider = TaskEvaluationDockerExecutionProvider(
        dispatch_key=requirements.dispatch_key,
        provider_settings=prepared.plan_join.settings.task_evaluation_provider,
        policy_settings=prepared.plan_join.settings.policy,
        runtime=runtime,
    )
    return provider, runner, invocation


def test_provider_runs_selected_task_leg_with_exact_mounts_and_reaps_resources(
    tmp_path,
    monkeypatch,
    task_evaluation_authority,
):
    provider, runner, invocation = _provider(
        tmp_path,
        monkeypatch,
        task_evaluation_authority,
    )

    completion = provider.execute_leg(invocation)

    assert (
        completion.provider_handle_id == invocation.provider_handle.provider_handle_id
    )
    assert completion.process_result.outcome is BoundedProcessOutcome.COMPLETED
    assert completion.result_payload == _RESULT_PAYLOAD
    assert runner.containers == {}
    assert runner.volumes == {}
    assert tuple(provider._runtime.trusted_root.glob("execution-*")) == ()
    commands = tuple(request.argv[5:] for request in runner.requests)
    evaluator_create = next(
        command
        for command in commands
        if command[:2] == ("container", "create")
        and "io.kapso.task-evaluation.role=evaluator" in command
    )
    mounts = tuple(
        evaluator_create[position + 1]
        for position, value in enumerate(evaluator_create[:-1])
        if value == "--mount"
    )
    assert len(mounts) == 2
    assert any(
        "dst=/kapso/input,readonly,bind-recursive=disabled" in mount for mount in mounts
    )
    assert any(
        "dst=/kapso/writable" in mount and "readonly" not in mount for mount in mounts
    )
    assert "--network" in evaluator_create
    assert evaluator_create[evaluator_create.index("--network") + 1] == "none"
    assert "--pull" in evaluator_create
    assert evaluator_create[evaluator_create.index("--pull") + 1] == "never"


def test_provider_projects_only_selected_bytes_and_blinded_request(
    tmp_path,
    monkeypatch,
    task_evaluation_authority,
):
    prepared, _requirements, _captured_invocation = task_evaluation_authority
    provider, runner, invocation = _provider(
        tmp_path,
        monkeypatch,
        task_evaluation_authority,
    )
    observed_sandbox_invocations = []

    def execute_sandbox(_self, sandbox_invocation):
        observed_sandbox_invocations.append(sandbox_invocation)
        request = BoundedProcessRequest(
            argv=("true",),
            trusted_root=tmp_path.resolve(),
            cwd=tmp_path.resolve(),
            timeout_seconds=1,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1,
            stderr_byte_limit=1,
            environment={},
        )
        return TaskEvaluationDockerSandboxCompletion(
            process_result=BoundedProcessResult(
                request=request,
                outcome=BoundedProcessOutcome.COMPLETED,
                returncode=0,
                stdout=b"",
                stderr=b"",
                stdout_bytes_observed=0,
                stderr_bytes_observed=0,
                duration_seconds=0.0,
            ),
            result_payload=None,
        )

    monkeypatch.setattr(
        TaskEvaluationDockerExecutionProvider,
        "_execute_sandbox",
        execute_sandbox,
    )

    provider.execute_leg(invocation)

    assert len(observed_sandbox_invocations) == 1
    sandbox_invocation = observed_sandbox_invocations[0]
    assert type(sandbox_invocation) is TaskEvaluationDockerSandboxInvocation
    assert sandbox_invocation.provider_handle_id == (
        invocation.provider_handle.provider_handle_id
    )
    assert sandbox_invocation.expert_source_contents == (
        invocation.selected_leg.expert_source.source_contents
    )
    assert {
        descriptor.relative_path
        for descriptor in sandbox_invocation.expert_source_files
    } == set(sandbox_invocation.expert_source_contents)
    assert sandbox_invocation.adapter_source_files == (
        invocation.adapter_runtime.source_files
    )
    assert sandbox_invocation.adapter_source_contents == (
        invocation.adapter_runtime.source_contents
    )
    assert not any(
        descriptor.relative_path.startswith("release_matrix_assets/")
        for descriptor in sandbox_invocation.adapter_source_files
    )
    assert {
        artifact.mount_path: artifact.source_contents
        for artifact in sandbox_invocation.task_artifacts
    } == {
        artifact.artifact.mount_path: artifact.source_contents
        for artifact in invocation.starting_artifacts
    }
    request_payload = invocation.task_evaluator_request.to_json_bytes()
    assert sandbox_invocation.request_payload == request_payload
    authority_request = prepared.plan_join.request
    for forbidden_id in (
        authority_request.evaluation_plan_id,
        authority_request.candidate_id,
        authority_request.authorization_transition_id,
        authority_request.cases[0].provenance_binding_id,
    ):
        assert forbidden_id.encode("utf-8") not in request_payload
    assert runner.requests == []


def test_provider_returns_no_result_for_a_technical_outcome_and_still_reaps(
    tmp_path,
    monkeypatch,
    task_evaluation_authority,
):
    provider, runner, invocation = _provider(
        tmp_path,
        monkeypatch,
        task_evaluation_authority,
    )
    runner.attach_outcome = BoundedProcessOutcome.TIMED_OUT
    runner.attach_returncode = 137

    completion = provider.execute_leg(invocation)

    assert completion.process_result.outcome is BoundedProcessOutcome.TIMED_OUT
    assert completion.result_payload is None
    assert runner.containers == {}
    assert runner.volumes == {}
    assert tuple(provider._runtime.trusted_root.glob("execution-*")) == ()


def test_cleanup_is_idempotent_and_never_starts_or_executes_a_container(
    tmp_path,
    monkeypatch,
    task_evaluation_authority,
):
    provider, runner, invocation = _provider(
        tmp_path,
        monkeypatch,
        task_evaluation_authority,
    )

    provider.cleanup_interrupted(invocation.provider_handle)
    first_cleanup_end = len(runner.requests)
    provider.cleanup_interrupted(invocation.provider_handle)

    assert runner.containers == {}
    assert runner.volumes == {}
    cleanup_commands = tuple(
        request.argv[5:] for request in runner.requests[:first_cleanup_end]
    )
    repeated_commands = tuple(
        request.argv[5:] for request in runner.requests[first_cleanup_end:]
    )
    for commands in (cleanup_commands, repeated_commands):
        assert not any(
            command[:2]
            in {
                ("container", "create"),
                ("container", "start"),
                ("container", "exec"),
            }
            for command in commands
        )


@pytest.mark.parametrize(
    "unsupported_requirements",
    (
        lambda requirements: replace(
            requirements,
            accelerator_class_id="cuda",
            accelerator_count=1,
        ),
        lambda requirements: replace(
            requirements,
            runtime_contract=replace(
                requirements.runtime_contract,
                architecture="unsupported-architecture",
            ),
        ),
    ),
)
def test_provider_rejects_unsupported_cases_without_contacting_docker(
    tmp_path,
    monkeypatch,
    task_evaluation_authority,
    unsupported_requirements,
):
    prepared, requirements, _invocation = task_evaluation_authority
    runtime, runner = _runtime(tmp_path, monkeypatch, prepared, requirements)
    provider = TaskEvaluationDockerExecutionProvider(
        dispatch_key=requirements.dispatch_key,
        provider_settings=prepared.plan_join.settings.task_evaluation_provider,
        policy_settings=prepared.plan_join.settings.policy,
        runtime=runtime,
    )

    with pytest.raises(
        TaskEvaluationDockerProviderError, match="not exactly realizable"
    ):
        provider.require_supported_execution(unsupported_requirements(requirements))

    assert runner.requests == []
