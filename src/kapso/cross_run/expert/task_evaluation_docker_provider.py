"""Concrete isolated Docker provider for task-evaluation cases."""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.canonical import require_content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    SourceFileDescriptor,
    TaskAdapterRuntimeContract,
)
from kapso.cross_run.docker.runtime import (
    DockerImageAuthority,
    PinnedDockerRuntime,
    read_verified_root_executable,
)
from kapso.cross_run.expert.task_evaluation_docker_resources import (
    TASK_EVALUATION_DOCKER_CONTAINER_HOSTNAME,
    TaskEvaluationDockerContainerObservation,
    TaskEvaluationDockerResourceIdentity,
    TaskEvaluationDockerResourceManager,
    TaskEvaluationDockerVolumeObservation,
    task_evaluation_docker_container_observations_match,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderKey,
    TaskEvaluationLegInvocation,
    TaskEvaluationProviderCompletion,
    TaskEvaluationProviderExecutionHandle,
    TaskEvaluationProviderSupportRequirements,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationSourceBase,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TASK_ADAPTER_RUNTIME_PROTOCOL_VERSION,
    TASK_EVALUATOR_ADAPTER_ROOT,
    TASK_EVALUATOR_PROTOCOL_VERSION,
    TASK_EVALUATOR_WRITABLE_ROOT,
)
from kapso.cross_run.expert.task_evaluation_provider_filesystem import (
    TaskEvaluationProviderArtifactInput,
    cleanup_task_evaluation_provider_workspace,
    materialize_task_evaluation_provider_inputs,
    materialize_verified_byte_tree,
    parse_task_evaluation_result_snapshot,
)
from kapso.cross_run.process import BoundedProcessOutcome, BoundedProcessResult
from kapso.cross_run.settings import (
    ExpertValidationPolicySettings,
    TaskEvaluationDockerProviderSettings,
)

_PROVIDER_DIRECTORY_NAME = "provider"
_HELPER_FILENAME = "busybox"
_CONTAINER_HELPER_ROOT = "/kapso/provider"
_CONTAINER_HELPER_PATH = f"{_CONTAINER_HELPER_ROOT}/{_HELPER_FILENAME}"
_CONTAINER_INPUT_ROOT = "/kapso/input"
_CONTAINER_HOME = "/kapso/home"
_EVALUATOR_ROLE = "evaluator"
_KEEPER_ROLE = "keeper"
_PROVIDER_ENVIRONMENT = (
    ("HOME", _CONTAINER_HOME),
    ("HOSTNAME", TASK_EVALUATION_DOCKER_CONTAINER_HOSTNAME),
)
_SUPPORTED_IMAGE_PLATFORMS_BY_HOST = {
    ("linux", "x86_64"): frozenset({("linux", "amd64", None)}),
    ("linux", "aarch64"): frozenset(
        {
            ("linux", "arm64", None),
            ("linux", "arm64", "v8"),
        }
    ),
}
TASK_EVALUATION_DOCKER_EXECUTION_PROVIDER_ID = (
    "kapso_task_evaluation_execution_provider"
)
TASK_EVALUATION_DOCKER_EXECUTION_PROVIDER_VERSION = (
    "kapso.task_evaluation_execution_provider.v1"
)
TASK_EVALUATION_DOCKER_EXECUTION_PROTOCOL_VERSION = "kapso.task_evaluation_execution.v1"
TASK_EVALUATION_DOCKER_SANDBOX_POLICY_VERSION = (
    "kapso.task_evaluation_sandbox.offline_readonly.v1"
)


class TaskEvaluationDockerSandboxError(RuntimeError):
    """The neutral Docker sandbox violates its exact byte or runtime authority."""


class TaskEvaluationDockerProviderError(TaskEvaluationDockerSandboxError):
    """A task-evaluation leg violates its exact Docker sandbox authority."""


@dataclass(frozen=True)
class TaskEvaluationDockerSandboxCompute:
    leg_wall_time_limit_seconds: int
    termination_grace_seconds: int
    cpu_millicore_limit: int
    memory_byte_limit: int
    shared_memory_byte_limit: int
    process_limit: int
    open_file_limit: int
    writable_inode_limit: int
    writable_storage_byte_limit: int
    output_byte_limit: int
    stdout_byte_limit: int
    stderr_byte_limit: int

    def __post_init__(self) -> None:
        limits = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if (
            any(type(limit) is not int or limit <= 0 for limit in limits)
            or self.termination_grace_seconds > self.leg_wall_time_limit_seconds
            or self.shared_memory_byte_limit > self.memory_byte_limit
            or self.output_byte_limit > self.writable_storage_byte_limit
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker sandbox compute is invalid"
            )


@dataclass(frozen=True)
class TaskEvaluationDockerSandboxInvocation:
    provider_handle_id: str
    adapter_runtime: TaskAdapterRuntimeContract
    evaluator_relative_path: str
    compute: TaskEvaluationDockerSandboxCompute
    expert_source_files: tuple[SourceFileDescriptor, ...]
    expert_source_contents: Mapping[str, bytes]
    adapter_source_files: tuple[SourceFileDescriptor, ...]
    adapter_source_contents: Mapping[str, bytes]
    task_artifacts: tuple[TaskEvaluationProviderArtifactInput, ...]
    request_payload: bytes

    def __post_init__(self) -> None:
        require_content_id(
            self.provider_handle_id,
            "task evaluation Docker sandbox provider handle",
        )
        evaluator_path = PurePosixPath(self.evaluator_relative_path)
        if (
            type(self.adapter_runtime) is not TaskAdapterRuntimeContract
            or not isinstance(self.evaluator_relative_path, str)
            or evaluator_path.is_absolute()
            or not evaluator_path.parts
            or ".." in evaluator_path.parts
            or str(evaluator_path) != self.evaluator_relative_path
            or type(self.compute) is not TaskEvaluationDockerSandboxCompute
            or type(self.expert_source_files) is not tuple
            or type(self.adapter_source_files) is not tuple
            or any(
                type(descriptor) is not SourceFileDescriptor
                for descriptor in self.expert_source_files + self.adapter_source_files
            )
            or not isinstance(self.expert_source_contents, Mapping)
            or not isinstance(self.adapter_source_contents, Mapping)
            or type(self.task_artifacts) is not tuple
            or any(
                type(artifact) is not TaskEvaluationProviderArtifactInput
                for artifact in self.task_artifacts
            )
            or not isinstance(self.request_payload, bytes)
            or not self.request_payload
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker sandbox invocation is invalid"
            )
        object.__setattr__(
            self,
            "expert_source_contents",
            MappingProxyType(dict(self.expert_source_contents)),
        )
        object.__setattr__(
            self,
            "adapter_source_contents",
            MappingProxyType(dict(self.adapter_source_contents)),
        )


@dataclass(frozen=True)
class TaskEvaluationDockerSandboxCompletion:
    process_result: BoundedProcessResult
    result_payload: bytes | None

    def __post_init__(self) -> None:
        if type(self.process_result) is not BoundedProcessResult or (
            self.result_payload is not None
            and not isinstance(self.result_payload, bytes)
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker sandbox completion is invalid"
            )


def task_evaluation_docker_provider_key_is_supported(
    dispatch_key: TaskEvaluationExecutionProviderKey,
    provider_settings: TaskEvaluationDockerProviderSettings,
    policy_settings: ExpertValidationPolicySettings,
) -> bool:
    """Return whether a key and policy name this exact implementation."""

    return not (
        type(dispatch_key) is not TaskEvaluationExecutionProviderKey
        or type(provider_settings) is not TaskEvaluationDockerProviderSettings
        or type(policy_settings) is not ExpertValidationPolicySettings
        or dispatch_key.execution_protocol_version
        != TASK_EVALUATION_DOCKER_EXECUTION_PROTOCOL_VERSION
        or dispatch_key.execution_provider_id
        != TASK_EVALUATION_DOCKER_EXECUTION_PROVIDER_ID
        or dispatch_key.execution_provider_version
        != TASK_EVALUATION_DOCKER_EXECUTION_PROVIDER_VERSION
        or dispatch_key.execution_provider_settings_digest
        != tree_or_blob_digest(provider_settings.to_json_bytes())
        or dispatch_key.sandbox_policy_version
        != TASK_EVALUATION_DOCKER_SANDBOX_POLICY_VERSION
        or dispatch_key.task_adapter_runtime_protocol_version
        != TASK_ADAPTER_RUNTIME_PROTOCOL_VERSION
        or dispatch_key.task_evaluator_protocol_version
        != TASK_EVALUATOR_PROTOCOL_VERSION
        or policy_settings.task_evaluation_execution_protocol_version
        != TASK_EVALUATION_DOCKER_EXECUTION_PROTOCOL_VERSION
        or policy_settings.task_evaluation_execution_provider_id
        != TASK_EVALUATION_DOCKER_EXECUTION_PROVIDER_ID
        or policy_settings.task_evaluation_execution_provider_version
        != TASK_EVALUATION_DOCKER_EXECUTION_PROVIDER_VERSION
        or policy_settings.task_evaluation_sandbox_policy_version
        != TASK_EVALUATION_DOCKER_SANDBOX_POLICY_VERSION
    )


def require_task_evaluation_docker_provider_key(
    dispatch_key: TaskEvaluationExecutionProviderKey,
    provider_settings: TaskEvaluationDockerProviderSettings,
    policy_settings: ExpertValidationPolicySettings,
) -> None:
    """Require the complete implementation-owned Docker dispatch authority."""

    if not task_evaluation_docker_provider_key_is_supported(
        dispatch_key,
        provider_settings,
        policy_settings,
    ):
        raise TaskEvaluationDockerProviderError(
            "task evaluation Docker provider key differs from implementation authority"
        )


def require_task_evaluation_docker_provider_support(
    requirements: TaskEvaluationProviderSupportRequirements,
    dispatch_key: TaskEvaluationExecutionProviderKey,
    provider_settings: TaskEvaluationDockerProviderSettings,
    policy_settings: ExpertValidationPolicySettings,
) -> None:
    """Reject a case that this exact CPU-only Docker implementation cannot run."""

    require_task_evaluation_docker_provider_key(
        dispatch_key,
        provider_settings,
        policy_settings,
    )
    if (
        type(requirements) is not TaskEvaluationProviderSupportRequirements
        or requirements.dispatch_key != dispatch_key
    ):
        raise TaskEvaluationDockerProviderError(
            "task evaluation Docker support differs from its dispatch authority"
        )
    compute = _task_evaluation_sandbox_compute(requirements)
    if not task_evaluation_docker_sandbox_support_is_exact(
        adapter_runtime=requirements.runtime_contract,
        compute=compute,
        accelerator_class_id=requirements.accelerator_class_id,
        accelerator_count=requirements.accelerator_count,
        provider_settings=provider_settings,
    ):
        raise TaskEvaluationDockerProviderError(
            "task evaluation Docker support requirements are not exactly realizable"
        )


def task_evaluation_docker_sandbox_support_is_exact(
    *,
    adapter_runtime: TaskAdapterRuntimeContract,
    compute: TaskEvaluationDockerSandboxCompute,
    accelerator_class_id: str | None,
    accelerator_count: int,
    provider_settings: TaskEvaluationDockerProviderSettings,
) -> bool:
    """Return whether the neutral CPU-only sandbox can realize this exact leg."""

    if (
        type(adapter_runtime) is not TaskAdapterRuntimeContract
        or type(compute) is not TaskEvaluationDockerSandboxCompute
        or type(accelerator_count) is not int
        or type(provider_settings) is not TaskEvaluationDockerProviderSettings
    ):
        return False
    environment = dict(adapter_runtime.environment)
    runtime_settings = provider_settings.runtime
    cpu_quota_numerator = (
        compute.cpu_millicore_limit * provider_settings.cpu_period_microseconds
    )
    supported_image_platforms = _SUPPORTED_IMAGE_PLATFORMS_BY_HOST.get(
        (
            runtime_settings.runtime_host_operating_system,
            runtime_settings.runtime_host_architecture,
        )
    )
    if (
        supported_image_platforms is None
        or (
            adapter_runtime.operating_system,
            adapter_runtime.architecture,
            adapter_runtime.architecture_variant,
        )
        not in supported_image_platforms
        or accelerator_count != 0
        or accelerator_class_id is not None
        or not environment.get("PATH")
        or compute.termination_grace_seconds >= runtime_settings.command_timeout_seconds
        or cpu_quota_numerator % 1000 != 0
        or cpu_quota_numerator // 1000 <= 0
    ):
        return False
    return True


def task_adapter_docker_image_authority(
    runtime: TaskAdapterRuntimeContract,
) -> DockerImageAuthority:
    """Project one task-adapter image into the neutral Docker authority."""

    if type(runtime) is not TaskAdapterRuntimeContract:
        raise TaskEvaluationDockerSandboxError(
            "task evaluation Docker image projection requires exact adapter runtime"
        )
    return DockerImageAuthority.mint(
        image_reference=runtime.image_reference,
        image_config_digest=runtime.image_config_digest,
        operating_system=runtime.operating_system,
        architecture=runtime.architecture,
        architecture_variant=runtime.architecture_variant,
    )


class TaskEvaluationDockerSandbox:
    """Run one provenance-erased evaluator closure in a fresh Docker sandbox."""

    def __init__(
        self,
        *,
        provider_settings: TaskEvaluationDockerProviderSettings,
        policy_settings: ExpertValidationPolicySettings,
        runtime: PinnedDockerRuntime,
    ) -> None:
        if (
            type(provider_settings) is not TaskEvaluationDockerProviderSettings
            or type(policy_settings) is not ExpertValidationPolicySettings
            or type(runtime) is not PinnedDockerRuntime
            or runtime.settings is not provider_settings.runtime
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker sandbox authorities are not exact"
            )
        if (
            provider_settings.container_user_id != os.geteuid()
            or provider_settings.container_group_id != os.getegid()
            or "," in str(runtime.trusted_root)
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker host identity cannot realize the sandbox"
            )
        if (
            policy_settings.task_evaluation_termination_grace_seconds
            >= runtime.settings.command_timeout_seconds
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker command timeout cannot contain graceful stop"
            )
        self._provider_settings = provider_settings
        self._policy_settings = policy_settings
        self._runtime = runtime
        self._resources = TaskEvaluationDockerResourceManager(
            runtime,
            provider_settings,
        )

    def _execute_sandbox(
        self,
        invocation: TaskEvaluationDockerSandboxInvocation,
    ) -> TaskEvaluationDockerSandboxCompletion:
        if type(invocation) is not TaskEvaluationDockerSandboxInvocation:
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker sandbox requires an exact invocation"
            )
        compute = invocation.compute
        adapter_runtime = invocation.adapter_runtime
        image = self._runtime.inspect_exact_image(
            task_adapter_docker_image_authority(adapter_runtime)
        )
        _require_task_adapter_image_policy(image, adapter_runtime)
        identity = self._resources.require_absent(invocation.provider_handle_id)

        with ExitStack() as ownership:
            ownership.callback(
                self._cleanup_owned_execution,
                invocation.provider_handle_id,
                identity,
            )
            layout = materialize_task_evaluation_provider_inputs(
                expert_source_files=invocation.expert_source_files,
                expert_source_contents=invocation.expert_source_contents,
                adapter_source_files=invocation.adapter_source_files,
                adapter_source_contents=invocation.adapter_source_contents,
                task_artifacts=invocation.task_artifacts,
                request_payload=invocation.request_payload,
                trusted_root=self._runtime.trusted_root,
                workspace_root=identity.workspace_root,
            )
            helper_root = self._materialize_helper(identity)
            volume = self._resources.create_writable_volume(
                identity,
                writable_storage_byte_limit=compute.writable_storage_byte_limit,
                writable_inode_limit=compute.writable_inode_limit,
            )
            keeper = self._create_keeper(
                identity,
                adapter_runtime,
                helper_root,
                volume,
                compute,
            )
            keeper = self._start_keeper(
                identity,
                keeper,
                volume,
                compute,
                adapter_runtime,
                helper_root,
            )
            evaluator = self._create_evaluator(
                identity,
                adapter_runtime,
                layout.input_root,
                volume,
                compute,
                invocation.evaluator_relative_path,
            )
            process_result = self._runtime.run_bounded(
                (
                    "container",
                    "start",
                    "--attach",
                    evaluator.container_id,
                ),
                timeout_seconds=compute.leg_wall_time_limit_seconds,
                cleanup_timeout_seconds=compute.termination_grace_seconds,
                stdout_byte_limit=compute.stdout_byte_limit,
                stderr_byte_limit=compute.stderr_byte_limit,
            )
            result_payload = self._finish_evaluator(
                identity,
                evaluator,
                keeper,
                volume,
                compute,
                process_result,
            )
            completion = TaskEvaluationDockerSandboxCompletion(
                process_result=process_result,
                result_payload=result_payload,
            )
        return completion

    def _cleanup_sandbox(self, provider_handle_id: str) -> None:
        identity = self._resources.cleanup_daemon_resources(provider_handle_id)
        cleanup_task_evaluation_provider_workspace(
            trusted_root=self._runtime.trusted_root,
            workspace_root=identity.workspace_root,
        )

    def _materialize_helper(
        self,
        identity: TaskEvaluationDockerResourceIdentity,
    ) -> Path:
        helper_bytes = read_verified_root_executable(
            Path(self._provider_settings.helper_executable_path),
            self._provider_settings.helper_executable_digest,
        )
        helper_root = identity.workspace_root / _PROVIDER_DIRECTORY_NAME
        materialize_verified_byte_tree(
            trusted_root=identity.workspace_root,
            destination_root=helper_root,
            descriptors=(
                SourceFileDescriptor(
                    relative_path=_HELPER_FILENAME,
                    digest=self._provider_settings.helper_executable_digest,
                    mode="100755",
                    size=len(helper_bytes),
                ),
            ),
            source_contents={_HELPER_FILENAME: helper_bytes},
        )
        return helper_root

    def _create_keeper(
        self,
        identity: TaskEvaluationDockerResourceIdentity,
        adapter_runtime: TaskAdapterRuntimeContract,
        helper_root: Path,
        volume: TaskEvaluationDockerVolumeObservation,
        compute: TaskEvaluationDockerSandboxCompute,
    ) -> TaskEvaluationDockerContainerObservation:
        arguments = (
            *_container_create_prefix(
                identity=identity,
                role=_KEEPER_ROLE,
                compute=compute,
                provider_settings=self._provider_settings,
                workdir="/",
            ),
            *_container_environment_arguments(adapter_runtime),
            "--mount",
            _bind_mount(helper_root, _CONTAINER_HELPER_ROOT),
            "--mount",
            _volume_mount(volume.name, readonly=True),
            "--entrypoint",
            _CONTAINER_HELPER_PATH,
            adapter_runtime.image_reference,
            "tail",
            "-f",
            "/dev/null",
        )
        result = self._runtime.run_control(arguments)
        container_id = _parse_created_container_id(result.stdout)
        evaluator, keeper, observed_volume = self._resources.observe(identity)
        if (
            evaluator is not None
            or keeper is None
            or keeper.container_id != container_id
            or observed_volume != volume
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation keeper creation changed its owned resources"
            )
        _require_container_contract(
            keeper,
            identity,
            adapter_runtime,
            compute,
            self._provider_settings,
            entrypoint=_CONTAINER_HELPER_PATH,
            command=("tail", "-f", "/dev/null"),
            workdir="/",
            expected_mounts=(
                _expected_bind_mount(helper_root, _CONTAINER_HELPER_ROOT),
                _expected_volume_mount(volume.name, readonly=True),
            ),
            expected_status="created",
            expected_exit_code=0,
        )
        return keeper

    def _start_keeper(
        self,
        identity: TaskEvaluationDockerResourceIdentity,
        keeper: TaskEvaluationDockerContainerObservation,
        volume: TaskEvaluationDockerVolumeObservation,
        compute: TaskEvaluationDockerSandboxCompute,
        adapter_runtime: TaskAdapterRuntimeContract,
        helper_root: Path,
    ) -> TaskEvaluationDockerContainerObservation:
        result = self._runtime.run_control(("container", "start", keeper.container_id))
        _require_exact_line(result.stdout, keeper.container_id)
        evaluator, running_keeper, observed_volume = self._resources.observe(identity)
        if (
            evaluator is not None
            or running_keeper is None
            or running_keeper.container_id != keeper.container_id
            or observed_volume != volume
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation keeper changed while starting"
            )
        _require_container_contract(
            running_keeper,
            identity,
            adapter_runtime,
            compute,
            self._provider_settings,
            entrypoint=_CONTAINER_HELPER_PATH,
            command=("tail", "-f", "/dev/null"),
            workdir="/",
            expected_mounts=(
                _expected_bind_mount(helper_root, _CONTAINER_HELPER_ROOT),
                _expected_volume_mount(volume.name, readonly=True),
            ),
            expected_status="running",
            expected_exit_code=0,
        )
        return running_keeper

    def _create_evaluator(
        self,
        identity: TaskEvaluationDockerResourceIdentity,
        adapter_runtime: TaskAdapterRuntimeContract,
        input_root: Path,
        volume: TaskEvaluationDockerVolumeObservation,
        compute: TaskEvaluationDockerSandboxCompute,
        evaluator_relative_path: str,
    ) -> TaskEvaluationDockerContainerObservation:
        evaluator_path = f"{TASK_EVALUATOR_ADAPTER_ROOT}/{evaluator_relative_path}"
        arguments = (
            *_container_create_prefix(
                identity=identity,
                role=_EVALUATOR_ROLE,
                compute=compute,
                provider_settings=self._provider_settings,
                workdir=TASK_EVALUATOR_ADAPTER_ROOT,
            ),
            *_container_environment_arguments(adapter_runtime),
            "--mount",
            _bind_mount(input_root, _CONTAINER_INPUT_ROOT),
            "--mount",
            _volume_mount(volume.name, readonly=False),
            "--entrypoint",
            evaluator_path,
            adapter_runtime.image_reference,
        )
        result = self._runtime.run_control(arguments)
        container_id = _parse_created_container_id(result.stdout)
        evaluator, keeper, observed_volume = self._resources.observe(identity)
        if (
            evaluator is None
            or evaluator.container_id != container_id
            or keeper is None
            or observed_volume != volume
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation evaluator creation changed its owned resources"
            )
        _require_container_contract(
            evaluator,
            identity,
            adapter_runtime,
            compute,
            self._provider_settings,
            entrypoint=evaluator_path,
            command=(),
            workdir=TASK_EVALUATOR_ADAPTER_ROOT,
            expected_mounts=(
                _expected_bind_mount(input_root, _CONTAINER_INPUT_ROOT),
                _expected_volume_mount(volume.name, readonly=False),
            ),
            expected_status="created",
            expected_exit_code=0,
        )
        return evaluator

    def _finish_evaluator(
        self,
        identity: TaskEvaluationDockerResourceIdentity,
        evaluator: TaskEvaluationDockerContainerObservation,
        keeper: TaskEvaluationDockerContainerObservation,
        volume: TaskEvaluationDockerVolumeObservation,
        compute: TaskEvaluationDockerSandboxCompute,
        process_result: BoundedProcessResult,
    ) -> bytes | None:
        evaluator_after, keeper_after, volume_after = self._resources.observe(identity)
        if (
            evaluator_after is None
            or evaluator_after.container_id != evaluator.container_id
            or not _container_execution_authority_matches_created(
                evaluator_after,
                evaluator,
            )
            or keeper_after is None
            or not task_evaluation_docker_container_observations_match(
                keeper_after,
                keeper,
            )
            or volume_after != volume
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation owned resources changed after evaluator execution"
            )
        if process_result.outcome is BoundedProcessOutcome.COMPLETED:
            _require_stopped_evaluator(
                evaluator_after,
                process_result.returncode,
                require_no_oom=process_result.returncode == 0,
            )
        else:
            evaluator_after = self._resources.stop_container(
                identity,
                evaluator_after,
                compute.termination_grace_seconds,
            )
            _require_stopped_evaluator_without_exit_authority(evaluator_after)
        self._resources.remove_container(identity, evaluator_after)
        if (
            process_result.outcome is not BoundedProcessOutcome.COMPLETED
            or process_result.returncode != 0
        ):
            return None
        self._runtime.require_live_authority()
        evaluator_absent, current_keeper, current_volume = self._resources.observe(
            identity
        )
        if (
            evaluator_absent is not None
            or not task_evaluation_docker_container_observations_match(
                current_keeper,
                keeper_after,
            )
            or current_volume != volume
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation result authority changed before snapshot"
            )
        maximum_result_bytes = min(
            compute.output_byte_limit,
            self._policy_settings.task_evaluation_result_byte_limit,
        )
        maximum_snapshot_bytes = (
            maximum_result_bytes
            + self._provider_settings.result_archive_overhead_byte_limit
        )
        snapshot = self._runtime.run_bounded(
            (
                "container",
                "exec",
                current_keeper.container_id,
                _CONTAINER_HELPER_PATH,
                "tar",
                "-C",
                TASK_EVALUATOR_WRITABLE_ROOT,
                "-cf",
                "-",
                ".",
            ),
            timeout_seconds=self._runtime.settings.command_timeout_seconds,
            cleanup_timeout_seconds=self._runtime.settings.cleanup_timeout_seconds,
            stdout_byte_limit=maximum_snapshot_bytes,
            stderr_byte_limit=self._runtime.settings.command_output_byte_limit,
        )
        if (
            snapshot.outcome is not BoundedProcessOutcome.COMPLETED
            or snapshot.returncode != 0
            or snapshot.stderr
        ):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation result snapshot command failed"
            )
        return parse_task_evaluation_result_snapshot(
            snapshot.stdout,
            expected_owner_id=self._provider_settings.container_user_id,
            expected_group_id=self._provider_settings.container_group_id,
            maximum_result_bytes=maximum_result_bytes,
            maximum_snapshot_bytes=maximum_snapshot_bytes,
        )

    def _cleanup_owned_execution(
        self,
        provider_handle_id: str,
        identity: TaskEvaluationDockerResourceIdentity,
    ) -> None:
        cleaned_identity = self._resources.cleanup_daemon_resources(provider_handle_id)
        if cleaned_identity != identity:
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker cleanup changed resource identity"
            )
        cleanup_task_evaluation_provider_workspace(
            trusted_root=self._runtime.trusted_root,
            workspace_root=identity.workspace_root,
        )


class TaskEvaluationDockerExecutionProvider(TaskEvaluationDockerSandbox):
    """Project one sealed task-evaluation leg into the neutral Docker sandbox."""

    def __init__(
        self,
        *,
        dispatch_key: TaskEvaluationExecutionProviderKey,
        provider_settings: TaskEvaluationDockerProviderSettings,
        policy_settings: ExpertValidationPolicySettings,
        runtime: PinnedDockerRuntime,
    ) -> None:
        require_task_evaluation_docker_provider_key(
            dispatch_key,
            provider_settings,
            policy_settings,
        )
        super().__init__(
            provider_settings=provider_settings,
            policy_settings=policy_settings,
            runtime=runtime,
        )
        self.dispatch_key = dispatch_key

    def require_supported_execution(
        self,
        requirements: TaskEvaluationProviderSupportRequirements,
    ) -> None:
        require_task_evaluation_docker_provider_support(
            requirements,
            self.dispatch_key,
            self._provider_settings,
            self._policy_settings,
        )

    def execute_leg(
        self,
        invocation: TaskEvaluationLegInvocation,
    ) -> TaskEvaluationProviderCompletion:
        if (
            type(invocation) is not TaskEvaluationLegInvocation
            or invocation.provider_handle.dispatch_key != self.dispatch_key
            or invocation.provider_handle.invocation_allocation
            != invocation.invocation_allocation
            or invocation.execution_requirements.dispatch_key != self.dispatch_key
        ):
            raise TaskEvaluationDockerProviderError(
                "task evaluation Docker invocation differs from provider authority"
            )
        requirements = invocation.execution_requirements
        self.require_supported_execution(requirements)
        adapter_runtime = invocation.adapter_runtime.manifest.runtime
        if (
            requirements.runtime_contract != adapter_runtime
            or requirements.task_evaluator_executable_path
            != invocation.adapter_runtime.manifest.task_evaluator.executable_path
        ):
            raise TaskEvaluationDockerProviderError(
                "task evaluation Docker invocation differs from supported runtime"
            )
        requested_mounts = {
            mount.starting_artifact_ref: mount.mount_path
            for mount in invocation.task_evaluator_request.starting_artifact_mounts
        }
        observed_mounts = {
            artifact.artifact.starting_artifact_ref: artifact.artifact.mount_path
            for artifact in invocation.starting_artifacts
        }
        if requested_mounts != observed_mounts:
            raise TaskEvaluationDockerProviderError(
                "task evaluation Docker artifacts differ from the evaluator request"
            )
        expert_source_files, expert_source_contents = _expert_source_closure(invocation)
        sandbox_completion = self._execute_sandbox(
            TaskEvaluationDockerSandboxInvocation(
                provider_handle_id=invocation.provider_handle.provider_handle_id,
                adapter_runtime=adapter_runtime,
                evaluator_relative_path=requirements.task_evaluator_executable_path,
                compute=_task_evaluation_sandbox_compute(requirements),
                expert_source_files=expert_source_files,
                expert_source_contents=expert_source_contents,
                adapter_source_files=invocation.adapter_runtime.source_files,
                adapter_source_contents=invocation.adapter_runtime.source_contents,
                task_artifacts=tuple(
                    sorted(
                        (
                            TaskEvaluationProviderArtifactInput(
                                mount_path=artifact.artifact.mount_path,
                                source_files=artifact.artifact.source_files,
                                source_contents=artifact.source_contents,
                            )
                            for artifact in invocation.starting_artifacts
                        ),
                        key=lambda artifact: artifact.mount_path,
                    )
                ),
                request_payload=invocation.task_evaluator_request.to_json_bytes(),
            )
        )
        return TaskEvaluationProviderCompletion(
            provider_handle_id=invocation.provider_handle.provider_handle_id,
            process_result=sandbox_completion.process_result,
            result_payload=sandbox_completion.result_payload,
        )

    def cleanup_interrupted(
        self,
        provider_handle: TaskEvaluationProviderExecutionHandle,
    ) -> None:
        if (
            type(provider_handle) is not TaskEvaluationProviderExecutionHandle
            or provider_handle.dispatch_key != self.dispatch_key
        ):
            raise TaskEvaluationDockerProviderError(
                "task evaluation Docker cleanup differs from provider authority"
            )
        self._cleanup_sandbox(provider_handle.provider_handle_id)


def _task_evaluation_sandbox_compute(
    requirements: TaskEvaluationProviderSupportRequirements,
) -> TaskEvaluationDockerSandboxCompute:
    return TaskEvaluationDockerSandboxCompute(
        leg_wall_time_limit_seconds=requirements.leg_wall_time_limit_seconds,
        termination_grace_seconds=requirements.termination_grace_seconds,
        cpu_millicore_limit=requirements.cpu_millicore_limit,
        memory_byte_limit=requirements.memory_byte_limit,
        shared_memory_byte_limit=requirements.shared_memory_byte_limit,
        process_limit=requirements.process_limit,
        open_file_limit=requirements.open_file_limit,
        writable_inode_limit=requirements.writable_inode_limit,
        writable_storage_byte_limit=requirements.writable_storage_byte_limit,
        output_byte_limit=requirements.output_byte_limit,
        stdout_byte_limit=requirements.stdout_byte_limit,
        stderr_byte_limit=requirements.stderr_byte_limit,
    )


def _expert_source_closure(
    invocation: TaskEvaluationLegInvocation,
) -> tuple[tuple[SourceFileDescriptor, ...], Mapping[str, bytes]]:
    source = invocation.selected_leg.expert_source
    if type(source) is VerifiedTaskEvaluationCandidate:
        return source.source_tree.files, source.source_contents
    if type(source) is VerifiedTaskEvaluationSourceBase:
        return (
            source.source_base_tree_receipt.source_extraction_receipt.source_tree_files,
            source.source_contents,
        )
    raise TaskEvaluationDockerProviderError(
        "task evaluation Docker invocation has no exact expert source"
    )


def _container_create_prefix(
    *,
    identity: TaskEvaluationDockerResourceIdentity,
    role: str,
    compute: TaskEvaluationDockerSandboxCompute,
    provider_settings: TaskEvaluationDockerProviderSettings,
    workdir: str,
) -> tuple[str, ...]:
    name = identity.evaluator_name if role == _EVALUATOR_ROLE else identity.keeper_name
    labels = identity.labels_for(role)
    label_arguments = tuple(
        argument
        for key, value in sorted(labels.items())
        for argument in ("--label", f"{key}={value}")
    )
    cpu_quota = (
        compute.cpu_millicore_limit * provider_settings.cpu_period_microseconds // 1000
    )
    return (
        "container",
        "create",
        "--name",
        name,
        *label_arguments,
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--security-opt",
        "seccomp=builtin",
        "--cgroupns",
        "private",
        "--ipc",
        "private",
        "--pids-limit",
        str(compute.process_limit),
        "--memory",
        str(compute.memory_byte_limit),
        "--memory-swap",
        str(compute.memory_byte_limit),
        "--oom-kill-disable=false",
        "--cpu-period",
        str(provider_settings.cpu_period_microseconds),
        "--cpu-quota",
        str(cpu_quota),
        "--shm-size",
        str(compute.shared_memory_byte_limit),
        "--ulimit",
        f"nofile={compute.open_file_limit}:{compute.open_file_limit}",
        "--restart",
        "no",
        "--log-driver",
        "none",
        "--hostname",
        TASK_EVALUATION_DOCKER_CONTAINER_HOSTNAME,
        "--user",
        f"{provider_settings.container_user_id}:{provider_settings.container_group_id}",
        "--workdir",
        workdir,
        "--runtime",
        provider_settings.runtime.runtime_default_runtime,
        "--stop-timeout",
        str(compute.termination_grace_seconds),
    )


def _bind_mount(source: Path, destination: str) -> str:
    return (
        f"type=bind,src={source},dst={destination},readonly,"
        "bind-recursive=disabled,bind-propagation=rprivate"
    )


def _container_environment(
    runtime: TaskAdapterRuntimeContract,
) -> tuple[tuple[str, str], ...]:
    environment = dict(runtime.environment)
    environment.update(_PROVIDER_ENVIRONMENT)
    return tuple(sorted(environment.items()))


def _container_environment_arguments(
    runtime: TaskAdapterRuntimeContract,
) -> tuple[str, ...]:
    return tuple(
        argument
        for key, value in _container_environment(runtime)
        for argument in ("--env", f"{key}={value}")
    )


def _volume_mount(name: str, *, readonly: bool) -> str:
    read_option = ",readonly" if readonly else ""
    return (
        f"type=volume,src={name},dst={TASK_EVALUATOR_WRITABLE_ROOT}"
        f"{read_option},volume-nocopy"
    )


def _expected_bind_mount(source: Path, destination: str) -> dict[str, Any]:
    return {
        "Destination": destination,
        "Propagation": "rprivate",
        "RW": False,
        "Source": str(source),
        "Type": "bind",
    }


def _expected_volume_mount(name: str, *, readonly: bool) -> dict[str, Any]:
    return {
        "Destination": TASK_EVALUATOR_WRITABLE_ROOT,
        "Driver": "local",
        "Name": name,
        "Propagation": "",
        "RW": not readonly,
        "Type": "volume",
    }


def _require_container_contract(
    observation: TaskEvaluationDockerContainerObservation,
    identity: TaskEvaluationDockerResourceIdentity,
    runtime: TaskAdapterRuntimeContract,
    compute: TaskEvaluationDockerSandboxCompute,
    provider_settings: TaskEvaluationDockerProviderSettings,
    *,
    entrypoint: str,
    command: tuple[str, ...],
    workdir: str,
    expected_mounts: tuple[Mapping[str, Any], ...],
    expected_status: str,
    expected_exit_code: int,
) -> None:
    payload = observation.payload
    config = _require_mapping(payload, "Config", "task evaluation container config")
    host = _require_mapping(
        payload, "HostConfig", "task evaluation container host config"
    )
    state = _require_mapping(payload, "State", "task evaluation container state")
    expected_name = (
        identity.evaluator_name
        if observation.role == _EVALUATOR_ROLE
        else identity.keeper_name
    )
    expected_command = list(command) if command else None
    expected_quota = (
        compute.cpu_millicore_limit * provider_settings.cpu_period_microseconds // 1000
    )
    expected_oom_kill_disable = False if expected_status == "created" else None
    if (
        observation.name != expected_name
        or payload.get("Image") != runtime.image_config_digest
        or payload.get("Path") != entrypoint
        or payload.get("Args") != list(command)
        or payload.get("RestartCount") != 0
        or config.get("Image") != runtime.image_reference
        or config.get("Hostname") != TASK_EVALUATION_DOCKER_CONTAINER_HOSTNAME
        or config.get("User")
        != (
            f"{provider_settings.container_user_id}:"
            f"{provider_settings.container_group_id}"
        )
        or not _container_environment_is_exact(config.get("Env"), runtime)
        or config.get("Entrypoint") != [entrypoint]
        or config.get("Cmd") != expected_command
        or config.get("WorkingDir") != workdir
        or config.get("Labels") != dict(identity.labels_for(observation.role))
        or config.get("StopTimeout") != compute.termination_grace_seconds
        or host.get("AutoRemove") is not False
        or host.get("Binds") is not None
        or host.get("CapAdd") is not None
        or host.get("CapDrop") != ["ALL"]
        or host.get("Cgroup") != ""
        or host.get("CgroupnsMode") != "private"
        or host.get("CpuPeriod") != provider_settings.cpu_period_microseconds
        or host.get("CpuQuota") != expected_quota
        or host.get("Devices") != []
        or host.get("DeviceRequests") is not None
        or host.get("DeviceCgroupRules") is not None
        or host.get("Dns") is not None
        or host.get("DnsOptions") != []
        or host.get("DnsSearch") != []
        or host.get("ExtraHosts") is not None
        or host.get("GroupAdd") is not None
        or host.get("IpcMode") != "private"
        or host.get("Links") is not None
        or host.get("LogConfig") != {"Type": "none", "Config": {}}
        or host.get("Memory") != compute.memory_byte_limit
        or host.get("MemorySwap") != compute.memory_byte_limit
        or host.get("NetworkMode") != "none"
        or host.get("OomKillDisable") is not expected_oom_kill_disable
        or host.get("PidMode") != ""
        or host.get("PidsLimit") != compute.process_limit
        or host.get("PortBindings") != {}
        or host.get("Privileged") is not False
        or host.get("PublishAllPorts") is not False
        or host.get("ReadonlyRootfs") is not True
        or host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0}
        or host.get("Runtime") != provider_settings.runtime.runtime_default_runtime
        or host.get("SecurityOpt") != ["no-new-privileges", "seccomp=builtin"]
        or host.get("ShmSize") != compute.shared_memory_byte_limit
        or host.get("UTSMode") != ""
        or host.get("UsernsMode") != ""
        or host.get("VolumeDriver") != ""
        or host.get("VolumesFrom") is not None
        or host.get("Ulimits")
        != [
            {
                "Name": "nofile",
                "Hard": compute.open_file_limit,
                "Soft": compute.open_file_limit,
            }
        ]
        or tuple(_normalized_mounts(payload)) != expected_mounts
        or not _container_has_exact_isolated_network(
            payload.get("NetworkSettings"),
            expected_status=expected_status,
        )
        or state.get("Status") != expected_status
        or state.get("Running") is not (expected_status == "running")
        or state.get("Paused") is not False
        or state.get("Restarting") is not False
        or state.get("OOMKilled") is not False
        or state.get("Dead") is not False
        or type(state.get("Pid")) is not int
        or (state.get("Pid") > 0) is not (expected_status == "running")
        or state.get("ExitCode") != expected_exit_code
        or state.get("Error") != ""
    ):
        raise TaskEvaluationDockerSandboxError(
            "task evaluation Docker container differs from exact sandbox authority"
        )


def _container_environment_is_exact(
    value: Any,
    runtime: TaskAdapterRuntimeContract,
) -> bool:
    if not isinstance(value, list) or any(
        not isinstance(assignment, str) or "=" not in assignment for assignment in value
    ):
        return False
    environment: dict[str, str] = {}
    for assignment in value:
        key, assigned_value = assignment.split("=", 1)
        if not key or key in environment:
            return False
        environment[key] = assigned_value
    return environment == dict(_container_environment(runtime))


def _require_task_adapter_image_policy(
    image: Mapping[str, Any],
    runtime: TaskAdapterRuntimeContract,
) -> None:
    config = _require_mapping(image, "Config", "task adapter image config")
    environment = config.get("Env")
    command = config.get("Cmd")
    volumes = config.get("Volumes")
    expected_environment = [
        f"{key}={value}" for key, value in sorted(runtime.environment.items())
    ]
    if (
        environment != expected_environment
        or (command is not None and command != [])
        or config.get("Entrypoint") is not None
        or (volumes is not None and volumes != {})
        or config.get("Healthcheck") is not None
    ):
        raise TaskEvaluationDockerSandboxError(
            "task adapter image violates the task evaluation sandbox policy"
        )


def _container_execution_authority_matches_created(
    current: TaskEvaluationDockerContainerObservation,
    created: TaskEvaluationDockerContainerObservation,
) -> bool:
    current_host = current.payload.get("HostConfig")
    created_host = created.payload.get("HostConfig")
    current_state = current.payload.get("State")
    if (
        not isinstance(current_host, Mapping)
        or not isinstance(created_host, Mapping)
        or not isinstance(current_state, Mapping)
        or current_host.get("OomKillDisable") is not None
        or created_host.get("OomKillDisable") is not False
        or not _container_has_exact_isolated_network(
            current.payload.get("NetworkSettings"),
            expected_status=current_state.get("Status"),
        )
    ):
        return False
    normalized_current_host = dict(current_host)
    normalized_current_host["OomKillDisable"] = False
    authority_fields = (
        "Args",
        "Config",
        "HostConfig",
        "Image",
        "Mounts",
        "Path",
        "RestartCount",
    )
    normalized_current_payload = {
        field: current.payload.get(field) for field in authority_fields
    }
    normalized_current_payload["HostConfig"] = normalized_current_host
    created_authority_payload = {
        field: created.payload.get(field) for field in authority_fields
    }
    return task_evaluation_docker_container_observations_match(
        TaskEvaluationDockerContainerObservation(
            container_id=current.container_id,
            name=current.name,
            role=current.role,
            payload=normalized_current_payload,
        ),
        TaskEvaluationDockerContainerObservation(
            container_id=created.container_id,
            name=created.name,
            role=created.role,
            payload=created_authority_payload,
        ),
    )


def _container_has_exact_isolated_network(
    value: Any,
    *,
    expected_status: Any,
) -> bool:
    if expected_status not in {"created", "running", "exited"}:
        return False
    if not isinstance(value, dict):
        return False
    networks = value.get("Networks")
    if not isinstance(networks, dict) or set(networks) != {"none"}:
        return False
    network = networks["none"]
    if not isinstance(network, dict):
        return False
    sandbox_id = value.get("SandboxID")
    sandbox_key = value.get("SandboxKey")
    network_id = network.get("NetworkID")
    endpoint_id = network.get("EndpointID")
    if expected_status == "created":
        if (
            sandbox_id != ""
            or sandbox_key != ""
            or network_id != ""
            or endpoint_id != ""
        ):
            return False
    elif expected_status == "running":
        if (
            not _is_docker_identifier(sandbox_id)
            or sandbox_key != f"/var/run/docker/netns/{sandbox_id[:12]}"
            or not _is_docker_identifier(network_id)
            or not _is_docker_identifier(endpoint_id)
        ):
            return False
    elif (
        sandbox_id != ""
        or sandbox_key != ""
        or not _is_docker_identifier(network_id)
        or endpoint_id != ""
    ):
        return False
    return value == {
        "SandboxID": sandbox_id,
        "SandboxKey": sandbox_key,
        "Ports": {},
        "Networks": {
            "none": {
                "IPAMConfig": None,
                "Links": None,
                "Aliases": None,
                "DriverOpts": None,
                "GwPriority": 0,
                "NetworkID": network_id,
                "EndpointID": endpoint_id,
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
    }


def _is_docker_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalized_mounts(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    mounts = payload.get("Mounts")
    if not isinstance(mounts, list):
        raise TaskEvaluationDockerSandboxError(
            "task evaluation Docker container mounts are not a list"
        )
    normalized = []
    for mount in mounts:
        if not isinstance(mount, dict):
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker container mount is not an object"
            )
        if mount.get("Type") == "bind":
            normalized.append(
                {
                    "Destination": mount.get("Destination"),
                    "Propagation": mount.get("Propagation"),
                    "RW": mount.get("RW"),
                    "Source": mount.get("Source"),
                    "Type": "bind",
                }
            )
        elif mount.get("Type") == "volume":
            normalized.append(
                {
                    "Destination": mount.get("Destination"),
                    "Driver": mount.get("Driver"),
                    "Name": mount.get("Name"),
                    "Propagation": mount.get("Propagation"),
                    "RW": mount.get("RW"),
                    "Type": "volume",
                }
            )
        else:
            raise TaskEvaluationDockerSandboxError(
                "task evaluation Docker container has an unsupported mount"
            )
    return tuple(sorted(normalized, key=lambda mount: mount["Destination"]))


def _require_stopped_evaluator(
    observation: TaskEvaluationDockerContainerObservation,
    expected_exit_code: int,
    *,
    require_no_oom: bool,
) -> None:
    state = _require_mapping(
        observation.payload,
        "State",
        "task evaluation stopped evaluator state",
    )
    if (
        observation.payload.get("RestartCount") != 0
        or state.get("Status") != "exited"
        or state.get("Running") is not False
        or state.get("Paused") is not False
        or state.get("Restarting") is not False
        or type(state.get("OOMKilled")) is not bool
        or (require_no_oom and state.get("OOMKilled") is not False)
        or state.get("Dead") is not False
        or state.get("Pid") != 0
        or state.get("ExitCode") != expected_exit_code
        or state.get("Error") != ""
    ):
        raise TaskEvaluationDockerSandboxError(
            "task evaluation evaluator did not stop with its exact process result"
        )


def _require_stopped_evaluator_without_exit_authority(
    observation: TaskEvaluationDockerContainerObservation,
) -> None:
    state = _require_mapping(
        observation.payload,
        "State",
        "task evaluation terminated evaluator state",
    )
    if (
        observation.payload.get("RestartCount") != 0
        or state.get("Status") != "exited"
        or state.get("Running") is not False
        or state.get("Paused") is not False
        or state.get("Restarting") is not False
        or type(state.get("OOMKilled")) is not bool
        or state.get("Dead") is not False
        or state.get("Pid") != 0
        or type(state.get("ExitCode")) is not int
        or state.get("Error") != ""
    ):
        raise TaskEvaluationDockerSandboxError(
            "task evaluation evaluator did not stop after its bounded outcome"
        )


def _parse_created_container_id(payload: bytes) -> str:
    if not isinstance(payload, bytes) or len(payload) != 65 or payload[-1:] != b"\n":
        raise TaskEvaluationDockerSandboxError(
            "task evaluation Docker create returned an invalid container identity"
        )
    container_id = payload[:-1].decode("ascii")
    if len(container_id) != 64 or any(
        character not in "0123456789abcdef" for character in container_id
    ):
        raise TaskEvaluationDockerSandboxError(
            "task evaluation Docker create returned an invalid container identity"
        )
    return container_id


def _require_exact_line(payload: bytes, expected: str) -> None:
    if payload != f"{expected}\n".encode():
        raise TaskEvaluationDockerSandboxError(
            "task evaluation Docker command returned an unexpected identity"
        )


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
    name: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TaskEvaluationDockerSandboxError(f"{name} is not an object")
    return value
