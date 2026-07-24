"""Request-bound registry bootstrap for concrete task evaluation."""

from __future__ import annotations

import fcntl
import os
import stat
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

from kapso.cross_run.docker.runtime import PinnedDockerRuntime
from kapso.cross_run.expert.task_evaluation_docker_provider import (
    TaskEvaluationDockerExecutionProvider,
    require_task_evaluation_docker_provider_support,
    task_evaluation_docker_provider_key_is_supported,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderKey,
    TaskEvaluationExecutionProviderRegistry,
    TaskEvaluationLegInvocation,
    TaskEvaluationProviderCompletion,
    TaskEvaluationProviderExecutionHandle,
    TaskEvaluationProviderSupportRequirements,
    project_prepared_task_evaluation_cases,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.settings import (
    DockerRuntimeSettings,
    ExpertValidationPolicySettings,
    TaskEvaluationDockerProviderSettings,
)


class TaskEvaluationDockerBootstrapError(ValueError):
    """A prepared task-evaluation request cannot bind the Docker provider."""


class TaskEvaluationDockerWorkspaceError(ValueError):
    """The configured Docker workspace cannot be trusted or initialized."""


def build_task_evaluation_docker_provider_registry(
    *,
    prepared_request: PreparedTaskEvaluationRequest,
    workspace_root: Path,
) -> TaskEvaluationExecutionProviderRegistry:
    """Resolve the complete request before initializing one lazy runtime."""

    if type(prepared_request) is not PreparedTaskEvaluationRequest:
        raise TaskEvaluationDockerBootstrapError(
            "task evaluation Docker bootstrap requires a prepared request"
        )
    settings = prepared_request.plan_join.settings
    trusted_root = configured_task_evaluation_docker_trusted_root(
        workspace_root,
        settings.task_evaluation_provider,
    )
    dispatch_keys = _distinct_supported_dispatch_keys(prepared_request)
    runtime_authority = TaskEvaluationDockerRuntimeAuthority(
        trusted_root=trusted_root,
        runtime_settings=settings.task_evaluation_provider.runtime,
    )
    registry = TaskEvaluationExecutionProviderRegistry(
        prepared_request,
        tuple(
            _LazyTaskEvaluationDockerExecutionProvider(
                dispatch_key=dispatch_key,
                provider_settings=settings.task_evaluation_provider,
                policy_settings=settings.policy,
                runtime_authority=runtime_authority,
            )
            for dispatch_key in dispatch_keys
        ),
    )
    prepare_task_evaluation_docker_trusted_root(
        workspace_root,
        trusted_root,
        settings.task_evaluation_provider,
    )
    return registry


def _distinct_supported_dispatch_keys(
    prepared_request: PreparedTaskEvaluationRequest,
) -> tuple[TaskEvaluationExecutionProviderKey, ...]:
    settings = prepared_request.plan_join.settings
    keys_by_identity: dict[tuple[str, ...], TaskEvaluationExecutionProviderKey] = {}
    for executable_case in project_prepared_task_evaluation_cases(prepared_request):
        dispatch_key = executable_case.provider_key
        if not task_evaluation_docker_provider_key_is_supported(
            dispatch_key,
            settings.task_evaluation_provider,
            settings.policy,
        ):
            raise TaskEvaluationDockerBootstrapError(
                "task evaluation Docker bootstrap encountered an unsupported key"
            )
        keys_by_identity[dispatch_key.identity] = dispatch_key
    return tuple(keys_by_identity[identity] for identity in sorted(keys_by_identity))


class TaskEvaluationDockerRuntimeAuthority:
    """Lazily initialize one runtime shared by an exact provider registry."""

    def __init__(
        self,
        *,
        trusted_root: Path,
        runtime_settings: DockerRuntimeSettings,
    ) -> None:
        self.trusted_root = trusted_root
        self.runtime_settings = runtime_settings
        self._runtime: PinnedDockerRuntime | None = None
        self._lock = Lock()

    def get(self) -> PinnedDockerRuntime:
        with self._lock:
            if self._runtime is None:
                self._runtime = PinnedDockerRuntime.create(
                    trusted_root=self.trusted_root,
                    settings=self.runtime_settings,
                )
            return self._runtime


class _LazyTaskEvaluationDockerExecutionProvider:
    def __init__(
        self,
        *,
        dispatch_key: TaskEvaluationExecutionProviderKey,
        provider_settings: TaskEvaluationDockerProviderSettings,
        policy_settings: ExpertValidationPolicySettings,
        runtime_authority: TaskEvaluationDockerRuntimeAuthority,
    ) -> None:
        if not task_evaluation_docker_provider_key_is_supported(
            dispatch_key,
            provider_settings,
            policy_settings,
        ):
            raise TaskEvaluationDockerBootstrapError(
                "task evaluation Docker provider key is unsupported"
            )
        self.dispatch_key = dispatch_key
        self._provider_settings = provider_settings
        self._policy_settings = policy_settings
        self._runtime_authority = runtime_authority

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
        return self._provider().execute_leg(invocation)

    def cleanup_interrupted(
        self,
        provider_handle: TaskEvaluationProviderExecutionHandle,
    ) -> None:
        self._provider().cleanup_interrupted(provider_handle)

    def _provider(self) -> TaskEvaluationDockerExecutionProvider:
        return TaskEvaluationDockerExecutionProvider(
            dispatch_key=self.dispatch_key,
            provider_settings=self._provider_settings,
            policy_settings=self._policy_settings,
            runtime=self._runtime_authority.get(),
        )


def configured_task_evaluation_docker_trusted_root(
    workspace_root: Path,
    provider_settings: TaskEvaluationDockerProviderSettings,
) -> Path:
    if (
        not isinstance(workspace_root, Path)
        or not workspace_root.is_absolute()
        or workspace_root != Path(os.path.abspath(workspace_root))
        or workspace_root.resolve() != workspace_root
        or workspace_root in {Path("/"), Path.home()}
    ):
        raise TaskEvaluationDockerWorkspaceError(
            "task evaluation Docker workspace root must be absolute and resolved"
        )
    trusted_root = workspace_root / provider_settings.workspace_path
    if (
        provider_settings.container_user_id != os.geteuid()
        or provider_settings.container_group_id != os.getegid()
        or "," in str(trusted_root)
    ):
        raise TaskEvaluationDockerWorkspaceError(
            "task evaluation Docker host identity cannot realize the configured root"
        )
    metadata = workspace_root.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise TaskEvaluationDockerWorkspaceError(
            "task evaluation Docker workspace root is unsafe"
        )
    if trusted_root.resolve() != trusted_root:
        raise TaskEvaluationDockerWorkspaceError(
            "task evaluation Docker configured root contains a symlink"
        )
    return trusted_root


def prepare_task_evaluation_docker_trusted_root(
    workspace_root: Path,
    trusted_root: Path,
    provider_settings: TaskEvaluationDockerProviderSettings,
) -> None:
    if trusted_root != workspace_root / provider_settings.workspace_path:
        raise TaskEvaluationDockerWorkspaceError(
            "task evaluation Docker configured root changed before initialization"
        )
    with ExitStack() as descriptors_to_close:
        descriptor = os.open(
            workspace_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors_to_close.callback(os.close, descriptor)
        bootstrap_descriptor = descriptor
        fcntl.flock(bootstrap_descriptor, fcntl.LOCK_EX)
        for part in Path(provider_settings.workspace_path).parts:
            exists = _configured_child_exists(descriptor, part)
            if not exists:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
            child_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            descriptors_to_close.callback(os.close, child_descriptor)
            if not exists:
                os.fchmod(child_descriptor, 0o700)
            metadata = os.fstat(child_descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise TaskEvaluationDockerWorkspaceError(
                    "task evaluation Docker configured hierarchy is unsafe"
                )
            descriptor = child_descriptor
        os.fsync(descriptor)
        fcntl.flock(bootstrap_descriptor, fcntl.LOCK_UN)


def _configured_child_exists(parent_descriptor: int, name: str) -> bool:
    return os.access(
        name,
        os.F_OK,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
