"""Request-derived registry bootstrap for concrete source replay execution."""

from __future__ import annotations

import fcntl
import os
import stat
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

from kapso.cross_run.expert.replay_docker_provider import (
    SourceReplayDockerExecutionProvider,
    require_source_replay_docker_provider_key,
    source_replay_docker_provider_key_is_supported,
)
from kapso.cross_run.expert.replay_docker_runtime import SourceReplayDockerRuntime
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayExecutionProviderKey,
    ExpertSourceReplayExecutionProviderRegistry,
    ExpertSourceReplayMatchedLegInvocation,
    ExpertSourceReplayProviderCompletion,
    SourceReplayProviderExecutionHandle,
    expert_source_replay_execution_provider_key,
)
from kapso.cross_run.expert.replay_request import PreparedExpertSourceReplayRequest
from kapso.cross_run.settings import (
    ExpertValidationPolicySettings,
    TaskEvaluationDockerProviderSettings,
)


class SourceReplayDockerBootstrapError(ValueError):
    """A prepared request cannot bind the concrete Docker implementation."""


class SourceReplayDockerProviderRegistry(ExpertSourceReplayExecutionProviderRegistry):
    """One exact prepared request bound to its lazy Docker implementations."""

    def __init__(
        self,
        *,
        prepared_request: PreparedExpertSourceReplayRequest,
        providers: tuple[_LazySourceReplayDockerExecutionProvider, ...],
    ) -> None:
        self._prepared_request = prepared_request
        super().__init__(providers)
        super().resolve_all(prepared_request)

    def resolve_all(
        self,
        prepared_request: PreparedExpertSourceReplayRequest,
    ):
        if prepared_request != self._prepared_request:
            raise SourceReplayDockerBootstrapError(
                "source replay Docker registry is bound to another prepared request"
            )
        return super().resolve_all(prepared_request)


def build_source_replay_docker_provider_registry(
    *,
    prepared_request: PreparedExpertSourceReplayRequest,
    workspace_root: Path,
) -> SourceReplayDockerProviderRegistry:
    """Resolve every key before initializing one shared lazy pinned runtime."""

    if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
        raise SourceReplayDockerBootstrapError(
            "source replay Docker bootstrap requires a prepared request"
        )
    settings = prepared_request.settings
    trusted_root = _configured_trusted_root(
        workspace_root,
        settings.task_evaluation_provider,
    )
    dispatch_keys = _distinct_supported_dispatch_keys(prepared_request)
    runtime_authority = _LazySourceReplayDockerRuntime(
        trusted_root=trusted_root,
        provider_settings=settings.task_evaluation_provider,
    )
    registry = SourceReplayDockerProviderRegistry(
        prepared_request=prepared_request,
        providers=tuple(
            _LazySourceReplayDockerExecutionProvider(
                dispatch_key=dispatch_key,
                provider_settings=settings.task_evaluation_provider,
                policy_settings=settings.policy,
                runtime_authority=runtime_authority,
            )
            for dispatch_key in dispatch_keys
        ),
    )
    _prepare_configured_trusted_root(
        workspace_root,
        trusted_root,
        settings.task_evaluation_provider,
    )
    return registry


def _distinct_supported_dispatch_keys(
    prepared_request: PreparedExpertSourceReplayRequest,
) -> tuple[ExpertSourceReplayExecutionProviderKey, ...]:
    keys_by_identity: dict[
        tuple[str, ...],
        ExpertSourceReplayExecutionProviderKey,
    ] = {}
    for materialized_case in prepared_request.cases:
        dispatch_key = expert_source_replay_execution_provider_key(materialized_case)
        if not source_replay_docker_provider_key_is_supported(
            dispatch_key,
            prepared_request.settings.task_evaluation_provider,
            prepared_request.settings.policy,
        ):
            raise SourceReplayDockerBootstrapError(
                "source replay Docker bootstrap encountered an unsupported key"
            )
        keys_by_identity[dispatch_key.identity] = dispatch_key
    return tuple(keys_by_identity[identity] for identity in sorted(keys_by_identity))


class _LazySourceReplayDockerRuntime:
    def __init__(
        self,
        *,
        trusted_root: Path,
        provider_settings: TaskEvaluationDockerProviderSettings,
    ) -> None:
        self.trusted_root = trusted_root
        self.provider_settings = provider_settings
        self._runtime = None
        self._lock = Lock()

    def get(self) -> SourceReplayDockerRuntime:
        with self._lock:
            if self._runtime is None:
                self._runtime = SourceReplayDockerRuntime.create(
                    trusted_root=self.trusted_root,
                    settings=self.provider_settings,
                )
            return self._runtime


class _LazySourceReplayDockerExecutionProvider:
    def __init__(
        self,
        *,
        dispatch_key: ExpertSourceReplayExecutionProviderKey,
        provider_settings: TaskEvaluationDockerProviderSettings,
        policy_settings: ExpertValidationPolicySettings,
        runtime_authority: _LazySourceReplayDockerRuntime,
    ) -> None:
        require_source_replay_docker_provider_key(
            dispatch_key,
            provider_settings,
            policy_settings,
        )
        self.dispatch_key = dispatch_key
        self._provider_settings = provider_settings
        self._policy_settings = policy_settings
        self._runtime_authority = runtime_authority

    def execute_leg(
        self,
        invocation: ExpertSourceReplayMatchedLegInvocation,
    ) -> ExpertSourceReplayProviderCompletion:
        return self._provider().execute_leg(invocation)

    def cleanup_interrupted(
        self,
        provider_handle: SourceReplayProviderExecutionHandle,
    ) -> None:
        self._provider().cleanup_interrupted(provider_handle)

    def _provider(self) -> SourceReplayDockerExecutionProvider:
        return SourceReplayDockerExecutionProvider(
            dispatch_key=self.dispatch_key,
            provider_settings=self._provider_settings,
            policy_settings=self._policy_settings,
            runtime=self._runtime_authority.get(),
        )


def _configured_trusted_root(
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
        raise SourceReplayDockerBootstrapError(
            "source replay Docker workspace root must be absolute and resolved"
        )
    trusted_root = workspace_root / provider_settings.workspace_path
    if (
        provider_settings.container_user_id != os.geteuid()
        or provider_settings.container_group_id != os.getegid()
        or "," in str(trusted_root)
    ):
        raise SourceReplayDockerBootstrapError(
            "source replay Docker host identity cannot realize the configured root"
        )
    metadata = workspace_root.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise SourceReplayDockerBootstrapError(
            "source replay Docker workspace root is unsafe"
        )
    if trusted_root.resolve() != trusted_root:
        raise SourceReplayDockerBootstrapError(
            "source replay Docker configured root contains a symlink"
        )
    return trusted_root


def _prepare_configured_trusted_root(
    workspace_root: Path,
    trusted_root: Path,
    provider_settings: TaskEvaluationDockerProviderSettings,
) -> None:
    if trusted_root != workspace_root / provider_settings.workspace_path:
        raise SourceReplayDockerBootstrapError(
            "source replay Docker configured root changed before initialization"
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
                raise SourceReplayDockerBootstrapError(
                    "source replay Docker configured hierarchy is unsafe"
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
