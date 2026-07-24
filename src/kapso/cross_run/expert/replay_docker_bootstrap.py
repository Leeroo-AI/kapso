"""Request-derived registry bootstrap for concrete source replay execution."""

from __future__ import annotations

from pathlib import Path

from kapso.cross_run.expert.replay_docker_provider import (
    SourceReplayDockerExecutionProvider,
    require_source_replay_docker_provider_support,
    require_source_replay_docker_provider_key,
    source_replay_docker_provider_key_is_supported,
)
from kapso.cross_run.expert.task_evaluation_docker_bootstrap import (
    TaskEvaluationDockerRuntimeAuthority,
    configured_task_evaluation_docker_trusted_root,
    prepare_task_evaluation_docker_trusted_root,
)
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
    trusted_root = configured_task_evaluation_docker_trusted_root(
        workspace_root,
        settings.task_evaluation_provider,
    )
    dispatch_keys = _distinct_supported_dispatch_keys(prepared_request)
    runtime_authority = TaskEvaluationDockerRuntimeAuthority(
        trusted_root=trusted_root,
        runtime_settings=settings.task_evaluation_provider.runtime,
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
    prepare_task_evaluation_docker_trusted_root(
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
        require_source_replay_docker_provider_support(
            materialized_case,
            dispatch_key,
            prepared_request.settings.task_evaluation_provider,
            prepared_request.settings.policy,
        )
        keys_by_identity[dispatch_key.identity] = dispatch_key
    return tuple(keys_by_identity[identity] for identity in sorted(keys_by_identity))


class _LazySourceReplayDockerExecutionProvider:
    def __init__(
        self,
        *,
        dispatch_key: ExpertSourceReplayExecutionProviderKey,
        provider_settings: TaskEvaluationDockerProviderSettings,
        policy_settings: ExpertValidationPolicySettings,
        runtime_authority: TaskEvaluationDockerRuntimeAuthority,
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
