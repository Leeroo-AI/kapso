"""Source-replay authority adapter for the neutral task-evaluation Docker sandbox."""

from __future__ import annotations

from typing import Mapping

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertSourceReplayComputeBinding,
    SourceFileDescriptor,
)
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayExecutionProviderKey,
    ExpertSourceReplayMatchedLegInvocation,
    ExpertSourceReplayProviderCompletion,
    SourceReplayProviderExecutionHandle,
    expert_source_replay_execution_provider_key,
)
from kapso.cross_run.expert.replay_request import MaterializedExpertSourceReplayCase
from kapso.cross_run.expert.task_evaluation_docker_provider import (
    TASK_EVALUATION_DOCKER_EXECUTION_PROTOCOL_VERSION,
    TASK_EVALUATION_DOCKER_EXECUTION_PROVIDER_ID,
    TASK_EVALUATION_DOCKER_EXECUTION_PROVIDER_VERSION,
    TASK_EVALUATION_DOCKER_SANDBOX_POLICY_VERSION,
    TaskEvaluationDockerSandbox,
    TaskEvaluationDockerSandboxCompute,
    TaskEvaluationDockerSandboxInvocation,
    task_evaluation_docker_sandbox_support_is_exact,
)
from kapso.cross_run.expert.task_evaluation_docker_runtime import (
    TaskEvaluationDockerRuntime,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationSourceBase,
)
from kapso.cross_run.expert.task_evaluation_provider_filesystem import (
    TaskEvaluationProviderArtifactInput,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TASK_ADAPTER_RUNTIME_PROTOCOL_VERSION,
    TASK_EVALUATOR_PROTOCOL_VERSION,
)
from kapso.cross_run.settings import (
    ExpertValidationPolicySettings,
    TaskEvaluationDockerProviderSettings,
)


class SourceReplayDockerProviderError(RuntimeError):
    """Source-replay authority cannot project into the neutral Docker sandbox."""


def source_replay_docker_provider_key_is_supported(
    dispatch_key: ExpertSourceReplayExecutionProviderKey,
    provider_settings: TaskEvaluationDockerProviderSettings,
    policy_settings: ExpertValidationPolicySettings,
) -> bool:
    """Return whether a source key names the one task-evaluation implementation."""

    return not (
        type(dispatch_key) is not ExpertSourceReplayExecutionProviderKey
        or type(provider_settings) is not TaskEvaluationDockerProviderSettings
        or type(policy_settings) is not ExpertValidationPolicySettings
        or dispatch_key.paired_execution_protocol_version
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


def require_source_replay_docker_provider_key(
    dispatch_key: ExpertSourceReplayExecutionProviderKey,
    provider_settings: TaskEvaluationDockerProviderSettings,
    policy_settings: ExpertValidationPolicySettings,
) -> None:
    """Require the complete source projection of the concrete implementation."""

    if not source_replay_docker_provider_key_is_supported(
        dispatch_key,
        provider_settings,
        policy_settings,
    ):
        raise SourceReplayDockerProviderError(
            "source replay Docker provider key differs from implementation authority"
        )


def require_source_replay_docker_provider_support(
    materialized_case: MaterializedExpertSourceReplayCase,
    dispatch_key: ExpertSourceReplayExecutionProviderKey,
    provider_settings: TaskEvaluationDockerProviderSettings,
    policy_settings: ExpertValidationPolicySettings,
) -> None:
    """Require that the neutral sandbox can realize this exact replay case."""

    require_source_replay_docker_provider_key(
        dispatch_key,
        provider_settings,
        policy_settings,
    )
    if (
        type(materialized_case) is not MaterializedExpertSourceReplayCase
        or expert_source_replay_execution_provider_key(materialized_case)
        != dispatch_key
    ):
        raise SourceReplayDockerProviderError(
            "source replay Docker support differs from its dispatch authority"
        )
    compute = materialized_case.request_case.compute_binding
    if not task_evaluation_docker_sandbox_support_is_exact(
        adapter_runtime=materialized_case.task_adapter.manifest.runtime,
        compute=_source_replay_sandbox_compute(compute),
        accelerator_class_id=compute.accelerator_class_id,
        accelerator_count=compute.accelerator_count,
        provider_settings=provider_settings,
    ):
        raise SourceReplayDockerProviderError(
            "source replay Docker support requirements are not exactly realizable"
        )


class SourceReplayDockerExecutionProvider(TaskEvaluationDockerSandbox):
    """Project one sealed source-replay leg into the neutral Docker sandbox."""

    def __init__(
        self,
        *,
        dispatch_key: ExpertSourceReplayExecutionProviderKey,
        provider_settings: TaskEvaluationDockerProviderSettings,
        policy_settings: ExpertValidationPolicySettings,
        runtime: TaskEvaluationDockerRuntime,
    ) -> None:
        require_source_replay_docker_provider_key(
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

    def execute_leg(
        self,
        invocation: ExpertSourceReplayMatchedLegInvocation,
    ) -> ExpertSourceReplayProviderCompletion:
        if (
            type(invocation) is not ExpertSourceReplayMatchedLegInvocation
            or invocation.provider_handle.dispatch_key != self.dispatch_key
            or expert_source_replay_execution_provider_key(invocation.materialized_case)
            != self.dispatch_key
        ):
            raise SourceReplayDockerProviderError(
                "source replay Docker invocation differs from provider authority"
            )
        require_source_replay_docker_provider_support(
            invocation.materialized_case,
            self.dispatch_key,
            self._provider_settings,
            self._policy_settings,
        )
        compute = invocation.materialized_case.request_case.compute_binding
        adapter = invocation.materialized_case.task_adapter
        requested_mounts = {
            mount.starting_artifact_ref: mount.mount_path
            for mount in invocation.task_evaluator_request.starting_artifact_mounts
        }
        observed_mounts = {
            artifact.artifact.starting_artifact_ref: artifact.artifact.mount_path
            for artifact in invocation.materialized_case.task_context.starting_artifacts
        }
        if requested_mounts != observed_mounts:
            raise SourceReplayDockerProviderError(
                "source replay task artifacts differ from the evaluator request"
            )
        expert_source_files, expert_source_contents = _expert_source_closure(invocation)
        sandbox_completion = self._execute_sandbox(
            TaskEvaluationDockerSandboxInvocation(
                provider_handle_id=invocation.provider_handle.provider_handle_id,
                adapter_runtime=adapter.manifest.runtime,
                evaluator_relative_path=adapter.manifest.task_evaluator.executable_path,
                compute=_source_replay_sandbox_compute(compute),
                expert_source_files=expert_source_files,
                expert_source_contents=expert_source_contents,
                adapter_source_files=adapter.evaluation_runtime_source_files,
                adapter_source_contents=adapter.evaluation_runtime_source_contents,
                task_artifacts=tuple(
                    sorted(
                        (
                            TaskEvaluationProviderArtifactInput(
                                mount_path=artifact.artifact.mount_path,
                                source_files=artifact.artifact.source_files,
                                source_contents=artifact.source_contents,
                            )
                            for artifact in invocation.materialized_case.task_context.starting_artifacts
                        ),
                        key=lambda artifact: artifact.mount_path,
                    )
                ),
                request_payload=invocation.task_evaluator_request.to_json_bytes(),
            )
        )
        return ExpertSourceReplayProviderCompletion(
            provider_handle_id=invocation.provider_handle.provider_handle_id,
            process_result=sandbox_completion.process_result,
            result_payload=sandbox_completion.result_payload,
        )

    def cleanup_interrupted(
        self,
        provider_handle: SourceReplayProviderExecutionHandle,
    ) -> None:
        if (
            type(provider_handle) is not SourceReplayProviderExecutionHandle
            or provider_handle.dispatch_key != self.dispatch_key
        ):
            raise SourceReplayDockerProviderError(
                "source replay Docker cleanup differs from provider authority"
            )
        self._cleanup_sandbox(provider_handle.provider_handle_id)


def _source_replay_sandbox_compute(
    compute: ExpertSourceReplayComputeBinding,
) -> TaskEvaluationDockerSandboxCompute:
    return TaskEvaluationDockerSandboxCompute(
        leg_wall_time_limit_seconds=compute.leg_wall_time_limit_seconds,
        termination_grace_seconds=compute.termination_grace_seconds,
        cpu_millicore_limit=compute.cpu_millicore_limit,
        memory_byte_limit=compute.memory_byte_limit,
        shared_memory_byte_limit=compute.shared_memory_byte_limit,
        process_limit=compute.process_limit,
        open_file_limit=compute.open_file_limit,
        writable_inode_limit=compute.writable_inode_limit,
        writable_storage_byte_limit=compute.writable_storage_byte_limit,
        output_byte_limit=compute.output_byte_limit,
        stdout_byte_limit=compute.stdout_byte_limit,
        stderr_byte_limit=compute.stderr_byte_limit,
    )


def _expert_source_closure(
    invocation: ExpertSourceReplayMatchedLegInvocation,
) -> tuple[tuple[SourceFileDescriptor, ...], Mapping[str, bytes]]:
    source = invocation.expert_source
    if type(source) is VerifiedTaskEvaluationCandidate:
        return source.source_tree.files, source.source_contents
    if type(source) is VerifiedTaskEvaluationSourceBase:
        return (
            source.source_base_tree_receipt.source_extraction_receipt.source_tree_files,
            source.source_contents,
        )
    raise SourceReplayDockerProviderError(
        "source replay invocation contains an unverified expert source"
    )
