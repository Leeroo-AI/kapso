"""Configuration-derived compute authority for task-evaluation cases."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import ExpertValidationStage
from kapso.cross_run.expert.promotion_contracts import ExpertReleaseMatrixMode
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationComputeBinding,
    TaskEvaluationLegKind,
)
from kapso.cross_run.settings import ExpertValidationSettings


class TaskEvaluationComputeError(ValueError):
    """Task-evaluation compute cannot be derived from exact configuration."""


def derive_release_matrix_compute_bindings(
    *,
    settings: ExpertValidationSettings,
    mode: ExpertReleaseMatrixMode,
    provenance_binding_ids: tuple[str, ...],
) -> Mapping[str, TaskEvaluationComputeBinding]:
    """Derive one exact configured compute envelope per adapter provenance."""

    if type(settings) is not ExpertValidationSettings:
        raise TaskEvaluationComputeError(
            "release matrix compute requires exact validation settings"
        )
    if type(mode) is not ExpertReleaseMatrixMode:
        raise TaskEvaluationComputeError(
            "release matrix compute requires an exact matrix mode"
        )
    ordered_provenance_ids = tuple(sorted(provenance_binding_ids))
    if not ordered_provenance_ids or len(ordered_provenance_ids) != len(
        set(ordered_provenance_ids)
    ):
        raise TaskEvaluationComputeError(
            "release matrix compute requires unique adapter provenances"
        )
    for provenance_id in ordered_provenance_ids:
        require_content_id(provenance_id, "release matrix compute provenance")
        if provenance_id.split(":sha256:", 1)[0] != (
            "expert-release-matrix-provenance-binding"
        ):
            raise TaskEvaluationComputeError(
                "release matrix compute provenance uses the wrong namespace"
            )
    evaluators = tuple(
        evaluator
        for evaluator in settings.policy.evaluators
        if evaluator.stage is ExpertValidationStage.RELEASE_MATRIX
    )
    if len(evaluators) != 1:
        raise TaskEvaluationComputeError(
            "release matrix compute requires one configured evaluator"
        )
    evaluator = evaluators[0]
    policy = settings.policy
    provider_settings_digest = tree_or_blob_digest(
        settings.task_evaluation_provider.to_json_bytes()
    )
    if mode is ExpertReleaseMatrixMode.BOOTSTRAP:
        schedules = {
            provenance_id: (TaskEvaluationLegKind.CANDIDATE,)
            for provenance_id in ordered_provenance_ids
        }
    else:
        parent_first = (
            TaskEvaluationLegKind.PARENT_CONTROL,
            TaskEvaluationLegKind.CANDIDATE,
        )
        candidate_first = tuple(reversed(parent_first))
        order_digest = tree_or_blob_digest(
            canonical_json_bytes(
                {
                    "execution_protocol_version": (
                        policy.task_evaluation_execution_protocol_version
                    ),
                    "provenance_binding_ids": ordered_provenance_ids,
                }
            )
        )
        starting_offset = int(order_digest[-1], 16) % 2
        schedules = {
            provenance_id: (
                parent_first
                if (position + starting_offset) % 2 == 0
                else candidate_first
            )
            for position, provenance_id in enumerate(ordered_provenance_ids)
        }
    return MappingProxyType(
        {
            provenance_id: TaskEvaluationComputeBinding.mint(
                execution_protocol_version=(
                    policy.task_evaluation_execution_protocol_version
                ),
                execution_provider_id=(policy.task_evaluation_execution_provider_id),
                execution_provider_version=(
                    policy.task_evaluation_execution_provider_version
                ),
                execution_provider_settings_digest=provider_settings_digest,
                sandbox_policy_version=policy.task_evaluation_sandbox_policy_version,
                leg_wall_time_limit_seconds=evaluator.timeout_seconds,
                termination_grace_seconds=(
                    policy.task_evaluation_termination_grace_seconds
                ),
                cpu_millicore_limit=policy.task_evaluation_cpu_millicore_limit,
                memory_byte_limit=policy.task_evaluation_memory_byte_limit,
                shared_memory_byte_limit=(
                    policy.task_evaluation_shared_memory_byte_limit
                ),
                process_limit=policy.task_evaluation_process_limit,
                open_file_limit=policy.task_evaluation_open_file_limit,
                writable_inode_limit=policy.task_evaluation_writable_inode_limit,
                writable_storage_byte_limit=(
                    policy.task_evaluation_writable_storage_byte_limit
                ),
                output_entry_limit=policy.artifact_entry_limit,
                output_byte_limit=policy.artifact_byte_limit,
                stdout_byte_limit=policy.task_evaluation_stdout_byte_limit,
                stderr_byte_limit=policy.task_evaluation_stderr_byte_limit,
                accelerator_class_id=policy.task_evaluation_accelerator_class_id,
                accelerator_count=policy.task_evaluation_accelerator_count,
                leg_order=schedules[provenance_id],
            )
            for provenance_id in ordered_provenance_ids
        }
    )
