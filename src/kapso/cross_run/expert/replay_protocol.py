"""Projection from verified replay authority into the blinded evaluator ABI."""

from kapso.cross_run.expert.replay_protocol_contracts import (
    ExpertSourceReplayProtocolError,
    ExpertSourceReplayInvocationAllocation,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TASK_EVALUATOR_PROTOCOL_VERSION,
    TaskEvaluatorRequest,
    TaskEvaluatorStartingArtifactMount,
)
from kapso.cross_run.expert.replay_request import MaterializedExpertSourceReplayCase


def build_task_evaluator_request(
    materialized_case: MaterializedExpertSourceReplayCase,
    invocation_allocation: ExpertSourceReplayInvocationAllocation,
) -> TaskEvaluatorRequest:
    """Build the common blinded request for one isolated evaluator spawn."""

    if not isinstance(materialized_case, MaterializedExpertSourceReplayCase):
        raise ExpertSourceReplayProtocolError(
            "task evaluator request requires a materialized replay case"
        )
    if not isinstance(invocation_allocation, ExpertSourceReplayInvocationAllocation):
        raise ExpertSourceReplayProtocolError(
            "task evaluator request requires a journal invocation allocation"
        )
    request_case = materialized_case.request_case
    if (
        invocation_allocation.execution_case_id != request_case.execution_case_id
        or invocation_allocation.execution_leg_id
        not in {
            request_case.control_leg.execution_leg_id,
            request_case.candidate_leg.execution_leg_id,
        }
    ):
        raise ExpertSourceReplayProtocolError(
            "task evaluator invocation allocation names another execution leg"
        )
    adapter = materialized_case.task_adapter.manifest
    if adapter.task_evaluator.protocol_version != TASK_EVALUATOR_PROTOCOL_VERSION:
        raise ExpertSourceReplayProtocolError(
            "task evaluator manifest protocol is unsupported"
        )
    terminal_attempt = materialized_case.episode.attempts[
        materialized_case.episode.terminal_attempt_revision
    ]
    context = materialized_case.episode.task_context_binding
    consumed_dimension_ids = adapter.context_binding.consumed_dimension_ids
    return TaskEvaluatorRequest(
        protocol_version=TASK_EVALUATOR_PROTOCOL_VERSION,
        opaque_invocation_id=invocation_allocation.opaque_invocation_id,
        input_contract_fingerprint=context.input_contract_fingerprint,
        target_contract_fingerprint=context.target_contract_fingerprint,
        evaluation_fingerprints=terminal_attempt.evaluation_fingerprints,
        context_dimensions={
            dimension_id: context.transfer_dimensions[dimension_id]
            for dimension_id in consumed_dimension_ids
        },
        starting_artifact_mounts=tuple(
            sorted(
                (
                    TaskEvaluatorStartingArtifactMount(
                        starting_artifact_ref=(artifact.artifact.starting_artifact_ref),
                        mount_path=artifact.artifact.mount_path,
                    )
                    for artifact in materialized_case.task_context.starting_artifacts
                ),
                key=lambda mount: mount.starting_artifact_ref,
            )
        ),
    )
