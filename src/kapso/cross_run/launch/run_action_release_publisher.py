"""Trusted construction and irreversible publication of one workload release."""

from __future__ import annotations

import os
from contextlib import ExitStack

from kapso.cross_run.launch.run_action_atomic_publication import (
    open_run_action_anonymous_file,
)
from kapso.cross_run.launch.run_action_control_candidate import (
    _CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY,
    _RunActionControlFileTransition,
    _RunActionFrozenControlFileCandidate,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
    RunActionCommittedContinuationCapability,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_resolved_workload import (
    _RELEASE_PUBLICATION_AUTHORITY,
    RunActionBlockedWorkloadLease,
)

_ANONYMOUS_FILE_MODE = 0o600


class RunActionReleasePublicationError(RuntimeError):
    """A release candidate or exact live publication authority is unsafe."""


def publish_run_action_workload_release_once(
    *,
    capability: RunActionCommittedContinuationCapability,
    blocked_workload_lease: RunActionBlockedWorkloadLease,
) -> RunActionWorkloadReleaseReceipt | None:
    """Freeze and no-replace link the sole coordinator-authorized receipt."""

    if (
        type(capability) is not RunActionCommittedContinuationCapability
        or type(blocked_workload_lease) is not RunActionBlockedWorkloadLease
    ):
        raise RunActionReleasePublicationError(
            "workload release publication requires exact live authorities"
        )
    blocked_workload_lease.require_current()
    resolved = blocked_workload_lease.resolved_workload_observation
    prepared = resolved.activation_revalidation_receipt.prepared_execution
    policy = prepared.preparation_claim.execution_policy
    control = prepared.control_directory
    with capability._begin_release_publication(
        resolved,
        _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
    ) as authorization:
        receipt = authorization._mint_receipt(
            _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
        )
        receipt_size_bound_bytes = authorization._receipt_size_bound(
            _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
        )
        with ExitStack() as descriptors:
            control_descriptor = (
                blocked_workload_lease._duplicate_release_control_descriptor(
                    _authority=_RELEASE_PUBLICATION_AUTHORITY,
                )
            )
            descriptors.callback(os.close, control_descriptor)
            anonymous_descriptor = open_run_action_anonymous_file(
                control_descriptor,
                _ANONYMOUS_FILE_MODE,
            )
            descriptors.callback(os.close, anonymous_descriptor)
            candidate = _RunActionFrozenControlFileCandidate(
                transition=_RunActionControlFileTransition.RELEASE,
                control_directory_descriptor=control_descriptor,
                anonymous_file_descriptor=anonymous_descriptor,
                predecessor_file_descriptor=None,
                owner_user_id=control.owner_user_id,
                owner_group_id=control.owner_group_id,
                payload_size_limit_bytes=receipt_size_bound_bytes,
                process_snapshot_size_limit_bytes=(
                    policy.supervisor_limits.process_snapshot_size_bytes
                ),
                payload=receipt.to_json_bytes(),
                _authority=_CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY,
            )
            descriptors.callback(candidate.close)
            blocked_workload_lease.require_current()
            return authorization._authorize_frozen_release_once(
                candidate=candidate,
                _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
            )


__all__ = [
    "RunActionReleasePublicationError",
    "publish_run_action_workload_release_once",
]
