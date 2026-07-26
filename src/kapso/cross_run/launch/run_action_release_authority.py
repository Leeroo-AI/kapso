"""Join workload release receipts to the actual typed durable event 5."""

from __future__ import annotations

from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionResolvedWorkloadObservation,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionReleaseAuthorizationObservation,
    RunActionReleaseContractError,
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionActivationRevalidationReceipt,
)


def mint_run_action_workload_release_receipt(
    *,
    activation_event: RunActionExecutionEvent,
    resolved_workload_observation: RunActionResolvedWorkloadObservation,
    release_authorization_observation: RunActionReleaseAuthorizationObservation,
) -> RunActionWorkloadReleaseReceipt:
    """Mint only after joining the full resolved graph to typed event 5."""

    _require_durable_activation_graph(
        activation_event,
        resolved_workload_observation,
    )
    receipt = RunActionWorkloadReleaseReceipt.mint(
        activation_event_id=activation_event.event_id,
        resolved_workload_observation=resolved_workload_observation,
        release_authorization_observation=release_authorization_observation,
    )
    require_run_action_workload_release_receipt_matches_event(
        receipt,
        activation_event,
    )
    return receipt


def require_run_action_workload_release_receipt_matches_event(
    receipt: RunActionWorkloadReleaseReceipt,
    activation_event: RunActionExecutionEvent,
) -> None:
    """Fail unless one parsed receipt is authority for this exact event 5."""

    if type(receipt) is not RunActionWorkloadReleaseReceipt:
        raise RunActionReleaseContractError(
            "workload release authority requires an exact receipt"
        )
    _require_durable_activation_graph(
        activation_event,
        receipt.resolved_workload_observation,
    )
    if receipt.activation_event_id != activation_event.event_id:
        raise RunActionReleaseContractError(
            "workload release receipt identifies another activation event"
        )


def _require_durable_activation_graph(
    activation_event: RunActionExecutionEvent,
    resolved_workload_observation: RunActionResolvedWorkloadObservation,
) -> None:
    if (
        type(activation_event) is not RunActionExecutionEvent
        or activation_event.event_number != 5
        or activation_event.event_kind
        is not RunActionExecutionEventKind.ACTIVATION_COMMITTED
        or type(activation_event.activation_revalidation_receipt)
        is not RunActionActivationRevalidationReceipt
        or type(resolved_workload_observation)
        is not RunActionResolvedWorkloadObservation
        or activation_event.activation_revalidation_receipt
        != resolved_workload_observation.activation_revalidation_receipt
    ):
        raise RunActionReleaseContractError(
            "workload release does not join the actual typed event 5"
        )


__all__ = [
    "mint_run_action_workload_release_receipt",
    "require_run_action_workload_release_receipt_matches_event",
]
