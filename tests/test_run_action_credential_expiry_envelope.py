"""Formal wire bound for irreversible credential-expiry termination."""

from __future__ import annotations

import pytest

import kapso.cross_run.launch.run_action_credential_expiry_envelope as envelope_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.launch.run_action_credential_contracts import (
    maximum_credential_retirement_intent,
)
from kapso.cross_run.launch.run_action_credential_expiry_envelope import (
    RunActionCredentialExpiryEnvelopeError,
    credential_expiry_termination_event_size_bound,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationAllocation,
    RunActionTerminalObservation,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from test_run_action_supervisor_contracts import (
    _activation_revalidation_receipt,
    _prepared_execution,
    _spawn_commit,
    _terminal_observation,
)
from test_run_action_termination_contracts import (
    _pre_release_loss,
    _pre_release_terminal,
)


def _expiry_event(*, terminal_branch: bool) -> RunActionExecutionEvent:
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    activation = _activation_revalidation_receipt(prepared, spawn)
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_authority=prepared.runtime_volume_authority,
    )
    intent = maximum_credential_retirement_intent(prepared, spawn)
    terminal = (
        _pre_release_terminal(
            activation,
            intent.activation_event_id,
            _terminal_observation(prepared, spawn),
        )
        if terminal_branch
        else None
    )
    loss = (
        None
        if terminal_branch
        else _pre_release_loss(
            activation,
            intent.activation_event_id,
        )
    )
    receipt = RunActionProviderTerminationReceipt.mint(
        disposition=RunActionProviderTerminationDisposition.INTERRUPTED,
        reason=RunActionProviderTerminationReason.CREDENTIAL_EXPIRED,
        activation_event_id=intent.activation_event_id,
        workload_release_adoption=None,
        terminal_observation=terminal,
        timeout_directive_publication=None,
        pre_release_main_loss_observation=loss,
        credential_retirement_intent=intent,
    )
    return RunActionExecutionEvent.mint(
        event_number=7,
        predecessor_event_id=content_id(
            RunActionExecutionEvent.CONTENT_NAMESPACE,
            {"fixture": "credential expiry event 6"},
        ),
        event_kind=RunActionExecutionEventKind.PROVIDER_TERMINATED,
        reservation=allocation.preparation_claim.reservation,
        preparation_allocation=None,
        prepared_execution=None,
        spawn_commit=None,
        activation_revalidation_receipt=None,
        credential_retirement_intent=None,
        provider_termination_receipt=receipt,
        result_receipt=None,
        result_decision=None,
        acceptance=None,
        workspace_after=None,
    )


def test_expiry_envelope_bounds_both_committable_physical_branches() -> None:
    loss_event = _expiry_event(terminal_branch=False)
    terminal_event = _expiry_event(terminal_branch=True)
    prepared = (
        terminal_event.provider_termination_receipt.terminal_observation.activation_revalidation_receipt.prepared_execution
    )
    spawn = (
        terminal_event.provider_termination_receipt.terminal_observation.activation_revalidation_receipt.spawn_commit
    )
    allocation = (
        terminal_event.provider_termination_receipt.terminal_observation.preparation_allocation
    )

    bound = credential_expiry_termination_event_size_bound(
        reservation=terminal_event.reservation,
        preparation_allocation=allocation,
        prepared_execution=prepared,
        spawn_commit=spawn,
    )

    assert len(loss_event.to_json_bytes()) <= bound
    assert len(terminal_event.to_json_bytes()) <= bound
    assert bound > 0


def test_expiry_envelope_schema_guard_fails_loud() -> None:
    with pytest.raises(
        RunActionCredentialExpiryEnvelopeError,
        match="envelope fields changed",
    ):
        envelope_module._sealed_wire(
            RunActionTerminalObservation,
            terminal_observation_id="missing-all-other-fields",
        )


def test_expiry_envelope_rejects_spliced_prepared_graph() -> None:
    prepared = _prepared_execution()
    spawn = _spawn_commit(prepared)
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_authority=prepared.runtime_volume_authority,
    )
    foreign_prepared = _prepared_execution(inode_offset=1)

    with pytest.raises(
        RunActionCredentialExpiryEnvelopeError,
        match="exact prepared spawn",
    ):
        credential_expiry_termination_event_size_bound(
            reservation=allocation.preparation_claim.reservation,
            preparation_allocation=allocation,
            prepared_execution=foreign_prepared,
            spawn_commit=spawn,
        )
