"""Durable non-secret contracts for pre-release credential retirement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import content_id, require_content_id
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_credential_broker import (
    RunActionCredentialLeaseStatus,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionActivationRevalidationReceipt,
    RunActionCredentialMode,
    RunActionPreparedExecution,
    run_action_credential_lease_authority_id,
    run_action_credential_lease_request,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit

_NANOSECONDS_PER_SECOND = 1_000_000_000


class RunActionCredentialContractError(ValueError):
    """Credential status or retirement evidence is malformed or spliced."""


class RunActionPreReleaseCredentialState(str, Enum):
    """Closed lease state before workload release."""

    VALID = "valid"
    RENEWAL_PENDING = "renewal_pending"
    EXPIRED = "expired"


@dataclass(frozen=True)
class RunActionPreReleaseCredentialObservation(StrictContract):
    """Clock-sandwiched non-secret lease status for one event-5 occurrence."""

    pre_release_credential_observation_id: str
    state: RunActionPreReleaseCredentialState
    activation_revalidation_receipt: RunActionActivationRevalidationReceipt
    credential_lease_status: RunActionCredentialLeaseStatus
    observed_before_realtime_nanoseconds: int
    observed_after_realtime_nanoseconds: int
    required_valid_until_realtime_nanoseconds: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-pre-release-credential-observation"
    IDENTITY_FIELD: ClassVar[str] = "pre_release_credential_observation_id"

    def _validate(self) -> None:
        if (
            type(self.state) is not RunActionPreReleaseCredentialState
            or type(self.activation_revalidation_receipt)
            is not RunActionActivationRevalidationReceipt
            or type(self.credential_lease_status) is not RunActionCredentialLeaseStatus
        ):
            raise RunActionCredentialContractError(
                "pre-release credential observation is malformed"
            )
        activation = self.activation_revalidation_receipt
        policy = activation.prepared_execution.preparation_claim.execution_policy
        credential_file = activation.credential_file_observation
        request = run_action_credential_lease_request(
            activation.prepared_execution,
            activation.spawn_commit,
        )
        expected_authority_id = run_action_credential_lease_authority_id(
            activation.prepared_execution,
            activation.spawn_commit,
        )
        status = self.credential_lease_status
        observed_before = self.observed_before_realtime_nanoseconds
        observed_after = self.observed_after_realtime_nanoseconds
        required_valid_until = self.required_valid_until_realtime_nanoseconds
        valid_until = status.valid_until_realtime_nanoseconds
        maximum_valid_until = (
            observed_after
            + policy.credential_policy.maximum_lease_seconds * _NANOSECONDS_PER_SECOND
        )
        derived_state = (
            RunActionPreReleaseCredentialState.EXPIRED
            if valid_until <= observed_after
            else (
                RunActionPreReleaseCredentialState.VALID
                if valid_until >= required_valid_until
                else RunActionPreReleaseCredentialState.RENEWAL_PENDING
            )
        )
        if (
            policy.credential_policy.mode is not RunActionCredentialMode.SUPERVISOR_FILE
            or credential_file is None
            or credential_file.content_authority_id != expected_authority_id
            or status.credential_lease_request_id != request.credential_lease_request_id
            or type(observed_before) is not int
            or observed_before <= 0
            or type(observed_after) is not int
            or observed_after < observed_before
            or observed_after > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            or type(required_valid_until) is not int
            or required_valid_until <= observed_after
            or required_valid_until > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            or maximum_valid_until > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            or valid_until > maximum_valid_until
            or self.state is not derived_state
        ):
            raise RunActionCredentialContractError(
                "pre-release credential observation is inconsistent"
            )


@dataclass(frozen=True)
class RunActionCredentialRetirementIntent(StrictContract):
    """Durable precedence: this event-5 occurrence may only be retired."""

    credential_retirement_intent_id: str
    activation_event_id: str
    pre_release_credential_observation_id: str
    credential_lease_status: RunActionCredentialLeaseStatus
    observed_before_realtime_nanoseconds: int
    observed_after_realtime_nanoseconds: int
    required_valid_until_realtime_nanoseconds: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-credential-retirement-intent"
    IDENTITY_FIELD: ClassVar[str] = "credential_retirement_intent_id"

    def _validate(self) -> None:
        activation_event_id = require_content_id(
            self.activation_event_id,
            "credential retirement activation event",
        )
        observation_id = require_content_id(
            self.pre_release_credential_observation_id,
            "credential retirement observation",
        )
        status = self.credential_lease_status
        if (
            activation_event_id.split(":sha256:", 1)[0] != "run-action-execution-event"
            or observation_id.split(":sha256:", 1)[0]
            != RunActionPreReleaseCredentialObservation.CONTENT_NAMESPACE
            or type(status) is not RunActionCredentialLeaseStatus
            or type(self.observed_before_realtime_nanoseconds) is not int
            or self.observed_before_realtime_nanoseconds <= 0
            or type(self.observed_after_realtime_nanoseconds) is not int
            or self.observed_after_realtime_nanoseconds
            < self.observed_before_realtime_nanoseconds
            or self.observed_after_realtime_nanoseconds
            > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            or type(self.required_valid_until_realtime_nanoseconds) is not int
            or self.required_valid_until_realtime_nanoseconds
            <= self.observed_after_realtime_nanoseconds
            or self.required_valid_until_realtime_nanoseconds
            > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            or status.valid_until_realtime_nanoseconds
            > self.observed_after_realtime_nanoseconds
        ):
            raise RunActionCredentialContractError(
                "credential retirement intent lacks exact expired authority"
            )


def credential_retirement_intent_matches_activation(
    intent: RunActionCredentialRetirementIntent,
    activation_event_id: str,
    activation: RunActionActivationRevalidationReceipt,
) -> bool:
    """Join one durable retirement decision to the selected event-5 activation."""

    if not (
        type(intent) is RunActionCredentialRetirementIntent
        and type(activation_event_id) is str
        and type(activation) is RunActionActivationRevalidationReceipt
        and intent.activation_event_id == activation_event_id
        and intent.credential_lease_status.credential_lease_request_id
        == run_action_credential_lease_request(
            activation.prepared_execution,
            activation.spawn_commit,
        ).credential_lease_request_id
        and intent.required_valid_until_realtime_nanoseconds
        == (
            intent.observed_after_realtime_nanoseconds
            + (
                activation.prepared_execution.preparation_claim.execution_policy.supervisor_limits.execution_timeout_seconds
                + activation.prepared_execution.preparation_claim.execution_policy.supervisor_limits.termination_grace_seconds
            )
            * _NANOSECONDS_PER_SECOND
        )
        and activation.credential_file_observation is not None
        and activation.credential_file_observation.content_authority_id
        == run_action_credential_lease_authority_id(
            activation.prepared_execution,
            activation.spawn_commit,
        )
    ):
        return False
    policy = activation.prepared_execution.preparation_claim.execution_policy
    maximum_valid_until = (
        intent.observed_after_realtime_nanoseconds
        + policy.credential_policy.maximum_lease_seconds * _NANOSECONDS_PER_SECOND
    )
    if maximum_valid_until > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER:
        return False
    observation = RunActionPreReleaseCredentialObservation.mint(
        state=RunActionPreReleaseCredentialState.EXPIRED,
        activation_revalidation_receipt=activation,
        credential_lease_status=intent.credential_lease_status,
        observed_before_realtime_nanoseconds=(
            intent.observed_before_realtime_nanoseconds
        ),
        observed_after_realtime_nanoseconds=(
            intent.observed_after_realtime_nanoseconds
        ),
        required_valid_until_realtime_nanoseconds=(
            intent.required_valid_until_realtime_nanoseconds
        ),
    )
    return (
        intent.pre_release_credential_observation_id
        == observation.pre_release_credential_observation_id
    )


def maximum_credential_retirement_intent(
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
) -> RunActionCredentialRetirementIntent:
    """Mint the widest numeric representation admitted by the expiry contract."""

    if (
        type(prepared_execution) is not RunActionPreparedExecution
        or type(spawn_commit) is not RunActionSpawnCommit
    ):
        raise RunActionCredentialContractError(
            "credential retirement envelope requires one exact prepared spawn"
        )
    policy = prepared_execution.preparation_claim.execution_policy
    required_duration_seconds = (
        policy.supervisor_limits.execution_timeout_seconds
        + policy.supervisor_limits.termination_grace_seconds
    )
    maximum_duration_seconds = max(
        required_duration_seconds,
        policy.credential_policy.maximum_lease_seconds,
    )
    observed_after = (
        RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
        - maximum_duration_seconds * _NANOSECONDS_PER_SECOND
    )
    required_valid_until = (
        observed_after + required_duration_seconds * _NANOSECONDS_PER_SECOND
    )
    request = run_action_credential_lease_request(
        prepared_execution,
        spawn_commit,
    )
    status = RunActionCredentialLeaseStatus.mint(
        credential_lease_request_id=request.credential_lease_request_id,
        valid_until_realtime_nanoseconds=observed_after,
    )
    activation_event_id = content_id(
        "run-action-execution-event",
        {"credential_retirement_size_bound": True},
    )
    return RunActionCredentialRetirementIntent.mint(
        activation_event_id=activation_event_id,
        pre_release_credential_observation_id=content_id(
            RunActionPreReleaseCredentialObservation.CONTENT_NAMESPACE,
            {"credential_retirement_size_bound": True},
        ),
        credential_lease_status=status,
        observed_before_realtime_nanoseconds=observed_after,
        observed_after_realtime_nanoseconds=observed_after,
        required_valid_until_realtime_nanoseconds=required_valid_until,
    )


__all__ = [
    "credential_retirement_intent_matches_activation",
    "maximum_credential_retirement_intent",
    "RunActionCredentialContractError",
    "RunActionCredentialRetirementIntent",
    "RunActionPreReleaseCredentialObservation",
    "RunActionPreReleaseCredentialState",
]
