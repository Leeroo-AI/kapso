"""Canonical authority for one irreversible run-action workload release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionResolvedWorkloadObservation,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialMode,
    RunActionSupervisorLimits,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)

_EXECUTION_EVENT_NAMESPACE = "run-action-execution-event"
_NANOSECONDS_PER_SECOND = 1_000_000_000


class RunActionReleaseContractError(ValueError):
    """A release receipt is malformed, spliced, stale, or insufficient."""


@dataclass(frozen=True)
class RunActionCredentialValidityObservation(StrictContract):
    """Non-secret proof that one delivered broker lease spans containment."""

    credential_validity_observation_id: str
    activated_credential_file_observation_id: str
    credential_lease_authority_id: str
    observed_at_realtime_nanoseconds: int
    valid_until_realtime_nanoseconds: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-credential-validity-observation"
    IDENTITY_FIELD: ClassVar[str] = "credential_validity_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.activated_credential_file_observation_id,
            "run-action-activated-file-observation",
            "credential validity activated file",
        )
        require_identifier(
            self.credential_lease_authority_id,
            "credential validity lease authority",
        )
        if (
            type(self.observed_at_realtime_nanoseconds) is not int
            or self.observed_at_realtime_nanoseconds <= 0
            or type(self.valid_until_realtime_nanoseconds) is not int
            or self.valid_until_realtime_nanoseconds
            <= self.observed_at_realtime_nanoseconds
        ):
            raise RunActionReleaseContractError(
                "credential validity interval is invalid"
            )


@dataclass(frozen=True)
class RunActionReleaseAuthorizationObservation(StrictContract):
    """The last external authorities observed before the release link."""

    release_authorization_observation_id: str
    security_observation: SecurityDenylistObservation
    authorized_at_boottime_nanoseconds: int
    authorized_at_realtime_nanoseconds: int
    credential_validity_observation: RunActionCredentialValidityObservation | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-release-authorization-observation"
    IDENTITY_FIELD: ClassVar[str] = "release_authorization_observation_id"

    def _validate(self) -> None:
        if (
            type(self.security_observation) is not SecurityDenylistObservation
            or self.security_observation.matched_revocations
            or type(self.authorized_at_boottime_nanoseconds) is not int
            or self.authorized_at_boottime_nanoseconds <= 0
            or type(self.authorized_at_realtime_nanoseconds) is not int
            or self.authorized_at_realtime_nanoseconds <= 0
            or (
                self.credential_validity_observation is not None
                and type(self.credential_validity_observation)
                is not RunActionCredentialValidityObservation
            )
        ):
            raise RunActionReleaseContractError(
                "release authorization observation is unsafe or invalid"
            )


@dataclass(frozen=True)
class RunActionWorkloadReleaseReceipt(StrictContract):
    """Complete crash-surviving authority published as ``control/release``."""

    workload_release_receipt_id: str
    activation_event_id: str
    resolved_workload_observation: RunActionResolvedWorkloadObservation
    release_authorization_observation: RunActionReleaseAuthorizationObservation

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-workload-release-receipt"
    IDENTITY_FIELD: ClassVar[str] = "workload_release_receipt_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.activation_event_id,
            _EXECUTION_EVENT_NAMESPACE,
            "workload release activation event",
        )
        if (
            type(self.resolved_workload_observation)
            is not RunActionResolvedWorkloadObservation
            or type(self.release_authorization_observation)
            is not RunActionReleaseAuthorizationObservation
        ):
            raise RunActionReleaseContractError(
                "workload release lacks its exact evidence graph"
            )
        resolved = self.resolved_workload_observation
        activation = resolved.activation_revalidation_receipt
        policy = activation.prepared_execution.preparation_claim.execution_policy
        reservation = activation.prepared_execution.preparation_claim.reservation
        authorization = self.release_authorization_observation
        credential_validity = authorization.credential_validity_observation
        credential_file = activation.credential_file_observation
        containment_realtime_deadline = (
            authorization.authorized_at_realtime_nanoseconds
            + (
                policy.supervisor_limits.execution_timeout_seconds
                + policy.supervisor_limits.termination_grace_seconds
            )
            * _NANOSECONDS_PER_SECOND
        )
        credential_mode = policy.credential_policy.mode
        if (
            authorization.security_observation.observation_id
            != reservation.frontier.security_observation_id
            or (credential_validity is None)
            != (credential_mode is RunActionCredentialMode.NONE)
            or (credential_file is None)
            != (credential_mode is RunActionCredentialMode.NONE)
            or (
                credential_validity is not None
                and (
                    credential_validity.activated_credential_file_observation_id
                    != credential_file.activated_file_observation_id
                    or credential_validity.credential_lease_authority_id
                    != credential_file.content_authority_id
                    or credential_validity.observed_at_realtime_nanoseconds
                    > authorization.authorized_at_realtime_nanoseconds
                    or credential_validity.valid_until_realtime_nanoseconds
                    < containment_realtime_deadline
                    or (
                        credential_validity.valid_until_realtime_nanoseconds
                        - credential_validity.observed_at_realtime_nanoseconds
                    )
                    > (
                        policy.credential_policy.maximum_lease_seconds
                        * _NANOSECONDS_PER_SECOND
                    )
                )
            )
            or len(self.to_json_bytes())
            > policy.supervisor_limits.release_receipt_size_bytes
        ):
            raise RunActionReleaseContractError(
                "workload release authorization differs from event-5 authority"
            )

    @property
    def host_boot_id(self) -> str:
        return self.resolved_workload_observation.host_boot_id

    @property
    def execution_deadline_boottime_nanoseconds(self) -> int:
        limits = self._supervisor_limits
        return (
            self.release_authorization_observation.authorized_at_boottime_nanoseconds
            + limits.execution_timeout_seconds * _NANOSECONDS_PER_SECOND
        )

    @property
    def containment_deadline_boottime_nanoseconds(self) -> int:
        return (
            self.execution_deadline_boottime_nanoseconds
            + self._supervisor_limits.termination_grace_seconds
            * _NANOSECONDS_PER_SECOND
        )

    @property
    def _supervisor_limits(self) -> RunActionSupervisorLimits:
        return (
            self.resolved_workload_observation.activation_revalidation_receipt.prepared_execution.preparation_claim.execution_policy.supervisor_limits
        )


def _require_namespaced_content_id(
    value: str,
    namespace: str,
    name: str,
) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionReleaseContractError(f"{name} uses another namespace")


__all__ = [
    "RunActionCredentialValidityObservation",
    "RunActionReleaseAuthorizationObservation",
    "RunActionReleaseContractError",
    "RunActionWorkloadReleaseReceipt",
]
