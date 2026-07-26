"""Typed evidence contracts for terminal run-action provider outcomes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import (
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierRunningContainerObservation,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseAdoption,
)
from kapso.cross_run.launch.run_action_result_authority import (
    run_action_terminal_result_evidence_matches,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionActivationRevalidationReceipt,
    RunActionPreparationAllocation,
    RunActionResultCaptureReceipt,
    RunActionRuntimeVolumeEvidence,
    RunActionTerminalObservation,
    RunActionVolumeKeeperEvidence,
    run_action_runtime_volume_occurrence_matches,
)

_BOOT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-" r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class RunActionTerminationContractError(ValueError):
    """Termination evidence is malformed, spliced, or outcome-incompatible."""


class RunActionProviderTerminationDisposition(str, Enum):
    """Whether the provider failed or the supervisor interrupted it."""

    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RunActionProviderTerminationReason(str, Enum):
    """Minimal mutually exclusive reason for one provider termination."""

    TIMEOUT = "timeout"
    OOM = "oom"
    NONZERO_EXIT = "nonzero_exit"
    EMPTY_RESULT = "empty_result"
    PRE_RELEASE_MAIN_LOSS = "pre_release_main_loss"


@dataclass(frozen=True)
class RunActionTimeoutDirective(StrictContract):
    """Fresh running proof sampled after one release-derived execution deadline."""

    timeout_directive_id: str
    activation_event_id: str
    workload_release_receipt_id: str
    workload_release_adoption_id: str
    host_boot_id: str
    execution_deadline_boottime_nanoseconds: int
    containment_deadline_boottime_nanoseconds: int
    observed_before_boottime_nanoseconds: int
    running_container_observation: RunActionBarrierRunningContainerObservation
    observed_after_boottime_nanoseconds: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-timeout-directive"
    IDENTITY_FIELD: ClassVar[str] = "timeout_directive_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.activation_event_id,
            "run-action-execution-event",
            "timeout activation event",
        )
        _require_namespaced_content_id(
            self.workload_release_receipt_id,
            "run-action-workload-release-receipt",
            "timeout workload release receipt",
        )
        _require_namespaced_content_id(
            self.workload_release_adoption_id,
            RunActionWorkloadReleaseAdoption.CONTENT_NAMESPACE,
            "timeout workload release adoption",
        )
        if (
            _BOOT_ID_PATTERN.fullmatch(self.host_boot_id) is None
            or type(self.running_container_observation)
            is not RunActionBarrierRunningContainerObservation
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.execution_deadline_boottime_nanoseconds,
                    self.containment_deadline_boottime_nanoseconds,
                    self.observed_before_boottime_nanoseconds,
                    self.observed_after_boottime_nanoseconds,
                )
            )
            or self.containment_deadline_boottime_nanoseconds
            <= self.execution_deadline_boottime_nanoseconds
            or self.observed_before_boottime_nanoseconds
            < self.execution_deadline_boottime_nanoseconds
            or self.observed_after_boottime_nanoseconds
            < self.observed_before_boottime_nanoseconds
        ):
            raise RunActionTerminationContractError(
                "run action timeout directive lacks a valid deadline observation"
            )


@dataclass(frozen=True)
class RunActionTimeoutDirectivePublicationReceipt(StrictContract):
    """Descriptor-read proof of the exact timeout directive linked after release."""

    timeout_directive_publication_receipt_id: str
    timeout_directive: RunActionTimeoutDirective
    workload_release_adoption_id: str
    prepared_control_directory_id: str
    control_mount_id: int
    control_device: int
    control_inode: int
    release_mount_id: int
    release_device: int
    release_inode: int
    relative_path: str
    file_type: str
    owner_user_id: int
    owner_group_id: int
    mode: int
    link_count: int
    size_bytes: int
    content_digest: str
    timeout_mount_id: int
    timeout_device: int
    timeout_inode: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-timeout-directive-publication"
    IDENTITY_FIELD: ClassVar[str] = "timeout_directive_publication_receipt_id"

    def _validate(self) -> None:
        if type(self.timeout_directive) is not RunActionTimeoutDirective:
            raise RunActionTerminationContractError(
                "timeout publication lacks one exact directive"
            )
        _require_namespaced_content_id(
            self.workload_release_adoption_id,
            RunActionWorkloadReleaseAdoption.CONTENT_NAMESPACE,
            "timeout publication workload release adoption",
        )
        _require_namespaced_content_id(
            self.prepared_control_directory_id,
            "run-action-prepared-runtime-directory",
            "timeout publication prepared control directory",
        )
        directive_payload = self.timeout_directive.to_json_bytes()
        if (
            self.timeout_directive.workload_release_adoption_id
            != self.workload_release_adoption_id
            or self.relative_path != "control/timeout"
            or self.file_type != "regular"
            or type(self.owner_user_id) is not int
            or self.owner_user_id <= 0
            or type(self.owner_group_id) is not int
            or self.owner_group_id <= 0
            or self.mode != 0o400
            or self.link_count != 1
            or self.size_bytes != len(directive_payload)
            or self.content_digest != tree_or_blob_digest(directive_payload)
            or self.timeout_mount_id != self.control_mount_id
            or self.timeout_device != self.control_device
            or self.release_mount_id != self.control_mount_id
            or self.release_device != self.control_device
            or len(
                {
                    self.control_inode,
                    self.release_inode,
                    self.timeout_inode,
                }
            )
            != 3
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.control_mount_id,
                    self.control_device,
                    self.control_inode,
                    self.release_mount_id,
                    self.release_device,
                    self.release_inode,
                    self.timeout_mount_id,
                    self.timeout_device,
                    self.timeout_inode,
                )
            )
        ):
            raise RunActionTerminationContractError(
                "run action timeout publication is not one exact linked directive"
            )


@dataclass(frozen=True)
class RunActionPreReleaseMainLossObservation(StrictContract):
    """Stable positive proof that only the unreleased main occurrence disappeared."""

    pre_release_main_loss_observation_id: str
    activation_event_id: str
    preparation_allocation: RunActionPreparationAllocation
    activation_revalidation_receipt: RunActionActivationRevalidationReceipt
    host_boot_id: str
    observed_before_boottime_nanoseconds: int
    first_complete_inventory_digest: str
    reobserved_volume_evidence: RunActionRuntimeVolumeEvidence
    reobserved_keeper_evidence: RunActionVolumeKeeperEvidence
    second_complete_inventory_digest: str
    observed_after_boottime_nanoseconds: int
    observed_runtime_volume_names: tuple[str, ...]
    observed_keeper_container_ids: tuple[str, ...]
    observed_main_container_ids: tuple[str, ...]
    missing_provider_execution_id: str
    control_mount_id: int
    control_device: int
    control_inode: int
    control_entry_count: int
    release_present: bool

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-pre-release-main-loss-observation"
    IDENTITY_FIELD: ClassVar[str] = "pre_release_main_loss_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.activation_event_id,
            "run-action-execution-event",
            "pre-release loss activation event",
        )
        if (
            type(self.preparation_allocation) is not RunActionPreparationAllocation
            or type(self.activation_revalidation_receipt)
            is not RunActionActivationRevalidationReceipt
            or type(self.reobserved_volume_evidence)
            is not RunActionRuntimeVolumeEvidence
            or type(self.reobserved_keeper_evidence)
            is not RunActionVolumeKeeperEvidence
        ):
            raise RunActionTerminationContractError(
                "pre-release main loss lacks its exact prepared evidence"
            )
        activation = self.activation_revalidation_receipt
        prepared = activation.prepared_execution
        allocation = self.preparation_allocation
        control = prepared.control_directory
        require_identifier(
            self.missing_provider_execution_id,
            "pre-release missing provider execution",
        )
        if (
            _BOOT_ID_PATTERN.fullmatch(self.host_boot_id) is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.first_complete_inventory_digest)
            is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.second_complete_inventory_digest)
            is None
            or self.first_complete_inventory_digest
            != self.second_complete_inventory_digest
            or type(self.observed_before_boottime_nanoseconds) is not int
            or self.observed_before_boottime_nanoseconds <= 0
            or type(self.observed_after_boottime_nanoseconds) is not int
            or self.observed_after_boottime_nanoseconds
            < self.observed_before_boottime_nanoseconds
            or allocation.preparation_claim != prepared.preparation_claim
            or allocation.runtime_volume_authority != prepared.runtime_volume_authority
            or not run_action_runtime_volume_occurrence_matches(
                self.reobserved_volume_evidence,
                activation.reobserved_volume_evidence,
            )
            or self.reobserved_keeper_evidence != activation.reobserved_keeper_evidence
            or self.observed_runtime_volume_names
            != (prepared.runtime_volume_authority.volume_name,)
            or self.observed_keeper_container_ids
            != (prepared.volume_keeper_evidence.container_id,)
            or self.observed_main_container_ids
            or self.missing_provider_execution_id
            != activation.spawn_commit.provider_execution_id
            or (
                self.control_mount_id,
                self.control_device,
                self.control_inode,
            )
            != (control.mount_id, control.device, control.inode)
            or self.control_entry_count != 0
            or self.release_present is not False
        ):
            raise RunActionTerminationContractError(
                "pre-release main loss observation is incomplete or spliced"
            )


@dataclass(frozen=True)
class RunActionProviderTerminationReceipt(StrictContract):
    """One complete mutually exclusive provider termination evidence graph."""

    provider_termination_receipt_id: str
    disposition: RunActionProviderTerminationDisposition
    reason: RunActionProviderTerminationReason
    activation_event_id: str
    activation_revalidation_receipt: RunActionActivationRevalidationReceipt
    workload_release_adoption: RunActionWorkloadReleaseAdoption | None
    terminal_observation: RunActionTerminalObservation | None
    timeout_directive_publication: RunActionTimeoutDirectivePublicationReceipt | None
    empty_result_capture_receipt: RunActionResultCaptureReceipt | None
    pre_release_main_loss_observation: RunActionPreReleaseMainLossObservation | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-provider-termination-receipt"
    IDENTITY_FIELD: ClassVar[str] = "provider_termination_receipt_id"

    def _validate(self) -> None:
        if (
            type(self.disposition) is not RunActionProviderTerminationDisposition
            or type(self.reason) is not RunActionProviderTerminationReason
            or type(self.activation_revalidation_receipt)
            is not RunActionActivationRevalidationReceipt
        ):
            raise RunActionTerminationContractError(
                "provider termination receipt has invalid typed authority"
            )
        _require_namespaced_content_id(
            self.activation_event_id,
            "run-action-execution-event",
            "provider termination activation event",
        )
        expected_disposition = (
            RunActionProviderTerminationDisposition.INTERRUPTED
            if self.reason
            in {
                RunActionProviderTerminationReason.TIMEOUT,
                RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
            }
            else RunActionProviderTerminationDisposition.FAILED
        )
        if self.disposition is not expected_disposition:
            raise RunActionTerminationContractError(
                "provider termination disposition differs from its reason"
            )
        if self.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS:
            self._validate_pre_release_main_loss()
            return
        self._validate_released_terminal()

    def _validate_pre_release_main_loss(self) -> None:
        loss = self.pre_release_main_loss_observation
        if (
            type(loss) is not RunActionPreReleaseMainLossObservation
            or loss.activation_revalidation_receipt
            != self.activation_revalidation_receipt
            or loss.activation_event_id != self.activation_event_id
            or self.workload_release_adoption is not None
            or self.terminal_observation is not None
            or self.timeout_directive_publication is not None
            or self.empty_result_capture_receipt is not None
        ):
            raise RunActionTerminationContractError(
                "pre-release main loss must be the sole termination evidence branch"
            )

    def _validate_released_terminal(self) -> None:
        adoption = self.workload_release_adoption
        terminal = self.terminal_observation
        if (
            type(adoption) is not RunActionWorkloadReleaseAdoption
            or type(terminal) is not RunActionTerminalObservation
            or self.pre_release_main_loss_observation is not None
            or adoption.workload_release_receipt.activation_event_id
            != self.activation_event_id
            or not _released_terminal_evidence_matches(
                terminal,
                self.activation_revalidation_receipt,
                adoption,
            )
        ):
            raise RunActionTerminationContractError(
                "released provider termination lacks its exact terminal occurrence"
            )
        if self.reason is RunActionProviderTerminationReason.TIMEOUT:
            if (
                type(self.timeout_directive_publication)
                is not RunActionTimeoutDirectivePublicationReceipt
                or self.empty_result_capture_receipt is not None
                or self.timeout_directive_publication.timeout_directive.activation_event_id
                != self.activation_event_id
                or not _timeout_publication_evidence_matches(
                    self.timeout_directive_publication,
                    self.activation_revalidation_receipt,
                    adoption,
                )
            ):
                raise RunActionTerminationContractError(
                    "timeout termination lacks its exact published directive"
                )
            return
        if self.timeout_directive_publication is not None:
            raise RunActionTerminationContractError(
                "published timeout authority has precedence over provider failure"
            )
        if self.reason is RunActionProviderTerminationReason.OOM:
            valid_outcome = (
                terminal.oom_killed is True
                and self.empty_result_capture_receipt is None
            )
        elif self.reason is RunActionProviderTerminationReason.NONZERO_EXIT:
            valid_outcome = (
                terminal.oom_killed is False
                and terminal.exit_code != 0
                and self.empty_result_capture_receipt is None
            )
        else:
            capture = self.empty_result_capture_receipt
            valid_outcome = (
                self.reason is RunActionProviderTerminationReason.EMPTY_RESULT
                and terminal.oom_killed is False
                and terminal.exit_code == 0
                and type(capture) is RunActionResultCaptureReceipt
                and capture.size_bytes == 0
                and capture.content_digest == tree_or_blob_digest(b"")
                and run_action_terminal_result_evidence_matches(
                    terminal,
                    capture,
                    self.activation_revalidation_receipt,
                    adoption,
                )
            )
        if not valid_outcome:
            raise RunActionTerminationContractError(
                "provider termination reason differs from terminal evidence"
            )


def _released_terminal_evidence_matches(
    terminal: RunActionTerminalObservation,
    activation: RunActionActivationRevalidationReceipt,
    adoption: RunActionWorkloadReleaseAdoption,
) -> bool:
    release = adoption.workload_release_receipt
    resolved = release.resolved_workload_observation
    if resolved.activation_revalidation_receipt != activation:
        return False
    prepared = activation.prepared_execution
    spawn = activation.spawn_commit
    running = resolved.running_container_observation
    return (
        terminal.activation_revalidation_receipt_id
        == activation.activation_revalidation_receipt_id
        and terminal.workload_release_adoption_id
        == adoption.workload_release_adoption_id
        and terminal.prepared_execution_id == prepared.prepared_execution_id
        and terminal.spawn_commit_id == spawn.spawn_commit_id
        and terminal.provider_execution_id == spawn.provider_execution_id
        and terminal.provider_execution_id == running.container_id
        and terminal.runtime_volume_authority_id
        == prepared.runtime_volume_authority.runtime_volume_authority_id
        and terminal.generation_nonce
        == prepared.runtime_volume_authority.generation_nonce
        and terminal.observed_inspect_projection
        == prepared.inert_container_evidence.issued_create_projection
        and terminal.observed_inspect_projection == running.observed_inspect_projection
        and terminal.started_at == running.started_at
    )


def _timeout_publication_evidence_matches(
    publication: RunActionTimeoutDirectivePublicationReceipt,
    activation: RunActionActivationRevalidationReceipt,
    adoption: RunActionWorkloadReleaseAdoption,
) -> bool:
    directive = publication.timeout_directive
    release = adoption.workload_release_receipt
    prepared = activation.prepared_execution
    control = prepared.control_directory
    authority = prepared.runtime_volume_authority
    released_running = (
        release.resolved_workload_observation.running_container_observation
    )
    timeout_running = directive.running_container_observation
    return (
        release.resolved_workload_observation.activation_revalidation_receipt
        == activation
        and directive.activation_event_id == release.activation_event_id
        and directive.workload_release_receipt_id == release.workload_release_receipt_id
        and directive.workload_release_adoption_id
        == adoption.workload_release_adoption_id
        and directive.host_boot_id == release.host_boot_id
        and directive.execution_deadline_boottime_nanoseconds
        == release.execution_deadline_boottime_nanoseconds
        and directive.containment_deadline_boottime_nanoseconds
        == release.containment_deadline_boottime_nanoseconds
        and timeout_running.container_id
        == activation.spawn_commit.provider_execution_id
        and timeout_running.container_id == released_running.container_id
        and timeout_running.observed_inspect_projection
        == prepared.inert_container_evidence.issued_create_projection
        and timeout_running.observed_inspect_projection
        == released_running.observed_inspect_projection
        and timeout_running.init_process_id == released_running.init_process_id
        and timeout_running.started_at == released_running.started_at
        and publication.workload_release_adoption_id
        == adoption.workload_release_adoption_id
        and publication.prepared_control_directory_id
        == control.prepared_runtime_directory_id
        and (
            publication.control_mount_id,
            publication.control_device,
            publication.control_inode,
        )
        == (control.mount_id, control.device, control.inode)
        and (
            publication.release_mount_id,
            publication.release_device,
            publication.release_inode,
        )
        == (
            adoption.release_mount_id,
            adoption.release_device,
            adoption.release_inode,
        )
        and publication.owner_user_id == authority.owner_user_id
        and publication.owner_group_id == authority.owner_group_id
    )


def _require_namespaced_content_id(
    value: str,
    namespace: str,
    name: str,
) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionTerminationContractError(f"{name} uses another namespace")


__all__ = [
    "RunActionPreReleaseMainLossObservation",
    "RunActionProviderTerminationDisposition",
    "RunActionProviderTerminationReason",
    "RunActionProviderTerminationReceipt",
    "RunActionTerminationContractError",
    "RunActionTimeoutDirective",
    "RunActionTimeoutDirectivePublicationReceipt",
]
