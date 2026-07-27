"""Typed evidence contracts for terminal run-action provider outcomes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_credential_contracts import (
    credential_retirement_intent_matches_activation,
    RunActionCredentialRetirementIntent,
)
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierRunningContainerObservation,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseAdoption,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionCreateInspectProjection,
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionActivationRevalidationReceipt,
    RunActionPreparationAllocation,
    RunActionRuntimeVolumeEvidence,
    RunActionTerminalObservation,
    RunActionVolumeKeeperEvidence,
    run_action_runtime_volume_occurrence_matches,
)

_BOOT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-" r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GENERATION_NONCE_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_DOCKER_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})T"
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:[.](?P<fraction>[0-9]{1,9}))?Z$"
)
_ZERO_DOCKER_TIMESTAMP = "0001-01-01T00:00:00Z"


def _bounded_positive_physical_integer(value: object) -> bool:
    return type(value) is int and 1 <= value <= RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER


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
    MISSING_RESULT = "missing_result"
    PRE_RELEASE_MAIN_LOSS = "pre_release_main_loss"
    PRE_RELEASE_MAIN_TERMINAL = "pre_release_main_terminal"
    CREDENTIAL_EXPIRED = "credential_expired"


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
                not _bounded_positive_physical_integer(value)
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
                not _bounded_positive_physical_integer(value)
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
    control_directory_topology: RunActionControlDirectoryTopology

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
            or not _bounded_positive_physical_integer(
                self.observed_before_boottime_nanoseconds
            )
            or not _bounded_positive_physical_integer(
                self.observed_after_boottime_nanoseconds
            )
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
            or self.control_directory_topology
            is not RunActionControlDirectoryTopology.EMPTY
        ):
            raise RunActionTerminationContractError(
                "pre-release main loss observation is incomplete or spliced"
            )


def run_action_pre_release_main_loss_observation_token(
    observation: RunActionPreReleaseMainLossObservation,
) -> str:
    """Bind the stable loss occurrence while excluding sampling time and usage."""

    if type(observation) is not RunActionPreReleaseMainLossObservation:
        raise RunActionTerminationContractError(
            "pre-release main loss token requires one exact observation"
        )
    return tree_or_blob_digest(
        canonical_json_bytes(
            {
                "activation_event_id": observation.activation_event_id,
                "preparation_allocation_id": (
                    observation.preparation_allocation.preparation_allocation_id
                ),
                "activation_revalidation_receipt_id": (
                    observation.activation_revalidation_receipt.activation_revalidation_receipt_id
                ),
                "host_boot_id": observation.host_boot_id,
                "complete_inventory_digest": (
                    observation.first_complete_inventory_digest
                ),
                "missing_provider_execution_id": (
                    observation.missing_provider_execution_id
                ),
                "control_mount_id": observation.control_mount_id,
                "control_device": observation.control_device,
                "control_inode": observation.control_inode,
                "control_entry_count": observation.control_entry_count,
                "control_directory_topology": (
                    observation.control_directory_topology.value
                ),
            }
        )
    )


@dataclass(frozen=True)
class RunActionPreReleaseTerminalContainerObservation(StrictContract):
    """Release-independent exited state of the exact event-5 main occurrence."""

    pre_release_terminal_container_observation_id: str
    prepared_execution_id: str
    spawn_commit_id: str
    provider_execution_id: str
    runtime_volume_authority_id: str
    generation_nonce: str
    activation_revalidation_receipt_id: str
    observed_inspect_projection: DockerRunActionCreateInspectProjection
    complete_inspection_digest: str
    container_status: str
    process_id: int
    restart_count: int
    paused: bool
    restarting: bool
    dead: bool
    started_at: str
    finished_at: str
    exit_code: int
    oom_killed: bool
    state_error: str

    CONTENT_NAMESPACE: ClassVar[str] = (
        "run-action-pre-release-terminal-container-observation"
    )
    IDENTITY_FIELD: ClassVar[str] = "pre_release_terminal_container_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.prepared_execution_id,
            "run-action-prepared-execution",
            "pre-release terminal prepared execution",
        )
        _require_namespaced_content_id(
            self.spawn_commit_id,
            "run-action-spawn-commit",
            "pre-release terminal spawn commit",
        )
        require_identifier(
            self.provider_execution_id,
            "pre-release terminal provider execution",
        )
        _require_namespaced_content_id(
            self.runtime_volume_authority_id,
            "run-action-runtime-volume-authority",
            "pre-release terminal runtime volume",
        )
        _require_namespaced_content_id(
            self.activation_revalidation_receipt_id,
            RunActionActivationRevalidationReceipt.CONTENT_NAMESPACE,
            "pre-release terminal activation revalidation",
        )
        started_at = _docker_timestamp_order_key(self.started_at)
        finished_at = _docker_timestamp_order_key(self.finished_at)
        if (
            type(self.observed_inspect_projection)
            is not DockerRunActionCreateInspectProjection
            or _GENERATION_NONCE_PATTERN.fullmatch(self.generation_nonce) is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.complete_inspection_digest) is None
            or self.container_status != "exited"
            or self.process_id != 0
            or self.restart_count != 0
            or self.paused is not False
            or self.restarting is not False
            or self.dead is not False
            or self.started_at == _ZERO_DOCKER_TIMESTAMP
            or self.finished_at == _ZERO_DOCKER_TIMESTAMP
            or started_at is None
            or finished_at is None
            or finished_at < started_at
            or type(self.exit_code) is not int
            or not 0 <= self.exit_code <= 255
            or type(self.oom_killed) is not bool
            or self.state_error != ""
        ):
            raise RunActionTerminationContractError(
                "pre-release terminal container observation is invalid"
            )


@dataclass(frozen=True)
class RunActionPreReleaseMainTerminalObservation(StrictContract):
    """Stable proof that the unreleased event-5 main exists and has exited."""

    pre_release_main_terminal_observation_id: str
    activation_event_id: str
    preparation_allocation: RunActionPreparationAllocation
    activation_revalidation_receipt: RunActionActivationRevalidationReceipt
    host_boot_id: str
    observed_before_boottime_nanoseconds: int
    first_complete_inventory_digest: str
    reobserved_volume_evidence: RunActionRuntimeVolumeEvidence
    reobserved_keeper_evidence: RunActionVolumeKeeperEvidence
    terminal_container_observation: RunActionPreReleaseTerminalContainerObservation
    second_complete_inventory_digest: str
    observed_after_boottime_nanoseconds: int
    observed_runtime_volume_names: tuple[str, ...]
    observed_keeper_container_ids: tuple[str, ...]
    observed_main_container_ids: tuple[str, ...]
    control_mount_id: int
    control_device: int
    control_inode: int
    control_entry_count: int
    control_directory_topology: RunActionControlDirectoryTopology

    CONTENT_NAMESPACE: ClassVar[str] = (
        "run-action-pre-release-main-terminal-observation"
    )
    IDENTITY_FIELD: ClassVar[str] = "pre_release_main_terminal_observation_id"

    def _validate(self) -> None:
        _require_namespaced_content_id(
            self.activation_event_id,
            "run-action-execution-event",
            "pre-release terminal activation event",
        )
        if (
            type(self.preparation_allocation) is not RunActionPreparationAllocation
            or type(self.activation_revalidation_receipt)
            is not RunActionActivationRevalidationReceipt
            or type(self.reobserved_volume_evidence)
            is not RunActionRuntimeVolumeEvidence
            or type(self.reobserved_keeper_evidence)
            is not RunActionVolumeKeeperEvidence
            or type(self.terminal_container_observation)
            is not RunActionPreReleaseTerminalContainerObservation
        ):
            raise RunActionTerminationContractError(
                "pre-release terminal lacks its exact prepared evidence"
            )
        activation = self.activation_revalidation_receipt
        prepared = activation.prepared_execution
        allocation = self.preparation_allocation
        control = prepared.control_directory
        spawn = activation.spawn_commit
        terminal = self.terminal_container_observation
        authority = prepared.runtime_volume_authority
        if (
            _BOOT_ID_PATTERN.fullmatch(self.host_boot_id) is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.first_complete_inventory_digest)
            is None
            or _SHA256_DIGEST_PATTERN.fullmatch(self.second_complete_inventory_digest)
            is None
            or self.first_complete_inventory_digest
            != self.second_complete_inventory_digest
            or not _bounded_positive_physical_integer(
                self.observed_before_boottime_nanoseconds
            )
            or not _bounded_positive_physical_integer(
                self.observed_after_boottime_nanoseconds
            )
            or self.observed_after_boottime_nanoseconds
            < self.observed_before_boottime_nanoseconds
            or allocation.preparation_claim != prepared.preparation_claim
            or allocation.runtime_volume_authority != authority
            or not run_action_runtime_volume_occurrence_matches(
                self.reobserved_volume_evidence,
                activation.reobserved_volume_evidence,
            )
            or self.reobserved_keeper_evidence != activation.reobserved_keeper_evidence
            or terminal.prepared_execution_id != prepared.prepared_execution_id
            or terminal.spawn_commit_id != spawn.spawn_commit_id
            or terminal.provider_execution_id != spawn.provider_execution_id
            or terminal.provider_execution_id
            != prepared.inert_container_evidence.container_id
            or terminal.runtime_volume_authority_id
            != authority.runtime_volume_authority_id
            or terminal.generation_nonce != authority.generation_nonce
            or terminal.activation_revalidation_receipt_id
            != activation.activation_revalidation_receipt_id
            or terminal.observed_inspect_projection
            != prepared.inert_container_evidence.issued_create_projection
            or self.observed_runtime_volume_names != (authority.volume_name,)
            or self.observed_keeper_container_ids
            != (prepared.volume_keeper_evidence.container_id,)
            or self.observed_main_container_ids != (spawn.provider_execution_id,)
            or (
                self.control_mount_id,
                self.control_device,
                self.control_inode,
            )
            != (control.mount_id, control.device, control.inode)
            or self.control_entry_count != 0
            or self.control_directory_topology
            is not RunActionControlDirectoryTopology.EMPTY
        ):
            raise RunActionTerminationContractError(
                "pre-release main terminal observation is incomplete or spliced"
            )


def run_action_pre_release_main_terminal_observation_token(
    observation: RunActionPreReleaseMainTerminalObservation,
) -> str:
    """Bind the stable present-exited occurrence without sampling times or usage."""

    if type(observation) is not RunActionPreReleaseMainTerminalObservation:
        raise RunActionTerminationContractError(
            "pre-release main terminal token requires one exact observation"
        )
    terminal = observation.terminal_container_observation
    return tree_or_blob_digest(
        canonical_json_bytes(
            {
                "activation_event_id": observation.activation_event_id,
                "preparation_allocation_id": (
                    observation.preparation_allocation.preparation_allocation_id
                ),
                "activation_revalidation_receipt_id": (
                    observation.activation_revalidation_receipt.activation_revalidation_receipt_id
                ),
                "host_boot_id": observation.host_boot_id,
                "complete_inventory_digest": (
                    observation.first_complete_inventory_digest
                ),
                "terminal_container_observation_id": (
                    terminal.pre_release_terminal_container_observation_id
                ),
                "complete_inspection_digest": terminal.complete_inspection_digest,
                "control_mount_id": observation.control_mount_id,
                "control_device": observation.control_device,
                "control_inode": observation.control_inode,
                "control_entry_count": observation.control_entry_count,
                "control_directory_topology": (
                    observation.control_directory_topology.value
                ),
            }
        )
    )


@dataclass(frozen=True)
class RunActionProviderTerminationReceipt(StrictContract):
    """One complete mutually exclusive provider termination evidence graph."""

    provider_termination_receipt_id: str
    disposition: RunActionProviderTerminationDisposition
    reason: RunActionProviderTerminationReason
    activation_event_id: str
    workload_release_adoption: RunActionWorkloadReleaseAdoption | None
    terminal_observation: (
        RunActionTerminalObservation | RunActionPreReleaseMainTerminalObservation | None
    )
    timeout_directive_publication: RunActionTimeoutDirectivePublicationReceipt | None
    pre_release_main_loss_observation: RunActionPreReleaseMainLossObservation | None
    credential_retirement_intent: RunActionCredentialRetirementIntent | None

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-provider-termination-receipt"
    IDENTITY_FIELD: ClassVar[str] = "provider_termination_receipt_id"

    def _validate(self) -> None:
        if (
            type(self.disposition) is not RunActionProviderTerminationDisposition
            or type(self.reason) is not RunActionProviderTerminationReason
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
                RunActionProviderTerminationReason.CREDENTIAL_EXPIRED,
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
        if self.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL:
            self._validate_pre_release_main_terminal()
            return
        if self.reason is RunActionProviderTerminationReason.CREDENTIAL_EXPIRED:
            self._validate_credential_expired()
            return
        self._validate_released_terminal()

    def _validate_pre_release_main_loss(self) -> None:
        loss = self.pre_release_main_loss_observation
        if (
            type(loss) is not RunActionPreReleaseMainLossObservation
            or loss.activation_event_id != self.activation_event_id
            or self.workload_release_adoption is not None
            or self.terminal_observation is not None
            or self.timeout_directive_publication is not None
            or self.credential_retirement_intent is not None
        ):
            raise RunActionTerminationContractError(
                "pre-release main loss must be the sole termination evidence branch"
            )

    def _validate_pre_release_main_terminal(self) -> None:
        terminal = self.terminal_observation
        if (
            type(terminal) is not RunActionPreReleaseMainTerminalObservation
            or terminal.activation_event_id != self.activation_event_id
            or self.workload_release_adoption is not None
            or self.timeout_directive_publication is not None
            or self.pre_release_main_loss_observation is not None
            or self.credential_retirement_intent is not None
        ):
            raise RunActionTerminationContractError(
                "pre-release main terminal must be the sole termination evidence branch"
            )

    def _validate_credential_expired(self) -> None:
        intent = self.credential_retirement_intent
        loss = self.pre_release_main_loss_observation
        terminal = self.terminal_observation
        physical_evidence_is_exact = (
            type(loss) is RunActionPreReleaseMainLossObservation
            and terminal is None
            and loss.activation_event_id == self.activation_event_id
        ) or (
            loss is None
            and type(terminal) is RunActionPreReleaseMainTerminalObservation
            and terminal.activation_event_id == self.activation_event_id
        )
        if (
            type(intent) is not RunActionCredentialRetirementIntent
            or intent.activation_event_id != self.activation_event_id
            or self.workload_release_adoption is not None
            or self.timeout_directive_publication is not None
            or not physical_evidence_is_exact
        ):
            raise RunActionTerminationContractError(
                "credential-expired termination lacks intent and physical evidence"
            )

    def _validate_released_terminal(self) -> None:
        adoption = self.workload_release_adoption
        terminal = self.terminal_observation
        if (
            type(adoption) is not RunActionWorkloadReleaseAdoption
            or type(terminal) is not RunActionTerminalObservation
            or self.pre_release_main_loss_observation is not None
            or self.credential_retirement_intent is not None
            or adoption.workload_release_receipt.activation_event_id
            != self.activation_event_id
        ):
            raise RunActionTerminationContractError(
                "released provider termination lacks its exact terminal occurrence"
            )
        activation = (
            adoption.workload_release_receipt.resolved_workload_observation.activation_revalidation_receipt
        )
        if not _released_terminal_evidence_matches(
            terminal,
            activation,
            adoption,
        ):
            raise RunActionTerminationContractError(
                "released provider termination lacks its exact terminal occurrence"
            )
        if self.reason is RunActionProviderTerminationReason.TIMEOUT:
            if type(
                self.timeout_directive_publication
            ) is not RunActionTimeoutDirectivePublicationReceipt or not run_action_timeout_publication_evidence_matches(
                self.timeout_directive_publication,
                self.activation_event_id,
                activation,
                adoption,
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
            valid_outcome = terminal.oom_killed is True
        elif self.reason is RunActionProviderTerminationReason.NONZERO_EXIT:
            valid_outcome = terminal.oom_killed is False and terminal.exit_code != 0
        else:
            valid_outcome = (
                self.reason is RunActionProviderTerminationReason.MISSING_RESULT
                and terminal.oom_killed is False
                and terminal.exit_code == 0
            )
        if not valid_outcome:
            raise RunActionTerminationContractError(
                "provider termination reason differs from terminal evidence"
            )

    @property
    def activation_revalidation_receipt(
        self,
    ) -> RunActionActivationRevalidationReceipt:
        """Derive activation evidence from the selected termination branch."""

        if self.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS:
            loss = self.pre_release_main_loss_observation
            if type(loss) is not RunActionPreReleaseMainLossObservation:
                raise RunActionTerminationContractError(
                    "pre-release termination lacks activation evidence"
                )
            return loss.activation_revalidation_receipt
        if self.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL:
            terminal = self.terminal_observation
            if type(terminal) is not RunActionPreReleaseMainTerminalObservation:
                raise RunActionTerminationContractError(
                    "pre-release termination lacks activation evidence"
                )
            return terminal.activation_revalidation_receipt
        if self.reason is RunActionProviderTerminationReason.CREDENTIAL_EXPIRED:
            loss = self.pre_release_main_loss_observation
            terminal = self.terminal_observation
            if type(loss) is RunActionPreReleaseMainLossObservation:
                return loss.activation_revalidation_receipt
            if type(terminal) is RunActionPreReleaseMainTerminalObservation:
                return terminal.activation_revalidation_receipt
            raise RunActionTerminationContractError(
                "credential-expired termination lacks activation evidence"
            )
        adoption = self.workload_release_adoption
        if type(adoption) is not RunActionWorkloadReleaseAdoption:
            raise RunActionTerminationContractError(
                "released termination lacks activation evidence"
            )
        return (
            adoption.workload_release_receipt.resolved_workload_observation.activation_revalidation_receipt
        )


def provider_termination_matches_durable_activation(
    receipt: RunActionProviderTerminationReceipt,
    activation_event_id: str,
    preparation_allocation: RunActionPreparationAllocation,
    activation_revalidation_receipt: RunActionActivationRevalidationReceipt,
) -> bool:
    """Join one complete termination graph to durable events 2 and 5."""

    if (
        type(receipt) is not RunActionProviderTerminationReceipt
        or type(activation_event_id) is not str
        or type(preparation_allocation) is not RunActionPreparationAllocation
        or type(activation_revalidation_receipt)
        is not RunActionActivationRevalidationReceipt
    ):
        return False
    prepared = activation_revalidation_receipt.prepared_execution
    if (
        receipt.activation_event_id != activation_event_id
        or receipt.activation_revalidation_receipt != activation_revalidation_receipt
        or preparation_allocation.preparation_claim != prepared.preparation_claim
        or preparation_allocation.runtime_volume_authority
        != prepared.runtime_volume_authority
    ):
        return False
    if receipt.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS:
        loss = receipt.pre_release_main_loss_observation
        return (
            type(loss) is RunActionPreReleaseMainLossObservation
            and loss.activation_event_id == activation_event_id
            and loss.preparation_allocation == preparation_allocation
        )
    if receipt.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL:
        terminal = receipt.terminal_observation
        return (
            type(terminal) is RunActionPreReleaseMainTerminalObservation
            and terminal.activation_event_id == activation_event_id
            and terminal.preparation_allocation == preparation_allocation
        )
    if receipt.reason is RunActionProviderTerminationReason.CREDENTIAL_EXPIRED:
        intent = receipt.credential_retirement_intent
        physical = (
            receipt.pre_release_main_loss_observation
            if receipt.pre_release_main_loss_observation is not None
            else receipt.terminal_observation
        )
        return (
            credential_retirement_intent_matches_activation(
                intent,
                activation_event_id,
                activation_revalidation_receipt,
            )
            and type(physical)
            in {
                RunActionPreReleaseMainLossObservation,
                RunActionPreReleaseMainTerminalObservation,
            }
            and physical.preparation_allocation == preparation_allocation
            and physical.activation_event_id == activation_event_id
        )
    adoption = receipt.workload_release_adoption
    return (
        type(adoption) is RunActionWorkloadReleaseAdoption
        and adoption.workload_release_receipt.activation_event_id == activation_event_id
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


def run_action_running_container_occurrence_matches(
    observed: RunActionBarrierRunningContainerObservation,
    expected: RunActionBarrierRunningContainerObservation,
) -> bool:
    """Match two fresh observations of one exact running occurrence."""

    return (
        type(observed) is RunActionBarrierRunningContainerObservation
        and type(expected) is RunActionBarrierRunningContainerObservation
        and observed.container_id == expected.container_id
        and observed.observed_inspect_projection == expected.observed_inspect_projection
        and observed.init_process_id == expected.init_process_id
        and observed.started_at == expected.started_at
    )


def run_action_timeout_directive_evidence_matches(
    directive: RunActionTimeoutDirective,
    activation_event_id: str,
    activation: RunActionActivationRevalidationReceipt,
    adoption: RunActionWorkloadReleaseAdoption,
) -> bool:
    """Join one timeout directive to its exact event-5 release occurrence."""

    if (
        type(directive) is not RunActionTimeoutDirective
        or type(activation_event_id) is not str
        or type(activation) is not RunActionActivationRevalidationReceipt
        or type(adoption) is not RunActionWorkloadReleaseAdoption
    ):
        return False
    release = adoption.workload_release_receipt
    prepared = activation.prepared_execution
    released_running = (
        release.resolved_workload_observation.running_container_observation
    )
    timeout_running = directive.running_container_observation
    return (
        release.activation_event_id == activation_event_id
        and release.resolved_workload_observation.activation_revalidation_receipt
        == activation
        and directive.activation_event_id == activation_event_id
        and directive.workload_release_receipt_id == release.workload_release_receipt_id
        and directive.workload_release_adoption_id
        == adoption.workload_release_adoption_id
        and directive.host_boot_id == release.host_boot_id
        and directive.execution_deadline_boottime_nanoseconds
        == release.execution_deadline_boottime_nanoseconds
        and directive.containment_deadline_boottime_nanoseconds
        == release.containment_deadline_boottime_nanoseconds
        and directive.observed_before_boottime_nanoseconds
        >= release.execution_deadline_boottime_nanoseconds
        and timeout_running.container_id
        == activation.spawn_commit.provider_execution_id
        and timeout_running.observed_inspect_projection
        == prepared.inert_container_evidence.issued_create_projection
        and run_action_running_container_occurrence_matches(
            timeout_running,
            released_running,
        )
    )


def run_action_timeout_publication_evidence_matches(
    publication: RunActionTimeoutDirectivePublicationReceipt,
    activation_event_id: str,
    activation: RunActionActivationRevalidationReceipt,
    adoption: RunActionWorkloadReleaseAdoption,
) -> bool:
    """Join one adopted timeout inode to its exact activation and release."""

    if (
        type(publication) is not RunActionTimeoutDirectivePublicationReceipt
        or type(activation_event_id) is not str
        or type(activation) is not RunActionActivationRevalidationReceipt
        or type(adoption) is not RunActionWorkloadReleaseAdoption
    ):
        return False
    directive = publication.timeout_directive
    prepared = activation.prepared_execution
    control = prepared.control_directory
    authority = prepared.runtime_volume_authority
    return (
        run_action_timeout_directive_evidence_matches(
            directive,
            activation_event_id,
            activation,
            adoption,
        )
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
        and 0
        < publication.size_bytes
        <= prepared.preparation_claim.execution_policy.supervisor_limits.timeout_directive_size_bytes
    )


def _require_namespaced_content_id(
    value: str,
    namespace: str,
    name: str,
) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionTerminationContractError(f"{name} uses another namespace")


def _docker_timestamp_order_key(value: object) -> tuple[int, ...] | None:
    if type(value) is not str:
        return None
    match = _DOCKER_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    fraction = match.group("fraction") or ""
    leap_year = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    month_lengths = (
        31,
        29 if leap_year else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    if (
        year <= 0
        or not 1 <= month <= len(month_lengths)
        or not 1 <= day <= month_lengths[month - 1]
        or not 0 <= hour < 24
        or not 0 <= minute < 60
        or not 0 <= second < 60
    ):
        return None
    return (
        year,
        month,
        day,
        hour,
        minute,
        second,
        int(fraction.ljust(9, "0")) if fraction else 0,
    )


__all__ = [
    "run_action_pre_release_main_loss_observation_token",
    "run_action_pre_release_main_terminal_observation_token",
    "RunActionPreReleaseMainLossObservation",
    "RunActionPreReleaseMainTerminalObservation",
    "RunActionPreReleaseTerminalContainerObservation",
    "RunActionProviderTerminationDisposition",
    "RunActionProviderTerminationReason",
    "RunActionProviderTerminationReceipt",
    "RunActionTerminationContractError",
    "RunActionTimeoutDirective",
    "RunActionTimeoutDirectivePublicationReceipt",
    "provider_termination_matches_durable_activation",
    "run_action_running_container_occurrence_matches",
    "run_action_timeout_directive_evidence_matches",
    "run_action_timeout_publication_evidence_matches",
]
