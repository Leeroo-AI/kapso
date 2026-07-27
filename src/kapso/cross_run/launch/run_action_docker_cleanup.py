"""Durably eligible, role-proved cleanup of terminal run-action resources."""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass, replace
from threading import get_ident, Lock
from weakref import WeakKeyDictionary

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    _DOCKER_CLEANUP_EXCLUSION_ISSUANCE,
    _DOCKER_CLEANUP_REMOVE_AUTHORITY,
    _docker_observation_and_cleanup_authorities_share_runtime,
    PinnedDockerCleanupAuthority,
    PinnedDockerCleanupExclusionLease,
    PinnedDockerRuntime,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionInertKeeperObservation,
    observe_allocation_inert_main_container,
    observe_allocation_keeper,
    observe_inert_main_container,
    observe_running_keeper,
    observe_runtime_volume,
    reobserve_pre_release_terminal_main_container_for_cleanup,
    reobserve_terminal_main_container_for_cleanup,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    target_command_from_main_projection,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    _run_action_observation_authority,
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionWorkloadReleaseAdoption,
)
from kapso.cross_run.launch.run_action_resource_finalization import (
    _issue_run_action_resource_finalization_authority,
    RunActionResourceFinalizationAuthority,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    open_run_action_control_directory,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_RECOVERY_AUTHORITY,
    RunActionExecutionEvent,
    RunActionExecutionStore,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    observe_docker_init_source,
    observe_supervisor_helper,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
    RunActionTerminalObservation,
    RunActionVolumeKeeperEvidence,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionPreReleaseMainTerminalObservation,
    RunActionProviderTerminationReason,
    RunActionTimeoutDirectivePublicationReceipt,
)
from kapso.cross_run.launch.run_action_timeout_adoption import (
    open_run_action_timeout_inspection,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings

_CLEANUP_MANAGER_LOCK = Lock()
_CLEANUP_MANAGER_AUTHORITIES: WeakKeyDictionary[
    DockerRunActionCleanupManager, PinnedDockerCleanupAuthority
] = WeakKeyDictionary()
_CLEANUP_EXCLUSION_LOCK = Lock()
_CLEANUP_EXCLUSION_SESSIONS: WeakKeyDictionary[
    PinnedDockerCleanupExclusionLease, object
] = WeakKeyDictionary()


class RunActionDockerCleanupError(RuntimeError):
    """Run-action Docker cleanup lacks exact durable or physical authority."""


@dataclass(frozen=True)
class _RunActionTerminalCleanupEvidence:
    allocation: RunActionPreparationAllocation
    prepared: RunActionPreparedExecution
    activation_event: RunActionExecutionEvent
    topology: RunActionControlDirectoryTopology
    workload_release_adoption: RunActionWorkloadReleaseAdoption | None
    timeout_directive_publication: RunActionTimeoutDirectivePublicationReceipt | None
    terminal_observation: (
        RunActionTerminalObservation | RunActionPreReleaseMainTerminalObservation | None
    )
    main_must_be_absent: bool


class DockerRunActionCleanupManager:
    """Removal-only Docker authority with no cleanup eligibility of its own."""

    def __init__(self, runtime: PinnedDockerRuntime) -> None:
        if type(runtime) is not PinnedDockerRuntime:
            raise RunActionDockerCleanupError(
                "run-action cleanup requires one pinned Docker runtime"
            )
        authority = runtime.issue_cleanup_authority()
        with _CLEANUP_MANAGER_LOCK:
            if _CLEANUP_MANAGER_AUTHORITIES.get(self) is not None:
                raise RunActionDockerCleanupError(
                    "run-action cleanup manager is already issued"
                )
            _CLEANUP_MANAGER_AUTHORITIES[self] = authority

    @property
    def runtime_settings(self) -> DockerRuntimeSettings:
        """Return settings from the exact issued cleanup authority."""

        return _cleanup_authority(self).settings


class _DockerRunActionResourceFinalizationDriver:
    """Finalize durable Docker terminals and seal later absence checks."""

    def __init__(
        self,
        *,
        action_store: RunActionExecutionStore,
        launch_settings: LaunchSettings,
        resource_manager: DockerRunActionResourceManager,
        cleanup_manager: DockerRunActionCleanupManager,
    ) -> None:
        if (
            type(action_store) is not RunActionExecutionStore
            or type(launch_settings) is not LaunchSettings
            or type(resource_manager) is not DockerRunActionResourceManager
            or type(cleanup_manager) is not DockerRunActionCleanupManager
        ):
            raise RunActionDockerCleanupError(
                "Docker finalization requires exact controller authorities"
            )
        if resource_manager.runtime_settings != cleanup_manager.runtime_settings:
            raise RunActionDockerCleanupError(
                "Docker finalization managers name different runtimes"
            )
        self._action_store = action_store
        self._launch_settings = launch_settings
        self._resource_manager = resource_manager
        self._cleanup_manager = cleanup_manager
        self._owner_process_id = os.getpid()

    def finalize_terminal(self, operation_id: str) -> None:
        """Reap eligible terminals and prove every terminal physically closed."""

        self._with_terminal_session(operation_id, reap=True)

    def require_terminal_absence(self, operation_id: str) -> None:
        """Independently reprove that one durable terminal owns no resources."""

        self._with_terminal_session(operation_id, reap=False)

    def _with_terminal_session(self, operation_id: str, *, reap: bool) -> None:
        self._require_owner_process()
        if type(operation_id) is not str or not operation_id or type(reap) is not bool:
            raise RunActionDockerCleanupError(
                "Docker finalization requires one exact terminal operation"
            )
        events = self._action_store.inspect().events_for(operation_id)
        with self._action_store._recovery_session(
            events[0].reservation,
            _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
        ) as session:
            if session.events != events:
                raise RunActionDockerCleanupError(
                    "run-action terminal changed before resource finalization"
                )
            tail_kind = events[-1].event_kind
            if tail_kind is RunActionExecutionEventKind.CANCELLED:
                return
            if tail_kind is RunActionExecutionEventKind.FRONTIER_INVALIDATED:
                allocation, prepared = _invalidated_frontier_evidence(events)
                if prepared is None and reap:
                    self._reap_invalidated_allocation(session, allocation)
                elif prepared is None or not reap:
                    self._require_allocation_absent(session, allocation)
                else:
                    self._reap_invalidated_prepared(
                        session,
                        allocation,
                        prepared,
                    )
                return
            evidence = _terminal_cleanup_evidence(events)
            _require_cleanup_runtime(
                evidence,
                self._resource_manager,
                self._cleanup_manager,
            )
            with ExitStack() as protections:
                exclusion = _issue_cleanup_exclusion(
                    session,
                    self._cleanup_manager,
                )
                protections.callback(
                    _unregister_cleanup_exclusion,
                    exclusion,
                    session,
                )
                protections.enter_context(exclusion)
                if reap:
                    _reap_terminal_resources_locked(
                        session=session,
                        evidence=evidence,
                        launch_settings=self._launch_settings,
                        resource_manager=self._resource_manager,
                        cleanup_manager=self._cleanup_manager,
                        exclusion=exclusion,
                    )
                else:
                    inventory = self._resource_manager.observe(evidence.allocation)
                    _require_cleanup_suffix(inventory)
                    _require_stable_absence(
                        session=session,
                        inventory=inventory,
                        resource_manager=self._resource_manager,
                        cleanup_manager=self._cleanup_manager,
                        exclusion=exclusion,
                    )

    def _require_allocation_absent(
        self,
        session,
        allocation: RunActionPreparationAllocation,
    ) -> None:
        _require_cleanup_runtime_for_allocation(
            allocation,
            self._resource_manager,
            self._cleanup_manager,
        )
        with ExitStack() as protections:
            exclusion = _issue_cleanup_exclusion(
                session,
                self._cleanup_manager,
            )
            protections.callback(
                _unregister_cleanup_exclusion,
                exclusion,
                session,
            )
            protections.enter_context(exclusion)
            inventory = self._resource_manager.observe(allocation)
            _require_cleanup_suffix(inventory)
            _require_stable_absence(
                session=session,
                inventory=inventory,
                resource_manager=self._resource_manager,
                cleanup_manager=self._cleanup_manager,
                exclusion=exclusion,
            )

    def _reap_invalidated_allocation(
        self,
        session,
        allocation: RunActionPreparationAllocation,
    ) -> None:
        _require_cleanup_runtime_for_allocation(
            allocation,
            self._resource_manager,
            self._cleanup_manager,
        )
        with ExitStack() as protections:
            exclusion = _issue_cleanup_exclusion(
                session,
                self._cleanup_manager,
            )
            protections.callback(
                _unregister_cleanup_exclusion,
                exclusion,
                session,
            )
            protections.enter_context(exclusion)
            inventory = self._resource_manager.observe(allocation)
            _require_cleanup_suffix(inventory)
            if inventory.is_absent:
                _require_stable_absence(
                    session=session,
                    inventory=inventory,
                    resource_manager=self._resource_manager,
                    cleanup_manager=self._cleanup_manager,
                    exclusion=exclusion,
                )
                return
            volume = _prove_allocation_volume_occurrence(
                inventory,
                self._resource_manager,
            )
            if inventory.keeper_container_id is not None:
                helper_evidence = observe_supervisor_helper(
                    allocation.preparation_claim.execution_policy
                )
                init_source_evidence = observe_docker_init_source(
                    allocation.preparation_claim.execution_policy
                )
                keeper = _prove_allocation_keeper_occurrence(
                    inventory,
                    self._resource_manager,
                    volume,
                    helper_evidence,
                    init_source_evidence,
                )
                if inventory.main_container_id is not None:
                    if type(keeper) is not RunActionVolumeKeeperEvidence:
                        raise RunActionDockerCleanupError(
                            "allocation-stage main lacks one exact running keeper"
                        )
                    _prove_allocation_inert_main_occurrence(
                        inventory,
                        self._resource_manager,
                        volume,
                        helper_evidence,
                        init_source_evidence,
                    )
                    inventory = _remove_main_once(
                        session=session,
                        inventory=inventory,
                        resource_manager=self._resource_manager,
                        cleanup_manager=self._cleanup_manager,
                        exclusion=exclusion,
                    )
                    volume = _prove_allocation_volume_occurrence(
                        inventory,
                        self._resource_manager,
                    )
                    keeper = _prove_allocation_keeper_occurrence(
                        inventory,
                        self._resource_manager,
                        volume,
                        helper_evidence,
                        init_source_evidence,
                    )
                    if type(keeper) is not RunActionVolumeKeeperEvidence:
                        raise RunActionDockerCleanupError(
                            "allocation-stage keeper changed after main removal"
                        )
                if type(keeper) is DockerRunActionInertKeeperObservation:
                    inventory = _remove_inert_keeper_once(
                        session=session,
                        inventory=inventory,
                        resource_manager=self._resource_manager,
                        cleanup_manager=self._cleanup_manager,
                        exclusion=exclusion,
                    )
                elif type(keeper) is RunActionVolumeKeeperEvidence:
                    inventory = _remove_keeper_once(
                        session=session,
                        inventory=inventory,
                        resource_manager=self._resource_manager,
                        cleanup_manager=self._cleanup_manager,
                        exclusion=exclusion,
                    )
                else:
                    raise RunActionDockerCleanupError(
                        "allocation-stage keeper lifecycle is not removable"
                    )
                volume = _prove_allocation_volume_occurrence(
                    inventory,
                    self._resource_manager,
                )
            if inventory.volume_present:
                if (
                    volume.volume_name
                    != allocation.runtime_volume_authority.volume_name
                ):
                    raise RunActionDockerCleanupError(
                        "allocation-stage volume changed before removal"
                    )
                inventory = _remove_volume_once(
                    session=session,
                    inventory=inventory,
                    resource_manager=self._resource_manager,
                    cleanup_manager=self._cleanup_manager,
                    exclusion=exclusion,
                )
            _require_stable_absence(
                session=session,
                inventory=inventory,
                resource_manager=self._resource_manager,
                cleanup_manager=self._cleanup_manager,
                exclusion=exclusion,
            )

    def _reap_invalidated_prepared(
        self,
        session,
        allocation: RunActionPreparationAllocation,
        prepared: RunActionPreparedExecution,
    ) -> None:
        _require_cleanup_runtime_for_prepared(
            allocation,
            prepared,
            self._resource_manager,
            self._cleanup_manager,
        )
        with ExitStack() as protections:
            exclusion = _issue_cleanup_exclusion(
                session,
                self._cleanup_manager,
            )
            protections.callback(
                _unregister_cleanup_exclusion,
                exclusion,
                session,
            )
            protections.enter_context(exclusion)
            inventory = self._resource_manager.observe(allocation)
            _require_cleanup_suffix(inventory)
            if inventory.is_absent:
                _require_stable_absence(
                    session=session,
                    inventory=inventory,
                    resource_manager=self._resource_manager,
                    cleanup_manager=self._cleanup_manager,
                    exclusion=exclusion,
                )
                return
            if inventory.keeper_container_id is not None:
                with open_run_action_control_directory(prepared) as control_lease:
                    if (
                        control_lease.topology
                        is not RunActionControlDirectoryTopology.EMPTY
                    ):
                        raise RunActionDockerCleanupError(
                            "invalidated prepared action changed its control topology"
                        )
                    volume = _prove_prepared_volume_occurrence(
                        inventory,
                        prepared,
                        self._resource_manager,
                    )
                    _prove_prepared_keeper_occurrence(
                        inventory,
                        prepared,
                        self._resource_manager,
                        volume,
                    )
                    control_lease.require_current()
                    if inventory.main_container_id is not None:
                        _prove_inert_main_occurrence(
                            inventory,
                            prepared,
                            self._resource_manager,
                            volume,
                        )
                        control_lease.require_current()
                        inventory = _remove_main_once(
                            session=session,
                            inventory=inventory,
                            resource_manager=self._resource_manager,
                            cleanup_manager=self._cleanup_manager,
                            exclusion=exclusion,
                        )
                        volume = _prove_prepared_volume_occurrence(
                            inventory,
                            prepared,
                            self._resource_manager,
                        )
                        _prove_prepared_keeper_occurrence(
                            inventory,
                            prepared,
                            self._resource_manager,
                            volume,
                        )
                        control_lease.require_current()
                    inventory = _remove_keeper_once(
                        session=session,
                        inventory=inventory,
                        resource_manager=self._resource_manager,
                        cleanup_manager=self._cleanup_manager,
                        exclusion=exclusion,
                    )
            if inventory.volume_present:
                _prove_prepared_volume_occurrence(
                    inventory,
                    prepared,
                    self._resource_manager,
                )
                inventory = _remove_volume_once(
                    session=session,
                    inventory=inventory,
                    resource_manager=self._resource_manager,
                    cleanup_manager=self._cleanup_manager,
                    exclusion=exclusion,
                )
            _require_stable_absence(
                session=session,
                inventory=inventory,
                resource_manager=self._resource_manager,
                cleanup_manager=self._cleanup_manager,
                exclusion=exclusion,
            )

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise RunActionDockerCleanupError(
                "Docker finalization authority cannot cross a process boundary"
            )


def issue_docker_run_action_resource_finalization_authority(
    *,
    action_store: RunActionExecutionStore,
    launch_settings: LaunchSettings,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
) -> RunActionResourceFinalizationAuthority:
    """Issue the sealed controller authority backed by one pinned Docker runtime."""

    driver = _DockerRunActionResourceFinalizationDriver(
        action_store=action_store,
        launch_settings=launch_settings,
        resource_manager=resource_manager,
        cleanup_manager=cleanup_manager,
    )
    return _issue_run_action_resource_finalization_authority(
        action_store=action_store,
        launch_settings=launch_settings,
        driver=driver,
    )


def _reap_terminal_resources_locked(
    *,
    session,
    evidence: _RunActionTerminalCleanupEvidence,
    launch_settings: LaunchSettings,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
    exclusion: PinnedDockerCleanupExclusionLease,
) -> DockerRunActionResourceInventory:
    inventory = resource_manager.observe(evidence.allocation)
    _require_cleanup_suffix(inventory)
    if inventory.is_absent:
        return _require_stable_absence(
            session=session,
            inventory=inventory,
            resource_manager=resource_manager,
            cleanup_manager=cleanup_manager,
            exclusion=exclusion,
        )
    if evidence.main_must_be_absent and inventory.main_container_id is not None:
        raise RunActionDockerCleanupError(
            "pre-release main reappeared after durable loss termination"
        )
    if inventory.keeper_container_id is not None:
        with open_run_action_timeout_inspection(
            activation_event=evidence.activation_event,
            launch_settings=launch_settings,
        ) as control_inspection:
            _require_control_evidence(control_inspection, evidence)
            volume = _prove_volume_occurrence(
                inventory,
                evidence,
                resource_manager,
            )
            _prove_keeper_occurrence(
                inventory,
                evidence,
                resource_manager,
                volume,
            )
            control_inspection.require_current()
            if inventory.main_container_id is not None:
                _prove_terminal_main_occurrence(
                    inventory,
                    evidence,
                    resource_manager,
                )
                control_inspection.require_current()
                inventory = _remove_main_once(
                    session=session,
                    inventory=inventory,
                    resource_manager=resource_manager,
                    cleanup_manager=cleanup_manager,
                    exclusion=exclusion,
                )
                volume = _prove_volume_occurrence(
                    inventory,
                    evidence,
                    resource_manager,
                )
                _prove_keeper_occurrence(
                    inventory,
                    evidence,
                    resource_manager,
                    volume,
                )
                control_inspection.require_current()
            inventory = _remove_keeper_once(
                session=session,
                inventory=inventory,
                resource_manager=resource_manager,
                cleanup_manager=cleanup_manager,
                exclusion=exclusion,
            )
    if inventory.volume_present:
        _prove_volume_occurrence(
            inventory,
            evidence,
            resource_manager,
        )
        inventory = _remove_volume_once(
            session=session,
            inventory=inventory,
            resource_manager=resource_manager,
            cleanup_manager=cleanup_manager,
            exclusion=exclusion,
        )
    return _require_stable_absence(
        session=session,
        inventory=inventory,
        resource_manager=resource_manager,
        cleanup_manager=cleanup_manager,
        exclusion=exclusion,
    )


def _terminal_cleanup_evidence(
    events: tuple[RunActionExecutionEvent, ...],
) -> _RunActionTerminalCleanupEvidence:
    if (
        type(events) is not tuple
        or len(events) < 6
        or any(type(event) is not RunActionExecutionEvent for event in events)
        or events[1].event_kind is not RunActionExecutionEventKind.PREPARATION_ALLOCATED
        or events[2].event_kind is not RunActionExecutionEventKind.EXECUTION_PREPARED
        or events[4].event_kind is not RunActionExecutionEventKind.ACTIVATION_COMMITTED
    ):
        raise RunActionDockerCleanupError(
            "run-action cleanup lacks a complete durable execution prefix"
        )
    allocation = events[1].preparation_allocation
    prepared = events[2].prepared_execution
    activation_event = events[4]
    if (
        type(allocation) is not RunActionPreparationAllocation
        or type(prepared) is not RunActionPreparedExecution
        or activation_event.activation_revalidation_receipt.prepared_execution
        != prepared
    ):
        raise RunActionDockerCleanupError(
            "run-action cleanup durable execution graph is spliced"
        )
    tail = events[-1]
    if (
        tail.event_kind is RunActionExecutionEventKind.RESULT_ACCEPTED
        and len(events) == 8
    ):
        result = events[5].result_receipt
        return _RunActionTerminalCleanupEvidence(
            allocation=allocation,
            prepared=prepared,
            activation_event=activation_event,
            topology=RunActionControlDirectoryTopology.RELEASED,
            workload_release_adoption=result.workload_release_adoption,
            timeout_directive_publication=None,
            terminal_observation=result.terminal_observation,
            main_must_be_absent=False,
        )
    provider_terminated_at_event_6 = (
        len(events) == 6
        and tail.event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED
        and tail.provider_termination_receipt.reason
        is not RunActionProviderTerminationReason.CREDENTIAL_EXPIRED
    )
    credential_expired_at_event_7 = (
        len(events) == 7
        and events[5].event_kind
        is RunActionExecutionEventKind.CREDENTIAL_RETIREMENT_REQUESTED
        and tail.event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED
        and tail.provider_termination_receipt.reason
        is RunActionProviderTerminationReason.CREDENTIAL_EXPIRED
        and tail.provider_termination_receipt.credential_retirement_intent
        == events[5].credential_retirement_intent
    )
    if not (provider_terminated_at_event_6 or credential_expired_at_event_7):
        raise RunActionDockerCleanupError(
            "run-action cleanup is not durably terminal and eligible"
        )
    termination = tail.provider_termination_receipt
    pre_release_loss = (
        termination.reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
    )
    pre_release_termination = termination.reason in {
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL,
        RunActionProviderTerminationReason.CREDENTIAL_EXPIRED,
    }
    topology = (
        RunActionControlDirectoryTopology.EMPTY
        if pre_release_termination
        else (
            RunActionControlDirectoryTopology.TIMED_OUT
            if termination.reason is RunActionProviderTerminationReason.TIMEOUT
            else RunActionControlDirectoryTopology.RELEASED
        )
    )
    return _RunActionTerminalCleanupEvidence(
        allocation=allocation,
        prepared=prepared,
        activation_event=activation_event,
        topology=topology,
        workload_release_adoption=termination.workload_release_adoption,
        timeout_directive_publication=termination.timeout_directive_publication,
        terminal_observation=termination.terminal_observation,
        main_must_be_absent=pre_release_loss,
    )


def _require_cleanup_runtime(
    evidence: _RunActionTerminalCleanupEvidence,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
) -> None:
    _require_cleanup_runtime_for_prepared(
        evidence.allocation,
        evidence.prepared,
        resource_manager,
        cleanup_manager,
    )


def _require_cleanup_runtime_for_prepared(
    allocation: RunActionPreparationAllocation,
    prepared: RunActionPreparedExecution,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
) -> None:
    cleanup_authority = _cleanup_authority(cleanup_manager)
    observation_authority = _run_action_observation_authority(resource_manager)
    settings = cleanup_authority.settings
    if (
        type(allocation) is not RunActionPreparationAllocation
        or type(prepared) is not RunActionPreparedExecution
        or prepared.preparation_claim != allocation.preparation_claim
        or prepared.runtime_volume_authority != allocation.runtime_volume_authority
        or resource_manager.runtime_settings != settings
        or prepared.preparation_claim.execution_policy.docker_runtime_settings_digest
        != tree_or_blob_digest(settings.to_json_bytes())
        or not _docker_observation_and_cleanup_authorities_share_runtime(
            observation_authority,
            cleanup_authority,
        )
    ):
        raise RunActionDockerCleanupError(
            "run-action cleanup inputs do not share durable Docker authority"
        )


def _require_cleanup_runtime_for_allocation(
    allocation: RunActionPreparationAllocation,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
) -> None:
    cleanup_authority = _cleanup_authority(cleanup_manager)
    observation_authority = _run_action_observation_authority(resource_manager)
    settings = cleanup_authority.settings
    if (
        type(allocation) is not RunActionPreparationAllocation
        or resource_manager.runtime_settings != settings
        or allocation.preparation_claim.execution_policy.docker_runtime_settings_digest
        != tree_or_blob_digest(settings.to_json_bytes())
        or not _docker_observation_and_cleanup_authorities_share_runtime(
            observation_authority,
            cleanup_authority,
        )
    ):
        raise RunActionDockerCleanupError(
            "invalidated run-action resources lack durable Docker authority"
        )


def _invalidated_frontier_evidence(
    events: tuple[RunActionExecutionEvent, ...],
) -> tuple[RunActionPreparationAllocation, RunActionPreparedExecution | None]:
    if (
        type(events) is not tuple
        or len(events) not in {3, 4}
        or events[-1].event_kind is not RunActionExecutionEventKind.FRONTIER_INVALIDATED
        or events[1].event_kind is not RunActionExecutionEventKind.PREPARATION_ALLOCATED
        or type(events[1].preparation_allocation) is not RunActionPreparationAllocation
        or (
            len(events) == 4
            and events[2].event_kind
            is not RunActionExecutionEventKind.EXECUTION_PREPARED
        )
    ):
        raise RunActionDockerCleanupError(
            "invalidated frontier lacks one pre-spawn allocation"
        )
    prepared = None if len(events) == 3 else events[2].prepared_execution
    if prepared is not None and type(prepared) is not RunActionPreparedExecution:
        raise RunActionDockerCleanupError(
            "invalidated frontier lacks its durable prepared occurrence"
        )
    return events[1].preparation_allocation, prepared


def _require_control_evidence(
    control_inspection,
    evidence: _RunActionTerminalCleanupEvidence,
) -> None:
    if (
        control_inspection.topology is not evidence.topology
        or control_inspection.workload_release_adoption
        != evidence.workload_release_adoption
        or control_inspection.timeout_directive_publication
        != evidence.timeout_directive_publication
    ):
        raise RunActionDockerCleanupError(
            "run-action cleanup control occurrence differs from terminal evidence"
        )


def _prove_volume_occurrence(
    inventory: DockerRunActionResourceInventory,
    evidence: _RunActionTerminalCleanupEvidence,
    resource_manager: DockerRunActionResourceManager,
):
    return _prove_prepared_volume_occurrence(
        inventory,
        evidence.prepared,
        resource_manager,
    )


def _prove_prepared_volume_occurrence(
    inventory: DockerRunActionResourceInventory,
    prepared: RunActionPreparedExecution,
    resource_manager: DockerRunActionResourceManager,
):
    if not inventory.volume_present:
        raise RunActionDockerCleanupError(
            "run-action cleanup keeper lacks its durable volume"
        )
    raw = resource_manager.inspect_volume(inventory)
    observed = observe_runtime_volume(
        raw,
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        resource_manager.runtime_settings,
    )
    if (
        observed.volume_occurrence_digest
        != prepared.runtime_volume_evidence.docker_volume_occurrence_digest
    ):
        raise RunActionDockerCleanupError(
            "run-action cleanup volume differs from its durable occurrence"
        )
    return observed


def _prove_allocation_volume_occurrence(
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
):
    if not inventory.volume_present:
        raise RunActionDockerCleanupError(
            "allocation-stage Docker resource lacks its exact volume"
        )
    allocation = inventory.preparation_allocation
    return observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        resource_manager.runtime_settings,
    )


def _prove_allocation_keeper_occurrence(
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    volume,
    helper_evidence,
    init_source_evidence,
) -> DockerRunActionInertKeeperObservation | RunActionVolumeKeeperEvidence:
    if inventory.keeper_container_id is None:
        raise RunActionDockerCleanupError(
            "allocation-stage Docker volume lacks its keeper"
        )
    allocation = inventory.preparation_allocation
    return observe_allocation_keeper(
        resource_manager.inspect_keeper(inventory),
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        volume,
        helper_evidence,
        init_source_evidence,
        resource_manager.runtime_settings,
    )


def _prove_allocation_inert_main_occurrence(
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    volume,
    helper_evidence,
    init_source_evidence,
) -> None:
    if inventory.main_container_id is None:
        raise RunActionDockerCleanupError(
            "allocation-stage Docker keeper lacks its main"
        )
    allocation = inventory.preparation_allocation
    observed = observe_allocation_inert_main_container(
        resource_manager.inspect_main(inventory),
        allocation.preparation_claim,
        allocation.runtime_volume_authority,
        volume,
        helper_evidence,
        init_source_evidence,
        resource_manager.runtime_settings,
    )
    if observed.container_id != inventory.main_container_id:
        raise RunActionDockerCleanupError(
            "allocation-stage main differs from its exact inventory"
        )


def _prove_keeper_occurrence(
    inventory: DockerRunActionResourceInventory,
    evidence: _RunActionTerminalCleanupEvidence,
    resource_manager: DockerRunActionResourceManager,
    volume,
) -> None:
    _prove_prepared_keeper_occurrence(
        inventory,
        evidence.prepared,
        resource_manager,
        volume,
    )


def _prove_prepared_keeper_occurrence(
    inventory: DockerRunActionResourceInventory,
    prepared: RunActionPreparedExecution,
    resource_manager: DockerRunActionResourceManager,
    volume,
) -> None:
    if inventory.keeper_container_id is None:
        raise RunActionDockerCleanupError(
            "run-action cleanup volume lacks its durable keeper"
        )
    projection = prepared.volume_keeper_evidence.issued_create_projection
    observed = observe_running_keeper(
        resource_manager.inspect_keeper(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        projection.helper_evidence,
        projection.docker_init_source_evidence,
        resource_manager.runtime_settings,
    )
    if observed != prepared.volume_keeper_evidence:
        raise RunActionDockerCleanupError(
            "run-action cleanup keeper differs from its durable occurrence"
        )


def _prove_inert_main_occurrence(
    inventory: DockerRunActionResourceInventory,
    prepared: RunActionPreparedExecution,
    resource_manager: DockerRunActionResourceManager,
    volume,
) -> None:
    if inventory.main_container_id is None:
        raise RunActionDockerCleanupError(
            "invalidated prepared action lacks its inert main"
        )
    projection = prepared.inert_container_evidence.issued_create_projection
    command = target_command_from_main_projection(projection)
    observed = observe_inert_main_container(
        resource_manager.inspect_main(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        command,
        projection.supervisor_helper_evidence,
        projection.docker_init_source_evidence,
        resource_manager.runtime_settings,
    )
    if observed != prepared.inert_container_evidence:
        raise RunActionDockerCleanupError(
            "invalidated main differs from its durable inert occurrence"
        )


def _prove_terminal_main_occurrence(
    inventory: DockerRunActionResourceInventory,
    evidence: _RunActionTerminalCleanupEvidence,
    resource_manager: DockerRunActionResourceManager,
) -> None:
    terminal = evidence.terminal_observation
    if type(terminal) is RunActionTerminalObservation:
        provider_execution_id = terminal.provider_execution_id
        reobserve_terminal_main_container_for_cleanup(
            resource_manager.inspect_main(inventory),
            terminal,
        )
    elif type(terminal) is RunActionPreReleaseMainTerminalObservation:
        container = terminal.terminal_container_observation
        provider_execution_id = container.provider_execution_id
        reobserve_pre_release_terminal_main_container_for_cleanup(
            resource_manager.inspect_main(inventory),
            container,
        )
    else:
        raise RunActionDockerCleanupError(
            "run-action cleanup main lacks durable terminal authority"
        )
    if inventory.main_container_id != provider_execution_id:
        raise RunActionDockerCleanupError(
            "run-action cleanup main differs from durable provider identity"
        )


def _remove_main_once(
    *,
    session,
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
    exclusion: PinnedDockerCleanupExclusionLease,
) -> DockerRunActionResourceInventory:
    if inventory.main_container_id is None:
        raise RunActionDockerCleanupError("run-action cleanup main is absent")
    _require_cleanup_exclusion(session, exclusion)
    current = resource_manager.observe(inventory.preparation_allocation)
    if current != inventory:
        raise RunActionDockerCleanupError(
            "run-action cleanup inventory changed before main removal"
        )
    _cleanup_authority(cleanup_manager)._remove_stopped_container_once(
        container_id=inventory.main_container_id,
        exclusion_lease=exclusion,
        _authority=_DOCKER_CLEANUP_REMOVE_AUTHORITY,
    )
    return _require_next_inventory(
        resource_manager,
        inventory,
        replace(inventory, main_container_id=None),
    )


def _remove_keeper_once(
    *,
    session,
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
    exclusion: PinnedDockerCleanupExclusionLease,
) -> DockerRunActionResourceInventory:
    if inventory.keeper_container_id is None or inventory.main_container_id is not None:
        raise RunActionDockerCleanupError(
            "run-action keeper cleanup lacks its exact suffix"
        )
    _require_cleanup_exclusion(session, exclusion)
    current = resource_manager.observe(inventory.preparation_allocation)
    if current != inventory:
        raise RunActionDockerCleanupError(
            "run-action cleanup inventory changed before keeper removal"
        )
    _cleanup_authority(cleanup_manager)._remove_running_keeper_once(
        container_id=inventory.keeper_container_id,
        exclusion_lease=exclusion,
        _authority=_DOCKER_CLEANUP_REMOVE_AUTHORITY,
    )
    return _require_next_inventory(
        resource_manager,
        inventory,
        replace(inventory, keeper_container_id=None),
    )


def _remove_inert_keeper_once(
    *,
    session,
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
    exclusion: PinnedDockerCleanupExclusionLease,
) -> DockerRunActionResourceInventory:
    if inventory.keeper_container_id is None or inventory.main_container_id is not None:
        raise RunActionDockerCleanupError(
            "run-action inert keeper cleanup lacks its exact suffix"
        )
    _require_cleanup_exclusion(session, exclusion)
    current = resource_manager.observe(inventory.preparation_allocation)
    if current != inventory:
        raise RunActionDockerCleanupError(
            "run-action cleanup inventory changed before inert keeper removal"
        )
    _cleanup_authority(cleanup_manager)._remove_stopped_container_once(
        container_id=inventory.keeper_container_id,
        exclusion_lease=exclusion,
        _authority=_DOCKER_CLEANUP_REMOVE_AUTHORITY,
    )
    return _require_next_inventory(
        resource_manager,
        inventory,
        replace(inventory, keeper_container_id=None),
    )


def _remove_volume_once(
    *,
    session,
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
    exclusion: PinnedDockerCleanupExclusionLease,
) -> DockerRunActionResourceInventory:
    if (
        not inventory.volume_present
        or inventory.keeper_container_id is not None
        or inventory.main_container_id is not None
    ):
        raise RunActionDockerCleanupError(
            "run-action volume cleanup lacks its exact suffix"
        )
    _require_cleanup_exclusion(session, exclusion)
    current = resource_manager.observe(inventory.preparation_allocation)
    if current != inventory:
        raise RunActionDockerCleanupError(
            "run-action cleanup inventory changed before volume removal"
        )
    _cleanup_authority(cleanup_manager)._remove_volume_once(
        volume_name=(
            inventory.preparation_allocation.runtime_volume_authority.volume_name
        ),
        exclusion_lease=exclusion,
        _authority=_DOCKER_CLEANUP_REMOVE_AUTHORITY,
    )
    return _require_next_inventory(
        resource_manager,
        inventory,
        replace(inventory, volume_inspection_digest=None),
    )


def _require_next_inventory(
    resource_manager: DockerRunActionResourceManager,
    previous: DockerRunActionResourceInventory,
    expected: DockerRunActionResourceInventory,
) -> DockerRunActionResourceInventory:
    observed = resource_manager.observe(previous.preparation_allocation)
    if observed != expected:
        raise RunActionDockerCleanupError(
            "run-action cleanup did not produce the exact next physical suffix"
        )
    return observed


def _require_stable_absence(
    *,
    session,
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    cleanup_manager: DockerRunActionCleanupManager,
    exclusion: PinnedDockerCleanupExclusionLease,
) -> DockerRunActionResourceInventory:
    _cleanup_authority(cleanup_manager)
    _require_cleanup_exclusion(session, exclusion)
    if not inventory.is_absent:
        raise RunActionDockerCleanupError(
            "run-action stable absence requires an absent inventory"
        )
    first = resource_manager.observe(inventory.preparation_allocation)
    second = resource_manager.observe(inventory.preparation_allocation)
    if first != inventory or second != inventory:
        raise RunActionDockerCleanupError(
            "run-action resources reappeared during stable absence proof"
        )
    return second


def _require_cleanup_suffix(inventory: DockerRunActionResourceInventory) -> None:
    volume_present = inventory.volume_inspection_digest is not None
    keeper_present = inventory.keeper_container_id is not None
    main_present = inventory.main_container_id is not None
    if (main_present and not keeper_present) or (keeper_present and not volume_present):
        raise RunActionDockerCleanupError(
            "run-action cleanup inventory is not an exact removable suffix"
        )


def _issue_cleanup_exclusion(
    session,
    cleanup_manager: DockerRunActionCleanupManager,
) -> PinnedDockerCleanupExclusionLease:
    session._require_active()
    authority = _cleanup_authority(cleanup_manager)
    exclusion = authority._issue_exclusion_lease(
        _authority=_DOCKER_CLEANUP_EXCLUSION_ISSUANCE,
    )
    with _CLEANUP_EXCLUSION_LOCK:
        if _CLEANUP_EXCLUSION_SESSIONS.get(exclusion) is not None:
            raise RunActionDockerCleanupError(
                "run-action cleanup exclusion is already registered"
            )
        _CLEANUP_EXCLUSION_SESSIONS[exclusion] = session
    return exclusion


def _require_cleanup_exclusion(
    session,
    exclusion: PinnedDockerCleanupExclusionLease,
) -> None:
    session._require_active()
    with _CLEANUP_EXCLUSION_LOCK:
        registered_session = _CLEANUP_EXCLUSION_SESSIONS.get(exclusion)
    if (
        registered_session is not session
        or session._owner_process_id != os.getpid()
        or exclusion._owner_thread_id != get_ident()
    ):
        raise RunActionDockerCleanupError(
            "run-action cleanup lost same-operation mutation exclusion"
        )
    exclusion.require_current()


def _unregister_cleanup_exclusion(
    exclusion: PinnedDockerCleanupExclusionLease,
    session,
) -> None:
    with _CLEANUP_EXCLUSION_LOCK:
        registered_session = _CLEANUP_EXCLUSION_SESSIONS.pop(exclusion, None)
    if registered_session is not session:
        raise RunActionDockerCleanupError(
            "run-action cleanup exclusion registration changed"
        )


def _cleanup_authority(
    manager: DockerRunActionCleanupManager,
) -> PinnedDockerCleanupAuthority:
    with _CLEANUP_MANAGER_LOCK:
        authority = _CLEANUP_MANAGER_AUTHORITIES.get(manager)
    if (
        type(manager) is not DockerRunActionCleanupManager
        or type(authority) is not PinnedDockerCleanupAuthority
        or authority._owner_process_id != os.getpid()
    ):
        raise RunActionDockerCleanupError(
            "run-action cleanup manager is unissued or foreign"
        )
    return authority


__all__ = [
    "DockerRunActionCleanupManager",
    "issue_docker_run_action_resource_finalization_authority",
    "RunActionDockerCleanupError",
]
