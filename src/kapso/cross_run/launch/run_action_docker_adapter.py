"""Stateless production composition for one sealed Docker run-action lifecycle."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Mapping
from weakref import WeakKeyDictionary

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import PinnedDockerRuntime
from kapso.cross_run.launch.run_action_activation_envelope import (
    activation_execution_event_size_bound,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionExecutionLifecycleIdentity,
)
from kapso.cross_run.launch.run_action_credential_retirement import (
    DockerRunActionCredentialRetirementManager,
    retire_run_action_expired_credential_once,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionVolumeObservation,
    observe_inert_main_container,
    observe_running_barrier_main_container,
    observe_running_keeper,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_preparation import (
    DockerRunActionPreparationManager,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_main_start import (
    DockerRunActionStartManager,
    inspect_run_action_inert_activation,
    start_run_action_barrier_once,
)
from kapso.cross_run.launch.run_action_natural_terminal import (
    resolve_run_action_natural_terminal_once,
)
from kapso.cross_run.launch.run_action_pre_release_main_loss import (
    capture_run_action_pre_release_main_loss_termination,
    inspect_run_action_pre_release_main_loss,
)
from kapso.cross_run.launch.run_action_pre_release_main_terminal import (
    capture_run_action_pre_release_main_terminal_termination,
    inspect_run_action_pre_release_main_terminal,
)
from kapso.cross_run.launch.run_action_prepared_envelope import (
    prepared_execution_event_size_bound,
)
from kapso.cross_run.launch.run_action_recovery import (
    RunActionActivationCapability,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnQuery,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionPreparationCapability,
    RunActionPreparationObservation,
    RunActionUnactivatedSpawnObservation,
    RunActionUnactivatedSpawnQuery,
    RunActionUnactivatedSpawnState,
)
from kapso.cross_run.launch.run_action_release_publisher import (
    publish_run_action_workload_release_once,
)
from kapso.cross_run.launch.run_action_resolved_workload import (
    open_run_action_blocked_workload,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    deliver_and_reobserve_runtime_volume_activation,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    DockerRunActionExecutionPolicy,
    RunActionActivationRevalidationReceipt,
    RunActionInertContainerEvidence,
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
    RunActionVolumeKeeperEvidence,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_supervisor_helper import (
    observe_docker_init_source,
    observe_supervisor_helper,
)
from kapso.cross_run.launch.run_action_terminal_inspection import (
    inspect_run_action_terminal,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_pre_release_main_loss_observation_token,
    run_action_pre_release_main_terminal_observation_token,
)
from kapso.cross_run.launch.run_action_timeout_containment import (
    contain_run_action_timeout_once,
    DockerRunActionContainmentManager,
)
from kapso.cross_run.launch.run_action_timeout_publisher import (
    publish_run_action_timeout_once,
)
from kapso.cross_run.launch.run_action_timeout_termination import (
    capture_run_action_timeout_termination,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings


class DockerRunActionAdapterError(RuntimeError):
    """A production adapter input or physical state is outside its sealed policy."""


class _DockerMainLifecycle(str, Enum):
    INERT = "inert"
    RUNNING = "running"
    EXITED = "exited"


@dataclass(frozen=True)
class _ExactPreActivationOccurrence:
    inventory: DockerRunActionResourceInventory
    volume: DockerRunActionVolumeObservation
    keeper: RunActionVolumeKeeperEvidence
    inert_main: RunActionInertContainerEvidence


@dataclass(frozen=True)
class _DockerRunActionAdapterState:
    execution_lifecycle_identity: RunActionExecutionLifecycleIdentity
    execution_policy: DockerRunActionExecutionPolicy
    command: DockerRunActionCommand
    resource_manager: DockerRunActionResourceManager
    preparation_manager: DockerRunActionPreparationManager
    start_manager: DockerRunActionStartManager
    containment_manager: DockerRunActionContainmentManager
    credential_retirement_manager: DockerRunActionCredentialRetirementManager
    docker_settings: DockerRuntimeSettings
    launch_settings: LaunchSettings
    owner_process_id: int


_ADAPTER_STATE_LOCK = Lock()
_ADAPTER_STATES: WeakKeyDictionary[
    DockerRunActionExecutionAdapter, _DockerRunActionAdapterState
] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class DockerRunActionExecutionAdapter:
    """One immutable lifecycle adapter over narrow, runtime-bound Docker managers."""

    execution_lifecycle_identity: RunActionExecutionLifecycleIdentity
    execution_policy: DockerRunActionExecutionPolicy
    _creation_process_id: int = field(repr=False)

    def __init__(
        self,
        *,
        execution_lifecycle_identity: RunActionExecutionLifecycleIdentity,
        execution_policy: DockerRunActionExecutionPolicy,
        command: DockerRunActionCommand,
        runtime: PinnedDockerRuntime,
        launch_settings: LaunchSettings,
    ) -> None:
        if (
            type(execution_lifecycle_identity)
            is not RunActionExecutionLifecycleIdentity
            or type(execution_policy) is not DockerRunActionExecutionPolicy
            or type(command) is not DockerRunActionCommand
            or type(runtime) is not PinnedDockerRuntime
            or type(launch_settings) is not LaunchSettings
            or execution_lifecycle_identity.kind is not execution_policy.kind
            or execution_lifecycle_identity.execution_policy_id
            != execution_policy.docker_execution_policy_id
            or command.command_template_id != execution_policy.command_template_id
            or tree_or_blob_digest(runtime.settings.to_json_bytes())
            != execution_policy.docker_runtime_settings_digest
        ):
            raise DockerRunActionAdapterError(
                "Docker run-action adapter composition differs from its identity"
            )
        resource_manager = DockerRunActionResourceManager(runtime)
        object.__setattr__(
            self,
            "execution_lifecycle_identity",
            execution_lifecycle_identity,
        )
        object.__setattr__(self, "execution_policy", execution_policy)
        object.__setattr__(self, "_creation_process_id", os.getpid())
        state = _DockerRunActionAdapterState(
            execution_lifecycle_identity=execution_lifecycle_identity,
            execution_policy=execution_policy,
            command=command,
            resource_manager=resource_manager,
            preparation_manager=DockerRunActionPreparationManager(
                runtime=runtime,
                resource_manager=resource_manager,
                launch_settings=launch_settings,
            ),
            start_manager=DockerRunActionStartManager(runtime),
            containment_manager=DockerRunActionContainmentManager(runtime),
            credential_retirement_manager=(
                DockerRunActionCredentialRetirementManager(runtime)
            ),
            docker_settings=runtime.settings,
            launch_settings=launch_settings,
            owner_process_id=os.getpid(),
        )
        observe_supervisor_helper(execution_policy)
        observe_docker_init_source(execution_policy)
        with _ADAPTER_STATE_LOCK:
            if _ADAPTER_STATES.get(self) is not None:
                raise DockerRunActionAdapterError(
                    "Docker run-action adapter is already issued"
                )
            _ADAPTER_STATES[self] = state

    def __copy__(self):
        raise DockerRunActionAdapterError("Docker run-action adapter cannot be copied")

    def __deepcopy__(self, memo):
        raise DockerRunActionAdapterError("Docker run-action adapter cannot be copied")

    def __reduce__(self):
        raise DockerRunActionAdapterError(
            "Docker run-action adapter cannot be serialized"
        )

    def prepared_event_size_bound(
        self,
        *,
        preparation_allocation: RunActionPreparationAllocation,
        predecessor_event_id: str,
    ) -> int:
        state = _adapter_state(self)
        _require_execution_policy(
            preparation_allocation.preparation_claim.execution_policy,
            state,
        )
        return prepared_execution_event_size_bound(
            preparation_allocation=preparation_allocation,
            predecessor_event_id=predecessor_event_id,
            command=state.command,
            runtime_settings=state.docker_settings,
        )

    def activation_event_size_bound(
        self,
        *,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        predecessor_event_id: str,
    ) -> int:
        state = _adapter_state(self)
        _require_execution_policy(
            prepared_execution.preparation_claim.execution_policy,
            state,
        )
        return activation_execution_event_size_bound(
            prepared_execution=prepared_execution,
            spawn_commit=spawn_commit,
            predecessor_event_id=predecessor_event_id,
        )

    def prepare(
        self,
        capability: RunActionPreparationCapability,
    ) -> RunActionPreparationObservation:
        state = _adapter_state(self)
        _require_execution_policy(
            capability.preparation_allocation.preparation_claim.execution_policy,
            state,
        )
        return state.preparation_manager.reconcile(capability, state.command)

    def stage_activation(
        self,
        capability: RunActionActivationCapability,
    ) -> RunActionActivationRevalidationReceipt:
        if type(capability) is not RunActionActivationCapability:
            raise DockerRunActionAdapterError(
                "Docker activation staging lacks its exact capability"
            )
        state = _adapter_state(self)
        prepared = capability.prepared_execution
        _require_execution_policy(prepared.preparation_claim.execution_policy, state)
        spawn = capability.spawn_commit
        allocation = _allocation_from_prepared(prepared)
        before = _observe_exact_pre_activation_occurrence(
            prepared=prepared,
            allocation=allocation,
            command=state.command,
            resource_manager=state.resource_manager,
            docker_settings=state.docker_settings,
        )
        activated = deliver_and_reobserve_runtime_volume_activation(
            prepared,
            spawn,
            before.volume,
            before.keeper,
            request_payload=capability.request_payload,
            credential_materialization=capability.credential_materialization,
            workspace_descriptor=capability.workspace_descriptor,
            settings=state.launch_settings,
        )
        after = _observe_exact_pre_activation_occurrence(
            prepared=prepared,
            allocation=allocation,
            command=state.command,
            resource_manager=state.resource_manager,
            docker_settings=state.docker_settings,
        )
        if (
            after.inventory != before.inventory
            or after.volume != before.volume
            or after.keeper != before.keeper
            or after.inert_main != before.inert_main
        ):
            raise DockerRunActionAdapterError(
                "Docker occurrence changed during activation delivery"
            )
        return RunActionActivationRevalidationReceipt.mint(
            prepared_execution=prepared,
            spawn_commit=spawn,
            reobserved_volume_evidence=activated.reobserved_volume_evidence,
            reobserved_keeper_evidence=after.keeper,
            reobserved_container_evidence=after.inert_main,
            activated_workspace_observation=(activated.activated_workspace_observation),
            activated_runtime_directory_observations=(
                activated.activated_runtime_directory_observations
            ),
            activated_sentinel_observation=(activated.activated_sentinel_observation),
            input_file_observation=activated.input_file_observation,
            credential_file_observation=activated.credential_file_observation,
        )

    def inspect_unactivated(
        self,
        query: RunActionUnactivatedSpawnQuery,
    ) -> RunActionUnactivatedSpawnObservation:
        if type(query) is not RunActionUnactivatedSpawnQuery:
            raise DockerRunActionAdapterError(
                "Docker unactivated inspection lacks its exact query"
            )
        state = _adapter_state(self)
        prepared = query.prepared_execution
        _require_execution_policy(prepared.preparation_claim.execution_policy, state)
        allocation = _allocation_from_prepared(prepared)
        inventory = state.resource_manager.observe(allocation)
        if not _inventory_is_complete(inventory):
            return RunActionUnactivatedSpawnObservation(
                state=RunActionUnactivatedSpawnState.UNKNOWN,
            )
        raw_main = state.resource_manager.inspect_main(inventory)
        if _main_lifecycle(raw_main) is not _DockerMainLifecycle.INERT:
            return RunActionUnactivatedSpawnObservation(
                state=RunActionUnactivatedSpawnState.UNKNOWN,
            )
        _observe_exact_pre_activation_occurrence(
            prepared=prepared,
            allocation=allocation,
            command=state.command,
            resource_manager=state.resource_manager,
            docker_settings=state.docker_settings,
            expected_inventory=inventory,
        )
        return RunActionUnactivatedSpawnObservation(
            state=RunActionUnactivatedSpawnState.INERT_ACTIVATABLE,
        )

    def inspect_committed(
        self,
        query: RunActionCommittedSpawnQuery,
    ) -> RunActionCommittedSpawnObservation:
        if type(query) is not RunActionCommittedSpawnQuery:
            raise DockerRunActionAdapterError(
                "Docker committed inspection lacks its exact query"
            )
        adapter_state = _adapter_state(self)
        _require_execution_policy(
            query.prepared_execution.preparation_claim.execution_policy,
            adapter_state,
        )
        inventory = adapter_state.resource_manager.observe(query.preparation_allocation)
        if _inventory_is_surviving_without_main(inventory):
            if (
                query.control_directory_topology
                is not RunActionControlDirectoryTopology.EMPTY
            ):
                raise DockerRunActionAdapterError(
                    "Docker main vanished after workload release"
                )
            observation = inspect_run_action_pre_release_main_loss(
                query=query,
                resource_manager=adapter_state.resource_manager,
                helper_evidence=observe_supervisor_helper(
                    adapter_state.execution_policy
                ),
                init_source_evidence=observe_docker_init_source(
                    adapter_state.execution_policy
                ),
                docker_settings=adapter_state.docker_settings,
            )
            return RunActionCommittedSpawnObservation(
                state=(RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE),
                observation_token=(
                    run_action_pre_release_main_loss_observation_token(observation)
                ),
            )
        if not _inventory_is_complete(inventory):
            return RunActionCommittedSpawnObservation(
                state=RunActionCommittedSpawnState.UNKNOWN,
                observation_token=None,
            )
        lifecycle = _main_lifecycle(
            adapter_state.resource_manager.inspect_main(inventory)
        )
        if lifecycle is _DockerMainLifecycle.INERT:
            if (
                query.control_directory_topology
                is not RunActionControlDirectoryTopology.EMPTY
            ):
                raise DockerRunActionAdapterError(
                    "released Docker occurrence cannot remain never-started"
                )
            return inspect_run_action_inert_activation(
                query=query,
                resource_manager=adapter_state.resource_manager,
                launch_settings=adapter_state.launch_settings,
            )
        if lifecycle is _DockerMainLifecycle.RUNNING:
            running = _observe_exact_running_occurrence(
                query=query,
                command=adapter_state.command,
                resource_manager=adapter_state.resource_manager,
                docker_settings=adapter_state.docker_settings,
                expected_inventory=inventory,
            )
            return RunActionCommittedSpawnObservation(
                state=RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
                observation_token=running.complete_inspection_digest,
            )
        helper = observe_supervisor_helper(adapter_state.execution_policy)
        init_source = observe_docker_init_source(adapter_state.execution_policy)
        if query.control_directory_topology is RunActionControlDirectoryTopology.EMPTY:
            observation = inspect_run_action_pre_release_main_terminal(
                query=query,
                resource_manager=adapter_state.resource_manager,
                command=adapter_state.command,
                helper_evidence=helper,
                init_source_evidence=init_source,
                docker_settings=adapter_state.docker_settings,
                launch_settings=adapter_state.launch_settings,
            )
            return RunActionCommittedSpawnObservation(
                state=(
                    RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE
                ),
                observation_token=(
                    run_action_pre_release_main_terminal_observation_token(observation)
                ),
            )
        terminal = inspect_run_action_terminal(
            query=query,
            resource_manager=adapter_state.resource_manager,
            command=adapter_state.command,
            helper_evidence=helper,
            init_source_evidence=init_source,
            docker_settings=adapter_state.docker_settings,
            launch_settings=adapter_state.launch_settings,
        )
        return RunActionCommittedSpawnObservation(
            state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
            observation_token=terminal.complete_inspection_digest,
        )

    def continue_committed_once(
        self,
        capability: RunActionCommittedContinuationCapability,
    ) -> RunActionContinuationOutcome:
        if type(capability) is not RunActionCommittedContinuationCapability:
            raise DockerRunActionAdapterError(
                "Docker continuation lacks its exact capability"
            )
        adapter_state = _adapter_state(self)
        query = capability.query
        _require_execution_policy(
            query.prepared_execution.preparation_claim.execution_policy,
            adapter_state,
        )
        state = capability.observation.state
        helper = observe_supervisor_helper(adapter_state.execution_policy)
        init_source = observe_docker_init_source(adapter_state.execution_policy)
        if query.credential_retirement_intent is not None:
            if state in {
                RunActionCommittedSpawnState.INERT_CONTINUABLE,
                RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
            }:
                retire_run_action_expired_credential_once(
                    capability=capability,
                    resource_manager=adapter_state.resource_manager,
                    retirement_manager=(adapter_state.credential_retirement_manager),
                    command=adapter_state.command,
                    helper_evidence=helper,
                    init_source_evidence=init_source,
                    docker_settings=adapter_state.docker_settings,
                    launch_settings=adapter_state.launch_settings,
                )
                return _pending_outcome()
            if state is RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE:
                return capture_run_action_pre_release_main_loss_termination(
                    capability=capability,
                    resource_manager=adapter_state.resource_manager,
                    helper_evidence=helper,
                    init_source_evidence=init_source,
                    docker_settings=adapter_state.docker_settings,
                )
            if (
                state
                is RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE
            ):
                return capture_run_action_pre_release_main_terminal_termination(
                    capability=capability,
                    resource_manager=adapter_state.resource_manager,
                    command=adapter_state.command,
                    helper_evidence=helper,
                    init_source_evidence=init_source,
                    docker_settings=adapter_state.docker_settings,
                    launch_settings=adapter_state.launch_settings,
                )
            raise DockerRunActionAdapterError(
                "credential retirement received another Docker state"
            )
        if state is RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE:
            return capture_run_action_pre_release_main_loss_termination(
                capability=capability,
                resource_manager=adapter_state.resource_manager,
                helper_evidence=helper,
                init_source_evidence=init_source,
                docker_settings=adapter_state.docker_settings,
            )
        if state is RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE:
            return capture_run_action_pre_release_main_terminal_termination(
                capability=capability,
                resource_manager=adapter_state.resource_manager,
                command=adapter_state.command,
                helper_evidence=helper,
                init_source_evidence=init_source,
                docker_settings=adapter_state.docker_settings,
                launch_settings=adapter_state.launch_settings,
            )
        if state is RunActionCommittedSpawnState.INERT_CONTINUABLE:
            volume = _observe_exact_volume(
                query.prepared_execution,
                query.preparation_allocation,
                adapter_state.resource_manager,
                adapter_state.docker_settings,
            )
            start_run_action_barrier_once(
                capability=capability,
                resource_manager=adapter_state.resource_manager,
                start_manager=adapter_state.start_manager,
                command=adapter_state.command,
                volume_observation=volume,
                helper_evidence=helper,
                init_source_evidence=init_source,
                docker_settings=adapter_state.docker_settings,
                launch_settings=adapter_state.launch_settings,
            )
            return _pending_outcome()
        if state is RunActionCommittedSpawnState.RUNNING_CONTINUABLE:
            if (
                query.control_directory_topology
                is RunActionControlDirectoryTopology.EMPTY
            ):
                running = _observe_exact_running_occurrence(
                    query=query,
                    command=adapter_state.command,
                    resource_manager=adapter_state.resource_manager,
                    docker_settings=adapter_state.docker_settings,
                )
                volume = _observe_exact_volume(
                    query.prepared_execution,
                    query.preparation_allocation,
                    adapter_state.resource_manager,
                    adapter_state.docker_settings,
                )
                with open_run_action_blocked_workload(
                    capability,
                    committed_running_observation=running,
                    resource_manager=adapter_state.resource_manager,
                    preparation_allocation=query.preparation_allocation,
                    command=adapter_state.command,
                    volume_observation=volume,
                    helper_evidence=helper,
                    init_source_evidence=init_source,
                    docker_settings=adapter_state.docker_settings,
                    launch_settings=adapter_state.launch_settings,
                ) as lease:
                    release = publish_run_action_workload_release_once(
                        capability=capability,
                        blocked_workload_lease=lease,
                    )
                if release is None:
                    raise DockerRunActionAdapterError(
                        "Docker workload release lost its no-replace publication"
                    )
                return _pending_outcome()
            if (
                query.control_directory_topology
                is RunActionControlDirectoryTopology.RELEASED
            ):
                publication = publish_run_action_timeout_once(
                    capability=capability,
                    resource_manager=adapter_state.resource_manager,
                    command=adapter_state.command,
                    helper_evidence=helper,
                    init_source_evidence=init_source,
                    docker_settings=adapter_state.docker_settings,
                    launch_settings=adapter_state.launch_settings,
                )
                if publication is None:
                    return _pending_outcome()
                return RunActionContinuationOutcome(
                    state=RunActionContinuationState.TIMEOUT_PUBLISHED,
                    result=None,
                    provider_termination_receipt=None,
                    timeout_directive_publication=publication,
                )
            contain_run_action_timeout_once(
                capability=capability,
                resource_manager=adapter_state.resource_manager,
                containment_manager=adapter_state.containment_manager,
                command=adapter_state.command,
                helper_evidence=helper,
                init_source_evidence=init_source,
                docker_settings=adapter_state.docker_settings,
                launch_settings=adapter_state.launch_settings,
            )
            return _pending_outcome()
        if state is RunActionCommittedSpawnState.TERMINAL_CONTINUABLE:
            if (
                query.control_directory_topology
                is RunActionControlDirectoryTopology.RELEASED
            ):
                return resolve_run_action_natural_terminal_once(
                    capability=capability,
                    resource_manager=adapter_state.resource_manager,
                    command=adapter_state.command,
                    helper_evidence=helper,
                    init_source_evidence=init_source,
                    docker_settings=adapter_state.docker_settings,
                    launch_settings=adapter_state.launch_settings,
                )
            if (
                query.control_directory_topology
                is RunActionControlDirectoryTopology.TIMED_OUT
            ):
                receipt = capture_run_action_timeout_termination(
                    capability=capability,
                    resource_manager=adapter_state.resource_manager,
                    command=adapter_state.command,
                    helper_evidence=helper,
                    init_source_evidence=init_source,
                    docker_settings=adapter_state.docker_settings,
                    launch_settings=adapter_state.launch_settings,
                )
                return RunActionContinuationOutcome(
                    state=RunActionContinuationState.PROVIDER_TERMINATED,
                    result=None,
                    provider_termination_receipt=receipt,
                    timeout_directive_publication=None,
                )
        raise DockerRunActionAdapterError(
            "Docker continuation state differs from its control topology"
        )


def _adapter_state(
    adapter: DockerRunActionExecutionAdapter,
) -> _DockerRunActionAdapterState:
    if (
        type(adapter) is not DockerRunActionExecutionAdapter
        or adapter._creation_process_id != os.getpid()
    ):
        raise DockerRunActionAdapterError(
            "Docker run-action adapter authority is substituted or foreign"
        )
    with _ADAPTER_STATE_LOCK:
        state = _ADAPTER_STATES.get(adapter)
    if (
        type(state) is not _DockerRunActionAdapterState
        or state.owner_process_id != os.getpid()
        or state.execution_lifecycle_identity
        is not adapter.execution_lifecycle_identity
        or state.execution_policy is not adapter.execution_policy
        or type(state.command) is not DockerRunActionCommand
        or state.command.command_template_id
        != adapter.execution_policy.command_template_id
        or type(state.resource_manager) is not DockerRunActionResourceManager
        or type(state.preparation_manager) is not DockerRunActionPreparationManager
        or type(state.start_manager) is not DockerRunActionStartManager
        or type(state.containment_manager) is not DockerRunActionContainmentManager
        or type(state.credential_retirement_manager)
        is not DockerRunActionCredentialRetirementManager
        or type(state.docker_settings) is not DockerRuntimeSettings
        or type(state.launch_settings) is not LaunchSettings
        or state.resource_manager.runtime_settings != state.docker_settings
        or state.preparation_manager.runtime_settings != state.docker_settings
        or state.start_manager.runtime_settings != state.docker_settings
        or state.containment_manager.runtime_settings != state.docker_settings
        or state.credential_retirement_manager.runtime_settings != state.docker_settings
    ):
        raise DockerRunActionAdapterError(
            "Docker run-action adapter authority is substituted or foreign"
        )
    return state


def _require_execution_policy(
    execution_policy: DockerRunActionExecutionPolicy,
    state: _DockerRunActionAdapterState,
) -> None:
    if (
        type(execution_policy) is not DockerRunActionExecutionPolicy
        or execution_policy != state.execution_policy
        or execution_policy.docker_execution_policy_id
        != state.execution_lifecycle_identity.execution_policy_id
    ):
        raise DockerRunActionAdapterError(
            "Docker operation differs from its sealed execution policy"
        )


def _allocation_from_prepared(
    prepared: RunActionPreparedExecution,
) -> RunActionPreparationAllocation:
    if type(prepared) is not RunActionPreparedExecution:
        raise DockerRunActionAdapterError(
            "Docker occurrence lacks a durable prepared execution"
        )
    return RunActionPreparationAllocation.mint(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_authority=prepared.runtime_volume_authority,
    )


def _observe_exact_pre_activation_occurrence(
    *,
    prepared: RunActionPreparedExecution,
    allocation: RunActionPreparationAllocation,
    command: DockerRunActionCommand,
    resource_manager: DockerRunActionResourceManager,
    docker_settings: DockerRuntimeSettings,
    expected_inventory: DockerRunActionResourceInventory | None = None,
) -> _ExactPreActivationOccurrence:
    inventory = resource_manager.observe(allocation)
    if (
        expected_inventory is not None and inventory != expected_inventory
    ) or not _inventory_is_complete(inventory):
        raise DockerRunActionAdapterError(
            "Docker pre-activation occurrence is incomplete or changed"
        )
    policy = prepared.preparation_claim.execution_policy
    helper = observe_supervisor_helper(policy)
    init_source = observe_docker_init_source(policy)
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        docker_settings,
    )
    keeper = observe_running_keeper(
        resource_manager.inspect_keeper(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        helper,
        init_source,
        docker_settings,
    )
    inert_main = observe_inert_main_container(
        resource_manager.inspect_main(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        command,
        helper,
        init_source,
        docker_settings,
    )
    if (
        inventory.main_container_id != prepared.inert_container_evidence.container_id
        or inventory.keeper_container_id != prepared.volume_keeper_evidence.container_id
        or volume.volume_occurrence_digest
        != prepared.runtime_volume_evidence.docker_volume_occurrence_digest
        or keeper != prepared.volume_keeper_evidence
        or inert_main != prepared.inert_container_evidence
        or resource_manager.observe(allocation) != inventory
    ):
        raise DockerRunActionAdapterError(
            "Docker pre-activation occurrence differs from durable preparation"
        )
    return _ExactPreActivationOccurrence(
        inventory=inventory,
        volume=volume,
        keeper=keeper,
        inert_main=inert_main,
    )


def _observe_exact_volume(
    prepared: RunActionPreparedExecution,
    allocation: RunActionPreparationAllocation,
    resource_manager: DockerRunActionResourceManager,
    docker_settings: DockerRuntimeSettings,
) -> DockerRunActionVolumeObservation:
    inventory = resource_manager.observe(allocation)
    if not _inventory_is_complete(inventory):
        raise DockerRunActionAdapterError(
            "Docker continuation lacks its complete resource occurrence"
        )
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        docker_settings,
    )
    if (
        volume.volume_occurrence_digest
        != prepared.runtime_volume_evidence.docker_volume_occurrence_digest
        or resource_manager.observe(allocation) != inventory
    ):
        raise DockerRunActionAdapterError(
            "Docker continuation volume differs from durable preparation"
        )
    return volume


def _observe_exact_running_occurrence(
    *,
    query: RunActionCommittedSpawnQuery,
    command: DockerRunActionCommand,
    resource_manager: DockerRunActionResourceManager,
    docker_settings: DockerRuntimeSettings,
    expected_inventory: DockerRunActionResourceInventory | None = None,
):
    inventory = resource_manager.observe(query.preparation_allocation)
    if (
        expected_inventory is not None and inventory != expected_inventory
    ) or not _inventory_is_complete(inventory):
        raise DockerRunActionAdapterError(
            "Docker running occurrence is incomplete or changed"
        )
    prepared = query.prepared_execution
    policy = prepared.preparation_claim.execution_policy
    volume = observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        docker_settings,
    )
    running = observe_running_barrier_main_container(
        resource_manager.inspect_main(inventory),
        prepared.preparation_claim,
        prepared.runtime_volume_authority,
        volume,
        command,
        observe_supervisor_helper(policy),
        observe_docker_init_source(policy),
        docker_settings,
    )
    if (
        running.container_id != query.spawn_commit.provider_execution_id
        or volume.volume_occurrence_digest
        != prepared.runtime_volume_evidence.docker_volume_occurrence_digest
        or resource_manager.observe(query.preparation_allocation) != inventory
    ):
        raise DockerRunActionAdapterError(
            "Docker running occurrence differs from durable activation"
        )
    return running


def _main_lifecycle(raw_main: Mapping[str, object]) -> _DockerMainLifecycle:
    if not isinstance(raw_main, Mapping):
        raise DockerRunActionAdapterError("Docker main inspection is not a mapping")
    state = raw_main.get("State")
    if not isinstance(state, Mapping) or type(state.get("Status")) is not str:
        raise DockerRunActionAdapterError("Docker main lifecycle is malformed")
    status = state["Status"]
    if status == "created":
        return _DockerMainLifecycle.INERT
    if status == "running":
        return _DockerMainLifecycle.RUNNING
    if status == "exited":
        return _DockerMainLifecycle.EXITED
    raise DockerRunActionAdapterError("Docker main lifecycle is not admitted")


def _inventory_is_complete(inventory: DockerRunActionResourceInventory) -> bool:
    return (
        type(inventory) is DockerRunActionResourceInventory
        and inventory.volume_present
        and inventory.keeper_container_id is not None
        and inventory.main_container_id is not None
    )


def _inventory_is_surviving_without_main(
    inventory: DockerRunActionResourceInventory,
) -> bool:
    return (
        type(inventory) is DockerRunActionResourceInventory
        and inventory.volume_present
        and inventory.keeper_container_id is not None
        and inventory.main_container_id is None
    )


def _pending_outcome() -> RunActionContinuationOutcome:
    return RunActionContinuationOutcome(
        state=RunActionContinuationState.PENDING,
        result=None,
        provider_termination_receipt=None,
        timeout_directive_publication=None,
    )


__all__ = [
    "DockerRunActionAdapterError",
    "DockerRunActionExecutionAdapter",
]
