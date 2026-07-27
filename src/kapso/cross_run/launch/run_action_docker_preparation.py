"""Exact Docker materialization of one durably allocated run-action."""

from __future__ import annotations

from threading import Lock
from typing import Any, Mapping
from weakref import WeakKeyDictionary

from kapso.cross_run.docker.runtime import (
    _DOCKER_PREPARATION_EXCLUSION_ISSUANCE,
    _DOCKER_PREPARATION_MUTATION_AUTHORITY,
    _docker_observation_and_preparation_authorities_share_runtime,
    PinnedDockerPreparationAuthority,
    PinnedDockerPreparationExclusionLease,
    PinnedDockerRuntime,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionInertKeeperObservation,
    DockerRunActionVolumeObservation,
    observe_allocation_keeper,
    observe_inert_keeper,
    observe_inert_main_container,
    observe_running_keeper,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
    keeper_create_arguments,
    main_create_arguments,
    require_run_action_image,
    volume_create_arguments,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    _run_action_observation_authority,
    DockerRunActionResourceInventory,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_recovery import (
    RunActionPreparationCapability,
    RunActionPreparationMode,
    RunActionPreparationObservation,
    RunActionPreparationOrigin,
    RunActionPreparationState,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    adopt_prepared_runtime_volume_layout,
    DockerRunActionPreparedVolumeObservation,
    materialize_runtime_volume_layout,
    observe_empty_runtime_volume,
    reobserve_runtime_volume_layout,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionDockerInitSourceEvidence,
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
    RunActionSupervisorHelperEvidence,
    RunActionVolumeKeeperEvidence,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    observe_docker_init_source,
    observe_supervisor_helper,
)
from kapso.cross_run.settings import DockerRuntimeSettings, LaunchSettings

_PREPARATION_MANAGER_LOCK = Lock()
_PREPARATION_MANAGER_AUTHORITIES: WeakKeyDictionary[
    DockerRunActionPreparationManager, PinnedDockerPreparationAuthority
] = WeakKeyDictionary()


class RunActionDockerPreparationError(RuntimeError):
    """A physical preparation differs from its durable allocation."""


class DockerRunActionPreparationManager:
    """Reconcile event 2 without exposing generic Docker mutation authority."""

    def __init__(
        self,
        *,
        runtime: PinnedDockerRuntime,
        resource_manager: DockerRunActionResourceManager,
        launch_settings: LaunchSettings,
    ) -> None:
        if (
            type(runtime) is not PinnedDockerRuntime
            or type(resource_manager) is not DockerRunActionResourceManager
            or type(launch_settings) is not LaunchSettings
        ):
            raise RunActionDockerPreparationError(
                "run-action preparation requires exact controller authorities"
            )
        authority = runtime.issue_preparation_authority()
        observation_authority = _run_action_observation_authority(resource_manager)
        if (
            not _docker_observation_and_preparation_authorities_share_runtime(
                observation_authority,
                authority,
            )
            or authority.settings != resource_manager.runtime_settings
        ):
            raise RunActionDockerPreparationError(
                "run-action preparation managers name different Docker runtimes"
            )
        self._resource_manager = resource_manager
        self._launch_settings = launch_settings
        with _PREPARATION_MANAGER_LOCK:
            if _PREPARATION_MANAGER_AUTHORITIES.get(self) is not None:
                raise RunActionDockerPreparationError(
                    "run-action preparation manager is already issued"
                )
            _PREPARATION_MANAGER_AUTHORITIES[self] = authority

    @property
    def runtime_settings(self) -> DockerRuntimeSettings:
        """Return settings from the exact issued preparation authority."""

        return _preparation_authority(self).settings

    def reconcile(
        self,
        capability: RunActionPreparationCapability,
        command: DockerRunActionCommand,
    ) -> RunActionPreparationObservation:
        """Create, adopt, or revalidate exactly as the capability permits."""

        if (
            type(capability) is not RunActionPreparationCapability
            or type(command) is not DockerRunActionCommand
        ):
            raise RunActionDockerPreparationError(
                "run-action preparation lacks exact capability and command"
            )
        allocation = capability.preparation_allocation
        mode = capability.mode
        durable = capability.durable_prepared_execution
        workspace_descriptor = capability.workspace_descriptor
        policy = allocation.preparation_claim.execution_policy
        if command.command_template_id != policy.command_template_id:
            raise RunActionDockerPreparationError(
                "run-action preparation command differs from durable policy"
            )
        if mode is RunActionPreparationMode.REVALIDATE_PREPARED:
            if type(durable) is not RunActionPreparedExecution:
                raise RunActionDockerPreparationError(
                    "prepared revalidation lacks its durable occurrence"
                )
            prepared = self._revalidate_prepared(
                allocation,
                durable,
                command,
            )
            return _preparation_observation(
                prepared,
                RunActionPreparationOrigin.REVALIDATED_PREPARED,
            )
        if durable is not None:
            raise RunActionDockerPreparationError(
                "allocation preparation carries a durable prepared occurrence"
            )
        inventory = self._resource_manager.observe(allocation)
        if mode is RunActionPreparationMode.REOPEN_ALLOCATED and _is_complete(
            inventory
        ):
            prepared = self._adopt_complete(allocation, command, inventory)
            return _preparation_observation(
                prepared,
                RunActionPreparationOrigin.REOPENED_ALLOCATION,
            )
        if not inventory.is_absent:
            return _unknown_preparation()
        if mode is RunActionPreparationMode.CREATE_ALLOCATED:
            origin = RunActionPreparationOrigin.NEWLY_MATERIALIZED
        elif mode is RunActionPreparationMode.REOPEN_ALLOCATED:
            origin = RunActionPreparationOrigin.MATERIALIZED_AFTER_PROVEN_ABSENCE
        else:
            raise RunActionDockerPreparationError(
                "run-action preparation mode is unsupported"
            )
        prepared = self._create_from_absence(
            allocation,
            command,
            workspace_descriptor=workspace_descriptor,
        )
        return _preparation_observation(prepared, origin)

    def _create_from_absence(
        self,
        allocation: RunActionPreparationAllocation,
        command: DockerRunActionCommand,
        *,
        workspace_descriptor: int | None,
    ) -> RunActionPreparedExecution | None:
        policy = allocation.preparation_claim.execution_policy
        image = _preparation_authority(self).inspect_exact_image(policy.image_authority)
        require_run_action_image(image, policy, self.runtime_settings)
        helper = observe_supervisor_helper(policy)
        init_source = observe_docker_init_source(policy)
        volume = self._create_volume(allocation)
        if volume is None:
            return None
        inert_keeper = self._create_keeper(
            allocation,
            volume,
            image,
            helper,
            init_source,
        )
        if inert_keeper is None:
            return None
        keeper = self._start_keeper(
            allocation,
            volume,
            inert_keeper,
            helper,
            init_source,
        )
        if keeper is None:
            return None
        empty_volume = observe_empty_runtime_volume(
            allocation.runtime_volume_authority,
            volume,
            keeper,
        )
        prepared_volume = materialize_runtime_volume_layout(
            allocation.preparation_claim,
            empty_volume,
            keeper,
            workspace_descriptor=workspace_descriptor,
            settings=self._launch_settings,
        )
        return self._create_main(
            allocation,
            command,
            volume,
            keeper,
            prepared_volume,
        )

    def _create_volume(
        self,
        allocation: RunActionPreparationAllocation,
    ) -> DockerRunActionVolumeObservation | None:
        authority = _preparation_authority(self)
        with _issue_exclusion(authority) as exclusion:
            before = self._resource_manager.observe(allocation)
            if not before.is_absent:
                raise RunActionDockerPreparationError(
                    "run-action volume creation lost exact absence"
                )
            exclusion.require_current()
            authority._create_volume_once(
                arguments=volume_create_arguments(
                    allocation.preparation_claim,
                    allocation.runtime_volume_authority,
                    self.runtime_settings,
                ),
                exclusion_lease=exclusion,
                _authority=_DOCKER_PREPARATION_MUTATION_AUTHORITY,
            )
            after = self._resource_manager.observe(allocation)
            exclusion.require_current()
            if after.is_absent:
                return None
            if not _is_volume_only(after):
                raise RunActionDockerPreparationError(
                    "run-action volume creation produced another resource suffix"
                )
            return _observe_volume(after, self._resource_manager)

    def _create_keeper(
        self,
        allocation: RunActionPreparationAllocation,
        volume: DockerRunActionVolumeObservation,
        image: Mapping[str, Any],
        helper: RunActionSupervisorHelperEvidence,
        init_source: RunActionDockerInitSourceEvidence,
    ) -> DockerRunActionInertKeeperObservation | None:
        authority = _preparation_authority(self)
        with _issue_exclusion(authority) as exclusion:
            before = self._resource_manager.observe(allocation)
            _require_volume_only(before, volume, self._resource_manager)
            current_helper = observe_supervisor_helper(
                allocation.preparation_claim.execution_policy
            )
            current_init_source = observe_docker_init_source(
                allocation.preparation_claim.execution_policy
            )
            if current_helper != helper or current_init_source != init_source:
                raise RunActionDockerPreparationError(
                    "run-action preparation executables changed before keeper creation"
                )
            exclusion.require_current()
            authority._create_container_once(
                arguments=keeper_create_arguments(
                    allocation.preparation_claim,
                    allocation.runtime_volume_authority,
                    image,
                    self.runtime_settings,
                ),
                exclusion_lease=exclusion,
                _authority=_DOCKER_PREPARATION_MUTATION_AUTHORITY,
            )
            after = self._resource_manager.observe(allocation)
            exclusion.require_current()
            if _is_volume_only(after):
                _require_volume_only(after, volume, self._resource_manager)
                return None
            if not _is_keeper_only(after):
                raise RunActionDockerPreparationError(
                    "run-action keeper creation produced another resource suffix"
                )
            current_volume = _observe_volume(after, self._resource_manager)
            if current_volume != volume:
                raise RunActionDockerPreparationError(
                    "run-action volume changed during keeper creation"
                )
            return observe_inert_keeper(
                self._resource_manager.inspect_keeper(after),
                allocation.preparation_claim,
                allocation.runtime_volume_authority,
                current_volume,
                current_helper,
                current_init_source,
                self.runtime_settings,
            )

    def _start_keeper(
        self,
        allocation: RunActionPreparationAllocation,
        volume: DockerRunActionVolumeObservation,
        inert_keeper: DockerRunActionInertKeeperObservation,
        helper: RunActionSupervisorHelperEvidence,
        init_source: RunActionDockerInitSourceEvidence,
    ) -> RunActionVolumeKeeperEvidence | None:
        authority = _preparation_authority(self)
        with _issue_exclusion(authority) as exclusion:
            before = self._resource_manager.observe(allocation)
            if (
                not _is_keeper_only(before)
                or before.keeper_container_id != inert_keeper.container_id
                or _observe_volume(before, self._resource_manager) != volume
                or observe_inert_keeper(
                    self._resource_manager.inspect_keeper(before),
                    allocation.preparation_claim,
                    allocation.runtime_volume_authority,
                    volume,
                    helper,
                    init_source,
                    self.runtime_settings,
                )
                != inert_keeper
            ):
                raise RunActionDockerPreparationError(
                    "run-action keeper changed before its preparation start"
                )
            exclusion.require_current()
            authority._start_created_container_once(
                container_id=inert_keeper.container_id,
                exclusion_lease=exclusion,
                _authority=_DOCKER_PREPARATION_MUTATION_AUTHORITY,
            )
            after = self._resource_manager.observe(allocation)
            exclusion.require_current()
            if (
                not _is_keeper_only(after)
                or after.keeper_container_id != inert_keeper.container_id
                or _observe_volume(after, self._resource_manager) != volume
            ):
                raise RunActionDockerPreparationError(
                    "run-action keeper start produced another resource suffix"
                )
            observed = observe_allocation_keeper(
                self._resource_manager.inspect_keeper(after),
                allocation.preparation_claim,
                allocation.runtime_volume_authority,
                volume,
                helper,
                init_source,
                self.runtime_settings,
            )
            if type(observed) is DockerRunActionInertKeeperObservation:
                if observed != inert_keeper:
                    raise RunActionDockerPreparationError(
                        "run-action inert keeper changed after start attempt"
                    )
                return None
            if type(observed) is not RunActionVolumeKeeperEvidence:
                raise RunActionDockerPreparationError(
                    "run-action keeper start returned an unknown lifecycle"
                )
            return observed

    def _create_main(
        self,
        allocation: RunActionPreparationAllocation,
        command: DockerRunActionCommand,
        volume: DockerRunActionVolumeObservation,
        keeper: RunActionVolumeKeeperEvidence,
        prepared_volume: DockerRunActionPreparedVolumeObservation,
    ) -> RunActionPreparedExecution | None:
        authority = _preparation_authority(self)
        policy = allocation.preparation_claim.execution_policy
        with _issue_exclusion(authority) as exclusion:
            before = self._resource_manager.observe(allocation)
            if not _is_keeper_only(before):
                raise RunActionDockerPreparationError(
                    "run-action main creation lost its keeper-only suffix"
                )
            current_volume = _observe_volume(before, self._resource_manager)
            helper = observe_supervisor_helper(policy)
            init_source = observe_docker_init_source(policy)
            current_keeper = observe_running_keeper(
                self._resource_manager.inspect_keeper(before),
                allocation.preparation_claim,
                allocation.runtime_volume_authority,
                current_volume,
                helper,
                init_source,
                self.runtime_settings,
            )
            adopted_volume = adopt_prepared_runtime_volume_layout(
                allocation,
                self._resource_manager,
                current_keeper,
                settings=self._launch_settings,
            )
            if (
                current_volume != volume
                or current_keeper != keeper
                or adopted_volume != prepared_volume
            ):
                raise RunActionDockerPreparationError(
                    "run-action prepared layout changed before main creation"
                )
            image = authority.inspect_exact_image(policy.image_authority)
            require_run_action_image(image, policy, self.runtime_settings)
            exclusion.require_current()
            authority._create_container_once(
                arguments=main_create_arguments(
                    allocation.preparation_claim,
                    allocation.runtime_volume_authority,
                    command,
                    image,
                    self.runtime_settings,
                ),
                exclusion_lease=exclusion,
                _authority=_DOCKER_PREPARATION_MUTATION_AUTHORITY,
            )
            after = self._resource_manager.observe(allocation)
            exclusion.require_current()
            if _is_keeper_only(after):
                if after != before:
                    raise RunActionDockerPreparationError(
                        "run-action resources changed after an unchanged main creation"
                    )
                return None
            if (
                not _is_complete(after)
                or after.volume_inspection_digest != before.volume_inspection_digest
                or after.keeper_container_id != before.keeper_container_id
            ):
                raise RunActionDockerPreparationError(
                    "run-action main creation produced another resource suffix"
                )
            prepared = _observe_complete_prepared(
                allocation=allocation,
                command=command,
                inventory=after,
                resource_manager=self._resource_manager,
                launch_settings=self._launch_settings,
                expected_volume=volume,
                expected_keeper=keeper,
                expected_prepared_volume=prepared_volume,
            )
            if self._resource_manager.observe(allocation) != after:
                raise RunActionDockerPreparationError(
                    "run-action resources changed after complete preparation"
                )
            return prepared

    def _adopt_complete(
        self,
        allocation: RunActionPreparationAllocation,
        command: DockerRunActionCommand,
        inventory: DockerRunActionResourceInventory,
    ) -> RunActionPreparedExecution:
        prepared = _observe_complete_prepared(
            allocation=allocation,
            command=command,
            inventory=inventory,
            resource_manager=self._resource_manager,
            launch_settings=self._launch_settings,
        )
        if self._resource_manager.observe(allocation) != inventory:
            raise RunActionDockerPreparationError(
                "run-action resources changed during allocation adoption"
            )
        return prepared

    def _revalidate_prepared(
        self,
        allocation: RunActionPreparationAllocation,
        durable: RunActionPreparedExecution,
        command: DockerRunActionCommand,
    ) -> RunActionPreparedExecution | None:
        inventory = self._resource_manager.observe(allocation)
        if not _is_complete(inventory):
            return None
        reobserved = _observe_complete_prepared(
            allocation=allocation,
            command=command,
            inventory=inventory,
            resource_manager=self._resource_manager,
            launch_settings=self._launch_settings,
        )
        if (
            reobserved != durable
            or self._resource_manager.observe(allocation) != inventory
        ):
            raise RunActionDockerPreparationError(
                "run-action prepared occurrence differs from durable event 3"
            )
        return durable


def _preparation_authority(
    manager: DockerRunActionPreparationManager,
) -> PinnedDockerPreparationAuthority:
    with _PREPARATION_MANAGER_LOCK:
        authority = _PREPARATION_MANAGER_AUTHORITIES.get(manager)
    if (
        type(manager) is not DockerRunActionPreparationManager
        or type(authority) is not PinnedDockerPreparationAuthority
    ):
        raise RunActionDockerPreparationError(
            "run-action preparation manager is unissued or foreign"
        )
    return authority


def _issue_exclusion(
    authority: PinnedDockerPreparationAuthority,
) -> PinnedDockerPreparationExclusionLease:
    return authority._issue_exclusion_lease(
        _authority=_DOCKER_PREPARATION_EXCLUSION_ISSUANCE,
    )


def _observe_volume(
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
) -> DockerRunActionVolumeObservation:
    return observe_runtime_volume(
        resource_manager.inspect_volume(inventory),
        inventory.preparation_allocation.preparation_claim,
        inventory.preparation_allocation.runtime_volume_authority,
        resource_manager.runtime_settings,
    )


def _require_volume_only(
    inventory: DockerRunActionResourceInventory,
    expected: DockerRunActionVolumeObservation,
    resource_manager: DockerRunActionResourceManager,
) -> None:
    if (
        not _is_volume_only(inventory)
        or _observe_volume(
            inventory,
            resource_manager,
        )
        != expected
    ):
        raise RunActionDockerPreparationError(
            "run-action volume-only occurrence changed"
        )


def _observe_complete_prepared(
    *,
    allocation: RunActionPreparationAllocation,
    command: DockerRunActionCommand,
    inventory: DockerRunActionResourceInventory,
    resource_manager: DockerRunActionResourceManager,
    launch_settings: LaunchSettings,
    expected_volume: DockerRunActionVolumeObservation | None = None,
    expected_keeper: RunActionVolumeKeeperEvidence | None = None,
    expected_prepared_volume: DockerRunActionPreparedVolumeObservation | None = None,
) -> RunActionPreparedExecution:
    if inventory.preparation_allocation != allocation or not _is_complete(inventory):
        raise RunActionDockerPreparationError(
            "complete preparation observation lacks its exact inventory"
        )
    claim = allocation.preparation_claim
    authority = allocation.runtime_volume_authority
    helper = observe_supervisor_helper(claim.execution_policy)
    init_source = observe_docker_init_source(claim.execution_policy)
    volume = _observe_volume(inventory, resource_manager)
    keeper = observe_running_keeper(
        resource_manager.inspect_keeper(inventory),
        claim,
        authority,
        volume,
        helper,
        init_source,
        resource_manager.runtime_settings,
    )
    prepared_volume = adopt_prepared_runtime_volume_layout(
        allocation,
        resource_manager,
        keeper,
        settings=launch_settings,
    )
    main = observe_inert_main_container(
        resource_manager.inspect_main(inventory),
        claim,
        authority,
        volume,
        command,
        helper,
        init_source,
        resource_manager.runtime_settings,
    )
    if (
        (expected_volume is not None and volume != expected_volume)
        or (expected_keeper is not None and keeper != expected_keeper)
        or (
            expected_prepared_volume is not None
            and prepared_volume != expected_prepared_volume
        )
    ):
        raise RunActionDockerPreparationError(
            "complete preparation differs from its preceding physical evidence"
        )
    prepared = RunActionPreparedExecution.mint(
        preparation_claim=claim,
        runtime_volume_authority=authority,
        runtime_volume_evidence=prepared_volume.runtime_volume_evidence,
        volume_keeper_evidence=keeper,
        input_delivery_slot=prepared_volume.input_delivery_slot,
        result_directory=prepared_volume.result_directory,
        control_directory=prepared_volume.control_directory,
        temporary_directory=prepared_volume.temporary_directory,
        credential_delivery_slot=prepared_volume.credential_delivery_slot,
        workspace_proof=prepared_volume.workspace_proof,
        layout_proof=prepared_volume.layout_proof,
        inert_container_evidence=main,
    )
    reopened = reobserve_runtime_volume_layout(
        prepared,
        volume,
        keeper,
        settings=launch_settings,
    )
    if reopened != prepared_volume:
        raise RunActionDockerPreparationError(
            "complete preparation changed during final layout reobservation"
        )
    return prepared


def _preparation_observation(
    prepared: RunActionPreparedExecution | None,
    origin: RunActionPreparationOrigin,
) -> RunActionPreparationObservation:
    if prepared is None:
        return _unknown_preparation()
    return RunActionPreparationObservation(
        state=RunActionPreparationState.EXACT_PREPARED,
        prepared_execution=prepared,
        origin=origin,
    )


def _unknown_preparation() -> RunActionPreparationObservation:
    return RunActionPreparationObservation(
        state=RunActionPreparationState.UNKNOWN,
        prepared_execution=None,
        origin=None,
    )


def _is_volume_only(inventory: DockerRunActionResourceInventory) -> bool:
    return (
        inventory.volume_present
        and inventory.keeper_container_id is None
        and inventory.main_container_id is None
    )


def _is_keeper_only(inventory: DockerRunActionResourceInventory) -> bool:
    return (
        inventory.volume_present
        and inventory.keeper_container_id is not None
        and inventory.main_container_id is None
    )


def _is_complete(inventory: DockerRunActionResourceInventory) -> bool:
    return (
        inventory.volume_present
        and inventory.keeper_container_id is not None
        and inventory.main_container_id is not None
    )


__all__ = [
    "DockerRunActionPreparationManager",
    "RunActionDockerPreparationError",
]
