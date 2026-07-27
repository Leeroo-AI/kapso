from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

import kapso.cross_run.launch.run_action_docker_preparation as preparation_module
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_docker_preparation import (
    DockerRunActionPreparationManager,
    RunActionDockerPreparationError,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceInventory,
)
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_PREPARATION_AUTHORITY,
    RunActionPreparationCapability,
    RunActionPreparationMode,
    RunActionPreparationOrigin,
    RunActionPreparationState,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationAllocation,
)
from test_run_action_docker_projection import _policy, docker_settings
from test_run_action_supervisor_contracts import (
    _claim,
    _prepared_execution,
    _volume_authority,
)

_COMMAND = DockerRunActionCommand.build(
    entrypoint="/bin/tool",
    arguments=("default",),
)
_GENERATION_NONCE = "1" * 32


@dataclass
class _InventoryResourceManager:
    inventory: DockerRunActionResourceInventory

    def observe(
        self,
        allocation: RunActionPreparationAllocation,
    ) -> DockerRunActionResourceInventory:
        assert allocation == self.inventory.preparation_allocation
        return self.inventory


class _PreparationDelegate:
    def __init__(
        self,
        manager: DockerRunActionPreparationManager,
        command: DockerRunActionCommand,
    ) -> None:
        self._manager = manager
        self._command = command

    def prepare(self, capability):
        return self._manager.reconcile(capability, self._command)


class _PreparationExclusion:
    def __init__(self) -> None:
        self.current_checks = 0

    def __enter__(self):
        return self

    def __exit__(self, *_arguments):
        return None

    def require_current(self):
        self.current_checks += 1


class _VolumeCreationAuthority:
    def __init__(self, docker_settings) -> None:
        self.settings = docker_settings
        self.exclusion = _PreparationExclusion()
        self.calls = []

    def _issue_exclusion_lease(self, *, _authority):
        assert _authority is preparation_module._DOCKER_PREPARATION_EXCLUSION_ISSUANCE
        return self.exclusion

    def _create_volume_once(
        self,
        *,
        arguments,
        exclusion_lease,
        _authority,
    ):
        assert exclusion_lease is self.exclusion
        assert _authority is preparation_module._DOCKER_PREPARATION_MUTATION_AUTHORITY
        self.calls.append(arguments)
        return object()


class _StageMutationAuthority(_VolumeCreationAuthority):
    def __init__(self, docker_settings) -> None:
        super().__init__(docker_settings)
        self.image = {}

    def inspect_exact_image(self, _image_authority):
        return self.image

    def _create_container_once(
        self,
        *,
        arguments,
        exclusion_lease,
        _authority,
    ):
        return self._record_mutation(
            arguments,
            exclusion_lease,
            _authority,
        )

    def _start_created_container_once(
        self,
        *,
        container_id,
        exclusion_lease,
        _authority,
    ):
        return self._record_mutation(
            ("container", "start", container_id),
            exclusion_lease,
            _authority,
        )

    def _record_mutation(self, arguments, exclusion_lease, _authority):
        assert exclusion_lease is self.exclusion
        assert _authority is preparation_module._DOCKER_PREPARATION_MUTATION_AUTHORITY
        self.calls.append(arguments)
        return object()


class _InventorySequenceResourceManager:
    def __init__(self, docker_settings, inventories) -> None:
        self.runtime_settings = docker_settings
        self._inventories = iter(inventories)

    def observe(self, allocation):
        inventory = next(self._inventories)
        assert inventory.preparation_allocation == allocation
        return inventory

    def inspect_keeper(self, _inventory):
        return {}


def _allocation(docker_settings):
    policy = _policy(
        docker_settings,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        command_template_id=_COMMAND.command_template_id,
    )
    claim = _claim(policy=policy)
    authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=claim,
        runtime_volume_authority=authority,
    )
    prepared = _prepared_execution(
        claim=claim,
        authority=authority,
    )
    return allocation, prepared


def _inventory(
    allocation,
    *,
    volume=False,
    keeper=False,
    main=False,
):
    return DockerRunActionResourceInventory(
        preparation_allocation=allocation,
        volume_inspection_digest=(tree_or_blob_digest(b"volume") if volume else None),
        keeper_container_id="b" * 64 if keeper else None,
        main_container_id="a" * 64 if main else None,
    )


def _manager_with_inventory(inventory):
    manager = object.__new__(DockerRunActionPreparationManager)
    manager._resource_manager = _InventoryResourceManager(inventory)
    manager._launch_settings = None
    return manager


def _invoke(
    manager,
    allocation,
    *,
    mode,
    durable=None,
    command=_COMMAND,
):
    capability = RunActionPreparationCapability(
        preparation_allocation=allocation,
        mode=mode,
        durable_prepared_execution=durable,
        workspace_descriptor=None,
        workspace_source_path=None,
        _authority=_RUN_ACTION_PREPARATION_AUTHORITY,
    )
    return capability._invoke_once(_PreparationDelegate(manager, command))


@pytest.mark.parametrize(
    ("mode", "expected_origin"),
    (
        (
            RunActionPreparationMode.CREATE_ALLOCATED,
            RunActionPreparationOrigin.NEWLY_MATERIALIZED,
        ),
        (
            RunActionPreparationMode.REOPEN_ALLOCATED,
            RunActionPreparationOrigin.MATERIALIZED_AFTER_PROVEN_ABSENCE,
        ),
    ),
)
def test_exact_absence_materializes_with_mode_specific_origin(
    docker_settings,
    monkeypatch,
    mode,
    expected_origin,
):
    allocation, prepared = _allocation(docker_settings)
    manager = _manager_with_inventory(_inventory(allocation))
    calls = []

    def create_from_absence(
        _manager,
        observed_allocation,
        observed_command,
        *,
        workspace_descriptor,
    ):
        calls.append(
            (
                observed_allocation,
                observed_command,
                workspace_descriptor,
            )
        )
        return prepared

    monkeypatch.setattr(
        DockerRunActionPreparationManager,
        "_create_from_absence",
        create_from_absence,
    )

    observation = _invoke(manager, allocation, mode=mode)

    assert observation.state is RunActionPreparationState.EXACT_PREPARED
    assert observation.prepared_execution == prepared
    assert observation.origin is expected_origin
    assert calls == [(allocation, _COMMAND, None)]


def test_reopen_adopts_only_a_complete_event_two_occurrence(
    docker_settings,
    monkeypatch,
):
    allocation, prepared = _allocation(docker_settings)
    complete = _inventory(allocation, volume=True, keeper=True, main=True)
    manager = _manager_with_inventory(complete)
    calls = []

    def adopt_complete(
        _manager,
        observed_allocation,
        observed_command,
        observed_inventory,
    ):
        calls.append(
            (
                observed_allocation,
                observed_command,
                observed_inventory,
            )
        )
        return prepared

    monkeypatch.setattr(
        DockerRunActionPreparationManager,
        "_adopt_complete",
        adopt_complete,
    )

    observation = _invoke(
        manager,
        allocation,
        mode=RunActionPreparationMode.REOPEN_ALLOCATED,
    )

    assert observation.state is RunActionPreparationState.EXACT_PREPARED
    assert observation.prepared_execution == prepared
    assert observation.origin is RunActionPreparationOrigin.REOPENED_ALLOCATION
    assert calls == [(allocation, _COMMAND, complete)]


@pytest.mark.parametrize(
    ("mode", "durable"),
    (
        (RunActionPreparationMode.CREATE_ALLOCATED, None),
        (RunActionPreparationMode.REOPEN_ALLOCATED, None),
        (RunActionPreparationMode.REVALIDATE_PREPARED, "prepared"),
    ),
)
@pytest.mark.parametrize(
    ("volume", "keeper", "main"),
    (
        (True, False, False),
        (True, True, False),
        (False, False, True),
        (False, True, False),
        (True, False, True),
        (False, True, True),
    ),
)
def test_partial_event_two_suffix_is_unknown_and_never_repaired(
    docker_settings,
    monkeypatch,
    mode,
    durable,
    volume,
    keeper,
    main,
):
    allocation, prepared = _allocation(docker_settings)
    manager = _manager_with_inventory(
        _inventory(
            allocation,
            volume=volume,
            keeper=keeper,
            main=main,
        )
    )

    def forbidden(*_arguments, **_keywords):
        raise AssertionError("partial preparation must not mutate or adopt")

    monkeypatch.setattr(
        DockerRunActionPreparationManager,
        "_create_from_absence",
        forbidden,
    )
    monkeypatch.setattr(
        DockerRunActionPreparationManager,
        "_adopt_complete",
        forbidden,
    )
    observation = _invoke(
        manager,
        allocation,
        mode=mode,
        durable=prepared if durable == "prepared" else None,
    )

    assert observation.state is RunActionPreparationState.UNKNOWN
    assert observation.prepared_execution is None
    assert observation.origin is None


def test_absent_prepared_occurrence_revalidates_as_unknown(
    docker_settings,
):
    allocation, prepared = _allocation(docker_settings)
    manager = _manager_with_inventory(_inventory(allocation))

    observation = _invoke(
        manager,
        allocation,
        mode=RunActionPreparationMode.REVALIDATE_PREPARED,
        durable=prepared,
    )

    assert observation.state is RunActionPreparationState.UNKNOWN


def test_revalidation_rejects_a_different_complete_prepared_occurrence(
    docker_settings,
    monkeypatch,
):
    allocation, durable = _allocation(docker_settings)
    replacement = _prepared_execution(
        claim=allocation.preparation_claim,
        authority=allocation.runtime_volume_authority,
        container_id="c" * 64,
        inode_offset=1,
    )
    manager = _manager_with_inventory(
        _inventory(allocation, volume=True, keeper=True, main=True)
    )
    monkeypatch.setattr(
        preparation_module,
        "_observe_complete_prepared",
        lambda **_arguments: replacement,
    )

    with pytest.raises(
        RunActionDockerPreparationError,
        match="differs from durable event 3",
    ):
        _invoke(
            manager,
            allocation,
            mode=RunActionPreparationMode.REVALIDATE_PREPARED,
            durable=durable,
        )


def test_revalidation_rejects_inventory_change_after_exact_reobservation(
    docker_settings,
    monkeypatch,
):
    allocation, durable = _allocation(docker_settings)
    first = _inventory(allocation, volume=True, keeper=True, main=True)
    changed = DockerRunActionResourceInventory(
        preparation_allocation=allocation,
        volume_inspection_digest=first.volume_inspection_digest,
        keeper_container_id=first.keeper_container_id,
        main_container_id="d" * 64,
    )
    manager = object.__new__(DockerRunActionPreparationManager)
    manager._resource_manager = _InventorySequenceResourceManager(
        docker_settings,
        (first, changed),
    )
    manager._launch_settings = None
    monkeypatch.setattr(
        preparation_module,
        "_observe_complete_prepared",
        lambda **_arguments: durable,
    )

    with pytest.raises(
        RunActionDockerPreparationError,
        match="differs from durable event 3",
    ):
        _invoke(
            manager,
            allocation,
            mode=RunActionPreparationMode.REVALIDATE_PREPARED,
            durable=durable,
        )


def test_prepared_revalidation_returns_only_the_durable_occurrence(
    docker_settings,
    monkeypatch,
):
    allocation, prepared = _allocation(docker_settings)
    complete = _inventory(allocation, volume=True, keeper=True, main=True)
    manager = _manager_with_inventory(complete)
    calls = []

    def revalidate(
        _manager,
        observed_allocation,
        durable,
        observed_command,
    ):
        calls.append((observed_allocation, durable, observed_command))
        return durable

    monkeypatch.setattr(
        DockerRunActionPreparationManager,
        "_revalidate_prepared",
        revalidate,
    )
    monkeypatch.setattr(
        DockerRunActionPreparationManager,
        "_create_from_absence",
        lambda *_arguments, **_keywords: (_ for _ in ()).throw(
            AssertionError("revalidation must not create")
        ),
    )

    observation = _invoke(
        manager,
        allocation,
        mode=RunActionPreparationMode.REVALIDATE_PREPARED,
        durable=prepared,
    )

    assert observation.state is RunActionPreparationState.EXACT_PREPARED
    assert observation.prepared_execution == prepared
    assert observation.origin is RunActionPreparationOrigin.REVALIDATED_PREPARED
    assert calls == [(allocation, prepared, _COMMAND)]


def test_durable_command_mismatch_fails_before_resource_observation(
    docker_settings,
):
    allocation, _prepared = _allocation(docker_settings)

    class _ForbiddenResourceManager:
        def observe(self, _allocation):
            raise AssertionError("command mismatch must precede Docker observation")

    manager = object.__new__(DockerRunActionPreparationManager)
    manager._resource_manager = _ForbiddenResourceManager()
    manager._launch_settings = None
    another_command = DockerRunActionCommand.build(
        entrypoint="/bin/tool",
        arguments=("another",),
    )

    with pytest.raises(
        RunActionDockerPreparationError,
        match="differs from durable policy",
    ):
        _invoke(
            manager,
            allocation,
            mode=RunActionPreparationMode.CREATE_ALLOCATED,
            command=another_command,
        )


def test_active_preparation_capability_rejects_foreign_thread_access(
    docker_settings,
):
    allocation, _prepared = _allocation(docker_settings)
    capability = RunActionPreparationCapability(
        preparation_allocation=allocation,
        mode=RunActionPreparationMode.CREATE_ALLOCATED,
        durable_prepared_execution=None,
        workspace_descriptor=None,
        workspace_source_path=None,
        _authority=_RUN_ACTION_PREPARATION_AUTHORITY,
    )

    with capability._begin_invocation():
        with ThreadPoolExecutor(max_workers=1) as execution:
            foreign_access = execution.submit(lambda: capability.preparation_allocation)
            with pytest.raises(
                RunActionRecoveryError,
                match="not in its one invocation",
            ):
                foreign_access.result()


def test_fresh_create_never_adopts_a_preexisting_complete_occurrence(
    docker_settings,
    monkeypatch,
):
    allocation, _prepared = _allocation(docker_settings)
    manager = _manager_with_inventory(
        _inventory(allocation, volume=True, keeper=True, main=True)
    )
    monkeypatch.setattr(
        DockerRunActionPreparationManager,
        "_adopt_complete",
        lambda *_arguments: (_ for _ in ()).throw(
            AssertionError("fresh create must not adopt")
        ),
    )

    observation = _invoke(
        manager,
        allocation,
        mode=RunActionPreparationMode.CREATE_ALLOCATED,
    )

    assert observation.state is RunActionPreparationState.UNKNOWN


@pytest.mark.parametrize("advanced", (False, True))
def test_volume_creation_uses_fresh_suffix_not_the_docker_response(
    docker_settings,
    monkeypatch,
    advanced,
):
    allocation, _prepared = _allocation(docker_settings)
    absent = _inventory(allocation)
    volume_only = _inventory(allocation, volume=True)
    manager = object.__new__(DockerRunActionPreparationManager)
    manager._resource_manager = _InventorySequenceResourceManager(
        docker_settings,
        (absent, volume_only if advanced else absent),
    )
    manager._launch_settings = None
    authority = _VolumeCreationAuthority(docker_settings)
    observed_volume = object()
    monkeypatch.setattr(
        preparation_module,
        "_preparation_authority",
        lambda _manager: authority,
    )

    def observe_volume(inventory, resource_manager):
        assert inventory == volume_only
        assert resource_manager is manager._resource_manager
        return observed_volume

    monkeypatch.setattr(
        preparation_module,
        "_observe_volume",
        observe_volume,
    )

    result = manager._create_volume(allocation)

    assert result is (observed_volume if advanced else None)
    assert authority.calls == [
        preparation_module.volume_create_arguments(
            allocation.preparation_claim,
            allocation.runtime_volume_authority,
            docker_settings,
        )
    ]
    assert authority.exclusion.current_checks == 2


@pytest.mark.parametrize("advanced", (False, True))
def test_keeper_creation_uses_fresh_suffix_not_the_docker_response(
    docker_settings,
    monkeypatch,
    advanced,
):
    allocation, _prepared = _allocation(docker_settings)
    volume_only = _inventory(allocation, volume=True)
    keeper_only = _inventory(allocation, volume=True, keeper=True)
    manager = object.__new__(DockerRunActionPreparationManager)
    manager._resource_manager = _InventorySequenceResourceManager(
        docker_settings,
        (volume_only, keeper_only if advanced else volume_only),
    )
    manager._launch_settings = None
    authority = _StageMutationAuthority(docker_settings)
    volume = object()
    helper = object()
    init_source = object()
    inert_keeper = object()
    projected_arguments = ("container", "create", "--name", "keeper")
    monkeypatch.setattr(
        preparation_module,
        "_preparation_authority",
        lambda _manager: authority,
    )
    monkeypatch.setattr(
        preparation_module,
        "_require_volume_only",
        lambda *_arguments: None,
    )
    monkeypatch.setattr(
        preparation_module,
        "observe_supervisor_helper",
        lambda _policy: helper,
    )
    monkeypatch.setattr(
        preparation_module,
        "observe_docker_init_source",
        lambda _policy: init_source,
    )
    monkeypatch.setattr(
        preparation_module,
        "keeper_create_arguments",
        lambda *_arguments: projected_arguments,
    )
    monkeypatch.setattr(
        preparation_module,
        "_observe_volume",
        lambda *_arguments: volume,
    )
    monkeypatch.setattr(
        preparation_module,
        "observe_inert_keeper",
        lambda *_arguments: inert_keeper,
    )

    result = manager._create_keeper(
        allocation,
        volume,
        authority.image,
        helper,
        init_source,
    )

    assert result is (inert_keeper if advanced else None)
    assert authority.calls == [projected_arguments]
    assert authority.exclusion.current_checks == 2


@pytest.mark.parametrize("advanced", (False, True))
def test_keeper_start_uses_fresh_lifecycle_not_the_docker_response(
    docker_settings,
    monkeypatch,
    advanced,
):
    class _InertKeeper:
        container_id = "b" * 64

    class _RunningKeeper:
        pass

    allocation, _prepared = _allocation(docker_settings)
    keeper_only = _inventory(allocation, volume=True, keeper=True)
    manager = object.__new__(DockerRunActionPreparationManager)
    manager._resource_manager = _InventorySequenceResourceManager(
        docker_settings,
        (keeper_only, keeper_only),
    )
    manager._launch_settings = None
    authority = _StageMutationAuthority(docker_settings)
    volume = object()
    inert_keeper = _InertKeeper()
    running_keeper = _RunningKeeper()
    monkeypatch.setattr(
        preparation_module,
        "_preparation_authority",
        lambda _manager: authority,
    )
    monkeypatch.setattr(
        preparation_module,
        "DockerRunActionInertKeeperObservation",
        _InertKeeper,
    )
    monkeypatch.setattr(
        preparation_module,
        "RunActionVolumeKeeperEvidence",
        _RunningKeeper,
    )
    monkeypatch.setattr(
        preparation_module,
        "_observe_volume",
        lambda *_arguments: volume,
    )
    monkeypatch.setattr(
        preparation_module,
        "observe_inert_keeper",
        lambda *_arguments: inert_keeper,
    )
    monkeypatch.setattr(
        preparation_module,
        "observe_allocation_keeper",
        lambda *_arguments: running_keeper if advanced else inert_keeper,
    )

    result = manager._start_keeper(
        allocation,
        volume,
        inert_keeper,
        object(),
        object(),
    )

    assert result is (running_keeper if advanced else None)
    assert authority.calls == [("container", "start", inert_keeper.container_id)]
    assert authority.exclusion.current_checks == 2


@pytest.mark.parametrize("advanced", (False, True))
def test_main_creation_uses_fresh_suffix_not_the_docker_response(
    docker_settings,
    monkeypatch,
    advanced,
):
    allocation, _prepared = _allocation(docker_settings)
    keeper_only = _inventory(allocation, volume=True, keeper=True)
    complete = _inventory(allocation, volume=True, keeper=True, main=True)
    inventories = [keeper_only, complete if advanced else keeper_only]
    if advanced:
        inventories.append(complete)
    manager = object.__new__(DockerRunActionPreparationManager)
    manager._resource_manager = _InventorySequenceResourceManager(
        docker_settings,
        inventories,
    )
    manager._launch_settings = None
    authority = _StageMutationAuthority(docker_settings)
    volume = object()
    keeper = object()
    prepared_volume = object()
    prepared = object()
    helper = object()
    init_source = object()
    projected_arguments = ("container", "create", "--name", "main")
    monkeypatch.setattr(
        preparation_module,
        "_preparation_authority",
        lambda _manager: authority,
    )
    monkeypatch.setattr(
        preparation_module,
        "_observe_volume",
        lambda *_arguments: volume,
    )
    monkeypatch.setattr(
        preparation_module,
        "observe_supervisor_helper",
        lambda _policy: helper,
    )
    monkeypatch.setattr(
        preparation_module,
        "observe_docker_init_source",
        lambda _policy: init_source,
    )
    monkeypatch.setattr(
        preparation_module,
        "observe_running_keeper",
        lambda *_arguments: keeper,
    )
    monkeypatch.setattr(
        preparation_module,
        "adopt_prepared_runtime_volume_layout",
        lambda *_arguments, **_keywords: prepared_volume,
    )
    monkeypatch.setattr(
        preparation_module,
        "require_run_action_image",
        lambda *_arguments: None,
    )
    monkeypatch.setattr(
        preparation_module,
        "main_create_arguments",
        lambda *_arguments: projected_arguments,
    )
    monkeypatch.setattr(
        preparation_module,
        "_observe_complete_prepared",
        lambda **_arguments: prepared,
    )

    result = manager._create_main(
        allocation,
        _COMMAND,
        volume,
        keeper,
        prepared_volume,
    )

    assert result is (prepared if advanced else None)
    assert authority.calls == [projected_arguments]
    assert authority.exclusion.current_checks == 2
