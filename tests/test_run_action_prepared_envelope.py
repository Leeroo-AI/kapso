"""Pre-mutation proof for the complete run-action event-3 wire shape."""

from __future__ import annotations

from dataclasses import replace

import pytest

import kapso.cross_run.launch.run_action_prepared_envelope as envelope_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
    main_barrier_command,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
)
from kapso.cross_run.launch.run_action_prepared_envelope import (
    prepared_execution_event_size_bound,
    RunActionPreparedEnvelopeError,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionContainerLabel,
    RunActionCredentialMode,
    RunActionPreparationAllocation,
    RunActionSupervisorContractError,
    issue_runtime_volume_authority,
    run_action_keeper_process_cgroup_path,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_docker_projection import _policy
from test_run_action_supervisor_contracts import (
    _claim,
    _prepared_execution,
    _remint_contract,
    _remint_policy,
    _remint_resource_limits,
    _remint_sandbox,
    _volume_authority,
)

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_PREDECESSOR_EVENT_ID = content_id(
    RunActionExecutionEvent.CONTENT_NAMESPACE,
    {"fixture": "prepared-envelope-predecessor"},
)


@pytest.fixture(scope="module")
def docker_settings():
    return CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker


def _envelope_case(
    docker_settings,
    *,
    workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
    credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    command_arguments=("default",),
    cgroup_parent_id=None,
):
    command = DockerRunActionCommand.build(
        entrypoint="/bin/tool",
        arguments=command_arguments,
    )
    policy = _policy(
        docker_settings,
        workspace_access=workspace_access,
        credential_mode=credential_mode,
        command_template_id=command.command_template_id,
    )
    if cgroup_parent_id is not None:
        policy = _remint_policy(
            policy,
            sandbox_spec=_remint_sandbox(
                policy.sandbox_spec,
                cgroup_parent_id=cgroup_parent_id,
            ),
        )
    claim = _claim(policy=policy)
    authority = _volume_authority(claim, nonce="1" * 32)
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=claim,
        runtime_volume_authority=authority,
    )
    return command, claim, allocation


def _prepared_event_size(prepared) -> int:
    event = RunActionExecutionEvent.mint(
        event_number=3,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
        event_kind=RunActionExecutionEventKind.EXECUTION_PREPARED,
        reservation=prepared.preparation_claim.reservation,
        preparation_allocation=None,
        prepared_execution=prepared,
        spawn_commit=None,
        activation_revalidation_receipt=None,
        credential_retirement_intent=None,
        provider_termination_receipt=None,
        result_receipt=None,
        result_decision=None,
        acceptance=None,
        workspace_after=None,
    )
    return len(event.to_json_bytes())


def _count_wire_value(value, expected) -> int:
    if value == expected:
        return 1
    if isinstance(value, dict):
        return sum(_count_wire_value(child, expected) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_wire_value(child, expected) for child in value)
    return 0


@pytest.mark.parametrize(
    ("workspace_access", "credential_mode"),
    (
        (RunFrontierWorkspaceAccess.NONE, RunActionCredentialMode.NONE),
        (
            RunFrontierWorkspaceAccess.NONE,
            RunActionCredentialMode.SUPERVISOR_FILE,
        ),
        (RunFrontierWorkspaceAccess.READ_ONLY, RunActionCredentialMode.NONE),
        (
            RunFrontierWorkspaceAccess.READ_ONLY,
            RunActionCredentialMode.SUPERVISOR_FILE,
        ),
        (RunFrontierWorkspaceAccess.EDIT_WORKSPACE, RunActionCredentialMode.NONE),
        (
            RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
            RunActionCredentialMode.SUPERVISOR_FILE,
        ),
    ),
)
def test_prepared_envelope_covers_every_optional_wire_shape(
    docker_settings,
    workspace_access,
    credential_mode,
):
    command, claim, allocation = _envelope_case(
        docker_settings,
        workspace_access=workspace_access,
        credential_mode=credential_mode,
    )
    prepared = _prepared_execution(
        claim=claim,
        authority=allocation.runtime_volume_authority,
    )

    first = prepared_execution_event_size_bound(
        preparation_allocation=allocation,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
        command=command,
        runtime_settings=docker_settings,
    )
    second = prepared_execution_event_size_bound(
        preparation_allocation=allocation,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
        command=command,
        runtime_settings=docker_settings,
    )

    assert first == second
    assert _prepared_event_size(prepared) <= first


def test_prepared_envelope_covers_maximum_cgroup_path_expansion(
    docker_settings,
):
    cgroup_parent_id = "a-" * 124 + "a.slice"
    assert len(cgroup_parent_id.encode("ascii")) == 255
    command, claim, allocation = _envelope_case(
        docker_settings,
        cgroup_parent_id=cgroup_parent_id,
    )
    keeper_cgroup_path = run_action_keeper_process_cgroup_path(
        claim.execution_policy,
        "f" * 64,
    )
    prepared_wire = envelope_module._prepared_execution_wire(
        allocation,
        command,
        docker_settings,
    )
    assert len(keeper_cgroup_path) == 16_578
    assert _count_wire_value(prepared_wire, keeper_cgroup_path) == 2


def test_prepared_envelope_measures_complete_escaped_utf8_command(
    docker_settings,
):
    short_command, _claim_value, short_allocation = _envelope_case(
        docker_settings,
        command_arguments=("default",),
    )
    long_command, _claim_value, long_allocation = _envelope_case(
        docker_settings,
        command_arguments=('quote"slash\\', "λ🙂", "line\nbreak"),
    )
    short_bound = prepared_execution_event_size_bound(
        preparation_allocation=short_allocation,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
        command=short_command,
        runtime_settings=docker_settings,
    )
    long_bound = prepared_execution_event_size_bound(
        preparation_allocation=long_allocation,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
        command=long_command,
        runtime_settings=docker_settings,
    )
    short_barrier = main_barrier_command(
        short_command,
        short_allocation.runtime_volume_authority.generation_nonce,
        docker_settings,
    )
    long_barrier = main_barrier_command(
        long_command,
        long_allocation.runtime_volume_authority.generation_nonce,
        docker_settings,
    )

    assert long_bound - short_bound == 2 * (
        len(canonical_json_bytes(long_barrier))
        - len(canonical_json_bytes(short_barrier))
    )


def test_prepared_envelope_rejects_command_or_runtime_substitution(
    docker_settings,
):
    command, _claim_value, allocation = _envelope_case(docker_settings)
    foreign_command = DockerRunActionCommand.build(
        entrypoint="/bin/tool",
        arguments=("foreign",),
    )
    foreign_settings = replace(
        docker_settings,
        run_action_barrier_poll_interval_seconds=(
            docker_settings.run_action_barrier_poll_interval_seconds + 1
        ),
    )

    with pytest.raises(RunActionPreparedEnvelopeError, match="Docker execution"):
        prepared_execution_event_size_bound(
            preparation_allocation=allocation,
            predecessor_event_id=_PREDECESSOR_EVENT_ID,
            command=foreign_command,
            runtime_settings=docker_settings,
        )
    with pytest.raises(RunActionPreparedEnvelopeError, match="Docker execution"):
        prepared_execution_event_size_bound(
            preparation_allocation=allocation,
            predecessor_event_id=_PREDECESSOR_EVENT_ID,
            command=command,
            runtime_settings=foreign_settings,
        )


def test_prepared_envelope_schema_guard_fails_loud():
    with pytest.raises(
        RunActionPreparedEnvelopeError,
        match="envelope fields changed",
    ):
        envelope_module._sealed_wire(
            RunActionContainerLabel,
            key="com.kapso.test",
        )
    with pytest.raises(
        RunActionPreparedEnvelopeError,
        match="envelope fields changed",
    ):
        envelope_module._sealed_wire(
            RunActionContainerLabel,
            key="com.kapso.test",
            value="present",
            added_field="schema-drift",
        )


def test_event_three_physical_integer_width_is_contractually_bounded(
    docker_settings,
):
    command, claim, allocation = _envelope_case(docker_settings)
    prepared = _prepared_execution(
        claim=claim,
        authority=allocation.runtime_volume_authority,
    )
    main_projection = prepared.inert_container_evidence.issued_create_projection
    keeper = prepared.volume_keeper_evidence
    keeper_projection = keeper.issued_create_projection
    over_limit = RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER + 1
    physical_field_cases = (
        (prepared.runtime_volume_evidence.sentinel_evidence, "mount_id"),
        (prepared.runtime_volume_evidence, "keeper_process_id"),
        (prepared.input_delivery_slot, "mount_id"),
        (prepared.result_directory, "mount_id"),
        (prepared.result_file, "mount_id"),
        (prepared.workspace_proof, "mount_id"),
        (main_projection.supervisor_helper_evidence, "mount_id"),
        (main_projection.docker_init_source_evidence, "mount_id"),
        (keeper.mounted_helper_evidence, "process_start_time_ticks"),
        (keeper, "process_id"),
        (main_projection, "nonauthoritative_raw_field_count"),
        (keeper_projection, "nonauthoritative_raw_field_count"),
    )

    for contract, field_name in physical_field_cases:
        with pytest.raises(RunActionSupervisorContractError):
            _remint_contract(contract, **{field_name: over_limit})

    helper = main_projection.supervisor_helper_evidence
    maximum_width_helper = _remint_contract(
        helper,
        mount_id=RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
        device=RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
        inode=RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    )
    assert maximum_width_helper.mount_id == RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
    assert prepared_execution_event_size_bound(
        preparation_allocation=allocation,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
        command=command,
        runtime_settings=docker_settings,
    ) >= _prepared_event_size(prepared)


def test_runtime_volume_authority_rejects_integer_wider_than_event_envelope(
    docker_settings,
):
    command, _claim_value, allocation = _envelope_case(docker_settings)
    policy = allocation.preparation_claim.execution_policy
    widened_limits = _remint_resource_limits(
        policy.docker_resource_limits,
        runtime_volume_size_bytes=RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER + 1,
    )
    widened_policy = _remint_policy(
        policy,
        docker_resource_limits=widened_limits,
    )
    widened_claim = _claim(policy=widened_policy)

    with pytest.raises(
        RunActionSupervisorContractError,
        match="runtime volume authority",
    ):
        issue_runtime_volume_authority(widened_claim, "1" * 32)

    assert (
        prepared_execution_event_size_bound(
            preparation_allocation=allocation,
            predecessor_event_id=_PREDECESSOR_EVENT_ID,
            command=command,
            runtime_settings=docker_settings,
        )
        > 0
    )
