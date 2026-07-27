"""Formal canonical bound for one future workload release receipt."""

from __future__ import annotations

from dataclasses import fields

import pytest

import kapso.cross_run.launch.run_action_release_envelope as envelope_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierInitProcessObservation,
    RunActionBarrierRunningContainerObservation,
    RunActionBarrierWrapperProcessObservation,
    RunActionMountInfoSnapshot,
    RunActionResolvedFileObservation,
    RunActionResolvedMountRootObservation,
    RunActionResolvedWorkloadObservation,
    RunActionResolvedWorkspaceObservation,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionCredentialValidityObservation,
    RunActionReleaseAuthorizationObservation,
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_release_envelope import (
    RunActionReleaseEnvelopeError,
    workload_release_receipt_size_bound,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionCredentialMode,
)
from test_run_action_barrier_contracts import _resolved_graph
from test_run_action_release_contracts import (
    _release_receipt,
    _remint,
    _security_observation,
)
from test_run_action_supervisor_contracts import (
    _claim,
    _execution_policy,
    _prepared_execution,
    _spawn_commit,
)

_TOPOLOGIES = (
    (RunFrontierWorkspaceAccess.NONE, RunActionCredentialMode.NONE, 6, 2, False),
    (
        RunFrontierWorkspaceAccess.NONE,
        RunActionCredentialMode.SUPERVISOR_FILE,
        7,
        3,
        False,
    ),
    (
        RunFrontierWorkspaceAccess.READ_ONLY,
        RunActionCredentialMode.NONE,
        7,
        2,
        True,
    ),
    (
        RunFrontierWorkspaceAccess.READ_ONLY,
        RunActionCredentialMode.SUPERVISOR_FILE,
        8,
        3,
        True,
    ),
    (
        RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        RunActionCredentialMode.NONE,
        7,
        2,
        True,
    ),
    (
        RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        RunActionCredentialMode.SUPERVISOR_FILE,
        8,
        3,
        True,
    ),
)


def _prepared_for(
    security,
    workspace_access,
    credential_mode,
    *,
    process_snapshot_size_bytes=None,
):
    policy = _execution_policy(
        kind=(
            RunFrontierActionKind.EMBEDDING
            if workspace_access is RunFrontierWorkspaceAccess.NONE
            else RunFrontierActionKind.CODING_AGENT
        ),
        workspace_access=workspace_access,
        credential_mode=credential_mode,
    )
    if process_snapshot_size_bytes is not None:
        limits = _remint(
            policy.supervisor_limits,
            process_snapshot_size_bytes=process_snapshot_size_bytes,
        )
        policy = _remint(policy, supervisor_limits=limits)
    return _prepared_execution(
        claim=_claim(
            policy=policy,
            security_observation_id=security.observation_id,
        )
    )


@pytest.mark.parametrize(
    (
        "workspace_access",
        "credential_mode",
        "root_count",
        "file_count",
        "has_workspace",
    ),
    _TOPOLOGIES,
)
def test_release_receipt_envelope_covers_every_topology_and_actual_receipt(
    workspace_access,
    credential_mode,
    root_count,
    file_count,
    has_workspace,
):
    security = _security_observation()
    prepared = _prepared_for(
        security,
        workspace_access,
        credential_mode,
    )
    resolved = _resolved_graph(prepared=prepared)
    activation = resolved.activation_revalidation_receipt

    first = workload_release_receipt_size_bound(
        prepared_execution=prepared,
        spawn_commit=activation.spawn_commit,
        required_security_observation=security,
    )
    second = workload_release_receipt_size_bound(
        prepared_execution=prepared,
        spawn_commit=activation.spawn_commit,
        required_security_observation=security,
    )
    receipt = _release_receipt(
        resolved,
        security,
        credential_validity=(
            credential_mode is RunActionCredentialMode.SUPERVISOR_FILE
        ),
    )
    activation_wire = envelope_module.activation_revalidation_receipt_wire_bound(
        prepared,
        activation.spawn_commit,
    )
    resolved_wire = envelope_module._resolved_workload_wire(
        prepared,
        activation.spawn_commit,
        activation_wire,
    )

    assert first == second
    assert len(receipt.to_json_bytes()) <= first
    assert (
        first
        <= prepared.preparation_claim.execution_policy.supervisor_limits.release_receipt_size_bytes
    )
    assert len(resolved_wire["resolved_mount_root_observations"]) == root_count
    assert len(resolved_wire["resolved_file_observations"]) == file_count
    assert (
        resolved_wire["resolved_workspace_observation"] is not None
    ) is has_workspace


def test_release_receipt_envelope_counts_complete_escaped_utf8_security():
    short_security = _security_observation()
    subjects = tuple(
        sorted(
            content_id("release-envelope-subject", {"position": position})
            for position in range(64)
        )
    )
    long_security = _remint(
        short_security,
        release_attestation_ref=('attestations/é/"escaped"/\\branch/' * 128),
        checked_subject_ids=subjects,
    )
    short_prepared = _prepared_for(
        short_security,
        RunFrontierWorkspaceAccess.READ_ONLY,
        RunActionCredentialMode.NONE,
    )
    long_prepared = _prepared_for(
        long_security,
        RunFrontierWorkspaceAccess.READ_ONLY,
        RunActionCredentialMode.NONE,
    )

    short_bound = workload_release_receipt_size_bound(
        prepared_execution=short_prepared,
        spawn_commit=_spawn_commit(short_prepared),
        required_security_observation=short_security,
    )
    long_bound = workload_release_receipt_size_bound(
        prepared_execution=long_prepared,
        spawn_commit=_spawn_commit(long_prepared),
        required_security_observation=long_security,
    )

    assert long_bound > short_bound
    with pytest.raises(RunActionReleaseEnvelopeError, match="security differs"):
        workload_release_receipt_size_bound(
            prepared_execution=short_prepared,
            spawn_commit=_spawn_commit(short_prepared),
            required_security_observation=long_security,
        )


def test_release_receipt_envelope_base64_arithmetic_covers_every_modulus():
    security = _security_observation()
    bounds = []
    for process_snapshot_size_bytes in (3, 4, 5):
        prepared = _prepared_for(
            security,
            RunFrontierWorkspaceAccess.READ_ONLY,
            RunActionCredentialMode.NONE,
            process_snapshot_size_bytes=process_snapshot_size_bytes,
        )
        bounds.append(
            workload_release_receipt_size_bound(
                prepared_execution=prepared,
                spawn_commit=_spawn_commit(prepared),
                required_security_observation=security,
            )
        )

    assert envelope_module._mount_info_base64_size(3) == 4
    assert envelope_module._mount_info_base64_size(4) == 8
    assert envelope_module._mount_info_base64_size(5) == 8
    assert bounds[1] - bounds[0] == 4
    assert bounds[2] == bounds[1]
    near_unsigned_64 = RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER - 1
    snapshot_wire = envelope_module._mount_info_snapshot_wire(near_unsigned_64)
    assert snapshot_wire["raw_payload_base64"] == ""
    assert envelope_module._mount_info_base64_size(near_unsigned_64) == (
        4 * ((near_unsigned_64 + 2) // 3)
    )


def test_release_receipt_envelope_maximizes_future_physical_values():
    security = _security_observation()
    prepared = _prepared_for(
        security,
        RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        RunActionCredentialMode.SUPERVISOR_FILE,
    )
    spawn = _spawn_commit(prepared)
    activation = envelope_module.activation_revalidation_receipt_wire_bound(
        prepared,
        spawn,
    )
    resolved = envelope_module._resolved_workload_wire(
        prepared,
        spawn,
        activation,
    )
    authorization = envelope_module._release_authorization_wire(
        activation,
        prepared,
        security,
    )

    assert resolved["running_container_observation"]["init_process_id"] == (
        RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
    )
    for process_name in (
        "init_process_observation",
        "wrapper_process_observation",
    ):
        process = resolved[process_name]
        assert {
            process[field_name]
            for field_name in (
                "process_id",
                "parent_process_id",
                "process_start_time_ticks",
                "mount_namespace_device",
                "mount_namespace_inode",
                "process_id_namespace_device",
                "process_id_namespace_inode",
                "root_mount_id",
                "root_device_major",
                "root_device_minor",
                "root_device",
                "root_inode",
                "executable_mount_id",
                "executable_device",
                "executable_inode",
            )
        } == {RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER}
    for root in resolved["resolved_mount_root_observations"]:
        assert {
            root[field_name]
            for field_name in (
                "source_mount_id",
                "source_device",
                "source_inode",
                "resolved_mount_id",
                "resolved_device",
                "resolved_inode",
                "mount_namespace_device",
                "mount_namespace_inode",
            )
        } == {RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER}
    for observed in resolved["resolved_file_observations"]:
        assert {
            observed[field_name] for field_name in ("mount_id", "device", "inode")
        } == {RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER}
    assert {
        authorization[field_name]
        for field_name in (
            "authorized_at_boottime_nanoseconds",
            "authorized_at_realtime_nanoseconds",
        )
    } == {RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER}
    credential = authorization["credential_validity_observation"]
    assert {
        credential[field_name]
        for field_name in (
            "observed_at_realtime_nanoseconds",
            "valid_until_realtime_nanoseconds",
        )
    } == {RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER}


@pytest.mark.parametrize(
    "contract_type",
    (
        RunActionWorkloadReleaseReceipt,
        RunActionReleaseAuthorizationObservation,
        RunActionCredentialValidityObservation,
        RunActionResolvedWorkloadObservation,
        RunActionBarrierRunningContainerObservation,
        RunActionBarrierInitProcessObservation,
        RunActionBarrierWrapperProcessObservation,
        RunActionMountInfoSnapshot,
        RunActionResolvedMountRootObservation,
        RunActionResolvedFileObservation,
        RunActionResolvedWorkspaceObservation,
    ),
)
def test_release_receipt_envelope_schema_guards_every_synthesized_contract(
    contract_type,
):
    values = {field.name: None for field in fields(contract_type)}
    assert envelope_module._sealed_wire(contract_type, **values) == values
    missing = dict(values)
    missing.pop(next(iter(missing)))
    with pytest.raises(RunActionReleaseEnvelopeError, match="fields changed"):
        envelope_module._sealed_wire(contract_type, **missing)
    with pytest.raises(RunActionReleaseEnvelopeError, match="fields changed"):
        envelope_module._sealed_wire(
            contract_type,
            **values,
            schema_drift=None,
        )


def test_release_receipt_envelope_rejects_spawn_substitution():
    security = _security_observation()
    prepared = _prepared_for(
        security,
        RunFrontierWorkspaceAccess.READ_ONLY,
        RunActionCredentialMode.NONE,
    )
    foreign = _prepared_for(
        security,
        RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        RunActionCredentialMode.NONE,
    )

    with pytest.raises(ValueError, match="spawn differs"):
        workload_release_receipt_size_bound(
            prepared_execution=prepared,
            spawn_commit=_spawn_commit(foreign),
            required_security_observation=security,
        )
