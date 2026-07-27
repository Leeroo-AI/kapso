from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest

import kapso.cross_run.launch.run_action_docker_cleanup as cleanup_module
import kapso.cross_run.launch.run_action_resource_finalization as finalization_module
import kapso.cross_run.launch.run_action_store as run_action_store_module
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_credential_broker import (
    RunActionCredentialLeaseStatus,
)
from kapso.cross_run.launch.run_action_credential_contracts import (
    RunActionCredentialRetirementIntent,
    RunActionPreReleaseCredentialObservation,
    RunActionPreReleaseCredentialState,
)
from kapso.cross_run.launch.run_action_docker_cleanup import (
    DockerRunActionCleanupManager,
    issue_docker_run_action_resource_finalization_authority,
    RunActionDockerCleanupError,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionInertKeeperObservation,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionExecutionEventKind,
)
from kapso.cross_run.launch.run_action_resource_finalization import (
    RunActionResourceFinalizationAuthority,
    RunActionResourceFinalizationError,
)
from kapso.cross_run.launch.run_action_store import (
    _RUN_ACTION_RECOVERY_AUTHORITY,
    RunActionResultDisposition,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
    run_action_credential_lease_request,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionPreReleaseMainTerminalObservation,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from test_launch_resolver import resolver_case
from test_run_action_docker_inspect import (
    _context,
    docker_settings,
)
from test_run_action_docker_resources import resource_context
from test_run_frontier_action_gate import (
    _boundary_identity as _frontier_boundary_identity,
)
from test_run_action_release_contracts import _release_adoption_for_event
from test_run_action_recovery import (
    _action_case,
    _activation_revalidation_receipt,
    _append_provider_terminated,
    _append_result_accepted,
    _append_result_received,
    _FakeExecutionAdapter,
    _remint_contract,
    _reserved_case,
    _terminal_observation,
)
from test_run_action_termination_contracts import (
    _pre_release_loss,
    _pre_release_terminal,
)
from test_run_action_supervisor_contracts import _execution_policy
from test_run_state_publisher import publisher_case


class _StaticCleanupControlLease:
    def __init__(
        self,
        *,
        topology,
        workload_release_adoption=None,
        timeout_directive_publication=None,
    ):
        self.topology = topology
        self.workload_release_adoption = workload_release_adoption
        self.timeout_directive_publication = timeout_directive_publication
        self.current_checks = 0

    def require_current(self):
        self.current_checks += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False


def _cleanup_commands(runner):
    return tuple(
        request.argv[5:]
        for request in runner.requests
        if request.argv[5:7]
        in {
            ("container", "rm"),
            ("volume", "rm"),
        }
    )


def _reserve_cleanup_operation(publisher_case, resource_manager):
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    execution_policy = _remint_contract(
        _execution_policy(
            kind=RunFrontierActionKind.CODING_AGENT,
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        ),
        docker_runtime_settings_digest=tree_or_blob_digest(
            resource_manager.runtime_settings.to_json_bytes()
        ),
    )
    base_boundary_identity = _frontier_boundary_identity(
        RunFrontierActionKind.CODING_AGENT
    )
    boundary_identity = _remint_contract(
        base_boundary_identity,
        execution_lifecycle_identity=_remint_contract(
            base_boundary_identity.execution_lifecycle_identity,
            execution_policy_id=execution_policy.docker_execution_policy_id,
        ),
    )
    reservation = gate.reserve(
        frontier,
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="cleanup_agent_call_0123456789abcdef",
        request_payload=b'{"prompt":"cleanup-state-machine"}',
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        boundary_identity=boundary_identity,
    )
    return gate, reservation, execution_policy


def _append_prepared_cleanup_prefix(gate, reservation, execution_policy):
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        allocation = session.allocate_preparation(execution_policy)
        prepared = _FakeExecutionAdapter._prepared_for_allocation(allocation)
        session.commit_prepared_execution(prepared)
    return allocation, prepared


def _append_cleanup_terminal(
    gate,
    reservation,
    execution_policy,
    terminal_kind,
):
    allocation, prepared = _append_prepared_cleanup_prefix(
        gate,
        reservation,
        execution_policy,
    )
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        spawn = session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )
        activation = _activation_revalidation_receipt(prepared, spawn)
        session.commit_activation(activation)
        adoption = _release_adoption_for_event(
            session.events[4],
            gate._security_authority.observation,
        )
        if terminal_kind == "result_accepted":
            result_payload = b'{"provider":"cleanup-result"}'
            result = _FakeExecutionAdapter._provider_result(
                prepared,
                spawn,
                activation,
                result_payload,
                adoption,
            )
            session.record_result(
                spawn_commit=spawn,
                workload_release_adoption=adoption,
                terminal_observation=result.terminal_observation,
                result_capture_receipt=result.result_capture_receipt,
                result_payload=result_payload,
            )
            session.decide_result(
                result_interpreter_identity=(
                    reservation.intent.boundary_identity.result_interpreter_identity
                ),
                disposition=RunActionResultDisposition.SUCCEEDED,
                accepted_result_payload=b'{"accepted":"cleanup-result"}',
                workspace_promotion=None,
            )
            workspace_binding = reservation.frontier.workspace_before
            session.accept_decision(
                workspace_after=(
                    None
                    if workspace_binding is None
                    else workspace_binding.to_identity()
                )
            )
        elif terminal_kind == "provider_terminated":
            terminal = _remint_contract(
                _terminal_observation(prepared, spawn, adoption),
                exit_code=137,
                oom_killed=True,
            )
            session.terminate_provider(
                RunActionProviderTerminationReceipt.mint(
                    disposition=RunActionProviderTerminationDisposition.FAILED,
                    reason=RunActionProviderTerminationReason.OOM,
                    activation_event_id=session.events[4].event_id,
                    workload_release_adoption=adoption,
                    terminal_observation=terminal,
                    timeout_directive_publication=None,
                    empty_result_capture_receipt=None,
                    pre_release_main_loss_observation=None,
                    credential_retirement_intent=None,
                )
            )
        elif terminal_kind == "pre_release_terminal":
            released_terminal = _terminal_observation(prepared, spawn, adoption)
            pre_release_terminal = _pre_release_terminal(
                activation,
                session.events[4].event_id,
                released_terminal,
            )
            session.terminate_provider(
                RunActionProviderTerminationReceipt.mint(
                    disposition=RunActionProviderTerminationDisposition.FAILED,
                    reason=(
                        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL
                    ),
                    activation_event_id=session.events[4].event_id,
                    workload_release_adoption=None,
                    terminal_observation=pre_release_terminal,
                    timeout_directive_publication=None,
                    empty_result_capture_receipt=None,
                    pre_release_main_loss_observation=None,
                    credential_retirement_intent=None,
                )
            )
        elif terminal_kind == "pre_release_loss":
            loss = _pre_release_loss(
                activation,
                session.events[4].event_id,
            )
            session.terminate_provider(
                RunActionProviderTerminationReceipt.mint(
                    disposition=RunActionProviderTerminationDisposition.FAILED,
                    reason=(RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS),
                    activation_event_id=session.events[4].event_id,
                    workload_release_adoption=None,
                    terminal_observation=None,
                    timeout_directive_publication=None,
                    empty_result_capture_receipt=None,
                    pre_release_main_loss_observation=loss,
                    credential_retirement_intent=None,
                )
            )
        elif terminal_kind == "credential_expired":
            request = run_action_credential_lease_request(prepared, spawn)
            status = RunActionCredentialLeaseStatus.mint(
                credential_lease_request_id=request.credential_lease_request_id,
                valid_until_realtime_nanoseconds=1,
            )
            supervisor_limits = (
                prepared.preparation_claim.execution_policy.supervisor_limits
            )
            required_valid_until = (
                2
                + (
                    supervisor_limits.execution_timeout_seconds
                    + supervisor_limits.termination_grace_seconds
                )
                * 1_000_000_000
            )
            credential_observation = RunActionPreReleaseCredentialObservation.mint(
                state=RunActionPreReleaseCredentialState.EXPIRED,
                activation_revalidation_receipt=activation,
                credential_lease_status=status,
                observed_before_realtime_nanoseconds=1,
                observed_after_realtime_nanoseconds=2,
                required_valid_until_realtime_nanoseconds=(required_valid_until),
            )
            intent = RunActionCredentialRetirementIntent.mint(
                activation_event_id=session.events[4].event_id,
                pre_release_credential_observation_id=(
                    credential_observation.pre_release_credential_observation_id
                ),
                credential_lease_status=status,
                observed_before_realtime_nanoseconds=1,
                observed_after_realtime_nanoseconds=2,
                required_valid_until_realtime_nanoseconds=required_valid_until,
            )
            session.commit_credential_retirement(intent)
            released_terminal = _terminal_observation(
                prepared,
                spawn,
                adoption,
            )
            pre_release_terminal = _pre_release_terminal(
                activation,
                session.events[4].event_id,
                released_terminal,
            )
            session.terminate_provider(
                RunActionProviderTerminationReceipt.mint(
                    disposition=(RunActionProviderTerminationDisposition.INTERRUPTED),
                    reason=RunActionProviderTerminationReason.CREDENTIAL_EXPIRED,
                    activation_event_id=session.events[4].event_id,
                    workload_release_adoption=None,
                    terminal_observation=pre_release_terminal,
                    timeout_directive_publication=None,
                    empty_result_capture_receipt=None,
                    pre_release_main_loss_observation=None,
                    credential_retirement_intent=intent,
                )
            )
        else:
            raise AssertionError("test terminal kind is unsupported")
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    return allocation, prepared, events


def _install_cleanup_resources(
    runner,
    allocation,
    prepared,
    *,
    main_role_proof=None,
):
    claim = allocation.preparation_claim
    authority = allocation.runtime_volume_authority
    keeper = prepared.volume_keeper_evidence
    main = prepared.inert_container_evidence
    runner.volumes[preparation_volume_name(claim)] = {
        "Name": preparation_volume_name(claim),
        "Labels": {
            label.key: label.value
            for label in preparation_volume_labels(claim, authority.generation_nonce)
        },
        "RoleProof": prepared.runtime_volume_evidence.docker_volume_occurrence_digest,
    }
    runner.containers[keeper.container_id] = {
        "Config": {
            "Labels": {
                label.key: label.value
                for label in preparation_keeper_container_labels(claim)
            }
        },
        "Id": keeper.container_id,
        "Name": f"/{preparation_keeper_container_name(claim)}",
        "RoleProof": keeper.volume_keeper_evidence_id,
        "State": {"Status": "running"},
    }
    runner.containers[main.container_id] = {
        "Config": {
            "Labels": {
                label.key: label.value for label in preparation_container_labels(claim)
            }
        },
        "Id": main.container_id,
        "Name": f"/{preparation_container_name(claim)}",
        "RoleProof": (
            main.inert_container_evidence_id
            if main_role_proof is None
            else main_role_proof
        ),
        "State": {"Status": "created"},
    }
    runner.running_container_ids.add(keeper.container_id)


def _install_cleanup_proof_adapters(monkeypatch, prepared, events):
    evidence = cleanup_module._terminal_cleanup_evidence(events)

    def observe_volume(raw, claim, authority, settings):
        assert claim == prepared.preparation_claim
        assert authority == prepared.runtime_volume_authority
        return SimpleNamespace(volume_occurrence_digest=raw["RoleProof"])

    def observe_keeper(
        raw,
        claim,
        authority,
        volume,
        helper_evidence,
        init_source_evidence,
        settings,
    ):
        assert claim == prepared.preparation_claim
        assert authority == prepared.runtime_volume_authority
        if (
            raw["RoleProof"]
            != prepared.volume_keeper_evidence.volume_keeper_evidence_id
        ):
            raise RunActionDockerCleanupError(
                "run-action cleanup keeper differs from its durable occurrence"
            )
        return prepared.volume_keeper_evidence

    def reobserve_terminal(raw, terminal):
        if raw["RoleProof"] != terminal.complete_inspection_digest:
            raise RunActionDockerCleanupError(
                "run-action cleanup main differs from durable terminal authority"
            )
        return terminal

    def reobserve_pre_release_terminal(raw, terminal):
        if raw["RoleProof"] != terminal.complete_inspection_digest:
            raise RunActionDockerCleanupError(
                "run-action cleanup main differs from durable terminal authority"
            )
        return terminal

    monkeypatch.setattr(cleanup_module, "observe_runtime_volume", observe_volume)
    monkeypatch.setattr(cleanup_module, "observe_running_keeper", observe_keeper)
    monkeypatch.setattr(
        cleanup_module,
        "reobserve_terminal_main_container_for_cleanup",
        reobserve_terminal,
    )
    monkeypatch.setattr(
        cleanup_module,
        "reobserve_pre_release_terminal_main_container_for_cleanup",
        reobserve_pre_release_terminal,
    )
    monkeypatch.setattr(
        cleanup_module,
        "open_run_action_timeout_inspection",
        lambda **_arguments: _StaticCleanupControlLease(
            topology=evidence.topology,
            workload_release_adoption=evidence.workload_release_adoption,
            timeout_directive_publication=evidence.timeout_directive_publication,
        ),
    )


def _install_prepared_cleanup_proof_adapters(monkeypatch, prepared):
    def observe_volume(raw, claim, authority, settings):
        assert claim == prepared.preparation_claim
        assert authority == prepared.runtime_volume_authority
        return SimpleNamespace(volume_occurrence_digest=raw["RoleProof"])

    def observe_keeper(
        raw,
        claim,
        authority,
        volume,
        helper_evidence,
        init_source_evidence,
        settings,
    ):
        if (
            raw["RoleProof"]
            != prepared.volume_keeper_evidence.volume_keeper_evidence_id
        ):
            raise RunActionDockerCleanupError(
                "invalidated keeper differs from its durable occurrence"
            )
        return prepared.volume_keeper_evidence

    def observe_inert(
        raw,
        claim,
        authority,
        volume,
        command,
        helper_evidence,
        init_source_evidence,
        settings,
    ):
        if (
            raw["RoleProof"]
            != prepared.inert_container_evidence.inert_container_evidence_id
        ):
            raise RunActionDockerCleanupError(
                "invalidated main differs from its durable inert occurrence"
            )
        return prepared.inert_container_evidence

    monkeypatch.setattr(cleanup_module, "observe_runtime_volume", observe_volume)
    monkeypatch.setattr(cleanup_module, "observe_running_keeper", observe_keeper)
    monkeypatch.setattr(cleanup_module, "observe_inert_main_container", observe_inert)
    monkeypatch.setattr(
        cleanup_module,
        "open_run_action_control_directory",
        lambda _prepared: _StaticCleanupControlLease(
            topology=RunActionControlDirectoryTopology.EMPTY
        ),
    )


def _install_allocation_cleanup_proof_adapters(
    monkeypatch,
    allocation,
    prepared,
):
    def observe_volume(raw, claim, authority, settings):
        assert claim == allocation.preparation_claim
        assert authority == allocation.runtime_volume_authority
        return SimpleNamespace(
            volume_name=authority.volume_name,
            volume_occurrence_digest=raw["RoleProof"],
        )

    def observe_helper(policy):
        assert policy == allocation.preparation_claim.execution_policy
        return SimpleNamespace(role="helper")

    def observe_init(policy):
        assert policy == allocation.preparation_claim.execution_policy
        return SimpleNamespace(role="init")

    def observe_keeper(
        raw,
        claim,
        authority,
        volume,
        helper_evidence,
        init_source_evidence,
        settings,
    ):
        assert claim == allocation.preparation_claim
        assert authority == allocation.runtime_volume_authority
        if (
            raw["RoleProof"]
            != prepared.volume_keeper_evidence.volume_keeper_evidence_id
        ):
            raise RunActionDockerCleanupError(
                "allocation-stage keeper differs from issued projection"
            )
        status = raw["State"]["Status"]
        if status == "created":
            projection = prepared.volume_keeper_evidence.issued_create_projection
            return DockerRunActionInertKeeperObservation(
                container_id=raw["Id"],
                issued_create_projection=projection,
                observed_inspect_projection=projection,
            )
        if status == "running":
            return prepared.volume_keeper_evidence
        raise RunActionDockerCleanupError(
            "allocation-stage keeper lifecycle is not removable"
        )

    def observe_main(
        raw,
        claim,
        authority,
        volume,
        helper_evidence,
        init_source_evidence,
        settings,
    ):
        if (
            raw["State"]["Status"] != "created"
            or raw["RoleProof"]
            != prepared.inert_container_evidence.inert_container_evidence_id
        ):
            raise RunActionDockerCleanupError(
                "allocation-stage main is not an exact inert occurrence"
            )
        return prepared.inert_container_evidence

    monkeypatch.setattr(cleanup_module, "observe_runtime_volume", observe_volume)
    monkeypatch.setattr(cleanup_module, "observe_supervisor_helper", observe_helper)
    monkeypatch.setattr(cleanup_module, "observe_docker_init_source", observe_init)
    monkeypatch.setattr(cleanup_module, "observe_allocation_keeper", observe_keeper)
    monkeypatch.setattr(
        cleanup_module,
        "observe_allocation_inert_main_container",
        observe_main,
    )


def _finalization_authority(
    publisher_case,
    gate,
    resource_manager,
    runner,
):
    return issue_docker_run_action_resource_finalization_authority(
        action_store=gate._action_store,
        launch_settings=publisher_case["settings"],
        resource_manager=resource_manager,
        cleanup_manager=DockerRunActionCleanupManager(runner.runtime),
    )


@pytest.mark.parametrize(
    ("terminal_kind", "terminal_event_kind"),
    (
        ("result_accepted", RunActionExecutionEventKind.RESULT_ACCEPTED),
        ("provider_terminated", RunActionExecutionEventKind.PROVIDER_TERMINATED),
        ("pre_release_terminal", RunActionExecutionEventKind.PROVIDER_TERMINATED),
        ("credential_expired", RunActionExecutionEventKind.PROVIDER_TERMINATED),
    ),
)
def test_terminal_cleanup_removes_exact_physical_suffix(
    publisher_case,
    resource_context,
    monkeypatch,
    terminal_kind,
    terminal_event_kind,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared, events = _append_cleanup_terminal(
        gate,
        reservation,
        execution_policy,
        terminal_kind,
    )
    terminal = cleanup_module._terminal_cleanup_evidence(events).terminal_observation
    terminal_container = (
        terminal.terminal_container_observation
        if type(terminal) is RunActionPreReleaseMainTerminalObservation
        else terminal
    )
    _install_cleanup_resources(
        runner,
        allocation,
        prepared,
        main_role_proof=terminal_container.complete_inspection_digest,
    )
    _install_cleanup_proof_adapters(monkeypatch, prepared, events)

    _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    ).finalize_terminal(reservation.intent.operation_id)

    assert events[-1].event_kind is terminal_event_kind
    assert not runner.containers
    assert not runner.volumes
    assert _cleanup_commands(runner) == (
        ("container", "rm", prepared.inert_container_evidence.container_id),
        (
            "container",
            "rm",
            "--force",
            prepared.volume_keeper_evidence.container_id,
        ),
        ("volume", "rm", allocation.runtime_volume_authority.volume_name),
    )


@pytest.mark.parametrize(
    ("remaining_suffix", "expected_roles"),
    (
        ("keeper_and_volume", ("keeper", "volume")),
        ("volume_only", ("volume",)),
    ),
)
def test_reconstructed_finalizer_resumes_exact_crash_suffix(
    publisher_case,
    resource_context,
    monkeypatch,
    remaining_suffix,
    expected_roles,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared, events = _append_cleanup_terminal(
        gate,
        reservation,
        execution_policy,
        "result_accepted",
    )
    terminal = cleanup_module._terminal_cleanup_evidence(events).terminal_observation
    _install_cleanup_resources(
        runner,
        allocation,
        prepared,
        main_role_proof=terminal.complete_inspection_digest,
    )
    _install_cleanup_proof_adapters(monkeypatch, prepared, events)
    runner.containers.pop(prepared.inert_container_evidence.container_id)
    if remaining_suffix == "volume_only":
        runner.containers.pop(prepared.volume_keeper_evidence.container_id)
        runner.running_container_ids.clear()

    reconstructed_manager = DockerRunActionResourceManager(runner.runtime)
    reconstructed_authority = _finalization_authority(
        publisher_case,
        gate,
        reconstructed_manager,
        runner,
    )
    reconstructed_authority.finalize_terminal(reservation.intent.operation_id)

    keeper_id = prepared.volume_keeper_evidence.container_id
    volume_name = allocation.runtime_volume_authority.volume_name
    assert not runner.containers
    assert not runner.volumes
    commands_by_role = {
        "keeper": ("container", "rm", "--force", keeper_id),
        "volume": ("volume", "rm", volume_name),
    }
    assert _cleanup_commands(runner) == tuple(
        commands_by_role[role] for role in expected_roles
    )


@pytest.mark.parametrize(
    "missing_resource",
    ("keeper", "volume"),
)
def test_cleanup_rejects_non_suffix_resource_mix_before_mutation(
    publisher_case,
    resource_context,
    monkeypatch,
    missing_resource,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared, events = _append_cleanup_terminal(
        gate,
        reservation,
        execution_policy,
        "result_accepted",
    )
    terminal = cleanup_module._terminal_cleanup_evidence(events).terminal_observation
    _install_cleanup_resources(
        runner,
        allocation,
        prepared,
        main_role_proof=terminal.complete_inspection_digest,
    )
    _install_cleanup_proof_adapters(monkeypatch, prepared, events)
    if missing_resource == "keeper":
        runner.containers.pop(prepared.volume_keeper_evidence.container_id)
        runner.running_container_ids.clear()
    else:
        runner.volumes.clear()

    with pytest.raises(
        RunActionDockerCleanupError,
        match="not an exact removable suffix",
    ):
        _finalization_authority(
            publisher_case,
            gate,
            resource_manager,
            runner,
        ).finalize_terminal(reservation.intent.operation_id)

    assert not _cleanup_commands(runner)


@pytest.mark.parametrize(
    ("substituted_resource", "error_match"),
    (
        ("volume", "volume differs from its durable occurrence"),
        ("keeper", "keeper differs from its durable occurrence"),
        ("main", "main differs from durable terminal authority"),
    ),
)
def test_cleanup_rejects_substituted_occurrence_before_mutation(
    publisher_case,
    resource_context,
    monkeypatch,
    substituted_resource,
    error_match,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared, events = _append_cleanup_terminal(
        gate,
        reservation,
        execution_policy,
        "result_accepted",
    )
    terminal = cleanup_module._terminal_cleanup_evidence(events).terminal_observation
    _install_cleanup_resources(
        runner,
        allocation,
        prepared,
        main_role_proof=terminal.complete_inspection_digest,
    )
    _install_cleanup_proof_adapters(monkeypatch, prepared, events)
    if substituted_resource == "volume":
        runner.volumes[allocation.runtime_volume_authority.volume_name]["RoleProof"] = (
            "sha256:" + "f" * 64
        )
    elif substituted_resource == "keeper":
        runner.containers[prepared.volume_keeper_evidence.container_id][
            "RoleProof"
        ] = "substituted-keeper"
    else:
        runner.containers[prepared.inert_container_evidence.container_id][
            "RoleProof"
        ] = "substituted-main"

    with pytest.raises(RunActionDockerCleanupError, match=error_match):
        _finalization_authority(
            publisher_case,
            gate,
            resource_manager,
            runner,
        ).finalize_terminal(reservation.intent.operation_id)

    assert not _cleanup_commands(runner)


def test_cleanup_rejects_resource_reappearance_after_one_transition(
    publisher_case,
    resource_context,
    monkeypatch,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared, events = _append_cleanup_terminal(
        gate,
        reservation,
        execution_policy,
        "result_accepted",
    )
    terminal = cleanup_module._terminal_cleanup_evidence(events).terminal_observation
    _install_cleanup_resources(
        runner,
        allocation,
        prepared,
        main_role_proof=terminal.complete_inspection_digest,
    )
    _install_cleanup_proof_adapters(monkeypatch, prepared, events)
    main_id = prepared.inert_container_evidence.container_id
    main_payload = deepcopy(runner.containers[main_id])

    def reappear_once():
        runner.containers[main_id] = main_payload
        runner.cleanup_post_mutation = None

    runner.cleanup_post_mutation = reappear_once

    with pytest.raises(
        RunActionDockerCleanupError,
        match="did not produce the exact next physical suffix",
    ):
        _finalization_authority(
            publisher_case,
            gate,
            resource_manager,
            runner,
        ).finalize_terminal(reservation.intent.operation_id)

    assert _cleanup_commands(runner) == (("container", "rm", main_id),)
    assert main_id in runner.containers


def _append_allocation_invalidation(gate, reservation, execution_policy):
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        allocation = session.allocate_preparation(execution_policy)
        session.invalidate_frontier()
    return allocation


def test_allocation_only_invalidation_proves_stable_absence(
    publisher_case,
    resource_context,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    _append_allocation_invalidation(gate, reservation, execution_policy)
    authority = _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    )

    authority.finalize_terminal(reservation.intent.operation_id)

    assert not _cleanup_commands(runner)


@pytest.mark.parametrize(
    ("residue", "expected_commands"),
    (
        ("volume", ("volume",)),
        ("created_keeper", ("created_keeper", "volume")),
        ("running_keeper", ("running_keeper", "volume")),
        ("inert_main", ("main", "running_keeper", "volume")),
    ),
)
def test_allocation_only_invalidation_reaps_exact_partial_residue(
    publisher_case,
    resource_context,
    monkeypatch,
    residue,
    expected_commands,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation = _append_allocation_invalidation(
        gate,
        reservation,
        execution_policy,
    )
    prepared = _FakeExecutionAdapter._prepared_for_allocation(allocation)
    _install_cleanup_resources(runner, allocation, prepared)
    _install_allocation_cleanup_proof_adapters(
        monkeypatch,
        allocation,
        prepared,
    )
    main_id = prepared.inert_container_evidence.container_id
    keeper_id = prepared.volume_keeper_evidence.container_id
    volume_name = allocation.runtime_volume_authority.volume_name
    if residue != "inert_main":
        runner.containers.pop(main_id)
    if residue == "volume":
        runner.containers.pop(keeper_id)
        runner.running_container_ids.clear()
    elif residue == "created_keeper":
        runner.containers[keeper_id]["State"]["Status"] = "created"
        runner.running_container_ids.clear()
    authority = _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    )

    with pytest.raises(
        RunActionDockerCleanupError,
        match="stable absence requires an absent inventory",
    ):
        authority.require_terminal_absence(reservation.intent.operation_id)
    assert not _cleanup_commands(runner)

    authority.finalize_terminal(reservation.intent.operation_id)

    commands = {
        "main": ("container", "rm", main_id),
        "created_keeper": ("container", "rm", keeper_id),
        "running_keeper": ("container", "rm", "--force", keeper_id),
        "volume": ("volume", "rm", volume_name),
    }
    assert _cleanup_commands(runner) == tuple(
        commands[role] for role in expected_commands
    )
    assert not runner.containers
    assert not runner.volumes


@pytest.mark.parametrize(
    ("invalid_state", "error_match"),
    (
        ("created_keeper_with_main", "main lacks one exact running keeper"),
        ("exited_keeper", "keeper lifecycle is not removable"),
        ("running_main", "main is not an exact inert occurrence"),
    ),
)
def test_allocation_only_invalidation_rejects_unsafe_lifecycle_before_mutation(
    publisher_case,
    resource_context,
    monkeypatch,
    invalid_state,
    error_match,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation = _append_allocation_invalidation(
        gate,
        reservation,
        execution_policy,
    )
    prepared = _FakeExecutionAdapter._prepared_for_allocation(allocation)
    _install_cleanup_resources(runner, allocation, prepared)
    _install_allocation_cleanup_proof_adapters(
        monkeypatch,
        allocation,
        prepared,
    )
    keeper_id = prepared.volume_keeper_evidence.container_id
    main_id = prepared.inert_container_evidence.container_id
    if invalid_state == "created_keeper_with_main":
        runner.containers[keeper_id]["State"]["Status"] = "created"
        runner.running_container_ids.clear()
    elif invalid_state == "exited_keeper":
        runner.containers.pop(main_id)
        runner.containers[keeper_id]["State"]["Status"] = "exited"
        runner.running_container_ids.clear()
    else:
        runner.containers[main_id]["State"]["Status"] = "running"

    with pytest.raises(RunActionDockerCleanupError, match=error_match):
        _finalization_authority(
            publisher_case,
            gate,
            resource_manager,
            runner,
        ).finalize_terminal(reservation.intent.operation_id)

    assert not _cleanup_commands(runner)


@pytest.mark.parametrize("removal_accepted", (True, False))
def test_allocation_volume_removal_progress_depends_only_on_fresh_inventory(
    publisher_case,
    resource_context,
    monkeypatch,
    removal_accepted,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation = _append_allocation_invalidation(
        gate,
        reservation,
        execution_policy,
    )
    prepared = _FakeExecutionAdapter._prepared_for_allocation(allocation)
    _install_cleanup_resources(runner, allocation, prepared)
    runner.containers.clear()
    runner.running_container_ids.clear()
    runner.cleanup_returncode = 1
    runner.cleanup_stderr = b"ambiguous Docker removal response"
    runner.cleanup_remove_target = removal_accepted
    _install_allocation_cleanup_proof_adapters(
        monkeypatch,
        allocation,
        prepared,
    )
    authority = _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    )

    if removal_accepted:
        authority.finalize_terminal(reservation.intent.operation_id)
        assert not runner.volumes
    else:
        with pytest.raises(
            RunActionDockerCleanupError,
            match="did not produce the exact next physical suffix",
        ):
            authority.finalize_terminal(reservation.intent.operation_id)
        assert runner.volumes

    assert _cleanup_commands(runner) == (
        ("volume", "rm", allocation.runtime_volume_authority.volume_name),
    )


@pytest.mark.parametrize("residue", ("created_keeper", "inert_main"))
def test_allocation_container_removals_ignore_ambiguous_responses_after_progress(
    publisher_case,
    resource_context,
    monkeypatch,
    residue,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation = _append_allocation_invalidation(
        gate,
        reservation,
        execution_policy,
    )
    prepared = _FakeExecutionAdapter._prepared_for_allocation(allocation)
    _install_cleanup_resources(runner, allocation, prepared)
    _install_allocation_cleanup_proof_adapters(
        monkeypatch,
        allocation,
        prepared,
    )
    main_id = prepared.inert_container_evidence.container_id
    keeper_id = prepared.volume_keeper_evidence.container_id
    if residue == "created_keeper":
        runner.containers.pop(main_id)
        runner.containers[keeper_id]["State"]["Status"] = "created"
        runner.running_container_ids.clear()
    runner.cleanup_returncode = 1
    runner.cleanup_stderr = b"ambiguous Docker removal response"

    _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    ).finalize_terminal(reservation.intent.operation_id)

    expected_prefix = (
        (("container", "rm", keeper_id),)
        if residue == "created_keeper"
        else (
            ("container", "rm", main_id),
            ("container", "rm", "--force", keeper_id),
        )
    )
    assert _cleanup_commands(runner) == (
        *expected_prefix,
        ("volume", "rm", allocation.runtime_volume_authority.volume_name),
    )
    assert not runner.containers
    assert not runner.volumes


def test_allocation_created_keeper_retries_only_after_unchanged_suffix(
    publisher_case,
    resource_context,
    monkeypatch,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation = _append_allocation_invalidation(
        gate,
        reservation,
        execution_policy,
    )
    prepared = _FakeExecutionAdapter._prepared_for_allocation(allocation)
    _install_cleanup_resources(runner, allocation, prepared)
    _install_allocation_cleanup_proof_adapters(
        monkeypatch,
        allocation,
        prepared,
    )
    main_id = prepared.inert_container_evidence.container_id
    keeper_id = prepared.volume_keeper_evidence.container_id
    runner.containers.pop(main_id)
    runner.containers[keeper_id]["State"]["Status"] = "created"
    runner.running_container_ids.clear()
    runner.cleanup_returncode = 1
    runner.cleanup_stderr = b"ambiguous Docker removal response"
    runner.cleanup_remove_target = False
    authority = _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    )

    with pytest.raises(
        RunActionDockerCleanupError,
        match="did not produce the exact next physical suffix",
    ):
        authority.finalize_terminal(reservation.intent.operation_id)

    runner.cleanup_returncode = 0
    runner.cleanup_stderr = b""
    runner.cleanup_remove_target = True
    authority.finalize_terminal(reservation.intent.operation_id)

    assert _cleanup_commands(runner) == (
        ("container", "rm", keeper_id),
        ("container", "rm", keeper_id),
        ("volume", "rm", allocation.runtime_volume_authority.volume_name),
    )
    assert not runner.containers
    assert not runner.volumes


def test_allocation_created_keeper_cleanup_is_serialized_and_mutates_once(
    publisher_case,
    resource_context,
    monkeypatch,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation = _append_allocation_invalidation(
        gate,
        reservation,
        execution_policy,
    )
    prepared = _FakeExecutionAdapter._prepared_for_allocation(allocation)
    _install_cleanup_resources(runner, allocation, prepared)
    _install_allocation_cleanup_proof_adapters(
        monkeypatch,
        allocation,
        prepared,
    )
    main_id = prepared.inert_container_evidence.container_id
    keeper_id = prepared.volume_keeper_evidence.container_id
    runner.containers.pop(main_id)
    runner.containers[keeper_id]["State"]["Status"] = "created"
    runner.running_container_ids.clear()
    authority = _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    )
    first_mutation = Event()
    release_first = Event()
    second_completed = Event()
    mutation_count = 0

    def hold_first_mutation():
        nonlocal mutation_count
        mutation_count += 1
        if mutation_count == 1:
            first_mutation.set()
            assert release_first.wait(runner.settings.command_timeout_seconds)

    def finalize_and_mark():
        authority.finalize_terminal(reservation.intent.operation_id)
        second_completed.set()

    runner.cleanup_post_mutation = hold_first_mutation
    with ThreadPoolExecutor(max_workers=2) as execution:
        first = execution.submit(
            authority.finalize_terminal,
            reservation.intent.operation_id,
        )
        assert first_mutation.wait(runner.settings.command_timeout_seconds)
        second = execution.submit(finalize_and_mark)
        second_was_serialized = not second_completed.wait(
            runner.settings.run_action_barrier_poll_interval_seconds
        )
        release_first.set()
        first.result(timeout=runner.settings.command_timeout_seconds)
        second.result(timeout=runner.settings.command_timeout_seconds)

    assert second_was_serialized
    assert _cleanup_commands(runner) == (
        ("container", "rm", keeper_id),
        ("volume", "rm", allocation.runtime_volume_authority.volume_name),
    )


def test_prepared_invalidation_reaps_proved_resources(
    publisher_case,
    resource_context,
    monkeypatch,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared = _append_prepared_cleanup_prefix(
        gate,
        reservation,
        execution_policy,
    )
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        session.invalidate_frontier()
    _install_cleanup_resources(runner, allocation, prepared)
    _install_prepared_cleanup_proof_adapters(monkeypatch, prepared)

    _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    ).finalize_terminal(reservation.intent.operation_id)

    assert not runner.containers
    assert not runner.volumes
    assert _cleanup_commands(runner) == (
        ("container", "rm", prepared.inert_container_evidence.container_id),
        (
            "container",
            "rm",
            "--force",
            prepared.volume_keeper_evidence.container_id,
        ),
        ("volume", "rm", allocation.runtime_volume_authority.volume_name),
    )


def test_cancelled_intent_is_cleanup_no_op(
    publisher_case,
    resource_context,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, _execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        session.cancel()
    authority = _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    )
    request_count = len(runner.requests)

    authority.finalize_terminal(reservation.intent.operation_id)
    authority.require_terminal_absence(reservation.intent.operation_id)

    assert len(runner.requests) == request_count
    assert not _cleanup_commands(runner)


def test_same_operation_cleanup_is_serialized_and_mutates_once(
    publisher_case,
    resource_context,
    monkeypatch,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared, events = _append_cleanup_terminal(
        gate,
        reservation,
        execution_policy,
        "result_accepted",
    )
    terminal = cleanup_module._terminal_cleanup_evidence(events).terminal_observation
    _install_cleanup_resources(
        runner,
        allocation,
        prepared,
        main_role_proof=terminal.complete_inspection_digest,
    )
    _install_cleanup_proof_adapters(monkeypatch, prepared, events)
    authority = _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    )
    first_mutation = Event()
    release_first = Event()
    second_completed = Event()
    mutation_count = 0

    def hold_first_mutation():
        nonlocal mutation_count
        mutation_count += 1
        if mutation_count == 1:
            first_mutation.set()
            assert release_first.wait(runner.settings.command_timeout_seconds)

    def finalize_and_mark():
        authority.finalize_terminal(reservation.intent.operation_id)
        second_completed.set()

    runner.cleanup_post_mutation = hold_first_mutation
    with ThreadPoolExecutor(max_workers=2) as execution:
        first = execution.submit(
            authority.finalize_terminal,
            reservation.intent.operation_id,
        )
        assert first_mutation.wait(runner.settings.command_timeout_seconds)
        second = execution.submit(finalize_and_mark)
        second_was_serialized = not second_completed.wait(
            runner.settings.run_action_barrier_poll_interval_seconds
        )
        release_first.set()
        first.result(timeout=runner.settings.command_timeout_seconds)
        second.result(timeout=runner.settings.command_timeout_seconds)

    assert second_was_serialized
    assert _cleanup_commands(runner) == (
        ("container", "rm", prepared.inert_container_evidence.container_id),
        (
            "container",
            "rm",
            "--force",
            prepared.volume_keeper_evidence.container_id,
        ),
        ("volume", "rm", allocation.runtime_volume_authority.volume_name),
    )


def test_cleanup_manager_has_no_standalone_mutation_or_eligibility_surface(
    resource_context,
):
    _resource_manager, runner, _allocation, _claim = resource_context
    manager = DockerRunActionCleanupManager(runner.runtime)

    assert not hasattr(manager, "remove")
    assert not hasattr(manager, "advance")
    assert not hasattr(cleanup_module, "advance_run_action_resource_cleanup_once")
    assert not hasattr(cleanup_module, "require_run_action_resources_stably_absent")


def test_result_acceptance_is_exact_terminal_cleanup_eligibility(publisher_case):
    _frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_accepted(gate, reservation)
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)

    evidence = cleanup_module._terminal_cleanup_evidence(events)

    assert events[-1].event_kind is RunActionExecutionEventKind.RESULT_ACCEPTED
    assert evidence.topology is RunActionControlDirectoryTopology.RELEASED
    assert (
        evidence.terminal_observation == events[5].result_receipt.terminal_observation
    )
    assert not evidence.main_must_be_absent


def test_provider_termination_is_exact_terminal_cleanup_eligibility(publisher_case):
    _frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_provider_terminated(gate, reservation)
    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)

    evidence = cleanup_module._terminal_cleanup_evidence(events)

    assert events[-1].event_kind is RunActionExecutionEventKind.PROVIDER_TERMINATED
    assert evidence.topology is RunActionControlDirectoryTopology.RELEASED
    assert (
        evidence.terminal_observation
        == events[-1].provider_termination_receipt.terminal_observation
    )


def test_pre_release_terminal_cleanup_is_empty_and_main_optional(
    publisher_case,
    resource_context,
    monkeypatch,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared, events = _append_cleanup_terminal(
        gate,
        reservation,
        execution_policy,
        "pre_release_terminal",
    )
    evidence = cleanup_module._terminal_cleanup_evidence(events)
    terminal = evidence.terminal_observation
    assert type(terminal) is RunActionPreReleaseMainTerminalObservation
    assert evidence.topology is RunActionControlDirectoryTopology.EMPTY
    assert evidence.workload_release_adoption is None
    assert evidence.timeout_directive_publication is None
    assert not evidence.main_must_be_absent

    _install_cleanup_resources(
        runner,
        allocation,
        prepared,
        main_role_proof=(
            terminal.terminal_container_observation.complete_inspection_digest
        ),
    )
    _install_cleanup_proof_adapters(monkeypatch, prepared, events)
    runner.containers.pop(prepared.inert_container_evidence.container_id)

    _finalization_authority(
        publisher_case,
        gate,
        resource_manager,
        runner,
    ).finalize_terminal(reservation.intent.operation_id)

    assert _cleanup_commands(runner) == (
        (
            "container",
            "rm",
            "--force",
            prepared.volume_keeper_evidence.container_id,
        ),
        ("volume", "rm", allocation.runtime_volume_authority.volume_name),
    )


def test_event5_tail_cannot_authorize_cleanup(
    publisher_case,
    resource_context,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared = _append_prepared_cleanup_prefix(
        gate,
        reservation,
        execution_policy,
    )
    with gate._action_store._recovery_session(
        reservation,
        _authority=_RUN_ACTION_RECOVERY_AUTHORITY,
    ) as session:
        spawn = session.commit_spawn(
            security_observation_id=reservation.frontier.security_observation_id,
            boundary_identity=reservation.intent.boundary_identity,
        )
        session.commit_activation(_activation_revalidation_receipt(prepared, spawn))
    _install_cleanup_resources(runner, allocation, prepared)

    with pytest.raises(
        RunActionDockerCleanupError,
        match="lacks a complete durable execution prefix",
    ):
        _finalization_authority(
            publisher_case,
            gate,
            resource_manager,
            runner,
        ).finalize_terminal(reservation.intent.operation_id)

    assert not _cleanup_commands(runner)


def test_pre_release_loss_still_rejects_main_reappearance(
    publisher_case,
    resource_context,
):
    resource_manager, runner, _fixture_allocation, _claim = resource_context
    gate, reservation, execution_policy = _reserve_cleanup_operation(
        publisher_case,
        resource_manager,
    )
    allocation, prepared, _events = _append_cleanup_terminal(
        gate,
        reservation,
        execution_policy,
        "pre_release_loss",
    )
    _install_cleanup_resources(runner, allocation, prepared)

    with pytest.raises(
        RunActionDockerCleanupError,
        match="pre-release main reappeared",
    ):
        _finalization_authority(
            publisher_case,
            gate,
            resource_manager,
            runner,
        ).finalize_terminal(reservation.intent.operation_id)

    assert not _cleanup_commands(runner)


def test_received_result_is_never_cleanup_eligibility(
    publisher_case,
    resource_context,
):
    _frontier, gate, reservation, _payload = _reserved_case(publisher_case)
    _append_result_received(gate, reservation)
    resource_manager, runner, _allocation, _claim = resource_context

    with pytest.raises(
        RunActionDockerCleanupError,
        match="not durably terminal and eligible",
    ):
        _finalization_authority(
            publisher_case,
            gate,
            resource_manager,
            runner,
        ).finalize_terminal(reservation.intent.operation_id)

    assert not tuple(
        request
        for request in runner.requests
        if request.argv[5:7]
        in {
            ("container", "rm"),
            ("volume", "rm"),
        }
    )


def test_volume_occurrence_digest_rejects_name_reuse(docker_settings):
    claim, authority, first_raw, _volume, _command, _helper, _init = _context(
        docker_settings
    )
    replacement_raw = deepcopy(first_raw)
    replacement_raw["CreatedAt"] = "2026-07-25T00:00:01Z"

    first = observe_runtime_volume(
        first_raw,
        claim,
        authority,
        docker_settings,
    )
    replacement = observe_runtime_volume(
        replacement_raw,
        claim,
        authority,
        docker_settings,
    )

    assert first.volume_name == replacement.volume_name
    assert first.volume_occurrence_digest != replacement.volume_occurrence_digest


def test_cleanup_manager_cannot_cross_process_boundary(
    resource_context,
    monkeypatch,
):
    _resource_manager, runner, _allocation, _claim = resource_context
    manager = DockerRunActionCleanupManager(runner.runtime)
    owner_process_id = cleanup_module.os.getpid()
    monkeypatch.setattr(cleanup_module.os, "getpid", lambda: owner_process_id + 1)

    with pytest.raises(
        RunActionDockerCleanupError,
        match="unissued or foreign",
    ):
        manager.runtime_settings


def test_finalization_authority_is_sealed_and_process_bound(
    publisher_case,
    resource_context,
    monkeypatch,
):
    _resource_manager, runner, _allocation, _claim = resource_context
    _frontier, gate, _reservation, _payload = _reserved_case(publisher_case)
    authority = _finalization_authority(
        publisher_case,
        gate,
        _resource_manager,
        runner,
    )
    binding = authority._require_current()

    with pytest.raises(
        RunActionResourceFinalizationError,
        match="lacks exact issuance",
    ):
        RunActionResourceFinalizationAuthority(
            action_store=binding[0],
            launch_settings=binding[1],
            driver=binding[2],
            _authority=object(),
        )

    owner_process_id = authority._owner_process_id
    monkeypatch.setattr(
        finalization_module.os,
        "getpid",
        lambda: owner_process_id + 1,
    )
    with pytest.raises(
        RunActionResourceFinalizationError,
        match="foreign or changed",
    ):
        authority.require_terminal_absence("operation_0123456789abcdef")
