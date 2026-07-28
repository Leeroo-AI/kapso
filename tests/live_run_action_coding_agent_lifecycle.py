"""Explicit native coding-agent image validation through the durable lifecycle.

Run directly after building the offline image:

    docker build -f docker/coding-agent/Dockerfile --target offline \
      -t kapso/coding-agent-offline:m9 .
    pytest -q tests/live_run_action_coding_agent_lifecycle.py -s
"""

from __future__ import annotations

import json
import subprocess
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from expert_live_docker_support import (
    require_setup_docker_success,
    run_setup_docker,
)
from kapso.core.config import load_config
from kapso.cross_run.docker.runtime import (
    DockerImageAuthority,
    PinnedDockerRuntime,
)
from kapso.cross_run.launch.run_action_coding_agent_consumer import (
    NATIVE_CODING_AGENT_CONSUMER_ID,
    NATIVE_CODING_AGENT_CONSUMER_VERSION,
)
from kapso.cross_run.launch.run_action_coding_agent_interpreter import (
    CodingAgentRunActionResultInterpreter,
    coding_agent_result_interpreter_identity,
)
from kapso.cross_run.launch.run_action_coding_agent_supervisor import (
    coding_agent_supervisor_command,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_docker_adapter import (
    DockerRunActionExecutionAdapter,
)
from kapso.cross_run.launch.run_action_docker_cleanup import (
    DockerRunActionCleanupManager,
    issue_docker_run_action_resource_finalization_authority,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_store import (
    RunActionExecutionEventKind,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    RunActionProviderTerminationReason,
)
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialMode,
    RunActionStaticEnvironmentVariable,
)
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from kapso.cross_run.settings import CrossRunSettings
from live_run_action_docker_projection import (
    _ORIGINAL_SUBPROCESS_RUN,
    _production_recovery_coordinator,
)
from test_run_action_coding_agent_contracts import (
    interpretation_policy,
    run_action_request,
)
from test_run_action_docker_projection import _policy
from test_launch_resolver import resolver_case
from test_run_action_supervisor_contracts import (
    _remint_contract,
    _remint_policy,
    _remint_sandbox,
)
from test_run_frontier_action_gate import _action_case, _boundary_identity
from test_run_state_publisher import publisher_case

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_OFFLINE_IMAGE = "kapso/coding-agent-offline:m9"


def _remove_registry_container(settings, docker_config_root, container_id):
    removal = run_setup_docker(
        settings,
        docker_config_root,
        ("container", "rm", "--force", container_id),
    )
    require_setup_docker_success(removal, "local registry cleanup")


def _remove_registry_image(settings, docker_config_root, image_reference):
    removal = run_setup_docker(
        settings,
        docker_config_root,
        ("image", "rm", image_reference),
    )
    require_setup_docker_success(removal, "local registry image cleanup")


def _publish_local_image(cleanup, settings, docker_config_root):
    registry = run_setup_docker(
        settings,
        docker_config_root,
        (
            "container",
            "run",
            "--detach",
            "--publish",
            "127.0.0.1::5000",
            "registry:2",
        ),
    )
    require_setup_docker_success(registry, "local registry start")
    registry_id = registry.stdout.strip().decode("ascii")
    cleanup.callback(
        _remove_registry_container,
        settings,
        docker_config_root,
        registry_id,
    )
    port = run_setup_docker(
        settings,
        docker_config_root,
        ("container", "port", registry_id, "5000/tcp"),
    )
    require_setup_docker_success(port, "local registry port")
    address = port.stdout.strip().decode("ascii")
    if not address.startswith("127.0.0.1:"):
        raise AssertionError("local registry received a non-loopback address")
    repository = f"{address}/kapso/coding-agent-offline"
    tagged_reference = f"{repository}:m9"
    tag = run_setup_docker(
        settings,
        docker_config_root,
        ("image", "tag", _OFFLINE_IMAGE, tagged_reference),
    )
    require_setup_docker_success(tag, "offline image tag")
    cleanup.callback(
        _remove_registry_image,
        settings,
        docker_config_root,
        tagged_reference,
    )
    push = run_setup_docker(
        settings,
        docker_config_root,
        ("image", "push", tagged_reference),
    )
    require_setup_docker_success(push, "offline image push")
    inspection = run_setup_docker(
        settings,
        docker_config_root,
        ("image", "inspect", tagged_reference),
    )
    require_setup_docker_success(inspection, "offline image inspection")
    images = json.loads(inspection.stdout)
    if len(images) != 1:
        raise AssertionError("offline image inspection is not singular")
    image = images[0]
    matching_digests = tuple(
        reference
        for reference in image["RepoDigests"]
        if reference.startswith(f"{repository}@sha256:")
    )
    if len(matching_digests) != 1:
        raise AssertionError("offline image lacks one local registry digest")
    return DockerImageAuthority.mint(
        image_reference=matching_digests[0],
        image_config_digest=image["Id"],
        operating_system="linux",
        architecture="amd64",
        architecture_variant=None,
    )


@pytest.mark.parametrize(
    ("workspace_access", "simulate_lost_installation"),
    (
        (RunFrontierWorkspaceAccess.READ_ONLY, False),
        (RunFrontierWorkspaceAccess.EDIT_WORKSPACE, False),
        (RunFrontierWorkspaceAccess.READ_ONLY, True),
    ),
)
def test_native_offline_image_completes_the_eight_event_lifecycle(
    tmp_path: Path,
    publisher_case,
    monkeypatch,
    workspace_access,
    simulate_lost_installation,
):
    monkeypatch.setattr(subprocess, "run", _ORIGINAL_SUBPROCESS_RUN)
    cross_run_settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    docker_settings = cross_run_settings.docker
    docker_config_root = tmp_path / "docker-config"
    docker_config_root.mkdir(mode=0o700)
    (docker_config_root / "config.json").write_bytes(b'{"auths":{}}\n')
    (docker_config_root / "config.json").chmod(0o400)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)

    with ExitStack() as cleanup:
        image_authority = _publish_local_image(
            cleanup,
            docker_settings,
            docker_config_root,
        )
        runtime = PinnedDockerRuntime.create(
            trusted_root=runtime_root.resolve(),
            settings=docker_settings,
        )
        resource_manager = DockerRunActionResourceManager(runtime)
        cleanup_manager = DockerRunActionCleanupManager(runtime)
        boundary = (
            RunSafetyBoundary.IDEATION
            if workspace_access is RunFrontierWorkspaceAccess.READ_ONLY
            else RunSafetyBoundary.IMPLEMENTATION
        )
        (
            _action_publisher,
            action_frontier,
            _security_authority,
            action_gate,
        ) = _action_case(
            publisher_case,
            boundary=boundary,
            resource_finalization_authority_factory=(
                lambda publisher: issue_docker_run_action_resource_finalization_authority(
                    action_store=publisher._action_store,
                    launch_settings=publisher._settings,
                    resource_manager=resource_manager,
                    cleanup_manager=cleanup_manager,
                )
            ),
        )
        workspace_descriptor, _workspace_identity = publisher_case[
            "active"
        ]._open_execution_workspace(cleanup)
        expected_workspace_commit = action_frontier.checkpoint.safety_state.derivative_frontier.evidence.branch_heads[
            publisher_case["settings"].workspace_git_branch
        ]
        source_frontier = inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=publisher_case["settings"],
            expected_commit_sha=expected_workspace_commit,
        )
        coding_policy = interpretation_policy(
            consumer_id=NATIVE_CODING_AGENT_CONSUMER_ID,
            consumer_version=NATIVE_CODING_AGENT_CONSUMER_VERSION,
            workspace_access=workspace_access,
            web_search_enabled=False,
        )
        request = run_action_request(
            coding_policy,
            predecessor_digest=(
                source_frontier.source_tree_digest
                if workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE
                else None
            ),
        )
        if workspace_access is RunFrontierWorkspaceAccess.EDIT_WORKSPACE:
            request = replace(request, prompt=f"OFFLINE_EDIT\n{request.prompt}")
        supervisor_command = coding_agent_supervisor_command(coding_policy)
        command = DockerRunActionCommand.build(
            entrypoint=supervisor_command[0],
            arguments=supervisor_command[1:],
        )
        base_policy = _policy(
            docker_settings,
            workspace_access=workspace_access,
            credential_mode=RunActionCredentialMode.NONE,
            command_template_id=command.command_template_id,
        )
        execution_policy = _remint_policy(
            base_policy,
            image_authority=image_authority,
            static_environment=(
                RunActionStaticEnvironmentVariable(key="LANG", value="C"),
                RunActionStaticEnvironmentVariable(
                    key="PATH",
                    value="/usr/local/bin:/usr/bin:/bin",
                ),
            ),
            sandbox_spec=_remint_sandbox(
                base_policy.sandbox_spec,
                capability_additions=("KILL", "SETGID", "SETPCAP", "SETUID"),
                supplementary_group_ids=(coding_policy.provider_group_id,),
                no_new_privileges=False,
                security_option_ids=(
                    "apparmor:docker-default",
                    "seccomp:builtin",
                ),
            ),
        )
        base_boundary = _boundary_identity(
            RunFrontierActionKind.CODING_AGENT,
            workspace_access,
        )
        boundary_identity = RunActionBoundaryIdentity.mint(
            kind=RunFrontierActionKind.CODING_AGENT,
            execution_lifecycle_identity=_remint_contract(
                base_boundary.execution_lifecycle_identity,
                execution_policy_id=execution_policy.docker_execution_policy_id,
            ),
            result_interpreter_identity=coding_agent_result_interpreter_identity(
                coding_policy
            ),
        )
        reservation = action_gate.reserve(
            action_frontier,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=boundary,
            operation_id=request.operation_id,
            request_payload=request.to_json_bytes(),
            workspace_access=workspace_access,
            boundary_identity=boundary_identity,
        )
        adapter = DockerRunActionExecutionAdapter(
            execution_lifecycle_identity=(
                boundary_identity.execution_lifecycle_identity
            ),
            execution_policy=execution_policy,
            command=command,
            runtime=runtime,
            launch_settings=cross_run_settings.launch,
        )
        interpreter = CodingAgentRunActionResultInterpreter(
            result_interpreter_identity=boundary_identity.result_interpreter_identity,
            interpretation_policy=coding_policy,
        )
        coordinator = _production_recovery_coordinator(
            action_gate,
            boundary_identity,
            adapter,
            interpreter,
        )

        started = coordinator.recover(action_frontier)
        assert not started.is_complete
        events = action_gate._action_store.inspect().events_for(
            reservation.intent.operation_id
        )
        spawn = tuple(
            event
            for event in events
            if event.event_kind is RunActionExecutionEventKind.SPAWN_COMMITTED
        )
        assert len(spawn) == 1
        active_inventory = resource_manager.observe(events[1].preparation_allocation)
        main_inspection = resource_manager.inspect_main(active_inventory)
        bind_sources = tuple(
            binding.rsplit(":", 2)[0]
            for binding in main_inspection["HostConfig"]["Binds"] or ()
        )
        mount_sources = tuple(mount["Source"] for mount in main_inspection["Mounts"])
        assert docker_settings.runtime_socket_path not in bind_sources + mount_sources
        assert (
            docker_settings.runtime_mutation_lock_path
            not in bind_sources + mount_sources
        )
        if simulate_lost_installation:
            main_stop = runtime.run_control(
                ("container", "stop", spawn[0].spawn_commit.provider_execution_id)
            )
            assert main_stop.returncode == 0
            keeper_stop = runtime.run_control(
                (
                    "container",
                    "stop",
                    events[2].prepared_execution.volume_keeper_evidence.container_id,
                )
            )
            assert keeper_stop.returncode == 0

            lost = coordinator.recover(action_frontier)

            lost_events = action_gate._action_store.inspect().events_for(
                reservation.intent.operation_id
            )
            assert lost.is_complete
            assert tuple(event.event_kind for event in lost_events) == (
                RunActionExecutionEventKind.INTENT_RESERVED,
                RunActionExecutionEventKind.PREPARATION_ALLOCATED,
                RunActionExecutionEventKind.EXECUTION_PREPARED,
                RunActionExecutionEventKind.SPAWN_COMMITTED,
                RunActionExecutionEventKind.ACTIVATION_COMMITTED,
                RunActionExecutionEventKind.PROVIDER_TERMINATED,
            )
            assert (
                lost_events[-1].provider_termination_receipt.reason
                is RunActionProviderTerminationReason.RUNTIME_INSTALLATION_LOST
            )
            assert resource_manager.observe(
                lost_events[1].preparation_allocation
            ).is_absent
            return
        released = coordinator.recover(action_frontier)
        assert not released.is_complete
        wait = runtime.run_control(
            ("container", "wait", spawn[0].spawn_commit.provider_execution_id)
        )
        assert wait.stdout == b"0\n"
        terminal = coordinator.recover(action_frontier)

        assert terminal.is_complete
        terminal_events = action_gate._action_store.inspect().events_for(
            reservation.intent.operation_id
        )
        assert tuple(event.event_kind for event in terminal_events) == (
            RunActionExecutionEventKind.INTENT_RESERVED,
            RunActionExecutionEventKind.PREPARATION_ALLOCATED,
            RunActionExecutionEventKind.EXECUTION_PREPARED,
            RunActionExecutionEventKind.SPAWN_COMMITTED,
            RunActionExecutionEventKind.ACTIVATION_COMMITTED,
            RunActionExecutionEventKind.RESULT_RECEIVED,
            RunActionExecutionEventKind.RESULT_DECIDED,
            RunActionExecutionEventKind.RESULT_ACCEPTED,
        )
        assert terminal.recovered_operations[-1].accepted_result_payload == (
            b'{"answer":"offline boundary passed"}'
        )
        assert resource_manager.observe(
            terminal_events[1].preparation_allocation
        ).is_absent
