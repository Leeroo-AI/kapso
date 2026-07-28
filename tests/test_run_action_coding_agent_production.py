from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import DockerImageAuthority
from kapso.cross_run.launch.run_action_coding_agent_production import (
    build_coding_agent_boundary_identity,
    build_coding_agent_execution_policy,
    build_coding_agent_interpretation_policy,
    ProductionCodingAgentPolicyError,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialMode,
)
from kapso.cross_run.settings import CodingAgentSettings, CrossRunSettings

_CONFIG_PATH = "src/kapso/config.yaml"


def _settings():
    return CrossRunSettings.from_dict(load_config(_CONFIG_PATH)["cross_run"])


def _agent(timeout_seconds=300):
    return CodingAgentSettings(
        cli="codex",
        model="gpt-5.6-sol",
        timeout_seconds=timeout_seconds,
        effort="xhigh",
        allowed_tools=("Read",),
    )


def _image():
    return DockerImageAuthority.mint(
        image_reference=(
            "registry.example/kapso/coding-agent@"
            + tree_or_blob_digest(b"production image")
        ),
        image_config_digest=tree_or_blob_digest(b"production image config"),
        operating_system="linux",
        architecture="amd64",
        architecture_variant=None,
    )


def test_production_builder_joins_configured_interpretation_docker_and_boundary():
    settings = _settings()
    interpretation = build_coding_agent_interpretation_policy(
        settings=settings,
        agent=_agent(),
        principal_id="kapso.ideation.generator",
        role="candidate_generator",
        workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        web_search_enabled=False,
        provider_network_enabled=True,
    )
    execution, command = build_coding_agent_execution_policy(
        settings=settings,
        image_authority=_image(),
        interpretation_policy=interpretation,
        credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
        egress_broker_socket_source_path="/private/egress/broker.sock",
    )
    boundary = build_coding_agent_boundary_identity(execution, interpretation)

    assert interpretation.egress_connect_authorities == (
        settings.launch.coding_agent_egress_connect_authorities
    )
    assert execution.command_template_id == command.command_template_id
    assert execution.docker_resource_limits.memory_size_bytes == (
        settings.launch.coding_agent_action_memory_size_bytes
    )
    assert execution.credential_policy.maximum_lease_seconds == (
        settings.launch.coding_agent_action_credential_lease_seconds
    )
    assert boundary.execution_lifecycle_identity.execution_policy_id == (
        execution.docker_execution_policy_id
    )
    assert boundary.result_interpreter_identity.interpretation_policy_id == (
        interpretation.interpretation_policy_id
    )


def test_production_builder_rejects_uncontained_timeout_and_credential_splice():
    settings = _settings()
    with pytest.raises(
        ProductionCodingAgentPolicyError,
        match="exceed configured authority",
    ):
        build_coding_agent_interpretation_policy(
            settings=settings,
            agent=_agent(
                settings.launch.coding_agent_action_execution_timeout_seconds + 1
            ),
            principal_id="kapso.ideation.generator",
            role="candidate_generator",
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            web_search_enabled=False,
            provider_network_enabled=True,
        )

    interpretation = build_coding_agent_interpretation_policy(
        settings=settings,
        agent=_agent(),
        principal_id="kapso.ideation.generator",
        role="candidate_generator",
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        web_search_enabled=False,
        provider_network_enabled=False,
    )
    with pytest.raises(
        ProductionCodingAgentPolicyError,
        match="requires its supervisor-delivered credential",
    ):
        build_coding_agent_execution_policy(
            settings=settings,
            image_authority=_image(),
            interpretation_policy=build_coding_agent_interpretation_policy(
                settings=settings,
                agent=_agent(),
                principal_id="kapso.ideation.generator",
                role="candidate_generator",
                workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
                web_search_enabled=False,
                provider_network_enabled=True,
            ),
            credential_mode=RunActionCredentialMode.NONE,
            egress_broker_socket_source_path="/private/egress/broker.sock",
        )

    with pytest.raises(ValueError, match="network-disabled"):
        replace(interpretation, web_search_enabled=True)
