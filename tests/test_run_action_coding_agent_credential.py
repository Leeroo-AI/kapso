from dataclasses import replace
from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.docker.runtime import DockerImageAuthority
from kapso.cross_run.launch.run_action_coding_agent_credential import (
    NativeCodexCredentialBroker,
    NativeCodexCredentialBrokerError,
)
from kapso.cross_run.launch.run_action_coding_agent_production import (
    build_coding_agent_execution_policy,
    build_coding_agent_interpretation_policy,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
)
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialLeaseRequest,
    RunActionCredentialMode,
    RunActionPreparedDeliverySlot,
    RunActionPreparedExecution,
)
from kapso.cross_run.settings import CodingAgentSettings, CrossRunSettings

_CONFIG_PATH = "src/kapso/config.yaml"


def _settings(tmp_path: Path):
    configured = CrossRunSettings.from_dict(load_config(_CONFIG_PATH)["cross_run"])
    credential = tmp_path / "auth.json"
    credential.write_bytes(b'{"credential":"test-secret"}')
    credential.chmod(0o600)
    launch = replace(
        configured.launch,
        coding_agent_codex_auth_source_path=credential.as_posix(),
        coding_agent_credential_lease_state_path="credential-leases.json",
    )
    return replace(configured, launch=launch)


def _policy(settings):
    interpretation = build_coding_agent_interpretation_policy(
        settings=settings,
        agent=CodingAgentSettings(
            cli="codex",
            model="gpt-5.6-sol",
            timeout_seconds=300,
            effort="xhigh",
            allowed_tools=("Read",),
        ),
        principal_id="kapso.ideation.generator",
        role="candidate_generator",
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        web_search_enabled=False,
        provider_network_enabled=True,
    )
    image = DockerImageAuthority.mint(
        image_reference=(
            "registry.example/kapso/coding-agent@" + tree_or_blob_digest(b"image")
        ),
        image_config_digest=tree_or_blob_digest(b"config"),
        operating_system="linux",
        architecture="amd64",
        architecture_variant=None,
    )
    execution, _command = build_coding_agent_execution_policy(
        settings=settings,
        image_authority=image,
        interpretation_policy=interpretation,
        credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    )
    return execution.credential_policy


def _request(policy):
    def fixture_id(namespace, label):
        return content_id(namespace, {"fixture": label})

    return RunActionCredentialLeaseRequest.mint(
        credential_policy=policy,
        reservation_id=fixture_id(RunActionReservation.CONTENT_NAMESPACE, "reserve"),
        prepared_execution_id=fixture_id(
            RunActionPreparedExecution.CONTENT_NAMESPACE,
            "prepared",
        ),
        spawn_commit_id=fixture_id(RunActionSpawnCommit.CONTENT_NAMESPACE, "spawn"),
        credential_delivery_slot_id=fixture_id(
            RunActionPreparedDeliverySlot.CONTENT_NAMESPACE,
            "delivery",
        ),
    )


def test_native_codex_broker_replays_exact_expiry_without_persisting_secret(
    tmp_path: Path,
):
    settings = _settings(tmp_path)
    request = _request(_policy(settings))
    state_root = (tmp_path / "state").resolve()
    broker = NativeCodexCredentialBroker(
        settings=settings.launch,
        state_root=state_root,
    )

    first = broker.issue_or_replay_exact(request)
    second = broker.issue_or_replay_exact(request)
    observed = broker.observe_exact(request)

    assert first.valid_until_realtime_nanoseconds == (
        second.valid_until_realtime_nanoseconds
    )
    assert observed.valid_until_realtime_nanoseconds == (
        first.valid_until_realtime_nanoseconds
    )
    state_payload = (state_root / "credential-leases.json").read_bytes()
    assert b"test-secret" not in state_payload
    assert tree_or_blob_digest(b'{"credential":"test-secret"}').encode() in (
        state_payload
    )


def test_native_codex_broker_fails_if_credential_changes_during_lease(tmp_path: Path):
    settings = _settings(tmp_path)
    request = _request(_policy(settings))
    broker = NativeCodexCredentialBroker(
        settings=settings.launch,
        state_root=(tmp_path / "state").resolve(),
    )
    broker.issue_or_replay_exact(request)
    credential = Path(settings.launch.coding_agent_codex_auth_source_path)
    credential.write_bytes(b'{"credential":"changed"}')
    credential.chmod(0o600)

    with pytest.raises(
        NativeCodexCredentialBrokerError,
        match="changed during an exact lease",
    ):
        broker.observe_exact(request)
