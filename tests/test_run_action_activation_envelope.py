"""Pre-delivery proof for the complete run-action event-5 wire shape."""

from __future__ import annotations

import pytest

import kapso.cross_run.launch.run_action_activation_envelope as envelope_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import CanonicalizationError, content_id
from kapso.cross_run.launch.run_action_activation_envelope import (
    RunActionActivationEnvelopeError,
    activation_execution_event_size_bound,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionContainerLabel,
    RunActionCredentialMode,
    RunActionSupervisorContractError,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_docker_projection import _policy
from test_run_action_supervisor_contracts import (
    _activation_revalidation_receipt,
    _claim,
    _prepared_execution,
    _remint_contract,
    _spawn_commit,
)

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_PREDECESSOR_EVENT_ID = content_id(
    RunActionExecutionEvent.CONTENT_NAMESPACE,
    {"fixture": "activation-envelope-predecessor"},
)


@pytest.fixture(scope="module")
def docker_settings():
    return CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker


def _activation_case(
    docker_settings,
    *,
    workspace_access: RunFrontierWorkspaceAccess,
    credential_mode: RunActionCredentialMode,
):
    policy = _policy(
        docker_settings,
        workspace_access=workspace_access,
        credential_mode=credential_mode,
    )
    prepared = _prepared_execution(claim=_claim(policy=policy))
    spawn = _spawn_commit(prepared)
    return prepared, spawn


def _activation_event_size(prepared, spawn) -> int:
    receipt = _activation_revalidation_receipt(prepared, spawn)
    event = RunActionExecutionEvent.mint(
        event_number=5,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
        event_kind=RunActionExecutionEventKind.ACTIVATION_COMMITTED,
        reservation=prepared.preparation_claim.reservation,
        preparation_allocation=None,
        prepared_execution=None,
        spawn_commit=None,
        activation_revalidation_receipt=receipt,
        provider_termination_receipt=None,
        result_receipt=None,
        result_decision=None,
        acceptance=None,
        workspace_after=None,
    )
    return len(event.to_json_bytes())


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
def test_activation_envelope_covers_every_optional_wire_shape(
    docker_settings,
    workspace_access,
    credential_mode,
):
    prepared, spawn = _activation_case(
        docker_settings,
        workspace_access=workspace_access,
        credential_mode=credential_mode,
    )

    first = activation_execution_event_size_bound(
        prepared_execution=prepared,
        spawn_commit=spawn,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
    )
    second = activation_execution_event_size_bound(
        prepared_execution=prepared,
        spawn_commit=spawn,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
    )

    assert first == second
    assert _activation_event_size(prepared, spawn) <= first


def test_activation_envelope_maxes_every_mutable_volume_integer(docker_settings):
    prepared, spawn = _activation_case(
        docker_settings,
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    )

    wire = envelope_module._reobserved_volume_wire(prepared.runtime_volume_evidence)

    assert {
        wire[field_name]
        for field_name in (
            "used_block_count",
            "used_size_bytes",
            "used_inode_count",
            "available_block_count",
            "available_size_bytes",
            "available_inode_count",
        )
    } == {RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER}
    assert activation_execution_event_size_bound(
        prepared_execution=prepared,
        spawn_commit=spawn,
        predecessor_event_id=_PREDECESSOR_EVENT_ID,
    ) >= _activation_event_size(prepared, spawn)


def test_activation_envelope_rejects_spawn_or_predecessor_substitution(
    docker_settings,
):
    prepared, spawn = _activation_case(
        docker_settings,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    foreign_prepared, foreign_spawn = _activation_case(
        docker_settings,
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        credential_mode=RunActionCredentialMode.NONE,
    )

    with pytest.raises(RunActionActivationEnvelopeError, match="spawn differs"):
        activation_execution_event_size_bound(
            prepared_execution=prepared,
            spawn_commit=foreign_spawn,
            predecessor_event_id=_PREDECESSOR_EVENT_ID,
        )
    with pytest.raises(
        RunActionActivationEnvelopeError,
        match="predecessor is not",
    ):
        activation_execution_event_size_bound(
            prepared_execution=foreign_prepared,
            spawn_commit=spawn,
            predecessor_event_id=content_id(
                "foreign-event",
                {"fixture": "wrong predecessor"},
            ),
        )


def test_activation_envelope_schema_guards_fail_loud(docker_settings, monkeypatch):
    with pytest.raises(
        RunActionActivationEnvelopeError,
        match="envelope fields changed",
    ):
        envelope_module._sealed_wire(
            RunActionContainerLabel,
            key="com.kapso.test",
        )
    prepared, _spawn = _activation_case(
        docker_settings,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    monkeypatch.setattr(
        envelope_module,
        "_RUNTIME_VOLUME_EVIDENCE_FIELDS",
        (*envelope_module._RUNTIME_VOLUME_EVIDENCE_FIELDS, "schema_drift"),
    )
    with pytest.raises(
        RunActionActivationEnvelopeError,
        match="RuntimeVolumeEvidence envelope fields changed",
    ):
        envelope_module._reobserved_volume_wire(prepared.runtime_volume_evidence)


def test_credential_authority_is_one_fixed_width_content_identity(
    docker_settings,
):
    prepared, spawn = _activation_case(
        docker_settings,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    )
    receipt = _activation_revalidation_receipt(prepared, spawn)
    credential = receipt.credential_file_observation
    assert credential is not None

    with pytest.raises(
        CanonicalizationError,
        match="credential lease authority",
    ):
        _remint_contract(
            credential,
            content_authority_id="unbounded.legacy.credential.authority",
        )
    with pytest.raises(
        RunActionSupervisorContractError,
        match="activated run action file observation",
    ):
        _remint_contract(
            credential,
            inode=RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER + 1,
        )
    assert credential.content_authority_id.startswith(
        "run-action-credential-lease-authority:sha256:"
    )
