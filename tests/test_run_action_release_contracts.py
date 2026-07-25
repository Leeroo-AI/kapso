"""Contracts and admission capacity for workload release authority."""

from __future__ import annotations

from dataclasses import fields

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    SecurityDenylistKind,
    SecurityDenylistRevocation,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_release_authority import (
    mint_run_action_workload_release_receipt,
    require_run_action_workload_release_receipt_matches_event,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionCredentialValidityObservation,
    RunActionReleaseAuthorizationObservation,
    RunActionReleaseContractError,
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialMode,
    RunActionSupervisorContractError,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from test_run_action_barrier_contracts import _resolved_graph
from test_run_action_supervisor_contracts import (
    _claim,
    _execution_policy,
    _prepared_execution,
)

_AUTHORIZED_BOOTTIME_NANOSECONDS = 50_000_000_000
_AUTHORIZED_REALTIME_NANOSECONDS = 1_800_000_000_000_000_000
_NANOSECONDS_PER_SECOND = 1_000_000_000


def _security_observation(*, generation=3):
    checked_subject_id = content_id(
        "run-action-security-subject",
        {"name": "release-contract"},
    )
    return SecurityDenylistObservation.mint(
        scope_id="test.release.scope",
        scope_contract_id=content_id(
            "domain-scope-contract",
            {"scope": "test.release.scope"},
        ),
        scope_repository_binding_hash=tree_or_blob_digest(b"repository binding"),
        snapshot_id=content_id(
            "security-denylist-snapshot",
            {"generation": generation},
        ),
        generation=generation,
        publication_id=content_id(
            "github-publication",
            {"generation": generation},
        ),
        repository_full_name="Leeroo-AI/kapso-knowledge",
        repository_node_id="repository-node",
        pointer_digest=tree_or_blob_digest(b"security pointer"),
        authority_commit_sha="a" * 40,
        release_attestation_ref="attestations/security/current",
        checked_subject_ids=(checked_subject_id,),
        matched_revocations=(),
    )


def _remint(contract, **changes):
    values = {
        field.name: getattr(contract, field.name)
        for field in fields(contract)
        if field.name != contract.IDENTITY_FIELD
    }
    values.update(changes)
    return type(contract).mint(**values)


def _resolved_for_security(
    security_observation,
    *,
    credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    release_receipt_size_bytes=None,
):
    policy = _execution_policy(
        kind=(
            RunFrontierActionKind.CODING_AGENT
            if credential_mode is RunActionCredentialMode.SUPERVISOR_FILE
            else RunFrontierActionKind.EMBEDDING
        ),
        workspace_access=(
            RunFrontierWorkspaceAccess.EDIT_WORKSPACE
            if credential_mode is RunActionCredentialMode.SUPERVISOR_FILE
            else RunFrontierWorkspaceAccess.NONE
        ),
        credential_mode=credential_mode,
    )
    if release_receipt_size_bytes is not None:
        limits = _remint(
            policy.supervisor_limits,
            release_receipt_size_bytes=release_receipt_size_bytes,
        )
        policy = _remint(policy, supervisor_limits=limits)
    claim = _claim(
        policy=policy,
        security_observation_id=security_observation.observation_id,
    )
    return _resolved_graph(prepared=_prepared_execution(claim=claim))


def _release_receipt(
    resolved,
    security_observation,
    *,
    credential_validity=True,
):
    activation = resolved.activation_revalidation_receipt
    limits = (
        activation.prepared_execution.preparation_claim.execution_policy.supervisor_limits
    )
    credential_file = activation.credential_file_observation
    validity = None
    if credential_validity:
        validity = RunActionCredentialValidityObservation.mint(
            activated_credential_file_observation_id=(
                credential_file.activated_file_observation_id
            ),
            credential_lease_authority_id=credential_file.content_authority_id,
            observed_at_realtime_nanoseconds=(
                _AUTHORIZED_REALTIME_NANOSECONDS - _NANOSECONDS_PER_SECOND
            ),
            valid_until_realtime_nanoseconds=(
                _AUTHORIZED_REALTIME_NANOSECONDS
                + (limits.execution_timeout_seconds + limits.termination_grace_seconds)
                * _NANOSECONDS_PER_SECOND
            ),
        )
    authorization = RunActionReleaseAuthorizationObservation.mint(
        security_observation=security_observation,
        authorized_at_boottime_nanoseconds=_AUTHORIZED_BOOTTIME_NANOSECONDS,
        authorized_at_realtime_nanoseconds=_AUTHORIZED_REALTIME_NANOSECONDS,
        credential_validity_observation=validity,
    )
    return mint_run_action_workload_release_receipt(
        activation_event=_activation_event(resolved),
        resolved_workload_observation=resolved,
        release_authorization_observation=authorization,
    )


def _activation_event(resolved, *, predecessor_label="event-four"):
    activation = resolved.activation_revalidation_receipt
    return RunActionExecutionEvent.mint(
        event_number=5,
        predecessor_event_id=content_id(
            "run-action-execution-event",
            {"fixture": predecessor_label},
        ),
        event_kind=RunActionExecutionEventKind.ACTIVATION_COMMITTED,
        reservation=activation.prepared_execution.preparation_claim.reservation,
        preparation_allocation=None,
        prepared_execution=None,
        spawn_commit=None,
        activation_revalidation_receipt=activation,
        result_receipt=None,
        result_decision=None,
        acceptance=None,
        terminal_reason=None,
        workspace_after=None,
    )


def test_release_receipt_round_trips_and_derives_same_boot_deadlines():
    security = _security_observation()
    resolved = _resolved_for_security(security)

    receipt = _release_receipt(resolved, security)

    limits = (
        resolved.activation_revalidation_receipt.prepared_execution.preparation_claim.execution_policy.supervisor_limits
    )
    assert (
        receipt.execution_deadline_boottime_nanoseconds
        == _AUTHORIZED_BOOTTIME_NANOSECONDS
        + limits.execution_timeout_seconds * _NANOSECONDS_PER_SECOND
    )
    assert (
        receipt.containment_deadline_boottime_nanoseconds
        == receipt.execution_deadline_boottime_nanoseconds
        + limits.termination_grace_seconds * _NANOSECONDS_PER_SECOND
    )
    assert receipt.host_boot_id == resolved.host_boot_id
    assert (
        RunActionWorkloadReleaseReceipt.from_json_bytes(receipt.to_json_bytes())
        == receipt
    )
    assert len(receipt.to_json_bytes()) <= limits.release_receipt_size_bytes


def test_credential_free_release_requires_and_accepts_no_validity_observation():
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )

    receipt = _release_receipt(
        resolved,
        security,
        credential_validity=False,
    )

    assert (
        receipt.release_authorization_observation.credential_validity_observation
        is None
    )


def test_release_receipt_rejects_security_generation_splice():
    security = _security_observation()
    resolved = _resolved_for_security(security)
    advanced = _security_observation(generation=security.generation + 1)

    with pytest.raises(
        RunActionReleaseContractError,
        match="differs from event-5 authority",
    ):
        _release_receipt(resolved, advanced)


def test_release_receipt_requires_credential_validity_through_containment():
    security = _security_observation()
    resolved = _resolved_for_security(security)
    receipt = _release_receipt(resolved, security)
    authorization = receipt.release_authorization_observation
    validity = authorization.credential_validity_observation

    too_short = _remint(
        validity,
        valid_until_realtime_nanoseconds=(
            validity.valid_until_realtime_nanoseconds - 1
        ),
    )
    changed_authorization = _remint(
        authorization,
        credential_validity_observation=too_short,
    )
    with pytest.raises(
        RunActionReleaseContractError,
        match="differs from event-5 authority",
    ):
        _remint(
            receipt,
            release_authorization_observation=changed_authorization,
        )


def test_credential_validity_cannot_exceed_policy_lease_authority():
    security = _security_observation()
    resolved = _resolved_for_security(security)
    receipt = _release_receipt(resolved, security)
    authorization = receipt.release_authorization_observation
    validity = authorization.credential_validity_observation
    policy = (
        resolved.activation_revalidation_receipt.prepared_execution.preparation_claim.execution_policy
    )
    maximum_interval = (
        policy.credential_policy.maximum_lease_seconds * _NANOSECONDS_PER_SECOND
    )
    exact = _remint(
        validity,
        observed_at_realtime_nanoseconds=(
            validity.valid_until_realtime_nanoseconds - maximum_interval
        ),
    )
    exact_authorization = _remint(
        authorization,
        credential_validity_observation=exact,
    )
    exact_receipt = _remint(
        receipt,
        release_authorization_observation=exact_authorization,
    )
    assert (
        exact_receipt.release_authorization_observation.credential_validity_observation
        == exact
    )

    too_long = _remint(
        exact,
        observed_at_realtime_nanoseconds=(exact.observed_at_realtime_nanoseconds - 1),
    )
    with pytest.raises(
        RunActionReleaseContractError,
        match="differs from event-5 authority",
    ):
        _remint(
            receipt,
            release_authorization_observation=_remint(
                authorization,
                credential_validity_observation=too_long,
            ),
        )


def test_credential_policy_must_be_capable_of_spanning_containment():
    policy = _execution_policy()
    required_seconds = (
        policy.supervisor_limits.execution_timeout_seconds
        + policy.supervisor_limits.termination_grace_seconds
    )
    exact_credential_policy = _remint(
        policy.credential_policy,
        maximum_lease_seconds=required_seconds,
    )
    exact_policy = _remint(
        policy,
        credential_policy=exact_credential_policy,
    )
    assert exact_policy.credential_policy.maximum_lease_seconds == required_seconds

    with pytest.raises(
        RunActionSupervisorContractError,
        match="credential lease cannot span containment",
    ):
        _remint(
            policy,
            credential_policy=_remint(
                policy.credential_policy,
                maximum_lease_seconds=required_seconds - 1,
            ),
        )


def test_release_receipt_rejects_missing_or_unexpected_credential_validity():
    security = _security_observation()
    credentialed = _resolved_for_security(security)
    credentialed_receipt = _release_receipt(credentialed, security)
    without_validity = _remint(
        credentialed_receipt.release_authorization_observation,
        credential_validity_observation=None,
    )
    with pytest.raises(
        RunActionReleaseContractError,
        match="differs from event-5 authority",
    ):
        _remint(
            credentialed_receipt,
            release_authorization_observation=without_validity,
        )

    credential_free = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )
    unexpected = _remint(
        credentialed_receipt.release_authorization_observation,
        security_observation=security,
    )
    with pytest.raises(
        RunActionReleaseContractError,
        match="differs from event-5 authority",
    ):
        RunActionWorkloadReleaseReceipt.mint(
            activation_event_id=credentialed_receipt.activation_event_id,
            resolved_workload_observation=credential_free,
            release_authorization_observation=unexpected,
        )


def test_release_receipt_must_fit_the_policy_reserved_bound():
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        release_receipt_size_bytes=1,
    )

    with pytest.raises(
        RunActionReleaseContractError,
        match="differs from event-5 authority",
    ):
        _release_receipt(resolved, security)


def test_release_receipt_rejects_non_event_identity():
    security = _security_observation()
    resolved = _resolved_for_security(security)
    receipt = _release_receipt(resolved, security)

    with pytest.raises(
        RunActionReleaseContractError,
        match="uses another namespace",
    ):
        _remint(
            receipt,
            activation_event_id=content_id(
                "run-action-spawn-commit",
                {"event": 5},
            ),
        )


def test_release_receipt_authority_rejects_another_same_namespace_event():
    security = _security_observation()
    resolved = _resolved_for_security(security)
    receipt = _release_receipt(resolved, security)
    other_event = _activation_event(
        resolved,
        predecessor_label="another-event-four",
    )

    with pytest.raises(
        RunActionReleaseContractError,
        match="identifies another activation event",
    ):
        require_run_action_workload_release_receipt_matches_event(
            receipt,
            other_event,
        )

    structurally_valid_wrong_id = _remint(
        receipt,
        activation_event_id=other_event.event_id,
    )
    with pytest.raises(
        RunActionReleaseContractError,
        match="identifies another activation event",
    ):
        require_run_action_workload_release_receipt_matches_event(
            structurally_valid_wrong_id,
            _activation_event(resolved),
        )


def test_release_authorization_rejects_matched_revocation():
    security = _security_observation()
    revocation = SecurityDenylistRevocation.mint(
        subject_id=security.checked_subject_ids[0],
        kind=SecurityDenylistKind.CONTAMINATION,
        reason_code="contaminated_release_dependency",
        evidence_ids=(content_id("security-evidence", {"case": "release-contract"}),),
        recorded_at="2026-07-23T00:00:00Z",
    )
    revoked = _remint(
        security,
        matched_revocations=(revocation,),
    )

    with pytest.raises(
        RunActionReleaseContractError,
        match="unsafe or invalid",
    ):
        RunActionReleaseAuthorizationObservation.mint(
            security_observation=revoked,
            authorized_at_boottime_nanoseconds=_AUTHORIZED_BOOTTIME_NANOSECONDS,
            authorized_at_realtime_nanoseconds=_AUTHORIZED_REALTIME_NANOSECONDS,
            credential_validity_observation=None,
        )
