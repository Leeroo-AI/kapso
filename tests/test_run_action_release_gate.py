"""Opaque final authorization and atomic publication of workload release."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from threading import get_ident, Thread
from types import SimpleNamespace

import pytest

import kapso.cross_run.launch.run_action_resolved_workload as workload_module
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    _RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnQuery,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_release_contracts import (
    RunActionCredentialValidityObservation,
    RunActionWorkloadReleaseReceipt,
)
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_control_candidate import (
    _RunActionFrozenControlFileCandidate,
    RunActionControlCandidateError,
)
from kapso.cross_run.launch.run_action_release_publisher import (
    publish_run_action_workload_release_once,
)
from kapso.cross_run.launch.run_action_resolved_workload import (
    RunActionBlockedWorkloadLease,
    RunActionResolvedWorkloadError,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialMode,
    RunActionPreparationAllocation,
)
from test_run_action_barrier_contracts import _resolved_graph
from test_run_action_release_contracts import (
    _activation_event,
    _resolved_for_security,
    _security_observation,
)

_INITIAL_BOOTTIME_NANOSECONDS = 50_000_000_000
_INITIAL_REALTIME_NANOSECONDS = 1_800_000_000_000_000_000
_CLOCK_STEP_NANOSECONDS = 1_000
_NANOSECONDS_PER_SECOND = 1_000_000_000


class _Clock:
    def __init__(self) -> None:
        self.boottime = _INITIAL_BOOTTIME_NANOSECONDS
        self.realtime = _INITIAL_REALTIME_NANOSECONDS

    def boottime_nanoseconds(self):
        observed = self.boottime
        self.boottime += _CLOCK_STEP_NANOSECONDS
        return observed

    def realtime_nanoseconds(self):
        observed = self.realtime
        self.realtime += _CLOCK_STEP_NANOSECONDS
        return observed


class _SecurityAuthority:
    def __init__(self, observation, *, before_return=None) -> None:
        self.observation = observation
        self.before_return = before_return
        self.calls = []

    def observe_exact_descendant_of(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
        required_ancestor,
    ):
        self.calls.append(
            (
                scope_id,
                scope_contract_id,
                checked_subject_ids,
                required_ancestor,
            )
        )
        if self.before_return is not None:
            self.before_return()
        return self.observation


class _RaisingSecurityAuthority(_SecurityAuthority):
    def observe_exact_descendant_of(self, **arguments):
        super().observe_exact_descendant_of(**arguments)
        raise RuntimeError("injected security authority failure")


class _CredentialAuthority:
    def __init__(self, clock, maximum_lease_seconds, *, expire_final=False) -> None:
        self.clock = clock
        self.maximum_lease_seconds = maximum_lease_seconds
        self.expire_final = expire_final
        self.calls = []

    def observe_exact(
        self,
        *,
        activated_credential_file_observation_id,
        credential_lease_authority_id,
    ):
        self.calls.append(
            (
                activated_credential_file_observation_id,
                credential_lease_authority_id,
            )
        )
        observed_at = self.clock.realtime_nanoseconds()
        valid_until = (
            observed_at + _CLOCK_STEP_NANOSECONDS
            if self.expire_final and len(self.calls) == 2
            else observed_at + self.maximum_lease_seconds * _NANOSECONDS_PER_SECOND
        )
        return RunActionCredentialValidityObservation.mint(
            activated_credential_file_observation_id=(
                activated_credential_file_observation_id
            ),
            credential_lease_authority_id=credential_lease_authority_id,
            observed_at_realtime_nanoseconds=observed_at,
            valid_until_realtime_nanoseconds=valid_until,
        )


class _ReleaseAdapter:
    def __init__(self, callback) -> None:
        self._callback = callback

    def continue_committed_once(self, capability):
        self._callback(capability)
        return RunActionContinuationOutcome(
            state=RunActionContinuationState.PENDING,
            result=None,
            provider_termination_receipt=None,
        )


def _capability(
    resolved,
    security,
    authority,
    *,
    credential_authority=None,
    clock=None,
    state=None,
):
    activation_event = _activation_event(resolved)
    prepared = resolved.activation_revalidation_receipt.prepared_execution
    release_clock = _SystemRunActionClock()
    if clock is not None:
        release_clock.boottime_nanoseconds = clock.boottime_nanoseconds
        release_clock.realtime_nanoseconds = clock.realtime_nanoseconds
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_authority=prepared.runtime_volume_authority,
    )
    return RunActionCommittedContinuationCapability(
        query=RunActionCommittedSpawnQuery(
            preparation_allocation=allocation,
            activation_event=activation_event,
            workload_release_adoption=None,
            timeout_directive_publication=None,
        ),
        observation=RunActionCommittedSpawnObservation(
            state=(
                RunActionCommittedSpawnState.RUNNING_CONTINUABLE
                if state is None
                else state
            ),
            observation_token=(
                resolved.running_container_observation.complete_inspection_digest
            ),
        ),
        required_security_observation=security,
        security_authority=authority,
        credential_validity_authority=credential_authority,
        release_clock=release_clock,
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )


def _lease(
    resolved,
    control_directory: Path,
    *,
    issued=True,
):
    control_directory.chmod(0o700)
    control_descriptor = os.open(
        control_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    lease = object.__new__(RunActionBlockedWorkloadLease)
    lease._owner_process_id = os.getpid()
    lease._owner_thread_id = get_ident()
    lease._closed = False
    lease._resolved_workload_observation = resolved
    lease._activation_event = _activation_event(resolved)
    lease._control_lease = SimpleNamespace(
        _control_descriptor=control_descriptor,
    )
    lease.test_current_calls = 0
    if issued:
        with workload_module._BLOCKED_WORKLOAD_LEASE_LOCK:
            workload_module._ISSUED_BLOCKED_WORKLOAD_LEASES[id(lease)] = lease
    return lease, control_descriptor


def _require_test_lease_current(self):
    self._require_issued()
    if (
        self._closed
        or self._owner_process_id != os.getpid()
        or self._owner_thread_id != get_ident()
    ):
        raise RunActionResolvedWorkloadError(
            "blocked workload lease is closed, forked, or on another thread"
        )
    self.test_current_calls += 1
    if tuple(os.listdir(self._control_lease._control_descriptor)):
        raise RunActionResolvedWorkloadError(
            "blocked workload release is already present"
        )


@pytest.fixture(autouse=True)
def _test_lease_physical_revalidation(monkeypatch):
    monkeypatch.setattr(
        RunActionBlockedWorkloadLease,
        "require_current",
        _require_test_lease_current,
    )


def _close_test_lease(lease, control_descriptor):
    with workload_module._BLOCKED_WORKLOAD_LEASE_LOCK:
        workload_module._ISSUED_BLOCKED_WORKLOAD_LEASES.pop(id(lease), None)
    os.close(control_descriptor)
    lease._closed = True


def test_atomic_release_publishes_only_after_final_exact_security(tmp_path):
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )
    authority = _SecurityAuthority(security)
    clock = _Clock()
    capability = _capability(resolved, security, authority, clock=clock)
    (tmp_path / "control").mkdir(mode=0o700, exist_ok=True)
    lease, descriptor = _lease(resolved, tmp_path / "control")
    published = []

    def release(active_capability):
        published.append(
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )
        )

    capability._invoke_once(_ReleaseAdapter(release))

    receipt = published[0]
    release_path = tmp_path / "control" / "release"
    assert (
        RunActionWorkloadReleaseReceipt.from_json_bytes(release_path.read_bytes())
        == receipt
    )
    assert stat.S_IMODE(release_path.stat().st_mode) == 0o400
    assert release_path.stat().st_nlink == 1
    assert lease.test_current_calls >= 4
    assert authority.calls == [
        (
            security.scope_id,
            security.scope_contract_id,
            security.checked_subject_ids,
            security,
        )
    ]
    assert not hasattr(capability, "_security_authority")
    assert not hasattr(capability, "authorize_workload_release_once")
    _close_test_lease(lease, descriptor)


def test_publication_api_has_no_adapter_supplied_authority_inputs():
    with pytest.raises(TypeError, match="unexpected keyword"):
        publish_run_action_workload_release_once(
            capability=None,
            blocked_workload_lease=None,
            clock=_Clock(),
        )


def test_unissued_frozen_candidate_cannot_consume_final_security():
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )
    authority = _SecurityAuthority(security)
    capability = _capability(
        resolved,
        security,
        authority,
        clock=_Clock(),
    )

    def attack(active_capability):
        with active_capability._begin_release_publication(
            resolved,
            _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
        ) as authorization:
            authorization._mint_receipt(
                _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
            )
            forged = object.__new__(_RunActionFrozenControlFileCandidate)
            with pytest.raises(
                RunActionControlCandidateError,
                match="unissued, spent, or foreign",
            ):
                authorization._authorize_frozen_release_once(
                    candidate=forged,
                    _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
                )

    capability._invoke_once(_ReleaseAdapter(attack))

    assert authority.calls == []


def test_unissued_spliced_graph_cannot_reach_security_or_link(tmp_path):
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )
    spliced = _resolved_graph(
        inode_offset=1,
        prepared=resolved.activation_revalidation_receipt.prepared_execution,
    )
    assert (
        spliced.running_container_observation.complete_inspection_digest
        == resolved.running_container_observation.complete_inspection_digest
    )
    authority = _SecurityAuthority(security)
    capability = _capability(resolved, security, authority)
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    lease, descriptor = _lease(spliced, control, issued=False)

    def release(active_capability):
        with pytest.raises(
            RunActionResolvedWorkloadError,
            match="unissued, closed, or foreign",
        ):
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )

    capability._invoke_once(_ReleaseAdapter(release))

    assert authority.calls == []
    assert tuple(control.iterdir()) == ()
    os.close(descriptor)


def test_security_advance_denies_link_and_burns_publication(tmp_path):
    required = _security_observation()
    advanced = _security_observation(generation=required.generation + 1)
    resolved = _resolved_for_security(
        required,
        credential_mode=RunActionCredentialMode.NONE,
    )
    authority = _SecurityAuthority(advanced)
    capability = _capability(resolved, required, authority)
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    lease, descriptor = _lease(resolved, control)

    def release(active_capability):
        assert (
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )
            is None
        )
        with pytest.raises(
            RunActionRecoveryError,
            match="lacks exact live authority",
        ):
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )

    capability._invoke_once(_ReleaseAdapter(release))

    assert len(authority.calls) == 1
    assert tuple(control.iterdir()) == ()
    _close_test_lease(lease, descriptor)


def test_security_exception_burns_publication_and_leaves_release_absent(tmp_path):
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )
    authority = _RaisingSecurityAuthority(security)
    capability = _capability(resolved, security, authority)
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    lease, descriptor = _lease(resolved, control)

    def release(active_capability):
        with pytest.raises(RuntimeError, match="injected security"):
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )
        with pytest.raises(
            RunActionRecoveryError,
            match="lacks exact live authority",
        ):
            active_capability._begin_release_publication(
                resolved,
                _authority=_RUN_ACTION_RELEASE_PUBLISHER_AUTHORITY,
            )

    capability._invoke_once(_ReleaseAdapter(release))

    assert tuple(control.iterdir()) == ()
    _close_test_lease(lease, descriptor)


def test_post_link_failure_leaves_canonical_receipt_for_adoption(
    tmp_path,
    monkeypatch,
):
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )
    authority = _SecurityAuthority(security)
    capability = _capability(
        resolved,
        security,
        authority,
        clock=_Clock(),
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    lease, descriptor = _lease(resolved, control)
    original = _RunActionFrozenControlFileCandidate._require_linked_file

    def fail_after_link(candidate):
        if candidate._state == "linked":
            raise RuntimeError("injected failure after irreversible link")
        return original(candidate)

    monkeypatch.setattr(
        _RunActionFrozenControlFileCandidate,
        "_require_linked_file",
        fail_after_link,
    )

    def release(active_capability):
        with pytest.raises(RuntimeError, match="after irreversible link"):
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )

    capability._invoke_once(_ReleaseAdapter(release))

    release_path = control / "release"
    receipt = RunActionWorkloadReleaseReceipt.from_json_bytes(release_path.read_bytes())
    assert receipt.activation_event_id == _activation_event(resolved).event_id
    assert stat.S_IMODE(release_path.stat().st_mode) == 0o400
    assert len(authority.calls) == 1
    _close_test_lease(lease, descriptor)


def test_release_commit_deadline_expiry_after_security_prevents_link(tmp_path):
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )
    clock = _Clock()
    commit_timeout = (
        resolved.activation_revalidation_receipt.prepared_execution.preparation_claim.execution_policy.supervisor_limits.release_commit_timeout_seconds
    )

    def expire_commit_window():
        clock.boottime = (
            _INITIAL_BOOTTIME_NANOSECONDS
            + (commit_timeout + 1) * _NANOSECONDS_PER_SECOND
        )

    authority = _SecurityAuthority(
        security,
        before_return=expire_commit_window,
    )
    capability = _capability(resolved, security, authority, clock=clock)
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    lease, descriptor = _lease(resolved, control)

    def release(active_capability):
        assert (
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )
            is None
        )

    capability._invoke_once(_ReleaseAdapter(release))

    assert tuple(control.iterdir()) == ()
    _close_test_lease(lease, descriptor)


def test_credential_revalidation_failure_prevents_security_and_link(tmp_path):
    security = _security_observation()
    resolved = _resolved_for_security(security)
    policy = (
        resolved.activation_revalidation_receipt.prepared_execution.preparation_claim.execution_policy
    )
    clock = _Clock()
    credential_authority = _CredentialAuthority(
        clock,
        policy.credential_policy.maximum_lease_seconds,
        expire_final=True,
    )
    security_authority = _SecurityAuthority(security)
    capability = _capability(
        resolved,
        security,
        security_authority,
        credential_authority=credential_authority,
        clock=clock,
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    lease, descriptor = _lease(resolved, control)

    def release(active_capability):
        with pytest.raises(
            RunActionRecoveryError,
            match="credential authority changed",
        ):
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )

    capability._invoke_once(_ReleaseAdapter(release))

    assert len(credential_authority.calls) == 2
    assert security_authority.calls == []
    assert tuple(control.iterdir()) == ()
    _close_test_lease(lease, descriptor)


def test_credentialed_release_revalidates_same_lease_through_containment(
    tmp_path,
):
    security = _security_observation()
    resolved = _resolved_for_security(security)
    policy = (
        resolved.activation_revalidation_receipt.prepared_execution.preparation_claim.execution_policy
    )
    clock = _Clock()
    credential_authority = _CredentialAuthority(
        clock,
        policy.credential_policy.maximum_lease_seconds,
    )
    security_authority = _SecurityAuthority(security)
    capability = _capability(
        resolved,
        security,
        security_authority,
        credential_authority=credential_authority,
        clock=clock,
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    lease, descriptor = _lease(resolved, control)
    published = []

    def release(active_capability):
        published.append(
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )
        )

    capability._invoke_once(_ReleaseAdapter(release))

    assert len(credential_authority.calls) == 2
    assert len(security_authority.calls) == 1
    assert (
        published[0].release_authorization_observation.credential_validity_observation
        is not None
    )
    assert (control / "release").read_bytes() == published[0].to_json_bytes()
    _close_test_lease(lease, descriptor)


def test_release_publication_is_owner_thread_and_running_state_bound(tmp_path):
    security = _security_observation()
    resolved = _resolved_for_security(
        security,
        credential_mode=RunActionCredentialMode.NONE,
    )
    authority = _SecurityAuthority(security)
    capability = _capability(
        resolved,
        security,
        authority,
        state=RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    lease, descriptor = _lease(resolved, control)
    failures = []

    def release(active_capability):
        def foreign_thread():
            with pytest.raises(
                RunActionResolvedWorkloadError,
                match="closed, forked, or on another thread",
            ):
                publish_run_action_workload_release_once(
                    capability=active_capability,
                    blocked_workload_lease=lease,
                )
            failures.append("thread")

        thread = Thread(target=foreign_thread)
        thread.start()
        thread.join()
        with pytest.raises(
            RunActionRecoveryError,
            match="lacks exact live authority",
        ):
            publish_run_action_workload_release_once(
                capability=active_capability,
                blocked_workload_lease=lease,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="terminal continuation lacks its trusted reinspection",
    ):
        capability._invoke_once(_ReleaseAdapter(release))

    assert failures == ["thread"]
    assert authority.calls == []
    _close_test_lease(lease, descriptor)
