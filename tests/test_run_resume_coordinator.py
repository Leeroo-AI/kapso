"""Policy-refreshed admission of a durable local run."""

from __future__ import annotations

from pathlib import Path

import pytest

import kapso.cross_run.launch.resume as resume_module
from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import (
    SecurityDenylistKind,
    SecurityDenylistRevocation,
)
from kapso.cross_run.launch.resume import (
    AdmittedRunResume,
    BlockedRunResume,
    RunResumeCoordinator,
    RunResumeError,
)
from kapso.cross_run.launch.resume_contracts import (
    RunEligibilityDisposition,
    RunReleaseUseMode,
    RunSafetyBoundary,
)
from kapso.cross_run.launch.run_state_publisher import RunStatePublisher
from kapso.cross_run.launch.workspace import StarterWorkspaceBuilder
from kapso.cross_run.launch.workspace_frontier import RunWorkspaceFrontierError
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import (
    _release_use_with_revocation,
    _security_observation,
)
from test_run_state_publisher import _genesis


class _ReleaseUseAuthority:
    def __init__(self, observation) -> None:
        self.observation = observation
        self.calls = []

    def observe_exact(self, *, scope_contract, checked_release_ids):
        self.calls.append((scope_contract, checked_release_ids))
        return self.observation


class _SecurityAuthority:
    def __init__(self, pin, revocations=()) -> None:
        self.pin = pin
        self.revocations = revocations
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
        return _security_observation(
            self.pin,
            checked_subject_ids,
            self.revocations,
            generation_offset=2,
        )


class _ForbiddenReleaseUseAuthority:
    def observe_exact(self, *, scope_contract, checked_release_ids):
        raise AssertionError("offline resume must not read knowledge CURRENT")


def _published_run(resolver_case, tmp_path):
    settings = resolver_case["resolver"]._settings
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    prepared = StarterWorkspaceBuilder(settings).build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="run-resume-coordinator",
        campaign_id="campaign-resume-coordinator",
    )
    active = prepared.activate()
    projection, bundle, checkpoint = _genesis(active, resolver_case)
    publisher = RunStatePublisher(active, settings.launch)
    permit = publisher.issue_publication_permit(None, checkpoint, bundle)
    published = publisher.publish(permit, checkpoint, bundle)
    run_root = active.run_root
    pin = active.bootstrap_pin
    active.close()
    return settings, run_root, pin, published


def _coordinator(
    *,
    settings,
    pin,
    release_use_observation,
    security_revocations=(),
):
    release_use = _ReleaseUseAuthority(release_use_observation)
    security = _SecurityAuthority(pin, security_revocations)
    return (
        RunResumeCoordinator(
            settings=settings,
            binding=pin.launch_manifest.launch_request.binding,
            security_authority=security,
            release_use_authority=release_use,
        ),
        release_use,
        security,
    )


def test_online_resume_publishes_policy_refreshed_successor(
    resolver_case,
    tmp_path,
):
    settings, run_root, pin, previous = _published_run(resolver_case, tmp_path)
    coordinator, release_use, security = _coordinator(
        settings=settings,
        pin=pin,
        release_use_observation=pin.launch_manifest.release_use_observation,
    )

    admitted = coordinator.resume(
        run_root,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    assert type(admitted) is AdmittedRunResume
    checkpoint = admitted.frontier.checkpoint
    assert checkpoint.predecessor_checkpoint_id == previous.run_checkpoint_id
    assert checkpoint.checkpoint_sequence == previous.checkpoint.checkpoint_sequence + 1
    assert checkpoint.last_stop is None
    assert checkpoint.safety_state.boundary is RunSafetyBoundary.RESUME
    assert checkpoint.safety_state.disposition is RunEligibilityDisposition.ELIGIBLE
    assert checkpoint.strategy_state == previous.checkpoint.strategy_state
    assert admitted.frontier.projection == previous.projection
    assert release_use.calls == [
        (
            pin.launch_manifest.scope_contract,
            (pin.launch_manifest.expert_manifest.release_id,),
        )
    ]
    assert security.calls[0][3] == previous.checkpoint.safety_state.security_observation
    admitted.close()


def test_offline_resume_uses_pin_and_is_reproducibility_only(
    resolver_case,
    tmp_path,
):
    settings, run_root, pin, _previous = _published_run(resolver_case, tmp_path)
    security = _SecurityAuthority(pin)
    coordinator = RunResumeCoordinator(
        settings=settings,
        binding=pin.launch_manifest.launch_request.binding,
        security_authority=security,
        release_use_authority=_ForbiddenReleaseUseAuthority(),
    )

    admitted = coordinator.resume(
        run_root,
        release_use_mode=RunReleaseUseMode.PINNED_OFFLINE,
    )

    assert type(admitted) is AdmittedRunResume
    assert (
        admitted.frontier.checkpoint.safety_state.disposition
        is RunEligibilityDisposition.REPRODUCIBILITY_ONLY
    )
    admitted.close()


def test_performance_revocation_preserves_reproducible_execution(
    resolver_case,
    tmp_path,
):
    settings, run_root, pin, _previous = _published_run(resolver_case, tmp_path)
    revoked = _release_use_with_revocation(pin)
    coordinator, _release_use, _security = _coordinator(
        settings=settings,
        pin=pin,
        release_use_observation=revoked,
    )

    admitted = coordinator.resume(
        run_root,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    assert type(admitted) is AdmittedRunResume
    assert (
        admitted.frontier.checkpoint.safety_state.disposition
        is RunEligibilityDisposition.REPRODUCIBILITY_ONLY
    )
    assert admitted.frontier.checkpoint.safety_state.release_use_observation == revoked
    admitted.close()


def test_security_revocation_publishes_block_and_releases_runtime(
    resolver_case,
    tmp_path,
):
    settings, run_root, pin, previous = _published_run(resolver_case, tmp_path)
    revocation = SecurityDenylistRevocation.mint(
        subject_id=pin.launch_manifest.expert_manifest.release_id,
        kind=SecurityDenylistKind.SECURITY,
        reason_code="unsafe_release",
        evidence_ids=(content_id("security-evidence", {"case": "resume"}),),
        recorded_at="2026-07-26T00:00:00Z",
    )
    coordinator, _release_use, _security = _coordinator(
        settings=settings,
        pin=pin,
        release_use_observation=pin.launch_manifest.release_use_observation,
        security_revocations=(revocation,),
    )

    blocked = coordinator.resume(
        run_root,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    assert type(blocked) is BlockedRunResume
    assert blocked.checkpoint.predecessor_checkpoint_id == previous.run_checkpoint_id
    assert (
        blocked.checkpoint.safety_state.disposition
        is RunEligibilityDisposition.SECURITY_BLOCKED
    )
    with StarterWorkspaceBuilder(settings).reopen(run_root) as reopened:
        durable = RunStatePublisher(reopened, settings.launch).load_reconciled()
        assert durable is not None
        assert durable.checkpoint == blocked.checkpoint


def test_security_block_replays_without_another_policy_read(
    resolver_case,
    tmp_path,
):
    settings, run_root, pin, _previous = _published_run(resolver_case, tmp_path)
    revocation = SecurityDenylistRevocation.mint(
        subject_id=pin.launch_manifest.expert_manifest.release_id,
        kind=SecurityDenylistKind.SECURITY,
        reason_code="unsafe_release",
        evidence_ids=(content_id("security-evidence", {"case": "replay"}),),
        recorded_at="2026-07-26T00:00:00Z",
    )
    coordinator, release_use, security = _coordinator(
        settings=settings,
        pin=pin,
        release_use_observation=pin.launch_manifest.release_use_observation,
        security_revocations=(revocation,),
    )
    first = coordinator.resume(
        run_root,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    policy_call_counts = (len(release_use.calls), len(security.calls))

    second = coordinator.resume(
        run_root,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    assert type(first) is BlockedRunResume
    assert type(second) is BlockedRunResume
    assert second.checkpoint == first.checkpoint
    assert (len(release_use.calls), len(security.calls)) == policy_call_counts
    with StarterWorkspaceBuilder(settings).reopen(run_root):
        pass


def test_failed_admission_validation_releases_runtime_immediately(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    settings, run_root, pin, _previous = _published_run(resolver_case, tmp_path)
    coordinator, _release_use, _security = _coordinator(
        settings=settings,
        pin=pin,
        release_use_observation=pin.launch_manifest.release_use_observation,
    )

    class _RejectedAdmission:
        def __init__(self, *, active_workspace, publisher, frontier):
            active_workspace.require_control_authority()
            publisher.require_current(frontier)
            raise RuntimeError("injected admission validation failure")

    monkeypatch.setattr(resume_module, "AdmittedRunResume", _RejectedAdmission)
    with pytest.raises(RuntimeError, match="injected admission"):
        coordinator.resume(
            run_root,
            release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
        )

    with StarterWorkspaceBuilder(settings).reopen(run_root):
        pass


def test_admitted_resume_rejects_cross_run_authority_splice(
    resolver_case,
    tmp_path,
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    settings, run_root_a, pin_a, _previous_a = _published_run(
        resolver_case,
        first_root,
    )
    _settings_b, run_root_b, pin_b, _previous_b = _published_run(
        resolver_case,
        second_root,
    )
    coordinator_a, _release_a, _security_a = _coordinator(
        settings=settings,
        pin=pin_a,
        release_use_observation=pin_a.launch_manifest.release_use_observation,
    )
    coordinator_b, _release_b, _security_b = _coordinator(
        settings=settings,
        pin=pin_b,
        release_use_observation=pin_b.launch_manifest.release_use_observation,
    )
    admitted_a = coordinator_a.resume(
        run_root_a,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    admitted_b = coordinator_b.resume(
        run_root_b,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    assert type(admitted_a) is AdmittedRunResume
    assert type(admitted_b) is AdmittedRunResume

    with pytest.raises(RunResumeError, match="exact live authority"):
        AdmittedRunResume(
            active_workspace=admitted_b.active_workspace,
            publisher=admitted_a.publisher,
            frontier=admitted_a.frontier,
        )

    admitted_a.close()
    admitted_b.close()


def test_dirty_workspace_fails_before_live_policy_refresh_and_releases_lock(
    resolver_case,
    tmp_path,
):
    settings, run_root, pin, _previous = _published_run(resolver_case, tmp_path)
    workspace = run_root / pin.installation_receipt.layout.workspace_relative_path
    source_file = next(
        path
        for path in sorted(workspace.rglob("*"))
        if path.is_file() and ".git" not in path.parts
    )
    source_file.write_bytes(source_file.read_bytes() + b"\ndirty\n")
    coordinator, release_use, security = _coordinator(
        settings=settings,
        pin=pin,
        release_use_observation=pin.launch_manifest.release_use_observation,
    )

    with pytest.raises(RunWorkspaceFrontierError):
        coordinator.resume(
            run_root,
            release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
        )

    assert release_use.calls == []
    assert security.calls == []
    with StarterWorkspaceBuilder(settings).reopen(run_root):
        pass
