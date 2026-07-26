"""Typed dormant composition of launch bootstrap and local resume."""

from __future__ import annotations

import re

import pytest

import kapso.cross_run.launch.bootstrap as bootstrap_module
from kapso.cross_run.contracts import CrossRunTaskBindingSettings
from kapso.cross_run.launch.bootstrap import (
    BootstrappedLaunchWorkspace,
    LaunchBootstrapCoordinator,
    LaunchBootstrapError,
    LaunchBootstrapIdentity,
)
from kapso.cross_run.launch.resume import (
    AdmittedRunResume,
    RunResumeCoordinator,
    RunResumeError,
)
from kapso.cross_run.launch.resume_contracts import RunReleaseUseMode
from kapso.cross_run.launch.run_state_publisher import RunStatePublisher
from kapso.cross_run.launch.workspace import StarterWorkspaceBuilder
from test_launch_resolver import resolver_case
from test_run_resume_coordinator import _coordinator, _published_run


class _UnusedReleaseUseAuthority:
    def observe_exact(self, *, scope_contract, checked_release_ids):
        raise AssertionError("fresh bootstrap must not invoke resume release-use")


class _UnusedSecurityAuthority:
    def observe_exact_descendant_of(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
        required_ancestor,
    ):
        raise AssertionError("fresh bootstrap must not invoke resume security")


def _fresh_coordinator(resolver_case, *, binding=None):
    settings = resolver_case["resolver"].settings
    return LaunchBootstrapCoordinator(
        settings=settings,
        binding=resolver_case["request"].binding if binding is None else binding,
        resolver=resolver_case["resolver"],
        resume_coordinator=RunResumeCoordinator(
            settings=settings,
            binding=resolver_case["request"].binding if binding is None else binding,
            security_authority=_UnusedSecurityAuthority(),
            release_use_authority=_UnusedReleaseUseAuthority(),
        ),
    )


def test_fresh_bootstrap_returns_exact_live_identity(resolver_case, tmp_path):
    run_root = (tmp_path / "run").absolute()

    bootstrapped = _fresh_coordinator(resolver_case).fresh(
        request=resolver_case["request"],
        run_root=run_root,
    )

    assert type(bootstrapped) is BootstrappedLaunchWorkspace
    assert re.fullmatch(r"run_[0-9a-f]{32}", bootstrapped.identity.run_id)
    assert re.fullmatch(r"campaign_[0-9a-f]{32}", bootstrapped.identity.campaign_id)
    assert bootstrapped.identity == LaunchBootstrapIdentity.from_bootstrap_pin(
        bootstrapped.active_workspace.bootstrap_pin
    )
    bootstrapped.active_workspace.require_control_authority()
    bootstrapped.close()


def test_binding_mismatch_fails_before_resolver_reads_current(
    resolver_case,
    tmp_path,
):
    mismatched_binding = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
    )
    coordinator = _fresh_coordinator(
        resolver_case,
        binding=mismatched_binding,
    )

    with pytest.raises(LaunchBootstrapError, match="task binding"):
        coordinator.fresh(
            request=resolver_case["request"],
            run_root=(tmp_path / "run").absolute(),
        )

    assert resolver_case["github"].resolve_counts == {
        artifact_kind: 0 for artifact_kind in resolver_case["github"].resolve_counts
    }
    assert resolver_case["task_adapters"].resolve_count == 0


def test_rejected_fresh_wrapper_releases_workspace_authority(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    run_root = (tmp_path / "run").absolute()

    class _RejectingBootstrappedWorkspace:
        def __init__(self, *, active_workspace, identity):
            active_workspace.require_control_authority()
            assert type(identity) is LaunchBootstrapIdentity
            raise LaunchBootstrapError("injected wrapper rejection")

    monkeypatch.setattr(
        bootstrap_module,
        "BootstrappedLaunchWorkspace",
        _RejectingBootstrappedWorkspace,
    )

    with pytest.raises(LaunchBootstrapError, match="injected"):
        _fresh_coordinator(resolver_case).fresh(
            request=resolver_case["request"],
            run_root=run_root,
        )

    settings = resolver_case["resolver"].settings
    with StarterWorkspaceBuilder(settings).reopen(run_root) as reopened:
        reopened.require_control_authority()


def test_resume_uses_local_pin_without_invoking_fresh_resolver(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    settings, run_root, pin, _previous = _published_run(resolver_case, tmp_path)
    resume_coordinator, _release_use, _security = _coordinator(
        settings=settings,
        pin=pin,
        release_use_observation=pin.launch_manifest.release_use_observation,
    )
    coordinator = LaunchBootstrapCoordinator(
        settings=settings,
        binding=resolver_case["request"].binding,
        resolver=resolver_case["resolver"],
        resume_coordinator=resume_coordinator,
    )

    def forbidden_resolve(_request):
        raise AssertionError("resume must not invoke fresh launch resolution")

    monkeypatch.setattr(resolver_case["resolver"], "resolve", forbidden_resolve)

    resumed = coordinator.resume(
        run_root,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    assert type(resumed) is AdmittedRunResume
    assert (
        resumed.active_workspace.bootstrap_pin.launch_manifest.launch_request.binding
        == resolver_case["request"].binding
    )
    resumed.close()


def test_resume_binding_rejection_releases_workspace_authority(
    resolver_case,
    tmp_path,
):
    settings, run_root, pin, previous = _published_run(resolver_case, tmp_path)
    mismatched_binding = CrossRunTaskBindingSettings(
        scope_id="ml_ai",
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
    )
    release_use = _UnusedReleaseUseAuthority()
    coordinator = LaunchBootstrapCoordinator(
        settings=settings,
        binding=mismatched_binding,
        resolver=resolver_case["resolver"],
        resume_coordinator=RunResumeCoordinator(
            settings=settings,
            binding=mismatched_binding,
            security_authority=_UnusedSecurityAuthority(),
            release_use_authority=release_use,
        ),
    )

    with pytest.raises(
        RunResumeError,
        match="local run pin differs from its configured task binding",
    ):
        coordinator.resume(
            run_root,
            release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
        )

    with StarterWorkspaceBuilder(settings).reopen(run_root) as reopened:
        reopened.require_control_authority()
        durable = RunStatePublisher(
            reopened,
            settings.launch,
        ).load_reconciled()
        assert durable is not None
        assert durable.checkpoint == previous.checkpoint
