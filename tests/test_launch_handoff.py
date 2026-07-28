"""Prepared fresh/resume handoff is the only paid-orchestration boundary."""

from __future__ import annotations

from types import SimpleNamespace

from kapso.cross_run.launch.bootstrap import LaunchBootstrapCoordinator
from kapso.cross_run.launch.handoff import (
    PreparedRunActionRecoveryHandoff,
    PreparedRunHandoff,
    prepare_fresh_run_handoff,
    prepare_run_action_recovery_handoff,
    prepare_resumed_run_handoff,
)
from kapso.cross_run.launch.resume_contracts import RunReleaseUseMode
from test_launch_bootstrap import _fresh_coordinator
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import _security_observation
from test_run_resume_coordinator import _coordinator, _published_run
from test_run_frontier_action_gate import _action_case, _reserve_ideation_agent
from test_run_state_publisher import publisher_case


class _DescendantSecurityAuthority:
    def observe_exact_descendant_of(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
        required_ancestor,
    ):
        assert scope_id == required_ancestor.scope_id
        assert scope_contract_id == required_ancestor.scope_contract_id
        pin = SimpleNamespace(
            launch_manifest=SimpleNamespace(
                security_observation=required_ancestor,
            )
        )
        return _security_observation(pin, checked_subject_ids)


def test_fresh_handoff_maps_pinned_workspace_before_exposing_genesis(
    resolver_case,
    tmp_path,
):
    coordinator = _fresh_coordinator(resolver_case)
    run_root = (tmp_path / "fresh").absolute()

    handoff = prepare_fresh_run_handoff(
        coordinator=coordinator,
        settings=resolver_case["resolver"].settings,
        security_authority=_DescendantSecurityAuthority(),
        request=resolver_case["request"],
        run_root=run_root,
        objective_direction="maximize",
    )

    assert type(handoff) is PreparedRunHandoff
    assert not handoff.resumed
    assert handoff.repository_memory.source_commit_sha == (
        handoff.active_workspace.bootstrap_pin.installation_receipt.workspace_baseline_commit_sha
    )
    assert handoff.frontier.checkpoint.checkpoint_sequence == 0
    handoff.close()


def test_resume_handoff_maps_the_refreshed_checkpoint_head(
    resolver_case,
    tmp_path,
):
    settings, run_root, pin, previous = _published_run(resolver_case, tmp_path)
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

    handoff = prepare_resumed_run_handoff(
        coordinator=coordinator,
        settings=settings,
        run_root=run_root,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )

    assert type(handoff) is PreparedRunHandoff
    assert handoff.resumed
    branch = settings.launch.workspace_git_branch
    assert handoff.repository_memory.source_commit_sha == (
        handoff.frontier.checkpoint.safety_state.derivative_frontier.evidence.branch_heads[
            branch
        ]
    )
    assert handoff.frontier.checkpoint.checkpoint_sequence == (
        previous.checkpoint.checkpoint_sequence + 1
    )
    handoff.close()


def test_action_recovery_handoff_opens_before_live_ledger_projection(
    resolver_case,
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    reservation = _reserve_ideation_agent(gate, frontier)
    run_root = publisher_case["active"].run_root
    publisher_case["active"].close()

    handoff = prepare_run_action_recovery_handoff(
        coordinator=_fresh_coordinator(resolver_case),
        settings=resolver_case["resolver"].settings,
        run_root=run_root,
    )

    assert type(handoff) is PreparedRunActionRecoveryHandoff
    assert handoff.frontier.run_checkpoint_id == frontier.run_checkpoint_id
    assert handoff.publisher.action_ledger_snapshot().operation_tails[
        0
    ].operation_id == (reservation.intent.operation_id)
    handoff.close()
