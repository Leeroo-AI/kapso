"""Production composition of terminal actions into fresh safety boundaries."""

from __future__ import annotations

import kapso.cross_run.launch.run_action_recovery as run_action_recovery_module

from kapso.cross_run.launch.boundary import publish_run_boundary
from kapso.cross_run.launch.resume_contracts import RunSafetyBoundary
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import _security_observation
from test_run_action_recovery import (
    _append_result_accepted,
    _append_result_received,
    _bind_successful_edit_result,
    _FakeExecutionAdapter,
    _FakeResultWorkspaceLease,
    _isolated_edit_candidate,
    _recovery_coordinator,
)
from test_run_frontier_action_gate import (
    _action_case,
    _reserve_ideation_agent,
    _reserve_implementation_agent,
    publisher_case,
)


class _CurrentReleaseUseAuthority:
    def __init__(self, pin):
        self.pin = pin
        self.calls = []

    def observe_exact(self, *, scope_contract, checked_release_ids):
        manifest = self.pin.launch_manifest
        assert scope_contract == manifest.scope_contract
        assert checked_release_ids == (manifest.expert_manifest.release_id,)
        self.calls.append((scope_contract, checked_release_ids))
        return manifest.release_use_observation


class _FreshSecurityAuthority:
    def __init__(self, pin):
        self.pin = pin
        self.calls = []

    def observe_exact_descendant_of(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
        required_ancestor,
    ):
        manifest = self.pin.launch_manifest
        assert scope_id == manifest.scope_contract.scope_id
        assert scope_contract_id == manifest.scope_contract.scope_contract_id
        self.calls.append((checked_subject_ids, required_ancestor))
        return _security_observation(
            self.pin,
            checked_subject_ids,
            generation_offset=(
                required_ancestor.generation
                - manifest.security_observation.generation
                + 1
            ),
        )


def _authorities(case):
    pin = case["active"].bootstrap_pin
    return _FreshSecurityAuthority(pin), _CurrentReleaseUseAuthority(pin)


def test_boundary_reconciles_terminal_read_only_action(publisher_case):
    publisher, frontier, _security, gate = _action_case(publisher_case)
    reservation = _reserve_ideation_agent(gate, frontier)
    _append_result_accepted(gate, reservation)
    security, release_use = _authorities(publisher_case)

    published = publish_run_boundary(
        publisher=publisher,
        frontier=frontier,
        security_authority=security,
        release_use_authority=release_use,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )

    assert published.checkpoint.safety_state.boundary is (
        RunSafetyBoundary.IMPLEMENTATION
    )
    assert published.projection.action_ledger.event_count == 8
    assert (
        published.checkpoint.safety_state.derivative_frontier.evidence.branch_heads
        == frontier.checkpoint.safety_state.derivative_frontier.evidence.branch_heads
    )
    assert len(security.calls) == 1
    assert len(release_use.calls) == 1


def test_boundary_accounts_for_one_promoted_direct_successor(
    publisher_case,
    tmp_path,
    monkeypatch,
):
    publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
    )
    reservation = _reserve_implementation_agent(
        gate,
        frontier,
        "boundary_publication",
    )
    adapter = _FakeExecutionAdapter(reservation.intent.boundary_identity)
    candidate = _isolated_edit_candidate(publisher_case, tmp_path)
    _bind_successful_edit_result(
        adapter,
        reservation,
        candidate,
        publisher_case["settings"],
    )
    monkeypatch.setattr(
        run_action_recovery_module,
        "open_run_action_result_workspace",
        lambda _prepared, _capture: _FakeResultWorkspaceLease(candidate),
    )
    _append_result_received(gate, reservation)
    _recovery_coordinator(gate, adapter).recover(frontier)
    security, release_use = _authorities(publisher_case)

    published = publish_run_boundary(
        publisher=publisher,
        frontier=frontier,
        security_authority=security,
        release_use_authority=release_use,
        boundary=RunSafetyBoundary.EVALUATION,
    )

    prior_evidence = frontier.checkpoint.safety_state.derivative_frontier.evidence
    evidence = published.checkpoint.safety_state.derivative_frontier.evidence
    branch = publisher_case["settings"].workspace_git_branch
    assert evidence.branch_heads[branch] != prior_evidence.branch_heads[branch]
    new_advances = tuple(
        advance
        for advance in evidence.branch_advances
        if advance not in prior_evidence.branch_advances
    )
    assert len(new_advances) == 1
    assert new_advances[0].authorization_safety_state_id == (
        frontier.checkpoint.safety_state.safety_state_id
    )
    assert published.projection.action_ledger.event_count == 8
