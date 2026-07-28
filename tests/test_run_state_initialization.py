"""Fresh-run genesis composition after verified BootstrapPin handoff."""

from __future__ import annotations

from dataclasses import fields

import pytest

from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpointStatus,
    RunStrategyKind,
)
from kapso.cross_run.launch.initialization import (
    InitializedRunState,
    initialize_run_state,
    RunStateInitializationError,
)
from kapso.cross_run.launch.workspace import StarterWorkspaceBuilder
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import _security_observation


class _RecordingSecurityAuthority:
    def __init__(self, pin):
        self.pin = pin
        self.checked_subject_ids = None
        self.call_count = 0

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
        assert required_ancestor == manifest.security_observation
        self.call_count += 1
        self.checked_subject_ids = checked_subject_ids
        return _security_observation(
            self.pin,
            checked_subject_ids,
            generation_offset=1,
        )


def _request_for_mode(request, search_mode):
    values = {
        field.name: getattr(request, field.name)
        for field in fields(request)
        if field.name != "launch_request_id"
    }
    values["search_mode"] = search_mode
    return type(request).mint(**values)


@pytest.mark.parametrize(
    ("search_mode", "expected_authorities"),
    (
        (
            "generic",
            {
                "action_ledger",
                "idea_archive",
                "experiment_history",
                "execution_journal",
            },
        ),
        (
            "benchmark_tree_search",
            {"action_ledger", "experiment_history", "execution_journal"},
        ),
    ),
)
def test_initialization_publishes_one_reconciled_empty_frontier(
    resolver_case,
    tmp_path,
    search_mode,
    expected_authorities,
):
    request = _request_for_mode(resolver_case["request"], search_mode)
    resolved = resolver_case["resolver"].resolve(request)
    active = (
        StarterWorkspaceBuilder(resolver_case["resolver"].settings)
        .build(
            resolved,
            (tmp_path / search_mode).absolute(),
            run_id=f"run-initialization-{search_mode}",
            campaign_id=f"campaign-initialization-{search_mode}",
        )
        .activate()
    )
    security = _RecordingSecurityAuthority(active.bootstrap_pin)

    initialized = initialize_run_state(
        active_workspace=active,
        launch_settings=resolver_case["resolver"].settings.launch,
        security_authority=security,
        objective_direction="maximize",
    )

    assert type(initialized) is InitializedRunState
    assert initialized.frontier.checkpoint.status is RunCheckpointStatus.ACTIVE
    assert initialized.frontier.checkpoint.checkpoint_sequence == 0
    assert initialized.frontier.projection.strategy_state.strategy_kind is (
        RunStrategyKind(search_mode)
    )
    assert (
        initialized.frontier.projection.strategy_state.describes_empty_durable_frontier()
    )
    assert initialized.frontier.checkpoint.safety_state.boundary_sequence == 0
    assert (
        not initialized.frontier.checkpoint.safety_state.derivative_frontier.derivatives
    )
    assert (
        set(
            initialized.frontier.checkpoint.safety_state.derivative_frontier.evidence.state_authority_digests
        )
        == expected_authorities
    )
    assert security.call_count == 1
    assert security.checked_subject_ids == (
        initialized.frontier.checkpoint.safety_state.security_observation.checked_subject_ids
    )
    assert initialized.publisher.load_reconciled() == initialized.frontier
    active.close()


def test_initialization_rejects_invalid_objective_before_publication(
    resolver_case,
    tmp_path,
):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    active = (
        StarterWorkspaceBuilder(resolver_case["resolver"].settings)
        .build(
            resolved,
            (tmp_path / "invalid-objective").absolute(),
            run_id="run-initialization-invalid-objective",
            campaign_id="campaign-initialization-invalid-objective",
        )
        .activate()
    )
    security = _RecordingSecurityAuthority(active.bootstrap_pin)

    with pytest.raises(RunStateInitializationError, match="exact configured"):
        initialize_run_state(
            active_workspace=active,
            launch_settings=resolver_case["resolver"].settings.launch,
            security_authority=security,
            objective_direction="sideways",
        )

    assert security.call_count == 0
    assert not (
        active.run_root
        / active.bootstrap_pin.installation_receipt.layout.run_checkpoint_relative_path
    ).exists()
    active.close()
