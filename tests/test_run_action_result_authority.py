from __future__ import annotations

from kapso.cross_run.launch.run_action_result_authority import (
    run_action_terminal_result_evidence_matches,
)
from test_run_action_barrier_contracts import _resolved_graph
from test_run_action_release_contracts import (
    _activation_event,
    _release_adoption_for_event,
    _security_observation,
)
from test_run_action_supervisor_contracts import (
    _activation_revalidation_receipt,
    _claim,
    _prepared_execution,
    _remint_contract,
    _result_capture_receipt,
    _spawn_commit,
    _terminal_observation,
)


def _result_graph():
    security = _security_observation()
    prepared = _prepared_execution(
        claim=_claim(security_observation_id=security.observation_id)
    )
    spawn = _spawn_commit(prepared)
    activation = _activation_revalidation_receipt(prepared, spawn)
    activation_event = _activation_event(
        _resolved_graph(prepared=prepared, activation=activation)
    )
    adoption = _release_adoption_for_event(activation_event, security)
    terminal = _terminal_observation(prepared, spawn, adoption)
    payload = b'{"provider":"complete"}'
    capture = _result_capture_receipt(
        prepared,
        activation,
        terminal,
        payload,
    )
    return activation, adoption, terminal, capture


def test_terminal_result_authority_joins_the_exact_released_occurrence():
    activation, adoption, terminal, capture = _result_graph()

    assert run_action_terminal_result_evidence_matches(
        terminal,
        capture,
        activation,
        adoption,
    )


def test_terminal_result_authority_rejects_release_terminal_and_capture_splices():
    activation, adoption, terminal, capture = _result_graph()
    security = _security_observation(generation=4)
    foreign_prepared = _prepared_execution(
        claim=_claim(security_observation_id=security.observation_id),
        inode_offset=9,
    )
    foreign_spawn = _spawn_commit(
        foreign_prepared,
        invocation_nonce="2" * 32,
    )
    foreign_activation = _activation_revalidation_receipt(
        foreign_prepared,
        foreign_spawn,
    )
    foreign_event = _activation_event(
        _resolved_graph(
            prepared=foreign_prepared,
            activation=foreign_activation,
        )
    )
    foreign_adoption = _release_adoption_for_event(foreign_event, security)

    assert not run_action_terminal_result_evidence_matches(
        terminal,
        capture,
        activation,
        foreign_adoption,
    )
    assert not run_action_terminal_result_evidence_matches(
        _remint_contract(
            terminal,
            started_at="2026-07-25T01:02:02.123456789Z",
        ),
        capture,
        activation,
        adoption,
    )
    assert not run_action_terminal_result_evidence_matches(
        _remint_contract(terminal, exit_code=17),
        capture,
        activation,
        adoption,
    )
    assert not run_action_terminal_result_evidence_matches(
        terminal,
        _remint_contract(capture, parent_inode=capture.parent_inode + 1),
        activation,
        adoption,
    )
