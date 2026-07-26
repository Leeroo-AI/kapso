"""Capability and outcome authority tests for provider-termination recovery."""

from __future__ import annotations

import pytest

import kapso.cross_run.launch.run_action_recovery as recovery
from kapso.cross_run.launch.run_action_clock import _SystemRunActionClock
from kapso.cross_run.launch.run_action_recovery import (
    _RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    RunActionCommittedContinuationCapability,
    RunActionCommittedSpawnObservation,
    RunActionCommittedSpawnState,
    RunActionContinuationOutcome,
    RunActionContinuationState,
    RunActionProviderResult,
    RunActionRecoveryCoordinator,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    run_action_pre_release_main_loss_observation_token,
    run_action_pre_release_main_terminal_observation_token,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
)
from test_run_action_release_contracts import (
    _security_observation as _release_security_observation,
)
from test_run_action_result_authority import _result_graph
from test_run_action_supervisor_contracts import (
    _remint_contract,
    _result_capture_receipt,
    _terminal_observation,
)
from test_run_action_terminal_inspection import (
    _configured_settings,
    _inspection_context,
    _SecurityAuthority,
)
from test_run_action_termination_contracts import (
    _pre_release_loss,
    _termination_graph,
)


class _PublicationFenceSource:
    def __init__(self):
        self.current_checks = 0
        self.closed = False

    def require_current(self):
        if self.closed:
            raise AssertionError("test publication fence source is closed")
        self.current_checks += 1

    def close(self):
        if self.closed:
            raise AssertionError("test publication fence source closed twice")
        self.closed = True


def _publication_fence():
    source = _PublicationFenceSource()
    fence = recovery.RunActionProviderTerminationPublicationFence(
        source=source,
        _authority=(
            recovery._RUN_ACTION_PROVIDER_TERMINATION_PUBLICATION_FENCE_AUTHORITY
        ),
    )
    return fence, source


def _capability(query, state, token):
    return RunActionCommittedContinuationCapability(
        query=query,
        observation=RunActionCommittedSpawnObservation(
            state=state,
            observation_token=token,
        ),
        required_security_observation=_release_security_observation(),
        security_authority=_SecurityAuthority(),
        credential_validity_authority=None,
        release_clock=_SystemRunActionClock(),
        _authority=_RUN_ACTION_COMMITTED_CONTINUATION_AUTHORITY,
    )


def _released_termination_case():
    query = _inspection_context(_configured_settings()[0])[0]
    terminal = _remint_contract(
        _terminal_observation(
            query.prepared_execution,
            query.spawn_commit,
            query.workload_release_adoption,
        ),
        exit_code=137,
        oom_killed=True,
    )
    receipt = RunActionProviderTerminationReceipt.mint(
        disposition=RunActionProviderTerminationDisposition.FAILED,
        reason=RunActionProviderTerminationReason.OOM,
        activation_event_id=query.activation_event.event_id,
        workload_release_adoption=query.workload_release_adoption,
        terminal_observation=terminal,
        timeout_directive_publication=None,
        empty_result_capture_receipt=None,
        pre_release_main_loss_observation=None,
    )
    capability = _capability(
        query,
        RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
        terminal.complete_inspection_digest,
    )
    return capability, query, terminal, receipt


def _register_terminal(capability, terminal):
    query, observation_token = capability._take_terminal_inspection_authority(
        _authority=recovery._RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY,
    )
    assert terminal.complete_inspection_digest == observation_token
    capability._complete_terminal_inspection(
        terminal,
        _authority=recovery._RUN_ACTION_TERMINAL_INSPECTION_AUTHORITY,
    )
    return query


def _register_termination(capability, receipt, publication_fence=None):
    query, retained_terminal, loss_observation_id = (
        capability._take_provider_termination_authority(
            _authority=recovery._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
        )
    )
    capability._complete_provider_termination(
        receipt,
        publication_fence,
        _authority=recovery._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
    )
    return query, retained_terminal, loss_observation_id


def test_registered_released_termination_is_the_only_admitted_receipt():
    capability, query, terminal, receipt = _released_termination_case()

    class _TrustedTerminationAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            assert _register_terminal(active_capability, terminal) == query
            registered = _register_termination(active_capability, receipt)
            assert registered == (query, terminal, None)
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PROVIDER_TERMINATED,
                result=None,
                provider_termination_receipt=receipt,
                timeout_directive_publication=None,
            )

    outcome = capability._invoke_once(_TrustedTerminationAdapter())

    assert outcome.provider_termination_receipt == receipt


def test_provider_termination_registration_excludes_result_capture():
    query = _inspection_context(_configured_settings()[0])[0]
    terminal = _terminal_observation(
        query.prepared_execution,
        query.spawn_commit,
        query.workload_release_adoption,
    )
    receipt = RunActionProviderTerminationReceipt.mint(
        disposition=RunActionProviderTerminationDisposition.FAILED,
        reason=RunActionProviderTerminationReason.EMPTY_RESULT,
        activation_event_id=query.activation_event.event_id,
        workload_release_adoption=query.workload_release_adoption,
        terminal_observation=terminal,
        timeout_directive_publication=None,
        empty_result_capture_receipt=_result_capture_receipt(
            query.prepared_execution,
            query.activation_revalidation_receipt,
            terminal,
            b"",
        ),
        pre_release_main_loss_observation=None,
    )
    capability = _capability(
        query,
        RunActionCommittedSpawnState.TERMINAL_CONTINUABLE,
        terminal.complete_inspection_digest,
    )

    class _ExclusiveTerminationAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            _register_terminal(active_capability, terminal)
            active_capability._take_provider_termination_authority(
                _authority=recovery._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
            )
            with pytest.raises(
                RunActionRecoveryError,
                match="result capture lacks exact live terminal authority",
            ):
                active_capability._take_result_capture_authority(
                    _authority=recovery._RUN_ACTION_RESULT_CAPTURE_AUTHORITY,
                )
            active_capability._complete_provider_termination(
                receipt,
                _authority=recovery._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PROVIDER_TERMINATED,
                result=None,
                provider_termination_receipt=receipt,
                timeout_directive_publication=None,
            )

    outcome = capability._invoke_once(_ExclusiveTerminationAdapter())

    assert outcome.provider_termination_receipt == receipt


def test_fabricated_receipt_cannot_bypass_private_registration():
    capability, _query, terminal, receipt = _released_termination_case()

    class _FabricatingAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            _register_terminal(active_capability, terminal)
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PROVIDER_TERMINATED,
                result=None,
                provider_termination_receipt=receipt,
                timeout_directive_publication=None,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="trusted outcome registration",
    ):
        capability._invoke_once(_FabricatingAdapter())


def test_registered_termination_cannot_be_discarded_as_pending():
    capability, _query, terminal, receipt = _released_termination_case()

    class _DiscardingAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            _register_terminal(active_capability, terminal)
            _register_termination(active_capability, receipt)
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PENDING,
                result=None,
                provider_termination_receipt=None,
                timeout_directive_publication=None,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="trusted outcome registration",
    ):
        capability._invoke_once(_DiscardingAdapter())


def test_cross_occurrence_receipt_cannot_complete_registration():
    capability, _query, terminal, _receipt = _released_termination_case()
    foreign_receipt = _termination_graph(RunActionProviderTerminationReason.OOM)

    class _SplicingAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            _register_terminal(active_capability, terminal)
            active_capability._take_provider_termination_authority(
                _authority=recovery._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
            )
            active_capability._complete_provider_termination(
                foreign_receipt,
                _authority=recovery._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
            )
            raise AssertionError("cross-occurrence receipt was registered")

    with pytest.raises(
        RunActionRecoveryError,
        match="completion lacks exact live authority",
    ):
        capability._invoke_once(_SplicingAdapter())


def _pre_release_termination_case():
    released_query = _inspection_context(_configured_settings()[0])[0]
    query = type(released_query)(
        preparation_allocation=released_query.preparation_allocation,
        activation_event=released_query.activation_event,
        workload_release_adoption=None,
        timeout_directive_publication=None,
    )
    loss = _pre_release_loss(
        query.activation_revalidation_receipt,
        query.activation_event.event_id,
    )
    receipt = RunActionProviderTerminationReceipt.mint(
        disposition=RunActionProviderTerminationDisposition.FAILED,
        reason=RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS,
        activation_event_id=query.activation_event.event_id,
        workload_release_adoption=None,
        terminal_observation=None,
        timeout_directive_publication=None,
        empty_result_capture_receipt=None,
        pre_release_main_loss_observation=loss,
    )
    capability = _capability(
        query,
        RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE,
        run_action_pre_release_main_loss_observation_token(loss),
    )
    return capability, query, loss, receipt


def test_pre_release_loss_registration_requires_the_exact_observation_token():
    capability, query, loss, receipt = _pre_release_termination_case()

    class _TrustedLossAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            publication_fence, source = _publication_fence()
            registered = _register_termination(
                active_capability,
                receipt,
                publication_fence,
            )
            observation_token = run_action_pre_release_main_loss_observation_token(loss)
            assert registered == (
                query,
                None,
                observation_token,
            )
            _TrustedLossAdapter.source = source
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PROVIDER_TERMINATED,
                result=None,
                provider_termination_receipt=receipt,
                timeout_directive_publication=None,
                provider_termination_publication_fence=publication_fence,
            )

    outcome = capability._invoke_once(_TrustedLossAdapter())

    assert outcome.provider_termination_receipt == receipt
    outcome.provider_termination_publication_fence.close()
    assert _TrustedLossAdapter.source.closed


def test_pre_release_loss_rejects_substituted_publication_fence_and_closes_real():
    capability, _query, _loss, receipt = _pre_release_termination_case()
    registered_fence, registered_source = _publication_fence()
    substituted_fence, substituted_source = _publication_fence()

    class _SubstitutingLossAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            _register_termination(
                active_capability,
                receipt,
                registered_fence,
            )
            return RunActionContinuationOutcome(
                state=RunActionContinuationState.PROVIDER_TERMINATED,
                result=None,
                provider_termination_receipt=receipt,
                timeout_directive_publication=None,
                provider_termination_publication_fence=substituted_fence,
            )

    with pytest.raises(
        RunActionRecoveryError,
        match="trusted termination",
    ):
        capability._invoke_once(_SubstitutingLossAdapter())

    assert registered_source.closed
    assert not substituted_source.closed
    substituted_fence.close()


def test_pre_release_loss_closes_registered_fence_when_adapter_raises():
    capability, _query, _loss, receipt = _pre_release_termination_case()
    registered_fence, registered_source = _publication_fence()

    class _RaisingLossAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            _register_termination(
                active_capability,
                receipt,
                registered_fence,
            )
            raise RuntimeError("adapter abandoned trusted pre-release outcome")

    with pytest.raises(RuntimeError, match="abandoned trusted"):
        capability._invoke_once(_RaisingLossAdapter())

    assert registered_source.closed


def test_result_and_termination_are_an_exact_outcome_xor():
    activation, adoption, terminal, _capture = _result_graph()
    payload = b'{"result":"exclusive"}'
    result = RunActionProviderResult(
        terminal_observation=terminal,
        result_capture_receipt=_result_capture_receipt(
            activation.prepared_execution,
            activation,
            terminal,
            payload,
        ),
        result_payload=payload,
    )
    termination = _termination_graph(RunActionProviderTerminationReason.OOM)

    with pytest.raises(RunActionRecoveryError, match="differs from its state"):
        RunActionContinuationOutcome(
            state=RunActionContinuationState.RESULT_CAPTURED,
            result=result,
            provider_termination_receipt=termination,
            timeout_directive_publication=None,
        )


def test_observation_and_outcome_matrix_is_closed():
    activation, _adoption, terminal, _capture = _result_graph()
    payload = b'{"result":"matrix"}'
    result = RunActionProviderResult(
        terminal_observation=terminal,
        result_capture_receipt=_result_capture_receipt(
            activation.prepared_execution,
            activation,
            terminal,
            payload,
        ),
        result_payload=payload,
    )
    termination = _termination_graph(RunActionProviderTerminationReason.OOM)
    timeout_publication = _termination_graph(
        RunActionProviderTerminationReason.TIMEOUT
    ).timeout_directive_publication
    outcomes = {
        RunActionContinuationState.PENDING: RunActionContinuationOutcome(
            state=RunActionContinuationState.PENDING,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        ),
        RunActionContinuationState.RESULT_CAPTURED: RunActionContinuationOutcome(
            state=RunActionContinuationState.RESULT_CAPTURED,
            result=result,
            provider_termination_receipt=None,
            timeout_directive_publication=None,
        ),
        RunActionContinuationState.TIMEOUT_PUBLISHED: RunActionContinuationOutcome(
            state=RunActionContinuationState.TIMEOUT_PUBLISHED,
            result=None,
            provider_termination_receipt=None,
            timeout_directive_publication=timeout_publication,
        ),
        RunActionContinuationState.PROVIDER_TERMINATED: (
            RunActionContinuationOutcome(
                state=RunActionContinuationState.PROVIDER_TERMINATED,
                result=None,
                provider_termination_receipt=termination,
                timeout_directive_publication=None,
            )
        ),
    }
    loss = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
    ).pre_release_main_loss_observation
    pre_release_terminal = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL
    ).terminal_observation
    observations = {
        RunActionCommittedSpawnState.INERT_CONTINUABLE: "sha256:" + "1" * 64,
        RunActionCommittedSpawnState.RUNNING_CONTINUABLE: "sha256:" + "2" * 64,
        RunActionCommittedSpawnState.TERMINAL_CONTINUABLE: "sha256:" + "3" * 64,
        RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE: (
            run_action_pre_release_main_loss_observation_token(loss)
        ),
        RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE: (
            run_action_pre_release_main_terminal_observation_token(pre_release_terminal)
        ),
        RunActionCommittedSpawnState.UNKNOWN: None,
    }
    admitted = {
        RunActionCommittedSpawnState.INERT_CONTINUABLE: {
            RunActionContinuationState.PENDING,
        },
        RunActionCommittedSpawnState.RUNNING_CONTINUABLE: {
            RunActionContinuationState.PENDING,
            RunActionContinuationState.TIMEOUT_PUBLISHED,
        },
        RunActionCommittedSpawnState.TERMINAL_CONTINUABLE: {
            RunActionContinuationState.PENDING,
            RunActionContinuationState.RESULT_CAPTURED,
            RunActionContinuationState.PROVIDER_TERMINATED,
        },
        RunActionCommittedSpawnState.PRE_RELEASE_MAIN_LOSS_CONTINUABLE: {
            RunActionContinuationState.PENDING,
            RunActionContinuationState.PROVIDER_TERMINATED,
        },
        RunActionCommittedSpawnState.PRE_RELEASE_MAIN_TERMINAL_CONTINUABLE: {
            RunActionContinuationState.PENDING,
            RunActionContinuationState.PROVIDER_TERMINATED,
        },
        RunActionCommittedSpawnState.UNKNOWN: set(),
    }
    for observation_state, token in observations.items():
        observation = RunActionCommittedSpawnObservation(
            state=observation_state,
            observation_token=token,
        )
        for outcome_state, outcome in outcomes.items():
            assert RunActionRecoveryCoordinator._continuation_outcome_allowed(
                observation,
                outcome,
            ) is (outcome_state in admitted[observation_state])


@pytest.mark.parametrize(
    "observation_state",
    (
        RunActionCommittedSpawnState.INERT_CONTINUABLE,
        RunActionCommittedSpawnState.RUNNING_CONTINUABLE,
    ),
)
def test_nonterminal_observation_cannot_consume_termination_authority(
    observation_state,
):
    query = _inspection_context(_configured_settings()[0])[0]
    capability = _capability(
        query,
        observation_state,
        "sha256:" + "4" * 64,
    )

    class _AuthorityConsumingAdapter:
        @staticmethod
        def continue_committed_once(active_capability):
            active_capability._take_provider_termination_authority(
                _authority=recovery._RUN_ACTION_PROVIDER_TERMINATION_AUTHORITY,
            )
            raise AssertionError("nonterminal observation consumed authority")

    with pytest.raises(
        RunActionRecoveryError,
        match="registration lacks exact live authority",
    ):
        capability._invoke_once(_AuthorityConsumingAdapter())
