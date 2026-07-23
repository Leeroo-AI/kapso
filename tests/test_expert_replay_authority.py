from __future__ import annotations

from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.expert.replay_authority import (
    ExpertSourceReplayFreshAuthorityCoordinator,
    ExpertSourceReplayFreshAuthorityError,
    SourceReplayCurrentReleaseObservation,
)
from kapso.cross_run.expert.replay_execution_store import (
    ExpertSourceReplayExecutionStore,
)
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalStoreError,
)
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayExecutionProviderRegistry,
    expert_source_replay_execution_provider_key,
)
from kapso.cross_run.expert.replay_protocol_contracts import (
    ExpertSourceReplayInvocationAllocation,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStoreError
from security_denylist_fixtures import matched_security_revocations
from test_expert_source_replay_request import _prepared, _request_fixture


class _ReservationAuthority:
    def __init__(self, store, calls):
        self.store = store
        self.calls = calls

    def reopen_source_replay_reservation(self, **request):
        self.calls.append("reservation")
        return self.store.reopen_source_replay_reservation(**request)


class _CurrentAuthority:
    def __init__(self, observation, calls):
        self.observation = observation
        self.calls = calls

    def current_release_observation(self, scope_id):
        self.calls.append("current")
        assert scope_id == self.observation.scope_id
        return self.observation


class _AdapterAuthority:
    def __init__(self, provider, calls):
        self.provider = provider
        self.calls = calls

    def resolve_exact(self, **request):
        self.calls.append("adapter")
        return self.provider.resolve_exact(**request)


class _DenylistAuthority:
    def __init__(self, calls, *, denied=False, substitute_checked=False):
        self.calls = calls
        self.denied = denied
        self.substitute_checked = substitute_checked
        self.checked_subject_ids = None

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.calls.append("denylist")
        self.checked_subject_ids = checked_subject_ids
        observed_subjects = (
            checked_subject_ids[:-1] if self.substitute_checked else checked_subject_ids
        )
        matched_subject_ids = (observed_subjects[0],) if self.denied else ()
        return SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=tree_or_blob_digest(b"scope binding"),
            snapshot_id=content_id(
                "security-denylist-snapshot",
                {"generation": 7},
            ),
            generation=7,
            publication_id=content_id(
                "github-publication",
                {"security_denylist_generation": 7},
            ),
            repository_full_name="Leeroo-AI/kapso-security",
            repository_node_id="security_repo_node",
            pointer_digest=tree_or_blob_digest(b"security CURRENT"),
            authority_commit_sha="b" * 40,
            release_attestation_ref="attestations/security-denylist",
            checked_subject_ids=observed_subjects,
            matched_revocations=matched_security_revocations(matched_subject_ids),
        )


@pytest.fixture
def authority(tmp_path):
    fixture = _request_fixture(tmp_path)
    prepared = _prepared(fixture)
    snapshot = fixture.validation_store.snapshot(prepared.request.candidate_id)
    assert snapshot is not None
    committed = fixture.validation_store.reserve_source_replay(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    execution_store = ExpertSourceReplayExecutionStore(
        (fixture.validation_store.root / "source-replay-executions").resolve(),
        fixture.validation_store.root,
        prepared.settings.policy,
    )
    parent = prepared.parent.release_manifest
    current_observation = SourceReplayCurrentReleaseObservation.mint(
        scope_id=parent.scope_id,
        release_id=parent.release_id,
        publication_id=content_id(
            "github-publication",
            {"release_id": parent.release_id},
        ),
        repository_full_name="Leeroo-AI/kapso-expert",
        repository_node_id="expert_repo_node",
        current_pointer_digest=tree_or_blob_digest(b"CURRENT"),
        current_pointer_commit_sha="a" * 40,
        validation_closure_ids=(),
    )
    calls = []
    denylist = _DenylistAuthority(calls)
    coordinator = ExpertSourceReplayFreshAuthorityCoordinator(
        _ReservationAuthority(fixture.validation_store, calls),
        execution_store,
        _CurrentAuthority(current_observation, calls),
        _AdapterAuthority(fixture.adapter_provider, calls),
        denylist,
    )
    with execution_store.reservation_session(
        reservation=committed.reservation,
        prepared_request=prepared,
    ) as session:
        invocation_permit = session.allocate_expected_leg()
        provider = SimpleNamespace(
            dispatch_key=expert_source_replay_execution_provider_key(prepared.cases[0]),
            execute_leg=lambda _invocation: None,
            cleanup_interrupted=lambda _provider_handle: None,
        )
        resolved_case = ExpertSourceReplayExecutionProviderRegistry(
            (provider,)
        ).resolve_all(prepared)[0]
        yield SimpleNamespace(
            fixture=fixture,
            prepared=prepared,
            reservation=committed.reservation,
            execution_store=execution_store,
            invocation_permit=invocation_permit,
            allocation=invocation_permit.require_current_allocation(execution_store),
            current_observation=current_observation,
            calls=calls,
            denylist=denylist,
            coordinator=coordinator,
            resolved_case=resolved_case,
            session=session,
        )


def test_fresh_spawn_authority_double_reopens_around_all_external_checks(authority):

    execution = authority.coordinator.commit_spawn(
        prepared_request=authority.prepared,
        reservation_id=authority.reservation.reservation_id,
        invocation_permit=authority.invocation_permit,
        resolved_case=authority.resolved_case,
    )
    spawn_event = authority.session.events[-1]
    fence = spawn_event.spawn_authority_fence

    assert authority.calls == [
        "reservation",
        "current",
        "adapter",
        "denylist",
        "reservation",
    ]
    assert fence.reservation_id == authority.reservation.reservation_id
    assert fence.execution_request_id == authority.prepared.request.execution_request_id
    assert fence.invocation_allocation == authority.allocation
    assert fence.security_subject_ids == authority.denylist.checked_subject_ids
    assert fence.current_release_observation == authority.current_observation
    assert len(fence.task_adapter_trust_observations) == 1
    adapter_observation = fence.task_adapter_trust_observations[0]
    assert adapter_observation.verifier_authority_subject_id in (
        authority.denylist.checked_subject_ids
    )
    assert set(authority.prepared.request.exact_dependency_ids).issubset(
        authority.denylist.checked_subject_ids
    )
    assert type(fence).from_json_bytes(fence.to_json_bytes()) == fence
    assert spawn_event.task_evaluator_request.opaque_invocation_id == (
        authority.allocation.opaque_invocation_id
    )
    assert execution is authority.session._spawn_permit


def test_fresh_spawn_authority_rejects_a_changed_current_before_adapter_work(
    authority,
):
    changed_current = SourceReplayCurrentReleaseObservation.mint(
        scope_id=authority.current_observation.scope_id,
        release_id=content_id("expert-base-release", {"changed": True}),
        publication_id=content_id("github-publication", {"changed": True}),
        repository_full_name=(authority.current_observation.repository_full_name),
        repository_node_id=authority.current_observation.repository_node_id,
        current_pointer_digest=tree_or_blob_digest(b"changed CURRENT"),
        current_pointer_commit_sha="b" * 40,
        validation_closure_ids=(),
    )
    authority.coordinator.current_release_authority.observation = changed_current

    with pytest.raises(ExpertSourceReplayFreshAuthorityError, match="current release"):
        authority.coordinator.commit_spawn(
            prepared_request=authority.prepared,
            reservation_id=authority.reservation.reservation_id,
            invocation_permit=authority.invocation_permit,
            resolved_case=authority.resolved_case,
        )

    assert authority.calls == ["reservation", "current"]


@pytest.mark.parametrize("failure", ("denied", "substituted"))
def test_fresh_spawn_authority_rejects_nonexact_or_denied_security_state(
    authority,
    failure,
):
    replacement = _DenylistAuthority(
        authority.calls,
        denied=failure == "denied",
        substitute_checked=failure == "substituted",
    )
    authority.coordinator.security_denylist_authority = replacement

    with pytest.raises(ExpertSourceReplayFreshAuthorityError, match="denylist"):
        authority.coordinator.commit_spawn(
            prepared_request=authority.prepared,
            reservation_id=authority.reservation.reservation_id,
            invocation_permit=authority.invocation_permit,
            resolved_case=authority.resolved_case,
        )

    assert authority.calls == ["reservation", "current", "adapter", "denylist"]


def test_fresh_spawn_authority_rejects_a_caller_minted_allocation(
    authority,
):
    request_case = authority.prepared.request.cases[0]
    other_leg_id = next(
        leg.execution_leg_id
        for leg in (request_case.control_leg, request_case.candidate_leg)
        if leg.execution_leg_id != authority.allocation.execution_leg_id
    )
    caller_allocation = ExpertSourceReplayInvocationAllocation(
        reservation_id=authority.reservation.reservation_id,
        execution_case_id=authority.allocation.execution_case_id,
        execution_leg_id=other_leg_id,
        invocation_nonce="fedcba9876543210fedcba9876543210",
    )

    with pytest.raises(
        ExpertSourceReplayFreshAuthorityError,
        match="live invocation allocation permit",
    ):
        authority.coordinator.commit_spawn(
            prepared_request=authority.prepared,
            reservation_id=authority.reservation.reservation_id,
            invocation_permit=caller_allocation,
            resolved_case=authority.resolved_case,
        )

    assert authority.calls == []


def test_fresh_spawn_authority_rejects_an_alternate_store_permit(authority):
    alternate_store = ExpertSourceReplayExecutionStore(
        (
            authority.fixture.validation_store.root
            / "alternate-source-replay-executions"
        ).resolve(),
        authority.fixture.validation_store.root,
        authority.prepared.settings.policy,
    )
    with alternate_store.reservation_session(
        reservation=authority.reservation,
        prepared_request=authority.prepared,
    ) as session:
        alternate_permit = session.allocate_expected_leg()
        with pytest.raises(
            ExecutionJournalStoreError,
            match="canonical live store lock",
        ):
            authority.coordinator.commit_spawn(
                prepared_request=authority.prepared,
                reservation_id=authority.reservation.reservation_id,
                invocation_permit=alternate_permit,
                resolved_case=authority.resolved_case,
            )

    assert authority.calls == []


def test_fresh_spawn_authority_propagates_revoked_adapter_trust(authority):

    def revoked_adapter(**_request):
        authority.calls.append("adapter")
        raise RuntimeError("historical verifier revoked")

    authority.coordinator.task_adapter_authority.resolve_exact = revoked_adapter

    with pytest.raises(RuntimeError, match="verifier revoked"):
        authority.coordinator.commit_spawn(
            prepared_request=authority.prepared,
            reservation_id=authority.reservation.reservation_id,
            invocation_permit=authority.invocation_permit,
            resolved_case=authority.resolved_case,
        )

    assert authority.calls == ["reservation", "current", "adapter"]


def test_fresh_spawn_authority_second_reopen_rejects_an_advanced_head(authority):
    original_observe = authority.denylist.observe_exact

    def advance_head(**request):
        observation = original_observe(**request)
        authority.fixture.current_release_provider.release_id = content_id(
            "expert-base-release",
            {"changed_during_spawn_authority": True},
        )
        authority.fixture.validation_store.publish_current_release_authority_invalidation(
            candidate_id=authority.prepared.request.candidate_id,
            expected_validation_state_id=(authority.reservation.authorization_state_id),
        )
        return observation

    authority.denylist.observe_exact = advance_head

    with pytest.raises(ExpertValidationStoreError, match="current head"):
        authority.coordinator.commit_spawn(
            prepared_request=authority.prepared,
            reservation_id=authority.reservation.reservation_id,
            invocation_permit=authority.invocation_permit,
            resolved_case=authority.resolved_case,
        )

    assert authority.calls == [
        "reservation",
        "current",
        "adapter",
        "denylist",
        "reservation",
    ]
