from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id
from kapso.cross_run.expert.validation_store import (
    ExpertValidationCompareAndSwapError,
    ExpertValidationStore,
)
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
    _release_matrix_fixture,
)
from test_expert_task_evaluation_preflight import (
    _CurrentAuthority,
    _coordinator,
    _current_observation,
    _expert_sources,
)


def _parent_prepared(tmp_path, monkeypatch):
    validation_store, snapshot, prepared_plan = _release_matrix_fixture(
        tmp_path,
        monkeypatch,
        rotate_active_adapter=True,
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    _candidate, parent = _expert_sources(prepared_plan)
    observation = _current_observation(prepared_plan)
    current_authority = _CurrentAuthority((observation, observation))
    coordinator, candidate_reader, parent_provider, adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=parent,
        current_authority=current_authority,
    )
    prepared = coordinator.build(plan_reservation)
    return (
        validation_store,
        snapshot,
        prepared,
        current_authority,
        candidate_reader,
        parent_provider,
        adapter_provider,
    )


def test_parent_task_evaluation_reserves_replays_and_reopens_offline(
    tmp_path,
    monkeypatch,
):
    (
        validation_store,
        snapshot,
        prepared,
        current_authority,
        candidate_reader,
        parent_provider,
        adapter_provider,
    ) = _parent_prepared(tmp_path, monkeypatch)
    provider_call_counts = (
        len(current_authority.calls),
        len(candidate_reader.calls),
        len(parent_provider.calls),
        len(adapter_provider.calls),
    )

    committed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    replayed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    reopened_store = ExpertValidationStore(
        validation_store.root,
        validation_store.state_root,
        validation_store.settings,
        validation_store.reducer,
    )
    reopened = reopened_store.reopen_task_evaluation_reservation(
        reservation_id=committed.reservation.reservation.reservation_id,
        prepared_request=prepared,
    )

    assert committed.replayed is False
    assert replayed.replayed is True
    assert replayed.reservation == committed.reservation
    assert reopened == committed.reservation
    assert committed.reservation.request == prepared.plan_join.request
    assert (
        committed.reservation.current_release_observation
        == prepared.current_release_observation
    )
    assert (
        committed.reservation.reservation.scope_id
        == prepared.plan_join.request.scope_id
    )
    assert committed.reservation.reservation.observed_current_release_id == (
        prepared.plan_join.request.parent_release_id
    )
    assert provider_call_counts == (
        len(current_authority.calls),
        len(candidate_reader.calls),
        len(parent_provider.calls),
        len(adapter_provider.calls),
    )


def test_bootstrap_reservation_persists_authenticated_absence(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared_plan, _active_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    observation = _current_observation(prepared_plan)
    current_authority = _CurrentAuthority((observation, observation))
    coordinator, _candidate_reader, parent_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=None,
        current_authority=current_authority,
    )
    prepared = coordinator.build(plan_reservation)

    committed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )

    assert parent_provider.calls == []
    assert committed.reservation.current_release_observation.release_id is None
    assert (
        committed.reservation.reservation.current_release_observation_id
        == observation.observation_id
    )
    assert committed.reservation.reservation.observed_current_release_id is None
    assert observation.default_branch_head_commit_sha == "a" * 40


def test_identical_replay_preserves_first_admission_observation(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    first = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    observation_values = prepared.current_release_observation.to_dict()
    observation_values.pop("observation_id")
    observation_values["default_branch_head_commit_sha"] = "c" * 40
    later_observation = type(prepared.current_release_observation).mint(
        **observation_values,
    )
    later_prepared = replace(
        prepared,
        current_release_observation=later_observation,
    )

    replayed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=later_prepared,
    )

    assert replayed.replayed is True
    assert replayed.reservation == first.reservation
    assert (
        replayed.reservation.current_release_observation
        == prepared.current_release_observation
    )
    assert replayed.reservation.current_release_observation != later_observation


def test_missing_persisted_observation_fails_journal_replay(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    committed = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    )
    observation_id = committed.reservation.current_release_observation.observation_id
    namespace, digest = observation_id.split(":sha256:", 1)
    observation_path = validation_store.object_root / namespace / f"{digest}.json"

    observation_path.unlink()

    with pytest.raises(ValueError):
        validation_store.snapshot(prepared.plan_join.request.candidate_id)


def test_concurrent_identical_task_evaluations_bind_one_alias(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )

    def reserve(_position):
        return validation_store.reserve_task_evaluation(
            expected_transition_id=snapshot.transition.transition_id,
            prepared_request=prepared,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(reserve, range(8)))

    assert sum(not result.replayed for result in results) == 1
    assert (
        len({result.reservation.reservation.reservation_id for result in results}) == 1
    )


def test_task_reservation_rejects_validation_head_advance(
    tmp_path,
    monkeypatch,
):
    validation_store, snapshot, prepared, *_providers = _parent_prepared(
        tmp_path,
        monkeypatch,
    )
    validation_store.reducer.current_release_provider.release_id = content_id(
        "expert-base-release",
        {"generation": "successor"},
    )
    validation_store.publish_current_release_authority_invalidation(
        candidate_id=prepared.plan_join.request.candidate_id,
        expected_validation_state_id=snapshot.state.validation_state_id,
    )

    with pytest.raises(ExpertValidationCompareAndSwapError):
        validation_store.reserve_task_evaluation(
            expected_transition_id=snapshot.transition.transition_id,
            prepared_request=prepared,
        )
