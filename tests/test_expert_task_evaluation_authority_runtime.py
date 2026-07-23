from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.expert.task_evaluation_authority import (
    TaskEvaluationFreshAuthorityCoordinator,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationAuthorityError,
)
from kapso.cross_run.expert.task_evaluation_execution import (
    TaskEvaluationExecutionProviderRegistry,
    project_prepared_task_evaluation_cases,
)
from kapso.cross_run.expert.task_evaluation_execution_store import (
    ExpertTaskEvaluationExecutionStore,
    TaskEvaluationSpawnPermit,
)
from kapso.cross_run.expert.validation_store import ExpertValidationStoreError
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from test_expert_release_matrix_reservation import (
    _bootstrap_release_matrix_fixture,
)
from test_expert_task_evaluation_preflight import (
    _CurrentAuthority,
    _coordinator,
    _current_observation,
)
from test_expert_task_evaluation_reservation import _parent_prepared


class _ReservationAuthority:
    def __init__(self, validation_store, calls):
        self.validation_store = validation_store
        self.calls = calls

    def reopen_task_evaluation_reservation(self, **request):
        self.calls.append("reservation")
        return self.validation_store.reopen_task_evaluation_reservation(**request)


class _CurrentReleaseAuthority:
    def __init__(self, observations, calls):
        self.observations = observations
        self.calls = calls

    def observe_task_evaluation_current(self, scope_id):
        self.calls.append("current")
        observation = self.observations[self.calls.count("current") - 1]
        assert observation.scope_id == scope_id
        return observation


class _AdapterAuthority:
    def __init__(self, adapters, calls):
        self.adapters = {
            (
                adapter.manifest.task_adapter_manifest_id,
                adapter.verification_receipt.verification_receipt_id,
            ): adapter
            for adapter in adapters
        }
        self.calls = calls

    def resolve_exact(
        self,
        *,
        task_adapter_manifest_id,
        verification_receipt_id,
    ):
        self.calls.append(f"adapter:{task_adapter_manifest_id}")
        return self.adapters[(task_adapter_manifest_id, verification_receipt_id)]


class _DenylistAuthority:
    def __init__(self, calls, *, denied=False, callback=None):
        self.calls = calls
        self.denied = denied
        self.callback = callback
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
        denied_subject_ids = (checked_subject_ids[0],) if self.denied else ()
        observation = SecurityDenylistObservation.mint(
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
            checked_subject_ids=checked_subject_ids,
            denied_subject_ids=denied_subject_ids,
        )
        if self.callback is not None:
            self.callback()
        return observation


class _ExecutionProvider:
    def __init__(self, dispatch_key):
        self.dispatch_key = dispatch_key
        self.execution_calls = []
        self.cleanup_calls = []

    def require_supported_execution(self, _requirements):
        return None

    def execute_leg(self, invocation):
        self.execution_calls.append(invocation)
        raise AssertionError("fresh authority must not execute the provider")

    def cleanup_interrupted(self, provider_handle):
        self.cleanup_calls.append(provider_handle)


def _moved_current(observation, head_commit_sha="c" * 40):
    values = observation.to_dict()
    values.pop("observation_id")
    values["default_branch_head_commit_sha"] = head_commit_sha
    return type(observation).mint(**values)


def _appeared_current(absence):
    release_id = content_id("expert-base-release", {"appeared": True})
    return type(absence).mint(
        scope_id=absence.scope_id,
        release_id=release_id,
        publication_id=content_id(
            "github-publication",
            {"release_id": release_id},
        ),
        repository_full_name=absence.repository_full_name,
        repository_node_id=absence.repository_node_id,
        default_branch_head_commit_sha="d" * 40,
        current_pointer_digest=tree_or_blob_digest(b"appeared CURRENT"),
        validation_closure_ids=(
            content_id("expert-validation", {"release_id": release_id}),
        ),
    )


def _bootstrap_prepared(tmp_path, monkeypatch):
    validation_store, snapshot, prepared_plan, _active_provider = (
        _bootstrap_release_matrix_fixture(tmp_path, monkeypatch)
    )
    plan_reservation = validation_store.reserve_release_matrix_plan(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_plan=prepared_plan,
    ).reservation
    admission_observation = _current_observation(prepared_plan)
    preflight, _reader, parent_provider, _adapter_provider = _coordinator(
        validation_store=validation_store,
        prepared_plan=prepared_plan,
        parent=None,
        current_authority=_CurrentAuthority(
            (admission_observation, admission_observation)
        ),
    )
    return (
        validation_store,
        snapshot,
        preflight.build(plan_reservation),
        parent_provider,
    )


def _runtime(tmp_path, monkeypatch, *, bootstrap=False):
    if bootstrap:
        validation_store, snapshot, prepared, parent_provider = _bootstrap_prepared(
            tmp_path, monkeypatch
        )
    else:
        validation_store, snapshot, prepared, *_providers = _parent_prepared(
            tmp_path,
            monkeypatch,
        )
        parent_provider = None
    reservation_snapshot = validation_store.reserve_task_evaluation(
        expected_transition_id=snapshot.transition.transition_id,
        prepared_request=prepared,
    ).reservation
    execution_store = ExpertTaskEvaluationExecutionStore(
        ExpertTaskEvaluationExecutionStore.canonical_root(validation_store.root),
        validation_store.root,
        prepared.plan_join.settings.policy,
    )
    provider_keys = tuple(
        sorted(
            {
                case.provider_key
                for case in project_prepared_task_evaluation_cases(prepared)
            },
            key=lambda key: key.identity,
        )
    )
    providers = tuple(_ExecutionProvider(key) for key in provider_keys)
    provider_registry = TaskEvaluationExecutionProviderRegistry(
        prepared,
        providers,
    )
    return SimpleNamespace(
        validation_store=validation_store,
        prepared=prepared,
        reservation_snapshot=reservation_snapshot,
        execution_store=execution_store,
        provider_registry=provider_registry,
        providers=providers,
        parent_provider=parent_provider,
    )


def _coordinator_for(runtime, calls, current_observations, denylist):
    return TaskEvaluationFreshAuthorityCoordinator(
        reservation_authority=_ReservationAuthority(
            runtime.validation_store,
            calls,
        ),
        execution_store=runtime.execution_store,
        current_release_authority=_CurrentReleaseAuthority(
            current_observations,
            calls,
        ),
        task_adapter_authority=_AdapterAuthority(
            runtime.prepared.adapters,
            calls,
        ),
        security_denylist_authority=denylist,
    )


def _expected_success_calls(prepared):
    return [
        "reservation",
        "current",
        *(
            f"adapter:{adapter.manifest.task_adapter_manifest_id}"
            for adapter in prepared.adapters
        ),
        "denylist",
        "current",
        "reservation",
    ]


def test_fresh_authority_double_observes_and_reopens_before_spawn(
    tmp_path,
    monkeypatch,
):
    runtime = _runtime(tmp_path, monkeypatch)
    calls = []
    fresh_current = _moved_current(runtime.prepared.current_release_observation)
    denylist = _DenylistAuthority(calls)
    coordinator = _coordinator_for(
        runtime,
        calls,
        (fresh_current, fresh_current),
        denylist,
    )

    with runtime.execution_store.reservation_session(
        reservation_snapshot=runtime.reservation_snapshot,
        prepared_request=runtime.prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        spawn_permit = coordinator.commit_spawn(
            prepared_request=runtime.prepared,
            reservation_id=(runtime.reservation_snapshot.reservation.reservation_id),
            invocation_permit=permit,
            provider_registry=runtime.provider_registry,
        )
        spawn_event = session.events[-1]

    assert type(spawn_permit) is TaskEvaluationSpawnPermit
    assert calls == _expected_success_calls(runtime.prepared)
    assert spawn_event.spawn_authority_fence.stable_current_release_observation == (
        fresh_current
    )
    assert spawn_event.spawn_authority_fence.security_subject_ids == (
        denylist.checked_subject_ids
    )
    assert all(not provider.execution_calls for provider in runtime.providers)


def test_fresh_authority_rejects_current_head_drift_after_denylist(
    tmp_path,
    monkeypatch,
):
    runtime = _runtime(tmp_path, monkeypatch)
    calls = []
    current_before = runtime.prepared.current_release_observation
    current_after = _moved_current(current_before)
    denylist = _DenylistAuthority(calls)
    coordinator = _coordinator_for(
        runtime,
        calls,
        (current_before, current_after),
        denylist,
    )

    with runtime.execution_store.reservation_session(
        reservation_snapshot=runtime.reservation_snapshot,
        prepared_request=runtime.prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        with pytest.raises(TaskEvaluationAuthorityError, match="changed"):
            coordinator.commit_spawn(
                prepared_request=runtime.prepared,
                reservation_id=(
                    runtime.reservation_snapshot.reservation.reservation_id
                ),
                invocation_permit=permit,
                provider_registry=runtime.provider_registry,
            )
        assert len(session.events) == 1

    assert calls == _expected_success_calls(runtime.prepared)[:-1]
    assert all(not provider.execution_calls for provider in runtime.providers)


def test_fresh_authority_rejects_denied_subject_before_second_current(
    tmp_path,
    monkeypatch,
):
    runtime = _runtime(tmp_path, monkeypatch)
    calls = []
    current = runtime.prepared.current_release_observation
    denylist = _DenylistAuthority(calls, denied=True)
    coordinator = _coordinator_for(
        runtime,
        calls,
        (current, current),
        denylist,
    )

    with runtime.execution_store.reservation_session(
        reservation_snapshot=runtime.reservation_snapshot,
        prepared_request=runtime.prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        with pytest.raises(TaskEvaluationAuthorityError, match="denylist"):
            coordinator.commit_spawn(
                prepared_request=runtime.prepared,
                reservation_id=(
                    runtime.reservation_snapshot.reservation.reservation_id
                ),
                invocation_permit=permit,
                provider_registry=runtime.provider_registry,
            )
        assert len(session.events) == 1

    assert calls == _expected_success_calls(runtime.prepared)[:-2]


def test_bootstrap_spawn_preserves_stable_authenticated_absence(
    tmp_path,
    monkeypatch,
):
    runtime = _runtime(tmp_path, monkeypatch, bootstrap=True)
    calls = []
    fresh_absence = _moved_current(runtime.prepared.current_release_observation)
    denylist = _DenylistAuthority(calls)
    coordinator = _coordinator_for(
        runtime,
        calls,
        (fresh_absence, fresh_absence),
        denylist,
    )

    with runtime.execution_store.reservation_session(
        reservation_snapshot=runtime.reservation_snapshot,
        prepared_request=runtime.prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        coordinator.commit_spawn(
            prepared_request=runtime.prepared,
            reservation_id=(runtime.reservation_snapshot.reservation.reservation_id),
            invocation_permit=permit,
            provider_registry=runtime.provider_registry,
        )
        fence = session.events[-1].spawn_authority_fence

    assert calls == _expected_success_calls(runtime.prepared)
    assert fence.stable_current_release_observation == fresh_absence
    assert fence.stable_current_release_observation.release_id is None
    assert runtime.parent_provider.calls == []


def test_bootstrap_spawn_rejects_a_release_appearing_before_adapter_work(
    tmp_path,
    monkeypatch,
):
    runtime = _runtime(tmp_path, monkeypatch, bootstrap=True)
    calls = []
    appeared = _appeared_current(runtime.prepared.current_release_observation)
    denylist = _DenylistAuthority(calls)
    coordinator = _coordinator_for(
        runtime,
        calls,
        (appeared, appeared),
        denylist,
    )

    with runtime.execution_store.reservation_session(
        reservation_snapshot=runtime.reservation_snapshot,
        prepared_request=runtime.prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        with pytest.raises(TaskEvaluationAuthorityError, match="current release"):
            coordinator.commit_spawn(
                prepared_request=runtime.prepared,
                reservation_id=(
                    runtime.reservation_snapshot.reservation.reservation_id
                ),
                invocation_permit=permit,
                provider_registry=runtime.provider_registry,
            )
        assert len(session.events) == 1

    assert calls == ["reservation", "current"]
    assert denylist.checked_subject_ids is None
    assert runtime.parent_provider.calls == []


def test_second_reopen_rejects_validation_head_change_without_spawn(
    tmp_path,
    monkeypatch,
):
    runtime = _runtime(tmp_path, monkeypatch)
    calls = []
    current = runtime.prepared.current_release_observation

    def advance_validation_head():
        runtime.validation_store.reducer.current_release_provider.release_id = (
            content_id("expert-base-release", {"advanced": True})
        )
        runtime.validation_store.publish_parent_authority_invalidation(
            candidate_id=runtime.prepared.plan_join.request.candidate_id,
            expected_validation_state_id=(
                runtime.reservation_snapshot.reservation.authorization_state_id
            ),
        )

    denylist = _DenylistAuthority(calls, callback=advance_validation_head)
    coordinator = _coordinator_for(
        runtime,
        calls,
        (current, current),
        denylist,
    )

    with runtime.execution_store.reservation_session(
        reservation_snapshot=runtime.reservation_snapshot,
        prepared_request=runtime.prepared,
    ) as session:
        permit = session.allocate_expected_leg()
        with pytest.raises(ExpertValidationStoreError, match="current head"):
            coordinator.commit_spawn(
                prepared_request=runtime.prepared,
                reservation_id=(
                    runtime.reservation_snapshot.reservation.reservation_id
                ),
                invocation_permit=permit,
                provider_registry=runtime.provider_registry,
            )
        assert len(session.events) == 1

    assert calls == _expected_success_calls(runtime.prepared)
    assert all(not provider.execution_calls for provider in runtime.providers)
