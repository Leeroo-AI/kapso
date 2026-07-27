import copy
import hashlib
import os
import pickle
import signal
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread

import pytest

from kapso.cross_run.launch import run_action_credential_broker as broker_module
from kapso.cross_run.launch.run_action_credential_broker import (
    RunActionCredentialBrokerBackend,
    RunActionCredentialBrokerError,
    RunActionCredentialBrokerRegistry,
    RunActionCredentialIssueResponse,
    RunActionCredentialLeaseStatus,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionCredentialMode,
    run_action_credential_lease_authority_id,
    run_action_credential_lease_request,
)
from test_run_action_supervisor_contracts import (
    _claim,
    _credential_policy,
    _execution_policy,
    _prepared_execution,
    _spawn_commit,
    _volume_authority,
)

_SECRET_CANARY = b"credential-canary-never-persist"
_VALID_UNTIL = 1_900_000_000_000_000_000


class _ReplayCredentialBroker(RunActionCredentialBrokerBackend):
    def __init__(
        self,
        *,
        payload=_SECRET_CANARY,
        valid_until=_VALID_UNTIL,
        response_request_id=None,
        status_request_id=None,
    ):
        super().__init__(
            broker_id="test.credential.broker",
            broker_protocol_version="test.credential.broker.v1",
        )
        self.payload = payload
        self.valid_until = valid_until
        self.response_request_id = response_request_id
        self.status_request_id = status_request_id
        self.issue_calls = []
        self.status_calls = []

    def issue_or_replay_exact(self, request):
        self.issue_calls.append(request)
        return RunActionCredentialIssueResponse(
            credential_lease_request_id=(
                request.credential_lease_request_id
                if self.response_request_id is None
                else self.response_request_id
            ),
            payload=self.payload,
            valid_until_realtime_nanoseconds=self.valid_until,
        )

    def observe_exact(self, request):
        self.status_calls.append(request)
        return RunActionCredentialLeaseStatus.mint(
            credential_lease_request_id=(
                request.credential_lease_request_id
                if self.status_request_id is None
                else self.status_request_id
            ),
            valid_until_realtime_nanoseconds=self.valid_until,
        )


def _credentialed_spawn(*, invocation_nonce="1" * 32):
    prepared = _prepared_execution()
    return prepared, _spawn_commit(prepared, invocation_nonce=invocation_nonce)


def _credential_free_policy():
    return _credential_policy(RunActionCredentialMode.NONE)


def _issue_materialization(registry, prepared, spawn):
    response = registry.issue_or_replay_exact(prepared, spawn)
    return registry.materialize_exact(response, prepared, spawn)


def test_lease_request_and_authority_are_exact_spawn_deterministic() -> None:
    prepared, spawn = _credentialed_spawn()
    first_request = run_action_credential_lease_request(prepared, spawn)
    second_request = run_action_credential_lease_request(prepared, spawn)
    first_authority = run_action_credential_lease_authority_id(prepared, spawn)

    other_spawn = _spawn_commit(prepared, invocation_nonce="2" * 32)
    other_request = run_action_credential_lease_request(prepared, other_spawn)
    other_authority = run_action_credential_lease_authority_id(prepared, other_spawn)

    assert first_request == second_request
    assert first_request.credential_policy == (
        prepared.preparation_claim.execution_policy.credential_policy
    )
    assert first_request.prepared_execution_id == prepared.prepared_execution_id
    assert first_request.spawn_commit_id == spawn.spawn_commit_id
    assert first_request.credential_delivery_slot_id == (
        prepared.credential_delivery_slot.prepared_delivery_slot_id
    )
    assert first_request != other_request
    assert first_authority != other_authority
    assert first_authority.startswith("run-action-credential-lease-authority:sha256:")


def test_registry_issues_one_opaque_single_use_materialization() -> None:
    prepared, spawn = _credentialed_spawn()
    backend = _ReplayCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))

    response = registry.issue_or_replay_exact(prepared, spawn)
    assert _SECRET_CANARY.decode("ascii") not in repr(response)
    materialization = registry.materialize_exact(response, prepared, spawn)
    authority_id, size_bytes, valid_until = (
        broker_module.require_run_action_credential_materialization(
            materialization,
            prepared,
            spawn,
            _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
        )
    )

    assert authority_id == run_action_credential_lease_authority_id(prepared, spawn)
    assert size_bytes == len(_SECRET_CANARY)
    assert valid_until == _VALID_UNTIL
    assert _SECRET_CANARY.decode("ascii") not in repr(materialization)
    assert hashlib.sha256(_SECRET_CANARY).hexdigest() not in repr(materialization)
    assert backend.issue_calls == [run_action_credential_lease_request(prepared, spawn)]

    payload = broker_module.consume_run_action_credential_materialization(
        materialization,
        prepared,
        spawn,
        _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
    )
    assert payload == _SECRET_CANARY
    with pytest.raises(RunActionCredentialBrokerError, match="spent"):
        broker_module.consume_run_action_credential_materialization(
            materialization,
            prepared,
            spawn,
            _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
        )


def test_unconsumed_materialization_can_be_burned_without_secret_repr() -> None:
    prepared, spawn = _credentialed_spawn()
    registry = RunActionCredentialBrokerRegistry((_ReplayCredentialBroker(),))
    materialization = _issue_materialization(registry, prepared, spawn)

    broker_module.burn_run_action_credential_materialization(
        materialization,
        _authority=broker_module._RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
    )
    broker_module.burn_run_action_credential_materialization(
        materialization,
        _authority=broker_module._RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
    )

    with pytest.raises(RunActionCredentialBrokerError, match="spent"):
        broker_module.require_run_action_credential_materialization(
            materialization,
            prepared,
            spawn,
            _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
        )


def test_adapter_cannot_replace_broker_private_materialization_bytes() -> None:
    prepared, spawn = _credentialed_spawn()
    registry = RunActionCredentialBrokerRegistry((_ReplayCredentialBroker(),))
    materialization = _issue_materialization(registry, prepared, spawn)

    with pytest.raises(AttributeError):
        materialization._payload = b"adapter-selected-forgery"
    with pytest.raises(AttributeError):
        materialization._valid_until_realtime_nanoseconds = (
            RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
        )
    with pytest.raises(AttributeError):
        materialization._request = run_action_credential_lease_request(
            prepared,
            spawn,
        )
    with pytest.raises(AttributeError):
        materialization._owner_process_id = os.getpid()
    with pytest.raises(AttributeError):
        materialization._state = "ready"
    assert _SECRET_CANARY.decode("ascii") not in repr(
        broker_module._ISSUED_CREDENTIAL_MATERIALIZATIONS
    )

    assert (
        broker_module.consume_run_action_credential_materialization(
            materialization,
            prepared,
            spawn,
            _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
        )
        == _SECRET_CANARY
    )


def test_materialization_rejects_wrong_spawn_thread_and_process() -> None:
    prepared, spawn = _credentialed_spawn()
    registry = RunActionCredentialBrokerRegistry((_ReplayCredentialBroker(),))
    materialization = _issue_materialization(registry, prepared, spawn)
    other_spawn = _spawn_commit(prepared, invocation_nonce="2" * 32)

    with pytest.raises(RunActionCredentialBrokerError, match="foreign"):
        broker_module.require_run_action_credential_materialization(
            materialization,
            prepared,
            other_spawn,
            _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
        )

    failures = []

    def use_from_foreign_thread():
        with pytest.raises(RunActionCredentialBrokerError, match="foreign"):
            broker_module.require_run_action_credential_materialization(
                materialization,
                prepared,
                spawn,
                _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
            )
        failures.append("rejected")

    thread = Thread(target=use_from_foreign_thread)
    thread.start()
    thread.join()
    assert failures == ["rejected"]

    child = os.fork()
    if child == 0:
        with pytest.raises(RunActionCredentialBrokerError, match="foreign"):
            broker_module.require_run_action_credential_materialization(
                materialization,
                prepared,
                spawn,
                _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
            )
        os._exit(37)
    _child_pid, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 37

    broker_module.burn_run_action_credential_materialization(
        materialization,
        _authority=broker_module._RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
    )


def test_registry_seals_policy_and_backend_method_identity() -> None:
    prepared, spawn = _credentialed_spawn()
    backend = _ReplayCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))
    registry.require_policy(
        prepared.preparation_claim.execution_policy.credential_policy
    )

    backend._broker_id = "substituted.credential.broker"
    with pytest.raises(RunActionCredentialBrokerError, match="changed"):
        registry.issue_or_replay_exact(prepared, spawn)

    missing = RunActionCredentialBrokerRegistry(())
    with pytest.raises(RunActionCredentialBrokerError, match="lacks one registered"):
        missing.require_policy(
            prepared.preparation_claim.execution_policy.credential_policy
        )
    missing.require_policy(_credential_free_policy())


def test_registry_seals_backend_protocol_and_class_methods(monkeypatch) -> None:
    prepared, spawn = _credentialed_spawn()
    backend = _ReplayCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))

    backend._broker_protocol_version = "test.credential.broker.v2"
    with pytest.raises(RunActionCredentialBrokerError, match="changed"):
        registry.issue_or_replay_exact(prepared, spawn)

    backend._broker_protocol_version = "test.credential.broker.v1"
    original_issue = _ReplayCredentialBroker.issue_or_replay_exact

    def substituted_issue(self, request):
        return original_issue(self, request)

    monkeypatch.setattr(
        _ReplayCredentialBroker,
        "issue_or_replay_exact",
        substituted_issue,
    )
    with pytest.raises(RunActionCredentialBrokerError, match="changed"):
        registry.issue_or_replay_exact(prepared, spawn)


@pytest.mark.parametrize(
    ("backend", "message"),
    (
        (
            _ReplayCredentialBroker(payload=b"x" * 4097),
            "payload exceeds",
        ),
        (
            _ReplayCredentialBroker(
                response_request_id=run_action_credential_lease_request(
                    *_credentialed_spawn(invocation_nonce="2" * 32)
                ).credential_lease_request_id
            ),
            "another issue response",
        ),
    ),
)
def test_registry_rejects_oversized_or_substituted_issue_response(
    backend,
    message,
) -> None:
    prepared, spawn = _credentialed_spawn()
    registry = RunActionCredentialBrokerRegistry((backend,))

    with pytest.raises(RunActionCredentialBrokerError, match=message):
        registry.issue_or_replay_exact(prepared, spawn)


@pytest.mark.parametrize(
    ("payload", "use_foreign_request", "message"),
    (
        (b"x" * 4097, False, "payload exceeds"),
        (_SECRET_CANARY, True, "another issue response"),
    ),
)
def test_registry_burns_rejected_backend_response_bytes(
    payload,
    use_foreign_request,
    message,
) -> None:
    prepared, spawn = _credentialed_spawn()
    foreign_request = run_action_credential_lease_request(
        *_credentialed_spawn(invocation_nonce="2" * 32)
    )

    class RetainingCredentialBroker(_ReplayCredentialBroker):
        def __init__(self):
            super().__init__(payload=payload)
            self.response = None

        def issue_or_replay_exact(self, request):
            self.issue_calls.append(request)
            self.response = RunActionCredentialIssueResponse(
                credential_lease_request_id=(
                    foreign_request.credential_lease_request_id
                    if use_foreign_request
                    else request.credential_lease_request_id
                ),
                payload=self.payload,
                valid_until_realtime_nanoseconds=self.valid_until,
            )
            return self.response

    backend = RetainingCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))

    with pytest.raises(RunActionCredentialBrokerError, match=message):
        registry.issue_or_replay_exact(prepared, spawn)

    assert backend.response._state == "spent"
    assert backend.response._payload is None


def test_retained_backend_response_cannot_rewrite_sealed_credential_bytes() -> None:
    prepared, spawn = _credentialed_spawn()
    foreign_request = run_action_credential_lease_request(
        *_credentialed_spawn(invocation_nonce="2" * 32)
    )

    class RetainingCredentialBroker(_ReplayCredentialBroker):
        def __init__(self):
            super().__init__()
            self.response = None

        def issue_or_replay_exact(self, request):
            self.response = super().issue_or_replay_exact(request)
            return self.response

    backend = RetainingCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))
    response = registry.issue_or_replay_exact(prepared, spawn)

    backend.response._payload = b"post-seal-backend-forgery"
    backend.response._valid_until_realtime_nanoseconds = (
        RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
    )
    backend.response._credential_lease_request_id = (
        foreign_request.credential_lease_request_id
    )
    backend.response._owner_process_id = 1
    backend.response._owner_thread_id = 1
    backend.response._state = "backend_response"
    assert response.valid_until_realtime_nanoseconds == _VALID_UNTIL
    materialization = registry.materialize_exact(response, prepared, spawn)

    authority_id, size_bytes, valid_until = (
        broker_module.require_run_action_credential_materialization(
            materialization,
            prepared,
            spawn,
            _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
        )
    )
    assert authority_id == run_action_credential_lease_authority_id(prepared, spawn)
    assert size_bytes == len(_SECRET_CANARY)
    assert valid_until == _VALID_UNTIL
    assert (
        broker_module.consume_run_action_credential_materialization(
            materialization,
            prepared,
            spawn,
            _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
        )
        == _SECRET_CANARY
    )


def test_registry_burns_response_when_backend_changes_during_issue() -> None:
    prepared, spawn = _credentialed_spawn()

    class SelfMutatingCredentialBroker(_ReplayCredentialBroker):
        def __init__(self):
            super().__init__()
            self.response = None

        def issue_or_replay_exact(self, request):
            self.response = super().issue_or_replay_exact(request)
            self._broker_protocol_version = "test.credential.broker.v2"
            return self.response

    backend = SelfMutatingCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))

    with pytest.raises(RunActionCredentialBrokerError, match="changed"):
        registry.issue_or_replay_exact(prepared, spawn)

    assert backend.response._state == "spent"
    assert backend.response._payload is None


def test_retained_backend_cannot_block_private_response_burn() -> None:
    prepared, spawn = _credentialed_spawn()

    class RetainingCredentialBroker(_ReplayCredentialBroker):
        def __init__(self):
            super().__init__()
            self.response = None

        def issue_or_replay_exact(self, request):
            self.response = super().issue_or_replay_exact(request)
            return self.response

    backend = RetainingCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))
    response = registry.issue_or_replay_exact(prepared, spawn)
    response._owner_process_id = 1
    response._owner_thread_id = 1
    response._state = "backend_response"
    response._payload = b"post-seal-forgery"

    registry.burn_issue_response_exact(response)

    issuance = broker_module._ISSUED_CREDENTIAL_RESPONSES[response]
    assert issuance.state == "spent"
    assert issuance.payload is None
    assert response._state == "spent"
    assert response._payload is None
    with pytest.raises(RunActionCredentialBrokerError, match="spent"):
        response.valid_until_realtime_nanoseconds


def test_registry_clears_malformed_unsealed_backend_response() -> None:
    prepared, spawn = _credentialed_spawn()

    class MutatingCredentialBroker(_ReplayCredentialBroker):
        def __init__(self):
            super().__init__()
            self.response = None

        def issue_or_replay_exact(self, request):
            self.response = super().issue_or_replay_exact(request)
            self.response._owner_process_id = 1
            self.response._owner_thread_id = 1
            self.response._state = "forged"
            return self.response

    backend = MutatingCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))

    with pytest.raises(RunActionCredentialBrokerError, match="broker authority"):
        registry.issue_or_replay_exact(prepared, spawn)

    assert backend.response._state == "spent"
    assert backend.response._payload is None


def test_backend_cannot_mutate_coordinator_lease_request_graph() -> None:
    prepared, spawn = _credentialed_spawn()
    original_policy = prepared.preparation_claim.execution_policy.credential_policy

    class RequestMutatingCredentialBroker(_ReplayCredentialBroker):
        def __init__(self):
            super().__init__()
            self.response = None

        def issue_or_replay_exact(self, request):
            object.__setattr__(
                request,
                "credential_policy",
                _credential_free_policy(),
            )
            self.response = RunActionCredentialIssueResponse(
                credential_lease_request_id=request.credential_lease_request_id,
                payload=self.payload,
                valid_until_realtime_nanoseconds=self.valid_until,
            )
            return self.response

    backend = RequestMutatingCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))

    with pytest.raises(RunActionCredentialBrokerError, match="changed its lease"):
        registry.issue_or_replay_exact(prepared, spawn)

    assert (
        prepared.preparation_claim.execution_policy.credential_policy == original_policy
    )
    assert backend.response._payload is None
    assert backend.response._state == "spent"


def test_backend_status_is_canonical_snapshot_not_retained_alias() -> None:
    prepared, spawn = _credentialed_spawn()

    class StatusRetainingCredentialBroker(_ReplayCredentialBroker):
        def __init__(self):
            super().__init__()
            self.status = None

        def observe_exact(self, request):
            self.status = super().observe_exact(request)
            return self.status

    backend = StatusRetainingCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))
    status = registry.observe_exact(
        prepared_execution=prepared,
        spawn_commit=spawn,
        activated_credential_file_observation_id=(
            "run-action-activated-file-observation:sha256:" + "a" * 64
        ),
    )

    assert status is not backend.status
    object.__setattr__(
        backend.status,
        "valid_until_realtime_nanoseconds",
        1,
    )
    assert status.valid_until_realtime_nanoseconds == _VALID_UNTIL


def test_registry_observes_same_request_without_secret_material() -> None:
    prepared, spawn = _credentialed_spawn()
    backend = _ReplayCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))
    activated_file_id = "run-action-activated-file-observation:sha256:" + "a" * 64

    status = registry.observe_exact(
        prepared_execution=prepared,
        spawn_commit=spawn,
        activated_credential_file_observation_id=activated_file_id,
    )

    request = run_action_credential_lease_request(prepared, spawn)
    assert backend.status_calls == [request]
    assert status.credential_lease_request_id == request.credential_lease_request_id
    assert status.valid_until_realtime_nanoseconds == _VALID_UNTIL
    assert _SECRET_CANARY not in status.to_json_bytes()
    assert hashlib.sha256(_SECRET_CANARY).hexdigest().encode("ascii") not in (
        status.to_json_bytes()
    )


def test_registry_rejects_status_for_another_exact_request() -> None:
    prepared, spawn = _credentialed_spawn()
    other_request = run_action_credential_lease_request(
        *_credentialed_spawn(invocation_nonce="2" * 32)
    )
    registry = RunActionCredentialBrokerRegistry(
        (
            _ReplayCredentialBroker(
                status_request_id=other_request.credential_lease_request_id
            ),
        )
    )

    with pytest.raises(
        RunActionCredentialBrokerError,
        match="another lease status",
    ):
        registry.observe_exact(
            prepared_execution=prepared,
            spawn_commit=spawn,
            activated_credential_file_observation_id=(
                "run-action-activated-file-observation:sha256:" + "a" * 64
            ),
        )


def test_concurrent_duplicate_issue_is_serialized_by_registry() -> None:
    prepared, spawn = _credentialed_spawn()

    class BlockingCredentialBroker(_ReplayCredentialBroker):
        def __init__(self):
            super().__init__()
            self.first_entered = Event()
            self.second_entered = Event()
            self.release_first = Event()
            self.in_flight = 0
            self.maximum_in_flight = 0
            self.state_lock = Lock()

        def issue_or_replay_exact(self, request):
            with self.state_lock:
                self.in_flight += 1
                self.maximum_in_flight = max(
                    self.maximum_in_flight,
                    self.in_flight,
                )
                invocation_position = len(self.issue_calls)
                self.issue_calls.append(request)
            if invocation_position == 0:
                self.first_entered.set()
                assert self.release_first.wait(timeout=5)
            else:
                self.second_entered.set()
            response = RunActionCredentialIssueResponse(
                credential_lease_request_id=request.credential_lease_request_id,
                payload=self.payload,
                valid_until_realtime_nanoseconds=self.valid_until,
            )
            with self.state_lock:
                self.in_flight -= 1
            return response

    backend = BlockingCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))

    def issue_and_burn():
        materialization = _issue_materialization(registry, prepared, spawn)
        authority_id, _size_bytes, _valid_until = (
            broker_module.require_run_action_credential_materialization(
                materialization,
                prepared,
                spawn,
                _authority=(broker_module._RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY),
            )
        )
        broker_module.burn_run_action_credential_materialization(
            materialization,
            _authority=broker_module._RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
        )
        return authority_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(issue_and_burn)
        assert backend.first_entered.wait(timeout=5)
        second = pool.submit(issue_and_burn)
        assert backend.second_entered.wait(timeout=0.1) is False
        assert backend.maximum_in_flight == 1
        backend.release_first.set()
        authorities = (first.result(), second.result())

    request = run_action_credential_lease_request(prepared, spawn)
    assert authorities == (
        run_action_credential_lease_authority_id(prepared, spawn),
        run_action_credential_lease_authority_id(prepared, spawn),
    )
    assert backend.issue_calls == [request, request]
    assert backend.maximum_in_flight == 1


def test_registry_response_and_materialization_reject_copy_and_forged_mint() -> None:
    prepared, spawn = _credentialed_spawn()
    registry = RunActionCredentialBrokerRegistry((_ReplayCredentialBroker(),))
    response = registry.issue_or_replay_exact(prepared, spawn)

    with pytest.raises(RunActionCredentialBrokerError, match="cannot be copied"):
        copy.copy(registry)
    with pytest.raises(RunActionCredentialBrokerError, match="cannot be copied"):
        copy.deepcopy(registry)
    with pytest.raises(RunActionCredentialBrokerError, match="serialized"):
        pickle.dumps(registry)
    with pytest.raises(RunActionCredentialBrokerError, match="cannot be copied"):
        copy.copy(response)
    with pytest.raises(RunActionCredentialBrokerError, match="cannot be copied"):
        copy.deepcopy(response)
    with pytest.raises(RunActionCredentialBrokerError, match="serialized"):
        pickle.dumps(response)

    materialization = registry.materialize_exact(response, prepared, spawn)
    with pytest.raises(RunActionCredentialBrokerError, match="cannot be copied"):
        copy.copy(materialization)
    with pytest.raises(RunActionCredentialBrokerError, match="cannot be copied"):
        copy.deepcopy(materialization)
    with pytest.raises(RunActionCredentialBrokerError, match="serialized"):
        pickle.dumps(materialization)
    with pytest.raises(RunActionCredentialBrokerError, match="invalid"):
        broker_module.RunActionCredentialMaterialization(
            request=run_action_credential_lease_request(prepared, spawn),
            credential_lease_authority_id=run_action_credential_lease_authority_id(
                prepared,
                spawn,
            ),
            payload=b"adapter-selected-forgery",
            valid_until_realtime_nanoseconds=_VALID_UNTIL,
            _authority=object(),
        )
    assert "RunActionCredentialMaterialization" not in broker_module.__all__
    broker_module.burn_run_action_credential_materialization(
        materialization,
        _authority=broker_module._RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
    )


def test_issue_response_is_bound_to_its_exact_registry() -> None:
    prepared, spawn = _credentialed_spawn()
    first_registry = RunActionCredentialBrokerRegistry((_ReplayCredentialBroker(),))
    second_registry = RunActionCredentialBrokerRegistry((_ReplayCredentialBroker(),))
    response = first_registry.issue_or_replay_exact(prepared, spawn)

    with pytest.raises(
        RunActionCredentialBrokerError,
        match="spent, cloned, or foreign",
    ):
        second_registry.materialize_exact(response, prepared, spawn)

    first_registry.burn_issue_response_exact(response)


def test_registry_rejects_foreign_process_before_backend_access() -> None:
    prepared, spawn = _credentialed_spawn()
    backend = _ReplayCredentialBroker()
    registry = RunActionCredentialBrokerRegistry((backend,))

    child = os.fork()
    if child == 0:
        with pytest.raises(RunActionCredentialBrokerError, match="foreign"):
            registry.issue_or_replay_exact(prepared, spawn)
        os._exit(37)
    _child_pid, status = os.waitpid(child, 0)

    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 37
    assert backend.issue_calls == []


def test_materialization_rejects_fork_before_inherited_lock_access() -> None:
    prepared, spawn = _credentialed_spawn()
    registry = RunActionCredentialBrokerRegistry((_ReplayCredentialBroker(),))
    materialization = _issue_materialization(registry, prepared, spawn)
    lock_held = Event()
    release_lock = Event()

    def hold_materialization_lock():
        with broker_module._CREDENTIAL_MATERIALIZATION_LOCK:
            lock_held.set()
            assert release_lock.wait(timeout=5)

    holder = Thread(target=hold_materialization_lock)
    holder.start()
    assert lock_held.wait(timeout=5)

    child = os.fork()
    if child == 0:
        signal.alarm(5)
        with pytest.raises(RunActionCredentialBrokerError, match="foreign"):
            broker_module.require_run_action_credential_materialization(
                materialization,
                prepared,
                spawn,
                _authority=broker_module._RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
            )
        signal.alarm(0)
        os._exit(37)

    release_lock.set()
    holder.join()
    _child_pid, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 37
    broker_module.burn_run_action_credential_materialization(
        materialization,
        _authority=broker_module._RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
    )


def test_credential_free_preparation_cannot_mint_a_lease_request() -> None:
    policy = _execution_policy(credential_mode=RunActionCredentialMode.NONE)
    claim = _claim(policy=policy)
    authority = _volume_authority(claim, nonce="1" * 32)
    prepared = _prepared_execution(claim=claim, authority=authority)
    spawn = _spawn_commit(prepared)
    registry = RunActionCredentialBrokerRegistry(())

    registry.require_policy(policy.credential_policy)
    with pytest.raises(ValueError, match="committed spawn"):
        run_action_credential_lease_request(prepared, spawn)


def test_status_rejects_unsigned_64_overflow() -> None:
    prepared, _spawn = _credentialed_spawn()
    request = run_action_credential_lease_request(prepared, _spawn)
    with pytest.raises(RunActionCredentialBrokerError, match="status"):
        RunActionCredentialLeaseStatus.mint(
            credential_lease_request_id=request.credential_lease_request_id,
            valid_until_realtime_nanoseconds=(RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER + 1),
        )
