"""Process-bound broker authority for spawn-specific credential leases."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import get_ident, Lock
from typing import ClassVar
from weakref import WeakKeyDictionary

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_spawn_contracts import RunActionSpawnCommit
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER,
    RunActionCredentialLeaseRequest,
    RunActionCredentialMode,
    RunActionCredentialPolicy,
    RunActionPreparedExecution,
    run_action_credential_lease_authority_id,
    run_action_credential_lease_authority_id_from_request,
    run_action_credential_lease_request,
)

_RUN_ACTION_CREDENTIAL_ISSUE_RESPONSE_AUTHORITY = object()
_RUN_ACTION_CREDENTIAL_MATERIALIZATION_MINT_AUTHORITY = object()
_RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY = object()
_RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY = object()
_CREDENTIAL_BROKER_REGISTRY_LOCK = Lock()
_CREDENTIAL_ISSUE_RESPONSE_LOCK = Lock()
_CREDENTIAL_MATERIALIZATION_LOCK = Lock()
_ISSUED_CREDENTIAL_BROKER_REGISTRIES: dict[int, "RunActionCredentialBrokerRegistry"] = (
    {}
)
_ISSUED_CREDENTIAL_RESPONSES: WeakKeyDictionary[
    "RunActionCredentialIssueResponse",
    "_RunActionCredentialIssueResponseIssuance",
] = WeakKeyDictionary()
_ISSUED_CREDENTIAL_MATERIALIZATIONS: WeakKeyDictionary[
    "RunActionCredentialMaterialization",
    "_RunActionCredentialMaterializationIssuance",
] = WeakKeyDictionary()


class RunActionCredentialBrokerError(RuntimeError):
    """A broker response or credential authority is unsafe or incompatible."""


@dataclass(frozen=True, repr=False)
class _RunActionCredentialIssueResponseIssuance:
    """Registry-private ownership transferred from one backend response."""

    credential_lease_request_id: str
    issuing_registry_id: int
    owner_process_id: int
    owner_thread_id: int
    payload: bytes | None
    valid_until_realtime_nanoseconds: int
    state: str

    def __repr__(self) -> str:
        return "_RunActionCredentialIssueResponseIssuance(payload=<redacted>)"


@dataclass(frozen=True)
class RunActionCredentialLeaseStatus(StrictContract):
    """Non-secret backend status for one deterministic lease request."""

    credential_lease_status_id: str
    credential_lease_request_id: str
    valid_until_realtime_nanoseconds: int

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-credential-lease-status"
    IDENTITY_FIELD: ClassVar[str] = "credential_lease_status_id"

    def _validate(self) -> None:
        request_id = require_content_id(
            self.credential_lease_request_id,
            "credential lease status request",
        )
        if (
            request_id.split(":sha256:", 1)[0]
            != RunActionCredentialLeaseRequest.CONTENT_NAMESPACE
            or type(self.valid_until_realtime_nanoseconds) is not int
            or self.valid_until_realtime_nanoseconds <= 0
            or self.valid_until_realtime_nanoseconds
            > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
        ):
            raise RunActionCredentialBrokerError("credential lease status is invalid")


class RunActionCredentialIssueResponse:
    """Ephemeral backend response whose secret payload has no public accessor."""

    __slots__ = (
        "__weakref__",
        "_credential_lease_request_id",
        "_owner_process_id",
        "_owner_thread_id",
        "_payload",
        "_state",
        "_valid_until_realtime_nanoseconds",
    )

    def __init__(
        self,
        *,
        credential_lease_request_id: str,
        payload: bytes,
        valid_until_realtime_nanoseconds: int,
    ) -> None:
        request_id = require_content_id(
            credential_lease_request_id,
            "credential issue response request",
        )
        if (
            request_id.split(":sha256:", 1)[0]
            != RunActionCredentialLeaseRequest.CONTENT_NAMESPACE
            or type(payload) is not bytes
            or not payload
            or type(valid_until_realtime_nanoseconds) is not int
            or valid_until_realtime_nanoseconds <= 0
            or valid_until_realtime_nanoseconds > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
        ):
            raise RunActionCredentialBrokerError("credential issue response is invalid")
        self._credential_lease_request_id = request_id
        self._payload = payload
        self._valid_until_realtime_nanoseconds = valid_until_realtime_nanoseconds
        self._owner_process_id = os.getpid()
        self._owner_thread_id = get_ident()
        self._state = "backend_response"

    @property
    def credential_lease_request_id(self) -> str:
        issuance = _ISSUED_CREDENTIAL_RESPONSES.get(self)
        if type(issuance) is _RunActionCredentialIssueResponseIssuance:
            if issuance.owner_process_id != os.getpid():
                raise RunActionCredentialBrokerError(
                    "credential issue response is spent, cloned, or foreign"
                )
            return issuance.credential_lease_request_id
        return self._credential_lease_request_id

    @property
    def valid_until_realtime_nanoseconds(self) -> int:
        issuance = _ISSUED_CREDENTIAL_RESPONSES.get(self)
        if issuance is None and (
            self._owner_process_id == os.getpid()
            and self._owner_thread_id == get_ident()
            and self._state == "backend_response"
        ):
            return self._valid_until_realtime_nanoseconds
        if (
            type(issuance) is not _RunActionCredentialIssueResponseIssuance
            or issuance.owner_process_id != os.getpid()
        ):
            raise RunActionCredentialBrokerError(
                "credential issue response is spent, cloned, or foreign"
            )
        with _CREDENTIAL_ISSUE_RESPONSE_LOCK:
            issuance = _ISSUED_CREDENTIAL_RESPONSES.get(self)
        if (
            type(issuance) is not _RunActionCredentialIssueResponseIssuance
            or issuance.owner_process_id != os.getpid()
            or issuance.owner_thread_id != get_ident()
            or issuance.state != "sealed"
        ):
            raise RunActionCredentialBrokerError(
                "credential issue response is spent, cloned, or foreign"
            )
        return issuance.valid_until_realtime_nanoseconds

    def _seal(
        self,
        *,
        registry_id: int,
        request: RunActionCredentialLeaseRequest,
        maximum_delivery_size_bytes: int,
        _authority: object,
    ) -> None:
        if _authority is not _RUN_ACTION_CREDENTIAL_ISSUE_RESPONSE_AUTHORITY:
            raise RunActionCredentialBrokerError(
                "credential issue response cannot receive broker authority"
            )
        if (
            type(registry_id) is not int
            or registry_id <= 0
            or type(request) is not RunActionCredentialLeaseRequest
            or self._owner_process_id != os.getpid()
            or self._owner_thread_id != get_ident()
            or self._state != "backend_response"
            or type(self._payload) is not bytes
            or type(self._valid_until_realtime_nanoseconds) is not int
            or self._valid_until_realtime_nanoseconds <= 0
            or self._valid_until_realtime_nanoseconds
            > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
        ):
            self._reject_backend_response()
            raise RunActionCredentialBrokerError(
                "credential issue response cannot receive broker authority"
            )
        if self._credential_lease_request_id != request.credential_lease_request_id:
            self._reject_backend_response()
            raise RunActionCredentialBrokerError(
                "credential broker returned another issue response"
            )
        if (
            type(maximum_delivery_size_bytes) is not int
            or maximum_delivery_size_bytes <= 0
        ):
            self._reject_backend_response()
            raise RunActionCredentialBrokerError(
                "credential issue response cannot receive broker authority"
            )
        if len(self._payload) > maximum_delivery_size_bytes:
            self._reject_backend_response()
            raise RunActionCredentialBrokerError(
                "credential broker payload exceeds its policy"
            )
        issuance = _RunActionCredentialIssueResponseIssuance(
            credential_lease_request_id=self._credential_lease_request_id,
            issuing_registry_id=registry_id,
            owner_process_id=self._owner_process_id,
            owner_thread_id=self._owner_thread_id,
            payload=self._payload,
            valid_until_realtime_nanoseconds=(self._valid_until_realtime_nanoseconds),
            state="sealed",
        )
        self._payload = None
        self._state = "sealed"
        with _CREDENTIAL_ISSUE_RESPONSE_LOCK:
            _ISSUED_CREDENTIAL_RESPONSES[self] = issuance

    def _reject_backend_response(self) -> None:
        """Drop bytes from one exact unsealed response rejected by the registry."""

        with _CREDENTIAL_ISSUE_RESPONSE_LOCK:
            if _ISSUED_CREDENTIAL_RESPONSES.get(self) is not None:
                raise RunActionCredentialBrokerError(
                    "credential issue response rejection authority changed"
                )
        self._payload = None
        self._state = "spent"

    def _take_payload(
        self,
        *,
        registry_id: int,
        request: RunActionCredentialLeaseRequest,
        _authority: object,
    ) -> tuple[bytes, int]:
        issuance = _ISSUED_CREDENTIAL_RESPONSES.get(self)
        if (
            type(issuance) is not _RunActionCredentialIssueResponseIssuance
            or issuance.owner_process_id != os.getpid()
        ):
            raise RunActionCredentialBrokerError(
                "credential issue response is spent, cloned, or foreign"
            )
        with _CREDENTIAL_ISSUE_RESPONSE_LOCK:
            issuance = _ISSUED_CREDENTIAL_RESPONSES.get(self)
            if (
                type(issuance) is not _RunActionCredentialIssueResponseIssuance
                or issuance.issuing_registry_id != registry_id
                or issuance.owner_process_id != os.getpid()
                or issuance.owner_thread_id != get_ident()
                or type(request) is not RunActionCredentialLeaseRequest
                or issuance.credential_lease_request_id
                != request.credential_lease_request_id
                or type(issuance.payload) is not bytes
                or issuance.state != "sealed"
                or _authority is not _RUN_ACTION_CREDENTIAL_ISSUE_RESPONSE_AUTHORITY
            ):
                raise RunActionCredentialBrokerError(
                    "credential issue response is spent, cloned, or foreign"
                )
            payload = issuance.payload
            valid_until = issuance.valid_until_realtime_nanoseconds
            _ISSUED_CREDENTIAL_RESPONSES[self] = (
                _RunActionCredentialIssueResponseIssuance(
                    credential_lease_request_id=(issuance.credential_lease_request_id),
                    issuing_registry_id=issuance.issuing_registry_id,
                    owner_process_id=issuance.owner_process_id,
                    owner_thread_id=issuance.owner_thread_id,
                    payload=None,
                    valid_until_realtime_nanoseconds=(
                        issuance.valid_until_realtime_nanoseconds
                    ),
                    state="spent",
                )
            )
        self._state = "spent"
        return payload, valid_until

    def _burn(
        self,
        *,
        registry_id: int,
        _authority: object,
    ) -> None:
        issuance = _ISSUED_CREDENTIAL_RESPONSES.get(self)
        if (
            type(issuance) is not _RunActionCredentialIssueResponseIssuance
            or issuance.owner_process_id != os.getpid()
            or _authority is not _RUN_ACTION_CREDENTIAL_ISSUE_RESPONSE_AUTHORITY
        ):
            raise RunActionCredentialBrokerError(
                "credential issue response cannot be burned by this caller"
            )
        with _CREDENTIAL_ISSUE_RESPONSE_LOCK:
            issuance = _ISSUED_CREDENTIAL_RESPONSES.get(self)
            if (
                type(issuance) is not _RunActionCredentialIssueResponseIssuance
                or issuance.issuing_registry_id != registry_id
                or issuance.owner_process_id != os.getpid()
                or issuance.owner_thread_id != get_ident()
                or issuance.state not in {"sealed", "spent"}
            ):
                raise RunActionCredentialBrokerError(
                    "credential issue response authority changed"
                )
            if issuance.state == "sealed":
                _ISSUED_CREDENTIAL_RESPONSES[self] = (
                    _RunActionCredentialIssueResponseIssuance(
                        credential_lease_request_id=(
                            issuance.credential_lease_request_id
                        ),
                        issuing_registry_id=issuance.issuing_registry_id,
                        owner_process_id=issuance.owner_process_id,
                        owner_thread_id=issuance.owner_thread_id,
                        payload=None,
                        valid_until_realtime_nanoseconds=(
                            issuance.valid_until_realtime_nanoseconds
                        ),
                        state="spent",
                    )
                )
        self._payload = None
        self._state = "spent"

    def __copy__(self):
        raise RunActionCredentialBrokerError(
            "credential issue response cannot be copied"
        )

    def __deepcopy__(self, memo):
        raise RunActionCredentialBrokerError(
            "credential issue response cannot be copied"
        )

    def __reduce__(self):
        raise RunActionCredentialBrokerError(
            "credential issue response cannot be serialized"
        )

    def __repr__(self) -> str:
        return "RunActionCredentialIssueResponse(payload=<redacted>)"


class RunActionCredentialBrokerBackend(ABC):
    """External idempotent issue/replay and status boundary."""

    def __init__(self, *, broker_id: str, broker_protocol_version: str) -> None:
        self._broker_id = require_identifier(
            broker_id,
            "run action credential broker",
        )
        self._broker_protocol_version = require_identifier(
            broker_protocol_version,
            "run action credential broker protocol",
        )

    @property
    def broker_id(self) -> str:
        return self._broker_id

    @property
    def broker_protocol_version(self) -> str:
        return self._broker_protocol_version

    @abstractmethod
    def issue_or_replay_exact(
        self,
        request: RunActionCredentialLeaseRequest,
    ) -> RunActionCredentialIssueResponse:
        """Return byte-identical material for every replay of one request."""

    @abstractmethod
    def observe_exact(
        self,
        request: RunActionCredentialLeaseRequest,
    ) -> RunActionCredentialLeaseStatus:
        """Return current non-secret status for the same external lease."""


@dataclass(frozen=True)
class _RegisteredCredentialBroker:
    """Captured implementation identity; runtime lookup cannot substitute it."""

    backend: RunActionCredentialBrokerBackend
    backend_type: type[RunActionCredentialBrokerBackend]
    broker_id: str
    broker_protocol_version: str
    issue_function: object
    observe_function: object

    def is_current(self) -> bool:
        return not (
            type(self.backend) is not self.backend_type
            or self.backend.broker_id != self.broker_id
            or self.backend.broker_protocol_version != self.broker_protocol_version
            or self.backend_type.issue_or_replay_exact is not self.issue_function
            or self.backend_type.observe_exact is not self.observe_function
        )

    def require_current(self) -> None:
        if not self.is_current():
            raise RunActionCredentialBrokerError(
                "registered credential broker implementation changed"
            )


@dataclass(frozen=True, repr=False)
class _RunActionCredentialMaterializationIssuance:
    """Broker-private authority retained outside the adapter-visible token."""

    request: RunActionCredentialLeaseRequest
    credential_lease_authority_id: str
    owner_process_id: int
    owner_thread_id: int
    payload: bytes
    valid_until_realtime_nanoseconds: int

    def __post_init__(self) -> None:
        if (
            type(self.request) is not RunActionCredentialLeaseRequest
            or self.credential_lease_authority_id
            != run_action_credential_lease_authority_id_from_request(self.request)
            or type(self.owner_process_id) is not int
            or self.owner_process_id <= 0
            or type(self.owner_thread_id) is not int
            or self.owner_thread_id <= 0
            or type(self.payload) is not bytes
            or not self.payload
            or type(self.valid_until_realtime_nanoseconds) is not int
            or self.valid_until_realtime_nanoseconds <= 0
            or self.valid_until_realtime_nanoseconds
            > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
        ):
            raise RunActionCredentialBrokerError(
                "credential materialization issuance is invalid"
            )

    def __repr__(self) -> str:
        return "_RunActionCredentialMaterializationIssuance(payload=<redacted>)"


class RunActionCredentialMaterialization:
    """Opaque, one-use, thread-bound credential bytes for the delivery leaf."""

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        *,
        request: RunActionCredentialLeaseRequest,
        credential_lease_authority_id: str,
        payload: bytes,
        valid_until_realtime_nanoseconds: int,
        _authority: object,
    ) -> None:
        if (
            type(request) is not RunActionCredentialLeaseRequest
            or type(payload) is not bytes
            or not payload
            or type(valid_until_realtime_nanoseconds) is not int
            or valid_until_realtime_nanoseconds <= 0
            or valid_until_realtime_nanoseconds > RUN_ACTION_MAXIMUM_PHYSICAL_INTEGER
            or _authority is not _RUN_ACTION_CREDENTIAL_MATERIALIZATION_MINT_AUTHORITY
        ):
            raise RunActionCredentialBrokerError(
                "credential materialization is invalid"
            )
        expected_authority_id = run_action_credential_lease_authority_id_from_request(
            request
        )
        if credential_lease_authority_id != expected_authority_id:
            raise RunActionCredentialBrokerError(
                "credential materialization authority differs from its request"
            )
        with _CREDENTIAL_MATERIALIZATION_LOCK:
            if _ISSUED_CREDENTIAL_MATERIALIZATIONS.get(self) is not None:
                raise RunActionCredentialBrokerError(
                    "credential materialization identity is already issued"
                )
            _ISSUED_CREDENTIAL_MATERIALIZATIONS[self] = (
                _RunActionCredentialMaterializationIssuance(
                    request=request,
                    credential_lease_authority_id=expected_authority_id,
                    owner_process_id=os.getpid(),
                    owner_thread_id=get_ident(),
                    payload=payload,
                    valid_until_realtime_nanoseconds=(valid_until_realtime_nanoseconds),
                )
            )

    def _require_exact(
        self,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        *,
        _authority: object,
    ) -> tuple[str, int, int]:
        _require_credential_materialization_owner_process(self)
        expected_request = run_action_credential_lease_request(
            prepared_execution,
            spawn_commit,
        )
        expected_authority_id = run_action_credential_lease_authority_id(
            prepared_execution,
            spawn_commit,
        )
        with _CREDENTIAL_MATERIALIZATION_LOCK:
            issuance = _ISSUED_CREDENTIAL_MATERIALIZATIONS.get(self)
            if (
                type(issuance) is not _RunActionCredentialMaterializationIssuance
                or issuance.owner_process_id != os.getpid()
                or issuance.owner_thread_id != get_ident()
                or issuance.request != expected_request
                or issuance.credential_lease_authority_id != expected_authority_id
                or _authority
                not in {
                    _RUN_ACTION_CREDENTIAL_DELIVERY_AUTHORITY,
                    _RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY,
                }
            ):
                raise RunActionCredentialBrokerError(
                    "credential materialization is spent, cloned, or foreign"
                )
            return (
                issuance.credential_lease_authority_id,
                len(issuance.payload),
                issuance.valid_until_realtime_nanoseconds,
            )

    def _consume(
        self,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        *,
        _authority: object,
    ) -> bytes:
        self._require_exact(
            prepared_execution,
            spawn_commit,
            _authority=_authority,
        )
        with _CREDENTIAL_MATERIALIZATION_LOCK:
            issuance = _ISSUED_CREDENTIAL_MATERIALIZATIONS.get(self)
            if (
                type(issuance) is not _RunActionCredentialMaterializationIssuance
                or issuance.request
                != run_action_credential_lease_request(
                    prepared_execution,
                    spawn_commit,
                )
                or issuance.credential_lease_authority_id
                != run_action_credential_lease_authority_id(
                    prepared_execution,
                    spawn_commit,
                )
            ):
                raise RunActionCredentialBrokerError(
                    "credential materialization changed before delivery"
                )
            payload = issuance.payload
            _ISSUED_CREDENTIAL_MATERIALIZATIONS.pop(self)
            return payload

    def _burn(self, *, _authority: object) -> None:
        if _authority is not _RUN_ACTION_CREDENTIAL_LIFECYCLE_AUTHORITY:
            raise RunActionCredentialBrokerError(
                "credential materialization cannot be burned by this caller"
            )
        issued_before_lock = _ISSUED_CREDENTIAL_MATERIALIZATIONS.get(self)
        if issued_before_lock is None:
            return
        if (
            type(issued_before_lock) is not _RunActionCredentialMaterializationIssuance
            or issued_before_lock.owner_process_id != os.getpid()
        ):
            raise RunActionCredentialBrokerError(
                "credential materialization issuance changed"
            )
        with _CREDENTIAL_MATERIALIZATION_LOCK:
            issuance = _ISSUED_CREDENTIAL_MATERIALIZATIONS.get(self)
            if issuance is None:
                return
            if (
                type(issuance) is not _RunActionCredentialMaterializationIssuance
                or issuance.owner_process_id != os.getpid()
                or issuance.owner_thread_id != get_ident()
            ):
                raise RunActionCredentialBrokerError(
                    "credential materialization issuance changed"
                )
            _ISSUED_CREDENTIAL_MATERIALIZATIONS.pop(self, None)

    def __copy__(self):
        raise RunActionCredentialBrokerError(
            "credential materialization cannot be copied"
        )

    def __deepcopy__(self, memo):
        raise RunActionCredentialBrokerError(
            "credential materialization cannot be copied"
        )

    def __reduce__(self):
        raise RunActionCredentialBrokerError(
            "credential materialization cannot be serialized"
        )

    def __repr__(self) -> str:
        return "RunActionCredentialMaterialization(payload=<redacted>)"


class RunActionCredentialBrokerRegistry:
    """Process-bound exact broker catalog shared by issue and validity paths."""

    def __init__(
        self,
        backends: tuple[RunActionCredentialBrokerBackend, ...],
    ) -> None:
        if type(backends) is not tuple or any(
            not isinstance(backend, RunActionCredentialBrokerBackend)
            for backend in backends
        ):
            raise RunActionCredentialBrokerError(
                "credential broker registry requires exact backend implementations"
            )
        registered: dict[tuple[str, str], _RegisteredCredentialBroker] = {}
        for backend in backends:
            backend_type = type(backend)
            if (
                backend_type is RunActionCredentialBrokerBackend
                or backend_type.issue_or_replay_exact
                is RunActionCredentialBrokerBackend.issue_or_replay_exact
                or backend_type.observe_exact
                is RunActionCredentialBrokerBackend.observe_exact
            ):
                raise RunActionCredentialBrokerError(
                    "credential broker backend is abstract or incomplete"
                )
            key = (backend.broker_id, backend.broker_protocol_version)
            if key in registered:
                raise RunActionCredentialBrokerError(
                    "credential broker registry contains a duplicate implementation"
                )
            registered[key] = _RegisteredCredentialBroker(
                backend=backend,
                backend_type=backend_type,
                broker_id=key[0],
                broker_protocol_version=key[1],
                issue_function=backend_type.issue_or_replay_exact,
                observe_function=backend_type.observe_exact,
            )
        self._registered = registered
        self._invocation_lock = Lock()
        self._owner_process_id = os.getpid()
        with _CREDENTIAL_BROKER_REGISTRY_LOCK:
            _ISSUED_CREDENTIAL_BROKER_REGISTRIES[id(self)] = self

    def require_policy(self, policy: RunActionCredentialPolicy) -> None:
        """Prove that a brokered policy has one sealed implementation."""

        self._require_owner_process()
        if type(policy) is not RunActionCredentialPolicy:
            raise RunActionCredentialBrokerError(
                "credential broker policy has another type"
            )
        if policy.mode is RunActionCredentialMode.NONE:
            return
        self._registered_for(policy).require_current()

    def issue_or_replay_exact(
        self,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
    ) -> RunActionCredentialIssueResponse:
        """Return one registry-sealed, ephemeral exact backend response."""

        self._require_owner_process()
        request = RunActionCredentialLeaseRequest.from_json_bytes(
            run_action_credential_lease_request(
                prepared_execution,
                spawn_commit,
            ).to_json_bytes()
        )
        policy = request.credential_policy
        registered = self._registered_for(policy)
        with self._invocation_lock:
            registered.require_current()
            backend_request = RunActionCredentialLeaseRequest.from_json_bytes(
                request.to_json_bytes()
            )
            response = registered.issue_function(registered.backend, backend_request)
            if backend_request != request:
                if type(response) is RunActionCredentialIssueResponse:
                    response._reject_backend_response()
                raise RunActionCredentialBrokerError(
                    "credential broker changed its lease request"
                )
            if (
                type(response) is RunActionCredentialIssueResponse
                and not registered.is_current()
            ):
                response._reject_backend_response()
                raise RunActionCredentialBrokerError(
                    "registered credential broker implementation changed"
                )
            registered.require_current()
            if type(response) is not RunActionCredentialIssueResponse:
                raise RunActionCredentialBrokerError(
                    "credential broker returned another issue response"
                )
            response._seal(
                registry_id=id(self),
                request=request,
                maximum_delivery_size_bytes=policy.maximum_delivery_size_bytes,
                _authority=_RUN_ACTION_CREDENTIAL_ISSUE_RESPONSE_AUTHORITY,
            )
        return response

    def materialize_exact(
        self,
        response: RunActionCredentialIssueResponse,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
    ) -> RunActionCredentialMaterialization:
        """Consume one sealed response into broker-minted delivery authority."""

        self._require_owner_process()
        if type(response) is not RunActionCredentialIssueResponse:
            raise RunActionCredentialBrokerError(
                "credential materialization lacks its sealed issue response"
            )
        request = RunActionCredentialLeaseRequest.from_json_bytes(
            run_action_credential_lease_request(
                prepared_execution,
                spawn_commit,
            ).to_json_bytes()
        )
        self._registered_for(request.credential_policy).require_current()
        payload, valid_until = response._take_payload(
            registry_id=id(self),
            request=request,
            _authority=_RUN_ACTION_CREDENTIAL_ISSUE_RESPONSE_AUTHORITY,
        )
        return RunActionCredentialMaterialization(
            request=request,
            credential_lease_authority_id=(
                run_action_credential_lease_authority_id(
                    prepared_execution,
                    spawn_commit,
                )
            ),
            payload=payload,
            valid_until_realtime_nanoseconds=valid_until,
            _authority=_RUN_ACTION_CREDENTIAL_MATERIALIZATION_MINT_AUTHORITY,
        )

    def burn_issue_response_exact(
        self,
        response: RunActionCredentialIssueResponse,
    ) -> None:
        """Deterministically drop one unused sealed backend response."""

        self._require_owner_process()
        if type(response) is not RunActionCredentialIssueResponse:
            raise RunActionCredentialBrokerError(
                "credential lifecycle lacks its exact issue response"
            )
        response._burn(
            registry_id=id(self),
            _authority=_RUN_ACTION_CREDENTIAL_ISSUE_RESPONSE_AUTHORITY,
        )

    def observe_exact(
        self,
        *,
        prepared_execution: RunActionPreparedExecution,
        spawn_commit: RunActionSpawnCommit,
        activated_credential_file_observation_id: str,
    ) -> RunActionCredentialLeaseStatus:
        """Return exact status through the same sealed broker that issued."""

        self._require_owner_process()
        activated_id = require_content_id(
            activated_credential_file_observation_id,
            "credential broker activated file",
        )
        if (
            activated_id.split(":sha256:", 1)[0]
            != "run-action-activated-file-observation"
        ):
            raise RunActionCredentialBrokerError(
                "credential broker activated file has another namespace"
            )
        request = RunActionCredentialLeaseRequest.from_json_bytes(
            run_action_credential_lease_request(
                prepared_execution,
                spawn_commit,
            ).to_json_bytes()
        )
        registered = self._registered_for(request.credential_policy)
        with self._invocation_lock:
            registered.require_current()
            backend_request = RunActionCredentialLeaseRequest.from_json_bytes(
                request.to_json_bytes()
            )
            backend_status = registered.observe_function(
                registered.backend,
                backend_request,
            )
            if backend_request != request:
                raise RunActionCredentialBrokerError(
                    "credential broker changed its status request"
                )
            registered.require_current()
        if type(backend_status) is not RunActionCredentialLeaseStatus:
            raise RunActionCredentialBrokerError(
                "credential broker returned another lease status"
            )
        status = RunActionCredentialLeaseStatus.from_json_bytes(
            backend_status.to_json_bytes()
        )
        if status.credential_lease_request_id != request.credential_lease_request_id:
            raise RunActionCredentialBrokerError(
                "credential broker returned another lease status"
            )
        return status

    def _registered_for(
        self,
        policy: RunActionCredentialPolicy,
    ) -> _RegisteredCredentialBroker:
        if (
            type(policy) is not RunActionCredentialPolicy
            or policy.mode is not RunActionCredentialMode.SUPERVISOR_FILE
            or not isinstance(policy.broker_id, str)
            or not isinstance(policy.broker_protocol_version, str)
        ):
            raise RunActionCredentialBrokerError(
                "credential broker lookup requires one brokered policy"
            )
        registered = self._registered.get(
            (policy.broker_id, policy.broker_protocol_version)
        )
        if type(registered) is not _RegisteredCredentialBroker:
            raise RunActionCredentialBrokerError(
                "credential policy lacks one registered broker implementation"
            )
        return registered

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise RunActionCredentialBrokerError(
                "credential broker registry is cloned or foreign"
            )
        with _CREDENTIAL_BROKER_REGISTRY_LOCK:
            issued = _ISSUED_CREDENTIAL_BROKER_REGISTRIES.get(id(self))
        if issued is not self:
            raise RunActionCredentialBrokerError(
                "credential broker registry is cloned or foreign"
            )

    def __copy__(self):
        raise RunActionCredentialBrokerError(
            "credential broker registry cannot be copied"
        )

    def __deepcopy__(self, memo):
        raise RunActionCredentialBrokerError(
            "credential broker registry cannot be copied"
        )

    def __reduce__(self):
        raise RunActionCredentialBrokerError(
            "credential broker registry cannot be serialized"
        )


def _require_credential_materialization_owner_process(
    materialization: RunActionCredentialMaterialization,
) -> None:
    issuance = _ISSUED_CREDENTIAL_MATERIALIZATIONS.get(materialization)
    if (
        type(issuance) is not _RunActionCredentialMaterializationIssuance
        or issuance.owner_process_id != os.getpid()
    ):
        raise RunActionCredentialBrokerError(
            "credential materialization is spent, cloned, or foreign"
        )


def require_run_action_credential_materialization(
    materialization: RunActionCredentialMaterialization,
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    *,
    _authority: object,
) -> tuple[str, int, int]:
    """Return only non-secret delivery metadata after exact authority checks."""

    if type(materialization) is not RunActionCredentialMaterialization:
        raise RunActionCredentialBrokerError(
            "credential delivery lacks its exact materialization"
        )
    return materialization._require_exact(
        prepared_execution,
        spawn_commit,
        _authority=_authority,
    )


def consume_run_action_credential_materialization(
    materialization: RunActionCredentialMaterialization,
    prepared_execution: RunActionPreparedExecution,
    spawn_commit: RunActionSpawnCommit,
    *,
    _authority: object,
) -> bytes:
    """Consume secret bytes exactly once inside the trusted delivery leaf."""

    if type(materialization) is not RunActionCredentialMaterialization:
        raise RunActionCredentialBrokerError(
            "credential delivery lacks its exact materialization"
        )
    return materialization._consume(
        prepared_execution,
        spawn_commit,
        _authority=_authority,
    )


def burn_run_action_credential_materialization(
    materialization: RunActionCredentialMaterialization,
    *,
    _authority: object,
) -> None:
    """Drop an unconsumed secret reference when activation authority closes."""

    if type(materialization) is not RunActionCredentialMaterialization:
        raise RunActionCredentialBrokerError(
            "credential lifecycle lacks its exact materialization"
        )
    materialization._burn(_authority=_authority)


__all__ = [
    "RunActionCredentialBrokerBackend",
    "RunActionCredentialBrokerError",
    "RunActionCredentialBrokerRegistry",
    "RunActionCredentialIssueResponse",
    "RunActionCredentialLeaseStatus",
]
