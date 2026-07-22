"""Exact execution-provider dispatch for expert source replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from kapso.cross_run.canonical import require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.expert.replay_protocol import (
    TaskEvaluatorInvocationAllocation,
    TaskEvaluatorRequest,
    build_task_evaluator_request,
)
from kapso.cross_run.expert.replay_request import (
    MaterializedExpertSourceReplayCase,
    PreparedExpertSourceReplayRequest,
)
from kapso.cross_run.process import BoundedProcessResult

_MISSING_PROVIDER_KEY = object()


class ExpertSourceReplayExecutionError(ValueError):
    """Source-replay execution dispatch is absent, ambiguous, or stale."""


@dataclass(frozen=True)
class ExpertSourceReplayExecutionProviderKey(StrictContract):
    paired_execution_protocol_version: str
    execution_provider_id: str
    execution_provider_version: str
    sandbox_policy_version: str
    task_adapter_runtime_protocol_version: str
    task_evaluator_protocol_version: str

    def _validate(self) -> None:
        for value, name in zip(self.identity, self.field_names, strict=True):
            require_identifier(value, f"source replay provider key {name}")

    @property
    def identity(self) -> tuple[str, ...]:
        return (
            self.paired_execution_protocol_version,
            self.execution_provider_id,
            self.execution_provider_version,
            self.sandbox_policy_version,
            self.task_adapter_runtime_protocol_version,
            self.task_evaluator_protocol_version,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return (
            "paired_execution_protocol_version",
            "execution_provider_id",
            "execution_provider_version",
            "sandbox_policy_version",
            "task_adapter_runtime_protocol_version",
            "task_evaluator_protocol_version",
        )


class ExpertSourceReplayExecutionProvider(Protocol):
    """An isolated provider registered under one exact implementation key."""

    dispatch_key: ExpertSourceReplayExecutionProviderKey

    def execute_leg(
        self,
        invocation: ExpertSourceReplayMatchedLegInvocation,
    ) -> ExpertSourceReplayProviderCompletion:
        """Execute one exact journal-owned scientific leg without retry."""


def expert_source_replay_execution_provider_key(
    materialized_case: MaterializedExpertSourceReplayCase,
) -> ExpertSourceReplayExecutionProviderKey:
    """Derive implementation dispatch only from the verified case authorities."""

    if not isinstance(materialized_case, MaterializedExpertSourceReplayCase):
        raise ExpertSourceReplayExecutionError(
            "source replay provider key requires a materialized case"
        )
    compute = materialized_case.request_case.compute_binding
    manifest = materialized_case.task_adapter.manifest
    return ExpertSourceReplayExecutionProviderKey(
        paired_execution_protocol_version=(compute.paired_execution_protocol_version),
        execution_provider_id=compute.execution_provider_id,
        execution_provider_version=compute.execution_provider_version,
        sandbox_policy_version=compute.sandbox_policy_version,
        task_adapter_runtime_protocol_version=(
            manifest.runtime.runtime_protocol_version
        ),
        task_evaluator_protocol_version=manifest.task_evaluator.protocol_version,
    )


@dataclass(frozen=True)
class SourceReplayProviderExecutionHandle(StrictContract):
    provider_handle_id: str
    dispatch_key: ExpertSourceReplayExecutionProviderKey
    invocation_allocation: TaskEvaluatorInvocationAllocation

    CONTENT_NAMESPACE = "source-replay-provider-execution-handle"
    IDENTITY_FIELD = "provider_handle_id"


def source_replay_provider_execution_handle(
    dispatch_key: ExpertSourceReplayExecutionProviderKey,
    invocation_allocation: TaskEvaluatorInvocationAllocation,
) -> SourceReplayProviderExecutionHandle:
    return SourceReplayProviderExecutionHandle.mint(
        dispatch_key=dispatch_key,
        invocation_allocation=invocation_allocation,
    )


@dataclass(frozen=True)
class ExpertSourceReplayMatchedLegInvocation:
    materialized_case: MaterializedExpertSourceReplayCase
    invocation_allocation: TaskEvaluatorInvocationAllocation
    task_evaluator_request: TaskEvaluatorRequest
    provider_handle: SourceReplayProviderExecutionHandle

    def __post_init__(self) -> None:
        if not isinstance(self.materialized_case, MaterializedExpertSourceReplayCase):
            raise ExpertSourceReplayExecutionError(
                "source replay invocation requires its materialized case"
            )
        expected_key = expert_source_replay_execution_provider_key(
            self.materialized_case
        )
        if (
            not isinstance(
                self.invocation_allocation,
                TaskEvaluatorInvocationAllocation,
            )
            or not isinstance(self.task_evaluator_request, TaskEvaluatorRequest)
            or not isinstance(
                self.provider_handle,
                SourceReplayProviderExecutionHandle,
            )
            or self.provider_handle.dispatch_key != expected_key
            or self.provider_handle.invocation_allocation != self.invocation_allocation
            or self.task_evaluator_request
            != build_task_evaluator_request(
                self.materialized_case,
                self.invocation_allocation,
            )
        ):
            raise ExpertSourceReplayExecutionError(
                "source replay invocation differs from its exact case authority"
            )


@dataclass(frozen=True)
class ExpertSourceReplayProviderCompletion:
    provider_handle_id: str
    process_result: BoundedProcessResult
    result_payload: bytes | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.provider_handle_id, str)
            or not self.provider_handle_id
            or type(self.process_result) is not BoundedProcessResult
            or (
                self.result_payload is not None
                and not isinstance(self.result_payload, bytes)
            )
        ):
            raise ExpertSourceReplayExecutionError(
                "source replay provider completion is invalid"
            )


@dataclass(frozen=True, slots=True, init=False)
class ResolvedExpertSourceReplayExecutionCase:
    """Runtime-only exact provider binding with no public invocation surface."""

    materialized_case: MaterializedExpertSourceReplayCase
    dispatch_key: ExpertSourceReplayExecutionProviderKey
    _provider: ExpertSourceReplayExecutionProvider = field(repr=False, compare=False)

    def __init__(
        self,
        materialized_case: MaterializedExpertSourceReplayCase,
        dispatch_key: ExpertSourceReplayExecutionProviderKey,
        provider: ExpertSourceReplayExecutionProvider,
    ) -> None:
        if not isinstance(materialized_case, MaterializedExpertSourceReplayCase):
            raise ExpertSourceReplayExecutionError(
                "resolved source replay execution requires a materialized case"
            )
        if not isinstance(dispatch_key, ExpertSourceReplayExecutionProviderKey):
            raise ExpertSourceReplayExecutionError(
                "resolved source replay execution requires a typed dispatch key"
            )
        if not callable(getattr(provider, "execute_leg", None)):
            raise ExpertSourceReplayExecutionError(
                "resolved source replay provider cannot execute a matched leg"
            )
        if (
            expert_source_replay_execution_provider_key(materialized_case)
            != dispatch_key
        ):
            raise ExpertSourceReplayExecutionError(
                "resolved source replay dispatch differs from its case"
            )
        object.__setattr__(self, "materialized_case", materialized_case)
        object.__setattr__(self, "dispatch_key", dispatch_key)
        object.__setattr__(self, "_provider", provider)
        self.require_current_provider_identity()

    def require_current_provider_identity(self) -> None:
        current_key = getattr(
            self._provider,
            "dispatch_key",
            _MISSING_PROVIDER_KEY,
        )
        if current_key != self.dispatch_key:
            raise ExpertSourceReplayExecutionError(
                "resolved source replay provider identity changed"
            )

    def _execute_leg(
        self,
        invocation: ExpertSourceReplayMatchedLegInvocation,
    ) -> ExpertSourceReplayProviderCompletion:
        self.require_current_provider_identity()
        execute_leg = getattr(self._provider, "execute_leg", None)
        if not callable(execute_leg):
            raise ExpertSourceReplayExecutionError(
                "source replay provider has no matched-leg executor"
            )
        completion = execute_leg(invocation)
        self.require_current_provider_identity()
        if type(completion) is not ExpertSourceReplayProviderCompletion:
            raise ExpertSourceReplayExecutionError(
                "source replay provider returned an untyped completion"
            )
        return completion


class ExpertSourceReplayExecutionProviderRegistry:
    """Resolve complete replay requests through exact full-key equality only."""

    def __init__(
        self,
        providers: tuple[ExpertSourceReplayExecutionProvider, ...],
    ) -> None:
        if not isinstance(providers, tuple) or not providers:
            raise ExpertSourceReplayExecutionError(
                "source replay execution providers must be a non-empty tuple"
            )
        providers_by_key: dict[
            ExpertSourceReplayExecutionProviderKey,
            ExpertSourceReplayExecutionProvider,
        ] = {}
        for provider in providers:
            advertised_key = getattr(
                provider,
                "dispatch_key",
                _MISSING_PROVIDER_KEY,
            )
            if not isinstance(
                advertised_key,
                ExpertSourceReplayExecutionProviderKey,
            ):
                raise ExpertSourceReplayExecutionError(
                    "source replay provider must advertise a typed dispatch key"
                )
            if not callable(getattr(provider, "execute_leg", None)):
                raise ExpertSourceReplayExecutionError(
                    "source replay provider must implement matched-leg execution"
                )
            if advertised_key in providers_by_key:
                raise ExpertSourceReplayExecutionError(
                    "source replay provider dispatch key is duplicated"
                )
            providers_by_key[advertised_key] = provider
        self._providers_by_key: Mapping[
            ExpertSourceReplayExecutionProviderKey,
            ExpertSourceReplayExecutionProvider,
        ] = MappingProxyType(providers_by_key)

    def resolve_all(
        self,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> tuple[ResolvedExpertSourceReplayExecutionCase, ...]:
        if not isinstance(prepared_request, PreparedExpertSourceReplayRequest):
            raise ExpertSourceReplayExecutionError(
                "source replay dispatch requires a prepared request"
            )
        prepared = PreparedExpertSourceReplayRequest(
            request=prepared_request.request,
            settings=prepared_request.settings,
            attempt=prepared_request.attempt,
            selection=prepared_request.selection,
            candidate=prepared_request.candidate,
            parent=prepared_request.parent,
            authorization_state=prepared_request.authorization_state,
            cases=prepared_request.cases,
        )
        case_keys = tuple(
            expert_source_replay_execution_provider_key(materialized_case)
            for materialized_case in prepared.cases
        )
        return self._resolve_cases(prepared.cases, case_keys)

    def _resolve_cases(
        self,
        materialized_cases: tuple[MaterializedExpertSourceReplayCase, ...],
        case_keys: tuple[ExpertSourceReplayExecutionProviderKey, ...],
    ) -> tuple[ResolvedExpertSourceReplayExecutionCase, ...]:
        if (
            not isinstance(materialized_cases, tuple)
            or not isinstance(case_keys, tuple)
            or len(materialized_cases) != len(case_keys)
        ):
            raise ExpertSourceReplayExecutionError(
                "source replay cases and provider keys must be aligned tuples"
            )
        missing_identities = tuple(
            sorted(
                {key.identity for key in case_keys if key not in self._providers_by_key}
            )
        )
        if missing_identities:
            raise ExpertSourceReplayExecutionError(
                "source replay execution provider keys are unsupported: "
                f"{missing_identities}"
            )
        return tuple(
            ResolvedExpertSourceReplayExecutionCase(
                materialized_case=materialized_case,
                dispatch_key=key,
                provider=self._providers_by_key[key],
            )
            for materialized_case, key in zip(
                materialized_cases,
                case_keys,
                strict=True,
            )
        )
