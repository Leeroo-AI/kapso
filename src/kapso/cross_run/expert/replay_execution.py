"""Exact execution-provider dispatch for expert source replay."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from kapso.cross_run.canonical import require_identifier
from kapso.cross_run.contracts import (
    ExpertSourceReplayExecutionLeg,
    ExpertSourceReplayExecutionLegKind,
    StrictContract,
)
from kapso.cross_run.expert.replay_protocol import build_task_evaluator_request
from kapso.cross_run.expert.replay_protocol_contracts import (
    TaskEvaluatorInvocationAllocation,
    TaskEvaluatorRequest,
)
from kapso.cross_run.expert.replay_request import (
    MaterializedExpertSourceReplayCase,
    PreparedExpertSourceReplayRequest,
    VerifiedExpertSourceReplayCandidate,
    VerifiedExpertSourceReplayParent,
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
    execution_provider_settings_digest: str
    sandbox_policy_version: str
    task_adapter_runtime_protocol_version: str
    task_evaluator_protocol_version: str

    def _validate(self) -> None:
        for value, name in zip(self.identity, self.field_names, strict=True):
            if name == "execution_provider_settings_digest":
                if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                    raise ExpertSourceReplayExecutionError(
                        "source replay provider settings digest is invalid"
                    )
            else:
                require_identifier(value, f"source replay provider key {name}")

    @property
    def identity(self) -> tuple[str, ...]:
        return (
            self.paired_execution_protocol_version,
            self.execution_provider_id,
            self.execution_provider_version,
            self.execution_provider_settings_digest,
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
            "execution_provider_settings_digest",
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

    def cleanup_interrupted(
        self,
        provider_handle: SourceReplayProviderExecutionHandle,
    ) -> None:
        """Idempotently remove daemon resources without executing the leg."""


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
        execution_provider_settings_digest=(compute.execution_provider_settings_digest),
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
    expert_source: (
        VerifiedExpertSourceReplayCandidate | VerifiedExpertSourceReplayParent
    )
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
            not _expert_source_matches_leg(
                self.materialized_case,
                self.invocation_allocation,
                self.expert_source,
            )
            or not isinstance(
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


def _expert_source_matches_leg(
    materialized_case: MaterializedExpertSourceReplayCase,
    invocation_allocation: TaskEvaluatorInvocationAllocation,
    expert_source: (
        VerifiedExpertSourceReplayCandidate | VerifiedExpertSourceReplayParent
    ),
) -> bool:
    if not isinstance(
        invocation_allocation,
        TaskEvaluatorInvocationAllocation,
    ):
        return False
    request_case = materialized_case.request_case
    if (
        invocation_allocation.execution_case_id != request_case.execution_case_id
        or invocation_allocation.execution_leg_id
        not in {
            request_case.control_leg.execution_leg_id,
            request_case.candidate_leg.execution_leg_id,
        }
    ):
        return False
    if (
        invocation_allocation.execution_leg_id
        == request_case.control_leg.execution_leg_id
    ):
        leg = request_case.control_leg
    else:
        leg = request_case.candidate_leg
    return _expert_source_matches_execution_leg(leg, expert_source)


def _expert_source_matches_execution_leg(
    execution_leg: ExpertSourceReplayExecutionLeg,
    expert_source: (
        VerifiedExpertSourceReplayCandidate | VerifiedExpertSourceReplayParent
    ),
) -> bool:
    if (
        execution_leg.kind is ExpertSourceReplayExecutionLegKind.CONTROL_PARENT
        and type(expert_source) is VerifiedExpertSourceReplayParent
    ):
        return (
            expert_source.release_manifest.release_id
            == execution_leg.expert_artifact_id
            and expert_source.parent_tree_receipt.parent_tree_receipt_id
            == execution_leg.expert_source_receipt_id
            and expert_source.parent_tree_receipt.parent_tree_hash
            == execution_leg.expert_tree_hash
        )
    if (
        execution_leg.kind is ExpertSourceReplayExecutionLegKind.CANDIDATE
        and type(expert_source) is VerifiedExpertSourceReplayCandidate
    ):
        return (
            expert_source.manifest.candidate_id == execution_leg.expert_artifact_id
            and expert_source.commit_record.commit_record_id
            == execution_leg.expert_source_receipt_id
            and expert_source.source_tree.tree_hash == execution_leg.expert_tree_hash
        )
    return False


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
    _candidate: VerifiedExpertSourceReplayCandidate = field(
        repr=False,
        compare=False,
    )
    _parent: VerifiedExpertSourceReplayParent = field(
        repr=False,
        compare=False,
    )
    _provider: ExpertSourceReplayExecutionProvider = field(repr=False, compare=False)

    def __init__(
        self,
        materialized_case: MaterializedExpertSourceReplayCase,
        dispatch_key: ExpertSourceReplayExecutionProviderKey,
        candidate: VerifiedExpertSourceReplayCandidate,
        parent: VerifiedExpertSourceReplayParent,
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
        if not callable(getattr(provider, "cleanup_interrupted", None)):
            raise ExpertSourceReplayExecutionError(
                "resolved source replay provider cannot clean an interrupted leg"
            )
        if (
            type(candidate) is not VerifiedExpertSourceReplayCandidate
            or type(parent) is not VerifiedExpertSourceReplayParent
        ):
            raise ExpertSourceReplayExecutionError(
                "resolved source replay execution requires exact expert sources"
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
        object.__setattr__(self, "_candidate", candidate)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_provider", provider)
        self._require_bound_expert_sources()
        self.require_current_provider_identity()

    def _require_bound_expert_sources(self) -> None:
        request_case = self.materialized_case.request_case
        for leg, source in (
            (request_case.control_leg, self._parent),
            (request_case.candidate_leg, self._candidate),
        ):
            if not _expert_source_matches_execution_leg(leg, source):
                raise ExpertSourceReplayExecutionError(
                    "resolved source replay expert source differs from its leg"
                )

    def require_exact_prepared_authority(
        self,
        prepared_request: PreparedExpertSourceReplayRequest,
    ) -> None:
        if (
            not isinstance(prepared_request, PreparedExpertSourceReplayRequest)
            or self._candidate != prepared_request.candidate
            or self._parent != prepared_request.parent
            or self.materialized_case not in prepared_request.cases
        ):
            raise ExpertSourceReplayExecutionError(
                "resolved source replay execution differs from prepared authority"
            )

    def _expert_source_for(
        self,
        invocation_allocation: TaskEvaluatorInvocationAllocation,
    ) -> VerifiedExpertSourceReplayCandidate | VerifiedExpertSourceReplayParent:
        request_case = self.materialized_case.request_case
        source = (
            self._parent
            if invocation_allocation.execution_leg_id
            == request_case.control_leg.execution_leg_id
            else self._candidate
        )
        if not _expert_source_matches_leg(
            self.materialized_case,
            invocation_allocation,
            source,
        ):
            raise ExpertSourceReplayExecutionError(
                "source replay allocation has no exact expert source"
            )
        return source

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
        if (
            type(invocation) is not ExpertSourceReplayMatchedLegInvocation
            or invocation.materialized_case != self.materialized_case
            or invocation.expert_source
            != self._expert_source_for(invocation.invocation_allocation)
        ):
            raise ExpertSourceReplayExecutionError(
                "source replay invocation differs from its resolved authority"
            )
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
            if not callable(getattr(provider, "cleanup_interrupted", None)):
                raise ExpertSourceReplayExecutionError(
                    "source replay provider must implement interrupted cleanup"
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
        return self._resolve_cases(
            prepared.cases,
            case_keys,
            prepared.candidate,
            prepared.parent,
        )

    def cleanup_interrupted(
        self,
        provider_handle: SourceReplayProviderExecutionHandle,
    ) -> None:
        if type(provider_handle) is not SourceReplayProviderExecutionHandle:
            raise ExpertSourceReplayExecutionError(
                "source replay cleanup requires an exact provider handle"
            )
        provider = self._providers_by_key.get(provider_handle.dispatch_key)
        if provider is None:
            raise ExpertSourceReplayExecutionError(
                "source replay interrupted provider key is unsupported"
            )
        if provider.dispatch_key != provider_handle.dispatch_key:
            raise ExpertSourceReplayExecutionError(
                "source replay interrupted provider identity changed"
            )
        cleanup_interrupted = getattr(provider, "cleanup_interrupted", None)
        if not callable(cleanup_interrupted):
            raise ExpertSourceReplayExecutionError(
                "source replay provider has no interrupted cleanup"
            )
        cleanup_interrupted(provider_handle)
        if provider.dispatch_key != provider_handle.dispatch_key:
            raise ExpertSourceReplayExecutionError(
                "source replay interrupted provider identity changed"
            )

    def _resolve_cases(
        self,
        materialized_cases: tuple[MaterializedExpertSourceReplayCase, ...],
        case_keys: tuple[ExpertSourceReplayExecutionProviderKey, ...],
        candidate: VerifiedExpertSourceReplayCandidate,
        parent: VerifiedExpertSourceReplayParent,
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
                candidate=candidate,
                parent=parent,
                provider=self._providers_by_key[key],
            )
            for materialized_case, key in zip(
                materialized_cases,
                case_keys,
                strict=True,
            )
        )
