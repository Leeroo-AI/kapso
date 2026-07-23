"""Exact provider dispatch for prepared task-evaluation cases."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping, Protocol

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import (
    EvaluationFingerprint,
    StrictContract,
    TaskAdapterRuntimeContract,
    TaskContextBinding,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationComputeBinding,
    TaskEvaluationExpertLeg,
    TaskEvaluationInvocationAllocation,
    TaskEvaluationLegKind,
)
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationAdapterRuntime,
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationSourceBase,
    VerifiedTaskEvaluationStartingArtifact,
)
from kapso.cross_run.expert.task_evaluation_preflight import (
    MaterializedTaskEvaluationCase,
    PreparedTaskEvaluationRequest,
)
from kapso.cross_run.expert.task_evaluation_protocol import (
    build_task_evaluation_evaluator_request,
)
from kapso.cross_run.expert.task_evaluation_reservation import (
    ExpertTaskEvaluationReservationSnapshot,
)
from kapso.cross_run.expert.task_evaluator_protocol import TaskEvaluatorRequest
from kapso.cross_run.process import BoundedProcessResult

_MISSING_PROVIDER_ATTRIBUTE = object()
_RESOLVED_CASE_SEAL = object()
_TASK_EVALUATION_INVOCATION_SEAL = object()


class TaskEvaluationExecutionError(ValueError):
    """Task-evaluation provider dispatch is absent, ambiguous, or stale."""


@dataclass(frozen=True)
class TaskEvaluationExecutionProviderKey(StrictContract):
    execution_protocol_version: str
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
                    raise TaskEvaluationExecutionError(
                        "task evaluation provider settings digest is invalid"
                    )
            else:
                require_identifier(value, f"task evaluation provider key {name}")

    @property
    def identity(self) -> tuple[str, ...]:
        return (
            self.execution_protocol_version,
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
            "execution_protocol_version",
            "execution_provider_id",
            "execution_provider_version",
            "execution_provider_settings_digest",
            "sandbox_policy_version",
            "task_adapter_runtime_protocol_version",
            "task_evaluator_protocol_version",
        )


@dataclass(frozen=True)
class TaskEvaluationProviderSupportRequirements:
    """Minimal non-scientific authority for deterministic provider admission."""

    dispatch_key: TaskEvaluationExecutionProviderKey
    runtime_contract: TaskAdapterRuntimeContract
    task_evaluator_executable_path: str
    leg_wall_time_limit_seconds: int
    termination_grace_seconds: int
    cpu_millicore_limit: int
    memory_byte_limit: int
    shared_memory_byte_limit: int
    process_limit: int
    open_file_limit: int
    writable_inode_limit: int
    writable_storage_byte_limit: int
    output_entry_limit: int
    output_byte_limit: int
    stdout_byte_limit: int
    stderr_byte_limit: int
    accelerator_class_id: str | None
    accelerator_count: int

    def __post_init__(self) -> None:
        executable_path = PurePosixPath(self.task_evaluator_executable_path)
        positive_limits = (
            self.leg_wall_time_limit_seconds,
            self.termination_grace_seconds,
            self.cpu_millicore_limit,
            self.memory_byte_limit,
            self.shared_memory_byte_limit,
            self.process_limit,
            self.open_file_limit,
            self.writable_inode_limit,
            self.writable_storage_byte_limit,
            self.output_entry_limit,
            self.output_byte_limit,
            self.stdout_byte_limit,
            self.stderr_byte_limit,
        )
        if (
            type(self.dispatch_key) is not TaskEvaluationExecutionProviderKey
            or type(self.runtime_contract) is not TaskAdapterRuntimeContract
            or not isinstance(self.task_evaluator_executable_path, str)
            or executable_path.is_absolute()
            or not executable_path.parts
            or ".." in executable_path.parts
            or str(executable_path) != self.task_evaluator_executable_path
            or any(type(value) is not int or value <= 0 for value in positive_limits)
            or self.termination_grace_seconds > self.leg_wall_time_limit_seconds
            or self.shared_memory_byte_limit > self.memory_byte_limit
            or self.output_entry_limit >= self.writable_inode_limit
            or self.output_byte_limit > self.writable_storage_byte_limit
            or type(self.accelerator_count) is not int
            or self.accelerator_count < 0
            or (self.accelerator_class_id is None) != (self.accelerator_count == 0)
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation provider support requirements are invalid"
            )
        if self.accelerator_class_id is not None:
            require_identifier(
                self.accelerator_class_id,
                "task evaluation provider accelerator class",
            )


@dataclass(frozen=True)
class ExecutableTaskEvaluationLeg:
    authority: TaskEvaluationExpertLeg
    expert_source: VerifiedTaskEvaluationCandidate | VerifiedTaskEvaluationSourceBase

    def __post_init__(self) -> None:
        if type(self.authority) is not TaskEvaluationExpertLeg or not (
            _task_evaluation_source_matches_leg(self.authority, self.expert_source)
        ):
            raise TaskEvaluationExecutionError(
                "executable task-evaluation leg differs from its expert source"
            )


@dataclass(frozen=True)
class ExecutableTaskEvaluationCase:
    evaluation_case_id: str
    compute_binding: TaskEvaluationComputeBinding
    task_context_binding: TaskContextBinding
    evaluation_fingerprints: tuple[EvaluationFingerprint, ...]
    adapter_runtime: VerifiedTaskEvaluationAdapterRuntime
    starting_artifacts: tuple[VerifiedTaskEvaluationStartingArtifact, ...]
    legs: tuple[ExecutableTaskEvaluationLeg, ...]

    def __post_init__(self) -> None:
        require_content_id(
            self.evaluation_case_id,
            "executable task evaluation case ID",
        )
        if (
            self.evaluation_case_id.split(":sha256:", 1)[0] != "task-evaluation-case"
            or type(self.compute_binding) is not TaskEvaluationComputeBinding
            or type(self.task_context_binding) is not TaskContextBinding
            or type(self.evaluation_fingerprints) is not tuple
            or not self.evaluation_fingerprints
            or any(
                type(fingerprint) is not EvaluationFingerprint
                for fingerprint in self.evaluation_fingerprints
            )
            or type(self.adapter_runtime) is not VerifiedTaskEvaluationAdapterRuntime
            or type(self.starting_artifacts) is not tuple
            or any(
                type(artifact) is not VerifiedTaskEvaluationStartingArtifact
                for artifact in self.starting_artifacts
            )
            or type(self.legs) is not tuple
            or not self.legs
            or any(type(leg) is not ExecutableTaskEvaluationLeg for leg in self.legs)
        ):
            raise TaskEvaluationExecutionError(
                "executable task-evaluation case is not an exact typed projection"
            )
        fingerprint_ids = tuple(
            fingerprint.evaluation_fingerprint_id
            for fingerprint in self.evaluation_fingerprints
        )
        artifact_ids = tuple(
            artifact.artifact.starting_artifact_content_id
            for artifact in self.starting_artifacts
        )
        leg_ids = tuple(leg.authority.leg_id for leg in self.legs)
        leg_kinds = tuple(leg.authority.kind for leg in self.legs)
        manifest = self.adapter_runtime.manifest
        if (
            fingerprint_ids != tuple(sorted(set(fingerprint_ids)))
            or artifact_ids != tuple(sorted(set(artifact_ids)))
            or leg_ids != tuple(sorted(set(leg_ids)))
            or set(leg_kinds) != set(self.compute_binding.leg_order)
            or len(leg_kinds) != len(set(leg_kinds))
            or self.task_context_binding.scope_contract_id != manifest.scope_contract_id
            or self.task_context_binding.task_family_id != manifest.task_family_id
            or self.task_context_binding.task_adapter_id != manifest.task_adapter_id
            or not set(manifest.context_binding.consumed_dimension_ids).issubset(
                self.task_context_binding.transfer_dimensions
            )
        ):
            raise TaskEvaluationExecutionError(
                "executable task-evaluation case closure is inconsistent"
            )

    @property
    def provider_key(self) -> TaskEvaluationExecutionProviderKey:
        compute = self.compute_binding
        manifest = self.adapter_runtime.manifest
        return TaskEvaluationExecutionProviderKey(
            execution_protocol_version=compute.execution_protocol_version,
            execution_provider_id=compute.execution_provider_id,
            execution_provider_version=compute.execution_provider_version,
            execution_provider_settings_digest=(
                compute.execution_provider_settings_digest
            ),
            sandbox_policy_version=compute.sandbox_policy_version,
            task_adapter_runtime_protocol_version=(
                manifest.runtime.runtime_protocol_version
            ),
            task_evaluator_protocol_version=manifest.task_evaluator.protocol_version,
        )


class TaskEvaluationExecutionProvider(Protocol):
    """An isolated provider registered under one exact implementation key."""

    dispatch_key: TaskEvaluationExecutionProviderKey

    def require_supported_execution(
        self,
        requirements: TaskEvaluationProviderSupportRequirements,
    ) -> None:
        """Reject deterministic provider incompatibility before reservation."""

    def execute_leg(
        self,
        invocation: TaskEvaluationLegInvocation,
    ) -> TaskEvaluationProviderCompletion:
        """Execute one exact journal-owned scientific leg without retry."""

    def cleanup_interrupted(
        self,
        provider_handle: TaskEvaluationProviderExecutionHandle,
    ) -> None:
        """Idempotently remove provider resources without executing the leg."""


@dataclass(frozen=True)
class TaskEvaluationProviderExecutionHandle(StrictContract):
    provider_handle_id: str
    dispatch_key: TaskEvaluationExecutionProviderKey
    invocation_allocation: TaskEvaluationInvocationAllocation

    CONTENT_NAMESPACE = "task-evaluation-provider-execution-handle"
    IDENTITY_FIELD = "provider_handle_id"


def task_evaluation_provider_execution_handle(
    dispatch_key: TaskEvaluationExecutionProviderKey,
    invocation_allocation: TaskEvaluationInvocationAllocation,
) -> TaskEvaluationProviderExecutionHandle:
    if (
        type(dispatch_key) is not TaskEvaluationExecutionProviderKey
        or type(invocation_allocation) is not TaskEvaluationInvocationAllocation
    ):
        raise TaskEvaluationExecutionError(
            "task evaluation provider handle requires exact dispatch and allocation"
        )
    return TaskEvaluationProviderExecutionHandle.mint(
        dispatch_key=dispatch_key,
        invocation_allocation=invocation_allocation,
    )


@dataclass(frozen=True, slots=True, init=False)
class TaskEvaluationLegInvocation:
    """Provider input constructible only by the journal-owned spawn path."""

    selected_leg: ExecutableTaskEvaluationLeg
    execution_requirements: TaskEvaluationProviderSupportRequirements
    task_context_binding: TaskContextBinding
    evaluation_fingerprints: tuple[EvaluationFingerprint, ...]
    adapter_runtime: VerifiedTaskEvaluationAdapterRuntime
    starting_artifacts: tuple[VerifiedTaskEvaluationStartingArtifact, ...]
    invocation_allocation: TaskEvaluationInvocationAllocation
    task_evaluator_request: TaskEvaluatorRequest
    provider_handle: TaskEvaluationProviderExecutionHandle

    def __init__(
        self,
        seal: object,
        *,
        registry: TaskEvaluationExecutionProviderRegistry,
        prepared_request: PreparedTaskEvaluationRequest,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        resolved_case: ResolvedTaskEvaluationCase,
        invocation_allocation: TaskEvaluationInvocationAllocation,
    ) -> None:
        if (
            seal is not _TASK_EVALUATION_INVOCATION_SEAL
            or type(registry) is not TaskEvaluationExecutionProviderRegistry
            or type(resolved_case) is not ResolvedTaskEvaluationCase
            or type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot
            or type(invocation_allocation) is not TaskEvaluationInvocationAllocation
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation invocation requires sealed journal authority"
            )
        expected_resolved_case = registry._resolved_case_for_allocation(
            prepared_request=prepared_request,
            reservation_snapshot=reservation_snapshot,
            invocation_allocation=invocation_allocation,
        )
        if resolved_case is not expected_resolved_case:
            raise TaskEvaluationExecutionError(
                "task evaluation invocation names a foreign provider resolution"
            )
        executable_case = resolved_case.executable_case
        matching_legs = tuple(
            leg
            for leg in executable_case.legs
            if leg.authority.leg_id == invocation_allocation.evaluation_leg_id
        )
        if (
            reservation_snapshot.reservation.reservation_id
            != invocation_allocation.reservation_id
            or executable_case.evaluation_case_id
            != invocation_allocation.evaluation_case_id
            or len(matching_legs) != 1
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation invocation differs from its reserved case and leg"
            )
        task_evaluator_request = build_task_evaluation_evaluator_request(
            prepared_request,
            reservation_snapshot,
            invocation_allocation,
        )
        provider_handle = task_evaluation_provider_execution_handle(
            resolved_case.dispatch_key,
            invocation_allocation,
        )
        object.__setattr__(self, "selected_leg", matching_legs[0])
        object.__setattr__(
            self,
            "execution_requirements",
            _task_evaluation_provider_support_requirements(executable_case),
        )
        object.__setattr__(
            self,
            "task_context_binding",
            executable_case.task_context_binding,
        )
        object.__setattr__(
            self,
            "evaluation_fingerprints",
            executable_case.evaluation_fingerprints,
        )
        object.__setattr__(self, "adapter_runtime", executable_case.adapter_runtime)
        object.__setattr__(
            self,
            "starting_artifacts",
            executable_case.starting_artifacts,
        )
        object.__setattr__(self, "invocation_allocation", invocation_allocation)
        object.__setattr__(self, "task_evaluator_request", task_evaluator_request)
        object.__setattr__(self, "provider_handle", provider_handle)


@dataclass(frozen=True)
class TaskEvaluationProviderCompletion:
    provider_handle_id: str
    process_result: BoundedProcessResult
    result_payload: bytes | None

    def __post_init__(self) -> None:
        if (
            require_content_id(
                self.provider_handle_id,
                "task evaluation provider completion handle",
            ).split(":sha256:", 1)[0]
            != TaskEvaluationProviderExecutionHandle.CONTENT_NAMESPACE
            or type(self.process_result) is not BoundedProcessResult
            or (
                self.result_payload is not None
                and not isinstance(self.result_payload, bytes)
            )
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation provider completion is invalid"
            )


@dataclass(frozen=True, slots=True, init=False)
class ResolvedTaskEvaluationCase:
    """Runtime-only exact provider binding with no public execution surface."""

    executable_case: ExecutableTaskEvaluationCase
    dispatch_key: TaskEvaluationExecutionProviderKey
    _provider: TaskEvaluationExecutionProvider = field(repr=False, compare=False)

    def __init__(
        self,
        seal: object,
        executable_case: ExecutableTaskEvaluationCase,
        provider: TaskEvaluationExecutionProvider,
    ) -> None:
        if (
            seal is not _RESOLVED_CASE_SEAL
            or type(executable_case) is not ExecutableTaskEvaluationCase
            or not callable(getattr(provider, "require_supported_execution", None))
            or not callable(getattr(provider, "execute_leg", None))
            or not callable(getattr(provider, "cleanup_interrupted", None))
        ):
            raise TaskEvaluationExecutionError(
                "resolved task evaluation requires exact case and provider authority"
            )
        dispatch_key = executable_case.provider_key
        if (
            getattr(provider, "dispatch_key", _MISSING_PROVIDER_ATTRIBUTE)
            != dispatch_key
        ):
            raise TaskEvaluationExecutionError(
                "resolved task evaluation provider key differs from its case"
            )
        object.__setattr__(self, "executable_case", executable_case)
        object.__setattr__(self, "dispatch_key", dispatch_key)
        object.__setattr__(self, "_provider", provider)
        self.require_current_provider_identity()

    def require_current_provider_identity(self) -> None:
        if (
            getattr(self._provider, "dispatch_key", _MISSING_PROVIDER_ATTRIBUTE)
            != self.dispatch_key
        ):
            raise TaskEvaluationExecutionError(
                "resolved task evaluation provider identity changed"
            )


class TaskEvaluationExecutionProviderRegistry:
    """Bind and resolve one prepared request through exact full-key equality."""

    def __init__(
        self,
        prepared_request: PreparedTaskEvaluationRequest,
        providers: tuple[TaskEvaluationExecutionProvider, ...],
    ) -> None:
        prepared = _reconstruct_prepared_request(prepared_request)
        executable_cases = project_prepared_task_evaluation_cases(prepared)
        if type(providers) is not tuple or not providers:
            raise TaskEvaluationExecutionError(
                "task evaluation execution providers must be a non-empty tuple"
            )
        providers_by_key: dict[
            TaskEvaluationExecutionProviderKey,
            TaskEvaluationExecutionProvider,
        ] = {}
        for provider in providers:
            advertised_key = getattr(
                provider,
                "dispatch_key",
                _MISSING_PROVIDER_ATTRIBUTE,
            )
            if type(advertised_key) is not TaskEvaluationExecutionProviderKey:
                raise TaskEvaluationExecutionError(
                    "task evaluation provider must advertise an exact dispatch key"
                )
            for method_name in (
                "require_supported_execution",
                "execute_leg",
                "cleanup_interrupted",
            ):
                if not callable(getattr(provider, method_name, None)):
                    raise TaskEvaluationExecutionError(
                        f"task evaluation provider lacks {method_name}"
                    )
            if advertised_key in providers_by_key:
                raise TaskEvaluationExecutionError(
                    "task evaluation provider dispatch key is duplicated"
                )
            providers_by_key[advertised_key] = provider
        self._providers_by_key: Mapping[
            TaskEvaluationExecutionProviderKey,
            TaskEvaluationExecutionProvider,
        ] = MappingProxyType(providers_by_key)
        self._prepared_request = prepared
        self._executable_cases = executable_cases
        required_provider_keys = {case.provider_key for case in executable_cases}
        if required_provider_keys.issubset(providers_by_key) and (
            set(providers_by_key) != required_provider_keys
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation providers differ from the exact required key set"
            )
        missing_identities = tuple(
            sorted(
                {
                    case.provider_key.identity
                    for case in executable_cases
                    if case.provider_key not in self._providers_by_key
                }
            )
        )
        if missing_identities:
            raise TaskEvaluationExecutionError(
                "task evaluation execution provider keys are unsupported: "
                f"{missing_identities}"
            )
        resolved_cases = []
        for executable_case in executable_cases:
            provider = self._providers_by_key[executable_case.provider_key]
            provider.require_supported_execution(
                _task_evaluation_provider_support_requirements(executable_case)
            )
            if provider.dispatch_key != executable_case.provider_key:
                raise TaskEvaluationExecutionError(
                    "task evaluation provider identity changed during support check"
                )
            resolved_cases.append(
                ResolvedTaskEvaluationCase(
                    seal=_RESOLVED_CASE_SEAL,
                    executable_case=executable_case,
                    provider=provider,
                )
            )
        self._resolved_cases = tuple(resolved_cases)

    def require_exact_prepared_authority(
        self,
        prepared_request: PreparedTaskEvaluationRequest,
    ) -> None:
        if _reconstruct_prepared_request(prepared_request) != self._prepared_request:
            raise TaskEvaluationExecutionError(
                "task evaluation registry differs from prepared authority"
            )

    def resolve_all(self) -> tuple[ResolvedTaskEvaluationCase, ...]:
        for resolved_case in self._resolved_cases:
            self._require_owned_resolved_case(resolved_case)
        return self._resolved_cases

    def _resolved_case_for_allocation(
        self,
        *,
        prepared_request: PreparedTaskEvaluationRequest,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        invocation_allocation: TaskEvaluationInvocationAllocation,
    ) -> ResolvedTaskEvaluationCase:
        """Resolve one journal allocation by its exact case-scoped leg authority."""

        self.require_exact_prepared_authority(prepared_request)
        if (
            type(reservation_snapshot) is not ExpertTaskEvaluationReservationSnapshot
            or type(invocation_allocation) is not TaskEvaluationInvocationAllocation
            or reservation_snapshot.request
            != self._prepared_request.plan_join.request
            or reservation_snapshot.plan_reservation
            != self._prepared_request.plan_join.plan_reservation
            or invocation_allocation.reservation_id
            != reservation_snapshot.reservation.reservation_id
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation allocation differs from its exact reservation"
            )
        matching_resolutions = tuple(
            resolved_case
            for resolved_case in self._resolved_cases
            if resolved_case.executable_case.evaluation_case_id
            == invocation_allocation.evaluation_case_id
            and any(
                leg.authority.leg_id == invocation_allocation.evaluation_leg_id
                for leg in resolved_case.executable_case.legs
            )
        )
        if len(matching_resolutions) != 1:
            raise TaskEvaluationExecutionError(
                "task evaluation allocation has no exact resolved case and leg"
            )
        resolved_case = matching_resolutions[0]
        self._require_owned_resolved_case(resolved_case)
        return resolved_case

    def _execute_journal_leg(
        self,
        *,
        prepared_request: PreparedTaskEvaluationRequest,
        reservation_snapshot: ExpertTaskEvaluationReservationSnapshot,
        resolved_case: ResolvedTaskEvaluationCase,
        invocation_allocation: TaskEvaluationInvocationAllocation,
    ) -> TaskEvaluationProviderCompletion:
        """Execute exactly once through one registry-owned journal resolution."""

        expected_resolved_case = self._resolved_case_for_allocation(
            prepared_request=prepared_request,
            reservation_snapshot=reservation_snapshot,
            invocation_allocation=invocation_allocation,
        )
        if resolved_case is not expected_resolved_case:
            raise TaskEvaluationExecutionError(
                "task evaluation journal execution names a foreign resolution"
            )
        invocation = TaskEvaluationLegInvocation(
            _TASK_EVALUATION_INVOCATION_SEAL,
            registry=self,
            prepared_request=prepared_request,
            reservation_snapshot=reservation_snapshot,
            resolved_case=resolved_case,
            invocation_allocation=invocation_allocation,
        )
        provider = resolved_case._provider
        execute_leg = getattr(provider, "execute_leg", None)
        if not callable(execute_leg):
            raise TaskEvaluationExecutionError(
                "task evaluation provider has no exact leg executor"
            )
        current_resolved_case = self._resolved_case_for_allocation(
            prepared_request=prepared_request,
            reservation_snapshot=reservation_snapshot,
            invocation_allocation=invocation_allocation,
        )
        if current_resolved_case is not resolved_case:
            raise TaskEvaluationExecutionError(
                "task evaluation journal resolution changed before execution"
            )
        completion = execute_leg(invocation)
        current_resolved_case = self._resolved_case_for_allocation(
            prepared_request=prepared_request,
            reservation_snapshot=reservation_snapshot,
            invocation_allocation=invocation_allocation,
        )
        if current_resolved_case is not resolved_case:
            raise TaskEvaluationExecutionError(
                "task evaluation journal resolution changed during execution"
            )
        if (
            type(completion) is not TaskEvaluationProviderCompletion
            or completion.provider_handle_id
            != invocation.provider_handle.provider_handle_id
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation provider returned a foreign completion"
            )
        return completion

    def _require_owned_resolved_case(
        self,
        resolved_case: ResolvedTaskEvaluationCase,
    ) -> None:
        if (
            type(resolved_case) is not ResolvedTaskEvaluationCase
            or not any(
                resolved_case is owned_case for owned_case in self._resolved_cases
            )
            or self._providers_by_key.get(resolved_case.dispatch_key)
            is not resolved_case._provider
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation provider resolution is not owned by this registry"
            )
        resolved_case.require_current_provider_identity()

    def cleanup_interrupted(
        self,
        provider_handle: TaskEvaluationProviderExecutionHandle,
    ) -> None:
        if type(provider_handle) is not TaskEvaluationProviderExecutionHandle:
            raise TaskEvaluationExecutionError(
                "task evaluation cleanup requires an exact provider handle"
            )
        allocation = provider_handle.invocation_allocation
        matching_cases = tuple(
            executable_case
            for executable_case in self._executable_cases
            if executable_case.evaluation_case_id == allocation.evaluation_case_id
            and any(
                leg.authority.leg_id == allocation.evaluation_leg_id
                for leg in executable_case.legs
            )
        )
        if (
            len(matching_cases) != 1
            or provider_handle.dispatch_key != matching_cases[0].provider_key
        ):
            raise TaskEvaluationExecutionError(
                "task evaluation cleanup handle differs from the bound request"
            )
        provider = self._providers_by_key.get(provider_handle.dispatch_key)
        if provider is None or provider.dispatch_key != provider_handle.dispatch_key:
            raise TaskEvaluationExecutionError(
                "task evaluation interrupted provider key is unavailable or changed"
            )
        provider.cleanup_interrupted(provider_handle)
        if provider.dispatch_key != provider_handle.dispatch_key:
            raise TaskEvaluationExecutionError(
                "task evaluation interrupted provider identity changed"
            )


def project_prepared_task_evaluation_cases(
    prepared_request: PreparedTaskEvaluationRequest,
) -> tuple[ExecutableTaskEvaluationCase, ...]:
    """Erase matrix provenance while retaining every provider input authority."""

    prepared = _reconstruct_prepared_request(prepared_request)
    return tuple(
        _project_materialized_case(prepared, materialized_case)
        for materialized_case in prepared.cases
    )


def _task_evaluation_provider_support_requirements(
    executable_case: ExecutableTaskEvaluationCase,
) -> TaskEvaluationProviderSupportRequirements:
    compute = executable_case.compute_binding
    manifest = executable_case.adapter_runtime.manifest
    return TaskEvaluationProviderSupportRequirements(
        dispatch_key=executable_case.provider_key,
        runtime_contract=manifest.runtime,
        task_evaluator_executable_path=manifest.task_evaluator.executable_path,
        leg_wall_time_limit_seconds=compute.leg_wall_time_limit_seconds,
        termination_grace_seconds=compute.termination_grace_seconds,
        cpu_millicore_limit=compute.cpu_millicore_limit,
        memory_byte_limit=compute.memory_byte_limit,
        shared_memory_byte_limit=compute.shared_memory_byte_limit,
        process_limit=compute.process_limit,
        open_file_limit=compute.open_file_limit,
        writable_inode_limit=compute.writable_inode_limit,
        writable_storage_byte_limit=compute.writable_storage_byte_limit,
        output_entry_limit=compute.output_entry_limit,
        output_byte_limit=compute.output_byte_limit,
        stdout_byte_limit=compute.stdout_byte_limit,
        stderr_byte_limit=compute.stderr_byte_limit,
        accelerator_class_id=compute.accelerator_class_id,
        accelerator_count=compute.accelerator_count,
    )


def _project_materialized_case(
    prepared: PreparedTaskEvaluationRequest,
    materialized_case: MaterializedTaskEvaluationCase,
) -> ExecutableTaskEvaluationCase:
    request_case = materialized_case.request_case
    signed_cases = tuple(
        signed_case
        for signed_case in materialized_case.adapter.manifest.release_matrix_cases
        if signed_case.release_matrix_case_id == request_case.release_matrix_case_id
    )
    if len(signed_cases) != 1:
        raise TaskEvaluationExecutionError(
            "task evaluation executable case lacks unique signed authority"
        )
    signed_case = signed_cases[0]
    sources_by_kind: dict[
        TaskEvaluationLegKind,
        VerifiedTaskEvaluationCandidate | VerifiedTaskEvaluationSourceBase,
    ] = {TaskEvaluationLegKind.CANDIDATE: prepared.candidate}
    if prepared.source_base is not None:
        sources_by_kind[TaskEvaluationLegKind.SOURCE_BASE_CONTROL] = prepared.source_base
    if (
        signed_case.evaluation_fingerprint_ids
        != request_case.evaluation_fingerprint_ids
        or signed_case.starting_artifact_ids != request_case.starting_artifact_ids
        or signed_case.task_context_binding.task_context_binding_id
        != request_case.task_context_binding_id
        or set(sources_by_kind) != {leg.kind for leg in request_case.legs}
    ):
        raise TaskEvaluationExecutionError(
            "task evaluation executable projection differs from prepared authority"
        )
    return ExecutableTaskEvaluationCase(
        evaluation_case_id=request_case.evaluation_case_id,
        compute_binding=request_case.compute_binding,
        task_context_binding=signed_case.task_context_binding,
        evaluation_fingerprints=signed_case.evaluation_fingerprints,
        adapter_runtime=materialized_case.adapter_runtime,
        starting_artifacts=materialized_case.starting_artifacts,
        legs=tuple(
            ExecutableTaskEvaluationLeg(
                authority=leg,
                expert_source=sources_by_kind[leg.kind],
            )
            for leg in request_case.legs
        ),
    )


def _task_evaluation_source_matches_leg(
    leg: TaskEvaluationExpertLeg,
    expert_source: VerifiedTaskEvaluationCandidate | VerifiedTaskEvaluationSourceBase,
) -> bool:
    if (
        leg.kind is TaskEvaluationLegKind.CANDIDATE
        and type(expert_source) is VerifiedTaskEvaluationCandidate
    ):
        return (
            expert_source.manifest.candidate_id == leg.expert_artifact_id
            and expert_source.commit_record.commit_record_id
            == leg.expert_source_receipt_id
            and expert_source.source_tree.tree_hash == leg.expert_tree_hash
        )
    if (
        leg.kind is TaskEvaluationLegKind.SOURCE_BASE_CONTROL
        and type(expert_source) is VerifiedTaskEvaluationSourceBase
    ):
        return (
            expert_source.release_manifest.release_id == leg.expert_artifact_id
            and expert_source.source_base_tree_receipt.source_base_tree_receipt_id
            == leg.expert_source_receipt_id
            and expert_source.source_base_tree_receipt.source_base_tree_hash
            == leg.expert_tree_hash
        )
    return False


def _reconstruct_prepared_request(
    prepared_request: PreparedTaskEvaluationRequest,
) -> PreparedTaskEvaluationRequest:
    if type(prepared_request) is not PreparedTaskEvaluationRequest:
        raise TaskEvaluationExecutionError(
            "task evaluation dispatch requires one exact prepared request"
        )
    return PreparedTaskEvaluationRequest(
        plan_join=prepared_request.plan_join,
        stored_candidate=prepared_request.stored_candidate,
        candidate=prepared_request.candidate,
        source_base=prepared_request.source_base,
        current_release_observation=prepared_request.current_release_observation,
        cases=prepared_request.cases,
    )
