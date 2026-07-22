"""Byte-closed task-evaluation requests before durable reservation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from kapso.cross_run.contracts import ExpertBaseReleaseManifest
from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixProvenanceBinding,
    ExpertReleaseMatrixProvenanceKind,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.task_evaluation_contracts import TaskEvaluationCase
from kapso.cross_run.expert.task_evaluation_materialization import (
    TaskEvaluationMaterializationLimits,
    VerifiedTaskEvaluationAdapterRuntime,
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationParent,
    VerifiedTaskEvaluationStartingArtifact,
    materialize_task_evaluation_starting_artifacts,
)
from kapso.cross_run.expert.task_evaluation_request import (
    PlanJoinedTaskEvaluationRequest,
    prepare_task_evaluation_request,
    validate_task_evaluation_candidate_authority,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    task_adapter_materialization_usage,
)
from kapso.cross_run.expert.triggers import ExpertParentTreeReceipt
from kapso.cross_run.expert.validation_store import (
    ExpertReleaseMatrixPlanReservationSnapshot,
)
from kapso.cross_run.settings import ExpertValidationSettings


class TaskEvaluationPreflightError(ValueError):
    """A materialized evaluation request differs from reserved authority."""


class TaskEvaluationPlanReservationAuthority(Protocol):
    def reopen_release_matrix_plan_reservation_snapshot(
        self,
        *,
        plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    ) -> ExpertReleaseMatrixPlanReservationSnapshot: ...


class TaskEvaluationCandidateReader(Protocol):
    def read(self, candidate_id: str) -> StoredExpertCandidate: ...


class TaskEvaluationParentProvider(Protocol):
    def materialize_exact(
        self,
        release_manifest: ExpertBaseReleaseManifest,
        parent_tree_receipt: ExpertParentTreeReceipt,
        limits: TaskEvaluationMaterializationLimits,
    ) -> VerifiedTaskEvaluationParent: ...


class TaskEvaluationAdapterProvider(Protocol):
    def resolve_exact_bounded(
        self,
        *,
        task_adapter_manifest_id: str,
        verification_receipt_id: str,
        maximum_entries: int,
        maximum_bytes: int,
        timeout_seconds: int,
    ) -> VerifiedTaskAdapter: ...


class TaskEvaluationCurrentReleaseAuthority(Protocol):
    def observe_task_evaluation_current(
        self,
        scope_id: str,
    ) -> TaskEvaluationCurrentReleaseObservation: ...


@dataclass(frozen=True)
class MaterializedTaskEvaluationCase:
    request_case: TaskEvaluationCase
    adapter: VerifiedTaskAdapter
    adapter_runtime: VerifiedTaskEvaluationAdapterRuntime
    starting_artifacts: tuple[VerifiedTaskEvaluationStartingArtifact, ...]

    def __post_init__(self) -> None:
        if (
            type(self.request_case) is not TaskEvaluationCase
            or type(self.adapter) is not VerifiedTaskAdapter
            or type(self.adapter_runtime) is not VerifiedTaskEvaluationAdapterRuntime
            or type(self.starting_artifacts) is not tuple
            or any(
                type(artifact) is not VerifiedTaskEvaluationStartingArtifact
                for artifact in self.starting_artifacts
            )
        ):
            raise TaskEvaluationPreflightError(
                "materialized task-evaluation case requires exact byte authorities"
            )
        if self.adapter_runtime != (
            VerifiedTaskEvaluationAdapterRuntime.from_verified_adapter(self.adapter)
        ):
            raise TaskEvaluationPreflightError(
                "materialized task-evaluation runtime differs from its full package"
            )


@dataclass(frozen=True)
class PreparedTaskEvaluationRequest:
    plan_join: PlanJoinedTaskEvaluationRequest
    stored_candidate: StoredExpertCandidate
    candidate: VerifiedTaskEvaluationCandidate
    parent: VerifiedTaskEvaluationParent | None
    current_release_observation: TaskEvaluationCurrentReleaseObservation
    cases: tuple[MaterializedTaskEvaluationCase, ...]

    def __post_init__(self) -> None:
        if (
            type(self.plan_join) is not PlanJoinedTaskEvaluationRequest
            or type(self.stored_candidate) is not StoredExpertCandidate
            or type(self.candidate) is not VerifiedTaskEvaluationCandidate
            or (
                self.parent is not None
                and type(self.parent) is not VerifiedTaskEvaluationParent
            )
            or type(self.current_release_observation)
            is not TaskEvaluationCurrentReleaseObservation
            or type(self.cases) is not tuple
            or any(
                type(case) is not MaterializedTaskEvaluationCase for case in self.cases
            )
        ):
            raise TaskEvaluationPreflightError(
                "prepared task evaluation requires exact typed authorities"
            )
        rederived = prepare_task_evaluation_request(
            plan_reservation=self.plan_join.plan_reservation,
            settings=self.plan_join.settings,
            stored_candidate=self.stored_candidate,
            candidate=self.candidate,
            parent=self.parent,
        )
        if rederived != self.plan_join:
            raise TaskEvaluationPreflightError(
                "prepared task-evaluation request differs from exact derivation"
            )
        packet = self.stored_candidate.closure.trigger_packet
        if (
            self.current_release_observation.scope_id != packet.scope_contract.scope_id
            or self.current_release_observation.release_id
            != self.plan_join.request.parent_release_id
        ):
            raise TaskEvaluationPreflightError(
                "prepared task-evaluation current authority differs from its request"
            )
        if (
            tuple(case.request_case for case in self.cases)
            != self.plan_join.request.cases
        ):
            raise TaskEvaluationPreflightError(
                "materialized task-evaluation case coverage is not exact"
            )
        self._validate_cases()
        _unique_task_evaluation_adapters(self.cases)

    def _validate_cases(self) -> None:
        plan = self.plan_join.plan_reservation.evaluation_plan
        provenances = {
            provenance.provenance_binding_id: provenance
            for provenance in plan.provenance_bindings
            if provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
        }
        authorities = {
            authority.adapter_authority_id: authority
            for authority in plan.adapter_authorities
        }
        for materialized_case in self.cases:
            request_case = materialized_case.request_case
            provenance = provenances[request_case.provenance_binding_id]
            authority = authorities[request_case.adapter_authority_id]
            signed_case = provenance.adapter_case
            if signed_case is None:
                raise TaskEvaluationPreflightError(
                    "materialized task-evaluation case lacks signed authority"
                )
            adapter = materialized_case.adapter
            if (
                provenance.adapter_authority_id != authority.adapter_authority_id
                or adapter.manifest != authority.task_adapter_manifest
                or adapter.verification_receipt != authority.verification_receipt
                or adapter.dependency_ids != authority.task_adapter_dependency_ids
                or signed_case.release_matrix_case_id
                != request_case.release_matrix_case_id
                or materialized_case.starting_artifacts
                != materialize_task_evaluation_starting_artifacts(
                    adapter=adapter,
                    signed_case=signed_case,
                )
            ):
                raise TaskEvaluationPreflightError(
                    "materialized task-evaluation case differs from plan authority"
                )

    @property
    def adapters(self) -> tuple[VerifiedTaskAdapter, ...]:
        return _unique_task_evaluation_adapters(self.cases)

    @property
    def entry_count(self) -> int:
        return task_evaluation_materialization_usage(
            candidate=self.candidate,
            parent=self.parent,
            adapters=self.adapters,
        )[0]

    @property
    def byte_count(self) -> int:
        return task_evaluation_materialization_usage(
            candidate=self.candidate,
            parent=self.parent,
            adapters=self.adapters,
        )[1]


class TaskEvaluationPreflightCoordinator:
    """Materialize one reserved matrix under exact local and external fences."""

    def __init__(
        self,
        *,
        settings: ExpertValidationSettings,
        plan_reservation_authority: TaskEvaluationPlanReservationAuthority,
        candidate_reader: TaskEvaluationCandidateReader,
        parent_provider: TaskEvaluationParentProvider,
        adapter_provider: TaskEvaluationAdapterProvider,
        current_release_authority: TaskEvaluationCurrentReleaseAuthority,
        monotonic_clock: Callable[[], float],
    ) -> None:
        if type(settings) is not ExpertValidationSettings:
            raise TaskEvaluationPreflightError(
                "task-evaluation preflight requires exact validation settings"
            )
        self.settings = settings
        self.plan_reservation_authority = plan_reservation_authority
        self.candidate_reader = candidate_reader
        self.parent_provider = parent_provider
        self.adapter_provider = adapter_provider
        self.current_release_authority = current_release_authority
        self.monotonic_clock = monotonic_clock

    def build(
        self,
        plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    ) -> PreparedTaskEvaluationRequest:
        if type(plan_reservation) is not ExpertReleaseMatrixPlanReservationSnapshot:
            raise TaskEvaluationPreflightError(
                "task-evaluation preflight requires an exact plan reservation"
            )
        deadline = (
            self.monotonic_clock()
            + self.settings.policy.task_evaluation_materialization_timeout_seconds
        )
        first_reservation = self._reopen_plan(plan_reservation)
        plan = first_reservation.evaluation_plan
        stored_candidate = self.candidate_reader.read(plan.candidate_id)
        if type(stored_candidate) is not StoredExpertCandidate:
            raise TaskEvaluationPreflightError(
                "task-evaluation candidate reader returned an unverified closure"
            )
        candidate = VerifiedTaskEvaluationCandidate(
            manifest=stored_candidate.closure.manifest,
            commit_record=stored_candidate.commit_record,
            source_tree=stored_candidate.closure.candidate_tree,
            source_contents=stored_candidate.closure.candidate_contents,
        )
        validate_task_evaluation_candidate_authority(
            plan_reservation=first_reservation,
            settings=self.settings,
            stored_candidate=stored_candidate,
            candidate=candidate,
        )
        self._require_deadline(deadline)
        scope_id = stored_candidate.closure.trigger_packet.scope_contract.scope_id
        current_before = self._observe_current(scope_id, plan.parent_release_id)
        self._require_deadline(deadline)
        parent = self._materialize_parent(
            stored_candidate=stored_candidate,
            candidate=candidate,
            deadline=deadline,
        )
        plan_join = prepare_task_evaluation_request(
            plan_reservation=first_reservation,
            settings=self.settings,
            stored_candidate=stored_candidate,
            candidate=candidate,
            parent=parent,
        )
        adapters_by_authority_id = self._materialize_adapters(
            plan_reservation=first_reservation,
            candidate=candidate,
            parent=parent,
            deadline=deadline,
        )
        current_after = self._observe_current(scope_id, plan.parent_release_id)
        if current_after != current_before:
            raise TaskEvaluationPreflightError(
                "task-evaluation current authority changed during materialization"
            )
        self._require_deadline(deadline)
        second_reservation = self._reopen_plan(first_reservation)
        if second_reservation != first_reservation:
            raise TaskEvaluationPreflightError(
                "task-evaluation plan reservation changed during materialization"
            )
        provenances = {
            provenance.provenance_binding_id: provenance
            for provenance in plan.provenance_bindings
            if provenance.provenance_kind
            is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
        }
        cases = tuple(
            self._materialized_case(
                request_case=request_case,
                provenance=provenances[request_case.provenance_binding_id],
                adapter=adapters_by_authority_id[request_case.adapter_authority_id],
            )
            for request_case in plan_join.request.cases
        )
        prepared = PreparedTaskEvaluationRequest(
            plan_join=plan_join,
            stored_candidate=stored_candidate,
            candidate=candidate,
            parent=parent,
            current_release_observation=current_after,
            cases=cases,
        )
        limits = self._configured_limits()
        if (
            prepared.entry_count > limits.maximum_entries
            or prepared.byte_count > limits.maximum_bytes
        ):
            raise TaskEvaluationPreflightError(
                "task-evaluation byte closure exceeds materialization limits"
            )
        self._require_deadline(deadline)
        return prepared

    def _reopen_plan(
        self,
        plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
    ) -> ExpertReleaseMatrixPlanReservationSnapshot:
        reopened = self.plan_reservation_authority.reopen_release_matrix_plan_reservation_snapshot(
            plan_reservation=plan_reservation,
        )
        if (
            type(reopened) is not ExpertReleaseMatrixPlanReservationSnapshot
            or reopened != plan_reservation
        ):
            raise TaskEvaluationPreflightError(
                "task-evaluation plan reservation is not current"
            )
        return reopened

    def _observe_current(
        self,
        scope_id: str,
        expected_release_id: str | None,
    ) -> TaskEvaluationCurrentReleaseObservation:
        observation = self.current_release_authority.observe_task_evaluation_current(
            scope_id
        )
        if (
            type(observation) is not TaskEvaluationCurrentReleaseObservation
            or observation.scope_id != scope_id
            or observation.release_id != expected_release_id
        ):
            raise TaskEvaluationPreflightError(
                "task-evaluation current release differs from reserved authority"
            )
        return observation

    def _materialize_parent(
        self,
        *,
        stored_candidate: StoredExpertCandidate,
        candidate: VerifiedTaskEvaluationCandidate,
        deadline: float,
    ) -> VerifiedTaskEvaluationParent | None:
        packet = stored_candidate.closure.trigger_packet
        if packet.parent_release is None or packet.parent_tree_receipt is None:
            if (
                packet.parent_release is not None
                or packet.parent_tree_receipt is not None
            ):
                raise TaskEvaluationPreflightError(
                    "task-evaluation candidate parent authority is partial"
                )
            return None
        limits = self._remaining_limits(
            candidate=candidate,
            parent=None,
            adapters=(),
            deadline=deadline,
        )
        parent = self.parent_provider.materialize_exact(
            packet.parent_release,
            packet.parent_tree_receipt,
            limits,
        )
        if type(parent) is not VerifiedTaskEvaluationParent:
            raise TaskEvaluationPreflightError(
                "task-evaluation parent provider returned an unverified closure"
            )
        self._require_deadline(deadline)
        return parent

    def _materialize_adapters(
        self,
        *,
        plan_reservation: ExpertReleaseMatrixPlanReservationSnapshot,
        candidate: VerifiedTaskEvaluationCandidate,
        parent: VerifiedTaskEvaluationParent | None,
        deadline: float,
    ) -> dict[str, VerifiedTaskAdapter]:
        plan = plan_reservation.evaluation_plan
        authority_ids = tuple(
            sorted(
                {
                    provenance.adapter_authority_id
                    for provenance in plan.provenance_bindings
                    if provenance.provenance_kind
                    is ExpertReleaseMatrixProvenanceKind.ADAPTER_CASE
                }
            )
        )
        authorities = {
            authority.adapter_authority_id: authority
            for authority in plan.adapter_authorities
        }
        resolved: dict[str, VerifiedTaskAdapter] = {}
        for authority_id in authority_ids:
            authority = authorities[authority_id]
            limits = self._remaining_limits(
                candidate=candidate,
                parent=parent,
                adapters=tuple(resolved.values()),
                deadline=deadline,
            )
            pin = authority.task_adapter_pin
            adapter = self.adapter_provider.resolve_exact_bounded(
                task_adapter_manifest_id=pin.task_adapter_manifest_id,
                verification_receipt_id=pin.verification_receipt_id,
                maximum_entries=limits.maximum_entries,
                maximum_bytes=limits.maximum_bytes,
                timeout_seconds=limits.timeout_seconds,
            )
            if (
                type(adapter) is not VerifiedTaskAdapter
                or adapter.manifest != authority.task_adapter_manifest
                or adapter.verification_receipt != authority.verification_receipt
                or adapter.dependency_ids != authority.task_adapter_dependency_ids
            ):
                raise TaskEvaluationPreflightError(
                    "task-evaluation adapter differs from reserved authority"
                )
            resolved[authority_id] = adapter
            self._require_deadline(deadline)
        return resolved

    @staticmethod
    def _materialized_case(
        *,
        request_case: TaskEvaluationCase,
        provenance: ExpertReleaseMatrixProvenanceBinding,
        adapter: VerifiedTaskAdapter,
    ) -> MaterializedTaskEvaluationCase:
        signed_case = provenance.adapter_case
        if signed_case is None:
            raise TaskEvaluationPreflightError(
                "task-evaluation provenance lacks a signed adapter case"
            )
        return MaterializedTaskEvaluationCase(
            request_case=request_case,
            adapter=adapter,
            adapter_runtime=VerifiedTaskEvaluationAdapterRuntime.from_verified_adapter(
                adapter
            ),
            starting_artifacts=materialize_task_evaluation_starting_artifacts(
                adapter=adapter,
                signed_case=signed_case,
            ),
        )

    def _configured_limits(self) -> TaskEvaluationMaterializationLimits:
        policy = self.settings.policy
        return TaskEvaluationMaterializationLimits(
            maximum_entries=policy.task_evaluation_materialization_entry_limit,
            maximum_bytes=policy.task_evaluation_materialization_byte_limit,
            timeout_seconds=policy.task_evaluation_materialization_timeout_seconds,
        )

    def _remaining_limits(
        self,
        *,
        candidate: VerifiedTaskEvaluationCandidate,
        parent: VerifiedTaskEvaluationParent | None,
        adapters: tuple[VerifiedTaskAdapter, ...],
        deadline: float,
    ) -> TaskEvaluationMaterializationLimits:
        limits = self._configured_limits()
        entry_count, byte_count = task_evaluation_materialization_usage(
            candidate=candidate,
            parent=parent,
            adapters=adapters,
        )
        remaining_entries = limits.maximum_entries - entry_count
        remaining_bytes = limits.maximum_bytes - byte_count
        remaining_seconds = int(deadline - self.monotonic_clock())
        if remaining_entries <= 0 or remaining_bytes <= 0 or remaining_seconds <= 0:
            raise TaskEvaluationPreflightError(
                "task-evaluation materialization budget is exhausted"
            )
        return TaskEvaluationMaterializationLimits(
            maximum_entries=remaining_entries,
            maximum_bytes=remaining_bytes,
            timeout_seconds=remaining_seconds,
        )

    def _require_deadline(self, deadline: float) -> None:
        if self.monotonic_clock() >= deadline:
            raise TaskEvaluationPreflightError(
                "task-evaluation materialization deadline expired"
            )


def task_evaluation_materialization_usage(
    *,
    candidate: VerifiedTaskEvaluationCandidate,
    parent: VerifiedTaskEvaluationParent | None,
    adapters: tuple[VerifiedTaskAdapter, ...],
) -> tuple[int, int]:
    """Count acquired bytes once; runtime/artifact projections are not copies."""

    if (
        type(candidate) is not VerifiedTaskEvaluationCandidate
        or (parent is not None and type(parent) is not VerifiedTaskEvaluationParent)
        or type(adapters) is not tuple
        or any(type(adapter) is not VerifiedTaskAdapter for adapter in adapters)
    ):
        raise TaskEvaluationPreflightError(
            "task-evaluation materialization usage requires exact byte authorities"
        )
    keyed_adapters: dict[tuple[str, str], VerifiedTaskAdapter] = {}
    for adapter in adapters:
        key = (
            adapter.manifest.task_adapter_manifest_id,
            adapter.verification_receipt.verification_receipt_id,
        )
        existing = keyed_adapters.get(key)
        if existing is not None and existing != adapter:
            raise TaskEvaluationPreflightError(
                "task-evaluation adapter identity has conflicting byte closures"
            )
        keyed_adapters[key] = adapter
    adapter_usages = tuple(
        task_adapter_materialization_usage(
            source_file_sizes=tuple(
                descriptor.size
                for descriptor in adapter.source_extraction_receipt.source_tree_files
            ),
            source_archive_sizes=(len(adapter.source_archive),),
            proof_object_sizes=tuple(
                len(payload) for payload in adapter.proof_objects.values()
            ),
            publisher_verification_sizes=(len(adapter.publisher_verification),),
        )
        for adapter in keyed_adapters.values()
    )
    return (
        candidate.entry_count
        + (0 if parent is None else parent.entry_count)
        + sum(usage[0] for usage in adapter_usages),
        candidate.byte_count
        + (0 if parent is None else parent.byte_count)
        + sum(usage[1] for usage in adapter_usages),
    )


def _unique_task_evaluation_adapters(
    cases: tuple[MaterializedTaskEvaluationCase, ...],
) -> tuple[VerifiedTaskAdapter, ...]:
    adapters: dict[tuple[str, str], VerifiedTaskAdapter] = {}
    for materialized_case in cases:
        adapter = materialized_case.adapter
        key = (
            adapter.manifest.task_adapter_manifest_id,
            adapter.verification_receipt.verification_receipt_id,
        )
        existing = adapters.get(key)
        if existing is not None and existing != adapter:
            raise TaskEvaluationPreflightError(
                "materialized task-evaluation package identity conflicts"
            )
        adapters[key] = adapter
    return tuple(adapters[key] for key in sorted(adapters))
