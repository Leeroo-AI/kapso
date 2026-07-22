"""Byte-closed task-evaluation requests before durable reservation."""

from __future__ import annotations

from dataclasses import dataclass

from kapso.cross_run.expert.promotion_contracts import (
    ExpertReleaseMatrixProvenanceKind,
)
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.expert.task_evaluation_contracts import TaskEvaluationCase
from kapso.cross_run.expert.task_evaluation_materialization import (
    VerifiedTaskEvaluationAdapterRuntime,
    VerifiedTaskEvaluationCandidate,
    VerifiedTaskEvaluationParent,
    VerifiedTaskEvaluationStartingArtifact,
    materialize_task_evaluation_starting_artifacts,
)
from kapso.cross_run.expert.task_evaluation_request import (
    PlanJoinedTaskEvaluationRequest,
    prepare_task_evaluation_request,
)
from kapso.cross_run.task_adapters import (
    VerifiedTaskAdapter,
    task_adapter_materialization_usage,
)


class TaskEvaluationPreflightError(ValueError):
    """A materialized evaluation request differs from reserved authority."""


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
