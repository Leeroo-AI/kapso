from collections.abc import Mapping

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    EvaluationFingerprint,
    ObjectiveDirection,
    TaskAdapterReleaseMatrixCase,
    TaskAdapterReleaseMatrixIndependenceGroup,
    TaskAdapterReleaseMatrixStartingArtifact,
    TaskContextBinding,
)


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def task_adapter_release_matrix_case(
    *,
    scope_contract_id: str,
    scope_id: str,
    task_family_id: str,
    task_adapter_id: str,
    evaluator_fingerprint: str,
    metric_directions: tuple[tuple[str, ObjectiveDirection], ...],
    transfer_dimensions: Mapping[str, object],
    label: str,
    starting_artifacts: tuple[TaskAdapterReleaseMatrixStartingArtifact, ...] = (),
    evaluation_bindings: tuple[tuple[str, str, ObjectiveDirection], ...] | None = None,
    seed_or_replicate_ids: tuple[str, ...] = ("seed-1",),
) -> TaskAdapterReleaseMatrixCase:
    context = TaskContextBinding.mint(
        scope_contract_id=scope_contract_id,
        scope_id=scope_id,
        task_family_id=task_family_id,
        task_adapter_id=task_adapter_id,
        capability_tags=("release.matrix",),
        input_contract_fingerprint=_digest(f"{label}:input"),
        target_contract_fingerprint=_digest(f"{label}:target"),
        starting_artifact_refs=tuple(
            sorted(artifact.starting_artifact_ref for artifact in starting_artifacts)
        ),
        method_fingerprint=_digest(f"{label}:method"),
        toolchain_fingerprint=_digest(f"{label}:toolchain"),
        dependency_runtime_fingerprint=_digest(f"{label}:runtime"),
        budget_hardware_envelope={"fixture": label},
        transfer_dimensions=dict(transfer_dimensions),
    )
    selected_evaluation_bindings = (
        tuple(
            (evaluator_fingerprint, metric_name, direction)
            for metric_name, direction in metric_directions
        )
        if evaluation_bindings is None
        else evaluation_bindings
    )
    fingerprints = tuple(
        sorted(
            (
                EvaluationFingerprint.mint(
                    benchmark_id=task_adapter_id,
                    dataset_version="v1",
                    split_version="release-matrix-v1",
                    evaluator_fingerprint=selected_evaluator_fingerprint,
                    metric_name=metric_name,
                    objective_direction=direction,
                    fidelity="full",
                    fraction=1.0,
                    seed_or_replicate_ids=seed_or_replicate_ids,
                    aggregation_protocol="arithmetic-mean",
                    judge_version=None,
                )
                for (
                    selected_evaluator_fingerprint,
                    metric_name,
                    direction,
                ) in selected_evaluation_bindings
            ),
            key=lambda fingerprint: fingerprint.evaluation_fingerprint_id,
        )
    )
    return TaskAdapterReleaseMatrixCase.mint(
        task_context_binding=context,
        independence_group=TaskAdapterReleaseMatrixIndependenceGroup.mint(
            lineage_root_digests=(_digest(f"{label}:lineage-root"),),
        ),
        evaluation_fingerprints=fingerprints,
        starting_artifacts=tuple(
            sorted(
                starting_artifacts,
                key=lambda artifact: artifact.starting_artifact_content_id,
            )
        ),
    )
