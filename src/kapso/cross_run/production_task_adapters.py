"""Canonical public task adapters for the production transport smoke."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ContextValueType,
    EvaluationFingerprint,
    ExpertScopeContract,
    SourceFileDescriptor,
    TaskAdapterContextBinding,
    TaskAdapterManifest,
    TaskAdapterReleaseMatrixCase,
    TaskAdapterReleaseMatrixIndependenceGroup,
    TaskAdapterRuntimeContract,
    TaskContextBinding,
    TaskEvaluatorBinding,
    TaskEvaluatorMetricComparisonBinding,
)
from kapso.cross_run.expert.task_evaluator_protocol import (
    TASK_ADAPTER_RUNTIME_PROTOCOL_VERSION,
    TASK_EVALUATOR_PROTOCOL_VERSION,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.task_adapter_authority import CanonicalTaskAdapterAuthority
from kapso.cross_run.task_adapter_store import (
    TaskAdapterAuthorityRegistry,
    TaskAdapterPackageStore,
)
from kapso.cross_run.task_adapters import TaskAdapterPackage


class ProductionTaskAdapterError(ValueError):
    """The public transport adapter authority is incomplete or inconsistent."""


_EVALUATOR_PATH = "evaluate.py"
_LOCK_PATH = "requirements.lock"
_SOURCE_ARCHIVE_REF = "task-adapter.tar"
_EVALUATOR_SOURCE = b"""#!/usr/local/bin/kapso-provider-python
import json
from pathlib import Path

request = json.loads(Path(\"/kapso/input/request.json\").read_text(encoding=\"utf-8\"))
score = 1.0
results = []
for fingerprint in request[\"evaluation_fingerprints\"]:
    replicate_values = {
        replicate_id: score
        for replicate_id in fingerprint[\"seed_or_replicate_ids\"]
    }
    results.append(
        {
            \"aggregate_value\": score,
            \"evaluation_fingerprint_id\": fingerprint[\"evaluation_fingerprint_id\"],
            \"replicate_values\": replicate_values,
        }
    )
result = {
    \"fingerprint_results\": results,
    \"opaque_invocation_id\": request[\"opaque_invocation_id\"],
    \"protocol_version\": request[\"protocol_version\"],
}
Path(\"/kapso/writable/result.json\").write_text(
    json.dumps(result, sort_keys=True, separators=(\",\", \":\")),
    encoding=\"utf-8\",
)
"""
_LOCK_CONTENT = b"python-standard-library-only\n"


def production_capture_evaluation_fingerprint(
    settings: CrossRunSettings,
    task_adapter_id: str,
) -> EvaluationFingerprint:
    """Return the measured fingerprint used by the replayable smoke capture."""

    dimensions = tuple(
        sorted(
            settings.expert.validation.policy.promotion.pareto_dimensions,
            key=lambda dimension: dimension.dimension_id,
        )
    )
    if not dimensions:
        raise ProductionTaskAdapterError(
            "production capture requires one promotion dimension"
        )
    dimension = dimensions[0]
    return EvaluationFingerprint.mint(
        benchmark_id=task_adapter_id,
        dataset_version="public_transport_capture",
        split_version="production_smoke_v1",
        evaluator_fingerprint=tree_or_blob_digest(_EVALUATOR_SOURCE),
        metric_name=dimension.dimension_id,
        objective_direction=dimension.direction,
        fidelity="full",
        fraction=1.0,
        seed_or_replicate_ids=("seed-1",),
        aggregation_protocol="arithmetic-mean",
        judge_version=None,
    )


def bootstrap_production_task_adapters(
    *,
    settings: CrossRunSettings,
    state_root: Path,
    scope_contract: ExpertScopeContract,
    image_inspection: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Publish and activate deterministic public adapters for every scope binding."""

    image = settings.launch.coding_agent_image
    if image is None:
        raise ProductionTaskAdapterError(
            "production task adapters require the pinned coding-agent image"
        )
    image_environment = _validated_image_environment(image_inspection)
    adapter_settings = settings.expert.task_adapters
    store = TaskAdapterPackageStore(
        state_root / adapter_settings.state_path,
        state_root,
        adapter_settings,
        TaskAdapterAuthorityRegistry(
            adapter_settings,
            tuple(
                CanonicalTaskAdapterAuthority(authority)
                for authority in adapter_settings.trusted_authorities
            ),
        ),
    )
    adapters = []
    for binding in scope_contract.task_adapter_contract:
        for task_adapter_id in binding.task_adapter_ids:
            package = _build_package(
                settings=settings,
                scope_contract=scope_contract,
                task_family_id=binding.task_family_id,
                task_adapter_id=task_adapter_id,
                image_environment=image_environment,
            )
            published = store.publish(package)
            activation_envelope = canonical_json_bytes(
                {
                    "scope_contract_id": scope_contract.scope_contract_id,
                    "task_adapter_id": task_adapter_id,
                    "task_family_id": binding.task_family_id,
                    "verification_receipt_id": (
                        published.verification_receipt.verification_receipt_id
                    ),
                }
            )
            activation = store.activate(
                scope_contract_id=scope_contract.scope_contract_id,
                task_family_id=binding.task_family_id,
                task_adapter_id=task_adapter_id,
                verification_receipt_id=(
                    published.verification_receipt.verification_receipt_id
                ),
                expected_activation_id=None,
                authority_envelope=activation_envelope,
            )
            adapters.append(
                {
                    "activation_id": activation.activation_id,
                    "task_adapter_id": task_adapter_id,
                    "task_adapter_manifest_id": (
                        package.manifest.task_adapter_manifest_id
                    ),
                    "task_family_id": binding.task_family_id,
                    "verification_receipt_id": (
                        published.verification_receipt.verification_receipt_id
                    ),
                }
            )
    return {
        "adapters": tuple(adapters),
        "image_config_digest": image.image_config_digest,
        "image_reference": image.image_reference,
    }


def _validated_image_environment(
    image_inspection: Mapping[str, Any],
) -> Mapping[str, str]:
    config = image_inspection.get("Config")
    if not isinstance(config, Mapping):
        raise ProductionTaskAdapterError("task-adapter image has no config")
    raw_environment = config.get("Env")
    if not isinstance(raw_environment, list) or any(
        not isinstance(assignment, str) or "=" not in assignment
        for assignment in raw_environment
    ):
        raise ProductionTaskAdapterError("task-adapter image environment is not exact")
    environment: dict[str, str] = {}
    for assignment in raw_environment:
        key, value = assignment.split("=", 1)
        if not key or key in environment:
            raise ProductionTaskAdapterError(
                "task-adapter image environment is ambiguous"
            )
        environment[key] = value
    if (
        not environment.get("PATH")
        or config.get("Entrypoint") is not None
        or config.get("Cmd") not in (None, [])
        or config.get("Volumes") not in (None, {})
        or config.get("Healthcheck") is not None
    ):
        raise ProductionTaskAdapterError(
            "task-adapter image violates the evaluator sandbox contract"
        )
    return dict(sorted(environment.items()))


def _build_package(
    *,
    settings: CrossRunSettings,
    scope_contract: ExpertScopeContract,
    task_family_id: str,
    task_adapter_id: str,
    image_environment: Mapping[str, str],
) -> TaskAdapterPackage:
    source_contents = {
        _EVALUATOR_PATH: _EVALUATOR_SOURCE,
        _LOCK_PATH: _LOCK_CONTENT,
    }
    source_files = tuple(
        SourceFileDescriptor(
            relative_path=relative_path,
            digest=tree_or_blob_digest(payload),
            mode="100755" if relative_path == _EVALUATOR_PATH else "100644",
            size=len(payload),
        )
        for relative_path, payload in sorted(source_contents.items())
    )
    tree_hash = source_tree_digest(
        {
            descriptor.relative_path: (
                descriptor.digest,
                descriptor.mode,
                descriptor.size,
            )
            for descriptor in source_files
        }
    )
    evaluator_fingerprint = tree_or_blob_digest(_EVALUATOR_SOURCE)
    image = settings.launch.coding_agent_image
    if image is None:
        raise ProductionTaskAdapterError(
            "production task adapter package has no pinned image"
        )
    metric_comparison_bindings = tuple(
        TaskEvaluatorMetricComparisonBinding(
            evaluator_fingerprint=evaluator_fingerprint,
            metric_name=dimension.dimension_id,
            objective_direction=dimension.direction,
            comparison_dimension_id=dimension.dimension_id,
            comparison_scale=1.0,
        )
        for dimension in settings.expert.validation.policy.promotion.pareto_dimensions
    )
    manifest = TaskAdapterManifest.mint(
        task_adapter_id=task_adapter_id,
        scope_contract_id=scope_contract.scope_contract_id,
        task_family_id=task_family_id,
        publisher_attestation={
            "fixture": "kapso-public-transport",
            "publisher": settings.github.publisher_login,
        },
        task_evaluator=TaskEvaluatorBinding(
            protocol_version=TASK_EVALUATOR_PROTOCOL_VERSION,
            executable_path=_EVALUATOR_PATH,
            supported_evaluator_fingerprints=(evaluator_fingerprint,),
            metric_comparison_bindings=metric_comparison_bindings,
        ),
        context_binding=TaskAdapterContextBinding(
            consumed_dimension_ids=scope_contract.required_context_dimensions,
        ),
        release_matrix_cases=tuple(
            sorted(
                (
                    _release_matrix_case(
                        scope_contract=scope_contract,
                        task_family_id=task_family_id,
                        task_adapter_id=task_adapter_id,
                        evaluator_fingerprint=evaluator_fingerprint,
                        metric_comparison_bindings=metric_comparison_bindings,
                        position=position,
                        replicate_count=(
                            settings.expert.validation.policy.promotion.minimum_replicates_per_cell
                        ),
                    )
                    for position in range(
                        settings.expert.validation.policy.promotion.minimum_distinct_context_lineage_pairs
                    )
                ),
                key=lambda case: case.release_matrix_case_id,
            )
        ),
        source_tree_ref=_SOURCE_ARCHIVE_REF,
        tree_hash=tree_hash,
        runtime=TaskAdapterRuntimeContract(
            runtime_protocol_version=TASK_ADAPTER_RUNTIME_PROTOCOL_VERSION,
            image_repository=image.image_reference.rsplit("@", 1)[0],
            image_manifest_digest=image.image_reference.rsplit("@", 1)[1],
            image_config_digest=image.image_config_digest,
            dependency_lock_path=_LOCK_PATH,
            dependency_lock_digest=tree_or_blob_digest(_LOCK_CONTENT),
            operating_system=image.operating_system,
            architecture=image.architecture,
            architecture_variant=image.architecture_variant,
            environment=image_environment,
        ),
        sanitation_report_id=content_id(
            "production-task-adapter-sanitation",
            {
                "scope_contract_id": scope_contract.scope_contract_id,
                "task_adapter_id": task_adapter_id,
                "tree_hash": tree_hash,
            },
        ),
        validation_refs=("kapso.production_task_adapter.validation.v1",),
    )
    source_archive = _source_archive(source_contents)
    proof_objects = {
        proof_ref: canonical_json_bytes(
            {
                "manifest_id": manifest.task_adapter_manifest_id,
                "outcome": "passed",
                "proof_ref": proof_ref,
                "tree_hash": manifest.tree_hash,
            }
        )
        for proof_ref in {manifest.sanitation_report_id, *manifest.validation_refs}
    }
    publisher_verification = canonical_json_bytes(
        {
            "archive_digest": tree_or_blob_digest(source_archive),
            "full_manifest_digest": tree_or_blob_digest(manifest.to_json_bytes()),
            "manifest_id": manifest.task_adapter_manifest_id,
            "proof_digests": {
                proof_ref: tree_or_blob_digest(payload)
                for proof_ref, payload in proof_objects.items()
            },
            "publisher_attestation": manifest.publisher_attestation,
            "tree_hash": manifest.tree_hash,
        }
    )
    return TaskAdapterPackage(
        manifest=manifest,
        source_archive=source_archive,
        proof_objects=proof_objects,
        publisher_verification=publisher_verification,
    )


def _release_matrix_case(
    *,
    scope_contract: ExpertScopeContract,
    task_family_id: str,
    task_adapter_id: str,
    evaluator_fingerprint: str,
    metric_comparison_bindings: tuple[TaskEvaluatorMetricComparisonBinding, ...],
    position: int,
    replicate_count: int,
) -> TaskAdapterReleaseMatrixCase:
    label = f"{task_family_id}:{task_adapter_id}:{position}"
    context = TaskContextBinding.mint(
        scope_contract_id=scope_contract.scope_contract_id,
        scope_id=scope_contract.scope_id,
        task_family_id=task_family_id,
        task_adapter_id=task_adapter_id,
        capability_tags=("release.matrix",),
        input_contract_fingerprint=tree_or_blob_digest(
            f"{label}:input".encode("utf-8")
        ),
        target_contract_fingerprint=tree_or_blob_digest(
            f"{label}:target".encode("utf-8")
        ),
        starting_artifact_refs=(),
        method_fingerprint=tree_or_blob_digest(f"{label}:method".encode("utf-8")),
        toolchain_fingerprint=tree_or_blob_digest(f"{label}:toolchain".encode("utf-8")),
        dependency_runtime_fingerprint=tree_or_blob_digest(
            f"{label}:runtime".encode("utf-8")
        ),
        budget_hardware_envelope={"transport_fixture": position},
        transfer_dimensions={
            schema.dimension_id: _dimension_value(schema.value_type, position)
            for schema in scope_contract.context_dimension_schemas
            if schema.dimension_id in scope_contract.required_context_dimensions
        },
    )
    fingerprints = tuple(
        sorted(
            (
                EvaluationFingerprint.mint(
                    benchmark_id=task_adapter_id,
                    dataset_version=f"public-transport-{position}",
                    split_version="production-smoke-v1",
                    evaluator_fingerprint=evaluator_fingerprint,
                    metric_name=binding.metric_name,
                    objective_direction=binding.objective_direction,
                    fidelity="full",
                    fraction=1.0,
                    seed_or_replicate_ids=tuple(
                        f"replicate-{replicate_position + 1}"
                        for replicate_position in range(replicate_count)
                    ),
                    aggregation_protocol="arithmetic-mean",
                    judge_version=None,
                )
                for binding in metric_comparison_bindings
            ),
            key=lambda fingerprint: fingerprint.evaluation_fingerprint_id,
        )
    )
    return TaskAdapterReleaseMatrixCase.mint(
        task_context_binding=context,
        independence_group=TaskAdapterReleaseMatrixIndependenceGroup.mint(
            lineage_root_digests=(
                tree_or_blob_digest(f"{label}:lineage".encode("utf-8")),
            ),
        ),
        evaluation_fingerprints=fingerprints,
        starting_artifacts=(),
    )


def _dimension_value(value_type: ContextValueType, position: int) -> object:
    if value_type is ContextValueType.STRING:
        return f"public-transport-{position}"
    if value_type is ContextValueType.INTEGER:
        return position
    if value_type is ContextValueType.NUMBER:
        return float(position)
    if value_type is ContextValueType.BOOLEAN:
        return position % 2 == 0
    if value_type is ContextValueType.STRING_ARRAY:
        return (f"public-transport-{position}",)
    raise ProductionTaskAdapterError("unsupported scope context dimension type")


def _source_archive(source_contents: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative_path, payload in sorted(source_contents.items()):
            member = tarfile.TarInfo(relative_path)
            member.size = len(payload)
            member.mode = 0o755 if relative_path == _EVALUATOR_PATH else 0o644
            member.uid = 0
            member.gid = 0
            member.mtime = 0
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


__all__ = [
    "ProductionTaskAdapterError",
    "bootstrap_production_task_adapters",
    "production_capture_evaluation_fingerprint",
]
