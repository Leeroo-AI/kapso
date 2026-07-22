"""Pure domain-neutral contracts for the blinded task-evaluator ABI."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.contracts import EvaluationFingerprint, StrictContract

TASK_ADAPTER_RUNTIME_PROTOCOL_VERSION = "kapso.task_adapter_runtime.v1"
TASK_EVALUATOR_PROTOCOL_VERSION = "kapso.task_evaluator.v1"
TASK_EVALUATOR_REQUEST_PATH = "/kapso/input/request.json"
TASK_EVALUATOR_EXPERT_ROOT = "/kapso/input/expert"
TASK_EVALUATOR_ADAPTER_ROOT = "/kapso/input/adapter"
TASK_EVALUATOR_TASK_ROOT = "/kapso/input/task"
TASK_EVALUATOR_WRITABLE_ROOT = "/kapso/writable"
TASK_EVALUATOR_RESULT_PATH = "/kapso/writable/result.json"

_INVOCATION_ID_PATTERN = re.compile(r"^task_evaluation_invocation_[0-9a-f]{32}$")
_SUPPORTED_AGGREGATION_PROTOCOLS = frozenset({"arithmetic-mean"})


class TaskEvaluatorProtocolError(ValueError):
    """The task-evaluator request or result violates its exact protocol."""


def _require_normalized_relative_path(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise TaskEvaluatorProtocolError(f"{name} must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or path.is_absolute()
        or ".." in path.parts
        or path == PurePosixPath(".")
        or path.as_posix() != value
    ):
        raise TaskEvaluatorProtocolError(f"{name} must be a normalized relative path")


def require_finite_float(value: Any, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TaskEvaluatorProtocolError(f"{name} must be a finite float")


def stable_arithmetic_mean(values: tuple[float, ...]) -> float:
    if not values or any(
        type(value) is not float or not math.isfinite(value) for value in values
    ):
        raise TaskEvaluatorProtocolError(
            "stable arithmetic mean requires finite floating-point values"
        )
    maximum_absolute_value = max(abs(value) for value in values)
    if maximum_absolute_value == 0.0:
        return 0.0
    normalized_mean = math.fsum(
        value / maximum_absolute_value for value in values
    ) / len(values)
    return normalized_mean * maximum_absolute_value


@dataclass(frozen=True)
class TaskEvaluatorStartingArtifactMount(StrictContract):
    starting_artifact_ref: str
    mount_path: str

    def _validate(self) -> None:
        if not isinstance(self.starting_artifact_ref, str) or not (
            self.starting_artifact_ref
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator starting_artifact_ref must be non-empty"
            )
        _require_normalized_relative_path(
            self.mount_path,
            "task evaluator starting artifact mount_path",
        )

    @property
    def materialized_path(self) -> str:
        return f"{TASK_EVALUATOR_TASK_ROOT}/{self.mount_path}"


@dataclass(frozen=True)
class TaskEvaluatorRequest(StrictContract):
    protocol_version: str
    opaque_invocation_id: str
    input_contract_fingerprint: str
    target_contract_fingerprint: str
    evaluation_fingerprints: tuple[EvaluationFingerprint, ...]
    context_dimensions: Mapping[str, Any]
    starting_artifact_mounts: tuple[TaskEvaluatorStartingArtifactMount, ...]

    def _validate(self) -> None:
        if self.protocol_version != TASK_EVALUATOR_PROTOCOL_VERSION:
            raise TaskEvaluatorProtocolError(
                "task evaluator request protocol is unsupported"
            )
        if (
            not isinstance(self.opaque_invocation_id, str)
            or _INVOCATION_ID_PATTERN.fullmatch(self.opaque_invocation_id) is None
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator invocation ID must be an opaque 128-bit identifier"
            )
        for value, name in (
            (self.input_contract_fingerprint, "input_contract_fingerprint"),
            (self.target_contract_fingerprint, "target_contract_fingerprint"),
        ):
            if (
                not isinstance(value, str)
                or not value.startswith("sha256:")
                or len(value) != 71
                or any(character not in "0123456789abcdef" for character in value[7:])
            ):
                raise TaskEvaluatorProtocolError(
                    f"task evaluator {name} must be a sha256 digest"
                )
        fingerprint_ids = tuple(
            fingerprint.evaluation_fingerprint_id
            for fingerprint in self.evaluation_fingerprints
        )
        if not fingerprint_ids or fingerprint_ids != tuple(
            sorted(set(fingerprint_ids))
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator fingerprints must be ID-sorted and unique"
            )
        unsupported_aggregation_protocols = tuple(
            sorted(
                {
                    fingerprint.aggregation_protocol
                    for fingerprint in self.evaluation_fingerprints
                }
                - _SUPPORTED_AGGREGATION_PROTOCOLS
            )
        )
        if unsupported_aggregation_protocols:
            raise TaskEvaluatorProtocolError(
                "task evaluator fingerprints use unsupported aggregation protocols: "
                f"{unsupported_aggregation_protocols}"
            )
        dimension_ids = tuple(self.context_dimensions)
        if dimension_ids != tuple(sorted(set(dimension_ids))):
            raise TaskEvaluatorProtocolError(
                "task evaluator context dimensions must be key-sorted and unique"
            )
        mount_refs = tuple(
            mount.starting_artifact_ref for mount in self.starting_artifact_mounts
        )
        if mount_refs != tuple(sorted(set(mount_refs))):
            raise TaskEvaluatorProtocolError(
                "task evaluator artifact mounts must be ref-sorted and unique"
            )
        mount_paths = tuple(
            PurePosixPath(mount.mount_path) for mount in self.starting_artifact_mounts
        )
        if len(mount_paths) != len(set(mount_paths)) or any(
            left in right.parents or right in left.parents
            for position, left in enumerate(mount_paths)
            for right in mount_paths[position + 1 :]
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator starting artifact mounts overlap"
            )


@dataclass(frozen=True)
class TaskEvaluatorFingerprintResult(StrictContract):
    evaluation_fingerprint_id: str
    aggregate_value: float
    replicate_values: Mapping[str, float]

    def _validate(self) -> None:
        require_content_id(
            self.evaluation_fingerprint_id,
            "task evaluator result evaluation_fingerprint_id",
        )
        if self.evaluation_fingerprint_id.split(":sha256:", 1)[0] != (
            "evaluation-fingerprint"
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator result must name an evaluation fingerprint"
            )
        require_finite_float(
            self.aggregate_value,
            "task evaluator aggregate value",
        )
        if not self.replicate_values:
            raise TaskEvaluatorProtocolError(
                "task evaluator result must contain replicate values"
            )
        for replicate_id, value in self.replicate_values.items():
            if not isinstance(replicate_id, str) or not replicate_id:
                raise TaskEvaluatorProtocolError(
                    "task evaluator replicate ID must be non-empty"
                )
            require_finite_float(value, "task evaluator replicate value")


@dataclass(frozen=True)
class TaskEvaluatorResult(StrictContract):
    protocol_version: str
    opaque_invocation_id: str
    fingerprint_results: tuple[TaskEvaluatorFingerprintResult, ...]

    def _validate(self) -> None:
        if self.protocol_version != TASK_EVALUATOR_PROTOCOL_VERSION:
            raise TaskEvaluatorProtocolError(
                "task evaluator result protocol is unsupported"
            )
        if (
            not isinstance(self.opaque_invocation_id, str)
            or _INVOCATION_ID_PATTERN.fullmatch(self.opaque_invocation_id) is None
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator result invocation ID is invalid"
            )
        fingerprint_ids = tuple(
            result.evaluation_fingerprint_id for result in self.fingerprint_results
        )
        if not fingerprint_ids or fingerprint_ids != tuple(
            sorted(set(fingerprint_ids))
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator results must be fingerprint-sorted and unique"
            )

    def validate_against(
        self,
        request: TaskEvaluatorRequest,
        aggregate_tolerance: float,
    ) -> None:
        if not isinstance(request, TaskEvaluatorRequest):
            raise TaskEvaluatorProtocolError(
                "task evaluator result requires a typed request"
            )
        if (
            self.protocol_version != request.protocol_version
            or self.opaque_invocation_id != request.opaque_invocation_id
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator result differs from its request identity"
            )
        if (
            type(aggregate_tolerance) is not float
            or not math.isfinite(aggregate_tolerance)
            or aggregate_tolerance < 0.0
        ):
            raise TaskEvaluatorProtocolError(
                "task evaluator aggregate tolerance must be a finite non-negative float"
            )
        expected_fingerprint_ids = tuple(
            fingerprint.evaluation_fingerprint_id
            for fingerprint in request.evaluation_fingerprints
        )
        observed_fingerprint_ids = tuple(
            result.evaluation_fingerprint_id for result in self.fingerprint_results
        )
        if observed_fingerprint_ids != expected_fingerprint_ids:
            raise TaskEvaluatorProtocolError(
                "task evaluator result fingerprint set differs from its request"
            )
        fingerprints_by_id = {
            fingerprint.evaluation_fingerprint_id: fingerprint
            for fingerprint in request.evaluation_fingerprints
        }
        for result in self.fingerprint_results:
            fingerprint = fingerprints_by_id[result.evaluation_fingerprint_id]
            if set(result.replicate_values) != set(fingerprint.seed_or_replicate_ids):
                raise TaskEvaluatorProtocolError(
                    "task evaluator result replicate set differs from its request"
                )
            expected_aggregate = stable_arithmetic_mean(
                tuple(
                    result.replicate_values[replicate_id]
                    for replicate_id in fingerprint.seed_or_replicate_ids
                )
            )
            if not math.isclose(
                result.aggregate_value,
                expected_aggregate,
                rel_tol=0.0,
                abs_tol=aggregate_tolerance,
            ):
                raise TaskEvaluatorProtocolError(
                    "task evaluator aggregate differs from its replicate values"
                )


def parse_task_evaluator_result(
    payload: bytes,
    request: TaskEvaluatorRequest,
    aggregate_tolerance: float,
) -> TaskEvaluatorResult:
    if not isinstance(payload, bytes):
        raise TaskEvaluatorProtocolError("task evaluator result payload must be bytes")
    result = TaskEvaluatorResult.from_json_bytes(payload)
    if result.to_json_bytes() != payload:
        raise TaskEvaluatorProtocolError(
            "task evaluator result bytes are not canonical"
        )
    result.validate_against(request, aggregate_tolerance)
    return result
