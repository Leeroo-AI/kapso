"""Strict, domain-neutral contracts shared by every cross-run module."""

from __future__ import annotations

import base64
import math
import re
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType, UnionType
from typing import (
    Any,
    ClassVar,
    Mapping,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from kapso.cross_run.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    content_id,
    freeze_json,
    normalize_utc_timestamp,
    parse_json_bytes,
    parse_utc_timestamp,
    require_content_id,
    require_identifier,
    source_tree_digest,
    to_json_value,
    tree_or_blob_digest,
    verify_declared_content_id,
)
from kapso.cross_run.agent_artifacts import (
    CodingAgentWorkspaceAccess,
    coding_agent_artifact_filenames,
)

_ContractType = TypeVar("_ContractType", bound="StrictContract")
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GITHUB_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?/[A-Za-z0-9._-]+$"
)
_CODING_AGENT_OPERATION_PATTERN = re.compile(r"^agent_call_[0-9a-f]{32}$")
EMPTY_EXPERT_TREE_DIGEST = tree_or_blob_digest(canonical_json_bytes(()))


class CrossRunContractError(ValueError):
    """Base class for strict cross-run contract failures."""


class ContractValidationError(CrossRunContractError):
    """A contract has an invalid field or relation."""


class MissingReferenceError(CrossRunContractError):
    """A required referenced object is absent."""


class IncompatibleArtifactError(CrossRunContractError):
    """Two valid artifacts cannot participate in the same operation."""


class IdentityConflictError(CrossRunContractError):
    """Two objects claim incompatible ownership of one identity."""


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be non-empty text")
    return value


def _require_text_tuple(
    values: tuple[str, ...], name: str, *, required: bool = False
) -> None:
    if required and not values:
        raise ContractValidationError(f"{name} must not be empty")
    for position, value in enumerate(values):
        _require_text(value, f"{name}[{position}]")


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_DIGEST_PATTERN.fullmatch(value):
        raise ContractValidationError(f"{name} must be a sha256 digest")
    return value


def _require_repository_coordinate(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _GITHUB_REPOSITORY_PATTERN.fullmatch(value):
        raise ContractValidationError(
            f"{name} must be a GitHub owner/repository coordinate"
        )
    return value


def _require_sorted_unique(values: tuple[str, ...], name: str) -> None:
    if not values:
        raise ContractValidationError(f"{name} must not be empty")
    if values != tuple(sorted(set(values))):
        raise ContractValidationError(f"{name} must be sorted and unique")
    for value in values:
        require_identifier(value, name)


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ContractValidationError(f"{name} must be unique")


def _require_relative_path(value: str, name: str) -> None:
    _require_text(value, name)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ContractValidationError(f"{name} must be a normalized relative path")


def _require_checksum_mapping(values: Mapping[str, str], name: str) -> None:
    if not values:
        raise ContractValidationError(f"{name} must not be empty")
    for path, digest in values.items():
        _require_relative_path(path, f"{name} key")
        _require_digest(digest, f"{name}[{path}]")


def _decode_value(annotation: Any, value: Any, name: str) -> Any:
    if annotation is Any:
        return freeze_json(value, name)
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, UnionType):
        if value is None and type(None) in arguments:
            return None
        non_null = tuple(
            argument for argument in arguments if argument is not type(None)
        )
        if len(non_null) != 1:
            raise ContractValidationError(f"{name} has unsupported union annotation")
        return _decode_value(non_null[0], value, name)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ContractValidationError(f"{name} must be an array")
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise ContractValidationError(f"{name} has unsupported tuple annotation")
        return tuple(
            _decode_value(arguments[0], child, f"{name}[{position}]")
            for position, child in enumerate(value)
        )
    if origin in (dict, Mapping, MappingABC):
        if not isinstance(value, MappingABC):
            raise ContractValidationError(f"{name} must be an object")
        key_type, value_type = arguments
        decoded: dict[Any, Any] = {}
        for key, child in value.items():
            decoded_key = _decode_value(key_type, key, f"{name} key")
            if decoded_key in decoded:
                raise ContractValidationError(f"{name} contains duplicate key")
            decoded[decoded_key] = _decode_value(
                value_type, child, f"{name}.{decoded_key}"
            )
        return MappingProxyType(decoded)
    if isinstance(annotation, type) and issubclass(annotation, StrictContract):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, MappingABC):
            raise ContractValidationError(f"{name} must be an object")
        return annotation.from_dict(value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        allowed = tuple(member.value for member in annotation)
        if value not in allowed:
            raise ContractValidationError(f"{name} must be one of {allowed}")
        return annotation(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ContractValidationError(f"{name} must be a string")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ContractValidationError(f"{name} must be a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ContractValidationError(f"{name} must be an integer")
        return value
    if annotation is float:
        if type(value) is not float:
            raise ContractValidationError(f"{name} must be a floating-point number")
        return freeze_json(value, name)
    raise ContractValidationError(f"{name} has unsupported annotation {annotation}")


@dataclass(frozen=True)
class StrictContract:
    """Base for exact-key, immutable, content-verifying records."""

    CONTENT_NAMESPACE: ClassVar[str | None] = None
    IDENTITY_FIELD: ClassVar[str | None] = None
    CONTENT_EXCLUDED_FIELDS: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        annotations = get_type_hints(type(self))
        for field in fields(self):
            decoded = _decode_value(
                annotations[field.name], getattr(self, field.name), field.name
            )
            object.__setattr__(self, field.name, decoded)
        self._validate()
        if self.CONTENT_NAMESPACE is not None and self.IDENTITY_FIELD is not None:
            verify_declared_content_id(
                self.to_dict(),
                namespace=self.CONTENT_NAMESPACE,
                identity_field=self.IDENTITY_FIELD,
                excluded_fields=self.CONTENT_EXCLUDED_FIELDS,
            )

    def _validate(self) -> None:
        """Validate relations that cannot be expressed by field types."""

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: to_json_value(getattr(self, field.name))
            for field in fields(self)
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(
        cls: type[_ContractType], payload: Mapping[str, Any]
    ) -> _ContractType:
        if not isinstance(payload, MappingABC):
            raise ContractValidationError(f"{cls.__name__} must be an object")
        expected = tuple(field.name for field in fields(cls))
        missing = tuple(sorted(set(expected) - set(payload)))
        unknown = tuple(sorted(set(payload) - set(expected)))
        if missing or unknown:
            raise ContractValidationError(
                f"{cls.__name__} fields mismatch; missing={missing}, unknown={unknown}"
            )
        return cls(**{name: payload[name] for name in expected})

    @classmethod
    def from_json_bytes(
        cls: type[_ContractType], payload: bytes | str
    ) -> _ContractType:
        parsed = parse_json_bytes(payload)
        if not isinstance(parsed, MappingABC):
            raise ContractValidationError(f"{cls.__name__} JSON must be an object")
        return cls.from_dict(parsed)

    @classmethod
    def mint(cls: type[_ContractType], **payload_without_id: Any) -> _ContractType:
        if cls.CONTENT_NAMESPACE is None or cls.IDENTITY_FIELD is None:
            raise ContractValidationError(f"{cls.__name__} is not content identified")
        expected = tuple(field.name for field in fields(cls))
        expected_without_id = tuple(
            name for name in expected if name != cls.IDENTITY_FIELD
        )
        missing = tuple(sorted(set(expected_without_id) - set(payload_without_id)))
        unknown = tuple(sorted(set(payload_without_id) - set(expected_without_id)))
        if missing or unknown:
            raise ContractValidationError(
                f"{cls.__name__} mint fields mismatch; missing={missing}, "
                f"unknown={unknown}"
            )
        annotations = get_type_hints(cls)
        decoded = {
            name: _decode_value(annotations[name], payload_without_id[name], name)
            for name in expected_without_id
        }
        preimage = {
            name: to_json_value(value)
            for name, value in decoded.items()
            if name not in cls.CONTENT_EXCLUDED_FIELDS
        }
        decoded[cls.IDENTITY_FIELD] = content_id(cls.CONTENT_NAMESPACE, preimage)
        return cls(**decoded)


class ContextValueType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING_ARRAY = "string_array"


class LineageRelation(str, Enum):
    SUPERSEDES = "supersedes"
    RENAME = "rename"
    SPLIT = "split"
    MERGE = "merge"


class ObjectiveDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class TransferCompatibility(str, Enum):
    EXACT_CONTEXT = "exact_context"
    ANALOGICAL = "analogical"
    INCOMPATIBLE = "incompatible"


class CompletionState(str, Enum):
    COMPLETE = "complete"
    STOPPED = "stopped"
    CRASHED = "crashed"


class ArtifactCompleteness(str, Enum):
    PRESENT = "present"
    ABSENT_BEFORE_FRONTIER = "absent_before_frontier"
    UNAVAILABLE = "unavailable"


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED_TECHNICAL = "failed_technical"
    INTERRUPTED = "interrupted"


class EpisodeEvaluationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    NOT_RUN = "not_run"


class ComparisonStatus(str, Enum):
    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"
    INCONCLUSIVE = "inconclusive"


class InterventionStructure(str, Enum):
    COUPLED = "coupled"
    ISOLATED_BY_ABLATION = "isolated_by_ablation"
    UNDETERMINED = "undetermined"


class PriorIdeaStatus(str, Enum):
    DEFERRED = "deferred"
    REJECTED = "rejected"
    UNEXECUTED = "unexecuted"


class AdmissionState(str, Enum):
    QUARANTINED = "quarantined"
    ADMITTED = "admitted"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class CandidateChangeKind(str, Enum):
    CAPABILITY = "capability"
    REPOSITORY_ARCHITECTURE = "repository_architecture"


class ExpertCapabilityLineageRelation(str, Enum):
    RENAME = "rename"
    SPLIT = "split"
    MERGE = "merge"
    RETIRE = "retire"


class ExpertCandidateOperationKind(str, Enum):
    BOOTSTRAP = "bootstrap"
    RESTRUCTURE = "restructure"
    GENERALIZE = "generalize"


class ExpertCandidateSanitationStatus(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"


class ExpertSanitationSeverity(str, Enum):
    INFORMATIONAL = "informational"
    WARNING = "warning"
    BLOCKING = "blocking"


class PublicationArtifactKind(str, Enum):
    KNOWLEDGE_SNAPSHOT = "knowledge_snapshot"
    EXPERT_BASE_RELEASE = "expert_base_release"


@dataclass(frozen=True)
class ScopeRepositorySettings(StrictContract):
    scope_id: str
    expert_repository: str
    knowledge_repository: str

    def _validate(self) -> None:
        require_identifier(self.scope_id, "scope_id")
        for name in ("expert_repository", "knowledge_repository"):
            _require_repository_coordinate(getattr(self, name), name)
        if self.expert_repository == self.knowledge_repository:
            raise IdentityConflictError(
                "expert and knowledge repositories must be distinct"
            )

    @property
    def binding_fingerprint(self) -> str:
        return tree_or_blob_digest(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class CrossRunTaskBindingSettings(StrictContract):
    scope_id: str
    task_family_id: str
    task_adapter_id: str

    def _validate(self) -> None:
        for name in ("scope_id", "task_family_id", "task_adapter_id"):
            require_identifier(getattr(self, name), name)


@dataclass(frozen=True)
class TaskFamilyDefinition(StrictContract):
    task_family_id: str
    capability_tags: tuple[str, ...]

    def _validate(self) -> None:
        require_identifier(self.task_family_id, "task_family_id")
        _require_sorted_unique(self.capability_tags, "capability_tags")


@dataclass(frozen=True)
class ContextDimensionSchema(StrictContract):
    dimension_id: str
    value_type: ContextValueType
    required: bool

    def _validate(self) -> None:
        require_identifier(self.dimension_id, "dimension_id")

    def validate_value(self, value: Any) -> None:
        if self.value_type is ContextValueType.STRING and not isinstance(value, str):
            raise ContractValidationError(f"{self.dimension_id} must be a string")
        if self.value_type is ContextValueType.INTEGER and type(value) is not int:
            raise ContractValidationError(f"{self.dimension_id} must be an integer")
        if self.value_type is ContextValueType.NUMBER and (
            type(value) not in (int, float)
        ):
            raise ContractValidationError(f"{self.dimension_id} must be numeric")
        if self.value_type is ContextValueType.BOOLEAN and type(value) is not bool:
            raise ContractValidationError(f"{self.dimension_id} must be a boolean")
        if self.value_type is ContextValueType.STRING_ARRAY:
            if not isinstance(value, tuple) or any(
                not isinstance(item, str) for item in value
            ):
                raise ContractValidationError(
                    f"{self.dimension_id} must be a string array"
                )


@dataclass(frozen=True)
class LineageEdge(StrictContract):
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    relation: LineageRelation

    def _validate(self) -> None:
        _require_sorted_unique(self.source_ids, "source_ids")
        _require_sorted_unique(self.target_ids, "target_ids")
        if set(self.source_ids) & set(self.target_ids):
            raise ContractValidationError("lineage source and target IDs must differ")
        if self.relation in (LineageRelation.SUPERSEDES, LineageRelation.RENAME):
            valid_shape = len(self.source_ids) == 1 and len(self.target_ids) == 1
        elif self.relation is LineageRelation.SPLIT:
            valid_shape = len(self.source_ids) == 1 and len(self.target_ids) > 1
        else:
            valid_shape = len(self.source_ids) > 1 and len(self.target_ids) == 1
        if not valid_shape:
            raise ContractValidationError(
                f"invalid {self.relation.value} lineage cardinality"
            )


@dataclass(frozen=True)
class TaskAdapterBinding(StrictContract):
    task_family_id: str
    task_adapter_ids: tuple[str, ...]

    def _validate(self) -> None:
        require_identifier(self.task_family_id, "task_family_id")
        _require_sorted_unique(self.task_adapter_ids, "task_adapter_ids")


@dataclass(frozen=True)
class ExpertScopeContract(StrictContract):
    scope_contract_id: str
    scope_id: str
    supersedes_scope_contract_id: str | None
    purpose: str
    explicit_non_goals: tuple[str, ...]
    task_family_ontology: tuple[TaskFamilyDefinition, ...]
    task_family_lineage: tuple[LineageEdge, ...]
    artifact_classes: tuple[str, ...]
    required_context_dimensions: tuple[str, ...]
    context_dimension_schemas: tuple[ContextDimensionSchema, ...]
    context_dimension_lineage: tuple[LineageEdge, ...]
    task_adapter_contract: tuple[TaskAdapterBinding, ...]
    sanitation_policy_ref: str
    validation_policy_ref: str
    repository_architecture_constraints: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-scope-contract"
    IDENTITY_FIELD: ClassVar[str] = "scope_contract_id"

    def _validate(self) -> None:
        require_identifier(self.scope_id, "scope_id")
        if self.supersedes_scope_contract_id is not None:
            require_content_id(
                self.supersedes_scope_contract_id, "supersedes_scope_contract_id"
            )
        _require_text(self.purpose, "purpose")
        _require_text_tuple(
            self.explicit_non_goals, "explicit_non_goals", required=True
        )
        _require_text_tuple(
            self.repository_architecture_constraints,
            "repository_architecture_constraints",
            required=True,
        )
        require_identifier(self.sanitation_policy_ref, "sanitation_policy_ref")
        require_identifier(self.validation_policy_ref, "validation_policy_ref")
        _require_sorted_unique(self.artifact_classes, "artifact_classes")
        family_ids = tuple(item.task_family_id for item in self.task_family_ontology)
        dimension_ids = tuple(
            item.dimension_id for item in self.context_dimension_schemas
        )
        if not family_ids or len(family_ids) != len(set(family_ids)):
            raise ContractValidationError(
                "task family ontology must be non-empty and unique"
            )
        if not dimension_ids or len(dimension_ids) != len(set(dimension_ids)):
            raise ContractValidationError(
                "context dimension schemas must be non-empty and unique"
            )
        if tuple(sorted(family_ids)) != family_ids:
            raise ContractValidationError("task_family_ontology must be sorted")
        if tuple(sorted(dimension_ids)) != dimension_ids:
            raise ContractValidationError("context_dimension_schemas must be sorted")
        _require_sorted_unique(
            self.required_context_dimensions, "required_context_dimensions"
        )
        if not set(self.required_context_dimensions).issubset(dimension_ids):
            raise MissingReferenceError(
                "required context dimension is absent from its schema registry"
            )
        binding_families = tuple(
            binding.task_family_id for binding in self.task_adapter_contract
        )
        if tuple(sorted(binding_families)) != binding_families:
            raise ContractValidationError("task_adapter_contract must be sorted")
        if set(binding_families) != set(family_ids):
            raise MissingReferenceError(
                "every task family must have exactly one adapter binding"
            )
        adapter_ids = tuple(
            adapter_id
            for binding in self.task_adapter_contract
            for adapter_id in binding.task_adapter_ids
        )
        if len(adapter_ids) != len(set(adapter_ids)):
            raise IdentityConflictError("task adapter IDs must be unique in a scope")
        current_families = set(family_ids)
        current_dimensions = set(dimension_ids)
        for edge in self.task_family_lineage:
            if not set(edge.target_ids).issubset(current_families):
                raise MissingReferenceError("task-family lineage target is not current")
        for edge in self.context_dimension_lineage:
            if not set(edge.target_ids).issubset(current_dimensions):
                raise MissingReferenceError("context lineage target is not current")

    def validate_binding(self, binding: CrossRunTaskBindingSettings) -> None:
        if binding.scope_id != self.scope_id:
            raise IncompatibleArtifactError("task binding names a different scope")
        bindings = {
            item.task_family_id: item.task_adapter_ids
            for item in self.task_adapter_contract
        }
        if binding.task_family_id not in bindings:
            raise IncompatibleArtifactError("unknown task family for scope")
        if binding.task_adapter_id not in bindings[binding.task_family_id]:
            raise IncompatibleArtifactError("unknown adapter for task family")


@dataclass(frozen=True)
class TaskContextBinding(StrictContract):
    task_context_binding_id: str
    scope_contract_id: str
    scope_id: str
    task_family_id: str
    task_adapter_id: str
    capability_tags: tuple[str, ...]
    input_contract_fingerprint: str
    target_contract_fingerprint: str
    starting_artifact_refs: tuple[str, ...]
    method_fingerprint: str
    toolchain_fingerprint: str
    dependency_runtime_fingerprint: str
    budget_hardware_envelope: Mapping[str, Any]
    transfer_dimensions: Mapping[str, Any]

    CONTENT_NAMESPACE: ClassVar[str] = "task-context-binding"
    IDENTITY_FIELD: ClassVar[str] = "task_context_binding_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        for name in ("scope_id", "task_family_id", "task_adapter_id"):
            require_identifier(getattr(self, name), name)
        _require_sorted_unique(self.capability_tags, "capability_tags")
        for name in (
            "input_contract_fingerprint",
            "target_contract_fingerprint",
            "method_fingerprint",
            "toolchain_fingerprint",
            "dependency_runtime_fingerprint",
        ):
            _require_digest(getattr(self, name), name)
        if self.starting_artifact_refs:
            _require_sorted_unique(
                self.starting_artifact_refs, "starting_artifact_refs"
            )
        if not self.budget_hardware_envelope:
            raise ContractValidationError("budget_hardware_envelope must not be empty")

    def validate_against(self, scope_contract: ExpertScopeContract) -> None:
        if self.scope_contract_id != scope_contract.scope_contract_id:
            raise IncompatibleArtifactError("context uses a different scope revision")
        scope_contract.validate_binding(
            CrossRunTaskBindingSettings(
                scope_id=self.scope_id,
                task_family_id=self.task_family_id,
                task_adapter_id=self.task_adapter_id,
            )
        )
        schemas = {
            schema.dimension_id: schema
            for schema in scope_contract.context_dimension_schemas
        }
        unknown = set(self.transfer_dimensions) - set(schemas)
        missing = set(scope_contract.required_context_dimensions) - set(
            self.transfer_dimensions
        )
        if unknown or missing:
            raise ContractValidationError(
                f"transfer dimensions mismatch; missing={tuple(sorted(missing))}, "
                f"unknown={tuple(sorted(unknown))}"
            )
        for dimension_id, value in self.transfer_dimensions.items():
            schemas[dimension_id].validate_value(value)

    def compatibility_with(self, other: TaskContextBinding) -> TransferCompatibility:
        if (
            self.scope_id != other.scope_id
            or self.scope_contract_id != other.scope_contract_id
        ):
            return TransferCompatibility.INCOMPATIBLE
        self_payload = self.to_dict()
        other_payload = other.to_dict()
        self_payload.pop("task_context_binding_id")
        other_payload.pop("task_context_binding_id")
        if self_payload == other_payload:
            return TransferCompatibility.EXACT_CONTEXT
        return TransferCompatibility.ANALOGICAL


@dataclass(frozen=True)
class EvaluationFingerprint(StrictContract):
    evaluation_fingerprint_id: str
    benchmark_id: str
    dataset_version: str
    split_version: str
    evaluator_fingerprint: str
    metric_name: str
    objective_direction: ObjectiveDirection
    fidelity: str
    fraction: float
    seed_or_replicate_ids: tuple[str, ...]
    aggregation_protocol: str
    judge_version: str | None

    CONTENT_NAMESPACE: ClassVar[str] = "evaluation-fingerprint"
    IDENTITY_FIELD: ClassVar[str] = "evaluation_fingerprint_id"

    def _validate(self) -> None:
        for name in ("benchmark_id", "dataset_version", "split_version", "metric_name"):
            require_identifier(getattr(self, name), name)
        _require_digest(self.evaluator_fingerprint, "evaluator_fingerprint")
        _require_text(self.fidelity, "fidelity")
        if not 0.0 < self.fraction <= 1.0:
            raise ContractValidationError("fraction must be in (0, 1]")
        _require_sorted_unique(self.seed_or_replicate_ids, "seed_or_replicate_ids")
        _require_text(self.aggregation_protocol, "aggregation_protocol")
        if self.judge_version is not None:
            _require_text(self.judge_version, "judge_version")

    def comparable_with(self, other: EvaluationFingerprint) -> bool:
        self_payload = self.to_dict()
        other_payload = other.to_dict()
        self_payload.pop("evaluation_fingerprint_id")
        other_payload.pop("evaluation_fingerprint_id")
        return self_payload == other_payload


@dataclass(frozen=True)
class ArtifactEnvironment(StrictContract):
    artifact_environment_id: str
    kapso_commit: str
    expert_base_release_id: str
    task_adapter_hash: str
    dependency_lock_hash: str

    CONTENT_NAMESPACE: ClassVar[str] = "artifact-environment"
    IDENTITY_FIELD: ClassVar[str] = "artifact_environment_id"

    def _validate(self) -> None:
        _require_text(self.kapso_commit, "kapso_commit")
        require_content_id(self.expert_base_release_id, "expert_base_release_id")
        _require_digest(self.task_adapter_hash, "task_adapter_hash")
        _require_digest(self.dependency_lock_hash, "dependency_lock_hash")


@dataclass(frozen=True)
class CaptureManifest(StrictContract):
    capture_manifest_id: str
    scope_contract_id: str
    scope_id: str
    run_id: str
    campaign_id: str
    capture_generation: int
    supersedes_capture_manifest_id: str | None
    checkpoint_frontier: int
    capture_watermarks: Mapping[str, int]
    configuration_fingerprint: str
    artifact_refs: Mapping[str, str]
    checksums: Mapping[str, str]
    captured_at: str

    CONTENT_NAMESPACE: ClassVar[str] = "capture-manifest"
    IDENTITY_FIELD: ClassVar[str] = "capture_manifest_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        for name in ("scope_id", "run_id", "campaign_id"):
            require_identifier(getattr(self, name), name)
        if self.capture_generation < 0 or self.checkpoint_frontier < 0:
            raise ContractValidationError(
                "capture generation/frontier must be non-negative"
            )
        if self.supersedes_capture_manifest_id is not None:
            require_content_id(
                self.supersedes_capture_manifest_id,
                "supersedes_capture_manifest_id",
            )
        if any(
            type(value) is not int or value < 0
            for value in self.capture_watermarks.values()
        ):
            raise ContractValidationError(
                "capture watermarks must be non-negative integers"
            )
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        if not self.artifact_refs:
            raise ContractValidationError("artifact_refs must not be empty")
        _require_checksum_mapping(self.checksums, "checksums")
        missing_checksums = set(self.artifact_refs.values()) - set(self.checksums)
        if missing_checksums:
            raise MissingReferenceError(
                f"capture artifact checksums are missing: {tuple(sorted(missing_checksums))}"
            )
        normalize_utc_timestamp(self.captured_at, "captured_at")


@dataclass(frozen=True)
class RunBundle(StrictContract):
    bundle_id: str
    scope_contract_id: str
    scope_id: str
    run_id: str
    campaign_id: str
    completion_state: CompletionState
    capture_generation: int
    supersedes_bundle_id: str | None
    checkpoint_frontier: int
    capture_watermarks: Mapping[str, int]
    configuration_fingerprint: str
    artifact_completeness: Mapping[str, ArtifactCompleteness]
    started_at: str
    captured_at: str
    kapso_commit: str
    launch_manifest_id: str
    knowledge_snapshot_id: str
    expert_base_release_id: str
    task_context_binding: TaskContextBinding
    artifact_environment: ArtifactEnvironment
    capture_descriptor_ref: str
    checkpoint_ref: str
    execution_event_journal_ref: str
    idea_archive_ref: str
    experiment_history_ref: str
    sanitation_report_ref: str
    branch_snapshot_refs: tuple[str, ...]
    run_log_refs: tuple[str, ...]
    checksums: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "run-bundle"
    IDENTITY_FIELD: ClassVar[str] = "bundle_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        for name in ("scope_id", "run_id", "campaign_id"):
            require_identifier(getattr(self, name), name)
        if self.capture_generation < 0 or self.checkpoint_frontier < 0:
            raise ContractValidationError(
                "capture generation/frontier must be non-negative"
            )
        if self.supersedes_bundle_id is not None:
            require_content_id(self.supersedes_bundle_id, "supersedes_bundle_id")
        if any(
            type(value) is not int or value < 0
            for value in self.capture_watermarks.values()
        ):
            raise ContractValidationError(
                "capture watermarks must be non-negative integers"
            )
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        if not self.artifact_completeness:
            raise ContractValidationError("artifact_completeness must not be empty")
        started_at = parse_utc_timestamp(self.started_at, "started_at")
        captured_at = parse_utc_timestamp(self.captured_at, "captured_at")
        if started_at > captured_at:
            raise ContractValidationError("run cannot be captured before it starts")
        _require_text(self.kapso_commit, "kapso_commit")
        for name in (
            "launch_manifest_id",
            "knowledge_snapshot_id",
            "expert_base_release_id",
        ):
            require_content_id(getattr(self, name), name)
        if self.task_context_binding.scope_contract_id != self.scope_contract_id:
            raise IncompatibleArtifactError(
                "bundle context uses another scope revision"
            )
        if self.task_context_binding.scope_id != self.scope_id:
            raise IncompatibleArtifactError("bundle context uses another scope")
        if (
            self.artifact_environment.expert_base_release_id
            != self.expert_base_release_id
        ):
            raise IncompatibleArtifactError(
                "bundle environment uses another expert release"
            )
        if self.artifact_environment.kapso_commit != self.kapso_commit:
            raise IncompatibleArtifactError(
                "bundle environment uses another Kapso commit"
            )
        for name in (
            "capture_descriptor_ref",
            "checkpoint_ref",
            "execution_event_journal_ref",
            "idea_archive_ref",
            "experiment_history_ref",
            "sanitation_report_ref",
        ):
            _require_relative_path(getattr(self, name), name)
        for name in ("branch_snapshot_refs", "run_log_refs"):
            for value in getattr(self, name):
                _require_relative_path(value, name)
        _require_checksum_mapping(self.checksums, "checksums")
        referenced_paths = {
            self.checkpoint_ref,
            self.capture_descriptor_ref,
            self.execution_event_journal_ref,
            self.idea_archive_ref,
            self.experiment_history_ref,
            self.sanitation_report_ref,
            *self.branch_snapshot_refs,
            *self.run_log_refs,
        }
        missing_checksums = referenced_paths - set(self.checksums)
        if missing_checksums:
            raise MissingReferenceError(
                f"bundle artifact checksums are missing: {tuple(sorted(missing_checksums))}"
            )


class EffectUncertaintyMethod(str, Enum):
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BundleArtifactRef(StrictContract):
    relative_path: str
    checksum: str

    def _validate(self) -> None:
        _require_relative_path(self.relative_path, "bundle artifact relative_path")
        _require_digest(self.checksum, "bundle artifact checksum")


@dataclass(frozen=True)
class RelativeEffect(StrictContract):
    evaluation_fingerprint_id: str
    metric_name: str
    objective_direction: ObjectiveDirection
    candidate_value: float
    source_parent_value: float
    raw_delta: float
    normalized_delta: float
    uncertainty: float | None
    uncertainty_method: EffectUncertaintyMethod

    def _validate(self) -> None:
        require_content_id(
            self.evaluation_fingerprint_id,
            "relative effect evaluation_fingerprint_id",
        )
        require_identifier(self.metric_name, "relative effect metric_name")
        expected_raw = self.candidate_value - self.source_parent_value
        if self.raw_delta != expected_raw:
            raise ContractValidationError("relative effect raw delta is inconsistent")
        direction = (
            1.0 if self.objective_direction is ObjectiveDirection.MAXIMIZE else -1.0
        )
        if self.normalized_delta != direction * self.raw_delta:
            raise ContractValidationError(
                "relative effect normalized delta is inconsistent"
            )
        if self.uncertainty_method is EffectUncertaintyMethod.UNAVAILABLE:
            if self.uncertainty is not None:
                raise ContractValidationError(
                    "unavailable relative-effect uncertainty must be null"
                )
        elif self.uncertainty is None or self.uncertainty < 0.0:
            raise ContractValidationError(
                "estimated relative-effect uncertainty must be non-negative"
            )


@dataclass(frozen=True)
class TransferAttempt(StrictContract):
    execution_revision: int
    captured_at: str
    execution_status: ExecutionStatus
    evaluation_status: EpisodeEvaluationStatus
    evaluation_fingerprints: tuple[EvaluationFingerprint, ...]
    score_of_record_fingerprint_id: str | None
    comparison_status: ComparisonStatus
    measurements: Mapping[str, Any]
    source_parent_effect: RelativeEffect | None
    intervention_ref: BundleArtifactRef | None
    intervention_structure: InterventionStructure
    feedback: tuple[str, ...]
    technical_difficulties: tuple[str, ...]
    confounders: tuple[str, ...]

    def _validate(self) -> None:
        if self.execution_revision < 0:
            raise ContractValidationError("execution_revision must be non-negative")
        normalize_utc_timestamp(self.captured_at, "captured_at")
        fingerprint_ids = tuple(
            fingerprint.evaluation_fingerprint_id
            for fingerprint in self.evaluation_fingerprints
        )
        if fingerprint_ids != tuple(sorted(set(fingerprint_ids))):
            raise ContractValidationError(
                "attempt evaluation fingerprints must be sorted and unique"
            )
        if self.evaluation_status is EpisodeEvaluationStatus.NOT_RUN and (
            self.evaluation_fingerprints
            or self.score_of_record_fingerprint_id is not None
            or self.measurements
        ):
            raise ContractValidationError(
                "not-run evaluation cannot have fingerprints or measurements"
            )
        if (
            self.evaluation_status is EpisodeEvaluationStatus.VALID
            and self.score_of_record_fingerprint_id not in fingerprint_ids
        ):
            raise ContractValidationError(
                "valid evaluation requires one score-of-record fingerprint"
            )
        if (
            self.evaluation_status
            in {EpisodeEvaluationStatus.INVALID, EpisodeEvaluationStatus.PARTIAL}
            and self.score_of_record_fingerprint_id is not None
        ):
            raise ContractValidationError(
                "non-valid evaluation cannot name a score-of-record fingerprint"
            )
        if (
            self.comparison_status is ComparisonStatus.COMPARABLE
            and self.evaluation_status is not EpisodeEvaluationStatus.VALID
        ):
            raise ContractValidationError(
                "comparable attempt requires a valid evaluation"
            )
        if self.comparison_status is ComparisonStatus.COMPARABLE and (
            self.source_parent_effect is None
            or self.source_parent_effect.evaluation_fingerprint_id
            != self.score_of_record_fingerprint_id
        ):
            raise ContractValidationError(
                "comparable attempt requires a source-parent effect"
            )
        if self.source_parent_effect is not None:
            score_fingerprint = next(
                (
                    fingerprint
                    for fingerprint in self.evaluation_fingerprints
                    if fingerprint.evaluation_fingerprint_id
                    == self.score_of_record_fingerprint_id
                ),
                None,
            )
            if (
                score_fingerprint is None
                or self.source_parent_effect.metric_name
                != score_fingerprint.metric_name
                or self.measurements.get(score_fingerprint.metric_name)
                != self.source_parent_effect.candidate_value
            ):
                raise ContractValidationError(
                    "source-parent effect does not match the score of record"
                )
        if (
            self.comparison_status is not ComparisonStatus.COMPARABLE
            and self.source_parent_effect is not None
        ):
            raise ContractValidationError(
                "non-comparable attempt cannot claim a source-parent effect"
            )
        if (
            self.evaluation_status is EpisodeEvaluationStatus.VALID
            and not self.measurements
        ):
            raise ContractValidationError("valid evaluation requires measurements")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for name, value in self.measurements.items()
        ):
            raise ContractValidationError(
                "attempt measurements must be finite named numbers"
            )
        if (
            self.execution_status is ExecutionStatus.COMPLETED
            and self.intervention_ref is None
        ):
            raise ContractValidationError(
                "completed execution requires an intervention artifact"
            )
        if (
            self.intervention_ref is None
            and self.intervention_structure is not InterventionStructure.UNDETERMINED
        ):
            raise ContractValidationError(
                "missing intervention cannot claim a known structure"
            )
        for name in ("feedback", "technical_difficulties", "confounders"):
            _require_text_tuple(getattr(self, name), name)


@dataclass(frozen=True)
class TransferEpisode(StrictContract):
    episode_id: str
    source: Mapping[str, str]
    source_bundle_id: str
    supersedes_projection_id: str | None
    task_context_binding: TaskContextBinding
    artifact_environment: ArtifactEnvironment
    proposal: str
    parent_episode_ref: str | None
    attempts: tuple[TransferAttempt, ...]
    terminal_attempt_revision: int
    safe_observation_refs: tuple[BundleArtifactRef, ...]
    sanitation_report_id: str
    derivation_refs: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "transfer-episode"
    IDENTITY_FIELD: ClassVar[str] = "episode_id"

    def _validate(self) -> None:
        required_source_keys = {
            "scope_id",
            "run_id",
            "campaign_id",
            "node_id",
            "idea_id",
            "batch_id",
        }
        if set(self.source) != required_source_keys:
            raise ContractValidationError(
                "episode source must contain exact globally qualified keys"
            )
        for key, value in self.source.items():
            require_identifier(value, f"source.{key}")
        if self.source["scope_id"] != self.task_context_binding.scope_id:
            raise IncompatibleArtifactError("episode source uses another scope")
        require_content_id(self.source_bundle_id, "source_bundle_id")
        if self.supersedes_projection_id is not None:
            require_content_id(
                self.supersedes_projection_id,
                "supersedes_projection_id",
            )
            if self.supersedes_projection_id == self.episode_id:
                raise ContractValidationError("episode cannot supersede itself")
        _require_text(self.proposal, "proposal")
        if self.parent_episode_ref is not None:
            require_content_id(self.parent_episode_ref, "parent_episode_ref")
        if not self.attempts:
            raise ContractValidationError("episode must contain at least one attempt")
        revisions = tuple(attempt.execution_revision for attempt in self.attempts)
        if revisions != tuple(range(len(self.attempts))):
            raise ContractValidationError(
                "attempt revisions must be ordered and gap-free"
            )
        if self.terminal_attempt_revision != revisions[-1]:
            raise ContractValidationError(
                "terminal_attempt_revision must name the final attempt"
            )
        require_content_id(self.sanitation_report_id, "sanitation_report_id")
        observation_keys = tuple(
            (reference.relative_path, reference.checksum)
            for reference in self.safe_observation_refs
        )
        if observation_keys != tuple(sorted(set(observation_keys))):
            raise ContractValidationError(
                "safe observation refs must be sorted and unique"
            )
        if self.derivation_refs:
            _require_sorted_unique(self.derivation_refs, "derivation_refs")
            for derivation_ref in self.derivation_refs:
                require_content_id(derivation_ref, "derivation_ref")


@dataclass(frozen=True)
class PriorIdea(StrictContract):
    prior_idea_id: str
    source_bundle_id: str
    supersedes_projection_id: str | None
    source: Mapping[str, str]
    proposal: str
    descriptor: Mapping[str, str]
    assumptions: tuple[str, ...]
    source_status: PriorIdeaStatus
    source_rationale: str
    source_evidence_refs: tuple[str, ...]
    task_context_binding: TaskContextBinding
    sanitation_report_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "prior-idea"
    IDENTITY_FIELD: ClassVar[str] = "prior_idea_id"

    def _validate(self) -> None:
        required_source_keys = {
            "scope_id",
            "run_id",
            "campaign_id",
            "batch_id",
            "idea_id",
        }
        if set(self.source) != required_source_keys:
            raise ContractValidationError(
                "prior idea source must contain campaign, batch, and idea IDs"
            )
        for key, value in self.source.items():
            require_identifier(value, f"source.{key}")
        if self.source["scope_id"] != self.task_context_binding.scope_id:
            raise IncompatibleArtifactError("prior idea source uses another scope")
        require_content_id(self.source_bundle_id, "source_bundle_id")
        if self.supersedes_projection_id is not None:
            require_content_id(
                self.supersedes_projection_id,
                "supersedes_projection_id",
            )
            if self.supersedes_projection_id == self.prior_idea_id:
                raise ContractValidationError("prior idea cannot supersede itself")
        _require_text(self.proposal, "proposal")
        descriptor_keys = {
            "approach_family",
            "expected_effect",
            "intervention_target",
            "mechanism",
        }
        if set(self.descriptor) != descriptor_keys or any(
            not isinstance(value, str) or not value
            for value in self.descriptor.values()
        ):
            raise ContractValidationError("prior idea descriptor is invalid")
        _require_text_tuple(self.assumptions, "assumptions")
        _require_text(self.source_rationale, "source_rationale")
        if self.source_evidence_refs:
            _require_sorted_unique(
                self.source_evidence_refs,
                "source_evidence_refs",
            )
        require_content_id(self.sanitation_report_id, "sanitation_report_id")


@dataclass(frozen=True)
class CodingAgentWorkspaceChangedFile(StrictContract):
    before: "SourceFileDescriptor | None"
    after: "SourceFileDescriptor"
    content_base64: str

    def _validate(self) -> None:
        if self.before is not None and (
            self.before.relative_path != self.after.relative_path
            or self.before == self.after
        ):
            raise ContractValidationError(
                "coding-agent workspace change has an invalid preimage"
            )
        if not isinstance(self.content_base64, str):
            raise ContractValidationError(
                "coding-agent workspace content must be base64 text"
            )
        content = base64.b64decode(self.content_base64, validate=True)
        if base64.b64encode(content).decode("ascii") != self.content_base64:
            raise ContractValidationError(
                "coding-agent workspace content must use canonical base64"
            )
        if (
            tree_or_blob_digest(content) != self.after.digest
            or len(content) != self.after.size
        ):
            raise ContractValidationError(
                "coding-agent workspace content differs from its descriptor"
            )

    @property
    def relative_path(self) -> str:
        return self.after.relative_path

    @property
    def content(self) -> bytes:
        return base64.b64decode(self.content_base64, validate=True)


@dataclass(frozen=True)
class CodingAgentWorkspaceDelta(StrictContract):
    workspace_delta_id: str
    baseline_tree_hash: str
    edited_tree_hash: str
    changed_files: tuple[CodingAgentWorkspaceChangedFile, ...]
    deleted_files: tuple["SourceFileDescriptor", ...]

    CONTENT_NAMESPACE: ClassVar[str] = "coding-agent-workspace-delta"
    IDENTITY_FIELD: ClassVar[str] = "workspace_delta_id"

    def _validate(self) -> None:
        _require_digest(self.baseline_tree_hash, "workspace baseline_tree_hash")
        _require_digest(self.edited_tree_hash, "workspace edited_tree_hash")
        if self.baseline_tree_hash == self.edited_tree_hash:
            raise ContractValidationError(
                "coding-agent workspace delta must change the tree"
            )
        changed_paths = tuple(change.relative_path for change in self.changed_files)
        deleted_paths = tuple(file.relative_path for file in self.deleted_files)
        for paths, name in (
            (changed_paths, "changed files"),
            (deleted_paths, "deleted files"),
        ):
            if paths != tuple(sorted(set(paths))):
                raise ContractValidationError(
                    f"coding-agent workspace {name} must be sorted and unique"
                )
        if not changed_paths and not deleted_paths:
            raise ContractValidationError(
                "coding-agent workspace delta contains no change"
            )
        if set(changed_paths) & set(deleted_paths):
            raise ContractValidationError(
                "coding-agent workspace changed and deleted files overlap"
            )

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(change.relative_path for change in self.changed_files)

    @property
    def deleted_paths(self) -> tuple[str, ...]:
        return tuple(file.relative_path for file in self.deleted_files)


@dataclass(frozen=True)
class CodingAgentOperationReceipt(StrictContract):
    operation_receipt_id: str
    operation_id: str
    principal_id: str
    role: str
    cli: str
    model: str
    effort: str
    workspace_access: CodingAgentWorkspaceAccess
    artifact_checksums: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "coding-agent-operation-receipt"
    IDENTITY_FIELD: ClassVar[str] = "operation_receipt_id"

    def _validate(self) -> None:
        if _CODING_AGENT_OPERATION_PATTERN.fullmatch(self.operation_id) is None:
            raise ContractValidationError("invalid coding-agent operation ID")
        for value, name in (
            (self.principal_id, "coding-agent principal_id"),
            (self.role, "coding-agent role"),
            (self.effort, "coding-agent effort"),
        ):
            require_identifier(value, name)
        if self.cli not in {"codex", "claude_code"}:
            raise ContractValidationError("invalid coding-agent CLI")
        if not isinstance(self.workspace_access, CodingAgentWorkspaceAccess):
            raise ContractValidationError("invalid coding-agent access mode")
        _require_text(self.model, "coding-agent model")
        _require_checksum_mapping(
            self.artifact_checksums,
            "coding-agent artifact_checksums",
        )
        if set(self.artifact_checksums) != set(
            coding_agent_artifact_filenames(self.workspace_access)
        ):
            raise MissingReferenceError(
                "coding-agent receipt requires the complete artifact set"
            )


@dataclass(frozen=True)
class ReviewAssertion(StrictContract):
    assertion_id: str
    subject_id: str
    reviewer_id: str
    reviewer_role: str
    rubric_version: str
    judgment: str
    rationale: str
    exact_evidence_refs: tuple[str, ...]
    supersedes_assertion_id: str | None
    review_operation_ref: str

    CONTENT_NAMESPACE: ClassVar[str] = "review-assertion"
    IDENTITY_FIELD: ClassVar[str] = "assertion_id"

    def _validate(self) -> None:
        require_content_id(self.subject_id, "subject_id")
        for name in ("reviewer_id", "reviewer_role", "rubric_version", "judgment"):
            require_identifier(getattr(self, name), name)
        _require_text(self.rationale, "rationale")
        _require_sorted_unique(self.exact_evidence_refs, "exact_evidence_refs")
        if self.supersedes_assertion_id is not None:
            require_content_id(self.supersedes_assertion_id, "supersedes_assertion_id")
        require_content_id(self.review_operation_ref, "review_operation_ref")


@dataclass(frozen=True)
class KnowledgeClaim(StrictContract):
    claim_id: str
    revision_id: str
    scope_contract_id: str
    statement: str
    mechanism: str
    applicability_predicates: Mapping[str, Any]
    explicit_exclusions: tuple[str, ...]
    supporting_episode_ids: tuple[str, ...]
    contradicting_episode_ids: tuple[str, ...]
    proposal_provenance: Mapping[str, Any]
    supersedes_revision_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "knowledge-claim-revision"
    IDENTITY_FIELD: ClassVar[str] = "revision_id"

    def _validate(self) -> None:
        require_identifier(self.claim_id, "claim_id")
        require_content_id(self.scope_contract_id, "scope_contract_id")
        _require_text(self.statement, "statement")
        _require_text(self.mechanism, "mechanism")
        if not self.applicability_predicates:
            raise ContractValidationError("claim applicability must not be empty")
        _require_text_tuple(
            self.explicit_exclusions, "explicit_exclusions", required=True
        )
        if not self.supporting_episode_ids and not self.contradicting_episode_ids:
            raise ContractValidationError(
                "claim must reference support or contradiction"
            )
        for name in (
            "supporting_episode_ids",
            "contradicting_episode_ids",
            "supersedes_revision_ids",
        ):
            values = getattr(self, name)
            if values:
                _require_sorted_unique(values, name)
                for value in values:
                    require_content_id(value, name)
        if set(self.supporting_episode_ids) & set(self.contradicting_episode_ids):
            raise ContractValidationError(
                "one episode cannot simultaneously support and contradict a claim"
            )
        if not self.proposal_provenance:
            raise ContractValidationError("proposal_provenance must not be empty")
        if self.revision_id in self.supersedes_revision_ids:
            raise ContractValidationError("claim revision cannot supersede itself")


@dataclass(frozen=True)
class CatalogEntryState(StrictContract):
    catalog_entry_state_id: str
    subject_payload_id: str
    catalog_generation: int
    predecessor_state_id: str | None
    configuration_fingerprint: str
    admission_state: AdmissionState
    superseded_by_payload_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...]
    revocation_ids: tuple[str, ...]
    taint_source_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "catalog-entry-state"
    IDENTITY_FIELD: ClassVar[str] = "catalog_entry_state_id"

    def _validate(self) -> None:
        require_content_id(self.subject_payload_id, "subject_payload_id")
        if self.catalog_generation < 0:
            raise ContractValidationError("catalog_generation must be non-negative")
        if self.predecessor_state_id is not None:
            require_content_id(self.predecessor_state_id, "predecessor_state_id")
            if self.predecessor_state_id == self.catalog_entry_state_id:
                raise ContractValidationError(
                    "catalog entry state cannot precede itself"
                )
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        for name in (
            "superseded_by_payload_ids",
            "assertion_ids",
            "revocation_ids",
            "taint_source_ids",
        ):
            values = getattr(self, name)
            if values:
                _require_sorted_unique(values, name)
                for value in values:
                    require_content_id(value, name)
        has_revocation_or_taint = bool(self.revocation_ids or self.taint_source_ids)
        if (self.admission_state is AdmissionState.REVOKED) != has_revocation_or_taint:
            raise ContractValidationError(
                "revoked state must match revocation or taint evidence"
            )
        if self.admission_state is AdmissionState.SUPERSEDED and not (
            self.superseded_by_payload_ids
        ):
            raise ContractValidationError(
                "superseded state must name successor payloads"
            )
        if self.superseded_by_payload_ids and self.admission_state not in {
            AdmissionState.SUPERSEDED,
            AdmissionState.REVOKED,
        }:
            raise ContractValidationError(
                "successor payloads require superseded or revoked state"
            )
        if self.admission_state is AdmissionState.ADMITTED and has_revocation_or_taint:
            raise ContractValidationError(
                "admitted state cannot carry revocation or taint"
            )


@dataclass(frozen=True)
class EmbeddingSidecar(StrictContract):
    embedding_space_id: str
    asset_ref: str
    checksum: str

    def _validate(self) -> None:
        require_content_id(self.embedding_space_id, "embedding_space_id")
        _require_text(self.asset_ref, "asset_ref")
        _require_digest(self.checksum, "checksum")


@dataclass(frozen=True)
class KnowledgeSnapshotManifest(StrictContract):
    snapshot_id: str
    scope_contract_id: str
    scope_id: str
    parent_snapshot_ids: tuple[str, ...]
    included_bundle_ids: tuple[str, ...]
    admitted_episode_ids: tuple[str, ...]
    admitted_prior_idea_ids: tuple[str, ...]
    active_claim_revision_ids: tuple[str, ...]
    catalog_generation: int
    configuration_fingerprint: str
    entry_state_refs: tuple[str, ...]
    included_assertion_ids: tuple[str, ...]
    included_revocation_ids: tuple[str, ...]
    proof_dependency_closure_ids: tuple[str, ...]
    sanitation_policy_version: str
    retrieval_policy_version: str
    embedding_sidecars: tuple[EmbeddingSidecar, ...]
    prompt_budget_policy: Mapping[str, Any]
    checksums: Mapping[str, str]
    published_at: str
    publisher_attestation: Mapping[str, Any]

    CONTENT_NAMESPACE: ClassVar[str] = "knowledge-snapshot"
    IDENTITY_FIELD: ClassVar[str] = "snapshot_id"
    CONTENT_EXCLUDED_FIELDS: ClassVar[tuple[str, ...]] = ("publisher_attestation",)

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        require_identifier(self.scope_id, "scope_id")
        if self.catalog_generation < 0:
            raise ContractValidationError("catalog_generation must be non-negative")
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        for name in (
            "parent_snapshot_ids",
            "included_bundle_ids",
            "admitted_episode_ids",
            "admitted_prior_idea_ids",
            "active_claim_revision_ids",
            "entry_state_refs",
            "included_assertion_ids",
            "included_revocation_ids",
            "proof_dependency_closure_ids",
        ):
            values = getattr(self, name)
            if values:
                _require_sorted_unique(values, name)
                for value in values:
                    require_content_id(value, name)
        required_proof_ids = set(self.admitted_episode_ids)
        required_proof_ids.update(self.admitted_prior_idea_ids)
        required_proof_ids.update(self.active_claim_revision_ids)
        required_proof_ids.update(self.entry_state_refs)
        required_proof_ids.update(self.included_assertion_ids)
        required_proof_ids.update(self.included_revocation_ids)
        if not required_proof_ids.issubset(self.proof_dependency_closure_ids):
            raise MissingReferenceError("snapshot proof closure is incomplete")
        _require_text(self.sanitation_policy_version, "sanitation_policy_version")
        _require_text(self.retrieval_policy_version, "retrieval_policy_version")
        sidecar_spaces = tuple(
            sidecar.embedding_space_id for sidecar in self.embedding_sidecars
        )
        _require_unique(sidecar_spaces, "embedding_sidecars")
        if not self.prompt_budget_policy:
            raise ContractValidationError("prompt_budget_policy must not be empty")
        _require_checksum_mapping(self.checksums, "checksums")
        for sidecar in self.embedding_sidecars:
            if self.checksums.get(sidecar.asset_ref) != sidecar.checksum:
                raise MissingReferenceError(
                    f"snapshot checksum closure omits sidecar {sidecar.asset_ref}"
                )
        normalize_utc_timestamp(self.published_at, "published_at")
        if not self.publisher_attestation:
            raise ContractValidationError("publisher_attestation must not be empty")


@dataclass(frozen=True)
class PriorKnowledgeSnapshot(StrictContract):
    prior_knowledge_snapshot_id: str
    source_snapshot_id: str
    query: Mapping[str, Any]
    retrieval_policy_version: str
    task_context_binding_id: str
    selected_records: tuple[Mapping[str, Any], ...]
    selected_record_ids: tuple[str, ...]
    proof_reference_ids: tuple[str, ...]
    selection_metadata: Mapping[str, Mapping[str, Any]]
    prompt_budget_policy: Mapping[str, Any]
    records_digest: str

    CONTENT_NAMESPACE: ClassVar[str] = "prior-knowledge-snapshot"
    IDENTITY_FIELD: ClassVar[str] = "prior_knowledge_snapshot_id"

    def _validate(self) -> None:
        require_content_id(self.source_snapshot_id, "source_snapshot_id")
        require_content_id(self.task_context_binding_id, "task_context_binding_id")
        if not self.query:
            raise ContractValidationError("prior knowledge query must not be empty")
        _require_text(self.retrieval_policy_version, "retrieval_policy_version")
        if len(self.selected_records) != len(self.selected_record_ids):
            raise ContractValidationError(
                "selected records and selected record IDs must align"
            )
        for position, (record, record_id) in enumerate(
            zip(self.selected_records, self.selected_record_ids)
        ):
            if set(record) != {"record_id", "record_kind", "payload"}:
                raise ContractValidationError(
                    f"selected_records[{position}] must be a complete record envelope"
                )
            if record["record_id"] != record_id:
                raise ContractValidationError(
                    f"selected_records[{position}] identity mismatch"
                )
            require_content_id(record["record_id"], "selected record ID")
            require_identifier(record["record_kind"], "selected record kind")
            if not isinstance(record["payload"], MappingABC) or not record["payload"]:
                raise ContractValidationError(
                    f"selected_records[{position}] payload must be complete"
                )
        if self.selected_record_ids:
            _require_sorted_unique(self.selected_record_ids, "selected_record_ids")
            for record_id in self.selected_record_ids:
                require_content_id(record_id, "selected_record_ids")
        if self.proof_reference_ids:
            _require_sorted_unique(self.proof_reference_ids, "proof_reference_ids")
            for proof_id in self.proof_reference_ids:
                require_content_id(proof_id, "proof_reference_ids")
        if set(self.selection_metadata) != set(self.selected_record_ids):
            raise ContractValidationError(
                "selection metadata must be keyed by every selected record exactly"
            )
        expected_selection_fields = {
            "compatibility",
            "evidence_quality",
            "lexical_score",
            "outcome",
            "proof_reference_ids",
            "rank",
            "recency",
            "retrieval_utility",
            "semantic_score",
        }
        ranks: list[int] = []
        for record_id in sorted(self.selection_metadata):
            metadata = self.selection_metadata[record_id]
            if not isinstance(metadata, MappingABC) or set(metadata) != (
                expected_selection_fields
            ):
                raise ContractValidationError("selection metadata fields are invalid")
            if metadata["compatibility"] not in {
                TransferCompatibility.EXACT_CONTEXT.value,
                TransferCompatibility.ANALOGICAL.value,
            }:
                raise ContractValidationError(
                    "selection compatibility must be exact or analogical"
                )
            if metadata["outcome"] not in {
                "positive",
                "negative",
                "inconclusive",
                "frontier",
            }:
                raise ContractValidationError("selection outcome is invalid")
            rank = metadata["rank"]
            evidence_quality = metadata["evidence_quality"]
            score_values = (
                metadata["lexical_score"],
                metadata["retrieval_utility"],
                metadata["semantic_score"],
            )
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ContractValidationError("selection rank is invalid")
            if (
                isinstance(evidence_quality, bool)
                or not isinstance(evidence_quality, int)
                or evidence_quality < 0
            ):
                raise ContractValidationError("selection evidence quality is invalid")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in score_values
            ):
                raise ContractValidationError("selection score is invalid")
            recency = metadata["recency"]
            if not isinstance(recency, str):
                raise ContractValidationError("selection recency must be text")
            if recency:
                normalize_utc_timestamp(recency, "selection recency")
            proof_ids = metadata["proof_reference_ids"]
            if not isinstance(proof_ids, tuple):
                raise ContractValidationError(
                    "selection proof references must be a tuple"
                )
            if proof_ids:
                _require_sorted_unique(proof_ids, "selection proof_reference_ids")
                for proof_id in proof_ids:
                    require_content_id(proof_id, "selection proof_reference_ids")
            if not set(proof_ids).issubset(self.proof_reference_ids):
                raise MissingReferenceError(
                    "selection proof references leave the packet proof closure"
                )
            ranks.append(rank)
        if tuple(sorted(ranks)) != tuple(range(len(self.selected_record_ids))):
            raise ContractValidationError("selection ranks must be unique and gap-free")
        if not self.prompt_budget_policy:
            raise ContractValidationError("prompt_budget_policy must not be empty")
        expected_digest = tree_or_blob_digest(
            canonical_json_bytes(self.selected_records)
        )
        if self.records_digest != expected_digest:
            raise ContractValidationError("selected-record digest mismatch")


@dataclass(frozen=True)
class SourceFileDescriptor(StrictContract):
    """One exact regular file in a content-addressed source tree."""

    relative_path: str
    digest: str
    mode: str
    size: int

    def _validate(self) -> None:
        _require_relative_path(self.relative_path, "source file path")
        if self.relative_path == ".":
            raise ContractValidationError("source file path cannot be the root")
        _require_digest(self.digest, "source file digest")
        if self.mode not in {"100644", "100755"}:
            raise ContractValidationError("source file mode is invalid")
        if type(self.size) is not int or self.size < 0:
            raise ContractValidationError("source file size is invalid")


@dataclass(frozen=True)
class ExpertSourceTreeManifest(StrictContract):
    source_tree_manifest_id: str
    tree_hash: str
    files: tuple[SourceFileDescriptor, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-tree"
    IDENTITY_FIELD: ClassVar[str] = "source_tree_manifest_id"

    def _validate(self) -> None:
        paths = tuple(file.relative_path for file in self.files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ContractValidationError(
                "expert source files must be non-empty, sorted, and unique"
            )
        source_paths = tuple(PurePosixPath(path) for path in paths)
        if any(
            source_path in other_path.parents
            for position, source_path in enumerate(source_paths)
            for other_path in source_paths[position + 1 :]
        ):
            raise ContractValidationError(
                "expert source files contain a file/directory collision"
            )
        expected_tree_hash = source_tree_digest(
            {
                file.relative_path: (file.digest, file.mode, file.size)
                for file in self.files
            }
        )
        if self.tree_hash != expected_tree_hash:
            raise ContractValidationError(
                "expert source tree hash differs from its file descriptor"
            )


@dataclass(frozen=True)
class ExpertModuleContract(StrictContract):
    module_contract_id: str
    module_id: str
    version: str
    purpose: str
    problem_signals: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    preconditions: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    dependency_capability_ids: tuple[str, ...]
    incompatible_capability_ids: tuple[str, ...]
    resource_bounds: Mapping[str, Any]
    dependency_license_manifest: Mapping[str, Any]
    supporting_episode_ids: tuple[str, ...]
    known_failure_episode_ids: tuple[str, ...]
    entrypoint_refs: tuple[str, ...]
    test_refs: tuple[str, ...]
    replay_refs: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-module-contract"
    IDENTITY_FIELD: ClassVar[str] = "module_contract_id"

    def _validate(self) -> None:
        require_identifier(self.module_id, "module_id")
        require_identifier(self.version, "version")
        _require_text(self.purpose, "purpose")
        required_text = {"problem_signals", "outputs"}
        for name in (
            "problem_signals",
            "inputs",
            "outputs",
            "preconditions",
            "incompatibilities",
        ):
            values = getattr(self, name)
            _require_text_tuple(values, name, required=name in required_text)
            if values != tuple(sorted(set(values))):
                raise ContractValidationError(f"{name} must be sorted and unique")
        for name in (
            "dependency_capability_ids",
            "incompatible_capability_ids",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ContractValidationError(f"{name} must be sorted and unique")
            for value in values:
                require_identifier(value, name)
            if self.module_id in values:
                raise ContractValidationError(f"{name} cannot reference the module")
        if set(self.dependency_capability_ids) & set(self.incompatible_capability_ids):
            raise ContractValidationError(
                "a capability cannot be both required and incompatible"
            )
        if not self.resource_bounds or not self.dependency_license_manifest:
            raise ContractValidationError(
                "resource and dependency/license manifests must not be empty"
            )
        for name in ("supporting_episode_ids", "known_failure_episode_ids"):
            values = getattr(self, name)
            if values:
                _require_sorted_unique(values, name)
                for value in values:
                    require_content_id(value, name)
        if set(self.supporting_episode_ids) & set(self.known_failure_episode_ids):
            raise ContractValidationError(
                "supporting and failure evidence must be disjoint"
            )
        for name in ("entrypoint_refs", "test_refs", "replay_refs"):
            values = getattr(self, name)
            if name != "replay_refs" and not values:
                raise ContractValidationError(f"{name} must not be empty")
            if values != tuple(sorted(set(values))):
                raise ContractValidationError(f"{name} must be sorted and unique")
            for value in values:
                _require_relative_path(value, name)


@dataclass(frozen=True)
class ExpertCapabilityNode(StrictContract):
    capability_id: str
    module_contract_ref: str
    owned_paths: tuple[str, ...]
    task_family_bindings: tuple[str, ...]

    def _validate(self) -> None:
        require_identifier(self.capability_id, "capability_id")
        require_content_id(self.module_contract_ref, "module_contract_ref")
        if not self.owned_paths:
            raise ContractValidationError("owned_paths must not be empty")
        if self.owned_paths != tuple(sorted(set(self.owned_paths))):
            raise ContractValidationError("owned_paths must be sorted and unique")
        for path in self.owned_paths:
            _require_relative_path(path, "owned_paths")
            if path == ".":
                raise ContractValidationError("a capability cannot own the root")
        if self.task_family_bindings != tuple(sorted(set(self.task_family_bindings))):
            raise ContractValidationError(
                "task_family_bindings must be sorted and unique"
            )
        for task_family_id in self.task_family_bindings:
            require_identifier(task_family_id, "task_family_bindings")


@dataclass(frozen=True)
class ExpertDependencyEdge(StrictContract):
    source_capability_id: str
    target_capability_id: str

    def _validate(self) -> None:
        require_identifier(self.source_capability_id, "source_capability_id")
        require_identifier(self.target_capability_id, "target_capability_id")
        if self.source_capability_id == self.target_capability_id:
            raise ContractValidationError("capability cannot depend on itself")


@dataclass(frozen=True)
class ExpertTaskAdapterBoundary(StrictContract):
    adapter_mount_path: str
    interface_entrypoint_refs: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    invariants: tuple[str, ...]

    def _validate(self) -> None:
        _require_relative_path(self.adapter_mount_path, "adapter_mount_path")
        if self.adapter_mount_path == ".":
            raise ContractValidationError("adapter mount path cannot be the root")
        for name in (
            "interface_entrypoint_refs",
            "inputs",
            "outputs",
            "invariants",
        ):
            values = getattr(self, name)
            if not values:
                raise ContractValidationError(f"{name} must not be empty")
            if values != tuple(sorted(set(values))):
                raise ContractValidationError(f"{name} must be sorted and unique")
            if name == "interface_entrypoint_refs":
                for value in values:
                    _require_relative_path(value, name)
            else:
                _require_text_tuple(values, name, required=True)


@dataclass(frozen=True)
class ExpertRepositoryMap(StrictContract):
    repository_map_id: str
    scope_contract_id: str
    capability_nodes: tuple[ExpertCapabilityNode, ...]
    dependency_edges: tuple[ExpertDependencyEdge, ...]
    task_adapter_boundary: ExpertTaskAdapterBoundary
    validation_entrypoints: tuple[str, ...]
    architecture_invariants: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-repository-map"
    IDENTITY_FIELD: ClassVar[str] = "repository_map_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        if not self.capability_nodes:
            raise ContractValidationError("repository map needs a capability")
        capability_ids = tuple(node.capability_id for node in self.capability_nodes)
        if capability_ids != tuple(sorted(set(capability_ids))):
            raise ContractValidationError("capability nodes must be sorted and unique")
        owned_paths = tuple(
            PurePosixPath(path)
            for node in self.capability_nodes
            for path in node.owned_paths
        )
        for position, owned_path in enumerate(owned_paths):
            for other_path in owned_paths[position + 1 :]:
                if (
                    owned_path == other_path
                    or owned_path in other_path.parents
                    or other_path in owned_path.parents
                ):
                    raise IdentityConflictError(
                        "capability owned paths must be prefix-disjoint"
                    )
        edge_pairs = tuple(
            (edge.source_capability_id, edge.target_capability_id)
            for edge in self.dependency_edges
        )
        if edge_pairs != tuple(sorted(set(edge_pairs))):
            raise ContractValidationError("dependency edges must be sorted and unique")
        known = set(capability_ids)
        graph: dict[str, set[str]] = {capability_id: set() for capability_id in known}
        for source_capability_id, target_capability_id in edge_pairs:
            if source_capability_id not in known or target_capability_id not in known:
                raise MissingReferenceError(
                    "dependency edge references unknown capability"
                )
            graph[source_capability_id].add(target_capability_id)
        visited: set[str] = set()
        active: set[str] = set()

        def visit(capability_id: str) -> None:
            if capability_id in active:
                raise ContractValidationError("capability dependency graph has a cycle")
            if capability_id in visited:
                return
            active.add(capability_id)
            for dependency_id in graph[capability_id]:
                visit(dependency_id)
            active.remove(capability_id)
            visited.add(capability_id)

        for capability_id in sorted(known):
            visit(capability_id)
        for name in ("validation_entrypoints", "architecture_invariants"):
            values = getattr(self, name)
            if not values or values != tuple(sorted(set(values))):
                raise ContractValidationError(
                    f"{name} must be non-empty, sorted, and unique"
                )
            if name == "validation_entrypoints":
                for path in values:
                    _require_relative_path(path, name)
            else:
                _require_text_tuple(values, name, required=True)


@dataclass(frozen=True)
class ExpertCapabilityLineage(StrictContract):
    source_capability_ids: tuple[str, ...]
    target_capability_ids: tuple[str, ...]
    relation: ExpertCapabilityLineageRelation
    evidence_ids: tuple[str, ...]

    def _validate(self) -> None:
        for name in ("source_capability_ids", "target_capability_ids"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ContractValidationError(f"{name} must be sorted and unique")
            for value in values:
                require_identifier(value, name)
        if not self.source_capability_ids:
            raise ContractValidationError("capability lineage needs a source")
        if set(self.source_capability_ids) & set(self.target_capability_ids):
            raise ContractValidationError(
                "capability lineage source and target overlap"
            )
        cardinality = {
            ExpertCapabilityLineageRelation.RENAME: (
                len(self.source_capability_ids) == 1
                and len(self.target_capability_ids) == 1
            ),
            ExpertCapabilityLineageRelation.SPLIT: (
                len(self.source_capability_ids) == 1
                and len(self.target_capability_ids) > 1
            ),
            ExpertCapabilityLineageRelation.MERGE: (
                len(self.source_capability_ids) > 1
                and len(self.target_capability_ids) == 1
            ),
            ExpertCapabilityLineageRelation.RETIRE: (
                len(self.source_capability_ids) == 1 and not self.target_capability_ids
            ),
        }
        if not cardinality[self.relation]:
            raise ContractValidationError(
                f"invalid {self.relation.value} capability lineage cardinality"
            )
        _require_sorted_unique(self.evidence_ids, "capability lineage evidence_ids")
        for evidence_id in self.evidence_ids:
            require_content_id(evidence_id, "capability lineage evidence_ids")


@dataclass(frozen=True)
class ExpertCandidatePatchChange(StrictContract):
    relative_path: str
    before: SourceFileDescriptor | None
    after: SourceFileDescriptor | None

    def _validate(self) -> None:
        _require_relative_path(self.relative_path, "candidate patch path")
        if self.relative_path == "." or (self.before is None and self.after is None):
            raise ContractValidationError("candidate patch change is empty")
        for descriptor in (self.before, self.after):
            if (
                descriptor is not None
                and descriptor.relative_path != self.relative_path
            ):
                raise ContractValidationError(
                    "candidate patch descriptor uses another path"
                )
        if self.before == self.after:
            raise ContractValidationError("candidate patch change has no effect")


@dataclass(frozen=True)
class ExpertCandidatePatch(StrictContract):
    patch_id: str
    parent_tree_hash: str
    candidate_tree_hash: str
    changes: tuple[ExpertCandidatePatchChange, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-patch"
    IDENTITY_FIELD: ClassVar[str] = "patch_id"

    def _validate(self) -> None:
        _require_digest(self.parent_tree_hash, "candidate patch parent_tree_hash")
        _require_digest(self.candidate_tree_hash, "candidate patch candidate_tree_hash")
        if self.parent_tree_hash == self.candidate_tree_hash:
            raise ContractValidationError("candidate patch must change the tree")
        paths = tuple(change.relative_path for change in self.changes)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ContractValidationError(
                "candidate patch changes must be non-empty, sorted, and unique"
            )


@dataclass(frozen=True)
class ExpertCandidateWorkspaceReceipt(StrictContract):
    """Kapso observation of one coding-agent workspace transformation."""

    workspace_receipt_id: str
    operation_receipt_id: str
    operation_id: str
    parent_tree_hash: str
    editable_parent_tree_hash: str
    edited_tree_hash: str
    changed_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-workspace"
    IDENTITY_FIELD: ClassVar[str] = "workspace_receipt_id"

    def _validate(self) -> None:
        require_content_id(self.operation_receipt_id, "operation_receipt_id")
        if _CODING_AGENT_OPERATION_PATTERN.fullmatch(self.operation_id) is None:
            raise ContractValidationError("invalid workspace operation ID")
        for value, name in (
            (self.parent_tree_hash, "workspace parent_tree_hash"),
            (self.editable_parent_tree_hash, "workspace editable_parent_tree_hash"),
            (self.edited_tree_hash, "workspace edited_tree_hash"),
        ):
            _require_digest(value, name)
        for name in ("changed_paths", "deleted_paths"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ContractValidationError(
                    f"workspace {name} must be sorted and unique"
                )
            for path in values:
                _require_relative_path(path, f"workspace {name}")
                if path == ".":
                    raise ContractValidationError(
                        f"workspace {name} cannot name the root"
                    )
        if not self.changed_paths and not self.deleted_paths:
            raise ContractValidationError("workspace receipt observes no source change")
        if set(self.changed_paths) & set(self.deleted_paths):
            raise ContractValidationError(
                "workspace changed and deleted paths must be disjoint"
            )


@dataclass(frozen=True)
class ExpertCandidateOperationRecord(StrictContract):
    operation_record_id: str
    operation_kind: ExpertCandidateOperationKind
    trigger_decision_id: str
    trigger_evidence_packet_id: str
    parent_tree_hash: str
    ancestor_candidate_ids: tuple[str, ...]
    configuration_fingerprint: str
    operation_preimage: Mapping[str, Any]
    operation_receipt: CodingAgentOperationReceipt
    workspace_receipt: ExpertCandidateWorkspaceReceipt
    final_output: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-operation"
    IDENTITY_FIELD: ClassVar[str] = "operation_record_id"

    def _validate(self) -> None:
        for value, name in (
            (self.trigger_decision_id, "trigger_decision_id"),
            (self.trigger_evidence_packet_id, "trigger_evidence_packet_id"),
        ):
            require_content_id(value, name)
        _require_digest(self.parent_tree_hash, "operation parent_tree_hash")
        _require_digest(
            self.configuration_fingerprint,
            "operation configuration_fingerprint",
        )
        if self.ancestor_candidate_ids:
            _require_sorted_unique(
                self.ancestor_candidate_ids,
                "operation ancestor_candidate_ids",
            )
            for candidate_id in self.ancestor_candidate_ids:
                require_content_id(candidate_id, "operation ancestor_candidate_ids")
        if not self.operation_preimage:
            raise ContractValidationError("candidate operation preimage is empty")
        expected_preimage_binding = {
            "ancestor_candidate_ids": self.ancestor_candidate_ids,
            "configuration_fingerprint": self.configuration_fingerprint,
            "operation_kind": self.operation_kind.value,
            "parent_tree_hash": self.parent_tree_hash,
            "trigger_decision_id": self.trigger_decision_id,
            "trigger_evidence_packet_id": self.trigger_evidence_packet_id,
        }
        observed_preimage_binding = {
            key: self.operation_preimage.get(key) for key in expected_preimage_binding
        }
        if canonical_json_bytes(observed_preimage_binding) != canonical_json_bytes(
            expected_preimage_binding
        ):
            raise ContractValidationError(
                "candidate operation preimage differs from its declared binding"
            )
        expected_operation_id = (
            "agent_call_"
            + tree_or_blob_digest(canonical_json_bytes(self.operation_preimage))[7:39]
        )
        if self.operation_receipt.operation_id != expected_operation_id:
            raise ContractValidationError(
                "candidate operation preimage differs from its receipt"
            )
        if (
            self.workspace_receipt.operation_receipt_id
            != self.operation_receipt.operation_receipt_id
            or self.workspace_receipt.operation_id
            != self.operation_receipt.operation_id
            or self.workspace_receipt.parent_tree_hash != self.parent_tree_hash
        ):
            raise ContractValidationError(
                "candidate workspace receipt differs from its operation"
            )
        if not isinstance(self.final_output, str) or not self.final_output.strip():
            raise ContractValidationError("candidate operation final output is empty")
        observed_final = parse_json_bytes(self.final_output.encode("utf-8"))
        if tree_or_blob_digest(self.final_output.encode("utf-8")) != (
            self.operation_receipt.artifact_checksums["final.json"]
        ):
            raise ContractValidationError(
                "candidate operation final output differs from its receipt"
            )
        if not isinstance(observed_final, MappingABC):
            raise ContractValidationError(
                "candidate operation final output must be an object"
            )
        expected_declarations = {
            "changed_paths": self.workspace_receipt.changed_paths,
            "deleted_paths": self.workspace_receipt.deleted_paths,
        }
        observed_declarations = {
            key: observed_final.get(key) for key in expected_declarations
        }
        if canonical_json_bytes(observed_declarations) != canonical_json_bytes(
            expected_declarations
        ):
            raise ContractValidationError(
                "candidate final output differs from the observed workspace delta"
            )


@dataclass(frozen=True)
class ExpertCandidateSanitationFinding(StrictContract):
    code: str
    relative_path: str
    evidence_digest: str
    severity: ExpertSanitationSeverity

    def _validate(self) -> None:
        require_identifier(self.code, "candidate sanitation finding code")
        _require_relative_path(
            self.relative_path,
            "candidate sanitation finding path",
        )
        _require_digest(
            self.evidence_digest,
            "candidate sanitation finding evidence_digest",
        )


@dataclass(frozen=True)
class ExpertCandidateSanitationReport(StrictContract):
    sanitation_report_id: str
    scope_contract_id: str
    candidate_tree_hash: str
    policy_version: str
    policy_fingerprint: str
    scanner_version: str
    status: ExpertCandidateSanitationStatus
    scanned_files: tuple[SourceFileDescriptor, ...]
    findings: tuple[ExpertCandidateSanitationFinding, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-sanitation"
    IDENTITY_FIELD: ClassVar[str] = "sanitation_report_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "sanitation scope_contract_id")
        for value, name in (
            (self.policy_version, "candidate sanitation policy_version"),
            (self.scanner_version, "candidate sanitation scanner_version"),
        ):
            require_identifier(value, name)
        _require_digest(
            self.policy_fingerprint,
            "candidate sanitation policy_fingerprint",
        )
        paths = tuple(file.relative_path for file in self.scanned_files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ContractValidationError(
                "candidate sanitation files must be non-empty, sorted, and unique"
            )
        if self.candidate_tree_hash != source_tree_digest(
            {
                file.relative_path: (file.digest, file.mode, file.size)
                for file in self.scanned_files
            }
        ):
            raise ContractValidationError(
                "candidate sanitation report scans another tree"
            )
        finding_keys = tuple(
            (finding.relative_path, finding.code, finding.evidence_digest)
            for finding in self.findings
        )
        if finding_keys != tuple(sorted(set(finding_keys))):
            raise ContractValidationError(
                "candidate sanitation findings must be sorted and unique"
            )
        if any(finding.relative_path not in paths for finding in self.findings):
            raise ContractValidationError(
                "candidate sanitation finding references an unscanned file"
            )
        has_blocking_finding = any(
            finding.severity is ExpertSanitationSeverity.BLOCKING
            for finding in self.findings
        )
        if has_blocking_finding != (
            self.status is ExpertCandidateSanitationStatus.REJECTED
        ):
            raise ContractValidationError(
                "candidate sanitation status differs from blocking findings"
            )


@dataclass(frozen=True)
class ExpertCandidateManifest(StrictContract):
    candidate_id: str
    scope_contract_id: str
    change_kind: CandidateChangeKind
    parent_release_id: str | None
    parent_repository_map_ref: str | None
    parent_tree_hash: str
    trigger_decision_id: str
    trigger_evidence_packet_id: str
    patch_ref: str
    patch_digest: str
    candidate_tree_ref: str
    candidate_tree_hash: str
    configuration_fingerprint: str
    module_contract_refs: tuple[str, ...]
    proposed_repository_map_ref: str
    semantic_book_digest: str
    proposer_operation_record_id: str
    source_dependency_ids: tuple[str, ...]
    ancestor_candidate_ids: tuple[str, ...]
    capability_lineage: tuple[ExpertCapabilityLineage, ...]
    sanitation_report_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate"
    IDENTITY_FIELD: ClassVar[str] = "candidate_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        if (self.parent_release_id is None) != (self.parent_repository_map_ref is None):
            raise ContractValidationError(
                "candidate parent release and repository map must appear together"
            )
        if self.parent_release_id is None:
            if (
                self.change_kind is not CandidateChangeKind.REPOSITORY_ARCHITECTURE
                or self.parent_tree_hash != EMPTY_EXPERT_TREE_DIGEST
            ):
                raise ContractValidationError(
                    "parentless candidate must bootstrap the canonical empty tree"
                )
        elif self.parent_tree_hash == EMPTY_EXPERT_TREE_DIGEST:
            raise ContractValidationError(
                "released parent candidate cannot use the canonical empty tree"
            )
        for value, name in (
            (self.parent_release_id, "parent_release_id"),
            (self.parent_repository_map_ref, "parent_repository_map_ref"),
        ):
            if value is not None:
                require_content_id(value, name)
        _require_digest(self.parent_tree_hash, "parent_tree_hash")
        for value, name in (
            (self.trigger_decision_id, "trigger_decision_id"),
            (self.trigger_evidence_packet_id, "trigger_evidence_packet_id"),
            (self.patch_ref, "patch_ref"),
            (self.candidate_tree_ref, "candidate_tree_ref"),
            (self.proposed_repository_map_ref, "proposed_repository_map_ref"),
            (self.proposer_operation_record_id, "proposer_operation_record_id"),
            (self.sanitation_report_id, "sanitation_report_id"),
        ):
            require_content_id(value, name)
        for value, name in (
            (self.patch_digest, "patch_digest"),
            (self.candidate_tree_hash, "candidate_tree_hash"),
            (self.configuration_fingerprint, "configuration_fingerprint"),
            (self.semantic_book_digest, "semantic_book_digest"),
        ):
            _require_digest(value, name)
        _require_sorted_unique(self.module_contract_refs, "module_contract_refs")
        for value in self.module_contract_refs:
            require_content_id(value, "module_contract_refs")
        _require_sorted_unique(self.source_dependency_ids, "source_dependency_ids")
        for value in self.source_dependency_ids:
            require_content_id(value, "source_dependency_ids")
        if self.ancestor_candidate_ids:
            _require_sorted_unique(
                self.ancestor_candidate_ids,
                "ancestor_candidate_ids",
            )
            for value in self.ancestor_candidate_ids:
                require_content_id(value, "ancestor_candidate_ids")
        lineage_keys = tuple(
            (
                lineage.relation.value,
                lineage.source_capability_ids,
                lineage.target_capability_ids,
            )
            for lineage in self.capability_lineage
        )
        if lineage_keys != tuple(sorted(set(lineage_keys))):
            raise ContractValidationError(
                "capability lineage must be sorted and unique"
            )


@dataclass(frozen=True)
class ExpertBaseReleaseManifest(StrictContract):
    release_id: str
    scope_contract_id: str
    scope_id: str
    parent_release_ids: tuple[str, ...]
    repository_map_ref: str
    module_versions: Mapping[str, str]
    semantic_book_digest: str
    configuration_fingerprint: str
    source_archive_ref: str
    dependency_closure_ids: tuple[str, ...]
    checksums: Mapping[str, str]
    test_matrix_results: Mapping[str, Any]
    approval_assertion_ids: tuple[str, ...]
    contamination_scanner_version: str
    dependency_lock_hash: str
    compatibility_envelope: Mapping[str, Any]
    publisher_attestation: Mapping[str, Any]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-base-release"
    IDENTITY_FIELD: ClassVar[str] = "release_id"
    CONTENT_EXCLUDED_FIELDS: ClassVar[tuple[str, ...]] = ("publisher_attestation",)

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        require_identifier(self.scope_id, "scope_id")
        if self.parent_release_ids:
            _require_sorted_unique(self.parent_release_ids, "parent_release_ids")
            for value in self.parent_release_ids:
                require_content_id(value, "parent_release_ids")
        require_content_id(self.repository_map_ref, "repository_map_ref")
        if not self.module_versions:
            raise ContractValidationError("module_versions must not be empty")
        for module_id, version in self.module_versions.items():
            require_identifier(module_id, "module_versions key")
            require_identifier(version, "module_versions value")
        _require_digest(self.semantic_book_digest, "semantic_book_digest")
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        _require_text(self.source_archive_ref, "source_archive_ref")
        source_archive_path = PurePosixPath(self.source_archive_ref)
        if (
            source_archive_path.is_absolute()
            or len(source_archive_path.parts) != 1
            or source_archive_path.as_posix() != self.source_archive_ref
            or not self.source_archive_ref.endswith((".tar", ".tar.zst"))
        ):
            raise ContractValidationError(
                "source_archive_ref must name one supported release asset"
            )
        _require_sorted_unique(self.dependency_closure_ids, "dependency_closure_ids")
        for value in self.dependency_closure_ids:
            require_content_id(value, "dependency_closure_ids")
        required_dependencies = {
            self.repository_map_ref,
            *self.approval_assertion_ids,
        }
        if not required_dependencies.issubset(self.dependency_closure_ids):
            raise MissingReferenceError(
                "expert release dependency closure is incomplete"
            )
        _require_checksum_mapping(self.checksums, "checksums")
        if self.source_archive_ref not in self.checksums:
            raise MissingReferenceError("expert source archive checksum is missing")
        if not self.test_matrix_results:
            raise ContractValidationError("test_matrix_results must not be empty")
        _require_sorted_unique(self.approval_assertion_ids, "approval_assertion_ids")
        for value in self.approval_assertion_ids:
            require_content_id(value, "approval_assertion_ids")
        _require_text(
            self.contamination_scanner_version, "contamination_scanner_version"
        )
        _require_digest(self.dependency_lock_hash, "dependency_lock_hash")
        if not self.compatibility_envelope or not self.publisher_attestation:
            raise ContractValidationError(
                "compatibility envelope and publisher attestation are required"
            )


@dataclass(frozen=True)
class TaskAdapterManifest(StrictContract):
    task_adapter_manifest_id: str
    task_adapter_id: str
    scope_contract_id: str
    task_family_id: str
    publisher_attestation: Mapping[str, Any]
    task_evaluator_binding: Mapping[str, Any]
    context_dimension_binding: Mapping[str, Any]
    source_tree_ref: str
    tree_hash: str
    dependency_runtime_contract: Mapping[str, Any]
    sanitation_report_id: str
    validation_refs: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-adapter-manifest"
    IDENTITY_FIELD: ClassVar[str] = "task_adapter_manifest_id"
    CONTENT_EXCLUDED_FIELDS: ClassVar[tuple[str, ...]] = ("publisher_attestation",)

    def _validate(self) -> None:
        require_identifier(self.task_adapter_id, "task_adapter_id")
        require_content_id(self.scope_contract_id, "scope_contract_id")
        require_identifier(self.task_family_id, "task_family_id")
        if not self.publisher_attestation:
            raise ContractValidationError("publisher_attestation must not be empty")
        if not self.task_evaluator_binding or not self.context_dimension_binding:
            raise ContractValidationError(
                "task/evaluator and context-dimension bindings are required"
            )
        _require_text(self.source_tree_ref, "source_tree_ref")
        _require_digest(self.tree_hash, "tree_hash")
        if not self.dependency_runtime_contract:
            raise ContractValidationError(
                "dependency_runtime_contract must not be empty"
            )
        require_content_id(self.sanitation_report_id, "sanitation_report_id")
        _require_sorted_unique(self.validation_refs, "validation_refs")


@dataclass(frozen=True)
class GitHubReleaseAsset(StrictContract):
    asset_id: str
    name: str
    media_type: str
    size: int
    sha256: str

    def _validate(self) -> None:
        require_identifier(self.asset_id, "asset_id")
        _require_relative_path(self.name, "name")
        _require_text(self.media_type, "media_type")
        if self.size < 0:
            raise ContractValidationError("asset size must be non-negative")
        _require_digest(self.sha256, "sha256")


@dataclass(frozen=True)
class GitHubPublicationRecord(StrictContract):
    publication_id: str
    artifact_kind: PublicationArtifactKind
    artifact_id: str
    repository_node_id: str
    repository_full_name: str
    commit_sha: str
    immutable_release_id: str
    tag: str
    assets: tuple[GitHubReleaseAsset, ...]
    release_attestation_ref: str
    published_at: str
    publisher_identity: str

    CONTENT_NAMESPACE: ClassVar[str] = "github-publication"
    IDENTITY_FIELD: ClassVar[str] = "publication_id"

    def _validate(self) -> None:
        require_content_id(self.artifact_id, "artifact_id")
        require_identifier(self.repository_node_id, "repository_node_id")
        _require_repository_coordinate(
            self.repository_full_name, "repository_full_name"
        )
        if not re.fullmatch(r"[0-9a-f]{40}", self.commit_sha):
            raise ContractValidationError("commit_sha must be 40 lowercase hex")
        require_identifier(self.immutable_release_id, "immutable_release_id")
        _require_text(self.tag, "tag")
        if not self.assets:
            raise ContractValidationError("publication must contain release assets")
        asset_names = tuple(asset.name for asset in self.assets)
        _require_unique(asset_names, "assets")
        _require_text(self.release_attestation_ref, "release_attestation_ref")
        normalize_utc_timestamp(self.published_at, "published_at")
        require_identifier(self.publisher_identity, "publisher_identity")


@dataclass(frozen=True)
class LaunchManifest(StrictContract):
    launch_manifest_id: str
    launch_request_hash: str
    scope_id: str
    scope_contract_id: str
    scope_repository_binding_hash: str
    configuration_fingerprint: str
    task_family_id: str
    task_adapter_id: str
    knowledge_snapshot_id: str
    knowledge_publication_ref: str
    expert_base_release_id: str
    expert_publication_ref: str
    embedding_space_id: str
    dependency_runtime_contract: Mapping[str, Any]
    sanitation_policy_generation: int
    security_denylist_generation: int
    expected_source_composition_hash: str
    publisher_attestation: Mapping[str, Any]

    CONTENT_NAMESPACE: ClassVar[str] = "launch-manifest"
    IDENTITY_FIELD: ClassVar[str] = "launch_manifest_id"
    CONTENT_EXCLUDED_FIELDS: ClassVar[tuple[str, ...]] = ("publisher_attestation",)

    def _validate(self) -> None:
        _require_digest(self.launch_request_hash, "launch_request_hash")
        for name in ("scope_id", "task_family_id", "task_adapter_id"):
            require_identifier(getattr(self, name), name)
        for name in (
            "scope_contract_id",
            "knowledge_snapshot_id",
            "knowledge_publication_ref",
            "expert_base_release_id",
            "expert_publication_ref",
            "embedding_space_id",
        ):
            require_content_id(getattr(self, name), name)
        _require_digest(
            self.scope_repository_binding_hash, "scope_repository_binding_hash"
        )
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        if not self.dependency_runtime_contract:
            raise ContractValidationError(
                "dependency_runtime_contract must not be empty"
            )
        if (
            self.sanitation_policy_generation < 0
            or self.security_denylist_generation < 0
        ):
            raise ContractValidationError("policy generations must be non-negative")
        _require_digest(
            self.expected_source_composition_hash,
            "expected_source_composition_hash",
        )
        if not self.publisher_attestation:
            raise ContractValidationError("publisher_attestation must not be empty")


@dataclass(frozen=True)
class BootstrapPin(StrictContract):
    bootstrap_pin_id: str
    launch_manifest_id: str
    launch_request_hash: str
    scope_id: str
    scope_contract_id: str
    task_family_id: str
    task_adapter_id: str
    knowledge_snapshot_id: str
    expert_base_release_id: str
    task_adapter_manifest_id: str
    workspace_tree_hash: str
    created_at: str

    CONTENT_NAMESPACE: ClassVar[str] = "bootstrap-pin"
    IDENTITY_FIELD: ClassVar[str] = "bootstrap_pin_id"

    def _validate(self) -> None:
        for name in (
            "launch_manifest_id",
            "scope_contract_id",
            "knowledge_snapshot_id",
            "expert_base_release_id",
            "task_adapter_manifest_id",
        ):
            require_content_id(getattr(self, name), name)
        _require_digest(self.launch_request_hash, "launch_request_hash")
        for name in ("scope_id", "task_family_id", "task_adapter_id"):
            require_identifier(getattr(self, name), name)
        _require_digest(self.workspace_tree_hash, "workspace_tree_hash")
        normalize_utc_timestamp(self.created_at, "created_at")
