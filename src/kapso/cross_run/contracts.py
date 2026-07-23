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
_EXPERT_MODULE_VERSION_PATTERN = re.compile(r"^v[1-9][0-9]*$")
_RUNTIME_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_ENVIRONMENT_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:ACCESS_KEY(?:_ID)?|ACCESS_TOKEN|API_KEY|AUTH_CONFIG|AUTH_TOKEN|"
    r"CREDENTIALS?|NETRC|OAUTH_TOKEN|PASSWORD|PASSWD|PAT|PRIVATE_KEY|"
    r"SECRET(?:_ACCESS_KEY)?|SECRETS?|TOKEN)(?:_|$)"
)
EMPTY_EXPERT_TREE_DIGEST = tree_or_blob_digest(canonical_json_bytes(()))
EXPERT_CANDIDATE_COMMIT_PATH = "COMMITTED.json"


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
            contract_types = tuple(
                argument
                for argument in non_null
                if isinstance(argument, type) and issubclass(argument, StrictContract)
            )
            if len(contract_types) != len(non_null):
                raise ContractValidationError(
                    f"{name} has unsupported union annotation"
                )
            matching_instances = tuple(
                contract_type
                for contract_type in contract_types
                if type(value) is contract_type
            )
            if len(matching_instances) == 1:
                return value
            if not isinstance(value, MappingABC):
                raise ContractValidationError(
                    f"{name} must be one recognized contract object"
                )
            matching_contracts = tuple(
                contract_type
                for contract_type in contract_types
                if contract_type.CONTENT_NAMESPACE is not None
                and contract_type.IDENTITY_FIELD is not None
                and isinstance(value.get(contract_type.IDENTITY_FIELD), str)
                and value[contract_type.IDENTITY_FIELD].split(":sha256:", 1)[0]
                == contract_type.CONTENT_NAMESPACE
            )
            if len(matching_contracts) != 1:
                raise ContractValidationError(
                    f"{name} does not identify one supported contract"
                )
            return _decode_value(matching_contracts[0], value, name)
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


class ExpertCandidateDerivationKind(str, Enum):
    AGENT_PROPOSAL = "agent_proposal"
    DETERMINISTIC_COMPOSITION = "deterministic_composition"
    DETERMINISTIC_RECOVERY_RESTORE = "deterministic_recovery_restore"


class ExpertCandidateSanitationStatus(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"


class ExpertValidationTrack(str, Enum):
    MECHANICAL_GENERAL_FIX = "mechanical_general_fix"
    BEHAVIORAL_CAPABILITY = "behavioral_capability"
    REPOSITORY_ARCHITECTURE = "repository_architecture"


class ExpertReviewDisposition(str, Enum):
    CORE_ELIGIBLE = "core_eligible"
    TASK_SPECIFIC = "task_specific"
    CONFOUNDED_OR_NOISY = "confounded_or_noisy"
    UNSAFE_OR_SPECIALIZED = "unsafe_or_specialized"


class ExpertPromotionState(str, Enum):
    INELIGIBLE = "ineligible"
    VALIDATING = "validating"
    FAILED = "failed"
    DISPUTED = "disputed"
    PARETO_RETAINED = "pareto_retained"
    APPROVED = "approved"
    RELEASE_USE_BLOCKED = "release_use_blocked"
    RELEASED = "released"
    REVOKED = "revoked"


class ExpertValidationStage(str, Enum):
    CONTRACT_SCHEMA = "contract_schema"
    IDENTITY_SECRETS_LICENSE_DEPENDENCY = "identity_secrets_license_dependency"
    STATIC_UNIT_SECURITY_RESOURCE = "static_unit_security_resource"
    SYNTHETIC_FRESH_TASK = "synthetic_fresh_task"
    SOURCE_RUN_REPLAY = "source_run_replay"
    DEVELOPMENT_ANCHORS = "development_anchors"
    CROSS_FAMILY_TRANSFER = "cross_family_transfer"
    SEALED_CANARY = "sealed_canary"
    AUTOMATED_REVIEW = "automated_review"
    RELEASE_MATRIX = "release_matrix"
    PUBLICATION_ELIGIBILITY = "publication_eligibility"


class ExpertSourceReplayExecutionLegKind(str, Enum):
    SOURCE_BASE_CONTROL = "source_base_control"
    CANDIDATE = "candidate"


class ExpertEvaluatorOutcome(str, Enum):
    PASSED = "passed"
    CANDIDATE_FAILED = "candidate_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    INCONCLUSIVE = "inconclusive"


class ExpertValidationAuthorityInvalidationKind(str, Enum):
    CURRENT_RELEASE_AUTHORITY_CHANGED = "current_release_authority_changed"


class ExpertSanitationSeverity(str, Enum):
    INFORMATIONAL = "informational"
    WARNING = "warning"
    BLOCKING = "blocking"


class PublicationArtifactKind(str, Enum):
    KNOWLEDGE_SNAPSHOT = "knowledge_snapshot"
    EXPERT_BASE_RELEASE = "expert_base_release"
    SECURITY_DENYLIST = "security_denylist"


class SecurityDenylistKind(str, Enum):
    SECURITY = "security"
    CONTAMINATION = "contamination"


SECURITY_DENYLIST_SCHEMA_VERSION = "kapso.security_denylist.v1"
SECURITY_DENYLIST_POLICY_VERSION = "kapso.security_revocation.v1"
SECURITY_DENYLIST_EVIDENCE_FILENAME = "security-denylist-evidence.json"


@dataclass(frozen=True)
class ScopeRepositorySettings(StrictContract):
    scope_id: str
    expert_repository: str
    knowledge_repository: str
    security_repository: str

    def _validate(self) -> None:
        require_identifier(self.scope_id, "scope_id")
        for name in (
            "expert_repository",
            "knowledge_repository",
            "security_repository",
        ):
            _require_repository_coordinate(getattr(self, name), name)
        repositories = {
            self.expert_repository,
            self.knowledge_repository,
            self.security_repository,
        }
        if len(repositories) != 3:
            raise IdentityConflictError("scope repositories must be distinct")

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
    task_adapter_manifest_id: str
    task_adapter_verification_receipt_id: str
    starting_artifact_content_ids: Mapping[str, str]
    dependency_lock_hash: str

    CONTENT_NAMESPACE: ClassVar[str] = "artifact-environment"
    IDENTITY_FIELD: ClassVar[str] = "artifact_environment_id"

    def _validate(self) -> None:
        _require_text(self.kapso_commit, "kapso_commit")
        require_content_id(self.expert_base_release_id, "expert_base_release_id")
        if self.expert_base_release_id.split(":sha256:", 1)[0] != (
            "expert-base-release"
        ):
            raise ContractValidationError(
                "expert_base_release_id must name an expert release"
            )
        require_content_id(
            self.task_adapter_manifest_id,
            "task_adapter_manifest_id",
        )
        if self.task_adapter_manifest_id.split(":sha256:", 1)[0] != (
            "task-adapter-manifest"
        ):
            raise ContractValidationError(
                "task_adapter_manifest_id must name a TaskAdapterManifest"
            )
        require_content_id(
            self.task_adapter_verification_receipt_id,
            "task_adapter_verification_receipt_id",
        )
        if self.task_adapter_verification_receipt_id.split(":sha256:", 1)[0] != (
            "task-adapter-verification-receipt"
        ):
            raise ContractValidationError(
                "task_adapter_verification_receipt_id must name a "
                "TaskAdapterVerificationReceipt"
            )
        for (
            artifact_ref,
            artifact_content_id,
        ) in self.starting_artifact_content_ids.items():
            _require_text(artifact_ref, "starting_artifact_content_ids key")
            require_content_id(
                artifact_content_id,
                "starting_artifact_content_ids value",
            )
            if artifact_content_id.split(":sha256:", 1)[0] != (
                "source-replay-starting-artifact"
            ):
                raise ContractValidationError(
                    "starting artifact content IDs must name source replay artifacts"
                )
        if len(self.starting_artifact_content_ids) != len(
            set(self.starting_artifact_content_ids.values())
        ):
            raise ContractValidationError(
                "starting artifact refs must name distinct content records"
            )
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
        if set(self.artifact_environment.starting_artifact_content_ids) != set(
            self.task_context_binding.starting_artifact_refs
        ):
            raise IncompatibleArtifactError(
                "bundle environment does not pin every starting artifact"
            )
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
        if set(self.artifact_environment.starting_artifact_content_ids) != set(
            self.task_context_binding.starting_artifact_refs
        ):
            raise IncompatibleArtifactError(
                "episode environment does not pin every starting artifact"
            )
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
    active_expert_release_use_revocation_ids: tuple[str, ...]
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
            "active_expert_release_use_revocation_ids",
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
        required_proof_ids.update(self.active_expert_release_use_revocation_ids)
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
        if (
            not isinstance(self.version, str)
            or _EXPERT_MODULE_VERSION_PATTERN.fullmatch(self.version) is None
        ):
            raise ContractValidationError(
                "expert module version must be a positive v-prefixed integer"
            )
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
    source_base_tree_hash: str
    candidate_tree_hash: str
    changes: tuple[ExpertCandidatePatchChange, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-patch"
    IDENTITY_FIELD: ClassVar[str] = "patch_id"

    def _validate(self) -> None:
        _require_digest(
            self.source_base_tree_hash, "candidate patch source_base_tree_hash"
        )
        _require_digest(self.candidate_tree_hash, "candidate patch candidate_tree_hash")
        if self.source_base_tree_hash == self.candidate_tree_hash:
            raise ContractValidationError("candidate patch must change the tree")
        paths = tuple(change.relative_path for change in self.changes)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ContractValidationError(
                "candidate patch changes must be non-empty, sorted, and unique"
            )


@dataclass(frozen=True)
class ExpertRecoveryRestorePatch(StrictContract):
    """Typed identity transform used only for whole-tree recovery."""

    patch_id: str
    restored_release_id: str
    source_base_tree_hash: str
    candidate_tree_hash: str
    changes: tuple[ExpertCandidatePatchChange, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-recovery-restore-patch"
    IDENTITY_FIELD: ClassVar[str] = "patch_id"

    def _validate(self) -> None:
        require_content_id(
            self.restored_release_id,
            "recovery restore patch release",
        )
        if self.restored_release_id.split(":sha256:", 1)[0] != "expert-base-release":
            raise ContractValidationError(
                "recovery restore patch release uses the wrong namespace"
            )
        _require_digest(
            self.source_base_tree_hash,
            "recovery restore source tree",
        )
        _require_digest(
            self.candidate_tree_hash,
            "recovery restore candidate tree",
        )
        if self.source_base_tree_hash != self.candidate_tree_hash or self.changes:
            raise ContractValidationError(
                "recovery restore patch must be an exact identity transform"
            )


@dataclass(frozen=True)
class ExpertCandidateCommitRecord(StrictContract):
    """Create-only package checksum closure for one expert candidate."""

    commit_record_id: str
    candidate_id: str
    file_checksums: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-commit"
    IDENTITY_FIELD: ClassVar[str] = "commit_record_id"

    def _validate(self) -> None:
        require_content_id(self.candidate_id, "candidate_id")
        if not self.file_checksums:
            raise ContractValidationError("candidate commit has no files")
        for relative_path, digest in self.file_checksums.items():
            path = PurePosixPath(relative_path)
            if (
                not relative_path
                or path.is_absolute()
                or path == PurePosixPath(".")
                or ".." in path.parts
                or path.as_posix() != relative_path
                or relative_path == EXPERT_CANDIDATE_COMMIT_PATH
            ):
                raise ContractValidationError("candidate commit file path is invalid")
            _require_digest(digest, "candidate commit file digest")


@dataclass(frozen=True)
class ExpertCandidateWorkspaceReceipt(StrictContract):
    """Kapso observation of one coding-agent workspace transformation."""

    workspace_receipt_id: str
    operation_receipt_id: str
    operation_id: str
    source_base_tree_hash: str
    editable_input_tree_hash: str
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
            (self.source_base_tree_hash, "workspace source_base_tree_hash"),
            (self.editable_input_tree_hash, "workspace editable_input_tree_hash"),
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
class ExpertProposerAuthority(StrictContract):
    authority_id: str
    principal_id: str
    role: str
    cli: str
    model: str
    effort: str
    timeout_seconds: int
    allowed_tools: tuple[str, ...]
    workspace_access: CodingAgentWorkspaceAccess
    workspace_maximum_entries: int
    workspace_maximum_bytes: int
    sensitive_file_glob_scan_max_depth: int

    CONTENT_NAMESPACE: ClassVar[str] = "expert-proposer-authority"
    IDENTITY_FIELD: ClassVar[str] = "authority_id"

    def _validate(self) -> None:
        for value, name in (
            (self.principal_id, "expert proposer principal_id"),
            (self.role, "expert proposer role"),
            (self.effort, "expert proposer effort"),
        ):
            require_identifier(value, name)
        if self.cli not in {"codex", "claude_code"}:
            raise ContractValidationError("invalid expert proposer CLI")
        _require_text(self.model, "expert proposer model")
        if self.allowed_tools != tuple(sorted(set(self.allowed_tools))):
            raise ContractValidationError(
                "expert proposer tools must be sorted and unique"
            )
        for value, name in (
            (self.timeout_seconds, "expert proposer timeout_seconds"),
            (self.workspace_maximum_entries, "expert proposer maximum entries"),
            (self.workspace_maximum_bytes, "expert proposer maximum bytes"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractValidationError(f"{name} must be positive")
        if (
            not isinstance(self.sensitive_file_glob_scan_max_depth, int)
            or isinstance(self.sensitive_file_glob_scan_max_depth, bool)
            or self.sensitive_file_glob_scan_max_depth < 0
        ):
            raise ContractValidationError(
                "expert proposer sensitive scan depth must be non-negative"
            )
        if self.workspace_access is not CodingAgentWorkspaceAccess.EDIT_WORKSPACE:
            raise ContractValidationError(
                "expert proposer authority requires editable workspace access"
            )


@dataclass(frozen=True)
class ExpertCandidateOperationRecord(StrictContract):
    operation_record_id: str
    operation_kind: ExpertCandidateOperationKind
    trigger_decision_id: str
    trigger_evidence_packet_id: str
    source_base_tree_hash: str
    ancestor_candidate_ids: tuple[str, ...]
    configuration_fingerprint: str
    proposer_authority: ExpertProposerAuthority
    operation_preimage: Mapping[str, Any]
    operation_receipt: CodingAgentOperationReceipt
    workspace_receipt: ExpertCandidateWorkspaceReceipt
    workspace_delta_ref: str
    workspace_delta_digest: str
    final_output: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-operation"
    IDENTITY_FIELD: ClassVar[str] = "operation_record_id"

    def _validate(self) -> None:
        for value, name in (
            (self.trigger_decision_id, "trigger_decision_id"),
            (self.trigger_evidence_packet_id, "trigger_evidence_packet_id"),
        ):
            require_content_id(value, name)
        _require_digest(self.source_base_tree_hash, "operation source_base_tree_hash")
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
        expected_preimage_fields = {
            "ancestor_candidate_ids",
            "configuration_fingerprint",
            "input_artifact_checksums",
            "mcp_configuration_fingerprint",
            "operation_kind",
            "source_base_tree_hash",
            "principal_id",
            "proposer_authority_id",
            "proposal_contract_version",
            "proposal_packet_digest",
            "trigger_decision_id",
            "trigger_evidence_packet_id",
        }
        if set(self.operation_preimage) != expected_preimage_fields:
            raise ContractValidationError(
                "candidate operation preimage fields are invalid"
            )
        input_checksums = self.operation_preimage.get("input_artifact_checksums")
        if not isinstance(input_checksums, MappingABC) or set(input_checksums) != {
            "invocation.json",
            "prior_knowledge.json",
            "prompt.txt",
            "response_schema.json",
        }:
            raise ContractValidationError(
                "candidate operation preimage input checksums are invalid"
            )
        _require_checksum_mapping(
            input_checksums,
            "candidate operation input_artifact_checksums",
        )
        _require_digest(
            self.operation_preimage.get("mcp_configuration_fingerprint"),
            "candidate operation MCP configuration fingerprint",
        )
        require_identifier(
            self.operation_preimage.get("proposal_contract_version"),
            "candidate proposal contract version",
        )
        require_identifier(
            self.operation_preimage.get("principal_id"),
            "candidate proposal principal ID",
        )
        require_content_id(
            self.operation_preimage.get("proposer_authority_id"),
            "candidate proposer authority ID",
        )
        _require_digest(
            self.operation_preimage.get("proposal_packet_digest"),
            "candidate proposal packet digest",
        )
        expected_preimage_binding = {
            "ancestor_candidate_ids": self.ancestor_candidate_ids,
            "configuration_fingerprint": self.configuration_fingerprint,
            "operation_kind": self.operation_kind.value,
            "source_base_tree_hash": self.source_base_tree_hash,
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
            self.operation_preimage["principal_id"]
            != self.operation_receipt.principal_id
            or self.operation_preimage["principal_id"]
            != self.proposer_authority.principal_id
            or self.operation_preimage["proposer_authority_id"]
            != self.proposer_authority.authority_id
        ):
            raise ContractValidationError(
                "candidate operation principal differs from its preimage"
            )
        if (
            self.workspace_receipt.operation_receipt_id
            != self.operation_receipt.operation_receipt_id
            or self.workspace_receipt.operation_id
            != self.operation_receipt.operation_id
            or self.workspace_receipt.source_base_tree_hash
            != self.source_base_tree_hash
        ):
            raise ContractValidationError(
                "candidate workspace receipt differs from its operation"
            )
        require_content_id(self.workspace_delta_ref, "workspace_delta_ref")
        _require_digest(self.workspace_delta_digest, "workspace_delta_digest")
        if (
            self.operation_receipt.artifact_checksums.get("workspace-delta.json")
            != self.workspace_delta_digest
        ):
            raise ContractValidationError(
                "candidate workspace delta differs from its operation receipt"
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
    source_base_release_id: str | None
    source_base_repository_map_ref: str | None
    source_base_tree_hash: str
    consumed_expert_release_ids: tuple[str, ...]
    derivation_kind: ExpertCandidateDerivationKind
    derivation_ref: str
    validation_context_ref: str
    patch_ref: str
    patch_digest: str
    candidate_tree_ref: str
    candidate_tree_hash: str
    configuration_fingerprint: str
    module_contract_refs: tuple[str, ...]
    proposed_repository_map_ref: str
    semantic_book_digest: str
    source_dependency_ids: tuple[str, ...]
    ancestor_candidate_ids: tuple[str, ...]
    capability_lineage: tuple[ExpertCapabilityLineage, ...]
    sanitation_report_id: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate"
    IDENTITY_FIELD: ClassVar[str] = "candidate_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        if (self.source_base_release_id is None) != (
            self.source_base_repository_map_ref is None
        ):
            raise ContractValidationError(
                "candidate source-base release and repository map must appear together"
            )
        if self.source_base_release_id is None:
            if (
                self.change_kind is not CandidateChangeKind.REPOSITORY_ARCHITECTURE
                or self.source_base_tree_hash != EMPTY_EXPERT_TREE_DIGEST
            ):
                raise ContractValidationError(
                    "parentless candidate must bootstrap the canonical empty tree"
                )
        elif self.source_base_tree_hash == EMPTY_EXPERT_TREE_DIGEST:
            raise ContractValidationError(
                "released parent candidate cannot use the canonical empty tree"
            )
        if self.consumed_expert_release_ids != tuple(
            sorted(set(self.consumed_expert_release_ids))
        ):
            raise ContractValidationError(
                "candidate consumed expert releases must be sorted and unique"
            )
        for release_id in self.consumed_expert_release_ids:
            require_content_id(release_id, "candidate consumed expert release")
            if release_id.split(":sha256:", 1)[0] != "expert-base-release":
                raise ContractValidationError(
                    "candidate consumed expert release uses the wrong namespace"
                )
        if self.source_base_release_id is not None and (
            self.source_base_release_id not in self.consumed_expert_release_ids
        ):
            raise MissingReferenceError(
                "candidate consumed expert releases omit its source base"
            )
        for value, name in (
            (self.source_base_release_id, "source_base_release_id"),
            (self.source_base_repository_map_ref, "source_base_repository_map_ref"),
        ):
            if value is not None:
                require_content_id(value, name)
        _require_digest(self.source_base_tree_hash, "source_base_tree_hash")
        for value, name in (
            (self.derivation_ref, "derivation_ref"),
            (self.validation_context_ref, "validation_context_ref"),
            (self.patch_ref, "patch_ref"),
            (self.candidate_tree_ref, "candidate_tree_ref"),
            (self.proposed_repository_map_ref, "proposed_repository_map_ref"),
            (self.sanitation_report_id, "sanitation_report_id"),
        ):
            require_content_id(value, name)
        if self.validation_context_ref.split(":sha256:", 1)[0] != (
            "expert-candidate-validation-context"
        ):
            raise ContractValidationError(
                "candidate validation context uses the wrong namespace"
            )
        expected_derivation_namespace = {
            ExpertCandidateDerivationKind.AGENT_PROPOSAL: (
                "expert-agent-proposal-derivation"
            ),
            ExpertCandidateDerivationKind.DETERMINISTIC_COMPOSITION: (
                "expert-deterministic-composition-derivation"
            ),
            ExpertCandidateDerivationKind.DETERMINISTIC_RECOVERY_RESTORE: (
                "expert-deterministic-recovery-restore-derivation"
            ),
        }[self.derivation_kind]
        if self.derivation_ref.split(":sha256:", 1)[0] != (
            expected_derivation_namespace
        ):
            raise ContractValidationError(
                "candidate derivation reference uses the wrong namespace"
            )
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
class ExpertSourceReplayCase(StrictContract):
    source_bundle_id: str
    episode_ids: tuple[str, ...]
    episode_reason_codes: Mapping[str, tuple[str, ...]]

    def _validate(self) -> None:
        require_content_id(self.source_bundle_id, "source replay bundle_id")
        if self.source_bundle_id.split(":sha256:", 1)[0] != "run-bundle":
            raise ContractValidationError("source replay case must name a RunBundle")
        _require_sorted_unique(self.episode_ids, "source replay episode_ids")
        if set(self.episode_reason_codes) != set(self.episode_ids):
            raise MissingReferenceError(
                "source replay case must explain every selected episode"
            )
        for episode_id in self.episode_ids:
            require_content_id(episode_id, "source replay episode_id")
            if episode_id.split(":sha256:", 1)[0] != "transfer-episode":
                raise ContractValidationError(
                    "source replay case must name TransferEpisodes"
                )
            reasons = self.episode_reason_codes[episode_id]
            if not reasons or reasons != tuple(sorted(set(reasons))):
                raise ContractValidationError(
                    "source replay episode reasons must be non-empty, sorted, and unique"
                )
            for reason in reasons:
                require_identifier(reason, "source replay episode reason")


@dataclass(frozen=True)
class ExpertSourceReplayAdapterPackagePin(StrictContract):
    source_adapter_pin_id: str
    scope_contract_id: str
    task_family_id: str
    task_adapter_id: str
    task_adapter_manifest_id: str
    verification_receipt_id: str
    episode_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-adapter-pin"
    IDENTITY_FIELD: ClassVar[str] = "source_adapter_pin_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "source adapter scope_contract_id",
            ),
            (
                self.task_adapter_manifest_id,
                "task-adapter-manifest",
                "source adapter task_adapter_manifest_id",
            ),
            (
                self.verification_receipt_id,
                "task-adapter-verification-receipt",
                "source adapter verification_receipt_id",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ContractValidationError(f"{name} must name a {namespace} record")
        require_identifier(self.task_family_id, "source adapter task_family_id")
        require_identifier(self.task_adapter_id, "source adapter task_adapter_id")
        _require_sorted_unique(self.episode_ids, "source adapter episode_ids")
        if not self.episode_ids:
            raise ContractValidationError(
                "source adapter pin must own at least one episode"
            )
        for episode_id in self.episode_ids:
            require_content_id(episode_id, "source adapter episode_id")
            if episode_id.split(":sha256:", 1)[0] != "transfer-episode":
                raise ContractValidationError(
                    "source adapter pin must name TransferEpisodes"
                )


@dataclass(frozen=True)
class ExpertSourceReplaySelection(StrictContract):
    source_replay_selection_id: str
    candidate_id: str
    candidate_tree_hash: str
    candidate_commit_record_id: str
    validation_context_id: str
    evidence_authority_ids: tuple[str, ...]
    validation_policy_id: str
    selection_policy_version: str
    configuration_fingerprint: str
    causal_episode_ids: tuple[str, ...]
    coverage_episode_ids: tuple[str, ...]
    selection_evidence_ids: tuple[str, ...]
    cases: tuple[ExpertSourceReplayCase, ...]
    source_adapter_pins: tuple[ExpertSourceReplayAdapterPackagePin, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-selection"
    IDENTITY_FIELD: ClassVar[str] = "source_replay_selection_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.candidate_id,
                "expert-candidate",
                "source replay candidate_id",
            ),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "source replay candidate_commit_record_id",
            ),
            (
                self.validation_context_id,
                "expert-candidate-validation-context",
                "source replay validation_context_id",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "source replay validation_policy_id",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ContractValidationError(f"{name} must name a {namespace} record")
        _require_sorted_unique(
            self.evidence_authority_ids,
            "source replay evidence_authority_ids",
        )
        for value in self.evidence_authority_ids:
            require_content_id(value, "source replay evidence_authority_ids")
        _require_digest(
            self.candidate_tree_hash,
            "source replay candidate_tree_hash",
        )
        require_identifier(
            self.selection_policy_version,
            "source replay selection_policy_version",
        )
        _require_digest(
            self.configuration_fingerprint,
            "source replay configuration_fingerprint",
        )
        for values, name in (
            (self.causal_episode_ids, "causal_episode_ids"),
            (self.coverage_episode_ids, "coverage_episode_ids"),
        ):
            if values != tuple(sorted(set(values))):
                raise ContractValidationError(f"{name} must be sorted and unique")
            for episode_id in values:
                require_content_id(episode_id, name)
        if set(self.causal_episode_ids) & set(self.coverage_episode_ids):
            raise ContractValidationError(
                "causal and coverage replay episodes must be disjoint"
            )
        selected_episode_ids = {
            *self.causal_episode_ids,
            *self.coverage_episode_ids,
        }
        if not selected_episode_ids:
            raise ContractValidationError(
                "source replay selection must contain an episode"
            )
        case_keys = tuple(case.source_bundle_id for case in self.cases)
        if not case_keys or case_keys != tuple(sorted(set(case_keys))):
            raise ContractValidationError(
                "source replay cases must be non-empty, sorted, and unique"
            )
        assigned_episode_ids = tuple(
            episode_id for case in self.cases for episode_id in case.episode_ids
        )
        if (
            len(assigned_episode_ids) != len(set(assigned_episode_ids))
            or set(assigned_episode_ids) != selected_episode_ids
        ):
            raise MissingReferenceError(
                "source replay cases must assign selected episodes exactly once"
            )
        adapter_pin_ids = tuple(
            pin.source_adapter_pin_id for pin in self.source_adapter_pins
        )
        if not adapter_pin_ids or adapter_pin_ids != tuple(
            sorted(set(adapter_pin_ids))
        ):
            raise ContractValidationError(
                "source adapter pins must be non-empty, sorted, and unique"
            )
        adapter_episode_ids = tuple(
            episode_id
            for pin in self.source_adapter_pins
            for episode_id in pin.episode_ids
        )
        if (
            len(adapter_episode_ids) != len(set(adapter_episode_ids))
            or set(adapter_episode_ids) != selected_episode_ids
        ):
            raise MissingReferenceError(
                "source adapter pins must assign selected episodes exactly once"
            )
        _require_sorted_unique(
            self.selection_evidence_ids,
            "source replay selection_evidence_ids",
        )
        for evidence_id in self.selection_evidence_ids:
            require_content_id(evidence_id, "source replay selection_evidence_ids")
        _require_sorted_unique(
            self.exact_dependency_ids,
            "source replay exact_dependency_ids",
        )
        required_dependencies = {
            self.candidate_id,
            self.candidate_commit_record_id,
            self.validation_context_id,
            *self.evidence_authority_ids,
            self.validation_policy_id,
            *self.selection_evidence_ids,
            *selected_episode_ids,
            *case_keys,
            *adapter_pin_ids,
            *(pin.task_adapter_manifest_id for pin in self.source_adapter_pins),
            *(pin.verification_receipt_id for pin in self.source_adapter_pins),
        }
        if required_dependencies != set(self.exact_dependency_ids):
            raise MissingReferenceError(
                "source replay selection dependency closure is not exact"
            )


@dataclass(frozen=True)
class ExpertSourceReplayStartingArtifact(StrictContract):
    starting_artifact_content_id: str
    starting_artifact_ref: str
    mount_path: str
    materialized_tree_hash: str
    source_files: tuple[SourceFileDescriptor, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "source-replay-starting-artifact"
    IDENTITY_FIELD: ClassVar[str] = "starting_artifact_content_id"

    def _validate(self) -> None:
        _require_text(self.starting_artifact_ref, "starting_artifact_ref")
        mount_path = PurePosixPath(self.mount_path)
        if (
            not self.mount_path
            or mount_path.is_absolute()
            or ".." in mount_path.parts
            or mount_path == PurePosixPath(".")
            or mount_path.as_posix() != self.mount_path
        ):
            raise ContractValidationError(
                "source replay artifact mount_path must be normalized and relative"
            )
        _require_digest(
            self.materialized_tree_hash,
            "source replay artifact materialized_tree_hash",
        )
        paths = tuple(descriptor.relative_path for descriptor in self.source_files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ContractValidationError(
                "source replay artifact files must be non-empty, sorted, and unique"
            )
        source_paths = tuple(PurePosixPath(path) for path in paths)
        if any(
            source_path in other_path.parents
            for position, source_path in enumerate(source_paths)
            for other_path in source_paths[position + 1 :]
        ):
            raise ContractValidationError(
                "source replay artifact files contain a file/directory collision"
            )
        expected_tree_hash = source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
                for descriptor in self.source_files
            }
        )
        if self.materialized_tree_hash != expected_tree_hash:
            raise ContractValidationError(
                "source replay artifact tree hash differs from its files"
            )


@dataclass(frozen=True)
class ExpertSourceReplayContextMaterializationReceipt(StrictContract):
    context_materialization_receipt_id: str
    task_context_binding_id: str
    input_contract_fingerprint: str
    target_contract_fingerprint: str
    starting_artifacts: tuple[ExpertSourceReplayStartingArtifact, ...]
    materializer_id: str
    materializer_version: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-context-materialization"
    IDENTITY_FIELD: ClassVar[str] = "context_materialization_receipt_id"

    def _validate(self) -> None:
        require_content_id(
            self.task_context_binding_id,
            "source replay context task_context_binding_id",
        )
        if self.task_context_binding_id.split(":sha256:", 1)[0] != (
            "task-context-binding"
        ):
            raise ContractValidationError(
                "source replay context receipt must name a TaskContextBinding"
            )
        for value, name in (
            (self.input_contract_fingerprint, "input_contract_fingerprint"),
            (self.target_contract_fingerprint, "target_contract_fingerprint"),
        ):
            _require_digest(value, f"source replay context {name}")
        artifact_ids = tuple(
            artifact.starting_artifact_content_id
            for artifact in self.starting_artifacts
        )
        if artifact_ids != tuple(sorted(set(artifact_ids))):
            raise ContractValidationError(
                "source replay starting artifacts must be ID-sorted and unique"
            )
        artifact_refs = tuple(
            artifact.starting_artifact_ref for artifact in self.starting_artifacts
        )
        mount_paths = tuple(
            PurePosixPath(artifact.mount_path) for artifact in self.starting_artifacts
        )
        if len(artifact_refs) != len(set(artifact_refs)):
            raise ContractValidationError(
                "source replay starting artifact refs must be unique"
            )
        if len(mount_paths) != len(set(mount_paths)) or any(
            mount_path in other_path.parents or other_path in mount_path.parents
            for position, mount_path in enumerate(mount_paths)
            for other_path in mount_paths[position + 1 :]
        ):
            raise ContractValidationError(
                "source replay starting artifact mounts overlap"
            )
        require_identifier(
            self.materializer_id,
            "source replay context materializer_id",
        )
        require_identifier(
            self.materializer_version,
            "source replay context materializer_version",
        )


@dataclass(frozen=True)
class ExpertSourceReplayExecutionLeg(StrictContract):
    execution_leg_id: str
    kind: ExpertSourceReplayExecutionLegKind
    expert_artifact_id: str
    expert_source_receipt_id: str
    expert_tree_hash: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-execution-leg"
    IDENTITY_FIELD: ClassVar[str] = "execution_leg_id"

    def _validate(self) -> None:
        if self.kind is ExpertSourceReplayExecutionLegKind.SOURCE_BASE_CONTROL:
            artifact_namespace = "expert-base-release"
            receipt_namespace = "expert-source-base-tree-receipt"
        else:
            artifact_namespace = "expert-candidate"
            receipt_namespace = "expert-candidate-commit"
        for value, namespace, name in (
            (self.expert_artifact_id, artifact_namespace, "expert_artifact_id"),
            (
                self.expert_source_receipt_id,
                receipt_namespace,
                "expert_source_receipt_id",
            ),
        ):
            require_content_id(value, f"source replay leg {name}")
            if value.split(":sha256:", 1)[0] != namespace:
                raise ContractValidationError(
                    f"source replay leg {name} must name a {namespace} record"
                )
        _require_digest(self.expert_tree_hash, "source replay leg expert_tree_hash")
        _require_sorted_unique(
            self.exact_dependency_ids,
            "source replay leg exact_dependency_ids",
        )
        if set(self.exact_dependency_ids) != {
            self.expert_artifact_id,
            self.expert_source_receipt_id,
        }:
            raise MissingReferenceError(
                "source replay leg dependency closure is not exact"
            )


@dataclass(frozen=True)
class ExpertSourceReplayComputeBinding(StrictContract):
    compute_binding_id: str
    paired_execution_protocol_version: str
    execution_provider_id: str
    execution_provider_version: str
    execution_provider_settings_digest: str
    sandbox_policy_version: str
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
    leg_order: tuple[ExpertSourceReplayExecutionLegKind, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-compute-binding"
    IDENTITY_FIELD: ClassVar[str] = "compute_binding_id"

    def _validate(self) -> None:
        for value, name in (
            (
                self.paired_execution_protocol_version,
                "paired_execution_protocol_version",
            ),
            (self.execution_provider_id, "execution_provider_id"),
            (self.execution_provider_version, "execution_provider_version"),
            (self.sandbox_policy_version, "sandbox_policy_version"),
        ):
            require_identifier(value, f"source replay compute {name}")
        _require_digest(
            self.execution_provider_settings_digest,
            "source replay compute execution_provider_settings_digest",
        )
        for value, name in (
            (self.leg_wall_time_limit_seconds, "leg_wall_time_limit_seconds"),
            (self.termination_grace_seconds, "termination_grace_seconds"),
            (self.cpu_millicore_limit, "cpu_millicore_limit"),
            (self.memory_byte_limit, "memory_byte_limit"),
            (self.shared_memory_byte_limit, "shared_memory_byte_limit"),
            (self.process_limit, "process_limit"),
            (self.open_file_limit, "open_file_limit"),
            (self.writable_inode_limit, "writable_inode_limit"),
            (
                self.writable_storage_byte_limit,
                "writable_storage_byte_limit",
            ),
            (self.output_entry_limit, "output_entry_limit"),
            (self.output_byte_limit, "output_byte_limit"),
            (self.stdout_byte_limit, "stdout_byte_limit"),
            (self.stderr_byte_limit, "stderr_byte_limit"),
        ):
            if type(value) is not int or value <= 0:
                raise ContractValidationError(
                    f"source replay compute {name} must be a positive integer"
                )
        if (
            self.termination_grace_seconds > self.leg_wall_time_limit_seconds
            or self.shared_memory_byte_limit > self.memory_byte_limit
            or self.output_entry_limit >= self.writable_inode_limit
            or self.output_byte_limit > self.writable_storage_byte_limit
        ):
            raise ContractValidationError(
                "source replay compute limits are internally inconsistent"
            )
        if type(self.accelerator_count) is not int or self.accelerator_count < 0:
            raise ContractValidationError(
                "source replay compute accelerator_count must be non-negative"
            )
        if (self.accelerator_class_id is None) != (self.accelerator_count == 0):
            raise ContractValidationError(
                "source replay compute accelerator class and count must be present together"
            )
        if self.accelerator_class_id is not None:
            require_identifier(
                self.accelerator_class_id,
                "source replay compute accelerator_class_id",
            )
        if len(self.leg_order) != 2 or set(self.leg_order) != set(
            ExpertSourceReplayExecutionLegKind
        ):
            raise ContractValidationError(
                "source replay compute leg_order must contain both legs exactly once"
            )


def expert_source_replay_matched_compute_digest(
    *,
    bundle_lineage_ids: tuple[str, ...],
    projection_manifest_id: str,
    episode_id: str,
    source_execution_revision: int,
    source_evaluation_fingerprint_ids: tuple[str, ...],
    source_score_of_record_fingerprint_id: str,
    task_context_binding_id: str,
    context_materialization_receipt_id: str,
    starting_artifact_content_ids: tuple[str, ...],
    task_adapter_manifest_id: str,
    verification_receipt_id: str,
    task_adapter_source_tree_hash: str,
    task_evaluator_digest: str,
    task_adapter_runtime_digest: str,
    task_adapter_context_binding_digest: str,
    compute_binding_id: str,
) -> str:
    """Derive the immutable environment shared by both replay legs."""

    return tree_or_blob_digest(
        canonical_json_bytes(
            {
                "bundle_lineage_ids": bundle_lineage_ids,
                "context_materialization_receipt_id": (
                    context_materialization_receipt_id
                ),
                "compute_binding_id": compute_binding_id,
                "episode_id": episode_id,
                "projection_manifest_id": projection_manifest_id,
                "source_evaluation_fingerprint_ids": (
                    source_evaluation_fingerprint_ids
                ),
                "source_execution_revision": source_execution_revision,
                "source_score_of_record_fingerprint_id": (
                    source_score_of_record_fingerprint_id
                ),
                "starting_artifact_content_ids": starting_artifact_content_ids,
                "task_adapter_manifest_id": task_adapter_manifest_id,
                "task_adapter_runtime_digest": task_adapter_runtime_digest,
                "task_adapter_context_binding_digest": (
                    task_adapter_context_binding_digest
                ),
                "task_adapter_source_tree_hash": task_adapter_source_tree_hash,
                "task_evaluator_digest": task_evaluator_digest,
                "task_context_binding_id": task_context_binding_id,
                "verification_receipt_id": verification_receipt_id,
            }
        )
    )


@dataclass(frozen=True)
class ExpertSourceReplayExecutionCase(StrictContract):
    execution_case_id: str
    source_bundle_id: str
    bundle_lineage_ids: tuple[str, ...]
    projection_manifest_id: str
    episode_id: str
    source_node_id: str
    source_execution_revision: int
    source_evaluation_fingerprint_ids: tuple[str, ...]
    source_score_of_record_fingerprint_id: str
    episode_reason_codes: tuple[str, ...]
    task_context_binding_id: str
    source_expert_base_release_id: str
    context_materialization_receipt_id: str
    starting_artifact_content_ids: tuple[str, ...]
    adapter_binding_id: str
    task_adapter_manifest_id: str
    verification_receipt_id: str
    task_adapter_source_tree_hash: str
    task_evaluator_digest: str
    task_adapter_runtime_digest: str
    task_adapter_context_binding_digest: str
    task_adapter_dependency_ids: tuple[str, ...]
    compute_binding: ExpertSourceReplayComputeBinding
    matched_compute_binding_digest: str
    control_leg: ExpertSourceReplayExecutionLeg
    candidate_leg: ExpertSourceReplayExecutionLeg
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-execution-case"
    IDENTITY_FIELD: ClassVar[str] = "execution_case_id"

    def _validate(self) -> None:
        if not self.bundle_lineage_ids or len(self.bundle_lineage_ids) != len(
            set(self.bundle_lineage_ids)
        ):
            raise ContractValidationError(
                "source replay bundle lineage must be non-empty and unique"
            )
        for bundle_id in self.bundle_lineage_ids:
            require_content_id(bundle_id, "source replay bundle_lineage_ids")
            if bundle_id.split(":sha256:", 1)[0] != "run-bundle":
                raise ContractValidationError(
                    "source replay bundle lineage must name RunBundles"
                )
        if self.bundle_lineage_ids[-1] != self.source_bundle_id:
            raise ContractValidationError(
                "source replay execution case must use the lineage tip"
            )
        for value, namespace, name in (
            (
                self.projection_manifest_id,
                "bundle-projection-manifest",
                "source replay projection_manifest_id",
            ),
            (self.episode_id, "transfer-episode", "source replay episode_id"),
            (
                self.task_context_binding_id,
                "task-context-binding",
                "source replay task_context_binding_id",
            ),
            (
                self.source_expert_base_release_id,
                "expert-base-release",
                "source replay source_expert_base_release_id",
            ),
            (
                self.context_materialization_receipt_id,
                "expert-source-replay-context-materialization",
                "source replay context_materialization_receipt_id",
            ),
            (
                self.adapter_binding_id,
                "task-adapter-binding",
                "source replay adapter_binding_id",
            ),
            (
                self.task_adapter_manifest_id,
                "task-adapter-manifest",
                "source replay task_adapter_manifest_id",
            ),
            (
                self.verification_receipt_id,
                "task-adapter-verification-receipt",
                "source replay verification_receipt_id",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ContractValidationError(f"{name} must name a {namespace} record")
        if not self.episode_reason_codes or self.episode_reason_codes != tuple(
            sorted(set(self.episode_reason_codes))
        ):
            raise ContractValidationError(
                "source replay execution reasons must be non-empty and unique"
            )
        for reason in self.episode_reason_codes:
            require_identifier(reason, "source replay execution reason")
        require_identifier(self.source_node_id, "source replay source_node_id")
        if self.source_execution_revision < 0:
            raise ContractValidationError(
                "source replay source_execution_revision must be non-negative"
            )
        if self.source_evaluation_fingerprint_ids:
            _require_sorted_unique(
                self.source_evaluation_fingerprint_ids,
                "source replay source_evaluation_fingerprint_ids",
            )
            for fingerprint_id in self.source_evaluation_fingerprint_ids:
                require_content_id(
                    fingerprint_id,
                    "source replay source_evaluation_fingerprint_ids",
                )
                if fingerprint_id.split(":sha256:", 1)[0] != ("evaluation-fingerprint"):
                    raise ContractValidationError(
                        "source replay fingerprints must name EvaluationFingerprints"
                    )
        require_content_id(
            self.source_score_of_record_fingerprint_id,
            "source replay source_score_of_record_fingerprint_id",
        )
        if (
            self.source_score_of_record_fingerprint_id.split(":sha256:", 1)[0]
            != "evaluation-fingerprint"
            or self.source_score_of_record_fingerprint_id
            not in self.source_evaluation_fingerprint_ids
        ):
            raise ContractValidationError(
                "source replay score of record must name one source fingerprint"
            )
        if self.starting_artifact_content_ids:
            _require_sorted_unique(
                self.starting_artifact_content_ids,
                "source replay starting_artifact_content_ids",
            )
            for artifact_id in self.starting_artifact_content_ids:
                require_content_id(
                    artifact_id,
                    "source replay starting_artifact_content_ids",
                )
                if artifact_id.split(":sha256:", 1)[0] != (
                    "source-replay-starting-artifact"
                ):
                    raise ContractValidationError(
                        "source replay artifacts must name starting-artifact records"
                    )
        _require_digest(
            self.task_adapter_source_tree_hash,
            "source replay task_adapter_source_tree_hash",
        )
        _require_digest(
            self.task_evaluator_digest,
            "source replay task_evaluator_digest",
        )
        _require_digest(
            self.task_adapter_runtime_digest,
            "source replay task_adapter_runtime_digest",
        )
        _require_digest(
            self.task_adapter_context_binding_digest,
            "source replay task_adapter_context_binding_digest",
        )
        _require_digest(
            self.matched_compute_binding_digest,
            "source replay matched_compute_binding_digest",
        )
        if self.matched_compute_binding_digest != (
            expert_source_replay_matched_compute_digest(
                bundle_lineage_ids=self.bundle_lineage_ids,
                projection_manifest_id=self.projection_manifest_id,
                episode_id=self.episode_id,
                source_execution_revision=self.source_execution_revision,
                source_evaluation_fingerprint_ids=(
                    self.source_evaluation_fingerprint_ids
                ),
                source_score_of_record_fingerprint_id=(
                    self.source_score_of_record_fingerprint_id
                ),
                task_context_binding_id=self.task_context_binding_id,
                context_materialization_receipt_id=(
                    self.context_materialization_receipt_id
                ),
                starting_artifact_content_ids=self.starting_artifact_content_ids,
                task_adapter_manifest_id=self.task_adapter_manifest_id,
                verification_receipt_id=self.verification_receipt_id,
                task_adapter_source_tree_hash=self.task_adapter_source_tree_hash,
                task_evaluator_digest=self.task_evaluator_digest,
                task_adapter_runtime_digest=self.task_adapter_runtime_digest,
                task_adapter_context_binding_digest=(
                    self.task_adapter_context_binding_digest
                ),
                compute_binding_id=self.compute_binding.compute_binding_id,
            )
        ):
            raise ContractValidationError(
                "source replay matched-compute digest differs from its case"
            )
        _require_sorted_unique(
            self.task_adapter_dependency_ids,
            "source replay task_adapter_dependency_ids",
        )
        for dependency_id in self.task_adapter_dependency_ids:
            require_content_id(
                dependency_id,
                "source replay task_adapter_dependency_ids",
            )
        if self.verification_receipt_id not in self.task_adapter_dependency_ids:
            raise MissingReferenceError(
                "source replay adapter dependencies omit the verification receipt"
            )
        if (
            self.control_leg.kind
            is not ExpertSourceReplayExecutionLegKind.SOURCE_BASE_CONTROL
            or self.candidate_leg.kind
            is not ExpertSourceReplayExecutionLegKind.CANDIDATE
            or self.control_leg.expert_artifact_id
            == self.candidate_leg.expert_artifact_id
        ):
            raise ContractValidationError(
                "source replay case requires distinct parent-control and candidate legs"
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "source replay execution case exact_dependency_ids",
        )
        expected_dependencies = {
            *self.bundle_lineage_ids,
            self.projection_manifest_id,
            self.episode_id,
            *self.source_evaluation_fingerprint_ids,
            self.task_context_binding_id,
            self.source_expert_base_release_id,
            self.context_materialization_receipt_id,
            *self.starting_artifact_content_ids,
            self.adapter_binding_id,
            self.task_adapter_manifest_id,
            self.verification_receipt_id,
            self.compute_binding.compute_binding_id,
            *self.task_adapter_dependency_ids,
            self.control_leg.execution_leg_id,
            *self.control_leg.exact_dependency_ids,
            self.candidate_leg.execution_leg_id,
            *self.candidate_leg.exact_dependency_ids,
        }
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise MissingReferenceError(
                "source replay execution case dependency closure is not exact"
            )


@dataclass(frozen=True)
class ExpertSourceReplayExecutionRequest(StrictContract):
    execution_request_id: str
    validation_attempt_id: str
    authorization_state_id: str
    source_replay_selection_id: str
    candidate_id: str
    candidate_tree_hash: str
    candidate_commit_record_id: str
    candidate_source_tree_manifest_id: str
    scope_contract_id: str
    source_base_release_id: str
    source_base_tree_receipt_id: str
    source_base_extraction_receipt_id: str
    source_base_tree_hash: str
    validation_policy_id: str
    configuration_fingerprint: str
    request_policy_version: str
    evaluator_id: str
    evaluator_role: str
    evaluator_version: str
    attempt_dependency_ids: tuple[str, ...]
    cases: tuple[ExpertSourceReplayExecutionCase, ...]
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-execution-request"
    IDENTITY_FIELD: ClassVar[str] = "execution_request_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "source replay validation_attempt_id",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "source replay authorization_state_id",
            ),
            (
                self.source_replay_selection_id,
                "expert-source-replay-selection",
                "source replay source_replay_selection_id",
            ),
            (self.candidate_id, "expert-candidate", "source replay candidate_id"),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "source replay candidate_commit_record_id",
            ),
            (
                self.candidate_source_tree_manifest_id,
                "expert-source-tree",
                "source replay candidate_source_tree_manifest_id",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "source replay scope_contract_id",
            ),
            (
                self.source_base_release_id,
                "expert-base-release",
                "source replay source_base_release_id",
            ),
            (
                self.source_base_tree_receipt_id,
                "expert-source-base-tree-receipt",
                "source replay source_base_tree_receipt_id",
            ),
            (
                self.source_base_extraction_receipt_id,
                "source-archive-extraction-receipt",
                "source replay source_base_extraction_receipt_id",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "source replay validation_policy_id",
            ),
        ):
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ContractValidationError(f"{name} must name a {namespace} record")
        _require_digest(self.candidate_tree_hash, "source replay candidate_tree_hash")
        _require_digest(
            self.source_base_tree_hash, "source replay source_base_tree_hash"
        )
        _require_digest(
            self.configuration_fingerprint,
            "source replay configuration_fingerprint",
        )
        require_identifier(
            self.request_policy_version,
            "source replay request_policy_version",
        )
        for value, name in (
            (self.evaluator_id, "source replay evaluator_id"),
            (self.evaluator_role, "source replay evaluator_role"),
            (self.evaluator_version, "source replay evaluator_version"),
        ):
            require_identifier(value, name)
        case_episode_ids = tuple(case.episode_id for case in self.cases)
        if not case_episode_ids or case_episode_ids != tuple(
            sorted(set(case_episode_ids))
        ):
            raise ContractValidationError(
                "source replay request cases must be episode-sorted and unique"
            )
        for case in self.cases:
            if (
                case.candidate_leg.expert_artifact_id != self.candidate_id
                or case.candidate_leg.expert_source_receipt_id
                != self.candidate_commit_record_id
                or case.candidate_leg.expert_tree_hash != self.candidate_tree_hash
                or case.control_leg.expert_artifact_id != self.source_base_release_id
                or case.control_leg.expert_source_receipt_id
                != self.source_base_tree_receipt_id
                or case.control_leg.expert_tree_hash != self.source_base_tree_hash
            ):
                raise ContractValidationError(
                    "source replay request legs differ from aggregate authority"
                )
        _require_sorted_unique(
            self.attempt_dependency_ids,
            "source replay request attempt_dependency_ids",
        )
        for dependency_id in self.attempt_dependency_ids:
            require_content_id(
                dependency_id,
                "source replay request attempt_dependency_ids",
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "source replay request exact_dependency_ids",
        )
        expected_dependencies = {
            self.validation_attempt_id,
            self.authorization_state_id,
            self.source_replay_selection_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.candidate_source_tree_manifest_id,
            self.scope_contract_id,
            self.source_base_release_id,
            self.source_base_tree_receipt_id,
            self.source_base_extraction_receipt_id,
            self.validation_policy_id,
            *self.attempt_dependency_ids,
            *(
                dependency_id
                for case in self.cases
                for dependency_id in (
                    case.execution_case_id,
                    *case.exact_dependency_ids,
                )
            ),
        }
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise MissingReferenceError(
                "source replay execution request dependency closure is not exact"
            )


@dataclass(frozen=True)
class ExpertSourceReplayExecutionReservation(StrictContract):
    reservation_id: str
    execution_request_id: str
    authorization_transition_id: str
    validation_attempt_id: str
    authorization_state_id: str
    candidate_id: str
    candidate_tree_hash: str
    expected_current_release_id: str
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-source-replay-execution-reservation"
    IDENTITY_FIELD: ClassVar[str] = "reservation_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.execution_request_id,
                "expert-source-replay-execution-request",
                "execution_request_id",
            ),
            (
                self.authorization_transition_id,
                "expert-validation-transition",
                "authorization_transition_id",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "validation_attempt_id",
            ),
            (
                self.authorization_state_id,
                "expert-candidate-validation-state",
                "authorization_state_id",
            ),
            (self.candidate_id, "expert-candidate", "candidate_id"),
            (
                self.expected_current_release_id,
                "expert-base-release",
                "expected_current_release_id",
            ),
        ):
            require_content_id(value, f"source replay reservation {name}")
            if value.split(":sha256:", 1)[0] != namespace:
                raise ContractValidationError(
                    f"source replay reservation {name} must name a {namespace} record"
                )
        _require_digest(
            self.candidate_tree_hash,
            "source replay reservation candidate_tree_hash",
        )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "source replay reservation exact_dependency_ids",
        )
        expected_dependencies = {
            self.execution_request_id,
            self.authorization_transition_id,
            self.validation_attempt_id,
            self.authorization_state_id,
            self.candidate_id,
            self.expected_current_release_id,
        }
        if set(self.exact_dependency_ids) != expected_dependencies:
            raise MissingReferenceError(
                "source replay reservation dependency closure is not exact"
            )


@dataclass(frozen=True)
class TaskAdapterPackagePin(StrictContract):
    adapter_binding_id: str
    task_adapter_manifest_id: str
    verification_receipt_id: str

    def _validate(self) -> None:
        for value, name in (
            (self.adapter_binding_id, "adapter_binding_id"),
            (self.task_adapter_manifest_id, "task_adapter_manifest_id"),
            (self.verification_receipt_id, "verification_receipt_id"),
        ):
            require_content_id(value, name)


@dataclass(frozen=True)
class ExpertCandidateEligibilityDecision(StrictContract):
    eligibility_decision_id: str
    candidate_id: str
    candidate_tree_hash: str
    candidate_commit_record_id: str
    scope_contract_id: str
    source_base_release_id: str | None
    validation_policy_id: str
    configuration_fingerprint: str
    eligible: bool
    validation_track: ExpertValidationTrack
    required_stages: tuple[ExpertValidationStage, ...]
    configured_task_family_ids: tuple[str, ...]
    task_adapter_pins: tuple[TaskAdapterPackagePin, ...]
    source_replay_selection: ExpertSourceReplaySelection | None
    exact_dependency_ids: tuple[str, ...]
    reason_code: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-eligibility"
    IDENTITY_FIELD: ClassVar[str] = "eligibility_decision_id"

    def _validate(self) -> None:
        for value, name in (
            (self.candidate_id, "candidate_id"),
            (self.candidate_commit_record_id, "candidate_commit_record_id"),
            (self.scope_contract_id, "scope_contract_id"),
            (self.validation_policy_id, "validation_policy_id"),
        ):
            require_content_id(value, name)
        if self.source_base_release_id is not None:
            require_content_id(self.source_base_release_id, "source_base_release_id")
        _require_digest(self.candidate_tree_hash, "candidate_tree_hash")
        _require_digest(
            self.configuration_fingerprint,
            "configuration_fingerprint",
        )
        require_identifier(self.reason_code, "eligibility reason_code")
        if not self.configured_task_family_ids or self.configured_task_family_ids != (
            tuple(sorted(set(self.configured_task_family_ids)))
        ):
            raise ContractValidationError(
                "configured task families must be non-empty, sorted, and unique"
            )
        for task_family_id in self.configured_task_family_ids:
            require_identifier(task_family_id, "configured_task_family_ids")
        pin_keys = tuple(pin.adapter_binding_id for pin in self.task_adapter_pins)
        if not pin_keys or pin_keys != tuple(sorted(set(pin_keys))):
            raise ContractValidationError(
                "task adapter package pins must be non-empty, sorted, and unique"
            )
        manifest_ids = tuple(
            pin.task_adapter_manifest_id for pin in self.task_adapter_pins
        )
        receipt_ids = tuple(
            pin.verification_receipt_id for pin in self.task_adapter_pins
        )
        if len(manifest_ids) != len(set(manifest_ids)) or len(receipt_ids) != len(
            set(receipt_ids)
        ):
            raise ContractValidationError(
                "task adapter package pins must reference unique packages"
            )
        _require_sorted_unique(self.exact_dependency_ids, "exact_dependency_ids")
        for value in self.exact_dependency_ids:
            require_content_id(value, "exact_dependency_ids")
        required_dependencies = {
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.validation_policy_id,
            *manifest_ids,
            *receipt_ids,
        }
        if self.source_base_release_id is not None:
            required_dependencies.add(self.source_base_release_id)
        source_replay_required = (
            self.eligible
            and ExpertValidationStage.SOURCE_RUN_REPLAY in self.required_stages
        )
        if (self.source_replay_selection is not None) != source_replay_required:
            raise ContractValidationError(
                "source replay stages require exactly one replay selection"
            )
        if self.source_replay_selection is not None:
            selection = self.source_replay_selection
            if (
                selection.candidate_id != self.candidate_id
                or selection.candidate_tree_hash != self.candidate_tree_hash
                or selection.candidate_commit_record_id
                != self.candidate_commit_record_id
                or selection.validation_policy_id != self.validation_policy_id
                or selection.configuration_fingerprint != self.configuration_fingerprint
            ):
                raise ContractValidationError(
                    "eligibility source replay authority differs"
                )
            required_dependencies.update(
                {
                    selection.source_replay_selection_id,
                    *selection.exact_dependency_ids,
                }
            )
        if not required_dependencies.issubset(self.exact_dependency_ids):
            raise MissingReferenceError(
                "eligibility decision dependency closure is incomplete"
            )
        if self.eligible != bool(self.required_stages):
            raise ContractValidationError(
                "only an eligible candidate may have a required stage plan"
            )
        if self.required_stages:
            stage_order = tuple(ExpertValidationStage)
            expected_order = tuple(
                stage for stage in stage_order if stage in self.required_stages
            )
            if (
                len(self.required_stages) != len(set(self.required_stages))
                or self.required_stages != expected_order
            ):
                raise ContractValidationError(
                    "required_stages must be unique and canonically ordered"
                )


@dataclass(frozen=True)
class ExpertValidationAttempt(StrictContract):
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    candidate_commit_record_id: str
    scope_contract_id: str
    source_base_release_id: str | None
    eligibility_decision_id: str
    validation_policy_id: str
    configuration_fingerprint: str
    validation_track: ExpertValidationTrack
    attempt_number: int
    predecessor_attempt_id: str | None
    required_stages: tuple[ExpertValidationStage, ...]
    configured_task_family_ids: tuple[str, ...]
    task_adapter_pins: tuple[TaskAdapterPackagePin, ...]
    source_replay_selection: ExpertSourceReplaySelection | None
    eligibility_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-validation-attempt"
    IDENTITY_FIELD: ClassVar[str] = "validation_attempt_id"

    def _validate(self) -> None:
        for value, name in (
            (self.candidate_id, "candidate_id"),
            (self.candidate_commit_record_id, "candidate_commit_record_id"),
            (self.scope_contract_id, "scope_contract_id"),
            (self.eligibility_decision_id, "eligibility_decision_id"),
            (self.validation_policy_id, "validation_policy_id"),
        ):
            require_content_id(value, name)
        if self.source_base_release_id is not None:
            require_content_id(self.source_base_release_id, "source_base_release_id")
        _require_digest(self.candidate_tree_hash, "candidate_tree_hash")
        _require_digest(
            self.configuration_fingerprint,
            "configuration_fingerprint",
        )
        if self.attempt_number <= 0:
            raise ContractValidationError("attempt_number must be positive")
        if (self.attempt_number == 1) != (self.predecessor_attempt_id is None):
            raise ContractValidationError(
                "only the first validation attempt may omit its predecessor"
            )
        if self.predecessor_attempt_id is not None:
            require_content_id(
                self.predecessor_attempt_id,
                "predecessor_attempt_id",
            )
        if not self.required_stages:
            raise ContractValidationError("required_stages must not be empty")
        if len(self.required_stages) != len(set(self.required_stages)):
            raise ContractValidationError("required_stages must be unique")
        stage_order = tuple(ExpertValidationStage)
        expected_order = tuple(
            stage for stage in stage_order if stage in self.required_stages
        )
        if self.required_stages != expected_order:
            raise ContractValidationError(
                "required_stages must follow the canonical evaluator order"
            )
        pin_keys = tuple(pin.adapter_binding_id for pin in self.task_adapter_pins)
        if not pin_keys or pin_keys != tuple(sorted(set(pin_keys))):
            raise ContractValidationError(
                "task adapter package pins must be non-empty, sorted, and unique"
            )
        manifest_ids = tuple(
            pin.task_adapter_manifest_id for pin in self.task_adapter_pins
        )
        receipt_ids = tuple(
            pin.verification_receipt_id for pin in self.task_adapter_pins
        )
        if len(manifest_ids) != len(set(manifest_ids)) or len(receipt_ids) != len(
            set(receipt_ids)
        ):
            raise ContractValidationError(
                "task adapter package pins must reference unique packages"
            )
        if not self.configured_task_family_ids or self.configured_task_family_ids != (
            tuple(sorted(set(self.configured_task_family_ids)))
        ):
            raise ContractValidationError(
                "configured task families must be non-empty, sorted, and unique"
            )
        for task_family_id in self.configured_task_family_ids:
            require_identifier(task_family_id, "configured_task_family_ids")
        _require_sorted_unique(
            self.eligibility_dependency_ids,
            "eligibility_dependency_ids",
        )
        for dependency_id in self.eligibility_dependency_ids:
            require_content_id(dependency_id, "eligibility_dependency_ids")
        required_dependencies = {
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.eligibility_decision_id,
            self.validation_policy_id,
            *manifest_ids,
            *receipt_ids,
        }
        if self.source_base_release_id is not None:
            required_dependencies.add(self.source_base_release_id)
        source_replay_required = (
            ExpertValidationStage.SOURCE_RUN_REPLAY in self.required_stages
        )
        if (self.source_replay_selection is not None) != source_replay_required:
            raise ContractValidationError(
                "source replay stages require exactly one replay selection"
            )
        if self.source_replay_selection is not None:
            selection = self.source_replay_selection
            if (
                selection.candidate_id != self.candidate_id
                or selection.candidate_tree_hash != self.candidate_tree_hash
                or selection.candidate_commit_record_id
                != self.candidate_commit_record_id
                or selection.validation_policy_id != self.validation_policy_id
                or selection.configuration_fingerprint != self.configuration_fingerprint
            ):
                raise ContractValidationError(
                    "validation attempt source replay authority differs"
                )
            required_dependencies.update(
                {
                    selection.source_replay_selection_id,
                    *selection.exact_dependency_ids,
                }
            )
        if not required_dependencies.issubset(self.eligibility_dependency_ids):
            raise MissingReferenceError(
                "validation attempt eligibility dependency closure is incomplete"
            )


@dataclass(frozen=True)
class ExpertSealedCanaryAggregate(StrictContract):
    candidate_id: str
    candidate_tree_hash: str
    evaluator_version: str
    evaluated_case_count: int
    aggregate_measurements: Mapping[str, float]

    def _validate(self) -> None:
        require_content_id(self.candidate_id, "sealed aggregate candidate_id")
        _require_digest(
            self.candidate_tree_hash,
            "sealed aggregate candidate_tree_hash",
        )
        require_identifier(
            self.evaluator_version,
            "sealed aggregate evaluator_version",
        )
        if self.evaluated_case_count <= 0:
            raise ContractValidationError(
                "sealed aggregate evaluated_case_count must be positive"
            )
        if not self.aggregate_measurements:
            raise ContractValidationError(
                "sealed aggregate measurements must not be empty"
            )
        for key, value in self.aggregate_measurements.items():
            require_identifier(key, "sealed aggregate measurement key")
            if not math.isfinite(value):
                raise ContractValidationError(
                    f"sealed aggregate measurement {key} must be finite"
                )


@dataclass(frozen=True)
class ExpertEvaluatorRun(StrictContract):
    evaluator_run_id: str
    validation_attempt_id: str
    candidate_id: str
    candidate_tree_hash: str
    stage: ExpertValidationStage
    evaluator_id: str
    evaluator_role: str
    evaluator_version: str
    exact_input_ids: tuple[str, ...]
    output_payloads_base64: Mapping[str, str]
    output_checksums: Mapping[str, str]
    measurements: Mapping[str, float]
    costs: Mapping[str, float]
    duration_seconds: float
    outcome: ExpertEvaluatorOutcome

    CONTENT_NAMESPACE: ClassVar[str] = "expert-evaluator-run"
    IDENTITY_FIELD: ClassVar[str] = "evaluator_run_id"

    def _validate(self) -> None:
        for value, name in (
            (self.validation_attempt_id, "validation_attempt_id"),
            (self.candidate_id, "candidate_id"),
        ):
            require_content_id(value, name)
        _require_digest(self.candidate_tree_hash, "candidate_tree_hash")
        for value, name in (
            (self.evaluator_id, "evaluator_id"),
            (self.evaluator_role, "evaluator_role"),
            (self.evaluator_version, "evaluator_version"),
        ):
            require_identifier(value, name)
        _require_sorted_unique(self.exact_input_ids, "exact_input_ids")
        for value in self.exact_input_ids:
            require_content_id(value, "exact_input_ids")
        _require_checksum_mapping(self.output_checksums, "output_checksums")
        if set(self.output_payloads_base64) != set(self.output_checksums):
            raise MissingReferenceError(
                "evaluator output payloads and checksums must name the same files"
            )
        for path, payload_base64 in self.output_payloads_base64.items():
            _require_relative_path(path, "output_payloads_base64 key")
            if not isinstance(payload_base64, str):
                raise ContractValidationError(
                    "evaluator output payload must be base64 text"
                )
            payload = base64.b64decode(payload_base64, validate=True)
            if tree_or_blob_digest(payload) != self.output_checksums[path]:
                raise ContractValidationError(
                    f"evaluator output checksum differs: {path}"
                )
        if self.duration_seconds < 0.0:
            raise ContractValidationError("duration_seconds must be non-negative")
        for key, value in self.measurements.items():
            require_identifier(key, "measurements key")
            if not math.isfinite(value):
                raise ContractValidationError(f"measurements[{key}] must be finite")
        for key, value in self.costs.items():
            require_identifier(key, "costs key")
            if not math.isfinite(value) or value < 0.0:
                raise ContractValidationError(
                    f"costs[{key}] must be finite and non-negative"
                )
        if self.stage is ExpertValidationStage.SEALED_CANARY:
            if set(self.output_payloads_base64) != {"aggregate.json"}:
                raise ContractValidationError(
                    "sealed canary output must contain only aggregate.json"
                )
            aggregate = ExpertSealedCanaryAggregate.from_json_bytes(
                base64.b64decode(
                    self.output_payloads_base64["aggregate.json"],
                    validate=True,
                )
            )
            if (
                aggregate.candidate_id != self.candidate_id
                or aggregate.candidate_tree_hash != self.candidate_tree_hash
                or aggregate.evaluator_version != self.evaluator_version
                or aggregate.aggregate_measurements != self.measurements
            ):
                raise ContractValidationError(
                    "sealed canary aggregate differs from evaluator run"
                )


@dataclass(frozen=True)
class ExpertEvaluatorAttestation(StrictContract):
    evaluator_attestation_id: str
    evaluator_run_id: str
    issuer_id: str
    trust_root_id: str | None
    predicate_digest: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-evaluator-attestation"
    IDENTITY_FIELD: ClassVar[str] = "evaluator_attestation_id"

    def _validate(self) -> None:
        require_content_id(self.evaluator_run_id, "evaluator_run_id")
        require_identifier(self.issuer_id, "attestation issuer_id")
        if self.trust_root_id is not None:
            require_identifier(self.trust_root_id, "attestation trust_root_id")
        _require_digest(self.predicate_digest, "attestation predicate_digest")


@dataclass(frozen=True)
class ExpertEvaluatorAttestationEnvelope(StrictContract):
    attestation: ExpertEvaluatorAttestation
    signature: str

    def _validate(self) -> None:
        _require_text(self.signature, "attestation signature")


@dataclass(frozen=True)
class ExpertEvaluatorResultRecord(StrictContract):
    evaluator_result_record_id: str
    evaluator_run: ExpertEvaluatorRun
    attestation_envelope: ExpertEvaluatorAttestationEnvelope

    CONTENT_NAMESPACE: ClassVar[str] = "expert-evaluator-result-record"
    IDENTITY_FIELD: ClassVar[str] = "evaluator_result_record_id"

    def _validate(self) -> None:
        attestation = self.attestation_envelope.attestation
        if (
            attestation.evaluator_run_id != self.evaluator_run.evaluator_run_id
            or attestation.issuer_id != self.evaluator_run.evaluator_id
            or attestation.predicate_digest
            != tree_or_blob_digest(self.evaluator_run.to_json_bytes())
        ):
            raise ContractValidationError(
                "evaluator result record attestation does not bind its run"
            )


@dataclass(frozen=True)
class ExpertAcceptedStageResultRef(StrictContract):
    stage: ExpertValidationStage
    stage_result_record_id: str

    def _validate(self) -> None:
        require_content_id(
            self.stage_result_record_id,
            "accepted stage_result_record_id",
        )
        evaluator_stages = {
            ExpertValidationStage.CONTRACT_SCHEMA,
            ExpertValidationStage.IDENTITY_SECRETS_LICENSE_DEPENDENCY,
            ExpertValidationStage.STATIC_UNIT_SECURITY_RESOURCE,
            ExpertValidationStage.SYNTHETIC_FRESH_TASK,
            ExpertValidationStage.DEVELOPMENT_ANCHORS,
            ExpertValidationStage.CROSS_FAMILY_TRANSFER,
            ExpertValidationStage.SEALED_CANARY,
        }
        if self.stage in evaluator_stages:
            expected_namespace = "expert-evaluator-result-record"
        elif self.stage is ExpertValidationStage.SOURCE_RUN_REPLAY:
            expected_namespace = "expert-source-replay-stage-result"
        elif self.stage is ExpertValidationStage.AUTOMATED_REVIEW:
            expected_namespace = "expert-automated-review-stage-result"
        elif self.stage is ExpertValidationStage.RELEASE_MATRIX:
            expected_namespace = "expert-release-matrix-stage-result"
        elif self.stage is ExpertValidationStage.PUBLICATION_ELIGIBILITY:
            expected_namespace = "expert-publication-eligibility-stage-result"
        else:
            raise ContractValidationError(
                "accepted stage result uses an unsupported stage"
            )
        if self.stage_result_record_id.split(":sha256:", 1)[0] != expected_namespace:
            raise ContractValidationError(
                "accepted stage result record uses the wrong namespace"
            )


@dataclass(frozen=True)
class ExpertValidationAuthorityInvalidation(StrictContract):
    authority_invalidation_id: str
    kind: ExpertValidationAuthorityInvalidationKind
    validation_attempt_id: str
    authorization_state_id: str
    candidate_id: str
    candidate_tree_hash: str
    scope_contract_id: str
    expected_current_release_id: str | None
    observed_current_release_id: str | None
    exact_dependency_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-validation-authority-invalidation"
    IDENTITY_FIELD: ClassVar[str] = "authority_invalidation_id"

    def _validate(self) -> None:
        for value, name in (
            (self.validation_attempt_id, "validation_attempt_id"),
            (self.authorization_state_id, "authorization_state_id"),
            (self.candidate_id, "candidate_id"),
            (self.scope_contract_id, "scope_contract_id"),
        ):
            require_content_id(value, f"authority invalidation {name}")
        _require_digest(
            self.candidate_tree_hash,
            "authority invalidation candidate_tree_hash",
        )
        if self.expected_current_release_id is not None:
            require_content_id(
                self.expected_current_release_id,
                "authority invalidation expected_current_release_id",
            )
            if (
                self.expected_current_release_id.split(":sha256:", 1)[0]
                != "expert-base-release"
            ):
                raise ContractValidationError(
                    "authority invalidation expected CURRENT release uses the wrong namespace"
                )
        if self.observed_current_release_id is not None:
            require_content_id(
                self.observed_current_release_id,
                "authority invalidation observed_current_release_id",
            )
            if (
                self.observed_current_release_id.split(":sha256:", 1)[0]
                != "expert-base-release"
            ):
                raise ContractValidationError(
                    "authority invalidation observed CURRENT release uses the wrong namespace"
                )
        if (
            self.expected_current_release_id is None
            and self.observed_current_release_id is None
        ):
            raise ContractValidationError(
                "authority invalidation cannot bind two absent CURRENT releases"
            )
        if self.observed_current_release_id == self.expected_current_release_id:
            raise ContractValidationError(
                "authority invalidation must observe changed CURRENT authority"
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "authority invalidation exact_dependency_ids",
        )
        required_dependencies = {
            self.validation_attempt_id,
            self.authorization_state_id,
            self.candidate_id,
            self.scope_contract_id,
        }
        if self.expected_current_release_id is not None:
            required_dependencies.add(self.expected_current_release_id)
        if self.observed_current_release_id is not None:
            required_dependencies.add(self.observed_current_release_id)
        if set(self.exact_dependency_ids) != required_dependencies:
            raise MissingReferenceError(
                "authority invalidation dependency closure is not exact"
            )


@dataclass(frozen=True)
class ExpertCandidateValidationState(StrictContract):
    validation_state_id: str
    validation_attempt_id: str | None
    candidate_id: str
    candidate_tree_hash: str
    predecessor_state_id: str | None
    promotion_state: ExpertPromotionState
    accepted_stage_results: tuple[ExpertAcceptedStageResultRef, ...]
    next_stage: ExpertValidationStage | None
    review_assertion_ids: tuple[str, ...]
    terminal_evidence_ids: tuple[str, ...]
    transition_evidence_id: str
    reason: str

    CONTENT_NAMESPACE: ClassVar[str] = "expert-candidate-validation-state"
    IDENTITY_FIELD: ClassVar[str] = "validation_state_id"

    def _validate(self) -> None:
        for value, name in (
            (self.candidate_id, "candidate_id"),
            (self.transition_evidence_id, "transition_evidence_id"),
        ):
            require_content_id(value, name)
        if self.validation_attempt_id is not None:
            require_content_id(self.validation_attempt_id, "validation_attempt_id")
        _require_digest(self.candidate_tree_hash, "candidate_tree_hash")
        if self.predecessor_state_id is not None:
            require_content_id(self.predecessor_state_id, "predecessor_state_id")
            if self.predecessor_state_id == self.validation_state_id:
                raise ContractValidationError("validation state cannot parent itself")
        accepted_stages = tuple(result.stage for result in self.accepted_stage_results)
        accepted_record_ids = tuple(
            result.stage_result_record_id for result in self.accepted_stage_results
        )
        if len(accepted_stages) != len(set(accepted_stages)) or len(
            accepted_record_ids
        ) != len(set(accepted_record_ids)):
            raise ContractValidationError(
                "accepted stage results must have unique stages and records"
            )
        for values, name in (
            (self.review_assertion_ids, "review_assertion_ids"),
            (self.terminal_evidence_ids, "terminal_evidence_ids"),
        ):
            if values:
                _require_sorted_unique(values, name)
                for value in values:
                    require_content_id(value, name)
        _require_text(self.reason, "reason")
        active = self.promotion_state is ExpertPromotionState.VALIDATING
        if active != (self.next_stage is not None):
            raise ContractValidationError(
                "only validating state may name a next evaluator stage"
            )
        if self.validation_attempt_id is None and self.promotion_state not in {
            ExpertPromotionState.INELIGIBLE,
            ExpertPromotionState.REVOKED,
        }:
            raise ContractValidationError(
                "only ineligible or revoked state may omit a validation attempt"
            )
        if self.promotion_state is ExpertPromotionState.INELIGIBLE and (
            self.validation_attempt_id is not None
            or self.accepted_stage_results
            or self.review_assertion_ids
            or not self.terminal_evidence_ids
        ):
            raise ContractValidationError(
                "ineligible state must contain only terminal eligibility evidence"
            )
        if active and self.terminal_evidence_ids:
            raise ContractValidationError(
                "validating state cannot contain terminal evidence"
            )
        terminal = self.promotion_state in {
            ExpertPromotionState.FAILED,
            ExpertPromotionState.DISPUTED,
            ExpertPromotionState.PARETO_RETAINED,
            ExpertPromotionState.APPROVED,
            ExpertPromotionState.RELEASE_USE_BLOCKED,
            ExpertPromotionState.RELEASED,
            ExpertPromotionState.REVOKED,
        }
        if terminal and not self.terminal_evidence_ids:
            raise ContractValidationError(
                "terminal validation state requires terminal evidence"
            )
        if self.promotion_state in {
            ExpertPromotionState.APPROVED,
            ExpertPromotionState.RELEASE_USE_BLOCKED,
            ExpertPromotionState.RELEASED,
        } and (
            not self.accepted_stage_results
            or not self.review_assertion_ids
            or self.predecessor_state_id is None
        ):
            raise ContractValidationError(
                "approved, release-use-blocked, or released state requires "
                "evaluated reviewed lineage"
            )
        if self.promotion_state is ExpertPromotionState.DISPUTED and (
            not self.accepted_stage_results
            or len(self.review_assertion_ids) < 2
            or self.predecessor_state_id is None
        ):
            raise ContractValidationError(
                "disputed state requires evaluated conflicting review lineage"
            )
        if self.promotion_state is ExpertPromotionState.PARETO_RETAINED and (
            not self.accepted_stage_results
            or not self.review_assertion_ids
            or self.predecessor_state_id is None
        ):
            raise ContractValidationError(
                "Pareto-retained state requires evaluated reviewed lineage"
            )
        if (
            self.promotion_state is ExpertPromotionState.FAILED
            and self.predecessor_state_id is None
        ):
            raise ContractValidationError(
                "failed state requires a validation predecessor"
            )


@dataclass(frozen=True)
class ExpertReleaseLineage(StrictContract):
    """Scientific source and remote activation ordering for one release."""

    source_base_release_id: str | None
    activation_predecessor_release_id: str | None

    def _validate(self) -> None:
        for value, name in (
            (self.source_base_release_id, "source_base_release_id"),
            (
                self.activation_predecessor_release_id,
                "activation_predecessor_release_id",
            ),
        ):
            if value is None:
                continue
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != "expert-base-release":
                raise ContractValidationError(f"{name} uses the wrong namespace")
        if (
            self.activation_predecessor_release_id is None
            and self.source_base_release_id is not None
        ):
            raise ContractValidationError(
                "release lineage cannot consume a source base without an "
                "activation predecessor"
            )


@dataclass(frozen=True)
class ExpertBaseReleaseManifest(StrictContract):
    release_id: str
    scope_contract_id: str
    scope_id: str
    lineage: ExpertReleaseLineage
    candidate_id: str
    candidate_commit_record_id: str
    candidate_tree_ref: str
    candidate_tree_hash: str
    candidate_derivation_ref: str
    candidate_validation_context_ref: str
    candidate_patch_ref: str
    candidate_sanitation_report_id: str
    candidate_ancestor_ids: tuple[str, ...]
    candidate_source_dependency_ids: tuple[str, ...]
    candidate_consumed_expert_release_ids: tuple[str, ...]
    repository_map_ref: str
    module_contract_refs: tuple[str, ...]
    module_versions: Mapping[str, str]
    semantic_book_digest: str
    validation_attempt_id: str
    approval_transition_id: str
    approval_state_id: str
    publication_eligibility_result_id: str
    release_matrix_stage_result_id: str
    release_matrix_report_id: str
    promotion_decision_id: str
    approval_assertion_ids: tuple[str, ...]
    validation_policy_id: str
    configuration_fingerprint: str
    source_archive_ref: str
    evidence_archive_ref: str
    evidence_manifest_ref: str
    test_matrix_summary_ref: str
    evidence_dependency_ids: tuple[str, ...]
    consumed_dependency_ids: tuple[str, ...]
    control_dependency_ids: tuple[str, ...]
    checksums: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-base-release"
    IDENTITY_FIELD: ClassVar[str] = "release_id"

    def _validate(self) -> None:
        namespaced_ids = (
            (self.scope_contract_id, "expert-scope-contract", "scope_contract_id"),
            (self.candidate_id, "expert-candidate", "candidate_id"),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "candidate_commit_record_id",
            ),
            (self.candidate_tree_ref, "expert-source-tree", "candidate_tree_ref"),
            (
                self.candidate_validation_context_ref,
                "expert-candidate-validation-context",
                "candidate_validation_context_ref",
            ),
            (
                self.candidate_patch_ref,
                (
                    "expert-recovery-restore-patch"
                    if self.candidate_derivation_ref.split(":sha256:", 1)[0]
                    == "expert-deterministic-recovery-restore-derivation"
                    else "expert-candidate-patch"
                ),
                "candidate_patch_ref",
            ),
            (
                self.candidate_sanitation_report_id,
                "expert-candidate-sanitation",
                "candidate_sanitation_report_id",
            ),
            (
                self.repository_map_ref,
                "expert-repository-map",
                "repository_map_ref",
            ),
            (
                self.validation_attempt_id,
                "expert-validation-attempt",
                "validation_attempt_id",
            ),
            (
                self.approval_transition_id,
                "expert-validation-transition",
                "approval_transition_id",
            ),
            (
                self.approval_state_id,
                "expert-candidate-validation-state",
                "approval_state_id",
            ),
            (
                self.publication_eligibility_result_id,
                "expert-publication-eligibility-stage-result",
                "publication_eligibility_result_id",
            ),
            (
                self.release_matrix_stage_result_id,
                "expert-release-matrix-stage-result",
                "release_matrix_stage_result_id",
            ),
            (
                self.release_matrix_report_id,
                "expert-release-matrix-report",
                "release_matrix_report_id",
            ),
            (
                self.promotion_decision_id,
                "expert-release-matrix-promotion-decision",
                "promotion_decision_id",
            ),
            (
                self.validation_policy_id,
                "expert-validation-policy",
                "validation_policy_id",
            ),
            (
                self.evidence_manifest_ref,
                "expert-release-evidence-manifest",
                "evidence_manifest_ref",
            ),
            (
                self.test_matrix_summary_ref,
                "expert-release-matrix-summary",
                "test_matrix_summary_ref",
            ),
        )
        for value, namespace, name in namespaced_ids:
            require_content_id(value, name)
            if value.split(":sha256:", 1)[0] != namespace:
                raise ContractValidationError(f"{name} uses the wrong namespace")
        require_content_id(self.candidate_derivation_ref, "candidate_derivation_ref")
        if self.candidate_derivation_ref.split(":sha256:", 1)[0] not in {
            "expert-agent-proposal-derivation",
            "expert-deterministic-composition-derivation",
            "expert-deterministic-recovery-restore-derivation",
        }:
            raise ContractValidationError(
                "candidate_derivation_ref uses the wrong namespace"
            )
        require_identifier(self.scope_id, "scope_id")
        if (
            self.lineage.source_base_release_id
            != self.lineage.activation_predecessor_release_id
        ):
            raise ContractValidationError(
                "ordinary expert release requires identical source base and "
                "activation predecessor"
            )
        _require_digest(self.candidate_tree_hash, "candidate_tree_hash")
        for values, name, required in (
            (self.candidate_ancestor_ids, "candidate_ancestor_ids", False),
            (
                self.candidate_source_dependency_ids,
                "candidate_source_dependency_ids",
                True,
            ),
            (
                self.candidate_consumed_expert_release_ids,
                "candidate_consumed_expert_release_ids",
                False,
            ),
            (self.module_contract_refs, "module_contract_refs", True),
            (self.approval_assertion_ids, "approval_assertion_ids", True),
        ):
            if required and not values:
                raise ContractValidationError(f"{name} must not be empty")
            if values:
                _require_sorted_unique(values, name)
                for value in values:
                    require_content_id(value, name)
        if any(
            value.split(":sha256:", 1)[0] != "expert-candidate"
            for value in self.candidate_ancestor_ids
        ):
            raise ContractValidationError(
                "candidate_ancestor_ids use the wrong namespace"
            )
        if any(
            value.split(":sha256:", 1)[0] != "expert-base-release"
            for value in self.candidate_consumed_expert_release_ids
        ):
            raise ContractValidationError(
                "candidate_consumed_expert_release_ids use the wrong namespace"
            )
        if self.lineage.source_base_release_id is not None and (
            self.lineage.source_base_release_id
            not in self.candidate_consumed_expert_release_ids
        ):
            raise MissingReferenceError(
                "release candidate consumption omits its source base"
            )
        if any(
            value.split(":sha256:", 1)[0] != "expert-module-contract"
            for value in self.module_contract_refs
        ):
            raise ContractValidationError(
                "module_contract_refs use the wrong namespace"
            )
        if not self.module_versions:
            raise ContractValidationError("module_versions must not be empty")
        for module_id, version in self.module_versions.items():
            require_identifier(module_id, "module_versions key")
            require_identifier(version, "module_versions value")
        _require_digest(self.semantic_book_digest, "semantic_book_digest")
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        for archive_ref, name in (
            (self.source_archive_ref, "source_archive_ref"),
            (self.evidence_archive_ref, "evidence_archive_ref"),
        ):
            _require_text(archive_ref, name)
            archive_path = PurePosixPath(archive_ref)
            if (
                archive_path.is_absolute()
                or len(archive_path.parts) != 1
                or archive_path.as_posix() != archive_ref
                or not archive_ref.endswith((".tar", ".tar.zst"))
            ):
                raise ContractValidationError(
                    f"{name} must name one supported release asset"
                )
        if self.source_archive_ref == self.evidence_archive_ref:
            raise ContractValidationError("release archive names must differ")
        _require_sorted_unique(self.evidence_dependency_ids, "evidence_dependency_ids")
        for value in self.evidence_dependency_ids:
            require_content_id(value, "evidence_dependency_ids")
        _require_sorted_unique(
            self.consumed_dependency_ids,
            "consumed_dependency_ids",
        )
        if not set(self.candidate_consumed_expert_release_ids).issubset(
            self.consumed_dependency_ids
        ):
            raise MissingReferenceError(
                "release consumed dependencies omit candidate release inputs"
            )
        if self.control_dependency_ids != tuple(
            sorted(set(self.control_dependency_ids))
        ):
            raise ContractValidationError(
                "control_dependency_ids must be sorted and unique"
            )
        for value in (
            *self.consumed_dependency_ids,
            *self.control_dependency_ids,
        ):
            require_content_id(value, "expert release categorized dependency")
        if set(self.consumed_dependency_ids) & set(self.control_dependency_ids):
            raise ContractValidationError(
                "expert release consumed and control dependencies overlap"
            )
        required_dependencies = {
            self.scope_contract_id,
            self.candidate_id,
            self.candidate_commit_record_id,
            self.candidate_tree_ref,
            self.candidate_derivation_ref,
            self.candidate_validation_context_ref,
            self.candidate_patch_ref,
            self.candidate_sanitation_report_id,
            *self.candidate_ancestor_ids,
            *self.candidate_source_dependency_ids,
            *self.candidate_consumed_expert_release_ids,
            self.repository_map_ref,
            *self.module_contract_refs,
            self.validation_attempt_id,
            self.approval_transition_id,
            self.approval_state_id,
            self.publication_eligibility_result_id,
            self.release_matrix_stage_result_id,
            self.release_matrix_report_id,
            self.promotion_decision_id,
            *self.approval_assertion_ids,
            self.validation_policy_id,
            self.evidence_manifest_ref,
            self.test_matrix_summary_ref,
        }
        if self.lineage.source_base_release_id is not None:
            required_dependencies.add(self.lineage.source_base_release_id)
        required_dependencies.update(self.evidence_dependency_ids)
        if set(self.consumed_dependency_ids) != required_dependencies:
            raise MissingReferenceError(
                "expert release consumed dependency closure is not exact"
            )
        if self.control_dependency_ids:
            raise ContractValidationError(
                "ordinary expert release cannot carry control dependencies"
            )
        _require_checksum_mapping(self.checksums, "checksums")
        if not {
            self.source_archive_ref,
            self.evidence_archive_ref,
        }.issubset(self.checksums):
            raise MissingReferenceError("expert release archive checksum is missing")


@dataclass(frozen=True)
class TaskEvaluatorMetricComparisonBinding(StrictContract):
    evaluator_fingerprint: str
    metric_name: str
    objective_direction: ObjectiveDirection
    comparison_dimension_id: str
    comparison_scale: float

    def _validate(self) -> None:
        _require_digest(
            self.evaluator_fingerprint,
            "task evaluator metric evaluator_fingerprint",
        )
        require_identifier(self.metric_name, "task evaluator metric_name")
        require_identifier(
            self.comparison_dimension_id,
            "task evaluator comparison_dimension_id",
        )
        if (
            type(self.comparison_scale) is not float
            or not math.isfinite(self.comparison_scale)
            or self.comparison_scale <= 0.0
        ):
            raise ContractValidationError(
                "task evaluator comparison_scale must be a finite positive float"
            )


@dataclass(frozen=True)
class TaskEvaluatorBinding(StrictContract):
    protocol_version: str
    executable_path: str
    supported_evaluator_fingerprints: tuple[str, ...]
    metric_comparison_bindings: tuple[TaskEvaluatorMetricComparisonBinding, ...]

    def _validate(self) -> None:
        require_identifier(self.protocol_version, "task evaluator protocol_version")
        _require_relative_path(
            self.executable_path,
            "task evaluator executable_path",
        )
        if not self.supported_evaluator_fingerprints or (
            self.supported_evaluator_fingerprints
            != tuple(sorted(set(self.supported_evaluator_fingerprints)))
        ):
            raise ContractValidationError(
                "task evaluator fingerprints must be non-empty, sorted, and unique"
            )
        for fingerprint in self.supported_evaluator_fingerprints:
            _require_digest(fingerprint, "task evaluator supported fingerprint")
        comparison_keys = tuple(
            (binding.evaluator_fingerprint, binding.metric_name)
            for binding in self.metric_comparison_bindings
        )
        if (
            not comparison_keys
            or comparison_keys != tuple(sorted(set(comparison_keys)))
            or {
                binding.evaluator_fingerprint
                for binding in self.metric_comparison_bindings
            }
            != set(self.supported_evaluator_fingerprints)
        ):
            raise ContractValidationError(
                "task evaluator metric comparison bindings must be sorted, unique, "
                "and cover every supported evaluator fingerprint"
            )


@dataclass(frozen=True)
class TaskAdapterContextBinding(StrictContract):
    consumed_dimension_ids: tuple[str, ...]

    def _validate(self) -> None:
        if self.consumed_dimension_ids != tuple(
            sorted(set(self.consumed_dimension_ids))
        ):
            raise ContractValidationError(
                "task adapter consumed_dimension_ids must be sorted and unique"
            )
        for dimension_id in self.consumed_dimension_ids:
            require_identifier(
                dimension_id,
                "task adapter consumed dimension",
            )


@dataclass(frozen=True)
class TaskAdapterRuntimeContract(StrictContract):
    runtime_protocol_version: str
    image_repository: str
    image_manifest_digest: str
    image_config_digest: str
    dependency_lock_path: str
    dependency_lock_digest: str
    operating_system: str
    architecture: str
    architecture_variant: str | None
    environment: Mapping[str, str]

    def _validate(self) -> None:
        require_identifier(
            self.runtime_protocol_version,
            "task adapter runtime protocol version",
        )
        if (
            re.fullmatch(
                r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[1-9][0-9]*)?"
                r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+",
                self.image_repository,
            )
            is None
        ):
            raise ContractValidationError(
                "task adapter runtime image_repository must be a normalized "
                "registry-qualified OCI repository"
            )
        registry = self.image_repository.split("/", 1)[0]
        if registry != "localhost" and "." not in registry and ":" not in registry:
            raise ContractValidationError(
                "task adapter runtime image_repository must name an explicit registry"
            )
        for digest, name in (
            (self.image_manifest_digest, "image_manifest_digest"),
            (self.image_config_digest, "image_config_digest"),
        ):
            _require_digest(digest, f"task adapter runtime {name}")
        _require_relative_path(
            self.dependency_lock_path,
            "task adapter runtime dependency_lock_path",
        )
        _require_digest(
            self.dependency_lock_digest,
            "task adapter runtime dependency_lock_digest",
        )
        require_identifier(
            self.operating_system,
            "task adapter runtime operating_system",
        )
        require_identifier(self.architecture, "task adapter runtime architecture")
        if self.architecture_variant is not None:
            require_identifier(
                self.architecture_variant,
                "task adapter runtime architecture_variant",
            )
        environment_keys = tuple(self.environment)
        if environment_keys != tuple(sorted(set(environment_keys))):
            raise ContractValidationError(
                "task adapter runtime environment must be key-sorted and unique"
            )
        for key, value in self.environment.items():
            if (
                _RUNTIME_ENVIRONMENT_KEY_PATTERN.fullmatch(key) is None
                or _SECRET_ENVIRONMENT_KEY_PATTERN.search(key) is not None
                or (value != "" and not value.isprintable())
            ):
                raise ContractValidationError(
                    "task adapter runtime environment must be fixed and non-secret"
                )
        if not self.environment.get("PATH"):
            raise ContractValidationError(
                "task adapter runtime environment must declare a non-empty PATH"
            )

    @property
    def image_reference(self) -> str:
        return f"{self.image_repository}@{self.image_manifest_digest}"


@dataclass(frozen=True)
class TaskAdapterReleaseMatrixIndependenceGroup(StrictContract):
    independence_group_id: str
    lineage_root_digests: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-adapter-release-matrix-independence-group"
    IDENTITY_FIELD: ClassVar[str] = "independence_group_id"

    def _validate(self) -> None:
        if not self.lineage_root_digests or self.lineage_root_digests != tuple(
            sorted(set(self.lineage_root_digests))
        ):
            raise ContractValidationError(
                "release matrix lineage roots must be non-empty, sorted, and unique"
            )
        for digest in self.lineage_root_digests:
            _require_digest(digest, "release matrix lineage root")


@dataclass(frozen=True)
class TaskAdapterReleaseMatrixStartingArtifact(StrictContract):
    starting_artifact_content_id: str
    starting_artifact_ref: str
    mount_path: str
    package_source_root: str
    materialized_tree_hash: str
    source_files: tuple[SourceFileDescriptor, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-adapter-release-matrix-starting-artifact"
    IDENTITY_FIELD: ClassVar[str] = "starting_artifact_content_id"

    def _validate(self) -> None:
        _require_text(
            self.starting_artifact_ref,
            "release matrix starting_artifact_ref",
        )
        _require_relative_path(self.mount_path, "release matrix artifact mount_path")
        if self.mount_path == ".":
            raise ContractValidationError(
                "release matrix artifact mount_path cannot be the workspace root"
            )
        _require_relative_path(
            self.package_source_root,
            "release matrix artifact package_source_root",
        )
        source_root = PurePosixPath(self.package_source_root)
        if (
            len(source_root.parts) < 2
            or source_root.parts[0] != "release_matrix_assets"
        ):
            raise ContractValidationError(
                "release matrix artifact source root must use its reserved subtree"
            )
        paths = tuple(descriptor.relative_path for descriptor in self.source_files)
        if not paths or paths != tuple(sorted(set(paths))):
            raise ContractValidationError(
                "release matrix artifact files must be non-empty, sorted, and unique"
            )
        source_paths = tuple(PurePosixPath(path) for path in paths)
        if any(
            source_path in other_path.parents
            for position, source_path in enumerate(source_paths)
            for other_path in source_paths[position + 1 :]
        ):
            raise ContractValidationError(
                "release matrix artifact files contain a path collision"
            )
        expected_tree_hash = source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
                for descriptor in self.source_files
            }
        )
        if self.materialized_tree_hash != expected_tree_hash:
            raise ContractValidationError(
                "release matrix artifact tree differs from its file closure"
            )


@dataclass(frozen=True)
class TaskAdapterReleaseMatrixCase(StrictContract):
    release_matrix_case_id: str
    task_context_binding: TaskContextBinding
    independence_group: TaskAdapterReleaseMatrixIndependenceGroup
    evaluation_fingerprints: tuple[EvaluationFingerprint, ...]
    starting_artifacts: tuple[TaskAdapterReleaseMatrixStartingArtifact, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "task-adapter-release-matrix-case"
    IDENTITY_FIELD: ClassVar[str] = "release_matrix_case_id"

    def _validate(self) -> None:
        fingerprint_ids = tuple(
            fingerprint.evaluation_fingerprint_id
            for fingerprint in self.evaluation_fingerprints
        )
        if not fingerprint_ids or fingerprint_ids != tuple(
            sorted(set(fingerprint_ids))
        ):
            raise ContractValidationError(
                "release matrix case fingerprints must be non-empty, sorted, and unique"
            )
        artifact_ids = tuple(
            artifact.starting_artifact_content_id
            for artifact in self.starting_artifacts
        )
        if artifact_ids != tuple(sorted(set(artifact_ids))):
            raise ContractValidationError(
                "release matrix case artifacts must be sorted and unique"
            )
        artifact_refs = tuple(
            artifact.starting_artifact_ref for artifact in self.starting_artifacts
        )
        if set(artifact_refs) != set(self.task_context_binding.starting_artifact_refs):
            raise ContractValidationError(
                "release matrix case artifacts differ from its task context"
            )
        if len(artifact_refs) != len(set(artifact_refs)):
            raise ContractValidationError(
                "release matrix case artifact refs must be unique"
            )
        mount_paths = tuple(
            PurePosixPath(artifact.mount_path) for artifact in self.starting_artifacts
        )
        if len(mount_paths) != len(set(mount_paths)) or any(
            left in right.parents or right in left.parents
            for position, left in enumerate(mount_paths)
            for right in mount_paths[position + 1 :]
        ):
            raise ContractValidationError("release matrix case artifact mounts overlap")

    @property
    def evaluation_fingerprint_ids(self) -> tuple[str, ...]:
        return tuple(
            fingerprint.evaluation_fingerprint_id
            for fingerprint in self.evaluation_fingerprints
        )

    @property
    def starting_artifact_ids(self) -> tuple[str, ...]:
        return tuple(
            artifact.starting_artifact_content_id
            for artifact in self.starting_artifacts
        )


@dataclass(frozen=True)
class TaskAdapterManifest(StrictContract):
    task_adapter_manifest_id: str
    task_adapter_id: str
    scope_contract_id: str
    task_family_id: str
    publisher_attestation: Mapping[str, Any]
    task_evaluator: TaskEvaluatorBinding
    context_binding: TaskAdapterContextBinding
    release_matrix_cases: tuple[TaskAdapterReleaseMatrixCase, ...]
    source_tree_ref: str
    tree_hash: str
    runtime: TaskAdapterRuntimeContract
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
        if self.task_evaluator.executable_path == self.runtime.dependency_lock_path:
            raise ContractValidationError(
                "task adapter evaluator and dependency lock paths must differ"
            )
        reserved_asset_root = PurePosixPath("release_matrix_assets")
        if any(
            PurePosixPath(path) == reserved_asset_root
            or reserved_asset_root in PurePosixPath(path).parents
            for path in (
                self.task_evaluator.executable_path,
                self.runtime.dependency_lock_path,
            )
        ):
            raise ContractValidationError(
                "task adapter runtime files cannot use the release matrix asset subtree"
            )
        case_ids = tuple(
            case.release_matrix_case_id for case in self.release_matrix_cases
        )
        if not case_ids or case_ids != tuple(sorted(set(case_ids))):
            raise ContractValidationError(
                "task adapter release matrix cases must be non-empty, sorted, and unique"
            )
        comparison_bindings = {
            (
                binding.evaluator_fingerprint,
                binding.metric_name,
            ): binding
            for binding in self.task_evaluator.metric_comparison_bindings
        }
        artifact_roots: dict[str, str] = {}
        scientific_projection_groups: dict[tuple[object, ...], str] = {}
        for case in self.release_matrix_cases:
            context = case.task_context_binding
            if (
                context.scope_contract_id != self.scope_contract_id
                or context.task_family_id != self.task_family_id
                or context.task_adapter_id != self.task_adapter_id
                or not set(self.context_binding.consumed_dimension_ids).issubset(
                    context.transfer_dimensions
                )
            ):
                raise ContractValidationError(
                    "task adapter release matrix case differs from its manifest binding"
                )
            scientific_projection = (
                context.scope_contract_id,
                context.scope_id,
                context.task_family_id,
                context.task_adapter_id,
                context.input_contract_fingerprint,
                context.target_contract_fingerprint,
                canonical_json_bytes(context.transfer_dimensions),
                tuple(
                    sorted(
                        {
                            (
                                fingerprint.benchmark_id,
                                fingerprint.dataset_version,
                                fingerprint.split_version,
                            )
                            for fingerprint in case.evaluation_fingerprints
                        }
                    )
                ),
                tuple(
                    sorted(
                        artifact.materialized_tree_hash
                        for artifact in case.starting_artifacts
                    )
                ),
            )
            prior_group_id = scientific_projection_groups.setdefault(
                scientific_projection,
                case.independence_group.independence_group_id,
            )
            if prior_group_id != case.independence_group.independence_group_id:
                raise ContractValidationError(
                    "identical release matrix cases cannot claim independent groups"
                )
            for fingerprint in case.evaluation_fingerprints:
                binding = comparison_bindings.get(
                    (
                        fingerprint.evaluator_fingerprint,
                        fingerprint.metric_name,
                    )
                )
                if (
                    binding is None
                    or binding.objective_direction
                    is not fingerprint.objective_direction
                ):
                    raise ContractValidationError(
                        "task adapter release matrix fingerprint lacks exact metric authority"
                    )
            for artifact in case.starting_artifacts:
                prior_artifact_id = artifact_roots.setdefault(
                    artifact.package_source_root,
                    artifact.starting_artifact_content_id,
                )
                if prior_artifact_id != artifact.starting_artifact_content_id:
                    raise ContractValidationError(
                        "release matrix asset root names multiple artifacts"
                    )
        roots = tuple(PurePosixPath(root) for root in artifact_roots)
        if any(
            left in right.parents or right in left.parents
            for position, left in enumerate(roots)
            for right in roots[position + 1 :]
        ):
            raise ContractValidationError("release matrix asset roots overlap")
        _require_text(self.source_tree_ref, "source_tree_ref")
        _require_digest(self.tree_hash, "tree_hash")
        require_content_id(self.sanitation_report_id, "sanitation_report_id")
        _require_sorted_unique(self.validation_refs, "validation_refs")


@dataclass(frozen=True)
class SecurityDenylistEvidence(StrictContract):
    evidence_id: str
    evidence_kind: str
    summary: str
    source_ids: tuple[str, ...]
    recorded_at: str

    CONTENT_NAMESPACE: ClassVar[str] = "security-denylist-evidence"
    IDENTITY_FIELD: ClassVar[str] = "evidence_id"

    def _validate(self) -> None:
        require_identifier(self.evidence_kind, "security denylist evidence_kind")
        _require_text(self.summary, "security denylist evidence summary")
        if not self.source_ids:
            raise ContractValidationError(
                "security denylist evidence requires source identities"
            )
        _require_sorted_unique(self.source_ids, "security denylist evidence sources")
        for source_id in self.source_ids:
            require_content_id(source_id, "security denylist evidence source")
        normalize_utc_timestamp(self.recorded_at, "security denylist evidence time")


@dataclass(frozen=True)
class SecurityDenylistEvidenceBundle(StrictContract):
    evidence_bundle_id: str
    evidence: tuple[SecurityDenylistEvidence, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "security-denylist-evidence-bundle"
    IDENTITY_FIELD: ClassVar[str] = "evidence_bundle_id"

    def _validate(self) -> None:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ContractValidationError(
                "security denylist evidence must be sorted and unique"
            )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {source_id for item in self.evidence for source_id in item.source_ids}
            )
        )


@dataclass(frozen=True)
class SecurityDenylistRevocation(StrictContract):
    revocation_id: str
    subject_id: str
    kind: SecurityDenylistKind
    reason_code: str
    evidence_ids: tuple[str, ...]
    recorded_at: str

    CONTENT_NAMESPACE: ClassVar[str] = "security-denylist-revocation"
    IDENTITY_FIELD: ClassVar[str] = "revocation_id"

    def _validate(self) -> None:
        require_content_id(self.subject_id, "security denylist subject_id")
        require_identifier(self.reason_code, "security denylist reason_code")
        if not self.evidence_ids:
            raise ContractValidationError(
                "security denylist revocation requires evidence"
            )
        _require_sorted_unique(
            self.evidence_ids,
            "security denylist evidence_ids",
        )
        for evidence_id in self.evidence_ids:
            require_content_id(evidence_id, "security denylist evidence_id")
        normalize_utc_timestamp(self.recorded_at, "security denylist recorded_at")


@dataclass(frozen=True)
class SecurityDenylistSnapshot(StrictContract):
    snapshot_id: str
    schema_version: str
    policy_version: str
    scope_id: str
    scope_contract_id: str
    scope_repository_binding_hash: str
    generation: int
    predecessor_snapshot_id: str | None
    evidence_bundle_id: str
    evidence_source_ids: tuple[str, ...]
    revocations: tuple[SecurityDenylistRevocation, ...]
    exact_dependency_ids: tuple[str, ...]
    checksums: Mapping[str, str]

    CONTENT_NAMESPACE: ClassVar[str] = "security-denylist-snapshot"
    IDENTITY_FIELD: ClassVar[str] = "snapshot_id"

    def _validate(self) -> None:
        if self.schema_version != SECURITY_DENYLIST_SCHEMA_VERSION:
            raise ContractValidationError(
                "security denylist schema version is unsupported"
            )
        if self.policy_version != SECURITY_DENYLIST_POLICY_VERSION:
            raise ContractValidationError(
                "security denylist policy version is unsupported"
            )
        require_identifier(self.scope_id, "security denylist scope_id")
        require_content_id(
            self.scope_contract_id,
            "security denylist scope_contract_id",
        )
        _require_digest(
            self.scope_repository_binding_hash,
            "security denylist scope_repository_binding_hash",
        )
        if type(self.generation) is not int or self.generation < 0:
            raise ContractValidationError(
                "security denylist generation must be non-negative"
            )
        if (self.predecessor_snapshot_id is None) != (self.generation == 0):
            raise ContractValidationError(
                "only security denylist generation zero may omit its predecessor"
            )
        if self.generation == 0 and self.revocations:
            raise ContractValidationError(
                "security denylist generation zero must be empty"
            )
        if self.predecessor_snapshot_id is not None:
            require_content_id(
                self.predecessor_snapshot_id,
                "security denylist predecessor_snapshot_id",
            )
            if self.predecessor_snapshot_id.split(":sha256:", 1)[0] != (
                "security-denylist-snapshot"
            ):
                raise ContractValidationError(
                    "security denylist predecessor uses the wrong namespace"
                )
        require_content_id(
            self.evidence_bundle_id,
            "security denylist evidence_bundle_id",
        )
        if self.evidence_bundle_id.split(":sha256:", 1)[0] != (
            "security-denylist-evidence-bundle"
        ):
            raise ContractValidationError(
                "security denylist evidence bundle uses the wrong namespace"
            )
        if self.evidence_source_ids != tuple(sorted(set(self.evidence_source_ids))):
            raise ContractValidationError(
                "security denylist evidence_source_ids must be sorted and unique"
            )
        for source_id in self.evidence_source_ids:
            require_content_id(source_id, "security denylist evidence source")
        revocation_ids = tuple(
            revocation.revocation_id for revocation in self.revocations
        )
        if revocation_ids != tuple(sorted(set(revocation_ids))):
            raise ContractValidationError(
                "security denylist revocations must be sorted and unique"
            )
        _require_sorted_unique(
            self.exact_dependency_ids,
            "security denylist exact_dependency_ids",
        )
        for dependency_id in self.exact_dependency_ids:
            require_content_id(dependency_id, "security denylist dependency")
        required_dependencies = {
            self.scope_contract_id,
            self.evidence_bundle_id,
            *self.evidence_source_ids,
            *(revocation.revocation_id for revocation in self.revocations),
            *(revocation.subject_id for revocation in self.revocations),
            *(
                evidence_id
                for revocation in self.revocations
                for evidence_id in revocation.evidence_ids
            ),
        }
        if self.predecessor_snapshot_id is not None:
            required_dependencies.add(self.predecessor_snapshot_id)
        if required_dependencies != set(self.exact_dependency_ids):
            raise MissingReferenceError(
                "security denylist dependency closure is not exact"
            )
        _require_checksum_mapping(self.checksums, "security denylist checksums")
        if set(self.checksums) != {SECURITY_DENYLIST_EVIDENCE_FILENAME}:
            raise ContractValidationError(
                "security denylist checksums must bind only its evidence bundle"
            )

    def validate_evidence_bundle(
        self,
        bundle: SecurityDenylistEvidenceBundle,
    ) -> None:
        if bundle.evidence_bundle_id != self.evidence_bundle_id:
            raise ContractValidationError(
                "security denylist evidence bundle identity differs"
            )
        referenced_evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for revocation in self.revocations
                    for evidence_id in revocation.evidence_ids
                }
            )
        )
        if (
            bundle.evidence_ids != referenced_evidence_ids
            or bundle.source_ids != self.evidence_source_ids
            or self.checksums[SECURITY_DENYLIST_EVIDENCE_FILENAME]
            != tree_or_blob_digest(bundle.to_json_bytes())
        ):
            raise ContractValidationError(
                "security denylist evidence bundle is not its exact closure"
            )


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
    security_denylist_snapshot_id: str
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
            "security_denylist_snapshot_id",
        ):
            require_content_id(getattr(self, name), name)
        if self.security_denylist_snapshot_id.split(":sha256:", 1)[0] != (
            "security-denylist-snapshot"
        ):
            raise ContractValidationError(
                "launch security denylist snapshot uses the wrong namespace"
            )
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
    security_denylist_snapshot_id: str
    security_denylist_generation: int
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
            "security_denylist_snapshot_id",
        ):
            require_content_id(getattr(self, name), name)
        if self.security_denylist_snapshot_id.split(":sha256:", 1)[0] != (
            "security-denylist-snapshot"
        ):
            raise ContractValidationError(
                "bootstrap security denylist snapshot uses the wrong namespace"
            )
        _require_digest(self.launch_request_hash, "launch_request_hash")
        for name in ("scope_id", "task_family_id", "task_adapter_id"):
            require_identifier(getattr(self, name), name)
        if (
            type(self.security_denylist_generation) is not int
            or self.security_denylist_generation < 0
        ):
            raise ContractValidationError(
                "bootstrap security denylist generation must be non-negative"
            )
        _require_digest(self.workspace_tree_hash, "workspace_tree_hash")
        normalize_utc_timestamp(self.created_at, "created_at")
