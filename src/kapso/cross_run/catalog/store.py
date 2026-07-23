"""Immutable fact storage and atomic generation publication for the catalog."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    parse_json_bytes,
    require_content_id,
    require_identifier,
)
from kapso.cross_run.contracts import (
    CatalogEntryState,
    ContractValidationError,
    MissingReferenceError,
    StrictContract,
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPERATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_CONTENT_ID_SEPARATOR = ":sha256:"
EXPERT_RELEASE_USE_REVOCATION_NAMESPACE = "expert-release-use-revocation"
_PROTECTED_FACT_NAMESPACES = (EXPERT_RELEASE_USE_REVOCATION_NAMESPACE,)


class CatalogStoreError(RuntimeError):
    """Base class for catalog persistence failures."""


class CatalogLayoutError(CatalogStoreError):
    """The catalog filesystem layout is unsafe or invalid."""


class CatalogCorruptionError(CatalogStoreError):
    """Persisted catalog bytes violate their declared identity or lineage."""


class CatalogNotInitializedError(CatalogStoreError):
    """No current catalog generation has been initialized."""


class CatalogCompareAndSwapError(CatalogStoreError):
    """The expected generation is no longer current."""

    def __init__(
        self,
        *,
        expected_generation_id: str,
        expected_generation_number: int,
        actual_generation_id: str,
        actual_generation_number: int,
    ) -> None:
        self.expected_generation_id = expected_generation_id
        self.expected_generation_number = expected_generation_number
        self.actual_generation_id = actual_generation_id
        self.actual_generation_number = actual_generation_number
        super().__init__(
            "catalog compare-and-swap conflict: "
            f"expected {expected_generation_number}/{expected_generation_id}, "
            f"actual {actual_generation_number}/{actual_generation_id}"
        )


class CatalogOperationConflictError(CatalogStoreError):
    """One operation identifier was reused for different immutable input."""


class CatalogClosureError(CatalogStoreError):
    """A generation or input delta has an incomplete object closure."""


class CatalogReducerError(CatalogStoreError):
    """A reducer returned a structurally impossible projection."""


class CatalogProtectedNamespaceError(CatalogStoreError):
    """A reserved fact namespace was submitted without its bound authority."""


def _require_digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ContractValidationError(f"{name} must be a sha256 digest")


def _require_operation_id(value: Any) -> None:
    if not isinstance(value, str) or _OPERATION_PATTERN.fullmatch(value) is None:
        raise ContractValidationError(
            "operation_id must be a filesystem-safe qualified identifier"
        )


def _require_sorted_content_ids(
    values: tuple[str, ...], name: str, *, required: bool = False
) -> None:
    if required and not values:
        raise ContractValidationError(f"{name} must not be empty")
    if values != tuple(sorted(set(values))):
        raise ContractValidationError(f"{name} must be sorted and unique")
    for value in values:
        require_content_id(value, name)


def _freeze_reference_mapping(
    values: Mapping[str, str],
    name: str,
    *,
    content_keys: bool,
) -> Mapping[str, str]:
    if not isinstance(values, MappingABC):
        raise ContractValidationError(f"{name} must be an object")
    frozen: dict[str, str] = {}
    for key, value in values.items():
        if content_keys:
            require_content_id(key, f"{name} key")
        else:
            require_identifier(key, f"{name} key")
        require_content_id(value, f"{name}[{key}]")
        frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class CatalogInputDelta(StrictContract):
    """One idempotent operation's additive scientific facts and proof closure."""

    input_delta_id: str
    scope_contract_id: str
    operation_id: str
    configuration_fingerprint: str
    added_object_ids: tuple[str, ...]
    dependency_closure_ids: tuple[str, ...]

    CONTENT_NAMESPACE = "catalog-input-delta"
    IDENTITY_FIELD = "input_delta_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        _require_operation_id(self.operation_id)
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        _require_sorted_content_ids(self.added_object_ids, "added_object_ids")
        _require_sorted_content_ids(
            self.dependency_closure_ids, "dependency_closure_ids"
        )
        if not set(self.added_object_ids).issubset(self.dependency_closure_ids):
            raise MissingReferenceError(
                "dependency_closure_ids must contain every added object"
            )


@dataclass(frozen=True)
class CatalogGenerationManifest(StrictContract):
    """Complete immutable reduction of all facts through one generation."""

    catalog_generation_id: str
    scope_contract_id: str
    generation_number: int
    parent_generation_id: str | None
    configuration_fingerprint: str
    fact_object_ids: tuple[str, ...]
    derived_object_ids: tuple[str, ...]
    applied_input_delta_ids: tuple[str, ...]
    bundle_frontier: Mapping[str, str]
    active_entry_state_ids: Mapping[str, str]

    CONTENT_NAMESPACE = "catalog-generation"
    IDENTITY_FIELD = "catalog_generation_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        if self.generation_number < 0:
            raise ContractValidationError("generation_number must be non-negative")
        if (self.generation_number == 0) != (self.parent_generation_id is None):
            raise ContractValidationError("only generation zero may omit its parent")
        if self.parent_generation_id is not None:
            require_content_id(self.parent_generation_id, "parent_generation_id")
            if self.parent_generation_id == self.catalog_generation_id:
                raise ContractValidationError("generation cannot parent itself")
        _require_digest(self.configuration_fingerprint, "configuration_fingerprint")
        _require_sorted_content_ids(self.fact_object_ids, "fact_object_ids")
        _require_sorted_content_ids(self.derived_object_ids, "derived_object_ids")
        if set(self.fact_object_ids) & set(self.derived_object_ids):
            raise ContractValidationError(
                "source facts and reducer-derived objects must be disjoint"
            )
        _require_sorted_content_ids(
            self.applied_input_delta_ids, "applied_input_delta_ids"
        )
        object.__setattr__(
            self,
            "bundle_frontier",
            _freeze_reference_mapping(
                self.bundle_frontier,
                "bundle_frontier",
                content_keys=False,
            ),
        )
        object.__setattr__(
            self,
            "active_entry_state_ids",
            _freeze_reference_mapping(
                self.active_entry_state_ids,
                "active_entry_state_ids",
                content_keys=True,
            ),
        )
        source_references = set(self.bundle_frontier.values())
        source_references.update(self.active_entry_state_ids)
        if not source_references.issubset(self.fact_object_ids):
            raise MissingReferenceError(
                "generation frontier and active subjects must be source facts"
            )
        if not set(self.active_entry_state_ids.values()).issubset(
            self.derived_object_ids
        ):
            raise MissingReferenceError(
                "active entry states must be target derived objects"
            )
        if self.generation_number == 0 and (
            self.fact_object_ids
            or self.derived_object_ids
            or self.applied_input_delta_ids
            or self.bundle_frontier
            or self.active_entry_state_ids
        ):
            raise ContractValidationError("generation zero must be empty")


@dataclass(frozen=True)
class CatalogDeltaManifest(StrictContract):
    """Reviewable exact transition, intentionally outside target identity."""

    catalog_delta_manifest_id: str
    scope_contract_id: str
    operation_id: str
    base_generation_id: str
    base_generation_number: int
    target_generation_id: str
    target_generation_number: int
    added_object_ids: tuple[str, ...]
    added_input_delta_ids: tuple[str, ...]
    target_derived_object_ids: tuple[str, ...]
    bundle_frontier_changes: Mapping[str, str]
    active_entry_state_changes: Mapping[str, str]

    CONTENT_NAMESPACE = "catalog-delta-manifest"
    IDENTITY_FIELD = "catalog_delta_manifest_id"

    def _validate(self) -> None:
        require_content_id(self.scope_contract_id, "scope_contract_id")
        _require_operation_id(self.operation_id)
        require_content_id(self.base_generation_id, "base_generation_id")
        require_content_id(self.target_generation_id, "target_generation_id")
        if self.base_generation_number < 0:
            raise ContractValidationError("base_generation_number must be non-negative")
        if self.target_generation_number != self.base_generation_number + 1:
            raise ContractValidationError(
                "target generation must immediately follow base generation"
            )
        if self.base_generation_id == self.target_generation_id:
            raise ContractValidationError(
                "catalog delta must change the generation identity"
            )
        _require_sorted_content_ids(self.added_object_ids, "added_object_ids")
        _require_sorted_content_ids(
            self.added_input_delta_ids,
            "added_input_delta_ids",
            required=True,
        )
        _require_sorted_content_ids(
            self.target_derived_object_ids, "target_derived_object_ids"
        )
        object.__setattr__(
            self,
            "bundle_frontier_changes",
            _freeze_reference_mapping(
                self.bundle_frontier_changes,
                "bundle_frontier_changes",
                content_keys=False,
            ),
        )
        object.__setattr__(
            self,
            "active_entry_state_changes",
            _freeze_reference_mapping(
                self.active_entry_state_changes,
                "active_entry_state_changes",
                content_keys=True,
            ),
        )


@dataclass(frozen=True)
class CatalogReductionRequest:
    """Complete reducer input; no precomputed state merge is accepted."""

    scope_contract_id: str
    generation_number: int
    parent_generation: CatalogGenerationManifest
    configuration_fingerprint: str
    fact_object_ids: tuple[str, ...]
    applied_input_delta_ids: tuple[str, ...]
    read_object_bytes: Callable[[str], bytes]


@dataclass(frozen=True)
class CatalogReduction:
    """Derived frontiers and states returned by a deterministic reducer."""

    bundle_frontier: Mapping[str, str]
    active_entry_state_ids: Mapping[str, str]
    derived_objects: tuple[StrictContract, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bundle_frontier",
            _freeze_reference_mapping(
                self.bundle_frontier,
                "bundle_frontier",
                content_keys=False,
            ),
        )
        object.__setattr__(
            self,
            "active_entry_state_ids",
            _freeze_reference_mapping(
                self.active_entry_state_ids,
                "active_entry_state_ids",
                content_keys=True,
            ),
        )
        if not isinstance(self.derived_objects, tuple):
            raise ContractValidationError("derived_objects must be a tuple")
        for record in self.derived_objects:
            if not isinstance(record, StrictContract):
                raise ContractValidationError(
                    "derived_objects must be strict immutable contracts"
                )


class CatalogReducer(Protocol):
    """Reduce a complete grow-only fact set from scratch."""

    def __call__(self, request: CatalogReductionRequest) -> CatalogReduction: ...


@dataclass(frozen=True)
class CatalogCommitResult:
    generation: CatalogGenerationManifest
    delta_manifest: CatalogDeltaManifest | None
    replayed: bool


CatalogCrashInjector = Callable[[str], None]


class CatalogStore:
    """Content-addressed objects plus one compare-and-swapped current pointer."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.objects_path = self.root / "objects"
        self.operations_path = self.root / "operations"
        self.staging_path = self.root / "staging"
        self.current_path = self.root / "current.json"
        self.lock_path = self.root / "catalog.lock"
        self._protected_namespace_authorities: dict[str, object | None] = {
            namespace: None for namespace in _PROTECTED_FACT_NAMESPACES
        }
        self._prepare_layout()

    def _bind_protected_namespace_authority(
        self,
        *,
        namespace: str,
        authority: object,
    ) -> None:
        if namespace not in self._protected_namespace_authorities:
            raise CatalogProtectedNamespaceError(
                "catalog namespace was not reserved for protected publication"
            )
        current = self._protected_namespace_authorities[namespace]
        if current is not None and current is not authority:
            raise CatalogProtectedNamespaceError(
                "catalog namespace already has another publication authority"
            )
        self._protected_namespace_authorities[namespace] = authority

    def initialize(
        self,
        *,
        scope_contract_id: str,
        configuration_fingerprint: str,
        crash_injector: CatalogCrashInjector | None = None,
    ) -> CatalogGenerationManifest:
        require_content_id(scope_contract_id, "scope_contract_id")
        _require_digest(configuration_fingerprint, "configuration_fingerprint")
        with self._locked():
            if os.path.lexists(self.current_path):
                current = self._read_current_locked()
                if current.scope_contract_id != scope_contract_id:
                    raise CatalogOperationConflictError(
                        "catalog is initialized for another scope contract"
                    )
                if current.generation_number == 0 and (
                    current.configuration_fingerprint != configuration_fingerprint
                ):
                    raise CatalogOperationConflictError(
                        "generation-zero configuration fingerprint differs"
                    )
                return current
            generation = CatalogGenerationManifest.mint(
                scope_contract_id=scope_contract_id,
                generation_number=0,
                parent_generation_id=None,
                configuration_fingerprint=configuration_fingerprint,
                fact_object_ids=(),
                derived_object_ids=(),
                applied_input_delta_ids=(),
                bundle_frontier={},
                active_entry_state_ids={},
            )
            self._write_contract(generation)
            self._inject(crash_injector, "generation_persisted")
            self._replace_pointer(generation)
            self._inject(crash_injector, "pointer_replaced")
            self._fsync_directory(self.root)
            self._inject(crash_injector, "pointer_directory_synced")
            return generation

    def read_current(self) -> CatalogGenerationManifest:
        with self._locked():
            return self._read_current_locked()

    def read_object_bytes(self, object_id: str) -> bytes:
        require_content_id(object_id, "object_id")
        path = self._object_path(object_id)
        payload = self._read_regular_bytes(path, "catalog object")
        self._verify_content_object_bytes(object_id, payload)
        return payload

    def read_contract(
        self, object_id: str, contract_type: type[StrictContract]
    ) -> StrictContract:
        record = contract_type.from_json_bytes(self.read_object_bytes(object_id))
        identity_field = contract_type.IDENTITY_FIELD
        if identity_field is None or getattr(record, identity_field) != object_id:
            raise CatalogCorruptionError(
                "stored contract does not own the requested object identity"
            )
        return record

    def publish(
        self,
        *,
        expected_generation_id: str,
        expected_generation_number: int,
        input_delta: CatalogInputDelta,
        objects: tuple[StrictContract, ...],
        reducer: CatalogReducer,
        crash_injector: CatalogCrashInjector | None = None,
    ) -> CatalogCommitResult:
        """Publish only if the exact expected pointer remains current."""
        return self._publish_compare_and_swap(
            expected_generation_id=expected_generation_id,
            expected_generation_number=expected_generation_number,
            input_delta=input_delta,
            objects=objects,
            reducer=reducer,
            crash_injector=crash_injector,
            protected_namespace=None,
            protected_authority=None,
        )

    def _publish_protected_namespace(
        self,
        *,
        namespace: str,
        authority: object,
        expected_generation_id: str,
        expected_generation_number: int,
        input_delta: CatalogInputDelta,
        objects: tuple[StrictContract, ...],
        reducer: CatalogReducer,
        crash_injector: CatalogCrashInjector | None = None,
    ) -> CatalogCommitResult:
        """Publish one reserved-namespace delta under its process-local authority."""
        return self._publish_compare_and_swap(
            expected_generation_id=expected_generation_id,
            expected_generation_number=expected_generation_number,
            input_delta=input_delta,
            objects=objects,
            reducer=reducer,
            crash_injector=crash_injector,
            protected_namespace=namespace,
            protected_authority=authority,
        )

    def _publish_compare_and_swap(
        self,
        *,
        expected_generation_id: str,
        expected_generation_number: int,
        input_delta: CatalogInputDelta,
        objects: tuple[StrictContract, ...],
        reducer: CatalogReducer,
        crash_injector: CatalogCrashInjector | None,
        protected_namespace: str | None,
        protected_authority: object | None,
    ) -> CatalogCommitResult:
        require_content_id(expected_generation_id, "expected_generation_id")
        if expected_generation_number < 0:
            raise ContractValidationError(
                "expected_generation_number must be non-negative"
            )
        self._require_namespace_publication(
            input_delta=input_delta,
            protected_namespace=protected_namespace,
            protected_authority=protected_authority,
        )
        prepared = self._prepare_objects(input_delta, objects)
        with self._locked():
            current = self._read_current_locked()
            replay = self._replay_or_conflict(current, input_delta, prepared)
            if replay is not None:
                return replay
            if (
                current.catalog_generation_id != expected_generation_id
                or current.generation_number != expected_generation_number
            ):
                raise CatalogCompareAndSwapError(
                    expected_generation_id=expected_generation_id,
                    expected_generation_number=expected_generation_number,
                    actual_generation_id=current.catalog_generation_id,
                    actual_generation_number=current.generation_number,
                )
            return self._commit_locked(
                current=current,
                input_delta=input_delta,
                prepared_objects=prepared,
                reducer=reducer,
                crash_injector=crash_injector,
            )

    def rebase(
        self,
        *,
        input_delta: CatalogInputDelta,
        objects: tuple[StrictContract, ...],
        reducer: CatalogReducer,
        crash_injector: CatalogCrashInjector | None = None,
    ) -> CatalogCommitResult:
        """Union additive input with the winner and fully reduce the new closure."""
        self._require_namespace_publication(
            input_delta=input_delta,
            protected_namespace=None,
            protected_authority=None,
        )
        prepared = self._prepare_objects(input_delta, objects)
        with self._locked():
            current = self._read_current_locked()
            replay = self._replay_or_conflict(current, input_delta, prepared)
            if replay is not None:
                return replay
            return self._commit_locked(
                current=current,
                input_delta=input_delta,
                prepared_objects=prepared,
                reducer=reducer,
                crash_injector=crash_injector,
            )

    def _require_namespace_publication(
        self,
        *,
        input_delta: CatalogInputDelta,
        protected_namespace: str | None,
        protected_authority: object | None,
    ) -> None:
        if type(input_delta) is not CatalogInputDelta:
            raise ContractValidationError("input_delta must be CatalogInputDelta")
        namespaces = {
            object_id.split(_CONTENT_ID_SEPARATOR, 1)[0]
            for object_id in input_delta.added_object_ids
        }
        protected = namespaces & set(self._protected_namespace_authorities)
        if protected_namespace is None:
            if protected:
                raise CatalogProtectedNamespaceError(
                    "protected catalog facts require their bound publication authority"
                )
            return
        if (
            protected_authority is None
            or protected != {protected_namespace}
            or namespaces != {protected_namespace}
            or self._protected_namespace_authorities.get(protected_namespace)
            is not protected_authority
        ):
            raise CatalogProtectedNamespaceError(
                "protected catalog publication is unbound or mixes namespaces"
            )

    def _commit_locked(
        self,
        *,
        current: CatalogGenerationManifest,
        input_delta: CatalogInputDelta,
        prepared_objects: Mapping[str, bytes],
        reducer: CatalogReducer,
        crash_injector: CatalogCrashInjector | None,
    ) -> CatalogCommitResult:
        if input_delta.scope_contract_id != current.scope_contract_id:
            raise CatalogClosureError("input delta belongs to another scope contract")
        candidate_fact_ids = tuple(
            sorted(set(current.fact_object_ids) | set(input_delta.added_object_ids))
        )
        if not set(input_delta.dependency_closure_ids).issubset(candidate_fact_ids):
            raise CatalogClosureError(
                "input delta dependency closure is absent from candidate facts"
            )
        self._write_operation_binding(input_delta)
        for object_id in sorted(prepared_objects):
            self._write_immutable_bytes(
                self._object_path(object_id), prepared_objects[object_id]
            )
        self._inject(crash_injector, "objects_persisted")
        self._write_contract(input_delta)
        self._inject(crash_injector, "input_delta_persisted")
        applied_delta_ids = tuple(
            sorted((*current.applied_input_delta_ids, input_delta.input_delta_id))
        )
        request = CatalogReductionRequest(
            scope_contract_id=current.scope_contract_id,
            generation_number=current.generation_number + 1,
            parent_generation=current,
            configuration_fingerprint=input_delta.configuration_fingerprint,
            fact_object_ids=candidate_fact_ids,
            applied_input_delta_ids=applied_delta_ids,
            read_object_bytes=self.read_object_bytes,
        )
        reduction = reducer(request)
        if not isinstance(reduction, CatalogReduction):
            raise CatalogReducerError("catalog reducer returned an invalid result")
        prepared_derived_objects = self._prepare_derived_objects(
            reduction.derived_objects
        )
        self._validate_reduction(
            current,
            request,
            reduction,
            prepared_derived_objects,
        )
        for object_id in sorted(prepared_derived_objects):
            self._write_immutable_bytes(
                self._object_path(object_id),
                prepared_derived_objects[object_id],
            )
        self._inject(crash_injector, "derived_objects_persisted")
        target = CatalogGenerationManifest.mint(
            scope_contract_id=current.scope_contract_id,
            generation_number=current.generation_number + 1,
            parent_generation_id=current.catalog_generation_id,
            configuration_fingerprint=input_delta.configuration_fingerprint,
            fact_object_ids=candidate_fact_ids,
            derived_object_ids=tuple(sorted(prepared_derived_objects)),
            applied_input_delta_ids=applied_delta_ids,
            bundle_frontier=reduction.bundle_frontier,
            active_entry_state_ids=reduction.active_entry_state_ids,
        )
        self._write_contract(target)
        self._inject(crash_injector, "generation_persisted")
        actual_added_object_ids = tuple(
            sorted(set(candidate_fact_ids) - set(current.fact_object_ids))
        )
        delta_manifest = CatalogDeltaManifest.mint(
            scope_contract_id=current.scope_contract_id,
            operation_id=input_delta.operation_id,
            base_generation_id=current.catalog_generation_id,
            base_generation_number=current.generation_number,
            target_generation_id=target.catalog_generation_id,
            target_generation_number=target.generation_number,
            added_object_ids=actual_added_object_ids,
            added_input_delta_ids=(input_delta.input_delta_id,),
            target_derived_object_ids=target.derived_object_ids,
            bundle_frontier_changes=self._mapping_changes(
                current.bundle_frontier, target.bundle_frontier
            ),
            active_entry_state_changes=self._mapping_changes(
                current.active_entry_state_ids,
                target.active_entry_state_ids,
            ),
        )
        self._write_contract(delta_manifest)
        self._inject(crash_injector, "delta_manifest_persisted")
        self._replace_pointer(target)
        self._inject(crash_injector, "pointer_replaced")
        self._fsync_directory(self.root)
        self._inject(crash_injector, "pointer_directory_synced")
        return CatalogCommitResult(
            generation=target,
            delta_manifest=delta_manifest,
            replayed=False,
        )

    def _prepare_objects(
        self,
        input_delta: CatalogInputDelta,
        objects: tuple[StrictContract, ...],
    ) -> Mapping[str, bytes]:
        if not isinstance(input_delta, CatalogInputDelta):
            raise ContractValidationError("input_delta must be CatalogInputDelta")
        prepared: dict[str, bytes] = {}
        for record in objects:
            if not isinstance(record, StrictContract):
                raise ContractValidationError(
                    "catalog objects must be strict immutable contracts"
                )
            if isinstance(record, CatalogEntryState):
                raise CatalogClosureError(
                    "CatalogEntryState must be created by the target reducer"
                )
            if record.CONTENT_EXCLUDED_FIELDS:
                raise ContractValidationError(
                    "attested payloads require a separate immutable envelope"
                )
            identity_field = record.IDENTITY_FIELD
            if identity_field is None:
                raise ContractValidationError(
                    "catalog objects must declare a content identity"
                )
            object_id = getattr(record, identity_field)
            require_content_id(object_id, "catalog object identity")
            payload = record.to_json_bytes()
            existing = prepared.get(object_id)
            if existing is not None and existing != payload:
                raise CatalogOperationConflictError(
                    "one object identity has conflicting in-memory bytes"
                )
            prepared[object_id] = payload
        if set(prepared) != set(input_delta.added_object_ids):
            raise CatalogClosureError(
                "objects must exactly match input_delta.added_object_ids"
            )
        return MappingProxyType(prepared)

    def _prepare_derived_objects(
        self, records: tuple[StrictContract, ...]
    ) -> Mapping[str, bytes]:
        prepared: dict[str, bytes] = {}
        for record in records:
            if record.CONTENT_EXCLUDED_FIELDS:
                raise CatalogReducerError(
                    "derived attested payloads require a separate envelope"
                )
            identity_field = record.IDENTITY_FIELD
            if identity_field is None:
                raise CatalogReducerError(
                    "derived objects must declare a content identity"
                )
            object_id = getattr(record, identity_field)
            require_content_id(object_id, "derived object identity")
            payload = record.to_json_bytes()
            existing = prepared.get(object_id)
            if existing is not None:
                raise CatalogReducerError(
                    "reducer returned a duplicate derived object identity"
                )
            prepared[object_id] = payload
        return MappingProxyType(prepared)

    def _replay_or_conflict(
        self,
        current: CatalogGenerationManifest,
        input_delta: CatalogInputDelta,
        prepared_objects: Mapping[str, bytes],
    ) -> CatalogCommitResult | None:
        bound_delta_id = self._read_operation_binding(input_delta.operation_id)
        if bound_delta_id is not None and bound_delta_id != input_delta.input_delta_id:
            raise CatalogOperationConflictError(
                "operation_id is already bound to another input delta"
            )
        if input_delta.input_delta_id not in current.applied_input_delta_ids:
            return None
        stored_delta = self.read_contract(input_delta.input_delta_id, CatalogInputDelta)
        if stored_delta.to_json_bytes() != input_delta.to_json_bytes():
            raise CatalogOperationConflictError(
                "applied input delta bytes conflict with replay"
            )
        for object_id, payload in prepared_objects.items():
            if self.read_object_bytes(object_id) != payload:
                raise CatalogOperationConflictError(
                    f"replayed object bytes conflict for {object_id}"
                )
        return CatalogCommitResult(
            generation=current,
            delta_manifest=None,
            replayed=True,
        )

    def _read_current_locked(self) -> CatalogGenerationManifest:
        if not os.path.lexists(self.current_path):
            raise CatalogNotInitializedError("catalog has no current generation")
        pointer_bytes = self._read_regular_bytes(
            self.current_path, "catalog current pointer"
        )
        pointer = parse_json_bytes(pointer_bytes)
        if not isinstance(pointer, MappingABC) or set(pointer) != {
            "catalog_generation_id",
            "generation_number",
        }:
            raise CatalogCorruptionError("catalog current pointer is malformed")
        if canonical_json_bytes(pointer) != pointer_bytes:
            raise CatalogCorruptionError("catalog current pointer is not canonical")
        generation_id = pointer["catalog_generation_id"]
        generation_number = pointer["generation_number"]
        require_content_id(generation_id, "catalog_generation_id")
        if type(generation_number) is not int or generation_number < 0:
            raise CatalogCorruptionError("catalog pointer generation_number is invalid")
        generation = self.read_contract(generation_id, CatalogGenerationManifest)
        if generation.generation_number != generation_number:
            raise CatalogCorruptionError(
                "catalog pointer generation number does not match manifest"
            )
        self._validate_generation_lineage(generation)
        return generation

    def _validate_generation_lineage(self, tip: CatalogGenerationManifest) -> None:
        lineage = [tip]
        cursor = tip
        while cursor.parent_generation_id is not None:
            parent = self.read_contract(
                cursor.parent_generation_id, CatalogGenerationManifest
            )
            if not isinstance(parent, CatalogGenerationManifest):
                raise CatalogCorruptionError("generation parent type is invalid")
            lineage.append(parent)
            cursor = parent
        lineage.reverse()
        if lineage[0].generation_number != 0:
            raise CatalogCorruptionError("catalog lineage does not terminate at G0")
        scope_contract_id = lineage[0].scope_contract_id
        for position, generation in enumerate(lineage):
            if generation.scope_contract_id != scope_contract_id:
                raise CatalogCorruptionError("catalog scope lineage changed")
            for object_id in generation.fact_object_ids:
                self.read_object_bytes(object_id)
            for object_id in generation.derived_object_ids:
                self.read_object_bytes(object_id)
            for delta_id in generation.applied_input_delta_ids:
                delta = self.read_contract(delta_id, CatalogInputDelta)
                if not isinstance(delta, CatalogInputDelta):
                    raise CatalogCorruptionError("input delta type is invalid")
                if delta.scope_contract_id != scope_contract_id:
                    raise CatalogCorruptionError("input delta scope lineage changed")
                if self._read_operation_binding(delta.operation_id) != delta_id:
                    raise CatalogCorruptionError(
                        "applied input delta operation binding is absent or changed"
                    )
                if not set(delta.dependency_closure_ids).issubset(
                    generation.fact_object_ids
                ):
                    raise CatalogCorruptionError(
                        "generation omits an input delta dependency"
                    )
            if position == 0:
                continue
            parent = lineage[position - 1]
            if generation.generation_number != parent.generation_number + 1:
                raise CatalogCorruptionError("catalog generation numbers have a gap")
            if generation.parent_generation_id != parent.catalog_generation_id:
                raise CatalogCorruptionError("catalog parent identity is inconsistent")
            new_delta_ids = set(generation.applied_input_delta_ids) - set(
                parent.applied_input_delta_ids
            )
            if len(new_delta_ids) != 1:
                raise CatalogCorruptionError(
                    "each catalog generation must apply exactly one input delta"
                )
            if not set(parent.applied_input_delta_ids).issubset(
                generation.applied_input_delta_ids
            ):
                raise CatalogCorruptionError("applied input delta history shrank")
            new_delta_id = next(iter(new_delta_ids))
            new_delta = self.read_contract(new_delta_id, CatalogInputDelta)
            if not isinstance(new_delta, CatalogInputDelta):
                raise CatalogCorruptionError("input delta type is invalid")
            expected_facts = set(parent.fact_object_ids) | set(
                new_delta.added_object_ids
            )
            if set(generation.fact_object_ids) != expected_facts:
                raise CatalogCorruptionError(
                    "generation fact closure is not the exact additive union"
                )
            if generation.configuration_fingerprint != (
                new_delta.configuration_fingerprint
            ):
                raise CatalogCorruptionError(
                    "generation configuration does not match applied input delta"
                )
            if not set(parent.bundle_frontier).issubset(generation.bundle_frontier):
                raise CatalogCorruptionError("bundle frontier keys were removed")
            if not set(parent.active_entry_state_ids).issubset(
                generation.active_entry_state_ids
            ):
                raise CatalogCorruptionError("active catalog subjects were removed")
            for subject_id, state_id in generation.active_entry_state_ids.items():
                state = self.read_contract(state_id, CatalogEntryState)
                if not isinstance(state, CatalogEntryState):
                    raise CatalogCorruptionError(
                        "active state object has an invalid type"
                    )
                if state.subject_payload_id != subject_id:
                    raise CatalogCorruptionError(
                        "active state does not own its mapped subject"
                    )
                if state.catalog_generation != generation.generation_number:
                    raise CatalogCorruptionError(
                        "active state names another catalog generation"
                    )
                if state.configuration_fingerprint != (
                    generation.configuration_fingerprint
                ):
                    raise CatalogCorruptionError(
                        "active state configuration differs from its generation"
                    )
                if state.predecessor_state_id != (
                    parent.active_entry_state_ids.get(subject_id)
                ):
                    raise CatalogCorruptionError(
                        "active state predecessor does not match parent projection"
                    )

    def _validate_reduction(
        self,
        parent: CatalogGenerationManifest,
        request: CatalogReductionRequest,
        reduction: CatalogReduction,
        prepared_derived_objects: Mapping[str, bytes],
    ) -> None:
        facts = set(request.fact_object_ids)
        if facts & set(prepared_derived_objects):
            raise CatalogReducerError(
                "reducer-derived objects cannot reuse source fact identities"
            )
        source_references = set(reduction.bundle_frontier.values())
        source_references.update(reduction.active_entry_state_ids)
        if not source_references.issubset(facts):
            raise CatalogReducerError(
                "reducer frontier or subjects leave the complete source fact set"
            )
        if not set(reduction.active_entry_state_ids.values()).issubset(
            prepared_derived_objects
        ):
            raise CatalogReducerError(
                "active states must be returned as target derived objects"
            )
        if not set(parent.bundle_frontier).issubset(reduction.bundle_frontier):
            raise CatalogReducerError("reducer removed a bundle frontier key")
        if not set(parent.active_entry_state_ids).issubset(
            reduction.active_entry_state_ids
        ):
            raise CatalogReducerError("reducer removed an active catalog subject")
        if len(reduction.active_entry_state_ids.values()) != len(
            set(reduction.active_entry_state_ids.values())
        ):
            raise CatalogReducerError("one derived state cannot own multiple subjects")
        for subject_id, state_id in reduction.active_entry_state_ids.items():
            state = CatalogEntryState.from_json_bytes(
                prepared_derived_objects[state_id]
            )
            if state.subject_payload_id != subject_id:
                raise CatalogReducerError(
                    "derived state does not own its active subject"
                )
            if state.catalog_generation != request.generation_number:
                raise CatalogReducerError(
                    "derived state was precomputed for another generation"
                )
            if state.configuration_fingerprint != (request.configuration_fingerprint):
                raise CatalogReducerError(
                    "derived state configuration differs from reduction"
                )
            if state.predecessor_state_id != (
                parent.active_entry_state_ids.get(subject_id)
            ):
                raise CatalogReducerError(
                    "derived state predecessor differs from parent projection"
                )

    def _write_contract(self, record: StrictContract) -> None:
        if record.CONTENT_EXCLUDED_FIELDS:
            raise ContractValidationError(
                "attested payloads require a separate immutable envelope"
            )
        identity_field = record.IDENTITY_FIELD
        if identity_field is None:
            raise ContractValidationError("persisted contracts need a content ID")
        object_id = getattr(record, identity_field)
        self._write_immutable_bytes(
            self._object_path(object_id), record.to_json_bytes()
        )

    def _write_operation_binding(self, input_delta: CatalogInputDelta) -> None:
        existing = self._read_operation_binding(input_delta.operation_id)
        if existing is not None and existing != input_delta.input_delta_id:
            raise CatalogOperationConflictError(
                "operation_id is already bound to another input delta"
            )
        binding = canonical_json_bytes(
            {
                "input_delta_id": input_delta.input_delta_id,
                "operation_id": input_delta.operation_id,
            }
        )
        self._write_immutable_bytes(
            self._operation_path(input_delta.operation_id), binding
        )

    def _read_operation_binding(self, operation_id: str) -> str | None:
        path = self._operation_path(operation_id)
        if not os.path.lexists(path):
            return None
        payload = self._read_regular_bytes(path, "catalog operation binding")
        parsed = parse_json_bytes(payload)
        if not isinstance(parsed, MappingABC) or set(parsed) != {
            "input_delta_id",
            "operation_id",
        }:
            raise CatalogCorruptionError("catalog operation binding is malformed")
        if canonical_json_bytes(parsed) != payload:
            raise CatalogCorruptionError("catalog operation binding is not canonical")
        if parsed["operation_id"] != operation_id:
            raise CatalogCorruptionError("catalog operation binding key mismatch")
        require_content_id(parsed["input_delta_id"], "input_delta_id")
        return parsed["input_delta_id"]

    def _replace_pointer(self, generation: CatalogGenerationManifest) -> None:
        if os.path.lexists(self.current_path):
            self._require_regular_file(self.current_path, "catalog current pointer")
        payload = canonical_json_bytes(
            {
                "catalog_generation_id": generation.catalog_generation_id,
                "generation_number": generation.generation_number,
            }
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.staging_path,
            prefix="current.",
            suffix=".tmp",
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        temporary.chmod(0o600)
        os.replace(temporary, self.current_path)
        self.current_path.chmod(0o600)

    def _write_immutable_bytes(self, path: Path, payload: bytes) -> None:
        if os.path.lexists(path):
            existing = self._read_regular_bytes(path, "immutable catalog file")
            if existing != payload:
                raise CatalogCorruptionError(
                    f"immutable catalog bytes conflict at {path.name}"
                )
            return
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.staging_path,
            prefix="object.",
            suffix=".tmp",
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
        self._fsync_directory(path.parent)

    def _verify_content_object_bytes(
        self, declared_object_id: str, payload: bytes
    ) -> None:
        parsed = parse_json_bytes(payload)
        if not isinstance(parsed, MappingABC):
            raise CatalogCorruptionError("catalog object must be a JSON object")
        if canonical_json_bytes(parsed) != payload:
            raise CatalogCorruptionError("catalog object bytes are not canonical")
        identity_fields = tuple(
            key for key, value in parsed.items() if value == declared_object_id
        )
        if len(identity_fields) != 1:
            raise CatalogCorruptionError(
                "catalog object must declare its identity exactly once"
            )
        namespace, separator, digest = declared_object_id.partition(
            _CONTENT_ID_SEPARATOR
        )
        if separator != _CONTENT_ID_SEPARATOR or not digest:
            raise CatalogCorruptionError("catalog object identity is malformed")
        identity_field = identity_fields[0]
        preimage = {
            key: value for key, value in parsed.items() if key != identity_field
        }
        if content_id(namespace, preimage) != declared_object_id:
            raise CatalogCorruptionError(
                "catalog object bytes do not match their content identity"
            )

    def _prepare_layout(self) -> None:
        self._reject_symlink_ancestors(self.root)
        if os.path.lexists(self.root) and not self.root.is_dir():
            raise CatalogLayoutError("catalog root must be a directory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        for directory in (
            self.objects_path,
            self.operations_path,
            self.staging_path,
        ):
            if os.path.lexists(directory) and (
                directory.is_symlink() or not directory.is_dir()
            ):
                raise CatalogLayoutError(
                    f"catalog path must be a real directory: {directory.name}"
                )
            directory.mkdir(exist_ok=True)
            directory.chmod(0o700)
        if os.path.lexists(self.lock_path):
            self._require_regular_file(self.lock_path, "catalog lock")
        lock_descriptor = os.open(
            self.lock_path,
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        lock_status = os.fstat(lock_descriptor)
        if not stat.S_ISREG(lock_status.st_mode):
            os.close(lock_descriptor)
            raise CatalogLayoutError("catalog lock must be a regular file")
        os.fsync(lock_descriptor)
        os.close(lock_descriptor)
        self.lock_path.chmod(0o600)
        self._fsync_directory(self.root)

    def _locked(self) -> Any:
        descriptor = os.open(
            self.lock_path,
            os.O_RDWR | os.O_NOFOLLOW,
        )
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            os.close(descriptor)
            raise CatalogLayoutError("catalog lock must be a regular file")
        handle = os.fdopen(descriptor, "r+b")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _object_path(self, object_id: str) -> Path:
        require_content_id(object_id, "object_id")
        return self.objects_path / f"{object_id}.json"

    def _operation_path(self, operation_id: str) -> Path:
        _require_operation_id(operation_id)
        digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        return self.operations_path / f"{digest}.json"

    @staticmethod
    def _mapping_changes(
        previous: Mapping[str, str], current: Mapping[str, str]
    ) -> Mapping[str, str]:
        return {
            key: value for key, value in current.items() if previous.get(key) != value
        }

    @staticmethod
    def _read_regular_bytes(path: Path, name: str) -> bytes:
        CatalogStore._require_regular_file(path, name)
        return path.read_bytes()

    @staticmethod
    def _require_regular_file(path: Path, name: str) -> None:
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise CatalogLayoutError(f"{name} must be a regular file")

    @staticmethod
    def _reject_symlink_ancestors(path: Path) -> None:
        absolute = path.absolute()
        candidates = tuple(reversed(absolute.parents)) + (absolute,)
        for candidate in candidates:
            if os.path.lexists(candidate) and candidate.is_symlink():
                raise CatalogLayoutError(f"catalog path traverses symlink: {candidate}")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.fsync(descriptor)
        os.close(descriptor)

    @staticmethod
    def _inject(crash_injector: CatalogCrashInjector | None, event: str) -> None:
        if crash_injector is not None:
            crash_injector(event)


__all__ = [
    "CatalogCrashInjector",
    "CatalogClosureError",
    "CatalogCommitResult",
    "CatalogCompareAndSwapError",
    "CatalogCorruptionError",
    "CatalogDeltaManifest",
    "CatalogGenerationManifest",
    "CatalogInputDelta",
    "CatalogLayoutError",
    "CatalogNotInitializedError",
    "CatalogOperationConflictError",
    "CatalogReducer",
    "CatalogReducerError",
    "CatalogReduction",
    "CatalogReductionRequest",
    "CatalogStore",
    "CatalogStoreError",
]
