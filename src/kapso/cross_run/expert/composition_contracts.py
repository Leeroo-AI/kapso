"""Stable planning contracts for deterministic expert candidate composition."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import ClassVar

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    require_identifier,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    CrossRunTaskBindingSettings,
    ExpertCandidateDerivationKind,
    ExpertCandidatePatch,
    ExpertCandidatePatchChange,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
    ExpertSourceTreeManifest,
    StrictContract,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.topology import (
    validate_expert_repository_topology,
    validate_expert_tree_ownership,
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExpertCompositionContractError(ValueError):
    """Composition plan or deterministic assessment is inconsistent."""


class ExpertCompositionDisposition(str, Enum):
    """Pure result of reducing approved sources against one current base."""

    CLEAN = "clean"
    ALREADY_PRESENT = "already_present"
    CONFLICTED = "conflicted"
    REQUIRES_RESTRUCTURE = "requires_restructure"


class ExpertCompositionConflictKind(str, Enum):
    """Closed conflict classes understood by the composition orchestrator."""

    SOURCE_BASE = "source_base"
    PATH_OVERLAP = "path_overlap"
    PATH_PREFIX = "path_prefix"
    CURRENT_PATH = "current_path"
    MODULE = "module"
    TOPOLOGY = "topology"
    CAPABILITY_LINEAGE = "capability_lineage"
    DEPENDENCY_GRAPH = "dependency_graph"
    CAPABILITY_INCOMPATIBILITY = "capability_incompatibility"
    RESOURCE_ENVELOPE = "resource_envelope"
    ADAPTER_BOUNDARY = "adapter_boundary"
    ARCHITECTURE = "architecture"


class ExpertCompositionConflictSubjectKind(str, Enum):
    """Canonical syntax used to identify a conflict subject."""

    AUTHORITY = "authority"
    PATH = "path"
    CAPABILITY = "capability"
    REPOSITORY = "repository"


_CONFLICT_SUBJECT_KINDS = {
    ExpertCompositionConflictKind.SOURCE_BASE: (
        ExpertCompositionConflictSubjectKind.AUTHORITY
    ),
    ExpertCompositionConflictKind.PATH_OVERLAP: (
        ExpertCompositionConflictSubjectKind.PATH
    ),
    ExpertCompositionConflictKind.PATH_PREFIX: (
        ExpertCompositionConflictSubjectKind.PATH
    ),
    ExpertCompositionConflictKind.CURRENT_PATH: (
        ExpertCompositionConflictSubjectKind.PATH
    ),
    ExpertCompositionConflictKind.MODULE: (
        ExpertCompositionConflictSubjectKind.CAPABILITY
    ),
    ExpertCompositionConflictKind.TOPOLOGY: (
        ExpertCompositionConflictSubjectKind.CAPABILITY
    ),
    ExpertCompositionConflictKind.CAPABILITY_LINEAGE: (
        ExpertCompositionConflictSubjectKind.CAPABILITY
    ),
    ExpertCompositionConflictKind.DEPENDENCY_GRAPH: (
        ExpertCompositionConflictSubjectKind.CAPABILITY
    ),
    ExpertCompositionConflictKind.CAPABILITY_INCOMPATIBILITY: (
        ExpertCompositionConflictSubjectKind.CAPABILITY
    ),
    ExpertCompositionConflictKind.RESOURCE_ENVELOPE: (
        ExpertCompositionConflictSubjectKind.CAPABILITY
    ),
    ExpertCompositionConflictKind.ADAPTER_BOUNDARY: (
        ExpertCompositionConflictSubjectKind.PATH
    ),
    ExpertCompositionConflictKind.ARCHITECTURE: (
        ExpertCompositionConflictSubjectKind.REPOSITORY
    ),
}
_RESTRUCTURE_CONFLICT_KINDS = {
    ExpertCompositionConflictKind.TOPOLOGY,
    ExpertCompositionConflictKind.CAPABILITY_LINEAGE,
    ExpertCompositionConflictKind.DEPENDENCY_GRAPH,
    ExpertCompositionConflictKind.ADAPTER_BOUNDARY,
    ExpertCompositionConflictKind.ARCHITECTURE,
}


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise ExpertCompositionContractError(f"{name} uses the wrong namespace")


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ExpertCompositionContractError(f"{name} must be a sha256 digest")


def expert_composition_configuration_fingerprint(
    *,
    composition_policy_version: str,
    composition_source_limit: int,
    candidate_entry_limit: int,
    candidate_byte_limit: int,
) -> str:
    """Identify every configured input that can change pure composition."""

    require_identifier(composition_policy_version, "composition policy version")
    if (
        type(composition_source_limit) is not int
        or composition_source_limit <= 0
        or type(candidate_entry_limit) is not int
        or candidate_entry_limit <= 0
        or type(candidate_byte_limit) is not int
        or candidate_byte_limit <= 0
    ):
        raise ExpertCompositionContractError(
            "composition limits must be positive integers"
        )
    return tree_or_blob_digest(
        canonical_json_bytes(
            {
                "candidate_byte_limit": candidate_byte_limit,
                "candidate_entry_limit": candidate_entry_limit,
                "composition_policy_version": composition_policy_version,
                "composition_source_limit": composition_source_limit,
            }
        )
    )


def _require_sorted_content_ids(
    values: tuple[str, ...],
    name: str,
    *,
    required: bool = True,
) -> None:
    if (required and not values) or values != tuple(sorted(set(values))):
        raise ExpertCompositionContractError(f"{name} must be sorted and unique")
    for value in values:
        require_content_id(value, name)


def _require_sorted_namespaced_content_ids(
    values: tuple[str, ...],
    namespace: str,
    name: str,
    *,
    required: bool = True,
) -> None:
    _require_sorted_content_ids(values, name, required=required)
    for value in values:
        _require_namespaced_id(value, namespace, name)


def _require_relative_paths(values: tuple[str, ...], name: str) -> None:
    if not values or values != tuple(sorted(set(values))):
        raise ExpertCompositionContractError(
            f"{name} must be non-empty, sorted, and unique"
        )
    paths = tuple(PurePosixPath(value) for value in values)
    if any(
        not value
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or value != path.as_posix()
        for value, path in zip(values, paths)
    ):
        raise ExpertCompositionContractError(f"{name} contains an invalid path")


@dataclass(frozen=True)
class ExpertCompositionBaseReference(StrictContract):
    """Stable projection of a verified release, excluding publication metadata."""

    base_reference_id: str
    release_id: str
    scope_contract_id: str
    scope_id: str
    source_tree_hash: str
    repository_map_id: str
    module_contract_ids: tuple[str, ...]
    semantic_book_digest: str
    release_configuration_fingerprint: str
    stable_authority_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-composition-base-reference"
    IDENTITY_FIELD: ClassVar[str] = "base_reference_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.release_id, "expert-base-release", "composition base release"),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "composition base scope contract",
            ),
            (
                self.repository_map_id,
                "expert-repository-map",
                "composition base repository map",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        require_identifier(self.scope_id, "composition base scope")
        _require_digest(self.source_tree_hash, "composition base source tree")
        _require_digest(self.semantic_book_digest, "composition base semantic book")
        _require_digest(
            self.release_configuration_fingerprint,
            "composition base release configuration",
        )
        _require_sorted_namespaced_content_ids(
            self.module_contract_ids,
            "expert-module-contract",
            "composition base module contracts",
        )
        _require_sorted_content_ids(
            self.stable_authority_ids,
            "composition base stable authorities",
        )
        expected_authorities = {
            self.release_id,
            self.scope_contract_id,
            self.repository_map_id,
            *self.module_contract_ids,
        }
        if set(self.stable_authority_ids) != expected_authorities:
            raise ExpertCompositionContractError(
                "composition base stable authority closure is not exact"
            )


@dataclass(frozen=True)
class ExpertCompositionSourceReference(StrictContract):
    """Stable candidate projection resolved against trusted stores before use."""

    source_reference_id: str
    candidate_id: str
    candidate_commit_record_id: str
    scope_contract_id: str
    change_kind: CandidateChangeKind
    derivation_kind: ExpertCandidateDerivationKind
    derivation_ref: str
    validation_context_ref: str
    origin_principal_ids: tuple[str, ...]
    source_base_release_id: str
    source_base_repository_map_id: str
    source_base_tree_hash: str
    candidate_tree_hash: str
    patch_id: str
    patch_digest: str
    proposed_repository_map_id: str
    module_contract_ids: tuple[str, ...]
    candidate_configuration_fingerprint: str
    stable_authority_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-composition-source-reference"
    IDENTITY_FIELD: ClassVar[str] = "source_reference_id"

    def _validate(self) -> None:
        for value, namespace, name in (
            (self.candidate_id, "expert-candidate", "composition source candidate"),
            (
                self.candidate_commit_record_id,
                "expert-candidate-commit",
                "composition source candidate commit",
            ),
            (
                self.scope_contract_id,
                "expert-scope-contract",
                "composition source scope contract",
            ),
            (
                self.derivation_ref,
                "expert-agent-proposal-derivation",
                "composition source derivation",
            ),
            (
                self.validation_context_ref,
                "expert-candidate-validation-context",
                "composition source validation context",
            ),
            (
                self.source_base_release_id,
                "expert-base-release",
                "composition source source-base release",
            ),
            (
                self.source_base_repository_map_id,
                "expert-repository-map",
                "composition source source-base repository map",
            ),
            (self.patch_id, "expert-candidate-patch", "composition source patch"),
            (
                self.proposed_repository_map_id,
                "expert-repository-map",
                "composition source proposed repository map",
            ),
        ):
            _require_namespaced_id(value, namespace, name)
        if self.derivation_kind is not ExpertCandidateDerivationKind.AGENT_PROPOSAL:
            raise ExpertCompositionContractError(
                "composition sources must be direct agent proposals"
            )
        if not self.origin_principal_ids or self.origin_principal_ids != tuple(
            sorted(set(self.origin_principal_ids))
        ):
            raise ExpertCompositionContractError(
                "composition source origin principals must be canonical and non-empty"
            )
        for principal_id in self.origin_principal_ids:
            require_identifier(principal_id, "composition source origin principal")
        for value, name in (
            (self.source_base_tree_hash, "composition source source-base tree"),
            (self.candidate_tree_hash, "composition source candidate tree"),
            (self.patch_digest, "composition source patch digest"),
            (
                self.candidate_configuration_fingerprint,
                "composition source candidate configuration",
            ),
        ):
            _require_digest(value, name)
        if self.source_base_tree_hash == self.candidate_tree_hash:
            raise ExpertCompositionContractError(
                "composition source candidate must change its source-base tree"
            )
        _require_sorted_namespaced_content_ids(
            self.module_contract_ids,
            "expert-module-contract",
            "composition source module contracts",
        )
        _require_sorted_content_ids(
            self.stable_authority_ids,
            "composition source stable authorities",
        )
        expected_authorities = {
            self.candidate_id,
            self.candidate_commit_record_id,
            self.scope_contract_id,
            self.derivation_ref,
            self.validation_context_ref,
            self.source_base_release_id,
            self.source_base_repository_map_id,
            self.patch_id,
            self.proposed_repository_map_id,
            *self.module_contract_ids,
        }
        if set(self.stable_authority_ids) != expected_authorities:
            raise ExpertCompositionContractError(
                "composition source stable authority closure is not exact"
            )


@dataclass(frozen=True)
class ExpertCompositionPlan(StrictContract):
    """Stable scientific composition identity, independent of temporal fences."""

    composition_plan_id: str
    scope_contract: ExpertScopeContract
    current_base: ExpertCompositionBaseReference
    sources: tuple[ExpertCompositionSourceReference, ...]
    active_task_bindings: tuple[CrossRunTaskBindingSettings, ...]
    composition_policy_version: str
    composition_source_limit: int
    candidate_entry_limit: int
    candidate_byte_limit: int
    configuration_fingerprint: str
    stable_authority_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-composition-plan"
    IDENTITY_FIELD: ClassVar[str] = "composition_plan_id"

    def _validate(self) -> None:
        if (
            type(self.scope_contract) is not ExpertScopeContract
            or type(self.current_base) is not ExpertCompositionBaseReference
            or type(self.sources) is not tuple
            or any(
                type(source) is not ExpertCompositionSourceReference
                for source in self.sources
            )
            or type(self.active_task_bindings) is not tuple
            or any(
                type(binding) is not CrossRunTaskBindingSettings
                for binding in self.active_task_bindings
            )
        ):
            raise ExpertCompositionContractError(
                "composition plan requires exact typed stable authorities"
            )
        require_identifier(
            self.composition_policy_version, "composition policy version"
        )
        expected_configuration_fingerprint = (
            expert_composition_configuration_fingerprint(
                composition_policy_version=self.composition_policy_version,
                composition_source_limit=self.composition_source_limit,
                candidate_entry_limit=self.candidate_entry_limit,
                candidate_byte_limit=self.candidate_byte_limit,
            )
        )
        _require_digest(
            self.configuration_fingerprint,
            "composition configuration fingerprint",
        )
        if self.configuration_fingerprint != expected_configuration_fingerprint:
            raise ExpertCompositionContractError(
                "composition configuration fingerprint is not exact"
            )
        source_keys = tuple(
            (source.candidate_id, source.source_reference_id) for source in self.sources
        )
        if (
            not self.sources
            or source_keys != tuple(sorted(set(source_keys)))
            or len({source.candidate_id for source in self.sources})
            != len(self.sources)
            or len(self.sources) > self.composition_source_limit
        ):
            raise ExpertCompositionContractError(
                "composition sources must be non-empty, canonical, and unique "
                "within the configured source limit"
            )
        binding_keys = tuple(
            (binding.task_family_id, binding.task_adapter_id)
            for binding in self.active_task_bindings
        )
        if not binding_keys or binding_keys != tuple(sorted(set(binding_keys))):
            raise ExpertCompositionContractError(
                "composition active task bindings must be canonical and non-empty"
            )
        for binding in self.active_task_bindings:
            self.scope_contract.validate_binding(binding)
        base = self.current_base
        if (
            base.scope_contract_id != self.scope_contract.scope_contract_id
            or base.scope_id != self.scope_contract.scope_id
            or any(
                source.scope_contract_id != self.scope_contract.scope_contract_id
                for source in self.sources
            )
        ):
            raise ExpertCompositionContractError(
                "composition plan authorities do not share one exact scope"
            )
        _require_sorted_content_ids(
            self.stable_authority_ids,
            "composition plan stable authorities",
        )
        expected_authorities = {
            self.scope_contract.scope_contract_id,
            base.base_reference_id,
            *base.stable_authority_ids,
            *(source.source_reference_id for source in self.sources),
            *(
                authority_id
                for source in self.sources
                for authority_id in source.stable_authority_ids
            ),
        }
        if self.scope_contract.supersedes_scope_contract_id is not None:
            expected_authorities.add(self.scope_contract.supersedes_scope_contract_id)
        if set(self.stable_authority_ids) != expected_authorities:
            raise ExpertCompositionContractError(
                "composition plan stable authority closure is not exact"
            )

    @property
    def source_reference_ids(self) -> tuple[str, ...]:
        return tuple(source.source_reference_id for source in self.sources)


@dataclass(frozen=True)
class ExpertCompositionConflict(StrictContract):
    """One deterministic reason a source cannot be merged mechanically."""

    conflict_id: str
    kind: ExpertCompositionConflictKind
    subject_kind: ExpertCompositionConflictSubjectKind
    subjects: tuple[str, ...]
    source_reference_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-composition-conflict"
    IDENTITY_FIELD: ClassVar[str] = "conflict_id"

    def _validate(self) -> None:
        if self.subject_kind is not _CONFLICT_SUBJECT_KINDS[self.kind]:
            raise ExpertCompositionContractError(
                "composition conflict subject kind differs from conflict kind"
            )
        if self.subject_kind is ExpertCompositionConflictSubjectKind.PATH:
            _require_relative_paths(self.subjects, "composition conflict subjects")
        elif self.subject_kind is ExpertCompositionConflictSubjectKind.CAPABILITY:
            if not self.subjects or self.subjects != tuple(sorted(set(self.subjects))):
                raise ExpertCompositionContractError(
                    "composition conflict subjects must be canonical"
                )
            for subject in self.subjects:
                require_identifier(subject, "composition conflict capability")
        elif self.subject_kind is ExpertCompositionConflictSubjectKind.AUTHORITY:
            _require_sorted_content_ids(
                self.subjects,
                "composition conflict authority subjects",
            )
        elif self.subjects != ("repository",):
            raise ExpertCompositionContractError(
                "repository conflict must use the canonical repository subject"
            )
        _require_sorted_namespaced_content_ids(
            self.source_reference_ids,
            "expert-composition-source-reference",
            "composition conflict source references",
        )
        if (
            self.kind is ExpertCompositionConflictKind.PATH_OVERLAP
            and len(self.source_reference_ids) < 2
        ):
            raise ExpertCompositionContractError(
                "cross-source path conflict requires multiple sources"
            )
        if (
            self.kind is ExpertCompositionConflictKind.PATH_PREFIX
            and len(self.subjects) < 2
        ):
            raise ExpertCompositionContractError(
                "path-prefix conflict requires multiple paths"
            )

    @property
    def canonical_key(self) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
        return (
            self.kind.value,
            self.subject_kind.value,
            self.subjects,
            self.source_reference_ids,
        )


@dataclass(frozen=True)
class ExpertCompositionAssessment(StrictContract):
    """Canonical complete partition produced by pure composition reduction."""

    assessment_id: str
    composition_plan: ExpertCompositionPlan
    disposition: ExpertCompositionDisposition
    applicable_source_reference_ids: tuple[str, ...]
    already_present_source_reference_ids: tuple[str, ...]
    conflicts: tuple[ExpertCompositionConflict, ...]
    stable_authority_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-composition-assessment"
    IDENTITY_FIELD: ClassVar[str] = "assessment_id"

    def _validate(self) -> None:
        if type(self.composition_plan) is not ExpertCompositionPlan:
            raise ExpertCompositionContractError(
                "composition assessment requires one exact plan"
            )
        for values, name in (
            (
                self.applicable_source_reference_ids,
                "composition applicable source references",
            ),
            (
                self.already_present_source_reference_ids,
                "composition already-present source references",
            ),
        ):
            _require_sorted_namespaced_content_ids(
                values,
                "expert-composition-source-reference",
                name,
                required=False,
            )
        if type(self.conflicts) is not tuple or any(
            type(conflict) is not ExpertCompositionConflict
            for conflict in self.conflicts
        ):
            raise ExpertCompositionContractError(
                "composition assessment conflicts are not typed"
            )
        conflict_keys = tuple(conflict.canonical_key for conflict in self.conflicts)
        if conflict_keys != tuple(sorted(set(conflict_keys))):
            raise ExpertCompositionContractError(
                "composition assessment conflicts must be canonical and unique"
            )
        applicable = set(self.applicable_source_reference_ids)
        already_present = set(self.already_present_source_reference_ids)
        conflict_sources = {
            source_reference_id
            for conflict in self.conflicts
            for source_reference_id in conflict.source_reference_ids
        }
        if (
            applicable & already_present
            or applicable & conflict_sources
            or already_present & conflict_sources
        ):
            raise ExpertCompositionContractError(
                "composition source classifications must be disjoint"
            )
        if applicable | already_present | conflict_sources != set(
            self.composition_plan.source_reference_ids
        ):
            raise ExpertCompositionContractError(
                "composition assessment must classify every planned source exactly"
            )
        sources_by_id = {
            source.source_reference_id: source
            for source in self.composition_plan.sources
        }
        if any(
            sources_by_id[source_reference_id].change_kind
            is not CandidateChangeKind.CAPABILITY
            for source_reference_id in applicable
        ):
            raise ExpertCompositionContractError(
                "only capability sources may be mechanically applicable"
            )
        architecture_sources = {
            source_reference_id
            for source_reference_id, source in sources_by_id.items()
            if source.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE
        }
        restructure_conflict_sources = {
            source_reference_id
            for conflict in self.conflicts
            if conflict.kind in _RESTRUCTURE_CONFLICT_KINDS
            for source_reference_id in conflict.source_reference_ids
        }
        if not (architecture_sources - already_present).issubset(
            restructure_conflict_sources
        ):
            raise ExpertCompositionContractError(
                "architecture source requires a restructure conflict"
            )
        if any(
            conflict.subject_kind is ExpertCompositionConflictSubjectKind.AUTHORITY
            and not set(conflict.subjects).issubset(
                self.composition_plan.stable_authority_ids
            )
            for conflict in self.conflicts
        ):
            raise ExpertCompositionContractError(
                "composition conflict cites authority outside its plan"
            )
        conflict_kinds = {conflict.kind for conflict in self.conflicts}
        has_restructure_conflict = bool(conflict_kinds & _RESTRUCTURE_CONFLICT_KINDS)
        if self.disposition is ExpertCompositionDisposition.CLEAN:
            valid_shape = bool(applicable) and not self.conflicts
        elif self.disposition is ExpertCompositionDisposition.ALREADY_PRESENT:
            valid_shape = (
                not applicable
                and already_present == set(self.composition_plan.source_reference_ids)
                and not self.conflicts
            )
        elif self.disposition is ExpertCompositionDisposition.CONFLICTED:
            valid_shape = bool(self.conflicts) and not has_restructure_conflict
        else:
            valid_shape = bool(self.conflicts) and has_restructure_conflict
        if not valid_shape:
            raise ExpertCompositionContractError(
                "composition assessment disposition contradicts its evidence"
            )
        _require_sorted_content_ids(
            self.stable_authority_ids,
            "composition assessment stable authorities",
        )
        expected_authorities = {
            self.composition_plan.composition_plan_id,
            *self.composition_plan.stable_authority_ids,
            *(conflict.conflict_id for conflict in self.conflicts),
        }
        if set(self.stable_authority_ids) != expected_authorities:
            raise ExpertCompositionContractError(
                "composition assessment stable authority closure is not exact"
            )


@dataclass(frozen=True)
class ExpertCompositionMaterialization(StrictContract):
    """Exact typed successor produced by one clean deterministic assessment."""

    materialization_id: str
    composition_assessment: ExpertCompositionAssessment
    source_base_tree: ExpertSourceTreeManifest
    patch: ExpertCandidatePatch
    source_tree: ExpertSourceTreeManifest
    repository_map: ExpertRepositoryMap
    module_contracts: tuple[ExpertModuleContract, ...]
    semantic_book_digest: str
    stable_authority_ids: tuple[str, ...]

    CONTENT_NAMESPACE: ClassVar[str] = "expert-composition-materialization"
    IDENTITY_FIELD: ClassVar[str] = "materialization_id"

    def _validate(self) -> None:
        if (
            type(self.composition_assessment) is not ExpertCompositionAssessment
            or type(self.source_base_tree) is not ExpertSourceTreeManifest
            or type(self.patch) is not ExpertCandidatePatch
            or type(self.source_tree) is not ExpertSourceTreeManifest
            or type(self.repository_map) is not ExpertRepositoryMap
            or type(self.module_contracts) is not tuple
            or any(
                type(module) is not ExpertModuleContract
                for module in self.module_contracts
            )
        ):
            raise ExpertCompositionContractError(
                "composition materialization requires exact typed authorities"
            )
        assessment = self.composition_assessment
        plan = assessment.composition_plan
        if assessment.disposition is not ExpertCompositionDisposition.CLEAN:
            raise ExpertCompositionContractError(
                "composition materialization requires a clean assessment"
            )
        if (
            self.source_base_tree.tree_hash != plan.current_base.source_tree_hash
            or self.patch.source_base_tree_hash != self.source_base_tree.tree_hash
            or self.patch.candidate_tree_hash != self.source_tree.tree_hash
            or self.repository_map.scope_contract_id
            != plan.scope_contract.scope_contract_id
        ):
            raise ExpertCompositionContractError(
                "composition materialization differs from its plan or source tree"
            )
        source_base_files = {
            descriptor.relative_path: descriptor
            for descriptor in self.source_base_tree.files
        }
        source_files = {
            descriptor.relative_path: descriptor
            for descriptor in self.source_tree.files
        }
        expected_changes = tuple(
            ExpertCandidatePatchChange(
                relative_path=path,
                before=source_base_files.get(path),
                after=source_files.get(path),
            )
            for path in sorted(set(source_base_files) | set(source_files))
            if source_base_files.get(path) != source_files.get(path)
        )
        if self.patch.changes != expected_changes:
            raise ExpertCompositionContractError(
                "composition materialization patch is not the exact tree transform"
            )
        module_contract_ids = tuple(
            sorted(module.module_contract_id for module in self.module_contracts)
        )
        validate_expert_repository_topology(
            self.repository_map,
            self.module_contracts,
            validation_error_type=ExpertCompositionContractError,
        )
        _require_digest(
            self.semantic_book_digest,
            "composition materialization semantic book",
        )
        expected_book = compile_expert_semantic_book(
            plan.scope_contract,
            self.repository_map,
            self.module_contracts,
        )
        if self.semantic_book_digest != expert_semantic_book_digest(expected_book):
            raise ExpertCompositionContractError(
                "composition materialization semantic book differs from its topology"
            )
        expected_controls = {
            EXPERT_BOOK_PATH: expected_book,
            EXPERT_REPOSITORY_MAP_PATH: self.repository_map.to_json_bytes(),
            **{
                expert_module_contract_path(module.module_contract_id): (
                    module.to_json_bytes()
                )
                for module in self.module_contracts
            },
        }
        if any(
            path not in source_files
            or source_files[path].digest != tree_or_blob_digest(payload)
            or source_files[path].size != len(payload)
            or source_files[path].mode != "100644"
            for path, payload in expected_controls.items()
        ):
            raise ExpertCompositionContractError(
                "composition materialization generated controls differ from topology"
            )
        validate_expert_tree_ownership(
            self.repository_map,
            self.module_contracts,
            source_files,
            validation_error_type=ExpertCompositionContractError,
        )
        _require_sorted_content_ids(
            self.stable_authority_ids,
            "composition materialization stable authorities",
        )
        expected_authorities = {
            assessment.assessment_id,
            *assessment.stable_authority_ids,
            self.source_base_tree.source_tree_manifest_id,
            self.patch.patch_id,
            self.source_tree.source_tree_manifest_id,
            self.repository_map.repository_map_id,
            *module_contract_ids,
        }
        if set(self.stable_authority_ids) != expected_authorities:
            raise ExpertCompositionContractError(
                "composition materialization stable authority closure is not exact"
            )
