"""Pure deterministic composition of approved expert candidate effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping

from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import (
    CandidateChangeKind,
    ExpertCandidatePatch,
    ExpertCandidatePatchChange,
    ExpertCapabilityNode,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertSourceTreeManifest,
    SourceFileDescriptor,
)
from kapso.cross_run.expert.candidate_context import (
    ExpertCandidateValidationContext,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_control_paths,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.composition_base import (
    ExpertCompositionBaseClosure,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionAssessment,
    ExpertCompositionConflict,
    ExpertCompositionConflictKind,
    ExpertCompositionConflictSubjectKind,
    ExpertCompositionDisposition,
    ExpertCompositionMaterialization,
    ExpertCompositionPlan,
    ExpertCompositionSourceReference,
)
from kapso.cross_run.expert.topology import (
    validate_expert_repository_topology,
    validate_expert_tree_ownership,
)


class ExpertCompositionError(ValueError):
    """Composition input closure or deterministic reduction is invalid."""


@dataclass(frozen=True)
class ExpertCompositionReduction:
    """One pure assessment and its optional exact clean successor bytes."""

    assessment: ExpertCompositionAssessment
    materialization: ExpertCompositionMaterialization | None
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if type(self.assessment) is not ExpertCompositionAssessment or not isinstance(
            self.source_contents,
            Mapping,
        ):
            raise ExpertCompositionError(
                "composition reduction requires exact assessment and source bytes"
            )
        contents = MappingProxyType(dict(self.source_contents))
        object.__setattr__(self, "source_contents", contents)
        if self.assessment.disposition is ExpertCompositionDisposition.CLEAN:
            if (
                type(self.materialization) is not ExpertCompositionMaterialization
                or self.materialization.composition_assessment != self.assessment
            ):
                raise ExpertCompositionError(
                    "clean composition reduction lacks its exact materialization"
                )
            descriptors = {
                descriptor.relative_path: descriptor
                for descriptor in self.materialization.source_tree.files
            }
            if set(contents) != set(descriptors):
                raise ExpertCompositionError(
                    "composition source bytes differ from materialized path closure"
                )
            for path, descriptor in descriptors.items():
                payload = contents[path]
                if (
                    type(payload) is not bytes
                    or len(payload) != descriptor.size
                    or tree_or_blob_digest(payload) != descriptor.digest
                ):
                    raise ExpertCompositionError(
                        f"composition source bytes differ from descriptor: {path}"
                    )
        elif self.materialization is not None or contents:
            raise ExpertCompositionError(
                "non-clean composition reduction cannot materialize source"
            )


@dataclass(frozen=True)
class ExpertCompositionReductionSource:
    """Stable source closure sufficient to replay the pure reducer."""

    source_reference: ExpertCompositionSourceReference
    validation_context: ExpertCandidateValidationContext
    patch: ExpertCandidatePatch
    candidate_tree: ExpertSourceTreeManifest
    repository_map: ExpertRepositoryMap
    module_contracts: tuple[ExpertModuleContract, ...]
    candidate_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if (
            type(self.source_reference) is not ExpertCompositionSourceReference
            or type(self.validation_context) is not ExpertCandidateValidationContext
            or type(self.patch) is not ExpertCandidatePatch
            or type(self.candidate_tree) is not ExpertSourceTreeManifest
            or type(self.repository_map) is not ExpertRepositoryMap
            or type(self.module_contracts) is not tuple
            or any(
                type(module) is not ExpertModuleContract
                for module in self.module_contracts
            )
            or not isinstance(self.candidate_contents, Mapping)
        ):
            raise ExpertCompositionError(
                "composition reduction source requires exact typed records"
            )
        contents = MappingProxyType(dict(self.candidate_contents))
        object.__setattr__(self, "candidate_contents", contents)
        source = self.source_reference
        parent_release = self.validation_context.parent_release
        parent_map = self.validation_context.parent_repository_map
        module_contract_ids = tuple(
            sorted(module.module_contract_id for module in self.module_contracts)
        )
        descriptors = {
            descriptor.relative_path: descriptor
            for descriptor in self.candidate_tree.files
        }
        if (
            parent_release is None
            or parent_map is None
            or source.validation_context_ref
            != self.validation_context.validation_context_id
            or source.scope_contract_id
            != self.validation_context.scope_contract.scope_contract_id
            or source.parent_release_id != parent_release.release_id
            or source.parent_repository_map_id != parent_map.repository_map_id
            or source.parent_tree_hash != self.validation_context.parent_tree_hash
            or source.candidate_tree_hash != self.candidate_tree.tree_hash
            or source.patch_id != self.patch.patch_id
            or source.patch_digest != tree_or_blob_digest(self.patch.to_json_bytes())
            or source.proposed_repository_map_id
            != self.repository_map.repository_map_id
            or source.module_contract_ids != module_contract_ids
            or set(contents) != set(descriptors)
        ):
            raise ExpertCompositionError(
                "composition reduction source differs from its stable reference"
            )
        for path, descriptor in descriptors.items():
            payload = contents[path]
            if (
                type(payload) is not bytes
                or len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.digest
            ):
                raise ExpertCompositionError(
                    f"composition reduction source bytes differ: {path}"
                )


@dataclass(frozen=True)
class _ModuleEffect:
    module_id: str
    before: ExpertModuleContract
    after: ExpertModuleContract


@dataclass(frozen=True)
class _SourceEffect:
    source: ExpertCompositionReductionSource
    editable_changes: tuple[ExpertCandidatePatchChange, ...]
    module_effects: tuple[_ModuleEffect, ...]
    already_present: bool
    individual_conflicts: tuple[ExpertCompositionConflict, ...]

    @property
    def source_reference_id(self) -> str:
        return self.source.source_reference.source_reference_id


@dataclass(frozen=True)
class _ComposedTree:
    patch: ExpertCandidatePatch
    source_tree: ExpertSourceTreeManifest
    repository_map: ExpertRepositoryMap
    module_contracts: tuple[ExpertModuleContract, ...]
    book_digest: str
    source_contents: Mapping[str, bytes]


class ExpertCompositionReducer:
    """Conservative three-way reducer over immutable checked source closures."""

    def __init__(
        self,
        *,
        candidate_entry_limit: int,
        candidate_byte_limit: int,
    ) -> None:
        if (
            type(candidate_entry_limit) is not int
            or candidate_entry_limit <= 0
            or type(candidate_byte_limit) is not int
            or candidate_byte_limit <= 0
        ):
            raise ExpertCompositionError(
                "composition candidate limits must be positive integers"
            )
        self.candidate_entry_limit = candidate_entry_limit
        self.candidate_byte_limit = candidate_byte_limit

    def reduce(
        self,
        *,
        plan: ExpertCompositionPlan,
        current_base: ExpertCompositionBaseClosure,
        sources: tuple[ExpertCompositionReductionSource, ...],
    ) -> ExpertCompositionReduction:
        self._require_exact_runtime_closure(
            plan=plan,
            current_base=current_base,
            sources=sources,
        )
        current_files = {
            descriptor.relative_path: descriptor
            for descriptor in current_base.source_files
        }
        current_modules = {
            module.module_id: module for module in current_base.module_contracts
        }
        effects = tuple(
            self._classify_source(
                source=source,
                current_base=current_base,
                current_files=current_files,
                current_modules=current_modules,
            )
            for source in sources
        )
        conflicts = [
            conflict for effect in effects for conflict in effect.individual_conflicts
        ]
        cross_source_conflicts = self._cross_source_conflicts(effects)
        conflicts.extend(cross_source_conflicts)
        conflicts = list(self._canonical_conflicts(tuple(conflicts)))
        conflict_source_ids = {
            source_reference_id
            for conflict in conflicts
            for source_reference_id in conflict.source_reference_ids
        }
        already_present_ids = tuple(
            sorted(
                effect.source_reference_id
                for effect in effects
                if effect.already_present
            )
        )
        applicable_ids = tuple(
            sorted(
                effect.source_reference_id
                for effect in effects
                if not effect.already_present
                and effect.source_reference_id not in conflict_source_ids
            )
        )
        if conflicts:
            assessment = self._assessment(
                plan=plan,
                applicable_source_reference_ids=applicable_ids,
                already_present_source_reference_ids=already_present_ids,
                conflicts=tuple(conflicts),
            )
            return ExpertCompositionReduction(
                assessment=assessment,
                materialization=None,
                source_contents={},
            )
        if not applicable_ids:
            assessment = self._assessment(
                plan=plan,
                applicable_source_reference_ids=(),
                already_present_source_reference_ids=already_present_ids,
                conflicts=(),
            )
            return ExpertCompositionReduction(
                assessment=assessment,
                materialization=None,
                source_contents={},
            )
        applicable_effects = tuple(
            effect
            for effect in effects
            if effect.source_reference_id in set(applicable_ids)
        )
        composed = self._compose_tree(
            plan=plan,
            current_base=current_base,
            effects=applicable_effects,
        )
        incompatibility_conflicts = self._capability_incompatibility_conflicts(
            module_contracts=composed.module_contracts,
            effects=applicable_effects,
        )
        if incompatibility_conflicts:
            incompatible_source_ids = {
                source_reference_id
                for conflict in incompatibility_conflicts
                for source_reference_id in conflict.source_reference_ids
            }
            assessment = self._assessment(
                plan=plan,
                applicable_source_reference_ids=tuple(
                    sorted(set(applicable_ids) - incompatible_source_ids)
                ),
                already_present_source_reference_ids=already_present_ids,
                conflicts=incompatibility_conflicts,
            )
            return ExpertCompositionReduction(
                assessment=assessment,
                materialization=None,
                source_contents={},
            )
        # Module resource bounds are opaque domain contracts and capability
        # candidates cannot rewrite them. The reducer therefore enforces only
        # the domain-neutral aggregate source-tree limits instead of inventing
        # arithmetic across unrelated resource schemas.
        if (
            len(composed.source_tree.files) > self.candidate_entry_limit
            or sum(descriptor.size for descriptor in composed.source_tree.files)
            > self.candidate_byte_limit
        ):
            resource_conflict = self._conflict(
                kind=ExpertCompositionConflictKind.RESOURCE_ENVELOPE,
                subject_kind=ExpertCompositionConflictSubjectKind.CAPABILITY,
                subjects=tuple(
                    sorted(
                        {
                            module_effect.module_id
                            for effect in applicable_effects
                            for module_effect in effect.module_effects
                        }
                    )
                ),
                source_reference_ids=applicable_ids,
            )
            assessment = self._assessment(
                plan=plan,
                applicable_source_reference_ids=(),
                already_present_source_reference_ids=already_present_ids,
                conflicts=(resource_conflict,),
            )
            return ExpertCompositionReduction(
                assessment=assessment,
                materialization=None,
                source_contents={},
            )
        assessment = self._assessment(
            plan=plan,
            applicable_source_reference_ids=applicable_ids,
            already_present_source_reference_ids=already_present_ids,
            conflicts=(),
        )
        parent_tree = ExpertSourceTreeManifest.mint(
            tree_hash=current_base.reference.source_tree_hash,
            files=current_base.source_files,
        )
        materialization_authorities = tuple(
            sorted(
                {
                    assessment.assessment_id,
                    *assessment.stable_authority_ids,
                    parent_tree.source_tree_manifest_id,
                    composed.patch.patch_id,
                    composed.source_tree.source_tree_manifest_id,
                    composed.repository_map.repository_map_id,
                    *(
                        module.module_contract_id
                        for module in composed.module_contracts
                    ),
                }
            )
        )
        materialization = ExpertCompositionMaterialization.mint(
            composition_assessment=assessment,
            parent_tree=parent_tree,
            patch=composed.patch,
            source_tree=composed.source_tree,
            repository_map=composed.repository_map,
            module_contracts=composed.module_contracts,
            semantic_book_digest=composed.book_digest,
            stable_authority_ids=materialization_authorities,
        )
        return ExpertCompositionReduction(
            assessment=assessment,
            materialization=materialization,
            source_contents=composed.source_contents,
        )

    @staticmethod
    def _require_exact_runtime_closure(
        *,
        plan: ExpertCompositionPlan,
        current_base: ExpertCompositionBaseClosure,
        sources: tuple[ExpertCompositionReductionSource, ...],
    ) -> None:
        if (
            type(plan) is not ExpertCompositionPlan
            or type(current_base) is not ExpertCompositionBaseClosure
            or type(sources) is not tuple
            or any(
                type(source) is not ExpertCompositionReductionSource
                for source in sources
            )
        ):
            raise ExpertCompositionError(
                "composition requires exact typed runtime inputs"
            )
        if (
            current_base.reference != plan.current_base
            or current_base.scope_contract != plan.scope_contract
        ):
            raise ExpertCompositionError(
                "composition base differs from its exact plan reference"
            )
        source_references = tuple(source.source_reference for source in sources)
        if source_references != plan.sources:
            raise ExpertCompositionError(
                "approved composition sources differ from the canonical plan"
            )

    def _classify_source(
        self,
        *,
        source: ExpertCompositionReductionSource,
        current_base: ExpertCompositionBaseClosure,
        current_files: Mapping[str, SourceFileDescriptor],
        current_modules: Mapping[str, ExpertModuleContract],
    ) -> _SourceEffect:
        source_reference = source.source_reference
        if source_reference.change_kind is CandidateChangeKind.REPOSITORY_ARCHITECTURE:
            already_present = (
                source_reference.candidate_tree_hash
                == current_base.reference.source_tree_hash
                and source.repository_map == current_base.repository_map
                and source.module_contracts == current_base.module_contracts
            )
            conflicts = (
                ()
                if already_present
                else (
                    self._conflict(
                        kind=ExpertCompositionConflictKind.ARCHITECTURE,
                        subject_kind=(ExpertCompositionConflictSubjectKind.REPOSITORY),
                        subjects=("repository",),
                        source_reference_ids=(source_reference.source_reference_id,),
                    ),
                )
            )
            return _SourceEffect(
                source=source,
                editable_changes=(),
                module_effects=(),
                already_present=already_present,
                individual_conflicts=conflicts,
            )
        if source_reference.change_kind is not CandidateChangeKind.CAPABILITY:
            raise ExpertCompositionError(
                "composition source uses an unknown candidate change kind"
            )
        parent_map = source.validation_context.parent_repository_map
        if parent_map is None:
            raise ExpertCompositionError(
                "capability composition source lacks parent topology"
            )
        parent_modules = {
            module.module_id: module
            for module in source.validation_context.parent_module_contracts
        }
        candidate_modules = {
            module.module_id: module for module in source.module_contracts
        }
        if set(parent_modules) != set(candidate_modules):
            raise ExpertCompositionError(
                "capability composition source changes module identity"
            )
        module_effects = tuple(
            _ModuleEffect(
                module_id=module_id,
                before=parent_modules[module_id],
                after=candidate_modules[module_id],
            )
            for module_id in sorted(parent_modules)
            if parent_modules[module_id] != candidate_modules[module_id]
        )
        parent_controls = set(
            expert_control_paths(source.validation_context.parent_module_contracts)
        )
        candidate_controls = set(expert_control_paths(source.module_contracts))
        editable_changes = tuple(
            change
            for change in source.patch.changes
            if change.relative_path not in parent_controls | candidate_controls
        )
        if not editable_changes or not module_effects:
            raise ExpertCompositionError(
                "capability composition source has no complete editable effect"
            )
        path_states = tuple(
            self._three_way_state(
                current_files.get(change.relative_path),
                change.before,
                change.after,
            )
            for change in editable_changes
        )
        module_states = tuple(
            self._three_way_state(
                current_modules.get(effect.module_id),
                effect.before,
                effect.after,
            )
            for effect in module_effects
        )
        already_present = all(state == "after" for state in path_states) and all(
            state == "after" for state in module_states
        )
        if already_present:
            return _SourceEffect(
                source=source,
                editable_changes=editable_changes,
                module_effects=module_effects,
                already_present=True,
                individual_conflicts=(),
            )
        conflicts: list[ExpertCompositionConflict] = []
        for change, state in zip(editable_changes, path_states):
            if state == "conflict":
                conflicts.append(
                    self._conflict(
                        kind=ExpertCompositionConflictKind.CURRENT_PATH,
                        subject_kind=ExpertCompositionConflictSubjectKind.PATH,
                        subjects=(change.relative_path,),
                        source_reference_ids=(source_reference.source_reference_id,),
                    )
                )
        for effect, state in zip(module_effects, module_states):
            if state == "conflict":
                conflicts.append(
                    self._conflict(
                        kind=ExpertCompositionConflictKind.MODULE,
                        subject_kind=(ExpertCompositionConflictSubjectKind.CAPABILITY),
                        subjects=(effect.module_id,),
                        source_reference_ids=(source_reference.source_reference_id,),
                    )
                )
        conflicts.extend(
            self._topology_conflicts(
                source_reference_id=source_reference.source_reference_id,
                parent_map=parent_map,
                current_map=current_base.repository_map,
            )
        )
        conflicts.extend(
            self._prospective_path_prefix_conflicts(
                source_reference_id=source_reference.source_reference_id,
                current_base=current_base,
                editable_changes=editable_changes,
            )
        )
        return _SourceEffect(
            source=source,
            editable_changes=editable_changes,
            module_effects=module_effects,
            already_present=False,
            individual_conflicts=self._canonical_conflicts(tuple(conflicts)),
        )

    @staticmethod
    def _three_way_state(current, before, after) -> str:
        if current == before:
            return "before"
        if current == after:
            return "after"
        return "conflict"

    def _topology_conflicts(
        self,
        *,
        source_reference_id: str,
        parent_map: ExpertRepositoryMap,
        current_map: ExpertRepositoryMap,
    ) -> tuple[ExpertCompositionConflict, ...]:
        conflicts: list[ExpertCompositionConflict] = []
        parent_nodes = {
            node.capability_id: node for node in parent_map.capability_nodes
        }
        current_nodes = {
            node.capability_id: node for node in current_map.capability_nodes
        }
        changed_capabilities = tuple(
            sorted(
                capability_id
                for capability_id in set(parent_nodes) | set(current_nodes)
                if capability_id not in parent_nodes
                or capability_id not in current_nodes
                or (
                    parent_nodes[capability_id].owned_paths,
                    parent_nodes[capability_id].task_family_bindings,
                )
                != (
                    current_nodes[capability_id].owned_paths,
                    current_nodes[capability_id].task_family_bindings,
                )
            )
        )
        if changed_capabilities:
            conflicts.append(
                self._conflict(
                    kind=ExpertCompositionConflictKind.TOPOLOGY,
                    subject_kind=ExpertCompositionConflictSubjectKind.CAPABILITY,
                    subjects=changed_capabilities,
                    source_reference_ids=(source_reference_id,),
                )
            )
        if parent_map.dependency_edges != current_map.dependency_edges:
            dependency_capabilities = tuple(
                sorted(
                    {
                        capability_id
                        for edge in (
                            *parent_map.dependency_edges,
                            *current_map.dependency_edges,
                        )
                        for capability_id in (
                            edge.source_capability_id,
                            edge.target_capability_id,
                        )
                    }
                )
            )
            conflicts.append(
                self._conflict(
                    kind=ExpertCompositionConflictKind.DEPENDENCY_GRAPH,
                    subject_kind=ExpertCompositionConflictSubjectKind.CAPABILITY,
                    subjects=dependency_capabilities,
                    source_reference_ids=(source_reference_id,),
                )
            )
        if parent_map.task_adapter_boundary != current_map.task_adapter_boundary:
            adapter_paths = tuple(
                sorted(
                    {
                        parent_map.task_adapter_boundary.adapter_mount_path,
                        current_map.task_adapter_boundary.adapter_mount_path,
                        *parent_map.task_adapter_boundary.interface_entrypoint_refs,
                        *current_map.task_adapter_boundary.interface_entrypoint_refs,
                    }
                )
            )
            conflicts.append(
                self._conflict(
                    kind=ExpertCompositionConflictKind.ADAPTER_BOUNDARY,
                    subject_kind=ExpertCompositionConflictSubjectKind.PATH,
                    subjects=adapter_paths,
                    source_reference_ids=(source_reference_id,),
                )
            )
        if (
            parent_map.validation_entrypoints != current_map.validation_entrypoints
            or parent_map.architecture_invariants != current_map.architecture_invariants
        ):
            conflicts.append(
                self._conflict(
                    kind=ExpertCompositionConflictKind.ARCHITECTURE,
                    subject_kind=ExpertCompositionConflictSubjectKind.REPOSITORY,
                    subjects=("repository",),
                    source_reference_ids=(source_reference_id,),
                )
            )
        return self._canonical_conflicts(tuple(conflicts))

    def _prospective_path_prefix_conflicts(
        self,
        *,
        source_reference_id: str,
        current_base: ExpertCompositionBaseClosure,
        editable_changes: tuple[ExpertCandidatePatchChange, ...],
    ) -> tuple[ExpertCompositionConflict, ...]:
        current_controls = set(expert_control_paths(current_base.module_contracts))
        prospective_paths = {
            descriptor.relative_path
            for descriptor in current_base.source_files
            if descriptor.relative_path not in current_controls
        }
        for change in editable_changes:
            prospective_paths.discard(change.relative_path)
            if change.after is not None:
                prospective_paths.add(change.relative_path)
        conflicts = tuple(
            self._conflict(
                kind=ExpertCompositionConflictKind.PATH_PREFIX,
                subject_kind=ExpertCompositionConflictSubjectKind.PATH,
                subjects=paths,
                source_reference_ids=(source_reference_id,),
            )
            for paths in self._path_prefix_pairs(tuple(sorted(prospective_paths)))
        )
        return self._canonical_conflicts(conflicts)

    def _cross_source_conflicts(
        self,
        effects: tuple[_SourceEffect, ...],
    ) -> tuple[ExpertCompositionConflict, ...]:
        active_effects = tuple(
            effect
            for effect in effects
            if not effect.already_present and not effect.individual_conflicts
        )
        source_ids_by_path: dict[str, set[str]] = {}
        source_ids_by_module: dict[str, set[str]] = {}
        for effect in active_effects:
            for change in effect.editable_changes:
                source_ids_by_path.setdefault(change.relative_path, set()).add(
                    effect.source_reference_id
                )
            for module_effect in effect.module_effects:
                source_ids_by_module.setdefault(module_effect.module_id, set()).add(
                    effect.source_reference_id
                )
        conflicts: list[ExpertCompositionConflict] = []
        for path, source_ids in sorted(source_ids_by_path.items()):
            if len(source_ids) > 1:
                conflicts.append(
                    self._conflict(
                        kind=ExpertCompositionConflictKind.PATH_OVERLAP,
                        subject_kind=ExpertCompositionConflictSubjectKind.PATH,
                        subjects=(path,),
                        source_reference_ids=tuple(sorted(source_ids)),
                    )
                )
        paths = tuple(sorted(source_ids_by_path))
        for shorter_path, longer_path in self._path_prefix_pairs(paths):
            source_ids = (
                source_ids_by_path[shorter_path] | source_ids_by_path[longer_path]
            )
            if len(source_ids) > 1:
                conflicts.append(
                    self._conflict(
                        kind=ExpertCompositionConflictKind.PATH_PREFIX,
                        subject_kind=ExpertCompositionConflictSubjectKind.PATH,
                        subjects=(shorter_path, longer_path),
                        source_reference_ids=tuple(sorted(source_ids)),
                    )
                )
        for module_id, source_ids in sorted(source_ids_by_module.items()):
            if len(source_ids) > 1:
                conflicts.append(
                    self._conflict(
                        kind=ExpertCompositionConflictKind.MODULE,
                        subject_kind=(ExpertCompositionConflictSubjectKind.CAPABILITY),
                        subjects=(module_id,),
                        source_reference_ids=tuple(sorted(source_ids)),
                    )
                )
        return self._canonical_conflicts(tuple(conflicts))

    def _capability_incompatibility_conflicts(
        self,
        *,
        module_contracts: tuple[ExpertModuleContract, ...],
        effects: tuple[_SourceEffect, ...],
    ) -> tuple[ExpertCompositionConflict, ...]:
        modules = {module.module_id: module for module in module_contracts}
        source_ids_by_module: dict[str, set[str]] = {}
        for effect in effects:
            for module_effect in effect.module_effects:
                source_ids_by_module.setdefault(module_effect.module_id, set()).add(
                    effect.source_reference_id
                )
        incompatible_pairs = {
            tuple(sorted((module.module_id, incompatible_module_id)))
            for module in module_contracts
            for incompatible_module_id in module.incompatible_capability_ids
            if incompatible_module_id in modules
        }
        # One approved candidate may change either or both sides as a tested unit.
        # Composition conflicts only when separate sources independently change
        # both sides of a declared incompatibility.
        conflicts = tuple(
            self._conflict(
                kind=ExpertCompositionConflictKind.CAPABILITY_INCOMPATIBILITY,
                subject_kind=ExpertCompositionConflictSubjectKind.CAPABILITY,
                subjects=pair,
                source_reference_ids=tuple(
                    sorted(
                        source_ids_by_module.get(pair[0], set())
                        | source_ids_by_module.get(pair[1], set())
                    )
                ),
            )
            for pair in sorted(incompatible_pairs)
            if source_ids_by_module.get(pair[0])
            and source_ids_by_module.get(pair[1])
            and len(source_ids_by_module[pair[0]] | source_ids_by_module[pair[1]]) > 1
        )
        return self._canonical_conflicts(conflicts)

    @staticmethod
    def _path_prefix_pairs(
        paths: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (shorter_path, longer_path)
            for shorter_position, shorter_path in enumerate(paths)
            for longer_path in paths[shorter_position + 1 :]
            if PurePosixPath(shorter_path) in PurePosixPath(longer_path).parents
            or PurePosixPath(longer_path) in PurePosixPath(shorter_path).parents
        )

    def _compose_tree(
        self,
        *,
        plan: ExpertCompositionPlan,
        current_base: ExpertCompositionBaseClosure,
        effects: tuple[_SourceEffect, ...],
    ) -> _ComposedTree:
        current_controls = set(expert_control_paths(current_base.module_contracts))
        current_files = {
            descriptor.relative_path: descriptor
            for descriptor in current_base.source_files
        }
        composed_files = {
            path: descriptor
            for path, descriptor in current_files.items()
            if path not in current_controls
        }
        composed_contents = {
            path: payload
            for path, payload in current_base.source_contents.items()
            if path not in current_controls
        }
        composed_modules = {
            module.module_id: module for module in current_base.module_contracts
        }
        for effect in effects:
            candidate_contents = effect.source.candidate_contents
            for change in effect.editable_changes:
                current = composed_files.get(change.relative_path)
                if current == change.after:
                    continue
                if current != change.before:
                    raise ExpertCompositionError(
                        "clean composition path changed during pure reduction"
                    )
                if change.after is None:
                    composed_files.pop(change.relative_path)
                    composed_contents.pop(change.relative_path)
                else:
                    payload = candidate_contents[change.relative_path]
                    if (
                        len(payload) != change.after.size
                        or tree_or_blob_digest(payload) != change.after.digest
                    ):
                        raise ExpertCompositionError(
                            "composition source bytes differ from patch descriptor"
                        )
                    composed_files[change.relative_path] = change.after
                    composed_contents[change.relative_path] = payload
            for module_effect in effect.module_effects:
                current_module = composed_modules.get(module_effect.module_id)
                if current_module == module_effect.after:
                    continue
                if current_module != module_effect.before:
                    raise ExpertCompositionError(
                        "clean composition module changed during pure reduction"
                    )
                composed_modules[module_effect.module_id] = module_effect.after
        modules = tuple(
            composed_modules[module_id] for module_id in sorted(composed_modules)
        )
        module_refs = {
            module.module_id: module.module_contract_id for module in modules
        }
        repository_map = ExpertRepositoryMap.mint(
            scope_contract_id=plan.scope_contract.scope_contract_id,
            capability_nodes=tuple(
                ExpertCapabilityNode(
                    capability_id=node.capability_id,
                    module_contract_ref=module_refs[node.capability_id],
                    owned_paths=node.owned_paths,
                    task_family_bindings=node.task_family_bindings,
                )
                for node in current_base.repository_map.capability_nodes
            ),
            dependency_edges=current_base.repository_map.dependency_edges,
            task_adapter_boundary=current_base.repository_map.task_adapter_boundary,
            validation_entrypoints=current_base.repository_map.validation_entrypoints,
            architecture_invariants=current_base.repository_map.architecture_invariants,
        )
        book = compile_expert_semantic_book(
            plan.scope_contract,
            repository_map,
            modules,
        )
        controls = {
            EXPERT_BOOK_PATH: book,
            EXPERT_REPOSITORY_MAP_PATH: repository_map.to_json_bytes(),
            **{
                expert_module_contract_path(module.module_contract_id): (
                    module.to_json_bytes()
                )
                for module in modules
            },
        }
        composed_contents.update(controls)
        descriptors = tuple(
            (
                composed_files[path]
                if path not in controls
                else SourceFileDescriptor(
                    relative_path=path,
                    digest=tree_or_blob_digest(payload),
                    mode="100644",
                    size=len(payload),
                )
            )
            for path, payload in sorted(composed_contents.items())
        )
        tree_hash = source_tree_digest(
            {
                descriptor.relative_path: (
                    descriptor.digest,
                    descriptor.mode,
                    descriptor.size,
                )
                for descriptor in descriptors
            }
        )
        source_tree = ExpertSourceTreeManifest.mint(
            tree_hash=tree_hash,
            files=descriptors,
        )
        composed_file_map = {
            descriptor.relative_path: descriptor for descriptor in descriptors
        }
        patch = ExpertCandidatePatch.mint(
            parent_tree_hash=current_base.reference.source_tree_hash,
            candidate_tree_hash=source_tree.tree_hash,
            changes=tuple(
                ExpertCandidatePatchChange(
                    relative_path=path,
                    before=current_files.get(path),
                    after=composed_file_map.get(path),
                )
                for path in sorted(set(current_files) | set(composed_file_map))
                if current_files.get(path) != composed_file_map.get(path)
            ),
        )
        validate_expert_repository_topology(
            repository_map,
            modules,
            validation_error_type=ExpertCompositionError,
        )
        validate_expert_tree_ownership(
            repository_map,
            modules,
            composed_file_map,
            validation_error_type=ExpertCompositionError,
        )
        return _ComposedTree(
            patch=patch,
            source_tree=source_tree,
            repository_map=repository_map,
            module_contracts=modules,
            book_digest=expert_semantic_book_digest(book),
            source_contents=MappingProxyType(dict(composed_contents)),
        )

    @staticmethod
    def _conflict(
        *,
        kind: ExpertCompositionConflictKind,
        subject_kind: ExpertCompositionConflictSubjectKind,
        subjects: tuple[str, ...],
        source_reference_ids: tuple[str, ...],
    ) -> ExpertCompositionConflict:
        return ExpertCompositionConflict.mint(
            kind=kind,
            subject_kind=subject_kind,
            subjects=tuple(sorted(set(subjects))),
            source_reference_ids=tuple(sorted(set(source_reference_ids))),
        )

    @staticmethod
    def _canonical_conflicts(
        conflicts: tuple[ExpertCompositionConflict, ...],
    ) -> tuple[ExpertCompositionConflict, ...]:
        by_key = {conflict.canonical_key: conflict for conflict in conflicts}
        return tuple(by_key[key] for key in sorted(by_key))

    @staticmethod
    def _assessment(
        *,
        plan: ExpertCompositionPlan,
        applicable_source_reference_ids: tuple[str, ...],
        already_present_source_reference_ids: tuple[str, ...],
        conflicts: tuple[ExpertCompositionConflict, ...],
    ) -> ExpertCompositionAssessment:
        if conflicts:
            restructure_kinds = {
                ExpertCompositionConflictKind.TOPOLOGY,
                ExpertCompositionConflictKind.CAPABILITY_LINEAGE,
                ExpertCompositionConflictKind.DEPENDENCY_GRAPH,
                ExpertCompositionConflictKind.ADAPTER_BOUNDARY,
                ExpertCompositionConflictKind.ARCHITECTURE,
            }
            disposition = (
                ExpertCompositionDisposition.REQUIRES_RESTRUCTURE
                if any(conflict.kind in restructure_kinds for conflict in conflicts)
                else ExpertCompositionDisposition.CONFLICTED
            )
        elif applicable_source_reference_ids:
            disposition = ExpertCompositionDisposition.CLEAN
        else:
            disposition = ExpertCompositionDisposition.ALREADY_PRESENT
        canonical_conflicts = ExpertCompositionReducer._canonical_conflicts(conflicts)
        stable_authority_ids = tuple(
            sorted(
                {
                    plan.composition_plan_id,
                    *plan.stable_authority_ids,
                    *(conflict.conflict_id for conflict in canonical_conflicts),
                }
            )
        )
        return ExpertCompositionAssessment.mint(
            composition_plan=plan,
            disposition=disposition,
            applicable_source_reference_ids=tuple(
                sorted(applicable_source_reference_ids)
            ),
            already_present_source_reference_ids=tuple(
                sorted(already_present_source_reference_ids)
            ),
            conflicts=canonical_conflicts,
            stable_authority_ids=stable_authority_ids,
        )
