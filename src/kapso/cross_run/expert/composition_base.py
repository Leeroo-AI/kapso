"""Pure verification of one stable expert composition base."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
    PublicationArtifactKind,
    SourceFileDescriptor,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.composition_contracts import (
    ExpertCompositionBaseReference,
)
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplayCurrentReleaseObservation,
)
from kapso.cross_run.expert.topology import (
    validate_expert_repository_topology,
    validate_expert_tree_ownership,
)
from kapso.cross_run.expert.triggers import ExpertParentTreeReceipt
from kapso.cross_run.github.materializer import SourceArchiveExtractionReceipt


class ExpertCompositionBaseError(ValueError):
    """A released expert tree cannot serve as an exact composition base."""


def expert_composition_base_security_subject_ids(
    closure: ExpertCompositionBaseClosure,
    current_observation: SourceReplayCurrentReleaseObservation,
) -> tuple[str, ...]:
    """Project the exact revocation closure of one authenticated current base."""

    if (
        type(closure) is not ExpertCompositionBaseClosure
        or type(current_observation) is not SourceReplayCurrentReleaseObservation
        or current_observation.scope_id != closure.scope_contract.scope_id
        or current_observation.release_id != closure.release_manifest.release_id
    ):
        raise ExpertCompositionBaseError(
            "base security projection requires one exact current base"
        )
    parent_receipt = closure.parent_tree_receipt
    return tuple(
        sorted(
            {
                closure.reference.base_reference_id,
                *closure.reference.stable_authority_ids,
                parent_receipt.parent_tree_receipt_id,
                parent_receipt.source_extraction_receipt.extraction_receipt_id,
                current_observation.observation_id,
                current_observation.publication_id,
                *current_observation.validation_closure_ids,
                *closure.release_manifest.dependency_closure_ids,
            }
        )
    )


def _module_contract_ids(
    module_contracts: tuple[ExpertModuleContract, ...],
) -> tuple[str, ...]:
    return tuple(sorted(module.module_contract_id for module in module_contracts))


def _mint_base_reference(
    *,
    scope_contract: ExpertScopeContract,
    release_manifest: ExpertBaseReleaseManifest,
    parent_tree_receipt: ExpertParentTreeReceipt,
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
) -> ExpertCompositionBaseReference:
    module_contract_ids = _module_contract_ids(module_contracts)
    stable_authority_ids = tuple(
        sorted(
            {
                release_manifest.release_id,
                scope_contract.scope_contract_id,
                repository_map.repository_map_id,
                *module_contract_ids,
            }
        )
    )
    return ExpertCompositionBaseReference.mint(
        release_id=release_manifest.release_id,
        scope_contract_id=scope_contract.scope_contract_id,
        scope_id=scope_contract.scope_id,
        source_tree_hash=parent_tree_receipt.parent_tree_hash,
        repository_map_id=repository_map.repository_map_id,
        module_contract_ids=module_contract_ids,
        semantic_book_digest=release_manifest.semantic_book_digest,
        release_configuration_fingerprint=(release_manifest.configuration_fingerprint),
        stable_authority_ids=stable_authority_ids,
    )


def _require_authority_types(
    *,
    scope_contract: ExpertScopeContract,
    release_manifest: ExpertBaseReleaseManifest,
    parent_tree_receipt: ExpertParentTreeReceipt,
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
) -> None:
    if (
        type(scope_contract) is not ExpertScopeContract
        or type(release_manifest) is not ExpertBaseReleaseManifest
        or type(parent_tree_receipt) is not ExpertParentTreeReceipt
        or type(repository_map) is not ExpertRepositoryMap
        or type(module_contracts) is not tuple
        or any(
            type(module_contract) is not ExpertModuleContract
            for module_contract in module_contracts
        )
    ):
        raise ExpertCompositionBaseError(
            "composition base requires exact typed release authorities"
        )


def _require_exact_types(
    *,
    reference: ExpertCompositionBaseReference,
    scope_contract: ExpertScopeContract,
    release_manifest: ExpertBaseReleaseManifest,
    parent_tree_receipt: ExpertParentTreeReceipt,
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
) -> None:
    if type(reference) is not ExpertCompositionBaseReference:
        raise ExpertCompositionBaseError(
            "composition base requires an exact stable reference"
        )
    _require_authority_types(
        scope_contract=scope_contract,
        release_manifest=release_manifest,
        parent_tree_receipt=parent_tree_receipt,
        repository_map=repository_map,
        module_contracts=module_contracts,
    )


def _require_release_joins(
    *,
    scope_contract: ExpertScopeContract,
    release_manifest: ExpertBaseReleaseManifest,
    parent_tree_receipt: ExpertParentTreeReceipt,
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
) -> None:
    module_contract_ids = _module_contract_ids(module_contracts)
    module_versions = {module.module_id: module.version for module in module_contracts}
    cache_receipt = parent_tree_receipt.cache_verification_receipt
    extraction_receipt = parent_tree_receipt.source_extraction_receipt
    source_archive_ref = release_manifest.source_archive_ref
    source_archive_digest = release_manifest.checksums[source_archive_ref]
    if (
        release_manifest.scope_contract_id != scope_contract.scope_contract_id
        or release_manifest.scope_id != scope_contract.scope_id
        or repository_map.scope_contract_id != scope_contract.scope_contract_id
    ):
        raise ExpertCompositionBaseError(
            "composition base release, map, and scope authority differ"
        )
    if (
        release_manifest.repository_map_ref != repository_map.repository_map_id
        or dict(release_manifest.module_versions) != module_versions
    ):
        raise ExpertCompositionBaseError(
            "composition base release differs from its exact topology"
        )
    if (
        parent_tree_receipt.release_id != release_manifest.release_id
        or parent_tree_receipt.repository_map_id != repository_map.repository_map_id
        or parent_tree_receipt.module_contract_ids != module_contract_ids
    ):
        raise ExpertCompositionBaseError(
            "composition base parent receipt differs from its release topology"
        )
    if (
        cache_receipt.artifact_kind is not PublicationArtifactKind.EXPERT_BASE_RELEASE
        or cache_receipt.artifact_id != release_manifest.release_id
        or cache_receipt.manifest_digest
        != tree_or_blob_digest(release_manifest.to_json_bytes())
        or cache_receipt.asset_digests.get(source_archive_ref) != source_archive_digest
    ):
        raise ExpertCompositionBaseError(
            "composition base cache receipt differs from its release"
        )
    if (
        extraction_receipt.artifact_id != release_manifest.release_id
        or extraction_receipt.source_archive_ref != source_archive_ref
        or extraction_receipt.source_archive_digest != source_archive_digest
        or extraction_receipt.source_tree_hash != parent_tree_receipt.parent_tree_hash
    ):
        raise ExpertCompositionBaseError(
            "composition base extraction differs from its release archive"
        )


def _require_source_byte_closure(
    *,
    parent_tree_receipt: ExpertParentTreeReceipt,
    source_contents: Mapping[str, bytes],
) -> dict[str, SourceFileDescriptor]:
    descriptors = parent_tree_receipt.source_extraction_receipt.source_tree_files
    files_by_path = {descriptor.relative_path: descriptor for descriptor in descriptors}
    if set(source_contents) != set(files_by_path):
        raise ExpertCompositionBaseError(
            "composition base bytes differ from the exact source-tree closure"
        )
    for path, descriptor in files_by_path.items():
        payload = source_contents[path]
        if (
            type(payload) is not bytes
            or len(payload) != descriptor.size
            or tree_or_blob_digest(payload) != descriptor.digest
        ):
            raise ExpertCompositionBaseError(
                f"composition base source bytes differ from descriptor: {path}"
            )
    return files_by_path


def _require_generated_controls(
    *,
    scope_contract: ExpertScopeContract,
    release_manifest: ExpertBaseReleaseManifest,
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
    source_contents: Mapping[str, bytes],
    source_files: Mapping[str, SourceFileDescriptor],
) -> None:
    expected_book = compile_expert_semantic_book(
        scope_contract,
        repository_map,
        module_contracts,
    )
    expected_controls = {
        EXPERT_BOOK_PATH: expected_book,
        EXPERT_REPOSITORY_MAP_PATH: repository_map.to_json_bytes(),
        **{
            expert_module_contract_path(module.module_contract_id): (
                module.to_json_bytes()
            )
            for module in module_contracts
        },
    }
    if any(
        source_contents.get(path) != payload
        for path, payload in expected_controls.items()
    ) or any(source_files[path].mode != "100644" for path in expected_controls):
        raise ExpertCompositionBaseError(
            "composition base generated controls differ from typed topology"
        )
    if (
        expert_semantic_book_digest(expected_book)
        != release_manifest.semantic_book_digest
    ):
        raise ExpertCompositionBaseError(
            "composition base semantic book differs from its release"
        )


@dataclass(frozen=True)
class ExpertCompositionBaseClosure:
    """Internally consistent release closure, without current-release authority."""

    reference: ExpertCompositionBaseReference
    scope_contract: ExpertScopeContract
    release_manifest: ExpertBaseReleaseManifest
    parent_tree_receipt: ExpertParentTreeReceipt
    repository_map: ExpertRepositoryMap
    module_contracts: tuple[ExpertModuleContract, ...]
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        _require_exact_types(
            reference=self.reference,
            scope_contract=self.scope_contract,
            release_manifest=self.release_manifest,
            parent_tree_receipt=self.parent_tree_receipt,
            repository_map=self.repository_map,
            module_contracts=self.module_contracts,
        )
        if not isinstance(self.source_contents, Mapping):
            raise ExpertCompositionBaseError(
                "composition base source contents must be a mapping"
            )
        frozen_contents = MappingProxyType(dict(self.source_contents))
        object.__setattr__(self, "source_contents", frozen_contents)
        _require_release_joins(
            scope_contract=self.scope_contract,
            release_manifest=self.release_manifest,
            parent_tree_receipt=self.parent_tree_receipt,
            repository_map=self.repository_map,
            module_contracts=self.module_contracts,
        )
        files_by_path = _require_source_byte_closure(
            parent_tree_receipt=self.parent_tree_receipt,
            source_contents=frozen_contents,
        )
        validate_expert_repository_topology(
            self.repository_map,
            self.module_contracts,
            validation_error_type=ExpertCompositionBaseError,
        )
        _require_generated_controls(
            scope_contract=self.scope_contract,
            release_manifest=self.release_manifest,
            repository_map=self.repository_map,
            module_contracts=self.module_contracts,
            source_contents=frozen_contents,
            source_files=files_by_path,
        )
        validate_expert_tree_ownership(
            self.repository_map,
            self.module_contracts,
            files_by_path,
            validation_error_type=ExpertCompositionBaseError,
        )
        expected_reference = _mint_base_reference(
            scope_contract=self.scope_contract,
            release_manifest=self.release_manifest,
            parent_tree_receipt=self.parent_tree_receipt,
            repository_map=self.repository_map,
            module_contracts=self.module_contracts,
        )
        if self.reference != expected_reference:
            raise ExpertCompositionBaseError(
                "composition base reference differs from verified release semantics"
            )

    @property
    def source_files(self) -> tuple[SourceFileDescriptor, ...]:
        return self.parent_tree_receipt.source_extraction_receipt.source_tree_files

    @property
    def source_tree(self) -> SourceArchiveExtractionReceipt:
        return self.parent_tree_receipt.source_extraction_receipt


def build_expert_composition_base_closure(
    *,
    scope_contract: ExpertScopeContract,
    release_manifest: ExpertBaseReleaseManifest,
    parent_tree_receipt: ExpertParentTreeReceipt,
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
    source_contents: Mapping[str, bytes],
) -> ExpertCompositionBaseClosure:
    """Check a release closure and derive its metadata-independent base identity."""

    _require_authority_types(
        scope_contract=scope_contract,
        release_manifest=release_manifest,
        parent_tree_receipt=parent_tree_receipt,
        repository_map=repository_map,
        module_contracts=module_contracts,
    )
    reference = _mint_base_reference(
        scope_contract=scope_contract,
        release_manifest=release_manifest,
        parent_tree_receipt=parent_tree_receipt,
        repository_map=repository_map,
        module_contracts=module_contracts,
    )
    return ExpertCompositionBaseClosure(
        reference=reference,
        scope_contract=scope_contract,
        release_manifest=release_manifest,
        parent_tree_receipt=parent_tree_receipt,
        repository_map=repository_map,
        module_contracts=module_contracts,
        source_contents=source_contents,
    )
