"""Authenticated GitHub CURRENT authority for expert composition bases."""

from __future__ import annotations

import os
import re

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
    PublicationArtifactKind,
)
from kapso.cross_run.expert.book import (
    EXPERT_MODULE_CONTRACT_ROOT,
    EXPERT_REPOSITORY_MAP_PATH,
    expert_module_contract_path,
)
from kapso.cross_run.expert.composition_base import (
    ExpertCompositionBaseClosure,
    build_expert_composition_base_closure,
    expert_composition_base_security_subject_ids,
)
from kapso.cross_run.expert.replay_authority_contracts import (
    SourceReplayCurrentReleaseObservation,
)
from kapso.cross_run.expert.release_authority import (
    AuthenticatedExpertReleaseActivation,
)
from kapso.cross_run.expert.triggers import ExpertSourceBaseTreeReceipt
from kapso.cross_run.github.materializer import (
    ExpertReleaseSourceSnapshot,
    GitHubArtifactMaterializer,
    MaterializedArtifact,
)
from kapso.cross_run.github.resolver import (
    CurrentArtifactPointer,
    GitHubArtifactResolver,
    ResolvedGitHubArtifact,
)
from kapso.cross_run.settings import ExpertSettings

EXPERT_COMPOSITION_BASE_PROVIDER_VERSION = (
    "kapso.github_expert_composition_base_provider.v1"
)
_MODULE_CONTRACT_PATH_PATTERN = re.compile(
    rf"^{re.escape(EXPERT_MODULE_CONTRACT_ROOT)}/[0-9a-f]{{64}}\.json$"
)
_CURRENT_EXPERT_COMPOSITION_BASE_SEAL = object()


class ExpertCompositionBaseProviderError(ValueError):
    """GitHub CURRENT cannot authorize one exact expert composition base."""


class CurrentExpertCompositionBase:
    """Process-local proof that a verified base was the authenticated CURRENT."""

    __slots__ = (
        "_closure",
        "_current_observation",
        "_owner_process_id",
        "_provider",
        "_resolved_current",
        "_security_subject_ids",
    )

    def __init__(
        self,
        seal: object,
        provider: GitHubExpertCompositionBaseProvider,
        *,
        closure: ExpertCompositionBaseClosure,
        current_observation: SourceReplayCurrentReleaseObservation,
        resolved_current: ResolvedGitHubArtifact,
    ) -> None:
        if seal is not _CURRENT_EXPERT_COMPOSITION_BASE_SEAL:
            raise ExpertCompositionBaseProviderError(
                "current composition base capability is not provider sealed"
            )
        security_subject_ids = expert_composition_base_security_subject_ids(
            closure,
            current_observation,
        )
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_closure", closure)
        object.__setattr__(self, "_current_observation", current_observation)
        object.__setattr__(self, "_resolved_current", resolved_current)
        object.__setattr__(self, "_security_subject_ids", security_subject_ids)

    def __setattr__(self, name, value) -> None:
        raise ExpertCompositionBaseProviderError(
            "current composition base capability is immutable"
        )

    def __reduce__(self):
        raise ExpertCompositionBaseProviderError(
            "current composition base capability cannot be serialized"
        )

    def __reduce_ex__(self, protocol):
        raise ExpertCompositionBaseProviderError(
            "current composition base capability cannot be serialized"
        )

    @property
    def closure(self) -> ExpertCompositionBaseClosure:
        self._require_owner_process()
        return self._closure

    @property
    def current_observation(self) -> SourceReplayCurrentReleaseObservation:
        self._require_owner_process()
        return self._current_observation

    @property
    def security_subject_ids(self) -> tuple[str, ...]:
        self._require_owner_process()
        return self._security_subject_ids

    def _require_owner_process(self) -> None:
        if self._owner_process_id != os.getpid():
            raise ExpertCompositionBaseProviderError(
                "current composition base capability is foreign"
            )

    def _require_bound(self, provider: GitHubExpertCompositionBaseProvider) -> None:
        self._require_owner_process()
        if self._provider is not provider:
            raise ExpertCompositionBaseProviderError(
                "current composition base capability belongs to another provider"
            )


class GitHubExpertCompositionBaseProvider:
    """Authenticate, materialize, and seal the exact current expert release."""

    __slots__ = ("_materializer", "_resolver", "_settings")

    def __init__(
        self,
        resolver: GitHubArtifactResolver,
        materializer: GitHubArtifactMaterializer,
        settings: ExpertSettings,
    ) -> None:
        if type(settings) is not ExpertSettings:
            raise ExpertCompositionBaseProviderError(
                "composition base provider requires exact expert settings"
            )
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_materializer", materializer)
        object.__setattr__(self, "_settings", settings)

    def __setattr__(self, name, value) -> None:
        raise ExpertCompositionBaseProviderError(
            "composition base provider authority is immutable"
        )

    @property
    def settings(self) -> ExpertSettings:
        return self._settings

    def resolve_current(
        self,
        scope_contract: ExpertScopeContract,
    ) -> CurrentExpertCompositionBase:
        if type(scope_contract) is not ExpertScopeContract:
            raise ExpertCompositionBaseProviderError(
                "composition base resolution requires one exact scope contract"
            )
        first = self._resolve_current(scope_contract.scope_id)
        materialized = self._materializer.materialize(first)
        if type(materialized) is not MaterializedArtifact:
            raise ExpertCompositionBaseProviderError(
                "composition base materializer returned an invalid artifact"
            )
        source_snapshot = self._materializer.inspect_expert_release_source(
            materialized,
            maximum_entries=self._settings.candidate_entry_limit,
            maximum_bytes=self._settings.candidate_byte_limit,
        )
        if type(source_snapshot) is not ExpertReleaseSourceSnapshot:
            raise ExpertCompositionBaseProviderError(
                "composition base materializer returned an invalid source snapshot"
            )
        closure = self._build_closure(
            scope_contract=scope_contract,
            pointer=first.pointer,
            materialized=materialized,
            source_snapshot=source_snapshot,
        )
        second = self._resolve_current(scope_contract.scope_id)
        self._require_same_current(first, second)
        current_observation = self._current_observation(second)
        return CurrentExpertCompositionBase(
            _CURRENT_EXPERT_COMPOSITION_BASE_SEAL,
            self,
            closure=closure,
            current_observation=current_observation,
            resolved_current=second,
        )

    def require_current(
        self,
        capability: CurrentExpertCompositionBase,
    ) -> SourceReplayCurrentReleaseObservation:
        """Diagnose live CURRENT; later admission must still fence persistence."""

        if type(capability) is not CurrentExpertCompositionBase:
            raise ExpertCompositionBaseProviderError(
                "composition base freshness requires its live capability"
            )
        capability._require_bound(self)
        current = self._resolve_current(capability.closure.scope_contract.scope_id)
        self._require_same_current(capability._resolved_current, current)
        return self._current_observation(current)

    def resolve_historical(
        self,
        scope_contract: ExpertScopeContract,
        activation: AuthenticatedExpertReleaseActivation,
    ) -> ExpertCompositionBaseClosure:
        """Materialize one provider-authenticated historical activation."""

        if (
            type(scope_contract) is not ExpertScopeContract
            or type(activation) is not AuthenticatedExpertReleaseActivation
        ):
            raise ExpertCompositionBaseProviderError(
                "historical composition base requires exact activation authority"
            )
        materialized = activation.materialized
        source_snapshot = self._materializer.inspect_expert_release_source(
            materialized,
            maximum_entries=self._settings.candidate_entry_limit,
            maximum_bytes=self._settings.candidate_byte_limit,
        )
        if type(source_snapshot) is not ExpertReleaseSourceSnapshot:
            raise ExpertCompositionBaseProviderError(
                "historical composition base source snapshot is invalid"
            )
        closure = self._build_closure(
            scope_contract=scope_contract,
            pointer=activation.pointer,
            materialized=materialized,
            source_snapshot=source_snapshot,
        )
        if closure.release_manifest != activation.manifest:
            raise ExpertCompositionBaseProviderError(
                "historical composition base differs from its activation"
            )
        return closure

    def _resolve_current(self, scope_id: str) -> ResolvedGitHubArtifact:
        resolved = self._resolver.resolve_current(
            scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
        )
        if type(resolved) is not ResolvedGitHubArtifact:
            raise ExpertCompositionBaseProviderError(
                "GitHub resolver returned an invalid current expert authority"
            )
        pointer = resolved.pointer
        publication = pointer.publication_record
        if (
            pointer.scope_id != scope_id
            or resolved.repositories.scope_id != scope_id
            or resolved.repositories.expert_repository
            != publication.repository_full_name
            or publication.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or publication.repository_full_name != resolved.policy.repository_full_name
            or publication.repository_node_id != resolved.policy.repository_node_id
        ):
            raise ExpertCompositionBaseProviderError(
                "resolved expert CURRENT has another scope, routing, kind, or repository"
            )
        return resolved

    @staticmethod
    def _require_same_current(
        first: ResolvedGitHubArtifact,
        second: ResolvedGitHubArtifact,
    ) -> None:
        if (
            first.repositories != second.repositories
            or first.policy != second.policy
            or first.pointer != second.pointer
        ):
            raise ExpertCompositionBaseProviderError(
                "expert CURRENT changed during composition base resolution"
            )

    @staticmethod
    def _current_observation(
        resolved: ResolvedGitHubArtifact,
    ) -> SourceReplayCurrentReleaseObservation:
        pointer = resolved.pointer
        publication = pointer.publication_record
        return SourceReplayCurrentReleaseObservation.mint(
            scope_id=pointer.scope_id,
            release_id=publication.artifact_id,
            publication_id=publication.publication_id,
            repository_full_name=resolved.policy.repository_full_name,
            repository_node_id=resolved.policy.repository_node_id,
            current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
            current_pointer_commit_sha=resolved.pointer_commit_sha,
            validation_closure_ids=pointer.validation_closure_ids,
        )

    @staticmethod
    def _build_closure(
        *,
        scope_contract: ExpertScopeContract,
        pointer: CurrentArtifactPointer,
        materialized: MaterializedArtifact,
        source_snapshot: ExpertReleaseSourceSnapshot,
    ) -> ExpertCompositionBaseClosure:
        publication = pointer.publication_record
        cache_receipt = materialized.receipt
        manifest = source_snapshot.release_manifest
        expected_assets = {asset.name: asset.sha256 for asset in publication.assets}
        if (
            cache_receipt.artifact_kind
            is not PublicationArtifactKind.EXPERT_BASE_RELEASE
            or cache_receipt.artifact_id != publication.artifact_id
            or cache_receipt.materialized_tree_digest
            != pointer.materialized_tree_digest
            or cache_receipt.manifest_relative_path != pointer.manifest_relative_path
            or cache_receipt.manifest_digest != pointer.manifest_digest
            or dict(cache_receipt.asset_digests) != expected_assets
            or manifest.release_id != publication.artifact_id
            or manifest.scope_id != scope_contract.scope_id
            or manifest.scope_contract_id != scope_contract.scope_contract_id
        ):
            raise ExpertCompositionBaseProviderError(
                "materialized expert base differs from GitHub or scope authority"
            )
        source_contents = source_snapshot.source_contents
        repository_map_payload = source_contents.get(EXPERT_REPOSITORY_MAP_PATH)
        if type(repository_map_payload) is not bytes:
            raise ExpertCompositionBaseProviderError(
                "expert base source omits its repository map"
            )
        repository_map = ExpertRepositoryMap.from_json_bytes(repository_map_payload)
        if repository_map_payload != repository_map.to_json_bytes():
            raise ExpertCompositionBaseProviderError(
                "expert base repository map is not canonical"
            )
        module_paths = tuple(
            sorted(
                path
                for path in source_contents
                if path == EXPERT_MODULE_CONTRACT_ROOT
                or path.startswith(f"{EXPERT_MODULE_CONTRACT_ROOT}/")
            )
        )
        if not module_paths or any(
            _MODULE_CONTRACT_PATH_PATTERN.fullmatch(path) is None
            for path in module_paths
        ):
            raise ExpertCompositionBaseProviderError(
                "expert base module-contract path closure is invalid"
            )
        modules = tuple(
            ExpertModuleContract.from_json_bytes(source_contents[path])
            for path in module_paths
        )
        if (
            len({module.module_id for module in modules}) != len(modules)
            or len({module.module_contract_id for module in modules}) != len(modules)
            or any(
                source_contents[path] != module.to_json_bytes()
                or path != expert_module_contract_path(module.module_contract_id)
                for path, module in zip(module_paths, modules)
            )
        ):
            raise ExpertCompositionBaseProviderError(
                "expert base module-contract closure is noncanonical or ambiguous"
            )
        canonical_modules = tuple(sorted(modules, key=lambda module: module.module_id))
        extraction_receipt = source_snapshot.source_extraction_receipt
        source_base_tree_receipt = ExpertSourceBaseTreeReceipt.mint(
            release_id=manifest.release_id,
            cache_verification_receipt=cache_receipt,
            source_extraction_receipt=extraction_receipt,
            source_base_tree_hash=extraction_receipt.source_tree_hash,
            repository_map_id=repository_map.repository_map_id,
            module_contract_ids=tuple(
                sorted(module.module_contract_id for module in canonical_modules)
            ),
            materializer_version=EXPERT_COMPOSITION_BASE_PROVIDER_VERSION,
        )
        return build_expert_composition_base_closure(
            scope_contract=scope_contract,
            release_manifest=manifest,
            source_base_tree_receipt=source_base_tree_receipt,
            repository_map=repository_map,
            module_contracts=canonical_modules,
            source_contents=source_contents,
        )
