"""Fail-closed resolution of one complete cross-run launch authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Protocol
from weakref import WeakValueDictionary

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_utc_timestamp,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertModuleContract,
    ExpertRepositoryMap,
    PublicationArtifactKind,
    TaskContextBinding,
)
from kapso.cross_run.embedding_space import EmbeddingSpace
from kapso.cross_run.expert.candidate_context import (
    ExpertCandidateValidationContext,
)
from kapso.cross_run.expert.promotion_stage_contracts import (
    ExpertReleaseMatrixStageResultRecord,
)
from kapso.cross_run.expert.release_use_policy_contracts import (
    ExpertReleaseUsePolicyObservation,
)
from kapso.cross_run.github.materializer import (
    ExpertReleaseSourceSnapshot,
    MaterializedArtifact,
)
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    GitHubArtifactActivationWitness,
    ResolvedGitHubArtifact,
)
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackage
from kapso.cross_run.launch.contracts import (
    LaunchCompatibilityPolicy,
    LaunchCompatibilityReceipt,
    LaunchCompatibilityAdmissionMode,
    LaunchContractError,
    LaunchExpertSourcePin,
    LaunchGitHubArtifactPin,
    LaunchManifest,
    LaunchRequest,
    LaunchStartingArtifact,
    LaunchStartingArtifactMaterializationReceipt,
    LaunchTaskAdapterPin,
    expected_launch_source_composition_hash,
    launch_security_subject_ids,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.task_adapters import (
    ActiveTaskAdapterBinding,
    ActiveTaskAdapterBindingProvider,
)

_EXPERT_RELEASE_RECORD_ROOT = ".kapso/expert/release-evidence/records"
_COMPATIBILITY_REASON_CODE = "verified_case_new_artifact_content"


class _ResolverAuthority:
    pass


_RESOLVER_AUTHORITY = _ResolverAuthority()
_ISSUED_RESOLVED_LAUNCHES: WeakValueDictionary[int, object] = WeakValueDictionary()
_RESOLVED_LAUNCH_AUTHORITY_LOCK = Lock()


class LaunchResolutionError(LaunchContractError):
    """The live authorities cannot admit one coherent launch tuple."""


class LaunchClock(Protocol):
    def now(self) -> str: ...


class SystemLaunchClock:
    def now(self) -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )


class LaunchGitHubResolver(Protocol):
    def resolve_current(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
    ) -> ResolvedGitHubArtifact: ...

    def read_artifact_intent(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
    ) -> ArtifactPublicationIntent | None: ...

    def resolve_artifact_activation_witness(
        self,
        scope_id: str,
        artifact_kind: PublicationArtifactKind,
        artifact_id: str,
        intent: ArtifactPublicationIntent,
        pointer,
        *,
        allow_missing: bool = False,
    ) -> GitHubArtifactActivationWitness | None: ...


class LaunchArtifactMaterializer(Protocol):
    def materialize(
        self,
        resolved: ResolvedGitHubArtifact,
    ) -> MaterializedArtifact: ...

    def inspect_expert_release_source(
        self,
        materialized: MaterializedArtifact,
        *,
        maximum_entries: int,
        maximum_bytes: int,
    ) -> ExpertReleaseSourceSnapshot: ...

    def read_verified_content_files(
        self,
        materialized: MaterializedArtifact,
        relative_paths: tuple[str, ...],
    ) -> Mapping[str, bytes]: ...


class LaunchReleaseUseAuthority(Protocol):
    def observe_exact(
        self,
        *,
        scope_contract,
        checked_release_ids: tuple[str, ...],
    ) -> ExpertReleaseUsePolicyObservation: ...


class LaunchSecurityAuthority(Protocol):
    def observe_exact(
        self,
        *,
        scope_id: str,
        scope_contract_id: str,
        checked_subject_ids: tuple[str, ...],
    ) -> SecurityDenylistObservation: ...


@dataclass(frozen=True)
class VerifiedLaunchStartingArtifact:
    """One launch input whose bytes exactly match its content-addressed descriptor."""

    artifact: LaunchStartingArtifact
    source_contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if type(self.artifact) is not LaunchStartingArtifact:
            raise LaunchResolutionError(
                "launch starting artifact must be one typed contract"
            )
        expected_paths = {
            descriptor.relative_path for descriptor in self.artifact.source_files
        }
        if set(self.source_contents) != expected_paths:
            raise LaunchResolutionError(
                "launch starting-artifact bytes differ from its descriptor paths"
            )
        for descriptor in self.artifact.source_files:
            payload = self.source_contents[descriptor.relative_path]
            if (
                type(payload) is not bytes
                or len(payload) != descriptor.size
                or tree_or_blob_digest(payload) != descriptor.digest
            ):
                raise LaunchResolutionError(
                    "launch starting-artifact bytes differ from a file descriptor"
                )
        object.__setattr__(
            self,
            "source_contents",
            MappingProxyType(dict(self.source_contents)),
        )


@dataclass(frozen=True)
class VerifiedLaunchStartingArtifacts:
    """Receipt plus every byte needed for deterministic workspace installation."""

    receipt: LaunchStartingArtifactMaterializationReceipt
    starting_artifacts: tuple[VerifiedLaunchStartingArtifact, ...]

    def __post_init__(self) -> None:
        if type(self.receipt) is not LaunchStartingArtifactMaterializationReceipt:
            raise LaunchResolutionError(
                "launch starting artifacts require one typed receipt"
            )
        if any(
            type(item) is not VerifiedLaunchStartingArtifact
            for item in self.starting_artifacts
        ):
            raise LaunchResolutionError(
                "launch starting artifacts contain an unverified byte closure"
            )
        if tuple(item.artifact for item in self.starting_artifacts) != (
            self.receipt.starting_artifacts
        ):
            raise LaunchResolutionError(
                "launch starting-artifact byte closures differ from their receipt"
            )

    @property
    def entry_count(self) -> int:
        return sum(len(item.artifact.source_files) for item in self.starting_artifacts)

    @property
    def byte_count(self) -> int:
        return sum(
            descriptor.size
            for item in self.starting_artifacts
            for descriptor in item.artifact.source_files
        )


class LaunchStartingArtifactProvider(Protocol):
    def materialize_exact(
        self,
        *,
        task_context_binding: TaskContextBinding,
        expected_artifact_content_ids: Mapping[str, str],
        maximum_entries: int,
        maximum_bytes: int,
    ) -> VerifiedLaunchStartingArtifacts: ...


@dataclass(frozen=True)
class ExpertLaunchEvidence:
    validation_context: ExpertCandidateValidationContext
    repository_map: ExpertRepositoryMap
    module_contracts: tuple[ExpertModuleContract, ...]
    release_matrix_stage_result: ExpertReleaseMatrixStageResultRecord

    def __post_init__(self) -> None:
        if (
            type(self.validation_context) is not ExpertCandidateValidationContext
            or type(self.repository_map) is not ExpertRepositoryMap
            or any(
                type(module) is not ExpertModuleContract
                for module in self.module_contracts
            )
            or type(self.release_matrix_stage_result)
            is not ExpertReleaseMatrixStageResultRecord
        ):
            raise LaunchResolutionError(
                "expert launch evidence requires exact typed evidence"
            )
        module_ids = tuple(
            module.module_contract_id for module in self.module_contracts
        )
        if module_ids != tuple(sorted(set(module_ids))):
            raise LaunchResolutionError(
                "expert launch module evidence must be sorted and unique"
            )


@dataclass(frozen=True)
class ResolvedLaunch:
    """Verified bytes and the sole immutable authority passed to bootstrap."""

    manifest: LaunchManifest
    expert_artifact: MaterializedArtifact
    expert_source: ExpertReleaseSourceSnapshot
    knowledge_artifact: MaterializedArtifact
    knowledge_package: KnowledgeSnapshotPackage
    task_adapter_binding: ActiveTaskAdapterBinding
    expert_evidence: ExpertLaunchEvidence
    starting_artifacts: VerifiedLaunchStartingArtifacts
    _resolver_authority: object

    def __post_init__(self) -> None:
        manifest = self.manifest
        if (
            type(manifest) is not LaunchManifest
            or type(self.expert_artifact) is not MaterializedArtifact
            or type(self.expert_source) is not ExpertReleaseSourceSnapshot
            or type(self.knowledge_artifact) is not MaterializedArtifact
            or type(self.knowledge_package) is not KnowledgeSnapshotPackage
            or type(self.task_adapter_binding) is not ActiveTaskAdapterBinding
            or type(self.expert_evidence) is not ExpertLaunchEvidence
            or type(self.starting_artifacts) is not VerifiedLaunchStartingArtifacts
            or self._resolver_authority is not _RESOLVER_AUTHORITY
            or self.expert_source.release_manifest != manifest.expert_manifest
            or self.expert_artifact.receipt != manifest.expert_component.cache_receipt
            or self.knowledge_package.manifest != manifest.knowledge_manifest
            or self.knowledge_artifact.receipt
            != manifest.knowledge_component.cache_receipt
            or self.task_adapter_binding.activation != manifest.task_adapter.activation
            or self.task_adapter_binding.verified_adapter.manifest
            != manifest.task_adapter.manifest
            or self.expert_evidence.repository_map != manifest.expert_repository_map
            or self.expert_evidence.module_contracts != manifest.expert_module_contracts
            or self.expert_evidence.validation_context.validation_context_id
            != manifest.expert_manifest.candidate_validation_context_ref
            or self.expert_evidence.validation_context.scope_contract
            != manifest.scope_contract
            or self.expert_evidence.release_matrix_stage_result.stage_result_record_id
            != manifest.expert_manifest.release_matrix_stage_result_id
            or self.expert_evidence.release_matrix_stage_result.release_matrix_report.release_matrix_report_id
            != manifest.expert_manifest.release_matrix_report_id
            or sum(
                authority.adapter_authority_id
                == manifest.compatibility_receipt.expert_release_matrix_adapter_authority_id
                and authority.task_adapter_manifest == manifest.task_adapter.manifest
                and authority.verification_receipt
                == manifest.task_adapter.verification_receipt
                for authority in self.expert_evidence.release_matrix_stage_result.release_matrix_report.evaluation_plan.adapter_authorities
            )
            != 1
            or self.starting_artifacts.receipt != manifest.starting_artifacts
        ):
            raise LaunchResolutionError(
                "resolved launch bytes differ from their launch manifest"
            )

    def require_resolver_authority(self) -> None:
        identity = id(self)
        with _RESOLVED_LAUNCH_AUTHORITY_LOCK:
            issued = _ISSUED_RESOLVED_LAUNCHES.pop(identity, None)
        if self._resolver_authority is not _RESOLVER_AUTHORITY or issued is not self:
            raise LaunchResolutionError("resolved launch lacks live resolver authority")


class LaunchResolver:
    """Resolve, verify, and freeze the only tuple a fresh run may consume."""

    def __init__(
        self,
        *,
        settings: CrossRunSettings,
        experiment_embedding_space: EmbeddingSpace,
        github_resolver: LaunchGitHubResolver,
        materializer: LaunchArtifactMaterializer,
        task_adapters: ActiveTaskAdapterBindingProvider,
        starting_artifacts: LaunchStartingArtifactProvider,
        release_use_authority: LaunchReleaseUseAuthority,
        security_authority: LaunchSecurityAuthority,
        clock: LaunchClock | None = None,
    ) -> None:
        if type(settings) is not CrossRunSettings:
            raise LaunchResolutionError("launch resolver requires exact settings")
        if type(experiment_embedding_space) is not EmbeddingSpace:
            raise LaunchResolutionError(
                "launch resolver requires one exact experiment embedding space"
            )
        self._settings = settings
        self._experiment_embedding_space = experiment_embedding_space
        self._github_resolver = github_resolver
        self._materializer = materializer
        self._task_adapters = task_adapters
        self._starting_artifacts = starting_artifacts
        self._release_use_authority = release_use_authority
        self._security_authority = security_authority
        self._clock = SystemLaunchClock() if clock is None else clock

    def resolve(self, request: LaunchRequest) -> ResolvedLaunch:
        if type(request) is not LaunchRequest:
            raise LaunchResolutionError("launch resolver requires one LaunchRequest")
        resolved_at = self._clock.now()
        resolved_time = parse_utc_timestamp(resolved_at, "launch resolved_at")
        scope_id = request.binding.scope_id

        expert_current = self._github_resolver.resolve_current(
            scope_id,
            PublicationArtifactKind.EXPERT_BASE_RELEASE,
        )
        knowledge_current = self._github_resolver.resolve_current(
            scope_id,
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        )
        self._require_shared_registry_binding(expert_current, knowledge_current)
        self._require_fresh(expert_current, resolved_time)
        self._require_fresh(knowledge_current, resolved_time)

        expert_artifact = self._materializer.materialize(expert_current)
        knowledge_artifact = self._materializer.materialize(knowledge_current)
        expert_source = self._materializer.inspect_expert_release_source(
            expert_artifact,
            maximum_entries=self._settings.github.source_entry_limit,
            maximum_bytes=self._settings.github.source_tree_size_bytes,
        )
        knowledge_package = KnowledgeSnapshotPackage.open(knowledge_artifact.content)
        self._require_materialized_identities(
            expert_current=expert_current,
            expert_artifact=expert_artifact,
            expert_source=expert_source,
            knowledge_current=knowledge_current,
            knowledge_artifact=knowledge_artifact,
            knowledge_package=knowledge_package,
        )

        scope_contract = knowledge_package.prepared.scope_contract
        scope_contract.validate_binding(request.binding)
        task_context = request.task_context_request.bind(
            binding=request.binding,
            scope_contract=scope_contract,
        )
        task_adapter_binding = self._task_adapters.resolve_active_binding(
            scope_contract_id=scope_contract.scope_contract_id,
            task_family_id=request.binding.task_family_id,
            task_adapter_id=request.binding.task_adapter_id,
        )
        task_adapter = self._task_adapter_pin(task_adapter_binding)
        self._require_runtime(request, task_adapter)
        self._require_consumed_dimensions(task_context, task_adapter)
        starting_artifacts = self._materialize_starting_artifacts(
            task_context,
            request.starting_artifact_content_ids,
        )

        expert_evidence = self._read_expert_evidence(
            expert_artifact,
            expert_source,
        )
        matrix_adapter_authority_id = self._require_expert_compatibility_evidence(
            request=request,
            scope_contract=scope_contract,
            task_adapter_binding=task_adapter_binding,
            evidence=expert_evidence,
            expert_source=expert_source,
        )
        compatibility_case_ids = self._compatible_adapter_case_ids(
            request,
            task_adapter_binding,
            starting_artifacts.receipt,
        )

        expert_component = self._component_pin(
            expert_current,
            expert_artifact,
        )
        knowledge_component = self._component_pin(
            knowledge_current,
            knowledge_artifact,
        )
        expert_source_pin = LaunchExpertSourcePin.mint(
            expert_release_id=expert_source.release_manifest.release_id,
            extraction_receipt=expert_source.source_extraction_receipt,
        )
        knowledge_embedding_space = self._knowledge_embedding_space()
        experiment_embedding_space = self._experiment_embedding_space
        snapshot_embedding_spaces = {
            sidecar.embedding_space_id
            for sidecar in knowledge_package.manifest.embedding_sidecars
        }
        if (
            snapshot_embedding_spaces
            and knowledge_embedding_space.embedding_space_id
            not in snapshot_embedding_spaces
        ) or (not snapshot_embedding_spaces and knowledge_package.retrieval_root_ids):
            raise LaunchResolutionError(
                "current knowledge snapshot lacks the configured embedding space"
            )

        release_use = self._release_use_authority.observe_exact(
            scope_contract=scope_contract,
            checked_release_ids=(expert_source.release_manifest.release_id,),
        )
        if release_use.matched_revocations:
            raise LaunchResolutionError(
                "current knowledge policy revokes the selected expert release"
            )
        self._require_release_use_pin(release_use, knowledge_component)

        policy = LaunchCompatibilityPolicy.mint(
            policy_version=self._settings.launch.compatibility_policy_version,
            admission_mode=(
                LaunchCompatibilityAdmissionMode.VERIFIED_CASE_NEW_ARTIFACT_CONTENT
            ),
            artifact_ttl_seconds=self._settings.launch.artifact_ttl_seconds,
        )
        source_composition_hash = expected_launch_source_composition_hash(
            expert_source_tree_hash=expert_source.release_manifest.candidate_tree_hash,
            expert_repository_map=expert_evidence.repository_map,
            task_adapter=task_adapter,
            starting_artifacts=starting_artifacts.receipt,
        )
        runtime_contract_digest = tree_or_blob_digest(
            canonical_json_bytes(request.dependency_runtime_contract)
        )
        compatibility = self._compatibility_receipt(
            policy=policy,
            request=request,
            task_context_binding_id=task_context.task_context_binding_id,
            scope_contract_id=scope_contract.scope_contract_id,
            expert_component=expert_component,
            expert_source=expert_source,
            knowledge_component=knowledge_component,
            knowledge_package=knowledge_package,
            task_adapter=task_adapter,
            release_use=release_use,
            expert_evidence=expert_evidence,
            matrix_adapter_authority_id=matrix_adapter_authority_id,
            compatibility_case_ids=compatibility_case_ids,
            starting_artifacts=starting_artifacts.receipt,
            knowledge_embedding_space_id=(knowledge_embedding_space.embedding_space_id),
            runtime_contract_digest=runtime_contract_digest,
            source_composition_hash=source_composition_hash,
            resolved_at=resolved_at,
        )
        security_subjects = launch_security_subject_ids(
            launch_request=request,
            scope_contract=scope_contract,
            task_context_binding=task_context,
            expert_component=expert_component,
            expert_manifest=expert_source.release_manifest,
            expert_source=expert_source_pin,
            expert_repository_map=expert_evidence.repository_map,
            expert_module_contracts=expert_evidence.module_contracts,
            knowledge_component=knowledge_component,
            knowledge_manifest=knowledge_package.manifest,
            task_adapter=task_adapter,
            starting_artifacts=starting_artifacts.receipt,
            knowledge_embedding_space=knowledge_embedding_space,
            experiment_embedding_space=experiment_embedding_space,
            release_use_observation=release_use,
            compatibility_receipt=compatibility,
        )
        security = self._security_authority.observe_exact(
            scope_id=scope_id,
            scope_contract_id=scope_contract.scope_contract_id,
            checked_subject_ids=security_subjects,
        )
        if security.matched_revocations:
            raise LaunchResolutionError(
                "security denylist rejects the selected launch dependency closure"
            )

        self._require_authorities_unchanged(
            request=request,
            expert_current=expert_current,
            knowledge_current=knowledge_current,
            expert_component=expert_component,
            knowledge_component=knowledge_component,
            task_adapter_binding=task_adapter_binding,
        )
        exact_dependencies = tuple(
            sorted({*security_subjects, security.observation_id})
        )
        manifest = LaunchManifest.mint(
            launch_request=request,
            launch_request_hash=request.request_hash,
            scope_contract=scope_contract,
            task_context_binding=task_context,
            scope_repositories=expert_current.repositories,
            scope_repository_binding_hash=(
                expert_current.repositories.binding_fingerprint
            ),
            configuration_fingerprint=request.configuration_fingerprint,
            expert_component=expert_component,
            expert_manifest=expert_source.release_manifest,
            expert_source=expert_source_pin,
            expert_repository_map=expert_evidence.repository_map,
            expert_module_contracts=expert_evidence.module_contracts,
            knowledge_component=knowledge_component,
            knowledge_manifest=knowledge_package.manifest,
            task_adapter=task_adapter,
            starting_artifacts=starting_artifacts.receipt,
            knowledge_embedding_space=knowledge_embedding_space,
            experiment_embedding_space=experiment_embedding_space,
            dependency_runtime_contract=request.dependency_runtime_contract,
            sanitation_policy_version=(
                knowledge_package.manifest.sanitation_policy_version
            ),
            security_observation=security,
            release_use_observation=release_use,
            compatibility_receipt=compatibility,
            expected_source_composition_hash=source_composition_hash,
            exact_dependency_ids=exact_dependencies,
        )
        resolved_launch = ResolvedLaunch(
            manifest=manifest,
            expert_artifact=expert_artifact,
            expert_source=expert_source,
            knowledge_artifact=knowledge_artifact,
            knowledge_package=knowledge_package,
            task_adapter_binding=task_adapter_binding,
            expert_evidence=expert_evidence,
            starting_artifacts=starting_artifacts,
            _resolver_authority=_RESOLVER_AUTHORITY,
        )
        with _RESOLVED_LAUNCH_AUTHORITY_LOCK:
            _ISSUED_RESOLVED_LAUNCHES[id(resolved_launch)] = resolved_launch
        return resolved_launch

    def _require_shared_registry_binding(
        self,
        expert_current: ResolvedGitHubArtifact,
        knowledge_current: ResolvedGitHubArtifact,
    ) -> None:
        if (
            expert_current.repositories != knowledge_current.repositories
            or expert_current.repositories.scope_id
            != knowledge_current.repositories.scope_id
        ):
            raise LaunchResolutionError(
                "expert and knowledge artifacts leave one registry binding"
            )

    def _require_fresh(
        self,
        resolved: ResolvedGitHubArtifact,
        resolved_time: datetime,
    ) -> None:
        publication_time = parse_utc_timestamp(
            resolved.pointer.publication_record.published_at,
            "launch publication time",
        )
        age_seconds = (resolved_time - publication_time).total_seconds()
        if age_seconds < 0 or age_seconds > self._settings.launch.artifact_ttl_seconds:
            raise LaunchResolutionError(
                "launch artifact is future-dated or exceeds its freshness policy"
            )

    @staticmethod
    def _require_materialized_identities(
        *,
        expert_current: ResolvedGitHubArtifact,
        expert_artifact: MaterializedArtifact,
        expert_source: ExpertReleaseSourceSnapshot,
        knowledge_current: ResolvedGitHubArtifact,
        knowledge_artifact: MaterializedArtifact,
        knowledge_package: KnowledgeSnapshotPackage,
    ) -> None:
        if (
            expert_artifact.receipt.artifact_id
            != expert_current.pointer.publication_record.artifact_id
            or expert_source.release_manifest.release_id
            != expert_artifact.receipt.artifact_id
            or knowledge_artifact.receipt.artifact_id
            != knowledge_current.pointer.publication_record.artifact_id
            or knowledge_package.manifest.snapshot_id
            != knowledge_artifact.receipt.artifact_id
        ):
            raise LaunchResolutionError(
                "materialized artifact identity differs from CURRENT"
            )

    @staticmethod
    def _task_adapter_pin(
        binding: ActiveTaskAdapterBinding,
    ) -> LaunchTaskAdapterPin:
        verified = binding.verified_adapter
        return LaunchTaskAdapterPin.mint(
            activation=binding.activation,
            manifest=verified.manifest,
            verification_receipt=verified.verification_receipt,
            source_extraction_receipt=verified.source_extraction_receipt,
        )

    @staticmethod
    def _require_runtime(
        request: LaunchRequest,
        task_adapter: LaunchTaskAdapterPin,
    ) -> None:
        actual_runtime = task_adapter.manifest.runtime.to_dict()
        runtime_digest = tree_or_blob_digest(canonical_json_bytes(actual_runtime))
        if (
            canonical_json_bytes(request.dependency_runtime_contract)
            != canonical_json_bytes(actual_runtime)
            or request.task_context_request.dependency_runtime_fingerprint
            != runtime_digest
        ):
            raise LaunchResolutionError(
                "launch runtime contract differs from the active task adapter"
            )

    @staticmethod
    def _require_consumed_dimensions(task_context, task_adapter) -> None:
        consumed = set(task_adapter.manifest.context_binding.consumed_dimension_ids)
        if not consumed.issubset(task_context.transfer_dimensions):
            raise LaunchResolutionError(
                "launch omits a context dimension consumed by the task adapter"
            )

    def _materialize_starting_artifacts(
        self,
        task_context: TaskContextBinding,
        expected_artifact_content_ids: Mapping[str, str],
    ) -> VerifiedLaunchStartingArtifacts:
        settings = self._settings.launch
        verified = self._starting_artifacts.materialize_exact(
            task_context_binding=task_context,
            expected_artifact_content_ids=expected_artifact_content_ids,
            maximum_entries=settings.starting_artifact_entry_limit,
            maximum_bytes=settings.starting_artifact_byte_limit,
        )
        if type(verified) is not VerifiedLaunchStartingArtifacts:
            raise LaunchResolutionError(
                "launch starting-artifact provider returned an unverified closure"
            )
        receipt = verified.receipt
        observed_artifact_ids = {
            artifact.starting_artifact_ref: artifact.starting_artifact_content_id
            for artifact in receipt.starting_artifacts
        }
        if (
            receipt.task_context_binding_id != task_context.task_context_binding_id
            or observed_artifact_ids != dict(expected_artifact_content_ids)
            or receipt.materializer_id != settings.starting_artifact_materializer_id
            or receipt.materializer_version
            != settings.starting_artifact_materializer_version
            or verified.entry_count > settings.starting_artifact_entry_limit
            or verified.byte_count > settings.starting_artifact_byte_limit
        ):
            raise LaunchResolutionError(
                "launch starting-artifact receipt differs from request or policy"
            )
        return verified

    def _read_expert_evidence(
        self,
        expert_artifact: MaterializedArtifact,
        expert_source: ExpertReleaseSourceSnapshot,
    ) -> ExpertLaunchEvidence:
        manifest = expert_source.release_manifest
        record_ids = (
            manifest.candidate_validation_context_ref,
            manifest.repository_map_ref,
            *manifest.module_contract_refs,
            manifest.release_matrix_stage_result_id,
        )
        paths = tuple(
            sorted(self._expert_record_path(record_id) for record_id in record_ids)
        )
        payloads = self._materializer.read_verified_content_files(
            expert_artifact,
            paths,
        )
        validation_context = self._parse_expert_record(
            payloads,
            manifest.candidate_validation_context_ref,
            ExpertCandidateValidationContext,
        )
        repository_map = self._parse_expert_record(
            payloads,
            manifest.repository_map_ref,
            ExpertRepositoryMap,
        )
        module_contracts = tuple(
            sorted(
                (
                    self._parse_expert_record(
                        payloads,
                        module_contract_id,
                        ExpertModuleContract,
                    )
                    for module_contract_id in manifest.module_contract_refs
                ),
                key=lambda module_contract: module_contract.module_contract_id,
            )
        )
        stage_result = self._parse_expert_record(
            payloads,
            manifest.release_matrix_stage_result_id,
            ExpertReleaseMatrixStageResultRecord,
        )
        return ExpertLaunchEvidence(
            validation_context=validation_context,
            repository_map=repository_map,
            module_contracts=module_contracts,
            release_matrix_stage_result=stage_result,
        )

    @staticmethod
    def _expert_record_path(record_id: str) -> str:
        namespace, digest = record_id.split(":sha256:", 1)
        return f"{_EXPERT_RELEASE_RECORD_ROOT}/{namespace}/{digest}.json"

    @classmethod
    def _parse_expert_record(cls, payloads, record_id, record_type):
        payload = payloads[cls._expert_record_path(record_id)]
        record = record_type.from_json_bytes(payload)
        if payload != record.to_json_bytes():
            raise LaunchResolutionError(
                "expert release evidence record is not canonical"
            )
        return record

    @staticmethod
    def _require_expert_compatibility_evidence(
        *,
        request: LaunchRequest,
        scope_contract,
        task_adapter_binding: ActiveTaskAdapterBinding,
        evidence: ExpertLaunchEvidence,
        expert_source: ExpertReleaseSourceSnapshot,
    ) -> str:
        manifest = expert_source.release_manifest
        context = evidence.validation_context
        repository_map = evidence.repository_map
        module_contracts = evidence.module_contracts
        stage = evidence.release_matrix_stage_result
        module_ids = tuple(module.module_contract_id for module in module_contracts)
        if (
            context.validation_context_id != manifest.candidate_validation_context_ref
            or context.scope_contract != scope_contract
            or request.binding not in context.active_task_bindings
            or repository_map.repository_map_id != manifest.repository_map_ref
            or repository_map.scope_contract_id != scope_contract.scope_contract_id
            or module_ids != manifest.module_contract_refs
            or dict(manifest.module_versions)
            != {module.module_id: module.version for module in module_contracts}
            or stage.stage_result_record_id != manifest.release_matrix_stage_result_id
            or stage.release_matrix_report.release_matrix_report_id
            != manifest.release_matrix_report_id
            or stage.scope_contract_id != scope_contract.scope_contract_id
        ):
            raise LaunchResolutionError(
                "expert release evidence differs from the requested scope tuple"
            )
        module_id_set = set(module_ids)
        applicable_nodes = tuple(
            node
            for node in repository_map.capability_nodes
            if request.binding.task_family_id in node.task_family_bindings
        )
        if not applicable_nodes or any(
            node.module_contract_ref not in module_id_set for node in applicable_nodes
        ):
            raise LaunchResolutionError(
                "expert repository has no validated capability for the task family"
            )
        verified = task_adapter_binding.verified_adapter
        matches = tuple(
            authority
            for authority in stage.release_matrix_report.evaluation_plan.adapter_authorities
            if authority.task_adapter_manifest == verified.manifest
            and authority.verification_receipt == verified.verification_receipt
        )
        if len(matches) != 1:
            raise LaunchResolutionError(
                "expert release was not accepted with the active task adapter package"
            )
        return matches[0].adapter_authority_id

    @staticmethod
    def _compatible_adapter_case_ids(
        request: LaunchRequest,
        task_adapter_binding: ActiveTaskAdapterBinding,
        starting_artifacts: LaunchStartingArtifactMaterializationReceipt,
    ) -> tuple[str, ...]:
        launch_context = request.task_context_request
        launch_mounts = {
            artifact.starting_artifact_ref: artifact.mount_path
            for artifact in starting_artifacts.starting_artifacts
        }
        compatible_cases = tuple(
            case.release_matrix_case_id
            for case in task_adapter_binding.verified_adapter.manifest.release_matrix_cases
            if case.task_context_binding.scope_id == request.binding.scope_id
            and case.task_context_binding.task_family_id
            == request.binding.task_family_id
            and case.task_context_binding.task_adapter_id
            == request.binding.task_adapter_id
            and case.task_context_binding.input_contract_fingerprint
            == launch_context.input_contract_fingerprint
            and case.task_context_binding.target_contract_fingerprint
            == launch_context.target_contract_fingerprint
            and case.task_context_binding.starting_artifact_refs
            == launch_context.starting_artifact_refs
            and case.task_context_binding.capability_tags
            == launch_context.capability_tags
            and case.task_context_binding.method_fingerprint
            == launch_context.method_fingerprint
            and case.task_context_binding.toolchain_fingerprint
            == launch_context.toolchain_fingerprint
            and case.task_context_binding.budget_hardware_envelope
            == launch_context.budget_hardware_envelope
            and case.task_context_binding.transfer_dimensions
            == launch_context.transfer_dimensions
            and {
                artifact.starting_artifact_ref: artifact.mount_path
                for artifact in case.starting_artifacts
            }
            == launch_mounts
        )
        if not compatible_cases:
            raise LaunchResolutionError(
                "launch task interface has no compatible verified adapter case"
            )
        return tuple(sorted(compatible_cases))

    def _component_pin(
        self,
        resolved: ResolvedGitHubArtifact,
        materialized: MaterializedArtifact,
    ) -> LaunchGitHubArtifactPin:
        pointer = resolved.pointer
        publication = pointer.publication_record
        intent = self._github_resolver.read_artifact_intent(
            resolved.repositories.scope_id,
            publication.artifact_kind,
            publication.artifact_id,
        )
        if intent is None:
            raise LaunchResolutionError(
                "CURRENT artifact lacks its immutable publication intent"
            )
        witness = self._github_resolver.resolve_artifact_activation_witness(
            resolved.repositories.scope_id,
            publication.artifact_kind,
            publication.artifact_id,
            intent,
            pointer,
        )
        if witness is None:
            raise LaunchResolutionError(
                "CURRENT artifact lacks its immutable activation witness"
            )
        return LaunchGitHubArtifactPin.mint(
            scope_id=resolved.repositories.scope_id,
            scope_repository_binding_hash=(resolved.repositories.binding_fingerprint),
            pointer=pointer,
            publication_intent=intent,
            authority_commit_sha=resolved.pointer_commit_sha,
            activation_witness=witness,
            cache_receipt=materialized.receipt,
        )

    def _knowledge_embedding_space(self) -> EmbeddingSpace:
        embeddings = self._settings.knowledge.embeddings
        if not embeddings.enabled:
            raise LaunchResolutionError(
                "cross-run launch requires configured knowledge embeddings"
            )
        return EmbeddingSpace.mint(
            provider=embeddings.provider,
            model=embeddings.model,
            dimensions=embeddings.dimensions,
            canonicalizer_version=embeddings.canonicalizer_version,
        )

    @staticmethod
    def _require_release_use_pin(
        observation: ExpertReleaseUsePolicyObservation,
        knowledge_component: LaunchGitHubArtifactPin,
    ) -> None:
        publication = knowledge_component.publication
        if (
            observation.scope_id != knowledge_component.scope_id
            or observation.scope_repository_binding_hash
            != knowledge_component.scope_repository_binding_hash
            or observation.knowledge_snapshot_id != knowledge_component.artifact_id
            or observation.knowledge_publication_id != publication.publication_id
            or observation.current_pointer_digest
            != knowledge_component.current_pointer_digest
            or observation.authority_commit_sha
            != knowledge_component.authority_commit_sha
        ):
            raise LaunchResolutionError(
                "release-use policy observation names another knowledge CURRENT"
            )

    @staticmethod
    def _compatibility_receipt(
        *,
        policy: LaunchCompatibilityPolicy,
        request: LaunchRequest,
        task_context_binding_id: str,
        scope_contract_id: str,
        expert_component: LaunchGitHubArtifactPin,
        expert_source: ExpertReleaseSourceSnapshot,
        knowledge_component: LaunchGitHubArtifactPin,
        knowledge_package: KnowledgeSnapshotPackage,
        task_adapter: LaunchTaskAdapterPin,
        release_use: ExpertReleaseUsePolicyObservation,
        expert_evidence: ExpertLaunchEvidence,
        matrix_adapter_authority_id: str,
        compatibility_case_ids: tuple[str, ...],
        starting_artifacts: LaunchStartingArtifactMaterializationReceipt,
        knowledge_embedding_space_id: str,
        runtime_contract_digest: str,
        source_composition_hash: str,
        resolved_at: str,
    ) -> LaunchCompatibilityReceipt:
        manifest = expert_source.release_manifest
        module_ids = tuple(
            module.module_contract_id for module in expert_evidence.module_contracts
        )
        starting_artifact_ids = tuple(
            sorted(request.starting_artifact_content_ids.values())
        )
        dependencies = tuple(
            sorted(
                {
                    policy.compatibility_policy_id,
                    request.launch_request_id,
                    task_context_binding_id,
                    scope_contract_id,
                    expert_component.component_pin_id,
                    manifest.release_id,
                    knowledge_component.component_pin_id,
                    knowledge_package.manifest.snapshot_id,
                    task_adapter.adapter_pin_id,
                    task_adapter.manifest.task_adapter_manifest_id,
                    task_adapter.verification_receipt.verification_receipt_id,
                    task_adapter.activation.activation_id,
                    knowledge_embedding_space_id,
                    release_use.observation_id,
                    expert_evidence.validation_context.validation_context_id,
                    expert_evidence.repository_map.repository_map_id,
                    *module_ids,
                    manifest.release_matrix_stage_result_id,
                    manifest.release_matrix_report_id,
                    matrix_adapter_authority_id,
                    *compatibility_case_ids,
                    starting_artifacts.materialization_receipt_id,
                    *starting_artifact_ids,
                }
            )
        )
        return LaunchCompatibilityReceipt.mint(
            policy=policy,
            launch_request_id=request.launch_request_id,
            task_context_binding_id=task_context_binding_id,
            scope_contract_id=scope_contract_id,
            expert_component_pin_id=expert_component.component_pin_id,
            expert_release_id=manifest.release_id,
            knowledge_component_pin_id=knowledge_component.component_pin_id,
            knowledge_snapshot_id=knowledge_package.manifest.snapshot_id,
            task_adapter_pin_id=task_adapter.adapter_pin_id,
            task_adapter_manifest_id=task_adapter.manifest.task_adapter_manifest_id,
            task_adapter_verification_receipt_id=(
                task_adapter.verification_receipt.verification_receipt_id
            ),
            task_adapter_activation_id=task_adapter.activation.activation_id,
            knowledge_embedding_space_id=knowledge_embedding_space_id,
            release_use_observation_id=release_use.observation_id,
            expert_validation_context_id=(
                expert_evidence.validation_context.validation_context_id
            ),
            expert_repository_map_id=(expert_evidence.repository_map.repository_map_id),
            expert_module_contract_ids=module_ids,
            expert_release_matrix_stage_result_id=(
                manifest.release_matrix_stage_result_id
            ),
            expert_release_matrix_report_id=manifest.release_matrix_report_id,
            expert_release_matrix_adapter_authority_id=(matrix_adapter_authority_id),
            task_adapter_compatibility_case_ids=compatibility_case_ids,
            starting_artifact_materialization_receipt_id=(
                starting_artifacts.materialization_receipt_id
            ),
            starting_artifact_content_ids=starting_artifact_ids,
            runtime_contract_digest=runtime_contract_digest,
            source_composition_hash=source_composition_hash,
            resolved_at=resolved_at,
            compatible=True,
            reason_code=_COMPATIBILITY_REASON_CODE,
            exact_dependency_ids=dependencies,
        )

    def _require_authorities_unchanged(
        self,
        *,
        request: LaunchRequest,
        expert_current: ResolvedGitHubArtifact,
        knowledge_current: ResolvedGitHubArtifact,
        expert_component: LaunchGitHubArtifactPin,
        knowledge_component: LaunchGitHubArtifactPin,
        task_adapter_binding: ActiveTaskAdapterBinding,
    ) -> None:
        current_components = (
            (
                PublicationArtifactKind.EXPERT_BASE_RELEASE,
                expert_current,
                expert_component,
            ),
            (
                PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
                knowledge_current,
                knowledge_component,
            ),
        )
        for artifact_kind, expected_current, component in current_components:
            observed_current = self._github_resolver.resolve_current(
                request.binding.scope_id,
                artifact_kind,
            )
            observed_intent = self._github_resolver.read_artifact_intent(
                request.binding.scope_id,
                artifact_kind,
                component.artifact_id,
            )
            observed_witness = (
                None
                if observed_intent is None
                else self._github_resolver.resolve_artifact_activation_witness(
                    request.binding.scope_id,
                    artifact_kind,
                    component.artifact_id,
                    observed_intent,
                    component.pointer,
                )
            )
            if (
                observed_current != expected_current
                or observed_intent != component.publication_intent
                or observed_witness != component.activation_witness
            ):
                raise LaunchResolutionError(
                    "launch authority changed during transactional resolution"
                )
        if (
            self._task_adapters.resolve_active_binding(
                scope_contract_id=(task_adapter_binding.activation.scope_contract_id),
                task_family_id=request.binding.task_family_id,
                task_adapter_id=request.binding.task_adapter_id,
            )
            != task_adapter_binding
        ):
            raise LaunchResolutionError(
                "launch authority changed during transactional resolution"
            )


__all__ = [
    "ExpertLaunchEvidence",
    "LaunchResolutionError",
    "LaunchResolver",
    "ResolvedLaunch",
    "SystemLaunchClock",
]
